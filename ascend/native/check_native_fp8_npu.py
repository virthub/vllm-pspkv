# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Compare native FP8 cache writes/reads with the PyTorch reference on NPU."""

import os

import torch
import torch.nn.functional as F
import torch_npu  # noqa: F401 - registers the NPU device

from vllm.v1.attention.backends.psp_kv_ascend_reference import (
    decode_e4m3,
    decode_e5m2,
    encode_e4m3,
    encode_e5m2,
)


def check_case(
    key: torch.Tensor,
    value: torch.Tensor,
    slots: torch.Tensor,
    formats: torch.Tensor,
    block_size: int,
) -> None:
    blocks = formats.numel()
    heads, dim = key.shape[1:]
    shape = (blocks, block_size, heads, dim)
    native_k = torch.zeros(shape, dtype=torch.uint8, device=key.device)
    native_v = torch.zeros_like(native_k)
    torch.ops.psp_kv_ascend_native.quantize_scatter_bf16(
        key, value, slots, formats, native_k, native_v
    )
    expected_k = torch.zeros_like(native_k)
    expected_v = torch.zeros_like(native_v)
    use_e5 = formats.index_select(0, slots // block_size)[:, None, None] == 2
    expected_k.view(-1, heads, dim).index_copy_(
        0, slots, torch.where(use_e5, encode_e5m2(key), encode_e4m3(key))
    )
    expected_v.view(-1, heads, dim).index_copy_(
        0, slots, torch.where(use_e5, encode_e5m2(value), encode_e4m3(value))
    )
    mismatch = (native_k != expected_k).nonzero()[:12].cpu().tolist()
    if mismatch:
        for block, offset, head, element in mismatch:
            print(
                "K mismatch",
                (block, offset, head, element),
                "format",
                formats[block].item(),
                "native",
                native_k[block, offset, head, element].item(),
                "reference",
                expected_k[block, offset, head, element].item(),
                flush=True,
            )
    torch.testing.assert_close(native_k, expected_k, rtol=0, atol=0)
    torch.testing.assert_close(native_v, expected_v, rtol=0, atol=0)
    for payload in (native_k, native_v):
        actual = torch.ops.psp_kv_ascend_native.decode_fp8(payload, formats)
        expected = torch.where(
            formats[:, None, None, None] == 2,
            decode_e5m2(payload),
            decode_e4m3(payload),
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, equal_nan=True)
    print(
        f"FP8 scatter/decode matched: {slots.numel()} tokens, "
        f"{heads * dim} elements per token, {blocks} physical blocks",
        flush=True,
    )


def check_decode_attention(device: torch.device) -> None:
    """Compare physical-page native decode against fully decoded torch attention."""
    torch.manual_seed(42)
    block_size, kv_heads, heads, dim = 16, 2, 4, 32
    formats = torch.tensor([2, 1, 1], dtype=torch.uint8, device=device)
    block_ids = torch.tensor([2, 0], dtype=torch.int32, device=device)
    key = torch.randn(25, kv_heads, dim, dtype=torch.bfloat16, device=device)
    value = torch.randn_like(key)
    query = torch.randn(1, heads, dim, dtype=torch.bfloat16, device=device)
    slots = torch.tensor(
        list(range(2 * block_size, 3 * block_size)) + list(range(9)),
        dtype=torch.int64,
        device=device,
    )
    shape = (3, block_size, kv_heads, dim)
    key_cache = torch.zeros(shape, dtype=torch.uint8, device=device)
    value_cache = torch.zeros_like(key_cache)
    torch.ops.psp_kv_ascend_native.quantize_scatter_bf16(
        key, value, slots, formats, key_cache, value_cache
    )
    full_keys = (
        torch.where(
            formats[:, None, None, None] == 2,
            decode_e5m2(key_cache),
            decode_e4m3(key_cache),
        )
        .index_select(0, block_ids.long())
        .reshape(-1, kv_heads, dim)
    )
    full_values = (
        torch.where(
            formats[:, None, None, None] == 2,
            decode_e5m2(value_cache),
            decode_e4m3(value_cache),
        )
        .index_select(0, block_ids.long())
        .reshape(-1, kv_heads, dim)
    )
    scale = dim**-0.5
    for seq_len in (1, 16, 17, 25):
        scores = torch.ops.psp_kv_ascend_native.decode_scores_bf16(
            query, key_cache, formats, block_ids, seq_len, scale
        )
        assert scores.shape == (heads, 16 if seq_len <= 16 else 32)
        actual = torch.ops.psp_kv_ascend_native.decode_weighted_value(
            F.softmax(scores, dim=-1), value_cache, formats, block_ids, seq_len
        )
        expanded_keys = full_keys[:seq_len].repeat_interleave(2, dim=1)
        expanded_values = full_values[:seq_len].repeat_interleave(2, dim=1)
        expected_scores = (
            torch.einsum("qhd,khd->hk", query.float(), expanded_keys) * scale
        )
        torch.testing.assert_close(
            scores[:, :seq_len], expected_scores, atol=2e-3, rtol=2e-3
        )
        expected = torch.einsum(
            "hk,khd->hd", F.softmax(expected_scores, dim=-1), expanded_values
        ).unsqueeze(0)
        torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-3)
        print(f"FP8 native attention matched: {seq_len} tokens", flush=True)

    # More than 40 query heads exercises the kernel's strided core loop.
    many_heads_query = torch.randn(1, 48, dim, dtype=torch.bfloat16, device=device)
    many_heads_scores = torch.ops.psp_kv_ascend_native.decode_scores_bf16(
        many_heads_query, key_cache, formats, block_ids, 25, scale
    )
    many_heads_output = torch.ops.psp_kv_ascend_native.decode_weighted_value(
        F.softmax(many_heads_scores, dim=-1),
        value_cache,
        formats,
        block_ids,
        25,
    )
    expected_scores = (
        torch.einsum(
            "qhd,khd->hk",
            many_heads_query.float(),
            full_keys[:25].repeat_interleave(24, dim=1),
        )
        * scale
    )
    expected_output = torch.einsum(
        "hk,khd->hd",
        F.softmax(expected_scores, dim=-1),
        full_values[:25].repeat_interleave(24, dim=1),
    ).unsqueeze(0)
    torch.testing.assert_close(
        many_heads_scores[:, :25], expected_scores, atol=2e-3, rtol=2e-3
    )
    torch.testing.assert_close(many_heads_output, expected_output, atol=2e-3, rtol=2e-3)
    print("FP8 native attention matched: 48 query heads", flush=True)


def main() -> None:
    library = os.environ["PSPKV_ASCEND_NATIVE_LIB"]
    torch.ops.load_library(library)
    torch.npu.set_device(0)
    device = torch.device("npu:0")
    block_size, heads, dim = 4, 2, 16
    formats = torch.tensor([1, 2, 1], dtype=torch.uint8, device=device)
    slots = torch.tensor([0, 5, 8, 6], dtype=torch.int64, device=device)
    # Cover normal and subnormal values, signed zero, overflow, and specials.
    edge_values = torch.tensor(
        [
            -float("inf"),
            -57344.0,
            -448.0,
            -240.0,
            -1.0,
            -0.1,
            -0.0078125,
            -0.001953125,
            -0.0,
            0.0,
            0.001953125,
            0.0078125,
            0.1,
            1.0,
            240.0,
            448.0,
            57344.0,
            float("inf"),
            float("nan"),
        ],
        dtype=torch.float32,
    )
    base = edge_values.repeat(
        (slots.numel() * heads * dim + len(edge_values) - 1) // len(edge_values)
    )[: slots.numel() * heads * dim].reshape(slots.numel(), heads, dim)
    check_case(
        base.to(torch.bfloat16).to(device),
        (base * -0.75).to(torch.bfloat16).to(device),
        slots,
        formats,
        block_size,
    )
    check_decode_attention(device)

    # Non-64-byte token strides must use the single-core fallback even when
    # the whole request exceeds one 256-element work chunk.
    block_size, heads, dim = 16, 2, 16
    formats = torch.tensor([1, 2, 1, 2], dtype=torch.uint8, device=device)
    slots = torch.arange(24, device=device)
    base = torch.linspace(-10, 10, slots.numel() * heads * dim).reshape(
        slots.numel(), heads, dim
    )
    check_case(
        base.to(torch.bfloat16).to(device),
        (base * 1.25).to(torch.bfloat16).to(device),
        slots,
        formats,
        block_size,
    )

    # The small case runs on one AI Core after cache-line partitioning. This
    # model-sized case exercises many cores and nonadjacent physical pages.
    block_size, heads, dim = 128, 8, 128
    formats = torch.tensor([1, 1, 2, 1, 2, 1], dtype=torch.uint8, device=device)
    slots = torch.cat(
        [torch.arange(start, start + block_size) for start in (0, 256, 512)]
    ).to(device)
    base = torch.linspace(-300, 300, slots.numel() * heads * dim).reshape(
        slots.numel(), heads, dim
    )
    check_case(
        base.to(torch.bfloat16).to(device),
        (base * -0.25).to(torch.bfloat16).to(device),
        slots,
        formats,
        block_size,
    )


if __name__ == "__main__":
    main()
