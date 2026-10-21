# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Verify the vectorized packed kernels and the native variable-width path.

Compares, on one NPU:
1. the UB-staged vectorized pack kernels against the scalar kernels
   (scores and weighted-value must match bit-for-bit);
2. the native variable-width arena attention (packed kernels over per-block
   offsets/formats/scales) against the eager reference path;
3. static (allocation-time) FP4 blocks against the same reference.

Run inside the vLLM-Ascend image with PSPKV_ASCEND_NATIVE_LIB set.
"""

import torch
import torch_npu  # noqa: F401 - registers the NPU device with PyTorch

from vllm.v1.attention.backends.psp_kv_ascend_arena import (
    VariableArenaKVCache,
    reference_variable_attention,
)
from vllm.v1.attention.backends.psp_kv_ascend_native import load_native_fp8_ops
from vllm.v1.attention.backends.psp_kv_ascend_ops import packed_lookup_tables

BLOCK = 128
HEADS = 2
KV_HEADS = 2
DIM = 128
DEVICE = torch.device("npu:0")


def _arena(
    static_code: int | None = None,
    static_scales: torch.Tensor | None = None,
    max_relative_rmse: float = 0.5,
) -> VariableArenaKVCache:
    # 6 blocks: enough for a 300-token sequence plus a second lifetime.
    block_bytes = BLOCK * KV_HEADS * DIM
    size = 6 * block_bytes
    return VariableArenaKVCache(
        num_blocks=6,
        block_size=BLOCK,
        num_kv_heads=KV_HEADS,
        head_size=DIM,
        device=DEVICE,
        k_buffer=torch.zeros(size, dtype=torch.uint8, device=DEVICE),
        v_buffer=torch.zeros(size, dtype=torch.uint8, device=DEVICE),
        avg_bytes=1.0,
        scalar_dtype=torch.bfloat16,
        max_relative_rmse=max_relative_rmse,
        layer_name="check.layer",
        static_code=static_code,
        static_scales=static_scales,
    )


def _cpu_randn(*shape: int, seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(*shape).bfloat16().to(DEVICE)


def _fill(arena: VariableArenaKVCache, num_tokens: int) -> tuple:
    key = _cpu_randn(num_tokens, KV_HEADS, DIM, seed=101)
    value = _cpu_randn(num_tokens, KV_HEADS, DIM, seed=102)
    slots = torch.arange(num_tokens, dtype=torch.int64, device=DEVICE)
    live = {int(slot) // BLOCK for slot in range(num_tokens)}
    arena.begin_writes(slots, live)
    arena.write_and_finalize(slots, key, value, native=True)
    arena.sync_device_metadata()
    return key, value


def _context(num_tokens: int) -> tuple[torch.Tensor, torch.Tensor]:
    # One extra decode token may land in a new block.
    max_blocks = (num_tokens + 1 + BLOCK - 1) // BLOCK
    tables = torch.arange(max_blocks, dtype=torch.int32, device=DEVICE)[None]
    lengths = torch.tensor([num_tokens], dtype=torch.int32, device=DEVICE)
    return tables, lengths


def check_vectorized_kernels(arena: VariableArenaKVCache, num_tokens: int) -> None:
    ops = load_native_fp8_ops()
    e4m3, e5m2, fp4 = packed_lookup_tables(DEVICE)
    query = _cpu_randn(1, HEADS, DIM, seed=201)
    tables, lengths = _context(num_tokens)
    scalar_scores = ops.packed_scores(
        query,
        arena.k_buffer,
        tables,
        lengths,
        arena.device_codes,
        arena.device_offsets,
        arena.scales,
        3,
        KV_HEADS,
        BLOCK,
        0.125,
    )
    vector_scores = ops.packed_scores_vec(
        query,
        arena.k_buffer,
        tables,
        lengths,
        arena.device_codes,
        arena.device_offsets,
        arena.scales,
        e4m3,
        e5m2,
        fp4,
        3,
        KV_HEADS,
        BLOCK,
        0.125,
    )
    torch.testing.assert_close(vector_scores, scalar_scores, rtol=0, atol=0)
    probabilities = torch.softmax(scalar_scores, dim=-1)
    scalar_values = ops.packed_weighted_value(
        probabilities,
        arena.v_buffer,
        tables,
        lengths,
        arena.device_codes,
        arena.device_offsets,
        arena.scales,
        3,
        KV_HEADS,
        DIM,
        BLOCK,
        0,
    )
    vector_values = ops.packed_weighted_value_vec(
        probabilities,
        arena.v_buffer,
        tables,
        lengths,
        arena.device_codes,
        arena.device_offsets,
        arena.scales,
        e4m3,
        e5m2,
        fp4,
        3,
        KV_HEADS,
        DIM,
        BLOCK,
        0,
    )
    torch.testing.assert_close(vector_values, scalar_values, rtol=0, atol=0)
    print(f"  vectorized kernels match scalar kernels ({num_tokens} tokens)")


def run_decode(
    arena: VariableArenaKVCache,
    num_tokens: int,
    native: bool,
    native_fused: bool = False,
) -> torch.Tensor:
    key = _cpu_randn(1, KV_HEADS, DIM, seed=301)
    value = _cpu_randn(1, KV_HEADS, DIM, seed=302)
    query = _cpu_randn(1, HEADS, DIM, seed=303)
    output = torch.zeros(1, HEADS, DIM, dtype=torch.bfloat16, device=DEVICE)
    tables, _ = _context(num_tokens)
    slot = torch.tensor([num_tokens], dtype=torch.int64, device=DEVICE)
    reference_variable_attention(
        query,
        key,
        value,
        arena,
        slot,
        tables,
        [1],
        [num_tokens + 1],
        output,
        0.125,
        native=native,
        native_fused=native_fused,
    )
    return output


def main() -> None:
    torch.npu.set_device(0)
    # FP4 blocks at an intentionally loose threshold exercise format 3.
    arena = _arena(max_relative_rmse=0.5)
    _fill(arena, 2 * BLOCK)
    counts = arena.format_counts()
    assert counts.get(3), f"expected FP4 blocks in the checker, got {counts}"
    print(f"  formats {counts}")
    check_vectorized_kernels(arena, 2 * BLOCK)

    native_arena = _arena(max_relative_rmse=0.5)
    _fill(native_arena, 2 * BLOCK)
    reference_arena = _arena(max_relative_rmse=0.5)
    _fill(reference_arena, 2 * BLOCK)
    native_out = run_decode(native_arena, 2 * BLOCK, native=True)
    reference_out = run_decode(reference_arena, 2 * BLOCK, native=False)
    torch.testing.assert_close(native_out, reference_out, rtol=1e-4, atol=1e-4)
    print("  native variable decode matches the reference path")

    fused_arena = _arena(max_relative_rmse=0.5)
    _fill(fused_arena, 2 * BLOCK)
    fused_out = run_decode(fused_arena, 2 * BLOCK, native=True, native_fused=True)
    torch.testing.assert_close(fused_out, reference_out, rtol=0.02, atol=0.02)
    print("  fused compressed-cache decode matches the reference path")

    scales = torch.full((KV_HEADS,), 0.25, dtype=torch.float32, device=DEVICE)
    static_arena = _arena(static_code=3, static_scales=scales)
    _fill(static_arena, 2 * BLOCK)
    assert static_arena._sizes[0] == BLOCK * KV_HEADS * DIM // 2
    static_out = run_decode(static_arena, 2 * BLOCK, native=True)
    static_reference = _arena(static_code=3, static_scales=scales)
    _fill(static_reference, 2 * BLOCK)
    static_out_ref = run_decode(static_reference, 2 * BLOCK, native=False)
    torch.testing.assert_close(static_out, static_out_ref, rtol=1e-4, atol=1e-4)
    print("  static allocation-time FP4 decode matches the reference path")

    # Grouped K/V-independent static scales (MXFP4-style, groups of 32).
    grouped = torch.stack(
        (
            torch.full((KV_HEADS, 4), 0.25, dtype=torch.float32, device=DEVICE),
            torch.full((KV_HEADS, 4), 1.0, dtype=torch.float32, device=DEVICE),
        ),
        dim=0,
    )
    grouped_native = _arena(static_code=3, static_scales=grouped)
    _fill(grouped_native, 2 * BLOCK)
    grouped_reference = _arena(static_code=3, static_scales=grouped)
    _fill(grouped_reference, 2 * BLOCK)
    torch.testing.assert_close(
        run_decode(grouped_native, 2 * BLOCK, native=True),
        run_decode(grouped_reference, 2 * BLOCK, native=False),
        rtol=1e-4,
        atol=1e-4,
    )
    print("  grouped static FP4 (K/V independent) matches the reference path")
    print("NPU vectorized kernels and allocation-time precision verified")


if __name__ == "__main__":
    main()
