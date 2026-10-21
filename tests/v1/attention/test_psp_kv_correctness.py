# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-block numerical validation of the PSP-KV kernels vs. the FP8 reference.

Covers every launcher (basic / enhanced / btkv) × FP8 format (e4m3 / e5m2) ×
MHA/GQA, plus the prefill quantize-and-scatter write path. Multi-block and
variable-context coverage lives in test_psp_kv_mixed_precision.py.
"""

import os

import pytest
import torch

from tests.v1.attention.psp_kv_reference import (
    dequantize_kv,
    pack_kv_cache,
    reference_paged_attention,
)
from vllm.v1.attention.backends.psp_kv_ascend_packed import PackedKVCache
from vllm.v1.attention.psp_kv_virthub import block_bytes

FMTS = [0, 1, 2, 3]  # 0 = lossless, 1 = e4m3, 2 = e5m2, 3 = fp4 (inner precision)
HEAD_SHAPES = [(8, 8), (8, 2)]  # (num_heads, num_kv_heads): MHA and GQA
DTYPES = [
    torch.float16,
    torch.bfloat16,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
]  # target precision
DEVICE = os.environ.get("PSPKV_TEST_DEVICE", "cuda")
pytestmark = pytest.mark.skip_global_cleanup if DEVICE == "npu" else []


def _psp_kv_ops_available():
    if DEVICE == "npu":
        try:
            from vllm.v1.attention.backends.psp_kv_ascend_native import (
                load_native_fp8_ops,
            )

            load_native_fp8_ops()
            return True
        except Exception:
            return False
    if not torch.cuda.is_available():
        return False
    try:
        import vllm._C  # noqa: F401  (registers torch.ops.psp_kv)

        _ops = getattr(torch.ops, "psp_kv", None)
        return _ops is not None and hasattr(_ops, "psp_attention_basic")
    except Exception:
        return False


SKIP_REASON = "CUDA and PSP-KV kernels are required"
skip_if_no_ops = pytest.mark.skipif(not _psp_kv_ops_available(), reason=SKIP_REASON)


class DummyLayer:
    def __init__(self, num_heads=8, num_kv_heads=8, head_size=128, scale=None):
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.num_kv_heads_per_partition = num_kv_heads
        self.head_size = head_size
        self.scale = scale if scale is not None else (1.0 / (head_size**0.5))


class _DummyCommon:
    """Plain stand-in for CommonAttentionMetadata; fields set per-test below."""


def _build_kv_cache(k_full, v_full, format_gen, fmt, dtype):
    """Build the byte buffer ([2, total_bytes] uint8) + block_offsets for one block.

    k_full/v_full: [ctx, num_kv_heads, head_size] fp16 (ctx == block_size).
    """
    k = k_full.permute(1, 0, 2).unsqueeze(0)  # [1, num_kv_heads, ctx, head_size]
    v = v_full.permute(1, 0, 2).unsqueeze(0)
    precision_table = torch.full((1,), fmt, dtype=torch.uint8, device=k_full.device)
    k_buf, v_buf, block_offsets = pack_kv_cache(
        k, v, format_gen, precision_table, dtype
    )
    return torch.stack([k_buf, v_buf]), block_offsets


def _run_and_compare(format_gen, num_heads, num_kv_heads, fmt, dtype):
    head_size = 128
    block_size = 16
    ctx = block_size
    softmax_scale = 1.0 / (head_size**0.5)

    if DEVICE == "npu":
        from vllm.v1.attention.backends.psp_kv_ascend_ops import PSPKVAscendPackedOps

        query_cpu = torch.randn(1, num_heads, head_size, dtype=torch.float16).to(dtype)
        k_cpu = torch.randn(ctx, num_kv_heads, head_size, dtype=torch.float16) * 2
        v_cpu = torch.randn(ctx, num_kv_heads, head_size, dtype=torch.float16) * 2
        cache_cpu, offsets_cpu = _build_kv_cache(k_cpu, v_cpu, format_gen, fmt, dtype)
        query = query_cpu.to(DEVICE)
        cache = cache_cpu.to(DEVICE)
        offsets = offsets_cpu.to(DEVICE)
        precision = torch.full((1,), fmt, dtype=torch.uint8, device=DEVICE)
        scales_cpu = (
            torch.tensor([[0.5, 1.0, 2.0, 0.5, 1.0, 2.0, 0.5, 1.0][:num_kv_heads]])
            if format_gen
            else None
        )
        scales = scales_cpu.to(DEVICE) if scales_cpu is not None else None
        tables = torch.zeros(1, 1, dtype=torch.int32, device=DEVICE)
        lengths = torch.tensor([ctx], dtype=torch.int32, device=DEVICE)
        output = torch.empty_like(query)
        ops = PSPKVAscendPackedOps()
        if format_gen == 0:
            ops.psp_attention_basic(
                query,
                cache[0],
                cache[1],
                tables,
                lengths,
                precision,
                offsets,
                softmax_scale,
                num_kv_heads,
                head_size,
                output,
            )
        elif format_gen == 1:
            ops.psp_attention_enhanced(
                query,
                cache[0],
                cache[1],
                tables,
                lengths,
                precision,
                offsets,
                scales,
                softmax_scale,
                num_kv_heads,
                head_size,
                output,
            )
        elif format_gen == 3:
            ops.psp_attention_scaled_basic(
                query,
                cache[0],
                cache[1],
                tables,
                lengths,
                precision,
                offsets,
                scales,
                softmax_scale,
                num_kv_heads,
                head_size,
                output,
            )
        else:
            descriptors = torch.zeros(1, 16, dtype=torch.int64, device=DEVICE)
            ops.psp_attention_btkv(
                query,
                cache[0],
                cache[1],
                tables,
                lengths,
                precision,
                offsets,
                scales,
                descriptors,
                softmax_scale,
                num_kv_heads,
                head_size,
                output,
            )
        reference = reference_paged_attention(
            query_cpu,
            cache_cpu[0],
            cache_cpu[1],
            tables.cpu(),
            lengths.cpu(),
            format_gen,
            precision.cpu(),
            scales_cpu,
            offsets_cpu,
            num_kv_heads,
            block_size,
            softmax_scale,
        )
        tolerance = 1.0 if dtype in (torch.float8_e4m3fn, torch.float8_e5m2) else 1e-2
        torch.testing.assert_close(
            output.cpu().float(), reference.float(), atol=tolerance, rtol=tolerance
        )
        return

    query = torch.randn(1, num_heads, head_size, dtype=torch.float16, device="cuda").to(
        dtype
    )
    k_full = (
        torch.randn(ctx, num_kv_heads, head_size, dtype=torch.float16, device="cuda")
        * 2.0
    )
    v_full = (
        torch.randn(ctx, num_kv_heads, head_size, dtype=torch.float16, device="cuda")
        * 2.0
    )

    kv_cache, block_offsets = _build_kv_cache(k_full, v_full, format_gen, fmt, dtype)
    precision_table = torch.full((1,), fmt, dtype=torch.uint8, device="cuda")
    scale_table = None
    tma_descriptors = None
    if format_gen != 0:
        scale_table = torch.tensor(
            [[0.5, 1.0, 2.0, 0.5, 1.0, 2.0, 0.5, 1.0][:num_kv_heads]],
            dtype=torch.float32,
            device="cuda",
        )
    if format_gen == 2:
        tma_descriptors = torch.zeros(1, 16, dtype=torch.int64, device="cuda")

    meta = _DummyCommon()
    meta.block_table_tensor = torch.zeros(1, 1, dtype=torch.int32, device="cuda")
    meta.context_lens = torch.tensor([ctx], dtype=torch.int32, device="cuda")
    meta.seq_lens = meta.context_lens
    meta.precision_table = precision_table
    meta.block_offsets = block_offsets
    meta.scale_table = scale_table
    meta.tma_descriptors = tma_descriptors
    meta.format_generation = format_gen
    meta.num_actual_tokens = 1
    meta.num_reqs = 1

    from vllm.v1.attention.backends.psp_kv_attn import (
        PSPKVAttentionImpl,
        PSPKVAttentionMetadata,
    )

    layer = DummyLayer(
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        scale=softmax_scale,
    )
    impl = PSPKVAttentionImpl(
        num_heads, head_size, softmax_scale, num_kv_heads, None, None, "auto"
    )
    output = torch.empty_like(query)
    impl.forward(
        layer=layer,
        query=query,
        key=torch.empty(0),
        value=torch.empty(0),
        kv_cache=kv_cache,
        attn_metadata=PSPKVAttentionMetadata(common_metadata=meta),
        output=output,
    )

    ref = reference_paged_attention(
        query,
        kv_cache[0],
        kv_cache[1],
        meta.block_table_tensor,
        meta.context_lens,
        format_gen,
        precision_table,
        scale_table,
        block_offsets,
        num_kv_heads,
        block_size,
        softmax_scale,
    )
    _atol = 1.0 if dtype in (torch.float8_e4m3fn, torch.float8_e5m2) else 1e-2
    torch.testing.assert_close(output.float(), ref.float(), atol=_atol, rtol=_atol)


@skip_if_no_ops
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("fmt", FMTS)
@pytest.mark.parametrize("num_heads,num_kv_heads", HEAD_SHAPES)
def test_basic_attention(num_heads, num_kv_heads, fmt, dtype):
    _run_and_compare(0, num_heads, num_kv_heads, fmt, dtype)


@skip_if_no_ops
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("fmt", FMTS)
@pytest.mark.parametrize("num_heads,num_kv_heads", HEAD_SHAPES)
def test_enhanced_attention(num_heads, num_kv_heads, fmt, dtype):
    _run_and_compare(1, num_heads, num_kv_heads, fmt, dtype)


@skip_if_no_ops
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("fmt", FMTS)
@pytest.mark.parametrize("num_heads,num_kv_heads", HEAD_SHAPES)
def test_btkv_attention(num_heads, num_kv_heads, fmt, dtype):
    if DEVICE == "cuda" and torch.cuda.get_device_capability()[0] < 9:
        pytest.skip("BT-KV requires Hopper (SM90+) or Blackwell")
    _run_and_compare(2, num_heads, num_kv_heads, fmt, dtype)


@skip_if_no_ops
@pytest.mark.skipif(DEVICE != "npu", reason="Ascend scaled basic layout only")
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("fmt", FMTS)
@pytest.mark.parametrize("num_heads,num_kv_heads", HEAD_SHAPES)
def test_scaled_basic_attention(num_heads, num_kv_heads, fmt, dtype):
    _run_and_compare(3, num_heads, num_kv_heads, fmt, dtype)


@skip_if_no_ops
@pytest.mark.skipif(DEVICE != "npu", reason="Ascend grouped FP4 scales only")
def test_scaled_basic_grouped_fp4_scatter_decode_and_attention():
    from vllm.v1.attention.backends.psp_kv_ascend_native import load_native_fp8_ops
    from vllm.v1.attention.backends.psp_kv_ascend_ops import PSPKVAscendPackedOps

    block_size, kv_heads, head_size = 16, 2, 128
    slots = torch.arange(block_size, dtype=torch.int64, device=DEVICE)
    keys = (torch.randint(-4, 5, (block_size, kv_heads, head_size)) * 0.5).to(
        torch.bfloat16
    )
    values = (torch.randint(-4, 5, (block_size, kv_heads, head_size)) * 0.125).to(
        torch.bfloat16
    )
    scales = torch.ones((1, 2, kv_heads, 8), dtype=torch.float32)
    scales[:, 0, :, 1::2] = 2.0
    scales[:, 1, :, :] = 0.25
    scales[:, 1, :, 1::2] = 0.5
    key_scale = scales[0, 0].repeat_interleave(16, dim=-1)
    value_scale = scales[0, 1].repeat_interleave(16, dim=-1)
    reference = PackedKVCache([3], block_size, kv_heads, head_size, torch.device("cpu"))
    reference.write(slots.cpu(), keys.float() / key_scale, values.float() / value_scale)

    key_cache = torch.zeros(reference.total_bytes, dtype=torch.uint8, device=DEVICE)
    value_cache = torch.zeros_like(key_cache)
    formats = torch.tensor([3], dtype=torch.uint8, device=DEVICE)
    offsets = torch.tensor([0], dtype=torch.int32, device=DEVICE)
    blocks = torch.tensor([[0]], dtype=torch.int32, device=DEVICE)
    native_scales = scales.to(DEVICE)
    prefill = torch.stack((keys, values), dim=1).to(DEVICE).contiguous()
    ops = PSPKVAscendPackedOps()
    ops.quantize_and_scatter_kv(
        prefill,
        key_cache,
        value_cache,
        blocks,
        slots,
        formats,
        offsets,
        native_scales,
        block_size,
        format_generation=3,
        preserve_scales=True,
        block_size=block_size,
    )
    assert torch.equal(key_cache.cpu(), reference.k_buffer)
    assert torch.equal(value_cache.cpu(), reference.v_buffer)

    decoded_k, decoded_v = load_native_fp8_ops().packed_dequant(
        key_cache,
        value_cache,
        offsets,
        formats,
        native_scales,
        block_size,
        head_size,
        0,
        3,
    )
    ref_k, ref_v = reference.gather([0])
    expected_k = (ref_k[0] * key_scale[None]).permute(1, 0, 2)
    expected_v = (ref_v[0] * value_scale[None]).permute(1, 0, 2)
    torch.testing.assert_close(decoded_k[0].cpu(), expected_k)
    torch.testing.assert_close(decoded_v[0].cpu(), expected_v)

    query = torch.ones((1, 4, head_size), dtype=torch.bfloat16, device=DEVICE)
    context_lens = torch.tensor([block_size], dtype=torch.int32, device=DEVICE)
    result = torch.empty_like(query)
    ops.psp_attention_scaled_basic(
        query,
        key_cache,
        value_cache,
        blocks,
        context_lens,
        formats,
        offsets,
        native_scales,
        head_size**-0.5,
        kv_heads,
        head_size,
        result,
        block_size,
    )
    expanded_k = expected_k.permute(1, 0, 2).repeat_interleave(2, dim=1)
    expanded_v = expected_v.permute(1, 0, 2).repeat_interleave(2, dim=1)
    scores = (
        torch.bmm(
            query.cpu().float().transpose(0, 1),
            expanded_k.transpose(0, 1).transpose(1, 2),
        )
        * head_size**-0.5
    )
    expected = torch.bmm(
        torch.softmax(scores, dim=-1), expanded_v.transpose(0, 1)
    ).transpose(0, 1)
    torch.testing.assert_close(result.cpu().float(), expected, rtol=0.02, atol=0.02)
    fused_result = torch.empty_like(query)
    ops.psp_attention_scaled_basic_fused(
        query,
        key_cache,
        value_cache,
        blocks,
        context_lens,
        formats,
        offsets,
        native_scales,
        head_size**-0.5,
        kv_heads,
        head_size,
        fused_result,
        block_size,
    )
    torch.testing.assert_close(
        fused_result.cpu().float(), expected, rtol=0.02, atol=0.02
    )


@skip_if_no_ops
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("fmt", FMTS)
def test_quantize_and_scatter_roundtrip(fmt, dtype):
    """The prefill write path quantizes fp16/bf16 -> the byte buffer."""
    num_tokens, num_kv_heads, head_size, block_size = 16, 8, 128, 16

    # Integer inputs in [0, 3] make FP4/FP8 quantization exact.
    prefill_cpu = torch.randint(
        0, 4, (num_tokens, 2, num_kv_heads, head_size), dtype=torch.float16
    ).to(dtype)
    prefill = prefill_cpu.to(DEVICE)

    precision_table = torch.full((1,), fmt, dtype=torch.uint8, device=DEVICE)
    block_offsets = torch.zeros(1, dtype=torch.int32, device=DEVICE)
    # one block's byte size
    scalar_size = 2 if dtype in (torch.float16, torch.bfloat16) else 1
    total = block_bytes(fmt, num_kv_heads, block_size, head_size, scalar_size)
    key_cache = torch.zeros(total, dtype=torch.uint8, device=DEVICE)
    value_cache = torch.zeros(total, dtype=torch.uint8, device=DEVICE)
    block_tables = torch.zeros(1, 1, dtype=torch.int32, device=DEVICE)
    slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device=DEVICE)
    scale_table = torch.zeros(1, num_kv_heads, dtype=torch.float32, device=DEVICE)

    if DEVICE == "npu":
        from vllm.v1.attention.backends.psp_kv_ascend_ops import PSPKVAscendPackedOps

        op = PSPKVAscendPackedOps().quantize_and_scatter_kv
    else:
        op = torch.ops.psp_kv.quantize_and_scatter_kv
    op(
        prefill,
        key_cache,
        value_cache,
        block_tables,
        slot_mapping,
        precision_table,
        block_offsets,
        scale_table,
        num_tokens,
    )

    # expected byte buffer via the reference packer
    k = prefill_cpu[:, 0].permute(1, 0, 2).unsqueeze(0)
    v = prefill_cpu[:, 1].permute(1, 0, 2).unsqueeze(0)
    expected_k, expected_v, _ = pack_kv_cache(k, v, 0, precision_table.cpu(), dtype)
    assert torch.equal(key_cache.cpu(), expected_k)
    assert torch.equal(value_cache.cpu(), expected_v)
    assert torch.equal(scale_table.cpu(), torch.ones(1, num_kv_heads))

    if DEVICE == "npu" and fmt == 3:
        # The live uniform-FP4 layout keeps token-major V and per-head scales.
        key_cache.zero_()
        value_cache.zero_()
        scale_table.fill_(2.0)
        op(
            prefill,
            key_cache,
            value_cache,
            block_tables,
            slot_mapping,
            precision_table,
            block_offsets,
            scale_table,
            num_tokens,
            format_generation=3,
            preserve_scales=True,
        )
        scaled_k, scaled_v, _ = pack_kv_cache(
            k, v, 3, precision_table.cpu(), dtype, scale=2.0
        )
        assert torch.equal(key_cache.cpu(), scaled_k)
        assert torch.equal(value_cache.cpu(), scaled_v)
        assert torch.equal(scale_table.cpu(), torch.full((1, num_kv_heads), 2.0))


@skip_if_no_ops
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("fmt", FMTS)
def test_dequantize_kv_staging_path(fmt, dtype):
    """The staging path outputs uniform target-precision KV from the packed cache."""
    num_blocks, num_kv_heads, block_size, head_size = 2, 8, 16, 128
    k = torch.randn(num_blocks, num_kv_heads, block_size, head_size) * 2.0
    v = torch.randn(num_blocks, num_kv_heads, block_size, head_size) * 2.0
    precision_cpu = torch.full((num_blocks,), fmt, dtype=torch.uint8)
    k_buf_cpu, v_buf_cpu, offsets_cpu = pack_kv_cache(k, v, 0, precision_cpu, dtype)
    precision_table = precision_cpu.to(DEVICE)
    k_buf, v_buf = k_buf_cpu.to(DEVICE), v_buf_cpu.to(DEVICE)
    block_offsets = offsets_cpu.to(DEVICE)
    scale_table = torch.ones(
        num_blocks, num_kv_heads, dtype=torch.float32, device=DEVICE
    )

    k_out = torch.empty(
        num_blocks, num_kv_heads, block_size, head_size, dtype=dtype, device=DEVICE
    )
    v_out = torch.empty_like(k_out)
    if DEVICE == "npu":
        from vllm.v1.attention.backends.psp_kv_ascend_ops import PSPKVAscendPackedOps

        op = PSPKVAscendPackedOps().psp_dequantize_kv
    else:
        op = torch.ops.psp_kv.psp_dequantize_kv
    op(
        k_buf,
        v_buf,
        block_offsets,
        precision_table,
        scale_table,
        num_blocks,
        num_kv_heads,
        block_size,
        head_size,
        k_out,
        v_out,
    )
    if DEVICE == "npu":
        torch.npu.synchronize()
    else:
        torch.accelerator.synchronize()

    ref_k, ref_v = dequantize_kv(
        k_buf_cpu,
        v_buf_cpu,
        offsets_cpu,
        precision_cpu,
        scale_table.cpu(),
        num_kv_heads,
        block_size,
        head_size,
        dtype,
    )
    torch.testing.assert_close(k_out.cpu().float(), ref_k.float(), atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(v_out.cpu().float(), ref_v.float(), atol=1e-2, rtol=1e-2)

    if DEVICE == "npu":
        # Continue the existing basic staging path into its dense consumer.
        query_cpu = torch.randn(1, num_kv_heads, head_size, dtype=torch.float16)
        query_cpu = query_cpu.to(dtype)
        query = query_cpu.to(DEVICE)
        tables = torch.tensor([[0, 1]], dtype=torch.int32, device=DEVICE)
        lengths = torch.tensor([2 * block_size], dtype=torch.int32, device=DEVICE)
        output = torch.empty_like(query)
        PSPKVAscendPackedOps().psp_attention_uniform(
            query,
            k_out,
            v_out,
            tables,
            lengths,
            1.0 / (head_size**0.5),
            num_kv_heads,
            head_size,
            output,
        )
        reference = reference_paged_attention(
            query_cpu,
            k_buf_cpu,
            v_buf_cpu,
            tables.cpu(),
            lengths.cpu(),
            0,
            precision_cpu,
            scale_table.cpu(),
            offsets_cpu,
            num_kv_heads,
            block_size,
            1.0 / (head_size**0.5),
        )
        tolerance = 1.0 if dtype in (torch.float8_e4m3fn, torch.float8_e5m2) else 1e-2
        torch.testing.assert_close(
            output.cpu().float(),
            reference.float(),
            atol=tolerance,
            rtol=tolerance,
        )
