# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Single-block numerical validation of the PSP-KV kernels vs. the FP8 reference.

Covers every launcher (basic / enhanced / btkv) × FP8 format (e4m3 / e5m2) ×
MHA/GQA, plus the prefill quantize-and-scatter write path. Multi-block and
variable-context coverage lives in test_psp_kv_mixed_precision.py.
"""

import pytest
import torch

from vllm.v1.attention.backends.psp_kv_attn import (
    PSPKVAttentionImpl,
    PSPKVAttentionMetadata,
)

from vllm.v1.attention.psp_kv_virthub import block_bytes

from tests.v1.attention.psp_kv_reference import (
    dequantize_kv,
    pack_kv_cache,
    reference_paged_attention,
)

FMTS = [0, 1, 2, 3]    # 0 = lossless, 1 = e4m3, 2 = e5m2, 3 = fp4 (inner precision)
HEAD_SHAPES = [(8, 8), (8, 2)]  # (num_heads, num_kv_heads): MHA and GQA
DTYPES = [torch.float16, torch.bfloat16, torch.float8_e4m3fn, torch.float8_e5m2]  # target precision


def _psp_kv_ops_available():
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
        self.scale = scale if scale is not None else (1.0 / (head_size ** 0.5))


class _DummyCommon:
    """Plain stand-in for CommonAttentionMetadata; fields set per-test below."""


def _build_kv_cache(k_full, v_full, format_gen, fmt, dtype):
    """Build the byte buffer ([2, total_bytes] uint8) + block_offsets for one block.

    k_full/v_full: [ctx, num_kv_heads, head_size] fp16 (ctx == block_size).
    """
    k = k_full.permute(1, 0, 2).unsqueeze(0)  # [1, num_kv_heads, ctx, head_size]
    v = v_full.permute(1, 0, 2).unsqueeze(0)
    precision_table = torch.full((1,), fmt, dtype=torch.uint8, device=k_full.device)
    k_buf, v_buf, block_offsets = pack_kv_cache(k, v, format_gen, precision_table, dtype)
    return torch.stack([k_buf, v_buf]), block_offsets


def _run_and_compare(format_gen, num_heads, num_kv_heads, fmt, dtype):
    head_size = 128
    block_size = 16
    ctx = block_size
    softmax_scale = 1.0 / (head_size ** 0.5)

    query = torch.randn(1, num_heads, head_size, dtype=torch.float16, device="cuda").to(dtype)
    k_full = torch.randn(ctx, num_kv_heads, head_size, dtype=torch.float16, device="cuda") * 2.0
    v_full = torch.randn(ctx, num_kv_heads, head_size, dtype=torch.float16, device="cuda") * 2.0

    kv_cache, block_offsets = _build_kv_cache(k_full, v_full, format_gen, fmt, dtype)
    precision_table = torch.full((1,), fmt, dtype=torch.uint8, device="cuda")
    scale_table = None
    tma_descriptors = None
    if format_gen != 0:
        scale_table = torch.tensor(
            [[0.5, 1.0, 2.0, 0.5, 1.0, 2.0, 0.5, 1.0][:num_kv_heads]],
            dtype=torch.float32, device="cuda",
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

    layer = DummyLayer(num_heads=num_heads, num_kv_heads=num_kv_heads, head_size=head_size, scale=softmax_scale)
    impl = PSPKVAttentionImpl(num_heads, head_size, softmax_scale, num_kv_heads, None, None, "auto")
    output = torch.empty_like(query)
    impl.forward(
        layer=layer, query=query, key=torch.empty(0), value=torch.empty(0),
        kv_cache=kv_cache,
        attn_metadata=PSPKVAttentionMetadata(common_metadata=meta),
        output=output,
    )

    ref = reference_paged_attention(
        query, kv_cache[0], kv_cache[1], meta.block_table_tensor, meta.context_lens,
        format_gen, precision_table, scale_table, block_offsets,
        num_kv_heads, block_size, softmax_scale,
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
    major, _ = torch.cuda.get_device_capability()
    if major < 9:
        pytest.skip("BT-KV requires Hopper (SM90+) or Blackwell")
    _run_and_compare(2, num_heads, num_kv_heads, fmt, dtype)


@skip_if_no_ops
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("fmt", FMTS)
def test_quantize_and_scatter_roundtrip(fmt, dtype):
    """The prefill write path quantizes fp16/bf16 -> the byte buffer."""
    num_tokens, num_kv_heads, head_size, block_size = 16, 8, 128, 16

    # integer-valued inputs in [0, 3] so fp4/fp8 quantization is exact (lossless is exact).
    prefill = torch.randint(0, 4, (num_tokens, 2, num_kv_heads, head_size),
                            dtype=torch.float16, device="cuda").to(dtype)

    precision_table = torch.full((1,), fmt, dtype=torch.uint8, device="cuda")
    block_offsets = torch.zeros(1, dtype=torch.int32, device="cuda")
    # one block's byte size
    scalar_size = 2 if dtype in (torch.float16, torch.bfloat16) else 1
    total = block_bytes(fmt, num_kv_heads, block_size, head_size, scalar_size)
    key_cache = torch.zeros(total, dtype=torch.uint8, device="cuda")
    value_cache = torch.zeros(total, dtype=torch.uint8, device="cuda")
    block_tables = torch.zeros(1, 1, dtype=torch.int32, device="cuda")
    slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device="cuda")  # block 0, offsets 0..15
    scale_table = torch.zeros(1, num_kv_heads, dtype=torch.float32, device="cuda")

    torch.ops.psp_kv.quantize_and_scatter_kv(
        prefill, key_cache, value_cache, block_tables, slot_mapping,
        precision_table, block_offsets, scale_table, num_tokens, 1,
    )

    # Per-block scale = max_abs / fp_max(fmt), matching the quantize kernel.
    # The test uses a single block, so max_abs is the global prefill max.
    fp_max = {0: 1.0, 1: 240.0, 2: 57344.0, 3: 3.0}[fmt]
    max_abs = prefill.float().abs().max().item()
    scale = max_abs / fp_max if fmt != 0 else 1.0
    if scale < 1e-8:
        scale = 1.0

    # expected byte buffer via the reference packer. The quantize-and-scatter
    # op writes V token-major (all formats), so use format_gen=1 for the scale.
    k = prefill[:, 0].permute(1, 0, 2).unsqueeze(0)  # [1, num_kv_heads, num_tokens, head_size]
    v = prefill[:, 1].permute(1, 0, 2).unsqueeze(0)
    expected_k, expected_v, _ = pack_kv_cache(k, v, 1, precision_table, dtype, scale=scale)
    assert torch.equal(key_cache, expected_k)
    assert torch.equal(value_cache, expected_v)
    assert torch.equal(
        scale_table, torch.full((1, num_kv_heads), scale, device="cuda"))


@skip_if_no_ops
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("fmt", FMTS)
def test_dequantize_kv_staging_path(fmt, dtype):
    """The staging path outputs uniform target-precision KV from the packed cache."""
    num_blocks, num_kv_heads, block_size, head_size = 2, 8, 16, 128
    k = torch.randn(num_blocks, num_kv_heads, block_size, head_size, device="cuda") * 2.0
    v = torch.randn(num_blocks, num_kv_heads, block_size, head_size, device="cuda") * 2.0
    precision_table = torch.full((num_blocks,), fmt, dtype=torch.uint8, device="cuda")
    k_buf, v_buf, block_offsets = pack_kv_cache(k, v, 0, precision_table, dtype)
    scale_table = torch.ones(num_blocks, num_kv_heads, dtype=torch.float32, device="cuda")

    k_out = torch.empty(num_blocks, num_kv_heads, block_size, head_size, dtype=dtype, device="cuda")
    v_out = torch.empty_like(k_out)
    torch.ops.psp_kv.psp_dequantize_kv(
        k_buf, v_buf, block_offsets, precision_table, scale_table,
        num_blocks, num_kv_heads, block_size, head_size, k_out, v_out,
    )
    torch.cuda.synchronize()

    ref_k, ref_v = dequantize_kv(k_buf, v_buf, block_offsets, precision_table,
                                 scale_table, num_kv_heads, block_size, head_size, dtype)
    torch.testing.assert_close(k_out.float(), ref_k.float(), atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(v_out.float(), ref_v.float(), atol=1e-2, rtol=1e-2)
