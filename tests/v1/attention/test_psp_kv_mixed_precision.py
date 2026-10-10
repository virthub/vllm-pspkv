# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Multi-block, variable-context validation of the PSP-KV kernels vs. the FP8
reference (basic / enhanced / btkv × e4m3/e5m2 × MHA/GQA), plus a check that the
per-block precision_table format selector is honoured.

See docs/PSP_KV_FORMAT_CONTRACT.md.
"""

import math

import pytest
import torch

from tests.v1.attention.psp_kv_reference import pack_kv_cache, reference_paged_attention

FMTS = [0, 1, 2, 3]  # 0 = lossless, 1 = e4m3, 2 = e5m2, 3 = fp4 (inner precision)
HEAD_SHAPES = [(8, 8), (8, 2)]
FORMATS = [0, 1, 2]  # basic, enhanced, btkv
DTYPES = [torch.float16, torch.bfloat16, torch.float8_e4m3fn, torch.float8_e5m2]  # target precision


def _get_ops():
    import vllm._C  # noqa: F401  (registers torch.ops.psp_kv)
    return getattr(torch.ops, "psp_kv", None)


def _build_multiblock_cache(num_blocks, num_kv_heads, block_size, head_dim, format_gen, precision_table, dtype, device):
    """Build multi-block byte buffer + block_offsets."""
    k = torch.randn(num_blocks, num_kv_heads, block_size, head_dim, device=device) * 2.0
    v_token_major = torch.randn(num_blocks, num_kv_heads, block_size, head_dim, device=device) * 2.0
    k_buf, v_buf, block_offsets = pack_kv_cache(k, v_token_major, format_gen, precision_table, dtype)
    return k_buf, v_buf, block_offsets


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for PSP-KV kernel execution")
class TestPSPKVMultiblock:

    @pytest.fixture(autouse=True)
    def setup_method(self):
        self.ops = _get_ops()
        if self.ops is None or not hasattr(self.ops, "psp_attention_enhanced"):
            pytest.skip("psp_attention_* op is not compiled or registered in torch.ops")

    @pytest.mark.parametrize("dtype", DTYPES)
    @pytest.mark.parametrize("format_gen", FORMATS)
    @pytest.mark.parametrize("fmt", FMTS)
    @pytest.mark.parametrize("num_heads,num_kv_heads", HEAD_SHAPES)
    def test_multiblock_matches_reference(self, format_gen, fmt, num_heads, num_kv_heads, dtype):
        if format_gen == 2 and torch.cuda.get_device_capability()[0] < 9:
            pytest.skip("BT-KV requires Hopper (SM90+) or Blackwell")

        torch.manual_seed(42)
        head_dim = 128
        block_size = 16
        num_seqs = 4
        max_blocks_per_seq = 8
        num_blocks = num_seqs * max_blocks_per_seq
        softmax_scale = 1.0 / (head_dim ** 0.5)

        context_lens = torch.tensor([17, 32, 63, 112], dtype=torch.int32, device="cuda")

        block_tables = torch.zeros((num_seqs, max_blocks_per_seq), dtype=torch.int32, device="cuda")
        curr_b = 0
        for i in range(num_seqs):
            needed = math.ceil(int(context_lens[i].item()) / block_size)
            block_tables[i, :needed] = torch.arange(curr_b, curr_b + needed, dtype=torch.int32, device="cuda")
            curr_b += needed

        query = torch.randn((num_seqs, num_heads, head_dim), dtype=torch.float16, device="cuda").to(dtype)
        precision_table = torch.full((num_blocks,), fmt, dtype=torch.uint8, device="cuda")
        key_cache, value_cache, block_offsets = _build_multiblock_cache(
            num_blocks, num_kv_heads, block_size, head_dim, format_gen, precision_table, dtype, "cuda"
        )
        scale_table = torch.rand(num_blocks, num_kv_heads, device="cuda") * 1.5 + 0.5

        ref_out = reference_paged_attention(
            query, key_cache, value_cache, block_tables, context_lens,
            format_gen, precision_table, scale_table, block_offsets,
            num_kv_heads, block_size, softmax_scale,
        )

        op = {
            0: self.ops.psp_attention_basic,
            1: self.ops.psp_attention_enhanced,
            2: self.ops.psp_attention_btkv,
        }[format_gen]

        kernel_out = torch.empty_like(query)
        if format_gen == 0:
            op(query, key_cache, value_cache, block_tables, context_lens,
               precision_table, block_offsets, softmax_scale, num_kv_heads, head_dim, kernel_out)
        elif format_gen == 1:
            op(query, key_cache, value_cache, block_tables, context_lens,
               precision_table, block_offsets, scale_table, softmax_scale, num_kv_heads, head_dim, kernel_out)
        else:
            tma_descriptors = torch.zeros(num_blocks, 16, dtype=torch.int64, device="cuda")
            op(query, key_cache, value_cache, block_tables, context_lens,
               precision_table, block_offsets, scale_table, tma_descriptors,
               softmax_scale, num_kv_heads, head_dim, kernel_out)
        torch.cuda.synchronize()

        kernel_out_f = kernel_out.float()
        assert not torch.isnan(kernel_out_f).any(), "Kernel produced NaN output"
        assert not torch.isinf(kernel_out_f).any(), "Kernel produced Inf output"
        _atol = 1.0 if dtype in (torch.float8_e4m3fn, torch.float8_e5m2) else 1e-2
        torch.testing.assert_close(kernel_out_f, ref_out.float(), atol=_atol, rtol=_atol)

    def test_precision_table_selects_fp8_format(self):
        """e4m3 vs e5m2 must dequantize the same bytes differently."""
        torch.manual_seed(0)
        head_dim = 128
        num_heads = num_kv_heads = 4
        block_size = 16
        num_seqs = 2
        ctx_len = 64
        num_blocks = 8
        softmax_scale = 1.0 / (head_dim ** 0.5)

        context_lens = torch.tensor([ctx_len, ctx_len], dtype=torch.int32, device="cuda")
        block_tables = torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=torch.int32, device="cuda")

        query = torch.randn((num_seqs, num_heads, head_dim), dtype=torch.float16, device="cuda")
        prec_e4m3 = torch.ones(num_blocks, dtype=torch.uint8, device="cuda")
        key_cache, value_cache, block_offsets = _build_multiblock_cache(
            num_blocks, num_kv_heads, block_size, head_dim, 1, prec_e4m3, torch.float16, "cuda"
        )
        scale_table = torch.ones((num_blocks, num_kv_heads), dtype=torch.float32, device="cuda")
        prec_e5m2 = torch.full((num_blocks,), 2, dtype=torch.uint8, device="cuda")

        def run(prec):
            out = torch.empty_like(query)
            self.ops.psp_attention_enhanced(
                query, key_cache, value_cache, block_tables, context_lens,
                prec, block_offsets, scale_table, softmax_scale, num_kv_heads, head_dim, out,
            )
            return out

        out_e4m3 = run(prec_e4m3)
        out_e5m2 = run(prec_e5m2)
        torch.cuda.synchronize()

        assert not torch.allclose(out_e4m3, out_e5m2, atol=1e-3, rtol=1e-3)
