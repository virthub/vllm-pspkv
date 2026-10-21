# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Multi-block, variable-context validation of the PSP-KV kernels vs. the FP8
reference (basic / enhanced / btkv × e4m3/e5m2 × MHA/GQA), plus a check that the
per-block precision_table format selector is honoured.

See docs/PSP_KV_FORMAT_CONTRACT.md.
"""

import math
import os

import pytest
import torch

from tests.v1.attention.psp_kv_reference import pack_kv_cache, reference_paged_attention

FMTS = [0, 1, 2, 3]  # 0 = lossless, 1 = e4m3, 2 = e5m2, 3 = fp4 (inner precision)
HEAD_SHAPES = [(8, 8), (8, 2)]
FORMATS = [0, 1, 2]  # basic, enhanced, btkv
DTYPES = [
    torch.float16,
    torch.bfloat16,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
]  # target precision
DEVICE = os.environ.get("PSPKV_TEST_DEVICE", "cuda")
pytestmark = pytest.mark.skip_global_cleanup if DEVICE == "npu" else []


def _get_ops():
    if DEVICE == "npu":
        from vllm.v1.attention.backends.psp_kv_ascend_native import (
            load_native_fp8_ops,
        )
        from vllm.v1.attention.backends.psp_kv_ascend_ops import PSPKVAscendPackedOps

        load_native_fp8_ops()
        return PSPKVAscendPackedOps()
    import vllm._C  # noqa: F401  (registers torch.ops.psp_kv)

    return getattr(torch.ops, "psp_kv", None)


def _build_multiblock_cache(
    num_blocks,
    num_kv_heads,
    block_size,
    head_dim,
    format_gen,
    precision_table,
    dtype,
    device,
):
    """Build multi-block byte buffer + block_offsets."""
    k = torch.randn(num_blocks, num_kv_heads, block_size, head_dim) * 2.0
    v_token_major = torch.randn(num_blocks, num_kv_heads, block_size, head_dim) * 2.0
    k_buf, v_buf, block_offsets = pack_kv_cache(
        k, v_token_major, format_gen, precision_table.cpu(), dtype
    )
    return k_buf.to(device), v_buf.to(device), block_offsets.to(device)


@pytest.mark.skipif(
    DEVICE == "cuda" and not torch.cuda.is_available(),
    reason="CUDA is required for PSP-KV CUDA kernel execution",
)
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
    def test_multiblock_matches_reference(
        self, format_gen, fmt, num_heads, num_kv_heads, dtype
    ):
        if (
            format_gen == 2
            and DEVICE == "cuda"
            and torch.cuda.get_device_capability()[0] < 9
        ):
            pytest.skip("BT-KV requires Hopper (SM90+) or Blackwell")

        torch.manual_seed(42)
        head_dim = 128
        block_size = 16
        num_seqs = 4
        max_blocks_per_seq = 8
        num_blocks = num_seqs * max_blocks_per_seq
        softmax_scale = 1.0 / (head_dim**0.5)

        context_lens = torch.tensor([17, 32, 63, 112], dtype=torch.int32, device=DEVICE)

        block_tables = torch.zeros(
            (num_seqs, max_blocks_per_seq), dtype=torch.int32, device=DEVICE
        )
        curr_b = 0
        for i in range(num_seqs):
            needed = math.ceil(int(context_lens[i].item()) / block_size)
            block_tables[i, :needed] = torch.arange(
                curr_b, curr_b + needed, dtype=torch.int32, device=DEVICE
            )
            curr_b += needed

        query = (
            torch.randn((num_seqs, num_heads, head_dim), dtype=torch.float16)
            .to(dtype)
            .to(DEVICE)
        )
        precision_table = torch.full(
            (num_blocks,), fmt, dtype=torch.uint8, device=DEVICE
        )
        key_cache, value_cache, block_offsets = _build_multiblock_cache(
            num_blocks,
            num_kv_heads,
            block_size,
            head_dim,
            format_gen,
            precision_table,
            dtype,
            DEVICE,
        )
        scale_table = torch.rand(num_blocks, num_kv_heads, device=DEVICE) * 1.5 + 0.5

        if DEVICE == "npu":
            # The original Python oracle uses byte views that torch_npu cannot
            # perform for every FP8 dtype. Keep the oracle on CPU while the
            # operator under test runs on NPU.
            ref_out = reference_paged_attention(
                query.cpu(),
                key_cache.cpu(),
                value_cache.cpu(),
                block_tables.cpu(),
                context_lens.cpu(),
                format_gen,
                precision_table.cpu(),
                scale_table.cpu(),
                block_offsets.cpu(),
                num_kv_heads,
                block_size,
                softmax_scale,
            )
        else:
            ref_out = reference_paged_attention(
                query,
                key_cache,
                value_cache,
                block_tables,
                context_lens,
                format_gen,
                precision_table,
                scale_table,
                block_offsets,
                num_kv_heads,
                block_size,
                softmax_scale,
            )

        op = {
            0: self.ops.psp_attention_basic,
            1: self.ops.psp_attention_enhanced,
            2: self.ops.psp_attention_btkv,
        }[format_gen]

        kernel_out = torch.empty_like(query)
        if format_gen == 0:
            op(
                query,
                key_cache,
                value_cache,
                block_tables,
                context_lens,
                precision_table,
                block_offsets,
                softmax_scale,
                num_kv_heads,
                head_dim,
                kernel_out,
            )
        elif format_gen == 1:
            op(
                query,
                key_cache,
                value_cache,
                block_tables,
                context_lens,
                precision_table,
                block_offsets,
                scale_table,
                softmax_scale,
                num_kv_heads,
                head_dim,
                kernel_out,
            )
        else:
            tma_descriptors = torch.zeros(
                num_blocks, 16, dtype=torch.int64, device=DEVICE
            )
            op(
                query,
                key_cache,
                value_cache,
                block_tables,
                context_lens,
                precision_table,
                block_offsets,
                scale_table,
                tma_descriptors,
                softmax_scale,
                num_kv_heads,
                head_dim,
                kernel_out,
            )
        if DEVICE == "npu":
            torch.npu.synchronize()
        else:
            torch.accelerator.synchronize()

        kernel_out_f = (
            kernel_out.cpu().float() if DEVICE == "npu" else kernel_out.float()
        )
        assert not torch.isnan(kernel_out_f).any(), "Kernel produced NaN output"
        assert not torch.isinf(kernel_out_f).any(), "Kernel produced Inf output"
        _atol = 1.0 if dtype in (torch.float8_e4m3fn, torch.float8_e5m2) else 1e-2
        torch.testing.assert_close(
            kernel_out_f.cpu(), ref_out.float(), atol=_atol, rtol=_atol
        )

    def test_precision_table_selects_fp8_format(self):
        """e4m3 vs e5m2 must dequantize the same bytes differently."""
        torch.manual_seed(0)
        head_dim = 128
        num_heads = num_kv_heads = 4
        block_size = 16
        num_seqs = 2
        ctx_len = 64
        num_blocks = 8
        softmax_scale = 1.0 / (head_dim**0.5)

        context_lens = torch.tensor(
            [ctx_len, ctx_len], dtype=torch.int32, device=DEVICE
        )
        block_tables = torch.tensor(
            [[0, 1, 2, 3], [4, 5, 6, 7]], dtype=torch.int32, device=DEVICE
        )

        query = torch.randn((num_seqs, num_heads, head_dim), dtype=torch.float16).to(
            DEVICE
        )
        prec_e4m3 = torch.ones(num_blocks, dtype=torch.uint8, device=DEVICE)
        key_cache, value_cache, block_offsets = _build_multiblock_cache(
            num_blocks,
            num_kv_heads,
            block_size,
            head_dim,
            1,
            prec_e4m3,
            torch.float16,
            DEVICE,
        )
        scale_table = torch.ones(
            (num_blocks, num_kv_heads), dtype=torch.float32, device=DEVICE
        )
        prec_e5m2 = torch.full((num_blocks,), 2, dtype=torch.uint8, device=DEVICE)

        def run(prec):
            out = torch.empty_like(query)
            self.ops.psp_attention_enhanced(
                query,
                key_cache,
                value_cache,
                block_tables,
                context_lens,
                prec,
                block_offsets,
                scale_table,
                softmax_scale,
                num_kv_heads,
                head_dim,
                out,
            )
            return out

        out_e4m3 = run(prec_e4m3)
        out_e5m2 = run(prec_e5m2)
        if DEVICE == "npu":
            torch.npu.synchronize()
        else:
            torch.accelerator.synchronize()

        assert not torch.allclose(out_e4m3, out_e5m2, atol=1e-3, rtol=1e-3)
