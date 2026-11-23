# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Correctness tests for the PSP-KV compressed-cache decode kernels.

The cache stores fp8/fp4 bytes plus per-scale-group floats, so its kernels can
fail in ways that model-level numerics hide: a transposed page layout, a scale
applied to the wrong group, or a V row read from the wrong head. Each test
below drives one invariant with an input whose correct output is arithmetic,
not a reference implementation:

* V == 1 everywhere -> every output element is 1 (softmax rows sum to one),
* V == 1 on a single dim -> only that output dim is non-zero (V's element
  order), and
* V == (kv_head + 1) / 8 -> each query head reads its own KV head's V row.
"""

import pytest
import torch
import vllm._C  # noqa: F401  (registers torch.ops.psp_kv)

HEAD_DIM = 128
BLOCK = 128
GROUP = 16
NUM_GROUPS = HEAD_DIM // GROUP
GQA = 4
FP8_MAX = 448.0

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="PSP-KV kernels are CUDA-only"
)


def _build_and_run(ctx: int, num_kv_heads: int, value: torch.Tensor) -> torch.Tensor:
    """Run decode attention over one block whose K is zero and whose V is given.

    Zero K makes every score equal, so the output is exactly the mean of the V
    rows regardless of the softmax scale. Both K and V are stored as fp8 with
    unit scales so the only thing under test is the kernel's indexing.
    """
    device = value.device
    num_blocks = 2
    k_block_bytes = BLOCK * HEAD_DIM
    v_block_bytes = BLOCK * HEAD_DIM
    page = num_kv_heads * (k_block_bytes + v_block_bytes)
    cache = torch.zeros(num_blocks * page, dtype=torch.uint8, device=device)

    k_offsets = torch.arange(num_blocks, dtype=torch.int32, device=device) * page
    v_offsets = k_offsets + num_kv_heads * k_block_bytes
    block_offsets = torch.cat([k_offsets, v_offsets]).contiguous()

    vq = value.to(torch.float8_e4m3fn).view(torch.uint8)
    # token-major page: row (t * num_kv_heads + h) holds token t, head h
    row_idx = torch.arange(ctx, device=device)[:, None] * num_kv_heads
    row_idx = row_idx + torch.arange(num_kv_heads, device=device)[None, :]
    rows = v_offsets[0] + row_idx * HEAD_DIM  # [ctx, num_kv_heads]
    for h in range(num_kv_heads):
        for t in range(ctx):
            start = int(rows[t, h])
            cache[start : start + HEAD_DIM] = vq[t, h]

    # float16: PSP-KV rejects float32 queries (its tf32 kernel overruns smem)
    query = torch.zeros(
        1, num_kv_heads * GQA, HEAD_DIM, device=device, dtype=torch.float16
    )
    block_table = torch.zeros((1, num_blocks), dtype=torch.int32, device=device)
    context_lens = torch.tensor([ctx], dtype=torch.int32, device=device)
    prec = torch.full((num_blocks,), 1 | (1 << 2), dtype=torch.uint8, device=device)
    scales = torch.ones(2 * num_blocks, num_kv_heads, NUM_GROUPS, device=device)

    out = torch.zeros_like(query)
    torch.ops.psp_kv.psp_attention_enhanced(
        query,
        cache,
        cache,
        block_table,
        context_lens,
        prec,
        block_offsets,
        scales,
        1.0 / HEAD_DIM**0.5,
        num_kv_heads,
        HEAD_DIM,
        0,
        False,
        1,
        1,
        1,
        False,
        False,
        out,
    )
    torch.cuda.synchronize()
    return out[0].float()


@pytest.mark.parametrize("num_kv_heads", [1, 8])
def test_output_is_mean_of_v_rows(num_kv_heads: int) -> None:
    ctx = 40
    value = torch.ones(ctx, num_kv_heads, HEAD_DIM, device="cuda")
    out = _build_and_run(ctx, num_kv_heads, value)
    torch.testing.assert_close(out, torch.ones_like(out), atol=1e-3, rtol=0)


@pytest.mark.parametrize("hot_dim", [0, 5, 127])
def test_v_element_order_is_preserved(hot_dim: int) -> None:
    ctx, num_kv_heads = 40, 8
    value = torch.zeros(ctx, num_kv_heads, HEAD_DIM, device="cuda")
    value[:, :, hot_dim] = 1.0
    out = _build_and_run(ctx, num_kv_heads, value)
    hot = (out[0].abs() > 0.5).nonzero().flatten().tolist()
    assert hot == [hot_dim], f"dim {hot_dim} landed in {hot}"


def test_each_query_head_reads_its_own_kv_head() -> None:
    ctx, num_kv_heads = 40, 8
    value = torch.zeros(ctx, num_kv_heads, HEAD_DIM, device="cuda")
    for h in range(num_kv_heads):
        value[:, h, :] = (h + 1) / 8.0
    out = _build_and_run(ctx, num_kv_heads, value)
    got = [round(float(out[h * GQA, 0]), 4) for h in range(num_kv_heads)]
    expected = [round((h + 1) / 8.0, 4) for h in range(num_kv_heads)]
    assert got == expected, f"got {got} expected {expected}"
