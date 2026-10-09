# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Executable spec for the PSP-KV FP8 dequant/quantize reference math.

These are CPU-only tests of the *reference model* (the exact math the CUDA
kernels implement, per docs/PSP_KV_FORMAT_CONTRACT.md). They are the golden
reference that the GPU correctness tests compare the kernels against. They do
not run the CUDA kernels themselves.
"""

import torch

from tests.v1.attention.psp_kv_reference import (
    dequantize_fp4_e2m1,
    dequantize_fp8,
    quantize_fp4_e2m1,
    quantize_fp8,
)


def test_dequantize_fp8_e4m3_known_values():
    """e4m3: byte layout is 1s-4e-3m with bias 7."""
    # 0b0_0111_000 = 1.0 * 2^(7-7) = 1.0
    assert dequantize_fp8(torch.tensor([0b00111000], dtype=torch.uint8), 1)[0].item() == 1.0
    # 0b1_0111_000 = -1.0
    assert dequantize_fp8(torch.tensor([0b10111000], dtype=torch.uint8), 1)[0].item() == -1.0
    # 0b0_1000_000 = 1.0 * 2^(8-7) = 2.0
    assert dequantize_fp8(torch.tensor([0b01000000], dtype=torch.uint8), 1)[0].item() == 2.0
    # subnormal: exp=0, mant=0b010 -> 0.010b * 2^-6 = 2 * 2^-9 = 0.00390625
    assert abs(dequantize_fp8(torch.tensor([0b00000010], dtype=torch.uint8), 1)[0].item() - 0.00390625) < 1e-6


def test_dequantize_fp8_matches_torch():
    """Our bit-exact dequant must match torch's native FP8 dequant."""
    bytes_ = torch.arange(256, dtype=torch.uint8)
    ours_e4m3 = dequantize_fp8(bytes_, 1)
    torch_e4m3 = bytes_.view(torch.float8_e4m3fn).to(torch.float32)
    torch.testing.assert_close(ours_e4m3, torch_e4m3, atol=0, rtol=0, equal_nan=True)

    ours_e5m2 = dequantize_fp8(bytes_, 2)
    torch_e5m2 = bytes_.view(torch.float8_e5m2).to(torch.float32)
    torch.testing.assert_close(ours_e5m2, torch_e5m2, atol=0, rtol=0, equal_nan=True)


def test_quantize_roundtrip_exact_integers():
    """Integers 0..8 are exactly representable in e4m3 and e5m2."""
    x = torch.arange(0, 9, dtype=torch.float32)
    for fmt in (1, 2):
        q = quantize_fp8(x, fmt)
        back = dequantize_fp8(q, fmt)
        torch.testing.assert_close(back, x, atol=0, rtol=0)


def test_quantize_produces_valid_fp8_bytes():
    """Quantized bytes dequantize back close to the original (within fp8 precision)."""
    x = torch.randn(64) * 4.0
    for fmt in (1, 2):
        q = quantize_fp8(x, fmt)
        assert q.dtype == torch.uint8
        back = dequantize_fp8(q, fmt)
        # e4m3 has ~2 decimal digits; e5m2 ~1.3. Relax tolerance accordingly.
        torch.testing.assert_close(back, x, atol=0.5, rtol=0.25)


def test_dequantize_fp4_e2m1_known_values():
    """e2m1: 1 sign, 2 exp, 1 mantissa."""
    assert dequantize_fp4_e2m1(torch.tensor([0b0010], dtype=torch.uint8))[0].item() == 1.0  # 1.0
    assert dequantize_fp4_e2m1(torch.tensor([0b0100], dtype=torch.uint8))[0].item() == 2.0  # 2.0
    assert dequantize_fp4_e2m1(torch.tensor([0b0101], dtype=torch.uint8))[0].item() == 3.0  # 3.0
    assert dequantize_fp4_e2m1(torch.tensor([0b1010], dtype=torch.uint8))[0].item() == -1.0  # -1.0


def test_quantize_fp4_e2m1_encodes_integers():
    """fp4 e2m1 encoding of integers 0..3 matches the CUDA helper."""
    q = quantize_fp4_e2m1(torch.tensor([0.0, 1.0, 2.0, 3.0]))
    assert q.tolist() == [0, 0b0010, 0b0100, 0b0101]


def test_dequantize_kv_interpolated_output():
    """dequantize_kv must produce uniform target-precision KV from the packed cache."""
    import torch
    from tests.v1.attention.psp_kv_reference import (
        dequantize_kv, pack_kv_cache, _dequant_elements, _bytes_per_head, _scalar_size,
    )
    torch.manual_seed(0)
    num_blocks, nkv, B, D = 2, 2, 16, 128
    # two blocks: fp8 e4m3 (fmt 1) and fp4 (fmt 3)
    precision_table = torch.tensor([1, 3], dtype=torch.uint8)
    k = torch.randn(num_blocks, nkv, B, D) * 2.0
    v = torch.randn(num_blocks, nkv, B, D) * 2.0
    k_buf, v_buf, offs = pack_kv_cache(k, v, 0, precision_table, torch.float16)
    scale_table = torch.rand(num_blocks, nkv) * 1.5 + 0.5
    k_out, v_out = dequantize_kv(k_buf, v_buf, offs, precision_table, scale_table,
                                 nkv, B, D, torch.float16)
    assert k_out.shape == (num_blocks, nkv, B, D)
    assert k_out.dtype == torch.float16
    # block 0 (fp8) equals fp8 dequant * scale
    from tests.v1.attention.psp_kv_reference import dequantize_fp8, quantize_fp8
    expected_k0 = dequantize_fp8(quantize_fp8(k[0], 1), 1) * scale_table[0][:, None, None]
    torch.testing.assert_close(k_out[0].float(), expected_k0, atol=1e-2, rtol=1e-2)
    # block 1 (fp4) equals fp4 dequant * scale (fp4 also applies scale_table)
    from tests.v1.attention.psp_kv_reference import dequantize_fp4_e2m1, quantize_fp4_e2m1
    expected_k1 = dequantize_fp4_e2m1(quantize_fp4_e2m1(k[1])) * scale_table[1][:, None, None]
    torch.testing.assert_close(k_out[1].float(), expected_k1, atol=1e-2, rtol=1e-2)
