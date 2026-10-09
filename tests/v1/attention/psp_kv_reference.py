# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Golden reference for the PSP-KV reference kernels, in pure PyTorch.

The KV cache is a *byte buffer* (variable-stride storage): each block's byte
offset comes from ``block_offsets``, and the per-block ``precision_table``
selects the inner precision:

  - 0 = lossless (fp16/bf16/fp8 target, 2 or 1 bytes/elem)
  - 1 = fp8 e4m3 (1 byte/elem)
  - 2 = fp8 e5m2 (1 byte/elem)
  - 3 = fp4 e2m1 (0.5 byte/elem, 2 elems packed per byte)

These functions reimplement the exact kernel math (docs/PSP_KV_FORMAT_CONTRACT.md).
"""

import torch


def dequantize_fp8(q: torch.Tensor, fmt: int) -> torch.Tensor:
    """Bit-exact reimplementation of the CUDA dequantize_fp8_* helpers (uint8 -> f32)."""
    q = q.to(torch.int32)
    if fmt == 2:
        sign = (q >> 7) & 1
        exp = (q >> 2) & 0x1F
        mant = q & 0x3
        normal = (1.0 + mant.to(torch.float32) / 4.0) * torch.pow(
            2.0, exp.to(torch.float32) - 15.0
        )
        subnormal = mant.to(torch.float32) * (2.0 ** -16.0)
        special = torch.where(mant == 0, float("inf"), float("nan"))
    else:
        sign = (q >> 7) & 1
        exp = (q >> 3) & 0xF
        mant = q & 0x7
        normal = (1.0 + mant.to(torch.float32) / 8.0) * torch.pow(
            2.0, exp.to(torch.float32) - 7.0
        )
        subnormal = mant.to(torch.float32) * (2.0 ** -9.0)
        special = torch.where(mant == 7, float("nan"), normal)
    val = torch.where(exp == 0, subnormal, normal)
    max_exp = 31 if fmt == 2 else 15
    val = torch.where(exp == max_exp, special, val)
    return torch.where(sign == 1, -val, val)


def quantize_fp8(x: torch.Tensor, fmt: int) -> torch.Tensor:
    """Quantize via torch's native FP8 types (matches CUDA for exact integers)."""
    if fmt == 2:
        return x.to(torch.float8_e5m2).view(torch.uint8)
    return x.to(torch.float8_e4m3fn).view(torch.uint8)


def dequantize_fp4_e2m1(q: torch.Tensor) -> torch.Tensor:
    """Bit-exact e2m1 dequant (nibble 0-15 -> float), matching the CUDA helper."""
    table = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, float("inf"), float("inf"),
         -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, float("-inf"), float("-inf")],
        dtype=torch.float32, device=q.device,
    )
    return table[q.to(torch.int32)]


def quantize_fp4_e2m1(x: torch.Tensor) -> torch.Tensor:
    """Nearest-neighbor e2m1 quantize (exact for integers 0..3)."""
    x = x.to(torch.float32)
    mag = x.abs()
    code = torch.where(mag < 0.25, 0,
            torch.where(mag < 0.75, 1,
            torch.where(mag < 1.25, 2,
            torch.where(mag < 1.75, 3,
            torch.where(mag < 2.5, 4, 5)))))
    code = code.to(torch.uint8)
    sign_bit = (x < 0).to(torch.uint8) * 8
    return code | sign_bit


def _scalar_size(dtype: torch.dtype) -> int:
    return 2 if dtype in (torch.float16, torch.bfloat16) else 1


def _bytes_per_head(fmt: int, block_size: int, head_size: int, scalar_size: int) -> int:
    if fmt == 0:
        return block_size * head_size * scalar_size
    if fmt == 3:
        return block_size * head_size // 2
    return block_size * head_size


def _pack_elements(x: torch.Tensor, fmt: int, dtype: torch.dtype, scale: float) -> torch.Tensor:
    """Pack a [..., block_size, head_size] float tensor into a uint8 byte buffer."""
    x = x.float()
    if fmt == 0:
        return x.to(dtype).reshape(-1).view(torch.uint8)
    if fmt == 3:
        q = quantize_fp4_e2m1(x / scale).reshape(-1)  # nibbles
        even = q[0::2]
        odd = q[1::2]
        return (even | (odd << 4)).to(torch.uint8)
    return quantize_fp8(x / scale, fmt).reshape(-1).to(torch.uint8)


def _dequant_elements(buf: torch.Tensor, fmt: int, dtype: torch.dtype,
                      shape: tuple, scale: float) -> torch.Tensor:
    """Dequantize a byte buffer into a float tensor of `shape`."""
    if fmt == 0:
        return buf.view(dtype).reshape(shape).float()
    if fmt == 3:
        low = (buf & 0x0F).to(torch.uint8)
        high = ((buf >> 4) & 0x0F).to(torch.uint8)
        nibbles = torch.stack([low, high], dim=-1).reshape(-1)
        return dequantize_fp4_e2m1(nibbles).reshape(shape) * scale
    return dequantize_fp8(buf, fmt).reshape(shape) * scale


def pack_kv_cache(k_full: torch.Tensor, v_full: torch.Tensor, format_gen: int,
                  precision_table: torch.Tensor, dtype: torch.dtype,
                  scale: float = 1.0):
    """Pack multi-block KV into a byte buffer + block_offsets.

    k_full/v_full: [num_blocks, num_kv_heads, block_size, head_size] (token-major).
    precision_table: [num_blocks] uint8.
    Returns (k_buf, v_buf, block_offsets).
    """
    num_blocks = k_full.shape[0]
    num_kv_heads = k_full.shape[1]
    block_size = k_full.shape[2]
    head_size = k_full.shape[3]
    scalar_size = _scalar_size(dtype)

    offsets = []
    block_bytes = []
    total = 0
    for b in range(num_blocks):
        fmt = int(precision_table[b].item())
        bb = _bytes_per_head(fmt, block_size, head_size, scalar_size) * num_kv_heads
        offsets.append(total)
        block_bytes.append(bb)
        total += bb

    k_buf = torch.zeros(total, dtype=torch.uint8, device=k_full.device)
    v_buf = torch.zeros(total, dtype=torch.uint8, device=k_full.device)
    for b in range(num_blocks):
        fmt = int(precision_table[b].item())
        k_bytes = _pack_elements(k_full[b], fmt, dtype, scale)
        if format_gen == 0:
            v_bytes = _pack_elements(v_full[b], fmt, dtype, scale)
        else:
            # enhanced/btkv V is stored transposed [head_size, block_size]
            # (i.e. [num_kv_heads, head_size, block_size] = V[h, d, t]).
            v_bytes = _pack_elements(v_full[b].permute(0, 2, 1), fmt, dtype, scale)
        off = offsets[b]
        k_buf[off:off + block_bytes[b]] = k_bytes
        v_buf[off:off + block_bytes[b]] = v_bytes

    return k_buf, v_buf, torch.tensor(offsets, dtype=torch.int32, device=k_full.device)


def dequantize_kv(k_cache: torch.Tensor, v_cache: torch.Tensor,
                  block_offsets: torch.Tensor, precision_table: torch.Tensor,
                  scale_table: torch.Tensor, num_kv_heads: int,
                  block_size: int, head_size: int, dtype: torch.dtype):
    """Reference for the staging path: compressed byte buffer -> uniform target-precision KV.

    Mirrors the kernel's launch_dequant_kv (scale applied only for fp8).
    """
    num_blocks = len(precision_table)
    k_out = torch.zeros(num_blocks, num_kv_heads, block_size, head_size,
                        dtype=dtype, device=k_cache.device)
    v_out = torch.zeros_like(k_out)
    scalar_size = _scalar_size(dtype)
    for b in range(num_blocks):
        fmt = int(precision_table[b].item())
        per_head = _bytes_per_head(fmt, block_size, head_size, scalar_size)
        bb = per_head * num_kv_heads
        off = int(block_offsets[b].item())
        scale = scale_table[b].float()[:, None, None] if fmt != 0 else 1.0
        k_out[b] = _dequant_elements(k_cache[off:off + bb], fmt, dtype,
                                     (num_kv_heads, block_size, head_size), scale)
        v_out[b] = _dequant_elements(v_cache[off:off + bb], fmt, dtype,
                                     (num_kv_heads, block_size, head_size), scale)
    return k_out, v_out


def reference_paged_attention(
    query: torch.Tensor,          # [num_tokens, num_heads, head_dim] fp16/bf16/fp8
    key_cache: torch.Tensor,      # uint8 byte buffer
    value_cache: torch.Tensor,    # uint8 byte buffer
    block_tables: torch.Tensor,   # int32 [num_seqs, max_blocks_per_seq]
    context_lens: torch.Tensor,   # int32 [num_seqs]
    format_gen: int,              # 0=basic, 1=enhanced, 2=btkv
    precision_table: torch.Tensor,  # uint8 [num_blocks]
    scale_table: torch.Tensor | None,  # fp32 [num_blocks, num_kv_heads]
    block_offsets: torch.Tensor,  # int32 [num_blocks]
    num_kv_heads: int,
    block_size: int,
    softmax_scale: float,
) -> torch.Tensor:
    """Paged attention reference matching the PSP-KV reference kernels."""
    num_heads = query.shape[1]
    head_dim = query.shape[2]
    device = query.device

    outputs = []
    for seq in range(query.shape[0]):
        ctx = int(context_lens[seq].item())
        if ctx == 0:
            outputs.append(torch.zeros(num_heads, head_dim, device=device))
            continue

        num_blocks_needed = (ctx + block_size - 1) // block_size
        block_ids = [int(block_tables[seq, i].item()) for i in range(num_blocks_needed)]

        k_blocks: list[torch.Tensor] = []
        v_blocks: list[torch.Tensor] = []
        for bid in block_ids:
            fmt = int(precision_table[bid].item())
            scalar_size = _scalar_size(query.dtype)
            per_head = _bytes_per_head(fmt, block_size, head_dim, scalar_size)
            bb = per_head * num_kv_heads
            off = int(block_offsets[bid].item())

            k = _dequant_elements(key_cache[off:off + bb], fmt, query.dtype,
                                  (num_kv_heads, block_size, head_dim), 1.0)
            if format_gen == 0:
                v = _dequant_elements(value_cache[off:off + bb], fmt, query.dtype,
                                      (num_kv_heads, block_size, head_dim), 1.0)
            else:
                v = _dequant_elements(value_cache[off:off + bb], fmt, query.dtype,
                                      (num_kv_heads, head_dim, block_size), 1.0).transpose(1, 2)
            if format_gen != 0:
                s = scale_table[bid].float()[:, None, None]
                k = k * s
                v = v * s

            k_blocks.append(k)
            v_blocks.append(v)

        k_seq = torch.cat(k_blocks, dim=1)[:, :ctx, :]
        v_seq = torch.cat(v_blocks, dim=1)[:, :ctx, :]

        if num_heads > num_kv_heads:
            g = num_heads // num_kv_heads
            k_seq = k_seq.repeat_interleave(g, dim=0)
            v_seq = v_seq.repeat_interleave(g, dim=0)

        q = query[seq].float()
        scores = torch.einsum("hd,htd->ht", q, k_seq) * softmax_scale
        weights = torch.softmax(scores, dim=-1)
        out = torch.einsum("ht,htd->hd", weights, v_seq)
        outputs.append(out)

    return torch.stack(outputs).to(query.dtype)
