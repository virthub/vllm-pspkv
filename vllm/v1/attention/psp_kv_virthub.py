# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reconciliation between Virthub's storage taxonomy and the PSP-KV kernels.

Virthub's precision crate (`src/precision/src/policy.rs`) defines a *policy*
taxonomy (PrecisionLevel), while its store crate (`src/store/src/psp_kv.rs`)
already lowers that policy to a *base_precision* matching the PSP-KV kernel
`precision_table` encoding. This module:

  1. reconciles Virthub ``PrecisionLevel`` -> PSP-KV ``base_precision``;
  2. adapts Virthub's ``PspKvSidecarDescriptor`` -> ``precision_table`` +
     ``block_offsets`` (the two tensors the kernels consume);
  3. computes per-block byte offsets for the variable-stride layout.
"""

import dataclasses
import math
import struct
from enum import IntEnum


class VirthubPrecisionLevel(IntEnum):
    """Virthub's allocation-time precision tier (policy.rs)."""
    Fp16 = 0          # lossless FP16/BF16
    Fp8 = 1           # symmetric FP8 (E4M3/E5M2)
    Fp8Residual = 2   # FP8 base + residual correction page
    Pruned = 3        # non-critical heads pruned (head_mask)


class PspKvBasePrecision(IntEnum):
    """The PSP-KV kernel precision_table encoding (matches sidecar base_precision)."""
    Lossless = 0      # FP16/BF16
    Fp8E4M3 = 1
    Fp8E5M2 = 2
    Fp4 = 3


def virthub_precision_to_base(level: VirthubPrecisionLevel | int,
                              fp8_format: str = "e4m3") -> PspKvBasePrecision:
    """Map Virthub's PrecisionLevel to the PSP-KV base_precision.

    - Fp16 -> Lossless (exact).
    - Fp8 / Fp8Residual -> Fp8 (e4m3 or e5m2); the residual page is a
      Virthub-side detail the kernels do not consume yet.
    - Pruned -> Fp4 (max compression proxy; head pruning is not a numerical
      format, so it maps to the lowest precision tier).
    """
    if isinstance(level, VirthubPrecisionLevel):
        level = int(level)
    level = int(level)
    if level == VirthubPrecisionLevel.Fp16:
        return PspKvBasePrecision.Lossless
    if level == VirthubPrecisionLevel.Pruned:
        return PspKvBasePrecision.Fp4
    # Fp8 or Fp8Residual
    return PspKvBasePrecision.Fp8E5M2 if fp8_format == "e5m2" else PspKvBasePrecision.Fp8E4M3


@dataclasses.dataclass
class PspKvSidecarDescriptor:
    """Mirror of virthub's PspKvSidecarDescriptor (src/store/src/psp_kv.rs)."""
    data_page_base_ptr: int = 0
    residual_page_ptr: int = 0
    head_presence_mask: int = 0
    base_precision: int = 1          # 0=FP16/BF16, 1=E4M3, 2=E5M2, 3=FP4
    compression_level: int = 1       # 0=lossless, 1=FP8, 2=FP8+residual, 3=pruned
    head_group_size: int = 1
    reserved_flags: int = 0
    per_head_scale_e8m0: list[int] = dataclasses.field(default_factory=lambda: [0] * 8)


def bytes_per_elem(base_precision: int, scalar_size: int = 2) -> float:
    return {0: float(scalar_size), 1: 1.0, 2: 1.0, 3: 0.5}[base_precision]


def block_bytes(base_precision: int, num_kv_heads: int, block_size: int,
                head_size: int, scalar_size: int = 2) -> int:
    """Byte size of one block (all heads) at the given precision."""
    per_head = int(block_size * head_size * bytes_per_elem(base_precision, scalar_size))
    return per_head * num_kv_heads


def compute_block_offsets(precisions: list[int], num_kv_heads: int,
                          block_size: int, head_size: int,
                          scalar_size: int = 2) -> list[int]:
    """Cumulative byte offsets for the variable-stride layout."""
    offsets = []
    total = 0
    for p in precisions:
        offsets.append(total)
        total += block_bytes(p, num_kv_heads, block_size, head_size, scalar_size)
    return offsets


def sidecar_to_tables(sidecars: list[PspKvSidecarDescriptor],
                      buffer_base_ptr: int = 0) -> tuple[list[int], list[int]]:
    """Adapt Virthub sidecars -> (precision_table, block_offsets).

    precision_table comes from ``base_precision`` (already the kernel encoding);
    block_offsets comes from ``data_page_base_ptr`` relative to the buffer base.
    """
    precision_table = [s.base_precision for s in sidecars]
    block_offsets = [s.data_page_base_ptr - buffer_base_ptr for s in sidecars]
    return precision_table, block_offsets


# bincode 1.3 serializes the Rust #[repr(C, align(64))] struct as the packed,
# little-endian sequence of its fields (no padding): 64 bytes total.
_SIDECAR_PACKED_SIZE = 64


def fp32_to_e8m0(scale: float) -> int:
    """Convert an fp32 per-head scale to E8M0 (value = 2^(bits - 127)).

    A non-positive scale means "unset", which maps to 127 (= 1.0), the
    identity scale. The kernel scale_table is zero-initialized, so this keeps
    the sidecar's default consistent with the lossless/unscaled behavior.
    """
    if scale <= 0.0:
        return 127
    e = int(round(math.log2(scale))) + 127
    return max(0, min(255, e))


def serialize_sidecar(base_precision: int,
                      per_head_scales: list[float] | None = None,
                      data_page_base_ptr: int = 0) -> bytes:
    """Serialize ``PspKvSidecarDescriptor`` to the virthub wire format.

    Matches ``bincode::serialize`` of the Rust struct (packed, little-endian):

        data_page_base_ptr  u64
        residual_page_ptr   u64
        head_presence_mask  u64
        base_precision      u8
        compression_level   u8
        head_group_size     u8
        reserved_flags      u8
        per_head_scale_e8m0 [u8; 8]
        reserved_padding    [u8; 28]
    """
    if per_head_scales is None:
        per_head_scales = []
    scales = [fp32_to_e8m0(s) for s in per_head_scales][:8]
    scales += [0] * (8 - len(scales))
    # base_precision -> compression_level (the Virthub policy tier).
    compression_level = {0: 0, 1: 1, 2: 1, 3: 3}[int(base_precision) & 3]
    header = struct.pack(
        "<QQQBBBB",
        data_page_base_ptr & 0xFFFFFFFFFFFFFFFF,
        0,                      # residual_page_ptr (unused)
        0xFFFFFFFFFFFFFFFF,     # head_presence_mask (all heads present)
        int(base_precision) & 0xFF,
        compression_level,
        1,                      # head_group_size
        0,                      # reserved_flags
    )
    out = header + bytes(scales) + bytes(28)
    assert len(out) == _SIDECAR_PACKED_SIZE
    return out
