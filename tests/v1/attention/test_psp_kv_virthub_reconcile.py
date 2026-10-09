# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the Virthub <-> PSP-KV precision reconciliation + sidecar adapter."""

from vllm.v1.attention.psp_kv_virthub import (
    PspKvBasePrecision,
    PspKvSidecarDescriptor,
    VirthubPrecisionLevel,
    block_bytes,
    compute_block_offsets,
    sidecar_to_tables,
    virthub_precision_to_base,
)


def test_precision_level_mapping():
    assert virthub_precision_to_base(VirthubPrecisionLevel.Fp16) == PspKvBasePrecision.Lossless
    assert virthub_precision_to_base(VirthubPrecisionLevel.Fp8) == PspKvBasePrecision.Fp8E4M3
    assert virthub_precision_to_base(VirthubPrecisionLevel.Fp8Residual) == PspKvBasePrecision.Fp8E4M3
    assert virthub_precision_to_base(VirthubPrecisionLevel.Pruned) == PspKvBasePrecision.Fp4
    # e5m2 selection
    assert virthub_precision_to_base(VirthubPrecisionLevel.Fp8, "e5m2") == PspKvBasePrecision.Fp8E5M2
    # raw int works too
    assert virthub_precision_to_base(1) == PspKvBasePrecision.Fp8E4M3


def test_block_bytes_per_precision():
    # block_size=16, head_size=128, 2 KV heads, scalar_size=2 (fp16/bf16)
    assert block_bytes(0, 2, 16, 128, 2) == 16 * 128 * 2 * 2   # lossless: 2 bytes
    assert block_bytes(1, 2, 16, 128, 2) == 16 * 128 * 1 * 2   # fp8: 1 byte
    assert block_bytes(3, 2, 16, 128, 2) == 16 * 128 // 2 * 2   # fp4: 0.5 byte


def test_compute_block_offsets_cumulative():
    # mixed precision: [lossless, fp8, fp4]
    precisions = [0, 1, 3]
    offsets = compute_block_offsets(precisions, num_kv_heads=2, block_size=16, head_size=128)
    assert offsets == [
        0,
        block_bytes(0, 2, 16, 128),
        block_bytes(0, 2, 16, 128) + block_bytes(1, 2, 16, 128),
    ]


def test_sidecar_to_tables():
    base = 0x1000
    sidecars = [
        PspKvSidecarDescriptor(data_page_base_ptr=base + 0, base_precision=0),
        PspKvSidecarDescriptor(data_page_base_ptr=base + 8192, base_precision=1),
        PspKvSidecarDescriptor(data_page_base_ptr=base + 12288, base_precision=3),
    ]
    precision_table, block_offsets = sidecar_to_tables(sidecars, buffer_base_ptr=base)
    assert precision_table == [0, 1, 3]
    assert block_offsets == [0, 8192, 12288]
