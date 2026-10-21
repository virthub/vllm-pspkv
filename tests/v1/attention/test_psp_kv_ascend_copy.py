# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Checks that the live FP8 backend copies formats with physical cache pages."""

import pytest
import torch

pytest.importorskip("vllm_ascend.attention.attention_v1")

from vllm.v1.attention.backends.psp_kv_ascend import (  # noqa: E402
    _REFERENCE_FP4_SCALES,
    _REFERENCE_PRECISION_TABLES,
    PSPKVAscendBackend,
)
from vllm.v1.kv_cache_interface import (  # noqa: E402
    FullAttentionSpec,
    PSPKVUniformFP4AttentionSpec,
)

pytestmark = pytest.mark.skip_global_cleanup


def test_uniform_fp4_page_budget_and_backend_shape(monkeypatch):
    fp8_spec = FullAttentionSpec(
        block_size=128,
        num_kv_heads=2,
        head_size=128,
        dtype=torch.uint8,
    )
    fp4_spec = PSPKVUniformFP4AttentionSpec(
        block_size=128,
        num_kv_heads=2,
        head_size=128,
        dtype=torch.uint8,
    )
    assert fp4_spec.page_size_bytes * 2 == fp8_spec.page_size_bytes
    assert type(fp4_spec.merge([fp4_spec, fp4_spec])) is PSPKVUniformFP4AttentionSpec
    monkeypatch.setenv("PSPKV_ASCEND_UNIFORM_FP4", "1")
    assert PSPKVAscendBackend.get_kv_cache_shape(3, 128, 2, 128) == (
        2,
        3,
        128,
        2,
        64,
    )


def test_copy_blocks_moves_fp4_scales_with_packed_bytes(monkeypatch):
    monkeypatch.setenv("PSPKV_ASCEND_UNIFORM_FP4", "1")
    cache = torch.arange(2 * 3 * 2 * 1 * 2, dtype=torch.uint8).reshape(2, 3, 2, 1, 2)
    before = cache.clone()
    scales = torch.tensor([[1.0], [2.0], [4.0]])
    pointer = cache[0].data_ptr()
    _REFERENCE_FP4_SCALES[pointer] = scales
    try:
        PSPKVAscendBackend.copy_blocks([cache], torch.tensor([[0, 1], [1, 0]]))
        torch.testing.assert_close(cache[:, 0], before[:, 1])
        torch.testing.assert_close(cache[:, 1], before[:, 0])
        assert scales[:, 0].tolist() == [2.0, 1.0, 4.0]
    finally:
        _REFERENCE_FP4_SCALES.pop(pointer, None)


def test_copy_blocks_rejects_missing_fp4_scales_before_writing(monkeypatch):
    monkeypatch.setenv("PSPKV_ASCEND_UNIFORM_FP4", "1")
    cache = torch.arange(16, dtype=torch.uint8).reshape(2, 2, 1, 1, 4)
    before = cache.clone()
    with pytest.raises(RuntimeError, match="without its scale table"):
        PSPKVAscendBackend.copy_blocks([cache], torch.tensor([[0, 1]]))
    torch.testing.assert_close(cache, before)


def test_copy_blocks_moves_fp8_format_table_with_kv_bytes():
    cache = torch.zeros((2, 3, 2, 1, 4), dtype=torch.uint8)
    cache[0, 0].fill_(0x38)
    cache[1, 0].fill_(0x40)
    cache[0, 1].fill_(0x3C)
    cache[1, 1].fill_(0x44)
    before = cache.clone()
    table = torch.tensor([1, 2, 1], dtype=torch.uint8)
    pointer = cache[0].data_ptr()
    _REFERENCE_PRECISION_TABLES[pointer] = table
    try:
        PSPKVAscendBackend.copy_blocks([cache], torch.tensor([[0, 1], [1, 0]]))
        torch.testing.assert_close(cache[:, 0], before[:, 1])
        torch.testing.assert_close(cache[:, 1], before[:, 0])
        assert table.tolist() == [2, 1, 1]
    finally:
        _REFERENCE_PRECISION_TABLES.pop(pointer, None)


def test_copy_blocks_rejects_missing_adaptive_table_before_writing(monkeypatch):
    monkeypatch.setenv("PSPKV_ASCEND_REFERENCE_FP8", "adaptive")
    cache = torch.arange(16, dtype=torch.uint8).reshape(2, 2, 1, 1, 4)
    before = cache.clone()
    with pytest.raises(RuntimeError, match="without its precision table"):
        PSPKVAscendBackend.copy_blocks([cache], torch.tensor([[0, 1]]))
    torch.testing.assert_close(cache, before)
