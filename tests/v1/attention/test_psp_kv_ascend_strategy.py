# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""The develop matrix must allocate real cache bytes for its four precisions."""

import pytest
import torch

from vllm.v1.attention.backends.psp_kv_ascend_strategy import (
    derive_develop_codes,
    read_strategy,
)
from vllm.v1.kv_cache_interface import PSPKVAttentionSpec


@pytest.fixture(autouse=True)
def clean_strategy(monkeypatch):
    for name in (
        "PSPKV_AVG_BYTES",
        "PSPKV_AVG_K_BYTES",
        "PSPKV_AVG_V_BYTES",
        "PSPKV_ADAPTIVE_PRECISION",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("PSPKV_NVIDIA_STRATEGIES", "1")
    read_strategy.cache_clear()
    yield
    read_strategy.cache_clear()


@pytest.mark.parametrize("generation", ("basic", "enhanced", "btkv"))
@pytest.mark.parametrize(
    "precision,element_bytes",
    (
        ("lossless", 2),
        ("e4m3", 1),
        ("e5m2", 1),
        ("fp4", 0.5),
    ),
)
def test_real_cache_budget(monkeypatch, generation, precision, element_bytes):
    monkeypatch.setenv("PSPKV_FORMAT", generation)
    monkeypatch.setenv("PSPKV_PRECISION", precision)
    strategy = read_strategy()
    assert strategy.element_bytes == element_bytes
    spec = PSPKVAttentionSpec(
        block_size=128, num_kv_heads=4, head_size=128, dtype=torch.uint8
    )
    assert spec.k_page_size_bytes == 128 * 4 * 128 * element_bytes
    assert spec.v_page_size_bytes == spec.k_page_size_bytes
    assert spec.page_size_bytes == 2 * spec.k_page_size_bytes


def test_rejects_underallocated_lossless(monkeypatch):
    monkeypatch.setenv("PSPKV_FORMAT", "enhanced")
    monkeypatch.setenv("PSPKV_PRECISION", "lossless")
    monkeypatch.setenv("PSPKV_AVG_BYTES", "1")
    with pytest.raises(ValueError, match="exact storage budget"):
        read_strategy()


def test_adaptive_matches_develop_amplitude_policy(monkeypatch):
    monkeypatch.setenv("PSPKV_PRECISION", "e4m3")
    monkeypatch.setenv("PSPKV_ADAPTIVE_PRECISION", "1")
    assert read_strategy().adaptive
    assert read_strategy().element_bytes == 1
    maxima = torch.tensor([[0.0, 0.0], [0.1, 0.2], [1.0, 0.0], [200.0, 100.0]])
    assert derive_develop_codes(maxima).tolist() == [1, 3, 1, 1]
