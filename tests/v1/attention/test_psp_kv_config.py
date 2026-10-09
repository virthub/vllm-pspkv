# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the PSP-KV config resolution (conf/pspkv.toml + env overrides).

These are CPU-only tests of `_read_psp_kv_format_generation` (launcher) and
`_read_psp_kv_precision` (e4m3/e5m2). They verify the precedence: env var ->
conf/pspkv.toml -> conf/virthub.toml (launcher only) -> default.
"""

import os

import pytest

from vllm.v1.worker.gpu_model_runner import (
    _read_psp_kv_format_generation,
    _read_psp_kv_precision,
    _read_psp_kv_storage_mode,
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in ("PSPKV_FORMAT", "VIRTHUB_FORMAT", "PSPKV_PRECISION",
              "PSPKV_STORAGE", "PSPKV_CONFIG", "VIRTHUB_CONFIG"):
        monkeypatch.delenv(k, raising=False)


def _write_toml(tmp_path, name, content):
    p = tmp_path / name
    p.write_text(content)
    return str(p)


def test_launcher_env_override(monkeypatch):
    monkeypatch.setenv("PSPKV_FORMAT", "2")
    assert _read_psp_kv_format_generation() == 2


def test_launcher_deprecated_alias(monkeypatch):
    monkeypatch.setenv("VIRTHUB_FORMAT", "0")
    assert _read_psp_kv_format_generation() == 0


def test_launcher_from_pspkv_toml(monkeypatch, tmp_path):
    p = _write_toml(tmp_path, "pspkv.toml", "[format]\ngeneration = 2\n")
    monkeypatch.setenv("PSPKV_CONFIG", p)
    assert _read_psp_kv_format_generation() == 2


def test_launcher_falls_back_to_virthub_toml(monkeypatch, tmp_path):
    # pspkv.toml missing -> fall back to virthub.toml's [precision].format_generation
    monkeypatch.setenv("PSPKV_CONFIG", str(tmp_path / "missing.toml"))
    p = _write_toml(tmp_path, "virthub.toml", "[precision]\nformat_generation = 2\n")
    monkeypatch.setenv("VIRTHUB_CONFIG", p)
    assert _read_psp_kv_format_generation() == 2


def test_precision_env_override(monkeypatch):
    monkeypatch.setenv("PSPKV_PRECISION", "e5m2")
    assert _read_psp_kv_precision() == 2


def test_precision_from_pspkv_toml(monkeypatch, tmp_path):
    p = _write_toml(tmp_path, "pspkv.toml", "[precision]\ninner_precision = \"e5m2\"\n")
    monkeypatch.setenv("PSPKV_CONFIG", p)
    assert _read_psp_kv_precision() == 2


def test_precision_default_is_e4m3(monkeypatch, tmp_path):
    monkeypatch.setenv("PSPKV_CONFIG", str(tmp_path / "missing.toml"))
    assert _read_psp_kv_precision() == 1


def test_precision_lossless_maps_to_0(monkeypatch):
    monkeypatch.setenv("PSPKV_PRECISION", "lossless")
    assert _read_psp_kv_precision() == 0


def test_precision_fp4_maps_to_3(monkeypatch):
    monkeypatch.setenv("PSPKV_PRECISION", "fp4")
    assert _read_psp_kv_precision() == 3


def test_precision_deprecated_fp8_format_alias(monkeypatch, tmp_path):
    p = _write_toml(tmp_path, "pspkv.toml", "[precision]\nfp8_format = \"e5m2\"\n")
    monkeypatch.setenv("PSPKV_CONFIG", p)
    assert _read_psp_kv_precision() == 2


def test_storage_mode_default_is_fixed(monkeypatch, tmp_path):
    monkeypatch.setenv("PSPKV_CONFIG", str(tmp_path / "missing.toml"))
    assert _read_psp_kv_storage_mode() == "fixed"


def test_storage_mode_from_pspkv_toml(monkeypatch, tmp_path):
    p = _write_toml(tmp_path, "pspkv.toml", "[storage]\nmode = \"variable\"\n")
    monkeypatch.setenv("PSPKV_CONFIG", p)
    assert _read_psp_kv_storage_mode() == "variable"


def test_storage_mode_env_override(monkeypatch):
    monkeypatch.setenv("PSPKV_STORAGE", "variable")
    assert _read_psp_kv_storage_mode() == "variable"


def test_storage_mode_invalid_falls_back_to_fixed(monkeypatch, tmp_path):
    p = _write_toml(tmp_path, "pspkv.toml", "[storage]\nmode = \"bogus\"\n")
    monkeypatch.setenv("PSPKV_CONFIG", p)
    assert _read_psp_kv_storage_mode() == "fixed"
