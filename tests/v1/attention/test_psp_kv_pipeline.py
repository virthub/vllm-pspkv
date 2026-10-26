# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from ascend.pipeline_runtime import (
    GRAPH_INPUTS,
    consume_deferred_rope,
    graph_signature,
    install_graph_bypass,
    pipeline_level,
    validate_graph_inputs,
)


def test_graph_reads_changed_content_and_rejects_replaced_buffers():
    source = SimpleNamespace(
        block_tables=torch.tensor([[3, 1]], dtype=torch.int32),
        slot_mapping=torch.tensor([0], dtype=torch.int64),
        psp_kv_device_seq_lens=torch.tensor([1], dtype=torch.int32),
    )
    owner = SimpleNamespace(attn_metadata={"layer": source})
    original = set(GRAPH_INPUTS)
    try:
        GRAPH_INPUTS.add(graph_signature(source))
        source.block_tables.copy_(torch.tensor([[1, 3]], dtype=torch.int32))
        source.psp_kv_device_seq_lens.fill_(129)
        source.slot_mapping.fill_(384)
        validate_graph_inputs(None, owner)
        source.psp_kv_device_seq_lens = source.psp_kv_device_seq_lens.clone()
        with pytest.raises(RuntimeError, match="address changed"):
            validate_graph_inputs(None, owner)
    finally:
        GRAPH_INPUTS.clear()
        GRAPH_INPUTS.update(original)


def test_pipeline_control_is_fixed_for_current_forward(monkeypatch, tmp_path):
    control = tmp_path / "mode.txt"
    control.write_text("0")
    monkeypatch.setenv("PSPKV_PIPELINE_CONTROL_FILE", str(control))
    owner = SimpleNamespace()
    assert pipeline_level(owner) == 0
    control.write_text("2")
    assert pipeline_level(owner) == 0
    assert pipeline_level(SimpleNamespace()) == 2


def test_deferred_rope_is_consumed_once_and_checks_pairing():
    query, key = torch.empty(1, 28, 128), torch.empty(1, 4, 128)
    record = (query.data_ptr(), torch.tensor([3]), torch.empty(4, 128))
    owner = SimpleNamespace(_pspkv_pending_rope={key.data_ptr(): record})
    assert consume_deferred_rope(owner, query, key) is record
    assert consume_deferred_rope(owner, query, key) is None
    owner._pspkv_pending_rope[key.data_ptr()] = record
    with pytest.raises(RuntimeError, match="pairing changed"):
        consume_deferred_rope(owner, query.clone(), key)


def test_native_bf16_rejects_custom_graph_ab(monkeypatch):
    monkeypatch.setenv("PSPKV_COMPARE_GRAPH_AB", "1")
    monkeypatch.setenv("PSPKV_COMPARE_FORMAT", "stock-bf16")
    with pytest.raises(ValueError, match="independent eager/graph processes"):
        install_graph_bypass()
