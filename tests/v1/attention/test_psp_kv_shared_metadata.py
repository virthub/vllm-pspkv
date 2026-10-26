# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from ascend.shared_metadata import prepare_metadata, reuse_enabled


def metadata(tables=None, lengths=None, dtype=torch.int32):
    return SimpleNamespace(
        block_tables=(
            torch.tensor([[7, 2, 9, 1]], dtype=dtype) if tables is None else tables
        ),
        seq_lens=torch.tensor([135, 0, 0], dtype=dtype),
        seq_lens_cpu=torch.tensor([135, 0, 0], dtype=torch.int32),
    )


def owner_for(source):
    return SimpleNamespace(attn_metadata={f"layer{i}": source for i in range(28)})


def prepare(owner, source, slots, lengths=(135,)):
    return prepare_metadata(
        owner, source.block_tables, slots, lengths, 128, torch.device("cpu")
    )


def test_borrows_existing_buffers_and_prepares_once_for_28_layers():
    source = metadata()
    owner = owner_for(source)
    slots = torch.tensor([134], dtype=torch.int64)
    rows = [prepare(owner, source, slots[:1]) for _ in range(28)]
    assert all(row is rows[0] for row in rows)
    row = rows[0]
    assert row.lengths.data_ptr() == source.seq_lens.data_ptr()
    assert row.tables.data_ptr() == source.block_tables.data_ptr()
    assert row.slots.data_ptr() == slots.data_ptr()
    assert row.get_cpu_lengths().data_ptr() == source.seq_lens_cpu.data_ptr()
    assert row.tables.tolist() == [[7, 2]]
    assert owner._pspkv_step_metadata.builds == 1
    assert owner._pspkv_step_metadata.hits == 27


def test_conversion_is_once_per_forward():
    source = metadata(dtype=torch.int64)
    owner = owner_for(source)
    slots = torch.tensor([134], dtype=torch.int32)
    row = prepare(owner, source, slots)
    for _ in range(27):
        assert prepare(owner, source, slots) is row
    assert row.tables.dtype == row.lengths.dtype == torch.int32
    assert row.slots.dtype == torch.int64
    assert row.tables.data_ptr() != source.block_tables.data_ptr()
    assert row.slots.data_ptr() != slots.data_ptr()
    assert not row.borrowed_lengths


def test_prefers_retained_device_lengths_to_plugins_cpu_field():
    source = metadata()
    source.seq_lens = torch.tensor([999], dtype=torch.int64)
    source.psp_kv_device_seq_lens = torch.tensor([135], dtype=torch.int32)
    row = prepare(owner_for(source), source, torch.tensor([134]))
    assert row.lengths.data_ptr() == source.psp_kv_device_seq_lens.data_ptr()
    assert row.lengths.tolist() == [135]


def test_new_forward_does_not_reuse_equal_length_request_mappings():
    source = metadata(dtype=torch.int64)
    slots = torch.tensor([134], dtype=torch.int32)
    first = prepare(owner_for(source), source, slots)
    source.block_tables.copy_(torch.tensor([[4, 3, 6, 8]]))
    slots.fill_(390)
    second = prepare(owner_for(source), source, slots)
    assert second is not first
    assert first.tables.tolist() == [[7, 2]]
    assert second.tables.tolist() == [[4, 3]]
    assert first.slots.tolist() == [134]
    assert second.slots.tolist() == [390]


def test_separate_cache_groups_have_separate_prepared_mappings():
    a, b = metadata(), metadata(tables=torch.tensor([[8, 6, 2, 5]]))
    owner = SimpleNamespace(attn_metadata=[{"a": a}, {"b": b}])
    slots = torch.tensor([134])
    before, after = prepare(owner, a, slots), prepare(owner, b, slots)
    assert before is not after
    assert after.tables.tolist() == [[8, 6]]
    assert owner._pspkv_step_metadata.builds == 2


def test_no_forward_owner_never_caches_across_calls():
    source = metadata()
    slots = torch.tensor([134])
    a, b = prepare(None, source, slots), prepare(None, source, slots)
    assert a is not b
    assert a.lengths.data_ptr() != b.lengths.data_ptr()


def test_fallback_lengths_are_created_once_and_cpu_lengths_are_lazy():
    source = SimpleNamespace(block_tables=torch.tensor([[0, 1, 2]]))
    owner = owner_for(source)
    slots = torch.tensor([134])
    row = prepare(owner, source, slots)
    assert not row.borrowed_lengths
    assert row.lengths.tolist() == [135]
    assert row.cpu_lengths is None
    cpu = row.get_cpu_lengths()
    assert row.get_cpu_lengths() is cpu
    assert prepare(owner, source, slots) is row
    changed = prepare(owner, source, slots, (136,))
    assert changed.lengths.tolist() == [136]
    assert changed is not row


def test_does_not_trigger_a_lazy_cpu_copy_property():
    class Source:
        block_tables = torch.tensor([[0, 1]])
        seq_lens = torch.tensor([135], dtype=torch.int32)

        @property
        def seq_lens_cpu(self):
            raise AssertionError("device-to-host copy must not be triggered")

    source = Source()
    row = prepare(owner_for(source), source, torch.tensor([134]))
    assert row.get_cpu_lengths().tolist() == [135]


def test_noncontiguous_multi_request_tables_are_trimmed_correctly():
    source = metadata(tables=torch.tensor([[7, 2, 9, 1], [5, 8, 3, 4]]))
    source.seq_lens = torch.tensor([127, 129, 0], dtype=torch.int32)
    source.seq_lens_cpu = source.seq_lens.clone()
    row = prepare(owner_for(source), source, torch.tensor([126, 128]), (127, 129))
    assert row.tables.tolist() == [[7, 2], [5, 8]]
    assert row.tables.is_contiguous()
    assert row.lengths.tolist() == [127, 129]


@pytest.mark.parametrize("field", ["seq_lens", "seq_lens_cpu"])
def test_rejects_builder_lengths_with_wrong_shape(field):
    source = metadata()
    setattr(source, field, torch.empty(0, dtype=torch.int32))
    with pytest.raises(ValueError, match="cover the requests"):
        prepare(owner_for(source), source, torch.tensor([134]))


def test_control_mode_is_fixed_within_a_forward(tmp_path, monkeypatch):
    control = tmp_path / "mode.txt"
    monkeypatch.setenv("PSPKV_METADATA_CONTROL_FILE", str(control))
    control.write_text("0")
    owner = SimpleNamespace()
    assert not reuse_enabled(owner)
    control.write_text("1")
    assert not reuse_enabled(owner)
    assert reuse_enabled(SimpleNamespace())


def test_default_mode_can_be_disabled(monkeypatch):
    monkeypatch.delenv("PSPKV_METADATA_CONTROL_FILE", raising=False)
    monkeypatch.delenv("PSPKV_ASCEND_SHARE_METADATA", raising=False)
    assert reuse_enabled(None)
    monkeypatch.setenv("PSPKV_ASCEND_SHARE_METADATA", "0")
    assert not reuse_enabled(None)
