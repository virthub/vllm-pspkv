# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Prepare packed-attention metadata once per vLLM forward, never per length.

The owner is the current ForwardContext. Keeping converted tensors on that
owner prevents stale page/slot mappings when a new request has the same length
or vLLM updates its persistent device buffers in place. Layer-specific KV,
precision and scale tensors are deliberately excluded.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch


def current_owner():
    from vllm.forward_context import (
        get_forward_context,
        is_forward_context_available,
    )

    return get_forward_context() if is_forward_context_available() else None


def reuse_enabled(owner) -> bool:
    """An optional test-only control file is read once per forward, not layer."""
    control = os.environ.get("PSPKV_METADATA_CONTROL_FILE")
    if not control:
        return os.environ.get("PSPKV_ASCEND_SHARE_METADATA", "1") == "1"
    if owner is not None and hasattr(owner, "_pspkv_metadata_mode"):
        return owner._pspkv_metadata_mode
    value = Path(control).read_text().strip()
    if value not in ("0", "1"):
        raise ValueError("metadata control file must contain 0 or 1")
    enabled = value == "1"
    if owner is not None:
        owner._pspkv_metadata_mode = enabled
    return enabled


@dataclass
class PreparedMetadata:
    tables: torch.Tensor
    slots: torch.Tensor
    lengths: torch.Tensor
    host_lengths: tuple[int, ...]
    cpu_lengths: torch.Tensor | None
    borrowed_lengths: bool
    # Retain source objects so their ids cannot be recycled during this forward.
    source: Any = None
    input_tables: torch.Tensor | None = None
    input_slots: torch.Tensor | None = None

    def get_cpu_lengths(self):
        if self.cpu_lengths is None:
            self.cpu_lengths = torch.tensor(self.host_lengths, dtype=torch.int32)
        return self.cpu_lengths


@dataclass
class ForwardMetadata:
    by_table: dict = field(default_factory=dict)
    prepared: dict = field(default_factory=dict)
    builds: int = 0
    hits: int = 0
    borrowed_lengths: int = 0
    created_lengths: int = 0


def _sources(value):
    if isinstance(value, dict):
        for child in value.values():
            yield from _sources(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _sources(child)
    elif value is not None:
        yield value


def _cache(owner):
    if owner is not None and hasattr(owner, "_pspkv_step_metadata"):
        return owner._pspkv_step_metadata
    cache = ForwardMetadata()
    if owner is not None:
        for metadata in _sources(getattr(owner, "attn_metadata", None)):
            tables = getattr(metadata, "block_tables", None)
            if isinstance(tables, torch.Tensor):
                cache.by_table[id(tables)] = metadata
        owner._pspkv_step_metadata = cache
    return cache


def prepare_metadata(owner, block_tables, slots, seq_lens, block_size, device):
    """Borrow builder-owned lengths and share trimmed/converted views by group.

    With no forward owner, preparation remains uncached. This supports direct
    operator callers without silently retaining tensors across arbitrary calls.
    Inputs must remain unchanged during a forward, as in vLLM's eager runner.
    """
    cache = _cache(owner)
    host_lengths = tuple(seq_lens)
    sequences = len(host_lengths)
    pages = (max(host_lengths) + block_size - 1) // block_size
    # Slots are often freshly sliced views in each layer; data_ptr includes
    # their storage offset. A distinct page-table object denotes a cache group.
    key = (
        id(block_tables),
        slots.data_ptr(),
        tuple(slots.shape),
        slots.stride(),
        slots.dtype,
        host_lengths,
        block_size,
        device,
    )
    if key in cache.prepared:
        cache.hits += 1
        return cache.prepared[key]
    metadata = cache.by_table.get(id(block_tables))
    source_lengths = getattr(metadata, "psp_kv_device_seq_lens", None)
    if source_lengths is None:
        source_lengths = getattr(metadata, "seq_lens", None)
    if isinstance(source_lengths, torch.Tensor):
        if source_lengths.ndim != 1 or source_lengths.numel() < sequences:
            raise ValueError("builder context lengths do not cover the requests")
        lengths = (
            source_lengths[:sequences]
            .to(
                device=device,
                dtype=torch.int32,
                non_blocking=(
                    source_lengths.device.type == "cpu" and source_lengths.is_pinned()
                ),
            )
            .contiguous()
        )
        borrowed = (
            source_lengths.device == device
            and source_lengths.dtype == torch.int32
            and source_lengths.is_contiguous()
        )
    else:
        lengths = torch.tensor(host_lengths, dtype=torch.int32, device=device)
        borrowed = False
    # Read only an existing CPU field. Never obtain CPU lengths by copying
    # the device tensor back to the host.
    cpu_source = (
        vars(metadata).get("seq_lens_cpu") if hasattr(metadata, "__dict__") else None
    )
    cpu_lengths = None
    if isinstance(cpu_source, torch.Tensor) and cpu_source.device.type == "cpu":
        if cpu_source.ndim != 1 or cpu_source.numel() < sequences:
            raise ValueError("builder CPU lengths do not cover the requests")
        cpu_lengths = cpu_source[:sequences].to(torch.int32).contiguous()
    prepared = PreparedMetadata(
        tables=block_tables[:sequences, :pages].to(torch.int32).contiguous(),
        slots=slots.to(torch.int64).contiguous(),
        lengths=lengths,
        host_lengths=host_lengths,
        cpu_lengths=cpu_lengths,
        borrowed_lengths=borrowed,
        source=metadata,
        input_tables=block_tables,
        input_slots=slots,
    )
    cache.prepared[key] = prepared
    cache.builds += 1
    cache.borrowed_lengths += int(borrowed)
    cache.created_lengths += int(not borrowed)
    return prepared
