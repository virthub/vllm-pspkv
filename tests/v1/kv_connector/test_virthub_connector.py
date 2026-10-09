# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Integration tests for the Virthub KV-cache storage connector.

These verify the fork-level wiring added by ``./install.sh --virthub``:

* the ``virthub`` connector is registered in ``KVConnectorFactory``,
* it resolves to ``virthub.vllm.VirthubKVConnector``,
* and the software-fallback (in-memory stub) save/load path works.

They run on CPU and skip if the ``virthub`` bindings are not installed.
"""

from types import SimpleNamespace

import pytest

pytest.importorskip(
    "virthub", reason="virthub bindings not installed (run ./install.sh --virthub)"
)

from virthub.vllm import VirthubKVConnector  # noqa: E402
from vllm.distributed.kv_transfer.kv_connector.factory import (  # noqa: E402
    KVConnectorFactory,
)


def _block(block_id=123):
    return SimpleNamespace(
        block_id=block_id, gpu_ptr=0x7FFF10000000, size=2097152
    )


def _worker(device_id=0):
    return SimpleNamespace(device_id=device_id)


def test_virthub_connector_is_registered():
    cls = KVConnectorFactory.get_connector_class_by_name("virthub")
    assert cls is VirthubKVConnector


def test_connector_initializes_with_in_memory_stub():
    connector = VirthubKVConnector({"transport": {"default_protocol": "tcp"}})
    try:
        assert connector._client is not None
        # Software fallback: no compiled Rust client → in-memory stub is used.
        assert type(connector._client).__name__ == "_StubRustConnector"
    finally:
        connector.close()


def test_save_kv_layer_registers_block():
    connector = VirthubKVConnector({})
    try:
        block = _block()
        worker = _worker()
        connector.save_kv_layer(0, [block], worker)
        assert block.block_id in connector.block_map
        assert block.block_id in connector._client.regions
    finally:
        connector.close()


def test_start_load_kv_swaps_in_registered_block():
    connector = VirthubKVConnector({})
    try:
        block = _block()
        worker = _worker()
        connector.save_kv_layer(0, [block], worker)
        # The stub's swap_in_remote_block is a no-op; the point is that every
        # registered block is reachable through start_load_kv without raising.
        connector.start_load_kv([block], worker)
    finally:
        connector.close()


def test_unregister_block_removes_from_map_and_regions():
    connector = VirthubKVConnector({})
    try:
        block = _block()
        connector.save_kv_layer(0, [block], _worker())
        connector.unregister_block(block.block_id)
        assert block.block_id not in connector.block_map
        assert block.block_id not in connector._client.regions
    finally:
        connector.close()


def test_close_clears_all_blocks():
    connector = VirthubKVConnector({})
    for i, block_id in enumerate((1, 2, 3)):
        connector.save_kv_layer(0, [_block(block_id)], _worker(i))
    assert len(connector.block_map) == 3
    connector.close()
    assert connector.block_map == {}
    assert connector._client.regions == {}
