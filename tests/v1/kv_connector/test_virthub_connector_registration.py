# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Virthub KV connector registration and adapter contract.

PSP-KV's cache-storage side (``PSPKVState``) talks to Virthub through the KV
connector API, so the ``virthub`` connector must be resolvable by name and must
be bound to the Model Runner V2 contract (the only GPU runner since vLLM 0.34).

Registration is lazy: vLLM must import fine without the optional ``virthub``
package and only fail when a config actually selects the connector.
"""

import pytest

from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory

try:  # pragma: no cover - depends on the optional package
    import virthub  # noqa: F401

    HAS_VIRTHUB = True
except ImportError:  # pragma: no cover
    HAS_VIRTHUB = False


def test_virthub_connector_is_registered():
    """The connector name resolves through the factory registry."""
    assert "virthub" in KVConnectorFactory._registry


@pytest.mark.skipif(HAS_VIRTHUB, reason="the virthub package is installed")
def test_resolution_without_package_raises_import_error():
    """Without the package, selecting the connector must be actionable.

    ``KeyError`` would mean the registration is missing; an ``ImportError``
    naming the package means the wiring is right and only the optional
    dependency is absent.
    """
    with pytest.raises(ImportError, match="virthub"):
        KVConnectorFactory.get_connector_class_by_name("virthub")


@pytest.mark.skipif(not HAS_VIRTHUB, reason="the virthub package is not installed")
def test_resolves_to_the_model_runner_v2_adapter():
    """The fork's adapter is bound to the package's V2 connector (MRV2)."""
    from virthub.vllm import VirthubKVConnectorV1, VirthubKVConnectorV2

    from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1
    from vllm.distributed.kv_transfer.kv_connector.v1.virthub_connector import (
        VirthubConnector,
    )

    cls = KVConnectorFactory.get_connector_class_by_name("virthub")
    assert cls is VirthubConnector
    assert issubclass(cls, KVConnectorBase_V1)
    assert issubclass(cls, VirthubKVConnectorV2)
    assert not issubclass(cls, VirthubKVConnectorV1)


@pytest.mark.skipif(not HAS_VIRTHUB, reason="the virthub package is not installed")
def test_load_and_save_hooks_accept_the_mr_v2_call_signature():
    """VLLM 0.34 calls these with a forward context and keyword arguments.

    The package implements them against an older generation (explicit blocks,
    ``async def wait_for_save``), which raised ``TypeError: start_load_kv() got
    an unexpected keyword argument 'scheduler_output'`` and killed the engine on
    the first request. The adapter must accept the new call shape.
    """
    from vllm.distributed.kv_transfer.kv_connector.v1.virthub_connector import (
        VirthubConnector,
    )

    connector = VirthubConnector(None, None, None)
    try:
        # Forward context + the keyword arguments MRV2 builds for the worker.
        connector.start_load_kv(
            object(),
            scheduler_output=object(),
            request_state_indices=[0],
            request_ids=["req-0"],
            num_tokens=1,
        )
        # Synchronous wait (the package's is a coroutine function).
        assert connector.wait_for_save() is None
        # vLLM calls save_kv_layer with a layer name, not block descriptors.
        connector.save_kv_layer("model.layers.0.attn", object(), object())
    finally:
        connector.close()


@pytest.mark.skipif(not HAS_VIRTHUB, reason="the virthub package is not installed")
def test_block_transfer_info_provider_round_trip():
    """PSPKVState registers a provider and the adapter forwards its answer."""
    from vllm.distributed.kv_transfer.kv_connector.v1.virthub_connector import (
        VirthubConnector,
    )

    connector = VirthubConnector(None, None, None)
    try:
        assert connector.block_transfer_info(7) is None
        connector.set_block_transfer_info_provider(lambda block_id: ("info", block_id))
        assert connector.block_transfer_info(7) == ("info", 7)
        connector.set_block_transfer_info_provider(None)
        assert connector.block_transfer_info(7) is None
    finally:
        connector.close()
