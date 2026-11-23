# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Virthub KV connector adapter for vLLM 0.34 (Model Runner V2).

The connector implementation ships in the optional ``virthub`` Python package
(``virthub.vllm``), which supports several vLLM generations: ``...V1`` for the
removed Model Runner V1 and ``...V2`` for Model Runner V2, the only GPU runner
since vLLM 0.34. The package selects V1 by default for backward compatibility,
so this adapter pins V2 and adapts the constructor to the API vLLM 0.34's
``KVConnectorFactory`` uses:

    cls(vllm_config, role, kv_cache_config)

The package expects its own config mapping instead, and it deliberately skips
``KVConnectorBase_V1.__init__``; the attributes vLLM reads (``role``,
``_kv_transfer_config``, ``_kv_cache_config``) are therefore populated here.

The adapter also owns ``set_block_transfer_info_provider``, the hook PSP-KV's
``PSPKVState`` uses to report a block's byte size and sidecar to the storage
layer (see ``vllm/v1/attention/psp_kv_virthub.py``).
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from typing import Any

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1
from vllm.logger import init_logger

logger = init_logger(__name__)

try:  # pragma: no cover - exercised only when the optional package is present
    from virthub.vllm import VirthubKVConnectorV2 as _VirthubConnectorImpl
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The 'virthub' package is required for the 'virthub' KV connector. "
        "Install it with './install.sh --virthub' (or "
        "'pip install -e tools/virthub/bindings/python')."
    ) from exc


def _to_mapping(config: Any) -> dict[str, Any]:
    """Best-effort conversion of a config object into the package's mapping."""
    if config is None:
        return {}
    if isinstance(config, dict):
        return dict(config)
    extra = getattr(config, "kv_connector_extra_config", None)
    if isinstance(extra, dict):
        return dict(extra)
    if dataclasses.is_dataclass(config):
        return dataclasses.asdict(config)
    return {}


class VirthubConnector(_VirthubConnectorImpl, KVConnectorBase_V1):
    """Virthub connector bound to vLLM 0.34's connector contract."""

    def __init__(
        self,
        vllm_config: Any = None,
        role: Any = None,
        kv_cache_config: Any = None,
        config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        connector_config = config
        if connector_config is None:
            connector_config = _to_mapping(
                getattr(vllm_config, "kv_transfer_config", None)
            )
        # The package's __init__ takes the connector config and skips the vLLM
        # base constructor, so vLLM's own bookkeeping is set here.
        _VirthubConnectorImpl.__init__(self, connector_config or {})
        self._vllm_config = vllm_config
        self._kv_transfer_config = getattr(vllm_config, "kv_transfer_config", None)
        self._kv_cache_config = kv_cache_config
        self._kv_cache_manager = None
        self._role = role
        self._connector_metadata = None
        self._block_transfer_info_provider: Callable[[int], Any] | None = None
        logger.info(
            "Initialized the Virthub KV connector (role=%s, Model Runner V2)",
            role,
        )

    # -- Model Runner V2 contract --------------------------------------------
    # The package implements these against older vLLM generations: blocks were
    # passed explicitly, and the save/load hooks were async. vLLM 0.34 calls them
    # with a forward context plus keyword arguments, synchronously. Block
    # registration with the storage layer now happens through PSP-KV's
    # ``set_block_transfer_info_provider`` hook instead of these callbacks.
    def start_load_kv(self, *args: Any, **kwargs: Any) -> None:
        """Load KV for this step; no-op unless called with explicit blocks."""
        if args and isinstance(args[0], (list, tuple)):
            _VirthubConnectorImpl.start_load_kv(self, args[0], *args[1:], **kwargs)

    def wait_for_save(self, *args: Any, **kwargs: Any) -> None:
        """Synchronous wait-for-save (the base API is not async in 0.34)."""

    def save_kv_layer(self, *args: Any, **kwargs: Any) -> None:
        """Persist one layer; vLLM 0.34 passes (layer_name, kv_cache, meta)."""
        # The legacy package signature is (layer_idx, kv_blocks, worker); only
        # that form carries block descriptors to register.
        if len(args) >= 2 and isinstance(args[1], (list, tuple)):
            _VirthubConnectorImpl.save_kv_layer(self, *args, **kwargs)

    # -- PSP-KV integration ----------------------------------------------------
    def set_block_transfer_info_provider(
        self, provider: Callable[[int], Any] | None
    ) -> None:
        """Register the callback that reports a block's size and sidecar."""
        self._block_transfer_info_provider = provider

    def block_transfer_info(self, block_id: int) -> Any | None:
        """Return the provider's block info, or None when nothing is registered."""
        if self._block_transfer_info_provider is None:
            return None
        return self._block_transfer_info_provider(block_id)
