# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PSP-KV model-runner state.

This is the vLLM V2 (``vllm/v1/worker/gpu``) home of the PSP-KV state that
lived in the V1 ``GPUModelRunner``: the env/TOML configuration, the per-block
precision/scale/TMA tables, the CPU dispatch flags, the introspection used by
the profiling scripts, and the per-step prefill quantization.

Every table keeps the V1 layout exactly:

* ``precision_table``: uint8, ``k_fmt | (v_fmt << 2)`` per kernel block, with
  the legacy ``4`` sentinel for uniform block-scaled int8.
* ``scale_table``: float32 ``[num_layers, 2 * blocks, num_kv_heads, groups]``
  (``groups = head_size // 16``), K rows then V rows.
* ``tma_descriptors``: int64 ``[blocks, 16]`` (BT-KV only).
* ``block_offsets``: int32 ``[2 * blocks]``, K offsets then V offsets.

The one unavoidable difference is ``block_offsets``: V2 allocates the KV cache
as fixed-stride pages, so a block's K and V bytes are adjacent inside one page
(``[K block | V block]``) instead of the V1 ``[all K blocks | all V blocks]``
region split. The offsets are absolute byte offsets into the layer's flat
byte buffer and the kernels only ever consume them through
``block_offsets``, so the numerics are unchanged.
"""

from __future__ import annotations

import os
from typing import Any

import torch

from vllm import envs
from vllm.config import VllmConfig
from vllm.logger import init_logger

logger = init_logger(__name__)

# Format-code <-> bytes-per-element, matching the precision_table encoding.
# 0 = lossless (target dtype), 1/2 = fp8 e4m3/e5m2, 3 = fp4/int4, 4 = int8.
_FMT_BYTES = {0: None, 1: 1.0, 2: 1.0, 3: 0.5, 4: 1.0}

_ACTIVE_STATE: PSPKVState | None = None


def set_active_psp_kv_state(state: PSPKVState | None) -> None:
    """Register the runner's PSP-KV state for metadata assembly."""
    global _ACTIVE_STATE
    _ACTIVE_STATE = state


def get_active_psp_kv_state() -> PSPKVState | None:
    return _ACTIVE_STATE


def _load_toml(path: str) -> dict:
    """Best-effort TOML load; returns {} on any failure."""
    try:
        try:
            import tomllib  # Python 3.11+
        except ImportError:  # pragma: no cover
            import tomli as tomllib  # type: ignore[no-redef]
        with open(path, "rb") as f:
            return tomllib.load(f)
    except Exception:
        return {}


def _inner_precision_to_code(fmt) -> int:
    """Map an inner-precision spec to the precision_table value.

    0 = lossless (fp16/bf16), 1 = fp8 e4m3, 2 = fp8 e5m2, 3 = fp4/int4,
    4 = block-scaled int8 (W8A8).
    """
    if isinstance(fmt, int):
        return max(0, min(4, fmt))
    s = str(fmt).strip().lower()
    if s in ("0", "lossless", "fp16", "bf16"):
        return 0
    if s in ("2", "e5m2"):
        return 2
    if s in ("3", "fp4", "int4", "e2m1", "nvfp4"):
        return 3
    if s in ("4", "int8", "w8a8"):
        return 4
    return 1  # "1", "e4m3", or anything else -> e4m3


def _read_psp_kv_format_generation() -> int:
    """Resolve the PSP-KV physical format generation (launcher).

    Precedence:
      1. ``PSPKV_FORMAT`` env var (``VIRTHUB_FORMAT`` is a deprecated alias).
      2. ``[format].generation`` in ``conf/pspkv.toml`` (path from
         ``PSPKV_CONFIG``), falling back to ``[precision].format_generation``
         in ``conf/virthub.toml`` (``VIRTHUB_CONFIG``) for backward
         compatibility.
      3. Default ``1`` (Enhanced format).
    """
    env_val = envs.PSPKV_FORMAT or os.getenv("VIRTHUB_FORMAT")
    if env_val is not None and env_val.strip() != "":
        return int(env_val)

    config_path = envs.PSPKV_CONFIG
    data = _load_toml(config_path)
    generation = data.get("format", {}).get("generation")
    if generation is not None:
        return int(generation)

    # Backward-compat: the launcher used to live in conf/virthub.toml.
    legacy_path = os.getenv("VIRTHUB_CONFIG", "./conf/virthub.toml")
    legacy = _load_toml(legacy_path)
    return int(legacy.get("precision", {}).get("format_generation", 1))


def _check_psp_kv_format_generation(generation: int) -> int:
    """Reject format generations this port cannot decode correctly.

    Generations 0 (Basic) and 2 (BT-KV) still assume head-major pages while the
    byte-buffer cache is allocated token-major, so they mis-decode whenever the
    model has more than one KV head. Only generation 1 honours the layout the
    runner builds.
    """
    if generation not in (0, 1, 2):
        raise ValueError(
            f"PSP-KV format generation {generation} is not supported; expected "
            "0 (Basic), 1 (Enhanced), or 2 (BT-KV)."
        )
    if generation != 1:
        names = {0: "Basic", 2: "BT-KV"}
        raise ValueError(
            f"PSP-KV format generation {generation} ({names[generation]}) is not "
            "supported by this port: its kernels assume head-major pages while "
            "the cache is allocated token-major, so it would mis-decode models "
            "with more than one KV head. Set PSPKV_FORMAT=1 (Enhanced)."
        )
    return generation


def _read_psp_kv_precision() -> int:
    """Resolve the default per-block inner precision.

    Precedence:
      1. ``PSPKV_PRECISION`` env var ("lossless"/"e4m3"/"e5m2"/"fp4", or 0-3).
      2. ``[precision].inner_precision`` in ``conf/pspkv.toml`` (path from
         ``PSPKV_CONFIG``); ``fp8_format`` is a deprecated alias.
      3. Default ``"e4m3"`` -> 1.
    """
    env_val = envs.PSPKV_PRECISION
    if env_val is not None and env_val.strip() != "":
        return _inner_precision_to_code(env_val)

    data = _load_toml(envs.PSPKV_CONFIG)
    precision = data.get("precision", {})
    fmt = precision.get("inner_precision", precision.get("fp8_format", "e4m3"))
    return _inner_precision_to_code(fmt)


def _read_psp_kv_storage_mode() -> str:
    """Resolve the KV-cache storage layout mode.

    Precedence:
      1. ``PSPKV_STORAGE`` env var.
      2. ``[storage].mode`` in ``conf/pspkv.toml`` (path from
         ``PSPKV_CONFIG``).
      3. Default ``"fixed"``.
    """
    env_val = envs.PSPKV_STORAGE
    if env_val is not None and env_val.strip() != "":
        return env_val.strip().lower()

    data = _load_toml(envs.PSPKV_CONFIG)
    mode = str(data.get("storage", {}).get("mode", "fixed")).strip().lower()
    return mode if mode in ("fixed", "variable") else "fixed"


def _read_psp_kv_profile() -> bool:
    """Whether to emit per-step PSP-KV prefill timing (PSPKV_PROFILE=1)."""
    return envs.PSPKV_PROFILE


def _read_psp_kv_adaptive_precision() -> bool:
    """Whether to derive per-block precision from the prefill magnitude."""
    return envs.PSPKV_ADAPTIVE_PRECISION


def _read_psp_kv_adaptive_e() -> float:
    """Target relative error for the adaptive precision derivation."""
    return envs.PSPKV_ADAPTIVE_E


def psp_kv_block_size() -> int:
    """Tokens per PSP-KV kernel block (must match ``PSP_KV_BLOCK_SIZE``)."""
    return envs.PSP_KV_BLOCK_SIZE


def _psp_specs(kv_cache_config) -> list:
    """PSP-KV cache specs in ``kv_cache_config`` (flattening uniform specs)."""
    specs = []
    for group in kv_cache_config.kv_cache_groups:
        specs.extend(_psp_specs_for_group(group))
    return specs


class PSPKVState:
    """PSP-KV tables, dispatch flags and per-step prefill quantization."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        max_num_tokens: int,
    ):
        self.model_config = vllm_config.model_config
        self.parallel_config = vllm_config.parallel_config
        self.cache_config = vllm_config.cache_config
        self.device = device
        self.dtype = self.model_config.dtype
        self.max_num_tokens = max_num_tokens
        self.num_kv_heads = self.model_config.get_num_kv_heads(self.parallel_config)
        self.num_layers = self.model_config.get_num_layers(self.parallel_config)
        self.head_size = self.model_config.get_head_size()
        self.lossless_bytes = torch.tensor([], dtype=self.dtype).element_size()

        # Parsed env/TOML configuration.
        self.format_generation = _check_psp_kv_format_generation(
            _read_psp_kv_format_generation()
        )
        self.precision_default = _read_psp_kv_precision()  # 1=e4m3, 2=e5m2
        # Lossless inner precision has no compression to offer and its decode
        # kernel is slower than FlashAttention (0.66-0.71x at batch 32), so it is
        # rejected rather than silently running a losing configuration.
        if self.precision_default == 0:
            raise ValueError(
                "PSPKV_PRECISION=lossless is not supported: a lossless PSP-KV "
                "cache stores the full-precision KV and only decodes more slowly "
                "than FlashAttention. Use the default backend for lossless KV, "
                "or an fp8/int8/int4 inner precision to compress the cache."
            )
        self.storage_mode = _read_psp_kv_storage_mode()
        self.psp_kv_profile = _read_psp_kv_profile()
        self.psp_kv_adaptive_precision = _read_psp_kv_adaptive_precision()
        # split-K: PSPKV_TC_NUM_SPLITS > 0 pins it, otherwise it follows the batch.
        manual_splits = int(os.environ.get("PSPKV_TC_NUM_SPLITS", "0") or 0)
        self._manual_num_splits = manual_splits > 0
        self.num_split_override: int | None = None
        self._auto_num_splits_applied = -1
        if not self._manual_num_splits:
            # Leave the variable unset so the wrapper's own default applies
            # until the first step picks a count (0 would divide by zero).
            os.environ.pop("PSPKV_TC_NUM_SPLITS", None)
        self.psp_kv_adaptive_e = _read_psp_kv_adaptive_e()

        # The default per-K/V precision (packed k_fmt | (v_fmt << 2)). The
        # adaptive derive later refines this per block; the cache byte budget is
        # derived from the packed default (whatever E produces) rather than a
        # fixed constant. An explicit PSPKV_AVG_BYTES env value still wins.
        if self.psp_kv_adaptive_precision:
            # Default codes for blocks the derivation has not touched: K is
            # fp4 once the tolerance loosens past ~0.5, else fp8; V is always fp4.
            self.psp_kv_default_k = (
                3 if self.psp_kv_adaptive_e >= 0.5 else 1
            )
            self.psp_kv_default_v = 3
            # The page geometry is a separate concern from the default code: it
            # must fit the widest format the derivation can emit for each
            # operand. That width is determined by the tolerance: at E >= 0.5
            # every K block is fp4, so reserving fp8 would waste half the K
            # region. The derivation is clamped to k_cap_fmt so a block can never
            # be assigned a wider K than the reservation.
            self.psp_kv_k_cap_fmt = 3 if self.psp_kv_adaptive_e >= 0.5 else 1
            geom_k = 0.5 if self.psp_kv_k_cap_fmt == 3 else 1.0
            geom_v = 0.5
        else:
            self.psp_kv_default_k = self.precision_default
            self.psp_kv_default_v = self.precision_default
            self.psp_kv_k_cap_fmt = self.psp_kv_default_k
            geom_k = self._fmt_bytes(self.psp_kv_default_k)
            geom_v = self._fmt_bytes(self.psp_kv_default_v)
        self.geom_k_bytes = geom_k
        self.geom_v_bytes = geom_v
        psp_kv_avg_bytes = (geom_k + geom_v) / 2.0
        os.environ.setdefault("PSPKV_AVG_BYTES", str(psp_kv_avg_bytes))

        # GPU tables (allocated in initialize()).
        self.precision_table: torch.Tensor | None = None
        self.scale_table: torch.Tensor | None = None
        self.tma_descriptors: torch.Tensor | None = None
        self.block_offsets: torch.Tensor | None = None
        self.block_sizes_cpu: torch.Tensor | None = None
        self.block_offsets_cpu: torch.Tensor | None = None
        self.precision_cpu: torch.Tensor | None = None
        self.scale_cpu: torch.Tensor | None = None

        # CPU-side dispatch flags.
        self.precision_is_lossless = False
        self.precision_fp8_fmt = 0
        self.precision_v_fmt = 0
        self.precision_k_fmt = 0
        self.precision_int_fmt = 0
        self.precision_k_lossless = False
        self.precision_v_lossless = False
        self.k_cache_bytes = 0
        self.v_cache_bytes = 0
        self.psp_kv_avg_bytes = psp_kv_avg_bytes

        # Cache-group geometry (fixed by initialize()).
        self.kv_cache_config = None
        self.num_blocks = 0
        self.block_size = 0
        self.kernel_block_size = 0
        self.page_bytes = 0
        self.k_block_bytes = 0
        self.v_block_bytes = 0
        self.attn_gid: int | None = None
        self.layer_names: list[str] = []
        self.layer_caches: list[tuple[int, torch.Tensor]] = []

        # Pinned host staging buffers and dirty-block batching (virthub).
        self.h_precision_table: torch.Tensor | None = None
        self.h_scale_table: torch.Tensor | None = None
        self.h_tma_descriptors: torch.Tensor | None = None
        self.h_has_scale: list[bool] = []
        self.max_dirty_blocks = 256
        self.h_dirty_indices: torch.Tensor | None = None
        self.d_dirty_indices: torch.Tensor | None = None
        self.h_dirty_scale_indices: torch.Tensor | None = None
        self.d_dirty_scale_indices: torch.Tensor | None = None
        self.h_scale_batch_buffer: torch.Tensor | None = None
        self.d_scale_batch_buffer: torch.Tensor | None = None
        self.h_prec_batch_buffer: torch.Tensor | None = None
        self.d_prec_batch_buffer: torch.Tensor | None = None
        self.dirty_block_ids: list[int] = []
        self.is_dirty: list[bool] = []

        # Prefill scratch, allocated with the tables.
        self.prefill_kv_scratch: torch.Tensor | None = None

    def _fmt_bytes(self, fmt: int) -> float:
        """Bytes per element for a precision_table format code."""
        b = _FMT_BYTES.get(fmt, 2.0)
        return float(self.lossless_bytes) if b is None else b

    # ------------------------------------------------------------------
    # Allocation
    # ------------------------------------------------------------------
    @property
    def enabled(self) -> bool:
        """Whether the model's KV cache is PSP-KV."""
        return self.attn_gid is not None

    def initialize(
        self,
        kv_cache_config,
        layer_caches: dict[str, torch.Tensor] | None = None,
    ) -> bool:
        """Allocate the PSP-KV tables for ``kv_cache_config``.

        Returns True when the model uses PSP-KV (and the tables were built).
        """
        specs = _psp_specs(kv_cache_config)
        if not specs:
            return False
        spec = specs[0]
        if int(spec.head_size_v) != int(spec.head_size):
            raise NotImplementedError(
                "PSP-KV requires head_size_v == head_size; got "
                f"{spec.head_size_v} vs {spec.head_size}."
            )

        total_blocks = kv_cache_config.num_blocks
        if not total_blocks:
            total_blocks = self.cache_config.num_gpu_blocks or 0
        if not total_blocks:
            return False

        # Kernel-level block count: hybrid models share a larger unified memory
        # block size across groups, so the number of kernel blocks is
        # num_blocks * (block_size // kernel_block_size).
        kernel_block_size = psp_kv_block_size()
        block_size = int(spec.block_size)
        if block_size % kernel_block_size != 0:
            raise ValueError(
                f"PSP-KV block size {block_size} must be a multiple of "
                f"PSP_KV_BLOCK_SIZE={kernel_block_size} (set both the build "
                "define and the runtime env var)."
            )
        total_blocks *= block_size // kernel_block_size
        if envs.PSPKV_DEBUG:
            print(
                f"[bs-debug] cache_spec.block_size={block_size} "
                f"kv_cache_config.num_blocks={kv_cache_config.num_blocks} "
                f"cache_config.block_size={self.cache_config.block_size} "
                f"kernel_block_size={kernel_block_size} total_blocks={total_blocks}",
                flush=True,
            )

        self.kv_cache_config = kv_cache_config
        self.num_blocks = total_blocks
        self.block_size = block_size
        self.kernel_block_size = kernel_block_size
        self.layer_names = self._psp_layer_names(kv_cache_config)
        self.attn_gid = self._psp_attn_gid(kv_cache_config)

        # Per-kernel-block page geometry: the page holds one kernel block's K
        # bytes followed by its V bytes (the split follows the configured
        # default precision). A manager block spans
        # block_size // kernel_block_size kernel blocks.
        blocks_per_page = block_size // kernel_block_size
        self.page_bytes = int(spec.page_size_bytes) // blocks_per_page
        base = kernel_block_size * self.num_kv_heads * self.head_size
        self.k_block_bytes = int(base * self.geom_k_bytes)
        self.v_block_bytes = self.page_bytes - self.k_block_bytes
        if layer_caches:
            self.bind_layer_caches(layer_caches)

        # GPU tables (static, never resized).
        # int8 is a uniform sentinel (K=V=int8), not packable in the 2-bit
        # fields.
        packed_default = (
            4
            if self.precision_default == 4
            else self.psp_kv_default_k | (self.psp_kv_default_v << 2)
        )
        self.precision_table = torch.full(
            (total_blocks,), packed_default, dtype=torch.uint8, device=self.device
        )
        # Per-layer, per-(block, head, scale-group) K/V scales. Each attention
        # layer has its own magnitude distribution, and low-precision formats
        # (fp4 e2m1) need a per-channel-group scale because the K cache spans
        # ~100x across dims.
        # Must match PSPKV_SCALE_GROUP_SIZE in the extension: 16 by default,
        # 32 when built for the m16n8k32 int8 kernel.
        scale_group = int(os.environ.get("PSPKV_SCALE_GROUP", "16"))
        if scale_group not in (16, 32):
            raise ValueError(
                f"PSPKV_SCALE_GROUP={scale_group} is not supported; use 16 or 32."
            )
        if self.head_size % scale_group:
            raise ValueError(
                f"head_size {self.head_size} is not a multiple of the scale group "
                f"{scale_group}."
            )
        self.scale_group = scale_group
        num_groups = self.head_size // scale_group
        self.scale_table = torch.zeros(
            self.num_layers,
            2 * total_blocks,
            self.num_kv_heads,
            num_groups,
            dtype=torch.float32,
            device=self.device,
        )
        self.tma_descriptors = torch.zeros(
            total_blocks, 16, dtype=torch.int64, device=self.device
        )
        self._compute_block_offsets()
        if envs.PSPKV_DEBUG:
            print(
                f"[bs-debug] total_blocks={total_blocks} "
                f"k_cache_bytes={self.k_cache_bytes} "
                f"head_size={self.head_size} num_kv_heads={self.num_kv_heads} "
                f"block_offsets[:4]={self.block_offsets[:4].tolist()}",
                flush=True,
            )

        # Register the variable-size transfer provider with the virthub
        # connector (generic hook; PSP-KV supplies size + sidecar).
        self._register_virthub_transfer_info_provider()

        # Pinned host staging buffers.
        self.h_precision_table = torch.full(
            (total_blocks,), packed_default, dtype=torch.uint8, pin_memory=True
        )
        self.h_scale_table = torch.zeros(
            2 * total_blocks, self.num_kv_heads, dtype=torch.float32, pin_memory=True
        )
        self.h_tma_descriptors = torch.zeros(
            total_blocks, 16, dtype=torch.int64, pin_memory=True
        )

        # Scale validity flag (prevents overwriting GPU scales with stale
        # zeros).
        self.h_has_scale = [False] * total_blocks

        # Batch management with chunking.
        self.max_dirty_blocks = 256
        self.h_dirty_indices = torch.empty(
            self.max_dirty_blocks, dtype=torch.long, pin_memory=True
        )
        self.d_dirty_indices = torch.empty(
            self.max_dirty_blocks, dtype=torch.long, device=self.device
        )
        self.h_dirty_scale_indices = torch.empty(
            self.max_dirty_blocks, dtype=torch.long, pin_memory=True
        )
        self.d_dirty_scale_indices = torch.empty(
            self.max_dirty_blocks, dtype=torch.long, device=self.device
        )
        self.h_prec_batch_buffer = torch.empty(
            self.max_dirty_blocks, dtype=torch.uint8, pin_memory=True
        )
        self.d_prec_batch_buffer = torch.empty(
            self.max_dirty_blocks, dtype=torch.uint8, device=self.device
        )
        self.h_scale_batch_buffer = torch.empty(
            (self.max_dirty_blocks, self.num_kv_heads),
            dtype=torch.float32,
            pin_memory=True,
        )
        self.d_scale_batch_buffer = torch.empty(
            (self.max_dirty_blocks, self.num_kv_heads),
            dtype=torch.float32,
            device=self.device,
        )

        self.dirty_block_ids = []
        self.is_dirty = [False] * total_blocks

        # Prefill scratch buffer with layer dimension.
        self.prefill_kv_scratch = torch.empty(
            (
                self.num_layers,
                self.max_num_tokens,
                2,
                self.num_kv_heads,
                self.head_size,
            ),
            dtype=self.dtype,
            device=self.device,
        )
        # Expose the scratch to the PSP-KV attention impl so the prefill
        # forward can write the projected K/V into it (quantize_and_scatter
        # reads it).
        from vllm.v1.attention.backends.psp_kv_attn import set_prefill_kv_scratch

        set_prefill_kv_scratch(self.prefill_kv_scratch)
        return True

    @staticmethod
    def _psp_attn_gid(kv_cache_config) -> int | None:
        for gid, group in enumerate(kv_cache_config.kv_cache_groups):
            if _psp_specs_for_group(group):
                return gid
        return None

    @staticmethod
    def _psp_layer_names(kv_cache_config) -> list[str]:
        for group in kv_cache_config.kv_cache_groups:
            if _psp_specs_for_group(group):
                return list(group.layer_names)
        return []

    def bind_layer_caches(self, kv_caches: dict[str, torch.Tensor]) -> None:
        """Record the per-layer flat byte caches used by the prefill scatter."""
        from vllm.model_executor.models.utils import extract_layer_index

        self.layer_caches = [
            (extract_layer_index(name), kv_caches[name])
            for name in self.layer_names
            if name in kv_caches and isinstance(kv_caches[name], torch.Tensor)
        ]

    def _compute_block_offsets(self) -> None:
        """Compute the per-block byte offsets into the flat layer cache.

        Each page holds one block's K bytes followed by its V bytes, so the
        offsets are ``b * page`` and ``b * page + k_block``. ``block_offsets``
        stacks K then V, matching the V1 layout of the array.
        """
        num_blocks = self.num_blocks
        block_ids = torch.arange(num_blocks, dtype=torch.int64, device=self.device)
        k_offsets = block_ids * self.page_bytes
        v_offsets = k_offsets + self.k_block_bytes
        self.block_offsets = torch.cat([k_offsets, v_offsets]).to(torch.int32)
        self.k_cache_bytes = num_blocks * self.k_block_bytes
        self.v_cache_bytes = num_blocks * self.v_block_bytes
        base = self.kernel_block_size * self.head_size * self.num_kv_heads
        if num_blocks > 0 and base > 0:
            self.psp_kv_avg_bytes = (
                self.k_cache_bytes + self.v_cache_bytes
            ) / (2 * num_blocks * base)
        else:
            self.psp_kv_avg_bytes = 1.0
        # CPU copies for the virthub connector's variable-size transfer
        # provider and sidecar serialization.
        block_sizes = torch.full(
            (num_blocks,),
            self.k_block_bytes + self.v_block_bytes,
            dtype=torch.int32,
        )
        self.block_sizes_cpu = block_sizes
        self.block_offsets_cpu = self.block_offsets.cpu()
        self.precision_cpu = self.precision_table.to(torch.uint8).cpu()
        self.scale_cpu = self.scale_table.to(torch.float32).cpu()
        self._refresh_dispatch_flags()

    def _refresh_dispatch_flags(self) -> None:
        """Update the CPU dispatch flags from the precision table."""
        # CPU-side dispatch flags (no per-step GPU->CPU sync): the tensor-core
        # paths require uniform lossless, uniform fp8, or the per-K/V combo of
        # uniform fp8 K + uniform fp4 V. The precision table is packed
        # k_fmt | (v_fmt << 2).
        p = self.precision_cpu
        if p is None or p.numel() == 0:
            self.precision_is_lossless = True
            self.precision_fp8_fmt = 0
            self.precision_v_fmt = 0
            self.precision_k_fmt = 0
            self.precision_int_fmt = 0
            self.precision_k_lossless = False
            self.precision_v_lossless = False
            return
        is_int8 = p == 4
        k_fmt = torch.where(
            is_int8, torch.full_like(p, 4), torch.where(p >= 4, p & 3, p)
        )
        v_fmt = torch.where(
            is_int8, torch.full_like(p, 4), torch.where(p >= 4, p >> 2, p)
        )
        self.precision_is_lossless = bool(
            (k_fmt == 0).all().item() and (v_fmt == 0).all().item()
        )
        # The K's uniform fp8 fmt (1=e4m3, 2=e5m2), else 0.
        self.precision_fp8_fmt = (
            int(k_fmt[0].item())
            if ((k_fmt == k_fmt[0]).all().item() and int(k_fmt[0].item()) in (1, 2))
            else 0
        )
        # The V's uniform fmt (3=fp4), else 0.
        self.precision_v_fmt = (
            int(v_fmt[0].item()) if (v_fmt == v_fmt[0]).all().item() else 0
        )
        # The K's uniform fmt (3=fp4), else 0.
        self.precision_k_fmt = (
            int(k_fmt[0].item())
            if ((k_fmt == k_fmt[0]).all().item() and int(k_fmt[0].item()) == 3)
            else 0
        )
        # The uniform integer TC fmt (3=int4, 4=int8), else 0.
        self.precision_int_fmt = (
            int(k_fmt[0].item())
            if (
                (k_fmt == k_fmt[0]).all().item()
                and (v_fmt == v_fmt[0]).all().item()
                and int(k_fmt[0].item()) in (3, 4)
            )
            else 0
        )
        # True when any K block is lossless; the per-block fp8/fp4 TC path can
        # only run when every K block is quantized.
        self.precision_k_lossless = bool((k_fmt == 0).any().item())
        self.precision_v_lossless = bool((v_fmt == 0).any().item())

    # ------------------------------------------------------------------
    # Introspection / control
    # ------------------------------------------------------------------
    def stats(self, num_used_blocks: int = 0) -> dict:
        """Precision mix + decode-dispatch flags + flat-cache byte footprint.

        Reports the K/V per-block format histogram (over the first
        ``num_used_blocks`` physical blocks, or all blocks when 0), the
        uniform-precision dispatch flags, and the flat-cache byte footprints.
        """
        p = self.precision_cpu
        num_blocks = int(p.numel()) if p is not None else 0
        if num_used_blocks > 0:
            p = p[: min(num_used_blocks, num_blocks)]
            num_blocks = int(p.numel())
        stats: dict[str, Any] = {
            "num_blocks": num_blocks,
            "precision_is_lossless": bool(self.precision_is_lossless),
            "precision_fp8_fmt": int(self.precision_fp8_fmt),
            "precision_v_fmt": int(self.precision_v_fmt),
            "precision_k_fmt": int(self.precision_k_fmt),
            "precision_int_fmt": int(self.precision_int_fmt),
            "precision_k_lossless": bool(self.precision_k_lossless),
            "precision_v_lossless": bool(self.precision_v_lossless),
            "k_cache_bytes": int(self.k_cache_bytes),
            "v_cache_bytes": int(self.v_cache_bytes),
            "psp_kv_avg_bytes": float(self.psp_kv_avg_bytes),
            "k_fmt_counts": {},
            "v_fmt_counts": {},
        }
        if num_blocks > 0:
            # Mirror the kernel-side accessors: int8 is the out-of-band sentinel
            # 4 and means uniform K=V=int8, so it must not be unpacked.
            pp = p.to(torch.int64)
            is_int8 = pp == 4
            k_fmt = torch.where(
                is_int8, torch.full_like(pp, 4), torch.where(pp >= 4, pp & 3, pp)
            )
            v_fmt = torch.where(
                is_int8, torch.full_like(pp, 4), torch.where(pp >= 4, pp >> 2, pp)
            )
            for f in (0, 1, 2, 3, 4):
                kc = int((k_fmt == f).sum().item())
                vc = int((v_fmt == f).sum().item())
                if kc:
                    stats["k_fmt_counts"][str(f)] = kc
                if vc:
                    stats["v_fmt_counts"][str(f)] = vc
        return stats

    def _auto_num_splits(self, num_reqs: int) -> int:
        """Split-K count for this step.

        Small batches have few CTAs, so more splits fill the SMs; larger batches
        already saturate and pay extra partials/merge traffic.
        """
        # Measured on Qwen3-4B: raising the count to 32 at batch 1 is a
        # consistent win (112 vs 89-110 tok/s across counts); at batch >= 3 the
        # partials/merge traffic dominates and 16 stays best.
        return 32 if num_reqs <= 2 else 16

    def _update_num_splits(self, num_reqs: int) -> None:
        if self._manual_num_splits or self.num_split_override is not None:
            return
        target = self._auto_num_splits(num_reqs)
        if target != self._auto_num_splits_applied:
            self.set_num_splits(target)
            self._auto_num_splits_applied = target

    def set_num_splits(self, num_splits: int) -> None:
        """Runtime override of the tensor-core split-K count (autotune).

        The C++ wrapper reads ``PSPKV_TC_NUM_SPLITS`` via ``getenv`` on every
        decode call, so writing it here (inside the engine-core process) takes
        effect for subsequent decodes without a reload.
        """
        os.environ["PSPKV_TC_NUM_SPLITS"] = str(int(num_splits))

    # ------------------------------------------------------------------
    # Metadata plumbing
    # ------------------------------------------------------------------
    def get_extra_common_attn_kwargs(self, group_idx: int, num_reqs: int) -> dict:
        """PSP-KV fields for ``CommonAttentionMetadata`` of group ``group_idx``.

        Returns an empty dict for non-PSP-KV groups (and before the tables
        exist), so the extra-kwargs hook stays PSP-KV only.
        """
        if self.attn_gid is None or group_idx != self.attn_gid:
            return {}
        if self.precision_table is None:
            return {}
        # Flush pending block-metadata updates before the metadata is built
        # (no-op while a CUDA graph is being captured).
        self._sync_dirty_blocks()
        self._update_num_splits(num_reqs)
        return {
            "precision_table": self.precision_table,
            "scale_table": self.scale_table,
            "tma_descriptors": (
                self.tma_descriptors if self.format_generation == 2 else None
            ),
            "block_offsets": self.block_offsets,
            "format_generation": self.format_generation,
            "precision_is_lossless": self.precision_is_lossless,
            "precision_fp8_fmt": self.precision_fp8_fmt,
            "precision_v_fmt": self.precision_v_fmt,
            "precision_k_fmt": self.precision_k_fmt,
            "precision_int_fmt": self.precision_int_fmt,
            "precision_k_lossless": self.precision_k_lossless,
            "precision_v_lossless": self.precision_v_lossless,
            "k_cache_bytes": self.k_cache_bytes,
            "v_cache_bytes": self.v_cache_bytes,
        }

    # ------------------------------------------------------------------
    # Prefill quantization (quantize-on-write)
    # ------------------------------------------------------------------
    def quantize_prefill(
        self,
        block_table: torch.Tensor,
        slot_mapping: torch.Tensor,
        num_tokens: int,
        has_prefill: bool,
    ) -> None:
        """Scatter this step's prefill K/V into the PSP-KV byte caches.

        The PSP-KV kernels are decode-only; prefill attention runs through
        FlashAttention, and the projected K/V written into the scratch buffer
        by the attention impl are quantized into the cache here.
        """
        scratch = self.prefill_kv_scratch
        if scratch is None or self.precision_table is None or num_tokens <= 0:
            return
        if not self.layer_caches:
            return

        prof = self.psp_kv_profile
        if prof:
            ev_derive = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            ev_quant = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            ev_derive[0].record()

        # Derive per-block precision from the prefill magnitude (only when
        # adaptive precision is enabled).
        if has_prefill and self.psp_kv_adaptive_precision:
            torch.ops.psp_kv.derive_precision_kv(
                scratch[0],
                slot_mapping,
                self.precision_table,
                self.scale_table[0],
                num_tokens,
                self.psp_kv_default_k,
                self.psp_kv_default_v,
                self.psp_kv_adaptive_e,
                self.psp_kv_k_cap_fmt,
            )
            self.precision_cpu = self.precision_table.to(torch.uint8).cpu()
            self._refresh_dispatch_flags()
        if prof:
            ev_derive[1].record()
            ev_quant[0].record()

        from vllm.v1.attention.backends.psp_kv_attn import psp_kv_flat_byte_caches

        for layer_idx, layer_kv_cache in self.layer_caches:
            k_cache, v_cache = psp_kv_flat_byte_caches(layer_kv_cache)
            # The verify (decode-like, no prefill) uses the scatter-only op so
            # the accepted blocks' scale is preserved.
            if has_prefill:
                if envs.PSPKV_DEBUG:
                    print(
                        f"[quant-debug] num_prefill_tokens={num_tokens} "
                        f"slot_map[:3]={slot_mapping[:3].tolist()} "
                        f"block_offsets[:3]={self.block_offsets[:3].tolist()}",
                        flush=True,
                    )
                torch.ops.psp_kv.quantize_and_scatter_kv(
                    scratch[layer_idx],
                    k_cache,
                    v_cache,
                    block_table,
                    slot_mapping,
                    self.precision_table,
                    self.block_offsets,
                    self.scale_table[layer_idx],
                    num_tokens,
                    self.format_generation,
                )
            else:
                torch.ops.psp_kv.quantize_scatter_multi(
                    scratch[layer_idx],
                    k_cache,
                    v_cache,
                    slot_mapping,
                    self.precision_table,
                    self.block_offsets,
                    self.scale_table[layer_idx],
                    num_tokens,
                    self.format_generation,
                )

        if prof:
            ev_quant[1].record()
            torch.cuda.synchronize()
            logger.warning(
                "[PSP-KV profile] prefill=%d tok derive=%.3fms quantize=%.3fms",
                num_tokens,
                ev_derive[0].elapsed_time(ev_derive[1]),
                ev_quant[0].elapsed_time(ev_quant[1]),
            )

    # ------------------------------------------------------------------
    # Virthub connector hooks (host-side block metadata)
    # ------------------------------------------------------------------
    def update_block_metadata(
        self, block_id: int, precision_level: int, scale: list[float] | None
    ) -> None:
        """Called by the connector when a block is allocated or changed."""
        if self.precision_table is None:
            return  # PSP-KV not initialized
        self.h_precision_table[block_id] = precision_level
        if scale is not None:
            self.h_scale_table[block_id] = torch.as_tensor(
                scale, dtype=torch.float32
            )
            self.h_has_scale[block_id] = True
        else:
            self.h_has_scale[block_id] = False

        if self.format_generation == 2:
            self._update_tma_descriptor_host(block_id)

        # Track dirty only once (prevent duplicates).
        if not self.is_dirty[block_id]:
            self.is_dirty[block_id] = True
            self.dirty_block_ids.append(block_id)

        # Flush if chunk is full.
        if len(self.dirty_block_ids) >= self.max_dirty_blocks:
            self._sync_dirty_blocks()

    def _update_tma_descriptor_host(self, block_id: int) -> None:
        """Generate a TMA descriptor for the given block (placeholder)."""
        # This function would call a CUDA helper to encode a CUtensorMap and
        # store it in self.h_tma_descriptors[block_id]. The reference BT-KV
        # kernel does not consume the descriptors yet.
        pass

    def _sync_dirty_blocks(self) -> None:
        """Flush all pending block metadata updates to the GPU in chunks."""
        if not self.dirty_block_ids:
            return

        # Guard against being called during CUDA graph capture.
        # Dynamic index_copy_ operations cannot be captured into a graph.
        if torch.cuda.is_current_stream_capturing():
            return

        current_stream = torch.cuda.current_stream(device=self.device)
        with torch.cuda.stream(current_stream):
            # Process dirty blocks in chunks to avoid buffer overflow.
            for i in range(0, len(self.dirty_block_ids), self.max_dirty_blocks):
                chunk = self.dirty_block_ids[i : i + self.max_dirty_blocks]
                k = len(chunk)

                # 1. Copy block IDs to pinned host buffer.
                self.h_dirty_indices[:k] = torch.as_tensor(chunk, dtype=torch.long)

                # 2. Async copy indices to GPU.
                self.d_dirty_indices[:k].copy_(
                    self.h_dirty_indices[:k], non_blocking=True
                )
                active_indices = self.d_dirty_indices[:k]

                # 3. Gather precision rows into the pinned buffer (async).
                torch.index_select(
                    self.h_precision_table,
                    0,
                    self.h_dirty_indices[:k],
                    out=self.h_prec_batch_buffer[:k],
                )
                self.d_prec_batch_buffer[:k].copy_(
                    self.h_prec_batch_buffer[:k], non_blocking=True
                )
                self.precision_table.index_copy_(
                    0, active_indices, self.d_prec_batch_buffer[:k]
                )

                # 4. Gather scale rows ONLY for blocks where host provided a
                # scale. This prevents overwriting GPU scales (set by
                # quantize_and_scatter_kv).
                scale_indices = [b for b in chunk if self.h_has_scale[b]]
                if scale_indices:
                    n_scale = len(scale_indices)
                    self.h_dirty_scale_indices[:n_scale] = torch.as_tensor(
                        scale_indices, dtype=torch.long
                    )
                    self.d_dirty_scale_indices[:n_scale].copy_(
                        self.h_dirty_scale_indices[:n_scale], non_blocking=True
                    )
                    d_s_indices = self.d_dirty_scale_indices[:n_scale]

                    torch.index_select(
                        self.h_scale_table,
                        0,
                        self.h_dirty_scale_indices[:n_scale],
                        out=self.h_scale_batch_buffer[:n_scale],
                    )
                    self.d_scale_batch_buffer[:n_scale].copy_(
                        self.h_scale_batch_buffer[:n_scale], non_blocking=True
                    )
                    self.scale_table.index_copy_(
                        0, d_s_indices, self.d_scale_batch_buffer[:n_scale]
                    )

                # 5. TMA descriptors for BT-KV.
                if self.format_generation == 2:
                    h_tma_batch = self.h_tma_descriptors[
                        self.h_dirty_indices[:k]
                    ].to(self.device, non_blocking=True)
                    self.tma_descriptors.index_copy_(
                        0, active_indices, h_tma_batch
                    )

            # Clear dirty flags and scale validity (only after all chunks
            # complete).
            for b_id in self.dirty_block_ids:
                self.is_dirty[b_id] = False
                self.h_has_scale[b_id] = False
            self.dirty_block_ids.clear()

            # Precision may have changed per-block, so refresh the CPU copies.
            self.precision_cpu = self.precision_table.to(torch.uint8).cpu()
            self._refresh_dispatch_flags()

    def _psp_kv_block_transfer_info(self, block_id):
        """Backend hook for the virthub connector."""
        from virthub import KVBlockTransferInfo  # type: ignore[import-not-found]

        from vllm.v1.attention.psp_kv_virthub import serialize_sidecar

        size_bytes = 0
        sizes = self.block_sizes_cpu
        if sizes is not None and 0 <= block_id < len(sizes):
            size_bytes = int(sizes[block_id].item())

        metadata = b""
        precisions = self.precision_cpu
        if precisions is not None and 0 <= block_id < len(precisions):
            base_precision = int(precisions[block_id].item())
            per_head = []
            scales = self.scale_cpu
            if scales is not None and 0 <= block_id < len(scales):
                per_head = scales[block_id].tolist()
            base_ptr = 0
            offsets = self.block_offsets_cpu
            if offsets is not None and 0 <= block_id < len(offsets):
                base_ptr = int(offsets[block_id].item())
            metadata = serialize_sidecar(base_precision, per_head, base_ptr)
        return KVBlockTransferInfo(size_bytes=size_bytes, metadata=metadata)

    def _register_virthub_transfer_info_provider(self) -> None:
        """Register the PSP-KV variable-size provider with the connector."""
        from vllm.distributed.kv_transfer import (
            get_kv_transfer_group,
            has_kv_transfer_group,
        )

        if not has_kv_transfer_group():
            return
        connector = get_kv_transfer_group()
        setter = getattr(connector, "set_block_transfer_info_provider", None)
        if callable(setter):
            setter(self._psp_kv_block_transfer_info)


def _psp_specs_for_group(group) -> list:
    from vllm.v1.kv_cache_interface import (
        PSPKVAttentionSpec,
        PSPKVSlidingWindowSpec,
        UniformTypeKVCacheSpecs,
    )

    spec = group.kv_cache_spec
    if isinstance(spec, UniformTypeKVCacheSpecs):
        return [
            layer_spec
            for layer_spec in spec.kv_cache_specs.values()
            if isinstance(layer_spec, (PSPKVAttentionSpec, PSPKVSlidingWindowSpec))
        ]
    return [
        s
        for s in (spec,)
        if isinstance(s, (PSPKVAttentionSpec, PSPKVSlidingWindowSpec))
    ]

