# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Correctness-first PSP-KV attention backend for vLLM-Ascend v0.20.2rc1.

The plugin still owns model execution, metadata creation and physical KV
allocation. This backend replaces its dense attention/cache operations with
the PSP-KV FP8 byte format. E4M3 is the default; the reference policy can
promote individual physical blocks to E5M2 based on incoming K/V values.
An opt-in uniform FP4 reference path allocates half-size pages. Mixed-size
live pages and fused NPU kernels remain future work.
"""

import math
import os
import weakref

import torch
from vllm_ascend.attention.attention_v1 import (
    AscendAttentionBackend,
    AscendAttentionBackendImpl,
    AscendMetadata,
)

from vllm.logger import init_logger
from vllm.v1.attention.backend import AttentionLayer, AttentionType
from vllm.v1.attention.backends.psp_kv_ascend_arena import (
    VariableArenaKVCache,
    reference_variable_attention,
)
from vllm.v1.attention.backends.psp_kv_ascend_diagnostics import FP4LayerDiagnostics
from vllm.v1.attention.backends.psp_kv_ascend_native import load_native_fp8_ops
from vllm.v1.attention.backends.psp_kv_ascend_packed import (
    PackedKVCache,
    reference_packed_attention,
)
from vllm.v1.attention.backends.psp_kv_ascend_reference import reference_attention

logger = init_logger(__name__)

# The plugin's copy_blocks callback receives cache tensors, not backend
# instances. Keep only weak references to per-layer tables and look them up by
# the K cache base pointer when a physical page is copied.
_REFERENCE_PRECISION_TABLES: weakref.WeakValueDictionary[int, torch.Tensor] = (
    weakref.WeakValueDictionary()
)
_REFERENCE_FP4_SCALES: weakref.WeakValueDictionary[int, torch.Tensor] = (
    weakref.WeakValueDictionary()
)
# The plugin's copy_blocks callback receives cache tensors, not backend
# instances. Keep weak references to per-layer variable arenas and look them
# up by the K cache base pointer when a physical page is copied.
_VARIABLE_ARENAS: weakref.WeakValueDictionary[int, VariableArenaKVCache] = (
    weakref.WeakValueDictionary()
)


class PSPKVAscendImpl(AscendAttentionBackendImpl):
    """Eager reference attention with FP8 or opt-in packed FP4 cache entries."""

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        sinks: torch.Tensor | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            alibi_slopes,
            sliding_window,
            kv_cache_dtype,
            logits_soft_cap,
            attn_type,
            kv_sharing_target_layer_name,
            sinks=sinks,
            **kwargs,
        )
        if self.kv_cache_dtype != "fp8":
            raise ValueError(
                "PSP-KV Ascend reference backend requires --kv-cache-dtype fp8"
            )
        if self.attn_type != AttentionType.DECODER:
            raise NotImplementedError("PSP-KV Ascend supports decoder attention only")
        if self.sliding_window is not None or self.alibi_slopes is not None:
            raise NotImplementedError(
                "PSP-KV Ascend does not support sliding window or ALiBi"
            )
        if logits_soft_cap is not None or sinks is not None:
            raise NotImplementedError(
                "PSP-KV Ascend does not support soft cap or attention sinks"
            )
        if kv_sharing_target_layer_name is not None:
            raise NotImplementedError(
                "PSP-KV Ascend does not support cross-layer KV sharing"
            )
        config = self.vllm_config
        if not config.model_config.enforce_eager:
            raise ValueError("PSP-KV Ascend reference backend requires --enforce-eager")
        if config.kv_transfer_config is not None:
            raise NotImplementedError("PSP-KV Ascend does not support KV transfer")
        if config.parallel_config.tensor_parallel_size != 1:
            raise NotImplementedError(
                "PSP-KV Ascend reference backend supports TP=1 only"
            )
        self.reference_fp8 = os.environ.get(
            "PSPKV_ASCEND_REFERENCE_FP8", "e4m3"
        ).lower()
        if self.reference_fp8 not in ("e4m3", "e5m2", "alternating", "adaptive"):
            raise ValueError(
                "PSPKV_ASCEND_REFERENCE_FP8 must be e4m3, e5m2, alternating, "
                "or adaptive"
            )
        self.adaptive_e4m3_max_abs: float | None = None
        if self.reference_fp8 == "adaptive":
            self.adaptive_e4m3_max_abs = float(
                os.environ.get("PSPKV_ASCEND_E4M3_MAX_ABS", "240")
            )
            if not 0 < self.adaptive_e4m3_max_abs <= 240 or not math.isfinite(
                self.adaptive_e4m3_max_abs
            ):
                raise ValueError("PSPKV_ASCEND_E4M3_MAX_ABS must be in (0, 240]")
        self.log_precision_transitions = (
            os.environ.get("PSPKV_ASCEND_LOG_PRECISION", "0") == "1"
        )
        self.reference_precision_table: torch.Tensor | None = None
        self.active_precision_table: torch.Tensor | None = None
        self.uniform_fp4 = os.environ.get("PSPKV_ASCEND_UNIFORM_FP4", "0") == "1"
        self.fp4_group_size = int(os.environ.get("PSPKV_ASCEND_FP4_GROUP_SIZE", "0"))
        self.native_fp4 = os.environ.get("PSPKV_ASCEND_NATIVE_FP4", "0") == "1"
        self.fp4_sdpa = os.environ.get("PSPKV_ASCEND_FP4_SDPA", "0") == "1"
        self.fp4_diagnostics = (
            FP4LayerDiagnostics(
                int(os.environ.get("PSPKV_ASCEND_FP4_DIAGNOSTICS_STEPS", "2"))
            )
            if os.environ.get("PSPKV_ASCEND_FP4_DIAGNOSTICS", "0") == "1"
            else None
        )
        self.native_fp8 = os.environ.get("PSPKV_ASCEND_NATIVE_FP8", "0") == "1"
        self.native_decode = os.environ.get("PSPKV_ASCEND_NATIVE_DECODE", "0") == "1"
        self.sdpa_attention = os.environ.get("PSPKV_ASCEND_SDPA_ATTENTION", "0") == "1"
        self.variable_precision = (
            os.environ.get("PSPKV_ASCEND_VARIABLE_PRECISION", "0") == "1"
        )
        self.variable_avg_bytes = float(os.environ.get("PSPKV_AVG_BYTES", "1.0"))
        self.variable_max_rmse = float(
            os.environ.get("PSPKV_ASCEND_MAX_RELATIVE_RMSE", "0.05")
        )
        self.variable_log_format = (
            os.environ.get("PSPKV_ASCEND_LOG_FORMAT_COUNTS", "0") == "1"
        )
        self.native_variable = (
            os.environ.get("PSPKV_ASCEND_NATIVE_VARIABLE", "0") == "1"
        )
        self.native_fused = os.environ.get("PSPKV_ASCEND_NATIVE_FUSED", "0") == "1"
        if self.native_fused and not self.native_variable:
            raise ValueError(
                "PSPKV_ASCEND_NATIVE_FUSED=1 requires PSPKV_ASCEND_NATIVE_VARIABLE=1"
            )
        self.calibrate = os.environ.get("PSPKV_ASCEND_CALIBRATE", "0") == "1"
        self.static_precision: dict[str, dict] = {}
        static_precision_path = os.environ.get("PSPKV_ASCEND_STATIC_PRECISION", "")
        if static_precision_path:
            if not self.variable_precision:
                raise ValueError(
                    "PSPKV_ASCEND_STATIC_PRECISION requires "
                    "PSPKV_ASCEND_VARIABLE_PRECISION=1"
                )
            import json

            with open(static_precision_path, encoding="utf-8") as handle:
                static_precision = json.load(handle)
            if not isinstance(static_precision, dict):
                raise ValueError("Static precision map must be a JSON object")
            for name, entry in static_precision.items():
                k_code = entry.get("k_code", entry.get("code"))
                v_code = entry.get("v_code", k_code)
                if k_code not in (1, 2, 3) or v_code not in (1, 2, 3):
                    raise ValueError(
                        f"Static precision for {name} must use codes 1, 2 or 3"
                    )
                groups = int(entry.get("groups", 1))
                if 3 in (k_code, v_code) and (groups < 1 or self.head_size % groups):
                    raise ValueError(
                        f"Static FP4 {name} group count must divide the head size"
                    )
                for stream, stream_code in (
                    ("k_scales", k_code),
                    ("v_scales", v_code),
                ):
                    if stream_code != 3:
                        continue
                    scales = entry.get(stream)
                    ok = isinstance(scales, list) and len(scales) == self.num_kv_heads
                    if ok and groups == 1:
                        ok = all(
                            isinstance(value, (int, float)) and value > 0
                            for value in scales
                        )
                    if ok and groups > 1:
                        ok = all(
                            isinstance(row, list)
                            and len(row) == groups
                            and all(
                                isinstance(value, (int, float)) and value > 0
                                for value in row
                            )
                            for row in scales
                        )
                    if not ok:
                        raise ValueError(
                            f"Static FP4 {name} needs positive {stream} "
                            f"for {self.num_kv_heads} KV heads"
                            + (f" and {groups} groups" if groups > 1 else "")
                        )
            self.static_precision = static_precision
            logger.info_once(
                "PSP-KV static allocation-time precision map loaded from %s "
                "for %d layers",
                static_precision_path,
                len(static_precision),
            )
        if self.calibrate and self.static_precision:
            raise ValueError(
                "PSPKV_ASCEND_CALIBRATE samples the dynamic policy and cannot "
                "run with a static precision map"
            )
        if self.variable_precision:
            if self.uniform_fp4:
                raise ValueError(
                    "PSP-KV variable precision and uniform FP4 are exclusive"
                )
            if self.reference_fp8 != "e4m3":
                raise ValueError("PSP-KV variable precision requires the e4m3 FP8 mode")
            if self.native_fp8 or self.native_fp4 or self.native_decode:
                raise NotImplementedError(
                    "PSP-KV variable precision has no native NPU kernels yet"
                )
            if self.sdpa_attention:
                raise NotImplementedError(
                    "PSP-KV variable precision uses the eager bmm attention path"
                )
            if config.cache_config.enable_prefix_caching:
                raise NotImplementedError(
                    "PSP-KV variable precision requires prefix caching disabled"
                )
            if config.scheduler_config.max_num_seqs != 1:
                raise NotImplementedError(
                    "PSP-KV variable precision supports one sequence at a time"
                )
            if not 0 < self.variable_avg_bytes <= 1.0:
                raise ValueError("PSPKV_AVG_BYTES must be in (0, 1]")
            if not math.isfinite(self.variable_max_rmse) or self.variable_max_rmse < 0:
                raise ValueError(
                    "PSPKV_ASCEND_MAX_RELATIVE_RMSE must be finite and nonnegative"
                )
            if self.fp4_group_size < 0 or (
                self.fp4_group_size and self.head_size % self.fp4_group_size
            ):
                raise ValueError(
                    "PSPKV_ASCEND_FP4_GROUP_SIZE must be zero or divide head size"
                )
            if config.model_config.dtype not in (torch.bfloat16, torch.float16):
                raise ValueError(
                    "PSP-KV variable precision requires BF16 or FP16 model K/V"
                )
            logger.info_once(
                "PSP-KV Ascend variable-width arena is active: %s bytes per "
                "block per K/V element budget, completed-block policy with "
                "max relative RMSE %s",
                self.variable_avg_bytes,
                self.variable_max_rmse,
            )
            if self.native_variable:
                native_ops = load_native_fp8_ops()
                if not all(
                    hasattr(native_ops, name)
                    for name in (
                        "packed_scatter",
                        "packed_scores",
                        "packed_weighted_value",
                    )
                ):
                    raise RuntimeError(
                        "PSP-KV native variable width requires a rebuilt "
                        "Ascend library with the packed kernels"
                    )
                if self.native_fused and not hasattr(native_ops, "packed_fused_decode"):
                    raise RuntimeError(
                        "PSP-KV fused decode requires a rebuilt Ascend library"
                    )
                logger.info_once(
                    "PSP-KV Ascend native variable path is active: packed "
                    "scatter and direct compressed-cache single-token "
                    "attention with per-block offsets and formats"
                )
        elif self.native_variable:
            raise ValueError(
                "PSPKV_ASCEND_NATIVE_VARIABLE=1 requires "
                "PSPKV_ASCEND_VARIABLE_PRECISION=1"
            )
        self.variable_arena: VariableArenaKVCache | None = None
        if self.sdpa_attention:
            logger.info_once(
                "PSP-KV Ascend SDPA attention dispatch is active; torch_npu may "
                "use its fused attention implementation"
            )
        if self.native_fp8:
            if self.uniform_fp4:
                raise ValueError("PSP-KV native FP8 and uniform FP4 are exclusive")
            if config.model_config.dtype != torch.bfloat16:
                raise ValueError("PSP-KV native FP8 currently requires BF16 model K/V")
            native_ops = load_native_fp8_ops()
            if self.native_decode and (
                not hasattr(native_ops, "decode_scores_bf16")
                or not hasattr(native_ops, "decode_weighted_value")
            ):
                raise RuntimeError(
                    "PSPKV_ASCEND_NATIVE_DECODE requires a rebuilt native library"
                )
            logger.info_once(
                "PSP-KV Ascend native FP8 cache scatter/decode is active; "
                "single-token attention uses the native compressed-cache "
                "path when PSPKV_ASCEND_NATIVE_DECODE=1"
            )
        if self.native_decode and not self.native_fp8:
            raise ValueError("PSPKV_ASCEND_NATIVE_DECODE=1 requires native FP8")
        if self.uniform_fp4:
            if self.reference_fp8 != "e4m3":
                raise ValueError("PSP-KV uniform FP4 cannot use an FP8 reference mode")
            if config.cache_config.enable_prefix_caching:
                raise NotImplementedError(
                    "PSP-KV uniform FP4 currently requires prefix caching disabled"
                )
            if config.scheduler_config.max_num_seqs != 1:
                raise NotImplementedError(
                    "PSP-KV uniform FP4 currently supports one sequence at a time"
                )
            if self.fp4_group_size < 0 or (
                self.fp4_group_size and self.head_size % self.fp4_group_size
            ):
                raise ValueError(
                    "PSPKV_ASCEND_FP4_GROUP_SIZE must be zero or divide head size"
                )
        elif self.fp4_group_size:
            raise ValueError("Grouped FP4 scales require uniform FP4 serving")
        if self.native_fp4:
            if not self.uniform_fp4:
                raise ValueError("Native FP4 requires PSPKV_ASCEND_UNIFORM_FP4=1")
            if config.model_config.dtype != torch.bfloat16:
                raise ValueError("Native FP4 currently requires BF16 model K/V")
            native_ops = load_native_fp8_ops()
            if not all(
                hasattr(native_ops, name)
                for name in ("packed_scatter", "packed_scores", "packed_weighted_value")
            ):
                raise RuntimeError("Native FP4 requires a rebuilt Ascend library")
            logger.info_once(
                "PSP-KV Ascend native uniform FP4 scatter and decode are active"
            )
        if self.fp4_sdpa:
            if not self.uniform_fp4:
                raise ValueError("FP4 SDPA requires PSPKV_ASCEND_UNIFORM_FP4=1")
            logger.info_once(
                "PSP-KV Ascend FP4 reference attention uses torch SDPA after "
                "decoding the packed cache"
            )
        if self.fp4_diagnostics is not None:
            if not self.uniform_fp4:
                raise ValueError("FP4 diagnostics requires uniform FP4 serving")
            logger.warning_once(
                "PSP-KV FP4 layer diagnostics are active; do not use this "
                "request for performance measurements"
            )
        self.packed_cache: PackedKVCache | None = None
        self.packed_scales: torch.Tensor | None = None

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: tuple[torch.Tensor, torch.Tensor],
        attn_metadata: AscendMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if output is None:
            raise ValueError("PSP-KV Ascend requires an output buffer")
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "PSP-KV Ascend does not support fused output quantization"
            )
        if attn_metadata is None:
            return output.fill_(0)
        if not attn_metadata.causal:
            raise NotImplementedError("PSP-KV Ascend requires causal attention")
        if layer._k_scale_float != 1.0 or layer._v_scale_float != 1.0:
            raise NotImplementedError("PSP-KV Ascend requires unit KV scales")
        if key is None or value is None or kv_cache is None or len(kv_cache) != 2:
            raise ValueError(
                "PSP-KV Ascend requires projected K/V and separate K/V caches"
            )

        k_cache, v_cache = kv_cache
        if self.variable_precision:
            if k_cache.ndim != 1 or v_cache.ndim != 1:
                raise ValueError(
                    "PSP-KV variable width requires flat uint8 byte arenas"
                )
            from vllm.v1.kv_cache_interface import _psp_kv_avg_bytes

            k_avg = _psp_kv_avg_bytes("k")
            v_avg = _psp_kv_avg_bytes("v")
            expected_k = 128 * self.num_kv_heads * self.head_size * k_avg
            expected_v = 128 * self.num_kv_heads * self.head_size * v_avg
            if (
                self.variable_arena is None
                or self.variable_arena.k_buffer.data_ptr() != k_cache.data_ptr()
                or self.variable_arena.k_buffer.numel() != k_cache.numel()
                or self.variable_arena.v_buffer.numel() != v_cache.numel()
            ):
                if (
                    not math.isfinite(expected_k)
                    or not math.isfinite(expected_v)
                    or expected_k <= 0
                    or expected_v <= 0
                    or k_cache.numel() % expected_k
                    or v_cache.numel() % expected_v
                    or k_cache.numel() // expected_k != v_cache.numel() // expected_v
                ):
                    raise ValueError(
                        "PSP-KV variable arena sizes are incompatible with the "
                        f"physical caches ({k_cache.numel()} K bytes, "
                        f"{v_cache.numel()} V bytes)"
                    )
                num_blocks = round(k_cache.numel() / expected_k)
                static_entry = self.static_precision.get(layer.layer_name)
                static_code = None
                static_v_code = None
                static_scales = None
                if static_entry is not None:
                    static_code = int(
                        static_entry.get("k_code", static_entry.get("code", 1))
                    )
                    static_v_code = int(static_entry.get("v_code", static_code))
                if static_entry is not None and 3 in (static_code, static_v_code):
                    # Each FP4 stream gets its own power-of-two scales
                    # (optionally per dimension group) plus one doubling of
                    # headroom. Non-FP4 streams keep a placeholder row.
                    groups = int(static_entry.get("groups", 1))
                    stream_scales = []
                    for stream, stream_code in (
                        ("k_scales", static_code),
                        ("v_scales", static_v_code),
                    ):
                        if stream_code == 3:
                            scales = torch.tensor(
                                static_entry[stream],
                                dtype=torch.float32,
                                device=k_cache.device,
                            ).reshape(self.num_kv_heads, groups)
                        else:
                            scales = torch.ones(
                                (self.num_kv_heads, groups),
                                dtype=torch.float32,
                                device=k_cache.device,
                            )
                        stream_scales.append(scales * 2.0)
                    static_scales = torch.stack(stream_scales, dim=0)
                elif static_entry is None and self.static_precision:
                    logger.warning_once(
                        "PSP-KV layer %s is missing from the static precision "
                        "map; falling back to the completed-block policy",
                        layer.layer_name,
                    )
                self.variable_arena = VariableArenaKVCache(
                    num_blocks,
                    128,
                    self.num_kv_heads,
                    self.head_size,
                    k_cache.device,
                    k_buffer=k_cache,
                    v_buffer=v_cache,
                    avg_bytes=k_avg,
                    v_avg_bytes=v_avg,
                    scalar_dtype=key.dtype,
                    fp4_group_size=self.fp4_group_size,
                    max_relative_rmse=self.variable_max_rmse,
                    layer_name=layer.layer_name,
                    static_code=static_code,
                    static_scales=static_scales,
                    calibrate=self.calibrate,
                    static_v_code=static_v_code,
                )
            arena = self.variable_arena
            assert arena is not None
            _VARIABLE_ARENAS[k_cache.data_ptr()] = arena
            actual = attn_metadata.num_actual_tokens
            result = reference_variable_attention(
                query,
                key,
                value,
                arena,
                attn_metadata.slot_mapping[:actual],
                attn_metadata.block_tables,
                attn_metadata.actual_seq_lengths_q,
                attn_metadata.seq_lens_list,
                output,
                self.scale,
                native=self.native_variable,
                native_fused=self.native_fused,
            )
            if self.variable_log_format and actual:
                logger.info(
                    "PSP-KV variable arena %s: formats %s, free %d/%d bytes",
                    layer.layer_name,
                    arena.format_counts(),
                    arena.arena_size - arena.free_ptr,
                    arena.arena_size,
                )
            return result
        if k_cache.shape[1] != 128:
            raise ValueError(
                "PSP-KV Ascend v0.20.2rc1 metadata requires 128-token blocks"
            )
        physical_head_size = self.head_size // 2 if self.uniform_fp4 else self.head_size
        if k_cache.shape[2:] != (self.num_kv_heads, physical_head_size):
            raise ValueError(
                "PSP-KV Ascend cache head dimensions do not match the model"
            )
        if self.uniform_fp4:
            if (
                self.packed_cache is None
                or self.packed_cache.k_buffer.data_ptr() != k_cache.data_ptr()
            ):
                self.packed_cache = PackedKVCache(
                    [3] * k_cache.shape[0],
                    128,
                    self.num_kv_heads,
                    self.head_size,
                    k_cache.device,
                    k_buffer=k_cache.reshape(-1),
                    v_buffer=v_cache.reshape(-1),
                    value_layout="token_major",
                    validate_layout=False,
                )
                scale_shape = (
                    (
                        k_cache.shape[0],
                        2,
                        self.num_kv_heads,
                        self.head_size // self.fp4_group_size,
                    )
                    if self.fp4_group_size
                    else (k_cache.shape[0], self.num_kv_heads)
                )
                self.packed_scales = torch.ones(
                    scale_shape,
                    dtype=torch.float32,
                    device=k_cache.device,
                )
            assert self.packed_scales is not None
            _REFERENCE_FP4_SCALES[k_cache.data_ptr()] = self.packed_scales
            actual = attn_metadata.num_actual_tokens
            result = reference_packed_attention(
                query,
                key,
                value,
                self.packed_cache,
                attn_metadata.slot_mapping[:actual],
                attn_metadata.block_tables,
                attn_metadata.actual_seq_lengths_q,
                attn_metadata.seq_lens_list,
                output,
                self.scale,
                self.packed_scales,
                native_fp4=self.native_fp4,
                use_sdpa=self.fp4_sdpa,
            )
            if self.fp4_diagnostics is not None and actual:
                if len(attn_metadata.seq_lens_list) != 1:
                    raise ValueError("FP4 diagnostics supports one request at a time")
                comparison = self.fp4_diagnostics.observe(
                    query[:actual],
                    key[:actual],
                    value[:actual],
                    output.view(-1, query.shape[1], self.head_size)[:actual],
                    self.packed_cache,
                    self.packed_scales,
                    attn_metadata.block_tables[0],
                    attn_metadata.seq_lens_list[0],
                    self.scale,
                )
                if comparison is not None:
                    logger.info(
                        "PSP-KV FP4 layer %s seq=%d K_RMSE=%.5f "
                        "K_FP8_RMSE=%.5f V_RMSE=%.5f V_FP8_RMSE=%.5f "
                        "ATTN_RMSE=%.5f K_ZERO=%.4f V_ZERO=%.4f",
                        layer.layer_name,
                        comparison.sequence_length,
                        comparison.fp4_key_rmse,
                        comparison.fp8_key_rmse,
                        comparison.fp4_value_rmse,
                        comparison.fp8_value_rmse,
                        comparison.attention_rmse,
                        comparison.fp4_key_zero_fraction,
                        comparison.fp4_value_zero_fraction,
                    )
            return result
        precision_table = getattr(attn_metadata, "psp_kv_precision_table", None)
        if precision_table is None and (
            self.reference_fp8 != "e4m3" or self.native_fp8
        ):
            if (
                self.reference_precision_table is None
                or self.reference_precision_table.shape[0] != k_cache.shape[0]
                or self.reference_precision_table.device != k_cache.device
            ):
                if self.reference_fp8 == "e5m2":
                    self.reference_precision_table = torch.full(
                        (k_cache.shape[0],), 2, dtype=torch.uint8, device=k_cache.device
                    )
                elif self.reference_fp8 == "alternating":
                    physical_blocks = torch.arange(
                        k_cache.shape[0], device=k_cache.device
                    )
                    self.reference_precision_table = (
                        physical_blocks.remainder(2) + 1
                    ).to(torch.uint8)
                else:
                    self.reference_precision_table = torch.ones(
                        k_cache.shape[0], dtype=torch.uint8, device=k_cache.device
                    )
            precision_table = self.reference_precision_table
        adaptive_threshold = (
            self.adaptive_e4m3_max_abs
            if precision_table is self.reference_precision_table
            else None
        )
        if precision_table is not None:
            # An externally supplied table may otherwise have no owner after
            # this forward; keep it alive for the plugin's later copy callback.
            self.active_precision_table = precision_table
            _REFERENCE_PRECISION_TABLES[k_cache.data_ptr()] = precision_table
        actual = attn_metadata.num_actual_tokens
        return reference_attention(
            query,
            key,
            value,
            k_cache,
            v_cache,
            attn_metadata.slot_mapping[:actual],
            attn_metadata.block_tables,
            attn_metadata.actual_seq_lengths_q,
            attn_metadata.seq_lens_list,
            output,
            self.scale,
            precision_table=precision_table,
            adaptive_e4m3_max_abs=adaptive_threshold,
            log_precision_transitions=self.log_precision_transitions,
            native_fp8=self.native_fp8,
            native_decode=self.native_decode,
            use_sdpa=self.sdpa_attention,
        )


class PSPKVAscendBackend(AscendAttentionBackend):
    """Use plugin metadata with PSP-KV's selected cache storage format."""

    supported_kv_cache_dtypes = ["fp8"]

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
        **kwargs,
    ) -> tuple[int, ...]:
        if os.environ.get("PSPKV_ASCEND_VARIABLE_PRECISION", "0") == "1":
            from vllm.v1.kv_cache_interface import _psp_kv_avg_bytes

            k_avg = _psp_kv_avg_bytes("k")
            v_avg = _psp_kv_avg_bytes("v")
            k_bytes = num_blocks * block_size * num_kv_heads * head_size * k_avg
            v_bytes = num_blocks * block_size * num_kv_heads * head_size * v_avg
            if (
                not math.isfinite(k_bytes)
                or not math.isfinite(v_bytes)
                or k_bytes <= 0
                or v_bytes <= 0
                or not float(k_bytes).is_integer()
                or not float(v_bytes).is_integer()
            ):
                raise ValueError(
                    "PSPKV_AVG_K_BYTES/PSPKV_AVG_V_BYTES do not yield "
                    f"integral byte arenas for {num_blocks} blocks"
                )
            logger.info_once(
                "PSP-KV variable-width cache: %s physical blocks, flat "
                "%s-byte K arena and %s-byte V arena (%s / %s bytes per "
                "element; worst case FP8 is %s per stream)",
                num_blocks,
                int(k_bytes),
                int(v_bytes),
                k_avg,
                v_avg,
                num_blocks * block_size * num_kv_heads * head_size,
            )
            return (2, int(k_bytes))
        if os.environ.get("PSPKV_ASCEND_UNIFORM_FP4", "0") == "1":
            if head_size % 2:
                raise ValueError("PSP-KV uniform FP4 requires an even head dimension")
            logger.info_once(
                "PSP-KV uniform FP4 cache: %s physical blocks, %s bytes per "
                "%s-token K/V page (E2M1, excluding scale metadata)",
                num_blocks,
                block_size * num_kv_heads * head_size,
                block_size,
            )
            return (2, num_blocks, block_size, num_kv_heads, head_size // 2)
        return AscendAttentionBackend.get_kv_cache_shape(
            num_blocks, block_size, num_kv_heads, head_size
        )

    @staticmethod
    def copy_blocks(kv_caches: list[torch.Tensor], src_to_dists: torch.Tensor) -> None:
        """Keep the physical FP8 format table in sync with copied KV pages."""
        if src_to_dists.numel() == 0:
            return
        if src_to_dists.ndim != 2 or src_to_dists.shape[1] != 2:
            raise ValueError("PSP-KV block copy requires [N, 2] block pairs")
        if os.environ.get("PSPKV_ASCEND_VARIABLE_PRECISION", "0") == "1":
            # The flat byte arenas have no fixed-stride block layout, so the
            # plugin's dense copy cannot touch them. Copy through each layer's
            # arena, which snapshots sources and requires matching formats.
            for kv_cache in kv_caches:
                arena = _VARIABLE_ARENAS.get(kv_cache[0].data_ptr())
                if arena is None:
                    raise RuntimeError(
                        "PSP-KV cannot copy variable-width KV without its arena"
                    )
                arena.copy_blocks(src_to_dists)
            return
        table_copies = []
        scale_copies = []
        for kv_cache in kv_caches:
            pointer = kv_cache[0].data_ptr()
            scales = _REFERENCE_FP4_SCALES.get(pointer)
            if os.environ.get("PSPKV_ASCEND_UNIFORM_FP4", "0") == "1":
                if scales is None:
                    raise RuntimeError(
                        "PSP-KV cannot copy FP4 KV without its scale table"
                    )
                src = src_to_dists[:, 0].to(device=scales.device, dtype=torch.int64)
                dst = src_to_dists[:, 1].to(device=scales.device, dtype=torch.int64)
                if bool(((src < 0) | (src >= scales.shape[0])).any()) or bool(
                    ((dst < 0) | (dst >= scales.shape[0])).any()
                ):
                    raise ValueError("PSP-KV block copy ID is outside the scale table")
                scale_copies.append((scales, dst, scales.index_select(0, src)))
                continue
            table = _REFERENCE_PRECISION_TABLES.get(pointer)
            if table is None:
                if os.environ.get("PSPKV_ASCEND_REFERENCE_FP8", "e4m3") != "e4m3":
                    raise RuntimeError(
                        "PSP-KV cannot copy mixed-format KV without its precision table"
                    )
                continue
            src = src_to_dists[:, 0].to(device=table.device, dtype=torch.int64)
            dst = src_to_dists[:, 1].to(device=table.device, dtype=torch.int64)
            if bool(((src < 0) | (src >= table.numel())).any()) or bool(
                ((dst < 0) | (dst >= table.numel())).any()
            ):
                raise ValueError("PSP-KV block copy ID is outside the precision table")
            table_copies.append((table, dst, table.index_select(0, src)))

        AscendAttentionBackend.copy_blocks(kv_caches, src_to_dists)
        for table, dst, source_formats in table_copies:
            table.index_copy_(0, dst, source_formats)
        for scales, dst, source_scales in scale_copies:
            scales.index_copy_(0, dst, source_scales)

    @staticmethod
    def get_impl_cls() -> type[PSPKVAscendImpl]:
        return PSPKVAscendImpl
