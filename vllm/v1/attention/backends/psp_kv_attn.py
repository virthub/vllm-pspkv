# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from vllm.config import VllmConfig
from vllm.model_executor.models.utils import extract_layer_index
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionImpl,
    AttentionLayer,
    AttentionMetadata,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
)
from vllm.v1.kv_cache_interface import AttentionSpec, _psp_kv_avg_bytes

# Module-level reference to the model runner's prefill scratch buffer
# [num_layers, max_num_tokens, 2, num_kv_heads, head_size]. Set by
# GPUModelRunner.initialize_psp_kv_tables via set_prefill_kv_scratch().
_prefill_kv_scratch: torch.Tensor | None = None


def set_prefill_kv_scratch(scratch: torch.Tensor | None) -> None:
    global _prefill_kv_scratch
    _prefill_kv_scratch = scratch


_verify_stats = {"max": 0.0, "sum": 0.0, "n": 0, "calls": 0}


def _verify_fp32_closeness(query, k_cache, v_cache, block_tables, context_lens,
                           precision_table, scale_table, block_offsets,
                           num_kv_heads, softmax_scale, output, format_gen):
    """Compare the PSP-KV decode output against the fp32 reference (opt-in).

    This is the correctness criterion: the max/mean absolute difference to a
    reference fp32 paged attention over the same byte cache. It is gated by
    PSPKV_VERIFY=1 and is never on the default path.
    """
    import sys as _sys
    _root = os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
    if _root not in _sys.path:
        _sys.path.insert(0, _root)
    from tests.v1.attention.psp_kv_reference import reference_paged_attention

    ref = reference_paged_attention(
        query, k_cache, v_cache, block_tables, context_lens, format_gen,
        precision_table, scale_table, block_offsets, num_kv_heads, 16,
        softmax_scale,
    )
    diff = (output.float() - ref.float()).abs()
    _verify_stats["max"] = max(_verify_stats["max"], diff.max().item())
    _verify_stats["sum"] += diff.sum().item()
    _verify_stats["n"] += diff.numel()
    _verify_stats["calls"] += 1
    if _verify_stats["calls"] % 128 == 0:
        mean = _verify_stats["sum"] / _verify_stats["n"]
        print(f"[fp32-verify] max_abs={_verify_stats['max']:.3e} "
              f"mean_abs={mean:.3e} elements={_verify_stats['n']}", flush=True)


@dataclass
class PSPKVAttentionMetadata(AttentionMetadata):
    """Container for PSP-KV attention metadata.

    Holds a reference to the common attention metadata, which includes
    precision tables, scale tables, and TMA descriptors.
    """
    common_metadata: CommonAttentionMetadata


class PSPKVAttentionMetadataBuilder(AttentionMetadataBuilder[PSPKVAttentionMetadata]):
    """Builder for PSP-KV attention metadata.

    Declares single-token-decode CUDA Graph support because the PSP-KV
    attention kernels are optimised for decode phases.
    """
    _cudagraph_support = AttentionCGSupport.UNIFORM_SINGLE_TOKEN_DECODE

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
        **kwargs,
    ) -> PSPKVAttentionMetadata:
        """Build PSP-KV metadata by wrapping the common metadata."""
        return PSPKVAttentionMetadata(common_metadata=common_attn_metadata)


def _get_psp_kv_ops():
    """Locate the PSP-KV op namespace registered by the C++ extension.

    The wrapper `csrc/attention/psp_kv_wrapper.cu` registers its ops under
    the `psp_kv` namespace via `TORCH_LIBRARY(psp_kv, m)`. It is compiled
    into the `_C` extension (not `_C_stable_libtorch`), because it uses the
    regular C++ libtorch API, which is incompatible with the stable ABI
    (TORCH_TARGET_VERSION) used by `_C_stable_libtorch`.

    Once `_C` is imported, the ops are accessible as
    `torch.ops.psp_kv.<op_name>`.
    """
    # Make sure the _C extension is loaded so that the ops are registered.
    try:
        import vllm._C  # noqa: F401
    except ImportError as exc:
        raise RuntimeError(
            "Failed to import vllm._C; the PSP-KV custom ops cannot be "
            "resolved. Ensure the extension is built."
        ) from exc

    ops = getattr(torch.ops, "psp_kv", None)
    if ops is None or not hasattr(ops, "psp_attention_basic"):
        raise RuntimeError(
            "PSP-KV custom ops not found at torch.ops.psp_kv. "
            "Ensure the _C extension is built and loaded, and that "
            "csrc/attention/psp_kv_wrapper.cu was compiled into it."
        )
    return ops


class PSPKVAttentionImpl(AttentionImpl[PSPKVAttentionMetadata]):
    """PSP-KV attention implementation.

    Dispatches to Basic, Enhanced, or BT-KV kernels. Prefill is not supported
    here; it must be handled via the scratch buffer and FlashAttention path.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        self.logits_soft_cap = 0 if logits_soft_cap is None else logits_soft_cap
        if sliding_window is None:
            self.sliding_window = (-1, -1)
        else:
            self.sliding_window = (sliding_window - 1, 0)

        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "Encoder self-attention and encoder/decoder cross-attention "
                "are not implemented for PSPKVAttentionImpl."
            )

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: PSPKVAttentionMetadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        if attn_metadata is None:
            # Profiling run: no metadata available yet.
            return output.fill_(0)

        # The PSP-KV ops require a contiguous query tensor.
        query = query.contiguous()

        meta = attn_metadata.common_metadata

        # ===== Prefill detection (safe for CUDA Graph) =====
        # Avoid GPU-CPU sync during graph capture.
        if not torch.cuda.is_current_stream_capturing():
            num_actual = getattr(meta, "num_actual_tokens", None)
            num_reqs = getattr(meta, "num_reqs", None)
            if num_actual is not None and num_reqs is not None and num_actual > num_reqs:
                # PSP-KV kernels are decode-only: prefill computes attention
                # directly from the projected Q/K/V (the cache is populated by
                # the model runner's quantize-and-scatter step afterwards).
                return self._prefill_forward(query, key, value, meta, output, layer)

        format_gen = meta.format_generation

        # Unpack key and value caches from unified cache [2, num_blocks, ...].
        k_cache = kv_cache[0]
        v_cache = kv_cache[1]

        # ----- Resolve block table and context length tensors safely -----
        block_tables = getattr(meta, "block_tables", None)
        if block_tables is None:
            block_tables = meta.block_table_tensor

        context_lens = getattr(meta, "context_lens", None)
        if context_lens is None:
            # seq_lens is the optimistic length (includes the current decode
            # token); the cache holds one fewer token, so subtract 1.
            context_lens = meta.seq_lens - 1

        # ----- Ensure required tensor invariants for CUDA kernels -----
        # The model runner must provide these in the correct format before
        # graph capture. Asserting here ensures early detection of mismatches.
        assert context_lens.dtype == torch.int32, (
            f"context_lens must be int32 (got {context_lens.dtype}); "
            "ensure the model runner prepares it as int32"
        )
        assert context_lens.is_contiguous(), (
            "context_lens must be contiguous; ensure the model runner provides "
            "a contiguous tensor"
        )
        assert block_tables.is_contiguous(), (
            "block_tables must be contiguous; ensure the model runner provides "
            "a contiguous tensor"
        )

        # ----- Resolve num_kv_heads (handling tensor parallelism) -----
        num_kv_heads = getattr(layer, "num_kv_heads_per_partition", None)
        if num_kv_heads is None:
            num_kv_heads = layer.num_kv_heads

        head_size = layer.head_size
        # The softmax scale lives on the impl (set from the constructor's
        # `scale` argument), not on the Attention layer.
        softmax_scale = self.scale

        # ----- Locate the PSP-KV op namespace -----
        # The wrapper registers ops under `torch.ops.psp_kv.*`.
        ops = _get_psp_kv_ops()

        # ----- Resolve per-block byte offsets -----
        block_size = 16  # PSP-KV block size
        num_blocks = meta.precision_table.numel()
        block_offsets = getattr(meta, "block_offsets", None)
        if block_offsets is None:
            # Fallback for callers that don't pass block_offsets through the
            # metadata (e.g. unit tests): derive a uniform stride from the
            # leading block's precision.
            prec = int(meta.precision_table[0].item()) if num_blocks > 0 else 1
            if prec == 0:
                elem_bytes = query.element_size()
            elif prec == 3:
                elem_bytes = 0.5
            else:
                elem_bytes = 1
            block_bytes = int(block_size * head_size * elem_bytes * num_kv_heads)
            block_offsets = torch.arange(
                num_blocks, dtype=torch.int32, device=query.device
            ) * block_bytes

        # ----- Dispatch to the correct PSP-KV kernel -----
        if format_gen == 0:
            # Basic format: staging path. Dequantize the compressed byte buffer
            # into a uniform target-precision buffer, then run the transparent
            # (uniform) consumer that needs no inner-precision metadata.
            k_staging = torch.empty(
                num_blocks, num_kv_heads, block_size, head_size,
                dtype=query.dtype, device=query.device,
            )
            v_staging = torch.empty_like(k_staging)
            scale_table = getattr(meta, "scale_table", None)
            if scale_table is None:
                scale_table = torch.ones(num_blocks, num_kv_heads,
                                         dtype=torch.float32, device=query.device)
            ops.psp_dequantize_kv(
                k_cache, v_cache, block_offsets, meta.precision_table,
                scale_table, num_blocks, num_kv_heads, block_size, head_size,
                k_staging, v_staging,
            )
            ops.psp_attention_uniform(
                query, k_staging, v_staging, block_tables, context_lens,
                softmax_scale, num_kv_heads, head_size, output,
            )
        elif format_gen == 1:
            # Enhanced format - precision_table + block_offsets + scale_table.
            ops.psp_attention_enhanced(
                query,
                k_cache,
                v_cache,
                block_tables,
                context_lens,
                meta.precision_table,
                block_offsets,
                meta.scale_table,
                softmax_scale,
                num_kv_heads,
                head_size,
                self.sliding_window[0] + self.sliding_window[1] + 1,
                output,
            )
        elif format_gen == 2:
            # BT-KV format - requires TMA descriptors.
            if meta.tma_descriptors is None:
                raise RuntimeError(
                    "BT-KV format requires TMA descriptors, but they are "
                    "missing in metadata."
                )
            ops.psp_attention_btkv(
                query,
                k_cache,
                v_cache,
                block_tables,
                context_lens,
                meta.precision_table,
                block_offsets,
                meta.scale_table,
                meta.tma_descriptors,
                softmax_scale,
                num_kv_heads,
                head_size,
                output,
            )
        else:
            raise ValueError(
                f"Unsupported PSP-KV format_generation: {format_gen}"
            )

        # Decode write-back: quantize this step's K/V with the block's existing
        # scale and scatter it into the byte cache so the next step sees it.
        slot_mapping = getattr(meta, "slot_mapping", None)
        if os.environ.get("PSPKV_VERIFY") == "1" and slot_mapping is not None:
            _verify_fp32_closeness(
                query, k_cache, v_cache, block_tables, context_lens,
                meta.precision_table, meta.scale_table, block_offsets,
                num_kv_heads, softmax_scale, output, format_gen)
        if slot_mapping is not None and key is not None and value is not None:
            ops.quantize_single_token(
                key, value, k_cache, v_cache,
                slot_mapping,
                meta.precision_table,
                block_offsets,
                meta.scale_table,
                format_gen,
            )

        return output

    def _prefill_forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        meta,
        output: torch.Tensor,
        layer,
    ) -> torch.Tensor:
        """Prefill attention via the scratch-buffer path.

        PSP-KV kernels are decode-only. During prefill the projected K/V are
        still in the ``key``/``value`` tensors (the model runner quantizes them
        into the byte cache after the forward pass). Compute causal attention
        directly from Q/K/V per sequence and write it to ``output``.
        """
        num_heads = query.size(1)
        num_kv_heads = key.size(1)
        scale = self.scale

        # GQA ratio: pass K/V un-broadcast and let the SDPA backend handle GQA
        # natively (enable_gqa=True), avoiding the repeat_interleave copy.
        g = num_heads // num_kv_heads

        query_start_loc = getattr(meta, "query_start_loc_cpu", None)
        if query_start_loc is None:
            query_start_loc = meta.query_start_loc.cpu()
        num_reqs = int(meta.num_reqs)

        # Prefill scratch: write the projected K/V so the model runner can
        # quantize them afterwards (the scratch is read by quantize_and_scatter_kv).
        scratch = _prefill_kv_scratch
        layer_idx = extract_layer_index(layer.layer_name) if scratch is not None else -1

        # Single-sequence fast path: one fused SDPA call, no Python loop.
        if num_reqs == 1:
            s = int(query_start_loc[0].item())
            e = int(query_start_loc[1].item())
            if e > s:
                if scratch is not None:
                    scratch[layer_idx, s:e, 0].copy_(key[s:e])
                    scratch[layer_idx, s:e, 1].copy_(value[s:e])
                q = query[s:e].transpose(0, 1).unsqueeze(0)   # [1, H, L, D]
                k = key[s:e].transpose(0, 1).unsqueeze(0)     # [1, H_kv, L, D]
                v = value[s:e].transpose(0, 1).unsqueeze(0)
                o = F.scaled_dot_product_attention(
                    q, k, v, is_causal=True, scale=scale, enable_gqa=(g > 1)
                )
                output[s:e].copy_(o.squeeze(0).transpose(0, 1).to(output.dtype))
            return output

        # Multi-sequence: per-sequence SDPA with native GQA (no repeat_interleave).
        parts = []
        for i in range(num_reqs):
            s = int(query_start_loc[i].item())
            e = int(query_start_loc[i + 1].item())
            if e <= s:
                continue
            if scratch is not None:
                scratch[layer_idx, s:e, 0].copy_(key[s:e])
                scratch[layer_idx, s:e, 1].copy_(value[s:e])
            o_i = F.scaled_dot_product_attention(
                query[s:e].transpose(0, 1).unsqueeze(0),
                key[s:e].transpose(0, 1).unsqueeze(0),
                value[s:e].transpose(0, 1).unsqueeze(0),
                is_causal=True,
                scale=scale,
                enable_gqa=(g > 1),
            )  # [1, H, L, D]
            parts.append(o_i.squeeze(0).transpose(0, 1).to(output.dtype))

        if parts:
            out = torch.cat(parts, dim=0)
            output[: out.size(0)].copy_(out)

        return output


class PSPKVAttentionBackend(AttentionBackend):
    """PSP-KV attention backend factory.

    Provides factory methods to instantiate the builder and implementation
    classes for PSP-KV attention.
    """

    # PSP-KV kernels dispatch on head_dim at runtime; block_size is fixed at 16.
    supported_dtypes = [torch.float16, torch.bfloat16, torch.float32]
    # The cache is a uint8 byte buffer: 1 byte/elem for the fp8 inner precisions.
    supported_kv_cache_dtypes = ["auto", "fp8"]

    @staticmethod
    def get_name() -> str:
        return "PSP_KV"

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        # Keep in sync with csrc/attention/psp_kv/common/head_dim_dispatch.cuh
        return [32, 64, 80, 96, 128, 160, 192, 256]

    @classmethod
    def get_supported_kernel_block_sizes(cls) -> list[int]:
        return [16]

    @classmethod
    def get_kv_cache_block_dim(
        cls,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> int:
        # The PSP-KV cache is a flat byte buffer (num_blocks is folded into the
        # byte footprint and tracked via block_offsets), so there is no block
        # dimension to re-layout for hybrid models. Returning 0 makes the
        # hybrid-layout update skip this group.
        return 0

    @staticmethod
    def get_impl_cls():
        return PSPKVAttentionImpl

    @staticmethod
    def get_builder_cls():
        return PSPKVAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        """Return the shape of the unified KV cache.

        The cache is a flat byte buffer
            [2, num_blocks * block_size * num_kv_heads * head_size * avg_bytes]
        where the first dimension separates key and value and the second is the
        per-K/V byte footprint. Per-block byte offsets are tracked separately
        (``block_offsets``) because blocks can have different precisions.
        """
        per_kv_bytes = int(
            num_blocks * block_size * num_kv_heads * head_size * _psp_kv_avg_bytes()
        )
        return (2, per_kv_bytes)
