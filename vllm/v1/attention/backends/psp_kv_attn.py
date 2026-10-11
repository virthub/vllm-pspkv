# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass

import torch

from vllm.config import VllmConfig
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
from vllm.v1.kv_cache_interface import AttentionSpec


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
            context_lens = meta.seq_lens

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

        # ----- Resolve per-block byte offsets (uniform fallback for fixed mode) -----
        block_size = 16  # PSP-KV block size
        num_blocks = meta.precision_table.numel()
        block_offsets = getattr(meta, "block_offsets", None)
        if block_offsets is None:
            # Fixed-stride fallback (storage.mode="fixed"): all blocks share the
            # same precision, so per-block byte offsets are uniform. Avoid a
            # device sync here by assuming fp8 (1 byte/elem) for the fallback.
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

        # GQA: broadcast K/V to the number of query heads.
        g = num_heads // num_kv_heads
        k = key.repeat_interleave(g, dim=1) if g > 1 else key
        v = value.repeat_interleave(g, dim=1) if g > 1 else value

        query_start_loc = getattr(meta, "query_start_loc_cpu", None)
        if query_start_loc is None:
            query_start_loc = meta.query_start_loc.cpu()
        num_reqs = int(meta.num_reqs)

        parts = []
        for i in range(num_reqs):
            s = int(query_start_loc[i].item())
            e = int(query_start_loc[i + 1].item())
            if e <= s:
                continue
            L = e - s
            q_i = query[s:e].float()  # [L, H, D]
            k_i = k[s:e].float()      # [L, H, D]
            v_i = v[s:e].float()      # [L, H, D]
            # scores: [H, L, L]  (plain cuBLAS matmul + softmax; no FlashAttention)
            scores = torch.einsum("lhd,mhd->hlm", q_i, k_i) * scale
            causal = torch.triu(
                torch.ones(L, L, dtype=torch.bool, device=query.device), diagonal=1
            )
            scores = scores.masked_fill(causal.unsqueeze(0), float("-inf"))
            weights = torch.softmax(scores, dim=-1)  # [H, L, L]
            o_i = torch.einsum("hlm,mhd->lhd", weights, v_i)  # [L, H, D]
            parts.append(o_i.to(output.dtype))

        if parts:
            out = torch.cat(parts, dim=0)
            output[: out.size(0)].copy_(out)

        return output


class PSPKVAttentionBackend(AttentionBackend):
    """PSP-KV attention backend factory.

    Provides factory methods to instantiate the builder and implementation
    classes for PSP-KV attention.
    """

    # PSP-KV kernels are hardcoded to head_dim == 128 and block_size == 16.
    supported_dtypes = [torch.float16, torch.bfloat16]
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

        The cache is laid out as
            [2, num_blocks, num_kv_heads, block_size, head_size]
        so that the first dimension separates key and value.
        """
        return (2, num_blocks, num_kv_heads, block_size, head_size)
