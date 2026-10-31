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


_FP4_E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, float("inf"), float("inf"),
             -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, float("-inf"), float("-inf")]

# [256, 2] byte lookup table: byte b -> [fp4(b & 0xF), fp4(b >> 4)].
# Lets the fp4 dequant be a single table-gather instead of bitwise_and + rshift
# + stack + gather (which was ~5 GPU kernels per block).
_FP4_BYTE_TABLE: torch.Tensor | None = None


def _fp4_byte_table(device: torch.device) -> torch.Tensor:
    global _FP4_BYTE_TABLE
    if _FP4_BYTE_TABLE is None or _FP4_BYTE_TABLE.device != device:
        t = torch.tensor(_FP4_E2M1, dtype=torch.float32, device=device)
        low = t.repeat(16)             # low[b] = t[b & 0xF] = t[b % 16]
        high = t.repeat_interleave(16)  # high[b] = t[b >> 4] = t[b // 16]
        _FP4_BYTE_TABLE = torch.stack([low, high], dim=1).contiguous()
    return _FP4_BYTE_TABLE


def _psp_fmt_k(p: int) -> int:
    return (p & 3) if p >= 4 else p


def _psp_fmt_v(p: int) -> int:
    return (p >> 2) if p >= 4 else p


def _dequant_fp8(q: torch.Tensor, fmt: int) -> torch.Tensor:
    """Bit-exact e4m3/e5m2 dequant (uint8 -> float32)."""
    q = q.to(torch.int32)
    if fmt == 2:  # e5m2
        sign = (q >> 7) & 1
        exp = (q >> 2) & 0x1F
        mant = q & 0x3
        normal = (1.0 + mant.float() / 4.0) * torch.pow(2.0, exp.float() - 15.0)
        subnormal = mant.float() * (2.0 ** -16.0)
        special = torch.where(mant == 0, float("inf"), float("nan"))
        max_exp = 31
    else:  # e4m3
        sign = (q >> 7) & 1
        exp = (q >> 3) & 0xF
        mant = q & 0x7
        normal = (1.0 + mant.float() / 8.0) * torch.pow(2.0, exp.float() - 7.0)
        subnormal = mant.float() * (2.0 ** -9.0)
        special = torch.where(mant == 7, float("nan"), normal)
        max_exp = 15
    val = torch.where(exp == 0, subnormal, normal)
    val = torch.where(exp == max_exp, special, val)
    return torch.where(sign == 1, -val, val)


def _dequant_one_block(cache: torch.Tensor, off: int, fmt: int,
                       num_kv_heads: int, block_size: int, head_size: int,
                       scalar_size: int, dtype: torch.dtype,
                       scale_table: torch.Tensor | None,
                       scale_row: int) -> torch.Tensor:
    """Dequantize one block's K or V into [num_kv_heads, block_size, head_size]."""
    if fmt == 0:
        nbytes = block_size * head_size * scalar_size * num_kv_heads
        return cache[off:off + nbytes].view(dtype).reshape(
            num_kv_heads, block_size, head_size).float()
    if fmt == 3:
        nbytes = (block_size * head_size // 2) * num_kv_heads
        buf = cache[off:off + nbytes]
        out = _fp4_byte_table(cache.device)[buf.to(torch.int64)].reshape(
            num_kv_heads, block_size, head_size)
    else:
        nbytes = block_size * head_size * num_kv_heads
        out = _dequant_fp8(cache[off:off + nbytes], fmt).reshape(
            num_kv_heads, block_size, head_size)
    if scale_table is not None:
        s = scale_table[scale_row].float()[:, None, None]
        out = out * s
    return out


def _dequant_prefix(k_cache, v_cache, block_tables, precision_table,
                    scale_table, block_offsets, seq_idx, context_len,
                    num_kv_heads, head_size, block_size, dtype):
    """Dequantize one request's cached prefix into [num_kv_heads, ctx, head_size]."""
    if context_len <= 0:
        return None, None
    n_blk = (context_len + block_size - 1) // block_size
    # Single fused kernel launch for the whole prefix (no per-block Python loop
    # and no GPU->CPU sync): dequantizes the sequence's physical blocks into
    # [n_blk, num_kv_heads, block_size, head_size].
    bids = block_tables[seq_idx, :n_blk]
    k_out = torch.empty(n_blk, num_kv_heads, block_size, head_size,
                        dtype=dtype, device=k_cache.device)
    v_out = torch.empty_like(k_out)
    _get_psp_kv_ops().psp_dequantize_prefix(
        k_cache, v_cache, bids, precision_table, block_offsets, scale_table,
        num_kv_heads, block_size, head_size, k_out, v_out)
    k = k_out.permute(1, 0, 2, 3).reshape(
        num_kv_heads, n_blk * block_size, head_size)[:, :context_len].contiguous()
    v = v_out.permute(1, 0, 2, 3).reshape(
        num_kv_heads, n_blk * block_size, head_size)[:, :context_len].contiguous()
    return k, v


_verify_stats = {"max": 0.0, "sum": 0.0, "n": 0, "calls": 0}


def _verify_fp32_closeness(query, k_cache, v_cache, block_tables, context_lens,
                           precision_table, scale_table, block_offsets,
                           num_kv_heads, softmax_scale, output, format_gen,
                           sliding_window=-1):
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
        softmax_scale, sliding_window,
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
        format_gen = meta.format_generation

        # Unpack key and value caches. The model runner passes a flat byte
        # buffer [K_bytes | V_bytes]; the tests pass a stacked [2, bytes].
        if kv_cache.dim() == 2 and kv_cache.size(0) == 2:
            k_cache = kv_cache[0]
            v_cache = kv_cache[1]
        elif getattr(meta, "k_cache_bytes", 0) > 0:
            k_bytes = meta.k_cache_bytes
            v_bytes = meta.v_cache_bytes
            k_cache = kv_cache[:k_bytes]
            v_cache = kv_cache[k_bytes:k_bytes + v_bytes]
        else:
            avg = float(os.environ.get("PSPKV_AVG_BYTES", "1.0"))
            k_bytes = int(kv_cache.numel() / (2 * avg))
            k_cache = kv_cache[:k_bytes]
            v_cache = kv_cache[k_bytes:]

        # ----- Resolve block table and context length tensors safely -----
        block_tables = getattr(meta, "block_tables", None)
        if block_tables is None:
            block_tables = meta.block_table_tensor

        context_lens = getattr(meta, "context_lens", None)
        if context_lens is None:
            # seq_lens already includes the current decode token's slot; the
            # current token is written to that slot before attention so the
            # decode attends over the full prefix plus itself.
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

        # Per-layer K/V scales: the runner stores the scale table with a
        # leading layer dimension; slice out this layer's scales (the tests
        # pass a 2D table, which is left as-is).
        scale_table = getattr(meta, "scale_table", None)
        if scale_table is not None and scale_table.dim() == 4:
            layer_name = getattr(layer, "layer_name", None)
            layer_idx = extract_layer_index(layer_name) if layer_name is not None else 0
            scale_table = scale_table[layer_idx]

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

        # ===== Prefill detection (safe for CUDA Graph) =====
        # Avoid GPU-CPU sync during graph capture.
        if not torch.cuda.is_current_stream_capturing():
            num_actual = getattr(meta, "num_actual_tokens", None)
            num_reqs = getattr(meta, "num_reqs", None)
            if num_actual is not None and num_reqs is not None and num_actual > num_reqs:
                # PSP-KV kernels are decode-only: prefill computes attention
                # from the projected Q/K/V plus any already-cached (quantized)
                # prefix; the new chunk is quantized-and-scattered afterwards.
                return self._prefill_forward(
                    query, key, value, meta, output, layer, k_cache, v_cache,
                    block_tables, block_offsets, num_kv_heads, head_size)

        # Decode write-back happens before attention so the current token is
        # already in its cache slot when the kernel reads `context_lens`
        # (= seq_lens) tokens, matching the FlashAttention baseline's self-
        # attention over [prefix | current token].
        slot_mapping = getattr(meta, "slot_mapping", None)
        if slot_mapping is not None and key is not None and value is not None:
            ops.quantize_single_token(
                key.contiguous(), value.contiguous(), k_cache, v_cache,
                slot_mapping,
                meta.precision_table,
                block_offsets,
                scale_table,
                format_gen,
            )

        # ----- Dispatch to the correct PSP-KV kernel -----
        if (format_gen == 1 and getattr(meta, "precision_is_lossless", False)
                and query.dtype in (torch.float16, torch.bfloat16)):
            self._lossless_flash_attn(
                query, k_cache, v_cache, block_tables, context_lens,
                meta, block_offsets, num_kv_heads, head_size, output,
                scale_table)
        elif format_gen == 0:
            # Basic format: staging path. Dequantize the compressed byte buffer
            # into a uniform target-precision buffer, then run the transparent
            # (uniform) consumer that needs no inner-precision metadata.
            k_staging = torch.empty(
                num_blocks, num_kv_heads, block_size, head_size,
                dtype=query.dtype, device=query.device,
            )
            v_staging = torch.empty_like(k_staging)
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
                scale_table,
                softmax_scale,
                num_kv_heads,
                head_size,
                self.sliding_window[0] + self.sliding_window[1] + 1,
                getattr(meta, "precision_is_lossless", False),
                getattr(meta, "precision_fp8_fmt", 0),
                getattr(meta, "precision_v_fmt", 0),
                getattr(meta, "precision_k_fmt", 0),
                getattr(meta, "precision_k_lossless", False),
                getattr(meta, "precision_v_lossless", False),
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
                scale_table,
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

        if os.environ.get("PSPKV_VERIFY") == "1" and slot_mapping is not None:
            _verify_fp32_closeness(
                query, k_cache, v_cache, block_tables, context_lens,
                meta.precision_table, scale_table, block_offsets,
                num_kv_heads, softmax_scale, output, format_gen,
                self.sliding_window[0] + self.sliding_window[1] + 1)

        return output

    def _lossless_flash_attn(
        self,
        query: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        block_tables: torch.Tensor,
        context_lens: torch.Tensor,
        meta,
        block_offsets: torch.Tensor,
        num_kv_heads: int,
        head_size: int,
        output: torch.Tensor,
        scale_table: torch.Tensor | None = None,
    ) -> torch.Tensor:
        from vllm.vllm_flash_attn.flash_attn_interface import (
            flash_attn_varlen_func,
        )

        scale = self.scale
        block_size = 16
        num_reqs = context_lens.size(0)
        num_actual = getattr(meta, "num_actual_tokens", num_reqs)

        ctx_cpu = [int(c) for c in context_lens.cpu().tolist()]
        n_blk = [(c + block_size - 1) // block_size for c in ctx_cpu]
        padded = [n * block_size for n in n_blk]
        total_padded = sum(padded)
        out_offsets = [0]
        for pl in padded:
            out_offsets.append(out_offsets[-1] + pl)
        max_blocks = max(n_blk) if n_blk else 1

        k_pre = torch.empty(total_padded, num_kv_heads, head_size,
                            dtype=query.dtype, device=query.device)
        v_pre = torch.empty_like(k_pre)
        if total_padded > 0:
            ctx_t = context_lens.to(torch.int32)
            offs_t = torch.tensor(out_offsets[:num_reqs], dtype=torch.int32,
                                  device=query.device)
            _get_psp_kv_ops().psp_dequantize_prefix_batch(
                k_cache, v_cache, block_tables[:, :max_blocks].contiguous(),
                ctx_t, offs_t, meta.precision_table, block_offsets,
                scale_table, num_kv_heads, block_size, head_size,
                k_pre, v_pre)

        # Gather the valid tokens (the pre-write already placed the current
        # token in its cache slot, so context_lens == seq_lens includes it).
        k_parts, v_parts = [], []
        k_lens = []
        for j in range(num_reqs):
            ctx = ctx_cpu[j]
            o0 = out_offsets[j]
            k_parts.append(k_pre[o0:o0 + ctx])
            v_parts.append(v_pre[o0:o0 + ctx])
            k_lens.append(ctx)

        k_big = torch.cat(k_parts, dim=0)
        v_big = torch.cat(v_parts, dim=0)
        cu_q = torch.arange(num_reqs + 1, dtype=torch.int32, device=query.device)
        cu_k = torch.cumsum(
            torch.tensor([0] + k_lens, dtype=torch.int32, device=query.device),
            dim=0).to(torch.int32)
        max_k = max(k_lens)

        o = flash_attn_varlen_func(
            q=query[:num_actual], k=k_big, v=v_big, out=output[:num_actual],
            cu_seqlens_q=cu_q, max_seqlen_q=1, max_seqlen_k=max_k,
            cu_seqlens_k=cu_k, softmax_scale=scale, causal=True, num_splits=0,
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
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        block_tables: torch.Tensor,
        block_offsets: torch.Tensor,
        num_kv_heads: int,
        head_size: int,
    ) -> torch.Tensor:
        """Prefill attention via the scratch-buffer path.

        PSP-KV kernels are decode-only. The projected K/V for the current chunk
        are in ``key``/``value`` (the model runner quantizes them into the byte
        cache after the forward pass). For chunked prefill the earlier tokens
        already live in the quantized cache, so dequantize that prefix and attend
        over [cached prefix | current chunk].
        """
        num_heads = query.size(1)
        scale = self.scale
        block_size = 16
        total_q = query.size(0)

        from vllm.vllm_flash_attn.flash_attn_interface import (
            flash_attn_varlen_func,
        )

        query_start_loc = meta.query_start_loc  # [num_reqs+1] int32 GPU
        num_reqs = int(meta.num_reqs)

        # Prefill scratch: write the projected K/V so the model runner can
        # quantize them afterwards (the scratch is read by quantize_and_scatter_kv).
        scratch = _prefill_kv_scratch
        layer_idx = extract_layer_index(layer.layer_name) if scratch is not None else -1
        scale_table = getattr(meta, "scale_table", None)
        if scale_table is not None and scale_table.dim() == 4:
            layer_name = getattr(layer, "layer_name", None)
            layer_idx = extract_layer_index(layer_name) if layer_name is not None else 0
            scale_table = scale_table[layer_idx]
        if scratch is not None:
            scratch[layer_idx, :total_q, 0].copy_(key)
            scratch[layer_idx, :total_q, 1].copy_(value)

        # Cached-prefix lengths: ctx = seq_lens - q_len (0 for a fresh prefill).
        q_lens = query_start_loc[1:] - query_start_loc[:-1]   # [num_reqs] int32
        ctx_lens = meta.seq_lens.to(q_lens.dtype) - q_lens    # [num_reqs]
        max_q = int(meta.max_query_len)

        if not bool((ctx_lens > 0).any().item()):
            # Fresh prefill: one batched causal flash-attn over the chunk.
            o = flash_attn_varlen_func(
                query, key, value, max_q, query_start_loc,
                max_q, query_start_loc,
                softmax_scale=scale, causal=True, num_splits=0,
            )
            output[:total_q].copy_(o.to(output.dtype))
            return output

        # Cached-prefix case: split the batch. Speculative-verify sequences
        # (uniform N tokens, ctx > 0) use the in-kernel multi-token attention so
        # the fp4/fp8 prefix dequant is amortized over N; the chunked-prefill
        # remainder uses the dense-buffer + left-padded flash-attn path.
        qs_cpu = query_start_loc.cpu().tolist()
        ctx_cpu = [int(c) for c in ctx_lens.cpu().tolist()]
        q_lens_cpu = [qs_cpu[i + 1] - qs_cpu[i] for i in range(num_reqs)]

        verify_idx = [i for i in range(num_reqs)
                      if ctx_cpu[i] > 0 and 1 < q_lens_cpu[i] <= 8]
        uniform_n = (len({q_lens_cpu[i] for i in verify_idx}) == 1
                     if verify_idx else False)

        if (verify_idx and uniform_n and meta.format_generation == 1 and
                not getattr(meta, "precision_is_lossless", False)):
            N = q_lens_cpu[verify_idx[0]]
            sub_q = torch.cat([query[qs_cpu[i]:qs_cpu[i + 1]] for i in verify_idx], dim=0)
            sub_k = torch.cat([key[qs_cpu[i]:qs_cpu[i + 1]] for i in verify_idx], dim=0)
            sub_v = torch.cat([value[qs_cpu[i]:qs_cpu[i + 1]] for i in verify_idx], dim=0)
            sub_bt = block_tables[verify_idx].contiguous()
            sub_ctx = ctx_lens[verify_idx].to(torch.int32)
            sub_out = torch.empty_like(sub_q)
            _get_psp_kv_ops().psp_attention_enhanced_mt(
                sub_q, sub_k, sub_v, k_cache, v_cache, sub_bt, sub_ctx,
                meta.precision_table, block_offsets, scale_table,
                scale, num_kv_heads, head_size,
                getattr(meta, "precision_fp8_fmt", 1), N, sub_out)
            off = 0
            for i in verify_idx:
                s, e = qs_cpu[i], qs_cpu[i + 1]
                output[s:e].copy_(sub_out[off:off + (e - s)].to(output.dtype))
                off += e - s

        rest_idx = [i for i in range(num_reqs) if i not in verify_idx]
        if not rest_idx:
            return output

        # Chunked-prefill remainder: batch-dequantize each cached prefix and run
        # one left-padded flash-attn over [prefix | chunk].
        rest_ql = [q_lens_cpu[i] for i in rest_idx]
        rest_ctx = [ctx_cpu[i] for i in rest_idx]
        n_blk_list = [(c + block_size - 1) // block_size for c in rest_ctx]
        padded_lens = [n * block_size for n in n_blk_list]
        total_padded = sum(padded_lens)
        out_offsets = [0]
        for pl in padded_lens:
            out_offsets.append(out_offsets[-1] + pl)
        max_blocks = max(n_blk_list) if n_blk_list else 1

        rest_bt = block_tables[rest_idx].contiguous()
        k_pre_big = torch.empty(total_padded, num_kv_heads, head_size,
                                dtype=query.dtype, device=query.device)
        v_pre_big = torch.empty_like(k_pre_big)
        if total_padded > 0:
            ctx_lens_t = torch.tensor(rest_ctx, dtype=torch.int32,
                                      device=query.device)
            offs_t = torch.tensor(out_offsets[:len(rest_idx)], dtype=torch.int32,
                                  device=query.device)
            _get_psp_kv_ops().psp_dequantize_prefix_batch(
                k_cache, v_cache, rest_bt[:, :max_blocks].contiguous(),
                ctx_lens_t, offs_t, meta.precision_table, block_offsets,
                scale_table, num_kv_heads, block_size, head_size,
                k_pre_big, v_pre_big)

        rest_q = torch.cat([query[qs_cpu[i]:qs_cpu[i + 1]] for i in rest_idx], dim=0)
        rest_k = torch.cat([key[qs_cpu[i]:qs_cpu[i + 1]] for i in rest_idx], dim=0)
        rest_v = torch.cat([value[qs_cpu[i]:qs_cpu[i + 1]] for i in rest_idx], dim=0)
        rest_q_start = [0]
        for ql in rest_ql:
            rest_q_start.append(rest_q_start[-1] + ql)

        k_parts, v_parts = [], []
        k_lens = []
        for j in range(len(rest_idx)):
            s, e = rest_q_start[j], rest_q_start[j + 1]
            ctx = rest_ctx[j]
            o0 = out_offsets[j]
            if ctx > 0:
                k_i = torch.cat([k_pre_big[o0:o0 + ctx], rest_k[s:e]], dim=0)
                v_i = torch.cat([v_pre_big[o0:o0 + ctx], rest_v[s:e]], dim=0)
            else:
                k_i = rest_k[s:e]
                v_i = rest_v[s:e]
            k_parts.append(k_i)
            v_parts.append(v_i)
            k_lens.append(ctx + (e - s))

        k_big = torch.cat(k_parts, dim=0)
        v_big = torch.cat(v_parts, dim=0)
        cu_k = torch.cumsum(
            torch.tensor([0] + k_lens, dtype=torch.int32, device=query.device),
            dim=0).to(torch.int32)
        max_k = int(max(k_lens))
        rest_cu_q = torch.tensor(rest_q_start, dtype=torch.int32,
                                 device=query.device).to(torch.int32)
        max_q_rest = int(max(rest_ql))

        o = flash_attn_varlen_func(
            rest_q, k_big, v_big, max_q_rest, rest_cu_q, max_k, cu_k,
            softmax_scale=scale, causal=True, num_splits=0,
        )
        off = 0
        for i in rest_idx:
            s, e = qs_cpu[i], qs_cpu[i + 1]
            output[s:e].copy_(o[off:off + (e - s)].to(output.dtype))
            off += e - s
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
        dtype: torch.dtype = torch.float16,
    ) -> tuple[int, ...]:
        """Return the shape of the unified KV cache.

        The cache is a single flat byte buffer of length K_bytes + V_bytes,
        where K and V have independent per-block byte offsets (``block_offsets``
        holds [K | V]). Keeping it one 1-D tensor lets K (fp8, 1 B/elem) and V
        (fp4, 0.5 B/elem) use different footprints.
        """
        per_kv_bytes = int(
            num_blocks * block_size * num_kv_heads * head_size * _psp_kv_avg_bytes(dtype)
        )
        return (2 * per_kv_bytes,)
