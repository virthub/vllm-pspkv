# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Opt-in Ascend execution of develop's uniform PSP-KV strategy matrix.

Same four inner precisions and token-major K/V as develop 00b78e9. No RMSE
selector, grouped calibration, asymmetric K/V, or BF16 shadow allocation.
"""

import json
import os
from copy import copy
from dataclasses import dataclass
from functools import lru_cache

import torch

FORMATS = ("basic", "enhanced", "btkv")
PRECISIONS = {"lossless": 0, "e4m3": 1, "e5m2": 2, "fp4": 3}
STATES = {}


@lru_cache(maxsize=1)
def load_cube_ops():
    library = os.environ.get("PSPKV_ASCEND_CUBE_LIB")
    if not library:
        raise RuntimeError("Cube attention requires PSPKV_ASCEND_CUBE_LIB")
    torch.ops.load_library(library)
    return torch.ops.psp_kv_ascend_cube


@lru_cache(maxsize=2)
def causal_mask(device):
    return torch.ones((2048, 2048), dtype=torch.bool).triu(1).to(device)


@lru_cache(maxsize=2)
def strategy_tile_permutation(device):
    from vllm.v1.attention.backends.psp_kv_ascend_reference import decode_e2m1

    index = torch.arange(32 * 128, dtype=torch.int32)
    permutation = (index % 128 * 32 + index // 128) * 4
    byte = torch.arange(256, dtype=torch.int32)
    nibbles = torch.stack((byte & 15, byte >> 4), dim=-1).flatten().to(torch.uint8)
    pair_table = decode_e2m1(nibbles).view(torch.int32)
    return torch.cat((permutation, pair_table)).to(device)


@dataclass(frozen=True)
class Strategy:
    generation: int
    precision: str
    adaptive: bool = False

    @property
    def code(self):
        return PRECISIONS[self.precision]

    @property
    def element_bytes(self):
        return (2.0, 1.0, 1.0, 0.5)[self.code]


@lru_cache(maxsize=1)
def read_strategy():
    configured_format = os.environ.get("PSPKV_FORMAT", "1").lower()
    generation = (
        FORMATS.index(configured_format)
        if configured_format in FORMATS
        else int(configured_format)
    )
    precision = os.environ.get("PSPKV_PRECISION", "e4m3").lower()
    precision = {
        "bf16": "lossless",
        "fp16": "lossless",
        "0": "lossless",
        "1": "e4m3",
        "2": "e5m2",
        "3": "fp4",
    }.get(precision, precision)
    if generation not in range(3) or precision not in PRECISIONS:
        raise ValueError(
            "NVIDIA matrix requires Basic/Enhanced/BTKV and lossless/E4M3/E5M2/FP4"
        )
    adaptive = os.environ.get("PSPKV_ADAPTIVE_PRECISION", "0") == "1"
    if adaptive and precision != "e4m3":
        raise ValueError(
            "The separate adaptive experiment uses the NVIDIA E4M3 default"
        )
    strategy = Strategy(generation, precision, adaptive)
    for name in ("PSPKV_AVG_BYTES", "PSPKV_AVG_K_BYTES", "PSPKV_AVG_V_BYTES"):
        if name in os.environ and float(os.environ[name]) != strategy.element_bytes:
            raise ValueError(
                "Uniform NVIDIA strategies require their exact storage budget"
            )
    return strategy


def derive_develop_codes(maxima):
    """Mirror develop's fixed E=1e-3, fp_max=240 amplitude selector."""
    worst = maxima.amax(dim=1)
    required = torch.log2(worst.clamp_min(1e-12) / (1e-3 * 240.0))
    return torch.where((worst == 0) | (required > 1), 1, 3).to(torch.uint8)


def strategy_attention(
    query, key, value, keys, values, metadata, output, softmax_scale, implementation
):
    from pipeline_runtime import (
        GRAPH_INPUTS,
        consume_deferred_rope,
        graph_signature,
        pipeline_level,
    )
    from shared_metadata import current_owner, prepare_metadata

    from vllm.v1.attention.backends.psp_kv_ascend_native import load_native_fp8_ops
    from vllm.v1.attention.backends.psp_kv_ascend_ops import (
        _forward_workspace,
        _scratch_block_table,
        packed_lookup_tables,
    )

    strategy = read_strategy()
    actual = metadata.num_actual_tokens
    if not actual:
        return output.zero_()
    if len(metadata.seq_lens_list) != 1:
        raise ValueError(
            "Uniform strategy integration currently requires a single request"
        )
    owner = current_owner()
    pipeline_level(owner)
    slots = metadata.slot_mapping[:actual]
    prepared = prepare_metadata(
        owner, metadata.block_tables, slots, metadata.seq_lens_list, 128, query.device
    )
    q = query[:actual].view(actual, -1, 128)
    k = key[:actual].view(actual, -1, 128)
    v = value[:actual].view(actual, -1, 128)
    result = output.view(-1, q.shape[1], 128)[:actual]
    heads = k.shape[1]
    page_bytes = int(128 * heads * 128 * strategy.element_bytes)
    if keys.ndim != 1 or values.shape != keys.shape or keys.numel() % page_bytes:
        raise ValueError("Strategy cache must be real, uniform flat byte pages")
    identity = (keys.data_ptr(), values.data_ptr())
    if identity not in STATES:
        blocks = keys.numel() // page_bytes
        codes = torch.full(
            (blocks,), strategy.code, dtype=torch.uint8, device=query.device
        )
        offsets = (
            torch.arange(blocks, dtype=torch.int32, device=query.device) * page_bytes
        )
        scales = torch.ones((blocks, heads), device=query.device)
        STATES[identity] = (codes, offsets, scales)
        print(
            "PSPKV_NVIDIA_STRATEGY "
            + json.dumps(
                {
                    "format": FORMATS[strategy.generation],
                    "precision": strategy.precision,
                    "adaptive": strategy.adaptive,
                    "bytes_per_element": strategy.element_bytes,
                    "physical_blocks": blocks,
                    "layout": "token_major_both",
                    "scales": "develop_shared_block_head_maxabs",
                    "native_writer": "direct_strided",
                    "decode_rope": "fused",
                    "decode_attention": os.environ.get(
                        "PSPKV_STRATEGY_DECODE_PATH", "fused"
                    ),
                    "shadow_cache": False,
                }
            ),
            flush=True,
        )
    codes, offsets, scales = STATES[identity]
    native = load_native_fp8_ops()
    # Prefill is unchunked, so its first token starts a new sequence lifetime.
    prefill = actual > 1
    if actual > 1 and metadata.seq_lens_list[0] != actual:
        raise NotImplementedError("Strategy comparison requires unchunked prefill")
    if prefill and strategy.code:
        sample = native.strategy_prefill_scales(k, v, prepared.slots, strategy.code)
        physical = (prepared.slots[::128] // 128).to(torch.int64)
        if strategy.adaptive:
            maxima = sample[:, :, 1]
            # develop derives the physical precision table from layer zero.
            # Share that choice across this forward, keeping per-layer scales.
            if not hasattr(owner, "_pspkv_strategy_selected_codes"):
                owner._pspkv_strategy_selected_codes = derive_develop_codes(maxima)
            selected = owner._pspkv_strategy_selected_codes
            codes.index_copy_(0, physical, selected)
            sizes = torch.where(codes == 3, page_bytes // 2, page_bytes)
            offsets.copy_((sizes.cumsum(0) - sizes).to(torch.int32))
            limits = torch.where(selected[:, None] == 3, 3.0, 240.0)
            selected_scales = maxima / limits
            selected_scales = torch.where(selected_scales < 1e-8, 1.0, selected_scales)
            scales.index_copy_(0, physical, selected_scales)
        else:
            scales.index_copy_(0, physical, sample[:, :, 0])
    rope = consume_deferred_rope(owner, q, k)
    if rope is not None:
        native.strategy_rope_scatter(
            q,
            k,
            v,
            rope[1],
            rope[2],
            keys,
            values,
            prepared.slots,
            codes,
            offsets,
            scales,
        )
    elif strategy.adaptive:
        native.strategy_scatter_codes(
            k, v, keys, values, prepared.slots, codes, offsets, scales
        )
    elif prefill and strategy.code <= 2:
        native.strategy_scatter_prefill_vec(
            k, v, keys, values, prepared.slots, offsets, scales, strategy.code
        )
    else:
        native.strategy_scatter_kv(
            k, v, keys, values, prepared.slots, offsets, scales, strategy.code
        )
    if actual == 1:
        full_graph = os.environ.get("PSPKV_ASCEND_FULL_GRAPH") == "1"
        tables = (
            metadata.block_tables[:1].to(torch.int32).contiguous()
            if full_graph
            else prepared.tables
        )
        if full_graph:
            if not prepared.borrowed_lengths:
                raise RuntimeError(
                    "Strategy graph requires runner-owned device lengths"
                )
            if getattr(owner, "capturing", False):
                GRAPH_INPUTS.add(graph_signature(prepared.source))
        e4, e5, _ = packed_lookup_tables(query.device)
        decode_path = os.environ.get("PSPKV_STRATEGY_DECODE_PATH", "fused")
        if decode_path == "cube":
            # Bound tile storage independently of context length. Layers on
            # the same stream consume it before the next layer reuses it.
            if not hasattr(owner, "_pspkv_cube_workspaces"):
                owner._pspkv_cube_workspaces = {}
            stream_key = (
                query.device,
                torch.npu.current_stream(query.device).npu_stream,
            )
            if stream_key not in owner._pspkv_cube_workspaces:
                owner._pspkv_cube_workspaces[stream_key] = torch.empty(
                    32 * 1024 * 1024, dtype=torch.uint8, device=query.device
                )
            load_cube_ops().attention(
                q.contiguous(),
                keys,
                values,
                tables,
                prepared.lengths,
                codes,
                offsets,
                scales,
                e4,
                e5,
                strategy_tile_permutation(query.device),
                result,
                owner._pspkv_cube_workspaces[stream_key],
                int(os.environ.get("PSPKV_ASCEND_SPLITS", "8")),
                softmax_scale,
            )
            return output
        if decode_path == "paged":
            workspace = _forward_workspace(q, tables, heads, 128, 128)
            if workspace is None:
                raise RuntimeError(
                    "Native attention requires forward-scoped scratch reuse"
                )
            native.gather_paged_bf16_vec_out(
                keys,
                values,
                tables,
                prepared.lengths,
                codes,
                offsets,
                scales,
                e4,
                e5,
                strategy_tile_permutation(query.device),
                3,
                heads,
                128,
                128,
                *workspace,
            )
            implementation.key_cache, implementation.value_cache = workspace
            paged_metadata = copy(metadata)
            paged_metadata.block_tables = _scratch_block_table(
                query.device, 1, tables.shape[1]
            )
            implementation.forward_paged_attention(q, paged_metadata, result)
            return output
        splits = int(os.environ.get("PSPKV_ASCEND_SPLITS", "8"))
        native.tile_fused_attention(
            q.contiguous(),
            keys,
            values,
            tables,
            prepared.lengths,
            codes,
            offsets,
            scales,
            e4,
            e5,
            strategy_tile_permutation(query.device),
            3,
            heads,
            128,
            softmax_scale,
            result,
            0,
            splits,
        )
    else:
        import torch_npu

        result.copy_(
            torch_npu.npu_fused_infer_attention_score(
                query=q.contiguous(),
                key=k.contiguous(),
                value=v.contiguous(),
                num_heads=q.shape[1],
                num_key_value_heads=heads,
                input_layout="TND",
                actual_seq_lengths=[actual],
                actual_seq_lengths_kv=[actual],
                atten_mask=causal_mask(query.device),
                sparse_mode=3,
                scale=softmax_scale,
            )[0]
        )
    return output
