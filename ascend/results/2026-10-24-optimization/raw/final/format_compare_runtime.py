# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Test-only adapter for running original packed formats in a live model.

The comparison launcher overlays the backend module in its own containers.
It replaces only that module's fixed-FP8 reference_attention callback.
Original public Basic/Enhanced/BTKV methods are called for single-token
decode, with their common attention block size changed from 16 to 128.
Default comparisons use the same native writer, unit FP8 scales and eager
prefill. Optional BF16 output-quality controls use two logical shadow pages
because the live plugin only allocates FP8 arenas. Their advertised engine
capacity and latency must not be interpreted as BF16 resource benchmarks.
This adapter is not installed into the normal serving backend.
"""

import json
import math
import os
from functools import lru_cache

import torch

from vllm.v1.attention.backends.psp_kv_ascend_native import load_native_fp8_ops
from vllm.v1.attention.backends.psp_kv_ascend_ops import PSPKVAscendPackedOps

NAMES = ("basic", "enhanced", "btkv")
STATES = {}
SHADOW_CACHES = {}


@lru_cache(maxsize=2)
def _causal_mask(device):
    return torch.ones((2048, 2048), dtype=torch.bool).triu(1).to(device)


class ServingOps(PSPKVAscendPackedOps):
    """Preserve public methods and select the live plugin's 128-token pages."""

    @staticmethod
    def _attention(*args, **kwargs):
        kwargs["block_size"] = 128
        return PSPKVAscendPackedOps._attention(*args, **kwargs)


def comparison_attention(
    query,
    key,
    value,
    k_cache,
    v_cache,
    slot_mapping,
    block_tables,
    query_ends,
    seq_lens,
    output,
    scale,
    **kwargs,
):
    name = os.environ["PSPKV_COMPARE_FORMAT"]
    generation = NAMES.index(name)
    precision = os.environ.get("PSPKV_COMPARE_PRECISION", "e4m3")
    code = {"bf16": 0, "e4m3": 1, "e5m2": 2}[precision]
    optimized = os.environ.get("PSPKV_COMPARE_OPTIMIZED", "0") == "1"
    paged_min_length = int(os.environ.get("PSPKV_COMPARE_PAGED_MIN_LENGTH", "0"))
    assert len(query_ends) == len(seq_lens) == 1
    assert k_cache.shape == v_cache.shape and k_cache.shape[1] == 128
    assert k_cache.dtype == v_cache.dtype == torch.uint8
    assert query.dtype == key.dtype == value.dtype == torch.bfloat16
    block_size, kv_heads, dimension = k_cache.shape[1:]
    actual = len(slot_mapping)
    result = output.view(-1, query.shape[1], dimension)
    result.zero_()
    if not actual:
        return output
    assert query_ends[-1] == actual
    num_queries = query_ends[0]
    prefix_length = seq_lens[0] - num_queries
    identity = (k_cache.data_ptr(), v_cache.data_ptr())
    if identity not in STATES:
        # The engine only accepts FP8 arenas. This small BF16 shadow is an
        # output-quality control for one sequence <=256 tokens, not a capacity
        # or speed benchmark. Remap its two pages to logical slots below.
        num_blocks = 2 if code == 0 else k_cache.shape[0]
        page_bytes = block_size * kv_heads * dimension * (2 if code == 0 else 1)
        if code == 0:
            SHADOW_CACHES[identity] = tuple(
                torch.empty(
                    num_blocks * page_bytes, dtype=torch.uint8, device=query.device
                )
                for _ in range(2)
            )
        codes = torch.full((num_blocks,), code, dtype=torch.uint8, device=query.device)
        offsets = (
            torch.arange(num_blocks, dtype=torch.int64, device=query.device)
            * page_bytes
        ).to(torch.int32)
        scales = torch.ones((num_blocks, kv_heads), device=query.device)
        descriptors = (
            torch.zeros((num_blocks, 16), dtype=torch.int64, device=query.device)
            if generation == 2
            else torch.empty(0, dtype=torch.int64, device=query.device)
        )
        STATES[identity] = (codes, offsets, scales, descriptors)
        print(
            "PSPKV_FORMAT_COMPARISON "
            + json.dumps(
                {
                    "format": name,
                    "precision": precision,
                    "block_size": block_size,
                    "physical_blocks": num_blocks,
                    "unit_scales": True,
                    "bf16_shadow_control": code == 0,
                    "native_writer": (
                        "packed_scatter_parallel"
                        if optimized and code in (1, 2)
                        else "packed_scatter"
                    ),
                    "optimized": optimized,
                    "paged_min_length": paged_min_length,
                    "attention_path": (
                        "length_dispatch_scalar_or_native_paged"
                        if optimized and paged_min_length
                        else (
                            "active_bf16_scratch_then_native_paged"
                            if optimized
                            else "scalar_packed_attention"
                        )
                    ),
                    "decode_entry": "psp_attention_" + name,
                    "metadata_bytes": sum(
                        t.numel() * t.element_size() for t in STATES[identity]
                    ),
                }
            ),
            flush=True,
        )
        if os.environ.get("PSPKV_COMPARE_DIAGNOSTICS") == "1":
            print(
                "PSPKV_FORMAT_INPUT "
                + json.dumps(
                    {
                        "format": name,
                        "precision": precision,
                        "tokens": actual,
                        "key_max_abs": key[:actual].abs().max().item(),
                        "value_max_abs": value[:actual].abs().max().item(),
                        "key_abs_ge_248": (key[:actual].abs() >= 248).sum().item(),
                        "value_abs_ge_248": (value[:actual].abs() >= 248).sum().item(),
                    }
                ),
                flush=True,
            )
    codes, offsets, scales, descriptors = STATES[identity]
    if code == 0:
        assert seq_lens[0] <= 256
        k_cache, v_cache = SHADOW_CACHES[identity]
        slot_mapping = torch.arange(
            prefix_length, prefix_length + actual, device=query.device
        )
    native = load_native_fp8_ops()
    source = torch.stack((key[:actual], value[:actual]), dim=1).contiguous()
    slots = slot_mapping.to(torch.int64).contiguous()
    if optimized and code in (1, 2):
        native.packed_scatter_parallel(
            source,
            k_cache.view(-1),
            v_cache.view(-1),
            slots,
            codes,
            offsets,
            scales,
            block_size,
            generation,
        )
    else:
        native.packed_scatter(
            source,
            k_cache.view(-1),
            v_cache.view(-1),
            slots,
            codes,
            offsets,
            scales,
            actual,
            block_size,
            generation,
            False,
        )
    if num_queries == 1:
        tables = (
            block_tables[:1, : math.ceil(seq_lens[0] / block_size)]
            .to(torch.int32)
            .contiguous()
        )
        if code == 0:
            tables = torch.arange(
                math.ceil(seq_lens[0] / block_size),
                dtype=torch.int32,
                device=query.device,
            ).unsqueeze(0)
        lengths = torch.tensor(seq_lens, dtype=torch.int32, device=query.device)
        args = [
            query[:1].contiguous(),
            k_cache.view(-1),
            v_cache.view(-1),
            tables,
            lengths,
            codes,
            offsets,
        ]
        if generation:
            args.append(scales)
        if generation == 2:
            args.append(descriptors)
        args.extend((scale, kv_heads, dimension, result[:1]))
        extra = (
            {
                "context_lens_cpu": torch.tensor(
                    seq_lens, dtype=torch.int32, device="cpu"
                )
            }
            if optimized and code in (1, 2) and seq_lens[0] >= paged_min_length
            else {}
        )
        getattr(ServingOps(), "psp_attention_" + name)(*args, **extra)
        return output
    if prefix_length:
        raise NotImplementedError("comparison adapter requires unchunked prefill")
    if optimized:
        import torch_npu

        result[:actual].copy_(
            torch_npu.npu_fused_infer_attention_score(
                query=query[:actual].contiguous(),
                key=key[:actual].contiguous(),
                value=value[:actual].contiguous(),
                num_heads=query.shape[1],
                num_key_value_heads=kv_heads,
                input_layout="TND",
                actual_seq_lengths=[actual],
                actual_seq_lengths_kv=[actual],
                atten_mask=_causal_mask(query.device),
                sparse_mode=3,
                scale=scale,
            )[0]
        )
        return output
    groups = query.shape[1] // kv_heads
    keys = key[:actual].float().repeat_interleave(groups, dim=1).transpose(0, 1)
    values = value[:actual].float().repeat_interleave(groups, dim=1).transpose(0, 1)
    scores = (
        torch.bmm(query[:actual].float().transpose(0, 1), keys.transpose(1, 2)) * scale
    )
    positions = torch.arange(actual, device=query.device)
    scores.masked_fill_(positions[None, :] > positions[:, None], float("-inf"))
    result[:actual].copy_(
        torch.bmm(torch.softmax(scores, dim=-1), values)
        .transpose(0, 1)
        .to(output.dtype)
    )
    return output
