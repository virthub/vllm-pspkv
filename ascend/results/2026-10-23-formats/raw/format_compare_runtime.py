# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Test-only adapter for running original packed formats in a live model.

The comparison launcher overlays the backend module in its own containers.
It replaces only that module's fixed-FP8 reference_attention callback.
Original public Basic/Enhanced/BTKV methods are called for single-token
decode, with their common attention block size changed from 16 to 128.
All formats use the same native writer, unit FP8 scales and eager prefill.
This adapter is not installed into the normal serving backend.
"""

import json
import math
import os

import torch

from vllm.v1.attention.backends.psp_kv_ascend_native import load_native_fp8_ops
from vllm.v1.attention.backends.psp_kv_ascend_ops import PSPKVAscendPackedOps

NAMES = ("basic", "enhanced", "btkv")
STATES = {}


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
    code = {"e4m3": 1, "e5m2": 2}[precision]
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
    identity = (k_cache.data_ptr(), v_cache.data_ptr())
    if identity not in STATES:
        num_blocks = k_cache.shape[0]
        page_bytes = block_size * kv_heads * dimension
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
                    "native_writer": "packed_scatter",
                    "decode_entry": "psp_attention_" + name,
                    "metadata_bytes": sum(
                        t.numel() * t.element_size() for t in STATES[identity]
                    ),
                }
            ),
            flush=True,
        )
    codes, offsets, scales, descriptors = STATES[identity]
    native = load_native_fp8_ops()
    native.packed_scatter(
        torch.stack((key[:actual], value[:actual]), dim=1).contiguous(),
        k_cache.view(-1),
        v_cache.view(-1),
        slot_mapping.to(torch.int64).contiguous(),
        codes,
        offsets,
        scales,
        actual,
        block_size,
        generation,
        False,
    )
    num_queries = query_ends[0]
    prefix_length = seq_lens[0] - num_queries
    if num_queries == 1:
        tables = (
            block_tables[:1, : math.ceil(seq_lens[0] / block_size)]
            .to(torch.int32)
            .contiguous()
        )
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
        getattr(ServingOps(), "psp_attention_" + name)(*args)
        return output
    if prefix_length:
        raise NotImplementedError("comparison adapter requires unchunked prefill")
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
