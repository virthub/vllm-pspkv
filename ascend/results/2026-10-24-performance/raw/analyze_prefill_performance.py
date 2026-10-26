# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Measure the exact eager-prefill ingredients and the one-block writer."""

# Each closure is measured immediately before the loop advances.
# ruff: noqa: B023

import argparse
import json
import math
from pathlib import Path

import torch
import torch_npu
from analyze_attention_performance import measure

from vllm.v1.attention.backends.psp_kv_ascend_native import load_native_fp8_ops


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.npu.set_device(0)
    torch.manual_seed(20261024)
    native = load_native_fp8_ops()
    heads, kv_heads, dim, block = 28, 4, 128, 128
    scale = 1 / math.sqrt(dim)
    rows = []
    for length in (3, 135):
        pages = math.ceil(length / block)
        q, k, v = [
            torch.randn(length, h, dim).bfloat16().to("npu")
            for h in (heads, kv_heads, kv_heads)
        ]
        out = torch.empty_like(q)
        causal = torch.triu(torch.ones(2048, 2048, dtype=torch.bool), diagonal=1).to(
            "npu"
        )

        def optimized_prefill():
            return torch_npu.npu_fused_infer_attention_score(
                query=q,
                key=k,
                value=v,
                num_heads=heads,
                num_key_value_heads=kv_heads,
                input_layout="TND",
                actual_seq_lengths=[length],
                actual_seq_lengths_kv=[length],
                atten_mask=causal,
                sparse_mode=3,
                scale=scale,
            )[0]

        def eager_prefill():
            keys = k.float().repeat_interleave(heads // kv_heads, dim=1).transpose(0, 1)
            values = (
                v.float().repeat_interleave(heads // kv_heads, dim=1).transpose(0, 1)
            )
            scores = torch.bmm(q.float().transpose(0, 1), keys.transpose(1, 2)) * scale
            positions = torch.arange(length, device=q.device)
            scores.masked_fill_(positions[None, :] > positions[:, None], float("-inf"))
            out.copy_(
                torch.bmm(torch.softmax(scores, dim=-1), values)
                .transpose(0, 1)
                .to(out.dtype)
            )
            return out

        baseline = optimized_prefill()
        candidate = eager_prefill()
        error = (baseline.float() - candidate.float()).abs().max().item()
        torch.testing.assert_close(
            baseline.float(), candidate.float(), atol=1e-2, rtol=1e-2
        )
        source = torch.stack((k, v), dim=1).contiguous()
        codes = torch.ones(pages, dtype=torch.uint8, device="npu")
        offsets = torch.arange(pages, dtype=torch.int32, device="npu") * (
            block * kv_heads * dim
        )
        scales = torch.ones(pages, kv_heads, device="npu")
        slots = torch.arange(length, dtype=torch.int64, device="npu")
        key_bytes, value_bytes = [
            torch.empty(pages * block * kv_heads * dim, dtype=torch.uint8, device="npu")
            for _ in range(2)
        ]
        row = {
            "length": length,
            "prefill_output_max_abs_difference": error,
            "optimized_bf16_fia_prefill": measure(optimized_prefill, 10, 3),
            "current_eager_fp32_prefill": measure(eager_prefill, 10, 3),
            "formats": {},
        }
        for generation, name in enumerate(("basic", "enhanced", "btkv")):

            def write():
                native.packed_scatter(
                    source,
                    key_bytes,
                    value_bytes,
                    slots,
                    codes,
                    offsets,
                    scales,
                    length,
                    block,
                    generation,
                    False,
                )

            row["formats"][name] = measure(write, 10, 3)
        old_k, old_v = [
            torch.empty(pages, block, kv_heads, dim, dtype=torch.uint8, device="npu")
            for _ in range(2)
        ]

        def existing_parallel_writer():
            native.quantize_scatter_bf16(k, v, slots, codes, old_k, old_v)

        row["existing_token_major_parallel_fp8_writer_control"] = measure(
            existing_parallel_writer, 10, 3
        )
        rows.append(row)
        args.output.write_text(json.dumps({"cases": rows}, indent=2))
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
