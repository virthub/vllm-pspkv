# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Compare the original Basic/Enhanced/BTKV NPU operator entry points.

Keep logical Q/K/V, physical block mappings, precision codes and unit scales
identical. Only the cache layout and public attention entry point change.
Timings cover attention (native score/value kernels and PyTorch softmax),
not packing, allocation, the precision policy, or a complete model.
"""

import argparse
import json
import math
import random
import statistics
import time
from pathlib import Path

import torch

from tests.v1.attention.psp_kv_reference import (
    pack_kv_cache,
    reference_paged_attention,
)
from vllm.v1.attention.backends.psp_kv_ascend_native import load_native_fp8_ops
from vllm.v1.attention.backends.psp_kv_ascend_ops import PSPKVAscendPackedOps

FORMATS = ("basic", "enhanced", "btkv")
PRECISIONS = {
    "bf16": (0,),
    "e4m3": (1,),
    "e5m2": (2,),
    "fp4": (3,),
    "mixed_fp8": (1, 2),
    "mixed_all": (0, 1, 2, 3),
}
WORKLOADS = (
    ("gqa_128", [128], 16, 8),
    ("gqa_512", [512], 16, 8),
    ("gqa_2048", [2048], 16, 8),
    ("mha_512", [512], 8, 8),
    ("batch4_partial", [17, 32, 63, 112], 8, 2),
)


def invoke(ops, generation, fixture, output):
    query, keys, values, tables, lengths, codes, offsets, scales, descriptors = fixture
    args = [query, keys, values, tables, lengths, codes, offsets]
    if generation:
        args.append(scales)
    if generation == 2:
        args.append(descriptors)
    args.extend(
        (1 / math.sqrt(query.shape[-1]), scales.shape[1], query.shape[-1], output)
    )
    getattr(ops, "psp_attention_" + FORMATS[generation])(*args)


def prepare(lengths, heads, kv_heads, pattern, seed):
    generator = torch.Generator().manual_seed(seed)
    block_size, dimension = 16, 128
    max_blocks = math.ceil(max(lengths) / block_size)
    needed = [math.ceil(length / block_size) for length in lengths]
    num_blocks = sum(needed) + 2
    permutation = torch.randperm(num_blocks, generator=generator)
    tables = torch.zeros(len(lengths), max_blocks, dtype=torch.int32)
    cursor = 0
    for seq, count in enumerate(needed):
        tables[seq, :count] = permutation[cursor : cursor + count]
        cursor += count
    codes = torch.tensor(
        [pattern[i % len(pattern)] for i in range(num_blocks)], dtype=torch.uint8
    )
    shape = (num_blocks, kv_heads, block_size, dimension)
    # Unit-scale FP4 stays finite, and all three formats see the same values.
    key = (torch.randn(shape, generator=generator) * 0.5).clamp(-2.5, 2.5)
    value = (torch.randn(shape, generator=generator) * 0.5).clamp(-2.5, 2.5)
    key, value = key.bfloat16(), value.bfloat16()
    query = torch.randn(len(lengths), heads, dimension, generator=generator).bfloat16()
    scales = torch.ones(num_blocks, kv_heads)
    context = torch.tensor(lengths, dtype=torch.int32)
    dense_keys, dense_values, dense_offsets = pack_kv_cache(
        key, value, 0, torch.zeros_like(codes), torch.bfloat16
    )
    dense = reference_paged_attention(
        query,
        dense_keys,
        dense_values,
        tables,
        context,
        0,
        torch.zeros_like(codes),
        scales,
        dense_offsets,
        kv_heads,
        block_size,
        1 / math.sqrt(dimension),
    )
    fixtures, references = [], []
    for generation in range(3):
        keys, values, offsets = pack_kv_cache(
            key, value, generation, codes, torch.bfloat16
        )
        references.append(
            reference_paged_attention(
                query,
                keys,
                values,
                tables,
                context,
                generation,
                codes,
                scales,
                offsets,
                kv_heads,
                block_size,
                1 / math.sqrt(dimension),
            )
        )
        fixtures.append(
            tuple(
                t.to("npu:0").contiguous()
                for t in (
                    query,
                    keys,
                    values,
                    tables,
                    context,
                    codes,
                    offsets,
                    scales,
                    torch.zeros(num_blocks, 16, dtype=torch.int64),
                )
            )
        )
    return fixtures, references, dense, codes.tolist()


def main():
    import torch_npu  # noqa: F401

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    torch.npu.set_device(0)
    load_native_fp8_ops()
    ops = PSPKVAscendPackedOps()
    result = {
        "method": "unit scales; BF16 query/output; shuffled physical pages; "
        "same logical Q/K/V and physical precision table",
        "timing": "each sample is a mean over repeated calls, not a request percentile",
        "rounds": args.rounds,
        "iterations": args.iterations,
        "warmups": args.warmups,
        "torch": torch.__version__,
        "cases": [],
    }
    workloads = WORKLOADS[:1] if args.quick else WORKLOADS
    modes = {"mixed_all": PRECISIONS["mixed_all"]} if args.quick else PRECISIONS
    ordering = random.Random(20261023)
    for workload, lengths, heads, kv_heads in workloads:
        for precision, pattern in modes.items():
            fixtures, references, dense, codes = prepare(
                lengths, heads, kv_heads, pattern, seed=20261023
            )
            outputs = [torch.empty_like(f[0]) for f in fixtures]
            row = {
                "workload": workload,
                "context_lengths": lengths,
                "query_heads": heads,
                "kv_heads": kv_heads,
                "head_dim": 128,
                "block_size": 16,
                "precision": precision,
                "physical_precision_codes": codes,
                "formats": {},
            }
            for generation, name in enumerate(FORMATS):
                invoke(ops, generation, fixtures[generation], outputs[generation])
                torch.npu.synchronize()
                output = outputs[generation].cpu().float()
                reference = references[generation].float()
                torch.testing.assert_close(output, reference, atol=1e-2, rtol=1e-2)
                assert bool(torch.isfinite(output).all())
                norm = max(dense.float().norm().item(), 1e-12)
                row["formats"][name] = {
                    "reference_max_abs_error": (output - reference).abs().max().item(),
                    "dense_bf16_relative_l2_error": (output - dense.float())
                    .norm()
                    .item()
                    / norm,
                    "cache_bytes": fixtures[generation][1].numel()
                    + fixtures[generation][2].numel(),
                    "wall_ms_samples": [],
                    "device_ms_samples": [],
                }
            cpu_outputs = [output.cpu() for output in outputs]
            row["basic_enhanced_bit_equal"] = torch.equal(
                cpu_outputs[0], cpu_outputs[1]
            )
            row["enhanced_btkv_bit_equal"] = torch.equal(cpu_outputs[1], cpu_outputs[2])
            for generation in range(3):
                for _ in range(args.warmups):
                    invoke(ops, generation, fixtures[generation], outputs[generation])
            torch.npu.synchronize()
            for _ in range(args.rounds):
                order = list(range(3))
                ordering.shuffle(order)
                for generation in order:
                    start = torch.npu.Event(enable_timing=True)
                    end = torch.npu.Event(enable_timing=True)
                    torch.npu.synchronize()
                    wall_start = time.perf_counter()
                    start.record()
                    for _ in range(args.iterations):
                        invoke(
                            ops, generation, fixtures[generation], outputs[generation]
                        )
                    end.record()
                    torch.npu.synchronize()
                    wall = (time.perf_counter() - wall_start) * 1000 / args.iterations
                    device = start.elapsed_time(end) / args.iterations
                    stats = row["formats"][FORMATS[generation]]
                    stats["wall_ms_samples"].append(wall)
                    stats["device_ms_samples"].append(device)
            for stats in row["formats"].values():
                stats["wall_ms_median"] = statistics.median(stats["wall_ms_samples"])
                stats["device_ms_median"] = statistics.median(
                    stats["device_ms_samples"]
                )
            result["cases"].append(row)
            args.output.write_text(json.dumps(result, indent=2))
            print(
                json.dumps(
                    {
                        "workload": workload,
                        "precision": precision,
                        "wall_ms": {
                            name: stats["wall_ms_median"]
                            for name, stats in row["formats"].items()
                        },
                        "cross_format_bit_equal": row["basic_enhanced_bit_equal"]
                        and row["enhanced_btkv_bit_equal"],
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
