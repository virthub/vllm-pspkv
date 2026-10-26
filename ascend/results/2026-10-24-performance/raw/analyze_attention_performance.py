# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Stage timing of the actual three-format path versus native paged attention.

Use isolated synthetic tensors with Qwen2.5-7B head geometry. Device event
times include stream scheduling and are not pipeline utilization counters.
The optional torch_npu profiler provides a task timeline for a short trace.
"""

# Timed closures and profiler callbacks finish before their loop advances.
# ruff: noqa: B023

import argparse
import json
import math
import statistics
import time
from pathlib import Path

import torch
import torch_npu  # noqa: F401

from tests.v1.attention.psp_kv_reference import pack_kv_cache, reference_paged_attention
from vllm.v1.attention.backends.psp_kv_ascend_native import load_native_fp8_ops
from vllm.v1.attention.backends.psp_kv_ascend_ops import (
    PSPKVAscendPackedOps,
)


def measure(function, iterations, rounds):
    for _ in range(3):
        function()
    torch.npu.synchronize()
    walls, devices = [], []
    for _ in range(rounds):
        start, end = (
            torch.npu.Event(enable_timing=True),
            torch.npu.Event(enable_timing=True),
        )
        torch.npu.synchronize()
        before = time.perf_counter()
        start.record()
        for _ in range(iterations):
            function()
        end.record()
        torch.npu.synchronize()
        walls.append((time.perf_counter() - before) * 1000 / iterations)
        devices.append(start.elapsed_time(end) / iterations)
    return {"wall_ms": statistics.median(walls), "event_ms": statistics.median(devices)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace-dir", type=Path)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()
    torch.npu.set_device(0)
    native = load_native_fp8_ops()
    ops = PSPKVAscendPackedOps()
    torch.manual_seed(20261024)
    heads, kv_heads, dim, block = 28, 4, 128, 128
    softmax_scale = 1 / math.sqrt(dim)
    data = {"torch": str(torch.__version__), "cases": [], "scope": __doc__}
    trace_calls = {}
    for batch, length in ((1, 32), (1, 160), (1, 512), (1, 2048), (4, 160)):
        pages = math.ceil(length / block)
        count = batch * pages
        keys = (
            (torch.randn(count, kv_heads, block, dim) * 0.5).clamp(-2.5, 2.5).bfloat16()
        )
        values = (torch.randn_like(keys.float()) * 0.5).clamp(-2.5, 2.5).bfloat16()
        q_cpu = torch.randn(batch, heads, dim).bfloat16()
        table_cpu = torch.arange(count, dtype=torch.int32).view(batch, pages)
        lens_cpu = torch.full((batch,), length, dtype=torch.int32)
        codes_cpu = torch.ones(count, dtype=torch.uint8)
        scales_cpu = torch.ones(count, kv_heads)
        query, table, lengths, codes, scales = [
            t.to("npu") for t in (q_cpu, table_cpu, lens_cpu, codes_cpu, scales_cpu)
        ]
        out = torch.empty_like(query)
        dense_k = keys.permute(0, 2, 1, 3).contiguous().to("npu")
        dense_v = values.permute(0, 2, 1, 3).contiguous().to("npu")

        def stock():
            torch_npu._npu_paged_attention(
                query=query,
                key_cache=dense_k,
                value_cache=dense_v,
                num_kv_heads=kv_heads,
                num_heads=heads,
                scale_value=softmax_scale,
                block_table=table,
                context_lens=lens_cpu,
                out=out,
            )

        stock()
        dense_output = out.cpu().clone()
        dense_codes = torch.zeros_like(codes_cpu)
        dense_bytes_k, dense_bytes_v, dense_offsets = pack_kv_cache(
            keys, values, 0, dense_codes, torch.bfloat16
        )
        dense_reference = reference_paged_attention(
            q_cpu,
            dense_bytes_k,
            dense_bytes_v,
            table_cpu,
            lens_cpu,
            0,
            dense_codes,
            scales_cpu,
            dense_offsets,
            kv_heads,
            block,
            softmax_scale,
        )
        torch.testing.assert_close(
            dense_output.float(), dense_reference.float(), atol=1e-2, rtol=1e-2
        )
        row = {
            "batch": batch,
            "length": length,
            "query_heads": heads,
            "kv_heads": kv_heads,
            "dimension": dim,
            "block_size": block,
            "stock_bf16_attention": measure(stock, args.iterations, args.rounds),
            "formats": {},
        }
        for generation, name in enumerate(("basic", "enhanced", "btkv")):
            k_cpu, v_cpu, offsets_cpu = pack_kv_cache(
                keys, values, generation, codes_cpu, torch.bfloat16
            )
            reference = reference_paged_attention(
                q_cpu,
                k_cpu,
                v_cpu,
                table_cpu,
                lens_cpu,
                generation,
                codes_cpu,
                scales_cpu,
                offsets_cpu,
                kv_heads,
                block,
                softmax_scale,
            )
            k, v, offsets = [t.to("npu") for t in (k_cpu, v_cpu, offsets_cpu)]
            kernel_scales = scales if generation else torch.empty(0, device="npu")
            writer_input = torch.stack(
                [
                    torch.stack(
                        (
                            keys[
                                seq * pages + (length - 1) // block,
                                :,
                                (length - 1) % block,
                            ],
                            values[
                                seq * pages + (length - 1) // block,
                                :,
                                (length - 1) % block,
                            ],
                        )
                    )
                    for seq in range(batch)
                ]
            ).to("npu")
            slots = (
                torch.arange(batch, device="npu", dtype=torch.int64) * pages * block
                + length
                - 1
            )

            def scatter():
                native.packed_scatter(
                    writer_input,
                    k,
                    v,
                    slots,
                    codes,
                    offsets,
                    scales,
                    batch,
                    block,
                    generation,
                    False,
                )

            def scores():
                return native.packed_scores(
                    query,
                    k,
                    table,
                    lengths,
                    codes,
                    offsets,
                    kernel_scales,
                    generation,
                    kv_heads,
                    block,
                    softmax_scale,
                )

            score = scores()
            probabilities = torch.softmax(score, dim=-1)

            def weighted():
                return native.packed_weighted_value(
                    probabilities,
                    v,
                    table,
                    lengths,
                    codes,
                    offsets,
                    kernel_scales,
                    generation,
                    kv_heads,
                    dim,
                    block,
                    0,
                )

            result = weighted()

            def full():
                return ops._attention(
                    generation,
                    query,
                    k,
                    v,
                    table,
                    lengths,
                    codes,
                    offsets,
                    scales if generation else None,
                    softmax_scale,
                    kv_heads,
                    dim,
                    out,
                    block_size=block,
                )

            full()
            torch.testing.assert_close(
                out.cpu().float(), reference.float(), atol=1e-2, rtol=1e-2
            )
            entry = {
                "reference_max_abs_error": (out.cpu().float() - reference.float())
                .abs()
                .max()
                .item(),
                "stock_dense_relative_l2_difference": (
                    out.cpu().float() - dense_output.float()
                )
                .norm()
                .item()
                / dense_output.float().norm().item(),
                "scatter_one_token_per_sequence": measure(
                    scatter, args.iterations, args.rounds
                ),
                "qk": measure(scores, args.iterations, args.rounds),
                "softmax": measure(
                    lambda: torch.softmax(score, dim=-1), args.iterations, args.rounds
                ),
                "weighted_value": measure(weighted, args.iterations, args.rounds),
                "cast_copy": measure(
                    lambda: out.copy_(result.to(out.dtype)),
                    args.iterations,
                    args.rounds,
                ),
                "full_attention": measure(full, args.iterations, args.rounds),
            }
            if generation == 0:

                def staged():
                    ops.psp_attention_scaled_basic_vec(
                        query,
                        k,
                        v,
                        table,
                        lengths,
                        codes,
                        offsets,
                        scales,
                        softmax_scale,
                        kv_heads,
                        dim,
                        out,
                        block_size=block,
                    )

                staged()
                torch.testing.assert_close(
                    out.cpu().float(), reference.float(), atol=1e-2, rtol=1e-2
                )
                entry["existing_staged_control"] = measure(
                    staged, args.iterations, args.rounds
                )
            row["formats"][name] = entry
            if batch == 1 and length == 160 and args.trace_dir and generation == 0:
                # Bind current buffers: the loop below profiles this case before
                # any fixture variables are reassigned by the next generation.
                trace_calls = {
                    "stock_bf16": stock,
                    "packed_scatter": scatter,
                    "packed_qk": scores,
                    "softmax": lambda: torch.softmax(score, dim=-1),
                    "packed_weighted_v": weighted,
                    "cast_copy": lambda: out.copy_(result.to(out.dtype)),
                    "staged_attention": staged,
                }
                for function in trace_calls.values():
                    function()
                torch.npu.synchronize()
                config = torch_npu.profiler._ExperimentalConfig(
                    profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
                    aic_metrics=torch_npu.profiler.AiCMetrics.PipeUtilization,
                )
                with torch_npu.profiler.profile(
                    activities=[
                        torch_npu.profiler.ProfilerActivity.CPU,
                        torch_npu.profiler.ProfilerActivity.NPU,
                    ],
                    on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                        str(args.trace_dir)
                    ),
                    experimental_config=config,
                ):
                    for _ in range(3):
                        for label, function in trace_calls.items():
                            with torch.autograd.profiler.record_function(label):
                                function()
                    torch.npu.synchronize()
        data["cases"].append(row)
        args.output.write_text(json.dumps(data, indent=2))
        print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
