# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Decompose the validated optimized path without changing its implementation."""

import argparse
import json
import math
import os
from pathlib import Path

import torch
import torch_npu
from analyze_attention_performance import measure
from format_compare_runtime import STATES, comparison_attention

from tests.v1.attention.psp_kv_reference import pack_kv_cache
from vllm.v1.attention.backends.psp_kv_ascend_native import load_native_fp8_ops
from vllm.v1.attention.backends.psp_kv_ascend_ops import PSPKVAscendPackedOps


def run_case(length, generation, trace_dir):
    block, heads, kv_heads, dim = 128, 28, 4, 128
    pages = math.ceil(length / block)
    name = ("basic", "enhanced", "btkv")[generation]
    native = load_native_fp8_ops()
    ops = PSPKVAscendPackedOps()
    shape = (pages, kv_heads, block, dim)
    key_cpu = (torch.randn(shape) * 0.5).clamp(-2.5, 2.5).bfloat16()
    value_cpu = (torch.randn(shape) * 0.5).clamp(-2.5, 2.5).bfloat16()
    codes_cpu = torch.ones(pages, dtype=torch.uint8)
    keys, values, offsets_cpu = pack_kv_cache(
        key_cpu, value_cpu, generation, codes_cpu, torch.bfloat16
    )
    k, v, offsets, codes = [t.to("npu") for t in (keys, values, offsets_cpu, codes_cpu)]
    query = torch.randn(1, heads, dim).bfloat16().to("npu")
    tables = torch.arange(pages, dtype=torch.int32, device="npu").view(1, pages)
    lengths_cpu = torch.tensor([length], dtype=torch.int32)
    lengths = lengths_cpu.to("npu")
    scales = torch.ones(pages, kv_heads, device="npu")
    kernel_scales = scales if generation else torch.empty(0, device="npu")
    out, scalar_out, fused_out = [torch.empty_like(query) for _ in range(3)]
    softmax_scale = 1 / math.sqrt(dim)

    def gather():
        return native.gather_paged_bf16(
            k,
            v,
            tables,
            lengths,
            codes,
            offsets,
            kernel_scales,
            generation,
            kv_heads,
            dim,
            block,
        )

    dense_k, dense_v = gather()

    def paged():
        torch_npu._npu_paged_attention(
            query=query,
            key_cache=dense_k,
            value_cache=dense_v,
            num_kv_heads=kv_heads,
            num_heads=heads,
            scale_value=softmax_scale,
            block_table=tables,
            context_lens=lengths_cpu,
            out=out,
        )

    arguments = (
        generation,
        query,
        k,
        v,
        tables,
        lengths,
        codes,
        offsets,
        scales if generation else None,
        softmax_scale,
        kv_heads,
        dim,
    )

    def scalar():
        ops._attention(*arguments, scalar_out, block_size=block)

    def optimized():
        ops._attention(
            *arguments, fused_out, block_size=block, context_lens_cpu=lengths_cpu
        )

    paged()
    scalar()
    optimized()
    torch.npu.synchronize()
    torch.testing.assert_close(
        out.cpu().float(), scalar_out.cpu().float(), atol=1e-2, rtol=1e-2
    )
    torch.testing.assert_close(fused_out.cpu(), out.cpu(), atol=1e-2, rtol=1e-2)
    last = length - 1
    key = key_cpu[last // block, :, last % block].unsqueeze(0).to("npu")
    value = value_cpu[last // block, :, last % block].unsqueeze(0).to("npu")
    source = torch.stack((key, value), dim=1).contiguous()
    slots = torch.tensor([last], dtype=torch.int64, device="npu")
    cache_k, cache_v = (
        k.view(pages, block, kv_heads, dim),
        v.view(pages, block, kv_heads, dim),
    )

    def scatter():
        native.packed_scatter_parallel(
            source, k, v, slots, codes, offsets, scales, block, generation
        )

    os.environ["PSPKV_COMPARE_FORMAT"] = name
    os.environ["PSPKV_COMPARE_PRECISION"] = "e4m3"
    os.environ["PSPKV_COMPARE_OPTIMIZED"] = "1"
    os.environ["PSPKV_COMPARE_PAGED_MIN_LENGTH"] = "64"
    STATES.clear()

    def callback():
        return comparison_attention(
            query,
            key,
            value,
            cache_k,
            cache_v,
            slots,
            tables,
            [1],
            [length],
            out,
            softmax_scale,
        )

    callback()
    torch.npu.synchronize()
    torch.testing.assert_close(
        out.cpu().float(), scalar_out.cpu().float(), atol=1e-2, rtol=1e-2
    )
    calls = {
        "gather_alloc_decode_write": gather,
        "native_paged_on_predecoded": paged,
        "optimized_wrapper_gather_plus_paged": optimized,
        "scalar_decode": scalar,
        "parallel_scatter_prepacked": scatter,
        "stack_kv": lambda: torch.stack((key, value), dim=1).contiguous(),
        "new_npu_length_tensor": lambda: torch.tensor(
            [length], dtype=torch.int32, device="npu"
        ),
        "new_cpu_length_tensor": lambda: torch.tensor(
            [length], dtype=torch.int32, device="cpu"
        ),
        "scratch_alloc_only": lambda: (
            torch.empty_like(dense_k),
            torch.empty_like(dense_v),
        ),
        "actual_decode_callback": callback,
    }
    result = {
        "length": length,
        "format": name,
        "selected_path": "scalar" if length < 64 else "gather_plus_paged",
        "bf16_scratch_bytes": 2 * dense_k.numel() * dense_k.element_size(),
        "stages": {label: measure(fn, 20, 5) for label, fn in calls.items()},
    }
    if trace_dir and length in (32, 160) and generation == 0:
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
                str(trace_dir / str(length))
            ),
            experimental_config=config,
        ):
            for _ in range(3):
                with torch.autograd.profiler.record_function(
                    "actual_optimized_callback"
                ):
                    callback()
                with torch.autograd.profiler.record_function(
                    "native_paged_without_gather"
                ):
                    paged()
            torch.npu.synchronize()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace-dir", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(20261024)
    torch.npu.set_device(0)
    report = {"cases": [], "scope": __doc__}
    for length in (32, 160, 512, 2048):
        for generation in range(3):
            row = run_case(length, generation, args.trace_dir)
            report["cases"].append(row)
            args.output.write_text(json.dumps(report, indent=2))
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
