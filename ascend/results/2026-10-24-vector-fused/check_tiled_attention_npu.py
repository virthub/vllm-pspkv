# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Validate and profile UB vector decode and optional tile-fused attention."""

# Closures are consumed within the current test case.
# ruff: noqa: B023

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import torch
import torch_npu
from analyze_attention_performance import measure

from tests.v1.attention.psp_kv_reference import pack_kv_cache
from vllm.v1.attention.backends.psp_kv_ascend_native import load_native_fp8_ops
from vllm.v1.attention.backends.psp_kv_ascend_ops import (
    PSPKVAscendPackedOps,
    _ATTENTION_IMPL,
    _tile_permutation,
    packed_lookup_tables,
)


def fixture(lengths, generation, dim=128, pattern="e4", nonunit=False):
    block, heads, kv_heads = 128, 28, 4
    pages = math.ceil(max(lengths) / block)
    count = pages * len(lengths)
    codes = torch.tensor(
        [
            (i % 4 if pattern == "mixed" else 2 if pattern == "e5" else 1)
            for i in range(count)
        ],
        dtype=torch.uint8,
    )
    key = (torch.randn(count, kv_heads, block, dim) * 0.3).clamp(-1.5, 1.5).bfloat16()
    value = (torch.randn_like(key.float()) * 0.3).clamp(-1.5, 1.5).bfloat16()
    k, v, offsets = pack_kv_cache(key, value, generation, codes, torch.bfloat16)
    tables = torch.randperm(count).to(torch.int32).view(len(lengths), pages)
    lens_cpu = torch.tensor(lengths, dtype=torch.int32)
    scales = (
        torch.rand(count, kv_heads) + 0.5 if nonunit else torch.ones(count, kv_heads)
    )
    k, v, tables, lens, codes, offsets, scales = [
        t.to("npu") for t in (k, v, tables, lens_cpu, codes, offsets, scales)
    ]
    query = torch.randn(len(lengths), heads, dim).bfloat16().to("npu")
    e4, e5, _ = packed_lookup_tables(query.device)
    permutation = _tile_permutation(query.device, dim)
    native = load_native_fp8_ops()
    common = (k, v, tables, lens, codes, offsets, scales)
    geometry = (generation, kv_heads, dim, block)

    def scalar_gather():
        return native.gather_paged_bf16(*common, *geometry)

    def vector_gather():
        return native.gather_paged_bf16_vec(*common, e4, e5, permutation, *geometry)

    dense_k, dense_v = scalar_gather()
    vector_k, vector_v = vector_gather()
    torch.npu.synchronize()
    torch.testing.assert_close(
        vector_k.cpu(), dense_k.cpu(), atol=0, rtol=0, equal_nan=True
    )
    torch.testing.assert_close(
        vector_v.cpu(), dense_v.cpu(), atol=0, rtol=0, equal_nan=True
    )
    output = torch.empty_like(query)
    scale = 1 / math.sqrt(dim)

    # Keep the canonical scratch table outside the measured closure.
    canonical = torch.arange(count, device="npu", dtype=torch.int32).view(
        len(lengths), pages
    )

    def native_paged():
        torch_npu._npu_paged_attention(
            query=query,
            key_cache=dense_k,
            value_cache=dense_v,
            num_kv_heads=kv_heads,
            num_heads=heads,
            scale_value=scale,
            block_table=canonical,
            context_lens=lens_cpu,
            out=output,
        )

    ops = PSPKVAscendPackedOps()

    def wrapper():
        ops._attention(
            generation,
            query,
            k,
            v,
            tables,
            lens,
            codes,
            offsets,
            scales if generation else None,
            scale,
            kv_heads,
            dim,
            output,
            block_size=block,
            context_lens_cpu=lens_cpu,
        )

    native_paged()
    torch.npu.synchronize()
    expected = output.cpu().clone()
    wrapper()
    torch.npu.synchronize()
    torch.testing.assert_close(output.cpu(), expected, atol=1e-2, rtol=1e-2)
    errors = {
        "public_max_abs": (output.cpu().float() - expected.float()).abs().max().item()
    }
    calls = {
        "scalar_gather": scalar_gather,
        "vector_gather": vector_gather,
        "native_paged": native_paged,
        "public_wrapper": wrapper,
    }
    if hasattr(native, "tile_fused_attention"):

        def fused():
            native.tile_fused_attention(
                query,
                *common,
                e4,
                e5,
                permutation,
                generation,
                kv_heads,
                block,
                scale,
                output,
                0,
            )

        fused()
        torch.npu.synchronize()
        torch.testing.assert_close(output.cpu(), expected, atol=1e-2, rtol=1e-2)
        errors["fused_max_abs"] = (
            (output.cpu().float() - expected.float()).abs().max().item()
        )
        calls["tile_fused"] = fused

    def profile_fused():
        return native.tile_fused_attention(
            query,
            *common,
            e4,
            e5,
            permutation,
            generation,
            kv_heads,
            block,
            scale,
            output,
            1,
        )

    return calls, profile_fused if "tile_fused" in calls else None, errors


def check_all_codes():
    native = load_native_fp8_ops()
    rows = []
    for dim in (128, 256):
        for code in (1, 2):
            for generation in range(3):
                size = 128 * 4 * dim
                cache = (torch.arange(size) % 256).byte().to("npu")
                table = torch.zeros(1, 1, dtype=torch.int32, device="npu")
                lengths = torch.tensor([128], dtype=torch.int32, device="npu")
                codes = torch.tensor([code], dtype=torch.uint8, device="npu")
                offsets = torch.zeros(1, dtype=torch.int32, device="npu")
                scales = torch.full((1, 4), 1.25, device="npu")
                e4, e5, _ = packed_lookup_tables(cache.device)
                args = (cache, cache, table, lengths, codes, offsets, scales)
                old = native.gather_paged_bf16(*args, generation, 4, dim, 128)
                new = native.gather_paged_bf16_vec(
                    *args,
                    e4,
                    e5,
                    _tile_permutation(cache.device, dim),
                    generation,
                    4,
                    dim,
                    128,
                )
                for before, after in zip(old, new):
                    before, after = before.cpu(), after.cpu()
                    mask = ~torch.isnan(before)
                    assert torch.equal(
                        before.view(torch.int16)[mask], after.view(torch.int16)[mask]
                    )
                    assert torch.equal(torch.isnan(before), torch.isnan(after))
                rows.append({"dim": dim, "code": code, "generation": generation})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace-dir", type=Path)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(20261025)
    torch.npu.set_device(0)
    report = {
        "checks": [],
        "timings": [],
        "all_codes": check_all_codes(),
        "attention_impl": _ATTENTION_IMPL,
        "torch": str(torch.__version__),
        "library_sha256": hashlib.sha256(
            Path(os.environ["PSPKV_ASCEND_NATIVE_LIB"]).read_bytes()
        ).hexdigest(),
    }
    for dim in (128,) if args.quick else (128, 256):
        for generation in range(3):
            for pattern in ("e4",) if args.quick else ("e4", "e5", "mixed"):
                for lengths in (
                    [3],
                    [135],
                    [512],
                    [3, 135],
                    [31, 33],
                    [127, 129],
                    [255, 257],
                ):
                    _, _, errors = fixture(
                        lengths, generation, dim, pattern, nonunit=generation != 0
                    )
                    report["checks"].append(
                        {
                            "dim": dim,
                            "generation": generation,
                            "pattern": pattern,
                            "lengths": lengths,
                            **errors,
                        }
                    )
    for length in (32, 160, 512, 2048):
        for generation in range(3):
            calls, profile_fused, errors = fixture([length], generation)
            row = {
                "length": length,
                "format": ("basic", "enhanced", "btkv")[generation],
                **errors,
                "stages": {name: measure(fn, 20, 5) for name, fn in calls.items()},
            }
            report["timings"].append(row)
            if profile_fused is not None:
                cycles = profile_fused().cpu()
                # 910B system counter is 50 MHz. Profiling adds barriers and
                # excludes one-time LUT/transpose-map initialization.
                row["instrumented_fused_per_head_us"] = (
                    cycles[:, :5].double() / 50
                ).tolist()
            args.output.write_text(json.dumps(report, indent=2))
            print(json.dumps(row), flush=True)
            if args.trace_dir and generation == 0 and length in (32, 160, 2048):
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
                        str(args.trace_dir / str(length))
                    ),
                    experimental_config=config,
                ):
                    for _ in range(3):
                        for name, fn in calls.items():
                            with torch.autograd.profiler.record_function(name):
                                fn()
                    torch.npu.synchronize()
    args.output.write_text(json.dumps(report, indent=2))
    print(
        f"PASS: {len(report['checks'])} numerical cases, "
        f"{len(report['timings'])} timed cases",
        flush=True,
    )


if __name__ == "__main__":
    main()
