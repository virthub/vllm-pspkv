# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Use develop's reference to verify and time the four optimized inner precisions."""

# All timing callbacks are consumed before the next precision/length.
# ruff: noqa: B023

import argparse
import importlib.util
import json
import math
from pathlib import Path

import torch
import torch_npu
from analyze_attention_performance import measure

from vllm.v1.attention.backends.psp_kv_ascend_native import load_native_fp8_ops
from vllm.v1.attention.backends.psp_kv_ascend_ops import (
    _tile_permutation,
    packed_lookup_tables,
)
from vllm.v1.attention.backends.psp_kv_ascend_reference import decode_e2m1, encode_e2m1


def verify_encoders(native):
    bits = torch.arange(65536, dtype=torch.int32).to(torch.int16)
    values = bits.view(torch.bfloat16).view(512, 1, 128)
    source = values.npu()
    slots = torch.arange(512, dtype=torch.int64, device="npu")
    scales = torch.ones(4, 1, device="npu")
    rows = []
    for code in range(4):
        size = (256, 128, 128, 64)[code]
        offsets = torch.arange(4, device="npu", dtype=torch.int32) * 128 * size
        k = torch.zeros(512 * size, dtype=torch.uint8, device="npu")
        v = torch.zeros_like(k)
        native.strategy_scatter_kv(source, source, k, v, slots, offsets, scales, code)
        if code == 0:
            expected = values.reshape(-1).view(torch.uint8)
            torch.testing.assert_close(k.cpu(), expected, rtol=0, atol=0)
        elif code in (1, 2):
            dtype = torch.float8_e4m3fn if code == 1 else torch.float8_e5m2
            limit = 448 if code == 1 else 57344
            expected = (
                values.float()
                .clamp(-limit, limit)
                .to(dtype)
                .view(torch.uint8)
                .flatten()
            )
            mask = ~values.isnan().flatten()
            torch.testing.assert_close(k.cpu()[mask], expected[mask], rtol=0, atol=0)
        else:
            nibble = encode_e2m1(values.float()).flatten()
            nibble = torch.where((nibble & 7) == 6, (nibble & 8) | 5, nibble)
            expected = nibble[::2] | (nibble[1::2] << 4)
            torch.testing.assert_close(k.cpu(), expected, rtol=0, atol=0)
        torch.testing.assert_close(k.cpu(), v.cpu(), rtol=0, atol=0)
        rows.append({"code": code, "bf16_inputs": 65536, "passed": True})
    return rows


def measure_graph(function, iterations):
    for _ in range(3):
        function()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    # Record multiple calls per replay, amortizing Python graph submission.
    with torch.npu.graph(graph):
        for _ in range(16):
            function()
    result = measure(graph.replay, iterations, 5)
    return {key: value / 16 for key, value in result.items()}


def verify_rope(native):
    from vllm_ascend.ops.triton.rope import rope_forward_triton
    from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

    init_device_properties_triton()
    phase = torch.randn(512, 64)
    phases = torch.cat((phase.cos(), phase.sin()), dim=-1)
    cases = 0
    for code in range(4):
        page_bytes = int(128 * 4 * 128 * (2, 1, 1, 0.5)[code])
        offsets = torch.arange(4, device="npu", dtype=torch.int32) * page_bytes
        codes = torch.full((4,), code, device="npu", dtype=torch.uint8)
        scales = torch.full((4, 4), 0.5, device="npu")
        for dtype in (torch.float32, torch.bfloat16):
            rotation = phases.to(dtype).npu()
            for position in (0, 127, 129, 511):
                qkv = torch.randn(1, 36, 128, device="npu").bfloat16()
                q, k, v = qkv[:, :28], qkv[:, 28:32], qkv[:, 32:]
                pos = torch.tensor([position], dtype=torch.int64, device="npu")
                expected_q, expected_k = rope_forward_triton(
                    q.clone(),
                    k.clone(),
                    cos_sin_cache=rotation,
                    positions=pos,
                    rope_dim=128,
                    is_neox_style=True,
                )
                before_k = torch.full(
                    (4 * page_bytes,), 0x55, dtype=torch.uint8, device="npu"
                )
                before_v = before_k.clone()
                after_k, after_v = before_k.clone(), before_v.clone()
                native.strategy_scatter_kv(
                    expected_k.view(1, 4, 128),
                    v,
                    before_k,
                    before_v,
                    pos,
                    offsets,
                    scales,
                    code,
                )
                native.strategy_rope_scatter(
                    q,
                    k,
                    v,
                    pos,
                    rotation,
                    after_k,
                    after_v,
                    pos,
                    codes,
                    offsets,
                    scales,
                )
                torch.testing.assert_close(q.cpu(), expected_q.cpu(), rtol=0, atol=0)
                torch.testing.assert_close(
                    after_k.cpu(), before_k.cpu(), rtol=0, atol=0
                )
                torch.testing.assert_close(
                    after_v.cpu(), before_v.cpu(), rtol=0, atol=0
                )
                cases += 1
    return cases


def verify_adaptive(native):
    """Exercise mixed FP8/FP4 pages and the zero/tiny scale guard on device."""
    source = torch.zeros(384, 1, 128, dtype=torch.bfloat16, device="npu")
    source[128:256].fill_(0.125)
    source[256:].fill_(2)
    slots = torch.arange(384, dtype=torch.int64, device="npu")
    sample = native.strategy_prefill_scales(source, source, slots, 1)
    torch.testing.assert_close(
        sample[:, 0, 1].cpu(), torch.tensor([0.0, 0.125, 2.0]), rtol=0, atol=0
    )
    assert sample[0, 0, 0].item() == 1.0
    codes = torch.tensor([1, 3, 1], dtype=torch.uint8, device="npu")
    offsets = torch.tensor([0, 16384, 24576], dtype=torch.int32, device="npu")
    scales = torch.tensor([[1.0], [0.125 / 3], [2.0 / 240]], device="npu")
    k = torch.zeros(40960, dtype=torch.uint8, device="npu")
    v = torch.zeros_like(k)
    native.strategy_scatter_codes(source, source, k, v, slots, codes, offsets, scales)
    expected_k, expected_v = torch.zeros_like(k), torch.zeros_like(v)
    for page, code in enumerate((1, 3, 1)):
        part = slice(page * 128, (page + 1) * 128)
        native.strategy_scatter_kv(
            source[part],
            source[part],
            expected_k,
            expected_v,
            slots[part],
            offsets,
            scales,
            code,
        )
    torch.testing.assert_close(k.cpu(), expected_k.cpu(), rtol=0, atol=0)
    torch.testing.assert_close(v.cpu(), expected_v.cpu(), rtol=0, atol=0)
    return {"mixed_page_codes": [1, 3, 1], "passed": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lengths", default="135,1536,8192")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--cube-library", type=Path)
    parser.add_argument("--profile-cube", action="store_true")
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location(
        "develop_reference", args.reference_file
    )
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    torch.set_num_threads(2)
    torch.manual_seed(20261026)
    torch.npu.set_device(0)
    native = load_native_fp8_ops()
    cube_workspace = None
    if args.cube_library:
        torch.ops.load_library(str(args.cube_library.resolve()))
        cube_workspace = torch.full(
            (32 * 1024 * 1024,), 85, dtype=torch.uint8, device="npu"
        )
    report = {
        "encoders": verify_encoders(native),
        "rope_cases": verify_rope(native),
        "adaptive": verify_adaptive(native),
        "cases": [],
    }
    args.output.write_text(json.dumps(report, indent=2))
    heads, kv_heads, dim, block = 28, 4, 128, 128
    e4, e5, _ = packed_lookup_tables(torch.device("npu:0"))
    permutation = _tile_permutation(torch.device("npu:0"), dim)
    byte = torch.arange(256, dtype=torch.int32)
    nibbles = torch.stack((byte & 15, byte >> 4), dim=-1).flatten().to(torch.uint8)
    permutation = torch.cat(
        (permutation.cpu(), decode_e2m1(nibbles).view(torch.int32))
    ).npu()
    for length in map(int, args.lengths.split(",")):
        pages = math.ceil(length / block)
        physical = torch.randperm(pages).to(torch.int32)
        table = physical[None].npu()
        lens_cpu = torch.tensor([length], dtype=torch.int32)
        lens = lens_cpu.npu()
        slots_cpu = (physical[:, None].long() * block + torch.arange(block)).flatten()[
            :length
        ]
        slots = slots_cpu.npu()
        key_cpu = (torch.randn(length, kv_heads, dim) * 0.5).bfloat16()
        value_cpu = (torch.randn_like(key_cpu.float()) * 0.5).bfloat16()
        # Strided QKV views match the model's direct writer input.
        source = torch.empty(
            length, heads + 2 * kv_heads, dim, dtype=torch.bfloat16, device="npu"
        )
        key, value = source[:, heads : heads + kv_heads], source[:, heads + kv_heads :]
        key.copy_(key_cpu)
        value.copy_(value_cpu)

        def check_source(phase):
            torch.testing.assert_close(key.cpu(), key_cpu, atol=0, rtol=0, msg=phase)
            torch.testing.assert_close(
                value.cpu(), value_cpu, atol=0, rtol=0, msg=phase
            )

        query_cpu = torch.randn(1, heads, dim).bfloat16()
        query = query_cpu.npu()
        out = torch.empty_like(query)
        dense_k = torch.zeros(pages * block, kv_heads, dim, dtype=torch.bfloat16)
        dense_v = torch.zeros_like(dense_k)
        dense_k[slots_cpu] = key_cpu
        dense_v[slots_cpu] = value_cpu
        dense_k = dense_k.reshape(pages, block, kv_heads, dim).npu()
        dense_v = dense_v.reshape(pages, block, kv_heads, dim).npu()

        def baseline():
            torch_npu._npu_paged_attention(
                query=query,
                key_cache=dense_k,
                value_cache=dense_v,
                num_kv_heads=kv_heads,
                num_heads=heads,
                scale_value=dim**-0.5,
                block_table=table,
                context_lens=lens_cpu,
                out=out,
            )

        row = {
            "length": length,
            "bf16": measure(baseline, args.iterations, 5),
            "bf16_graph": measure_graph(baseline, args.iterations),
            "precisions": [],
        }
        baseline()
        bf16_output = out.cpu().float().clone()
        for code, name in enumerate(("lossless", "e4m3", "e5m2", "fp4")):
            if cube_workspace is not None:
                torch.testing.assert_close(key.cpu(), key_cpu, atol=0, rtol=0)
                torch.testing.assert_close(value.cpu(), value_cpu, atol=0, rtol=0)
            page_bytes = int(block * kv_heads * dim * (2, 1, 1, 0.5)[code])
            offsets = torch.arange(pages, dtype=torch.int32, device="npu") * page_bytes
            codes = torch.full((pages,), code, dtype=torch.uint8, device="npu")
            scales = torch.ones(pages, kv_heads, device="npu")
            k = torch.full((pages * page_bytes,), 0xAA, dtype=torch.uint8, device="npu")
            v = torch.full_like(k, 0xAA)
            if code:
                sample = native.strategy_prefill_scales(key, value, slots, code)
                scales.index_copy_(0, physical.long().npu(), sample[:, :, 0])
                limit = (0, 240, 57344, 3)[code]
                expected = []
                for start in range(0, length, block):
                    expected.append(
                        torch.maximum(
                            key_cpu[start : start + block].abs().float().amax((0, 2)),
                            value_cpu[start : start + block].abs().float().amax((0, 2)),
                        )
                        / limit
                    )
                torch.testing.assert_close(
                    sample[:, :, 0].cpu(), torch.stack(expected), atol=1e-7, rtol=1e-5
                )
            native.strategy_scatter_kv(key, value, k, v, slots, offsets, scales, code)
            expected = reference.reference_paged_attention(
                query_cpu,
                k.cpu(),
                v.cpu(),
                physical[None],
                lens_cpu,
                1,
                codes.cpu(),
                scales.cpu(),
                offsets.cpu(),
                kv_heads,
                block,
                dim**-0.5,
            )
            case = {
                "precision": name,
                "kv_bytes": 2 * pages * page_bytes,
                "relative_l2_vs_unquantized": None,
                "splits": [],
            }

            def write_prefill():
                native.strategy_scatter_kv(
                    key, value, k, v, slots, offsets, scales, code
                )

            case["prefill_writer"] = measure(write_prefill, args.iterations, 5)
            if code:
                case["prefill_scale_estimation"] = measure(
                    lambda: native.strategy_prefill_scales(key, value, slots, code),
                    args.iterations,
                    5,
                )
            scratch_k, scratch_v = torch.empty_like(dense_k), torch.empty_like(dense_v)
            canonical = torch.arange(pages, dtype=torch.int32, device="npu")[None]

            def decode():
                native.gather_paged_bf16_vec_out(
                    k,
                    v,
                    table,
                    lens,
                    codes,
                    offsets,
                    scales,
                    e4,
                    e5,
                    permutation,
                    3,
                    kv_heads,
                    dim,
                    block,
                    scratch_k,
                    scratch_v,
                )

            def staged():
                decode()
                torch_npu._npu_paged_attention(
                    query=query,
                    key_cache=scratch_k,
                    value_cache=scratch_v,
                    num_kv_heads=kv_heads,
                    num_heads=heads,
                    scale_value=dim**-0.5,
                    block_table=canonical,
                    context_lens=lens_cpu,
                    out=out,
                )

            staged()
            torch.testing.assert_close(out.cpu(), expected, atol=2e-2, rtol=2e-2)
            case["staged_paged"] = {
                "decode_graph": measure_graph(decode, args.iterations),
                "total_graph": measure_graph(staged, args.iterations),
                "max_abs_error": (out.cpu().float() - expected.float())
                .abs()
                .max()
                .item(),
            }
            if cube_workspace is not None:
                check_source("after staged attention")
            for splits in (1, 4, 8, 16):
                print("CHECK", length, name, splits, flush=True)

                def fused():
                    native.tile_fused_attention(
                        query,
                        k,
                        v,
                        table,
                        lens,
                        codes,
                        offsets,
                        scales,
                        e4,
                        e5,
                        permutation,
                        3,
                        kv_heads,
                        block,
                        dim**-0.5,
                        out,
                        0,
                        splits,
                    )

                fused()
                actual = out.cpu().float()
                torch.testing.assert_close(
                    actual, expected.float(), atol=2e-2, rtol=2e-2
                )
                case["relative_l2_vs_unquantized"] = (
                    (actual - bf16_output).norm() / bf16_output.norm()
                ).item()
                timing = measure(fused, args.iterations, 5)
                timing["graph"] = measure_graph(fused, args.iterations)
                timing.update(
                    splits=splits,
                    max_abs_error=(actual - expected.float()).abs().max().item(),
                )
                case["splits"].append(timing)
                if cube_workspace is not None:
                    check_source(f"after Vector attention, split {splits}")
            if cube_workspace is not None:
                case["cube"] = []
                saved_tails = []
                if length % block:
                    row_bytes = int(dim * (2, 1, 1, 0.5)[code])
                    for cache in (k, v):
                        for head in range(kv_heads):
                            start = (
                                int(physical[-1]) * page_bytes
                                + (head * block + length % block) * row_bytes
                            )
                            end = (
                                int(physical[-1]) * page_bytes
                                + (head + 1) * block * row_bytes
                            )
                            tail = cache[start:end]
                            saved_tails.append((tail, tail.clone()))
                            # Unused rows decode to NaNs in every supported
                            # format. Cube must zero them before QK and PV.
                            tail.fill_(255)
                cache_snapshots = (k.cpu(), v.cpu())
                for cube_splits in (1, 4, 8, 16):

                    def cube():
                        torch.ops.psp_kv_ascend_cube.attention(
                            query,
                            k,
                            v,
                            table,
                            lens,
                            codes,
                            offsets,
                            scales,
                            e4,
                            e5,
                            permutation,
                            out,
                            cube_workspace,
                            cube_splits,
                            dim**-0.5,
                        )

                    cube()
                    check_source("after one Cube call")
                    actual = out.cpu()
                    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
                    timing = measure(cube, args.iterations, 5)
                    check_source("after eager Cube timing")
                    timing.update(
                        splits=cube_splits,
                        graph=measure_graph(cube, args.iterations),
                        max_abs_error=(actual.float() - expected.float())
                        .abs()
                        .max()
                        .item(),
                    )
                    if args.profile_cube and length == 8192:
                        stats = torch.empty(
                            (kv_heads * cube_splits, 8), dtype=torch.int64, device="npu"
                        )
                        torch.ops.psp_kv_ascend_cube.attention(
                            query,
                            k,
                            v,
                            table,
                            lens,
                            codes,
                            offsets,
                            scales,
                            e4,
                            e5,
                            permutation,
                            out,
                            cube_workspace,
                            cube_splits,
                            dim**-0.5,
                            stats,
                        )
                        torch.testing.assert_close(
                            out.cpu(), expected, atol=2e-2, rtol=2e-2
                        )
                        host_stats = stats.cpu()
                        active_stats = host_stats[host_stats[:, 5] > 0]
                        totals = active_stats.sum(0).tolist()
                        timing["profile"] = {
                            "fields": [
                                "decode",
                                "qk_with_transport",
                                "softmax_with_transport",
                                "pv_with_transport",
                                "job_total",
                                "tiles",
                                "reserved",
                                "reserved",
                            ],
                            "active_jobs": active_stats.shape[0],
                            "cycle_sums": totals,
                            "rows": host_stats.tolist(),
                        }
                    case["cube"].append(timing)
                    check_source(f"after Cube attention, split {cube_splits}")
                    torch.testing.assert_close(
                        k.cpu(), cache_snapshots[0], atol=0, rtol=0
                    )
                    torch.testing.assert_close(
                        v.cpu(), cache_snapshots[1], atol=0, rtol=0
                    )
                    assert (cube_workspace[28 * 1024 * 1024 :] == 85).all().item()
                if length == 8192:
                    cube_graph = torch.npu.NPUGraph()
                    with torch.npu.graph(cube_graph):
                        cube()
                    for changed_length in (3, 135, 1536, 8192, 17, 8191):
                        lens.fill_(changed_length)
                        changed_query = torch.randn_like(query_cpu)
                        query.copy_(changed_query)
                        cube_graph.replay()
                        changed_reference = reference.reference_paged_attention(
                            changed_query,
                            k.cpu(),
                            v.cpu(),
                            physical[None],
                            torch.tensor([changed_length], dtype=torch.int32),
                            1,
                            codes.cpu(),
                            scales.cpu(),
                            offsets.cpu(),
                            kv_heads,
                            block,
                            dim**-0.5,
                        )
                        torch.testing.assert_close(
                            out.cpu(), changed_reference, atol=2e-2, rtol=2e-2
                        )
                    lens.fill_(length)
                    query.copy_(query_cpu)
                    case["cube_dynamic_graph_replays_checked"] = 6
                for tail, saved in saved_tails:
                    tail.copy_(saved)
                case["cube_poisoned_tail_checked"] = bool(saved_tails)
            row["precisions"].append(case)
            print(json.dumps({"length": length, **case}), flush=True)
            if length == 8192:
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph):
                    fused()
                for changed_length in (3, 135, 1536, 8192, 17, 8191):
                    lens.fill_(changed_length)
                    changed_query = torch.randn_like(query_cpu)
                    query.copy_(changed_query)
                    graph.replay()
                    changed_reference = reference.reference_paged_attention(
                        changed_query,
                        k.cpu(),
                        v.cpu(),
                        physical[None],
                        torch.tensor([changed_length], dtype=torch.int32),
                        1,
                        codes.cpu(),
                        scales.cpu(),
                        offsets.cpu(),
                        kv_heads,
                        block,
                        dim**-0.5,
                    )
                    torch.testing.assert_close(
                        out.cpu(), changed_reference, atol=2e-2, rtol=2e-2
                    )
                lens.fill_(length)
                query.copy_(query_cpu)
                case["dynamic_graph_replays_checked"] = 6
        report["cases"].append(row)
        args.output.write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
