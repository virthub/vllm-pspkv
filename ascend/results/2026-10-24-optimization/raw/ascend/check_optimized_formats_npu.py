# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Validate parallel scatter and active-page/native-attention optimization.

Compares full arena bytes (including untouched bytes) against the original
writer. Covers shuffled unique slots, skips, repeated calls, mixed precision,
partial pages, noncontiguous physical page order and multi-sequence decode.
"""

# Closures are measured immediately within the iteration that defines them.
# ruff: noqa: B023

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch_npu  # noqa: F401

from tests.v1.attention.psp_kv_reference import pack_kv_cache
from vllm.v1.attention.backends.psp_kv_ascend_native import load_native_fp8_ops
from vllm.v1.attention.backends.psp_kv_ascend_ops import PSPKVAscendPackedOps


def timed(call, repeats=20):
    for _ in range(3):
        call()
    torch.npu.synchronize()
    before = time.perf_counter()
    for _ in range(repeats):
        call()
    torch.npu.synchronize()
    return (time.perf_counter() - before) * 1000 / repeats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=20)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.npu.set_device(0)
    torch.manual_seed(20261024)
    native = load_native_fp8_ops()
    ops = PSPKVAscendPackedOps()
    block, heads, kv_heads, dim = args.block_size, 28, 4, args.head_dim
    report = {"writes": [], "attention": []}
    for generation, name in enumerate(("basic", "enhanced", "btkv")):
        for pattern in ((1,), (2,), (0, 1, 2, 3)):
            count = 8
            codes_cpu = torch.tensor(
                [pattern[i % len(pattern)] for i in range(count)], dtype=torch.uint8
            )
            shape = (count, kv_heads, block, dim)
            initial = torch.zeros(shape, dtype=torch.bfloat16)
            kb, vb, offsets_cpu = pack_kv_cache(
                initial, initial, generation, codes_cpu, torch.bfloat16
            )
            old_k, old_v, new_k, new_v = [
                torch.full_like(t, 0x55).to("npu") for t in (kb, vb, kb, vb)
            ]
            codes, offsets = codes_cpu.to("npu"), offsets_cpu.to("npu")
            scales = torch.ones(count, kv_heads, device="npu")
            for tokens in (3, 135, 257):
                slots_cpu = torch.randperm(count * block)[:tokens].to(torch.int64)
                slots_cpu[0] = -1
                slots = slots_cpu.to("npu")
                source = (
                    (torch.randn(tokens, 2, kv_heads, dim) * 0.5)
                    .clamp(-2.5, 2.5)
                    .bfloat16()
                    .to("npu")
                )
                native.packed_scatter(
                    source,
                    old_k,
                    old_v,
                    slots,
                    codes,
                    offsets,
                    scales,
                    tokens,
                    block,
                    generation,
                    True,
                )
                native.packed_scatter_parallel(
                    source,
                    new_k,
                    new_v,
                    slots,
                    codes,
                    offsets,
                    scales,
                    block,
                    generation,
                )
                torch.npu.synchronize()
                assert torch.equal(old_k.cpu(), new_k.cpu()), (
                    name,
                    pattern,
                    tokens,
                    "K",
                )
                assert torch.equal(old_v.cpu(), new_v.cpu()), (
                    name,
                    pattern,
                    tokens,
                    "V",
                )
                report["writes"].append(
                    {
                        "format": name,
                        "pattern": pattern,
                        "tokens": tokens,
                        "byte_equal": True,
                    }
                )
        for lengths in ([3], [135], [160], [512], [2048], [3, 135]):
            pages = math.ceil(max(lengths) / block)
            count = len(lengths) * pages
            codes_cpu = (
                torch.tensor([i % 4 for i in range(count)], dtype=torch.uint8)
                if lengths == [512]
                else torch.ones(count, dtype=torch.uint8)
            )
            key = (torch.randn(count, kv_heads, block, dim) * 0.5).bfloat16()
            value = (torch.randn_like(key.float()) * 0.5).bfloat16()
            kb, vb, offsets_cpu = pack_kv_cache(
                key, value, generation, codes_cpu, torch.bfloat16
            )
            table_cpu = torch.randperm(count).to(torch.int32).view(len(lengths), pages)
            lens_cpu = torch.tensor(lengths, dtype=torch.int32)
            k, v, codes, offsets, table, lens = [
                t.to("npu")
                for t in (kb, vb, codes_cpu, offsets_cpu, table_cpu, lens_cpu)
            ]
            scales = torch.ones(count, kv_heads, device="npu")
            query = torch.randn(len(lengths), heads, dim).bfloat16().to("npu")
            before, after = torch.empty_like(query), torch.empty_like(query)
            arguments = (
                generation,
                query,
                k,
                v,
                table,
                lens,
                codes,
                offsets,
                scales if generation else None,
                1 / math.sqrt(dim),
                kv_heads,
                dim,
            )

            def original():
                ops._attention(*arguments, before, block_size=block)

            def optimized():
                public = [query, k, v, table, lens, codes, offsets]
                if generation:
                    public.append(scales)
                if generation == 2:
                    public.append(
                        torch.zeros(count, 16, dtype=torch.int64, device="npu")
                    )
                public.extend((1 / math.sqrt(dim), kv_heads, dim, after))
                getattr(ops, "psp_attention_" + name)(
                    *public, block_size=block, context_lens_cpu=lens_cpu
                )

            original()
            if block != 128:
                try:
                    optimized()
                except ValueError as error:
                    assert "128-token" in str(error)
                else:
                    raise AssertionError("Unsupported native page size was accepted")
                report["attention"].append(
                    {
                        "format": name,
                        "lengths": lengths,
                        "unsupported_page_size_rejected": True,
                    }
                )
                continue
            optimized()
            torch.npu.synchronize()
            torch.testing.assert_close(
                before.cpu().float(), after.cpu().float(), atol=1e-2, rtol=1e-2
            )
            assert bool(torch.isfinite(after.cpu()).all())
            row = {
                "format": name,
                "lengths": lengths,
                "max_abs_error": (before.cpu().float() - after.cpu().float())
                .abs()
                .max()
                .item(),
                "precision_codes": codes_cpu.tolist(),
                "original_ms": timed(original, args.repeats),
                "optimized_ms": timed(optimized, args.repeats),
                "bf16_scratch_bytes": len(lengths)
                * pages
                * block
                * kv_heads
                * dim
                * 2
                * 2,
            }
            report["attention"].append(row)
            args.output.write_text(json.dumps(report, indent=2))
            print(json.dumps(row), flush=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(
        f"PASS: {len(report['writes'])} writer cases and "
        f"{len(report['attention'])} attention cases",
        flush=True,
    )


if __name__ == "__main__":
    main()
