# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Check direct K/V writes, untouched bytes, strides and graph metadata updates."""

# Closures are executed before the enclosing test iteration advances.
# ruff: noqa: B023

import argparse
import json
from pathlib import Path

import torch
import torch_npu  # noqa: F401

from tests.v1.attention.psp_kv_reference import pack_kv_cache
from vllm.v1.attention.backends.psp_kv_ascend_native import load_native_fp8_ops
from vllm.v1.attention.backends.psp_kv_ascend_ops import (
    _tile_permutation,
    packed_lookup_tables,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.npu.set_device(0)
    torch.manual_seed(20261026)
    torch.set_num_threads(2)
    native = load_native_fp8_ops()
    report = {"writes": [], "graph_replays": []}
    block, heads, dim, pages = 128, 4, 128, 4
    for generation in range(3):
        for pattern in ((1,), (2,), (0, 1, 2, 3)):
            codes_cpu = torch.tensor(
                [pattern[i % len(pattern)] for i in range(pages)], dtype=torch.uint8
            )
            dense = torch.zeros(pages, heads, block, dim, dtype=torch.bfloat16)
            kb, vb, offsets_cpu = pack_kv_cache(
                dense, dense, generation, codes_cpu, torch.bfloat16
            )
            codes, offsets = codes_cpu.npu(), offsets_cpu.npu()
            scales = torch.ones(pages, heads, device="npu") * 0.75
            for tokens in (0, 1, 3, 135, 257):
                for strided in (False, True):
                    source = torch.randn(tokens, 36, dim, device="npu").bfloat16()
                    key = source[:, 28:32]
                    value = source[:, 32:36]
                    if not strided:
                        key, value = key.contiguous(), value.contiguous()
                    slots = torch.randperm(pages * block)[:tokens].long()
                    if tokens > 1:
                        slots[0] = -1
                    slots = slots.npu()
                    old_k, old_v = (
                        torch.full_like(kb, 0x55).npu(),
                        torch.full_like(vb, 0x55).npu(),
                    )
                    new_k, new_v = old_k.clone(), old_v.clone()
                    packed = torch.stack((key, value), dim=1)
                    native.packed_scatter_parallel(
                        packed,
                        old_k,
                        old_v,
                        slots,
                        codes,
                        offsets,
                        scales,
                        block,
                        generation,
                    )
                    native.packed_scatter_kv(
                        key,
                        value,
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
                    assert torch.equal(old_k.cpu(), new_k.cpu())
                    assert torch.equal(old_v.cpu(), new_v.cpu())
                    report["writes"].append(
                        {
                            "generation": generation,
                            "pattern": pattern,
                            "tokens": tokens,
                            "strided": strided,
                        }
                    )
        # Capture the writer + tile attention once. Change lengths, page order,
        # slot and K/V content in place before every replay, including a new
        # request with a previously used length. Compare to a fresh eager call.
        codes = torch.ones(pages, dtype=torch.uint8, device="npu")
        offsets = torch.arange(pages, dtype=torch.int32, device="npu") * (
            block * heads * dim
        )
        scales = torch.ones(pages, heads, device="npu")
        key_cache = torch.zeros(
            pages * block * heads * dim, dtype=torch.uint8, device="npu"
        )
        value_cache = torch.zeros_like(key_cache)
        tables = torch.arange(pages, dtype=torch.int32, device="npu").view(1, pages)
        lengths = torch.tensor([1], dtype=torch.int32, device="npu")
        slots = torch.zeros(1, dtype=torch.int64, device="npu")
        key = torch.randn(1, heads, dim, device="npu").bfloat16()
        value = torch.randn_like(key)
        query = torch.randn(1, 28, dim, device="npu").bfloat16()
        output = torch.empty_like(query)
        e4, e5, _ = packed_lookup_tables(query.device)
        permutation = _tile_permutation(query.device, dim)

        def run():
            native.packed_scatter_kv(
                key,
                value,
                key_cache,
                value_cache,
                slots,
                codes,
                offsets,
                scales,
                block,
                generation,
            )
            native.tile_fused_attention(
                query,
                key_cache,
                value_cache,
                tables,
                lengths,
                codes,
                offsets,
                scales,
                e4,
                e5,
                permutation,
                generation,
                heads,
                block,
                dim**-0.5,
                output,
                0,
            )

        for _ in range(3):
            run()
        torch.npu.synchronize()
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            run()
        for i, length in enumerate((3, 127, 128, 129, 160, 257, 3)):
            order = torch.randperm(pages).int()
            tables.copy_(order.view(1, pages))
            lengths.fill_(length)
            slots.fill_(
                int(order[(length - 1) // block]) * block + (length - 1) % block
            )
            key.normal_()
            value.normal_()
            query.normal_()
            # Poison the output to check that attention overwrites every row.
            output.fill_(float("nan"))
            graph.replay()
            observed = output.cpu().clone()
            output.fill_(float("nan"))
            run()
            expected = output.cpu()
            torch.testing.assert_close(observed, expected, atol=0, rtol=0)
            assert torch.isfinite(observed).all()
            report["graph_replays"].append(
                {"generation": generation, "length": length, "step": i}
            )
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: len(v) for k, v in report.items()}), flush=True)


if __name__ == "__main__":
    main()
