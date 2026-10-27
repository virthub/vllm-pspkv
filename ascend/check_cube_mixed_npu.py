# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Check mixed physical precisions and two sequences against develop's oracle."""

import argparse
import importlib.util
import json
from pathlib import Path

import torch
import torch_npu  # noqa: F401

from vllm.v1.attention.backends.psp_kv_ascend_native import load_native_fp8_ops
from vllm.v1.attention.backends.psp_kv_ascend_ops import (
    _tile_permutation,
    packed_lookup_tables,
)
from vllm.v1.attention.backends.psp_kv_ascend_reference import decode_e2m1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--reference-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location(
        "develop_reference", args.reference_file
    )
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    torch.set_num_threads(2)
    torch.manual_seed(20261027)
    torch.npu.set_device(0)
    native = load_native_fp8_ops()
    torch.ops.load_library(str(args.library.resolve()))
    length, heads, kv_heads, dim, block = 700, 28, 4, 128, 128
    pages = (length + block - 1) // block
    physical = torch.randperm(pages).int()
    codes = torch.tensor([0, 1, 2, 3, 1, 3], dtype=torch.uint8)
    widths = torch.tensor([2, 1, 1, 0.5])[codes.long()]
    sizes = (widths * block * kv_heads * dim).int()
    offsets = sizes.cumsum(0).int() - sizes
    slots = (physical[:, None].long() * block + torch.arange(block)).flatten()[:length]
    key = torch.randn(length, kv_heads, dim).bfloat16()
    value = torch.randn_like(key.float()).bfloat16()
    scales = torch.ones(pages, kv_heads)
    limits = (1, 240, 57344, 3)
    for logical, page in enumerate(physical.tolist()):
        code = int(codes[page])
        if code:
            start, end = logical * block, (logical + 1) * block
            scales[page] = (
                torch.maximum(
                    key[start:end].float().abs().amax((0, 2)),
                    value[start:end].float().abs().amax((0, 2)),
                )
                / limits[code]
            )
    source = torch.empty(
        length, heads + 2 * kv_heads, dim, dtype=torch.bfloat16, device="npu"
    )
    source[:, heads : heads + kv_heads].copy_(key)
    source[:, heads + kv_heads :].copy_(value)
    codes_npu, offsets_npu, scales_npu = codes.npu(), offsets.npu(), scales.npu()
    k = torch.full((int(sizes.sum()),), 0xAA, dtype=torch.uint8, device="npu")
    v = torch.full_like(k, 0xAA)
    native.strategy_scatter_codes(
        source[:, heads : heads + kv_heads],
        source[:, heads + kv_heads :],
        k,
        v,
        slots.npu(),
        codes_npu,
        offsets_npu,
        scales_npu,
    )
    table = physical[None].expand(2, -1).contiguous()
    table_npu = table.npu()
    lengths = torch.tensor([700, 135], dtype=torch.int32)
    lens = lengths.npu()
    query_cpu = torch.randn(2, heads, dim).bfloat16()
    query, output = (
        query_cpu.npu(),
        torch.empty((2, heads, dim), dtype=torch.bfloat16, device="npu"),
    )
    e4, e5, _ = packed_lookup_tables(torch.device("npu:0"))
    byte = torch.arange(256, dtype=torch.int32)
    nibbles = torch.stack((byte & 15, byte >> 4), -1).flatten().byte()
    permutation = torch.cat(
        (
            _tile_permutation(torch.device("npu:0"), dim).cpu(),
            decode_e2m1(nibbles).view(torch.int32),
        )
    ).npu()
    workspace = torch.full((32 * 1024 * 1024,), 85, dtype=torch.uint8, device="npu")
    cache_k, cache_v = k.cpu(), v.cpu()
    rows = []
    for splits in (1, 4, 8, 16):
        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            torch.ops.psp_kv_ascend_cube.attention(
                query,
                k,
                v,
                table_npu,
                lens,
                codes_npu,
                offsets_npu,
                scales_npu,
                e4,
                e5,
                permutation,
                output,
                workspace,
                splits,
                dim**-0.5,
            )
        for changed in ((700, 135), (135, 700), (3, 511), (511, 700)):
            lens.copy_(torch.tensor(changed, dtype=torch.int32).npu())
            query_cpu = torch.randn_like(query_cpu)
            query.copy_(query_cpu)
            graph.replay()
            expected = reference.reference_paged_attention(
                query_cpu,
                cache_k,
                cache_v,
                table,
                torch.tensor(changed, dtype=torch.int32),
                1,
                codes,
                scales,
                offsets,
                kv_heads,
                block,
                dim**-0.5,
            )
            actual = output.cpu()
            torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)
            torch.testing.assert_close(k.cpu(), cache_k, atol=0, rtol=0)
            torch.testing.assert_close(v.cpu(), cache_v, atol=0, rtol=0)
            assert (workspace[28 * 1024 * 1024 :] == 85).all().item()
            rows.append(
                {
                    "splits": splits,
                    "lengths": changed,
                    "max_abs_error": (actual.float() - expected.float())
                    .abs()
                    .max()
                    .item(),
                }
            )
    args.output.write_text(
        json.dumps({"physical_codes": codes.tolist(), "cases": rows}, indent=2)
    )
    print(
        json.dumps(
            {
                "passed": len(rows),
                "max_abs_error": max(r["max_abs_error"] for r in rows),
            }
        )
    )


if __name__ == "__main__":
    main()
