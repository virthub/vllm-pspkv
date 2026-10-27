# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import json
from pathlib import Path

import torch
import torch_npu  # noqa: F401

# Timed closures finish before the next fixture.
# ruff: noqa: B023
from vllm.v1.attention.backends.psp_kv_ascend_native import load_native_fp8_ops

parser = argparse.ArgumentParser(
    description="Check and time tiled FP8 writes against the scalar oracle."
)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
torch.manual_seed(20261027)
native = load_native_fp8_ops()
torch.npu.set_device(0)
rows = []
# Same bytes as scalar oracle over every BF16 pattern, then real scaled strided tiles.
for length in (512, 3, 135, 8192):
    heads = 1 if length == 512 else 4
    if length == 512:
        x = (
            torch.arange(65536, dtype=torch.int32)
            .to(torch.int16)
            .view(torch.bfloat16)
            .reshape(512, 1, 128)
            .npu()
        )
        k = v = x
    else:
        qkv = torch.randn(length, 36, 128, device="npu").bfloat16()
        k, v = qkv[:, 28:32], qkv[:, 32:]
    pages = (length + 127) // 128
    perm = torch.randperm(pages, device="npu")
    slots = (
        (perm[:, None] * 128 + torch.arange(128, device="npu"))
        .flatten()[:length]
        .long()
    )
    for code in (0, 1, 2):
        size = 128 * heads * 128 * (2 if code == 0 else 1)
        offsets = torch.arange(pages, device="npu", dtype=torch.int32) * size
        scales = (
            torch.ones(pages, heads, device="npu")
            if length == 512
            else torch.rand(pages, heads, device="npu") + 0.01
        )
        a = torch.full((pages * size,), 85, dtype=torch.uint8, device="npu")
        b = a.clone()
        c = a.clone()
        d = a.clone()
        native.strategy_scatter_kv(k, v, a, b, slots, offsets, scales, code)
        native.strategy_scatter_prefill_vec(k, v, c, d, slots, offsets, scales, code)
        equal = ((a == c) & (b == d)).cpu()
        if not equal.all():
            idx = (~equal).nonzero().flatten()
            print(
                "MISMATCH",
                length,
                code,
                idx[:30],
                a.cpu()[idx[:30]],
                c.cpu()[idx[:30]],
                flush=True,
            )
            raise AssertionError(int(idx.numel()))

        def measure(fn):
            for _ in range(3):
                fn()
            torch.npu.synchronize()
            start, end = (
                torch.npu.Event(enable_timing=True),
                torch.npu.Event(enable_timing=True),
            )
            start.record()
            for _ in range(20):
                fn()
            end.record()
            torch.npu.synchronize()
            return start.elapsed_time(end) / 20

        t_old = measure(
            lambda: native.strategy_scatter_kv(k, v, a, b, slots, offsets, scales, code)
        )
        t_new = measure(
            lambda: native.strategy_scatter_prefill_vec(
                k, v, c, d, slots, offsets, scales, code
            )
        )
        row = dict(length=length, code=code, old_ms=t_old, new_ms=t_new, passed=True)
        rows.append(row)
        print(row, flush=True)
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(rows, indent=2))
