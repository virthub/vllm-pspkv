#!/usr/bin/env python3
# Benchmark the three PSP-KV launchers (basic / enhanced / btkv) across target
# precisions and inner precisions, and compare against an SDPA (FlashAttention)
# baseline. Reports per-decode-step latency in microseconds.
#
# Usage:
#   .venv/bin/python scripts/benchmark_psp_kv.py [--num-heads N] [--num-kv-heads K]
#       [--head-dim D] [--context-len L] [--iters I]
#
# Requires a CUDA GPU and a built PSP-KV extension (torch.ops.psp_kv).

import argparse
import os
import sys
import time

import torch
import torch.nn.functional as F

# Allow running as a standalone script (add the project root to sys.path).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tests.v1.attention.psp_kv_reference import pack_kv_cache

LAUNCHERS = ["basic", "enhanced", "btkv"]
TARGET_DTYPES = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp8_e4m3": torch.float8_e4m3fn,
    "fp8_e5m2": torch.float8_e5m2,
}
INNER_FMTS = {"lossless": 0, "e4m3": 1, "e5m2": 2, "fp4": 3}


def _get_ops():
    import vllm._C  # noqa: F401  (registers torch.ops.psp_kv)
    return getattr(torch.ops, "psp_kv", None)


def _time(fn, iters, warmup):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters  # ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-heads", type=int, default=32)
    ap.add_argument("--num-kv-heads", type=int, default=8)
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--context-len", type=int, default=1024)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=5)
    args = ap.parse_args()

    ops = _get_ops()
    if ops is None or not hasattr(ops, "psp_attention_enhanced"):
        raise SystemExit("torch.ops.psp_kv not available; build the extension first.")

    num_heads = args.num_heads
    num_kv_heads = args.num_kv_heads
    head_dim = args.head_dim
    context_len = args.context_len
    block_size = 16
    num_blocks = (context_len + block_size - 1) // block_size
    context_len = num_blocks * block_size  # round up to whole blocks
    scale = 1.0 / (head_dim ** 0.5)
    device = "cuda"

    torch.manual_seed(0)
    # Full-precision reference K/V (fp16), [num_blocks, kv_heads, block_size, head_dim].
    k_full = torch.randn(num_blocks, num_kv_heads, block_size, head_dim, device=device) * 0.5
    v_full = torch.randn(num_blocks, num_kv_heads, block_size, head_dim, device=device) * 0.5

    # ---------- Baseline: SDPA (FlashAttention) on the dequantized fp16 K/V ----------
    g = num_heads // num_kv_heads
    k_seq = k_full.permute(1, 0, 2, 3).reshape(num_kv_heads, context_len, head_dim)
    v_seq = v_full.permute(1, 0, 2, 3).reshape(num_kv_heads, context_len, head_dim)
    k_seq = k_seq.repeat_interleave(g, dim=0).unsqueeze(0)  # [1, H, N, D]
    v_seq = v_seq.repeat_interleave(g, dim=0).unsqueeze(0)

    def sdpa_baseline(dtype):
        q = torch.randn(1, num_heads, 1, head_dim, device=device).to(dtype)
        kk = k_seq.to(dtype)
        vv = v_seq.to(dtype)
        return F.scaled_dot_product_attention(q, kk, vv, is_causal=False, scale=scale)

    # Full-precision SDPA baselines (fp8 has no fused SDPA backend).
    baseline = {}
    for name in ("fp16", "bf16"):
        baseline[name] = _time(
            lambda d=TARGET_DTYPES[name]: sdpa_baseline(d), args.iters, args.warmup
        )

    # ---------- PSP-KV launchers ----------
    block_tables = torch.arange(num_blocks, dtype=torch.int32, device=device).unsqueeze(0)
    context_lens = torch.tensor([context_len], dtype=torch.int32, device=device)
    tma = torch.zeros(num_blocks, 16, dtype=torch.int64, device=device)

    results = {}  # (launcher, inner, target) -> us
    for launcher_idx, launcher in enumerate(LAUNCHERS):
        for inner_name, fmt in INNER_FMTS.items():
            precision_table = torch.full((num_blocks,), fmt, dtype=torch.uint8, device=device)
            scale_table = torch.ones(num_blocks, num_kv_heads, device=device)
            # Cache layout depends on the launcher (V transposed for enhanced/btkv).
            k_buf, v_buf, block_offsets = pack_kv_cache(
                k_full, v_full, launcher_idx, precision_table, torch.float16
            )
            for target_name, dtype in TARGET_DTYPES.items():
                q = torch.randn(1, num_heads, head_dim, device=device).to(dtype)
                out = torch.empty_like(q)

                def run(li=launcher_idx, qq=q, kb=k_buf, vb=v_buf, oo=out):
                    if li == 0:
                        ops.psp_attention_basic(
                            qq, kb, vb, block_tables, context_lens, precision_table,
                            block_offsets, scale, num_kv_heads, head_dim, oo,
                        )
                    elif li == 1:
                        ops.psp_attention_enhanced(
                            qq, kb, vb, block_tables, context_lens, precision_table,
                            block_offsets, scale_table, scale, num_kv_heads, head_dim, oo,
                        )
                    else:
                        ops.psp_attention_btkv(
                            qq, kb, vb, block_tables, context_lens, precision_table,
                            block_offsets, scale_table, tma, scale, num_kv_heads, head_dim, oo,
                        )

                ms = _time(run, args.iters, args.warmup)
                results[(launcher, inner_name, target_name)] = ms * 1000.0  # us

    # ---------- Report ----------
    print(f"Shape: heads={num_heads} kv_heads={num_kv_heads} head_dim={head_dim} "
          f"context={context_len} blocks={num_blocks}")
    print(f"Baseline SDPA (FlashAttention), us/decode:")
    for name in baseline:
        print(f"  {name:10s} {baseline[name] * 1000.0:8.2f} us")
    print()

    header = f"{'launcher':<10} {'inner':<10} " + "".join(f"{t:>12s}" for t in TARGET_DTYPES)
    print(header)
    for launcher in LAUNCHERS:
        for inner_name in INNER_FMTS:
            row = f"{launcher:<10} {inner_name:<10} "
            for target_name in TARGET_DTYPES:
                row += f"{results[(launcher, inner_name, target_name)]:>12.2f}"
            print(row)

    print()
    print(f"Slowdown vs SDPA-fp16 baseline (kernel_us / baseline_us):")
    base_us = baseline["fp16"] * 1000.0
    for launcher in LAUNCHERS:
        for inner_name in INNER_FMTS:
            row = f"{launcher:<10} {inner_name:<10} "
            for target_name in TARGET_DTYPES:
                us = results[(launcher, inner_name, target_name)]
                row += f"{us / base_us:>11.0f}x "
            print(row)


if __name__ == "__main__":
    main()
