#!/usr/bin/env python3
"""Evaluate PSP-KV KV-cache memory savings (CPU-only).

Reports the variable-stride byte footprint for a per-block precision mix and
compares it against the fp8 / fp16 baselines and the allocation reserved by the
``PSPKV_AVG_BYTES`` budget. No GPU or vLLM import is required.

Example:
    .venv/bin/python scripts/evaluate_memory_psp_kv.py \
        --num-blocks 64 --mix 0.5,0.5,0.0 --avg-bytes 0.75
"""

import argparse
import os

# bytes-per-element for each PSP-KV inner-precision format.
# 0 = lossless (fp16), 1 = fp8 e4m3, 2 = fp8 e5m2, 3 = fp4 e2m1.
ELEM_BYTES = {0: 2.0, 1: 1.0, 2: 1.0, 3: 0.5}
FMT_NAME = {0: "lossless(fp16)", 1: "fp8 e4m3", 2: "fp8 e5m2", 3: "fp4 e2m1"}


def _fmt_banner(title: str) -> str:
    return f"\n{title}\n" + "-" * 60


def main() -> None:
    p = argparse.ArgumentParser(
        description="Evaluate PSP-KV KV-cache memory savings (CPU-only)."
    )
    p.add_argument("--num-blocks", type=int, default=64)
    p.add_argument("--block-size", type=int, default=16)
    p.add_argument("--num-kv-heads", type=int, default=8)
    p.add_argument("--head-size", type=int, default=128)
    p.add_argument(
        "--mix",
        default="0.5,0.5,0.0",
        help="fp8,fp4,lossless fractions that sum to 1.0 (default: 0.5,0.5,0.0)",
    )
    p.add_argument(
        "--avg-bytes",
        type=float,
        default=None,
        help="allocation budget (default: $PSPKV_AVG_BYTES, else 1.0)",
    )
    args = p.parse_args()

    fp8_frac, fp4_frac, lossless_frac = (float(x) for x in args.mix.split(","))
    if abs(fp8_frac + fp4_frac + lossless_frac - 1.0) > 1e-6:
        raise SystemExit("error: --mix fractions must sum to 1.0")

    num_blocks = args.num_blocks
    # Number of elements in ONE of K or V for a single block.
    elems_per_block = args.block_size * args.num_kv_heads * args.head_size

    # Build a representative per-block precision sequence from the mix.
    n_fp8 = round(num_blocks * fp8_frac)
    n_fp4 = round(num_blocks * fp4_frac)
    n_lossless = num_blocks - n_fp8 - n_fp4
    fmt_seq = [1] * n_fp8 + [3] * n_fp4 + [0] * n_lossless

    # Variable-stride footprint (per K/V), matching the model runner's
    # block_offsets cumulative prefix sum.
    variable_per_kv = sum(elems_per_block * ELEM_BYTES[f] for f in fmt_seq)
    fp8_per_kv = num_blocks * elems_per_block * 1.0
    fp16_per_kv = num_blocks * elems_per_block * 2.0

    avg_elem = variable_per_kv / (num_blocks * elems_per_block)

    # Allocation reserved by PSPKV_AVG_BYTES (per K/V).
    avg_bytes = (
        args.avg_bytes
        if args.avg_bytes is not None
        else float(os.environ.get("PSPKV_AVG_BYTES", "1.0"))
    )
    alloc_per_kv = num_blocks * elems_per_block * avg_bytes

    print(
        f"Geometry: {num_blocks} blocks x {args.block_size} tokens x "
        f"{args.num_kv_heads} heads x {args.head_size} dim"
    )
    print(
        f"Precision mix: fp8={fp8_frac:.2f} fp4={fp4_frac:.2f} "
        f"lossless={lossless_frac:.2f}"
    )
    print(f"  distribution: {n_fp8} fp8, {n_fp4} fp4, {n_lossless} lossless")

    print(_fmt_banner("Per-K/V footprint"))
    print(f"  fp16 baseline       {fp16_per_kv:>12.0f} B   (2.00 B/elem)")
    print(f"  fp8  baseline       {fp8_per_kv:>12.0f} B   (1.00 B/elem)")
    print(f"  variable footprint  {variable_per_kv:>12.0f} B   ({avg_elem:.2f} B/elem)")

    print(_fmt_banner("Reduction (variable footprint)"))
    print(f"  vs fp16: {fp16_per_kv / variable_per_kv:5.2f}x smaller")
    print(f"  vs fp8 : {fp8_per_kv / variable_per_kv:5.2f}x smaller")

    print(_fmt_banner(f"Allocation (PSPKV_AVG_BYTES={avg_bytes:.2f})"))
    print(f"  reserved per K/V   {alloc_per_kv:>12.0f} B")
    if variable_per_kv > alloc_per_kv:
        over = variable_per_kv - alloc_per_kv
        print(
            f"  !!! OVERFLOW: variable footprint exceeds allocation by "
            f"{over:.0f} B ({variable_per_kv / alloc_per_kv:.2f}x)"
        )
    else:
        headroom = alloc_per_kv - variable_per_kv
        print(
            f"  fits: {headroom:.0f} B headroom "
            f"({variable_per_kv / alloc_per_kv:.1%} used)"
        )


if __name__ == "__main__":
    main()
