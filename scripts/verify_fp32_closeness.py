#!/usr/bin/env python3
"""Verify PSP-KV correctness as fp32-reference closeness.

Correctness is defined as the numerical distance between the PSP-KV decode
attention output and a reference fp32 paged attention over the *same* byte
cache, per layer and per decode step. This is the intended correctness
criterion: the PSP-KV kernel is numerically correct when the max/mean absolute
difference stays at the fp32/bf16 rounding level (<= ~1e-4), independent of
greedy text divergence.

The comparison runs inside the engine via the PSPKV_VERIFY=1 hook; this script
just drives a generation and reports the accumulated stats printed by the hook.

Usage:
  BENCH_MODEL=Qwen/Qwen3-4B PSPKV_PRECISION=lossless \
      .venv/bin/python scripts/verify_fp32_closeness.py [--max-tokens N]
"""

import argparse
import os

os.environ["PSPKV_VERIFY"] = "1"
os.environ.setdefault("PSPKV_FORMAT", "1")
os.environ.setdefault("PSPKV_PRECISION", "lossless")
os.environ.setdefault("PSPKV_STORAGE", "fixed")

from vllm import LLM, SamplingParams  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-tokens", type=int, default=16)
    args = ap.parse_args()

    model = os.environ.get("BENCH_MODEL", "Qwen/Qwen3-4B")
    llm = LLM(
        model=model,
        enforce_eager=True,
        max_model_len=1024,
        gpu_memory_utilization=0.6,
        attention_backend="PSP_KV",
    )
    prompt = " ".join(["The quick brown fox jumps over the lazy dog."] * 2)
    sp = SamplingParams(max_tokens=args.max_tokens, temperature=0, ignore_eos=True)
    llm.generate([prompt], sp)


if __name__ == "__main__":
    main()
