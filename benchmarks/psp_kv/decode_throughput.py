#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Decode-only throughput for the PSP-KV backend.

Prefill is excluded by timing a short and a long generation and taking the
difference, so the reported number isolates the decode loop:

    tok/s = batch * (long_tokens - short_tokens) / (t_long - t_short)

For every model/context/batch cell the FlashAttention backend is measured first
and used as the baseline, then each requested PSP-KV inner precision.

Usage:
    python benchmarks/psp_kv/decode_throughput.py
    PSPKV_BENCH_MODELS=Qwen/Qwen3-0.6B python benchmarks/psp_kv/decode_throughput.py
    PSPKV_BENCH_PRECISIONS=int8,e4m3,adaptive ...
"""

import os
import sys
import time

_REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("PSPKV_FORMAT", "1")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

from vllm import LLM, SamplingParams  # noqa: E402

BLOCK_SIZE = int(os.environ.get("PSPKV_BLOCK_SIZE", "128"))
SHORT_TOKENS = 8
LONG_TOKENS = 72
WARMUP_TOKENS = 8
OUT_TOKENS = LONG_TOKENS - SHORT_TOKENS

MODELS = [
    m.strip()
    for m in os.environ.get(
        "PSPKV_BENCH_MODELS", "Qwen/Qwen3-0.6B,Qwen/Qwen3-4B"
    ).split(",")
    if m.strip()
]
CONTEXTS = [
    int(c) for c in os.environ.get("PSPKV_BENCH_CONTEXTS", "512,2048").split(",")
]
BATCHES = [int(b) for b in os.environ.get("PSPKV_BENCH_BATCHES", "32").split(",")]
PRECISIONS = [
    p.strip()
    for p in os.environ.get("PSPKV_BENCH_PRECISIONS", "int8,e4m3,adaptive").split(",")
    if p.strip()
]
MAX_MODEL_LEN = max(CONTEXTS) + LONG_TOKENS + WARMUP_TOKENS + 128

PROMPT = "The quick brown fox jumps over the lazy dog. "


def build(model: str, backend: str, precision: str | None = None) -> LLM:
    """Build an engine for one backend/precision combination."""
    os.environ["PSPKV_ADAPTIVE_PRECISION"] = "0"
    if precision == "adaptive":
        os.environ["PSPKV_ADAPTIVE_PRECISION"] = "1"
    elif precision:
        os.environ["PSPKV_PRECISION"] = precision
    return LLM(
        model=model,
        attention_backend=backend,
        block_size=BLOCK_SIZE,
        enforce_eager=True,
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=float(os.environ.get("PSPKV_BENCH_GPU_UTIL", "0.85")),
        dtype="float16",
        enable_prefix_caching=False,
    )


def _timed(llm: LLM, ids: list[int], batch: int, ntok: int) -> float:
    params = SamplingParams(max_tokens=ntok, temperature=0, ignore_eos=True)
    start = time.perf_counter()
    llm.generate([ids] * batch, params, use_tqdm=False)
    return time.perf_counter() - start


def decode_tokens_per_second(llm: LLM, ctx: int, batch: int) -> float:
    """Decode-loop throughput, with the prefill contribution subtracted."""
    tokenizer = llm.get_tokenizer()
    ids = tokenizer(PROMPT * (ctx // 9 + 1), add_special_tokens=False)["input_ids"][
        :ctx
    ]
    _timed(llm, ids, min(batch, 2), WARMUP_TOKENS)
    short = _timed(llm, ids, batch, SHORT_TOKENS)
    long = _timed(llm, ids, batch, LONG_TOKENS)
    return batch * OUT_TOKENS / max(long - short, 1e-6)


def main() -> int:
    print(
        f"PSP-KV decode throughput: block={BLOCK_SIZE} "
        f"contexts={CONTEXTS} batches={BATCHES} precisions={PRECISIONS}"
    )
    for model in MODELS:
        tag = model.split("/")[-1]
        for ctx in CONTEXTS:
            for batch in BATCHES:
                baseline = build(model, "FLASH_ATTN")
                base_tps = decode_tokens_per_second(baseline, ctx, batch)
                del baseline

                row = f"DEC {tag:9s} ctx={ctx:5d} b={batch:2d} baseline={base_tps:7.1f}"
                for precision in PRECISIONS:
                    try:
                        llm = build(model, "PSP_KV", precision)
                        tps = decode_tokens_per_second(llm, ctx, batch)
                        row += f" {precision}={tps:7.1f} ({tps / base_tps:5.3f}x)"
                        del llm
                    except Exception as exc:  # noqa: BLE001 - report and continue
                        row += f" {precision}=FAILED({type(exc).__name__})"
                print(row, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
