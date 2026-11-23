#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PSP-KV quality and cost evaluation, one configuration per engine.

For every requested inner precision (plus the adaptive mode) this reports:

* decode-path perplexity — the log-probabilities of *generated* tokens, which is
  the only valid quality measure here: prefill runs through FlashAttention over
  full-precision K/V, so prompt NLL would not see the cache at all;
* greedy agreement with the FlashAttention baseline;
* decode throughput;
* the reserved KV footprint and the observed per-block precision histogram.

Usage:
    python scripts/eval_quality.py
    PSPKV_EVAL_PRECISIONS=int8,e4m3 python scripts/eval_quality.py
    PSPKV_EVAL_MODEL=Qwen/Qwen3-4B PSPKV_EVAL_ADAPTIVE_E=0.01 \
        python scripts/eval_quality.py

Each configuration builds its own engine and patches the environment before the
next one starts, so the reported numbers are not contaminated by a previous
precision's page geometry.
"""

import math
import os
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("PSPKV_FORMAT", "1")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

from vllm import LLM, SamplingParams  # noqa: E402

MODEL = os.environ.get("PSPKV_EVAL_MODEL", "Qwen/Qwen3-0.6B")
BLOCK_SIZE = int(os.environ.get("PSPKV_BLOCK_SIZE", "128"))
PRECISIONS = [
    p.strip()
    for p in os.environ.get("PSPKV_EVAL_PRECISIONS", "int8,e4m3,int4,adaptive").split(
        ","
    )
    if p.strip()
]
ADAPTIVE_E = os.environ.get("PSPKV_EVAL_ADAPTIVE_E", "0.01")
DTYPE = os.environ.get("PSPKV_EVAL_DTYPE", "float16")
QUALITY_PROMPT = (
    "The history of the printing press begins in the fifteenth century. " * 4
)
QUALITY_TOKENS = int(os.environ.get("PSPKV_EVAL_TOKENS", "64"))
BENCH_PROMPT = "The quick brown fox jumps over the lazy dog. " * 4
BENCH_TOKENS = int(os.environ.get("PSPKV_EVAL_BENCH_TOKENS", "64"))


def build(backend: str, precision: str) -> LLM:
    """Build one engine, isolating the precision/adaptive environment."""
    os.environ["PSPKV_ADAPTIVE_PRECISION"] = "0"
    if precision == "adaptive":
        os.environ["PSPKV_ADAPTIVE_PRECISION"] = "1"
        os.environ["PSPKV_ADAPTIVE_E"] = ADAPTIVE_E
    else:
        os.environ["PSPKV_PRECISION"] = precision
    return LLM(
        model=MODEL,
        attention_backend=backend,
        block_size=BLOCK_SIZE,
        enforce_eager=True,
        max_model_len=512,
        gpu_memory_utilization=float(os.environ.get("PSPKV_EVAL_GPU_UTIL", "0.6")),
        dtype=DTYPE,
        enable_prefix_caching=False,
    )


def quality(llm: LLM) -> tuple[float, list[int]]:
    """Decode-path perplexity and the generated tokens it was measured on."""
    tokenizer = llm.get_tokenizer()
    ids = tokenizer(QUALITY_PROMPT, add_special_tokens=False)["input_ids"]
    out = llm.generate(
        [ids],
        SamplingParams(
            max_tokens=QUALITY_TOKENS, temperature=0, ignore_eos=True, logprobs=1
        ),
        use_tqdm=False,
    )
    seq = out[0].outputs[0]
    logprobs = [next(iter(m.values())).logprob for m in (seq.logprobs or []) if m]
    if not logprobs:
        return float("nan"), seq.token_ids
    return math.exp(-sum(logprobs) / len(logprobs)), seq.token_ids


def throughput(llm: LLM, batch: int) -> float:
    tokenizer = llm.get_tokenizer()
    ids = tokenizer(BENCH_PROMPT, add_special_tokens=False)["input_ids"]
    params = SamplingParams(max_tokens=BENCH_TOKENS, temperature=0, ignore_eos=True)
    llm.generate(
        [ids] * batch, SamplingParams(max_tokens=8, temperature=0), use_tqdm=False
    )
    start = time.perf_counter()
    llm.generate([ids] * batch, params, use_tqdm=False)
    return batch * BENCH_TOKENS / max(time.perf_counter() - start, 1e-6)


def runner_of(llm: LLM):
    try:
        return llm.llm_engine.model_executor.driver_worker.model_runner
    except AttributeError:
        return None


def kv_gib(llm: LLM) -> float:
    runner = runner_of(llm)
    config = getattr(runner, "kv_cache_config", None)
    if config is None:
        return float("nan")
    total = 0
    for group in config.kv_cache_groups:
        spec = getattr(group, "kv_cache_spec", None)
        first = getattr(spec, "first_spec", spec)
        total += config.num_blocks * getattr(first, "page_size_bytes", 0)
    return total / 2**30


def precision_histogram(llm: LLM) -> dict[str, dict[str, int]]:
    runner = runner_of(llm)
    if runner is None or not hasattr(runner, "psp_kv_stats"):
        return {}
    try:
        stats = runner.psp_kv_stats(0)
    except Exception:  # noqa: BLE001 - introspection only
        return {}
    return {k: v for k, v in stats.items() if k.endswith("_counts")}


def main() -> int:
    print(
        f"model: {MODEL}   dtype: {DTYPE}   block size: {BLOCK_SIZE}   "
        f"quality: {QUALITY_TOKENS} generated tokens"
    )
    baseline = build("FLASH_ATTN", PRECISIONS[0])
    base_ppl, base_tokens = quality(baseline)
    base_tp = throughput(baseline, 1)
    base_kv = kv_gib(baseline)
    del baseline
    print(
        f"{'baseline':10s} ppl={base_ppl:7.4f} tok/s b1={base_tp:7.1f} "
        f"kv={base_kv:5.2f}GiB"
    )

    for precision in PRECISIONS:
        label = f"adaptive@{ADAPTIVE_E}" if precision == "adaptive" else precision
        try:
            llm = build("PSP_KV", precision)
            ppl, tokens = quality(llm)
            agree = sum(a == b for a, b in zip(tokens, base_tokens))
            tp = throughput(llm, 1)
            print(
                f"{label:10s} ppl={ppl:7.4f} agree={agree}/{len(base_tokens)} "
                f"tok/s b1={tp:7.1f} ({tp / base_tp:5.3f}x) "
                f"kv={kv_gib(llm):5.2f}GiB fmt={precision_histogram(llm)}",
                flush=True,
            )
            del llm
        except Exception as exc:  # noqa: BLE001 - one bad config must not stop the sweep
            print(f"{label:10s} FAILED: {type(exc).__name__}: {exc}"[:200], flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
