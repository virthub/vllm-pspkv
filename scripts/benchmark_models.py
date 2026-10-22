#!/usr/bin/env python3
"""Per-model prefill/decode benchmark for PSP-KV vs the FlashAttention baseline.

Measures prefill latency (first token) and decode throughput (tokens/sec) for a
list of cached models, with an optional inner-precision setting. Reports a
table plus the PSP-KV vs baseline ratios.

Usage:
  .venv/bin/python scripts/benchmark_models.py \
      --models Qwen/Qwen3-0.6B Qwen/Qwen3-4B \
      --precision lossless --input-len 512 --output-len 64
"""
import argparse
import os
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from vllm import LLM, SamplingParams  # noqa: E402


def _make_llm(model, backend, gpu_util):
    kwargs = dict(
        model=model,
        enforce_eager=True,
        max_model_len=2048,
        gpu_memory_utilization=gpu_util,
        dtype="bfloat16",
    )
    if backend == "PSP_KV":
        kwargs["attention_backend"] = "PSP_KV"
    return LLM(**kwargs)


def _run(llm, prompt, n_tokens):
    sp = SamplingParams(max_tokens=n_tokens, temperature=0, ignore_eos=True)
    out = llm.generate([prompt], sp)
    return out


def _time_prefill(llm, prompt):
    sp = SamplingParams(max_tokens=1, temperature=0, ignore_eos=True)
    import torch
    llm.generate([prompt], sp)  # warmup
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(3):
        llm.generate([prompt], sp)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / 3 * 1e3  # ms


def _time_decode(llm, prompt, out_len):
    import torch
    sp = SamplingParams(max_tokens=out_len, temperature=0, ignore_eos=True)
    llm.generate([prompt], sp)  # warmup
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(3):
        llm.generate([prompt], sp)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / 3  # s per full (prefill+decode)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=["Qwen/Qwen3-0.6B", "Qwen/Qwen3-4B"])
    ap.add_argument("--precision", default="lossless")
    ap.add_argument("--input-len", type=int, default=512)
    ap.add_argument("--output-len", type=int, default=64)
    ap.add_argument("--gpu-util", type=float, default=0.6)
    args = ap.parse_args()

    os.environ["PSPKV_PRECISION"] = args.precision
    os.environ["PSPKV_FORMAT"] = "1"

    prompt = " ".join(["The quick brown fox jumps over the lazy dog."] * 20)
    prompt = prompt[: args.input_len]

    print(f"precision={args.precision} input_len={args.input_len} output_len={args.output_len}")
    print(f"{'model':28s} {'backend':12s} {'prefill_ms':>10s} {'decode_tok/s':>12s} {'ratio':>6s}")

    results = {}
    for model in args.models:
        for backend in ["baseline", "PSP_KV"]:
            tag = f"{model}:{backend}"
            try:
                llm = _make_llm(model, backend, args.gpu_util)
                prefill_ms = _time_prefill(llm, prompt)
                full_s = _time_decode(llm, prompt, args.output_len)
                # decode-only time = full - prefill (approximate)
                decode_s = full_s - prefill_ms / 1e3
                tok_s = args.output_len / max(decode_s, 1e-6)
                results[tag] = (prefill_ms, tok_s)
                print(f"{model:28s} {backend:12s} {prefill_ms:10.2f} {tok_s:12.1f} {'':>6s}")
                del llm
            except Exception as e:
                print(f"{model:28s} {backend:12s} FAILED: {str(e)[:60]}")
                results[tag] = None

    print("\nRatios (PSP_KV / baseline, >1 = PSP-KV faster):")
    for model in args.models:
        b = results.get(f"{model}:baseline")
        p = results.get(f"{model}:PSP_KV")
        if b and p:
            print(f"{model:28s} decode {p[1]/b[1]:.2f}x  prefill {p[0]/b[0]:.2f}x")


if __name__ == "__main__":
    main()
