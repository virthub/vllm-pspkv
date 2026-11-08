#!/usr/bin/env python3
"""Kernel-level decode profiling: where does a decode step spend GPU time?"""
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import argparse
import torch
from torch.profiler import profile, ProfilerActivity
from vllm import LLM, SamplingParams

MODEL = "Qwen/Qwen3-4B"


def make_llm(prec, batch):
    if prec == "baseline":
        return LLM(model=MODEL, enforce_eager=True, max_model_len=2048,
                   gpu_memory_utilization=0.6, dtype="float16")
    os.environ["PSPKV_PRECISION"] = prec
    os.environ["PSPKV_ADAPTIVE_PRECISION"] = "0"
    os.environ.setdefault("PSPKV_FORMAT", "1")
    return LLM(model=MODEL, enforce_eager=True, max_model_len=2048,
               gpu_memory_utilization=0.6, dtype="float16",
               attention_backend="PSP_KV", enable_chunked_prefill=False,
               enable_prefix_caching=False)


def profile_decode(prec, batch, ctx):
    llm = make_llm(prec, batch)
    tok = llm.get_tokenizer()
    sentence = "The quick brown fox jumps over the lazy dog. "
    ids = tok(sentence * (ctx // len(tok(sentence)["input_ids"]) + 1),
              add_special_tokens=False)["input_ids"]
    batch_ids = [ids] * batch
    sp1 = SamplingParams(max_tokens=1, temperature=0, ignore_eos=True)
    # prefill + warmup
    llm.generate(batch_ids, sp1, use_tqdm=False)
    torch.cuda.synchronize()
    # profile a few decode steps
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(4):
            llm.generate(batch_ids, sp1, use_tqdm=False)
        torch.cuda.synchronize()
    evs = prof.key_averages()
    rows = []
    for e in evs:
        rows.append((e.self_device_time_total, e.count, e.key))
    rows.sort(reverse=True)
    total = sum(r[0] for r in rows)
    print(f"\n=== {prec}  batch={batch} ctx={ctx}  (top kernels by self CUDA time) ===")
    for t, c, k in rows[:14]:
        print(f"  {t/1e3:8.1f} us  {t/total*100:5.1f}%  x{c:<4d} {k[:70]}")
    del llm
    torch.cuda.empty_cache()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prec", default="e4m3")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--ctx", type=int, default=512)
    a = ap.parse_args()
    profile_decode(a.prec, a.batch, a.ctx)
