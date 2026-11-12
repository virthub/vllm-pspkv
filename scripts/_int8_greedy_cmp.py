#!/usr/bin/env python3
"""Focused greedy-match comparison for a single model: baseline vs PSP-KV
precisions (e4m3 / int8) at a fixed context length.

Usage:
  .venv/bin/python scripts/_int8_greedy_cmp.py Qwen/Qwen3-1.7B 512 32 e4m3 int8
"""
import argparse
import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402


def make_llm(model, prec, max_len):
    if prec == "baseline":
        return LLM(model=model, enforce_eager=True, max_model_len=max_len,
                   gpu_memory_utilization=0.6, dtype="float16", block_size=128)
    os.environ["PSP_KV_BLOCK_SIZE"] = "128"
    os.environ["PSPKV_PRECISION"] = prec
    os.environ["PSPKV_ADAPTIVE_PRECISION"] = "0"
    os.environ.setdefault("PSPKV_FORMAT", "1")
    return LLM(model=model, enforce_eager=True, max_model_len=max_len,
               gpu_memory_utilization=0.6, dtype="float16",
               attention_backend="PSP_KV", enable_chunked_prefill=False,
               enable_prefix_caching=False, block_size=128)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", default="Qwen/Qwen3-1.7B")
    ap.add_argument("ctx", type=int, default=512)
    ap.add_argument("out_len", type=int, default=32)
    ap.add_argument("precs", nargs="+", default=["e4m3", "int8"])
    args = ap.parse_args()

    probe = make_llm(args.model, "baseline", args.ctx + args.out_len)
    tok = probe.get_tokenizer()
    del probe
    torch.cuda.empty_cache()

    sentence = "The quick brown fox jumps over the lazy dog. "
    n = len(tok(sentence, add_special_tokens=False)["input_ids"])
    ids = tok(sentence, add_special_tokens=False)["input_ids"]
    ids = (ids * (args.ctx // n + 1))[:args.ctx]
    prompts = [ids]

    sp = SamplingParams(max_tokens=args.out_len, temperature=0, ignore_eos=True)

    ref = None
    results = {}
    for prec in ["baseline"] + args.precs:
        llm = make_llm(args.model, prec, args.ctx + args.out_len)
        out = [list(o.outputs[0].token_ids) for o in llm.generate(prompts, sp, use_tqdm=False)]
        if prec == "baseline":
            ref = out
            print(f"baseline={out}", flush=True)
        else:
            m = sum(a == b for ra, oa in zip(ref, out) for a, b in zip(ra, oa))
            rate = m / max(sum(len(r) for r in ref), 1)
            results[prec] = rate
            print(f"{prec}={out}", flush=True)
            print(f"greedy_match[{prec}]={rate*100:.1f}% (ctx={args.ctx}, out={args.out_len})", flush=True)
        del llm
        torch.cuda.empty_cache()

    print(f"\nSUMMARY model={args.model} ctx={args.ctx} out={args.out_len} "
          + "  ".join(f"{p}={results[p]*100:.1f}%" for p in args.precs), flush=True)


if __name__ == "__main__":
    main()
