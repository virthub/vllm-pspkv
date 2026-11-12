#!/usr/bin/env python3
"""Exact _bench_full.py-style greedy match (short acc_prompts) for int8 vs e4m3."""
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ["PSP_KV_BLOCK_SIZE"] = "128"

import torch
from vllm import LLM, SamplingParams

MODEL = "Qwen/Qwen3-1.7B"
OUT = 32


def make_llm(prec):
    if prec == "baseline":
        return LLM(model=MODEL, enforce_eager=True, max_model_len=4096,
                   gpu_memory_utilization=0.6, dtype="float16", block_size=128)
    os.environ["PSPKV_PRECISION"] = prec
    os.environ["PSPKV_ADAPTIVE_PRECISION"] = "0"
    os.environ.setdefault("PSPKV_FORMAT", "1")
    return LLM(model=MODEL, enforce_eager=True, max_model_len=4096,
               gpu_memory_utilization=0.6, dtype="float16",
               attention_backend="PSP_KV", enable_chunked_prefill=False,
               enable_prefix_caching=False, block_size=128)


def main():
    probe = make_llm("baseline")
    tok = probe.get_tokenizer()
    del probe
    torch.cuda.empty_cache()

    acc_prompts = [
        tok("The quick brown fox jumps over the lazy dog. ",
            add_special_tokens=False)["input_ids"],
        tok("Machine learning models compress knowledge into hidden states. " * 2,
            add_special_tokens=False)["input_ids"],
    ]
    sp = SamplingParams(max_tokens=OUT, temperature=0, ignore_eos=True)

    ref = None
    res = {}
    for prec in ["baseline", "e4m3", "int8"]:
        llm = make_llm(prec)
        out = [list(o.outputs[0].token_ids) for o in llm.generate(acc_prompts, sp, use_tqdm=False)]
        if prec == "baseline":
            ref = out
            print(f"baseline={out}", flush=True)
        else:
            m = sum(a == b for ra, oa in zip(ref, out) for a, b in zip(ra, oa))
            res[prec] = m / max(sum(len(r) for r in ref), 1)
            print(f"{prec}={out}", flush=True)
            print(f"greedy_match[{prec}]={res[prec]*100:.1f}%", flush=True)
        del llm
        torch.cuda.empty_cache()
    print(f"\nSUMMARY bench_full-style " + "  ".join(f"{p}={res[p]*100:.1f}%" for p in res), flush=True)


if __name__ == "__main__":
    main()
