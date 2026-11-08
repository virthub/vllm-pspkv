#!/usr/bin/env python3
"""8K decode 3-way: PSP-KV fp8 vs PSP-KV lossless(f16) vs baseline(default)."""
import os, time, gc
os.environ.setdefault("HF_HUB_OFFLINE","1"); os.environ.setdefault("TRANSFORMERS_OFFLINE","1")
import torch
from vllm import LLM, SamplingParams

MODEL = "Qwen/Qwen3-4B"
CTX = 8192
N_TOK = 64


def make_prompt(tok, n):
    sent = "The quick brown fox jumps over the lazy dog. "
    ids = tok(sent, add_special_tokens=False)["input_ids"]
    reps = (n + len(ids) - 1) // len(ids) + 1
    return (ids * reps)[:n]


def bench(kind):
    os.environ.pop("PSPKV_TC_NUM_SPLITS", None)
    if kind == "fp8":
        os.environ["PSPKV_PRECISION"] = "e4m3"
        os.environ["PSPKV_ADAPTIVE_PRECISION"] = "0"
        os.environ.setdefault("PSPKV_FORMAT", "1")
        kw = dict(attention_backend="PSP_KV", block_size=128)
    elif kind == "lossless":
        os.environ["PSPKV_PRECISION"] = "lossless"
        os.environ["PSPKV_ADAPTIVE_PRECISION"] = "0"
        os.environ.setdefault("PSPKV_FORMAT", "1")
        kw = dict(attention_backend="PSP_KV", block_size=128)
    else:
        kw = dict()
    llm = LLM(model=MODEL, enforce_eager=True, max_model_len=CTX + N_TOK,
              gpu_memory_utilization=0.6, dtype="float16",
              enable_chunked_prefill=False, enable_prefix_caching=False, **kw)
    tok = llm.get_tokenizer()
    ids = make_prompt(tok, CTX)
    sp = SamplingParams(max_tokens=N_TOK, temperature=0, ignore_eos=True)
    llm.generate([ids], sp, use_tqdm=False)  # warmup
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    llm.generate([ids], sp, use_tqdm=False)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    del llm; gc.collect(); torch.cuda.empty_cache()
    return N_TOK / dt


for kind in ["fp8", "lossless", "baseline"]:
    try:
        ts = bench(kind)
        print(f"{kind:10s}  {ts:7.1f} tok/s", flush=True)
    except Exception as e:
        print(f"{kind:10s}  ERROR: {str(e)[:100]}", flush=True)
