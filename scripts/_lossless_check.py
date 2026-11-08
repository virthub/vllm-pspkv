#!/usr/bin/env python3
"""Run one kind (lossless|fp8|baseline) at 8K, print tok/s + tokens."""
import os, sys, time
os.environ.setdefault("HF_HUB_OFFLINE","1"); os.environ.setdefault("TRANSFORMERS_OFFLINE","1")
import torch
from vllm import LLM, SamplingParams

kind = sys.argv[1] if len(sys.argv) > 1 else "lossless"
MODEL = "Qwen/Qwen3-4B"; CTX = 8192; N_TOK = 16

def make_prompt(tok, n):
    sent = "The quick brown fox jumps over the lazy dog. "
    ids = tok(sent, add_special_tokens=False)["input_ids"]
    return (ids * ((n + len(ids) - 1) // len(ids) + 1))[:n]

kw = dict()
if kind in ("lossless", "fp8"):
    os.environ["PSPKV_PRECISION"] = "e4m3" if kind == "fp8" else "lossless"
    os.environ["PSPKV_ADAPTIVE_PRECISION"] = "0"
    os.environ.setdefault("PSPKV_FORMAT", "1")
    kw = dict(attention_backend="PSP_KV", block_size=128)

llm = LLM(model=MODEL, enforce_eager=True, max_model_len=CTX + N_TOK,
          gpu_memory_utilization=0.6, dtype="float16",
          enable_chunked_prefill=False, enable_prefix_caching=False, **kw)
tok = llm.get_tokenizer()
ids = make_prompt(tok, CTX)
sp = SamplingParams(max_tokens=N_TOK, temperature=0, ignore_eos=True)
llm.generate([ids], sp, use_tqdm=False)  # warmup
torch.cuda.synchronize()
out = llm.generate([ids], sp, use_tqdm=False)
toks = out[0].outputs[0].token_ids
torch.cuda.synchronize()
t0 = time.perf_counter()
llm.generate([ids], sp, use_tqdm=False)
torch.cuda.synchronize()
dt = time.perf_counter() - t0
print(f"RESULT kind={kind} {N_TOK/dt:7.1f} tok/s tokens={toks}", flush=True)
