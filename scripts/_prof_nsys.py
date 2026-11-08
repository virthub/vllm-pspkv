#!/usr/bin/env python3
"""Minimal 1-token 8K decode for nsys (prefix-cached, no prefill in the profiled step)."""
import os
os.environ.setdefault("HF_HUB_OFFLINE","1"); os.environ.setdefault("TRANSFORMERS_OFFLINE","1")
os.environ["PSPKV_PRECISION"]="e4m3"; os.environ["PSPKV_ADAPTIVE_PRECISION"]="0"; os.environ.setdefault("PSPKV_FORMAT","1")
import torch
from vllm import LLM, SamplingParams

MODEL="Qwen/Qwen3-4B"; CTX=8192; N_TOK=1

def make_prompt(tok, n):
    sent="The quick brown fox jumps over the lazy dog. "
    ids=tok(sent,add_special_tokens=False)["input_ids"]
    return (ids*((n+len(ids)-1)//len(ids)+1))[:n]

llm=LLM(model=MODEL, enforce_eager=True, max_model_len=CTX+N_TOK, gpu_memory_utilization=0.6,
        dtype="float16", attention_backend="PSP_KV", enable_chunked_prefill=False,
        enable_prefix_caching=True, block_size=128)
tok=llm.get_tokenizer(); ids=make_prompt(tok, CTX)
sp=SamplingParams(max_tokens=N_TOK, temperature=0, ignore_eos=True)
llm.generate([ids], sp, use_tqdm=False)  # warmup (prefill + decode), caches the 8K KV
torch.cuda.synchronize()
llm.generate([ids], sp, use_tqdm=False)  # decode only (KV reused)
torch.cuda.synchronize()
print("DONE", flush=True)
