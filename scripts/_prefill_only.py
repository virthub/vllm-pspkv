#!/usr/bin/env python3
"""Minimal prefill-only run for nsys profiling (4B, 512-token prompt)."""
import os, sys
os.environ.setdefault("HF_HUB_OFFLINE","1"); os.environ.setdefault("TRANSFORMERS_OFFLINE","1")
import torch
from vllm import LLM, SamplingParams

backend = sys.argv[1] if len(sys.argv) > 1 else "pspkv"
ctx = 512
kw = dict(block_size=128)
if backend == "pspkv":
    os.environ["PSP_KV_BLOCK_SIZE"] = "128"
    os.environ["PSPKV_PRECISION"] = "e4m3"
    os.environ["PSPKV_ADAPTIVE_PRECISION"] = "0"
    os.environ.setdefault("PSPKV_FORMAT", "1")
    kw["attention_backend"] = "PSP_KV"
llm = LLM(model="Qwen/Qwen3-4B", enforce_eager=True, max_model_len=ctx+8,
          gpu_memory_utilization=0.6, dtype="float16",
          enable_chunked_prefill=False, enable_prefix_caching=False, **kw)
tok = llm.get_tokenizer()
sentence = "The quick brown fox jumps over the lazy dog. "
ids = tok(sentence * (ctx // len(tok(sentence)["input_ids"]) + 1), add_special_tokens=False)["input_ids"]
sp1 = SamplingParams(max_tokens=1, temperature=0, ignore_eos=True)
llm.generate([ids]*8, sp1, use_tqdm=False); torch.cuda.synchronize()
print("done", flush=True)
