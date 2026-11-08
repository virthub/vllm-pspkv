#!/usr/bin/env python3
"""Minimal decode-only script for nsys kernel profiling."""
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import torch
from vllm import LLM, SamplingParams

prec = os.environ.get("PREC", "e4m3")
batch = int(os.environ.get("BATCH", "1"))
ctx = int(os.environ.get("CTX", "512"))

if prec == "baseline":
    llm = LLM(model="Qwen/Qwen3-4B", enforce_eager=True, max_model_len=2048,
              gpu_memory_utilization=0.6, dtype="float16")
else:
    os.environ["PSPKV_PRECISION"] = prec
    os.environ["PSPKV_ADAPTIVE_PRECISION"] = "0"
    os.environ.setdefault("PSPKV_FORMAT", "1")
    llm = LLM(model="Qwen/Qwen3-4B", enforce_eager=True, max_model_len=2048,
              gpu_memory_utilization=0.6, dtype="float16",
              attention_backend="PSP_KV", enable_chunked_prefill=False,
              enable_prefix_caching=False)

tok = llm.get_tokenizer()
sentence = "The quick brown fox jumps over the lazy dog. "
ids = tok(sentence * (ctx // len(tok(sentence)["input_ids"]) + 1),
          add_special_tokens=False)["input_ids"]
batch_ids = [ids] * batch
sp1 = SamplingParams(max_tokens=1, temperature=0, ignore_eos=True)
# prefill
llm.generate(batch_ids, sp1, use_tqdm=False)
torch.cuda.synchronize()
# decode steps (nsys captures this region)
for _ in range(20):
    llm.generate(batch_ids, sp1, use_tqdm=False)
torch.cuda.synchronize()
print("done", flush=True)
