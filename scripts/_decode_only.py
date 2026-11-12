#!/usr/bin/env python3
"""Minimal decode-only run for ncu profiling (4B, 512 ctx, batch 1, int8)."""
import os, sys
os.environ.setdefault("HF_HUB_OFFLINE","1"); os.environ.setdefault("TRANSFORMERS_OFFLINE","1")
os.environ["PSP_KV_BLOCK_SIZE"] = "128"
os.environ["PSPKV_PRECISION"] = "int8"
os.environ["PSPKV_ADAPTIVE_PRECISION"] = "0"
os.environ.setdefault("PSPKV_FORMAT", "1")
import torch
from vllm import LLM, SamplingParams

ctx = 512
llm = LLM(model="Qwen/Qwen3-4B", enforce_eager=True, max_model_len=ctx+8,
          gpu_memory_utilization=0.6, dtype="float16",
          enable_chunked_prefill=False, enable_prefix_caching=False,
          block_size=128, attention_backend="PSP_KV")
tok = llm.get_tokenizer()
sentence = "The quick brown fox jumps over the lazy dog. "
ids = tok(sentence * (ctx // len(tok(sentence)["input_ids"]) + 1), add_special_tokens=False)["input_ids"]
b = [ids]*1
sp1 = SamplingParams(max_tokens=1, temperature=0, ignore_eos=True)
sp = SamplingParams(max_tokens=16, temperature=0, ignore_eos=True)
llm.generate(b, sp1, use_tqdm=False); torch.cuda.synchronize()
llm.generate(b, sp, use_tqdm=False); torch.cuda.synchronize()
print("done", flush=True)
