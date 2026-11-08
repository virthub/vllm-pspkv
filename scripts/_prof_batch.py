#!/usr/bin/env python3
"""Profile a batch-32 decode step (torch profiler) to find the kernel bottleneck."""
import os, sys
os.environ.setdefault("HF_HUB_OFFLINE","1"); os.environ.setdefault("TRANSFORMERS_OFFLINE","1")
mode = sys.argv[1] if len(sys.argv) > 1 else "baseline"
import torch
from vllm import LLM, SamplingParams

BATCH = 32
if mode == "pspkv":
    os.environ["PSPKV_PRECISION"] = "e4m3"
    os.environ["PSPKV_ADAPTIVE_PRECISION"] = "0"
    os.environ.setdefault("PSPKV_FORMAT", "1")
    kw = dict(attention_backend="PSP_KV", block_size=128)
else:
    kw = dict(block_size=128)
llm = LLM(model="Qwen/Qwen3-4B", enforce_eager=True, max_model_len=2048,
          gpu_memory_utilization=0.6, dtype="float16",
          enable_chunked_prefill=False, enable_prefix_caching=False, **kw)
tok = llm.get_tokenizer()
sentence = "The quick brown fox jumps over the lazy dog. "
ids = tok(sentence * (512 // len(tok(sentence)["input_ids"]) + 1), add_special_tokens=False)["input_ids"]
b = [ids]*BATCH
sp1 = SamplingParams(max_tokens=1, temperature=0, ignore_eos=True)
sp = SamplingParams(max_tokens=4, temperature=0, ignore_eos=True)
llm.generate(b, sp1, use_tqdm=False); torch.cuda.synchronize()

from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CUDA]) as prof:
    llm.generate(b, sp, use_tqdm=False)
    torch.cuda.synchronize()
print(f"=== {mode} batch={BATCH} ===")
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15))
