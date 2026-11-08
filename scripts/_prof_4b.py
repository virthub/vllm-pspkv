#!/usr/bin/env python3
"""Profile the 4B decode at batch 32: kernel time breakdown."""
import os, sys, time
os.environ.setdefault("HF_HUB_OFFLINE","1"); os.environ.setdefault("TRANSFORMERS_OFFLINE","1")
import torch
from vllm import LLM, SamplingParams

batch = int(sys.argv[1]) if len(sys.argv) > 1 else 32
ctx = int(sys.argv[2]) if len(sys.argv) > 2 else 512

os.environ["PSP_KV_BLOCK_SIZE"] = "128"
os.environ["PSPKV_PRECISION"] = "e4m3"
os.environ["PSPKV_ADAPTIVE_PRECISION"] = "0"
os.environ.setdefault("PSPKV_FORMAT", "1")

llm = LLM(model="Qwen/Qwen3-4B", enforce_eager=True, max_model_len=ctx+64,
          gpu_memory_utilization=0.6, dtype="float16",
          enable_chunked_prefill=False, enable_prefix_caching=False,
          block_size=128, attention_backend="PSP_KV")
tok = llm.get_tokenizer()
sentence = "The quick brown fox jumps over the lazy dog. "
ids = tok(sentence * (ctx // len(tok(sentence)["input_ids"]) + 1), add_special_tokens=False)["input_ids"]
sp = SamplingParams(max_tokens=16, temperature=0, ignore_eos=True)
b = [ids]*batch
llm.generate(b, SamplingParams(max_tokens=1, temperature=0, ignore_eos=True), use_tqdm=False)
torch.cuda.synchronize()

from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CUDA]) as prof:
    llm.generate(b, sp, use_tqdm=False)
    torch.cuda.synchronize()

# aggregate kernel times by name
agg = {}
for evt in prof.key_averages():
    if evt.device_type == torch.autograd.DeviceType.CUDA:
        agg.setdefault(evt.key, 0.0)
        agg[evt.key] += evt.self_device_time_total
rows = sorted(agg.items(), key=lambda x: -x[1])
tot = sum(v for _, v in rows)
print(f"=== total CUDA {tot/1e3:.2f} ms, top kernels ===")
for k, v in rows[:20]:
    print(f"{v/1e3:8.2f} ms  {v/tot*100:5.1f}%  {k[:80]}")
