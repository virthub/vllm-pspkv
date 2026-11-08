#!/usr/bin/env python3
"""One invocation per (backend, block, batch): PURE decode throughput."""
import os, sys, time
os.environ.setdefault("HF_HUB_OFFLINE","1"); os.environ.setdefault("TRANSFORMERS_OFFLINE","1")
import torch
from vllm import LLM, SamplingParams

backend = sys.argv[1] if len(sys.argv) > 1 else "pspkv"
block = int(sys.argv[2]) if len(sys.argv) > 2 else 16
batch = int(sys.argv[3]) if len(sys.argv) > 3 else 32
ctx = int(sys.argv[4]) if len(sys.argv) > 4 else 512

kw = dict(block_size=block)
if backend == "pspkv":
    os.environ["PSP_KV_BLOCK_SIZE"] = str(block)
    os.environ["PSPKV_PRECISION"] = "e4m3"
    os.environ["PSPKV_ADAPTIVE_PRECISION"] = "0"
    os.environ.setdefault("PSPKV_FORMAT", "1")
    kw["attention_backend"] = "PSP_KV"

llm = LLM(model="Qwen/Qwen3-4B", enforce_eager=True, max_model_len=ctx+64,
          gpu_memory_utilization=0.6, dtype="float16",
          enable_chunked_prefill=False, enable_prefix_caching=False, **kw)
tok = llm.get_tokenizer()
sentence = "The quick brown fox jumps over the lazy dog. "
ids = tok(sentence * (ctx // len(tok(sentence)["input_ids"]) + 1), add_special_tokens=False)["input_ids"]
sp1 = SamplingParams(max_tokens=1, temperature=0, ignore_eos=True)
sp = SamplingParams(max_tokens=64, temperature=0, ignore_eos=True)
b = [ids]*batch

# prefill time (3 runs, averaged)
llm.generate(b, sp1, use_tqdm=False); torch.cuda.synchronize()
t0 = time.perf_counter()
for _ in range(3):
    llm.generate(b, sp1, use_tqdm=False)
torch.cuda.synchronize()
pf_ms = (time.perf_counter() - t0) / 3 * 1e3

# decode time (subtract prefill)
llm.generate(b, sp1, use_tqdm=False); torch.cuda.synchronize()
t0 = time.perf_counter()
llm.generate(b, sp, use_tqdm=False)
torch.cuda.synchronize()
decode_s = (time.perf_counter() - t0) - pf_ms / 1e3
tok_s = batch * 64 / max(decode_s, 1e-6)
print(f"RESULT backend={backend} block={block} batch={batch} ctx={ctx} pf_ms={pf_ms:.1f} {tok_s:.0f} tok/s", flush=True)
