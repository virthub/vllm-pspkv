#!/usr/bin/env python3
"""One invocation per (prec, splits, batch): decode throughput, block_size=128."""
import os, sys, time
os.environ.setdefault("HF_HUB_OFFLINE","1"); os.environ.setdefault("TRANSFORMERS_OFFLINE","1")
import torch
from vllm import LLM, SamplingParams

prec = sys.argv[1] if len(sys.argv) > 1 else "e4m3"
splits = int(sys.argv[2]) if len(sys.argv) > 2 else 16
batch = int(sys.argv[3]) if len(sys.argv) > 3 else 32
ctx = int(sys.argv[4]) if len(sys.argv) > 4 else 512

os.environ["PSPKV_PRECISION"] = prec
os.environ["PSPKV_ADAPTIVE_PRECISION"] = "0"
os.environ.setdefault("PSPKV_FORMAT", "1")
os.environ["PSPKV_TC_NUM_SPLITS"] = str(splits)

llm = LLM(model="Qwen/Qwen3-4B", enforce_eager=True, max_model_len=ctx+64,
          gpu_memory_utilization=0.6, dtype="float16",
          attention_backend="PSP_KV", enable_chunked_prefill=False,
          enable_prefix_caching=False, block_size=128)
tok = llm.get_tokenizer()
sentence = "The quick brown fox jumps over the lazy dog. "
ids = tok(sentence * (ctx // len(tok(sentence)["input_ids"]) + 1), add_special_tokens=False)["input_ids"]
sp1 = SamplingParams(max_tokens=1, temperature=0, ignore_eos=True)
sp = SamplingParams(max_tokens=64, temperature=0, ignore_eos=True)
b = [ids]*batch
llm.generate(b, sp1, use_tqdm=False); torch.cuda.synchronize()
t0 = time.perf_counter()
llm.generate(b, sp, use_tqdm=False); torch.cuda.synchronize()
dt = time.perf_counter() - t0
print(f"RESULT prec={prec} splits={splits} batch={batch} ctx={ctx} {batch*64/dt:.0f} tok/s", flush=True)
