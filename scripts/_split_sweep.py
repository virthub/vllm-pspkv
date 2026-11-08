#!/usr/bin/env python3
"""Sweep PSPKV_TC_NUM_SPLITS for fp8 decode throughput at high batch."""
import os, time
os.environ.setdefault("HF_HUB_OFFLINE","1"); os.environ.setdefault("TRANSFORMERS_OFFLINE","1")
import torch
from vllm import LLM, SamplingParams

def mk(prec, splits):
    os.environ["PSPKV_PRECISION"] = prec
    os.environ["PSPKV_ADAPTIVE_PRECISION"] = "0"
    os.environ.setdefault("PSPKV_FORMAT", "1")
    os.environ["PSPKV_TC_NUM_SPLITS"] = str(splits)
    return LLM(model="Qwen/Qwen3-4B", enforce_eager=True, max_model_len=2048,
               gpu_memory_utilization=0.6, dtype="float16",
               attention_backend="PSP_KV", enable_chunked_prefill=False,
               enable_prefix_caching=False)

tok = mk("e4m3", 16).get_tokenizer()
sentence = "The quick brown fox jumps over the lazy dog. "
ids = tok(sentence * (512 // len(tok(sentence)["input_ids"]) + 1), add_special_tokens=False)["input_ids"]
sp1 = SamplingParams(max_tokens=1, temperature=0, ignore_eos=True)
sp = SamplingParams(max_tokens=64, temperature=0, ignore_eos=True)

for splits in [2, 4, 8, 16, 32]:
    llm = mk("e4m3", splits)
    for batch in [8, 32]:
        b = [ids]*batch
        llm.generate(b, sp1, use_tqdm=False); torch.cuda.synchronize()
        t0 = time.perf_counter()
        llm.generate(b, sp, use_tqdm=False); torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        tok_s = batch*64/dt
        print(f"splits={splits:2d} batch={batch:2d}  {tok_s:7.0f} tok/s", flush=True)
    del llm; torch.cuda.empty_cache()
