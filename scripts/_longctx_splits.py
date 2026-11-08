#!/usr/bin/env python3
"""Long-context decode: sweep PSPKV_TC_NUM_SPLITS at 8K ctx, batch 1 (one load)."""
import os, time
os.environ.setdefault("HF_HUB_OFFLINE","1"); os.environ.setdefault("TRANSFORMERS_OFFLINE","1")
os.environ["PSPKV_PRECISION"] = "e4m3"
os.environ["PSPKV_ADAPTIVE_PRECISION"] = "0"
os.environ.setdefault("PSPKV_FORMAT", "1")
import torch
from vllm import LLM, SamplingParams

MODEL = "Qwen/Qwen3-4B"
CTX = 8192
N_TOK = 64


def make_prompt(tok, n):
    sent = "The quick brown fox jumps over the lazy dog. "
    ids = tok(sent, add_special_tokens=False)["input_ids"]
    reps = (n + len(ids) - 1) // len(ids) + 1
    return (ids * reps)[:n]


llm = LLM(model=MODEL, enforce_eager=True, max_model_len=CTX + N_TOK,
          gpu_memory_utilization=0.6, dtype="float16",
          attention_backend="PSP_KV", enable_chunked_prefill=False,
          enable_prefix_caching=False, block_size=128)
tok = llm.get_tokenizer()
ids = make_prompt(tok, CTX)
sp = SamplingParams(max_tokens=N_TOK, temperature=0, ignore_eos=True)

for splits in [2, 4, 8, 16, 32]:
    os.environ["PSPKV_TC_NUM_SPLITS"] = str(splits)
    llm.generate([ids], sp, use_tqdm=False)  # warmup
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    llm.generate([ids], sp, use_tqdm=False)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    print(f"ctx={CTX} splits={splits:2d}  {N_TOK/dt:7.1f} tok/s", flush=True)
