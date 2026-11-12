#!/usr/bin/env python3
"""Compare e4m3 (fp8 TC sub-tile split-K) greedy tokens vs lossless reference at ctx=512."""
import os, sys
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import torch
from vllm import LLM, SamplingParams

MODEL = "Qwen/Qwen3-4B"; CTX = 512; N_TOK = 16
tok_prompt = None
toks = {}
for prec in ("lossless", "e4m3"):
    os.environ["PSPKV_PRECISION"] = prec
    os.environ["PSPKV_ADAPTIVE_PRECISION"] = "0"
    os.environ.setdefault("PSPKV_FORMAT", "1")
    llm = LLM(model=MODEL, enforce_eager=True, max_model_len=CTX+N_TOK,
              gpu_memory_utilization=0.6, dtype="float16",
              attention_backend="PSP_KV", enable_chunked_prefill=False,
              enable_prefix_caching=False, block_size=128)
    tok = llm.get_tokenizer()
    sent = "The quick brown fox jumps over the lazy dog. "
    ids = tok(sent, add_special_tokens=False)["input_ids"]
    ids = (ids * ((CTX + len(ids) - 1) // len(ids) + 1))[:CTX]
    tok_prompt = ids
    sp = SamplingParams(max_tokens=N_TOK, temperature=0, ignore_eos=True)
    out = llm.generate([ids], sp, use_tqdm=False)
    toks[prec] = out[0].outputs[0].token_ids
    print(f"PREC={prec} tokens={toks[prec]}", flush=True)
    del llm
    torch.cuda.empty_cache()

a, b = toks["lossless"], toks["e4m3"]
match = sum(1 for x, y in zip(a, b) if x == y)
print(f"MATCH {match}/{N_TOK}  lossless==e4m3: {a == b}", flush=True)
