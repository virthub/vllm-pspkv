#!/usr/bin/env python3
"""Compare lossless vs baseline logprobs for the first 2 decode tokens."""
import os, sys
os.environ.setdefault("HF_HUB_OFFLINE","1"); os.environ.setdefault("TRANSFORMERS_OFFLINE","1")
import torch
from vllm import LLM, SamplingParams

kind = sys.argv[1] if len(sys.argv) > 1 else "lossless"
MODEL = "Qwen/Qwen3-4B"; CTX = 128; N_TOK = 2
kw = dict()
if kind == "lossless":
    os.environ["PSPKV_PRECISION"] = "lossless"
    os.environ["PSPKV_ADAPTIVE_PRECISION"] = "0"
    os.environ.setdefault("PSPKV_FORMAT", "1")
    kw = dict(attention_backend="PSP_KV", block_size=128)
llm = LLM(model=MODEL, enforce_eager=True, max_model_len=CTX+N_TOK,
          gpu_memory_utilization=0.6, dtype="float16",
          enable_chunked_prefill=False, enable_prefix_caching=False, **kw)
tok = llm.get_tokenizer()
sent = "The quick brown fox jumps over the lazy dog. "
ids = tok(sent, add_special_tokens=False)["input_ids"]
ids = (ids * ((CTX + len(ids) - 1) // len(ids) + 1))[:CTX]
sp = SamplingParams(max_tokens=N_TOK, temperature=0, ignore_eos=True, logprobs=3)
out = llm.generate([ids], sp, use_tqdm=False)[0]
for i, o in enumerate(out.outputs[0].logprobs):
    top = {k: round(v, 4) for k, v in o.items()}
    print(f"{kind} token{i} logprobs={top}")
