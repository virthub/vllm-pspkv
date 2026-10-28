#!/usr/bin/env python3
"""Minimal fp4 prefill+decode for nsys bisect (batch threshold)."""
import argparse
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ["PSPKV_PRECISION"] = "fp4"
os.environ["PSPKV_ADAPTIVE_PRECISION"] = "0"
os.environ.setdefault("PSPKV_FORMAT", "1")

import torch  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--input-len", type=int, default=256)
ap.add_argument("--output-len", type=int, default=8)
args = ap.parse_args()

llm = LLM(model="Qwen/Qwen3-4B", enforce_eager=True, max_model_len=2048,
          gpu_memory_utilization=0.6, dtype="float16",
          attention_backend="PSP_KV", enable_chunked_prefill=False,
          enable_prefix_caching=False)
tok = llm.get_tokenizer()
s = tok("The quick brown fox jumps over the lazy dog. " * 40,
        add_special_tokens=False)["input_ids"][:args.input_len]
ids = [s] * args.batch
sp1 = SamplingParams(max_tokens=1, temperature=0, ignore_eos=True)
llm.generate(ids, sp1, use_tqdm=False)
torch.cuda.synchronize()
sp = SamplingParams(max_tokens=args.output_len, temperature=0, ignore_eos=True)
llm.generate(ids, sp, use_tqdm=False)
torch.cuda.synchronize()
print("DONE", args.batch, args.input_len)
