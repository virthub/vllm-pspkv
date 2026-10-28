#!/usr/bin/env python3
"""Minimal batched decode for nsys profiling (baseline backend, no PSP_KV)."""
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402

llm = LLM(model="Qwen/Qwen3-4B", enforce_eager=True, max_model_len=2048,
          gpu_memory_utilization=0.6, dtype="float16",
          enable_chunked_prefill=False, enable_prefix_caching=False)

tok = llm.get_tokenizer()
sentence = "The quick brown fox jumps over the lazy dog. "
tok_ids = tok(sentence * 24, add_special_tokens=False)["input_ids"]
batch = 8
ids = [tok_ids] * batch

sp1 = SamplingParams(max_tokens=1, temperature=0, ignore_eos=True)
llm.generate(ids, sp1, use_tqdm=False)
sp = SamplingParams(max_tokens=8, temperature=0, ignore_eos=True)
llm.generate(ids, sp, use_tqdm=False)

torch.cuda.synchronize()
llm.generate(ids, sp, use_tqdm=False)
torch.cuda.synchronize()
print("DONE")
