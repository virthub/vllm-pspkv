#!/usr/bin/env python3
"""Profile a batched decode step with vLLM's built-in torch profiler (CPU+CUDA).

Usage:
  .venv/bin/python scripts/_profile_decode.py [baseline|pspkv] [out_dir]
"""
import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

mode = sys.argv[1] if len(sys.argv) > 1 else "baseline"
out_dir = sys.argv[2] if len(sys.argv) > 2 else "/tmp/prof_decode"

import torch  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402
from vllm.config.profiler import ProfilerConfig  # noqa: E402

pc = ProfilerConfig(profiler="torch", torch_profiler_dir=out_dir)

if mode == "pspkv":
    os.environ["PSPKV_PRECISION"] = "lossless"
    os.environ["PSPKV_ADAPTIVE_PRECISION"] = "0"
    os.environ.setdefault("PSPKV_FORMAT", "1")
    llm = LLM(model="Qwen/Qwen3-4B", enforce_eager=True, max_model_len=2048,
              gpu_memory_utilization=0.6, dtype="float16",
              attention_backend="PSP_KV", enable_chunked_prefill=False,
              enable_prefix_caching=False, profiler_config=pc)
else:
    llm = LLM(model="Qwen/Qwen3-4B", enforce_eager=True, max_model_len=2048,
              gpu_memory_utilization=0.6, dtype="float16",
              enable_chunked_prefill=False, enable_prefix_caching=False,
              profiler_config=pc)

tok = llm.get_tokenizer()
tok_ids = tok("The quick brown fox jumps over the lazy dog. " * 24,
              add_special_tokens=False)["input_ids"]
ids = [tok_ids] * 8

sp1 = SamplingParams(max_tokens=1, temperature=0, ignore_eos=True)
llm.generate(ids, sp1, use_tqdm=False)  # prefill warmup

# Start profiling, run a decode step, stop.
llm.llm_engine.engine_core.profile(True)
sp = SamplingParams(max_tokens=8, temperature=0, ignore_eos=True)
llm.generate(ids, sp, use_tqdm=False)
llm.llm_engine.engine_core.profile(False)
print("PROFILED", out_dir)
