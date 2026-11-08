#!/usr/bin/env python3
"""Test the opt-in native-fp8-MMA / fp8-query decode paths vs the default fp8 TC."""
import os, time
os.environ.setdefault("HF_HUB_OFFLINE","1"); os.environ.setdefault("TRANSFORMERS_OFFLINE","1")
import torch
from vllm import LLM, SamplingParams

def mk(env):
    os.environ["PSPKV_PRECISION"] = "e4m3"
    os.environ["PSPKV_ADAPTIVE_PRECISION"] = "0"
    os.environ.setdefault("PSPKV_FORMAT", "1")
    for k in ["PSPKV_NATIVE_FP8_MMA", "PSPKV_FP8_QUERY", "PSPKV_TC_NUM_SPLITS"]:
        os.environ.pop(k, None)
    for k, v in env.items():
        os.environ[k] = v
    return LLM(model="Qwen/Qwen3-4B", enforce_eager=True, max_model_len=2048,
               gpu_memory_utilization=0.6, dtype="float16",
               attention_backend="PSP_KV", enable_chunked_prefill=False,
               enable_prefix_caching=False)

tok = mk({}).get_tokenizer()
sentence = "The quick brown fox jumps over the lazy dog. "
ids = tok(sentence * (512 // len(tok(sentence)["input_ids"]) + 1), add_special_tokens=False)["input_ids"]
sp1 = SamplingParams(max_tokens=1, temperature=0, ignore_eos=True)
sp = SamplingParams(max_tokens=64, temperature=0, ignore_eos=True)

for name, env in [
    ("default(fp16-dequant)", {}),
    ("native-fp8-mma", {"PSPKV_NATIVE_FP8_MMA": "1"}),
    ("fp8-query", {"PSPKV_FP8_QUERY": "1"}),
    ("native+splits4", {"PSPKV_NATIVE_FP8_MMA": "1", "PSPKV_TC_NUM_SPLITS": "4"}),
]:
    llm = mk(env)
    for batch in [1, 32]:
        b = [ids]*batch
        llm.generate(b, sp1, use_tqdm=False); torch.cuda.synchronize()
        t0 = time.perf_counter()
        llm.generate(b, sp, use_tqdm=False); torch.cuda.synchronize()
        tok_s = batch*64/(time.perf_counter()-t0)
        print(f"{name:22s} batch={batch:2d}  {tok_s:7.0f} tok/s", flush=True)
    del llm; torch.cuda.empty_cache()
