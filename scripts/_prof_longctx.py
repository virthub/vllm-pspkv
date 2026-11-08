#!/usr/bin/env python3
"""Profile 8K decode via vLLM's built-in torch profiler."""
import os, glob, json
os.environ.setdefault("HF_HUB_OFFLINE","1"); os.environ.setdefault("TRANSFORMERS_OFFLINE","1")
os.environ["PSPKV_PRECISION"] = "e4m3"
os.environ["PSPKV_ADAPTIVE_PRECISION"] = "0"
os.environ.setdefault("PSPKV_FORMAT", "1")
import torch
from vllm import LLM, SamplingParams
from vllm.config.profiler import ProfilerConfig

MODEL = "Qwen/Qwen3-4B"
CTX = 8192
N_TOK = 1
OUT = "/tmp/prof_longctx"

def make_prompt(tok, n):
    sent = "The quick brown fox jumps over the lazy dog. "
    ids = tok(sent, add_special_tokens=False)["input_ids"]
    reps = (n + len(ids) - 1) // len(ids) + 1
    return (ids * reps)[:n]

pc = ProfilerConfig(profiler="torch", torch_profiler_dir=OUT)
llm = LLM(model=MODEL, enforce_eager=True, max_model_len=CTX + N_TOK,
          gpu_memory_utilization=0.6, dtype="float16",
          attention_backend="PSP_KV", enable_chunked_prefill=False,
          enable_prefix_caching=False, block_size=128, profiler_config=pc)
tok = llm.get_tokenizer()
ids = make_prompt(tok, CTX)
sp = SamplingParams(max_tokens=N_TOK, temperature=0, ignore_eos=True)
llm.generate([ids], sp, use_tqdm=False)  # warmup
torch.cuda.synchronize()

llm.llm_engine.engine_core.profile(True)
llm.generate([ids], sp, use_tqdm=False)
llm.llm_engine.engine_core.profile(False)

# Parse the chrome trace JSON for CUDA kernel durations.
traces = glob.glob(os.path.join(OUT, "*.json"))
print(f"traces: {traces}", flush=True)
from collections import defaultdict
agg = defaultdict(float)
for tp in traces:
    data = json.load(open(tp))
    for evt in data.get("traceEvents", []):
        if evt.get("cat") == "kernel":
            name = evt.get("name", "")[:60]
            dur = evt.get("dur", 0)
            agg[name] += dur
tot = sum(agg.values())
print(f"total CUDA time {tot/1e3:.2f} ms over {N_TOK} tok")
for k, v in sorted(agg.items(), key=lambda x: -x[1])[:18]:
    print(f"  {v/1e3:8.2f} ms  {v/tot*100:5.1f}%  {k}")
