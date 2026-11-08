#!/usr/bin/env python3
"""Probe the lossless dispatch: precision table + stats + greedy tokens."""
import os, sys
os.environ.setdefault("HF_HUB_OFFLINE","1"); os.environ.setdefault("TRANSFORMERS_OFFLINE","1")
prec = sys.argv[1] if len(sys.argv) > 1 else "lossless"
os.environ["PSPKV_PRECISION"] = prec
os.environ["PSPKV_ADAPTIVE_PRECISION"] = "0"
os.environ.setdefault("PSPKV_FORMAT", "1")
import torch
from vllm import LLM, SamplingParams

MODEL = "Qwen/Qwen3-4B"; CTX = 128; N_TOK = 8
llm = LLM(model=MODEL, enforce_eager=True, max_model_len=CTX+N_TOK,
          gpu_memory_utilization=0.6, dtype="float16",
          attention_backend="PSP_KV", enable_chunked_prefill=False,
          enable_prefix_caching=False, block_size=128)
tok = llm.get_tokenizer()
sent = "The quick brown fox jumps over the lazy dog. "
ids = tok(sent, add_special_tokens=False)["input_ids"]
ids = (ids * ((CTX + len(ids) - 1) // len(ids) + 1))[:CTX]
sp = SamplingParams(max_tokens=N_TOK, temperature=0, ignore_eos=True)
out = llm.generate([ids], sp, use_tqdm=False)
toks = out[0].outputs[0].token_ids
print(f"PREC={prec} tokens={toks}", flush=True)
# dump the runner's PSP-KV precision stats
try:
    core = llm.llm_engine.engine_core
    # SyncMPClient: find the model executor's runner
    runner = None
    for attr in ("model_executor", "engine_core"):
        o = getattr(core, attr, None)
    # Try the worker's runner via the model_executor
    me = core.model_executor
    for name in dir(me):
        if "runner" in name.lower():
            r = getattr(me, name)
            if hasattr(r, "psp_kv_stats"):
                st = r.psp_kv_stats(4)
                print(f"PREC={prec} stats={ {k: v for k, v in st.items() if 'precision' in k or 'fmt' in k or 'avg' in k} }", flush=True)
                break
except Exception as e:
    print(f"PREC={prec} probe-err={str(e)[:120]}", flush=True)
