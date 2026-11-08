#!/usr/bin/env python3
"""Validate + benchmark the token-major fp8 kernel (PSPKV_TC_TOKEN_MAJOR=1)."""
import os, time
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import torch
from vllm import LLM, SamplingParams

MODEL = "Qwen/Qwen3-4B"


def mk(prec, tm, splits):
    os.environ["PSPKV_PRECISION"] = prec
    os.environ["PSPKV_ADAPTIVE_PRECISION"] = "0"
    os.environ.setdefault("PSPKV_FORMAT", "1")
    if tm:
        os.environ["PSPKV_TC_TOKEN_MAJOR"] = "1"
    else:
        os.environ.pop("PSPKV_TC_TOKEN_MAJOR", None)
    os.environ["PSPKV_TC_NUM_SPLITS"] = str(splits)
    return LLM(model=MODEL, enforce_eager=True, max_model_len=2048,
               gpu_memory_utilization=0.6, dtype="float16",
               attention_backend="PSP_KV", enable_chunked_prefill=False,
               enable_prefix_caching=False)


tok = mk("e4m3", False, 16).get_tokenizer()
sentence = "The quick brown fox jumps over the lazy dog. "
ids = tok(sentence * (512 // len(tok(sentence)["input_ids"]) + 1),
          add_special_tokens=False)["input_ids"]
sp1 = SamplingParams(max_tokens=1, temperature=0, ignore_eos=True)
sp = SamplingParams(max_tokens=64, temperature=0, ignore_eos=True)
sp32 = SamplingParams(max_tokens=32, temperature=0, ignore_eos=True)

# Reference tokens (default fp8 path) for the match check.
ref = mk("e4m3", False, 16)
ref_tokens = [list(o.outputs[0].token_ids) for o in
              ref.generate([ids, ids], sp32, use_tqdm=False)]
del ref; torch.cuda.empty_cache()

for tm in [False, True]:
    llm = mk("e4m3", tm, 16)
    out = [list(o.outputs[0].token_ids) for o in
           llm.generate([ids, ids], sp32, use_tqdm=False)]
    match = sum(a == b for ra, oa in zip(ref_tokens, out)
                for a, b in zip(ra, oa)) / max(sum(len(r) for r in ref_tokens), 1)
    print(f"TM={int(tm)} match-vs-default={match*100:.1f}%  tok0={out[0][:6]}", flush=True)

    for batch in [1, 32]:
        b = [ids] * batch
        llm.generate(b, sp1, use_tqdm=False); torch.cuda.synchronize()
        t0 = time.perf_counter()
        llm.generate(b, sp, use_tqdm=False); torch.cuda.synchronize()
        tok_s = batch * 64 / (time.perf_counter() - t0)
        print(f"TM={int(tm)} batch={batch:2d}  {tok_s:7.0f} tok/s", flush=True)
    del llm; torch.cuda.empty_cache()
