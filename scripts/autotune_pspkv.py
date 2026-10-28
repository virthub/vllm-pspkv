"""Runtime split-K autotuner for PSP-KV decode (Option C).

FlashInfer's ``fast_decode_plan`` benchmarks its compiled kernel variants and
picks the fastest at init. PSP-KV's kernels are build-time compiled, but its one
runtime tunable (the tensor-core split-K count) is read from
``PSPKV_TC_NUM_SPLITS`` on every decode call. This script sweeps that count at
runtime -- without reloading the model -- and leaves the fastest one set.

Usage:
  .venv/bin/python scripts/autotune_pspkv.py <model> [precision] [out_len]
"""
import os
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from vllm import LLM, SamplingParams  # noqa: E402

model = sys.argv[1]
precision = sys.argv[2] if len(sys.argv) > 2 else "e4m3"
out_len = int(sys.argv[3]) if len(sys.argv) > 3 else 128
candidates = [2, 4, 8, 16, 24, 32]

if precision == "lossless":
    os.environ["PSPKV_PRECISION"] = "lossless"
elif precision == "e4m3":
    os.environ["PSPKV_PRECISION"] = "e4m3"
elif precision == "fp4":
    os.environ["PSPKV_PRECISION"] = "fp4"
os.environ.pop("PSPKV_ADAPTIVE_PRECISION", None)

llm = LLM(model=model, enforce_eager=True, max_model_len=2048,
          gpu_memory_utilization=0.6, dtype="float16",
          attention_backend="PSP_KV", enable_chunked_prefill=False,
          enable_prefix_caching=False)
rpc = llm.llm_engine.engine_core.collective_rpc

tok = llm.get_tokenizer()
sent = "The quick brown fox jumps over the lazy dog. "
prompt = sent * (1024 // len(tok(sent)["input_ids"]) + 1)
sp = SamplingParams(max_tokens=out_len, temperature=0, ignore_eos=True)

best = (None, -1.0)
for s in candidates:
    rpc("psp_kv_set_num_splits", args=(s,))
    llm.generate([prompt], sp)          # warmup under this split
    llm.generate([prompt], sp)          # warmup (cudagraph/jit settle)
    t0 = time.perf_counter()
    for _ in range(3):
        llm.generate([prompt], sp)
    tok_s = out_len * 3 / (time.perf_counter() - t0)
    print(f"split={s:2d}  {tok_s:7.1f} tok/s")
    if tok_s > best[1]:
        best = (s, tok_s)

rpc("psp_kv_set_num_splits", args=(best[0],))
print(f"selected split={best[0]}  ({best[1]:.1f} tok/s)")
