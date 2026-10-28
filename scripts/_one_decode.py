"""Measure one decode config. Usage: _one_decode.py <label> <backend|''> <splits|''>"""
import os
import sys
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from vllm import LLM, SamplingParams  # noqa: E402

label = sys.argv[1]
backend = sys.argv[2]
splits = sys.argv[3] if len(sys.argv) > 3 else ""

if splits:
    os.environ["PSPKV_TC_NUM_SPLITS"] = splits
if backend == "PSP_KV":
    os.environ["PSPKV_PRECISION"] = "e4m3"
    os.environ.pop("PSPKV_ADAPTIVE_PRECISION", None)
    if label.startswith("lossless"):
        os.environ["PSPKV_PRECISION"] = "lossless"

kw = dict(model="Qwen/Qwen3-4B", enforce_eager=True, max_model_len=2048,
          gpu_memory_utilization=0.6, dtype="float16",
          enable_chunked_prefill=False, enable_prefix_caching=False)
if backend:
    kw["attention_backend"] = backend
llm = LLM(**kw)
tok = llm.get_tokenizer()
sent = "The quick brown fox jumps over the lazy dog. "
prompt = sent * (1024 // len(tok(sent)["input_ids"]) + 1)
sp = SamplingParams(max_tokens=128, temperature=0, ignore_eos=True)
llm.generate([prompt], sp)
llm.generate([prompt], sp)
t0 = time.perf_counter()
for _ in range(3):
    llm.generate([prompt], sp)
print(f"{label}: {128 * 3 / (time.perf_counter() - t0):.1f} tok/s", flush=True)
