"""Sweep PSPKV_TC_NUM_SPLITS on 4B e4m3 decode."""
import os
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from vllm import LLM, SamplingParams  # noqa: E402

model = "Qwen/Qwen3-4B"
tokn = None

for splits in [1, 2, 4, 8, 16]:
    os.environ["PSPKV_TC_NUM_SPLITS"] = str(splits)
    os.environ["PSPKV_PRECISION"] = "e4m3"
    os.environ.pop("PSPKV_ADAPTIVE_PRECISION", None)
    llm = LLM(model=model, enforce_eager=True, max_model_len=2048,
              gpu_memory_utilization=0.6, dtype="float16",
              attention_backend="PSP_KV", enable_chunked_prefill=False,
              enable_prefix_caching=False)
    if tokn is None:
        tokn = llm.get_tokenizer()
    sentence = "The quick brown fox jumps over the lazy dog. "
    prompt = sentence * (1024 // len(tokn(sentence)["input_ids"]) + 1)
    sp = SamplingParams(max_tokens=1, temperature=0, ignore_eos=True)
    llm.generate([prompt], sp)  # warmup
    sp128 = SamplingParams(max_tokens=128, temperature=0, ignore_eos=True)
    llm.generate([prompt], sp128)  # warmup decode
    t0 = time.perf_counter()
    for _ in range(3):
        llm.generate([prompt], sp128)
    dt = time.perf_counter() - t0
    print(f"splits={splits:2d}  decode={128*3/dt:7.1f} tok/s")
    del llm
