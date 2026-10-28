"""Measure FlashInfer fp16 vs fp8 KV-cache decode on 4B."""
import os
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from vllm import LLM, SamplingParams  # noqa: E402

for kv_dtype in ["auto", "fp8"]:
    llm = LLM(model="Qwen/Qwen3-4B", enforce_eager=True, max_model_len=2048,
              gpu_memory_utilization=0.6, dtype="float16",
              kv_cache_dtype=kv_dtype, enable_chunked_prefill=False,
              enable_prefix_caching=False)
    tok = llm.get_tokenizer()
    sent = "The quick brown fox jumps over the lazy dog. "
    prompt = sent * (1024 // len(tok(sent)["input_ids"]) + 1)
    sp = SamplingParams(max_tokens=128, temperature=0, ignore_eos=True)
    llm.generate([prompt], sp)
    llm.generate([prompt], sp)
    t0 = time.perf_counter()
    for _ in range(3):
        llm.generate([prompt], sp)
    print(f"flashinfer kv={kv_dtype}: {128*3/(time.perf_counter()-t0):.1f} tok/s",
          flush=True)
    del llm
