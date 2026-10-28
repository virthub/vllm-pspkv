"""Minimal decode-only run for nsys kernel-level profiling."""
import os
import sys

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from vllm import LLM, SamplingParams  # noqa: E402

model = sys.argv[1]
backend = sys.argv[2] if len(sys.argv) > 2 else None
out_len = int(sys.argv[3]) if len(sys.argv) > 3 else 32

kwargs = dict(model=model, enforce_eager=True, max_model_len=2048,
              gpu_memory_utilization=0.6, dtype="float16",
              enable_prefix_caching=False)
if backend:
    kwargs["attention_backend"] = backend

llm = LLM(**kwargs)
tok = llm.get_tokenizer()
sentence = "The quick brown fox jumps over the lazy dog. "
prompt = sentence * (1024 // len(tok(sentence)["input_ids"]) + 1)
sp = SamplingParams(max_tokens=out_len, temperature=0, ignore_eos=True)
llm.generate([prompt], sp)
del llm
