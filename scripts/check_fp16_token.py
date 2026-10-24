import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ["PSPKV_PRECISION"] = "lossless"
from vllm import LLM, SamplingParams

sp = SamplingParams(max_tokens=1, temperature=0, ignore_eos=True)
prompt = ("The quick brown fox jumps over the lazy dog. " * 26)[:512]

for backend in ["baseline", "PSP_KV"]:
    kw = dict(model="Qwen/Qwen3-4B", enforce_eager=True, max_model_len=2048,
              gpu_memory_utilization=0.6, dtype="float16")
    if backend == "PSP_KV":
        kw["attention_backend"] = "PSP_KV"
    llm = LLM(**kw)
    o = llm.generate([prompt], sp)
    print(backend, "token", o[0].outputs[0].token_ids,
          "logprob", o[0].outputs[0].logprobs)
    del llm
