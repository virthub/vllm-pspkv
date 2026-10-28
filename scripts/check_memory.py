import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
from vllm import LLM

for prec in ["baseline", "lossless", "e4m3", "fp4"]:
    kw = dict(model="Qwen/Qwen3-4B", enforce_eager=True, max_model_len=2048,
              gpu_memory_utilization=0.6, dtype="float16")
    if prec != "baseline":
        os.environ["PSPKV_PRECISION"] = prec
        kw["attention_backend"] = "PSP_KV"
    llm = LLM(**kw)
    cc = llm.llm_engine.vllm_config.cache_config
    print(prec, "num_gpu_blocks", cc.num_gpu_blocks)
    del llm
