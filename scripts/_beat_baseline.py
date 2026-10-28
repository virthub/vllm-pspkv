"""Measure baseline vs PSP-KV decode tok/s for a few split-K settings."""
import os
import subprocess
import sys

CODE = r'''
import os,time,sys
from vllm import LLM, SamplingParams
model="Qwen/Qwen3-4B"
backend=os.environ.get("BEAT_BACKEND","")
splits=os.environ.get("BEAT_SPLITS","")
if splits: os.environ["PSPKV_TC_NUM_SPLITS"]=splits
if backend=="PSP_KV": os.environ["PSPKV_PRECISION"]="e4m3"
kw=dict(model=model,enforce_eager=True,max_model_len=2048,gpu_memory_utilization=0.6,dtype="float16",enable_chunked_prefill=False,enable_prefix_caching=False)
if backend: kw["attention_backend"]=backend
llm=LLM(**kw)
tok=llm.get_tokenizer(); sent="The quick brown fox jumps over the lazy dog. "; prompt=sent*(1024//len(tok(sent)["input_ids"])+1)
sp=SamplingParams(max_tokens=128,temperature=0,ignore_eos=True)
llm.generate([prompt],sp); llm.generate([prompt],sp)
t0=time.perf_counter()
for _ in range(3): llm.generate([prompt],sp)
print(f"{os.environ.get('BEAT_LABEL','?')}: {128*3/(time.perf_counter()-t0):.1f} tok/s", flush=True)
'''

cases = [
    ("baseline", "", ""),
    ("lossless-s16", "PSP_KV", "16"),
    ("e4m3-s8", "PSP_KV", "8"),
    ("e4m3-s16", "PSP_KV", "16"),
    ("e4m3-s32", "PSP_KV", "32"),
]

for label, backend, splits in cases:
    env = dict(os.environ)
    env["BEAT_BACKEND"] = backend
    env["BEAT_SPLITS"] = splits
    env["BEAT_LABEL"] = label
    if backend == "PSP_KV":
        env["PSPKV_PRECISION"] = "e4m3"
        env["PSPKV_ADAPTIVE_PRECISION"] = ""
        env.pop("PSPKV_ADAPTIVE_PRECISION", None)
    r = subprocess.run([sys.executable, "-c", CODE], env=env,
                       capture_output=True, text=True)
    out = [l for l in r.stdout.splitlines() if "tok/s" in l]
    print(out[-1] if out else f"{label}: FAILED {r.stderr[-200:]}")
