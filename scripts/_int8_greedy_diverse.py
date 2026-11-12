#!/usr/bin/env python3
"""Diverse-prompt greedy-match for int8 vs e4m3 at 512 ctx (Qwen3-1.7B)."""
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ["PSP_KV_BLOCK_SIZE"] = "128"

import torch
from vllm import LLM, SamplingParams

MODEL = "Qwen/Qwen3-1.7B"
CTX = 512
OUT = 32

_SENTENCES = [
    "The quick brown fox jumps over the lazy dog. ",
    "Machine learning models compress knowledge into hidden states. ",
    "Attention mechanisms weigh the relevance of every token in the sequence. ",
    "Quantum computers exploit superposition and entanglement for parallelism. ",
    "The Pacific ocean covers more area than all landmasses combined. ",
    "Gradient descent follows the steepest slope of the loss landscape. ",
    "Photosynthesis converts sunlight into chemical energy in plants. ",
    "Encryption transforms plaintext into ciphertext using a secret key. ",
    "The mitochondria are often called the powerhouse of the cell. ",
    "A compiler translates high-level source code into machine instructions. ",
]


def build_prompt(tok):
    ids = []
    i = 0
    while len(ids) < CTX:
        ids += tok(_SENTENCES[i % len(_SENTENCES)], add_special_tokens=False)["input_ids"]
        i += 1
    return ids[:CTX]


def make_llm(prec):
    if prec == "baseline":
        return LLM(model=MODEL, enforce_eager=True, max_model_len=CTX + OUT,
                   gpu_memory_utilization=0.6, dtype="float16", block_size=128)
    os.environ["PSPKV_PRECISION"] = prec
    os.environ["PSPKV_ADAPTIVE_PRECISION"] = "0"
    os.environ.setdefault("PSPKV_FORMAT", "1")
    return LLM(model=MODEL, enforce_eager=True, max_model_len=CTX + OUT,
               gpu_memory_utilization=0.6, dtype="float16",
               attention_backend="PSP_KV", enable_chunked_prefill=False,
               enable_prefix_caching=False, block_size=128)


def main():
    probe = make_llm("baseline")
    tok = probe.get_tokenizer()
    del probe
    torch.cuda.empty_cache()
    prompts = [build_prompt(tok), build_prompt(tok)[:CTX // 2]]
    sp = SamplingParams(max_tokens=OUT, temperature=0, ignore_eos=True)

    ref = None
    res = {}
    for prec in ["baseline", "e4m3", "int8"]:
        llm = make_llm(prec)
        out = [list(o.outputs[0].token_ids) for o in llm.generate(prompts, sp, use_tqdm=False)]
        if prec == "baseline":
            ref = out
            print(f"baseline={out}", flush=True)
        else:
            m = sum(a == b for ra, oa in zip(ref, out) for a, b in zip(ra, oa))
            res[prec] = m / max(sum(len(r) for r in ref), 1)
            print(f"{prec}={out}", flush=True)
            print(f"greedy_match[{prec}]={res[prec]*100:.1f}%", flush=True)
        del llm
        torch.cuda.empty_cache()
    print(f"\nSUMMARY diverse ctx={CTX} out={OUT} "
          + "  ".join(f"{p}={res[p]*100:.1f}%" for p in res), flush=True)


if __name__ == "__main__":
    main()
