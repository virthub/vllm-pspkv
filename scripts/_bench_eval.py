#!/usr/bin/env python3
"""Complete PSP-KV vs baseline performance evaluation.

For the FlashAttention fp16 baseline and each PSP-KV inner precision:
  - KV-cache capacity (num_gpu_blocks)
  - prefill latency and aggregate decode tokens/s at several batch sizes
  - greedy-decode token match rate vs the fp16 baseline
"""
import argparse
import os
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402

MODEL = "Qwen/Qwen3-4B"
BLOCK = 16
ACC_LEN = 40


def make_llm(prec, gpu_util):
    if prec == "baseline":
        return LLM(model=MODEL, enforce_eager=True, max_model_len=2048,
                   gpu_memory_utilization=gpu_util, dtype="float16")
    os.environ["PSPKV_PRECISION"] = prec
    os.environ["PSPKV_ADAPTIVE_PRECISION"] = "0"
    os.environ.setdefault("PSPKV_FORMAT", "1")
    return LLM(model=MODEL, enforce_eager=True, max_model_len=2048,
               gpu_memory_utilization=gpu_util, dtype="float16",
               attention_backend="PSP_KV", enable_chunked_prefill=False,
               enable_prefix_caching=False)


def prefill_ms(llm, batch_ids):
    sp = SamplingParams(max_tokens=1, temperature=0, ignore_eos=True)
    llm.generate(batch_ids, sp, use_tqdm=False)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(3):
        llm.generate(batch_ids, sp, use_tqdm=False)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / 3 * 1e3


def decode_tok_s(llm, batch_ids, out_len, pf_ms):
    sp = SamplingParams(max_tokens=out_len, temperature=0, ignore_eos=True)
    llm.generate(batch_ids, sp, use_tqdm=False)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    llm.generate(batch_ids, sp, use_tqdm=False)
    torch.cuda.synchronize()
    decode_s = (time.perf_counter() - t0) - pf_ms / 1e3
    return len(batch_ids) * out_len / max(decode_s, 1e-6)


def greedy_tokens(llm, prompts):
    sp = SamplingParams(max_tokens=ACC_LEN, temperature=0, ignore_eos=True)
    out = llm.generate(prompts, sp, use_tqdm=False)
    return [list(o.outputs[0].token_ids) for o in out]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input-len", type=int, default=1024)
    ap.add_argument("--output-len", type=int, default=128)
    ap.add_argument("--batches", nargs="+", type=int, default=[1, 4, 8, 16])
    ap.add_argument("--gpu-util", type=float, default=0.6)
    args = ap.parse_args()

    # Geometry + prompts (shared across backends).
    probe = make_llm("baseline", args.gpu_util)
    hf = probe.llm_engine.vllm_config.model_config.hf_config
    n_layers = hf.num_hidden_layers
    n_kv = getattr(hf, "num_key_value_heads", hf.num_attention_heads)
    head_dim = getattr(hf, "head_size", hf.hidden_size // hf.num_attention_heads)
    tok = probe.get_tokenizer()
    sentence = "The quick brown fox jumps over the lazy dog. "
    n_rep = max(1, args.input_len // len(tok(sentence)["input_ids"]) + 1)
    prompt_ids = tok(sentence * n_rep, add_special_tokens=False)["input_ids"]
    prompt_len = len(prompt_ids)
    del probe
    torch.cuda.empty_cache()

    acc_texts = [
        "The quick brown fox jumps over the lazy dog. ",
        "Machine learning models compress knowledge into hidden states. " * 3,
        "In a hole in the ground there lived a hobbit. " * 3,
    ]
    acc_prompts = [tok(t, add_special_tokens=False)["input_ids"] for t in acc_texts]

    backends = [
        ("baseline(fp16)", "baseline"),
        ("pspkv-lossless", "lossless"),
        ("pspkv-e4m3", "e4m3"),
        ("pspkv-e5m2", "e5m2"),
        ("pspkv-fp4", "fp4"),
    ]

    # Reference greedy outputs (fp16 baseline).
    ref_llm = make_llm("baseline", args.gpu_util)
    ref_out = greedy_tokens(ref_llm, acc_prompts)
    del ref_llm
    torch.cuda.empty_cache()

    rows = []
    for name, prec in backends:
        llm = make_llm(prec, args.gpu_util)
        blocks = llm.llm_engine.vllm_config.cache_config.num_gpu_blocks
        max_batch = blocks // ((prompt_len + BLOCK - 1) // BLOCK)

        results = {}
        for b in args.batches:
            ids = [prompt_ids] * b
            pf = prefill_ms(llm, ids)
            results[b] = decode_tok_s(llm, ids, args.output_len, pf)

        acc = greedy_tokens(llm, acc_prompts)
        match = sum(a == b for ra, oa in zip(ref_out, acc)
                    for a, b in zip(ra, oa))
        total = sum(len(ra) for ra in ref_out)
        match_rate = match / max(total, 1)

        rows.append((name, blocks, max_batch, results, match_rate))
        del llm
        torch.cuda.empty_cache()

    # ===== report =====
    print(f"\n=== {MODEL}  layers={n_layers} kv_heads={n_kv} head_dim={head_dim} "
          f"ctx={prompt_len}  gpu_util={args.gpu_util} ===")

    base_blocks = rows[0][1]
    print("\n=== KV-cache capacity (num_gpu_blocks) ===")
    for name, blocks, _, _, _ in rows:
        print(f"  {name:18s} {blocks:6d} blocks  {blocks/base_blocks:6.2f}x vs baseline")

    print(f"\n=== aggregate decode tokens/s (out_len={args.output_len}) ===")
    hdr = f"{'batch':>6s}" + "".join(f"{name:>17s}" for name, *_ in rows)
    print(hdr)
    for b in args.batches:
        line = f"{b:6d}"
        for _, _, _, results, _ in rows:
            line += f"{results[b]:17.1f}"
        print(line)

    print("\n=== decode tokens/s ratio vs baseline ===")
    base_res = rows[0][3]
    print(f"{'batch':>6s}" + "".join(f"{name:>17s}" for name, *_ in rows[1:]))
    for b in args.batches:
        line = f"{b:6d}"
        for _, _, _, results, _ in rows[1:]:
            line += f"{results[b]/base_res[b]:16.2f}x "
        print(line)

    bmax = args.batches[-1]
    print("\n=== memory-limited aggregate throughput (linear extrapolation) ===")
    base_agg = rows[0][2] * rows[0][3][bmax] / bmax
    for name, blocks, max_batch, results, _ in rows:
        agg = max_batch * results[bmax] / bmax
        print(f"  {name:18s} {blocks:6d} blocks  max_batch={max_batch:5d}  "
              f"agg={agg:9.0f} tok/s  {agg/base_agg:6.2f}x vs baseline")

    print(f"\n=== greedy-decode token match rate vs fp16 baseline "
          f"({ACC_LEN} tokens x {len(acc_prompts)} prompts) ===")
    for name, _, _, _, mr in rows:
        print(f"  {name:18s} {mr*100:6.2f}%")


if __name__ == "__main__":
    main()
