#!/usr/bin/env python3
"""Benchmark memory-savings -> batch-size -> aggregate decode tokens/s.

For each precision config this reports:
  * KV cache capacity (num_gpu_blocks) at a fixed gpu_memory_utilization;
  * average bytes/element (from the flat cache introspection);
  * aggregate decode tokens/s at several batch sizes, showing the throughput
    scales with the memory capacity, not the per-token latency.
"""
import argparse
import os
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402


def _make_llm(model, prec, adaptive, E, gpu_util):
    os.environ["PSPKV_PRECISION"] = prec
    os.environ["PSPKV_ADAPTIVE_PRECISION"] = "1" if adaptive else "0"
    if E is not None:
        os.environ["PSPKV_ADAPTIVE_E"] = str(E)
    os.environ.setdefault("PSPKV_FORMAT", "1")
    return LLM(model=model, enforce_eager=True, max_model_len=2048,
               gpu_memory_utilization=gpu_util, dtype="float16",
               attention_backend="PSP_KV", enable_chunked_prefill=False,
               enable_prefix_caching=False)


def _prefill_ms(llm, tok_ids, sp1):
    llm.generate(tok_ids, sp1, use_tqdm=False)  # warmup
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(2):
        llm.generate(tok_ids, sp1, use_tqdm=False)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / 2 * 1e3


def _aggregate_tok_s(llm, tok_ids, out_len, prefill_ms):
    sp = SamplingParams(max_tokens=out_len, temperature=0, ignore_eos=True)
    llm.generate(tok_ids, sp, use_tqdm=False)  # warmup
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    llm.generate(tok_ids, sp, use_tqdm=False)
    torch.cuda.synchronize()
    full_s = time.perf_counter() - t0
    decode_s = full_s - prefill_ms / 1e3
    return len(tok_ids) * out_len / max(decode_s, 1e-6)


def _stats(llm, num_used_blocks):
    try:
        r = llm.llm_engine.engine_core.collective_rpc(
            "psp_kv_stats", args=(num_used_blocks,))
        return r[0] if r else {}
    except Exception as e:
        return {"error": str(e)[:60]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--input-len", type=int, default=1024)
    ap.add_argument("--output-len", type=int, default=64)
    ap.add_argument("--batches", nargs="+", type=int, default=[1, 8, 32, 64])
    ap.add_argument("--gpu-util", type=float, default=0.6)
    args = ap.parse_args()

    configs = [
        ("lossless", False, None),
        ("e4m3", False, None),
        ("fp4", False, None),
        ("e4m3", True, 1e-3),
    ]

    # Load once to get the model config + tokenizer (baseline-style).
    base = LLM(model=args.model, enforce_eager=True, max_model_len=2048,
               gpu_memory_utilization=args.gpu_util, dtype="float16")
    hf = base.llm_engine.vllm_config.model_config.hf_config
    n_layers = hf.num_hidden_layers
    n_kv = getattr(hf, "num_key_value_heads", hf.num_attention_heads)
    head_dim = getattr(hf, "head_size", hf.hidden_size // hf.num_attention_heads)
    tok = base.get_tokenizer()
    sentence = "The quick brown fox jumps over the lazy dog. "
    n_repeats = max(1, args.input_len // len(tok(sentence)["input_ids"]) + 1)
    prompt = sentence * n_repeats
    tok_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    prompt_len = len(tok_ids)
    num_used_blocks = (prompt_len + 15) // 16
    del base

    # Per-token KV bytes = 2 (K,V) * layers * kv_heads * head_dim * bytes/elem.
    print(f"\n=== {args.model}  layers={n_layers} kv_heads={n_kv} "
          f"head_dim={head_dim}  context={prompt_len} tokens "
          f"({num_used_blocks} blocks) ===")
    print(f"{'config':18s} {'avg_B/e':>7s} {'KV/token':>9s} "
          f"{'blocks':>7s} {'max_batch':>9s}")
    print("-" * 60)

    capacity = {}
    for tag, adaptive, E in configs:
        llm = _make_llm(args.model, tag.split("-")[0], adaptive, E, args.gpu_util)
        sp1 = SamplingParams(max_tokens=1, temperature=0, ignore_eos=True)
        _prefill_ms(llm, [tok_ids], sp1)
        st = _stats(llm, num_used_blocks)
        blocks = llm.llm_engine.vllm_config.cache_config.num_gpu_blocks
        avg = st.get("psp_kv_avg_bytes", 0.0)
        kv_per_token = 2 * n_layers * n_kv * head_dim * avg
        max_batch = blocks // num_used_blocks
        capacity[tag] = (blocks, max_batch, avg)
        print(f"{tag:18s} {avg:7.3f} {kv_per_token/1024:7.1f}KB "
              f"{blocks:7d} {max_batch:9d}")
        del llm
        torch.cuda.empty_cache()

    # Throughput vs batch for the two extreme formats (lossless vs fp4).
    print(f"\n=== aggregate decode tokens/s vs batch ===")
    print(f"{'batch':>6s} {'lossless':>12s} {'fp4':>12s} {'fp4/lossless':>13s}")
    print("-" * 50)
    for tag, adaptive, E in [("lossless", False, None), ("fp4", False, None)]:
        llm = _make_llm(args.model, tag, adaptive, E, args.gpu_util)
        sp1 = SamplingParams(max_tokens=1, temperature=0, ignore_eos=True)
        results = {}
        for b in args.batches:
            batch_ids = [tok_ids] * b
            pf = _prefill_ms(llm, batch_ids, sp1)
            results[b] = _aggregate_tok_s(llm, batch_ids, args.output_len, pf)
        capacity[tag] = (capacity[tag][0], capacity[tag][1], capacity[tag][2], results)
        del llm
        torch.cuda.empty_cache()

    for b in args.batches:
        lt = capacity["lossless"][3][b]
        ft = capacity["fp4"][3][b]
        print(f"{b:6d} {lt:12.1f} {ft:12.1f} {ft/lt:13.2f}x")

    # Combined: aggregate throughput at the memory-limited max batch.
    _, lb_max, _, lr = capacity["lossless"]
    _, fb_max, _, fr = capacity["fp4"]
    lt = lr[max(args.batches)]
    ft = fr[max(args.batches)]
    print(f"\n=== memory-limited aggregate throughput (gpu_util={args.gpu_util}) ===")
    print(f"lossless: max_batch={lb_max}  ~= {lb_max * (lt/max(args.batches)):.0f} tok/s "
          f"(scaled from batch {max(args.batches)})")
    print(f"fp4:      max_batch={fb_max}  ~= {fb_max * (ft/max(args.batches)):.0f} tok/s")
    print(f"fp4 uplift ~= {fb_max * ft / (lb_max * lt):.2f}x")


if __name__ == "__main__":
    main()
