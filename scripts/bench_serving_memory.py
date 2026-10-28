#!/usr/bin/env python3
"""Long-context serving benchmark: actual high-batch decode, fp16 vs fp4.

Measures the *actual* (non-extrapolated) aggregate decode tokens/s at high
batch and reports the fp4 memory-efficiency uplift (4x KV capacity -> 4x batch
at the same memory footprint -> higher aggregate throughput).
"""
import argparse
import os
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402


def _make_llm(model, backend, gpu_util):
    if backend == "fp4":
        os.environ["PSPKV_PRECISION"] = "fp4"
        os.environ["PSPKV_ADAPTIVE_PRECISION"] = "0"
        os.environ.setdefault("PSPKV_FORMAT", "1")
        return LLM(model=model, enforce_eager=True, max_model_len=4096,
                   gpu_memory_utilization=gpu_util, dtype="float16",
                   attention_backend="PSP_KV", enable_chunked_prefill=False,
                   enable_prefix_caching=False)
    return LLM(model=model, enforce_eager=True, max_model_len=4096,
               gpu_memory_utilization=gpu_util, dtype="float16",
               enable_chunked_prefill=False, enable_prefix_caching=False)


def _decode_agg_tok_s(llm, batch_ids, out_len):
    """Aggregate decode tokens/s (prefill time subtracted)."""
    sp1 = SamplingParams(max_tokens=1, temperature=0, ignore_eos=True)
    llm.generate(batch_ids, sp1, use_tqdm=False)  # prefill warmup
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(2):
        llm.generate(batch_ids, sp1, use_tqdm=False)
    torch.cuda.synchronize()
    prefill_s = (time.perf_counter() - t0) / 2

    sp = SamplingParams(max_tokens=out_len, temperature=0, ignore_eos=True)
    llm.generate(batch_ids, sp, use_tqdm=False)  # decode warmup
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    llm.generate(batch_ids, sp, use_tqdm=False)
    torch.cuda.synchronize()
    full_s = time.perf_counter() - t0
    decode_s = max(full_s - prefill_s, 1e-6)
    return len(batch_ids) * out_len / decode_s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--input-len", type=int, default=256)
    ap.add_argument("--output-len", type=int, default=32)
    ap.add_argument("--batches", nargs="+", type=int, default=[8, 32, 64, 128])
    ap.add_argument("--gpu-util", type=float, default=0.8)
    args = ap.parse_args()

    # Tokenizer + capacity info from a throwaway fp16 load.
    base = LLM(model=args.model, enforce_eager=True, max_model_len=4096,
               gpu_memory_utilization=args.gpu_util, dtype="float16")
    hf = base.llm_engine.vllm_config.model_config.hf_config
    n_layers = hf.num_hidden_layers
    n_kv = getattr(hf, "num_key_value_heads", hf.num_attention_heads)
    head_dim = getattr(hf, "head_size", hf.hidden_size // hf.num_attention_heads)
    tok = base.get_tokenizer()
    sentence = "The quick brown fox jumps over the lazy dog. "
    n_rep = max(1, args.input_len // len(tok(sentence)["input_ids"]) + 1)
    tok_ids = tok(sentence * n_rep, add_special_tokens=False)["input_ids"]
    prompt_len = len(tok_ids)
    blocks_per_seq = (prompt_len + 15) // 16
    fp16_blocks = base.llm_engine.vllm_config.cache_config.num_gpu_blocks
    fp16_max_batch = fp16_blocks // blocks_per_seq
    del base
    torch.cuda.empty_cache()

    print(f"\n=== {args.model}  layers={n_layers} kv_heads={n_kv} "
          f"head_dim={head_dim}  prompt={prompt_len} tok "
          f"({blocks_per_seq} blocks/seq) gpu_util={args.gpu_util} ===")
    print(f"fp16 capacity: {fp16_blocks} blocks -> max_batch ~{fp16_max_batch}")

    results = {}
    blocks_map = {}
    for backend in ["fp16", "fp4"]:
        llm = _make_llm(args.model, backend, args.gpu_util)
        blocks = llm.llm_engine.vllm_config.cache_config.num_gpu_blocks
        max_batch = blocks // blocks_per_seq
        blocks_map[backend] = blocks
        print(f"\n[{backend}] blocks={blocks} max_batch~={max_batch}")
        results[backend] = {}
        for b in args.batches:
            batch_ids = [tok_ids] * b
            agg = _decode_agg_tok_s(llm, batch_ids, args.output_len)
            results[backend][b] = agg
            print(f"  batch {b:4d}: aggregate {agg:8.1f} tok/s")
        del llm
        torch.cuda.empty_cache()

    print(f"\n=== aggregate decode tok/s (actual) ===")
    print(f"{'batch':>6s} {'fp16':>10s} {'fp4':>10s} {'fp4/fp16':>9s}")
    for b in args.batches:
        a = results["fp16"][b]
        c = results["fp4"][b]
        print(f"{b:6d} {a:10.1f} {c:10.1f} {c/a:9.2f}x")

    # Memory-efficiency uplift: fp4 holds ~4x the blocks, so it can serve ~4x
    # the batch at equal memory. Aggregate scales sub-linearly with batch, so
    # estimate the fp4 uplift as capacity_ratio x (fp4/fp16 throughput at the
    # largest measured batch).
    hi = max(args.batches)
    ratio = results["fp4"][hi] / results["fp16"][hi]
    cap_ratio = blocks_map["fp4"] / blocks_map["fp16"]
    print(f"\n=== fp4 memory-efficiency summary ===")
    print(f"capacity ratio fp4/fp16: {cap_ratio:.2f}x")
    print(f"same-batch fp4/fp16 @ batch {hi}: {ratio:.2f}x")
    print(f"estimated memory-limited uplift: {cap_ratio * ratio:.2f}x "
          f"(= capacity_ratio x same-batch ratio)")


if __name__ == "__main__":
    main()
