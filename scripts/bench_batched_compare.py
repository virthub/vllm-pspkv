#!/usr/bin/env python3
"""Compare batched decode: PSP-KV (lossless/e4m3/fp4) vs the FlashInfer baseline.

Reports aggregate decode tokens/s at several batch sizes, the KV capacity
(num_gpu_blocks), and the memory-limited aggregate throughput for each backend.
"""
import argparse
import os
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402


def _pspkv_llm(model, prec, gpu_util):
    os.environ["PSPKV_PRECISION"] = prec
    os.environ["PSPKV_ADAPTIVE_PRECISION"] = "0"
    os.environ.setdefault("PSPKV_FORMAT", "1")
    return LLM(model=model, enforce_eager=True, max_model_len=2048,
               gpu_memory_utilization=gpu_util, dtype="float16",
               attention_backend="PSP_KV", enable_chunked_prefill=False,
               enable_prefix_caching=False)


def _baseline_llm(model, gpu_util):
    return LLM(model=model, enforce_eager=True, max_model_len=2048,
               gpu_memory_utilization=gpu_util, dtype="float16")


def _prefill_ms(llm, tok_ids, sp1):
    llm.generate(tok_ids, sp1, use_tqdm=False)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(2):
        llm.generate(tok_ids, sp1, use_tqdm=False)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / 2 * 1e3


def _aggregate_tok_s(llm, tok_ids, out_len, prefill_ms):
    sp = SamplingParams(max_tokens=out_len, temperature=0, ignore_eos=True)
    llm.generate(tok_ids, sp, use_tqdm=False)
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
    except Exception:
        return {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--input-len", type=int, default=1024)
    ap.add_argument("--output-len", type=int, default=64)
    ap.add_argument("--batches", nargs="+", type=int, default=[1, 4, 8])
    ap.add_argument("--gpu-util", type=float, default=0.6)
    args = ap.parse_args()

    base = _baseline_llm(args.model, args.gpu_util)
    hf = base.llm_engine.vllm_config.model_config.hf_config
    n_layers = hf.num_hidden_layers
    n_kv = getattr(hf, "num_key_value_heads", hf.num_attention_heads)
    head_dim = getattr(hf, "head_size", hf.hidden_size // hf.num_attention_heads)
    tok = base.get_tokenizer()
    sentence = "The quick brown fox jumps over the lazy dog. "
    n_repeats = max(1, args.input_len // len(tok(sentence)["input_ids"]) + 1)
    tok_ids = tok(sentence * n_repeats, add_special_tokens=False)["input_ids"]
    prompt_len = len(tok_ids)
    num_used_blocks = (prompt_len + 15) // 16
    del base
    torch.cuda.empty_cache()

    sp1 = SamplingParams(max_tokens=1, temperature=0, ignore_eos=True)

    print(f"\n=== {args.model}  layers={n_layers} kv_heads={n_kv} "
          f"head_dim={head_dim}  context={prompt_len} tokens "
          f"({num_used_blocks} blocks)  gpu_util={args.gpu_util} ===")

    # Collect (name, blocks, {batch: tok_s}) for baseline + each PSP-KV fmt.
    rows = []
    for name, make in [
        ("baseline(fi-fp16)", lambda: _baseline_llm(args.model, args.gpu_util)),
        ("pspkv-lossless", lambda: _pspkv_llm(args.model, "lossless", args.gpu_util)),
        ("pspkv-e4m3", lambda: _pspkv_llm(args.model, "e4m3", args.gpu_util)),
        ("pspkv-fp4", lambda: _pspkv_llm(args.model, "fp4", args.gpu_util)),
    ]:
        llm = make()
        blocks = llm.llm_engine.vllm_config.cache_config.num_gpu_blocks
        max_batch = blocks // num_used_blocks
        st = _stats(llm, num_used_blocks)
        avg = st.get("psp_kv_avg_bytes", 2.0 if "pspkv" not in name else 0.0)
        results = {}
        for b in args.batches:
            batch_ids = [tok_ids] * b
            pf = _prefill_ms(llm, batch_ids, sp1)
            results[b] = _aggregate_tok_s(llm, batch_ids, args.output_len, pf)
        rows.append((name, blocks, max_batch, avg, results))
        del llm
        torch.cuda.empty_cache()

    # Same-batch aggregate tok/s.
    print(f"\n=== aggregate decode tokens/s (same batch) ===")
    hdr = f"{'batch':>6s}"
    for name, *_ in rows:
        hdr += f" {name:>16s}"
    print(hdr)
    for b in args.batches:
        line = f"{b:6d}"
        for _, _, _, _, results in rows:
            line += f" {results[b]:16.1f}"
        print(line)

    # Ratio vs baseline at each batch.
    print(f"\n=== ratio vs baseline(fi-fp16) at same batch ===")
    base_res = rows[0][4]
    print(f"{'batch':>6s} " + " ".join(f"{name:>16s}" for name, *_ in rows[1:]))
    for b in args.batches:
        print(f"{b:6d} " + " ".join(
            f"{results[b]/base_res[b]:16.2f}x" for _, _, _, _, results in rows[1:]))

    # Memory-limited aggregate throughput (linear extrapolation from max batch).
    print(f"\n=== memory-limited aggregate throughput (linear extrapolation) ===")
    bmax = max(args.batches)
    print(f"{'backend':>16s} {'blocks':>7s} {'max_batch':>9s} "
          f"{'agg_tok/s':>10s} {'vs_base':>8s}")
    base_agg = rows[0][2] * rows[0][4][bmax] / bmax
    for name, blocks, max_batch, avg, results in rows:
        agg = max_batch * results[bmax] / bmax
        print(f"{name:>16s} {blocks:7d} {max_batch:9d} "
              f"{agg:10.0f} {agg/base_agg:7.2f}x")


if __name__ == "__main__":
    main()
