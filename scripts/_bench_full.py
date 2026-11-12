#!/usr/bin/env python3
"""Comprehensive PSP-KV evaluation across models / target precision / workloads.

For each (model, target dtype): build the FlashAttention baseline and the
PSP-KV inner precisions (lossless/e4m3/e5m2/fp4) and report:
  - KV-cache capacity (num_gpu_blocks)
  - decode tokens/s at several batch sizes and context lengths
  - prefill latency
  - greedy-decode token match rate vs the fp16 baseline
"""
import argparse
import os
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402

BLOCK = 16


def make_llm(model, prec, dtype, gpu_util, max_len=4096):
    if prec == "baseline":
        return LLM(model=model, enforce_eager=True, max_model_len=max_len,
                   gpu_memory_utilization=gpu_util, dtype=dtype, block_size=128)
    os.environ["PSPKV_PRECISION"] = prec
    os.environ["PSPKV_ADAPTIVE_PRECISION"] = "0"
    os.environ.setdefault("PSPKV_FORMAT", "1")
    return LLM(model=model, enforce_eager=True, max_model_len=max_len,
               gpu_memory_utilization=gpu_util, dtype=dtype,
               attention_backend="PSP_KV", enable_chunked_prefill=False,
               enable_prefix_caching=False, block_size=128)


def prefill_ms(llm, batch_ids):
    sp = SamplingParams(max_tokens=1, temperature=0, ignore_eos=True)
    llm.generate(batch_ids, sp, use_tqdm=False)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(3):
        llm.generate(batch_ids, sp, use_tqdm=False)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / 3 * 1e3


def decode_tok_s(llm, batch_ids, out_len):
    sp = SamplingParams(max_tokens=out_len, temperature=0, ignore_eos=True)
    sp1 = SamplingParams(max_tokens=1, temperature=0, ignore_eos=True)
    # prefill at the SAME batch (1 token), measured separately
    llm.generate(batch_ids, sp1, use_tqdm=False)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    llm.generate(batch_ids, sp1, use_tqdm=False)
    torch.cuda.synchronize()
    pf_s = time.perf_counter() - t0
    # decode: prefill + out_len tokens, minus the prefill
    llm.generate(batch_ids, sp1, use_tqdm=False)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    llm.generate(batch_ids, sp, use_tqdm=False)
    torch.cuda.synchronize()
    decode_s = (time.perf_counter() - t0) - pf_s
    return len(batch_ids) * out_len / max(decode_s, 1e-6)


def greedy_match(llm, prompts, ref, out_len=32):
    sp = SamplingParams(max_tokens=out_len, temperature=0, ignore_eos=True)
    out = [list(o.outputs[0].token_ids) for o in llm.generate(prompts, sp, use_tqdm=False)]
    m = sum(a == b for ra, oa in zip(ref, out) for a, b in zip(ra, oa))
    return m / max(sum(len(r) for r in ref), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=["Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B"])
    ap.add_argument("--input-lens", nargs="+", type=int, default=[512, 2048])
    ap.add_argument("--batches", nargs="+", type=int, default=[1, 8, 32])
    ap.add_argument("--output-len", type=int, default=64)
    ap.add_argument("--gpu-util", type=float, default=0.6)
    ap.add_argument("--bf16-model", default="Qwen/Qwen3-4B",
                    help="one model to also run at bf16 target precision")
    args = ap.parse_args()

    backends = [("baseline", "baseline"), ("lossless", "lossless"),
                ("e4m3", "e4m3"), ("e5m2", "e5m2"), ("fp4", "fp4"),
                ("int8", "int8")]

    # tokenizer + prompts (shared per model)
    toks = {}
    for m in args.models:
        probe = make_llm(m, "baseline", "float16", args.gpu_util)
        toks[m] = probe.get_tokenizer()
        del probe
        torch.cuda.empty_cache()

    for model in args.models:
        for dtype in (["float16", "bfloat16"] if model == args.bf16_model else ["float16"]):
            tok = toks[model]
            sentence = "The quick brown fox jumps over the lazy dog. "
            prompts = {il: tok(sentence * (il // len(tok(sentence)["input_ids"]) + 1),
                               add_special_tokens=False)["input_ids"] for il in args.input_lens}
            acc_prompts = [tok("The quick brown fox jumps over the lazy dog. ",
                               add_special_tokens=False)["input_ids"],
                           tok("Machine learning models compress knowledge into hidden states. " * 2,
                               add_special_tokens=False)["input_ids"]]

            ref = None
            rows = []
            for name, prec in backends:
                llm = make_llm(model, prec, dtype, args.gpu_util)
                blocks = llm.llm_engine.vllm_config.cache_config.num_gpu_blocks
                thr = {}  # (input_len, batch) -> tok/s
                pf = {}   # input_len -> ms
                for il in args.input_lens:
                    ids1 = [prompts[il]]
                    pf[il] = prefill_ms(llm, ids1)
                    for b in args.batches:
                        ids = [prompts[il]] * b
                        thr[(il, b)] = decode_tok_s(llm, ids, args.output_len)
                # accuracy (only fp16 target)
                acc = None
                if dtype == "float16":
                    if ref is None and prec == "baseline":
                        sp = SamplingParams(max_tokens=32, temperature=0, ignore_eos=True)
                        ref = [list(o.outputs[0].token_ids) for o in
                               llm.generate(acc_prompts, sp, use_tqdm=False)]
                    if ref is not None:
                        acc = greedy_match(llm, acc_prompts, ref)
                rows.append((name, blocks, thr, pf, acc))
                del llm
                torch.cuda.empty_cache()

            print(f"\n{'='*100}")
            print(f"MODEL={model}  dtype={dtype}")
            print(f"{'='*100}")
            base_blocks = rows[0][1]
            print("KV capacity (blocks): " + "  ".join(
                f"{name}={blocks}({blocks/base_blocks:.2f}x)" for name, blocks, *_ in rows))
            for il in args.input_lens:
                print(f"\n  context={il}  prefill(ms): " + "  ".join(
                    f"{name}={pf[il]:.1f}" for name, _, _, pf, _ in rows))
                hdr = f"  {'batch':>5s} " + "".join(f"{name:>14s}" for name, *_ in rows)
                print(hdr)
                for b in args.batches:
                    line = f"  {b:5d} "
                    for _, _, thr, _, _ in rows:
                        line += f"{thr[(il, b)]:14.1f}"
                    print(line)
            if dtype == "float16":
                print("  greedy match vs baseline: " + "  ".join(
                    f"{name}={r_acc*100:.1f}%" for name, _, _, _, r_acc in rows
                    if r_acc is not None))


if __name__ == "__main__":
    main()
