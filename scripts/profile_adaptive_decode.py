#!/usr/bin/env python3
"""Profile adaptive per-K/V decode across models and inner-precision targets.

For each model x precision config this reports:
  * decode throughput (tok/s) and prefill latency (ms), measured on the
    PSP_KV backend and compared against the FlashAttention baseline;
  * the per-block K/V precision mix actually derived by the adaptive path;
  * which decode kernel the uniform-precision flags select
    (lossless TC / fp8 TC / per-K/V TC / scalar);
  * the KV-cache memory capacity and average bytes-per-element.

Configs cover the fixed targets (lossless / e4m3 / fp4) plus the adaptive
per-K/V path at several PSPKV_ADAPTIVE_E tolerances.

Usage:
  .venv/bin/python scripts/profile_adaptive_decode.py \
      --models facebook/opt-125m Qwen/Qwen3-0.6B Qwen/Qwen3-1.7B Qwen/Qwen3-4B \
      --input-len 1024 --output-len 128
"""
import argparse
import os
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402

FMT_NAME = {0: "lossless", 1: "e4m3", 2: "e5m2", 3: "fp4"}


def _kernel_path(stats: dict, gqa: int) -> str:
    if stats.get("precision_is_lossless"):
        return "lossless-TC"
    fp8 = stats.get("precision_fp8_fmt", 0)
    v = stats.get("precision_v_fmt", 0)
    k_lossless = stats.get("precision_k_lossless", False)
    v_lossless = stats.get("precision_v_lossless", False)
    if fp8 != 0 and v == fp8:
        return "fp8-TC"
    if not k_lossless and not v_lossless:
        return "perKV-TC" if gqa >= 4 else "perKV-scalar(gqa<4)"
    return "scalar"


def _fmt_mix(counts: dict) -> str:
    if not counts:
        return "-"
    total = sum(counts.values())
    return ",".join(
        f"{FMT_NAME[int(k)]}={v / total * 100:.0f}%"
        for k, v in sorted(counts.items(), key=lambda kv: int(kv[0]))
    )


def _make_llm(model, config, gpu_util, input_len):
    os.environ["PSPKV_PRECISION"] = config["prec"]
    os.environ["PSPKV_ADAPTIVE_PRECISION"] = "1" if config.get("adaptive") else "0"
    if config.get("E") is not None:
        os.environ["PSPKV_ADAPTIVE_E"] = str(config["E"])
    os.environ.setdefault("PSPKV_FORMAT", "1")
    kw = dict(
        model=model,
        enforce_eager=True,
        max_model_len=2048,
        gpu_memory_utilization=gpu_util,
        dtype="float16",
        attention_backend="PSP_KV",
        enable_chunked_prefill=False,
        enable_prefix_caching=False,
    )
    return LLM(**kw)


def _time_prefill(llm, prompt, sp1):
    llm.generate([prompt], sp1)  # warmup
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(3):
        llm.generate([prompt], sp1)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / 3 * 1e3  # ms


def _time_decode(llm, prompt, out_len, prefill_ms):
    sp = SamplingParams(max_tokens=out_len, temperature=0, ignore_eos=True)
    llm.generate([prompt], sp)  # warmup
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(3):
        llm.generate([prompt], sp)
    torch.cuda.synchronize()
    full_s = (time.perf_counter() - t0) / 3
    decode_s = full_s - prefill_ms / 1e3
    return out_len / max(decode_s, 1e-6)


def _stats(llm, num_used_blocks):
    try:
        r = llm.llm_engine.engine_core.collective_rpc(
            "psp_kv_stats", args=(num_used_blocks,))
        return r[0] if r else {}
    except Exception as e:
        return {"error": str(e)[:60]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+",
                    default=["facebook/opt-125m", "Qwen/Qwen3-0.6B",
                             "Qwen/Qwen3-1.7B", "Qwen/Qwen3-4B"])
    ap.add_argument("--input-len", type=int, default=1024)
    ap.add_argument("--output-len", type=int, default=128)
    ap.add_argument("--gpu-util", type=float, default=0.6)
    args = ap.parse_args()

    configs = [
        {"prec": "lossless"},
        {"prec": "e4m3"},
        {"prec": "fp4"},
        {"prec": "e4m3", "adaptive": True, "E": 1e-3},
        {"prec": "e4m3", "adaptive": True, "E": 1e-2},
        {"prec": "e4m3", "adaptive": True, "E": 1e-4},
        {"prec": "fp4", "adaptive": True, "E": 1e-3},
    ]
    for model in args.models:
        # Baseline (FlashAttention) for reference.
        base_kw = dict(model=model, enforce_eager=True, max_model_len=2048,
                       gpu_memory_utilization=args.gpu_util, dtype="float16")
        try:
            base = LLM(**base_kw)
            hf = base.llm_engine.vllm_config.model_config.hf_config
            n_heads = hf.num_attention_heads
            n_kv = getattr(hf, "num_key_value_heads", n_heads)
            gqa = n_heads // n_kv
            tok = base.get_tokenizer()
            # Build a prompt of ~input_len tokens (repeat a sentence).
            sentence = "The quick brown fox jumps over the lazy dog. "
            n_repeats = max(1, args.input_len // len(tok(sentence)["input_ids"]) + 1)
            prompt = (sentence * n_repeats)
            prompt_len = len(tok(prompt, add_special_tokens=False)["input_ids"])
            num_used_blocks = (prompt_len + 15) // 16
            sp1 = SamplingParams(max_tokens=1, temperature=0, ignore_eos=True)
            base_pf = _time_prefill(base, prompt, sp1)
            base_tok = _time_decode(base, prompt, args.output_len, base_pf)
            base_blocks = base.llm_engine.vllm_config.cache_config.num_gpu_blocks
            del base
        except Exception as e:
            print(f"{model}: baseline FAILED {str(e)[:60]}")
            continue

        print(f"\n=== {model}  heads={n_heads} kv={n_kv} gqa={gqa} "
              f"prompt_tokens={prompt_len} ===")
        print(f"{'config':26s} {'prefill_ms':>10s} {'decode_tok/s':>12s} "
              f"{'vs_base':>8s} {'kernel':>20s} {'K mix':>22s} "
              f"{'V mix':>22s} {'avg_B/e':>8s} {'tokens':>8s}")
        for cfg in configs:
            tag = cfg["prec"]
            if cfg.get("adaptive"):
                tag += f"-adp(E={cfg['E']:g})"
            try:
                llm = _make_llm(model, cfg, args.gpu_util, args.input_len)
                sp1 = SamplingParams(max_tokens=1, temperature=0, ignore_eos=True)
                pf = _time_prefill(llm, prompt, sp1)
                st = _stats(llm, num_used_blocks)  # right after a pure prefill
                tok = _time_decode(llm, prompt, args.output_len, pf)
                blocks = llm.llm_engine.vllm_config.cache_config.num_gpu_blocks
                tokens = blocks * 16
                kpath = _kernel_path(st, gqa)
                kmix = _fmt_mix(st.get("k_fmt_counts", {}))
                vmix = _fmt_mix(st.get("v_fmt_counts", {}))
                avg = st.get("psp_kv_avg_bytes", 0.0)
                print(f"{tag:26s} {pf:10.2f} {tok:12.1f} "
                      f"{tok / base_tok:7.2f}x {kpath:>20s} {kmix:>22s} "
                      f"{vmix:>22s} {avg:8.3f} {tokens:8d}")
                del llm
            except Exception as e:
                print(f"{tag:26s} FAILED {str(e)[:60]}")


if __name__ == "__main__":
    main()
