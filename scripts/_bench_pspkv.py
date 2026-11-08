"""Benchmark PSP-KV (block_size=128) vs default attention across LLMs.

Run via `ssh localhost` with the PSP_KV build env (PSP_KV_BLOCK_SIZE=128
PSP_KV_TOKEN_MAJOR=1) and the fp8 precision env vars.
"""
import time
import gc
import torch
from vllm import LLM, SamplingParams

MODELS = [
    "Qwen/Qwen3-0.6B",
    "Qwen/Qwen3-1.7B",
    "Qwen/Qwen3-4B",
    "Qwen/Qwen2.5-1.5B",
]
SEQ_LEN = 512
N_TOK = 128


def bench(model, backend, block_size=None):
    kwargs = dict(
        model=model,
        enforce_eager=True,
        max_model_len=1024,
        gpu_memory_utilization=0.6,
        dtype="float16",
        enable_chunked_prefill=False,
        enable_prefix_caching=False,
    )
    if backend == "PSP_KV":
        kwargs["attention_backend"] = "PSP_KV"
        kwargs["block_size"] = block_size
    llm = LLM(**kwargs)
    tok = llm.get_tokenizer()
    sent = "The quick brown fox jumps over the lazy dog. " * (SEQ_LEN // 9 + 1)
    ids = tok(sent, add_special_tokens=False)["input_ids"][:SEQ_LEN]
    sp = SamplingParams(max_tokens=N_TOK, temperature=0, ignore_eos=True)
    llm.generate([ids], sp, use_tqdm=False)  # warmup + compile
    torch.cuda.synchronize()
    t0 = time.time()
    llm.generate([ids], sp, use_tqdm=False)
    torch.cuda.synchronize()
    dt = time.time() - t0
    tok_s = N_TOK / dt
    ms_step = dt / N_TOK * 1000.0
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    return tok_s, ms_step


def main():
    print(f"{'model':24s} {'PSP-KV128 tok/s':>15s} {'ms/step':>8s} "
          f"{'baseline tok/s':>15s} {'ms/step':>8s} {'speedup':>8s}")
    for m in MODELS:
        try:
            ps_tok, ps_ms = bench(m, "PSP_KV", block_size=128)
            bl_tok, bl_ms = bench(m, "default")
            print(f"{m:24s} {ps_tok:15.1f} {ps_ms:8.2f} "
                  f"{bl_tok:15.1f} {bl_ms:8.2f} {ps_tok / bl_tok:8.2f}x")
        except Exception as e:
            print(f"{m:24s} ERROR: {str(e)[:80]}")


if __name__ == "__main__":
    main()
