"""Long-context decode: PSP-KV (fp8 cache) vs baseline (fp16 cache).

Quantifies the fp8 KV cache's memory/bandwidth advantage as the context grows.
Run via `ssh localhost` with the PSP_KV build env + fp8 precision env vars.
"""
import time
import gc
import torch
from vllm import LLM, SamplingParams

MODEL = "Qwen/Qwen3-4B"
CONTEXTS = [1024, 2048, 4096, 8192]
N_TOK = 64  # decode tokens per measurement


def make_prompt(tok, n):
    sent = "The quick brown fox jumps over the lazy dog. "
    ids = tok(sent, add_special_tokens=False)["input_ids"]
    reps = (n + len(ids) - 1) // len(ids) + 1
    return (ids * reps)[:n]


def bench(model, backend, ctx, block_size=None):
    kwargs = dict(
        model=model,
        enforce_eager=True,
        max_model_len=ctx + N_TOK,
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
    ids = make_prompt(tok, ctx)
    sp = SamplingParams(max_tokens=N_TOK, temperature=0, ignore_eos=True)
    llm.generate([ids], sp, use_tqdm=False)  # warmup (prefill + decode)
    torch.cuda.synchronize()
    t0 = time.time()
    llm.generate([ids], sp, use_tqdm=False)
    torch.cuda.synchronize()
    dt = time.time() - t0
    tok_s = N_TOK / dt
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    return tok_s


def main():
    print(f"model={MODEL}  decode {N_TOK} tok at context")
    print(f"{'context':>8s} {'PSP-KV128 tok/s':>16s} {'baseline tok/s':>16s} {'speedup':>8s}")
    for ctx in CONTEXTS:
        try:
            ps = bench(MODEL, "PSP_KV", ctx, block_size=128)
            bl = bench(MODEL, "default", ctx)
            print(f"{ctx:8d} {ps:16.1f} {bl:16.1f} {ps / bl:8.2f}x", flush=True)
        except Exception as e:
            print(f"{ctx:8d} ERROR: {str(e)[:90]}", flush=True)


if __name__ == "__main__":
    main()
