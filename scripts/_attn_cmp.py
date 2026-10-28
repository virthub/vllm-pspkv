"""Time FlashAttention decode vs PSP-KV lossless attention at batch 8."""
import os
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch  # noqa: E402
from vllm.vllm_flash_attn.flash_attn_interface import flash_attn_varlen_func  # noqa: E402

B, H, KVH, D, C = 8, 32, 8, 80, 941
scale = 1.0 / (D ** 0.5)

q = torch.randn(B, H, D, dtype=torch.float16, device="cuda")
k = torch.randn(B * C, KVH, D, dtype=torch.float16, device="cuda")
v = torch.randn(B * C, KVH, D, dtype=torch.float16, device="cuda")
cu_q = torch.arange(0, B + 1, dtype=torch.int32, device="cuda")
cu_k = torch.arange(0, B * C + 1, C, dtype=torch.int32, device="cuda")


def time_fa(iters=100):
    for _ in range(10):
        flash_attn_varlen_func(q, k, v, 1, cu_q, C, cu_k,
                               softmax_scale=scale, causal=False, num_splits=0)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        flash_attn_varlen_func(q, k, v, 1, cu_q, C, cu_k,
                               softmax_scale=scale, causal=False, num_splits=0)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def time_pspkv(iters=100):
    from vllm.v1.attention.backends.psp_kv_attn import _get_psp_kv_ops
    ops = _get_psp_kv_ops()
    num_blocks = (C + 15) // 16
    eb = 2
    block_bytes = int(16 * D * eb * KVH)
    total_bytes = block_bytes * num_blocks
    kv_cache = torch.zeros(2 * total_bytes, dtype=torch.uint8, device="cuda")
    kc, vc = kv_cache[:total_bytes], kv_cache[total_bytes:]
    pt = torch.zeros(num_blocks, dtype=torch.uint8, device="cuda")
    bo = torch.arange(num_blocks, dtype=torch.int32, device="cuda") * block_bytes
    bo = torch.cat([bo, bo])
    st = torch.ones(2 * num_blocks, KVH, dtype=torch.float32, device="cuda")
    query = torch.randn(B, H, D, dtype=torch.float16, device="cuda")
    block_tables = torch.zeros(B, num_blocks, dtype=torch.int32, device="cuda")
    context_lens = torch.full((B,), C - 5, dtype=torch.int32, device="cuda")
    out = torch.empty_like(query)
    args = (query, kc, vc, block_tables, context_lens, pt, bo, st,
            scale, KVH, D, -1, True, 0, 0, 0, True, True, out)
    for _ in range(10):
        ops.psp_attention_enhanced(*args)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        ops.psp_attention_enhanced(*args)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


fa = time_fa()
pk = time_pspkv()
print(f"FlashAttention decode (batch 8): {fa*1e6:7.1f} us")
print(f"PSP-KV lossless   (batch 8): {pk*1e6:7.1f} us")
print(f"PSP-KV / FlashAttention: {pk/fa:.2f}x")
