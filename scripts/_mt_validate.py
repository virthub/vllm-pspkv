#!/usr/bin/env python3
"""Validate the multi-token fp4 attention: dequant amortization vs N."""
import os
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch  # noqa: E402
from vllm.v1.attention.backends.psp_kv_attn import _get_psp_kv_ops  # noqa: E402

ops = _get_psp_kv_ops()

B, H, KVH, D, C = 64, 32, 8, 80, 128
scale = 1.0 / (D ** 0.5)
num_blocks = (C + 15) // 16
eb = 0.5
block_bytes = int(16 * D * eb * KVH)
total = block_bytes * num_blocks

kv = torch.zeros(2 * total, dtype=torch.uint8, device="cuda")
kc, vc = kv[:total], kv[total:]
pt = torch.full((num_blocks,), 15, dtype=torch.uint8, device="cuda")
bo = torch.arange(num_blocks, dtype=torch.int32, device="cuda") * block_bytes
bo = torch.cat([bo, bo])
st = torch.ones(2 * num_blocks, KVH, dtype=torch.float32, device="cuda")
bt = torch.zeros(B, num_blocks, dtype=torch.int32, device="cuda")
cl = torch.full((B,), C - 8, dtype=torch.int32, device="cuda")


def time_mt(N, iters=50):
    q = torch.randn(B * N, H, D, dtype=torch.float16, device="cuda")
    key = torch.randn(B * N, KVH, D, dtype=torch.float16, device="cuda")
    value = torch.randn(B * N, KVH, D, dtype=torch.float16, device="cuda")
    out = torch.empty_like(q)
    for _ in range(10):
        ops.psp_attention_enhanced_mt(
            q, key, value, kc, vc, bt, cl, pt, bo, st, scale, KVH, D, 3, N, out)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        ops.psp_attention_enhanced_mt(
            q, key, value, kc, vc, bt, cl, pt, bo, st, scale, KVH, D, 3, N, out)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


for N in [1, 2, 4]:
    t = time_mt(N)
    print(f"N={N}: {t*1e6:7.1f} us/call  {t*1e6/N:7.1f} us/token")
