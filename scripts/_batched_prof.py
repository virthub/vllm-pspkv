"""Isolate batched-decode overhead: attention vs write-back at batch 1 vs 8.

Times `psp_attention_enhanced` and `quantize_single_token` directly (lossless
format) so we can attribute the batch>1 slowdown to the attention kernel or the
per-token K/V write-back.
"""
import os
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch  # noqa: E402
from vllm.v1.attention.backends.psp_kv_attn import _get_psp_kv_ops  # noqa: E402

ops = _get_psp_kv_ops()

num_blocks = 59
num_kv_heads = 8
num_heads = 32
head_dim = 80
block_size = 16
softmax_scale = 1.0 / (head_dim ** 0.5)
num_layers = 36


def build_cache(batch):
    # lossless: 2 bytes/elem, uniform strides.
    precision_table = torch.zeros(num_blocks, dtype=torch.uint8, device="cuda")
    eb = 2
    block_bytes = int(block_size * head_dim * eb * num_kv_heads)
    total_bytes = block_bytes * num_blocks
    kv_cache = torch.zeros(2 * total_bytes, dtype=torch.uint8, device="cuda")
    k_cache = kv_cache[:total_bytes]
    v_cache = kv_cache[total_bytes:]
    bo = torch.arange(num_blocks, dtype=torch.int32, device="cuda") * block_bytes
    block_offsets = torch.cat([bo, bo])
    scale_table = torch.ones(2 * num_blocks, num_kv_heads, dtype=torch.float32, device="cuda")
    return k_cache, v_cache, precision_table, block_offsets, scale_table


def time_attn(batch, iters=100):
    kc, vc, pt, bo, st = build_cache(batch)
    query = torch.randn(batch, num_heads, head_dim, dtype=torch.float16, device="cuda")
    block_tables = torch.zeros(batch, num_blocks, dtype=torch.int32, device="cuda")
    context_lens = torch.full((batch,), num_blocks * block_size - 5, dtype=torch.int32, device="cuda")
    out = torch.empty_like(query)
    args = (query, kc, vc, block_tables, context_lens, pt, bo, st,
            softmax_scale, num_kv_heads, head_dim, -1,
            True, 0, 0, 0, True, True, out)
    for _ in range(10):
        ops.psp_attention_enhanced(*args)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        ops.psp_attention_enhanced(*args)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def time_writeback(batch, iters=200):
    kc, vc, pt, bo, st = build_cache(batch)
    key = torch.randn(batch, num_kv_heads, head_dim, dtype=torch.float16, device="cuda")
    value = torch.randn(batch, num_kv_heads, head_dim, dtype=torch.float16, device="cuda")
    slot_mapping = torch.arange(batch, dtype=torch.int64, device="cuda")
    for _ in range(10):
        ops.quantize_single_token(key, value, kc, vc, slot_mapping, pt, bo, st, 0)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        ops.quantize_single_token(key, value, kc, vc, slot_mapping, pt, bo, st, 0)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


for b in (1, 8):
    a = time_attn(b)
    w = time_writeback(b)
    # per decode step: 36 layers x (attention + writeback)
    step_attn = a * num_layers
    step_wb = w * num_layers
    step_total = step_attn + step_wb
    print(f"batch {b}: attn={a*1e6:7.1f}us  writeback={w*1e6:7.1f}us  "
          f"per-step attn={step_attn*1e3:7.2f}ms wb={step_wb*1e3:7.2f}ms "
          f"total={step_total*1e3:7.2f}ms  wb_frac={step_wb/step_total*100:.0f}%")
