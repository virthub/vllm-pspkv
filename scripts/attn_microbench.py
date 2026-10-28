"""Micro-benchmark: time PSP-KV attention kernels directly (no FFN).

Builds a synthetic 4B-shaped cache (8 kv heads, 128 dim, 59 blocks) and times
the decode attention op for each precision, isolating the attention kernel from
the rest of the model.
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
head_dim = 128
block_size = 16
softmax_scale = 1.0 / (head_dim ** 0.5)


def build_cache(fmt):
    """Uniform precision table (legacy <4) with a packed byte buffer."""
    precision_table = torch.full((num_blocks,), fmt, dtype=torch.uint8, device="cuda")
    # per-block bytes per head for this fmt
    if fmt == 0:
        eb = 2
    elif fmt == 3:
        eb = 0.5
    else:
        eb = 1
    block_bytes = int(block_size * head_dim * eb * num_kv_heads)
    total_bytes = block_bytes * num_blocks
    # flat [K_bytes | V_bytes]
    kv_cache = torch.zeros(2 * total_bytes, dtype=torch.uint8, device="cuda")
    # uniform stride offsets
    block_offsets = torch.arange(num_blocks, dtype=torch.int32, device="cuda") * block_bytes
    block_offsets = torch.cat([block_offsets, block_offsets])
    scale_table = torch.ones(2 * num_blocks, num_kv_heads, dtype=torch.float32, device="cuda")
    k_cache = kv_cache[:total_bytes]
    v_cache = kv_cache[total_bytes:]
    return k_cache, v_cache, precision_table, block_offsets, scale_table


query = torch.randn(1, num_heads, head_dim, dtype=torch.float16, device="cuda")
block_tables = torch.zeros(1, num_blocks, dtype=torch.int32, device="cuda")
context_lens = torch.tensor([num_blocks * block_size - 5], dtype=torch.int32, device="cuda")

for name, fmt in [("lossless", 0), ("e4m3", 1), ("fp4", 3)]:
    kc, vc, pt, bo, st = build_cache(fmt)
    out = torch.empty_like(query)
    # dispatch flags for uniform fmt (legacy <4)
    is_lossless = (fmt == 0)
    fp8_fmt = fmt if fmt in (1, 2) else 0
    v_fmt = fmt
    k_fmt = 3 if fmt == 3 else 0
    k_lossless = (fmt == 0)
    v_lossless = (fmt == 0)
    args = (query, kc, vc, block_tables, context_lens, pt, bo, st,
            softmax_scale, num_kv_heads, head_dim, -1,
            is_lossless, fp8_fmt, v_fmt, k_fmt, k_lossless, v_lossless, out)
    # warmup
    for _ in range(20):
        ops.psp_attention_enhanced(*args)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(200):
        ops.psp_attention_enhanced(*args)
    torch.cuda.synchronize()
    dt = (time.perf_counter() - t0) / 200
    print(f"{name:10s}  {dt*1e6:8.1f} us/decode-attn")
