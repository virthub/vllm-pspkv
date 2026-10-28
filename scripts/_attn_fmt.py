"""Time PSP-KV attention: lossless vs fp8 (e4m3) at batch 1 vs 8."""
import os
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch  # noqa: E402
from vllm.v1.attention.backends.psp_kv_attn import _get_psp_kv_ops  # noqa: E402

H, KVH, D, C = 32, 8, 80, 941
scale = 1.0 / (D ** 0.5)


def time_attn(B, fmt):
    ops = _get_psp_kv_ops()
    num_blocks = (C + 15) // 16
    if fmt == "lossless":
        eb, is_lossless, fp8_fmt, kf, vf, kl, vl, prec = 2, True, 0, 0, 0, True, True, 0
    elif fmt == "e4m3":
        eb, is_lossless, fp8_fmt, kf, vf, kl, vl, prec = 1, False, 1, 1, 1, False, False, 5
    else:
        raise ValueError(fmt)
    block_bytes = int(16 * D * eb * KVH)
    total_bytes = block_bytes * num_blocks
    kv_cache = torch.zeros(2 * total_bytes, dtype=torch.uint8, device="cuda")
    kc, vc = kv_cache[:total_bytes], kv_cache[total_bytes:]
    pt = torch.full((num_blocks,), prec, dtype=torch.uint8, device="cuda")
    bo = torch.arange(num_blocks, dtype=torch.int32, device="cuda") * block_bytes
    bo = torch.cat([bo, bo])
    st = torch.ones(2 * num_blocks, KVH, dtype=torch.float32, device="cuda")
    query = torch.randn(B, H, D, dtype=torch.float16, device="cuda")
    block_tables = torch.zeros(B, num_blocks, dtype=torch.int32, device="cuda")
    context_lens = torch.full((B,), C - 5, dtype=torch.int32, device="cuda")
    out = torch.empty_like(query)
    args = (query, kc, vc, block_tables, context_lens, pt, bo, st,
            scale, KVH, D, -1, is_lossless, fp8_fmt, vf, kf, kl, vl, out)
    for _ in range(20):
        ops.psp_attention_enhanced(*args)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(200):
        ops.psp_attention_enhanced(*args)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / 200


for B in [1, 8]:
    l = time_attn(B, "lossless")
    e = time_attn(B, "e4m3")
    print(f"batch {B}: lossless {l*1e6:6.1f}us  e4m3 {e*1e6:6.1f}us  "
          f"e4m3/lossless {e/l:.2f}x")
