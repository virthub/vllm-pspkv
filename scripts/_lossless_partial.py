#!/usr/bin/env python3
"""Test the C++ lossless kernel with a PARTIAL block (decode case)."""
import os, sys
os.environ.setdefault("HF_HUB_OFFLINE","1"); os.environ.setdefault("TRANSFORMERS_OFFLINE","1")
import torch
from vllm.v1.attention.backends.psp_kv_attn import _get_psp_kv_ops
ops = _get_psp_kv_ops()

B, H, KVH, D = 1, 32, 8, 128
C = int(sys.argv[1]) if len(sys.argv) > 1 else 129
BLOCK = 128
num_blocks = (C + BLOCK - 1) // BLOCK
block_bytes = BLOCK * KVH * D * 2
total_bytes = block_bytes * num_blocks

torch.manual_seed(0)
q = torch.randn(B, H, D, dtype=torch.float16, device="cuda")
k = torch.randn(C, KVH, D, dtype=torch.float16, device="cuda")
v = torch.randn(C, KVH, D, dtype=torch.float16, device="cuda")

kv_cache = torch.zeros(2 * total_bytes, dtype=torch.uint8, device="cuda")
kc, vc = kv_cache[:total_bytes], kv_cache[total_bytes:]
# token-major [block][token][head][dim]; C tokens fill block0 + start of block1
kc.view(torch.float16)[:C * KVH * D] = k.reshape(-1)
vc.view(torch.float16)[:C * KVH * D] = v.reshape(-1)

pt = torch.zeros(num_blocks, dtype=torch.uint8, device="cuda")
bo = torch.arange(num_blocks, dtype=torch.int32, device="cuda") * block_bytes
bo = torch.cat([bo, bo])  # [2*num_blocks] K offsets | V offsets
st = torch.ones(2 * num_blocks, KVH, dtype=torch.float32, device="cuda")
block_tables = torch.zeros(B, num_blocks, dtype=torch.int32, device="cuda")
for b in range(num_blocks):
    block_tables[0, b] = b
context_lens = torch.full((B,), C, dtype=torch.int32, device="cuda")
out = torch.empty(B, H, D, dtype=torch.float16, device="cuda")
scale = 1.0 / (D ** 0.5)
args = (q, kc, vc, block_tables, context_lens, pt, bo, st,
        scale, KVH, D, -1, True, 0, 0, 0, True, True, out)
ops.psp_attention_enhanced(*args)
torch.cuda.synchronize()

q_ref = q[0].float()
g = H // KVH
k_r = k.float().repeat_interleave(g, dim=1)
v_r = v.float().repeat_interleave(g, dim=1)
scores = torch.einsum('hd,chd->ch', q_ref, k_r) * scale
p = torch.softmax(scores, dim=0)
o_ref = torch.einsum('ch,chd->hd', p, v_r)
err = (out[0].float() - o_ref).abs()
print(f"C={C} num_blocks={num_blocks} lossless max_abs_err={err.max().item():.3e} mean={err.mean().item():.3e}")
