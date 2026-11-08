"""Numerical comparison: PSP-KV lossless attention vs fp32 torch reference."""
import os
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import torch
from vllm.v1.attention.backends.psp_kv_attn import _get_psp_kv_ops

ops = _get_psp_kv_ops()
B, H, KVH, D, C = 1, 32, 8, 128, 512
scale = 1.0 / (D ** 0.5)
num_blocks = (C + 127) // 128
eb = 2  # fp16
block_bytes = int(128 * D * eb * KVH)
total_bytes = block_bytes * num_blocks

torch.manual_seed(0)
q = torch.randn(B, H, D, dtype=torch.float16, device="cuda")
k = torch.randn(num_blocks * 128, KVH, D, dtype=torch.float16, device="cuda")
v = torch.randn(num_blocks * 128, KVH, D, dtype=torch.float16, device="cuda")

kv_cache = torch.zeros(2 * total_bytes, dtype=torch.uint8, device="cuda")
kc, vc = kv_cache[:total_bytes], kv_cache[total_bytes:]
# token-major layout: [block][token][head][dim]  (PSP_KV_TOKEN_MAJOR=1)
for b in range(num_blocks):
    kblk = k[b * 128:(b + 1) * 128]  # [128, KVH, D]
    vblk = v[b * 128:(b + 1) * 128]
    kc.view(torch.float16)[b * (128 * KVH * D): (b + 1) * (128 * KVH * D)] = kblk.reshape(-1)
    vc.view(torch.float16)[b * (128 * KVH * D): (b + 1) * (128 * KVH * D)] = vblk.reshape(-1)

pt = torch.zeros(num_blocks, dtype=torch.uint8, device="cuda")
bo = torch.arange(num_blocks, dtype=torch.int32, device="cuda") * block_bytes
bo = torch.cat([bo, bo])
st = torch.ones(2 * num_blocks, KVH, dtype=torch.float32, device="cuda")
block_tables = torch.zeros(B, num_blocks, dtype=torch.int32, device="cuda")
for b in range(num_blocks):
    block_tables[0, b] = b
context_lens = torch.full((B,), C, dtype=torch.int32, device="cuda")
out = torch.empty(B, H, D, dtype=torch.float16, device="cuda")
args = (q, kc, vc, block_tables, context_lens, pt, bo, st,
        scale, KVH, D, -1, True, 0, 0, 0, True, True, out)
ops.psp_attention_enhanced(*args)
torch.cuda.synchronize()

# fp32 torch reference (decode: single query, full context).
q_ref = q[0].float()  # [H, D]
k_ref = k.float()     # [C, KVH, D]
v_ref = v.float()
# expand for GQA (H/KVH = 4)
g = H // KVH
k_r = k_ref.repeat_interleave(g, dim=1)  # [C, H, D]
v_r = v_ref.repeat_interleave(g, dim=1)
scores = torch.einsum('hd,chd->ch', q_ref, k_r) * scale  # [C, H]
p = torch.softmax(scores, dim=0)  # softmax over tokens
o_ref = torch.einsum('ch,chd->hd', p, v_r)

o = out[0].float()
err = (o - o_ref).abs()
print(f"PSP-KV lossless max abs err = {err.max().item():.3e}, mean = {err.mean().item():.3e}")

# Flash-attention fp16 reference for the same input.
from vllm.vllm_flash_attn.flash_attn_interface import flash_attn_varlen_func
g = H // KVH
q_fa = q  # [1, H, D]
k_fa = k.repeat_interleave(g, dim=1).unsqueeze(0)  # [1, C, H, D]
v_fa = v.repeat_interleave(g, dim=1).unsqueeze(0)
cu_q = torch.tensor([0, 1], dtype=torch.int32, device="cuda")
cu_k = torch.tensor([0, C], dtype=torch.int32, device="cuda")
o_fa = flash_attn_varlen_func(q_fa, k_fa, v_fa, 1, cu_q, C, cu_k,
                              softmax_scale=scale, causal=False, num_splits=0)
o_fa = o_fa[0].float()
err_fa = (o_fa - o_ref).abs()
print(f"FA         max abs err = {err_fa.max().item():.3e}, mean = {err_fa.mean().item():.3e}")
err_diff = (o - o_fa).abs()
print(f"PSP-KV vs FA max abs diff = {err_diff.max().item():.3e}")
