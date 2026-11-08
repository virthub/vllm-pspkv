#!/usr/bin/env python3
"""Round-trip: write K/V via quantize kernels, read back via dequant, check shift."""
import os, sys
os.environ.setdefault("HF_HUB_OFFLINE","1"); os.environ.setdefault("TRANSFORMERS_OFFLINE","1")
import torch
from vllm.v1.attention.backends.psp_kv_attn import _get_psp_kv_ops

ops = _get_psp_kv_ops()
D = 128
KVH = 8
BLOCK = 128
NB = 4  # physical blocks
EB = 2   # fp16 bytes (lossless)

# token-major cache: [block][token][head][dim]
k_cache = torch.zeros(NB * BLOCK * KVH * D * EB, dtype=torch.uint8, device="cuda")
v_cache = torch.zeros(NB * BLOCK * KVH * D * EB, dtype=torch.uint8, device="cuda")

# uniform lossless precision + uniform block offsets (fp16 stride)
pt = torch.zeros(NB, dtype=torch.uint8, device="cuda")
block_bytes = BLOCK * KVH * D * EB
bo = torch.arange(2 * NB, dtype=torch.int32, device="cuda") * block_bytes
st = torch.zeros(2 * NB * KVH * (D // 16), dtype=torch.float32, device="cuda")

# known K/V: value encodes (token, head, dim) so we can detect any shift
k_src = torch.arange(130, dtype=torch.float32, device="cuda").reshape(130, 1, 1).expand(130, KVH, D).contiguous().half()
v_src = (k_src + 1000).contiguous()

# prefill: write 128 tokens via quantize_and_scatter (block 0, offsets 0..127)
slot_map_pre = torch.arange(128, dtype=torch.int64, device="cuda")
# decode: write token 128 -> slot 128 (block 1 offset 0), token 129 -> slot 129
slot_map_dec = torch.tensor([128, 129], dtype=torch.int64, device="cuda")

# prefill scratch layout: [token, 2, KVH, D]
scratch = torch.zeros(128, 2, KVH, D, dtype=torch.float16, device="cuda")
scratch[:, 0] = k_src[:128]
scratch[:, 1] = v_src[:128]
block_tables_pre = torch.tensor([[0]], dtype=torch.int32, device="cuda")
ops.quantize_and_scatter_kv(scratch, k_cache, v_cache, block_tables_pre, slot_map_pre, pt, bo, st, 128, 1)

# decode: quantize_single_token for tokens 128, 129
ops.quantize_single_token(k_src[128:130].contiguous(), v_src[128:130].contiguous(),
                          k_cache, v_cache, slot_map_dec, pt, bo, st, 1)

# dequantize context_lens = 130 (prompt 128 + 2 decode)
ctx = torch.tensor([130], dtype=torch.int32, device="cuda")
offs = torch.tensor([0], dtype=torch.int32, device="cuda")
block_tables = torch.tensor([[0, 1]], dtype=torch.int32, device="cuda")  # seq uses physical blocks 0,1
k_out = torch.empty(2 * BLOCK, KVH, D, dtype=torch.float16, device="cuda")
v_out = torch.empty_like(k_out)
ops.psp_dequantize_prefix_batch(k_cache, v_cache, block_tables, ctx, offs, pt, bo, st,
                                KVH, BLOCK, D, k_out, v_out)

# compare k_out[:130] vs k_src[:130]
got = k_out[:130].float()
exp = k_src[:130].float()
err = (got - exp).abs().max().item()
print(f"K roundtrip max_abs_err={err:.6f}")
for t in [0, 1, 127, 128, 129]:
    print(f"  token {t}: got={got[t,0,0].item():.1f} exp={exp[t,0,0].item():.1f} match={bool((got[t]-exp[t]).abs().max() < 1e-3)}")
