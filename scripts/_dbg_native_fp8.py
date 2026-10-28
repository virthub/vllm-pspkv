"""Debug: verify native fp8 QK is correct modulo Q quantization."""
import os, sys
os.environ["PSPKV_NATIVE_FP8_MMA"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from tests.v1.attention.test_psp_kv_correctness import (
    _build_kv_cache, _DummyCommon,
)
from tests.v1.attention.psp_kv_reference import reference_paged_attention
from vllm.v1.attention.backends.psp_kv_attn import (
    PSPKVAttentionImpl, PSPKVAttentionMetadata,
)

num_heads, num_kv_heads, fmt, head_size = 8, 8, 1, 128
block_size, ctx = 16, 16
softmax_scale = 1.0 / (head_size ** 0.5)
dtype = torch.float16

query = torch.randn(1, num_heads, head_size, dtype=dtype, device="cuda")
k_full = torch.randn(ctx, num_kv_heads, head_size, dtype=dtype, device="cuda") * 2.0
v_full = torch.randn(ctx, num_kv_heads, head_size, dtype=dtype, device="cuda") * 2.0

kv_cache, block_offsets = _build_kv_cache(k_full, v_full, 1, fmt, dtype)
precision_table = torch.full((1,), fmt, dtype=torch.uint8, device="cuda")
scale_table = torch.tensor([[0.5, 1.0, 2.0, 0.5, 1.0, 2.0, 0.5, 1.0][:num_kv_heads]],
                           dtype=torch.float32, device="cuda")
scale_table = torch.cat([scale_table] * 2)

meta = _DummyCommon()
meta.block_table_tensor = torch.zeros(1, 1, dtype=torch.int32, device="cuda")
meta.context_lens = torch.tensor([ctx], dtype=torch.int32, device="cuda")
meta.seq_lens = meta.context_lens
meta.precision_table = precision_table
meta.block_offsets = block_offsets
meta.scale_table = scale_table
meta.tma_descriptors = None
meta.format_generation = 1
meta.num_actual_tokens = 1
meta.num_reqs = 1
meta.precision_is_lossless = False
meta.precision_fp8_fmt = fmt
meta.precision_v_fmt = fmt
meta.precision_k_fmt = 0
meta.precision_k_lossless = False
meta.precision_v_lossless = False

class L: pass
layer = L(); layer.num_heads = num_heads; layer.num_kv_heads = num_kv_heads
layer.head_size = head_size; layer.scale = softmax_scale
impl = PSPKVAttentionImpl(num_heads, head_size, softmax_scale, num_kv_heads, None, None, "auto")
output = torch.empty_like(query)
impl.forward(layer=layer, query=query, key=torch.empty(0), value=torch.empty(0),
             kv_cache=kv_cache, attn_metadata=PSPKVAttentionMetadata(common_metadata=meta),
             output=output)

# Reference with fp16 Q (tight) and with Q quantized to fp8 (matching the kernel).
ref = reference_paged_attention(query, kv_cache[0], kv_cache[1], meta.block_table_tensor,
                                meta.context_lens, 1, precision_table, scale_table,
                                block_offsets, num_kv_heads, block_size, softmax_scale)

# Q quantize to fp8 with per-head max_abs/240 scale (replicating the kernel).
qf = query.float()
qmax = qf.abs().max(dim=-1, keepdim=True).values
qscale = (qmax / 240.0).clamp(min=1e-8)
qf8 = (qf / qscale).clamp(-448, 448).to(torch.float8_e4m3fn).float()
qdeq = (qf8 * qscale).to(dtype)
ref_q = reference_paged_attention(qdeq, kv_cache[0], kv_cache[1], meta.block_table_tensor,
                                  meta.context_lens, 1, precision_table, scale_table,
                                  block_offsets, num_kv_heads, block_size, softmax_scale)

def me(a, b):
    return (a.float() - b.float()).abs().max().item()

print(f"native vs fp16-ref : {me(output, ref):.4f}")
print(f"native vs fp8Q-ref : {me(output, ref_q):.4f}")
print(f"fp8Q-ref vs fp16-ref : {me(ref_q, ref):.4f}")
