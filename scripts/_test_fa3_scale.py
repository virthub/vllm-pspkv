import torch
from vllm.vllm_flash_attn import flash_attn_varlen_func

torch.manual_seed(0)
dev = 'cuda'
HQ, HK, D, SQ, SK = 4, 2, 128, 1, 128
g = HQ // HK

# Realistic magnitudes: Q large (~10), K/V small (~0.1)
q = torch.randn(SQ, HQ, D, device=dev, dtype=torch.float16) * 10.0
k = torch.randn(SK, HK, D, device=dev, dtype=torch.float16) * 0.1
v = torch.randn(SK, HK, D, device=dev, dtype=torch.float16) * 0.1

FP_MAX = 240.0
q_scale = q.float().abs().amax() / FP_MAX
k_scale = k.float().abs().amax() / FP_MAX
v_scale = v.float().abs().amax() / FP_MAX

q_fp8 = q.to(torch.float8_e4m3fn)  # direct, no scaling (FA3 non-softcap path ignores q_descale)
k_fp8 = (k.float() / k_scale).to(torch.float8_e4m3fn)
v_fp8 = (v.float() / v_scale).to(torch.float8_e4m3fn)

cu_q = torch.tensor([0, SQ], dtype=torch.int32, device=dev)
cu_k = torch.tensor([0, SK], dtype=torch.int32, device=dev)
q_descale = None
k_descale = k_scale.expand(1, HK, 1).contiguous()   # 3D per-block, non-zero stride2
v_descale = v_scale.expand(1, HK, 1).contiguous()

out = flash_attn_varlen_func(
    q_fp8, k_fp8, v_fp8, max_seqlen_q=SQ, cu_seqlens_q=cu_q, max_seqlen_k=SK,
    cu_seqlens_k=cu_k, causal=True, softmax_scale=D ** -0.5,
    q_descale=q_descale, k_descale=k_descale, v_descale=v_descale, fa_version=3)

# reference: dequant then fp32 attention
q_d = q_fp8.float()                 # Q is direct (no scale)
k_d = k_fp8.float() * k_scale
v_d = v_fp8.float() * v_scale
k_d = k_d.repeat_interleave(g, dim=1)
v_d = v_d.repeat_interleave(g, dim=1)
scores = torch.einsum('qhd,khd->qhk', q_d, k_d) / (D ** 0.5)
weights = torch.softmax(scores, dim=-1)
ref = torch.einsum('qhk,khd->qhd', weights, v_d)

print("q_scale", q_scale.item(), "k_scale", k_scale.item(), "v_scale", v_scale.item())
print("out[0,0,:4]", out.float()[0,0,:4].tolist())
print("ref[0,0,:4]", ref[0,0,:4].tolist())
o = out.float()
print("out absmax per head", o.abs().amax(dim=(0,2)).tolist())
print("out max loc", (o.abs() == o.abs().max()).nonzero().tolist())
print("out max", o.abs().max().item(), "ref max", ref.abs().max().item())
