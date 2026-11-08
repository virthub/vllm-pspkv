import torch
from vllm.vllm_flash_attn import flash_attn_varlen_func

torch.manual_seed(0)
dev = 'cuda'
HQ, HK, D, SQ, SK = 2, 2, 64, 1, 256
NB = 2
BS = SK // NB

q = torch.randn(SQ, HQ, D, device=dev, dtype=torch.float16) * 0.5
k = torch.randn(SK, HK, D, device=dev, dtype=torch.float16) * 0.5
v = torch.randn(SK, HK, D, device=dev, dtype=torch.float16) * 0.5
q_fp8 = q.to(torch.float8_e4m3fn)
k_fp8 = k.to(torch.float8_e4m3fn)
v_fp8 = v.to(torch.float8_e4m3fn)

cu_q = torch.tensor([0, SQ], dtype=torch.int32, device=dev)
cu_k = torch.tensor([0, SK], dtype=torch.int32, device=dev)

# No descale at all
out_none = flash_attn_varlen_func(q_fp8, k_fp8, v_fp8,
    max_seqlen_q=SQ, cu_seqlens_q=cu_q, max_seqlen_k=SK, cu_seqlens_k=cu_k,
    causal=True, softmax_scale=D ** -0.5, fa_version=3)

scores = torch.einsum('qhd,khd->hqk', q_fp8.float(), k_fp8.float()) / (D ** 0.5)
probs = torch.softmax(scores, dim=-1)
ref = torch.einsum('hqk,khd->qhd', probs, v_fp8.float())
print("NO-DESCALE diff:", (out_none.float() - ref).abs().max().item())

# Uniform descale = 2.0 for K, 3.0 for V
kd = torch.full((1, HK, NB), 2.0, device=dev)
vd = torch.full((1, HK, NB), 3.0, device=dev)
out_uni = flash_attn_varlen_func(q_fp8, k_fp8, v_fp8,
    max_seqlen_q=SQ, cu_seqlens_q=cu_q, max_seqlen_k=SK, cu_seqlens_k=cu_k,
    causal=True, softmax_scale=D ** -0.5,
    q_descale=None, k_descale=kd, v_descale=vd, fa_version=3)

k_deq = k_fp8.float() * 2.0
v_deq = v_fp8.float() * 3.0
scores = torch.einsum('qhd,khd->hqk', q_fp8.float(), k_deq) / (D ** 0.5)
probs = torch.softmax(scores, dim=-1)
ref2 = torch.einsum('hqk,khd->qhd', probs, v_deq)
print("UNIFORM descale diff:", (out_uni.float() - ref2).abs().max().item())
