import torch
from vllm.vllm_flash_attn import flash_attn_varlen_func

torch.manual_seed(0)
dev = 'cuda'
HQ, HK, D, SQ, SK = 2, 2, 64, 1, 256
NB = 2

q = torch.randn(SQ, HQ, D, device=dev, dtype=torch.float16) * 0.5
k = torch.randn(SK, HK, D, device=dev, dtype=torch.float16) * 0.5
v = torch.randn(SK, HK, D, device=dev, dtype=torch.float16) * 0.5
q_fp8 = q.to(torch.float8_e4m3fn)
k_fp8 = k.to(torch.float8_e4m3fn)
v_fp8 = v.to(torch.float8_e4m3fn)

cu_q = torch.tensor([0, SQ], dtype=torch.int32, device=dev)
cu_k = torch.tensor([0, SK], dtype=torch.int32, device=dev)
kw = dict(max_seqlen_q=SQ, cu_seqlens_q=cu_q, max_seqlen_k=SK, cu_seqlens_k=cu_k,
          causal=True, softmax_scale=D ** -0.5, fa_version=3)

# 2D descale (per-head, uniform 2.0)
kd2 = torch.full((1, HK), 2.0, device=dev)
vd2 = torch.full((1, HK), 3.0, device=dev)
out2 = flash_attn_varlen_func(q_fp8, k_fp8, v_fp8, q_descale=None, k_descale=kd2, v_descale=vd2, **kw)

# 3D descale (per-block, uniform 2.0 across blocks)
kd3 = torch.full((1, HK, NB), 2.0, device=dev)
vd3 = torch.full((1, HK, NB), 3.0, device=dev)
out3 = flash_attn_varlen_func(q_fp8, k_fp8, v_fp8, q_descale=None, k_descale=kd3, v_descale=vd3, **kw)

d = (out2.float() - out3.float()).abs()
print("2D-vs-3D uniform descale: max abs diff", d.max().item())
print("MATCH" if d.max().item() < 1e-3 else "MISMATCH")

# per-block variation: block0=1.0, block1=2.0 (K); reference via per-token dequant
kd_var = torch.tensor([[[1.0, 2.0]] * HK], device=dev)
out_var = flash_attn_varlen_func(q_fp8, k_fp8, v_fp8, q_descale=None, k_descale=kd_var, v_descale=None, **kw)
# reference: FA3 with 2D descale applied to a pre-scaled K (block0 *1.0, block1 *2.0)
k_scaled = k_fp8.float().clone()
k_scaled[128:] = k_scaled[128:] * 2.0
k_scaled_fp8 = (k_scaled * 1.0).to(torch.float8_e4m3fn)  # requantize after scaling
out_ref = flash_attn_varlen_func(q_fp8, k_scaled_fp8, v_fp8, q_descale=None, k_descale=None, v_descale=None, **kw)
d2 = (out_var.float() - out_ref.float()).abs()
print("per-block K descale vs pre-scaled reference: max abs diff", d2.max().item())
