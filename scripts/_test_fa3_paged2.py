import torch
from vllm.vllm_flash_attn import flash_attn_varlen_func

torch.manual_seed(0)
dev = 'cuda'
B, HQ, HK, D, SQ, SK = 1, 4, 2, 128, 1, 256
PS = 128
NB = SK // PS
num_pages = 4

k = torch.randn(num_pages, PS, HK, D, device=dev, dtype=torch.float16) * 0.5
v = torch.randn(num_pages, PS, HK, D, device=dev, dtype=torch.float16) * 0.5
q = torch.randn(SQ, HQ, D, device=dev, dtype=torch.float16) * 0.5
k_descale = torch.tensor([[[0.7, 1.4]] * HK], device=dev)
v_descale = torch.tensor([[[0.9, 1.6]] * HK], device=dev)

def quant(x, scale_per_blk):
    out = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    for b in range(NB):
        out[b] = (x[b] / scale_per_blk[0, :, b:b+1]).to(torch.float8_e4m3fn)
    return out

q_fp8 = q.to(torch.float8_e4m3fn)
k_fp8 = quant(k, k_descale)
v_fp8 = quant(v, v_descale)

block_table = torch.tensor([[0, 1]], dtype=torch.int32, device=dev)
cu_q = torch.tensor([0, SQ], dtype=torch.int32, device=dev)
seqused_k = torch.tensor([SK], dtype=torch.int32, device=dev)
common = dict(max_seqlen_q=SQ, cu_seqlens_q=cu_q, max_seqlen_k=SK,
              causal=True, softmax_scale=D ** -0.5, q_descale=None, fa_version=3)

# Paged (token-major)
out_paged = flash_attn_varlen_func(q_fp8, k_fp8, v_fp8, seqused_k=seqused_k, block_table=block_table,
                                   k_descale=k_descale, v_descale=v_descale, **common)

# Non-paged: contiguous [SK, HK, D] (the first NB pages, token-major)
k_np = k_fp8[:NB].reshape(SK, HK, D).contiguous()
v_np = v_fp8[:NB].reshape(SK, HK, D).contiguous()
out_np = flash_attn_varlen_func(q_fp8, k_np, v_np, cu_seqlens_k=torch.tensor([0, SK], dtype=torch.int32, device=dev),
                                k_descale=k_descale, v_descale=v_descale, **common)

d = (out_paged.float() - out_np.float()).abs()
print("paged-vs-nonpaged max abs diff", d.max().item())
print("MATCH" if d.max().item() < 1e-3 else "MISMATCH")
