import torch
from vllm.vllm_flash_attn import flash_attn_varlen_func

torch.manual_seed(0)
dev = 'cuda'
HQ, HK, D, SQ, SK = 2, 2, 64, 1, 256   # 256 = 2 blocks of 128
NB = 2
BS = SK // NB  # block size

q = torch.randn(SQ, HQ, D, device=dev, dtype=torch.float16) * 0.5
k = torch.randn(SK, HK, D, device=dev, dtype=torch.float16) * 0.5
v = torch.randn(SK, HK, D, device=dev, dtype=torch.float16) * 0.5

k_descale = torch.tensor([[[0.7, 1.4]] * HK], device=dev)  # [1, HK, NB]
v_descale = torch.tensor([[[0.9, 1.6]] * HK], device=dev)

def quant(x, scale_per_blk):
    out = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    for b in range(NB):
        sl = slice(b * BS, (b + 1) * BS)
        out[sl] = (x[sl] / scale_per_blk[0, :, b:b+1]).to(torch.float8_e4m3fn)
    return out

q_fp8 = q.to(torch.float8_e4m3fn)
k_fp8 = quant(k, k_descale)
v_fp8 = quant(v, v_descale)

cu_q = torch.tensor([0, SQ], dtype=torch.int32, device=dev)
cu_k = torch.tensor([0, SK], dtype=torch.int32, device=dev)

out = flash_attn_varlen_func(
    q_fp8, k_fp8, v_fp8,
    max_seqlen_q=SQ, cu_seqlens_q=cu_q, max_seqlen_k=SK, cu_seqlens_k=cu_k,
    causal=True, softmax_scale=D ** -0.5,
    q_descale=None, k_descale=k_descale, v_descale=v_descale, fa_version=3)

k_deq = k_fp8.float()
v_deq = v_fp8.float()
for b in range(NB):
    sl = slice(b * BS, (b + 1) * BS)
    k_deq[sl] = k_deq[sl] * k_descale[0, :, b:b+1]
    v_deq[sl] = v_deq[sl] * v_descale[0, :, b:b+1]

scores = torch.einsum('qhd,khd->hqk', q_fp8.float(), k_deq)  # [HK, SQ, SK]
scores = scores / (D ** 0.5)
probs = torch.softmax(scores, dim=-1)
ref = torch.einsum('hqk,khd->qhd', probs, v_deq)  # [SQ, HK, D]
ref = ref.repeat_interleave(HQ // HK, dim=1)  # [SQ, HQ, D]

diff = (out.float() - ref).abs()
print("out shape", tuple(out.shape), "max abs diff", diff.max().item(), "mean", diff.mean().item())
print("PASS" if diff.max().item() < 0.05 else "FAIL")
