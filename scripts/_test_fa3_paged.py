import torch
from vllm.vllm_flash_attn import flash_attn_varlen_func

torch.manual_seed(0)
dev = 'cuda'
B, HQ, HK, D, SQ, SK = 1, 4, 2, 128, 1, 256   # head_dim=128, 2 pages of 128 tokens
PS = 128  # page_size = block_size
NB = SK // PS  # 2 pages
num_pages = 4  # physical pool

# token-major paged K/V: [num_pages, PS, HK, D] fp8
k = torch.randn(num_pages, PS, HK, D, device=dev, dtype=torch.float16) * 0.5
v = torch.randn(num_pages, PS, HK, D, device=dev, dtype=torch.float16) * 0.5
q = torch.randn(SQ, HQ, D, device=dev, dtype=torch.float16) * 0.5

# per-(block, head) scale; distinct per block
k_descale = torch.tensor([[[0.7, 1.4]] * HK], device=dev)  # [B, HK, NB]
v_descale = torch.tensor([[[0.9, 1.6]] * HK], device=dev)

def quant(x, scale_per_blk):
    out = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    for b in range(NB):
        out[b] = (x[b] / scale_per_blk[0, :, b:b+1]).to(torch.float8_e4m3fn)
    return out

q_fp8 = q.to(torch.float8_e4m3fn)
k_fp8 = quant(k, k_descale)
v_fp8 = quant(v, v_descale)

# block_table: [B, NB] -> physical page indices (0, 1)
block_table = torch.tensor([[0, 1]], dtype=torch.int32, device=dev)
cu_q = torch.tensor([0, SQ], dtype=torch.int32, device=dev)
seqused_k = torch.tensor([SK], dtype=torch.int32, device=dev)

out = flash_attn_varlen_func(
    q_fp8, k_fp8, v_fp8,
    max_seqlen_q=SQ, cu_seqlens_q=cu_q, max_seqlen_k=SK, seqused_k=seqused_k,
    causal=True, softmax_scale=D ** -0.5, block_table=block_table,
    q_descale=None, k_descale=k_descale, v_descale=v_descale, fa_version=3)

# reference: token-major dequant
k_deq = k_fp8.float()
v_deq = v_fp8.float()
for b in range(NB):
    k_deq[b] = k_deq[b] * k_descale[0, :, b:b+1]
    v_deq[b] = v_deq[b] * v_descale[0, :, b:b+1]
k_seq = k_deq[:NB].reshape(SK, HK, D)  # [SK, HK, D]
v_seq = v_deq[:NB].reshape(SK, HK, D)
qseq = q_fp8.float().squeeze(0)  # [HQ, D]
# GQA: HK -> HQ
g = HQ // HK
k_seq = k_seq.repeat_interleave(g, dim=1)  # [SK, HQ, D]
v_seq = v_seq.repeat_interleave(g, dim=1)
scores = torch.einsum('hd,thd->ht', qseq, k_seq) / (D ** 0.5)
weights = torch.softmax(scores, dim=-1)
ref = torch.einsum('ht,thd->hd', weights, v_seq)

diff = (out.float().squeeze(0) - ref).abs()
print("out shape", tuple(out.shape), "max abs diff", diff.max().item())
print("PASS" if diff.max().item() < 0.05 else "FAIL")
