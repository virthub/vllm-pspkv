import torch
from vllm.vllm_flash_attn import flash_attn_varlen_func
torch.manual_seed(0)
dev='cuda'; HQ,HK,D,SQ,SK=2,2,128,1,256
q=torch.randn(SQ,HQ,D,device=dev,dtype=torch.float16)*0.5
k=torch.randn(SK,HK,D,device=dev,dtype=torch.float16)*0.5
v=torch.randn(SK,HK,D,device=dev,dtype=torch.float16)*0.5
qf=q.to(torch.float8_e4m3fn); kf=k.to(torch.float8_e4m3fn); vf=v.to(torch.float8_e4m3fn)
cuq=torch.tensor([0,SQ],dtype=torch.int32,device=dev); cuk=torch.tensor([0,SK],dtype=torch.int32,device=dev)
NB=SK//128
def run(kd, vd):
    return flash_attn_varlen_func(qf,kf,vf,max_seqlen_q=SQ,cu_seqlens_q=cuq,max_seqlen_k=SK,cu_seqlens_k=cuk,
        causal=True,softmax_scale=D**-0.5,q_descale=None,k_descale=kd,v_descale=vd,fa_version=3)
a=run(torch.full((1,HK,NB),1.0,device=dev), torch.full((1,HK,NB),1.0,device=dev))
b=run(torch.tensor([[[2.0,0.5]]*HK],device=dev), torch.tensor([[[2.0,0.5]]*HK],device=dev))
print("uniform vs per-block diff", (a.float()-b.float()).abs().max().item())
