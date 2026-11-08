// Minimal CUTE gemm using the SM120 f8f6f4 m16n8k32 atom, to verify the fp8
// MMA works on SM120 and to get the correct fragment layout.
#include <cuda_runtime.h>
#include <cstdio>
#include <cute/tensor.hpp>
#include <cute/atom/mma_traits_sm120.hpp>
#include <cute/arch/mma_sm120.hpp>
#include <cutlass/numeric_types.h>

using namespace cute;

__global__ void cute_fp8_gemm(const uint8_t* A, const uint8_t* B, float* C)
{
    using MMA_T = SM120_16x8x32_TN<float_e4m3_t, float_e4m3_t, float>;
    auto mma = make_tiled_mma(MMA_T{});
    auto a = make_tensor(make_gmem_ptr(A), make_shape(Int<16>{}, Int<32>{}), make_stride(Int<32>{}, Int<1>{}));
    auto b = make_tensor(make_gmem_ptr(B), make_shape(Int<8>{}, Int<32>{}), make_stride(Int<32>{}, Int<1>{}));
    auto c = make_tensor(make_gmem_ptr(C), make_shape(Int<16>{}, Int<8>{}), make_stride(Int<8>{}, Int<1>{}));
    auto thr = mma.get_slice(threadIdx.x);
    auto tCrA = thr.partition_fragment_A(a);
    auto tCrB = thr.partition_fragment_B(b);
    auto tCrC = thr.partition_fragment_C(c);
    clear(tCrC);
    copy(a, tCrA);
    copy(b, tCrB);
    gemm(mma, tCrC, tCrA, tCrB);
    copy(tCrC, c);
}

__host__ __device__ uint8_t q(float v){ __nv_fp8_e4m3 f=(__nv_fp8_e4m3)v; return *reinterpret_cast<uint8_t*>(&f); }
__host__ __device__ float dq(uint8_t q){ __nv_fp8_e4m3 f; *reinterpret_cast<uint8_t*>(&f)=q; return (float)f; }

int main(){
    const int M=16,N=8,K=32;
    float hA[M*K], hB[K*N];
    for(int i=0;i<M*K;i++) hA[i]=(float)((i*7)%23-11)/8.0f;
    for(int i=0;i<K*N;i++) hB[i]=(float)((i*5)%17-8)/8.0f;
    uint8_t hAq[M*K], hBq[K*N];
    for(int m=0;m<M;m++) for(int k=0;k<K;k++) hAq[m*K+k]=q(hA[m*K+k]);
    for(int n=0;n<N;n++) for(int k=0;k<K;k++) hBq[n*K+k]=q(hB[k*N+n]);
    uint8_t *dA,*dB; float *dC;
    cudaMalloc(&dA,M*K); cudaMalloc(&dB,K*N); cudaMalloc(&dC,M*N*4);
    cudaMemcpy(dA,hAq,M*K,cudaMemcpyHostToDevice);
    cudaMemcpy(dB,hBq,K*N,cudaMemcpyHostToDevice);
    cudaMemset(dC,0,M*N*4);
    cute_fp8_gemm<<<1,32>>>(dA,dB,dC);
    float hC[M*N]; cudaMemcpy(hC,dC,M*N*4,cudaMemcpyDeviceToHost);
    float ref[M*N];
    for(int m=0;m<M;m++) for(int n=0;n<N;n++){ float s=0; for(int k=0;k<K;k++) s+=dq(hAq[m*K+k])*dq(hBq[n*K+k]); ref[m*N+n]=s; }
    float mx=0; int bad=0;
    for(int m=0;m<M;m++) for(int n=0;n<N;n++){ float e=fabsf(hC[m*N+n]-ref[m*N+n]); if(e>mx)mx=e; if(e>0.5f){bad++; if(bad<=4)printf("  C[%d][%d]=%.2f ref %.2f\n",m,n,hC[m*N+n],ref[m*N+n]);} }
    printf("cute gemm maxerr=%.4f bad=%d\n",mx,bad);
    return bad?1:0;
}
