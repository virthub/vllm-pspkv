// MMA throughput microbenchmark with ILP (8 independent accumulators) to hide latency.
#include <cuda_runtime.h>
#include <mma.h>
#include <cstdio>
#include <stdint.h>
using namespace nvcuda::wmma;

#define ITERS 1000000

__global__ void bench_fp16(float* out) {
    fragment<matrix_a,16,16,16,__half,row_major> a;
    fragment<matrix_b,16,16,16,__half,col_major> b;
    fragment<accumulator,16,16,16,float> c[8];
    fill_fragment(a, __float2half(0.5f));
    fill_fragment(b, __float2half(0.5f));
    #pragma unroll
    for (int j = 0; j < 8; j++) fill_fragment(c[j], 0.0f);
    for (int i = 0; i < ITERS; i++) {
        #pragma unroll
        for (int j = 0; j < 8; j++) mma_sync(c[j], a, b, c[j]);
    }
    float s = 0;
    #pragma unroll
    for (int j = 0; j < 8; j++) s += c[j].x[0];
    if (threadIdx.x == 0) out[0] = s;
}

__global__ void bench_int8(int* out) {
    fragment<matrix_a,16,16,16,signed char,row_major> a;
    fragment<matrix_b,16,16,16,signed char,col_major> b;
    fragment<accumulator,16,16,16,int> c[8];
    fill_fragment(a, (signed char)1);
    fill_fragment(b, (signed char)1);
    #pragma unroll
    for (int j = 0; j < 8; j++) fill_fragment(c[j], 0);
    for (int i = 0; i < ITERS; i++) {
        #pragma unroll
        for (int j = 0; j < 8; j++) mma_sync(c[j], a, b, c[j]);
    }
    int s = 0;
    #pragma unroll
    for (int j = 0; j < 8; j++) s += c[j].x[0];
    if (threadIdx.x == 0) out[0] = s;
}

// int8 mma with the "saturate" flag off but using .u8 (unsigned) to probe variant.
__global__ void bench_bf16(float* out) {
    fragment<matrix_a,16,16,16,__nv_bfloat16,row_major> a;
    fragment<matrix_b,16,16,16,__nv_bfloat16,col_major> b;
    fragment<accumulator,16,16,16,float> c[8];
    fill_fragment(a, __float2bfloat16(0.5f));
    fill_fragment(b, __float2bfloat16(0.5f));
    #pragma unroll
    for (int j = 0; j < 8; j++) fill_fragment(c[j], 0.0f);
    for (int i = 0; i < ITERS; i++) {
        #pragma unroll
        for (int j = 0; j < 8; j++) mma_sync(c[j], a, b, c[j]);
    }
    float s = 0;
    #pragma unroll
    for (int j = 0; j < 8; j++) s += c[j].x[0];
    if (threadIdx.x == 0) out[0] = s;
}

template <typename F, typename T>
float run(F kern, T** opp, const char* name) {
    cudaMallocManaged(opp, 4);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    kern<<<1, 32>>>(*opp);
    cudaDeviceSynchronize();
    cudaEventRecord(e0);
    kern<<<1, 32>>>(*opp);
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);
    float ms; cudaEventElapsedTime(&ms, e0, e1);
    // 8 MMAs per iter, each 16x16x16 MACs = 4096
    float gmacs = (float)ITERS * 8.0f * 16.0f * 16.0f * 16.0f / 1e9f;
    float tops = gmacs / (ms / 1e3f);
    printf("%-8s: %.3f ms  %.1f GOPS/warp\n", name, ms, tops);
    return tops;
}

int main() {
    int dev; cudaGetDevice(&dev);
    cudaDeviceProp p; cudaGetDeviceProperties(&p, dev);
    printf("GPU: %s  sm=%d.%d\n", p.name, p.major, p.minor);
    float* fout; int* iout;
    run(bench_fp16, &fout, "fp16");
    run(bench_bf16, &fout, "bf16");
    run(bench_int8, &iout, "int8");
    return 0;
}
