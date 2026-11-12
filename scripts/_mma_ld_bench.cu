// Measures load_matrix_sync + mma_sync throughput (incl. shared-memory loads),
// fp16 vs int8, to isolate whether int8 fragment loads are the bottleneck.
#include <cuda_runtime.h>
#include <mma.h>
#include <cstdio>
using namespace nvcuda::wmma;

#define ITERS 100000
#define NT8 8

__global__ void bench_fp16(float* out) {
    __shared__ __half a_s[16*16], b_s[16*16];
    for (int i = threadIdx.x; i < 16*16; i += 32) { a_s[i] = __float2half(0.5f); b_s[i] = __float2half(0.5f); }
    __syncthreads();
    fragment<matrix_a,16,16,16,__half,row_major> a;
    fragment<matrix_b,16,16,16,__half,col_major> b;
    fragment<accumulator,16,16,16,float> c[8];
    load_matrix_sync(a, a_s, 16);
    #pragma unroll
    for (int j = 0; j < 8; j++) fill_fragment(c[j], 0.0f);
    for (int i = 0; i < ITERS; i++) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            load_matrix_sync(b, b_s, 16);
            mma_sync(c[j], a, b, c[j]);
        }
    }
    float s = 0;
    #pragma unroll
    for (int j = 0; j < 8; j++) s += c[j].x[0];
    if (threadIdx.x == 0) out[0] = s;
}

__global__ void bench_int8(int* out) {
    __shared__ signed char a_s[16*16], b_s[16*16];
    for (int i = threadIdx.x; i < 16*16; i += 32) { a_s[i] = 1; b_s[i] = 1; }
    __syncthreads();
    fragment<matrix_a,16,16,16,signed char,row_major> a;
    fragment<matrix_b,16,16,16,signed char,col_major> b;
    fragment<accumulator,16,16,16,int> c[8];
    load_matrix_sync(a, a_s, 16);
    #pragma unroll
    for (int j = 0; j < 8; j++) fill_fragment(c[j], 0);
    for (int i = 0; i < ITERS; i++) {
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            load_matrix_sync(b, b_s, 16);
            mma_sync(c[j], a, b, c[j]);
        }
    }
    int s = 0;
    #pragma unroll
    for (int j = 0; j < 8; j++) s += c[j].x[0];
    if (threadIdx.x == 0) out[0] = s;
}

template <typename F, typename T>
float run(F kern, T** opp, const char* name) {
    cudaMallocManaged(opp, 4);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    kern<<<1, 32>>>(*opp); cudaDeviceSynchronize();
    cudaEventRecord(e0); kern<<<1, 32>>>(*opp); cudaEventRecord(e1); cudaEventSynchronize(e1);
    float ms; cudaEventElapsedTime(&ms, e0, e1);
    float gmacs = (float)ITERS * 8.0f * 16.0f * 16.0f * 16.0f / 1e9f;
    printf("%-8s: %.3f ms  %.1f GOP-MAC/warp  (%.1f G-tiles/s)\n", name, ms, gmacs/(ms/1e3f), (float)ITERS*8.0f/(ms/1e3f)/1e6f);
    return 0;
}

int main() {
    int dev; cudaGetDevice(&dev);
    cudaDeviceProp p; cudaGetDeviceProperties(&p, dev);
    printf("GPU: %s sm=%d.%d\n", p.name, p.major, p.minor);
    float* fo; int* io;
    run(bench_fp16, &fo, "fp16 ld+mma");
    run(bench_int8, &io, "int8 ld+mma");
    return 0;
}
