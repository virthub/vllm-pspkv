// Standalone MMA throughput microbenchmark for SM120 (RTX 5090).
#include <cuda_runtime.h>
#include <mma.h>
#include <cstdio>
#include <stdint.h>
using namespace nvcuda::wmma;

#define ITERS 400000

__global__ void bench_fp16(float* out) {
    fragment<matrix_a,16,16,16,__half,row_major> a;
    fragment<matrix_b,16,16,16,__half,col_major> b;
    fragment<accumulator,16,16,16,float> c;
    fill_fragment(a, __float2half(0.5f));
    fill_fragment(b, __float2half(0.5f));
    fill_fragment(c, 0.0f);
    for (int i = 0; i < ITERS; i++) mma_sync(c, a, b, c);
    if (threadIdx.x == 0) out[0] = c.x[0];
}

__global__ void bench_int8(int* out) {
    fragment<matrix_a,16,16,16,signed char,row_major> a;
    fragment<matrix_b,16,16,16,signed char,col_major> b;
    fragment<accumulator,16,16,16,int> c;
    fill_fragment(a, (signed char)1);
    fill_fragment(b, (signed char)1);
    fill_fragment(c, 0);
    for (int i = 0; i < ITERS; i++) mma_sync(c, a, b, c);
    if (threadIdx.x == 0) out[0] = c.x[0];
}

// [int4 removed: unsupported on sm_120a]
#if 0
// int4 mma.sync.aligned.m16n16k32.row.col.s32.s4.s4.s32 via inline PTX.
__global__ void bench_int4(int* out) {
    uint32_t a0 = 0x11111111u, a1 = 0x11111111u, a2 = 0x11111111u, a3 = 0x11111111u;
    uint32_t b0 = 0x11111111u, b1 = 0x11111111u;
    uint32_t c0 = 0, c1 = 0, c2 = 0, c3 = 0;
    for (int i = 0; i < ITERS; i++) {
        asm volatile(
            "mma.sync.aligned.m16n16k32.row.col.s32.s4.s4.s32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+r"(c0), "+r"(c1), "+r"(c2), "+r"(c3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
    }
    if (threadIdx.x == 0) out[0] = (int)c0;
}
#endif

static void* out;
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
    float gmacs = (float)ITERS * 16.0f * 16.0f * 16.0f / 1e9f;
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
    run(bench_int8, &iout, "int8");
    return 0;
}
