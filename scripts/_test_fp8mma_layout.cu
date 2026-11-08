// fp8 mma.sync.kind::f8f6f4.m16n8k32 fragment layout (SM120).
// Standard 32-K sum. Decoded layout:
//   A: M = r%2 + 2*(lane/4);  K = 4*(r/2) + i + 8*(r%2) + 16*(lane/16)
//   B: N = lane/4;            K = 4*r + i + 8*(lane%4)
//   C: row0 = 2*(lane/4), col0 = 2*(lane%4), 2x2 block
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cstdio>
#include <cstdint>
#include <cmath>

__host__ __device__ __forceinline__ uint8_t quant_fp8(float v){ __nv_fp8_e4m3 f=(__nv_fp8_e4m3)v; return *reinterpret_cast<uint8_t*>(&f); }
__host__ __device__ __forceinline__ float dequant_fp8(uint8_t q){ __nv_fp8_e4m3 f; *reinterpret_cast<uint8_t*>(&f)=q; return (float)f; }
__device__ __forceinline__ void fp8_mma(float c[4], const uint32_t a[4], const uint32_t b[2]) {
    asm volatile("mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__global__ void test_kernel(const uint8_t* A, const uint8_t* B, float* C) {
    const int lane = threadIdx.x & 31;
    uint32_t a[4];
    #pragma unroll
    for (int r = 0; r < 4; r++) {
        int mm = (r % 2) + 2 * (lane / 4);
        int kk = 4 * (r / 2) + 8 * (r % 2) + 16 * (lane / 16);
        uint32_t reg = 0;
        #pragma unroll
        for (int i = 0; i < 4; i++) reg |= (uint32_t)A[mm * 32 + kk + i] << (8 * i);
        a[r] = reg;
    }
    int nn = lane / 4;
    uint32_t b[2];
    #pragma unroll
    for (int r = 0; r < 2; r++) {
        int kk = 4 * r + 8 * (lane % 4);
        uint32_t reg = 0;
        #pragma unroll
        for (int i = 0; i < 4; i++) reg |= (uint32_t)B[nn * 32 + kk + i] << (8 * i);
        b[r] = reg;
    }
    float c[4] = {0,0,0,0};
    fp8_mma(c, a, b);
    int row0 = 2 * (lane / 4), col0 = 2 * (lane % 4);
    C[(row0+0)*8 + col0+0] = c[0];
    C[(row0+0)*8 + col0+1] = c[1];
    C[(row0+1)*8 + col0+0] = c[2];
    C[(row0+1)*8 + col0+1] = c[3];
}

int main() {
    const int M = 16, N = 8, K = 32;
    float hA[M*K], hB[K*N];
    for (int i = 0; i < M*K; i++) hA[i] = (float)((i*7)%23 - 11) / 8.0f;
    for (int i = 0; i < K*N; i++) hB[i] = (float)((i*5)%17 - 8) / 8.0f;
    uint8_t hAq[M*K], hBq[K*N];
    for (int m = 0; m < M; m++) for (int k = 0; k < K; k++) hAq[m*K+k] = quant_fp8(hA[m*K+k]);
    for (int k = 0; k < K; k++) for (int n = 0; n < N; n++) hBq[n*K+k] = quant_fp8(hB[k*N+n]);
    float ref[M*N];
    for (int m = 0; m < M; m++) for (int n = 0; n < N; n++) {
        float s = 0; for (int k = 0; k < K; k++) s += dequant_fp8(hAq[m*K+k]) * dequant_fp8(hBq[n*K+k]);
        ref[m*N+n] = s;
    }
    uint8_t *dA, *dB; float *dC;
    cudaMalloc(&dA, M*K); cudaMalloc(&dB, K*N); cudaMalloc(&dC, M*N*4);
    cudaMemcpy(dA, hAq, M*K, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hBq, K*N, cudaMemcpyHostToDevice);
    cudaMemset(dC, 0, M*N*4);
    test_kernel<<<1, 32>>>(dA, dB, dC);
    float hC[M*N]; cudaMemcpy(hC, dC, M*N*4, cudaMemcpyDeviceToHost);
    float mx = 0; int bad = 0;
    for (int m = 0; m < M; m++) for (int n = 0; n < N; n++) {
        float e = fabsf(hC[m*N+n] - ref[m*N+n]);
        if (e > mx) mx = e;
        if (e > 0.5f) { bad++; if (bad <= 4) printf("  C[%d][%d]=%.2f ref %.2f\n", m, n, hC[m*N+n], ref[m*N+n]); }
    }
    printf("maxerr=%.4f bad=%d\n", mx, bad);
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return bad ? 1 : 0;
}
