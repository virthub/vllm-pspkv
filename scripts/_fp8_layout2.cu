// Verify fp8 m16n8k32 fragment layout (corrected per PTX ISA: A cols = 8t+i).
#include <cuda_fp8.h>
#include <cstdio>
#include <cstdint>
#include <cstring>

__device__ __forceinline__ uint8_t f2e4m3(float x) {
  if (x == 0.0f) return 0;
  uint32_t u = __float_as_uint(x);
  uint32_t s = (u >> 31) & 1; int e = ((u >> 23) & 0xff) - 127; uint32_t m = u & 0x7fffff;
  if (e < -6) return (uint8_t)(s << 7);
  if (e > 7) return (uint8_t)((s << 7) | 0x7e);
  uint32_t mm = (m | 0x800000) >> (23 - 3);
  return (uint8_t)((s << 7) | ((e + 7) << 3) | (mm & 0x7));
}

__device__ __forceinline__ void mma_fp8(float d[4], const uint32_t a[4],
                                       const uint32_t b[2], const float c[4]) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
      : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
        "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}

// A: 16x32 row-major. g=lane>>2, t=lane&3. Each thread holds rows {g,g+8} and
// 8 cols {8t..8t+7}. Registers: a0 = row g cols 8t+0..3, a1 = row g+8 cols
// 8t+0..3, a2 = row g cols 8t+4..7, a3 = row g+8 cols 8t+4..7.
__device__ __forceinline__ void load_a(uint32_t a[4], const uint8_t* A, int lane) {
  const int g = lane >> 2, t = lane & 3;
  const uint8_t* p0 = A + g * 32 + t * 8;
  const uint8_t* p1 = p0 + 8 * 32;
  a[0] = (uint32_t)p0[0] | ((uint32_t)p0[1] << 8) | ((uint32_t)p0[2] << 16) | ((uint32_t)p0[3] << 24);
  a[1] = (uint32_t)p1[0] | ((uint32_t)p1[1] << 8) | ((uint32_t)p1[2] << 16) | ((uint32_t)p1[3] << 24);
  a[2] = (uint32_t)p0[4] | ((uint32_t)p0[5] << 8) | ((uint32_t)p0[6] << 16) | ((uint32_t)p0[7] << 24);
  a[3] = (uint32_t)p1[4] | ((uint32_t)p1[5] << 8) | ((uint32_t)p1[6] << 16) | ((uint32_t)p1[7] << 24);
}

// B: 32x8 col-major, stored B[n][k] (n*32 + k). B element i = B[k=8t+i][n=g].
__device__ __forceinline__ void load_b(uint32_t b[2], const uint8_t* B, int lane) {
  const int g = lane >> 2, t = lane & 3;
  const uint8_t* p = B + g * 32 + t * 8;
  b[0] = (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
  b[1] = (uint32_t)p[4] | ((uint32_t)p[5] << 8) | ((uint32_t)p[6] << 16) | ((uint32_t)p[7] << 24);
}

__global__ void k(const uint8_t* A, const uint8_t* B, float* C) {
  const int lane = threadIdx.x & 31;
  uint32_t a[4], b[2]; float c[4] = {0,0,0,0}, d[4];
  load_a(a, A, lane);
  load_b(b, B, lane);
  mma_fp8(d, a, b, c);
  const int g = lane >> 2, t = lane & 3;
  C[g * 8 + t * 2 + 0] = d[0];
  C[g * 8 + t * 2 + 1] = d[1];
  C[(g + 8) * 8 + t * 2 + 0] = d[2];
  C[(g + 8) * 8 + t * 2 + 1] = d[3];
}

int main() {
  const int M = 16, K = 32, N = 8;
  float Ah[M * K], Bh[K * N], Ch[M * N];
  auto hf2e4 = [](float x) -> uint8_t {
    if (x == 0.0f) return 0;
    uint32_t u; memcpy(&u, &x, 4);
    uint32_t s = (u >> 31) & 1; int e = ((u >> 23) & 0xff) - 127; uint32_t m = u & 0x7fffff;
    if (e < -6) return (uint8_t)(s << 7);
    if (e > 7) return (uint8_t)((s << 7) | 0x7e);
    uint32_t mm = (m | 0x800000) >> (23 - 3);
    return (uint8_t)((s << 7) | ((e + 7) << 3) | (mm & 0x7));
  };
  for (int i = 0; i < M * K; i++) Ah[i] = (i % 7) * 0.5f - 1.5f;
  for (int i = 0; i < K * N; i++) Bh[i] = (i % 5) * 0.25f - 0.5f;
  for (int i = 0; i < M; i++) for (int j = 0; j < N; j++) {
    float s = 0; for (int kk = 0; kk < K; kk++) s += Ah[i*K+kk]*Bh[kk*N+j]; Ch[i*N+j] = s; }

  uint8_t Aq[M * K], Bq[N * K];
  for (int i = 0; i < M * K; i++) Aq[i] = hf2e4(Ah[i]);
  for (int n = 0; n < N; n++) for (int kk = 0; kk < K; kk++) Bq[n*K+kk] = hf2e4(Bh[kk*N+n]);

  uint8_t *dA, *dB; float *dC;
  cudaMalloc(&dA, M*K); cudaMalloc(&dB, N*K); cudaMalloc(&dC, M*N*4);
  cudaMemcpy(dA, Aq, M*K, cudaMemcpyHostToDevice);
  cudaMemcpy(dB, Bq, N*K, cudaMemcpyHostToDevice);
  k<<<1,32>>>(dA, dB, dC);
  float Cout[M*N]; cudaMemcpy(Cout, dC, M*N*4, cudaMemcpyDeviceToHost);
  float me = 0;
  for (int i = 0; i < M*N; i++) me = fmaxf(me, fabsf(Cout[i]-Ch[i]));
  printf("max_err = %g\n", me);
  return me > 0.5f ? 1 : 0;
}
