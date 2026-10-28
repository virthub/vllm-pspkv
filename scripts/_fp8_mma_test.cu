// Standalone verification of the fp8 m16n8k32 fragment load layout.
// Compile: nvcc -arch=sm_90 -o /tmp/fp8mma _fp8_mma_test.cu && /tmp/fp8mma
#include <cuda_fp8.h>
#include <cstdio>
#include <cstdint>

__device__ __forceinline__ uint8_t f2e4m3(float x) {
  // simple round-to-nearest float -> e4m3 (sign 1, exp 4, mant 3)
  if (x == 0.0f) return 0;
  uint32_t u = __float_as_uint(x);
  uint32_t s = (u >> 31) & 1;
  int e = ((u >> 23) & 0xff) - 127;
  uint32_t m = u & 0x7fffff;
  // clamp exponent to e4m3 range [-6, 7]
  if (e < -6) return s << 7;
  if (e > 7) return (s << 7) | 0x7e;  // max
  // add implicit bit, shift to 3 mantissa bits with rounding
  uint32_t mm = (m | 0x800000) >> (23 - 3);
  int exp = e + 7;
  return (uint8_t)((s << 7) | (exp << 3) | (mm & 0x7));
}

__device__ __forceinline__ float e4m3_2f(uint8_t q) {
  int s = (q >> 7) & 1, e = (q >> 3) & 0xf, m = q & 0x7;
  float v;
  if (e == 0) v = ldexpf((float)m, -6);
  else v = ldexpf((float)(m + 8), e - 7 - 3);
  return s ? -v : v;
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

// Candidate A layout (16x32 row-major). g = lane>>2 (0..7), t = lane&3 (0..3).
__device__ __forceinline__ void load_a(uint32_t a[4], const uint8_t* A, int lane) {
  const int g = lane >> 2, t = (lane & 3) * 2;
  const uint8_t* p0 = A + g * 32 + t;        // row g
  const uint8_t* p1 = p0 + 8 * 32;           // row g+8
  // a0 = {A[g][t], A[g][t+1], A[g+8][t], A[g+8][t+1]}
  a[0] = (uint32_t)p0[0] | ((uint32_t)p0[1] << 8) | ((uint32_t)p1[0] << 16) | ((uint32_t)p1[1] << 24);
  a[1] = (uint32_t)p0[8] | ((uint32_t)p0[9] << 8) | ((uint32_t)p1[8] << 16) | ((uint32_t)p1[9] << 24);
  a[2] = (uint32_t)p0[16] | ((uint32_t)p0[17] << 8) | ((uint32_t)p1[16] << 16) | ((uint32_t)p1[17] << 24);
  a[3] = (uint32_t)p0[24] | ((uint32_t)p0[25] << 8) | ((uint32_t)p1[24] << 16) | ((uint32_t)p1[25] << 24);
}

// Candidate B layout (32x8 col-major). B is stored as B[k][n] (k contiguous).
// col-major: B[n][k] -> memory index n*32 + k.
__device__ __forceinline__ void load_b(uint32_t b[2], const uint8_t* B, int lane) {
  const int g = lane >> 2, t = lane & 3;
  // B is 32x8; the mma "col" layout means B^T is consumed. Candidate mapping:
  // b0 = {B[t*2][g], B[t*2+1][g], B[t*2+8][g], B[t*2+9][g]}? -- see below
  // Use row-major storage B[k][n] with k contiguous (k in [0,32), n in [0,8)).
  const uint8_t* p = B + (t * 2) * 8 + g;  // B[k=t*2][n=g]
  // 8 elements per thread: k in {2t,2t+1} and n in {g}, plus k+8, k+16, k+24
  const uint8_t* q0 = B + (t * 2) * 8 + g;
  const uint8_t* q1 = B + (t * 2 + 8) * 8 + g;
  const uint8_t* q2 = B + (t * 2 + 16) * 8 + g;
  const uint8_t* q3 = B + (t * 2 + 24) * 8 + g;
  b[0] = (uint32_t)q0[0] | ((uint32_t)q0[1] << 8) | ((uint32_t)q1[0] << 16) | ((uint32_t)q1[1] << 24);
  b[1] = (uint32_t)q2[0] | ((uint32_t)q2[1] << 8) | ((uint32_t)q3[0] << 16) | ((uint32_t)q3[1] << 24);
}

__global__ void test_kernel(const uint8_t* A, const uint8_t* B, float* C) {
  const int lane = threadIdx.x & 31;
  uint32_t a[4], b[2];
  float c[4] = {0, 0, 0, 0}, d[4];
  load_a(a, A, lane);
  load_b(b, B, lane);
  mma_fp8(d, a, b, c);
  // write C fragment (16x8 row-major): each thread holds 4 floats.
  const int g = lane >> 2, t = lane & 3;
  // C fragment layout for m16n8k32: row = g, col = t*2 + {0,1}, plus row g+8
  C[g * 8 + t * 2 + 0] = d[0];
  C[g * 8 + t * 2 + 1] = d[1];
  C[(g + 8) * 8 + t * 2 + 0] = d[2];
  C[(g + 8) * 8 + t * 2 + 1] = d[3];
}

int main() {
  const int M = 16, K = 32, N = 8;
  float Ah[M * K], Bh[K * N], Ch[M * N];
  for (int i = 0; i < M * K; i++) Ah[i] = (i % 7) * 0.5f - 1.0f;
  for (int i = 0; i < K * N; i++) Bh[i] = (i % 5) * 0.25f - 0.5f;
  // CPU reference: C = A @ B (A is MxK row-major, B is KxN row-major here)
  for (int i = 0; i < M; i++)
    for (int j = 0; j < N; j++) {
      float s = 0;
      for (int k = 0; k < K; k++) s += Ah[i * K + k] * Bh[k * N + j];
      Ch[i * N + j] = s;
    }
  // quantize to fp8 on host
  uint8_t Aq[M * K], Bq[K * N];
  for (int i = 0; i < M * K; i++) Aq[i] = (uint8_t)(i % 7 ? (i % 7) : 1);  // placeholder replaced below
  // NOTE: host-side f2e4m3 requires __device__; do a simple host quant
  // Recompute host quant with host float->e4m3 (duplicate of device logic)
  auto hf2e4 = [](float x) -> uint8_t {
    if (x == 0.0f) return 0;
    uint32_t u; memcpy(&u, &x, 4);
    uint32_t s = (u >> 31) & 1;
    int e = ((u >> 23) & 0xff) - 127;
    uint32_t m = u & 0x7fffff;
    if (e < -6) return (uint8_t)(s << 7);
    if (e > 7) return (uint8_t)((s << 7) | 0x7e);
    uint32_t mm = (m | 0x800000) >> (23 - 3);
    return (uint8_t)((s << 7) | ((e + 7) << 3) | (mm & 0x7));
  };
  for (int i = 0; i < M * K; i++) Aq[i] = hf2e4(Ah[i]);
  for (int i = 0; i < K * N; i++) Bq[i] = hf2e4(Bh[i]);

  uint8_t *dA, *dB; float *dC;
  cudaMalloc(&dA, M * K); cudaMalloc(&dB, K * N); cudaMalloc(&dC, M * N);
  cudaMemcpy(dA, Aq, M * K, cudaMemcpyHostToDevice);
  cudaMemcpy(dB, Bq, K * N, cudaMemcpyHostToDevice);
  test_kernel<<<1, 32>>>(dA, dB, dC);
  float Cout[M * N];
  cudaMemcpy(Cout, dC, M * N * 4, cudaMemcpyDeviceToHost);
  float max_err = 0;
  for (int i = 0; i < M; i++)
    for (int j = 0; j < N; j++)
      max_err = fmaxf(max_err, fabsf(Cout[i * N + j] - Ch[i * N + j]));
  printf("max_err = %g\n", max_err);
  return max_err > 0.5f ? 1 : 0;
}
