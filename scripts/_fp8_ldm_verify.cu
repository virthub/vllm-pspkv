// Definitive fp8 m16n8k32 layout verification: ldmatrix.m16n16.x2.trans.b8 for
// A + FlashInfer-style scalar B, random data, vs CPU reference.
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

// B [32][8] col-major (B[n][k] = n*32 + k). Correct layout (FlashInfer-style):
//   b0 = B[k=tid*4+0..3][n=gid], b1 = B[k=tid*4+16..19][n=gid]
__device__ __forceinline__ void load_b(uint32_t b[2], const uint8_t* B, int lane) {
  const int gid = lane >> 2, tid = lane & 3;
  const uint8_t* p = B + gid * 32 + tid * 4;
  b[0] = (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
  const uint8_t* q = p + 16;
  b[1] = (uint32_t)q[0] | ((uint32_t)q[1] << 8) | ((uint32_t)q[2] << 16) | ((uint32_t)q[3] << 24);
}

__global__ void k(const uint8_t* AT, const uint8_t* B, float* C) {
  const int lane = threadIdx.x;
  __shared__ uint8_t sAT[32 * 16];
  __shared__ uint8_t sB[8 * 32];
  // A^T [32][16] row-major: sAT[k*16+m] = A[m][k]
  for (int i = lane; i < 32 * 16; i += 32) sAT[i] = AT[i];
  for (int i = lane; i < 8 * 32; i += 32) sB[i] = B[i];
  __syncthreads();
  uint32_t a[4], b[2];
  uint32_t addr = (uint32_t)__cvta_generic_to_shared(sAT + lane * 16);
  asm volatile("ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) : "r"(addr));
  load_b(b, sB, lane);
  float c[4] = {0, 0, 0, 0}, d[4];
  mma_fp8(d, a, b, c);
  const int gid = lane >> 2, tid = lane & 3;
  // C [16][8]: c0=C[gid][2tid], c1=C[gid][2tid+1], c2=C[gid+8][2tid], c3=C[gid+8][2tid+1]
  C[gid * 8 + 2 * tid + 0] = d[0];
  C[gid * 8 + 2 * tid + 1] = d[1];
  C[(gid + 8) * 8 + 2 * tid + 0] = d[2];
  C[(gid + 8) * 8 + 2 * tid + 1] = d[3];
}

int main() {
  const int M = 16, K = 32, N = 8;
  float Ah[M * K], Bh[K * N], Ch[M * N];
  srand(42);
  for (int i = 0; i < M * K; i++) Ah[i] = (rand() % 1024) * 0.5f - 256.0f;
  for (int i = 0; i < K * N; i++) Bh[i] = (rand() % 1024) * 0.5f - 256.0f;
  for (int i = 0; i < M; i++) for (int j = 0; j < N; j++) {
    float s = 0; for (int kk = 0; kk < K; kk++) s += Ah[i*K+kk]*Bh[kk*N+j]; Ch[i*N+j] = s; }
  uint8_t Aq[M * K], ATq[K * M], Bq[N * K];
  for (int i = 0; i < M * K; i++) Aq[i] = f2e4m3(Ah[i]);
  for (int kk = 0; kk < K; kk++) for (int m = 0; m < M; m++) ATq[kk*M+m] = Aq[m*K+kk];
  for (int n = 0; n < N; n++) for (int kk = 0; kk < K; kk++) Bq[n*K+kk] = f2e4m3(Bh[kk*N+n]);
  uint8_t *dAT, *dB; float *dC;
  cudaMalloc(&dAT, K*M); cudaMalloc(&dB, N*K); cudaMalloc(&dC, M*N*4);
  cudaMemcpy(dAT, ATq, K*M, cudaMemcpyHostToDevice);
  cudaMemcpy(dB, Bq, N*K, cudaMemcpyHostToDevice);
  k<<<1, 32>>>(dAT, dB, dC);
  float Cout[M*N]; cudaMemcpy(Cout, dC, M*N*4, cudaMemcpyDeviceToHost);
  float me = 0;
  for (int i = 0; i < M*N; i++) me = fmaxf(me, fabsf(Cout[i]-Ch[i]));
  printf("max_err = %g\n", me);
  return me > 50.0f ? 1 : 0;
}
