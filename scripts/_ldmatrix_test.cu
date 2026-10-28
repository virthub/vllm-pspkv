// Empirically decode ldmatrix.m16n16.x2.trans.b8 fragment layout.
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

__host__ __device__ __forceinline__ float e4m32f(uint8_t q) {
  __nv_fp8_e4m3 f = *reinterpret_cast<const __nv_fp8_e4m3*>(&q);
  return (float)f;
}

// mode 0 -> fill col (c), mode 1 -> fill row (r)
__global__ void k(uint32_t* out, int mode) {
  __shared__ uint8_t smem[32 * 16];
  const int lane = threadIdx.x;
  for (int i = lane; i < 32 * 16; i += 32) {
    int r = i / 16, c = i % 16;
    smem[i] = f2e4m3((mode == 0) ? (float)c : (float)r);
  }
  __syncthreads();
  uint32_t addr = (uint32_t)__cvta_generic_to_shared(smem + lane * 16);
  uint32_t d0, d1, d2, d3;
  asm volatile("ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(d0), "=r"(d1), "=r"(d2), "=r"(d3) : "r"(addr));
  out[lane * 4 + 0] = d0;
  out[lane * 4 + 1] = d1;
  out[lane * 4 + 2] = d2;
  out[lane * 4 + 3] = d3;
}

int main() {
  uint32_t *d, *h;
  cudaMalloc(&d, 32 * 4 * 4);
  h = (uint32_t*)malloc(32 * 4 * 4);
  for (int mode = 0; mode < 2; mode++) {
    k<<<1, 32>>>(d, mode);
    cudaMemcpy(h, d, 32 * 4 * 4, cudaMemcpyDeviceToHost);
    printf("mode %d (%s):\n", mode, mode == 0 ? "col" : "row");
    for (int lane = 0; lane < 32; lane++) {
      printf("  lane %2d: ", lane);
      for (int reg = 0; reg < 4; reg++) {
        uint32_t v = h[lane * 4 + reg];
        printf("[");
        for (int b = 0; b < 4; b++) {
          uint8_t q = (v >> (8 * b)) & 0xff;
          printf("%.0f ", e4m32f(q));
        }
        printf("] ");
      }
      printf("\n");
    }
  }
  return 0;
}
