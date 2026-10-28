// Native fp8 tensor-core MMA (SM89+) primitives for the PSP-KV decode kernels.
//
// This replaces the current fp8->fp16 dequant + fp16 m16n16k16 WMMA with the
// native mma.sync.aligned.m16n8k32 fp8 instruction (fp32 accumulate), so fp8
// K/V are fed to the tensor cores directly (no dequant pass, half the shared-
// memory traffic).
//
// The PTX and fragment sizes are taken verbatim from CUTLASS
// cutlass/arch/mma_sm89.h (Mma<GemmShape<16,8,32>, 32, float_e4m3_t, RowMajor,
// float_e4m3_t, ColumnMajor, float, RowMajor>).
//
// Fragment layouts below were verified against a CPU reference with exact fp8
// values (scripts/_fp8_ldm_verify.cu, max_err = 0). NOTE: an earlier revision
// used a consecutive-column layout (8t+i) that only passed because the test
// pattern was periodic; random data exposed it as wrong. The correct layout is
// INTERLEAVED (4t+0..3 and 16+4t+0..3), which is also what
// ldmatrix.m16n16.x2.trans.b8 produces directly.
#pragma once

#include <cstdint>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

// fp8 e4m3 is one byte; a 32-bit register holds 4 of them. The A operand is a
// 16x32 row-major tile -> 4 u32 registers/thread. The B operand is a 32x8
// column-major tile -> 2 u32 registers/thread. The C/D accumulator is a 16x8
// fp32 tile -> 4 floats/thread.
__device__ __forceinline__ void psp_fp8_mma_m16n8k32(
    float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2],
    const float (&c)[4]) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
      : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
        "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}

// Same instruction but for fp8 e5m2 K (the "B" operand is e5m2; A is e4m3).
__device__ __forceinline__ void psp_fp8_mma_m16n8k32_e5m2b(
    float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2],
    const float (&c)[4]) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e5m2.f32 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
      : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
        "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}

// ============================================================================
// Verified fragment layouts (max_err = 0 with exact fp8 values).
//
// Warp lanes: groupID = lane>>2 (0..7), threadID_in_group = lane&3 (0..3),
// written g and t below.
//
// A (16x32 row-major fp8), 4 u32/thread. Rows {g, g+8}; 8 INTERLEAVED columns
// per thread {4t..4t+3} and {16+4t..16+4t+3}:
//   a0 = row g   cols 4t+0..3
//   a1 = row g+8 cols 4t+0..3
//   a2 = row g   cols 16+4t+0..3
//   a3 = row g+8 cols 16+4t+0..3
// `ld_k` is the row stride of the A matrix in bytes (32 for an isolated tile,
// or HEAD_DIM for a head-major Q slice).
//
// B (32x8 col-major fp8), 2 u32/thread. B element i = B[k][n] with 8
// INTERLEAVED k-values per thread {4t..4t+3} and {16+4t..16+4t+3}:
//   b0 = B[k=4t+0..3][n=g]
//   b1 = B[k=16+4t+0..3][n=g]
// In col-major storage B[n][k] = n*ld_k + k, that is
//   b0 = base[g*ld_k + 4t + 0..3]
//   b1 = base[g*ld_k + 16 + 4t + 0..3]
//
// C/D accumulator (16x8 fp32), 4 floats/thread:
//   d0 = C[g][2t], d1 = C[g][2t+1], d2 = C[g+8][2t], d3 = C[g+8][2t+1]
// ============================================================================

__device__ __forceinline__ void psp_fp8_load_a(uint32_t (&a)[4],
                                               const uint8_t* A, int lane,
                                               int ld_k) {
  const int g = lane >> 2, t = lane & 3;
  const uint8_t* p0 = A + g * ld_k + t * 4;
  const uint8_t* p1 = p0 + 8 * ld_k;
  a[0] = (uint32_t)p0[0] | ((uint32_t)p0[1] << 8) | ((uint32_t)p0[2] << 16) | ((uint32_t)p0[3] << 24);
  a[1] = (uint32_t)p1[0] | ((uint32_t)p1[1] << 8) | ((uint32_t)p1[2] << 16) | ((uint32_t)p1[3] << 24);
  a[2] = (uint32_t)p0[16] | ((uint32_t)p0[17] << 8) | ((uint32_t)p0[18] << 16) | ((uint32_t)p0[19] << 24);
  a[3] = (uint32_t)p1[16] | ((uint32_t)p1[17] << 8) | ((uint32_t)p1[18] << 16) | ((uint32_t)p1[19] << 24);
}

__device__ __forceinline__ void psp_fp8_load_b(uint32_t (&b)[2],
                                               const uint8_t* B, int lane,
                                               int ld_k) {
  const int g = lane >> 2, t = lane & 3;
  const uint8_t* p = B + g * ld_k + t * 4;
  b[0] = (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
  b[1] = (uint32_t)p[16] | ((uint32_t)p[17] << 8) | ((uint32_t)p[18] << 16) | ((uint32_t)p[19] << 24);
}

// C/D accumulator (16x8 fp32), 4 floats/thread:
//   d0 = C[g][2t], d1 = C[g][2t+1], d2 = C[g+8][2t], d3 = C[g+8][2t+1]
__device__ __forceinline__ void psp_fp8_store_c(const float (&d)[4], float* C,
                                                int lane, int ld_n) {
  const int g = lane >> 2, t = lane & 3;
  C[g * ld_n + 2 * t + 0] = d[0];
  C[g * ld_n + 2 * t + 1] = d[1];
  C[(g + 8) * ld_n + 2 * t + 0] = d[2];
  C[(g + 8) * ld_n + 2 * t + 1] = d[3];
}

// ============================================================================
// Fast path: load A via ldmatrix.m16n16.x2.trans.b8 (SM100a/SM120a only).
//
// Loads a [32x16] source (A transposed: AT[k*16 + m] = A[m][k]) and produces
// the [16x32] A operand directly, matching the interleaved layout above with
// no scalar byte loads. Each lane provides the address of its row (16 bytes =
// 16 fp8). REQUIRES a sm_120a (or sm_100a) build; sm_120 rejects the `.m16n16`
// shape. There is no non-trans form of `m16n16.b8`.
// ============================================================================
__device__ __forceinline__ void psp_fp8_ldmatrix_a(uint32_t (&a)[4],
                                                   const uint8_t* AT, int lane) {
  uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(AT + lane * 16));
  asm volatile(
      "ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8 {%0,%1,%2,%3}, [%4];\n"
      : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
      : "r"(addr));
}
