// Verify fp8 m16n8k32 via CUTLASS MmaTensorOp + tile iterator (ground truth).
#include <cutlass/cutlass.h>
#include <cutlass/gemm/warp/mma_tensor_op.h>
#include <cutlass/gemm/warp/default_mma_tensor_op.h>
#include <cutlass/gemm/warp/mma_tensor_op_tile_iterator.h>
#include <cutlass/tensor_ref.h>
#include <cutlass/layout/tensor_op_multiplicand_sm80.h>
#include <cstdio>
#include <cstdint>
#include <cstring>

using namespace cutlass;
using LayoutA = cutlass::layout::TensorOpMultiplicandCongruous<8, 32>;
using LayoutB = cutlass::layout::TensorOpMultiplicandCongruous<8, 32>;

using Mma = cutlass::gemm::warp::DefaultMmaTensorOp<
    cutlass::gemm::GemmShape<16, 8, 32>, cutlass::gemm::GemmShape<16, 8, 32>,
    cutlass::float_e4m3_t, cutlass::layout::RowMajor,
    cutlass::float_e4m3_t, cutlass::layout::ColumnMajor,
    float, cutlass::layout::RowMajor>::Type;

using AIter = cutlass::gemm::warp::MmaTensorOpMultiplicandTileIterator<
    cutlass::gemm::GemmShape<16, 32, 32>, cutlass::gemm::Operand::kA,
    cutlass::float_e4m3_t, LayoutA, cutlass::gemm::GemmShape<16, 8, 32>,
    1, 32, Mma::kPartitionsK>;

using BIter = cutlass::gemm::warp::MmaTensorOpMultiplicandTileIterator<
    cutlass::gemm::GemmShape<32, 8, 32>, cutlass::gemm::Operand::kB,
    cutlass::float_e4m3_t, LayoutB, cutlass::gemm::GemmShape<16, 8, 32>,
    1, 32, Mma::kPartitionsK>;

// A: 16x32 row-major, B: 32x8 col-major (B[n][k], k contiguous). Both stride 32.
__global__ void test(const float_e4m3_t* gA, const float_e4m3_t* gB, float* C) {
  const int lane = threadIdx.x & 31;
  __shared__ float_e4m3_t shA[16 * 32];
  __shared__ float_e4m3_t shB[8 * 32];
  const int i = threadIdx.x;
  for (int x = i; x < 16 * 32; x += 32) shA[x] = gA[x];
  for (int x = i; x < 8 * 32; x += 32) shB[x] = gB[x];
  __syncthreads();

  TensorRef<float_e4m3_t, LayoutA> refA(shA, LayoutA(32));
  TensorRef<float_e4m3_t, LayoutB> refB(shB, LayoutB(32));
  AIter a_iter(refA, lane);
  BIter b_iter(refB, lane);
  Mma::FragmentA a; Mma::FragmentB b; Mma::FragmentC c;
  a_iter.load(a); b_iter.load(b);
  for (int j = 0; j < 4; j++) c[j] = 0.0f;
  Mma mma; mma(c, a, b, c);

  const int g = lane >> 2, t = lane & 3;
  C[g * 8 + t * 2 + 0] = c[0];
  C[g * 8 + t * 2 + 1] = c[1];
  C[(g + 8) * 8 + t * 2 + 0] = c[2];
  C[(g + 8) * 8 + t * 2 + 1] = c[3];
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
  for (int i = 0; i < M; i++)
    for (int j = 0; j < N; j++) { float s = 0; for (int k = 0; k < K; k++) s += Ah[i*K+k]*Bh[k*N+j]; Ch[i*N+j] = s; }

  float_e4m3_t Aq[M * K], Bq[N * K];  // B stored col-major: B[n][k]
  for (int i = 0; i < M * K; i++) Aq[i] = float_e4m3_t(hf2e4(Ah[i]));
  for (int n = 0; n < N; n++) for (int k = 0; k < K; k++) Bq[n*K+k] = float_e4m3_t(hf2e4(Bh[k*N+n]));

  float_e4m3_t *dA, *dB; float *dC;
  cudaMalloc(&dA, M*K); cudaMalloc(&dB, N*K); cudaMalloc(&dC, M*N*4);
  cudaMemcpy(dA, Aq, M*K, cudaMemcpyHostToDevice);
  cudaMemcpy(dB, Bq, N*K, cudaMemcpyHostToDevice);
  test<<<1, 32>>>(dA, dB, dC);
  float Cout[M*N]; cudaMemcpy(Cout, dC, M*N*4, cudaMemcpyDeviceToHost);
  float max_err = 0;
  for (int i = 0; i < M; i++) for (int j = 0; j < N; j++) max_err = fmaxf(max_err, fabsf(Cout[i*N+j]-Ch[i*N+j]));
  printf("max_err = %g\n", max_err);
  return max_err > 0.5f ? 1 : 0;
}
