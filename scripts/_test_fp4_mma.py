"""Minimal SM120 block-scaled fp4 (MXF4) MMA test.

Verifies that `warp.MmaMXF4Op` (the SM120-native block-scaled fp4 atom,
elem_type = f4E2M1FN, sf_type = f8E8M0FNU) compiles and, ideally, runs a
`cute.gemm` with the E8M0 scale-factor operands.

Run (requires a Python 3.12/3.13 env with cutlass-dsl >= 4.5.2 that has the
`ThrMma` API, and GPU access via `ssh localhost`):

    ssh -o StrictHostKeyChecking=no localhost \
      '<env>/bin/python /root/vllm-pspkv/scripts/_test_fp4_mma.py'

Status (2026-11-05):
  - atom compiles: MLIR `!cute_nvgpu.SM120.mma_bs<16x8x64, vec_size=32,
    elem_type=(f4E2M1FN, f4E2M1FN, f32), sf_type=f8E8M0FNU, use_sf_layout_TV=false>`.
  - block-scaled gemm plumbing (`cute.gemm(atom, d, [A, SFA], [B, SFB], c)`)
    dispatches to the C++ backend.
  - BLOCKER: the E8M0 scale-factor fragment layout is unknown. Raw rank-2 SFA
    -> rank-mismatch verify error; partitioning `[M, K//32]` via partition_A
    -> `std::bad_variant_access: Unexpected index` in the C++ backend.
"""
import torch
import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import warp


def main():
    @cute.jit
    def fp4_gemm(mA, mB, mSFA, mSFB, mC):
        tiled_mma = cute.make_tiled_mma(
            cute.make_mma_atom(
                warp.MmaMXF4Op(cutlass.Float4E2M1FN, cutlass.Float32, cutlass.Float8E8M0FNU)
            ),
            (1, 1, 1),
        )
        thr_mma = tiled_mma.get_slice(0)
        mma_atom = cute.make_mma_atom(tiled_mma.op)
        tA = thr_mma.make_fragment_A(thr_mma.partition_A(mA))
        tB = thr_mma.make_fragment_B(thr_mma.partition_B(mB))
        tSFA = thr_mma.make_fragment_A(thr_mma.partition_A(mSFA))
        tSFB = thr_mma.make_fragment_B(thr_mma.partition_B(mSFB))
        tC = thr_mma.partition_C(mC)
        cute.gemm(mma_atom, tC, [tA, tSFA], [tB, tSFB], tC)
        cute.copy(tC, mC)

    M, N, K = 16, 8, 64
    dev = "cuda"
    A = torch.randn(M, K, device=dev).to(torch.float32)
    B = torch.randn(N, K, device=dev).to(torch.float32)
    C = torch.zeros(M, N, device=dev, dtype=torch.float32)
    # TODO: derive the correct E8M0 scale-factor fragment shape/layout
    # (sf_vec_size=32, use_sf_layout_TV=false).
    SFA = torch.randn(M, K // 32, device=dev, dtype=torch.float32)
    SFB = torch.randn(N, K // 32, device=dev, dtype=torch.float32)
    fp4_gemm(A, B, SFA, SFB, C)
    torch.cuda.synchronize()
    print("fp4 gemm ran OK, C:", C.cpu().tolist())


if __name__ == "__main__":
    main()
