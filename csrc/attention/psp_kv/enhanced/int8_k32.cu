// SPDX-License-Identifier: Apache-2.0
// PROTOTYPE (not wired into the engine default path).
//
// Int8 decode with 32-dim scale groups and the raw integer tensor-core
// instruction mma.sync.m16n8k32 (SASS: IMMA.16832.S8.S8 on SM120).
//
// Why: the production int8 kernel (`paged_attention_enhanced_int_tc_4warp_`)
// uses wmma m16n16k16.s8, so the QK accumulator must be converted to fp32,
// scaled and summed once per 16-dim K chunk (head_dim/16 = 8 chunks). With 32
// int8 elements per K step the same reduction needs only head_dim/32 = 4
// rescale steps, and each SASS IMMA covers twice the MACs (4096 vs 2048), so
// both the MMA issue count and the int32->fp32 rescale count halve.
//
// Semantics: out = softmax(scale * q.k) v, with q quantized to int8 per 32-dim
// group by max-abs (q_scale[g] = max_abs/127), k/v int8 with per-(block,
// kv_head, 32-dim group) float32 scales, and the softmax probabilities
// quantized to int8 (7-bit) exactly like the production int8 path.
//
// Structure is the production GQA layout: one CTA (one warp) per
// (kv_head, seq, split), the gqa query heads in the 16-row tile (zero padded),
// fp32 online softmax, split-K partials + merge.
//
// QK uses the raw PTX m16n8k32 (A=Q row-major, B=K^T col-major, which is
// exactly the token-major K tile). PV keeps wmma m16n16k16.s8: integer mma.sync
// only supports `.row.col`, so an m16n8k32 PV would need V transposed in smem
// (see scripts/k32probe/mma_probe.cu). Consequence: the PV rescale count is
// unchanged; only QK gets the 8 -> 4 rescale steps.
//
// TILE=16 tokens, not 32. Measured on the RTX 5090 D: sm_120 exposes only
// 100 KB of shared memory per SM, so with 32-token K/V double buffers (24.8 KB
// per warp) the kernel was capped at 4 CTAs/SM and ran 15% slower than the
// production path despite executing 29% fewer instructions. At TILE=16
// (12.0 KB, 128 registers) it fits 8 warps/SM, matching the 4-warp production
// kernel's occupancy, and the instruction savings translate into wall time.
//
// The quantized Q staging area aliases the first K tile buffer (it is consumed
// into registers before the prologue cp.async), and the score/prob column
// strides are padded (TILE+4 / TILE+16) so the 8 row groups land on distinct
// smem banks: a TILE-sized stride is 8-way bank conflicted on the row-wise
// softmax reads/writes.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

#include "common/quantization_utils.cuh"
#include "common/head_dim_dispatch.cuh"
#include "common/psp_kv_config.cuh"

#include <torch/library.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

using namespace nvcuda;
using namespace nvcuda::wmma;

namespace psp_kv_k32 {

// ---------------------------------------------------------------------------
// Raw int8 mma.sync.m16n8k32
//
// ptxas accepts the integer form only with the ".dtype.atype.btype.ctype"
// ordering and only for .row.col (A row-major, B col-major); ".satfinite" and
// ".col.row" are rejected for this shape. Fragment mapping (verified against a
// CPU reference in scripts/k32probe/mma_probe.cu):
//   A(16x32, 4 x .b32): r0=(row gid, k 4*tid), r1=(row gid+8, k 4*tid),
//                       r2=(row gid, k 16+4*tid), r3=(row gid+8, k 16+4*tid)
//   B(32x8,  2 x .b32): r0=(k 4*tid, col gid), r1=(k 16+4*tid, col gid)
//   D(16x8,  4 x .s32): d0=(gid, 2*tid), d1=(gid, 2*tid+1),
//                       d2=(gid+8, 2*tid), d3=(gid+8, 2*tid+1)
// with gid = lane >> 2, tid = lane & 3.
// ---------------------------------------------------------------------------
__device__ __forceinline__ void mma_k32(int32_t (&d)[4], const uint32_t (&a)[4],
                                        const uint32_t (&b)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10};\n"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
          "r"(0));
}

// Token-major K/V tile -> padded smem tile (rows of `head_dim` bytes, stride
// `smem_stride`). One cp.async 16-byte chunk per (token, 16-dim chunk).
__device__ __forceinline__ void k32_cp_async_tile(signed char* smem,
                                                  const uint8_t* gmem, int lane,
                                                  int tokens, int head_dim,
                                                  int num_kv_heads,
                                                  int smem_stride) {
    const int chunks_per_row = head_dim / 16;
    const int total_chunks = tokens * chunks_per_row;
    #pragma unroll
    for (int i = lane; i < total_chunks; i += 32) {
        const int t = i / chunks_per_row;
        const int c = i - t * chunks_per_row;
        const uint8_t* src = gmem + (size_t)t * num_kv_heads * head_dim + c * 16;
        signed char* dst = smem + (size_t)t * smem_stride + c * 16;
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(
                         (unsigned)__cvta_generic_to_shared(dst)),
                     "l"(src));
    }
    asm volatile("cp.async.commit_group;");
}

template <int HEAD_DIM, typename scalar_t>
__global__ void __launch_bounds__(32) paged_attention_enhanced_int8_k32_kernel(
    float* __restrict__ partials,
    const scalar_t* __restrict__ q,
    const uint8_t* __restrict__ k_cache,
    const uint8_t* __restrict__ v_cache,
    const int* __restrict__ block_tables,
    const int* __restrict__ context_lens,
    const uint32_t* __restrict__ block_offsets,
    const float* __restrict__ scale_table,  // [2, num_blocks, num_kv_heads, HEAD_DIM/32]
    int max_blocks_per_seq,
    int num_heads,
    int num_kv_heads,
    float scale,
    int num_splits,
    int num_total_blocks)
{
    constexpr int TILE = 16;                     // tokens per sub-tile
    constexpr int NK = HEAD_DIM / 32;            // QK reduction steps (group-32)
    constexpr int NT = TILE / 8;                 // QK n-tiles of 8 tokens
    constexpr int NQ8 = HEAD_DIM / 16;           // PV dim chunks / wmma n-tiles
    constexpr int NGRP = HEAD_DIM / 32;          // scale groups
    constexpr int SUB_TILES = BLOCK_SIZE / TILE; // sub-tiles per block
    constexpr int BLOCK_BYTES = BLOCK_SIZE * HEAD_DIM;
    constexpr int SMEM_STRIDE = HEAD_DIM + 16;   // padded row stride (see below)
    // Column strides padded past TILE so the 8 row groups (gid) land on
    // distinct banks: a TILE stride maps all 8 rows of a column onto one bank
    // (8-way conflict on the row-wise softmax reads and the prob writes).
    // SSC_SCORE is float (4-byte) addressed; SSC_PROB must stay a multiple of
    // 16 for the wmma A-fragment load.
    constexpr int SSC_SCORE = TILE + 4;   // 36 floats
    constexpr int SSC_PROB = TILE + 16;   // 48 bytes

    static_assert(HEAD_DIM % 32 == 0, "k32 probe needs head_dim % 32 == 0");
    static_assert(BLOCK_SIZE % TILE == 0, "k32 probe needs block_size % TILE == 0");

    __shared__ __align__(16) signed char sk[2][TILE * SMEM_STRIDE];
    __shared__ __align__(16) signed char sv[2][TILE * SMEM_STRIDE];
    __shared__ __align__(16) float s_score[16 * SSC_SCORE];
    __shared__ __align__(16) signed char s_prob[16 * SSC_PROB];

    // Quantized Q is staged in the first K tile buffer: it is consumed into
    // registers before the loop, and the prologue K cp.async runs afterwards.
    signed char* s_q_int8 = sk[0];

    const int kv_head = blockIdx.x;
    const int seq_idx = blockIdx.y;
    const int split_idx = blockIdx.z;
    const int gqa = num_heads / num_kv_heads;
    const int tid = threadIdx.x;
    const int lane = tid & 31, gid = lane >> 2, tig = lane & 3;
    const float log2e = 1.4426950408889634f;

    fragment<accumulator, 16, 16, 16, float> acc[NQ8];
    fragment<matrix_a, 16, 16, 16, signed char, row_major> pa_i[TILE / 16];
    fragment<matrix_b, 16, 16, 16, signed char, row_major> vb_i;
    fragment<accumulator, 16, 16, 16, int> c_pv[2];

    // --- Q quantize: per-32-dim-group max-abs scale, rows [0, gqa) valid ------
    for (int i = lane; i < 16 * SMEM_STRIDE; i += 32) s_q_int8[i] = 0;
    const int head_base = kv_head * gqa;
    float q_scale[NGRP];
    #pragma unroll
    for (int g = 0; g < NGRP; g++) q_scale[g] = 0.0f;
    for (int i = lane; i < gqa * HEAD_DIM; i += 32) {
        const int h = head_base + i / HEAD_DIM;
        const int d = i % HEAD_DIM;
        const int g = d / 32;
        const float qv = psp_to_float<scalar_t>(
            q[(int64_t)seq_idx * num_heads * HEAD_DIM + (int64_t)h * HEAD_DIM + d]);
        q_scale[g] = fmaxf(q_scale[g], fabsf(qv));
    }
    #pragma unroll
    for (int g = 0; g < NGRP; g++) {
        #pragma unroll
        for (int o = 16; o > 0; o >>= 1)
            q_scale[g] = fmaxf(q_scale[g], __shfl_xor_sync(0xffffffffu, q_scale[g], o));
        q_scale[g] /= 127.0f;
        if (q_scale[g] < 1e-8f) q_scale[g] = 1.0f;
    }
    for (int i = lane; i < gqa * HEAD_DIM; i += 32) {
        const int row = i / HEAD_DIM;
        const int h = head_base + row;
        const int d = i % HEAD_DIM;
        const int g = d / 32;
        const float qv = psp_to_float<scalar_t>(
            q[(int64_t)seq_idx * num_heads * HEAD_DIM + (int64_t)h * HEAD_DIM + d]);
        s_q_int8[row * SMEM_STRIDE + d] = quantize_int8(qv / q_scale[g]);
    }
    __syncwarp();

    // Q fragments: A(16x32), one per 32-dim K step, register resident.
    uint32_t a_q[NK][4];
    #pragma unroll
    for (int kt = 0; kt < NK; kt++) {
        const int k0 = kt * 32 + tig * 4;
        a_q[kt][0] = *reinterpret_cast<const uint32_t*>(s_q_int8 + gid * SMEM_STRIDE + k0);
        a_q[kt][1] = *reinterpret_cast<const uint32_t*>(s_q_int8 + (gid + 8) * SMEM_STRIDE + k0);
        a_q[kt][2] = *reinterpret_cast<const uint32_t*>(s_q_int8 + gid * SMEM_STRIDE + k0 + 16);
        a_q[kt][3] = *reinterpret_cast<const uint32_t*>(s_q_int8 + (gid + 8) * SMEM_STRIDE + k0 + 16);
    }
    #pragma unroll
    for (int nt8 = 0; nt8 < NQ8; nt8++) fill_fragment(acc[nt8], 0.0f);

    float rmax0 = -1e30f, rsum0 = 0.0f;   // row gid
    float rmax1 = -1e30f, rsum1 = 0.0f;   // row gid + 8

    const int context_len = context_lens[seq_idx];
    const int num_blocks = (context_len + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int blocks_per_split = (num_blocks + num_splits - 1) / num_splits;
    const int split_start = split_idx * blocks_per_split;
    const int split_end = min(split_start + blocks_per_split, num_blocks);
    const int chunk_blocks = split_end - split_start;
    const int num_tiles = chunk_blocks * SUB_TILES;

    // Prologue: prefetch tile 0 (K then V = 2 async groups).
    if (num_tiles > 0) {
        const int pb0 = block_tables[seq_idx * max_blocks_per_seq + split_start];
#if PSP_KV_TOKEN_MAJOR
        const uint8_t* k0 = k_cache + block_offsets[pb0] + (size_t)kv_head * HEAD_DIM;
        const uint8_t* v0 = v_cache + block_offsets[num_total_blocks + pb0]
                          + (size_t)kv_head * HEAD_DIM;
        k32_cp_async_tile(sk[0], k0, lane, TILE, HEAD_DIM, num_kv_heads, SMEM_STRIDE);
        k32_cp_async_tile(sv[0], v0, lane, TILE, HEAD_DIM, num_kv_heads, SMEM_STRIDE);
#else
        const uint8_t* k0 = k_cache + block_offsets[pb0]
                          + (size_t)kv_head * BLOCK_BYTES;
        const uint8_t* v0 = v_cache + block_offsets[num_total_blocks + pb0]
                          + (size_t)kv_head * BLOCK_BYTES;
        k32_cp_async_tile(sk[0], k0, lane, TILE, HEAD_DIM, 1, SMEM_STRIDE);
        k32_cp_async_tile(sv[0], v0, lane, TILE, HEAD_DIM, 1, SMEM_STRIDE);
#endif
    }



    for (int n = 0; n < num_tiles; n++) {
        const int b = split_start + n / SUB_TILES;
        const int sub = n % SUB_TILES;
        const int cur = n & 1;
        const int nxt = cur ^ 1;

        if (n + 1 < num_tiles) {
            const int b1 = split_start + (n + 1) / SUB_TILES;
            const int sub1 = (n + 1) % SUB_TILES;
            const int pb1 = block_tables[seq_idx * max_blocks_per_seq + b1];
#if PSP_KV_TOKEN_MAJOR
            const uint8_t* k1 = k_cache + block_offsets[pb1]
                              + (size_t)sub1 * TILE * (num_kv_heads * HEAD_DIM)
                              + (size_t)kv_head * HEAD_DIM;
            const uint8_t* v1 = v_cache + block_offsets[num_total_blocks + pb1]
                              + (size_t)sub1 * TILE * (num_kv_heads * HEAD_DIM)
                              + (size_t)kv_head * HEAD_DIM;
            k32_cp_async_tile(sk[nxt], k1, lane, TILE, HEAD_DIM, num_kv_heads, SMEM_STRIDE);
            k32_cp_async_tile(sv[nxt], v1, lane, TILE, HEAD_DIM, num_kv_heads, SMEM_STRIDE);
#else
            const uint8_t* k1 = k_cache + block_offsets[pb1]
                              + (size_t)kv_head * BLOCK_BYTES
                              + (size_t)sub1 * TILE * HEAD_DIM;
            const uint8_t* v1 = v_cache + block_offsets[num_total_blocks + pb1]
                              + (size_t)kv_head * BLOCK_BYTES
                              + (size_t)sub1 * TILE * HEAD_DIM;
            k32_cp_async_tile(sk[nxt], k1, lane, TILE, HEAD_DIM, 1, SMEM_STRIDE);
            k32_cp_async_tile(sv[nxt], v1, lane, TILE, HEAD_DIM, 1, SMEM_STRIDE);
#endif
        }

        if (n + 1 < num_tiles) {
            asm volatile("cp.async.wait_group 2;");
        } else {
            asm volatile("cp.async.wait_group 0;");
        }
        __syncwarp();

        const int start_token = b * BLOCK_SIZE + sub * TILE;
        const int tokens_in_block = min(TILE, context_len - start_token);
        const int physical_block = block_tables[seq_idx * max_blocks_per_seq + b];
        float k_sc[NGRP], v_sc[NGRP];
        #pragma unroll
        for (int g = 0; g < NGRP; g++) {
            k_sc[g] = scale_table[((int64_t)physical_block * num_kv_heads + kv_head) * NGRP + g];
            v_sc[g] = scale_table[((int64_t)num_total_blocks * num_kv_heads
                                   + (int64_t)physical_block * num_kv_heads + kv_head) * NGRP + g];
        }

        // --- QK^T: 4 n-tiles x 4 k-steps, one int32->fp32 rescale per 32 dims --
        signed char* skw = sk[cur];
        #pragma unroll
        for (int nt = 0; nt < NT; nt++) {
            int32_t ci[4];
            float cf[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            #pragma unroll
            for (int kt = 0; kt < NK; kt++) {
                const signed char* row = skw + (nt * 8 + gid) * SMEM_STRIDE + kt * 32 + tig * 4;
                uint32_t bi[2];
                bi[0] = *reinterpret_cast<const uint32_t*>(row);
                bi[1] = *reinterpret_cast<const uint32_t*>(row + 16);
                mma_k32(ci, a_q[kt], bi);
                const float gs = scale * k_sc[kt] * q_scale[kt];
                cf[0] += __int2float_rn(ci[0]) * gs;
                cf[1] += __int2float_rn(ci[1]) * gs;
                cf[2] += __int2float_rn(ci[2]) * gs;
                cf[3] += __int2float_rn(ci[3]) * gs;
            }
            const int c0 = nt * 8 + tig * 2;
            s_score[gid * SSC_SCORE + c0] = cf[0];
            s_score[gid * SSC_SCORE + c0 + 1] = cf[1];
            s_score[(gid + 8) * SSC_SCORE + c0] = cf[2];
            s_score[(gid + 8) * SSC_SCORE + c0 + 1] = cf[3];
        }
        __syncwarp();

        // --- online softmax, 2 rows per lane (rows gid and gid+8) --------------
        #pragma unroll
        for (int rr = 0; rr < 2; rr++) {
            const int r = gid + rr * 8;
            const float* srow = s_score + r * SSC_SCORE;
            signed char* prow = s_prob + r * SSC_PROB;
            float mx = -1e30f;
            for (int cc = 0; cc < tokens_in_block; cc++)
                mx = fmaxf(mx, srow[cc]);
            float& rmax = rr ? rmax1 : rmax0;
            float& rsum = rr ? rsum1 : rsum0;
            const float m_new = fmaxf(rmax, mx);
            const float rescale_old = exp2f((rmax - m_new) * log2e);
            const float rescale_new = exp2f((mx - m_new) * log2e);
            rsum *= rescale_old;
            float tsum = 0.0f;
            // Unnormalized prob quantized to int8 (7 bits): e and rescale_new
            // are in (0, 1], so e*rescale_new*127 fits [0,127]. The /127 is
            // folded into the V scale after the PV MMA. Four bytes per store.
            #pragma unroll
            for (int c4 = 0; c4 < TILE; c4 += 4) {
                uint32_t packed = 0;
                #pragma unroll
                for (int j = 0; j < 4; j++) {
                    const int cc = c4 + j;
                    const float e = (cc < tokens_in_block)
                        ? exp2f((srow[cc] - mx) * log2e) : 0.0f;
                    packed |= (uint32_t)(uint8_t)(signed char)
                        __float2int_rn(e * rescale_new * 127.0f) << (8 * j);
                    tsum += e;
                }
                *reinterpret_cast<uint32_t*>(prow + c4) = packed;
            }
            rsum += tsum * rescale_new;
            rmax = m_new;
            #pragma unroll
            for (int nt8 = 0; nt8 < NQ8; nt8++) {
                if (rr == 0) {
                    acc[nt8].x[0] *= rescale_old; acc[nt8].x[1] *= rescale_old;
                    acc[nt8].x[4] *= rescale_old; acc[nt8].x[5] *= rescale_old;
                } else {
                    acc[nt8].x[2] *= rescale_old; acc[nt8].x[3] *= rescale_old;
                    acc[nt8].x[6] *= rescale_old; acc[nt8].x[7] *= rescale_old;
                }
            }
        }
        __syncwarp();

        // --- PV: wmma int8, one int32->fp32 rescale per dim chunk per tile ---
        signed char* svw = sv[cur];
        #pragma unroll
        for (int h = 0; h < TILE / 16; h++)
            load_matrix_sync(pa_i[h], s_prob + h * 16, SSC_PROB);
        // PV stays on wmma m16n16k16 (integer mma has no col-row layout, so an
        // m16n8k32 PV would need V transposed in smem). The int32 accumulator
        // is converted + scaled once per dim chunk per token tile; with a
        // 32-token tile the two 16-token K halves could share that rescale
        // (group-32 V scale), which is where the remaining PV win sits.
        #pragma unroll
        for (int nt8 = 0; nt8 < NQ8; nt8++) {
            fill_fragment(c_pv[nt8 & 1], 0);
            #pragma unroll
            for (int h = 0; h < TILE / 16; h++) {
                load_matrix_sync(vb_i, svw + h * 16 * SMEM_STRIDE + nt8 * 16, SMEM_STRIDE);
                mma_sync(c_pv[nt8 & 1], pa_i[h], vb_i, c_pv[nt8 & 1]);
            }
            const float vs = v_sc[nt8 >> 1] / 127.0f;
            #pragma unroll
            for (int i = 0; i < c_pv[nt8 & 1].num_elements; i++)
                acc[nt8].x[i] += __int2float_rn(c_pv[nt8 & 1].x[i]) * vs;
        }
        __syncwarp();
    }

    // Partial outputs: same layout as the production int8 path so the existing
    // merge kernel math applies (per (kv_head, seq, split, head, dim)).
    const int64_t p_base =
        (((int64_t)kv_head * gridDim.y + seq_idx) * num_splits + split_idx) * gqa + gid;
    #pragma unroll
    for (int nt8 = 0; nt8 < NQ8; nt8++) {
        const int c0 = nt8 * 16 + tig * 2, c1 = nt8 * 16 + 8 + tig * 2;
        if (gid < gqa) {
            float* p = partials + p_base * (2 + HEAD_DIM);
            p[2 + c0] = acc[nt8].x[0];
            p[2 + c0 + 1] = acc[nt8].x[1];
            p[2 + c1] = acc[nt8].x[4];
            p[2 + c1 + 1] = acc[nt8].x[5];
        }
    }
    if (tig == 0 && gid < gqa) {
        partials[p_base * (2 + HEAD_DIM) + 0] = rmax0;
        partials[p_base * (2 + HEAD_DIM) + 1] = rsum0;
    }
}

// Split-K merge: one CTA per (kv_head, seq), same partials layout as above.
template <int HEAD_DIM, typename scalar_t>
__global__ void paged_attention_enhanced_int8_k32_merge_kernel(
    scalar_t* __restrict__ out,
    const float* __restrict__ partials,
    int num_kv_heads,
    int num_seqs,
    int num_heads,
    int num_splits)
{
    const int kv_head = blockIdx.x;
    const int seq_idx = blockIdx.y;
    const int tid = threadIdx.x;
    const int gqa = num_heads / num_kv_heads;
    const int64_t hstride = 2 + HEAD_DIM;

    for (int h = 0; h < gqa; ++h) {
        const float* base = partials
            + (((int64_t)kv_head * num_seqs + seq_idx) * num_splits) * gqa * hstride
            + h * hstride;
        float global_max = -1e30f;
        for (int s = 0; s < num_splits; ++s)
            global_max = fmaxf(global_max, base[s * gqa * hstride + 0]);
        float global_sum = 0.0f;
        for (int s = 0; s < num_splits; ++s)
            global_sum += base[s * gqa * hstride + 1]
                        * __expf(base[s * gqa * hstride + 0] - global_max);
        const float inv_sum = 1.0f / global_sum;
        const int head = kv_head * gqa + h;
        for (int d = tid; d < HEAD_DIM; d += blockDim.x) {
            float o = 0.0f;
            for (int s = 0; s < num_splits; ++s)
                o += base[s * gqa * hstride + 2 + d]
                   * __expf(base[s * gqa * hstride + 0] - global_max);
            out[(int64_t)seq_idx * num_heads * HEAD_DIM + (int64_t)head * HEAD_DIM + d] =
                psp_from_float<scalar_t>(o * inv_sum);
        }
    }
}

template <typename scalar_t>
void launch_paged_attention_enhanced_int8_k32(
    scalar_t* out,
    const scalar_t* q,
    const uint8_t* k_cache,
    const uint8_t* v_cache,
    const int* block_tables,
    const int* context_lens,
    const uint32_t* block_offsets,
    const float* scale_table,
    float* partials,
    int max_blocks_per_seq,
    int num_seqs,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    float scale,
    int num_total_blocks,
    int num_splits,
    cudaStream_t stream = 0)
{
    psp_kv_dispatch_head_dim(head_dim, [&](auto hd) {
        constexpr int HEAD_DIM = decltype(hd)::value;
        if constexpr (HEAD_DIM % 32 == 0) {
            const dim3 grid(num_kv_heads, num_seqs, num_splits);
            paged_attention_enhanced_int8_k32_kernel<HEAD_DIM, scalar_t>
                <<<grid, 32, 0, stream>>>(
                    partials, q, k_cache, v_cache, block_tables, context_lens,
                    block_offsets, scale_table, max_blocks_per_seq, num_heads,
                    num_kv_heads, scale, num_splits, num_total_blocks);
            paged_attention_enhanced_int8_k32_merge_kernel<HEAD_DIM, scalar_t>
                <<<dim3(num_kv_heads, num_seqs), 128, 0, stream>>>(
                    out, partials, num_kv_heads, num_seqs, num_heads, num_splits);
        }
    });

    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "int8 k32 launch failed: %s\n", cudaGetErrorString(err));
    }
}

template void launch_paged_attention_enhanced_int8_k32<half>(
    half*, const half*, const uint8_t*, const uint8_t*, const int*, const int*,
    const uint32_t*, const float*, float*, int, int, int, int, int, float, int,
    int, cudaStream_t);
template void launch_paged_attention_enhanced_int8_k32<__nv_bfloat16>(
    __nv_bfloat16*, const __nv_bfloat16*, const uint8_t*, const uint8_t*,
    const int*, const int*, const uint32_t*, const float*, float*, int, int, int,
    int, int, float, int, int, cudaStream_t);

}  // namespace psp_kv_k32

// ---------------------------------------------------------------------------
// Op: psp_kv.int8_k32_probe
//   query          [num_seqs, num_heads, head_dim] fp16/bf16
//   key_cache      uint8 byte buffer, token-major [block][token][kv_head][dim]
//   value_cache    uint8 byte buffer, same layout
//   block_tables   [num_seqs, max_blocks_per_seq] int32
//   context_lens   [num_seqs] int32
//   block_offsets  [2 * num_blocks] int32 byte offsets (K section, V section)
//   scale_table    [2, num_blocks, num_kv_heads, head_dim/32] float32
//   out            [num_seqs, num_heads, head_dim]
// ---------------------------------------------------------------------------
namespace psp_kv_k32 {

void psp_attention_enhanced_int8_k32(
    at::Tensor& query,
    at::Tensor& key_cache,
    at::Tensor& value_cache,
    at::Tensor& block_tables,
    at::Tensor& context_lens,
    at::Tensor& block_offsets,
    at::Tensor& scale_table,
    double softmax_scale,
    int64_t num_kv_heads,
    int64_t head_size,
    int64_t num_splits,
    at::Tensor& out)
{
    TORCH_CHECK(query.is_contiguous(), "query must be contiguous");
    TORCH_CHECK(key_cache.is_contiguous(), "key_cache must be contiguous");
    TORCH_CHECK(value_cache.is_contiguous(), "value_cache must be contiguous");
    TORCH_CHECK(block_tables.is_contiguous(), "block_tables must be contiguous");
    TORCH_CHECK(context_lens.is_contiguous(), "context_lens must be contiguous");
    TORCH_CHECK(block_offsets.is_contiguous(), "block_offsets must be contiguous");
    TORCH_CHECK(scale_table.is_contiguous(), "scale_table must be contiguous");
    TORCH_CHECK(out.is_contiguous(), "out must be contiguous");

    TORCH_CHECK(block_tables.scalar_type() == at::kInt, "block_tables must be int32");
    TORCH_CHECK(context_lens.scalar_type() == at::kInt, "context_lens must be int32");
    TORCH_CHECK(block_offsets.scalar_type() == at::kInt, "block_offsets must be int32");
    TORCH_CHECK(scale_table.scalar_type() == at::kFloat, "scale_table must be float32");
    TORCH_CHECK(key_cache.scalar_type() == at::kByte, "key_cache must be uint8");
    TORCH_CHECK(value_cache.scalar_type() == at::kByte, "value_cache must be uint8");
    TORCH_CHECK(query.scalar_type() == at::kHalf || query.scalar_type() == at::kBFloat16,
                "int8_k32_probe supports fp16/bf16 query only");

    const int num_heads = static_cast<int>(query.size(1));
    const int head_dim = static_cast<int>(head_size);
    const int kv_heads = static_cast<int>(num_kv_heads);
    const int num_seqs = static_cast<int>(block_tables.size(0));
    const int max_blocks_per_seq = static_cast<int>(block_tables.size(1));

    TORCH_CHECK(query.size(2) == head_size, "query head_dim mismatch");
    TORCH_CHECK(context_lens.size(0) == num_seqs, "context_lens count mismatch");
    TORCH_CHECK(scale_table.dim() == 4 && scale_table.size(0) == 2
                && scale_table.size(2) == num_kv_heads
                && scale_table.size(3) == head_size / 32,
                "scale_table must be [2, num_blocks, num_kv_heads, head_dim/32]");
    TORCH_CHECK(head_size % 32 == 0, "int8_k32_probe needs head_dim % 32 == 0");
    TORCH_CHECK(BLOCK_SIZE % 32 == 0, "int8_k32_probe needs block_size % 32 == 0");
    TORCH_CHECK(num_heads % kv_heads == 0 && num_heads / kv_heads <= 16,
                "int8_k32_probe needs 1 <= num_heads/num_kv_heads <= 16");

    const int num_blocks = static_cast<int>(scale_table.size(1));
    const int gqa = num_heads / kv_heads;
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    auto partials = at::empty(
        {(int64_t)num_seqs * kv_heads * num_splits * gqa * (2 + head_dim)},
        at::TensorOptions().dtype(at::kFloat).device(key_cache.device()));

    if (query.scalar_type() == at::kHalf) {
        launch_paged_attention_enhanced_int8_k32<half>(
            reinterpret_cast<half*>(out.data_ptr()),
            reinterpret_cast<const half*>(query.data_ptr()),
            key_cache.data_ptr<uint8_t>(), value_cache.data_ptr<uint8_t>(),
            block_tables.data_ptr<int>(), context_lens.data_ptr<int>(),
            reinterpret_cast<const uint32_t*>(block_offsets.data_ptr<int>()),
            scale_table.data_ptr<float>(), partials.data_ptr<float>(),
            max_blocks_per_seq, num_seqs, num_heads, kv_heads, head_dim,
            static_cast<float>(softmax_scale), num_blocks,
            static_cast<int>(num_splits), stream);
    } else {
        launch_paged_attention_enhanced_int8_k32<__nv_bfloat16>(
            reinterpret_cast<__nv_bfloat16*>(out.data_ptr()),
            reinterpret_cast<const __nv_bfloat16*>(query.data_ptr()),
            key_cache.data_ptr<uint8_t>(), value_cache.data_ptr<uint8_t>(),
            block_tables.data_ptr<int>(), context_lens.data_ptr<int>(),
            reinterpret_cast<const uint32_t*>(block_offsets.data_ptr<int>()),
            scale_table.data_ptr<float>(), partials.data_ptr<float>(),
            max_blocks_per_seq, num_seqs, num_heads, kv_heads, head_dim,
            static_cast<float>(softmax_scale), num_blocks,
            static_cast<int>(num_splits), stream);
    }
}

}  // namespace psp_kv_k32

// TORCH_LIBRARY_FRAGMENT (not TORCH_LIBRARY): the psp_kv namespace is owned by
// csrc/attention/psp_kv_wrapper.cu, and torch allows only one TORCH_LIBRARY per
// namespace. A fragment adds this prototype op without touching that file.
TORCH_LIBRARY_FRAGMENT(psp_kv, m) {
    m.def(
        "psp_attention_enhanced_int8_k32("
        "Tensor query, "
        "Tensor key_cache, "
        "Tensor value_cache, "
        "Tensor block_tables, "
        "Tensor context_lens, "
        "Tensor block_offsets, "
        "Tensor scale_table, "
        "float softmax_scale, "
        "int num_kv_heads, "
        "int head_size, "
        "int num_splits, "
        "Tensor! out"
        ") -> ()");
}

TORCH_LIBRARY_IMPL(psp_kv, CUDA, m) {
    m.impl("psp_attention_enhanced_int8_k32", &psp_kv_k32::psp_attention_enhanced_int8_k32);
}
