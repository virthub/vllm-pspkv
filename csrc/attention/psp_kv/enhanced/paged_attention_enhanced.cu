// virthub/kernels/enhanced_format/paged_attention_enhanced.cu
//
// Enhanced Format Paged Attention Kernel (split-K + split-H)
//
// Paged attention for the Enhanced PSP-KV format. KV cache blocks are stored in
// quantized (FP8/FP4/lossless) representation with per-(block, head) float32
// scales. Layout: K token-major [B_t, D_head], V token-major [B_t, D_head].
//
// Decode-only 2D split: the context is split across `num_splits` blocks (split-K)
// and the head dimension across 2 blocks (split-H). Each block computes the full
// score (dot product) but only its half of the V weighted sum, so the per-thread
// accumulator is HALF the head dim (less register pressure -> higher occupancy).
// A second tiny kernel merges the split-K partials.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

#include "common/quantization_utils.cuh"
#include "common/head_dim_dispatch.cuh"
#include "fp8_mma_sm89.cuh"

using namespace nvcuda;
using namespace nvcuda::wmma;

#define BLOCK_SIZE 16
#define THREADS 128
#define NUM_SPLITS 2
#define NUM_SPLITS_H 2

__device__ inline uint32_t psp_f2u_mono(float f) {
    uint32_t u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
__device__ inline float psp_u2f_mono(uint32_t u) {
    return __uint_as_float((u & 0x80000000u) ? (u & 0x7FFFFFFFu) : ~u);
}

template <typename scalar_t, int HEAD_DIM>
__global__ void paged_attention_enhanced_split_kernel(
    float* __restrict__ partials,
    const scalar_t* __restrict__ q,
    const uint8_t* __restrict__ k_cache,
    const uint8_t* __restrict__ v_cache,
    const float* __restrict__ k_scales,
    const float* __restrict__ v_scales,
    const int* __restrict__ block_tables,
    const int* __restrict__ context_lens,
    const uint8_t* __restrict__ precision_table,
    const uint32_t* __restrict__ block_offsets,
    int max_blocks_per_seq,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    float scale,
    int num_splits,
    int sliding_window,
    int num_total_blocks)
{
    constexpr int HALF = HEAD_DIM / 2;

    const int head_idx = blockIdx.x;
    const int seq_idx = blockIdx.y;
    const int split_idx = blockIdx.z / NUM_SPLITS_H;  // split-K
    const int split_h = blockIdx.z % NUM_SPLITS_H;    // split-H (0 or 1)
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    const int context_len = context_lens[seq_idx];
    const int kv_head = head_idx / (num_heads / num_kv_heads);

    __shared__ float s_query[HEAD_DIM];
    __shared__ uint32_t s_max;
    __shared__ float s_sum;
    __shared__ float s_acc[HALF];

    const scalar_t* q_ptr = q + ((int64_t)seq_idx * num_heads + head_idx) * head_dim;
    for (int i = tid; i < HEAD_DIM; i += nthreads) s_query[i] = psp_to_float(q_ptr[i]);
    if (tid == 0) {
        s_max = psp_f2u_mono(-INFINITY);
        s_sum = 0.0f;
        for (int d = 0; d < HALF; ++d) s_acc[d] = 0.0f;
    }
    __syncthreads();

    // Sliding window: only the trailing `sliding_window` tokens attend.
    // The first window block may be partial (window_start not block-aligned).
    const int window_start = (sliding_window > 0 && sliding_window < context_len)
        ? context_len - sliding_window : 0;
    const int start_block = window_start / BLOCK_SIZE;
    const int num_blocks = (context_len + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int win_blocks = num_blocks - start_block;
    const int blocks_per_split = (win_blocks + num_splits - 1) / num_splits;
    const int split_start = start_block + split_idx * blocks_per_split;
    const int split_end = min(split_start + blocks_per_split, num_blocks);

    float local_max = -INFINITY;
    float local_sum = 0.0f;
    float acc[HALF];
    for (int d = 0; d < HALF; ++d) acc[d] = 0.0f;

    const int dim_offset = split_h * HALF;

    for (int b = split_start + tid; b < split_end; b += nthreads) {
        const int physical_block = block_tables[seq_idx * max_blocks_per_seq + b];
        const int packed = (int)precision_table[physical_block];
        const int k_fmt = psp_kv_fmt_k(packed);
        const int v_fmt = psp_kv_fmt_v(packed);
        constexpr int NG = HEAD_DIM / PSP_KV_SCALE_GROUP;
        float k_sc[NG], v_sc[NG];
        const int gbase = (physical_block * num_kv_heads + kv_head) * NG;
        if (k_fmt == 0) {
            #pragma unroll
            for (int g = 0; g < NG; ++g) k_sc[g] = scale;
        } else {
            #pragma unroll
            for (int g = 0; g < NG; ++g) k_sc[g] = k_scales[gbase + g] * scale;
        }
        if (v_fmt == 0) {
            #pragma unroll
            for (int g = 0; g < NG; ++g) v_sc[g] = 1.0f;
        } else {
            #pragma unroll
            for (int g = 0; g < NG; ++g) v_sc[g] = v_scales[gbase + g];
        }
        int k_block_bytes;
        if (k_fmt == 0)      k_block_bytes = BLOCK_SIZE * HEAD_DIM * (int)sizeof(scalar_t);
        else if (k_fmt == 3) k_block_bytes = BLOCK_SIZE * HEAD_DIM / 2;
        else                 k_block_bytes = BLOCK_SIZE * HEAD_DIM;
        int v_block_bytes;
        if (v_fmt == 0)      v_block_bytes = BLOCK_SIZE * HEAD_DIM * (int)sizeof(scalar_t);
        else if (v_fmt == 3) v_block_bytes = BLOCK_SIZE * HEAD_DIM / 2;
        else                 v_block_bytes = BLOCK_SIZE * HEAD_DIM;
        const uint8_t* k_base = k_cache + block_offsets[physical_block]
                              + (size_t)kv_head * k_block_bytes;
        const uint8_t* v_base = v_cache + block_offsets[num_total_blocks + physical_block]
                              + (size_t)kv_head * v_block_bytes;
        const int start_token = b * BLOCK_SIZE;
        const int token_offset = (b == start_block) ? (window_start - start_token) : 0;
        const int tokens_in_block = min(BLOCK_SIZE, context_len - start_token) - token_offset;
        const int combo = k_fmt * 4 + v_fmt;
        switch (combo) {
            case 0:  psp_decode_block<scalar_t, HEAD_DIM, 0, 0, HEAD_DIM/2>(s_query, local_max, local_sum, acc, k_base, v_base, tokens_in_block, k_sc, v_sc, dim_offset, token_offset); break;
            case 1:  psp_decode_block<scalar_t, HEAD_DIM, 0, 1, HEAD_DIM/2>(s_query, local_max, local_sum, acc, k_base, v_base, tokens_in_block, k_sc, v_sc, dim_offset, token_offset); break;
            case 2:  psp_decode_block<scalar_t, HEAD_DIM, 0, 2, HEAD_DIM/2>(s_query, local_max, local_sum, acc, k_base, v_base, tokens_in_block, k_sc, v_sc, dim_offset, token_offset); break;
            case 3:  psp_decode_block<scalar_t, HEAD_DIM, 0, 3, HEAD_DIM/2>(s_query, local_max, local_sum, acc, k_base, v_base, tokens_in_block, k_sc, v_sc, dim_offset, token_offset); break;
            case 4:  psp_decode_block<scalar_t, HEAD_DIM, 1, 0, HEAD_DIM/2>(s_query, local_max, local_sum, acc, k_base, v_base, tokens_in_block, k_sc, v_sc, dim_offset, token_offset); break;
            case 5:  psp_decode_block<scalar_t, HEAD_DIM, 1, 1, HEAD_DIM/2>(s_query, local_max, local_sum, acc, k_base, v_base, tokens_in_block, k_sc, v_sc, dim_offset, token_offset); break;
            case 6:  psp_decode_block<scalar_t, HEAD_DIM, 1, 2, HEAD_DIM/2>(s_query, local_max, local_sum, acc, k_base, v_base, tokens_in_block, k_sc, v_sc, dim_offset, token_offset); break;
            case 7:  psp_decode_block<scalar_t, HEAD_DIM, 1, 3, HEAD_DIM/2>(s_query, local_max, local_sum, acc, k_base, v_base, tokens_in_block, k_sc, v_sc, dim_offset, token_offset); break;
            case 8:  psp_decode_block<scalar_t, HEAD_DIM, 2, 0, HEAD_DIM/2>(s_query, local_max, local_sum, acc, k_base, v_base, tokens_in_block, k_sc, v_sc, dim_offset, token_offset); break;
            case 9:  psp_decode_block<scalar_t, HEAD_DIM, 2, 1, HEAD_DIM/2>(s_query, local_max, local_sum, acc, k_base, v_base, tokens_in_block, k_sc, v_sc, dim_offset, token_offset); break;
            case 10: psp_decode_block<scalar_t, HEAD_DIM, 2, 2, HEAD_DIM/2>(s_query, local_max, local_sum, acc, k_base, v_base, tokens_in_block, k_sc, v_sc, dim_offset, token_offset); break;
            case 11: psp_decode_block<scalar_t, HEAD_DIM, 2, 3, HEAD_DIM/2>(s_query, local_max, local_sum, acc, k_base, v_base, tokens_in_block, k_sc, v_sc, dim_offset, token_offset); break;
            case 12: psp_decode_block<scalar_t, HEAD_DIM, 3, 0, HEAD_DIM/2>(s_query, local_max, local_sum, acc, k_base, v_base, tokens_in_block, k_sc, v_sc, dim_offset, token_offset); break;
            case 13: psp_decode_block<scalar_t, HEAD_DIM, 3, 1, HEAD_DIM/2>(s_query, local_max, local_sum, acc, k_base, v_base, tokens_in_block, k_sc, v_sc, dim_offset, token_offset); break;
            case 14: psp_decode_block<scalar_t, HEAD_DIM, 3, 2, HEAD_DIM/2>(s_query, local_max, local_sum, acc, k_base, v_base, tokens_in_block, k_sc, v_sc, dim_offset, token_offset); break;
            default: psp_decode_block<scalar_t, HEAD_DIM, 3, 3, HEAD_DIM/2>(s_query, local_max, local_sum, acc, k_base, v_base, tokens_in_block, k_sc, v_sc, dim_offset, token_offset); break;
        }
    }

    atomicMax(&s_max, psp_f2u_mono(local_max));
    __syncthreads();
    const float gmax = psp_u2f_mono(s_max);
    if (local_max > -INFINITY) {
        const float g_rescale = __expf(local_max - gmax);
        local_sum *= g_rescale;
        for (int d = 0; d < HALF; ++d) acc[d] *= g_rescale;
    }
    atomicAdd(&s_sum, local_sum);
    for (int d = 0; d < HALF; ++d) atomicAdd(&s_acc[d], acc[d]);
    __syncthreads();

    if (tid == 0) {
        const int64_t p_idx =
            (((int64_t)seq_idx * num_heads + head_idx) * NUM_SPLITS_H + split_h)
            * num_splits + split_idx;
        float* p = partials + p_idx * (2 + HALF);
        p[0] = gmax;
        p[1] = s_sum;
        for (int d = 0; d < HALF; ++d) p[2 + d] = s_acc[d];
    }
}

template <typename scalar_t, int HEAD_DIM>
__global__ void paged_attention_enhanced_merge_kernel(
    scalar_t* __restrict__ out,
    const float* __restrict__ partials,
    int num_heads,
    int num_seqs,
    int num_splits)
{
    constexpr int HALF = HEAD_DIM / 2;
    const int head_idx = blockIdx.x;
    const int seq_idx = blockIdx.y;
    const int split_h = blockIdx.z;
    const int tid = threadIdx.x;

    const int64_t base =
        (((int64_t)seq_idx * num_heads + head_idx) * NUM_SPLITS_H + split_h) * num_splits;
    float global_max = -INFINITY;
    for (int s = 0; s < num_splits; ++s) {
        global_max = fmaxf(global_max, partials[(base + s) * (2 + HALF) + 0]);
    }
    float global_sum = 0.0f;
    float o = 0.0f;
    if (tid < HALF) {
        for (int s = 0; s < num_splits; ++s) {
            const float* p = partials + (base + s) * (2 + HALF);
            const float rescale = __expf(p[0] - global_max);
            global_sum += p[1] * rescale;
            o += p[2 + tid] * rescale;
        }
    }
    if (tid < HALF) {
        const float inv_sum = 1.0f / global_sum;
        out[((int64_t)seq_idx * num_heads + head_idx) * HEAD_DIM + split_h * HALF + tid] =
            psp_from_float<scalar_t>(o * inv_sum);
    }
}

// ============================================================================
// Tensor-core decode path (bf16 lossless inner precision).
//
// One warp per (kv_head, split-K chunk). The m16 tile holds the `gqa` query
// heads (padded with zeros); K/V are the per-kv_head bf16 byte buffer loaded
// via cp.async + wmma. The per-warp online softmax keeps a register-resident
// accumulator; partials (max, sum, acc) are merged by a second kernel.
// ============================================================================

#define TC_NUM_SPLITS 8
#define TC_THREADS 32

__device__ __forceinline__ void cp_async_tc(uint8_t* smem, const uint8_t* gmem,
                                            int tid, int bytes) {
    const int chunks = bytes / 16;
    #pragma unroll
    for (int i = tid; i < chunks; i += 32) {
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(
            (unsigned)__cvta_generic_to_shared(smem + i * 16)), "l"(gmem + i * 16));
    }
    asm volatile("cp.async.commit_group;");
}

template <int HEAD_DIM, typename scalar_t>
__global__ void paged_attention_enhanced_tc_kernel(
    float* __restrict__ partials,
    const scalar_t* __restrict__ q,
    const uint8_t* __restrict__ k_cache,
    const uint8_t* __restrict__ v_cache,
    const int* __restrict__ block_tables,
    const int* __restrict__ context_lens,
    const uint32_t* __restrict__ block_offsets,
    int max_blocks_per_seq,
    int num_heads,
    int num_kv_heads,
    float scale,
    int num_splits,
    int sliding_window,
    int num_total_blocks)
{
    constexpr int NT8 = HEAD_DIM / 16;
    constexpr int BLOCK_BYTES = BLOCK_SIZE * HEAD_DIM * 2;  // bf16/fp16 lossless

    const int kv_head = blockIdx.x;
    const int seq_idx = blockIdx.y;
    const int split_idx = blockIdx.z;
    const int gqa = num_heads / num_kv_heads;
    const int tid = threadIdx.x;
    const int lane = tid & 31, group = lane >> 2, tig = lane & 3;
    const float log2e = 1.4426950408889634f;

    __shared__ scalar_t s_q[16 * HEAD_DIM];
    __shared__ scalar_t sk[2][16 * HEAD_DIM];
    __shared__ scalar_t sv[2][16 * HEAD_DIM];
    __shared__ float s_score[16 * 16];
    __shared__ scalar_t s_prob[16 * 16];

    fragment<matrix_a, 16, 16, 16, scalar_t, row_major> a[NT8];
    fragment<matrix_b, 16, 16, 16, scalar_t, col_major> b_frag;
    fragment<matrix_b, 16, 16, 16, scalar_t, row_major> vb;
    fragment<matrix_a, 16, 16, 16, scalar_t, row_major> pa;
    fragment<accumulator, 16, 16, 16, float> c;
    fragment<accumulator, 16, 16, 16, float> acc[NT8];

    // Padded Q: rows [0, gqa) = the query heads, rest zero.
    for (int i = tid; i < 16 * HEAD_DIM; i += 32) s_q[i] = psp_from_float<scalar_t>(0.0f);
    __syncthreads();
    const int head_base = kv_head * gqa;
    for (int i = tid; i < gqa * HEAD_DIM; i += 32) {
        int h = head_base + i / HEAD_DIM;
        int d = i % HEAD_DIM;
        s_q[i] = q[(int64_t)seq_idx * num_heads * HEAD_DIM + (int64_t)h * HEAD_DIM + d];
    }
    #pragma unroll
    for (int kt = 0; kt < NT8; kt++) load_matrix_sync(a[kt], s_q + kt * 16, HEAD_DIM);
    #pragma unroll
    for (int nt8 = 0; nt8 < NT8; nt8++) fill_fragment(acc[nt8], 0.0f);

    float rmax0 = -1e30f, rmax1 = -1e30f, rsum0 = 0.0f, rsum1 = 0.0f;

    const int context_len = context_lens[seq_idx];
    const int num_blocks = (context_len + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int blocks_per_split = (num_blocks + num_splits - 1) / num_splits;
    const int split_start = split_idx * blocks_per_split;
    const int split_end = min(split_start + blocks_per_split, num_blocks);
    const int chunk_blocks = split_end - split_start;

    // Sliding-window mask: the decode query (position context_len) attends
    // only to tokens >= window_start; <= 0 means no window.
    const int window_start = (sliding_window > 0)
        ? max(0, context_len - sliding_window + 1) : 0;

    const int block_bytes_per_head = BLOCK_BYTES;

    // Prologue: prefetch block 0 into the first buffer. Guarded so empty
    // splits (num_splits > num_blocks) never dereference an out-of-range block.
    if (chunk_blocks > 0) {
        const int pb0 = block_tables[seq_idx * max_blocks_per_seq + split_start];
        const uint8_t* k0 = k_cache + block_offsets[pb0] + (size_t)kv_head * block_bytes_per_head;
        const uint8_t* v0 = v_cache + block_offsets[num_total_blocks + pb0] + (size_t)kv_head * block_bytes_per_head;
        cp_async_tc(reinterpret_cast<uint8_t*>(sk[0]), k0, tid, BLOCK_BYTES);
        cp_async_tc(reinterpret_cast<uint8_t*>(sv[0]), v0, tid, BLOCK_BYTES);
    }

    for (int n = 0; n < chunk_blocks; n++) {
        const int b = split_start + n;
        const int cur = n & 1;
        const int nxt = cur ^ 1;

        // Prefetch block n+1 (2 async groups) so it overlaps the current
        // compute.
        if (n + 1 < chunk_blocks) {
            const int pb1 = block_tables[seq_idx * max_blocks_per_seq + b + 1];
            const uint8_t* k1 = k_cache + block_offsets[pb1] + (size_t)kv_head * block_bytes_per_head;
            const uint8_t* v1 = v_cache + block_offsets[num_total_blocks + pb1] + (size_t)kv_head * block_bytes_per_head;
            cp_async_tc(reinterpret_cast<uint8_t*>(sk[nxt]), k1, tid, BLOCK_BYTES);
            cp_async_tc(reinterpret_cast<uint8_t*>(sv[nxt]), v1, tid, BLOCK_BYTES);
        }

        // Wait for the current block's copy, keeping the 2 prefetch groups
        // outstanding so they overlap the compute below.
        if (n + 1 < chunk_blocks) {
            asm volatile("cp.async.wait_group 2;");
        } else {
            asm volatile("cp.async.wait_group 0;");
        }
        __syncthreads();

        const int start_token = b * BLOCK_SIZE;
        const int tokens_in_block = min(BLOCK_SIZE, context_len - start_token);

        // QK -> scores [16, 16]
        fill_fragment(c, 0.0f);
        #pragma unroll
        for (int kt = 0; kt < NT8; kt++) {
            load_matrix_sync(b_frag, sk[cur] + kt * 16, HEAD_DIM);
            mma_sync(c, a[kt], b_frag, c);
        }
        for (int i = 0; i < c.num_elements; i++) c.x[i] *= scale;
        store_matrix_sync(s_score, c, 16, mem_row_major);
        __syncthreads();

        // per-row online softmax (2 rows per thread), masked to valid tokens
        for (int r = group; r < 16; r += 8) {
            float mx = -1e30f;
            for (int cc = 0; cc < tokens_in_block; cc++)
                if (start_token + cc >= window_start) mx = fmaxf(mx, s_score[r * 16 + cc]);
            float& rmax = (r == group) ? rmax0 : rmax1;
            float& rsum = (r == group) ? rsum0 : rsum1;
            float m_new = fmaxf(rmax, mx);
            float rescale_old = exp2f((rmax - m_new) * log2e);
            float rescale_new = exp2f((mx - m_new) * log2e);
            rsum *= rescale_old;
            float tsum = 0.0f;
            for (int cc = 0; cc < 16; cc++) {
                float e = (cc < tokens_in_block && start_token + cc >= window_start)
                    ? exp2f((s_score[r * 16 + cc] - mx) * log2e) : 0.0f;
                s_prob[r * 16 + cc] = psp_from_float<scalar_t>(e * rescale_new);
                tsum += e;
            }
            rsum += tsum * rescale_new;
            rmax = m_new;
            #pragma unroll
            for (int nt8 = 0; nt8 < NT8; nt8++) {
                if (r == group) {
                    acc[nt8].x[0] *= rescale_old; acc[nt8].x[1] *= rescale_old;
                    acc[nt8].x[4] *= rescale_old; acc[nt8].x[5] *= rescale_old;
                } else {
                    acc[nt8].x[2] *= rescale_old; acc[nt8].x[3] *= rescale_old;
                    acc[nt8].x[6] *= rescale_old; acc[nt8].x[7] *= rescale_old;
                }
            }
        }
        __syncthreads();

        // PV: acc += prob @ V_tile
        load_matrix_sync(pa, s_prob, 16);
        #pragma unroll
        for (int nt8 = 0; nt8 < NT8; nt8++) {
            load_matrix_sync(vb, sv[cur] + nt8 * 16, HEAD_DIM);
            mma_sync(acc[nt8], pa, vb, acc[nt8]);
        }
        __syncthreads();
    }

    // Output partials: per (kv_head, seq, split, head, dim).
    const int64_t p_base =
        (((int64_t)kv_head * gridDim.y + seq_idx) * num_splits + split_idx) * gqa + group;
    #pragma unroll
    for (int nt8 = 0; nt8 < NT8; nt8++) {
        int c0 = nt8 * 16 + tig * 2, c1 = nt8 * 16 + 8 + tig * 2;
        if (group < gqa) {
            float* p = partials + p_base * (2 + HEAD_DIM);
            p[2 + c0]      = acc[nt8].x[0];
            p[2 + c0 + 1]  = acc[nt8].x[1];
            p[2 + c1]      = acc[nt8].x[4];
            p[2 + c1 + 1]  = acc[nt8].x[5];
        }
    }
    if (tig == 0 && group < gqa) {
        partials[p_base * (2 + HEAD_DIM) + 0] = rmax0;
        partials[p_base * (2 + HEAD_DIM) + 1] = rsum0;
    }
}

// ---------------------------------------------------------------------------
// tf32 tensor-core decode for the fp32 target: fp32 query + fp32 lossless cache
// (4 B/elem) through the tf32 m16n16k8 mma (10-bit mantissa). The contraction K
// is 8, so the QK dims and the PV tokens each split into 8-wide chunks.
// ---------------------------------------------------------------------------
template <int HEAD_DIM>
__global__ void paged_attention_enhanced_tf32_tc_kernel(
    float* __restrict__ partials,
    const float* __restrict__ q,
    const uint8_t* __restrict__ k_cache,
    const uint8_t* __restrict__ v_cache,
    const int* __restrict__ block_tables,
    const int* __restrict__ context_lens,
    const uint32_t* __restrict__ block_offsets,
    int max_blocks_per_seq,
    int num_heads,
    int num_kv_heads,
    float scale,
    int num_splits,
    int sliding_window,
    int num_total_blocks)
{
    constexpr int NT8_Q = HEAD_DIM / 8;    // QK contraction (dims) in k=8
    constexpr int NT8_PV = HEAD_DIM / 16;  // PV N (dims) in n=16
    constexpr int BLOCK_BYTES = BLOCK_SIZE * HEAD_DIM * (int)sizeof(float);

    const int kv_head = blockIdx.x;
    const int seq_idx = blockIdx.y;
    const int split_idx = blockIdx.z;
    const int gqa = num_heads / num_kv_heads;
    const int tid = threadIdx.x;
    const int lane = tid & 31, group = lane >> 2, tig = lane & 3;
    const float log2e = 1.4426950408889634f;

    __shared__ float s_q[16 * HEAD_DIM];
    __shared__ float sk[2][16 * HEAD_DIM];
    __shared__ float sv[2][16 * HEAD_DIM];
    __shared__ float s_score[16 * 16];
    __shared__ float s_prob[16 * 16];

    fragment<matrix_a, 16, 16, 8, precision::tf32, row_major> a[NT8_Q];
    fragment<matrix_b, 16, 16, 8, precision::tf32, col_major> b_frag;
    fragment<matrix_b, 16, 16, 8, precision::tf32, row_major> vb;
    fragment<matrix_a, 16, 16, 8, precision::tf32, row_major> pa;
    fragment<accumulator, 16, 16, 8, float> c;
    fragment<accumulator, 16, 16, 8, float> acc[NT8_PV];

    // Padded Q: rows [0, gqa) = the query heads, rest zero.
    for (int i = tid; i < 16 * HEAD_DIM; i += 32) s_q[i] = 0.0f;
    __syncthreads();
    const int head_base = kv_head * gqa;
    for (int i = tid; i < gqa * HEAD_DIM; i += 32) {
        int h = head_base + i / HEAD_DIM;
        int d = i % HEAD_DIM;
        s_q[i] = q[(int64_t)seq_idx * num_heads * HEAD_DIM + (int64_t)h * HEAD_DIM + d];
    }
    #pragma unroll
    for (int kt = 0; kt < NT8_Q; kt++) load_matrix_sync(a[kt], s_q + kt * 8, HEAD_DIM);
    #pragma unroll
    for (int nt8 = 0; nt8 < NT8_PV; nt8++) fill_fragment(acc[nt8], 0.0f);

    float rmax0 = -1e30f, rmax1 = -1e30f, rsum0 = 0.0f, rsum1 = 0.0f;

    const int context_len = context_lens[seq_idx];
    const int num_blocks = (context_len + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int blocks_per_split = (num_blocks + num_splits - 1) / num_splits;
    const int split_start = split_idx * blocks_per_split;
    const int split_end = min(split_start + blocks_per_split, num_blocks);
    const int chunk_blocks = split_end - split_start;

    const int window_start = (sliding_window > 0)
        ? max(0, context_len - sliding_window + 1) : 0;

    const int block_bytes_per_head = BLOCK_BYTES;

    // Prologue: prefetch block 0.
    {
        const int pb0 = block_tables[seq_idx * max_blocks_per_seq + split_start];
        const uint8_t* k0 = k_cache + block_offsets[pb0] + (size_t)kv_head * block_bytes_per_head;
        const uint8_t* v0 = v_cache + block_offsets[num_total_blocks + pb0] + (size_t)kv_head * block_bytes_per_head;
        cp_async_tc(reinterpret_cast<uint8_t*>(sk[0]), k0, tid, BLOCK_BYTES);
        cp_async_tc(reinterpret_cast<uint8_t*>(sv[0]), v0, tid, BLOCK_BYTES);
    }

    for (int n = 0; n < chunk_blocks; n++) {
        const int b = split_start + n;
        const int cur = n & 1;
        const int nxt = cur ^ 1;

        if (n + 1 < chunk_blocks) {
            const int pb1 = block_tables[seq_idx * max_blocks_per_seq + b + 1];
            const uint8_t* k1 = k_cache + block_offsets[pb1] + (size_t)kv_head * block_bytes_per_head;
            const uint8_t* v1 = v_cache + block_offsets[num_total_blocks + pb1] + (size_t)kv_head * block_bytes_per_head;
            cp_async_tc(reinterpret_cast<uint8_t*>(sk[nxt]), k1, tid, BLOCK_BYTES);
            cp_async_tc(reinterpret_cast<uint8_t*>(sv[nxt]), v1, tid, BLOCK_BYTES);
        }

        if (n + 1 < chunk_blocks) {
            asm volatile("cp.async.wait_group 2;");
        } else {
            asm volatile("cp.async.wait_group 0;");
        }
        __syncthreads();

        const int start_token = b * BLOCK_SIZE;
        const int tokens_in_block = min(BLOCK_SIZE, context_len - start_token);

        // QK -> scores [16, 16]
        fill_fragment(c, 0.0f);
        #pragma unroll
        for (int kt = 0; kt < NT8_Q; kt++) {
            load_matrix_sync(b_frag, sk[cur] + kt * 8, HEAD_DIM);
            mma_sync(c, a[kt], b_frag, c);
        }
        for (int i = 0; i < c.num_elements; i++) c.x[i] *= scale;
        store_matrix_sync(s_score, c, 16, mem_row_major);
        __syncthreads();

        // per-row online softmax (2 rows per thread), masked to valid tokens
        for (int r = group; r < 16; r += 8) {
            float mx = -1e30f;
            for (int cc = 0; cc < tokens_in_block; cc++)
                if (start_token + cc >= window_start) mx = fmaxf(mx, s_score[r * 16 + cc]);
            float& rmax = (r == group) ? rmax0 : rmax1;
            float& rsum = (r == group) ? rsum0 : rsum1;
            float m_new = fmaxf(rmax, mx);
            float rescale_old = exp2f((rmax - m_new) * log2e);
            float rescale_new = exp2f((mx - m_new) * log2e);
            rsum *= rescale_old;
            float tsum = 0.0f;
            for (int cc = 0; cc < 16; cc++) {
                float e = (cc < tokens_in_block && start_token + cc >= window_start)
                    ? exp2f((s_score[r * 16 + cc] - mx) * log2e) : 0.0f;
                s_prob[r * 16 + cc] = e * rescale_new;
                tsum += e;
            }
            rsum += tsum * rescale_new;
            rmax = m_new;
            #pragma unroll
            for (int nt8 = 0; nt8 < NT8_PV; nt8++) {
                if (r == group) {
                    acc[nt8].x[0] *= rescale_old; acc[nt8].x[1] *= rescale_old;
                    acc[nt8].x[4] *= rescale_old; acc[nt8].x[5] *= rescale_old;
                } else {
                    acc[nt8].x[2] *= rescale_old; acc[nt8].x[3] *= rescale_old;
                    acc[nt8].x[6] *= rescale_old; acc[nt8].x[7] *= rescale_old;
                }
            }
        }
        __syncthreads();

        // PV: acc += prob @ V_tile (two 8-token chunks per tf32 k=8)
        #pragma unroll
        for (int nt8 = 0; nt8 < NT8_PV; nt8++) {
            #pragma unroll
            for (int tt = 0; tt < 2; tt++) {
                load_matrix_sync(pa, s_prob + tt * 8, 16);
                load_matrix_sync(vb, sv[cur] + tt * 8 * HEAD_DIM + nt8 * 16, HEAD_DIM);
                mma_sync(acc[nt8], pa, vb, acc[nt8]);
            }
        }
        __syncthreads();
    }

    // Output partials: per (kv_head, seq, split, head, dim).
    const int64_t p_base =
        (((int64_t)kv_head * gridDim.y + seq_idx) * num_splits + split_idx) * gqa + group;
    #pragma unroll
    for (int nt8 = 0; nt8 < NT8_PV; nt8++) {
        int c0 = nt8 * 16 + tig * 2, c1 = nt8 * 16 + 8 + tig * 2;
        if (group < gqa) {
            float* p = partials + p_base * (2 + HEAD_DIM);
            p[2 + c0]      = acc[nt8].x[0];
            p[2 + c0 + 1]  = acc[nt8].x[1];
            p[2 + c1]      = acc[nt8].x[4];
            p[2 + c1 + 1]  = acc[nt8].x[5];
        }
    }
    if (tig == 0 && group < gqa) {
        partials[p_base * (2 + HEAD_DIM) + 0] = rmax0;
        partials[p_base * (2 + HEAD_DIM) + 1] = rsum0;
    }
}

template <int HEAD_DIM, typename scalar_t>
__global__ void paged_attention_enhanced_tc_merge_kernel(
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

    for (int h = 0; h < gqa; ++h) {
        const float* base = partials
            + (((int64_t)kv_head * num_seqs + seq_idx) * num_splits) * gqa * (2 + HEAD_DIM)
            + h * (2 + HEAD_DIM);
        float global_max = -1e30f;
        for (int s = 0; s < num_splits; ++s) {
            global_max = fmaxf(global_max, base[s * gqa * (2 + HEAD_DIM) + 0]);
        }
        float global_sum = 0.0f;
        for (int s = 0; s < num_splits; ++s) {
            float rescale = __expf(base[s * gqa * (2 + HEAD_DIM) + 0] - global_max);
            global_sum += base[s * gqa * (2 + HEAD_DIM) + 1] * rescale;
        }
        float inv_sum = 1.0f / global_sum;
        const int head = kv_head * gqa + h;
        for (int d = tid; d < HEAD_DIM; d += blockDim.x) {
            float o = 0.0f;
            for (int s = 0; s < num_splits; ++s) {
                o += base[s * gqa * (2 + HEAD_DIM) + 2 + d]
                     * __expf(base[s * gqa * (2 + HEAD_DIM) + 0] - global_max);
            }
            out[(int64_t)seq_idx * num_heads * HEAD_DIM + (int64_t)head * HEAD_DIM + d] =
                psp_from_float<scalar_t>(o * inv_sum);
        }
    }
}

// Fused merge + decode write-back. Each (kv_head, seq) block merges the split-K
// partials into the output and then writes the current token's K/V for this
// kv_head into the byte cache (num_tokens == num_seqs for decode), removing the
// separate quantize_single_token launch per layer.
template <int HEAD_DIM, typename scalar_t>
__global__ void paged_attention_enhanced_tc_merge_writeback_kernel(
    scalar_t* __restrict__ out,
    const float* __restrict__ partials,
    const scalar_t* __restrict__ key,       // [num_tokens, num_kv_heads, head_dim]
    const scalar_t* __restrict__ value,
    uint8_t* __restrict__ key_cache, uint8_t* __restrict__ value_cache,
    const int64_t* __restrict__ slot_mapping,  // [num_tokens]
    const uint8_t* __restrict__ precision_table,
    const uint32_t* __restrict__ block_offsets,
    const float* __restrict__ scale_table,
    int num_kv_heads, int num_seqs, int num_heads, int num_splits,
    int num_blocks)
{
    const int kv_head = blockIdx.x;
    const int seq_idx = blockIdx.y;
    const int tid = threadIdx.x;
    const int gqa = num_heads / num_kv_heads;

    for (int h = 0; h < gqa; ++h) {
        const float* base = partials
            + (((int64_t)kv_head * num_seqs + seq_idx) * num_splits) * gqa * (2 + HEAD_DIM)
            + h * (2 + HEAD_DIM);
        float global_max = -1e30f;
        for (int s = 0; s < num_splits; ++s) {
            global_max = fmaxf(global_max, base[s * gqa * (2 + HEAD_DIM) + 0]);
        }
        float global_sum = 0.0f;
        for (int s = 0; s < num_splits; ++s) {
            float rescale = __expf(base[s * gqa * (2 + HEAD_DIM) + 0] - global_max);
            global_sum += base[s * gqa * (2 + HEAD_DIM) + 1] * rescale;
        }
        float inv_sum = 1.0f / global_sum;
        const int head = kv_head * gqa + h;
        for (int d = tid; d < HEAD_DIM; d += blockDim.x) {
            float o = 0.0f;
            for (int s = 0; s < num_splits; ++s) {
                o += base[s * gqa * (2 + HEAD_DIM) + 2 + d]
                     * __expf(base[s * gqa * (2 + HEAD_DIM) + 0] - global_max);
            }
            out[(int64_t)seq_idx * num_heads * HEAD_DIM + (int64_t)head * HEAD_DIM + d] =
                psp_from_float<scalar_t>(o * inv_sum);
        }
    }

    // Decode write-back (one thread per (seq, kv_head) block): quantize this
    // token's K/V row with the block's existing scale (or on-the-fly for a
    // freshly-allocated block) and scatter into the byte cache.
    if (tid == 0) {
        const int block_size = 16;
        const int64_t slot = slot_mapping[seq_idx];
        const int block_id = static_cast<int>(slot / block_size);
        const int block_offset = static_cast<int>(slot % block_size);
        const int packed = static_cast<int>(precision_table[block_id]);
        const int k_fmt = psp_kv_fmt_k(packed);
        const int v_fmt = psp_kv_fmt_v(packed);

        auto bytes_per_head = [&](int fmt) -> int {
            if (fmt == 0)      return block_size * HEAD_DIM * (int)sizeof(scalar_t);
            else if (fmt == 3) return block_size * HEAD_DIM / 2;
            else               return block_size * HEAD_DIM;
        };
        auto bytes_per_row = [&](int fmt) -> int {
            if (fmt == 0)      return HEAD_DIM * (int)sizeof(scalar_t);
            else if (fmt == 3) return HEAD_DIM / 2;
            else               return HEAD_DIM;
        };
        auto fp_max = [&](int fmt) -> float {
            if (fmt == 3) return 3.0f;
            if (fmt == 2) return 57344.0f;
            return 240.0f;
        };
        auto compute_scale = [&](const scalar_t* row, int fmt) -> float {
            if (fmt == 0) return 1.0f;
            float m = 0.0f;
            for (int i = 0; i < HEAD_DIM; ++i) m = fmaxf(m, fabsf(psp_to_float(row[i])));
            float s = m / fp_max(fmt);
            return (s < 1e-8f) ? 1.0f : s;
        };

        const int k_block_bytes = bytes_per_head(k_fmt);
        const int v_block_bytes = bytes_per_head(v_fmt);
        const int k_row_bytes = bytes_per_row(k_fmt);

        const int64_t hh = (int64_t)num_blocks * num_kv_heads * psp_kv_num_groups(HEAD_DIM);
        const scalar_t* k_row = key + ((size_t)seq_idx * num_kv_heads + kv_head) * HEAD_DIM;
        const scalar_t* v_row = value + ((size_t)seq_idx * num_kv_heads + kv_head) * HEAD_DIM;
        const float k_scale_stored = scale_table[((int64_t)block_id * num_kv_heads + kv_head) * psp_kv_num_groups(HEAD_DIM)];
        const float v_scale_stored = scale_table[hh + ((int64_t)block_id * num_kv_heads + kv_head) * psp_kv_num_groups(HEAD_DIM)];
        const float k_scale = (k_scale_stored > 1e-8f) ? k_scale_stored : compute_scale(k_row, k_fmt);
        const float v_scale = (v_scale_stored > 1e-8f) ? v_scale_stored : compute_scale(v_row, v_fmt);
        uint8_t* k_out = key_cache + block_offsets[block_id]
                       + (size_t)kv_head * k_block_bytes + block_offset * k_row_bytes;
        uint8_t* v_out = value_cache + block_offsets[num_blocks + block_id]
                       + (size_t)kv_head * v_block_bytes;

        switch (k_fmt) {
            case 0: psp_quantize_row_k<scalar_t, 0>(k_row, k_out, HEAD_DIM, 0, k_scale); break;
            case 1: psp_quantize_row_k<scalar_t, 1>(k_row, k_out, HEAD_DIM, 0, k_scale); break;
            case 2: psp_quantize_row_k<scalar_t, 2>(k_row, k_out, HEAD_DIM, 0, k_scale); break;
            default: psp_quantize_row_k<scalar_t, 3>(k_row, k_out, HEAD_DIM, 0, k_scale); break;
        }
        switch (v_fmt) {
            case 0: psp_quantize_row_v<scalar_t, 0>(v_row, v_out, HEAD_DIM, HEAD_DIM, 0, v_scale, block_offset); break;
            case 1: psp_quantize_row_v<scalar_t, 1>(v_row, v_out, HEAD_DIM, HEAD_DIM, 0, v_scale, block_offset); break;
            case 2: psp_quantize_row_v<scalar_t, 2>(v_row, v_out, HEAD_DIM, HEAD_DIM, 0, v_scale, block_offset); break;
            default: psp_quantize_row_v<scalar_t, 3>(v_row, v_out, HEAD_DIM, HEAD_DIM, 0, v_scale, block_offset); break;
        }
    }
}

// ============================================================================
// fp8 (e4m3/e5m2) tensor-core decode path.
//
// The KV byte buffer stores fp8 (1 byte/elem); this kernel converts each block
// to fp16 in shared memory via the vectorized __nv_cvt_fp8x2_to_halfraw2
// (1 cvt per 2 elements, exact: fp8's 3 mantissa bits fit in fp16's 10) and
// runs the fp16 WMMA m16n16k16 tiles. The per-(block, kv_head) scale is folded
// into the fp32 mma accumulator to preserve the fp32-scalar reference.
// ============================================================================

template <int HEAD_DIM, typename scalar_t>
__global__ void paged_attention_enhanced_fp8_tc_kernel(
    float* __restrict__ partials,
    const scalar_t* __restrict__ q,
    const uint8_t* __restrict__ k_cache,
    const uint8_t* __restrict__ v_cache,
    const int* __restrict__ block_tables,
    const int* __restrict__ context_lens,
    const uint32_t* __restrict__ block_offsets,
    const float* __restrict__ scale_table,  // [2, num_blocks, num_kv_heads]
    int max_blocks_per_seq,
    int num_heads,
    int num_kv_heads,
    float scale,
    int num_splits,
    int fp8_fmt,         // 1=e4m3, 2=e5m2
    int num_total_blocks)
{
    constexpr int NT8 = HEAD_DIM / 16;
    constexpr int BLOCK_BYTES = BLOCK_SIZE * HEAD_DIM;  // fp8 = 1 byte/elem

    const int kv_head = blockIdx.x;
    const int seq_idx = blockIdx.y;
    const int split_idx = blockIdx.z;
    const int gqa = num_heads / num_kv_heads;
    const int tid = threadIdx.x;
    const int lane = tid & 31, group = lane >> 2, tig = lane & 3;
    const float log2e = 1.4426950408889634f;

    __shared__ __half s_q[16 * HEAD_DIM];
    __shared__ __half sk[2][16 * HEAD_DIM];
    __shared__ __half sv[2][16 * HEAD_DIM];
    __shared__ float s_score[16 * 16];
    __shared__ __half s_prob[16 * 16];
    __shared__ __half fp4_lut[16];

    fragment<matrix_a, 16, 16, 16, __half, row_major> a[NT8];
    fragment<matrix_b, 16, 16, 16, __half, col_major> b_frag;
    fragment<matrix_b, 16, 16, 16, __half, row_major> vb;
    fragment<matrix_a, 16, 16, 16, __half, row_major> pa;
    fragment<accumulator, 16, 16, 16, float> c;
    fragment<accumulator, 16, 16, 16, float> acc[NT8];

    // Padded Q: rows [0, gqa) = the query heads, rest zero.
    for (int i = tid; i < 16 * HEAD_DIM; i += 32) s_q[i] = __float2half(0.0f);
    __syncthreads();
    const int head_base = kv_head * gqa;
    for (int i = tid; i < gqa * HEAD_DIM; i += 32) {
        int h = head_base + i / HEAD_DIM;
        int d = i % HEAD_DIM;
        s_q[i] = __float2half(psp_to_float<scalar_t>(
            q[(int64_t)seq_idx * num_heads * HEAD_DIM + (int64_t)h * HEAD_DIM + d]));
    }
    #pragma unroll
    for (int kt = 0; kt < NT8; kt++) load_matrix_sync(a[kt], s_q + kt * 16, HEAD_DIM);
    #pragma unroll
    for (int nt8 = 0; nt8 < NT8; nt8++) fill_fragment(acc[nt8], 0.0f);

    float rmax0 = -1e30f, rmax1 = -1e30f, rsum0 = 0.0f, rsum1 = 0.0f;

    // fp4 nibble LUT (16 half values): replaces the scalar dequantize_fp4_e2m1
    // arithmetic with a shared-memory lookup in the K/V dequant loop.
    if (tid < 16) fp4_lut[tid] = __float2half(dequantize_fp4_e2m1((uint8_t)tid));
    __syncthreads();

    const int context_len = context_lens[seq_idx];
    const int num_blocks = (context_len + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int blocks_per_split = (num_blocks + num_splits - 1) / num_splits;
    const int split_start = split_idx * blocks_per_split;
    const int split_end = min(split_start + blocks_per_split, num_blocks);
    const int chunk_blocks = split_end - split_start;

    const int block_bytes_per_head = BLOCK_BYTES;

    const __nv_fp8_interpretation_t interp = (fp8_fmt == 2) ? __NV_E5M2 : __NV_E4M3;

    // Prologue: prefetch block 0 into the first buffer. Guarded so empty
    // splits (num_splits > num_blocks) never dereference an out-of-range block.
    if (chunk_blocks > 0) {
        const int pb0 = block_tables[seq_idx * max_blocks_per_seq + split_start];
        const uint8_t* k0 = k_cache + block_offsets[pb0] + (size_t)kv_head * block_bytes_per_head;
        const uint8_t* v0 = v_cache + block_offsets[num_total_blocks + pb0] + (size_t)kv_head * block_bytes_per_head;
        cp_async_tc(reinterpret_cast<uint8_t*>(sk[0]), k0, tid, BLOCK_BYTES);
        cp_async_tc(reinterpret_cast<uint8_t*>(sv[0]), v0, tid, BLOCK_BYTES);
    }

    for (int n = 0; n < chunk_blocks; n++) {
        const int b = split_start + n;
        const int cur = n & 1;
        const int nxt = cur ^ 1;

        // Prefetch block n+1 (2 async groups) so it overlaps the current
        // compute.
        if (n + 1 < chunk_blocks) {
            const int pb1 = block_tables[seq_idx * max_blocks_per_seq + b + 1];
            const uint8_t* k1 = k_cache + block_offsets[pb1] + (size_t)kv_head * block_bytes_per_head;
            const uint8_t* v1 = v_cache + block_offsets[num_total_blocks + pb1] + (size_t)kv_head * block_bytes_per_head;
            cp_async_tc(reinterpret_cast<uint8_t*>(sk[nxt]), k1, tid, BLOCK_BYTES);
            cp_async_tc(reinterpret_cast<uint8_t*>(sv[nxt]), v1, tid, BLOCK_BYTES);
        }

        // Wait for the current block's copy, keeping the 2 prefetch groups
        // outstanding so they overlap the compute below.
        if (n + 1 < chunk_blocks) {
            asm volatile("cp.async.wait_group 2;");
        } else {
            asm volatile("cp.async.wait_group 0;");
        }
        __syncthreads();

        const int start_token = b * BLOCK_SIZE;
        const int tokens_in_block = min(BLOCK_SIZE, context_len - start_token);
        const int physical_block = block_tables[seq_idx * max_blocks_per_seq + b];
        const float k_scale = scale_table[((int64_t)physical_block * num_kv_heads + kv_head) * psp_kv_num_groups(HEAD_DIM)];
        const float v_scale = scale_table[((int64_t)num_total_blocks * num_kv_heads
                                          + (int64_t)physical_block * num_kv_heads + kv_head) * psp_kv_num_groups(HEAD_DIM)];

        // fp8 -> fp16 dequant in place (reverse order so the 2x fp16 write
        // never clobbers a not-yet-read fp8).
        uint8_t* skv = reinterpret_cast<uint8_t*>(sk[cur]);
        uint8_t* svv = reinterpret_cast<uint8_t*>(sv[cur]);
        for (int p = (BLOCK_SIZE * HEAD_DIM / 2) - 1 - tid; p >= 0; p -= 32) {
            __nv_fp8x2_storage_t kb = *reinterpret_cast<const __nv_fp8x2_storage_t*>(skv + p * 2);
            __nv_fp8x2_storage_t vb = *reinterpret_cast<const __nv_fp8x2_storage_t*>(svv + p * 2);
            __half2 kh = __half2(__nv_cvt_fp8x2_to_halfraw2(kb, interp));
            __half2 vh = __half2(__nv_cvt_fp8x2_to_halfraw2(vb, interp));
            sk[cur][p * 2 + 0] = kh.x;
            sk[cur][p * 2 + 1] = kh.y;
            sv[cur][p * 2 + 0] = vh.x;
            sv[cur][p * 2 + 1] = vh.y;
        }
        __syncthreads();

        // QK -> scores [16, 16], then fold the fp32 softmax + K scales.
        fill_fragment(c, 0.0f);
        #pragma unroll
        for (int kt = 0; kt < NT8; kt++) {
            load_matrix_sync(b_frag, sk[cur] + kt * 16, HEAD_DIM);
            mma_sync(c, a[kt], b_frag, c);
        }
        const float qk_scale = scale * k_scale;
        for (int i = 0; i < c.num_elements; i++) c.x[i] *= qk_scale;
        store_matrix_sync(s_score, c, 16, mem_row_major);
        __syncthreads();

        // per-row online softmax (2 rows per thread), masked to valid tokens
        for (int r = group; r < 16; r += 8) {
            float mx = -1e30f;
            for (int cc = 0; cc < tokens_in_block; cc++) mx = fmaxf(mx, s_score[r * 16 + cc]);
            float& rmax = (r == group) ? rmax0 : rmax1;
            float& rsum = (r == group) ? rsum0 : rsum1;
            float m_new = fmaxf(rmax, mx);
            float rescale_old = exp2f((rmax - m_new) * log2e);
            float rescale_new = exp2f((mx - m_new) * log2e);
            rsum *= rescale_old;
            float tsum = 0.0f;
            for (int cc = 0; cc < 16; cc++) {
                float e = (cc < tokens_in_block)
                    ? exp2f((s_score[r * 16 + cc] - mx) * log2e) : 0.0f;
                s_prob[r * 16 + cc] = __float2half(e * rescale_new);
                tsum += e;
            }
            rsum += tsum * rescale_new;
            rmax = m_new;
            #pragma unroll
            for (int nt8 = 0; nt8 < NT8; nt8++) {
                if (r == group) {
                    acc[nt8].x[0] *= rescale_old; acc[nt8].x[1] *= rescale_old;
                    acc[nt8].x[4] *= rescale_old; acc[nt8].x[5] *= rescale_old;
                } else {
                    acc[nt8].x[2] *= rescale_old; acc[nt8].x[3] *= rescale_old;
                    acc[nt8].x[6] *= rescale_old; acc[nt8].x[7] *= rescale_old;
                }
            }
        }
        __syncthreads();

        // PV: acc += (prob @ V_tile) * v_scale (fp32 V scale applied after the
        // mma to preserve the reference precision).
        load_matrix_sync(pa, s_prob, 16);
        #pragma unroll
        for (int nt8 = 0; nt8 < NT8; nt8++) {
            fill_fragment(c, 0.0f);
            load_matrix_sync(vb, sv[cur] + nt8 * 16, HEAD_DIM);
            mma_sync(c, pa, vb, c);
            for (int i = 0; i < c.num_elements; i++) acc[nt8].x[i] += c.x[i] * v_scale;
        }
        __syncthreads();
    }

    // Output partials: per (kv_head, seq, split, head, dim).
    const int64_t p_base =
        (((int64_t)kv_head * gridDim.y + seq_idx) * num_splits + split_idx) * gqa + group;
    #pragma unroll
    for (int nt8 = 0; nt8 < NT8; nt8++) {
        int c0 = nt8 * 16 + tig * 2, c1 = nt8 * 16 + 8 + tig * 2;
        if (group < gqa) {
            float* p = partials + p_base * (2 + HEAD_DIM);
            p[2 + c0]      = acc[nt8].x[0];
            p[2 + c0 + 1]  = acc[nt8].x[1];
            p[2 + c1]      = acc[nt8].x[4];
            p[2 + c1 + 1]  = acc[nt8].x[5];
        }
    }
    if (tig == 0 && group < gqa) {
        partials[p_base * (2 + HEAD_DIM) + 0] = rmax0;
        partials[p_base * (2 + HEAD_DIM) + 1] = rsum0;
    }
}

// ============================================================================
// Native-fp8 tensor-core decode (SM120a): uniform fp8 K/V fed to the tensor
// cores directly via m16n8k32 fp8 MMA. Q is quantized to fp8 and stored
// TRANSPOSED in smem so ldmatrix.m16n16.x2.trans.b8 produces the A operand
// with no scalar loads. The PV stays on fp16 WMMA for now (V dequant).
// ============================================================================
template <int HEAD_DIM, typename scalar_t>
__global__ void paged_attention_enhanced_fp8_native_tc_kernel(
    float* __restrict__ partials,
    const scalar_t* __restrict__ q,
    const uint8_t* __restrict__ k_cache,
    const uint8_t* __restrict__ v_cache,
    const int* __restrict__ block_tables,
    const int* __restrict__ context_lens,
    const uint32_t* __restrict__ block_offsets,
    const float* __restrict__ scale_table,  // [2, num_blocks, num_kv_heads]
    int max_blocks_per_seq,
    int num_heads,
    int num_kv_heads,
    float scale,
    int num_splits,
    int fp8_fmt,         // 1=e4m3, 2=e5m2
    int num_total_blocks)
{
    constexpr int NT8 = HEAD_DIM / 16;
    constexpr int BLOCK_BYTES = BLOCK_SIZE * HEAD_DIM;  // fp8 = 1 byte/elem

    const int kv_head = blockIdx.x;
    const int seq_idx = blockIdx.y;
    const int split_idx = blockIdx.z;
    const int gqa = num_heads / num_kv_heads;
    const int tid = threadIdx.x;
    const int lane = tid & 31, group = lane >> 2, tig = lane & 3;
    const float log2e = 1.4426950408889634f;

    __shared__ __half sk[2][16 * HEAD_DIM];   // raw fp8 K (BLOCK_BYTES used)
    __shared__ __half sv[2][16 * HEAD_DIM];   // fp8 V -> fp16 dequant
    __shared__ float s_score[16 * 16];
    __shared__ __half s_prob[16 * 16];

    fragment<matrix_b, 16, 16, 16, __half, row_major> vb;
    fragment<matrix_a, 16, 16, 16, __half, row_major> pa;
    fragment<accumulator, 16, 16, 16, float> c;
    fragment<accumulator, 16, 16, 16, float> acc[NT8];
    #pragma unroll
    for (int nt8 = 0; nt8 < NT8; nt8++) fill_fragment(acc[nt8], 0.0f);

    // Register-resident Q quantization: quantize the A-fragment elements
    // directly into the interleaved mma layout (no smem round-trip), reducing
    // the per-head max over the 4 lanes sharing a head with warp shuffles.
    // This avoids the 16-way bank conflict of a transposed smem write.
    constexpr int NT32 = HEAD_DIM / 32;
    const int head_base = kv_head * gqa;
    const int h_lo = head_base + group;
    const int h_hi = head_base + group + 8;
    const bool lo_valid = (group < gqa);
    const bool hi_valid = (group + 8 < gqa);
    const int64_t q_base = (int64_t)seq_idx * num_heads * HEAD_DIM;

    float qmax_lo = 0.0f, qmax_hi = 0.0f;
    #pragma unroll
    for (int kt = 0; kt < NT32; kt++) {
        const int d0 = kt * 32 + 4 * tig;
        if (lo_valid) {
            const scalar_t* plo = q + q_base + (int64_t)h_lo * HEAD_DIM + d0;
            qmax_lo = fmaxf(qmax_lo, fabsf(psp_to_float(plo[0])));
            qmax_lo = fmaxf(qmax_lo, fabsf(psp_to_float(plo[1])));
            qmax_lo = fmaxf(qmax_lo, fabsf(psp_to_float(plo[2])));
            qmax_lo = fmaxf(qmax_lo, fabsf(psp_to_float(plo[3])));
            qmax_lo = fmaxf(qmax_lo, fabsf(psp_to_float(plo[16])));
            qmax_lo = fmaxf(qmax_lo, fabsf(psp_to_float(plo[17])));
            qmax_lo = fmaxf(qmax_lo, fabsf(psp_to_float(plo[18])));
            qmax_lo = fmaxf(qmax_lo, fabsf(psp_to_float(plo[19])));
        }
        if (hi_valid) {
            const scalar_t* phi = q + q_base + (int64_t)h_hi * HEAD_DIM + d0;
            qmax_hi = fmaxf(qmax_hi, fabsf(psp_to_float(phi[0])));
            qmax_hi = fmaxf(qmax_hi, fabsf(psp_to_float(phi[1])));
            qmax_hi = fmaxf(qmax_hi, fabsf(psp_to_float(phi[2])));
            qmax_hi = fmaxf(qmax_hi, fabsf(psp_to_float(phi[3])));
            qmax_hi = fmaxf(qmax_hi, fabsf(psp_to_float(phi[16])));
            qmax_hi = fmaxf(qmax_hi, fabsf(psp_to_float(phi[17])));
            qmax_hi = fmaxf(qmax_hi, fabsf(psp_to_float(phi[18])));
            qmax_hi = fmaxf(qmax_hi, fabsf(psp_to_float(phi[19])));
        }
    }
    qmax_lo = fmaxf(qmax_lo, __shfl_xor_sync(0xffffffff, qmax_lo, 1));
    qmax_lo = fmaxf(qmax_lo, __shfl_xor_sync(0xffffffff, qmax_lo, 2));
    qmax_hi = fmaxf(qmax_hi, __shfl_xor_sync(0xffffffff, qmax_hi, 1));
    qmax_hi = fmaxf(qmax_hi, __shfl_xor_sync(0xffffffff, qmax_hi, 2));
    const float s_lo = (qmax_lo < 2.4e-6f) ? 1.0f : qmax_lo / 240.0f;
    const float s_hi = (qmax_hi < 2.4e-6f) ? 1.0f : qmax_hi / 240.0f;

    uint32_t q_a[NT32][4];
    #pragma unroll
    for (int kt = 0; kt < NT32; kt++) {
        const int d0 = kt * 32 + 4 * tig;
        const int d1 = d0 + 16;
        float lo0[4] = {0, 0, 0, 0}, lo1[4] = {0, 0, 0, 0};
        float hi0[4] = {0, 0, 0, 0}, hi1[4] = {0, 0, 0, 0};
        if (lo_valid) {
            const scalar_t* plo = q + q_base + (int64_t)h_lo * HEAD_DIM + d0;
            lo0[0] = psp_to_float(plo[0]); lo0[1] = psp_to_float(plo[1]);
            lo0[2] = psp_to_float(plo[2]); lo0[3] = psp_to_float(plo[3]);
            lo1[0] = psp_to_float(plo[16]); lo1[1] = psp_to_float(plo[17]);
            lo1[2] = psp_to_float(plo[18]); lo1[3] = psp_to_float(plo[19]);
        }
        if (hi_valid) {
            const scalar_t* phi = q + q_base + (int64_t)h_hi * HEAD_DIM + d0;
            hi0[0] = psp_to_float(phi[0]); hi0[1] = psp_to_float(phi[1]);
            hi0[2] = psp_to_float(phi[2]); hi0[3] = psp_to_float(phi[3]);
            hi1[0] = psp_to_float(phi[16]); hi1[1] = psp_to_float(phi[17]);
            hi1[2] = psp_to_float(phi[18]); hi1[3] = psp_to_float(phi[19]);
        }
        q_a[kt][0] = (uint32_t)quantize_fp8(lo0[0] / s_lo, 1)
                   | ((uint32_t)quantize_fp8(lo0[1] / s_lo, 1) << 8)
                   | ((uint32_t)quantize_fp8(lo0[2] / s_lo, 1) << 16)
                   | ((uint32_t)quantize_fp8(lo0[3] / s_lo, 1) << 24);
        q_a[kt][1] = (uint32_t)quantize_fp8(hi0[0] / s_hi, 1)
                   | ((uint32_t)quantize_fp8(hi0[1] / s_hi, 1) << 8)
                   | ((uint32_t)quantize_fp8(hi0[2] / s_hi, 1) << 16)
                   | ((uint32_t)quantize_fp8(hi0[3] / s_hi, 1) << 24);
        q_a[kt][2] = (uint32_t)quantize_fp8(lo1[0] / s_lo, 1)
                   | ((uint32_t)quantize_fp8(lo1[1] / s_lo, 1) << 8)
                   | ((uint32_t)quantize_fp8(lo1[2] / s_lo, 1) << 16)
                   | ((uint32_t)quantize_fp8(lo1[3] / s_lo, 1) << 24);
        q_a[kt][3] = (uint32_t)quantize_fp8(hi1[0] / s_hi, 1)
                   | ((uint32_t)quantize_fp8(hi1[1] / s_hi, 1) << 8)
                   | ((uint32_t)quantize_fp8(hi1[2] / s_hi, 1) << 16)
                   | ((uint32_t)quantize_fp8(hi1[3] / s_hi, 1) << 24);
    }

    float rmax0 = -1e30f, rmax1 = -1e30f, rsum0 = 0.0f, rsum1 = 0.0f;

    const int context_len = context_lens[seq_idx];
    const int num_blocks = (context_len + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int blocks_per_split = (num_blocks + num_splits - 1) / num_splits;
    const int split_start = split_idx * blocks_per_split;
    const int split_end = min(split_start + blocks_per_split, num_blocks);
    const int chunk_blocks = split_end - split_start;

    const int block_bytes_per_head = BLOCK_BYTES;

    const __nv_fp8_interpretation_t interp = (fp8_fmt == 2) ? __NV_E5M2 : __NV_E4M3;

    if (chunk_blocks > 0) {
        const int pb0 = block_tables[seq_idx * max_blocks_per_seq + split_start];
        const uint8_t* k0 = k_cache + block_offsets[pb0] + (size_t)kv_head * block_bytes_per_head;
        const uint8_t* v0 = v_cache + block_offsets[num_total_blocks + pb0] + (size_t)kv_head * block_bytes_per_head;
        cp_async_tc(reinterpret_cast<uint8_t*>(sk[0]), k0, tid, BLOCK_BYTES);
        cp_async_tc(reinterpret_cast<uint8_t*>(sv[0]), v0, tid, BLOCK_BYTES);
    }

    for (int n = 0; n < chunk_blocks; n++) {
        const int b = split_start + n;
        const int cur = n & 1;
        const int nxt = cur ^ 1;

        if (n + 1 < chunk_blocks) {
            const int pb1 = block_tables[seq_idx * max_blocks_per_seq + b + 1];
            const uint8_t* k1 = k_cache + block_offsets[pb1] + (size_t)kv_head * block_bytes_per_head;
            const uint8_t* v1 = v_cache + block_offsets[num_total_blocks + pb1] + (size_t)kv_head * block_bytes_per_head;
            cp_async_tc(reinterpret_cast<uint8_t*>(sk[nxt]), k1, tid, BLOCK_BYTES);
            cp_async_tc(reinterpret_cast<uint8_t*>(sv[nxt]), v1, tid, BLOCK_BYTES);
        }
        if (n + 1 < chunk_blocks) {
            asm volatile("cp.async.wait_group 2;");
        } else {
            asm volatile("cp.async.wait_group 0;");
        }
        __syncthreads();

        const int start_token = b * BLOCK_SIZE;
        const int tokens_in_block = min(BLOCK_SIZE, context_len - start_token);
        const int physical_block = block_tables[seq_idx * max_blocks_per_seq + b];
        const float k_scale = scale_table[((int64_t)physical_block * num_kv_heads + kv_head) * psp_kv_num_groups(HEAD_DIM)];
        const float v_scale = scale_table[((int64_t)num_total_blocks * num_kv_heads
                                          + (int64_t)physical_block * num_kv_heads + kv_head) * psp_kv_num_groups(HEAD_DIM)];

        // fp8 -> fp16 dequant of V only (K stays raw fp8 for the native MMA).
        uint8_t* svv = reinterpret_cast<uint8_t*>(sv[cur]);
        for (int p = (BLOCK_SIZE * HEAD_DIM / 2) - 1 - tid; p >= 0; p -= 32) {
            __nv_fp8x2_storage_t vb2 = *reinterpret_cast<const __nv_fp8x2_storage_t*>(svv + p * 2);
            __half2 vh = __half2(__nv_cvt_fp8x2_to_halfraw2(vb2, interp));
            sv[cur][p * 2 + 0] = vh.x;
            sv[cur][p * 2 + 1] = vh.y;
        }
        __syncthreads();

        // QK native fp8 MMA: S = Q_fp8 @ K_fp8. A is register-resident (q_a);
        // B loaded with the interleaved scalar layout.
        uint8_t* skv = reinterpret_cast<uint8_t*>(sk[cur]);
        float score[2][4];
        #pragma unroll
        for (int nt = 0; nt < 2; nt++)
            #pragma unroll
            for (int i = 0; i < 4; i++) score[nt][i] = 0.0f;
        #pragma unroll
        for (int kt = 0; kt < NT32; kt++) {
            #pragma unroll
            for (int nt = 0; nt < 2; nt++) {
                uint32_t kb[2];
                psp_fp8_load_b(kb, skv + nt * 8 * HEAD_DIM + kt * 32, lane, HEAD_DIM);
                psp_fp8_mma_m16n8k32(score[nt], q_a[kt], kb, score[nt]);
            }
        }
        // Fold softmax_scale * k_scale * per-head q_scale into the scores.
        const float qk0 = scale * k_scale * s_lo;
        const float qk1 = scale * k_scale * s_hi;
        #pragma unroll
        for (int nt = 0; nt < 2; nt++) {
            score[nt][0] *= qk0; score[nt][1] *= qk0;
            score[nt][2] *= qk1; score[nt][3] *= qk1;
            psp_fp8_store_c(score[nt], s_score + nt * 8, lane, 16);
        }
        __syncthreads();

        // per-row online softmax (2 rows per thread), masked to valid tokens
        for (int r = group; r < 16; r += 8) {
            float mx = -1e30f;
            for (int cc = 0; cc < tokens_in_block; cc++) mx = fmaxf(mx, s_score[r * 16 + cc]);
            float& rmax = (r == group) ? rmax0 : rmax1;
            float& rsum = (r == group) ? rsum0 : rsum1;
            float m_new = fmaxf(rmax, mx);
            float rescale_old = exp2f((rmax - m_new) * log2e);
            float rescale_new = exp2f((mx - m_new) * log2e);
            rsum *= rescale_old;
            float tsum = 0.0f;
            for (int cc = 0; cc < 16; cc++) {
                float e = (cc < tokens_in_block)
                    ? exp2f((s_score[r * 16 + cc] - mx) * log2e) : 0.0f;
                s_prob[r * 16 + cc] = __float2half(e * rescale_new);
                tsum += e;
            }
            rsum += tsum * rescale_new;
            rmax = m_new;
            #pragma unroll
            for (int nt8 = 0; nt8 < NT8; nt8++) {
                if (r == group) {
                    acc[nt8].x[0] *= rescale_old; acc[nt8].x[1] *= rescale_old;
                    acc[nt8].x[4] *= rescale_old; acc[nt8].x[5] *= rescale_old;
                } else {
                    acc[nt8].x[2] *= rescale_old; acc[nt8].x[3] *= rescale_old;
                    acc[nt8].x[6] *= rescale_old; acc[nt8].x[7] *= rescale_old;
                }
            }
        }
        __syncthreads();

        // PV: acc += (prob @ V_tile) * v_scale (fp16 WMMA, unchanged).
        load_matrix_sync(pa, s_prob, 16);
        #pragma unroll
        for (int nt8 = 0; nt8 < NT8; nt8++) {
            fill_fragment(c, 0.0f);
            load_matrix_sync(vb, sv[cur] + nt8 * 16, HEAD_DIM);
            mma_sync(c, pa, vb, c);
            for (int i = 0; i < c.num_elements; i++) acc[nt8].x[i] += c.x[i] * v_scale;
        }
        __syncthreads();
    }

    // Output partials: per (kv_head, seq, split, head, dim).
    const int64_t p_base =
        (((int64_t)kv_head * gridDim.y + seq_idx) * num_splits + split_idx) * gqa + group;
    #pragma unroll
    for (int nt8 = 0; nt8 < NT8; nt8++) {
        int c0 = nt8 * 16 + tig * 2, c1 = nt8 * 16 + 8 + tig * 2;
        if (group < gqa) {
            float* p = partials + p_base * (2 + HEAD_DIM);
            p[2 + c0]      = acc[nt8].x[0];
            p[2 + c0 + 1]  = acc[nt8].x[1];
            p[2 + c1]      = acc[nt8].x[4];
            p[2 + c1 + 1]  = acc[nt8].x[5];
        }
    }
    if (tig == 0 && group < gqa) {
        partials[p_base * (2 + HEAD_DIM) + 0] = rmax0;
        partials[p_base * (2 + HEAD_DIM) + 1] = rsum0;
    }
}

// ============================================================================
// Per-K/V tensor-core decode: K fp8 (e4m3/e5m2) + V fp4 (e2m1). Same fp16
// m16n16k16 tiles as the fp8 path, but the V is dequantized from fp4 (2 elems
// per byte) with its own byte stride and its own per-(block, head) scale.
// ============================================================================
template <int HEAD_DIM, typename scalar_t>
__global__ void paged_attention_enhanced_fp8pv_tc_kernel(
    scalar_t* __restrict__ out,
    float* __restrict__ partials,
    const scalar_t* __restrict__ q,
    const uint8_t* __restrict__ k_cache,
    const uint8_t* __restrict__ v_cache,
    const int* __restrict__ block_tables,
    const int* __restrict__ context_lens,
    const uint32_t* __restrict__ block_offsets,  // [K | V]
    const float* __restrict__ scale_table,       // [2, num_blocks, num_kv_heads]
    const uint8_t* __restrict__ precision_table, // per-block k_fmt | (v_fmt<<2)
    int max_blocks_per_seq,
    int num_heads,
    int num_kv_heads,
    float scale,
    int num_splits,
    int fp8_fmt,          // 1=e4m3, 2=e5m2 (used only for fp8 K)
    int num_total_blocks)
{
    constexpr int NT8 = HEAD_DIM / 16;
    // K/V byte size per (block, kv_head): fp8 = 1 B/elem, fp4 = 0.5 B/elem.
    constexpr int BLOCK_BYTES_FP8 = BLOCK_SIZE * HEAD_DIM;
    constexpr int BLOCK_BYTES_FP4 = BLOCK_SIZE * HEAD_DIM / 2;

    const int kv_head = blockIdx.x;
    const int seq_idx = blockIdx.y;
    const int split_idx = blockIdx.z;
    const int gqa = num_heads / num_kv_heads;
    const int tid = threadIdx.x;
    const int lane = tid & 31, group = lane >> 2, tig = lane & 3;
    const float log2e = 1.4426950408889634f;

    __shared__ __half s_q[16 * HEAD_DIM];
    __shared__ __half sk[2][16 * HEAD_DIM];
    __shared__ __half sv[2][16 * HEAD_DIM];
    __shared__ float s_score[16 * 16];
    __shared__ __half s_prob[16 * 16];
    __shared__ __half fp4_lut[16];

    fragment<matrix_a, 16, 16, 16, __half, row_major> a[NT8];
    fragment<matrix_b, 16, 16, 16, __half, col_major> b_frag;
    fragment<matrix_b, 16, 16, 16, __half, row_major> vb;
    fragment<matrix_a, 16, 16, 16, __half, row_major> pa;
    fragment<accumulator, 16, 16, 16, float> c;
    fragment<accumulator, 16, 16, 16, float> acc[NT8];

    for (int i = tid; i < 16 * HEAD_DIM; i += 32) s_q[i] = __float2half(0.0f);
    __syncthreads();
    const int head_base = kv_head * gqa;
    for (int i = tid; i < gqa * HEAD_DIM; i += 32) {
        int h = head_base + i / HEAD_DIM;
        int d = i % HEAD_DIM;
        s_q[i] = __float2half(psp_to_float<scalar_t>(
            q[(int64_t)seq_idx * num_heads * HEAD_DIM + (int64_t)h * HEAD_DIM + d]));
    }
    #pragma unroll
    for (int kt = 0; kt < NT8; kt++) load_matrix_sync(a[kt], s_q + kt * 16, HEAD_DIM);
    #pragma unroll
    for (int nt8 = 0; nt8 < NT8; nt8++) fill_fragment(acc[nt8], 0.0f);

    float rmax0 = -1e30f, rmax1 = -1e30f, rsum0 = 0.0f, rsum1 = 0.0f;

    // fp4 nibble LUT (16 half values): replaces the scalar dequantize_fp4_e2m1
    // arithmetic with a shared-memory lookup in the K/V dequant loop.
    if (tid < 16) fp4_lut[tid] = __float2half(dequantize_fp4_e2m1((uint8_t)tid));
    __syncthreads();

    const int context_len = context_lens[seq_idx];
    const int num_blocks = (context_len + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int blocks_per_split = (num_blocks + num_splits - 1) / num_splits;
    const int split_start = split_idx * blocks_per_split;
    const int split_end = min(split_start + blocks_per_split, num_blocks);
    const int chunk_blocks = split_end - split_start;

    // Prologue: prefetch block 0. K/V byte sizes depend on that block's formats.
    // Guarded so empty splits (num_splits > num_blocks) never dereference a
    // block table entry past the sequence's context.
    if (chunk_blocks > 0) {
        const int pb0 = block_tables[seq_idx * max_blocks_per_seq + split_start];
        const int p0_packed = (int)precision_table[pb0];
        const int p0_kfmt = psp_kv_fmt_k(p0_packed);
        const int p0_vfmt = psp_kv_fmt_v(p0_packed);
        const int kb0 = (p0_kfmt == 3) ? BLOCK_BYTES_FP4 : BLOCK_BYTES_FP8;
        const int vb0 = (p0_vfmt == 3) ? BLOCK_BYTES_FP4 : BLOCK_BYTES_FP8;
        const uint8_t* k0 = k_cache + block_offsets[pb0]
                          + (size_t)kv_head * kb0;
        const uint8_t* v0 = v_cache + block_offsets[num_total_blocks + pb0]
                          + (size_t)kv_head * vb0;
        cp_async_tc(reinterpret_cast<uint8_t*>(sk[0]), k0, tid, kb0);
        cp_async_tc(reinterpret_cast<uint8_t*>(sv[0]), v0, tid, vb0);
    }

    for (int n = 0; n < chunk_blocks; n++) {
        const int b = split_start + n;
        const int cur = n & 1;
        const int nxt = cur ^ 1;

        if (n + 1 < chunk_blocks) {
            const int pb1 = block_tables[seq_idx * max_blocks_per_seq + b + 1];
            const int p1_packed = (int)precision_table[pb1];
            const int p1_kfmt = psp_kv_fmt_k(p1_packed);
            const int p1_vfmt = psp_kv_fmt_v(p1_packed);
            const int kb1 = (p1_kfmt == 3) ? BLOCK_BYTES_FP4 : BLOCK_BYTES_FP8;
            const int vb1 = (p1_vfmt == 3) ? BLOCK_BYTES_FP4 : BLOCK_BYTES_FP8;
            const uint8_t* k1 = k_cache + block_offsets[pb1]
                              + (size_t)kv_head * kb1;
            const uint8_t* v1 = v_cache + block_offsets[num_total_blocks + pb1]
                              + (size_t)kv_head * vb1;
            cp_async_tc(reinterpret_cast<uint8_t*>(sk[nxt]), k1, tid, kb1);
            cp_async_tc(reinterpret_cast<uint8_t*>(sv[nxt]), v1, tid, vb1);
        }

        if (n + 1 < chunk_blocks) {
            asm volatile("cp.async.wait_group 2;");
        } else {
            asm volatile("cp.async.wait_group 0;");
        }
        __syncthreads();

        const int start_token = b * BLOCK_SIZE;
        const int tokens_in_block = min(BLOCK_SIZE, context_len - start_token);
        const int physical_block = block_tables[seq_idx * max_blocks_per_seq + b];
        const int packed = (int)precision_table[physical_block];
        const int k_fmt = psp_kv_fmt_k(packed);
        const int v_fmt = psp_kv_fmt_v(packed);
        const int64_t v_base_scale = (int64_t)num_total_blocks * num_kv_heads * psp_kv_num_groups(HEAD_DIM);
        const float k_scale = scale_table[((int64_t)physical_block * num_kv_heads + kv_head) * psp_kv_num_groups(HEAD_DIM)];
        const float v_scale = scale_table[v_base_scale + ((int64_t)physical_block * num_kv_heads + kv_head) * psp_kv_num_groups(HEAD_DIM)];

        // K/V -> fp16 in place. fp4 uses the shared-memory 16-entry nibble LUT
        // (2 LUT loads/byte) instead of the scalar dequantize_fp4_e2m1; fp8 uses
        // the hardware fp8x4 convert.
        uint8_t* skv = reinterpret_cast<uint8_t*>(sk[cur]);
        uint8_t* svv = reinterpret_cast<uint8_t*>(sv[cur]);
        for (int p = (BLOCK_SIZE * HEAD_DIM / 4) - 1 - tid; p >= 0; p -= 32) {
            if (k_fmt == 3) {
                uint16_t kb = *reinterpret_cast<const uint16_t*>(skv + p * 2);
                uint8_t b0 = (uint8_t)(kb & 0xFFu);
                uint8_t b1 = (uint8_t)(kb >> 8);
                sk[cur][p * 4 + 0] = fp4_lut[b0 & 0x0F];
                sk[cur][p * 4 + 1] = fp4_lut[b0 >> 4];
                sk[cur][p * 4 + 2] = fp4_lut[b1 & 0x0F];
                sk[cur][p * 4 + 3] = fp4_lut[b1 >> 4];
            } else {
                float4 k4 = psp_packed_dequant4<scalar_t>(skv, p * 4, k_fmt);
                sk[cur][p * 4 + 0] = __float2half(k4.x);
                sk[cur][p * 4 + 1] = __float2half(k4.y);
                sk[cur][p * 4 + 2] = __float2half(k4.z);
                sk[cur][p * 4 + 3] = __float2half(k4.w);
            }
            if (v_fmt == 3) {
                uint16_t vb = *reinterpret_cast<const uint16_t*>(svv + p * 2);
                uint8_t b0 = (uint8_t)(vb & 0xFFu);
                uint8_t b1 = (uint8_t)(vb >> 8);
                sv[cur][p * 4 + 0] = fp4_lut[b0 & 0x0F];
                sv[cur][p * 4 + 1] = fp4_lut[b0 >> 4];
                sv[cur][p * 4 + 2] = fp4_lut[b1 & 0x0F];
                sv[cur][p * 4 + 3] = fp4_lut[b1 >> 4];
            } else {
                float4 v4 = psp_packed_dequant4<scalar_t>(svv, p * 4, v_fmt);
                sv[cur][p * 4 + 0] = __float2half(v4.x);
                sv[cur][p * 4 + 1] = __float2half(v4.y);
                sv[cur][p * 4 + 2] = __float2half(v4.z);
                sv[cur][p * 4 + 3] = __float2half(v4.w);
            }
        }
        __syncthreads();

        // QK -> scores [16, 16], then fold the fp32 softmax + K scales.
        fill_fragment(c, 0.0f);
        #pragma unroll
        for (int kt = 0; kt < NT8; kt++) {
            load_matrix_sync(b_frag, sk[cur] + kt * 16, HEAD_DIM);
            mma_sync(c, a[kt], b_frag, c);
        }
        const float qk_scale = scale * k_scale;
        for (int i = 0; i < c.num_elements; i++) c.x[i] *= qk_scale;
        store_matrix_sync(s_score, c, 16, mem_row_major);
        __syncthreads();

        for (int r = group; r < 16; r += 8) {
            float mx = -1e30f;
            for (int cc = 0; cc < tokens_in_block; cc++) mx = fmaxf(mx, s_score[r * 16 + cc]);
            float& rmax = (r == group) ? rmax0 : rmax1;
            float& rsum = (r == group) ? rsum0 : rsum1;
            float m_new = fmaxf(rmax, mx);
            float rescale_old = exp2f((rmax - m_new) * log2e);
            float rescale_new = exp2f((mx - m_new) * log2e);
            rsum *= rescale_old;
            float tsum = 0.0f;
            for (int cc = 0; cc < 16; cc++) {
                float e = (cc < tokens_in_block)
                    ? exp2f((s_score[r * 16 + cc] - mx) * log2e) : 0.0f;
                s_prob[r * 16 + cc] = __float2half(e * rescale_new);
                tsum += e;
            }
            rsum += tsum * rescale_new;
            rmax = m_new;
            #pragma unroll
            for (int nt8 = 0; nt8 < NT8; nt8++) {
                if (r == group) {
                    acc[nt8].x[0] *= rescale_old; acc[nt8].x[1] *= rescale_old;
                    acc[nt8].x[4] *= rescale_old; acc[nt8].x[5] *= rescale_old;
                } else {
                    acc[nt8].x[2] *= rescale_old; acc[nt8].x[3] *= rescale_old;
                    acc[nt8].x[6] *= rescale_old; acc[nt8].x[7] *= rescale_old;
                }
            }
        }
        __syncthreads();

        // PV: acc += (prob @ V_tile) * v_scale.
        load_matrix_sync(pa, s_prob, 16);
        #pragma unroll
        for (int nt8 = 0; nt8 < NT8; nt8++) {
            fill_fragment(c, 0.0f);
            load_matrix_sync(vb, sv[cur] + nt8 * 16, HEAD_DIM);
            mma_sync(c, pa, vb, c);
            for (int i = 0; i < c.num_elements; i++) acc[nt8].x[i] += c.x[i] * v_scale;
        }
        __syncthreads();
    }

    // Output partials: per (kv_head, seq, split, head, dim).
    const int64_t p_base =
        (((int64_t)kv_head * gridDim.y + seq_idx) * num_splits + split_idx) * gqa + group;
    #pragma unroll
    for (int nt8 = 0; nt8 < NT8; nt8++) {
        int c0 = nt8 * 16 + tig * 2, c1 = nt8 * 16 + 8 + tig * 2;
        if (group < gqa) {
            float* p = partials + p_base * (2 + HEAD_DIM);
            p[2 + c0]      = acc[nt8].x[0];
            p[2 + c0 + 1]  = acc[nt8].x[1];
            p[2 + c1]      = acc[nt8].x[4];
            p[2 + c1 + 1]  = acc[nt8].x[5];
        }
    }
    if (tig == 0 && group < gqa) {
        partials[p_base * (2 + HEAD_DIM) + 0] = rmax0;
        partials[p_base * (2 + HEAD_DIM) + 1] = rsum0;
    }
}

// ============================================================================
// Multi-token per-K/V tensor-core attention (batched-token path). Same fp8/fp4
// in-kernel dequant + fp16 m16n16k16 MMA as the decode kernel, but N query
// tokens per sequence share one K/V block read + dequant, amortizing the fp4
// dequant across tokens (the speculative-verify / mini-prefill shape).
// ============================================================================
template <int HEAD_DIM, int MAX_Q, typename scalar_t>
__global__ void paged_attention_enhanced_fp8pv_mt_kernel(
    scalar_t* __restrict__ out,               // [num_tokens, num_heads, HEAD_DIM]
    const scalar_t* __restrict__ q,           // [num_tokens, num_heads, HEAD_DIM]
    const scalar_t* __restrict__ key,         // [num_tokens, num_kv_heads, HEAD_DIM]
    const scalar_t* __restrict__ value,       // [num_tokens, num_kv_heads, HEAD_DIM]
    const uint8_t* __restrict__ k_cache,
    const uint8_t* __restrict__ v_cache,
    const int* __restrict__ block_tables,
    const int* __restrict__ context_lens,
    const uint32_t* __restrict__ block_offsets,
    const float* __restrict__ scale_table,
    const uint8_t* __restrict__ precision_table,
    int max_blocks_per_seq,
    int num_heads,
    int num_kv_heads,
    float scale,
    int fp8_fmt,
    int num_total_blocks,
    int num_query_tokens)
{
    constexpr int NT8 = HEAD_DIM / 16;
    constexpr int BLOCK_BYTES_FP8 = BLOCK_SIZE * HEAD_DIM;
    constexpr int BLOCK_BYTES_FP4 = BLOCK_SIZE * HEAD_DIM / 2;
    constexpr int BLOCK_BYTES_LOSSLESS = BLOCK_SIZE * HEAD_DIM * (int)sizeof(scalar_t);
    auto block_bytes = [&](int fmt) -> int {
        if (fmt == 0)      return BLOCK_BYTES_LOSSLESS;
        else if (fmt == 3) return BLOCK_BYTES_FP4;
        else               return BLOCK_BYTES_FP8;
    };

    const int kv_head = blockIdx.x;
    const int seq_idx = blockIdx.y;
    const int gqa = num_heads / num_kv_heads;
    const int tid = threadIdx.x;
    const int lane = tid & 31, group = lane >> 2, tig = lane & 3;
    const float log2e = 1.4426950408889634f;

    __shared__ __half s_q[16 * HEAD_DIM];
    __shared__ __half sk[2][16 * HEAD_DIM];
    __shared__ __half sv[2][16 * HEAD_DIM];
    __shared__ float s_score[16 * 16];
    __shared__ __half s_prob[16 * 16];
    __shared__ __half fp4_lut[16];

    fragment<matrix_a, 16, 16, 16, __half, row_major> a[NT8];
    fragment<matrix_b, 16, 16, 16, __half, col_major> b_frag;
    fragment<matrix_b, 16, 16, 16, __half, row_major> vb;
    fragment<matrix_a, 16, 16, 16, __half, row_major> pa;
    fragment<accumulator, 16, 16, 16, float> c;
    fragment<accumulator, 16, 16, 16, float> acc[MAX_Q][NT8];

    float rmax[MAX_Q][2], rsum[MAX_Q][2];
    #pragma unroll
    for (int t = 0; t < MAX_Q; t++) {
        rmax[t][0] = rmax[t][1] = -1e30f;
        rsum[t][0] = rsum[t][1] = 0.0f;
        #pragma unroll
        for (int nt8 = 0; nt8 < NT8; nt8++) fill_fragment(acc[t][nt8], 0.0f);
    }

    if (tid < 16) fp4_lut[tid] = __float2half(dequantize_fp4_e2m1((uint8_t)tid));
    // Zero the padded Q rows (gqa..15) once; the per-token reload only fills
    // rows 0..gqa-1, and the m16n16k16 MMA reads all 16 rows.
    for (int i = tid; i < 16 * HEAD_DIM; i += 32) s_q[i] = __float2half(0.0f);
    __syncthreads();

    const int context_len = context_lens[seq_idx];
    const int num_blocks = (context_len + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int head_base = kv_head * gqa;

    if (num_blocks > 0) {
        const int pb0 = block_tables[seq_idx * max_blocks_per_seq];
        const int p0_packed = (int)precision_table[pb0];
        const int kb0 = block_bytes(psp_kv_fmt_k(p0_packed));
        const int vb0 = block_bytes(psp_kv_fmt_v(p0_packed));
        cp_async_tc(reinterpret_cast<uint8_t*>(sk[0]),
                    k_cache + block_offsets[pb0] + (size_t)kv_head * kb0, tid, kb0);
        cp_async_tc(reinterpret_cast<uint8_t*>(sv[0]),
                    v_cache + block_offsets[num_total_blocks + pb0] + (size_t)kv_head * vb0, tid, vb0);
    }

    for (int n = 0; n < num_blocks; n++) {
        const int cur = n & 1;
        const int nxt = cur ^ 1;
        if (n + 1 < num_blocks) {
            const int pb1 = block_tables[seq_idx * max_blocks_per_seq + n + 1];
            const int p1_packed = (int)precision_table[pb1];
            const int kb1 = block_bytes(psp_kv_fmt_k(p1_packed));
            const int vb1 = block_bytes(psp_kv_fmt_v(p1_packed));
            cp_async_tc(reinterpret_cast<uint8_t*>(sk[nxt]),
                        k_cache + block_offsets[pb1] + (size_t)kv_head * kb1, tid, kb1);
            cp_async_tc(reinterpret_cast<uint8_t*>(sv[nxt]),
                        v_cache + block_offsets[num_total_blocks + pb1] + (size_t)kv_head * vb1, tid, vb1);
        }
        if (n + 1 < num_blocks) { asm volatile("cp.async.wait_group 2;"); }
        else { asm volatile("cp.async.wait_group 0;"); }
        __syncthreads();

        const int start_token = n * BLOCK_SIZE;
        const int tokens_in_block = min(BLOCK_SIZE, context_len - start_token);
        const int physical_block = block_tables[seq_idx * max_blocks_per_seq + n];
        const int packed = (int)precision_table[physical_block];
        const int k_fmt = psp_kv_fmt_k(packed);
        const int v_fmt = psp_kv_fmt_v(packed);
        const int64_t v_base_scale = (int64_t)num_total_blocks * num_kv_heads * psp_kv_num_groups(HEAD_DIM);
        const float k_scale = scale_table[((int64_t)physical_block * num_kv_heads + kv_head) * psp_kv_num_groups(HEAD_DIM)];
        const float v_scale = scale_table[v_base_scale + ((int64_t)physical_block * num_kv_heads + kv_head) * psp_kv_num_groups(HEAD_DIM)];

        uint8_t* skv = reinterpret_cast<uint8_t*>(sk[cur]);
        uint8_t* svv = reinterpret_cast<uint8_t*>(sv[cur]);
        for (int p = (BLOCK_SIZE * HEAD_DIM / 4) - 1 - tid; p >= 0; p -= 32) {
            if (k_fmt == 3) {
                uint16_t kb = *reinterpret_cast<const uint16_t*>(skv + p * 2);
                uint8_t b0 = (uint8_t)(kb & 0xFFu), b1 = (uint8_t)(kb >> 8);
                sk[cur][p * 4 + 0] = fp4_lut[b0 & 0x0F];
                sk[cur][p * 4 + 1] = fp4_lut[b0 >> 4];
                sk[cur][p * 4 + 2] = fp4_lut[b1 & 0x0F];
                sk[cur][p * 4 + 3] = fp4_lut[b1 >> 4];
            } else {
                float4 k4 = psp_packed_dequant4<scalar_t>(skv, p * 4, k_fmt);
                sk[cur][p * 4 + 0] = __float2half(k4.x);
                sk[cur][p * 4 + 1] = __float2half(k4.y);
                sk[cur][p * 4 + 2] = __float2half(k4.z);
                sk[cur][p * 4 + 3] = __float2half(k4.w);
            }
            if (v_fmt == 3) {
                uint16_t vb = *reinterpret_cast<const uint16_t*>(svv + p * 2);
                uint8_t b0 = (uint8_t)(vb & 0xFFu), b1 = (uint8_t)(vb >> 8);
                sv[cur][p * 4 + 0] = fp4_lut[b0 & 0x0F];
                sv[cur][p * 4 + 1] = fp4_lut[b0 >> 4];
                sv[cur][p * 4 + 2] = fp4_lut[b1 & 0x0F];
                sv[cur][p * 4 + 3] = fp4_lut[b1 >> 4];
            } else {
                float4 v4 = psp_packed_dequant4<scalar_t>(svv, p * 4, v_fmt);
                sv[cur][p * 4 + 0] = __float2half(v4.x);
                sv[cur][p * 4 + 1] = __float2half(v4.y);
                sv[cur][p * 4 + 2] = __float2half(v4.z);
                sv[cur][p * 4 + 3] = __float2half(v4.w);
            }
        }
        __syncthreads();

        #pragma unroll
        for (int t = 0; t < MAX_Q; t++) {
            if (t >= num_query_tokens) break;
            for (int i = tid; i < gqa * HEAD_DIM; i += 32) {
                int h = head_base + i / HEAD_DIM;
                int d = i % HEAD_DIM;
                s_q[i] = __float2half(psp_to_float<scalar_t>(
                    q[((int64_t)seq_idx * num_query_tokens + t) * num_heads * HEAD_DIM
                      + (int64_t)h * HEAD_DIM + d]));
            }
            __syncthreads();
            #pragma unroll
            for (int kt = 0; kt < NT8; kt++) load_matrix_sync(a[kt], s_q + kt * 16, HEAD_DIM);

            fill_fragment(c, 0.0f);
            #pragma unroll
            for (int kt = 0; kt < NT8; kt++) {
                load_matrix_sync(b_frag, sk[cur] + kt * 16, HEAD_DIM);
                mma_sync(c, a[kt], b_frag, c);
            }
            const float qk_scale = scale * k_scale;
            for (int i = 0; i < c.num_elements; i++) c.x[i] *= qk_scale;
            store_matrix_sync(s_score, c, 16, mem_row_major);
            __syncthreads();

            for (int r = group; r < 16; r += 8) {
                float mx = -1e30f;
                for (int cc = 0; cc < tokens_in_block; cc++) mx = fmaxf(mx, s_score[r * 16 + cc]);
                float& rmaxv = (r == group) ? rmax[t][0] : rmax[t][1];
                float& rsumv = (r == group) ? rsum[t][0] : rsum[t][1];
                float m_new = fmaxf(rmaxv, mx);
                float rescale_old = exp2f((rmaxv - m_new) * log2e);
                float rescale_new = exp2f((mx - m_new) * log2e);
                rsumv *= rescale_old;
                float tsum = 0.0f;
                for (int cc = 0; cc < 16; cc++) {
                    float e = (cc < tokens_in_block)
                        ? exp2f((s_score[r * 16 + cc] - mx) * log2e) : 0.0f;
                    s_prob[r * 16 + cc] = __float2half(e * rescale_new);
                    tsum += e;
                }
                rsumv += tsum * rescale_new;
                rmaxv = m_new;
                #pragma unroll
                for (int nt8 = 0; nt8 < NT8; nt8++) {
                    if (r == group) {
                        acc[t][nt8].x[0] *= rescale_old; acc[t][nt8].x[1] *= rescale_old;
                        acc[t][nt8].x[4] *= rescale_old; acc[t][nt8].x[5] *= rescale_old;
                    } else {
                        acc[t][nt8].x[2] *= rescale_old; acc[t][nt8].x[3] *= rescale_old;
                        acc[t][nt8].x[6] *= rescale_old; acc[t][nt8].x[7] *= rescale_old;
                    }
                }
            }
            __syncthreads();

            load_matrix_sync(pa, s_prob, 16);
            #pragma unroll
            for (int nt8 = 0; nt8 < NT8; nt8++) {
                fill_fragment(c, 0.0f);
                load_matrix_sync(vb, sv[cur] + nt8 * 16, HEAD_DIM);
                mma_sync(c, pa, vb, c);
                for (int i = 0; i < c.num_elements; i++) acc[t][nt8].x[i] += c.x[i] * v_scale;
            }
            __syncthreads();
        }
    }

    // Causal draft-draft attention: query token t attends to draft tokens
    // 0..t-1 (the chunk-chunk block missing from the context-only path). Reuses
    // the sk/sv double buffers (free after the context loop); the fp16 draft
    // K/V are stored transposed (dim * 16 + token) to match the context `sk`
    // layout, with token slots [num_query_tokens, 16) zeroed.
    if (num_query_tokens > 1) {
        for (int i = tid; i < 16 * HEAD_DIM; i += 32) {
            sk[0][i] = __float2half(0.0f);
            sv[0][i] = __float2half(0.0f);
        }
        for (int j = 0; j < num_query_tokens; j++) {
            const int64_t db = ((int64_t)seq_idx * num_query_tokens + j) * num_kv_heads * HEAD_DIM
                             + (int64_t)kv_head * HEAD_DIM;
            for (int d = tid; d < HEAD_DIM; d += 32) {
                sk[0][j * HEAD_DIM + d] = __float2half(psp_to_float<scalar_t>(key[db + d]));
                sv[0][j * HEAD_DIM + d] = __float2half(psp_to_float<scalar_t>(value[db + d]));
            }
        }
        __syncthreads();

        #pragma unroll
        for (int t = 0; t < MAX_Q; t++) {
            if (t >= num_query_tokens) break;
            for (int i = tid; i < gqa * HEAD_DIM; i += 32) {
                int h = head_base + i / HEAD_DIM;
                int d = i % HEAD_DIM;
                s_q[i] = __float2half(psp_to_float<scalar_t>(
                    q[((int64_t)seq_idx * num_query_tokens + t) * num_heads * HEAD_DIM
                      + (int64_t)h * HEAD_DIM + d]));
            }
            __syncthreads();
            #pragma unroll
            for (int kt = 0; kt < NT8; kt++) load_matrix_sync(a[kt], s_q + kt * 16, HEAD_DIM);

            fill_fragment(c, 0.0f);
            #pragma unroll
            for (int kt = 0; kt < NT8; kt++) {
                load_matrix_sync(b_frag, sk[0] + kt * 16, HEAD_DIM);
                mma_sync(c, a[kt], b_frag, c);
            }
            for (int i = 0; i < c.num_elements; i++) c.x[i] *= scale;
            store_matrix_sync(s_score, c, 16, mem_row_major);
            __syncthreads();

            for (int r = group; r < 16; r += 8) {
                float mx = -1e30f;
                for (int cc = 0; cc <= t; cc++) mx = fmaxf(mx, s_score[r * 16 + cc]);
                float& rmaxv = (r == group) ? rmax[t][0] : rmax[t][1];
                float& rsumv = (r == group) ? rsum[t][0] : rsum[t][1];
                float m_new = fmaxf(rmaxv, mx);
                float rescale_old = exp2f((rmaxv - m_new) * log2e);
                float rescale_new = exp2f((mx - m_new) * log2e);
                rsumv *= rescale_old;
                float tsum = 0.0f;
                for (int cc = 0; cc < 16; cc++) {
                    float e = (cc <= t) ? exp2f((s_score[r * 16 + cc] - mx) * log2e) : 0.0f;
                    s_prob[r * 16 + cc] = __float2half(e * rescale_new);
                    tsum += e;
                }
                rsumv += tsum * rescale_new;
                rmaxv = m_new;
                #pragma unroll
                for (int nt8 = 0; nt8 < NT8; nt8++) {
                    if (r == group) {
                        acc[t][nt8].x[0] *= rescale_old; acc[t][nt8].x[1] *= rescale_old;
                        acc[t][nt8].x[4] *= rescale_old; acc[t][nt8].x[5] *= rescale_old;
                    } else {
                        acc[t][nt8].x[2] *= rescale_old; acc[t][nt8].x[3] *= rescale_old;
                        acc[t][nt8].x[6] *= rescale_old; acc[t][nt8].x[7] *= rescale_old;
                    }
                }
            }
            __syncthreads();

            load_matrix_sync(pa, s_prob, 16);
            #pragma unroll
            for (int nt8 = 0; nt8 < NT8; nt8++) {
                fill_fragment(c, 0.0f);
                load_matrix_sync(vb, sv[0] + nt8 * 16, HEAD_DIM);
                mma_sync(c, pa, vb, c);
                for (int i = 0; i < c.num_elements; i++) acc[t][nt8].x[i] += c.x[i];
            }
            __syncthreads();
        }
    }

    #pragma unroll
    for (int t = 0; t < MAX_Q; t++) {
        if (t >= num_query_tokens) break;
        const int out_head = head_base + group;
        if (group < gqa) {
            #pragma unroll
            for (int nt8 = 0; nt8 < NT8; nt8++) {
                int c0 = nt8 * 16 + tig * 2, c1 = nt8 * 16 + 8 + tig * 2;
                float inv = 1.0f / rsum[t][0];
                scalar_t* o = out + ((int64_t)seq_idx * num_query_tokens + t) * num_heads * HEAD_DIM
                             + (int64_t)out_head * HEAD_DIM;
                o[c0] = psp_from_float<scalar_t>(acc[t][nt8].x[0] * inv);
                o[c0 + 1] = psp_from_float<scalar_t>(acc[t][nt8].x[1] * inv);
                o[c1] = psp_from_float<scalar_t>(acc[t][nt8].x[4] * inv);
                o[c1 + 1] = psp_from_float<scalar_t>(acc[t][nt8].x[5] * inv);
            }
        }
    }
}

// ============================================================================
// fp8-query tensor-core decode path (PSPKV_FP8_QUERY=1). Quantizes the query to
// e4m3 (per-head max-abs scale) and reads the fp8 K cache directly for the QK
// m16n8k32 mma; the PV reuses the fp16 mma with the fp8 V dequantized. Enabled
// only behind the flag because the query quantization exceeds the fp32-query
// reference tolerance (~4.6%).
// ============================================================================
template <int HEAD_DIM, typename scalar_t>
__global__ void paged_attention_enhanced_fp8q_tc_kernel(
    float* __restrict__ partials,
    const scalar_t* __restrict__ q,
    const uint8_t* __restrict__ k_cache,
    const uint8_t* __restrict__ v_cache,
    const int* __restrict__ block_tables,
    const int* __restrict__ context_lens,
    const uint32_t* __restrict__ block_offsets,
    const float* __restrict__ scale_table,
    int max_blocks_per_seq,
    int num_heads,
    int num_kv_heads,
    float scale,
    int num_splits,
    int fp8_fmt,
    int num_total_blocks)
{
    constexpr int NT8 = HEAD_DIM / 16;
    constexpr int NK32 = HEAD_DIM / 32;
    constexpr int BLOCK_BYTES = BLOCK_SIZE * HEAD_DIM;

    const int kv_head = blockIdx.x;
    const int seq_idx = blockIdx.y;
    const int split_idx = blockIdx.z;
    const int gqa = num_heads / num_kv_heads;
    const int tid = threadIdx.x;
    const int lane = tid & 31, group = lane >> 2, tig = lane & 3;
    const float log2e = 1.4426950408889634f;
    // m16n8k32 fragment layout: each thread holds 4 low-half + 4 high-half k
    // cols. A rows r0=lane/4, r1=r0+8; B n-col n=lane/4; k-base t4=(lane%4)*4.
    const int r0 = lane / 4, r1 = r0 + 8;
    const int n = lane / 4, t4 = (lane % 4) * 4;
    const __nv_fp8_interpretation_t interp = (fp8_fmt == 2) ? __NV_E5M2 : __NV_E4M3;

    __shared__ uint8_t s_q_fp8[16 * HEAD_DIM];
    __shared__ uint8_t sk_fp8[2][16 * HEAD_DIM];
    __shared__ __half sv[2][16 * HEAD_DIM];
    __shared__ float s_score[16 * 16];
    __shared__ __half s_prob[16 * 16];
    __shared__ unsigned int s_qscale_bits[16];

    fragment<matrix_a, 16, 16, 16, __half, row_major> pa;
    fragment<matrix_b, 16, 16, 16, __half, row_major> vb;
    fragment<accumulator, 16, 16, 16, float> c;
    fragment<accumulator, 16, 16, 16, float> acc[NT8];
    #pragma unroll
    for (int nt8 = 0; nt8 < NT8; nt8++) fill_fragment(acc[nt8], 0.0f);

    // Query -> fp8 (e4m3) with the per-head max-abs scale.
    for (int i = tid; i < 16; i += 32) s_qscale_bits[i] = 0u;
    __syncthreads();
    const int head_base = kv_head * gqa;
    for (int i = tid; i < gqa * HEAD_DIM; i += 32) {
        int h = i / HEAD_DIM, d = i % HEAD_DIM;
        float qv = psp_to_float<scalar_t>(q[(int64_t)seq_idx * num_heads * HEAD_DIM
                                     + (int64_t)(head_base + h) * HEAD_DIM + d]);
        atomicMax(&s_qscale_bits[h], __float_as_uint(fabsf(qv)));
    }
    __syncthreads();
    const float qs0 = fmaxf(__uint_as_float(s_qscale_bits[r0]), 1e-6f);
    const float qs1 = fmaxf(__uint_as_float(s_qscale_bits[r1]), 1e-6f);
    for (int i = tid; i < gqa * HEAD_DIM; i += 32) {
        int h = i / HEAD_DIM, d = i % HEAD_DIM;
        float qv = psp_to_float<scalar_t>(q[(int64_t)seq_idx * num_heads * HEAD_DIM
                                     + (int64_t)(head_base + h) * HEAD_DIM + d]);
        float qs = fmaxf(__uint_as_float(s_qscale_bits[h]), 1e-6f);
        s_q_fp8[h * HEAD_DIM + d] = (uint8_t)__nv_cvt_float_to_fp8(qv / qs, __NV_SATFINITE, __NV_E4M3);
    }
    for (int i = tid; i < (16 - gqa) * HEAD_DIM; i += 32) {
        int h = gqa + i / HEAD_DIM, d = i % HEAD_DIM;
        s_q_fp8[h * HEAD_DIM + d] = 0;
    }
    __syncthreads();

    float rmax0 = -1e30f, rmax1 = -1e30f, rsum0 = 0.0f, rsum1 = 0.0f;

    const int context_len = context_lens[seq_idx];
    const int num_blocks = (context_len + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int blocks_per_split = (num_blocks + num_splits - 1) / num_splits;
    const int split_start = split_idx * blocks_per_split;
    const int split_end = min(split_start + blocks_per_split, num_blocks);
    const int chunk_blocks = split_end - split_start;
    const int block_bytes_per_head = BLOCK_BYTES;

    for (int nb = 0; nb < chunk_blocks; nb++) {
        const int b = split_start + nb;
        const int start_token = b * BLOCK_SIZE;
        const int tokens_in_block = min(BLOCK_SIZE, context_len - start_token);
        const int physical_block = block_tables[seq_idx * max_blocks_per_seq + b];
        const uint8_t* k_base = k_cache + block_offsets[physical_block]
                              + (size_t)kv_head * block_bytes_per_head;
        const uint8_t* v_base = v_cache + block_offsets[num_total_blocks + physical_block]
                              + (size_t)kv_head * block_bytes_per_head;

        int cur = nb & 1;
        const float k_scale = scale_table[((int64_t)physical_block * num_kv_heads + kv_head) * psp_kv_num_groups(HEAD_DIM)];
        const float v_scale = scale_table[((int64_t)num_total_blocks * num_kv_heads
                                          + (int64_t)physical_block * num_kv_heads + kv_head) * psp_kv_num_groups(HEAD_DIM)];

        // K: fp8 direct uint4 copy. V: fp8 -> fp16 (exact vectorized cvt).
        constexpr int VEC = 16;
        for (int i = tid; i < BLOCK_SIZE * HEAD_DIM / VEC; i += 32) {
            *reinterpret_cast<uint4*>(sk_fp8[cur] + i * VEC) =
                *reinterpret_cast<const uint4*>(k_base + i * VEC);
        }
        for (int i = tid; i < BLOCK_SIZE * HEAD_DIM / 2; i += 32) {
            __nv_fp8x2_storage_t vb = *reinterpret_cast<const __nv_fp8x2_storage_t*>(v_base + i * 2);
            __half2 vh = __half2(__nv_cvt_fp8x2_to_halfraw2(vb, interp));
            sv[cur][i * 2 + 0] = vh.x;
            sv[cur][i * 2 + 1] = vh.y;
        }
        __syncthreads();

        // QK in fp8 (m16n8k32): scores [16, 16] = 2 n8 tiles x NK32 k32 chunks.
        const float qk_scale = scale * k_scale;
        for (int n8 = 0; n8 < 2; n8++) {
            float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f;
            #pragma unroll
            for (int kt = 0; kt < NK32; kt++) {
                uint32_t a[4];
                a[0] = *reinterpret_cast<const uint32_t*>(s_q_fp8 + r0 * HEAD_DIM + kt * 32 + t4 + 0);
                a[1] = *reinterpret_cast<const uint32_t*>(s_q_fp8 + r1 * HEAD_DIM + kt * 32 + t4 + 0);
                a[2] = *reinterpret_cast<const uint32_t*>(s_q_fp8 + r0 * HEAD_DIM + kt * 32 + t4 + 16);
                a[3] = *reinterpret_cast<const uint32_t*>(s_q_fp8 + r1 * HEAD_DIM + kt * 32 + t4 + 16);
                const int tok = n8 * 8 + n;
                uint32_t bv[2];
                bv[0] = *reinterpret_cast<const uint32_t*>(sk_fp8[cur] + tok * HEAD_DIM + kt * 32 + t4 + 0);
                bv[1] = *reinterpret_cast<const uint32_t*>(sk_fp8[cur] + tok * HEAD_DIM + kt * 32 + t4 + 16);
                asm volatile(
                    "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                    : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
                    : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(bv[0]), "r"(bv[1]));
            }
            c0 *= qs0 * qk_scale; c1 *= qs0 * qk_scale;
            c2 *= qs1 * qk_scale; c3 *= qs1 * qk_scale;
            const int col0 = tig * 2;
            s_score[r0 * 16 + n8 * 8 + col0] = c0;
            s_score[r0 * 16 + n8 * 8 + col0 + 1] = c1;
            s_score[r1 * 16 + n8 * 8 + col0] = c2;
            s_score[r1 * 16 + n8 * 8 + col0 + 1] = c3;
        }
        __syncthreads();

        // per-row online softmax (2 rows per thread), masked to valid tokens.
        for (int r = group; r < 16; r += 8) {
            float mx = -1e30f;
            for (int cc = 0; cc < tokens_in_block; cc++) mx = fmaxf(mx, s_score[r * 16 + cc]);
            float& rmax = (r == group) ? rmax0 : rmax1;
            float& rsum = (r == group) ? rsum0 : rsum1;
            float m_new = fmaxf(rmax, mx);
            float rescale_old = exp2f((rmax - m_new) * log2e);
            float rescale_new = exp2f((mx - m_new) * log2e);
            rsum *= rescale_old;
            float tsum = 0.0f;
            for (int cc = 0; cc < 16; cc++) {
                float e = (cc < tokens_in_block)
                    ? exp2f((s_score[r * 16 + cc] - mx) * log2e) : 0.0f;
                s_prob[r * 16 + cc] = __float2half(e * rescale_new);
                tsum += e;
            }
            rsum += tsum * rescale_new;
            rmax = m_new;
            #pragma unroll
            for (int nt8 = 0; nt8 < NT8; nt8++) {
                if (r == group) {
                    acc[nt8].x[0] *= rescale_old; acc[nt8].x[1] *= rescale_old;
                    acc[nt8].x[4] *= rescale_old; acc[nt8].x[5] *= rescale_old;
                } else {
                    acc[nt8].x[2] *= rescale_old; acc[nt8].x[3] *= rescale_old;
                    acc[nt8].x[6] *= rescale_old; acc[nt8].x[7] *= rescale_old;
                }
            }
        }
        __syncthreads();

        // PV: acc += (prob @ V_tile) * v_scale (fp32 V scale after the mma).
        load_matrix_sync(pa, s_prob, 16);
        #pragma unroll
        for (int nt8 = 0; nt8 < NT8; nt8++) {
            fill_fragment(c, 0.0f);
            load_matrix_sync(vb, sv[cur] + nt8 * 16, HEAD_DIM);
            mma_sync(c, pa, vb, c);
            for (int i = 0; i < c.num_elements; i++) acc[nt8].x[i] += c.x[i] * v_scale;
        }
        __syncthreads();
    }

    const int64_t p_base =
        (((int64_t)kv_head * gridDim.y + seq_idx) * num_splits + split_idx) * gqa + group;
    #pragma unroll
    for (int nt8 = 0; nt8 < NT8; nt8++) {
        int c0 = nt8 * 16 + tig * 2, c1 = nt8 * 16 + 8 + tig * 2;
        if (group < gqa) {
            float* p = partials + p_base * (2 + HEAD_DIM);
            p[2 + c0]      = acc[nt8].x[0];
            p[2 + c0 + 1]  = acc[nt8].x[1];
            p[2 + c1]      = acc[nt8].x[4];
            p[2 + c1 + 1]  = acc[nt8].x[5];
        }
    }
    if (tig == 0 && group < gqa) {
        partials[p_base * (2 + HEAD_DIM) + 0] = rmax0;
        partials[p_base * (2 + HEAD_DIM) + 1] = rsum0;
    }
}

template <typename scalar_t>
void launch_paged_attention_enhanced_fp8q_tc(
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
    int fp8_fmt,
    int num_total_blocks,
    cudaStream_t stream = 0)
{
    const int num_splits = TC_NUM_SPLITS;
    dim3 grid(num_kv_heads, num_seqs, num_splits);
    psp_kv_dispatch_head_dim(head_dim, [&](auto hd) {
        constexpr int HEAD_DIM = decltype(hd)::value;
        paged_attention_enhanced_fp8q_tc_kernel<HEAD_DIM, scalar_t><<<grid, TC_THREADS, 0, stream>>>(
            partials, q, k_cache, v_cache,
            block_tables, context_lens, block_offsets, scale_table,
            max_blocks_per_seq, num_heads, num_kv_heads, scale, num_splits, fp8_fmt,
            num_total_blocks);
        paged_attention_enhanced_tc_merge_kernel<HEAD_DIM, scalar_t>
            <<<dim3(num_kv_heads, num_seqs), 128, 0, stream>>>(
            out, partials, num_kv_heads, num_seqs, num_heads, num_splits);
    });
}

template <typename scalar_t>
void launch_paged_attention_enhanced_tc(
    scalar_t* out,
    const scalar_t* q,
    const uint8_t* k_cache,
    const uint8_t* v_cache,
    const int* block_tables,
    const int* context_lens,
    const uint32_t* block_offsets,
    float* partials,
    int max_blocks_per_seq,
    int num_seqs,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    float scale,
    int sliding_window,
    int num_splits,
    int num_total_blocks,
    cudaStream_t stream = 0)
{
    dim3 grid(num_kv_heads, num_seqs, num_splits);
    psp_kv_dispatch_head_dim(head_dim, [&](auto hd) {
        constexpr int HEAD_DIM = decltype(hd)::value;
        paged_attention_enhanced_tc_kernel<HEAD_DIM, scalar_t><<<grid, TC_THREADS, 0, stream>>>(
            partials, q, k_cache, v_cache,
            block_tables, context_lens, block_offsets,
            max_blocks_per_seq, num_heads, num_kv_heads, scale, num_splits, sliding_window,
            num_total_blocks);
        paged_attention_enhanced_tc_merge_kernel<HEAD_DIM, scalar_t>
            <<<dim3(num_kv_heads, num_seqs), 128, 0, stream>>>(
            out, partials, num_kv_heads, num_seqs, num_heads, num_splits);
    });

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Enhanced TC paged attention launch failed: %s\n", cudaGetErrorString(err));
    }
}

void launch_paged_attention_enhanced_tf32_tc(
    float* out,
    const float* q,
    const uint8_t* k_cache,
    const uint8_t* v_cache,
    const int* block_tables,
    const int* context_lens,
    const uint32_t* block_offsets,
    float* partials,
    int max_blocks_per_seq,
    int num_seqs,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    float scale,
    int sliding_window,
    int num_total_blocks,
    cudaStream_t stream = 0)
{
    const int num_splits = TC_NUM_SPLITS;
    dim3 grid(num_kv_heads, num_seqs, num_splits);
    psp_kv_dispatch_head_dim(head_dim, [&](auto hd) {
        constexpr int HEAD_DIM = decltype(hd)::value;
        if constexpr (HEAD_DIM <= 128) {
            paged_attention_enhanced_tf32_tc_kernel<HEAD_DIM><<<grid, TC_THREADS, 0, stream>>>(
                partials, q, k_cache, v_cache,
                block_tables, context_lens, block_offsets,
                max_blocks_per_seq, num_heads, num_kv_heads, scale, num_splits, sliding_window,
                num_total_blocks);
            paged_attention_enhanced_tc_merge_kernel<HEAD_DIM, float>
                <<<dim3(num_kv_heads, num_seqs), 128, 0, stream>>>(
                out, partials, num_kv_heads, num_seqs, num_heads, num_splits);
        }
    });

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Enhanced tf32 TC paged attention launch failed: %s\n", cudaGetErrorString(err));
    }
}

template <typename scalar_t>
void launch_paged_attention_enhanced_fp8_tc(
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
    int fp8_fmt,
    int num_total_blocks,
    int num_splits,
    cudaStream_t stream = 0)
{
    dim3 grid(num_kv_heads, num_seqs, num_splits);
    psp_kv_dispatch_head_dim(head_dim, [&](auto hd) {
        constexpr int HEAD_DIM = decltype(hd)::value;
        paged_attention_enhanced_fp8_tc_kernel<HEAD_DIM, scalar_t><<<grid, TC_THREADS, 0, stream>>>(
            partials, q, k_cache, v_cache,
            block_tables, context_lens, block_offsets, scale_table,
            max_blocks_per_seq, num_heads, num_kv_heads, scale, num_splits, fp8_fmt,
            num_total_blocks);
        paged_attention_enhanced_tc_merge_kernel<HEAD_DIM, scalar_t>
            <<<dim3(num_kv_heads, num_seqs), 128, 0, stream>>>(
            out, partials, num_kv_heads, num_seqs, num_heads, num_splits);
    });

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Enhanced fp8 TC paged attention launch failed: %s\n", cudaGetErrorString(err));
    }
}

template <typename scalar_t>
void launch_paged_attention_enhanced_fp8_native_tc(
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
    int fp8_fmt,
    int num_total_blocks,
    int num_splits,
    cudaStream_t stream = 0)
{
    dim3 grid(num_kv_heads, num_seqs, num_splits);
    psp_kv_dispatch_head_dim(head_dim, [&](auto hd) {
        constexpr int HEAD_DIM = decltype(hd)::value;
        paged_attention_enhanced_fp8_native_tc_kernel<HEAD_DIM, scalar_t>
            <<<grid, TC_THREADS, 0, stream>>>(
            partials, q, k_cache, v_cache,
            block_tables, context_lens, block_offsets, scale_table,
            max_blocks_per_seq, num_heads, num_kv_heads, scale, num_splits, fp8_fmt,
            num_total_blocks);
        paged_attention_enhanced_tc_merge_kernel<HEAD_DIM, scalar_t>
            <<<dim3(num_kv_heads, num_seqs), 128, 0, stream>>>(
            out, partials, num_kv_heads, num_seqs, num_heads, num_splits);
    });

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Enhanced native fp8 TC paged attention launch failed: %s\n", cudaGetErrorString(err));
    }
}

template <typename scalar_t>
void launch_paged_attention_enhanced_fp8pv_tc(
    scalar_t* out,
    const scalar_t* q,
    const uint8_t* k_cache,
    const uint8_t* v_cache,
    const int* block_tables,
    const int* context_lens,
    const uint32_t* block_offsets,
    const float* scale_table,
    const uint8_t* precision_table,
    float* partials,
    int max_blocks_per_seq,
    int num_seqs,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    float scale,
    int fp8_fmt,
    int k_fmt,
    int num_total_blocks,
    int num_splits,
    cudaStream_t stream = 0)
{
    dim3 grid(num_kv_heads, num_seqs, num_splits);
    psp_kv_dispatch_head_dim(head_dim, [&](auto hd) {
        constexpr int HEAD_DIM = decltype(hd)::value;
        paged_attention_enhanced_fp8pv_tc_kernel<HEAD_DIM, scalar_t>
            <<<grid, TC_THREADS, 0, stream>>>(
            out, partials, q, k_cache, v_cache,
            block_tables, context_lens, block_offsets, scale_table,
            precision_table,
            max_blocks_per_seq, num_heads, num_kv_heads, scale, num_splits,
            fp8_fmt, num_total_blocks);
        paged_attention_enhanced_tc_merge_kernel<HEAD_DIM, scalar_t>
            <<<dim3(num_kv_heads, num_seqs), 128, 0, stream>>>(
            out, partials, num_kv_heads, num_seqs, num_heads, num_splits);
    });

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Enhanced fp8+fp4 TC paged attention launch failed: %s\n", cudaGetErrorString(err));
    }
}

template <typename scalar_t>
void launch_paged_attention_enhanced_fp8pv_mt(
    scalar_t* out,
    const scalar_t* q,
    const scalar_t* key,
    const scalar_t* value,
    const uint8_t* k_cache,
    const uint8_t* v_cache,
    const int* block_tables,
    const int* context_lens,
    const uint32_t* block_offsets,
    const float* scale_table,
    const uint8_t* precision_table,
    int max_blocks_per_seq,
    int num_tokens,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    float scale,
    int fp8_fmt,
    int num_total_blocks,
    int num_query_tokens,
    cudaStream_t stream = 0)
{
    const int num_seqs = num_tokens / num_query_tokens;
    dim3 grid(num_kv_heads, num_seqs);
    psp_kv_dispatch_head_dim(head_dim, [&](auto hd) {
        constexpr int HEAD_DIM = decltype(hd)::value;
        paged_attention_enhanced_fp8pv_mt_kernel<HEAD_DIM, 8, scalar_t>
            <<<grid, TC_THREADS, 0, stream>>>(
            out, q, key, value, k_cache, v_cache,
            block_tables, context_lens, block_offsets, scale_table,
            precision_table,
            max_blocks_per_seq, num_heads, num_kv_heads, scale, fp8_fmt,
            num_total_blocks, num_query_tokens);
    });
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Enhanced fp8+fp4 MT paged attention launch failed: %s\n", cudaGetErrorString(err));
    }
}

template void launch_paged_attention_enhanced_fp8pv_mt<half>(
    half*, const half*, const half*, const half*, const uint8_t*, const uint8_t*,
    const int*, const int*, const uint32_t*, const float*, const uint8_t*, int, int,
    int, int, int, float, int, int, int, cudaStream_t);
template void launch_paged_attention_enhanced_fp8pv_mt<__nv_bfloat16>(
    __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*,
    const uint8_t*, const uint8_t*, const int*, const int*, const uint32_t*, const float*,
    const uint8_t*, int, int, int, int, int, float, int, int, int, cudaStream_t);

template <typename scalar_t>
void launch_paged_attention_enhanced(
    scalar_t* out,
    const scalar_t* q,
    const uint8_t* k_cache,
    const uint8_t* v_cache,
    const float* k_scales,
    const float* v_scales,
    const int* block_tables,
    const int* context_lens,
    const uint8_t* precision_table,
    const uint32_t* block_offsets,
    float* partials,
    int max_blocks_per_seq,
    int num_tokens,
    int num_seqs,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    float scale,
    int sliding_window,
    int num_blocks,
    cudaStream_t stream = 0)
{
    dim3 grid(num_heads, num_seqs, NUM_SPLITS * NUM_SPLITS_H);
    psp_kv_dispatch_head_dim(head_dim, [&](auto hd) {
        constexpr int HEAD_DIM = decltype(hd)::value;
        paged_attention_enhanced_split_kernel<scalar_t, HEAD_DIM><<<grid, THREADS, 0, stream>>>(
            partials, q, k_cache, v_cache,
            k_scales, v_scales,
            block_tables, context_lens, precision_table, block_offsets,
            max_blocks_per_seq, num_heads, num_kv_heads, head_dim, scale, NUM_SPLITS,
            sliding_window, num_blocks);
        paged_attention_enhanced_merge_kernel<scalar_t, HEAD_DIM>
            <<<dim3(num_heads, num_seqs, NUM_SPLITS_H), THREADS, 0, stream>>>(
            out, partials, num_heads, num_seqs, NUM_SPLITS);
    });

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Enhanced paged attention kernel launch failed: %s\n", cudaGetErrorString(err));
    }
}

template void launch_paged_attention_enhanced<half>(
    half*, const half*, const uint8_t*, const uint8_t*,
    const float*, const float*, const int*, const int*, const uint8_t*, const uint32_t*,
    float*, int, int, int, int, int, int, float, int, int, cudaStream_t);
template void launch_paged_attention_enhanced<__nv_bfloat16>(
    __nv_bfloat16*, const __nv_bfloat16*, const uint8_t*, const uint8_t*,
    const float*, const float*, const int*, const int*, const uint8_t*, const uint32_t*,
    float*, int, int, int, int, int, int, float, int, int, cudaStream_t);
template void launch_paged_attention_enhanced<float>(
    float*, const float*, const uint8_t*, const uint8_t*,
    const float*, const float*, const int*, const int*, const uint8_t*, const uint32_t*,
    float*, int, int, int, int, int, int, float, int, int, cudaStream_t);
template void launch_paged_attention_enhanced<__nv_fp8_e4m3>(
    __nv_fp8_e4m3*, const __nv_fp8_e4m3*, const uint8_t*, const uint8_t*,
    const float*, const float*, const int*, const int*, const uint8_t*, const uint32_t*,
    float*, int, int, int, int, int, int, float, int, int, cudaStream_t);
template void launch_paged_attention_enhanced<__nv_fp8_e5m2>(
    __nv_fp8_e5m2*, const __nv_fp8_e5m2*, const uint8_t*, const uint8_t*,
    const float*, const float*, const int*, const int*, const uint8_t*, const uint32_t*,
    float*, int, int, int, int, int, int, float, int, int, cudaStream_t);

template void launch_paged_attention_enhanced_tc<half>(
    half*, const half*, const uint8_t*, const uint8_t*,
    const int*, const int*, const uint32_t*, float*,
    int, int, int, int, int, float, int, int, int, cudaStream_t);
template void launch_paged_attention_enhanced_tc<__nv_bfloat16>(
    __nv_bfloat16*, const __nv_bfloat16*, const uint8_t*, const uint8_t*,
    const int*, const int*, const uint32_t*, float*,
    int, int, int, int, int, float, int, int, int, cudaStream_t);

template void launch_paged_attention_enhanced_fp8_tc<half>(
    half*, const half*, const uint8_t*, const uint8_t*,
    const int*, const int*, const uint32_t*, const float*, float*,
    int, int, int, int, int, float, int, int, int, cudaStream_t);
template void launch_paged_attention_enhanced_fp8_tc<__nv_bfloat16>(
    __nv_bfloat16*, const __nv_bfloat16*, const uint8_t*, const uint8_t*,
    const int*, const int*, const uint32_t*, const float*, float*,
    int, int, int, int, int, float, int, int, int, cudaStream_t);

template void launch_paged_attention_enhanced_fp8_native_tc<half>(
    half*, const half*, const uint8_t*, const uint8_t*,
    const int*, const int*, const uint32_t*, const float*, float*,
    int, int, int, int, int, float, int, int, int, cudaStream_t);
template void launch_paged_attention_enhanced_fp8_native_tc<__nv_bfloat16>(
    __nv_bfloat16*, const __nv_bfloat16*, const uint8_t*, const uint8_t*,
    const int*, const int*, const uint32_t*, const float*, float*,
    int, int, int, int, int, float, int, int, int, cudaStream_t);

template void launch_paged_attention_enhanced_fp8q_tc<half>(
    half*, const half*, const uint8_t*, const uint8_t*,
    const int*, const int*, const uint32_t*, const float*, float*,
    int, int, int, int, int, float, int, int, cudaStream_t);
template void launch_paged_attention_enhanced_fp8q_tc<__nv_bfloat16>(
    __nv_bfloat16*, const __nv_bfloat16*, const uint8_t*, const uint8_t*,
    const int*, const int*, const uint32_t*, const float*, float*,
    int, int, int, int, int, float, int, int, cudaStream_t);

template void launch_paged_attention_enhanced_fp8pv_tc<half>(
    half*, const half*, const uint8_t*, const uint8_t*,
    const int*, const int*, const uint32_t*, const float*, const uint8_t*, float*,
    int, int, int, int, int, float, int, int, int, int, cudaStream_t);
template void launch_paged_attention_enhanced_fp8pv_tc<__nv_bfloat16>(
    __nv_bfloat16*, const __nv_bfloat16*, const uint8_t*, const uint8_t*,
    const int*, const int*, const uint32_t*, const float*, const uint8_t*, float*,
    int, int, int, int, int, float, int, int, int, int, cudaStream_t);
