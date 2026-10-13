// virthub/kernels/btkv_format/paged_attention_btkv.cu
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
#include <math.h>
#include <stdint.h>
#include <stdio.h>

#include "common/quantization_utils.cuh"
#include "common/head_dim_dispatch.cuh"

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
__global__ void paged_attention_btkv_split_kernel(
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
    int num_splits)
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

    const int num_blocks = (context_len + BLOCK_SIZE - 1) / BLOCK_SIZE;
    const int blocks_per_split = (num_blocks + num_splits - 1) / num_splits;
    const int split_start = split_idx * blocks_per_split;
    const int split_end = min(split_start + blocks_per_split, num_blocks);

    float local_max = -INFINITY;
    float local_sum = 0.0f;
    float acc[HALF];
    for (int d = 0; d < HALF; ++d) acc[d] = 0.0f;

    const int dim_offset = split_h * HALF;

    for (int b = split_start + tid; b < split_end; b += nthreads) {
        const int physical_block = block_tables[seq_idx * max_blocks_per_seq + b];
        const int fp8_fmt = (int)precision_table[physical_block];
        const float k_scale = k_scales[physical_block * num_kv_heads + kv_head];
        const float v_scale = v_scales[physical_block * num_kv_heads + kv_head];
        const float k_scale_eff = (fp8_fmt == 0) ? 1.0f : k_scale;
        const float v_scale_eff = (fp8_fmt == 0) ? 1.0f : v_scale;
        const float combined_scale = k_scale_eff * scale;
        int block_bytes_per_head;
        if (fp8_fmt == 0)      block_bytes_per_head = BLOCK_SIZE * HEAD_DIM * (int)sizeof(scalar_t);
        else if (fp8_fmt == 3) block_bytes_per_head = BLOCK_SIZE * HEAD_DIM / 2;
        else                   block_bytes_per_head = BLOCK_SIZE * HEAD_DIM;
        const uint8_t* k_base = k_cache + block_offsets[physical_block]
                              + (size_t)kv_head * block_bytes_per_head;
        const uint8_t* v_base = v_cache + block_offsets[physical_block]
                              + (size_t)kv_head * block_bytes_per_head;
        const int start_token = b * BLOCK_SIZE;
        const int tokens_in_block = min(BLOCK_SIZE, context_len - start_token);
        switch (fp8_fmt) {
            case 0: psp_decode_block<scalar_t, HEAD_DIM, 0, HEAD_DIM/2>(s_query, local_max, local_sum, acc, k_base, v_base, tokens_in_block, combined_scale, v_scale_eff, dim_offset); break;
            case 1: psp_decode_block<scalar_t, HEAD_DIM, 1, HEAD_DIM/2>(s_query, local_max, local_sum, acc, k_base, v_base, tokens_in_block, combined_scale, v_scale_eff, dim_offset); break;
            case 2: psp_decode_block<scalar_t, HEAD_DIM, 2, HEAD_DIM/2>(s_query, local_max, local_sum, acc, k_base, v_base, tokens_in_block, combined_scale, v_scale_eff, dim_offset); break;
            default: psp_decode_block<scalar_t, HEAD_DIM, 3, HEAD_DIM/2>(s_query, local_max, local_sum, acc, k_base, v_base, tokens_in_block, combined_scale, v_scale_eff, dim_offset); break;
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
__global__ void paged_attention_btkv_merge_kernel(
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

template <typename scalar_t>
void launch_paged_attention_btkv(
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
    cudaStream_t stream = 0)
{
    dim3 grid(num_heads, num_seqs, NUM_SPLITS * NUM_SPLITS_H);
    psp_kv_dispatch_head_dim(head_dim, [&](auto hd) {
        constexpr int HEAD_DIM = decltype(hd)::value;
        paged_attention_btkv_split_kernel<scalar_t, HEAD_DIM><<<grid, THREADS, 0, stream>>>(
            partials, q, k_cache, v_cache,
            k_scales, v_scales,
            block_tables, context_lens, precision_table, block_offsets,
            max_blocks_per_seq, num_heads, num_kv_heads, head_dim, scale, NUM_SPLITS);
        paged_attention_btkv_merge_kernel<scalar_t, HEAD_DIM>
            <<<dim3(num_heads, num_seqs, NUM_SPLITS_H), THREADS, 0, stream>>>(
            out, partials, num_heads, num_seqs, NUM_SPLITS);
    });

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "Enhanced paged attention kernel launch failed: %s\n", cudaGetErrorString(err));
    }
}

template void launch_paged_attention_btkv<half>(
    half*, const half*, const uint8_t*, const uint8_t*,
    const float*, const float*, const int*, const int*, const uint8_t*, const uint32_t*,
    float*, int, int, int, int, int, int, float, cudaStream_t);
template void launch_paged_attention_btkv<__nv_bfloat16>(
    __nv_bfloat16*, const __nv_bfloat16*, const uint8_t*, const uint8_t*,
    const float*, const float*, const int*, const int*, const uint8_t*, const uint32_t*,
    float*, int, int, int, int, int, int, float, cudaStream_t);
template void launch_paged_attention_btkv<__nv_fp8_e4m3>(
    __nv_fp8_e4m3*, const __nv_fp8_e4m3*, const uint8_t*, const uint8_t*,
    const float*, const float*, const int*, const int*, const uint8_t*, const uint32_t*,
    float*, int, int, int, int, int, int, float, cudaStream_t);
template void launch_paged_attention_btkv<__nv_fp8_e5m2>(
    __nv_fp8_e5m2*, const __nv_fp8_e5m2*, const uint8_t*, const uint8_t*,
    const float*, const float*, const int*, const int*, const uint8_t*, const uint32_t*,
    float*, int, int, int, int, int, int, float, cudaStream_t);
