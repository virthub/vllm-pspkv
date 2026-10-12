// virthub/kernels/btkv_format/paged_attention_btkv.cu
//
// BT-KV Format Paged Attention Kernel
//
// Paged attention for the BT-KV PSP-KV format. KV cache blocks are stored in
// quantized (FP8/FP4/lossless) representation with per-(block, head) float32
// scales. K is token-major [B_t, D_head], V is transposed [D_head, B_t].
//
// Decode-only: one thread block per (query head, sequence), threads cooperate
// on the context scan (two-phase online softmax with a block reduction).

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>

#include "common/quantization_utils.cuh"
#include "common/head_dim_dispatch.cuh"

#define BLOCK_SIZE 16
#define THREADS 128

__device__ inline uint32_t psp_f2u_mono(float f) {
    uint32_t u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
__device__ inline float psp_u2f_mono(uint32_t u) {
    return __uint_as_float((u & 0x80000000u) ? (u & 0x7FFFFFFFu) : ~u);
}

template <typename scalar_t, int HEAD_DIM>
__global__ void paged_attention_btkv_kernel(
    scalar_t* __restrict__ out,
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
    float scale)
{
    const int head_idx = blockIdx.x;
    const int seq_idx = blockIdx.y;
    const int tid = threadIdx.x;
    const int nthreads = blockDim.x;

    const int context_len = context_lens[seq_idx];
    const int kv_head = head_idx / (num_heads / num_kv_heads);

    __shared__ float s_query[HEAD_DIM];
    __shared__ uint32_t s_max;
    __shared__ float s_sum;
    __shared__ float s_acc[HEAD_DIM];

    const scalar_t* q_ptr = q + ((int64_t)seq_idx * num_heads + head_idx) * head_dim;
    for (int i = tid; i < HEAD_DIM; i += nthreads) s_query[i] = psp_to_float(q_ptr[i]);
    if (tid == 0) {
        s_max = psp_f2u_mono(-INFINITY);
        s_sum = 0.0f;
        for (int d = 0; d < HEAD_DIM; ++d) s_acc[d] = 0.0f;
    }
    __syncthreads();

    const int num_blocks = (context_len + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Phase 1: per-thread max over this thread's assigned blocks.
    float local_max = -INFINITY;
    for (int b = tid; b < num_blocks; b += nthreads) {
        const int physical_block = block_tables[seq_idx * max_blocks_per_seq + b];
        const int fp8_fmt = (int)precision_table[physical_block];
        const float k_scale = k_scales[physical_block * num_kv_heads + kv_head];
        int block_bytes_per_head;
        if (fp8_fmt == 0)      block_bytes_per_head = BLOCK_SIZE * HEAD_DIM * (int)sizeof(scalar_t);
        else if (fp8_fmt == 3) block_bytes_per_head = BLOCK_SIZE * HEAD_DIM / 2;
        else                   block_bytes_per_head = BLOCK_SIZE * HEAD_DIM;
        const uint8_t* k_base = k_cache + block_offsets[physical_block]
                              + (size_t)kv_head * block_bytes_per_head;
        const int start_token = b * BLOCK_SIZE;
        const int tokens_in_block = min(BLOCK_SIZE, context_len - start_token);
        for (int t = 0; t < tokens_in_block; ++t) {
            float s = 0.0f;
            const int elem_base = t * HEAD_DIM;
            for (int d = 0; d < HEAD_DIM; ++d) {
                s += s_query[d] * psp_packed_dequant<scalar_t>(k_base, elem_base + d, fp8_fmt, k_scale);
            }
            s *= scale;
            local_max = fmaxf(local_max, s);
        }
    }
    atomicMax(&s_max, psp_f2u_mono(local_max));
    __syncthreads();
    const float gmax = psp_u2f_mono(s_max);

    // Phase 2: per-thread sum + weighted V (transposed V[d, t]).
    float local_sum = 0.0f;
    float acc[HEAD_DIM];
    for (int d = 0; d < HEAD_DIM; ++d) acc[d] = 0.0f;
    for (int b = tid; b < num_blocks; b += nthreads) {
        const int physical_block = block_tables[seq_idx * max_blocks_per_seq + b];
        const int fp8_fmt = (int)precision_table[physical_block];
        const float k_scale = k_scales[physical_block * num_kv_heads + kv_head];
        const float v_scale = v_scales[physical_block * num_kv_heads + kv_head];
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
        for (int t = 0; t < tokens_in_block; ++t) {
            float s = 0.0f;
            const int elem_base = t * HEAD_DIM;
            for (int d = 0; d < HEAD_DIM; ++d) {
                s += s_query[d] * psp_packed_dequant<scalar_t>(k_base, elem_base + d, fp8_fmt, k_scale);
            }
            s *= scale;
            const float w = expf(s - gmax);
            local_sum += w;
            for (int d = 0; d < HEAD_DIM; ++d) {
                acc[d] += w * psp_packed_dequant<scalar_t>(v_base, d * BLOCK_SIZE + t, fp8_fmt, v_scale);
            }
        }
    }
    atomicAdd(&s_sum, local_sum);
    for (int d = 0; d < HEAD_DIM; ++d) atomicAdd(&s_acc[d], acc[d]);
    __syncthreads();

    if (tid == 0) {
        const float inv_sum = 1.0f / s_sum;
        scalar_t* out_ptr = out + ((int64_t)seq_idx * num_heads + head_idx) * head_dim;
        for (int d = 0; d < HEAD_DIM; ++d) out_ptr[d] = psp_from_float<scalar_t>(s_acc[d] * inv_sum);
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
    int max_blocks_per_seq,
    int num_tokens,
    int num_seqs,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    float scale,
    cudaStream_t stream = 0)
{
    // Decode: one block per (head, seq), THREADS threads cooperate per head.
    dim3 grid(num_heads, num_seqs);
    psp_kv_dispatch_head_dim(head_dim, [&](auto hd) {
        constexpr int HEAD_DIM = decltype(hd)::value;
        paged_attention_btkv_kernel<scalar_t, HEAD_DIM><<<grid, THREADS, 0, stream>>>(
            out, q, k_cache, v_cache,
            k_scales, v_scales,
            block_tables, context_lens, precision_table, block_offsets,
            max_blocks_per_seq, num_heads, num_kv_heads, head_dim, scale);
    });

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "BT-KV paged attention kernel launch failed: %s\n", cudaGetErrorString(err));
    }
}

template void launch_paged_attention_btkv<half>(
    half*, const half*, const uint8_t*, const uint8_t*,
    const float*, const float*, const int*, const int*, const uint8_t*, const uint32_t*,
    int, int, int, int, int, int, float, cudaStream_t);
template void launch_paged_attention_btkv<__nv_bfloat16>(
    __nv_bfloat16*, const __nv_bfloat16*, const uint8_t*, const uint8_t*,
    const float*, const float*, const int*, const int*, const uint8_t*, const uint32_t*,
    int, int, int, int, int, int, float, cudaStream_t);
template void launch_paged_attention_btkv<__nv_fp8_e4m3>(
    __nv_fp8_e4m3*, const __nv_fp8_e4m3*, const uint8_t*, const uint8_t*,
    const float*, const float*, const int*, const int*, const uint8_t*, const uint32_t*,
    int, int, int, int, int, int, float, cudaStream_t);
template void launch_paged_attention_btkv<__nv_fp8_e5m2>(
    __nv_fp8_e5m2*, const __nv_fp8_e5m2*, const uint8_t*, const uint8_t*,
    const float*, const float*, const int*, const int*, const uint8_t*, const uint32_t*,
    int, int, int, int, int, int, float, cudaStream_t);
