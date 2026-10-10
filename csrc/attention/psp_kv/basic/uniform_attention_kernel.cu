// Uniform-consumer paged attention kernel (Basic-format staging path).
//
// Reads a dequantized, uniform target-precision KV cache (the output of
// launch_dequant_kv / psp_dequantize_kv). No precision_table, block_offsets,
// or scale_table are needed here: the inner precision is transparent.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include "common/quantization_utils.cuh"
#include "common/head_dim_dispatch.cuh"

#define BLOCK_SIZE 16

template <typename scalar_t, int HEAD_DIM>
__global__ void paged_attention_uniform_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ q,
    const scalar_t* __restrict__ k_cache,   // [num_blocks, num_kv_heads, block_size, head_dim]
    const scalar_t* __restrict__ v_cache,
    const int* __restrict__ block_tables,
    const int* __restrict__ context_lens,
    int max_blocks_per_seq,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    float scale)
{
    int token_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int seq_idx = blockIdx.z;

    if (token_idx != seq_idx || token_idx >= context_lens[seq_idx]) return;

    int kv_head = head_idx / (num_heads / num_kv_heads);

    const scalar_t* q_ptr = q + (token_idx * num_heads + head_idx) * head_dim;
    float query[HEAD_DIM];
    #pragma unroll
    for (int i = 0; i < HEAD_DIM; ++i) query[i] = psp_to_float(q_ptr[i]);

    float acc[HEAD_DIM] = {0.0f};
    float max_score = -INFINITY;
    float sum_exp = 0.0f;

    int num_blocks = (context_lens[seq_idx] + BLOCK_SIZE - 1) / BLOCK_SIZE;
    for (int block_idx = 0; block_idx < num_blocks; ++block_idx) {
        int physical_block = block_tables[seq_idx * max_blocks_per_seq + block_idx];
        int start_token = block_idx * BLOCK_SIZE;
        int tokens_in_block = min(BLOCK_SIZE, context_lens[seq_idx] - start_token);

        const scalar_t* k_block = k_cache + ((size_t)physical_block * num_kv_heads + kv_head) * BLOCK_SIZE * HEAD_DIM;
        const scalar_t* v_block = v_cache + ((size_t)physical_block * num_kv_heads + kv_head) * BLOCK_SIZE * HEAD_DIM;

        for (int t = 0; t < tokens_in_block; ++t) {
            float score = 0.0f;
            const scalar_t* k_token = k_block + t * HEAD_DIM;
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; ++d) {
                score += query[d] * psp_to_float(k_token[d]);
            }
            score *= scale;

            float new_max = fmaxf(max_score, score);
            float exp_old = expf(max_score - new_max);
            float exp_new = expf(score - new_max);
            sum_exp = sum_exp * exp_old + exp_new;
            max_score = new_max;

            const scalar_t* v_token = v_block + t * HEAD_DIM;
            float weight = exp_new;
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; ++d) {
                acc[d] = acc[d] * exp_old + weight * psp_to_float(v_token[d]);
            }
        }
    }

    float inv_sum = 1.0f / sum_exp;
    scalar_t* out_ptr = out + (token_idx * num_heads + head_idx) * head_dim;
    #pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) {
        out_ptr[d] = psp_from_float<scalar_t>(acc[d] * inv_sum);
    }
}

template <typename scalar_t>
void launch_paged_attention_uniform(
    scalar_t* out,
    const scalar_t* q,
    const scalar_t* k_cache,
    const scalar_t* v_cache,
    const int* block_tables,
    const int* context_lens,
    int max_blocks_per_seq,
    int num_tokens,
    int num_seqs,
    int num_heads,
    int num_kv_heads,
    int head_dim,
    float scale,
    cudaStream_t stream = 0)
{
    dim3 grid(num_tokens, num_heads, num_seqs);
    psp_kv_dispatch_head_dim(head_dim, [&](auto hd) {
        constexpr int HEAD_DIM = decltype(hd)::value;
        paged_attention_uniform_kernel<scalar_t, HEAD_DIM><<<grid, 1, 0, stream>>>(
        out, q, k_cache, v_cache, block_tables, context_lens,
        max_blocks_per_seq, num_heads, num_kv_heads, head_dim, scale);
    });
}

template void launch_paged_attention_uniform<half>(
    half*, const half*, const half*, const half*,
    const int*, const int*, int, int, int, int, int, int, float, cudaStream_t);
template void launch_paged_attention_uniform<__nv_bfloat16>(
    __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*, const __nv_bfloat16*,
    const int*, const int*, int, int, int, int, int, int, float, cudaStream_t);
template void launch_paged_attention_uniform<__nv_fp8_e4m3>(
    __nv_fp8_e4m3*, const __nv_fp8_e4m3*, const __nv_fp8_e4m3*, const __nv_fp8_e4m3*,
    const int*, const int*, int, int, int, int, int, int, float, cudaStream_t);
template void launch_paged_attention_uniform<__nv_fp8_e5m2>(
    __nv_fp8_e5m2*, const __nv_fp8_e5m2*, const __nv_fp8_e5m2*, const __nv_fp8_e5m2*,
    const int*, const int*, int, int, int, int, int, int, float, cudaStream_t);
