#include "common/quantization_utils.cuh"
#define BLOCK_SIZE 16
#define MAX_KV_HEADS 8
template <typename scalar_t, int HEAD_DIM>
__global__ void paged_attention_basic_kernel(
    scalar_t* __restrict__ out,
    const scalar_t* __restrict__ q,
    const uint8_t* __restrict__ k_cache,
    const uint8_t* __restrict__ v_cache,
    const float* __restrict__ k_scales,
    const float* __restrict__ k_zero_points,
    const float* __restrict__ v_scales,
    const float* __restrict__ v_zero_points,
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
    int token_idx = blockIdx.x;          // one token per block
    int head_idx = blockIdx.y;           // one query head per block
    int seq_idx = blockIdx.z;            // one sequence per grid dimension

    if (token_idx != seq_idx || token_idx >= context_lens[seq_idx]) return;

    // Determine which KV head this query head maps to (GQA)
    int kv_head = head_idx / (num_heads / num_kv_heads);

    // Load query vector for this token and head
    const scalar_t* q_ptr = q + (token_idx * num_heads + head_idx) * head_dim;
    float query[HEAD_DIM];
    #pragma unroll
    for (int i = 0; i < HEAD_DIM; ++i) {
        query[i] = psp_to_float(q_ptr[i]);
    }

    // Accumulators for output
    float acc[HEAD_DIM] = {0.0f};
    float max_score = -INFINITY;
    float sum_exp = 0.0f;

    // Iterate over blocks belonging to this sequence
    int num_blocks = (context_lens[seq_idx] + BLOCK_SIZE - 1) / BLOCK_SIZE;
    for (int block_idx = 0; block_idx < num_blocks; ++block_idx) {
        int physical_block = block_tables[seq_idx * max_blocks_per_seq + block_idx];

        // Compute start token index within this block
        int start_token = block_idx * BLOCK_SIZE;
        int tokens_in_block = min(BLOCK_SIZE, context_lens[seq_idx] - start_token);

        // Per-block FP8 format (1=e4m3, 2=e5m2; 0 reserved for FP16).
        int fp8_fmt = (int)precision_table[physical_block];

        // Load scale for this KV head and block (zero-points are unused for FP8).
        float k_scale = k_scales[physical_block * num_kv_heads + kv_head];
        float v_scale = v_scales[physical_block * num_kv_heads + kv_head];

        // Byte stride for this block's precision.
        int block_bytes_per_head;
        if (fp8_fmt == 0)      block_bytes_per_head = BLOCK_SIZE * HEAD_DIM * (int)sizeof(scalar_t);
        else if (fp8_fmt == 3) block_bytes_per_head = BLOCK_SIZE * HEAD_DIM / 2;
        else                   block_bytes_per_head = BLOCK_SIZE * HEAD_DIM;

        const uint8_t* k_block_base = k_cache + block_offsets[physical_block]
                                    + (size_t)kv_head * block_bytes_per_head;
        const uint8_t* v_block_base = v_cache + block_offsets[physical_block]
                                    + (size_t)kv_head * block_bytes_per_head;



        // Iterate over tokens in this block
        for (int t = 0; t < tokens_in_block; ++t) {
            // Compute dot product between query and key (dequantize key on‑the‑fly)
            float score = 0.0f;
            int elem_base = t * HEAD_DIM;
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; ++d) {
                float k_val = psp_packed_dequant<scalar_t>(k_block_base, elem_base + d, fp8_fmt, k_scale);
                score += query[d] * k_val;
            }
            score *= scale;

            // Update running max and sum for online softmax
            float new_max = fmaxf(max_score, score);
            float exp_old = expf(max_score - new_max);
            float exp_new = expf(score - new_max);
            sum_exp = sum_exp * exp_old + exp_new;
            max_score = new_max;

            // Dequantize value vector and accumulate with the current score (will be normalized later)
            float weight = exp_new; // temporary unnormalized weight
            #pragma unroll
            for (int d = 0; d < HEAD_DIM; ++d) {
                float v_val = psp_packed_dequant<scalar_t>(v_block_base, elem_base + d, fp8_fmt, v_scale);
                acc[d] = acc[d] * exp_old + weight * v_val;
            }
        }
    }

    // Normalize accumulators by sum_exp
    float inv_sum = 1.0f / sum_exp;
    scalar_t* out_ptr = out + (token_idx * num_heads + head_idx) * head_dim;
    #pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) {
        out_ptr[d] = psp_from_float<scalar_t>(acc[d] * inv_sum);
    }
}
