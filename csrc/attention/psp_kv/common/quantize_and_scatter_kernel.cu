// virthub/kernels/common/quantize_and_scatter_kernel.cu
//
// Quantize-and-scatter kernel for the PSP-KV prefill write path.
//
// Each fp16/bf16/fp8 value is quantized to the per-block inner precision
// selected by the precision_table (lossless / fp8 e4m3 / fp8 e5m2 / fp4 e2m1)
// as  q = quantize(x / scale, fmt), with a per-(block, head) scale estimated
// from the block's max-abs value:
//
//   scale = max_abs(block, head) / fp_max(fmt)
//
// This is the inverse of the attention kernels' dequant
// (x = dequantize(q, fmt) * scale), so a quantize -> attention round-trip is
// internally consistent and the fp8/fp4 value range is adapted per block.
//
// The scale is written to scale_table[num_blocks, num_kv_heads], which the
// decode kernels read as k_scales/v_scales.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

#include "quantization_utils.cuh"

// Largest finite value representable for each precision (matches the
// dequantize_* helpers in quantization_utils.cuh).
__device__ inline float psp_kv_fp_max(int fmt) {
    if (fmt == 3) return 3.0f;      // fp4 e2m1
    if (fmt == 2) return 57344.0f;  // fp8 e5m2
    return 240.0f;                  // fp8 e4m3 (default)
}

// ---------------------------------------------------------------------------
// Pass 1: per-(block, head) max-abs via atomic-max reduction.
// ---------------------------------------------------------------------------
template <typename scalar_t>
__global__ void estimate_block_scales_kernel(
    const scalar_t* __restrict__ prefill_scratch,  // [num_tokens, 2, num_kv_heads, head_dim]
    const int64_t* __restrict__ slot_mapping,      // [num_tokens]
    const uint8_t* __restrict__ precision_table,   // [num_blocks]
    float* __restrict__ max_abs,                   // [num_blocks, num_kv_heads]
    int num_tokens, int num_kv_heads, int block_size, int head_dim)
{
    // One thread per (token, head): num_kv_heads x more parallelism than the
    // old one-thread-per-token layout, so each thread reduces only 2*head_dim
    // elements instead of 2*num_kv_heads*head_dim (a short fmaxf chain).
    const int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int token = global_tid / num_kv_heads;
    const int h = global_tid % num_kv_heads;
    if (token >= num_tokens) return;

    const int64_t slot = slot_mapping[token];
    const int block_id = static_cast<int>(slot / block_size);
    const int fp8_fmt = static_cast<int>(precision_table[block_id]);
    if (fp8_fmt == 0) return;  // lossless: scale stays 1.0

    const scalar_t* k_row = prefill_scratch +
        ((static_cast<int64_t>(token) * 2 + 0) * num_kv_heads + h) * head_dim;
    const scalar_t* v_row = prefill_scratch +
        ((static_cast<int64_t>(token) * 2 + 1) * num_kv_heads + h) * head_dim;

    float m = 0.0f;
    #pragma unroll 4
    for (int d = 0; d < head_dim; ++d) {
        m = fmaxf(m, fabsf(psp_to_float(k_row[d])));
        m = fmaxf(m, fabsf(psp_to_float(v_row[d])));
    }

    // max_abs is >= 0, so its IEEE-754 bit pattern is monotonically ordered
    // and a signed int atomicMax yields the float max.
    atomicMax(reinterpret_cast<int*>(&max_abs[block_id * num_kv_heads + h]),
              __float_as_int(m));
}

// ---------------------------------------------------------------------------
// Pass 2: convert max-abs -> scale and store into scale_table.
// ---------------------------------------------------------------------------
__global__ void finalize_block_scales_kernel(
    const float* __restrict__ max_abs,             // [num_blocks, num_kv_heads]
    const uint8_t* __restrict__ precision_table,   // [num_blocks]
    float* __restrict__ scale_table,               // [num_blocks, num_kv_heads]
    int num_blocks, int num_kv_heads)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_blocks * num_kv_heads) return;

    int block_id = idx / num_kv_heads;
    int fp8_fmt = static_cast<int>(precision_table[block_id]);

    float scale;
    if (fp8_fmt == 0) {
        scale = 1.0f;  // lossless: no quantization
    } else {
        scale = max_abs[idx] / psp_kv_fp_max(fp8_fmt);
        if (scale < 1e-8f) scale = 1.0f;  // all-zero block: avoid zero scale
    }
    scale_table[idx] = scale;
}

// ---------------------------------------------------------------------------
// Pass 3: quantize each token's K/V with the per-block scale and scatter.
// ---------------------------------------------------------------------------
template <typename scalar_t>
__global__ void quantize_and_scatter_kernel(
    const scalar_t* __restrict__ prefill_scratch,  // [num_tokens, 2, num_kv_heads, head_dim]
    uint8_t* __restrict__ key_cache,           // byte buffer
    uint8_t* __restrict__ value_cache,         // byte buffer
    const int64_t* __restrict__ slot_mapping,  // [num_tokens]
    const uint8_t* __restrict__ precision_table,  // [num_blocks] (0=lossless, 1=e4m3, 2=e5m2, 3=fp4)
    const uint32_t* __restrict__ block_offsets,   // [num_blocks] byte offsets
    const float* __restrict__ scale_table,     // [num_blocks, num_kv_heads]
    int num_tokens, int num_kv_heads, int block_size, int head_dim,
    int format_generation)
{
    // One thread per (token, head): num_kv_heads x more parallelism than the
    // old one-thread-per-token layout, so each thread quantizes one row pair.
    const int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int token = global_tid / num_kv_heads;
    const int h = global_tid % num_kv_heads;
    if (token >= num_tokens) return;

    const int64_t slot = slot_mapping[token];
    const int block_id = static_cast<int>(slot / block_size);
    const int block_offset = static_cast<int>(slot % block_size);
    const int fp8_fmt = static_cast<int>(precision_table[block_id]);

    int block_bytes_per_head;
    if (fp8_fmt == 0)      block_bytes_per_head = block_size * head_dim * (int)sizeof(scalar_t);
    else if (fp8_fmt == 3) block_bytes_per_head = block_size * head_dim / 2;
    else                   block_bytes_per_head = block_size * head_dim;
    int bytes_per_row;
    if (fp8_fmt == 0)      bytes_per_row = head_dim * (int)sizeof(scalar_t);
    else if (fp8_fmt == 3) bytes_per_row = head_dim / 2;
    else                   bytes_per_row = head_dim;

    const float scale = scale_table[static_cast<int64_t>(block_id) * num_kv_heads + h];

    const scalar_t* k_row = prefill_scratch +
        ((static_cast<int64_t>(token) * 2 + 0) * num_kv_heads + h) * head_dim;
    const scalar_t* v_row = prefill_scratch +
        ((static_cast<int64_t>(token) * 2 + 1) * num_kv_heads + h) * head_dim;

    uint8_t* k_out = key_cache + block_offsets[block_id]
                   + (size_t)h * block_bytes_per_head
                   + block_offset * bytes_per_row;
    uint8_t* v_out = value_cache + block_offsets[block_id]
                   + (size_t)h * block_bytes_per_head;

    // Branchless per-block dispatch: the format is fixed per block, so the
    // quantize loop below is compiled once per format without a runtime
    // fmt branch inside the element loop.
    switch (fp8_fmt) {
        case 0: psp_quantize_row<scalar_t, 0>(k_row, v_row, k_out, v_out, head_dim, scale, block_offset); break;
        case 1: psp_quantize_row<scalar_t, 1>(k_row, v_row, k_out, v_out, head_dim, scale, block_offset); break;
        case 2: psp_quantize_row<scalar_t, 2>(k_row, v_row, k_out, v_out, head_dim, scale, block_offset); break;
        default: psp_quantize_row<scalar_t, 3>(k_row, v_row, k_out, v_out, head_dim, scale, block_offset); break;
    }
}

// ---------------------------------------------------------------------------
// Launch
// ---------------------------------------------------------------------------
template <typename scalar_t>
void launch_quantize_and_scatter(
    const scalar_t* prefill_scratch,
    uint8_t* key_cache,
    uint8_t* value_cache,
    const int64_t* slot_mapping,
    const uint8_t* precision_table,
    const uint32_t* block_offsets,
    float* scale_table,
    float* max_abs,   // temp [num_blocks, num_kv_heads] workspace (zeroed)
    int num_tokens, int num_blocks, int num_kv_heads, int block_size, int head_dim,
    int format_generation, cudaStream_t stream)
{
    const int threads = 256;
    // Both the estimate and the quantize now use one thread per (token, head).
    const int pair_blocks = ((int64_t)num_tokens * num_kv_heads + threads - 1) / threads;
    const int scale_blocks = (num_blocks * num_kv_heads + threads - 1) / threads;

    estimate_block_scales_kernel<scalar_t><<<pair_blocks, threads, 0, stream>>>(
        prefill_scratch, slot_mapping, precision_table, max_abs,
        num_tokens, num_kv_heads, block_size, head_dim);

    finalize_block_scales_kernel<<<scale_blocks, threads, 0, stream>>>(
        max_abs, precision_table, scale_table, num_blocks, num_kv_heads);

    quantize_and_scatter_kernel<scalar_t><<<pair_blocks, threads, 0, stream>>>(
        prefill_scratch, key_cache, value_cache, slot_mapping, precision_table,
        block_offsets, scale_table, num_tokens, num_kv_heads, block_size, head_dim,
        format_generation);
}

template void launch_quantize_and_scatter<half>(
    const half*, uint8_t*, uint8_t*, const int64_t*, const uint8_t*, const uint32_t*,
    float*, float*, int, int, int, int, int, int, cudaStream_t);
template void launch_quantize_and_scatter<__nv_bfloat16>(
    const __nv_bfloat16*, uint8_t*, uint8_t*, const int64_t*, const uint8_t*, const uint32_t*,
    float*, float*, int, int, int, int, int, int, cudaStream_t);
template void launch_quantize_and_scatter<__nv_fp8_e4m3>(
    const __nv_fp8_e4m3*, uint8_t*, uint8_t*, const int64_t*, const uint8_t*, const uint32_t*,
    float*, float*, int, int, int, int, int, int, cudaStream_t);
template void launch_quantize_and_scatter<__nv_fp8_e5m2>(
    const __nv_fp8_e5m2*, uint8_t*, uint8_t*, const int64_t*, const uint8_t*, const uint32_t*,
    float*, float*, int, int, int, int, int, int, cudaStream_t);
