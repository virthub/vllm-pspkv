// virthub/kernels/common/quantize_and_scatter_kernel.cu
//
// Reference quantize-and-scatter kernel for the PSP-KV prefill write path.
//
// Each fp16 value is quantized to real FP8 (e4m3 or e5m2, selected per-block by
// the precision_table) as  q = quantize_fp8(x / scale, fmt)  with a unit scale
// (scale = 1.0). This is the inverse of the reference attention kernels'
// dequant (x = dequantize_fp8(q, fmt) * scale), so a quantize -> attention
// round-trip is internally consistent.
//
// Per-block scale estimation (max-abs -> scale) is a follow-up; this reference
// uses unit scale and relies on the input being already in a bounded range.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

#include "quantization_utils.cuh"

template <typename scalar_t>
__global__ void quantize_and_scatter_kernel(
    const scalar_t* __restrict__ prefill_scratch,  // [num_tokens, 2, num_kv_heads, head_dim]
    uint8_t* __restrict__ key_cache,           // byte buffer
    uint8_t* __restrict__ value_cache,         // byte buffer
    const int64_t* __restrict__ slot_mapping,  // [num_tokens]
    const uint8_t* __restrict__ precision_table,  // [num_blocks] (0=lossless, 1=e4m3, 2=e5m2)
    const uint32_t* __restrict__ block_offsets,   // [num_blocks] byte offsets
    float* __restrict__ scale_table,           // [num_blocks, num_kv_heads]
    int num_tokens, int num_kv_heads, int block_size, int head_dim)
{
    int token = blockIdx.x * blockDim.x + threadIdx.x;
    if (token >= num_tokens) return;

    const int64_t slot = slot_mapping[token];
    const int block_id = static_cast<int>(slot / block_size);
    const int block_offset = static_cast<int>(slot % block_size);
    const int fp8_fmt = static_cast<int>(precision_table[block_id]);

    const float scale = 1.0f;  // unit-scale reference

    int block_bytes_per_head;
    if (fp8_fmt == 0)      block_bytes_per_head = block_size * head_dim * (int)sizeof(scalar_t);
    else if (fp8_fmt == 3) block_bytes_per_head = block_size * head_dim / 2;
    else                   block_bytes_per_head = block_size * head_dim;
    int bytes_per_row;
    if (fp8_fmt == 0)      bytes_per_row = head_dim * (int)sizeof(scalar_t);
    else if (fp8_fmt == 3) bytes_per_row = head_dim / 2;
    else                   bytes_per_row = head_dim;

    for (int h = 0; h < num_kv_heads; ++h) {
        scale_table[static_cast<int64_t>(block_id) * num_kv_heads + h] = scale;

        const scalar_t* k_row = prefill_scratch +
            ((static_cast<int64_t>(token) * 2 + 0) * num_kv_heads + h) * head_dim;
        const scalar_t* v_row = prefill_scratch +
            ((static_cast<int64_t>(token) * 2 + 1) * num_kv_heads + h) * head_dim;

        uint8_t* k_out = key_cache + block_offsets[block_id]
                       + (size_t)h * block_bytes_per_head
                       + block_offset * bytes_per_row;
        // Enhanced/BT-KV store V transposed [head_dim, block_size]:
        // element (d, t) lives at byte d*block_size + t.
        uint8_t* v_out = value_cache + block_offsets[block_id]
                       + (size_t)h * block_bytes_per_head;

        for (int d = 0; d < head_dim; ++d) {
            float k = psp_to_float(k_row[d]);
            float v = psp_to_float(v_row[d]);
            psp_packed_quantize<scalar_t>(k_out, d, k, fp8_fmt, scale);
            psp_packed_quantize<scalar_t>(v_out, d * block_size + block_offset, v, fp8_fmt, scale);
        }
    }
}

template <typename scalar_t>
void launch_quantize_and_scatter(
    const scalar_t* prefill_scratch,
    uint8_t* key_cache,
    uint8_t* value_cache,
    const int64_t* slot_mapping,
    const uint8_t* precision_table,
    const uint32_t* block_offsets,
    float* scale_table,
    int num_tokens, int num_kv_heads, int block_size, int head_dim,
    cudaStream_t stream)
{
    const int threads = 256;
    const int blocks = (num_tokens + threads - 1) / threads;
    quantize_and_scatter_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        prefill_scratch, key_cache, value_cache, slot_mapping, precision_table,
        block_offsets, scale_table, num_tokens, num_kv_heads, block_size, head_dim);
}

template void launch_quantize_and_scatter<half>(
    const half*, uint8_t*, uint8_t*, const int64_t*, const uint8_t*, const uint32_t*,
    float*, int, int, int, int, cudaStream_t);
template void launch_quantize_and_scatter<__nv_bfloat16>(
    const __nv_bfloat16*, uint8_t*, uint8_t*, const int64_t*, const uint8_t*, const uint32_t*,
    float*, int, int, int, int, cudaStream_t);
template void launch_quantize_and_scatter<__nv_fp8_e4m3>(
    const __nv_fp8_e4m3*, uint8_t*, uint8_t*, const int64_t*, const uint8_t*, const uint32_t*,
    float*, int, int, int, int, cudaStream_t);
template void launch_quantize_and_scatter<__nv_fp8_e5m2>(
    const __nv_fp8_e5m2*, uint8_t*, uint8_t*, const int64_t*, const uint8_t*, const uint32_t*,
    float*, int, int, int, int, cudaStream_t);
