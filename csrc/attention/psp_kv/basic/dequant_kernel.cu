// virthub/kernels/basic_format/dequant_kernel.cu
//
// Basic-format dequantization: reconstructs a uniform target-precision tensor
// from the compressed byte buffer. This is the "staging buffer" path that makes
// the inner precision (lossless / fp8 / fp4) transparent to the data consumer.
//
// The consumer (a simple attention kernel) reads k_out/v_out at the target
// precision and needs no precision_table / block_offsets / scale knowledge.

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <stdint.h>

#include "common/quantization_utils.cuh"

template <typename scalar_t>
__global__ void dequant_kv_kernel(
    const uint8_t* __restrict__ k_cache,
    const uint8_t* __restrict__ v_cache,
    const uint32_t* __restrict__ block_offsets,
    const uint8_t* __restrict__ precision_table,
    const float* __restrict__ scale_table,
    scalar_t* __restrict__ k_out,
    scalar_t* __restrict__ v_out,
    int num_blocks, int num_kv_heads, int block_size, int head_dim)
{
    const int elems_per_head = block_size * head_dim;
    const int elems_per_block = elems_per_head * num_kv_heads;
    const int total = elems_per_block * num_blocks;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int block = idx / elems_per_block;
    int rem = idx % elems_per_block;
    int head = rem / elems_per_head;
    int elem = rem % elems_per_head;

    int fmt = (int)precision_table[block];
    float scale = (fmt != 0) ? scale_table[block * num_kv_heads + head] : 1.0f;

    int block_bytes_per_head;
    if (fmt == 0)      block_bytes_per_head = elems_per_head * (int)sizeof(scalar_t);
    else if (fmt == 3) block_bytes_per_head = elems_per_head / 2;
    else               block_bytes_per_head = elems_per_head;

    const uint8_t* k_base = k_cache + block_offsets[block] + (size_t)head * block_bytes_per_head;
    const uint8_t* v_base = v_cache + block_offsets[block] + (size_t)head * block_bytes_per_head;

    k_out[idx] = psp_from_float<scalar_t>(psp_packed_dequant<scalar_t>(k_base, elem, fmt, scale));
    v_out[idx] = psp_from_float<scalar_t>(psp_packed_dequant<scalar_t>(v_base, elem, fmt, scale));
}

template <typename scalar_t>
void launch_dequant_kv(
    const uint8_t* k_cache, const uint8_t* v_cache,
    const uint32_t* block_offsets, const uint8_t* precision_table,
    const float* scale_table,
    scalar_t* k_out, scalar_t* v_out,
    int num_blocks, int num_kv_heads, int block_size, int head_dim,
    cudaStream_t stream)
{
    int total = num_blocks * num_kv_heads * block_size * head_dim;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    dequant_kv_kernel<scalar_t><<<blocks, threads, 0, stream>>>(
        k_cache, v_cache, block_offsets, precision_table, scale_table,
        k_out, v_out, num_blocks, num_kv_heads, block_size, head_dim);
}

template void launch_dequant_kv<half>(
    const uint8_t*, const uint8_t*, const uint32_t*, const uint8_t*, const float*,
    half*, half*, int, int, int, int, cudaStream_t);
template void launch_dequant_kv<__nv_bfloat16>(
    const uint8_t*, const uint8_t*, const uint32_t*, const uint8_t*, const float*,
    __nv_bfloat16*, __nv_bfloat16*, int, int, int, int, cudaStream_t);
template void launch_dequant_kv<float>(
    const uint8_t*, const uint8_t*, const uint32_t*, const uint8_t*, const float*,
    float*, float*, int, int, int, int, cudaStream_t);
template void launch_dequant_kv<__nv_fp8_e4m3>(
    const uint8_t*, const uint8_t*, const uint32_t*, const uint8_t*, const float*,
    __nv_fp8_e4m3*, __nv_fp8_e4m3*, int, int, int, int, cudaStream_t);
template void launch_dequant_kv<__nv_fp8_e5m2>(
    const uint8_t*, const uint8_t*, const uint32_t*, const uint8_t*, const float*,
    __nv_fp8_e5m2*, __nv_fp8_e5m2*, int, int, int, int, cudaStream_t);
