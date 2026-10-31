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
    if (fmt == 3) return 7.0f;      // fp4 (linear int4, magnitude 0..7)
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
    float* __restrict__ max_abs,                   // [2 * num_blocks, num_kv_heads, num_groups] (K | V)
    int num_tokens, int num_kv_heads, int block_size, int head_dim, int num_blocks)
{
    // One thread per (token, head, scale-group). Each thread reduces one
    // PSP_KV_SCALE_GROUP-wide slice; per-group partials are combined by the
    // atomicMax below so low-precision formats get a per-channel-magnitude
    // scale (a single max-abs scale cannot cover the K cache's dynamic range).
    const int num_groups = psp_kv_num_groups(head_dim);
    const int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int token = global_tid / (num_kv_heads * num_groups);
    const int h = (global_tid / num_groups) % num_kv_heads;
    const int group = global_tid % num_groups;
    if (token >= num_tokens) return;

    const int64_t slot = slot_mapping[token];
    const int block_id = static_cast<int>(slot / block_size);
    const int fp8_fmt = static_cast<int>(precision_table[block_id]);
    if (fp8_fmt == 0) return;  // lossless: scale stays 1.0

    const scalar_t* k_row = prefill_scratch +
        ((static_cast<int64_t>(token) * 2 + 0) * num_kv_heads + h) * head_dim
        + group * PSP_KV_SCALE_GROUP;
    const scalar_t* v_row = prefill_scratch +
        ((static_cast<int64_t>(token) * 2 + 1) * num_kv_heads + h) * head_dim
        + group * PSP_KV_SCALE_GROUP;

    // Separate K and V magnitudes so the derive can pick the precision for
    // each (K ~ fp8 range, V ~ much smaller -> fp4).
    float mk = 0.0f, mv = 0.0f;
    #pragma unroll
    for (int d = 0; d < PSP_KV_SCALE_GROUP; ++d) {
        mk = fmaxf(mk, fabsf(psp_to_float(k_row[d])));
        mv = fmaxf(mv, fabsf(psp_to_float(v_row[d])));
    }

    // max_abs is >= 0, so its IEEE-754 bit pattern is monotonically ordered
    // and a signed int atomicMax yields the float max. Layout: [K | V], each
    // with (block, head, group) innermost.
    const int k_idx = (block_id * num_kv_heads + h) * num_groups + group;
    const int v_base = num_blocks * num_kv_heads * num_groups;
    atomicMax(reinterpret_cast<int*>(&max_abs[k_idx]), __float_as_int(mk));
    atomicMax(reinterpret_cast<int*>(&max_abs[v_base + k_idx]), __float_as_int(mv));
}

// ---------------------------------------------------------------------------
// Pass 1b: decode-token max-abs for the derive. Reads K/V separately (no
// host-side stack) and only fires for the FIRST token of a block (the block's
// precision is fixed once its first token is written; later tokens reuse it).
// ---------------------------------------------------------------------------
template <typename scalar_t>
__global__ void estimate_decode_block_scales_kernel(
    const scalar_t* __restrict__ key,       // [num_tokens, num_kv_heads, head_dim]
    const scalar_t* __restrict__ value,     // [num_tokens, num_kv_heads, head_dim]
    const int64_t* __restrict__ slot_mapping,  // [num_tokens]
    float* __restrict__ max_abs,            // [2 * num_blocks, num_kv_heads, num_groups]
    int num_tokens, int num_kv_heads, int block_size, int head_dim, int num_blocks)
{
    const int num_groups = psp_kv_num_groups(head_dim);
    const int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int token = global_tid / (num_kv_heads * num_groups);
    const int h = (global_tid / num_groups) % num_kv_heads;
    const int group = global_tid % num_groups;
    if (token >= num_tokens) return;

    const int64_t slot = slot_mapping[token];
    const int block_id = static_cast<int>(slot / block_size);
    // Only the first token of a block triggers a (re)derive; subsequent tokens
    // in the same block are quantized with that block's already-derived format.
    if ((slot % block_size) != 0) return;

    const scalar_t* k_row = key + ((int64_t)token * num_kv_heads + h) * head_dim + group * PSP_KV_SCALE_GROUP;
    const scalar_t* v_row = value + ((int64_t)token * num_kv_heads + h) * head_dim + group * PSP_KV_SCALE_GROUP;

    float mk = 0.0f, mv = 0.0f;
    #pragma unroll
    for (int d = 0; d < PSP_KV_SCALE_GROUP; ++d) {
        mk = fmaxf(mk, fabsf(psp_to_float(k_row[d])));
        mv = fmaxf(mv, fabsf(psp_to_float(v_row[d])));
    }
    const int k_idx = (block_id * num_kv_heads + h) * num_groups + group;
    const int v_base = num_blocks * num_kv_heads * num_groups;
    atomicMax(reinterpret_cast<int*>(&max_abs[k_idx]), __float_as_int(mk));
    atomicMax(reinterpret_cast<int*>(&max_abs[v_base + k_idx]), __float_as_int(mv));
}

// ---------------------------------------------------------------------------
// Pass 2: convert max-abs -> scale and store into scale_table.
// ---------------------------------------------------------------------------
__global__ void finalize_block_scales_kernel(
    const float* __restrict__ max_abs,             // [2, num_blocks, num_kv_heads, num_groups] (K | V)
    const uint8_t* __restrict__ precision_table,   // [num_blocks] packed
    float* __restrict__ scale_table,               // [2, num_blocks, num_kv_heads, num_groups] (K | V)
    int num_blocks, int num_kv_heads, int head_dim)
{
    const int num_groups = psp_kv_num_groups(head_dim);
    const int stride = num_kv_heads * num_groups;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_blocks * stride) return;

    int block_id = idx / stride;
    int packed = static_cast<int>(precision_table[block_id]);

    auto finalize = [&](int fmt, int m_base, int s_base) {
        float scale;
        if (fmt == 0) {
            scale = 1.0f;  // lossless: no quantization
        } else if (fmt == 3) {
            scale = max_abs[m_base + idx] / psp_kv_fp_max(fmt);  // fp4: per-group
        } else {
            // fp8: broadcast the head's max-abs across its groups so the
            // tensor-core kernels (which apply one scale per block+head) read a
            // consistent scale from group 0.
            const int group = idx % num_groups;
            const int head_base = idx - group;
            float m = 0.0f;
            for (int g = 0; g < num_groups; ++g)
                m = fmaxf(m, max_abs[m_base + head_base + g]);
            scale = m / psp_kv_fp_max(fmt);
        }
        if (scale < 1e-8f) scale = 1.0f;  // all-zero block: avoid zero scale
        scale_table[s_base + idx] = scale;
    };

    const int vb = num_blocks * stride;
    finalize(psp_kv_fmt_k(packed), 0, 0);        // K
    finalize(psp_kv_fmt_v(packed), vb, vb);      // V
}

// Decode finalize: only refreshes a scale when that operand's max-abs was
// estimated this step (non-zero); already-derived blocks keep their scale.
__global__ void finalize_block_scales_decode_kernel(
    const float* __restrict__ max_abs,             // [2, num_blocks, num_kv_heads, num_groups] (K | V)
    const uint8_t* __restrict__ precision_table,   // [num_blocks] packed
    float* __restrict__ scale_table,               // [2, num_blocks, num_kv_heads, num_groups] (K | V)
    int num_blocks, int num_kv_heads, int head_dim)
{
    const int num_groups = psp_kv_num_groups(head_dim);
    const int stride = num_kv_heads * num_groups;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_blocks * stride) return;

    int block_id = idx / stride;
    int packed = static_cast<int>(precision_table[block_id]);

    auto finalize_one = [&](int fmt, int m_base, int s_base) {
        float scale;
        if (fmt == 0) {
            float m = max_abs[m_base + idx];
            if (m == 0.0f) return;  // not estimated this step: keep existing scale
            scale = 1.0f;
        } else if (fmt == 3) {
            float m = max_abs[m_base + idx];
            if (m == 0.0f) return;
            scale = m / psp_kv_fp_max(fmt);
        } else {
            const int group = idx % num_groups;
            const int head_base = idx - group;
            float m = 0.0f;
            for (int g = 0; g < num_groups; ++g)
                m = fmaxf(m, max_abs[m_base + head_base + g]);
            if (m == 0.0f) return;  // not estimated this step
            scale = m / psp_kv_fp_max(fmt);
        }
        if (scale < 1e-8f) scale = 1.0f;
        scale_table[s_base + idx] = scale;
    };

    const int vb = num_blocks * stride;
    finalize_one(psp_kv_fmt_k(packed), 0, 0);
    finalize_one(psp_kv_fmt_v(packed), vb, vb);
}

// ---------------------------------------------------------------------------
// Pass 2b (optional): derive per-block precision from the max-abs magnitude.
// Reuses the Pass-1 max_abs (no separate Torch reduction): maps the worst-head
// max-abs to lossless/fp8/fp4 via a logarithmic threshold, writing
// precision_table. Mirrors _derive_precision_from_scratch in the model runner.
// ---------------------------------------------------------------------------
__global__ void derive_precision_kernel(
    const float* __restrict__ max_abs,        // [2, num_blocks, num_kv_heads, num_groups] (K | V)
    uint8_t* __restrict__ precision_table,    // [num_blocks] packed k_fmt | (v_fmt << 2)
    int num_blocks, int num_kv_heads, int head_dim, int k_default, int v_default, float E)
{
    int block_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (block_id >= num_blocks) return;

    const int num_groups = psp_kv_num_groups(head_dim);
    const float fp_max = 240.0f; // fp8 e4m3 max finite

    auto derive = [&](int base, int default_kv) -> uint8_t {
        float m = 0.0f;
        for (int h = 0; h < num_kv_heads; ++h) {
            for (int g = 0; g < num_groups; ++g) {
                m = fmaxf(m, max_abs[base + (block_id * num_kv_heads + h) * num_groups + g]);
            }
        }
        float required_bits = log2f(fmaxf(m, 1e-12f) / (E * fp_max));
        if (m == 0.0f) {
            return static_cast<uint8_t>(default_kv);
        } else if (required_bits > 1.0f) {
            return 1;  // fp8 e4m3
        } else {
            return 3;  // fp4 e2m1
        }
    };

    const int vb = num_blocks * num_kv_heads * num_groups;
    const uint8_t k_prec = derive(0, k_default);
    // V adapts too: fp4 by default, upgraded to fp8 when a block's worst-head
    // magnitude needs more than ~2 mantissa bits within tolerance E. The
    // per-K/V tensor-core kernel reads both formats per block, so a mixed
    // fp4/fp8 V table now stays on the tensor-core path.
    const uint8_t v_prec = derive(vb, v_default);
    precision_table[block_id] = k_prec | (v_prec << 2);
}

// ---------------------------------------------------------------------------
// Decode derive: like derive_precision_kernel but only OVERWRITES a block when
// its max-abs was actually estimated (non-zero). Blocks the estimate skipped
// (already-derived blocks, or non-first-token decode steps) keep their format.
// ---------------------------------------------------------------------------
__global__ void derive_precision_decode_kernel(
    const float* __restrict__ max_abs,     // [2, num_blocks, num_kv_heads, num_groups] (K | V)
    uint8_t* __restrict__ precision_table, // [num_blocks] packed k_fmt | (v_fmt << 2)
    int num_blocks, int num_kv_heads, int head_dim, int k_default, int v_default, float E)
{
    int block_id = blockIdx.x * blockDim.x + threadIdx.x;
    if (block_id >= num_blocks) return;

    const int num_groups = psp_kv_num_groups(head_dim);
    const float fp_max = 240.0f;

    auto worst = [&](int base) {
        float m = 0.0f;
        for (int h = 0; h < num_kv_heads; ++h)
            for (int g = 0; g < num_groups; ++g)
                m = fmaxf(m, max_abs[base + (block_id * num_kv_heads + h) * num_groups + g]);
        return m;
    };
    auto derive = [&](float m, int default_kv) -> uint8_t {
        float required_bits = log2f(fmaxf(m, 1e-12f) / (E * fp_max));
        if (required_bits > 1.0f) return 1;   // fp8 e4m3
        return 3;                              // fp4 e2m1
    };

    const int vb = num_blocks * num_kv_heads * num_groups;
    const float mk = worst(0);
    const float mv = worst(vb);
    // Skip blocks whose max-abs was not estimated this step (keep existing).
    if (mk == 0.0f && mv == 0.0f) return;
    // Decode blocks keep their K precision (K default is already fp8, and
    // changing a block's K byte size would shift the V region and misalign the
    // already-written V data). Only V is re-derived here.
    const uint8_t k_prec = psp_kv_fmt_k(precision_table[block_id]);
    const uint8_t v_prec = (mv == 0.0f) ? psp_kv_fmt_v(precision_table[block_id])
                                        : derive(mv, v_default);
    precision_table[block_id] = k_prec | (v_prec << 2);
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
    const uint32_t* __restrict__ block_offsets,   // [2 * num_blocks] byte offsets (K | V)
    const float* __restrict__ scale_table,     // [2 * num_blocks, num_kv_heads] (K | V)
    int num_tokens, int num_kv_heads, int block_size, int head_dim,
    int num_blocks, int format_generation)
{
    // One thread per (token, head, scale-group): each thread quantizes one
    // PSP_KV_SCALE_GROUP-wide slice of a row pair with that group's scale.
    const int num_groups = psp_kv_num_groups(head_dim);
    const int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int token = global_tid / (num_kv_heads * num_groups);
    const int h = (global_tid / num_groups) % num_kv_heads;
    const int group = global_tid % num_groups;
    if (token >= num_tokens) return;

    const int64_t slot = slot_mapping[token];
    const int block_id = static_cast<int>(slot / block_size);
    const int block_offset = static_cast<int>(slot % block_size);
    const int packed = static_cast<int>(precision_table[block_id]);
    const int k_fmt = psp_kv_fmt_k(packed);
    const int v_fmt = psp_kv_fmt_v(packed);

    auto bytes_per_head = [&](int fmt) -> int {
        if (fmt == 0)      return block_size * head_dim * (int)sizeof(scalar_t);
        else if (fmt == 3) return block_size * head_dim / 2;
        else               return block_size * head_dim;
    };
    auto bytes_per_row = [&](int fmt) -> int {
        if (fmt == 0)      return head_dim * (int)sizeof(scalar_t);
        else if (fmt == 3) return head_dim / 2;
        else               return head_dim;
    };

    const int k_block_bytes = bytes_per_head(k_fmt);
    const int v_block_bytes = bytes_per_head(v_fmt);
    const int k_row_bytes = bytes_per_row(k_fmt);

    const int64_t hh = (int64_t)num_blocks * num_kv_heads * num_groups;

    const int elem_base = group * PSP_KV_SCALE_GROUP;
    const scalar_t* k_row = prefill_scratch +
        ((static_cast<int64_t>(token) * 2 + 0) * num_kv_heads + h) * head_dim + elem_base;
    const scalar_t* v_row = prefill_scratch +
        ((static_cast<int64_t>(token) * 2 + 1) * num_kv_heads + h) * head_dim + elem_base;

    // Use the stored per-group scale; for a fresh block (scale 0) fall back to
    // an on-the-fly scale from this group's magnitude so decode-like steps
    // (speculative verify) don't quantize with scale 0.
    auto compute_scale = [&](const scalar_t* row, int fmt) -> float {
        if (fmt == 0) return 1.0f;
        float m = 0.0f;
        for (int i = 0; i < PSP_KV_SCALE_GROUP; ++i) m = fmaxf(m, fabsf(psp_to_float(row[i])));
        float s = m / psp_kv_fp_max(fmt);
        return (s < 1e-8f) ? 1.0f : s;
    };
    const int64_t gidx = ((int64_t)block_id * num_kv_heads + h) * num_groups + group;
    const float k_scale_stored = scale_table[gidx];
    const float v_scale_stored = scale_table[hh + gidx];
    const float k_scale = (k_scale_stored > 1e-8f) ? k_scale_stored : compute_scale(k_row, k_fmt);
    const float v_scale = (v_scale_stored > 1e-8f) ? v_scale_stored : compute_scale(v_row, v_fmt);

    uint8_t* k_out = key_cache + block_offsets[block_id]
                   + (size_t)h * k_block_bytes
                   + block_offset * k_row_bytes;
    uint8_t* v_out = value_cache + block_offsets[num_blocks + block_id]
                   + (size_t)h * v_block_bytes;

    switch (k_fmt) {
        case 0: psp_quantize_row_k<scalar_t, 0>(k_row, k_out, PSP_KV_SCALE_GROUP, elem_base, k_scale); break;
        case 1: psp_quantize_row_k<scalar_t, 1>(k_row, k_out, PSP_KV_SCALE_GROUP, elem_base, k_scale); break;
        case 2: psp_quantize_row_k<scalar_t, 2>(k_row, k_out, PSP_KV_SCALE_GROUP, elem_base, k_scale); break;
        default: psp_quantize_row_k<scalar_t, 3>(k_row, k_out, PSP_KV_SCALE_GROUP, elem_base, k_scale); break;
    }
    switch (v_fmt) {
        case 0: psp_quantize_row_v<scalar_t, 0>(v_row, v_out, head_dim, PSP_KV_SCALE_GROUP, elem_base, v_scale, block_offset); break;
        case 1: psp_quantize_row_v<scalar_t, 1>(v_row, v_out, head_dim, PSP_KV_SCALE_GROUP, elem_base, v_scale, block_offset); break;
        case 2: psp_quantize_row_v<scalar_t, 2>(v_row, v_out, head_dim, PSP_KV_SCALE_GROUP, elem_base, v_scale, block_offset); break;
        default: psp_quantize_row_v<scalar_t, 3>(v_row, v_out, head_dim, PSP_KV_SCALE_GROUP, elem_base, v_scale, block_offset); break;
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
    // The estimate and the quantize use one thread per (token, head, group).
    const int num_groups = psp_kv_num_groups(head_dim);
    const int pair_blocks =
        ((int64_t)num_tokens * num_kv_heads * num_groups + threads - 1) / threads;
    const int scale_blocks =
        (num_blocks * num_kv_heads * num_groups + threads - 1) / threads;

    estimate_block_scales_kernel<scalar_t><<<pair_blocks, threads, 0, stream>>>(
        prefill_scratch, slot_mapping, precision_table, max_abs,
        num_tokens, num_kv_heads, block_size, head_dim, num_blocks);

    finalize_block_scales_kernel<<<scale_blocks, threads, 0, stream>>>(
        max_abs, precision_table, scale_table, num_blocks, num_kv_heads, head_dim);

    quantize_and_scatter_kernel<scalar_t><<<pair_blocks, threads, 0, stream>>>(
        prefill_scratch, key_cache, value_cache, slot_mapping, precision_table,
        block_offsets, scale_table, num_tokens, num_kv_heads, block_size, head_dim,
        num_blocks, format_generation);
}

// Scatter-only variant: quantize the scratch with the EXISTING precision/scale
// (on-the-fly scale for fresh blocks) and scatter. No estimate/finalize pass,
// so decode-like multi-token steps (speculative verify) don't clobber the
// accepted blocks' already-finalized scale.
template <typename scalar_t>
void launch_quantize_scatter_multi(
    const scalar_t* prefill_scratch,
    uint8_t* key_cache,
    uint8_t* value_cache,
    const int64_t* slot_mapping,
    const uint8_t* precision_table,
    const uint32_t* block_offsets,
    const float* scale_table,
    int num_tokens, int num_blocks, int num_kv_heads, int block_size, int head_dim,
    int format_generation, cudaStream_t stream)
{
    const int threads = 256;
    const int num_groups = psp_kv_num_groups(head_dim);
    const int pair_blocks =
        ((int64_t)num_tokens * num_kv_heads * num_groups + threads - 1) / threads;
    quantize_and_scatter_kernel<scalar_t><<<pair_blocks, threads, 0, stream>>>(
        prefill_scratch, key_cache, value_cache, slot_mapping, precision_table,
        block_offsets, scale_table, num_tokens, num_kv_heads, block_size, head_dim,
        num_blocks, format_generation);
}

template void launch_quantize_scatter_multi<half>(
    const half*, uint8_t*, uint8_t*, const int64_t*, const uint8_t*, const uint32_t*,
    const float*, int, int, int, int, int, int, cudaStream_t);
template void launch_quantize_scatter_multi<__nv_bfloat16>(
    const __nv_bfloat16*, uint8_t*, uint8_t*, const int64_t*, const uint8_t*, const uint32_t*,
    const float*, int, int, int, int, int, int, cudaStream_t);
template void launch_quantize_scatter_multi<float>(
    const float*, uint8_t*, uint8_t*, const int64_t*, const uint8_t*, const uint32_t*,
    const float*, int, int, int, int, int, int, cudaStream_t);
template void launch_quantize_scatter_multi<__nv_fp8_e4m3>(
    const __nv_fp8_e4m3*, uint8_t*, uint8_t*, const int64_t*, const uint8_t*, const uint32_t*,
    const float*, int, int, int, int, int, int, cudaStream_t);
template void launch_quantize_scatter_multi<__nv_fp8_e5m2>(
    const __nv_fp8_e5m2*, uint8_t*, uint8_t*, const int64_t*, const uint8_t*, const uint32_t*,
    const float*, int, int, int, int, int, int, cudaStream_t);

template void launch_quantize_and_scatter<half>(
    const half*, uint8_t*, uint8_t*, const int64_t*, const uint8_t*, const uint32_t*,
    float*, float*, int, int, int, int, int, int, cudaStream_t);
template void launch_quantize_and_scatter<__nv_bfloat16>(
    const __nv_bfloat16*, uint8_t*, uint8_t*, const int64_t*, const uint8_t*, const uint32_t*,
    float*, float*, int, int, int, int, int, int, cudaStream_t);
template void launch_quantize_and_scatter<float>(
    const float*, uint8_t*, uint8_t*, const int64_t*, const uint8_t*, const uint32_t*,
    float*, float*, int, int, int, int, int, int, cudaStream_t);
template void launch_quantize_and_scatter<__nv_fp8_e4m3>(
    const __nv_fp8_e4m3*, uint8_t*, uint8_t*, const int64_t*, const uint8_t*, const uint32_t*,
    float*, float*, int, int, int, int, int, int, cudaStream_t);
template void launch_quantize_and_scatter<__nv_fp8_e5m2>(
    const __nv_fp8_e5m2*, uint8_t*, uint8_t*, const int64_t*, const uint8_t*, const uint32_t*,
    float*, float*, int, int, int, int, int, int, cudaStream_t);

// Fused precision derivation: Pass-1 max-abs + the derive kernel. Writes the
// per-block precision_table (no separate Torch pass in the model runner).
template <typename scalar_t>
void launch_derive_precision(
    const scalar_t* prefill_scratch,
    const int64_t* slot_mapping,
    uint8_t* precision_table,
    float* max_abs,
    int num_tokens, int num_blocks, int num_kv_heads, int block_size, int head_dim,
    int k_default, int v_default, float E, cudaStream_t stream)
{
    const int threads = 256;
    const int num_groups = psp_kv_num_groups(head_dim);
    const int pair_blocks = ((int64_t)num_tokens * num_kv_heads * num_groups + threads - 1) / threads;
    const int block_blocks = (num_blocks + threads - 1) / threads;

    estimate_block_scales_kernel<scalar_t><<<pair_blocks, threads, 0, stream>>>(
        prefill_scratch, slot_mapping, precision_table, max_abs,
        num_tokens, num_kv_heads, block_size, head_dim, num_blocks);
    derive_precision_kernel<<<block_blocks, threads, 0, stream>>>(
        max_abs, precision_table, num_blocks, num_kv_heads, head_dim, k_default, v_default, E);
}

template void launch_derive_precision<half>(
    const half*, const int64_t*, uint8_t*, float*, int, int, int, int, int, int, int, float, cudaStream_t);
template void launch_derive_precision<__nv_bfloat16>(
    const __nv_bfloat16*, const int64_t*, uint8_t*, float*, int, int, int, int, int, int, int, float, cudaStream_t);
template void launch_derive_precision<float>(
    const float*, const int64_t*, uint8_t*, float*, int, int, int, int, int, int, int, float, cudaStream_t);
template void launch_derive_precision<__nv_fp8_e4m3>(
    const __nv_fp8_e4m3*, const int64_t*, uint8_t*, float*, int, int, int, int, int, int, int, float, cudaStream_t);
template void launch_derive_precision<__nv_fp8_e5m2>(
    const __nv_fp8_e5m2*, const int64_t*, uint8_t*, float*, int, int, int, int, int, int, int, float, cudaStream_t);

// Decode derive: estimate the first-token max-abs, derive the block precision,
// and finalize the block's scale so the following quantize_single_token sees a
// consistent (precision, scale) pair. Reuses the prefill derive + finalize.
template <typename scalar_t>
void launch_derive_precision_decode(
    const scalar_t* key, const scalar_t* value,
    const int64_t* slot_mapping,
    uint8_t* precision_table, float* scale_table, float* max_abs,
    int num_tokens, int num_blocks, int num_kv_heads, int block_size, int head_dim,
    int k_default, int v_default, float E, cudaStream_t stream)
{
    const int threads = 256;
    const int num_groups = psp_kv_num_groups(head_dim);
    const int pair_blocks = ((int64_t)num_tokens * num_kv_heads * num_groups + threads - 1) / threads;
    const int block_blocks = (num_blocks + threads - 1) / threads;
    const int scale_blocks = (num_blocks * num_kv_heads * num_groups + threads - 1) / threads;

    estimate_decode_block_scales_kernel<scalar_t><<<pair_blocks, threads, 0, stream>>>(
        key, value, slot_mapping, max_abs,
        num_tokens, num_kv_heads, block_size, head_dim, num_blocks);
    derive_precision_decode_kernel<<<block_blocks, threads, 0, stream>>>(
        max_abs, precision_table, num_blocks, num_kv_heads, head_dim, k_default, v_default, E);
    finalize_block_scales_decode_kernel<<<scale_blocks, threads, 0, stream>>>(
        max_abs, precision_table, scale_table, num_blocks, num_kv_heads, head_dim);
}

template void launch_derive_precision_decode<half>(
    const half*, const half*, const int64_t*, uint8_t*, float*, float*, int, int, int, int, int, int, int, float, cudaStream_t);
template void launch_derive_precision_decode<__nv_bfloat16>(
    const __nv_bfloat16*, const __nv_bfloat16*, const int64_t*, uint8_t*, float*, float*, int, int, int, int, int, int, int, float, cudaStream_t);
template void launch_derive_precision_decode<float>(
    const float*, const float*, const int64_t*, uint8_t*, float*, float*, int, int, int, int, int, int, int, float, cudaStream_t);
template void launch_derive_precision_decode<__nv_fp8_e4m3>(
    const __nv_fp8_e4m3*, const __nv_fp8_e4m3*, const int64_t*, uint8_t*, float*, float*, int, int, int, int, int, int, int, float, cudaStream_t);
template void launch_derive_precision_decode<__nv_fp8_e5m2>(
    const __nv_fp8_e5m2*, const __nv_fp8_e5m2*, const int64_t*, uint8_t*, float*, float*, int, int, int, int, int, int, int, float, cudaStream_t);

// Quantize + scatter a single decode token using the block's EXISTING scale
// (no scale estimation: the prefill already populated scale_table). The key and
// value are passed separately (no host-side stack) so the hot decode write-back
// path avoids a tensor allocation.
template <typename scalar_t>
__global__ void quantize_single_token_kernel(
    const scalar_t* __restrict__ key,       // [num_tokens, num_kv_heads, head_dim]
    const scalar_t* __restrict__ value,     // [num_tokens, num_kv_heads, head_dim]
    uint8_t* __restrict__ key_cache, uint8_t* __restrict__ value_cache,
    const int64_t* __restrict__ slot_mapping,   // [num_tokens]
    const uint8_t* __restrict__ precision_table,
    const uint32_t* __restrict__ block_offsets,
    const float* __restrict__ scale_table,
    int num_tokens, int num_kv_heads, int block_size, int head_dim, int num_blocks)
{
    const int num_groups = psp_kv_num_groups(head_dim);
    const int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int token = global_tid / (num_kv_heads * num_groups);
    const int h = (global_tid / num_groups) % num_kv_heads;
    const int group = global_tid % num_groups;
    if (token >= num_tokens) return;

    const int64_t slot = slot_mapping[token];
    const int block_id = static_cast<int>(slot / block_size);
    const int block_offset = static_cast<int>(slot % block_size);
    const int packed = static_cast<int>(precision_table[block_id]);
    const int k_fmt = psp_kv_fmt_k(packed);
    const int v_fmt = psp_kv_fmt_v(packed);

    auto bytes_per_head = [&](int fmt) -> int {
        if (fmt == 0)      return block_size * head_dim * (int)sizeof(scalar_t);
        else if (fmt == 3) return block_size * head_dim / 2;
        else               return block_size * head_dim;
    };
    auto bytes_per_row = [&](int fmt) -> int {
        if (fmt == 0)      return head_dim * (int)sizeof(scalar_t);
        else if (fmt == 3) return head_dim / 2;
        else               return head_dim;
    };

    const int k_block_bytes = bytes_per_head(k_fmt);
    const int v_block_bytes = bytes_per_head(v_fmt);
    const int k_row_bytes = bytes_per_row(k_fmt);

    const int64_t hh = (int64_t)num_blocks * num_kv_heads * num_groups;
    const int elem_base = group * PSP_KV_SCALE_GROUP;
    const scalar_t* k_row = key + ((size_t)token * num_kv_heads + h) * head_dim + elem_base;
    const scalar_t* v_row = value + ((size_t)token * num_kv_heads + h) * head_dim + elem_base;

    // A freshly-allocated decode block has scale 0 in scale_table (it was only
    // derived for the prefill). Fall back to an on-the-fly scale from this
    // group's magnitude so the first token of a new block isn't quantized with
    // scale 0 (which would saturate to +/-max).
    auto compute_scale = [&](const scalar_t* row, int fmt) -> float {
        if (fmt == 0) return 1.0f;  // lossless: no scaling
        float m = 0.0f;
        for (int i = 0; i < PSP_KV_SCALE_GROUP; ++i) m = fmaxf(m, fabsf(psp_to_float(row[i])));
        float s = m / psp_kv_fp_max(fmt);
        return (s < 1e-8f) ? 1.0f : s;
    };
    const int64_t gidx = ((int64_t)block_id * num_kv_heads + h) * num_groups + group;
    const float k_scale_stored = scale_table[gidx];
    const float v_scale_stored = scale_table[hh + gidx];
    const float k_scale = (k_scale_stored > 1e-8f) ? k_scale_stored : compute_scale(k_row, k_fmt);
    const float v_scale = (v_scale_stored > 1e-8f) ? v_scale_stored : compute_scale(v_row, v_fmt);
    uint8_t* k_out = key_cache + block_offsets[block_id]
                   + (size_t)h * k_block_bytes
                   + block_offset * k_row_bytes;
    uint8_t* v_out = value_cache + block_offsets[num_blocks + block_id]
                   + (size_t)h * v_block_bytes;

    switch (k_fmt) {
        case 0: psp_quantize_row_k<scalar_t, 0>(k_row, k_out, PSP_KV_SCALE_GROUP, elem_base, k_scale); break;
        case 1: psp_quantize_row_k<scalar_t, 1>(k_row, k_out, PSP_KV_SCALE_GROUP, elem_base, k_scale); break;
        case 2: psp_quantize_row_k<scalar_t, 2>(k_row, k_out, PSP_KV_SCALE_GROUP, elem_base, k_scale); break;
        default: psp_quantize_row_k<scalar_t, 3>(k_row, k_out, PSP_KV_SCALE_GROUP, elem_base, k_scale); break;
    }
    switch (v_fmt) {
        case 0: psp_quantize_row_v<scalar_t, 0>(v_row, v_out, head_dim, PSP_KV_SCALE_GROUP, elem_base, v_scale, block_offset); break;
        case 1: psp_quantize_row_v<scalar_t, 1>(v_row, v_out, head_dim, PSP_KV_SCALE_GROUP, elem_base, v_scale, block_offset); break;
        case 2: psp_quantize_row_v<scalar_t, 2>(v_row, v_out, head_dim, PSP_KV_SCALE_GROUP, elem_base, v_scale, block_offset); break;
        default: psp_quantize_row_v<scalar_t, 3>(v_row, v_out, head_dim, PSP_KV_SCALE_GROUP, elem_base, v_scale, block_offset); break;
    }
}

template <typename scalar_t>
void launch_quantize_single_token(
    const scalar_t* key, const scalar_t* value,
    uint8_t* key_cache, uint8_t* value_cache,
    const int64_t* slot_mapping,
    const uint8_t* precision_table, const uint32_t* block_offsets,
    const float* scale_table,
    int num_tokens, int num_kv_heads, int block_size, int head_dim, int num_blocks,
    cudaStream_t stream)
{
    const int threads = 256;
    const int num_groups = psp_kv_num_groups(head_dim);
    const int total = num_tokens * num_kv_heads * num_groups;
    const int pair_blocks = (total + threads - 1) / threads;
    quantize_single_token_kernel<scalar_t><<<pair_blocks, threads, 0, stream>>>(
        key, value, key_cache, value_cache, slot_mapping, precision_table,
        block_offsets, scale_table, num_tokens, num_kv_heads, block_size, head_dim,
        num_blocks);
}

template void launch_quantize_single_token<half>(
    const half*, const half*, uint8_t*, uint8_t*, const int64_t*, const uint8_t*,
    const uint32_t*, const float*, int, int, int, int, int, cudaStream_t);
template void launch_quantize_single_token<__nv_bfloat16>(
    const __nv_bfloat16*, const __nv_bfloat16*, uint8_t*, uint8_t*, const int64_t*,
    const uint8_t*, const uint32_t*, const float*, int, int, int, int, int, cudaStream_t);
template void launch_quantize_single_token<float>(
    const float*, const float*, uint8_t*, uint8_t*, const int64_t*,
    const uint8_t*, const uint32_t*, const float*, int, int, int, int, int, cudaStream_t);
template void launch_quantize_single_token<__nv_fp8_e4m3>(
    const __nv_fp8_e4m3*, const __nv_fp8_e4m3*, uint8_t*, uint8_t*, const int64_t*,
    const uint8_t*, const uint32_t*, const float*, int, int, int, int, int, cudaStream_t);
template void launch_quantize_single_token<__nv_fp8_e5m2>(
    const __nv_fp8_e5m2*, const __nv_fp8_e5m2*, uint8_t*, uint8_t*, const int64_t*,
    const uint8_t*, const uint32_t*, const float*, int, int, int, int, int, cudaStream_t);
