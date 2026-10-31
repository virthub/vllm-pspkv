// virthub/kernels/common/quantization_utils.cuh

#ifndef PSP_KV_QUANTIZATION_UTILS_CUH
#define PSP_KV_QUANTIZATION_UTILS_CUH

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <math.h>
#include <stdint.h>

// Per-scale-group granularity. The per-(block, head) scale is split into
// groups of PSP_KV_SCALE_GROUP dims so low-precision formats (fp4 e2m1 has a
// ~6x representable range) can track the per-channel magnitude spread: the K
// cache spans ~100x across dims, which a single max-abs scale cannot cover.
// The scale table's innermost dimension becomes head_dim / PSP_KV_SCALE_GROUP.
#define PSP_KV_SCALE_GROUP 16

__device__ __host__ inline int psp_kv_num_groups(int head_dim) {
    return head_dim / PSP_KV_SCALE_GROUP;
}

// Flat index into a per-(block, head, group) scale table (within one K or V
// section). The caller adds the V section offset (num_blocks * num_kv_heads *
// num_groups) for the value scales.
__device__ inline int psp_kv_scale_idx(int block, int kv_head, int num_kv_heads,
                                       int num_groups, int group) {
    return (block * num_kv_heads + kv_head) * num_groups + group;
}

// Forward declarations of the FP8 helpers (defined later in this header).
__host__ __device__ inline float dequantize_fp8_e4m3(uint8_t q);
__host__ __device__ inline float dequantize_fp8_e5m2(uint8_t q);
__host__ __device__ inline uint8_t quantize_fp8_e4m3(float value);
__host__ __device__ inline uint8_t quantize_fp8_e5m2(float value);

// Scalar (query/output) conversion helpers for templated kernels. PSP-KV
// supports fp16, bf16, and fp8 (e4m3/e5m2) "target precision" (query/output).
template <typename scalar_t> __device__ inline float psp_to_float(scalar_t x) {
    return (float)x;
}
template <> __device__ inline float psp_to_float<half>(half x) {
    return __half2float(x);
}
template <> __device__ inline float psp_to_float<__nv_bfloat16>(__nv_bfloat16 x) {
    return __bfloat162float(x);
}
template <> __device__ inline float psp_to_float<__nv_fp8_e4m3>(__nv_fp8_e4m3 x) {
    return dequantize_fp8_e4m3(*reinterpret_cast<const uint8_t*>(&x));
}
template <> __device__ inline float psp_to_float<__nv_fp8_e5m2>(__nv_fp8_e5m2 x) {
    return dequantize_fp8_e5m2(*reinterpret_cast<const uint8_t*>(&x));
}

template <typename scalar_t> __device__ inline scalar_t psp_from_float(float x) {
    return (scalar_t)x;
}
template <> __device__ inline half psp_from_float<half>(float x) {
    return __float2half(x);
}
template <> __device__ inline __nv_bfloat16 psp_from_float<__nv_bfloat16>(float x) {
    return __float2bfloat16(x);
}
template <> __device__ inline __nv_fp8_e4m3 psp_from_float<__nv_fp8_e4m3>(float x) {
    __nv_fp8_e4m3 r;
    *reinterpret_cast<uint8_t*>(&r) = quantize_fp8_e4m3(x);
    return r;
}
template <> __device__ inline __nv_fp8_e5m2 psp_from_float<__nv_fp8_e5m2>(float x) {
    __nv_fp8_e5m2 r;
    *reinterpret_cast<uint8_t*>(&r) = quantize_fp8_e5m2(x);
    return r;
}

// Portable float<->uint32 bit conversion for both host and device.
__host__ __device__ inline uint32_t float_as_uint(float value) {
    union { float f; uint32_t u; } conv;
    conv.f = value;
    return conv.u;
}
__host__ __device__ inline float uint_as_float(uint32_t value) {
    union { uint32_t u; float f; } conv;
    conv.u = value;
    return conv.f;
}

// FP8 E4M3 (1 sign, 4 exponent, 3 mantissa, bias 7)
__host__ __device__ inline uint8_t quantize_fp8_e4m3(float value) {
    // Native hardware conversion (cvt.rn.fp8.e4m3.f32 on SM 89+); the caller
    // normalizes to the fp8 range so no saturation is needed.
    __nv_fp8_e4m3 f = (__nv_fp8_e4m3)value;
    return *reinterpret_cast<const uint8_t*>(&f);
}

__host__ __device__ inline float dequantize_fp8_e4m3(uint8_t q) {
    // Native hardware conversion (cvt.rn.f32.e4m3 on SM 89+); exact for e4m3.
    __nv_fp8_e4m3 f = *reinterpret_cast<const __nv_fp8_e4m3*>(&q);
    return (float)f;
}

// FP8 E5M2 (1 sign, 5 exponent, 2 mantissa, bias 15)
__host__ __device__ inline uint8_t quantize_fp8_e5m2(float value) {
    // Native hardware conversion (cvt.rn.fp8.e5m2.f32 on SM 89+); the caller
    // normalizes to the fp8 range so no saturation is needed.
    __nv_fp8_e5m2 f = (__nv_fp8_e5m2)value;
    return *reinterpret_cast<const uint8_t*>(&f);
}

__host__ __device__ inline float dequantize_fp8_e5m2(uint8_t q) {
    // Native hardware conversion (cvt.rn.f32.e5m2 on SM 89+); exact for e5m2.
    __nv_fp8_e5m2 f = *reinterpret_cast<const __nv_fp8_e5m2*>(&q);
    return (float)f;
}

// Forward declarations for the FP4 helpers (defined later in this header).
__host__ __device__ inline uint8_t quantize_fp4_e2m1(float value);
__host__ __device__ inline float dequantize_fp4_e2m1(uint8_t q);

// Dispatch helpers keyed by the per-block precision_table value:
//   0 -> lossless (fp16/bf16, 16-bit)
//   1 -> FP8 e4m3 (default)
//   2 -> FP8 e5m2
//   3 -> FP4 e2m1 (low nibble of the 16-bit value)
__host__ __device__ inline float dequantize_fp8(uint8_t q, int fmt) {
    return (fmt == 2) ? dequantize_fp8_e5m2(q) : dequantize_fp8_e4m3(q);
}

__host__ __device__ inline uint8_t quantize_fp8(float value, int fmt) {
    return (fmt == 2) ? quantize_fp8_e5m2(value) : quantize_fp8_e4m3(value);
}

// Read one element from the 16-bit cache container and dequantize it.
//
// The KV cache is a 16-bit container (matching the target precision). The
// precision_table selects the per-block interpretation:
//   fmt == 0 -> lossless: the 16-bit value is fp16/bf16, read directly.
//   fmt 1/2  -> fp8: the fp8 byte sits in the low byte of the 16-bit value.
//   fmt == 3 -> fp4: the fp4 nibble sits in the low nibble of the 16-bit value.
template <typename scalar_t>
__device__ inline float psp_cache_dequant(
    const scalar_t* ptr, int idx, int fmt, float scale)
{
    if (fmt == 0) {
        return psp_to_float(ptr[idx]);
    }
    if (sizeof(scalar_t) == 1) {
        // 8-bit container (fp8 target): the value is the fp8 byte; fp4 in low nibble.
        uint8_t raw = *reinterpret_cast<const uint8_t*>(&ptr[idx]);
        if (fmt == 3) {
            return dequantize_fp4_e2m1((uint8_t)(raw & 0x0Fu)) * scale;
        }
        return dequantize_fp8(raw, fmt) * scale;
    }
    uint16_t raw = *reinterpret_cast<const uint16_t*>(&ptr[idx]);
    if (fmt == 3) {
        return dequantize_fp4_e2m1((uint8_t)(raw & 0x0Fu)) * scale;
    }
    return dequantize_fp8((uint8_t)(raw & 0xFFu), fmt) * scale;
}

// Write one element into the cache container (16-bit for fp16/bf16 target,
// 8-bit for fp8 target).
template <typename scalar_t>
__device__ inline void psp_cache_quantize(
    scalar_t* ptr, int idx, float x, int fmt, float scale)
{
    if (fmt == 0) {
        ptr[idx] = psp_from_float<scalar_t>(x);
    } else if (sizeof(scalar_t) == 1) {
        // 8-bit container (fp8 target): fp8 byte or fp4 low nibble.
        if (fmt == 3) {
            uint8_t q = quantize_fp4_e2m1(x / scale);
            uint8_t raw = (uint8_t)(q & 0x0Fu);
            ptr[idx] = *reinterpret_cast<scalar_t*>(&raw);
        } else {
            uint8_t q = quantize_fp8(x / scale, fmt);
            ptr[idx] = *reinterpret_cast<scalar_t*>(&q);
        }
    } else if (fmt == 3) {
        uint8_t q = quantize_fp4_e2m1(x / scale);
        uint16_t raw = (uint16_t)(q & 0x0Fu);  // low nibble
        ptr[idx] = *reinterpret_cast<scalar_t*>(&raw);
    } else {
        uint8_t q = quantize_fp8(x / scale, fmt);
        uint16_t raw = (uint16_t)q;  // 0x00XX
        ptr[idx] = *reinterpret_cast<scalar_t*>(&raw);
    }
}

// FP4 as a linear signed 4-bit (int4) value: sign-magnitude with magnitude
// 0..7. The per-(block, head, group) scale already absorbs the dynamic range,
// so spending the 4 bits on a uniform mantissa (vs e2m1's coarse 1-bit mantissa
// + 2 exponent bits) roughly halves the quantization error and brings fp4 close
// to fp8 e4m3 accuracy for attention.
__host__ __device__ inline uint8_t quantize_fp4_e2m1(float value) {
    if (isnan(value)) return 0x0;
    float r = rintf(value);
    r = fminf(fmaxf(r, -7.0f), 7.0f);
    uint8_t sign = (r < 0.0f) ? 0x8u : 0x0u;
    return (uint8_t)(sign | ((uint8_t)fabsf(r) & 0x7u));
}

__host__ __device__ inline float dequantize_fp4_e2m1(uint8_t q) {
    int mag = q & 0x7;
    float v = (float)mag;
    return (q & 0x8) ? -v : v;
}

// Variable-stride (byte-buffer) dequant/quantize. The cache is a uint8 byte
// buffer; each block's byte offset comes from `block_offsets`, and the per-element
// byte size follows the precision (lossless = sizeof(scalar_t), fp8 = 1, fp4 = 0.5).
template <typename scalar_t>
__device__ inline float psp_packed_dequant(
    const uint8_t* base, int elem_idx, int fmt, float scale)
{
    if (fmt == 0) {
        const scalar_t* s = reinterpret_cast<const scalar_t*>(base + elem_idx * sizeof(scalar_t));
        return psp_to_float(*s);
    }
    if (fmt == 3) {
        const uint8_t* b = base + (elem_idx >> 1);
        uint8_t nibble = (elem_idx & 1) ? ((*b >> 4) & 0x0Fu) : (*b & 0x0Fu);
        return dequantize_fp4_e2m1(nibble) * scale;
    }
    return dequantize_fp8(base[elem_idx], fmt) * scale;
}

// Vectorized dequant: 4 packed elements -> float4 in one load + one conversion.
// ``base`` is the block byte pointer and ``elem_idx`` is the logical element
// index (0-based within the block), matching ``psp_packed_dequant``'s
// addressing so the byte offset is derived from the format.
template <typename scalar_t>
__device__ inline float4 psp_packed_dequant4(const uint8_t* base, int elem_idx, int fmt) {
    if (fmt == 0) {
        const scalar_t* s = reinterpret_cast<const scalar_t*>(
            base + elem_idx * sizeof(scalar_t));
        return make_float4(
            psp_to_float(s[0]), psp_to_float(s[1]),
            psp_to_float(s[2]), psp_to_float(s[3]));
    }
    if (fmt == 3) {
        // 4 fp4 nibbles packed into 2 bytes.
        uint16_t packed = *reinterpret_cast<const uint16_t*>(base + (elem_idx >> 1));
        return make_float4(
            dequantize_fp4_e2m1((uint8_t)(packed & 0x0Fu)),
            dequantize_fp4_e2m1((uint8_t)((packed >> 4) & 0x0Fu)),
            dequantize_fp4_e2m1((uint8_t)((packed >> 8) & 0x0Fu)),
            dequantize_fp4_e2m1((uint8_t)((packed >> 12) & 0x0Fu)));
    }
    // fp8: 4 bytes -> float4 via native hardware conversion.
    uint32_t packed = *reinterpret_cast<const uint32_t*>(base + elem_idx);
    if (fmt == 2) {
        __nv_fp8x4_e5m2 f = *reinterpret_cast<const __nv_fp8x4_e5m2*>(&packed);
        return (float4)f;
    }
    __nv_fp8x4_e4m3 f = *reinterpret_cast<const __nv_fp8x4_e4m3*>(&packed);
    return (float4)f;
}

// Lossless (FMT=0) dequant of 4 elements: vectorized via the native 2-wide
// packed types so the bf16/fp16 case is 2 loads + 2 converts instead of 4
// scalar loads + 4 converts (fp8 already uses the fp8x4 hardware path below).
template <typename scalar_t> struct psp_lossless_dequant4 {
    static __device__ float4 load(const scalar_t* s) {
        return make_float4(
            psp_to_float(s[0]), psp_to_float(s[1]),
            psp_to_float(s[2]), psp_to_float(s[3]));
    }
};
template <> struct psp_lossless_dequant4<__nv_bfloat16> {
    static __device__ float4 load(const __nv_bfloat16* s) {
        const __nv_bfloat162* p = reinterpret_cast<const __nv_bfloat162*>(s);
        float2 f0 = __bfloat1622float2(p[0]);
        float2 f1 = __bfloat1622float2(p[1]);
        return make_float4(f0.x, f0.y, f1.x, f1.y);
    }
};
template <> struct psp_lossless_dequant4<half> {
    static __device__ float4 load(const half* s) {
        const __half2* p = reinterpret_cast<const __half2*>(s);
        float2 f0 = __half22float2(p[0]);
        float2 f1 = __half22float2(p[1]);
        return make_float4(f0.x, f0.y, f1.x, f1.y);
    }
};
template <> struct psp_lossless_dequant4<float> {
    static __device__ float4 load(const float* s) {
        return *reinterpret_cast<const float4*>(s);
    }
};

// Compile-time-format vectorized dequant: FMT is a template constant, so the
// branch is eliminated at compile time (no per-element runtime dispatch).
template <typename scalar_t, int FMT>
__device__ inline float4 psp_dequant4_ct(const uint8_t* base, int elem_idx) {
    if (FMT == 0) {
        const scalar_t* s = reinterpret_cast<const scalar_t*>(
            base + elem_idx * sizeof(scalar_t));
        return psp_lossless_dequant4<scalar_t>::load(s);
    }
    if (FMT == 3) {
        uint16_t packed = *reinterpret_cast<const uint16_t*>(base + (elem_idx >> 1));
        return make_float4(
            dequantize_fp4_e2m1((uint8_t)(packed & 0x0Fu)),
            dequantize_fp4_e2m1((uint8_t)((packed >> 4) & 0x0Fu)),
            dequantize_fp4_e2m1((uint8_t)((packed >> 8) & 0x0Fu)),
            dequantize_fp4_e2m1((uint8_t)((packed >> 12) & 0x0Fu)));
    }
    uint32_t packed = *reinterpret_cast<const uint32_t*>(base + elem_idx);
    if (FMT == 2) {
        __nv_fp8x4_e5m2 f = *reinterpret_cast<const __nv_fp8x4_e5m2*>(&packed);
        return (float4)f;
    }
    __nv_fp8x4_e4m3 f = *reinterpret_cast<const __nv_fp8x4_e4m3*>(&packed);
    return (float4)f;
}

// Branchless decode of one block's token range for a compile-time format.
// The K dot product spans the full HEAD_DIM; the V weighted sum covers a
// DIM_COUNT-wide dim subset starting at dim_offset (split-H uses HEAD_DIM/2,
// non-split uses HEAD_DIM with dim_offset=0). `k_scales`/`v_scales` are the
// per-scale-group (PSP_KV_SCALE_GROUP-wide) arrays for this block+head; the
// softmax scale is already folded into k_scales by the caller.
template <typename scalar_t, int HEAD_DIM, int KFMT, int VFMT, int DIM_COUNT>
__device__ inline void psp_decode_block(
    const float* __restrict__ s_query,
    float& local_max, float& local_sum, float* __restrict__ acc,
    const uint8_t* __restrict__ k_base, const uint8_t* __restrict__ v_base,
    int tokens_in_block, const float* __restrict__ k_scales,
    const float* __restrict__ v_scales, int dim_offset, int token_offset)
{
    constexpr int NUM_GROUPS = HEAD_DIM / PSP_KV_SCALE_GROUP;
    for (int t = 0; t < tokens_in_block; ++t) {
        float s_raw = 0.0f;
        const int elem_base = (token_offset + t) * HEAD_DIM;
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; d += 4) {
            const float ks = k_scales[d / PSP_KV_SCALE_GROUP];
            float4 k4 = psp_dequant4_ct<scalar_t, KFMT>(k_base, elem_base + d);
            s_raw = fmaf(s_query[d], k4.x * ks, s_raw);
            s_raw = fmaf(s_query[d + 1], k4.y * ks, s_raw);
            s_raw = fmaf(s_query[d + 2], k4.z * ks, s_raw);
            s_raw = fmaf(s_query[d + 3], k4.w * ks, s_raw);
        }
        const float s = s_raw;  // softmax scale already folded into k_scales
        if (s > local_max) {
            const float rescale = __expf(local_max - s);
            local_sum *= rescale;
            for (int d = 0; d < DIM_COUNT; ++d) acc[d] *= rescale;
            local_max = s;
        }
        const float w = __expf(s - local_max);
        local_sum += w;
        #pragma unroll
        for (int d = 0; d < DIM_COUNT; d += 4) {
            const float vs = v_scales[(dim_offset + d) / PSP_KV_SCALE_GROUP];
            float4 v4 = psp_dequant4_ct<scalar_t, VFMT>(v_base, elem_base + dim_offset + d);
            acc[d]     = fmaf(w, v4.x * vs, acc[d]);
            acc[d + 1] = fmaf(w, v4.y * vs, acc[d + 1]);
            acc[d + 2] = fmaf(w, v4.z * vs, acc[d + 2]);
            acc[d + 3] = fmaf(w, v4.w * vs, acc[d + 3]);
        }
    }
}

template <typename scalar_t>
__device__ inline void psp_packed_quantize(
    uint8_t* base, int elem_idx, float x, int fmt, float scale)
{
    if (fmt == 0) {
        scalar_t* s = reinterpret_cast<scalar_t*>(base + elem_idx * sizeof(scalar_t));
        *s = psp_from_float<scalar_t>(x);
    } else if (fmt == 3) {
        uint8_t* b = base + (elem_idx >> 1);
        uint8_t q = quantize_fp4_e2m1(x / scale);
        if (elem_idx & 1) {
            *b = (*b & 0x0Fu) | ((q & 0x0Fu) << 4);  // high nibble
        } else {
            *b = (*b & 0xF0u) | (q & 0x0Fu);  // low nibble
        }
    } else {
        base[elem_idx] = quantize_fp8(x / scale, fmt);
    }
}

// Compile-time-format quantize of one element (no runtime fmt branch).
template <typename scalar_t, int FMT>
__device__ inline void psp_quantize_ct(uint8_t* base, int elem_idx, float x, float scale) {
    if (FMT == 0) {
        scalar_t* s = reinterpret_cast<scalar_t*>(base + elem_idx * sizeof(scalar_t));
        *s = psp_from_float<scalar_t>(x);
    } else if (FMT == 3) {
        uint8_t* b = base + (elem_idx >> 1);
        uint8_t q = quantize_fp4_e2m1(x / scale);
        if (elem_idx & 1) *b = (*b & 0x0Fu) | ((q & 0x0Fu) << 4);
        else *b = (*b & 0xF0u) | (q & 0x0Fu);
    } else {
        base[elem_idx] = (FMT == 2) ? quantize_fp8_e5m2(x / scale) : quantize_fp8_e4m3(x / scale);
    }
}

// Branchless quantize of one K/V row pair for a compile-time format.
template <typename scalar_t, int FMT>
__device__ inline void psp_quantize_row(
    const scalar_t* __restrict__ k_row, const scalar_t* __restrict__ v_row,
    uint8_t* __restrict__ k_out, uint8_t* __restrict__ v_out,
    int head_dim, int chunk_dim, int elem_base, float scale, int block_offset)
{
    for (int d = 0; d < chunk_dim; ++d) {
        const float k = psp_to_float(k_row[d]);
        const float v = psp_to_float(v_row[d]);
        psp_quantize_ct<scalar_t, FMT>(k_out, elem_base + d, k, scale);
        psp_quantize_ct<scalar_t, FMT>(v_out, block_offset * head_dim + elem_base + d, v, scale);
    }
}

// Per-K/V quantize: quantizes one row (K or V) with its own format + scale.
template <typename scalar_t, int FMT>
__device__ inline void psp_quantize_row_k(
    const scalar_t* __restrict__ row, uint8_t* __restrict__ out,
    int chunk_dim, int elem_base, float scale)
{
    for (int d = 0; d < chunk_dim; ++d) {
        psp_quantize_ct<scalar_t, FMT>(out, elem_base + d, psp_to_float(row[d]), scale);
    }
}

template <typename scalar_t, int FMT>
__device__ inline void psp_quantize_row_v(
    const scalar_t* __restrict__ row, uint8_t* __restrict__ out,
    int head_dim, int chunk_dim, int elem_base, float scale, int block_offset)
{
    for (int d = 0; d < chunk_dim; ++d) {
        psp_quantize_ct<scalar_t, FMT>(out, block_offset * head_dim + elem_base + d,
                                       psp_to_float(row[d]), scale);
    }
}

// Packed per-block precision helpers: the value >= 4 is packed
// k_fmt | (v_fmt << 2); the legacy value < 4 is a uniform K/V format.
__device__ inline int psp_kv_fmt_k(int p) { return (p >= 4) ? (p & 3) : p; }
__device__ inline int psp_kv_fmt_v(int p) { return (p >= 4) ? (p >> 2) : p; }

// Bytes per element for a block's precision (fp4 packs 2 elems per byte).
__device__ inline int psp_bytes_per_elem(int fmt, int scalar_size) {
    if (fmt == 0) return scalar_size;
    if (fmt == 3) return 1;  // two fp4 values per byte; callers use elem_idx>>1
    return 1;
}

// FP4 pair packing / unpacking
__host__ __device__ inline uint8_t pack_fp4_pair(uint8_t a, uint8_t b) {
    return (a << 4) | (b & 0x0F);
}

__host__ __device__ inline void unpack_fp4_pair(uint8_t packed, uint8_t *high, uint8_t *low) {
    *high = packed >> 4;
    *low = packed & 0x0F;
}

// E8M0 scale exponent conversion
__host__ __device__ inline float e8m0_to_f32(uint8_t exponent) {
    return powf(2.0f, (float)exponent);
}

__host__ __device__ inline uint8_t f32_to_e8m0(float scale) {
    if (isnan(scale)) return 0;
    if (isinf(scale) && scale > 0) return 255;
    if (scale <= 0.0f) return 0;
    float exp = log2f(scale);
    int e = (int)roundf(exp);
    if (e < 0) return 0;
    if (e > 255) return 255;
    return (uint8_t)e;
}

#endif
