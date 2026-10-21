// virthub/kernels/common/quantization_utils.cuh

#ifndef PSP_KV_QUANTIZATION_UTILS_CUH
#define PSP_KV_QUANTIZATION_UTILS_CUH

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <math.h>
#include <stdint.h>

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

// FP4 E2M1 (1 sign, 2 exponent, 1 mantissa, bias 1)
__host__ __device__ inline uint8_t quantize_fp4_e2m1(float value) {
    if (isnan(value)) return 0x7;
    if (value == 0.0f) return 0x0;

    uint32_t bits = float_as_uint(value);
    uint8_t sign = (bits >> 31) & 0x1;
    float abs_value = fabsf(value);

    if (isinf(abs_value)) return (sign << 3) | 0x5;  // clamp to max (3.0)

    int f32_exp = ((bits >> 23) & 0xFF) - 127;
    uint32_t f32_mant = bits & 0x7FFFFF;

    int fp4_exp = f32_exp + 1;
    uint8_t mant = f32_mant >> (23 - 1);
    uint8_t round_bit = (f32_mant >> (23 - 1 - 1)) & 1;
    uint8_t sticky = (f32_mant & ((1 << (23 - 1 - 1)) - 1)) != 0;

    if (round_bit && (sticky || (mant & 1))) {
        mant++;
        if (mant == 0x2) {
            mant = 0;
            fp4_exp++;
        }
    }

    if (fp4_exp >= 3) {
        return (sign << 3) | 0x5;  // clamp to max (3.0)
    } else if (fp4_exp <= 0) {
        if (fp4_exp < -1) {
            return sign << 3;      // underflow to zero
        }
        // subnormal
        mant = ((0x2 | mant) >> (1 - fp4_exp)) & 0x1;
        fp4_exp = 0;
    }

    return (sign << 3) | ((fp4_exp & 0x3) << 1) | mant;
}

__host__ __device__ inline float dequantize_fp4_e2m1(uint8_t q) {
    // 16-entry LUT for the e2m1 (1 sign, 2 exp, 1 mantissa) format. The
    // `static const` array lives in constant memory on device, so the lookup
    // is a single constant read instead of bit manipulation + branches.
    static const float lut[16] = {
        0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, INFINITY, INFINITY,
        -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -INFINITY, -INFINITY
    };
    return lut[q & 0x0Fu];
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
// non-split uses HEAD_DIM with dim_offset=0).
template <typename scalar_t, int HEAD_DIM, int FMT, int DIM_COUNT>
__device__ inline void psp_decode_block(
    const float* __restrict__ s_query,
    float& local_max, float& local_sum, float* __restrict__ acc,
    const uint8_t* __restrict__ k_base, const uint8_t* __restrict__ v_base,
    int tokens_in_block, float combined_scale, float v_scale_eff, int dim_offset,
    int token_offset)
{
    for (int t = 0; t < tokens_in_block; ++t) {
        float s_raw = 0.0f;
        const int elem_base = (token_offset + t) * HEAD_DIM;
        #pragma unroll
        for (int d = 0; d < HEAD_DIM; d += 4) {
            float4 k4 = psp_dequant4_ct<scalar_t, FMT>(k_base, elem_base + d);
            s_raw = fmaf(s_query[d], k4.x, s_raw);
            s_raw = fmaf(s_query[d + 1], k4.y, s_raw);
            s_raw = fmaf(s_query[d + 2], k4.z, s_raw);
            s_raw = fmaf(s_query[d + 3], k4.w, s_raw);
        }
        const float s = s_raw * combined_scale;
        if (s > local_max) {
            const float rescale = __expf(local_max - s);
            local_sum *= rescale;
            for (int d = 0; d < DIM_COUNT; ++d) acc[d] *= rescale;
            local_max = s;
        }
        const float w = __expf(s - local_max);
        local_sum += w;
        const float w_scaled = w * v_scale_eff;
        #pragma unroll
        for (int d = 0; d < DIM_COUNT; d += 4) {
            float4 v4 = psp_dequant4_ct<scalar_t, FMT>(v_base, elem_base + dim_offset + d);
            acc[d]     = fmaf(w_scaled, v4.x, acc[d]);
            acc[d + 1] = fmaf(w_scaled, v4.y, acc[d + 1]);
            acc[d + 2] = fmaf(w_scaled, v4.z, acc[d + 2]);
            acc[d + 3] = fmaf(w_scaled, v4.w, acc[d + 3]);
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
    int head_dim, float scale, int block_offset)
{
    for (int d = 0; d < head_dim; ++d) {
        const float k = psp_to_float(k_row[d]);
        const float v = psp_to_float(v_row[d]);
        psp_quantize_ct<scalar_t, FMT>(k_out, d, k, scale);
        psp_quantize_ct<scalar_t, FMT>(v_out, block_offset * head_dim + d, v, scale);
    }
}

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
