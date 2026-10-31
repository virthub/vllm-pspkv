// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// PSP-KV wrapper.
//
// This file lives in the _C extension (NOT _C_stable_libtorch). The stable
// extension is compiled with TORCH_TARGET_VERSION defined, which causes
// torch/library.h and friends to emit
//   #error "This file should not be included when either TORCH_STABLE_ONLY
//           or TORCH_TARGET_VERSION is defined."
// The regular _C extension does not set that macro, so we register the ops
// here using the ordinary TORCH_LIBRARY mechanism.
//
// We deliberately do NOT include <torch/extension.h>, because it pulls in
// <torch/python.h> (and therefore pybind11 + Python.h), and both _C and
// _C_stable_libtorch are compiled with Py_LIMITED_API=3, under which modern
// pybind11 headers fail to compile. Only <torch/library.h> is needed for the
// dispatcher registration.
//
// Once registered, the ops are callable from Python as:
//
//     torch.ops.psp_kv.psp_attention_basic(...)
//     torch.ops.psp_kv.psp_attention_enhanced(...)
//     torch.ops.psp_kv.psp_attention_btkv(...)
//     torch.ops.psp_kv.quantize_and_scatter_kv(...)
//
// TODO: replace the "not yet wired" bodies with calls to the actual kernel
// launchers in csrc/attention/psp_kv/{basic,enhanced,btkv}/*.cu once their
// launcher signatures are confirmed.

#include <torch/library.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cstdint>
#include <cstdlib>

// Per-scale-group granularity (matches PSP_KV_SCALE_GROUP in
// attention/psp_kv/common/quantization_utils.cuh; this translation unit only
// forward-declares the kernels so it needs a host copy for shape checks).
#define PSP_KV_SCALE_GROUP 16
static inline int psp_kv_num_groups(int head_dim) {
    return head_dim / PSP_KV_SCALE_GROUP;
}

// Forward declarations of the PSP-KV kernel launchers (compiled into _C).
// Templated on the query/output "target precision" (fp16 or bf16).
template <typename scalar_t>
void launch_paged_attention_basic(
    scalar_t* out, const scalar_t* q,
    const uint8_t* k_cache, const uint8_t* v_cache,
    const float* k_scales, const float* k_zero_points,
    const float* v_scales, const float* v_zero_points,
    const int* block_tables, const int* context_lens,
    const uint8_t* precision_table, const uint32_t* block_offsets,
    float* partials,
    int max_blocks_per_seq, int num_tokens, int num_seqs,
    int num_heads, int num_kv_heads, int head_dim, float scale,
    cudaStream_t stream);

template <typename scalar_t>
void launch_paged_attention_enhanced(
    scalar_t* out, const scalar_t* q,
    const uint8_t* k_cache, const uint8_t* v_cache,
    const float* k_scales, const float* v_scales,
    const int* block_tables, const int* context_lens,
    const uint8_t* precision_table, const uint32_t* block_offsets,
    float* partials,
    int max_blocks_per_seq, int num_tokens, int num_seqs,
    int num_heads, int num_kv_heads, int head_dim, float scale,
    int sliding_window, int num_blocks, cudaStream_t stream);

template <typename scalar_t>
void launch_paged_attention_enhanced_tc(
    scalar_t* out, const scalar_t* q,
    const uint8_t* k_cache, const uint8_t* v_cache,
    const int* block_tables, const int* context_lens,
    const uint32_t* block_offsets,
    float* partials,
    int max_blocks_per_seq, int num_seqs,
    int num_heads, int num_kv_heads, int head_dim, float scale,
    int sliding_window, int num_splits, int num_total_blocks, cudaStream_t stream);

void launch_paged_attention_enhanced_tf32_tc(
    float* out, const float* q,
    const uint8_t* k_cache, const uint8_t* v_cache,
    const int* block_tables, const int* context_lens,
    const uint32_t* block_offsets,
    float* partials,
    int max_blocks_per_seq, int num_seqs,
    int num_heads, int num_kv_heads, int head_dim, float scale,
    int sliding_window, int num_total_blocks, cudaStream_t stream);

template <typename scalar_t>
void launch_paged_attention_enhanced_fp8_tc(
    scalar_t* out, const scalar_t* q,
    const uint8_t* k_cache, const uint8_t* v_cache,
    const int* block_tables, const int* context_lens,
    const uint32_t* block_offsets, const float* scale_table,
    float* partials,
    int max_blocks_per_seq, int num_seqs,
    int num_heads, int num_kv_heads, int head_dim, float scale, int fp8_fmt,
    int num_total_blocks, int num_splits, cudaStream_t stream);

template <typename scalar_t>
void launch_paged_attention_enhanced_fp8_native_tc(
    scalar_t* out, const scalar_t* q,
    const uint8_t* k_cache, const uint8_t* v_cache,
    const int* block_tables, const int* context_lens,
    const uint32_t* block_offsets, const float* scale_table,
    float* partials,
    int max_blocks_per_seq, int num_seqs,
    int num_heads, int num_kv_heads, int head_dim, float scale, int fp8_fmt,
    int num_total_blocks, int num_splits, cudaStream_t stream);

template <typename scalar_t>
void launch_paged_attention_enhanced_fp8q_tc(
    scalar_t* out, const scalar_t* q,
    const uint8_t* k_cache, const uint8_t* v_cache,
    const int* block_tables, const int* context_lens,
    const uint32_t* block_offsets, const float* scale_table,
    float* partials,
    int max_blocks_per_seq, int num_seqs,
    int num_heads, int num_kv_heads, int head_dim, float scale, int fp8_fmt,
    int num_total_blocks, cudaStream_t stream);

template <typename scalar_t>
void launch_paged_attention_enhanced_fp8pv_tc(
    scalar_t* out, const scalar_t* q,
    const uint8_t* k_cache, const uint8_t* v_cache,
    const int* block_tables, const int* context_lens,
    const uint32_t* block_offsets, const float* scale_table,
    const uint8_t* precision_table,
    float* partials,
    int max_blocks_per_seq, int num_seqs,
    int num_heads, int num_kv_heads, int head_dim, float scale, int fp8_fmt,
    int k_fmt, int num_total_blocks, int num_splits, cudaStream_t stream);

template <typename scalar_t>
void launch_paged_attention_enhanced_fp8pv_mt(
    scalar_t* out, const scalar_t* q, const scalar_t* key, const scalar_t* value,
    const uint8_t* k_cache, const uint8_t* v_cache,
    const int* block_tables, const int* context_lens,
    const uint32_t* block_offsets, const float* scale_table,
    const uint8_t* precision_table,
    int max_blocks_per_seq, int num_tokens,
    int num_heads, int num_kv_heads, int head_dim, float scale, int fp8_fmt,
    int num_total_blocks, int num_query_tokens, cudaStream_t stream);

template <typename scalar_t>
void launch_paged_attention_btkv(
    scalar_t* out, const scalar_t* q,
    const uint8_t* k_cache, const uint8_t* v_cache,
    const float* k_scales, const float* v_scales,
    const int* block_tables, const int* context_lens,
    const uint8_t* precision_table, const uint32_t* block_offsets,
    float* partials,
    int max_blocks_per_seq, int num_tokens, int num_seqs,
    int num_heads, int num_kv_heads, int head_dim, float scale,
    cudaStream_t stream);

template <typename scalar_t>
void launch_quantize_and_scatter(
    const scalar_t* prefill_scratch,
    uint8_t* key_cache, uint8_t* value_cache,
    const int64_t* slot_mapping,
    const uint8_t* precision_table, const uint32_t* block_offsets,
    float* scale_table, float* max_abs,
    int num_tokens, int num_blocks, int num_kv_heads, int block_size, int head_dim,
    int format_generation, cudaStream_t stream);

template <typename scalar_t>
void launch_quantize_scatter_multi(
    const scalar_t* prefill_scratch,
    uint8_t* key_cache, uint8_t* value_cache,
    const int64_t* slot_mapping,
    const uint8_t* precision_table, const uint32_t* block_offsets,
    const float* scale_table,
    int num_tokens, int num_blocks, int num_kv_heads, int block_size, int head_dim,
    int format_generation, cudaStream_t stream);

template <typename scalar_t>
void launch_derive_precision(
    const scalar_t* prefill_scratch,
    const int64_t* slot_mapping,
    uint8_t* precision_table,
    float* max_abs,
    int num_tokens, int num_blocks, int num_kv_heads, int block_size, int head_dim,
    int k_default, int v_default, float E, cudaStream_t stream);

template <typename scalar_t>
void launch_derive_precision_decode(
    const scalar_t* key, const scalar_t* value,
    const int64_t* slot_mapping,
    uint8_t* precision_table, float* scale_table, float* max_abs,
    int num_tokens, int num_blocks, int num_kv_heads, int block_size, int head_dim,
    int k_default, int v_default, float E, cudaStream_t stream);

template <typename scalar_t>
void launch_quantize_single_token(
    const scalar_t* key, const scalar_t* value,
    uint8_t* key_cache, uint8_t* value_cache,
    const int64_t* slot_mapping,
    const uint8_t* precision_table, const uint32_t* block_offsets,
    const float* scale_table,
    int num_tokens, int num_kv_heads, int block_size, int head_dim, int num_blocks,
    cudaStream_t stream);

template <typename scalar_t>
void launch_dequant_kv(
    const uint8_t* k_cache, const uint8_t* v_cache,
    const uint32_t* block_offsets, const uint8_t* precision_table,
    const float* scale_table,
    scalar_t* k_out, scalar_t* v_out,
    int num_blocks, int num_kv_heads, int block_size, int head_dim,
    cudaStream_t stream);

template <typename scalar_t>
void launch_dequant_prefix(
    const uint8_t* k_cache, const uint8_t* v_cache,
    const int* block_table, const uint8_t* precision_table,
    const uint32_t* block_offsets, const float* scale_table,
    scalar_t* k_out, scalar_t* v_out,
    int n_blk, int num_blocks, int num_kv_heads, int block_size, int head_dim,
    cudaStream_t stream);

template <typename scalar_t>
void launch_dequant_prefix_batch(
    const uint8_t* k_cache, const uint8_t* v_cache,
    const int* block_tables, const int* ctx_lens, const int* out_offsets,
    const uint8_t* precision_table, const uint32_t* block_offsets,
    const float* scale_table,
    scalar_t* k_out, scalar_t* v_out,
    int num_seqs, int max_blocks_per_seq, int num_blocks,
    int num_kv_heads, int block_size, int head_dim,
    cudaStream_t stream);

template <typename scalar_t>
void launch_paged_attention_uniform(
    scalar_t* out, const scalar_t* q,
    const scalar_t* k_cache, const scalar_t* v_cache,
    const int* block_tables, const int* context_lens,
    int max_blocks_per_seq, int num_tokens, int num_seqs,
    int num_heads, int num_kv_heads, int head_dim, float scale,
    cudaStream_t stream);

namespace psp_kv {

// Dispatch a launch call for the given query dtype (fp16 / bf16 / fp8).
#define PSP_KV_DISPATCH_SCALAR(query, scalar_body)                  \
    do {                                                            \
        if ((query).scalar_type() == at::kHalf) {                   \
            using scalar_t = half;                                  \
            scalar_body                                              \
        } else if ((query).scalar_type() == at::kBFloat16) {        \
            using scalar_t = __nv_bfloat16;                         \
            scalar_body                                              \
        } else if ((query).scalar_type() == at::kFloat) {           \
            using scalar_t = float;                                 \
            scalar_body                                              \
        } else if ((query).scalar_type() == at::kFloat8_e4m3fn) {   \
            using scalar_t = __nv_fp8_e4m3;                         \
            scalar_body                                              \
        } else {                                                    \
            using scalar_t = __nv_fp8_e5m2;                         \
            scalar_body                                              \
        }                                                           \
    } while (0)

// Dispatch a tensor-core launch for the half/bf16 query dtypes (the WMMA mma
// m16n8k16 tiles only support fp16 and bf16, not fp32/fp8).
#define PSP_KV_DISPATCH_HALF_BF16(query, scalar_body)           \
    do {                                                        \
        if ((query).scalar_type() == at::kHalf) {               \
            using scalar_t = half;                              \
            scalar_body                                          \
        } else {                                                \
            using scalar_t = __nv_bfloat16;                     \
            scalar_body                                          \
        }                                                       \
    } while (0)

// ----------------------------------------------------------------------------
// Basic Format
// ----------------------------------------------------------------------------
void psp_attention_basic(
    at::Tensor& query,           // [num_tokens, num_heads, head_dim]
    at::Tensor& key_cache,       // [num_blocks, num_kv_heads, block_size, head_dim]
    at::Tensor& value_cache,     // [num_blocks, num_kv_heads, block_size, head_dim]
    at::Tensor& block_tables,    // [num_seqs, max_blocks_per_seq] int32
    at::Tensor& context_lens,    // [num_seqs] int32
    at::Tensor& precision_table, // [num_blocks] uint8
    at::Tensor& block_offsets,   // [num_blocks] int32 byte offsets
    double softmax_scale,
    int64_t num_kv_heads,
    int64_t head_size,
    at::Tensor& out)             // [num_tokens, num_heads, head_dim]
{
    TORCH_CHECK(query.is_contiguous(), "query must be contiguous");
    TORCH_CHECK(key_cache.is_contiguous(), "key_cache must be contiguous");
    TORCH_CHECK(value_cache.is_contiguous(), "value_cache must be contiguous");
    TORCH_CHECK(block_tables.is_contiguous(), "block_tables must be contiguous");
    TORCH_CHECK(context_lens.is_contiguous(), "context_lens must be contiguous");
    TORCH_CHECK(precision_table.is_contiguous(), "precision_table must be contiguous");
    TORCH_CHECK(block_offsets.is_contiguous(), "block_offsets must be contiguous");
    TORCH_CHECK(out.is_contiguous(), "out must be contiguous");

    TORCH_CHECK(block_tables.scalar_type() == at::kInt, "block_tables must be int32");
    TORCH_CHECK(context_lens.scalar_type() == at::kInt, "context_lens must be int32");
    TORCH_CHECK(precision_table.scalar_type() == at::kByte, "precision_table must be uint8");
    TORCH_CHECK(block_offsets.scalar_type() == at::kInt, "block_offsets must be int32");
    TORCH_CHECK(key_cache.scalar_type() == at::kByte, "key_cache must be uint8 (byte buffer)");
    TORCH_CHECK(value_cache.scalar_type() == at::kByte, "value_cache must be uint8 (byte buffer)");
    TORCH_CHECK(query.scalar_type() == out.scalar_type(), "query and out dtype mismatch");

    const int num_tokens = static_cast<int>(query.size(0));
    const int num_heads = static_cast<int>(query.size(1));
    const int num_blocks = static_cast<int>(precision_table.size(0));
    const int kv_heads = static_cast<int>(num_kv_heads);
    const int block_size = 16;  // PSP-KV block size
    const int head_dim = static_cast<int>(head_size);
    const int num_seqs = static_cast<int>(block_tables.size(0));
    const int max_blocks_per_seq = static_cast<int>(block_tables.size(1));

    TORCH_CHECK(context_lens.size(0) == num_seqs, "context_lens count mismatch with num_seqs");
    TORCH_CHECK(num_tokens == num_seqs, "Decode attention requires num_tokens == num_seqs");
    TORCH_CHECK(kv_heads == num_kv_heads, "key_cache num_kv_heads mismatch");
    TORCH_CHECK(head_dim == head_size, "head size mismatch");
    TORCH_CHECK(query.size(2) == head_size, "query head_dim mismatch");
    TORCH_CHECK(out.size(0) == num_tokens && out.size(1) == num_heads &&
                out.size(2) == head_size, "output shape mismatch");

    TORCH_CHECK(query.scalar_type() == at::kHalf || query.scalar_type() == at::kBFloat16 || query.scalar_type() == at::kFloat || query.scalar_type() == at::kFloat8_e4m3fn || query.scalar_type() == at::kFloat8_e5m2,
                "psp_attention_basic supports fp16/bf16/fp8 target precision only");

    // Basic has no out-of-band scale_table (in-band metadata); use unit scale
    // + zero zero-point so the reference kernel dequantizes x = 1.0 * (q - 0).
    auto opts = at::TensorOptions().dtype(at::kFloat).device(key_cache.device());
    auto k_scales = at::ones({(int64_t)num_blocks, (int64_t)kv_heads, (int64_t)psp_kv_num_groups(head_dim)}, opts);
    auto v_scales = at::ones({(int64_t)num_blocks, (int64_t)kv_heads, (int64_t)psp_kv_num_groups(head_dim)}, opts);
    auto k_zero_points = at::zeros({(int64_t)num_blocks, (int64_t)kv_heads}, opts);
    auto v_zero_points = at::zeros({(int64_t)num_blocks, (int64_t)kv_heads}, opts);

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    auto partials = at::empty(
        {(int64_t)num_seqs * num_heads * 2 * 2 * (2 + head_dim / 2)},
        at::TensorOptions().dtype(at::kFloat).device(key_cache.device()));
    PSP_KV_DISPATCH_SCALAR(query, {
        launch_paged_attention_basic<scalar_t>(
            reinterpret_cast<scalar_t*>(out.data_ptr()),
            reinterpret_cast<const scalar_t*>(query.data_ptr()),
            key_cache.data_ptr<uint8_t>(),
            value_cache.data_ptr<uint8_t>(),
            k_scales.data_ptr<float>(),
            k_zero_points.data_ptr<float>(),
            v_scales.data_ptr<float>(),
            v_zero_points.data_ptr<float>(),
            block_tables.data_ptr<int>(),
            context_lens.data_ptr<int>(),
            precision_table.data_ptr<uint8_t>(),
            reinterpret_cast<const uint32_t*>(block_offsets.data_ptr<int>()),
            partials.data_ptr<float>(),
            max_blocks_per_seq, num_tokens, num_seqs, num_heads, kv_heads, head_dim,
            static_cast<float>(softmax_scale), stream);
    });
}

// ----------------------------------------------------------------------------
// Enhanced Format
// ----------------------------------------------------------------------------
void psp_attention_enhanced(
    at::Tensor& query,
    at::Tensor& key_cache,
    at::Tensor& value_cache,
    at::Tensor& block_tables,
    at::Tensor& context_lens,
    at::Tensor& precision_table,
    at::Tensor& block_offsets,   // [num_blocks] int32 byte offsets
    at::Tensor& scale_table,     // [num_blocks, num_kv_heads] float32
    double softmax_scale,
    int64_t num_kv_heads,
    int64_t head_size,
    int64_t sliding_window,
    bool is_lossless,
    int64_t fp8_fmt,
    int64_t v_fmt,
    int64_t k_fmt,
    bool k_lossless,
    bool v_lossless,
    at::Tensor& out)
{
    TORCH_CHECK(query.is_contiguous(), "query must be contiguous");
    TORCH_CHECK(key_cache.is_contiguous(), "key_cache must be contiguous");
    TORCH_CHECK(value_cache.is_contiguous(), "value_cache must be contiguous");
    TORCH_CHECK(block_tables.is_contiguous(), "block_tables must be contiguous");
    TORCH_CHECK(context_lens.is_contiguous(), "context_lens must be contiguous");
    TORCH_CHECK(precision_table.is_contiguous(), "precision_table must be contiguous");
    TORCH_CHECK(block_offsets.is_contiguous(), "block_offsets must be contiguous");
    TORCH_CHECK(scale_table.is_contiguous(), "scale_table must be contiguous");
    TORCH_CHECK(out.is_contiguous(), "out must be contiguous");

    TORCH_CHECK(block_tables.scalar_type() == at::kInt, "block_tables must be int32");
    TORCH_CHECK(context_lens.scalar_type() == at::kInt, "context_lens must be int32");
    TORCH_CHECK(precision_table.scalar_type() == at::kByte, "precision_table must be uint8");
    TORCH_CHECK(block_offsets.scalar_type() == at::kInt, "block_offsets must be int32");
    TORCH_CHECK(scale_table.scalar_type() == at::kFloat, "scale_table must be float32");
    TORCH_CHECK(key_cache.scalar_type() == at::kByte, "key_cache must be uint8 (byte buffer)");
    TORCH_CHECK(value_cache.scalar_type() == at::kByte, "value_cache must be uint8 (byte buffer)");
    TORCH_CHECK(query.scalar_type() == out.scalar_type(), "query and out dtype mismatch");

    const int num_tokens = static_cast<int>(query.size(0));
    const int num_heads = static_cast<int>(query.size(1));
    const int num_blocks = static_cast<int>(precision_table.size(0));
    const int kv_heads = static_cast<int>(num_kv_heads);
    const int block_size = 16;  // PSP-KV block size
    const int head_dim = static_cast<int>(head_size);
    const int num_seqs = static_cast<int>(block_tables.size(0));
    const int max_blocks_per_seq = static_cast<int>(block_tables.size(1));

    TORCH_CHECK(context_lens.size(0) == num_seqs, "context_lens count mismatch with num_seqs");
    TORCH_CHECK(num_tokens == num_seqs, "Decode attention requires num_tokens == num_seqs");
    TORCH_CHECK(kv_heads == num_kv_heads, "key_cache num_kv_heads mismatch");
    TORCH_CHECK(head_dim == head_size, "head size mismatch");
    TORCH_CHECK(scale_table.size(0) == num_blocks
                || scale_table.size(0) == 2 * num_blocks,
                "scale_table num_blocks mismatch");
    TORCH_CHECK(scale_table.size(1) == kv_heads, "scale_table num_kv_heads mismatch");
    TORCH_CHECK(scale_table.dim() == 2
                || scale_table.size(2) == psp_kv_num_groups(head_dim),
                "scale_table num_groups mismatch");

    TORCH_CHECK(query.scalar_type() == at::kHalf || query.scalar_type() == at::kBFloat16 || query.scalar_type() == at::kFloat || query.scalar_type() == at::kFloat8_e4m3fn || query.scalar_type() == at::kFloat8_e5m2,
                "psp_attention_enhanced supports fp16/bf16/fp8 target precision only");
    TORCH_CHECK(scale_table.scalar_type() == at::kFloat,
                "scale_table must be float32");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    // Split-K count (PSPKV_TC_NUM_SPLITS, default 8) is runtime-tunable per
    // model size: larger models saturate with fewer splits, smaller models
    // want more parallelism per head.
    const int tc_num_splits = getenv("PSPKV_TC_NUM_SPLITS")
        ? atoi(getenv("PSPKV_TC_NUM_SPLITS")) : 16;

    // Tensor-core path: bf16 lossless (all blocks FMT=0). The kernel applies
    // the sliding-window mask itself, so the window no longer forces a fallback.
    // `is_lossless`/`fp8_fmt` are precomputed CPU-side by the model runner to
    // avoid a GPU->CPU sync per decode step.
    if ((query.scalar_type() == at::kHalf || query.scalar_type() == at::kBFloat16)
        && is_lossless && head_dim <= 256) {
        auto tc_partials = at::empty(
            {(int64_t)num_seqs * kv_heads * tc_num_splits * (num_heads / kv_heads) * (2 + head_dim)},
            at::TensorOptions().dtype(at::kFloat).device(key_cache.device()));
        PSP_KV_DISPATCH_HALF_BF16(query, {
            launch_paged_attention_enhanced_tc<scalar_t>(
                reinterpret_cast<scalar_t*>(out.data_ptr()),
                reinterpret_cast<const scalar_t*>(query.data_ptr()),
                key_cache.data_ptr<uint8_t>(),
                value_cache.data_ptr<uint8_t>(),
                block_tables.data_ptr<int>(),
                context_lens.data_ptr<int>(),
                reinterpret_cast<const uint32_t*>(block_offsets.data_ptr<int>()),
                tc_partials.data_ptr<float>(),
                max_blocks_per_seq, num_seqs, num_heads, kv_heads, head_dim,
                static_cast<float>(softmax_scale), static_cast<int>(sliding_window),
                tc_num_splits, num_blocks, stream);
        });
        return;
    }

    // Tensor-core path: uniform fp8 inner precision (all blocks e4m3 or e5m2).
    // PSPKV_FP8_QUERY=1 selects the fp8-query kernel (query -> e4m3 for the QK
    // mma); it trades ~4.6% accuracy for the faster path.
    static const bool fp8_query = (getenv("PSPKV_FP8_QUERY") != nullptr);
    if ((query.scalar_type() == at::kHalf || query.scalar_type() == at::kBFloat16)
        && sliding_window <= 0 && fp8_fmt != 0 && v_fmt == fp8_fmt
        && head_dim <= 256 && fp8_query) {
        auto tc_partials = at::empty(
            {(int64_t)num_seqs * kv_heads * 8 * (num_heads / kv_heads) * (2 + head_dim)},
            at::TensorOptions().dtype(at::kFloat).device(key_cache.device()));
        PSP_KV_DISPATCH_HALF_BF16(query, {
            launch_paged_attention_enhanced_fp8q_tc<scalar_t>(
                reinterpret_cast<scalar_t*>(out.data_ptr()),
                reinterpret_cast<const scalar_t*>(query.data_ptr()),
                key_cache.data_ptr<uint8_t>(),
                value_cache.data_ptr<uint8_t>(),
                block_tables.data_ptr<int>(),
                context_lens.data_ptr<int>(),
                reinterpret_cast<const uint32_t*>(block_offsets.data_ptr<int>()),
                scale_table.data_ptr<float>(),
                tc_partials.data_ptr<float>(),
                max_blocks_per_seq, num_seqs, num_heads, kv_heads, head_dim,
                static_cast<float>(softmax_scale), fp8_fmt, num_blocks, stream);
        });
        return;
    }
    // Native fp8 MMA path (opt-in): uniform fp8 K/V fed to the m16n8k32 tensor
    // cores directly via ldmatrix, dropping the fp8->fp16 dequant on K.
    // Gated behind PSPKV_NATIVE_FP8_MMA because Q is quantized to fp8 and the
    // kernel needs the sm_120a build (ldmatrix.m16n16.b8).
    static const bool native_fp8 = (getenv("PSPKV_NATIVE_FP8_MMA") != nullptr);
    if (native_fp8
        && (query.scalar_type() == at::kHalf || query.scalar_type() == at::kBFloat16)
        && sliding_window <= 0 && fp8_fmt != 0 && v_fmt == fp8_fmt
        && head_dim % 32 == 0 && head_dim <= 256) {
        auto tc_partials = at::empty(
            {(int64_t)num_seqs * kv_heads * tc_num_splits * (num_heads / kv_heads) * (2 + head_dim)},
            at::TensorOptions().dtype(at::kFloat).device(key_cache.device()));
        PSP_KV_DISPATCH_HALF_BF16(query, {
            launch_paged_attention_enhanced_fp8_native_tc<scalar_t>(
                reinterpret_cast<scalar_t*>(out.data_ptr()),
                reinterpret_cast<const scalar_t*>(query.data_ptr()),
                key_cache.data_ptr<uint8_t>(),
                value_cache.data_ptr<uint8_t>(),
                block_tables.data_ptr<int>(),
                context_lens.data_ptr<int>(),
                reinterpret_cast<const uint32_t*>(block_offsets.data_ptr<int>()),
                scale_table.data_ptr<float>(),
                tc_partials.data_ptr<float>(),
                max_blocks_per_seq, num_seqs, num_heads, kv_heads, head_dim,
                static_cast<float>(softmax_scale), fp8_fmt, num_blocks,
                tc_num_splits, stream);
        });
        return;
    }
    if ((query.scalar_type() == at::kHalf || query.scalar_type() == at::kBFloat16)
        && sliding_window <= 0 && fp8_fmt != 0 && v_fmt == fp8_fmt
        && head_dim <= 256) {
        auto tc_partials = at::empty(
            {(int64_t)num_seqs * kv_heads * tc_num_splits * (num_heads / kv_heads) * (2 + head_dim)},
            at::TensorOptions().dtype(at::kFloat).device(key_cache.device()));
        PSP_KV_DISPATCH_HALF_BF16(query, {
            launch_paged_attention_enhanced_fp8_tc<scalar_t>(
                reinterpret_cast<scalar_t*>(out.data_ptr()),
                reinterpret_cast<const scalar_t*>(query.data_ptr()),
                key_cache.data_ptr<uint8_t>(),
                value_cache.data_ptr<uint8_t>(),
                block_tables.data_ptr<int>(),
                context_lens.data_ptr<int>(),
                reinterpret_cast<const uint32_t*>(block_offsets.data_ptr<int>()),
                scale_table.data_ptr<float>(),
                tc_partials.data_ptr<float>(),
                max_blocks_per_seq, num_seqs, num_heads, kv_heads, head_dim,
                static_cast<float>(softmax_scale), fp8_fmt, num_blocks,
                tc_num_splits, stream);
        });
        return;
    }

    // Per-K/V tensor-core path: per-block fp8/fp4 K + per-block fp8/fp4 V
    // (packed precision_table k_fmt|(v_fmt<<2), the adaptive per-K/V default).
    // Both formats are read per block so the dynamic precision prediction can
    // emit a mixed fp8/fp4 table and still stay on the tensor-core path. The
    // dequant only pays off when enough query heads share one KV head
    // (GQA >= 4); below that the scalar path's per-head parallelism wins.
    if ((query.scalar_type() == at::kHalf || query.scalar_type() == at::kBFloat16)
        && sliding_window <= 0 && head_dim <= 256
        && !k_lossless && !v_lossless
        && k_fmt != 3 && v_fmt != 3  // fp4 needs per-group scales: scalar path
        && (num_heads / kv_heads) >= 4) {
        auto tc_partials = at::empty(
            {(int64_t)num_seqs * kv_heads * tc_num_splits * (num_heads / kv_heads) * (2 + head_dim)},
            at::TensorOptions().dtype(at::kFloat).device(key_cache.device()));
        PSP_KV_DISPATCH_HALF_BF16(query, {
            launch_paged_attention_enhanced_fp8pv_tc<scalar_t>(
                reinterpret_cast<scalar_t*>(out.data_ptr()),
                reinterpret_cast<const scalar_t*>(query.data_ptr()),
                key_cache.data_ptr<uint8_t>(),
                value_cache.data_ptr<uint8_t>(),
                block_tables.data_ptr<int>(),
                context_lens.data_ptr<int>(),
                reinterpret_cast<const uint32_t*>(block_offsets.data_ptr<int>()),
                scale_table.data_ptr<float>(),
                precision_table.data_ptr<uint8_t>(),
                tc_partials.data_ptr<float>(),
                max_blocks_per_seq, num_seqs, num_heads, kv_heads, head_dim,
                static_cast<float>(softmax_scale), static_cast<int>(fp8_fmt),
                static_cast<int>(k_fmt), num_blocks, tc_num_splits, stream);
        });
        return;
    }

    // fp32 target + lossless cache -> tf32 tensor-core decode (m16n16k8).
    if (query.scalar_type() == at::kFloat
        && sliding_window <= 0 && fp8_fmt == 0 && head_dim <= 128) {
        auto tc_partials = at::empty(
            {(int64_t)num_seqs * kv_heads * 8 * (num_heads / kv_heads) * (2 + head_dim)},
            at::TensorOptions().dtype(at::kFloat).device(key_cache.device()));
        launch_paged_attention_enhanced_tf32_tc(
            reinterpret_cast<float*>(out.data_ptr()),
            reinterpret_cast<const float*>(query.data_ptr()),
            key_cache.data_ptr<uint8_t>(),
            value_cache.data_ptr<uint8_t>(),
            block_tables.data_ptr<int>(),
            context_lens.data_ptr<int>(),
            reinterpret_cast<const uint32_t*>(block_offsets.data_ptr<int>()),
            tc_partials.data_ptr<float>(),
            max_blocks_per_seq, num_seqs, num_heads, kv_heads, head_dim,
            static_cast<float>(softmax_scale), static_cast<int>(sliding_window), num_blocks, stream);
        return;
    }

    // Split-K x split-H scratch: [num_seqs*num_heads*2*2, 2+head_dim/2] partial.
    auto partials = at::empty(
        {(int64_t)num_seqs * num_heads * 2 * 2 * (2 + head_dim / 2)},
        at::TensorOptions().dtype(at::kFloat).device(key_cache.device()));
    PSP_KV_DISPATCH_SCALAR(query, {
        launch_paged_attention_enhanced<scalar_t>(
            reinterpret_cast<scalar_t*>(out.data_ptr()),
            reinterpret_cast<const scalar_t*>(query.data_ptr()),
            key_cache.data_ptr<uint8_t>(),
            value_cache.data_ptr<uint8_t>(),
            scale_table.data_ptr<float>(),
            scale_table.data_ptr<float>() + (int64_t)num_blocks * kv_heads * psp_kv_num_groups(head_dim),
            block_tables.data_ptr<int>(),
            context_lens.data_ptr<int>(),
            precision_table.data_ptr<uint8_t>(),
            reinterpret_cast<const uint32_t*>(block_offsets.data_ptr<int>()),
            partials.data_ptr<float>(),
            max_blocks_per_seq, num_tokens, num_seqs, num_heads, kv_heads, head_dim,
            static_cast<float>(softmax_scale), static_cast<int>(sliding_window),
            num_blocks, stream);
    });
}

// ----------------------------------------------------------------------------
// BT-KV Format
// ----------------------------------------------------------------------------
void psp_attention_btkv(
    at::Tensor& query,
    at::Tensor& key_cache,
    at::Tensor& value_cache,
    at::Tensor& block_tables,
    at::Tensor& context_lens,
    at::Tensor& precision_table,
    at::Tensor& block_offsets,   // [num_blocks] int32 byte offsets
    at::Tensor& scale_table,
    at::Tensor& tma_descriptors, // [num_blocks, 16] uint64
    double softmax_scale,
    int64_t num_kv_heads,
    int64_t head_size,
    at::Tensor& out)
{
    TORCH_CHECK(query.is_contiguous(), "query must be contiguous");
    TORCH_CHECK(key_cache.is_contiguous(), "key_cache must be contiguous");
    TORCH_CHECK(value_cache.is_contiguous(), "value_cache must be contiguous");
    TORCH_CHECK(block_tables.is_contiguous(), "block_tables must be contiguous");
    TORCH_CHECK(context_lens.is_contiguous(), "context_lens must be contiguous");
    TORCH_CHECK(precision_table.is_contiguous(), "precision_table must be contiguous");
    TORCH_CHECK(block_offsets.is_contiguous(), "block_offsets must be contiguous");
    TORCH_CHECK(scale_table.is_contiguous(), "scale_table must be contiguous");
    TORCH_CHECK(tma_descriptors.is_contiguous(), "tma_descriptors must be contiguous");
    TORCH_CHECK(out.is_contiguous(), "out must be contiguous");

    TORCH_CHECK(block_tables.scalar_type() == at::kInt, "block_tables must be int32");
    TORCH_CHECK(context_lens.scalar_type() == at::kInt, "context_lens must be int32");
    TORCH_CHECK(precision_table.scalar_type() == at::kByte, "precision_table must be uint8");
    TORCH_CHECK(block_offsets.scalar_type() == at::kInt, "block_offsets must be int32");
    TORCH_CHECK(scale_table.scalar_type() == at::kFloat, "scale_table must be float32");
    TORCH_CHECK(tma_descriptors.scalar_type() == at::kLong, "tma_descriptors must be int64/uint64");
    TORCH_CHECK(key_cache.scalar_type() == at::kByte, "key_cache must be uint8 (byte buffer)");
    TORCH_CHECK(value_cache.scalar_type() == at::kByte, "value_cache must be uint8 (byte buffer)");
    TORCH_CHECK(query.scalar_type() == out.scalar_type(), "query and out dtype mismatch");

    // TMA descriptor alignment checks.
    const int64_t desc_stride = tma_descriptors.stride(0);
    TORCH_CHECK(desc_stride * sizeof(uint64_t) % 64 == 0,
                "Per-block TMA descriptor stride must be a multiple of 64 bytes");
    uintptr_t desc_base = reinterpret_cast<uintptr_t>(tma_descriptors.data_ptr());
    TORCH_CHECK(desc_base % 64 == 0,
                "Base address of TMA descriptors must be 64-byte aligned");

    const int num_tokens = static_cast<int>(query.size(0));
    const int num_heads = static_cast<int>(query.size(1));
    const int num_blocks = static_cast<int>(precision_table.size(0));
    const int kv_heads = static_cast<int>(num_kv_heads);
    const int block_size = 16;  // PSP-KV block size
    const int head_dim = static_cast<int>(head_size);
    const int num_seqs = static_cast<int>(block_tables.size(0));
    const int max_blocks_per_seq = static_cast<int>(block_tables.size(1));

    TORCH_CHECK(context_lens.size(0) == num_seqs, "context_lens count mismatch with num_seqs");
    TORCH_CHECK(num_tokens == num_seqs, "Decode attention requires num_tokens == num_seqs");
    TORCH_CHECK(kv_heads == num_kv_heads, "key_cache num_kv_heads mismatch");
    TORCH_CHECK(head_dim == head_size, "head size mismatch");

    TORCH_CHECK(query.scalar_type() == at::kHalf || query.scalar_type() == at::kBFloat16 || query.scalar_type() == at::kFloat || query.scalar_type() == at::kFloat8_e4m3fn || query.scalar_type() == at::kFloat8_e5m2,
                "psp_attention_btkv supports fp16/bf16/fp8 target precision only");

    // The reference BT-KV kernel does not consume the TMA descriptors; they are
    // only validated here. scale_table is passed through as float32.
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    auto partials = at::empty(
        {(int64_t)num_seqs * num_heads * 2 * 2 * (2 + head_dim / 2)},
        at::TensorOptions().dtype(at::kFloat).device(key_cache.device()));
    PSP_KV_DISPATCH_SCALAR(query, {
        launch_paged_attention_btkv<scalar_t>(
            reinterpret_cast<scalar_t*>(out.data_ptr()),
            reinterpret_cast<const scalar_t*>(query.data_ptr()),
            key_cache.data_ptr<uint8_t>(),
            value_cache.data_ptr<uint8_t>(),
            scale_table.data_ptr<float>(),
            scale_table.data_ptr<float>(),
            block_tables.data_ptr<int>(),
            context_lens.data_ptr<int>(),
            precision_table.data_ptr<uint8_t>(),
            reinterpret_cast<const uint32_t*>(block_offsets.data_ptr<int>()),
            partials.data_ptr<float>(),
            max_blocks_per_seq, num_tokens, num_seqs, num_heads, kv_heads, head_dim,
            static_cast<float>(softmax_scale), stream);
    });
}

// ----------------------------------------------------------------------------
// Prefill quantize-and-scatter helper
// ----------------------------------------------------------------------------
void quantize_and_scatter_kv(
    at::Tensor& prefill_scratch, // [num_tokens, 2, num_kv_heads, head_dim] FP16/BF16
    at::Tensor& key_cache,       // [num_blocks, num_kv_heads, block_size, head_dim] FP8
    at::Tensor& value_cache,     // [num_blocks, num_kv_heads, block_size, head_dim] FP8
    at::Tensor& block_tables,    // [num_seqs, max_blocks_per_seq] int32
    at::Tensor& slot_mapping,    // [num_tokens] int64
    at::Tensor& precision_table, // [num_blocks] uint8 (1=e4m3, 2=e5m2)
    at::Tensor& block_offsets,   // [num_blocks] int32 byte offsets
    at::Tensor& scale_table,     // [num_blocks, num_kv_heads] float32
    int64_t num_tokens,
    int64_t format_generation)
{
    TORCH_CHECK(num_tokens > 0, "num_tokens must be positive");
    TORCH_CHECK(num_tokens <= prefill_scratch.size(0),
                "num_tokens exceeds prefill scratch capacity");
    TORCH_CHECK(prefill_scratch.is_contiguous(), "prefill_scratch must be contiguous");
    TORCH_CHECK(key_cache.is_contiguous(), "key_cache must be contiguous");
    TORCH_CHECK(value_cache.is_contiguous(), "value_cache must be contiguous");
    TORCH_CHECK(block_tables.is_contiguous(), "block_tables must be contiguous");
    TORCH_CHECK(slot_mapping.is_contiguous(), "slot_mapping must be contiguous");
    TORCH_CHECK(precision_table.is_contiguous(), "precision_table must be contiguous");
    TORCH_CHECK(block_offsets.is_contiguous(), "block_offsets must be contiguous");
    TORCH_CHECK(scale_table.is_contiguous(), "scale_table must be contiguous");

    TORCH_CHECK(prefill_scratch.scalar_type() == at::kHalf || prefill_scratch.scalar_type() == at::kBFloat16 || prefill_scratch.scalar_type() == at::kFloat || prefill_scratch.scalar_type() == at::kFloat8_e4m3fn || prefill_scratch.scalar_type() == at::kFloat8_e5m2,
                "quantize_and_scatter_kv supports fp16/bf16/fp32/fp8 prefill scratch only");
    TORCH_CHECK(block_tables.scalar_type() == at::kInt, "block_tables must be int32");
    TORCH_CHECK(slot_mapping.scalar_type() == at::kLong, "slot_mapping must be int64");
    TORCH_CHECK(precision_table.scalar_type() == at::kByte, "precision_table must be uint8");
    TORCH_CHECK(block_offsets.scalar_type() == at::kInt, "block_offsets must be int32");
    TORCH_CHECK(scale_table.scalar_type() == at::kFloat, "scale_table must be float32");
    TORCH_CHECK(key_cache.scalar_type() == at::kByte, "key_cache must be uint8 (byte buffer)");
    TORCH_CHECK(value_cache.scalar_type() == at::kByte, "value_cache must be uint8 (byte buffer)");

    const int num_kv_heads = static_cast<int>(prefill_scratch.size(2));
    const int block_size = 16;  // PSP-KV block size
    const int head_dim = static_cast<int>(prefill_scratch.size(3));
    const int num_blocks = static_cast<int>(precision_table.size(0));

    // Temporary per-(block, head) max-abs workspace for scale estimation,
    // laid out as [K | V] (2 * num_blocks rows).
    auto max_abs = at::zeros({2 * (int64_t)num_blocks * num_kv_heads * psp_kv_num_groups(head_dim)}, scale_table.options());

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    PSP_KV_DISPATCH_SCALAR(prefill_scratch, {
        launch_quantize_and_scatter<scalar_t>(
            reinterpret_cast<const scalar_t*>(prefill_scratch.data_ptr()),
            key_cache.data_ptr<uint8_t>(),
            value_cache.data_ptr<uint8_t>(),
            slot_mapping.data_ptr<int64_t>(),
            precision_table.data_ptr<uint8_t>(),
            reinterpret_cast<const uint32_t*>(block_offsets.data_ptr<int>()),
            scale_table.data_ptr<float>(),
            max_abs.data_ptr<float>(),
            static_cast<int>(num_tokens), num_blocks, num_kv_heads, block_size, head_dim,
            static_cast<int>(format_generation), stream);
    });
}

// Scatter-only write-back for decode-like multi-token steps (speculative
// verify). Quantizes the scratch with the existing precision/scale (on-the-fly
// scale for fresh blocks) and scatters; no estimate/finalize pass.
void quantize_scatter_multi(
    at::Tensor& prefill_scratch, // [num_tokens, 2, num_kv_heads, head_dim]
    at::Tensor& key_cache,
    at::Tensor& value_cache,
    at::Tensor& slot_mapping,    // [num_tokens] int64
    at::Tensor& precision_table, // [num_blocks] uint8
    at::Tensor& block_offsets,   // [2 * num_blocks] int32
    at::Tensor& scale_table,     // [2 * num_blocks, num_kv_heads] float32
    int64_t num_tokens,
    int64_t format_generation)
{
    TORCH_CHECK(num_tokens > 0, "num_tokens must be positive");
    TORCH_CHECK(num_tokens <= prefill_scratch.size(0),
                "num_tokens exceeds prefill scratch capacity");
    TORCH_CHECK(prefill_scratch.is_contiguous(), "prefill_scratch must be contiguous");
    TORCH_CHECK(key_cache.scalar_type() == at::kByte, "key_cache must be uint8");
    TORCH_CHECK(value_cache.scalar_type() == at::kByte, "value_cache must be uint8");
    TORCH_CHECK(slot_mapping.scalar_type() == at::kLong, "slot_mapping must be int64");
    TORCH_CHECK(precision_table.scalar_type() == at::kByte, "precision_table must be uint8");
    TORCH_CHECK(block_offsets.scalar_type() == at::kInt, "block_offsets must be int32");
    TORCH_CHECK(scale_table.scalar_type() == at::kFloat, "scale_table must be float32");

    const int num_kv_heads = static_cast<int>(prefill_scratch.size(2));
    const int block_size = 16;
    const int head_dim = static_cast<int>(prefill_scratch.size(3));
    const int num_blocks = static_cast<int>(precision_table.size(0));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    PSP_KV_DISPATCH_SCALAR(prefill_scratch, {
        launch_quantize_scatter_multi<scalar_t>(
            reinterpret_cast<const scalar_t*>(prefill_scratch.data_ptr()),
            key_cache.data_ptr<uint8_t>(),
            value_cache.data_ptr<uint8_t>(),
            slot_mapping.data_ptr<int64_t>(),
            precision_table.data_ptr<uint8_t>(),
            reinterpret_cast<const uint32_t*>(block_offsets.data_ptr<int>()),
            scale_table.data_ptr<float>(),
            static_cast<int>(num_tokens), num_blocks, num_kv_heads, block_size, head_dim,
            static_cast<int>(format_generation), stream);
    });
}

// ----------------------------------------------------------------------------
// Fused precision derivation: estimate max-abs + derive per-block precision.
// ----------------------------------------------------------------------------
void derive_precision_kv(
    at::Tensor& prefill_scratch, // [num_tokens, 2, num_kv_heads, head_dim]
    at::Tensor& slot_mapping,    // [num_tokens] int64
    at::Tensor& precision_table, // [num_blocks] uint8 (in/out)
    at::Tensor& scale_table,     // [num_blocks, num_kv_heads] float32 (scale_table reused as max-abs workspace dtype)
    int64_t num_tokens,
    int64_t k_default,
    int64_t v_default,
    double E)
{
    TORCH_CHECK(num_tokens > 0, "num_tokens must be positive");
    TORCH_CHECK(prefill_scratch.is_contiguous(), "prefill_scratch must be contiguous");
    TORCH_CHECK(slot_mapping.is_contiguous(), "slot_mapping must be contiguous");
    TORCH_CHECK(precision_table.is_contiguous(), "precision_table must be contiguous");

    const int num_kv_heads = static_cast<int>(prefill_scratch.size(2));
    const int block_size = 16;
    const int head_dim = static_cast<int>(prefill_scratch.size(3));
    const int num_blocks = static_cast<int>(precision_table.size(0));

    auto max_abs = at::zeros({2 * (int64_t)num_blocks * num_kv_heads * psp_kv_num_groups(head_dim)}, scale_table.options());

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    PSP_KV_DISPATCH_SCALAR(prefill_scratch, {
        launch_derive_precision<scalar_t>(
            reinterpret_cast<const scalar_t*>(prefill_scratch.data_ptr()),
            slot_mapping.data_ptr<int64_t>(),
            precision_table.data_ptr<uint8_t>(),
            max_abs.data_ptr<float>(),
            static_cast<int>(num_tokens), num_blocks, num_kv_heads, block_size, head_dim,
            static_cast<int>(k_default), static_cast<int>(v_default),
            static_cast<float>(E), stream);
    });
}

// Decode derive: adapt a new block's K/V precision from the first token's
// magnitude (and refresh its scale), so decode-generated KV is predicted the
// same way prefill KV is. Called only when adaptive precision is enabled.
void derive_precision_decode_kv(
    at::Tensor& key,             // [num_tokens, num_kv_heads, head_dim]
    at::Tensor& value,           // [num_tokens, num_kv_heads, head_dim]
    at::Tensor& slot_mapping,    // [num_tokens] int64
    at::Tensor& precision_table, // [num_blocks] uint8 (in/out)
    at::Tensor& scale_table,     // [2, num_blocks, num_kv_heads] float32 (in/out)
    at::Tensor& max_abs,         // [2, num_blocks, num_kv_heads] float32 (scratch, zeroed)
    int64_t num_tokens,
    int64_t k_default,
    int64_t v_default,
    double E)
{
    TORCH_CHECK(num_tokens > 0, "num_tokens must be positive");
    TORCH_CHECK(key.is_contiguous() && value.is_contiguous(), "key/value must be contiguous");
    TORCH_CHECK(slot_mapping.is_contiguous(), "slot_mapping must be contiguous");
    TORCH_CHECK(precision_table.is_contiguous(), "precision_table must be contiguous");
    TORCH_CHECK(scale_table.is_contiguous(), "scale_table must be contiguous");
    TORCH_CHECK(max_abs.is_contiguous(), "max_abs must be contiguous");
    TORCH_CHECK(max_abs.scalar_type() == at::kFloat, "max_abs must be float32");

    const int num_kv_heads = static_cast<int>(key.size(1));
    const int block_size = 16;
    const int head_dim = static_cast<int>(key.size(2));
    const int num_blocks = static_cast<int>(precision_table.size(0));

    // The scratch is zeroed by the caller each step so the atomicMax in the
    // estimate kernel starts from a clean slate.
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    PSP_KV_DISPATCH_SCALAR(key, {
        launch_derive_precision_decode<scalar_t>(
            reinterpret_cast<const scalar_t*>(key.data_ptr()),
            reinterpret_cast<const scalar_t*>(value.data_ptr()),
            slot_mapping.data_ptr<int64_t>(),
            precision_table.data_ptr<uint8_t>(),
            scale_table.data_ptr<float>(),
            max_abs.data_ptr<float>(),
            static_cast<int>(num_tokens), num_blocks, num_kv_heads, block_size, head_dim,
            static_cast<int>(k_default), static_cast<int>(v_default),
            static_cast<float>(E), stream);
    });
}

// ----------------------------------------------------------------------------
// Decode write-back: quantize one token's K/V with the block's existing scale.
// ----------------------------------------------------------------------------
void quantize_single_token(
    at::Tensor& key,             // [num_tokens, num_kv_heads, head_dim] fp16/bf16
    at::Tensor& value,           // [num_tokens, num_kv_heads, head_dim] fp16/bf16
    at::Tensor& key_cache,       // uint8 byte buffer
    at::Tensor& value_cache,     // uint8 byte buffer
    at::Tensor& slot_mapping,    // [num_tokens] int64
    at::Tensor& precision_table, // [num_blocks] uint8
    at::Tensor& block_offsets,   // [num_blocks] int32
    at::Tensor& scale_table,     // [num_blocks, num_kv_heads] float32
    int64_t format_generation)
{
    TORCH_CHECK(key.is_contiguous() && value.is_contiguous(),
                "key and value must be contiguous");
    TORCH_CHECK(key_cache.scalar_type() == at::kByte, "key_cache must be uint8");
    TORCH_CHECK(value_cache.scalar_type() == at::kByte, "value_cache must be uint8");

    const int num_tokens = static_cast<int>(key.size(0));
    const int num_kv_heads = static_cast<int>(key.size(1));
    const int block_size = 16;
    const int head_dim = static_cast<int>(key.size(2));
    const int num_blocks = static_cast<int>(precision_table.size(0));

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    PSP_KV_DISPATCH_SCALAR(key, {
        launch_quantize_single_token<scalar_t>(
            reinterpret_cast<const scalar_t*>(key.data_ptr()),
            reinterpret_cast<const scalar_t*>(value.data_ptr()),
            key_cache.data_ptr<uint8_t>(),
            value_cache.data_ptr<uint8_t>(),
            slot_mapping.data_ptr<int64_t>(),
            precision_table.data_ptr<uint8_t>(),
            reinterpret_cast<const uint32_t*>(block_offsets.data_ptr<int>()),
            scale_table.data_ptr<float>(),
            num_tokens, num_kv_heads, block_size, head_dim, num_blocks, stream);
    });
}

// ----------------------------------------------------------------------------
// Dequantize-and-interpolate helper (Basic-format staging path)
// ----------------------------------------------------------------------------
void psp_dequantize_kv(
    at::Tensor& k_cache,        // uint8 byte buffer
    at::Tensor& v_cache,        // uint8 byte buffer
    at::Tensor& block_offsets,  // [num_blocks] int32
    at::Tensor& precision_table,// [num_blocks] uint8
    at::Tensor& scale_table,    // [num_blocks, num_kv_heads] float32
    int64_t num_blocks,
    int64_t num_kv_heads,
    int64_t block_size,
    int64_t head_size,
    at::Tensor& k_out,          // [num_blocks, num_kv_heads, block_size, head_size]
    at::Tensor& v_out)
{
    TORCH_CHECK(k_cache.scalar_type() == at::kByte, "k_cache must be uint8");
    TORCH_CHECK(v_cache.scalar_type() == at::kByte, "v_cache must be uint8");
    TORCH_CHECK(block_offsets.scalar_type() == at::kInt, "block_offsets must be int32");
    TORCH_CHECK(precision_table.scalar_type() == at::kByte, "precision_table must be uint8");
    TORCH_CHECK(scale_table.scalar_type() == at::kFloat, "scale_table must be float32");
    TORCH_CHECK(k_out.scalar_type() == v_out.scalar_type(), "k_out and v_out dtype mismatch");

    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    PSP_KV_DISPATCH_SCALAR(k_out, {
        launch_dequant_kv<scalar_t>(
            k_cache.data_ptr<uint8_t>(),
            v_cache.data_ptr<uint8_t>(),
            reinterpret_cast<const uint32_t*>(block_offsets.data_ptr<int>()),
            precision_table.data_ptr<uint8_t>(),
            scale_table.data_ptr<float>(),
            reinterpret_cast<scalar_t*>(k_out.data_ptr()),
            reinterpret_cast<scalar_t*>(v_out.data_ptr()),
            static_cast<int>(num_blocks), static_cast<int>(num_kv_heads),
            static_cast<int>(block_size), static_cast<int>(head_size), stream);
    });
}

void psp_dequantize_prefix(
    at::Tensor& k_cache,          // uint8 byte buffer
    at::Tensor& v_cache,          // uint8 byte buffer
    at::Tensor& block_table,      // [n_blk] int32 physical block ids
    at::Tensor& precision_table,  // [num_blocks] uint8
    at::Tensor& block_offsets,    // [num_blocks] int32 byte offsets
    at::Tensor& scale_table,      // [num_blocks, num_kv_heads] float32
    int64_t num_kv_heads,
    int64_t block_size,
    int64_t head_size,
    at::Tensor& k_out,            // [n_blk, num_kv_heads, block_size, head_size]
    at::Tensor& v_out)
{
    TORCH_CHECK(k_cache.scalar_type() == at::kByte, "k_cache must be uint8");
    TORCH_CHECK(v_cache.scalar_type() == at::kByte, "v_cache must be uint8");
    TORCH_CHECK(block_table.scalar_type() == at::kInt, "block_table must be int32");
    TORCH_CHECK(precision_table.scalar_type() == at::kByte, "precision_table must be uint8");
    TORCH_CHECK(block_offsets.scalar_type() == at::kInt, "block_offsets must be int32");
    TORCH_CHECK(scale_table.scalar_type() == at::kFloat, "scale_table must be float32");
    TORCH_CHECK(k_out.scalar_type() == v_out.scalar_type(), "k_out and v_out dtype mismatch");

    const int n_blk = static_cast<int>(block_table.size(0));
    const int num_blocks = static_cast<int>(precision_table.size(0));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    PSP_KV_DISPATCH_SCALAR(k_out, {
        launch_dequant_prefix<scalar_t>(
            k_cache.data_ptr<uint8_t>(),
            v_cache.data_ptr<uint8_t>(),
            block_table.data_ptr<int>(),
            precision_table.data_ptr<uint8_t>(),
            reinterpret_cast<const uint32_t*>(block_offsets.data_ptr<int>()),
            scale_table.data_ptr<float>(),
            reinterpret_cast<scalar_t*>(k_out.data_ptr()),
            reinterpret_cast<scalar_t*>(v_out.data_ptr()),
            n_blk, num_blocks, static_cast<int>(num_kv_heads),
            static_cast<int>(block_size), static_cast<int>(head_size), stream);
    });
}

void psp_dequantize_prefix_batch(
    at::Tensor& k_cache,          // uint8 byte buffer
    at::Tensor& v_cache,          // uint8 byte buffer
    at::Tensor& block_tables,     // [num_seqs, max_blocks_per_seq] int32
    at::Tensor& ctx_lens,         // [num_seqs] int32 cached-prefix lengths
    at::Tensor& out_offsets,      // [num_seqs] int32 padded prefix-sum
    at::Tensor& precision_table,  // [num_blocks] uint8
    at::Tensor& block_offsets,    // [num_blocks] int32 byte offsets
    at::Tensor& scale_table,      // [num_blocks, num_kv_heads] float32
    int64_t num_kv_heads,
    int64_t block_size,
    int64_t head_size,
    at::Tensor& k_out,            // [total_padded, num_kv_heads, head_size]
    at::Tensor& v_out)
{
    TORCH_CHECK(k_cache.scalar_type() == at::kByte, "k_cache must be uint8");
    TORCH_CHECK(v_cache.scalar_type() == at::kByte, "v_cache must be uint8");
    TORCH_CHECK(block_tables.scalar_type() == at::kInt, "block_tables must be int32");
    TORCH_CHECK(ctx_lens.scalar_type() == at::kInt, "ctx_lens must be int32");
    TORCH_CHECK(out_offsets.scalar_type() == at::kInt, "out_offsets must be int32");
    TORCH_CHECK(precision_table.scalar_type() == at::kByte, "precision_table must be uint8");
    TORCH_CHECK(block_offsets.scalar_type() == at::kInt, "block_offsets must be int32");
    TORCH_CHECK(scale_table.scalar_type() == at::kFloat, "scale_table must be float32");
    TORCH_CHECK(k_out.scalar_type() == v_out.scalar_type(), "k_out and v_out dtype mismatch");

    const int num_seqs = static_cast<int>(block_tables.size(0));
    const int max_blocks_per_seq = static_cast<int>(block_tables.size(1));
    const int num_blocks = static_cast<int>(precision_table.size(0));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();
    PSP_KV_DISPATCH_SCALAR(k_out, {
        launch_dequant_prefix_batch<scalar_t>(
            k_cache.data_ptr<uint8_t>(),
            v_cache.data_ptr<uint8_t>(),
            block_tables.data_ptr<int>(),
            ctx_lens.data_ptr<int>(),
            out_offsets.data_ptr<int>(),
            precision_table.data_ptr<uint8_t>(),
            reinterpret_cast<const uint32_t*>(block_offsets.data_ptr<int>()),
            scale_table.data_ptr<float>(),
            reinterpret_cast<scalar_t*>(k_out.data_ptr()),
            reinterpret_cast<scalar_t*>(v_out.data_ptr()),
            num_seqs, max_blocks_per_seq, num_blocks,
            static_cast<int>(num_kv_heads), static_cast<int>(block_size),
            static_cast<int>(head_size), stream);
    });
}

// Multi-token per-K/V tensor-core attention (batched-token path). N query
// tokens per sequence share one K/V read + dequant.
void psp_attention_enhanced_mt(
    at::Tensor& query,          // [num_tokens, num_heads, head_size]
    at::Tensor& key,            // [num_tokens, num_kv_heads, head_size] fp16
    at::Tensor& value,          // [num_tokens, num_kv_heads, head_size] fp16
    at::Tensor& key_cache,      // uint8 byte buffer
    at::Tensor& value_cache,
    at::Tensor& block_tables,   // [num_seqs, max_blocks] int32
    at::Tensor& context_lens,   // [num_seqs] int32
    at::Tensor& precision_table,
    at::Tensor& block_offsets,
    at::Tensor& scale_table,
    double softmax_scale,
    int64_t num_kv_heads,
    int64_t head_size,
    int64_t fp8_fmt,
    int64_t num_query_tokens,
    at::Tensor& out)
{
    TORCH_CHECK(query.is_contiguous(), "query must be contiguous");
    TORCH_CHECK(key.scalar_type() == query.scalar_type(), "key dtype mismatch");
    TORCH_CHECK(value.scalar_type() == query.scalar_type(), "value dtype mismatch");
    TORCH_CHECK(key_cache.scalar_type() == at::kByte, "key_cache must be uint8");
    TORCH_CHECK(value_cache.scalar_type() == at::kByte, "value_cache must be uint8");
    TORCH_CHECK(block_tables.scalar_type() == at::kInt, "block_tables must be int32");
    TORCH_CHECK(context_lens.scalar_type() == at::kInt, "context_lens must be int32");
    TORCH_CHECK(precision_table.scalar_type() == at::kByte, "precision_table must be uint8");
    TORCH_CHECK(block_offsets.scalar_type() == at::kInt, "block_offsets must be int32");
    TORCH_CHECK(scale_table.scalar_type() == at::kFloat, "scale_table must be float32");
    TORCH_CHECK(out.scalar_type() == query.scalar_type(), "query and out dtype mismatch");

    const int num_tokens = static_cast<int>(query.size(0));
    const int num_heads = static_cast<int>(query.size(1));
    const int num_seqs = static_cast<int>(block_tables.size(0));
    const int max_blocks_per_seq = static_cast<int>(block_tables.size(1));
    const int num_total_blocks = static_cast<int>(precision_table.size(0));
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    PSP_KV_DISPATCH_HALF_BF16(query, {
        launch_paged_attention_enhanced_fp8pv_mt<scalar_t>(
            reinterpret_cast<scalar_t*>(out.data_ptr()),
            reinterpret_cast<const scalar_t*>(query.data_ptr()),
            reinterpret_cast<const scalar_t*>(key.data_ptr()),
            reinterpret_cast<const scalar_t*>(value.data_ptr()),
            key_cache.data_ptr<uint8_t>(),
            value_cache.data_ptr<uint8_t>(),
            block_tables.data_ptr<int>(),
            context_lens.data_ptr<int>(),
            reinterpret_cast<const uint32_t*>(block_offsets.data_ptr<int>()),
            scale_table.data_ptr<float>(),
            precision_table.data_ptr<uint8_t>(),
            max_blocks_per_seq, num_tokens, num_heads,
            static_cast<int>(num_kv_heads), static_cast<int>(head_size),
            static_cast<float>(softmax_scale), static_cast<int>(fp8_fmt),
            num_total_blocks, static_cast<int>(num_query_tokens), stream);
    });
}

// ----------------------------------------------------------------------------
// Uniform-consumer paged attention (Basic-format staging path)
// ----------------------------------------------------------------------------
void psp_attention_uniform(
    at::Tensor& query,      // [num_tokens, num_heads, head_dim] target precision
    at::Tensor& key_cache,  // [num_blocks, num_kv_heads, block_size, head_dim]
    at::Tensor& value_cache,
    at::Tensor& block_tables,
    at::Tensor& context_lens,
    double softmax_scale,
    int64_t num_kv_heads,
    int64_t head_size,
    at::Tensor& out)
{
    TORCH_CHECK(key_cache.scalar_type() == query.scalar_type(), "key_cache dtype must match query");
    TORCH_CHECK(value_cache.scalar_type() == query.scalar_type(), "value_cache dtype must match query");

    int num_tokens = query.size(0);
    int num_seqs = context_lens.size(0);
    int max_blocks_per_seq = block_tables.size(1);
    int num_heads = query.size(1);
    cudaStream_t stream = at::cuda::getCurrentCUDAStream().stream();

    PSP_KV_DISPATCH_SCALAR(query, {
        launch_paged_attention_uniform<scalar_t>(
            reinterpret_cast<scalar_t*>(out.data_ptr()),
            reinterpret_cast<const scalar_t*>(query.data_ptr()),
            reinterpret_cast<const scalar_t*>(key_cache.data_ptr()),
            reinterpret_cast<const scalar_t*>(value_cache.data_ptr()),
            block_tables.data_ptr<int>(),
            context_lens.data_ptr<int>(),
            max_blocks_per_seq, num_tokens, num_seqs,
            num_heads, static_cast<int>(num_kv_heads),
            static_cast<int>(head_size), static_cast<float>(softmax_scale), stream);
    });
}

}  // namespace psp_kv

// ============================================================================
// Op registration via the PyTorch dispatcher.
// ============================================================================

TORCH_LIBRARY(psp_kv, m) {
    m.def(
        "psp_attention_basic("
        "Tensor query, "
        "Tensor key_cache, "
        "Tensor value_cache, "
        "Tensor block_tables, "
        "Tensor context_lens, "
        "Tensor precision_table, "
        "Tensor block_offsets, "
        "float softmax_scale, "
        "int num_kv_heads, "
        "int head_size, "
        "Tensor! out"
        ") -> ()");

    m.def(
        "psp_attention_enhanced("
        "Tensor query, "
        "Tensor key_cache, "
        "Tensor value_cache, "
        "Tensor block_tables, "
        "Tensor context_lens, "
        "Tensor precision_table, "
        "Tensor block_offsets, "
        "Tensor scale_table, "
        "float softmax_scale, "
        "int num_kv_heads, "
        "int head_size, "
        "int sliding_window, "
        "bool is_lossless, "
        "int fp8_fmt, "
        "int v_fmt, "
        "int k_fmt, "
        "bool k_lossless, "
        "bool v_lossless, "
        "Tensor! out"
        ") -> ()");

    m.def(
        "psp_attention_btkv("
        "Tensor query, "
        "Tensor key_cache, "
        "Tensor value_cache, "
        "Tensor block_tables, "
        "Tensor context_lens, "
        "Tensor precision_table, "
        "Tensor block_offsets, "
        "Tensor scale_table, "
        "Tensor tma_descriptors, "
        "float softmax_scale, "
        "int num_kv_heads, "
        "int head_size, "
        "Tensor! out"
        ") -> ()");

    m.def(
        "quantize_and_scatter_kv("
        "Tensor prefill_scratch, "
        "Tensor! key_cache, "
        "Tensor! value_cache, "
        "Tensor block_tables, "
        "Tensor slot_mapping, "
        "Tensor precision_table, "
        "Tensor block_offsets, "
        "Tensor! scale_table, "
        "int num_tokens, "
        "int format_generation"
        ") -> ()");

    m.def(
        "quantize_scatter_multi("
        "Tensor prefill_scratch, "
        "Tensor! key_cache, "
        "Tensor! value_cache, "
        "Tensor slot_mapping, "
        "Tensor precision_table, "
        "Tensor block_offsets, "
        "Tensor scale_table, "
        "int num_tokens, "
        "int format_generation"
        ") -> ()");

    m.def(
        "derive_precision_kv("
        "Tensor prefill_scratch, "
        "Tensor slot_mapping, "
        "Tensor! precision_table, "
        "Tensor scale_table, "
        "int num_tokens, "
        "int k_default, "
        "int v_default, "
        "float E"
        ") -> ()");

    m.def(
        "derive_precision_decode_kv("
        "Tensor key, "
        "Tensor value, "
        "Tensor slot_mapping, "
        "Tensor! precision_table, "
        "Tensor! scale_table, "
        "Tensor! max_abs, "
        "int num_tokens, "
        "int k_default, "
        "int v_default, "
        "float E"
        ") -> ()");

    m.def(
        "quantize_single_token("
        "Tensor key, "
        "Tensor value, "
        "Tensor! key_cache, "
        "Tensor! value_cache, "
        "Tensor slot_mapping, "
        "Tensor precision_table, "
        "Tensor block_offsets, "
        "Tensor scale_table, "
        "int format_generation"
        ") -> ()");

    m.def(
        "psp_dequantize_kv("
        "Tensor k_cache, "
        "Tensor v_cache, "
        "Tensor block_offsets, "
        "Tensor precision_table, "
        "Tensor scale_table, "
        "int num_blocks, "
        "int num_kv_heads, "
        "int block_size, "
        "int head_size, "
        "Tensor! k_out, "
        "Tensor! v_out"
        ") -> ()");

    m.def(
        "psp_dequantize_prefix("
        "Tensor k_cache, "
        "Tensor v_cache, "
        "Tensor block_table, "
        "Tensor precision_table, "
        "Tensor block_offsets, "
        "Tensor scale_table, "
        "int num_kv_heads, "
        "int block_size, "
        "int head_size, "
        "Tensor! k_out, "
        "Tensor! v_out"
        ") -> ()");

    m.def(
        "psp_dequantize_prefix_batch("
        "Tensor k_cache, "
        "Tensor v_cache, "
        "Tensor block_tables, "
        "Tensor ctx_lens, "
        "Tensor out_offsets, "
        "Tensor precision_table, "
        "Tensor block_offsets, "
        "Tensor scale_table, "
        "int num_kv_heads, "
        "int block_size, "
        "int head_size, "
        "Tensor! k_out, "
        "Tensor! v_out"
        ") -> ()");

    m.def(
        "psp_attention_uniform("
        "Tensor query, "
        "Tensor key_cache, "
        "Tensor value_cache, "
        "Tensor block_tables, "
        "Tensor context_lens, "
        "float softmax_scale, "
        "int num_kv_heads, "
        "int head_size, "
        "Tensor! out"
        ") -> ()");

    m.def(
        "psp_attention_enhanced_mt("
        "Tensor query, "
        "Tensor key, "
        "Tensor value, "
        "Tensor key_cache, "
        "Tensor value_cache, "
        "Tensor block_tables, "
        "Tensor context_lens, "
        "Tensor precision_table, "
        "Tensor block_offsets, "
        "Tensor scale_table, "
        "float softmax_scale, "
        "int num_kv_heads, "
        "int head_size, "
        "int fp8_fmt, "
        "int num_query_tokens, "
        "Tensor! out"
        ") -> ()");
}

TORCH_LIBRARY_IMPL(psp_kv, CUDA, m) {
    m.impl("psp_attention_basic",     &psp_kv::psp_attention_basic);
    m.impl("psp_attention_enhanced",  &psp_kv::psp_attention_enhanced);
    m.impl("psp_attention_enhanced_mt", &psp_kv::psp_attention_enhanced_mt);
    m.impl("psp_attention_btkv",      &psp_kv::psp_attention_btkv);
    m.impl("quantize_and_scatter_kv", &psp_kv::quantize_and_scatter_kv);
    m.impl("quantize_scatter_multi", &psp_kv::quantize_scatter_multi);
    m.impl("derive_precision_kv", &psp_kv::derive_precision_kv);
    m.impl("derive_precision_decode_kv", &psp_kv::derive_precision_decode_kv);
    m.impl("quantize_single_token", &psp_kv::quantize_single_token);
    m.impl("psp_dequantize_kv", &psp_kv::psp_dequantize_kv);
    m.impl("psp_dequantize_prefix", &psp_kv::psp_dequantize_prefix);
    m.impl("psp_dequantize_prefix_batch", &psp_kv::psp_dequantize_prefix_batch);
    m.impl("psp_attention_uniform", &psp_kv::psp_attention_uniform);
}
