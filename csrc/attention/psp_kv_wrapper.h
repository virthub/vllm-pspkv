// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#pragma once

// Do NOT use <torch/extension.h> here: it pulls in <torch/python.h> (pybind11
// + Python.h), and both _C and _C_stable_libtorch are compiled with
// Py_LIMITED_API=3, under which modern pybind11 headers fail to compile.
// <torch/types.h> provides torch::Tensor without the Python bindings.
#include <torch/types.h>

// ----------------------------------------------------------------------------
// PSP‑KV attention kernels (host wrappers)
// ----------------------------------------------------------------------------

namespace psp_kv {

/**
 * Basic format: in‑band metadata, symmetric 1D layout.
 *
 * @param query          [num_tokens, num_heads, head_dim] input query (FP16/BF16/FP32)
 * @param key_cache      [num_blocks, num_kv_heads, block_size, head_dim] FP8 cache
 * @param value_cache    [num_blocks, num_kv_heads, block_size, head_dim] FP8 cache
 * @param block_tables   [num_seqs, max_blocks_per_seq] int32 block table
 * @param context_lens   [num_seqs] int32 sequence lengths
 * @param precision_table [num_blocks] uint8 precision level per block
 * @param softmax_scale  scaling factor for attention (float)
 * @param num_kv_heads   number of KV heads
 * @param head_size      head dimension
 * @param out            [num_tokens, num_heads, head_dim] output tensor (in‑place)
 */
void psp_attention_basic(
    torch::Tensor& query,
    torch::Tensor& key_cache,
    torch::Tensor& value_cache,
    torch::Tensor& block_tables,
    torch::Tensor& context_lens,
    torch::Tensor& precision_table,
    torch::Tensor& block_offsets,
    double softmax_scale,
    int64_t num_kv_heads,
    int64_t head_size,
    torch::Tensor& out);

/**
 * Enhanced format: out‑of‑band scale_table, asymmetric 1D layout.
 *
 * @param scale_table    [num_blocks, num_kv_heads] float32 per‑head scales
 * (additional parameters same as basic)
 */
void psp_attention_enhanced(
    torch::Tensor& query,
    torch::Tensor& key_cache,
    torch::Tensor& value_cache,
    torch::Tensor& block_tables,
    torch::Tensor& context_lens,
    torch::Tensor& precision_table,
    torch::Tensor& block_offsets,
    torch::Tensor& scale_table,
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
    torch::Tensor& out);

/**
 * BT‑KV format: 2D block‑tiled with TMA descriptors.
 *
 * @param tma_descriptors [num_blocks, 16] uint64 TMA descriptors (128 bytes each)
 */
void psp_attention_btkv(
    torch::Tensor& query,
    torch::Tensor& key_cache,
    torch::Tensor& value_cache,
    torch::Tensor& block_tables,
    torch::Tensor& context_lens,
    torch::Tensor& precision_table,
    torch::Tensor& block_offsets,
    torch::Tensor& scale_table,
    torch::Tensor& tma_descriptors,
    double softmax_scale,
    int64_t num_kv_heads,
    int64_t head_size,
    torch::Tensor& out);

/**
 * Quantize FP16/BF16 prefill scratch buffer and scatter into FP8 cache.
 *
 * @param prefill_scratch [num_tokens, 2, num_kv_heads, head_dim] FP16/BF16 scratch
 * @param key_cache      [num_blocks, num_kv_heads, block_size, head_dim] FP8 key cache
 * @param value_cache    [num_blocks, num_kv_heads, block_size, head_dim] FP8 value cache
 * @param block_tables   [num_seqs, max_blocks_per_seq] int32 block table
 * @param slot_mapping   [num_tokens] int64 slot indices
 * @param scale_table    [num_blocks, num_kv_heads] float32 scale table (updated in‑place)
 * @param num_tokens     number of tokens to process
 */
void quantize_and_scatter_kv(
    torch::Tensor& prefill_scratch,
    torch::Tensor& key_cache,
    torch::Tensor& value_cache,
    torch::Tensor& block_tables,
    torch::Tensor& slot_mapping,
    torch::Tensor& precision_table,
    torch::Tensor& block_offsets,
    torch::Tensor& scale_table,
    int64_t num_tokens,
    int64_t format_generation);

void psp_dequantize_kv(
    torch::Tensor& k_cache,
    torch::Tensor& v_cache,
    torch::Tensor& block_offsets,
    torch::Tensor& precision_table,
    torch::Tensor& scale_table,
    int64_t num_blocks,
    int64_t num_kv_heads,
    int64_t block_size,
    int64_t head_size,
    torch::Tensor& k_out,
    torch::Tensor& v_out);

void psp_attention_uniform(
    torch::Tensor& query,
    torch::Tensor& key_cache,
    torch::Tensor& value_cache,
    torch::Tensor& block_tables,
    torch::Tensor& context_lens,
    double softmax_scale,
    int64_t num_kv_heads,
    int64_t head_size,
    torch::Tensor& out);

}  // namespace psp_kv