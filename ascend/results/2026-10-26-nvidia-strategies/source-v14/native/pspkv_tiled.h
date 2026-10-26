// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#pragma once

// Included after the reference codecs in pspkv_fp8.asc. All working storage is
// private to one AIV. Tables and the transpose map are loaded once, not per
// row.
constexpr uint32_t PSPKV_TILE = 32;

class PspkvTileDecoder {
 public:
  __aicore__ inline void Init(AscendC::TPipe& pipe, uint32_t dim, GM_ADDR e4,
                              GM_ADDR e5, GM_ADDR permutation,
                              bool pair_lut = false) {
    dimension = dim;
    count = PSPKV_TILE * dim;
    fp4_from_gm = pair_lut;
    if (pair_lut)
      fp4_global.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(permutation) +
                                 count);
    pipe.InitBuffer(inputBuffer, count);
    pipe.InitBuffer(halfBuffer, count * 2);
    pipe.InitBuffer(indexBuffer, count * 4);
    pipe.InitBuffer(decodedBuffer, count * 4);
    pipe.InitBuffer(transposedBuffer, count * 4);
    pipe.InitBuffer(permutationBuffer, count * 4);
    pipe.InitBuffer(tableBuffer, 1024 * 4);
    pipe.InitBuffer(outputBuffer, count * 2);
    bytes = inputBuffer.Get<uint8_t>();
    halves = halfBuffer.Get<half>();
    indices = indexBuffer.Get<int32_t>();
    decoded = decodedBuffer.Get<float>();
    transposed = transposedBuffer.Get<float>();
    permutationLocal = permutationBuffer.Get<uint32_t>();
    table = tableBuffer.Get<float>();
    bf16 = outputBuffer.Get<bfloat16_t>();
    AscendC::GlobalTensor<float> e4Global, e5Global;
    AscendC::GlobalTensor<uint32_t> permutationGlobal;
    e4Global.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(e4));
    e5Global.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(e5));
    permutationGlobal.SetGlobalBuffer(
        reinterpret_cast<__gm__ uint32_t*>(permutation));
    AscendC::DataCopy(table, e4Global, 256);
    AscendC::DataCopy(table[256], e5Global, 256);
    AscendC::DataCopy(permutationLocal, permutationGlobal, count);
    pipe_barrier(PIPE_ALL);
  }

  __aicore__ inline AscendC::LocalTensor<float> Load(
      AscendC::GlobalTensor<uint8_t>& source, uint32_t base, uint8_t code,
      float scale, uint32_t token, uint32_t block_size, bool transpose,
      bool token_major = true) {
    if (code == 0 && !transpose) {
      AscendC::DataCopy(halves.ReinterpretCast<uint8_t>(),
                        source[base + token * dimension * 2], count * 2);
      pipe_barrier(PIPE_ALL);
      AscendC::Cast(decoded, halves.ReinterpretCast<bfloat16_t>(),
                    AscendC::RoundMode::CAST_NONE, count);
      pipe_barrier(PIPE_V);
      return decoded;
    }
    if (code == 3 && !transpose) {
      if (!fp4_initialized) {
        if (fp4_from_gm) {
          AscendC::DataCopy(table[512], fp4_global, 512);
        } else {
          for (uint32_t byte = 0; byte < 256; ++byte) {
            table.SetValue(512 + byte * 2, DecodeFP4(byte & 15));
            table.SetValue(513 + byte * 2, DecodeFP4(byte >> 4));
          }
        }
        pipe_barrier(PIPE_ALL);
        fp4_initialized = true;
      }
      AscendC::DataCopy(bytes, source[base + token * dimension / 2], count / 2);
      pipe_barrier(PIPE_ALL);
      AscendC::Cast(halves, bytes, AscendC::RoundMode::CAST_NONE, count / 2);
      pipe_barrier(PIPE_V);
      AscendC::Cast(indices, halves, AscendC::RoundMode::CAST_RINT, count / 2);
      pipe_barrier(PIPE_V);
      auto offsets = transposed.ReinterpretCast<uint32_t>();
      AscendC::CreateVecIndex(offsets.ReinterpretCast<int32_t>(),
                              static_cast<int32_t>(0), count);
      pipe_barrier(PIPE_V);
      AscendC::ShiftRight(offsets, offsets, static_cast<uint32_t>(1), count);
      pipe_barrier(PIPE_V);
      AscendC::ShiftLeft(offsets, offsets, static_cast<uint32_t>(2), count);
      pipe_barrier(PIPE_V);
      AscendC::Gather(decoded.ReinterpretCast<uint32_t>(),
                      indices.ReinterpretCast<uint32_t>(), offsets, 0, count);
      pipe_barrier(PIPE_V);
      AscendC::CreateVecIndex(indices, static_cast<int32_t>(0), count);
      pipe_barrier(PIPE_V);
      AscendC::ShiftRight(offsets, indices.ReinterpretCast<uint32_t>(),
                          static_cast<uint32_t>(1), count);
      pipe_barrier(PIPE_V);
      AscendC::ShiftLeft(offsets, offsets, static_cast<uint32_t>(1), count);
      pipe_barrier(PIPE_V);
      AscendC::Sub(indices, indices, offsets.ReinterpretCast<int32_t>(), count);
      AscendC::ShiftLeft(decoded.ReinterpretCast<uint32_t>(),
                         decoded.ReinterpretCast<uint32_t>(),
                         static_cast<uint32_t>(1), count);
      pipe_barrier(PIPE_V);
      AscendC::Add(indices, indices, decoded.ReinterpretCast<int32_t>(), count);
      pipe_barrier(PIPE_V);
      AscendC::ShiftLeft(indices.ReinterpretCast<uint32_t>(),
                         indices.ReinterpretCast<uint32_t>(),
                         static_cast<uint32_t>(2), count);
      pipe_barrier(PIPE_V);
      AscendC::Gather(decoded, table, indices.ReinterpretCast<uint32_t>(),
                      512 * 4, count);
      pipe_barrier(PIPE_V);
      if (scale != 1.0f) {
        AscendC::Muls(decoded, decoded, scale, count);
        pipe_barrier(PIPE_V);
      }
      return decoded;
    }
    if (code != 1 && code != 2) {
      // Preserve mixed BF16/FP4 contracts; the FP8 hot path below contains
      // no per-element scalar loads, conversions or stores.
      auto result = transpose && token_major ? transposed : decoded;
      for (uint32_t row = 0; row < PSPKV_TILE; ++row) {
        for (uint32_t dim = 0; dim < dimension; ++dim) {
          const uint32_t element = transpose ? dim * block_size + token + row
                                             : (token + row) * dimension + dim;
          result.SetValue(transpose && !token_major ? dim * PSPKV_TILE + row
                                                    : row * dimension + dim,
                          ReadPacked(source, base, element, code, 0, scale));
        }
      }
      pipe_barrier(PIPE_ALL);
      return result;
    }
    if (transpose) {
      AscendC::DataCopyParams copy{
          static_cast<uint16_t>(dimension), 1,
          static_cast<uint16_t>((block_size - PSPKV_TILE) / 32), 0};
      AscendC::DataCopy(bytes, source[base + token], copy);
    } else {
      AscendC::DataCopy(bytes, source[base + token * dimension], count);
    }
    pipe_barrier(PIPE_ALL);
    // 910B has no native FP8 cast. Byte -> integer offsets -> FP32 UB LUT
    // is entirely vectorized and also preserves subnormals and signed zero.
    AscendC::Cast(halves, bytes, AscendC::RoundMode::CAST_NONE, count);
    pipe_barrier(PIPE_V);
    AscendC::Cast(indices, halves, AscendC::RoundMode::CAST_RINT, count);
    pipe_barrier(PIPE_V);
    auto offsets = indices.ReinterpretCast<uint32_t>();
    AscendC::ShiftLeft(offsets, offsets, static_cast<uint32_t>(2), count);
    pipe_barrier(PIPE_V);
    AscendC::Gather(decoded, table, offsets, code == 2 ? 1024 : 0, count);
    pipe_barrier(PIPE_V);
    if (scale != 1.0f) {
      AscendC::Muls(decoded, decoded, scale, count);
      pipe_barrier(PIPE_V);
    }
    if (transpose && token_major) {
      AscendC::Gather(transposed, decoded, permutationLocal, 0, count);
      pipe_barrier(PIPE_V);
      return transposed;
    }
    return decoded;
  }

  __aicore__ inline AscendC::LocalTensor<bfloat16_t> Round(
      AscendC::LocalTensor<float> values) {
    AscendC::Cast(bf16, values, AscendC::RoundMode::CAST_RINT, count);
    pipe_barrier(PIPE_V);
    return bf16;
  }

  __aicore__ inline void RoundInPlace(AscendC::LocalTensor<float> values) {
    Round(values);
    AscendC::Cast(values, bf16, AscendC::RoundMode::CAST_NONE, count);
    pipe_barrier(PIPE_V);
  }

  __aicore__ inline AscendC::LocalTensor<float> DotProducts(
      AscendC::LocalTensor<float> query, AscendC::LocalTensor<float> keys,
      AscendC::LocalTensor<float> partials) {
    // The lookup indices are dead after Load. Reuse their UB allocation
    // for a broadcast query/product rather than reserving another matrix.
    auto product = indices.ReinterpretCast<float>();
    for (uint32_t dim = 0; dim < dimension; dim += 64) {
      AscendC::Copy(product[dim], query[dim], 64, PSPKV_TILE,
                    {1, 1, static_cast<uint16_t>(dimension / 8), 0});
    }
    pipe_barrier(PIPE_V);
    AscendC::Mul(product, keys, product, count);
    pipe_barrier(PIPE_V);
    AscendC::WholeReduceSum(partials, product, 64, count / 64, 1, 1, 8);
    pipe_barrier(PIPE_V);
    return partials;
  }

  __aicore__ inline void WeightedSum(AscendC::LocalTensor<float> values,
                                     AscendC::LocalTensor<float> probabilities,
                                     AscendC::LocalTensor<float> accumulator,
                                     float alpha, bool transpose) {
    auto offsets = indices.ReinterpretCast<uint32_t>();
    AscendC::CreateVecIndex(indices, static_cast<int32_t>(0), count);
    pipe_barrier(PIPE_V);
    AscendC::ShiftRight(offsets, offsets,
                        static_cast<uint32_t>(dimension == 128 ? 7 : 8), count);
    pipe_barrier(PIPE_V);
    AscendC::ShiftLeft(offsets, offsets, static_cast<uint32_t>(2), count);
    pipe_barrier(PIPE_V);
    auto weights = transpose ? decoded : transposed;
    AscendC::Gather(weights, probabilities, offsets, 0, count);
    pipe_barrier(PIPE_V);
    AscendC::Mul(values, values, weights, count);
    pipe_barrier(PIPE_V);
    // Tree reduction across tile rows, retaining all dimensions in SIMD.
    for (uint32_t remaining = count / 2; remaining >= dimension;
         remaining /= 2) {
      AscendC::Add(values, values, values[remaining], remaining);
      pipe_barrier(PIPE_V);
    }
    AscendC::Muls(accumulator, accumulator, alpha, dimension);
    pipe_barrier(PIPE_V);
    AscendC::Add(accumulator, accumulator, values, dimension);
    pipe_barrier(PIPE_V);
  }

  __aicore__ inline void WeightedSumTransposed(
      AscendC::LocalTensor<float> values,
      AscendC::LocalTensor<float> probabilities,
      AscendC::LocalTensor<float> accumulator, float alpha) {
    // Keep Enhanced/BTKV V in its existing [dimension, token] layout in UB.
    // Broadcast probabilities, then reduce each contiguous token row directly.
    for (uint32_t dim = 0; dim < dimension; dim += 128) {
      const uint32_t repeats = dimension - dim < 128 ? dimension - dim : 128;
      AscendC::Mul(values[dim * PSPKV_TILE], values[dim * PSPKV_TILE],
                   probabilities, PSPKV_TILE, repeats, {1, 1, 1, 4, 4, 0});
      pipe_barrier(PIPE_V);
      AscendC::WholeReduceSum(transposed[dim], values[dim * PSPKV_TILE],
                              PSPKV_TILE, repeats, 1, 1, 4);
    }
    pipe_barrier(PIPE_V);
    AscendC::Muls(accumulator, accumulator, alpha, dimension);
    pipe_barrier(PIPE_V);
    AscendC::Add(accumulator, accumulator, transposed, dimension);
    pipe_barrier(PIPE_V);
  }

  uint32_t dimension, count;
  AscendC::LocalTensor<bfloat16_t> bf16;

 private:
  bool fp4_initialized = false;
  bool fp4_from_gm = false;
  AscendC::GlobalTensor<float> fp4_global;
  AscendC::TBuf<AscendC::TPosition::VECCALC> inputBuffer, halfBuffer,
      indexBuffer, decodedBuffer, transposedBuffer, permutationBuffer,
      tableBuffer, outputBuffer;
  AscendC::LocalTensor<uint8_t> bytes;
  AscendC::LocalTensor<half> halves;
  AscendC::LocalTensor<int32_t> indices;
  AscendC::LocalTensor<uint32_t> permutationLocal;
  AscendC::LocalTensor<float> decoded, transposed, table;
};

__global__ __vector__ void PspkvGatherPagedBF16Vec(
    GM_ADDR key_cache, GM_ADDR value_cache, GM_ADDR block_tables,
    GM_ADDR context_lens, GM_ADDR formats, GM_ADDR offsets, GM_ADDR scales,
    GM_ADDR e4, GM_ADDR e5, GM_ADDR permutation, GM_ADDR key_output,
    GM_ADDR value_output, uint32_t seqs, uint32_t max_blocks, uint32_t heads,
    uint32_t dimension, uint32_t block_size, uint32_t generation,
    uint32_t pair_lut) {
  AscendC::GlobalTensor<uint8_t> keys, values, codes;
  AscendC::GlobalTensor<int32_t> tables, lengths, byte_offsets;
  AscendC::GlobalTensor<float> per_head_scales;
  AscendC::GlobalTensor<bfloat16_t> key_out, value_out;
  keys.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(key_cache));
  values.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(value_cache));
  codes.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(formats));
  tables.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(block_tables));
  lengths.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(context_lens));
  byte_offsets.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(offsets));
  per_head_scales.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(scales));
  key_out.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(key_output));
  value_out.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(value_output));
  AscendC::TPipe pipe;
  PspkvTileDecoder decoder;
  decoder.Init(pipe, dimension, e4, e5, permutation, pair_lut);
  const uint32_t chunks = block_size / PSPKV_TILE;
  const uint32_t tasks = seqs * max_blocks * heads * chunks;
  for (uint32_t task = AscendC::GetBlockIdx(); task < 2 * tasks;
       task += AscendC::GetBlockNum()) {
    const bool is_key = task < tasks;
    uint32_t index = task % tasks;
    const uint32_t token = index % chunks * PSPKV_TILE;
    index /= chunks;
    const uint32_t head = index % heads;
    index /= heads;
    const uint32_t logical_block = index % max_blocks;
    const uint32_t seq = index / max_blocks;
    const uint32_t length = lengths.GetValue(seq);
    // The vendor consumer masks by length. Do not clear wholly inactive
    // scratch pages on the opt-in strategy route, especially for short input
    // captured in a graph with a long maximum context.
    if (generation == 3 && logical_block * block_size >= length) continue;
    const uint32_t start = logical_block * block_size + token;
    auto local = decoder.bf16;
    if (start >= length) {
      AscendC::Duplicate(local.ReinterpretCast<uint16_t>(),
                         static_cast<uint16_t>(0), decoder.count);
    } else {
      const uint32_t physical =
          tables.GetValue(seq * max_blocks + logical_block);
      const uint8_t code = codes.GetValue(physical);
      const uint32_t base =
          byte_offsets.GetValue(physical) +
          head * PackedBytesPerHead(code, block_size, dimension, 0);
      const float scale =
          code == 0 || generation == 0
              ? 1.0f
              : per_head_scales.GetValue(physical * heads + head);
      local = decoder.Round(decoder.Load(
          is_key ? keys : values, base, code, scale, token, block_size,
          !is_key && (generation == 1 || generation == 2)));
      if (start + PSPKV_TILE > length) {
        AscendC::Duplicate(
            local[(length - start) * dimension].ReinterpretCast<uint16_t>(),
            static_cast<uint16_t>(0),
            (start + PSPKV_TILE - length) * dimension);
      }
    }
    pipe_barrier(PIPE_ALL);
    const uint32_t destination =
        ((seq * max_blocks + logical_block) * block_size + token) * heads *
            dimension +
        head * dimension;
    AscendC::DataCopyParams copy{
        PSPKV_TILE, static_cast<uint16_t>(dimension * 2 / 32), 0,
        static_cast<uint16_t>((heads - 1) * dimension * 2 / 32)};
    AscendC::DataCopy((is_key ? key_out : value_out)[destination], local, copy);
    pipe_barrier(PIPE_ALL);
  }
}

template <bool Profile>
__global__ __vector__ void PspkvTileFusedAttention(
    GM_ADDR query, GM_ADDR key_cache, GM_ADDR value_cache, GM_ADDR block_tables,
    GM_ADDR context_lens, GM_ADDR formats, GM_ADDR offsets, GM_ADDR scales,
    GM_ADDR e4, GM_ADDR e5, GM_ADDR permutation, GM_ADDR output,
    GM_ADDR statistics, uint32_t seqs, uint32_t heads, uint32_t kv_heads,
    uint32_t dimension, uint32_t block_size, uint32_t max_blocks,
    uint32_t generation, float softmax_scale, GM_ADDR split_partials,
    uint32_t splits, uint32_t pair_lut) {
  AscendC::GlobalTensor<uint8_t> keys, values, codes;
  AscendC::GlobalTensor<int32_t> tables, lengths, byte_offsets;
  AscendC::GlobalTensor<float> per_head_scales;
  AscendC::GlobalTensor<bfloat16_t> q, out;
  AscendC::GlobalTensor<float> split_output;
  AscendC::GlobalTensor<int64_t> stats;
  keys.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(key_cache));
  values.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(value_cache));
  codes.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(formats));
  tables.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(block_tables));
  lengths.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(context_lens));
  byte_offsets.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(offsets));
  per_head_scales.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(scales));
  q.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(query));
  split_output.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(split_partials));
  out.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(output));
  if constexpr (Profile)
    stats.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(statistics));
  AscendC::TPipe pipe;
  PspkvTileDecoder decoder;
  decoder.Init(pipe, dimension, e4, e5, permutation, pair_lut);
  AscendC::TBuf<AscendC::TPosition::VECCALC> queryBuffer, queryFloatBuffer,
      accumulatorBuffer, partialBuffer, scoreBuffer, temporaryBuffer,
      reductionBuffer, offsetBuffer;
  pipe.InitBuffer(queryBuffer, dimension * 2);
  pipe.InitBuffer(queryFloatBuffer, dimension * 4);
  pipe.InitBuffer(accumulatorBuffer, dimension * 4);
  pipe.InitBuffer(partialBuffer, PSPKV_TILE * dimension / 64 * 4);
  pipe.InitBuffer(scoreBuffer, PSPKV_TILE * 4);
  pipe.InitBuffer(temporaryBuffer, PSPKV_TILE * 4);
  pipe.InitBuffer(reductionBuffer, 64);
  pipe.InitBuffer(offsetBuffer, PSPKV_TILE * 4);
  auto qLocal = queryBuffer.Get<bfloat16_t>();
  auto qFloat = queryFloatBuffer.Get<float>();
  auto accumulator = accumulatorBuffer.Get<float>();
  auto partials = partialBuffer.Get<float>();
  auto scores = scoreBuffer.Get<float>();
  auto temporary = temporaryBuffer.Get<float>();
  auto reductions = reductionBuffer.Get<float>();
  auto partialOffsets = offsetBuffer.Get<uint32_t>();
  for (uint32_t i = 0; i < PSPKV_TILE; ++i) {
    partialOffsets.SetValue(i, i * (dimension / 64) * 4);
  }
  pipe_barrier(PIPE_ALL);
  for (uint32_t job = AscendC::GetBlockIdx(); job < seqs * heads * splits;
       job += AscendC::GetBlockNum()) {
    const uint32_t row = job / splits, split = job % splits;
    const uint32_t seq = row / heads;
    const uint32_t kv_head = (row % heads) / (heads / kv_heads);
    const uint32_t length = lengths.GetValue(seq);
    int64_t decode_cycles = 0, qk_cycles = 0, softmax_cycles = 0, pv_cycles = 0;
    int64_t begin = 0, stamp = 0;
    if constexpr (Profile) begin = AscendC::GetSystemCycle();
    AscendC::DataCopy(qLocal, q[row * dimension], dimension);
    pipe_barrier(PIPE_ALL);
    AscendC::Cast(qFloat, qLocal, AscendC::RoundMode::CAST_NONE, dimension);
    AscendC::Duplicate(accumulator, 0.0f, dimension);
    pipe_barrier(PIPE_V);
    float maximum = -3.402823466e38f, denominator = 0.0f;
    const uint32_t tile_count = (length + PSPKV_TILE - 1) / PSPKV_TILE;
    const uint32_t span = (tile_count + splits - 1) / splits * PSPKV_TILE;
    const uint32_t split_end =
        (split + 1) * span < length ? (split + 1) * span : length;
    for (uint32_t start = split * span; start < split_end;
         start += PSPKV_TILE) {
      if constexpr (Profile) {
        pipe_barrier(PIPE_ALL);
        stamp = AscendC::GetSystemCycle();
      }
      const uint32_t physical =
          tables.GetValue(seq * max_blocks + start / block_size);
      const uint8_t code = codes.GetValue(physical);
      const uint32_t base =
          byte_offsets.GetValue(physical) +
          kv_head * PackedBytesPerHead(code, block_size, dimension, 0);
      const float scale =
          code == 0 || generation == 0
              ? 1.0f
              : per_head_scales.GetValue(physical * kv_heads + kv_head);
      const uint32_t valid =
          length - start < PSPKV_TILE ? length - start : PSPKV_TILE;
      auto key = decoder.Load(keys, base, code, scale, start % block_size,
                              block_size, false);
      // Unscaled E4M3/E5M2 (and BF16/FP4 fallback values) are exactly
      // representable in BF16. Only a non-unit scale needs BF16 rounding to
      // match the materialized/native attention path.
      if (scale != 1.0f) decoder.RoundInPlace(key);
      if constexpr (Profile) {
        pipe_barrier(PIPE_ALL);
        auto now = AscendC::GetSystemCycle();
        decode_cycles += now - stamp;
        stamp = now;
      }
      decoder.DotProducts(qFloat, key, partials);
      AscendC::Duplicate(scores, -3.402823466e38f, PSPKV_TILE);
      pipe_barrier(PIPE_V);
      AscendC::Gather(scores, partials, partialOffsets, 0, valid);
      pipe_barrier(PIPE_V);
      for (uint32_t part = 1; part < dimension / 64; ++part) {
        AscendC::Gather(temporary, partials, partialOffsets, part * 4, valid);
        pipe_barrier(PIPE_V);
        AscendC::Add(scores, scores, temporary, valid);
        pipe_barrier(PIPE_V);
      }
      AscendC::Muls(scores, scores, softmax_scale, valid);
      pipe_barrier(PIPE_V);
      if constexpr (Profile) {
        pipe_barrier(PIPE_ALL);
        auto now = AscendC::GetSystemCycle();
        qk_cycles += now - stamp;
        stamp = now;
      }
      AscendC::WholeReduceMax(reductions, scores, PSPKV_TILE, 1, 1, 1, 8,
                              AscendC::ReduceOrder::ORDER_ONLY_VALUE);
      pipe_barrier(PIPE_ALL);
      const float tile_max = reductions.GetValue(0);
      const float next_max = tile_max > maximum ? tile_max : maximum;
      reductions.SetValue(8, maximum - next_max);
      pipe_barrier(PIPE_ALL);
      AscendC::Adds(scores, scores, -next_max, PSPKV_TILE);
      pipe_barrier(PIPE_V);
      AscendC::Exp(scores, scores, PSPKV_TILE);
      AscendC::Exp(reductions[8], reductions[8], 1);
      pipe_barrier(PIPE_V);
      AscendC::WholeReduceSum(reductions, scores, PSPKV_TILE, 1, 1, 1, 8);
      pipe_barrier(PIPE_ALL);
      const float alpha = reductions.GetValue(8);
      denominator = denominator * alpha + reductions.GetValue(0);
      maximum = next_max;
      if constexpr (Profile) {
        auto now = AscendC::GetSystemCycle();
        softmax_cycles += now - stamp;
        stamp = now;
      }
      const bool transposed_v = generation == 1 || generation == 2;
      const bool direct_v = transposed_v && valid == PSPKV_TILE;
      auto value = decoder.Load(values, base, code, scale, start % block_size,
                                block_size, transposed_v, !direct_v);
      if (scale != 1.0f) decoder.RoundInPlace(value);
      if constexpr (Profile) {
        pipe_barrier(PIPE_ALL);
        auto now = AscendC::GetSystemCycle();
        decode_cycles += now - stamp;
        stamp = now;
      }
      // Explicitly zero padding: an unwritten cache byte can decode to
      // NaN/Inf, and multiplying it by a zero probability is not a mask.
      if (valid < PSPKV_TILE) {
        AscendC::Duplicate(value[valid * dimension], 0.0f,
                           (PSPKV_TILE - valid) * dimension);
        pipe_barrier(PIPE_V);
      }
      if (direct_v)
        decoder.WeightedSumTransposed(value, scores, accumulator, alpha);
      else
        decoder.WeightedSum(value, scores, accumulator, alpha, transposed_v);
      if constexpr (Profile) {
        pipe_barrier(PIPE_ALL);
        pv_cycles += AscendC::GetSystemCycle() - stamp;
      }
    }
    if (splits > 1) {
      const uint32_t base = job * (dimension + 16);
      // The last PV accumulation is a Vector write. MTE3 must not read it
      // until that pipeline completes; profiling used to hide this hazard.
      pipe_barrier(PIPE_ALL);
      AscendC::DataCopy(split_output[base + 16], accumulator, dimension);
      reductions.SetValue(0, maximum);
      reductions.SetValue(1, denominator);
      pipe_barrier(PIPE_ALL);
      AscendC::DataCopy(split_output[base], reductions, 16);
      pipe_barrier(PIPE_ALL);
    } else {
      AscendC::Muls(accumulator, accumulator,
                    denominator == 0.0f ? 0.0f : 1.0f / denominator, dimension);
      pipe_barrier(PIPE_V);
      AscendC::Cast(qLocal, accumulator, AscendC::RoundMode::CAST_RINT,
                    dimension);
      pipe_barrier(PIPE_ALL);
      AscendC::DataCopy(out[row * dimension], qLocal, dimension);
      pipe_barrier(PIPE_ALL);
    }
    if constexpr (Profile) {
      // One full cache line per head prevents scalar GM write races.
      stats.SetValue(job * 8, decode_cycles);
      stats.SetValue(job * 8 + 1, qk_cycles);
      stats.SetValue(job * 8 + 2, softmax_cycles);
      stats.SetValue(job * 8 + 3, pv_cycles);
      stats.SetValue(job * 8 + 4, AscendC::GetSystemCycle() - begin);
      stats.SetValue(job * 8 + 5, 0);
      stats.SetValue(job * 8 + 6, 0);
      stats.SetValue(job * 8 + 7, 0);
    }
  }
}
