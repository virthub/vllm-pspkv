// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#pragma once

// Single-token NeoX RoPE + FP8 write. Q is rotated in place, K is rounded
// to BF16 in UB and encoded directly into its page without a GM K round trip.
// Each Q/K head and each transposed V dimension row has a unique owner.
__global__ __vector__ void PspkvRopeScatter(
    GM_ADDR query, GM_ADDR key, GM_ADDR value, GM_ADDR positions,
    GM_ADDR cos_sin, GM_ADDR key_cache, GM_ADDR value_cache, GM_ADDR slots,
    GM_ADDR formats, GM_ADDR offsets, uint32_t q_heads, uint32_t kv_heads,
    uint32_t dimension, uint32_t block_size, uint32_t generation,
    uint32_t cos_float) {
  AscendC::GlobalTensor<bfloat16_t> q, k;
  AscendC::GlobalTensor<uint8_t> v, keys, values, codes;
  AscendC::GlobalTensor<int64_t> position, slot;
  AscendC::GlobalTensor<int32_t> byte_offsets;
  q.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(query));
  k.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(key));
  v.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(value));
  keys.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(key_cache));
  values.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(value_cache));
  codes.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(formats));
  position.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(positions));
  slot.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(slots));
  byte_offsets.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(offsets));
  AscendC::TPipe pipe;
  AscendC::TBuf<AscendC::TPosition::VECCALC> raw_buffer, cos_raw_buffer,
      input_buffer, cos_buffer, temp_buffer, result_buffer;
  pipe.InitBuffer(raw_buffer, dimension * 2);
  pipe.InitBuffer(cos_raw_buffer, dimension * 2);
  pipe.InitBuffer(input_buffer, dimension * 4);
  pipe.InitBuffer(cos_buffer, dimension * 4);
  pipe.InitBuffer(temp_buffer, dimension * 4);
  pipe.InitBuffer(result_buffer, dimension * 4);
  auto raw = raw_buffer.Get<bfloat16_t>();
  auto cos_raw = cos_raw_buffer.Get<bfloat16_t>();
  auto input = input_buffer.Get<float>();
  auto cs = cos_buffer.Get<float>();
  auto temp = temp_buffer.Get<float>();
  auto result = result_buffer.Get<float>();
  const uint32_t half_dim = dimension / 2;
  const uint32_t value_jobs = generation == 0 ? kv_heads : kv_heads * dimension;
  const int64_t current_slot = slot.GetValue(0);
  const uint32_t physical = current_slot < 0 ? 0 : current_slot / block_size;
  const uint32_t within = current_slot < 0 ? 0 : current_slot % block_size;
  const uint32_t cache_base = byte_offsets.GetValue(physical);
  const uint8_t code = codes.GetValue(physical);
  bool loaded_cos = false;
  for (uint32_t job = AscendC::GetBlockIdx();
       job < q_heads + kv_heads + value_jobs; job += AscendC::GetBlockNum()) {
    if (job < q_heads + kv_heads) {
      const bool is_query = job < q_heads;
      if (!is_query && current_slot < 0) continue;
      const uint32_t head = is_query ? job : job - q_heads;
      if (!loaded_cos) {
        const uint32_t pos = position.GetValue(0);
        if (cos_float) {
          AscendC::GlobalTensor<float> cache;
          cache.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(cos_sin));
          AscendC::DataCopy(cs, cache[pos * dimension], dimension);
        } else {
          AscendC::GlobalTensor<bfloat16_t> cache;
          cache.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(cos_sin));
          AscendC::DataCopy(cos_raw, cache[pos * dimension], dimension);
          pipe_barrier(PIPE_ALL);
          AscendC::Cast(cs, cos_raw, AscendC::RoundMode::CAST_NONE, dimension);
        }
        loaded_cos = true;
      }
      AscendC::DataCopy(raw, (is_query ? q : k)[head * dimension], dimension);
      pipe_barrier(PIPE_ALL);
      AscendC::Cast(input, raw, AscendC::RoundMode::CAST_NONE, dimension);
      pipe_barrier(PIPE_V);
      AscendC::Mul(temp, input, cs, half_dim);
      AscendC::Mul(temp[half_dim], input[half_dim], cs[half_dim], half_dim);
      pipe_barrier(PIPE_V);
      AscendC::Sub(result, temp, temp[half_dim], half_dim);
      pipe_barrier(PIPE_V);
      AscendC::Mul(temp, input[half_dim], cs, half_dim);
      AscendC::Mul(temp[half_dim], input, cs[half_dim], half_dim);
      pipe_barrier(PIPE_V);
      AscendC::Add(result[half_dim], temp, temp[half_dim], half_dim);
      pipe_barrier(PIPE_V);
      AscendC::Cast(raw, result, AscendC::RoundMode::CAST_RINT, dimension);
      pipe_barrier(PIPE_ALL);
      if (is_query) {
        AscendC::DataCopy(q[head * dimension], raw, dimension);
      } else {
        const uint32_t base = cache_base + head * block_size * dimension;
        auto bits = raw.ReinterpretCast<uint16_t>();
        for (uint32_t d = 0; d < dimension; ++d) {
          const uint32_t fp32 = static_cast<uint32_t>(bits.GetValue(d)) << 16;
          keys.SetValue(base + within * dimension + d,
                        code == 2 ? EncodeE5M2(fp32) : EncodeE4M3(fp32));
        }
      }
      pipe_barrier(PIPE_ALL);
    } else if (current_slot >= 0) {
      const uint32_t local = job - q_heads - kv_heads;
      const uint32_t head = generation == 0 ? local : local / dimension;
      const uint32_t base = cache_base + head * block_size * dimension;
      const uint32_t count = generation == 0 ? dimension : 1;
      for (uint32_t i = 0; i < count; ++i) {
        const uint32_t d = generation == 0 ? i : local % dimension;
        const uint32_t element =
            generation == 0 ? within * dimension + d : d * block_size + within;
        WritePacked(v, values, head * dimension + d, base, element, code, 0,
                    1.0f);
      }
    }
  }
}
