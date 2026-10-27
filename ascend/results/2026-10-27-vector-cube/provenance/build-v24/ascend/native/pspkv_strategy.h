// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#pragma once

// Opt-in develop-compatible codecs. The legacy reference remains unchanged.
__aicore__ inline uint8_t StrategyEncode(float value, uint32_t code) {
  union {
    float f;
    uint32_t u;
  } bits;
  bits.f = value;
  if (code == 3) {
    const uint8_t old = EncodeE2M1(bits.u);
    return (old & 7) == 6 ? static_cast<uint8_t>((old & 8) | 5) : old;
  }
  const uint32_t sign = (bits.u >> 24) & 128;
  const uint32_t mag = bits.u & 0x7fffffffU;
  const uint32_t mantissa = code == 2 ? 2 : 3;
  const int32_t bias = code == 2 ? 15 : 7;
  const uint32_t maximum = code == 2 ? 123 : 126;
  if (mag > 0x7f800000U) return 127;
  if (mag == 0x7f800000U) return sign | maximum;
  const int32_t exponent = static_cast<int32_t>(mag >> 23) - 127 + bias;
  uint32_t encoded;
  if (exponent <= 0) {
    // Round the entire significand once, including gradual underflow.
    const int32_t shift = 24 - static_cast<int32_t>(mantissa) - exponent;
    if (shift > 31) return sign;
    const uint32_t significand = (mag & 0x7fffffU) | 0x800000U;
    const uint32_t halfway = 1U << (shift - 1);
    encoded =
        (significand + halfway - 1 + ((significand >> shift) & 1)) >> shift;
  } else {
    const uint32_t shift = 23 - mantissa;
    const uint32_t rounded =
        mag + (1U << (shift - 1)) - 1 + ((mag >> shift) & 1);
    encoded = (rounded >> shift) - ((127 - bias) << mantissa);
  }
  return sign | (encoded > maximum ? maximum : encoded);
}

// A head owns a whole packed row, avoiding FP4 read/modify/write races.
__global__ __vector__ void PspkvStrategyScatter(
    GM_ADDR key, GM_ADDR value, GM_ADDR key_cache, GM_ADDR value_cache,
    GM_ADDR slots, GM_ADDR offsets, GM_ADDR scales, uint32_t tokens,
    uint32_t heads, uint32_t dim, uint32_t block_size, uint32_t code,
    uint32_t kt, uint32_t kh, uint32_t vt, uint32_t vh, GM_ADDR formats) {
  AscendC::GlobalTensor<bfloat16_t> k, v;
  AscendC::GlobalTensor<uint8_t> ko, vo;
  AscendC::GlobalTensor<int64_t> slot;
  AscendC::GlobalTensor<int32_t> offset;
  AscendC::GlobalTensor<float> scale;
  AscendC::GlobalTensor<uint8_t> codes;
  if (code == 4)
    codes.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(formats));
  k.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(key));
  v.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(value));
  ko.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(key_cache));
  vo.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(value_cache));
  slot.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(slots));
  offset.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(offsets));
  scale.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(scales));
  AscendC::TPipe pipe;
  AscendC::TBuf<AscendC::TPosition::VECCALC> rawBuf, floatBuf, encodedBuf;
  pipe.InitBuffer(rawBuf, dim * 2);
  pipe.InitBuffer(floatBuf, dim * 4);
  pipe.InitBuffer(encodedBuf, dim * 2);
  auto raw = rawBuf.Get<bfloat16_t>();
  auto real = floatBuf.Get<float>();
  auto encoded = encodedBuf.Get<uint8_t>();
  for (uint32_t job = AscendC::GetBlockIdx(); job < tokens * heads * 2;
       job += AscendC::GetBlockNum()) {
    const uint32_t stream = job % 2, token = job / (heads * 2),
                   head = job / 2 % heads;
    const int64_t physical_slot = slot.GetValue(token);
    if (physical_slot < 0) continue;
    const uint32_t block = physical_slot / block_size,
                   within = physical_slot % block_size;
    const uint32_t row_code = code == 4 ? codes.GetValue(block) : code;
    const uint32_t row_bytes = row_code == 0   ? dim * 2
                               : row_code == 3 ? dim / 2
                                               : dim;
    const uint32_t base =
        offset.GetValue(block) + (head * block_size + within) * row_bytes;
    AscendC::DataCopy(
        raw,
        (stream ? v
                : k)[token * (stream ? vt : kt) + head * (stream ? vh : kh)],
        dim);
    pipe_barrier(PIPE_ALL);
    if (row_code == 0) {
      AscendC::DataCopy((stream ? vo : ko)[base],
                        raw.ReinterpretCast<uint8_t>(), row_bytes);
    } else {
      AscendC::Cast(real, raw, AscendC::RoundMode::CAST_NONE, dim);
      pipe_barrier(PIPE_V);
      AscendC::Muls(real, real, 1.0f / scale.GetValue(block * heads + head),
                    dim);
      pipe_barrier(PIPE_ALL);
      for (uint32_t d = 0; d < dim; d += row_code == 3 ? 2 : 1) {
        uint8_t result = StrategyEncode(real.GetValue(d), row_code);
        if (row_code == 3)
          result |= StrategyEncode(real.GetValue(d + 1), row_code) << 4;
        encoded.SetValue(row_code == 3 ? d / 2 : d, result);
      }
      pipe_barrier(PIPE_ALL);
      AscendC::DataCopy((stream ? vo : ko)[base], encoded, row_bytes);
    }
    pipe_barrier(PIPE_ALL);
  }
}

// One pass per page/head, outside decode. Same shared K/V max-abs policy as
// develop's estimate_block_scales_kernel; no host reads or synchronization.
__global__ __vector__ void PspkvStrategyScales(
    GM_ADDR key, GM_ADDR value, GM_ADDR slots, GM_ADDR scales, uint32_t tokens,
    uint32_t heads, uint32_t dim, uint32_t block_size, uint32_t code,
    uint32_t kt, uint32_t kh, uint32_t vt, uint32_t vh) {
  AscendC::GlobalTensor<bfloat16_t> k, v;
  AscendC::GlobalTensor<int64_t> slot;
  AscendC::GlobalTensor<float> scale;
  k.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(key));
  v.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(value));
  slot.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(slots));
  scale.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(scales));
  AscendC::TPipe pipe;
  AscendC::TBuf<AscendC::TPosition::VECCALC> rawBuf, floatBuf, reduceBuf,
      workBuf;
  pipe.InitBuffer(rawBuf, block_size * dim * 2);
  pipe.InitBuffer(floatBuf, block_size * dim * 4);
  pipe.InitBuffer(reduceBuf, 64);
  pipe.InitBuffer(workBuf, 2048);
  auto raw = rawBuf.Get<bfloat16_t>();
  auto real = floatBuf.Get<float>();
  auto reduced = reduceBuf.Get<float>();
  auto work = workBuf.Get<float>();
  const uint32_t pages = (tokens + block_size - 1) / block_size;
  for (uint32_t job = AscendC::GetBlockIdx(); job < pages * heads;
       job += AscendC::GetBlockNum()) {
    const uint32_t first = job / heads * block_size, head = job % heads;
    const int64_t physical_slot = slot.GetValue(first);
    if (physical_slot < 0) continue;
    const uint32_t physical = physical_slot / block_size;
    const uint32_t valid =
        tokens - first < block_size ? tokens - first : block_size;
    float maximum = 0.0f;
    for (uint32_t stream = 0; stream < 2; ++stream) {
      const uint32_t stride = stream ? vt : kt;
      AscendC::DataCopyParams copy{
          static_cast<uint16_t>(valid), static_cast<uint16_t>(dim * 2 / 32),
          static_cast<uint16_t>((stride - dim) * 2 / 32), 0};
      AscendC::DataCopy(
          raw, (stream ? v : k)[first * stride + head * (stream ? vh : kh)],
          copy);
      pipe_barrier(PIPE_ALL);
      AscendC::Cast(real, raw, AscendC::RoundMode::CAST_NONE, valid * dim);
      pipe_barrier(PIPE_V);
      AscendC::Abs(real, real, valid * dim);
      pipe_barrier(PIPE_V);
      AscendC::ReduceMax(reduced, real, work, valid * dim, false);
      pipe_barrier(PIPE_ALL);
      const float current = reduced.GetValue(0);
      maximum = current > maximum ? current : maximum;
    }
    const float limit = code == 3 ? 3.0f : code == 2 ? 57344.0f : 240.0f;
    // One cache line per job; the caller scatters this compact sample table
    // into physical page metadata with one vectorized index_copy_.
    const float normalized = maximum / limit;
    reduced.SetValue(0, normalized < 1e-8f ? 1.0f : normalized);
    reduced.SetValue(1, maximum);
    pipe_barrier(PIPE_ALL);
    AscendC::DataCopy(scale[job * 16], reduced, 16);
    pipe_barrier(PIPE_ALL);
  }
}
