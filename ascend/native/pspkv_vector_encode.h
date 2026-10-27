// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#pragma once

// 910B has no native float-to-FP8 conversion. Encode the normal and gradual
// underflow paths in SIMD, including RNE ties, finite saturation and NaNs.
class PspkvVectorEncoder {
 public:
  __aicore__ inline void Init(AscendC::TPipe& pipe, uint32_t elements) {
    count = elements;
    pipe.InitBuffer(realBuf, count * 4);
    pipe.InitBuffer(magBuf, count * 4);
    pipe.InitBuffer(workBuf, count * 4);
    pipe.InitBuffer(integerBuf, count * 4);
    pipe.InitBuffer(resultBuf, count * 4);
    pipe.InitBuffer(subBuf, count * 4);
    pipe.InitBuffer(halfBuf, count * 2);
    pipe.InitBuffer(byteBuf, count);
    pipe.InitBuffer(maskBuf, (count + 255) / 256 * 32);
    real = realBuf.Get<float>();
    bytes = byteBuf.Get<uint8_t>();
  }

  __aicore__ inline void Encode(uint32_t code) {
    const uint32_t mantissa = code == 2 ? 2 : 3;
    const uint32_t bias = code == 2 ? 15 : 7;
    const uint32_t shift = 23 - mantissa;
    auto mag = magBuf.Get<uint32_t>();
    auto work = workBuf.Get<uint32_t>();
    auto integer = integerBuf.Get<int32_t>();
    auto result = resultBuf.Get<float>();
    auto sub = subBuf.Get<float>();
    auto mask = maskBuf.Get<uint8_t>();
    auto halves = halfBuf.Get<half>();
    AscendC::ShiftLeft(mag, real.ReinterpretCast<uint32_t>(),
                       static_cast<uint32_t>(1), count);
    pipe_barrier(PIPE_V);
    AscendC::ShiftRight(mag, mag, static_cast<uint32_t>(1), count);
    pipe_barrier(PIPE_V);
    AscendC::ShiftRight(work, mag, shift, count);
    pipe_barrier(PIPE_V);
    AscendC::ShiftLeft(work, work, static_cast<uint32_t>(31), count);
    pipe_barrier(PIPE_V);
    AscendC::ShiftRight(work, work, static_cast<uint32_t>(31), count);
    AscendC::Adds(integer, mag.ReinterpretCast<int32_t>(),
                  static_cast<int32_t>((1U << (shift - 1)) - 1), count);
    pipe_barrier(PIPE_V);
    AscendC::Add(integer, integer, work.ReinterpretCast<int32_t>(), count);
    pipe_barrier(PIPE_V);
    AscendC::ShiftRight(integer.ReinterpretCast<uint32_t>(),
                        integer.ReinterpretCast<uint32_t>(), shift, count);
    pipe_barrier(PIPE_V);
    AscendC::Adds(integer, integer,
                  -static_cast<int32_t>((127 - bias) << mantissa), count);
    pipe_barrier(PIPE_V);
    AscendC::Cast(result, integer, AscendC::RoundMode::CAST_NONE, count);
    // Subnormals: round the whole significand once, rather than double
    // rounding through FP16/BF16 or discarding the lowest normal exponent.
    AscendC::Muls(sub, mag.ReinterpretCast<float>(),
                  code == 2 ? 65536.0f : 512.0f, count);
    pipe_barrier(PIPE_V);
    AscendC::Cast(integer, sub, AscendC::RoundMode::CAST_RINT, count);
    pipe_barrier(PIPE_V);
    AscendC::Cast(sub, integer, AscendC::RoundMode::CAST_NONE, count);
    AscendC::Compares(mask, mag.ReinterpretCast<float>(),
                      code == 2 ? 0.00006103515625f : 0.015625f,
                      AscendC::CMPMODE::LT, count);
    pipe_barrier(PIPE_V);
    AscendC::Select(result, mask, sub, result,
                    AscendC::SELMODE::VSEL_TENSOR_TENSOR_MODE, count);
    pipe_barrier(PIPE_V);
    AscendC::Mins(result, result, code == 2 ? 123.0f : 126.0f, count);
    AscendC::ShiftRight(work, real.ReinterpretCast<uint32_t>(),
                        static_cast<uint32_t>(31), count);
    pipe_barrier(PIPE_V);
    AscendC::ShiftLeft(work, work, static_cast<uint32_t>(7), count);
    pipe_barrier(PIPE_V);
    AscendC::Cast(sub, work.ReinterpretCast<int32_t>(),
                  AscendC::RoundMode::CAST_NONE, count);
    pipe_barrier(PIPE_V);
    AscendC::Add(result, result, sub, count);
    AscendC::Compare(mask, mag.ReinterpretCast<float>(),
                     mag.ReinterpretCast<float>(), AscendC::CMPMODE::EQ, count);
    pipe_barrier(PIPE_V);
    AscendC::Select(result, mask, result, 127.0f,
                    AscendC::SELMODE::VSEL_TENSOR_SCALAR_MODE, count);
    pipe_barrier(PIPE_V);
    AscendC::Cast(halves, result, AscendC::RoundMode::CAST_NONE, count);
    pipe_barrier(PIPE_V);
    AscendC::Cast(bytes, halves, AscendC::RoundMode::CAST_RINT, count);
    pipe_barrier(PIPE_ALL);
  }

  AscendC::LocalTensor<float> real;
  AscendC::LocalTensor<uint8_t> bytes;

 private:
  uint32_t count;
  AscendC::TBuf<AscendC::TPosition::VECCALC> realBuf, magBuf, workBuf,
      integerBuf, resultBuf, subBuf, halfBuf, byteBuf, maskBuf;
};

// Logical 16-token tiles never cross a 128-token physical page. Source K/V
// retain their projection strides; destination rows are owned by one block.
__global__ __vector__ void PspkvPrefillScatterVec(
    GM_ADDR key, GM_ADDR value, GM_ADDR key_cache, GM_ADDR value_cache,
    GM_ADDR slots, GM_ADDR offsets, GM_ADDR scales, uint32_t tokens,
    uint32_t heads, uint32_t code, uint32_t kt, uint32_t kh, uint32_t vt,
    uint32_t vh) {
  constexpr uint32_t rows = 16, dim = 128, count = rows * dim;
  AscendC::GlobalTensor<bfloat16_t> k, v;
  AscendC::GlobalTensor<uint8_t> ko, vo;
  AscendC::GlobalTensor<int64_t> slot;
  AscendC::GlobalTensor<int32_t> offset;
  AscendC::GlobalTensor<float> scale;
  k.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(key));
  v.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(value));
  ko.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(key_cache));
  vo.SetGlobalBuffer(reinterpret_cast<__gm__ uint8_t*>(value_cache));
  slot.SetGlobalBuffer(reinterpret_cast<__gm__ int64_t*>(slots));
  offset.SetGlobalBuffer(reinterpret_cast<__gm__ int32_t*>(offsets));
  scale.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(scales));
  AscendC::TPipe pipe;
  AscendC::TBuf<AscendC::TPosition::VECCALC> rawBuf;
  pipe.InitBuffer(rawBuf, count * 2);
  auto raw = rawBuf.Get<bfloat16_t>();
  PspkvVectorEncoder encoder;
  encoder.Init(pipe, count);
  const uint32_t tiles = (tokens + rows - 1) / rows;
  for (uint32_t job = AscendC::GetBlockIdx(); job < tiles * heads * 2;
       job += AscendC::GetBlockNum()) {
    const uint32_t stream = job % 2, head = job / 2 % heads;
    const uint32_t first = job / (heads * 2) * rows;
    const uint32_t valid = tokens - first < rows ? tokens - first : rows;
    const int64_t position = slot.GetValue(first);
    if (position < 0) continue;
    const uint32_t page = position / 128, within = position % 128;
    if (valid < rows) {
      AscendC::Duplicate(raw.ReinterpretCast<uint16_t>(),
                         static_cast<uint16_t>(0), count);
      pipe_barrier(PIPE_ALL);
    }
    const uint32_t stride = stream ? vt : kt;
    AscendC::DataCopyParams copy{static_cast<uint16_t>(valid), 8,
                                 static_cast<uint16_t>((stride - dim) / 16), 0};
    AscendC::DataCopy(
        raw, (stream ? v : k)[first * stride + head * (stream ? vh : kh)],
        copy);
    pipe_barrier(PIPE_ALL);
    const uint32_t row_bytes = code == 0 ? 256 : 128;
    const uint32_t base =
        offset.GetValue(page) + (head * 128 + within) * row_bytes;
    if (code == 0) {
      AscendC::DataCopy((stream ? vo : ko)[base],
                        raw.ReinterpretCast<uint8_t>(), valid * 256);
    } else {
      AscendC::Cast(encoder.real, raw, AscendC::RoundMode::CAST_NONE, count);
      pipe_barrier(PIPE_V);
      AscendC::Muls(encoder.real, encoder.real,
                    1.0f / scale.GetValue(page * heads + head), count);
      pipe_barrier(PIPE_V);
      encoder.Encode(code);
      AscendC::DataCopy((stream ? vo : ko)[base], encoder.bytes, valid * dim);
    }
    pipe_barrier(PIPE_ALL);
  }
}
