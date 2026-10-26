// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#pragma once

// Stable online-softmax merge: partial accumulators stay FP32 until here.
__global__ __vector__ void PspkvMergeSplits(GM_ADDR partials, GM_ADDR output,
                                            uint32_t rows, uint32_t dim,
                                            uint32_t splits) {
  AscendC::GlobalTensor<float> source;
  AscendC::GlobalTensor<bfloat16_t> out;
  source.SetGlobalBuffer(reinterpret_cast<__gm__ float*>(partials));
  out.SetGlobalBuffer(reinterpret_cast<__gm__ bfloat16_t*>(output));
  AscendC::TPipe pipe;
  AscendC::TBuf<AscendC::TPosition::VECCALC> accumBuf, valueBuf, weightBuf,
      outBuf, partialBuf;
  pipe.InitBuffer(accumBuf, dim * 4);
  pipe.InitBuffer(valueBuf, dim * 4);
  pipe.InitBuffer(weightBuf, 128);
  pipe.InitBuffer(outBuf, dim * 2);
  pipe.InitBuffer(partialBuf, splits * (dim + 16) * 4);
  auto accum = accumBuf.Get<float>(), value = valueBuf.Get<float>();
  auto weights = weightBuf.Get<float>();
  auto result = outBuf.Get<bfloat16_t>();
  auto local = partialBuf.Get<float>();
  for (uint32_t row = AscendC::GetBlockIdx(); row < rows;
       row += AscendC::GetBlockNum()) {
    AscendC::DataCopy(local, source[row * splits * (dim + 16)],
                      splits * (dim + 16));
    pipe_barrier(PIPE_ALL);
    float maximum = -3.402823466e38f;
    for (uint32_t s = 0; s < splits; ++s) {
      const float current = local.GetValue(s * (dim + 16));
      maximum = current > maximum ? current : maximum;
    }
    for (uint32_t s = 0; s < splits; ++s) {
      const uint32_t base = s * (dim + 16);
      weights.SetValue(s, local.GetValue(base) - maximum);
    }
    pipe_barrier(PIPE_ALL);
    AscendC::Exp(weights, weights, splits);
    AscendC::Duplicate(accum, 0.0f, dim);
    pipe_barrier(PIPE_ALL);
    float denominator = 0.0f;
    for (uint32_t s = 0; s < splits; ++s) {
      const uint32_t base = s * (dim + 16);
      const float weight = weights.GetValue(s);
      denominator += weight * local.GetValue(base + 1);
      AscendC::Muls(value, local[base + 16], weight, dim);
      pipe_barrier(PIPE_V);
      AscendC::Add(accum, accum, value, dim);
      pipe_barrier(PIPE_V);
    }
    AscendC::Muls(accum, accum, denominator > 0.0f ? 1.0f / denominator : 0.0f,
                  dim);
    pipe_barrier(PIPE_V);
    AscendC::Cast(result, accum, AscendC::RoundMode::CAST_RINT, dim);
    pipe_barrier(PIPE_ALL);
    AscendC::DataCopy(out[row * dim], result, dim);
    pipe_barrier(PIPE_ALL);
  }
}
