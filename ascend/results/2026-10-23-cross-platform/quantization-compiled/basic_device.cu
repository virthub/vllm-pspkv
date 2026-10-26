#include "common/quantization_utils.cuh"
#define BLOCK_SIZE 16
#define MAX_KV_HEADS 8

extern "C" __global__ void pspkv_quantization_probe(
    const float* input, uint8_t* encoded, float* decoded) {
    int i = threadIdx.x;
    if (i < 16) {
        encoded[i] = quantize_fp8_e4m3(input[i]);
        decoded[i] = dequantize_fp8_e4m3(encoded[i]);
    }
}
