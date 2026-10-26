// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

// Included after the reference codecs in pspkv_fp8.asc. All working storage is
// private to one AIV. Tables and the transpose map are loaded once, not per row.
constexpr uint32_t PSPKV_TILE = 32;

class PspkvTileDecoder {
public:
    __aicore__ inline void Init(AscendC::TPipe& pipe, uint32_t dim,
                               GM_ADDR e4, GM_ADDR e5, GM_ADDR permutation) {
        dimension = dim;
        count = PSPKV_TILE * dim;
        pipe.InitBuffer(inputBuffer, count);
        pipe.InitBuffer(halfBuffer, count * 2);
        pipe.InitBuffer(indexBuffer, count * 4);
        pipe.InitBuffer(decodedBuffer, count * 4);
        pipe.InitBuffer(transposedBuffer, count * 4);
        pipe.InitBuffer(permutationBuffer, count * 4);
        pipe.InitBuffer(tableBuffer, 512 * 4);
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
        permutationGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t*>(permutation));
        AscendC::DataCopy(table, e4Global, 256);
        AscendC::DataCopy(table[256], e5Global, 256);
        AscendC::DataCopy(permutationLocal, permutationGlobal, count);
        pipe_barrier(PIPE_ALL);
    }

    __aicore__ inline AscendC::LocalTensor<float> Load(
        AscendC::GlobalTensor<uint8_t>& source, uint32_t base, uint8_t code,
        float scale, uint32_t token, uint32_t block_size, bool transpose) {
        if (code != 1 && code != 2) {
            // Preserve mixed BF16/FP4 contracts; the FP8 hot path below contains
            // no per-element scalar loads, conversions or stores.
            for (uint32_t row = 0; row < PSPKV_TILE; ++row) {
                for (uint32_t dim = 0; dim < dimension; ++dim) {
                    const uint32_t element = transpose
                        ? dim * block_size + token + row
                        : (token + row) * dimension + dim;
                    decoded.SetValue(row * dimension + dim,
                                     ReadPacked(source, base, element, code, 0, scale));
                }
            }
            pipe_barrier(PIPE_ALL);
            return decoded;
        }
        if (transpose) {
            AscendC::DataCopyParams copy{static_cast<uint16_t>(dimension), 1,
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
        if (transpose) {
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

    uint32_t dimension, count;
    AscendC::LocalTensor<bfloat16_t> bf16;
private:
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
    uint32_t dimension, uint32_t block_size, uint32_t generation) {
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
    decoder.Init(pipe, dimension, e4, e5, permutation);
    const uint32_t chunks = block_size / PSPKV_TILE;
    const uint32_t tasks = seqs * max_blocks * heads * chunks;
    for (uint32_t task = AscendC::GetBlockIdx(); task < 2 * tasks;
         task += AscendC::GetBlockNum()) {
        const bool is_key = task < tasks;
        uint32_t index = task % tasks;
        const uint32_t token = index % chunks * PSPKV_TILE; index /= chunks;
        const uint32_t head = index % heads; index /= heads;
        const uint32_t logical_block = index % max_blocks;
        const uint32_t seq = index / max_blocks;
        const uint32_t length = lengths.GetValue(seq);
        const uint32_t start = logical_block * block_size + token;
        auto local = decoder.bf16;
        if (start >= length) {
            AscendC::Duplicate(local.ReinterpretCast<uint16_t>(), static_cast<uint16_t>(0), decoder.count);
        } else {
            const uint32_t physical = tables.GetValue(seq * max_blocks + logical_block);
            const uint8_t code = codes.GetValue(physical);
            const uint32_t base = byte_offsets.GetValue(physical)
                + head * PackedBytesPerHead(code, block_size, dimension, 0);
            const float scale = code == 0 || generation == 0
                ? 1.0f : per_head_scales.GetValue(physical * heads + head);
            local = decoder.Round(decoder.Load(is_key ? keys : values, base,
                code, scale, token, block_size, !is_key && generation != 0));
            if (start + PSPKV_TILE > length) {
                AscendC::Duplicate(local[(length - start) * dimension].ReinterpretCast<uint16_t>(),
                    static_cast<uint16_t>(0), (start + PSPKV_TILE - length) * dimension);
            }
        }
        pipe_barrier(PIPE_ALL);
        const uint32_t destination = ((seq * max_blocks + logical_block) * block_size
                                      + token) * heads * dimension + head * dimension;
        AscendC::DataCopyParams copy{PSPKV_TILE, static_cast<uint16_t>(dimension * 2 / 32),
            0, static_cast<uint16_t>((heads - 1) * dimension * 2 / 32)};
        AscendC::DataCopy((is_key ? key_out : value_out)[destination], local, copy);
        pipe_barrier(PIPE_ALL);
    }
}
