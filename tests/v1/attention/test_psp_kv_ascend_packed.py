# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Contract checks for PSP-KV's variable-stride Ascend reference storage."""

import pytest
import torch

from vllm.v1.attention.backends.psp_kv_ascend_diagnostics import FP4LayerDiagnostics
from vllm.v1.attention.backends.psp_kv_ascend_packed import (
    PackedKVCache,
    reference_packed_attention,
)
from vllm.v1.attention.backends.psp_kv_ascend_reference import (
    decode_e2m1,
    encode_e2m1,
)

pytestmark = pytest.mark.skip_global_cleanup


def test_e2m1_codes_match_psp_kv_cuda_reference():
    values = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, -0.5, -1.0, -3.0, 4.0])
    encoded = encode_e2m1(values)
    assert encoded.tolist() == [0, 1, 2, 3, 4, 5, 9, 10, 13, 6]
    torch.testing.assert_close(decode_e2m1(encoded[:9]), values[:9])
    assert torch.isposinf(decode_e2m1(encoded[9]))


def test_packed_offsets_nibble_order_and_physical_block_lookup():
    cache = PackedKVCache([1, 3, 2, 3], 2, 2, 4, torch.device("cpu"))
    assert cache.block_offsets.tolist() == [0, 16, 24, 40]
    assert cache.total_bytes == 48
    assert cache.dense_fp8_bytes == 64
    assert cache.k_buffer.numel() + cache.v_buffer.numel() == 96

    slots = torch.tensor([6, 7, 0, 1, 2, 3, 4, 5])
    rows = (
        torch.tensor(
            [
                [0.5, 1.0, 1.5, 2.0],
                [2.0, 1.5, 1.0, 0.5],
                [1.125] * 4,
                [1.125] * 4,
                [0.0, 0.5, 1.0, 1.5],
                [1.5, 1.0, 0.5, 0.0],
                [1.125] * 4,
                [1.125] * 4,
            ]
        )
        .unsqueeze(1)
        .expand(-1, 2, -1)
        .clone()
    )
    rows[:, 1] *= -1
    cache.write(slots, -rows, rows)

    # Even element is the low nibble, matching psp_packed_quantize.
    assert cache.v_buffer[40:42].tolist() == [0x21, 0x43]
    keys, values = cache.gather([3, 0, 2, 1])
    assert keys.shape == (4, 2, 2, 4)
    torch.testing.assert_close(values[0, 0, 0], rows[0, 0])
    torch.testing.assert_close(values[0, 0, 1], rows[0, 1])
    torch.testing.assert_close(values[1, 0, 0], rows[2, 0])
    torch.testing.assert_close(values[2, 0, 0], torch.ones(4))
    torch.testing.assert_close(values[3, 0, 0], rows[4, 0])
    torch.testing.assert_close(keys[0, 0, 0], -rows[0, 0])


def test_transposed_value_bytes_match_enhanced_reader_indexing():
    cache = PackedKVCache([3], 2, 1, 4, torch.device("cpu"), value_layout="transposed")
    rows = torch.tensor([[0.5, 1.0, 1.5, 2.0], [2.0, 1.5, 1.0, 0.5]]).unsqueeze(1)
    cache.write(torch.tensor([0, 1]), rows, rows)

    assert cache.k_buffer.tolist() == [0x21, 0x43, 0x34, 0x12]
    assert cache.v_buffer.tolist() == [0x41, 0x32, 0x23, 0x14]
    keys, values = cache.gather([0])
    torch.testing.assert_close(keys[0, :, 0], rows[:, 0])
    torch.testing.assert_close(values[0, :, 0], rows[:, 0])


@pytest.mark.parametrize("code", [0, 1, 2, 3])
def test_transposed_values_survive_out_of_order_writes_across_heads(code):
    cache = PackedKVCache(
        [code], 3, 2, 4, torch.device("cpu"), value_layout="transposed"
    )
    slots = torch.tensor([2, 0, 1])
    rows = torch.tensor(
        [
            [[0.5, 1.0, 1.5, 2.0], [-0.5, -1.0, -1.5, -2.0]],
            [[2.0, 1.5, 1.0, 0.5], [-2.0, -1.5, -1.0, -0.5]],
            [[1.0, 2.0, 0.5, 1.5], [-1.0, -2.0, -0.5, -1.5]],
        ]
    )
    cache.write(slots, rows, rows)
    expected = torch.empty_like(rows)
    expected[slots] = rows

    keys, values = cache.gather([0])
    torch.testing.assert_close(keys[0], expected)
    torch.testing.assert_close(values[0], expected)
    assert not torch.equal(cache.k_buffer, cache.v_buffer)


def test_packed_block_copy_snapshots_sources_and_rejects_format_mismatch():
    cache = PackedKVCache(
        [1, 3, 1, 3], 2, 1, 4, torch.device("cpu"), value_layout="transposed"
    )
    rows = torch.tensor(
        [
            [0.5, 1.0, 1.5, 2.0],
            [2.0, 1.5, 1.0, 0.5],
            [1.0, 0.5, 2.0, 1.5],
            [1.5, 2.0, 0.5, 1.0],
            [0.5, 0.5, 1.0, 1.0],
            [2.0, 2.0, 1.5, 1.5],
            [1.0, 1.5, 0.5, 2.0],
            [2.0, 0.5, 1.5, 1.0],
        ]
    ).unsqueeze(1)
    cache.write(torch.arange(8), rows, -rows)
    old_keys, old_values = cache.gather([0, 1, 2, 3])
    original_key_bytes = cache.k_buffer.clone()
    original_value_bytes = cache.v_buffer.clone()

    with pytest.raises(ValueError, match="matching formats"):
        cache.copy_blocks([(0, 1)])
    with pytest.raises(ValueError, match="unique destinations"):
        cache.copy_blocks([(0, 2), (2, 2)])
    with pytest.raises(ValueError, match="outside the arena"):
        cache.copy_blocks([(4, 0)])
    torch.testing.assert_close(cache.k_buffer, original_key_bytes)
    torch.testing.assert_close(cache.v_buffer, original_value_bytes)

    cache.copy_blocks([(0, 2), (2, 0), (1, 3), (3, 1)])
    keys, values = cache.gather([0, 1, 2, 3])
    torch.testing.assert_close(keys, old_keys[[2, 3, 0, 1]])
    torch.testing.assert_close(values, old_values[[2, 3, 0, 1]])
    assert cache.k_buffer[:8].tolist() == original_key_bytes[12:20].tolist()


def test_repack_relocates_pages_without_changing_original_arena():
    cache = PackedKVCache(
        [1, 1, 1, 1], 2, 1, 4, torch.device("cpu"), value_layout="transposed"
    )
    rows = torch.tensor([[0.5, 1.0, 1.5, 2.0], [2.0, 1.5, 1.0, 0.5]] * 4).unsqueeze(1)
    cache.write(torch.arange(8), rows, -rows)
    original_key_bytes = cache.k_buffer.clone()
    original_value_bytes = cache.v_buffer.clone()

    compact = cache.repack([1, 3, 1, 3])
    assert compact.block_offsets.tolist() == [0, 8, 12, 20]
    assert compact.total_bytes == 24 < cache.total_bytes == 32
    torch.testing.assert_close(compact.k_buffer[:8], original_key_bytes[:8])
    torch.testing.assert_close(compact.k_buffer[12:20], original_key_bytes[16:24])
    keys, values = compact.gather([0, 1, 2, 3])
    torch.testing.assert_close(keys.reshape(8, 1, 4), rows)
    torch.testing.assert_close(values.reshape(8, 1, 4), -rows)
    restored = compact.repack([1, 1, 1, 1])
    torch.testing.assert_close(restored.gather([0, 1, 2, 3])[0], keys)
    torch.testing.assert_close(cache.k_buffer, original_key_bytes)
    torch.testing.assert_close(cache.v_buffer, original_value_bytes)

    unsafe = PackedKVCache([1], 2, 1, 4, torch.device("cpu"))
    unsafe.write(torch.tensor([0]), torch.full((1, 1, 4), 4.0), torch.zeros((1, 1, 4)))
    before = unsafe.k_buffer.clone()
    with pytest.raises(ValueError, match="must be in"):
        unsafe.repack([3])
    torch.testing.assert_close(unsafe.k_buffer, before)


@pytest.mark.parametrize("use_sdpa", [False, True])
def test_uniform_fp4_attention_promotes_scale_and_reuses_block(use_sdpa):
    cache = PackedKVCache([3, 3], 4, 1, 4, torch.device("cpu"))
    scales = torch.ones((2, 1), dtype=torch.float32)
    block_tables = torch.tensor([[1, 0]], dtype=torch.int32)
    initial_key = torch.tensor([[[0.5, 1.0, 1.5, 2.0]], [[1.0, 1.0, 1.0, 1.0]]])
    initial_value = initial_key.clone()
    prefill_output = torch.empty((2, 2, 4))
    reference_packed_attention(
        torch.zeros_like(prefill_output),
        initial_key,
        initial_value,
        cache,
        torch.tensor([4, 5]),
        block_tables,
        [2],
        [2],
        prefill_output,
        0.5,
        scales,
        use_sdpa=use_sdpa,
    )
    assert scales.tolist() == [[1.0], [1.0]]
    torch.testing.assert_close(prefill_output[0, 0], initial_value[0, 0])
    torch.testing.assert_close(prefill_output[1, 0], initial_value[:, 0].mean(dim=0))

    old_bytes = cache.v_buffer.clone()
    decode_output = torch.empty((1, 2, 4))
    reference_packed_attention(
        torch.zeros_like(decode_output),
        torch.full((1, 1, 4), 4.0),
        torch.full((1, 1, 4), 4.0),
        cache,
        torch.tensor([6]),
        block_tables,
        [1],
        [3],
        decode_output,
        0.5,
        scales,
        use_sdpa=use_sdpa,
    )
    assert scales.tolist() == [[1.0], [2.0]]
    assert not torch.equal(cache.v_buffer, old_bytes)
    _, read_values = cache.gather([1])
    expected = (read_values[0, :3, 0] * 2).mean(dim=0)
    torch.testing.assert_close(decode_output[0, 0], expected)
    torch.testing.assert_close(decode_output[0, 1], expected)

    reused_output = torch.empty((1, 2, 4))
    reference_packed_attention(
        torch.zeros_like(reused_output),
        torch.ones((1, 1, 4)),
        torch.ones((1, 1, 4)),
        cache,
        torch.tensor([4]),
        block_tables,
        [1],
        [1],
        reused_output,
        0.5,
        scales,
        use_sdpa=use_sdpa,
    )
    assert scales.tolist() == [[1.0], [0.5]]
    torch.testing.assert_close(reused_output[0, 0], torch.ones(4))


def test_uniform_fp4_small_values_use_page_range_and_promote_on_decode():
    cache = PackedKVCache([3], 4, 1, 4, torch.device("cpu"))
    scales = torch.ones((1, 1), dtype=torch.float32)
    blocks = torch.tensor([[0]], dtype=torch.int32)
    small = torch.full((1, 1, 4), 0.125)
    output = torch.empty((1, 1, 4))
    reference_packed_attention(
        torch.zeros_like(output),
        small,
        small,
        cache,
        torch.tensor([0]),
        blocks,
        [1],
        [1],
        output,
        0.5,
        scales,
    )
    assert scales.item() == 0.0625
    assert cache.gather([0])[0][0, 0, 0, 0].item() == 2.0

    large = torch.full((1, 1, 4), 0.75)
    reference_packed_attention(
        torch.zeros_like(output),
        large,
        large,
        cache,
        torch.tensor([1]),
        blocks,
        [1],
        [2],
        output,
        0.5,
        scales,
    )
    assert scales.item() == 0.25
    cached_values = cache.gather([0])[1][0, :2, 0, 0] * scales.item()
    torch.testing.assert_close(cached_values, torch.tensor([0.125, 0.75]))
    torch.testing.assert_close(output[0, 0], torch.full((4,), 0.4375))


def test_uniform_fp4_batch_scatter_keeps_head_and_block_order():
    cache = PackedKVCache([3, 3], 2, 2, 4, torch.device("cpu"))
    slots = torch.tensor([3, 0, 2, 1])
    rows = torch.tensor(
        [
            [[0.5, 1.0, 1.5, 2.0], [-0.5, -1.0, -1.5, -2.0]],
            [[2.0, 1.5, 1.0, 0.5], [-2.0, -1.5, -1.0, -0.5]],
            [[1.0, 2.0, 0.5, 1.5], [-1.0, -2.0, -0.5, -1.5]],
            [[1.5, 2.0, 0.5, 1.0], [-1.5, -2.0, -0.5, -1.0]],
        ]
    )
    cache.write(slots, rows, -rows)
    keys, values = cache.gather([1, 0])
    expected = torch.empty_like(rows)
    expected[slots] = rows
    torch.testing.assert_close(keys.reshape(4, 2, 4), expected[[2, 3, 0, 1]])
    torch.testing.assert_close(values, -keys)

    with pytest.raises(ValueError, match="unique slots"):
        cache.write(torch.tensor([1, 1]), rows[:2], -rows[:2])


def test_packed_rejects_unsafe_fp4_values_and_insufficient_arena():
    with pytest.raises(ValueError, match="even head"):
        PackedKVCache([3], 2, 1, 3, torch.device("cpu"))

    cache = PackedKVCache([3], 2, 1, 4, torch.device("cpu"))
    with pytest.raises(ValueError, match="must be in"):
        cache.write(
            torch.tensor([0]), torch.full((1, 1, 4), 4.0), torch.zeros((1, 1, 4))
        )

    with pytest.raises(ValueError, match="too small"):
        PackedKVCache(
            [1, 3],
            2,
            1,
            4,
            torch.device("cpu"),
            k_buffer=torch.zeros(10, dtype=torch.uint8),
        )
    with pytest.raises(ValueError, match="value layout"):
        PackedKVCache([1], 2, 1, 4, torch.device("cpu"), value_layout="invalid")

    cache.precision_table[0] = 1
    with pytest.raises(ValueError, match="relocation"):
        cache.gather([0])


@pytest.mark.parametrize("scalar_dtype", [torch.bfloat16, torch.float16])
def test_packed_lossless_block_and_variable_offsets(scalar_dtype):
    cache = PackedKVCache(
        [0, 3, 3, 3, 1],
        1,
        1,
        4,
        torch.device("cpu"),
        scalar_dtype=scalar_dtype,
    )
    assert cache.block_offsets.tolist() == [0, 8, 10, 12, 14]
    assert cache.total_bytes == 18 < cache.dense_fp8_bytes == 20

    rows = torch.tensor(
        [[1.125] * 4, [1.5] * 4, [2.0] * 4, [0.5] * 4, [1.125] * 4]
    ).unsqueeze(1)
    cache.write(torch.arange(5), rows, -rows)
    keys, values = cache.gather([0, 3, 4])
    torch.testing.assert_close(keys[0, 0, 0], rows[0, 0])
    torch.testing.assert_close(values[0, 0, 0], -rows[0, 0])
    torch.testing.assert_close(keys[1, 0, 0], rows[3, 0])
    torch.testing.assert_close(keys[2, 0, 0], rows[4, 0])


def test_fp4_layer_diagnostics_find_decode_error_against_fp8():
    cache = PackedKVCache([3], 4, 1, 4, torch.device("cpu"))
    scales = torch.ones((1, 1), dtype=torch.float32)
    diagnostics = FP4LayerDiagnostics(max_decode_steps=1)
    query = torch.zeros((1, 1, 4))
    small = torch.full((1, 1, 4), 0.125)
    large = torch.full((1, 1, 4), 2.0)
    table = torch.tensor([0], dtype=torch.int32)
    cache.write(torch.tensor([0]), small, small)
    assert (
        diagnostics.observe(query, small, small, small, cache, scales, table, 1, 0.5)
        is None
    )

    cache.write(torch.tensor([1]), large, large)
    comparison = diagnostics.observe(
        query,
        large,
        large,
        torch.ones_like(large),
        cache,
        scales,
        table,
        2,
        0.5,
    )
    assert comparison is not None
    assert comparison.sequence_length == 2
    assert comparison.fp4_key_rmse > comparison.fp8_key_rmse
    assert comparison.fp4_value_rmse > comparison.fp8_value_rmse
    assert comparison.attention_rmse > 0
    assert comparison.fp4_value_zero_fraction == 0.5
    assert (
        diagnostics.observe(
            query, large, large, torch.ones_like(large), cache, scales, table, 3, 0.5
        )
        is None
    )


@pytest.mark.parametrize("use_sdpa", [False, True])
def test_grouped_fp4_scales_preserve_small_values_during_key_promotion(use_sdpa):
    cache = PackedKVCache([3], 4, 1, 4, torch.device("cpu"))
    # [physical block, K/V, KV head, two 2-dimensional groups]
    scales = torch.ones((1, 2, 1, 2), dtype=torch.float32)
    blocks = torch.tensor([[0]], dtype=torch.int32)
    query = torch.zeros((1, 1, 4))
    first_key = torch.tensor([[[16.0, 16.0, 0.25, 0.25]]])
    second_key = torch.tensor([[[32.0, 32.0, 0.25, 0.25]]])
    small_value = torch.full((1, 1, 4), 0.125)
    output = torch.empty_like(query)

    reference_packed_attention(
        query,
        first_key,
        small_value,
        cache,
        torch.tensor([0]),
        blocks,
        [1],
        [1],
        output,
        0.5,
        scales,
        use_sdpa=use_sdpa,
    )
    assert scales[0, 0, 0, 0] == 8.0
    assert scales[0, 1, 0, 0] == 0.0625
    reference_packed_attention(
        query,
        second_key,
        small_value,
        cache,
        torch.tensor([1]),
        blocks,
        [1],
        [2],
        output,
        0.5,
        scales,
        use_sdpa=use_sdpa,
    )
    assert scales[0, 0, 0, 0] == 16.0
    assert scales[0, 1, 0, 0] == 0.0625
    _, decoded_values = cache.gather([0])
    decoded_values *= scales[0, 1, :, :].repeat_interleave(2, dim=-1)[None, None]
    torch.testing.assert_close(decoded_values[0, :2], small_value.expand(2, -1, -1))
    torch.testing.assert_close(output, small_value)
