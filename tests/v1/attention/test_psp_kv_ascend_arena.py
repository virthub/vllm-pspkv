# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CPU checks for the online variable-width PSP-KV Ascend arena."""

import pytest
import torch

from vllm.v1.attention.backends.psp_kv_ascend_arena import (
    VariableArenaKVCache,
    live_block_ids,
    reference_variable_attention,
)
from vllm.v1.attention.backends.psp_kv_ascend_reference import (
    decode_e2m1,
    decode_e4m3,
    encode_e2m1,
)

pytestmark = pytest.mark.skip_global_cleanup

BLOCK = 4
HEADS = 2
DIM = 8
FP8_BLOCK_BYTES = BLOCK * HEADS * DIM  # 64


def _arena(
    num_blocks: int = 3,
    avg_bytes: float = 1.0,
    fp4_group_size: int = 0,
    max_relative_rmse: float = 0.05,
    scalar_dtype: torch.dtype = torch.bfloat16,
    static_code: int | None = None,
    static_scales: torch.Tensor | None = None,
    static_v_code: int | None = None,
) -> VariableArenaKVCache:
    size = num_blocks * FP8_BLOCK_BYTES * avg_bytes
    return VariableArenaKVCache(
        num_blocks=num_blocks,
        block_size=BLOCK,
        num_kv_heads=HEADS,
        head_size=DIM,
        device=torch.device("cpu"),
        k_buffer=torch.zeros(int(size), dtype=torch.uint8),
        v_buffer=torch.zeros(int(size), dtype=torch.uint8),
        avg_bytes=avg_bytes,
        scalar_dtype=scalar_dtype,
        fp4_group_size=fp4_group_size,
        max_relative_rmse=max_relative_rmse,
        static_code=static_code,
        static_scales=static_scales,
        static_v_code=static_v_code,
    )


def _write_open_block(
    arena: VariableArenaKVCache,
    block_id: int,
    key: torch.Tensor,
    value: torch.Tensor,
    live: set[int] | None = None,
) -> None:
    slots = torch.arange(block_id * BLOCK, (block_id + 1) * BLOCK, dtype=torch.int64)
    if live is None:
        live = set(range(block_id + 1))
    arena.begin_writes(slots, live)
    arena.write_and_finalize(slots, key, value)


def _exact_fp4_block() -> torch.Tensor:
    """Values exactly representable in E2M1 at unit scale."""
    row = torch.tensor([0.0, 1.0, -2.0, 3.0, 0.5, -1.5, 2.0, -3.0])
    return row.repeat(BLOCK, HEADS, 1).bfloat16()


def _small_random_block() -> torch.Tensor:
    """Small values that stay E4M3: FP4 error exceeds 0.05, E4M3 does not."""
    return (torch.rand(BLOCK, HEADS, DIM) * 0.2).bfloat16()


def test_open_write_and_e4m3_roundtrip():
    arena = _arena(max_relative_rmse=0.1)
    key = torch.randn(BLOCK, HEADS, DIM).bfloat16()
    value = torch.randn(BLOCK, HEADS, DIM).bfloat16()
    _write_open_block(arena, 0, key, value)

    # Small random values meet the limit in E4M3 (FP4 error is larger).
    assert arena._codes[0] == 1 and arena._finalized[0]
    decoded_keys, decoded_values = arena.gather_blocks([0])
    # E4M3 quantization error only; the head-major layout must roundtrip
    # without reordering.
    torch.testing.assert_close(decoded_keys[0], key.float(), atol=1 / 64, rtol=0.07)
    torch.testing.assert_close(decoded_values[0], value.float(), atol=1 / 64, rtol=0.07)


def test_completed_blocks_select_mixed_formats_and_shrink_tail():
    arena = _arena(max_relative_rmse=0.1)
    key0 = _exact_fp4_block()
    value0 = -key0
    key1 = torch.randn(BLOCK, HEADS, DIM).bfloat16()
    value1 = torch.randn(BLOCK, HEADS, DIM).bfloat16()

    _write_open_block(arena, 0, key0, value0)
    _write_open_block(arena, 1, key1, value1)

    assert arena._codes[0] == 3, "exact FP4 block should select FP4"
    assert arena._sizes[0] == FP8_BLOCK_BYTES // 2
    # Block 0 was the tail when it shrank, so block 1 starts right after it.
    assert arena._offsets[1] == FP8_BLOCK_BYTES // 2
    assert arena._sizes[1] == FP8_BLOCK_BYTES
    assert arena.free_ptr == FP8_BLOCK_BYTES // 2 + FP8_BLOCK_BYTES

    decoded, _ = arena.gather_blocks([0])
    torch.testing.assert_close(decoded[0], key0.float())


def test_tail_shrink_reclaims_fp4_upper_half():
    arena = _arena()
    key = _exact_fp4_block()
    _write_open_block(arena, 0, key, -key)
    assert arena._codes[0] == 3
    assert arena.free_ptr == FP8_BLOCK_BYTES // 2
    assert arena._sizes[0] == FP8_BLOCK_BYTES // 2

    # A new block after the shrunken tail starts right at the tail.
    key1 = _small_random_block()
    _write_open_block(arena, 1, key1, key1)
    assert arena._offsets[1] == FP8_BLOCK_BYTES // 2
    assert arena.free_ptr == FP8_BLOCK_BYTES // 2 + FP8_BLOCK_BYTES


def test_slot_zero_restarts_block_lifetime():
    arena = _arena(max_relative_rmse=0.1)
    key0 = _small_random_block()
    _write_open_block(arena, 0, key0, key0)
    assert arena._codes[0] == 1

    # The same physical block receives slot 0 again: a fresh lifetime that
    # abandons the previous reservation.
    key1 = _small_random_block()
    slots = torch.arange(0, BLOCK, dtype=torch.int64)
    arena.begin_writes(slots, {0})
    arena.write_and_finalize(slots, key1, key1)
    assert arena._written[0] == BLOCK
    decoded, _ = arena.gather_blocks([0])
    # The new lifetime's values are read, not the abandoned ones.
    torch.testing.assert_close(decoded[0], key1.float(), atol=1 / 64, rtol=0.07)


def test_free_dead_blocks_and_reset_empty_arena():
    arena = _arena()
    key = _small_random_block()
    _write_open_block(arena, 0, key, key)
    _write_open_block(arena, 1, key, key)
    assert arena.free_ptr == 2 * FP8_BLOCK_BYTES

    # No live blocks: everything is released and the free pointer resets.
    arena.begin_writes(torch.empty(0, dtype=torch.int64), set())
    assert arena.free_ptr == 0
    assert arena._codes == [-1, -1, -1]

    # Allocating again starts at offset zero.
    slots = torch.arange(0, BLOCK, dtype=torch.int64)
    arena.begin_writes(slots, {0})
    arena.write_and_finalize(slots, key, key)
    assert arena._offsets[0] == 0


def test_compaction_reclaims_holes_on_overflow():
    arena = _arena(num_blocks=4, avg_bytes=0.75)  # 192 bytes of arena
    key = _small_random_block()
    slots0 = torch.arange(0, BLOCK, dtype=torch.int64)
    slots1 = torch.arange(BLOCK, 2 * BLOCK, dtype=torch.int64)
    slots2 = torch.arange(2 * BLOCK, 3 * BLOCK, dtype=torch.int64)
    arena.begin_writes(slots0, {0})
    arena.write_and_finalize(slots0, key, key)
    arena.begin_writes(slots1, {0, 1})
    arena.write_and_finalize(slots1, key, key)
    arena.begin_writes(slots2, {0, 1, 2})
    arena.write_and_finalize(slots2, key, key)
    assert arena.free_ptr == 3 * FP8_BLOCK_BYTES

    # Free block 0; its bytes become a hole. Allocating block 3 would
    # overflow, so compaction must first move live blocks 1 and 2 to the
    # front of the arena.
    slots3 = torch.arange(3 * BLOCK, 4 * BLOCK, dtype=torch.int64)
    arena.begin_writes(slots3, {1, 2, 3})
    arena.write_and_finalize(slots3, key, key)
    assert arena._offsets[1] == 0
    assert arena._offsets[2] == FP8_BLOCK_BYTES
    assert arena._offsets[3] == 2 * FP8_BLOCK_BYTES
    assert arena.free_ptr == 3 * FP8_BLOCK_BYTES
    decoded, _ = arena.gather_blocks([1, 2, 3])
    torch.testing.assert_close(decoded[0], key.float(), atol=1 / 64, rtol=0.07)


def test_compaction_preserves_pages_when_physical_ids_are_out_of_order():
    arena = _arena(num_blocks=3)
    key_one = torch.full((BLOCK, HEADS, DIM), 0.125).bfloat16()
    key_zero = torch.full((BLOCK, HEADS, DIM), 1.0).bfloat16()
    _write_open_block(arena, 1, key_one, -key_one, {0, 1})
    _write_open_block(arena, 0, key_zero, -key_zero, {0, 1})

    # ID 0 currently follows ID 1 in the byte arena. Moving ID 0 to
    # offset zero overwrites ID 1's old region unless both are snapshotted.
    arena.compact({0, 1})
    decoded, decoded_values = arena.gather_blocks([0, 1])
    torch.testing.assert_close(decoded[0], key_zero.float())
    torch.testing.assert_close(decoded[1], key_one.float())
    torch.testing.assert_close(decoded_values, -decoded)


def test_overflow_fails_closed():
    arena = _arena(num_blocks=2, avg_bytes=0.625)  # 80 bytes; two FP8 blocks need 128
    key = _small_random_block()
    _write_open_block(arena, 0, key, key)
    slots = torch.arange(BLOCK, 2 * BLOCK, dtype=torch.int64)
    with pytest.raises(RuntimeError, match="overflow"):
        arena.begin_writes(slots, {0, 1})
        arena.write_and_finalize(slots, key, key)


def test_copy_blocks_same_format_only():
    arena = _arena(max_relative_rmse=0.1)
    key0 = _small_random_block()
    key1 = _small_random_block()
    _write_open_block(arena, 0, key0, key0)
    _write_open_block(arena, 1, key1, key1)
    arena.copy_blocks(torch.tensor([[0, 1]]))
    decoded_before, _ = arena.gather_blocks([0])
    decoded_after, _ = arena.gather_blocks([1])
    torch.testing.assert_close(decoded_after, decoded_before)

    # Cross-format copies need a repack; the arena rejects them.
    arena2 = _arena(max_relative_rmse=0.1)
    exact = _exact_fp4_block()
    random = _small_random_block()
    _write_open_block(arena2, 0, exact, exact)  # FP4
    _write_open_block(arena2, 1, random, random)  # E4M3
    with pytest.raises(ValueError, match="matching formats"):
        arena2.copy_blocks(torch.tensor([[0, 1]]))


def test_grouped_fp4_scales_roundtrip():
    arena = _arena(fp4_group_size=4, max_relative_rmse=0.5)
    key = torch.randn(BLOCK, HEADS, DIM).bfloat16() * 0.05
    value = torch.randn(BLOCK, HEADS, DIM).bfloat16() * 4.0
    _write_open_block(arena, 0, key, value)
    assert arena._codes[0] == 3
    assert arena.scales.shape == (3, 2, HEADS, DIM // 4)
    decoded_keys, decoded_values = arena.gather_blocks([0])
    assert decoded_keys[0].shape == (BLOCK, HEADS, DIM)
    # The gather must apply the exact same per-group scales the packer used.
    key_divisor = arena.scales[0][0].repeat_interleave(4, dim=-1)[None]
    expected = decode_e2m1(encode_e2m1(key.float() / key_divisor)) * key_divisor
    torch.testing.assert_close(decoded_keys[0], expected)
    # Small K values must not be erased by the much larger V stream.
    assert (decoded_keys[0] - key.float()).abs().mean().item() < 0.02


def test_reference_attention_prefill_matches_bruteforce():
    arena = _arena(max_relative_rmse=0.1)
    key = torch.randn(6, HEADS, DIM).bfloat16()
    value = torch.randn(6, HEADS, DIM).bfloat16()
    query = torch.randn(6, HEADS, DIM)
    output = torch.zeros(6, HEADS, DIM)
    slots = torch.tensor([0, 1, 2, 3, 4, 5])
    block_tables = torch.tensor([[0, 1, -1]], dtype=torch.int32)

    reference_variable_attention(
        query, key, value, arena, slots, block_tables, [6], [6], output, 0.1
    )
    scores = torch.einsum("qhd,khd->qhk", query, key.float())
    scores *= 0.1
    mask = torch.triu(torch.ones(6, 6, dtype=torch.bool), diagonal=1)
    scores.masked_fill_(mask[:, None, :], float("-inf"))
    probabilities = torch.softmax(scores, dim=-1)
    expected = torch.einsum("qhk,khd->qhd", probabilities, value.float())
    torch.testing.assert_close(output, expected)


def test_cross_page_prefill_finalizes_before_reusing_scratch():
    arena = _arena(num_blocks=3, avg_bytes=0.75, max_relative_rmse=0.1)
    exact = _exact_fp4_block()
    key = torch.cat((exact, exact[:2]))
    value = -key
    query = torch.randn(6, HEADS, DIM)
    output = torch.empty_like(query)
    # Logical blocks arrive in physical order 2, 0. The FP4 shrink must
    # happen before reserving block 0 in the 0.75-byte-per-element arena.
    slots = torch.tensor([8, 9, 10, 11, 0, 1])
    block_tables = torch.tensor([[2, 0, -1]], dtype=torch.int32)

    reference_variable_attention(
        query, key, value, arena, slots, block_tables, [6], [6], output, 0.1
    )

    assert arena._codes[2] == 3
    assert arena._offsets[2] == 0
    assert arena._offsets[0] == FP8_BLOCK_BYTES // 2
    decoded_key, decoded_value = arena.gather_blocks([2, 0])
    torch.testing.assert_close(decoded_key[0], exact.float())
    torch.testing.assert_close(decoded_value[0], -exact.float())
    assert bool(torch.isfinite(output).all())


def test_reference_attention_decode_reads_compressed_history():
    arena = _arena(max_relative_rmse=0.1)
    key = torch.randn(5, HEADS, DIM).bfloat16()
    value = torch.randn(5, HEADS, DIM).bfloat16()
    slots = torch.tensor([0, 1, 2, 3, 4])
    block_tables = torch.tensor([[0, 1, -1]], dtype=torch.int32)
    arena.begin_writes(slots, {0, 1})
    arena.write_and_finalize(slots, key, value)

    # Decode token 5: seq_len 6, query length 1.
    k_new = torch.randn(1, HEADS, DIM).bfloat16()
    v_new = torch.randn(1, HEADS, DIM).bfloat16()
    query = torch.randn(1, HEADS, DIM)
    output = torch.zeros(1, HEADS, DIM)
    slots = torch.tensor([5])
    reference_variable_attention(
        query, k_new, v_new, arena, slots, block_tables, [1], [6], output, 0.1
    )
    # The just-written token is read from the compressed cache, like the
    # fixed-stride reference. Recompute from the arena bytes.
    key_blocks, value_blocks = arena.gather_blocks([0, 1])
    all_keys = key_blocks.reshape(-1, HEADS, DIM)[:6]
    all_values = value_blocks.reshape(-1, HEADS, DIM)[:6]
    scores = torch.einsum("qhd,khd->qhk", query, all_keys)
    scores *= 0.1
    probabilities = torch.softmax(scores, dim=-1)
    expected = torch.einsum("qhk,khd->qhd", probabilities, all_values)
    torch.testing.assert_close(output, expected)


def test_live_block_ids_covers_tables_and_writes():
    block_tables = torch.tensor([[3, 7, -1, -1]], dtype=torch.int32)
    slots = torch.tensor([5], dtype=torch.int64)
    live = live_block_ids(block_tables, [9], slots, BLOCK)
    assert live == {3, 7, 1}


def test_invalid_inputs_are_rejected():
    arena = _arena()
    key = torch.randn(2, HEADS, DIM).bfloat16()
    with pytest.raises(ValueError, match="unique slots"):
        arena.begin_writes(torch.tensor([0, 0]), {0})
        arena.write_and_finalize(torch.tensor([0, 0]), key, key)
    with pytest.raises(ValueError, match="shapes"):
        arena.write_and_finalize(torch.tensor([0]), key[:1, :, :3], key[:1])
    with pytest.raises(ValueError, match="outside"):
        arena.write_and_finalize(
            torch.tensor([arena.num_blocks * BLOCK]), key[:1], key[:1]
        )
    arena.begin_writes(torch.tensor([0, 2]), {0})
    with pytest.raises(ValueError, match="consecutive"):
        arena.write_and_finalize(torch.tensor([0, 2]), key, key)


def test_static_fp4_reserves_half_and_roundtrips():
    scales = torch.full((HEADS,), 0.25, dtype=torch.float32)
    arena = _arena(num_blocks=3, static_code=3, static_scales=scales)
    # FP4 rows are exactly representable at unit scale.
    row = torch.tensor([0.0, 0.25, -0.5, 0.75, 0.125, -0.375, 0.5, -0.75])
    key = row.repeat(BLOCK, HEADS, 1).bfloat16()
    value = -key
    _write_open_block(arena, 0, key, value)
    _write_open_block(arena, 1, key, value)

    # Allocation already reserved half a block per K/V: no completion policy
    # and no tail shrink are involved.
    assert arena._codes[0] == 3 and arena._finalized[0]
    assert arena._sizes[0] == FP8_BLOCK_BYTES // 2
    assert arena._offsets[1] == FP8_BLOCK_BYTES // 2
    assert arena.free_ptr == FP8_BLOCK_BYTES
    decoded_keys, decoded_values = arena.gather_blocks([0, 1])
    torch.testing.assert_close(decoded_keys[0], key.float())
    torch.testing.assert_close(decoded_values[0], value.float())
    # A same-format copy keeps the calibrated bytes.
    _write_open_block(arena, 2, value, key)
    arena.copy_blocks(torch.tensor([[0, 2]]))
    copied, _ = arena.gather_blocks([2])
    torch.testing.assert_close(copied[0], key.float())


def test_static_e4m3_writes_exact_bytes():
    arena = _arena(num_blocks=2, static_code=1)
    key = torch.randn(BLOCK, HEADS, DIM).bfloat16()
    value = torch.randn(BLOCK, HEADS, DIM).bfloat16()
    _write_open_block(arena, 0, key, value)
    assert arena._codes[0] == 1
    assert arena._sizes[0] == FP8_BLOCK_BYTES
    # The calibrated E4M3 path stores the same bytes as the reference encoder.
    expected = arena.k_buffer[:FP8_BLOCK_BYTES].view(HEADS, BLOCK, DIM)
    torch.testing.assert_close(
        decode_e4m3(expected).permute(1, 0, 2), key.float(), atol=1 / 64, rtol=0.07
    )


def test_static_fp4_clamps_outliers_to_finite_codes():
    scales = torch.full((HEADS,), 0.25, dtype=torch.float32)
    arena = _arena(num_blocks=2, static_code=3, static_scales=scales)
    key = torch.full((BLOCK, HEADS, DIM), 100.0, dtype=torch.bfloat16)
    _write_open_block(arena, 0, key, key)
    decoded, _ = arena.gather_blocks([0])
    # 3 * 0.25 * 2 headroom = 1.5 is the clamped magnitude, encoded finitely.
    assert bool(torch.isfinite(decoded).all())
    assert decoded.abs().max().item() <= 1.5


def test_static_precision_map_parser(tmp_path):
    import subprocess
    import sys

    log = tmp_path / "calibration.log"
    log.write_text(
        "PSP-KV calibration layer=model.layers.0.attn code=3 "
        "k_max=0.5,0.25 v_max=0.5,0.25\n"
        "PSP-KV calibration layer=model.layers.0.attn code=3 "
        "k_max=1.0,0.5 v_max=1.0,0.5\n"
        "PSP-KV calibration layer=model.layers.1.attn code=2 "
        "k_max=9.0,9.0 v_max=9.0,9.0\n"
        "PSP-KV calibration layer=model.layers.1.attn code=1 "
        "k_max=2.0,2.0 v_max=2.0,2.0\n"
        "PSP-KV calibration layer=model.layers.1.attn code=1 "
        "k_max=2.0,2.0 v_max=2.0,2.0\n",
        encoding="utf-8",
    )
    output = tmp_path / "map.json"
    script = (
        __import__("pathlib").Path(__file__).resolve().parents[3]
        / "ascend/calibrate_precision.py"
    )
    subprocess.run(
        [sys.executable, str(script), "--log", str(log), "--output", str(output)],
        check=True,
        capture_output=True,
    )
    import json

    mapping = json.loads(output.read_text(encoding="utf-8"))
    assert mapping["model.layers.0.attn"]["code"] == 3
    # 1.0 / 3 rounds up to the next power of two: 0.5.
    assert mapping["model.layers.0.attn"]["k_scales"] == [0.5, 0.25]
    assert mapping["model.layers.0.attn"]["v_scales"] == [0.5, 0.25]
    assert mapping["model.layers.1.attn"]["code"] == 1


def test_static_grouped_fp4_independent_kv_scales():
    # K gets a fine scale, V a coarse one: K values must not be erased by a
    # large V stream, and each dimension group keeps its own scale.
    # Group 0 (dims 0-3) uses 0.125 for K and 1.0 for V; group 1 uses 1.0/4.0.
    k_scales = torch.tensor([[0.125, 1.0]], dtype=torch.float32).repeat(HEADS, 1)
    v_scales = torch.tensor([[1.0, 4.0]], dtype=torch.float32).repeat(HEADS, 1)
    scales = torch.stack((k_scales, v_scales), dim=0)  # [2, heads, 2 groups]
    arena = _arena(num_blocks=2, static_code=3, static_scales=scales)
    assert arena.scales.shape == (2, 2, HEADS, 2)
    assert arena.static_scale_groups == 2

    # Exact FP4 multiples per group: 0.125 * {0..3} for K, 1.0 * {0..3} for V.
    k_row = torch.tensor([0.0, 0.125, 0.25, 0.375, 0.0, 1.0, 2.0, 3.0])
    # V group 0 scale 1.0 (max 3.0), group 1 scale 4.0 (max 12.0): both exact.
    v_row = torch.tensor([0.0, 1.0, 2.0, 3.0, 0.0, 4.0, 8.0, 12.0])
    key = k_row.repeat(BLOCK, HEADS, 1).bfloat16()
    value = v_row.repeat(BLOCK, HEADS, 1).bfloat16()
    _write_open_block(arena, 0, key, value)
    decoded_k, decoded_v = arena.gather_blocks([0])
    torch.testing.assert_close(decoded_k[0], key.float())
    torch.testing.assert_close(decoded_v[0], value.float())

    # Group 0 clamps K at 0.125*3=0.375 and group 1 at 3.0.
    clamped = arena._clamp_static_fp4(
        torch.full((1, HEADS, DIM), 100.0, dtype=torch.bfloat16), 0
    )
    assert clamped[0, 0, 0].item() == 0.375
    assert clamped[0, 0, -1].item() == 3.0


def test_calibrate_precision_parser_with_groups(tmp_path):
    import json
    import subprocess
    import sys

    log = tmp_path / "calibration.log"
    log.write_text(
        "PSP-KV calibration layer=model.layers.0.attn code=3 groups=2 "
        "k_max=0.25,1.0 v_max=2.0,4.0\n"
        "PSP-KV calibration layer=model.layers.0.attn code=3 groups=2 "
        "k_max=0.5,2.0 v_max=4.0,8.0\n",
        encoding="utf-8",
    )
    output = tmp_path / "map.json"
    script = (
        __import__("pathlib").Path(__file__).resolve().parents[3]
        / "ascend/calibrate_precision.py"
    )
    subprocess.run(
        [sys.executable, str(script), "--log", str(log), "--output", str(output)],
        check=True,
        capture_output=True,
    )
    entry = json.loads(output.read_text(encoding="utf-8"))["model.layers.0.attn"]
    assert entry["code"] == 3 and entry["groups"] == 2
    # One KV head, two groups: k group maxima 0.5 and 2.0 ->
    # 2^-2 and 2^0; v maxima 4.0 and 8.0 -> 2^1 and 2^2.
    assert entry["k_scales"] == [[0.25, 1.0]]
    assert entry["v_scales"] == [[2.0, 4.0]]


def test_mixed_static_kv_codes_reserve_per_stream():
    # K stays E4M3 (1 byte/element) while V is packed FP4 (half a byte), so
    # each stream keeps its own reservation and byte offset.
    scales = torch.stack(
        (
            torch.ones(HEADS, 1, dtype=torch.float32),
            torch.full((HEADS, 1), 0.25, dtype=torch.float32),
        ),
        dim=0,
    )
    arena = _arena(
        num_blocks=3,
        avg_bytes=1.0,
        static_code=1,
        static_v_code=3,
        static_scales=scales,
    )
    key = torch.randn(BLOCK, HEADS, DIM).bfloat16()
    value = (
        torch.tensor([0.0, 0.125, 0.25, 0.375, 0.0, 0.5, 0.75, 0.25])
        .repeat(BLOCK, HEADS, 1)
        .bfloat16()
    )
    _write_open_block(arena, 0, key, value)
    assert arena._codes[0] == 1 and arena._v_codes[0] == 3
    assert arena._sizes[0] == FP8_BLOCK_BYTES
    assert arena._v_sizes[0] == FP8_BLOCK_BYTES // 2
    assert arena.free_ptr == FP8_BLOCK_BYTES
    assert arena.v_free_ptr == FP8_BLOCK_BYTES // 2

    _write_open_block(arena, 1, key, value)
    assert arena._offsets[1] == FP8_BLOCK_BYTES
    assert arena._v_offsets[1] == FP8_BLOCK_BYTES // 2

    decoded_k, decoded_v = arena.gather_blocks([0, 1])
    torch.testing.assert_close(decoded_k[0], key.float(), atol=1 / 64, rtol=0.07)
    # V is exact: every value is an E2M1 multiple of the 0.25 scale.
    torch.testing.assert_close(decoded_v[0], value.float())
    # Same-format copies move each stream with its own size.
    _write_open_block(arena, 2, key, value)
    arena.copy_blocks(torch.tensor([[0, 2]]))
    copied_k, copied_v = arena.gather_blocks([2])
    torch.testing.assert_close(copied_k, decoded_k[:1])
    torch.testing.assert_close(copied_v, decoded_v[:1])


def test_psp_kv_spec_per_stream_page_bytes(monkeypatch):
    import torch as _torch

    from vllm.v1.kv_cache_interface import PSPKVAttentionSpec

    monkeypatch.setenv("PSPKV_AVG_BYTES", "1.0")
    monkeypatch.setenv("PSPKV_AVG_K_BYTES", "1.0")
    monkeypatch.setenv("PSPKV_AVG_V_BYTES", "0.5")
    spec = PSPKVAttentionSpec(
        block_size=128,
        num_kv_heads=8,
        head_size=128,
        head_size_v=128,
        dtype=_torch.uint8,
    )
    assert spec.k_page_size_bytes == 128 * 8 * 128
    assert spec.v_page_size_bytes == 128 * 8 * 64
    assert spec.real_page_size_bytes == 128 * 8 * 128 + 128 * 8 * 64
    # An empty override falls back to the shared budget.
    monkeypatch.setenv("PSPKV_AVG_V_BYTES", "")
    assert spec.v_page_size_bytes == 128 * 8 * 128
