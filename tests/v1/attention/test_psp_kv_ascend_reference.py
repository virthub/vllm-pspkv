# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""CPU checks for the FP8 formats used by the Ascend reference path."""

import importlib.util
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

pytestmark = pytest.mark.skip_global_cleanup

_REFERENCE_PATH = (
    Path(__file__).resolve().parents[3]
    / "vllm/v1/attention/backends/psp_kv_ascend_reference.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "psp_kv_ascend_reference", _REFERENCE_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
_CODEC = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_CODEC)


def test_e4m3_known_codes():
    values = torch.tensor([0.0, -0.0, 0.5, 1.0, -1.0, 2.0, 1 / 512, 240.0])
    encoded = _CODEC.encode_e4m3(values)
    assert encoded.dtype == torch.uint8
    assert encoded.tolist() == [0x00, 0x00, 0x30, 0x38, 0xB8, 0x40, 0x01, 0x77]

    expected = torch.tensor([0.0, 0.0, 0.5, 1.0, -1.0, 2.0, 1 / 512, 240.0])
    torch.testing.assert_close(_CODEC.decode_e4m3(encoded), expected)


def test_e4m3_quantization_error_and_reserved_codes():
    generator = torch.Generator().manual_seed(194)
    values = torch.randn(4096, generator=generator).clamp(-8.0, 8.0)
    decoded = _CODEC.decode_e4m3(_CODEC.encode_e4m3(values))
    torch.testing.assert_close(decoded, values, atol=1 / 64, rtol=0.07)

    encoded = _CODEC.encode_e4m3(
        torch.tensor([float("inf"), -float("inf"), float("nan")])
    )
    assert encoded.tolist() == [0x78, 0xF8, 0x7F]
    assert torch.isnan(_CODEC.decode_e4m3(encoded)[-1])


def test_e5m2_known_codes_and_reserved_values():
    values = torch.tensor(
        [
            0.0,
            -0.0,
            0.5,
            1.0,
            -1.0,
            2.0,
            2**-16,
            57344.0,
            float("inf"),
            -float("inf"),
            float("nan"),
        ]
    )
    encoded = _CODEC.encode_e5m2(values)
    assert encoded.tolist() == [
        0x00,
        0x00,
        0x38,
        0x3C,
        0xBC,
        0x40,
        0x01,
        0x7B,
        0x7C,
        0xFC,
        0x7F,
    ]
    decoded = _CODEC.decode_e5m2(encoded)
    torch.testing.assert_close(decoded[:8], values[:8])
    assert torch.isposinf(decoded[8])
    assert torch.isneginf(decoded[9])
    assert torch.isnan(decoded[10])


def _dense_attention(query, key, value, causal):
    # Independent PyTorch attention for exactly representable E4M3 values.
    groups = query.shape[1] // key.shape[1]
    key = key.repeat_interleave(groups, dim=1)
    value = value.repeat_interleave(groups, dim=1)
    return F.scaled_dot_product_attention(
        query.transpose(0, 1),
        key.transpose(0, 1),
        value.transpose(0, 1),
        is_causal=causal,
        scale=0.5,
    ).transpose(0, 1)


def test_reference_prefill_and_decode_follow_physical_block_table():
    generator = torch.Generator().manual_seed(210)
    k_cache = torch.zeros((4, 4, 2, 4), dtype=torch.uint8)
    v_cache = torch.zeros_like(k_cache)
    table = torch.tensor([[3, 1], [0, -1]], dtype=torch.int32)
    q = torch.randint(-2, 3, (7, 4, 4), generator=generator).float()
    k = torch.randint(-2, 3, (7, 2, 4), generator=generator).float()
    v = torch.randint(-2, 3, (7, 2, 4), generator=generator).float()
    output = torch.empty_like(q)

    _CODEC.reference_attention(
        q,
        k,
        v,
        k_cache,
        v_cache,
        torch.tensor([12, 13, 14, 15, 4, 0, 1]),
        table,
        [5, 7],
        [5, 2],
        output,
        0.5,
    )
    torch.testing.assert_close(output[:5], _dense_attention(q[:5], k[:5], v[:5], True))
    torch.testing.assert_close(output[5:], _dense_attention(q[5:], k[5:], v[5:], True))
    assert k_cache[3, 0].tolist() == _CODEC.encode_e4m3(k[0]).tolist()
    assert k_cache[1, 0].tolist() == _CODEC.encode_e4m3(k[4]).tolist()
    assert k_cache[0, 0].tolist() == _CODEC.encode_e4m3(k[5]).tolist()

    q_next = torch.randint(-2, 3, (2, 4, 4), generator=generator).float()
    k_next = torch.randint(-2, 3, (2, 2, 4), generator=generator).float()
    v_next = torch.randint(-2, 3, (2, 2, 4), generator=generator).float()
    decode_output = torch.empty_like(q_next)
    _CODEC.reference_attention(
        q_next,
        k_next,
        v_next,
        k_cache,
        v_cache,
        torch.tensor([5, 2]),
        table,
        [1, 2],
        [6, 3],
        decode_output,
        0.5,
    )
    torch.testing.assert_close(
        decode_output[:1],
        _dense_attention(
            q_next[:1],
            torch.cat((k[:5], k_next[:1])),
            torch.cat((v[:5], v_next[:1])),
            False,
        ),
    )
    torch.testing.assert_close(
        decode_output[1:],
        _dense_attention(
            q_next[1:],
            torch.cat((k[5:], k_next[1:])),
            torch.cat((v[5:], v_next[1:])),
            False,
        ),
    )


def test_sdpa_attention_matches_reference_for_prefix_chunk():
    generator = torch.Generator().manual_seed(917)
    block_size = 4
    num_heads, num_kv_heads, head_size = 4, 2, 8
    prefix_len, chunk_len = 4, 3
    prefix_key = torch.randint(
        -2, 3, (prefix_len, num_kv_heads, head_size), generator=generator
    ).float()
    prefix_value = torch.randint(
        -2, 3, (prefix_len, num_kv_heads, head_size), generator=generator
    ).float()
    key = torch.randint(
        -2, 3, (chunk_len, num_kv_heads, head_size), generator=generator
    ).float()
    value = torch.randint(
        -2, 3, (chunk_len, num_kv_heads, head_size), generator=generator
    ).float()
    query = torch.randint(
        -2, 3, (chunk_len, num_heads, head_size), generator=generator
    ).float()
    block_tables = torch.tensor([[2, 0, 1]], dtype=torch.int32)

    outputs = []
    for use_sdpa in (False, True):
        k_cache = torch.zeros(
            (3, block_size, num_kv_heads, head_size), dtype=torch.uint8
        )
        v_cache = torch.zeros_like(k_cache)
        k_cache[2, :prefix_len] = _CODEC.encode_e4m3(prefix_key)
        v_cache[2, :prefix_len] = _CODEC.encode_e4m3(prefix_value)
        output = torch.empty_like(query)
        _CODEC.reference_attention(
            query,
            key,
            value,
            k_cache,
            v_cache,
            torch.tensor([4, 5, 6]),
            block_tables,
            [chunk_len],
            [prefix_len + chunk_len],
            output,
            1.0 / (head_size**0.5),
            use_sdpa=use_sdpa,
        )
        outputs.append(output)

    torch.testing.assert_close(outputs[1], outputs[0], rtol=1e-5, atol=1e-6)


def test_sdpa_attention_matches_reference_for_single_token_decode():
    generator = torch.Generator().manual_seed(921)
    block_size, num_kv_heads, num_heads, head_size = 4, 2, 4, 8
    k_cache = torch.zeros((2, block_size, num_kv_heads, head_size), dtype=torch.uint8)
    v_cache = torch.zeros_like(k_cache)
    prefix_key = torch.randint(
        -2, 3, (block_size, num_kv_heads, head_size), generator=generator
    ).float()
    prefix_value = torch.randint(
        -2, 3, (block_size, num_kv_heads, head_size), generator=generator
    ).float()
    k_cache[0] = _CODEC.encode_e4m3(prefix_key)
    v_cache[0] = _CODEC.encode_e4m3(prefix_value)
    query = torch.randint(-2, 3, (1, num_heads, head_size), generator=generator).float()
    key = torch.randint(
        -2, 3, (1, num_kv_heads, head_size), generator=generator
    ).float()
    value = torch.randint(
        -2, 3, (1, num_kv_heads, head_size), generator=generator
    ).float()
    outputs = []

    for use_sdpa in (False, True):
        k_work, v_work = k_cache.clone(), v_cache.clone()
        output = torch.empty_like(query)
        _CODEC.reference_attention(
            query,
            key,
            value,
            k_work,
            v_work,
            torch.tensor([block_size]),
            torch.tensor([[0, 1]], dtype=torch.int32),
            [1],
            [block_size + 1],
            output,
            1.0 / (head_size**0.5),
            use_sdpa=use_sdpa,
        )
        outputs.append(output)

    torch.testing.assert_close(outputs[1], outputs[0], rtol=1e-5, atol=1e-6)


def test_mixed_fp8_blocks_use_physical_precision_table_on_decode():
    k_cache = torch.zeros((4, 2, 1, 1), dtype=torch.uint8)
    v_cache = torch.zeros_like(k_cache)
    precision_table = torch.tensor([1, 1, 1, 2], dtype=torch.uint8)
    block_tables = torch.tensor([[3, 1]], dtype=torch.int32)
    query = torch.zeros((3, 1, 1))
    key = torch.zeros_like(query)
    value = torch.full_like(query, 1.125)

    _CODEC.reference_attention(
        query,
        key,
        value,
        k_cache,
        v_cache,
        torch.tensor([6, 7, 2]),
        block_tables,
        [3],
        [3],
        torch.empty_like(query),
        1.0,
        precision_table=precision_table,
    )
    assert v_cache[3, 0, 0, 0].item() == 0x3C  # E5M2 rounds 1.125 to 1.0.
    assert v_cache[1, 0, 0, 0].item() == 0x39  # E4M3 preserves 1.125.

    output = torch.empty((1, 1, 1))
    _CODEC.reference_attention(
        torch.zeros_like(output),
        torch.zeros_like(output),
        torch.full_like(output, 2.0),
        k_cache,
        v_cache,
        torch.tensor([3]),
        block_tables,
        [1],
        [4],
        output,
        1.0,
        precision_table=precision_table,
    )
    torch.testing.assert_close(output, torch.full_like(output, 1.28125))


def test_adaptive_fp8_promotes_reencodes_and_resets_reused_block():
    k_cache = torch.zeros((2, 4, 1, 1), dtype=torch.uint8)
    v_cache = torch.zeros_like(k_cache)
    precision_table = torch.ones(2, dtype=torch.uint8)
    block_tables = torch.tensor([[1, 0]], dtype=torch.int32)

    def write(slots, keys, values, seq_len):
        query = torch.zeros((len(slots), 1, 1))
        key = torch.tensor(keys).reshape(-1, 1, 1)
        value = torch.tensor(values).reshape(-1, 1, 1)
        output = torch.empty_like(query)
        _CODEC.reference_attention(
            query,
            key,
            value,
            k_cache,
            v_cache,
            torch.tensor(slots),
            block_tables,
            [len(slots)],
            [seq_len],
            output,
            1.0,
            precision_table=precision_table,
            adaptive_e4m3_max_abs=240.0,
        )
        return output

    write([4, 5], [0.0, 0.0], [1.125, 1.125], 2)
    assert precision_table.tolist() == [1, 1]
    assert v_cache[1, :2, 0, 0].tolist() == [0x39, 0x39]

    output = write([6], [0.0], [300.0], 3)
    assert precision_table.tolist() == [1, 2]
    assert v_cache[1, :2, 0, 0].tolist() == [0x3C, 0x3C]
    expected = _CODEC.decode_e5m2(v_cache[1, :3, 0, 0]).mean()
    torch.testing.assert_close(output[0, 0, 0], expected)

    # A different physical block still uses E4M3.
    block_tables[0, 0] = 0
    write([0], [0.0], [1.125], 1)
    assert precision_table.tolist() == [1, 2]
    assert v_cache[0, 0, 0, 0].item() == 0x39

    # Slot offset zero marks a fresh lifetime for the recycled physical block.
    block_tables[0, 0] = 1
    write([4], [0.0], [1.125], 1)
    assert precision_table.tolist() == [1, 1]
    assert v_cache[1, 0, 0, 0].item() == 0x39


def test_adaptive_fp8_selects_format_from_keys_and_values():
    k_cache = torch.zeros((2, 2, 1, 1), dtype=torch.uint8)
    v_cache = torch.zeros_like(k_cache)
    precision_table = torch.ones(2, dtype=torch.uint8)
    _CODEC.reference_attention(
        torch.zeros((3, 1, 1)),
        torch.tensor([300.0, 0.0, 0.0]).reshape(3, 1, 1),
        torch.tensor([0.0, 1.125, 1.125]).reshape(3, 1, 1),
        k_cache,
        v_cache,
        torch.tensor([2, 3, 0]),
        torch.tensor([[1, 0]], dtype=torch.int32),
        [3],
        [3],
        torch.empty((3, 1, 1)),
        1.0,
        precision_table=precision_table,
        adaptive_e4m3_max_abs=240.0,
    )
    assert precision_table.tolist() == [1, 2]
    assert k_cache[1, 0, 0, 0].item() == _CODEC.encode_e5m2(torch.tensor(300.0)).item()
    assert v_cache[1, 1, 0, 0].item() == 0x3C
    assert v_cache[0, 0, 0, 0].item() == 0x39
