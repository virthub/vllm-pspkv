# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU-only unit tests for the PSP-KV KV-cache layout and op contract.

These verify the *shape/container contract* that the GPU kernels rely on
(layout, key/value splitting, and the registered custom-op surface). They do
not execute the CUDA kernels themselves, so they run without a GPU.
"""

import pytest
import torch

from vllm.v1.attention.backends.psp_kv_attn import PSPKVAttentionBackend


@pytest.mark.parametrize("cache_dtype_str", ["auto", "fp8", "float16", "bfloat16"])
def test_get_kv_cache_shape_is_dtype_independent(cache_dtype_str):
    """The PSP-KV cache layout does not change with the cache dtype."""
    shape = PSPKVAttentionBackend.get_kv_cache_shape(
        num_blocks=64,
        block_size=16,
        num_kv_heads=8,
        head_size=128,
        cache_dtype_str=cache_dtype_str,
    )
    # Flat byte buffer: (2, num_blocks * block_size * num_kv_heads * head_size).
    assert shape == (2, 64 * 16 * 8 * 128)


@pytest.mark.parametrize(
    "num_blocks,block_size,num_kv_heads,head_size",
    [
        (1, 16, 1, 64),        # minimal single-block
        (1024, 16, 8, 128),    # typical decode
        (256, 32, 2, 128),     # GQA with larger block size
        (512, 16, 8, 256),     # long head dim
    ],
)
def test_get_kv_cache_shape_geometry(num_blocks, block_size, num_kv_heads, head_size):
    shape = PSPKVAttentionBackend.get_kv_cache_shape(
        num_blocks=num_blocks,
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
    )
    # Flat byte buffer: the second axis is the total per-K/V byte footprint.
    per_kv_bytes = num_blocks * block_size * num_kv_heads * head_size
    assert shape == (2, per_kv_bytes)
    # Leading axis separates key (index 0) from value (index 1).
    assert shape[0] == 2


def test_kv_cache_leading_axis_is_key_value():
    """forward() must slice the unified cache into key/value along dim 0."""
    num_blocks, num_kv_heads, block_size, head_size = 4, 2, 16, 64
    shape = PSPKVAttentionBackend.get_kv_cache_shape(
        num_blocks, block_size, num_kv_heads, head_size
    )
    # An FP8 cache uses a uint8 container; the shape itself is dtype-agnostic.
    cache = torch.zeros(shape, dtype=torch.uint8)
    key_cache, value_cache = cache[0], cache[1]
    per_kv_bytes = num_blocks * num_kv_heads * block_size * head_size
    assert key_cache.shape == (per_kv_bytes,)
    assert value_cache.shape == (per_kv_bytes,)
    # The two halves must be distinct, contiguous slices of the same storage.
    assert key_cache.data_ptr() != value_cache.data_ptr()
    assert key_cache.is_contiguous() and value_cache.is_contiguous()


def test_cache_ops_are_registered():
    """The PSP-KV cache ops (read + write) must be registered on torch.ops.psp_kv."""
    import vllm._C  # noqa: F401  (side effect: registers the ops)

    pkv = getattr(torch.ops, "psp_kv", None)
    assert pkv is not None, "torch.ops.psp_kv namespace not registered"

    for op in (
        "psp_attention_basic",
        "psp_attention_enhanced",
        "psp_attention_btkv",
        "quantize_and_scatter_kv",
    ):
        assert hasattr(pkv, op), f"PSP-KV cache op {op!r} is not registered"


def test_quantize_and_scatter_kv_schema():
    """The cache-write op must expose the documented argument order."""
    import vllm._C  # noqa: F401

    op = torch.ops.psp_kv.quantize_and_scatter_kv
    schema = op._schemas[""]
    arg_names = [a.name for a in schema.arguments]
    assert arg_names == [
        "prefill_scratch",
        "key_cache",
        "value_cache",
        "block_tables",
        "slot_mapping",
        "precision_table",
        "block_offsets",
        "scale_table",
        "num_tokens",
        "format_generation",
    ]


@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for op dispatch"
)
def test_quantize_and_scatter_kv_rejects_wrong_dtype():
    """The cache-write op validates its FP8-cache contract before launching.

    prefill_scratch must be fp16/bf16 (it is quantized into the fp8 cache).
    This validation fires before the kernel launcher, so it verifies the write
    contract even while the launcher is still a stub.
    """
    import vllm._C  # noqa: F401

    op = torch.ops.psp_kv.quantize_and_scatter_kv
    num_tokens, num_kv_heads, block_size, head_dim = 4, 2, 16, 64
    num_blocks = 8
    with pytest.raises(
        RuntimeError, match="supports fp16/bf16"
    ):
        op(
            torch.zeros(num_tokens, 2, num_kv_heads, head_dim, dtype=torch.int64, device="cuda"),
            torch.zeros(num_blocks * num_kv_heads * block_size * head_dim, dtype=torch.uint8, device="cuda"),
            torch.zeros(num_blocks * num_kv_heads * block_size * head_dim, dtype=torch.uint8, device="cuda"),
            torch.zeros(1, num_blocks, dtype=torch.int32, device="cuda"),
            torch.zeros(num_tokens, dtype=torch.int64, device="cuda"),
            torch.zeros(num_blocks, dtype=torch.uint8, device="cuda"),
            torch.zeros(num_blocks, dtype=torch.int32, device="cuda"),
            torch.zeros(num_blocks, num_kv_heads, dtype=torch.float32, device="cuda"),
            num_tokens,
            1,
        )
