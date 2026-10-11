# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from unittest.mock import MagicMock, patch

from vllm.v1.attention.backends.psp_kv_attn import (
    PSPKVAttentionBackend,
    PSPKVAttentionMetadata,
    PSPKVAttentionMetadataBuilder,
    PSPKVAttentionImpl,
)


class DummyLayer:
    """Dummy attention layer with required attributes."""
    def __init__(self):
        self.num_kv_heads = 8
        self.num_kv_heads_per_partition = 8
        self.head_size = 128
        self.scale = 0.125


class DummyMetadata:
    """Dummy CommonAttentionMetadata with required fields."""
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


def _make_impl() -> PSPKVAttentionImpl:
    """Construct a PSPKVAttentionImpl with dummy constructor arguments."""
    return PSPKVAttentionImpl(
        num_heads=8,
        head_size=128,
        scale=0.125,
        num_kv_heads=8,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
    )


def _make_ops_mock() -> MagicMock:
    """Return a MagicMock standing in for the `torch.ops.psp_kv` namespace."""
    return MagicMock()


def test_psp_kv_attention_backend_get_name():
    assert PSPKVAttentionBackend.get_name() == "PSP_KV"


def test_psp_kv_attention_backend_get_impl_cls():
    assert PSPKVAttentionBackend.get_impl_cls() == PSPKVAttentionImpl


def test_psp_kv_attention_backend_get_builder_cls():
    assert PSPKVAttentionBackend.get_builder_cls() == PSPKVAttentionMetadataBuilder


def test_psp_kv_attention_backend_get_kv_cache_shape():
    shape = PSPKVAttentionBackend.get_kv_cache_shape(
        num_blocks=1024,
        block_size=16,
        num_kv_heads=8,
        head_size=128,
    )
    assert shape == (2, 1024, 8, 16, 128)


def test_psp_kv_metadata_builder_build():
    common_meta = DummyMetadata()
    builder = PSPKVAttentionMetadataBuilder(
        kv_cache_spec=MagicMock(),
        layer_names=["layer"],
        vllm_config=MagicMock(),
        device=torch.device("cpu"),
    )
    meta = builder.build(
        common_prefix_len=0,
        common_attn_metadata=common_meta,
        fast_build=False,
    )
    assert isinstance(meta, PSPKVAttentionMetadata)
    assert meta.common_metadata is common_meta


@patch("torch.cuda.is_current_stream_capturing", return_value=False)
@patch("vllm.v1.attention.backends.psp_kv_attn._get_psp_kv_ops")
def test_psp_kv_attention_impl_forward_dispatch_basic(mock_get_ops, _mock_capturing):
    # Setup
    layer = DummyLayer()
    query = torch.randn(1, 8, 128)
    kv_cache = torch.zeros(2, 10, 8, 16, 128, dtype=torch.uint8)
    meta = PSPKVAttentionMetadata(
        common_metadata=DummyMetadata(
            format_generation=0,
            precision_table=torch.zeros(10, dtype=torch.uint8),
            block_offsets=torch.zeros(10, dtype=torch.int32),
            block_table_tensor=torch.zeros(1, 10, dtype=torch.int32),
            context_lens=torch.zeros(1, dtype=torch.int32),
            num_actual_tokens=1,
            num_reqs=1,
        )
    )
    output = torch.empty_like(query)
    # The backend resolves ops via _get_psp_kv_ops(); patch that seam.
    mock_ops = _make_ops_mock()
    mock_ops.psp_dequantize_kv = MagicMock()
    mock_ops.psp_attention_uniform = MagicMock()
    mock_get_ops.return_value = mock_ops

    impl = _make_impl()
    impl.forward(
        layer=layer,
        query=query,
        key=torch.empty(0),
        value=torch.empty(0),
        kv_cache=kv_cache,
        attn_metadata=meta,
        output=output,
    )

    # Basic format uses the staging path: dequantize -> uniform consumer.
    mock_ops.psp_dequantize_kv.assert_called_once()
    mock_ops.psp_attention_uniform.assert_called_once()
    args, _ = mock_ops.psp_attention_uniform.call_args
    assert args[0] is query
    # args[1]/args[2] are the freshly allocated staging buffers.
    assert args[1].shape == (10, 8, 16, 128)
    assert args[2].shape == (10, 8, 16, 128)
    assert args[3] is meta.common_metadata.block_table_tensor
    assert args[4] is meta.common_metadata.context_lens
    assert args[5] == layer.scale
    assert args[6] == layer.num_kv_heads
    assert args[7] == layer.head_size
    assert args[8] is output


@patch("torch.cuda.is_current_stream_capturing", return_value=False)
@patch("vllm.v1.attention.backends.psp_kv_attn._get_psp_kv_ops")
def test_psp_kv_attention_impl_forward_dispatch_enhanced(mock_get_ops, _mock_capturing):
    layer = DummyLayer()
    query = torch.randn(1, 8, 128)
    kv_cache = torch.zeros(2, 10, 8, 16, 128, dtype=torch.uint8)
    meta = PSPKVAttentionMetadata(
        common_metadata=DummyMetadata(
            format_generation=1,
            precision_table=torch.zeros(10, dtype=torch.uint8),
            block_offsets=torch.zeros(10, dtype=torch.int32),
            scale_table=torch.zeros(10, 8, dtype=torch.float32),
            block_table_tensor=torch.zeros(1, 10, dtype=torch.int32),
            context_lens=torch.zeros(1, dtype=torch.int32),
            num_actual_tokens=1,
            num_reqs=1,
        )
    )
    output = torch.empty_like(query)
    mock_ops = _make_ops_mock()
    mock_ops.psp_attention_enhanced = MagicMock()
    mock_get_ops.return_value = mock_ops

    impl = _make_impl()
    impl.forward(
        layer=layer,
        query=query,
        key=torch.empty(0),
        value=torch.empty(0),
        kv_cache=kv_cache,
        attn_metadata=meta,
        output=output,
    )

    mock_ops.psp_attention_enhanced.assert_called_once()
    args, _ = mock_ops.psp_attention_enhanced.call_args
    assert args[5] is meta.common_metadata.precision_table
    assert args[6] is meta.common_metadata.block_offsets
    assert args[7] is meta.common_metadata.scale_table


@patch("torch.cuda.is_current_stream_capturing", return_value=False)
@patch("vllm.v1.attention.backends.psp_kv_attn._get_psp_kv_ops")
def test_psp_kv_attention_impl_forward_dispatch_btkv(mock_get_ops, _mock_capturing):
    layer = DummyLayer()
    query = torch.randn(1, 8, 128)
    kv_cache = torch.zeros(2, 10, 8, 16, 128, dtype=torch.uint8)
    tma_descriptors = torch.zeros(10, 16, dtype=torch.uint64)
    meta = PSPKVAttentionMetadata(
        common_metadata=DummyMetadata(
            format_generation=2,
            precision_table=torch.zeros(10, dtype=torch.uint8),
            block_offsets=torch.zeros(10, dtype=torch.int32),
            scale_table=torch.zeros(10, 8, dtype=torch.float32),
            tma_descriptors=tma_descriptors,
            block_table_tensor=torch.zeros(1, 10, dtype=torch.int32),
            context_lens=torch.zeros(1, dtype=torch.int32),
            num_actual_tokens=1,
            num_reqs=1,
        )
    )
    output = torch.empty_like(query)
    mock_ops = _make_ops_mock()
    mock_ops.psp_attention_btkv = MagicMock()
    mock_get_ops.return_value = mock_ops

    impl = _make_impl()
    impl.forward(
        layer=layer,
        query=query,
        key=torch.empty(0),
        value=torch.empty(0),
        kv_cache=kv_cache,
        attn_metadata=meta,
        output=output,
    )

    mock_ops.psp_attention_btkv.assert_called_once()
    args, _ = mock_ops.psp_attention_btkv.call_args
    assert args[6] is meta.common_metadata.block_offsets
    assert args[7] is meta.common_metadata.scale_table
    assert args[8] is meta.common_metadata.tma_descriptors


@patch("torch.cuda.is_current_stream_capturing", return_value=False)
def test_psp_kv_attention_impl_forward_prefill_computes_attention(_mock_capturing):
    layer = DummyLayer()
    query = torch.randn(2, 8, 128)
    key = torch.randn(2, 8, 128)
    value = torch.randn(2, 8, 128)
    kv_cache = torch.zeros(2, 10, 8, 16, 128, dtype=torch.uint8)
    # num_actual_tokens > num_reqs triggers prefill detection
    meta = PSPKVAttentionMetadata(
        common_metadata=DummyMetadata(
            format_generation=1,
            precision_table=torch.zeros(10, dtype=torch.uint8),
            block_offsets=torch.zeros(10, dtype=torch.int32),
            scale_table=torch.zeros(10, 8, dtype=torch.float32),
            block_table_tensor=torch.zeros(1, 10, dtype=torch.int32),
            context_lens=torch.zeros(1, dtype=torch.int32),
            num_actual_tokens=2,
            num_reqs=1,
            query_start_loc_cpu=torch.tensor([0, 2], dtype=torch.int64),
        )
    )
    output = torch.empty_like(query)

    impl = _make_impl()
    # Prefill computes attention directly from Q/K/V (no NotImplementedError).
    impl.forward(
        layer=layer,
        query=query,
        key=key,
        value=value,
        kv_cache=kv_cache,
        attn_metadata=meta,
        output=output,
    )
    assert output.shape == query.shape
    assert not torch.isnan(output).any()


@patch("torch.cuda.is_current_stream_capturing", return_value=False)
def test_psp_kv_attention_impl_forward_btkv_missing_tma_raises(_mock_capturing):
    layer = DummyLayer()
    query = torch.randn(1, 8, 128)
    kv_cache = torch.zeros(2, 10, 8, 16, 128, dtype=torch.uint8)
    meta = PSPKVAttentionMetadata(
        common_metadata=DummyMetadata(
            format_generation=2,
            precision_table=torch.zeros(10, dtype=torch.uint8),
            block_offsets=torch.zeros(10, dtype=torch.int32),
            scale_table=torch.zeros(10, 8, dtype=torch.float32),
            tma_descriptors=None,  # Missing TMA descriptors
            block_table_tensor=torch.zeros(1, 10, dtype=torch.int32),
            context_lens=torch.zeros(1, dtype=torch.int32),
            num_actual_tokens=1,
            num_reqs=1,
        )
    )
    output = torch.empty_like(query)

    impl = _make_impl()
    with pytest.raises(RuntimeError, match="BT-KV format requires TMA descriptors"):
        impl.forward(
            layer=layer,
            query=query,
            key=torch.empty(0),
            value=torch.empty(0),
            kv_cache=kv_cache,
            attn_metadata=meta,
            output=output,
        )


@patch("torch.cuda.is_current_stream_capturing", return_value=False)
@patch("vllm.v1.attention.backends.psp_kv_attn._get_psp_kv_ops")
def test_psp_kv_attention_impl_forward_uses_impl_scale(mock_get_ops, _mock_capturing):
    class DummyLayerNoScale:
        num_kv_heads = 8
        num_kv_heads_per_partition = 8
        head_size = 128
        # NOTE: the Attention layer has no `scale` attribute; the softmax
        # scale lives on the impl (set from the constructor).

    layer = DummyLayerNoScale()
    query = torch.randn(1, 8, 128)
    kv_cache = torch.zeros(2, 10, 8, 16, 128, dtype=torch.uint8)
    meta = PSPKVAttentionMetadata(
        common_metadata=DummyMetadata(
            format_generation=0,
            precision_table=torch.zeros(10, dtype=torch.uint8),
            block_offsets=torch.zeros(10, dtype=torch.int32),
            block_table_tensor=torch.zeros(1, 10, dtype=torch.int32),
            context_lens=torch.zeros(1, dtype=torch.int32),
            num_actual_tokens=1,
            num_reqs=1,
        )
    )
    output = torch.empty_like(query)
    mock_ops = _make_ops_mock()
    mock_ops.psp_dequantize_kv = MagicMock()
    mock_ops.psp_attention_uniform = MagicMock()
    mock_get_ops.return_value = mock_ops

    impl = _make_impl()  # scale=0.125

    impl.forward(
        layer=layer,
        query=query,
        key=torch.empty(0),
        value=torch.empty(0),
        kv_cache=kv_cache,
        attn_metadata=meta,
        output=output,
    )

    mock_ops.psp_attention_uniform.assert_called_once()
    args, _ = mock_ops.psp_attention_uniform.call_args
    assert args[5] == impl.scale


def test_psp_kv_metadata_builder_build_accepts_kwargs():
    builder = PSPKVAttentionMetadataBuilder(
        kv_cache_spec=MagicMock(),
        layer_names=["layer"],
        vllm_config=MagicMock(),
        device=torch.device("cpu"),
    )
    # Extra kwargs should be accepted
    meta = builder.build(
        common_prefix_len=0,
        common_attn_metadata=DummyMetadata(),
        fast_build=False,
        extra_arg=123,
    )
    assert isinstance(meta, PSPKVAttentionMetadata)


@patch("torch.cuda.is_current_stream_capturing", return_value=True)
@patch("vllm.v1.attention.backends.psp_kv_attn._get_psp_kv_ops")
def test_psp_kv_attention_impl_forward_skips_prefill_check_during_capture(
    mock_get_ops, mock_capturing
):
    # Even if prefill condition is met, during capture the check is skipped.
    layer = DummyLayer()
    query = torch.randn(1, 8, 128)
    kv_cache = torch.zeros(2, 10, 8, 16, 128, dtype=torch.uint8)
    meta = PSPKVAttentionMetadata(
        common_metadata=DummyMetadata(
            format_generation=0,
            precision_table=torch.zeros(10, dtype=torch.uint8),
            block_offsets=torch.zeros(10, dtype=torch.int32),
            block_table_tensor=torch.zeros(1, 10, dtype=torch.int32),
            context_lens=torch.zeros(1, dtype=torch.int32),
            num_actual_tokens=2,
            num_reqs=1,
        )
    )
    output = torch.empty_like(query)

    mock_ops = _make_ops_mock()
    mock_ops.psp_dequantize_kv = MagicMock()
    mock_ops.psp_attention_uniform = MagicMock()
    mock_get_ops.return_value = mock_ops

    impl = _make_impl()
    # Should not raise NotImplementedError
    impl.forward(
        layer=layer,
        query=query,
        key=torch.empty(0),
        value=torch.empty(0),
        kv_cache=kv_cache,
        attn_metadata=meta,
        output=output,
    )
    mock_ops.psp_attention_uniform.assert_called_once()
