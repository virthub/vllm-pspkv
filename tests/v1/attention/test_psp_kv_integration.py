# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock
import pytest
import torch

from vllm.v1.attention.backend import CommonAttentionMetadata
from vllm.v1.attention.backends.psp_kv_attn import (
    PSPKVAttentionImpl,
    PSPKVAttentionMetadata,
)


def _get_ops():
    """Resolve the PSP-KV op namespace registered by the _C extension."""
    import vllm._C  # noqa: F401  (registers torch.ops.psp_kv)
    return getattr(torch.ops, "psp_kv", None)


def _make_byte_cache(precision_table, num_kv_heads, block_size, head_dim, dtype, device):
    """Build a variable-stride byte-buffer cache + block offsets."""
    from vllm.v1.attention.psp_kv_virthub import block_bytes, compute_block_offsets

    scalar_size = 2 if dtype in (torch.float16, torch.bfloat16) else 1
    precisions = precision_table.tolist()
    offsets = compute_block_offsets(precisions, num_kv_heads, block_size, head_dim, scalar_size)
    total = sum(block_bytes(p, num_kv_heads, block_size, head_dim, scalar_size) for p in precisions)
    # 0x3C is finite in fp16/bf16 (lossless), e4m3, and e5m2 (1.5 in e4m3)
    k_buf = torch.full((total,), 0x3C, dtype=torch.uint8, device=device)
    v_buf = torch.full((total,), 0x3C, dtype=torch.uint8, device=device)
    kv_cache = torch.stack([k_buf, v_buf])
    block_offsets = torch.tensor(offsets, dtype=torch.int32, device=device)
    return kv_cache, block_offsets


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for integration tests")
class TestPSPKVIntegration:

    @pytest.fixture(autouse=True)
    def setup_method(self):
        self.ops = _get_ops()
        if self.ops is None or not hasattr(self.ops, "psp_attention_enhanced"):
            pytest.skip("PSP-KV custom ops are not compiled or registered.")

        # Fixed model parameters for decode tests
        self.batch_size = 4
        self.num_heads = 8
        self.num_kv_heads = 2
        self.head_dim = 128
        self.block_size = 16
        self.max_blocks_per_seq = 8
        self.max_num_blocks = 64
        self.dtype = torch.bfloat16
        self.device = torch.device("cuda:0")
        self.softmax_scale = float(1.0 / (self.head_dim ** 0.5))

    def _create_static_decode_environment(self):
        """
        Allocate static GPU buffers matching vLLM V1 GPUModelRunner specifications.
        """
        # Static query & output buffers
        query = torch.randn(
            (self.batch_size, self.num_heads, self.head_dim),
            dtype=self.dtype,
            device=self.device,
        )
        output = torch.zeros_like(query)

        # Static metadata tables
        block_tables = torch.zeros(
            (self.batch_size, self.max_blocks_per_seq),
            dtype=torch.int32,
            device=self.device,
        )
        context_lens = torch.zeros(self.batch_size, dtype=torch.int32, device=self.device)
        precision_table = torch.zeros(self.max_num_blocks, dtype=torch.uint8, device=self.device)
        scale_table = torch.ones(
            (self.max_num_blocks, self.num_kv_heads),
            dtype=torch.float32,
            device=self.device,
        )
        kv_cache, block_offsets = _make_byte_cache(
            precision_table, self.num_kv_heads, self.block_size, self.head_dim,
            self.dtype, self.device,
        )

        # Populate realistic block mapping and sequence lengths
        for seq_idx in range(self.batch_size):
            allocated_blocks = 4
            context_lens[seq_idx] = allocated_blocks * self.block_size
            block_tables[seq_idx, :allocated_blocks] = torch.arange(
                seq_idx * allocated_blocks,
                (seq_idx + 1) * allocated_blocks,
                dtype=torch.int32,
                device=self.device,
            )

        # Construct layer mock
        layer = MagicMock()
        layer.num_kv_heads = self.num_kv_heads
        layer.num_kv_heads_per_partition = self.num_kv_heads
        layer.head_size = self.head_dim
        layer.scale = self.softmax_scale

        # Construct common metadata mock
        common_meta = MagicMock(spec=CommonAttentionMetadata)
        common_meta.block_tables = block_tables
        common_meta.context_lens = context_lens
        common_meta.precision_table = precision_table
        common_meta.scale_table = scale_table
        common_meta.block_offsets = block_offsets
        common_meta.tma_descriptors = None
        common_meta.format_generation = 1  # Enhanced format
        common_meta.num_actual_tokens = self.batch_size
        common_meta.num_reqs = self.batch_size

        attn_metadata = PSPKVAttentionMetadata(common_metadata=common_meta)

        return layer, query, kv_cache, attn_metadata, output

    def test_eager_multi_step_decode(self):
        """
        Verify that sequential decode steps execute stably in eager mode
        as context lengths monotonically increase.
        """
        layer, query, kv_cache, metadata, output = self._create_static_decode_environment()
        impl = PSPKVAttentionImpl(self.num_heads, self.head_dim, self.softmax_scale, self.num_kv_heads, None, None, "auto")
        context_lens = metadata.common_metadata.context_lens

        num_decode_steps = 10
        for step in range(num_decode_steps):
            query.normal_()
            impl.forward(
                layer=layer,
                query=query,
                key=None,
                value=None,
                kv_cache=kv_cache,
                attn_metadata=metadata,
                output=output,
            )

            assert not torch.isnan(output).any(), f"NaN detected at decode step {step}"
            assert not torch.isinf(output).any(), f"Inf detected at decode step {step}"

            # Advance context length by 1 token per request
            context_lens.add_(1)

        torch.cuda.synchronize()

    def test_cuda_graph_capture_and_replay(self):
        """
        Test that FULL_DECODE_ONLY CUDA Graph captures without graph breaks,
        and replaying the graph produces outputs matching eager execution.
        """
        layer, query, kv_cache, metadata, output = self._create_static_decode_environment()
        impl = PSPKVAttentionImpl(self.num_heads, self.head_dim, self.softmax_scale, self.num_kv_heads, None, None, "auto")

        # 1. Warmup step (crucial for PyTorch allocator and lazy CUDA context initialization)
        warmup_stream = torch.cuda.Stream()
        warmup_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warmup_stream):
            for _ in range(3):
                impl.forward(
                    layer=layer,
                    query=query,
                    key=None,
                    value=None,
                    kv_cache=kv_cache,
                    attn_metadata=metadata,
                    output=output,
                )
        torch.cuda.current_stream().wait_stream(warmup_stream)

        # 2. Capture CUDA Graph
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            impl.forward(
                layer=layer,
                query=query,
                key=None,
                value=None,
                kv_cache=kv_cache,
                attn_metadata=metadata,
                output=output,
            )

        # 3. Test graph replay with updated input data
        # Note: In CUDA Graphs, inputs must be updated in-place into the captured memory buffers
        test_query = torch.randn_like(query)
        query.copy_(test_query)

        graph.replay()
        torch.cuda.synchronize()
        graph_output = output.clone()

        # 4. Run eager baseline with identical query
        eager_output = torch.empty_like(output)
        impl.forward(
            layer=layer,
            query=test_query,
            key=None,
            value=None,
            kv_cache=kv_cache,
            attn_metadata=metadata,
            output=eager_output,
        )
        torch.cuda.synchronize()

        # Check exact numerical parity between Graph replay and Eager execution
        max_diff = torch.max(torch.abs(graph_output - eager_output)).item()
        assert max_diff < 1e-4, f"Mismatch between Graph replay and Eager execution: max diff {max_diff}"

    def test_cuda_graph_dynamic_precision_mutation(self):
        """
        Simulate cache memory pressure during decode:
        Mutate block precision and scale tables in-place on the GPU,
        then replay the captured graph to ensure memory updates take immediate effect.
        """
        layer, query, kv_cache, metadata, output = self._create_static_decode_environment()
        impl = PSPKVAttentionImpl(self.num_heads, self.head_dim, self.softmax_scale, self.num_kv_heads, None, None, "auto")
        precision_table = metadata.common_metadata.precision_table
        scale_table = metadata.common_metadata.scale_table

        # Warmup and capture
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            impl.forward(
                layer=layer,
                query=query,
                key=None,
                value=None,
                kv_cache=kv_cache,
                attn_metadata=metadata,
                output=output,
            )

        # Baseline run with precision 0
        graph.replay()
        torch.cuda.synchronize()
        base_output = output.clone()

        # Dynamically compress cache: set active blocks to precision level 1 (FP8 compressed)
        # and adjust scales via in-place mutation (simulating _sync_dirty_blocks)
        precision_table[:16].fill_(1)
        scale_table[:16].mul_(0.85)

        # Replay graph without recapturing
        graph.replay()
        torch.cuda.synchronize()
        mutated_output = output.clone()

        # Verify output changed as a direct result of the precision & scale mutations
        diff = torch.max(torch.abs(mutated_output - base_output)).item()
        assert diff > 1e-4, "Graph replay output did not react to precision and scale table mutations!"
        assert not torch.isnan(mutated_output).any(), "Graph replay produced NaN after precision mutation"