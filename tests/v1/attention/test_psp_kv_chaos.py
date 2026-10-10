# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import MagicMock
import random
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
    """Build a variable-stride byte-buffer cache + block offsets for stress tests."""
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


class MockGPUModelRunnerSyncHarness:
    """
    Simulates the chunked synchronization engine in GPUModelRunner (Section 7.2)
    to stress-test batch transfers and scale validity flags under chaos conditions.
    """

    def __init__(self, max_num_blocks: int, num_kv_heads: int, device: torch.device):
        self.max_num_blocks = max_num_blocks
        self.num_kv_heads = num_kv_heads
        self.device = device
        self.max_dirty_blocks = 256

        # GPU static tables
        self.precision_table = torch.zeros(self.max_num_blocks, dtype=torch.uint8, device=self.device)
        self.scale_table = torch.ones(self.max_num_blocks, self.num_kv_heads, dtype=torch.float32, device=self.device)

        # Host pinned staging buffers
        self.h_precision_table = torch.zeros(self.max_num_blocks, dtype=torch.uint8, pin_memory=True)
        self.h_scale_table = torch.ones(self.max_num_blocks, self.num_kv_heads, dtype=torch.float32, pin_memory=True)
        self.h_has_scale = [False] * self.max_num_blocks

        # Sized staging buffers (sized to max_dirty_blocks per chunk)
        self.h_dirty_indices = torch.empty(self.max_dirty_blocks, dtype=torch.long, pin_memory=True)
        self.d_dirty_indices = torch.empty(self.max_dirty_blocks, dtype=torch.long, device=self.device)

        self.dirty_block_ids: list[int] = []
        self.is_dirty = [False] * self.max_num_blocks

    def update_block_metadata(self, block_id: int, precision_level: int, scale: list[float] | None):
        self.h_precision_table[block_id] = precision_level
        if scale is not None:
            self.h_scale_table[block_id] = torch.as_tensor(scale, dtype=torch.float32)
            self.h_has_scale[block_id] = True
        else:
            self.h_has_scale[block_id] = False

        if not self.is_dirty[block_id]:
            self.is_dirty[block_id] = True
            self.dirty_block_ids.append(block_id)

    def sync_dirty_blocks(self):
        if not self.dirty_block_ids:
            return

        current_stream = torch.cuda.current_stream(device=self.device)
        with torch.cuda.stream(current_stream):
            for i in range(0, len(self.dirty_block_ids), self.max_dirty_blocks):
                chunk = self.dirty_block_ids[i:i + self.max_dirty_blocks]
                k = len(chunk)

                # 1. Pinned host copy
                self.h_dirty_indices[:k] = torch.as_tensor(chunk, dtype=torch.long)

                # 2. Async H2D indices transfer
                self.d_dirty_indices[:k].copy_(self.h_dirty_indices[:k], non_blocking=True)
                active_indices = self.d_dirty_indices[:k]

                # 3. Transfer precision
                h_prec_batch = self.h_precision_table[self.h_dirty_indices[:k]].to(self.device, non_blocking=True)
                self.precision_table.index_copy_(0, active_indices, h_prec_batch)

                # 4. Selective scale transfer only when scale was explicitly provided
                scale_indices = [b for b in chunk if self.h_has_scale[b]]
                if scale_indices:
                    s_chunk = torch.as_tensor(scale_indices, dtype=torch.long)
                    h_scale_batch = self.h_scale_table[s_chunk].to(self.device, non_blocking=True)
                    d_s_indices = s_chunk.to(self.device, non_blocking=True)
                    self.scale_table.index_copy_(0, d_s_indices, h_scale_batch)

            # Invariant: clear dirty flags outside chunk iteration
            for b_id in self.dirty_block_ids:
                self.is_dirty[b_id] = False
                self.h_has_scale[b_id] = False
            self.dirty_block_ids.clear()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for chaos tests")
class TestPSPKVChaos:

    @pytest.fixture(autouse=True)
    def setup_method(self):
        self.ops = _get_ops()
        if self.ops is None or not hasattr(self.ops, "psp_attention_enhanced"):
            pytest.skip("PSP-KV custom ops are not compiled or registered.")
        self.device = torch.device("cuda:0")

    def test_chunked_sync_under_heavy_burst_churn(self):
        """
        Chaos Test 1: Heavy Churn Stress.
        Simulate 1200 random block updates across 2048 total blocks.
        Verify that chunking logic handles boundary rollovers, selective scale updates,
        and leaves no stale CPU/GPU sync state.
        """
        total_blocks = 2048
        num_kv_heads = 4
        harness = MockGPUModelRunnerSyncHarness(total_blocks, num_kv_heads, self.device)

        # Pre-seed GPU scales with a known marker (e.g. 5.55)
        harness.scale_table.fill_(5.55)

        # Mutate 1200 random blocks (well exceeding chunk limit of 256)
        random.seed(42)
        churn_blocks = random.sample(range(total_blocks), 1200)

        expected_precisions = {}
        expected_scales = {}

        for idx, block_id in enumerate(churn_blocks):
            prec = random.choice([0, 1])
            # Only half the blocks get an updated scale; the other half should NOT overwrite GPU
            if idx % 2 == 0:
                scale = [1.0 + (block_id % 10) * 0.1] * num_kv_heads
                harness.update_block_metadata(block_id, prec, scale)
                expected_scales[block_id] = scale[0]
            else:
                harness.update_block_metadata(block_id, prec, None)
                expected_scales[block_id] = 5.55  # Retains previous GPU value

            expected_precisions[block_id] = prec

        # Trigger sync
        harness.sync_dirty_blocks()
        torch.cuda.synchronize()

        # Validate GPU table states
        gpu_prec = harness.precision_table.cpu().tolist()
        gpu_scale = harness.scale_table.cpu().tolist()

        for block_id in churn_blocks:
            assert gpu_prec[block_id] == expected_precisions[block_id], (
                f"Precision mismatch at block {block_id}: expected {expected_precisions[block_id]}, got {gpu_prec[block_id]}"
            )
            assert abs(gpu_scale[block_id][0] - expected_scales[block_id]) < 1e-4, (
                f"Scale overwrite failure at block {block_id}: expected {expected_scales[block_id]}, got {gpu_scale[block_id][0]}"
            )

        # Verify dirty queue drained
        assert len(harness.dirty_block_ids) == 0
        assert not any(harness.is_dirty)
        assert not any(harness.h_has_scale)

    def test_extreme_cache_fragmentation_and_shuffled_tables(self):
        """
        Chaos Test 2: Extreme Cache Fragmentation.
        Construct completely non-contiguous, reverse-mapped, and randomly dispersed
        block allocation tables across sequences with jagged context lengths.
        """
        batch_size = 8
        num_heads = 8
        num_kv_heads = 2
        head_dim = 128
        block_size = 16
        max_blocks_per_seq = 16
        total_blocks = 256

        query = torch.randn((batch_size, num_heads, head_dim), dtype=torch.bfloat16, device=self.device)
        output = torch.empty_like(query)

        # Jagged context lengths
        context_lens = torch.tensor([13, 47, 80, 115, 128, 200, 241, 256], dtype=torch.int32, device=self.device)

        # Shuffled, non-contiguous block tables
        all_blocks = list(range(total_blocks))
        random.seed(1337)
        random.shuffle(all_blocks)

        block_tables = torch.zeros((batch_size, max_blocks_per_seq), dtype=torch.int32, device=self.device)
        cursor = 0
        for i in range(batch_size):
            num_blocks_needed = (int(context_lens[i].item()) + block_size - 1) // block_size
            seq_blocks = all_blocks[cursor:cursor + num_blocks_needed]
            cursor += num_blocks_needed
            block_tables[i, :num_blocks_needed] = torch.tensor(seq_blocks, dtype=torch.int32, device=self.device)

        precision_table = torch.randint(0, 2, (total_blocks,), dtype=torch.uint8, device=self.device)
        scale_table = torch.empty((total_blocks, num_kv_heads), dtype=torch.float32, device=self.device).uniform_(0.8, 1.2)
        kv_cache, block_offsets = _make_byte_cache(
            precision_table, num_kv_heads, block_size, head_dim, torch.bfloat16, self.device
        )

        layer = MagicMock()
        layer.num_kv_heads = num_kv_heads
        layer.num_kv_heads_per_partition = num_kv_heads
        layer.head_size = head_dim
        layer.scale = 1.0 / (head_dim ** 0.5)

        common_meta = MagicMock(spec=CommonAttentionMetadata)
        common_meta.block_tables = block_tables
        common_meta.context_lens = context_lens
        common_meta.precision_table = precision_table
        common_meta.scale_table = scale_table
        common_meta.block_offsets = block_offsets
        common_meta.tma_descriptors = None
        common_meta.format_generation = 1
        common_meta.num_actual_tokens = batch_size
        common_meta.num_reqs = batch_size

        metadata = PSPKVAttentionMetadata(common_metadata=common_meta)
        impl = PSPKVAttentionImpl(num_heads, head_dim, layer.scale, num_kv_heads, None, None, "auto")

        # Forward execution under extreme fragmentation
        impl.forward(
            layer=layer,
            query=query,
            key=None,
            value=None,
            kv_cache=kv_cache,
            attn_metadata=metadata,
            output=output,
        )
        torch.cuda.synchronize()

        assert not torch.isnan(output).any(), "Fragmented block lookup produced NaN values"
        assert not torch.isinf(output).any(), "Fragmented block lookup produced Inf values"

    def test_adversarial_numerical_scales(self):
        """
        Chaos Test 3: Numerical Edge Cases.
        Inject subnormal, zero, and huge scale factors to verify lack of kernel crashes,
        memory faults, or illegal memory exceptions.
        """
        batch_size = 2
        num_heads = 4
        num_kv_heads = 2
        head_dim = 128
        block_size = 16
        total_blocks = 16

        query = torch.randn((batch_size, num_heads, head_dim), dtype=torch.float16, device=self.device)
        output = torch.empty_like(query)

        block_tables = torch.zeros((batch_size, 4), dtype=torch.int32, device=self.device)
        block_tables[0] = torch.tensor([0, 1, 2, 3], dtype=torch.int32, device=self.device)
        block_tables[1] = torch.tensor([4, 5, 6, 7], dtype=torch.int32, device=self.device)

        context_lens = torch.tensor([64, 64], dtype=torch.int32, device=self.device)
        precision_table = torch.ones(total_blocks, dtype=torch.uint8, device=self.device)

        # Adversarial scales: 0.0, very small, large, and standard values
        scale_table = torch.tensor([
            [0.0, 1e-12],
            [1e-7, 0.0],
            [100.0, 50.0],
            [1.0, 1.0],
            [0.0, 0.0],
            [1e-5, 1e-5],
            [1.0, 10.0],
            [0.5, 0.5],
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
            [1.0, 1.0],
        ], dtype=torch.float32, device=self.device)
        kv_cache, block_offsets = _make_byte_cache(
            precision_table, num_kv_heads, block_size, head_dim, torch.float16, self.device
        )

        layer = MagicMock()
        layer.num_kv_heads = num_kv_heads
        layer.num_kv_heads_per_partition = num_kv_heads
        layer.head_size = head_dim
        layer.scale = 1.0 / (head_dim ** 0.5)

        common_meta = MagicMock(spec=CommonAttentionMetadata)
        common_meta.block_tables = block_tables
        common_meta.context_lens = context_lens
        common_meta.precision_table = precision_table
        common_meta.scale_table = scale_table
        common_meta.block_offsets = block_offsets
        common_meta.tma_descriptors = None
        common_meta.format_generation = 1
        common_meta.num_actual_tokens = batch_size
        common_meta.num_reqs = batch_size

        metadata = PSPKVAttentionMetadata(common_metadata=common_meta)
        impl = PSPKVAttentionImpl(num_heads, head_dim, layer.scale, num_kv_heads, None, None, "auto")

        # Must execute without hardware trap or illegal access
        impl.forward(
            layer=layer,
            query=query,
            key=None,
            value=None,
            kv_cache=kv_cache,
            attn_metadata=metadata,
            output=output,
        )
        torch.cuda.synchronize()

        # The kernel should handle zero-scale gracefully without throwing CUDA driver errors
        assert output.shape == query.shape