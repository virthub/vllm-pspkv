# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Compare the online variable-width arena and attention on CPU and 910B3."""

import torch
import torch_npu  # noqa: F401 - registers the NPU device with PyTorch

from vllm.v1.attention.backends.psp_kv_ascend_arena import (
    VariableArenaKVCache,
    reference_variable_attention,
)


def _arena(
    device: torch.device,
    num_blocks: int = 4,
    block_size: int = 4,
    num_kv_heads: int = 2,
    head_size: int = 8,
    avg_bytes: float = 0.75,
    fp4_group_size: int = 0,
    max_relative_rmse: float = 0.1,
) -> VariableArenaKVCache:
    block_bytes = block_size * num_kv_heads * head_size
    size = int(num_blocks * block_bytes * avg_bytes)
    return VariableArenaKVCache(
        num_blocks=num_blocks,
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        device=device,
        k_buffer=torch.zeros(size, dtype=torch.uint8, device=device),
        v_buffer=torch.zeros(size, dtype=torch.uint8, device=device),
        avg_bytes=avg_bytes,
        scalar_dtype=torch.bfloat16,
        fp4_group_size=fp4_group_size,
        max_relative_rmse=max_relative_rmse,
    )


def _cpu_randn(*shape: int, seed: int, device: torch.device) -> torch.Tensor:
    """Random BF16 data that is bit-identical on CPU and NPU."""
    torch.manual_seed(seed)
    return torch.randn(*shape).bfloat16().to(device)


def run_mixed_lifetime(device: torch.device) -> tuple[torch.Tensor, ...]:
    """Exact FP4 block, random E4M3 block, tail shrink, reuse, and reads."""
    arena = _arena(device)
    exact = (
        torch.tensor(
            [[0.0, 1.0, -2.0, 3.0, 0.5, -1.5, 2.0, -3.0]] * 2,
            dtype=torch.bfloat16,
        )
        .repeat(4, 1, 1)
        .to(device)
    )
    random = _cpu_randn(4, 2, 8, seed=11, device=device)

    slots0 = torch.arange(0, 4, dtype=torch.int64, device=device)
    arena.begin_writes(slots0, {0})
    arena.write_and_finalize(slots0, exact, -exact)
    slots1 = torch.arange(4, 8, dtype=torch.int64, device=device)
    arena.begin_writes(slots1, {0, 1})
    arena.write_and_finalize(slots1, random, random)
    read_exact, read_random = arena.gather_blocks([0, 1])

    # Reuse block 0 from slot zero: a fresh lifetime.
    fresh = _cpu_randn(4, 2, 8, seed=13, device=device)
    arena.begin_writes(slots0, {0, 1})
    arena.write_and_finalize(slots0, fresh, fresh)
    read_fresh, _ = arena.gather_blocks([0])

    return (
        torch.tensor(arena._codes, dtype=torch.int32),
        torch.tensor(arena._offsets, dtype=torch.int64),
        torch.tensor(arena._sizes, dtype=torch.int64),
        arena.k_buffer.cpu().clone(),
        arena.v_buffer.cpu().clone(),
        arena.scales.cpu().clone(),
        read_exact.cpu().clone(),
        read_random.cpu().clone(),
        read_fresh.cpu().clone(),
    )


def run_compaction(device: torch.device) -> tuple[torch.Tensor, ...]:
    """Compaction moves live blocks when a hole would cause overflow."""
    arena = _arena(device, num_blocks=4, avg_bytes=0.75, max_relative_rmse=0.05)
    torch.manual_seed(23)
    key = (torch.rand(4, 2, 8) * 0.2).bfloat16().to(device)
    for block_id in range(3):
        slots = torch.arange(
            block_id * 4, (block_id + 1) * 4, dtype=torch.int64, device=device
        )
        arena.begin_writes(slots, set(range(block_id + 1)))
        arena.write_and_finalize(slots, key, key)
    slots = torch.arange(12, 16, dtype=torch.int64, device=device)
    arena.begin_writes(slots, {1, 2, 3})
    arena.write_and_finalize(slots, key, key)
    read, _ = arena.gather_blocks([1, 2, 3])
    return (
        torch.tensor(arena._offsets, dtype=torch.int64),
        arena.k_buffer.cpu().clone(),
        arena.v_buffer.cpu().clone(),
        read.cpu().clone(),
    )


def run_variable_attention(device: torch.device) -> tuple[torch.Tensor, ...]:
    """Prefill and decode over mixed-format pages on the selected device."""
    arena = _arena(device, max_relative_rmse=0.1)
    block_tables = torch.tensor([[0, 1, -1, -1]], dtype=torch.int32, device=device)

    key = _cpu_randn(6, 2, 8, seed=31, device=device)
    value = _cpu_randn(6, 2, 8, seed=37, device=device)
    query = _cpu_randn(6, 2, 8, seed=41, device=device)
    output = torch.zeros(6, 2, 8, device=device)
    slots = torch.tensor([0, 1, 2, 3, 4, 5], device=device)
    reference_variable_attention(
        query, key, value, arena, slots, block_tables, [6], [6], output, 0.1
    )

    k_new = _cpu_randn(1, 2, 8, seed=43, device=device)
    v_new = _cpu_randn(1, 2, 8, seed=47, device=device)
    q_new = _cpu_randn(1, 2, 8, seed=53, device=device)
    decode_out = torch.zeros(1, 2, 8, device=device)
    reference_variable_attention(
        q_new,
        k_new,
        v_new,
        arena,
        torch.tensor([6], device=device),
        block_tables,
        [1],
        [7],
        decode_out,
        0.1,
    )
    return (
        output.cpu().clone(),
        decode_out.cpu().clone(),
        arena.k_buffer.cpu().clone(),
        arena.v_buffer.cpu().clone(),
        arena.scales.cpu().clone(),
    )


def main() -> None:
    torch.npu.set_device(0)
    for cpu_result, npu_result in zip(
        run_mixed_lifetime(torch.device("cpu")),
        run_mixed_lifetime(torch.device("npu:0")),
    ):
        torch.testing.assert_close(npu_result, cpu_result, rtol=0, atol=0)
    for cpu_result, npu_result in zip(
        run_compaction(torch.device("cpu")),
        run_compaction(torch.device("npu:0")),
    ):
        torch.testing.assert_close(npu_result, cpu_result, rtol=0, atol=0)
    cpu_attn = run_variable_attention(torch.device("cpu"))
    npu_attn = run_variable_attention(torch.device("npu:0"))
    for index, (cpu_result, npu_result) in enumerate(zip(cpu_attn, npu_attn)):
        if index < 2:
            # bmm/softmax may differ by a few ulps between CPU and NPU math
            # libraries; the storage bytes and scales must still match.
            torch.testing.assert_close(npu_result, cpu_result, rtol=1e-6, atol=1e-6)
        else:
            torch.testing.assert_close(npu_result, cpu_result, rtol=0, atol=0)
    print("NPU variable-width arena, compaction and attention match CPU")


if __name__ == "__main__":
    main()
