# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Compare variable-stride FP8/FP4 KV bytes and reads on CPU and 910B3."""

import torch
import torch_npu  # noqa: F401 - registers the NPU device with PyTorch

from vllm.v1.attention.backends.psp_kv_ascend_packed import (
    PackedKVCache,
    reference_packed_attention,
)
from vllm.v1.attention.backends.psp_kv_ascend_policy import (
    pack_completed_blocks_by_error,
)


def run(device: torch.device) -> tuple[torch.Tensor, ...]:
    cache = PackedKVCache([1, 3, 2, 3], 2, 2, 4, device)
    slots = torch.tensor([6, 7, 0, 1, 2, 3, 4, 5], device=device)
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
            ],
            device=device,
        )
        .unsqueeze(1)
        .expand(-1, 2, -1)
        .clone()
    )
    rows[:, 1] *= -1
    cache.write(slots, -rows, rows)
    assert cache.block_offsets.cpu().tolist() == [0, 16, 24, 40]
    assert cache.total_bytes == 48 and cache.dense_fp8_bytes == 64
    assert cache.v_buffer[40:42].cpu().tolist() == [0x21, 0x43]
    keys, values = cache.gather([3, 0, 2, 1])
    return (
        cache.k_buffer.cpu().clone(),
        cache.v_buffer.cpu().clone(),
        keys.cpu().clone(),
        values.cpu().clone(),
    )


def run_lossless(
    device: torch.device, scalar_dtype: torch.dtype
) -> tuple[torch.Tensor, ...]:
    cache = PackedKVCache([0, 3, 3, 3, 1], 1, 1, 4, device, scalar_dtype=scalar_dtype)
    assert cache.block_offsets.cpu().tolist() == [0, 8, 10, 12, 14]
    assert cache.total_bytes == 18
    rows = torch.tensor(
        [[1.125] * 4, [1.5] * 4, [2.0] * 4, [0.5] * 4, [1.125] * 4],
        device=device,
    ).unsqueeze(1)
    cache.write(torch.arange(5, device=device), rows, -rows)
    keys, values = cache.gather([0, 3, 4])
    return (
        cache.k_buffer.cpu().clone(),
        cache.v_buffer.cpu().clone(),
        keys.cpu().clone(),
        values.cpu().clone(),
    )


def run_transposed(device: torch.device) -> tuple[torch.Tensor, ...]:
    cache = PackedKVCache([0, 1, 2, 3], 2, 1, 4, device, value_layout="transposed")
    rows = torch.tensor(
        [[0.5, 1.0, 1.5, 2.0], [2.0, 1.5, 1.0, 0.5]] * 4,
        device=device,
    ).unsqueeze(1)
    cache.write(torch.arange(8, device=device), rows, rows)
    assert cache.v_buffer[-4:].cpu().tolist() == [0x41, 0x32, 0x23, 0x14]
    keys, values = cache.gather([0, 1, 2, 3])
    return (
        cache.k_buffer.cpu().clone(),
        cache.v_buffer.cpu().clone(),
        keys.cpu().clone(),
        values.cpu().clone(),
    )


def run_copy_and_repack(device: torch.device) -> tuple[torch.Tensor, ...]:
    cache = PackedKVCache([1, 1, 1, 1], 2, 1, 4, device, value_layout="transposed")
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
        ],
        device=device,
    ).unsqueeze(1)
    cache.write(torch.arange(8, device=device), rows, -rows)
    compact = cache.repack([1, 3, 1, 3])
    assert compact.total_bytes == 24
    assert compact.block_offsets.cpu().tolist() == [0, 8, 12, 20]
    compact.copy_blocks([(0, 2), (2, 0), (1, 3), (3, 1)])
    keys, values = compact.gather([0, 1, 2, 3])
    return (
        cache.k_buffer.cpu().clone(),
        cache.v_buffer.cpu().clone(),
        compact.k_buffer.cpu().clone(),
        compact.v_buffer.cpu().clone(),
        keys.cpu().clone(),
        values.cpu().clone(),
    )


def run_uniform_fp4_attention(device: torch.device) -> tuple[torch.Tensor, ...]:
    cache = PackedKVCache([3, 3], 4, 1, 4, device)
    scales = torch.ones((2, 1), dtype=torch.float32, device=device)
    block_tables = torch.tensor([[1, 0]], dtype=torch.int32, device=device)
    initial = torch.tensor(
        [[[0.5, 1.0, 1.5, 2.0]], [[1.0, 1.0, 1.0, 1.0]]], device=device
    )
    prefill = torch.empty((2, 2, 4), device=device)
    reference_packed_attention(
        torch.zeros_like(prefill),
        initial,
        initial,
        cache,
        torch.tensor([4, 5], device=device),
        block_tables,
        [2],
        [2],
        prefill,
        0.5,
        scales,
    )
    decode = torch.empty((1, 2, 4), device=device)
    incoming = torch.full((1, 1, 4), 4.0, device=device)
    reference_packed_attention(
        torch.zeros_like(decode),
        incoming,
        incoming,
        cache,
        torch.tensor([6], device=device),
        block_tables,
        [1],
        [3],
        decode,
        0.5,
        scales,
    )
    assert scales.cpu().tolist() == [[1.0], [2.0]]
    return (
        cache.k_buffer.cpu().clone(),
        cache.v_buffer.cpu().clone(),
        scales.cpu().clone(),
        prefill.cpu().clone(),
        decode.cpu().clone(),
    )


def run_uniform_fp4_small_values(device: torch.device) -> tuple[torch.Tensor, ...]:
    cache = PackedKVCache([3], 4, 1, 4, device)
    scales = torch.ones((1, 1), dtype=torch.float32, device=device)
    output = torch.empty((1, 1, 4), device=device)
    blocks = torch.tensor([[0]], dtype=torch.int32, device=device)
    for slot, seq_len, magnitude in ((0, 1, 0.125), (1, 2, 0.75)):
        incoming = torch.full((1, 1, 4), magnitude, device=device)
        reference_packed_attention(
            torch.zeros_like(output),
            incoming,
            incoming,
            cache,
            torch.tensor([slot], device=device),
            blocks,
            [1],
            [seq_len],
            output,
            0.5,
            scales,
        )
    assert scales.item() == 0.25
    return (
        cache.k_buffer.cpu().clone(),
        cache.v_buffer.cpu().clone(),
        scales.cpu().clone(),
        output.cpu().clone(),
    )


def run_quality_policy(device: torch.device) -> tuple[torch.Tensor, ...]:
    key = torch.tensor(
        [
            [[0.125, 0.25, 0.125, 0.25]] * 2,
            [[1.125, 1.125, 1.125, 1.125]] * 2,
            [[512.0, 1.125, 512.0, 1.125]] * 2,
            [[1.0078125, 1.0078125, 1.0078125, 1.0078125]] * 2,
        ],
        dtype=torch.bfloat16,
        device=device,
    ).unsqueeze(2)
    cache, scales, choices = pack_completed_blocks_by_error(key, -key, 0.001)
    assert [choice.code for choice in choices] == [3, 1, 2, 0]
    read_key, read_value = cache.gather([0, 1, 2, 3])
    return (
        cache.precision_table.cpu().clone(),
        cache.block_offsets.cpu().clone(),
        cache.k_buffer.cpu().clone(),
        cache.v_buffer.cpu().clone(),
        scales.cpu().clone(),
        (read_key * scales[:, None, :, None]).cpu().clone(),
        (read_value * scales[:, None, :, None]).cpu().clone(),
    )


def main() -> None:
    torch.npu.set_device(0)
    for cpu_result, npu_result in zip(
        run(torch.device("cpu")), run(torch.device("npu:0"))
    ):
        torch.testing.assert_close(npu_result, cpu_result, rtol=0, atol=0)
    for scalar_dtype in (torch.bfloat16, torch.float16):
        for cpu_result, npu_result in zip(
            run_lossless(torch.device("cpu"), scalar_dtype),
            run_lossless(torch.device("npu:0"), scalar_dtype),
        ):
            torch.testing.assert_close(npu_result, cpu_result, rtol=0, atol=0)
    for cpu_result, npu_result in zip(
        run_transposed(torch.device("cpu")), run_transposed(torch.device("npu:0"))
    ):
        torch.testing.assert_close(npu_result, cpu_result, rtol=0, atol=0)
    for cpu_result, npu_result in zip(
        run_copy_and_repack(torch.device("cpu")),
        run_copy_and_repack(torch.device("npu:0")),
    ):
        torch.testing.assert_close(npu_result, cpu_result, rtol=0, atol=0)
    for cpu_result, npu_result in zip(
        run_uniform_fp4_attention(torch.device("cpu")),
        run_uniform_fp4_attention(torch.device("npu:0")),
    ):
        torch.testing.assert_close(npu_result, cpu_result, rtol=0, atol=0)
    for cpu_result, npu_result in zip(
        run_uniform_fp4_small_values(torch.device("cpu")),
        run_uniform_fp4_small_values(torch.device("npu:0")),
    ):
        torch.testing.assert_close(npu_result, cpu_result, rtol=0, atol=0)
    for cpu_result, npu_result in zip(
        run_quality_policy(torch.device("cpu")),
        run_quality_policy(torch.device("npu:0")),
    ):
        torch.testing.assert_close(npu_result, cpu_result, rtol=0, atol=0)
    print("NPU mixed-format selection, packed storage and attention match CPU")


if __name__ == "__main__":
    main()
