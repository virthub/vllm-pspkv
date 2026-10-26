# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Compare the PSP-KV FP8 reference operations on CPU and one visible NPU."""

import importlib.util
from pathlib import Path

import torch
import torch_npu  # noqa: F401 - registers the NPU device with PyTorch

from vllm.v1.attention.backends.psp_kv_ascend import (
    _REFERENCE_PRECISION_TABLES,
    PSPKVAscendBackend,
)

_REFERENCE_PATH = (
    Path(__file__).resolve().parents[1]
    / "vllm/v1/attention/backends/psp_kv_ascend_reference.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "psp_kv_ascend_reference", _REFERENCE_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
_CODEC = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_CODEC)


def _run_mixed_blocks(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    k_cache = torch.zeros((4, 2, 1, 1), dtype=torch.uint8, device=device)
    v_cache = torch.zeros_like(k_cache)
    precision_table = torch.tensor([1, 1, 1, 2], dtype=torch.uint8, device=device)
    block_tables = torch.tensor([[3, 1]], dtype=torch.int32, device=device)
    query = torch.zeros((3, 1, 1), device=device)
    key = torch.zeros_like(query)
    value = torch.full_like(query, 1.125)
    _CODEC.reference_attention(
        query,
        key,
        value,
        k_cache,
        v_cache,
        torch.tensor([6, 7, 2], device=device),
        block_tables,
        [3],
        [3],
        torch.empty_like(query),
        1.0,
        precision_table=precision_table,
    )

    output = torch.empty((1, 1, 1), device=device)
    _CODEC.reference_attention(
        torch.zeros_like(output),
        torch.zeros_like(output),
        torch.full_like(output, 2.0),
        k_cache,
        v_cache,
        torch.tensor([3], device=device),
        block_tables,
        [1],
        [4],
        output,
        1.0,
        precision_table=precision_table,
    )
    return v_cache.cpu(), output.cpu()


def _run_adaptive_blocks(
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    k_cache = torch.zeros((2, 4, 1, 1), dtype=torch.uint8, device=device)
    v_cache = torch.zeros_like(k_cache)
    precision_table = torch.ones(2, dtype=torch.uint8, device=device)
    block_tables = torch.tensor([[1, 0]], dtype=torch.int32, device=device)

    def write(slots: list[int], values: list[float], seq_len: int) -> torch.Tensor:
        query = torch.zeros((len(slots), 1, 1), device=device)
        key = torch.zeros_like(query)
        value = torch.tensor(values, device=device).reshape(-1, 1, 1)
        output = torch.empty_like(query)
        _CODEC.reference_attention(
            query,
            key,
            value,
            k_cache,
            v_cache,
            torch.tensor(slots, device=device),
            block_tables,
            [len(slots)],
            [seq_len],
            output,
            1.0,
            precision_table=precision_table,
            adaptive_e4m3_max_abs=240.0,
        )
        return output

    write([4, 5], [1.125, 1.125], 2)
    output = write([6], [300.0], 3)
    assert precision_table.cpu().tolist() == [1, 2]
    assert v_cache[1, :2, 0, 0].cpu().tolist() == [0x3C, 0x3C]
    result = (
        precision_table.cpu().clone(),
        v_cache.cpu().clone(),
        output.cpu().clone(),
    )

    # Reusing the physical block resets its format before writing new bytes.
    write([4], [1.125], 1)
    assert precision_table.cpu().tolist() == [1, 1]
    assert v_cache[1, 0, 0, 0].item() == 0x39
    return result


def _run_backend_copy(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    cache = torch.zeros((2, 3, 2, 1, 4), dtype=torch.uint8, device=device)
    cache[0, 0].fill_(0x38)
    cache[1, 0].fill_(0x40)
    cache[0, 1].fill_(0x3C)
    cache[1, 1].fill_(0x44)
    table = torch.tensor([1, 2, 1], dtype=torch.uint8, device=device)
    pointer = cache[0].data_ptr()
    _REFERENCE_PRECISION_TABLES[pointer] = table
    try:
        pairs = torch.tensor([[0, 1], [1, 0]], device=device)
        PSPKVAscendBackend.copy_blocks([cache], pairs)
        return cache.cpu().clone(), table.cpu().clone()
    finally:
        _REFERENCE_PRECISION_TABLES.pop(pointer, None)


def main() -> None:
    torch.npu.set_device(0)
    values = torch.tensor(
        [0.0, 0.5, 1.0, 1.125, -1.0, 2.0, 2**-16, 240.0],
        dtype=torch.float32,
    )
    for encoder, decoder in (
        (_CODEC.encode_e4m3, _CODEC.decode_e4m3),
        (_CODEC.encode_e5m2, _CODEC.decode_e5m2),
    ):
        cpu_bytes = encoder(values)
        npu_bytes = encoder(values.npu()).cpu()
        torch.testing.assert_close(npu_bytes, cpu_bytes, rtol=0, atol=0)
        torch.testing.assert_close(
            decoder(npu_bytes.npu()).cpu(), decoder(cpu_bytes), rtol=0, atol=0
        )

    cpu_cache, cpu_output = _run_mixed_blocks(torch.device("cpu"))
    npu_cache, npu_output = _run_mixed_blocks(torch.device("npu:0"))
    torch.testing.assert_close(npu_cache, cpu_cache, rtol=0, atol=0)
    torch.testing.assert_close(npu_output, cpu_output, rtol=1e-5, atol=1e-5)
    cpu_adaptive = _run_adaptive_blocks(torch.device("cpu"))
    npu_adaptive = _run_adaptive_blocks(torch.device("npu:0"))
    for cpu_result, npu_result in zip(cpu_adaptive, npu_adaptive):
        torch.testing.assert_close(npu_result, cpu_result, rtol=1e-5, atol=1e-5)
    for cpu_result, npu_result in zip(
        _run_backend_copy(torch.device("cpu")),
        _run_backend_copy(torch.device("npu:0")),
    ):
        torch.testing.assert_close(npu_result, cpu_result, rtol=0, atol=0)
    print("NPU FP8 codecs, adaptive transitions and backend block copy match CPU")


if __name__ == "__main__":
    main()
