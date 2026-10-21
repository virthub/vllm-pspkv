# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Benchmark decode attention paths for the variable-width arena on 910B3.

Measures one decode step (single query token over a 256-token context) for
the eager reference path and the native packed-kernel path, with Qwen3-0.6B
attention geometry (16 query heads, 8 KV heads, head_dim 128, 28 layers).

Run inside the vLLM-Ascend image with PSPKV_ASCEND_NATIVE_LIB set.
"""

import time

import torch
import torch_npu  # noqa: F401 - registers the NPU device with PyTorch

from vllm.v1.attention.backends.psp_kv_ascend_arena import (
    VariableArenaKVCache,
    reference_variable_attention,
)

BLOCK = 128
HEADS = 16
KV_HEADS = 8
DIM = 128
LAYERS = 28
CONTEXT = 256
STEPS = 20
WARMUP_STEPS = 3
DEVICE = torch.device("npu:0")


def _cpu_randn(*shape: int, seed: int) -> torch.Tensor:
    torch.manual_seed(seed)
    return torch.randn(*shape).bfloat16().to(DEVICE)


def _build() -> tuple[VariableArenaKVCache, torch.Tensor]:
    block_bytes = BLOCK * KV_HEADS * DIM
    num_blocks = CONTEXT // BLOCK + 2
    arena = VariableArenaKVCache(
        num_blocks=num_blocks,
        block_size=BLOCK,
        num_kv_heads=KV_HEADS,
        head_size=DIM,
        device=DEVICE,
        k_buffer=torch.zeros(
            num_blocks * block_bytes, dtype=torch.uint8, device=DEVICE
        ),
        v_buffer=torch.zeros(
            num_blocks * block_bytes, dtype=torch.uint8, device=DEVICE
        ),
        avg_bytes=1.0,
        scalar_dtype=torch.bfloat16,
        max_relative_rmse=0.05,
        layer_name="bench.layer",
    )
    key = _cpu_randn(CONTEXT, KV_HEADS, DIM, seed=11)
    value = _cpu_randn(CONTEXT, KV_HEADS, DIM, seed=12)
    slots = torch.arange(CONTEXT, dtype=torch.int64, device=DEVICE)
    arena.begin_writes(slots, {0, 1})
    arena.write_and_finalize(slots, key, value)
    tables = torch.tensor(list(range(num_blocks)), dtype=torch.int32, device=DEVICE)[
        None
    ]
    return arena, tables


def _time_path(native: bool, native_fused: bool = False) -> float:
    # Each layer owns its own cache. Every decode step writes a new token;
    # repeating the same physical slot would not model vLLM's cache lifetime.
    built = [_build() for _ in range(LAYERS)]
    arenas = [arena for arena, _ in built]
    tables = built[0][1]
    slots = [
        torch.tensor([CONTEXT + step], dtype=torch.int64, device=DEVICE)
        for step in range(WARMUP_STEPS + STEPS)
    ]
    keys = _cpu_randn(LAYERS, KV_HEADS, DIM, seed=31)
    values = _cpu_randn(LAYERS, KV_HEADS, DIM, seed=32)
    queries = _cpu_randn(LAYERS, HEADS, DIM, seed=33)
    output = torch.zeros(1, HEADS, DIM, dtype=torch.bfloat16, device=DEVICE)

    def step(step_index: int) -> None:
        context = CONTEXT + step_index + 1
        for layer in range(LAYERS):
            reference_variable_attention(
                queries[layer : layer + 1],
                keys[layer : layer + 1],
                values[layer : layer + 1],
                arenas[layer],
                slots[step_index],
                tables,
                [1],
                [context],
                output,
                0.125,
                native=native,
                native_fused=native_fused,
            )

    for step_index in range(WARMUP_STEPS):
        step(step_index)
    torch.npu.synchronize()
    start = time.perf_counter()
    for step_index in range(WARMUP_STEPS, WARMUP_STEPS + STEPS):
        step(step_index)
    torch.npu.synchronize()
    return (time.perf_counter() - start) / STEPS


def main() -> None:
    torch.npu.set_device(0)
    results = {}
    for native, native_fused, label in (
        (False, False, "reference"),
        (True, False, "native packed kernels"),
        (True, True, "native fused decode"),
    ):
        decode_seconds = _time_path(native, native_fused)
        results[label] = decode_seconds
        print(
            f"{label:24s} {decode_seconds * 1e3:8.2f} ms/step "
            f"({LAYERS} layers) -> {1.0 / decode_seconds:6.1f} steps/s"
        )
    speedup = results["reference"] / results["native packed kernels"]
    print(f"native speedup over reference: {speedup:.2f}x")
    fused_ratio = results["native packed kernels"] / results["native fused decode"]
    print(f"fused speedup over separate native kernels: {fused_ratio:.2f}x")


if __name__ == "__main__":
    main()
