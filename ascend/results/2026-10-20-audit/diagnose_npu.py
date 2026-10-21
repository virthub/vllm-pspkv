# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Supplemental probes for checks interrupted by the two recorded failures."""

import importlib.util
import json

import torch
import torch_npu  # noqa: F401

from vllm.v1.attention.backends.psp_kv_ascend_arena import VariableArenaKVCache
from vllm.v1.attention.backends.psp_kv_ascend_packed import (
    PackedKVCache,
    reference_packed_attention,
)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def scales(device):
    cache = PackedKVCache([3], 4, 1, 4, device)
    table = torch.ones((1, 1), device=device)
    output = torch.empty((1, 1, 4), device=device)
    blocks = torch.tensor([[0]], dtype=torch.int32, device=device)
    trace = []
    for slot, magnitude in ((0, 0.125), (1, 0.75)):
        incoming = torch.full((1, 1, 4), magnitude, device=device)
        reference_packed_attention(
            torch.zeros_like(output),
            incoming,
            incoming,
            cache,
            torch.tensor([slot], device=device),
            blocks,
            [1],
            [slot + 1],
            output,
            0.5,
            table,
        )
        trace.append(
            {
                "magnitude": magnitude,
                "scale": table.item(),
                "output": output.cpu().tolist(),
            }
        )
    quarter = torch.tensor([0.25], device=device)
    return {
        "trace": trace,
        "log2_quarter": torch.log2(quarter).item(),
        "ceil_log2_quarter": torch.ceil(torch.log2(quarter)).item(),
        "pow2_minus2": torch.pow(2.0, torch.tensor([-2.0], device=device)).item(),
    }


def main():
    torch.npu.set_device(0)
    result = {
        "fp4_scales_cpu": scales(torch.device("cpu")),
        "fp4_scales_npu": scales(torch.device("npu:0")),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)
    root = "/workspace/pspkv-src/ascend/"
    packed = load("packed_checker", root + "check_packed_reference_npu.py")
    result["completed_block_policy_comparisons"] = [
        {
            "tensor": name,
            "exact": torch.equal(cpu, npu),
            "max_abs_error": (cpu.float() - npu.float()).abs().max().item(),
        }
        for name, cpu, npu in zip(
            ("precision", "offsets", "k_bytes", "v_bytes", "scales", "k", "v"),
            packed.run_quality_policy(torch.device("cpu")),
            packed.run_quality_policy(torch.device("npu:0")),
        )
    ]
    native = load("native_checker", root + "check_variable_native_npu.py")
    device = torch.device("npu:0")
    per_head = torch.full((2,), 0.25, device=device)
    grouped = torch.stack(
        (
            torch.full((2, 4), 0.25, device=device),
            torch.full((2, 4), 1.0, device=device),
        )
    )
    for label, table in (("static_fp4", per_head), ("grouped_static_fp4", grouped)):
        a = native._arena(static_code=3, static_scales=table)
        b = native._arena(static_code=3, static_scales=table)
        native._fill(a, 256)
        native._fill(b, 256)
        try:
            torch.testing.assert_close(
                native.run_decode(a, 256, native=True),
                native.run_decode(b, 256, native=False),
                rtol=1e-4,
                atol=1e-4,
            )
            result[label + "_native_reference_parity"] = True
        except Exception as exc:
            result[label + "_native_reference_parity"] = str(exc)
    block_bytes = 128 * 2 * 128

    def asymmetric():
        return VariableArenaKVCache(
            num_blocks=6,
            block_size=128,
            num_kv_heads=2,
            head_size=128,
            device=device,
            k_buffer=torch.zeros(6 * block_bytes, dtype=torch.uint8, device=device),
            v_buffer=torch.zeros(3 * block_bytes, dtype=torch.uint8, device=device),
            avg_bytes=1.0,
            v_avg_bytes=0.5,
            scalar_dtype=torch.bfloat16,
            static_code=1,
            static_v_code=3,
            static_scales=grouped,
        )

    a, b = asymmetric(), asymmetric()
    native._fill(a, 256)
    native._fill(b, 256)
    torch.testing.assert_close(
        native.run_decode(a, 256, native=True),
        native.run_decode(b, 256, native=False),
        rtol=1e-4,
        atol=1e-4,
    )
    result["asymmetric_native_reference_parity"] = True
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
