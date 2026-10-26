# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Compare fused RoPE/write with the installed plugin's Triton RoPE + writer."""

import argparse
import json
from pathlib import Path

import torch
import torch_npu  # noqa: F401
from vllm_ascend.ops.triton.rope import rope_forward_triton
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

from vllm.v1.attention.backends.psp_kv_ascend_native import load_native_fp8_ops


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.npu.set_device(0)
    init_device_properties_triton()
    torch.manual_seed(20261026)
    native = load_native_fp8_ops()
    rows = []
    pages, block, heads, dim = 4, 128, 4, 128
    phases = torch.randn(512, dim // 2)
    cos_sin = torch.cat((phases.cos(), phases.sin()), dim=1)
    for generation in range(3):
        for code in (1, 2):
            codes = torch.full((pages,), code, dtype=torch.uint8, device="npu")
            offsets = (
                torch.arange(pages, dtype=torch.int32, device="npu")
                * block
                * heads
                * dim
            )
            scales = torch.ones(pages, heads, device="npu")
            for dtype in (torch.bfloat16, torch.float32):
                cache = cos_sin.to(dtype).npu()
                for pos in (0, 127, 129, 511):
                    qkv = torch.randn(1, 36, dim, device="npu").bfloat16()
                    query, key, value = qkv[:, :28], qkv[:, 28:32], qkv[:, 32:]
                    q0, k0, v0 = query.clone(), key.clone(), value.clone()
                    positions = torch.tensor([pos], dtype=torch.long, device="npu")
                    slots = torch.tensor([pos], dtype=torch.long, device="npu")
                    before_k = torch.full(
                        (pages * block * heads * dim,),
                        0x55,
                        dtype=torch.uint8,
                        device="npu",
                    )
                    before_v = torch.full_like(before_k, 0x55)
                    after_k, after_v = before_k.clone(), before_v.clone()
                    q_expected, k_expected = rope_forward_triton(
                        q0,
                        k0,
                        cos_sin_cache=cache,
                        positions=positions,
                        rope_dim=dim,
                        is_neox_style=True,
                    )
                    native.packed_scatter_parallel(
                        torch.stack((k_expected, v0), dim=1),
                        before_k,
                        before_v,
                        slots,
                        codes,
                        offsets,
                        scales,
                        block,
                        generation,
                    )
                    native.rope_scatter_kv(
                        query,
                        key,
                        value,
                        positions,
                        cache,
                        after_k,
                        after_v,
                        slots,
                        codes,
                        offsets,
                        block,
                        generation,
                    )
                    observed, expected = query.cpu(), q_expected.cpu()
                    assert torch.equal(observed, expected), (
                        generation,
                        code,
                        dtype,
                        pos,
                        (observed.float() - expected.float()).abs().max().item(),
                    )
                    assert torch.equal(before_k.cpu(), after_k.cpu())
                    assert torch.equal(before_v.cpu(), after_v.cpu())
                    rows.append(
                        {
                            "generation": generation,
                            "code": code,
                            "cache_dtype": str(dtype),
                            "position": pos,
                        }
                    )
    args.output.write_text(json.dumps({"cases": rows}, indent=2))
    print(json.dumps({"rope_cases": len(rows)}), flush=True)


if __name__ == "__main__":
    main()
