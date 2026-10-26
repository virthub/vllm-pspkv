# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Verify per-forward scratch reuse with distinct per-layer mixed KV caches."""

import argparse
import json
import math
from pathlib import Path
from types import SimpleNamespace

import torch
import torch_npu  # noqa: F401

from tests.v1.attention.psp_kv_reference import pack_kv_cache
from vllm.forward_context import override_forward_context
from vllm.v1.attention.backends.psp_kv_ascend_ops import PSPKVAscendPackedOps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.npu.set_device(0)
    torch.manual_seed(20261026)
    rows = []
    ops = PSPKVAscendPackedOps()
    for generation in range(3):
        for seq_lens in ([257], [1536], [257, 333]):
            block, heads, dim = 128, 4, 128
            per_seq = math.ceil(max(seq_lens) / block)
            pages = per_seq * len(seq_lens)
            codes_cpu = (torch.arange(pages) % 4).byte()
            table = torch.randperm(pages).int().view(len(seq_lens), per_seq).npu()
            cpu_lens = torch.tensor(seq_lens, dtype=torch.int32)
            lens = cpu_lens.npu()
            codes = codes_cpu.npu()
            scales = torch.ones(pages, heads, device="npu")
            owner = SimpleNamespace(_pspkv_pipeline_level=1)
            workspace_pointers = []
            for _ in range(4):
                shape = (pages, heads, block, dim)
                key = (torch.randn(shape) * 0.25).bfloat16()
                value = (torch.randn(shape) * 0.25).bfloat16()
                kb, vb, offsets_cpu = pack_kv_cache(
                    key, value, generation, codes_cpu, torch.bfloat16
                )
                query = torch.randn(len(seq_lens), 28, dim, device="npu").bfloat16()
                output = torch.empty_like(query)
                common = (
                    generation,
                    query,
                    kb.npu(),
                    vb.npu(),
                    table,
                    lens,
                    codes,
                    offsets_cpu.npu(),
                    scales,
                    dim**-0.5,
                    heads,
                    dim,
                    output,
                )
                ops._attention(*common, block_size=block, context_lens_cpu=cpu_lens)
                expected = output.cpu().clone()
                output.fill_(float("nan"))
                with override_forward_context(owner):
                    ops._attention(*common, block_size=block, context_lens_cpu=cpu_lens)
                torch.testing.assert_close(output.cpu(), expected, atol=0, rtol=0)
                assert len(owner._pspkv_paged_workspaces) == 1
                workspace = next(iter(owner._pspkv_paged_workspaces.values()))
                workspace_pointers.append(tuple(t.data_ptr() for t in workspace))
            assert len(set(workspace_pointers)) == 1
            rows.append({"generation": generation, "lengths": seq_lens, "layers": 4})
    args.output.write_text(json.dumps({"cases": rows}, indent=2))
    print(
        json.dumps({"workspace_cases": len(rows), "layer_comparisons": len(rows) * 4}),
        flush=True,
    )


if __name__ == "__main__":
    main()
