# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Export identical packed inputs and measured NPU outputs for CUDA checks."""

import argparse
import json
from pathlib import Path

import torch
import torch_npu  # noqa: F401
from compare_formats import FORMATS, PRECISIONS, WORKLOADS, invoke, prepare

from vllm.v1.attention.backends.psp_kv_ascend_native import load_native_fp8_ops
from vllm.v1.attention.backends.psp_kv_ascend_ops import PSPKVAscendPackedOps


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.npu.set_device(0)
    load_native_fp8_ops()
    ops = PSPKVAscendPackedOps()
    package = {"torch": str(torch.__version__), "cases": []}
    workloads = (*WORKLOADS, ("qwen25_gqa_128", [128], 28, 4))
    for workload, lengths, heads, kv_heads in workloads:
        for precision, pattern in PRECISIONS.items():
            fixtures, references, _, _ = prepare(
                lengths, heads, kv_heads, pattern, seed=20261023
            )
            outputs = []
            for generation, name in enumerate(FORMATS):
                output = torch.empty_like(fixtures[generation][0])
                invoke(ops, generation, fixtures[generation], output)
                torch.npu.synchronize()
                output = output.cpu()
                torch.testing.assert_close(
                    output.float(),
                    references[generation].float(),
                    atol=1e-2,
                    rtol=1e-2,
                )
                assert bool(torch.isfinite(output).all())
                outputs.append(output)
            package["cases"].append(
                {
                    "workload": workload,
                    "precision": precision,
                    "fixtures": [tuple(t.cpu() for t in f) for f in fixtures],
                    "npu_outputs": outputs,
                    "references": references,
                }
            )
            print(
                json.dumps({"workload": workload, "precision": precision}), flush=True
            )
    torch.save(package, args.output)
    print(f"Saved {len(package['cases']) * 3} outputs to {args.output}", flush=True)


if __name__ == "__main__":
    main()
