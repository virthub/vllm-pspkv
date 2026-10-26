# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Profile the four Qwen2.5-7B projection shapes with device pipe counters.

Standalone diagnostics only: repeated weights may hit cache, so these timings
and counters do not represent a full model step or justify changing weights.
"""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import torch_npu


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.npu.set_device(0)
    torch.manual_seed(20261026)
    torch.set_num_threads(2)
    tensors = []
    rows = []
    for name, inner, outer in (
        ("qkv", 3584, 4608),
        ("o_proj", 3584, 3584),
        ("gate_up", 3584, 37888),
        ("down", 18944, 3584),
    ):
        weight = torch.randn(outer, inner, device="npu", dtype=torch.bfloat16)
        for tokens in (1, 16, 135):
            x = torch.randn(tokens, inner, device="npu", dtype=torch.bfloat16)
            out = torch.empty(tokens, outer, device="npu", dtype=torch.bfloat16)
            for _ in range(3):
                torch.mm(x, weight.t(), out=out)
            torch.npu.synchronize()
            start = time.perf_counter()
            for _ in range(20):
                torch.mm(x, weight.t(), out=out)
            torch.npu.synchronize()
            ms = (time.perf_counter() - start) * 1000 / 20
            row = {
                "projection": name,
                "tokens": tokens,
                "inner": inner,
                "outer": outer,
                "host_loop_ms": ms,
                "useful_flops": 2 * tokens * inner * outer,
                "weight_bytes": weight.numel() * weight.element_size(),
            }
            rows.append(row)
            tensors.append((name, tokens, x, weight))
    for metric_name in ("PipeUtilization", "MemoryAccess"):
        config = torch_npu.profiler._ExperimentalConfig(
            profiler_level=torch_npu.profiler.ProfilerLevel.Level2,
            aic_metrics=getattr(torch_npu.profiler.AiCMetrics, metric_name),
        )
        with torch_npu.profiler.profile(
            activities=[
                torch_npu.profiler.ProfilerActivity.CPU,
                torch_npu.profiler.ProfilerActivity.NPU,
            ],
            experimental_config=config,
            record_shapes=True,
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                str(args.output_dir / metric_name)
            ),
        ) as profiler:
            for name, tokens, x, weight in tensors:
                with torch.autograd.profiler.record_function(f"{name}_m{tokens}"):
                    F.linear(x, weight)
                profiler.step()
            torch.npu.synchronize()
    (args.output_dir / "timings.json").write_text(json.dumps(rows, indent=2))
    print(json.dumps(rows), flush=True)


if __name__ == "__main__":
    main()
