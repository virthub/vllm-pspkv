# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Check cross-layer metadata reuse against identical pre-optimization calls."""

import argparse
import json
import math
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch_npu
from analyze_attention_performance import measure
from format_compare_runtime import STATES, comparison_attention

from tests.v1.attention.psp_kv_reference import pack_kv_cache
from vllm.forward_context import override_forward_context
from vllm.v1.attention.backends.psp_kv_ascend import (
    AscendAttentionMetadataBuilder,
    PSPKVAscendBackend,
    PSPKVAscendMetadataBuilder,
)


def run_fixture(length, generation, layer_count):
    block, kv_heads, dim, heads = 128, 4, 128, 28
    pages = math.ceil(length / block)
    tables_cpu = torch.randperm(pages).to(torch.int32).view(1, pages)
    tables = tables_cpu.to("npu")
    cpu_lengths = torch.tensor([length], dtype=torch.int32)
    device_lengths = cpu_lengths.to("npu")
    slot = tables_cpu[0, (length - 1) // block].item() * block + (length - 1) % block
    slots = torch.tensor([slot], dtype=torch.int64, device="npu")
    source = SimpleNamespace(
        block_tables=tables,
        seq_lens=cpu_lengths,
        seq_lens_cpu=cpu_lengths,
        psp_kv_device_seq_lens=device_lengths,
    )
    layers = []
    for _ in range(layer_count):
        kc = (torch.randn(pages, kv_heads, block, dim) * 0.3).bfloat16()
        vc = (torch.randn_like(kc.float()) * 0.3).bfloat16()
        k, v, _ = pack_kv_cache(
            kc, vc, generation, torch.ones(pages, dtype=torch.uint8), torch.bfloat16
        )
        k, v = [t.to("npu").view(pages, block, kv_heads, dim) for t in (k, v)]
        key = kc[slot // block, :, slot % block].unsqueeze(0).to("npu")
        value = vc[slot // block, :, slot % block].unsqueeze(0).to("npu")
        query = torch.randn(1, heads, dim).bfloat16().to("npu")
        output = torch.empty_like(query)
        layers.append((query, key, value, k, v, output))

    def step():
        owner = SimpleNamespace(
            attn_metadata={str(i): source for i in range(layer_count)}
        )
        with override_forward_context(owner):
            for query, key, value, k, v, output in layers:
                comparison_attention(
                    query,
                    key,
                    value,
                    k,
                    v,
                    slots[:1],
                    tables,
                    [1],
                    [length],
                    output,
                    1 / math.sqrt(dim),
                )
        return owner

    name = ("basic", "enhanced", "btkv")[generation]
    os.environ["PSPKV_COMPARE_FORMAT"] = name
    os.environ["PSPKV_COMPARE_PRECISION"] = "e4m3"
    os.environ["PSPKV_COMPARE_OPTIMIZED"] = "1"
    os.environ["PSPKV_COMPARE_PAGED_MIN_LENGTH"] = "64"
    STATES.clear()
    os.environ["PSPKV_ASCEND_SHARE_METADATA"] = "0"
    step()
    torch.npu.synchronize()
    before = [row[-1].cpu().clone() for row in layers]
    os.environ["PSPKV_ASCEND_SHARE_METADATA"] = "1"
    owner = step()
    torch.npu.synchronize()
    for row, expected in zip(layers, before):
        torch.testing.assert_close(row[-1].cpu(), expected, atol=0, rtol=0)
    cache = owner._pspkv_step_metadata
    assert cache.builds == 1 and cache.hits == layer_count - 1
    assert cache.borrowed_lengths == 1 and cache.created_lengths == 0
    prepared = next(iter(cache.prepared.values()))
    assert prepared.lengths.data_ptr() == device_lengths.data_ptr()
    return step, {
        "format": name,
        "length": length,
        "layers": layer_count,
        "outputs_bit_equal": True,
        "builds": cache.builds,
        "hits": cache.hits,
        "borrowed_lengths": cache.borrowed_lengths,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trace-dir", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(20261025)
    torch.npu.set_device(0)
    os.environ.pop("PSPKV_METADATA_CONTROL_FILE", None)
    assert PSPKVAscendBackend.get_builder_cls() is PSPKVAscendMetadataBuilder
    # Ascend's CPU seq_lens is intentionally distinct from the common NPU view.
    cpu = torch.tensor([135], dtype=torch.int32)
    device = cpu.to("npu")
    built = SimpleNamespace(seq_lens=cpu)
    with patch.object(AscendAttentionMetadataBuilder, "build", return_value=built):
        builder = object.__new__(PSPKVAscendMetadataBuilder)
        result = builder.build(0, SimpleNamespace(seq_lens=device))
        assert result.psp_kv_device_seq_lens is device
        assert result.seq_lens is cpu
    report = {"checks": [], "timings": []}
    for generation in range(3):
        for length in (3, 32, 135, 160, 257):
            _, row = run_fixture(length, generation, 4)
            report["checks"].append(row)
    for generation in range(3):
        for length in (32, 160):
            step, row = run_fixture(length, generation, 28)
            for enabled in (False, True):
                os.environ["PSPKV_ASCEND_SHARE_METADATA"] = str(int(enabled))
                row["after" if enabled else "before"] = measure(step, 5, 5)
                if args.trace_dir and generation == 0 and length == 160:
                    config = torch_npu.profiler._ExperimentalConfig(
                        profiler_level=torch_npu.profiler.ProfilerLevel.Level1
                    )
                    with torch_npu.profiler.profile(
                        activities=[
                            torch_npu.profiler.ProfilerActivity.CPU,
                            torch_npu.profiler.ProfilerActivity.NPU,
                        ],
                        experimental_config=config,
                        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(
                            str(args.trace_dir / ("after" if enabled else "before"))
                        ),
                    ):
                        for _ in range(6):
                            step()
                        torch.npu.synchronize()
            report["timings"].append(row)
            args.output.write_text(json.dumps(report, indent=2))
            print(json.dumps(row), flush=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(
        "PASS: builder ownership, 15 multi-layer cases and six 28-layer comparisons",
        flush=True,
    )


if __name__ == "__main__":
    main()
