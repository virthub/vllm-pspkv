# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Frozen diagnostic: retain failures and continue independent checks."""

import json
import runpy
import traceback
from pathlib import Path

import torch
import torch_npu  # noqa: F401

torch.npu.set_device(0)
results = {}
for name in (
    "check_fp8_reference_npu.py",
    "check_packed_reference_npu.py",
    "check_variable_npu.py",
    "check_variable_native_npu.py",
):
    print(f"BEGIN {name}", flush=True)
    try:
        runpy.run_path(f"/workspace/pspkv-src/ascend/{name}", run_name="__main__")
        results[name] = {"passed": True}
    except Exception as exc:
        traceback.print_exc()
        results[name] = {"passed": False, "error": repr(exc)}
    print(f"END {name}: {results[name]}", flush=True)

for device in ("cpu", "npu:0"):
    x = torch.tensor([0.125, 0.75], dtype=torch.float32, device=device)
    log = torch.log2(x / 3.0)
    print("SCALE_DIAGNOSTIC", device, {
        "log2": log.cpu().tolist(),
        "ceil": torch.ceil(log).cpu().tolist(),
        "scale": torch.pow(2.0, torch.ceil(log)).cpu().tolist(),
    }, flush=True)

Path("/audit/results/checks-summary.json").write_text(json.dumps(results, indent=2))
raise SystemExit(0 if all(row["passed"] for row in results.values()) else 1)
