# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Test-container entry point: initialize NPU before ATB loads in spawn workers.

This file is also executed as __mp_main__ by Python's spawn bootstrap. Keep
device initialization at module scope so child workers follow the same order.
Only use this entry point for the single-device comparison containers.
"""

import os

import torch
import torch_npu  # noqa: F401

torch.npu.set_device(0)

if (
    os.environ.get("PSPKV_COMPARE_GRAPH_AB") == "1"
    or os.environ.get("PSPKV_ASCEND_FULL_GRAPH") == "1"
):
    from pipeline_runtime import install_graph_bypass

    install_graph_bypass()


from vllm.v1.attention.backends.psp_kv_ascend_native import load_native_fp8_ops
_audit_native = load_native_fp8_ops()
def _audit_once(name):
    function = getattr(_audit_native, name)
    emitted = False
    def call(*args, **kwargs):
        nonlocal emitted
        if not emitted:
            print("PSPKV_EXECUTION_AUDIT " + name, flush=True)
            emitted = True
        return function(*args, **kwargs)
    setattr(_audit_native, name, call)
for _audit_name in ("strategy_scatter_kv", "strategy_rope_scatter", "gather_paged_bf16_vec_out"):
    _audit_once(_audit_name)

if __name__ == "__main__":
    from vllm.entrypoints.cli.main import main

    main()
