# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Test-container entry point: initialize NPU before ATB loads in spawn workers.

This file is also executed as __mp_main__ by Python's spawn bootstrap. Keep
device initialization at module scope so child workers follow the same order.
Only use this entry point for the single-device comparison containers.
"""

import torch
import torch_npu  # noqa: F401

torch.npu.set_device(0)

if __name__ == "__main__":
    from vllm.entrypoints.cli.main import main

    main()
