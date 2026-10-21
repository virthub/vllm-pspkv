# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Optional Ascend C operators for the PSP-KV reference backend."""

import os
from functools import lru_cache

import torch


@lru_cache(maxsize=1)
def load_native_fp8_ops() -> object:
    """Load the explicitly built extension; never silently change the backend."""
    library = os.environ.get("PSPKV_ASCEND_NATIVE_LIB")
    if not library:
        raise RuntimeError("PSP-KV native operators require PSPKV_ASCEND_NATIVE_LIB")
    if not os.path.isfile(library):
        raise FileNotFoundError(f"PSP-KV native library not found: {library}")
    torch.ops.load_library(library)
    return torch.ops.psp_kv_ascend_native
