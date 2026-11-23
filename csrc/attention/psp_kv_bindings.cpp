// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Python module entry point for the PSP-KV extension (_C).
//
// The PSP-KV ops are registered by the TORCH_LIBRARY(psp_kv, m) /
// TORCH_LIBRARY_IMPL(psp_kv, CUDA, m) static initializers in
// csrc/attention/psp_kv_wrapper.cu, so importing this module is enough to make
// torch.ops.psp_kv.* resolve.
//
// The wrapper (and this file) is compiled with Py_LIMITED_API (the extension
// uses USE_SABI), under which pybind11 headers do not compile. The module
// therefore uses vLLM's plain limited-API module registration instead of
// <torch/extension.h> / PYBIND11_MODULE; it exposes no Python-level API of its
// own.

#include "core/registration.h"

REGISTER_EXTENSION(TORCH_EXTENSION_NAME)
