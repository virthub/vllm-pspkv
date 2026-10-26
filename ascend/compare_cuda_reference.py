# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Run original CUDA device-kernel bodies against exported NPU fixtures.

NVRTC compiles the unchanged device definitions and quantization header from
the specified Git revision. Host launchers and PyTorch registration are not
built. CUDA Driver API calls reproduce their grid and argument conventions.
The BTKV reference body does not use TMA, so this does not validate hardware
TMA on Hopper/Blackwell. A CPU-only PyTorch is enough to read fixture tensors.
"""

import argparse
import ctypes as c
import hashlib
import json
import subprocess
from pathlib import Path

import torch

NAMES = ("basic", "enhanced", "btkv")


def bind(library, name, arguments):
    function = getattr(library, name)
    function.argtypes = arguments
    function.restype = c.c_int
    return function


def check(status):
    if status:
        raise RuntimeError(f"CUDA/NVRTC returned status {status}")


class Driver:
    def __init__(self):
        self.library = c.CDLL("/usr/lib/wsl/lib/libcuda.so.1")
        self.init = bind(self.library, "cuInit", [c.c_uint])
        self.get_device = bind(
            self.library, "cuDeviceGet", [c.POINTER(c.c_int), c.c_int]
        )
        self.get_name = bind(
            self.library, "cuDeviceGetName", [c.c_void_p, c.c_int, c.c_int]
        )
        self.capability = bind(
            self.library,
            "cuDeviceComputeCapability",
            [c.POINTER(c.c_int), c.POINTER(c.c_int), c.c_int],
        )
        self.create_context = bind(
            self.library, "cuCtxCreate_v2", [c.POINTER(c.c_void_p), c.c_uint, c.c_int]
        )
        self.allocate = bind(
            self.library, "cuMemAlloc_v2", [c.POINTER(c.c_uint64), c.c_size_t]
        )
        self.free = bind(self.library, "cuMemFree_v2", [c.c_uint64])
        self.to_device = bind(
            self.library, "cuMemcpyHtoD_v2", [c.c_uint64, c.c_void_p, c.c_size_t]
        )
        self.to_host = bind(
            self.library, "cuMemcpyDtoH_v2", [c.c_void_p, c.c_uint64, c.c_size_t]
        )
        self.load = bind(
            self.library, "cuModuleLoadData", [c.POINTER(c.c_void_p), c.c_void_p]
        )
        self.get_function = bind(
            self.library,
            "cuModuleGetFunction",
            [c.POINTER(c.c_void_p), c.c_void_p, c.c_char_p],
        )
        self.launch = bind(
            self.library,
            "cuLaunchKernel",
            [
                c.c_void_p,
                *([c.c_uint] * 7),
                c.c_void_p,
                c.POINTER(c.c_void_p),
                c.c_void_p,
            ],
        )
        self.synchronize = bind(self.library, "cuCtxSynchronize", [])
        check(self.init(0))
        self.device = c.c_int()
        check(self.get_device(c.byref(self.device), 0))
        self.context = c.c_void_p()
        check(self.create_context(c.byref(self.context), 0, self.device))
        buffer = c.create_string_buffer(256)
        check(self.get_name(buffer, len(buffer), self.device))
        self.name = buffer.value.decode()
        major, minor = c.c_int(), c.c_int()
        check(self.capability(c.byref(major), c.byref(minor), self.device))
        self.arch = f"compute_{major.value}{minor.value}"

    def upload(self, tensor):
        raw = tensor.contiguous().view(torch.uint8).numpy().tobytes()
        pointer = c.c_uint64()
        check(self.allocate(c.byref(pointer), max(len(raw), 1)))
        if raw:
            check(self.to_device(pointer, c.c_char_p(raw), len(raw)))
        return pointer


def compile_kernel(nvidia, arch, name, revision, output_dir, probe=False):
    root = "csrc/attention/psp_kv/"
    path = root + f"{name}/paged_attention_{name}.cu"
    original = subprocess.check_output(["git", "show", revision + ":" + path]).decode()
    header = subprocess.check_output(
        ["git", "show", revision + ":" + root + "common/quantization_utils.cuh"]
    )
    start = original.index("template <typename scalar_t, int HEAD_DIM>")
    end = original.index("\ntemplate <typename scalar_t>\nvoid launch_", start)
    body = original[start:end]
    if probe:
        body = """
extern "C" __global__ void pspkv_quantization_probe(
    const float* input, uint8_t* encoded, float* decoded) {
    int i = threadIdx.x;
    if (i < 16) {
        encoded[i] = quantize_fp8_e4m3(input[i]);
        decoded[i] = dequantize_fp8_e4m3(encoded[i]);
    }
}
"""
    source = (
        '#include "common/quantization_utils.cuh"\n'
        "#define BLOCK_SIZE 16\n#define MAX_KV_HEADS 8\n" + body
    ).encode()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{name}_device.cu").write_bytes(source)
    (output_dir / "quantization_utils.cuh").write_bytes(header)
    nvrtc = c.CDLL(str(nvidia / "cuda_nvrtc/lib/libnvrtc.so.12"))
    create = bind(
        nvrtc,
        "nvrtcCreateProgram",
        [
            c.POINTER(c.c_void_p),
            c.c_char_p,
            c.c_char_p,
            c.c_int,
            c.POINTER(c.c_char_p),
            c.POINTER(c.c_char_p),
        ],
    )
    add_name = bind(nvrtc, "nvrtcAddNameExpression", [c.c_void_p, c.c_char_p])
    compile_program = bind(
        nvrtc, "nvrtcCompileProgram", [c.c_void_p, c.c_int, c.POINTER(c.c_char_p)]
    )
    log_size = bind(
        nvrtc, "nvrtcGetProgramLogSize", [c.c_void_p, c.POINTER(c.c_size_t)]
    )
    get_log = bind(nvrtc, "nvrtcGetProgramLog", [c.c_void_p, c.c_void_p])
    ptx_size = bind(nvrtc, "nvrtcGetPTXSize", [c.c_void_p, c.POINTER(c.c_size_t)])
    get_ptx = bind(nvrtc, "nvrtcGetPTX", [c.c_void_p, c.c_void_p])
    get_name = bind(
        nvrtc, "nvrtcGetLoweredName", [c.c_void_p, c.c_char_p, c.POINTER(c.c_char_p)]
    )
    destroy = bind(nvrtc, "nvrtcDestroyProgram", [c.POINTER(c.c_void_p)])
    program = c.c_void_p()
    check(
        create(
            c.byref(program),
            source,
            path.encode(),
            3,
            (c.c_char_p * 3)(
                header,
                b"// NVRTC supplies the CUDA math builtins directly.\n"
                b"#include <math_constants.h>\n"
                b"#define NAN CUDART_NAN_F\n#define INFINITY CUDART_INF_F\n",
                b"typedef unsigned char uint8_t;\n"
                b"typedef unsigned short uint16_t;\n"
                b"typedef unsigned int uint32_t;\n"
                b"typedef unsigned long long uint64_t;\n"
                b"typedef long long int64_t;\n",
            ),
            (c.c_char_p * 3)(b"common/quantization_utils.cuh", b"math.h", b"stdint.h"),
        )
    )
    expression = (
        b"pspkv_quantization_probe"
        if probe
        else f"paged_attention_{name}_kernel<__nv_bfloat16,128>".encode()
    )
    check(add_name(program, expression))
    options = [
        "--std=c++17",
        "--gpu-architecture=" + arch,
        "--include-path=" + str(nvidia / "cuda_runtime/include"),
    ]
    cccl = nvidia / "cuda_cccl/include"
    if cccl.exists():
        options.append("--include-path=" + str(cccl))
    status = compile_program(
        program,
        len(options),
        (c.c_char_p * len(options))(*(x.encode() for x in options)),
    )
    size = c.c_size_t()
    check(log_size(program, c.byref(size)))
    log = c.create_string_buffer(size.value)
    check(get_log(program, log))
    (output_dir / f"{name}-compile.log").write_text(log.value.decode())
    if status:
        raise RuntimeError(
            f"{name} compilation failed ({status}):\n{log.value.decode()}"
        )
    check(ptx_size(program, c.byref(size)))
    ptx = c.create_string_buffer(size.value)
    check(get_ptx(program, ptx))
    lowered = c.c_char_p()
    check(get_name(program, expression, c.byref(lowered)))
    symbol = lowered.value
    check(destroy(c.byref(program)))
    return (
        ptx.raw,
        symbol,
        {
            "source": path,
            "source_sha256": hashlib.sha256(original.encode()).hexdigest(),
            "header_sha256": hashlib.sha256(header).hexdigest(),
            "compiler_options": options,
        },
    )


def run_case(driver, function, name, fixture):
    query, keys, values, tables, lengths, codes, offsets, scales, _ = fixture
    output = torch.empty_like(query)
    zeros = torch.zeros_like(scales)
    tensors = [
        output,
        query,
        keys,
        values,
        tables,
        lengths,
        codes,
        offsets,
        scales,
        zeros,
    ]
    pointers = [driver.upload(tensor) for tensor in tensors]
    out, q, k, v, bt, lens, prec, offs, sc, z = pointers
    parameters = [out, q, k, v]
    parameters.extend((sc, z, sc, z) if name == "basic" else (sc, sc))
    parameters.extend(
        (
            bt,
            lens,
            prec,
            offs,
            c.c_int(tables.shape[1]),
            c.c_int(query.shape[1]),
            c.c_int(scales.shape[1]),
            c.c_int(query.shape[2]),
            c.c_float(1 / query.shape[2] ** 0.5),
        )
    )
    arguments = (c.c_void_p * len(parameters))(*(c.addressof(p) for p in parameters))
    check(
        driver.launch(
            function,
            query.shape[0],
            query.shape[1],
            query.shape[0],
            1,
            1,
            1,
            0,
            None,
            arguments,
            None,
        )
    )
    check(driver.synchronize())
    storage = c.create_string_buffer(output.numel() * output.element_size())
    check(driver.to_host(storage, out, len(storage)))
    result = (
        torch.frombuffer(bytearray(storage.raw), dtype=query.dtype)
        .reshape(query.shape)
        .clone()
    )
    for pointer in pointers:
        check(driver.free(pointer))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda-libs", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-revision", default="f613ce1")
    parser.add_argument("--quantization-probe", action="store_true")
    args = parser.parse_args()
    artifacts = args.output.parent / (
        "quantization-compiled" if args.quantization_probe else "cuda-compiled"
    )
    driver = Driver()
    data = {
        "gpu": driver.name,
        "architecture": driver.arch,
        "source_revision": args.source_revision,
        "scope": (
            "original CUDA device bodies; original host registration "
            "and TMA path not tested"
        ),
        "kernels": {},
        "cases": [],
    }
    functions, modules = {}, []
    for name in ("basic",) if args.quantization_probe else NAMES:
        ptx, symbol, metadata = compile_kernel(
            args.cuda_libs,
            driver.arch,
            name,
            args.source_revision,
            artifacts,
            probe=args.quantization_probe,
        )
        module, function = c.c_void_p(), c.c_void_p()
        check(driver.load(c.byref(module), c.c_char_p(ptx)))
        check(driver.get_function(c.byref(function), module, symbol))
        modules.append(module)
        functions[name] = function
        data["kernels"][name] = metadata
        print("compiled", name, flush=True)
    if args.quantization_probe:
        data["scope"] = "original CUDA quantization header; E4M3 encode/decode probe"
        inputs = torch.tensor(
            [
                0,
                1,
                240,
                248,
                256,
                272,
                288,
                320,
                352,
                384,
                416,
                418,
                432,
                448,
                -418,
                -448,
            ],
            dtype=torch.float32,
        )
        encoded = torch.empty(16, dtype=torch.uint8)
        decoded = torch.empty(16, dtype=torch.float32)
        pointers = [driver.upload(t) for t in (inputs, encoded, decoded)]
        parameters = (c.c_void_p * 3)(*(c.addressof(p) for p in pointers))
        check(
            driver.launch(
                functions["basic"], 1, 1, 1, 32, 1, 1, 0, None, parameters, None
            )
        )
        check(driver.synchronize())
        encoded_buffer, decoded_buffer = (
            c.create_string_buffer(16),
            c.create_string_buffer(64),
        )
        check(driver.to_host(encoded_buffer, pointers[1], 16))
        check(driver.to_host(decoded_buffer, pointers[2], 64))
        decoded = torch.frombuffer(bytearray(decoded_buffer.raw), dtype=torch.float32)
        native = inputs.to(torch.float8_e4m3fn).float()
        data["quantization_probe"] = [
            {
                "input": x,
                "original_cuda_byte": byte,
                "original_cuda_decoded": y,
                "torch_e4m3fn_decoded": z,
            }
            for x, byte, y, z in zip(
                inputs.tolist(), encoded_buffer.raw, decoded.tolist(), native.tolist()
            )
        ]
        for pointer in pointers:
            check(driver.free(pointer))
        print(json.dumps(data["quantization_probe"], indent=2), flush=True)
    if args.fixtures:
        # Older exporters stored torch.__version__ as this known string subclass.
        with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
            package = torch.load(args.fixtures, map_location="cpu", weights_only=True)
        data["fixtures_sha256"] = hashlib.sha256(args.fixtures.read_bytes()).hexdigest()
        for case in package["cases"]:
            for generation, name in enumerate(NAMES):
                actual = run_case(
                    driver, functions[name], name, case["fixtures"][generation]
                )
                npu = case["npu_outputs"][generation]
                reference = case["references"][generation]
                torch.testing.assert_close(
                    actual.float(), npu.float(), atol=1e-2, rtol=1e-2
                )
                torch.testing.assert_close(
                    actual.float(), reference.float(), atol=1e-2, rtol=1e-2
                )
                assert bool(torch.isfinite(actual).all())
                row = {
                    "workload": case["workload"],
                    "precision": case["precision"],
                    "format": name,
                    "cuda_npu_bit_equal": torch.equal(actual, npu),
                    "cuda_npu_max_abs_error": (actual.float() - npu.float())
                    .abs()
                    .max()
                    .item(),
                    "cuda_cpu_max_abs_error": (actual.float() - reference.float())
                    .abs()
                    .max()
                    .item(),
                }
                data["cases"].append(row)
                print(json.dumps(row), flush=True)
    args.output.write_text(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
