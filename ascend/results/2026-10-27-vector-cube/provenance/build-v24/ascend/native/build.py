# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Build the PSP-KV Ascend C extension with the active torch_npu environment.

Run with a uv-managed ``.venv/bin/python`` inside the vLLM-Ascend image.
The output shared library is written to the caller-supplied path; it is not
installed into vLLM or the shared vLLM-Ascend plugin.
"""

import argparse
import os
import subprocess
import sysconfig
from pathlib import Path

import torch
import torch_npu
from torch.utils.cpp_extension import include_paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--arch", default="dav-2201", help="910B3 target architecture")
    parser.add_argument("--source", default="pspkv_fp8.asc")
    args = parser.parse_args()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).with_name(args.source)
    torch_dir = Path(torch.__file__).parent
    torch_npu_dir = Path(torch_npu.__file__).parent
    includes = [
        *include_paths(),
        sysconfig.get_path("include"),
        str(torch_npu_dir / "include"),
    ]
    sdk = Path(
        os.environ.get("ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest")
    )
    asc = sdk / "aarch64-linux/asc"
    if asc.exists():
        includes.extend([str(asc), str(asc / "include/adv_api")])
    libraries = [torch_dir / "lib", torch_npu_dir / "lib"]
    command = [
        "bisheng",
        "-x",
        "asc",
        f"--npu-arch={args.arch}",
        "-shared",
        "-fPIC",
        "-O2",
        "-std=c++17",
        f"-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}",
        str(source),
        "-o",
        str(output),
        *[f"-I{path}" for path in includes],
        *[f"-L{path}" for path in libraries],
        "-ltorch_npu",
        "-ltorch",
        "-ltorch_cpu",
        "-lc10",
    ]
    if args.source == "pspkv_cube.asc":
        command.extend(
            [f"-L{sdk / 'aarch64-linux/lib64'}", "-ltiling_api", "-lplatform"]
        )
    print("Building", output, "for", args.arch, flush=True)
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
