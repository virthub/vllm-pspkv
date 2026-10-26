# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Capture two extra requests after latency measurement; never mix samples."""

import argparse
import json
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--control-file", type=Path)
    parser.add_argument("--trace-root", type=Path)
    parser.add_argument("--after-level", type=int, default=1)
    args = parser.parse_args()
    measured = json.loads(args.input.read_text())
    base = f"http://127.0.0.1:{args.port}"

    def post(path, payload):
        request = urllib.request.Request(
            base + path,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=180) as response:
            return response.read()

    results = []
    for enabled in (False, True) if args.control_file else (None,):
        if args.control_file:
            args.control_file.write_text(str(args.after_level) if enabled else "0")
        old_dirs = set(args.trace_root.iterdir()) if args.trace_root else set()
        requests = []
        post("/start_profile", {})
        try:
            for profile in ("cross_page", "short"):
                response = json.loads(
                    post(
                        "/v1/completions",
                        {
                            "model": measured["model"],
                            "prompt": measured["profiles"][profile],
                            "max_tokens": 4,
                            "temperature": 0,
                            "seed": 123,
                            "ignore_eos": True,
                        },
                    )
                )
                assert response["usage"]["completion_tokens"] == 4
                requests.append({"profile": profile, "response": response})
        finally:
            post("/stop_profile", {})
        if args.control_file:
            new_dirs = (
                set(args.trace_root.iterdir()) - old_dirs if args.trace_root else set()
            )
            results.append(
                {
                    "metadata_reuse": enabled,
                    "requests": requests,
                    "trace_directories": sorted(path.name for path in new_dirs),
                }
            )
        else:
            results = requests
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
