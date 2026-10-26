# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Summarize small eager precision probes without claiming benchmark significance."""

import argparse
import ast
import json
import statistics
from pathlib import Path

import regex as re


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path, help="Frozen audit directory")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for name in (
        "stock-bf16",
        "fixed-e5",
        "adaptive-fp8",
        "calibration",
        "dynamic-conservative",
        "dynamic-fp4",
        "static-fp8",
        "uniform-fp4",
        "static-fp4-grouped",
        "asymmetric-k8v4",
    ):
        path = args.directory / "results" / f"{name}.json"
        log_path = path.with_name(f"{name}-server.log")
        if not path.exists():
            rows.append({"case": name, "complete": False})
            continue
        data = json.loads(path.read_text())
        log = log_path.read_text() if log_path.exists() else ""
        capacity = re.search(r"KV cache size: ([\d,]+) tokens", log)
        observed = set()
        for entry in re.findall(r"formats (\{[^\n]+?\})", log):
            observed.update(ast.literal_eval(entry))
        row = {
            "case": name,
            "complete": data["complete"],
            "output_tokens": data["output_tokens"],
            "allocator_kv_tokens": int(capacity[1].replace(",", ""))
            if capacity
            else None,
            "observed_k_codes_in_variable_arena_logs": sorted(observed),
            "quality_probes": data["quality_probes"],
        }
        for profile in ("short", "cross_page"):
            requests = [x for x in data["requests"] if x["profile"] == profile]
            row[profile] = {
                "samples": len(requests),
                "p50_seconds": statistics.median(x["seconds"] for x in requests),
                "texts": list(dict.fromkeys(x["text"] for x in requests)),
            }
        rows.append(row)
    result = {
        "scope": "Qwen2.5-7B base, eager, concurrency=1, 16 output tokens; "
        "two timed requests per input; functionality/output review only",
        "format_count_note": "Observed K codes are a union over log snapshots, "
        "not a distribution or proof of arena saturation; V may differ.",
        "cases": rows,
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    for row in rows:
        print(
            row["case"],
            row.get("allocator_kv_tokens"),
            *(
                round(row[p]["p50_seconds"], 3) if p in row else None
                for p in ("short", "cross_page")
            ),
            "complete=",
            row["complete"],
        )


if __name__ == "__main__":
    main()
