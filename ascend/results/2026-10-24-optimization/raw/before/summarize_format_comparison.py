# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Summarize saved operator and streaming model comparison measurements."""

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

import regex as re

FORMATS = ("basic", "enhanced", "btkv")


def percentile(values, quantile):
    ordered = sorted(values)
    index = (len(ordered) - 1) * quantile
    low, high = math.floor(index), math.ceil(index)
    return ordered[low] + (ordered[high] - ordered[low]) * (index - low)


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    operator_path = args.input_dir / "operators.json"
    operators = (
        json.loads(operator_path.read_text())
        if operator_path.exists()
        else {"cases": []}
    )
    rows = []
    for case in operators["cases"]:
        for name in FORMATS:
            value = case["formats"][name]
            rows.append(
                {
                    "workload": case["workload"],
                    "precision": case["precision"],
                    "format": name,
                    "wall_ms_median": value["wall_ms_median"],
                    "wall_ms_min": min(value["wall_ms_samples"]),
                    "wall_ms_max": max(value["wall_ms_samples"]),
                    "device_ms_median": value["device_ms_median"],
                    "kv_data_bytes": value["cache_bytes"],
                    "reference_max_abs_error": value["reference_max_abs_error"],
                    "dense_bf16_relative_l2_error": value[
                        "dense_bf16_relative_l2_error"
                    ],
                }
            )
    write_csv(args.output_dir / "operators.csv", rows)
    summary = {
        "operator_combinations": len(rows),
        "all_operator_outputs_bit_equal": (
            all(
                case["basic_enhanced_bit_equal"] and case["enhanced_btkv_bit_equal"]
                for case in operators["cases"]
            )
            if operators["cases"]
            else None
        ),
        "max_operator_reference_error": max(
            (row["reference_max_abs_error"] for row in rows), default=None
        ),
    }
    serving, responses, resources = [], {}, {}
    for name in ("stock-bf16", *FORMATS):
        path = args.input_dir / f"{name}-serving.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        responses[name] = data
        server_log = args.input_dir / f"{name}-server.log"
        if server_log.exists():
            log = server_log.read_text(errors="replace")
            capacity = re.search(r"GPU KV cache size: ([\d,]+) tokens", log)
            records = [
                json.loads(match)
                for match in re.findall(r"PSPKV_FORMAT_COMPARISON (\{[^\n]+\})", log)
            ]
            resources[name] = {
                "kv_token_capacity": (
                    int(capacity[1].replace(",", "")) if capacity else None
                ),
                "recorded_attention_layers": len(records),
                "adapter_metadata_bytes": sum(r["metadata_bytes"] for r in records),
                "decode_entries": sorted({r["decode_entry"] for r in records}),
            }
        for profile in ("short", "cross_page"):
            requests = [row for row in data["requests"] if row["profile"] == profile]
            if not requests:
                continue
            seconds = [row["seconds"] for row in requests]
            first = [row["ttft_ms"] for row in requests]
            tpot = [row["tpot_ms"] for row in requests]
            serving.append(
                {
                    "format": name,
                    "profile": profile,
                    "requests": len(requests),
                    "prompt_tokens": requests[0]["usage"]["prompt_tokens"],
                    "output_tokens": requests[0]["usage"]["completion_tokens"],
                    "latency_s_p50": statistics.median(seconds),
                    "latency_s_p95": percentile(seconds, 0.95),
                    "ttft_ms_p50": statistics.median(first),
                    "ttft_ms_p95": percentile(first, 0.95),
                    "tpot_ms_p50": statistics.median(tpot),
                    "tpot_ms_p95": percentile(tpot, 0.95),
                    "serial_output_tokens_per_s": sum(
                        row["usage"]["completion_tokens"] for row in requests
                    )
                    / sum(seconds),
                    "identical_repeated_outputs": len({row["text"] for row in requests})
                    == 1,
                }
            )
    write_csv(args.output_dir / "serving.csv", serving)
    summary["serving"] = serving
    summary["resources"] = resources
    summary["measured_generation_requests"] = sum(row["requests"] for row in serving)
    compared = {name: responses[name] for name in FORMATS if name in responses}
    if len(compared) == 3:
        summary["cross_format_output_equal"] = {
            profile: len(
                {
                    row["text"]
                    for data in compared.values()
                    for row in data["requests"]
                    if row["profile"] == profile
                }
            )
            == 1
            for profile in ("short", "cross_page")
        }
        summary["quality_probe_outputs_equal"] = all(
            len(
                {
                    data["quality_probes"][index]["response"]["text"]
                    for data in compared.values()
                }
            )
            == 1
            for index in range(2)
        )
    if "stock-bf16" in responses:
        baseline = responses["stock-bf16"]
        summary["stock_bf16_output_equal"] = {
            name: {
                profile: {
                    row["text"]
                    for row in baseline["requests"]
                    if row["profile"] == profile
                }
                == {
                    row["text"] for row in data["requests"] if row["profile"] == profile
                }
                for profile in ("short", "cross_page")
            }
            for name, data in compared.items()
        }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2)
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
