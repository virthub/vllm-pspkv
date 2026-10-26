# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Compare extra model traces, excluding them from latency measurements."""

import argparse
import collections
import csv
import json
from pathlib import Path


def analyze(directory, layers):
    source = directory / "ASCEND_PROFILER_OUTPUT"
    rows = list(csv.DictReader((source / "kernel_details.csv").open()))
    for row in rows:
        row["start"] = float(row["Start Time(us)"])
        row["end"] = row["start"] + float(row["Duration(us)"])
    rows.sort(key=lambda r: r["start"])
    grouped = collections.defaultdict(lambda: {"count": 0, "device_us": 0.0})
    for row in rows:
        group = grouped[row["Name"]]
        group["count"] += 1
        group["device_us"] += float(row["Duration(us)"])
    fused = [r for r in rows if "PspkvTileFusedAttention" in r["Name"]]

    def busy(start, end):
        intervals = sorted(
            (max(start, r["start"]), min(end, r["end"]))
            for r in rows
            if r["start"] < end and r["end"] > start
        )
        total, last = 0.0, start
        for begin, finish in intervals:
            if finish > last:
                total += finish - max(last, begin)
                last = finish
        return total

    window_us, kernel_us = 0.0, 0.0
    assert len(fused) % layers == 0
    for i in range(len(fused) - 1):
        if i % layers == layers - 1:
            continue
        begin, end = fused[i]["start"], fused[i + 1]["start"]
        window_us += end - begin
        kernel_us += busy(begin, end)
    apis = {
        r["API Name"]: r for r in csv.DictReader((source / "api_statistic.csv").open())
    }
    host = collections.defaultdict(lambda: {"count": 0, "self_us": 0.0})
    for row in csv.DictReader((source / "operator_details.csv").open()):
        item = host[row["Name"]]
        item["count"] += 1
        item["self_us"] += float(row["Host Self Duration(us)"])
    return {
        "trace_directory": directory.name,
        "kernel_count": len(rows),
        "fused_calls": len(fused),
        "sync_copies": int(apis.get("aclrtMemcpy", {}).get("Count", 0)),
        "stream_syncs": int(apis.get("aclrtSynchronizeStream", {}).get("Count", 0)),
        "async_copies": int(apis.get("aclrtMemcpyAsync", {}).get("Count", 0)),
        "device_kernel_sum_us": sum(float(r["Duration(us)"]) for r in rows),
        "decode_layer_windows_us": window_us,
        "decode_layer_kernel_union_us": kernel_us,
        "kernels": dict(
            sorted(grouped.items(), key=lambda x: x[1]["device_us"], reverse=True)
        ),
        "host_ops": dict(
            sorted(host.items(), key=lambda x: x[1]["self_us"], reverse=True)
        ),
        "apis": apis,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", type=int, default=28)
    args = parser.parse_args()
    sessions = json.loads((args.input_dir / "basic-profile-requests.json").read_text())
    result = {
        "scope": (
            "Two extra 4-token requests per mode; profiler overhead included. "
            "Kernel interval coverage is not unprofiled whole-device utilization."
        )
    }
    for session in sessions:
        mode = "after" if session["metadata_reuse"] else "before"
        assert len(session["trace_directories"]) == 1
        directory = (
            args.input_dir / "model-profiler-basic" / session["trace_directories"][0]
        )
        result[mode] = analyze(directory, args.layers)
    args.output.write_text(json.dumps(result, indent=2))
    print(
        json.dumps(
            {
                mode: {
                    key: value
                    for key, value in stats.items()
                    if key not in ("kernels", "host_ops", "apis")
                }
                for mode, stats in result.items()
                if mode != "scope"
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
