# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Summarize sequential, same-card strategy sweeps bracketed by BF16 runs."""

import argparse
import json
import random
import statistics
from pathlib import Path

import regex as re


def load_run(path):
    data = json.loads(path.read_text())
    if not data["complete"]:
        raise ValueError(f"Incomplete run: {path}")
    grouped = {}
    for request in data["requests"]:
        grouped.setdefault(request["profile"], []).append(request)
    for profile, samples in grouped.items():
        if len(samples) != data["pairs_per_profile"]:
            raise ValueError(f"Missing samples: {path}, {profile}")
    return grouped


def capacity(path):
    match = re.search(r"KV cache size: ([\d,]+) tokens", path.read_text())
    return int(match[1].replace(",", "")) if match else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap", type=int, default=5000)
    args = parser.parse_args()
    rng = random.Random(20261026)
    report = {
        "design": "same_card_sequential_bf16_before_candidates_bf16_after",
        "ci": "independent bootstrap of medians; baseline stratified by phase",
        "confidence": 0.95,
        "baseline": [],
        "rows": [],
    }
    for layout in ("basic", "enhanced", "btkv"):
        root = args.results / layout
        before = load_run(root / "bf16-before.json")
        after = load_run(root / "bf16-after.json")
        for profile in before:
            left = [r["seconds"] for r in before[profile]]
            right = [r["seconds"] for r in after[profile]]
            report["baseline"].append(
                {
                    "layout": layout,
                    "profile": profile,
                    "before_seconds": statistics.median(left),
                    "after_seconds": statistics.median(right),
                    "drift_percent": (
                        statistics.median(right) / statistics.median(left) - 1
                    )
                    * 100,
                    "kv_tokens_before": capacity(root / "bf16-before-server.log"),
                    "kv_tokens_after": capacity(root / "bf16-after-server.log"),
                }
            )
        for precision in ("lossless", "e4m3", "e5m2", "fp4", "adaptive"):
            candidate = load_run(root / f"{precision}.json")
            for profile, requests in candidate.items():
                left = [r["seconds"] for r in before[profile]]
                right = [r["seconds"] for r in after[profile]]
                samples = [r["seconds"] for r in requests]
                ratios = sorted(
                    statistics.median(
                        rng.choices(left, k=len(left))
                        + rng.choices(right, k=len(right))
                    )
                    / statistics.median(rng.choices(samples, k=len(samples)))
                    for _ in range(args.bootstrap)
                )
                row = {
                    "layout": layout,
                    "precision": precision,
                    "profile": profile,
                    "samples": len(samples),
                    "seconds": statistics.median(samples),
                    "bf16_seconds": statistics.median(left + right),
                    "speedup": statistics.median(left + right)
                    / statistics.median(samples),
                    "speedup_ci95": [
                        ratios[int(args.bootstrap * 0.025)],
                        ratios[min(len(ratios) - 1, int(args.bootstrap * 0.975))],
                    ],
                    "ttft_ms": statistics.median(r["ttft_ms"] for r in requests),
                    "stream_tpot_estimate_ms": statistics.median(
                        r["tpot_ms"] for r in requests
                    ),
                    "kv_tokens": capacity(root / f"{precision}-server.log"),
                    "outputs": sorted({r["text"] for r in requests}),
                }
                report["rows"].append(row)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
