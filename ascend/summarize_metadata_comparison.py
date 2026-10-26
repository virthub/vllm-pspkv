# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Summarize adjacent, randomized before/after metadata-reuse request pairs."""

import argparse
import csv
import json
import math
import random
import statistics
from pathlib import Path


def percentile(values, fraction):
    ordered = sorted(values)
    pos = (len(ordered) - 1) * fraction
    low, high = math.floor(pos), math.ceil(pos)
    return ordered[low] + (ordered[high] - ordered[low]) * (pos - low)


def paired_interval(pairs):
    rng = random.Random(20261025)
    samples = []
    for _ in range(10000):
        chosen = [pairs[rng.randrange(len(pairs))] for _ in pairs]
        samples.append(
            100 * (1 - sum(a for b, a in chosen) / sum(b for b, a in chosen))
        )
    return percentile(samples, 0.025), percentile(samples, 0.975)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, outputs = [], {}
    baseline_path = args.input_dir / "stock-bf16-serving.json"
    baseline = json.loads(baseline_path.read_text()) if baseline_path.exists() else None
    for name in ("basic", "enhanced", "btkv"):
        path = args.input_dir / f"{name}-serving.json"
        if not path.exists():
            continue
        data = json.loads(path.read_text())
        assert data["paired_metadata"]
        outputs[name] = {}
        for profile in ("short", "cross_page"):
            records = [r for r in data["requests"] if r["profile"] == profile]
            before = [r for r in records if not r["metadata_reuse"]]
            after = [r for r in records if r["metadata_reuse"]]
            by_pair = {}
            for r in records:
                by_pair.setdefault((r["round"], r["pair"]), {})[r["metadata_reuse"]] = r
            assert all(len(pair) == 2 for pair in by_pair.values())
            pairs = [
                (v[False]["seconds"], v[True]["seconds"]) for v in by_pair.values()
            ]
            low, high = paired_interval(pairs)
            b = statistics.median(r["seconds"] for r in before)
            a = statistics.median(r["seconds"] for r in after)
            reference = (
                statistics.median(
                    r["seconds"]
                    for r in baseline["requests"]
                    if r["profile"] == profile
                )
                if baseline
                else None
            )
            equal = {r["text"] for r in before} == {r["text"] for r in after}
            outputs[name][profile] = equal
            rows.append(
                {
                    "format": name,
                    "profile": profile,
                    "pairs": len(pairs),
                    "bf16_p50_s": reference,
                    "before_p50_s": b,
                    "after_p50_s": a,
                    "p50_reduction_percent": 100 * (1 - a / b),
                    "paired_mean_reduction_percent": 100
                    * (1 - sum(a for b, a in pairs) / sum(b for b, a in pairs)),
                    "paired_mean_bootstrap95_low": low,
                    "paired_mean_bootstrap95_high": high,
                    "before_p95_s": percentile([r["seconds"] for r in before], 0.95),
                    "after_p95_s": percentile([r["seconds"] for r in after], 0.95),
                    "before_tpot_ms": statistics.median(r["tpot_ms"] for r in before),
                    "after_tpot_ms": statistics.median(r["tpot_ms"] for r in after),
                    "before_ttft_ms": statistics.median(r["ttft_ms"] for r in before),
                    "after_ttft_ms": statistics.median(r["ttft_ms"] for r in after),
                    "output_text_equal": equal,
                }
            )
    result = {
        "rows": rows,
        "output_text_equal": outputs,
        "interval_scope": (
            "Paired mean latency reduction, not the P50 reduction; "
            "10000 paired bootstrap resamples, no cross-server/run confidence claim."
        ),
    }
    (args.output_dir / "comparison.json").write_text(json.dumps(result, indent=2))
    if rows:
        with (args.output_dir / "comparison.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
