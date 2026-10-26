# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Summarize matched model ablations and text/log-probability changes."""

import argparse
import csv
import json
import random
import statistics
from pathlib import Path


def percentile(values, q):
    values = sorted(values)
    pos = (len(values) - 1) * q
    low = int(pos)
    return values[low] + (values[min(low + 1, len(values) - 1)] - values[low]) * (
        pos - low
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--exclude-formats", default="")
    args = parser.parse_args()
    rows, quality = [], []
    for path in sorted(args.input_dir.glob("*-serving.json")):
        name = path.name.removesuffix("-serving.json")
        if name in args.exclude_formats.split(","):
            continue
        data = json.loads(path.read_text())
        if "modes" not in data and data.get("paired_pipeline"):
            after_level = data["after_level"]
            data["modes"] = [0, after_level]
            data["all_stage_texts_equal"] = data.get("pipeline_outputs_equal")
            for row in data["requests"] + data["quality_probes"]:
                row["pipeline_mode"] = after_level if row["pipeline_optimized"] else 0
        if not data.get("all_stage_texts_equal"):
            raise ValueError(f"missing or failed cross-stage output check: {name}")
        for profile in data["profiles"]:
            for mode in data["modes"]:
                samples = [
                    r
                    for r in data["requests"]
                    if r["profile"] == profile and r["pipeline_mode"] == mode
                ]
                baseline = [
                    r
                    for r in data["requests"]
                    if r["profile"] == profile and r["pipeline_mode"] == 0
                ]
                by_pair = {(r["round"], r["pair"]): r for r in baseline}
                pairs = [
                    (by_pair[(r["round"], r["pair"])]["seconds"], r["seconds"])
                    for r in samples
                ]
                rng = random.Random(20261026)
                boot = []
                for _ in range(10000):
                    draw = rng.choices(pairs, k=len(pairs))
                    boot.append(
                        (1 - sum(b for a, b in draw) / sum(a for a, b in draw)) * 100
                    )
                median = statistics.median(r["seconds"] for r in samples)
                base_median = statistics.median(r["seconds"] for r in baseline)
                rows.append(
                    {
                        "format": name,
                        "profile": profile,
                        "input_tokens": data["prompt_token_counts"][profile],
                        "mode": mode,
                        "samples": len(samples),
                        "p50_seconds": median,
                        "p95_seconds": percentile(
                            [r["seconds"] for r in samples], 0.95
                        ),
                        "ttft_p50_ms": statistics.median(r["ttft_ms"] for r in samples),
                        "tpot_p50_ms": statistics.median(r["tpot_ms"] for r in samples),
                        "p50_reduction_pct": (1 - median / base_median) * 100,
                        "paired_mean_reduction_pct": (
                            1 - sum(b for a, b in pairs) / sum(a for a, b in pairs)
                        )
                        * 100,
                        "paired_mean_ci95_low": percentile(boot, 0.025),
                        "paired_mean_ci95_high": percentile(boot, 0.975),
                    }
                )
        comparisons = []
        for mode in data["modes"]:
            maximum = 0.0
            changed_tokens = 0
            compared_logprobs = 0
            base = {
                (r["profile"], r["round"], r["pair"]): r
                for r in data["requests"]
                if r["pipeline_mode"] == 0
            }
            for row in data["requests"]:
                if row["pipeline_mode"] != mode:
                    continue
                original = base[(row["profile"], row["round"], row["pair"])]
                if row["text"] != original["text"]:
                    raise ValueError(f"paired generated text changed: {name}, {mode}")
                if len(row.get("token_logprobs", [])) != len(
                    original.get("token_logprobs", [])
                ):
                    raise ValueError(f"generated token count changed: {name}, {mode}")
                for (a, ap), (b, bp) in zip(
                    original.get("token_logprobs", []), row.get("token_logprobs", [])
                ):
                    changed_tokens += a != b
                    if ap is not None and bp is not None:
                        compared_logprobs += 1
                        maximum = max(maximum, abs(ap - bp))
            base_probes = {
                r["prompt"]: r["response"]
                for r in data["quality_probes"]
                if r["pipeline_mode"] == 0
            }
            for probe in data["quality_probes"]:
                if probe["pipeline_mode"] != mode:
                    continue
                observed = probe["response"]
                original = base_probes[probe["prompt"]]
                if observed["text"] != original["text"]:
                    raise ValueError(f"quality probe changed: {name}, {mode}")
                for (a, ap), (b, bp) in zip(
                    original.get("token_logprobs", []),
                    observed.get("token_logprobs", []),
                ):
                    changed_tokens += a != b
                    if ap is not None and bp is not None:
                        compared_logprobs += 1
                        maximum = max(maximum, abs(ap - bp))
            comparisons.append(
                {
                    "mode": mode,
                    "max_generated_token_logprob_delta": maximum
                    if compared_logprobs
                    else None,
                    "changed_generated_tokens": changed_tokens
                    if compared_logprobs
                    else None,
                    "compared_logprobs": compared_logprobs,
                }
            )
        quality.append(
            {
                "format": name,
                "all_stage_texts_equal": True,
                "quality_prompts": len({r["prompt"] for r in data["quality_probes"]}),
                "quality_probe_requests": len(data["quality_probes"]),
                "logprob_comparisons": comparisons,
            }
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "comparison.json").write_text(
        json.dumps({"latency": rows, "quality": quality}, indent=2)
    )
    with (args.output_dir / "comparison.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"groups": len(rows), "formats": len(quality)}))


if __name__ == "__main__":
    main()
