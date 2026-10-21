#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Freeze a per-layer allocation-time precision map from calibration logs.

Serve Qwen3 (or the target model) with the dynamic route and sampling on:

    PSPKV_ASCEND_VARIABLE_PRECISION=1 \
    PSPKV_ASCEND_CALIBRATE=1 \
    PSPKV_ASCEND_LOG_FORMAT_COUNTS=1 \
    bash ascend/smoke_910b3.sh

Send representative prompts, capture ``docker logs <container>``, then run:

    python ascend/calibrate_precision.py --log run.log --output map.json

Serve with ``PSPKV_ASCEND_STATIC_PRECISION=map.json`` to freeze the formats
at allocation time: FP4 blocks reserve half the bytes up front, no
completed-block policy or scratch retention runs, and the calibrated scale
makes FP4 writes a single encode. The map records the most common code per
layer (ties resolve to the larger format) and, for FP4 layers, per-KV-head
power-of-two scales with one doubling of headroom applied at runtime.
"""

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import regex as re

_CALIBRATION_LINE = re.compile(
    r"PSP-KV calibration layer=(\S+) code=(\d)(?: groups=(\d+))? "
    r"k_max=(\S+) v_max=(\S+)"
)


def _pow2_scale(maximum: float) -> float:
    """Smallest power-of-two scale that keeps ``maximum`` inside FP4 range."""
    if maximum <= 0 or not math.isfinite(maximum):
        return 1.0
    exponent = math.ceil(math.log2(maximum / 3.0))
    return max(2.0**exponent, 2.0**-24)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--k-code",
        type=int,
        choices=(1, 2, 3),
        default=None,
        help="force the K-stream code (default: the sampled policy choice)",
    )
    parser.add_argument(
        "--v-code",
        type=int,
        choices=(1, 2, 3),
        default=None,
        help="force the V-stream code (default: the sampled policy choice)",
    )
    args = parser.parse_args()

    code_counts: dict[str, Counter] = defaultdict(Counter)
    key_maxima: dict[str, list[list[float]]] = defaultdict(list)
    value_maxima: dict[str, list[list[float]]] = defaultdict(list)
    group_counts: dict[str, set[int]] = defaultdict(set)
    for line in args.log.read_text(encoding="utf-8", errors="replace").splitlines():
        match = _CALIBRATION_LINE.search(line)
        if match is None:
            continue
        layer = match.group(1)
        code_counts[layer][int(match.group(2))] += 1
        group_counts[layer].add(int(match.group(3) or 1))
        key_maxima[layer].append([float(v) for v in match.group(4).split(",")])
        value_maxima[layer].append([float(v) for v in match.group(5).split(",")])
    if not code_counts:
        raise SystemExit("no 'PSP-KV calibration' lines found in the log")

    result: dict[str, dict] = {}
    for layer, counts in sorted(code_counts.items()):
        # Ties resolve to the larger format: code 2 (E5M2) > 1 (E4M3) > 3 (FP4).
        code = sorted(counts.items(), key=lambda item: (-item[1], -item[0]))[0][0]
        k_code = args.k_code if args.k_code is not None else code
        v_code = args.v_code if args.v_code is not None else code
        entry: dict = {
            "code": code,
            "k_code": k_code,
            "v_code": v_code,
            "samples": sum(counts.values()),
        }
        if 3 in (k_code, v_code):
            groups = group_counts[layer].pop()
            if group_counts[layer]:
                raise SystemExit(
                    f"{layer} mixes calibration group counts: {group_counts[layer]}"
                )
            heads = len(key_maxima[layer][0]) // groups
            flattened = [
                max(row[i] for row in key_maxima[layer]) for i in range(heads * groups)
            ]
            flat_values = [
                max(row[i] for row in value_maxima[layer])
                for i in range(heads * groups)
            ]
            entry["groups"] = groups
            if groups == 1:
                entry["k_scales"] = [_pow2_scale(v) for v in flattened]
                entry["v_scales"] = [_pow2_scale(v) for v in flat_values]
            else:
                entry["k_scales"] = [
                    [
                        _pow2_scale(flattened[head * groups + group])
                        for group in range(groups)
                    ]
                    for head in range(heads)
                ]
                entry["v_scales"] = [
                    [
                        _pow2_scale(flat_values[head * groups + group])
                        for group in range(groups)
                    ]
                    for head in range(heads)
                ]
        result[layer] = entry

    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    codes = Counter(entry["code"] for entry in result.values())
    print(
        f"wrote {args.output} for {len(result)} layers: "
        f"E4M3={codes.get(1, 0)} E5M2={codes.get(2, 0)} FP4={codes.get(3, 0)}"
    )
    if not args.output.stat().st_size:
        raise SystemExit("calibration output is empty")


if __name__ == "__main__":
    main()
