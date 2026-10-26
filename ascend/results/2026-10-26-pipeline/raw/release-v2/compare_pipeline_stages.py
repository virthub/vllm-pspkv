# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Randomized four-stage ablations in the same loaded model and decode graph."""

import argparse
import json
import random
import time
import urllib.error
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--control-file", type=Path, required=True)
    parser.add_argument("--modes", default="0,1,2,3")
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--requests-per-profile", type=int, default=4)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--ready-timeout", type=int, default=600)
    parser.add_argument("--long-input-tokens", type=int, default=135)
    args = parser.parse_args()
    base = f"http://127.0.0.1:{args.port}"
    model = args.model
    tested_modes = tuple(int(mode) for mode in args.modes.split(","))
    assert tested_modes and all(mode in (0, 1, 2, 3) for mode in tested_modes)

    def post(path, payload):
        request = urllib.request.Request(
            base + path,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=180) as response:
            return json.load(response)

    deadline = time.monotonic() + args.ready_timeout
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(base + "/health", timeout=3):
                break
        except (OSError, urllib.error.URLError):
            time.sleep(3)
    else:
        raise RuntimeError("server readiness timed out")
    long_text = (
        "请阅读记录，最后用中文回答中国的首都是哪里。\n"
        + "记录：北京是中国的首都，上海是中国的城市。\n" * 256
        + "\n中国的首都是"
    )
    tokens = post("/tokenize", {"model": model, "prompt": long_text})["tokens"]
    assert 15 <= args.long_input_tokens <= len(tokens)
    profiles = {
        "short": "北京是中国的",
        "cross_page": tokens[: args.long_input_tokens - 15] + tokens[-15:],
    }
    prompt_counts = {
        "short": len(
            post("/tokenize", {"model": model, "prompt": profiles["short"]})["tokens"]
        ),
        "cross_page": args.long_input_tokens,
    }
    data = {
        "model": model,
        "rounds": args.rounds,
        "requests_per_profile_per_round": args.requests_per_profile,
        "warmups_per_profile": args.warmups,
        "output_tokens": 32,
        "concurrency": 1,
        "profiles": profiles,
        "prompt_token_counts": prompt_counts,
        "requests": [],
        "paired_pipeline": args.control_file is not None,
        "modes": tested_modes,
    }

    def generate(prompt):
        payload = {
            "model": model,
            "prompt": prompt,
            "max_tokens": 32,
            "temperature": 0,
            "seed": 123,
            "ignore_eos": True,
            "stream": True,
            "stream_options": {"include_usage": True},
            "logprobs": 1,
        }
        request = urllib.request.Request(
            base + "/v1/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        started = time.perf_counter()
        first = last = None
        text, usage = "", None
        logprobs = []
        with urllib.request.urlopen(request, timeout=180) as response:
            for line in response:
                if not line.startswith(b"data: "):
                    continue
                event = line[6:].strip()
                if event == b"[DONE]":
                    break
                chunk = json.loads(event)
                if "error" in chunk:
                    raise RuntimeError(chunk["error"])
                if chunk.get("usage"):
                    usage = chunk["usage"]
                for choice in chunk.get("choices", []):
                    if choice.get("logprobs"):
                        logprobs.extend(
                            zip(
                                choice["logprobs"]["tokens"],
                                choice["logprobs"]["token_logprobs"],
                            )
                        )
                    piece = choice.get("text", "")
                    if piece:
                        last = time.perf_counter()
                        first = first if first is not None else last
                        text += piece
        ended = time.perf_counter()
        assert usage is not None and usage["completion_tokens"] == 32
        assert first is not None and last is not None
        return {
            "seconds": ended - started,
            "ttft_ms": (first - started) * 1000,
            "tpot_ms": (last - first) * 1000 / 31,
            "usage": usage,
            "text": text,
            "token_logprobs": logprobs,
        }

    def select_mode(enabled):
        # The server reads once per ForwardContext, before any attention layer.
        if args.control_file:
            args.control_file.write_text(str(enabled))

    for prompt in profiles.values():
        for enabled in tested_modes:
            select_mode(enabled)
            for _ in range(args.warmups):
                generate(prompt)
    ordering = random.Random(20261023)
    for round_index in range(args.rounds):
        schedule = list(profiles) * args.requests_per_profile
        ordering.shuffle(schedule)
        for pair, profile in enumerate(schedule):
            modes = list(tested_modes)
            ordering.shuffle(modes)
            for enabled in modes:
                select_mode(enabled)
                row = generate(profiles[profile])
                row.update(
                    {
                        "profile": profile,
                        "round": round_index,
                        "pair": pair,
                        "pipeline_mode": enabled,
                    }
                )
                assert row["usage"]["prompt_tokens"] == prompt_counts[profile]
                data["requests"].append(row)
                args.output.write_text(json.dumps(data, ensure_ascii=False, indent=2))
                print(
                    json.dumps(
                        {
                            k: row[k]
                            for k in (
                                "profile",
                                "round",
                                "pair",
                                "pipeline_mode",
                                "seconds",
                                "ttft_ms",
                            )
                        }
                    ),
                    flush=True,
                )
    data["quality_probes"] = []
    for enabled in tested_modes:
        select_mode(enabled)
        for prompt in (
            "1+1=",
            "The capital of France is",
            "2+3=",
            "10-4=",
            "3*4=",
            "中国的首都是",
            "日本的首都是",
            "The opposite of hot is",
            "A week has",
            "水的化学式是",
            "太阳从东方",
            "The largest ocean is",
        ):
            data["quality_probes"].append(
                {
                    "prompt": prompt,
                    "pipeline_mode": enabled,
                    "response": generate(prompt),
                }
            )
    text_checks = []
    for profile in profiles:
        outputs = [
            {
                r["text"]
                for r in data["requests"]
                if r["profile"] == profile and r["pipeline_mode"] == mode
            }
            for mode in tested_modes
        ]
        text_checks.append(all(group == outputs[0] for group in outputs))
    for prompt in {r["prompt"] for r in data["quality_probes"]}:
        texts = {
            r["response"]["text"]
            for r in data["quality_probes"]
            if r["prompt"] == prompt
        }
        text_checks.append(len(texts) == 1)
    data["all_stage_texts_equal"] = all(text_checks)
    args.output.write_text(json.dumps(data, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
