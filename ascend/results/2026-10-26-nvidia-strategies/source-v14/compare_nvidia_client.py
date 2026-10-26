# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Fixed-token requests with interleaved lengths and optional paired servers."""

import argparse
import json
import random
import time
import urllib.error
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-port", type=int, required=True)
    parser.add_argument("--candidate-port", type=int, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--pairs", type=int, default=12)
    parser.add_argument("--long-tokens", type=int, default=8192)
    parser.add_argument("--output-tokens", type=int, default=32)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B")
    parser.add_argument("--single-mode", choices=("bf16", "candidate"))
    args = parser.parse_args()
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def request(port, path, payload=None):
        return opener.open(
            urllib.request.Request(
                f"http://127.0.0.1:{port}{path}",
                data=None if payload is None else json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
            ),
            timeout=300,
        )

    ports = {"bf16": args.baseline_port, "candidate": args.candidate_port}
    if args.single_mode:
        ports = {args.single_mode: ports[args.single_mode]}
    for port in set(ports.values()):
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            try:
                with request(port, "/health"):
                    break
            except (OSError, urllib.error.URLError):
                time.sleep(2)
        else:
            raise RuntimeError(f"Server {port} did not become ready")
    text = (
        "请阅读记录，最后用中文回答中国的首都是哪里。\n"
        + "记录：北京是中国的首都，上海是中国的城市。\n" * 2000
        + "\n中国的首都是"
    )
    with request(
        next(iter(ports.values())), "/tokenize", {"model": args.model, "prompt": text}
    ) as response:
        tokens = json.load(response)["tokens"]
    profiles = {
        "short_3": "北京是中国的",
        "cross_page_135": tokens[:120] + tokens[-15:],
        f"long_{args.long_tokens}": tokens[: args.long_tokens - 15] + tokens[-15:],
    }
    data = {
        "case": args.case,
        "model": args.model,
        "output_tokens": args.output_tokens,
        "pairs_per_profile": args.pairs,
        "concurrency": 1,
        "paired_same_physical_card": args.single_mode is None,
        "requests": [],
        "complete": False,
    }

    def generate(port, prompt):
        payload = {
            "model": args.model,
            "prompt": prompt,
            "max_tokens": args.output_tokens,
            "temperature": 0,
            "seed": 123,
            "ignore_eos": True,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        started = time.perf_counter()
        first = last = None
        text, usage = "", None
        with request(port, "/v1/completions", payload) as response:
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
                    piece = choice.get("text", "")
                    if piece:
                        last = time.perf_counter()
                        first = first if first is not None else last
                        text += piece
        ended = time.perf_counter()
        assert usage and usage["completion_tokens"] == args.output_tokens
        assert first is not None and last is not None
        return {
            "seconds": ended - started,
            "ttft_ms": (first - started) * 1000,
            "tpot_ms": (last - first) * 1000 / (args.output_tokens - 1),
            "usage": usage,
            "text": text,
        }

    for prompt in profiles.values():
        for port in ports.values():
            for _ in range(2):
                generate(port, prompt)
    ordering = random.Random(20261026)
    # Interleave input lengths too, so graph replay must read changing metadata.
    for pair in range(args.pairs):
        names = list(profiles)
        ordering.shuffle(names)
        for profile in names:
            modes = list(ports)
            ordering.shuffle(modes)
            for mode in modes:
                result = generate(ports[mode], profiles[profile])
                expected_tokens = int(profile.rsplit("_", 1)[1])
                assert result["usage"]["prompt_tokens"] == expected_tokens
                result.update(pair=pair, profile=profile, mode=mode)
                data["requests"].append(result)
                args.output.write_text(json.dumps(data, ensure_ascii=False, indent=2))
                print(
                    json.dumps(
                        {
                            "pair": pair,
                            "profile": profile,
                            "mode": mode,
                            "seconds": result["seconds"],
                        }
                    ),
                    flush=True,
                )
    data["complete"] = True
    args.output.write_text(json.dumps(data, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
