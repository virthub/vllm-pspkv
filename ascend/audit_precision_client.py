# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Small completion probes for precision-route functionality and output review.

This is an eager, single-request smoke audit, not a throughput benchmark.
"""

import argparse
import json
import time
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B")
    parser.add_argument("--tokens", type=int, default=16)
    parser.add_argument("--repeats", type=int, default=2)
    args = parser.parse_args()
    base = f"http://127.0.0.1:{args.port}"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def post(path, payload):
        request = urllib.request.Request(
            base + path,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        started = time.perf_counter()
        with opener.open(request, timeout=240) as response:
            result = json.load(response)
        return result, time.perf_counter() - started

    text = (
        "请阅读记录，最后用中文回答中国的首都是哪里。\n"
        + "记录：北京是中国的首都，上海是中国的城市。\n" * 20
        + "\n中国的首都是"
    )
    tokenized, _ = post("/tokenize", {"model": args.model, "prompt": text})
    tokens = tokenized["tokens"]
    profiles = {"short": "北京是中国的", "cross_page": tokens[:120] + tokens[-15:]}
    data = {
        "case": args.case,
        "model": args.model,
        "output_tokens": args.tokens,
        "scope": "single-request eager smoke audit; no performance significance claim",
        "requests": [],
        "quality_probes": [],
        "complete": False,
    }

    def generate(prompt):
        result, elapsed = post(
            "/v1/completions",
            {
                "model": args.model,
                "prompt": prompt,
                "temperature": 0,
                "seed": 123,
                "max_tokens": args.tokens,
                "ignore_eos": True,
            },
        )
        assert result["usage"]["completion_tokens"] == args.tokens
        return {
            "seconds": elapsed,
            "usage": result["usage"],
            "text": result["choices"][0]["text"],
        }

    generate(profiles["short"])
    for repeat in range(args.repeats):
        for profile, prompt in profiles.items():
            row = generate(prompt)
            if profile == "cross_page":
                assert row["usage"]["prompt_tokens"] == 135
            row.update(profile=profile, repeat=repeat)
            data["requests"].append(row)
            args.output.write_text(json.dumps(data, ensure_ascii=False, indent=2))
            print(json.dumps(row, ensure_ascii=False), flush=True)
    for prompt in ("1+1=", "The capital of France is"):
        data["quality_probes"].append({"prompt": prompt, "response": generate(prompt)})
    data["complete"] = True
    args.output.write_text(json.dumps(data, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
