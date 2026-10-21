# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Record reproducible serving probes; generated text is not an accuracy score."""

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base = f"http://127.0.0.1:{args.port}"

    def request(path, payload=None, timeout=180):
        data = None if payload is None else json.dumps(payload).encode()
        req = urllib.request.Request(
            base + path, data=data, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.load(response)

    deadline = time.monotonic() + 600
    last_error = None
    while time.monotonic() < deadline:
        try:
            request("/health", timeout=3)
            break
        except json.JSONDecodeError:
            # vLLM /health returns HTTP 200 with an empty body.
            break
        except (OSError, urllib.error.URLError) as exc:
            last_error = str(exc)
            time.sleep(3)
    else:
        raise RuntimeError(f"server did not become ready: {last_error}")

    model = "Qwen/Qwen3-0.6B"
    long_text = (
        "请阅读记录，最后用中文回答中国的首都是哪里。\n"
        + "记录：北京是中国的首都，上海是中国的城市。\n" * 20
        + "\n中国的首都是"
    )
    tokens = request("/tokenize", {"model": model, "prompt": long_text})["tokens"]
    assert len(tokens) >= 135
    long_prompt = tokens[:120] + tokens[-15:]
    cases = [
        ("warmup", "你好", 2),
        ("capital", "北京是中国的", 32),
        ("arithmetic", "1+1=", 16),
        ("english", "The capital of France is", 16),
        ("cross_page", long_prompt, 32),
        ("capital_replay", "北京是中国的", 32),
    ]
    results = []
    try:
        for name, prompt, max_tokens in cases:
            started = time.monotonic()
            result = request(
                "/v1/completions",
                {
                    "model": model,
                    "prompt": prompt,
                    "max_tokens": max_tokens,
                    "temperature": 0,
                    "seed": 123,
                },
            )
            row = {
                "case": name,
                "seconds": time.monotonic() - started,
                "response": result,
            }
            results.append(row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
    finally:
        args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2))
    by_name = {row["case"]: row["response"] for row in results}
    assert by_name["cross_page"]["usage"]["prompt_tokens"] == 135
    replay_equal = (
        by_name["capital"]["choices"][0]["text"]
        == by_name["capital_replay"]["choices"][0]["text"]
    )
    print(
        json.dumps(
            {"cross_page_prompt_tokens": 135, "capital_replay_equal": replay_equal}
        ),
        flush=True,
    )
    if not replay_equal:
        raise AssertionError("greedy output changed after intervening requests")


if __name__ == "__main__":
    main()
