#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare PSP-KV against the FlashAttention backend on greedy decode.

Runs the same prompt through both backends and reports the generated tokens and
whether they match. PSP-KV must reproduce the FlashAttention output exactly for
every supported inner precision, which makes this the end-to-end correctness
gate (`./run.sh --test-integration`).

Usage:
    python scripts/compare_backends.py
    PSPKV_TEST_MODEL=Qwen/Qwen3-1.7B python scripts/compare_backends.py
    PSPKV_TEST_PRECISIONS=int8,e4m3,int4 \
        python scripts/compare_backends.py --no-cuda-graph

Exit status is non-zero when any precision disagrees with the baseline.
"""

import argparse
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("PSPKV_FORMAT", "1")
os.environ.setdefault("PSPKV_ADAPTIVE_PRECISION", "0")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

from vllm import LLM, SamplingParams  # noqa: E402

MODEL = os.environ.get("PSPKV_TEST_MODEL", "Qwen/Qwen3-0.6B")
PRECISIONS = [
    p.strip()
    for p in os.environ.get("PSPKV_TEST_PRECISIONS", "int8,e4m3,int4").split(",")
    if p.strip()
]
PROMPT = "The quick brown fox jumps over the lazy dog. " * 4
BLOCK_SIZE = int(os.environ.get("PSPKV_BLOCK_SIZE", "128"))
N_TOKENS = 8
PROMPT_TOKENS = 128


def generate(backend: str, precision: str, eager: bool) -> list[int]:
    """Greedy tokens for one backend/precision combination."""
    os.environ["PSPKV_PRECISION"] = precision
    llm = LLM(
        model=MODEL,
        attention_backend=backend,
        block_size=BLOCK_SIZE,
        enforce_eager=eager,
        max_model_len=256,
        gpu_memory_utilization=float(os.environ.get("PSPKV_TEST_GPU_UTIL", "0.6")),
        dtype="float16",
        enable_prefix_caching=False,
    )
    tokenizer = llm.get_tokenizer()
    ids = tokenizer(PROMPT, add_special_tokens=False)["input_ids"][:PROMPT_TOKENS]
    out = llm.generate(
        [ids],
        SamplingParams(max_tokens=N_TOKENS, temperature=0, ignore_eos=True),
        use_tqdm=False,
    )
    tokens = out[0].outputs[0].token_ids
    del llm
    return tokens


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-cuda-graph",
        action="store_true",
        help="also exercise one PSP-KV run with CUDA graphs enabled",
    )
    args = parser.parse_args()

    print(f"model: {MODEL}   block size: {BLOCK_SIZE}")
    baseline = generate("FLASH_ATTN", PRECISIONS[0], eager=True)
    print(f"baseline       : {baseline}")

    failures = 0
    for precision in PRECISIONS:
        tokens = generate("PSP_KV", precision, eager=True)
        ok = tokens == baseline
        failures += 0 if ok else 1
        print(f"psp {precision:9s}: {tokens}  match={ok}")

    if not args.no_cuda_graph:
        tokens = generate("PSP_KV", PRECISIONS[0], eager=False)
        ok = tokens == baseline
        failures += 0 if ok else 1
        print(f"psp {PRECISIONS[0]:9s} (cuda graphs): {tokens}  match={ok}")

    if failures:
        print(f"FAILED: {failures} configuration(s) diverged from FlashAttention")
        return 1
    print("OK: every configuration reproduces the FlashAttention output")
    return 0


if __name__ == "__main__":
    sys.exit(main())
