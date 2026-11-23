#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline accuracy attribution for the PSP-KV cache formats.

Captures the fp16 K/V that the prefill path hands to the quantizer from a real
model, then answers the format questions analytically — no engine, no second
model load:

* scale granularity: per-16-group vs per-32-group vs per-head scales, for fp8,
  int8 and fp4 (the writer broadcasts fp8 scales across a head, while int8 and
  fp4 keep per-group scales);
* format pairs: attention-output error for every (K, V) format combination;
* probability quantisation: the cost of the integer PV path, which rounds the
  softmax weights to int8 (``round(p * 127)``).

Errors are relative L2 against a reference built from the *same* quantized
values, so quantisation error is charged to the format, not to the comparison.

Usage:
    python scripts/eval_kv_formats.py
    PSPKV_EVAL_MODEL=Qwen/Qwen3-4B PSPKV_EVAL_TOKENS=256 \
        python scripts/eval_kv_formats.py
"""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("PSPKV_FORMAT", "1")
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

import torch  # noqa: E402

from vllm import LLM, SamplingParams  # noqa: E402
from vllm.v1.worker import psp_kv_state as state_module  # noqa: E402

MODEL = os.environ.get("PSPKV_EVAL_MODEL", "Qwen/Qwen3-4B")
BLOCK_SIZE = int(os.environ.get("PSPKV_BLOCK_SIZE", "128"))
TOKENS = int(os.environ.get("PSPKV_EVAL_TOKENS", "256"))
GROUP = 16
GQA = 4
FP8_MAX = 448.0
INT8_MAX = 127.0
FP4_MAX = 7.0
PROMPT = "The history of the printing press begins in the fifteenth century. " * 10

_CAPTURED: list[torch.Tensor] = []
_ORIGINAL_QUANTIZE = state_module.PSPKVState.quantize_prefill


def _capture(self, *args, **kwargs):
    """Grab the pre-quantisation fp16 K/V of the first prefill."""
    if not _CAPTURED and self.prefill_kv_scratch is not None:
        _CAPTURED.append(
            self.prefill_kv_scratch[0, :TOKENS].detach().float().cpu().clone()
        )
    return _ORIGINAL_QUANTIZE(self, *args, **kwargs)


def _engine_quantize(target, scale, quantizer):
    """Quantise/dequantise ``target`` with one scale per 16-dim group."""
    out = torch.empty_like(target)
    for g in range(0, target.shape[-1], GROUP):
        window = target[..., g : g + GROUP]
        local = window.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / scale
        out[..., g : g + GROUP] = quantizer(window, local.expand_as(window))
    return out


def _engine_quantize_wide(target, scale, quantizer, group):
    """Same, with a configurable scale-group width."""
    out = torch.empty_like(target)
    for g in range(0, target.shape[-1], group):
        window = target[..., g : g + group]
        local = window.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / scale
        out[..., g : g + group] = quantizer(window, local.expand_as(window))
    return out


def _head_quantize(target, scale, quantizer):
    """Quantise with one scale for the whole 128-dim head."""
    local = target.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8) / scale
    return quantizer(target, local.expand_as(target))


def _q_fp8(value, scale):
    return (value / scale).to(torch.float8_e4m3fn).float() * scale


def _q_int8(value, scale):
    return torch.round(value / scale).clamp(-INT8_MAX, INT8_MAX) * scale


def _q_fp4(value, scale):
    return torch.round(value / scale).clamp(-FP4_MAX, FP4_MAX) * scale


def _relative_error(estimate: torch.Tensor, reference: torch.Tensor) -> float:
    delta = (estimate - reference).float()
    return (delta.norm() / reference.norm()).item()


def _attention(query, keys, values) -> torch.Tensor:
    heads, dim = query.shape
    out = torch.empty_like(query)
    softmax_scale = dim**-0.5
    for head in range(heads):
        kv_head = head // GQA
        probs = torch.softmax((query[head] @ keys[:, kv_head].T) * softmax_scale, -1)
        out[head] = probs @ values[:, kv_head]
    return out


def main() -> int:
    state_module.PSPKVState.quantize_prefill = _capture
    os.environ["PSPKV_PRECISION"] = "int8"
    engine = LLM(
        model=MODEL,
        attention_backend="PSP_KV",
        block_size=BLOCK_SIZE,
        enforce_eager=True,
        max_model_len=512,
        gpu_memory_utilization=float(os.environ.get("PSPKV_EVAL_GPU_UTIL", "0.7")),
        dtype="float16",
        enable_prefix_caching=False,
    )
    tokenizer = engine.get_tokenizer()
    ids = tokenizer(PROMPT, add_special_tokens=False)["input_ids"][:TOKENS]
    engine.generate([ids], SamplingParams(max_tokens=2, temperature=0), use_tqdm=False)
    del engine
    if not _CAPTURED:
        print("no K/V capture: the prefill path did not run")
        return 1

    kv = _CAPTURED[0]
    tokens, _, num_kv_heads, head_dim = kv.shape
    device = kv.device
    # One decode-style query token for all heads: the invariants here are about
    # reading the cache, not about processing many query positions.
    query = torch.randn(1, num_kv_heads * GQA, head_dim, device=device)
    query = query * kv[:, 0].abs().mean() * 3
    query = query[0]
    keys_ref, values_ref = kv[:, 0], kv[:, 1]
    reference = _attention(query, keys_ref, values_ref)

    print(
        f"model: {MODEL}   tokens: {tokens}   "
        f"kv heads: {num_kv_heads}   dim: {head_dim}"
    )

    print("\nscale granularity (attention-output relative error)")
    for name, quantizer, scale, layout in (
        ("e4m3", _q_fp8, FP8_MAX, "group16"),
        ("int8", _q_int8, INT8_MAX, "group16"),
        ("fp4", _q_fp4, FP4_MAX, "group16"),
    ):
        for group in (16, 32, 64, head_dim):
            if layout == "group16" and group == 16:
                keys = _engine_quantize(keys_ref, scale, quantizer)
                values = _engine_quantize(values_ref, scale, quantizer)
            else:
                keys = _engine_quantize_wide(keys_ref, scale, quantizer, group)
                values = _engine_quantize_wide(values_ref, scale, quantizer, group)
            error = _relative_error(_attention(query, keys, values), reference)
            print(f"  {name:5s} group={group:3d} rel_err={error:.5f}")

    print("\nformat pairs (K, V) -> attention-output relative error")
    for key_name, key_fn, key_scale, key_layout in (
        ("e4m3", _q_fp8, FP8_MAX, _head_quantize),
        ("int8", _q_int8, INT8_MAX, _engine_quantize),
        ("fp4", _q_fp4, FP4_MAX, _engine_quantize),
    ):
        keys = key_layout(keys_ref, key_scale, key_fn)
        for value_name, value_fn, value_scale, value_layout in (
            ("e4m3", _q_fp8, FP8_MAX, _head_quantize),
            ("int8", _q_int8, INT8_MAX, _engine_quantize),
            ("fp4", _q_fp4, FP4_MAX, _engine_quantize),
        ):
            values = value_layout(values_ref, value_scale, value_fn)
            error = _relative_error(_attention(query, keys, values), reference)
            print(f"  K={key_name:5s} V={value_name:5s} rel_err={error:.5f}")

    print("\ninteger PV (softmax probabilities rounded to int8)")
    keys = _head_quantize(keys_ref, FP8_MAX, _q_fp8)
    values = _engine_quantize(values_ref, FP4_MAX, _q_fp4)
    fp32_reference = _attention(query, keys, values)
    hybrid = torch.empty_like(fp32_reference)
    softmax_scale = head_dim**-0.5
    for head in range(query.shape[0]):
        kv_head = head // GQA
        probs = torch.softmax((query[head] @ keys[:, kv_head].T) * softmax_scale, -1)
        rounded = torch.round(probs * 127.0).float()
        for g in range(0, head_dim, GROUP):
            window = values[:, kv_head, g : g + GROUP]
            scale = window.abs().amax().clamp_min(1e-8) / FP4_MAX
            window_int = torch.round(window / scale).clamp(-FP4_MAX, FP4_MAX)
            hybrid[head, g : g + GROUP] = (rounded @ window_int) * (scale / 127.0)
    print(f"  rel_err={_relative_error(hybrid, fp32_reference):.5f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
