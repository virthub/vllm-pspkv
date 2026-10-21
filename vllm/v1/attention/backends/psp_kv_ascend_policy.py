# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Quality-gated precision choice for completed PSP-KV Ascend blocks.

The live variable-width Ascend arena calls this selector after a full block's
K/V values are available. It is not an allocation-time predictor.
"""

import math
from dataclasses import dataclass

import torch

from vllm.v1.attention.backends.psp_kv_ascend_packed import PackedKVCache
from vllm.v1.attention.backends.psp_kv_ascend_reference import (
    decode_e2m1,
    decode_e4m3,
    decode_e5m2,
    encode_e2m1,
    encode_e4m3,
    encode_e5m2,
)


@dataclass(frozen=True)
class BlockPrecisionChoice:
    """Selected byte format and scales for one complete physical block."""

    code: int  # 0=BF16/FP16, 1=E4M3, 2=E5M2, 3=E2M1
    scales: torch.Tensor  # float32 [heads] or [2, heads, groups]
    relative_rmse: float


def _relative_rmse(
    key: torch.Tensor,
    value: torch.Tensor,
    reconstructed_key: torch.Tensor,
    reconstructed_value: torch.Tensor,
) -> float:
    if not bool(
        torch.isfinite(reconstructed_key).all()
        and torch.isfinite(reconstructed_value).all()
    ):
        return math.inf

    # A large K outlier must not hide a completely erased V stream. Check
    # each stream and head independently before accepting a compressed block.
    def stream_error(actual: torch.Tensor, reconstructed: torch.Tensor) -> torch.Tensor:
        normalizer = actual.abs().amax(dim=(0, 2)).clamp_min(1e-30)
        signal = actual / normalizer[None, :, None]
        error = (reconstructed - actual) / normalizer[None, :, None]
        error_energy = error.square().sum(dim=(0, 2))
        signal_energy = signal.square().sum(dim=(0, 2)).clamp_min(1e-30)
        return (error_energy / signal_energy).sqrt()

    return float(
        torch.maximum(
            stream_error(key, reconstructed_key),
            stream_error(value, reconstructed_value),
        ).max()
    )


def select_completed_block_precision(
    key: torch.Tensor,
    value: torch.Tensor,
    max_relative_rmse: float,
    fp4_group_size: int | None = None,
    allow_lossless: bool = True,
) -> BlockPrecisionChoice:
    """Pick the smallest format whose worst-head K/V error meets the limit.

    ``key`` and ``value`` are full [tokens, KV heads, head dim] blocks. The
    decision is deterministic for fixed inputs and checks FP4, E4M3, then
    E5M2 before preserving the model's BF16/FP16 values. The legacy FP4
    format shares one power-of-two scale between K and V for each KV head.
    Setting ``fp4_group_size`` gives K and V separate scales for each group
    of dimensions, matching the grouped online FP4 reference format.

    With ``allow_lossless=False`` the choice can never exceed the FP8 worst
    case: when no FP4/FP8 format meets the limit, E5M2 is selected and the
    actual error is reported even if it exceeds ``max_relative_rmse``. This
    keeps the online arena within its fixed byte budget, because BF16 blocks
    would need two bytes per element. The threshold is therefore not a hard
    model-quality guarantee in the live path.
    """
    if key.ndim != 3 or key.shape != value.shape or min(key.shape) <= 0:
        raise ValueError("Precision selection requires matching nonempty K/V blocks")
    if key.device != value.device or key.dtype != value.dtype:
        raise ValueError("Precision selection requires matching K/V device and dtype")
    if key.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("Precision selection requires BF16 or FP16 model K/V")
    if not math.isfinite(max_relative_rmse) or max_relative_rmse < 0:
        raise ValueError("Precision error limit must be finite and nonnegative")
    if fp4_group_size is not None and (
        fp4_group_size <= 0 or key.shape[2] % fp4_group_size
    ):
        raise ValueError("FP4 group size must divide the head dimension")

    key_float, value_float = key.float(), value.float()
    if not bool(torch.isfinite(key_float).all() and torch.isfinite(value_float).all()):
        raise ValueError("Precision selection requires finite K/V values")
    scale_shape = (
        (2, key.shape[1], key.shape[2] // fp4_group_size)
        if fp4_group_size is not None
        else (key.shape[1],)
    )
    ones = torch.ones(scale_shape, dtype=torch.float32, device=key.device)

    if key.shape[2] % 2 == 0:
        if fp4_group_size is None:
            maxima = torch.maximum(
                key_float.abs().amax(dim=(0, 2)),
                value_float.abs().amax(dim=(0, 2)),
            )
        else:
            maxima = (
                torch.stack((key_float, value_float), dim=1)
                .abs()
                .reshape(
                    key.shape[0],
                    2,
                    key.shape[1],
                    key.shape[2] // fp4_group_size,
                    fp4_group_size,
                )
                .amax(dim=(0, 4))
            )
        scales = torch.where(
            maxima > 0,
            torch.pow(2.0, torch.ceil(torch.log2(maxima / 3.0))).clamp_min(2**-24),
            torch.ones_like(maxima),
        )
        scales = torch.where(maxima > 3.0 * scales, scales * 2.0, scales)
        if fp4_group_size is None:
            key_factor = value_factor = scales[None, :, None]
        else:
            key_factor = scales[0].repeat_interleave(fp4_group_size, dim=-1)[None]
            value_factor = scales[1].repeat_interleave(fp4_group_size, dim=-1)[None]
        reconstructed_key = (
            decode_e2m1(encode_e2m1(key_float / key_factor)) * key_factor
        )
        reconstructed_value = (
            decode_e2m1(encode_e2m1(value_float / value_factor)) * value_factor
        )
        error = _relative_rmse(
            key_float, value_float, reconstructed_key, reconstructed_value
        )
        if error <= max_relative_rmse:
            return BlockPrecisionChoice(3, scales, error)

    e5m2_error: float = 0.0
    for code, encode, decode in (
        (1, encode_e4m3, decode_e4m3),
        (2, encode_e5m2, decode_e5m2),
    ):
        reconstructed_key = decode(encode(key_float))
        reconstructed_value = decode(encode(value_float))
        error = _relative_rmse(
            key_float, value_float, reconstructed_key, reconstructed_value
        )
        if code == 2:
            e5m2_error = error
        if error <= max_relative_rmse:
            return BlockPrecisionChoice(code, ones, error)

    if not allow_lossless:
        return BlockPrecisionChoice(2, ones, e5m2_error)
    return BlockPrecisionChoice(0, ones, 0.0)


def pack_completed_blocks_by_error(
    key: torch.Tensor,
    value: torch.Tensor,
    max_relative_rmse: float,
    value_layout: str = "token_major",
    fp4_group_size: int | None = None,
) -> tuple[PackedKVCache, torch.Tensor, tuple[BlockPrecisionChoice, ...]]:
    """Select formats and pack full blocks into genuinely variable-size arenas.

    Inputs are [blocks, tokens, KV heads, head dim]. The returned scale table
    must be applied to FP4 rows after ``cache.gather``. The byte offsets and
    precision table live in ``cache``. This is an offline reference operation;
    it does not manage vLLM's live physical block IDs or lifetime events.
    """
    if key.ndim != 4 or key.shape != value.shape or min(key.shape) <= 0:
        raise ValueError("Packed precision selection requires full K/V blocks")
    choices = tuple(
        select_completed_block_precision(
            key[index], value[index], max_relative_rmse, fp4_group_size
        )
        for index in range(key.shape[0])
    )
    cache = PackedKVCache(
        [choice.code for choice in choices],
        key.shape[1],
        key.shape[2],
        key.shape[3],
        key.device,
        scalar_dtype=key.dtype,
        value_layout=value_layout,
    )
    scales = torch.stack([choice.scales for choice in choices])
    for block_id, choice in enumerate(choices):
        slots = torch.arange(
            block_id * cache.block_size,
            (block_id + 1) * cache.block_size,
            device=key.device,
        )
        if choice.code == 3 and fp4_group_size is not None:
            key_divisor = choice.scales[0].repeat_interleave(fp4_group_size, dim=-1)[
                None
            ]
            value_divisor = choice.scales[1].repeat_interleave(fp4_group_size, dim=-1)[
                None
            ]
        else:
            key_divisor = value_divisor = (
                choice.scales[None, :, None] if choice.code == 3 else 1.0
            )
        cache.write(
            slots,
            key[block_id].float() / key_divisor,
            value[block_id].float() / value_divisor,
        )
    return cache, scales, choices
