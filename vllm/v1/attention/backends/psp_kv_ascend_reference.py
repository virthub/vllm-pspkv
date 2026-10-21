# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Reference implementation of PSP-KV FP8 caching and attention.

This matches the fixed-stride E4M3 path in
``csrc/attention/psp_kv/common/quantization_utils.cuh``. It uses integer
operations instead of a native ``torch.float8`` cast, which is unavailable on
some Ascend 910B software stacks. The kernels here prioritize correctness
over throughput and are intended for eager-mode bring-up.
"""

import logging

import torch
import torch.nn.functional as F

from vllm.v1.attention.backends.psp_kv_ascend_native import load_native_fp8_ops

logger = logging.getLogger(__name__)


def encode_e4m3(values: torch.Tensor) -> torch.Tensor:
    """Encode fp16/bf16/fp32 values into PSP-KV E4M3 bytes (unit scale)."""
    bits = values.float().contiguous().view(torch.int32)
    sign = (bits >> 31) & 1
    magnitude = bits & 0x7FFFFFFF
    exponent = ((magnitude >> 23) & 0xFF) - 127 + 7
    fraction = magnitude & 0x7FFFFF
    mantissa = fraction >> 20
    round_bit = (fraction >> 19) & 1
    sticky = (fraction & 0x7FFFF) != 0
    round_up = (round_bit != 0) & (sticky | ((mantissa & 1) != 0))
    mantissa = mantissa + round_up.to(torch.int32)
    exponent = exponent + (mantissa >> 3)
    mantissa = mantissa & 7

    normal = (sign << 7) | (exponent << 3) | mantissa
    shift = (1 - exponent).clamp(0, 31)
    subnormal = (sign << 7) | (((8 | mantissa) >> shift) & 7)
    encoded = torch.where(exponent <= 0, subnormal, normal)
    encoded = torch.where(exponent < -7, sign << 7, encoded)
    encoded = torch.where(exponent >= 15, (sign << 7) | 0x78, encoded)
    encoded = torch.where(magnitude == 0, 0, encoded)
    encoded = torch.where(magnitude == 0x7F800000, (sign << 7) | 0x78, encoded)
    encoded = torch.where(magnitude > 0x7F800000, 0x7F, encoded)
    return encoded.to(torch.uint8)


def decode_e4m3(values: torch.Tensor) -> torch.Tensor:
    """Decode PSP-KV E4M3 bytes to fp32; 0x7f/0xff denote NaN."""
    bits = values.to(torch.int32)
    sign = (bits >> 7) & 1
    exponent = (bits >> 3) & 15
    mantissa = bits & 7

    # Build the exact IEEE-754 fp32 bits for normal E4M3 values. The CUDA
    # reference treats exponent 15, mantissa 0..6 as finite numbers.
    normal_bits = ((exponent + 120) << 23) | (mantissa << 20) | (sign << 31)
    normal = normal_bits.contiguous().view(torch.float32)
    subnormal = mantissa.to(torch.float32) * (1.0 / 512.0)
    subnormal = torch.where(sign != 0, -subnormal, subnormal)
    decoded = torch.where(exponent == 0, subnormal, normal)
    return torch.where(
        (exponent == 15) & (mantissa == 7),
        torch.full_like(decoded, float("nan")),
        decoded,
    )


def encode_e5m2(values: torch.Tensor) -> torch.Tensor:
    """Encode values using the PSP-KV E5M2 format (unit scale)."""
    bits = values.float().contiguous().view(torch.int32)
    sign = (bits >> 31) & 1
    magnitude = bits & 0x7FFFFFFF
    exponent = ((magnitude >> 23) & 0xFF) - 127 + 15
    fraction = magnitude & 0x7FFFFF
    mantissa = fraction >> 21
    round_bit = (fraction >> 20) & 1
    sticky = (fraction & 0xFFFFF) != 0
    round_up = (round_bit != 0) & (sticky | ((mantissa & 1) != 0))
    mantissa = mantissa + round_up.to(torch.int32)
    exponent = exponent + (mantissa >> 2)
    mantissa = mantissa & 3

    normal = (sign << 7) | (exponent << 2) | mantissa
    shift = (1 - exponent).clamp(0, 31)
    subnormal = (sign << 7) | (((4 | mantissa) >> shift) & 3)
    encoded = torch.where(exponent <= 0, subnormal, normal)
    encoded = torch.where(exponent < -1, sign << 7, encoded)
    encoded = torch.where(exponent >= 31, (sign << 7) | 0x7C, encoded)
    encoded = torch.where(magnitude == 0, 0, encoded)
    encoded = torch.where(magnitude == 0x7F800000, (sign << 7) | 0x7C, encoded)
    encoded = torch.where(magnitude > 0x7F800000, 0x7F, encoded)
    return encoded.to(torch.uint8)


def decode_e5m2(values: torch.Tensor) -> torch.Tensor:
    """Decode PSP-KV E5M2 bytes to fp32."""
    bits = values.to(torch.int32)
    sign = (bits >> 7) & 1
    exponent = (bits >> 2) & 31
    mantissa = bits & 3

    normal_bits = ((exponent + 112) << 23) | (mantissa << 21) | (sign << 31)
    normal = normal_bits.contiguous().view(torch.float32)
    subnormal = mantissa.to(torch.float32) * (1.0 / 65536.0)
    subnormal = torch.where(sign != 0, -subnormal, subnormal)
    infinity = torch.full_like(normal, float("inf"))
    infinity = torch.where(sign != 0, -infinity, infinity)
    decoded = torch.where(exponent == 0, subnormal, normal)
    decoded = torch.where((exponent == 31) & (mantissa == 0), infinity, decoded)
    return torch.where(
        (exponent == 31) & (mantissa != 0),
        torch.full_like(decoded, float("nan")),
        decoded,
    )


def encode_e2m1(values: torch.Tensor) -> torch.Tensor:
    """Encode the PSP-KV FP4 E2M1 nibble (unit scale)."""
    bits = values.float().contiguous().view(torch.int32)
    sign = (bits >> 31) & 1
    magnitude = bits & 0x7FFFFFFF
    exponent = ((magnitude >> 23) & 0xFF) - 127 + 1
    fraction = magnitude & 0x7FFFFF
    mantissa = fraction >> 22
    round_bit = (fraction >> 21) & 1
    sticky = (fraction & 0x1FFFFF) != 0
    round_up = (round_bit != 0) & (sticky | ((mantissa & 1) != 0))
    mantissa = mantissa + round_up.to(torch.int32)
    exponent = exponent + (mantissa >> 1)
    mantissa = mantissa & 1

    normal = (sign << 3) | (exponent << 1) | mantissa
    shift = (1 - exponent).clamp(0, 31)
    subnormal = (sign << 3) | (((2 | mantissa) >> shift) & 1)
    encoded = torch.where(exponent <= 0, subnormal, normal)
    encoded = torch.where(exponent < -1, sign << 3, encoded)
    encoded = torch.where(exponent >= 3, (sign << 3) | 0x6, encoded)
    encoded = torch.where(magnitude == 0, 0, encoded)
    encoded = torch.where(magnitude == 0x7F800000, (sign << 3) | 0x6, encoded)
    encoded = torch.where(magnitude > 0x7F800000, 0x7, encoded)
    return encoded.to(torch.uint8)


def decode_e2m1(values: torch.Tensor) -> torch.Tensor:
    """Decode the low four bits of PSP-KV FP4 E2M1 to fp32."""
    bits = values.to(torch.int32) & 0xF
    sign = (bits >> 3) & 1
    exponent = (bits >> 1) & 3
    mantissa = bits & 1
    normal_bits = ((exponent + 126) << 23) | (mantissa << 22) | (sign << 31)
    normal = normal_bits.contiguous().view(torch.float32)
    subnormal = mantissa.to(torch.float32) * 0.5
    subnormal = torch.where(sign != 0, -subnormal, subnormal)
    infinity = torch.full_like(normal, float("inf"))
    infinity = torch.where(sign != 0, -infinity, infinity)
    decoded = torch.where(exponent == 0, subnormal, normal)
    return torch.where(exponent == 3, infinity, decoded)


def select_adaptive_fp8_blocks(
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slots: torch.Tensor,
    precision_table: torch.Tensor,
    e4m3_max_abs: float,
    log_transitions: bool = False,
) -> None:
    """Choose a format from incoming K/V and keep existing block bytes valid.

    A block starts in E4M3. A value outside its configured safe range promotes
    the whole block to E5M2 for the rest of that block's lifetime. Prior E4M3
    bytes are transcoded before the new slots are written. Slot offset zero
    starts a new lifetime when the allocator reuses a physical block.

    This eager reference policy assumes the usual contiguous, increasing slot
    writes within each block and no prefix sharing or cache transfer.
    """
    block_size = k_cache.shape[1]
    block_ids = slots // block_size
    for block_id in torch.unique(block_ids).tolist():
        incoming = block_ids == block_id
        first_offset = int((slots[incoming] % block_size).min().item())
        if first_offset == 0:
            precision_table[block_id] = 1

        block_key = key[incoming].float()
        block_value = value[incoming].float()
        # NaNs have an explicit code in both formats. Ignore them for the
        # range decision; positive/negative infinity still selects E5M2.
        abs_key = block_key.abs()
        abs_value = block_value.abs()
        max_key = torch.where(torch.isnan(abs_key), 0.0, abs_key)
        max_value = torch.where(torch.isnan(abs_value), 0.0, abs_value)
        needs_e5m2 = bool(
            (max_key > e4m3_max_abs).any() or (max_value > e4m3_max_abs).any()
        )
        if not needs_e5m2 or int(precision_table[block_id].item()) == 2:
            continue

        if first_offset:
            k_prefix = k_cache[block_id, :first_offset]
            v_prefix = v_cache[block_id, :first_offset]
            k_prefix.copy_(encode_e5m2(decode_e4m3(k_prefix)))
            v_prefix.copy_(encode_e5m2(decode_e4m3(v_prefix)))
        precision_table[block_id] = 2
        if log_transitions:
            logger.info(
                "PSP-KV adaptive FP8 block %d: E4M3 -> E5M2 (previous tokens: %d)",
                block_id,
                first_offset,
            )


def reference_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    block_tables: torch.Tensor,
    query_ends: list[int],
    seq_lens: list[int],
    output: torch.Tensor,
    scale: float,
    precision_table: torch.Tensor | None = None,
    adaptive_e4m3_max_abs: float | None = None,
    log_precision_transitions: bool = False,
    native_fp8: bool = False,
    use_sdpa: bool = False,
    native_decode: bool = False,
) -> torch.Tensor:
    """Write compressed KV, then compute causal GQA from the logical blocks.

    ``query_ends`` contains cumulative query-token counts; ``seq_lens``
    contains context lengths including those query tokens. This function
    deliberately accepts the same logical mappings for CPU and NPU tests.
    """
    if k_cache.dtype != torch.uint8 or v_cache.dtype != torch.uint8:
        raise ValueError("PSP-KV Ascend requires uint8 KV cache tensors")
    if k_cache.shape != v_cache.shape or k_cache.ndim != 4:
        raise ValueError("PSP-KV Ascend expects [blocks, slots, heads, dim] K/V caches")
    if not k_cache.is_contiguous() or not v_cache.is_contiguous():
        raise NotImplementedError("PSP-KV Ascend requires contiguous K/V caches")
    block_size = k_cache.shape[1]
    if precision_table is not None:
        if (
            precision_table.dtype != torch.uint8
            or precision_table.ndim != 1
            or precision_table.numel() != k_cache.shape[0]
            or precision_table.device != k_cache.device
        ):
            raise ValueError("PSP-KV precision table must match the physical blocks")
        if bool(((precision_table != 1) & (precision_table != 2)).any()):
            raise NotImplementedError("PSP-KV Ascend supports only E4M3/E5M2 blocks")
    if adaptive_e4m3_max_abs is not None and precision_table is None:
        raise ValueError("PSP-KV adaptive FP8 requires a physical precision table")
    if native_fp8 and (precision_table is None or key.dtype != torch.bfloat16):
        raise ValueError("PSP-KV native FP8 needs BF16 K/V and a precision table")
    if native_decode and (not native_fp8 or query.dtype != torch.bfloat16):
        raise ValueError("PSP-KV native decode needs native FP8 and BF16 queries")
    num_kv_heads, head_size = k_cache.shape[2:]
    num_heads = query.shape[1]
    if key.shape[1:] != (num_kv_heads, head_size) or value.shape[1:] != key.shape[1:]:
        raise ValueError("PSP-KV Ascend cache head dimensions do not match K/V")
    if num_heads % num_kv_heads:
        raise ValueError("PSP-KV Ascend query heads must be divisible by KV heads")

    actual = len(slot_mapping)
    output_view = output.view(-1, num_heads, head_size)
    output_view.zero_()
    if actual == 0:
        return output
    if len(query_ends) != len(seq_lens) or query_ends[-1] != actual:
        raise ValueError("PSP-KV Ascend received inconsistent request lengths")

    slots = slot_mapping.to(torch.int64)
    if adaptive_e4m3_max_abs is not None:
        select_adaptive_fp8_blocks(
            key[:actual],
            value[:actual],
            k_cache,
            v_cache,
            slots,
            precision_table,
            adaptive_e4m3_max_abs,
            log_precision_transitions,
        )
    if native_fp8:
        load_native_fp8_ops().quantize_scatter_bf16(
            key[:actual].contiguous(),
            value[:actual].contiguous(),
            slots.contiguous(),
            precision_table,
            k_cache,
            v_cache,
        )
    else:
        encoded_keys = encode_e4m3(key[:actual])
        encoded_values = encode_e4m3(value[:actual])
        if precision_table is not None:
            use_e5m2 = (
                precision_table.index_select(0, slots // block_size).view(actual, 1, 1)
                == 2
            )
            encoded_keys = torch.where(
                use_e5m2, encode_e5m2(key[:actual]), encoded_keys
            )
            encoded_values = torch.where(
                use_e5m2, encode_e5m2(value[:actual]), encoded_values
            )
        k_cache.view(-1, num_kv_heads, head_size).index_copy_(0, slots, encoded_keys)
        v_cache.view(-1, num_kv_heads, head_size).index_copy_(0, slots, encoded_values)

    for req, seq_len in enumerate(seq_lens):
        start = 0 if req == 0 else query_ends[req - 1]
        end = query_ends[req]
        num_queries = end - start
        if num_queries == 0:
            continue
        prefix_len = seq_len - num_queries
        if prefix_len < 0:
            raise ValueError("PSP-KV Ascend received a negative prefix length")

        if native_decode and num_queries == 1:
            num_blocks = (seq_len + block_size - 1) // block_size
            block_ids = block_tables[req, :num_blocks].to(torch.int32).contiguous()
            native_ops = load_native_fp8_ops()
            scores = native_ops.decode_scores_bf16(
                query[start:end].contiguous(),
                k_cache,
                precision_table,
                block_ids,
                seq_len,
                scale,
            )
            probabilities = torch.softmax(scores, dim=-1)
            result = native_ops.decode_weighted_value(
                probabilities, v_cache, precision_table, block_ids, seq_len
            )
            output_view[start:end].copy_(result.to(output.dtype))
            continue

        # Prefill uses current projected K/V directly, like the CUDA scratch
        # path. Decode reads the full context from the compressed cache.
        if prefix_len == 0 and num_queries > 1:
            all_keys = key[start:end].float()
            all_values = value[start:end].float()
        else:
            num_blocks = (seq_len + block_size - 1) // block_size
            block_ids = block_tables[req, :num_blocks].to(torch.int64)
            key_blocks = k_cache.index_select(0, block_ids)
            value_blocks = v_cache.index_select(0, block_ids)
            if native_fp8:
                block_formats = precision_table.index_select(0, block_ids)
                all_keys = load_native_fp8_ops().decode_fp8(key_blocks, block_formats)
                all_values = load_native_fp8_ops().decode_fp8(
                    value_blocks, block_formats
                )
            else:
                all_keys = decode_e4m3(key_blocks)
                all_values = decode_e4m3(value_blocks)
                if precision_table is not None:
                    use_e5m2 = (
                        precision_table.index_select(0, block_ids).view(-1, 1, 1, 1)
                        == 2
                    )
                    all_keys = torch.where(use_e5m2, decode_e5m2(key_blocks), all_keys)
                    all_values = torch.where(
                        use_e5m2, decode_e5m2(value_blocks), all_values
                    )
            all_keys = all_keys.reshape(-1, num_kv_heads, head_size)[:seq_len]
            all_values = all_values.reshape(-1, num_kv_heads, head_size)[:seq_len]
            if num_queries > 1:
                all_keys = torch.cat((all_keys[:prefix_len], key[start:end].float()))
                all_values = torch.cat(
                    (all_values[:prefix_len], value[start:end].float())
                )

        groups = num_heads // num_kv_heads
        if groups > 1:
            all_keys = all_keys.repeat_interleave(groups, dim=1)
            all_values = all_values.repeat_interleave(groups, dim=1)

        q = query[start:end].float().transpose(0, 1)
        k = all_keys.transpose(0, 1)
        v = all_values.transpose(0, 1)
        if use_sdpa:
            # SDPA can dispatch to torch_npu's fused attention implementation.
            # For a decode query (Q length 1), all cached keys are causal, so
            # no mask is needed. For chunked prefill after a cached prefix,
            # SDPA's built-in causal mask is top-left aligned and would hide
            # valid prefix keys; provide the explicit absolute-position mask.
            query_positions = prefix_len + torch.arange(
                num_queries, device=query.device
            )
            if num_queries > 1 and prefix_len > 0:
                key_positions = torch.arange(seq_len, device=query.device)
                attention_mask = key_positions[None, :] <= query_positions[:, None]
                is_causal = False
            else:
                attention_mask = None
                is_causal = num_queries > 1
            sdpa_dtype = (
                query.dtype
                if query.dtype in (torch.float16, torch.bfloat16)
                else q.dtype
            )
            result = (
                F.scaled_dot_product_attention(
                    q.to(sdpa_dtype).unsqueeze(0),
                    k.to(sdpa_dtype).unsqueeze(0),
                    v.to(sdpa_dtype).unsqueeze(0),
                    attn_mask=attention_mask,
                    dropout_p=0.0,
                    is_causal=is_causal,
                    scale=scale,
                )
                .squeeze(0)
                .transpose(0, 1)
            )
        else:
            scores = torch.bmm(q, k.transpose(1, 2)) * scale
            key_positions = torch.arange(seq_len, device=query.device)
            query_positions = prefix_len + torch.arange(
                num_queries, device=query.device
            )
            scores.masked_fill_(
                key_positions[None, :] > query_positions[:, None], float("-inf")
            )
            probabilities = torch.softmax(scores, dim=-1)
            result = torch.bmm(probabilities, v).transpose(0, 1)
        output_view[start:end].copy_(result.to(output.dtype))

    return output
