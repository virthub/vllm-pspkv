# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Correctness reference for variable-stride PSP-KV 16/8/4-bit KV storage.

Each physical block has an allocation-time format and a byte offset. K and V
use separate byte arenas with the same offsets. K is head-major, token-major;
V may be token-major or transposed [head, dimension, token]. The latter matches
the PSP-KV enhanced CUDA reader, while its current writer uses token-major V.

The variable-stride layout is not wired to vLLM-Ascend's fixed-size KV
allocator. The opt-in serving route uses uniform FP4 pages and this codec;
the mixed-format copy and relocation contract remains standalone.
"""

import torch
import torch.nn.functional as F

from vllm.v1.attention.backends.psp_kv_ascend_reference import (
    decode_e2m1,
    decode_e4m3,
    decode_e5m2,
    encode_e2m1,
    encode_e4m3,
    encode_e5m2,
)


class PackedKVCache:
    """A fixed set of physical blocks with per-block 16/8/4-bit strides."""

    def __init__(
        self,
        precision_codes: list[int],
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        device: torch.device,
        k_buffer: torch.Tensor | None = None,
        v_buffer: torch.Tensor | None = None,
        scalar_dtype: torch.dtype = torch.bfloat16,
        value_layout: str = "token_major",
        validate_layout: bool = True,
    ) -> None:
        if not precision_codes or any(
            code not in (0, 1, 2, 3) for code in precision_codes
        ):
            raise ValueError("Packed PSP-KV supports 16-bit, E4M3, E5M2 and FP4")
        if block_size <= 0 or num_kv_heads <= 0 or head_size <= 0:
            raise ValueError("Packed PSP-KV dimensions must be positive")
        if 3 in precision_codes and head_size % 2:
            raise ValueError("Packed FP4 requires an even head dimension")
        if scalar_dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("Lossless packed blocks require FP16 or BF16 dtype")
        if value_layout not in ("token_major", "transposed"):
            raise ValueError(
                "Packed PSP-KV value layout must be token_major or transposed"
            )

        self.block_size = block_size
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.scalar_dtype = scalar_dtype
        self.value_layout = value_layout
        self.validate_layout = validate_layout
        self.precision_table = torch.tensor(
            precision_codes, dtype=torch.uint8, device=device
        )
        self._codes = tuple(precision_codes)
        offsets: list[int] = []
        total_bytes = 0
        for code in precision_codes:
            offsets.append(total_bytes)
            row_bytes = self._row_bytes(code)
            total_bytes += block_size * num_kv_heads * row_bytes
        self.block_offsets = torch.tensor(offsets, dtype=torch.int64, device=device)
        self._offsets = tuple(offsets)
        self.total_bytes = total_bytes
        self.dense_fp8_bytes = (
            len(precision_codes) * block_size * num_kv_heads * head_size
        )

        def make_buffer(buffer: torch.Tensor | None) -> torch.Tensor:
            if buffer is None:
                return torch.zeros(total_bytes, dtype=torch.uint8, device=device)
            if (
                buffer.dtype != torch.uint8
                or buffer.ndim != 1
                or not buffer.is_contiguous()
                or buffer.device != device
                or buffer.numel() < total_bytes
            ):
                raise ValueError("Packed PSP-KV arena is too small or incompatible")
            return buffer[:total_bytes]

        self.k_buffer = make_buffer(k_buffer)
        self.v_buffer = make_buffer(v_buffer)

    def _row_bytes(self, code: int) -> int:
        if code == 0:
            return self.head_size * 2
        return self.head_size // 2 if code == 3 else self.head_size

    def _check_layout(self) -> None:
        if (
            tuple(self.precision_table.tolist()) != self._codes
            or tuple(self.block_offsets.tolist()) != self._offsets
        ):
            raise ValueError("Changing packed formats or offsets requires relocation")

    def _block_bytes(self, block_id: int) -> int:
        return (
            self.block_size * self.num_kv_heads * self._row_bytes(self._codes[block_id])
        )

    def copy_blocks(self, src_to_dsts: torch.Tensor | list[tuple[int, int]]) -> None:
        """Copy physical pages without changing their format or byte contents.

        All source pages are snapshotted before writing destinations, so a
        mapping that swaps two pages has the same semantics as vLLM's indexed
        block copy. A target with a different format needs ``repack`` instead.
        """
        self._check_layout()
        mapping = torch.as_tensor(src_to_dsts)
        if mapping.numel() == 0:
            return
        if (
            mapping.ndim != 2
            or mapping.shape[1] != 2
            or mapping.dtype not in (torch.int32, torch.int64)
        ):
            raise ValueError("Packed PSP-KV block copy requires integer [N, 2] pairs")
        pairs = mapping.tolist()
        destinations = [dst for _, dst in pairs]
        if len(set(destinations)) != len(destinations):
            raise ValueError("Packed PSP-KV block copy requires unique destinations")
        for src, dst in pairs:
            if not 0 <= src < len(self._codes) or not 0 <= dst < len(self._codes):
                raise ValueError("Packed PSP-KV block copy ID is outside the arena")
            if self._codes[src] != self._codes[dst]:
                raise ValueError("Packed PSP-KV block copy requires matching formats")

        snapshots = []
        for src, _ in pairs:
            offset = self._offsets[src]
            end = offset + self._block_bytes(src)
            snapshots.append(
                (self.k_buffer[offset:end].clone(), self.v_buffer[offset:end].clone())
            )
        for (_, dst), (key_bytes, value_bytes) in zip(pairs, snapshots):
            offset = self._offsets[dst]
            end = offset + self._block_bytes(dst)
            self.k_buffer[offset:end].copy_(key_bytes)
            self.v_buffer[offset:end].copy_(value_bytes)

    def repack(self, precision_codes: list[int]) -> "PackedKVCache":
        """Relocate all pages into a new compact arena with new formats.

        Unchanged-format pages are copied byte-for-byte. Changed pages are
        decoded and encoded again, which can lose precision and can reject a
        value outside FP4's unit-scale range. The old arena remains intact.
        """
        self._check_layout()
        if len(precision_codes) != len(self._codes):
            raise ValueError("Packed PSP-KV repack must retain physical block IDs")
        result = PackedKVCache(
            precision_codes,
            self.block_size,
            self.num_kv_heads,
            self.head_size,
            self.k_buffer.device,
            scalar_dtype=self.scalar_dtype,
            value_layout=self.value_layout,
        )
        for block_id, old_code in enumerate(self._codes):
            if precision_codes[block_id] == old_code:
                old_start = self._offsets[block_id]
                new_start = result._offsets[block_id]
                size = self._block_bytes(block_id)
                result.k_buffer[new_start : new_start + size].copy_(
                    self.k_buffer[old_start : old_start + size]
                )
                result.v_buffer[new_start : new_start + size].copy_(
                    self.v_buffer[old_start : old_start + size]
                )
            else:
                keys, values = self.gather([block_id])
                slots = torch.arange(
                    block_id * self.block_size,
                    (block_id + 1) * self.block_size,
                    device=self.k_buffer.device,
                )
                result.write(slots, keys[0], values[0])
        return result

    def write(
        self, slots: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> None:
        """Write projected K/V at vLLM physical slots using each block's format."""
        expected_shape = (slots.numel(), self.num_kv_heads, self.head_size)
        if (
            slots.ndim != 1
            or key.shape != expected_shape
            or value.shape != expected_shape
        ):
            raise ValueError("Packed PSP-KV slots and K/V shapes do not match")
        if (
            slots.device != self.k_buffer.device
            or key.device != self.k_buffer.device
            or value.device != self.k_buffer.device
        ):
            raise ValueError("Packed PSP-KV tensors must be on the same device")
        if self.validate_layout:
            self._check_layout()

        if self.value_layout == "token_major" and set(self._codes) == {3}:
            self._write_uniform_fp4(slots, key, value)
            return
        for token_index, slot in enumerate(slots.to(torch.int64).tolist()):
            if not 0 <= slot < self.precision_table.numel() * self.block_size:
                raise ValueError("Packed PSP-KV slot is outside the physical arena")
            block_id, token_offset = divmod(slot, self.block_size)
            code = self._codes[block_id]
            block_base = self._offsets[block_id]
            row_bytes = self._row_bytes(code)

            for head in range(self.num_kv_heads):
                head_start = block_base + head * self.block_size * row_bytes
                row_start = head_start + token_offset * row_bytes
                for source, arena, transposed in (
                    (key[token_index, head], self.k_buffer, False),
                    (
                        value[token_index, head],
                        self.v_buffer,
                        self.value_layout == "transposed",
                    ),
                ):
                    if code == 0:
                        payload = (
                            source.to(self.scalar_dtype).contiguous().view(torch.uint8)
                        )
                    elif code == 3:
                        if bool((~torch.isfinite(source) | (source.abs() > 3)).any()):
                            raise ValueError("Unit-scale FP4 values must be in [-3, 3]")
                        nibbles = encode_e2m1(source)
                        payload = nibbles[0::2] | (nibbles[1::2] << 4)
                    elif code == 2:
                        payload = encode_e5m2(source)
                    else:
                        payload = encode_e4m3(source)
                    if not transposed:
                        arena[row_start : row_start + row_bytes].copy_(payload)
                    elif code == 3:
                        # Repack one full head in this correctness reference.
                        # A future NPU scatter kernel must update nibbles in
                        # place without a read/modify/write race.
                        data = arena[
                            head_start : head_start + self.block_size * row_bytes
                        ]
                        unpacked = (
                            torch.stack((data & 0xF, data >> 4), dim=-1)
                            .reshape(self.head_size, self.block_size)
                            .clone()
                        )
                        unpacked[:, token_offset] = nibbles
                        flat = unpacked.reshape(-1)
                        data.copy_(flat[0::2] | (flat[1::2] << 4))
                    else:
                        bytes_per_element = 2 if code == 0 else 1
                        arena[
                            head_start : head_start + self.block_size * row_bytes
                        ].view(self.head_size, self.block_size, bytes_per_element)[
                            :, token_offset
                        ].copy_(payload.reshape(self.head_size, bytes_per_element))

    def _write_uniform_fp4(
        self, slots: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> None:
        """Scatter all token/head FP4 byte rows in two indexed writes."""
        slots = slots.to(torch.int64)
        if bool(((slots < 0) | (slots >= len(self._codes) * self.block_size)).any()):
            raise ValueError("Packed PSP-KV slot is outside the physical arena")
        if torch.unique(slots).numel() != slots.numel():
            raise ValueError("Uniform FP4 scatter requires unique slots")
        block_ids = slots // self.block_size
        token_offsets = slots % self.block_size
        heads = torch.arange(self.num_kv_heads, device=slots.device)
        row_ids = (
            block_ids[:, None] * (self.num_kv_heads * self.block_size)
            + heads[None, :] * self.block_size
            + token_offsets[:, None]
        ).reshape(-1)
        row_bytes = self.head_size // 2

        for source, arena in ((key, self.k_buffer), (value, self.v_buffer)):
            if bool((~torch.isfinite(source) | (source.abs() > 3)).any()):
                raise ValueError("Unit-scale FP4 values must be in [-3, 3]")
            codes = encode_e2m1(source)
            payload = (codes[..., 0::2] | (codes[..., 1::2] << 4)).reshape(
                -1, row_bytes
            )
            arena.view(-1, row_bytes).index_copy_(0, row_ids, payload)

    def gather(
        self, physical_block_ids: torch.Tensor | list[int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Read full physical blocks as fp32 [blocks, tokens, heads, dim]."""
        block_ids = torch.as_tensor(physical_block_ids).to(torch.int64).tolist()
        if self.validate_layout:
            self._check_layout()
        if not block_ids:
            shape = (0, self.block_size, self.num_kv_heads, self.head_size)
            empty = torch.empty(shape, dtype=torch.float32, device=self.k_buffer.device)
            return empty, empty.clone()
        if self.value_layout == "token_major" and set(self._codes) == {3}:
            return self._gather_uniform_fp4(block_ids)

        def read_arena(arena: torch.Tensor, transposed: bool) -> torch.Tensor:
            blocks = []
            for block_id in block_ids:
                if not 0 <= block_id < self.precision_table.numel():
                    raise ValueError("Packed PSP-KV block ID is outside the arena")
                code = self._codes[block_id]
                block_base = self._offsets[block_id]
                row_bytes = self._row_bytes(code)
                heads = []
                for head in range(self.num_kv_heads):
                    head_start = block_base + head * self.block_size * row_bytes
                    data = arena[head_start : head_start + self.block_size * row_bytes]
                    if code == 0:
                        # Clone to a zero-offset tensor before reinterpreting
                        # two-byte scalars; a preceding FP4 page may be odd-sized.
                        values = data.clone().view(self.scalar_dtype).float()
                    elif code == 3:
                        nibbles = torch.stack((data & 0xF, data >> 4), dim=-1)
                        values = decode_e2m1(nibbles.reshape(-1))
                    elif code == 2:
                        values = decode_e5m2(data)
                    else:
                        values = decode_e4m3(data)
                    if transposed:
                        values = values.reshape(self.head_size, self.block_size).T
                    else:
                        values = values.reshape(self.block_size, self.head_size)
                    heads.append(values)
                blocks.append(torch.stack(heads, dim=1))
            return torch.stack(blocks)

        return read_arena(self.k_buffer, False), read_arena(
            self.v_buffer, self.value_layout == "transposed"
        )

    def _gather_uniform_fp4(
        self, block_ids: list[int]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if any(not 0 <= block_id < len(self._codes) for block_id in block_ids):
            raise ValueError("Packed PSP-KV block ID is outside the arena")
        indices = torch.tensor(
            block_ids, dtype=torch.int64, device=self.k_buffer.device
        )
        shape = (
            len(self._codes),
            self.num_kv_heads,
            self.block_size,
            self.head_size // 2,
        )

        def read(arena: torch.Tensor) -> torch.Tensor:
            data = arena.view(shape).index_select(0, indices).permute(0, 2, 1, 3)
            nibbles = torch.stack((data & 0xF, data >> 4), dim=-1)
            return decode_e2m1(
                nibbles.reshape(
                    len(block_ids), self.block_size, self.num_kv_heads, self.head_size
                )
            )

        return read(self.k_buffer), read(self.v_buffer)


def reference_packed_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    cache: PackedKVCache,
    slot_mapping: torch.Tensor,
    block_tables: torch.Tensor,
    query_ends: list[int],
    seq_lens: list[int],
    output: torch.Tensor,
    attention_scale: float,
    block_scales: torch.Tensor,
    native_fp4: bool = False,
    use_sdpa: bool = False,
) -> torch.Tensor:
    """Eager causal GQA using uniform FP4 pages and selectable scale granularity.

    With ``native_fp4``, the NPU kernels write every FP4 page and consume the
    compressed cache directly for single-token decode. The completed-block
    scale policy and multi-token prefill remain in this reference wrapper.
    """
    if any(code != 3 for code in cache._codes):
        raise ValueError("Packed attention currently requires uniform FP4 pages")
    legacy_scales = block_scales.shape == (len(cache._codes), cache.num_kv_heads)
    grouped_scales = (
        block_scales.ndim == 4
        and block_scales.shape[:3] == (len(cache._codes), 2, cache.num_kv_heads)
        and block_scales.shape[3] > 0
        and cache.head_size % block_scales.shape[3] == 0
    )
    if (
        not (legacy_scales or grouped_scales)
        or block_scales.dtype != torch.float32
        or block_scales.device != cache.k_buffer.device
    ):
        raise ValueError(
            "FP4 scale table must be [blocks, heads] or [blocks, 2, heads, groups]"
        )

    def stream_scales(selected: torch.Tensor, stream: int) -> torch.Tensor:
        """Expand selected pages' scales to [pages/tokens, heads, dimension]."""
        if legacy_scales:
            return selected[:, :, None]
        return selected[:, stream].repeat_interleave(
            cache.head_size // block_scales.shape[3], dim=-1
        )

    actual = slot_mapping.numel()
    num_heads = query.shape[1]
    if (
        query.ndim != 3
        or query.shape[0] < actual
        or query.shape[2] != cache.head_size
        or key.ndim != 3
        or key.shape[0] < actual
        or key.shape[1:] != (cache.num_kv_heads, cache.head_size)
        or value.shape != key.shape
        or num_heads % cache.num_kv_heads
    ):
        raise ValueError("Packed attention Q/K/V shapes do not match the cache")
    output_view = output.view(-1, num_heads, cache.head_size)
    output_view.zero_()
    if actual == 0:
        return output
    if len(query_ends) != len(seq_lens) or query_ends[-1] != actual:
        raise ValueError("Packed attention received inconsistent request lengths")

    slots = slot_mapping.to(torch.int64)
    block_size = cache.block_size
    slot_blocks = slots // block_size
    native = None
    native_offsets = None
    if native_fp4:
        from vllm.v1.attention.backends.psp_kv_ascend_ops import PSPKVAscendPackedOps

        native = PSPKVAscendPackedOps()
        native_offsets = cache.block_offsets.to(torch.int32)

    def write_cache(
        physical_slots: torch.Tensor,
        real_keys: torch.Tensor,
        real_values: torch.Tensor,
    ) -> None:
        if native is None:
            scales = block_scales.index_select(
                0, (physical_slots // block_size).to(torch.int64)
            )
            cache.write(
                physical_slots,
                real_keys.float() / stream_scales(scales, 0),
                real_values.float() / stream_scales(scales, 1),
            )
            return
        assert native_offsets is not None
        projected = torch.stack(
            (real_keys.to(key.dtype), real_values.to(value.dtype)), dim=1
        ).contiguous()
        native.quantize_and_scatter_kv(
            projected,
            cache.k_buffer,
            cache.v_buffer,
            block_tables,
            physical_slots,
            cache.precision_table,
            native_offsets,
            block_scales,
            projected.shape[0],
            format_generation=3,
            preserve_scales=True,
            block_size=block_size,
        )

    for block_id in torch.unique(slot_blocks).tolist():
        incoming = slot_blocks == block_id
        first_offset = int((slots[incoming] % block_size).min().item())
        incoming_key = key[incoming].float()
        incoming_value = value[incoming].float()
        if legacy_scales:
            max_abs = torch.maximum(
                incoming_key.abs().amax(dim=(0, 2)),
                incoming_value.abs().amax(dim=(0, 2)),
            )
        else:
            group_size = cache.head_size // block_scales.shape[3]
            max_abs = (
                torch.stack((incoming_key, incoming_value), dim=1)
                .abs()
                .reshape(-1, 2, cache.num_kv_heads, block_scales.shape[3], group_size)
                .amax(dim=(0, 4))
            )
        if not bool(torch.isfinite(max_abs).all()):
            raise ValueError("FP4 scale selection requires finite K/V values")
        old_scales = block_scales[block_id].clone()
        # A new page should use the FP4 range even for small K/V values;
        # starting every page at scale 1 would round many values to zero.
        required = torch.where(
            max_abs > 0,
            torch.pow(2.0, torch.ceil(torch.log2(max_abs / 3.0))).clamp_min(2**-24),
            torch.ones_like(max_abs),
        )
        # log2/ceil may round down at an exact power-of-two boundary on NPU.
        # Compare in the original value domain before encoding E2M1.
        required = torch.where(max_abs > 3.0 * required, required * 2.0, required)
        new_scales = (
            required if first_offset == 0 else torch.maximum(old_scales, required)
        )
        if not torch.equal(new_scales, old_scales):
            if first_offset:
                old_keys, old_values = cache.gather([block_id])
                old_keys = old_keys[0, :first_offset] * stream_scales(
                    old_scales[None], 0
                )
                old_values = old_values[0, :first_offset] * stream_scales(
                    old_scales[None], 1
                )
            block_scales[block_id].copy_(new_scales)
            if first_offset:
                prefix_slots = torch.arange(
                    block_id * block_size,
                    block_id * block_size + first_offset,
                    device=cache.k_buffer.device,
                )
                write_cache(prefix_slots, old_keys, old_values)

    write_cache(slots, key[:actual], value[:actual])

    for req, seq_len in enumerate(seq_lens):
        start = 0 if req == 0 else query_ends[req - 1]
        end = query_ends[req]
        num_queries = end - start
        if num_queries == 0:
            continue
        prefix_len = seq_len - num_queries
        if prefix_len < 0:
            raise ValueError("Packed attention received a negative prefix length")
        if native is not None and num_queries == 1:
            assert native_offsets is not None
            context_lens = torch.tensor(
                [seq_len], dtype=torch.int32, device=query.device
            )
            native.psp_attention_scaled_basic(
                query[start:end].contiguous(),
                cache.k_buffer,
                cache.v_buffer,
                block_tables[req : req + 1].contiguous(),
                context_lens,
                cache.precision_table,
                native_offsets,
                block_scales,
                attention_scale,
                cache.num_kv_heads,
                cache.head_size,
                output_view[start:end],
                block_size,
            )
            continue
        if prefix_len == 0 and num_queries > 1:
            all_keys = key[start:end].float()
            all_values = value[start:end].float()
        else:
            num_blocks = (seq_len + block_size - 1) // block_size
            physical_ids = block_tables[req, :num_blocks].to(torch.int64)
            key_blocks, value_blocks = cache.gather(physical_ids)
            scales = block_scales.index_select(0, physical_ids)
            all_keys = (key_blocks * stream_scales(scales, 0)[:, None]).reshape(
                -1, cache.num_kv_heads, cache.head_size
            )[:seq_len]
            all_values = (value_blocks * stream_scales(scales, 1)[:, None]).reshape(
                -1, cache.num_kv_heads, cache.head_size
            )[:seq_len]
            if num_queries > 1:
                all_keys = torch.cat((all_keys[:prefix_len], key[start:end].float()))
                all_values = torch.cat(
                    (all_values[:prefix_len], value[start:end].float())
                )

        groups = num_heads // cache.num_kv_heads
        if groups > 1:
            all_keys = all_keys.repeat_interleave(groups, dim=1)
            all_values = all_values.repeat_interleave(groups, dim=1)
        if use_sdpa:
            query_tokens = query[start:end]
            key_positions = torch.arange(seq_len, device=query.device)
            query_positions = prefix_len + torch.arange(
                num_queries, device=query.device
            )
            causal_mask = (
                key_positions[None, :] <= query_positions[:, None]
                if prefix_len and num_queries > 1
                else None
            )
            result = F.scaled_dot_product_attention(
                query_tokens.transpose(0, 1).unsqueeze(0),
                all_keys.to(query.dtype).transpose(0, 1).unsqueeze(0),
                all_values.to(query.dtype).transpose(0, 1).unsqueeze(0),
                attn_mask=causal_mask,
                dropout_p=0.0,
                is_causal=prefix_len == 0 and num_queries > 1,
                scale=attention_scale,
            )[0].transpose(0, 1)
            output_view[start:end].copy_(result.to(output.dtype))
            continue
        scores = (
            torch.bmm(
                query[start:end].float().transpose(0, 1),
                all_keys.transpose(0, 1).transpose(1, 2),
            )
            * attention_scale
        )
        key_positions = torch.arange(seq_len, device=query.device)
        query_positions = prefix_len + torch.arange(num_queries, device=query.device)
        scores.masked_fill_(
            key_positions[None, :] > query_positions[:, None], float("-inf")
        )
        result = torch.bmm(
            torch.softmax(scores, dim=-1), all_values.transpose(0, 1)
        ).transpose(0, 1)
        output_view[start:end].copy_(result.to(output.dtype))
    return output
