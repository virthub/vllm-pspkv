# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Online variable-width compact KV arena for the Ascend PSP-KV path.

Each attention layer owns two flat uint8 byte arenas (K and V) allocated by
the worker with an average bytes-per-element budget (``PSPKV_AVG_BYTES``).
Physical blocks start a new lifetime on a slot-0 write: the backend reserves
the FP8 worst case, stores incoming K/V as E4M3, and keeps the original
model values of the currently open block in a small per-layer scratch. When
a block receives its 128th token, the completed-block precision policy runs
on those original values - exactly like the standalone reference - and the
block is kept as E4M3, rewritten as E5M2, or packed to half size as FP4. An
FP4 tail block reclaims its unused upper half.

The arena is correctness-first: holes left by freed or shrunk blocks are
reclaimed by a whole-arena compaction when the free pointer would overflow,
and an irreducible overflow fails closed with a clear error. The lossless
BF16 format is never selected online because it would exceed the FP8 budget.

This reference assumes one sequence at a time, no prefix caching, no KV
transfer, contiguous increasing slot writes within a block, and eager mode.
"""

import logging
import math
import os

import torch

from vllm.v1.attention.backends.psp_kv_ascend_policy import (
    select_completed_block_precision,
)
from vllm.v1.attention.backends.psp_kv_ascend_reference import (
    decode_e2m1,
    decode_e4m3,
    decode_e5m2,
    encode_e2m1,
    encode_e4m3,
    encode_e5m2,
)

logger = logging.getLogger(__name__)

# Block format codes shared with the standalone references:
# 1 = E4M3, 2 = E5M2, 3 = packed E2M1 FP4. -1 marks a free block. The
# lossless 16-bit code 0 is deliberately absent from the online arena
# because it would need two bytes per element and can exceed the budget.
FREE = -1


class VariableArenaKVCache:
    """Per-layer compact byte arena with per-block precision and offsets."""

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        device: torch.device,
        k_buffer: torch.Tensor,
        v_buffer: torch.Tensor,
        avg_bytes: float,
        v_avg_bytes: float | None = None,
        scalar_dtype: torch.dtype = torch.bfloat16,
        fp4_group_size: int = 0,
        max_relative_rmse: float = 0.05,
        layer_name: str = "",
        static_code: int | None = None,
        static_scales: torch.Tensor | None = None,
        calibrate: bool = False,
        static_v_code: int | None = None,
    ) -> None:
        if num_blocks <= 0 or block_size <= 0 or num_kv_heads <= 0 or head_size <= 0:
            raise ValueError("Variable arena dimensions must be positive")
        if head_size % 2:
            raise ValueError("Variable arena FP4 requires an even head dimension")
        if scalar_dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("Variable arena requires FP16 or BF16 model K/V")
        if not 0 < avg_bytes <= 1.0:
            raise ValueError("PSPKV_AVG_BYTES must be in (0, 1]")
        if v_avg_bytes is None:
            v_avg_bytes = avg_bytes
        if not 0 < v_avg_bytes <= 1.0:
            raise ValueError("PSPKV_AVG_V_BYTES must be in (0, 1]")
        if fp4_group_size < 0 or (fp4_group_size and head_size % fp4_group_size):
            raise ValueError("FP4 group size must be zero or divide the head size")
        if not math.isfinite(max_relative_rmse) or max_relative_rmse < 0:
            raise ValueError("Precision error limit must be finite and nonnegative")
        if static_code is not None and static_code not in (1, 2, 3):
            raise ValueError("Static precision must be E4M3, E5M2 or packed FP4")
        if static_v_code is not None and static_v_code not in (1, 2, 3):
            raise ValueError("Static V precision must be E4M3, E5M2 or packed FP4")
        if static_v_code is not None and static_code is None:
            raise ValueError("A static V precision needs a static K precision")
        if 3 in (static_code, static_v_code) and fp4_group_size:
            raise ValueError(
                "Static FP4 takes its scale granularity from the calibrated "
                "scale tensor, not PSPKV_ASCEND_FP4_GROUP_SIZE"
            )
        if 3 in (static_code, static_v_code) and static_scales is None:
            raise ValueError("Static FP4 requires calibrated scales")
        static_scale_groups = 1
        if static_scales is not None:
            if static_scales.dtype != torch.float32 or static_scales.device != device:
                raise ValueError(
                    "Static FP4 scales must be float32 on the arena device"
                )
            if static_scales.shape == (num_kv_heads,):
                # Shared K/V power-of-two scale per KV head.
                static_scales = static_scales[None, :, None].expand(
                    2, num_kv_heads, 1
                ).contiguous()
            elif (
                static_scales.ndim == 3
                and static_scales.shape[0] == 2
                and static_scales.shape[1] == num_kv_heads
                and static_scales.shape[2] > 0
                and head_size % static_scales.shape[2] == 0
            ):
                static_scales = static_scales.contiguous()
                static_scale_groups = static_scales.shape[2]
            else:
                raise ValueError(
                    "Static FP4 scales must be [KV heads] or "
                    "[2, KV heads, groups]"
                )

        self.num_blocks = num_blocks
        self.block_size = block_size
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.scalar_dtype = scalar_dtype
        self.fp4_group_size = fp4_group_size
        self.max_relative_rmse = max_relative_rmse
        self.layer_name = layer_name
        # A calibration-frozen format turns the completed-block policy into a
        # true allocation-time predictor: the format (and FP4 scale) is known
        # before any K/V exists, so reservations are exact and no re-encode
        # or scratch retention runs at completion time.
        self.static_code = static_code
        self.static_v_code = (
            static_v_code if static_v_code is not None else static_code
        )
        self.static_scales = static_scales
        self.static_scale_groups = static_scale_groups
        self.calibrate = calibrate

        # The FP8 worst case reservation per block for one arena.
        self.block_bytes_fp8 = block_size * num_kv_heads * head_size
        self.row_bytes_fp8 = head_size

        def check_buffer(
            buffer: torch.Tensor, name: str, avg: float
        ) -> torch.Tensor:
            expected = num_blocks * self.block_bytes_fp8 * avg
            if (
                buffer.dtype != torch.uint8
                or buffer.ndim != 1
                or not buffer.is_contiguous()
                or buffer.device != device
                or buffer.numel() != expected
            ):
                raise ValueError(
                    f"Variable arena {name} buffer must be a flat uint8 tensor "
                    f"of exactly {expected} bytes on {device}"
                )
            return buffer

        self.avg_bytes = avg_bytes
        self.v_avg_bytes = v_avg_bytes
        self.k_buffer = check_buffer(k_buffer, "K", avg_bytes)
        self.v_buffer = check_buffer(v_buffer, "V", v_avg_bytes)
        self.arena_size = self.k_buffer.numel()
        self.v_arena_size = self.v_buffer.numel()

        # CPU-side per-block metadata; the scale tensor lives on device.
        self._codes = [FREE] * num_blocks
        self._v_codes = [FREE] * num_blocks
        self._offsets = [-1] * num_blocks
        self._v_offsets = [-1] * num_blocks
        self._sizes = [0] * num_blocks
        self._v_sizes = [0] * num_blocks
        self._written = [0] * num_blocks
        self._finalized = [False] * num_blocks
        # Physical blocks with a current lifetime; iterating this set keeps
        # liveness bookkeeping O(live blocks) instead of O(num_blocks).
        self._allocated: set[int] = set()
        self._pending_block_starts: set[int] = set()
        self._live_block_ids: set[int] = set()
        self.free_ptr = 0
        self.v_free_ptr = 0
        # Device mirrors of the per-block metadata for the native Ascend C
        # kernels, refreshed lazily on mutation. Free blocks keep code 255
        # and offset 0; native readers only touch live blocks.
        self.device_codes = torch.full(
            (num_blocks,), 255, dtype=torch.uint8, device=device
        )
        self.device_offsets = torch.zeros(num_blocks, dtype=torch.int32, device=device)
        self.device_v_codes = torch.full(
            (num_blocks,), 255, dtype=torch.uint8, device=device
        )
        self.device_v_offsets = torch.zeros(
            num_blocks, dtype=torch.int32, device=device
        )
        self._device_metadata_dirty = True
        if 3 in (static_code, self.static_v_code):
            # Any static FP4 stream keeps K/V (and optional dimension groups)
            # scales in the [blocks, 2, heads, groups] layout.
            scale_shape = (num_blocks, 2, num_kv_heads, static_scale_groups)
        elif fp4_group_size:
            scale_shape = (
                num_blocks, 2, num_kv_heads, head_size // fp4_group_size
            )
        else:
            scale_shape = (num_blocks, num_kv_heads)
        self.scales = torch.ones(scale_shape, dtype=torch.float32, device=device)

        # Original model values of the block currently receiving writes, so
        # the completed-block policy sees unquantized K/V like the standalone
        # reference. Only one block is open at a time because writes are
        # contiguous and increasing within a single sequence.
        self.scratch_keys = torch.zeros(
            (block_size, num_kv_heads, head_size),
            dtype=scalar_dtype,
            device=device,
        )
        self.scratch_values = torch.zeros_like(self.scratch_keys)
        self.scratch_block_id = -1

    # ------------------------------------------------------------------
    # Lifetime management
    # ------------------------------------------------------------------

    def _mark_metadata_dirty(self) -> None:
        self._device_metadata_dirty = True

    def sync_device_metadata(self) -> None:
        """Refresh the uint8/int32 device mirrors of codes and offsets.

        Called by the native Ascend C path before launching kernels; the
        copies only happen when per-block metadata actually changed.
        """
        if not self._device_metadata_dirty:
            return
        self.device_codes.copy_(
            torch.tensor(
                [code if code != FREE else 255 for code in self._codes],
                dtype=torch.uint8,
                device=self.device_codes.device,
            )
        )
        self.device_offsets.copy_(
            torch.tensor(
                [offset if offset >= 0 else 0 for offset in self._offsets],
                dtype=torch.int32,
                device=self.device_offsets.device,
            )
        )
        self.device_v_codes.copy_(
            torch.tensor(
                [code if code != FREE else 255 for code in self._v_codes],
                dtype=torch.uint8,
                device=self.device_v_codes.device,
            )
        )
        self.device_v_offsets.copy_(
            torch.tensor(
                [offset if offset >= 0 else 0 for offset in self._v_offsets],
                dtype=torch.int32,
                device=self.device_v_offsets.device,
            )
        )
        self._device_metadata_dirty = False

    def _free_dead(self, live_block_ids: set[int]) -> None:
        """Release blocks that no request references and no write touches."""
        dead = self._allocated - live_block_ids
        for block_id in dead:
            self._codes[block_id] = FREE
            self._v_codes[block_id] = FREE
            self._offsets[block_id] = -1
            self._v_offsets[block_id] = -1
            self._sizes[block_id] = 0
            self._v_sizes[block_id] = 0
            self._written[block_id] = 0
            self._finalized[block_id] = False
            self._reset_scale_row(block_id)
            self._allocated.discard(block_id)
        if dead:
            self._mark_metadata_dirty()
        if self.scratch_block_id in dead:
            self.scratch_block_id = -1
        if not self._allocated:
            # The whole arena is empty again; restart it from offset zero so
            # the next sequence allocates without residual holes.
            self.free_ptr = 0
            self.v_free_ptr = 0

    def _reset_scale_row(self, block_id: int) -> None:
        self.scales[block_id] = torch.ones_like(self.scales[block_id])

    def _stream_codes(self) -> tuple[int | None, int | None]:
        if self.static_code is None:
            return None, None
        return self.static_code, self.static_v_code

    def _block_bytes_for(self, code: int) -> int:
        return self.block_bytes_fp8 // 2 if code == 3 else self.block_bytes_fp8

    def _allocate(self, block_id: int, live_block_ids: set[int]) -> None:
        """Reserve the block's bytes for a new lifetime.

        Dynamic mode reserves the FP8 worst case because the completed-block
        policy may keep any format. Static (calibrated) mode already knows
        the format at allocation time, so FP4 blocks reserve half the bytes.
        """
        k_code, v_code = self._stream_codes()
        k_reservation = (
            self._block_bytes_for(k_code)
            if k_code is not None
            else self.block_bytes_fp8
        )
        v_reservation = (
            self._block_bytes_for(v_code)
            if v_code is not None
            else self.block_bytes_fp8
        )
        needs_compaction = (
            self.free_ptr + k_reservation > self.arena_size
            or self.v_free_ptr + v_reservation > self.v_arena_size
        )
        if needs_compaction:
            self.compact(live_block_ids)
        for name, reservation, free, capacity, avg in (
            ("K", k_reservation, self.free_ptr, self.arena_size, self.avg_bytes),
            (
                "V",
                v_reservation,
                self.v_free_ptr,
                self.v_arena_size,
                self.v_avg_bytes,
            ),
        ):
            if free + reservation > capacity:
                raise RuntimeError(
                    "PSP-KV variable arena overflow: the current precision "
                    f"mix needs {free + reservation} {name} bytes but the "
                    f"{name} arena only holds {capacity} ({self.num_blocks} "
                    f"physical blocks at {capacity / self.num_blocks:.2f} "
                    f"bytes per block per {name}). Raise PSPKV_AVG_BYTES or "
                    "PSPKV_AVG_K_BYTES/PSPKV_AVG_V_BYTES, raise "
                    "PSPKV_ASCEND_MAX_RELATIVE_RMSE, or serve fewer requests "
                    "before recycling the arena."
                )
        if k_code is not None:
            self._codes[block_id] = k_code
            self._v_codes[block_id] = v_code
            if 3 in (k_code, v_code):
                self.scales[block_id] = self.static_scales
            else:
                self._reset_scale_row(block_id)
        else:
            self._codes[block_id] = 1
            self._v_codes[block_id] = 1
            self._reset_scale_row(block_id)
        self._offsets[block_id] = self.free_ptr
        self._sizes[block_id] = k_reservation
        self._v_offsets[block_id] = self.v_free_ptr
        self._v_sizes[block_id] = v_reservation
        self._written[block_id] = 0
        self._finalized[block_id] = False
        self._allocated.add(block_id)
        self._mark_metadata_dirty()
        self.free_ptr += k_reservation
        self.v_free_ptr += v_reservation

    def begin_writes(self, slots: torch.Tensor, live_block_ids: set[int]) -> None:
        """Free dead blocks and record fresh lifetimes for slot-0 writes.

        Reservations happen as each block is written. This lets a completed
        FP4 block shrink before the next block needs space in the arena.
        """
        self._free_dead(live_block_ids)
        self._live_block_ids = live_block_ids.copy()
        slot_list = slots.to(torch.int64).tolist()
        self._pending_block_starts = {
            slot // self.block_size for slot in slot_list if slot % self.block_size == 0
        }
        if self.scratch_block_id in self._pending_block_starts:
            # The scratch block is being rewritten from slot zero; its
            # retained values belong to the abandoned lifetime.
            self.scratch_block_id = -1

    # ------------------------------------------------------------------
    # Writes, completion policy, compaction
    # ------------------------------------------------------------------

    def write_and_finalize(
        self,
        slots: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        native: bool = False,
    ) -> None:
        """Encode incoming K/V as E4M3, retain originals, finalize full blocks.

        Blocks are processed in incoming token order. A block that
        receives its last token is finalized immediately from the scratch
        copy of its original values, before any later block is touched.
        With ``native``, the byte scatter runs through the Ascend C
        ``packed_scatter`` kernel; scratch bookkeeping and the completed-block
        policy remain in this reference wrapper.
        """
        slots = slots.to(torch.int64)
        if slots.numel() == 0:
            return
        if torch.unique(slots).numel() != slots.numel():
            raise ValueError("Variable arena scatter requires unique slots")
        if (
            slots.device != self.k_buffer.device
            or key.device != self.k_buffer.device
            or value.device != self.k_buffer.device
        ):
            raise ValueError("Variable arena tensors must be on the same device")
        expected_shape = (slots.numel(), self.num_kv_heads, self.head_size)
        if key.shape != expected_shape or value.shape != expected_shape:
            raise ValueError("Variable arena slots and K/V shapes do not match")
        if bool(((slots < 0) | (slots >= self.num_blocks * self.block_size)).any()):
            raise ValueError("Variable arena slot is outside the physical arena")

        slot_list = slots.tolist()
        groups: dict[int, list[int]] = {}
        for token_index, slot in enumerate(slot_list):
            groups.setdefault(slot // self.block_size, []).append(token_index)

        # Dict insertion order follows the incoming token order, which is
        # logical sequence order even when physical block IDs are shuffled.
        # Finish each block before the next one reuses the single scratch.
        for block_id in groups:
            if block_id in self._pending_block_starts:
                self._allocate(block_id, self._live_block_ids)
                self._pending_block_starts.remove(block_id)
            token_indices = groups[block_id]
            token_offsets = [
                slot_list[index] % self.block_size for index in token_indices
            ]
            self._switch_scratch_to(block_id)
            self._track_open_tokens(block_id, token_indices, token_offsets, key, value)
            if native and self._codes[block_id] == self._v_codes[block_id]:
                indices = torch.tensor(token_indices, device=slots.device)
                self._scatter_open_tokens_native(
                    slots[indices], key[indices], value[indices]
                )
            else:
                self._scatter_open_tokens_e4m3(
                    block_id, token_indices, token_offsets, key, value
                )
            if (
                self._written[block_id] == self.block_size
                and not self._finalized[block_id]
            ):
                self._finalize_block(block_id)

    def _switch_scratch_to(self, block_id: int) -> None:
        if self.static_code is not None:
            # Static precision never consults the original values.
            return
        if self.scratch_block_id == block_id:
            return
        if self.scratch_block_id != -1:
            if self._written[self.scratch_block_id] != self.block_size:
                raise ValueError(
                    "Variable arena cannot switch away from an incomplete "
                    "block while retaining only one scratch block"
                )
            if not self._finalized[self.scratch_block_id]:
                self._finalize_block(self.scratch_block_id)
        self.scratch_block_id = block_id

    def _track_open_tokens(
        self,
        block_id: int,
        token_indices: list[int],
        token_offsets: list[int],
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        """Validate the open block, count tokens, and retain original values."""
        expected_k, expected_v = self._stream_codes()
        expected_k = expected_k if expected_k is not None else 1
        expected_v = expected_v if expected_v is not None else 1
        if (
            self._codes[block_id] != expected_k
            or self._v_codes[block_id] != expected_v
            or self._finalized[block_id]
        ):
            raise ValueError(
                "Variable arena writes require an open block of the "
                "allocated format; a write to slot 0 of this block should "
                "have started a lifetime"
            )
        first_unwritten = self._written[block_id]
        if sorted(token_offsets) != list(
            range(first_unwritten, first_unwritten + len(token_offsets))
        ):
            raise ValueError(
                "Variable arena writes must cover consecutive, unwritten "
                "slots within each block"
            )
        self._written[block_id] += len(token_offsets)
        if self.static_code is not None:
            # The format is fixed at allocation, so the completed-block
            # policy never needs the original values.
            return
        offset_tensor = torch.tensor(
            token_offsets, dtype=torch.int64, device=key.device
        )
        index_tensor = torch.tensor(token_indices, dtype=torch.int64, device=key.device)
        self.scratch_keys.index_copy_(0, offset_tensor, key[index_tensor])
        self.scratch_values.index_copy_(0, offset_tensor, value[index_tensor])

    def _scatter_open_tokens_e4m3(
        self,
        block_id: int,
        token_indices: list[int],
        token_offsets: list[int],
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> None:
        """Encode the open block's tokens in each stream's format."""
        index_tensor = torch.tensor(token_indices, dtype=torch.int64, device=key.device)
        for stream, (source, arena, codes, offsets) in enumerate(
            (
                (key, self.k_buffer, self._codes, self._offsets),
                (value, self.v_buffer, self._v_codes, self._v_offsets),
            )
        ):
            code = codes[block_id]
            if code == 3:
                self._scatter_stream_fp4(
                    block_id, stream, token_indices, token_offsets, source,
                    arena, offsets,
                )
                continue
            encoder = encode_e5m2 if code == 2 else encode_e4m3
            row_bytes = self.head_size
            rows = []
            base_row = offsets[block_id] // row_bytes
            for token_offset in token_offsets:
                for head in range(self.num_kv_heads):
                    rows.append(base_row + head * self.block_size + token_offset)
            row_ids = torch.tensor(rows, dtype=torch.int64, device=key.device)
            encoded = encoder(source[index_tensor]).reshape(-1, self.head_size)
            arena.view(-1, row_bytes).index_copy_(0, row_ids, encoded)

    def _fp4_factor(
        self, block_id: int, stream: int, leading: bool
    ) -> torch.Tensor:
        """Per-element FP4 scale factor for one K/V stream.

        Static scales are [2, heads, groups]; dynamic scales are either
        per-head [blocks, heads] or per-head grouped
        [blocks, 2, heads, groups]. ``leading`` selects [1, heads, dim] for
        ``[tokens, heads, dim]`` tensors or [heads, 1, dim] for
        ``[heads, tokens, dim]`` tensors.
        """
        scale_row = self.scales[block_id]
        # Legacy per-head dynamic scales have no stream dimension.
        row = scale_row if scale_row.ndim == 1 else scale_row[stream]
        if row.ndim == 2:
            group = self.head_size // row.shape[-1]
            expanded = row.repeat_interleave(group, dim=-1)
            return expanded[None] if leading else expanded[:, None]
        return row[None, :, None] if leading else row[:, None, None]

    def _clamp_static_fp4(
        self, values: torch.Tensor, stream: int
    ) -> torch.Tensor:
        """Clamp values to the calibrated FP4 range so unseen outliers do not
        encode as the E2M1 infinity code."""
        codes = self._stream_codes()
        if codes[stream] != 3:
            return values
        row = self.static_scales[stream] * 3.0
        if row.ndim == 2:
            group = self.head_size // row.shape[-1]
            limit = row.repeat_interleave(group, dim=-1)
        else:
            limit = row.expand(self.num_kv_heads, self.head_size)
        limit = limit[None]
        return values.float().clamp(-limit, limit).to(values.dtype)

    def _scatter_stream_fp4(
        self,
        block_id: int,
        stream: int,
        token_indices: list[int],
        token_offsets: list[int],
        source: torch.Tensor,
        arena: torch.Tensor,
        offsets: list[int],
    ) -> None:
        """Pack one stream's FP4 rows with its calibrated scale table."""
        row_bytes = self.head_size // 2
        rows = []
        base_row = offsets[block_id] // row_bytes
        for token_offset in token_offsets:
            for head in range(self.num_kv_heads):
                rows.append(base_row + head * self.block_size + token_offset)
        row_ids = torch.tensor(rows, dtype=torch.int64, device=source.device)
        index_tensor = torch.tensor(
            token_indices, dtype=torch.int64, device=source.device
        )
        clamped = self._clamp_static_fp4(source[index_tensor], stream)
        factors = self._fp4_factor(block_id, stream, leading=True)
        positive = clamped.float() / factors
        nibbles = encode_e2m1(positive)
        payload = (nibbles[..., 0::2] | (nibbles[..., 1::2] << 4)).reshape(
            -1, row_bytes
        )
        arena.view(-1, row_bytes).index_copy_(0, row_ids, payload)

    def _scatter_open_tokens_native(
        self, slots: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> None:
        from vllm.v1.attention.backends.psp_kv_ascend_native import (
            load_native_fp8_ops,
        )

        self.sync_device_metadata()
        stacked = torch.stack(
            (
                self._clamp_static_fp4(key, 0),
                self._clamp_static_fp4(value, 1),
            ),
            dim=1,
        ).contiguous()
        if self._codes[slots[0].item() // self.block_size] != self._v_codes[
            slots[0].item() // self.block_size
        ]:
            raise ValueError(
                "Native scatter needs matching K/V formats; use the Python "
                "scatter for mixed-precision blocks"
            )
        load_native_fp8_ops().packed_scatter(
            stacked,
            self.k_buffer,
            self.v_buffer,
            slots,
            self.device_codes,
            self.device_offsets,
            self.scales,
            slots.numel(),
            self.block_size,
            3,
            True,
        )

    def _finalize_block(self, block_id: int) -> None:
        if self.static_code is not None:
            # The format was fixed at allocation; nothing to choose.
            self._finalized[block_id] = True
            return
        if self.scratch_block_id != block_id:
            raise ValueError(
                "Variable arena can only finalize the block whose original "
                "values are still in the scratch"
            )
        block_key = self.scratch_keys
        block_value = self.scratch_values
        choice = select_completed_block_precision(
            block_key,
            block_value,
            self.max_relative_rmse,
            self.fp4_group_size or None,
            allow_lossless=False,
        )
        if self.calibrate:
            groups = int(os.environ.get("PSPKV_ASCEND_CALIBRATE_GROUPS", "1"))
            if groups < 1 or self.head_size % groups:
                raise ValueError(
                    "PSPKV_ASCEND_CALIBRATE_GROUPS must divide the head size"
                )

            def group_maxima(tensor: torch.Tensor) -> list[str]:
                magnitude = tensor.float().abs()
                if groups == 1:
                    maxima = magnitude.amax(dim=(0, 2))
                else:
                    group_size = self.head_size // groups
                    maxima = magnitude.reshape(
                        self.block_size,
                        self.num_kv_heads,
                        groups,
                        group_size,
                    ).amax(dim=(0, 3))
                return [f"{value:.6g}" for value in maxima.flatten().tolist()]

            logger.info(
                "PSP-KV calibration layer=%s code=%d groups=%d k_max=%s v_max=%s",
                self.layer_name,
                choice.code,
                groups,
                ",".join(group_maxima(block_key)),
                ",".join(group_maxima(block_value)),
            )
        self._codes[block_id] = choice.code
        self._v_codes[block_id] = choice.code
        self._mark_metadata_dirty()
        if choice.code == 3:
            self._pack_fp4_block(block_id, block_key, block_value, choice.scales)
        elif choice.code == 2:
            self._rewrite_block(block_id, encode_e5m2, block_key, block_value)
        else:
            self.scales[block_id] = torch.ones_like(self.scales[block_id])
        self._finalized[block_id] = True

    def _rewrite_block(self, block_id: int, encoder, block_key, block_value) -> None:
        for arena, source, offset, size in (
            (
                self.k_buffer,
                block_key,
                self._offsets[block_id],
                self._sizes[block_id],
            ),
            (
                self.v_buffer,
                block_value,
                self._v_offsets[block_id],
                self._v_sizes[block_id],
            ),
        ):
            payload = encoder(source.float()).permute(1, 0, 2).reshape(-1)
            arena[offset : offset + size].copy_(payload)

    def _pack_fp4_block(
        self,
        block_id: int,
        block_key: torch.Tensor,
        block_value: torch.Tensor,
        scales: torch.Tensor,
    ) -> None:
        k_offset = self._offsets[block_id]
        v_offset = self._v_offsets[block_id]
        if self.fp4_group_size:
            key_factor = scales[0].repeat_interleave(self.fp4_group_size, dim=-1)[None]
            value_factor = scales[1].repeat_interleave(self.fp4_group_size, dim=-1)[
                None
            ]
        else:
            key_factor = value_factor = scales[None, :, None]
        self.scales[block_id] = scales
        packed_size = self.block_bytes_fp8 // 2
        for arena, source, factor, offset in (
            (self.k_buffer, block_key.float(), key_factor, k_offset),
            (self.v_buffer, block_value.float(), value_factor, v_offset),
        ):
            nibbles = encode_e2m1(source / factor)
            payload = nibbles[..., 0::2] | (nibbles[..., 1::2] << 4)
            arena[offset : offset + packed_size].copy_(
                payload.permute(1, 0, 2).reshape(-1)
            )
        self._sizes[block_id] = packed_size
        self._v_sizes[block_id] = packed_size
        if k_offset + self.block_bytes_fp8 == self.free_ptr:
            # The block was the K arena tail; reclaim its unused upper half.
            self.free_ptr = k_offset + packed_size
        if v_offset + self.block_bytes_fp8 == self.v_free_ptr:
            self.v_free_ptr = v_offset + packed_size

    def compact(self, live_block_ids: set[int]) -> None:
        """Rebuild the arena contiguously from the live blocks only.

        Formats never change during compaction, so every live block is a
        byte-for-byte move. Visit blocks in current byte-offset order, not
        physical ID order. Each move goes left, so it cannot overwrite a
        later source; only the current source needs a snapshot.
        """
        live = sorted(self._allocated & live_block_ids, key=self._offsets.__getitem__)
        moved = False
        for arena, codes, offsets, sizes, attr in (
            (self.k_buffer, self._codes, self._offsets, self._sizes, "free_ptr"),
            (
                self.v_buffer,
                self._v_codes,
                self._v_offsets,
                self._v_sizes,
                "v_free_ptr",
            ),
        ):
            running = 0
            # Blocks were already ordered by the K byte offsets; the V stream
            # keeps the same block order because lifetimes advance together.
            for block_id in live:
                size = sizes[block_id]
                old_offset = offsets[block_id]
                if old_offset == running:
                    running += size
                    continue
                moved = True
                region = arena[old_offset : old_offset + size].clone()
                arena[running : running + size].copy_(region)
                offsets[block_id] = running
                running += size
            setattr(self, attr, running)
        if moved:
            self._mark_metadata_dirty()

    # ------------------------------------------------------------------
    # Reads and copies
    # ------------------------------------------------------------------

    def gather_blocks(
        self, physical_block_ids: list[int] | torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Decode full blocks as fp32 [blocks, tokens, heads, dim].

        FP4 blocks are multiplied by their per-block scales so callers see
        model-scale values. The final partial block of a request must still
        be truncated to the request's context length.
        """
        block_ids = torch.as_tensor(physical_block_ids).to(torch.int64).tolist()

        def read_arena(
            arena: torch.Tensor,
            stream: int,
            code_table: list[int],
            offset_table: list[int],
            size_table: list[int],
        ) -> torch.Tensor:
            blocks = []
            for block_id in block_ids:
                if not 0 <= block_id < self.num_blocks:
                    raise ValueError("Variable arena block ID is outside the arena")
                code = code_table[block_id]
                if code == FREE:
                    raise ValueError("Variable arena read of a block with no lifetime")
                offset = offset_table[block_id]
                if code == 3:
                    region = arena[offset : offset + size_table[block_id]].view(
                        self.num_kv_heads, self.block_size, self.head_size // 2
                    )
                    nibbles = torch.stack((region & 0xF, region >> 4), dim=-1)
                    values = decode_e2m1(
                        nibbles.reshape(
                            self.num_kv_heads,
                            self.block_size,
                            self.head_size,
                        )
                    )
                    values = values * self._fp4_factor(
                        block_id, stream, leading=False
                    )
                    values = values.permute(1, 0, 2)
                else:
                    region = arena[offset : offset + size_table[block_id]].view(
                        self.num_kv_heads, self.block_size, self.head_size
                    )
                    decoder = decode_e5m2 if code == 2 else decode_e4m3
                    values = decoder(region).permute(1, 0, 2)
                blocks.append(values)
            if not blocks:
                shape = (
                    0,
                    self.block_size,
                    self.num_kv_heads,
                    self.head_size,
                )
                return torch.empty(shape, dtype=torch.float32, device=arena.device)
            return torch.stack(blocks)

        return (
            read_arena(
                self.k_buffer, 0, self._codes, self._offsets, self._sizes
            ),
            read_arena(
                self.v_buffer,
                1,
                self._v_codes,
                self._v_offsets,
                self._v_sizes,
            ),
        )

    def copy_blocks(self, src_to_dsts: torch.Tensor | list[tuple[int, int]]) -> None:
        """Copy physical pages without changing their format or bytes."""
        mapping = torch.as_tensor(src_to_dsts)
        if mapping.numel() == 0:
            return
        if mapping.ndim != 2 or mapping.shape[1] != 2:
            raise ValueError("Variable arena block copy requires [N, 2] pairs")
        pairs = mapping.tolist()
        destinations = [dst for _, dst in pairs]
        if len(set(destinations)) != len(destinations):
            raise ValueError("Variable arena block copy requires unique destinations")
        for src, dst in pairs:
            if not 0 <= src < self.num_blocks or not 0 <= dst < self.num_blocks:
                raise ValueError("Variable arena block copy ID is outside the arena")
            if self._codes[src] == FREE or self._codes[dst] == FREE:
                raise ValueError("Variable arena block copy requires allocated blocks")
            if (
                self._codes[src] != self._codes[dst]
                or self._v_codes[src] != self._v_codes[dst]
            ):
                raise ValueError("Variable arena block copy requires matching formats")
            if self._sizes[src] != self._sizes[dst] or self._v_sizes[src] != self._v_sizes[dst]:
                raise ValueError("Variable arena block copy requires matching sizes")
        snapshots = []
        for src, _ in pairs:
            k_size = self._sizes[src]
            v_size = self._v_sizes[src]
            k_offset = self._offsets[src]
            v_offset = self._v_offsets[src]
            snapshots.append(
                (
                    self.k_buffer[k_offset : k_offset + k_size].clone(),
                    self.v_buffer[v_offset : v_offset + v_size].clone(),
                )
            )
        for (_, dst), (k_bytes, v_bytes) in zip(pairs, snapshots):
            k_size = self._sizes[dst]
            v_size = self._v_sizes[dst]
            k_offset = self._offsets[dst]
            v_offset = self._v_offsets[dst]
            self.k_buffer[k_offset : k_offset + k_size].copy_(k_bytes)
            self.v_buffer[v_offset : v_offset + v_size].copy_(v_bytes)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def format_counts(self) -> dict[int, int]:
        """Count allocated blocks per format code for observability."""
        counts: dict[int, int] = {}
        for code in self._codes:
            if code != FREE:
                counts[code] = counts.get(code, 0) + 1
        return counts


def live_block_ids(
    block_tables: torch.Tensor,
    seq_lens: list[int],
    slots: torch.Tensor,
    block_size: int,
) -> set[int]:
    """Physical blocks referenced by the current requests or writes."""
    live: set[int] = set()
    for req, seq_len in enumerate(seq_lens):
        num_blocks = (seq_len + block_size - 1) // block_size
        table = block_tables[req, :num_blocks].to(torch.int64)
        live.update(int(block) for block in table.tolist() if block >= 0)
    live.update((slots.to(torch.int64) // block_size).tolist())
    return live


def reference_variable_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    arena: VariableArenaKVCache,
    slot_mapping: torch.Tensor,
    block_tables: torch.Tensor,
    query_ends: list[int],
    seq_lens: list[int],
    output: torch.Tensor,
    scale: float,
    native: bool = False,
    native_fused: bool = False,
) -> torch.Tensor:
    """Eager causal GQA over the variable-width compact arena.

    Writes land in open E4M3 reservations; blocks that reach their last
    slot are finalized by the completed-block policy before attention reads
    them. Multi-token prefill uses the current projected K/V directly like
    the fixed-stride reference; decode reads the compressed history. With
    ``native``, single-token decode attention runs through the Ascend C
    packed kernels over the per-block offsets, formats and scales, avoiding
    the whole-block FP32 decode; multi-token prefill keeps this reference.
    """
    num_heads = query.shape[1]
    head_size = arena.head_size
    num_kv_heads = arena.num_kv_heads
    block_size = arena.block_size
    if native_fused and not native:
        raise ValueError("Fused variable attention requires native cache writes")
    if num_heads % num_kv_heads:
        raise ValueError("PSP-KV Ascend query heads must be divisible by KV heads")
    if query.ndim != 3 or query.shape[2] != head_size:
        raise ValueError("Variable attention query shape does not match the arena")
    output_view = output.view(-1, num_heads, head_size)
    output_view.zero_()
    actual = slot_mapping.numel()
    if actual == 0:
        return output
    if len(query_ends) != len(seq_lens) or query_ends[-1] != actual:
        raise ValueError("Variable attention received inconsistent request lengths")

    slots = slot_mapping.to(torch.int64)
    live = live_block_ids(block_tables, seq_lens, slots, block_size)
    arena.begin_writes(slots, live)
    arena.write_and_finalize(slots, key[:actual], value[:actual], native=native)

    for req, seq_len in enumerate(seq_lens):
        start = 0 if req == 0 else query_ends[req - 1]
        end = query_ends[req]
        num_queries = end - start
        if num_queries == 0:
            continue
        prefix_len = seq_len - num_queries
        if prefix_len < 0:
            raise ValueError("Variable attention received a negative prefix length")

        if native and num_queries == 1:
            from vllm.v1.attention.backends.psp_kv_ascend_ops import (
                PSPKVAscendPackedOps,
            )

            arena.sync_device_metadata()
            if block_tables.dtype != torch.int32:
                tables = block_tables.to(torch.int32)
            else:
                tables = block_tables
            context_lens = torch.tensor(
                seq_lens, dtype=torch.int32, device=query.device
            )
            operator = (
                PSPKVAscendPackedOps.psp_attention_scaled_basic_fused
                if native_fused
                else PSPKVAscendPackedOps.psp_attention_scaled_basic_vec
            )
            operator(
                PSPKVAscendPackedOps(),
                query[start:end].contiguous(),
                arena.k_buffer,
                arena.v_buffer,
                tables.contiguous(),
                context_lens,
                arena.device_codes,
                arena.device_offsets,
                arena.scales,
                scale,
                num_kv_heads,
                head_size,
                output_view[start:end],
                block_size=block_size,
                value_precision_table=arena.device_v_codes,
                value_block_offsets=arena.device_v_offsets,
            )
            continue
        if prefix_len == 0 and num_queries > 1:
            all_keys = key[start:end].float()
            all_values = value[start:end].float()
        else:
            num_blocks = (seq_len + block_size - 1) // block_size
            block_ids = block_tables[req, :num_blocks]
            key_blocks, value_blocks = arena.gather_blocks(
                block_ids.to(torch.int64).tolist()
            )
            all_keys = key_blocks.reshape(-1, num_kv_heads, head_size)[:seq_len]
            all_values = value_blocks.reshape(-1, num_kv_heads, head_size)[:seq_len]
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
        scores = torch.bmm(q, k.transpose(1, 2)) * scale
        key_positions = torch.arange(seq_len, device=query.device)
        query_positions = prefix_len + torch.arange(num_queries, device=query.device)
        scores.masked_fill_(
            key_positions[None, :] > query_positions[:, None], float("-inf")
        )
        probabilities = torch.softmax(scores, dim=-1)
        result = torch.bmm(probabilities, v).transpose(0, 1)
        output_view[start:end].copy_(result.to(output.dtype))

    return output
