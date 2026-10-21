# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Quality-gated mixed-precision packing of complete Ascend KV blocks."""

import pytest
import torch

from vllm.v1.attention.backends.psp_kv_ascend_policy import (
    pack_completed_blocks_by_error,
    select_completed_block_precision,
)

pytestmark = pytest.mark.skip_global_cleanup


def _example_blocks() -> torch.Tensor:
    return torch.tensor(
        [
            [[0.125, 0.25, 0.125, 0.25]] * 2,
            [[1.125, 1.125, 1.125, 1.125]] * 2,
            [[512.0, 1.125, 512.0, 1.125]] * 2,
            [[1.0078125, 1.0078125, 1.0078125, 1.0078125]] * 2,
        ],
        dtype=torch.bfloat16,
    ).unsqueeze(2)


def test_policy_selects_four_formats_and_packs_real_byte_offsets():
    keys = _example_blocks()
    values = -keys
    cache, scales, choices = pack_completed_blocks_by_error(
        keys, values, max_relative_rmse=0.001
    )

    assert [choice.code for choice in choices] == [3, 1, 2, 0]
    assert cache.precision_table.tolist() == [3, 1, 2, 0]
    assert cache.block_offsets.tolist() == [0, 4, 12, 20]
    assert cache.total_bytes == 36  # one arena; all-BF16 would need 64 bytes
    assert cache.k_buffer.numel() + cache.v_buffer.numel() == 72
    assert scales[:, 0].tolist() == [0.125, 1.0, 1.0, 1.0]

    decoded_keys, decoded_values = cache.gather([0, 1, 2, 3])
    decoded_keys *= scales[:, None, :, None]
    decoded_values *= scales[:, None, :, None]
    torch.testing.assert_close(decoded_keys[[0, 1, 3]], keys[[0, 1, 3]].float())
    torch.testing.assert_close(decoded_values[[0, 1, 3]], values[[0, 1, 3]].float())
    relative_error = (decoded_keys[2] - keys[2].float()).norm() / keys[2].float().norm()
    assert relative_error.item() <= 0.001


def test_policy_uses_worst_kv_head_and_rejects_invalid_inputs():
    key = torch.full((2, 2, 4), 0.125, dtype=torch.bfloat16)
    value = key.clone()
    key[:, 1] = 1.0078125
    value[:, 1] = 1.0078125
    choice = select_completed_block_precision(key, value, 0.001)
    assert choice.code == 0  # head 0 fits FP4, but head 1 needs BF16

    with pytest.raises(ValueError, match="nonnegative"):
        select_completed_block_precision(key, value, -0.1)
    with pytest.raises(ValueError, match="finite K/V"):
        key[0, 0, 0] = float("nan")
        select_completed_block_precision(key, value, 0.1)


def test_grouped_fp4_preserves_small_values_hidden_by_a_key_outlier():
    key = torch.tensor(
        [[[256.0, 0.25, 0.25, 0.25]], [[256.0, 0.25, 0.25, 0.25]]],
        dtype=torch.bfloat16,
    )
    value = torch.full_like(key, 0.25)

    # One scale shared by K and V erases V entirely. The error gate must
    # consider V independently even though K has much more signal energy.
    assert select_completed_block_precision(key, value, 0.001).code == 1
    grouped_choice = select_completed_block_precision(
        key, value, 0.001, fp4_group_size=1
    )
    assert grouped_choice.code == 3
    assert grouped_choice.scales.shape == (2, 1, 4)

    next_key = torch.full_like(key, 1.0078125)
    next_value = next_key.clone()
    cache, scales, choices = pack_completed_blocks_by_error(
        torch.stack((key, next_key)),
        torch.stack((value, next_value)),
        max_relative_rmse=0.001,
        fp4_group_size=1,
    )
    assert [choice.code for choice in choices] == [3, 0]
    assert cache.block_offsets.tolist() == [0, 4]
    assert cache.total_bytes == 20  # per stream; all-BF16 needs 32
    decoded_key, decoded_value = cache.gather([0, 1])
    group_key_scale = scales[0, 0].repeat_interleave(1, dim=-1)
    group_value_scale = scales[0, 1].repeat_interleave(1, dim=-1)
    torch.testing.assert_close(decoded_key[0] * group_key_scale[None], key.float())
    torch.testing.assert_close(
        decoded_value[0] * group_value_scale[None], value.float()
    )
    torch.testing.assert_close(decoded_key[1], next_key.float())
    torch.testing.assert_close(decoded_value[1], next_value.float())
