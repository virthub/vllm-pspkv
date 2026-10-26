# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Forward-scoped controls for isolated pipeline ablation experiments."""

import json
import os
from pathlib import Path

GRAPH_INPUTS = set()


def graph_signature(metadata):
    return tuple(
        tensor.data_ptr()
        for tensor in (
            metadata.block_tables,
            metadata.slot_mapping,
            metadata.psp_kv_device_seq_lens,
        )
    )


def validate_graph_inputs(update_stream, forward_context, *args, **kwargs):
    """Custom kernels read runner-owned device metadata on every replay.

    They do not capture vendor attention task handles or CPU sequence lengths.
    Refuse replay if the runner replaces one of the captured input buffers.
    """
    metadata = forward_context.attn_metadata
    if not isinstance(metadata, dict):
        raise ValueError("single-group full graph requires a metadata dictionary")
    for source in metadata.values():
        if graph_signature(source) not in GRAPH_INPUTS:
            raise RuntimeError("PSP-KV graph metadata address changed; replay refused")


def experiment_mode(owner):
    if owner is not None and hasattr(owner, "_pspkv_experiment_mode"):
        return owner._pspkv_experiment_mode
    control = os.environ.get("PSPKV_PIPELINE_CONTROL_FILE")
    level = int(
        Path(control).read_text().strip()
        if control
        else os.environ.get("PSPKV_ASCEND_PIPELINE_LEVEL", "1")
    )
    if level not in (0, 1, 2, 3):
        raise ValueError(
            "pipeline mode must be 0 (old), 1 (direct), 2 (RoPE), or 3 (graph)"
        )
    if owner is not None:
        owner._pspkv_experiment_mode = level
    return level


def pipeline_level(owner):
    level = min(experiment_mode(owner), 2)
    if owner is not None:
        owner._pspkv_pipeline_level = level
    return level


def install_graph_bypass():
    """Test-only eager/graph ablation inside the same loaded model process.

    Capture with mode 3 first. Modes 0/1/2 call the same underlying runnable
    eagerly; mode 3 replays the captured optimized decode graph. Never patch
    the vendor module on disk or use this switch without explicit opt-in.
    """
    ab_test = os.environ.get("PSPKV_COMPARE_GRAPH_AB") == "1"
    if not ab_test and os.environ.get("PSPKV_ASCEND_FULL_GRAPH") != "1":
        return
    from vllm_ascend.compilation.acl_graph import ACLGraphWrapper

    from vllm.config import CUDAGraphMode
    from vllm.forward_context import get_forward_context

    original = ACLGraphWrapper.__call__
    if getattr(original, "_pspkv_ab", False):
        return

    def call(self, *args, **kwargs):
        owner = get_forward_context()
        mode = experiment_mode(owner) if ab_test else 3
        runtime_mode = owner.cudagraph_runtime_mode
        if mode != 3:
            owner.cudagraph_runtime_mode = CUDAGraphMode.NONE
        selected = owner.cudagraph_runtime_mode == self.runtime_mode
        entry = self.concrete_aclgraph_entries.get(owner.batch_descriptor)
        action = (
            "replay"
            if selected and entry is not None and entry.aclgraph is not None
            else "capture"
            if selected
            else "eager"
        )
        if action == "replay" and os.environ.get("PSPKV_COMPARE_FORMAT") in (
            "basic",
            "enhanced",
            "btkv",
        ):
            # mode=0 compilation intentionally bypasses vendor CPU attention
            # task updates, so validate our device-buffer contract here too.
            validate_graph_inputs(None, owner)
        if not hasattr(self, "_pspkv_ab_seen"):
            self._pspkv_ab_seen = set()
        if (mode, action) not in self._pspkv_ab_seen:
            print(
                "PSPKV_GRAPH_AB " + json.dumps({"mode": mode, "action": action}),
                flush=True,
            )
            self._pspkv_ab_seen.add((mode, action))
        try:
            return original(self, *args, **kwargs)
        finally:
            owner.cudagraph_runtime_mode = runtime_mode

    call._pspkv_ab = True
    ACLGraphWrapper.__call__ = call


def install_deferred_rope():
    """Opt-in adapter: defer supported decode RoPE to the cache writer.

    Prefill, other RoPE styles/dtypes and calls outside a real forward keep
    the plugin implementation. Pending tensors live only on this forward.
    """
    import torch
    import vllm_ascend.ops.rotary_embedding as rotary
    from shared_metadata import current_owner

    from vllm.model_executor.models.qwen2 import Qwen2Attention

    original = rotary.rope_forward_oot
    original_forward = Qwen2Attention.forward
    if getattr(original_forward, "_pspkv_deferred", False):
        return

    def deferred(
        positions,
        query,
        key,
        cos_sin_cache,
        head_size,
        rotary_dim,
        is_neox_style,
        offsets=None,
    ):
        owner = current_owner()
        if (
            owner is not None
            and getattr(owner, "attn_metadata", None) is not None
            and query.shape[0] == key.shape[0] == positions.numel() == 1
            and head_size == rotary_dim == 128
            and is_neox_style
            and offsets is None
            and query.dtype == key.dtype == torch.bfloat16
            and cos_sin_cache.dtype in (torch.bfloat16, torch.float32)
            and pipeline_level(owner) >= 2
            and os.environ.get("PSPKV_COMPARE_PRECISION", "e4m3") in ("e4m3", "e5m2")
            and os.environ.get("PSPKV_COMPARE_OPTIMIZED") == "1"
        ):
            from vllm.v1.attention.backends.psp_kv_ascend_native import (
                load_native_fp8_ops,
            )

            if not hasattr(load_native_fp8_ops(), "rope_scatter_kv"):
                return original(
                    positions,
                    query,
                    key,
                    cos_sin_cache,
                    head_size,
                    rotary_dim,
                    is_neox_style,
                    offsets,
                )
            if not hasattr(owner, "_pspkv_pending_rope"):
                owner._pspkv_pending_rope = {}
            owner._pspkv_pending_rope[key.data_ptr()] = (
                query.data_ptr(),
                positions,
                cos_sin_cache,
            )
            return query, key
        return original(
            positions,
            query,
            key,
            cos_sin_cache,
            head_size,
            rotary_dim,
            is_neox_style,
            offsets,
        )

    def forward(self, positions, hidden_states):
        owner = current_owner()
        rope = self.rotary_emb
        if not (
            owner is not None
            and getattr(owner, "attn_metadata", None) is not None
            and positions.numel() == hidden_states.shape[0] == 1
            and hidden_states.dtype == torch.bfloat16
            and not self.qk_norm
            and self.dual_chunk_attention_config is None
            and rope.head_size == rope.rotary_dim == 128
            and rope.is_neox_style
            and pipeline_level(owner) >= 2
        ):
            return original_forward(self, positions, hidden_states)
        qkv, _ = self.qkv_proj(hidden_states)
        query, key, value = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        query, key = deferred(
            positions,
            query,
            key,
            rope.cos_sin_cache,
            rope.head_size,
            rope.rotary_dim,
            rope.is_neox_style,
        )
        output = self.attn(query, key, value)
        output, _ = self.o_proj(output)
        return output

    # The plugin registered its low-level RoPE function as a custom operator
    # earlier. Rebinding that function does not change the registered callable.
    # Intercept the Qwen2 model's decode entry before it invokes that operator.
    forward._pspkv_deferred = True
    Qwen2Attention.forward = forward


def consume_deferred_rope(owner, query, key):
    pending = getattr(owner, "_pspkv_pending_rope", {}) if owner is not None else {}
    record = pending.pop(key.data_ptr(), None)
    if record is not None and record[0] != query.data_ptr():
        raise RuntimeError("deferred RoPE query/key pairing changed")
    return record
