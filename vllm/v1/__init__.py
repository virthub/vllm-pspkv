"""vLLM V1 engine architecture.

``v1`` names the engine stack — scheduler, KV cache, attention backends,
executor — not a model-runner generation. The current model runner is Model
Runner V2, in :mod:`vllm.v1.worker.gpu.model_runner`; Model Runner V1 no longer
exists, and ``VLLM_USE_V2_MODEL_RUNNER`` only rejects values other than ``1``.
"""
