#!/bin/bash
cd /root/vllm-pspkv
MODELS="facebook/opt-125m Qwen/Qwen3-0.6B Qwen/Qwen3-1.7B Qwen/Qwen2.5-1.5B Qwen/Qwen3-4B Qwen/Qwen3.5-2B Qwen/Qwen3.5-4B"
for dt in bfloat16 float16 float32; do
  echo "==================== dtype=$dt ===================="
  .venv/bin/python scripts/benchmark_models.py --models $MODELS --precision lossless --dtype $dt --input-len 512 --output-len 128 2>&1 | grep -vE "Processed prompts|INFO|WARNING|rank0"
done
echo "ALL_DONE"
