#!/bin/bash
cd /root/vllm-pspkv
MODELS="facebook/opt-125m Qwen/Qwen3-0.6B Qwen/Qwen3-1.7B Qwen/Qwen2.5-1.5B Qwen/Qwen3-4B google/gemma-3-270m google/gemma-3-1b-it google/gemma-3-1b-pt google/gemma-3-4b-it Qwen/Qwen3.5-0.8B Qwen/Qwen3.5-2B Qwen/Qwen3.5-4B"
for prec in lossless e4m3 fp4; do
  echo "==================== precision=$prec ===================="
  .venv/bin/python scripts/benchmark_models.py --models $MODELS --precision $prec --dtype float16 --input-len 512 --output-len 128 2>&1 | grep -vE "Processed prompts|INFO|WARNING|rank0"
done
echo "ALL_DONE"
