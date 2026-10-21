#!/usr/bin/env bash
set -euo pipefail

# Run the correctness-first PSP-KV backend on one otherwise idle 910B3.
# The route file is a copy of vllm-ascend v0.20.2rc1's platform.py with
# patches/v0.20.2rc1-platform-psp-kv.patch applied.

PSPKV_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${NPU_ID:?Set NPU_ID to an idle, usable 910B3 after checking npu-smi info}"
HOST_PORT="${HOST_PORT:-18083}"
CONTAINER_NAME="${CONTAINER_NAME:-pspkv-ascend-ref-npu${NPU_ID}}"
ROUTE_PLATFORM="${ROUTE_PLATFORM:-/root/zhang/pspkv-ascend-overlay/v0.20.2rc1/vllm_ascend/platform.py}"
ROUTE_MODEL_RUNNER="${ROUTE_MODEL_RUNNER:-/root/zhang/pspkv-ascend-overlay/v0.20.2rc1/vllm_ascend/worker/model_runner_v1.py}"
MODEL_CACHE="${MODEL_CACHE:-/root/vllm-ascend-smoke-cache}"
MODEL="${MODEL:-Qwen/Qwen3-0.6B}"
IMAGE="${IMAGE:-quay.io/ascend/vllm-ascend:v0.20.2rc1}"
IMAGE_PYTHONPATH="$(
  docker image inspect "$IMAGE" --format '{{range .Config.Env}}{{println .}}{{end}}' \
    | sed -n 's/^PYTHONPATH=//p'
)"

test -f "$PSPKV_ROOT/vllm/v1/attention/backends/psp_kv_ascend.py"
test -f "$ROUTE_PLATFORM"
FP4_RUNNER_MOUNT=()
if [[ "${PSPKV_ASCEND_UNIFORM_FP4:-0}" == "1" || "${PSPKV_ASCEND_VARIABLE_PRECISION:-0}" == "1" ]]; then
  test -f "$ROUTE_MODEL_RUNNER"
  FP4_RUNNER_MOUNT=(
    -v "$ROUTE_MODEL_RUNNER:/vllm-workspace/vllm-ascend/vllm_ascend/worker/model_runner_v1.py:ro"
  )
fi
NATIVE_FP8_MOUNT=()
NATIVE_FP8_LIB_CONTAINER=/workspace/pspkv-native/libpspkv_ascend_native.so
if [[ "${PSPKV_ASCEND_NATIVE_FP8:-0}" == "1" || "${PSPKV_ASCEND_NATIVE_FP4:-0}" == "1" || "${PSPKV_ASCEND_NATIVE_VARIABLE:-0}" == "1" ]]; then
  NATIVE_FP8_LIB_HOST="${PSPKV_ASCEND_NATIVE_LIB_HOST:-/root/zhang/pspkv-native-build/libpspkv_ascend_native.so}"
  test -f "$NATIVE_FP8_LIB_HOST"
  NATIVE_FP8_MOUNT=(-v "$NATIVE_FP8_LIB_HOST:$NATIVE_FP8_LIB_CONTAINER:ro")
fi
test -d "$MODEL_CACHE"
test -c "/dev/davinci${NPU_ID}"

if ! docker run --rm \
  --platform linux/arm64 \
  --device "/dev/davinci${NPU_ID}" \
  --device /dev/davinci_manager \
  --device /dev/devmm_svm \
  --device /dev/hisi_hdc \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/sbin/npu-smi:/usr/local/bin/npu-smi:ro \
  -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64 \
  -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  "$IMAGE" npu-smi info; then
  echo "NPU ${NPU_ID} is not accessible inside Docker; model launch skipped." >&2
  exit 1
fi

docker run -d \
  --name "$CONTAINER_NAME" \
  --platform linux/arm64 \
  --shm-size=1g \
  --device "/dev/davinci${NPU_ID}" \
  --device /dev/davinci_manager \
  --device /dev/devmm_svm \
  --device /dev/hisi_hdc \
  -v /usr/local/dcmi:/usr/local/dcmi \
  -v /usr/local/sbin/npu-smi:/usr/local/bin/npu-smi:ro \
  -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64 \
  -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info \
  -v /etc/ascend_install.info:/etc/ascend_install.info \
  -v "$MODEL_CACHE:/root/.cache" \
  -v "$PSPKV_ROOT:/workspace/pspkv-src:ro" \
  -v "$ROUTE_PLATFORM:/vllm-workspace/vllm-ascend/vllm_ascend/platform.py:ro" \
  "${FP4_RUNNER_MOUNT[@]}" \
  "${NATIVE_FP8_MOUNT[@]}" \
  -p "127.0.0.1:${HOST_PORT}:8000" \
  -e "PYTHONPATH=/workspace/pspkv-src:${IMAGE_PYTHONPATH}" \
  -e VLLM_USE_MODELSCOPE=True \
  -e "PSPKV_ASCEND_REFERENCE_FP8=${PSPKV_ASCEND_REFERENCE_FP8:-e4m3}" \
  -e "PSPKV_ASCEND_E4M3_MAX_ABS=${PSPKV_ASCEND_E4M3_MAX_ABS:-240}" \
  -e "PSPKV_ASCEND_LOG_PRECISION=${PSPKV_ASCEND_LOG_PRECISION:-0}" \
  -e "PSPKV_ASCEND_UNIFORM_FP4=${PSPKV_ASCEND_UNIFORM_FP4:-0}" \
  -e "PSPKV_ASCEND_VARIABLE_PRECISION=${PSPKV_ASCEND_VARIABLE_PRECISION:-0}" \
  -e "PSPKV_ASCEND_MAX_RELATIVE_RMSE=${PSPKV_ASCEND_MAX_RELATIVE_RMSE:-0.05}" \
  -e "PSPKV_AVG_BYTES=${PSPKV_AVG_BYTES:-1.0}" \
  -e "PSPKV_AVG_K_BYTES=${PSPKV_AVG_K_BYTES:-}" \
  -e "PSPKV_AVG_V_BYTES=${PSPKV_AVG_V_BYTES:-}" \
  -e "PSPKV_ASCEND_LOG_FORMAT_COUNTS=${PSPKV_ASCEND_LOG_FORMAT_COUNTS:-0}" \
  -e "PSPKV_ASCEND_CALIBRATE=${PSPKV_ASCEND_CALIBRATE:-0}" \
  -e "PSPKV_ASCEND_CALIBRATE_GROUPS=${PSPKV_ASCEND_CALIBRATE_GROUPS:-1}" \
  -e "PSPKV_ASCEND_STATIC_PRECISION=${PSPKV_ASCEND_STATIC_PRECISION:-}" \
  -e "PSPKV_ASCEND_FP4_GROUP_SIZE=${PSPKV_ASCEND_FP4_GROUP_SIZE:-0}" \
  -e "PSPKV_ASCEND_NATIVE_FP4=${PSPKV_ASCEND_NATIVE_FP4:-0}" \
  -e "PSPKV_ASCEND_FP4_SDPA=${PSPKV_ASCEND_FP4_SDPA:-0}" \
  -e "PSPKV_ASCEND_FP4_DIAGNOSTICS=${PSPKV_ASCEND_FP4_DIAGNOSTICS:-0}" \
  -e "PSPKV_ASCEND_FP4_DIAGNOSTICS_STEPS=${PSPKV_ASCEND_FP4_DIAGNOSTICS_STEPS:-2}" \
  -e "PSPKV_ASCEND_NATIVE_FP8=${PSPKV_ASCEND_NATIVE_FP8:-0}" \
  -e "PSPKV_ASCEND_NATIVE_DECODE=${PSPKV_ASCEND_NATIVE_DECODE:-0}" \
  -e "PSPKV_ASCEND_NATIVE_VARIABLE=${PSPKV_ASCEND_NATIVE_VARIABLE:-0}" \
  -e "PSPKV_ASCEND_NATIVE_FUSED=${PSPKV_ASCEND_NATIVE_FUSED:-0}" \
  -e "PSPKV_ASCEND_SDPA_ATTENTION=${PSPKV_ASCEND_SDPA_ATTENTION:-0}" \
  -e "PSPKV_ASCEND_NATIVE_LIB=$NATIVE_FP8_LIB_CONTAINER" \
  "$IMAGE" \
  vllm serve "$MODEL" \
    --host 0.0.0.0 --port 8000 \
    --tensor-parallel-size 1 \
    --block-size 128 \
    --max-num-seqs 1 \
    --max-model-len 256 \
    --gpu-memory-utilization 0.35 \
    --attention-backend PSP_KV \
    --kv-cache-dtype fp8 \
    --no-enable-prefix-caching \
    --enforce-eager

echo "Container: $CONTAINER_NAME"
echo "Logs: docker logs -f $CONTAINER_NAME"
echo "API: http://127.0.0.1:${HOST_PORT}/v1/completions"
