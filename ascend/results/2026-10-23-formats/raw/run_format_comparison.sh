#!/usr/bin/env bash
# Test-only model adapter: keep the source checkout and installed plugin intact.
set -u
: "${NPU_ID:?Choose an idle 910B3}"
: "${COMPARISON_DIR:?Set the directory containing the comparison tools}"
SOURCE_ROOT="${SOURCE_ROOT:-/root/zhang/vllm-pspkv}"
IMAGE="${IMAGE:-quay.io/ascend/vllm-ascend:v0.20.2rc1}"
BUILD_DIR="${BUILD_DIR:-/root/zhang/pspkv-native-build}"
MODEL_CACHE="${MODEL_CACHE:-/root/vllm-ascend-smoke-cache}"
PLATFORM_OVERLAY="${PLATFORM_OVERLAY:-/root/zhang/pspkv-ascend-overlay/v0.20.2rc1/vllm_ascend/platform.py}"
HOST_PORT="${HOST_PORT:-18130}"
RUN_ID="${RUN_ID:-20261023}"
ACTIVE_CONTAINER=""

cleanup() {
  if [[ -n "$ACTIVE_CONTAINER" ]]; then
    docker stop --time 30 "$ACTIVE_CONTAINER" >/dev/null 2>&1 || true
    ACTIVE_CONTAINER=""
  fi
}
trap cleanup EXIT INT TERM

cp "$SOURCE_ROOT/vllm/v1/attention/backends/psp_kv_ascend.py" \
  "$COMPARISON_DIR/backend_overlay.py"
cat >> "$COMPARISON_DIR/backend_overlay.py" <<'PY'

# Explicit opt-in for the isolated format comparison containers only.
if os.environ.get("PSPKV_COMPARE_FORMAT"):
    from format_compare_runtime import comparison_attention

    reference_attention = comparison_attention
PY

for format in basic enhanced btkv; do
  name="pspkv-format-${RUN_ID}-${format}"
  ACTIVE_CONTAINER="$name"
  rc=0
  docker run -d --name "$name" --platform linux/arm64 --shm-size=1g \
    --device "/dev/davinci${NPU_ID}" --device /dev/davinci_manager \
    --device /dev/devmm_svm --device /dev/hisi_hdc \
    -v /usr/local/dcmi:/usr/local/dcmi:ro \
    -v /usr/local/sbin/npu-smi:/usr/local/bin/npu-smi:ro \
    -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64:ro \
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info:ro \
    -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
    -v "$MODEL_CACHE:/root/.cache" \
    -v "$SOURCE_ROOT:/workspace/pspkv-src:ro" \
    -v "$PLATFORM_OVERLAY:/vllm-workspace/vllm-ascend/vllm_ascend/platform.py:ro" \
    -v "$COMPARISON_DIR:/comparison:ro" \
    -v "$COMPARISON_DIR/backend_overlay.py:/workspace/pspkv-src/vllm/v1/attention/backends/psp_kv_ascend.py:ro" \
    -v "$BUILD_DIR/libpspkv_ascend_native.so:/comparison-native/libpspkv_ascend_native.so:ro" \
    -p "127.0.0.1:${HOST_PORT}:8000" \
    -e PYTHONPATH=/comparison:/workspace/pspkv-src:/vllm-workspace/vllm-ascend \
    -e VLLM_USE_MODELSCOPE=True \
    -e "PSPKV_COMPARE_FORMAT=$format" -e PSPKV_COMPARE_PRECISION=e4m3 \
    -e PSPKV_ASCEND_NATIVE_LIB=/comparison-native/libpspkv_ascend_native.so \
    "$IMAGE" vllm serve Qwen/Qwen3-0.6B \
    --host 0.0.0.0 --port 8000 --seed 123 \
    --tensor-parallel-size 1 --block-size 128 --max-num-seqs 1 \
    --max-model-len 256 --gpu-memory-utilization 0.35 \
    --attention-backend PSP_KV --kv-cache-dtype fp8 \
    --no-enable-prefix-caching --enforce-eager \
    >"$COMPARISON_DIR/$format-launch.log" 2>&1 || rc=$?
  if [[ "$rc" == 0 ]]; then
    docker run --rm --network host --platform linux/arm64 \
      --entrypoint /workspace/pspkv-native-build/.venv/bin/python \
      -v "$BUILD_DIR:/workspace/pspkv-native-build:ro" \
      -v "$COMPARISON_DIR:/comparison" "$IMAGE" \
      /comparison/compare_formats_client.py --port "$HOST_PORT" \
      --output "/comparison/$format-serving.json" \
      >"$COMPARISON_DIR/$format-client.log" 2>&1 || rc=$?
    docker logs "$name" >"$COMPARISON_DIR/$format-server.log" 2>&1
    docker inspect "$name" >"$COMPARISON_DIR/$format-inspect.json"
    npu-smi info -t proc-mem -i "$NPU_ID" -c 0 \
      >"$COMPARISON_DIR/$format-process-memory.log" 2>&1
    npu-smi info >"$COMPARISON_DIR/$format-npu.log" 2>&1
    cleanup
  fi
  printf '%s\t%s\n' "$format" "$rc" | tee -a "$COMPARISON_DIR/serving-status.tsv"
done
date -u +%FT%TZ > "$COMPARISON_DIR/serving-finished.txt"
