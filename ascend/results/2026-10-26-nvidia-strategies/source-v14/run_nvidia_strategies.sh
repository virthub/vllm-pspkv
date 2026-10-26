#!/usr/bin/env bash
# Same physical card and execution mode; BF16 brackets the precision sweep.
set -euo pipefail
: "${NPU_ID:?Choose an idle physical card}"
: "${STRATEGY_DIR:?Choose the frozen source/results directory}"
: "${STRATEGY_FORMAT:?basic, enhanced, or btkv}"
STRATEGY_IMAGE=quay.io/ascend/vllm-ascend:v0.20.2rc1
STRATEGY_BUILD=/root/zhang/pspkv-native-build
STRATEGY_SOURCE="${SOURCE_ROOT:-/root/zhang/vllm-pspkv}"
STRATEGY_OVERLAY=/root/zhang/pspkv-ascend-overlay/v0.20.2rc1
STRATEGY_BASE_PORT="${BASE_PORT:-18220}"
STRATEGY_TEST_PORT="$((STRATEGY_BASE_PORT + 1))"
STRATEGY_BASE="pspkv-nvidia-${RUN_TAG:-20261026}-$STRATEGY_FORMAT-bf16"
STRATEGY_ACTIVE=""
mkdir -p "$STRATEGY_DIR/results/$STRATEGY_FORMAT"
cleanup() {
  [[ -z "$STRATEGY_ACTIVE" ]] || docker stop --time 30 "$STRATEGY_ACTIVE" >/dev/null 2>&1 || true
  docker stop --time 30 "$STRATEGY_BASE" >/dev/null 2>&1 || true
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

launch() {
  local name="$1" port="$2" precision="$3"
  local vars=() mounts=() attention=() import_path=/vllm-workspace/vllm-ascend
  local graph='{"mode":3,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1]}'
  if [[ "$precision" != native ]]; then
    import_path=/strategy:/workspace/pspkv-src:/vllm-workspace/vllm-ascend
    local bytes=1
    [[ "$precision" != lossless ]] || bytes=2
    [[ "$precision" != fp4 ]] || bytes=0.5
    local generation
    case "$STRATEGY_FORMAT" in
      basic) generation=0 ;;
      enhanced) generation=1 ;;
      btkv) generation=2 ;;
      *) echo "Unknown format: $STRATEGY_FORMAT" >&2; return 2 ;;
    esac
    local adaptive=0 selected_precision="$precision"
    if [[ "$precision" == adaptive ]]; then adaptive=1; selected_precision=e4m3; fi
    vars=(
      -e PSPKV_NVIDIA_STRATEGIES=1 -e "PSPKV_FORMAT=$generation"
      -e "PSPKV_PRECISION=$selected_precision" -e "PSPKV_AVG_BYTES=$bytes"
      -e "PSPKV_ADAPTIVE_PRECISION=$adaptive"
      -e "PSPKV_COMPARE_FORMAT=$STRATEGY_FORMAT" -e "PSPKV_COMPARE_PRECISION=$precision"
      -e PSPKV_COMPARE_OPTIMIZED=1 -e PSPKV_ASCEND_PIPELINE_LEVEL=3
      -e PSPKV_ASCEND_SHARE_METADATA=1 -e PSPKV_ASCEND_FULL_GRAPH=1
      -e "PSPKV_ASCEND_SPLITS=${ATTENTION_SPLITS:-8}"
      -e "PSPKV_STRATEGY_DECODE_PATH=${DECODE_PATH:-paged}"
      -e PSPKV_STRATEGY_INIT_PA_GRAPH=1
      -e PSPKV_ASCEND_NATIVE_LIB=/strategy-native/libpspkv_strategy.so
    )
    attention=(--attention-backend PSP_KV --kv-cache-dtype fp8)
    mounts=(
      -v "${NATIVE_LIBRARY_PATH:-$STRATEGY_DIR/../${NATIVE_LIBRARY:-libpspkv_strategy.so}}:/strategy-native/libpspkv_strategy.so:ro"
      -v "$STRATEGY_OVERLAY/vllm_ascend/platform.py:/vllm-workspace/vllm-ascend/vllm_ascend/platform.py:ro"
      -v "$STRATEGY_OVERLAY/vllm_ascend/worker/model_runner_v1.py:/vllm-workspace/vllm-ascend/vllm_ascend/worker/model_runner_v1.py:ro"
      -v "$STRATEGY_DIR/backend.py:/workspace/pspkv-src/vllm/v1/attention/backends/psp_kv_ascend.py:ro"
      -v "$STRATEGY_DIR/strategy.py:/workspace/pspkv-src/vllm/v1/attention/backends/psp_kv_ascend_strategy.py:ro"
      -v "$STRATEGY_DIR/ops.py:/workspace/pspkv-src/vllm/v1/attention/backends/psp_kv_ascend_ops.py:ro"
      -v "$STRATEGY_DIR/attention.py:/workspace/pspkv-src/vllm/model_executor/layers/attention/attention.py:ro"
      -v "$STRATEGY_DIR/kv_cache_interface.py:/workspace/pspkv-src/vllm/v1/kv_cache_interface.py:ro"
    )
    if [[ "${CANDIDATE_COMPILE_MODE:-0}" == 0 ]]; then
      graph='{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1]}'
    fi
  fi
  docker run -d --name "$name" --platform linux/arm64 --shm-size=1g \
    --device "/dev/davinci$NPU_ID" --device /dev/davinci_manager \
    --device /dev/devmm_svm --device /dev/hisi_hdc \
    --volumes-from pspkv-vector-fused-dev-20261024 \
    -v "$STRATEGY_SOURCE:/workspace/pspkv-src:ro" \
    -v "$STRATEGY_DIR:/strategy:ro" \
    -v /root/vllm-ascend-smoke-cache:/root/.cache \
    "${mounts[@]}" -p "127.0.0.1:$port:8000" \
    -e "PYTHONPATH=$import_path" -e VLLM_USE_MODELSCOPE=True \
    -e VLLM_WORKER_MULTIPROC_METHOD=spawn "${vars[@]}" \
    --entrypoint bash "$STRATEGY_IMAGE" -lc \
    'source /usr/local/Ascend/ascend-toolkit/set_env.sh; exec /workspace/pspkv-native-build/.venv/bin/python /strategy/serve_initialized_npu.py "$@"' \
    bash serve Qwen/Qwen2.5-7B --host 0.0.0.0 --port 8000 --seed 123 \
    --tensor-parallel-size 1 --block-size 128 --max-num-seqs 1 \
    --max-model-len "${MAX_MODEL_LEN:-8320}" --max-num-batched-tokens "${MAX_MODEL_LEN:-8320}" \
    --gpu-memory-utilization 0.35 --no-enable-prefix-caching --no-enable-chunked-prefill \
    --additional-config '{"pa_shape_list":[1]}' \
    --compilation-config "$graph" "${attention[@]}"
}

wait_ready() {
  local port="$1" name="$2"
  for _ in $(seq 1 240); do
    curl --noproxy '*' -fsS --max-time 2 "http://127.0.0.1:$port/health" >/dev/null 2>&1 && return 0
    [[ "$(docker inspect -f '{{.State.Running}}' "$name")" == true ]] || return 1
    sleep 2
  done
  return 1
}

run_client() {
  local mode="$1" output="$2"
  docker run --rm --network host --platform linux/arm64 \
    -v "$STRATEGY_BUILD:/workspace/pspkv-native-build:ro" -v "$STRATEGY_DIR:/strategy" \
    --entrypoint /workspace/pspkv-native-build/.venv/bin/python "$STRATEGY_IMAGE" \
    /strategy/compare_nvidia_client.py --baseline-port "$STRATEGY_BASE_PORT" \
    --candidate-port "$STRATEGY_TEST_PORT" --case "$STRATEGY_FORMAT:$output" \
    --single-mode "$mode" --pairs "${BENCHMARK_PAIRS:-12}" \
    --long-tokens "${LONG_INPUT_TOKENS:-8192}" \
    --output "/strategy/results/$STRATEGY_FORMAT/$output.json" \
    > "$STRATEGY_DIR/results/$STRATEGY_FORMAT/$output-client.log" 2>&1
}

run_baseline() {
  local tag="$1"
  STRATEGY_BASE="pspkv-nvidia-${RUN_TAG:-20261026}-$STRATEGY_FORMAT-$tag"
  launch "$STRATEGY_BASE" "$STRATEGY_BASE_PORT" native > "$STRATEGY_DIR/results/$STRATEGY_FORMAT/$tag-launch.log" 2>&1
  if wait_ready "$STRATEGY_BASE_PORT" "$STRATEGY_BASE"; then
    run_client bf16 "$tag"
  else
    docker logs "$STRATEGY_BASE" > "$STRATEGY_DIR/results/$STRATEGY_FORMAT/$tag-server.log" 2>&1
    return 1
  fi
  docker logs "$STRATEGY_BASE" > "$STRATEGY_DIR/results/$STRATEGY_FORMAT/$tag-server.log" 2>&1
  docker inspect "$STRATEGY_BASE" > "$STRATEGY_DIR/results/$STRATEGY_FORMAT/$tag-inspect.json"
  docker stop --time 30 "$STRATEGY_BASE" >/dev/null
}

STRATEGY_FAILED=0
if [[ "${RUN_BASELINES:-1}" == 1 && "${RUN_INITIAL_BASELINE:-1}" == 1 ]]; then
  run_baseline bf16-before
fi
for precision in ${STRATEGY_PRECISIONS:-lossless e4m3 e5m2 fp4 adaptive}; do
  STRATEGY_ACTIVE="pspkv-nvidia-${RUN_TAG:-20261026}-$STRATEGY_FORMAT-$precision"
  rc=0
  launch "$STRATEGY_ACTIVE" "$STRATEGY_TEST_PORT" "$precision" > "$STRATEGY_DIR/results/$STRATEGY_FORMAT/$precision-launch.log" 2>&1
  if wait_ready "$STRATEGY_TEST_PORT" "$STRATEGY_ACTIVE"; then
    run_client candidate "$precision" || rc=$?
  else
    rc=1
  fi
  docker logs "$STRATEGY_ACTIVE" > "$STRATEGY_DIR/results/$STRATEGY_FORMAT/$precision-server.log" 2>&1
  docker inspect "$STRATEGY_ACTIVE" > "$STRATEGY_DIR/results/$STRATEGY_FORMAT/$precision-inspect.json"
  docker stop --time 30 "$STRATEGY_ACTIVE" >/dev/null
  STRATEGY_ACTIVE=""
  printf '%s\t%s\n' "$precision" "$rc" >> "$STRATEGY_DIR/results/$STRATEGY_FORMAT/status.tsv"
  if [[ "$rc" != 0 ]]; then STRATEGY_FAILED=1; fi
done
if [[ "${RUN_BASELINES:-1}" == 1 ]]; then run_baseline bf16-after; fi
exit "$STRATEGY_FAILED"
