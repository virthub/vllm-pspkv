#!/usr/bin/env bash
# Test precision routes in isolated, immutable-source containers.
set -euo pipefail
: "${NPU_ID:?Choose an idle NPU}"
: "${AUDIT_DIR:?Choose the frozen audit directory}"
: "${AUDIT_CASES:?Space-separated route names}"
PSPKV_SOURCE="${SOURCE_ROOT:-/root/zhang/vllm-pspkv}"
PSPKV_BUILD=/root/zhang/pspkv-native-build
PSPKV_IMAGE=quay.io/ascend/vllm-ascend:v0.20.2rc1
PSPKV_DRIVER=pspkv-vector-fused-dev-20261024
PSPKV_MODEL="${MODEL:-Qwen/Qwen2.5-7B}"
PSPKV_PORT="${HOST_PORT:-18142}"
PSPKV_CONTAINER=""
cleanup() {
  if [[ -n "$PSPKV_CONTAINER" ]]; then
    docker stop --time 30 "$PSPKV_CONTAINER" >/dev/null 2>&1 || true
    PSPKV_CONTAINER=""
  fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
mkdir -p "$AUDIT_DIR/results"
read -r -a PSPKV_CASES <<< "$AUDIT_CASES"
for route in "${PSPKV_CASES[@]}"; do
  PSPKV_CONTAINER="pspkv-precision-audit-${RUN_TAG:-20261026}-$route"
  args=(--attention-backend PSP_KV --kv-cache-dtype fp8)
  runner=()
  vars=(
    -e PSPKV_ASCEND_REFERENCE_FP8=e4m3
    -e PSPKV_ASCEND_NATIVE_FP8=1
    -e PSPKV_ASCEND_NATIVE_DECODE=1
    -e PSPKV_ASCEND_LOG_FORMAT_COUNTS=1
    -e PSPKV_AVG_BYTES=1.0
  )
  case "$route" in
    stock-bf16) args=(); vars=() ;;
    fixed-e5) vars+=(-e PSPKV_ASCEND_REFERENCE_FP8=e5m2) ;;
    adaptive-fp8)
      vars+=(-e PSPKV_ASCEND_REFERENCE_FP8=adaptive -e PSPKV_ASCEND_LOG_PRECISION=1) ;;
    calibration|dynamic-conservative|dynamic-fp4|static-fp8|static-fp4-grouped|asymmetric-k8v4)
      vars+=(
        -e PSPKV_ASCEND_NATIVE_FP8=0 -e PSPKV_ASCEND_NATIVE_DECODE=0
        -e PSPKV_ASCEND_VARIABLE_PRECISION=1 -e PSPKV_ASCEND_NATIVE_VARIABLE=1
        -e PSPKV_ASCEND_MAX_RELATIVE_RMSE=0.05
      )
      runner=(-v /root/zhang/pspkv-ascend-overlay/v0.20.2rc1/vllm_ascend/worker/model_runner_v1.py:/vllm-workspace/vllm-ascend/vllm_ascend/worker/model_runner_v1.py:ro)
      case "$route" in
        calibration) vars+=(-e PSPKV_ASCEND_CALIBRATE=1 -e PSPKV_ASCEND_CALIBRATE_GROUPS=4) ;;
        dynamic-conservative) vars+=(-e PSPKV_AVG_BYTES=0.75) ;;
        dynamic-fp4) vars+=(-e PSPKV_ASCEND_MAX_RELATIVE_RMSE=2.0) ;;
        static-fp8|static-fp4-grouped|asymmetric-k8v4)
          for _ in $(seq 1 150); do
            [[ -f "$AUDIT_DIR/maps-ready.txt" ]] && break
            sleep 2
          done
          test -f "$AUDIT_DIR/maps-ready.txt"
          vars+=(-e "PSPKV_ASCEND_STATIC_PRECISION=/audit/$route-map.json")
          if [[ "$route" == static-fp4-grouped ]]; then
            vars+=(-e PSPKV_AVG_BYTES=0.5)
          elif [[ "$route" == asymmetric-k8v4 ]]; then
            vars+=(-e PSPKV_AVG_K_BYTES=1.0 -e PSPKV_AVG_V_BYTES=0.5)
          fi
          ;;
      esac
      ;;
    uniform-fp4)
      vars+=(
        -e PSPKV_ASCEND_NATIVE_FP8=0 -e PSPKV_ASCEND_NATIVE_DECODE=0
        -e PSPKV_ASCEND_UNIFORM_FP4=1 -e PSPKV_ASCEND_NATIVE_FP4=1
      )
      runner=(-v /root/zhang/pspkv-ascend-overlay/v0.20.2rc1/vllm_ascend/worker/model_runner_v1.py:/vllm-workspace/vllm-ascend/vllm_ascend/worker/model_runner_v1.py:ro)
      ;;
    *) echo "Unknown route: $route" >&2; exit 2 ;;
  esac
  mounts=(
    -v /root/zhang/pspkv-ascend-overlay/v0.20.2rc1/vllm_ascend/platform.py:/vllm-workspace/vllm-ascend/vllm_ascend/platform.py:ro
    -v "$AUDIT_DIR/backend.py:/workspace/pspkv-src/vllm/v1/attention/backends/psp_kv_ascend.py:ro"
    -v "$AUDIT_DIR/ops.py:/workspace/pspkv-src/vllm/v1/attention/backends/psp_kv_ascend_ops.py:ro"
  )
  if [[ "$route" == stock-bf16 ]]; then
    mounts=()
    import_path=/vllm-workspace/vllm-ascend
  else
    import_path=/audit:/workspace/pspkv-src:/vllm-workspace/vllm-ascend
  fi
  docker run -d --name "$PSPKV_CONTAINER" --platform linux/arm64 --shm-size=1g \
    --device "/dev/davinci$NPU_ID" --device /dev/davinci_manager \
    --device /dev/devmm_svm --device /dev/hisi_hdc \
    --volumes-from "$PSPKV_DRIVER" \
    -v "$PSPKV_SOURCE:/workspace/pspkv-src:ro" \
    -v "$AUDIT_DIR:/audit:ro" \
    -v /root/vllm-ascend-smoke-cache:/root/.cache \
    "${mounts[@]}" "${runner[@]}" \
    -p "127.0.0.1:$PSPKV_PORT:8000" \
    -e "PYTHONPATH=$import_path" -e VLLM_USE_MODELSCOPE=True \
    -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
    -e PSPKV_ASCEND_NATIVE_LIB=/audit/libpspkv_pipeline.so \
    "${vars[@]}" --entrypoint bash "$PSPKV_IMAGE" -lc \
    'source /usr/local/Ascend/ascend-toolkit/set_env.sh; exec /workspace/pspkv-native-build/.venv/bin/python /audit/serve_initialized_npu.py "$@"' \
    bash serve "$PSPKV_MODEL" --host 0.0.0.0 --port 8000 --seed 123 \
    --tensor-parallel-size 1 --block-size 128 --max-num-seqs 1 \
    --max-model-len 256 --gpu-memory-utilization 0.35 \
    "${args[@]}" --no-enable-prefix-caching --enforce-eager \
    > "$AUDIT_DIR/results/$route-launch.log" 2>&1
  ready=0
  for _ in $(seq 1 150); do
    if curl --noproxy '*' -fsS --max-time 2 "http://127.0.0.1:$PSPKV_PORT/health" >/dev/null 2>&1; then
      ready=1; break
    fi
    [[ "$(docker inspect -f '{{.State.Running}}' "$PSPKV_CONTAINER")" == true ]] || break
    sleep 2
  done
  rc=1
  if [[ "$ready" == 1 ]]; then
    rc=0
    docker run --rm --network host --platform linux/arm64 \
      -v "$PSPKV_BUILD:/workspace/pspkv-native-build:ro" -v "$AUDIT_DIR:/audit" \
      --entrypoint /workspace/pspkv-native-build/.venv/bin/python "$PSPKV_IMAGE" \
      /audit/audit_precision_client.py --port "$PSPKV_PORT" --case "$route" \
      --model "$PSPKV_MODEL" --output "/audit/results/$route.json" \
      > "$AUDIT_DIR/results/$route-client.log" 2>&1 || rc=$?
  fi
  docker logs "$PSPKV_CONTAINER" > "$AUDIT_DIR/results/$route-server.log" 2>&1
  docker inspect "$PSPKV_CONTAINER" > "$AUDIT_DIR/results/$route-inspect.json"
  cleanup
  printf '%s\t%s\n' "$route" "$rc" >> "$AUDIT_DIR/results/status-$NPU_ID.tsv"
  if [[ "$route" == calibration && "$rc" == 0 ]]; then
    docker run --rm --platform linux/arm64 \
      -v "$PSPKV_BUILD:/workspace/pspkv-native-build:ro" -v "$AUDIT_DIR:/audit" \
      --entrypoint bash "$PSPKV_IMAGE" -c '
        set -e
        for spec in "static-fp8:1:1" "static-fp4-grouped:3:3" "asymmetric-k8v4:1:3"; do
          IFS=: read -r name k v <<< "$spec"
          /workspace/pspkv-native-build/.venv/bin/python /audit/calibrate_precision.py \
            --log /audit/results/calibration-server.log --k-code "$k" --v-code "$v" \
            --output "/audit/$name-map.json"
        done
        touch /audit/maps-ready.txt
      ' > "$AUDIT_DIR/results/calibration-maps.log" 2>&1
  fi
done
