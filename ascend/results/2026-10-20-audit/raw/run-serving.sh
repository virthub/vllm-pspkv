#!/usr/bin/env bash
# Run serially on NPU 4; retain logs and stopped containers for inspection.
set -u
AUDIT=/root/zhang/pspkv-audit-20261020
SRC=/root/zhang/vllm-pspkv
IMAGE=quay.io/ascend/vllm-ascend:v0.20.2rc1
PORT=18120

while [[ "$(docker inspect pspkv-audit-20261020-checks --format '{{.State.Running}}')" == true ]]; do
  sleep 5
done

run_mode() {
  local mode="$1"
  shift
  local name="pspkv-audit-20261020-${mode}"
  local rc=0
  if [[ "$mode" == stock-bf16 ]]; then
    docker run -d --name "$name" --platform linux/arm64 --shm-size=1g \
      --device /dev/davinci4 --device /dev/davinci_manager \
      --device /dev/devmm_svm --device /dev/hisi_hdc \
      -v /usr/local/dcmi:/usr/local/dcmi:ro \
      -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64:ro \
      -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info:ro \
      -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
      -v /root/vllm-ascend-smoke-cache:/root/.cache \
      -p "127.0.0.1:${PORT}:8000" -e VLLM_USE_MODELSCOPE=True \
      "$IMAGE" vllm serve Qwen/Qwen3-0.6B --host 0.0.0.0 --port 8000 \
      --tensor-parallel-size 1 --block-size 128 --max-num-seqs 1 \
      --max-model-len 256 --gpu-memory-utilization 0.35 \
      --no-enable-prefix-caching --enforce-eager >"$AUDIT/$mode-launch.log" 2>&1 || rc=$?
  else
    env NPU_ID=4 HOST_PORT="$PORT" CONTAINER_NAME="$name" "$@" \
      bash "$SRC/ascend/smoke_910b3.sh" >"$AUDIT/$mode-launch.log" 2>&1 || rc=$?
  fi
  if [[ "$rc" == 0 ]]; then
    docker run --rm --network host --platform linux/arm64 \
      --entrypoint /workspace/pspkv-native-build/.venv/bin/python \
      -v /root/zhang/pspkv-native-build:/workspace/pspkv-native-build:ro \
      -v "$AUDIT:/audit" "$IMAGE" /audit/smoke_client.py \
      --port "$PORT" --output "/audit/$mode-responses.json" \
      >"$AUDIT/$mode-client.log" 2>&1 || rc=$?
    docker logs "$name" >"$AUDIT/$mode-server.log" 2>&1
    docker inspect "$name" >"$AUDIT/$mode-inspect.json"
    docker stop --time 30 "$name" >>"$AUDIT/$mode-launch.log" 2>&1
  fi
  printf '%s\t%s\n' "$mode" "$rc" | tee -a "$AUDIT/serving-status.tsv"
}

run_mode stock-bf16
run_mode native-fp8 PSPKV_ASCEND_NATIVE_FP8=1 PSPKV_ASCEND_NATIVE_DECODE=1
run_mode variable-native PSPKV_ASCEND_VARIABLE_PRECISION=1 PSPKV_ASCEND_NATIVE_VARIABLE=1 PSPKV_ASCEND_LOG_FORMAT_COUNTS=1
run_mode static-fp8 PSPKV_ASCEND_VARIABLE_PRECISION=1 PSPKV_ASCEND_NATIVE_VARIABLE=1 PSPKV_ASCEND_STATIC_PRECISION=/workspace/pspkv-src/ascend/precision_e4m3.json
run_mode native-fp4 PSPKV_ASCEND_UNIFORM_FP4=1 PSPKV_ASCEND_NATIVE_FP4=1
run_mode grouped-fp4 PSPKV_ASCEND_UNIFORM_FP4=1 PSPKV_ASCEND_NATIVE_FP4=1 PSPKV_ASCEND_FP4_GROUP_SIZE=1
run_mode asymmetric PSPKV_ASCEND_VARIABLE_PRECISION=1 PSPKV_ASCEND_NATIVE_VARIABLE=1 PSPKV_ASCEND_STATIC_PRECISION=/workspace/pspkv-src/ascend/precision_mix4_k1v3.json PSPKV_AVG_K_BYTES=1.0 PSPKV_AVG_V_BYTES=0.5
date -u +%FT%TZ > "$AUDIT/serving-finished.txt"
