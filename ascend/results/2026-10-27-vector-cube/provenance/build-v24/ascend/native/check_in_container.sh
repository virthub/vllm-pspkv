#!/usr/bin/env bash
set -euo pipefail

PSPKV_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
: "${NPU_ID:?Set NPU_ID to a free 910B3 device}"
IMAGE="${IMAGE:-quay.io/ascend/vllm-ascend:v0.20.2rc1}"
BUILD_DIR="${PSPKV_NATIVE_BUILD_DIR:-/root/zhang/pspkv-native-build}"
IMAGE_PYTHONPATH="$(
  docker image inspect "$IMAGE" --format '{{range .Config.Env}}{{println .}}{{end}}' \
    | sed -n 's/^PYTHONPATH=//p'
)"
test -f "$BUILD_DIR/libpspkv_ascend_native.so"
test -c "/dev/davinci${NPU_ID}"

docker run --rm --platform linux/arm64 --entrypoint /bin/bash \
  --device "/dev/davinci${NPU_ID}" \
  --device /dev/davinci_manager \
  --device /dev/devmm_svm \
  --device /dev/hisi_hdc \
  -v "$PSPKV_ROOT:/workspace/pspkv-src:ro" \
  -v "$BUILD_DIR:/workspace/pspkv-native-build:ro" \
  -v /usr/local/dcmi:/usr/local/dcmi:ro \
  -v /usr/local/sbin/npu-smi:/usr/local/bin/npu-smi:ro \
  -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64:ro \
  -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info:ro \
  -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
  -e "PYTHONPATH=/workspace/pspkv-src:${IMAGE_PYTHONPATH}" \
  -e PSPKV_ASCEND_NATIVE_LIB=/workspace/pspkv-native-build/libpspkv_ascend_native.so \
  "$IMAGE" -lc '
    set -euo pipefail
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    /workspace/pspkv-native-build/.venv/bin/python \
      /workspace/pspkv-src/ascend/native/check_native_fp8_npu.py
  '
