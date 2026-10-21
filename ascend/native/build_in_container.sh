#!/usr/bin/env bash
set -euo pipefail

# Build only in the disposable vLLM-Ascend image, keeping host CANN and Python
# installations untouched. The output directory is intentionally outside git.
PSPKV_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
IMAGE="${IMAGE:-quay.io/ascend/vllm-ascend:v0.20.2rc1}"
BUILD_DIR="${PSPKV_NATIVE_BUILD_DIR:-/root/zhang/pspkv-native-build}"
UV_BIN="${PSPKV_NATIVE_UV:-/tmp/pspkv-test-uv}"
test -x "$UV_BIN"
mkdir -p "$BUILD_DIR"

docker run --rm --platform linux/arm64 --entrypoint /bin/bash \
  -v "$PSPKV_ROOT:/workspace/pspkv-src:ro" \
  -v "$BUILD_DIR:/workspace/pspkv-native-build" \
  -v "$UV_BIN:/usr/local/bin/uv:ro" \
  -v /usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64:ro \
  -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info:ro \
  -v /etc/ascend_install.info:/etc/ascend_install.info:ro \
  "$IMAGE" -lc '
    set -euo pipefail
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    if test ! -x /workspace/pspkv-native-build/.venv/bin/python; then
      uv venv --system-site-packages --python /usr/local/python3.11.15/bin/python3 \
        /workspace/pspkv-native-build/.venv
    fi
    /workspace/pspkv-native-build/.venv/bin/python \
      /workspace/pspkv-src/ascend/native/build.py \
      /workspace/pspkv-native-build/libpspkv_ascend_native.so
  '

echo "Built $BUILD_DIR/libpspkv_ascend_native.so"
