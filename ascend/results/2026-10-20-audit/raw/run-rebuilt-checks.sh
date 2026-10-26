set -u
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd /workspace/pspkv-src
P=/workspace/pspkv-native-build/.venv/bin/python
run_check() {
  label="$1"
  shift
  timeout 600 "$@" >"/audit/rebuilt-$label.log" 2>&1
  rc=$?
  printf '%s\t%s\n' "$label" "$rc" | tee -a /audit/rebuilt-status.tsv
}
run_check native_fp8 "$P" ascend/native/check_native_fp8_npu.py
run_check diagnostics "$P" /audit/diagnose_npu.py
run_check correctness "$P" -m pytest --confcutdir=tests/v1/attention -o cache_dir=/tmp/pspkv-audit-pytest tests/v1/attention/test_psp_kv_correctness.py -q
run_check mixed_precision "$P" -m pytest --confcutdir=tests/v1/attention -o cache_dir=/tmp/pspkv-audit-pytest tests/v1/attention/test_psp_kv_mixed_precision.py -q
date -u +%FT%TZ > /audit/rebuilt-finished.txt