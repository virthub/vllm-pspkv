set -u
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd /workspace/pspkv-src
run_check() {
  label="$1"
  shift
  timeout 600 "$@" >"/audit/$label.log" 2>&1
  rc=$?
  printf '%s\t%s\n' "$label" "$rc" | tee -a /audit/status.tsv
}
P=/workspace/pspkv-native-build/.venv/bin/python
run_check environment "$P" -c 'import json,sys,torch,torch_npu,importlib.metadata as m; print(json.dumps({"python":sys.version,"torch":torch.__version__,"torch_npu":m.version("torch-npu"),"vllm":m.version("vllm"),"vllm_ascend":m.version("vllm-ascend"),"npu":torch.npu.get_device_name(0)},indent=2))'
run_check fp8_reference "$P" ascend/check_fp8_reference_npu.py
run_check packed_reference "$P" ascend/check_packed_reference_npu.py
run_check variable_reference "$P" ascend/check_variable_npu.py
run_check native_fp8 "$P" ascend/native/check_native_fp8_npu.py
run_check native_variable "$P" ascend/check_variable_native_npu.py
run_check cpu_tests "$P" -m pytest --confcutdir=tests/v1/attention -o cache_dir=/tmp/pspkv-audit-pytest tests/v1/attention/test_psp_kv_ascend_reference.py tests/v1/attention/test_psp_kv_ascend_packed.py tests/v1/attention/test_psp_kv_ascend_policy.py tests/v1/attention/test_psp_kv_ascend_copy.py tests/v1/attention/test_psp_kv_ascend_arena.py -q
run_check correctness "$P" -m pytest --confcutdir=tests/v1/attention -o cache_dir=/tmp/pspkv-audit-pytest tests/v1/attention/test_psp_kv_correctness.py -q
run_check mixed_precision "$P" -m pytest --confcutdir=tests/v1/attention -o cache_dir=/tmp/pspkv-audit-pytest tests/v1/attention/test_psp_kv_mixed_precision.py -q
date -u +%FT%TZ > /audit/checks-finished.txt
