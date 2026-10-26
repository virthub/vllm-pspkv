set -u
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd /workspace/pspkv-src
/workspace/pspkv-native-build/.venv/bin/python /comparison/compare_formats_control.py --workload gqa_512 --precision bf16 --precision e4m3 --shared-buffers --rounds 7 --iterations 20 --output /comparison/control.json > /comparison/control.log 2>&1
rc=$?
printf 'shared_buffers\t%s\n' "$rc" > /comparison/control-status.tsv
exit "$rc"