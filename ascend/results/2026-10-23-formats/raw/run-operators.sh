set -u
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd /workspace/pspkv-src
P=/workspace/pspkv-native-build/.venv/bin/python
"$P" /comparison/compare_formats.py --quick --rounds 2 --iterations 5 --warmups 3 --output /comparison/pilot.json > /comparison/pilot.log 2>&1
rc=$?
printf 'pilot\t%s\n' "$rc" >> /comparison/operator-status.tsv
if [[ "$rc" == 0 ]]; then
 "$P" /comparison/compare_formats.py --output /comparison/operators.json > /comparison/operators.log 2>&1
 rc=$?
 printf 'operators\t%s\n' "$rc" >> /comparison/operator-status.tsv
fi
date -u +%FT%TZ > /comparison/operators-finished.txt
exit "$rc"