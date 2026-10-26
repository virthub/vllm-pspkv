# 本轮检查命令

CPU，仓库根目录（85 passed）：

```bash
.venv/bin/python -m pytest -q --confcutdir=tests/v1/attention tests/v1/attention/test_psp_kv_ascend_reference.py tests/v1/attention/test_psp_kv_ascend_packed.py tests/v1/attention/test_psp_kv_ascend_policy.py tests/v1/attention/test_psp_kv_ascend_arena.py tests/v1/attention/test_psp_kv_ascend_strategy.py tests/v1/attention/test_psp_kv_shared_metadata.py tests/v1/attention/test_psp_kv_pipeline.py
```

原有 NPU 接口回归（263 passed，原生库 v8）：

```bash
docker run --rm --name pspkv-nvidia-regression-v8-20261026 --platform linux/arm64 --device /dev/davinci3 --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc --volumes-from pspkv-vector-fused-dev-20261024 -v /root/zhang/pspkv-nvidia-strategies-20261026/source-v1:/workspace/pspkv-src:ro -v /root/zhang/pspkv-nvidia-strategies-20261026:/strategy -v /root/zhang/pspkv-nvidia-strategies-20261026/serving-v9/backend.py:/workspace/pspkv-src/vllm/v1/attention/backends/psp_kv_ascend.py:ro -v /root/zhang/pspkv-nvidia-strategies-20261026/serving-v9/ops.py:/workspace/pspkv-src/vllm/v1/attention/backends/psp_kv_ascend_ops.py:ro -v /root/zhang/pspkv-nvidia-strategies-20261026/serving-v9/kv_cache_interface.py:/workspace/pspkv-src/vllm/v1/kv_cache_interface.py:ro -e PYTHONPATH=/strategy/serving-v9:/workspace/pspkv-src:/vllm-workspace/vllm-ascend -e PSPKV_TEST_DEVICE=npu -e PSPKV_ASCEND_NATIVE_LIB=/strategy/libpspkv_strategy_v8.so --entrypoint bash quay.io/ascend/vllm-ascend:v0.20.2rc1 -lc 'source /usr/local/Ascend/ascend-toolkit/set_env.sh; cd /workspace/pspkv-src; /workspace/pspkv-native-build/.venv/bin/python -m pytest -q --confcutdir=tests/v1/attention tests/v1/attention/test_psp_kv_correctness.py tests/v1/attention/test_psp_kv_mixed_precision.py tests/v1/attention/test_psp_kv_ascend_copy.py' > /root/zhang/pspkv-nvidia-strategies-20261026/regression-v8.log 2>&1
```

四精度、RoPE、图重放、prefill 写入及注意力分阶段计时：

```bash
docker run --rm --name pspkv-strategy-check-v9-20261026 --platform linux/arm64 --device /dev/davinci3 --device /dev/davinci_manager --device /dev/devmm_svm --device /dev/hisi_hdc --volumes-from pspkv-vector-fused-dev-20261024 -v /root/zhang/pspkv-nvidia-strategies-20261026:/strategy -v /root/zhang/pspkv-nvidia-strategies-20261026/source-v1:/workspace/pspkv-src:ro -v /root/zhang/vllm-pspkv-develop-00b78e9:/develop:ro -e PYTHONPATH=/workspace/pspkv-src:/vllm-workspace/vllm-ascend -e PSPKV_ASCEND_NATIVE_LIB=/strategy/libpspkv_strategy_v8.so --entrypoint bash quay.io/ascend/vllm-ascend:v0.20.2rc1 -lc 'source /usr/local/Ascend/ascend-toolkit/set_env.sh; /workspace/pspkv-native-build/.venv/bin/python /strategy/checks-v9/ascend/check_nvidia_strategies_npu.py --reference-file /develop/tests/v1/attention/psp_kv_reference.py --output /strategy/checks-v9/result.json' >/root/zhang/pspkv-nvidia-strategies-20261026/checks-v9/check.log 2>&1
```

模型正式测试使用报告中的 serving-v14 启动器；各格式使用卡 4 / 6 / 7。RUN_INITIAL_BASELINE=0，前置原生 BF16 数据来源见 BASELINE-PROVENANCE.json。
