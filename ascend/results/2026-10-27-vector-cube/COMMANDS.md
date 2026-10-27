# 复现环境与命令

## 源码与容器

本地工作分支 `ascend`，本轮改动基于 `5b10bc5f036933b74303f23ecc12a6d0e56fc544`。
实验目录 `/root/zhang/pspkv-cube-20261027`，与原仓库及之前的冻结实验隔离。
容器 `pspkv-cube-dev-20261027` 将该目录挂载为 `/experiment`，仅使用物理 NPU 4。
原有 uv 环境为 `/workspace/pspkv-native-build/.venv/bin/python`。

只在原有设备进程退出、确认卡空闲后开始下一轮 NPU 检查/模型服务。
没有重置设备或停止其他用户的任务。

## 构建

以下命令在相同镜像的容器内执行；`SOURCE` 是对应冻结源码目录。

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
/workspace/pspkv-native-build/.venv/bin/python \
  "$SOURCE/ascend/native/build.py" /experiment/libpspkv_native.so
/workspace/pspkv-native-build/.venv/bin/python \
  "$SOURCE/ascend/native/build.py" /experiment/libpspkv_cube.so \
  --source pspkv_cube.asc
```

编译目标为 `dav-2201`。原有纯 Vector native 库与新的 Cube 库分开构建、加载。
各轮目录不覆盖；最终有效源码/库哈希见 `manifest.json`。

## 编码、四精度、图重放与 profile

容器中的 `PYTHONPATH=/workspace/pspkv-src` 指向只读 vLLM 源码；
新增 checker 在 `/experiment/checks-source/ascend` 中执行，保持其配套 timing module 可导入。

```bash
export PSPKV_ASCEND_NATIVE_LIB=/experiment/libpspkv_native.so
/workspace/pspkv-native-build/.venv/bin/python \
  /experiment/check_vector_prefill_npu.py \
  --output /experiment/vector-prefill.json
/workspace/pspkv-native-build/.venv/bin/python \
  /experiment/checks-source/ascend/check_nvidia_strategies_npu.py \
  --reference-file /experiment/serving-encode/develop-reference.py \
  --cube-library /experiment/libpspkv_cube.so \
  --profile-cube --lengths 3,135,1536,8192 --iterations 10 \
  --output /experiment/cube-matrix.json
```

第二个命令同时检查四种精度、随机物理页、GQA、四档 split-K、NaN 尾页、输入/缓存不变、
workspace 边界和动态 query/length 图重放。计时不包含输入构造和精度转换。
每个 profile 外围记录设备 event；graph 内包含 16 次调用，结果除以 16。
cycle profile 是独立的一次插桩调用，不能用于替代未插桩正常路径的时延。

NVIDIA oracle 文件来自 `develop` 的 `00b78e954225947fbd05281fc19bd9f22164f851`。
本轮没有 NVIDIA 硬件运行结果。

## 原有 NPU 回归

```bash
PSPKV_TEST_DEVICE=npu /workspace/pspkv-native-build/.venv/bin/python -m pytest \
  tests/v1/attention/test_psp_kv_correctness.py \
  tests/v1/attention/test_psp_kv_mixed_precision.py \
  tests/v1/attention/test_psp_kv_ascend_copy.py -q -p no:cacheprovider
```

## CPU 回归

在本地仓库执行。仅运行这些纯 CPU 检查，使用 `--confcutdir` 排除依赖完整 GPU 测试环境的
根 conftest；没有把 NPU 检查替换为跳过或 CPU 模拟。

```bash
.venv/bin/python -m pytest --confcutdir=tests/v1/attention \
  tests/v1/attention/test_psp_kv_ascend_strategy.py \
  tests/v1/attention/test_psp_kv_shared_metadata.py \
  tests/v1/attention/test_psp_kv_pipeline.py \
  tests/v1/attention/test_psp_kv_ascend_policy.py \
  tests/v1/attention/test_psp_kv_ascend_reference.py \
  tests/v1/attention/test_psp_kv_ascend_packed.py -q
```

## 整模型消融

`STRATEGY_DIR` 包含冻结的启动器、策略、后端与 runtime 文件，每一轮使用新目录。
物理卡 6 的两次消融分别设置 `STRATEGY_DIR=.../serving-scalar-npu6` 和
`.../serving-vector-npu6`，只切换 strategy.py 中的 prefill writer 调用。

```bash
NPU_ID=6 STRATEGY_FORMAT=enhanced \
STRATEGY_DIR=/root/zhang/pspkv-cube-20261027/serving-vector-npu6 \
SOURCE_ROOT=/root/zhang/pspkv-nvidia-strategies-20261026/source-v1 \
NATIVE_LIBRARY_PATH=/root/zhang/pspkv-cube-20261027/libpspkv_vec_v3.so \
STRATEGY_PRECISIONS=e4m3 BENCHMARK_PAIRS=6 \
RUN_TAG=vector6-20261027 BASE_PORT=18950 RUN_INITIAL_BASELINE=0 \
bash /root/zhang/pspkv-cube-20261027/serving-vector-npu6/run_nvidia_strategies.sh
```

最终物理卡 4 的前置 BF16 使用冻结的 `run_baseline_only.sh`，其内容保留同一启动器的函数，
仅执行 `run_baseline bf16-before`，不启动候选。先停止本轮自己的 NPU 检查容器以释放设备。
随后按以下两轮顺序执行，前者跳过重复基线，后者在候选后补跑 BF16。

```bash
NPU_ID=4 STRATEGY_FORMAT=enhanced \
STRATEGY_DIR=/root/zhang/pspkv-cube-20261027/serving-paged-npu4 \
SOURCE_ROOT=/root/zhang/pspkv-nvidia-strategies-20261026/source-v1 \
NATIVE_LIBRARY_PATH=/root/zhang/pspkv-cube-20261027/libpspkv_native_v24.so \
DECODE_PATH=paged STRATEGY_PRECISIONS=e4m3 BENCHMARK_PAIRS=6 \
RUN_TAG=paged4-20261027 BASE_PORT=18974 RUN_BASELINES=0 \
bash /root/zhang/pspkv-cube-20261027/serving-paged-npu4/run_nvidia_strategies.sh

NPU_ID=4 STRATEGY_FORMAT=enhanced \
STRATEGY_DIR=/root/zhang/pspkv-cube-20261027/serving-pipeline-npu4 \
SOURCE_ROOT=/root/zhang/pspkv-nvidia-strategies-20261026/source-v1 \
NATIVE_LIBRARY_PATH=/root/zhang/pspkv-cube-20261027/libpspkv_native_v24.so \
CUBE_LIBRARY_PATH=/root/zhang/pspkv-cube-20261027/libpspkv_cube_v28.so \
DECODE_PATH=cube ATTENTION_SPLITS=4 STRATEGY_PRECISIONS=e4m3 \
BENCHMARK_PAIRS=6 RUN_TAG=pipeline4-20261027 BASE_PORT=18976 RUN_INITIAL_BASELINE=0 \
bash /root/zhang/pspkv-cube-20261027/serving-pipeline-npu4/run_nvidia_strategies.sh
```

额外混合精度/双序列检查：

```bash
/workspace/pspkv-native-build/.venv/bin/python \
  /experiment/checks-source/ascend/check_cube_mixed_npu.py \
  --reference-file /experiment/serving-encode/develop-reference.py \
  --library /experiment/libpspkv_cube_v28.so \
  --output /experiment/cube-v28-mixed.json
```

默认 native 库对应 `build-v24`，最终 Cube 库对应 `build-v28`；活动 native 源码的 SHA
与冻结构建输入逐项一致。完整的服务入口、metadata、ops 与 checker 副本在 `provenance/`。
模型配置文件 SHA 也记录在 manifest。动态库保留在服务器，不进入 Git。

## 静态检查

激活源码执行 pre-commit，冻结实验快照保持原样；新 `.asc` 显式执行 clang-format 检查，
因为通用 C/C++ 文件过滤器未必识别该后缀。原始日志中的 vendor 警告保留，不做文本修饰。
