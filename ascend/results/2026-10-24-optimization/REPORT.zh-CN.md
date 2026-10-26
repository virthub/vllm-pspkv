# 2026-10-24 三格式第一轮性能优化与实测

在华为 Ascend 910B3 上完成代码优化、数值回归和 Qwen2.5-7B 基础版前后对照。
三格式的跨页输入耗时明显下降；短输入没有稳定加速证据；仍未超过原生 BF16。
本轮保持原量化编码规则，因此此前观察到的生成重复问题仍然存在。

## 完整模型结果

同一天、同一张物理 NPU 4，单请求、输入 3 / 135 token、输出固定 32 token。
每配置预热 4 次、计时 24 次（每类 12 次），另有 2 个输出探针。
表中为请求耗时 P50；“跨页”指跨越 128-token 页面边界，不是长上下文压力测试。

| 配置 | 短输入，优化前 | 短输入，优化后 | 跨页，优化前 | 跨页，优化后 | 跨页耗时降低 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Basic | 1.799 s | 1.813 s | 3.302 s | 1.754 s | 46.9% |
| Enhanced | 1.814 s | 1.786 s | 2.661 s | 1.771 s | 33.5% |
| BTKV 参考兼容 | 1.805 s | 1.762 s | 2.630 s | 1.710 s | 35.0% |

Basic 短输入耗时增加约 0.8%，Enhanced/BTKV 分别降低约 1.6%/2.4%；
这些小幅差别不足以宣称稳定收益，短输入应视为基本持平。
本次没有给这些差异计算统计置信区间。

同日原生 BF16 对照为短输入 **1.526 s**、跨页 **1.542 s**。
优化后三格式跨页仍比它慢约 11%～15%，没有宣称已经超越基线。

| 跨页指标 | Basic 前 → 后 | Enhanced 前 → 后 | BTKV 前 → 后 |
| --- | ---: | ---: | ---: |
| TTFT P50 | 278.57 → 108.61 ms | 274.36 → 111.47 ms | 274.52 → 108.57 ms |
| TPOT P50 | 97.45 → 52.83 ms | 76.93 → 53.47 ms | 75.96 → 51.77 ms |
| 整请求 P95 | 3.456 → 2.022 s | 2.962 → 2.326 s | 2.764 → 1.826 s |

每类仅 12 个样本，P95 为小样本插值估计，不代替正式尾延迟与稳定性基准。
TPOT 按客户端首尾非空流式片段的时间与 31 个间隔估算，不是设备内核耗时。
完整数据见 [comparison.csv](comparison.csv)、[comparison.json](comparison.json)、
[优化前汇总](before/summary.json)、[优化后汇总](after/summary.json)。

## 实际修改

### 按缓存行所有权并行写入

新增 `packed_scatter_parallel`，保留 Basic、Enhanced 和 BTKV 的原始压缩布局。
K 和 Basic V 按完整 token/head 行分工；转置 V 按完整维度行分工，保证不同
工作块不写同一个 cache line，FP4 共用字节也由同一工作块处理。
该接口要求唯一的有效 slot，负 slot 跳过；由调用方初始化单位 FP8 缩放，
不再对每个维度反复写相同的缩放值。

独立 NPU 2 对同样 135-token 输入测得，每层：

| 格式 | 原写入器 | 新并行写入器 | 加速比 |
| --- | ---: | ---: | ---: |
| Basic | 5.746 ms | 0.242 ms | 23.7× |
| Enhanced | 5.974 ms | 0.405 ms | 14.8× |
| BTKV | 5.990 ms | 0.412 ms | 14.6× |

该探针用于分离算子成本，不把其倍数直接当作模型加速倍数。
详见 [write-bench.json](raw/write-bench.json)。

### 活跃页解码后调用原生分页注意力

新增 `gather_paged_bf16`，通过当前请求的 block table 定位压缩页，按完整
head 行分工，元数据读取和索引计算移出逐元素循环，尾部填零使用向量操作，
经 UB 批量搬出到临时 BF16 K/V。随后调用 `torch_npu._npu_paged_attention`，
复用其硬件实现，替换原先逐头串行计算的 QK、softmax 和 V 加权路径。

公开 `psp_attention_basic/enhanced/btkv` 接口新增可选
`context_lens_cpu=` 和 `block_size=` 参数。未传 CPU 长度元数据的旧调用保持
原行为。调用方应直接提供已有 CPU 长度，避免每层从 NPU 回读并同步。

新路径限制为 BF16 Q/O、128 或 256 的 head dimension、**128-token 页长**。
256-token 页长的原生分页算子在边界试验中产生了错误数值，已在 Python
入口明确拒绝；不再让这类输入静默进入新路径。旧标量路径仍可按原合同使用。

最终库在头数 28 / KV 头数 4 / dim 128、长度 160 的算子对照中，Basic
注意力约从 3 ms 降到约 0.2 ms，Enhanced/BTKV 从约 2 ms 降到约 0.23 ms。
具体阶段结果见 [checks-v2.json](raw/checks-v2.json)。

### 原生 prefill 与短序列选路

隔离模型适配层的 prefill 改用原生 FIA，保持因果 mask 和 GQA 头数配置。
首轮尝试所有 decode 都使用新分页路径，Basic 短输入出现了轻微退化：
1.799 → 1.841 s。该试验保存在 [pilot](pilot/summary.json)，未取代最终结果。

最终配置使用 `PAGED_MIN_LENGTH=64`：上下文长度小于 64 时保留原标量 decode，
达到 64 后使用临时解码 + 原生分页路径；并行写入和融合 prefill 都保留。
这里是计算路径选择，未启用自动混合精度策略。阈值只针对当前小规模测试
进行调整，尚未证明它适合所有模型和负载。

## 内存和实现边界

- 三格式分配器容量前后均为 **261,504 KV token**；原生 BF16 为 130,688。
- 仍保存原布局的 FP8 持久缓存，没有把全 arena 偷换为 BF16 缓存。
- 135-token 输入的两个 BF16 临时 K/V 缓冲区合计 **512 KiB/调用**。
  这不包括分配器缓存、厂商算子 workspace 和约 4 MiB 的共享因果 mask。
- 实测 EngineCore HBM：Basic/Enhanced 从约 22,067 MB 变为 22,259 MB，
  BTKV 从约 22,073 MB 变为 22,265 MB，均增加约 **192 MB**。
  没有逐项拆分 allocator 缓存和厂商 workspace，不能把整个额外占用称作
  只有 512 KiB，也不能预设较高显存预算下容量/可运行性仍完全不变。
- 临时解码路径增加了压缩数据读取、BF16 写出和后续读取。它是恢复原生
  计算效率的过渡优化，并非直接消费压缩数据的融合注意力，也不是原生 FP8
  矩阵计算；仍有额外搬运，不能预设长上下文会超过 BF16。
- 非单位缩放值会在临时缓冲区中额外舍入到 BF16，不能声称与旧 FP32 累积
  逐位一致。已覆盖的单位缩放及混合页面数值通过容差；其他缩放分布需另测。
- BTKV 仍是参考兼容布局，没有实现或使用 NVIDIA TMA。
- 本轮是显式三格式测试接入，不会自动替换安装环境的通用服务后端。
  TP=1、eager、单序列、关闭 prefix cache、最大模型长度 256 的条件未改变。

## 正确性、质量与回归

- 五组原有 CPU 参考测试：**54 passed，1 skipped**。
- 原有单页/多页/混合精度 NPU 回归：**258 passed**，使用最终优化库。
- 最终扩展矩阵：81 组随机 slot、重复调用、未写区域和混合页面写入检查，
  与原写入器完整字节一致；36 组公开接口注意力对照通过；另有 18 组
  不支持页长的拒绝检查。覆盖 dim 128/256、非连续物理页、尾页及多序列。
- 注意力对照采用 atol=rtol=1e-2，最大绝对误差 **0.0078125**。
  不能称为注意力输出逐位一致。结果见 `raw/checks-*-final.json`。
- 原生 FIA prefill 与 eager 参考也通过 atol=rtol=1e-2；样本最大绝对差
  0.015625，由相对容差允许，详见写入阶段探针。
- 优化后三格式的短句、跨页输出与两个探针彼此一致。与优化前相比，短句和
  两个探针逐字一致，跨页标点/续写不同，不能声称模型输出完全无变化。
- 短句“之一”重复仍出现。本轮没有修改 E4M3 编码器的大值饱和规则，故性能
  改善不是量化质量修复，不构成正式模型质量验收。

初期页长 256 的失败日志保留于 `raw/checks-extra.log`；最终的限制与回归见
[checks-limits.log](raw/checks-limits.log)。
代码和测试工具的相关 pre-commit 检查通过；这不等于全项目存量类型问题已解决。

## 环境、实验顺序与复现

镜像 `quay.io/ascend/vllm-ascend:v0.20.2rc1`，Torch 2.10.0、torch_npu 2.10.0，
Qwen2.5-7B BF16 权重、temperature=0、seed=123、ignore_eos=True。
并发 1、页长 128、显存预算 0.35、前缀缓存关闭、最大模型长度 256。
每个模型容器都单独启动并完成测量，再停止后启动下一配置。

顺序是：同日原生基线 + 优化前三格式、首轮全分页三格式、最终选路三格式。
正式前后表包含 7 配置、210 次生成、168 次计时；另有首轮探索 90 次生成、
72 次计时，其结果单独保留。模型测试全部使用 NPU 4；部分小规模回归和
阶段探针使用 NPU 2 并限制 CPU 线程，没有两个模型服务同时争用 NPU 4。
服务器其他卡仍有既有服务，未宣称整机独占或统计显著性。

原始库保留为：
`/root/zhang/pspkv-native-build/libpspkv_ascend_native.so`，SHA256
`72f17c1718609e04e3fad08404d832ab7fb2553fc42520fc31b06f1de1428ed1`。
优化库：`/root/zhang/pspkv-opt-20261024/build/libpspkv_ascend_native_v2.so`，SHA256
`6431a9e4782eb78edf21487d0ff455e04bfab0d68414989fea531b08f5250779`。
服务器原项目未覆盖；优化源码/包装器和实际编译快照独立保留。

服务器复现时选择新的 RUN_ID，避免容器同名：

```bash
NPU_ID=4 \
COMPARISON_DIR=/root/zhang/pspkv-opt-20261024/final \
RUN_ID=manual-opt-check \
MODEL=Qwen/Qwen2.5-7B \
COMPARE_OPTIMIZED=1 PAGED_MIN_LENGTH=64 \
NATIVE_LIBRARY=/root/zhang/pspkv-opt-20261024/build/libpspkv_ascend_native_v2.so \
OPS_OVERLAY=/root/zhang/pspkv-opt-20261024/vllm/v1/attention/backends/psp_kv_ascend_ops.py \
bash /root/zhang/pspkv-opt-20261024/final/run_format_comparison.sh
```

实现位置：[原生算子](../../../ascend/native/pspkv_fp8.asc)、
[Python 三格式接口](../../../vllm/v1/attention/backends/psp_kv_ascend_ops.py)、
[模型适配层](../../../ascend/format_compare_runtime.py)、
[回归脚本](../../../ascend/check_optimized_formats_npu.py)。
