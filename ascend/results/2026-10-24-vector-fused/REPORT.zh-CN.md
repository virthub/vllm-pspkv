# UB 向量解码与流式融合注意力：逐步实现和性能分析

本轮完成两项实现：将 FP8 解码从逐元素标量循环改为 UB 内向量查表；
再实现逐 tile 解码、在线 softmax 和 PV 累加的融合注意力。
两条路径均保留，用于按负载选择和前后对照。

融合内核目前使用 AIV SIMD，没有在该内核内调用 Cube。
长序列测得原生矩阵注意力组合更快，因此默认自动选择只在已测短上下文形状下融合。
不同层的 K/V 本身不同，仍需要解码本层的数据；本轮消除的是独立全量解码阶段和
完整 BF16 临时 K/V 写回，不是把一层的 K/V 跨层复用。

## 测试环境和版本

- Ascend 910B3 64 GiB，CANN 9.0.0，镜像 `quay.io/ascend/vllm-ascend:v0.20.2rc1`。
- 阶段 1 算子与全部整模型对照使用物理 NPU 4。
  融合迭代、最终数值回归和算子采样使用另一张空闲物理 NPU 2。
- 头数为 Qwen2.5-7B 的 28 Q heads / 4 KV heads / dim 128，页长 128。
  数值检查额外覆盖 dim 256、多序列、非单位缩放与混合页面。
- 算子计时为每阶段预热 3 次，5 轮 × 每轮 20 次，取轮均值的中位数。
  主机计时包括提交和等待；Event 计时也可能包含主机提交间隙。
  纯设备内核时间另外使用 profiler 获取，每个采样入口重复 3 次。
- 阶段 1 库 SHA256：
  `1ebebb718f8eaf9ca666aa27a5adf528c0bbf8efcd7026494ad9383476eb6538`。
- 最终库 SHA256：
  `8980346ea75b271574f5e01ea03c5c641b202b1fb3855c86c1627162df35c358`。

原始标量解码函数保留在同一动态库内，算子对照不会依赖历史机器上的计时。
本轮没有修改 FP8 编码、精度选择策略或原有饱和规则；BTKV 仍为原始参考布局的兼容实现。

## 第一步：按 tile 向量解码

新增 `gather_paged_bf16_vec`，每次搬运 32 个 token 的一个 KV head：

1. 从 GM 按块搬入 FP8 字节，转置 V 使用带步长的块搬运。
2. E4M3/E5M2 表和转置索引在一个工作块内只搬入 UB 一次。
3. 在 UB 内执行向量 Cast、索引计算、Gather 查表、缩放、BF16 转换。
4. 按输出的 head 间隔批量写出；尾部明确填零。

热路径不再逐元素调用标量 `ReadPacked`、位解码和 `SetValue`。
混合页面中的 BF16/FP4 保留正确性参考分支，没有宣称这些分支也已向量化。
910B 的 Cast 支持范围见 [CANN Cast 文档的 A2 表](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0073.html)，
本实现通过向量查表处理 FP8，未依赖较新架构的原生 FP8 转换指令。

阶段 1 的独立解码调用，包含框架提交、分配和同步等待：

| 长度 | 格式 | 原标量解码 | UB 向量解码 |
| --- | --- | ---: | ---: |
| 160 | Basic | 0.171 ms | 0.047 ms |
| 160 | Enhanced | 0.224 ms | 0.045 ms |
| 160 | BTKV | 0.214 ms | 0.044 ms |
| 2048 | Basic | 2.011 ms | 0.051 ms |
| 2048 | Enhanced | 2.468 ms | 0.045 ms |
| 2048 | BTKV | 2.459 ms | 0.052 ms |

Basic 的纯设备内核采样：

| 长度 | 原解码内核 | 向量解码内核 | 内核耗时加速比 |
| --- | ---: | ---: | ---: |
| 32 | 47.22 μs | 8.53 μs | 5.5× |
| 160 | 167.83 μs | 9.79 μs | 17.1× |
| 2048 | 1990.57 μs | 23.43 μs | 85.0× |

长度 160 的内核标量流水线比例约从 93.9% 降到 42.1%；长度 2048 约从
97.2% 降到 22.6%，向量流水线比例约为 44.9%。这些是内核内部流水线计数，
不是整卡利用率。相比向量解码设备时间，框架调用成本已经不能忽略。
后面仍要执行原生分页注意力，并保留 BF16 临时缓存，因此第一步尚未消除中间物化。

数据见 [stage1.json](results/stage1.json)、[PMU 聚合](pmu-summary.json)。

## 第二步：解码与注意力流式融合

新增 `tile_fused_attention`：一个工作块负责一个 query head，循环消费当前序列的 tile。
query、查找表、当前 tile、概率与输出累加器保留在 UB，最大值和归一化状态在片上更新。
每个 tile 进行 QK 向量归约、稳定的在线 softmax 和 PV 累加，最后只写输出。
没有完整 BF16 K/V 临时缓存，也没有全局 attention score / probability 矩阵。

为减少 UB 占用，查表索引使用完的内存会复用于 QK 乘积与概率广播索引。
dim 128 时解码器约占 86 KiB UB，融合额外工作缓冲约 2 KiB；
dim 256 的融合缓冲约为 173.4 KiB。该估计按显式 `InitBuffer` 大小相加，
不是内存分析器测得的硬件资源占用，也不意味着实现了双缓冲。

第一版融合已经正确，但长上下文比原生注意力组合慢。按每个 query head 的插桩计时，
2048 token 的 Enhanced 主要成本约为解码 188 μs、PV 115 μs、QK 34 μs。
据此继续做了两个调整：

- Enhanced/BTKV 的完整 V tile 保持 `[dimension, token]` 布局，直接广播概率、
  沿 token 维归约，省掉一次转置和概率展开。
- 单位缩放时，解码的 FP8 值可由 BF16 精确表示，省去融合内部多余的 BF16 往返转换。
  非单位缩放仍保留相应舍入，以匹配物化后原生注意力的数值。

Enhanced/BTKV 在 2048 token 的融合调用从约 0.345 ms 降至约 0.216 ms。
最终各格式的直接融合调用如下，均为主机计时，不能直接当作设备内核执行时间：

| 上下文长度 | Basic | Enhanced | BTKV |
| --- | ---: | ---: | ---: |
| 32 | 0.034 ms | 0.035 ms | 0.036 ms |
| 160 | 0.034 ms | 0.034 ms | 0.034 ms |
| 512 | 0.075 ms | 0.063 ms | 0.063 ms |
| 2048 | 0.269 ms | 0.216 ms | 0.216 ms |

Basic 的最终纯设备融合内核约为长度 32：11.53 μs，160：27.89 μs，2048：264.17 μs。
长度 160 的原生注意力组合两个设备内核合计约 26 μs，因此融合不意味着设备运算
必然更快；它还减少了算子提交、临时张量和全局中间数据。
长度 2048 时两个组合内核约为 53 μs，明显优于当前 Basic 融合内核。
本轮不把较长序列的融合路径设为默认。

数据见 [第一版融合](results/stage2.json)、[V 布局改进](results/stage2b.json)、
[最终融合与自动选路](results/final.json)、[展开 CSV](operators.csv)。

## 融合后哪些阶段还在耗时

最终插桩版本在 2048 token 下，每个 query head 的平均阶段时间为：

| 格式 | 解码 K/V | QK | 在线 softmax | PV | 头内总时间 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Basic | 108.88 μs | 33.36 μs | 11.42 μs | 115.64 μs | 271.77 μs |
| Enhanced | 131.13 μs | 33.44 μs | 11.40 μs | 42.52 μs | 220.95 μs |
| BTKV | 130.88 μs | 33.46 μs | 11.41 μs | 42.50 μs | 220.70 μs |

阶段时间来自 `profile_mode=1`，使用
[910B 的 50 MHz 系统计数器](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0282.html)。
插桩有额外同步；头内总时间不包括此前统一加载的表和转置索引。
这些列不能作为未插桩内核的精确阶段占比，更不能直接乘 28 推导整模型延迟。

当前长序列融合的主要限制已经从逐维标量位解码变成了：

- 每个 query head 独立遍历 tile。Qwen 的 GQA 中 7 个 Q head 共享一个 KV head，
  当前实现仍可能重复搬运和解码同一份 K/V；没有做到组内共享一次解码。
- Basic 的 V 仍要做行间归约，成本高于可以直接利用转置布局的 Enhanced/BTKV。
- tile 之间保留了同步和在线归一化依赖，没有实现 AIC/AIV 联合流水线和双缓冲。

例如 Basic 长度 2048 的融合内核向量流水线比例约为 84.5%，仍比原生矩阵路径慢。
这说明向量单元忙碌并不等于已经达到最快算法或最高整卡效率。
更长上下文的下一步应优先考虑 GQA 组内复用、矩阵化 QK/PV 与 AIC/AIV 协作。

## 自动选路和使用方式

默认 `PSPKV_ASCEND_ATTENTION_IMPL=auto`：BF16、单序列、dim 128、128-token 页、
页表最多两页时使用融合。此阈值覆盖本轮 Qwen2.5-7B 的短输入和跨页输入。
较长页表或其他形状，若提供已有 CPU 长度，则走向量解码加原生分页注意力；
不满足相关条件时保留原路径。页表形状判断避免了为选路从设备回读长度。

也可在导入模块前指定 `vector`、`fused`、`gather`，分别用于阶段 1、强制融合和
上一轮标量解码对照。具体参数见 [Ascend README](../../README.md)。
`auto` 是当前测量范围内的保守规则，没有证明它对所有模型、并发和长度都是最优。

## 整模型对照

使用 Qwen2.5-7B 基础版，BF16 权重，TP 1，单请求，eager，关闭前缀缓存，
128-token 页，`max_model_len=256`，显存预算比例 0.35，温度 0、seed 123。
输入分别为 3 / 135 token，固定生成 32 token。每类输入预热 4 次，再计时 12 次。
下表采用请求耗时 P50；跨页测试不是长上下文压力测试。

| 配置 | 短输入 | 跨页输入 |
| --- | ---: | ---: |
| 本轮原生 BF16 | 1.603 s | 1.577 s |
| 第一步 Basic，向量解码 | 1.742 s | 1.740 s |
| 第一步 Enhanced，向量解码 | 1.799 s | 1.796 s |
| 第一步 BTKV，向量解码 | 1.787 s | 1.718 s |
| 第二步 Basic，自动选路使用融合 | 1.729 s | 1.670 s |
| 第二步 Enhanced，自动选路使用融合 | 1.658 s | 1.658 s |
| 第二步 BTKV，自动选路使用融合 | 1.626 s | 1.663 s |

融合相对第一步，跨页请求耗时分别降低约 **4.0%、7.7%、3.2%**；
短输入分别降低约 **0.7%、7.8%、9.0%**。
Basic 短输入的 0.7% 差别不足以宣称稳定收益。每类仅 12 个样本，未计算置信区间，
其他差异也以本轮观察值报告，不作普遍或稳定加速保证。
优化后跨页仍比本轮 BF16 慢约 **5.1%～5.9%**；短输入约慢 **1.5%～7.8%**。
三格式跨页 TPOT 从阶段 1 的 51.78～54.35 ms 降至 49.81～50.33 ms，
本轮 BF16 对照为 47.57 ms。

三格式的最终短句、跨页输出及两个算术/常识探针彼此一致。
这不是与原始 BF16 模型的质量等价证明；对照的是保留原量化编码后的实现。
此前短输入中的重复生成仍然出现，本轮性能改动没有修复该质量问题。
详细请求、P95、输出和资源数据见
[阶段 2 汇总](summary-stage2/summary.json)、[逐阶段对照](model-comparison.csv)。

第一步内核加速没有等比例变为整请求加速。请求仍包括模型权重、MLP、元数据构造、
K/V 拼接、压缩写入以及框架提交。旧设备任务也可能与主机提交重叠，不能把
独立内核节省的时间直接加到整模型上。没有完整时间线时，不给这些原因分配精确比例。

本轮 BF16 的分配器容量为 130,688 token，三格式为 261,504，仍有近两倍 KV 容量收益。
容量、进程显存和请求延迟分别统计；未以 token 容量代替速度指标。

## 整模型采样确认的新瓶颈

Basic 的正常延迟测试结束后，额外采集两个请求：输入 135 / 3 token，各输出 4 token。
采样请求没有计入延迟样本。时间线中共 3560 个设备 kernel，其中：

- 融合注意力恰好 168 次，即 2 请求 × 3 个 decode 步 × 28 层。
  旧标量 QK 和完整页解码调用均为 0，确认模型实际执行了新融合路径。
- 融合注意力设备执行时间合计约 3.21 ms，平均每层调用约 19.10 μs。
- 两类主要矩阵乘法设备时间合计约 106.97 ms，占本次设备 kernel 时间总和
  约 82.9%；融合注意力约占 2.5%。这些比例不是整请求延迟占比。
- 主机侧记录到 168 次 `aclrtMemcpy`，合计约 5.17 ms，以及 168 次
  `aclrtSynchronizeStream`，合计约 1.95 ms。次数与逐层创建 NPU 长度张量的
  代码路径一致，是明确值得消除的同步点；本轮没有把全部 API 调用逐一归因到源代码。
- 在同一个 decode 步内，相邻融合层调用之间取 162 个窗口，排除 token 边界和
  请求切换，窗口总长约 308.62 ms，已记录设备 kernel 的区间并集约为 74.03 ms。
  其余时间不能简单称为 CPU 计算，也包括提交、同步、拷贝和运行时等待。

采样会增加主机开销，因此上述窗口比例不能直接当作正常运行的整卡利用率。
它与每层重复的长度张量构造、K/V 拼接和大量 eager 小算子相符，解释了为什么
单独解码内核节省很多时间，而端到端收益较小。融合后还存在主机路径和模型
矩阵乘法成本，继续只优化 FP8 位解码已不是足够的端到端优化策略。

下一步优先复用同一 decode 步跨层不变的元数据、消除重复同步拷贝，
再评估图执行及适配器的小操作合并。长序列融合则需要 GQA 组内共享与矩阵化。
数据见 [整模型采样聚合](model-profile-summary.json)；
原时间线位于 `serving-stage2/model-profiler-basic`。
Basic 的采样后进程显存可能包含 profiler 缓冲，不能用它与未采样配置直接比较总显存。

## 数值检查和边界

- 最终 126 组注意力数值用例：三格式、dim 128/256、E4M3/E5M2/混合页面、
  非单位缩放、乱序物理页、多序列及 tile/page 边界。输出对照容差 `atol=rtol=0.01`，
  测得融合最大绝对差为 0.0078125。
- 12 组全编码检查，每组覆盖 256 个 FP8 字节值。非 NaN 结果的 BF16 位模式与旧解码一致，
  包括符号零和无穷；NaN 位置一致，不要求 NaN payload 相同。
- 原有正确性和混合精度 NPU 测试 258 项通过；CPU 参考、布局与策略测试 54 项通过。
- 新解码保留混合 BF16/FP4 的标量分支。本轮没有宣称解决 FP4 质量、
  自动混合精度策略、更多模型的质量评测或 Blackwell 原生 BTKV 特性。

测试曾因开发容器占有物理 NPU 4，导致新容器初始化返回 `Resource_Busy`。
停止开发容器后相同设备探针恢复，随后整模型串行运行。
启动入口额外先初始化 NPU，再创建 spawn worker，所有本轮整模型配置使用相同入口。
故障日志单独保留，不计入性能样本；没有停止服务器上的其他业务容器。

## 审阅和复现文件

- [新 kernel](../../native/pspkv_tiled.h)、[注册和校验](../../native/pspkv_fp8.asc)。
- [Python 调度](../../../vllm/v1/attention/backends/psp_kv_ascend_ops.py)。
- [数值与算子采样脚本](../../check_tiled_attention_npu.py)。
- [模型对照脚本](../../run_format_comparison.sh)、[单卡启动入口](../../serve_initialized_npu.py)。
- [额外模型采样客户端](../../profile_model_client.py)：采样请求与延迟样本分开。
- [原有 NPU 回归日志](results/legacy-regression.log)、[最终验证日志](results/final-validation.log)、
  [CPU 日志](cpu-tests.txt)、[pre-commit 日志](pre-commit.txt)。
- [阶段 1 模型汇总](summary-stage1/summary.json)、[阶段 1 原始模型记录](serving-stage1-driver)。
- [阶段 2 模型汇总](summary-stage2/summary.json)、[阶段 2 原始模型记录](serving-stage2)。
- [本轮代码补丁](code-from-v2.patch)、[源码与库校验和](manifest.json)、
  [结束时设备状态及原始库校验和](remote-final-state.txt)。

服务器的独立工作目录为 `/root/zhang/pspkv-vector-fused-20261024`。
库由 uv 管理的 `/workspace/pspkv-native-build/.venv/bin/python` 构建和测试，
未替换原始 `/root/zhang/pspkv-native-build` 下的库或服务器原始源码目录。

在物理 NPU 4 空闲、开发容器已停止的前提下，按本轮配置重复最终模型对照：

```bash
PSPKV_REVIEW=/root/zhang/pspkv-vector-fused-20261024/final-review
PSPKV_REPRO=/root/zhang/pspkv-vector-fused-20261024/reproduce
mkdir -p "$PSPKV_REPRO"
cp "$PSPKV_REVIEW"/ascend/{run_format_comparison.sh,serve_initialized_npu.py,format_compare_runtime.py,compare_formats_client.py,profile_model_client.py,summarize_format_comparison.py} "$PSPKV_REPRO/"
NPU_ID=4 DRIVER_CONTAINER=pspkv-vector-fused-dev-20261024 \
  COMPARISON_DIR="$PSPKV_REPRO" \
  NATIVE_LIBRARY=/root/zhang/pspkv-vector-fused-20261024/build/libpspkv_final.so \
  OPS_OVERLAY="$PSPKV_REVIEW/vllm/v1/attention/backends/psp_kv_ascend_ops.py" \
  MODEL=Qwen/Qwen2.5-7B INCLUDE_STOCK=1 \
  RUN_ID="tiled-reproduce-$(date +%Y%m%d%H%M%S)" HOST_PORT=18134 \
  COMPARE_OPTIMIZED=1 PAGED_MIN_LENGTH=64 ATTENTION_IMPL=auto \
  CLIENT_WARMUPS=4 CLIENT_ROUNDS=3 CLIENT_REQUESTS=4 \
  bash "$PSPKV_REPRO/run_format_comparison.sh"
```

阶段 1 可使用 `ATTENTION_IMPL=vector` 和 `build/libpspkv_stage1.so` 重复。
启用 `PROFILE_MODEL=1` 会在 Basic 延迟测试后额外采样两个请求。
以上只适用于本轮单设备容器工作流，不替代正式部署配置。
