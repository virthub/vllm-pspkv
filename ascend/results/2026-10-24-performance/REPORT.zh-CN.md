# 昇腾 PSP-KV 性能瓶颈分析

2026-10-24 在 worker03、物理 NPU 4 / 容器逻辑 NPU 0（Ascend 910B3）
完成源代码审计、阶段微基准与 CPU/NPU profiler 采样。

结论：当前路径的确没有合理使用昇腾计算资源。已经实现了压缩存储和参考
计算，但本次 Basic/Enhanced/BTKV 走的是大量标量读写与串行乘加的实现，
不能按“硬件优化已完成”理解。首 token 的主要可疑瓶颈是单块写缓存，
持续解码的明确瓶颈是 QK 和 V 加权。这些问题不是增加几个 launch block
或只融合 softmax 就能解决的。

## 方法与证据范围

- 原生库、源代码与之前 7B 三格式测试相同；未修改生产实现、未重新编译库。
  关键哈希见 [source-sha256.txt](raw/source-sha256.txt)。
- 微基准使用 Qwen2.5-7B 的 28 Q 头 / 4 KV 头 / head_dim 128，块长 128。
  输入为有界随机 BF16，FP8 E4M3 缓存、单位缩放，三格式逻辑输入一致。
- 长度 32/160/512/2048、batch=1，另测 batch=4、长度 160。
  阶段计时各预热 3 次、3 轮、每轮 10 次，报告轮均值的中位数。
- 同时记录主机 wall time 与 NPU Event 时间。Event 包含流内调度，不等同于
  单个内核的纯执行时间；profiler 另提供设备任务时间和流水线计数器。
- 原生 BF16 对照直接使用插件现有 `_npu_paged_attention` 接口，按其要求
  传 CPU 长度张量；不是自写 BF16 标量参考。其结果与 CPU BF16 参考校验。
- 压缩注意力结果与 CPU 压缩参考校验，使用 atol=rtol=1e-2；本轮最大绝对
  误差为原始 JSON 记录值。该数值校验不代表模型量化质量合格。
- profiler 在 batch=1、长度 160 下，对原生 BF16、Basic 各阶段和既有
  staged 控制采样 3 次，启用 Level1 / PipeUtilization。
- 本轮没有对完整 7B 服务重新采集 timeline。原生服务可按条件选择分页注意力
  或 FIA；微基准的算子倍数不能直接当作端到端模型加速倍数。

原始计时见 [stages.json](raw/stages.json)、[attention.csv](attention.csv)，
硬件计数汇总见 [pipeline-metrics.json](pipeline-metrics.json)。
`raw/profiler/` 内保存 kernel_details.csv、operator_details.csv 和 timeline。

## 解码：QK 与 V 加权是主要瓶颈

单层、单请求、长度 160，wall time：

| 阶段 | Basic | Enhanced | BTKV |
| --- | ---: | ---: | ---: |
| 写入该序列一个新 token | 0.055 ms | 0.061 ms | 0.060 ms |
| QK | 0.830 ms | 0.999 ms | 1.003 ms |
| softmax | 0.038 ms | 0.038 ms | 0.039 ms |
| V 加权 | 2.278 ms | 1.070 ms | 1.074 ms |
| FP32 → BF16 与拷贝 | 0.077 ms | 0.085 ms | 0.083 ms |
| 连续调用的完整注意力 | 3.102 ms | 2.066 ms | 2.072 ms |

原生 BF16 分页注意力对照为 **0.065 ms**。Basic / Enhanced / BTKV 的
完整注意力分别约为它的 47.6 / 31.7 / 31.8 倍。
各阶段分开测量会改变调度和缓存状态，分项之和不要求精确等于完整路径。

### 硬件计数器确认主要在标量流水线上执行

| 采样任务 | Block Num | AIV Scalar Ratio | AIV Vec Ratio | Cube utilization |
| --- | ---: | ---: | ---: | ---: |
| PackedScatter | 1 | 98.93% | 工具显示 0 | 0 |
| PackedScores | 28 | 99.8% | 工具显示 0 | 0 |
| PackedWeightedValue | 28 | 99.9% | 工具显示 0 | 0 |
| PackedScoresVec | 28 | 99.4% | 工具显示 0 | 0 |
| PackedWeightedValueVec | 28 | 99.6% | 工具显示 0 | 0 |

这是**采样算子内部**的流水线比例，不是整卡利用率。比例可能重叠，不能相加
当作资源分配。零值受计数器显示精度限制。该结论也不代表模型其他层没用 Cube：
线性层仍可使用矩阵计算单元。

原生分页对照的任务为 `PagedAttentionMaskNdKernel` / `MIX_AIC`，
Block Num=14、Mix Block Num=28，并有非零矩阵计算计数。
自定义 QK/V 是 `AI_VECTOR_CORE`，但内部主要执行标量代码；
`__vector__` 修饰函数不等于函数体已经采用 SIMD 向量乘加。

华为文档明确说明 GlobalTensor/LocalTensor 的 GetValue、SetValue 属于
标量读写，通过 PIPE_S 执行；连续访问可命中 DataCache，并非每次都访问 HBM。
依据：[Scalar Unit 读写原理](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_10_00031.html)。

### 代码中的具体问题

1. `PspkvPackedScores` 按请求×Q头分工，头内逐 token、逐维度循环。
   长度方向没有分块并行；每个点积的 128 维通过普通 float 乘加累积。
2. `PspkvPackedWeightedValue` 同样按请求×Q头分工，再逐维度、逐 token
   累积。Basic 的 V 是 token-major，这个循环沿 token 读取同一维，步长
   是 head_dim；Enhanced 转置 V 后改善该访问，所以 V 阶段变快。
   这与布局收益一致，但本轮没有单独采集 cache miss / HBM 流量来完全归因。
3. Query、格式、块偏移、缩放等数据在内层反复读取；没有充分用 UB 缓存
   并复用 Q、整页 K/V 和量化元数据，也没有针对 GQA 共享 K/V 进行复用。
   28 个 Q 头共享 4 个 KV 头，即每个 KV 头对应 7 个 Q 头；重复计算/
   读取不等于实际 HBM 流量必然翻 7 倍，因为存在缓存。
4. host wrapper 最多启动 min(batch×heads,40) 个工作块。
   batch=1 时为 28；每块仍串行扫过整个长度。不能把“28/40”直接当成整卡
   利用率，更不能把 block 上限改大就认为标量乘加会自动变成向量计算。
5. `_attention` 分开调用 QK、torch.softmax、V 加权和输出转换，分配 FP32
   中间张量，增加启动与中间读写。但本例 softmax+转换仅约 0.12 ms；
   主成本约 3 ms 的 QK/V 不改，单独减少这点开销不足以追上原生算子。

源码入口：
[原生内核](../../../ascend/native/pspkv_fp8.asc)、
[Python 算子包装](../../../vllm/v1/attention/backends/psp_kv_ascend_ops.py)。

## 现有 Vec/Fused 还不能当成完整高性能实现

这次三格式的入口调用 `packed_scores` / `packed_weighted_value`，
没有调用 `packed_scores_vec` / `packed_weighted_value_vec`。
额外测试既有 scaled-basic staged 路径，在单位缩放下与 Basic 输入兼容：
长度 160 时为 **1.728 ms**，比 Basic 快约 44%，但仍约是原生对照的 26.5 倍。

原因是 staged 实现虽然用 DataCopy 搬到 UB，并通过查表解码，但 QK/V 的
主体仍用 LocalTensor.GetValue 和 float 循环；profiler 也显示约 99% 的
标量比例。其名字里的 Vec 不代表已采用完整向量乘加/归约。
另一个 fused 路径把部分 score/softmax 留在 UB，但源码明确保留标量 QK/V，
没有在本轮切换模型使用它，也不能预设切过去就会达到原生性能。

BTKV 当前与 Enhanced 共用计算与转置 V 布局，描述符不参与硬件加速，
不应期待它在当前实现中自动获得独立的速度优势。

## 首 token：单块写缓存尤其严重

另在独立容器串行测量真实测试输入长度 3/135，保持相同头数和页长。
前缀结果使用 `prefill-isolated.json`，主阶段数据使用首次隔离测试的
`raw/stages.json` 快照；未使用存在其他测试任务重叠的测量数据。

| 135-token 前缀，每层 | wall time |
| --- | ---: |
| 当前 Basic packed_scatter | 5.738 ms |
| 当前 Enhanced packed_scatter | 5.965 ms |
| 当前 BTKV packed_scatter | 5.959 ms |
| 既有 token-major 并行 FP8 写入器控制 | 0.102 ms |
| 当前 eager FP32 prefill 注意力 | 0.368 ms |
| 原生 BF16 FIA prefill 对照 | 0.124 ms |

并行写入器的布局和元数据处理也不同，所以 56 倍差距不能全部归因于核心数，
它不能原样替换三格式写入器；但它说明 FP8 编码本身不必耗时数毫秒。
当前 packed_scatter 对本轮固定 FP8 路径只启动 1 个 block；这个 block
循环全部 token×KV头×维度，还在维度循环中反复把相同缩放写为 1。
prefill 的 135 个 token 在此串行处理，成本随长度增长。

Basic 5.738 ms ×28 层约 160.7 ms，而前次完整模型跨页 TTFT 相对基线
增加约 178.7 ms；Enhanced/BTKV 对应的写入估算也约为 167 ms。
仅 eager FP32 prefill 与 FIA 的差值乘 28 层约为 6.8 ms。
**这是写缓存为首 token 主要瓶颈的强证据**，但不是完整模型 timeline 的
精确归因；算子调度、模型输入分布与层间重叠可能改变实际贡献。

原始记录见 [prefill-isolated.json](raw/prefill-isolated.json)。
FIA/eager 输出通过 atol=rtol=1e-2 校验，最大绝对差 0.015625，
不代表逐位相同。这里只比较实现开销，不评估真实模型量化质量。

## 场景和算法收益应分开判断

前次完整模型测试固定 E4M3，没有运行自动混合精度选择。因此“精度选择策略
太慢”不是这张表的原因。当前收益是 FP8 减少缓存字节数；并没有把注意力
计算变成原生 FP8 矩阵运算，而是在读取时解码成 float 再标量乘加。

batch=1、长度只有 3/135，缓存没有接近容量上限，容量翻倍不会自动加快
一次请求。模型权重、MLP 等成本也没有因 KV 缓存变小而减半。
不过，当前实现即使长度增加到 2048，Basic/Enhanced 仍约为 47.54/25.91 ms，
原生分页对照约 0.070 ms。**只改成长上下文，不能解决当前标量实现的问题。**
多并发能增加任务数，但现有 block 上限和内核结构仍限制吞吐；原服务路径的
单序列等限制也需先处理，不能把提升并发当作立即可用的修复。

## 建议的优化顺序与验收

1. 先保证量化质量：修复/明确 E4M3 大值饱和边界与缩放策略，重测 7B，
   避免用质量退化后的输出衡量性能收益。
2. 优先改写 packed_scatter：按 token×head×连续字节块分工，批量搬运和
   编码，避免重复写单位缩放。分块需处理 64-byte cache line 一致性、
   FP4 两元素共享字节等约束；不能只把 launch 的 1 改成 40。
3. 用实际 SIMD 的反量化、乘加、归约替换 QK/V 的逐元素循环，Q 和页面数据
   分块放入 UB，减少每元素格式分支/元数据读取，并复用 GQA 的 K/V。
   单 token GEMV 可先走高效向量路径，适当 batch/形状再使用 Cube/Matmul，
   不要求每种小形状都强行用矩阵单元。
4. 在上述基础上融合 QK/在线 softmax/V，减少中间 GM 张量；必要时沿 KV
   长度切分，增加并行度并做数值稳定归并，而不是每个头始终只分配一块。
5. prefill 尽量保留原生融合注意力，压缩写入单独优化；预分配元数据、复用
   缓冲区，减少 stack、类型转换、CPU→NPU 长度构造和 Python 调用。
6. 最后在质量合格前提下重跑同模型、同批量、同上下文的端到端测试，再扩大
   上下文/并发，报告吞吐、TTFT、TPOT、P95 与容量收益。进一步对完整服务
   采集 timeline，验证这些单算子改进是否真正降低端到端延迟。

本轮只分析和采样，没有修改生产内核。两个测试容器已退出，物理 NPU 4
无测试进程，见 [final-containers.txt](raw/final-containers.txt) 和
[npu-after.log](raw/npu-after.log)。工具检查见 [pre-commit.txt](pre-commit.txt)。
服务器测试目录：`/root/zhang/pspkv-perf-analysis-20261024`。
