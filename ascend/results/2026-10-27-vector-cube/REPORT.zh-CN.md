# FP8 向量编码与 Vector/Cube 注意力优化

## 结论与口径

已完成并验证长 prefill 的 BF16→E4M3/E5M2 向量编码。8192 token 的单层
E4M3 写入从 7.023 ms 降到 0.139 ms，约 50.6 倍；同卡整模型请求从
1.3634 s 降到 1.1856 s，耗时降低 13.0%。这些是不同层次的指标，不能把
单个 kernel 的 50 倍加速写成整模型加速。

新编码的长输入首 token 延迟接近 BF16，但生成阶段仍慢于 BF16。
Vector/Cube 联合注意力、GQA 复用、V 双缓冲和异步 P×V 期间的 K 预取已实现并通过检查。
最终同卡 7B 实测：8192-token 请求由分阶段 FP8 的 1.1840 s 降到 1.1487 s，降低约 3.0%；
原生 BF16 前后共 12 次的中位数为 1.0816 s，融合 FP8 仍慢约 6.2%。
不能将本轮结果表述为量化推理已经超过原生 BF16，也不能把数值误差检查当作模型质量评测。

## 环境与公平性

- Ascend 910B3；Qwen/Qwen2.5-7B 基础版，BF16 权重，TP=1、并发=1。
- 镜像 `quay.io/ascend/vllm-ascend:v0.20.2rc1`，CANN 9.0.0，编译目标 `dav-2201`。
- 输入严格为 3、135、8192 token；固定输出 32 token，temperature=0，seed=123，忽略 EOS。
- block size=128，max model length/max batched tokens=8320，memory utilization=0.35。
- 关闭 prefix caching 和 chunked prefill。候选保留 mode=0 的 RoPE 适配器及完整
  decode ACL graph；原生 BF16 使用 mode=3 的原生完整 decode graph。
- 每个长度独立预热两次、采样六次，随机交替输入长度；表中均为请求端中位数。
- 先在物理卡 7 完成新编码与 BF16 检查，再在空闲物理卡 6 完成消融：旧编码→BF16→新编码→BF16。
  主结论只使用卡 6 的同卡结果。卡 7 随后被其他任务使用，补测的 ResourceBusy 结果不计入有效数据。
- 消融两边使用同一个 native v3 库、相同 decode、相同策略和 graph 配置，仅切换 prefill 写入入口。
  这样不会把 Cube、不同精度、不同卡或不同编译路径的差别算进编码优化。
- Basic、Enhanced、BTKV 在 Ascend 的主策略入口共用软件实现；本轮消融使用 Enhanced E4M3
  作为公共实现的代表，不把一次结果复制成三套独立硬件实测。

## 第一步：批量编码与写入

### 改了什么

`ascend/native/pspkv_vector_encode.h` 新增 `PspkvVectorEncoder` 与
`PspkvPrefillScatterVec`，通过 `strategy_scatter_prefill_vec` 接入策略后端。

1. 一个任务处理一个 K 或 V head 的 16×128 tile，最多使用 40 个 Vector block。
2. 按 projection 输出的 token/head stride 批量 DataCopy 到 UB，不先 stack/cat 整段 K/V。
3. 以向量位移、加减、Cast、Compare/Select 实现 E4M3/E5M2：保留 RNE ties、渐进下溢、
   饱和、符号及 NaN 编码。没有把不支持的硬件 FP8 cast 当作已实现功能。
4. 编码所需中间量留在当前 tile 的 UB，向量计算后直接写真实字节缓存；lossless 采用批量 DMA。
5. 页/head 的 scale 和 offset 每个 tile 读取，替代旧的逐元素读取与编码。

保留原有跨层元数据复用、设备长度、直接 strided 写入、decode RoPE/scatter 融合、
工作区同流复用和 ACL graph。没有另起一条漏掉旧优化的对照路径。

适用范围为当前单请求、从新序列起始的 unchunked prefill，head dimension=128。
FP4 与幅值自适应仍走其原有编码入口；不能把 FP8 编码加速声称为四种精度全部加速。
本轮没有实现 prefill RoPE/量化/scatter 三合一、chunked prefill 重叠或固定 prefill graph。

### 单层编码计时

NPU event，3 次预热、20 次循环；输入为真实 projection stride、随机物理页和非单位 scale。
完整原始数据见 [最终编码检查](ablation-npu6/vector-prefill-final.json)。

| 输入 token | 精度 | 旧写入 ms | 向量写入 ms | 说明 |
| ---: | --- | ---: | ---: | --- |
| 8192 | lossless BF16 | 0.6000 | 0.0499 | 主要来自批量搬运 |
| 8192 | E4M3 | 7.0235 | 0.1387 | 消除逐元素标量编码 |
| 8192 | E5M2 | 7.0136 | 0.1393 | 同一 SIMD 编码框架 |

短输入的 Python/launch 成本占比较高，收益不能按 8192 token 的比例外推。
这些数字只包含写入，不包含 scale 估计、QKV projection、RoPE 或 attention。

### 整模型消融

主结果来自 [旧编码](ablation-npu6/serving-scalar-npu6/results/enhanced/e4m3.json)、
[新编码](ablation-npu6/serving-vector-npu6/results/enhanced/e4m3.json) 和各自随后运行的 BF16。
统计明细与样本范围见 [summary](ablation-summary.json)。

| 配置 | 3-token 请求 s | 135-token 请求 s | 8192-token 请求 s | 长输入 TTFT ms | 长输入 TPOT ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| 旧编码 E4M3 | 0.5265 | 0.5230 | 1.3634 | 794.01 | 18.36 |
| 旧编码后 BF16 | 0.4954 | 0.4909 | 1.0844 | 618.75 | 15.03 |
| 向量编码 E4M3 | 0.5242 | 0.5219 | 1.1856 | 621.44 | 18.21 |
| 新编码后 BF16 | 0.4935 | 0.4927 | 1.0854 | 618.40 | 15.03 |

- 长输入整体省约 177.8 ms，降低 13.0%；TTFT 省约 172.6 ms，降低 21.7%。
- 单层写入省约 6.885 ms，28 层合计约 192.8 ms。这只是解释量级的估算，不是逐层 profiler 累加实测。
- 新编码 TPOT 相对旧编码变化很小，符合这次只修改 prefill 的事实。
- 新编码整体仍约比本轮 BF16 慢 9.2%；不能称为已经超过 BF16。
- 三种输入的旧/新 E4M3 输出文本集合完全一致。它说明优化没有改变这些样本的计算语义，
  不证明量化模型在长问答、数学或知识任务上的质量等同于 BF16，也不说明模型能力提高。

## 第二、三步：直接消费压缩 tile 的 Vector/Cube 注意力

### 数据流与硬件映射

最终源码为 `build-v28`，纯 Vector native 库为 `build-v24`；入口及内存边界见
[改动说明](CHANGES.zh-CN.md)。一次 attention 处理一个层自己的 K/V，不跨层复用不同的 K/V 内容。

```text
AIV0：解码 K → 当前 K tile ───────────────┐
AIV1：解码 V → V slot 0 / 1（提前一块）   │
                                        ↓
                   Cube：QKᵀ → Vector：online softmax
                                        ↓
                   Cube：P×V（异步） ── 与下一块 K 预取重叠
                                        ↓
                      FP32 分段结果 → 稳定合并 → BF16 输出
```

- 每次处理 512 token，跨四个 128-token 物理页。codec/offset/scale 在页内复用。
- 每个 KV head 的一次解码供七个 Q head 使用；Cube 的 M 维补齐至 16。
- 每个 Cube 组使用两个 AIV：AIV0 解码 K、执行 softmax 并发起矩阵乘；AIV1 解码 V，
  使用两个 GM slot 形成生产/消费队列。下一块 V 的解码可与当前注意力重叠。
- AIV0 在 P×V 异步执行期间预取下一块 K；消费矩阵结果之前调用 `WaitIterateAll`。
- split-K=4 在 8K 场景产生 16 个有效组；检查还覆盖 1/8/16。更多 split 不一定更快，
  因为会增加合并、空分段及多轮执行成本。
- LUT、解码与 softmax 中间量留在当前 AIV 的 UB；跨层复用的是设备 metadata 和同流工作区，
  不能把跨 kernel 的 UB 内容当作永久缓存。
- 32 MiB 调用方工作区包含系统区、每组有界 tile/信号区和 FP32 split 输出，逐层同流复用。
  没有生成完整上下文的 BF16 KV 页。8K 时原分阶段 K/V scratch 约 16 MiB，
  因此不能仅靠“有界 tile”声称这一长度的所有临时内存已经更小。

这里使用 BF16 Cube 矩阵乘，FP8/FP4 是存储格式。UB 与 Cube L1 之间仍经有界 GM tile，
不能称为完全片上、零 HBM 往返。尚未采集 L2 命中率、HBM 实际字节数或 Cube 利用率计数器。

### 每一步的 kernel 收益

同一 checker、Qwen head geometry、8192 token、split-K=4，表中均为未插桩正常路径的
每层 graph event 时间，单位 ms。各阶段在不同时段运行；频率/温度漂移不能完全排除。
最终版本与同轮 BF16/分阶段路径比较，避免只引用跨轮绝对耗时。
原始数据及汇总见 [阶段汇总](kernel-stages.json)。

| 阶段 | lossless | E4M3 | E5M2 | FP4 |
| --- | ---: | ---: | ---: | ---: |
| 128-token tile，Vector/Cube 串行 | 0.1860 | 0.2699 | 0.2705 | 0.6080 |
| 512-token tile，显式事件同步 | 0.1155 | 0.1997 | 0.1999 | 0.5371 |
| 双 Vector，V 双缓冲 | 0.0815 | 0.1227 | 0.1227 | 0.2913 |
| 再加异步 P×V 期间的 K 预取 | 0.0734 | 0.1150 | 0.1151 | 0.2835 |

扩大 tile 后，E4M3 约降低 26.0%；双 Vector/V 缓冲再降低约 38.6%；
K 预取再降低约 6.2%。从首个有效 Cube 原型到最终版本，约降低 57.4%。
这说明仅把乘法移到 Cube 并不够，前几版主要时间仍花在解码/搬运和等待。

最终一轮相同输入、相同物理卡的参照如下；没有用单次 Python eager 提交对比 graph。

| 精度 | 分阶段 decode＋原生 PA | 最终 Cube 流水线 | 结果 |
| --- | ---: | ---: | --- |
| lossless | 0.0903 | 0.0734 | 降低 18.6% |
| E4M3 | 0.1225 | 0.1150 | 降低 6.1% |
| E5M2 | 0.1236 | 0.1151 | 降低 6.9% |
| FP4 | 0.2614 | 0.2835 | 慢约 8.5%，不作为收益推荐 |
| 原生 BF16（独立参照） | 0.0480 | — | 仍最快 |

最终 FP8 每层仍约为原生 BF16 的 2.4 倍耗时。FP4 读端数学路径虽跑通，
解码代价和原有量化误差都还较大，不能从 4-bit 存储密度推出性能或质量更好。
表中的 lossless 是 PSP-KV 无损 KV 路由，布局和图接线仍与原生 BF16 基线不同。

### 哪一步还在花时间

512-token 串行版中，E4M3 的解码/GM tile 写入区间占四个阶段 cycle 总和的 77.2%。
分配第二个 AIV 后，主 AIV 只解码 K，并让 V 的工作重叠，前台 K 区间占比变为 59.2%。

最终版本的记录如下。占比是有效 job 的 cycle 累加占比，不是全芯片 wall time 占比，
各区间包含自身的传输/同步，也不是纯 Cube 算术时间。

| 最终 E4M3 区间 | 占四阶段 cycle 总和 |
| --- | ---: |
| 前台 K 解码/写入 | 16.5% |
| QK（含调用、等待和 score 搬入） | 14.8% |
| online softmax / rescale / P 写入 | 14.5% |
| PV 区间（含下一块 K 预取、V 信号等待） | 54.3% |

最后一行不能直接解读为“P×V 乘法最慢”：其中约 89.3% 的区间时间用于下一块 K 的预取。
V 等待计时约占四阶段总和的 0.5%，且包含读信号本身的成本。
前台 K 与预取 K 的工作量没有凭空消失，只是部分工作与 Cube 执行重叠了。
原始 cycle 字段 6/7 在 v28 中分别记录 K 预取/V 等待；旧 checker 的原始字段标签仍为 reserved，
[处理后的阶段汇总](kernel-stages.json) 按实际代码解释，保留原始 JSON 不改写。

### 最终整模型对照

物理卡 4 顺序执行：BF16-before → 向量写入＋分阶段 FP8 → 向量写入＋Cube 流水线 → BF16-after。
两个候选唯一的路径差别是 decode consumer；均保留此前的 RoPE/scatter、元数据与 graph 优化。
Cube 固定 split-K=4，所有长度共用同一张动态 decode graph，没有按测试长度重新捕获最优图。

| 配置 | KV token 容量 | 3-token 请求 s | 135-token 请求 s | 8192-token 请求 s | 长输入 TTFT ms | 长输入 TPOT ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| BF16-before | 113,664 | 0.4976 | 0.4962 | 1.0831 | 616.09 | 15.08 |
| FP8 分阶段，已用向量写入 | 218,752 | 0.5201 | 0.5228 | 1.1840 | 620.50 | 18.20 |
| FP8 最终 Cube 流水线 | 218,752 | 0.5192 | 0.5185 | 1.1487 | 611.20 | 17.31 |
| BF16-after | 113,664 | 0.4914 | 0.4923 | 1.0796 | 615.05 | 14.97 |

- 长输入请求耗时降低 2.99%，后续生成 TPOT 降低约 4.90%。短/跨页输入的整体差异较小，
  与基线短请求的漂移接近，不声称它们已有稳定显著收益。
- BF16 长请求前后漂移约 0.32%。最终 FP8 相对两次 BF16 合并中位数仍慢 6.20%。
- 基于本次各六个样本的独立 bootstrap，长输入耗时降低的条件 95% 区间约为 [2.75%, 3.56%]；
  短输入与跨页区间包含零。它不能涵盖跨天/跨进程环境差异，不替代多轮正式服务基准。
- 新流水线没有修改 prefill attention，因此不把 TTFT 的约 9 ms 波动全归因于 decode kernel。
- 分配器容量约为 BF16 的 1.925 倍；候选在本轮仍使用相同显存预算，并未让整个进程显存减半。
  容量是总 KV 池，当前 max_model_len=8320、max_num_seqs=1 并未随容量一起提高。
- 短输入和 135-token 输入的输出文本集合与分阶段 FP8 相同；8192-token 文本不同，且出现重复。
  BF16 与两种 FP8 的原始输出均保留。没有做 PPL、长文 QA 或任务集准确率评测，
  不能声称融合后模型能力提升、质量等同 BF16，或长上下文质量已经可用。

统计与原始结果见 [model summary](model-summary.json)、[effect/区间](model-effects.json)、
[前置 BF16](model-npu4/serving-bf16-npu4b/results/enhanced/bf16-before.json)、
[分阶段 FP8](model-npu4/serving-paged-npu4/results/enhanced/e4m3.json)、
[融合 FP8](model-npu4/serving-pipeline-npu4/results/enhanced/e4m3.json)、
[后置 BF16](model-npu4/serving-pipeline-npu4/results/enhanced/bf16-after.json)。

## 剩余瓶颈与后续方向

1. **Vector 读端与 tile 传输仍是核心成本。** 每个 FP8 字节仍需展开为索引、查表、scale 与 BF16。
   下一步可测试按 page/head 预计算带 scale 的小 LUT、复用 UB，以及双缓冲解码输出以减少
   每 32 token 的等待。FP4 可评估将两个缩放后的 BF16 打包进 32-bit LUT，减少展开指令。
   这些是后续方案，本轮未实现，不能提前计入收益。
2. **跨核通信和 GM 桥接尚未消失。** 高阶 Matmul 降低了实现风险，但仍有消息、布局整理、
   GM tile 与图内多个 kernel 的成本。更低层的 Vector/Cube 同步与 L1/L0 调度需要另做实现和验证。
   本轮没有完成全片上 attention，也未证明 GM tile 全部驻留 L2。
3. **不是所有请求都由 KV 带宽主导。** Qwen 7B 日志中的 BF16 权重约 14.28 GiB。
   按 28 层、4 KV head、128 维计算，8K KV 的理论读体量约 448 MiB/步（BF16）或 224 MiB/步（FP8）。
   压缩减少的 KV 数据远小于单请求时的权重体量；这些字节数是量级分析，不是 HBM 计数器实测。
   因而本轮单并发 8K 测试中，容量收益显著，整模型时延收益有限。
4. **模型其他计算、写入和图接线仍有成本。** decode RoPE/scatter 中尚有标量编码，
   候选使用的编译适配路径也与原生不同。整模型还包含图参数更新和调度，不能把所有 TPOT 差值都归因于单个 Matmul；
   不能用单个 attention microbenchmark 推断这些阶段的精确占比；
   后续需要完整模型时间线。当前策略 serving 接线仍限制单请求；高并发测试需先扩展接线，
   更长上下文的吞吐和可承载请求数也尚未测试。

默认保持分阶段主路径，新的 `cube` 路径显式启用。FP8 在本轮长输入有小幅可测收益；
FP4 consumer 仍未优于分阶段实现。完全片上、prefill 三合一、chunked prefill 重叠、
固定 prefill graph、原生 FP8 转换指令和系统性质量评估均不应标为已完成。

## 验证与复现

- 向量编码：lossless/E4M3/E5M2 × 512/3/135/8192 token，共 12 个夹具；512-token 夹具覆盖
  全部 65,536 种 BF16 位模式，包括 subnormal、signed zero、Inf 和 NaN。与旧标量 writer 字节完全一致。
- 原有 NPU 回归：263 passed，覆盖三种入口、四种精度、混合精度页、attention 与缓存复制。
- 最终相关 CPU 回归：62 passed；另保留早期 85 项检查记录，二者选择范围不同。
- 新编码整模型：旧/新编码各三个长度、各六次请求，文本集合一致；两个 BF16 进程用于观察同卡漂移。
- 最终 Cube：64 个固定精度/长度/split 组合，记录的最大绝对误差为 0.0078125；
  另有 24 个动态长度/query 重放检查。混合物理精度页的双序列图重放 16 组，
  记录最大绝对误差 0.015625。容差为 atol=rtol=2e-2，且检查源 K/V、缓存不变及工作区边界。
  数值结果与 develop 的 CPU oracle 比较；P 使用 BF16，因此不要求 bitwise attention 一致。

全部 NPU 测试均使用 uv 创建的 `/workspace/pspkv-native-build/.venv/bin/python`，未使用系统 Python/pip。
构建、检查、服务命令及源码/库哈希见 [COMMANDS](COMMANDS.md)；实现位置见 [CHANGES](CHANGES.zh-CN.md)。

官方接口依据：系统/用户 workspace 的区分遵循
[CANN workspace 约定](https://www.hiascend.com/document/detail/zh/canncommercial/82RC1/opdevg/Ascendcopdevg/atlas_ascendc_10_0092.html)；
Vector/Cube 注册与同步参考
[CANN 9.0 融合 Matmul](https://www.hiascend.com/doc_center/source/en/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_10_0050.html)。
