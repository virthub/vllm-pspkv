# 跨层元数据复用：实现、配对测试与剩余瓶颈

已完成当前三格式推理路径的元数据复用，并验证了输出和性能。
在同一个已加载模型进程内交替开关，Basic / Enhanced / BTKV 的请求耗时 P50
降低约 **8.4%～11.3%**。同步拷贝及流同步从各 168 次降至 0。
本轮保持上一轮融合/量化动态库不变，收益来自元数据准备和提交路径。

## 实现

通用 `CommonAttentionMetadata` 已持有设备长度，但 Ascend 普通 self-attention
builder 将 CPU 长度放入 `AscendMetadata.seq_lens`，原适配器随后又在每层构造 NPU 张量。

本轮新增 `PSPKVAscendMetadataBuilder`，保留原有设备长度引用
`psp_kv_device_seq_lens`。不修改 vendor 插件的 CPU 字段或原生接口。
`shared_metadata.py` 在当前 `ForwardContext` 内准备并共享：

- 当前有效请求的设备长度视图，以及已有 CPU 长度视图；
- 当前有效页表的裁剪与 int32 / 连续化结果；
- slot mapping 的 int64 / 连续化结果。

缓存按页表对象等信息区分缓存组，持有源对象直到本次 forward 结束。
下一步重新绑定当时的映射；没有基于长度的全局 LRU，也不跨请求保存转换后的页表。
各层 K/V、精度、偏移和缩放保持独立。

适配器默认 `PSPKV_ASCEND_SHARE_METADATA=1`，设为 `0` 恢复逐层准备。
单独调用而没有 forward owner 时不持久缓存；缺少设备长度时支持一次性转换的兼容路径。
常规运行不读取控制文件，配对基准的控制文件仅用于同一进程内切换实验模式。

核心修改见 [backend builder](../../../vllm/v1/attention/backends/psp_kv_ascend.py)、
[元数据准备器](../../shared_metadata.py)、[三格式接入](../../format_compare_runtime.py)。

## 数值和同步检查

- 本地 CPU 测试 **67 项通过**，其中新增元数据测试 13 项，覆盖同长度新请求切换、
  原地更新的持久输入缓冲、缓存组隔离、dtype 转换、多请求裁剪和 CPU 长度惰性读取等。
- NPU **15 组多层用例**通过：三格式，长度 3/32/135/160/257，每层独立 K/V，
  优化前后输出逐位一致。
- 六组 28 层测试均确认：一次 forward 只准备一次，命中 27 次，设备长度直接借用。
- 原有正确性、混合精度和后端拷贝回归 **263 项通过**。
- 三格式全部计时请求及算术/常识探针，开关两种模式的输出文本一致。

隔离测试连续执行 6 个 decode 步，每步 28 层；整模型额外采样两个各生成 4 token
的请求，也包含 6 个 decode 步。两种场景均验证了重复同步的消除。

| 整模型采样指标 | 关闭复用 | 开启复用 |
| --- | ---: | ---: |
| `aclrtMemcpy` 同步拷贝 | 168 | **0** |
| `aclrtSynchronizeStream` | 168 | **0** |
| `aclrtMemcpyAsync` | 120 | 120 |
| 元数据等路径的 `aclnnInplaceCopy_CastAiCore_Cast` | 256 | **40** |
| 设备 kernel 总数 | 3560 | 3344 |
| 融合注意力调用数 | 168 | 168 |

216 次 Cast 的减少对应每次 forward 转换一次、其余 27 层复用的模式
（本次采样共 8 次 forward）。步级输入更新的异步拷贝依然存在，未宣称消除全部传输。
实际模型日志 `PSPKV_SHARED_METADATA` 确认 `forward_scoped=true`、
`borrowed_device_lengths=true`、`npu:0`、`torch.int32`。

28 层注意力回调链的独立主机计时如下。该链包含缓存写入和注意力，
不包含模型线性层、MLP 和服务端调度，因此不能将加速幅度直接套用到整模型。

| 长度 | 格式 | 关闭复用 | 开启复用 |
| --- | --- | ---: | ---: |
| 32 | Basic | 13.040 ms | 9.014 ms |
| 32 | Enhanced | 13.953 ms | 9.911 ms |
| 32 | BTKV | 14.513 ms | 10.080 ms |
| 160 | Basic | 13.636 ms | 8.716 ms |
| 160 | Enhanced | 15.400 ms | 9.824 ms |
| 160 | BTKV | 14.235 ms | 10.186 ms |

数据见 [CPU 日志](cpu-tests.txt)、[NPU 回归](raw/results/regression.log)、
[多层验证及计时](raw/results/micro.json)、[微基准 API 计数](micro-api-summary.json)。

## 整模型配对测试

环境为 Ascend 910B3、Qwen2.5-7B 基础版、BF16 权重、TP 1、单请求、eager，
关闭前缀缓存，128-token 页，最大模型长度 256，显存预算比例 0.35。
输入为 3 / 135 token，固定生成 32 token，温度 0、seed 123。

三格式各加载一次模型。每种输入、每种模式先预热 4 次，然后做 12 对相邻请求，
每对随机决定先关闭还是先开启复用。每类每模式 12 个样本，三格式共 144 个计时请求，
另有 24 个 BF16 对照请求。开关控制每个 forward 读取一次，两种模式有相同的实验控制开销。
额外 profiler 请求全部在计时结束后执行。

下表为请求 P50；降低百分比按本轮配对数据的两组 P50 计算：

| 配置 | 短输入：关闭 → 开启 | 降低 | 跨页输入：关闭 → 开启 | 降低 |
| --- | ---: | ---: | ---: | ---: |
| Basic | 1.643 → **1.479 s** | **10.0%** | 1.652 → **1.489 s** | **9.9%** |
| Enhanced | 1.679 → **1.502 s** | **10.5%** | 1.752 → **1.553 s** | **11.3%** |
| BTKV 参考兼容 | 1.645 → **1.507 s** | **8.4%** | 1.663 → **1.501 s** | **9.7%** |

跨页 TPOT P50 分别从 49.81 / 52.64 / 50.25 ms，降至
**44.70 / 46.72 / 45.15 ms**，每个后续 token 减少约 5.1～5.9 ms。

此外，对配对均值的耗时降低做 10000 次配对 bootstrap，95% 区间为：

| 配置 | 短输入区间 | 跨页区间 |
| --- | ---: | ---: |
| Basic | 6.8%～11.9% | 8.7%～13.9% |
| Enhanced | 6.6%～11.2% | 9.3%～13.6% |
| BTKV | 3.6%～9.4% | 7.7%～12.0% |

这些区间针对配对均值，不是上表 P50 降低的区间；仅反映本轮样本，
未包含跨时段、跨服务器或其他负载的不确定性。全部区间均为正，支持本轮复用带来收益。

本轮 BF16 P50 为短输入 **1.490 s**、跨页 **1.510 s**。
Basic 两项略低于 BF16；Enhanced 分别约高 0.9% / 2.9%；
BTKV 短输入约高 1.1%，跨页约低 0.6%。整体已经接近 BF16。
BF16 是独立进程对照，没有与三格式逐请求配对，因此不据这些小幅差异宣称稳定超越。
本轮 BF16 KV 容量仍为 130,688 token，FP8 为 261,504，容量收益保留。

完整 P50/P95、TTFT、TPOT、配对均值及区间见
[comparison.csv](summary/comparison.csv)、[comparison.json](summary/comparison.json)。
跨页输入只是 135 token，没有替代长上下文或并发吞吐基准。

## 现在还有哪些耗时

Basic 的前后额外采样采用同一模型、同一组 135 / 3-token 输入，各输出 4 token。
以下数据来自插桩后的时间线，不能当作正常运行的整卡利用率或请求耗时占比。

### 收益主要来自主机等待和提交路径

关闭与开启复用，设备 kernel 执行时间总和分别约为 **128.883 / 128.735 ms**，
几乎不变。融合注意力始终调用 168 次，矩阵乘法和 FP8 写入次数不变。

在同一个 decode 步内，相邻融合层之间取 162 个窗口，排除 token 边界和请求切换：

| 指标 | 关闭复用 | 开启复用 |
| --- | ---: | ---: |
| 层间窗口总长 | 306.14 ms | **275.03 ms** |
| 已记录设备 kernel 区间并集 | 73.78 ms | 73.65 ms |

窗口缩短约 31.11 ms，而设备计算量基本不变，与消除每层同步、减少元数据转换的方向一致。
未被这些 kernel 覆盖的区间还包括主机提交、同步、拷贝和运行时等待，不能全部称为 CPU 计算。

### 设备计算主要在模型矩阵乘法

开启复用后的设备 kernel 时间中：

- 两类主要 MatMul 合计约 **106.72 ms，占 82.9%**；权重与 MLP/投影计算是主要设备成本。
- FP8 写入合计约 **8.69 ms，占 6.8%**，其中包含两个 prefill，不能直接当作 decode 写入占比。
- 融合注意力合计约 **3.25 ms，占 2.5%**，已经不是当前短上下文的首要设备热点。

这些是设备执行时间占比，不是总请求延迟占比。

### eager 下的调用和张量管理仍明显

同次采样的主机 self time 中，RoPE 包装调用约 **55.25 ms**，统一注意力调用约
**49.74 ms**；`empty_tensor` 仍调用 **4264 次**，合计约 **25.84 ms**。
RoPE 的设备计算只有约毫秒级，因此需要进一步检查包装层、Triton 提交及框架开销，
不能把该主机耗时都归为旋转位置编码的数学计算。

K/V `stack`、压缩写入与输出清零依然逐层执行，后续可评估减少张量创建、
让写入器直接接收 K/V，以及计算图对 eager 提交链的影响。更长上下文融合仍需要
GQA 组内复用和矩阵化；本轮没有修改这些算法或宣称对应收益。

分析见 [model-profile-summary.json](model-profile-summary.json)，
原时间线位于 `raw/comparison/model-profiler-basic`。

## 代码、边界与复现

- [代码补丁](code.patch)、[源码与动态库校验和](manifest.json)。
- [结束时设备状态](remote-final-state.txt)，本轮测试容器均已退出。
- [pre-commit](pre-commit.txt)。
- [配对客户端](../../compare_metadata_client.py)、[统计脚本](../../summarize_metadata_comparison.py)、
  [采样分析](../../analyze_metadata_profiles.py)、[NPU 验证](../../check_shared_metadata_npu.py)。
- 原始服务端配置、请求、输出及脚本快照保存在 `raw/comparison`。

动态库仍为 `8980346ea75b271574f5e01ea03c5c641b202b1fb3855c86c1627162df35c358`。
本轮没有重编译或替换量化/注意力 kernel，没有改变原有 FP8 编码和质量问题。
已验证范围为 eager、单请求；图重放、并发调度和其他模型仍需单独验证。
实际三格式推理接入仍使用独立适配层，原服务器源码和 vendor 插件未被覆盖。

服务器审阅目录为 `/root/zhang/pspkv-metadata-20261025/final-review`。
以下在空闲物理 NPU 4 上重复配对测试；已有开发容器应保持停止状态：

```bash
PSPKV_META_REVIEW=/root/zhang/pspkv-metadata-20261025/final-review
PSPKV_META_REPRO=/root/zhang/pspkv-metadata-20261025/reproduce
mkdir -p "$PSPKV_META_REPRO"
cp "$PSPKV_META_REVIEW"/ascend/{run_format_comparison.sh,serve_initialized_npu.py,format_compare_runtime.py,shared_metadata.py,compare_formats_client.py,compare_metadata_client.py,profile_model_client.py} "$PSPKV_META_REPRO/"
NPU_ID=4 DRIVER_CONTAINER=pspkv-vector-fused-dev-20261024 \
  COMPARISON_DIR="$PSPKV_META_REPRO" \
  BACKEND_SOURCE="$PSPKV_META_REVIEW/vllm/v1/attention/backends/psp_kv_ascend.py" \
  OPS_OVERLAY=/root/zhang/pspkv-vector-fused-20261024/ops-final.py \
  NATIVE_LIBRARY=/root/zhang/pspkv-vector-fused-20261024/build/libpspkv_final.so \
  MODEL=Qwen/Qwen2.5-7B INCLUDE_STOCK=1 \
  RUN_ID="metadata-reproduce-$(date +%Y%m%d%H%M%S)" HOST_PORT=18134 \
  COMPARE_OPTIMIZED=1 PAGED_MIN_LENGTH=64 ATTENTION_IMPL=auto \
  PAIRED_METADATA=1 PROFILE_MODEL=1 \
  CLIENT_WARMUPS=4 CLIENT_ROUNDS=3 CLIENT_REQUESTS=4 \
  bash "$PSPKV_META_REPRO/run_format_comparison.sh"
```
