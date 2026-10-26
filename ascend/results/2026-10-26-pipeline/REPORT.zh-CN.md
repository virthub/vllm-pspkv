# Qwen2.5-7B：写入、RoPE、工作区与计算图优化

已完成直接 K/V 写入、有效输出免清零、单 token RoPE/写入融合、
跨层工作区复用，以及受限范围的 decode ACL Graph。
在同一个已加载模型内随机切换四个阶段，Basic / Enhanced / BTKV 请求耗时
P50 相对本轮原 eager 路径降低 **67.8%～70.8%**，约为 **3.1～3.4 倍速度**。
不开计算图，仅直接写入和 RoPE 融合，累计也降低约 **19%～22%**。

FP8 三格式的全部配对生成文本和已生成 token 的 API logprob 保持一致。
当前速度已接近原生 BF16 图模式，同时保留约两倍 KV token 容量。
这些短上下文、单请求结果不能代表并发吞吐、完整准确率或生产后端的全功能迁移。

## 测试条件与四个阶段

Ascend 910B3，Qwen2.5-7B **基础版**，BF16 权重，TP 1，单请求，
128-token 页，显存预算比例 0.35，关闭前缀缓存，temperature 0、seed 123。
输入 3 / 135 token，强制生成 32 token；FP8 主表请求返回 logprobs。
最大模型长度 256。模型、权重、FP8 编码、单位缩放及三种布局保持原样。
BTKV 为参考布局兼容实现。

每种格式只加载一次模型。每种输入、每种模式预热 3 次，随后进行 12 组配对：
每组随机排列四种模式，完成一个请求后再切换。三格式共有 **288 个计时请求**，
另有 **144 个质量探针请求**；profiler 请求在计时后单独运行。

| 阶段 | 行为 |
| --- | --- |
| 0 | 本轮原 eager 路径，保留上一轮跨层元数据复用与融合注意力 |
| 1 | 直接 K/V 写入，移除有效输出清零；适用长输入时复用工作区 |
| 2 | 阶段 1 加单 token RoPE/缓存写入融合 |
| 3 | 重放阶段 2 的 decode 图，prefill 保持原路径 |

控制文件只用于实验，每次 forward 固定一次模式。原生 BF16 使用独立进程对照。
当前实现接入独立三格式适配器；服务器原源码目录与原共享动态库未被覆盖。

## 整模型速度：逐步收益

以下是请求耗时 P50，单位秒；最后一列相对于同一行阶段 0 计算。

| 格式 | 输入 token | 阶段 0 | 阶段 1 | 阶段 2 | 阶段 3 | 总耗时降低 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Basic | 3 | 1.660 | 1.534 | 1.344 | **0.485** | **70.8%** |
| Basic | 135 | 1.663 | 1.514 | 1.340 | **0.494** | **70.3%** |
| Enhanced | 3 | 1.578 | 1.493 | 1.249 | **0.496** | **68.6%** |
| Enhanced | 135 | 1.586 | 1.498 | 1.279 | **0.507** | **68.0%** |
| BTKV 参考兼容 | 3 | 1.631 | 1.518 | 1.271 | **0.491** | **69.9%** |
| BTKV 参考兼容 | 135 | 1.563 | 1.496 | 1.263 | **0.503** | **67.8%** |

- 直接写入和免清零作为一组修改，P50 降低约 **4.3%～9.0%**；未拆分两者的独立收益。
- 在阶段 1 基础上加入实际 RoPE 融合，进一步降低约 **11.5%～16.4%**；
  相对阶段 0 累计降低约 **19%～22%**。
- 最大收益来自图重放，消除了 decode 中大量逐层 Python 包装和算子提交。
  百分比不能直接相加。
- 跨页输入的 TPOT P50：Basic **50.16 → 13.93 ms**，
  Enhanced **47.75 → 14.39 ms**，BTKV **47.14 → 14.21 ms**。

TPOT 使用流式首末非空文本事件估计；TTFT 也含文本缓冲与服务端开销，
不能直接作为 prefill kernel 时间。

对配对均值降低做 10000 次 bootstrap，阶段 3 的 95% 区间均为正，约为
**67.3%～71.9%**。这些区间针对配对均值，不是上表 P50 的区间。
阶段 1 的 BTKV 跨页区间约为 **-0.6%～9.2%**，单看这一步不能保证统计显著。
完整 P50/P95、TTFT、TPOT、配对区间见
[主表 JSON](summary/comparison.json) / [CSV](summary/comparison.csv)。

## 与原生 BF16 的比较

原生 BF16 单独启动，使用相同模型、输入与 32-token 输出长度。
每类输入 12 个计时样本，另有 12 类质量探针。

| 配置 | 物理卡 | 短输入 P50 | 135-token 输入 P50 |
| --- | ---: | ---: | ---: |
| BF16，原生 eager | 2 | 1.536 s | 1.555 s |
| BF16，原生 FULL_DECODE_ONLY | 2 | 0.499 s | 0.497 s |
| BF16，原生 FULL，捕获大小 1 | 4 | 0.502 s | 0.510 s |
| Basic，阶段 3 | 4 | 0.485 s | 0.494 s |
| Enhanced，阶段 3 | 4 | 0.496 s | 0.507 s |
| BTKV，阶段 3 | 4 | 0.491 s | 0.503 s |

原生两种图模式的计时请求和 12 类探针，文本均与原生 eager 一致。
图模式只捕获大小 1，主表的多 token prefill 回退 eager。
FP8 与 BF16 图模式已经接近；独立进程、不同采样时段和部分配置的卡号不同，
不能依据约 1%～4% 的小差异宣称稳定超越 BF16。

分配器容量保持为 BF16 图模式 **130,816**、FP8 **261,504 KV token**。
这是同一缓存预算下可存储 token 数的收益，不表示整个进程显存减半。
原生 BF16 也能从计算图获益，因此主表约三倍速度不能算作 KV 压缩独有收益。

**排除的实验：**测试用同进程图旁路切换器不适用于 vendor BF16 的 CPU 注意力
任务更新路径，出现重复输出；该次数据未进入有效比较。
随后采用上述独立原生启动方式复核，输出对照通过。
最终启动器对 BF16 使用固定模式独立进程，运行时拒绝将自定义图旁路应用于 BF16。
有效数据与排除原因见 [baseline-summary.json](baseline-summary.json)，
原始失败记录仍保留在 `raw/final-stages/stock-bf16-serving.json`。

## 1536-token 输入

物理卡 2，最大长度 2048，仍生成 32 token。每类输入做 6 对相邻请求。
较长输入使用已有 Vector 解码加原生矩阵化分页注意力，**没有开启整模型图**。

| 格式 | 阶段 0 P50 | 阶段 2 P50 | 降低 |
| --- | ---: | ---: | ---: |
| Basic | 1.722 s | **1.517 s** | **11.9%** |
| Enhanced | 1.724 s | **1.460 s** | **15.3%** |
| BTKV 参考兼容 | 1.699 s | **1.492 s** | **12.2%** |

这是直接写入、免清零、RoPE 融合和工作区复用的合计收益，
没有把其中任何一项单独归因为 12%～15%。优化前后生成文本一致。
工作区复用减少分配次数，各层 KV 仍分别解码并写入临时 BF16 活跃页。
本轮没有实现长上下文的一体化 Vector/Cube attention kernel。
详见 [长输入 JSON](long-summary/comparison.json) / [CSV](long-summary/comparison.csv)。

## 实际时间线：主要省在何处

Basic 的额外采样：135 / 3-token 输入，各生成 4 token；分别测阶段 0 / 3。
共 8 次 forward，含 6 个 decode 步、每步 28 层。

| 指标 | 阶段 0 | 阶段 3 |
| --- | ---: | ---: |
| 设备 kernel 数 | 3344 | **2672** |
| 主机 `aclrtLaunchKernelWithHostArgs` 次数 | 3112 | **904** |
| 主机 `empty_tensor` 次数 | 4264 | **1322** |
| 原 Triton RoPE kernel 数 | 224 | **56，仅 prefill** |
| `PspkvRopeScatter` 次数 | 0 | **168，全部 decode 层** |
| Stack Pack kernel 数 | 224 | **0** |
| 融合注意力次数 | 168 | 168 |
| 设备 kernel 时间总和 | 129.109 ms | **126.144 ms** |

设备计算只降低约 **2.3%**，端到端请求却降低约 70%，说明最大收益来自
主机包装、张量管理及提交间隙，而非把矩阵乘法本身加速三倍。
在 162 个相邻 decode 层窗口中，窗口总长 **313.49 → 71.81 ms**，
已记录 kernel 区间并集 **73.99 → 71.74 ms**。这些局部窗口接近连续执行，
不能据此宣称整个请求或整卡算术利用率达到 100%。

原来每层的同步拷贝没有回来。图框架另有 **6 次**
`aclrtSynchronizeStreamWithTimeout`，对应每个 decode 步一次，
采样主机等待合计约 33.94 ms；等待与设备工作重叠，不能直接加到设备计算时间上，
也不能视为删除后必然节省的请求耗时。
完整统计见 [final-model-profile-summary.json](final-model-profile-summary.json)。

## 推理结果与剩余瓶颈

- FP8 主表三格式的所有配对文本和 12 类探针文本均一致。
  对每种格式、每种模式比较 1152 个已生成 token 的 API logprob，最大差值为 **0**。
  这不扩展为对全部词表 logits、困惑度或通用任务准确率的证明。
- 原有 FP8 输出的重复等质量现象仍保留；权重、编码规则和精度没有改变。
  本轮没有把既有量化质量问题计作已经解决。
- 图后主要 MatMul 合计约 **106.82 ms，占设备 kernel 时间的 84.7%**。
  独立投影算子计数器中，单 token gate/up 的 MAC 活跃比例约 **10.5%**，
  MTE2 搬运管线约 **98.6%**，更支持权重搬运限制的判断。
  这是独立重复权重测试；管线可重叠，缓存会影响结果，不能当作整模型利用率。
- Prefill 压缩写入仍有成本；本次采样的 56 次直接 prefill 写入约 6.87 ms，
  168 次 decode 融合写入约 1.33 ms。更长 prefill 的量化写入值得继续分析。
- 较长上下文仍有活跃页 BF16 临时数据往返。GQA 组内片上复用、
  Vector/Cube 融合流水以及多请求调度尚需单独实现和验证。
- 另尝试了插件权重预取，在较早的直接写入图候选版本上未见收益，未默认开启。
  该探索不混入最终四阶段结果，见 [prefetch-summary.json](prefetch-summary.json)。

硬件计数器见 [matmul-summary.json](matmul-summary.json)。

## 改了哪些代码与验证

逐处实现、支持边界及调用关系见 **[CHANGES.zh-CN.md](CHANGES.zh-CN.md)**。
核心改动为 `packed_scatter_kv`、`rope_scatter_kv`、
`gather_paged_bf16_vec_out`、Qwen2 decode 入口、工作区所有权和图输入校验。
相对本轮开始快照的核心代码差异见 [code.patch](code.patch)，
源码及动态库指纹见 [manifest.json](manifest.json)。

- CPU **71 项通过**；NPU 原正确性、混合精度与拷贝回归 **263 项通过**。
- **90 组**直接写入比较缓存字节，覆盖 stride、混合精度页、跳过 slot、空输入。
- **48 组** RoPE/写入与原 Triton RoPE 加旧写入器一致。
- **21 次**图重放覆盖动态长度、跨页、页表换序和同长度新请求。
- **9 组、36 次**跨层工作区输出比较通过。
- 验收同时核对整模型实际 kernel，避免把仅通过独立测试的算子算作已经接入。

见 [CPU](cpu-tests.txt)、[NPU 回归](raw/results/regression.log)、
[写入与图](raw/results/correctness.json)、[RoPE](raw/results/rope-correctness.json)、
[工作区](raw/results/workspace-correctness.json)。最终采样使用服务器固定的
`release-v2` 快照；收尾又增加原生 BF16 实验边界检查，不改变 FP8 计算路径。

## 复现与材料位置

服务器目录：`/root/zhang/pspkv-pipeline-20261026`。
`final-review` 提供最终源码与本报告；`release-v2` 为主表计算实现的固定快照，
`final-stages` / `final-long` 保留原始请求、服务端日志及完整 profiler 数据。
本地 `raw` 包含用于本报告的原始 JSON、日志和 profiler CSV。
动态库为 `libpspkv_pipeline.so`，SHA256：

```text
482c2e32311b74685df8dd09165720dd6098b2523f5b719a84aa5aeca886c712
```

在确认物理卡空闲后，用固定副本重复 FP8 四阶段测试：

```bash
PSPKV_REVIEW=/root/zhang/pspkv-pipeline-20261026/final-review
PSPKV_REPRO=/root/zhang/pspkv-pipeline-20261026/reproduce
mkdir -p "$PSPKV_REPRO"
cp "$PSPKV_REVIEW"/ascend/{format_compare_runtime.py,pipeline_runtime.py,shared_metadata.py,serve_initialized_npu.py,compare_formats_client.py,compare_pipeline_client.py,compare_pipeline_stages.py,profile_model_client.py} "$PSPKV_REPRO/"
NPU_ID=4 DRIVER_CONTAINER=pspkv-vector-fused-dev-20261024 \
  COMPARISON_DIR="$PSPKV_REPRO" \
  BACKEND_SOURCE="$PSPKV_REVIEW/vllm/v1/attention/backends/psp_kv_ascend.py" \
  OPS_OVERLAY="$PSPKV_REVIEW/vllm/v1/attention/backends/psp_kv_ascend_ops.py" \
  NATIVE_LIBRARY=/root/zhang/pspkv-pipeline-20261026/libpspkv_pipeline.so \
  MODEL=Qwen/Qwen2.5-7B RUN_ID="pipeline-$(date +%Y%m%d%H%M%S)" \
  HOST_PORT=18136 COMPARE_OPTIMIZED=1 PAGED_MIN_LENGTH=64 \
  ATTENTION_IMPL=fused GRAPH_MODE=full PIPELINE_STAGES=1 INCLUDE_STOCK=0 \
  PROFILE_MODEL=1 CLIENT_WARMUPS=3 CLIENT_ROUNDS=3 CLIENT_REQUESTS=4 \
  bash "$PSPKV_REVIEW/ascend/run_format_comparison.sh"
```

长输入复现使用新的比较目录，改为 `GRAPH_MODE=eager PIPELINE_STAGES=0`、
`PAIRED_PIPELINE=1 PIPELINE_AFTER_LEVEL=2 MAX_MODEL_LEN=2048`、
`LONG_INPUT_TOKENS=1536 ATTENTION_IMPL=auto`。
BF16 则使用独立目录，设置 `COMPARE_FORMATS=stock-bf16 PIPELINE_STAGES=0`、
`PIPELINE_CLIENT=1`，分别运行 `GRAPH_MODE=eager` 和 `GRAPH_MODE=full`。
不要用自定义的同进程旁路切换替代原生 BF16 图对照。

统计命令如下，FP8 四阶段及长输入分别汇总：

```bash
.venv/bin/python ascend/summarize_pipeline_stages.py \
  --input-dir ascend/results/2026-10-26-pipeline/raw/final-stages \
  --exclude-formats stock-bf16 \
  --output-dir ascend/results/2026-10-26-pipeline/summary
.venv/bin/python ascend/summarize_pipeline_stages.py \
  --input-dir ascend/results/2026-10-26-pipeline/raw/final-long \
  --output-dir ascend/results/2026-10-26-pipeline/long-summary
```

原生 BF16 独立进程汇总见 `baseline-summary.json`，原始无效 A/B 数据须排除。
当前整模型图验证限于上述 Qwen2.5-7B、单请求、最大长度 256；
正常服务后端集成、并发和其他模型仍需后续工作。
