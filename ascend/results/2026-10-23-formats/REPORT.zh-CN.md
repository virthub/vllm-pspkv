# 2026-10-23 Basic / Enhanced / BTKV 昇腾对照测试

本次在 worker03 的 Ascend 910B3 上完成了两层对照：三种原始算子入口的
数值与延迟测试，以及通过隔离测试适配层接入 Qwen3-0.6B 的流式服务测试。
这次比较固定精度和输入，仅改变缓存布局与原始注意力入口。

结论：三种路径在已测输入上输出一致；Enhanced 在较长上下文的当前标量
注意力实现中明显快于 Basic；BTKV 与 Enhanced 没有体现独立算法的稳定优势。
当前 BTKV 仍是兼容实现，没有使用 NVIDIA TMA 或对应的昇腾硬件加速。

## 环境、代码与隔离

- 服务器：worker03，Ascend 910B3，NPU 4。
- 镜像：`quay.io/ascend/vllm-ascend:v0.20.2rc1`；实际镜像 ID 见
  [image-id.txt](raw/image-id.txt)。
- Python 3.11.15、torch `2.10.0+cpu`、torch_npu `2.10.0`、vLLM
  `0.20.2+empty`、vLLM-Ascend `0.20.2rc1`。
- 使用服务器已有 `/root/zhang/vllm-pspkv` 工作树及原生库。
  服务器基础提交为 `dd2cf6d`，包含未提交的迁移实现；完整状态和关键文件
  哈希分别保存在 [source-status.txt](raw/source-status.txt) 与
  [source-sha256.txt](raw/source-sha256.txt)。
- 原生库 SHA256 为
  `72f17c1718609e04e3fad08404d832ab7fb2553fc42520fc31b06f1de1428ed1`。
- 所有实验串行使用同一张 NPU；原始源码和插件只读挂载。模型对照通过独立
  `backend_overlay.py` 替换测试容器内的固定 FP8 回调，没有改写服务器原文件。
- 测试前后原工作树状态及关键源码/库的哈希均一致。所有本轮容器已停止，
  NPU 4 已无运行进程。证据见 [final-containers.tsv](raw/final-containers.tsv)
  及 [npu-after.log](raw/npu-after.log)。

## 算子级对照

直接调用 `psp_attention_basic`、`psp_attention_enhanced`、
`psp_attention_btkv`，保留各自布局和参数检查。三者采用完全相同的逻辑 Q/K/V、
物理块映射和精度代码，缩放统一为 1，输出和 Query 均为 BF16。

工作负载包含：

- GQA：16 个 Q 头、8 个 KV 头，上下文 128、512、2048。
- MHA：8 个 Q 头、8 个 KV 头，上下文 512。
- 四请求 GQA：8 个 Q 头、2 个 KV 头，长度 17、32、63、112，覆盖尾页不满。
- head dimension 为 128，块长为 16，物理块顺序随机打乱。

每种负载分别测试无损 BF16、E4M3、E5M2、FP4、混合 E4M3/E5M2，以及
同一缓存中混合 BF16/E4M3/E5M2/FP4，合计 **90 个格式/负载/精度组合**。
精度表在测试中预先指定，验证的是消费混合块的能力，未运行自动精度策略。

每个组合先预热 5 次，再执行 5 轮，每轮 20 次。每轮随机排列三个格式的计时
顺序，记录主机总耗时和 NPU Event 耗时。下表使用各轮单次平均主机耗时的中位数，
单位为 ms；它不是单请求的 P50/P95。计时包含原生 QK、PyTorch softmax、
原生 V 加权和调用开销，不包含打包、分配与精度决策。

| GQA，上下文 512 | Basic | Enhanced | BTKV |
| --- | ---: | ---: | ---: |
| 无损 BF16 | 10.358 | 5.123 | 4.517 |
| FP8 E4M3 | 11.722 | 6.782 | 6.766 |
| FP8 E5M2 | 11.718 | 6.472 | 6.456 |
| FP4 | 11.014 | 5.704 | 5.705 |
| 混合 E4M3/E5M2 | 11.730 | 6.667 | 6.631 |
| 混合四种格式 | 11.377 | 5.923 | 5.927 |

| FP8 E4M3，改变负载 | Basic | Enhanced | BTKV |
| --- | ---: | ---: | ---: |
| GQA，上下文 128 | 2.926 | 1.718 | 1.722 |
| GQA，上下文 512 | 11.722 | 6.782 | 6.766 |
| GQA，上下文 2048 | 46.783 | 26.952 | 27.017 |
| MHA，上下文 512 | 11.635 | 6.763 | 6.738 |
| 四请求，尾页不满 | 2.611 | 1.515 | 1.514 |

全部 90 个组合通过参考数值校验；相同负载和精度下，三个格式的 NPU 输出
逐位一致。相对 CPU 参考结果的最大绝对误差为 **0.00048828125**。
三种格式的 K/V 数据字节数也完全相同，改变布局本身未减少数据容量。
详细数据见 [operators.csv](operators.csv) 和 [operators.json](raw/operators.json)。

首轮 BF16 的 Enhanced/BTKV 存在一处约 12% 的差异，因此补做了共享缓冲区的
控制实验：三个路径复用同一组 Query、K、V、块表、精度表和缩放表内存地址；
仅在计时外填入对应的 V 布局并充分预热，7 轮，每轮 20 次。

| 同址控制，GQA，上下文 512 | Basic | Enhanced | BTKV |
| --- | ---: | ---: | ---: |
| 无损 BF16 | 10.421 | 4.537 | 4.537 |
| FP8 E4M3 | 11.715 | 6.775 | 6.778 |

同址且充分预热时 Enhanced/BTKV 的差异低于 0.1%。因此首轮个别差异不足以
证明 BTKV 存在独立算法收益。其原因可能涉及分配地址或缓存状态，本轮没有
使用硬件计数器进一步归因。控制实验见 [control.json](raw/control.json)。

从代码看，Enhanced 和 BTKV 使用相同的转置 V 布局及相同的计算内核，
BTKV 描述符只做形状/对齐检查，没有被硬件 TMA 使用。Enhanced 相对 Basic
的注意力优势与 V 读取布局的变化一致，但本轮没有单独 profile 访存瓶颈。

FP4 输入限定在 [-2.5, 2.5]，用于保证单位缩放下有限。它的算子一致性结果
不能说明真实模型的 FP4 量化质量，也不覆盖此前发现的动态缩放边界错误。

## 完整模型服务对照

为填补此前只测 scaled-basic 在线路径的空缺，新增了显式的测试适配层。
它让三个容器分别直接调用原始 Basic、Enhanced、BTKV 的公开 decode 方法，
使用同一 `packed_scatter` 写入路径；公共内部计算的块长从默认 16 设为插件
要求的 128，其他原始格式行为保留。每个模型的 28 层均记录了实际入口：
`psp_attention_basic`、`psp_attention_enhanced`、`psp_attention_btkv`。

这不是将七种精度配置重新命名为三个格式，也不是自动选精度测试。三条路径
全部固定 E4M3、单位缩放、相同原生库和相同 eager prefill；Enhanced/BTKV 的
V 写为转置布局，Basic 的 V 保持 token-major。适配层仅用于本轮隔离测试，
没有加入常规后端的默认在线切换功能。

统一配置：Qwen3-0.6B、TP=1、eager、单序列、块长 128、最大长度 256、
`gpu-memory-utilization=0.35`，禁用前缀缓存，temperature=0，seed=123，
固定生成 32 token 并忽略 EOS。短提示词为 3 token，跨页提示词为 135 token。
每种输入先预热两次，再做三轮、每轮四请求，共每条路径 24 个计时请求。
另有每条路径 4 个预热请求、2 个算术/英文续写探针。

**72 个计时生成请求全部完成，加上预热与探针合计 90 个生成请求。**
每种格式内部的重复输出一致；三种格式之间的短提示词、跨页提示词及两个
附加探针的输出也完全一致。

| 输入 | 格式 | 请求数 | 总耗时 P50 / P95（秒） | TTFT P50（ms） | TPOT P50（ms） | 串行输出 token/s |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 3-token 短输入 | Basic | 12 | 1.987 / 2.104 | 132.31 | 59.73 | 16.06 |
| 3-token 短输入 | Enhanced | 12 | 2.091 / 2.222 | 142.91 | 62.74 | 15.15 |
| 3-token 短输入 | BTKV | 12 | 1.955 / 2.079 | 129.69 | 58.84 | 16.33 |
| 135-token 跨页输入 | Basic | 12 | 3.102 / 3.193 | 413.78 | 86.72 | 10.29 |
| 135-token 跨页输入 | Enhanced | 12 | 2.516 / 2.618 | 413.34 | 67.81 | 12.68 |
| 135-token 跨页输入 | BTKV | 12 | 2.562 / 2.682 | 413.26 | 69.29 | 12.44 |

跨页输入下，Enhanced 的总耗时中位数比 Basic 降低约 18.9%，BTKV 降低约
17.4%。短输入下没有一致的加速趋势，Enhanced 反而略慢。不同布局对写入和
读取的影响不同，算子级的加速不能直接等同于整个模型的加速。

这些是客户端观测的串行流式指标；TPOT 按首个非空内容到最后内容的时间及
输出 token 数估算。每格只有 12 个请求，P95 是描述性的小样本估计；模型按
Basic、Enhanced、BTKV 的顺序分别启动，未做多次独立重启和顺序轮换。因此
不能据此给相近的结果排名或外推生产并发能力。本轮没有测原生 BF16 后端基线。

跨页生成文本出现了北京/上海交替续写，三条路径完全相同。它说明当前三种
格式没有产生不同的模型输出，不代表语义准确率已经合格。本轮未测任务准确率、
perplexity 或真实长上下文检索质量。原始文本见 `raw/*-serving.json`。

## 容量与内存

| 格式 | KV token 槽位容量 | EngineCore 进程内存读数（MB） | 适配层元数据字节数 |
| --- | ---: | ---: | ---: |
| Basic | 377,344 | 22,081 | 3,054,128 |
| Enhanced | 377,344 | 22,081 | 3,054,128 |
| BTKV | 377,344 | 22,091 | 13,619,760 |

三者均为 FP8，KV 数据预算约为 20.16 GiB，所以分配器容量相同。
BTKV 额外分配描述符表，元数据增加 10,565,632 字节，与进程内存多约 10 MB
相符；当前描述符没有带来额外硬件计算收益。进程内存读数来自
`npu-smi info -t proc-mem`，不等于单独的 KV 数据占用，也不等于整机 HBM。
服务最大长度为 256，没有实际运行 377,344-token 的单请求。

## 复现与交付

工具文件：

- `ascend/compare_formats.py`：独立算子矩阵和共享地址控制选项。
- `ascend/format_compare_runtime.py`：本轮专用的模型测试适配。
- `ascend/compare_formats_client.py`：预热、流式测量及输出记录。
- `ascend/run_format_comparison.sh`：隔离容器的启动与停止。
- `ascend/summarize_format_comparison.py`：生成 CSV 和 JSON 摘要。

服务器记录目录：`/root/zhang/pspkv-format-compare-20261023`。
完整执行脚本见 `raw/run-operators.sh`、`raw/run_format_comparison.sh`、
`raw/run-control.sh`；启动、返回码和环境文件均在 [raw/](raw/)。
模型原始数据和统计摘要见 [serving.csv](serving.csv) 与 [summary.json](summary.json)。

本轮新增工具的适用 pre-commit 检查通过；没有对原项目全部代码重新做类型
检查，也没有修复此前记录的 39 个类型错误、融合 decode 接口或 FP4 缩放问题。
本轮源码和结果保存在当前工作区，尚未提交或推送新的 GitHub 版本。
