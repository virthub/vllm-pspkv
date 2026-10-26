# Ascend 其他精度策略检查与最新 develop 差异

2026-10-26 在华为 Ascend 910B3 上重新检查。结论：固定 FP8 的受限快速路径已有
可复现的功能与性能结果；其他精度策略目前仍处于实验阶段，不能宣称全部迁移完成。
本轮还确认，当前 Ascend 编码与缩放逻辑没有完整对齐最新 NVIDIA `develop`。

## 本轮实际测试了什么

- Qwen2.5-7B **基础版**，BF16 权重，TP 1，单请求，128-token 页。
- 最大长度 256，显存预算比例 0.35，关闭前缀缓存，全部使用 eager。
- 输入 3 / 135 token，强制生成 **16 token**，temperature 0、seed 123。
- 每种配置先预热一次，然后两类输入各测两次，再测算术和英文首都两个探针。
- 共 10 次配置启动，70 个生成请求；其中包含一次校准启动。
- 物理卡 4：校准、BF16、动态保守策略、放宽门限的动态 FP4；
  卡 2：统一 FP4、静态 FP8、分组 FP4、K8/V4；卡 3：E5M2、自适应 FP8。

这是功能和输出抽查，每类输入只有两个计时样本，且不同配置分布在不同卡和时段。
以下时间不能作为正式性能排名，也不能与此前生成 32 token、开启图重放的主表直接比较。
本轮整模型使用普通 Ascend 精度路由；此前三格式适配器的固定 FP8 图优化没有自动应用到这些路由。

10 种配置均完成请求，客户端均记录 `complete=true`。这只说明服务可执行。
原始请求、文本、服务器日志、启动配置及校准表见 [raw-v2](raw-v2)，
机器可读结果见 [summary.json](summary.json)。

## 整模型结果

耗时为两次请求的 P50，单位秒。容量为分配器日志报告的逻辑 KV token 数。

| 精度策略 | KV token 容量 | 短输入 | 跨页输入 | 本轮输出观察与完成情况 |
| --- | ---: | ---: | ---: | --- |
| 原生 BF16 | 130,688 | 0.690 | 0.693 | 短句正常，作为独立进程对照 |
| 固定 E5M2 | 261,504 | 0.823 | 1.585 | 短输入出现重复外文，算术续写也退化 |
| 自适应 E4M3/E5M2 | 261,504 | 1.618 | 2.396 | 可以切换格式，短输入仍有重复退化 |
| 动态变长，平均 0.75 byte，门限 0.05 | 348,672 | 1.638 | 2.856 | 观察到 E4M3/E5M2，没有观察到 FP4；短句重复标点 |
| 动态变长，平均 1 byte，门限 2.0 | 261,504 | 1.604 | 2.710 | 完整页观察到 FP4；跨页输出发生变化，质量未验证 |
| 静态 E4M3 | 261,504 | 1.407 | 1.640 | 路由可执行，短句仍有重复标点 |
| 统一 FP4，K/V 共用每头缩放 | 523,136 | 2.050 | 2.140 | 短输入、跨页输入均明显乱码或重复 |
| 静态 FP4，K/V 独立每 32 维缩放 | 523,136 | 1.670 | 2.101 | 使用本次 7B 校准数据，仍明显退化 |
| 静态 K=E4M3、V=FP4，每 32 维缩放 | 348,672 | 3.708 | 4.121 | 重复、答非所问；尚未达到可用质量 |

校准启动本身为动态变长、平均 1 byte、门限 0.05，容量 261,504，
短/跨页时间为 1.600 / 2.792 秒；它用于收集校准值，没有加入策略优劣比较。

FP4 分组表来自本次 Qwen2.5-7B 校准，覆盖全部 28 层、4 个 KV 头，
K/V 分别保存 4 组缩放，未沿用旧 0.6B 模型的数据。
校准仅有两份已完成页观测/层，并非大规模校准集，因此不能代表最优 FP4 质量。
校准工具日志的 E4M3/E5M2 计数表示原始选择结果；静态测试实际读取表内的
`k_code` / `v_code` 强制覆盖，三张表分别是 `(1,1)`、`(3,3)`、`(1,3)`。

输出例子：统一 FP4 在短输入后产生 `SingleOrDefault` 等无关字符串和重复片段；
分组 FP4 的 `1+1=` 探针续写成多行零；K8/V4 的首都探针混入无关内容。
BF16 在相同探针上也不是完整准确率基准：本轮仅观察续写，未测困惑度、任务得分或长文本质量。

## 容量与格式选择不能混为一谈

1. 固定 E4M3 与 E5M2 都是每元素 1 byte，自适应两者不会再增加容量。
2. 统一/静态 FP4 每元素 0.5 byte，报告容量约为 BF16 的四倍；输出质量尚不合格。
3. K8/V4 平均每个 K/V 元素 0.75 byte，报告容量约为 BF16 的 2.67 倍。
4. 动态平均 0.75 byte 的容量由预算公式计算。本轮保守门限只选到 FP8，
   没有证明实际数据能填满全部逻辑页；达到容量上限可能触发 arena 溢出保护。
5. 放宽门限确实让完整页选择 FP4，但本例平均预算仍为 1 byte，所以分配器报告容量仍是两倍。

动态策略只在页完成后选择精度。3-token 输入加 16-token 生成没有写满 128-token 页，
因此动态 FP4 配置的短输入实际仍走 FP8，不能用这行短输入证明 FP4 的质量。
日志中的格式集合是多次快照的并集，不能相加为全模型占比；`format_counts()` 只统计 K，
K8/V4 的 V 格式须结合静态表和启动配置判断。

## 算子检查通过项与失败项

本地 CPU 回归 **71 passed**。NPU 的原始 correctness、mixed precision、copy 测试
**263 passed**，包含 Basic / Enhanced / BTKV 参考兼容布局，以及预先指定的
16/8/4-bit 和混合格式。它们证明指定输入和误差范围内的算子行为，不能替代整模型质量测试。

独立 NPU 检查的完整结果见 [checks-summary.json](raw-v2/results/checks-summary.json)：

| 检查 | 结果 |
| --- | --- |
| FP8 编解码、自适应转换、块复制 | 通过 |
| 变长 arena 生命周期、回收/搬移、参考注意力 | 通过 |
| Packed 混合格式、无损存储、统一 FP4 等综合检查 | 在 FP4 小数值缩放检查处失败 |
| 原生变长/向量化/融合综合检查 | 向量与标量一致、普通原生 decode 通过；融合入口参数错误 |

失败细节保留在 [NPU 日志](raw-v2/results/npu-checks.log)，本轮没有修改这些实现或隐藏失败：

- `reference_packed_attention()` 使用 `pow(2, ceil(log2(...)))` 产生缩放。
  NPU 上期望的 0.25 实际为 `0.2499999850988388`，而 CPU 得到精确 0.25。
  后续范围比较会将这个略小的值再次放大，`run_uniform_fp4_small_values()` 失败。
  相同形式的计算还存在于完成页精度选择器中，需要一起修复并验证边界。
- `reference_variable_attention(native_fused=True)` 传入 `value_precision_table`
  和 `value_block_offsets`，而 `psp_attention_scaled_basic_fused()` 不接受这两个参数，
  抛出 TypeError。默认普通原生变长路由没有开启这个开关，因此本轮服务请求仍可完成。
  不能因此声称动态精度/K8V4 已接通固定 FP8 的融合快速路径。

初次组合检查在遇到异常后仍执行了 pytest，因此外层退出码可能为零。
本轮额外使用逐项捕获、最终汇总失败的诊断脚本，确认上述两个失败并未被 263 项通过掩盖。
原始组合日志见 [npu-pytest-and-initial-checks.log](npu-pytest-and-initial-checks.log)。

## FP16、BF16 及其他策略

- 独立 packed 存储支持 code 0 的 BF16/FP16，无损存储和三布局注意力有算子测试。
  BF16/FP16 都占 2 byte；两者之间切换不构成相对 BF16 的 KV 容量压缩。
- 在线变长 arena 只容纳 FP8/FP4。选择器在预算约束下禁止回退 16-bit，
  最坏回退到 E5M2，即使实际误差超过门限。因此门限不是整模型质量保证。
- 当前 Ascend 服务没有实现 FP8 加 residual、按头剪枝、完整 Virthub 搬移或硬件原生 BTKV。
  NVIDIA 结构体中存在字段也不等于端到端算法已接通；最新 sidecar 仍写入空 residual 指针和全头掩码。
- 当前 FP4 是仓库现有软件打包和缩放路径，不能表述为已实现硬件原生 NVFP4/MXFP4。

本轮整模型是在共享精度路由上审计，没有对九种策略再重复 Basic/Enhanced/BTKV 三套整模型测试。
三种布局与压缩精度属于不同维度；已通过的三布局算子测试不表示各布局的全部精度快速路径均已完成。

## develop 已下载，当前尚未合并

最新取回的提交是 `00b78e954225947fbd05281fc19bd9f22164f851`，提交时间
2026-10-24 11:17:46 +08:00。完整源码位置：

- 本机 Git worktree：`/home/zmh/workspace/vllm-pspkv-develop-20261026`。
- 华为源码快照：`/root/zhang/vllm-pspkv-develop-00b78e9`，`SOURCE_COMMIT` 记录完整提交号。

`ascend` 工作区没有混入这个分支。对照清单见 [develop-updates.txt](develop-updates.txt)，
相关源代码差异见 [develop-source.patch](develop-source.patch)。本轮没有运行新版 CUDA 测试。
旧的 108 项 CUDA/NPU 对照固定在 `f613ce1`，不能作为本次 `00b78e9` 的对齐证明。

主要变化与迁移缺口：

| 最新 NVIDIA 代码变化 | 当前 Ascend 状态 |
| --- | --- |
| FP8 使用原生 E4M3/E5M2 转换 | 仍保留旧编码，部分边界行为不同 |
| FP8/FP4 按块/头缩放，量化使用 x/scale，读取乘 scale | 固定 FP8 快速路径使用单位缩放，没有完整对齐 |
| FP4 编码溢出钳位到 code 5，即 3.0 | Ascend 旧编码溢出到 code 6，解码得到 Inf |
| FP16、无损 BF16、FP8 与 FP8-query Tensor Core decode | 没有对应的完整 Ascend Cube 路径 |
| 异步流水加载、分块 prefill 量化、head dim 256、滑窗等扩展 | 当前优化验证范围不能覆盖这些功能 |

CPU 编码诊断显示，当前 Ascend 的 E4M3 对 418 编码再解码得到 256，
标准 E4M3FN 转换得到 416；这是源码差异的一个可复现例子。
见 [codec-diagnostic.json](codec-diagnostic.json)。这份诊断没有执行 NVIDIA CUDA。
最新 FP4 解码表仍将 code 6/7 解释为 Inf，变化在编码器的钳位行为，不能误写为新版已把 4/6 当作有限数。

## 后续应优先补齐什么

1. 固定最新 develop 为参考，先对齐 FP8/FP4 编码、缩放与越界行为，再做 CPU/NPU/CUDA 共用数据对照。
2. 修复 FP4 精确幂缩放和动态融合入口；验证 K/V 独立元数据，而非只消除参数报错。
3. 扩大校准集，测试困惑度/任务准确率、长上下文及饱和 arena，确认质量和真实容量。
4. 在数值与质量合格后，将元数据复用、直接写入、RoPE/图重放扩展到支持的精度组合。

本轮提交保留已完成的固定 FP8 优化和这些精度检查工具，不把尚未完成的策略标记为完成。

## 复现与证据

CPU：

```bash
.venv/bin/python -m pytest -q --confcutdir=tests/v1/attention \
  tests/v1/attention/test_psp_kv_ascend_reference.py \
  tests/v1/attention/test_psp_kv_ascend_packed.py \
  tests/v1/attention/test_psp_kv_ascend_policy.py \
  tests/v1/attention/test_psp_kv_ascend_arena.py \
  tests/v1/attention/test_psp_kv_shared_metadata.py \
  tests/v1/attention/test_psp_kv_pipeline.py
```

服务启动按 [run_precision_audit.sh](../../run_precision_audit.sh) 的冻结版本执行，
服务端目录为 `/root/zhang/pspkv-precision-audit-v2-20261026`。
静态三策略须先运行 calibration 生成校准表，卡号须选择空闲设备。
脚本需要对应容器镜像、驱动挂载和 vendor overlay；本次实际 overlay 快照及源码 SHA 已保存到 raw-v2。
原生库 SHA256 为 `482c2e32311b74685df8dd09165720dd6098b2523f5b719a84aa5aeca886c712`。
原共享库和服务端原始项目没有被本轮覆盖。

初次健康检查受服务器本机代理配置影响，未执行模型探针；修正为回环地址绕过代理后，
在独立 v2 目录重新启动全部配置，上表只使用 v2 的完整结果。
服务端 arena 源码与本机文件仅排版不同，Python AST 相同，见
[arena-source-equivalence.json](arena-source-equivalence.json)。

汇总：

```bash
.venv/bin/python ascend/summarize_precision_audit.py \
  ascend/results/2026-10-26-precision-audit/raw-v2 \
  --output ascend/results/2026-10-26-precision-audit/summary.json
```
