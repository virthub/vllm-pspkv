# 本轮改动与实现方式

## 直接读取独立 K/V

改动位于 [native 算子](../../native/pspkv_fp8.asc) 的
`packed_scatter_kv` / `PspkvPackedScatterParallel<true>`，以及
[模型适配器](../../format_compare_runtime.py)。

原来每层用 `torch.stack((key, value), dim=1).contiguous()` 创建来源张量，
写入器再从这个副本读取、编码、写入压缩页。现在直接传入 K、V 和各自的
token/head stride，兼容 QKV 投影切片产生的非连续视图。缓存布局、页偏移、
单位缩放及 FP8 编码规则保持原样。

原堆叠接口保留为对照和兼容路径。三种布局继续沿用原来的缓存行归属：
K 和 Basic V 按 token/head 分工，Enhanced/BTKV 的转置 V 按 dimension 分工，
避免多个核同时修改同一缓存行。单次调用的有效 slot 必须唯一。

## 减少不必要的输出初始化

适配器不再清零即将被注意力完整覆盖的有效输出行。有 padding 时仅清理
额外行；空输入仍返回零输出。验证中先将输出填成 NaN，再检查计算结果，
避免把未写入内容误认为正确输出。

## 将单 token RoPE 与缓存写入合并

核心实现在 [pspkv_rope.h](../../native/pspkv_rope.h)，入口为
`rope_scatter_kv`。每个 Q/K head 和每个转置 V 维度行由唯一核处理。

1. 把当前位置的 cos/sin 和当前 Q/K head 搬到 UB。
2. 使用 Vector 指令在 FP32 中完成 NeoX 旋转，再按原规则舍入到 BF16。
3. Q 写回原视图，供后续注意力使用。
4. K 在 UB 舍入后直接编码写入 FP8 页，省去旋转后 K 的显存回写与重读。
5. V 在同一个 kernel 内写入对应缓存布局。

这是一项 Vector 融合，没有把整层注意力或 MLP 融合进去。UB 中的数据仅在
本次 kernel 内复用，不能把它视为跨 kernel、跨层的持久缓存。

[pipeline_runtime.py](../../pipeline_runtime.py) 在 Qwen2 的单 token decode
入口延后 RoPE，让注意力适配器调用融合写入器。必须拦截模型入口：插件注册的
`torch.ops.vllm.npu_rotary_embedding` 已持有原函数，仅重绑定模块中的函数名
不会改变实际算子。最终验收同时检查生成结果和 profiler 中的 `PspkvRopeScatter`。

支持范围限定为单 token、BF16 Q/K/V、完整 NeoX RoPE、head dim 128、
128-token 页、固定单位缩放 E4M3/E5M2。Prefill、QK norm、其他 RoPE 类型和
缺少新算子的旧动态库保留兼容路径。此次没有改变权重或修复原有 FP8 编码规则。

## 跨层复用活跃页工作区

[packed ops](../../../vllm/v1/attention/backends/psp_kv_ascend_ops.py) 的
`_forward_workspace` 和 native 的 `gather_paged_bf16_vec_out` 配合使用。

较长上下文仍使用 Vector 解码加原生分页注意力。原来每层分配两个 BF16 活跃页
张量；现在同一个 forward、同一个设备流、相同形状只分配一对工作区。
上一层注意力在该流上消费完毕后，下一层覆盖工作区。不同设备流使用不同条目，
下一次 forward 重新建立所属关系。

各层 KV 内容仍需分别解码；复用的是存储空间，并非复用不同层的 K/V 数值。
这不是全量 BF16 缓存副本，也没有消除长上下文路径的临时 BF16 显存往返。
没有在本轮新增一体化长上下文 Vector/Cube attention kernel。

## Decode ACL Graph

修改涉及 [后端边界检查](../../../vllm/v1/attention/backends/psp_kv_ascend.py)、
`pipeline_runtime.py`、适配器、[启动器](../../run_format_comparison.sh) 和
[测试入口](../../serve_initialized_npu.py)。

图内保留压缩 KV 页读写、动态设备长度、动态 slot 和完整页表。长度与页表内容
由 runner 在每步更新；图不会固化捕获时的一页裁剪结果。重放前核对元数据的
设备地址，地址变化则拒绝重放。图外不再逐层执行 Python 包装、分配临时张量和
提交单个 kernel。

当前自定义整模型图支持范围限制为 TP 1、单请求、BF16、head dim 128、
最大长度 256，继续关闭前缀缓存。较长上下文的本轮对照使用 eager。正常参考
后端的 eager 限制保留，只有显式启用的独立三格式适配器允许这条图路径。

该插件版本的 BF16 原生图需要 `mode=3` 初始化注意力任务参数。
自定义路径使用 `mode=0` 加 ACL Graph，kernel 每步直接读设备元数据，不依赖
vendor 的 CPU 注意力任务更新。两条路径都实际验证图重放。

## 对照实验

[四阶段客户端](../../compare_pipeline_stages.py) 在同一个加载后的模型内，
对每组输入随机排列四种模式，逐个完成请求后再切换：

| 模式 | 改动 |
| --- | --- |
| 0 | 原 eager 路径，已有跨层元数据复用保持开启 |
| 1 | 直接 K/V 写入、有效输出免清零；适用长输入时复用工作区 |
| 2 | 模式 1 加单 token RoPE/写入融合 |
| 3 | 重放捕获的模式 2 decode 图，prefill 保持原路径 |

BF16 对照使用独立进程的原生 eager/图模式，避免把图收益误算为 FP8 独有收益。
原生 BF16 的同进程旁路实验出现重复输出，已排除并增加边界检查；独立原生启动
的图模式通过了与 eager 的输出对照。切换文件仅用于自定义图的实验，
每个 forward 固定一次模式；普通环境变量模式没有文件轮询。

[统计脚本](../../summarize_pipeline_stages.py) 分别计算 P50/P95、TTFT、TPOT、
配对均值的 bootstrap 区间，以及生成文本和 token logprob 差异。不能把四种模式
混合统计，也不能把对自身 eager 的加速当成对 BF16 图模式的加速。

独立算子测试覆盖 90 组写入、48 组 RoPE/写入、21 次动态元数据图重放，以及
9 组工作区场景、36 次跨层输出比较。另有 CPU 71 项和原 NPU 回归 263 项通过。
少量短句/常识探针不能替代完整准确率或困惑度评估。
