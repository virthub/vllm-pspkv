# 本轮代码变化及策略边界

本轮在 Ascend 分支上新增显式开关 `PSPKV_NVIDIA_STRATEGIES=1`，对齐
NVIDIA `develop` 提交 `00b78e954225947fbd05281fc19bd9f22164f851` 的精度集合。
没有把整个 develop 合并进当前 vLLM-Ascend 依赖版本，也没有迁移 CUDA 指令本身。

## 保留什么，移出什么

主矩阵为 Basic / Enhanced / BTKV × lossless / E4M3 / E5M2 / FP4。
lossless 存储当前模型的 BF16 K/V，不代表模型权重改成 FP16。
幅值自适应另测，采用 develop 的 `E=1e-3, fp_max=240` 判定，选择 E4M3 或 FP4。
旧版 RMSE 门限、分组校准 FP4、K8/V4 等实验代码和历史报告保留，均不进入主矩阵。

当前 develop 的字节页内布局为 `[head, token, dimension]`，K/V 相同。
Ascend 的三种格式入口复用公共软件实现；它们不是三套独立的华为硬件算法。
Basic 可对应分阶段解码与统一注意力消费者；Enhanced / BTKV 在主对照中也使用
这条优化回退路径。独立测试另外覆盖直接读取压缩 tile 的 Vector 融合实现。
没有实现 CUDA TMA、Blackwell 原生 FP4 或新的 Vector/Cube 融合 FlashAttention。

## 存储与服务接线

- [strategy 模块](../../../vllm/v1/attention/backends/psp_kv_ascend_strategy.py)
  管理真实字节缓存、每页格式、字节偏移及每层缩放；没有为每层持有 BF16 缓存副本。
- [Attention 缓存规格](../../../vllm/model_executor/layers/attention/attention.py)
  和 [容量规格](../../../vllm/v1/kv_cache_interface.py) 按 2 / 1 / 1 / 0.5 byte
  配置 KV 分配器。工作区、缩放元数据及模型权重不属于这个元素字节数。
- [Ascend 后端](../../../vllm/v1/attention/backends/psp_kv_ascend.py)
  接入新路由和设备长度；分页路径委托 vendor 的图任务更新，直接融合路径读取设备元数据。
  继续限制 TP=1、单请求、BF16 权重、128 维 head、关闭前缀缓存和 chunked prefill。

## 编码及预填充

[pspkv_strategy.h](../../native/pspkv_strategy.h) 新增 develop 兼容编码器。
FP8 使用有限值饱和、RNE 和次正规数处理；FP4 使用 CUDA 位编码规则并将溢出饱和到 3。
保留旧编码器供原接口和历史实验使用，避免静默改变历史结果。

预填充每页、每 KV head 对 K/V 共用 max-abs 缩放，分母分别为 240 / 57344 / 3，
极小缩放回退到 1。缩放估计在设备执行；没有逐层读回 K/V 数值到 CPU。
自适应由第一层幅值选择物理页格式，跨层复用该选择，每层保留自己的缩放。

写入器直接读取 QKV 投影的 strided K/V 视图，省去 `stack/contiguous` 来源副本。
每个任务拥有完整打包行，避免 FP4 半字节写入竞争。
编码器目前仍有逐元素标量位运算；这会影响长 prefill，不能把向量解码优化说成编码也已完全向量化。

## Decode 优化如何复用

1. [shared_metadata.py](../../shared_metadata.py) 的跨层页表、slot 和设备长度复用继续开启。
2. [pipeline_runtime.py](../../pipeline_runtime.py) 将 RoPE 延后到缓存写入；
   [pspkv_rope.h](../../native/pspkv_rope.h) 在 UB 完成旋转、舍入、编码，覆盖四精度。
3. [pspkv_tiled.h](../../native/pspkv_tiled.h) 使用 Vector LUT 解码 FP8；
   FP4 用字节到双元素查表，替代逐元素标量解包。配对 LUT 提前生成并搬入 UB。
   UB 仅在一次 kernel 中有效，不是跨 kernel 的持久缓存。
4. 分阶段路径复用原有 `_forward_workspace`：同一 forward、同一设备流的各层依次
   覆盖同一对 BF16 活跃页工作区，再调用华为原生分页注意力。跳过完全不活跃的页，
   最后一页的无效 tile 清零。仍然存在每层解码和 GM 临时写回。
5. 直接融合路径逐 tile 读压缩 K/V、UB 解码、online softmax，并用
   [pspkv_split.h](../../native/pspkv_split.h) 合并 Split-K 的 FP32 部分结果。
   修复了 Vector 累加器写入 MTE3 前缺少屏障的问题。
   当前 QK/PV 在 Vector 上执行，长上下文性能不能代表 Cube 融合内核的潜力。
6. 两条路径均验证 decode 图重放。调用审计发现 `mode=3` 的 AOT 编译绕过 Python
   RoPE 适配器；因此最终候选使用 `mode=0` 捕获完整 decode 图，显式初始化原生分页注意力
   的图参数，继续由 vendor runner 在图发射后更新任务。没有在重放前重复更新任务。
   BF16 保留原生 `mode=3` 图模式，前一轮候选 `mode=3` 结果作为独立编译路径对照。

## 检查与复现入口

- [设备检查](../../check_nvidia_strategies_npu.py)：全 BF16 位模式编码、RoPE、混合页、
  不同长度与 Split-K、变长图重放，并计时原生 BF16、解码、分阶段注意力、融合注意力及 prefill 写入。
- [服务启动器](../../run_nvidia_strategies.sh)：同一卡按 BF16-before、四精度、自适应、
  BF16-after 的顺序执行；任何候选失败最终返回非零。
- [请求客户端](../../compare_nvidia_client.py)：两次预热后，交替测试 3 / 135 / 8192
  输入 token，每次固定生成 32 token，前缀缓存关闭。
- [统计脚本](../../summarize_nvidia_strategies.py)：中位数、基线前后漂移、独立 bootstrap
  速度比区间。没有把顺序请求伪称为配对采样。

具体测量值及可用性结论见本目录的 `REPORT.zh-CN.md`。
