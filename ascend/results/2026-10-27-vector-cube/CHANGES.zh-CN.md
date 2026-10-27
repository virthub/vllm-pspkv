# 本轮实现位置与边界

## 长 prefill 编码

- [SIMD 编码与 tile writer](../../native/pspkv_vector_encode.h)：16-token × 128-dim，
  RNE、subnormal、饱和与 NaN 路径向量化；一个任务负责一组 K/V head tile。
  工作数据约 58 KiB/AIV，strided DMA 直接接 projection 的 K/V view。
- [native 入口](../../native/pspkv_fp8.asc)：注册 `strategy_scatter_prefill_vec`，
  检查 BF16 head view、dtype、device、alignment 和页面元数据。
- [策略接线](../../../vllm/v1/attention/backends/psp_kv_ascend_strategy.py)：
  固定 lossless/E4M3/E5M2 的 unchunked prefill 使用新入口；decode RoPE/scatter、
  自适应与 FP4 保留原有入口。布局名称与精度策略仍沿用 develop 的集合。

编码结果与旧 writer 字节一致。这一改动不改变量化策略，也不修改模型权重。

## Vector/Cube 注意力

[pspkv_cube.asc](../../native/pspkv_cube.asc) 是单独构建的可选库。

- BF16 Q，真实压缩 K/V byte cache，设备端 block table/length/code/offset/scale。
- 每个 KV head 处理整个 GQA 组。Qwen2.5-7B 中一次解码的 K/V 供七个 Q head 使用；
  Cube 的 M 维补齐至 16，额外行不进入最终输出。
- 每个 tile 跨四个 128-token 物理页，按页读取 code/offset/scale，再按 32 token 解码。
  每页元数据在 tile 内复用，不按每个元素读取。
- Vector 经 UB LUT/Cast/scale 得到 BF16；高阶 Matmul 在 Cube 上执行 QK、P×V；
  Vector 执行 FP32 online softmax、max/rescale/accumulate，P 转成 BF16 供第二次矩阵乘。
- split-K 按上下文 tile 分割，部分结果保持 FP32，使用稳定的 max/denominator 合并。
- 尾页在进入矩阵乘前清零无效 K/V；softmax 另外屏蔽无效 score。
  不能仅把概率设为零，因为 `0 × NaN` 仍会污染结果。
- 跨 DMA/Vector/Scalar 使用显式 HardEvent；对齐后的 score 尾部用向量 Duplicate，
  仅处理最多七个非对齐边界元素。
- 系统 workspace、tile 与 split 输出分别保留；split 输出由调用方持有，避免每层临时分配。
  workspace 按同一 forward、同一 NPU stream 复用。

两个 AIV 均参与工作：AIV0 解码 K 并执行 softmax/发起 Cube，AIV1 解码 V。
V 使用两个有界 GM slot，按单调 ticket 握手复用，使下一 tile 的 V 解码与当前注意力重叠。
在 P×V 异步执行期间，AIV0 还会把下一块 K 预取到已释放的 K slot；消费结果前调用
`WaitIterateAll`。这些路径已经通过正常调用和动态图重放检查。

每次图重放先清理私有跨核计数器。信号位各占独立 cache line，使用 volatile 标量访问与
显式 DCCI，避免将轮询值当成普通不变输入。工作区包含 16 MiB 系统区、每组 448 KiB
有界 tile/信号区和从 26 MiB 开始的 FP32 split 输出；当前单请求集成复用 32 MiB。

这是一条 **FP8 存储、BF16 Cube 计算** 的路径；没有使用 910B3 原生 FP8 矩阵乘。
高阶 API 通过有界 GM tile 连接 Vector UB 与 Cube L1，不等于完全没有 HBM 访问。
没有分配全上下文的 BF16 KV 临时页；GM tile 是否实际命中 L2 不能靠代码布局推定，
本轮没有 L2/HBM 硬件计数器证据。

## 调用与图重放

- [策略后端](../../../vllm/v1/attention/backends/psp_kv_ascend_strategy.py)
  增加 `PSPKV_STRATEGY_DECODE_PATH=cube` 与 `PSPKV_ASCEND_CUBE_LIB`。
- [图参数入口](../../../vllm/v1/attention/backends/psp_kv_ascend.py)
  对 `cube` 使用设备长度/表的验证路径，避免更新不存在的 vendor PA task。
- [启动脚本](../../run_nvidia_strategies.sh) 为可选 Cube 库增加只读挂载，
  `CUBE_LIBRARY_PATH` 必须显式指定。主对照默认仍使用 `paged`。
- 原有跨层元数据、设备长度、同流工作区、RoPE/scatter 与 decode graph 保留。
  不在 graph replay 前重复更新 vendor PA 参数。

## 检查工具

- [向量 writer 检查](../../check_vector_prefill_npu.py)：穷举 BF16 位模式，
  比较旧/新 writer 字节与 NPU event 延迟。
- [策略/Cube 检查](../../check_nvidia_strategies_npu.py)：新增可选 Cube 库、profile、
  缓存/源输入不变、NaN 尾页、workspace 边界及动态长度图重放检查。
- Cube 的可选 cycle 输出统计 decode、QK、softmax、P×V；这些区间包含各自搬运/同步，
  不能把 `QK` 区间直接称为 Cube 纯计算时间或硬件利用率。
- [构建器](../../native/build.py) 保留原 native 默认入口，新增 `--source pspkv_cube.asc`，
  使用当前 SDK 的 tiling/platform 库。

## 已修复的实验问题

初期把编译器调整后的用户 workspace 参数再次按系统区偏移，且将其直接当成系统指针，
导致越界与后续输入被改写。最终使用 `GetSysWorkSpacePtr()` 注册 Matmul，
用户 tile 从编译器提供的用户区起始位置计算；失败的输出不进入有效性能表。

扩大 tile 后，直接复用解码 UB 需要补齐 MTE/Vector/Scalar 之间的事件；
仅凭普通 pipeline barrier 的实验版本未通过完整数值检查，未接入有效版本。

默认主对照仍为 `paged`；新的 `cube` 路径通过显式环境变量选择。
本轮不会根据单个长度的 kernel 结果自动切换所有生产请求的路径，实际收益与限制见报告。
FP4 的编码和量化策略未改动，其模型质量问题不会因融合注意力而自动消失。
