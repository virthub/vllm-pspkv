# Ascend 实验结果索引

这些结果对应各报告列出的源码、配置与实验时间，不能视为所有精度和模型的通用结论。
`raw`、`before` 及其他实验目录中的源码快照属于历史证据，运行和修改代码请使用
`ascend/` 与 `vllm/v1/attention/backends/` 中的当前版本。

建议依次阅读：

1. [向量编码与 Vector/Cube 流水线](2026-10-27-vector-cube/REPORT.zh-CN.md)：prefill 标量瓶颈、双 Vector 核、V 双缓冲、K 预取，以及整模型同卡消融。
2. [本轮代码修改说明](2026-10-27-vector-cube/CHANGES.zh-CN.md)：实际实现、UB/GM 工作区、动态图及仍未完成的硬件路径。
3. [NVIDIA 策略矩阵与长上下文对照](2026-10-26-nvidia-strategies/REPORT.zh-CN.md)：四精度、独立自适应、8192-token 输入及阶段瓶颈。
4. [策略接线代码修改说明](2026-10-26-nvidia-strategies/CHANGES.zh-CN.md)：新策略路由、编码、RoPE、向量解码和图执行。
5. [前一轮精度策略审计](2026-10-26-precision-audit/REPORT.zh-CN.md)：旧实验策略的完成情况，以及当时与 develop 的差距。
6. [固定 FP8 性能结果](2026-10-26-pipeline/REPORT.zh-CN.md)：直接写入、RoPE 融合、图重放的逐步收益与适用范围。
7. [对应固定 FP8 代码修改说明](2026-10-26-pipeline/CHANGES.zh-CN.md)。
8. [跨层元数据优化](2026-10-25-metadata/REPORT.zh-CN.md)：元数据复用、同步消除与回归验证。
9. [Vector 与融合注意力](2026-10-24-vector-fused/REPORT.zh-CN.md)：较早阶段的算子优化。
10. [Qwen2.5-7B 三格式初始对照](2026-10-23-qwen25-7b/REPORT.zh-CN.md)。
11. [旧版 CUDA 跨平台对照](2026-10-23-cross-platform/REPORT.zh-CN.md)：仅适用于该报告固定的旧版提交。

仓库保留报告、原始请求/输出、配置、测试日志、导出的 profiler CSV 与代码快照。
未提交硬件采集二进制、profiler 数据库、大型 Chrome 时间线和动态库；它们仍在
各报告记录的华为服务器 `/root/zhang/` 原始实验目录中。
构建出的动态库使用报告中的 SHA256 标识，请按源码与工具链重新构建。
