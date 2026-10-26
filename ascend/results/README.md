# Ascend 实验结果索引

这些结果对应各报告列出的源码、配置与实验时间，不能视为所有精度和模型的通用结论。
`raw`、`before` 及其他实验目录中的源码快照属于历史证据，运行和修改代码请使用
`ascend/` 与 `vllm/v1/attention/backends/` 中的当前版本。

建议依次阅读：

1. [最新精度策略检查](2026-10-26-precision-audit/REPORT.zh-CN.md)：FP8 以外策略的完成情况，以及与最新 develop 的差距。
2. [最新固定 FP8 性能结果](2026-10-26-pipeline/REPORT.zh-CN.md)：直接写入、RoPE 融合、图重放的逐步收益与适用范围。
3. [对应代码修改说明](2026-10-26-pipeline/CHANGES.zh-CN.md)：每项修改的实现位置。
4. [跨层元数据优化](2026-10-25-metadata/REPORT.zh-CN.md)：元数据复用、同步消除与回归验证。
5. [Vector 与融合注意力](2026-10-24-vector-fused/REPORT.zh-CN.md)：较早阶段的算子优化。
6. [Qwen2.5-7B 三格式初始对照](2026-10-23-qwen25-7b/REPORT.zh-CN.md)。
7. [旧版 CUDA 跨平台对照](2026-10-23-cross-platform/REPORT.zh-CN.md)：仅适用于该报告固定的旧版提交。

仓库保留报告、原始请求/输出、配置、测试日志、导出的 profiler CSV 与代码快照。
未提交硬件采集二进制、profiler 数据库、大型 Chrome 时间线和动态库；它们仍在
各报告记录的华为服务器 `/root/zhang/` 原始实验目录中。
构建出的动态库使用报告中的 SHA256 标识，请按源码与工具链重新构建。
