# 2026-10-20 昇腾迁移实机复核

本次在 `worker03` 的 `/root/zhang/vllm-pspkv` 实测。结论：FP8 推理、在线变长存储、静态精度映射和部分 Ascend C 算子已可运行；FP4 质量、可选融合 decode 接口和 NPU 缩放边界仍有问题，不能判定为完成生产迁移。

## 环境与代码

- 主机：Ascend 910B3；服务测试串行使用空闲 NPU 4，补充诊断使用空闲 NPU 2。
- 镜像：`quay.io/ascend/vllm-ascend:v0.20.2rc1`。
- Python 3.11.15、torch `2.10.0+cpu`、torch_npu `2.10.0`、vLLM `0.20.2+empty`、vLLM-Ascend `0.20.2rc1`。Python 使用服务器已有的 uv 虚拟环境。
- 远端 Git HEAD 为 `dd2cf6d`，昇腾迁移代码有未提交和未跟踪内容，不能仅用提交号复现工作树。
- [source-manifest.json](source-manifest.json) 中核对的 30 个迁移相关文件，本地与远端 SHA256 全部一致。这不是整个仓库所有文件的一致性声明。
- 模型为 Qwen3-0.6B，TP=1、eager、块长 128、最大长度 256、单序列、禁用前缀缓存、显存利用率参数 0.35。

## 测试结果

原始输出保存在 [raw/](raw/)，可复现的客户端与服务驱动在本文所在目录；服务器原件在 `/root/zhang/pspkv-audit-20261020`。

| 检查 | 结果 | 证据 |
| --- | --- | --- |
| FP8 CPU/NPU 编解码、混合格式、提升及页复制 | 通过 | [fp8_reference.log](raw/fp8_reference.log) |
| 打包存储综合 NPU checker | 失败：FP4 小数值缩放与 CPU 不一致 | [packed_reference.log](raw/packed_reference.log) |
| 在线变长 arena、压实、生命周期及参考注意力 | 通过 | [variable_reference.log](raw/variable_reference.log) |
| 原生 FP8 写入、读回、单 token 注意力、多核心与非对齐样例 | 通过 | [native_fp8.log](raw/native_fp8.log) |
| 原生变长综合 checker | 前两部分通过，在可选融合 decode 接口处失败 | [native_variable.log](raw/native_variable.log) |
| 五个昇腾 CPU 测试文件 | 59 passed | [cpu_tests.log](raw/cpu_tests.log) |
| 原有算子正确性测试，`PSPKV_TEST_DEVICE=npu` | 161 passed | [correctness.log](raw/correctness.log) |
| 原有混合精度测试，`PSPKV_TEST_DEVICE=npu` | 97 passed | [mixed_precision.log](raw/mixed_precision.log) |
| 补测静态 FP4、K/V 独立分组 FP4、非对称 K/V 原生 decode | 与参考实现一致，容差 `1e-4` | [diagnostics.log](raw/diagnostics.log) |

317 项 pytest 用例通过，不包含两个失败的独立 checker；不能把 pytest 通过写成全部验证通过。补充诊断没有修改或屏蔽原始 checker 的失败结果。

## 七种在线服务配置

每种配置执行六个生成请求：预热、中文续写、简单算术、英文续写、135-token 跨页提示词、重复中文续写。共 42 个生成请求，全部成功；每条配置的跨页输入均为 135 token，重复中文请求输出均与第一次相同。

| 路径 | 分配器报告的 KV token 容量 | 重复生成 32 token 的耗时 | 本次输出观察 |
| --- | ---: | ---: | --- |
| 原生 vLLM-Ascend BF16 基线 | 188,672 | 1.750 s | 短句可读 |
| PSP-KV 原生固定 FP8 | 377,344 | 2.009 s | 短句可读 |
| PSP-KV 原生在线变长，预算 1.0 | 377,344 | 3.771 s | 短句可读，观测到 E4M3/E5M2 两种块格式 |
| PSP-KV 原生静态 FP8 map | 377,344 | 3.003 s | 短句可读 |
| PSP-KV 原生统一 FP4，每头共享缩放 | 754,688 | 4.628 s | 明显乱码和混杂字符 |
| PSP-KV 原生统一 FP4，K/V 独立逐维缩放 | 754,816 | 12.858 s | 短句可读，跨页回答仍不稳定 |
| PSP-KV 非对称静态 K=FP8/V=FP4，每 32 维分组 | 503,168 | 7.083 s | 中英文混杂，明显质量退化 |

完整输出见 [summary.json](summary.json) 及 `raw/*-responses.json`，参数见 [run-serving.sh](run-serving.sh) 和 `raw/*-inspect.json`。

这些耗时来自每条路径的一次重复短请求，包含 prefill、decode 和 HTTP 开销，只用于现场诊断，不是正式吞吐或延迟基准。首次相同请求有额外初始化耗时，未用来填表。跨请求精确复现说明此组样例中的块复用稳定，不证明所有生命周期边界均已覆盖。

容量是约 20.16 GiB KV 数据预算下的分配器槽位数，未完整计入后端另外创建的缩放表、元数据和 scratch，不代表整机 HBM 实际节省。逐维缩放 FP4 相比每头缩放还需要更多缩放元数据。不同运行可能因预算取整差一页。

FP8 各路径和 BF16 的生成文本并不全部相同；变长路径在长提示词下会选 E5M2，固定路径默认 E4M3。静态 map 也固定了特定层的格式。本次结果不支持“所有 FP8 模式输出逐字节一致”的笼统结论。所有可读性描述都是少量原始 completion 的人工观察，没有测 perplexity、任务准确率或长上下文检索质量。

## 两个明确的问题

### 1. 融合 decode 的 Python 接口没有跟随非对称 K/V 参数更新

`reference_variable_attention()` 给向量和融合两个分支都传入 `value_precision_table` 和 `value_block_offsets`。`PSPKVAscendPackedOps.psp_attention_scaled_basic_fused()` 尚未接收这两个参数，启用融合分支立即出现 `TypeError`，发生在调用 NPU kernel 之前。

因此 `PSPKV_ASCEND_NATIVE_FUSED=1` 当前不能算可用。普通 `PSPKV_ASCEND_NATIVE_VARIABLE=1` 分支已经接受这些参数，完成了本次服务测试；静态与非对称参考一致性也补测通过。

### 2. NPU 上的幂函数误差会触发 FP4 缩放过度提升

按顺序写入 0.125 和 0.75 时，CPU 的缩放为 0.0625 → 0.25，NPU 为约 0.062499996 → 0.49999997。对应参考注意力输出分别约为 0.4375 和 0.375，已经超过单纯元数据末位差异。

诊断中 NPU 的 `pow(2, -2)` 得到约 0.249999985；当前算法在幂计算后检查 `max_abs > 3 * required`，这个微小的向下误差会让刚好位于边界的值再提升一档。这是本次失败的直接数值证据，见 [diagnostics.log](raw/diagnostics.log)。

完成块策略的另一个 CPU/NPU 样例中，格式、偏移和缓存字节完全一致，但缩放差约 `7.45e-9`，读回 K/V 差约 `1.49e-8`；原 checker 要求所有张量严格相等，也会在后续这一处失败。必须区分可接受的浮点末位差异与前述会改变缩放档位、实际改变结果的问题。

## 完成情况与建议顺序

已完成可运行原型：插件选择后端、真正的 FP8/FP4 字节存储、在线变长 arena、静态格式映射、K/V 不同预算、部分 Ascend C 写入和 decode 算子，以及单卡单序列的在线推理。

尚未完成验收：高质量低于 FP8 的实际压缩、相对原生 BF16 的性能优势、可选融合 decode、NPU 缩放边界、完整任务质量评测、长上下文与高占用压力测试。当前后端还显式限制 TP=1、eager，变长/FP4 路径禁用前缀缓存且只允许单序列；KV transfer/Virthub 未接入。

建议先修复两个可复现的问题，保持 FP8 为基线，再做 FP4 量化策略和任务质量评测，最后扩展性能、长上下文、多序列和 KV transfer。不要仅依据 HTTP 200、slot 容量或旧文档的“已通过”判断完整迁移完成。

## 源码重建

通过 `PSPKV_NATIVE_BUILD_DIR=/root/zhang/pspkv-audit-20261020/native-rebuild bash ascend/native/build_in_container.sh` 在独立目录重建，退出码 0。编译器报告 92 条 warning，主要涉及 `cce_global` 属性；原有共享库没有被替换。

新旧共享库的 SHA256 完全相同：`72f17c1718609e04e3fad08404d832ab7fb2553fc42520fc31b06f1de1428ed1`。这次重建复现了之前执行服务测试的二进制。

对重建库再次运行原生 FP8 检查器、补充静态/非对称诊断、161 项正确性测试和 97 项混合精度测试，均通过，见 [rebuilt-status.tsv](raw/rebuilt-status.tsv) 和 `raw/rebuilt-*.log`。重复执行的 258 项没有计入前文 317 项的独立用例数。

本次没有修改项目实现或停止已有业务服务；所有本次创建的测试容器最终均已停止，NPU 2 和 4 均无运行进程，见 [final-containers.tsv](raw/final-containers.tsv) 和 [final-npu-smi.log](raw/final-npu-smi.log)。服务器项目的测试前后 `git status --short` 一致。报告与诊断脚本属于本次新增的审计材料。
