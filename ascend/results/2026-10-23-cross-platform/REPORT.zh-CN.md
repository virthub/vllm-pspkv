# 原始 CUDA 与昇腾参考计算的数值对照

2026-10-23 实测：108 个格式/负载/精度组合通过 CUDA—NPU 数值校验；
最大绝对误差为 **0.0009765625**，51 个组合的 BF16 输出逐位一致。
这是本次新增的原始 CUDA 设备函数对照，不是找到了历史 NVIDIA 测试日志。

## BTKV 的实现范围

原始代码中存在可执行的 BTKV 参考内核，但完整 TMA 优化路径没有接入该内核。
[原始包装代码](../../../csrc/attention/psp_kv_wrapper.cu) 第 345 行明确说明：
参考 BTKV 内核不消费 TMA 描述符，包装层仅检查描述符。
[BTKV 内核](../../../csrc/attention/psp_kv/btkv/paged_attention_btkv.cu)
使用普通 CUDA 加载、逐元素反量化和注意力计算。原仓库另有
`tma_descriptors.cu`，并纳入构建；存在描述符构造函数不代表当前注意力计算
已经使用 TMA。

昇腾的 `psp_attention_btkv` 同样只校验描述符形状/步长，之后调用公共注意力
实现，采用与 Enhanced 相同的转置 V 布局。这里没有模拟或移植 NVIDIA TMA。
因此“BTKV 跑通”应准确表述为“BTKV 参考布局、计算和接口兼容路径跑通”。

TMA 从 Hopper 架构已经提供，H100 即支持，并非 Blackwell 独有。
依据：[NVIDIA Hopper Tuning Guide](https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html#tensor-memory-accelerator)。
本次 RTX 4060 Ti 为 compute capability 8.9，仅运行不依赖 TMA 的参考计算。
原 README 和 CUDA 测试中仍保留 Hopper+ 的 BTKV 硬件要求/跳过条件；
本次直接编译执行参考设备函数，不经过该测试跳过条件，不能据此宣称已经
验证 README 中描述的硬件原生 BTKV 路径。

## 测试方法与复现

- CUDA 源码固定在迁移前提交 `f613ce1337fb4e8815cbf1c6d485d9096dd8d933`。
- 本机显卡 RTX 4060 Ti 16 GB，驱动 610.88；CUDA NVRTC 12.6.85，
  CUDA runtime headers 12.6.77，CPU PyTorch 2.14.1+cpu。
- 通过 NVRTC 编译原始文件中未改写的三个设备内核函数和原始量化头文件。
  排除 PyTorch 注册和原始 host launcher，使用 CUDA Driver API 按原 launcher
  的网格与参数约定启动。NVRTC 的虚拟 `math.h` 提供 NVIDIA 数学常量，
  函数由 NVRTC 内建实现；虚拟 `stdint.h` 提供固定宽度整数类型。
- 昇腾使用 worker03 空闲 NPU 2、原生库与先前格式测试一致；库 SHA256：
  `72f17c1718609e04e3fad08404d832ab7fb2553fc42520fc31b06f1de1428ed1`。
- 将 CPU 生成并打包、实际交给 NPU 的输入张量和 NPU 输出保存为同一份
  `fixtures.pt`，再将这些确切字节上传 CUDA 显存计算；不是两端分别随机生成。
- 既比较 CUDA—NPU，也比较 CUDA—CPU 参考输出，要求有限且通过
  `atol=1e-2, rtol=1e-2`。另统计所有元素最大误差；实测绝对误差均小于 1e-3。

负载为之前五组：GQA 长度 128/512/2048、MHA 长度 512、四请求尾页不满；
新增 Qwen2.5-7B 对应的 28 个 Q 头、4 个 KV 头、长度 128。
共六负载 × 六精度 × 三格式 = 108 项。head dimension 128、页长 16、
Query/输出为 BF16、单位缩放、随机物理页顺序。精度包括 BF16、E4M3、E5M2、
FP4、两种 FP8 混合、四种精度混合，精度表预先指定。
原始随机 KV 限制在 [-2.5, 2.5]，因此不覆盖动态 FP4 缩放缺陷。

| KV 精度 | 组合数 | CUDA—NPU 逐位一致数 | 最大绝对误差 |
| --- | ---: | ---: | ---: |
| BF16 | 18 | 3 | 0.00048828125 |
| E4M3 | 18 | 6 | 0.0009765625 |
| E5M2 | 18 | 12 | 0.000244140625 |
| FP4 | 18 | 9 | 0.00006103515625 |
| E4M3/E5M2 混合 | 18 | 9 | 0.0001220703125 |
| BF16/E4M3/E5M2/FP4 混合 | 18 | 12 | 0.00006103515625 |

原始输入留在服务器：
`/root/zhang/pspkv-cross-platform-20261023/fixtures.pt`。
本机副本：`/tmp/pspkv-cross-platform-data/fixtures.pt`。
文件 SHA256：`cc92570012b60b41780bf5c5fcdc6356d9a0cd467e6201be82cb158d2a302ac3`。
138 MB 张量文件未加入仓库；原始 CUDA 源码/头文件 SHA、逐项误差及编译参数
保存在 [cuda-reference.json](cuda-reference.json)，运行日志见
[cuda-run.log](cuda-run.log)。

导出脚本为 [export_format_fixtures.py](../../export_format_fixtures.py)，
CUDA 执行脚本为 [compare_cuda_reference.py](../../compare_cuda_reference.py)。
在有相同 NVRTC/headers 和 CUDA 驱动的本机复现：

```bash
.venv/bin/python ascend/compare_cuda_reference.py \
  --cuda-libs /tmp/pspkv-cuda-reference/.venv/lib/python3.12/site-packages/nvidia \
  --fixtures /tmp/pspkv-cross-platform-data/fixtures.pt \
  --output /tmp/cuda-reference.json
```

## 历史 NVIDIA 结果核查与结论边界

### 真实模型分布补充：原 CUDA E4M3 编码大值饱和

7B 测试中发现真实 K 最大绝对值 418。另使用 `--quantization-probe`
编译一个仅调用原始量化头文件的 CUDA 小内核，在同一 RTX 4060 Ti 上实测：
320、384、418、448 均被原始编码器编码为 `0x78` 并解码为 256；
PyTorch E4M3FN 对 418 的编码再解码为 416，对 448 为 448。
负值也有相应饱和。见 [cuda-quantization-probe.json](cuda-quantization-probe.json)
及 [quantization-probe.log](quantization-probe.log)。
昇腾编码代码使用相同的 exponent>=15 规则；这段范围的问题已存在于原 CUDA
参考量化器。主 108 项测试使用 CPU 打包的相同 KV 字节，且输入幅度限制为
2.5，检验的是消费缓存的注意力算术，因此不能发现在线编码器的这项问题。
这也说明“CUDA—NPU 注意力输出接近”不能单独作为模型质量验收标准。
7B 的 BF16 缓存控制恢复了基线输出，但尚未完成编码器修复与质量重测，
详见 [7B 报告](../2026-10-23-qwen25-7b/REPORT.zh-CN.md)。

### 可找到的原仓库材料

在本地可见分支和服务器已检查的 PSP-KV 文件中，没有找到可归属具体 GPU、
模型、命令和输入的历史 Basic/Enhanced/BTKV 原始运行结果或输出张量。
这不能证明作者从未运行过测试。

`origin/main` 的 `39c54d2be35230ce71d3d1762da751cc8426769c` 提交包含：

- `scripts/benchmark_psp_kv.py`：使用 CUDA Event 测三个算子入口及 SDPA，
  它是可执行测试脚本，文件本身不构成已运行的证据。
- `scripts/evaluate_memory_psp_kv.py`：在 CPU 上根据精度比例和布局公式算
  字节数，不是 NVIDIA 显存实测，也不是模型吞吐结果。

本次数值对照支持：在这些固定输入和已覆盖精度下，迁移后的算术结果与原始
CUDA 参考计算一致到小于 1e-3 的绝对误差。它不验证整个 CUDA 扩展的 host
注册/参数检查，不验证硬件 TMA，不构成 Qwen2.5-7B 的 NVIDIA 端完整模型
结果，也没有执行公平的跨硬件速度比较。自动精度策略、长上下文模型质量、
动态缩放和生产稳定性仍需各自的测试。

本轮新增对照工具的相关 pre-commit 检查通过，见
[pre-commit.txt](pre-commit.txt)。这不改变之前迁移核心代码仍有 39 个
mypy 错误的审计结论。本次未修复这些存量类型错误，也未修改生产内核。
