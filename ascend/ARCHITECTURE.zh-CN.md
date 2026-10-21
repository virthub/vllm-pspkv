# PSP-KV 项目与昇腾迁移：从一次请求看各层如何工作

本文描述这个仓库当前的代码与已验证行为。根目录的 `README.md` 主要介绍原有
CUDA/Virthub 设计；昇腾 910B3 的实际启动方式和限制以 `ascend/README.md` 为准。

## 先理解 KV-cache

模型每处理一个 token，每个注意力层都会从该 token 的隐藏状态计算 Query、Key、
Value（Q/K/V）。下一步生成 token 时，新的 Q 要与此前 token 的 K 做相似度计算，
再用相似度加权此前的 V。如果每一步都重算所有旧 token 的 K/V，就会重复做大量
工作。KV-cache 把每一层、每个 KV 头、每个旧 token 的 K/V 留在设备显存中。

KV-cache 不是模型权重，也不是 HTTP 服务的缓存。一次请求结束后，调度器可以
回收它占用的物理块。当前测试禁用了前缀缓存，所以不会跨请求复用历史 KV。

常规 BF16 K/V 每个数各占 2 字节；FP8 各占 1 字节；两个 FP4 数可打包成 1
字节。在测试模型的一个 128-token 物理页中，K+V 的原始数据分别大约需要
512 KiB、256 KiB 和 128 KiB。比例只针对 KV 数据；模型权重、计算临时空间、
页元数据与 FP4 的缩放表仍占显存。vLLM 可能把省出的预算用于分配更多页，所以
`npu-smi` 看到的总显存占用不一定减半。

## 启动时的调用链

服务器上用于昇腾的入口是 `ascend/smoke_910b3.sh`，不是 CUDA 用的根目录
`run.sh`。脚本先让隔离容器检查所选 NPU 是否可见，再启动
`quay.io/ascend/vllm-ascend:v0.20.2rc1` 镜像，在容器中执行 `vllm serve
Qwen/Qwen3-0.6B`。它把本仓库只读挂载到 `/workspace/pspkv-src`，并通过
`PYTHONPATH` 让镜像使用这份 vLLM Python 代码。它还把一份经过补丁修改的
`vllm_ascend/platform.py` 挂载到插件位置；启用统一 FP4 时再挂载经过补丁修改的
`worker/model_runner_v1.py`。启用原生 FP8 时还会挂载预先编译的 Ascend C
共享库。安装在镜像内的插件本体没有就地修改。

简化调用链：

```text
Docker / ascend/smoke_910b3.sh
  -> vllm serve (vLLM CLI)
  -> API Server + AsyncLLM + EngineCore
  -> Scheduler / KVCacheManager / BlockPool
  -> vLLM-Ascend worker/model runner
  -> Qwen3 模型各层的 Attention
  -> vLLM-Ascend platform.py 选择 PSPKVAscendBackend
  -> psp_kv_ascend.py 的 PSPKVAscendImpl.forward
  -> psp_kv_ascend_reference.py 或 psp_kv_ascend_packed.py
  -> 默认：PyTorch 张量运算 -> torch_npu -> CANN -> 910B3
  -> 原生 FP8 开关：psp_kv_ascend_native.py -> torch.ops -> Ascend C -> 910B3
     （仅缓存写入/解码；注意力矩阵计算仍用 PyTorch/torch_npu）
```

CLI 实现在 `vllm/entrypoints/cli/main.py` 和 `serve.py`；API Server 位于
`vllm/entrypoints/openai/api_server.py`。例如客户端 POST
`/v1/completions` 时，`completion/api_router.py` 接收 JSON，
`completion/serving.py` 处理提示词与采样参数，然后调用
`AsyncLLM.generate()`。API 地址是调用模型的入口，模型本身不会主动去请求这个
HTTP 地址。`AsyncLLM` 把请求交给独立的 EngineCore；
`vllm/v1/engine/core.py` 的 `step()` 调用调度器、模型执行器，再把结果送回 API。

启动时，vLLM-Ascend 插件向 vLLM 注册 NPU 平台。原版插件不知道本仓库的
`PSP_KV` 后端，`ascend/patches/v0.20.2rc1-platform-psp-kv.patch` 才使显式
`--attention-backend PSP_KV` 解析到
`vllm.v1.attention.backends.psp_kv_ascend.PSPKVAscendBackend`。因此只把本仓库
放上服务器、或只安装 vLLM-Ascend，都不足以走到这个后端。

## 一个请求怎样找到自己的 KV

调度器主要记录“这个请求已计算多少 token、下次需要计算多少 token”，然后向
`KVCacheManager` 申请新的物理块。这里没有一个必须独立运行的“prefill 服务”
或“decode 服务”：提示词的多个 token 通常在一次或数次调度中计算，后续生成
通常一次一个 token；调度器用已计算 token 数追踪进度。

两个索引尤其重要：

- `block_tables[请求, 逻辑块号]`：把请求内部的第几个块映射到显存中的物理块号。
- `slot_mapping[本次 token]`：指出本次产生的 K/V 写进哪个物理块的哪个槽位。

假设块长 128，某请求的 144 个提示词 token 要占两个逻辑块；调度器可能把它们
放到物理块 23 和 7。第 0 个 token 写入 `23 * 128 + 0` 槽，第 128 个 token
写入 `7 * 128 + 0` 槽。注意力后端读历史时按该请求的 `block_tables` 找 23、7，
再按物理块的格式和缩放表解码。实际物理号由调度器决定，并不要求连续。

Qwen3 模型代码在 `vllm/model_executor/models/qwen3.py` 中计算每层的 Q/K/V
并调用 `Attention`。`Attention` 根据 `--attention-backend PSP_KV` 选择相应
实现。`psp_kv_ascend.py` 的 `forward()` 收到本次 Q/K/V、NPU 上的 KV-cache
张量，以及插件构造的 `slot_mapping`、`block_tables`、序列长度等元数据。
它把本次 K/V 编码并写入物理槽；注意力计算读该请求的历史 K/V，得到输出交还
模型。模型经过剩余层后生成下一个 token，调度器继续下一轮，直到停止条件。

提示词首次计算（prefill）时，参考后端用本次刚算出的原始 K/V 参与当前 token
的注意力，同时把它们编码进 cache。后续生成（decode）要反复读取已经编码的
历史 K/V，因此量化误差会在后续生成中体现。KV-cache 位于 NPU HBM；Python
保存的是张量引用、表和控制逻辑，实际张量运算由 `torch_npu` 交给 CANN。
默认参考路径没有专用的高性能昇腾融合算子。可选原生路径已有缓存写入和解码
算子，但它们是标量正确性实现，还不是高性能融合注意力算子。

## 当前在线推理到底怎样选择精度

启动命令的 `--kv-cache-dtype fp8` 让插件分配 `uint8` 容器；它本身不等于
“每个块一定按 E4M3 解释”。最终解释方式由当前启用的 PSP-KV 路径决定：

| 在线模式 | 如何选择 | 实际页大小 | 当前结论 |
| --- | --- | --- | --- |
| 默认 `e4m3` | 所有块按 E4M3 编码 | 固定 FP8 页 | 已运行模型，输出连贯 |
| `PSPKV_ASCEND_NATIVE_FP8=1` | 仍使用上述 FP8 格式策略；把 BF16→FP8 写入和 FP8→FP32 解码交给 Ascend C | 固定 FP8 页 | 910B3 字节级测试和 Qwen3-0.6B 跨页请求已通过；注意力计算仍是参考路径 |
| `PSPKV_ASCEND_REFERENCE_FP8=adaptive` | 新块先 E4M3；写入值绝对值超过配置阈值时，把整个物理块提升到 E5M2，并重编码已有值 | 固定 FP8 页；两种格式都占 8 bit | 已做参考路径验证；不是按模型质量选精度 |
| `PSPKV_ASCEND_UNIFORM_FP4=1` | 所有物理页一开始就指定为 FP4；每个块、每个 KV 头按 K/V 幅值选 2 的幂缩放，后续数值变大时重编码旧值 | 固定、真正打包的 FP4 页 | 已启动并完成跨页请求，但生成质量不合格 |
| `PSPKV_ASCEND_VARIABLE_PRECISION=1` | 页按平均字节预算分配；块写满后由完成块策略在原始 K/V 上选 FP4/E4M3/E5M2，按块维护偏移与缩放，压实回收空洞，超预算失败关闭 | 真正变长字节 arena | 已接入在线服务；CPU 单测通过；910B3 上 Qwen3-0.6B 短/跨页请求 HTTP 200、字节级 checker 通过、保守阈值下输出与固定 FP8 完全一致；FP4 需阈值 2.0 才被选中且输出乱码 |
| `+ PSPKV_ASCEND_NATIVE_VARIABLE=1` | 同上，但写入走 Ascend C `packed_scatter`，单 token 解码走 `packed_scores_vec`/`packed_weighted_value_vec`（UB 分块、8 行/批、Python 参考编解码器生成 LUT、FP4 走 256 项表） | 同上 | 910B3 实测：vec 与标量内核逐位一致；单步解码 947→132 ms（7.19×）；端到端 32 token 40.94→5.25 s（7.8×） |
| `+ PSPKV_ASCEND_STATIC_PRECISION=<map.json>` | 校准运行（`PSPKV_ASCEND_CALIBRATE=1`）冻结每层格式与 FP4 缩放，分配时即定格式：FP4 直接预留半尺寸，无完成块策略/scratch/重编码；缺失层回退动态策略 | 分配时确定变长页 | 910B3 实测：全 E4M3 map 输出连贯 32 token 3.08 s（13.3×）；全 FP4 map + `PSPKV_AVG_BYTES=0.5` 分配 5896 块（FP8 的 2.0×），输出乱码（与既有 FP4 结论一致） |

FP8 的提升逻辑位于 `psp_kv_ascend_reference.py` 的
`select_adaptive_fp8_blocks()`；环境阈值在 `psp_kv_ascend.py` 中读取。
E4M3 有更多有效数字位，E5M2 有更大的动态范围；这种规则主要是避免超出
E4M3 范围，不会节省 E4M3 相对 E5M2 的空间。

统一 FP4 的启动开关在 `psp_kv_ascend.py` 中读取；
`psp_kv_ascend_packed.py` 中的 `reference_packed_attention()` 根据本块
K/V 的最大绝对值选每头缩放比例、必要时重编码，然后调用 `PackedKVCache`
写入和读出两个数共一个字节的 E2M1 数据。
`PSPKVUniformFP4AttentionSpec.real_page_size_bytes` 负责告知 vLLM 每页所需
字节数，`get_kv_cache_shape()` 给出与之吻合的物理张量形状，插件 worker 补丁
确保 V 张量也按打包后的维度解释。三个地方必须一致，否则即使 API 能启动，
也可能分配错误的页或读错数据。

服务器实测同预算下 FP4 大约可容纳 754k 个 KV token 槽，FP8 大约 377k。
但同一 Qwen3-0.6B、同一提示词，FP8 生成连贯中文，统一 FP4 出现乱码。
目前不能断言具体原因是 FP4 本身、缩放粒度还是 FP4 路径中的其他错误；
**HTTP 200 和容量翻倍不代表模型质量合格。**

## 本次新增的“按块选精度”到哪一步

`psp_kv_ascend_policy.py` 是对**已完成的 KV 块**运行的独立参考策略。
它拿到该块全部 K/V 原值后，依次试编码和解码：

1. FP4 E2M1：每个 KV 头使用一个 K/V 共享的 2 的幂缩放比例；
2. FP8 E4M3；
3. FP8 E5M2；
4. 如果前三种的误差都超过阈值，则保留 BF16/FP16（在线路径用
   `allow_lossless=False` 禁止这一步，超过阈值时回退 E5M2，保证不超过
   FP8 的字节预算）。

误差是“原值与读回值的均方根差 / 原值均方根”，分别对各 KV 头计算，然后取
最差的那个头。选择第一个不超过 `max_relative_rmse` 的格式。选择函数是
`select_completed_block_precision()`；
`pack_completed_blocks_by_error()` 将多个块放入 `PackedKVCache`，产出每块
`precision_table`、实际 `block_offsets` 和 FP4 缩放表。不同精度的块在字节
arena 中占不同空间，这一步确实验证了变长布局，而不是仅把 FP4 放进一个
8-bit 或 16-bit 的固定槽。单测中的四块选择 `[FP4, E4M3, E5M2, BF16]`，
单个 K 或 V arena 占 36 字节；四块全 BF16 则是 64 字节。

### 在线变长服务路径（`PSPKV_ASCEND_VARIABLE_PRECISION=1`）

该策略已经接入在线服务，由 `psp_kv_ascend_arena.py` 的
`VariableArenaKVCache` 维护每个注意力层的一组扁平 uint8 字节 arena。调度器
按平均字节预算 `PSPKV_AVG_BYTES`（默认 1.0，即 FP8 最坏情况；0.625/0.75
在 128-token 块下合法）通过 `PSPKVAttentionSpec.real_page_size_bytes` 预算
每页字节，worker 补丁使 `get_kv_cache_shape` 返回的扁平 `(2, per_kv_bytes)`
形状能同时用于 K/V。后端自己维护每块的偏移和格式：

- 块在 slot 0 写入时开始新生命周期，在自由指针处预留 FP8 最坏情况大小，
  入站 K/V 先按 E4M3 写入，同时把当前打开块的原始值保留在每层的小型
  scratch 中；
- 块写满 128 个 token 时，立即在**原始值**上运行完成块策略（与独立参考
  一致，不是对已 E4M3 量化的值再量化），块保持 E4M3、改写为 E5M2 或打包
  成半尺寸 FP4；
- 位于 arena 尾部的 FP4 块回收其上半部分；空闲块留下的空洞在自由指针
  溢出时由全 arena 压实回收；
- 实际混合精度超过预算时**失败关闭**，抛出明确错误而不是写坏字节；
- 生命周期检测依据当前请求的 block_tables 与本次写入的 slot（单序列时
  精确）；slot 0 重写同一物理块即开始新生命周期。

读取路径按每块格式和偏移解码，FP4 块乘回缩放表；prefill 用当前投影 K/V，
decode 读压缩历史。`PSPKV_ASCEND_MAX_RELATIVE_RMSE`（默认 0.05）是策略
阈值，`PSPKV_ASCEND_FP4_GROUP_SIZE` 选择 FP4 缩放粒度，
`PSPKV_ASCEND_LOG_FORMAT_COUNTS=1` 打印每层格式计数与 arena 占用。该路径
是正确性参考实现（整块解码、bmm 注意力、压实按字节搬移），不是融合高性能
实现；仍假设单序列、无前缀缓存、无 KV transfer。

## Python、CUDA、Rust、昇腾各负责什么

- Python 的 vLLM API、调度器、模型执行器和注意力后端负责请求、模型运算、
  物理块分配、格式选择与元数据。当前昇腾路径主要在这一层实现参考算法。
- 原项目 `vllm/v1/attention/backends/psp_kv_attn.py` 调用
  `torch.ops.psp_kv.*`；这些 op 在 `csrc/attention/psp_kv_wrapper.cu` 注册，
  底层 `csrc/attention/psp_kv/**/*.cu` 是 NVIDIA CUDA 实现。它们不会因为
  把代码放到 910B3 服务器就自动变成昇腾算子。
- Virthub 是外部存储管理组件。仓库的 `install.sh` 可选下载到
  `third_party/virthub`，并在连接器工厂懒加载其 Python 接口；当前 checkout
  没有 `third_party/virthub` 源码。本次昇腾测试没有启用 KV connector，
  `PSPKVAscendImpl` 当前也显式拒绝 KV transfer，所以 Virthub 的 Rust
  存储、跨设备迁移等没有参与这些推理请求。
- 910B3 默认运行 PyTorch/torch_npu 参考读写与注意力；可选加载
  `ascend/native/pspkv_fp8.asc` 编译的 Ascend C 扩展，把 BF16 K/V 编成
  E4M3/E5M2 字节并按物理槽写入、按块格式解码历史 KV。该算子已在边界值、
  非 64 字节对齐的单核心回退、384-token 多核心样本以及 Qwen3-0.6B 实际推理中验证。
  它仍用标量全局
  内存访问；注意力的 QK、softmax、AV 仍由 PyTorch/torch_npu 完成。
  尚未实现与现有 CUDA kernel 对应的高性能融合注意力算子，因此当前性能
  数字不能作为基准。

## 迁移状态与下一步

已验证：插件能选中 PSP-KV Ascend 后端；固定页 FP8 推理；FP8 范围驱动的
E4M3/E5M2 提升；统一 FP4 的实际打包页和跨页请求；独立参考中的 BF16/FP8/
FP4 按块选择与变长 arena；原生 FP8 缓存写入/解码的多核心字节级正确性和
模型跨页推理；**在线变长服务路径**（`PSPKV_ASCEND_VARIABLE_PRECISION=1`）
的平均字节预算页池、完成块策略、按块格式/偏移/缩放的读写、slot 0 生命周期
重置、压实回收与溢出失败关闭（CPU 17 项单测、容器内测试、910B3 字节级
checker 均通过）。2026-10-17 在 910B3 上用 Qwen3-0.6B 实测：0.75 预算下
分配 3931 个物理块（FP8 最坏情况 2948 个）；阈值 0.05/0.5 全部选 E4M3，
0.25 有 16/28 层选 E5M2，输出与固定 FP8 路径逐字节一致；阈值提到 2.0 才
选中 FP4，此时输出乱码。

**原生变长算子路径**（`PSPKV_ASCEND_NATIVE_VARIABLE=1`）已实现并实测：
写入用 `packed_scatter`，单 token 解码用新增的 `packed_scores_vec` /
`packed_weighted_value_vec`——每批 8 个 token 行用 `DataCopy` 搬入 UB，
query 行按头只搬一次，字节经 Python 参考编解码器生成的 FP32 查找表解码
（E4M3/E5M2 各 256 项，FP4 表补零到 256 项；16 项小表的 UB 拷贝在 910B3
上不可靠），累加顺序与标量内核一致，因此 vec 结果与标量内核逐位相同。
910B3 实测：单步解码 947→132 ms（7.19×）；端到端 32 token 40.94→5.25 s
（7.8×）。已知数据搬运对齐约束：`DataCopy` 需要 32 字节粒度，概率行必须
上取整（否则最后一个 token 的概率丢失，实测会造成约 0.04 的输出偏差）。

**分配时静态精度预测器**（`PSPKV_ASCEND_STATIC_PRECISION`）已实现并实测：
校准运行（`PSPKV_ASCEND_CALIBRATE=1`）在完成块策略处记录每层选码与每
KV 头的 K/V 最大幅值；`ascend/calibrate_precision.py` 冻结成 JSON map
（每层格式、FP4 每头 2 的幂缩放、运行期乘 2 留余量并对越界值钳制）。
静态模式下格式在分配时确定：FP4 直接预留半尺寸，跳过完成块策略、
scratch 与重编码。910B3 实测：全 E4M3 map（27 层 E4M3 + 1 层 E5M2）
输出连贯、32 token 3.08 s（13.3×）；全 FP4 map + `PSPKV_AVG_BYTES=0.5`
分配 5896 个物理块（FP8 的 2.0×），无溢出，32 token 3.25 s，但输出
乱码——容量与分配时决策已验证，FP4 质量仍未解决。注意 vLLM 每个 KV
cache group 只有一个 page size，因此异构 per-layer 格式无法把 FP4 省的
容量转给需要 E4M3 的层。

FP4 质量调查（2026-10-17）：校准发现该模型 **K 与 V 幅值相差约 1000 倍**
（layer-0 的 K 每头 2 的幂缩放 32–256，V 为 0.25–0.5）。原先 K/V 共享
一个缩放，V 被 K 的大缩放除后约等于清零，这本身就是乱码的根因。静态 FP4
已改为 K/V 独立、可按 32 维分组的 `[2, KV heads, groups]` 缩放，校准日志
也按 (stream, head, group) 记录最大幅值。但即使独立+分组缩放，E2M1 只有
1 位尾数，对真实 K 仍过粗，全 FP4 服务输出依旧乱码；NPU checker 证明
分组静态路径本身正确（存储逐字节一致、解码与参考 1e-4 内一致），因此是
格式精度问题而非实现错误。非对称精度（K=E4M3、V=FP4）现已端到端实现：`PSPKV_AVG_K_BYTES` /
`PSPKV_AVG_V_BYTES` 分别定两流预算，spec 的每页字节数为
`k_page_size_bytes + v_page_size_bytes`，worker overlay
（`patches/v0.20.2rc1-model-runner-asym-kv.patch`）按预算比例拆分插件的
K/V 缓冲并暴露扁平字节 arena。910B3 实测 K=1.0 / V=0.5 时分配
**3931 个物理块（FP8 的 1.33×）**，K arena 515244032 字节、V arena
257622016 字节，混合 map 服务 HTTP 200。但质量仍不合格：V-only FP4
在每头缩放时输出乱码，改为每 32 维分组缩放后仅部分可读
（`首都理工大学science Science & ...`）。因此 E2M1 的 1 位尾数对该模型的
V 同样过粗。结论：该模型上保质量可用的昇腾配置仍是 FP8（E4M3/E5M2，
1 B/元素）；不对称分配器已可用，适用于 V 用 4 bit 仍能接受质量的模型或
新格式。

下一步：若必须省 KV 内存，需要新的数值格式（例如 FP4 加 per-channel 偏移
/更细缩放，或用 V 专用低比特加 K 的混合格式并改插件为 K/V 非等大缓冲），
并用真实任务指标验收；再接入 Virthub KV 转移；随后把仍在参考路径上的
多 token prefill/整块解码与 Python 策略替换为向量化/融合算子。
