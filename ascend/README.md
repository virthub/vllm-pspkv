# Ascend 910B3 reference path

This directory provides a bring-up route for PSP-KV FP8 and uniform FP4 reference formats
on vLLM-Ascend `v0.20.2rc1` with the vLLM fork at `dd2cf6d`. The vLLM fork
contains the new `psp_kv_ascend.py` backend and its pure PyTorch reference
implementation. The patch in `patches/` changes the plugin's backend selector
to choose that backend when `--attention-backend PSP_KV` is requested.

The plugin continues to allocate KV pages and build the slot mapping and
block tables. The reference backend encodes K/V into one-byte E4M3 entries
by default, writes the physical slots, reads the logical blocks, decodes
them, and runs causal GQA. Prefill uses projected K/V directly for current
tokens; decode uses compressed K/V. The reference implementation also accepts
an optional physical-block precision table (1=E4M3, 2=E5M2) so its read/write
path can handle mixed FP8 blocks. The plugin does not yet produce that table
or implement a block precision policy, so normal serving uses E4M3 unless the
reference backend is placed in `adaptive` mode.
When the plugin copies an existing physical page, the PSP-KV backend copies its
per-layer FP8 format entry with the K/V bytes. In mixed-format modes, it fails
closed if the format table has not been registered. This handles the current
fixed-stride FP8 cache; the standalone variable-stride copy below is not yet
connected to the plugin callback.
For an end-to-end format check, set `PSPKV_ASCEND_REFERENCE_FP8=e5m2` or
`PSPKV_ASCEND_REFERENCE_FP8=alternating` when invoking `smoke_910b3.sh`. The
second mode assigns E4M3/E5M2 by physical block ID and is only a diagnostic
pattern; it is not a precision policy based on model data.
Set `PSPKV_ASCEND_REFERENCE_FP8=adaptive` to select the format from incoming
K/V. A physical block starts in E4M3. If any new K/V component has absolute
value above `PSPKV_ASCEND_E4M3_MAX_ABS` (default 240, the maximum finite
value before this repository's E4M3 encoder emits its overflow code), the
block promotes to E5M2.
Existing E4M3 bytes in that block are transcoded before the new K/V write, and
subsequent reads use the updated physical precision table. The block stays
E5M2 until the allocator starts a new lifetime by writing slot offset zero.
This policy also promotes for infinities; NaNs are encoded by both formats.
It is a reference data-range policy, not the Virthub allocation-time
predictor. Already-quantized older values cannot regain precision during a
promotion. The reference path assumes contiguous writes within each block,
`--no-enable-prefix-caching`, and no KV transfer. E4M3 and E5M2 both take one
byte per element, so adaptive mode does not save additional memory over fixed
FP8. It performs NPU-to-host synchronizations and is unsuitable for a
performance benchmark.
The fixed-FP8 and uniform-FP4 serving routes use fixed-stride pages. The
opt-in variable-width route described below connects a compact FP4/FP8 arena
to the live allocator and scheduler. It still has no online lossless BF16
format, Virthub transfer, TMA equivalent, or fully fused NPU attention
kernel. The dynamic format policy is evaluated only when a block completes.

An opt-in Ascend C extension in `native/` now compiles the BF16-to-FP8 cache
scatter and mixed E4M3/E5M2 cache decode into one shared library. Set
`PSPKV_ASCEND_NATIVE_FP8=1` to use it while keeping the existing reference
attention calculation (`torch.bmm` and softmax). The native kernels use scalar
global-memory accesses, so this stage establishes a native execution path but
is **not** the high-performance attention implementation. It is limited to
BF16 model K/V, fixed-width FP8 pages, eager mode, and the other serving limits
above. The extension compiled for `dav-2201` and passed the NPU byte-for-byte
check, including a 24-token unaligned-stride fallback and a 384-token/
1024-element multi-core scatter across nonadjacent physical pages. On a 910B3,
Qwen3-0.6B returned HTTP 200 for both a 3-token
prompt and a 149-token prompt crossing a 128-token page boundary; EngineCore mapped the
native shared library. The normal reference path remains the default.

Set `PSPKV_ASCEND_NATIVE_FP8=1 PSPKV_ASCEND_NATIVE_DECODE=1` to opt into
direct compressed-cache single-token attention. An Ascend C score kernel reads
E4M3/E5M2 K bytes from physical pages using the block table; PyTorch computes
softmax on the padded score rows; a second Ascend C kernel reads compressed V
bytes and produces the weighted sum. No full-context FP32 K/V tensors are
created for these queries. Multi-token prefill still uses the reference path.
These kernels currently use scalar global-memory access and are a correctness
milestone, not a throughput claim. The standalone NPU checker compares their
scores and outputs against decoded-cache PyTorch attention for mixed physical
formats, GQA, a partial last page, and 48 query heads across 40 AI Cores.

The native extension also exposes the original PSP-KV packed-cache operator
contract through `psp_kv_ascend_ops.py`: basic, enhanced and BTKV attention,
packed quantize/scatter, packed dequantization, and the basic path's uniform
attention consumer. The byte format is selected per physical block by the
caller's `precision_table`: `0` is uncompressed BF16/FP16 (or target FP8), `1`
is E4M3, `2` is E5M2, and `3` is packed E2M1 FP4. `block_offsets` identifies
each variable-width page, and `scale_table` supplies per-head FP4 scales when
needed. The CUDA kernels also **consume** these tables; they do not choose a
format from model error inside the attention kernel. The original
`test_psp_kv_correctness.py` and `test_psp_kv_mixed_precision.py` run on NPU
with `PSPKV_TEST_DEVICE=npu` and the native library path set. These native
kernels for basic, enhanced and BTKV remain scalar correctness
implementations. The live variable-width route uses the packed scatter
kernel and vectorized packed score/value kernels for single-token decode;
its block format and byte offset are managed by `VariableArenaKVCache`.

An experimental attention dispatch can be enabled with
`PSPKV_ASCEND_SDPA_ATTENTION=1` when starting `smoke_910b3.sh`. It replaces
the reference `bmm`/softmax calculation with PyTorch SDPA, allowing
torch_npu to select its fused attention implementation where supported. It
still decodes the selected KV pages to FP32 first, so this is not a fused
compressed-KV kernel and does not avoid the temporary decoded tensors. The
CPU test checks causal chunk-prefill parity against the eager reference; the
actual torch_npu dispatch and performance must be measured on 910B3. Keep the
flag off if SDPA is unsupported by the installed torch_npu build.

Build and validate the native extension on the server without changing the
installed vLLM-Ascend package:

```bash
bash /root/zhang/vllm-pspkv/ascend/native/build_in_container.sh
NPU_ID=<free_npu_id> \
  bash /root/zhang/vllm-pspkv/ascend/native/check_in_container.sh
```

The second command checks scatter bytes and decoded values against the
PyTorch reference for both FP8 formats, including zeros, subnormal values,
overflow, infinities and NaNs. It also checks the single-core fallback for
token strides smaller than a 64-byte cache line and a multi-core, model-sized
layout. The checker also compares native single-token attention with decoded
PyTorch attention for GQA and mixed-format pages. Once it passes, start a model
with the native cache path using a fresh port and container name:

```bash
NPU_ID=<free_npu_id> HOST_PORT=18085 \
  CONTAINER_NAME=pspkv-ascend-native-fp8-test \
  PSPKV_ASCEND_NATIVE_FP8=1 \
  PSPKV_ASCEND_NATIVE_DECODE=1 \
  bash /root/zhang/vllm-pspkv/ascend/smoke_910b3.sh
curl -sS --max-time 120 http://127.0.0.1:18085/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-0.6B","prompt":"北京是中国的","max_tokens":16,"temperature":0}'
```

This response establishes that the extension participated in cache writes
and single-token compressed-cache attention. It does not establish good
throughput or final model quality.

`psp_kv_ascend_packed.py` is the next, standalone storage reference. It builds
byte offsets from an allocation-time physical precision table: `0` stores
BF16/FP16 without quantization (two bytes), `1`/`2` store FP8 E4M3/E5M2 (one
byte), and `3` packs FP4 E2M1 (two values per byte). Its K and V byte arenas
use the same offset table. K is head-major and token-major; V can be
token-major or transposed `[head, dimension, token]`. The existing CUDA packed
writer uses token-major V, while its enhanced attention reader indexes
transposed V. This layout mismatch must be resolved when integrating the
packed path; the standalone reference tests both layouts explicitly. The reference
refuses FP4 values outside [-3, 3] at unit scale to avoid silently writing an
infinity code. A four-block test layout `[E4M3, FP4, E5M2, FP4]` occupies
48 bytes per K/V arena versus 64 bytes with all-FP8 blocks; a second test
mixes BF16, FP8 and FP4 blocks. The CPU and 910B3 checks compare byte-for-byte
contents and logical block reads, including the transposed FP4 byte order.
The standalone cache can copy pages with matching formats using their byte
offsets, including overlapping source/destination mappings. It rejects direct
copies between different formats. Its `repack` operation allocates a new
compact arena when formats change, copies unchanged pages byte-for-byte and
re-encodes changed pages; the old arena remains intact. Re-encoding can lose
precision, and unit-scale FP4 rejects values outside [-3, 3].

`psp_kv_ascend_policy.py` adds a quality-gated **completed-block** reference
on top of that compact storage. For each full K/V block it tries FP4 E2M1
with a shared power-of-two K/V scale per head, then unit-scale FP8 E4M3 and
E5M2. It selects the first format whose worst KV head has relative K/V root
mean square reconstruction error at or below `max_relative_rmse`; otherwise
it preserves the model's BF16/FP16 values (`allow_lossless=False` keeps the
online arena within the FP8 budget and falls back to E5M2). The packer
returns the selected precision codes, per-head scales and a `PackedKVCache`
with actual mixed-width byte offsets. A four-block BF16 fixture selects
`[FP4, E4M3, E5M2, BF16]`; one K or V byte arena occupies 36 bytes versus
64 bytes at all-BF16. The CPU tests and 910B3 checker compare every selected
code, byte, offset and decoded value. This threshold measures local KV
reconstruction error, not final model quality; the synthetic `0.001` test
threshold is not a production setting. This reference runs **after** all
K/V values in a block are available. The live variable-width route
(`PSPKV_ASCEND_VARIABLE_PRECISION=1`, below) calls the same selection on the
block's original values as soon as its last token is written, so it is a
completed-block policy rather than an allocation-time predictor.

The mixed-format, variable-stride `PackedKVCache` reference is **not itself
used by `vllm serve`**; the live variable-width route keeps its own
per-layer `VariableArenaKVCache` because live blocks appear and disappear
dynamically. The opt-in serving route instead assigns FP4 to every page
at allocation time. `PSPKVUniformFP4AttentionSpec` budgets half as many bytes
per page as FP8; the backend exposes a correspondingly packed physical shape
and reads the same bytes using the standalone codec. Its per-physical-block,
per-KV-head power-of-two scales live in a separate float32 tensor. A new page
chooses its scale from the incoming K/V magnitude, including magnitudes below
one; existing values are requantized if a later write needs a larger scale.
Writing slot zero starts a new page lifetime. Prefill uses current projected
K/V directly, while decode uses the packed history. The page still has a fixed
FP4 stride; this route does not choose BF16, FP8 or FP4 for individual
blocks.

`PSPKV_ASCEND_UNIFORM_FP4=1 PSPKV_ASCEND_NATIVE_FP4=1` opts that separate
uniform-FP4 serving route into native packed scatter and direct single-token
compressed-cache attention. It retains the Python scale-promotion policy and
multi-token prefill attention, so it is a partial native model path rather
than the mixed-width scheduler. With a 128-multiple head dimension, the
token-major native FP4 writer can split 64-byte-aligned token rows across up
to 40 AI Cores. The standalone transposed FP4 layout still uses one core to avoid
writes to a shared byte. This route
requires the rebuilt library, BF16 model K/V, eager mode, block size 128,
prefix caching disabled, and one sequence at a time. The scale is chosen per
physical block and KV head; the **precision code stays FP4 for every page**.
For a separate test container, use `smoke_910b3.sh` with both variables set
and an idle NPU/unused host port.

On 2026-10-17, the native FP4 route was measured on one 910B3 with
Qwen3-0.6B. The same benchmark produced 7.83 output tokens/s with native FP4,
16.32 with native fixed-width FP8, and 18.26 with the stock BF16 backend.
Native FP4 was 3.88 times faster than this repository's PyTorch FP4 reference
(2.02 output tokens/s), but it was slower than both usable alternatives. The
tested greedy prompt also produced corrupted text with **both** FP4 paths.
The complete commands, results, cache capacities and limits are in
[`results/910b3/README.md`](results/910b3/README.md). These measurements do not
establish a production speedup or acceptable FP4 model quality.

Variable-width serving is now available as an opt-in online route:
`PSPKV_ASCEND_VARIABLE_PRECISION=1`. The worker allocates two flat uint8
byte arenas per layer using an average bytes-per-element budget
(`PSPKV_AVG_BYTES`, default 1.0 = FP8 worst case; 0.625 and 0.75 are valid
with 128-token blocks and any even head dimension). Each physical block
starts a lifetime on a slot-0 write with an FP8-size reservation, stores
incoming K/V as E4M3, and keeps the open block's original values in a small
per-layer scratch. When a block receives its 128th token, the
completed-block policy from `psp_kv_ascend_policy.py` runs on those
original values and keeps the block as E4M3, rewrites it as E5M2, or packs
it to half-size FP4; an FP4 block at the arena tail reclaims its upper
half. `PSPKV_ASCEND_MAX_RELATIVE_RMSE` (default 0.05) is the policy
threshold, and `PSPKV_ASCEND_FP4_GROUP_SIZE` selects the FP4 scale
granularity. The lossless BF16 format is never selected online because it
would exceed the FP8 byte budget.

The scheduler still reserves a single `page_size_bytes` per block, but now
that size is the *average* budget, and the backend's `get_kv_cache_shape`
returns a flat `(2, per_kv_bytes)` shape that the patched plugin worker
accepts. Blocks are placed by the backend at its own offsets inside the
arenas; the plugin never indexes a fixed-shape page tensor. Holes left by
freed blocks or mid-arena FP4 shrinks are reclaimed by a whole-arena
compaction that runs when the free pointer would overflow. If the actual
precision mix still exceeds the budget, allocation fails closed with a
clear error instead of corrupting bytes. The route assumes one sequence at
a time, no prefix caching, no KV transfer, and contiguous increasing slot
writes, like the uniform FP4 route. `PSPKV_ASCEND_LOG_FORMAT_COUNTS=1`
logs per-layer format counts and arena occupancy after each step.

```bash
NPU_ID=<idle_npu_id> HOST_PORT=18086 \
  CONTAINER_NAME=pspkv-ascend-variable-test \
  PSPKV_ASCEND_VARIABLE_PRECISION=1 \
  PSPKV_AVG_BYTES=0.75 \
  PSPKV_ASCEND_MAX_RELATIVE_RMSE=0.05 \
  bash /root/zhang/vllm-pspkv/ascend/smoke_910b3.sh
curl -sS --max-time 120 http://127.0.0.1:18086/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-0.6B","prompt":"北京是中国的","max_tokens":16,"temperature":0}'
```

With `PSPKV_AVG_BYTES=0.75` the scheduler budgets 0.75 bytes per KV element
instead of 1.0, so the same HBM budget yields about 1.33x the KV slots of
the FP8 route when the completed blocks are FP4-heavy. A workload whose
blocks all stay E4M3 overflows the arena and fails closed; lower the
threshold or raise the budget for such a workload. The arena is a
correctness reference: `gather_blocks` and the bmm attention path decode
whole blocks, and compaction copies live bytes. This is not a fused
high-performance implementation.

On 2026-10-17 the route was measured on one 910B3 with Qwen3-0.6B and
`PSPKV_AVG_BYTES=0.75` (3931 physical blocks versus 2948 at the FP8 worst
case). The short and 135-token cross-page prompts returned HTTP 200 with
coherent output, and the standalone NPU checker matched CPU byte-for-byte
for storage and within 1e-6 for the bmm/softmax attention math. The
completed-block policy kept every block in E4M3 at thresholds 0.05 and
0.5, selected E5M2 for 16 of 28 layers at 0.25, and only selected FP4
when the threshold was raised to 2.0 - at which point the output became
incoherent, consistent with the uniform FP4 finding. The amax-normalized
policy metric makes FP4 hard to select for real KV values; with this
model the online route therefore reproduces the FP8 path exactly at
conservative thresholds and does not yet yield per-block memory savings
without a quality loss. The byte-identical output against the fixed FP8
reference for the same prompt is the correctness baseline to preserve
when tuning thresholds.

## Native packed-kernel path (`PSPKV_ASCEND_NATIVE_VARIABLE=1`)

The variable-width route has an opt-in native path. Writes go through the
Ascend C `packed_scatter` kernel with the arena's per-block byte offsets,
formats and scales; single-token decode attention runs through
`packed_scores_vec` and `packed_weighted_value_vec`, which stage 8 token
rows at a time into UB with `DataCopy` instead of one global load per
element, load the query row once per head, and decode cache bytes through
FP32 lookup tables generated by the Python reference codec. The tables
make every decoded value bit-identical to the scalar kernels, and the
arithmetic order (per-position dot, per-dimension accumulation) is
unchanged, so the vectorized results match the scalar kernels exactly.
Multi-token prefill keeps the eager reference path, like the fixed-FP8
native route. `ascend/check_variable_native_npu.py` verifies the vec
kernels against the scalar kernels bit-for-bit, the native decode against
the reference at 1e-4, and a static FP4 arena against the reference.
`ascend/bench_variable_npu.py` times one decode step with Qwen3-0.6B
geometry on 910B3.

Measured on one 910B3 (Qwen3-0.6B geometry: 16 query heads, 8 KV heads,
head_dim 128, 28 layers, 256-token context, one decode token):

| path | per decode step | steps/s |
| --- | --- | --- |
| eager reference (`native=False`) | 947 ms | 1.1 |
| native packed kernels (`native=True`) | 132 ms | 7.6 |

The native path is **7.19x** faster for the attention portion. End-to-end
`vllm serve` on the same card with a 135-token prompt and 32 greedy output
tokens: the reference variable path took 40.94 s (0.78 tok/s), the native
dynamic-policy path 5.25 s (6.1 tok/s, **7.8x**), and the native static
E4M3 map 3.08 s (10.4 tok/s, **13.3x**). Short-prompt output stays
coherent (`首都，也是中国最大的城市。...`) and the E4M3 static run matches
the fixed-FP8 baseline text. The checker
(`ascend/check_variable_native_npu.py`) confirms the vectorized scores and
weighted-value kernels are bit-identical to the scalar kernels, the native
decode matches the reference path, and a static FP4 arena matches the
reference decode.

## Allocation-time static precision (`PSPKV_ASCEND_STATIC_PRECISION`)

A calibration run freezes per-layer formats so the precision decision is
made at allocation time instead of at block completion. Serve with
`PSPKV_ASCEND_VARIABLE_PRECISION=1 PSPKV_ASCEND_CALIBRATE=1` and send
representative prompts; the completed-block policy then logs one
`PSP-KV calibration layer=... code=... k_max=... v_max=...` line per
finished block. Freeze the map with:

```bash
docker logs <container> > /tmp/calibration.log 2>&1
python ascend/calibrate_precision.py \
  --log /tmp/calibration.log --output /tmp/precision_map.json
```

The map records the most common code per layer (ties resolve to the
larger format) and, for FP4 layers, per-KV-head power-of-two scales from
the largest sampled magnitudes. Serving with
`PSPKV_ASCEND_STATIC_PRECISION=/tmp/precision_map.json` then:

- assigns every block its format when it is allocated, so an FP4 block
  reserves half the bytes immediately - no FP8 worst-case reservation, no
  completion-time re-encode, no scratch retention, no compaction churn;
- multiplies the calibrated scale by two at runtime for one
  power-of-two of headroom and clamps unseen outliers before encoding so
  they never become the E2M1 infinity code;
- keeps the completed-block policy only for layers missing from the map
  (logged once).

Because vLLM budgets each KV cache group with one page size, the
capacity win requires the calibrated formats to be uniform enough that
`PSPKV_AVG_BYTES` can cover them: an all-FP4 map with
`PSPKV_AVG_BYTES=0.5` doubles the KV slots, while a map that mixes E4M3
and FP4 layers needs the FP8 worst case and saves execution overhead but
not capacity.

Measured on one 910B3 with the calibration workflow above:

- Calibrating at `max_relative_rmse=0.05` froze 27 layers as E4M3 and one
  as E5M2. Serving with that map at `PSPKV_AVG_BYTES=1.0` allocated the
  same 2948 physical blocks as the FP8 budget, kept the output coherent,
  skipped the completed-block policy entirely, and served 32 greedy
  tokens in 3.08 s (10.4 tok/s).
- Calibrating at `2.0` froze all 28 layers as FP4 with per-head scales.
  Serving that map at `PSPKV_AVG_BYTES=0.5` allocated **5896 physical
  blocks (2.0x the FP8 route)** with no overflow and served 32 tokens in
  3.25 s. The generated text was incoherent, matching the earlier
  uniform-FP4 and dynamic-FP4 findings. The allocation-time machinery and
  the doubled capacity are verified; FP4 model quality for this model is
  not.

The static path therefore delivers the allocation-time precision decision
and removes completion-time re-encoding when calibration is trustworthy;
whether it saves capacity in practice still depends on the model
tolerating the calibrated format.

### FP4 quality investigation

Calibrating with `PSPKV_ASCEND_CALIBRATE_GROUPS` (1 = per KV head,
4 = per 32 dimensions) exposed the cause of the earlier FP4 failures:
**K and V magnitudes differ by roughly 1000x** on this model (layer-0 K
power-of-two scales 32-256 versus V scales 0.25-0.5). The original
format shared one scale between K and V, so V was divided by K's large
scale and rounded to zero, which alone garbles attention. Static FP4 now
takes independent `[2, KV heads, groups]` scales (`{"k_code": 1,
"v_code": 3}` selects asymmetric K/V precision; `groups` selects the
dimension-group granularity), and the calibration log records
per-(stream, head, group) maxima.

Even with independent and per-32-dimension scales, E2M1's one mantissa
bit remains too coarse for this model's K: serving an all-FP4 map still
produced incoherent text. The NPU checker confirms the grouped static
path is correct (bit-exact storage, 1e-4 decode parity with the
reference), so the loss is the format itself, not the implementation.

Asymmetric K/V precision (K=E4M3, V=FP4) is implemented end to end.
`PSPKV_AVG_K_BYTES` and `PSPKV_AVG_V_BYTES` size the two arenas
independently, `PSPKVAttentionSpec` budgets `k_page_size_bytes +
v_page_size_bytes` per block, and the worker overlay
(`patches/v0.20.2rc1-model-runner-asym-kv.patch`) scales the plugin's
K:V buffer split by those budgets while exposing flat byte arenas. On
910B3 with `PSPKV_AVG_K_BYTES=1.0 PSPKV_AVG_V_BYTES=0.5` this allocated
**3931 physical blocks (1.33x the FP8 route)** with a 515244032-byte K
arena and a 257622016-byte V arena, and the mixed map served HTTP 200.

Quality, however, still fails: V-only FP4 with calibrated per-head V
scales produced incoherent text, and per-32-dimension V scales improved
it only to partially broken output (`首都理工大学science Science & ...`).
E2M1's one mantissa bit is therefore too coarse for this model's V as
well, not just K. On this model the useful, quality-preserving Ascend
configuration remains FP8 (E4M3/E5M2) at 1 byte per element; the
asymmetric allocator is available for models or formats where 4-bit V
is acceptable.

To stage the worker overlay on a fresh server, apply the platform, FP4
worker, variable-space worker and asymmetric-K/V worker patches in that
order (see the patch list below).

The overlay is already staged at
`/root/zhang/pspkv-ascend-overlay/v0.20.2rc1/vllm_ascend/platform.py` on the
current server; do not reapply the patch there. The uniform FP4 and
variable-width routes also need an overlay copy of the plugin worker because
vLLM-Ascend v0.20.2rc1 otherwise reshapes V using the logical full head
dimension even after the page budget and backend shape have changed. On a
fresh server, stage both overlay files without changing the installed plugin.
For an existing stopped test container:

```bash
mkdir -p /root/zhang/pspkv-ascend-overlay/v0.20.2rc1/vllm_ascend
docker cp \
  pspkv-fork-dd2cf6d-npu2:/vllm-workspace/vllm-ascend/vllm_ascend/platform.py \
  /root/zhang/pspkv-ascend-overlay/v0.20.2rc1/vllm_ascend/platform.py
patch --directory=/root/zhang/pspkv-ascend-overlay/v0.20.2rc1 \
  --strip=1 \
  < /root/zhang/vllm-pspkv/ascend/patches/v0.20.2rc1-platform-psp-kv.patch
mkdir -p /root/zhang/pspkv-ascend-overlay/v0.20.2rc1/vllm_ascend/worker
docker cp \
  pspkv-fork-dd2cf6d-npu2:/vllm-workspace/vllm-ascend/vllm_ascend/worker/model_runner_v1.py \
  /root/zhang/pspkv-ascend-overlay/v0.20.2rc1/vllm_ascend/worker/model_runner_v1.py
patch --directory=/root/zhang/pspkv-ascend-overlay/v0.20.2rc1 \
  --strip=1 \
  < /root/zhang/vllm-pspkv/ascend/patches/v0.20.2rc1-model-runner-fp4.patch
patch --directory=/root/zhang/pspkv-ascend-overlay/v0.20.2rc1 \
  --strip=1 \
  < /root/zhang/vllm-pspkv/ascend/patches/v0.20.2rc1-model-runner-variable.patch
patch --directory=/root/zhang/pspkv-ascend-overlay/v0.20.2rc1 \
  --strip=1 \
  < /root/zhang/vllm-pspkv/ascend/patches/v0.20.2rc1-model-runner-asym-kv.patch
```

First check `npu-smi info` and choose an idle, usable NPU. This server's card
assignments change over time; check again before each run. Supply the selected
NPU explicitly:

```bash
NPU_ID=<idle_npu_id> HOST_PORT=18083 \
  bash /root/zhang/vllm-pspkv/ascend/smoke_910b3.sh
```

For example, a separate mixed-format test can use
`PSPKV_ASCEND_REFERENCE_FP8=alternating` and a different free port and
container name. The 135-token example below then spans multiple physical
blocks and exercises both byte formats when the chosen blocks have different
parity.

To exercise the data-dependent policy on the model, use a fresh name and port
and enable transition logs. The low threshold is **only for testing**; with
the default threshold the small Qwen model may never need E5M2:

```bash
NPU_ID=<idle_npu_id> HOST_PORT=18083 \
  CONTAINER_NAME=pspkv-ascend-adaptive-test \
  PSPKV_ASCEND_REFERENCE_FP8=adaptive \
  PSPKV_ASCEND_E4M3_MAX_ABS=0.01 \
  PSPKV_ASCEND_LOG_PRECISION=1 \
  bash /root/zhang/vllm-pspkv/ascend/smoke_910b3.sh
```

After the API starts, send the short and long completion requests below, then
look for `PSP-KV adaptive FP8 block ... E4M3 -> E5M2` in
`docker logs pspkv-ascend-adaptive-test`. A successful HTTP response alone
does not prove that the policy selected either format. The CPU tests and NPU
checker at the end of this document also assert that the precision table and
cached bytes change together, that a mixed-format read is correct, and that
a recycled block resets to E4M3. Stop only this test container when finished.

To exercise live uniform FP4 allocation and attention, use a free port and
an idle NPU with a fresh container name:

```bash
NPU_ID=<idle_npu_id> HOST_PORT=18084 \
  CONTAINER_NAME=pspkv-ascend-fp4-test \
  PSPKV_ASCEND_UNIFORM_FP4=1 \
  bash /root/zhang/vllm-pspkv/ascend/smoke_910b3.sh
curl -sS --max-time 120 http://127.0.0.1:18084/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-0.6B","prompt":"北京是中国的","max_tokens":2,"temperature":0}'
```

Keep `--kv-cache-dtype fp8` in this bring-up script: that selects the byte
storage dtype, while the explicit environment flag changes page budgeting and
the codec to packed E2M1. The FP4 reference is limited to eager decoder
attention, one request at a time, no prefix caching or KV transfer. It makes
each KV page half the size of an FP8 page before the small scale-table cost;
under a fixed `--gpu-memory-utilization` budget, vLLM may use the saving to
allocate more pages rather than report lower total HBM use. Do not infer
throughput or model quality from this short correctness smoke test.

On this 910B3 server, Qwen3-0.6B served with uniform FP4 returned HTTP 200
for both a 16-token decode and a 144-token prompt spanning two pages, and
vLLM allocated about 754k KV token slots versus about 377k with FP8 at the
same memory budget. **The FP4 generated text was incoherent** on the 16-token
request. With the same model, NPU and prompt, PSP-KV FP8 generated coherent
Chinese. Choosing a smaller initial FP4 scale for small K/V values passed the
CPU and NPU reference checks but did not fix the model output. The failure
could reflect the uniform FP4 precision policy or another FP4-only error; the
cause has not been isolated. Treat this path as a storage and execution probe,
not a model-quality validation. The plugin's route log still says "FP8
reference attention backend" in this opt-in mode; the `PSP-KV uniform FP4
cache` log and doubled KV slot count confirm the actual packed page shape.

The script first checks whether that card is visible to an isolated Docker
container. A card can appear idle in host `npu-smi info` yet fail this check
with a driver `Resource_Busy` or `device is used` error. If that happens, the
script stops before starting the model; resolve the device issue or choose a
usable card. It mounts both the vLLM fork and the one-file plugin overlay
read-only, so the normal Ascend service stays intact. Replace the port if
18083 is in use.

Wait for `docker logs -f pspkv-ascend-ref-npu<idle_npu_id>` to report server
startup, then:

```bash
curl -sS --max-time 120 http://127.0.0.1:18083/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-0.6B","prompt":"北京是中国的","max_tokens":16,"temperature":0}'
```

Check the log for `Using the PSP-KV Ascend FP8 reference attention backend`
and `npu-smi info` for an inference process on the selected NPU. The first
launch may be slow. If it fails, keep the container and collect
`docker logs --tail 200 pspkv-ascend-ref-npu<idle_npu_id>`.

The short request checks initial prefill and several decode steps. To exercise
more than one 128-token KV block, make a longer prompt and inspect the returned
`usage.prompt_tokens`; keep it below the configured 256-token model limit:

```bash
PROMPT=$(printf '北京是中国的%.0s' {1..45})
curl -sS --max-time 120 http://127.0.0.1:18083/v1/completions \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"Qwen/Qwen3-0.6B\",\"prompt\":\"$PROMPT\",\"max_tokens\":16,\"temperature\":0}"
```

After testing, stop only the container created by this script, for example
`docker stop pspkv-ascend-ref-npu<idle_npu_id>`. A stopped container retains
its name; remove that stopped test container before reusing the same name, or
set `CONTAINER_NAME` for the next run.

To check E4M3/E5M2 byte operations, mixed physical blocks and adaptive
promotions on one visible
NPU, use the image's PyTorch installation through a disposable `uv` environment
inside the test container:

```bash
docker exec pspkv-ascend-ref-npu<idle_npu_id> \
  bash -lc 'curl -LsSf https://astral.sh/uv/install.sh | sh'
docker exec pspkv-ascend-ref-npu<idle_npu_id> \
  /root/.local/bin/uv venv --system-site-packages \
  --python /usr/local/python3.11.15/bin/python3 /tmp/pspkv-test/.venv
docker exec pspkv-ascend-ref-npu<idle_npu_id> \
  /tmp/pspkv-test/.venv/bin/python \
  /workspace/pspkv-src/ascend/check_fp8_reference_npu.py
docker exec pspkv-ascend-ref-npu<idle_npu_id> \
  /tmp/pspkv-test/.venv/bin/python \
  /workspace/pspkv-src/ascend/check_packed_reference_npu.py
docker exec pspkv-ascend-ref-npu<idle_npu_id> \
  /tmp/pspkv-test/.venv/bin/python -m pytest \
  --confcutdir=/workspace/pspkv-src/tests/v1/attention \
  -o cache_dir=/tmp/pspkv-test/.pytest_cache \
  /workspace/pspkv-src/tests/v1/attention/test_psp_kv_ascend_reference.py \
  /workspace/pspkv-src/tests/v1/attention/test_psp_kv_ascend_packed.py \
  /workspace/pspkv-src/tests/v1/attention/test_psp_kv_ascend_policy.py \
  /workspace/pspkv-src/tests/v1/attention/test_psp_kv_ascend_copy.py -q
```
