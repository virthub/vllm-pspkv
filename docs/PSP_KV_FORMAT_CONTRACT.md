# PSP-KV format contract

Single source of truth for the three PSP-KV physical storage formats and how the
vLLM wrapper (`csrc/attention/psp_kv_wrapper.cu`) maps onto the CUDA kernels
(`csrc/attention/psp_kv/{basic,enhanced,btkv}/*.cu`).

## Formats

| | Basic (`format=0`) | Enhanced (`format=1`) | BT-KV (`format=2`) |
|---|---|---|---|
| KV container | `uint8` (FP8 bits) | `uint8` (FP8 bits) | `uint8` (FP8 bits) |
| K layout | token-major `[B_t, D]` | token-major `[B_t, D]` | 2D tile `[16,64]` |
| V layout | token-major `[B_t, D]` | **transposed** `[D, B_t]` | **transposed** `[D, B_t]` |
| Dequant | `dequantize_fp8(q, fmt) · scale` | `dequantize_fp8(q, fmt) · scale` | `dequantize_fp8(q, fmt) · scale` |
| FP8 format | `precision_table[block]` (1=e4m3, 2=e5m2) | same | same |
| Scale metadata | unit scale (no `scale_table`) | `scale_table` fp32 `[B, H]` | `scale_table` fp32 `[B, H]` |
| Hardware gate | any CUDA | any CUDA | SM90+ (TMA descriptors) |

Compile-time constants shared by the reference kernels: `BLOCK_SIZE=16`,
`HEAD_DIM=128`, `MAX_KV_HEADS=8`.

## Precision model (two axes)

**Target precision** — the query/output dtype. The kernels are templated on it
and support **fp16, bf16, and fp8 e4m3/e5m2** (`half` / `__nv_bfloat16` /
`__nv_fp8_e4m3` / `__nv_fp8_e5m2`).

**Inner precision** — the per-block KV storage precision, selected by the
`precision_table` (uint8 `[num_blocks]`) and chosen on-demand by the PSP-KV
precision predictor. Each target precision has **three** inner precision levels:

| `precision_table` value | Inner precision | Bits/elem | Status |
|---|---|---|---|
| `0` | **lossless** (fp16 for fp16 target, bf16 for bf16 target) | 16 | implemented |
| `1` | **fp8** e4m3 | 8 | implemented |
| `2` | **fp8** e5m2 | 8 | implemented |
| `3` | **fp4** e2m1 | 4 | implemented |

## Wrapper → kernel bridge

- `precision_table` (uint8 `[num_blocks]`) selects the per-block inner precision:
  `0` = lossless (reserved), `1` = fp8 e4m3 (default), `2` = fp8 e5m2,
  `3` = fp4 (reserved).
- `scale_table` (fp32 `[num_blocks, num_kv_heads]`):
  - **basic**: no out-of-band scale → unit scale (1.0).
  - **enhanced / btkv**: passed through directly as fp32 (no E8M0 conversion).

The KV cache is a **16-bit container** (fp16/bf16, matching the target
precision). Dequant per element is driven by `precision_table[block]`:

- lossless (`0`): read the 16-bit value directly (no scale).
- fp8 (`1`/`2`): `dequantize_fp8(low_byte, fmt) · scale` (the fp8 byte sits in the
  low byte of the 16-bit container). `dequantize_fp8` is `quantization_utils.cuh`'s
  bit-exact e4m3/e5m2 decoder.
- fp4 (`3`): `dequantize_fp4_e2m1(low_nibble) · scale` (the fp4 nibble sits in the
  low nibble of the 16-bit container).

Quantize (prefill write path) is the inverse: lossless stores the 16-bit value;
fp8 stores `quantize_fp8(x / scale, fmt)` in the low byte; fp4 stores
`quantize_fp4_e2m1(x / scale)` in the low nibble.

## Known gaps (reference vs. intended spec)

1. **Unit-scale prefill.** `quantize_and_scatter_kv` quantizes with a unit scale;
   per-block/head scale estimation (max-abs) is a follow-up.
2. **Fixed 16-bit container (no memory savings).** The cache is a fixed 16-bit
   container, so fp8 blocks waste the unused high byte and fp4 blocks waste the
   unused high 12 bits. Variable-stride/pooled storage (the compression benefit)
   is Virthub's storage-management role.
3. **Extension placement.** The kernels were compiled into `_C_stable_libtorch`
   while the wrapper lives in `_C`; the kernels are now compiled into `_C`
   (they are torch-agnostic raw CUDA) so the wrapper can call `launch_*`
   directly.
