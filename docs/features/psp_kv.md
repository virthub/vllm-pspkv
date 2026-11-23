# PSP-KV (Adaptive-Precision KV Cache)

PSP-KV stores the KV cache as a flat byte buffer where each physical block
carries its own inner precision, so a deployment can keep 16-bit blocks where
they matter and 8-bit or 4-bit blocks everywhere else while still decoding with
tensor cores.

The cache is one flat byte buffer per layer. Per-block metadata — an inner
precision code, per-head scales, and a byte offset — is held in GPU tables that
the model runner refreshes every step, so the kernels never re-plan the layout.
The on-disk encoding is specified in
[PSP_KV_FORMAT_CONTRACT.md](../PSP_KV_FORMAT_CONTRACT.md).

## Usage

PSP-KV runs on Model Runner V2 (the only GPU runner since v0.34.0) and requires
its kernel block size to match the framework block size:

```bash
PSP_KV_BLOCK_SIZE=128 vllm serve Qwen/Qwen3-0.6B \
    --attention-backend PSP_KV \
    --block-size 128
```

The repository scripts wrap this flow: `./install.sh` provisions the environment
and builds the extension, and `./run.sh` is the single entry point for every
command — `--serve` sets the environment variables below before starting the
server, `--test-unit`/`--test-integration`/`--test-virthub` run the test suites,
and `--eval-quality`/`--eval-formats`/`--bench` reproduce the quality and
performance numbers quoted here. See the [README](../../README.md) for the
command-to-script map.

| requirement | value |
|---|---|
| Attention backend | `--attention-backend PSP_KV` |
| Block size | `--block-size 128`, matching the compile-time `PSP_KV_BLOCK_SIZE` |
| KV cache dtype | `auto` or `fp8` |
| Activation dtype | `float16` or `bfloat16` (prefill runs through FlashAttention) |
| Head size | one of 32, 64, 80, 96, 128, 160, 192, 256 |

`PSP_KV_BLOCK_SIZE` is a compile-time define in the PSP-KV extension, and the
Python side reads the same variable, so a mismatch between the build, the
runtime environment, and `--block-size` is rejected at backend selection rather
than silently producing a wrong layout.

## Cache budget

The page reserved per block is the average bytes-per-element of the configured
inner precision (`PSPKV_AVG_BYTES`, else derived from `PSPKV_PRECISION`):
16-bit for `lossless`, 1 for `e4m3`/`e5m2`/`int8`, 0.5 for `fp4`/`int4`. This
replaces the 16-bit worst case the default specs reserve, which is where the
memory saving comes from. The page is a fixed size per block: per-block
precision changes what is *stored* in a block, not the reservation.

## Virthub connector

Blocks can be handed to [Virthub](https://github.com/virthub/virthub)'s storage
layer through the `virthub` KV connector. The connector is registered lazily in
`KVConnectorFactory`, so vLLM imports without the package and only a config that
selects it fails, with an `ImportError` naming `virthub`. The fork's adapter
(`vllm/distributed/kv_transfer/kv_connector/v1/virthub_connector.py`) binds the
package to Model Runner V2 and adapts its callbacks to the vLLM 0.34 signatures
(`start_load_kv(forward_context, **kwargs)`, synchronous `wait_for_save`,
`save_kv_layer(layer_name, kv_cache, metadata)`), because the package targets an
older generation where blocks were passed explicitly and the save hook was
async.

```bash
./install.sh --virthub    # best-effort install of the optional package
vllm serve <model> --attention-backend PSP_KV \
    --kv-transfer-config '{"kv_connector": "virthub", "kv_role": "kv_both"}'
```

[`vllm/v1/attention/psp_kv_virthub.py`](../../vllm/v1/attention/psp_kv_virthub.py)
reconciles Virthub's `PrecisionLevel` policy taxonomy with the kernel
`precision_table` encoding, converts per-head scales to E8M0, and packs the
64-byte per-block sidecar the connector ships (`serialize_sidecar`).
`PSPKVState` registers a block-transfer-info provider with the connector so
variable-size blocks report their byte size and sidecar.

## Configuration

`PSPKV_PRECISION`, `PSPKV_FORMAT`, and `PSPKV_STORAGE` fall back to
[conf/pspkv.toml](../../conf/pspkv.toml) when unset; the environment always wins.
Point `PSPKV_CONFIG` at another TOML to change which file is read.

### Inner precision (`PSPKV_PRECISION`)

| value | code | bytes/elem | notes |
|---|---|---|---|
| `lossless` | 0 | 2 | rejected: stores full-precision KV and decodes 0.66–0.71× slower than FlashAttention |
| `e4m3` | 1 | 1 | 8-bit float, 4-bit exponent / 3-bit mantissa |
| `e5m2` | 2 | 1 | 8-bit float, 5-bit exponent / 2-bit mantissa |
| `fp4` / `int4` | 3 | 0.5 | 4-bit, decoded on the integer tensor-core path, or as the V operand of the per-K/V tensor-core path |
| `int8` (default) | 4 | 1 | block-scaled 8-bit integer, native int8 tensor cores |

### Adaptive per-block precision

`PSPKV_ADAPTIVE_PRECISION=1` derives each block's K and V formats from the
block's largest magnitude at prefill (`required_bits = log2(max_abs / (E * 240))`,
fp8 above one bit, fp4 below). The page geometry reserves the widest format the
derivation can emit for each operand and the derivation is clamped to it, so a
block can never be written wider than its reservation:

* `E < 0.5` reserves 1 byte/element for K and 0.5 for V; K stays fp8, V is fp4.
* `E >= 0.5` reserves 0.5 bytes/element for K as well, which raises the block
  capacity by 50% and keeps every K block at fp4.

Measured on Qwen3-4B with 64 generated tokens: adaptive reaches ppl 1.0198 at
`E = 0.01`, against 1.0265 for `int8` and 1.0270 for `int4`. The mixed
fp8-K/fp4-V table decodes on the per-K/V tensor-core kernel at ~1540 tok/s
(ctx 2048, batch 32) versus ~2500 for uniform `int8`, so adaptive trades decode
speed for a smaller cache.

### Format generation (`PSPKV_FORMAT`)

| value | format | decode path |
|---|---|---|
| 0 | Basic | not supported by this port (head-major page assumption) |
| 1 (default) | Enhanced | split-K fp8/int8 tensor-core decode, per-block mixed precision |
| 2 | BT-KV | not supported by this port (head-major page assumption) |

Generations 0 and 2 are rejected at startup: their kernels assume head-major
pages, while the byte-buffer cache is allocated token-major, so they would
mis-decode any model with more than one KV head.

### Environment variables

| variable | type | default | description |
|---|---|---|---|
| `PSPKV_PRECISION` | str | from TOML (`int8`) | Inner precision, see the table above |
| `PSPKV_FORMAT` | int | from TOML (`1`) | Format generation |
| `PSPKV_STORAGE` | str | from TOML (`fixed`) | `fixed` slot width or `variable` per-block stride |
| `PSPKV_CONFIG` | str | `./conf/pspkv.toml` | Path of the TOML holding the sections above |
| `PSPKV_AVG_BYTES` | float | unset | Override the bytes-per-element reserved per block |
| `PSPKV_ADAPTIVE_PRECISION` | bool | `0` | Derive each block's K/V precision from its magnitude, see below |
| `PSPKV_ADAPTIVE_E` | float | `1e-3` | Target relative error for the derivation above; `>= 0.5` keeps every K block at fp4 |
| `PSP_KV_BLOCK_SIZE` | int | `128` | Tokens per kernel block; must match the build and `--block-size` |
| `PSP_KV_TOKEN_MAJOR` | int | `1` | Kernel block layout: 1 token-major, 0 head-major (compile-time) |
| `PSPKV_TC_NUM_SPLITS` | int | `0` | Split-K count for the tensor-core decode kernel; 0 = auto |
| `PSPKV_DEBUG` | int | `0` | Emit kernel-level debug traces |
| `PSPKV_TIME` | bool | `0` | Accumulate per-step quantize/attention GPU timing |
| `PSPKV_PROFILE` | bool | `0` | Emit per-step prefill timing |
| `PSPKV_VERIFY` | bool | `0` | Cross-check the fused path against the reference kernel |
| `PSPKV_FA3` | bool | `0` | Route Enhanced-format decode through the FlashAttention-3 kernel |

Boolean variables accept `1`, `true`, or `on`.

## Limitations

- Speculative decoding is not supported and is rejected at startup: the kernels
  decode one query token per request, so multi-token verify rows have no path.
- Adaptive precision (`PSPKV_ADAPTIVE_PRECISION`) does not shrink the cache:
  blocks are fixed-size pages, so a 4-bit block occupies the same reservation
  as an 8-bit one. Set `PSPKV_AVG_BYTES` to budget for the mix you actually
  store.
- The PSP-KV extension is built outside `_C_stable_libtorch`, because the
  wrapper uses the regular libtorch API rather than the stable ABI that vLLM
  0.34 moved its CUDA ops to. This is the one place PSP-KV diverges from the
  upstream build layout.
