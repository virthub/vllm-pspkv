# vLLM with PSP-KV

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![CUDA 13.0](https://img.shields.io/badge/CUDA-13.0-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![GPU](https://img.shields.io/badge/tested-RTX%205090%20(SM120)-76b900.svg)](#status-and-limitations)

A fork of **vLLM 0.34.0** that integrates the **PSP-KV (Precision-Scalable Paged
KV-Cache)** backend from [Virthub](https://github.com/virthub/virthub). Each KV
cache block carries its own inner precision (FP8, FP4, block-scaled INT8 or
lossless), decided per block at write time, so a single cache can mix
precisions and the decode kernels read only the bytes a block actually needs.

---

## Key features

- **Per-block inner precision** — `int8` (default), `e4m3`, `e5m2`, `fp4`/`int4`,
  with a per-block `precision_table` and per-scale-group float scales.
- **Adaptive precision** (`PSPKV_ADAPTIVE_PRECISION=1`) — derives each block's
  K/V format from that block's magnitude at prefill, instead of a single
  process-wide choice.
- **Enhanced format** — out-of-band sidecar metadata, token-major pages, split-K
  tensor-core decode (fp8, int8 and per-K/V mixed fp8+fp4 paths).
- **Model Runner V2** — the fork tracks vLLM 0.34.0, whose only GPU runner is
  MRV2; the PSP-KV state is created inside `GPUModelRunner` and the KV-cache
  spec is routed through `customize_attention_spec()`.
- **Virthub KV connector** (optional) — hands blocks to Virthub's storage layer
  with a packed per-block sidecar (precision, scale groups, page pointer).
- **CUDA graphs** (full decode) and **zero-copy metadata** via pinned host
  buffers with dirty-block batching.

> **Formats:** only the **Enhanced** generation (`PSPKV_FORMAT=1`) is supported
> by this port. The Basic (0) and BT-KV (2) generations are rejected at startup
> because their kernels assume head-major pages while the byte-buffer cache is
> allocated token-major.

---

## Installation

```bash
./install.sh
```

The script creates the virtual environment (`.venv34` by default), installs the
matching upstream `vllm==0.34.0` wheel to supply vLLM's own compiled extensions,
then builds the PSP-KV extension (`vllm/_C.abi3.so`) from `csrc/` with the small
standalone CMake project in `cmake/psp_kv`.

Prerequisites: **Python 3.10–3.13** (3.12 tested), a **CUDA toolkit** with
`nvcc` for your GPU (`CUDA_HOME`, default `120a`), `cmake` and (recommended)
`ninja`.

```bash
./install.sh --virthub            # also provision the optional Virthub package
./install.sh --cuda-arch 90a      # build the kernels for a different SM
./install.sh --no-build           # environment only
./install.sh --clean              # remove the venv and all build artifacts
```

The PSP-KV extension can be rebuilt at any time without touching the rest of
vLLM:

```bash
./run.sh --build
```

`install.sh` is the only setup entry point and never needs to be followed by a
second step; every build artifact it produces (the CMake tree, the wheel cache
and the compiled extension) lives in `build/` or `vllm/_C.abi3.so`, both ignored
by git.

### Repository layout

| path | contents |
|---|---|
| `vllm/`, `csrc/`, `tests/`, `cmake/` | this fork's source (the PSP-KV package, kernels, tests and the standalone extension build) |
| `conf/`, `docs/features/psp_kv.md`, `install.sh`, `run.sh` | fork configuration, documentation and the two entry points |
| `scripts/`, `benchmarks/psp_kv/` | tracked tooling, each reachable through `./run.sh` (see below) |
| `recipes/`, `rust/`, `tools/`, `docker/`, `examples/` | upstream vLLM content, kept unchanged for parity |
| `build/`, `tools/virthub/` | generated: the extension build tree and the optional fetched Virthub sources |

---

## Quick start

```bash
# Serve with the default settings (int8 inner precision, Enhanced format)
./run.sh --serve --model Qwen/Qwen3-0.6B

# Adaptive per-block precision
./run.sh --serve --model Qwen/Qwen3-0.6B --precision adaptive --enforce-eager
```

The script sets the required environment (`VLLM_ATTENTION_BACKEND=PSP_KV`,
`PSPKV_CONFIG`, `PSPKV_FORMAT`, `PSPKV_PRECISION`) and starts
`vllm serve` with `--attention-backend PSP_KV --block-size 128`.

Manual launch:

```bash
export VLLM_ATTENTION_BACKEND=PSP_KV
export PSPKV_CONFIG=./conf/pspkv.toml
export PSPKV_FORMAT=1

vllm serve Qwen/Qwen3-0.6B \
    --attention-backend PSP_KV \
    --block-size 128 \
    --kv-transfer-config '{"kv_connector": "virthub", "kv_role": "kv_both"}'
```

`--block-size` is fixed at 128 for this build: the kernels are compiled with
`PSP_KV_BLOCK_SIZE=128`, and the engine refuses a mismatch.

---

## Testing

```bash
./run.sh --test-unit          # kernel correctness + Virthub adapter (CPU, fast)
./run.sh --test-integration   # decode output vs FlashAttention (GPU)
./run.sh --test-virthub       # Virthub connector registration/adapter tests
./run.sh --test              # unit + Virthub + integration
./run.sh --eval-quality       # per-precision ppl, agreement, throughput, KV footprint
./run.sh --eval-formats       # offline scale-granularity and format-pair attribution
./run.sh --bench              # decode-throughput benchmark (GPU)
```

`run.sh` is the single entry point: every tracked script is invoked through it,
and no script under `scripts/` or `benchmarks/` is meant to be run by hand.

| command | script |
|---|---|
| `--build` | `cmake/psp_kv` (CMake project) |
| `--test-unit` | `tests/kernels/attention/test_psp_kv_kernels.py`, `tests/v1/attention/test_psp_kv_virthub_reconcile.py` |
| `--test-integration` | `scripts/compare_backends.py` |
| `--test-virthub` | `tests/v1/kv_connector/test_virthub_connector_registration.py` |
| `--eval-quality` | `scripts/eval_quality.py` |
| `--eval-formats` | `scripts/eval_kv_formats.py` |
| `--bench` | `benchmarks/psp_kv/decode_throughput.py` |

The kernel tests drive invariants whose correct result is arithmetic rather than
a reference implementation — softmax row normalisation, V element order and GQA
head mapping — so a failure names the broken property (a transposed page layout,
a mis-scaled group) instead of an opaque diff.

---

## Configuration

`conf/pspkv.toml` selects the format generation and the default inner precision;
`conf/virthub.toml` is still read as a backward-compatible fallback
(`VIRTHUB_CONFIG`) for the legacy launcher location.

```toml
[format]
generation = 1            # 0 = Basic (unsupported), 1 = Enhanced, 2 = BT-KV (unsupported)

[precision]
inner_precision = "int8"  # lossless | e4m3 | e5m2 | fp4 | int8

[storage]
mode = "fixed"            # fixed-width pages; "variable" with Virthub
```

### Inner precisions

| value | bytes/elem | notes |
|---|---|---|
| `int8` (default) | 1 | block-scaled 8-bit integer, native int8 tensor cores |
| `e4m3` | 1 | FP8, 4-bit exponent / 3-bit mantissa |
| `e5m2` | 1 | FP8, 5-bit exponent / 2-bit mantissa |
| `fp4` / `int4` | 0.5 | 4-bit, integer tensor-core path |
| `lossless` | 2 | rejected: it stores the full-precision values and decodes slower than FlashAttention |

### Environment variables

| variable | default | description |
|---|---|---|
| `PSPKV_PRECISION` | from TOML (`int8`) | inner precision |
| `PSPKV_FORMAT` | from TOML (`1`) | format generation; only 1 is supported |
| `PSPKV_ADAPTIVE_PRECISION` | `0` | per-block precision derived from block magnitude |
| `PSPKV_ADAPTIVE_E` | `1e-3` | target relative error for the derivation (`>= 0.5` keeps K at fp4) |
| `PSPKV_CONFIG` | `./conf/pspkv.toml` | launcher configuration file |
| `VIRTHUB_CONFIG` | `./conf/virthub.toml` | legacy launcher configuration fallback |
| `PSPKV_TC_NUM_SPLITS` | `0` (auto) | split-K count for the tensor-core decode kernel |
| `PSPKV_DEBUG`, `PSPKV_TIME`, `PSPKV_PROFILE` | `0` | kernel/step diagnostics |

The full reference — including the adaptive-precision geometry rules, the decode
paths and the measured costs — lives in
[docs/features/psp_kv.md](docs/features/psp_kv.md).

---

## Virthub KV connector

PSP-KV can hand KV blocks to [Virthub](https://github.com/virthub/virthub)'s
storage layer. The connector name is registered lazily, so vLLM imports and runs
without the package; only a config that selects it needs the dependency:

```bash
./install.sh --virthub        # best effort; skipped if the package cannot be built
vllm serve <model> --attention-backend PSP_KV \
    --kv-transfer-config '{"kv_connector": "virthub", "kv_role": "kv_both"}'
```

The connector is adapted to vLLM 0.34's Model Runner V2 contract by
`vllm/distributed/kv_transfer/kv_connector/v1/virthub_connector.py` (the package
targets an older generation: explicit block arguments and an async save hook),
and the server has been exercised end to end with
`--kv-transfer-config '{"kv_connector": "virthub"}'`.

The per-block sidecar (base precision, E8M0 per-head scales, page pointer) is
produced by `vllm/v1/attention/psp_kv_virthub.py`, which also reconciles
Virthub's `PrecisionLevel` policy taxonomy with the kernel `precision_table`
encoding. `./run.sh --serve --kv-connector none` serves without a connector.

---

## Status and limitations

Measured on an RTX 5090 D with Qwen3-4B (decode-only, 2048-token context, batch
32), relative to the FlashAttention backend:

| configuration | throughput | quality |
|---|---|---|
| `int8` (default) | ~1.1× FlashAttention | ppl 1.0265 |
| `e4m3` | ~0.93× | ppl 1.0273 |
| adaptive (fp8 K + fp4 V) | ~0.8× | **ppl 1.0198** |

These are single-model, single-configuration numbers; treat them as indicative.

- Speculative decoding is rejected at startup: the decode kernels take one query
  token per request, so multi-token verify rows have no path.
- Adaptive precision does not shrink the cache with `mode = "fixed"`: blocks are
  fixed-size pages, so a 4-bit block reserves the same space as an 8-bit one.
- The PSP-KV extension is built outside vLLM's `_C_stable_libtorch` because it
  uses the regular libtorch API rather than the stable ABI.
- The fork has been exercised on a single GPU (SM120); other architectures build
  with `./install.sh --cuda-arch`.

---

## Documentation and license

- PSP-KV internals: [docs/features/psp_kv.md](docs/features/psp_kv.md)
- Upstream vLLM: [docs.vllm.ai](https://docs.vllm.ai)
- Virthub: [github.com/virthub/virthub](https://github.com/virthub/virthub)

This project inherits vLLM's [Apache-2.0 license](LICENSE); the PSP-KV
integration and scripts in this fork are released under the same terms.
