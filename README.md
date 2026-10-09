# vLLM with PSP‑KV

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![CUDA 12.4+](https://img.shields.io/badge/CUDA-12.4%2B-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![GCC 12](https://img.shields.io/badge/GCC-12-blue.svg)](https://gcc.gnu.org/)

This is a **customised fork of vLLM 0.20.2** that integrates the **PSP‑KV (Precision‑Scalable Paged KV‑Cache)** backend from [Virthub](https://github.com/virthub/virthub).  
PSP‑KV reduces GPU memory consumption for KV caches by up to **50%** through per‑block precision scaling (FP16, FP8, FP8+residual, and pruned), while maintaining model quality via a deterministic, allocation‑time precision predictor.

---

## ✨ Key Features

- **Precision‑Scalable KV Cache** – each block can store FP16, FP8, FP8+residual, or pruned values based on runtime memory pressure.
- **Three Physical Formats**:
  - **Basic** – in‑band metadata, symmetric 1D layout.
  - **Enhanced** – out‑of‑band sidecar, asymmetric 1D layout, fused dequantisation.
  - **BT‑KV** – hardware‑native 2D block‑tiled layout for Hopper (SM90+) and Blackwell (SM100+) GPUs (requires CUDA 12.3+).
- **Direct integration** with vLLM’s **V1 engine** via a custom `AttentionBackend`.
- **Zero‑copy metadata updates** using pinned host buffers and chunked batch synchronisation.
- **CUDA Graph support** (full decode only) for low‑overhead inference.
- **Multi‑tier store** (VRAM → DRAM → NVMe) and distributed memory mesh (via Virthub’s KLNK engine) – currently in software fallback mode (RDMA pending).

> **Note:** BT‑KV format requires **CUDA 12.3+** and **Hopper (SM90+)** or **Blackwell (SM100+)**. Basic and Enhanced formats work on **Ampere (SM80+) and newer**.

---

## 📦 Installation

```bash
./install.sh
```

Run the script to automatically set up the environment, install dependencies, and build the vLLM extension.  
Prerequisites: **Python 3.10**, **CUDA 12.4+**, and **GCC 12**.

---

## 🧪 Testing

We provide a `run.sh` script to run various test suites:

```bash
# Run all tests (unit + integration)
./run.sh --test

# Run only unit tests (fast)
./run.sh --test-unit

# Run end‑to‑end integration tests (requires GPU)
./run.sh --test-integration
```

These tests validate the PSP‑KV backend, kernel correctness, and end‑to‑end inference with a small model (e.g., `facebook/opt-125m`).

---

## 🚀 Usage

The `run.sh` script automatically sets the required environment variables (`VLLM_ATTENTION_BACKEND=PSP_KV` and `VIRTHUB_CONFIG=./conf/virthub.toml`) for you. You can also override them manually if needed.

### Quick Start with `run.sh`

```bash
# Start the vLLM server with default settings (Enhanced format)
./run.sh --serve --model facebook/opt-125m

# Use a different model and override the format to BT‑KV (if on Hopper+)
./run.sh --serve --model meta-llama/Llama-2-7b-hf --format 2

# Disable CUDA graphs for debugging
./run.sh --serve --model facebook/opt-125m --enforce-eager
```

### Manual Server Launch

If you prefer to run `vllm serve` directly, set the environment variables explicitly:

```bash
export VLLM_ATTENTION_BACKEND=PSP_KV
export VIRTHUB_CONFIG=./conf/virthub.toml

vllm serve facebook/opt-125m \
    --kv-connector virthub \
    --kv-role kv_both
```

### Configuration File

The precision predictor is configured via `conf/virthub.toml`. A default file is provided in the repository. You can adjust the thresholds and format generation to suit your hardware and workload.

```toml
[general]
log_level = "info"

[precision]
format_generation = 1          # 0=Basic, 1=Enhanced, 2=BT‑KV
sink_window = 16
local_window = 64
critical_layer_count = 2
elevated_pressure_threshold = 0.78
nominal_pressure_threshold = 0.70
critical_pressure_threshold = 0.88
critical_relax_threshold = 0.82

[vllm]
connector_version = "v2"
```

To use a custom config, set the `VIRTHUB_CONFIG` environment variable:

```bash
export VIRTHUB_CONFIG=/path/to/your/config.toml
```

You can also temporarily override the format generation without editing the file by setting `VIRTHUB_FORMAT`:

```bash
export VIRTHUB_FORMAT=2   # Override to BT‑KV
```

Or use the `--format` option in `run.sh`:

```bash
./run.sh --serve --format 2
```

---

## 📖 Documentation

- For general vLLM usage, see the [official vLLM documentation](https://docs.vllm.ai).
- For PSP‑KV internals and the precision predictor, refer to the [Virthub repository](https://github.com/virthub/virthub).

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).  
The original vLLM code is licensed under the Apache‑2.0 license.
