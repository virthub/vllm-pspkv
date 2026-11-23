#!/bin/bash
# run.sh – command runner for vLLM with PSP-KV.
#
# Every Python invocation goes through the project virtual environment
# ($VENV_DIR, default .venv34) created by ./install.sh; no `uv` is required.
#
#   ./run.sh --test-unit          CPU unit tests (kernels + Virthub adapter)
#   ./run.sh --test-integration   GPU end-to-end check against FlashAttention
#   ./run.sh --serve --model ...  start an OpenAI-compatible server
#   ./run.sh --build              (re)build the PSP-KV extension
#   ./run.sh --clean              remove build artifacts
#
# `--help` and `--clean*` work even when the environment is not set up yet.

set -euo pipefail

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
VENV_DIR="${VENV_DIR:-.venv34}"
BUILD_DIR="${BUILD_DIR:-build/psp}"
PSPKV_CUDA_ARCH="${PSPKV_CUDA_ARCH:-120a}"
PSPKV_SCALE_GROUP="${PSPKV_SCALE_GROUP:-16}"
DEFAULT_MODEL="${DEFAULT_MODEL:-Qwen/Qwen3-0.6B}"
PSPKV_TEST_MODEL="${PSPKV_TEST_MODEL:-Qwen/Qwen3-0.6B}"

# ------------------------------------------------------------------
# Logging (colour only on a TTY)
# ------------------------------------------------------------------
if [[ -t 1 ]]; then
    BOLD=$(tput bold 2>/dev/null || echo "")
    RED=$(tput setaf 1 2>/dev/null || echo "")
    GREEN=$(tput setaf 2 2>/dev/null || echo "")
    YELLOW=$(tput setaf 3 2>/dev/null || echo "")
    BLUE=$(tput setaf 4 2>/dev/null || echo "")
    RESET=$(tput sgr0 2>/dev/null || echo "")
else
    BOLD="" RED="" GREEN="" YELLOW="" BLUE="" RESET=""
fi

log_info()    { echo "${BLUE}${BOLD}[INFO]${RESET} $*"; }
log_success() { echo "${GREEN}${BOLD}[SUCCESS]${RESET} $*"; }
log_warning() { echo "${YELLOW}${BOLD}[WARNING]${RESET} $*" >&2; }
log_error()   { echo "${RED}${BOLD}[ERROR]${RESET} $*" >&2; exit 1; }

# ------------------------------------------------------------------
# Project root
# ------------------------------------------------------------------
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"
[[ -d vllm ]] || log_error "Please run this script from the vLLM project root directory."

show_help() {
    cat <<EOF
${BOLD}Usage:${RESET} $0 [COMMAND] [OPTIONS]

${BOLD}Commands:${RESET}
  ${BOLD}--test, -t${RESET}              Run the CPU unit tests and the GPU integration check.
  ${BOLD}--test-unit${RESET}             Run only the CPU unit tests (fast).
  ${BOLD}--test-integration${RESET}      Compare PSP-KV decode output against FlashAttention (GPU).
  ${BOLD}--test-virthub${RESET}          Run the Virthub adapter/registration tests.
  ${BOLD}--eval-quality${RESET}          Per-precision perplexity, agreement, throughput and KV footprint.
  ${BOLD}--eval-formats${RESET}          Offline scale-granularity and format-pair accuracy attribution.
  ${BOLD}--serve, -s${RESET}             Start the vLLM server with PSP-KV.
  ${BOLD}--bench, -b${RESET}             Decode-throughput benchmark (GPU).
  ${BOLD}--build${RESET}                 (Re)build the PSP-KV extension.
  ${BOLD}--clean${RESET}                 Remove generated artifacts (build trees, caches, extension).
  ${BOLD}--clean-all${RESET}             Also remove the virtual environment, fetched sources and
                          every compiled extension (re-run ./install.sh afterwards).
  ${BOLD}--help, -h${RESET}              Show this help message.

${BOLD}Options for --serve:${RESET}
  --model MODEL            Model to serve (default: $DEFAULT_MODEL).
  --port PORT              Port to listen on (default: 8000).
  --precision PREC         PSP-KV inner precision: int8 (default), e4m3, e5m2,
                           int4/fp4. Set "adaptive" to let each block be derived
                           from its magnitude (PSPKV_ADAPTIVE_PRECISION=1).
  --format 0|1|2           PSP-KV format generation; only 1 (Enhanced) is
                           supported by this port.
  --block-size N           KV cache block size (default: 128; must match the
                           kernel build).
  --kv-connector NAME      KV transfer connector (default: virthub). Pass "none"
                           to serve without a KV connector.
  --kv-role ROLE           KV connector role (default: kv_both).
  --enforce-eager          Disable CUDA graphs.
  --dry-run                Print the server command instead of starting it.
  --extra "ARGS"           Extra arguments appended to the vllm serve command.

${BOLD}Environment:${RESET}
  VENV_DIR                 Virtual environment (default: $VENV_DIR).
  PSPKV_CONFIG             Launcher config (default: ./conf/pspkv.toml).
  VIRTHUB_CONFIG           Legacy launcher config (default: ./conf/virthub.toml).
  PSPKV_PRECISION          Inner precision (also settable with --precision).
  PSPKV_FORMAT             Format generation (also settable with --format).
  PSPKV_CUDA_ARCH          CUDA arch used by --build (default: $PSPKV_CUDA_ARCH).

${BOLD}Examples:${RESET}
  $0 --test-unit
  $0 --serve --model Qwen/Qwen3-0.6B --precision int8
  $0 --serve --model meta-llama/Llama-3.1-8B-Instruct --precision adaptive
  $0 --build --clean
EOF
}

# ------------------------------------------------------------------
# Environment
# ------------------------------------------------------------------
setup_env() {
    [[ -x "$VENV_DIR/bin/python" ]] || \
        log_error "Virtual environment not found at $VENV_DIR. Run ./install.sh first (or set VENV_DIR)."
    PY="$VENV_DIR/bin/python"

    if [[ -z "${CUDA_HOME:-}" ]]; then
        for candidate in /usr/local/cuda /usr/local/cuda-13.0 /usr/local/cuda-12.8; do
            if [[ -x "$candidate/bin/nvcc" ]]; then CUDA_HOME="$candidate"; break; fi
        done
    fi
    if [[ -n "${CUDA_HOME:-}" ]]; then
        export CUDA_HOME
        export PATH="$CUDA_HOME/bin:$PATH"
        export CUDACXX="${CUDACXX:-$CUDA_HOME/bin/nvcc}"
    fi

    # PSP-KV launcher configuration. The backend name is the registered
    # attention backend; the TOML paths are fallbacks for the legacy names.
    export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-PSP_KV}"
    export PSPKV_CONFIG="${PSPKV_CONFIG:-./conf/pspkv.toml}"
    export VIRTHUB_CONFIG="${VIRTHUB_CONFIG:-./conf/virthub.toml}"
    export PSPKV_FORMAT="${PSPKV_FORMAT:-1}"
    export PSPKV_PRECISION="${PSPKV_PRECISION:-int8}"
    # Keep the kernels' block size and the engine's in sync by default.
    export PSPKV_BLOCK_SIZE="${PSPKV_BLOCK_SIZE:-128}"
    export PSP_KV_BLOCK_SIZE="${PSP_KV_BLOCK_SIZE:-$PSPKV_BLOCK_SIZE}"
    export PSP_KV_TOKEN_MAJOR="${PSP_KV_TOKEN_MAJOR:-1}"
}

# ------------------------------------------------------------------
# Build
# ------------------------------------------------------------------
run_build() {
    setup_env
    command -v cmake >/dev/null 2>&1 || log_error "cmake is required to build the PSP-KV extension."
    [[ -x "${CUDA_HOME:-}/bin/nvcc" ]] || log_error "nvcc not found; set CUDA_HOME."
    TORCH_CMAKE="$("$PY" -c 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__), "share", "cmake"))')"
    GENERATOR=()
    command -v ninja >/dev/null 2>&1 && GENERATOR=(-G Ninja)
    # A build tree generated from a different source directory cannot be reused.
    if [[ -f "$BUILD_DIR/CMakeCache.txt" ]] && \
       ! grep -q "CMAKE_HOME_DIRECTORY:INTERNAL=$PWD/cmake/psp_kv" "$BUILD_DIR/CMakeCache.txt"; then
        log_warning "Discarding stale build cache in $BUILD_DIR (different source directory)."
        rm -rf "$BUILD_DIR"
    fi

    log_info "Configuring the PSP-KV extension (arch=$PSPKV_CUDA_ARCH, scale_group=$PSPKV_SCALE_GROUP) ..."
    cmake -S cmake/psp_kv -B "$BUILD_DIR" "${GENERATOR[@]}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DPython_EXECUTABLE="$PROJECT_ROOT/$VENV_DIR/bin/python" \
        -DCMAKE_PREFIX_PATH="$TORCH_CMAKE" \
        -DPSPKV_CUDA_ARCH="$PSPKV_CUDA_ARCH" \
        -DPSPKV_SCALE_GROUP_SIZE="$PSPKV_SCALE_GROUP" >/dev/null
    log_info "Building vllm/_C.abi3.so ..."
    cmake --build "$BUILD_DIR" --target _C
    [[ -f vllm/_C.abi3.so ]] || log_error "Build finished but vllm/_C.abi3.so is missing."
    log_success "PSP-KV extension built."
}

# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------
run_unit_tests() {
    setup_env
    if ! "$PY" -c "import pytest" >/dev/null 2>&1; then
        log_warning "pytest is not installed in $VENV_DIR; installing ..."
        "$PY" -m pip install --quiet pytest || log_error "Could not install pytest."
    fi
    # --noconftest: these suites are self-contained and tests/conftest.py pulls
    # in heavy optional dependencies (transformers, tblib, PIL, ...).
    log_info "Running PSP-KV unit tests ..."
    "$PY" -m pytest \
        tests/kernels/attention/test_psp_kv_kernels.py \
        tests/v1/attention/test_psp_kv_virthub_reconcile.py \
        -q --noconftest || log_error "Unit tests failed."
    log_success "Unit tests passed."
}

run_virthub_tests() {
    setup_env
    log_info "Running Virthub adapter tests ..."
    "$PY" -m pytest \
        tests/v1/kv_connector/test_virthub_connector_registration.py \
        tests/v1/attention/test_psp_kv_virthub_reconcile.py \
        -q --noconftest || log_error "Virthub tests failed."
    if ! "$PY" -c "import virthub" >/dev/null 2>&1; then
        log_warning "The virthub package is not installed: the connector name is"
        log_warning "registered but resolving it will raise ImportError."
        log_warning "Run ./install.sh --virthub to install it."
    fi
    log_success "Virthub tests passed."
}

run_integration_tests() {
    setup_env
    log_info "GPUs and model: $PSPKV_TEST_MODEL"
    if ! "$PY" -c "import torch; assert torch.cuda.is_available()" >/dev/null 2>&1; then
        log_warning "No CUDA device available; skipping the integration check."
        return 0
    fi
    [[ -f scripts/compare_backends.py ]] || \
        log_error "scripts/compare_backends.py is missing; cannot run the integration check."
    log_info "Comparing PSP-KV decode output against FlashAttention ..."
    PSP_KV_BLOCK_SIZE="$PSPKV_BLOCK_SIZE" "$PY" scripts/compare_backends.py
    log_success "Integration check passed (tokens match FlashAttention)."
}

run_all_tests() {
    run_unit_tests
    run_virthub_tests
    run_integration_tests
    log_success "All tests passed."
}

# ------------------------------------------------------------------
# Serve
# ------------------------------------------------------------------
serve_model() {
    local MODEL="$DEFAULT_MODEL"
    local PORT="8000"
    local PRECISION=""
    local FORMAT=""
    local BLOCK_SIZE="128"
    local ENFORCE_EAGER=""
    local KV_CONNECTOR="virthub"
    local KV_ROLE="kv_both"
    local EXTRA=""
    local DRY_RUN=0

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --model)         MODEL="$2"; shift 2 ;;
            --dry-run)       DRY_RUN=1; shift ;;
            --port)          PORT="$2"; shift 2 ;;
            --precision)     PRECISION="$2"; shift 2 ;;
            --format)        FORMAT="$2"; shift 2 ;;
            --block-size)    BLOCK_SIZE="$2"; shift 2 ;;
            --kv-connector)  KV_CONNECTOR="$2"; shift 2 ;;
            --kv-role)       KV_ROLE="$2"; shift 2 ;;
            --enforce-eager) ENFORCE_EAGER="--enforce-eager"; shift ;;
            --extra)         EXTRA="$2"; shift 2 ;;
            *)               log_error "Unknown option for --serve: $1" ;;
        esac
    done

    setup_env
    [[ -n "$PRECISION" ]] && export PSPKV_PRECISION="$PRECISION"
    [[ -n "$FORMAT" ]] && export PSPKV_FORMAT="$FORMAT"
    if [[ "$PSPKV_PRECISION" == "adaptive" ]]; then
        export PSPKV_ADAPTIVE_PRECISION=1
    fi
    if [[ "$PSPKV_FORMAT" != "1" ]]; then
        log_warning "PSPKV_FORMAT=$PSPKV_FORMAT is not supported by this port (only 1 = Enhanced)."
        log_warning "PSP-KV will refuse to start; use --format 1."
    fi
    if [[ "$BLOCK_SIZE" != "$PSPKV_BLOCK_SIZE" ]]; then
        log_error "block size $BLOCK_SIZE does not match the kernel build (PSPKV_BLOCK_SIZE=$PSPKV_BLOCK_SIZE)."
    fi

    local TRANSFER_ARGS=()
    if [[ "$KV_CONNECTOR" != "none" ]]; then
        TRANSFER_ARGS=(--kv-transfer-config "{\"kv_connector\": \"$KV_CONNECTOR\", \"kv_role\": \"$KV_ROLE\"}")
        if ! "$PY" -c "import virthub" >/dev/null 2>&1; then
            log_warning "The '$KV_CONNECTOR' connector needs the virthub package."
            log_warning "Install it with ./install.sh --virthub, or pass --kv-connector none."
        fi
    else
        log_info "Serving without a KV transfer connector."
    fi

    log_info "Model: $MODEL   port: $PORT"
    log_info "Precision: $PSPKV_PRECISION   format: $PSPKV_FORMAT   block size: $BLOCK_SIZE"
    log_info "Config: $PSPKV_CONFIG"

    if [[ "$DRY_RUN" == "1" ]]; then
        log_info "Dry run; the server command would be:"
        echo "  $VENV_DIR/bin/python -m vllm.entrypoints.openai.api_server \\"
        echo "    --model $MODEL --port $PORT --attention-backend PSP_KV \\"
        echo "    --block-size $BLOCK_SIZE ${TRANSFER_ARGS[*]:-} $ENFORCE_EAGER $EXTRA"
        return 0
    fi

    # shellcheck disable=SC2086
    exec "$VENV_DIR/bin/python" -m vllm.entrypoints.openai.api_server \
        --model "$MODEL" \
        --port "$PORT" \
        --attention-backend PSP_KV \
        --block-size "$BLOCK_SIZE" \
        "${TRANSFER_ARGS[@]}" \
        $ENFORCE_EAGER $EXTRA
}

# ------------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------------
run_eval_quality() {
    setup_env
    [[ -f scripts/eval_quality.py ]] || log_error "scripts/eval_quality.py is missing."
    if ! "$PY" -c "import torch; assert torch.cuda.is_available()" >/dev/null 2>&1; then
        log_warning "No CUDA device available; skipping the quality evaluation."
        return 0
    fi
    log_info "Quality and cost evaluation (one engine per configuration) ..."
    PSP_KV_BLOCK_SIZE="$PSPKV_BLOCK_SIZE" "$PY" scripts/eval_quality.py
}

run_eval_formats() {
    setup_env
    [[ -f scripts/eval_kv_formats.py ]] || log_error "scripts/eval_kv_formats.py is missing."
    if ! "$PY" -c "import torch; assert torch.cuda.is_available()" >/dev/null 2>&1; then
        log_warning "No CUDA device available; skipping the format evaluation."
        return 0
    fi
    log_info "Offline format/scale-granularity attribution ..."
    PSP_KV_BLOCK_SIZE="$PSPKV_BLOCK_SIZE" "$PY" scripts/eval_kv_formats.py
}

# ------------------------------------------------------------------
# Bench
# ------------------------------------------------------------------
run_benchmarks() {
    setup_env
    if ! "$PY" -c "import torch; assert torch.cuda.is_available()" >/dev/null 2>&1; then
        log_warning "No CUDA device available; skipping the benchmark."
        return 0
    fi
    local BENCH="benchmarks/psp_kv/decode_throughput.py"
    [[ -f "$BENCH" ]] || log_error "$BENCH is missing."
    log_info "Decode-throughput benchmark (FlashAttention baseline vs PSP-KV) ..."
    "$PY" "$BENCH"
    log_success "Benchmark completed."
}

# ------------------------------------------------------------------
# Clean
# ------------------------------------------------------------------
run_clean() {
    log_info "Removing generated build artifacts and caches ..."
    # Everything here is regenerated by ./run.sh --build (or ./install.sh) and
    # nothing else in the tree depends on it.
    rm -rf "$BUILD_DIR" build build_psp .build .deps dist ./*.egg-info
    rm -rf .ruff_cache .pytest_cache
    # The PSP-KV extension and the stale probe build; the upstream extensions
    # that `install.sh` copies from the wheel are kept so vLLM still imports.
    find vllm -maxdepth 1 -name "_C.abi3.so" -delete 2>/dev/null || true
    find vllm -maxdepth 1 -name "_psp_*probe*.so" -delete 2>/dev/null || true
    find . -maxdepth 3 -name "CMakeCache.txt" -delete 2>/dev/null || true
    find . -maxdepth 1 -name "bench_*.txt" -delete 2>/dev/null || true
    find . -path "./$VENV_DIR" -prune -o -type d -name "__pycache__" \
        -prune -exec rm -rf {} + 2>/dev/null || true
    log_success "Clean completed. Re-run ./run.sh --build to rebuild the extension."
}

run_clean_all() {
    run_clean
    log_info "Removing the virtual environment and the fetched Virthub sources ..."
    rm -rf "$VENV_DIR" .venv tools/virthub
    # The upstream extensions are re-copied by install.sh; drop them too so a
    # deep clean leaves only tracked files behind.
    find vllm -maxdepth 1 -name "*.so" -delete 2>/dev/null || true
    log_success "Deep clean completed. Re-run ./install.sh to rebuild."
}

# ==================================================================
# Dispatch
# ==================================================================
if [[ $# -eq 0 ]]; then
    show_help
    exit 0
fi

COMMAND="$1"
shift

case "$COMMAND" in
    --clean)     run_clean; exit 0 ;;
    --clean-all) run_clean_all; exit 0 ;;
    --help|-h)   show_help; exit 0 ;;
esac

case "$COMMAND" in
    --build)            run_build ;;
    --test|-t)          run_all_tests ;;
    --test-unit)        run_unit_tests ;;
    --test-integration) run_integration_tests ;;
    --test-virthub)     run_virthub_tests ;;
    --eval-quality)     run_eval_quality ;;
    --eval-formats)     run_eval_formats ;;
    --serve|-s)         serve_model "$@" ;;
    --bench|-b)         run_benchmarks ;;
    *)                  log_error "Unknown command: $COMMAND (try --help)" ;;
esac
