#!/bin/bash
# run.sh – Unified command runner for vLLM with PSP‑KV
# Usage: ./run.sh [COMMAND] [OPTIONS]
#
# Uses `uv` as the default environment manager. All Python invocations go
# through `uv run`, which automatically uses the project virtual environment
# (`.venv`) and ensures dependencies are present.
#
# `--clean` and `--help` are handled before any uv/venv checks, so they work
# even if the environment is not yet set up.

set -e

# ------------------------------------------------------------------
# Coloured logging (safe for non‑TTY)
# ------------------------------------------------------------------
if [[ -t 1 ]]; then
    readonly BOLD=$(tput bold 2>/dev/null || echo "")
    readonly RED=$(tput setaf 1 2>/dev/null || echo "")
    readonly GREEN=$(tput setaf 2 2>/dev/null || echo "")
    readonly YELLOW=$(tput setaf 3 2>/dev/null || echo "")
    readonly BLUE=$(tput setaf 4 2>/dev/null || echo "")
    readonly RESET=$(tput sgr0 2>/dev/null || echo "")
else
    readonly BOLD="" RED="" GREEN="" YELLOW="" BLUE="" RESET=""
fi

log_info()    { echo "${BLUE}${BOLD}[INFO]${RESET} $*"; }
log_success() { echo "${GREEN}${BOLD}[SUCCESS]${RESET} $*"; }
log_warning() { echo "${YELLOW}${BOLD}[WARNING]${RESET} $*" >&2; }
log_error()   { echo "${RED}${BOLD}[ERROR]${RESET} $*" >&2; exit 1; }

# ------------------------------------------------------------------
# Ensure we are in the project root
# ------------------------------------------------------------------
if [ ! -f "setup.py" ] || [ ! -d "vllm" ]; then
    log_error "Please run this script from the vLLM project root directory."
fi

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

# ==================================================================
# Function definitions
# ==================================================================

# ------------------------------------------------------------------
# ensure_uv – make sure the `uv` CLI is available, installing it via
# pip if necessary.
# ------------------------------------------------------------------
ensure_uv() {
    if command -v uv &> /dev/null; then
        return 0
    fi

    log_warning "uv is not installed. Attempting to install it via pip ..."

    if command -v pip3 &> /dev/null; then
        pip3 install --user uv || pip3 install uv || \
            log_error "Failed to install uv via pip3. Please install it manually: https://docs.astral.sh/uv/"
    elif command -v pip &> /dev/null; then
        pip install --user uv || pip install uv || \
            log_error "Failed to install uv via pip. Please install it manually: https://docs.astral.sh/uv/"
    else
        log_error "Neither pip nor pip3 found. Please install uv manually: https://docs.astral.sh/uv/"
    fi

    # Refresh PATH for user‑installed binaries (pip --user installs to ~/.local/bin)
    if [ -d "$HOME/.local/bin" ]; then
        export PATH="$HOME/.local/bin:$PATH"
    fi

    if ! command -v uv &> /dev/null; then
        log_error "uv was installed but is not on PATH. Please add ~/.local/bin to your PATH."
    fi

    log_success "uv installed: $(uv --version)"
}

# ------------------------------------------------------------------
# ensure_venv – verify the project virtual environment exists.
# ------------------------------------------------------------------
ensure_venv() {
    if [ ! -d ".venv" ]; then
        log_error "Virtual environment .venv not found. Please run ./install.sh first."
    fi
}

# ------------------------------------------------------------------
# setup_env – compiler and CUDA environment variables.
# These are needed if a test triggers a rebuild.
# ------------------------------------------------------------------
setup_env() {
    export CC=/usr/bin/gcc-12
    export CXX=/usr/bin/g++-12
    export CUDAHOSTCXX=/usr/bin/g++-12
    export TORCH_CUDA_ARCH_LIST="8.9"

    export CUDA_HOME=/usr/local/cuda
    export PATH="$CUDA_HOME/bin:$PATH"
    export CUDACXX="$CUDA_HOME/bin/nvcc"

    # pythoncapi-compat header (defensive, only needed if rebuild happens)
    PYTHONCAPI_COMPAT_DIR="/tmp/pythoncapi-compat"
    if [ -f "$PYTHONCAPI_COMPAT_DIR/include/pythoncapi_compat.h" ]; then
        export CXXFLAGS="-I$PYTHONCAPI_COMPAT_DIR/include ${CXXFLAGS:-}"
        export CFLAGS="-I$PYTHONCAPI_COMPAT_DIR/include ${CFLAGS:-}"
    fi

    export CMAKE_ARGS="-DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12"

    # Disable Triton kernels (matches CMakeLists.txt option)
    export VLLM_BUILD_TRITON_KERNELS="${VLLM_BUILD_TRITON_KERNELS:-OFF}"

    # PSP‑KV environment variables
    export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-PSP_KV}"
    export VIRTHUB_CONFIG="${VIRTHUB_CONFIG:-./conf/virthub.toml}"
    export VIRTHUB_FORMAT="${VIRTHUB_FORMAT:-1}"
}

# ------------------------------------------------------------------
# Help message
# ------------------------------------------------------------------
show_help() {
    cat << EOF
${BOLD}Usage:${RESET} $0 [COMMAND] [OPTIONS]

${BOLD}Commands:${RESET}
  ${BOLD}--test, -t${RESET}                Run all tests (unit + integration).
  ${BOLD}--test-unit${RESET}               Run only unit tests (fast).
  ${BOLD}--test-integration${RESET}        Run end‑to‑end inference tests (requires GPU).
  ${BOLD}--test-virthub${RESET}            Run Virthub KV-cache connector tests (requires ./install.sh --virthub).
  ${BOLD}--serve, -s${RESET}               Start the vLLM server with PSP‑KV.
  ${BOLD}--bench, -b${RESET}               Run performance benchmarks.
  ${BOLD}--clean${RESET}                   Remove build artifacts and generated files.
  ${BOLD}--help, -h${RESET}                Show this help message.

${BOLD}Options for --serve:${RESET}
  --model MODEL_NAME            Model to serve (default: facebook/opt-125m).
  --port PORT                   Port to listen on (default: 8000).
  --format 0|1|2                Override PSP‑KV format generation (0=Basic, 1=Enhanced, 2=BT‑KV).
  --enforce-eager               Disable CUDA graphs (useful for debugging).
  --kv-connector CONNECTOR      KV connector to use (default: virthub).
  --kv-role ROLE                KV connector role (default: kv_both).

${BOLD}Environment Variables (optional):${RESET}
  VLLM_ATTENTION_BACKEND   Attention backend (default: PSP_KV).
  VIRTHUB_CONFIG           Path to config file (default: ./conf/virthub.toml).
  VIRTHUB_FORMAT           Override format (can also use --format).

${BOLD}Examples:${RESET}
  $0 --test
  $0 --serve --model meta-llama/Llama-2-7b-hf --port 8080 --format 2
  $0 --clean
EOF
}

# ------------------------------------------------------------------
# Test functions
# ------------------------------------------------------------------
run_unit_tests() {
    log_info "Running unit tests..."
    # --noconftest: the PSP-KV tests are self-contained and do not use any
    # fixture from tests/conftest.py, which pulls in heavy optional deps
    # (transformers, huggingface_hub, PIL). Skipping it lets the CPU unit
    # tests run without the full vLLM test dependency stack.
    uv run --no-sync pytest tests/v1/attention/test_psp_kv_*.py -v --noconftest \
        || log_error "Unit tests failed."
    log_success "Unit tests passed."
}

run_integration_tests() {
    log_info "Running integration tests (requires GPU and a small model)..."

    local MODEL="facebook/opt-125m"
    local PORT="8000"

    log_info "Starting vLLM server with $MODEL on port $PORT ..."
    uv run --no-sync vllm serve "$MODEL" \
        --port "$PORT" \
        --attention-backend psp_kv \
        --kv-cache-dtype fp8 \
        --kv-transfer-config '{"kv_connector": "virthub", "kv_role": "kv_both"}' \
        --enforce-eager \
        --max-num-seqs 1 \
        --max-model-len 64 \
        --gpu-memory-utilization 0.3 &
    local SERVER_PID=$!

    log_info "Waiting for server to start (up to 60s)..."
    local RETRIES=60
    local COUNT=0
    until curl -s --max-time 5 "http://localhost:$PORT/health" >/dev/null 2>&1 \
       || curl -s --max-time 5 "http://localhost:$PORT/v1/models" >/dev/null 2>&1; do
        sleep 1
        COUNT=$((COUNT + 1))
        if [ $COUNT -ge $RETRIES ]; then
            kill $SERVER_PID 2>/dev/null || true
            log_error "Server failed to start within $RETRIES seconds."
        fi
    done

    log_info "Sending test request..."
    local RESPONSE=$(curl -s --max-time 60 -X POST "http://localhost:$PORT/v1/completions" \
         -H "Content-Type: application/json" \
         -d '{"model":"'"$MODEL"'","prompt":"The capital of France is","max_tokens":8,"temperature":0.0}')

    if echo "$RESPONSE" | grep -q '"text"'; then
        log_success "Integration test passed: received valid response."
    else
        kill $SERVER_PID 2>/dev/null || true
        log_error "Integration test failed: invalid response from server.\nResponse: $RESPONSE"
    fi

    kill $SERVER_PID 2>/dev/null || true
    log_success "Integration tests passed."
}

run_all_tests() {
    log_info "Running all tests..."
    run_unit_tests
    run_integration_tests
    log_success "All tests passed!"
}

run_virthub_tests() {
    log_info "Running Virthub KV-cache connector tests..."
    if ! uv run --no-sync python -c "import virthub" >/dev/null 2>&1; then
        log_error "Virthub bindings are not installed. Run ./install.sh --virthub first."
    fi

    # 1. Fork-level integration test (self-contained; skip the fork's heavy conftest).
    uv run --no-sync pytest tests/v1/kv_connector/test_virthub_connector.py \
        -v --noconftest || log_error "Virthub connector tests failed."

    # 2. Virthub's own vLLM connector unit tests (in-memory stubs, no GPU/daemon).
    uv run --no-sync pytest \
        third_party/virthub/tests/python/test_vllm_connector.py \
        -v || log_error "Virthub unit tests failed."

    log_success "Virthub connector tests passed."
}

# ------------------------------------------------------------------
# Serve function
# ------------------------------------------------------------------
serve_model() {
    local MODEL="facebook/opt-125m"
    local PORT="8000"
    local ENFORCE_EAGER=""
    local KV_CONNECTOR="virthub"
    local KV_ROLE="kv_both"
    local ATTENTION_BACKEND="psp_kv"
    local FORMAT=""

    while [[ $# -gt 0 ]]; do
        case "$1" in
            --model)             MODEL="$2"; shift 2 ;;
            --port)              PORT="$2"; shift 2 ;;
            --format)            FORMAT="$2"; shift 2 ;;
            --enforce-eager)     ENFORCE_EAGER="--enforce-eager"; shift ;;
            --kv-connector)      KV_CONNECTOR="$2"; shift 2 ;;
            --kv-role)           KV_ROLE="$2"; shift 2 ;;
            --attention-backend) ATTENTION_BACKEND="$2"; shift 2 ;;
            *)                   log_error "Unknown option for --serve: $1" ;;
        esac
    done

    if [ -n "$FORMAT" ]; then
        export VIRTHUB_FORMAT="$FORMAT"
        log_info "Override format generation to $FORMAT"
    fi

    log_info "Starting vLLM server with model $MODEL on port $PORT ..."
    log_info "Attention backend: $ATTENTION_BACKEND"
    log_info "Config file: $VIRTHUB_CONFIG"
    if [ -n "$VIRTHUB_FORMAT" ]; then
        log_info "Format override: $VIRTHUB_FORMAT"
    fi

    uv run --no-sync vllm serve "$MODEL" \
        --port "$PORT" \
        --attention-backend "$ATTENTION_BACKEND" \
        --kv-cache-dtype fp8 \
        --kv-transfer-config "{\"kv_connector\": \"$KV_CONNECTOR\", \"kv_role\": \"$KV_ROLE\"}" \
        $ENFORCE_EAGER
}

# ------------------------------------------------------------------
# Benchmark (placeholder)
# ------------------------------------------------------------------
run_benchmarks() {
    log_info "Running benchmarks (not yet implemented)..."
    log_warning "Please implement benchmark scripts as needed."
    log_success "Benchmark completed."
}

# ------------------------------------------------------------------
# Clean function – removes build artifacts and generated files.
# Does NOT require uv or .venv.
# ------------------------------------------------------------------
run_clean() {
    log_info "Cleaning build artifacts and generated files..."

    if [ -d "build" ]; then
        log_info "Removing build/ ..."
        rm -rf build
    fi

    for dir in dist *.egg-info; do
        if [ -d "$dir" ]; then
            log_info "Removing $dir ..."
            rm -rf "$dir"
        fi
    done

    log_info "Removing compiled extensions in vllm/ ..."
    find vllm -maxdepth 2 -name "*.so" -type f -delete 2>/dev/null || true
    find vllm -maxdepth 2 -name "_C*.so" -type f -delete 2>/dev/null || true

    if [ -f "vllm/_version.py" ]; then
        log_info "Removing vllm/_version.py ..."
        rm -f vllm/_version.py
    fi

    log_info "Removing __pycache__ and .pyc files ..."
    find . -type d -name "__pycache__" -prune -exec rm -rf {} + 2>/dev/null || true
    find . -type f -name "*.pyc" -delete 2>/dev/null || true

    find . -maxdepth 2 -name "CMakeCache.txt" -delete 2>/dev/null || true
    find . -maxdepth 2 -name "cmake_install.cmake" -delete 2>/dev/null || true

    for f in error.log full_build.log cmake_output.log; do
        if [ -f "$f" ]; then
            log_info "Removing $f ..."
            rm -f "$f"
        fi
    done

    log_success "Clean completed."
}

# ==================================================================
# Main command parser
# ==================================================================

if [ $# -eq 0 ]; then
    show_help
    exit 0
fi

COMMAND="$1"
shift

# ------------------------------------------------------------------
# Early-exit commands that do NOT require uv or .venv
# ------------------------------------------------------------------
case "$COMMAND" in
    --clean)
        run_clean
        exit 0
        ;;
    --help|-h)
        show_help
        exit 0
        ;;
esac

# ------------------------------------------------------------------
# Writable-dir fallback for uv.
#
# Containers / CI sandboxes sometimes mount $HOME (or ~/.cache and ~/.local)
# read-only. uv needs a writable cache and managed-Python install dir. Redirect
# them into the project tree only when their default location is not writable,
# so a normal user setup is left untouched. (install.sh applies the same fix.)
# ------------------------------------------------------------------
RUNTIME_DIR="${RUNTIME_DIR:-$PROJECT_ROOT/.build}"
mkdir -p "$RUNTIME_DIR"

_writable_path() {
    local p="$1"
    while [[ -n "$p" && ! -e "$p" ]]; do
        p="$(dirname "$p")"
    done
    [[ -n "$p" && -w "$p" ]]
}

if ! _writable_path "${UV_CACHE_DIR:-$HOME/.cache/uv}"; then
    export UV_CACHE_DIR="$RUNTIME_DIR/uv-cache"
    log_info "uv cache dir is not writable; using $UV_CACHE_DIR"
fi

if ! _writable_path "${UV_PYTHON_INSTALL_DIR:-$HOME/.local/share/uv/python}"; then
    export UV_PYTHON_INSTALL_DIR="$RUNTIME_DIR/uv-python"
    log_info "uv Python install dir is not writable; using $UV_PYTHON_INSTALL_DIR"
fi

# ------------------------------------------------------------------
# For all remaining commands we need uv and the project venv
# ------------------------------------------------------------------
ensure_uv
ensure_venv

# Deactivate conda if active (avoids wrong interpreter in PATH)
if command -v conda &> /dev/null; then
    conda deactivate 2>/dev/null || true
fi

# Verify Python version inside the uv environment
PYTHON_VERSION=$(uv run --no-sync python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
if [ "$PYTHON_VERSION" != "3.10" ]; then
    log_warning "Python version is $PYTHON_VERSION, but vLLM 0.20.2 expects Python 3.10."
fi

# Set up compiler / CUDA / PSP‑KV environment
setup_env

# Ensure pytest is available inside the uv environment
if ! uv run --no-sync python -c "import pytest" >/dev/null 2>&1; then
    log_info "Installing pytest into the uv environment ..."
    uv pip install pytest || log_warning "Could not install pytest; unit tests may fail."
fi

# ------------------------------------------------------------------
# Dispatch remaining commands
# ------------------------------------------------------------------
case "$COMMAND" in
    --test|-t)              run_all_tests ;;
    --test-unit)            run_unit_tests ;;
    --test-integration)     run_integration_tests ;;
    --test-virthub)         run_virthub_tests ;;
    --serve|-s)             serve_model "$@" ;;
    --bench|-b)             run_benchmarks ;;
    *)                      log_error "Unknown command: $COMMAND" ;;
esac
