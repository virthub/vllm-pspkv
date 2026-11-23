#!/bin/bash
# install.sh – provision the environment for vLLM with PSP-KV.
#
# This fork ships the PSP-KV kernels as a *separate* extension (`vllm/_C.abi3.so`)
# built on top of an unmodified upstream vLLM wheel. Installing therefore has
# three steps, all handled here:
#
#   1. create (or reuse) a Python virtual environment;
#   2. install the matching upstream vLLM wheel, which supplies vLLM's own
#      compiled extensions (FlashAttention, CUTLASS, MoE, ...);
#   3. build the PSP-KV extension from `csrc/` with the small standalone CMake
#      project in `cmake/psp_kv` and place it next to the sources.
#
# Virthub (the KV-cache storage daemon that PSP-KV can hand blocks to) is
# optional and installed best-effort with `--virthub`.
#
# Usage:
#   ./install.sh                     # full setup (venv + wheel + PSP extension)
#   ./install.sh --virthub           # ... and try to provision Virthub too
#   ./install.sh --no-build          # environment only, skip the PSP extension
#   ./install.sh --clean             # remove the venv and build artifacts
#   ./install.sh --help

set -euo pipefail

# ------------------------------------------------------------------
# Configuration (override through the environment)
# ------------------------------------------------------------------
VLLM_VERSION="${VLLM_VERSION:-0.34.0}"
VENV_DIR="${VENV_DIR:-.venv34}"
PYTHON_BIN="${PYTHON_BIN:-}"
BUILD_DIR="${BUILD_DIR:-build/psp}"
BUILD_JOBS="${BUILD_JOBS:-}"
PSPKV_CUDA_ARCH="${PSPKV_CUDA_ARCH:-120a}"
PSPKV_SCALE_GROUP="${PSPKV_SCALE_GROUP:-16}"
VIRTHUB_COMMIT="${VIRTHUB_COMMIT:-4c85bed03cb59336752f77f9569ceb52e3d3bcdf}"
VIRTHUB_DIR="${VIRTHUB_DIR:-tools/virthub}"

DO_BUILD=1
DO_VIRTHUB=0
DO_CLEAN=0
SKIP_DEPS=0

# ------------------------------------------------------------------
# Coloured logging (safe for non-TTY)
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

show_help() {
    cat <<EOF
${BOLD}Usage:${RESET} $0 [OPTIONS]

${BOLD}Options:${RESET}
  ${BOLD}--python PATH${RESET}        Python interpreter used to create the venv
                          (default: python3.12, then python3)
  ${BOLD}--venv DIR${RESET}           Virtual environment directory (default: $VENV_DIR)
  ${BOLD}--vllm-version VER${RESET}  Upstream vLLM wheel to install (default: $VLLM_VERSION)
  ${BOLD}--no-build${RESET}           Skip building the PSP-KV extension
  ${BOLD}--no-deps${RESET}            Skip installing the upstream wheel (use an
                          existing vLLM installation for its extensions)
  ${BOLD}--virthub${RESET}            Best-effort install of the optional Virthub
                          package (KV-cache storage connector)
  ${BOLD}--cuda-arch ARCH${RESET}     CUDA architecture for the PSP kernels
                          (default: $PSPKV_CUDA_ARCH)
  ${BOLD}--scale-group N${RESET}      16 or 32-wide KV scale groups (default: $PSPKV_SCALE_GROUP)
  ${BOLD}--jobs N${RESET}             Parallel build jobs (default: all cores)
  ${BOLD}--clean${RESET}              Remove the venv, build artifacts and the
                          compiled PSP extension
  ${BOLD}--help, -h${RESET}           Show this help message

${BOLD}Environment:${RESET}
  VLLM_VERSION, VENV_DIR, PYTHON_BIN, BUILD_JOBS, PSPKV_CUDA_ARCH,
  PSPKV_SCALE_GROUP, VIRTHUB_COMMIT

${BOLD}Examples:${RESET}
  $0
  $0 --virthub --cuda-arch 120a
  $0 --no-build --venv .venv
EOF
}

# ------------------------------------------------------------------
# Argument parsing
# ------------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case "$1" in
        --python)        PYTHON_BIN="$2"; shift 2 ;;
        --venv)          VENV_DIR="$2"; shift 2 ;;
        --vllm-version)  VLLM_VERSION="$2"; shift 2 ;;
        --no-build)      DO_BUILD=0; shift ;;
        --no-deps)       SKIP_DEPS=1; shift ;;
        --virthub)       DO_VIRTHUB=1; shift ;;
        --cuda-arch)     PSPKV_CUDA_ARCH="$2"; shift 2 ;;
        --scale-group)   PSPKV_SCALE_GROUP="$2"; shift 2 ;;
        --jobs)          BUILD_JOBS="$2"; shift 2 ;;
        --clean)         DO_CLEAN=1; shift ;;
        --help|-h)       show_help; exit 0 ;;
        *)               log_error "Unknown option: $1 (try --help)" ;;
    esac
done

# ------------------------------------------------------------------
# Project root
# ------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
[[ -d vllm ]] || log_error "Please run this script from the vLLM project root directory."
[[ -f cmake/psp_kv/CMakeLists.txt ]] || \
    log_error "cmake/psp_kv/CMakeLists.txt is missing; the PSP-KV build project is required."

# ------------------------------------------------------------------
# Clean
# ------------------------------------------------------------------
if [[ "$DO_CLEAN" == "1" ]]; then
    log_info "Cleaning the environment and build artifacts ..."
    rm -rf "$VENV_DIR" build build_psp .build dist ./*.egg-info "$VIRTHUB_DIR"
    find vllm -maxdepth 1 -name "_C.abi3.so" -delete 2>/dev/null || true
    find vllm -maxdepth 1 -name "_C*.so" -delete 2>/dev/null || true
    log_success "Clean completed."
    exit 0
fi

# ------------------------------------------------------------------
# Python interpreter
# ------------------------------------------------------------------
if [[ -z "$PYTHON_BIN" ]]; then
    for candidate in python3.12 python3; do
        if command -v "$candidate" >/dev/null 2>&1; then
            PYTHON_BIN="$candidate"
            break
        fi
    done
fi
[[ -n "$PYTHON_BIN" ]] || log_error "No Python interpreter found (use --python PATH)."
PY_VERSION="$("$PYTHON_BIN" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
log_info "Using Python $PY_VERSION ($PYTHON_BIN)"
case "$PY_VERSION" in
    3.10|3.11|3.12|3.13) ;;
    *) log_warning "Python $PY_VERSION is untested; vLLM $VLLM_VERSION supports 3.10-3.13." ;;
esac

# ------------------------------------------------------------------
# Virtual environment
# ------------------------------------------------------------------
if [[ ! -d "$VENV_DIR" ]]; then
    log_info "Creating virtual environment in $VENV_DIR ..."
    "$PYTHON_BIN" -m venv "$VENV_DIR"
fi
PY="$VENV_DIR/bin/python"
[[ -x "$PY" ]] || log_error "$VENV_DIR does not contain a usable python."

# Some environments create venvs without pip (`uv venv`, distributions with
# ensurepip disabled). Try the built-in bootstraps, then any system pip, then
# the official wheel bootstrap script.
if ! "$PY" -m pip --version >/dev/null 2>&1; then
    log_info "Bootstrapping pip in $VENV_DIR ..."
    "$PY" -m ensurepip --upgrade >/dev/null 2>&1 || true
fi
if ! "$PY" -m pip --version >/dev/null 2>&1; then
    for sys_python in python3 python; do
        if command -v "$sys_python" >/dev/null 2>&1 && \
           "$sys_python" -m pip --version >/dev/null 2>&1; then
            "$sys_python" -m pip --python "$PY" install --quiet pip >/dev/null 2>&1 || true
            break
        fi
    done
fi
if ! "$PY" -m pip --version >/dev/null 2>&1 && command -v curl >/dev/null 2>&1; then
    log_info "Fetching get-pip.py ..."
    if curl -sSL --max-time 120 https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py 2>/dev/null; then
        "$PY" /tmp/get-pip.py --quiet >/dev/null 2>&1 || true
        rm -f /tmp/get-pip.py
    fi
fi
"$PY" -m pip --version >/dev/null 2>&1 || \
    log_error "Could not bootstrap pip in $VENV_DIR; create the venv with 'python -m venv' instead."
"$PY" -m pip install --quiet --upgrade pip >/dev/null 2>&1 || \
    log_warning "Could not upgrade pip; continuing with the existing version."

# Developer tooling used by ./run.sh --test.
"$PY" -m pip install --quiet pytest >/dev/null 2>&1 || \
    log_warning "Could not install pytest; ./run.sh --test-unit will not work."

# ------------------------------------------------------------------
# Upstream vLLM wheel (provides vLLM's own compiled extensions)
# ------------------------------------------------------------------
if [[ "$SKIP_DEPS" == "1" ]]; then
    log_info "Skipping the upstream wheel (--no-deps)."
else
    INSTALLED="$("$PY" - <<'PYCODE' 2>/dev/null || true
try:
    import importlib.metadata as md
    print(md.version("vllm"))
except Exception:
    print("")
PYCODE
)"
    if [[ -n "$INSTALLED" ]]; then
        log_info "Upstream vLLM $INSTALLED is already installed in $VENV_DIR."
    else
        log_info "Installing upstream vLLM $VLLM_VERSION (this pulls torch and the prebuilt kernels) ..."
        "$PY" -m pip install "vllm==$VLLM_VERSION" || \
            log_error "Could not install vLLM $VLLM_VERSION. Set --vllm-version or use --no-deps."
        log_success "Upstream vLLM installed."
    fi

    # The fork runs from the source tree, so vLLM's compiled extensions must be
    # visible next to the sources. Copy them out of the installed wheel (never
    # overwrite the PSP extension, which is built here).
    SITE_PKG="$("$PY" -c 'import vllm, os; print(os.path.dirname(vllm.__file__))')"
    if [[ "$(cd "$SITE_PKG" && pwd)" != "$(pwd)/vllm" ]]; then
        log_info "Copying vLLM's compiled extensions from $SITE_PKG ..."
        for so in "$SITE_PKG"/*.so; do
            [[ -e "$so" ]] || continue
            base="$(basename "$so")"
            [[ "$base" == "_C.abi3.so" ]] && continue
            cp -f "$so" "vllm/$base"
        done
        log_success "Compiled extensions copied (PSP-KV extension excluded)."
    else
        log_info "vLLM is already imported from the source tree; nothing to copy."
    fi
fi

# ------------------------------------------------------------------
# CUDA toolchain sanity
# ------------------------------------------------------------------
if [[ "$DO_BUILD" == "1" ]]; then
    if [[ -z "${CUDA_HOME:-}" ]]; then
        for candidate in /usr/local/cuda /usr/local/cuda-13.0 /usr/local/cuda-12.8; do
            [[ -x "$candidate/bin/nvcc" ]] && CUDA_HOME="$candidate" && break
        done
    fi
    [[ -n "${CUDA_HOME:-}" && -x "$CUDA_HOME/bin/nvcc" ]] || \
        log_error "nvcc not found. Install the CUDA toolkit or set CUDA_HOME."
    export CUDA_HOME
    export PATH="$CUDA_HOME/bin:$PATH"
    command -v cmake >/dev/null 2>&1 || log_error "cmake is required to build the PSP-KV extension."
    command -v ninja >/dev/null 2>&1 || log_warning "ninja not found; using the default CMake generator."
fi

# ------------------------------------------------------------------
# Build the PSP-KV extension
# ------------------------------------------------------------------
if [[ "$DO_BUILD" == "1" ]]; then
    TORCH_CMAKE="$("$PY" -c 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__), "share", "cmake"))')"
    [[ -d "$TORCH_CMAKE" ]] || log_error "torch CMake package not found; install torch in $VENV_DIR."

    GENERATOR=()
    command -v ninja >/dev/null 2>&1 && GENERATOR=(-G Ninja)
    JOBS=()
    [[ -n "$BUILD_JOBS" ]] && JOBS=(-j "$BUILD_JOBS")

    # A cached build tree generated from a different source directory (e.g. an
    # older layout) cannot be reused; drop it and configure fresh.
    if [[ -f "$BUILD_DIR/CMakeCache.txt" ]] && \
       ! grep -q "CMAKE_HOME_DIRECTORY:INTERNAL=$PWD/cmake/psp_kv" "$BUILD_DIR/CMakeCache.txt"; then
        log_warning "Discarding stale build cache in $BUILD_DIR (different source directory)."
        rm -rf "$BUILD_DIR"
    fi

    log_info "Configuring the PSP-KV extension (arch=$PSPKV_CUDA_ARCH, scale_group=$PSPKV_SCALE_GROUP) ..."
    cmake -S cmake/psp_kv -B "$BUILD_DIR" "${GENERATOR[@]}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DPython_EXECUTABLE="$PWD/$PY" \
        -DCMAKE_PREFIX_PATH="$TORCH_CMAKE" \
        -DPSPKV_CUDA_ARCH="$PSPKV_CUDA_ARCH" \
        -DPSPKV_SCALE_GROUP_SIZE="$PSPKV_SCALE_GROUP"

    log_info "Building vllm/_C.abi3.so (this takes a few minutes on a cold cache) ..."
    cmake --build "$BUILD_DIR" --target _C "${JOBS[@]}"
    [[ -f vllm/_C.abi3.so ]] || log_error "Build finished but vllm/_C.abi3.so is missing."
    log_success "PSP-KV extension built: vllm/_C.abi3.so"
fi

# ------------------------------------------------------------------
# Optional: Virthub (KV-cache storage connector)
# ------------------------------------------------------------------
if [[ "$DO_VIRTHUB" == "1" ]]; then
    PYPROJ="$VIRTHUB_DIR/bindings/python"
    if "$PY" -c "import virthub" >/dev/null 2>&1; then
        log_info "Virthub is already importable in $VENV_DIR."
    else
        if [[ ! -f "$PYPROJ/pyproject.toml" ]]; then
            log_info "Fetching Virthub $VIRTHUB_COMMIT ..."
            mkdir -p "$(dirname "$VIRTHUB_DIR")"
            fetched=0
            for base in "" "https://ghproxy.net" "https://ghfast.top"; do
                url="${base:+$base/}https://github.com/virthub/virthub/archive/${VIRTHUB_COMMIT}.tar.gz"
                tmp="$(mktemp -d)"
                if curl -sSL --max-time 600 -o "$tmp/virthub.tar.gz" "$url" 2>/dev/null && \
                   tar -tzf "$tmp/virthub.tar.gz" >/dev/null 2>&1; then
                    rm -rf "$VIRTHUB_DIR"
                    mkdir -p "$VIRTHUB_DIR"
                    tar -xzf "$tmp/virthub.tar.gz" -C "$VIRTHUB_DIR" --strip-components=1
                    fetched=1
                fi
                rm -rf "$tmp"
                [[ "$fetched" == "1" ]] && break
            done
            if [[ "$fetched" != "1" ]]; then
                log_warning "Could not fetch Virthub; continuing without it."
            fi
        fi

        if [[ -f "$PYPROJ/pyproject.toml" ]]; then
            # The bindings are pure Python; the native Rust client is optional.
            # Installing them directly (instead of delegating to Virthub's own
            # installer, which provisions a separate venv and system packages)
            # keeps this environment self-contained.
            log_info "Installing the Virthub bindings (pure Python) into $VENV_DIR ..."
            "$PY" -m pip install --quiet toml >/dev/null 2>&1 || \
                log_warning "Could not install 'toml' (the connector imports it)."
            if "$PY" -m pip install --quiet -e "$PYPROJ" >/dev/null 2>&1; then
                log_success "Virthub bindings installed."
                log_info "The native Rust client is optional; build it later with"
                log_info "maturin if you need the real transport instead of the stub."
            else
                log_warning "Virthub bindings failed to install; continuing without them."
            fi
        fi
    fi
    log_info "The 'virthub' KV connector is adapted to Model Runner V2 in"
    log_info "vllm/distributed/kv_transfer/kv_connector/v1/virthub_connector.py;"
    log_info "select it with --kv-transfer-config '{\"kv_connector\": \"virthub\"}'."
fi

# ------------------------------------------------------------------
# Summary
# ------------------------------------------------------------------
echo
log_success "Installation completed."
cat <<EOF

${BOLD}Next steps${RESET}
  ./run.sh --test-unit          # CPU unit tests (PSP-KV kernels + Virthub adapter)
  ./run.sh --test-integration   # GPU end-to-end check against FlashAttention
  ./run.sh --serve --model Qwen/Qwen3-0.6B
  ./run.sh --help

PSP-KV reads its launcher configuration from conf/pspkv.toml (PSPKV_CONFIG);
conf/virthub.toml is kept as a backward-compatible fallback (VIRTHUB_CONFIG).
EOF
