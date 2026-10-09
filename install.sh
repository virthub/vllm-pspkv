#!/usr/bin/env bash
# install.sh – Build and install vLLM with PSP-KV using GCC 12
#
# Uses `uv` as the default environment manager. Torch is installed from a
# mirror first (unless the user has already configured one), then the official
# PyTorch index. Dependency tarballs are fetched via a list of GitHub mirrors
# with fallback. Existing user mirror configuration is detected and preferred,
# but our fast mirrors are always kept as a fallback.
#
# Optional external projects (DeepGEMM, FlashMLA, QuTLASS, Triton kernels)
# are disabled by default because they are not required by PSP-KV and their
# network fetches routinely fail. Enable them by exporting the corresponding
# VLLM_BUILD_* variables (e.g. VLLM_BUILD_DEEPGEMM=ON) before running this
# script.
#
# Required external projects (CUTLASS, vLLM Flash Attention) are pre-fetched
# into .deps/ so that CMake never attempts a network clone for them.
#
# Virthub (KV-cache storage management) is installed BY DEFAULT into
# third_party/virthub via Virthub's own install.sh; pass --no-virthub to skip.

set -euo pipefail

# ------------------------------------------------------------------
# Coloured logging (safe for non-TTY)
# ------------------------------------------------------------------
if [[ -t 1 ]]; then
    readonly BOLD=$(tput bold 2>/dev/null || true)
    readonly RED=$(tput setaf 1 2>/dev/null || true)
    readonly GREEN=$(tput setaf 2 2>/dev/null || true)
    readonly YELLOW=$(tput setaf 3 2>/dev/null || true)
    readonly BLUE=$(tput setaf 4 2>/dev/null || true)
    readonly RESET=$(tput sgr0 2>/dev/null || true)
else
    readonly BOLD="" RED="" GREEN="" YELLOW="" BLUE="" RESET=""
fi

log_info()    { echo "${BLUE}${BOLD}[INFO]${RESET} $*"; }
log_success() { echo "${GREEN}${BOLD}[SUCCESS]${RESET} $*"; }
log_warning() { echo "${YELLOW}${BOLD}[WARNING]${RESET} $*" >&2; }
log_error()   { echo "${RED}${BOLD}[ERROR]${RESET} $*" >&2; exit 1; }

# ------------------------------------------------------------------
# Project root (symlink-safe)
# ------------------------------------------------------------------
get_script_dir() {
    local source="${BASH_SOURCE[0]}"
    while [[ -h "$source" ]]; do
        local dir
        dir="$(cd -P "$(dirname "$source")" && pwd)"
        source="$(readlink "$source")"
        [[ "$source" != /* ]] && source="$dir/$source"
    done
    cd -P "$(dirname "$source")" && pwd
}

readonly PROJECT_DIR="$(get_script_dir)"
readonly VENV_DIR="$PROJECT_DIR/.venv"
readonly DEPS_DIR="$PROJECT_DIR/.deps"
readonly PYTHONCAPI_COMPAT_DIR="/tmp/pythoncapi-compat"

# ------------------------------------------------------------------
# Writable-dir fallback for tools that default into $HOME
#
# Containers and CI sandboxes sometimes mount $HOME (or just ~/.cache
# and ~/.local) read-only. uv needs a writable cache directory and, if it
# has to download a managed Python, a writable install directory. Redirect
# both into the project tree (which the build writes to anyway) only when
# their default location is not writable, so a normal user setup is left
# untouched. The override is also exported for the nested `uv pip install`
# and build steps, which inherit the same cache.
# ------------------------------------------------------------------
RUNTIME_DIR="${RUNTIME_DIR:-$PROJECT_DIR/.build}"
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

cd "$PROJECT_DIR"
log_info "Project root: $PROJECT_DIR"

# ==================================================================
# CLI flags
#
#   Virthub (KV-cache storage management) is installed BY DEFAULT. It is
#   fetched into third_party/virthub and installed via Virthub's own
#   install.sh (Rust toolchain, deps, Python bindings).
#   --no-virthub  skip the Virthub install.
# ==================================================================
VLLM_BUILD_VIRTHUB="${VLLM_BUILD_VIRTHUB:-ON}"
for arg in "$@"; do
    case "$arg" in
        --virthub)    VLLM_BUILD_VIRTHUB=ON ;;
        --no-virthub) VLLM_BUILD_VIRTHUB=OFF ;;
        *) log_warning "Unknown argument: $arg (ignored)" ;;
    esac
done

# ==================================================================
# Detach from any active venv so `python`/`pip` resolve outside it.
# This prevents us from picking up the project venv's own interpreter,
# which we may delete and recreate later.
# ==================================================================
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    log_info "Deactivating active virtual environment: $VIRTUAL_ENV"
    NEW_PATH=""
    IFS=':' read -ra PARTS <<< "$PATH"
    for p in "${PARTS[@]}"; do
        [[ "$p" == "$VIRTUAL_ENV/bin"* ]] && continue
        NEW_PATH+="${NEW_PATH:+:}$p"
    done
    export PATH="$NEW_PATH"
    unset VIRTUAL_ENV
fi

if [[ ":$PATH:" == *":$VENV_DIR/bin:"* ]]; then
    NEW_PATH=""
    IFS=':' read -ra PARTS <<< "$PATH"
    for p in "${PARTS[@]}"; do
        [[ "$p" == "$VENV_DIR/bin"* ]] && continue
        NEW_PATH+="${NEW_PATH:+:}$p"
    done
    export PATH="$NEW_PATH"
fi

# ==================================================================
# Detect existing user mirror configuration
# ==================================================================

user_has_pypi_mirror() {
    [[ -n "${UV_INDEX_URL:-}"        ]] && return 0
    [[ -n "${UV_DEFAULT_INDEX:-}"    ]] && return 0
    [[ -n "${PIP_INDEX_URL:-}"       ]] && return 0
    [[ -n "${PIP_EXTRA_INDEX_URL:-}" ]] && return 0
    [[ -n "${UV_CONFIG_FILE:-}"  && -f "${UV_CONFIG_FILE}"  ]] && return 0
    [[ -n "${PIP_CONFIG_FILE:-}" && -f "${PIP_CONFIG_FILE}" ]] && return 0
    [[ -f "$HOME/.config/uv/uv.toml"   ]] && return 0
    [[ -f "$HOME/.config/pip/pip.conf" ]] && return 0
    [[ -f "$HOME/.pip/pip.conf"        ]] && return 0
    [[ -f "$PROJECT_DIR/uv.toml"       ]] && return 0
    [[ -f "/etc/pip.conf"              ]] && return 0
    if [[ -f "$PROJECT_DIR/pyproject.toml" ]] && \
       grep -qE '^\[tool\.uv\]|^\[tool\.pip\]' "$PROJECT_DIR/pyproject.toml" 2>/dev/null; then
        return 0
    fi
    return 1
}

user_has_git_mirror() {
    git config --get-regexp '^url\..*\.insteadof$' 2>/dev/null \
        | grep -qi 'github\.com' && return 0
    return 1
}

# ==================================================================
# Mirror configuration
#
# Only prefix-proxy style mirrors are included here. These all work as:
#   {mirror_base}/{original_github_url}
# so mirror_url() can construct URLs uniformly.
#
# We ALWAYS keep our fast mirror list. If a user has configured their own
# mirror, we PREFER it (by prepending an empty entry that makes uv/git use
# the user config), but never replace our list entirely — otherwise a broken
# user mirror makes the whole install unusably slow.
# ==================================================================

PIP_MIRRORS=(
    "https://pypi.tuna.tsinghua.edu.cn/simple"
    "https://mirrors.aliyun.com/pypi/simple"
    "https://pypi.org/simple"
)

TORCH_MIRRORS=(
    "https://mirrors.aliyun.com/pypi/simple"
    "https://pypi.tuna.tsinghua.edu.cn/simple"
    "https://pypi.org/simple"
)

GITHUB_MIRRORS=(
    "https://ghfast.top"
    "https://ghproxy.net"
    ""
)

if user_has_git_mirror; then
    log_info "Detected existing git insteadOf rule for github."
    log_info "  Will try user's github config first, then built-in mirrors as fallback."
    GITHUB_MIRRORS=("" "https://ghfast.top" "https://ghproxy.net")
else
    log_info "No existing git insteadOf rule; using built-in github mirror list."
fi

if user_has_pypi_mirror; then
    log_info "Detected existing pip/uv mirror configuration — using it first."
    PIP_MIRRORS=("" "https://pypi.tuna.tsinghua.edu.cn/simple" "https://mirrors.aliyun.com/pypi/simple" "https://pypi.org/simple")
    TORCH_MIRRORS=("" "https://mirrors.aliyun.com/pypi/simple" "https://pypi.tuna.tsinghua.edu.cn/simple" "https://pypi.org/simple")
else
    log_info "No existing pip/uv mirror; using built-in mirror list."
fi

mirror_url() {
    local base="$1" path="$2"
    if [[ -z "$base" ]]; then
        echo "$path"
    else
        echo "${base}/${path}"
    fi
}

# ==================================================================
# uv bootstrap
# ==================================================================
add_common_uv_paths() {
    if [[ -d "$HOME/.local/bin" && ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
        export PATH="$HOME/.local/bin:$PATH"
    fi
}

ensure_uv_installed() {
    add_common_uv_paths
    command -v uv &>/dev/null && return 0
    log_info "uv not found. Installing ..."

    local installed=0
    if command -v curl &>/dev/null; then
        if curl -LsSf --retry 3 --connect-timeout 20 \
                https://astral.sh/uv/install.sh | sh; then
            installed=1
        fi
    fi
    if [[ "$installed" -eq 0 ]] && command -v wget &>/dev/null; then
        if wget -qO- --tries=3 --timeout=20 \
                https://astral.sh/uv/install.sh | sh; then
            installed=1
        fi
    fi
    if [[ "$installed" -eq 0 ]]; then
        log_warning "Astral installer failed; falling back to pip."
        for mirror in "${PIP_MIRRORS[@]}"; do
            local pargs=()
            [[ -n "$mirror" ]] && pargs+=(-i "$mirror")
            log_info "  pip install uv ${pargs[*]:-}"
            if pip3 install --user "${pargs[@]}" uv 2>/dev/null; then
                installed=1; break
            fi
        done
    fi
    add_common_uv_paths
    command -v uv &>/dev/null || log_error "Failed to install uv."
    log_info "uv installed: $(uv --version)"
}
ensure_uv_installed
log_info "Using uv: $(uv --version)"

# ==================================================================
# Deactivate conda
# ==================================================================
if command -v conda &>/dev/null; then
    conda deactivate 2>/dev/null || true
fi

# ==================================================================
# Git config (avoid HTTP/2 failures on large clones)
# ==================================================================
git config --global http.version HTTP/1.1 2>/dev/null || true
git config --global http.postBuffer 524288000 2>/dev/null || true
git config --global core.compression 0 2>/dev/null || true

# ==================================================================
# apt helper
# ==================================================================
install_packages() {
    local packages=("$@")
    [[ "$(id -u)" -eq 0 ]] || \
        log_error "Missing packages: ${packages[*]}. Install manually."
    command -v apt-get &>/dev/null || log_error "apt-get not found."
    log_info "Installing: ${packages[*]} ..."
    apt-get update -qq
    apt-get install -y "${packages[@]}"
    log_success "Packages installed."
}

# ==================================================================
# GCC 12
# ==================================================================
GCC12_PATH=$(command -v gcc-12 2>/dev/null || true)
GXX12_PATH=$(command -v g++-12 2>/dev/null || true)
if [[ -z "$GCC12_PATH" || -z "$GXX12_PATH" ]]; then
    log_warning "GCC 12 not found."
    install_packages gcc-12 g++-12
    GCC12_PATH=$(command -v gcc-12)
    GXX12_PATH=$(command -v g++-12)
fi
log_info "GCC 12: $GCC12_PATH"
log_info "G++  12: $GXX12_PATH"

# ==================================================================
# Python 3.10 detection
#
# Rules:
#   - Never accept a Python inside $VENV_DIR (we may delete that dir).
#   - Prefer a uv-managed interpreter, then system python3.10.
#   - Install via uv if needed; fall back to conda.
# ==================================================================
PYTHON310=""

is_inside_venv() {
    [[ -n "$1" && "$1" == "$VENV_DIR"* ]]
}

# 1. uv-managed python 3.10
if uv python find 3.10 &>/dev/null; then
    cand="$(uv python find 3.10 || true)"
    if [[ -n "$cand" ]] && ! is_inside_venv "$cand"; then
        PYTHON310="$cand"
        log_info "Found uv-managed Python 3.10: $PYTHON310"
    fi
fi

# 2. System python3.10 / python3 / python (skip any inside $VENV_DIR)
if [[ -z "$PYTHON310" ]]; then
    for cand_name in python3.10 python3 python; do
        cand_path="$(command -v "$cand_name" 2>/dev/null || true)"
        [[ -z "$cand_path" ]] && continue
        is_inside_venv "$cand_path" && continue
        if "$cand_path" -c 'import sys; sys.exit(0 if sys.version_info[:2]==(3,10) else 1)' 2>/dev/null; then
            PYTHON310="$cand_path"
            log_info "Found system Python 3.10: $PYTHON310"
            break
        fi
    done
fi

# 3. Try installing via uv
if [[ -z "$PYTHON310" ]]; then
    log_info "Attempting to install Python 3.10 via uv ..."
    if uv python install 3.10; then
        cand="$(uv python find 3.10 || true)"
        if [[ -n "$cand" ]] && ! is_inside_venv "$cand"; then
            PYTHON310="$cand"
            log_success "Python 3.10 installed via uv: $PYTHON310"
        fi
    fi
fi

# 4. Conda fallback
if [[ -z "$PYTHON310" ]] && command -v conda &>/dev/null; then
    log_info "Falling back to conda (env 'py310') ..."
    conda create -y -n py310 python=3.10
    eval "$(conda shell.bash hook)"
    conda activate py310
    PYTHON310="$(command -v python)"
    log_success "Conda env py310: $PYTHON310"
fi

[[ -n "$PYTHON310" ]] || log_error "Python 3.10 not available."
log_info "Using Python 3.10: $PYTHON310"

# ==================================================================
# Torch install helpers
# ==================================================================
verify_torch() {
    [[ -x "$VENV_DIR/bin/python" ]] || return 1
    "$VENV_DIR/bin/python" -c \
        'import os, torch, sys; p = os.path.join(os.path.dirname(torch.__file__), "include", "torch", "headeronly", "util", "Float8_e4m3fnuz.h"); sys.exit(0 if os.path.isfile(p) else 1)' \
        >/dev/null 2>&1
}

try_install_torch_from() {
    local index="$1" pin="$2"
    local pkgs
    if [[ -n "$pin" ]]; then
        pkgs="torch==$pin torchvision torchaudio"
    else
        pkgs="torch torchvision torchaudio"
    fi
    local args=()
    [[ -n "$index" ]] && args+=(--index-url "$index")
    log_info "  Trying: $pkgs ${args[*]:-}"
    # shellcheck disable=SC2086
    uv pip install --python "$VENV_DIR/bin/python" $pkgs "${args[@]}" >/dev/null 2>&1
}

install_torch_with_fallback() {
    # Pin to the exact torch version this tree targets (see pyproject.toml
    # build-system.requires and CMakeLists.txt TORCH_SUPPORTED_VERSION_CUDA).
    # Newer torch (2.14.x) requires C++20 while vLLM builds with C++17, so
    # "latest" is NOT safe. Fall back to "latest" only as a last resort.
    local pins=("2.11.0" "")
    local indexes
    if [[ ${#TORCH_MIRRORS[@]} -eq 0 ]]; then
        indexes=("")
    else
        indexes=("${TORCH_MIRRORS[@]}")
    fi
    for index in "${indexes[@]}"; do
        for pin in "${pins[@]}"; do
            if try_install_torch_from "$index" "$pin"; then
                if verify_torch; then
                    local ver
                    ver=$("$VENV_DIR/bin/python" -c 'import torch; print(torch.__version__)' 2>/dev/null || echo unknown)
                    log_success "Installed torch $ver from ${index:-user config}"
                    return 0
                fi
            fi
        done
    done
    log_error "Could not install a working torch with torch/headeronly."
}

# ==================================================================
# Create or reuse .venv
# ==================================================================
if [[ -d "$VENV_DIR" ]]; then
    if ! verify_torch 2>/dev/null; then
        log_warning "Existing .venv lacks working torch — recreating .venv"
        rm -rf "$VENV_DIR"
    else
        log_info "Reusing existing virtual environment at $VENV_DIR"
    fi
fi

if [[ ! -d "$VENV_DIR" ]]; then
    # Guarantee PYTHON310 is still valid (in case it was inside the deleted venv)
    if [[ ! -x "$PYTHON310" ]]; then
        log_warning "PYTHON310 '$PYTHON310' is no longer valid; re-detecting ..."
        PYTHON310=""
        for cand_name in python3.10 python3 python; do
            cand_path="$(command -v "$cand_name" 2>/dev/null || true)"
            [[ -z "$cand_path" ]] && continue
            [[ "$cand_path" == "$VENV_DIR"* ]] && continue
            if "$cand_path" -c 'import sys; sys.exit(0 if sys.version_info[:2]==(3,10) else 1)' 2>/dev/null; then
                PYTHON310="$cand_path"; break
            fi
        done
        if [[ -z "$PYTHON310" ]]; then
            PYTHON310="$(uv python find 3.10 2>/dev/null || true)"
        fi
        if [[ -z "$PYTHON310" ]] && command -v conda &>/dev/null; then
            conda create -y -n py310 python=3.10
            eval "$(conda shell.bash hook)"
            conda activate py310
            PYTHON310="$(command -v python)"
        fi
        [[ -n "$PYTHON310" ]] || log_error "Could not find a valid Python 3.10."
        log_info "Re-detected Python 3.10: $PYTHON310"
    fi

    log_info "Creating .venv with Python 3.10 ..."
    uv venv "$VENV_DIR" --python "$PYTHON310"

    log_info "Installing PyTorch ..."
    install_torch_with_fallback
fi

# ------------------------------------------------------------------
# Ensure build dependencies are present — ALWAYS, even on reuse.
# ------------------------------------------------------------------
log_info "Ensuring build dependencies are installed ..."
_deps_ok=0
for mirror in "${PIP_MIRRORS[@]}"; do
    _args=()
    [[ -n "$mirror" ]] && _args+=(-i "$mirror")
    log_info "  uv pip install ... ${_args[*]:-}"
    if uv pip install --python "$VENV_DIR/bin/python" \
            "${_args[@]}" \
            ninja wheel "setuptools>=77.0.3,<81.0.0" packaging setuptools_scm cmake numpy jinja2; then
        _deps_ok=1
        break
    fi
done
[[ "$_deps_ok" -eq 1 ]] || log_warning "Some build dependencies may not have installed."

PYTHON_VERSION=$("$VENV_DIR/bin/python" -c \
    'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
if [[ "$PYTHON_VERSION" != "3.10" ]]; then
    log_warning "Python is $PYTHON_VERSION (vLLM expects 3.10)"
else
    log_info "Python version: $PYTHON_VERSION (OK)"
fi
TORCH_VERSION=$("$VENV_DIR/bin/python" -c \
    'import torch; print(torch.__version__)' 2>/dev/null || echo unknown)
log_info "PyTorch version: $TORCH_VERSION"

# ==================================================================
# pythoncapi-compat
# ==================================================================
fetch_pythoncapi_compat() {
    if [[ -f "$PYTHONCAPI_COMPAT_DIR/include/pythoncapi_compat.h" ]]; then
        log_info "pythoncapi-compat already present"
        return 0
    fi
    log_info "Fetching pythoncapi-compat ..."

    local tar="/tmp/pythoncapi-compat.tar.gz"

    # Tarball first — git clone is unreliable on restricted egress links.
    for ref in "master" "main"; do
        for base in "${GITHUB_MIRRORS[@]}"; do
            local url
            url="$(mirror_url "$base" "https://github.com/python/pythoncapi-compat/archive/refs/heads/${ref}.tar.gz")"
            log_info "  tarball (ref $ref): $url"
            rm -f "$tar"
            if curl -L \
                    --retry 3 --retry-delay 3 \
                    --http1.1 \
                    --connect-timeout 15 \
                    --max-time 600 \
                    --speed-limit 1024 --speed-time 30 \
                    --progress-bar \
                    -o "$tar" "$url" 2>/dev/null; then
                if tar -tzf "$tar" >/dev/null 2>&1; then
                    rm -rf "$PYTHONCAPI_COMPAT_DIR"
                    mkdir -p "$PYTHONCAPI_COMPAT_DIR"
                    tar -xzf "$tar" -C "$PYTHONCAPI_COMPAT_DIR" --strip-components=1
                    rm -f "$tar"
                    if [[ -f "$PYTHONCAPI_COMPAT_DIR/include/pythoncapi_compat.h" ]]; then
                        log_success "pythoncapi-compat fetched from $url"
                        return 0
                    fi
                fi
            fi
            rm -f "$tar"
        done
    done

    # Git clone fallback (with explicit timeout so a dead mirror can't hang).
    for base in "${GITHUB_MIRRORS[@]}"; do
        local url
        url="$(mirror_url "$base" "https://github.com/python/pythoncapi-compat.git")"
        log_info "  git clone $url"
        rm -rf "$PYTHONCAPI_COMPAT_DIR"
        if timeout 120 git clone --depth 1 --quiet "$url" "$PYTHONCAPI_COMPAT_DIR" 2>/dev/null; then
            if [[ -f "$PYTHONCAPI_COMPAT_DIR/include/pythoncapi_compat.h" ]]; then
                log_success "pythoncapi-compat fetched from $url"
                return 0
            fi
        fi
        rm -rf "$PYTHONCAPI_COMPAT_DIR"
    done
    log_warning "Could not fetch pythoncapi-compat."
    return 1
}

fetch_pythoncapi_compat || true
if [[ -f "$PYTHONCAPI_COMPAT_DIR/include/pythoncapi_compat.h" ]]; then
    export CXXFLAGS="-I$PYTHONCAPI_COMPAT_DIR/include ${CXXFLAGS:-}"
    export CFLAGS="-I$PYTHONCAPI_COMPAT_DIR/include ${CFLAGS:-}"
    log_info "pythoncapi-compat: $PYTHONCAPI_COMPAT_DIR/include"
fi

# ==================================================================
# Compiler / CUDA env
# ==================================================================
export CC="$GCC12_PATH"
export CXX="$GXX12_PATH"
export CUDAHOSTCXX="$GXX12_PATH"

export CUDA_HOME=/usr/local/cuda
export PATH="$CUDA_HOME/bin:$PATH"
export CUDACXX="$CUDA_HOME/bin/nvcc"

# ==================================================================
# Auto-detect GPU SM
# ==================================================================
detect_sm() {
    command -v nvidia-smi &>/dev/null || { echo ""; return 0; }
    local sm
    sm="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
        | head -n1 | tr -d ' ' || true)"
    # Only accept a real compute capability like "8.9" / "10.0". When the
    # driver is unavailable, nvidia-smi can print its error to *stdout*, so
    # validate the output instead of trusting it blindly.
    if [[ "$sm" =~ ^[0-9]+\.[0-9]+$ ]]; then
        echo "$sm"
    fi
    return 0
}
# `detect_sm` must never abort the script when nvidia-smi is unavailable
# (e.g. no GPU, or driver permissions): it always returns 0, and an invalid
# reading yields empty output so the fallback arch is used.
DETECTED_SM="$(detect_sm || true)"
if [[ -n "$DETECTED_SM" ]]; then
    export TORCH_CUDA_ARCH_LIST="$DETECTED_SM"
    log_info "Detected GPU SM: $DETECTED_SM → TORCH_CUDA_ARCH_LIST=$DETECTED_SM"
else
    export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.9}"
    log_warning "Could not detect GPU; using $TORCH_CUDA_ARCH_LIST"
fi

# ==================================================================
# Optional external projects
#
# These are disabled by default. None of them are required by PSP-KV,
# and their network fetches routinely fail (DeepGEMM in particular
# recursively initialises git submodules that cannot be suppressed with
# FETCHCONTENT_UPDATES_DISCONNECTED=ON).
#
# Enable any of them by exporting the corresponding variable (e.g.
# VLLM_BUILD_DEEPGEMM=ON) before running this script.
# ==================================================================
export VLLM_BUILD_TRITON_KERNELS="${VLLM_BUILD_TRITON_KERNELS:-OFF}"
export VLLM_BUILD_DEEPGEMM="${VLLM_BUILD_DEEPGEMM:-OFF}"
export VLLM_BUILD_FLASHMLA="${VLLM_BUILD_FLASHMLA:-OFF}"
export VLLM_BUILD_QUTLASS="${VLLM_BUILD_QUTLASS:-OFF}"
log_info "VLLM_BUILD_TRITON_KERNELS=$VLLM_BUILD_TRITON_KERNELS"
log_info "VLLM_BUILD_DEEPGEMM=$VLLM_BUILD_DEEPGEMM"
log_info "VLLM_BUILD_FLASHMLA=$VLLM_BUILD_FLASHMLA"
log_info "VLLM_BUILD_QUTLASS=$VLLM_BUILD_QUTLASS"

# ==================================================================
# CUTLASS fetch: mirrors + tarball + git fallback
#
# CUTLASS is always required — vLLM's CUTLASS extensions are part of
# the core _C build.
# ==================================================================
fetch_cutlass() {
    local target="$DEPS_DIR/cutlass-src"
    [[ -f "$target/CMakeLists.txt" ]] && return 0

    log_warning "CUTLASS missing — attempting download ..."
    local tar="/tmp/cutlass-v4.4.2.tar.gz"

    # 1. Tarball from each mirror (fastest, retries cleanly)
    for base in "${GITHUB_MIRRORS[@]}"; do
        local url
        url="$(mirror_url "$base" "https://github.com/nvidia/cutlass/archive/refs/tags/v4.4.2.tar.gz")"
        log_info "  tarball: $url"

        # --continue-at - : resume if a partial file exists (the server may
        #   ignore it; the --retry loop then restarts cleanly).
        # A LOW --speed-limit (1 KB/s) detects a fully stalled transfer and
        # aborts it so we can try the next mirror, while still allowing slow
        # but steady links (tens of KB/s). Generous --max-time so a
        # slow-but-progressing transfer can finish.
        if curl -L \
                --retry 5 --retry-delay 3 \
                --http1.1 \
                --connect-timeout 15 \
                --max-time 3600 \
                --speed-limit 1024 --speed-time 30 \
                --continue-at - \
                --progress-bar \
                -o "$tar" "$url"; then
            if tar -tzf "$tar" >/dev/null 2>&1; then
                rm -rf "$target"
                mkdir -p "$target"
                tar -xzf "$tar" -C "$target" --strip-components=1
                rm -f "$tar"
                if [[ -f "$target/CMakeLists.txt" ]]; then
                    log_success "CUTLASS via tarball to $target"
                    return 0
                fi
            fi
        fi
        rm -f "$tar"
    done

    # 2. Last resort: git clone with mirrors
    for base in "${GITHUB_MIRRORS[@]}"; do
        local url
        url="$(mirror_url "$base" "https://github.com/nvidia/cutlass.git")"
        log_info "  git clone: $url"
        rm -rf "$target"
        if timeout 600 git clone --depth 1 --branch v4.4.2 --progress \
                "$url" "$target" 2>&1 | tail -n 5; then
            if [[ -f "$target/CMakeLists.txt" ]]; then
                log_success "CUTLASS via git to $target"
                return 0
            fi
        fi
    done

    log_error "Could not fetch CUTLASS from any mirror."
}

# ==================================================================
# vLLM Flash Attention fetch: mirrors + tarball + git fallback
#
# The upstream repo is https://github.com/vllm-project/flash-attention.git
# (the "vllm-flash-attn" project). It provides standard FA2/FA3 kernels and
# is required for the _vllm_fa2_C / _vllm_fa3_C extensions. Unlike the
# optional projects, this one is always required.
# ==================================================================
# Exact commits pinned by cmake/external_projects/vllm_flash_attn.cmake (GIT_TAG)
# and by that file's CUTLASS submodule. Using these instead of an unpinned
# "main" keeps the FA kernels ABI-compatible with the CUTLASS vLLM was tested
# against (an unpinned main pulls a newer CUTLASS API).
FA_PIN_COMMIT="f5bc33cfc02c744d24a2e9d50e6db656de40611c"
FA_CUTLASS_COMMIT="62750a2b75c802660e4894434dc55e839f322277"

fetch_vllm_flash_attn() {
    local target="$DEPS_DIR/vllm-flash-attn-src"
    [[ -f "$target/CMakeLists.txt" ]] && return 0

    log_warning "vLLM Flash Attention missing — attempting download ..."
    local tar="/tmp/vllm-flash-attn.tar.gz"

    # 1. Pinned commit tarball (preferred; matches the CMake GIT_TAG).
    for base in "${GITHUB_MIRRORS[@]}"; do
        local url
        url="$(mirror_url "$base" "https://github.com/vllm-project/flash-attention/archive/${FA_PIN_COMMIT}.tar.gz")"
        log_info "  tarball (pinned commit): $url"
        rm -f "$tar"
        if curl -L \
                --retry 5 --retry-delay 3 \
                --http1.1 \
                --connect-timeout 15 \
                --max-time 3600 \
                --speed-limit 1024 --speed-time 30 \
                --continue-at - \
                --progress-bar \
                -o "$tar" "$url" 2>/dev/null; then
            if tar -tzf "$tar" >/dev/null 2>&1; then
                rm -rf "$target"
                mkdir -p "$target"
                tar -xzf "$tar" -C "$target" --strip-components=1
                rm -f "$tar"
                if [[ -f "$target/CMakeLists.txt" ]]; then
                    log_success "vLLM Flash Attention (pinned commit) via tarball to $target"
                    return 0
                fi
            fi
        fi
        rm -f "$tar"
    done

    # 2. Fall back to a few common branch/tag names.
    for ref in "main" "master" "v2.7.4.post1" "v2.7.3" "v2.7.2.post1"; do
        for base in "${GITHUB_MIRRORS[@]}"; do
            local url
            url="$(mirror_url "$base" "https://github.com/vllm-project/flash-attention/archive/refs/heads/${ref}.tar.gz")"
            log_info "  tarball (ref $ref): $url"
            rm -f "$tar"
            if curl -L \
                    --retry 5 --retry-delay 3 \
                    --http1.1 \
                    --connect-timeout 15 \
                    --max-time 3600 \
                    --speed-limit 1024 --speed-time 30 \
                    --continue-at - \
                    --progress-bar \
                    -o "$tar" "$url" 2>/dev/null; then
                if tar -tzf "$tar" >/dev/null 2>&1; then
                    rm -rf "$target"
                    mkdir -p "$target"
                    tar -xzf "$tar" -C "$target" --strip-components=1
                    rm -f "$tar"
                    if [[ -f "$target/CMakeLists.txt" ]]; then
                        log_success "vLLM Flash Attention (ref $ref) via tarball to $target"
                        return 0
                    fi
                fi
            fi
            rm -f "$tar"
        done
    done

    # 2. Last resort: git clone with mirrors (no branch pin → default branch).
    for base in "${GITHUB_MIRRORS[@]}"; do
        local url
        url="$(mirror_url "$base" "https://github.com/vllm-project/flash-attention.git")"
        log_info "  git clone: $url"
        rm -rf "$target"
        if timeout 600 git clone --depth 1 --progress \
                "$url" "$target" 2>&1 | tail -n 5; then
            if [[ -f "$target/CMakeLists.txt" ]]; then
                log_success "vLLM Flash Attention via git to $target"
                return 0
            fi
        fi
    done

    log_error "Could not fetch vLLM Flash Attention from any mirror."
}

mkdir -p "$DEPS_DIR"
fetch_cutlass
fetch_vllm_flash_attn

# ------------------------------------------------------------------
# Populate vllm-flash-attn's CUTLASS git submodule.
#
# vllm-flash-attn keeps CUTLASS as a git submodule at csrc/cutlass. Tarball
# fetches do not include submodule contents, so csrc/cutlass ends up empty and
# its flash_api.cpp fails with `cutlass/numeric_types.h: No such file or
# directory`. Fetch the exact CUTLASS revision that flash-attn pins as its
# submodule and link its include/ tree into place. (The ROCm submodules are
# unused for a CUDA build.)
# ------------------------------------------------------------------
fetch_flash_attn_cutlass() {
    local fa_cutlass="$DEPS_DIR/vllm-flash-attn-src/csrc/cutlass"
    [[ -d "$fa_cutlass" ]] || return 1
    [[ -e "$fa_cutlass/include" ]] && return 0

    local src="$DEPS_DIR/cutlass-fa-src"
    if [[ ! -f "$src/CMakeLists.txt" ]]; then
        log_info "Fetching vllm-flash-attn's pinned CUTLASS ..."
        local tar="/tmp/cutlass-fa.tar.gz"
        for base in "${GITHUB_MIRRORS[@]}"; do
            local url
            url="$(mirror_url "$base" "https://github.com/NVIDIA/cutlass/archive/${FA_CUTLASS_COMMIT}.tar.gz")"
            log_info "  tarball: $url"
            rm -f "$tar"
            if curl -L \
                    --retry 5 --retry-delay 3 \
                    --http1.1 \
                    --connect-timeout 15 \
                    --max-time 3600 \
                    --speed-limit 1024 --speed-time 30 \
                    --continue-at - \
                    --progress-bar \
                    -o "$tar" "$url" 2>/dev/null; then
                if tar -tzf "$tar" >/dev/null 2>&1; then
                    rm -rf "$src"
                    mkdir -p "$src"
                    tar -xzf "$tar" -C "$src" --strip-components=1
                    rm -f "$tar"
                    [[ -f "$src/CMakeLists.txt" ]] && break
                fi
            fi
            rm -f "$tar"
        done
    fi

    if [[ -d "$src/include" ]]; then
        ln -s "$src/include" "$fa_cutlass/include"
        log_info "Linked pinned CUTLASS into vllm-flash-attn (csrc/cutlass/include)."
        return 0
    fi
    log_warning "Could not fetch vllm-flash-attn's pinned CUTLASS."
    return 1
}
fetch_flash_attn_cutlass || true

# ==================================================================
# Virthub (KV-cache storage management) — installed by default (best-effort)
#
# Virthub provides the multi-tier KV-cache store (VRAM/DRAM/NVMe), the
# allocation-time precision predictor, and the vLLM KV connector. Installation is
# delegated to Virthub's own `install.sh` (which provisions the Rust toolchain,
# system deps, its own virtualenv, and the Python bindings). The PSP-KV CUDA
# kernels are already vendored under csrc/attention/psp_kv and do NOT depend on
# Virthub.
# ==================================================================
VIRTHUB_COMMIT="4c85bed03cb59336752f77f9569ceb52e3d3bcdf"
VIRTHUB_DIR="$PROJECT_DIR/third_party/virthub"

fetch_virthub() {
    local target="$VIRTHUB_DIR"
    [[ -f "$target/bindings/python/pyproject.toml" ]] && return 0

    log_info "Fetching Virthub ..."
    local tar="/tmp/virthub.tar.gz"
    mkdir -p "$(dirname "$target")"

    for base in "${GITHUB_MIRRORS[@]}"; do
        local url
        url="$(mirror_url "$base" "https://github.com/virthub/virthub/archive/${VIRTHUB_COMMIT}.tar.gz")"
        log_info "  tarball: $url"
        rm -f "$tar"
        if curl -L \
                --retry 5 --retry-delay 3 \
                --http1.1 \
                --connect-timeout 15 \
                --max-time 3600 \
                --speed-limit 1024 --speed-time 30 \
                --continue-at - \
                --progress-bar \
                -o "$tar" "$url" 2>/dev/null; then
            if tar -tzf "$tar" >/dev/null 2>&1; then
                rm -rf "$target"
                mkdir -p "$target"
                tar -xzf "$tar" -C "$target" --strip-components=1
                rm -f "$tar"
                if [[ -f "$target/bindings/python/pyproject.toml" ]]; then
                    log_success "Virthub fetched to $target"
                    return 0
                fi
            fi
        fi
        rm -f "$tar"
    done

    log_warning "Could not fetch Virthub from any mirror."
    return 1
}

if [[ "$VLLM_BUILD_VIRTHUB" == "ON" ]]; then
    # Virthub is installed by default, but its own installer provisions a Rust
    # toolchain + system packages (sudo) and creates its own virtualenv, which
    # can fail in constrained environments. Treat it as best-effort so a
    # Virthub failure never blocks the core vLLM + PSP-KV build.
    if fetch_virthub; then
        # Delegate the actual installation to Virthub's own installer.
        # --no-patch-vllm because this fork already registers the connector in
        # KVConnectorFactory at the source level.
        log_info "Running Virthub's own installer ..."
        if bash "$VIRTHUB_DIR/install.sh" --no-patch-vllm; then
            log_success "Virthub installed via its own install.sh."
        else
            log_warning "Virthub installer failed; continuing without Virthub."
        fi
    else
        log_warning "Skipping Virthub (fetch failed); continuing without it."
    fi
fi

# ==================================================================
# Build CMAKE_ARGS
# ==================================================================
EXTRA_CMAKE_ARGS=""

# Optional external projects (must match the CMake options).
EXTRA_CMAKE_ARGS+=" -DVLLM_BUILD_TRITON_KERNELS=$VLLM_BUILD_TRITON_KERNELS"
EXTRA_CMAKE_ARGS+=" -DVLLM_BUILD_DEEPGEMM=$VLLM_BUILD_DEEPGEMM"
EXTRA_CMAKE_ARGS+=" -DVLLM_BUILD_FLASHMLA=$VLLM_BUILD_FLASHMLA"
EXTRA_CMAKE_ARGS+=" -DVLLM_BUILD_QUTLASS=$VLLM_BUILD_QUTLASS"

# CUTLASS (always required)
if [[ -f "$DEPS_DIR/cutlass-src/CMakeLists.txt" ]]; then
    export VLLM_CUTLASS_SRC_DIR="$DEPS_DIR/cutlass-src"
    export FETCHCONTENT_SOURCE_DIR_CUTLASS="$DEPS_DIR/cutlass-src"
    EXTRA_CMAKE_ARGS+=" -DVLLM_CUTLASS_SRC_DIR=$DEPS_DIR/cutlass-src"
    EXTRA_CMAKE_ARGS+=" -DFETCHCONTENT_SOURCE_DIR_CUTLASS=$DEPS_DIR/cutlass-src"
    log_info "Using local CUTLASS: $VLLM_CUTLASS_SRC_DIR"
fi

# vLLM Flash Attention (always required — provides standard attention).
if [[ -f "$DEPS_DIR/vllm-flash-attn-src/CMakeLists.txt" ]]; then
    export FETCHCONTENT_SOURCE_DIR_VLLM_FLASH_ATTN="$DEPS_DIR/vllm-flash-attn-src"
    EXTRA_CMAKE_ARGS+=" -DFETCHCONTENT_SOURCE_DIR_VLLM_FLASH_ATTN=$DEPS_DIR/vllm-flash-attn-src"
    log_info "Using local vLLM Flash Attention: $DEPS_DIR/vllm-flash-attn-src"
else
    log_error "vLLM Flash Attention missing at $DEPS_DIR/vllm-flash-attn-src. Aborting."
fi

# Optional dependencies — only exported if their VLLM_BUILD_* flag is ON
# AND their source tree exists locally.
if [[ "$VLLM_BUILD_DEEPGEMM" == "ON" ]]; then
    dir="$DEPS_DIR/deepgemm-src"
    if [[ -f "$dir/CMakeLists.txt" ]]; then
        export FETCHCONTENT_SOURCE_DIR_DEEPGEMM="$dir"
        EXTRA_CMAKE_ARGS+=" -DFETCHCONTENT_SOURCE_DIR_DEEPGEMM=$dir"
    fi
fi

if [[ "$VLLM_BUILD_FLASHMLA" == "ON" ]]; then
    dir="$DEPS_DIR/flashmla-src"
    if [[ -f "$dir/CMakeLists.txt" ]]; then
        export FETCHCONTENT_SOURCE_DIR_FLASHMLA="$dir"
        EXTRA_CMAKE_ARGS+=" -DFETCHCONTENT_SOURCE_DIR_FLASHMLA=$dir"
    fi
fi

if [[ "$VLLM_BUILD_QUTLASS" == "ON" ]]; then
    dir="$DEPS_DIR/qutlass-src"
    if [[ -f "$dir/CMakeLists.txt" ]]; then
        export FETCHCONTENT_SOURCE_DIR_QUTLASS="$dir"
        EXTRA_CMAKE_ARGS+=" -DFETCHCONTENT_SOURCE_DIR_QUTLASS=$dir"
    fi
fi

if [[ "$VLLM_BUILD_TRITON_KERNELS" == "ON" ]] && \
   [[ -f "/root/triton_kernels-src/CMakeLists.txt" ]]; then
    export FETCHCONTENT_SOURCE_DIR_TRITON_KERNELS="/root/triton_kernels-src"
    EXTRA_CMAKE_ARGS+=" -DFETCHCONTENT_SOURCE_DIR_TRITON_KERNELS=/root/triton_kernels-src"
fi

EXTRA_CMAKE_ARGS+=" -DFETCHCONTENT_UPDATES_DISCONNECTED=ON"

export CMAKE_ARGS="-DCMAKE_CUDA_HOST_COMPILER=$GXX12_PATH ${EXTRA_CMAKE_ARGS}"
log_info "CMAKE_ARGS=$CMAKE_ARGS"

# ==================================================================
# Final sanity checks
# ==================================================================
[[ -f "$DEPS_DIR/cutlass-src/CMakeLists.txt" ]] || \
    log_error "CUTLASS incomplete at $DEPS_DIR/cutlass-src."
[[ -f "$DEPS_DIR/vllm-flash-attn-src/CMakeLists.txt" ]] || \
    log_error "vLLM Flash Attention incomplete at $DEPS_DIR/vllm-flash-attn-src."

# ==================================================================
# Editable install
# ==================================================================
log_info "Running: uv pip install -e . --no-build-isolation -v"
uv pip install --python "$VENV_DIR/bin/python" \
    -e . --no-build-isolation -v

# ==================================================================
# Final success message
# ==================================================================
log_success "Build completed successfully!"
log_info "You can now run vLLM with: ./run.sh --serve"
