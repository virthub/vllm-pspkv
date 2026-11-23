# Standalone build for the PSP-KV extension (`vllm/_C.abi3.so`).
#
# The rest of vLLM's compiled extensions come from the matching upstream wheel
# (see install.sh); this project builds only the PSP-KV kernels, which are not
# part of upstream vLLM. It is a small, self-contained project so the extension
# can be rebuilt in seconds instead of re-running vLLM's full CMake build.
#
# Usage (from the repository root):
#   cmake -S cmake/psp_kv -B build_psp/out -G Ninja \
#     -DPython_EXECUTABLE="$(which python)" \
#     -DCMAKE_PREFIX_PATH="$(python -c 'import torch,os;print(os.path.dirname(torch.__file__) + "/share/cmake")')"
#   cmake --build build_psp/out --target _C
#
# Knobs:
#   -DPSPKV_CUDA_ARCH=120a         CUDA arch for the PSP kernels
#   -DPSPKV_SCALE_GROUP_SIZE=16    16 (default) or 32 scale granularity
#   -DPSPKV_BLOCK_SIZE=128         tokens per kernel block (must match --block-size)
#   -DPSPKV_TOKEN_MAJOR=1          1 = token-major pages, 0 = head-major legacy

cmake_minimum_required(VERSION 3.26)
project(psp_kv LANGUAGES CXX CUDA)

get_filename_component(REPO "${CMAKE_CURRENT_LIST_DIR}/../.." ABSOLUTE)

find_package(Python COMPONENTS Interpreter Development.Module Development.SABIModule REQUIRED)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD 20)

# Resolve the target architecture *before* Torch is found: TorchConfig seeds
# CMAKE_CUDA_ARCHITECTURES from TORCH_CUDA_ARCH_LIST (which defaults to a long
# list including compute_50, where the wmma headers this project relies on are
# unavailable). Forcing the cache entry first keeps a single-arch build.
if(NOT DEFINED PSPKV_CUDA_ARCH OR "$ENV{PSPKV_CUDA_ARCH}" STREQUAL "")
  if(DEFINED ENV{PSPKV_CUDA_ARCH} AND NOT "$ENV{PSPKV_CUDA_ARCH}" STREQUAL "")
    set(PSPKV_CUDA_ARCH "$ENV{PSPKV_CUDA_ARCH}")
  elseif(NOT DEFINED PSPKV_CUDA_ARCH)
    set(PSPKV_CUDA_ARCH "120a")
  endif()
endif()
set(CMAKE_CUDA_ARCHITECTURES "${PSPKV_CUDA_ARCH}" CACHE STRING
    "CUDA architectures for the PSP-KV kernels" FORCE)

# TorchConfig derives and injects one -gencode flag per entry of
# TORCH_CUDA_ARCH_LIST, whose default spans compute_50..compute_120. compile for
# compute_50 fails here because the wmma headers need sm_70+, so pin the list to
# the single architecture this project targets.
if(PSPKV_CUDA_ARCH MATCHES "^([0-9]+)([a-z]?)$")
  set(_arch_num "${CMAKE_MATCH_1}")
  set(_arch_suffix "${CMAKE_MATCH_2}")
  string(LENGTH "${_arch_num}" _arch_len)
  if(_arch_len EQUAL 2)
    string(SUBSTRING "${_arch_num}" 0 1 _arch_major)
    string(SUBSTRING "${_arch_num}" 1 1 _arch_minor)
  elseif(_arch_len EQUAL 3)
    string(SUBSTRING "${_arch_num}" 0 2 _arch_major)
    string(SUBSTRING "${_arch_num}" 2 1 _arch_minor)
  else()
    set(_arch_major "${_arch_num}")
    set(_arch_minor "0")
  endif()
  # Strip a leading zero from the minor version (9.0a, 12.0a, ...).
  string(REGEX REPLACE "^0+([0-9])$" "\\1" _arch_minor "${_arch_minor}")
  set(ENV{TORCH_CUDA_ARCH_LIST} "${_arch_major}.${_arch_minor}${_arch_suffix}")
  message(STATUS "TORCH_CUDA_ARCH_LIST=${_arch_major}.${_arch_minor}${_arch_suffix}")
endif()

find_package(Torch REQUIRED)
set(CMAKE_CUDA_ARCHITECTURES "${PSPKV_CUDA_ARCH}")

if(NOT DEFINED PSPKV_SCALE_GROUP_SIZE)
  if(DEFINED ENV{PSPKV_SCALE_GROUP} AND NOT "$ENV{PSPKV_SCALE_GROUP}" STREQUAL "")
    set(PSPKV_SCALE_GROUP_SIZE "$ENV{PSPKV_SCALE_GROUP}")
  else()
    set(PSPKV_SCALE_GROUP_SIZE 16)
  endif()
endif()

if(NOT DEFINED PSPKV_BLOCK_SIZE)
  set(PSPKV_BLOCK_SIZE 128)
endif()

if(NOT DEFINED PSPKV_TOKEN_MAJOR)
  set(PSPKV_TOKEN_MAJOR 1)
endif()

set(SRCS
  ${REPO}/csrc/attention/psp_kv_bindings.cpp
  ${REPO}/csrc/attention/psp_kv_wrapper.cu
  ${REPO}/csrc/attention/psp_kv/basic/paged_attention_basic.cu
  ${REPO}/csrc/attention/psp_kv/basic/dequant_kernel.cu
  ${REPO}/csrc/attention/psp_kv/basic/uniform_attention_kernel.cu
  ${REPO}/csrc/attention/psp_kv/enhanced/paged_attention_enhanced.cu
  ${REPO}/csrc/attention/psp_kv/enhanced/dequant_kernel.cu
  ${REPO}/csrc/attention/psp_kv/enhanced/int8_k32.cu
  ${REPO}/csrc/attention/psp_kv/btkv/paged_attention_btkv.cu
  ${REPO}/csrc/attention/psp_kv/btkv/tma_descriptors.cu
  ${REPO}/csrc/attention/psp_kv/common/quantize_and_scatter_kernel.cu)

Python_add_library(_C MODULE USE_SABI 3 WITH_SOABI ${SRCS})
target_compile_definitions(_C PRIVATE
  PSP_KV_BLOCK_SIZE=${PSPKV_BLOCK_SIZE}
  PSP_KV_TOKEN_MAJOR=${PSPKV_TOKEN_MAJOR}
  TORCH_EXTENSION_NAME=_C
  PSPKV_SCALE_GROUP_SIZE=${PSPKV_SCALE_GROUP_SIZE}
  _GLIBCXX_USE_CXX11_ABI=1)
target_include_directories(_C PRIVATE
  ${REPO}/csrc ${REPO}/csrc/attention/psp_kv ${TORCH_INCLUDE_DIRS})
target_link_libraries(_C PRIVATE ${TORCH_LIBRARIES} CUDA::cudart CUDA::cuda_driver)
set_target_properties(_C PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${REPO}/vllm
  CUDA_ARCHITECTURES "${PSPKV_CUDA_ARCH}")
