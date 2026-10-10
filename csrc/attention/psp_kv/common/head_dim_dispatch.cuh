// Runtime head_dim -> compile-time constant dispatch.
//
// The PSP-KV kernels use stack arrays sized by HEAD_DIM (float query[HEAD_DIM],
// float acc[HEAD_DIM]) and unrolled loops, so HEAD_DIM must be a compile-time
// constant. Launch functions dispatch the runtime head_dim to the nearest
// supported compile-time instantiation via this helper.

#pragma once

#include <stdexcept>
#include <string>
#include <type_traits>

// Supported head sizes. Keep in sync with the PSP-KV backend's declared
// get_supported_head_sizes().
template <typename F>
inline void psp_kv_dispatch_head_dim(int head_dim, F&& fn) {
    switch (head_dim) {
        case 32:  return fn(std::integral_constant<int, 32>{});
        case 64:  return fn(std::integral_constant<int, 64>{});
        case 80:  return fn(std::integral_constant<int, 80>{});
        case 96:  return fn(std::integral_constant<int, 96>{});
        case 128: return fn(std::integral_constant<int, 128>{});
        case 160: return fn(std::integral_constant<int, 160>{});
        case 192: return fn(std::integral_constant<int, 192>{});
        case 256: return fn(std::integral_constant<int, 256>{});
        default:
            throw std::invalid_argument(
                "PSP-KV: unsupported head_dim " + std::to_string(head_dim));
    }
}
