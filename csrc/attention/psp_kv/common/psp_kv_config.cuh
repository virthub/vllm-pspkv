#pragma once
// Shared PSP-KV configuration constants.
//
// BLOCK_SIZE: number of tokens per KV-cache block. Must be a power of two.
//   16  = default: fine-grained per-block adaptive precision.
//   128 = FA3-aligned kBlockN (for head_dim ~96); coarser per-block granularity.
// Override at compile time with -DPSP_KV_BLOCK_SIZE=<n>.
#ifndef PSP_KV_BLOCK_SIZE
#define PSP_KV_BLOCK_SIZE 16
#endif
#define BLOCK_SIZE PSP_KV_BLOCK_SIZE

// PSP_KV_TOKEN_MAJOR: within a block, store K/V as [token][head][dim]
// (token-major, matching FA3's paged layout) instead of the default
// [head][token][dim] (head-major). Override with -DPSP_KV_TOKEN_MAJOR=1.
#ifndef PSP_KV_TOKEN_MAJOR
#define PSP_KV_TOKEN_MAJOR 0
#endif
