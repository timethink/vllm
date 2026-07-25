// Copyright (c) 2026, ByteV2 contributors.
//
// This file is consumed by the vLLM FlashAttention-2 extension.  It keeps the
// ByteV2-specific global-memory decode separate from the FA2 attention
// mainloop so the latter can retain its original MMA, softmax, split, and
// combine implementation.

#pragma once

#include <cstdint>

#include "byte_v2_layout.cuh"

namespace vllm::byte_v2::fa2 {

using Layout = ByteV2PageLayoutV5<>;
using Policy = Layout::TilePolicy;
using RawLayout = ByteV2RawStagingLayout<Policy, Layout::NumKvHeadsValue>;

static_assert(Policy::CodecTokenBlock == 16);
static_assert(Policy::CodecDimBlock == 16);
static_assert(Policy::AllocBlockTokens == 16);
static_assert(Policy::HeadDim == 128);
static_assert(Policy::HeadDimV == 128);
static_assert(Layout::NumKvHeadsValue == 8);
static_assert(Layout::CodecExponentCodeBits == 4);
static_assert(!Layout::IncludeRawPayloadValue);
static_assert(Layout::OutlierEntriesPerTileValue <= 0x1ff);
static_assert(Layout::OutlierPoolEntriesValue <= 0x7ff);
static_assert(RawLayout::SlotSizeBytes == 65536);
static_assert(RawLayout::ValueBaseBytes == 32768);

#ifndef VLLM_BYTE_V2_FA2_STAGE_MODE
  #define VLLM_BYTE_V2_FA2_STAGE_MODE 2
#endif

#ifndef VLLM_BYTE_V2_FA2_SIDEBAND_PREFETCH_MODE
  #define VLLM_BYTE_V2_FA2_SIDEBAND_PREFETCH_MODE 2
#endif

#ifndef VLLM_BYTE_V2_FA2_REUSE_KV_SMEM
  #define VLLM_BYTE_V2_FA2_REUSE_KV_SMEM 1
#endif

// 0 keeps the per-8-element-vector staging baseline. 1 cooperatively remaps
// each thread to one 8-row by 16-dimension quadrant and uses 16B low-plane
// plus 8B code-plane copies. Mode 2 combines two adjacent code rows into one
// 16B copy and reaches the compressed byte-count lower bound.
static constexpr int kStageMode = VLLM_BYTE_V2_FA2_STAGE_MODE;
static_assert(kStageMode >= 0 && kStageMode <= 2);

// 0 preserves the post-copy sideband loads. Mode 1 starts the page-global
// overflow-marker load before payload staging; mode 2 also starts the
// per-head fallback-mask load, and mode 3 starts the outlier-mask load. The
// values remain authoritative: they are consumed after all payload copies are
// issued, with the same fail-closed checks.
static constexpr int kSidebandPrefetchMode =
    VLLM_BYTE_V2_FA2_SIDEBAND_PREFETCH_MODE;
static_assert(kSidebandPrefetchMode >= 0 && kSidebandPrefetchMode <= 3);

// FA2 can optionally reuse one 32 KiB KV tile for both K and V. The mainloop
// then serializes K -> QK -> V -> PV -> next K, retaining N128 and the original
// floating-point operation order while reducing dynamic shared memory from
// 80 KiB to 48 KiB. The base loader enables this for split-K only; an explicit
// derived loader can opt the sufficiently large nonsplit grid into the same
// schedule.
static constexpr bool kReuseKvSmem = VLLM_BYTE_V2_FA2_REUSE_KV_SMEM != 0;

__device__ __forceinline__ uint32_t load_u32(const uint8_t* page, int offset) {
  return *reinterpret_cast<const uint32_t*>(page + offset);
}

// Keep this as an explicit demand load so the generated SASS can be checked
// for issue before the independent cp.async payload copies. This is a
// scheduling optimization for immutable committed pages, not a synchronization
// primitive for concurrent cache writers.
__device__ __forceinline__ uint32_t load_u32_early(const uint8_t* page,
                                                   int offset) {
  uint32_t value;
  const void* address = page + offset;
  asm volatile("ld.global.u32 %0, [%1];\n"
               : "=r"(value)
               : "l"(address)
               : "memory");
  return value;
}

__device__ __forceinline__ uint64_t load_u64(const uint8_t* page, int offset) {
  return *reinterpret_cast<const uint64_t*>(page + offset);
}

__device__ __forceinline__ uint16_t load_u16(const uint8_t* page, int offset) {
  return *reinterpret_cast<const uint16_t*>(page + offset);
}

__device__ __forceinline__ uint32_t decode_high_bytes(uint32_t byte_codes,
                                                      uint32_t base_bytes) {
  const uint32_t deltas = byte_codes & 0x07070707u;
  const uint32_t signs = (byte_codes & 0x08080808u) << 4;
  // The encoder restricts base to [0, 120], so adding a delta in [0, 7]
  // cannot carry between packed bytes.
  return signs | (base_bytes + deltas);
}

__device__ __forceinline__ uint4 decode_bf16x8(uint64_t lows, uint32_t codes,
                                               uint8_t base) {
  constexpr uint32_t kPackedNibbleMask = 0x0f0f0f0fu;
  constexpr uint32_t kInterleaveLow = 0x5140u;
  constexpr uint32_t kInterleaveHigh = 0x7362u;

  const uint32_t even_codes = codes & kPackedNibbleMask;
  const uint32_t odd_codes = (codes >> 4) & kPackedNibbleMask;
  const uint32_t base_bytes = static_cast<uint32_t>(base) * 0x01010101u;
  const uint32_t even_highs = decode_high_bytes(even_codes, base_bytes);
  const uint32_t odd_highs = decode_high_bytes(odd_codes, base_bytes);
  const uint32_t highs_0123 =
      __byte_perm(even_highs, odd_highs, kInterleaveLow);
  const uint32_t highs_4567 =
      __byte_perm(even_highs, odd_highs, kInterleaveHigh);
  const uint32_t lows_0123 = static_cast<uint32_t>(lows);
  const uint32_t lows_4567 = static_cast<uint32_t>(lows >> 32);
  return {__byte_perm(lows_0123, highs_0123, kInterleaveLow),
          __byte_perm(lows_0123, highs_0123, kInterleaveHigh),
          __byte_perm(lows_4567, highs_4567, kInterleaveLow),
          __byte_perm(lows_4567, highs_4567, kInterleaveHigh)};
}

// The persistent raw sidecar uses the existing ByteV2 staging layout:
// [K/V, kv_head, row, dim].  Each FA2 copy thread owns sixteen aligned BF16
// values for eight rows, so it can populate the final shared-memory tile with
// the same 16-byte transactions as raw paged FA2.  Returning true tells the
// post-copy decoder that this thread's page is already materialized BF16.
template <typename DstTensor>
__device__ __forceinline__ bool stage_raw_page_to_fa2_smem(
    const uint8_t* raw_side_base, int num_raw_slots,
    const int* page_to_raw_slot, int physical_page, int kv_head, int row0,
    int valid_rows, DstTensor& dst) {
  if (page_to_raw_slot == nullptr) {
    return false;
  }
  const int raw_slot = page_to_raw_slot[physical_page];
  if (raw_slot == -1) {
    return false;
  }
  if (raw_side_base == nullptr || raw_slot < 0 || raw_slot >= num_raw_slots ||
      reinterpret_cast<uintptr_t>(raw_side_base) % alignof(uint4) != 0) {
    __trap();
  }

  constexpr int kElementsPerVector = 8;
  constexpr int kRowsPerThread = 8;
  constexpr int kDimVectorsPerThread = 2;
  const int tidx = static_cast<int>(threadIdx.x);
  const int row_in_page0 = kRowsPerThread * ((tidx >> 3) & 1);
  const int dim_in_tile0 = kElementsPerVector * (tidx & 1);
  const uint8_t* raw_slot_base =
      raw_side_base + static_cast<int64_t>(raw_slot) * RawLayout::SlotSizeBytes;

#pragma unroll
  for (int k = 0; k < kDimVectorsPerThread; ++k) {
    const int dim_tile = 4 * k + ((tidx & 7) >> 1);
    const int dim0 = dim_tile * Policy::CodecDimBlock + dim_in_tile0;
#pragma unroll 1
    for (int m = 0; m < kRowsPerThread; ++m) {
      const int row = row_in_page0 + m;
      const bool valid = row0 + m < valid_rows;
      const int raw_offset =
          ((kv_head * Policy::AllocBlockTokens + row) * Policy::HeadDim +
           dim0) *
          RawLayout::RawElementBytesValue;
      cute::SM80_CP_ASYNC_CACHEALWAYS_ZFILL<uint4>::copy(
          *reinterpret_cast<const uint4*>(raw_slot_base + raw_offset),
          *reinterpret_cast<uint4*>(&dst(0, m, k)), valid);
    }
  }
  return true;
}

__device__ __forceinline__ bool thread_page_is_raw(int num_pages,
                                                   int num_raw_slots,
                                                   const int* page_to_raw_slot,
                                                   const int* block_table,
                                                   int n_block, int block_n) {
  if (page_to_raw_slot == nullptr) {
    return false;
  }
  const int logical_page =
      n_block * (block_n / Policy::AllocBlockTokens) +
      static_cast<int>(threadIdx.x) / Policy::CodecDimBlock;
  const int physical_page = block_table[logical_page];
  if (physical_page < 0 || physical_page >= num_pages) {
    __trap();
  }
  const int raw_slot = page_to_raw_slot[physical_page];
  if (raw_slot < -1 || raw_slot >= num_raw_slots) {
    __trap();
  }
  return raw_slot >= 0;
}

template <bool IsValue, typename DstTensor, typename CoordTensor>
__device__ __forceinline__ void stage_tile_to_fa2_smem_scalar(
    const uint8_t* byte_v2_cache, const uint8_t* raw_side_base,
    const int* page_to_raw_slot, int64_t page_stride_bytes, int num_pages,
    int num_raw_slots, const int* block_table, int kv_head, int n_block,
    int block_n, int valid_rows, DstTensor& dst, const CoordTensor& coords) {
  constexpr int kElementsPerVector = 8;
  constexpr int kRowsPerThread = 8;
  constexpr int kDimVectorsPerThread = 2;

  // This mapping is the exact sm80 FA2 BF16/D128/M64/N128 4-warp
  // GmemTiledCopyKV layout.  For thread tidx = 8 * group + column:
  //   dst(v, m, k) = (8 * group + m, 8 * column + v + 64 * k).
  // Therefore each (m, k) slice is one aligned 16-byte shared-memory store,
  // and every thread's eight rows reside in a single 16-token ByteV2 page.
  CUTE_STATIC_ASSERT_V(cute::size<0>(dst) == cute::Int<8>{});
  CUTE_STATIC_ASSERT_V(cute::size<1>(dst) == cute::Int<8>{});
  CUTE_STATIC_ASSERT_V(cute::size<2>(dst) == cute::Int<2>{});
  CUTE_STATIC_ASSERT_V(cute::size<0>(coords) == cute::Int<8>{});
  CUTE_STATIC_ASSERT_V(cute::size<1>(coords) == cute::Int<8>{});
  CUTE_STATIC_ASSERT_V(cute::size<2>(coords) == cute::Int<2>{});
  if (byte_v2_cache == nullptr || block_table == nullptr ||
      reinterpret_cast<uintptr_t>(byte_v2_cache) % alignof(uint4) != 0 ||
      page_stride_bytes != Layout::PageSizeBytes || num_pages <= 0 ||
      kv_head < 0 || kv_head >= Layout::NumKvHeadsValue || n_block < 0 ||
      valid_rows < 0 || valid_rows > block_n || block_n != 128 ||
      page_stride_bytes % alignof(uint4) != 0) {
    __trap();
  }

  const int tidx = static_cast<int>(threadIdx.x);
  const int row0 = kRowsPerThread * (tidx >> 3);
  if (row0 >= valid_rows) {
    return;
  }

  const int pages_per_fa2_block = block_n / Policy::AllocBlockTokens;
  const int logical_page =
      n_block * pages_per_fa2_block + tidx / (2 * kElementsPerVector);
  const int physical_page = block_table[logical_page];
  if (physical_page < 0 || physical_page >= num_pages) {
    __trap();
  }
  if (stage_raw_page_to_fa2_smem(raw_side_base, num_raw_slots, page_to_raw_slot,
                                 physical_page, kv_head, row0, valid_rows,
                                 dst)) {
    return;
  }
  const uint8_t* page =
      byte_v2_cache + static_cast<int64_t>(physical_page) * page_stride_bytes;

  const int row_in_page0 = kRowsPerThread * ((tidx >> 3) & 1);
  const int dim_in_tile0 = kElementsPerVector * (tidx & 1);

  // Stage the compressed 8-byte low plane and 4-byte nibble plane directly
  // into the first 12 bytes of each final 16-byte FA2 vector slot. The copy is
  // committed by the unchanged FA2 mainloop and can overlap QK or PV. After
  // wait_group, the same thread expands the slot in place to eight BF16s.
#pragma unroll 1
  for (int k = 0; k < kDimVectorsPerThread; ++k) {
    const int dim_tile = 4 * k + ((tidx & 7) >> 1);
    const int payload_offset =
        IsValue ? Layout::v_payload_offset(kv_head, dim_tile)
                : Layout::k_payload_offset(kv_head, dim_tile);
#pragma unroll 1
    for (int m = 0; m < kRowsPerThread; ++m) {
      const int elem_idx =
          (row_in_page0 + m) * Policy::CodecDimBlock + dim_in_tile0;
      const bool valid = row0 + m < valid_rows;
      auto* slot = reinterpret_cast<uint8_t*>(&dst(0, m, k));
      cute::SM80_CP_ASYNC_CACHEALWAYS_ZFILL<uint64_t>::copy(
          *reinterpret_cast<const uint64_t*>(page + payload_offset + elem_idx),
          *reinterpret_cast<uint64_t*>(slot), valid);
      cute::SM80_CP_ASYNC_CACHEALWAYS_ZFILL<uint32_t>::copy(
          *reinterpret_cast<const uint32_t*>(
              page + payload_offset + Policy::CodecTileElems + elem_idx / 2),
          *reinterpret_cast<uint32_t*>(slot + sizeof(uint64_t)), valid);
    }
  }

  // V5 has no raw-payload fallback. A committed page carrying either an
  // overflow marker or a fallback bit cannot be decoded exactly.
  if (load_u32(page, Layout::OutlierPoolOverflowOffset) != 0) {
    __trap();
  }
  const uint32_t fallback_mask =
      load_u32(page, IsValue ? Layout::v_fallback_mask_offset(kv_head)
                             : Layout::k_fallback_mask_offset(kv_head));
  const uint32_t outlier_mask =
      load_u32(page, IsValue ? Layout::v_outlier_mask_offset(kv_head)
                             : Layout::k_outlier_mask_offset(kv_head));

  // The last word of the first two slots is unused by compressed staging.
  // Keep all state needed after QK/PV in shared memory instead of extending
  // its live range in registers across the FA2 MMA pipeline.
#pragma unroll 1
  for (int k = 0; k < kDimVectorsPerThread; ++k) {
    const int dim_tile = 4 * k + ((tidx & 7) >> 1);
    const int codec_tile_idx = IsValue ? Layout::v_tile_index(dim_tile)
                                       : Layout::k_tile_index(dim_tile);
    const uint32_t tile_bit = uint32_t{1} << codec_tile_idx;
    if ((fallback_mask & tile_bit) != 0) {
      __trap();
    }

    const uint8_t base =
        page[IsValue ? Layout::v_base_offset(kv_head, dim_tile)
                     : Layout::k_base_offset(kv_head, dim_tile)];
    const bool has_outlier = (outlier_mask & tile_bit) != 0;
    int count = 0;
    int pool_index = 0;
    if (has_outlier) {
      count = IsValue ? Layout::v_outlier_count(page, kv_head, dim_tile)
                      : Layout::k_outlier_count(page, kv_head, dim_tile);
      pool_index = IsValue
                       ? Layout::v_outlier_pool_index(page, kv_head, dim_tile)
                       : Layout::k_outlier_pool_index(page, kv_head, dim_tile);
      if (count < 0 || count > Layout::OutlierEntriesPerTileValue ||
          pool_index < 0 ||
          pool_index + count > Layout::OutlierPoolEntriesValue) {
        __trap();
      }
    }
    constexpr int kHasOutlierShift = 8;
    constexpr int kCountShift = 9;
    constexpr int kPoolIndexShift = 18;
    const uint32_t packed_meta =
        static_cast<uint32_t>(base) |
        (static_cast<uint32_t>(has_outlier) << kHasOutlierShift) |
        (static_cast<uint32_t>(count) << kCountShift) |
        (static_cast<uint32_t>(pool_index) << kPoolIndexShift);
    auto* meta_slot = reinterpret_cast<uint8_t*>(&dst(0, 0, k));
    auto* page_slot = reinterpret_cast<uint8_t*>(&dst(0, 1, k));
    *reinterpret_cast<uint32_t*>(meta_slot + 12) = packed_meta;
    *reinterpret_cast<int*>(page_slot + 12) = physical_page;
  }
}

template <typename DstTensor, typename CoordTensor>
__device__ __forceinline__ void decode_staged_tile_to_fa2_smem_scalar(
    const uint8_t* byte_v2_cache, int64_t page_stride_bytes, int num_pages,
    int num_raw_slots, const int* page_to_raw_slot, const int* block_table,
    int n_block, int block_n, int valid_rows, DstTensor& dst,
    const CoordTensor& coords) {
  constexpr int kElementsPerVector = 8;
  constexpr int kRowsPerThread = 8;
  constexpr int kDimVectorsPerThread = 2;
  CUTE_STATIC_ASSERT_V(cute::size<0>(dst) == cute::Int<8>{});
  CUTE_STATIC_ASSERT_V(cute::size<1>(dst) == cute::Int<8>{});
  CUTE_STATIC_ASSERT_V(cute::size<2>(dst) == cute::Int<2>{});
  CUTE_STATIC_ASSERT_V(cute::size<0>(coords) == cute::Int<8>{});
  CUTE_STATIC_ASSERT_V(cute::size<1>(coords) == cute::Int<8>{});
  CUTE_STATIC_ASSERT_V(cute::size<2>(coords) == cute::Int<2>{});

  const int tidx = static_cast<int>(threadIdx.x);
  const int row0 = kRowsPerThread * (tidx >> 3);
  const uint4 zero = {0, 0, 0, 0};
  if (row0 >= valid_rows) {
#pragma unroll 1
    for (int k = 0; k < kDimVectorsPerThread; ++k) {
#pragma unroll 1
      for (int m = 0; m < kRowsPerThread; ++m) {
        *reinterpret_cast<uint4*>(&dst(0, m, k)) = zero;
      }
    }
    return;
  }
  if (thread_page_is_raw(num_pages, num_raw_slots, page_to_raw_slot,
                         block_table, n_block, block_n)) {
    return;
  }

  const int row_in_page0 = kRowsPerThread * ((tidx >> 3) & 1);
  const int dim_in_tile0 = kElementsPerVector * (tidx & 1);
  constexpr int kHasOutlierShift = 8;
  constexpr int kCountShift = 9;
  constexpr int kPoolIndexShift = 18;
  constexpr uint32_t kCountMask = 0x1ffu;
  constexpr uint32_t kPoolIndexMask = 0x7ffu;

#pragma unroll 1
  for (int k = 0; k < kDimVectorsPerThread; ++k) {
    auto* meta_slot = reinterpret_cast<uint8_t*>(&dst(0, 0, k));
    auto* page_slot = reinterpret_cast<uint8_t*>(&dst(0, 1, k));
    const uint32_t packed_meta =
        *reinterpret_cast<const uint32_t*>(meta_slot + 12);
    const int physical_page = *reinterpret_cast<const int*>(page_slot + 12);
    const uint8_t base = static_cast<uint8_t>(packed_meta);
    const bool has_outlier = ((packed_meta >> kHasOutlierShift) & 1u) != 0;
    const int count =
        static_cast<int>((packed_meta >> kCountShift) & kCountMask);
    const int pool_index =
        static_cast<int>((packed_meta >> kPoolIndexShift) & kPoolIndexMask);

#pragma unroll 1
    for (int m = 0; m < kRowsPerThread; ++m) {
      uint4 decoded = zero;
      if (row0 + m < valid_rows) {
        const uint4 staged = *reinterpret_cast<const uint4*>(&dst(0, m, k));
        const uint64_t lows = static_cast<uint64_t>(staged.x) |
                              (static_cast<uint64_t>(staged.y) << 32);
        decoded = decode_bf16x8(lows, staged.z, base);
      }
      *reinterpret_cast<uint4*>(&dst(0, m, k)) = decoded;
    }

    if (!has_outlier) {
      continue;
    }
    const uint8_t* page =
        byte_v2_cache + static_cast<int64_t>(physical_page) * page_stride_bytes;
    // Baseline vectors are already in shared memory. Scan backwards and patch
    // matching high bytes so duplicate entries retain the scalar decoder's
    // "first entry wins" behavior.
#pragma unroll 1
    for (int entry_idx = count - 1; entry_idx >= 0; --entry_idx) {
      const int entry_offset =
          Layout::OutlierPoolBaseBytes +
          (pool_index + entry_idx) * Layout::OutlierEntryBytes;
      const uint16_t entry = load_u16(page, entry_offset);
      const int outlier_elem = static_cast<int>(
          Layout::OutlierEntryPolicy::decode_elem_index(entry));
      const int outlier_row = outlier_elem / Policy::CodecDimBlock;
      const int outlier_dim = outlier_elem % Policy::CodecDimBlock;
      const int m = outlier_row - row_in_page0;
      const int v = outlier_dim - dim_in_tile0;
      if (m >= 0 && m < kRowsPerThread && v >= 0 && v < kElementsPerVector &&
          row0 + m < valid_rows) {
        auto* dst_bytes = reinterpret_cast<uint8_t*>(&dst(v, m, k));
        dst_bytes[1] = static_cast<uint8_t>(
            Layout::OutlierEntryPolicy::decode_value_bits(entry));
      }
    }
  }
}

template <typename DstTensor>
__device__ __forceinline__ uint8_t* paired_vector_slot(DstTensor& dst, int m,
                                                       int k, int side) {
  auto* own_slot = reinterpret_cast<uint8_t*>(&dst(0, m, k));
  // The FA2 SmemLayoutKV Swizzle<3,3,3> places the adjacent 8-BF16 vector
  // 16 bytes above or below this thread's vector, alternating by row.
  const int partner_delta = side == (m & 1) ? 16 : -16;
  return own_slot + partner_delta;
}

template <bool IsValue, typename DstTensor, typename CoordTensor>
__device__ __forceinline__ void stage_tile_to_fa2_smem_paired_16(
    const uint8_t* byte_v2_cache, const uint8_t* raw_side_base,
    const int* page_to_raw_slot, int64_t page_stride_bytes, int num_pages,
    int num_raw_slots, const int* block_table, int kv_head, int n_block,
    int block_n, int valid_rows, DstTensor& dst, const CoordTensor& coords) {
  constexpr int kRowsPerThread = 8;

  CUTE_STATIC_ASSERT_V(cute::size<0>(dst) == cute::Int<8>{});
  CUTE_STATIC_ASSERT_V(cute::size<1>(dst) == cute::Int<8>{});
  CUTE_STATIC_ASSERT_V(cute::size<2>(dst) == cute::Int<2>{});
  CUTE_STATIC_ASSERT_V(cute::size<0>(coords) == cute::Int<8>{});
  CUTE_STATIC_ASSERT_V(cute::size<1>(coords) == cute::Int<8>{});
  CUTE_STATIC_ASSERT_V(cute::size<2>(coords) == cute::Int<2>{});
  if (byte_v2_cache == nullptr || block_table == nullptr ||
      reinterpret_cast<uintptr_t>(byte_v2_cache) % alignof(uint4) != 0 ||
      page_stride_bytes != Layout::PageSizeBytes || num_pages <= 0 ||
      kv_head < 0 || kv_head >= Layout::NumKvHeadsValue || n_block < 0 ||
      valid_rows < 0 || valid_rows > block_n || block_n != 128 ||
      page_stride_bytes % alignof(uint4) != 0) {
    __trap();
  }

  // tidx = 16 * page + 8 * row_half + column. Even columns own one
  // lower-half dim tile through dst k=0; odd columns own one upper-half dim
  // tile through dst k=1. This assigns every 8x16 page quadrant to exactly
  // one thread while retaining FA2's final shared-memory layout.
  const int tidx = static_cast<int>(threadIdx.x);
  const int row0 = kRowsPerThread * (tidx >> 3);
  if (row0 >= valid_rows) {
    return;
  }
  const int side = tidx & 1;
  const int dst_k = side;
  const int dim_tile = 4 * side + ((tidx & 7) >> 1);
  const int logical_page = n_block * (block_n / Policy::AllocBlockTokens) +
                           tidx / Policy::CodecDimBlock;
  const int physical_page = block_table[logical_page];
  if (physical_page < 0 || physical_page >= num_pages) {
    __trap();
  }
  if (stage_raw_page_to_fa2_smem(raw_side_base, num_raw_slots, page_to_raw_slot,
                                 physical_page, kv_head, row0, valid_rows,
                                 dst)) {
    return;
  }
  const uint8_t* page =
      byte_v2_cache + static_cast<int64_t>(physical_page) * page_stride_bytes;
  uint32_t prefetched_overflow_marker = 0;
  if constexpr (kSidebandPrefetchMode >= 1) {
    prefetched_overflow_marker =
        load_u32_early(page, Layout::OutlierPoolOverflowOffset);
  }
  uint32_t prefetched_fallback_mask = 0;
  if constexpr (kSidebandPrefetchMode >= 2) {
    prefetched_fallback_mask =
        load_u32_early(page, IsValue ? Layout::v_fallback_mask_offset(kv_head)
                                     : Layout::k_fallback_mask_offset(kv_head));
  }
  uint32_t prefetched_outlier_mask = 0;
  if constexpr (kSidebandPrefetchMode >= 3) {
    prefetched_outlier_mask =
        load_u32_early(page, IsValue ? Layout::v_outlier_mask_offset(kv_head)
                                     : Layout::k_outlier_mask_offset(kv_head));
  }
  const int row_in_page0 = kRowsPerThread * ((tidx >> 3) & 1);
  const int payload_offset = IsValue
                                 ? Layout::v_payload_offset(kv_head, dim_tile)
                                 : Layout::k_payload_offset(kv_head, dim_tile);

  // The low plane temporarily occupies this thread's vector slot. The code
  // plane occupies the adjacent vector that the same thread will produce
  // after wait_group. No other thread reads or writes either slot.
#pragma unroll 1
  for (int m = 0; m < kRowsPerThread; ++m) {
    const int row_in_page = row_in_page0 + m;
    const bool valid = row0 + m < valid_rows;
    auto* own_slot = reinterpret_cast<uint8_t*>(&dst(0, m, dst_k));
    auto* partner_slot = paired_vector_slot(dst, m, dst_k, side);
    cute::SM80_CP_ASYNC_CACHEALWAYS_ZFILL<uint4>::copy(
        *reinterpret_cast<const uint4*>(page + payload_offset +
                                        row_in_page * Policy::CodecDimBlock),
        *reinterpret_cast<uint4*>(own_slot), valid);
    if constexpr (kStageMode == 1) {
      cute::SM80_CP_ASYNC_CACHEALWAYS_ZFILL<uint64_t>::copy(
          *reinterpret_cast<const uint64_t*>(
              page + payload_offset + Policy::CodecTileElems +
              row_in_page * Policy::CodecDimBlock / 2),
          *reinterpret_cast<uint64_t*>(partner_slot), valid);
    }
  }
  if constexpr (kStageMode == 2) {
#pragma unroll 1
    for (int row_pair = 0; row_pair < kRowsPerThread / 2; ++row_pair) {
      const int m = 2 * row_pair;
      const int row_in_page = row_in_page0 + m;
      const bool valid = row0 + m < valid_rows;
      auto* code_slot = paired_vector_slot(dst, m, dst_k, side);
      cute::SM80_CP_ASYNC_CACHEALWAYS_ZFILL<uint4>::copy(
          *reinterpret_cast<const uint4*>(
              page + payload_offset + Policy::CodecTileElems +
              row_in_page * Policy::CodecDimBlock / 2),
          *reinterpret_cast<uint4*>(code_slot), valid);
    }
  }

  // V5 has no raw payload from which to recover an invalid compressed page.
  uint32_t overflow_marker;
  if constexpr (kSidebandPrefetchMode >= 1) {
    overflow_marker = prefetched_overflow_marker;
  } else {
    overflow_marker = load_u32(page, Layout::OutlierPoolOverflowOffset);
  }
  if (overflow_marker != 0) {
    __trap();
  }
  uint32_t fallback_mask;
  if constexpr (kSidebandPrefetchMode >= 2) {
    fallback_mask = prefetched_fallback_mask;
  } else {
    fallback_mask =
        load_u32(page, IsValue ? Layout::v_fallback_mask_offset(kv_head)
                               : Layout::k_fallback_mask_offset(kv_head));
  }
  uint32_t outlier_mask;
  if constexpr (kSidebandPrefetchMode >= 3) {
    outlier_mask = prefetched_outlier_mask;
  } else {
    outlier_mask =
        load_u32(page, IsValue ? Layout::v_outlier_mask_offset(kv_head)
                               : Layout::k_outlier_mask_offset(kv_head));
  }
  const int codec_tile_idx =
      IsValue ? Layout::v_tile_index(dim_tile) : Layout::k_tile_index(dim_tile);
  const uint32_t tile_bit = uint32_t{1} << codec_tile_idx;
  if ((fallback_mask & tile_bit) != 0) {
    __trap();
  }

  const uint8_t base = page[IsValue ? Layout::v_base_offset(kv_head, dim_tile)
                                    : Layout::k_base_offset(kv_head, dim_tile)];
  const bool has_outlier = (outlier_mask & tile_bit) != 0;
  int count = 0;
  int pool_index = 0;
  if (has_outlier) {
    count = IsValue ? Layout::v_outlier_count(page, kv_head, dim_tile)
                    : Layout::k_outlier_count(page, kv_head, dim_tile);
    pool_index = IsValue
                     ? Layout::v_outlier_pool_index(page, kv_head, dim_tile)
                     : Layout::k_outlier_pool_index(page, kv_head, dim_tile);
    if (count < 0 || count > Layout::OutlierEntriesPerTileValue ||
        pool_index < 0 ||
        pool_index + count > Layout::OutlierPoolEntriesValue) {
      __trap();
    }
  }
  constexpr int kHasOutlierShift = 8;
  constexpr int kCountShift = 9;
  constexpr int kPoolIndexShift = 18;
  const uint32_t packed_meta =
      static_cast<uint32_t>(base) |
      (static_cast<uint32_t>(has_outlier) << kHasOutlierShift) |
      (static_cast<uint32_t>(count) << kCountShift) |
      (static_cast<uint32_t>(pool_index) << kPoolIndexShift);
  constexpr int kMetadataRow = kStageMode == 1 ? 0 : 1;
  constexpr int kMetadataOffset = kStageMode == 1 ? 8 : 0;
  auto* metadata_slot = paired_vector_slot(dst, kMetadataRow, dst_k, side);
  *reinterpret_cast<uint32_t*>(metadata_slot + kMetadataOffset) = packed_meta;
  *reinterpret_cast<int*>(metadata_slot + kMetadataOffset + 4) = physical_page;
}

template <typename DstTensor, typename CoordTensor>
__device__ __forceinline__ void decode_staged_tile_to_fa2_smem_paired_16(
    const uint8_t* byte_v2_cache, int64_t page_stride_bytes, int num_pages,
    int num_raw_slots, const int* page_to_raw_slot, const int* block_table,
    int n_block, int block_n, int valid_rows, DstTensor& dst,
    const CoordTensor& coords) {
  constexpr int kElementsPerVector = 8;
  constexpr int kRowsPerThread = 8;
  CUTE_STATIC_ASSERT_V(cute::size<0>(dst) == cute::Int<8>{});
  CUTE_STATIC_ASSERT_V(cute::size<1>(dst) == cute::Int<8>{});
  CUTE_STATIC_ASSERT_V(cute::size<2>(dst) == cute::Int<2>{});
  CUTE_STATIC_ASSERT_V(cute::size<0>(coords) == cute::Int<8>{});
  CUTE_STATIC_ASSERT_V(cute::size<1>(coords) == cute::Int<8>{});
  CUTE_STATIC_ASSERT_V(cute::size<2>(coords) == cute::Int<2>{});

  const int tidx = static_cast<int>(threadIdx.x);
  const int row0 = kRowsPerThread * (tidx >> 3);
  const int side = tidx & 1;
  const int dst_k = side;
  const uint4 zero = {0, 0, 0, 0};
  if (row0 >= valid_rows) {
#pragma unroll 1
    for (int m = 0; m < kRowsPerThread; ++m) {
      auto* own_slot = reinterpret_cast<uint4*>(&dst(0, m, dst_k));
      auto* partner_slot =
          reinterpret_cast<uint4*>(paired_vector_slot(dst, m, dst_k, side));
      *own_slot = zero;
      *partner_slot = zero;
    }
    return;
  }
  if (thread_page_is_raw(num_pages, num_raw_slots, page_to_raw_slot,
                         block_table, n_block, block_n)) {
    return;
  }

  constexpr int kHasOutlierShift = 8;
  constexpr int kCountShift = 9;
  constexpr int kPoolIndexShift = 18;
  constexpr uint32_t kCountMask = 0x1ffu;
  constexpr uint32_t kPoolIndexMask = 0x7ffu;
  constexpr int kMetadataRow = kStageMode == 1 ? 0 : 1;
  constexpr int kMetadataOffset = kStageMode == 1 ? 8 : 0;
  auto* metadata_slot = paired_vector_slot(dst, kMetadataRow, dst_k, side);
  const uint32_t packed_meta =
      *reinterpret_cast<const uint32_t*>(metadata_slot + kMetadataOffset);
  const int physical_page =
      *reinterpret_cast<const int*>(metadata_slot + kMetadataOffset + 4);
  const uint8_t base = static_cast<uint8_t>(packed_meta);
  const bool has_outlier = ((packed_meta >> kHasOutlierShift) & 1u) != 0;
  const int count = static_cast<int>((packed_meta >> kCountShift) & kCountMask);
  const int pool_index =
      static_cast<int>((packed_meta >> kPoolIndexShift) & kPoolIndexMask);

  if constexpr (kStageMode == 1) {
#pragma unroll 1
    for (int m = 0; m < kRowsPerThread; ++m) {
      auto* own_slot = reinterpret_cast<uint8_t*>(&dst(0, m, dst_k));
      auto* partner_slot = paired_vector_slot(dst, m, dst_k, side);
      uint4 low_decoded = zero;
      uint4 high_decoded = zero;
      if (row0 + m < valid_rows) {
        const uint4 staged_lows = *reinterpret_cast<const uint4*>(own_slot);
        const uint64_t staged_codes =
            *reinterpret_cast<const uint64_t*>(partner_slot);
        const uint64_t low_lows = static_cast<uint64_t>(staged_lows.x) |
                                  (static_cast<uint64_t>(staged_lows.y) << 32);
        const uint64_t high_lows = static_cast<uint64_t>(staged_lows.z) |
                                   (static_cast<uint64_t>(staged_lows.w) << 32);
        low_decoded =
            decode_bf16x8(low_lows, static_cast<uint32_t>(staged_codes), base);
        high_decoded = decode_bf16x8(
            high_lows, static_cast<uint32_t>(staged_codes >> 32), base);
      }
      auto* low_slot = side == 0 ? own_slot : partner_slot;
      auto* high_slot = side == 0 ? partner_slot : own_slot;
      *reinterpret_cast<uint4*>(low_slot) = low_decoded;
      *reinterpret_cast<uint4*>(high_slot) = high_decoded;
    }
  } else {
    // Decode odd before even. The even partner slot contains both rows' code
    // bytes, so odd output leaves it intact and even output overwrites it only
    // after consuming the remaining low eight bytes.
#pragma unroll 1
    for (int row_pair = 0; row_pair < kRowsPerThread / 2; ++row_pair) {
      auto* code_slot = paired_vector_slot(dst, 2 * row_pair, dst_k, side);
#pragma unroll
      for (int row_phase = 0; row_phase < 2; ++row_phase) {
        const int m = 2 * row_pair + (1 - row_phase);
        auto* own_slot = reinterpret_cast<uint8_t*>(&dst(0, m, dst_k));
        auto* partner_slot = paired_vector_slot(dst, m, dst_k, side);
        uint4 low_decoded = zero;
        uint4 high_decoded = zero;
        if (row0 + m < valid_rows) {
          const uint4 staged_lows = *reinterpret_cast<const uint4*>(own_slot);
          const uint64_t staged_codes =
              *reinterpret_cast<const uint64_t*>(code_slot + (m & 1) * 8);
          const uint64_t low_lows =
              static_cast<uint64_t>(staged_lows.x) |
              (static_cast<uint64_t>(staged_lows.y) << 32);
          const uint64_t high_lows =
              static_cast<uint64_t>(staged_lows.z) |
              (static_cast<uint64_t>(staged_lows.w) << 32);
          low_decoded = decode_bf16x8(
              low_lows, static_cast<uint32_t>(staged_codes), base);
          high_decoded = decode_bf16x8(
              high_lows, static_cast<uint32_t>(staged_codes >> 32), base);
        }
        auto* low_slot = side == 0 ? own_slot : partner_slot;
        auto* high_slot = side == 0 ? partner_slot : own_slot;
        *reinterpret_cast<uint4*>(low_slot) = low_decoded;
        *reinterpret_cast<uint4*>(high_slot) = high_decoded;
      }
    }
  }

  if (!has_outlier) {
    return;
  }
  const uint8_t* page =
      byte_v2_cache + static_cast<int64_t>(physical_page) * page_stride_bytes;
  const int row_in_page0 = kRowsPerThread * ((tidx >> 3) & 1);
  // Scan backwards so duplicate entries preserve scalar first-entry-wins.
#pragma unroll 1
  for (int entry_idx = count - 1; entry_idx >= 0; --entry_idx) {
    const int entry_offset =
        Layout::OutlierPoolBaseBytes +
        (pool_index + entry_idx) * Layout::OutlierEntryBytes;
    const uint16_t entry = load_u16(page, entry_offset);
    const int outlier_elem =
        static_cast<int>(Layout::OutlierEntryPolicy::decode_elem_index(entry));
    const int outlier_row = outlier_elem / Policy::CodecDimBlock;
    const int outlier_dim = outlier_elem % Policy::CodecDimBlock;
    const int m = outlier_row - row_in_page0;
    if (m >= 0 && m < kRowsPerThread && row0 + m < valid_rows) {
      auto* own_slot = reinterpret_cast<uint8_t*>(&dst(0, m, dst_k));
      auto* partner_slot = paired_vector_slot(dst, m, dst_k, side);
      auto* low_slot = side == 0 ? own_slot : partner_slot;
      auto* high_slot = side == 0 ? partner_slot : own_slot;
      auto* dst_bytes =
          outlier_dim < kElementsPerVector
              ? low_slot + 2 * outlier_dim
              : high_slot + 2 * (outlier_dim - kElementsPerVector);
      dst_bytes[1] = static_cast<uint8_t>(
          Layout::OutlierEntryPolicy::decode_value_bits(entry));
    }
  }
}

struct Loader {
  static constexpr int PageBlockSize = Policy::AllocBlockTokens;
  static constexpr int PageSizeBytes = Layout::PageSizeBytes;
  static constexpr int BlockN = 128;
  static constexpr int Threads = 128;
  static constexpr bool ReuseKvSmem = kReuseKvSmem;
  static constexpr bool ReuseKvSmemNonsplit = false;

  template <bool IsValue, typename Params, typename DstTensor,
            typename CoordTensor>
  __device__ static __forceinline__ void stage_tile_to_fa2_smem(
      const Params& params, const int* block_table, int kv_head, int n_block,
      int block_n, int valid_rows, DstTensor& dst, const CoordTensor& coords) {
    const auto* byte_v2_cache =
        reinterpret_cast<const uint8_t*>(params.blockmask);
    const auto* raw_slot_base =
        reinterpret_cast<const uint8_t*>(IsValue ? params.v_ptr : params.k_ptr);
    const auto* page_to_raw_slot =
        reinterpret_cast<const int*>(params.vnew_ptr);
    const int num_pages = static_cast<int>(params.k_batch_stride);
    const int num_raw_slots = static_cast<int>(params.v_batch_stride);
    if constexpr (kStageMode == 0) {
      fa2::stage_tile_to_fa2_smem_scalar<IsValue>(
          byte_v2_cache, raw_slot_base, page_to_raw_slot, PageSizeBytes,
          num_pages, num_raw_slots, block_table, kv_head, n_block, block_n,
          valid_rows, dst, coords);
    } else {
      fa2::stage_tile_to_fa2_smem_paired_16<IsValue>(
          byte_v2_cache, raw_slot_base, page_to_raw_slot, PageSizeBytes,
          num_pages, num_raw_slots, block_table, kv_head, n_block, block_n,
          valid_rows, dst, coords);
    }
  }

  template <typename Params, typename DstTensor, typename CoordTensor>
  __device__ static __forceinline__ void decode_staged_tile_to_fa2_smem(
      const Params& params, const int* block_table, int n_block, int block_n,
      int valid_rows, DstTensor& dst, const CoordTensor& coords) {
    const auto* byte_v2_cache =
        reinterpret_cast<const uint8_t*>(params.blockmask);
    const auto* page_to_raw_slot =
        reinterpret_cast<const int*>(params.vnew_ptr);
    const int num_pages = static_cast<int>(params.k_batch_stride);
    const int num_raw_slots = static_cast<int>(params.v_batch_stride);
    if constexpr (kStageMode == 0) {
      fa2::decode_staged_tile_to_fa2_smem_scalar(
          byte_v2_cache, PageSizeBytes, num_pages, num_raw_slots,
          page_to_raw_slot, block_table, n_block, block_n, valid_rows, dst,
          coords);
    } else {
      fa2::decode_staged_tile_to_fa2_smem_paired_16(
          byte_v2_cache, PageSizeBytes, num_pages, num_raw_slots,
          page_to_raw_slot, block_table, n_block, block_n, valid_rows, dst,
          coords);
    }
  }
};

struct LoaderReuseKvSmemNonsplit : Loader {
  static constexpr bool ReuseKvSmemNonsplit = true;
};

}  // namespace vllm::byte_v2::fa2
