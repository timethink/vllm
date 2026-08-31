// Copyright (c) 2026, ByteV2 contributors.
//
// This file is consumed by the vLLM FlashAttention-2 extension.  It keeps the
// ByteV2-specific global-memory decode separate from the FA2 attention
// mainloop so the latter can retain its original MMA, softmax, split, and
// combine implementation.

#pragma once

#include <cstdint>
#include <type_traits>

#include "byte_v2_layout.cuh"

namespace vllm::byte_v2::fa2 {

enum class ExternalKvFormat : int {
  ByteV2 = 0,
  SplitZip = 1,
  StaticW16 = 2,
  StaticW16Canonical = 3,
};

#ifndef VLLM_BYTE_V2_FA2_LAYOUT_VERSION
  #define VLLM_BYTE_V2_FA2_LAYOUT_VERSION 6
#endif

#if VLLM_BYTE_V2_FA2_LAYOUT_VERSION == 6
using Layout = ByteV2PageLayoutV6<>;
#elif VLLM_BYTE_V2_FA2_LAYOUT_VERSION == 7
using Layout = ByteV2PageLayoutV7<>;
#else
  #error "VLLM_BYTE_V2_FA2_LAYOUT_VERSION must be 6 or 7"
#endif

using SplitZipLayout = ByteV2PageLayoutV5<>;
using Policy = Layout::TilePolicy;
using RawLayout = ByteV2RawStagingLayout<Policy, Layout::NumKvHeadsValue>;

// Reader-only arithmetic Static-W16 page.  The dense representation keeps
// one sign/mantissa byte and one four-bit contiguous exponent-window code per
// BF16 value.  Sixteen tagged ranges index one page-shared 128-entry exact
// patch pool.  A page whose aggregate demand exceeds that pool is served by
// the authoritative raw sidecar.
struct StaticW16Layout {
  static constexpr int NumKvHeadsValue = 8;
  static constexpr int PageSizeBytes = 49792;
  static constexpr int ValuesPerChunk = 16 * 128;
  static constexpr int SignMantissaBaseBytes = 0;
  static constexpr int PackedCodeBaseBytes = 32768;
  static constexpr int HeaderBaseBytes = 49152;
  static constexpr int EscapeBaseBytes = 49280;
  static constexpr int EscapeCapacity = 128;
  static constexpr int EscapeEntryBytes = 4;
  static constexpr int StatusOffset = HeaderBaseBytes + 64;
  static constexpr int KBaseOffset = HeaderBaseBytes + 68;
  static constexpr int VBaseOffset = HeaderBaseBytes + 72;
  static constexpr uint32_t RawFallbackStatus = 1u;
  static constexpr uint32_t CanonicalMetadataStatus = 2u;
  static constexpr uint32_t RangeTag = 0x5732u;

  __host__ __device__ static constexpr int chunk_index(bool is_value,
                                                       int kv_head) {
    return (is_value ? NumKvHeadsValue : 0) + kv_head;
  }

  __host__ __device__ static constexpr int range_offset(bool is_value,
                                                        int kv_head) {
    return HeaderBaseBytes +
           chunk_index(is_value, kv_head) * static_cast<int>(sizeof(uint32_t));
  }

  __host__ __device__ static constexpr int escape_offset(int entry) {
    return EscapeBaseBytes + entry * EscapeEntryBytes;
  }

  __host__ __device__ static constexpr int range_start(uint32_t range) {
    return static_cast<int>(range & 0xffu);
  }

  __host__ __device__ static constexpr int range_count(uint32_t range) {
    return static_cast<int>((range >> 8) & 0xffu);
  }

  __host__ __device__ static constexpr bool range_has_tag(uint32_t range) {
    return (range >> 16) == RangeTag;
  }

  __host__ __device__ static constexpr int canonical_base(uint32_t meta) {
    return static_cast<int>(meta & 0xffu);
  }

  __host__ __device__ static constexpr int canonical_start(uint32_t meta) {
    return static_cast<int>((meta >> 8) & 0xffu);
  }

  __host__ __device__ static constexpr int canonical_count(uint32_t meta) {
    return static_cast<int>((meta >> 16) & 0xffu);
  }
};

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
static_assert(Layout::PageSizeBytes ==
              (VLLM_BYTE_V2_FA2_LAYOUT_VERSION == 6 ? 50560 : 50304));
static_assert(SplitZipLayout::PageSizeBytes == 52096);
static_assert(StaticW16Layout::PageSizeBytes == 49792);
static_assert(StaticW16Layout::EscapeBaseBytes +
                  StaticW16Layout::EscapeCapacity *
                      StaticW16Layout::EscapeEntryBytes ==
              StaticW16Layout::PageSizeBytes);
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

#ifndef VLLM_BYTE_V2_FA2_SHARED_HEAD_METADATA_BYTES
  #define VLLM_BYTE_V2_FA2_SHARED_HEAD_METADATA_BYTES 0
#endif

// Profile-only Static-W16 reader experiment.  Compact K pages are staged in
// a page-local compressed layout and decoded by each consuming warp directly
// into its MMA B fragment.  Keep this disabled in production until the
// duplicated per-warp decode and register footprint pass the full A/B gate.
#ifndef VLLM_BYTE_V2_FA2_K_DIRECT_FRAGMENT
  #define VLLM_BYTE_V2_FA2_K_DIRECT_FRAGMENT 0
#endif

// Profile-only upper-bound and cooperative-routing switch for the exact
// Static-W16 sparse exponent repair. Mode 0 is the production reader; mode 1
// skips compact-page repairs and is intentionally non-bitwise when escapes
// are present; mode 2 loads each escape once per 16-thread page group and
// routes it deterministically to the owning lane; mode 3 overlaps a bounded
// escape prefetch with async payload staging and lets owners filter a shared
// broadcast cache.
#ifndef VLLM_BYTE_V2_FA2_STATIC_W16_ESCAPE_PATCH_MODE
  #define VLLM_BYTE_V2_FA2_STATIC_W16_ESCAPE_PATCH_MODE 0
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
static constexpr bool kStaticW16KDirectFragment =
    VLLM_BYTE_V2_FA2_K_DIRECT_FRAGMENT != 0;
static constexpr int kStaticW16EscapePatchMode =
    VLLM_BYTE_V2_FA2_STATIC_W16_ESCAPE_PATCH_MODE;
static_assert(kStaticW16EscapePatchMode >= 0 && kStaticW16EscapePatchMode <= 3);

// The reuse schedule has 48 KiB of FA2-owned shared memory: 16 KiB for Q and
// one 32 KiB tile aliased by K and V.  Keep one descriptor for each 16-token
// page after that region.  A K stage rebuilds the descriptors cooperatively;
// the matching V stage consumes the same immutable snapshot.
static constexpr int kFa2ReuseSmemBytes = 48 * 1024;
static constexpr int kFa2QSmemBytes = 16 * 1024;
static constexpr int kFa2KvSmemBytes = 32 * 1024;
static constexpr int kFa2SeparateKvSmemBytes =
    kFa2QSmemBytes + 2 * kFa2KvSmemBytes;
static constexpr int kPagesPerFa2Tile = 128 / Policy::AllocBlockTokens;
static constexpr int kHotHeadMetadataBytes = 32;
static_assert(kPagesPerFa2Tile == 8);
static_assert(kFa2QSmemBytes + kFa2KvSmemBytes == kFa2ReuseSmemBytes);

template <typename PageLayout>
struct SharedPageDescriptorTraits {
  static constexpr bool IsPrimaryLayout = std::is_same_v<PageLayout, Layout>;
  static constexpr int RequestedHeadMetadataBytes =
      VLLM_BYTE_V2_FA2_SHARED_HEAD_METADATA_BYTES;
  static constexpr int HeadMetadataBytes =
      IsPrimaryLayout && RequestedHeadMetadataBytes != 0
          ? RequestedHeadMetadataBytes
          : PageLayout::KvHeadRequiredMetaBytes;

  static_assert(HeadMetadataBytes == kHotHeadMetadataBytes ||
                HeadMetadataBytes == PageLayout::KvHeadRequiredMetaBytes);
  static_assert(HeadMetadataBytes % alignof(uint4) == 0);
  static_assert(PageLayout::v_outlier_mask_offset(0) -
                    PageLayout::kv_head_meta_offset(0) + sizeof(uint32_t) ==
                kHotHeadMetadataBytes);
  static_assert(PageLayout::k_outlier_count_offset(0, 0) -
                    PageLayout::kv_head_meta_offset(0) ==
                kHotHeadMetadataBytes);
};

template <typename PageLayout>
struct alignas(16) SharedPageDescriptor {
  static constexpr int HeadMetadataBytes =
      SharedPageDescriptorTraits<PageLayout>::HeadMetadataBytes;

  int physical_page;
  int raw_slot;
  uint32_t overflow_marker;
  uint32_t reserved;
  uint8_t head_metadata[HeadMetadataBytes];
};

template <typename PageLayout>
static constexpr int kSharedPageDescriptorBytes =
    kPagesPerFa2Tile * sizeof(SharedPageDescriptor<PageLayout>);

static_assert(alignof(SharedPageDescriptor<Layout>) == alignof(uint4));
static_assert(alignof(SharedPageDescriptor<SplitZipLayout>) == alignof(uint4));
static_assert(sizeof(SharedPageDescriptor<Layout>) ==
              4 * sizeof(uint32_t) +
                  SharedPageDescriptor<Layout>::HeadMetadataBytes);
static_assert(sizeof(SharedPageDescriptor<SplitZipLayout>) ==
              4 * sizeof(uint32_t) +
                  SharedPageDescriptor<SplitZipLayout>::HeadMetadataBytes);

template <typename PageLayout>
__device__ __forceinline__ SharedPageDescriptor<PageLayout>*
shared_page_descriptors() {
  extern __shared__ char smem_[];
  return reinterpret_cast<SharedPageDescriptor<PageLayout>*>(
      smem_ + kFa2ReuseSmemBytes);
}

struct alignas(16) StaticW16SharedPageDescriptor {
  // The K producer validates these indices before publishing the descriptor.
  // Shared-path V and decode consumers only read descriptors for valid pages.
  int physical_page;
  int raw_slot;
  uint32_t status;
  uint32_t k_base;
  uint32_t v_base;
  uint32_t k_range;
  uint32_t v_range;
  uint32_t reserved;
};

static_assert(sizeof(StaticW16SharedPageDescriptor) == 32);
static constexpr int kStaticW16SharedPageDescriptorBytes =
    kPagesPerFa2Tile * sizeof(StaticW16SharedPageDescriptor);
static constexpr int kStaticW16PrefetchedEscapesPerPage = 16;
static constexpr int kStaticW16PrefetchedEscapeBytes =
    kPagesPerFa2Tile * kStaticW16PrefetchedEscapesPerPage * sizeof(uint32_t);
static_assert(kStaticW16PrefetchedEscapeBytes == 512);
static_assert(kFa2ReuseSmemBytes + kStaticW16SharedPageDescriptorBytes +
                  kStaticW16PrefetchedEscapeBytes <=
              50 * 1024);

template <int DescriptorSmemOffset = kFa2ReuseSmemBytes>
__device__ __forceinline__ StaticW16SharedPageDescriptor*
static_w16_shared_page_descriptors() {
  extern __shared__ char smem_[];
  return reinterpret_cast<StaticW16SharedPageDescriptor*>(smem_ +
                                                          DescriptorSmemOffset);
}

template <int DescriptorSmemOffset = kFa2ReuseSmemBytes>
__device__ __forceinline__ uint32_t* static_w16_shared_prefetched_escapes() {
  extern __shared__ char smem_[];
  return reinterpret_cast<uint32_t*>(smem_ + DescriptorSmemOffset +
                                     kStaticW16SharedPageDescriptorBytes);
}

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

// SplitZip stores one sign/mantissa byte and one 4-bit exponent-window code
// per BF16 value.  The offline page encoder orders each contiguous 16-entry
// codebook by exponent, so the exact exponent is base + code.  Reconstruct
// both BF16 bytes here: unlike ByteV2, SplitZip's low plane does not retain
// the exponent low bit.
__device__ __forceinline__ uint32_t select_byte_high_bits(uint32_t yes,
                                                          uint32_t no) {
  uint32_t result;
  // PTX LUT 0xe4 implements C ? A : B.  Selecting with 0x80808080 takes
  // each byte's high bit from `yes` and all remaining bits from `no`.
  asm("lop3.b32 %0, %1, %2, 0x80808080, 0xe4;"
      : "=r"(result)
      : "r"(yes), "r"(no));
  return result;
}

__device__ __forceinline__ uint4 decode_splitzip_bf16x8(uint64_t sign_mantissas,
                                                        uint32_t codes,
                                                        uint8_t base) {
  constexpr uint32_t kPackedNibbleMask = 0x0f0f0f0fu;
  constexpr uint32_t kInterleaveLow = 0x5140u;
  constexpr uint32_t kInterleaveHigh = 0x7362u;

  const uint32_t sm_0123 = static_cast<uint32_t>(sign_mantissas);
  const uint32_t sm_4567 = static_cast<uint32_t>(sign_mantissas >> 32);
  const uint32_t even_codes = codes & kPackedNibbleMask;
  const uint32_t odd_codes = (codes >> 4) & kPackedNibbleMask;
  const uint32_t base_bytes = static_cast<uint32_t>(base) * 0x01010101u;
  const uint32_t even_exponents = base_bytes + even_codes;
  const uint32_t odd_exponents = base_bytes + odd_codes;
  const uint32_t exponents_0123 =
      __byte_perm(even_exponents, odd_exponents, kInterleaveLow);
  const uint32_t exponents_4567 =
      __byte_perm(even_exponents, odd_exponents, kInterleaveHigh);

  const uint32_t exponent_lows_0123 = exponents_0123 << 7;
  const uint32_t exponent_lows_4567 = exponents_4567 << 7;
  const uint32_t exponent_highs_0123 = exponents_0123 >> 1;
  const uint32_t exponent_highs_4567 = exponents_4567 >> 1;
  const uint32_t lows_0123 = select_byte_high_bits(exponent_lows_0123, sm_0123);
  const uint32_t lows_4567 = select_byte_high_bits(exponent_lows_4567, sm_4567);
  const uint32_t highs_0123 =
      select_byte_high_bits(sm_0123, exponent_highs_0123);
  const uint32_t highs_4567 =
      select_byte_high_bits(sm_4567, exponent_highs_4567);
  return {__byte_perm(lows_0123, highs_0123, kInterleaveLow),
          __byte_perm(lows_0123, highs_0123, kInterleaveHigh),
          __byte_perm(lows_4567, highs_4567, kInterleaveLow),
          __byte_perm(lows_4567, highs_4567, kInterleaveHigh)};
}

template <bool SplitZipFormat>
__device__ __forceinline__ uint4 decode_codec_bf16x8(uint64_t lows,
                                                     uint32_t codes,
                                                     uint8_t base) {
  if constexpr (SplitZipFormat) {
    return decode_splitzip_bf16x8(lows, codes, base);
  } else {
    return decode_bf16x8(lows, codes, base);
  }
}

// The persistent raw sidecar uses the existing ByteV2 staging layout:
// [K/V, kv_head, row, dim].  Each FA2 copy thread owns sixteen aligned BF16
// values for eight rows, so it can populate the final shared-memory tile with
// the same 16-byte transactions as raw paged FA2.  Returning true tells the
// post-copy decoder that this thread's page is already materialized BF16.
template <typename DstTensor>
__device__ __forceinline__ bool stage_raw_slot_to_fa2_smem(
    const uint8_t* raw_side_base, int num_raw_slots, int raw_slot, int kv_head,
    int row0, int valid_rows, DstTensor& dst) {
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

template <typename DstTensor>
__device__ __forceinline__ bool stage_raw_page_to_fa2_smem(
    const uint8_t* raw_side_base, int num_raw_slots,
    const int* page_to_raw_slot, int physical_page, int kv_head, int row0,
    int valid_rows, DstTensor& dst) {
  if (page_to_raw_slot == nullptr) {
    return false;
  }
  return stage_raw_slot_to_fa2_smem(raw_side_base, num_raw_slots,
                                    page_to_raw_slot[physical_page], kv_head,
                                    row0, valid_rows, dst);
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

template <bool IsValue, typename PageLayout = Layout, typename DstTensor,
          typename CoordTensor>
__device__ __forceinline__ void stage_tile_to_fa2_smem_scalar(
    const uint8_t* byte_v2_cache, const uint8_t* raw_side_base,
    const int* page_to_raw_slot, int64_t page_stride_bytes, int num_pages,
    int num_raw_slots, const int* block_table, int kv_head, int n_block,
    int block_n, int valid_rows, DstTensor& dst, const CoordTensor& coords) {
  using Layout = PageLayout;
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

  // The compact page has no embedded raw-payload fallback. A committed page
  // carrying either an
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

template <bool SplitZipFormat = false, typename PageLayout = Layout,
          typename DstTensor, typename CoordTensor>
__device__ __forceinline__ void decode_staged_tile_to_fa2_smem_scalar(
    const uint8_t* byte_v2_cache, int64_t page_stride_bytes, int num_pages,
    int num_raw_slots, const int* page_to_raw_slot, const int* block_table,
    int n_block, int block_n, int valid_rows, DstTensor& dst,
    const CoordTensor& coords) {
  using Layout = PageLayout;
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
        decoded = decode_codec_bf16x8<SplitZipFormat>(lows, staged.z, base);
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
        const uint16_t replacement = static_cast<uint16_t>(
            Layout::OutlierEntryPolicy::decode_value_bits(entry));
        if constexpr (SplitZipFormat) {
          auto* dst_bits = reinterpret_cast<uint16_t*>(dst_bytes);
          *dst_bits =
              static_cast<uint16_t>((*dst_bits & 0x807fu) | (replacement << 7));
        } else {
          dst_bytes[1] = static_cast<uint8_t>(replacement);
        }
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

template <bool IsValue, bool UseSharedPageDescriptor,
          typename PageLayout = Layout, typename DstTensor,
          typename CoordTensor>
__device__ __forceinline__ void stage_tile_to_fa2_smem_paired_16(
    const uint8_t* byte_v2_cache, const uint8_t* raw_side_base,
    const int* page_to_raw_slot, int64_t page_stride_bytes, int num_pages,
    int num_raw_slots, const int* block_table, int kv_head, int n_block,
    int block_n, int valid_rows, DstTensor& dst, const CoordTensor& coords) {
  using Layout = PageLayout;
  using Descriptor = SharedPageDescriptor<Layout>;
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
  const int page_in_tile = tidx / Policy::CodecDimBlock;
  Descriptor* descriptor = nullptr;
  int physical_page = -1;
  int raw_slot = -1;

  if constexpr (UseSharedPageDescriptor && !IsValue) {
    // Every warp owns two complete 16-thread page groups. Keep all 32 lanes
    // participating through the warp barrier, including a padded page group,
    // so a partial final tile never reads its padded block-table entries.
    const bool page_valid =
        page_in_tile * Policy::AllocBlockTokens < valid_rows;
    const int lane_in_warp = tidx & (warpSize - 1);
    const int lane_in_page = tidx & (Policy::CodecDimBlock - 1);
    const int leader_lane = lane_in_warp & ~(Policy::CodecDimBlock - 1);
    uint32_t overflow_marker = 0;
    if (page_valid && lane_in_page == 0) {
      const int logical_page =
          n_block * (block_n / Policy::AllocBlockTokens) + page_in_tile;
      physical_page = block_table[logical_page];
      if (physical_page < 0 || physical_page >= num_pages) {
        __trap();
      }
      if (page_to_raw_slot != nullptr) {
        raw_slot = page_to_raw_slot[physical_page];
        if (raw_slot < -1 || raw_slot >= num_raw_slots) {
          __trap();
        }
      }
      if (raw_slot < 0) {
        const uint8_t* page =
            byte_v2_cache +
            static_cast<int64_t>(physical_page) * page_stride_bytes;
        overflow_marker =
            load_u32_early(page, Layout::OutlierPoolOverflowOffset);
      }
    }
    physical_page = __shfl_sync(0xffffffffu, physical_page, leader_lane);
    raw_slot = __shfl_sync(0xffffffffu, raw_slot, leader_lane);

    descriptor = shared_page_descriptors<Layout>() + page_in_tile;
    if (page_valid && lane_in_page == 0) {
      descriptor->physical_page = physical_page;
      descriptor->raw_slot = raw_slot;
      descriptor->overflow_marker = overflow_marker;
      descriptor->reserved = 0;
    }
    if (page_valid && raw_slot < 0 &&
        lane_in_page < Descriptor::HeadMetadataBytes / sizeof(uint4)) {
      const uint8_t* page =
          byte_v2_cache +
          static_cast<int64_t>(physical_page) * page_stride_bytes;
      const int chunk_offset = lane_in_page * sizeof(uint4);
      const uint4 metadata_chunk = *reinterpret_cast<const uint4*>(
          page + Layout::kv_head_meta_offset(kv_head) + chunk_offset);
      *reinterpret_cast<uint4*>(descriptor->head_metadata + chunk_offset) =
          metadata_chunk;
    }
    __syncwarp();
  }

  if (row0 >= valid_rows) {
    return;
  }

  if constexpr (UseSharedPageDescriptor) {
    descriptor = shared_page_descriptors<Layout>() + page_in_tile;
    if constexpr (IsValue) {
      physical_page = descriptor->physical_page;
      raw_slot = descriptor->raw_slot;
    }
    if (physical_page < 0 || physical_page >= num_pages || raw_slot < -1 ||
        raw_slot >= num_raw_slots) {
      __trap();
    }
  } else {
    const int logical_page =
        n_block * (block_n / Policy::AllocBlockTokens) + page_in_tile;
    physical_page = block_table[logical_page];
    if (physical_page < 0 || physical_page >= num_pages) {
      __trap();
    }
  }

  const int side = tidx & 1;
  const int dst_k = side;
  const int dim_tile = 4 * side + ((tidx & 7) >> 1);
  if constexpr (UseSharedPageDescriptor) {
    if (stage_raw_slot_to_fa2_smem(raw_side_base, num_raw_slots, raw_slot,
                                   kv_head, row0, valid_rows, dst)) {
      return;
    }
  } else {
    if (stage_raw_page_to_fa2_smem(raw_side_base, num_raw_slots,
                                   page_to_raw_slot, physical_page, kv_head,
                                   row0, valid_rows, dst)) {
      return;
    }
  }
  const uint8_t* page =
      byte_v2_cache + static_cast<int64_t>(physical_page) * page_stride_bytes;
  uint32_t prefetched_overflow_marker = 0;
  if constexpr (!UseSharedPageDescriptor && kSidebandPrefetchMode >= 1) {
    prefetched_overflow_marker =
        load_u32_early(page, Layout::OutlierPoolOverflowOffset);
  }
  uint32_t prefetched_fallback_mask = 0;
  if constexpr (!UseSharedPageDescriptor && kSidebandPrefetchMode >= 2) {
    prefetched_fallback_mask =
        load_u32_early(page, IsValue ? Layout::v_fallback_mask_offset(kv_head)
                                     : Layout::k_fallback_mask_offset(kv_head));
  }
  uint32_t prefetched_outlier_mask = 0;
  if constexpr (!UseSharedPageDescriptor && kSidebandPrefetchMode >= 3) {
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

  // A compact page has no embedded raw payload from which to recover an
  // invalid compressed page.
  uint32_t overflow_marker;
  if constexpr (UseSharedPageDescriptor) {
    overflow_marker = descriptor->overflow_marker;
  } else if constexpr (kSidebandPrefetchMode >= 1) {
    overflow_marker = prefetched_overflow_marker;
  } else {
    overflow_marker = load_u32(page, Layout::OutlierPoolOverflowOffset);
  }
  if (overflow_marker != 0) {
    __trap();
  }
  uint32_t fallback_mask;
  uint32_t outlier_mask;
  if constexpr (UseSharedPageDescriptor) {
    const int head_metadata_offset = Layout::kv_head_meta_offset(kv_head);
    fallback_mask =
        load_u32(descriptor->head_metadata,
                 (IsValue ? Layout::v_fallback_mask_offset(kv_head)
                          : Layout::k_fallback_mask_offset(kv_head)) -
                     head_metadata_offset);
    outlier_mask = load_u32(descriptor->head_metadata,
                            (IsValue ? Layout::v_outlier_mask_offset(kv_head)
                                     : Layout::k_outlier_mask_offset(kv_head)) -
                                head_metadata_offset);
  } else if constexpr (kSidebandPrefetchMode >= 2) {
    fallback_mask = prefetched_fallback_mask;
  } else {
    fallback_mask =
        load_u32(page, IsValue ? Layout::v_fallback_mask_offset(kv_head)
                               : Layout::k_fallback_mask_offset(kv_head));
  }
  if constexpr (!UseSharedPageDescriptor) {
    if constexpr (kSidebandPrefetchMode >= 3) {
      outlier_mask = prefetched_outlier_mask;
    } else {
      outlier_mask =
          load_u32(page, IsValue ? Layout::v_outlier_mask_offset(kv_head)
                                 : Layout::k_outlier_mask_offset(kv_head));
    }
  }
  const int codec_tile_idx =
      IsValue ? Layout::v_tile_index(dim_tile) : Layout::k_tile_index(dim_tile);
  const uint32_t tile_bit = uint32_t{1} << codec_tile_idx;
  if ((fallback_mask & tile_bit) != 0) {
    __trap();
  }

  const int head_metadata_offset = Layout::kv_head_meta_offset(kv_head);
  const uint8_t base =
      UseSharedPageDescriptor
          ? descriptor->head_metadata
                [(IsValue ? Layout::v_base_offset(kv_head, dim_tile)
                          : Layout::k_base_offset(kv_head, dim_tile)) -
                 head_metadata_offset]
          : page[IsValue ? Layout::v_base_offset(kv_head, dim_tile)
                         : Layout::k_base_offset(kv_head, dim_tile)];
  const bool has_outlier = (outlier_mask & tile_bit) != 0;
  int count = 0;
  int pool_index = 0;
  if (has_outlier) {
    if constexpr (UseSharedPageDescriptor &&
                  Descriptor::HeadMetadataBytes ==
                      Layout::KvHeadRequiredMetaBytes) {
      const int count_offset =
          (IsValue ? Layout::v_outlier_count_offset(kv_head, dim_tile)
                   : Layout::k_outlier_count_offset(kv_head, dim_tile)) -
          head_metadata_offset;
      const int pool_index_offset =
          (IsValue ? Layout::v_outlier_pool_index_offset(kv_head, dim_tile)
                   : Layout::k_outlier_pool_index_offset(kv_head, dim_tile)) -
          head_metadata_offset;
      if constexpr (Layout::OutlierDescriptorBytesValue == 1) {
        const uint16_t packed_descriptor =
            load_u16(descriptor->head_metadata, count_offset);
        count = static_cast<int>(packed_descriptor & 0xffu) + 1;
        pool_index = static_cast<int>(packed_descriptor >> 8);
      } else {
        count =
            static_cast<int>(load_u16(descriptor->head_metadata, count_offset));
        pool_index = static_cast<int>(
            load_u16(descriptor->head_metadata, pool_index_offset));
      }
    } else {
      if constexpr (Layout::OutlierDescriptorBytesValue == 1) {
        const int count_offset =
            IsValue ? Layout::v_outlier_count_offset(kv_head, dim_tile)
                    : Layout::k_outlier_count_offset(kv_head, dim_tile);
        const uint16_t packed_descriptor = load_u16(page, count_offset);
        count = static_cast<int>(packed_descriptor & 0xffu) + 1;
        pool_index = static_cast<int>(packed_descriptor >> 8);
      } else {
        count = IsValue ? Layout::v_outlier_count(page, kv_head, dim_tile)
                        : Layout::k_outlier_count(page, kv_head, dim_tile);
        pool_index =
            IsValue ? Layout::v_outlier_pool_index(page, kv_head, dim_tile)
                    : Layout::k_outlier_pool_index(page, kv_head, dim_tile);
      }
    }
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
  if constexpr (!UseSharedPageDescriptor) {
    *reinterpret_cast<int*>(metadata_slot + kMetadataOffset + 4) =
        physical_page;
  }
}

template <bool UseSharedPageDescriptor, bool SplitZipFormat = false,
          typename PageLayout = Layout, typename DstTensor,
          typename CoordTensor>
__device__ __forceinline__ void decode_staged_tile_to_fa2_smem_paired_16(
    const uint8_t* byte_v2_cache, int64_t page_stride_bytes, int num_pages,
    int num_raw_slots, const int* page_to_raw_slot, const int* block_table,
    int n_block, int block_n, int valid_rows, DstTensor& dst,
    const CoordTensor& coords) {
  using Layout = PageLayout;
  using Descriptor = SharedPageDescriptor<Layout>;
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
  const Descriptor* descriptor = nullptr;
  if constexpr (UseSharedPageDescriptor) {
    descriptor =
        shared_page_descriptors<Layout>() + tidx / Policy::CodecDimBlock;
    if (descriptor->physical_page < 0 ||
        descriptor->physical_page >= num_pages || descriptor->raw_slot < -1 ||
        descriptor->raw_slot >= num_raw_slots) {
      __trap();
    }
    if (descriptor->raw_slot >= 0) {
      return;
    }
  } else {
    if (thread_page_is_raw(num_pages, num_raw_slots, page_to_raw_slot,
                           block_table, n_block, block_n)) {
      return;
    }
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
      UseSharedPageDescriptor
          ? descriptor->physical_page
          : *reinterpret_cast<const int*>(metadata_slot + kMetadataOffset + 4);
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
        low_decoded = decode_codec_bf16x8<SplitZipFormat>(
            low_lows, static_cast<uint32_t>(staged_codes), base);
        high_decoded = decode_codec_bf16x8<SplitZipFormat>(
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
          low_decoded = decode_codec_bf16x8<SplitZipFormat>(
              low_lows, static_cast<uint32_t>(staged_codes), base);
          high_decoded = decode_codec_bf16x8<SplitZipFormat>(
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
      const uint16_t replacement = static_cast<uint16_t>(
          Layout::OutlierEntryPolicy::decode_value_bits(entry));
      if constexpr (SplitZipFormat) {
        auto* dst_bits = reinterpret_cast<uint16_t*>(dst_bytes);
        *dst_bits =
            static_cast<uint16_t>((*dst_bits & 0x807fu) | (replacement << 7));
      } else {
        dst_bytes[1] = static_cast<uint8_t>(replacement);
      }
    }
  }
}

__device__ __forceinline__ uint32_t
validate_and_pack_static_w16_meta(uint32_t base, uint32_t range) {
  const int range_start = StaticW16Layout::range_start(range);
  const int range_count = StaticW16Layout::range_count(range);
  if (base > 240 || !StaticW16Layout::range_has_tag(range) ||
      range_start > StaticW16Layout::EscapeCapacity ||
      range_count > StaticW16Layout::EscapeCapacity - range_start) {
    __trap();
  }
  return base | ((range & 0xffffu) << 8);
}

template <bool IsValue, bool UseSharedPageDescriptor, bool CanonicalMetadata,
          int DescriptorSmemOffset = kFa2ReuseSmemBytes, typename DstTensor,
          typename CoordTensor>
__device__ __forceinline__ void stage_static_w16_tile_to_fa2_smem(
    const uint8_t* static_w16_cache, const uint8_t* raw_side_base,
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
  if (static_w16_cache == nullptr || block_table == nullptr ||
      reinterpret_cast<uintptr_t>(static_w16_cache) % alignof(uint4) != 0 ||
      page_stride_bytes != StaticW16Layout::PageSizeBytes || num_pages <= 0 ||
      kv_head < 0 || kv_head >= StaticW16Layout::NumKvHeadsValue ||
      n_block < 0 || valid_rows < 0 || valid_rows > block_n || block_n != 128 ||
      page_stride_bytes % alignof(uint4) != 0) {
    __trap();
  }

  const int tidx = static_cast<int>(threadIdx.x);
  const int row0 = kRowsPerThread * (tidx >> 3);
  const int page_in_tile = tidx / Policy::CodecDimBlock;
  StaticW16SharedPageDescriptor* descriptor = nullptr;
  int physical_page = -1;
  int raw_slot = -1;

  if constexpr (UseSharedPageDescriptor && !IsValue) {
    const bool page_valid =
        page_in_tile * Policy::AllocBlockTokens < valid_rows;
    const int lane_in_warp = tidx & (warpSize - 1);
    const int lane_in_page = tidx & (Policy::CodecDimBlock - 1);
    const int leader_lane = lane_in_warp & ~(Policy::CodecDimBlock - 1);
    descriptor = static_w16_shared_page_descriptors<DescriptorSmemOffset>() +
                 page_in_tile;
    if (page_valid && lane_in_page == 0) {
      const int logical_page =
          n_block * (block_n / Policy::AllocBlockTokens) + page_in_tile;
      physical_page = block_table[logical_page];
      if (physical_page < 0 || physical_page >= num_pages) {
        __trap();
      }
      if (page_to_raw_slot != nullptr) {
        raw_slot = page_to_raw_slot[physical_page];
        if (raw_slot < -1 || raw_slot >= num_raw_slots) {
          __trap();
        }
      }
      if constexpr (DescriptorSmemOffset == kFa2SeparateKvSmemBytes) {
        descriptor->physical_page = physical_page;
        descriptor->raw_slot = raw_slot;
      }
    }
    if constexpr (DescriptorSmemOffset == kFa2SeparateKvSmemBytes) {
      // A warp-specialized producer is reached through a role branch.  Use a
      // shared half-warp broadcast here instead of a full-mask shuffle so the
      // producer mapping remains valid under independent thread scheduling.
      __syncwarp();
      if (page_valid) {
        physical_page = descriptor->physical_page;
        raw_slot = descriptor->raw_slot;
      }
    } else {
      physical_page = __shfl_sync(0xffffffffu, physical_page, leader_lane);
      raw_slot = __shfl_sync(0xffffffffu, raw_slot, leader_lane);
    }
    if (page_valid) {
      const uint8_t* page =
          static_w16_cache +
          static_cast<int64_t>(physical_page) * page_stride_bytes;
      if (lane_in_page == 0) {
        const uint32_t status =
            load_u32_early(page, StaticW16Layout::StatusOffset);
        const bool page_is_raw = raw_slot >= 0;
        constexpr uint32_t kExpectedMetadataStatus =
            CanonicalMetadata ? StaticW16Layout::CanonicalMetadataStatus : 0u;
        if ((status & ~StaticW16Layout::RawFallbackStatus) !=
                kExpectedMetadataStatus ||
            ((status & StaticW16Layout::RawFallbackStatus) != 0) !=
                page_is_raw) {
          __trap();
        }
        if constexpr (DescriptorSmemOffset != kFa2SeparateKvSmemBytes) {
          descriptor->physical_page = physical_page;
          descriptor->raw_slot = raw_slot;
        }
        descriptor->status = status;
        descriptor->reserved = 0;
      } else if (lane_in_page == 1 || lane_in_page == 2) {
        uint32_t packed_meta = 0;
        if (raw_slot < 0) {
          const bool meta_is_value = lane_in_page == 2;
          const uint32_t wire_meta = load_u32(
              page, StaticW16Layout::range_offset(meta_is_value, kv_head));
          if constexpr (CanonicalMetadata) {
            // The version bit is published only after prepare validates every
            // descriptor.  The reader can therefore consume the canonical
            // base/start/count word without rebuilding or revalidating it.
            packed_meta = wire_meta;
          } else {
            const uint32_t base =
                load_u32(page, meta_is_value ? StaticW16Layout::VBaseOffset
                                             : StaticW16Layout::KBaseOffset);
            packed_meta = validate_and_pack_static_w16_meta(base, wire_meta);
          }
        }
        // The shared128 specialization uses these two fields as validated
        // packed metadata.  The profile-only cap8 reader retains their legacy
        // meaning in its separate staging function.
        if (lane_in_page == 1) {
          descriptor->k_base = packed_meta;
        } else {
          descriptor->v_base = packed_meta;
        }
      }
    }
    __syncwarp();
  }

  if (row0 >= valid_rows) {
    return;
  }

  uint32_t status = 0;
  uint32_t base = 0;
  uint32_t wire_meta = 0;
  uint32_t packed_meta = 0;
  if constexpr (UseSharedPageDescriptor) {
    descriptor = static_w16_shared_page_descriptors<DescriptorSmemOffset>() +
                 page_in_tile;
    if constexpr (IsValue) {
      physical_page = descriptor->physical_page;
      raw_slot = descriptor->raw_slot;
    }
    packed_meta = IsValue ? descriptor->v_base : descriptor->k_base;
  } else {
    const int logical_page =
        n_block * (block_n / Policy::AllocBlockTokens) + page_in_tile;
    physical_page = block_table[logical_page];
    if (physical_page < 0 || physical_page >= num_pages) {
      __trap();
    }
    if (page_to_raw_slot != nullptr) {
      raw_slot = page_to_raw_slot[physical_page];
      if (raw_slot < -1 || raw_slot >= num_raw_slots) {
        __trap();
      }
    }
    const uint8_t* page =
        static_w16_cache +
        static_cast<int64_t>(physical_page) * page_stride_bytes;
    status = load_u32_early(page, StaticW16Layout::StatusOffset);
    wire_meta = load_u32(page, StaticW16Layout::range_offset(IsValue, kv_head));
    if constexpr (CanonicalMetadata) {
      packed_meta = wire_meta;
    } else {
      base = load_u32(page, IsValue ? StaticW16Layout::VBaseOffset
                                    : StaticW16Layout::KBaseOffset);
    }
  }

  if constexpr (!UseSharedPageDescriptor) {
    const bool page_is_raw = raw_slot >= 0;
    constexpr uint32_t kExpectedMetadataStatus =
        CanonicalMetadata ? StaticW16Layout::CanonicalMetadataStatus : 0u;
    if ((status & ~StaticW16Layout::RawFallbackStatus) !=
            kExpectedMetadataStatus ||
        ((status & StaticW16Layout::RawFallbackStatus) != 0) != page_is_raw) {
      __trap();
    }
  }
  if (stage_raw_slot_to_fa2_smem(raw_side_base, num_raw_slots, raw_slot,
                                 kv_head, row0, valid_rows, dst)) {
    return;
  }
  if constexpr (!UseSharedPageDescriptor && !CanonicalMetadata) {
    packed_meta = validate_and_pack_static_w16_meta(base, wire_meta);
  }

  const int side = tidx & 1;
  const int dst_k = side;
  const int dim_tile = 4 * side + ((tidx & 7) >> 1);
  const int row_in_page0 = kRowsPerThread * ((tidx >> 3) & 1);
  const int chunk = StaticW16Layout::chunk_index(IsValue, kv_head);
  const int sign_mantissa_base = StaticW16Layout::SignMantissaBaseBytes +
                                 chunk * StaticW16Layout::ValuesPerChunk;
  const int packed_code_base = StaticW16Layout::PackedCodeBaseBytes +
                               chunk * (StaticW16Layout::ValuesPerChunk / 2);
  const uint8_t* page = static_w16_cache +
                        static_cast<int64_t>(physical_page) * page_stride_bytes;

#pragma unroll 1
  for (int m = 0; m < kRowsPerThread; ++m) {
    const int row_in_page = row_in_page0 + m;
    const bool valid = row0 + m < valid_rows;
    auto* own_slot = reinterpret_cast<uint8_t*>(&dst(0, m, dst_k));
    cute::SM80_CP_ASYNC_CACHEALWAYS_ZFILL<uint4>::copy(
        *reinterpret_cast<const uint4*>(page + sign_mantissa_base +
                                        row_in_page * Policy::HeadDim +
                                        dim_tile * Policy::CodecDimBlock),
        *reinterpret_cast<uint4*>(own_slot), valid);
  }
#pragma unroll 1
  for (int m = 0; m < kRowsPerThread; ++m) {
    const int row_in_page = row_in_page0 + m;
    const bool valid = row0 + m < valid_rows;
    auto* code_slot = paired_vector_slot(dst, m, dst_k, side);
    cute::SM80_CP_ASYNC_CACHEALWAYS_ZFILL<uint64_t>::copy(
        *reinterpret_cast<const uint64_t*>(
            page + packed_code_base + row_in_page * (Policy::HeadDim / 2) +
            dim_tile * (Policy::CodecDimBlock / 2)),
        *reinterpret_cast<uint64_t*>(code_slot), valid);
  }

  auto* metadata_slot = paired_vector_slot(dst, 0, dst_k, side);
  *reinterpret_cast<uint32_t*>(metadata_slot + 8) = packed_meta;
  if constexpr (!UseSharedPageDescriptor) {
    *reinterpret_cast<int*>(metadata_slot + 12) = physical_page;
  }

  if constexpr (UseSharedPageDescriptor && kStaticW16EscapePatchMode == 3) {
    const int start = static_cast<int>((packed_meta >> 8) & 0xffu);
    const int count = static_cast<int>((packed_meta >> 16) & 0xffu);
    if (count <= kStaticW16PrefetchedEscapesPerPage) {
      const int lane_in_page = tidx & (Policy::CodecDimBlock - 1);
      uint32_t* prefetched =
          static_w16_shared_prefetched_escapes<DescriptorSmemOffset>() +
          page_in_tile * kStaticW16PrefetchedEscapesPerPage;
#pragma unroll 1
      for (int entry_idx = lane_in_page; entry_idx < count;
           entry_idx += Policy::CodecDimBlock) {
        const uint32_t entry =
            load_u32(page, StaticW16Layout::escape_offset(start + entry_idx));
        const int local_pos = static_cast<int>(entry & 0xffffu);
        if (local_pos < 0 || local_pos >= StaticW16Layout::ValuesPerChunk) {
          __trap();
        }
        const int owner_row_group = local_pos >> 10;
        const int owner_dim_tile = (local_pos & 0x7f) >> 4;
        const int owner_lane = owner_row_group * 8 + (owner_dim_tile & 3) * 2 +
                               (owner_dim_tile >> 2);
        prefetched[entry_idx] =
            (entry & 0x00ffffffu) | (static_cast<uint32_t>(owner_lane) << 24);
      }
    }
  }
}

template <bool UseSharedPageDescriptor,
          int DescriptorSmemOffset = kFa2ReuseSmemBytes, typename DstTensor,
          typename CoordTensor>
__device__ __forceinline__ void decode_static_w16_tile_to_fa2_smem(
    const uint8_t* static_w16_cache, int64_t page_stride_bytes, int num_pages,
    int num_raw_slots, const int* page_to_raw_slot, const int* block_table,
    int n_block, int block_n, int valid_rows, DstTensor& dst,
    const CoordTensor& coords, int logical_tidx = -1, int m_begin = 0,
    int m_end = 8, bool use_packed_meta_override = false,
    uint32_t packed_meta_override = 0) {
  constexpr int kElementsPerVector = 8;
  constexpr int kRowsPerThread = 8;
  CUTE_STATIC_ASSERT_V(cute::size<0>(dst) == cute::Int<8>{});
  CUTE_STATIC_ASSERT_V(cute::size<1>(dst) == cute::Int<8>{});
  CUTE_STATIC_ASSERT_V(cute::size<2>(dst) == cute::Int<2>{});
  CUTE_STATIC_ASSERT_V(cute::size<0>(coords) == cute::Int<8>{});
  CUTE_STATIC_ASSERT_V(cute::size<1>(coords) == cute::Int<8>{});
  CUTE_STATIC_ASSERT_V(cute::size<2>(coords) == cute::Int<2>{});

  if (m_begin < 0 || m_begin > m_end || m_end > kRowsPerThread) {
    __trap();
  }
  const int tidx =
      logical_tidx >= 0 ? logical_tidx : static_cast<int>(threadIdx.x);
  const int row0 = kRowsPerThread * (tidx >> 3);
  const int side = tidx & 1;
  const int dst_k = side;
  const uint4 zero = {0, 0, 0, 0};
  if (row0 >= valid_rows) {
#pragma unroll 1
    for (int m = m_begin; m < m_end; ++m) {
      auto* own_slot = reinterpret_cast<uint4*>(&dst(0, m, dst_k));
      auto* partner_slot =
          reinterpret_cast<uint4*>(paired_vector_slot(dst, m, dst_k, side));
      *own_slot = zero;
      *partner_slot = zero;
    }
    return;
  }

  const StaticW16SharedPageDescriptor* descriptor = nullptr;
  if constexpr (UseSharedPageDescriptor) {
    descriptor = static_w16_shared_page_descriptors<DescriptorSmemOffset>() +
                 tidx / Policy::CodecDimBlock;
    if (descriptor->raw_slot >= 0) {
      return;
    }
  } else if (thread_page_is_raw(num_pages, num_raw_slots, page_to_raw_slot,
                                block_table, n_block, block_n)) {
    return;
  }

  auto* metadata_slot = paired_vector_slot(dst, 0, dst_k, side);
  const uint32_t packed_meta =
      use_packed_meta_override
          ? packed_meta_override
          : *reinterpret_cast<const uint32_t*>(metadata_slot + 8);
  const int physical_page =
      UseSharedPageDescriptor
          ? descriptor->physical_page
          : *reinterpret_cast<const int*>(metadata_slot + 12);
  const uint8_t base = static_cast<uint8_t>(packed_meta);
  const int start = static_cast<int>((packed_meta >> 8) & 0xffu);
  const int count = static_cast<int>((packed_meta >> 16) & 0xffu);

#pragma unroll 1
  for (int m = m_begin; m < m_end; ++m) {
    auto* own_slot = reinterpret_cast<uint8_t*>(&dst(0, m, dst_k));
    auto* partner_slot = paired_vector_slot(dst, m, dst_k, side);
    uint4 low_decoded = zero;
    uint4 high_decoded = zero;
    if (row0 + m < valid_rows) {
      const uint4 staged_sign_mantissas =
          *reinterpret_cast<const uint4*>(own_slot);
      const uint64_t staged_codes =
          *reinterpret_cast<const uint64_t*>(partner_slot);
      const uint64_t low_sign_mantissas =
          static_cast<uint64_t>(staged_sign_mantissas.x) |
          (static_cast<uint64_t>(staged_sign_mantissas.y) << 32);
      const uint64_t high_sign_mantissas =
          static_cast<uint64_t>(staged_sign_mantissas.z) |
          (static_cast<uint64_t>(staged_sign_mantissas.w) << 32);
      low_decoded = decode_splitzip_bf16x8(
          low_sign_mantissas, static_cast<uint32_t>(staged_codes), base);
      high_decoded = decode_splitzip_bf16x8(
          high_sign_mantissas, static_cast<uint32_t>(staged_codes >> 32), base);
    }
    auto* low_slot = side == 0 ? own_slot : partner_slot;
    auto* high_slot = side == 0 ? partner_slot : own_slot;
    *reinterpret_cast<uint4*>(low_slot) = low_decoded;
    *reinterpret_cast<uint4*>(high_slot) = high_decoded;
  }

  if constexpr (kStaticW16EscapePatchMode == 1) {
    // Deliberately inexact upper bound: raw-fallback pages returned above and
    // remain exact, while compact pages omit only sparse exponent repair.
    return;
  }

  if (count == 0) {
    return;
  }
  if constexpr (!UseSharedPageDescriptor) {
    if (start > StaticW16Layout::EscapeCapacity ||
        count > StaticW16Layout::EscapeCapacity - start) {
      __trap();
    }
  }
  const uint8_t* page = static_w16_cache +
                        static_cast<int64_t>(physical_page) * page_stride_bytes;
  const int row_in_page0 = kRowsPerThread * ((tidx >> 3) & 1);
  const int dim_tile = 4 * side + ((tidx & 7) >> 1);
  const int dim_base = dim_tile * Policy::CodecDimBlock;

  if constexpr (UseSharedPageDescriptor && kStaticW16EscapePatchMode == 3) {
    const int page_in_tile = tidx / Policy::CodecDimBlock;
    const bool full_page =
        (page_in_tile + 1) * Policy::CodecTokenBlock <= valid_rows;
    if (full_page && count <= kStaticW16PrefetchedEscapesPerPage) {
      const int lane_in_warp = tidx & (warpSize - 1);
      const int lane_in_page = tidx & (Policy::CodecDimBlock - 1);
      const int page_leader = lane_in_warp & ~(Policy::CodecDimBlock - 1);
      const unsigned int page_lane_mask = 0xffffu << page_leader;
      __syncwarp(page_lane_mask);
      const uint32_t* prefetched =
          static_w16_shared_prefetched_escapes<DescriptorSmemOffset>() +
          page_in_tile * kStaticW16PrefetchedEscapesPerPage;
#pragma unroll 1
      for (int entry_idx = count - 1; entry_idx >= 0; --entry_idx) {
        const uint32_t entry = prefetched[entry_idx];
        if (static_cast<int>(entry >> 24) == lane_in_page) {
          const int local_pos = static_cast<int>(entry & 0xffffu);
          const int m = (local_pos >> 7) & 7;
          const int v = local_pos & (Policy::CodecDimBlock - 1);
          if (m < m_begin || m >= m_end) {
            continue;
          }
          auto* own_slot = reinterpret_cast<uint8_t*>(&dst(0, m, dst_k));
          auto* partner_slot = paired_vector_slot(dst, m, dst_k, side);
          auto* low_slot = side == 0 ? own_slot : partner_slot;
          auto* high_slot = side == 0 ? partner_slot : own_slot;
          auto* dst_bits = reinterpret_cast<uint16_t*>(
              v < kElementsPerVector
                  ? low_slot + 2 * v
                  : high_slot + 2 * (v - kElementsPerVector));
          const uint16_t exponent =
              static_cast<uint16_t>((entry >> 16) & 0xffu);
          *dst_bits =
              static_cast<uint16_t>((*dst_bits & 0x807fu) | (exponent << 7));
        }
      }
      return;
    }
  }

  if constexpr (kStaticW16EscapePatchMode == 2) {
    // A sealed compact page normally has all 16 rows. Keep the legacy owner
    // scan as a defensive fallback if a future wire admits a partial compact
    // page, because inactive rows returned before this half-warp exchange.
    const int page_in_tile = tidx / Policy::CodecDimBlock;
    const bool full_page =
        (page_in_tile + 1) * Policy::CodecTokenBlock <= valid_rows;
    if (full_page) {
      const int lane_in_warp = tidx & (warpSize - 1);
      const int lane_in_page = tidx & (Policy::CodecDimBlock - 1);
      const int page_leader = lane_in_warp & ~(Policy::CodecDimBlock - 1);
      const unsigned int page_lane_mask = 0xffffu << page_leader;

#pragma unroll 1
      for (int batch = 0; batch < count; batch += Policy::CodecDimBlock) {
        const int entry_idx = count - 1 - batch - lane_in_page;
        uint32_t owned_entry = 0;
        int owner_lane = -1;
        if (entry_idx >= 0) {
          owned_entry =
              load_u32(page, StaticW16Layout::escape_offset(start + entry_idx));
          const int local_pos = static_cast<int>(owned_entry & 0xffffu);
          if (local_pos < 0 || local_pos >= StaticW16Layout::ValuesPerChunk) {
            __trap();
          }
          const int owner_row_group = local_pos >> 10;
          const int owner_dim_tile = (local_pos & 0x7f) >> 4;
          owner_lane = owner_row_group * 8 + (owner_dim_tile & 3) * 2 +
                       (owner_dim_tile >> 2);
        }

        const int batch_count = min(count - batch, Policy::CodecDimBlock);
#pragma unroll 1
        for (int source_lane = 0; source_lane < batch_count; ++source_lane) {
          const uint32_t entry = __shfl_sync(page_lane_mask, owned_entry,
                                             page_leader + source_lane);
          const int routed_owner = __shfl_sync(page_lane_mask, owner_lane,
                                               page_leader + source_lane);
          if (routed_owner == lane_in_page) {
            const int local_pos = static_cast<int>(entry & 0xffffu);
            const int m = (local_pos >> 7) & 7;
            const int v = local_pos & (Policy::CodecDimBlock - 1);
            if (m < m_begin || m >= m_end) {
              continue;
            }
            auto* own_slot = reinterpret_cast<uint8_t*>(&dst(0, m, dst_k));
            auto* partner_slot = paired_vector_slot(dst, m, dst_k, side);
            auto* low_slot = side == 0 ? own_slot : partner_slot;
            auto* high_slot = side == 0 ? partner_slot : own_slot;
            auto* dst_bits = reinterpret_cast<uint16_t*>(
                v < kElementsPerVector
                    ? low_slot + 2 * v
                    : high_slot + 2 * (v - kElementsPerVector));
            const uint16_t exponent =
                static_cast<uint16_t>((entry >> 16) & 0xffu);
            *dst_bits =
                static_cast<uint16_t>((*dst_bits & 0x807fu) | (exponent << 7));
          }
        }
      }
      return;
    }
  }

#pragma unroll 1
  for (int entry_idx = count - 1; entry_idx >= 0; --entry_idx) {
    const uint32_t entry =
        load_u32(page, StaticW16Layout::escape_offset(start + entry_idx));
    const int local_pos = static_cast<int>(entry & 0xffffu);
    if (local_pos < 0 || local_pos >= StaticW16Layout::ValuesPerChunk) {
      __trap();
    }
    const int outlier_row = local_pos / Policy::HeadDim;
    const int outlier_dim = local_pos % Policy::HeadDim;
    const int m = outlier_row - row_in_page0;
    const int v = outlier_dim - dim_base;
    if (m >= m_begin && m < m_end && v >= 0 && v < Policy::CodecDimBlock &&
        row0 + m < valid_rows) {
      auto* own_slot = reinterpret_cast<uint8_t*>(&dst(0, m, dst_k));
      auto* partner_slot = paired_vector_slot(dst, m, dst_k, side);
      auto* low_slot = side == 0 ? own_slot : partner_slot;
      auto* high_slot = side == 0 ? partner_slot : own_slot;
      auto* dst_bits = reinterpret_cast<uint16_t*>(
          v < kElementsPerVector ? low_slot + 2 * v
                                 : high_slot + 2 * (v - kElementsPerVector));
      const uint16_t exponent = static_cast<uint16_t>((entry >> 16) & 0xffu);
      *dst_bits =
          static_cast<uint16_t>((*dst_bits & 0x807fu) | (exponent << 7));
    }
  }
}

// K-direct-fragment staging uses one 4 KiB shared-memory slot per 16-token
// page. A compact page occupies 2 KiB of sign/mantissa bytes followed by
// 1 KiB of packed exponent codes. A raw page uses the whole slot as BF16.
// The remaining compact-page KiB starts with a sparse-escape presence bitmap;
// retaining a fixed page stride keeps compact/raw addressing uniform in the
// consumer.
static constexpr int kStaticW16DirectPageSlotBytes = 4 * 1024;
static constexpr int kStaticW16DirectSignMantissaBytes = 2 * 1024;
static constexpr int kStaticW16DirectPackedCodeOffset = 2 * 1024;
static constexpr int kStaticW16DirectPackedCodeBytes = 1 * 1024;
static constexpr int kStaticW16DirectEscapeMaskOffset = 3 * 1024;
static constexpr int kStaticW16DirectEscapeMaskBytes =
    StaticW16Layout::ValuesPerChunk / 8;
static_assert(kPagesPerFa2Tile * kStaticW16DirectPageSlotBytes ==
              kFa2KvSmemBytes);
static_assert(kStaticW16DirectSignMantissaBytes +
                  kStaticW16DirectPackedCodeBytes <=
              kStaticW16DirectPageSlotBytes);
static_assert(kStaticW16DirectEscapeMaskBytes == 256);
static_assert(kStaticW16DirectEscapeMaskOffset +
                  kStaticW16DirectEscapeMaskBytes <=
              kStaticW16DirectPageSlotBytes);

__device__ __forceinline__ uint8_t* static_w16_direct_k_smem() {
  extern __shared__ char smem_[];
  return reinterpret_cast<uint8_t*>(smem_) + kFa2QSmemBytes;
}

template <bool CanonicalMetadata, typename DstTensor, typename CoordTensor>
__device__ __forceinline__ void stage_static_w16_k_tile_direct_fragment(
    const uint8_t* static_w16_cache, const uint8_t* raw_side_base,
    const int* page_to_raw_slot, int64_t page_stride_bytes, int num_pages,
    int num_raw_slots, const int* block_table, int kv_head, int n_block,
    int block_n, int valid_rows, DstTensor& dst, const CoordTensor& coords) {
  CUTE_STATIC_ASSERT_V(cute::size<0>(dst) == cute::Int<8>{});
  CUTE_STATIC_ASSERT_V(cute::size<1>(dst) == cute::Int<8>{});
  CUTE_STATIC_ASSERT_V(cute::size<2>(dst) == cute::Int<2>{});
  CUTE_STATIC_ASSERT_V(cute::size<0>(coords) == cute::Int<8>{});
  CUTE_STATIC_ASSERT_V(cute::size<1>(coords) == cute::Int<8>{});
  CUTE_STATIC_ASSERT_V(cute::size<2>(coords) == cute::Int<2>{});
  if (static_w16_cache == nullptr || block_table == nullptr ||
      reinterpret_cast<uintptr_t>(static_w16_cache) % alignof(uint4) != 0 ||
      page_stride_bytes != StaticW16Layout::PageSizeBytes || num_pages <= 0 ||
      kv_head < 0 || kv_head >= StaticW16Layout::NumKvHeadsValue ||
      n_block < 0 || valid_rows < 0 || valid_rows > block_n || block_n != 128 ||
      page_stride_bytes % alignof(uint4) != 0) {
    __trap();
  }

  const int tidx = static_cast<int>(threadIdx.x);
  const int page_in_tile = tidx / Policy::CodecTokenBlock;
  const int row_in_page = tidx & (Policy::CodecTokenBlock - 1);
  const bool page_valid = page_in_tile * Policy::CodecTokenBlock < valid_rows;
  const int lane_in_warp = tidx & (warpSize - 1);
  const int lane_in_page = tidx & (Policy::CodecDimBlock - 1);
  const int leader_lane = lane_in_warp & ~(Policy::CodecDimBlock - 1);
  const unsigned int page_lane_mask = 0xffffu << leader_lane;
  int physical_page = -1;
  int raw_slot = -1;

  if (page_valid && lane_in_page == 0) {
    const int logical_page =
        n_block * (block_n / Policy::AllocBlockTokens) + page_in_tile;
    physical_page = block_table[logical_page];
    if (physical_page < 0 || physical_page >= num_pages) {
      __trap();
    }
    if (page_to_raw_slot != nullptr) {
      raw_slot = page_to_raw_slot[physical_page];
      if (raw_slot < -1 || raw_slot >= num_raw_slots) {
        __trap();
      }
    }
  }
  physical_page = __shfl_sync(0xffffffffu, physical_page, leader_lane);
  raw_slot = __shfl_sync(0xffffffffu, raw_slot, leader_lane);

  StaticW16SharedPageDescriptor* descriptor =
      static_w16_shared_page_descriptors() + page_in_tile;
  if (page_valid) {
    const uint8_t* page =
        static_w16_cache +
        static_cast<int64_t>(physical_page) * page_stride_bytes;
    if (lane_in_page == 0) {
      const uint32_t status =
          load_u32_early(page, StaticW16Layout::StatusOffset);
      const bool page_is_raw = raw_slot >= 0;
      constexpr uint32_t kExpectedMetadataStatus =
          CanonicalMetadata ? StaticW16Layout::CanonicalMetadataStatus : 0u;
      if ((status & ~StaticW16Layout::RawFallbackStatus) !=
              kExpectedMetadataStatus ||
          ((status & StaticW16Layout::RawFallbackStatus) != 0) != page_is_raw) {
        __trap();
      }
      descriptor->physical_page = physical_page;
      descriptor->raw_slot = raw_slot;
      descriptor->status = status;
      descriptor->reserved = 0;
    } else if (lane_in_page == 1 || lane_in_page == 2) {
      uint32_t packed_meta = 0;
      if (raw_slot < 0) {
        const bool meta_is_value = lane_in_page == 2;
        const uint32_t wire_meta = load_u32(
            page, StaticW16Layout::range_offset(meta_is_value, kv_head));
        if constexpr (CanonicalMetadata) {
          packed_meta = wire_meta;
        } else {
          const uint32_t base =
              load_u32(page, meta_is_value ? StaticW16Layout::VBaseOffset
                                           : StaticW16Layout::KBaseOffset);
          packed_meta = validate_and_pack_static_w16_meta(base, wire_meta);
        }
      }
      if (lane_in_page == 1) {
        descriptor->k_base = packed_meta;
      } else {
        descriptor->v_base = packed_meta;
      }
    }
  }
  __syncwarp();

  // Most Static-W16 values use the contiguous exponent window, but testing
  // descriptor->k_range alone would make every MMA fragment rescan the sparse
  // escape list.  Materialize one exact presence bit per K value in the
  // otherwise-unused compact-page KiB.  The direct consumer can then keep the
  // common path entirely in shared memory and call the global escape lookup
  // only for a fragment that actually contains an escaped value.
  if (page_valid && raw_slot < 0) {
    uint8_t* page_slot = static_w16_direct_k_smem() +
                         page_in_tile * kStaticW16DirectPageSlotBytes;
    auto* escape_mask =
        reinterpret_cast<uint4*>(page_slot + kStaticW16DirectEscapeMaskOffset);
    escape_mask[lane_in_page] = make_uint4(0, 0, 0, 0);
    __syncwarp(page_lane_mask);

    const uint32_t packed_meta = descriptor->k_base;
    const int start = static_cast<int>((packed_meta >> 8) & 0xffu);
    const int count = static_cast<int>((packed_meta >> 16) & 0xffu);
    const uint8_t* page =
        static_w16_cache +
        static_cast<int64_t>(physical_page) * page_stride_bytes;
    auto* escape_mask_words = reinterpret_cast<unsigned int*>(escape_mask);
    for (int entry_idx = lane_in_page; entry_idx < count;
         entry_idx += Policy::CodecDimBlock) {
      const uint32_t entry =
          load_u32(page, StaticW16Layout::escape_offset(start + entry_idx));
      const unsigned int local_pos = entry & 0xffffu;
      if (local_pos >= StaticW16Layout::ValuesPerChunk) {
        __trap();
      }
      atomicOr(escape_mask_words + local_pos / 32, 1u << (local_pos & 31));
    }
  }
  __syncwarp();

  if (tidx >= valid_rows) {
    return;
  }
  physical_page = descriptor->physical_page;
  raw_slot = descriptor->raw_slot;
  uint8_t* page_slot =
      static_w16_direct_k_smem() + page_in_tile * kStaticW16DirectPageSlotBytes;

  if (raw_slot >= 0) {
    if (raw_side_base == nullptr || raw_slot >= num_raw_slots ||
        reinterpret_cast<uintptr_t>(raw_side_base) % alignof(uint4) != 0) {
      __trap();
    }
    const uint8_t* raw_slot_base =
        raw_side_base +
        static_cast<int64_t>(raw_slot) * RawLayout::SlotSizeBytes;
    const int raw_row_offset =
        (kv_head * Policy::AllocBlockTokens + row_in_page) * Policy::HeadDim *
        RawLayout::RawElementBytesValue;
#pragma unroll
    for (int vector = 0; vector < Policy::HeadDim / 8; ++vector) {
      cute::SM80_CP_ASYNC_CACHEALWAYS_ZFILL<uint4>::copy(
          *reinterpret_cast<const uint4*>(raw_slot_base + raw_row_offset +
                                          vector * sizeof(uint4)),
          *reinterpret_cast<uint4*>(page_slot +
                                    row_in_page * Policy::HeadDim *
                                        RawLayout::RawElementBytesValue +
                                    vector * sizeof(uint4)),
          true);
    }
    return;
  }

  const uint8_t* page = static_w16_cache +
                        static_cast<int64_t>(physical_page) * page_stride_bytes;
  const int chunk = StaticW16Layout::chunk_index(false, kv_head);
  const int sign_mantissa_offset = StaticW16Layout::SignMantissaBaseBytes +
                                   chunk * StaticW16Layout::ValuesPerChunk +
                                   row_in_page * Policy::HeadDim;
  const int packed_code_offset = StaticW16Layout::PackedCodeBaseBytes +
                                 chunk * (StaticW16Layout::ValuesPerChunk / 2) +
                                 row_in_page * (Policy::HeadDim / 2);
#pragma unroll
  for (int vector = 0; vector < Policy::HeadDim / 16; ++vector) {
    cute::SM80_CP_ASYNC_CACHEALWAYS_ZFILL<uint4>::copy(
        *reinterpret_cast<const uint4*>(page + sign_mantissa_offset +
                                        vector * sizeof(uint4)),
        *reinterpret_cast<uint4*>(page_slot + row_in_page * Policy::HeadDim +
                                  vector * sizeof(uint4)),
        true);
  }
#pragma unroll
  for (int vector = 0; vector < Policy::HeadDim / 32; ++vector) {
    cute::SM80_CP_ASYNC_CACHEALWAYS_ZFILL<uint4>::copy(
        *reinterpret_cast<const uint4*>(page + packed_code_offset +
                                        vector * sizeof(uint4)),
        *reinterpret_cast<uint4*>(page_slot + kStaticW16DirectPackedCodeOffset +
                                  row_in_page * (Policy::HeadDim / 2) +
                                  vector * sizeof(uint4)),
        true);
  }
}

template <typename Params>
__device__ __forceinline__ uint16_t load_static_w16_direct_k_bf16_bits(
    const Params& params, int row, int dim, int valid_rows) {
  if (row < 0 || row >= valid_rows || dim < 0 || dim >= Policy::HeadDim) {
    return 0;
  }
  const int page_in_tile = row / Policy::CodecTokenBlock;
  const int row_in_page = row & (Policy::CodecTokenBlock - 1);
  const StaticW16SharedPageDescriptor* descriptor =
      static_w16_shared_page_descriptors() + page_in_tile;
  const uint8_t* page_slot =
      static_w16_direct_k_smem() + page_in_tile * kStaticW16DirectPageSlotBytes;
  if (descriptor->raw_slot >= 0) {
    return reinterpret_cast<const uint16_t*>(
        page_slot)[row_in_page * Policy::HeadDim + dim];
  }

  const uint8_t sign_mantissa = page_slot[row_in_page * Policy::HeadDim + dim];
  const uint8_t packed_codes =
      page_slot[kStaticW16DirectPackedCodeOffset +
                row_in_page * (Policy::HeadDim / 2) + dim / 2];
  const uint8_t code =
      static_cast<uint8_t>((packed_codes >> (4 * (dim & 1))) & 0x0fu);
  const uint32_t packed_meta = descriptor->k_base;
  uint16_t exponent =
      static_cast<uint16_t>(static_cast<uint8_t>(packed_meta) + code);
  const int start = static_cast<int>((packed_meta >> 8) & 0xffu);
  const int count = static_cast<int>((packed_meta >> 16) & 0xffu);

  if (count != 0) {
    const auto* cache = reinterpret_cast<const uint8_t*>(params.blockmask);
    const uint8_t* page =
        cache + static_cast<int64_t>(descriptor->physical_page) *
                    StaticW16Layout::PageSizeBytes;
    const int local_pos = row_in_page * Policy::HeadDim + dim;
#pragma unroll 1
    for (int entry_idx = count - 1; entry_idx >= 0; --entry_idx) {
      const uint32_t entry =
          load_u32(page, StaticW16Layout::escape_offset(start + entry_idx));
      if (static_cast<int>(entry & 0xffffu) == local_pos) {
        exponent = static_cast<uint16_t>((entry >> 16) & 0xffu);
      }
    }
  }

  const uint16_t low =
      static_cast<uint16_t>((sign_mantissa & 0x7fu) | ((exponent & 1u) << 7));
  const uint16_t high =
      static_cast<uint16_t>((sign_mantissa & 0x80u) | (exponent >> 1));
  return static_cast<uint16_t>(low | (high << 8));
}

__device__ __forceinline__ uint16_t
static_w16_reconstruct_bf16_bits(uint8_t sign_mantissa, uint16_t exponent) {
  const uint16_t low =
      static_cast<uint16_t>((sign_mantissa & 0x7fu) | ((exponent & 1u) << 7));
  const uint16_t high =
      static_cast<uint16_t>((sign_mantissa & 0x80u) | (exponent >> 1));
  return static_cast<uint16_t>(low | (high << 8));
}

static __device__ __noinline__ uint32_t patch_static_w16_direct_k_exponents(
    const uint8_t* cache, int physical_page, int start, int count, int local_0,
    uint32_t packed_exponents) {
  const uint8_t* page = cache + static_cast<int64_t>(physical_page) *
                                    StaticW16Layout::PageSizeBytes;
  const int local_1 = local_0 + 1;
  const int local_8 = local_0 + 8;
  const int local_9 = local_0 + 9;
#pragma unroll 1
  for (int entry_idx = count - 1; entry_idx >= 0; --entry_idx) {
    const uint32_t entry =
        load_u32(page, StaticW16Layout::escape_offset(start + entry_idx));
    const int local_pos = static_cast<int>(entry & 0xffffu);
    const uint32_t exponent = (entry >> 16) & 0xffu;
    int shift = -1;
    if (local_pos == local_0) {
      shift = 0;
    } else if (local_pos == local_1) {
      shift = 8;
    } else if (local_pos == local_8) {
      shift = 16;
    } else if (local_pos == local_9) {
      shift = 24;
    }
    if (shift >= 0) {
      packed_exponents =
          (packed_exponents & ~(0xffu << shift)) | (exponent << shift);
    }
  }
  return packed_exponents;
}

template <typename Params>
__device__ __forceinline__ uint2 load_static_w16_direct_k_bf16x4_bits(
    const Params& params, int row, int dim_pair, int valid_rows) {
  if (row < 0 || row >= valid_rows) {
    return make_uint2(0, 0);
  }
  const int page_in_tile = row / Policy::CodecTokenBlock;
  const int row_in_page = row & (Policy::CodecTokenBlock - 1);
  const StaticW16SharedPageDescriptor* descriptor =
      static_w16_shared_page_descriptors() + page_in_tile;
  const uint8_t* page_slot =
      static_w16_direct_k_smem() + page_in_tile * kStaticW16DirectPageSlotBytes;
  if (descriptor->raw_slot >= 0) {
    const auto* raw_row = reinterpret_cast<const uint16_t*>(page_slot) +
                          row_in_page * Policy::HeadDim;
    return make_uint2(
        *reinterpret_cast<const uint32_t*>(raw_row + dim_pair),
        *reinterpret_cast<const uint32_t*>(raw_row + dim_pair + 8));
  }

  const uint8_t* sign_mantissa_row = page_slot + row_in_page * Policy::HeadDim;
  const uint8_t* packed_code_row = page_slot +
                                   kStaticW16DirectPackedCodeOffset +
                                   row_in_page * (Policy::HeadDim / 2);
  const uint16_t sign_mantissas_01 =
      *reinterpret_cast<const uint16_t*>(sign_mantissa_row + dim_pair);
  const uint16_t sign_mantissas_89 =
      *reinterpret_cast<const uint16_t*>(sign_mantissa_row + dim_pair + 8);
  const uint8_t codes_01 = packed_code_row[dim_pair / 2];
  const uint8_t codes_89 = packed_code_row[dim_pair / 2 + 4];
  const uint32_t packed_meta = descriptor->k_base;
  const uint16_t base = static_cast<uint8_t>(packed_meta);
  uint32_t packed_exponents =
      static_cast<uint32_t>(base + (codes_01 & 0x0fu)) |
      (static_cast<uint32_t>(base + (codes_01 >> 4)) << 8) |
      (static_cast<uint32_t>(base + (codes_89 & 0x0fu)) << 16) |
      (static_cast<uint32_t>(base + (codes_89 >> 4)) << 24);
  const int row_base = row_in_page * Policy::HeadDim;
  const int local_0 = row_base + dim_pair;
  const auto* escape_mask_words = reinterpret_cast<const uint32_t*>(
      page_slot + kStaticW16DirectEscapeMaskOffset);
  const uint32_t wanted_mask = 0x303u << (local_0 & 31);
  if ((escape_mask_words[local_0 / 32] & wanted_mask) != 0) {
    const auto* cache = reinterpret_cast<const uint8_t*>(params.blockmask);
    packed_exponents = patch_static_w16_direct_k_exponents(
        cache, descriptor->physical_page,
        static_cast<int>((packed_meta >> 8) & 0xffu),
        static_cast<int>((packed_meta >> 16) & 0xffu), local_0,
        packed_exponents);
  }

  const uint16_t bf16_0 =
      static_w16_reconstruct_bf16_bits(static_cast<uint8_t>(sign_mantissas_01),
                                       static_cast<uint8_t>(packed_exponents));
  const uint16_t bf16_1 = static_w16_reconstruct_bf16_bits(
      static_cast<uint8_t>(sign_mantissas_01 >> 8),
      static_cast<uint8_t>(packed_exponents >> 8));
  const uint16_t bf16_8 = static_w16_reconstruct_bf16_bits(
      static_cast<uint8_t>(sign_mantissas_89),
      static_cast<uint8_t>(packed_exponents >> 16));
  const uint16_t bf16_9 = static_w16_reconstruct_bf16_bits(
      static_cast<uint8_t>(sign_mantissas_89 >> 8),
      static_cast<uint8_t>(packed_exponents >> 24));
  return make_uint2(
      static_cast<uint32_t>(bf16_0) | (static_cast<uint32_t>(bf16_1) << 16),
      static_cast<uint32_t>(bf16_8) | (static_cast<uint32_t>(bf16_9) << 16));
}

template <typename Params, typename AccTensor, typename AFragment,
          typename ASmemTensor, typename KSmemTensor, typename TiledMma,
          typename TiledCopyA, typename ThrCopyA>
__device__ __forceinline__ void static_w16_direct_k_fragment_gemm(
    const Params& params, int valid_rows, AccTensor& acc, AFragment& tCrA,
    const ASmemTensor& tCsA, const KSmemTensor& sK, TiledMma tiled_mma,
    TiledCopyA smem_tiled_copy_A, ThrCopyA smem_thr_copy_A) {
  CUTE_STATIC_ASSERT_V(cute::size<1>(tCrA) == cute::size<1>(acc));

  auto thr_mma = tiled_mma.get_thread_slice(threadIdx.x);
  auto direct_sK = cute::make_tensor(
      sK.data(),
      cute::make_layout(cute::make_shape(cute::Int<8>{}, cute::Int<16>{}),
                        cute::GenRowMajor{}));
  auto tCrB = thr_mma.partition_fragment_B(direct_sK);
  CUTE_STATIC_ASSERT_V(cute::size<0>(tCrB) == cute::Int<4>{});
  CUTE_STATIC_ASSERT_V(cute::size<1>(tCrB) == cute::Int<2>{});
  CUTE_STATIC_ASSERT_V(cute::size<2>(tCrB) == cute::Int<1>{});
  auto tCrA_copy_view = smem_thr_copy_A.retile_D(tCrA);
  cute::copy(smem_tiled_copy_A, tCsA(_, _, cute::_0{}),
             tCrA_copy_view(_, _, cute::_0{}));

#pragma unroll
  for (int k = 0; k < cute::size<2>(tCrA); ++k) {
    if (k < cute::size<2>(tCrA) - 1) {
      cute::copy(smem_tiled_copy_A, tCsA(_, _, k + 1),
                 tCrA_copy_view(_, _, k + 1));
    }
#pragma unroll
    for (int n = 0; n < cute::size<2>(acc); ++n) {
      const int lane = static_cast<int>(threadIdx.x) & (warpSize - 1);
      const int row = (lane >> 2) + n * 8;
      const int dim_pair = (lane & 3) * 2 + k * 16;
      const uint2 decoded = load_static_w16_direct_k_bf16x4_bits(
          params, row, dim_pair, valid_rows);
      auto fragment = tCrB(_, cute::_0{}, cute::_0{});
      auto* fragment_bits = reinterpret_cast<uint32_t*>(&fragment(0));
      fragment_bits[0] = decoded.x;
      fragment_bits[1] = decoded.y;
      const auto& mma_atom =
          static_cast<const typename TiledMma::Atom&>(tiled_mma);
      cute::gemm(mma_atom, tCrA(_, cute::_0{}, k), fragment,
                 acc(_, cute::_0{}, n));
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
  static constexpr bool DirectFragmentK = false;
  static constexpr int SharedStorageBytes = 0;

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
      fa2::stage_tile_to_fa2_smem_paired_16<IsValue, false>(
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
      fa2::decode_staged_tile_to_fa2_smem_paired_16<false>(
          byte_v2_cache, PageSizeBytes, num_pages, num_raw_slots,
          page_to_raw_slot, block_table, n_block, block_n, valid_rows, dst,
          coords);
    }
  }
};

struct LoaderReuseKvSmemNonsplit : Loader {
  static constexpr bool ReuseKvSmemNonsplit = true;
};

// This variant is selected only when FA2 aliases K and V shared memory. K
// builds one page descriptor cooperatively per 16-thread group and V reuses
// it, while the legacy Loader remains available to the non-reuse schedule.
struct LoaderSharedPageDescriptor : Loader {
  static constexpr bool ReuseKvSmemNonsplit = true;
  static constexpr int SharedStorageBytes = kSharedPageDescriptorBytes<Layout>;

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
      fa2::stage_tile_to_fa2_smem_paired_16<IsValue, true>(
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
      fa2::decode_staged_tile_to_fa2_smem_paired_16<true>(
          byte_v2_cache, PageSizeBytes, num_pages, num_raw_slots,
          page_to_raw_slot, block_table, n_block, block_n, valid_rows, dst,
          coords);
    }
  }
};

// Reader-only SplitZip page variants reuse the V5 envelope and asynchronous
// staging schedule.  Their dense plane contains sign/mantissa bytes rather
// than ByteV2 low bytes, and pooled 2-byte entries carry the true exponent
// rather than a replacement high byte.  The page writer is intentionally
// outside the realtime attention path.
struct SplitZipLoader : Loader {
  static constexpr int PageSizeBytes = SplitZipLayout::PageSizeBytes;

  template <bool IsValue, typename Params, typename DstTensor,
            typename CoordTensor>
  __device__ static __forceinline__ void stage_tile_to_fa2_smem(
      const Params& params, const int* block_table, int kv_head, int n_block,
      int block_n, int valid_rows, DstTensor& dst, const CoordTensor& coords) {
    const auto* splitzip_cache =
        reinterpret_cast<const uint8_t*>(params.blockmask);
    const auto* raw_slot_base =
        reinterpret_cast<const uint8_t*>(IsValue ? params.v_ptr : params.k_ptr);
    const auto* page_to_raw_slot =
        reinterpret_cast<const int*>(params.vnew_ptr);
    const int num_pages = static_cast<int>(params.k_batch_stride);
    const int num_raw_slots = static_cast<int>(params.v_batch_stride);
    if constexpr (kStageMode == 0) {
      fa2::stage_tile_to_fa2_smem_scalar<IsValue, SplitZipLayout>(
          splitzip_cache, raw_slot_base, page_to_raw_slot, PageSizeBytes,
          num_pages, num_raw_slots, block_table, kv_head, n_block, block_n,
          valid_rows, dst, coords);
    } else {
      fa2::stage_tile_to_fa2_smem_paired_16<IsValue, false, SplitZipLayout>(
          splitzip_cache, raw_slot_base, page_to_raw_slot, PageSizeBytes,
          num_pages, num_raw_slots, block_table, kv_head, n_block, block_n,
          valid_rows, dst, coords);
    }
  }

  template <typename Params, typename DstTensor, typename CoordTensor>
  __device__ static __forceinline__ void decode_staged_tile_to_fa2_smem(
      const Params& params, const int* block_table, int n_block, int block_n,
      int valid_rows, DstTensor& dst, const CoordTensor& coords) {
    const auto* splitzip_cache =
        reinterpret_cast<const uint8_t*>(params.blockmask);
    const auto* page_to_raw_slot =
        reinterpret_cast<const int*>(params.vnew_ptr);
    const int num_pages = static_cast<int>(params.k_batch_stride);
    const int num_raw_slots = static_cast<int>(params.v_batch_stride);
    if constexpr (kStageMode == 0) {
      fa2::decode_staged_tile_to_fa2_smem_scalar<true, SplitZipLayout>(
          splitzip_cache, PageSizeBytes, num_pages, num_raw_slots,
          page_to_raw_slot, block_table, n_block, block_n, valid_rows, dst,
          coords);
    } else {
      fa2::decode_staged_tile_to_fa2_smem_paired_16<false, true,
                                                    SplitZipLayout>(
          splitzip_cache, PageSizeBytes, num_pages, num_raw_slots,
          page_to_raw_slot, block_table, n_block, block_n, valid_rows, dst,
          coords);
    }
  }
};

struct SplitZipLoaderReuseKvSmemNonsplit : SplitZipLoader {
  static constexpr bool ReuseKvSmemNonsplit = true;
};

struct SplitZipLoaderSharedPageDescriptor : SplitZipLoader {
  static constexpr bool ReuseKvSmemNonsplit = true;
  static constexpr int SharedStorageBytes =
      kSharedPageDescriptorBytes<SplitZipLayout>;

  template <bool IsValue, typename Params, typename DstTensor,
            typename CoordTensor>
  __device__ static __forceinline__ void stage_tile_to_fa2_smem(
      const Params& params, const int* block_table, int kv_head, int n_block,
      int block_n, int valid_rows, DstTensor& dst, const CoordTensor& coords) {
    const auto* splitzip_cache =
        reinterpret_cast<const uint8_t*>(params.blockmask);
    const auto* raw_slot_base =
        reinterpret_cast<const uint8_t*>(IsValue ? params.v_ptr : params.k_ptr);
    const auto* page_to_raw_slot =
        reinterpret_cast<const int*>(params.vnew_ptr);
    const int num_pages = static_cast<int>(params.k_batch_stride);
    const int num_raw_slots = static_cast<int>(params.v_batch_stride);
    if constexpr (kStageMode == 0) {
      fa2::stage_tile_to_fa2_smem_scalar<IsValue, SplitZipLayout>(
          splitzip_cache, raw_slot_base, page_to_raw_slot, PageSizeBytes,
          num_pages, num_raw_slots, block_table, kv_head, n_block, block_n,
          valid_rows, dst, coords);
    } else {
      fa2::stage_tile_to_fa2_smem_paired_16<IsValue, true, SplitZipLayout>(
          splitzip_cache, raw_slot_base, page_to_raw_slot, PageSizeBytes,
          num_pages, num_raw_slots, block_table, kv_head, n_block, block_n,
          valid_rows, dst, coords);
    }
  }

  template <typename Params, typename DstTensor, typename CoordTensor>
  __device__ static __forceinline__ void decode_staged_tile_to_fa2_smem(
      const Params& params, const int* block_table, int n_block, int block_n,
      int valid_rows, DstTensor& dst, const CoordTensor& coords) {
    const auto* splitzip_cache =
        reinterpret_cast<const uint8_t*>(params.blockmask);
    const auto* page_to_raw_slot =
        reinterpret_cast<const int*>(params.vnew_ptr);
    const int num_pages = static_cast<int>(params.k_batch_stride);
    const int num_raw_slots = static_cast<int>(params.v_batch_stride);
    if constexpr (kStageMode == 0) {
      fa2::decode_staged_tile_to_fa2_smem_scalar<true, SplitZipLayout>(
          splitzip_cache, PageSizeBytes, num_pages, num_raw_slots,
          page_to_raw_slot, block_table, n_block, block_n, valid_rows, dst,
          coords);
    } else {
      fa2::decode_staged_tile_to_fa2_smem_paired_16<true, true, SplitZipLayout>(
          splitzip_cache, PageSizeBytes, num_pages, num_raw_slots,
          page_to_raw_slot, block_table, n_block, block_n, valid_rows, dst,
          coords);
    }
  }
};

// Reader-only arithmetic Static-W16 variants.  They preserve the original
// FA2 mainloop and only replace its global-to-shared K/V policy.  Sealed
// compact pages use the fixed 49,792-byte wire; overflow and partial pages
// must be mapped to an authoritative 65,536-byte raw sidecar.
template <bool CanonicalMetadata>
struct StaticW16LoaderImpl : Loader {
  static constexpr int PageSizeBytes = StaticW16Layout::PageSizeBytes;

  template <bool IsValue, typename Params, typename DstTensor,
            typename CoordTensor>
  __device__ static __forceinline__ void stage_tile_to_fa2_smem(
      const Params& params, const int* block_table, int kv_head, int n_block,
      int block_n, int valid_rows, DstTensor& dst, const CoordTensor& coords) {
    const auto* cache = reinterpret_cast<const uint8_t*>(params.blockmask);
    const auto* raw_slot_base =
        reinterpret_cast<const uint8_t*>(IsValue ? params.v_ptr : params.k_ptr);
    const auto* page_to_raw_slot =
        reinterpret_cast<const int*>(params.vnew_ptr);
    const int num_pages = static_cast<int>(params.k_batch_stride);
    const int num_raw_slots = static_cast<int>(params.v_batch_stride);
    fa2::stage_static_w16_tile_to_fa2_smem<IsValue, false, CanonicalMetadata>(
        cache, raw_slot_base, page_to_raw_slot, PageSizeBytes, num_pages,
        num_raw_slots, block_table, kv_head, n_block, block_n, valid_rows, dst,
        coords);
  }

  template <typename Params, typename DstTensor, typename CoordTensor>
  __device__ static __forceinline__ void decode_staged_tile_to_fa2_smem(
      const Params& params, const int* block_table, int n_block, int block_n,
      int valid_rows, DstTensor& dst, const CoordTensor& coords) {
    const auto* cache = reinterpret_cast<const uint8_t*>(params.blockmask);
    const auto* page_to_raw_slot =
        reinterpret_cast<const int*>(params.vnew_ptr);
    const int num_pages = static_cast<int>(params.k_batch_stride);
    const int num_raw_slots = static_cast<int>(params.v_batch_stride);
    fa2::decode_static_w16_tile_to_fa2_smem<false>(
        cache, PageSizeBytes, num_pages, num_raw_slots, page_to_raw_slot,
        block_table, n_block, block_n, valid_rows, dst, coords);
  }
};

struct StaticW16Loader : StaticW16LoaderImpl<false> {};

struct StaticW16CanonicalLoader : StaticW16LoaderImpl<true> {};

struct StaticW16LoaderReuseKvSmemNonsplit : StaticW16Loader {
  static constexpr bool ReuseKvSmemNonsplit = true;
};

struct StaticW16CanonicalLoaderReuseKvSmemNonsplit : StaticW16CanonicalLoader {
  static constexpr bool ReuseKvSmemNonsplit = true;
};

template <bool CanonicalMetadata, int DescriptorSmemOffset = kFa2ReuseSmemBytes>
struct StaticW16LoaderSharedPageDescriptorImpl
    : StaticW16LoaderImpl<CanonicalMetadata> {
  static constexpr int PageSizeBytes = StaticW16Layout::PageSizeBytes;
  static constexpr bool ReuseKvSmemNonsplit = true;
  static constexpr bool DirectFragmentK = kStaticW16KDirectFragment;
  static constexpr int SharedStorageBytes =
      kStaticW16SharedPageDescriptorBytes +
      (kStaticW16EscapePatchMode == 3 ? kStaticW16PrefetchedEscapeBytes : 0);

  template <bool IsValue, typename Params, typename DstTensor,
            typename CoordTensor>
  __device__ static __forceinline__ void stage_tile_to_fa2_smem(
      const Params& params, const int* block_table, int kv_head, int n_block,
      int block_n, int valid_rows, DstTensor& dst, const CoordTensor& coords) {
    const auto* cache = reinterpret_cast<const uint8_t*>(params.blockmask);
    const auto* raw_slot_base =
        reinterpret_cast<const uint8_t*>(IsValue ? params.v_ptr : params.k_ptr);
    const auto* page_to_raw_slot =
        reinterpret_cast<const int*>(params.vnew_ptr);
    const int num_pages = static_cast<int>(params.k_batch_stride);
    const int num_raw_slots = static_cast<int>(params.v_batch_stride);
    if constexpr (DirectFragmentK && !IsValue) {
      fa2::stage_static_w16_k_tile_direct_fragment<CanonicalMetadata>(
          cache, raw_slot_base, page_to_raw_slot, PageSizeBytes, num_pages,
          num_raw_slots, block_table, kv_head, n_block, block_n, valid_rows,
          dst, coords);
    } else {
      fa2::stage_static_w16_tile_to_fa2_smem<IsValue, true, CanonicalMetadata,
                                             DescriptorSmemOffset>(
          cache, raw_slot_base, page_to_raw_slot, PageSizeBytes, num_pages,
          num_raw_slots, block_table, kv_head, n_block, block_n, valid_rows,
          dst, coords);
    }
  }

  template <typename Params, typename DstTensor, typename CoordTensor>
  __device__ static __forceinline__ void decode_staged_tile_to_fa2_smem(
      const Params& params, const int* block_table, int n_block, int block_n,
      int valid_rows, DstTensor& dst, const CoordTensor& coords) {
    const auto* cache = reinterpret_cast<const uint8_t*>(params.blockmask);
    const auto* page_to_raw_slot =
        reinterpret_cast<const int*>(params.vnew_ptr);
    const int num_pages = static_cast<int>(params.k_batch_stride);
    const int num_raw_slots = static_cast<int>(params.v_batch_stride);
    fa2::decode_static_w16_tile_to_fa2_smem<true, DescriptorSmemOffset>(
        cache, PageSizeBytes, num_pages, num_raw_slots, page_to_raw_slot,
        block_table, n_block, block_n, valid_rows, dst, coords);
  }

  template <typename Params, typename AccTensor, typename AFragment,
            typename ASmemTensor, typename KSmemTensor, typename TiledMma,
            typename TiledCopyA, typename ThrCopyA>
  __device__ static __forceinline__ void direct_fragment_k_gemm(
      const Params& params, int valid_rows, AccTensor& acc, AFragment& tCrA,
      const ASmemTensor& tCsA, const KSmemTensor& sK, TiledMma tiled_mma,
      TiledCopyA smem_tiled_copy_A, ThrCopyA smem_thr_copy_A) {
    static_assert(DirectFragmentK);
    fa2::static_w16_direct_k_fragment_gemm(params, valid_rows, acc, tCrA, tCsA,
                                           sK, tiled_mma, smem_tiled_copy_A,
                                           smem_thr_copy_A);
  }
};

struct StaticW16LoaderSharedPageDescriptor
    : StaticW16LoaderSharedPageDescriptorImpl<false> {};

struct StaticW16CanonicalLoaderSharedPageDescriptor
    : StaticW16LoaderSharedPageDescriptorImpl<true> {};

// Profile-only one-CTA producer/consumer loader.  Separate K and V tiles use
// the full 80 KiB FA2 shared-memory layout, so descriptors must live after V
// rather than at the production aliased-KV 48 KiB boundary.
struct StaticW16CanonicalLoaderOneCtaOverlap
    : StaticW16LoaderSharedPageDescriptorImpl<true, kFa2SeparateKvSmemBytes> {
  static constexpr bool ReuseKvSmem = false;
  static constexpr bool ReuseKvSmemNonsplit = false;
  static constexpr bool DirectFragmentK = false;

  template <bool IsValue, typename Params, typename DstTensor,
            typename CoordTensor>
  __device__ static __forceinline__ void decode_staged_tile_to_fa2_smem_7p1c(
      const Params& params, const int* block_table, int n_block, int block_n,
      int valid_rows, DstTensor& dst, const CoordTensor& coords,
      int logical_tidx, bool helper_producer) {
    static_assert(
        kStaticW16EscapePatchMode == 0,
        "the 7-producer profile currently requires owner-local escape scans");
    // Warp 4 remains the logical consumer warp 0, so helper warps 5--7 keep
    // logical producer indices 32--127 and can reuse the same CUTE slices in
    // the output epilogue.  Producer warp 0 consequently remains unsplit.
    const bool has_helper = logical_tidx >= warpSize;
    const int m_begin = helper_producer ? 4 : 0;
    const int m_end = helper_producer ? 8 : (has_helper ? 4 : 8);
    const auto* cache = reinterpret_cast<const uint8_t*>(params.blockmask);
    const auto* page_to_raw_slot =
        reinterpret_cast<const int*>(params.vnew_ptr);
    const int num_pages = static_cast<int>(params.k_batch_stride);
    const int num_raw_slots = static_cast<int>(params.v_batch_stride);
    const StaticW16SharedPageDescriptor* descriptor =
        fa2::static_w16_shared_page_descriptors<kFa2SeparateKvSmemBytes>() +
        logical_tidx / Policy::CodecDimBlock;
    const uint32_t packed_meta_override =
        IsValue ? descriptor->v_base : descriptor->k_base;
    fa2::decode_static_w16_tile_to_fa2_smem<true, kFa2SeparateKvSmemBytes>(
        cache, PageSizeBytes, num_pages, num_raw_slots, page_to_raw_slot,
        block_table, n_block, block_n, valid_rows, dst, coords, logical_tidx,
        m_begin, m_end, /*use_packed_meta_override=*/true,
        packed_meta_override);
  }
};

}  // namespace vllm::byte_v2::fa2
