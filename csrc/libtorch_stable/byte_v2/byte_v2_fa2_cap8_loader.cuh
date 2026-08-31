// Copyright (c) 2026, ByteV2 contributors.
//
// Profile-only Static-W16 cap8 loader.  This header is included only by the
// dedicated cap8 CUDA translation units so production ByteV2/SplitZip/
// shared128 template instantiations keep their original compilation unit.

#pragma once

#include "byte_v2_fa2_loader.cuh"

namespace vllm::byte_v2::fa2 {

// Historical Static-W16 cap8 interpretation of the same fixed 49,792-byte
// page.  Only the directory and exact-patch overlay differ from shared128:
// the first 64 header bytes are 16 untagged uint32 counts, and every K/V-head
// chunk owns eight fixed uint32 patch slots.  This layout exists solely for a
// same-binary reader A/B; production writers and routes continue to use
// StaticW16Layout above.
struct StaticW16Cap8Layout {
  static constexpr int NumKvHeadsValue = StaticW16Layout::NumKvHeadsValue;
  static constexpr int PageSizeBytes = StaticW16Layout::PageSizeBytes;
  static constexpr int ValuesPerChunk = StaticW16Layout::ValuesPerChunk;
  static constexpr int SignMantissaBaseBytes =
      StaticW16Layout::SignMantissaBaseBytes;
  static constexpr int PackedCodeBaseBytes =
      StaticW16Layout::PackedCodeBaseBytes;
  static constexpr int HeaderBaseBytes = StaticW16Layout::HeaderBaseBytes;
  static constexpr int EscapeBaseBytes = StaticW16Layout::EscapeBaseBytes;
  static constexpr int EscapesPerChunk = 8;
  static constexpr int EscapeEntryBytes = StaticW16Layout::EscapeEntryBytes;
  static constexpr int StatusOffset = StaticW16Layout::StatusOffset;
  static constexpr int KBaseOffset = StaticW16Layout::KBaseOffset;
  static constexpr int VBaseOffset = StaticW16Layout::VBaseOffset;

  __host__ __device__ static constexpr int chunk_index(bool is_value,
                                                       int kv_head) {
    return (is_value ? NumKvHeadsValue : 0) + kv_head;
  }

  __host__ __device__ static constexpr int count_offset(bool is_value,
                                                        int kv_head) {
    return HeaderBaseBytes +
           chunk_index(is_value, kv_head) * static_cast<int>(sizeof(uint32_t));
  }

  __host__ __device__ static constexpr int escape_offset(bool is_value,
                                                         int kv_head,
                                                         int entry) {
    return escape_offset(chunk_index(is_value, kv_head), entry);
  }

  __host__ __device__ static constexpr int escape_offset(int chunk, int entry) {
    return EscapeBaseBytes +
           (chunk * EscapesPerChunk + entry) * EscapeEntryBytes;
  }
};

static_assert(StaticW16Cap8Layout::EscapeBaseBytes +
                  2 * StaticW16Cap8Layout::NumKvHeadsValue *
                      StaticW16Cap8Layout::EscapesPerChunk *
                      StaticW16Cap8Layout::EscapeEntryBytes ==
              StaticW16Cap8Layout::PageSizeBytes);

template <bool IsValue, bool UseSharedPageDescriptor, typename DstTensor,
          typename CoordTensor>
__device__ __forceinline__ void stage_static_w16_cap8_tile_to_fa2_smem(
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
      page_stride_bytes != StaticW16Cap8Layout::PageSizeBytes ||
      num_pages <= 0 || kv_head < 0 ||
      kv_head >= StaticW16Cap8Layout::NumKvHeadsValue || n_block < 0 ||
      valid_rows < 0 || valid_rows > block_n || block_n != 128 ||
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
    descriptor = static_w16_shared_page_descriptors() + page_in_tile;
    if (page_valid) {
      const uint8_t* page =
          static_w16_cache +
          static_cast<int64_t>(physical_page) * page_stride_bytes;
      if (lane_in_page == 0) {
        descriptor->physical_page = physical_page;
        descriptor->raw_slot = raw_slot;
        descriptor->status =
            load_u32_early(page, StaticW16Cap8Layout::StatusOffset);
        descriptor->reserved = 0;
      } else if (lane_in_page == 1) {
        descriptor->k_base = load_u32(page, StaticW16Cap8Layout::KBaseOffset);
      } else if (lane_in_page == 2) {
        descriptor->v_base = load_u32(page, StaticW16Cap8Layout::VBaseOffset);
      } else if (lane_in_page == 3) {
        descriptor->k_range =
            load_u32(page, StaticW16Cap8Layout::count_offset(false, kv_head));
      } else if (lane_in_page == 4) {
        descriptor->v_range =
            load_u32(page, StaticW16Cap8Layout::count_offset(true, kv_head));
      }
    }
    __syncwarp();
  }

  if (row0 >= valid_rows) {
    return;
  }

  uint32_t status = 0;
  uint32_t base = 0;
  uint32_t count = 0;
  if constexpr (UseSharedPageDescriptor) {
    descriptor = static_w16_shared_page_descriptors() + page_in_tile;
    if constexpr (IsValue) {
      physical_page = descriptor->physical_page;
      raw_slot = descriptor->raw_slot;
    }
    if (physical_page < 0 || physical_page >= num_pages || raw_slot < -1 ||
        raw_slot >= num_raw_slots) {
      __trap();
    }
    status = descriptor->status;
    base = IsValue ? descriptor->v_base : descriptor->k_base;
    count = IsValue ? descriptor->v_range : descriptor->k_range;
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
    status = load_u32_early(page, StaticW16Cap8Layout::StatusOffset);
    base = load_u32(page, IsValue ? StaticW16Cap8Layout::VBaseOffset
                                  : StaticW16Cap8Layout::KBaseOffset);
    count = load_u32(page, StaticW16Cap8Layout::count_offset(IsValue, kv_head));
  }

  const bool page_is_raw = raw_slot >= 0;
  if ((status & ~uint32_t{1}) != 0 || ((status & 1u) != 0) != page_is_raw) {
    __trap();
  }
  if (stage_raw_slot_to_fa2_smem(raw_side_base, num_raw_slots, raw_slot,
                                 kv_head, row0, valid_rows, dst)) {
    return;
  }
  if (base > 240 || count > StaticW16Cap8Layout::EscapesPerChunk) {
    __trap();
  }

  const int side = tidx & 1;
  const int dst_k = side;
  const int dim_tile = 4 * side + ((tidx & 7) >> 1);
  const int row_in_page0 = kRowsPerThread * ((tidx >> 3) & 1);
  const int chunk = StaticW16Cap8Layout::chunk_index(IsValue, kv_head);
  const int sign_mantissa_base = StaticW16Cap8Layout::SignMantissaBaseBytes +
                                 chunk * StaticW16Cap8Layout::ValuesPerChunk;
  const int packed_code_base =
      StaticW16Cap8Layout::PackedCodeBaseBytes +
      chunk * (StaticW16Cap8Layout::ValuesPerChunk / 2);
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

  // Carry the fixed overlay owner with the per-thread metadata because the
  // FA2 decode callback intentionally has no IsValue/kv_head arguments.
  const uint32_t packed_meta =
      base | (count << 8) | (static_cast<uint32_t>(chunk) << 16);
  constexpr int kMetadataRow = 0;
  constexpr int kMetadataOffset = 8;
  auto* metadata_slot = paired_vector_slot(dst, kMetadataRow, dst_k, side);
  *reinterpret_cast<uint32_t*>(metadata_slot + kMetadataOffset) = packed_meta;
  if constexpr (!UseSharedPageDescriptor) {
    *reinterpret_cast<int*>(metadata_slot + kMetadataOffset + 4) =
        physical_page;
  }
}

template <bool UseSharedPageDescriptor, typename DstTensor,
          typename CoordTensor>
__device__ __forceinline__ void decode_static_w16_cap8_tile_to_fa2_smem(
    const uint8_t* static_w16_cache, int64_t page_stride_bytes, int num_pages,
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

  const StaticW16SharedPageDescriptor* descriptor = nullptr;
  if constexpr (UseSharedPageDescriptor) {
    descriptor =
        static_w16_shared_page_descriptors() + tidx / Policy::CodecDimBlock;
    if (descriptor->physical_page < 0 ||
        descriptor->physical_page >= num_pages || descriptor->raw_slot < -1 ||
        descriptor->raw_slot >= num_raw_slots) {
      __trap();
    }
    if (descriptor->raw_slot >= 0) {
      return;
    }
  } else if (thread_page_is_raw(num_pages, num_raw_slots, page_to_raw_slot,
                                block_table, n_block, block_n)) {
    return;
  }

  constexpr int kMetadataRow = 0;
  constexpr int kMetadataOffset = 8;
  auto* metadata_slot = paired_vector_slot(dst, kMetadataRow, dst_k, side);
  const uint32_t packed_meta =
      *reinterpret_cast<const uint32_t*>(metadata_slot + kMetadataOffset);
  const int physical_page =
      UseSharedPageDescriptor
          ? descriptor->physical_page
          : *reinterpret_cast<const int*>(metadata_slot + kMetadataOffset + 4);
  const uint8_t base = static_cast<uint8_t>(packed_meta);
  const int count = static_cast<int>((packed_meta >> 8) & 0xffu);
  const int chunk = static_cast<int>((packed_meta >> 16) & 0xffu);
  if (count > StaticW16Cap8Layout::EscapesPerChunk || chunk < 0 ||
      chunk >= 2 * StaticW16Cap8Layout::NumKvHeadsValue) {
    __trap();
  }

#pragma unroll 1
  for (int m = 0; m < kRowsPerThread; ++m) {
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

  if (count == 0) {
    return;
  }
  const uint8_t* page = static_w16_cache +
                        static_cast<int64_t>(physical_page) * page_stride_bytes;
  const int row_in_page0 = kRowsPerThread * ((tidx >> 3) & 1);
  const int dim_tile = 4 * side + ((tidx & 7) >> 1);
  const int dim_base = dim_tile * Policy::CodecDimBlock;

#pragma unroll 1
  for (int entry_idx = count - 1; entry_idx >= 0; --entry_idx) {
    const uint32_t entry =
        load_u32(page, StaticW16Cap8Layout::escape_offset(chunk, entry_idx));
    const int local_pos = static_cast<int>(entry & 0xffffu);
    if (local_pos < 0 || local_pos >= StaticW16Cap8Layout::ValuesPerChunk) {
      __trap();
    }
    const int outlier_row = local_pos / Policy::HeadDim;
    const int outlier_dim = local_pos % Policy::HeadDim;
    const int m = outlier_row - row_in_page0;
    const int v = outlier_dim - dim_base;
    if (m >= 0 && m < kRowsPerThread && v >= 0 && v < Policy::CodecDimBlock &&
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

// Profile-only cap8 readers.  They intentionally live beside, rather than
// parameterize, the production shared128 readers so the production template
// instantiation does not acquire a runtime directory/overlay branch.
struct StaticW16Cap8Loader : Loader {
  static constexpr int PageSizeBytes = StaticW16Cap8Layout::PageSizeBytes;

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
    fa2::stage_static_w16_cap8_tile_to_fa2_smem<IsValue, false>(
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
    fa2::decode_static_w16_cap8_tile_to_fa2_smem<false>(
        cache, PageSizeBytes, num_pages, num_raw_slots, page_to_raw_slot,
        block_table, n_block, block_n, valid_rows, dst, coords);
  }
};

struct StaticW16Cap8LoaderReuseKvSmemNonsplit : StaticW16Cap8Loader {
  static constexpr bool ReuseKvSmemNonsplit = true;
};

struct StaticW16Cap8LoaderSharedPageDescriptor : StaticW16Cap8Loader {
  static constexpr bool ReuseKvSmemNonsplit = true;
  static constexpr int SharedStorageBytes = kStaticW16SharedPageDescriptorBytes;

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
    fa2::stage_static_w16_cap8_tile_to_fa2_smem<IsValue, true>(
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
    fa2::decode_static_w16_cap8_tile_to_fa2_smem<true>(
        cache, PageSizeBytes, num_pages, num_raw_slots, page_to_raw_slot,
        block_table, n_block, block_n, valid_rows, dst, coords);
  }
};

}  // namespace vllm::byte_v2::fa2
