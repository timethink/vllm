#include "byte_v2_layout.cuh"

#include "../torch_utils.h"

#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>

#include <cuda_runtime.h>

#include <cstdint>
#include <limits>

namespace {

using RawLayout = vllm::byte_v2::ByteV2RawStagingLayout<>;

struct StaticW16Layout {
  static constexpr int kNumKvHeads = 8;
  static constexpr int kBlockSize = 16;
  static constexpr int kHeadDim = 128;
  static constexpr int kChunksPerPage = 16;
  static constexpr int kValuesPerChunk = kBlockSize * kHeadDim;
  static constexpr int kPairsPerChunk = kValuesPerChunk / 2;
  static constexpr int kSignMantissaOffset = 0;
  static constexpr int kCodeOffset = 32768;
  static constexpr int kHeaderOffset = 49152;
  static constexpr int kEscapeOffset = 49280;
  static constexpr int kPageBytes = 49792;
  static constexpr int kRawPageBytes = 65536;
  static constexpr int kEscapeCapacity = 128;
  static constexpr int kQ1Threads = 1024;
  static constexpr int kStatusOffset = kHeaderOffset + 64;
  static constexpr int kKBaseOffset = kHeaderOffset + 68;
  static constexpr int kVBaseOffset = kHeaderOffset + 72;
  static constexpr int kValidRowsOffset = kHeaderOffset + 76;
  static constexpr int kPageEscapeCountOffset = kHeaderOffset + 80;
  static constexpr int kRawFallbackStatus = 1;
  static constexpr int kCanonicalMetadataStatus = 2;
  static constexpr int kCanonicalRawFallbackStatus =
      kCanonicalMetadataStatus | kRawFallbackStatus;
  static constexpr int kLockedRawSlot = -2;

  __device__ static int range_offset(int chunk) {
    return kHeaderOffset + chunk * static_cast<int>(sizeof(int32_t));
  }

  __device__ static int escape_offset(int index) {
    return kEscapeOffset + index * static_cast<int>(sizeof(int32_t));
  }

  __device__ static uint32_t pack_metadata(int base, int start, int count) {
    const uint32_t encoded_base =
        static_cast<uint32_t>(base < 0 ? 0 : (base > 255 ? 255 : base));
    const uint32_t encoded_start =
        static_cast<uint32_t>(start < 0 ? 0 : (start > 255 ? 255 : start));
    const uint32_t encoded_count =
        static_cast<uint32_t>(count < 0 ? 0 : (count > 255 ? 255 : count));
    return encoded_base | (encoded_start << 8) | (encoded_count << 16);
  }

  __device__ static int metadata_base(uint32_t metadata) {
    return static_cast<int>(metadata & 0xffu);
  }

  __device__ static int metadata_start(uint32_t metadata) {
    return static_cast<int>((metadata >> 8) & 0xffu);
  }

  __device__ static int metadata_count(uint32_t metadata) {
    return static_cast<int>((metadata >> 16) & 0xffu);
  }

  __device__ static bool metadata_is_canonical(uint32_t metadata) {
    return (metadata >> 24) == 0;
  }
};

static_assert(RawLayout::SlotSizeBytes == StaticW16Layout::kRawPageBytes);

bool pointer_is_aligned(const void* pointer, std::size_t alignment) {
  return reinterpret_cast<std::uintptr_t>(pointer) % alignment == 0;
}

__device__ __forceinline__ uint16_t load_u16(const uint8_t* ptr,
                                             int64_t offset) {
  return static_cast<uint16_t>(ptr[offset]) |
         static_cast<uint16_t>(ptr[offset + 1]) << 8;
}

__device__ __forceinline__ void store_u16(uint8_t* ptr, int64_t offset,
                                          uint16_t value) {
  ptr[offset] = static_cast<uint8_t>(value);
  ptr[offset + 1] = static_cast<uint8_t>(value >> 8);
}

__device__ __forceinline__ int32_t load_i32(const uint8_t* ptr, int offset) {
  return *reinterpret_cast<const int32_t*>(ptr + offset);
}

__device__ __forceinline__ void store_i32(uint8_t* ptr, int offset,
                                          int32_t value) {
  *reinterpret_cast<int32_t*>(ptr + offset) = value;
}

__device__ __forceinline__ int64_t raw_offset(int chunk, int local_pos) {
  const int side = chunk / StaticW16Layout::kNumKvHeads;
  const int head = chunk % StaticW16Layout::kNumKvHeads;
  const int row = local_pos / StaticW16Layout::kHeadDim;
  const int dim = local_pos % StaticW16Layout::kHeadDim;
  return side == 0 ? RawLayout::key_offset(head, row, dim)
                   : RawLayout::value_offset(head, row, dim);
}

__device__ __forceinline__ int64_t raw_chunk_offset(int chunk) {
  const int side = chunk / StaticW16Layout::kNumKvHeads;
  const int head = chunk % StaticW16Layout::kNumKvHeads;
  return side == 0 ? RawLayout::key_offset(head, 0, 0)
                   : RawLayout::value_offset(head, 0, 0);
}

__device__ __forceinline__ void fail_closed(int32_t* fatal) {
  if (fatal != nullptr) {
    atomicExch(fatal, 1);
  }
  __trap();
}

__device__ int32_t pop_raw_slot(const int32_t* free_slots, int32_t* free_count,
                                int32_t num_slots, int32_t* fatal) {
  int32_t observed = atomicAdd(free_count, 0);
  while (observed > 0) {
    if (observed > num_slots) {
      fail_closed(fatal);
      return -1;
    }
    const int32_t prior = atomicCAS(free_count, observed, observed - 1);
    if (prior == observed) {
      const int32_t slot = free_slots[observed - 1];
      if (slot < 0 || slot >= num_slots) {
        fail_closed(fatal);
        return -1;
      }
      return slot;
    }
    observed = prior;
  }
  fail_closed(fatal);
  return -1;
}

__device__ void push_raw_slot(int32_t slot, int32_t* free_slots,
                              int32_t* free_count, int32_t num_slots,
                              int32_t* fatal) {
  const int32_t index = atomicAdd(free_count, 1);
  if (index < 0 || index >= num_slots || slot < 0 || slot >= num_slots) {
    fail_closed(fatal);
    return;
  }
  free_slots[index] = slot;
  __threadfence();
}

__device__ void pack_chunk(const uint8_t* raw_page, uint8_t* compact_page,
                           int chunk, int base, int* shared_count,
                           int* shared_start, uint32_t* shared_records) {
  if (threadIdx.x == 0) {
    *shared_count = 0;
    *shared_start = 0;
  }
  __syncthreads();

  for (int pair = threadIdx.x; pair < StaticW16Layout::kPairsPerChunk;
       pair += blockDim.x) {
    const int pos0 = pair * 2;
    const int pos1 = pos0 + 1;
    const uint16_t bits0 = load_u16(raw_page, raw_offset(chunk, pos0));
    const uint16_t bits1 = load_u16(raw_page, raw_offset(chunk, pos1));
    const int exp0 = (bits0 >> 7) & 0xff;
    const int exp1 = (bits1 >> 7) & 0xff;
    const int delta0 = exp0 - base;
    const int delta1 = exp1 - base;
    const bool escape0 = delta0 < 0 || delta0 >= 16;
    const bool escape1 = delta1 < 0 || delta1 >= 16;
    const uint8_t sm0 =
        static_cast<uint8_t>(((bits0 >> 8) & 0x80) | (bits0 & 0x7f));
    const uint8_t sm1 =
        static_cast<uint8_t>(((bits1 >> 8) & 0x80) | (bits1 & 0x7f));
    compact_page[StaticW16Layout::kSignMantissaOffset +
                 chunk * StaticW16Layout::kValuesPerChunk + pos0] = sm0;
    compact_page[StaticW16Layout::kSignMantissaOffset +
                 chunk * StaticW16Layout::kValuesPerChunk + pos1] = sm1;
    compact_page[StaticW16Layout::kCodeOffset +
                 chunk * StaticW16Layout::kPairsPerChunk + pair] =
        static_cast<uint8_t>((delta0 & 0x0f) | ((delta1 & 0x0f) << 4));

    if (escape0) {
      const int index = atomicAdd(shared_count, 1);
      if (index < StaticW16Layout::kEscapeCapacity) {
        shared_records[index] = static_cast<uint32_t>((exp0 << 16) | pos0);
      }
    }
    if (escape1) {
      const int index = atomicAdd(shared_count, 1);
      if (index < StaticW16Layout::kEscapeCapacity) {
        shared_records[index] = static_cast<uint32_t>((exp1 << 16) | pos1);
      }
    }
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    *shared_start =
        atomicAdd(reinterpret_cast<int32_t*>(
                      compact_page + StaticW16Layout::kPageEscapeCountOffset),
                  *shared_count);
    store_i32(compact_page, StaticW16Layout::range_offset(chunk),
              static_cast<int32_t>(StaticW16Layout::pack_metadata(
                  base, *shared_start, *shared_count)));
  }
  __syncthreads();
  const int available = *shared_start < StaticW16Layout::kEscapeCapacity
                            ? StaticW16Layout::kEscapeCapacity - *shared_start
                            : 0;
  const int stored = *shared_count < available ? *shared_count : available;
  for (int index = threadIdx.x; index < stored; index += blockDim.x) {
    store_i32(compact_page,
              StaticW16Layout::escape_offset(*shared_start + index),
              static_cast<int32_t>(shared_records[index]));
  }
  __syncthreads();
}

__device__ int pack_page_chunks_deterministic(const uint8_t* raw_page,
                                              uint8_t* compact_page, int k_base,
                                              int v_base) {
  constexpr unsigned int kFullWarpMask = 0xffffffffu;
  constexpr int kWarpSize = 32;
  constexpr int kNumWarps = StaticW16Layout::kQ1Threads / kWarpSize;
  constexpr int kWarpsPerChunk = kNumWarps / StaticW16Layout::kChunksPerPage;
  constexpr int kPairsPerPartition =
      StaticW16Layout::kPairsPerChunk / kWarpsPerChunk;
  static_assert(StaticW16Layout::kQ1Threads == 512 ||
                StaticW16Layout::kQ1Threads == 1024);
  static_assert(StaticW16Layout::kPairsPerChunk % kWarpsPerChunk == 0);
  __shared__ int shared_partition_counts[kNumWarps];
  __shared__ int shared_starts[StaticW16Layout::kChunksPerPage];
  __shared__ uint32_t
      shared_records[kNumWarps * StaticW16Layout::kEscapeCapacity];
  __shared__ int shared_total;

  const int lane = threadIdx.x % kWarpSize;
  const int warp = threadIdx.x / kWarpSize;
  const int chunk = warp / kWarpsPerChunk;
  const int partition = warp % kWarpsPerChunk;
  const int pair_begin = partition * kPairsPerPartition;
  const int pair_end = pair_begin + kPairsPerPartition;

  // Pass one writes the dense payload and counts escapes independently for
  // each chunk.  Contiguous warp partitions preserve local-position order
  // when a 1024-thread experiment assigns two warps to one chunk.
  const int base = chunk < StaticW16Layout::kNumKvHeads ? k_base : v_base;
  const auto* raw_pairs =
      reinterpret_cast<const uint32_t*>(raw_page + raw_chunk_offset(chunk));
  auto* sign_mantissa = compact_page + StaticW16Layout::kSignMantissaOffset +
                        chunk * StaticW16Layout::kValuesPerChunk;
  auto* sign_mantissa_pairs = reinterpret_cast<uint16_t*>(sign_mantissa);
  auto* codes = compact_page + StaticW16Layout::kCodeOffset +
                chunk * StaticW16Layout::kPairsPerChunk;
  const unsigned int lower_lanes =
      lane == 0 ? 0u : (static_cast<unsigned int>(1) << lane) - 1u;
  int partition_count = 0;
  for (int pair = pair_begin + lane; pair < pair_end; pair += kWarpSize) {
    const int pos0 = pair * 2;
    const int pos1 = pos0 + 1;
    const uint32_t pair_bits = raw_pairs[pair];
    const uint16_t bits0 = static_cast<uint16_t>(pair_bits);
    const uint16_t bits1 = static_cast<uint16_t>(pair_bits >> 16);
    const int exp0 = (bits0 >> 7) & 0xff;
    const int exp1 = (bits1 >> 7) & 0xff;
    const int delta0 = exp0 - base;
    const int delta1 = exp1 - base;
    const bool escape0 = delta0 < 0 || delta0 >= 16;
    const bool escape1 = delta1 < 0 || delta1 >= 16;
    const uint16_t sm0 = ((bits0 >> 8) & 0x80) | (bits0 & 0x7f);
    const uint16_t sm1 = ((bits1 >> 8) & 0x80) | (bits1 & 0x7f);
    sign_mantissa_pairs[pair] = sm0 | (sm1 << 8);
    codes[pair] =
        static_cast<uint8_t>((delta0 & 0x0f) | ((delta1 & 0x0f) << 4));
    const unsigned int escape0_lanes = __ballot_sync(kFullWarpMask, escape0);
    const unsigned int escape1_lanes = __ballot_sync(kFullWarpMask, escape1);
    const int preceding = __popc(escape0_lanes & lower_lanes) +
                          __popc(escape1_lanes & lower_lanes);
    const int escape0_index = partition_count + preceding;
    const int escape1_index = escape0_index + static_cast<int>(escape0);
    if (escape0 && escape0_index < StaticW16Layout::kEscapeCapacity) {
      shared_records[warp * StaticW16Layout::kEscapeCapacity + escape0_index] =
          static_cast<uint32_t>((exp0 << 16) | pos0);
    }
    if (escape1 && escape1_index < StaticW16Layout::kEscapeCapacity) {
      shared_records[warp * StaticW16Layout::kEscapeCapacity + escape1_index] =
          static_cast<uint32_t>((exp1 << 16) | pos1);
    }
    partition_count += __popc(escape0_lanes) + __popc(escape1_lanes);
  }
  if (lane == 0) {
    shared_partition_counts[warp] = partition_count;
  }
  __syncthreads();

  // One fixed-order exclusive scan makes the shared escape directory
  // bitwise deterministic even though chunks were encoded concurrently.
  if (threadIdx.x == 0) {
    int total = 0;
    for (int chunk = 0; chunk < StaticW16Layout::kChunksPerPage; ++chunk) {
      int count = 0;
      for (int partition = 0; partition < kWarpsPerChunk; ++partition) {
        count += shared_partition_counts[chunk * kWarpsPerChunk + partition];
      }
      const int base = chunk < StaticW16Layout::kNumKvHeads ? k_base : v_base;
      shared_starts[chunk] = total;
      store_i32(compact_page, StaticW16Layout::range_offset(chunk),
                static_cast<int32_t>(
                    StaticW16Layout::pack_metadata(base, total, count)));
      total += count;
    }
    store_i32(compact_page, StaticW16Layout::kPageEscapeCountOffset, total);
    shared_total = total;
  }
  __syncthreads();

  // No directory payload is needed for an exact dense page or for a page
  // that is already known to exceed the shared escape capacity and remain raw.
  if (shared_total == 0 || shared_total > StaticW16Layout::kEscapeCapacity) {
    return shared_total;
  }

  // Scatter the already ordered per-partition records into the deterministic
  // chunk-prefix directory.  No second raw-page scan is needed.
  int partition_start = shared_starts[chunk];
  for (int previous = 0; previous < partition; ++previous) {
    partition_start +=
        shared_partition_counts[chunk * kWarpsPerChunk + previous];
  }
  const int records = shared_partition_counts[warp];
  for (int index = lane; index < records; index += kWarpSize) {
    store_i32(
        compact_page, StaticW16Layout::escape_offset(partition_start + index),
        static_cast<int32_t>(
            shared_records[warp * StaticW16Layout::kEscapeCapacity + index]));
  }
  __syncthreads();
  return shared_total;
}

__device__ void append_q1_raw_tail(const uint16_t* key, int64_t key_head_stride,
                                   const uint16_t* value,
                                   int64_t value_head_stride, uint8_t* raw_page,
                                   int row, bool vector_aligned) {
  constexpr int kValuesPerVector = sizeof(uint4) / sizeof(uint16_t);
  constexpr int kVectorsPerHead = StaticW16Layout::kHeadDim / kValuesPerVector;
  constexpr int kVectorsPerToken =
      2 * StaticW16Layout::kNumKvHeads * kVectorsPerHead;
  if (vector_aligned) {
    for (int index = threadIdx.x; index < kVectorsPerToken;
         index += blockDim.x) {
      const int side = index / (StaticW16Layout::kNumKvHeads * kVectorsPerHead);
      const int side_index =
          index % (StaticW16Layout::kNumKvHeads * kVectorsPerHead);
      const int head = side_index / kVectorsPerHead;
      const int vector = side_index % kVectorsPerHead;
      const uint16_t* source = side == 0 ? key + head * key_head_stride
                                         : value + head * value_head_stride;
      const int chunk = side * StaticW16Layout::kNumKvHeads + head;
      auto* destination = reinterpret_cast<uint4*>(
          raw_page + raw_chunk_offset(chunk) +
          row * StaticW16Layout::kHeadDim * sizeof(uint16_t));
      destination[vector] = reinterpret_cast<const uint4*>(source)[vector];
    }
    return;
  }

  constexpr int kValuesPerToken =
      2 * StaticW16Layout::kNumKvHeads * StaticW16Layout::kHeadDim;
  for (int index = threadIdx.x; index < kValuesPerToken; index += blockDim.x) {
    const int side =
        index / (StaticW16Layout::kNumKvHeads * StaticW16Layout::kHeadDim);
    const int side_index =
        index % (StaticW16Layout::kNumKvHeads * StaticW16Layout::kHeadDim);
    const int head = side_index / StaticW16Layout::kHeadDim;
    const int dim = side_index % StaticW16Layout::kHeadDim;
    const uint16_t bits = side == 0 ? key[head * key_head_stride + dim]
                                    : value[head * value_head_stride + dim];
    const int64_t offset = side == 0 ? RawLayout::key_offset(head, row, dim)
                                     : RawLayout::value_offset(head, row, dim);
    store_u16(raw_page, offset, bits);
  }
}

__device__ bool compact_page_is_safe(const uint8_t* page, int rows) {
  if (rows != StaticW16Layout::kBlockSize) {
    return false;
  }
  const int total = load_i32(page, StaticW16Layout::kPageEscapeCountOffset);
  if (total < 0 || total > StaticW16Layout::kEscapeCapacity) {
    return false;
  }
  int summed_count = 0;
  for (int chunk = 0; chunk < StaticW16Layout::kChunksPerPage; ++chunk) {
    const uint32_t metadata = static_cast<uint32_t>(
        load_i32(page, StaticW16Layout::range_offset(chunk)));
    const int base = StaticW16Layout::metadata_base(metadata);
    const int start = StaticW16Layout::metadata_start(metadata);
    const int count = StaticW16Layout::metadata_count(metadata);
    const int expected_base =
        load_i32(page, chunk < StaticW16Layout::kNumKvHeads
                           ? StaticW16Layout::kKBaseOffset
                           : StaticW16Layout::kVBaseOffset);
    if (!StaticW16Layout::metadata_is_canonical(metadata) || base > 240 ||
        base != expected_base || start > StaticW16Layout::kEscapeCapacity ||
        count > StaticW16Layout::kEscapeCapacity - start) {
      return false;
    }
    summed_count += count;
    if (count == 0) {
      continue;
    }
    for (int previous = 0; previous < chunk; ++previous) {
      const uint32_t previous_metadata = static_cast<uint32_t>(
          load_i32(page, StaticW16Layout::range_offset(previous)));
      const int previous_start =
          StaticW16Layout::metadata_start(previous_metadata);
      const int previous_count =
          StaticW16Layout::metadata_count(previous_metadata);
      if (previous_count != 0 && start < previous_start + previous_count &&
          previous_start < start + count) {
        return false;
      }
    }
  }
  return summed_count == total;
}

__global__ void static_w16_reset_staging_headers_kernel(
    uint8_t* kv_cache, int64_t cache_stride, const int32_t* staging_to_physical,
    const int32_t* valid_rows, int64_t num_staging_slots, int64_t num_blocks,
    int k_base, int v_base, int32_t* fatal) {
  const int64_t staging_slot = blockIdx.x;
  if (staging_slot >= num_staging_slots) {
    return;
  }
  const int32_t physical = staging_to_physical[staging_slot];
  const int rows = valid_rows[staging_slot];
  if (physical == -1 && rows == 0) {
    return;
  }
  if (physical < 0 || physical >= num_blocks || rows <= 0 ||
      rows > StaticW16Layout::kBlockSize) {
    if (threadIdx.x == 0) {
      fail_closed(fatal);
    }
    return;
  }
  uint8_t* page = kv_cache + static_cast<int64_t>(physical) * cache_stride;
  for (int offset = threadIdx.x; offset < 128; offset += blockDim.x) {
    page[StaticW16Layout::kHeaderOffset + offset] = 0;
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    store_i32(page, StaticW16Layout::kStatusOffset,
              StaticW16Layout::kCanonicalRawFallbackStatus);
    store_i32(page, StaticW16Layout::kKBaseOffset, k_base);
    store_i32(page, StaticW16Layout::kVBaseOffset, v_base);
    store_i32(page, StaticW16Layout::kValidRowsOffset, rows);
    store_i32(page, StaticW16Layout::kPageEscapeCountOffset, 0);
  }
}

__global__ void static_w16_pack_staging_chunks_kernel(
    const uint8_t* raw_staging, int64_t raw_stride, uint8_t* kv_cache,
    int64_t cache_stride, const int32_t* staging_to_physical,
    const int32_t* valid_rows, int64_t num_staging_slots, int64_t num_blocks,
    int k_base, int v_base, int32_t* fatal) {
  const int64_t staging_slot = blockIdx.x;
  const int chunk = blockIdx.y;
  if (staging_slot >= num_staging_slots ||
      chunk >= StaticW16Layout::kChunksPerPage) {
    return;
  }
  const int32_t physical = staging_to_physical[staging_slot];
  const int rows = valid_rows[staging_slot];
  if (physical == -1 && rows == 0) {
    return;
  }
  if (physical < 0 || physical >= num_blocks || rows <= 0 ||
      rows > StaticW16Layout::kBlockSize) {
    if (threadIdx.x == 0) {
      fail_closed(fatal);
    }
    return;
  }
  uint8_t* page = kv_cache + static_cast<int64_t>(physical) * cache_stride;
  __shared__ int shared_count;
  __shared__ int shared_start;
  __shared__ uint32_t shared_records[StaticW16Layout::kEscapeCapacity];
  if (rows == StaticW16Layout::kBlockSize) {
    const uint8_t* raw_page = raw_staging + staging_slot * raw_stride;
    pack_chunk(raw_page, page, chunk,
               chunk < StaticW16Layout::kNumKvHeads ? k_base : v_base,
               &shared_count, &shared_start, shared_records);
  }
}

__global__ void static_w16_demote_compact_staging_pages_kernel(
    uint8_t* kv_cache, int64_t cache_stride, const int32_t* staging_to_physical,
    const int32_t* valid_rows, int64_t num_staging_slots, int64_t num_blocks,
    int32_t* page_to_raw_slot, int32_t* free_slots, int32_t* free_count,
    int32_t num_raw_slots, bool retain_safe_full_pages, int32_t* fatal) {
  const int64_t staging_slot = blockIdx.x;
  if (staging_slot >= num_staging_slots || threadIdx.x != 0) {
    return;
  }
  const int32_t physical = staging_to_physical[staging_slot];
  const int rows = valid_rows[staging_slot];
  if (physical == -1 && rows == 0) {
    return;
  }
  if (physical < 0 || physical >= num_blocks || rows <= 0 ||
      rows > StaticW16Layout::kBlockSize) {
    fail_closed(fatal);
    return;
  }
  uint8_t* page = kv_cache + static_cast<int64_t>(physical) * cache_stride;
  if (!compact_page_is_safe(page, rows)) {
    return;
  }
  if (retain_safe_full_pages && rows == StaticW16Layout::kBlockSize) {
    return;
  }
  __threadfence();
  const int32_t raw_slot = page_to_raw_slot[physical];
  if (raw_slot < -1 || raw_slot >= num_raw_slots) {
    fail_closed(fatal);
    return;
  }
  if (raw_slot >= 0) {
    if (atomicCAS(page_to_raw_slot + physical, raw_slot, -1) != raw_slot) {
      fail_closed(fatal);
      return;
    }
    push_raw_slot(raw_slot, free_slots, free_count, num_raw_slots, fatal);
  }
  store_i32(page, StaticW16Layout::kStatusOffset,
            StaticW16Layout::kCanonicalMetadataStatus);
  __threadfence();
}

__global__ void static_w16_persist_fallback_staging_pages_kernel(
    const uint8_t* raw_staging, int64_t raw_stride, uint8_t* kv_cache,
    int64_t cache_stride, uint8_t* raw_pages, int64_t raw_page_stride,
    const int32_t* staging_to_physical, const int32_t* valid_rows,
    int64_t num_staging_slots, int64_t num_blocks, int32_t* page_to_raw_slot,
    int32_t* free_slots, int32_t* free_count, int32_t num_raw_slots,
    bool retain_safe_full_pages, int32_t* fatal) {
  const int64_t staging_slot = blockIdx.x;
  if (staging_slot >= num_staging_slots) {
    return;
  }
  const int32_t physical = staging_to_physical[staging_slot];
  const int rows = valid_rows[staging_slot];
  if (physical == -1 && rows == 0) {
    return;
  }
  if (physical < 0 || physical >= num_blocks || rows <= 0 ||
      rows > StaticW16Layout::kBlockSize) {
    if (threadIdx.x == 0) {
      fail_closed(fatal);
    }
    return;
  }
  uint8_t* page = kv_cache + static_cast<int64_t>(physical) * cache_stride;
  __shared__ int shared_safe;
  if (threadIdx.x == 0) {
    shared_safe = compact_page_is_safe(page, rows) ? 1 : 0;
  }
  __syncthreads();
  if (shared_safe != 0 &&
      !(retain_safe_full_pages && rows == StaticW16Layout::kBlockSize)) {
    return;
  }

  __shared__ int32_t shared_raw_slot;
  if (threadIdx.x == 0) {
    int32_t raw_slot = page_to_raw_slot[physical];
    if (raw_slot < -1 || raw_slot >= num_raw_slots) {
      fail_closed(fatal);
      raw_slot = -1;
    }
    if (raw_slot < 0) {
      raw_slot = pop_raw_slot(free_slots, free_count, num_raw_slots, fatal);
    }
    shared_raw_slot = raw_slot;
  }
  __syncthreads();
  const int32_t raw_slot = shared_raw_slot;
  if (raw_slot < 0 || raw_slot >= num_raw_slots) {
    return;
  }
  const auto* source =
      reinterpret_cast<const uint4*>(raw_staging + staging_slot * raw_stride);
  auto* destination = reinterpret_cast<uint4*>(
      raw_pages + static_cast<int64_t>(raw_slot) * raw_page_stride);
  constexpr int kVectors = StaticW16Layout::kRawPageBytes / sizeof(uint4);
  for (int index = threadIdx.x; index < kVectors; index += blockDim.x) {
    destination[index] = source[index];
  }
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) {
    store_i32(page, StaticW16Layout::kStatusOffset,
              StaticW16Layout::kCanonicalRawFallbackStatus);
    __threadfence();
    const int32_t observed =
        atomicCAS(page_to_raw_slot + physical, -1, raw_slot);
    if (observed != -1 && observed != raw_slot) {
      fail_closed(fatal);
      return;
    }
    __threadfence();
  }
}

__global__ void static_w16_hydrate_staging_chunks_kernel(
    uint8_t* raw_staging, int64_t raw_stride, const uint8_t* kv_cache,
    int64_t cache_stride, const uint8_t* raw_pages, int64_t raw_page_stride,
    const int32_t* page_to_raw_slot, const int32_t* staging_to_physical,
    const int32_t* valid_rows, int64_t num_staging_slots, int64_t num_blocks,
    int32_t num_raw_slots, int32_t* fatal) {
  const int64_t staging_slot = blockIdx.x;
  const int chunk = blockIdx.y;
  if (staging_slot >= num_staging_slots ||
      chunk >= StaticW16Layout::kChunksPerPage) {
    return;
  }
  const int32_t physical = staging_to_physical[staging_slot];
  const int rows = valid_rows[staging_slot];
  if (physical == -1 && rows == 0) {
    return;
  }
  if (physical < 0 || physical >= num_blocks || rows <= 0 ||
      rows > StaticW16Layout::kBlockSize) {
    if (threadIdx.x == 0) {
      fail_closed(fatal);
    }
    return;
  }
  const uint8_t* page =
      kv_cache + static_cast<int64_t>(physical) * cache_stride;
  __shared__ int shared_raw_slot;
  __shared__ int shared_count;
  __shared__ int shared_start;
  __shared__ int shared_base;
  if (threadIdx.x == 0) {
    const int status = load_i32(page, StaticW16Layout::kStatusOffset);
    const int raw_slot = page_to_raw_slot[physical];
    const int expected_status =
        raw_slot >= 0 ? StaticW16Layout::kCanonicalRawFallbackStatus
                      : StaticW16Layout::kCanonicalMetadataStatus;
    if (raw_slot < -1 || raw_slot >= num_raw_slots ||
        status != expected_status) {
      fail_closed(fatal);
    }
    int count = 0;
    int start = 0;
    int base = 0;
    if (raw_slot < 0) {
      const uint32_t metadata = static_cast<uint32_t>(
          load_i32(page, StaticW16Layout::range_offset(chunk)));
      base = StaticW16Layout::metadata_base(metadata);
      start = StaticW16Layout::metadata_start(metadata);
      count = StaticW16Layout::metadata_count(metadata);
      if (!StaticW16Layout::metadata_is_canonical(metadata) || base > 240 ||
          start > StaticW16Layout::kEscapeCapacity ||
          count > StaticW16Layout::kEscapeCapacity - start) {
        fail_closed(fatal);
      }
    } else {
      base = load_i32(page, chunk < StaticW16Layout::kNumKvHeads
                                ? StaticW16Layout::kKBaseOffset
                                : StaticW16Layout::kVBaseOffset);
    }
    shared_raw_slot = raw_slot;
    shared_count = count;
    shared_start = start;
    shared_base = base;
    if (raw_slot < 0 && (shared_base < 0 || shared_base > 240)) {
      fail_closed(fatal);
    }
  }
  __syncthreads();

  uint8_t* destination = raw_staging + staging_slot * raw_stride;
  for (int pos = threadIdx.x; pos < StaticW16Layout::kValuesPerChunk;
       pos += blockDim.x) {
    if (pos / StaticW16Layout::kHeadDim >= rows) {
      continue;
    }
    uint16_t bits;
    if (shared_raw_slot >= 0) {
      const uint8_t* raw_page =
          raw_pages + static_cast<int64_t>(shared_raw_slot) * raw_page_stride;
      bits = load_u16(raw_page, raw_offset(chunk, pos));
    } else {
      const uint8_t sign_mantissa =
          page[StaticW16Layout::kSignMantissaOffset +
               chunk * StaticW16Layout::kValuesPerChunk + pos];
      const uint8_t code_byte =
          page[StaticW16Layout::kCodeOffset +
               chunk * StaticW16Layout::kPairsPerChunk + pos / 2];
      const int code = (pos & 1) == 0 ? code_byte & 0x0f : code_byte >> 4;
      int exponent = shared_base + code;
      for (int index = 0; index < shared_count; ++index) {
        const int entry = load_i32(
            page, StaticW16Layout::escape_offset(shared_start + index));
        if ((entry & 0xffff) == pos) {
          exponent = (entry >> 16) & 0xff;
        }
      }
      bits = static_cast<uint16_t>(((sign_mantissa & 0x80) << 8) |
                                   (exponent << 7) | (sign_mantissa & 0x7f));
    }
    store_u16(destination, raw_offset(chunk, pos), bits);
  }
}

__global__ void static_w16_update_raw_tail_q1_kernel(
    const uint16_t* key, int64_t key_token_stride, int64_t key_head_stride,
    const uint16_t* value, int64_t value_token_stride,
    int64_t value_head_stride, uint8_t* kv_cache, int64_t cache_stride,
    uint8_t* raw_pages, int64_t raw_page_stride, const int64_t* slot_mapping,
    int64_t num_tokens, int64_t num_blocks, int32_t* page_to_raw_slot,
    int32_t* free_slots, int32_t* free_count, int32_t num_raw_slots,
    int32_t* fatal, int k_base, int v_base, bool vector_aligned) {
  __shared__ int32_t shared_physical;
  __shared__ int32_t shared_row;
  __shared__ int32_t shared_raw_slot;
  __shared__ int32_t shared_new_slot;
  const int64_t token = blockIdx.x;
  if (token >= num_tokens) {
    return;
  }
  if (threadIdx.x == 0) {
    const int64_t logical_slot = slot_mapping[token];
    if (logical_slot < 0) {
      shared_physical = -1;
      shared_row = -1;
      shared_raw_slot = -1;
      shared_new_slot = 0;
    } else {
      const int64_t physical64 = logical_slot / StaticW16Layout::kBlockSize;
      const int row = logical_slot % StaticW16Layout::kBlockSize;
      if (physical64 < 0 || physical64 >= num_blocks || row < 0 ||
          row >= StaticW16Layout::kBlockSize) {
        fail_closed(fatal);
        return;
      }
      const int32_t physical = static_cast<int32_t>(physical64);
      const uint8_t* compact_page =
          kv_cache + static_cast<int64_t>(physical) * cache_stride;
      int32_t raw_slot = page_to_raw_slot[physical];
      int32_t new_slot = 0;
      if (raw_slot < -1 || raw_slot >= num_raw_slots) {
        fail_closed(fatal);
        raw_slot = -1;
      }
      const int status = load_i32(compact_page, StaticW16Layout::kStatusOffset);
      const int expected_status =
          raw_slot >= 0 ? StaticW16Layout::kCanonicalRawFallbackStatus
                        : StaticW16Layout::kCanonicalMetadataStatus;
      const bool uninitialized_page =
          status == 0 && raw_slot < 0 && row == 0 &&
          load_i32(compact_page, StaticW16Layout::kValidRowsOffset) == 0;
      if (status != expected_status && !uninitialized_page) {
        fail_closed(fatal);
        return;
      }
      if (raw_slot < 0) {
        if (row != 0 ||
            load_i32(compact_page, StaticW16Layout::kValidRowsOffset) != 0) {
          fail_closed(fatal);
          return;
        }
        if (atomicCAS(page_to_raw_slot + physical, -1,
                      StaticW16Layout::kLockedRawSlot) != -1) {
          fail_closed(fatal);
          return;
        }
        raw_slot = pop_raw_slot(free_slots, free_count, num_raw_slots, fatal);
        new_slot = 1;
      } else {
        const int old_rows =
            load_i32(compact_page, StaticW16Layout::kValidRowsOffset);
        if (old_rows < 1 || old_rows >= StaticW16Layout::kBlockSize ||
            row != old_rows) {
          fail_closed(fatal);
          return;
        }
        // Interior Q1 appends target request-private mutable tail pages.  The
        // vLLM block manager gives each scheduled token a unique writable
        // slot, and the following reader is ordered after this kernel on the
        // same stream.  Keep the map lock only for row 15, where this CTA may
        // release or retain the raw slot while sealing the page.
        if (row == StaticW16Layout::kBlockSize - 1) {
          if (atomicCAS(page_to_raw_slot + physical, raw_slot,
                        StaticW16Layout::kLockedRawSlot) != raw_slot) {
            fail_closed(fatal);
            return;
          }
        }
      }
      shared_physical = physical;
      shared_row = row;
      shared_raw_slot = raw_slot;
      shared_new_slot = new_slot;
    }
  }
  __syncthreads();

  const int32_t physical = shared_physical;
  const int row = shared_row;
  const int32_t raw_slot = shared_raw_slot;
  if (physical < 0 || raw_slot < 0) {
    return;
  }
  uint8_t* compact_page =
      kv_cache + static_cast<int64_t>(physical) * cache_stride;
  uint8_t* raw_page =
      raw_pages + static_cast<int64_t>(raw_slot) * raw_page_stride;
  if (shared_new_slot != 0) {
    for (int offset = threadIdx.x; offset < 640; offset += blockDim.x) {
      compact_page[StaticW16Layout::kHeaderOffset + offset] = 0;
    }
    __syncthreads();
  }
  if (threadIdx.x == 0) {
    if (shared_new_slot != 0) {
      store_i32(compact_page, StaticW16Layout::kStatusOffset,
                StaticW16Layout::kCanonicalRawFallbackStatus);
      store_i32(compact_page, StaticW16Layout::kKBaseOffset, k_base);
      store_i32(compact_page, StaticW16Layout::kVBaseOffset, v_base);
    }
    store_i32(compact_page, StaticW16Layout::kValidRowsOffset, row + 1);
  }

  append_q1_raw_tail(key + token * key_token_stride, key_head_stride,
                     value + token * value_token_stride, value_head_stride,
                     raw_page, row, vector_aligned);
  if (row != StaticW16Layout::kBlockSize - 1) {
    if (shared_new_slot != 0) {
      // Row 0 must publish its newly allocated slot only after the raw payload
      // is complete.  Existing interior pages are already published and need
      // neither a CTA barrier nor a device-wide fence on this same stream.
      __syncthreads();
      if (threadIdx.x == 0) {
        __threadfence();
        if (atomicCAS(page_to_raw_slot + physical,
                      StaticW16Layout::kLockedRawSlot,
                      raw_slot) != StaticW16Layout::kLockedRawSlot) {
          fail_closed(fatal);
        }
      }
    }
    return;
  }
  __syncthreads();

  const int page_escape_count =
      pack_page_chunks_deterministic(raw_page, compact_page, k_base, v_base);
  if (threadIdx.x == 0) {
    // The fixed-order prefix constructs canonical, contiguous, non-overlapping
    // ranges.  Q1 therefore needs only the capacity result; generic staging
    // commits retain compact_page_is_safe because they use atomic allocation.
    if (page_escape_count <= StaticW16Layout::kEscapeCapacity) {
      store_i32(compact_page, StaticW16Layout::kStatusOffset,
                StaticW16Layout::kCanonicalMetadataStatus);
      __threadfence();
      if (atomicCAS(page_to_raw_slot + physical,
                    StaticW16Layout::kLockedRawSlot,
                    -1) != StaticW16Layout::kLockedRawSlot) {
        fail_closed(fatal);
        return;
      }
      push_raw_slot(raw_slot, free_slots, free_count, num_raw_slots, fatal);
    } else {
      store_i32(compact_page, StaticW16Layout::kStatusOffset,
                StaticW16Layout::kCanonicalRawFallbackStatus);
      __threadfence();
      if (atomicCAS(page_to_raw_slot + physical,
                    StaticW16Layout::kLockedRawSlot,
                    raw_slot) != StaticW16Layout::kLockedRawSlot) {
        fail_closed(fatal);
        return;
      }
    }
    __threadfence();
  }
}

void check_static_w16_common(torch::stable::Tensor& kv_cache,
                             torch::stable::Tensor& raw_pages,
                             torch::stable::Tensor& page_to_raw_slot,
                             torch::stable::Tensor& free_slots,
                             torch::stable::Tensor& free_count,
                             torch::stable::Tensor& fatal) {
  using torch::headeronly::ScalarType;
  STD_TORCH_CHECK(kv_cache.device().is_cuda(), "kv_cache must be CUDA");
  const auto device = kv_cache.device();
  STD_TORCH_CHECK(raw_pages.device() == device &&
                      page_to_raw_slot.device() == device &&
                      free_slots.device() == device &&
                      free_count.device() == device && fatal.device() == device,
                  "Static-W16 cache state must share one CUDA device");
  STD_TORCH_CHECK(kv_cache.scalar_type() == ScalarType::Byte &&
                      raw_pages.scalar_type() == ScalarType::Byte,
                  "Static-W16 compact and raw pages must be uint8");
  STD_TORCH_CHECK(page_to_raw_slot.scalar_type() == ScalarType::Int &&
                      free_slots.scalar_type() == ScalarType::Int &&
                      free_count.scalar_type() == ScalarType::Int &&
                      fatal.scalar_type() == ScalarType::Int,
                  "Static-W16 allocator state must be int32");
  STD_TORCH_CHECK(kv_cache.dim() == 2 && kv_cache.size(0) > 0 &&
                      kv_cache.size(0) <= std::numeric_limits<int32_t>::max() &&
                      kv_cache.size(1) == StaticW16Layout::kPageBytes &&
                      kv_cache.stride(1) == 1 &&
                      kv_cache.stride(0) >= StaticW16Layout::kPageBytes,
                  "Static-W16 cache must be contiguous [num_blocks, 49792]");
  STD_TORCH_CHECK(
      raw_pages.dim() == 2 && raw_pages.size(0) > 0 &&
          raw_pages.size(0) <= std::numeric_limits<int32_t>::max() &&
          raw_pages.size(1) >= StaticW16Layout::kRawPageBytes &&
          raw_pages.stride(1) == 1 &&
          raw_pages.stride(0) >= StaticW16Layout::kRawPageBytes,
      "Static-W16 raw sidecar must contain contiguous 65536-byte pages");
  STD_TORCH_CHECK(
      kv_cache.stride(0) % alignof(int32_t) == 0 &&
          raw_pages.stride(0) % alignof(uint4) == 0 &&
          pointer_is_aligned(kv_cache.const_data_ptr(), alignof(int32_t)) &&
          pointer_is_aligned(raw_pages.const_data_ptr(), alignof(uint4)),
      "Static-W16 cache storage does not satisfy native vector alignment");
  STD_TORCH_CHECK(
      page_to_raw_slot.dim() == 1 &&
          page_to_raw_slot.size(0) == kv_cache.size(0) &&
          page_to_raw_slot.stride(0) == 1 && free_slots.dim() == 1 &&
          free_slots.size(0) == raw_pages.size(0) &&
          free_slots.stride(0) == 1 && free_count.dim() == 1 &&
          free_count.size(0) == 1 && free_count.stride(0) == 1 &&
          fatal.dim() == 1 && fatal.size(0) == 1 && fatal.stride(0) == 1,
      "Static-W16 allocator tensors have invalid shapes");
}

void check_bases(int64_t k_base, int64_t v_base) {
  STD_TORCH_CHECK(k_base >= 0 && k_base <= 240 && v_base >= 0 && v_base <= 240,
                  "Static-W16 K/V bases must be in [0, 240]");
}

void check_staging_descriptors(torch::stable::Tensor& raw_staging,
                               torch::stable::Tensor& staging_to_physical,
                               torch::stable::Tensor& valid_rows,
                               const torch::stable::Tensor& kv_cache) {
  using torch::headeronly::ScalarType;
  STD_TORCH_CHECK(raw_staging.device() == kv_cache.device() &&
                      staging_to_physical.device() == kv_cache.device() &&
                      valid_rows.device() == kv_cache.device(),
                  "Static-W16 staging tensors must share the cache device");
  STD_TORCH_CHECK(raw_staging.scalar_type() == ScalarType::Byte &&
                      staging_to_physical.scalar_type() == ScalarType::Int &&
                      valid_rows.scalar_type() == ScalarType::Int,
                  "Static-W16 staging dtypes are invalid");
  STD_TORCH_CHECK(raw_staging.dim() == 2 &&
                      raw_staging.size(1) >= StaticW16Layout::kRawPageBytes &&
                      raw_staging.stride(1) == 1 &&
                      raw_staging.stride(0) >= StaticW16Layout::kRawPageBytes &&
                      staging_to_physical.dim() == 1 &&
                      staging_to_physical.stride(0) == 1 &&
                      valid_rows.dim() == 1 && valid_rows.stride(0) == 1 &&
                      raw_staging.size(0) >= staging_to_physical.size(0) &&
                      valid_rows.size(0) >= staging_to_physical.size(0),
                  "Static-W16 staging tensors have invalid shapes");
  STD_TORCH_CHECK(
      raw_staging.stride(0) % alignof(uint4) == 0 &&
          pointer_is_aligned(raw_staging.const_data_ptr(), alignof(uint4)),
      "Static-W16 raw staging does not satisfy uint4 alignment");
}

}  // namespace

void byte_v2_static_w16_hydrate_raw_staging_from_hybrid_cache(
    torch::stable::Tensor& raw_staging, torch::stable::Tensor& kv_cache,
    torch::stable::Tensor& raw_pages, torch::stable::Tensor& page_to_raw_slot,
    torch::stable::Tensor& staging_to_physical,
    torch::stable::Tensor& valid_rows, torch::stable::Tensor& fatal) {
  // Hydration only consumes the allocator map, but validate the complete state
  // through temporary stack tensors supplied by the caller-independent checks
  // below instead of silently accepting an inconsistent page geometry.
  using torch::headeronly::ScalarType;
  STD_TORCH_CHECK(kv_cache.device().is_cuda(), "kv_cache must be CUDA");
  STD_TORCH_CHECK(raw_pages.device() == kv_cache.device() &&
                      page_to_raw_slot.device() == kv_cache.device() &&
                      fatal.device() == kv_cache.device(),
                  "Static-W16 hydrate state must share one CUDA device");
  STD_TORCH_CHECK(kv_cache.scalar_type() == ScalarType::Byte &&
                      raw_pages.scalar_type() == ScalarType::Byte &&
                      page_to_raw_slot.scalar_type() == ScalarType::Int &&
                      fatal.scalar_type() == ScalarType::Int,
                  "Static-W16 hydrate dtypes are invalid");
  STD_TORCH_CHECK(
      kv_cache.dim() == 2 && kv_cache.size(0) > 0 &&
          kv_cache.size(0) <= std::numeric_limits<int32_t>::max() &&
          kv_cache.size(1) == StaticW16Layout::kPageBytes &&
          kv_cache.stride(1) == 1 &&
          kv_cache.stride(0) >= StaticW16Layout::kPageBytes &&
          raw_pages.dim() == 2 && raw_pages.size(0) > 0 &&
          raw_pages.size(0) <= std::numeric_limits<int32_t>::max() &&
          raw_pages.size(1) >= StaticW16Layout::kRawPageBytes &&
          raw_pages.stride(1) == 1 &&
          raw_pages.stride(0) >= StaticW16Layout::kRawPageBytes &&
          page_to_raw_slot.dim() == 1 &&
          page_to_raw_slot.size(0) == kv_cache.size(0) &&
          page_to_raw_slot.stride(0) == 1 && fatal.dim() == 1 &&
          fatal.size(0) == 1,
      "Static-W16 hydrate state shapes are invalid");
  STD_TORCH_CHECK(
      kv_cache.stride(0) % alignof(int32_t) == 0 &&
          raw_pages.stride(0) % alignof(uint4) == 0 &&
          pointer_is_aligned(kv_cache.const_data_ptr(), alignof(int32_t)) &&
          pointer_is_aligned(raw_pages.const_data_ptr(), alignof(uint4)),
      "Static-W16 hydrate storage does not satisfy native vector alignment");
  check_staging_descriptors(raw_staging, staging_to_physical, valid_rows,
                            kv_cache);
  const int64_t num_slots = staging_to_physical.size(0);
  if (num_slots == 0) {
    return;
  }
  const torch::stable::accelerator::DeviceGuard guard(
      kv_cache.get_device_index());
  const cudaStream_t stream =
      get_current_cuda_stream(kv_cache.get_device_index());
  static_w16_hydrate_staging_chunks_kernel<<<
      dim3(static_cast<unsigned int>(num_slots),
           StaticW16Layout::kChunksPerPage),
      256, 0, stream>>>(
      reinterpret_cast<uint8_t*>(raw_staging.mutable_data_ptr()),
      raw_staging.stride(0),
      reinterpret_cast<const uint8_t*>(kv_cache.const_data_ptr()),
      kv_cache.stride(0),
      reinterpret_cast<const uint8_t*>(raw_pages.const_data_ptr()),
      raw_pages.stride(0), page_to_raw_slot.const_data_ptr<int32_t>(),
      staging_to_physical.const_data_ptr<int32_t>(),
      valid_rows.const_data_ptr<int32_t>(), num_slots, kv_cache.size(0),
      static_cast<int32_t>(raw_pages.size(0)),
      fatal.mutable_data_ptr<int32_t>());
  const cudaError_t error = cudaGetLastError();
  STD_TORCH_CHECK(error == cudaSuccess, "Static-W16 hydrate launch failed: ",
                  cudaGetErrorString(error));
}

void byte_v2_static_w16_commit_raw_staging_to_hybrid_cache(
    torch::stable::Tensor& raw_staging, torch::stable::Tensor& kv_cache,
    torch::stable::Tensor& raw_pages, torch::stable::Tensor& page_to_raw_slot,
    torch::stable::Tensor& free_slots, torch::stable::Tensor& free_count,
    torch::stable::Tensor& fatal, torch::stable::Tensor& staging_to_physical,
    torch::stable::Tensor& valid_rows, int64_t k_base, int64_t v_base,
    bool retain_safe_full_pages) {
  check_static_w16_common(kv_cache, raw_pages, page_to_raw_slot, free_slots,
                          free_count, fatal);
  check_staging_descriptors(raw_staging, staging_to_physical, valid_rows,
                            kv_cache);
  check_bases(k_base, v_base);
  const int64_t num_slots = staging_to_physical.size(0);
  if (num_slots == 0) {
    return;
  }
  const torch::stable::accelerator::DeviceGuard guard(
      kv_cache.get_device_index());
  const cudaStream_t stream =
      get_current_cuda_stream(kv_cache.get_device_index());
  static_w16_reset_staging_headers_kernel<<<
      static_cast<unsigned int>(num_slots), 128, 0, stream>>>(
      reinterpret_cast<uint8_t*>(kv_cache.mutable_data_ptr()),
      kv_cache.stride(0), staging_to_physical.const_data_ptr<int32_t>(),
      valid_rows.const_data_ptr<int32_t>(), num_slots, kv_cache.size(0),
      static_cast<int>(k_base), static_cast<int>(v_base),
      fatal.mutable_data_ptr<int32_t>());
  cudaError_t error = cudaGetLastError();
  STD_TORCH_CHECK(
      error == cudaSuccess,
      "Static-W16 header reset launch failed: ", cudaGetErrorString(error));

  static_w16_pack_staging_chunks_kernel<<<dim3(static_cast<unsigned int>(
                                                   num_slots),
                                               StaticW16Layout::kChunksPerPage),
                                          256, 0, stream>>>(
      reinterpret_cast<const uint8_t*>(raw_staging.const_data_ptr()),
      raw_staging.stride(0),
      reinterpret_cast<uint8_t*>(kv_cache.mutable_data_ptr()),
      kv_cache.stride(0), staging_to_physical.const_data_ptr<int32_t>(),
      valid_rows.const_data_ptr<int32_t>(), num_slots, kv_cache.size(0),
      static_cast<int>(k_base), static_cast<int>(v_base),
      fatal.mutable_data_ptr<int32_t>());
  error = cudaGetLastError();
  STD_TORCH_CHECK(error == cudaSuccess,
                  "Static-W16 pack launch failed: ", cudaGetErrorString(error));

  static_w16_demote_compact_staging_pages_kernel<<<
      static_cast<unsigned int>(num_slots), 32, 0, stream>>>(
      reinterpret_cast<uint8_t*>(kv_cache.mutable_data_ptr()),
      kv_cache.stride(0), staging_to_physical.const_data_ptr<int32_t>(),
      valid_rows.const_data_ptr<int32_t>(), num_slots, kv_cache.size(0),
      page_to_raw_slot.mutable_data_ptr<int32_t>(),
      free_slots.mutable_data_ptr<int32_t>(),
      free_count.mutable_data_ptr<int32_t>(),
      static_cast<int32_t>(raw_pages.size(0)), retain_safe_full_pages,
      fatal.mutable_data_ptr<int32_t>());
  error = cudaGetLastError();
  STD_TORCH_CHECK(error == cudaSuccess, "Static-W16 demote launch failed: ",
                  cudaGetErrorString(error));

  static_w16_persist_fallback_staging_pages_kernel<<<
      static_cast<unsigned int>(num_slots), 256, 0, stream>>>(
      reinterpret_cast<const uint8_t*>(raw_staging.const_data_ptr()),
      raw_staging.stride(0),
      reinterpret_cast<uint8_t*>(kv_cache.mutable_data_ptr()),
      kv_cache.stride(0),
      reinterpret_cast<uint8_t*>(raw_pages.mutable_data_ptr()),
      raw_pages.stride(0), staging_to_physical.const_data_ptr<int32_t>(),
      valid_rows.const_data_ptr<int32_t>(), num_slots, kv_cache.size(0),
      page_to_raw_slot.mutable_data_ptr<int32_t>(),
      free_slots.mutable_data_ptr<int32_t>(),
      free_count.mutable_data_ptr<int32_t>(),
      static_cast<int32_t>(raw_pages.size(0)), retain_safe_full_pages,
      fatal.mutable_data_ptr<int32_t>());
  error = cudaGetLastError();
  STD_TORCH_CHECK(
      error == cudaSuccess,
      "Static-W16 fallback persist launch failed: ", cudaGetErrorString(error));
}

void byte_v2_static_w16_update_hybrid_cache_raw_tail_q1(
    torch::stable::Tensor& key, torch::stable::Tensor& value,
    torch::stable::Tensor& kv_cache, torch::stable::Tensor& raw_pages,
    torch::stable::Tensor& slot_mapping,
    torch::stable::Tensor& page_to_raw_slot, torch::stable::Tensor& free_slots,
    torch::stable::Tensor& free_count, torch::stable::Tensor& fatal,
    int64_t k_base, int64_t v_base) {
  using torch::headeronly::ScalarType;
  check_static_w16_common(kv_cache, raw_pages, page_to_raw_slot, free_slots,
                          free_count, fatal);
  check_bases(k_base, v_base);
  STD_TORCH_CHECK(key.device() == kv_cache.device() &&
                      value.device() == kv_cache.device() &&
                      slot_mapping.device() == kv_cache.device(),
                  "Static-W16 Q1 tensors must share the cache device");
  STD_TORCH_CHECK(key.scalar_type() == ScalarType::BFloat16 &&
                      value.scalar_type() == ScalarType::BFloat16 &&
                      slot_mapping.scalar_type() == ScalarType::Long,
                  "Static-W16 Q1 requires BF16 K/V and int64 slot mapping");
  STD_TORCH_CHECK(
      key.dim() == 3 && value.dim() == 3 && key.size(0) > 0 &&
          value.size(0) == key.size(0) &&
          key.size(0) <= std::numeric_limits<int32_t>::max() &&
          key.size(1) == StaticW16Layout::kNumKvHeads &&
          value.size(1) == StaticW16Layout::kNumKvHeads &&
          key.size(2) == StaticW16Layout::kHeadDim &&
          value.size(2) == StaticW16Layout::kHeadDim && key.stride(2) == 1 &&
          value.stride(2) == 1 && slot_mapping.dim() == 1 &&
          slot_mapping.size(0) == key.size(0) && slot_mapping.stride(0) == 1,
      "Static-W16 Q1 requires matching [N, 8, 128] K/V and N slots");
  const torch::stable::accelerator::DeviceGuard guard(
      kv_cache.get_device_index());
  const cudaStream_t stream =
      get_current_cuda_stream(kv_cache.get_device_index());
  constexpr int64_t kValuesPerVector = sizeof(uint4) / sizeof(uint16_t);
  const bool vector_aligned =
      key.stride(0) % kValuesPerVector == 0 &&
      key.stride(1) % kValuesPerVector == 0 &&
      value.stride(0) % kValuesPerVector == 0 &&
      value.stride(1) % kValuesPerVector == 0 &&
      pointer_is_aligned(key.const_data_ptr(), alignof(uint4)) &&
      pointer_is_aligned(value.const_data_ptr(), alignof(uint4));
  const int64_t num_tokens = key.size(0);
  static_w16_update_raw_tail_q1_kernel<<<static_cast<unsigned int>(num_tokens),
                                         StaticW16Layout::kQ1Threads, 0,
                                         stream>>>(
      reinterpret_cast<const uint16_t*>(key.const_data_ptr()), key.stride(0),
      key.stride(1), reinterpret_cast<const uint16_t*>(value.const_data_ptr()),
      value.stride(0), value.stride(1),
      reinterpret_cast<uint8_t*>(kv_cache.mutable_data_ptr()),
      kv_cache.stride(0),
      reinterpret_cast<uint8_t*>(raw_pages.mutable_data_ptr()),
      raw_pages.stride(0), slot_mapping.const_data_ptr<int64_t>(), num_tokens,
      kv_cache.size(0), page_to_raw_slot.mutable_data_ptr<int32_t>(),
      free_slots.mutable_data_ptr<int32_t>(),
      free_count.mutable_data_ptr<int32_t>(),
      static_cast<int32_t>(raw_pages.size(0)),
      fatal.mutable_data_ptr<int32_t>(), static_cast<int>(k_base),
      static_cast<int>(v_base), vector_aligned);
  const cudaError_t error = cudaGetLastError();
  STD_TORCH_CHECK(
      error == cudaSuccess,
      "Static-W16 raw-tail Q1 launch failed: ", cudaGetErrorString(error));
}
