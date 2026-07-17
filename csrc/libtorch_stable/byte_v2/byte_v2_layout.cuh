#pragma once

#include <cstdint>

namespace vllm::byte_v2 {

constexpr int kByteV2MaxMacroPages = 8;
constexpr int kByteV2MacroDescriptorBytes = 128;

#if defined(__CUDACC__) || defined(__HIPCC__)
  #define VLLM_BYTE_V2_HOST_DEVICE __host__ __device__
#else
  #define VLLM_BYTE_V2_HOST_DEVICE
#endif

VLLM_BYTE_V2_HOST_DEVICE constexpr int cdiv(int a, int b) {
  return (a + b - 1) / b;
}

VLLM_BYTE_V2_HOST_DEVICE constexpr int required_bits(int max_inclusive_value) {
  int bits = 1;
  int value = max_inclusive_value;
  while (value >>= 1) {
    ++bits;
  }
  return bits;
}

template <int CodecTokenBlock_ = 16, int CodecDimBlock_ = 16,
          int AllocBlockTokens_ = 16, int ComputeBlockN_ = 64,
          int HeadDim_ = 128, int HeadDimV_ = 128>
struct ByteV2TilePolicy {
  static_assert(CodecTokenBlock_ > 0);
  static_assert(CodecDimBlock_ > 0);
  static_assert(AllocBlockTokens_ > 0);
  static_assert(ComputeBlockN_ > 0);
  static_assert(HeadDim_ > 0);
  static_assert(HeadDimV_ > 0);
  static_assert(AllocBlockTokens_ % CodecTokenBlock_ == 0);
  static_assert(HeadDim_ % CodecDimBlock_ == 0);
  static_assert(HeadDimV_ % CodecDimBlock_ == 0);
  static_assert(ComputeBlockN_ % AllocBlockTokens_ == 0);
  static_assert((CodecTokenBlock_ * CodecDimBlock_) % 2 == 0);

  static constexpr int CodecTokenBlock = CodecTokenBlock_;
  static constexpr int CodecDimBlock = CodecDimBlock_;
  static constexpr int AllocBlockTokens = AllocBlockTokens_;
  static constexpr int ComputeBlockN = ComputeBlockN_;
  static constexpr int HeadDim = HeadDim_;
  static constexpr int HeadDimV = HeadDimV_;

  static constexpr int CodecTileElems = CodecTokenBlock * CodecDimBlock;
  static constexpr int CodecPackedElems = CodecTileElems / 2;
  static constexpr int KDimTiles = HeadDim / CodecDimBlock;
  static constexpr int VDimTiles = HeadDimV / CodecDimBlock;
  static constexpr int CodecTokenTilesPerAllocBlock =
      AllocBlockTokens / CodecTokenBlock;
  static constexpr int AllocBlocksPerComputeTile =
      ComputeBlockN / AllocBlockTokens;
  static constexpr int CodecTokenTilesPerComputeTile =
      ComputeBlockN / CodecTokenBlock;
  static constexpr int CodecTilesPerKPage =
      CodecTokenTilesPerAllocBlock * KDimTiles;
  static constexpr int CodecTilesPerVPage =
      CodecTokenTilesPerAllocBlock * VDimTiles;
};

using DefaultByteV2TilePolicy = ByteV2TilePolicy<>;

template <typename Policy = DefaultByteV2TilePolicy, int LowBytesPerElem = 1,
          int ExponentCodeBits = 4, bool OutlierHighSideband = false>
struct ByteV2CodecPayloadPolicy {
  static_assert(LowBytesPerElem > 0);
  static_assert(ExponentCodeBits > 0);

  static constexpr int LowBytesPerElemValue = LowBytesPerElem;
  static constexpr int ExponentCodeBitsValue = ExponentCodeBits;
  static constexpr bool OutlierHighSidebandValue = OutlierHighSideband;
  static constexpr int LowBytesPerCodecTile =
      Policy::CodecTileElems * LowBytesPerElem;
  static constexpr int CodeBytesPerCodecTile =
      cdiv(Policy::CodecTileElems * ExponentCodeBits, 8);
  static constexpr int BytesPerCodecTile =
      LowBytesPerCodecTile + CodeBytesPerCodecTile;
};

template <typename Policy = DefaultByteV2TilePolicy, int ValueBits = 16>
struct ByteV2OutlierEntryPolicy {
  static_assert(ValueBits > 0);

  static constexpr int ElemBits = required_bits(Policy::CodecTileElems - 1);
  static constexpr int ValueBitsValue = ValueBits;
  static constexpr int EntryBits = ElemBits + ValueBits;
  static constexpr int EntryBytes = cdiv(EntryBits, 8);
  static constexpr int MaxElemIndex = (1 << ElemBits) - 1;
  static constexpr uint32_t ElemMask = (uint32_t{1} << ElemBits) - 1;
  static constexpr uint32_t ValueMask = (uint32_t{1} << ValueBits) - 1;

  VLLM_BYTE_V2_HOST_DEVICE static constexpr uint32_t encode(
      uint32_t elem_index, uint32_t value_bits) {
    return ((value_bits & ValueMask) << ElemBits) | (elem_index & ElemMask);
  }

  VLLM_BYTE_V2_HOST_DEVICE static constexpr uint32_t decode_elem_index(
      uint32_t entry) {
    return entry & ElemMask;
  }

  VLLM_BYTE_V2_HOST_DEVICE static constexpr uint32_t decode_value_bits(
      uint32_t entry) {
    return (entry >> ElemBits) & ValueMask;
  }
};

template <typename Policy = DefaultByteV2TilePolicy,
          typename CodecPayloadPolicy = ByteV2CodecPayloadPolicy<Policy>,
          int NumKvHeads = 8, int PageHeaderBytes = 128,
          int KvHeadMetaBytes = 64, int AlignmentBytes = 128,
          int RawElementBytes = 2, int OutlierValueBits = 8,
          int OutlierEntriesPerTile = 256, bool IncludeRawPayload = false>
struct ByteV2PageLayoutV4 {
  static_assert(NumKvHeads > 0);
  static_assert(PageHeaderBytes > 0);
  static_assert(KvHeadMetaBytes > 0);
  static_assert(AlignmentBytes > 0);
  static_assert(RawElementBytes > 0);
  static_assert(OutlierValueBits > 0);
  static_assert(OutlierEntriesPerTile > 0);
  static_assert(IncludeRawPayload ||
                OutlierEntriesPerTile >= Policy::CodecTileElems);

  using TilePolicy = Policy;
  using OutlierEntryPolicy = ByteV2OutlierEntryPolicy<Policy, OutlierValueBits>;

  static constexpr int NumKvHeadsValue = NumKvHeads;
  static constexpr int RawElementBytesValue = RawElementBytes;
  static constexpr int OutlierEntriesPerTileValue = OutlierEntriesPerTile;
  static constexpr bool IncludeRawPayloadValue = IncludeRawPayload;
  static constexpr bool PagePooledOutliersValue = false;

  static constexpr int align_up(int value) {
    return cdiv(value, AlignmentBytes) * AlignmentBytes;
  }

  static constexpr int MacroPages = Policy::AllocBlocksPerComputeTile;
  static constexpr int KvHeadRequiredMetaBytes =
      32 + Policy::CodecTilesPerKPage + Policy::CodecTilesPerVPage;
  static_assert(KvHeadMetaBytes >= KvHeadRequiredMetaBytes);
  static constexpr int MetadataBytes =
      PageHeaderBytes + NumKvHeads * KvHeadMetaBytes;
  static constexpr int AlignedMetadataBytes = align_up(MetadataBytes);
  static constexpr int CodecLowBytesPerElem =
      CodecPayloadPolicy::LowBytesPerElemValue;
  static constexpr int CodecExponentCodeBits =
      CodecPayloadPolicy::ExponentCodeBitsValue;
  static constexpr bool CodecOutlierHighSideband =
      CodecPayloadPolicy::OutlierHighSidebandValue;
  static constexpr int CodecPayloadBytesPerTile =
      CodecPayloadPolicy::BytesPerCodecTile;
  static constexpr int KPayloadBytesPerKvHead =
      Policy::CodecTilesPerKPage * CodecPayloadBytesPerTile;
  static constexpr int VPayloadBytesPerKvHead =
      Policy::CodecTilesPerVPage * CodecPayloadBytesPerTile;
  static constexpr int AlignedKPayloadBytesPerKvHead =
      align_up(KPayloadBytesPerKvHead);
  static constexpr int AlignedVPayloadBytesPerKvHead =
      align_up(VPayloadBytesPerKvHead);
  static constexpr int CompressedPayloadBytes =
      NumKvHeads *
      (AlignedKPayloadBytesPerKvHead + AlignedVPayloadBytesPerKvHead);
  static constexpr int OutlierEntryBytes = OutlierEntryPolicy::EntryBytes;
  static constexpr int OutlierPayloadBytesPerTile =
      OutlierEntriesPerTile * OutlierEntryBytes;
  static constexpr int KOutlierPayloadBytesPerKvHead =
      Policy::CodecTilesPerKPage * OutlierPayloadBytesPerTile;
  static constexpr int VOutlierPayloadBytesPerKvHead =
      Policy::CodecTilesPerVPage * OutlierPayloadBytesPerTile;
  static constexpr int AlignedKOutlierPayloadBytesPerKvHead =
      align_up(KOutlierPayloadBytesPerKvHead);
  static constexpr int AlignedVOutlierPayloadBytesPerKvHead =
      align_up(VOutlierPayloadBytesPerKvHead);
  static constexpr int OutlierPayloadBytes =
      NumKvHeads * (AlignedKOutlierPayloadBytesPerKvHead +
                    AlignedVOutlierPayloadBytesPerKvHead);
  static constexpr int RawKPayloadBytesPerKvHead =
      Policy::AllocBlockTokens * Policy::HeadDim * RawElementBytes;
  static constexpr int RawVPayloadBytesPerKvHead =
      Policy::AllocBlockTokens * Policy::HeadDimV * RawElementBytes;
  static constexpr int AlignedRawKPayloadBytesPerKvHead =
      align_up(RawKPayloadBytesPerKvHead);
  static constexpr int AlignedRawVPayloadBytesPerKvHead =
      align_up(RawVPayloadBytesPerKvHead);
  static constexpr int RawPayloadBytes =
      IncludeRawPayload ? NumKvHeads * (AlignedRawKPayloadBytesPerKvHead +
                                        AlignedRawVPayloadBytesPerKvHead)
                        : 0;
  static constexpr int PayloadBytes =
      CompressedPayloadBytes + OutlierPayloadBytes + RawPayloadBytes;
  static constexpr int PageSizeBytes = AlignedMetadataBytes + PayloadBytes;

  static constexpr int KPayloadBaseBytes = AlignedMetadataBytes;
  static constexpr int VPayloadBaseBytes =
      KPayloadBaseBytes + NumKvHeads * AlignedKPayloadBytesPerKvHead;
  static constexpr int KOutlierPayloadBaseBytes =
      VPayloadBaseBytes + NumKvHeads * AlignedVPayloadBytesPerKvHead;
  static constexpr int VOutlierPayloadBaseBytes =
      KOutlierPayloadBaseBytes +
      NumKvHeads * AlignedKOutlierPayloadBytesPerKvHead;
  static constexpr int RawKPayloadBaseBytes =
      VOutlierPayloadBaseBytes +
      NumKvHeads * AlignedVOutlierPayloadBytesPerKvHead;
  static constexpr int RawVPayloadBaseBytes =
      RawKPayloadBaseBytes +
      (IncludeRawPayload ? NumKvHeads * AlignedRawKPayloadBytesPerKvHead : 0);

  VLLM_BYTE_V2_HOST_DEVICE static constexpr int kv_head_meta_offset(
      int kv_head) {
    return PageHeaderBytes + kv_head * KvHeadMetaBytes;
  }

  VLLM_BYTE_V2_HOST_DEVICE static constexpr int k_fallback_mask_offset(
      int kv_head) {
    return kv_head_meta_offset(kv_head);
  }

  VLLM_BYTE_V2_HOST_DEVICE static constexpr int v_fallback_mask_offset(
      int kv_head) {
    return kv_head_meta_offset(kv_head) + 4;
  }

  VLLM_BYTE_V2_HOST_DEVICE static constexpr int k_outlier_mask_offset(
      int kv_head) {
    return kv_head_meta_offset(kv_head) + 24;
  }

  VLLM_BYTE_V2_HOST_DEVICE static constexpr int v_outlier_mask_offset(
      int kv_head) {
    return kv_head_meta_offset(kv_head) + 28;
  }

  VLLM_BYTE_V2_HOST_DEVICE static constexpr int k_base_offset(
      int kv_head, int dim_tile, int token_tile = 0) {
    return kv_head_meta_offset(kv_head) + 8 +
           dim_tile * Policy::CodecTokenTilesPerAllocBlock + token_tile;
  }

  VLLM_BYTE_V2_HOST_DEVICE static constexpr int v_base_offset(
      int kv_head, int dim_tile, int token_tile = 0) {
    return kv_head_meta_offset(kv_head) + 8 + Policy::CodecTilesPerKPage +
           token_tile * Policy::VDimTiles + dim_tile;
  }

  VLLM_BYTE_V2_HOST_DEVICE static constexpr int k_outlier_count_offset(
      int kv_head, int dim_tile, int token_tile = 0) {
    return kv_head_meta_offset(kv_head) + 32 +
           dim_tile * Policy::CodecTokenTilesPerAllocBlock + token_tile;
  }

  VLLM_BYTE_V2_HOST_DEVICE static constexpr int v_outlier_count_offset(
      int kv_head, int dim_tile, int token_tile = 0) {
    return kv_head_meta_offset(kv_head) + 32 + Policy::CodecTilesPerKPage +
           token_tile * Policy::VDimTiles + dim_tile;
  }

  VLLM_BYTE_V2_HOST_DEVICE static int k_outlier_count(const uint8_t* page,
                                                      int kv_head, int dim_tile,
                                                      int token_tile = 0) {
    return static_cast<int>(
        page[k_outlier_count_offset(kv_head, dim_tile, token_tile)]);
  }

  VLLM_BYTE_V2_HOST_DEVICE static int v_outlier_count(const uint8_t* page,
                                                      int kv_head, int dim_tile,
                                                      int token_tile = 0) {
    return static_cast<int>(
        page[v_outlier_count_offset(kv_head, dim_tile, token_tile)]);
  }

  VLLM_BYTE_V2_HOST_DEVICE static void set_k_outlier_count(
      uint8_t* page, int kv_head, int dim_tile, int token_tile, int count) {
    page[k_outlier_count_offset(kv_head, dim_tile, token_tile)] =
        static_cast<uint8_t>(count);
  }

  VLLM_BYTE_V2_HOST_DEVICE static void set_v_outlier_count(
      uint8_t* page, int kv_head, int dim_tile, int token_tile, int count) {
    page[v_outlier_count_offset(kv_head, dim_tile, token_tile)] =
        static_cast<uint8_t>(count);
  }

  VLLM_BYTE_V2_HOST_DEVICE static constexpr int k_payload_offset(
      int kv_head, int dim_tile, int token_tile = 0) {
    return KPayloadBaseBytes + kv_head * AlignedKPayloadBytesPerKvHead +
           (dim_tile * Policy::CodecTokenTilesPerAllocBlock + token_tile) *
               CodecPayloadBytesPerTile;
  }

  VLLM_BYTE_V2_HOST_DEVICE static constexpr int v_payload_offset(
      int kv_head, int dim_tile, int token_tile = 0) {
    return VPayloadBaseBytes + kv_head * AlignedVPayloadBytesPerKvHead +
           (token_tile * Policy::VDimTiles + dim_tile) *
               CodecPayloadBytesPerTile;
  }

  VLLM_BYTE_V2_HOST_DEVICE static constexpr int k_outlier_payload_offset(
      int kv_head, int dim_tile, int token_tile = 0, int entry_idx = 0) {
    return KOutlierPayloadBaseBytes +
           kv_head * AlignedKOutlierPayloadBytesPerKvHead +
           (dim_tile * Policy::CodecTokenTilesPerAllocBlock + token_tile) *
               OutlierPayloadBytesPerTile +
           entry_idx * OutlierEntryBytes;
  }

  VLLM_BYTE_V2_HOST_DEVICE static constexpr int v_outlier_payload_offset(
      int kv_head, int dim_tile, int token_tile = 0, int entry_idx = 0) {
    return VOutlierPayloadBaseBytes +
           kv_head * AlignedVOutlierPayloadBytesPerKvHead +
           (token_tile * Policy::VDimTiles + dim_tile) *
               OutlierPayloadBytesPerTile +
           entry_idx * OutlierEntryBytes;
  }

  VLLM_BYTE_V2_HOST_DEVICE static int k_outlier_payload_offset(
      const uint8_t*, int kv_head, int dim_tile, int token_tile = 0,
      int entry_idx = 0) {
    return k_outlier_payload_offset(kv_head, dim_tile, token_tile, entry_idx);
  }

  VLLM_BYTE_V2_HOST_DEVICE static int v_outlier_payload_offset(
      const uint8_t*, int kv_head, int dim_tile, int token_tile = 0,
      int entry_idx = 0) {
    return v_outlier_payload_offset(kv_head, dim_tile, token_tile, entry_idx);
  }

  VLLM_BYTE_V2_HOST_DEVICE static constexpr int raw_key_offset(int kv_head,
                                                               int row,
                                                               int dim) {
    return RawKPayloadBaseBytes + kv_head * AlignedRawKPayloadBytesPerKvHead +
           (row * Policy::HeadDim + dim) * RawElementBytes;
  }

  VLLM_BYTE_V2_HOST_DEVICE static constexpr int raw_value_offset(int kv_head,
                                                                 int row,
                                                                 int dim) {
    return RawVPayloadBaseBytes + kv_head * AlignedRawVPayloadBytesPerKvHead +
           (row * Policy::HeadDimV + dim) * RawElementBytes;
  }
};

template <typename Policy = DefaultByteV2TilePolicy,
          typename CodecPayloadPolicy = ByteV2CodecPayloadPolicy<Policy>,
          int NumKvHeads = 8, int PageHeaderBytes = 128,
          int KvHeadMetaBytes = 96, int AlignmentBytes = 128,
          int OutlierValueBits = 8, int OutlierPoolEntries = 1024>
struct ByteV2PageLayoutV5 {
  static_assert(NumKvHeads > 0);
  static_assert(PageHeaderBytes >= 8);
  static_assert(KvHeadMetaBytes > 0);
  static_assert(AlignmentBytes > 0);
  static_assert(OutlierValueBits > 0);
  static_assert(OutlierPoolEntries >= Policy::CodecTileElems);
  static_assert(OutlierPoolEntries <= 65535);
  static_assert(Policy::CodecTilesPerKPage <= 32);
  static_assert(Policy::CodecTilesPerVPage <= 32);

  using TilePolicy = Policy;
  using OutlierEntryPolicy = ByteV2OutlierEntryPolicy<Policy, OutlierValueBits>;

  static constexpr int NumKvHeadsValue = NumKvHeads;
  static constexpr int RawElementBytesValue = 2;
  static constexpr int OutlierEntriesPerTileValue = Policy::CodecTileElems;
  static constexpr int OutlierPoolEntriesValue = OutlierPoolEntries;
  static constexpr bool IncludeRawPayloadValue = false;
  static constexpr bool PagePooledOutliersValue = true;

  static constexpr int align_up(int value) {
    return cdiv(value, AlignmentBytes) * AlignmentBytes;
  }

  static constexpr int MacroPages = Policy::AllocBlocksPerComputeTile;
  static constexpr int TilesPerKvHead =
      Policy::CodecTilesPerKPage + Policy::CodecTilesPerVPage;
  static constexpr int KvHeadRequiredMetaBytes = 32 + 4 * TilesPerKvHead;
  static_assert(KvHeadMetaBytes >= KvHeadRequiredMetaBytes);
  static constexpr int MetadataBytes =
      PageHeaderBytes + NumKvHeads * KvHeadMetaBytes;
  static constexpr int AlignedMetadataBytes = align_up(MetadataBytes);
  static constexpr int CodecLowBytesPerElem =
      CodecPayloadPolicy::LowBytesPerElemValue;
  static constexpr int CodecExponentCodeBits =
      CodecPayloadPolicy::ExponentCodeBitsValue;
  static constexpr bool CodecOutlierHighSideband =
      CodecPayloadPolicy::OutlierHighSidebandValue;
  static constexpr int CodecPayloadBytesPerTile =
      CodecPayloadPolicy::BytesPerCodecTile;
  static constexpr int KPayloadBytesPerKvHead =
      Policy::CodecTilesPerKPage * CodecPayloadBytesPerTile;
  static constexpr int VPayloadBytesPerKvHead =
      Policy::CodecTilesPerVPage * CodecPayloadBytesPerTile;
  static constexpr int AlignedKPayloadBytesPerKvHead =
      align_up(KPayloadBytesPerKvHead);
  static constexpr int AlignedVPayloadBytesPerKvHead =
      align_up(VPayloadBytesPerKvHead);
  static constexpr int CompressedPayloadBytes =
      NumKvHeads *
      (AlignedKPayloadBytesPerKvHead + AlignedVPayloadBytesPerKvHead);
  static constexpr int OutlierEntryBytes = OutlierEntryPolicy::EntryBytes;
  static constexpr int OutlierPoolBytes =
      OutlierPoolEntries * OutlierEntryBytes;
  static constexpr int PayloadBytes = CompressedPayloadBytes + OutlierPoolBytes;
  static constexpr int PageSizeBytes = AlignedMetadataBytes + PayloadBytes;

  static constexpr int OutlierPoolUsedOffset = 0;
  static constexpr int OutlierPoolOverflowOffset = 4;
  static constexpr int KPayloadBaseBytes = AlignedMetadataBytes;
  static constexpr int VPayloadBaseBytes =
      KPayloadBaseBytes + NumKvHeads * AlignedKPayloadBytesPerKvHead;
  static constexpr int OutlierPoolBaseBytes =
      VPayloadBaseBytes + NumKvHeads * AlignedVPayloadBytesPerKvHead;
  static constexpr int RawKPayloadBaseBytes = PageSizeBytes;
  static constexpr int RawVPayloadBaseBytes = PageSizeBytes;

  VLLM_BYTE_V2_HOST_DEVICE static uint16_t load_u16(const uint8_t* page,
                                                    int offset) {
    return static_cast<uint16_t>(page[offset]) |
           static_cast<uint16_t>(page[offset + 1]) << 8;
  }

  VLLM_BYTE_V2_HOST_DEVICE static void store_u16(uint8_t* page, int offset,
                                                 uint16_t value) {
    page[offset] = static_cast<uint8_t>(value & 0xff);
    page[offset + 1] = static_cast<uint8_t>(value >> 8);
  }

  VLLM_BYTE_V2_HOST_DEVICE static constexpr int kv_head_meta_offset(
      int kv_head) {
    return PageHeaderBytes + kv_head * KvHeadMetaBytes;
  }

  VLLM_BYTE_V2_HOST_DEVICE static constexpr int k_fallback_mask_offset(
      int kv_head) {
    return kv_head_meta_offset(kv_head);
  }

  VLLM_BYTE_V2_HOST_DEVICE static constexpr int v_fallback_mask_offset(
      int kv_head) {
    return kv_head_meta_offset(kv_head) + 4;
  }

  VLLM_BYTE_V2_HOST_DEVICE static constexpr int k_outlier_mask_offset(
      int kv_head) {
    return kv_head_meta_offset(kv_head) + 24;
  }

  VLLM_BYTE_V2_HOST_DEVICE static constexpr int v_outlier_mask_offset(
      int kv_head) {
    return kv_head_meta_offset(kv_head) + 28;
  }

  VLLM_BYTE_V2_HOST_DEVICE static constexpr int k_tile_index(
      int dim_tile, int token_tile = 0) {
    return dim_tile * Policy::CodecTokenTilesPerAllocBlock + token_tile;
  }

  VLLM_BYTE_V2_HOST_DEVICE static constexpr int v_tile_index(
      int dim_tile, int token_tile = 0) {
    return token_tile * Policy::VDimTiles + dim_tile;
  }

  VLLM_BYTE_V2_HOST_DEVICE static constexpr int k_base_offset(
      int kv_head, int dim_tile, int token_tile = 0) {
    return kv_head_meta_offset(kv_head) + 8 +
           k_tile_index(dim_tile, token_tile);
  }

  VLLM_BYTE_V2_HOST_DEVICE static constexpr int v_base_offset(
      int kv_head, int dim_tile, int token_tile = 0) {
    return kv_head_meta_offset(kv_head) + 8 + Policy::CodecTilesPerKPage +
           v_tile_index(dim_tile, token_tile);
  }

  VLLM_BYTE_V2_HOST_DEVICE static constexpr int k_outlier_count_offset(
      int kv_head, int dim_tile, int token_tile = 0) {
    return kv_head_meta_offset(kv_head) + 32 +
           2 * k_tile_index(dim_tile, token_tile);
  }

  VLLM_BYTE_V2_HOST_DEVICE static constexpr int v_outlier_count_offset(
      int kv_head, int dim_tile, int token_tile = 0) {
    return kv_head_meta_offset(kv_head) + 32 +
           2 * (Policy::CodecTilesPerKPage +
                v_tile_index(dim_tile, token_tile));
  }

  VLLM_BYTE_V2_HOST_DEVICE static constexpr int k_outlier_pool_index_offset(
      int kv_head, int dim_tile, int token_tile = 0) {
    return kv_head_meta_offset(kv_head) + 32 + 2 * TilesPerKvHead +
           2 * k_tile_index(dim_tile, token_tile);
  }

  VLLM_BYTE_V2_HOST_DEVICE static constexpr int v_outlier_pool_index_offset(
      int kv_head, int dim_tile, int token_tile = 0) {
    return kv_head_meta_offset(kv_head) + 32 + 2 * TilesPerKvHead +
           2 * (Policy::CodecTilesPerKPage +
                v_tile_index(dim_tile, token_tile));
  }

  VLLM_BYTE_V2_HOST_DEVICE static int k_outlier_count(const uint8_t* page,
                                                      int kv_head, int dim_tile,
                                                      int token_tile = 0) {
    return static_cast<int>(
        load_u16(page, k_outlier_count_offset(kv_head, dim_tile, token_tile)));
  }

  VLLM_BYTE_V2_HOST_DEVICE static int v_outlier_count(const uint8_t* page,
                                                      int kv_head, int dim_tile,
                                                      int token_tile = 0) {
    return static_cast<int>(
        load_u16(page, v_outlier_count_offset(kv_head, dim_tile, token_tile)));
  }

  VLLM_BYTE_V2_HOST_DEVICE static void set_k_outlier_count(
      uint8_t* page, int kv_head, int dim_tile, int token_tile, int count) {
    store_u16(page, k_outlier_count_offset(kv_head, dim_tile, token_tile),
              static_cast<uint16_t>(count));
  }

  VLLM_BYTE_V2_HOST_DEVICE static void set_v_outlier_count(
      uint8_t* page, int kv_head, int dim_tile, int token_tile, int count) {
    store_u16(page, v_outlier_count_offset(kv_head, dim_tile, token_tile),
              static_cast<uint16_t>(count));
  }

  VLLM_BYTE_V2_HOST_DEVICE static int k_outlier_pool_index(const uint8_t* page,
                                                           int kv_head,
                                                           int dim_tile,
                                                           int token_tile = 0) {
    return static_cast<int>(load_u16(
        page, k_outlier_pool_index_offset(kv_head, dim_tile, token_tile)));
  }

  VLLM_BYTE_V2_HOST_DEVICE static int v_outlier_pool_index(const uint8_t* page,
                                                           int kv_head,
                                                           int dim_tile,
                                                           int token_tile = 0) {
    return static_cast<int>(load_u16(
        page, v_outlier_pool_index_offset(kv_head, dim_tile, token_tile)));
  }

  VLLM_BYTE_V2_HOST_DEVICE static void set_k_outlier_pool_index(
      uint8_t* page, int kv_head, int dim_tile, int token_tile,
      int pool_index) {
    store_u16(page, k_outlier_pool_index_offset(kv_head, dim_tile, token_tile),
              static_cast<uint16_t>(pool_index));
  }

  VLLM_BYTE_V2_HOST_DEVICE static void set_v_outlier_pool_index(
      uint8_t* page, int kv_head, int dim_tile, int token_tile,
      int pool_index) {
    store_u16(page, v_outlier_pool_index_offset(kv_head, dim_tile, token_tile),
              static_cast<uint16_t>(pool_index));
  }

  VLLM_BYTE_V2_HOST_DEVICE static constexpr int k_payload_offset(
      int kv_head, int dim_tile, int token_tile = 0) {
    return KPayloadBaseBytes + kv_head * AlignedKPayloadBytesPerKvHead +
           k_tile_index(dim_tile, token_tile) * CodecPayloadBytesPerTile;
  }

  VLLM_BYTE_V2_HOST_DEVICE static constexpr int v_payload_offset(
      int kv_head, int dim_tile, int token_tile = 0) {
    return VPayloadBaseBytes + kv_head * AlignedVPayloadBytesPerKvHead +
           v_tile_index(dim_tile, token_tile) * CodecPayloadBytesPerTile;
  }

  VLLM_BYTE_V2_HOST_DEVICE static int k_outlier_payload_offset(
      const uint8_t* page, int kv_head, int dim_tile, int token_tile = 0,
      int entry_idx = 0) {
    return OutlierPoolBaseBytes +
           (k_outlier_pool_index(page, kv_head, dim_tile, token_tile) +
            entry_idx) *
               OutlierEntryBytes;
  }

  VLLM_BYTE_V2_HOST_DEVICE static int v_outlier_payload_offset(
      const uint8_t* page, int kv_head, int dim_tile, int token_tile = 0,
      int entry_idx = 0) {
    return OutlierPoolBaseBytes +
           (v_outlier_pool_index(page, kv_head, dim_tile, token_tile) +
            entry_idx) *
               OutlierEntryBytes;
  }

  VLLM_BYTE_V2_HOST_DEVICE static constexpr int raw_key_offset(int, int, int) {
    return RawKPayloadBaseBytes;
  }

  VLLM_BYTE_V2_HOST_DEVICE static constexpr int raw_value_offset(int, int,
                                                                 int) {
    return RawVPayloadBaseBytes;
  }
};

template <typename Policy = DefaultByteV2TilePolicy, int NumKvHeads = 8,
          int RawElementBytes = 2, int AlignmentBytes = 128>
struct ByteV2RawStagingLayout {
  static_assert(NumKvHeads > 0);
  static_assert(RawElementBytes > 0);
  static_assert(AlignmentBytes > 0);

  using TilePolicy = Policy;

  static constexpr int NumKvHeadsValue = NumKvHeads;
  static constexpr int RawElementBytesValue = RawElementBytes;

  static constexpr int align_up(int value) {
    return cdiv(value, AlignmentBytes) * AlignmentBytes;
  }

  static constexpr int KeyBytes =
      NumKvHeads * Policy::AllocBlockTokens * Policy::HeadDim * RawElementBytes;
  static constexpr int ValueBytes = NumKvHeads * Policy::AllocBlockTokens *
                                    Policy::HeadDimV * RawElementBytes;
  static constexpr int AlignedKeyBytes = align_up(KeyBytes);
  static constexpr int AlignedValueBytes = align_up(ValueBytes);
  static constexpr int ValueBaseBytes = AlignedKeyBytes;
  static constexpr int SlotSizeBytes = AlignedKeyBytes + AlignedValueBytes;

  VLLM_BYTE_V2_HOST_DEVICE static constexpr int key_offset(int kv_head, int row,
                                                           int dim) {
    return ((kv_head * Policy::AllocBlockTokens + row) * Policy::HeadDim +
            dim) *
           RawElementBytes;
  }

  VLLM_BYTE_V2_HOST_DEVICE static constexpr int value_offset(int kv_head,
                                                             int row, int dim) {
    return ValueBaseBytes +
           ((kv_head * Policy::AllocBlockTokens + row) * Policy::HeadDimV +
            dim) *
               RawElementBytes;
  }
};

template <typename Layout = ByteV2PageLayoutV4<>>
struct ByteV2MacroDesc {
  static_assert(Layout::MacroPages <= kByteV2MaxMacroPages);

  int32_t physical_blocks[kByteV2MaxMacroPages];
  uint16_t valid_rows[kByteV2MaxMacroPages];
  uint16_t compressed_mask;
  uint16_t outlier_page_mask;
  uint16_t reserved0;
  uint16_t reserved1;
  uint32_t k_payload_offsets[kByteV2MaxMacroPages];
  uint32_t v_payload_offsets[kByteV2MaxMacroPages];
  uint32_t reserved2;
  uint32_t reserved3;
};

static_assert(sizeof(ByteV2MacroDesc<>) == kByteV2MacroDescriptorBytes);

template <typename Layout = ByteV2PageLayoutV4<>>
struct ByteV2PagedKVManager {
  static_assert(Layout::MacroPages <= kByteV2MaxMacroPages);

  const int32_t* block_table;
  int64_t block_table_size;

  VLLM_BYTE_V2_HOST_DEVICE ByteV2MacroDesc<Layout> make_macro_desc(
      int first_block_idx, int seq_len, int kv_head,
      uint16_t compressed_mask_override = 0xffff,
      uint16_t outlier_page_mask = 0) const {
    ByteV2MacroDesc<Layout> desc{};
    desc.compressed_mask = 0;
    desc.outlier_page_mask = 0;
    desc.reserved0 = 0;
    desc.reserved1 = 0;
    desc.reserved2 = 0;
    desc.reserved3 = 0;

    for (int i = 0; i < kByteV2MaxMacroPages; ++i) {
      desc.physical_blocks[i] = -1;
      desc.valid_rows[i] = 0;
      desc.k_payload_offsets[i] = 0;
      desc.v_payload_offsets[i] = 0;
    }

    const auto k_payload_offset = Layout::k_payload_offset(kv_head, 0);
    const auto v_payload_offset = Layout::v_payload_offset(kv_head, 0);
    uint16_t valid_mask = 0;
    for (int page_idx = 0; page_idx < Layout::MacroPages; ++page_idx) {
      const int block_idx = first_block_idx + page_idx;
      const int token_start = block_idx * Layout::TilePolicy::AllocBlockTokens;
      int rows = seq_len - token_start;
      if (rows <= 0 || block_idx < 0 || block_idx >= block_table_size) {
        continue;
      }
      if (rows > Layout::TilePolicy::AllocBlockTokens) {
        rows = Layout::TilePolicy::AllocBlockTokens;
      }
      const int32_t physical_block = block_table[block_idx];
      if (physical_block < 0) {
        continue;
      }
      desc.physical_blocks[page_idx] = physical_block;
      desc.valid_rows[page_idx] = static_cast<uint16_t>(rows);
      desc.k_payload_offsets[page_idx] = k_payload_offset;
      desc.v_payload_offsets[page_idx] = v_payload_offset;
      valid_mask |= uint16_t{1} << page_idx;
    }

    desc.compressed_mask = compressed_mask_override == 0xffff
                               ? valid_mask
                               : compressed_mask_override & valid_mask;
    desc.outlier_page_mask = outlier_page_mask & valid_mask;
    return desc;
  }
};

}  // namespace vllm::byte_v2

#undef VLLM_BYTE_V2_HOST_DEVICE
