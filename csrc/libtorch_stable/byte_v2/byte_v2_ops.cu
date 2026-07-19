#include "byte_v2_layout.cuh"

#include "../async_util.cuh"
#include "../dispatch_utils.h"
#include "../torch_utils.h"

#include <torch/csrc/stable/ops.h>
#include <torch/csrc/stable/tensor.h>

#include <cfloat>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <cutlass/numeric_types.h>
#include <cute/tensor.hpp>

#include <limits>
#include <vector>

namespace {

using ByteV2DefaultLayout = vllm::byte_v2::ByteV2PageLayoutV5<>;
using ByteV2DefaultPolicy = ByteV2DefaultLayout::TilePolicy;
using ByteV2HighBytePayloadPolicy =
    vllm::byte_v2::ByteV2CodecPayloadPolicy<ByteV2DefaultPolicy, 1, 8>;
using ByteV2HighByteLayout =
    vllm::byte_v2::ByteV2PageLayoutV5<ByteV2DefaultPolicy,
                                      ByteV2HighBytePayloadPolicy>;
using ByteV2SidebandHighPayloadPolicy =
    vllm::byte_v2::ByteV2CodecPayloadPolicy<ByteV2DefaultPolicy, 1, 4, true>;
using ByteV2SidebandHighLayout =
    vllm::byte_v2::ByteV2PageLayoutV5<ByteV2DefaultPolicy,
                                      ByteV2SidebandHighPayloadPolicy>;
using ByteV2DefaultRawStagingLayout = vllm::byte_v2::ByteV2RawStagingLayout<>;
using ByteV2BN128Policy =
    vllm::byte_v2::ByteV2TilePolicy<16, 16, 16, 128, 128, 128>;
using ByteV2BN128Layout = vllm::byte_v2::ByteV2PageLayoutV5<ByteV2BN128Policy>;

template <int kBlockM_, int kBlockN_, int kHeadDim_, int kNWarps_>
struct ByteV2Fa2LikeCutePvTraits {
  using Element = cutlass::bfloat16_t;
  using ElementAccum = float;

  static constexpr int kBlockM = kBlockM_;
  static constexpr int kBlockN = kBlockN_;
  static constexpr int kHeadDim = kHeadDim_;
  static constexpr int kNWarps = kNWarps_;
  static constexpr int kNThreads = kNWarps * 32;
  static constexpr int kBlockKSmem = kHeadDim % 64 == 0 ? 64 : 32;
  static constexpr int kSwizzle = kBlockKSmem == 32 ? 2 : 3;

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  using MmaAtomArch = cute::MMA_Atom<cute::SM80_16x8x16_F32BF16BF16F32_TN>;
#else
  // Keep the CUTE skeleton compilable for the current sm75 build target.
  // The actual ByteV2 BF16 tensor-core path must be enabled only for sm80+.
  using MmaAtomArch = cute::MMA_Atom<cute::SM75_16x8x8_F32F16F16F32_TN>;
#endif

  using TiledMma = cute::TiledMMA<
      MmaAtomArch,
      cute::Layout<cute::Shape<cute::Int<kNWarps>, cute::_1, cute::_1>>,
      cute::Tile<cute::Int<16 * kNWarps>, cute::_16, cute::_16>>;

  using SmemLayoutAtom = decltype(cute::composition(
      cute::Swizzle<kSwizzle, 3, 3>{},
      cute::Layout<cute::Shape<cute::_8, cute::Int<kBlockKSmem>>,
                   cute::Stride<cute::Int<kBlockKSmem>, cute::_1>>{}));
  using SmemLayoutKV = decltype(cute::tile_to_shape(
      SmemLayoutAtom{},
      cute::Shape<cute::Int<kBlockN>, cute::Int<kHeadDim>>{}));
  using SmemLayoutVtransposed = decltype(cute::composition(
      SmemLayoutKV{},
      cute::make_layout(cute::Shape<cute::Int<kHeadDim>, cute::Int<kBlockN>>{},
                        cute::GenRowMajor{})));
  using SmemLayoutVtransposedNoSwizzle =
      decltype(cute::get_nonswizzle_portion(SmemLayoutVtransposed{}));
  using SmemCopyAtomV = cute::Copy_Atom<cute::SM75_U16x8_LDSM_T, Element>;
  static constexpr int kVStorageElems = cute::size(SmemLayoutKV{});
};

using ByteV2Fa2LikeCutePvProbe = ByteV2Fa2LikeCutePvTraits<16, 64, 128, 1>;
static_assert(ByteV2Fa2LikeCutePvProbe::kBlockM == 16);
static_assert(ByteV2Fa2LikeCutePvProbe::kNThreads == 32);
static_assert(ByteV2Fa2LikeCutePvProbe::kBlockKSmem == 64);

template <int kBlockM_, int kBlockK_, int kBlockN_>
struct ByteV2Fa2LikeCutePvTileTraits {
  using Element = cutlass::bfloat16_t;
  using ElementAccum = float;

  static constexpr int kBlockM = kBlockM_;
  static constexpr int kBlockK = kBlockK_;
  static constexpr int kBlockN = kBlockN_;
  static constexpr int kNThreads = 32;
  static constexpr int kVBlockKSmem = kBlockN;
  static constexpr int kVSwizzle =
      kVBlockKSmem == 16 ? 1 : (kVBlockKSmem == 32 ? 2 : 3);

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  using MmaAtomArch = cute::MMA_Atom<cute::SM80_16x8x16_F32BF16BF16F32_TN>;
#else
  using MmaAtomArch = cute::MMA_Atom<cute::SM75_16x8x8_F32F16F16F32_TN>;
#endif

  using TiledMma = cute::TiledMMA<
      MmaAtomArch, cute::Layout<cute::Shape<cute::_1, cute::_1, cute::_1>>,
      cute::Tile<cute::Int<kBlockM>, cute::Int<kBlockN>, cute::_16>>;
  using SmemLayoutAtomV = decltype(cute::composition(
      cute::Swizzle<kVSwizzle, 3, 3>{},
      cute::Layout<cute::Shape<cute::_8, cute::Int<kVBlockKSmem>>,
                   cute::Stride<cute::Int<kVBlockKSmem>, cute::_1>>{}));
  using SmemLayoutV = decltype(cute::tile_to_shape(
      SmemLayoutAtomV{},
      cute::Shape<cute::Int<kBlockK>, cute::Int<kBlockN>>{}));
  using SmemLayoutVt = decltype(cute::composition(
      SmemLayoutV{},
      cute::make_layout(cute::Shape<cute::Int<kBlockN>, cute::Int<kBlockK>>{},
                        cute::GenRowMajor{})));
  using SmemLayoutVtNoSwizzle =
      decltype(cute::get_nonswizzle_portion(SmemLayoutVt{}));
  using SmemCopyAtomV = cute::Copy_Atom<cute::SM75_U16x8_LDSM_T, Element>;
  static constexpr int kVStorageElems = cute::size(SmemLayoutV{});
};

using ByteV2Fa2LikeCutePvTileProbe = ByteV2Fa2LikeCutePvTileTraits<16, 64, 16>;
static_assert(ByteV2Fa2LikeCutePvTileProbe::kNThreads == 32);

template <int kBlockM_, int kBlockN_, int kHeadDim_, int kNWarps_>
struct ByteV2Fa2LikeCuteQkTraits {
  using Element = cutlass::bfloat16_t;
  using ElementAccum = float;

  static constexpr int kBlockM = kBlockM_;
  static constexpr int kBlockN = kBlockN_;
  static constexpr int kHeadDim = kHeadDim_;
  static constexpr int kNWarps = kNWarps_;
  static constexpr int kBlockKSmem =
      kHeadDim % 64 == 0 ? 64 : (kHeadDim % 32 == 0 ? 32 : 16);
  static constexpr int kSwizzle =
      kBlockKSmem == 16 ? 1 : (kBlockKSmem == 32 ? 2 : 3);

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  using MmaAtomArch = cute::MMA_Atom<cute::SM80_16x8x16_F32BF16BF16F32_TN>;
#else
  using MmaAtomArch = cute::MMA_Atom<cute::SM75_16x8x8_F32F16F16F32_TN>;
#endif

  using TiledMma = cute::TiledMMA<
      MmaAtomArch,
      cute::Layout<cute::Shape<cute::Int<kNWarps>, cute::_1, cute::_1>>,
      cute::Tile<cute::Int<16 * kNWarps>, cute::_16, cute::_16>>;
  using SmemLayoutAtomQ = decltype(cute::composition(
      cute::Swizzle<kSwizzle, 3, 3>{},
      cute::Layout<cute::Shape<cute::_8, cute::Int<kBlockKSmem>>,
                   cute::Stride<cute::Int<kBlockKSmem>, cute::_1>>{}));
  using SmemLayoutQ = decltype(cute::tile_to_shape(
      SmemLayoutAtomQ{},
      cute::Shape<cute::Int<kBlockM>, cute::Int<kHeadDim>>{}));
  using SmemLayoutKV = decltype(cute::tile_to_shape(
      SmemLayoutAtomQ{},
      cute::Shape<cute::Int<kBlockN>, cute::Int<kHeadDim>>{}));
  using SmemCopyAtom = cute::Copy_Atom<cute::SM75_U32x4_LDSM_N, Element>;
  static constexpr int kQStorageElems = cute::size(SmemLayoutQ{});
  static constexpr int kKStorageElems = cute::size(SmemLayoutKV{});
};

__device__ __forceinline__ cutlass::bfloat16_t
byte_v2_bf16_bits_to_cutlass_bfloat16(uint16_t bits) {
  cutlass::bfloat16_t value;
  *reinterpret_cast<uint16_t*>(&value) = bits;
  return value;
}

__device__ __forceinline__ cutlass::bfloat16_t
byte_v2_float_to_cutlass_bfloat16(float value) {
  union {
    float f32;
    uint32_t u32;
  } in;
  in.f32 = value;
  const uint32_t lsb = (in.u32 >> 16) & 1;
  const uint32_t rounding_bias = 0x7fff + lsb;
  return byte_v2_bf16_bits_to_cutlass_bfloat16(
      static_cast<uint16_t>((in.u32 + rounding_bias) >> 16));
}

template <typename Tensor0, typename Tensor1, typename Tensor2,
          typename Tensor3, typename Tensor4, typename TiledMma,
          typename TiledCopyA, typename TiledCopyB, typename ThrCopyA,
          typename ThrCopyB>
__device__ __forceinline__ void byte_v2_cute_gemm_smem(
    Tensor0& acc, Tensor1& tCrA, Tensor2& tCrB, const Tensor3& tCsA,
    const Tensor4& tCsB, TiledMma tiled_mma, TiledCopyA smem_tiled_copy_A,
    TiledCopyB smem_tiled_copy_B, ThrCopyA smem_thr_copy_A,
    ThrCopyB smem_thr_copy_B) {
  using namespace cute;
  auto tCrA_copy_view = smem_thr_copy_A.retile_D(tCrA);
  auto tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
  cute::copy(smem_tiled_copy_A, tCsA(_, _, _0{}), tCrA_copy_view(_, _, _0{}));
  cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tCrB_copy_view(_, _, _0{}));
#pragma unroll
  for (int i = 0; i < cute::size<2>(tCrA); ++i) {
    if (i < cute::size<2>(tCrA) - 1) {
      cute::copy(smem_tiled_copy_A, tCsA(_, _, i + 1),
                 tCrA_copy_view(_, _, i + 1));
      cute::copy(smem_tiled_copy_B, tCsB(_, _, i + 1),
                 tCrB_copy_view(_, _, i + 1));
    }
    cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
  }
}

template <typename MmaTraits, typename Layout>
__device__ __forceinline__ auto byte_v2_convert_layout_acc_Aregs(
    Layout acc_layout) {
  using namespace cute;
  using X = Underscore;
  static_assert(decltype(cute::size<0>(acc_layout))::value == 4);
  static_assert(decltype(cute::rank(acc_layout))::value == 3);
  constexpr int mma_shape_k = cute::get<2>(typename MmaTraits::Shape_MNK{});
  static_assert(mma_shape_k == 8 || mma_shape_k == 16);
  if constexpr (mma_shape_k == 8) {
    return acc_layout;
  } else {
    auto divided = cute::logical_divide(acc_layout, cute::Shape<X, X, _2>{});
    return cute::make_layout(
        cute::make_layout(cute::get<0>(divided), cute::get<2, 0>(divided)),
        cute::get<1>(divided), cute::get<2, 1>(divided));
  }
}

template <typename Layout>
__device__ __forceinline__ auto byte_v2_convert_layout_acc_rowcol(
    Layout acc_layout) {
  using namespace cute;
  static_assert(decltype(cute::size<0>(acc_layout))::value == 4);
  static_assert(decltype(cute::rank(acc_layout))::value == 3);
  auto divided = cute::logical_divide(acc_layout, cute::Shape<_2>{});
  return cute::make_layout(
      cute::make_layout(cute::get<0, 1>(divided), cute::get<1>(divided)),
      cute::make_layout(cute::get<0, 0>(divided), cute::get<2>(divided)));
}

__device__ __forceinline__ float byte_v2_quad_sum(float value) {
  value += __shfl_xor_sync(0xffffffff, value, 1);
  value += __shfl_xor_sync(0xffffffff, value, 2);
  return value;
}

__device__ __forceinline__ float byte_v2_quad_max(float value) {
  value = fmaxf(value, __shfl_xor_sync(0xffffffff, value, 1));
  value = fmaxf(value, __shfl_xor_sync(0xffffffff, value, 2));
  return value;
}

template <typename Tensor0, typename Tensor1, typename Tensor2,
          typename Tensor3, typename TiledMma, typename TiledCopyB,
          typename ThrCopyB>
__device__ __forceinline__ void byte_v2_cute_gemm_rs(
    Tensor0& acc, Tensor1& tCrA, Tensor2& tCrB, const Tensor3& tCsB,
    TiledMma tiled_mma, TiledCopyB smem_tiled_copy_B,
    ThrCopyB smem_thr_copy_B) {
  using namespace cute;
  CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));
  CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));
  CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));
  auto tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
  CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));
  cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tCrB_copy_view(_, _, _0{}));
#pragma unroll
  for (int i = 0; i < cute::size<2>(tCrA); ++i) {
    if (i < cute::size<2>(tCrA) - 1) {
      cute::copy(smem_tiled_copy_B, tCsB(_, _, i + 1),
                 tCrB_copy_view(_, _, i + 1));
    }
    cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
  }
}

static_assert(ByteV2DefaultPolicy::CodecTokenBlock == 16);
static_assert(ByteV2DefaultPolicy::CodecDimBlock == 16);
static_assert(ByteV2DefaultPolicy::AllocBlockTokens == 16);
static_assert(ByteV2DefaultPolicy::HeadDim == 128);
static_assert(ByteV2DefaultPolicy::HeadDimV == 128);
static_assert(ByteV2DefaultRawStagingLayout::SlotSizeBytes == 65536);
static_assert(ByteV2HighByteLayout::CodecPayloadBytesPerTile ==
              ByteV2DefaultPolicy::CodecTileElems * 2);
static_assert(ByteV2HighByteLayout::PageSizeBytes >
              ByteV2DefaultLayout::PageSizeBytes);
static_assert(ByteV2SidebandHighLayout::PageSizeBytes ==
              ByteV2DefaultLayout::PageSizeBytes);
static_assert(ByteV2SidebandHighLayout::CodecOutlierHighSideband);
static_assert(ByteV2BN128Layout::MacroPages == 8);
static_assert(ByteV2BN128Layout::PageSizeBytes ==
              ByteV2DefaultLayout::PageSizeBytes);

enum ByteV2DirectDiagnosticMode : int {
  kByteV2DirectDiagnosticCurrent = 0,
  kByteV2DirectDiagnosticDecodeStageOnly = 1,
  kByteV2DirectDiagnosticFakeDecodeZero = 2,
  kByteV2DirectDiagnosticRawSameSkeleton = 3,
  kByteV2DirectDiagnosticKDecodeStageOnly = 4,
  kByteV2DirectDiagnosticVDecodeStageOnly = 5,
  kByteV2DirectDiagnosticKDecodeQkOnly = 6,
  kByteV2DirectDiagnosticVDecodePvOnly = 7,
  kByteV2DirectDiagnosticKDecodeProducer3StageOnly = 8,
  kByteV2DirectDiagnosticVDecodeProducer3StageOnly = 9,
  kByteV2DirectDiagnosticKDecodeProducer2StageOnly = 10,
  kByteV2DirectDiagnosticVDecodeProducer2StageOnly = 11,
  kByteV2DirectDiagnosticKDecodeNoStoreStageOnly = 12,
  kByteV2DirectDiagnosticVDecodeNoStoreStageOnly = 13,
  kByteV2DirectDiagnosticKDecodeLowOnlyStageOnly = 14,
  kByteV2DirectDiagnosticVDecodeLowOnlyStageOnly = 15,
  kByteV2DirectDiagnosticKDecodeHighOnlyStageOnly = 16,
  kByteV2DirectDiagnosticVDecodeHighOnlyStageOnly = 17,
  kByteV2DirectDiagnosticPhaseProfile = 18,
  kByteV2DirectDiagnosticStageWindow16 = 19,
  kByteV2DirectDiagnosticStageWindow32 = 20,
  kByteV2DirectDiagnosticStageWindow16Specialized = 21,
  kByteV2DirectDiagnosticQkWindow16Profile = 22,
  kByteV2DirectDiagnosticEffectiveM16Profile = 23,
  kByteV2DirectDiagnosticVDecodePvNoGemmOnly = 24,
  kByteV2DirectDiagnosticVDecodePvNoAccumOnly = 25,
  kByteV2DirectDiagnosticKDecodeQkNoOutput = 26,
  kByteV2DirectDiagnosticUnnormalizedPartitionOutput = 27,
  kByteV2DirectDiagnosticCodeReady = 28,
};

enum ByteV2PayloadFormat : int {
  kByteV2PayloadFormatDefault = 0,
  kByteV2PayloadFormatHighByte = 1,
  kByteV2PayloadFormatSidebandHigh = 2,
};

constexpr int kByteV2PageUnsafeAny = 1;
constexpr int kByteV2PageUnsafeK = 2;
constexpr int kByteV2PageUnsafeV = 4;

void check_byte_v2_tile_policy(const std::vector<int64_t>& tile_policy) {
  STD_TORCH_CHECK(tile_policy.size() >= 6 && tile_policy.size() <= 16,
                  "ByteV2 tile_policy must contain 6 to 16 values");
  const int64_t codec_token_block = tile_policy[0];
  const int64_t codec_dim_block = tile_policy[1];
  const int64_t alloc_block_tokens = tile_policy[2];
  const int64_t compute_block_n = tile_policy[3];
  const int64_t head_dim = tile_policy[4];
  const int64_t head_dim_v = tile_policy[5];

  STD_TORCH_CHECK(codec_token_block > 0, "codec_token_block must be positive");
  STD_TORCH_CHECK(codec_dim_block > 0, "codec_dim_block must be positive");
  STD_TORCH_CHECK(alloc_block_tokens > 0,
                  "alloc_block_tokens must be positive");
  STD_TORCH_CHECK(compute_block_n > 0, "compute_block_n must be positive");
  STD_TORCH_CHECK(head_dim > 0, "head_dim must be positive");
  STD_TORCH_CHECK(head_dim_v > 0, "head_dim_v must be positive");
  STD_TORCH_CHECK(alloc_block_tokens % codec_token_block == 0,
                  "alloc_block_tokens must be divisible by codec_token_block");
  STD_TORCH_CHECK(head_dim % codec_dim_block == 0,
                  "head_dim must be divisible by codec_dim_block");
  STD_TORCH_CHECK(head_dim_v % codec_dim_block == 0,
                  "head_dim_v must be divisible by codec_dim_block");
  STD_TORCH_CHECK(compute_block_n % alloc_block_tokens == 0,
                  "compute_block_n must be divisible by alloc_block_tokens");
}

bool byte_v2_use_raw_fallback(const std::vector<int64_t>& tile_policy) {
  return tile_policy.size() >= 7 && tile_policy[6] != 0;
}

bool byte_v2_assume_no_fallback_no_outlier(
    const std::vector<int64_t>& tile_policy) {
  return tile_policy.size() >= 8 && tile_policy[7] != 0;
}

bool byte_v2_use_gqa_packed(const std::vector<int64_t>& tile_policy) {
  return tile_policy.size() >= 9 && tile_policy[8] != 0;
}

bool byte_v2_use_gqa_fa2_like(const std::vector<int64_t>& tile_policy) {
  return tile_policy.size() >= 10 && tile_policy[9] != 0;
}

bool byte_v2_use_gqa_fa2_qk_mma(const std::vector<int64_t>& tile_policy) {
  return tile_policy.size() >= 11 && tile_policy[10] != 0;
}

bool byte_v2_use_gqa_fa2_mainloop(const std::vector<int64_t>& tile_policy) {
  return tile_policy.size() >= 12 && tile_policy[11] != 0;
}

bool byte_v2_use_gqa_fa2_multiwarp(const std::vector<int64_t>& tile_policy) {
  return tile_policy.size() >= 13 && tile_policy[12] != 0;
}

bool byte_v2_use_gqa_fa2_direct(const std::vector<int64_t>& tile_policy) {
  return tile_policy.size() >= 14 && tile_policy[13] != 0;
}

int byte_v2_gqa_fa2_direct_diagnostic_mode(
    const std::vector<int64_t>& tile_policy) {
  const int mode = tile_policy.size() >= 15 ? static_cast<int>(tile_policy[14])
                                            : kByteV2DirectDiagnosticCurrent;
  STD_TORCH_CHECK(mode >= kByteV2DirectDiagnosticCurrent &&
                      mode <= kByteV2DirectDiagnosticCodeReady,
                  "ByteV2 FA2-direct diagnostic mode must be in [0, 28]");
  return mode;
}

int byte_v2_payload_format(const std::vector<int64_t>& tile_policy) {
  const int format = tile_policy.size() >= 16
                         ? static_cast<int>(tile_policy[15])
                         : kByteV2PayloadFormatDefault;
  STD_TORCH_CHECK(format == kByteV2PayloadFormatDefault ||
                      format == kByteV2PayloadFormatHighByte ||
                      format == kByteV2PayloadFormatSidebandHigh,
                  "ByteV2 payload format must be 0 (default), 1 "
                  "(high-byte upper-bound), or 2 (outlier high sideband)");
  return format;
}

__device__ __forceinline__ uint8_t byte_v2_code_nibble(uint16_t bits) {
  const uint8_t high = static_cast<uint8_t>(bits >> 8);
  const uint8_t sign = static_cast<uint8_t>(high >> 7);
  const uint8_t high7 = static_cast<uint8_t>(high & 0x7f);
  return static_cast<uint8_t>((sign << 3) | (high7 & 0x07));
}

__device__ __forceinline__ uint8_t byte_v2_delta_code_nibble(uint16_t bits,
                                                             int base) {
  const uint8_t high = static_cast<uint8_t>(bits >> 8);
  const uint8_t sign = static_cast<uint8_t>(high >> 7);
  const uint8_t high7 = static_cast<uint8_t>(high & 0x7f);
  return static_cast<uint8_t>((sign << 3) | ((high7 - base) & 0x07));
}

__device__ __forceinline__ uint8_t byte_v2_high7(uint16_t bits) {
  return static_cast<uint8_t>((bits >> 8) & 0x7f);
}

__device__ __forceinline__ uint8_t
byte_v2_high_byte_from_base_and_code(uint8_t base, uint8_t code) {
  const uint8_t sign = static_cast<uint8_t>(code >> 3);
  const uint8_t delta = static_cast<uint8_t>(code & 0x07);
  return static_cast<uint8_t>((sign << 7) | (base + delta));
}

__device__ __forceinline__ bool byte_v2_high7_in_window(uint16_t bits,
                                                        int base) {
  const int high7 = static_cast<int>(byte_v2_high7(bits));
  return high7 >= base && high7 < base + 8;
}

__device__ __forceinline__ float byte_v2_bf16_bits_to_float(uint16_t bits) {
  union {
    uint32_t u32;
    float f32;
  } value;
  value.u32 = static_cast<uint32_t>(bits) << 16;
  return value.f32;
}

__device__ __forceinline__ uint16_t byte_v2_float_to_bf16_bits(float value) {
  union {
    float f32;
    uint32_t u32;
  } in;
  in.f32 = value;
  const uint32_t lsb = (in.u32 >> 16) & 1;
  const uint32_t rounding_bias = 0x7fff + lsb;
  return static_cast<uint16_t>((in.u32 + rounding_bias) >> 16);
}

__device__ __forceinline__ uint16_t
byte_v2_load_u16_bytes(const uint8_t* __restrict__ base, int64_t offset) {
  return static_cast<uint16_t>(base[offset]) |
         static_cast<uint16_t>(base[offset + 1]) << 8;
}

__device__ __forceinline__ uint16_t
byte_v2_load_aligned_u16(const uint8_t* __restrict__ base, int64_t offset) {
  return *reinterpret_cast<const uint16_t*>(base + offset);
}

__device__ __forceinline__ uint64_t
byte_v2_load_aligned_u64(const uint8_t* __restrict__ base, int64_t offset) {
  return *reinterpret_cast<const uint64_t*>(base + offset);
}

__device__ __forceinline__ uint32_t
byte_v2_load_u32_bytes(const uint8_t* __restrict__ base, int64_t offset) {
  return static_cast<uint32_t>(base[offset]) |
         (static_cast<uint32_t>(base[offset + 1]) << 8) |
         (static_cast<uint32_t>(base[offset + 2]) << 16) |
         (static_cast<uint32_t>(base[offset + 3]) << 24);
}

__device__ __forceinline__ void byte_v2_store_u16_bytes(
    uint8_t* __restrict__ base, int64_t offset, uint16_t bits) {
  base[offset] = static_cast<uint8_t>(bits & 0xff);
  base[offset + 1] = static_cast<uint8_t>(bits >> 8);
}

__device__ __forceinline__ int byte_v2_outlier_segment_capacity(int count) {
  int capacity = 1;
  while (capacity < count) {
    capacity <<= 1;
  }
  return capacity;
}

template <typename Layout>
__device__ __forceinline__ int byte_v2_allocate_outlier_segment(
    uint8_t* __restrict__ page, int count) {
  if constexpr (!Layout::PagePooledOutliersValue) {
    return 0;
  } else {
    const int capacity = byte_v2_outlier_segment_capacity(count);
    const unsigned int pool_index = atomicAdd(
        reinterpret_cast<unsigned int*>(page + Layout::OutlierPoolUsedOffset),
        static_cast<unsigned int>(capacity));
    if (pool_index + capacity > Layout::OutlierPoolEntriesValue) {
      atomicExch(reinterpret_cast<unsigned int*>(
                     page + Layout::OutlierPoolOverflowOffset),
                 1u);
      __trap();
    }
    return static_cast<int>(pool_index);
  }
}

template <typename Layout>
__device__ __forceinline__ void byte_v2_set_outlier_descriptor(
    uint8_t* __restrict__ page, int kv_side, int kv_head, int dim_tile,
    int token_tile, int count, int pool_index) {
  if (kv_side == 0) {
    Layout::set_k_outlier_count(page, kv_head, dim_tile, token_tile, count);
    if constexpr (Layout::PagePooledOutliersValue) {
      Layout::set_k_outlier_pool_index(page, kv_head, dim_tile, token_tile,
                                       pool_index);
    }
  } else {
    Layout::set_v_outlier_count(page, kv_head, dim_tile, token_tile, count);
    if constexpr (Layout::PagePooledOutliersValue) {
      Layout::set_v_outlier_pool_index(page, kv_head, dim_tile, token_tile,
                                       pool_index);
    }
  }
}

struct ByteV2PageUnsafeStats {
  int fallback_tiles;
  int outlier_tiles;
  int k_unsafe_tiles;
  int v_unsafe_tiles;
};

__device__ __forceinline__ ByteV2PageUnsafeStats
byte_v2_collect_page_unsafe_stats(const uint8_t* __restrict__ page) {
  constexpr uint32_t kKTileMask =
      (uint32_t{1} << ByteV2DefaultPolicy::CodecTilesPerKPage) - 1;
  constexpr uint32_t kVTileMask =
      (uint32_t{1} << ByteV2DefaultPolicy::CodecTilesPerVPage) - 1;

  ByteV2PageUnsafeStats stats{0, 0, 0, 0};
#pragma unroll
  for (int kv_head = 0; kv_head < ByteV2DefaultLayout::NumKvHeadsValue;
       ++kv_head) {
    const uint32_t k_fallback =
        byte_v2_load_u32_bytes(
            page, ByteV2DefaultLayout::k_fallback_mask_offset(kv_head)) &
        kKTileMask;
    const uint32_t v_fallback =
        byte_v2_load_u32_bytes(
            page, ByteV2DefaultLayout::v_fallback_mask_offset(kv_head)) &
        kVTileMask;
    const uint32_t k_outlier =
        byte_v2_load_u32_bytes(
            page, ByteV2DefaultLayout::k_outlier_mask_offset(kv_head)) &
        kKTileMask;
    const uint32_t v_outlier =
        byte_v2_load_u32_bytes(
            page, ByteV2DefaultLayout::v_outlier_mask_offset(kv_head)) &
        kVTileMask;
    stats.fallback_tiles += __popc(k_fallback) + __popc(v_fallback);
    stats.outlier_tiles += __popc(k_outlier) + __popc(v_outlier);
    stats.k_unsafe_tiles += __popc(k_fallback | k_outlier);
    stats.v_unsafe_tiles += __popc(v_fallback | v_outlier);
  }
  return stats;
}

__device__ __forceinline__ int byte_v2_page_unsafe_flag_from_stats(
    const ByteV2PageUnsafeStats& stats) {
  int flag = 0;
  if (stats.k_unsafe_tiles != 0) {
    flag |= kByteV2PageUnsafeK;
  }
  if (stats.v_unsafe_tiles != 0) {
    flag |= kByteV2PageUnsafeV;
  }
  if (flag != 0) {
    flag |= kByteV2PageUnsafeAny;
  }
  return flag;
}

__device__ __forceinline__ int byte_v2_page_unsafe_flag(
    const uint8_t* __restrict__ page) {
  return byte_v2_page_unsafe_flag_from_stats(
      byte_v2_collect_page_unsafe_stats(page));
}

__device__ __forceinline__ int byte_v2_page_is_unsafe(
    const uint8_t* __restrict__ page) {
  return byte_v2_page_unsafe_flag(page) != 0;
}

template <typename Layout, bool IsValue>
__device__ __forceinline__ float byte_v2_load_raw_elem(
    const uint8_t* __restrict__ page, int kv_head, int row, int dim);

template <typename Layout, bool IsValue>
__device__ __forceinline__ float byte_v2_load_payload_elem(
    const uint8_t* __restrict__ page, uint32_t payload_base_offset, int kv_head,
    int row_in_page, int dim) {
  using Policy = typename Layout::TilePolicy;

  constexpr int kCodecDimBlock = Policy::CodecDimBlock;
  constexpr int kCodecTileElems = Policy::CodecTileElems;
  constexpr int kPayloadBytesPerTile = Layout::CodecPayloadBytesPerTile;

  const int dim_tile = dim / kCodecDimBlock;
  const int dim_in_tile = dim % kCodecDimBlock;
  const int token_tile = row_in_page / Policy::CodecTokenBlock;
  const int row_in_tile = row_in_page % Policy::CodecTokenBlock;
  const int elem_idx = row_in_tile * kCodecDimBlock + dim_in_tile;
  int codec_tile_idx;
  if constexpr (IsValue) {
    codec_tile_idx = token_tile * Policy::VDimTiles + dim_tile;
  } else {
    codec_tile_idx =
        dim_tile * Policy::CodecTokenTilesPerAllocBlock + token_tile;
  }
  const uint32_t fallback_mask =
      IsValue ? byte_v2_load_u32_bytes(page,
                                       Layout::v_fallback_mask_offset(kv_head))
              : byte_v2_load_u32_bytes(page,
                                       Layout::k_fallback_mask_offset(kv_head));
  if ((fallback_mask & (uint32_t{1} << codec_tile_idx)) != 0) {
    if constexpr (Layout::IncludeRawPayloadValue) {
      return byte_v2_load_raw_elem<Layout, IsValue>(page, kv_head, row_in_page,
                                                    dim);
    }
  }

  const int tile_offset =
      payload_base_offset + codec_tile_idx * kPayloadBytesPerTile;
  const uint8_t low = page[tile_offset + elem_idx];
  const uint8_t packed_code =
      page[tile_offset + kCodecTileElems + elem_idx / 2];
  const uint8_t code =
      (elem_idx & 1) ? (packed_code >> 4) : (packed_code & 0x0f);
  const uint8_t base =
      IsValue ? page[Layout::v_base_offset(kv_head, dim_tile, token_tile)]
              : page[Layout::k_base_offset(kv_head, dim_tile, token_tile)];
  uint8_t high = byte_v2_high_byte_from_base_and_code(base, code);

  const uint32_t outlier_mask =
      IsValue
          ? byte_v2_load_u32_bytes(page, Layout::v_outlier_mask_offset(kv_head))
          : byte_v2_load_u32_bytes(page,
                                   Layout::k_outlier_mask_offset(kv_head));
  if ((outlier_mask & (uint32_t{1} << codec_tile_idx)) != 0) {
    const int outlier_count =
        IsValue ? Layout::v_outlier_count(page, kv_head, dim_tile, token_tile)
                : Layout::k_outlier_count(page, kv_head, dim_tile, token_tile);
    const int outlier_base_offset =
        IsValue ? Layout::v_outlier_payload_offset(page, kv_head, dim_tile,
                                                   token_tile)
                : Layout::k_outlier_payload_offset(page, kv_head, dim_tile,
                                                   token_tile);
    if constexpr (Layout::CodecOutlierHighSideband) {
      high = page[outlier_base_offset + elem_idx];
      return byte_v2_bf16_bits_to_float((static_cast<uint16_t>(high) << 8) |
                                        low);
    }
    if (outlier_count == Layout::OutlierEntriesPerTileValue - 1 ||
        outlier_count > elem_idx) {
      const uint16_t direct_entry = byte_v2_load_u16_bytes(
          page, outlier_base_offset + elem_idx * Layout::OutlierEntryBytes);
      if (Layout::OutlierEntryPolicy::decode_elem_index(direct_entry) ==
          static_cast<uint32_t>(elem_idx)) {
        high = static_cast<uint8_t>(
            Layout::OutlierEntryPolicy::decode_value_bits(direct_entry));
        return byte_v2_bf16_bits_to_float((static_cast<uint16_t>(high) << 8) |
                                          low);
      }
    }
#pragma unroll 1
    for (int entry_idx = 0; entry_idx < outlier_count; ++entry_idx) {
      const uint16_t entry = byte_v2_load_u16_bytes(
          page, outlier_base_offset + entry_idx * Layout::OutlierEntryBytes);
      if (Layout::OutlierEntryPolicy::decode_elem_index(entry) ==
          static_cast<uint32_t>(elem_idx)) {
        high = static_cast<uint8_t>(
            Layout::OutlierEntryPolicy::decode_value_bits(entry));
        break;
      }
    }
  }

  return byte_v2_bf16_bits_to_float((static_cast<uint16_t>(high) << 8) | low);
}

template <typename Layout, bool IsValue>
__device__ __forceinline__ uint16_t
byte_v2_load_payload_elem_no_fallback_no_outlier_bits(
    const uint8_t* __restrict__ page, uint32_t payload_base_offset, int kv_head,
    int row_in_page, int dim) {
  using Policy = typename Layout::TilePolicy;

  constexpr int kCodecDimBlock = Policy::CodecDimBlock;
  constexpr int kCodecTileElems = Policy::CodecTileElems;
  constexpr int kPayloadBytesPerTile = Layout::CodecPayloadBytesPerTile;

  const int dim_tile = dim / kCodecDimBlock;
  const int dim_in_tile = dim % kCodecDimBlock;
  const int token_tile = row_in_page / Policy::CodecTokenBlock;
  const int row_in_tile = row_in_page % Policy::CodecTokenBlock;
  const int elem_idx = row_in_tile * kCodecDimBlock + dim_in_tile;
  int codec_tile_idx;
  if constexpr (IsValue) {
    codec_tile_idx = token_tile * Policy::VDimTiles + dim_tile;
  } else {
    codec_tile_idx =
        dim_tile * Policy::CodecTokenTilesPerAllocBlock + token_tile;
  }

  const int tile_offset =
      payload_base_offset + codec_tile_idx * kPayloadBytesPerTile;
  const uint8_t low = page[tile_offset + elem_idx];
  const uint8_t packed_code =
      page[tile_offset + kCodecTileElems + elem_idx / 2];
  const uint8_t code =
      (elem_idx & 1) ? (packed_code >> 4) : (packed_code & 0x0f);
  const uint8_t base =
      IsValue ? page[Layout::v_base_offset(kv_head, dim_tile, token_tile)]
              : page[Layout::k_base_offset(kv_head, dim_tile, token_tile)];
  const uint8_t high = byte_v2_high_byte_from_base_and_code(base, code);

  return (static_cast<uint16_t>(high) << 8) | low;
}

template <typename Layout>
__device__ __forceinline__ void
byte_v2_load_k_payload_elem_pair_from_fixed_tile_no_outlier_bits(
    const uint8_t* __restrict__ page, uint32_t tile_offset, int row_in_page,
    int dim_in_tile_even, uint8_t base, uint16_t& bits0, uint16_t& bits1) {
  using Policy = typename Layout::TilePolicy;

  constexpr int kCodecDimBlock = Policy::CodecDimBlock;
  constexpr int kCodecTileElems = Policy::CodecTileElems;

  const int elem_idx = row_in_page * kCodecDimBlock + dim_in_tile_even;
  const uint16_t low_pair =
      byte_v2_load_aligned_u16(page, tile_offset + elem_idx);
  const uint8_t packed_code =
      page[tile_offset + kCodecTileElems + elem_idx / 2];
  const uint8_t high0 =
      byte_v2_high_byte_from_base_and_code(base, packed_code & 0x0f);
  const uint8_t high1 =
      byte_v2_high_byte_from_base_and_code(base, packed_code >> 4);

  bits0 = (static_cast<uint16_t>(high0) << 8) | (low_pair & 0x00ff);
  bits1 = (static_cast<uint16_t>(high1) << 8) | (low_pair >> 8);
}

template <typename Layout, bool IsValue>
__device__ __forceinline__ float
byte_v2_load_payload_elem_no_fallback_no_outlier(
    const uint8_t* __restrict__ page, uint32_t payload_base_offset, int kv_head,
    int row_in_page, int dim) {
  return byte_v2_bf16_bits_to_float(
      byte_v2_load_payload_elem_no_fallback_no_outlier_bits<Layout, IsValue>(
          page, payload_base_offset, kv_head, row_in_page, dim));
}

template <typename Layout, bool IsValue>
__device__ __forceinline__ float byte_v2_load_raw_elem(
    const uint8_t* __restrict__ page, int kv_head, int row, int dim) {
  const int64_t offset = IsValue ? Layout::raw_value_offset(kv_head, row, dim)
                                 : Layout::raw_key_offset(kv_head, row, dim);
  return byte_v2_bf16_bits_to_float(byte_v2_load_u16_bytes(page, offset));
}

template <typename Layout, bool IsValue, bool UseRawFallback,
          bool AssumeNoFallbackNoOutlier>
__device__ __forceinline__ float byte_v2_load_decode_elem(
    const uint8_t* __restrict__ page, uint32_t payload_base_offset, int kv_head,
    int row, int dim) {
  if constexpr (UseRawFallback) {
    return byte_v2_load_raw_elem<Layout, IsValue>(page, kv_head, row, dim);
  } else if constexpr (AssumeNoFallbackNoOutlier) {
    return byte_v2_load_payload_elem_no_fallback_no_outlier<Layout, IsValue>(
        page, payload_base_offset, kv_head, row, dim);
  } else {
    return byte_v2_load_payload_elem<Layout, IsValue>(page, payload_base_offset,
                                                      kv_head, row, dim);
  }
}

template <typename Layout, bool IsValue, bool UseRawFallback,
          bool AssumeNoFallbackNoOutlier, bool UsePageUnsafeFlags>
__device__ __forceinline__ float byte_v2_load_decode_elem_guarded(
    const uint8_t* __restrict__ page, uint32_t payload_base_offset, int kv_head,
    int row, int dim, bool page_is_unsafe) {
  if constexpr (UseRawFallback) {
    return byte_v2_load_raw_elem<Layout, IsValue>(page, kv_head, row, dim);
  } else if constexpr (AssumeNoFallbackNoOutlier) {
    if constexpr (UsePageUnsafeFlags) {
      if (page_is_unsafe) {
        return byte_v2_load_payload_elem<Layout, IsValue>(
            page, payload_base_offset, kv_head, row, dim);
      }
    }
    return byte_v2_load_payload_elem_no_fallback_no_outlier<Layout, IsValue>(
        page, payload_base_offset, kv_head, row, dim);
  } else {
    return byte_v2_load_payload_elem<Layout, IsValue>(page, payload_base_offset,
                                                      kv_head, row, dim);
  }
}

template <typename Layout>
struct ByteV2PayloadTileDescriptor {
  const uint8_t* page;
  uint32_t payload_offset;
  int outlier_payload_offset;
  int outlier_count;
  uint8_t base;
  bool fallback_hit;
  bool outlier_hit;
};

template <typename Layout, bool IsValue>
__device__ __forceinline__ ByteV2PayloadTileDescriptor<Layout>
byte_v2_make_payload_tile_descriptor_from_masks(
    const uint8_t* __restrict__ page, uint32_t payload_base_offset, int kv_head,
    int dim_tile, uint32_t fallback_mask, uint32_t outlier_mask) {
  using Policy = typename Layout::TilePolicy;
  static_assert(Policy::AllocBlockTokens == Policy::CodecTokenBlock);

  constexpr int kPayloadBytesPerTile = Layout::CodecPayloadBytesPerTile;
  const int codec_tile_idx = dim_tile;
  const uint32_t tile_mask = uint32_t{1} << codec_tile_idx;

  ByteV2PayloadTileDescriptor<Layout> desc;
  desc.page = page;
  desc.payload_offset =
      payload_base_offset + codec_tile_idx * kPayloadBytesPerTile;
  desc.outlier_payload_offset = 0;
  desc.outlier_count = 0;
  desc.base = IsValue ? page[Layout::v_base_offset(kv_head, dim_tile, 0)]
                      : page[Layout::k_base_offset(kv_head, dim_tile, 0)];
  desc.fallback_hit = false;
  desc.outlier_hit = false;

  desc.fallback_hit = (fallback_mask & tile_mask) != 0;
  desc.outlier_hit = (outlier_mask & tile_mask) != 0;
  if (desc.outlier_hit) {
    desc.outlier_count =
        IsValue ? Layout::v_outlier_count(page, kv_head, dim_tile, 0)
                : Layout::k_outlier_count(page, kv_head, dim_tile, 0);
    desc.outlier_payload_offset =
        IsValue ? Layout::v_outlier_payload_offset(page, kv_head, dim_tile, 0)
                : Layout::k_outlier_payload_offset(page, kv_head, dim_tile, 0);
  }

  return desc;
}

template <typename Layout, bool IsValue>
__device__ __forceinline__ ByteV2PayloadTileDescriptor<Layout>
byte_v2_make_payload_tile_descriptor(const uint8_t* __restrict__ page,
                                     uint32_t payload_base_offset, int kv_head,
                                     int dim_tile, bool use_masks) {
  uint32_t fallback_mask = 0;
  uint32_t outlier_mask = 0;
  if (use_masks) {
    fallback_mask = IsValue
                        ? byte_v2_load_u32_bytes(
                              page, Layout::v_fallback_mask_offset(kv_head))
                        : byte_v2_load_u32_bytes(
                              page, Layout::k_fallback_mask_offset(kv_head));
    outlier_mask = IsValue ? byte_v2_load_u32_bytes(
                                 page, Layout::v_outlier_mask_offset(kv_head))
                           : byte_v2_load_u32_bytes(
                                 page, Layout::k_outlier_mask_offset(kv_head));
  }

  return byte_v2_make_payload_tile_descriptor_from_masks<Layout, IsValue>(
      page, payload_base_offset, kv_head, dim_tile, fallback_mask,
      outlier_mask);
}

template <typename Layout, bool IsValue>
__device__ __forceinline__ float byte_v2_load_payload_elem_from_tile_descriptor(
    const ByteV2PayloadTileDescriptor<Layout>& desc, int kv_head, int row,
    int dim, int dim_in_tile) {
  using Policy = typename Layout::TilePolicy;
  static_assert(Policy::AllocBlockTokens == Policy::CodecTokenBlock);

  constexpr int kCodecDimBlock = Policy::CodecDimBlock;
  constexpr int kCodecTileElems = Policy::CodecTileElems;

  const int elem_idx = row * kCodecDimBlock + dim_in_tile;
  if (desc.fallback_hit) {
    if constexpr (Layout::IncludeRawPayloadValue) {
      return byte_v2_load_raw_elem<Layout, IsValue>(desc.page, kv_head, row,
                                                    dim);
    }
  }

  const uint8_t low = desc.page[desc.payload_offset + elem_idx];
  const uint8_t packed_code =
      desc.page[desc.payload_offset + kCodecTileElems + elem_idx / 2];
  const uint8_t code =
      (elem_idx & 1) ? (packed_code >> 4) : (packed_code & 0x0f);
  uint8_t high = byte_v2_high_byte_from_base_and_code(desc.base, code);

  if constexpr (Layout::CodecOutlierHighSideband) {
    if (desc.outlier_hit) {
      high = desc.page[desc.outlier_payload_offset + elem_idx];
      return byte_v2_bf16_bits_to_float((static_cast<uint16_t>(high) << 8) |
                                        low);
    }
  }

  if (desc.outlier_hit) {
    if (desc.outlier_count == Layout::OutlierEntriesPerTileValue - 1 ||
        desc.outlier_count > elem_idx) {
      const uint16_t direct_entry = byte_v2_load_u16_bytes(
          desc.page,
          desc.outlier_payload_offset + elem_idx * Layout::OutlierEntryBytes);
      if (Layout::OutlierEntryPolicy::decode_elem_index(direct_entry) ==
          static_cast<uint32_t>(elem_idx)) {
        high = static_cast<uint8_t>(
            Layout::OutlierEntryPolicy::decode_value_bits(direct_entry));
        return byte_v2_bf16_bits_to_float((static_cast<uint16_t>(high) << 8) |
                                          low);
      }
    }
#pragma unroll 1
    for (int entry_idx = 0; entry_idx < desc.outlier_count; ++entry_idx) {
      const uint16_t entry = byte_v2_load_u16_bytes(
          desc.page,
          desc.outlier_payload_offset + entry_idx * Layout::OutlierEntryBytes);
      if (Layout::OutlierEntryPolicy::decode_elem_index(entry) ==
          static_cast<uint32_t>(elem_idx)) {
        high = static_cast<uint8_t>(
            Layout::OutlierEntryPolicy::decode_value_bits(entry));
        break;
      }
    }
  }

  return byte_v2_bf16_bits_to_float((static_cast<uint16_t>(high) << 8) | low);
}

template <typename Layout>
__device__ __forceinline__ uint16_t
byte_v2_load_payload_elem_from_safe_tile_descriptor_fixed_dim_bits(
    const ByteV2PayloadTileDescriptor<Layout>& desc, int row, int dim_in_tile) {
  using Policy = typename Layout::TilePolicy;
  static_assert(Policy::AllocBlockTokens == Policy::CodecTokenBlock);

  constexpr int kCodecDimBlock = Policy::CodecDimBlock;
  constexpr int kCodecTileElems = Policy::CodecTileElems;
  constexpr int kCodeBytesPerRow = kCodecDimBlock / 2;

  const int low_offset = row * kCodecDimBlock + dim_in_tile;
  const int code_offset = row * kCodeBytesPerRow + (dim_in_tile >> 1);
  const uint8_t low = desc.page[desc.payload_offset + low_offset];
  const uint8_t packed_code =
      desc.page[desc.payload_offset + kCodecTileElems + code_offset];
  const uint8_t code =
      (dim_in_tile & 1) ? (packed_code >> 4) : (packed_code & 0x0f);
  const uint8_t high = byte_v2_high_byte_from_base_and_code(desc.base, code);
  return (static_cast<uint16_t>(high) << 8) | low;
}

template <typename Layout>
__device__ __forceinline__ void
byte_v2_load_payload_elem_pair_from_safe_tile_descriptor_fixed_dim_bits(
    const ByteV2PayloadTileDescriptor<Layout>& desc, int row,
    int dim_in_tile_even, uint16_t& bits0, uint16_t& bits1) {
  using Policy = typename Layout::TilePolicy;
  static_assert(Policy::AllocBlockTokens == Policy::CodecTokenBlock);

  constexpr int kCodecDimBlock = Policy::CodecDimBlock;
  constexpr int kCodecTileElems = Policy::CodecTileElems;
  constexpr int kCodeBytesPerRow = kCodecDimBlock / 2;

  const int low_offset = row * kCodecDimBlock + dim_in_tile_even;
  const int code_offset = row * kCodeBytesPerRow + (dim_in_tile_even >> 1);
  const uint16_t low_pair =
      byte_v2_load_aligned_u16(desc.page, desc.payload_offset + low_offset);
  const uint8_t packed_code =
      desc.page[desc.payload_offset + kCodecTileElems + code_offset];
  const uint8_t high0 =
      byte_v2_high_byte_from_base_and_code(desc.base, packed_code & 0x0f);
  const uint8_t high1 =
      byte_v2_high_byte_from_base_and_code(desc.base, packed_code >> 4);

  bits0 = (static_cast<uint16_t>(high0) << 8) | (low_pair & 0x00ff);
  bits1 = (static_cast<uint16_t>(high1) << 8) | (low_pair >> 8);
}

template <int LowElem, int CodeElem>
__device__ __forceinline__ uint16_t
byte_v2_make_payload_bits_from_low_and_codes(uint64_t low_chunk,
                                             uint64_t packed_codes,
                                             uint8_t base) {
  static_assert(LowElem >= 0 && LowElem < 8);
  static_assert(CodeElem >= 0 && CodeElem < 16);
  const uint16_t low =
      static_cast<uint16_t>((low_chunk >> (LowElem * 8)) & 0xffULL);
  const uint8_t code =
      static_cast<uint8_t>((packed_codes >> (CodeElem * 4)) & 0x0fULL);
  const uint16_t high =
      static_cast<uint16_t>(byte_v2_high_byte_from_base_and_code(base, code));
  return static_cast<uint16_t>((high << 8) | low);
}

template <int Elem>
__device__ __forceinline__ uint16_t byte_v2_make_payload_bits_from_low_and_high(
    uint64_t low_chunk, uint64_t high_chunk) {
  static_assert(Elem >= 0 && Elem < 8);
  const uint16_t low =
      static_cast<uint16_t>((low_chunk >> (Elem * 8)) & 0xffULL);
  const uint16_t high =
      static_cast<uint16_t>((high_chunk >> (Elem * 8)) & 0xffULL);
  return static_cast<uint16_t>((high << 8) | low);
}

template <typename Layout, bool LoadPackedCodes = true>
__device__ __forceinline__ void
byte_v2_load_payload_elem_hex_from_safe_tile_descriptor_fixed_dim_bits(
    const ByteV2PayloadTileDescriptor<Layout>& desc, int row,
    int dim_in_tile_hex, uint16_t& bits0, uint16_t& bits1, uint16_t& bits2,
    uint16_t& bits3, uint16_t& bits4, uint16_t& bits5, uint16_t& bits6,
    uint16_t& bits7, uint16_t& bits8, uint16_t& bits9, uint16_t& bits10,
    uint16_t& bits11, uint16_t& bits12, uint16_t& bits13, uint16_t& bits14,
    uint16_t& bits15) {
  using Policy = typename Layout::TilePolicy;
  static_assert(Policy::AllocBlockTokens == Policy::CodecTokenBlock);
  static_assert(Policy::CodecDimBlock % 16 == 0);

  constexpr int kCodecDimBlock = Policy::CodecDimBlock;
  constexpr int kCodecTileElems = Policy::CodecTileElems;
  constexpr int kCodeBytesPerRow = kCodecDimBlock / 2;

  const int low_offset = row * kCodecDimBlock + dim_in_tile_hex;
  const int code_offset = row * kCodeBytesPerRow + (dim_in_tile_hex >> 1);
  const uint64_t low0 =
      byte_v2_load_aligned_u64(desc.page, desc.payload_offset + low_offset);
  const uint64_t low1 =
      byte_v2_load_aligned_u64(desc.page, desc.payload_offset + low_offset + 8);
  if constexpr (Layout::CodecExponentCodeBits == 8) {
    const uint64_t high0 = byte_v2_load_aligned_u64(
        desc.page, desc.payload_offset + kCodecTileElems + low_offset);
    const uint64_t high1 = byte_v2_load_aligned_u64(
        desc.page, desc.payload_offset + kCodecTileElems + low_offset + 8);
    bits0 = byte_v2_make_payload_bits_from_low_and_high<0>(low0, high0);
    bits1 = byte_v2_make_payload_bits_from_low_and_high<1>(low0, high0);
    bits2 = byte_v2_make_payload_bits_from_low_and_high<2>(low0, high0);
    bits3 = byte_v2_make_payload_bits_from_low_and_high<3>(low0, high0);
    bits4 = byte_v2_make_payload_bits_from_low_and_high<4>(low0, high0);
    bits5 = byte_v2_make_payload_bits_from_low_and_high<5>(low0, high0);
    bits6 = byte_v2_make_payload_bits_from_low_and_high<6>(low0, high0);
    bits7 = byte_v2_make_payload_bits_from_low_and_high<7>(low0, high0);
    bits8 = byte_v2_make_payload_bits_from_low_and_high<0>(low1, high1);
    bits9 = byte_v2_make_payload_bits_from_low_and_high<1>(low1, high1);
    bits10 = byte_v2_make_payload_bits_from_low_and_high<2>(low1, high1);
    bits11 = byte_v2_make_payload_bits_from_low_and_high<3>(low1, high1);
    bits12 = byte_v2_make_payload_bits_from_low_and_high<4>(low1, high1);
    bits13 = byte_v2_make_payload_bits_from_low_and_high<5>(low1, high1);
    bits14 = byte_v2_make_payload_bits_from_low_and_high<6>(low1, high1);
    bits15 = byte_v2_make_payload_bits_from_low_and_high<7>(low1, high1);
  } else {
    static_assert(Layout::CodecExponentCodeBits == 4);
    uint64_t packed_codes;
    if constexpr (LoadPackedCodes) {
      packed_codes = byte_v2_load_aligned_u64(
          desc.page, desc.payload_offset + kCodecTileElems + code_offset);
    } else {
      // Diagnostic upper bound: keep the downstream nibble extraction and
      // high-byte reconstruction, but make the code word register-ready.
      packed_codes = static_cast<uint64_t>(desc.base) * 0x0101010101010101ULL;
    }

    bits0 = byte_v2_make_payload_bits_from_low_and_codes<0, 0>(
        low0, packed_codes, desc.base);
    bits1 = byte_v2_make_payload_bits_from_low_and_codes<1, 1>(
        low0, packed_codes, desc.base);
    bits2 = byte_v2_make_payload_bits_from_low_and_codes<2, 2>(
        low0, packed_codes, desc.base);
    bits3 = byte_v2_make_payload_bits_from_low_and_codes<3, 3>(
        low0, packed_codes, desc.base);
    bits4 = byte_v2_make_payload_bits_from_low_and_codes<4, 4>(
        low0, packed_codes, desc.base);
    bits5 = byte_v2_make_payload_bits_from_low_and_codes<5, 5>(
        low0, packed_codes, desc.base);
    bits6 = byte_v2_make_payload_bits_from_low_and_codes<6, 6>(
        low0, packed_codes, desc.base);
    bits7 = byte_v2_make_payload_bits_from_low_and_codes<7, 7>(
        low0, packed_codes, desc.base);
    bits8 = byte_v2_make_payload_bits_from_low_and_codes<0, 8>(
        low1, packed_codes, desc.base);
    bits9 = byte_v2_make_payload_bits_from_low_and_codes<1, 9>(
        low1, packed_codes, desc.base);
    bits10 = byte_v2_make_payload_bits_from_low_and_codes<2, 10>(
        low1, packed_codes, desc.base);
    bits11 = byte_v2_make_payload_bits_from_low_and_codes<3, 11>(
        low1, packed_codes, desc.base);
    bits12 = byte_v2_make_payload_bits_from_low_and_codes<4, 12>(
        low1, packed_codes, desc.base);
    bits13 = byte_v2_make_payload_bits_from_low_and_codes<5, 13>(
        low1, packed_codes, desc.base);
    bits14 = byte_v2_make_payload_bits_from_low_and_codes<6, 14>(
        low1, packed_codes, desc.base);
    bits15 = byte_v2_make_payload_bits_from_low_and_codes<7, 15>(
        low1, packed_codes, desc.base);
  }
}

template <typename Layout>
__device__ __forceinline__ void
byte_v2_load_payload_elem_hex_from_outlier_high_sideband_fixed_dim_bits(
    const ByteV2PayloadTileDescriptor<Layout>& desc, int row,
    int dim_in_tile_hex, uint16_t& bits0, uint16_t& bits1, uint16_t& bits2,
    uint16_t& bits3, uint16_t& bits4, uint16_t& bits5, uint16_t& bits6,
    uint16_t& bits7, uint16_t& bits8, uint16_t& bits9, uint16_t& bits10,
    uint16_t& bits11, uint16_t& bits12, uint16_t& bits13, uint16_t& bits14,
    uint16_t& bits15) {
  using Policy = typename Layout::TilePolicy;
  static_assert(Layout::CodecOutlierHighSideband);
  static_assert(Policy::AllocBlockTokens == Policy::CodecTokenBlock);
  static_assert(Policy::CodecDimBlock % 16 == 0);

  constexpr int kCodecDimBlock = Policy::CodecDimBlock;

  const int low_offset = row * kCodecDimBlock + dim_in_tile_hex;
  const uint64_t low0 =
      byte_v2_load_aligned_u64(desc.page, desc.payload_offset + low_offset);
  const uint64_t low1 =
      byte_v2_load_aligned_u64(desc.page, desc.payload_offset + low_offset + 8);
  const uint64_t high0 = byte_v2_load_aligned_u64(
      desc.page, desc.outlier_payload_offset + low_offset);
  const uint64_t high1 = byte_v2_load_aligned_u64(
      desc.page, desc.outlier_payload_offset + low_offset + 8);
  bits0 = byte_v2_make_payload_bits_from_low_and_high<0>(low0, high0);
  bits1 = byte_v2_make_payload_bits_from_low_and_high<1>(low0, high0);
  bits2 = byte_v2_make_payload_bits_from_low_and_high<2>(low0, high0);
  bits3 = byte_v2_make_payload_bits_from_low_and_high<3>(low0, high0);
  bits4 = byte_v2_make_payload_bits_from_low_and_high<4>(low0, high0);
  bits5 = byte_v2_make_payload_bits_from_low_and_high<5>(low0, high0);
  bits6 = byte_v2_make_payload_bits_from_low_and_high<6>(low0, high0);
  bits7 = byte_v2_make_payload_bits_from_low_and_high<7>(low0, high0);
  bits8 = byte_v2_make_payload_bits_from_low_and_high<0>(low1, high1);
  bits9 = byte_v2_make_payload_bits_from_low_and_high<1>(low1, high1);
  bits10 = byte_v2_make_payload_bits_from_low_and_high<2>(low1, high1);
  bits11 = byte_v2_make_payload_bits_from_low_and_high<3>(low1, high1);
  bits12 = byte_v2_make_payload_bits_from_low_and_high<4>(low1, high1);
  bits13 = byte_v2_make_payload_bits_from_low_and_high<5>(low1, high1);
  bits14 = byte_v2_make_payload_bits_from_low_and_high<6>(low1, high1);
  bits15 = byte_v2_make_payload_bits_from_low_and_high<7>(low1, high1);
}

template <int LowElem>
__device__ __forceinline__ uint16_t
byte_v2_make_payload_low_only_bits(uint64_t low_chunk) {
  static_assert(LowElem >= 0 && LowElem < 8);
  return static_cast<uint16_t>((low_chunk >> (LowElem * 8)) & 0xffULL);
}

template <int CodeElem>
__device__ __forceinline__ uint16_t
byte_v2_make_payload_high_only_bits(uint64_t packed_codes, uint8_t base) {
  static_assert(CodeElem >= 0 && CodeElem < 16);
  const uint8_t code =
      static_cast<uint8_t>((packed_codes >> (CodeElem * 4)) & 0x0fULL);
  const uint16_t high =
      static_cast<uint16_t>(byte_v2_high_byte_from_base_and_code(base, code));
  return static_cast<uint16_t>(high << 8);
}

template <typename Layout>
__device__ __forceinline__ void
byte_v2_load_payload_elem_hex_low_only_from_safe_tile_descriptor_fixed_dim_bits(
    const ByteV2PayloadTileDescriptor<Layout>& desc, int row,
    int dim_in_tile_hex, uint16_t& bits0, uint16_t& bits1, uint16_t& bits2,
    uint16_t& bits3, uint16_t& bits4, uint16_t& bits5, uint16_t& bits6,
    uint16_t& bits7, uint16_t& bits8, uint16_t& bits9, uint16_t& bits10,
    uint16_t& bits11, uint16_t& bits12, uint16_t& bits13, uint16_t& bits14,
    uint16_t& bits15) {
  using Policy = typename Layout::TilePolicy;
  static_assert(Policy::AllocBlockTokens == Policy::CodecTokenBlock);
  static_assert(Policy::CodecDimBlock % 16 == 0);

  constexpr int kCodecDimBlock = Policy::CodecDimBlock;

  const int low_offset = row * kCodecDimBlock + dim_in_tile_hex;
  const uint64_t low0 =
      byte_v2_load_aligned_u64(desc.page, desc.payload_offset + low_offset);
  const uint64_t low1 =
      byte_v2_load_aligned_u64(desc.page, desc.payload_offset + low_offset + 8);

  bits0 = byte_v2_make_payload_low_only_bits<0>(low0);
  bits1 = byte_v2_make_payload_low_only_bits<1>(low0);
  bits2 = byte_v2_make_payload_low_only_bits<2>(low0);
  bits3 = byte_v2_make_payload_low_only_bits<3>(low0);
  bits4 = byte_v2_make_payload_low_only_bits<4>(low0);
  bits5 = byte_v2_make_payload_low_only_bits<5>(low0);
  bits6 = byte_v2_make_payload_low_only_bits<6>(low0);
  bits7 = byte_v2_make_payload_low_only_bits<7>(low0);
  bits8 = byte_v2_make_payload_low_only_bits<0>(low1);
  bits9 = byte_v2_make_payload_low_only_bits<1>(low1);
  bits10 = byte_v2_make_payload_low_only_bits<2>(low1);
  bits11 = byte_v2_make_payload_low_only_bits<3>(low1);
  bits12 = byte_v2_make_payload_low_only_bits<4>(low1);
  bits13 = byte_v2_make_payload_low_only_bits<5>(low1);
  bits14 = byte_v2_make_payload_low_only_bits<6>(low1);
  bits15 = byte_v2_make_payload_low_only_bits<7>(low1);
}

template <typename Layout>
__device__ __forceinline__ void
byte_v2_load_payload_elem_hex_high_only_from_safe_tile_descriptor_fixed_dim_bits(
    const ByteV2PayloadTileDescriptor<Layout>& desc, int row,
    int dim_in_tile_hex, uint16_t& bits0, uint16_t& bits1, uint16_t& bits2,
    uint16_t& bits3, uint16_t& bits4, uint16_t& bits5, uint16_t& bits6,
    uint16_t& bits7, uint16_t& bits8, uint16_t& bits9, uint16_t& bits10,
    uint16_t& bits11, uint16_t& bits12, uint16_t& bits13, uint16_t& bits14,
    uint16_t& bits15) {
  using Policy = typename Layout::TilePolicy;
  static_assert(Policy::AllocBlockTokens == Policy::CodecTokenBlock);
  static_assert(Policy::CodecDimBlock % 16 == 0);

  constexpr int kCodecDimBlock = Policy::CodecDimBlock;
  constexpr int kCodecTileElems = Policy::CodecTileElems;
  constexpr int kCodeBytesPerRow = kCodecDimBlock / 2;

  const int code_offset = row * kCodeBytesPerRow + (dim_in_tile_hex >> 1);
  const uint64_t packed_codes = byte_v2_load_aligned_u64(
      desc.page, desc.payload_offset + kCodecTileElems + code_offset);

  bits0 = byte_v2_make_payload_high_only_bits<0>(packed_codes, desc.base);
  bits1 = byte_v2_make_payload_high_only_bits<1>(packed_codes, desc.base);
  bits2 = byte_v2_make_payload_high_only_bits<2>(packed_codes, desc.base);
  bits3 = byte_v2_make_payload_high_only_bits<3>(packed_codes, desc.base);
  bits4 = byte_v2_make_payload_high_only_bits<4>(packed_codes, desc.base);
  bits5 = byte_v2_make_payload_high_only_bits<5>(packed_codes, desc.base);
  bits6 = byte_v2_make_payload_high_only_bits<6>(packed_codes, desc.base);
  bits7 = byte_v2_make_payload_high_only_bits<7>(packed_codes, desc.base);
  bits8 = byte_v2_make_payload_high_only_bits<8>(packed_codes, desc.base);
  bits9 = byte_v2_make_payload_high_only_bits<9>(packed_codes, desc.base);
  bits10 = byte_v2_make_payload_high_only_bits<10>(packed_codes, desc.base);
  bits11 = byte_v2_make_payload_high_only_bits<11>(packed_codes, desc.base);
  bits12 = byte_v2_make_payload_high_only_bits<12>(packed_codes, desc.base);
  bits13 = byte_v2_make_payload_high_only_bits<13>(packed_codes, desc.base);
  bits14 = byte_v2_make_payload_high_only_bits<14>(packed_codes, desc.base);
  bits15 = byte_v2_make_payload_high_only_bits<15>(packed_codes, desc.base);
}

__device__ __forceinline__ void byte_v2_accumulate_16_bf16_bits_for_diagnostic(
    uint32_t& sink, uint16_t bits0, uint16_t bits1, uint16_t bits2,
    uint16_t bits3, uint16_t bits4, uint16_t bits5, uint16_t bits6,
    uint16_t bits7, uint16_t bits8, uint16_t bits9, uint16_t bits10,
    uint16_t bits11, uint16_t bits12, uint16_t bits13, uint16_t bits14,
    uint16_t bits15) {
  sink ^= static_cast<uint32_t>(bits0) | (static_cast<uint32_t>(bits1) << 16);
  sink ^= static_cast<uint32_t>(bits2) | (static_cast<uint32_t>(bits3) << 16);
  sink ^= static_cast<uint32_t>(bits4) | (static_cast<uint32_t>(bits5) << 16);
  sink ^= static_cast<uint32_t>(bits6) | (static_cast<uint32_t>(bits7) << 16);
  sink ^= static_cast<uint32_t>(bits8) | (static_cast<uint32_t>(bits9) << 16);
  sink ^= static_cast<uint32_t>(bits10) | (static_cast<uint32_t>(bits11) << 16);
  sink ^= static_cast<uint32_t>(bits12) | (static_cast<uint32_t>(bits13) << 16);
  sink ^= static_cast<uint32_t>(bits14) | (static_cast<uint32_t>(bits15) << 16);
}

template <typename Layout>
__device__ __forceinline__ uint16_t
byte_v2_outlier_high_bits_from_entry(uint16_t entry) {
  const uint16_t high = static_cast<uint16_t>(
      Layout::OutlierEntryPolicy::decode_value_bits(entry));
  return static_cast<uint16_t>(high << 8);
}

template <typename Layout>
__device__ __forceinline__ void
byte_v2_patch_payload_elem_hex_outliers_from_tile_descriptor_bits(
    const ByteV2PayloadTileDescriptor<Layout>& desc, int row,
    int dim_in_tile_hex, uint16_t& bits0, uint16_t& bits1, uint16_t& bits2,
    uint16_t& bits3, uint16_t& bits4, uint16_t& bits5, uint16_t& bits6,
    uint16_t& bits7, uint16_t& bits8, uint16_t& bits9, uint16_t& bits10,
    uint16_t& bits11, uint16_t& bits12, uint16_t& bits13, uint16_t& bits14,
    uint16_t& bits15) {
  using Policy = typename Layout::TilePolicy;
  static_assert(Policy::AllocBlockTokens == Policy::CodecTokenBlock);
  static_assert(Policy::CodecDimBlock % 16 == 0);

  constexpr int kCodecDimBlock = Policy::CodecDimBlock;
  const int elem_start = row * kCodecDimBlock + dim_in_tile_hex;
  const int elem_end = elem_start + 16;

  // Cache writers append outlier entries in elem_idx order.
#pragma unroll 1
  for (int entry_idx = 0; entry_idx < desc.outlier_count; ++entry_idx) {
    const uint16_t entry = byte_v2_load_u16_bytes(
        desc.page,
        desc.outlier_payload_offset + entry_idx * Layout::OutlierEntryBytes);
    const int elem_idx =
        static_cast<int>(Layout::OutlierEntryPolicy::decode_elem_index(entry));
    if (elem_idx < elem_start) {
      continue;
    }
    if (elem_idx >= elem_end) {
      break;
    }
    const uint16_t patched =
        byte_v2_outlier_high_bits_from_entry<Layout>(entry);
    switch (elem_idx - elem_start) {
      case 0:
        bits0 = static_cast<uint16_t>((patched & 0xff00) | (bits0 & 0x00ff));
        break;
      case 1:
        bits1 = static_cast<uint16_t>((patched & 0xff00) | (bits1 & 0x00ff));
        break;
      case 2:
        bits2 = static_cast<uint16_t>((patched & 0xff00) | (bits2 & 0x00ff));
        break;
      case 3:
        bits3 = static_cast<uint16_t>((patched & 0xff00) | (bits3 & 0x00ff));
        break;
      case 4:
        bits4 = static_cast<uint16_t>((patched & 0xff00) | (bits4 & 0x00ff));
        break;
      case 5:
        bits5 = static_cast<uint16_t>((patched & 0xff00) | (bits5 & 0x00ff));
        break;
      case 6:
        bits6 = static_cast<uint16_t>((patched & 0xff00) | (bits6 & 0x00ff));
        break;
      case 7:
        bits7 = static_cast<uint16_t>((patched & 0xff00) | (bits7 & 0x00ff));
        break;
      case 8:
        bits8 = static_cast<uint16_t>((patched & 0xff00) | (bits8 & 0x00ff));
        break;
      case 9:
        bits9 = static_cast<uint16_t>((patched & 0xff00) | (bits9 & 0x00ff));
        break;
      case 10:
        bits10 = static_cast<uint16_t>((patched & 0xff00) | (bits10 & 0x00ff));
        break;
      case 11:
        bits11 = static_cast<uint16_t>((patched & 0xff00) | (bits11 & 0x00ff));
        break;
      case 12:
        bits12 = static_cast<uint16_t>((patched & 0xff00) | (bits12 & 0x00ff));
        break;
      case 13:
        bits13 = static_cast<uint16_t>((patched & 0xff00) | (bits13 & 0x00ff));
        break;
      case 14:
        bits14 = static_cast<uint16_t>((patched & 0xff00) | (bits14 & 0x00ff));
        break;
      default:
        bits15 = static_cast<uint16_t>((patched & 0xff00) | (bits15 & 0x00ff));
        break;
    }
  }
}

__device__ __forceinline__ uint4 byte_v2_pack_8_bf16_bits(
    uint16_t bits0, uint16_t bits1, uint16_t bits2, uint16_t bits3,
    uint16_t bits4, uint16_t bits5, uint16_t bits6, uint16_t bits7) {
  return make_uint4(
      static_cast<uint32_t>(bits0) | (static_cast<uint32_t>(bits1) << 16),
      static_cast<uint32_t>(bits2) | (static_cast<uint32_t>(bits3) << 16),
      static_cast<uint32_t>(bits4) | (static_cast<uint32_t>(bits5) << 16),
      static_cast<uint32_t>(bits6) | (static_cast<uint32_t>(bits7) << 16));
}

template <typename Element, typename SmemLayout>
__device__ __forceinline__ void byte_v2_store_16_bf16_bits_to_smem(
    Element* smem, SmemLayout layout, int row, int col, uint16_t bits0,
    uint16_t bits1, uint16_t bits2, uint16_t bits3, uint16_t bits4,
    uint16_t bits5, uint16_t bits6, uint16_t bits7, uint16_t bits8,
    uint16_t bits9, uint16_t bits10, uint16_t bits11, uint16_t bits12,
    uint16_t bits13, uint16_t bits14, uint16_t bits15) {
  const int offset0 = layout(row, col);
  const int offset1 = layout(row, col + 8);
  *reinterpret_cast<uint4*>(smem + offset0) = byte_v2_pack_8_bf16_bits(
      bits0, bits1, bits2, bits3, bits4, bits5, bits6, bits7);
  *reinterpret_cast<uint4*>(smem + offset1) = byte_v2_pack_8_bf16_bits(
      bits8, bits9, bits10, bits11, bits12, bits13, bits14, bits15);
}

template <typename Layout>
__device__ __forceinline__ float
byte_v2_load_payload_elem_from_safe_tile_descriptor_fixed_dim(
    const ByteV2PayloadTileDescriptor<Layout>& desc, int row, int dim_in_tile) {
  return byte_v2_bf16_bits_to_float(
      byte_v2_load_payload_elem_from_safe_tile_descriptor_fixed_dim_bits<
          Layout>(desc, row, dim_in_tile));
}

template <int NumThreads>
__device__ __forceinline__ float byte_v2_block_sum(float value,
                                                   float* reduce_smem) {
  static_assert(NumThreads % 32 == 0);
  constexpr int kNumWarps = NumThreads / 32;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  if (lane == 0) {
    reduce_smem[warp] = value;
  }
  __syncthreads();

  value = threadIdx.x < kNumWarps ? reduce_smem[lane] : 0.0f;
  if (warp == 0) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      value += __shfl_down_sync(0xffffffff, value, offset);
    }
    if (lane == 0) {
      reduce_smem[0] = value;
    }
  }
  __syncthreads();
  const float block_value = reduce_smem[0];
  __syncthreads();
  return block_value;
}

template <int NumThreads>
__device__ __forceinline__ float byte_v2_block_sum_thread0(float value,
                                                           float* reduce_smem) {
  static_assert(NumThreads % 32 == 0);
  constexpr int kNumWarps = NumThreads / 32;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  if (lane == 0) {
    reduce_smem[warp] = value;
  }
  __syncthreads();

  float block_value = 0.0f;
  value = threadIdx.x < kNumWarps ? reduce_smem[lane] : 0.0f;
  if (warp == 0) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      value += __shfl_down_sync(0xffffffff, value, offset);
    }
    if (lane == 0) {
      block_value = value;
    }
  }
  __syncthreads();
  return block_value;
}

template <int NumThreads>
__device__ __forceinline__ float byte_v2_block_max(float value,
                                                   float* reduce_smem) {
  static_assert(NumThreads % 32 == 0);
  constexpr int kNumWarps = NumThreads / 32;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;

#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value = fmaxf(value, __shfl_down_sync(0xffffffff, value, offset));
  }
  if (lane == 0) {
    reduce_smem[warp] = value;
  }
  __syncthreads();

  value = threadIdx.x < kNumWarps ? reduce_smem[lane] : -FLT_MAX;
  if (warp == 0) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      value = fmaxf(value, __shfl_down_sync(0xffffffff, value, offset));
    }
    if (lane == 0) {
      reduce_smem[0] = value;
    }
  }
  __syncthreads();
  const float block_value = reduce_smem[0];
  __syncthreads();
  return block_value;
}

template <typename Layout, bool UseRawFallback, bool AssumeNoFallbackNoOutlier,
          int NumThreads>
__global__ void byte_v2_paged_decode_attention_kernel(
    uint16_t* __restrict__ output, const uint16_t* __restrict__ query,
    const uint8_t* __restrict__ kv_cache,
    const int32_t* __restrict__ block_tables,
    const int32_t* __restrict__ seq_lens, float scale, int64_t q_stride_token,
    int64_t q_stride_head, int64_t q_stride_dim, int64_t out_stride_token,
    int64_t out_stride_head, int64_t out_stride_dim,
    int64_t kv_cache_stride_block, int64_t block_table_stride_seq,
    int max_num_blocks_per_seq) {
  using Policy = typename Layout::TilePolicy;

  constexpr int kBlockSize = Policy::AllocBlockTokens;

  const int seq_idx = blockIdx.y;
  const int head_idx = blockIdx.x;
  const int dim = threadIdx.x;
  const int num_heads = gridDim.x;
  const int q_per_kv = num_heads / Layout::NumKvHeadsValue;
  const int kv_head_idx = head_idx / q_per_kv;
  const int32_t seq_len = seq_lens[seq_idx];

  const int64_t q_offset = static_cast<int64_t>(seq_idx) * q_stride_token +
                           static_cast<int64_t>(head_idx) * q_stride_head +
                           dim * q_stride_dim;
  const float q_value = byte_v2_bf16_bits_to_float(query[q_offset]);

  float acc = 0.0f;
  float softmax_m = -FLT_MAX;
  float softmax_l = 0.0f;

  __shared__ float reduce_smem[NumThreads / 32];

  const int32_t* block_table =
      block_tables + static_cast<int64_t>(seq_idx) * block_table_stride_seq;

  __shared__ float shared_scores[Policy::ComputeBlockN];
  __shared__ float shared_probs[Policy::ComputeBlockN];
  __shared__ float shared_new_m;
  __shared__ float shared_new_l;
  __shared__ float shared_alpha;

  constexpr uint32_t kPayloadBaseOffset = Layout::k_payload_offset(0, 0);
  constexpr uint32_t vPayloadBaseOffset = Layout::v_payload_offset(0, 0);
  const uint32_t k_payload_base_offset =
      kPayloadBaseOffset + kv_head_idx * Layout::AlignedKPayloadBytesPerKvHead;
  const uint32_t v_payload_base_offset =
      vPayloadBaseOffset + kv_head_idx * Layout::AlignedVPayloadBytesPerKvHead;
  const int v_dim_tile = dim / Policy::CodecDimBlock;
  const int v_dim_in_tile = dim % Policy::CodecDimBlock;

  for (int32_t tile_start = 0; tile_start < seq_len;
       tile_start += Policy::ComputeBlockN) {
    const int32_t tile_end =
        min(tile_start + Policy::ComputeBlockN, static_cast<int32_t>(seq_len));
    const int32_t tile_len = tile_end - tile_start;

#pragma unroll 1
    for (int32_t tile_offset = 0; tile_offset < tile_len; ++tile_offset) {
      const int32_t kv_token_idx = tile_start + tile_offset;
      const int32_t logical_block = kv_token_idx / kBlockSize;
      if (logical_block < 0 || logical_block >= max_num_blocks_per_seq) {
        if (threadIdx.x == 0) {
          shared_scores[tile_offset] = -FLT_MAX;
        }
        continue;
      }

      const int32_t physical_block = block_table[logical_block];
      if (physical_block < 0) {
        if (threadIdx.x == 0) {
          shared_scores[tile_offset] = -FLT_MAX;
        }
        continue;
      }

      const int row = kv_token_idx - logical_block * kBlockSize;
      const uint8_t* page = kv_cache + static_cast<int64_t>(physical_block) *
                                           kv_cache_stride_block;
      const float k_value =
          byte_v2_load_decode_elem<Layout, false, UseRawFallback,
                                   AssumeNoFallbackNoOutlier>(
              page, k_payload_base_offset, kv_head_idx, row, dim);
      const float qk_partial = q_value * k_value;
      const float qk_sum =
          byte_v2_block_sum_thread0<NumThreads>(qk_partial, reduce_smem);

      if (threadIdx.x == 0) {
        shared_scores[tile_offset] = qk_sum * scale;
      }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
      float tile_m = -FLT_MAX;
#pragma unroll 1
      for (int32_t tile_offset = 0; tile_offset < tile_len; ++tile_offset) {
        tile_m = fmaxf(tile_m, shared_scores[tile_offset]);
      }
      const float new_m = fmaxf(softmax_m, tile_m);
      const float alpha = __expf(softmax_m - new_m);
      float tile_l = 0.0f;
#pragma unroll 1
      for (int32_t tile_offset = 0; tile_offset < tile_len; ++tile_offset) {
        const float prob = __expf(shared_scores[tile_offset] - new_m);
        shared_probs[tile_offset] = prob;
        tile_l += prob;
      }
      shared_alpha = alpha;
      shared_new_m = new_m;
      shared_new_l = softmax_l * alpha + tile_l;
    }
    __syncthreads();

    float pv = 0.0f;
#pragma unroll 1
    for (int32_t tile_offset = 0; tile_offset < tile_len; ++tile_offset) {
      const int32_t kv_token_idx = tile_start + tile_offset;
      const int32_t logical_block = kv_token_idx / kBlockSize;
      if (logical_block < 0 || logical_block >= max_num_blocks_per_seq) {
        continue;
      }
      const int32_t physical_block = block_table[logical_block];
      if (physical_block < 0) {
        continue;
      }

      const int row = kv_token_idx - logical_block * kBlockSize;
      const uint8_t* page = kv_cache + static_cast<int64_t>(physical_block) *
                                           kv_cache_stride_block;
      const float v_value =
          byte_v2_load_decode_elem<Layout, true, UseRawFallback,
                                   AssumeNoFallbackNoOutlier>(
              page, v_payload_base_offset, kv_head_idx, row, dim);
      pv += shared_probs[tile_offset] * v_value;
    }
    acc = acc * shared_alpha + pv;
    softmax_l = shared_new_l;
    softmax_m = shared_new_m;
    __syncthreads();
  }

  const float out_value = softmax_l > 0.0f ? acc / softmax_l : 0.0f;
  const int64_t out_offset = static_cast<int64_t>(seq_idx) * out_stride_token +
                             static_cast<int64_t>(head_idx) * out_stride_head +
                             dim * out_stride_dim;
  output[out_offset] = byte_v2_float_to_bf16_bits(out_value);
}

template <typename Layout, bool UseRawFallback, bool AssumeNoFallbackNoOutlier,
          bool UsePageUnsafeFlags, int NumThreads>
__global__ void byte_v2_paged_decode_attention_split_k_kernel(
    float* __restrict__ tmp_out, float* __restrict__ exp_sums,
    float* __restrict__ max_logits, const uint16_t* __restrict__ query,
    const uint8_t* __restrict__ kv_cache,
    const int32_t* __restrict__ page_unsafe_flags,
    const int32_t* __restrict__ block_tables,
    const int32_t* __restrict__ seq_lens, float scale, int64_t q_stride_token,
    int64_t q_stride_head, int64_t q_stride_dim, int64_t kv_cache_stride_block,
    int64_t block_table_stride_seq, int max_num_blocks_per_seq,
    int max_num_partitions, int partition_size) {
  using Policy = typename Layout::TilePolicy;
  (void)max_logits;

  constexpr int kBlockSize = Policy::AllocBlockTokens;

  const int seq_idx = blockIdx.y;
  const int head_idx = blockIdx.x;
  const int partition_idx = blockIdx.z;
  const int dim = threadIdx.x;
  const int num_heads = gridDim.x;
  const int q_per_kv = num_heads / Layout::NumKvHeadsValue;
  const int kv_head_idx = head_idx / q_per_kv;
  const int32_t seq_len = seq_lens[seq_idx];
  const int32_t partition_start = partition_idx * partition_size;
  const int32_t partition_end =
      min(partition_start + partition_size, static_cast<int32_t>(seq_len));

  const int64_t stats_offset =
      (static_cast<int64_t>(seq_idx) * num_heads + head_idx) *
          max_num_partitions +
      partition_idx;
  const int64_t tmp_base =
      stats_offset * static_cast<int64_t>(Policy::HeadDimV);

  if (partition_start >= seq_len || partition_start >= partition_end) {
    if (threadIdx.x == 0) {
      exp_sums[stats_offset] = -FLT_MAX;
    }
    tmp_out[tmp_base + dim] = 0.0f;
    return;
  }

  const int64_t q_offset = static_cast<int64_t>(seq_idx) * q_stride_token +
                           static_cast<int64_t>(head_idx) * q_stride_head +
                           dim * q_stride_dim;
  const float q_value = byte_v2_bf16_bits_to_float(query[q_offset]);

  float acc = 0.0f;
  float softmax_m = -FLT_MAX;
  float softmax_l = 0.0f;

  __shared__ float reduce_smem[NumThreads / 32];
  __shared__ float shared_scores[Policy::ComputeBlockN];
  __shared__ float shared_probs[Policy::ComputeBlockN];
  __shared__ float shared_new_m;
  __shared__ float shared_new_l;
  __shared__ float shared_alpha;

  const int32_t* block_table =
      block_tables + static_cast<int64_t>(seq_idx) * block_table_stride_seq;

  constexpr uint32_t kPayloadBaseOffset = Layout::k_payload_offset(0, 0);
  constexpr uint32_t vPayloadBaseOffset = Layout::v_payload_offset(0, 0);
  const uint32_t k_payload_base_offset =
      kPayloadBaseOffset + kv_head_idx * Layout::AlignedKPayloadBytesPerKvHead;
  const uint32_t v_payload_base_offset =
      vPayloadBaseOffset + kv_head_idx * Layout::AlignedVPayloadBytesPerKvHead;
  const int dim_tile = dim / Policy::CodecDimBlock;
  const int dim_in_tile = dim % Policy::CodecDimBlock;

  for (int32_t tile_start = partition_start; tile_start < partition_end;
       tile_start += Policy::ComputeBlockN) {
    const int32_t tile_end =
        min(tile_start + Policy::ComputeBlockN, partition_end);
    const int32_t tile_len = tile_end - tile_start;

    const int32_t first_logical_block = tile_start / kBlockSize;
    const int32_t last_logical_block = (tile_end + kBlockSize - 1) / kBlockSize;
#pragma unroll 1
    for (int32_t logical_block = first_logical_block;
         logical_block < last_logical_block; ++logical_block) {
      const int32_t block_token_start = logical_block * kBlockSize;
      const int row_start = max(tile_start - block_token_start, 0);
      const int row_end = min(tile_end - block_token_start, kBlockSize);
      if (logical_block >= max_num_blocks_per_seq) {
        if (threadIdx.x == 0) {
#pragma unroll 1
          for (int row = row_start; row < row_end; ++row) {
            shared_scores[block_token_start + row - tile_start] = -FLT_MAX;
          }
        }
        continue;
      }

      const int32_t physical_block = block_table[logical_block];
      if (physical_block < 0) {
        if (threadIdx.x == 0) {
#pragma unroll 1
          for (int row = row_start; row < row_end; ++row) {
            shared_scores[block_token_start + row - tile_start] = -FLT_MAX;
          }
        }
        continue;
      }

      const uint8_t* page = kv_cache + static_cast<int64_t>(physical_block) *
                                           kv_cache_stride_block;
      bool page_is_unsafe = false;
      if constexpr (UsePageUnsafeFlags) {
        page_is_unsafe = page_unsafe_flags[physical_block] != 0;
      }
      bool use_payload_masks = !AssumeNoFallbackNoOutlier;
      if constexpr (AssumeNoFallbackNoOutlier && UsePageUnsafeFlags) {
        use_payload_masks = page_is_unsafe;
      }
      const auto k_desc = byte_v2_make_payload_tile_descriptor<Layout, false>(
          page, k_payload_base_offset, kv_head_idx, dim_tile,
          use_payload_masks);
#pragma unroll 1
      for (int row = row_start; row < row_end; ++row) {
        const int32_t tile_offset = block_token_start + row - tile_start;
        float k_value;
        if constexpr (UseRawFallback) {
          k_value =
              byte_v2_load_raw_elem<Layout, false>(page, kv_head_idx, row, dim);
        } else {
          k_value =
              byte_v2_load_payload_elem_from_tile_descriptor<Layout, false>(
                  k_desc, kv_head_idx, row, dim, dim_in_tile);
        }
        const float qk_partial = q_value * k_value;
        const float qk_sum =
            byte_v2_block_sum_thread0<NumThreads>(qk_partial, reduce_smem);

        if (threadIdx.x == 0) {
          shared_scores[tile_offset] = qk_sum * scale;
        }
      }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
      float tile_m = -FLT_MAX;
#pragma unroll 1
      for (int32_t tile_offset = 0; tile_offset < tile_len; ++tile_offset) {
        tile_m = fmaxf(tile_m, shared_scores[tile_offset]);
      }
      const float new_m = fmaxf(softmax_m, tile_m);
      const float alpha = __expf(softmax_m - new_m);
      float tile_l = 0.0f;
#pragma unroll 1
      for (int32_t tile_offset = 0; tile_offset < tile_len; ++tile_offset) {
        const float prob = __expf(shared_scores[tile_offset] - new_m);
        shared_probs[tile_offset] = prob;
        tile_l += prob;
      }
      shared_alpha = alpha;
      shared_new_m = new_m;
      shared_new_l = softmax_l * alpha + tile_l;
    }
    __syncthreads();

    float pv = 0.0f;
#pragma unroll 1
    for (int32_t logical_block = first_logical_block;
         logical_block < last_logical_block; ++logical_block) {
      if (logical_block >= max_num_blocks_per_seq) {
        continue;
      }
      const int32_t physical_block = block_table[logical_block];
      if (physical_block < 0) {
        continue;
      }

      const int32_t block_token_start = logical_block * kBlockSize;
      const int row_start = max(tile_start - block_token_start, 0);
      const int row_end = min(tile_end - block_token_start, kBlockSize);
      const uint8_t* page = kv_cache + static_cast<int64_t>(physical_block) *
                                           kv_cache_stride_block;
      bool page_is_unsafe = false;
      if constexpr (UsePageUnsafeFlags) {
        page_is_unsafe = page_unsafe_flags[physical_block] != 0;
      }
      bool use_payload_masks = !AssumeNoFallbackNoOutlier;
      if constexpr (AssumeNoFallbackNoOutlier && UsePageUnsafeFlags) {
        use_payload_masks = page_is_unsafe;
      }
      const auto v_desc = byte_v2_make_payload_tile_descriptor<Layout, true>(
          page, v_payload_base_offset, kv_head_idx, dim_tile,
          use_payload_masks);
#pragma unroll 1
      for (int row = row_start; row < row_end; ++row) {
        const int32_t tile_offset = block_token_start + row - tile_start;
        float v_value;
        if constexpr (UseRawFallback) {
          v_value =
              byte_v2_load_raw_elem<Layout, true>(page, kv_head_idx, row, dim);
        } else {
          v_value =
              byte_v2_load_payload_elem_from_tile_descriptor<Layout, true>(
                  v_desc, kv_head_idx, row, dim, dim_in_tile);
        }
        pv += shared_probs[tile_offset] * v_value;
      }
    }
    acc = acc * shared_alpha + pv;
    softmax_l = shared_new_l;
    softmax_m = shared_new_m;
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    exp_sums[stats_offset] =
        softmax_l > 0.0f ? logf(softmax_l) + softmax_m : -FLT_MAX;
  }
  tmp_out[tmp_base + dim] = softmax_l > 0.0f ? acc / softmax_l : 0.0f;
}

template <typename Layout, bool UsePageUnsafeFlags, int NumThreads>
__global__ void
byte_v2_paged_decode_attention_split_k_gqa4_no_fallback_no_outlier_kernel(
    float* __restrict__ tmp_out, float* __restrict__ exp_sums,
    float* __restrict__ max_logits, const uint16_t* __restrict__ query,
    const uint8_t* __restrict__ kv_cache,
    const int32_t* __restrict__ page_unsafe_flags,
    const int32_t* __restrict__ block_tables,
    const int32_t* __restrict__ seq_lens, float scale, int64_t q_stride_token,
    int64_t q_stride_head, int64_t q_stride_dim, int64_t kv_cache_stride_block,
    int64_t block_table_stride_seq, int max_num_blocks_per_seq,
    int max_num_partitions, int partition_size) {
  using Policy = typename Layout::TilePolicy;
  (void)max_logits;

  constexpr int kBlockSize = Policy::AllocBlockTokens;
  constexpr int kQPerKv = 4;
  constexpr int kNumQHeads = Layout::NumKvHeadsValue * kQPerKv;

  const int seq_idx = blockIdx.y;
  const int kv_head_idx = blockIdx.x;
  const int partition_idx = blockIdx.z;
  const int dim = threadIdx.x;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int32_t seq_len = seq_lens[seq_idx];
  const int32_t partition_start = partition_idx * partition_size;
  const int32_t partition_end =
      min(partition_start + partition_size, static_cast<int32_t>(seq_len));
  static_assert(NumThreads == kQPerKv * 32);
  static_assert(Policy::HeadDim % 32 == 0);
  constexpr int kQkDimsPerThread = Policy::HeadDim / 32;

  const int64_t stats_base =
      (static_cast<int64_t>(seq_idx) * kNumQHeads + kv_head_idx * kQPerKv) *
          max_num_partitions +
      partition_idx;
  const int64_t tmp_base = stats_base * static_cast<int64_t>(Policy::HeadDimV);

  if (partition_start >= seq_len || partition_start >= partition_end) {
    if (threadIdx.x == 0) {
#pragma unroll
      for (int q_group = 0; q_group < kQPerKv; ++q_group) {
        exp_sums[stats_base +
                 static_cast<int64_t>(q_group) * max_num_partitions] = -FLT_MAX;
      }
    }
#pragma unroll
    for (int q_group = 0; q_group < kQPerKv; ++q_group) {
      tmp_out[tmp_base +
              static_cast<int64_t>(q_group) * max_num_partitions *
                  Policy::HeadDimV +
              dim] = 0.0f;
    }
    return;
  }

  float q_values[kQkDimsPerThread];
#pragma unroll
  for (int q_dim_iter = 0; q_dim_iter < kQkDimsPerThread; ++q_dim_iter) {
    const int head_idx = kv_head_idx * kQPerKv + warp;
    const int q_dim = q_dim_iter * 32 + lane;
    const int64_t q_offset = static_cast<int64_t>(seq_idx) * q_stride_token +
                             static_cast<int64_t>(head_idx) * q_stride_head +
                             q_dim * q_stride_dim;
    q_values[q_dim_iter] = byte_v2_bf16_bits_to_float(query[q_offset]);
  }

  float acc[kQPerKv];
  float softmax_m[kQPerKv];
  float softmax_l[kQPerKv];
#pragma unroll
  for (int q_group = 0; q_group < kQPerKv; ++q_group) {
    acc[q_group] = 0.0f;
    softmax_m[q_group] = -FLT_MAX;
    softmax_l[q_group] = 0.0f;
  }

  __shared__ float shared_scores[kQPerKv][Policy::ComputeBlockN];
  __shared__ float shared_probs[kQPerKv][Policy::ComputeBlockN];
  __shared__ float shared_new_m[kQPerKv];
  __shared__ float shared_new_l[kQPerKv];
  __shared__ float shared_alpha[kQPerKv];
  __shared__ uint16_t shared_k_tile[Policy::AllocBlockTokens][Policy::HeadDim];

  const int32_t* block_table =
      block_tables + static_cast<int64_t>(seq_idx) * block_table_stride_seq;

  constexpr uint32_t kPayloadBaseOffset = Layout::k_payload_offset(0, 0);
  constexpr uint32_t vPayloadBaseOffset = Layout::v_payload_offset(0, 0);
  const uint32_t k_payload_base_offset =
      kPayloadBaseOffset + kv_head_idx * Layout::AlignedKPayloadBytesPerKvHead;
  const uint32_t v_payload_base_offset =
      vPayloadBaseOffset + kv_head_idx * Layout::AlignedVPayloadBytesPerKvHead;
  const int v_dim_tile = dim / Policy::CodecDimBlock;
  const int v_dim_in_tile = dim % Policy::CodecDimBlock;

  for (int32_t tile_start = partition_start; tile_start < partition_end;
       tile_start += Policy::ComputeBlockN) {
    const int32_t tile_end =
        min(tile_start + Policy::ComputeBlockN, partition_end);
    const int32_t tile_len = tile_end - tile_start;

    const int32_t first_logical_block = tile_start / kBlockSize;
    const int32_t last_logical_block = (tile_end + kBlockSize - 1) / kBlockSize;
#pragma unroll 1
    for (int32_t logical_block = first_logical_block;
         logical_block < last_logical_block; ++logical_block) {
      const int32_t block_token_start = logical_block * kBlockSize;
      const int row_start = max(tile_start - block_token_start, 0);
      const int row_end = min(tile_end - block_token_start, kBlockSize);
      if (logical_block >= max_num_blocks_per_seq) {
        if (threadIdx.x == 0) {
#pragma unroll 1
          for (int row = row_start; row < row_end; ++row) {
            const int32_t tile_offset = block_token_start + row - tile_start;
#pragma unroll
            for (int q_group = 0; q_group < kQPerKv; ++q_group) {
              shared_scores[q_group][tile_offset] = -FLT_MAX;
            }
          }
        }
        continue;
      }

      const int32_t physical_block = block_table[logical_block];
      if (physical_block < 0) {
        if (threadIdx.x == 0) {
#pragma unroll 1
          for (int row = row_start; row < row_end; ++row) {
            const int32_t tile_offset = block_token_start + row - tile_start;
#pragma unroll
            for (int q_group = 0; q_group < kQPerKv; ++q_group) {
              shared_scores[q_group][tile_offset] = -FLT_MAX;
            }
          }
        }
        continue;
      }

      const uint8_t* page = kv_cache + static_cast<int64_t>(physical_block) *
                                           kv_cache_stride_block;
      bool page_is_unsafe = false;
      if constexpr (UsePageUnsafeFlags) {
        page_is_unsafe = page_unsafe_flags[physical_block] != 0;
      }
      if (page_is_unsafe) {
#pragma unroll 1
        for (int row = row_start; row < row_end; ++row) {
          const int32_t tile_offset = block_token_start + row - tile_start;
          float qk_sum = 0.0f;
#pragma unroll
          for (int q_dim_iter = 0; q_dim_iter < kQkDimsPerThread;
               ++q_dim_iter) {
            const int q_dim = q_dim_iter * 32 + lane;
            const float k_value = byte_v2_load_payload_elem<Layout, false>(
                page, k_payload_base_offset, kv_head_idx, row, q_dim);
            qk_sum += q_values[q_dim_iter] * k_value;
          }
#pragma unroll
          for (int offset = 16; offset > 0; offset >>= 1) {
            qk_sum += __shfl_down_sync(0xffffffff, qk_sum, offset);
          }
          if (lane == 0) {
            shared_scores[warp][tile_offset] = qk_sum * scale;
          }
        }
      } else {
        const int staged_rows = row_end - row_start;
        const int staged_elems = staged_rows * Policy::HeadDim;
        const int staged_pairs = staged_elems / 2;
        const int thread_k_dim = (threadIdx.x * 2) & (Policy::HeadDim - 1);
        const int thread_dim_tile = thread_k_dim / Policy::CodecDimBlock;
        const int thread_dim_in_tile = thread_k_dim % Policy::CodecDimBlock;
        const uint32_t thread_k_tile_offset =
            k_payload_base_offset +
            thread_dim_tile * Layout::CodecPayloadBytesPerTile;
        const uint8_t thread_k_base =
            page[Layout::k_base_offset(kv_head_idx, thread_dim_tile, 0)];
#pragma unroll 1
        for (int pair = threadIdx.x; pair < staged_pairs; pair += NumThreads) {
          const int elem = pair * 2;
          const int row_rel = elem / Policy::HeadDim;
          const int row = row_start + row_rel;
          uint16_t bits0;
          uint16_t bits1;
          byte_v2_load_k_payload_elem_pair_from_fixed_tile_no_outlier_bits<
              Layout>(page, thread_k_tile_offset, row, thread_dim_in_tile,
                      thread_k_base, bits0, bits1);
          shared_k_tile[row][thread_k_dim] = bits0;
          shared_k_tile[row][thread_k_dim + 1] = bits1;
        }
        __syncthreads();
#pragma unroll 1
        for (int row = row_start; row < row_end; ++row) {
          const int32_t tile_offset = block_token_start + row - tile_start;
          float qk_sum = 0.0f;
#pragma unroll
          for (int q_dim_iter = 0; q_dim_iter < kQkDimsPerThread;
               ++q_dim_iter) {
            const int q_dim = q_dim_iter * 32 + lane;
            const float k_value =
                byte_v2_bf16_bits_to_float(shared_k_tile[row][q_dim]);
            qk_sum += q_values[q_dim_iter] * k_value;
          }
#pragma unroll
          for (int offset = 16; offset > 0; offset >>= 1) {
            qk_sum += __shfl_down_sync(0xffffffff, qk_sum, offset);
          }
          if (lane == 0) {
            shared_scores[warp][tile_offset] = qk_sum * scale;
          }
        }
        __syncthreads();
      }
    }
    __syncthreads();

    float tile_m = -FLT_MAX;
#pragma unroll 1
    for (int32_t tile_offset = lane; tile_offset < tile_len;
         tile_offset += 32) {
      tile_m = fmaxf(tile_m, shared_scores[warp][tile_offset]);
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      tile_m = fmaxf(tile_m, __shfl_down_sync(0xffffffff, tile_m, offset));
    }
    tile_m = __shfl_sync(0xffffffff, tile_m, 0);
    const float new_m = fmaxf(softmax_m[warp], tile_m);
    const float alpha = __expf(softmax_m[warp] - new_m);

    float tile_l = 0.0f;
#pragma unroll 1
    for (int32_t tile_offset = lane; tile_offset < tile_len;
         tile_offset += 32) {
      const float prob = __expf(shared_scores[warp][tile_offset] - new_m);
      shared_probs[warp][tile_offset] = prob;
      tile_l += prob;
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      tile_l += __shfl_down_sync(0xffffffff, tile_l, offset);
    }
    if (lane == 0) {
      shared_alpha[warp] = alpha;
      shared_new_m[warp] = new_m;
      shared_new_l[warp] = softmax_l[warp] * alpha + tile_l;
    }
    __syncthreads();

    float pv[kQPerKv];
#pragma unroll
    for (int q_group = 0; q_group < kQPerKv; ++q_group) {
      pv[q_group] = 0.0f;
    }
#pragma unroll 1
    for (int32_t logical_block = first_logical_block;
         logical_block < last_logical_block; ++logical_block) {
      if (logical_block >= max_num_blocks_per_seq) {
        continue;
      }
      const int32_t physical_block = block_table[logical_block];
      if (physical_block < 0) {
        continue;
      }

      const int32_t block_token_start = logical_block * kBlockSize;
      const int row_start = max(tile_start - block_token_start, 0);
      const int row_end = min(tile_end - block_token_start, kBlockSize);
      const uint8_t* page = kv_cache + static_cast<int64_t>(physical_block) *
                                           kv_cache_stride_block;
      bool page_is_unsafe = false;
      if constexpr (UsePageUnsafeFlags) {
        page_is_unsafe = page_unsafe_flags[physical_block] != 0;
      }
      if (page_is_unsafe) {
#pragma unroll 1
        for (int row = row_start; row < row_end; ++row) {
          const int32_t tile_offset = block_token_start + row - tile_start;
          const float v_value = byte_v2_load_payload_elem<Layout, true>(
              page, v_payload_base_offset, kv_head_idx, row, dim);
          for (int q_group = 0; q_group < kQPerKv; ++q_group) {
            pv[q_group] += shared_probs[q_group][tile_offset] * v_value;
          }
        }
      } else {
        const auto v_desc = byte_v2_make_payload_tile_descriptor<Layout, true>(
            page, v_payload_base_offset, kv_head_idx, v_dim_tile, false);
#pragma unroll 1
        for (int row = row_start; row < row_end; ++row) {
          const int32_t tile_offset = block_token_start + row - tile_start;
          const float v_value =
              byte_v2_load_payload_elem_from_safe_tile_descriptor_fixed_dim<
                  Layout>(v_desc, row, v_dim_in_tile);
          for (int q_group = 0; q_group < kQPerKv; ++q_group) {
            pv[q_group] += shared_probs[q_group][tile_offset] * v_value;
          }
        }
      }
    }
#pragma unroll
    for (int q_group = 0; q_group < kQPerKv; ++q_group) {
      acc[q_group] = acc[q_group] * shared_alpha[q_group] + pv[q_group];
      softmax_l[q_group] = shared_new_l[q_group];
      softmax_m[q_group] = shared_new_m[q_group];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
#pragma unroll
    for (int q_group = 0; q_group < kQPerKv; ++q_group) {
      const int64_t stats_offset =
          stats_base + static_cast<int64_t>(q_group) * max_num_partitions;
      exp_sums[stats_offset] =
          softmax_l[q_group] > 0.0f
              ? logf(softmax_l[q_group]) + softmax_m[q_group]
              : -FLT_MAX;
    }
  }
#pragma unroll
  for (int q_group = 0; q_group < kQPerKv; ++q_group) {
    const int64_t out_offset =
        tmp_base +
        static_cast<int64_t>(q_group) * max_num_partitions * Policy::HeadDimV +
        dim;
    tmp_out[out_offset] =
        softmax_l[q_group] > 0.0f ? acc[q_group] / softmax_l[q_group] : 0.0f;
  }
}

template <typename Layout, bool UsePageUnsafeFlags, bool UseQkMma,
          bool UseFa2Mainloop, bool UseFa2Multiwarp, bool UseFa2Direct,
          int QHeadsPerKv, int QGroupTile, int NumThreads,
          int DirectDiagnosticMode = kByteV2DirectDiagnosticCurrent,
          int SpeculativeQueryLen = 0, bool UseRaggedSpeculativeQ4 = false>
__global__ void
byte_v2_paged_decode_attention_split_k_gqa4_fa2_like_no_fallback_no_outlier_kernel(
    float* __restrict__ tmp_out, float* __restrict__ exp_sums,
    float* __restrict__ max_logits, const uint16_t* __restrict__ query,
    const uint8_t* __restrict__ kv_cache,
    const int32_t* __restrict__ page_unsafe_flags,
    const int32_t* __restrict__ block_tables,
    const int32_t* __restrict__ seq_lens,
    const int32_t* __restrict__ query_start_locs, int num_actual_tokens,
    float scale, int64_t q_stride_token, int64_t q_stride_head,
    int64_t q_stride_dim, int64_t kv_cache_stride_block,
    int64_t block_table_stride_seq, int max_num_blocks_per_seq,
    int max_num_partitions, int partition_size) {
  using Policy = typename Layout::TilePolicy;
  (void)max_logits;

  constexpr int kBlockSize = Policy::AllocBlockTokens;
  constexpr int kQPerKv = QGroupTile;
  constexpr int kQHeadsPerKv = QHeadsPerKv;
  constexpr int kGroupsPerKv = (QHeadsPerKv + QGroupTile - 1) / QGroupTile;
  constexpr int kNumQHeads = Layout::NumKvHeadsValue * kQHeadsPerKv;

  const int seq_idx = blockIdx.y;
  const int kv_group_idx = blockIdx.x;
  const int kv_head_idx = kv_group_idx / kGroupsPerKv;
  const int q_group_block = kv_group_idx - kv_head_idx * kGroupsPerKv;
  const int q_group_base = q_group_block * QGroupTile;
  int query_start = seq_idx * SpeculativeQueryLen;
  int request_query_len = SpeculativeQueryLen;
  if constexpr (UseRaggedSpeculativeQ4) {
    query_start = max(min(query_start_locs[seq_idx], num_actual_tokens), 0);
    const int query_end =
        max(min(query_start_locs[seq_idx + 1], num_actual_tokens), query_start);
    request_query_len = min(query_end - query_start, SpeculativeQueryLen);
  }
  int valid_q_rows = min(QGroupTile, QHeadsPerKv - q_group_base);
  if constexpr (UseRaggedSpeculativeQ4) {
    valid_q_rows =
        max(min(request_query_len * 4 - q_group_base, QGroupTile), 0);
  }
  const int partition_idx = blockIdx.z;
  const int dim = threadIdx.x;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int32_t seq_len = seq_lens[seq_idx];
  const int32_t effective_partition_size =
      UseFa2Direct ? partition_size : Policy::ComputeBlockN;
  const int32_t partition_start = partition_idx * effective_partition_size;
  const int32_t partition_end = min(partition_start + effective_partition_size,
                                    static_cast<int32_t>(seq_len));
  static_assert(QHeadsPerKv > 0);
  static_assert(QGroupTile > 0);
  static_assert(QGroupTile <= QHeadsPerKv);
  static_assert(!UseFa2Direct || QGroupTile <= 64);
  static_assert(UseFa2Direct || NumThreads == kQPerKv * 32);
  static_assert(Policy::HeadDim % 32 == 0);
  static_assert(Policy::ComputeBlockN == 64);
  static_assert(!UseFa2Mainloop || UseQkMma);
  static_assert(!UseFa2Multiwarp || UseQkMma);
  static_assert(!UseFa2Direct || UseQkMma);
  static_assert(!(UseFa2Direct && UseFa2Mainloop));
  static_assert(!(UseFa2Direct && UseFa2Multiwarp));
  static_assert(SpeculativeQueryLen == 0 || UseFa2Direct);
  static_assert(SpeculativeQueryLen == 0 ||
                QHeadsPerKv == SpeculativeQueryLen * 4);
  static_assert(SpeculativeQueryLen == 0 || QGroupTile <= 64);
  static_assert(!UseRaggedSpeculativeQ4 || SpeculativeQueryLen == 4);
  static_assert(!UseRaggedSpeculativeQ4 || QHeadsPerKv == 16);
  static_assert(DirectDiagnosticMode >= kByteV2DirectDiagnosticCurrent &&
                DirectDiagnosticMode <= kByteV2DirectDiagnosticCodeReady);
  static_assert(UseFa2Direct ||
                DirectDiagnosticMode == kByteV2DirectDiagnosticCurrent);
  using CutePvTraits =
      ByteV2Fa2LikeCutePvTraits<16, Policy::ComputeBlockN, Policy::HeadDimV, 1>;
  using CutePvTileTraits =
      ByteV2Fa2LikeCutePvTileTraits<16, Policy::ComputeBlockN, 16>;
  using CuteQkTraits =
      ByteV2Fa2LikeCuteQkTraits<16, Policy::ComputeBlockN, Policy::HeadDim, 1>;
  using CuteDirectPvTraits =
      ByteV2Fa2LikeCutePvTraits<16, Policy::ComputeBlockN, Policy::HeadDimV, 1>;
  using CuteDirectQkTraits =
      ByteV2Fa2LikeCuteQkTraits<16, Policy::ComputeBlockN, Policy::HeadDim, 1>;
  static_assert(CutePvTraits::kBlockN == Policy::ComputeBlockN);
  static_assert(CutePvTraits::kHeadDim == Policy::HeadDimV);
  static_assert(CutePvTileTraits::kBlockK == Policy::ComputeBlockN);
  static_assert(CutePvTileTraits::kBlockN == 16);
  static_assert(CuteQkTraits::kBlockN == Policy::ComputeBlockN);
  static_assert(CuteQkTraits::kHeadDim == Policy::HeadDim);
  static_assert(CuteDirectPvTraits::kBlockN == Policy::ComputeBlockN);
  static_assert(CuteDirectPvTraits::kHeadDim == Policy::HeadDimV);
  static_assert(CuteDirectQkTraits::kBlockN == Policy::ComputeBlockN);
  static_assert(CuteDirectQkTraits::kHeadDim == Policy::HeadDim);
  static_assert(CuteDirectPvTraits::kBlockM == CuteDirectQkTraits::kBlockM);
  constexpr int kDirectBlockM = CuteDirectQkTraits::kBlockM;
  constexpr int kDirectComputeWarps =
      (kQPerKv + kDirectBlockM - 1) / kDirectBlockM;
  static_assert(kDirectBlockM == 16);
  static_assert(kDirectComputeWarps <= NumThreads / 32);
  constexpr bool kDirectDiagPhaseProfile =
      DirectDiagnosticMode == kByteV2DirectDiagnosticPhaseProfile ||
      DirectDiagnosticMode == kByteV2DirectDiagnosticEffectiveM16Profile;
  constexpr bool kDirectDiagStageWindow =
      DirectDiagnosticMode == kByteV2DirectDiagnosticStageWindow16 ||
      DirectDiagnosticMode == kByteV2DirectDiagnosticStageWindow32 ||
      DirectDiagnosticMode == kByteV2DirectDiagnosticStageWindow16Specialized;
  constexpr bool kDirectDiagStageWindow16Specialized =
      DirectDiagnosticMode == kByteV2DirectDiagnosticStageWindow16Specialized;
  constexpr bool kDirectDiagClockProfile =
      kDirectDiagPhaseProfile || kDirectDiagStageWindow;
  constexpr bool kDirectDiagUnnormalizedPartitionOutput =
      DirectDiagnosticMode ==
      kByteV2DirectDiagnosticUnnormalizedPartitionOutput;
  constexpr bool kDirectDiagCurrentLike =
      DirectDiagnosticMode == kByteV2DirectDiagnosticCurrent ||
      DirectDiagnosticMode == kByteV2DirectDiagnosticCodeReady ||
      kDirectDiagPhaseProfile || kDirectDiagUnnormalizedPartitionOutput;
  constexpr bool kDirectDiagStageOnly =
      DirectDiagnosticMode == kByteV2DirectDiagnosticDecodeStageOnly ||
      DirectDiagnosticMode == kByteV2DirectDiagnosticKDecodeStageOnly ||
      DirectDiagnosticMode == kByteV2DirectDiagnosticVDecodeStageOnly ||
      DirectDiagnosticMode ==
          kByteV2DirectDiagnosticKDecodeProducer3StageOnly ||
      DirectDiagnosticMode ==
          kByteV2DirectDiagnosticVDecodeProducer3StageOnly ||
      DirectDiagnosticMode ==
          kByteV2DirectDiagnosticKDecodeProducer2StageOnly ||
      DirectDiagnosticMode ==
          kByteV2DirectDiagnosticVDecodeProducer2StageOnly ||
      DirectDiagnosticMode == kByteV2DirectDiagnosticKDecodeNoStoreStageOnly ||
      DirectDiagnosticMode == kByteV2DirectDiagnosticVDecodeNoStoreStageOnly ||
      DirectDiagnosticMode == kByteV2DirectDiagnosticKDecodeLowOnlyStageOnly ||
      DirectDiagnosticMode == kByteV2DirectDiagnosticVDecodeLowOnlyStageOnly ||
      DirectDiagnosticMode == kByteV2DirectDiagnosticKDecodeHighOnlyStageOnly ||
      DirectDiagnosticMode == kByteV2DirectDiagnosticVDecodeHighOnlyStageOnly ||
      kDirectDiagStageWindow;
  constexpr bool kDirectDiagNeedsCompressedK =
      kDirectDiagCurrentLike ||
      DirectDiagnosticMode == kByteV2DirectDiagnosticDecodeStageOnly ||
      DirectDiagnosticMode == kByteV2DirectDiagnosticKDecodeStageOnly ||
      DirectDiagnosticMode == kByteV2DirectDiagnosticKDecodeQkOnly ||
      DirectDiagnosticMode == kByteV2DirectDiagnosticKDecodeQkNoOutput ||
      DirectDiagnosticMode ==
          kByteV2DirectDiagnosticKDecodeProducer3StageOnly ||
      DirectDiagnosticMode ==
          kByteV2DirectDiagnosticKDecodeProducer2StageOnly ||
      DirectDiagnosticMode == kByteV2DirectDiagnosticKDecodeNoStoreStageOnly ||
      DirectDiagnosticMode == kByteV2DirectDiagnosticKDecodeLowOnlyStageOnly ||
      DirectDiagnosticMode == kByteV2DirectDiagnosticKDecodeHighOnlyStageOnly ||
      kDirectDiagStageWindow;
  constexpr bool kDirectDiagNeedsCompressedV =
      kDirectDiagCurrentLike ||
      DirectDiagnosticMode == kByteV2DirectDiagnosticDecodeStageOnly ||
      DirectDiagnosticMode == kByteV2DirectDiagnosticVDecodeStageOnly ||
      DirectDiagnosticMode == kByteV2DirectDiagnosticVDecodePvOnly ||
      DirectDiagnosticMode == kByteV2DirectDiagnosticVDecodePvNoGemmOnly ||
      DirectDiagnosticMode == kByteV2DirectDiagnosticVDecodePvNoAccumOnly ||
      DirectDiagnosticMode ==
          kByteV2DirectDiagnosticVDecodeProducer3StageOnly ||
      DirectDiagnosticMode ==
          kByteV2DirectDiagnosticVDecodeProducer2StageOnly ||
      DirectDiagnosticMode == kByteV2DirectDiagnosticVDecodeNoStoreStageOnly ||
      DirectDiagnosticMode == kByteV2DirectDiagnosticVDecodeLowOnlyStageOnly ||
      DirectDiagnosticMode == kByteV2DirectDiagnosticVDecodeHighOnlyStageOnly ||
      kDirectDiagStageWindow;
  constexpr bool kDirectDiagDecodeNoStore =
      DirectDiagnosticMode == kByteV2DirectDiagnosticKDecodeNoStoreStageOnly ||
      DirectDiagnosticMode == kByteV2DirectDiagnosticVDecodeNoStoreStageOnly;
  constexpr bool kDirectDiagDecodeLowOnly =
      DirectDiagnosticMode == kByteV2DirectDiagnosticKDecodeLowOnlyStageOnly ||
      DirectDiagnosticMode == kByteV2DirectDiagnosticVDecodeLowOnlyStageOnly;
  constexpr bool kDirectDiagDecodeHighOnly =
      DirectDiagnosticMode == kByteV2DirectDiagnosticKDecodeHighOnlyStageOnly ||
      DirectDiagnosticMode == kByteV2DirectDiagnosticVDecodeHighOnlyStageOnly;
  constexpr bool kDirectDiagStoreDecodedK =
      kDirectDiagNeedsCompressedK && !kDirectDiagDecodeNoStore &&
      !kDirectDiagDecodeLowOnly && !kDirectDiagDecodeHighOnly;
  constexpr bool kDirectDiagStoreDecodedV =
      kDirectDiagNeedsCompressedV && !kDirectDiagDecodeNoStore &&
      !kDirectDiagDecodeLowOnly && !kDirectDiagDecodeHighOnly;
  constexpr bool kDirectDiagNeedsKSmem =
      kDirectDiagStoreDecodedK ||
      DirectDiagnosticMode == kByteV2DirectDiagnosticFakeDecodeZero ||
      DirectDiagnosticMode == kByteV2DirectDiagnosticRawSameSkeleton;
  constexpr bool kDirectDiagNeedsVSmem =
      kDirectDiagStoreDecodedV ||
      DirectDiagnosticMode == kByteV2DirectDiagnosticFakeDecodeZero ||
      DirectDiagnosticMode == kByteV2DirectDiagnosticRawSameSkeleton;
  constexpr bool kDirectDiagUseFakePvProbs =
      DirectDiagnosticMode == kByteV2DirectDiagnosticVDecodePvOnly ||
      DirectDiagnosticMode == kByteV2DirectDiagnosticVDecodePvNoGemmOnly ||
      DirectDiagnosticMode == kByteV2DirectDiagnosticVDecodePvNoAccumOnly;
  constexpr bool kDirectDiagSkipPv =
      DirectDiagnosticMode == kByteV2DirectDiagnosticKDecodeQkOnly ||
      DirectDiagnosticMode == kByteV2DirectDiagnosticKDecodeQkNoOutput;
  constexpr bool kDirectDiagSkipFinalOutput =
      DirectDiagnosticMode == kByteV2DirectDiagnosticKDecodeQkNoOutput;
  constexpr bool kDirectDiagSkipPvGemm =
      DirectDiagnosticMode == kByteV2DirectDiagnosticVDecodePvNoGemmOnly;
  constexpr bool kDirectDiagSkipPvAccum =
      DirectDiagnosticMode == kByteV2DirectDiagnosticVDecodePvNoAccumOnly;
  constexpr bool kDirectDiagProducer3StageOnly =
      DirectDiagnosticMode ==
          kByteV2DirectDiagnosticKDecodeProducer3StageOnly ||
      DirectDiagnosticMode == kByteV2DirectDiagnosticVDecodeProducer3StageOnly;
  constexpr bool kDirectDiagProducer2StageOnly =
      DirectDiagnosticMode ==
          kByteV2DirectDiagnosticKDecodeProducer2StageOnly ||
      DirectDiagnosticMode == kByteV2DirectDiagnosticVDecodeProducer2StageOnly;
  constexpr int kDirectDiagStagingWarpCount =
      kDirectDiagProducer3StageOnly ? 3
                                    : (kDirectDiagProducer2StageOnly ? 2 : 4);
  constexpr int kDirectDiagStagingThreadBase =
      (4 - kDirectDiagStagingWarpCount) * 32;
  constexpr int kDirectDiagStagingThreads = kDirectDiagStagingWarpCount * 32;
  constexpr int kDirectDiagStageWindowDim =
      (DirectDiagnosticMode == kByteV2DirectDiagnosticStageWindow16 ||
       kDirectDiagStageWindow16Specialized)
          ? 16
          : (DirectDiagnosticMode == kByteV2DirectDiagnosticStageWindow32
                 ? 32
                 : 128);
  static_assert(kDirectDiagStageWindowDim % 16 == 0);
  static_assert(kDirectDiagStageWindowDim <= Policy::HeadDim);
  constexpr int kDirectDiagStageWindowHexes = kDirectDiagStageWindowDim / 16;
  // A half warp owns all 16 rows of one dim hex in speculative CTAs. This
  // matches the CuTe shared layout and avoids conflicting row-wide stores.
  constexpr bool kDirectSpecRowFastStaging = SpeculativeQueryLen > 0;

  const int64_t stats_base = (static_cast<int64_t>(seq_idx) * kNumQHeads +
                              kv_head_idx * kQHeadsPerKv + q_group_base) *
                                 max_num_partitions +
                             partition_idx;
  const int64_t tmp_base = stats_base * static_cast<int64_t>(Policy::HeadDimV);

  if constexpr (UseRaggedSpeculativeQ4) {
    if (valid_q_rows == 0) {
      return;
    }
  }

  if (partition_start >= seq_len || partition_start >= partition_end) {
    if (threadIdx.x == 0) {
#pragma unroll
      for (int q_group = 0; q_group < kQPerKv; ++q_group) {
        if (q_group < valid_q_rows) {
          exp_sums[stats_base + static_cast<int64_t>(q_group) *
                                    max_num_partitions] = -FLT_MAX;
        }
      }
    }
#pragma unroll
    for (int q_group = 0; q_group < kQPerKv; ++q_group) {
      if (q_group < valid_q_rows) {
        tmp_out[tmp_base +
                static_cast<int64_t>(q_group) * max_num_partitions *
                    Policy::HeadDimV +
                dim] = 0.0f;
      }
    }
    return;
  }

  constexpr int kStateRows = UseFa2Direct ? kDirectBlockM : kQPerKv;
  float softmax_m[kStateRows];
  float softmax_l[kStateRows];
#pragma unroll
  for (int q_group = 0; q_group < kStateRows; ++q_group) {
    softmax_m[q_group] = -FLT_MAX;
    softmax_l[q_group] = 0.0f;
  }

  __shared__ float shared_scores[kQPerKv][Policy::ComputeBlockN];
  __shared__ float shared_probs[kQPerKv][Policy::ComputeBlockN];
  __shared__ float shared_new_m[kQPerKv];
  __shared__ float shared_new_l[kQPerKv];
  __shared__ __align__(16) cutlass::bfloat16_t
      shared_v_cute[kQPerKv][CutePvTileTraits::kVStorageElems];

  const int32_t* block_table =
      block_tables + static_cast<int64_t>(seq_idx) * block_table_stride_seq;

  constexpr uint32_t kPayloadBaseOffset = Layout::k_payload_offset(0, 0);
  constexpr uint32_t vPayloadBaseOffset = Layout::v_payload_offset(0, 0);
  const uint32_t k_payload_base_offset =
      kPayloadBaseOffset + kv_head_idx * Layout::AlignedKPayloadBytesPerKvHead;
  const uint32_t v_payload_base_offset =
      vPayloadBaseOffset + kv_head_idx * Layout::AlignedVPayloadBytesPerKvHead;

  const int32_t tile_start = partition_start;
  const int32_t tile_end = partition_end;
  const int32_t tile_len = tile_end - tile_start;
  const int32_t first_logical_block = tile_start / kBlockSize;
  const int32_t last_logical_block = (tile_end + kBlockSize - 1) / kBlockSize;

  if constexpr (UseFa2Direct) {
    __shared__ __align__(16) typename CuteDirectQkTraits::Element
        shared_q_cute[kDirectComputeWarps * CuteDirectQkTraits::kQStorageElems];
    __shared__ __align__(16) typename CuteDirectQkTraits::Element
        shared_k_cute[CuteDirectQkTraits::kKStorageElems];
    __shared__ __align__(16) typename CuteDirectPvTraits::Element
        shared_v_full[CuteDirectPvTraits::kVStorageElems];

    const int direct_compute_warp = min(warp, kDirectComputeWarps - 1);
    const int direct_row_base = direct_compute_warp * kDirectBlockM;
    const int direct_valid_q_rows =
        max(min(valid_q_rows - direct_row_base, kDirectBlockM), 0);
    auto sQ = cute::make_tensor(
        cute::make_smem_ptr(&shared_q_cute[direct_compute_warp *
                                           CuteDirectQkTraits::kQStorageElems]),
        typename CuteDirectQkTraits::SmemLayoutQ{});
    auto sK = cute::make_tensor(cute::make_smem_ptr(&shared_k_cute[0]),
                                typename CuteDirectQkTraits::SmemLayoutKV{});
    auto sV = cute::make_tensor(cute::make_smem_ptr(&shared_v_full[0]),
                                typename CuteDirectPvTraits::SmemLayoutKV{});
    uint32_t direct_diag_decode_sink = 0;
    const bool direct_phase_profile_block = max_logits != nullptr &&
                                            blockIdx.x == 0 &&
                                            blockIdx.y == 0 && blockIdx.z == 0;
    uint64_t direct_phase_total_start = 0;
    uint64_t direct_phase_stage_cycles = 0;
    uint64_t direct_phase_qk_cycles = 0;
    uint64_t direct_phase_softmax_cycles = 0;
    uint64_t direct_phase_pv_cycles = 0;
    uint64_t direct_phase_stage_wait_cycles = 0;
    uint64_t direct_phase_compute_wait_cycles = 0;
    uint64_t direct_phase_compute_region_cycles = 0;
    int direct_phase_tile_count = 0;
    if constexpr (kDirectDiagPhaseProfile) {
      if (direct_phase_profile_block && lane == 0) {
        direct_phase_total_start = clock64();
      }
    }

    const bool direct_q_async = q_stride_dim == 1;
    if (direct_q_async) {
      constexpr int kQChunkElems = 16 / sizeof(uint16_t);
      constexpr int kQChunksPerRow = Policy::HeadDim / kQChunkElems;
      static_assert(Policy::HeadDim % kQChunkElems == 0);
      const auto q_layout = typename CuteDirectQkTraits::SmemLayoutQ{};
#pragma unroll 1
      for (int chunk = threadIdx.x;
           chunk < kDirectComputeWarps * kDirectBlockM * kQChunksPerRow;
           chunk += NumThreads) {
        const int row = chunk / kQChunksPerRow;
        const int col = (chunk - row * kQChunksPerRow) * kQChunkElems;
        const int q_tile = row / kDirectBlockM;
        const int row_in_tile = row - q_tile * kDirectBlockM;
        auto* q_dst =
            &shared_q_cute[q_tile * CuteDirectQkTraits::kQStorageElems +
                           q_layout(row_in_tile, col)];
        if (row < valid_q_rows) {
          int q_token_idx = seq_idx;
          int head_idx = kv_head_idx * kQHeadsPerKv + q_group_base + row;
          if constexpr (SpeculativeQueryLen > 0) {
            const int virtual_row = q_group_base + row;
            q_token_idx = query_start + virtual_row / 4;
            head_idx = kv_head_idx * 4 + virtual_row % 4;
          }
          const int64_t q_offset =
              static_cast<int64_t>(q_token_idx) * q_stride_token +
              static_cast<int64_t>(head_idx) * q_stride_head + col;
          vllm::cuda_async::cp_async_shared_global_16_cg(q_dst,
                                                         query + q_offset);
        } else {
          *reinterpret_cast<int4*>(q_dst) = make_int4(0, 0, 0, 0);
        }
      }
      vllm::cuda_async::cp_async_commit_group();
    } else {
#pragma unroll 1
      for (int elem = threadIdx.x;
           elem < kDirectComputeWarps * kDirectBlockM * Policy::HeadDim;
           elem += NumThreads) {
        const int row = elem / Policy::HeadDim;
        const int col = elem - row * Policy::HeadDim;
        const int q_tile = row / kDirectBlockM;
        const int row_in_tile = row - q_tile * kDirectBlockM;
        uint16_t bits = 0;
        if (row < valid_q_rows) {
          int q_token_idx = seq_idx;
          int head_idx = kv_head_idx * kQHeadsPerKv + q_group_base + row;
          if constexpr (SpeculativeQueryLen > 0) {
            const int virtual_row = q_group_base + row;
            q_token_idx = query_start + virtual_row / 4;
            head_idx = kv_head_idx * 4 + virtual_row % 4;
          }
          const int64_t q_offset =
              static_cast<int64_t>(q_token_idx) * q_stride_token +
              static_cast<int64_t>(head_idx) * q_stride_head +
              col * q_stride_dim;
          bits = query[q_offset];
        }
        const auto q_layout = typename CuteDirectQkTraits::SmemLayoutQ{};
        shared_q_cute[q_tile * CuteDirectQkTraits::kQStorageElems +
                      q_layout(row_in_tile, col)] =
            byte_v2_bf16_bits_to_cutlass_bfloat16(bits);
      }
    }

    typename CuteDirectPvTraits::TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(lane);

    auto acc_o = cute::partition_fragment_C(
        tiled_mma,
        cute::Shape<cute::Int<kDirectBlockM>, cute::Int<Policy::HeadDimV>>{});
    cute::clear(acc_o);

    auto cO = cute::make_identity_tensor(
        cute::Shape<cute::Int<kDirectBlockM>, cute::Int<Policy::HeadDimV>>{});
    auto tCcO = thr_mma.partition_C(cO);
    auto acc_o_rowcol = cute::make_tensor(
        acc_o.data(), byte_v2_convert_layout_acc_rowcol(acc_o.layout()));
    auto tCcO_rowcol = cute::make_tensor(
        tCcO.data(), byte_v2_convert_layout_acc_rowcol(tCcO.layout()));
    constexpr int kDirectThreadRows =
        decltype(cute::size<0>(acc_o_rowcol))::value;
    static_assert(kDirectThreadRows == 2);
    float direct_softmax_m[kDirectThreadRows];
    float direct_softmax_l[kDirectThreadRows];
#pragma unroll
    for (int row = 0; row < kDirectThreadRows; ++row) {
      direct_softmax_m[row] = -FLT_MAX;
      direct_softmax_l[row] = 0.0f;
    }

#pragma unroll 1
    for (int32_t direct_tile_start = partition_start;
         direct_tile_start < partition_end;
         direct_tile_start += Policy::ComputeBlockN) {
      const int32_t direct_tile_end =
          min(direct_tile_start + Policy::ComputeBlockN, partition_end);
      const int32_t direct_tile_len = direct_tile_end - direct_tile_start;
      const int32_t direct_first_logical_block = direct_tile_start / kBlockSize;
      const int32_t direct_last_logical_block =
          (direct_tile_end + kBlockSize - 1) / kBlockSize;
      uint64_t direct_phase_stage_start = 0;
      uint64_t direct_phase_stage_done = 0;
      if constexpr (kDirectDiagClockProfile) {
        if (direct_phase_profile_block && threadIdx.x == 0) {
          direct_phase_stage_start = clock64();
          ++direct_phase_tile_count;
        }
      }

      if (direct_tile_len < Policy::ComputeBlockN) {
#pragma unroll 1
        for (int elem = threadIdx.x;
             elem < Policy::ComputeBlockN * Policy::HeadDim;
             elem += NumThreads) {
          const int row = elem / Policy::HeadDim;
          const int col = elem - row * Policy::HeadDim;
          if constexpr (kDirectDiagNeedsKSmem) {
            sK(row, col) = byte_v2_bf16_bits_to_cutlass_bfloat16(0);
          }
          if constexpr (kDirectDiagNeedsVSmem) {
            sV(row, col) = byte_v2_bf16_bits_to_cutlass_bfloat16(0);
          }
        }
        __syncthreads();
      }

#pragma unroll 1
      for (int32_t logical_block = direct_first_logical_block;
           logical_block < direct_last_logical_block; ++logical_block) {
        const int32_t block_token_start = logical_block * kBlockSize;
        const int row_start = max(direct_tile_start - block_token_start, 0);
        const int row_end =
            min(direct_tile_end - block_token_start, kBlockSize);
        const int staged_rows = row_end - row_start;
        if (logical_block >= max_num_blocks_per_seq) {
#pragma unroll 1
          for (int elem = threadIdx.x; elem < staged_rows * Policy::HeadDim;
               elem += NumThreads) {
            const int row_rel = elem / Policy::HeadDim;
            const int q_dim = elem - row_rel * Policy::HeadDim;
            const int row = row_start + row_rel;
            const int32_t tile_offset =
                block_token_start + row - direct_tile_start;
            if constexpr (kDirectDiagNeedsKSmem) {
              sK(tile_offset, q_dim) = byte_v2_bf16_bits_to_cutlass_bfloat16(0);
            }
            if constexpr (kDirectDiagNeedsVSmem) {
              sV(tile_offset, q_dim) = byte_v2_bf16_bits_to_cutlass_bfloat16(0);
            }
          }
          continue;
        }
        const int32_t physical_block = block_table[logical_block];
        if (physical_block < 0) {
#pragma unroll 1
          for (int elem = threadIdx.x; elem < staged_rows * Policy::HeadDim;
               elem += NumThreads) {
            const int row_rel = elem / Policy::HeadDim;
            const int q_dim = elem - row_rel * Policy::HeadDim;
            const int row = row_start + row_rel;
            const int32_t tile_offset =
                block_token_start + row - direct_tile_start;
            if constexpr (kDirectDiagNeedsKSmem) {
              sK(tile_offset, q_dim) = byte_v2_bf16_bits_to_cutlass_bfloat16(0);
            }
            if constexpr (kDirectDiagNeedsVSmem) {
              sV(tile_offset, q_dim) = byte_v2_bf16_bits_to_cutlass_bfloat16(0);
            }
          }
          continue;
        }

        if constexpr (DirectDiagnosticMode ==
                      kByteV2DirectDiagnosticFakeDecodeZero) {
#pragma unroll 1
          for (int elem = threadIdx.x; elem < staged_rows * Policy::HeadDim;
               elem += NumThreads) {
            const int row_rel = elem / Policy::HeadDim;
            const int q_dim = elem - row_rel * Policy::HeadDim;
            const int row = row_start + row_rel;
            const int32_t tile_offset =
                block_token_start + row - direct_tile_start;
            sK(tile_offset, q_dim) = byte_v2_bf16_bits_to_cutlass_bfloat16(0);
            sV(tile_offset, q_dim) = byte_v2_bf16_bits_to_cutlass_bfloat16(0);
          }
          continue;
        }

        if constexpr (DirectDiagnosticMode ==
                      kByteV2DirectDiagnosticRawSameSkeleton) {
          static_assert(ByteV2DefaultRawStagingLayout::TilePolicy::HeadDim ==
                        Policy::HeadDim);
          static_assert(ByteV2DefaultRawStagingLayout::TilePolicy::HeadDimV ==
                        Policy::HeadDimV);
          static_assert(
              ByteV2DefaultRawStagingLayout::TilePolicy::AllocBlockTokens ==
              Policy::AllocBlockTokens);
          const uint8_t* raw_slot =
              kv_cache +
              static_cast<int64_t>(physical_block) * kv_cache_stride_block;
#pragma unroll 1
          for (int elem = threadIdx.x; elem < staged_rows * Policy::HeadDim;
               elem += NumThreads) {
            const int row_rel = elem / Policy::HeadDim;
            const int q_dim = elem - row_rel * Policy::HeadDim;
            const int row = row_start + row_rel;
            const int32_t tile_offset =
                block_token_start + row - direct_tile_start;
            const uint16_t k_bits = byte_v2_load_u16_bytes(
                raw_slot, ByteV2DefaultRawStagingLayout::key_offset(
                              kv_head_idx, row, q_dim));
            const uint16_t v_bits = byte_v2_load_u16_bytes(
                raw_slot, ByteV2DefaultRawStagingLayout::value_offset(
                              kv_head_idx, row, q_dim));
            sK(tile_offset, q_dim) =
                byte_v2_bf16_bits_to_cutlass_bfloat16(k_bits);
            sV(tile_offset, q_dim) =
                byte_v2_bf16_bits_to_cutlass_bfloat16(v_bits);
          }
          continue;
        }

        const uint8_t* page = kv_cache + static_cast<int64_t>(physical_block) *
                                             kv_cache_stride_block;
        int page_unsafe_flag = 0;
        if constexpr (UsePageUnsafeFlags) {
          page_unsafe_flag = page_unsafe_flags[physical_block];
        }
        const bool page_is_unsafe = page_unsafe_flag != 0;
        const bool page_unsafe_has_side_bits =
            (page_unsafe_flag & (kByteV2PageUnsafeK | kByteV2PageUnsafeV)) != 0;
        const bool k_page_is_unsafe =
            page_is_unsafe && (!page_unsafe_has_side_bits ||
                               (page_unsafe_flag & kByteV2PageUnsafeK) != 0);
        const bool v_page_is_unsafe =
            page_is_unsafe && (!page_unsafe_has_side_bits ||
                               (page_unsafe_flag & kByteV2PageUnsafeV) != 0);
        uint32_t k_fallback_mask = 0;
        uint32_t k_outlier_mask = 0;
        uint32_t v_fallback_mask = 0;
        uint32_t v_outlier_mask = 0;
        if (page_is_unsafe) {
          if constexpr (kDirectDiagNeedsCompressedK) {
            if (k_page_is_unsafe) {
              k_fallback_mask = byte_v2_load_u32_bytes(
                  page, Layout::k_fallback_mask_offset(kv_head_idx));
              k_outlier_mask = byte_v2_load_u32_bytes(
                  page, Layout::k_outlier_mask_offset(kv_head_idx));
            }
          }
          if constexpr (kDirectDiagNeedsCompressedV) {
            if (v_page_is_unsafe) {
              v_fallback_mask = byte_v2_load_u32_bytes(
                  page, Layout::v_fallback_mask_offset(kv_head_idx));
              v_outlier_mask = byte_v2_load_u32_bytes(
                  page, Layout::v_outlier_mask_offset(kv_head_idx));
            }
          }
        }
        if (page_is_unsafe) {
          constexpr int kDimHexes = Policy::HeadDim / 16;
          static_assert(Policy::HeadDim % 16 == 0);
          static_assert(Policy::CodecDimBlock == 16);
          static_assert((kDimHexes & (kDimHexes - 1)) == 0);
          static_assert(kDirectDiagStagingThreads % kDimHexes == 0);
          const int staging_thread = threadIdx.x - kDirectDiagStagingThreadBase;
          const bool staging_thread_in_range =
              staging_thread >= 0 && staging_thread < kDirectDiagStagingThreads;
          const int dim_hex =
              staging_thread_in_range
                  ? (kDirectSpecRowFastStaging
                         ? staging_thread / Policy::CodecTokenBlock
                         : (staging_thread & (kDimHexes - 1)))
                  : 0;
          const bool staging_thread_active =
              staging_thread_in_range && dim_hex < kDirectDiagStageWindowHexes;
          const int q_dim = dim_hex * 16;
          const int dim_tile = q_dim / Policy::CodecDimBlock;
          const int dim_in_tile = q_dim % Policy::CodecDimBlock;
          const auto k_desc0 =
              byte_v2_make_payload_tile_descriptor_from_masks<Layout, false>(
                  page, k_payload_base_offset, kv_head_idx, dim_tile,
                  k_fallback_mask, k_outlier_mask);
          const auto v_desc0 =
              byte_v2_make_payload_tile_descriptor_from_masks<Layout, true>(
                  page, v_payload_base_offset, kv_head_idx, dim_tile,
                  v_fallback_mask, v_outlier_mask);
#pragma unroll 1
          for (int row_rel =
                   kDirectSpecRowFastStaging
                       ? staging_thread & (Policy::CodecTokenBlock - 1)
                       : staging_thread / kDimHexes;
               staging_thread_active && row_rel < staged_rows;
               row_rel += kDirectSpecRowFastStaging
                              ? Policy::CodecTokenBlock
                              : kDirectDiagStagingThreads / kDimHexes) {
            const int row = row_start + row_rel;
            const int32_t tile_offset =
                block_token_start + row - direct_tile_start;
            uint16_t bits0;
            uint16_t bits1;
            uint16_t bits2;
            uint16_t bits3;
            uint16_t bits4;
            uint16_t bits5;
            uint16_t bits6;
            uint16_t bits7;
            uint16_t bits8;
            uint16_t bits9;
            uint16_t bits10;
            uint16_t bits11;
            uint16_t bits12;
            uint16_t bits13;
            uint16_t bits14;
            uint16_t bits15;
            if constexpr (kDirectDiagNeedsCompressedK) {
              if (k_desc0.fallback_hit) {
#pragma unroll
                for (int col = 0; col < 16; ++col) {
                  const int dim = q_dim + col;
                  const float k_value =
                      byte_v2_load_payload_elem_from_tile_descriptor<Layout,
                                                                     false>(
                          k_desc0, kv_head_idx, row, dim, dim_in_tile + col);
                  sK(tile_offset, dim) =
                      byte_v2_float_to_cutlass_bfloat16(k_value);
                }
              } else {
                if constexpr (Layout::CodecOutlierHighSideband) {
                  if (k_desc0.outlier_hit) {
                    byte_v2_load_payload_elem_hex_from_outlier_high_sideband_fixed_dim_bits<
                        Layout>(k_desc0, row, dim_in_tile, bits0, bits1, bits2,
                                bits3, bits4, bits5, bits6, bits7, bits8, bits9,
                                bits10, bits11, bits12, bits13, bits14, bits15);
                  } else {
                    byte_v2_load_payload_elem_hex_from_safe_tile_descriptor_fixed_dim_bits<
                        Layout, DirectDiagnosticMode !=
                                    kByteV2DirectDiagnosticCodeReady>(
                        k_desc0, row, dim_in_tile, bits0, bits1, bits2, bits3,
                        bits4, bits5, bits6, bits7, bits8, bits9, bits10,
                        bits11, bits12, bits13, bits14, bits15);
                  }
                } else {
                  byte_v2_load_payload_elem_hex_from_safe_tile_descriptor_fixed_dim_bits<
                      Layout,
                      DirectDiagnosticMode != kByteV2DirectDiagnosticCodeReady>(
                      k_desc0, row, dim_in_tile, bits0, bits1, bits2, bits3,
                      bits4, bits5, bits6, bits7, bits8, bits9, bits10, bits11,
                      bits12, bits13, bits14, bits15);
                  if (k_desc0.outlier_hit) {
                    byte_v2_patch_payload_elem_hex_outliers_from_tile_descriptor_bits<
                        Layout>(k_desc0, row, dim_in_tile, bits0, bits1, bits2,
                                bits3, bits4, bits5, bits6, bits7, bits8, bits9,
                                bits10, bits11, bits12, bits13, bits14, bits15);
                  }
                }
                byte_v2_store_16_bf16_bits_to_smem(
                    &shared_k_cute[0],
                    typename CuteDirectQkTraits::SmemLayoutKV{}, tile_offset,
                    q_dim, bits0, bits1, bits2, bits3, bits4, bits5, bits6,
                    bits7, bits8, bits9, bits10, bits11, bits12, bits13, bits14,
                    bits15);
              }
            }
            if constexpr (kDirectDiagNeedsCompressedV) {
              if (v_desc0.fallback_hit) {
#pragma unroll
                for (int col = 0; col < 16; ++col) {
                  const int dim = q_dim + col;
                  const float v_value =
                      byte_v2_load_payload_elem_from_tile_descriptor<Layout,
                                                                     true>(
                          v_desc0, kv_head_idx, row, dim, dim_in_tile + col);
                  sV(tile_offset, dim) =
                      byte_v2_float_to_cutlass_bfloat16(v_value);
                }
              } else {
                if constexpr (Layout::CodecOutlierHighSideband) {
                  if (v_desc0.outlier_hit) {
                    byte_v2_load_payload_elem_hex_from_outlier_high_sideband_fixed_dim_bits<
                        Layout>(v_desc0, row, dim_in_tile, bits0, bits1, bits2,
                                bits3, bits4, bits5, bits6, bits7, bits8, bits9,
                                bits10, bits11, bits12, bits13, bits14, bits15);
                  } else {
                    byte_v2_load_payload_elem_hex_from_safe_tile_descriptor_fixed_dim_bits<
                        Layout, DirectDiagnosticMode !=
                                    kByteV2DirectDiagnosticCodeReady>(
                        v_desc0, row, dim_in_tile, bits0, bits1, bits2, bits3,
                        bits4, bits5, bits6, bits7, bits8, bits9, bits10,
                        bits11, bits12, bits13, bits14, bits15);
                  }
                } else {
                  byte_v2_load_payload_elem_hex_from_safe_tile_descriptor_fixed_dim_bits<
                      Layout,
                      DirectDiagnosticMode != kByteV2DirectDiagnosticCodeReady>(
                      v_desc0, row, dim_in_tile, bits0, bits1, bits2, bits3,
                      bits4, bits5, bits6, bits7, bits8, bits9, bits10, bits11,
                      bits12, bits13, bits14, bits15);
                  if (v_desc0.outlier_hit) {
                    byte_v2_patch_payload_elem_hex_outliers_from_tile_descriptor_bits<
                        Layout>(v_desc0, row, dim_in_tile, bits0, bits1, bits2,
                                bits3, bits4, bits5, bits6, bits7, bits8, bits9,
                                bits10, bits11, bits12, bits13, bits14, bits15);
                  }
                }
                byte_v2_store_16_bf16_bits_to_smem(
                    &shared_v_full[0],
                    typename CuteDirectPvTraits::SmemLayoutKV{}, tile_offset,
                    q_dim, bits0, bits1, bits2, bits3, bits4, bits5, bits6,
                    bits7, bits8, bits9, bits10, bits11, bits12, bits13, bits14,
                    bits15);
              }
            }
          }
        } else {
          if constexpr (kDirectDiagStageWindow16Specialized) {
            static_assert(Policy::CodecDimBlock == 16);
            static_assert(Policy::HeadDim >= 16);
            if (threadIdx.x < staged_rows) {
              constexpr int q_dim = 0;
              constexpr int dim_tile = 0;
              constexpr int dim_in_tile = 0;
              const int row = row_start + threadIdx.x;
              const int32_t tile_offset =
                  block_token_start + row - direct_tile_start;
              const auto k_desc0 =
                  byte_v2_make_payload_tile_descriptor<Layout, false>(
                      page, k_payload_base_offset, kv_head_idx, dim_tile,
                      false);
              const auto v_desc0 =
                  byte_v2_make_payload_tile_descriptor<Layout, true>(
                      page, v_payload_base_offset, kv_head_idx, dim_tile,
                      false);
              uint16_t bits0;
              uint16_t bits1;
              uint16_t bits2;
              uint16_t bits3;
              uint16_t bits4;
              uint16_t bits5;
              uint16_t bits6;
              uint16_t bits7;
              uint16_t bits8;
              uint16_t bits9;
              uint16_t bits10;
              uint16_t bits11;
              uint16_t bits12;
              uint16_t bits13;
              uint16_t bits14;
              uint16_t bits15;
              byte_v2_load_payload_elem_hex_from_safe_tile_descriptor_fixed_dim_bits<
                  Layout>(k_desc0, row, dim_in_tile, bits0, bits1, bits2, bits3,
                          bits4, bits5, bits6, bits7, bits8, bits9, bits10,
                          bits11, bits12, bits13, bits14, bits15);
              byte_v2_store_16_bf16_bits_to_smem(
                  &shared_k_cute[0],
                  typename CuteDirectQkTraits::SmemLayoutKV{}, tile_offset,
                  q_dim, bits0, bits1, bits2, bits3, bits4, bits5, bits6, bits7,
                  bits8, bits9, bits10, bits11, bits12, bits13, bits14, bits15);
              byte_v2_load_payload_elem_hex_from_safe_tile_descriptor_fixed_dim_bits<
                  Layout>(v_desc0, row, dim_in_tile, bits0, bits1, bits2, bits3,
                          bits4, bits5, bits6, bits7, bits8, bits9, bits10,
                          bits11, bits12, bits13, bits14, bits15);
              byte_v2_store_16_bf16_bits_to_smem(
                  &shared_v_full[0],
                  typename CuteDirectPvTraits::SmemLayoutKV{}, tile_offset,
                  q_dim, bits0, bits1, bits2, bits3, bits4, bits5, bits6, bits7,
                  bits8, bits9, bits10, bits11, bits12, bits13, bits14, bits15);
            }
          } else {
            constexpr int kDimHexes = Policy::HeadDim / 16;
            static_assert(Policy::HeadDim % 16 == 0);
            static_assert((kDimHexes & (kDimHexes - 1)) == 0);
            static_assert(kDirectDiagStagingThreads % kDimHexes == 0);
            const int staging_thread =
                threadIdx.x - kDirectDiagStagingThreadBase;
            const bool staging_thread_in_range =
                staging_thread >= 0 &&
                staging_thread < kDirectDiagStagingThreads;
            const int dim_hex =
                staging_thread_in_range
                    ? (kDirectSpecRowFastStaging
                           ? staging_thread / Policy::CodecTokenBlock
                           : (staging_thread & (kDimHexes - 1)))
                    : 0;
            const bool staging_thread_active =
                staging_thread_in_range &&
                dim_hex < kDirectDiagStageWindowHexes;
            const int q_dim = dim_hex * 16;
            const int dim_tile = q_dim / Policy::CodecDimBlock;
            const int dim_in_tile = q_dim % Policy::CodecDimBlock;
            const auto k_desc0 =
                byte_v2_make_payload_tile_descriptor<Layout, false>(
                    page, k_payload_base_offset, kv_head_idx, dim_tile, false);
            const auto v_desc0 =
                byte_v2_make_payload_tile_descriptor<Layout, true>(
                    page, v_payload_base_offset, kv_head_idx, dim_tile, false);
#pragma unroll 1
            for (int row_rel =
                     kDirectSpecRowFastStaging
                         ? staging_thread & (Policy::CodecTokenBlock - 1)
                         : staging_thread / kDimHexes;
                 staging_thread_active && row_rel < staged_rows;
                 row_rel += kDirectSpecRowFastStaging
                                ? Policy::CodecTokenBlock
                                : kDirectDiagStagingThreads / kDimHexes) {
              const int row = row_start + row_rel;
              const int32_t tile_offset =
                  block_token_start + row - direct_tile_start;
              uint16_t bits0;
              uint16_t bits1;
              uint16_t bits2;
              uint16_t bits3;
              uint16_t bits4;
              uint16_t bits5;
              uint16_t bits6;
              uint16_t bits7;
              uint16_t bits8;
              uint16_t bits9;
              uint16_t bits10;
              uint16_t bits11;
              uint16_t bits12;
              uint16_t bits13;
              uint16_t bits14;
              uint16_t bits15;
              if constexpr (kDirectDiagNeedsCompressedK) {
                if constexpr (kDirectDiagDecodeLowOnly) {
                  byte_v2_load_payload_elem_hex_low_only_from_safe_tile_descriptor_fixed_dim_bits<
                      Layout>(k_desc0, row, dim_in_tile, bits0, bits1, bits2,
                              bits3, bits4, bits5, bits6, bits7, bits8, bits9,
                              bits10, bits11, bits12, bits13, bits14, bits15);
                } else if constexpr (kDirectDiagDecodeHighOnly) {
                  byte_v2_load_payload_elem_hex_high_only_from_safe_tile_descriptor_fixed_dim_bits<
                      Layout>(k_desc0, row, dim_in_tile, bits0, bits1, bits2,
                              bits3, bits4, bits5, bits6, bits7, bits8, bits9,
                              bits10, bits11, bits12, bits13, bits14, bits15);
                } else {
                  byte_v2_load_payload_elem_hex_from_safe_tile_descriptor_fixed_dim_bits<
                      Layout,
                      DirectDiagnosticMode != kByteV2DirectDiagnosticCodeReady>(
                      k_desc0, row, dim_in_tile, bits0, bits1, bits2, bits3,
                      bits4, bits5, bits6, bits7, bits8, bits9, bits10, bits11,
                      bits12, bits13, bits14, bits15);
                }
                if constexpr (kDirectDiagStoreDecodedK) {
                  byte_v2_store_16_bf16_bits_to_smem(
                      &shared_k_cute[0],
                      typename CuteDirectQkTraits::SmemLayoutKV{}, tile_offset,
                      q_dim, bits0, bits1, bits2, bits3, bits4, bits5, bits6,
                      bits7, bits8, bits9, bits10, bits11, bits12, bits13,
                      bits14, bits15);
                } else if constexpr (!kDirectDiagStoreDecodedK) {
                  byte_v2_accumulate_16_bf16_bits_for_diagnostic(
                      direct_diag_decode_sink, bits0, bits1, bits2, bits3,
                      bits4, bits5, bits6, bits7, bits8, bits9, bits10, bits11,
                      bits12, bits13, bits14, bits15);
                }
              }
              if constexpr (kDirectDiagNeedsCompressedV) {
                if constexpr (kDirectDiagDecodeLowOnly) {
                  byte_v2_load_payload_elem_hex_low_only_from_safe_tile_descriptor_fixed_dim_bits<
                      Layout>(v_desc0, row, dim_in_tile, bits0, bits1, bits2,
                              bits3, bits4, bits5, bits6, bits7, bits8, bits9,
                              bits10, bits11, bits12, bits13, bits14, bits15);
                } else if constexpr (kDirectDiagDecodeHighOnly) {
                  byte_v2_load_payload_elem_hex_high_only_from_safe_tile_descriptor_fixed_dim_bits<
                      Layout>(v_desc0, row, dim_in_tile, bits0, bits1, bits2,
                              bits3, bits4, bits5, bits6, bits7, bits8, bits9,
                              bits10, bits11, bits12, bits13, bits14, bits15);
                } else {
                  byte_v2_load_payload_elem_hex_from_safe_tile_descriptor_fixed_dim_bits<
                      Layout,
                      DirectDiagnosticMode != kByteV2DirectDiagnosticCodeReady>(
                      v_desc0, row, dim_in_tile, bits0, bits1, bits2, bits3,
                      bits4, bits5, bits6, bits7, bits8, bits9, bits10, bits11,
                      bits12, bits13, bits14, bits15);
                }
                if constexpr (kDirectDiagStoreDecodedV) {
                  byte_v2_store_16_bf16_bits_to_smem(
                      &shared_v_full[0],
                      typename CuteDirectPvTraits::SmemLayoutKV{}, tile_offset,
                      q_dim, bits0, bits1, bits2, bits3, bits4, bits5, bits6,
                      bits7, bits8, bits9, bits10, bits11, bits12, bits13,
                      bits14, bits15);
                } else if constexpr (!kDirectDiagStoreDecodedV) {
                  byte_v2_accumulate_16_bf16_bits_for_diagnostic(
                      direct_diag_decode_sink, bits0, bits1, bits2, bits3,
                      bits4, bits5, bits6, bits7, bits8, bits9, bits10, bits11,
                      bits12, bits13, bits14, bits15);
                }
              }
            }
          }
        }
      }
      if constexpr (kDirectDiagClockProfile) {
        if (direct_phase_profile_block && threadIdx.x == 0) {
          direct_phase_stage_done = clock64();
        }
      }
      uint64_t direct_phase_stage_wait_start = 0;
      if constexpr (kDirectDiagClockProfile) {
        if (direct_phase_profile_block && lane == 0) {
          direct_phase_stage_wait_start = clock64();
        }
      }
      if (direct_q_async && direct_tile_start == partition_start) {
        vllm::cuda_async::cp_async_wait_group<0>();
      }
      __syncthreads();
      if constexpr (kDirectDiagClockProfile) {
        if (direct_phase_profile_block && lane == 0) {
          const uint64_t direct_phase_stage_wait_done = clock64();
          direct_phase_stage_wait_cycles +=
              direct_phase_stage_wait_done - direct_phase_stage_wait_start;
          if (threadIdx.x == 0) {
            direct_phase_stage_cycles +=
                direct_phase_stage_done - direct_phase_stage_start;
          }
        }
      }

      if constexpr (kDirectDiagStageOnly) {
        continue;
      }

      uint64_t direct_phase_compute_region_start = 0;
      if constexpr (kDirectDiagPhaseProfile) {
        if (direct_phase_profile_block && lane == 0) {
          direct_phase_compute_region_start = clock64();
        }
      }
      if (warp < kDirectComputeWarps) {
        auto acc_s = cute::partition_fragment_C(
            tiled_mma, cute::Shape<cute::Int<kDirectBlockM>,
                                   cute::Int<Policy::ComputeBlockN>>{});
        cute::clear(acc_s);

        auto cS = cute::make_identity_tensor(
            cute::Shape<cute::Int<kDirectBlockM>,
                        cute::Int<Policy::ComputeBlockN>>{});
        auto tCcS = thr_mma.partition_C(cS);
        auto scores = cute::make_tensor(
            acc_s.data(), byte_v2_convert_layout_acc_rowcol(acc_s.layout()));
        auto tCcS_rowcol = cute::make_tensor(
            tCcS.data(), byte_v2_convert_layout_acc_rowcol(tCcS.layout()));
        static_assert(decltype(cute::size<0>(scores))::value ==
                      kDirectThreadRows);
        float row_l[kDirectThreadRows];
        float new_m[kDirectThreadRows];
        float alpha[kDirectThreadRows];

        if constexpr (kDirectDiagUseFakePvProbs) {
#pragma unroll
          for (int row = 0; row < kDirectThreadRows; ++row) {
            row_l[row] = 0.0f;
            new_m[row] = 0.0f;
            alpha[row] = direct_softmax_l[row] > 0.0f ? 1.0f : 0.0f;
          }
#pragma unroll
          for (int row = 0; row < kDirectThreadRows; ++row) {
            const int logical_row = cute::get<0>(tCcS_rowcol(row, 0));
            int32_t row_seq_len = seq_len;
            if constexpr (SpeculativeQueryLen > 0) {
              const int virtual_row =
                  q_group_base + direct_row_base + logical_row;
              row_seq_len = seq_len - (request_query_len - 1) + virtual_row / 4;
            }
#pragma unroll
            for (int col = 0; col < cute::size<1>(scores); ++col) {
              const int logical_col = cute::get<1>(tCcS_rowcol(row, col));
              const bool token_valid =
                  logical_col < direct_tile_len &&
                  direct_tile_start + logical_col < row_seq_len;
              const float prob =
                  logical_row < direct_valid_q_rows && token_valid ? 1.0f
                                                                   : 0.0f;
              scores(row, col) = prob;
              row_l[row] += prob;
            }
          }
#pragma unroll
          for (int row = 0; row < kDirectThreadRows; ++row) {
            row_l[row] = byte_v2_quad_sum(row_l[row]);
          }
        } else {
          uint64_t direct_phase_qk_start = 0;
          uint64_t direct_phase_softmax_start = 0;
          if constexpr (kDirectDiagPhaseProfile) {
            if (direct_phase_profile_block && threadIdx.x == 0) {
              direct_phase_qk_start = clock64();
            }
          }
          auto tCrQ = thr_mma.partition_fragment_A(sQ);
          auto tCrK = thr_mma.partition_fragment_B(sK);
          auto smem_tiled_copy_Q = cute::make_tiled_copy_A(
              typename CuteDirectQkTraits::SmemCopyAtom{}, tiled_mma);
          auto smem_tiled_copy_K = cute::make_tiled_copy_B(
              typename CuteDirectQkTraits::SmemCopyAtom{}, tiled_mma);
          auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(lane);
          auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(lane);
          auto tCsQ = smem_thr_copy_Q.partition_S(sQ);
          auto tCsK = smem_thr_copy_K.partition_S(sK);
          byte_v2_cute_gemm_smem(acc_s, tCrQ, tCrK, tCsQ, tCsK, tiled_mma,
                                 smem_tiled_copy_Q, smem_tiled_copy_K,
                                 smem_thr_copy_Q, smem_thr_copy_K);
          if constexpr (kDirectDiagPhaseProfile) {
            if (direct_phase_profile_block && threadIdx.x == 0) {
              const uint64_t direct_phase_qk_done = clock64();
              direct_phase_qk_cycles +=
                  direct_phase_qk_done - direct_phase_qk_start;
              direct_phase_softmax_start = direct_phase_qk_done;
            }
          }

          float row_m[kDirectThreadRows];
#pragma unroll
          for (int row = 0; row < kDirectThreadRows; ++row) {
            const int logical_row = cute::get<0>(tCcS_rowcol(row, 0));
            int32_t row_seq_len = seq_len;
            if constexpr (SpeculativeQueryLen > 0) {
              const int virtual_row =
                  q_group_base + direct_row_base + logical_row;
              row_seq_len = seq_len - (request_query_len - 1) + virtual_row / 4;
            }
            row_m[row] = -FLT_MAX;
#pragma unroll
            for (int col = 0; col < cute::size<1>(scores); ++col) {
              const int logical_col = cute::get<1>(tCcS_rowcol(row, col));
              const bool token_valid =
                  logical_col < direct_tile_len &&
                  direct_tile_start + logical_col < row_seq_len;
              const float score =
                  logical_row < direct_valid_q_rows && token_valid
                      ? scores(row, col) * scale
                      : -FLT_MAX;
              scores(row, col) = score;
              row_m[row] = fmaxf(row_m[row], score);
            }
          }
#pragma unroll
          for (int row = 0; row < kDirectThreadRows; ++row) {
            row_m[row] = byte_v2_quad_max(row_m[row]);
          }
#pragma unroll
          for (int row = 0; row < kDirectThreadRows; ++row) {
            row_l[row] = 0.0f;
            new_m[row] = fmaxf(direct_softmax_m[row], row_m[row]);
            alpha[row] = direct_softmax_l[row] > 0.0f
                             ? __expf(direct_softmax_m[row] - new_m[row])
                             : 0.0f;
          }
#pragma unroll
          for (int row = 0; row < kDirectThreadRows; ++row) {
#pragma unroll
            for (int col = 0; col < cute::size<1>(scores); ++col) {
              const float score = scores(row, col);
              const float prob = score != -FLT_MAX && new_m[row] != -FLT_MAX
                                     ? __expf(score - new_m[row])
                                     : 0.0f;
              scores(row, col) = prob;
              row_l[row] += prob;
            }
          }
#pragma unroll
          for (int row = 0; row < kDirectThreadRows; ++row) {
            row_l[row] = byte_v2_quad_sum(row_l[row]);
          }
          if constexpr (kDirectDiagPhaseProfile) {
            if (direct_phase_profile_block && threadIdx.x == 0) {
              direct_phase_softmax_cycles +=
                  clock64() - direct_phase_softmax_start;
            }
          }
        }

        if constexpr (!kDirectDiagSkipPv) {
          uint64_t direct_phase_pv_start = 0;
          if constexpr (kDirectDiagPhaseProfile) {
            if (direct_phase_profile_block && threadIdx.x == 0) {
              direct_phase_pv_start = clock64();
            }
          }
          if constexpr (kDirectDiagSkipPvGemm) {
#pragma unroll
            for (int row = 0; row < kDirectThreadRows; ++row) {
              const int logical_row = cute::get<0>(tCcO_rowcol(row, 0));
#pragma unroll
              for (int col = 0; col < cute::size<1>(acc_o_rowcol); ++col) {
                acc_o_rowcol(row, col) =
                    logical_row < direct_valid_q_rows
                        ? acc_o_rowcol(row, col) * alpha[row]
                        : 0.0f;
              }
            }
          } else {
            auto rP =
                cute::make_tensor_like<typename CuteDirectPvTraits::Element>(
                    acc_s);
#pragma unroll
            for (int elem = 0; elem < cute::size(rP); ++elem) {
              rP(elem) = byte_v2_float_to_cutlass_bfloat16(acc_s(elem));
            }
            auto tCrP = cute::make_tensor(
                rP.data(),
                byte_v2_convert_layout_acc_Aregs<
                    typename CuteDirectPvTraits::TiledMma>(rP.layout()));

            auto sVt = cute::make_tensor(
                sV.data(),
                typename CuteDirectPvTraits::SmemLayoutVtransposed{});
            auto sVtNoSwizzle = cute::make_tensor(
                sV.data().get(),
                typename CuteDirectPvTraits::SmemLayoutVtransposedNoSwizzle{});
            auto tCrV = thr_mma.partition_fragment_B(sVtNoSwizzle);
            auto acc_o_tile = cute::partition_fragment_C(
                tiled_mma, cute::Shape<cute::Int<kDirectBlockM>,
                                       cute::Int<Policy::HeadDimV>>{});
            cute::clear(acc_o_tile);
            auto acc_o_tile_rowcol = cute::make_tensor(
                acc_o_tile.data(),
                byte_v2_convert_layout_acc_rowcol(acc_o_tile.layout()));
            auto smem_tiled_copy_V = cute::make_tiled_copy_B(
                typename CuteDirectPvTraits::SmemCopyAtomV{}, tiled_mma);
            auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(lane);
            auto tCsV = smem_thr_copy_V.partition_S(sVt);
            byte_v2_cute_gemm_rs(acc_o_tile, tCrP, tCrV, tCsV, tiled_mma,
                                 smem_tiled_copy_V, smem_thr_copy_V);

            if constexpr (kDirectDiagSkipPvAccum) {
              float pv_diag_sum = 0.0f;
#pragma unroll
              for (int elem = 0; elem < cute::size(acc_o_tile); ++elem) {
                pv_diag_sum += acc_o_tile(elem);
              }
              acc_o(0) += pv_diag_sum;
            } else {
#pragma unroll
              for (int row = 0; row < kDirectThreadRows; ++row) {
                const int logical_row = cute::get<0>(tCcO_rowcol(row, 0));
#pragma unroll
                for (int col = 0; col < cute::size<1>(acc_o_rowcol); ++col) {
                  acc_o_rowcol(row, col) =
                      logical_row < direct_valid_q_rows
                          ? acc_o_rowcol(row, col) * alpha[row] +
                                acc_o_tile_rowcol(row, col)
                          : 0.0f;
                }
              }
            }
          }
          if constexpr (kDirectDiagPhaseProfile) {
            if (direct_phase_profile_block && threadIdx.x == 0) {
              direct_phase_pv_cycles += clock64() - direct_phase_pv_start;
            }
          }
        }
#pragma unroll
        for (int row = 0; row < kDirectThreadRows; ++row) {
          direct_softmax_l[row] =
              direct_softmax_l[row] * alpha[row] + row_l[row];
          direct_softmax_m[row] = new_m[row];
        }
      }
      uint64_t direct_phase_compute_wait_start = 0;
      if constexpr (kDirectDiagPhaseProfile) {
        if (direct_phase_profile_block && lane == 0) {
          direct_phase_compute_wait_start = clock64();
        }
      }
      __syncthreads();
      if constexpr (kDirectDiagPhaseProfile) {
        if (direct_phase_profile_block && lane == 0) {
          const uint64_t direct_phase_compute_wait_done = clock64();
          direct_phase_compute_wait_cycles +=
              direct_phase_compute_wait_done - direct_phase_compute_wait_start;
          direct_phase_compute_region_cycles +=
              direct_phase_compute_wait_done -
              direct_phase_compute_region_start;
        }
      }
    }

    if constexpr (kDirectDiagStageOnly) {
      if (threadIdx.x == 0) {
#pragma unroll
        for (int q_group = 0; q_group < kQPerKv; ++q_group) {
          if (q_group < valid_q_rows) {
            exp_sums[stats_base +
                     static_cast<int64_t>(q_group) * max_num_partitions] = 0.0f;
          }
        }
      }
#pragma unroll
      for (int q_group = 0; q_group < kQPerKv; ++q_group) {
        if (q_group < valid_q_rows) {
          tmp_out[tmp_base +
                  static_cast<int64_t>(q_group) * max_num_partitions *
                      Policy::HeadDimV +
                  dim] =
              q_group == 0 ? static_cast<float>(direct_diag_decode_sink) : 0.0f;
        }
      }
      if constexpr (kDirectDiagClockProfile) {
        if (direct_phase_profile_block && lane == 0) {
          if (threadIdx.x == 0) {
            max_logits[0] =
                static_cast<float>(clock64() - direct_phase_total_start);
            max_logits[1] = static_cast<float>(direct_phase_stage_cycles);
            max_logits[2] = static_cast<float>(direct_phase_qk_cycles);
            max_logits[3] = static_cast<float>(direct_phase_softmax_cycles);
            max_logits[4] = static_cast<float>(direct_phase_pv_cycles);
            max_logits[13] = static_cast<float>(direct_phase_tile_count);
            max_logits[14] = static_cast<float>(Policy::ComputeBlockN);
            max_logits[15] = static_cast<float>(partition_size);
          }
          max_logits[5 + warp] =
              static_cast<float>(direct_phase_stage_wait_cycles);
          max_logits[9 + warp] =
              static_cast<float>(direct_phase_compute_wait_cycles);
          max_logits[16 + warp] =
              static_cast<float>(direct_phase_compute_region_cycles);
        }
      }
      return;
    }

    if constexpr (kDirectDiagSkipFinalOutput) {
      if (warp < kDirectComputeWarps && (lane & 3) == 0) {
#pragma unroll
        for (int row = 0; row < kDirectThreadRows; ++row) {
          const int logical_row = cute::get<0>(tCcO_rowcol(row, 0));
          if (logical_row < direct_valid_q_rows) {
            const int global_q_group = direct_row_base + logical_row;
            exp_sums[stats_base + static_cast<int64_t>(global_q_group) *
                                      max_num_partitions] =
                direct_softmax_m[row] + direct_softmax_l[row];
          }
        }
      }
      return;
    }

    if (warp < kDirectComputeWarps) {
#pragma unroll
      for (int row = 0; row < kDirectThreadRows; ++row) {
        const int logical_row = cute::get<0>(tCcO_rowcol(row, 0));
        const int global_q_group = direct_row_base + logical_row;
        const float denom = direct_softmax_l[row];
#pragma unroll
        for (int col = 0; col < cute::size<1>(acc_o_rowcol); ++col) {
          const int logical_col = cute::get<1>(tCcO_rowcol(row, col));
          if (logical_row < direct_valid_q_rows &&
              logical_col < Policy::HeadDimV) {
            float out_value;
            if constexpr (kDirectDiagUnnormalizedPartitionOutput) {
              out_value = acc_o_rowcol(row, col);
            } else {
              out_value = denom > 0.0f ? acc_o_rowcol(row, col) / denom : 0.0f;
            }
            const int64_t out_offset = tmp_base +
                                       static_cast<int64_t>(global_q_group) *
                                           max_num_partitions *
                                           Policy::HeadDimV +
                                       logical_col;
            tmp_out[out_offset] = out_value;
          }
        }
      }

      if ((lane & 3) == 0) {
#pragma unroll
        for (int row = 0; row < kDirectThreadRows; ++row) {
          const int logical_row = cute::get<0>(tCcO_rowcol(row, 0));
          if (logical_row < direct_valid_q_rows) {
            const int global_q_group = direct_row_base + logical_row;
            const int64_t stats_offset =
                stats_base +
                static_cast<int64_t>(global_q_group) * max_num_partitions;
            if constexpr (kDirectDiagUnnormalizedPartitionOutput) {
              exp_sums[stats_offset] = direct_softmax_l[row];
              max_logits[stats_offset] = direct_softmax_m[row];
            } else {
              exp_sums[stats_offset] =
                  direct_softmax_l[row] > 0.0f
                      ? logf(direct_softmax_l[row]) + direct_softmax_m[row]
                      : -FLT_MAX;
            }
          }
        }
      }
    }
    if constexpr (kDirectDiagClockProfile) {
      if (direct_phase_profile_block && lane == 0) {
        if (threadIdx.x == 0) {
          max_logits[0] =
              static_cast<float>(clock64() - direct_phase_total_start);
          max_logits[1] = static_cast<float>(direct_phase_stage_cycles);
          max_logits[2] = static_cast<float>(direct_phase_qk_cycles);
          max_logits[3] = static_cast<float>(direct_phase_softmax_cycles);
          max_logits[4] = static_cast<float>(direct_phase_pv_cycles);
          max_logits[13] = static_cast<float>(direct_phase_tile_count);
          max_logits[14] = static_cast<float>(Policy::ComputeBlockN);
          max_logits[15] = static_cast<float>(partition_size);
        }
        max_logits[5 + warp] =
            static_cast<float>(direct_phase_stage_wait_cycles);
        max_logits[9 + warp] =
            static_cast<float>(direct_phase_compute_wait_cycles);
        max_logits[16 + warp] =
            static_cast<float>(direct_phase_compute_region_cycles);
      }
    }
    return;
  }

  if constexpr (UseFa2Multiwarp) {
    __shared__ int shared_token_valid[Policy::ComputeBlockN];
    __shared__ __align__(16) typename CuteQkTraits::Element
        shared_q_cute[CuteQkTraits::kQStorageElems];
    __shared__ __align__(16) typename CuteQkTraits::Element
        shared_k_cute[CuteQkTraits::kKStorageElems];

    auto sQ = cute::make_tensor(cute::make_smem_ptr(&shared_q_cute[0]),
                                typename CuteQkTraits::SmemLayoutQ{});
    auto sK = cute::make_tensor(cute::make_smem_ptr(&shared_k_cute[0]),
                                typename CuteQkTraits::SmemLayoutKV{});

#pragma unroll 1
    for (int elem = threadIdx.x; elem < 16 * Policy::HeadDim;
         elem += NumThreads) {
      const int row = elem / Policy::HeadDim;
      const int col = elem - row * Policy::HeadDim;
      uint16_t bits = 0;
      if (row < kQPerKv) {
        const int head_idx = kv_head_idx * kQPerKv + row;
        const int64_t q_offset =
            static_cast<int64_t>(seq_idx) * q_stride_token +
            static_cast<int64_t>(head_idx) * q_stride_head + col * q_stride_dim;
        bits = query[q_offset];
      }
      sQ(row, col) = byte_v2_bf16_bits_to_cutlass_bfloat16(bits);
    }

#pragma unroll 1
    for (int elem = threadIdx.x; elem < Policy::ComputeBlockN * Policy::HeadDim;
         elem += NumThreads) {
      const int row = elem / Policy::HeadDim;
      const int col = elem - row * Policy::HeadDim;
      sK(row, col) = byte_v2_bf16_bits_to_cutlass_bfloat16(0);
    }
    for (int tile_offset = threadIdx.x; tile_offset < Policy::ComputeBlockN;
         tile_offset += NumThreads) {
      shared_token_valid[tile_offset] = 0;
    }
    __syncthreads();

#pragma unroll 1
    for (int32_t logical_block = first_logical_block;
         logical_block < last_logical_block; ++logical_block) {
      const int32_t block_token_start = logical_block * kBlockSize;
      const int row_start = max(tile_start - block_token_start, 0);
      const int row_end = min(tile_end - block_token_start, kBlockSize);
      if (logical_block >= max_num_blocks_per_seq) {
        continue;
      }
      const int32_t physical_block = block_table[logical_block];
      if (physical_block < 0) {
        continue;
      }

      const uint8_t* page = kv_cache + static_cast<int64_t>(physical_block) *
                                           kv_cache_stride_block;
      bool page_is_unsafe = false;
      if constexpr (UsePageUnsafeFlags) {
        page_is_unsafe = page_unsafe_flags[physical_block] != 0;
      }
      const int staged_rows = row_end - row_start;
#pragma unroll 1
      for (int row_rel = threadIdx.x; row_rel < staged_rows;
           row_rel += NumThreads) {
        const int row = row_start + row_rel;
        const int32_t tile_offset = block_token_start + row - tile_start;
        shared_token_valid[tile_offset] = 1;
      }

      if (page_is_unsafe) {
#pragma unroll 1
        for (int elem = threadIdx.x; elem < staged_rows * Policy::HeadDim;
             elem += NumThreads) {
          const int row_rel = elem / Policy::HeadDim;
          const int q_dim = elem - row_rel * Policy::HeadDim;
          const int row = row_start + row_rel;
          const int32_t tile_offset = block_token_start + row - tile_start;
          const float k_value = byte_v2_load_payload_elem<Layout, false>(
              page, k_payload_base_offset, kv_head_idx, row, q_dim);
          sK(tile_offset, q_dim) = byte_v2_float_to_cutlass_bfloat16(k_value);
        }
      } else {
#pragma unroll 1
        for (int elem = threadIdx.x; elem < staged_rows * Policy::HeadDim;
             elem += NumThreads) {
          const int row_rel = elem / Policy::HeadDim;
          const int q_dim = elem - row_rel * Policy::HeadDim;
          const int row = row_start + row_rel;
          const int32_t tile_offset = block_token_start + row - tile_start;
          const uint16_t bits =
              byte_v2_load_payload_elem_no_fallback_no_outlier_bits<Layout,
                                                                    false>(
                  page, k_payload_base_offset, kv_head_idx, row, q_dim);
          sK(tile_offset, q_dim) = byte_v2_bf16_bits_to_cutlass_bfloat16(bits);
        }
      }
    }
    __syncthreads();

    typename CuteQkTraits::TiledMma qk_mma;
    auto qk_thr_mma = qk_mma.get_thread_slice(lane);
    auto tCrQ = qk_thr_mma.partition_fragment_A(sQ);
    auto tCrK = qk_thr_mma.partition_fragment_B(sK);
    auto acc_s = cute::partition_fragment_C(
        qk_mma, cute::Shape<cute::Int<16>, cute::Int<Policy::ComputeBlockN>>{});
    cute::clear(acc_s);
    auto smem_tiled_copy_Q =
        cute::make_tiled_copy_A(typename CuteQkTraits::SmemCopyAtom{}, qk_mma);
    auto smem_tiled_copy_K =
        cute::make_tiled_copy_B(typename CuteQkTraits::SmemCopyAtom{}, qk_mma);
    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(lane);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(lane);
    auto tCsQ = smem_thr_copy_Q.partition_S(sQ);
    auto tCsK = smem_thr_copy_K.partition_S(sK);
    byte_v2_cute_gemm_smem(acc_s, tCrQ, tCrK, tCsQ, tCsK, qk_mma,
                           smem_tiled_copy_Q, smem_tiled_copy_K,
                           smem_thr_copy_Q, smem_thr_copy_K);

    auto cS = cute::make_identity_tensor(
        cute::Shape<cute::Int<16>, cute::Int<Policy::ComputeBlockN>>{});
    auto tCcS = qk_thr_mma.partition_C(cS);
    float row_m[kQPerKv];
#pragma unroll
    for (int q_group = 0; q_group < kQPerKv; ++q_group) {
      row_m[q_group] = -FLT_MAX;
    }
#pragma unroll
    for (int elem = 0; elem < cute::size(acc_s); ++elem) {
      const int row = cute::get<0>(tCcS(elem));
      const int col = cute::get<1>(tCcS(elem));
      if (row < kQPerKv) {
        const float score = col < tile_len && shared_token_valid[col] != 0
                                ? acc_s(elem) * scale
                                : -FLT_MAX;
        row_m[row] = fmaxf(row_m[row], score);
      }
    }
#pragma unroll
    for (int q_group = 0; q_group < kQPerKv; ++q_group) {
#pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1) {
        row_m[q_group] =
            fmaxf(row_m[q_group],
                  __shfl_down_sync(0xffffffff, row_m[q_group], offset));
      }
      row_m[q_group] = __shfl_sync(0xffffffff, row_m[q_group], 0);
    }

    float row_l[kQPerKv];
#pragma unroll
    for (int q_group = 0; q_group < kQPerKv; ++q_group) {
      row_l[q_group] = 0.0f;
    }
#pragma unroll
    for (int elem = 0; elem < cute::size(acc_s); ++elem) {
      const int row = cute::get<0>(tCcS(elem));
      const int col = cute::get<1>(tCcS(elem));
      float prob = 0.0f;
      if (row < kQPerKv && col < tile_len && shared_token_valid[col] != 0) {
        prob = __expf(acc_s(elem) * scale - row_m[row]);
        row_l[row] += prob;
      }
      acc_s(elem) = prob;
    }
#pragma unroll
    for (int q_group = 0; q_group < kQPerKv; ++q_group) {
#pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1) {
        row_l[q_group] += __shfl_down_sync(0xffffffff, row_l[q_group], offset);
      }
      row_l[q_group] = __shfl_sync(0xffffffff, row_l[q_group], 0);
    }

    auto rP = cute::make_tensor_like<typename CutePvTileTraits::Element>(acc_s);
#pragma unroll
    for (int elem = 0; elem < cute::size(rP); ++elem) {
      rP(elem) = byte_v2_float_to_cutlass_bfloat16(acc_s(elem));
    }

#pragma unroll
    for (int n_iter = 0; n_iter < 2; ++n_iter) {
      const int tile_n = n_iter * kQPerKv + warp;
      const int n_base = tile_n * 16;
      auto sV = cute::make_tensor(
          cute::make_smem_ptr(
              reinterpret_cast<typename CutePvTileTraits::Element*>(
                  &shared_v_cute[warp][0])),
          typename CutePvTileTraits::SmemLayoutV{});
#pragma unroll 1
      for (int elem = lane; elem < Policy::ComputeBlockN * 16; elem += 32) {
        const int row = elem / 16;
        const int col = elem - row * 16;
        sV(row, col) = byte_v2_bf16_bits_to_cutlass_bfloat16(0);
      }
      __syncwarp();

#pragma unroll 1
      for (int32_t logical_block = first_logical_block;
           logical_block < last_logical_block; ++logical_block) {
        if (logical_block >= max_num_blocks_per_seq) {
          continue;
        }
        const int32_t physical_block = block_table[logical_block];
        if (physical_block < 0) {
          continue;
        }

        const int32_t block_token_start = logical_block * kBlockSize;
        const int row_start = max(tile_start - block_token_start, 0);
        const int row_end = min(tile_end - block_token_start, kBlockSize);
        const int staged_rows = row_end - row_start;
        const uint8_t* page = kv_cache + static_cast<int64_t>(physical_block) *
                                             kv_cache_stride_block;
        bool page_is_unsafe = false;
        if constexpr (UsePageUnsafeFlags) {
          page_is_unsafe = page_unsafe_flags[physical_block] != 0;
        }

        if (page_is_unsafe) {
#pragma unroll 1
          for (int elem = lane; elem < staged_rows * 16; elem += 32) {
            const int row_rel = elem / 16;
            const int col = elem - row_rel * 16;
            const int row = row_start + row_rel;
            const int32_t tile_offset = block_token_start + row - tile_start;
            const int v_dim = n_base + col;
            const float v_value = byte_v2_load_payload_elem<Layout, true>(
                page, v_payload_base_offset, kv_head_idx, row, v_dim);
            sV(tile_offset, col) = byte_v2_float_to_cutlass_bfloat16(v_value);
          }
        } else {
#pragma unroll 1
          for (int elem = lane; elem < staged_rows * 16; elem += 32) {
            const int row_rel = elem / 16;
            const int col = elem - row_rel * 16;
            const int row = row_start + row_rel;
            const int32_t tile_offset = block_token_start + row - tile_start;
            const int v_dim = n_base + col;
            const uint16_t bits =
                byte_v2_load_payload_elem_no_fallback_no_outlier_bits<Layout,
                                                                      true>(
                    page, v_payload_base_offset, kv_head_idx, row, v_dim);
            sV(tile_offset, col) = byte_v2_bf16_bits_to_cutlass_bfloat16(bits);
          }
        }
      }
      __syncwarp();

      typename CutePvTileTraits::TiledMma pv_mma;
      auto pv_thr_mma = pv_mma.get_thread_slice(lane);
      auto tCrP = cute::make_tensor(
          rP.data(),
          byte_v2_convert_layout_acc_Aregs<typename CutePvTileTraits::TiledMma>(
              rP.layout()));
      auto sVt = cute::make_tensor(sV.data(),
                                   typename CutePvTileTraits::SmemLayoutVt{});
      auto sVtNoSwizzle = cute::make_tensor(
          sV.data().get(), typename CutePvTileTraits::SmemLayoutVtNoSwizzle{});
      auto tCrV = pv_thr_mma.partition_fragment_B(sVtNoSwizzle);
      auto acc_o = cute::partition_fragment_C(
          pv_mma, cute::Shape<cute::Int<16>, cute::Int<16>>{});
      cute::clear(acc_o);
      auto smem_tiled_copy_V = cute::make_tiled_copy_B(
          typename CutePvTileTraits::SmemCopyAtomV{}, pv_mma);
      auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(lane);
      auto tCsV = smem_thr_copy_V.partition_S(sVt);
      byte_v2_cute_gemm_rs(acc_o, tCrP, tCrV, tCsV, pv_mma, smem_tiled_copy_V,
                           smem_thr_copy_V);

      auto cO = cute::make_identity_tensor(
          cute::Shape<cute::Int<16>, cute::Int<16>>{});
      auto tCcO = pv_thr_mma.partition_C(cO);
#pragma unroll
      for (int elem = 0; elem < cute::size(acc_o); ++elem) {
        const int q_group = cute::get<0>(tCcO(elem));
        const int col = cute::get<1>(tCcO(elem));
        if (q_group < kQPerKv && col < 16) {
          const float denom = row_l[q_group];
          const float out_value = denom > 0.0f ? acc_o(elem) / denom : 0.0f;
          const int64_t out_offset = tmp_base +
                                     static_cast<int64_t>(q_group) *
                                         max_num_partitions * Policy::HeadDimV +
                                     n_base + col;
          tmp_out[out_offset] = out_value;
        }
      }
      __syncwarp();
    }

    if (warp == 0 && lane == 0) {
#pragma unroll
      for (int q_group = 0; q_group < kQPerKv; ++q_group) {
        const int64_t stats_offset =
            stats_base + static_cast<int64_t>(q_group) * max_num_partitions;
        exp_sums[stats_offset] = row_l[q_group] > 0.0f
                                     ? logf(row_l[q_group]) + row_m[q_group]
                                     : -FLT_MAX;
      }
    }
    return;
  }

  if constexpr (UseFa2Mainloop) {
    __shared__ int shared_token_valid[Policy::ComputeBlockN];
    __shared__ __align__(16) typename CuteQkTraits::Element
        shared_q_cute[CuteQkTraits::kQStorageElems];
    __shared__ __align__(16) typename CuteQkTraits::Element
        shared_k_cute[CuteQkTraits::kKStorageElems];
    __shared__ __align__(16) typename CutePvTraits::Element
        shared_v_full[CutePvTraits::kVStorageElems];

    auto sQ = cute::make_tensor(cute::make_smem_ptr(&shared_q_cute[0]),
                                typename CuteQkTraits::SmemLayoutQ{});
    auto sK = cute::make_tensor(cute::make_smem_ptr(&shared_k_cute[0]),
                                typename CuteQkTraits::SmemLayoutKV{});
    auto sV = cute::make_tensor(cute::make_smem_ptr(&shared_v_full[0]),
                                typename CutePvTraits::SmemLayoutKV{});

#pragma unroll 1
    for (int elem = threadIdx.x; elem < 16 * Policy::HeadDim;
         elem += NumThreads) {
      const int row = elem / Policy::HeadDim;
      const int col = elem - row * Policy::HeadDim;
      uint16_t bits = 0;
      if (row < kQPerKv) {
        const int head_idx = kv_head_idx * kQPerKv + row;
        const int64_t q_offset =
            static_cast<int64_t>(seq_idx) * q_stride_token +
            static_cast<int64_t>(head_idx) * q_stride_head + col * q_stride_dim;
        bits = query[q_offset];
      }
      sQ(row, col) = byte_v2_bf16_bits_to_cutlass_bfloat16(bits);
    }

#pragma unroll 1
    for (int elem = threadIdx.x; elem < Policy::ComputeBlockN * Policy::HeadDim;
         elem += NumThreads) {
      const int row = elem / Policy::HeadDim;
      const int col = elem - row * Policy::HeadDim;
      sK(row, col) = byte_v2_bf16_bits_to_cutlass_bfloat16(0);
      sV(row, col) = byte_v2_bf16_bits_to_cutlass_bfloat16(0);
    }
    for (int tile_offset = threadIdx.x; tile_offset < Policy::ComputeBlockN;
         tile_offset += NumThreads) {
      shared_token_valid[tile_offset] = 0;
    }
    __syncthreads();

#pragma unroll 1
    for (int32_t logical_block = first_logical_block;
         logical_block < last_logical_block; ++logical_block) {
      const int32_t block_token_start = logical_block * kBlockSize;
      const int row_start = max(tile_start - block_token_start, 0);
      const int row_end = min(tile_end - block_token_start, kBlockSize);
      if (logical_block >= max_num_blocks_per_seq) {
        continue;
      }
      const int32_t physical_block = block_table[logical_block];
      if (physical_block < 0) {
        continue;
      }

      const uint8_t* page = kv_cache + static_cast<int64_t>(physical_block) *
                                           kv_cache_stride_block;
      bool page_is_unsafe = false;
      if constexpr (UsePageUnsafeFlags) {
        page_is_unsafe = page_unsafe_flags[physical_block] != 0;
      }
      const int staged_rows = row_end - row_start;
#pragma unroll 1
      for (int row_rel = threadIdx.x; row_rel < staged_rows;
           row_rel += NumThreads) {
        const int row = row_start + row_rel;
        const int32_t tile_offset = block_token_start + row - tile_start;
        shared_token_valid[tile_offset] = 1;
      }

      if (page_is_unsafe) {
#pragma unroll 1
        for (int elem = threadIdx.x; elem < staged_rows * Policy::HeadDim;
             elem += NumThreads) {
          const int row_rel = elem / Policy::HeadDim;
          const int q_dim = elem - row_rel * Policy::HeadDim;
          const int row = row_start + row_rel;
          const int32_t tile_offset = block_token_start + row - tile_start;
          const float k_value = byte_v2_load_payload_elem<Layout, false>(
              page, k_payload_base_offset, kv_head_idx, row, q_dim);
          const float v_value = byte_v2_load_payload_elem<Layout, true>(
              page, v_payload_base_offset, kv_head_idx, row, q_dim);
          sK(tile_offset, q_dim) = byte_v2_float_to_cutlass_bfloat16(k_value);
          sV(tile_offset, q_dim) = byte_v2_float_to_cutlass_bfloat16(v_value);
        }
      } else {
#pragma unroll 1
        for (int elem = threadIdx.x; elem < staged_rows * Policy::HeadDim;
             elem += NumThreads) {
          const int row_rel = elem / Policy::HeadDim;
          const int q_dim = elem - row_rel * Policy::HeadDim;
          const int row = row_start + row_rel;
          const int32_t tile_offset = block_token_start + row - tile_start;
          const uint16_t k_bits =
              byte_v2_load_payload_elem_no_fallback_no_outlier_bits<Layout,
                                                                    false>(
                  page, k_payload_base_offset, kv_head_idx, row, q_dim);
          const uint16_t v_bits =
              byte_v2_load_payload_elem_no_fallback_no_outlier_bits<Layout,
                                                                    true>(
                  page, v_payload_base_offset, kv_head_idx, row, q_dim);
          sK(tile_offset, q_dim) =
              byte_v2_bf16_bits_to_cutlass_bfloat16(k_bits);
          sV(tile_offset, q_dim) =
              byte_v2_bf16_bits_to_cutlass_bfloat16(v_bits);
        }
      }
    }
    __syncthreads();

    if (warp == 0) {
      typename CutePvTraits::TiledMma tiled_mma;
      auto thr_mma = tiled_mma.get_thread_slice(lane);
      auto tCrQ = thr_mma.partition_fragment_A(sQ);
      auto tCrK = thr_mma.partition_fragment_B(sK);
      auto acc_s = cute::partition_fragment_C(
          tiled_mma,
          cute::Shape<cute::Int<16>, cute::Int<Policy::ComputeBlockN>>{});
      cute::clear(acc_s);

      auto smem_tiled_copy_Q = cute::make_tiled_copy_A(
          typename CuteQkTraits::SmemCopyAtom{}, tiled_mma);
      auto smem_tiled_copy_K = cute::make_tiled_copy_B(
          typename CuteQkTraits::SmemCopyAtom{}, tiled_mma);
      auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(lane);
      auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(lane);
      auto tCsQ = smem_thr_copy_Q.partition_S(sQ);
      auto tCsK = smem_thr_copy_K.partition_S(sK);
      byte_v2_cute_gemm_smem(acc_s, tCrQ, tCrK, tCsQ, tCsK, tiled_mma,
                             smem_tiled_copy_Q, smem_tiled_copy_K,
                             smem_thr_copy_Q, smem_thr_copy_K);

      auto cS = cute::make_identity_tensor(
          cute::Shape<cute::Int<16>, cute::Int<Policy::ComputeBlockN>>{});
      auto tCcS = thr_mma.partition_C(cS);
      float row_m[kQPerKv];
#pragma unroll
      for (int q_group = 0; q_group < kQPerKv; ++q_group) {
        row_m[q_group] = -FLT_MAX;
      }
#pragma unroll
      for (int elem = 0; elem < cute::size(acc_s); ++elem) {
        const int row = cute::get<0>(tCcS(elem));
        const int col = cute::get<1>(tCcS(elem));
        if (row < kQPerKv) {
          const float score = col < tile_len && shared_token_valid[col] != 0
                                  ? acc_s(elem) * scale
                                  : -FLT_MAX;
          row_m[row] = fmaxf(row_m[row], score);
        }
      }
#pragma unroll
      for (int q_group = 0; q_group < kQPerKv; ++q_group) {
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
          row_m[q_group] =
              fmaxf(row_m[q_group],
                    __shfl_down_sync(0xffffffff, row_m[q_group], offset));
        }
        row_m[q_group] = __shfl_sync(0xffffffff, row_m[q_group], 0);
      }

      float row_l[kQPerKv];
#pragma unroll
      for (int q_group = 0; q_group < kQPerKv; ++q_group) {
        row_l[q_group] = 0.0f;
      }
#pragma unroll
      for (int elem = 0; elem < cute::size(acc_s); ++elem) {
        const int row = cute::get<0>(tCcS(elem));
        const int col = cute::get<1>(tCcS(elem));
        float prob = 0.0f;
        if (row < kQPerKv && col < tile_len && shared_token_valid[col] != 0) {
          prob = __expf(acc_s(elem) * scale - row_m[row]);
          row_l[row] += prob;
        }
        acc_s(elem) = prob;
      }
#pragma unroll
      for (int q_group = 0; q_group < kQPerKv; ++q_group) {
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
          row_l[q_group] +=
              __shfl_down_sync(0xffffffff, row_l[q_group], offset);
        }
        row_l[q_group] = __shfl_sync(0xffffffff, row_l[q_group], 0);
      }

      auto rP = cute::make_tensor_like<typename CutePvTraits::Element>(acc_s);
#pragma unroll
      for (int elem = 0; elem < cute::size(rP); ++elem) {
        rP(elem) = byte_v2_float_to_cutlass_bfloat16(acc_s(elem));
      }
      auto tCrP = cute::make_tensor(
          rP.data(),
          byte_v2_convert_layout_acc_Aregs<typename CutePvTraits::TiledMma>(
              rP.layout()));

      auto sVt = cute::make_tensor(
          sV.data(), typename CutePvTraits::SmemLayoutVtransposed{});
      auto sVtNoSwizzle = cute::make_tensor(
          sV.data().get(),
          typename CutePvTraits::SmemLayoutVtransposedNoSwizzle{});
      auto tCrV = thr_mma.partition_fragment_B(sVtNoSwizzle);
      auto acc_o = cute::partition_fragment_C(
          tiled_mma, cute::Shape<cute::Int<16>, cute::Int<Policy::HeadDimV>>{});
      cute::clear(acc_o);
      auto smem_tiled_copy_V = cute::make_tiled_copy_B(
          typename CutePvTraits::SmemCopyAtomV{}, tiled_mma);
      auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(lane);
      auto tCsV = smem_thr_copy_V.partition_S(sVt);
      byte_v2_cute_gemm_rs(acc_o, tCrP, tCrV, tCsV, tiled_mma,
                           smem_tiled_copy_V, smem_thr_copy_V);

      auto cO = cute::make_identity_tensor(
          cute::Shape<cute::Int<16>, cute::Int<Policy::HeadDimV>>{});
      auto tCcO = thr_mma.partition_C(cO);
#pragma unroll
      for (int elem = 0; elem < cute::size(acc_o); ++elem) {
        const int q_group = cute::get<0>(tCcO(elem));
        const int col = cute::get<1>(tCcO(elem));
        if (q_group < kQPerKv && col < Policy::HeadDimV) {
          const float denom = row_l[q_group];
          const float out_value = denom > 0.0f ? acc_o(elem) / denom : 0.0f;
          const int64_t out_offset = tmp_base +
                                     static_cast<int64_t>(q_group) *
                                         max_num_partitions * Policy::HeadDimV +
                                     col;
          tmp_out[out_offset] = out_value;
        }
      }

      if (lane == 0) {
#pragma unroll
        for (int q_group = 0; q_group < kQPerKv; ++q_group) {
          const int64_t stats_offset =
              stats_base + static_cast<int64_t>(q_group) * max_num_partitions;
          exp_sums[stats_offset] = row_l[q_group] > 0.0f
                                       ? logf(row_l[q_group]) + row_m[q_group]
                                       : -FLT_MAX;
        }
      }
    }
    return;
  }

  if constexpr (UseQkMma) {
    __shared__ int shared_token_valid[Policy::ComputeBlockN];
    __shared__ __align__(16) typename CuteQkTraits::Element
        shared_q_cute[CuteQkTraits::kQStorageElems];
    __shared__ __align__(16) typename CuteQkTraits::Element
        shared_k_cute[CuteQkTraits::kKStorageElems];

    auto sQ = cute::make_tensor(cute::make_smem_ptr(&shared_q_cute[0]),
                                typename CuteQkTraits::SmemLayoutQ{});
    auto sK = cute::make_tensor(cute::make_smem_ptr(&shared_k_cute[0]),
                                typename CuteQkTraits::SmemLayoutKV{});

#pragma unroll 1
    for (int elem = threadIdx.x; elem < 16 * Policy::HeadDim;
         elem += NumThreads) {
      const int row = elem / Policy::HeadDim;
      const int col = elem - row * Policy::HeadDim;
      uint16_t bits = 0;
      if (row < kQPerKv) {
        const int head_idx = kv_head_idx * kQPerKv + row;
        const int64_t q_offset =
            static_cast<int64_t>(seq_idx) * q_stride_token +
            static_cast<int64_t>(head_idx) * q_stride_head + col * q_stride_dim;
        bits = query[q_offset];
      }
      sQ(row, col) = byte_v2_bf16_bits_to_cutlass_bfloat16(bits);
    }

#pragma unroll 1
    for (int elem = threadIdx.x; elem < Policy::ComputeBlockN * Policy::HeadDim;
         elem += NumThreads) {
      const int row = elem / Policy::HeadDim;
      const int col = elem - row * Policy::HeadDim;
      sK(row, col) = byte_v2_bf16_bits_to_cutlass_bfloat16(0);
    }
    for (int tile_offset = threadIdx.x; tile_offset < Policy::ComputeBlockN;
         tile_offset += NumThreads) {
      shared_token_valid[tile_offset] = 0;
    }
    __syncthreads();

#pragma unroll 1
    for (int32_t logical_block = first_logical_block;
         logical_block < last_logical_block; ++logical_block) {
      const int32_t block_token_start = logical_block * kBlockSize;
      const int row_start = max(tile_start - block_token_start, 0);
      const int row_end = min(tile_end - block_token_start, kBlockSize);
      if (logical_block >= max_num_blocks_per_seq) {
        continue;
      }
      const int32_t physical_block = block_table[logical_block];
      if (physical_block < 0) {
        continue;
      }

      const uint8_t* page = kv_cache + static_cast<int64_t>(physical_block) *
                                           kv_cache_stride_block;
      bool page_is_unsafe = false;
      if constexpr (UsePageUnsafeFlags) {
        page_is_unsafe = page_unsafe_flags[physical_block] != 0;
      }
      const int staged_rows = row_end - row_start;
#pragma unroll 1
      for (int row_rel = threadIdx.x; row_rel < staged_rows;
           row_rel += NumThreads) {
        const int row = row_start + row_rel;
        const int32_t tile_offset = block_token_start + row - tile_start;
        shared_token_valid[tile_offset] = 1;
      }

      if (page_is_unsafe) {
#pragma unroll 1
        for (int elem = threadIdx.x; elem < staged_rows * Policy::HeadDim;
             elem += NumThreads) {
          const int row_rel = elem / Policy::HeadDim;
          const int q_dim = elem - row_rel * Policy::HeadDim;
          const int row = row_start + row_rel;
          const int32_t tile_offset = block_token_start + row - tile_start;
          const float k_value = byte_v2_load_payload_elem<Layout, false>(
              page, k_payload_base_offset, kv_head_idx, row, q_dim);
          sK(tile_offset, q_dim) = byte_v2_float_to_cutlass_bfloat16(k_value);
        }
      } else {
#pragma unroll 1
        for (int elem = threadIdx.x; elem < staged_rows * Policy::HeadDim;
             elem += NumThreads) {
          const int row_rel = elem / Policy::HeadDim;
          const int q_dim = elem - row_rel * Policy::HeadDim;
          const int row = row_start + row_rel;
          const int32_t tile_offset = block_token_start + row - tile_start;
          const uint16_t bits =
              byte_v2_load_payload_elem_no_fallback_no_outlier_bits<Layout,
                                                                    false>(
                  page, k_payload_base_offset, kv_head_idx, row, q_dim);
          sK(tile_offset, q_dim) = byte_v2_bf16_bits_to_cutlass_bfloat16(bits);
        }
      }
    }
    __syncthreads();

    if (warp == 0) {
      typename CuteQkTraits::TiledMma tiled_mma;
      auto thr_mma = tiled_mma.get_thread_slice(lane);
      auto tCrQ = thr_mma.partition_fragment_A(sQ);
      auto tCrK = thr_mma.partition_fragment_B(sK);
      auto acc_s = cute::partition_fragment_C(
          tiled_mma,
          cute::Shape<cute::Int<16>, cute::Int<Policy::ComputeBlockN>>{});
      cute::clear(acc_s);
      auto smem_tiled_copy_Q = cute::make_tiled_copy_A(
          typename CuteQkTraits::SmemCopyAtom{}, tiled_mma);
      auto smem_tiled_copy_K = cute::make_tiled_copy_B(
          typename CuteQkTraits::SmemCopyAtom{}, tiled_mma);
      auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(lane);
      auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(lane);
      auto tCsQ = smem_thr_copy_Q.partition_S(sQ);
      auto tCsK = smem_thr_copy_K.partition_S(sK);
      byte_v2_cute_gemm_smem(acc_s, tCrQ, tCrK, tCsQ, tCsK, tiled_mma,
                             smem_tiled_copy_Q, smem_tiled_copy_K,
                             smem_thr_copy_Q, smem_thr_copy_K);

      auto cS = cute::make_identity_tensor(
          cute::Shape<cute::Int<16>, cute::Int<Policy::ComputeBlockN>>{});
      auto tCcS = thr_mma.partition_C(cS);
#pragma unroll
      for (int elem = 0; elem < cute::size(acc_s); ++elem) {
        const int row = cute::get<0>(tCcS(elem));
        const int col = cute::get<1>(tCcS(elem));
        if (row < kQPerKv) {
          shared_scores[row][col] =
              col < tile_len && shared_token_valid[col] != 0
                  ? acc_s(elem) * scale
                  : -FLT_MAX;
        }
      }
    }
    __syncthreads();
  } else {
    constexpr int kQkDimsPerThread = Policy::HeadDim / 32;
    float q_values[kQkDimsPerThread];
#pragma unroll
    for (int q_dim_iter = 0; q_dim_iter < kQkDimsPerThread; ++q_dim_iter) {
      const int head_idx = kv_head_idx * kQPerKv + warp;
      const int q_dim = q_dim_iter * 32 + lane;
      const int64_t q_offset = static_cast<int64_t>(seq_idx) * q_stride_token +
                               static_cast<int64_t>(head_idx) * q_stride_head +
                               q_dim * q_stride_dim;
      q_values[q_dim_iter] = byte_v2_bf16_bits_to_float(query[q_offset]);
    }

    __shared__ uint16_t
        shared_k_tile[Policy::AllocBlockTokens][Policy::HeadDim];
#pragma unroll 1
    for (int32_t logical_block = first_logical_block;
         logical_block < last_logical_block; ++logical_block) {
      const int32_t block_token_start = logical_block * kBlockSize;
      const int row_start = max(tile_start - block_token_start, 0);
      const int row_end = min(tile_end - block_token_start, kBlockSize);
      if (logical_block >= max_num_blocks_per_seq) {
        if (threadIdx.x == 0) {
#pragma unroll 1
          for (int row = row_start; row < row_end; ++row) {
            const int32_t tile_offset = block_token_start + row - tile_start;
#pragma unroll
            for (int q_group = 0; q_group < kQPerKv; ++q_group) {
              shared_scores[q_group][tile_offset] = -FLT_MAX;
            }
          }
        }
        continue;
      }
      const int32_t physical_block = block_table[logical_block];
      if (physical_block < 0) {
        if (threadIdx.x == 0) {
#pragma unroll 1
          for (int row = row_start; row < row_end; ++row) {
            const int32_t tile_offset = block_token_start + row - tile_start;
#pragma unroll
            for (int q_group = 0; q_group < kQPerKv; ++q_group) {
              shared_scores[q_group][tile_offset] = -FLT_MAX;
            }
          }
        }
        continue;
      }

      const uint8_t* page = kv_cache + static_cast<int64_t>(physical_block) *
                                           kv_cache_stride_block;
      bool page_is_unsafe = false;
      if constexpr (UsePageUnsafeFlags) {
        page_is_unsafe = page_unsafe_flags[physical_block] != 0;
      }
      if (page_is_unsafe) {
#pragma unroll 1
        for (int row = row_start; row < row_end; ++row) {
          const int32_t tile_offset = block_token_start + row - tile_start;
          float qk_sum = 0.0f;
#pragma unroll
          for (int q_dim_iter = 0; q_dim_iter < kQkDimsPerThread;
               ++q_dim_iter) {
            const int q_dim = q_dim_iter * 32 + lane;
            const float k_value = byte_v2_load_payload_elem<Layout, false>(
                page, k_payload_base_offset, kv_head_idx, row, q_dim);
            qk_sum += q_values[q_dim_iter] * k_value;
          }
#pragma unroll
          for (int offset = 16; offset > 0; offset >>= 1) {
            qk_sum += __shfl_down_sync(0xffffffff, qk_sum, offset);
          }
          if (lane == 0) {
            shared_scores[warp][tile_offset] = qk_sum * scale;
          }
        }
      } else {
        const int staged_rows = row_end - row_start;
        const int staged_elems = staged_rows * Policy::HeadDim;
        const int staged_pairs = staged_elems / 2;
        const int thread_k_dim = (threadIdx.x * 2) & (Policy::HeadDim - 1);
        const int thread_dim_tile = thread_k_dim / Policy::CodecDimBlock;
        const int thread_dim_in_tile = thread_k_dim % Policy::CodecDimBlock;
        const uint32_t thread_k_tile_offset =
            k_payload_base_offset +
            thread_dim_tile * Layout::CodecPayloadBytesPerTile;
        const uint8_t thread_k_base =
            page[Layout::k_base_offset(kv_head_idx, thread_dim_tile, 0)];
#pragma unroll 1
        for (int pair = threadIdx.x; pair < staged_pairs; pair += NumThreads) {
          const int elem = pair * 2;
          const int row_rel = elem / Policy::HeadDim;
          const int row = row_start + row_rel;
          uint16_t bits0;
          uint16_t bits1;
          byte_v2_load_k_payload_elem_pair_from_fixed_tile_no_outlier_bits<
              Layout>(page, thread_k_tile_offset, row, thread_dim_in_tile,
                      thread_k_base, bits0, bits1);
          shared_k_tile[row][thread_k_dim] = bits0;
          shared_k_tile[row][thread_k_dim + 1] = bits1;
        }
        __syncthreads();
#pragma unroll 1
        for (int row = row_start; row < row_end; ++row) {
          const int32_t tile_offset = block_token_start + row - tile_start;
          float qk_sum = 0.0f;
#pragma unroll
          for (int q_dim_iter = 0; q_dim_iter < kQkDimsPerThread;
               ++q_dim_iter) {
            const int q_dim = q_dim_iter * 32 + lane;
            const float k_value =
                byte_v2_bf16_bits_to_float(shared_k_tile[row][q_dim]);
            qk_sum += q_values[q_dim_iter] * k_value;
          }
#pragma unroll
          for (int offset = 16; offset > 0; offset >>= 1) {
            qk_sum += __shfl_down_sync(0xffffffff, qk_sum, offset);
          }
          if (lane == 0) {
            shared_scores[warp][tile_offset] = qk_sum * scale;
          }
        }
        __syncthreads();
      }
    }
    __syncthreads();
  }

  float tile_m = -FLT_MAX;
#pragma unroll 1
  for (int32_t tile_offset = lane; tile_offset < tile_len; tile_offset += 32) {
    tile_m = fmaxf(tile_m, shared_scores[warp][tile_offset]);
  }
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    tile_m = fmaxf(tile_m, __shfl_down_sync(0xffffffff, tile_m, offset));
  }
  tile_m = __shfl_sync(0xffffffff, tile_m, 0);
  const float new_m = fmaxf(softmax_m[warp], tile_m);
  const float alpha = __expf(softmax_m[warp] - new_m);

  float tile_l = 0.0f;
#pragma unroll 1
  for (int32_t tile_offset = lane; tile_offset < tile_len; tile_offset += 32) {
    const float prob = __expf(shared_scores[warp][tile_offset] - new_m);
    shared_probs[warp][tile_offset] = prob;
    tile_l += prob;
  }
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    tile_l += __shfl_down_sync(0xffffffff, tile_l, offset);
  }
  if (lane == 0) {
    shared_new_m[warp] = new_m;
    shared_new_l[warp] = softmax_l[warp] * alpha + tile_l;
  }
  __syncthreads();

  if (warp < kQPerKv) {
#pragma unroll
    for (int n_iter = 0; n_iter < 2; ++n_iter) {
      const int tile_n = n_iter * kQPerKv + warp;
      const int n_base = tile_n * 16;
      auto sV = cute::make_tensor(
          cute::make_smem_ptr(
              reinterpret_cast<typename CutePvTileTraits::Element*>(
                  &shared_v_cute[warp][0])),
          typename CutePvTileTraits::SmemLayoutV{});
#pragma unroll 1
      for (int elem = lane; elem < Policy::ComputeBlockN * 16; elem += 32) {
        const int row = elem / 16;
        const int col = elem - row * 16;
        sV(row, col) = byte_v2_bf16_bits_to_cutlass_bfloat16(0);
      }
      __syncwarp();

#pragma unroll 1
      for (int32_t logical_block = first_logical_block;
           logical_block < last_logical_block; ++logical_block) {
        if (logical_block >= max_num_blocks_per_seq) {
          continue;
        }
        const int32_t physical_block = block_table[logical_block];
        if (physical_block < 0) {
          continue;
        }

        const int32_t block_token_start = logical_block * kBlockSize;
        const int row_start = max(tile_start - block_token_start, 0);
        const int row_end = min(tile_end - block_token_start, kBlockSize);
        const int staged_rows = row_end - row_start;
        const uint8_t* page = kv_cache + static_cast<int64_t>(physical_block) *
                                             kv_cache_stride_block;
        bool page_is_unsafe = false;
        if constexpr (UsePageUnsafeFlags) {
          page_is_unsafe = page_unsafe_flags[physical_block] != 0;
        }

        if (page_is_unsafe) {
#pragma unroll 1
          for (int elem = lane; elem < staged_rows * 16; elem += 32) {
            const int row_rel = elem / 16;
            const int col = elem - row_rel * 16;
            const int row = row_start + row_rel;
            const int32_t tile_offset = block_token_start + row - tile_start;
            const int v_dim = n_base + col;
            const float v_value = byte_v2_load_payload_elem<Layout, true>(
                page, v_payload_base_offset, kv_head_idx, row, v_dim);
            sV(tile_offset, col) = byte_v2_float_to_cutlass_bfloat16(v_value);
          }
        } else {
#pragma unroll 1
          for (int elem = lane; elem < staged_rows * 16; elem += 32) {
            const int row_rel = elem / 16;
            const int col = elem - row_rel * 16;
            const int row = row_start + row_rel;
            const int32_t tile_offset = block_token_start + row - tile_start;
            const int v_dim = n_base + col;
            const uint16_t bits =
                byte_v2_load_payload_elem_no_fallback_no_outlier_bits<Layout,
                                                                      true>(
                    page, v_payload_base_offset, kv_head_idx, row, v_dim);
            sV(tile_offset, col) = byte_v2_bf16_bits_to_cutlass_bfloat16(bits);
          }
        }
      }
      __syncwarp();

      typename CutePvTileTraits::TiledMma tiled_mma;
      auto thr_mma = tiled_mma.get_thread_slice(lane);
      auto sVt = cute::make_tensor(sV.data(),
                                   typename CutePvTileTraits::SmemLayoutVt{});
      auto sVtNoSwizzle = cute::make_tensor(
          sV.data().get(), typename CutePvTileTraits::SmemLayoutVtNoSwizzle{});
      auto p_acc = cute::partition_fragment_C(
          tiled_mma,
          cute::Shape<cute::Int<16>, cute::Int<Policy::ComputeBlockN>>{});
      auto cP = cute::make_identity_tensor(
          cute::Shape<cute::Int<16>, cute::Int<Policy::ComputeBlockN>>{});
      auto tCcP = thr_mma.partition_C(cP);
      auto rP =
          cute::make_tensor_like<typename CutePvTileTraits::Element>(p_acc);
#pragma unroll
      for (int elem = 0; elem < cute::size(rP); ++elem) {
        const int row = cute::get<0>(tCcP(elem));
        const int col = cute::get<1>(tCcP(elem));
        const float prob =
            row < kQPerKv && col < tile_len ? shared_probs[row][col] : 0.0f;
        rP(elem) = byte_v2_float_to_cutlass_bfloat16(prob);
      }
      auto tCrP = cute::make_tensor(
          rP.data(),
          byte_v2_convert_layout_acc_Aregs<typename CutePvTileTraits::TiledMma>(
              rP.layout()));
      auto tCrV = thr_mma.partition_fragment_B(sVtNoSwizzle);
      auto acc_o = cute::partition_fragment_C(
          tiled_mma, cute::Shape<cute::Int<16>, cute::Int<16>>{});
      cute::clear(acc_o);
      auto smem_tiled_copy_V = cute::make_tiled_copy_B(
          typename CutePvTileTraits::SmemCopyAtomV{}, tiled_mma);
      auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(lane);
      auto tCsV = smem_thr_copy_V.partition_S(sVt);
      byte_v2_cute_gemm_rs(acc_o, tCrP, tCrV, tCsV, tiled_mma,
                           smem_tiled_copy_V, smem_thr_copy_V);

      auto cO = cute::make_identity_tensor(
          cute::Shape<cute::Int<16>, cute::Int<16>>{});
      auto tCcO = thr_mma.partition_C(cO);
#pragma unroll
      for (int elem = 0; elem < cute::size(acc_o); ++elem) {
        const int q_group = cute::get<0>(tCcO(elem));
        const int col = cute::get<1>(tCcO(elem));
        if (q_group < kQPerKv && col < 16) {
          const float denom = shared_new_l[q_group];
          const float out_value = denom > 0.0f ? acc_o(elem) / denom : 0.0f;
          const int64_t out_offset = tmp_base +
                                     static_cast<int64_t>(q_group) *
                                         max_num_partitions * Policy::HeadDimV +
                                     n_base + col;
          tmp_out[out_offset] = out_value;
        }
      }
      __syncwarp();
    }
  }

#pragma unroll
  for (int q_group = 0; q_group < kQPerKv; ++q_group) {
    softmax_l[q_group] = shared_new_l[q_group];
    softmax_m[q_group] = shared_new_m[q_group];
  }
  __syncthreads();

  if (threadIdx.x == 0) {
#pragma unroll
    for (int q_group = 0; q_group < kQPerKv; ++q_group) {
      const int64_t stats_offset =
          stats_base + static_cast<int64_t>(q_group) * max_num_partitions;
      exp_sums[stats_offset] =
          softmax_l[q_group] > 0.0f
              ? logf(softmax_l[q_group]) + softmax_m[q_group]
              : -FLT_MAX;
    }
  }
}
template <typename Layout, int NumThreads>
__global__ void byte_v2_paged_decode_attention_split_k_reduce_kernel(
    uint16_t* __restrict__ output, const float* __restrict__ tmp_out,
    const float* __restrict__ exp_sums, const float* __restrict__ max_logits,
    const int32_t* __restrict__ seq_lens, int64_t out_stride_token,
    int64_t out_stride_head, int64_t out_stride_dim, int max_num_partitions,
    int partition_size) {
  using Policy = typename Layout::TilePolicy;
  (void)max_logits;

  const int seq_idx = blockIdx.y;
  const int head_idx = blockIdx.x;
  const int dim = threadIdx.x;
  const int num_heads = gridDim.x;
  const int32_t seq_len = seq_lens[seq_idx];
  const int num_partitions = (seq_len + partition_size - 1) / partition_size;

  const int64_t stats_base =
      (static_cast<int64_t>(seq_idx) * num_heads + head_idx) *
      max_num_partitions;
  const int64_t tmp_base = stats_base * static_cast<int64_t>(Policy::HeadDimV);
  const int64_t out_offset = static_cast<int64_t>(seq_idx) * out_stride_token +
                             static_cast<int64_t>(head_idx) * out_stride_head +
                             dim * out_stride_dim;

  if (num_partitions <= 0) {
    output[out_offset] = byte_v2_float_to_bf16_bits(0.0f);
    return;
  }
  if (num_partitions == 1) {
    output[out_offset] = byte_v2_float_to_bf16_bits(tmp_out[tmp_base + dim]);
    return;
  }

  extern __shared__ float shared_stats[];
  float* shared_weights = shared_stats;
  __shared__ float reduce_smem[NumThreads / 32];

  float local_lse_max = -FLT_MAX;
#pragma unroll 1
  for (int idx = threadIdx.x; idx < num_partitions; idx += blockDim.x) {
    const float value = exp_sums[stats_base + idx];
    local_lse_max = fmaxf(local_lse_max, value);
  }
  __syncthreads();
  const float global_lse_max =
      byte_v2_block_max<NumThreads>(local_lse_max, reduce_smem);

  float local_sum = 0.0f;
#pragma unroll 1
  for (int idx = threadIdx.x; idx < num_partitions; idx += blockDim.x) {
    const float partition_lse = exp_sums[stats_base + idx];
    const float weight = partition_lse == -FLT_MAX
                             ? 0.0f
                             : __expf(partition_lse - global_lse_max);
    shared_weights[idx] = weight;
    local_sum += weight;
  }
  __syncthreads();
  const float global_sum =
      byte_v2_block_sum<NumThreads>(local_sum, reduce_smem);

  float acc = 0.0f;
#pragma unroll 1
  for (int idx = 0; idx < num_partitions; ++idx) {
    acc +=
        tmp_out[tmp_base + idx * Policy::HeadDimV + dim] * shared_weights[idx];
  }
  const float out_value = global_sum > 0.0f ? acc / global_sum : 0.0f;
  output[out_offset] = byte_v2_float_to_bf16_bits(out_value);
}

template <typename Layout, bool UsePageUnsafeFlags, int QHeadsPerKv,
          int QGroupTile, int NumThreads>
__global__ void
byte_v2_paged_decode_attention_split_k_gqa4_qk_window16_diagnostic_kernel(
    float* __restrict__ tmp_out, float* __restrict__ exp_sums,
    float* __restrict__ max_logits, const uint16_t* __restrict__ query,
    const uint8_t* __restrict__ kv_cache,
    const int32_t* __restrict__ page_unsafe_flags,
    const int32_t* __restrict__ block_tables,
    const int32_t* __restrict__ seq_lens, float scale, int64_t q_stride_token,
    int64_t q_stride_head, int64_t q_stride_dim, int64_t kv_cache_stride_block,
    int64_t block_table_stride_seq, int max_num_blocks_per_seq,
    int max_num_partitions, int partition_size) {
  using Policy = typename Layout::TilePolicy;
  (void)scale;

  constexpr int kBlockSize = Policy::AllocBlockTokens;
  constexpr int kQPerKv = QGroupTile;
  constexpr int kGroupsPerKv = (QHeadsPerKv + QGroupTile - 1) / QGroupTile;
  constexpr int kNumQHeads = Layout::NumKvHeadsValue * QHeadsPerKv;
  constexpr int kWindowDim = 16;
  constexpr int kWindowsPerTile = Policy::HeadDim / kWindowDim;
  static_assert(QHeadsPerKv == 4);
  static_assert(QGroupTile == 4);
  static_assert(NumThreads == Policy::HeadDim);
  static_assert(Policy::HeadDim % kWindowDim == 0);
  static_assert(Policy::CodecDimBlock == kWindowDim);
  static_assert(Policy::ComputeBlockN == 64);

  using WindowQkTraits =
      ByteV2Fa2LikeCuteQkTraits<16, Policy::ComputeBlockN, kWindowDim, 1>;

  const int seq_idx = blockIdx.y;
  const int kv_group_idx = blockIdx.x;
  const int kv_head_idx = kv_group_idx / kGroupsPerKv;
  const int q_group_block = kv_group_idx - kv_head_idx * kGroupsPerKv;
  const int q_group_base = q_group_block * QGroupTile;
  const int valid_q_rows = min(QGroupTile, QHeadsPerKv - q_group_base);
  const int partition_idx = blockIdx.z;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int dim = threadIdx.x;
  const int32_t seq_len = seq_lens[seq_idx];
  const int32_t partition_start = partition_idx * partition_size;
  const int32_t partition_end =
      min(partition_start + partition_size, static_cast<int32_t>(seq_len));
  const int64_t stats_base = (static_cast<int64_t>(seq_idx) * kNumQHeads +
                              kv_head_idx * QHeadsPerKv + q_group_base) *
                                 max_num_partitions +
                             partition_idx;
  const int64_t tmp_base = stats_base * static_cast<int64_t>(Policy::HeadDimV);

  if (partition_start >= seq_len || partition_start >= partition_end) {
    if (threadIdx.x == 0) {
#pragma unroll
      for (int q_group = 0; q_group < kQPerKv; ++q_group) {
        if (q_group < valid_q_rows) {
          exp_sums[stats_base + static_cast<int64_t>(q_group) *
                                    max_num_partitions] = -FLT_MAX;
        }
      }
    }
#pragma unroll
    for (int q_group = 0; q_group < kQPerKv; ++q_group) {
      if (q_group < valid_q_rows) {
        tmp_out[tmp_base +
                static_cast<int64_t>(q_group) * max_num_partitions *
                    Policy::HeadDimV +
                dim] = 0.0f;
      }
    }
    return;
  }

  __shared__ __align__(16) typename WindowQkTraits::Element
      shared_q_window[WindowQkTraits::kQStorageElems];
  __shared__ __align__(16) typename WindowQkTraits::Element
      shared_k_window[WindowQkTraits::kKStorageElems];

  auto sQ = cute::make_tensor(cute::make_smem_ptr(&shared_q_window[0]),
                              typename WindowQkTraits::SmemLayoutQ{});
  auto sK = cute::make_tensor(cute::make_smem_ptr(&shared_k_window[0]),
                              typename WindowQkTraits::SmemLayoutKV{});

  const int32_t* block_table =
      block_tables + static_cast<int64_t>(seq_idx) * block_table_stride_seq;

  constexpr uint32_t kPayloadBaseOffset = Layout::k_payload_offset(0, 0);
  const uint32_t k_payload_base_offset =
      kPayloadBaseOffset + kv_head_idx * Layout::AlignedKPayloadBytesPerKvHead;

  const bool profile_block = max_logits != nullptr && blockIdx.x == 0 &&
                             blockIdx.y == 0 && blockIdx.z == 0;
  uint64_t total_start = 0;
  uint64_t stage_cycles = 0;
  uint64_t qk_cycles = 0;
  uint64_t stage_wait_cycles = 0;
  uint64_t qk_wait_cycles = 0;
  int direct_tile_count = 0;
  int qk_window_count = 0;
  if (profile_block && lane == 0) {
    total_start = clock64();
  }

  typename WindowQkTraits::TiledMma qk_mma;
  auto qk_thr_mma = qk_mma.get_thread_slice(lane);
  auto smem_tiled_copy_Q =
      cute::make_tiled_copy_A(typename WindowQkTraits::SmemCopyAtom{}, qk_mma);
  auto smem_tiled_copy_K =
      cute::make_tiled_copy_B(typename WindowQkTraits::SmemCopyAtom{}, qk_mma);
  auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(lane);
  auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(lane);
  float qk_sink = 0.0f;

#pragma unroll 1
  for (int32_t tile_start = partition_start; tile_start < partition_end;
       tile_start += Policy::ComputeBlockN) {
    const int32_t tile_end =
        min(tile_start + Policy::ComputeBlockN, partition_end);
    const int32_t tile_len = tile_end - tile_start;
    const int32_t first_logical_block = tile_start / kBlockSize;
    const int32_t last_logical_block = (tile_end + kBlockSize - 1) / kBlockSize;
    if (profile_block && threadIdx.x == 0) {
      ++direct_tile_count;
    }

    auto acc_s = cute::partition_fragment_C(
        qk_mma, cute::Shape<cute::Int<16>, cute::Int<Policy::ComputeBlockN>>{});
    cute::clear(acc_s);

#pragma unroll 1
    for (int dim_window = 0; dim_window < Policy::HeadDim;
         dim_window += kWindowDim) {
      uint64_t stage_start = 0;
      uint64_t stage_done = 0;
      if (profile_block && threadIdx.x == 0) {
        stage_start = clock64();
        ++qk_window_count;
      }

#pragma unroll 1
      for (int elem = threadIdx.x; elem < 16 * kWindowDim; elem += NumThreads) {
        const int row = elem / kWindowDim;
        const int col = elem - row * kWindowDim;
        uint16_t bits = 0;
        if (row < valid_q_rows) {
          const int head_idx = kv_head_idx * QHeadsPerKv + q_group_base + row;
          const int64_t q_offset =
              static_cast<int64_t>(seq_idx) * q_stride_token +
              static_cast<int64_t>(head_idx) * q_stride_head +
              static_cast<int64_t>(dim_window + col) * q_stride_dim;
          bits = query[q_offset];
        }
        sQ(row, col) = byte_v2_bf16_bits_to_cutlass_bfloat16(bits);
      }

      if (tile_len < Policy::ComputeBlockN) {
#pragma unroll 1
        for (int elem = threadIdx.x; elem < Policy::ComputeBlockN * kWindowDim;
             elem += NumThreads) {
          const int row = elem / kWindowDim;
          const int col = elem - row * kWindowDim;
          sK(row, col) = byte_v2_bf16_bits_to_cutlass_bfloat16(0);
        }
      }

#pragma unroll 1
      for (int32_t logical_block = first_logical_block;
           logical_block < last_logical_block; ++logical_block) {
        const int32_t block_token_start = logical_block * kBlockSize;
        const int row_start = max(tile_start - block_token_start, 0);
        const int row_end = min(tile_end - block_token_start, kBlockSize);
        const int staged_rows = row_end - row_start;
        const int32_t tile_offset_base = block_token_start - tile_start;
        const bool invalid_logical_block =
            logical_block >= max_num_blocks_per_seq;
        const int32_t physical_block =
            invalid_logical_block ? -1 : block_table[logical_block];
        bool zero_block = invalid_logical_block || physical_block < 0;
        const uint8_t* page =
            zero_block ? nullptr
                       : kv_cache + static_cast<int64_t>(physical_block) *
                                        kv_cache_stride_block;
        if constexpr (UsePageUnsafeFlags) {
          if (!zero_block) {
            zero_block = page_unsafe_flags[physical_block] != 0;
          }
        }

        if (threadIdx.x < staged_rows) {
          const int row = row_start + threadIdx.x;
          const int32_t tile_offset = tile_offset_base + row;
          uint16_t bits0 = 0;
          uint16_t bits1 = 0;
          uint16_t bits2 = 0;
          uint16_t bits3 = 0;
          uint16_t bits4 = 0;
          uint16_t bits5 = 0;
          uint16_t bits6 = 0;
          uint16_t bits7 = 0;
          uint16_t bits8 = 0;
          uint16_t bits9 = 0;
          uint16_t bits10 = 0;
          uint16_t bits11 = 0;
          uint16_t bits12 = 0;
          uint16_t bits13 = 0;
          uint16_t bits14 = 0;
          uint16_t bits15 = 0;
          if (!zero_block) {
            const int dim_tile = dim_window / Policy::CodecDimBlock;
            constexpr int dim_in_tile = 0;
            const auto k_desc =
                byte_v2_make_payload_tile_descriptor<Layout, false>(
                    page, k_payload_base_offset, kv_head_idx, dim_tile, false);
            byte_v2_load_payload_elem_hex_from_safe_tile_descriptor_fixed_dim_bits<
                Layout>(k_desc, row, dim_in_tile, bits0, bits1, bits2, bits3,
                        bits4, bits5, bits6, bits7, bits8, bits9, bits10,
                        bits11, bits12, bits13, bits14, bits15);
          }
          byte_v2_store_16_bf16_bits_to_smem(
              &shared_k_window[0], typename WindowQkTraits::SmemLayoutKV{},
              tile_offset, 0, bits0, bits1, bits2, bits3, bits4, bits5, bits6,
              bits7, bits8, bits9, bits10, bits11, bits12, bits13, bits14,
              bits15);
        }
      }

      if (profile_block && threadIdx.x == 0) {
        stage_done = clock64();
      }
      uint64_t stage_wait_start = 0;
      if (profile_block && lane == 0) {
        stage_wait_start = clock64();
      }
      __syncthreads();
      if (profile_block && lane == 0) {
        stage_wait_cycles += clock64() - stage_wait_start;
        if (threadIdx.x == 0) {
          stage_cycles += stage_done - stage_start;
        }
      }

      if (warp == 0) {
        uint64_t qk_start = 0;
        if (profile_block && threadIdx.x == 0) {
          qk_start = clock64();
        }
        auto tCrQ = qk_thr_mma.partition_fragment_A(sQ);
        auto tCrK = qk_thr_mma.partition_fragment_B(sK);
        auto tCsQ = smem_thr_copy_Q.partition_S(sQ);
        auto tCsK = smem_thr_copy_K.partition_S(sK);
        byte_v2_cute_gemm_smem(acc_s, tCrQ, tCrK, tCsQ, tCsK, qk_mma,
                               smem_tiled_copy_Q, smem_tiled_copy_K,
                               smem_thr_copy_Q, smem_thr_copy_K);
        if (profile_block && threadIdx.x == 0) {
          qk_cycles += clock64() - qk_start;
        }
      }

      uint64_t qk_wait_start = 0;
      if (profile_block && lane == 0) {
        qk_wait_start = clock64();
      }
      __syncthreads();
      if (profile_block && lane == 0) {
        qk_wait_cycles += clock64() - qk_wait_start;
      }
    }

    if (warp == 0) {
#pragma unroll
      for (int elem = 0; elem < cute::size(acc_s); ++elem) {
        qk_sink += acc_s(elem);
      }
    }
  }

  if (threadIdx.x == 0) {
#pragma unroll
    for (int q_group = 0; q_group < kQPerKv; ++q_group) {
      if (q_group < valid_q_rows) {
        exp_sums[stats_base +
                 static_cast<int64_t>(q_group) * max_num_partitions] = 0.0f;
      }
    }
  }
#pragma unroll
  for (int q_group = 0; q_group < kQPerKv; ++q_group) {
    if (q_group < valid_q_rows) {
      tmp_out[tmp_base +
              static_cast<int64_t>(q_group) * max_num_partitions *
                  Policy::HeadDimV +
              dim] = (q_group == 0 && warp == 0) ? qk_sink : 0.0f;
    }
  }

  if (profile_block && lane == 0) {
    if (threadIdx.x == 0) {
      max_logits[0] = static_cast<float>(clock64() - total_start);
      max_logits[1] = static_cast<float>(stage_cycles);
      max_logits[2] = static_cast<float>(qk_cycles);
      max_logits[3] = 0.0f;
      max_logits[4] = 0.0f;
      max_logits[13] = static_cast<float>(direct_tile_count);
      max_logits[14] = static_cast<float>(Policy::ComputeBlockN);
      max_logits[15] = static_cast<float>(partition_size);
      max_logits[20] = static_cast<float>(qk_window_count);
      max_logits[21] = static_cast<float>(kWindowDim);
      max_logits[22] = static_cast<float>(kWindowsPerTile);
    }
    max_logits[5 + warp] = static_cast<float>(stage_wait_cycles);
    max_logits[9 + warp] = static_cast<float>(qk_wait_cycles);
    max_logits[16 + warp] = 0.0f;
  }
}

template <typename Layout, bool UseUnnormalizedOutput = false,
          int SpeculativeQueryLen = 0>
__global__ void byte_v2_paged_decode_attention_split_k_reduce_warp_kernel(
    uint16_t* __restrict__ output, const float* __restrict__ tmp_out,
    const float* __restrict__ exp_sums, const float* __restrict__ max_logits,
    const int32_t* __restrict__ seq_lens, int64_t out_stride_token,
    int64_t out_stride_head, int64_t out_stride_dim, int max_num_partitions,
    int partition_size) {
  using Policy = typename Layout::TilePolicy;
  if constexpr (!UseUnnormalizedOutput) {
    (void)max_logits;
  }

  constexpr int kWarpSize = 32;
  constexpr int kWarpsPerBlock = 4;
  const int seq_idx = blockIdx.y;
  const int head_idx = blockIdx.x;
  const int lane = threadIdx.x & (kWarpSize - 1);
  const int warp = threadIdx.x >> 5;
  const int dim = blockIdx.z * kWarpsPerBlock + warp;
  const int num_heads = gridDim.x;
  const int32_t seq_len = seq_lens[seq_idx];
  const int num_partitions = (seq_len + partition_size - 1) / partition_size;
  if (dim >= Policy::HeadDimV) {
    return;
  }

  const int64_t stats_base =
      (static_cast<int64_t>(seq_idx) * num_heads + head_idx) *
      max_num_partitions;
  const int64_t tmp_base = stats_base * static_cast<int64_t>(Policy::HeadDimV);
  int out_token_idx = seq_idx;
  int out_head_idx = head_idx;
  if constexpr (SpeculativeQueryLen > 0) {
    constexpr int kQueryHeadsPerKv = 4;
    constexpr int kVirtualRowsPerKv = SpeculativeQueryLen * kQueryHeadsPerKv;
    const int kv_head_idx = head_idx / kVirtualRowsPerKv;
    const int virtual_row = head_idx - kv_head_idx * kVirtualRowsPerKv;
    out_token_idx =
        seq_idx * SpeculativeQueryLen + virtual_row / kQueryHeadsPerKv;
    out_head_idx =
        kv_head_idx * kQueryHeadsPerKv + virtual_row % kQueryHeadsPerKv;
  }
  const int64_t out_offset =
      static_cast<int64_t>(out_token_idx) * out_stride_token +
      static_cast<int64_t>(out_head_idx) * out_stride_head +
      dim * out_stride_dim;

  if (num_partitions <= 0) {
    if (lane == 0) {
      output[out_offset] = byte_v2_float_to_bf16_bits(0.0f);
    }
    return;
  }
  if (num_partitions == 1) {
    if (lane == 0) {
      float out_value = tmp_out[tmp_base + dim];
      if constexpr (UseUnnormalizedOutput) {
        const float denom = exp_sums[stats_base];
        out_value = denom > 0.0f ? out_value / denom : 0.0f;
      }
      output[out_offset] = byte_v2_float_to_bf16_bits(out_value);
    }
    return;
  }

  float local_lse_max = -FLT_MAX;
#pragma unroll 1
  for (int idx = lane; idx < num_partitions; idx += kWarpSize) {
    float partition_max;
    if constexpr (UseUnnormalizedOutput) {
      partition_max = max_logits[stats_base + idx];
    } else {
      partition_max = exp_sums[stats_base + idx];
    }
    local_lse_max = fmaxf(local_lse_max, partition_max);
  }
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    local_lse_max = fmaxf(local_lse_max,
                          __shfl_down_sync(0xffffffff, local_lse_max, offset));
  }
  const float global_lse_max = __shfl_sync(0xffffffff, local_lse_max, 0);

  float local_sum = 0.0f;
  float local_acc = 0.0f;
#pragma unroll 1
  for (int idx = lane; idx < num_partitions; idx += kWarpSize) {
    float weight;
    if constexpr (UseUnnormalizedOutput) {
      const float partition_m = max_logits[stats_base + idx];
      weight =
          partition_m == -FLT_MAX ? 0.0f : __expf(partition_m - global_lse_max);
      local_sum += exp_sums[stats_base + idx] * weight;
    } else {
      const float partition_lse = exp_sums[stats_base + idx];
      weight = partition_lse == -FLT_MAX
                   ? 0.0f
                   : __expf(partition_lse - global_lse_max);
      local_sum += weight;
    }
    local_acc += tmp_out[tmp_base + idx * Policy::HeadDimV + dim] * weight;
  }
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    local_acc += __shfl_down_sync(0xffffffff, local_acc, offset);
  }
  if (lane == 0) {
    const float out_value = local_sum > 0.0f ? local_acc / local_sum : 0.0f;
    output[out_offset] = byte_v2_float_to_bf16_bits(out_value);
  }
}

template <typename Layout, int SpeculativeQueryLen,
          bool UseRaggedSpeculativeQ4 = false>
__global__ void
byte_v2_paged_decode_attention_split_k_reduce_speculative_row_kernel(
    uint16_t* __restrict__ output, const float* __restrict__ tmp_out,
    const float* __restrict__ exp_sums, const int32_t* __restrict__ seq_lens,
    const int32_t* __restrict__ query_start_locs, int num_actual_tokens,
    int64_t out_stride_token, int64_t out_stride_head, int64_t out_stride_dim,
    int max_num_partitions, int partition_size) {
  using Policy = typename Layout::TilePolicy;
  static_assert(SpeculativeQueryLen > 0);

  constexpr int kWarpSize = 32;
  constexpr int kQueryHeadsPerKv = 4;
  constexpr int kVirtualRowsPerKv = SpeculativeQueryLen * kQueryHeadsPerKv;
  constexpr int kNumVirtualHeads = Layout::NumKvHeadsValue * kVirtualRowsPerKv;
  static_assert(Policy::HeadDimV == 128);
  static_assert(!UseRaggedSpeculativeQ4 || SpeculativeQueryLen == 4);

  extern __shared__ float shared_partition_weights[];
  float* const shared_sum = shared_partition_weights + max_num_partitions;
  const int seq_idx = blockIdx.y;
  const int head_idx = blockIdx.x;
  const int warp = threadIdx.x >> 5;
  const int lane = threadIdx.x & (kWarpSize - 1);
  const int dim = threadIdx.x;
  const int32_t seq_len = seq_lens[seq_idx];
  const int num_partitions =
      max(min(static_cast<int>(
                  (static_cast<int64_t>(seq_len) + partition_size - 1) /
                  partition_size),
              max_num_partitions),
          0);
  const int64_t stats_base =
      (static_cast<int64_t>(seq_idx) * kNumVirtualHeads + head_idx) *
      max_num_partitions;
  const int64_t tmp_base = stats_base * static_cast<int64_t>(Policy::HeadDimV);
  const int kv_head_idx = head_idx / kVirtualRowsPerKv;
  const int virtual_row = head_idx - kv_head_idx * kVirtualRowsPerKv;
  int query_start = seq_idx * SpeculativeQueryLen;
  int request_query_len = SpeculativeQueryLen;
  if constexpr (UseRaggedSpeculativeQ4) {
    query_start = max(min(query_start_locs[seq_idx], num_actual_tokens), 0);
    const int query_end =
        max(min(query_start_locs[seq_idx + 1], num_actual_tokens), query_start);
    request_query_len = min(query_end - query_start, SpeculativeQueryLen);
    if (virtual_row / kQueryHeadsPerKv >= request_query_len) {
      return;
    }
  }
  const int out_token_idx = query_start + virtual_row / kQueryHeadsPerKv;
  const int out_head_idx =
      kv_head_idx * kQueryHeadsPerKv + virtual_row % kQueryHeadsPerKv;
  const int64_t out_offset =
      static_cast<int64_t>(out_token_idx) * out_stride_token +
      static_cast<int64_t>(out_head_idx) * out_stride_head +
      dim * out_stride_dim;

  if (num_partitions <= 0) {
    output[out_offset] = byte_v2_float_to_bf16_bits(0.0f);
    return;
  }
  if (num_partitions == 1) {
    output[out_offset] = byte_v2_float_to_bf16_bits(tmp_out[tmp_base + dim]);
    return;
  }

  if (warp == 0) {
    float local_lse_max = -FLT_MAX;
#pragma unroll 1
    for (int idx = lane; idx < num_partitions; idx += kWarpSize) {
      local_lse_max = fmaxf(local_lse_max, exp_sums[stats_base + idx]);
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      local_lse_max = fmaxf(
          local_lse_max, __shfl_down_sync(0xffffffff, local_lse_max, offset));
    }
    const float global_lse_max = __shfl_sync(0xffffffff, local_lse_max, 0);

    float local_sum = 0.0f;
#pragma unroll 1
    for (int idx = lane; idx < num_partitions; idx += kWarpSize) {
      const float partition_lse = exp_sums[stats_base + idx];
      const float weight = partition_lse == -FLT_MAX
                               ? 0.0f
                               : __expf(partition_lse - global_lse_max);
      shared_partition_weights[idx] = weight;
      local_sum += weight;
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }
    if (lane == 0) {
      *shared_sum = local_sum;
    }
  }
  __syncthreads();

  float acc = 0.0f;
#pragma unroll 1
  for (int idx = 0; idx < num_partitions; ++idx) {
    acc += tmp_out[tmp_base + idx * Policy::HeadDimV + dim] *
           shared_partition_weights[idx];
  }
  const float out_value = *shared_sum > 0.0f ? acc / *shared_sum : 0.0f;
  output[out_offset] = byte_v2_float_to_bf16_bits(out_value);
}

template <int ComputeBlockN, int NumThreads>
__global__ void byte_v2_prefill_attention_kernel(
    uint16_t* __restrict__ output, const uint16_t* __restrict__ query,
    const uint16_t* __restrict__ key, const uint16_t* __restrict__ value,
    const int32_t* __restrict__ query_start_loc, int64_t num_tokens,
    int64_t num_reqs, int64_t max_query_len, int64_t num_heads,
    int64_t num_kv_heads, bool causal, float scale, int64_t q_stride_token,
    int64_t q_stride_head, int64_t q_stride_dim, int64_t k_stride_token,
    int64_t k_stride_head, int64_t k_stride_dim, int64_t v_stride_token,
    int64_t v_stride_head, int64_t v_stride_dim, int64_t out_stride_token,
    int64_t out_stride_head, int64_t out_stride_dim) {
  constexpr int kHeadDim = ByteV2DefaultPolicy::HeadDim;
  static_assert(NumThreads == kHeadDim);

  const int64_t head_idx = blockIdx.x;
  const int64_t req_query_idx = blockIdx.y;
  const int64_t req_idx = req_query_idx / max_query_len;
  const int64_t query_offset_in_req = req_query_idx - req_idx * max_query_len;
  const int dim = threadIdx.x;
  if (req_idx >= num_reqs) {
    return;
  }

  const int64_t q_per_kv = num_heads / num_kv_heads;
  const int64_t kv_head_idx = head_idx / q_per_kv;

  int32_t seq_start = query_start_loc[req_idx];
  int32_t seq_end = query_start_loc[req_idx + 1];
  if (seq_start < 0) {
    seq_start = 0;
  }
  if (seq_end < seq_start) {
    seq_end = seq_start;
  }
  if (seq_start > num_tokens) {
    return;
  }
  if (seq_end > num_tokens) {
    seq_end = static_cast<int32_t>(num_tokens);
  }
  const int32_t q_len = seq_end - seq_start;
  if (query_offset_in_req >= q_len) {
    return;
  }

  const int64_t token_idx = seq_start + query_offset_in_req;

  int32_t kv_end = causal ? static_cast<int32_t>(token_idx + 1) : seq_end;
  if (kv_end > seq_end) {
    kv_end = seq_end;
  }

  const int64_t q_offset = token_idx * q_stride_token +
                           head_idx * q_stride_head + dim * q_stride_dim;
  const float q_value = byte_v2_bf16_bits_to_float(query[q_offset]);

  float acc = 0.0f;
  float softmax_m = -FLT_MAX;
  float softmax_l = 0.0f;

  __shared__ float reduce_smem[NumThreads / 32];
  __shared__ float shared_scores[ComputeBlockN];
  __shared__ float shared_probs[ComputeBlockN];
  __shared__ float shared_new_m;
  __shared__ float shared_new_l;
  __shared__ float shared_alpha;

  for (int32_t tile_start = seq_start; tile_start < kv_end;
       tile_start += ComputeBlockN) {
    const int32_t tile_end =
        min(tile_start + ComputeBlockN, static_cast<int32_t>(kv_end));
    const int32_t tile_len = tile_end - tile_start;

    for (int32_t tile_offset = 0; tile_offset < tile_len; ++tile_offset) {
      const int32_t kv_token_idx = tile_start + tile_offset;
      const int64_t k_offset =
          static_cast<int64_t>(kv_token_idx) * k_stride_token +
          kv_head_idx * k_stride_head + dim * k_stride_dim;
      const float k_value = byte_v2_bf16_bits_to_float(key[k_offset]);
      const float qk_partial = q_value * k_value;
      const float qk_sum =
          byte_v2_block_sum<NumThreads>(qk_partial, reduce_smem);

      if (threadIdx.x == 0) {
        shared_scores[tile_offset] = qk_sum * scale;
      }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
      float tile_m = -FLT_MAX;
      for (int32_t tile_offset = 0; tile_offset < tile_len; ++tile_offset) {
        tile_m = fmaxf(tile_m, shared_scores[tile_offset]);
      }
      const float new_m = fmaxf(softmax_m, tile_m);
      const float alpha = __expf(softmax_m - new_m);
      float tile_l = 0.0f;
      for (int32_t tile_offset = 0; tile_offset < tile_len; ++tile_offset) {
        const float prob = __expf(shared_scores[tile_offset] - new_m);
        shared_probs[tile_offset] = prob;
        tile_l += prob;
      }
      shared_alpha = alpha;
      shared_new_m = new_m;
      shared_new_l = softmax_l * alpha + tile_l;
    }
    __syncthreads();

    float pv = 0.0f;
    for (int32_t tile_offset = 0; tile_offset < tile_len; ++tile_offset) {
      const int32_t kv_token_idx = tile_start + tile_offset;
      const int64_t v_offset =
          static_cast<int64_t>(kv_token_idx) * v_stride_token +
          kv_head_idx * v_stride_head + dim * v_stride_dim;
      const float v_value = byte_v2_bf16_bits_to_float(value[v_offset]);
      pv += shared_probs[tile_offset] * v_value;
    }
    acc = acc * shared_alpha + pv;
    softmax_l = shared_new_l;
    softmax_m = shared_new_m;
    __syncthreads();
  }

  const float out_value = softmax_l > 0.0f ? acc / softmax_l : 0.0f;
  const int64_t out_offset = token_idx * out_stride_token +
                             head_idx * out_stride_head + dim * out_stride_dim;
  output[out_offset] = byte_v2_float_to_bf16_bits(out_value);
}

__global__ void byte_v2_reshape_and_cache_kernel(
    const uint16_t* __restrict__ key, const uint16_t* __restrict__ value,
    uint8_t* __restrict__ kv_cache, const int64_t* __restrict__ slot_mapping,
    int64_t num_tokens, int64_t num_heads, int64_t key_stride_token,
    int64_t key_stride_head, int64_t key_stride_dim, int64_t value_stride_token,
    int64_t value_stride_head, int64_t value_stride_dim,
    int64_t kv_cache_stride_block) {
  constexpr int kCodecDimBlock = ByteV2DefaultPolicy::CodecDimBlock;
  constexpr int kCodecTileElems = ByteV2DefaultPolicy::CodecTileElems;
  constexpr int kPairsPerDimTile = kCodecDimBlock / 2;
  constexpr int kKDimTiles = ByteV2DefaultPolicy::KDimTiles;
  constexpr int kVDimTiles = ByteV2DefaultPolicy::VDimTiles;
  constexpr int kBlockSize = ByteV2DefaultPolicy::AllocBlockTokens;
  constexpr int kMetadataBytes = ByteV2DefaultLayout::AlignedMetadataBytes;
  constexpr int kPayloadBytesPerTile =
      ByteV2DefaultLayout::CodecPayloadBytesPerTile;
  constexpr int kCodeBase = kCodecTileElems;

  const int64_t work_items =
      num_tokens * 2 * num_heads * kKDimTiles * kPairsPerDimTile;
  for (int64_t work_idx = blockIdx.x * blockDim.x + threadIdx.x;
       work_idx < work_items;
       work_idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    int64_t tmp = work_idx;
    const int pair_in_dim_tile = tmp % kPairsPerDimTile;
    tmp /= kPairsPerDimTile;
    const int dim_tile = tmp % kKDimTiles;
    tmp /= kKDimTiles;
    const int head_idx = tmp % num_heads;
    tmp /= num_heads;
    const int kv_side = tmp & 1;
    const int64_t token_idx = tmp >> 1;

    const int64_t slot_idx = slot_mapping[token_idx];
    if (slot_idx < 0) {
      continue;
    }

    const int64_t physical_block = slot_idx / kBlockSize;
    const int token_offset = static_cast<int>(slot_idx % kBlockSize);
    const int token_tile = token_offset / ByteV2DefaultPolicy::CodecTokenBlock;
    const int row_in_tile = token_offset % ByteV2DefaultPolicy::CodecTokenBlock;
    uint8_t* __restrict__ page =
        kv_cache + physical_block * kv_cache_stride_block;

    if (kv_side == 0 && head_idx == 0 && dim_tile == 0 &&
        pair_in_dim_tile == 0) {
      for (int i = 0; i < kMetadataBytes; ++i) {
        page[i] = 0;
      }
    }

    const int dim0 = dim_tile * kCodecDimBlock + pair_in_dim_tile * 2;
    const int dim1 = dim0 + 1;
    const uint16_t* __restrict__ src = kv_side == 0 ? key : value;
    const int64_t stride_token =
        kv_side == 0 ? key_stride_token : value_stride_token;
    const int64_t stride_head =
        kv_side == 0 ? key_stride_head : value_stride_head;
    const int64_t stride_dim = kv_side == 0 ? key_stride_dim : value_stride_dim;
    const int64_t src_base = token_idx * stride_token + head_idx * stride_head;
    const uint16_t bits0 = src[src_base + dim0 * stride_dim];
    const uint16_t bits1 = src[src_base + dim1 * stride_dim];

    int64_t tile_offset;
    if (kv_side == 0) {
      tile_offset =
          ByteV2DefaultLayout::KPayloadBaseBytes +
          head_idx * ByteV2DefaultLayout::AlignedKPayloadBytesPerKvHead +
          (dim_tile * ByteV2DefaultPolicy::CodecTokenTilesPerAllocBlock +
           token_tile) *
              kPayloadBytesPerTile;
    } else {
      tile_offset =
          ByteV2DefaultLayout::VPayloadBaseBytes +
          head_idx * ByteV2DefaultLayout::AlignedVPayloadBytesPerKvHead +
          (token_tile * kVDimTiles + dim_tile) * kPayloadBytesPerTile;
    }

    const int elem_base = row_in_tile * kCodecDimBlock + pair_in_dim_tile * 2;
    page[tile_offset + elem_base] = static_cast<uint8_t>(bits0 & 0xff);
    page[tile_offset + elem_base + 1] = static_cast<uint8_t>(bits1 & 0xff);
    page[tile_offset + kCodeBase + elem_base / 2] =
        byte_v2_code_nibble(bits0) |
        static_cast<uint8_t>(byte_v2_code_nibble(bits1) << 4);

    if constexpr (ByteV2DefaultLayout::IncludeRawPayloadValue) {
      const int64_t raw_offset0 = kv_side == 0
                                      ? ByteV2DefaultLayout::raw_key_offset(
                                            head_idx, token_offset, dim0)
                                      : ByteV2DefaultLayout::raw_value_offset(
                                            head_idx, token_offset, dim0);
      const int64_t raw_offset1 = kv_side == 0
                                      ? ByteV2DefaultLayout::raw_key_offset(
                                            head_idx, token_offset, dim1)
                                      : ByteV2DefaultLayout::raw_value_offset(
                                            head_idx, token_offset, dim1);
      byte_v2_store_u16_bytes(page, raw_offset0, bits0);
      byte_v2_store_u16_bytes(page, raw_offset1, bits1);
    }
  }
}

__global__ void byte_v2_clear_page_metadata_from_slots_kernel(
    uint8_t* __restrict__ kv_cache, const int64_t* __restrict__ slot_mapping,
    int64_t num_tokens, int64_t kv_cache_stride_block) {
  constexpr int kBlockSize = ByteV2DefaultPolicy::AllocBlockTokens;
  constexpr int kMetadataBytes = ByteV2DefaultLayout::AlignedMetadataBytes;

  const int64_t work_items = num_tokens * kMetadataBytes;
  for (int64_t work_idx = blockIdx.x * blockDim.x + threadIdx.x;
       work_idx < work_items;
       work_idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int64_t token_idx = work_idx / kMetadataBytes;
    const int metadata_byte = static_cast<int>(work_idx % kMetadataBytes);
    const int64_t slot_idx = slot_mapping[token_idx];
    if (slot_idx < 0) {
      continue;
    }
    const int64_t physical_block = slot_idx / kBlockSize;
    kv_cache[physical_block * kv_cache_stride_block + metadata_byte] = 0;
  }
}

__global__ void byte_v2_clear_page_metadata_from_staging_kernel(
    uint8_t* __restrict__ kv_cache,
    const int32_t* __restrict__ staging_to_physical_block,
    int64_t num_staging_slots, int64_t kv_cache_stride_block) {
  constexpr int kMetadataBytes = ByteV2DefaultLayout::AlignedMetadataBytes;

  const int64_t work_items = num_staging_slots * kMetadataBytes;
  for (int64_t work_idx = blockIdx.x * blockDim.x + threadIdx.x;
       work_idx < work_items;
       work_idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int64_t staging_slot = work_idx / kMetadataBytes;
    const int metadata_byte = static_cast<int>(work_idx % kMetadataBytes);
    const int32_t physical_block = staging_to_physical_block[staging_slot];
    if (physical_block < 0) {
      continue;
    }
    kv_cache[static_cast<int64_t>(physical_block) * kv_cache_stride_block +
             metadata_byte] = 0;
  }
}

__global__ void byte_v2_prepare_single_token_page_kernel(
    uint8_t* __restrict__ kv_cache, const int64_t* __restrict__ slot_mapping,
    int32_t* __restrict__ page_unsafe_flags, int64_t kv_cache_blocks,
    int64_t kv_cache_stride_block) {
  constexpr int kBlockSize = ByteV2DefaultPolicy::AllocBlockTokens;
  constexpr int kMetadataBytes = ByteV2DefaultLayout::AlignedMetadataBytes;
  constexpr int kTilesPerSide = ByteV2DefaultPolicy::KDimTiles;
  constexpr int kTilesPerHead = 2 * kTilesPerSide;
  constexpr int kTotalTiles =
      ByteV2DefaultLayout::NumKvHeadsValue * kTilesPerHead;

  const int64_t slot_idx = slot_mapping[0];
  if (slot_idx < 0) {
    return;
  }
  const int64_t physical_block = slot_idx / kBlockSize;
  if (physical_block < 0 || physical_block >= kv_cache_blocks) {
    return;
  }
  uint8_t* __restrict__ page =
      kv_cache + physical_block * kv_cache_stride_block;
  if (slot_idx % kBlockSize == 0) {
    for (int metadata_byte = threadIdx.x; metadata_byte < kMetadataBytes;
         metadata_byte += blockDim.x) {
      page[metadata_byte] = 0;
    }
    if (threadIdx.x == 0 && page_unsafe_flags != nullptr) {
      page_unsafe_flags[physical_block] = 0;
    }
    return;
  }

  if constexpr (ByteV2DefaultLayout::PagePooledOutliersValue) {
    __shared__ uint16_t
        shared_entries[ByteV2DefaultLayout::OutlierPoolEntriesValue];
    const unsigned int old_pool_used = *reinterpret_cast<unsigned int*>(
        page + ByteV2DefaultLayout::OutlierPoolUsedOffset);
    if (old_pool_used > ByteV2DefaultLayout::OutlierPoolEntriesValue) {
      __trap();
    }
    const uint16_t* __restrict__ old_entries =
        reinterpret_cast<const uint16_t*>(
            page + ByteV2DefaultLayout::OutlierPoolBaseBytes);
    for (int entry_idx = threadIdx.x; entry_idx < old_pool_used;
         entry_idx += blockDim.x) {
      shared_entries[entry_idx] = old_entries[entry_idx];
    }
    __syncthreads();

    if (threadIdx.x == 0) {
      int new_pool_used = 0;
      for (int linear_tile = 0; linear_tile < kTotalTiles; ++linear_tile) {
        const int kv_head = linear_tile / kTilesPerHead;
        const int tile_in_head = linear_tile % kTilesPerHead;
        const int kv_side = tile_in_head / kTilesPerSide;
        const int dim_tile = tile_in_head % kTilesPerSide;
        const int count = kv_side == 0 ? ByteV2DefaultLayout::k_outlier_count(
                                             page, kv_head, dim_tile, 0)
                                       : ByteV2DefaultLayout::v_outlier_count(
                                             page, kv_head, dim_tile, 0);
        if (count == 0) {
          continue;
        }
        const int old_pool_index =
            kv_side == 0 ? ByteV2DefaultLayout::k_outlier_pool_index(
                               page, kv_head, dim_tile, 0)
                         : ByteV2DefaultLayout::v_outlier_pool_index(
                               page, kv_head, dim_tile, 0);
        const int capacity = byte_v2_outlier_segment_capacity(count);
        if (new_pool_used + capacity >
            ByteV2DefaultLayout::OutlierPoolEntriesValue) {
          __trap();
        }
        for (int entry_idx = 0; entry_idx < count; ++entry_idx) {
          byte_v2_store_u16_bytes(
              page,
              ByteV2DefaultLayout::OutlierPoolBaseBytes +
                  (new_pool_used + entry_idx) *
                      ByteV2DefaultLayout::OutlierEntryBytes,
              shared_entries[old_pool_index + entry_idx]);
        }
        if (kv_side == 0) {
          ByteV2DefaultLayout::set_k_outlier_pool_index(page, kv_head, dim_tile,
                                                        0, new_pool_used);
        } else {
          ByteV2DefaultLayout::set_v_outlier_pool_index(page, kv_head, dim_tile,
                                                        0, new_pool_used);
        }
        new_pool_used += capacity;
      }
      *reinterpret_cast<unsigned int*>(
          page + ByteV2DefaultLayout::OutlierPoolUsedOffset) =
          static_cast<unsigned int>(new_pool_used);
    }
  }
}

template <bool WriteOutlierHighSideband>
__global__ void byte_v2_reshape_and_cache_block_direct_kernel(
    const uint16_t* __restrict__ key, const uint16_t* __restrict__ value,
    uint8_t* __restrict__ kv_cache, const int64_t* __restrict__ slot_mapping,
    int64_t num_tokens, int64_t num_heads, int64_t key_stride_token,
    int64_t key_stride_head, int64_t key_stride_dim, int64_t value_stride_token,
    int64_t value_stride_head, int64_t value_stride_dim,
    int64_t kv_cache_stride_block) {
  constexpr int kCodecTokenBlock = ByteV2DefaultPolicy::CodecTokenBlock;
  constexpr int kCodecDimBlock = ByteV2DefaultPolicy::CodecDimBlock;
  constexpr int kCodecTileElems = ByteV2DefaultPolicy::CodecTileElems;
  constexpr int kPairsPerRow = kCodecDimBlock / 2;
  constexpr int kPairsPerCodecTile = kCodecTileElems / 2;
  constexpr int kKDimTiles = ByteV2DefaultPolicy::KDimTiles;
  constexpr int kVDimTiles = ByteV2DefaultPolicy::VDimTiles;
  constexpr int kBlockSize = ByteV2DefaultPolicy::AllocBlockTokens;
  constexpr int kPayloadBytesPerTile =
      ByteV2DefaultLayout::CodecPayloadBytesPerTile;
  constexpr int kCodeBase = kCodecTileElems;

  const int64_t source_chunk_start =
      static_cast<int64_t>(blockIdx.x) * kCodecTokenBlock;
  const int64_t source_chunk_end =
      min(source_chunk_start + kCodecTokenBlock, num_tokens);
  if (source_chunk_start >= source_chunk_end) {
    return;
  }

  int64_t tile_group = blockIdx.y;
  const int dim_tile = tile_group % kKDimTiles;
  tile_group /= kKDimTiles;
  const int head_idx = tile_group % num_heads;
  tile_group /= num_heads;
  const int kv_side = tile_group;

  __shared__ int shared_base;
  __shared__ int shared_fallback;

  for (int64_t page_token_start = source_chunk_start;
       page_token_start < source_chunk_end; ++page_token_start) {
    const int64_t first_slot_idx = slot_mapping[page_token_start];
    if (first_slot_idx < 0) {
      continue;
    }
    const int64_t physical_block = first_slot_idx / kBlockSize;
    if (page_token_start > 0) {
      const int64_t previous_slot_idx = slot_mapping[page_token_start - 1];
      if (previous_slot_idx >= 0 &&
          previous_slot_idx / kBlockSize == physical_block) {
        continue;
      }
    }

    int page_token_count = 1;
    while (page_token_count < kBlockSize &&
           page_token_start + page_token_count < num_tokens) {
      const int64_t next_slot_idx =
          slot_mapping[page_token_start + page_token_count];
      if (next_slot_idx < 0 || next_slot_idx / kBlockSize != physical_block) {
        break;
      }
      ++page_token_count;
    }

    uint8_t* __restrict__ page =
        kv_cache + physical_block * kv_cache_stride_block;
    if (threadIdx.x == 0) {
      int high_counts[128];
      for (int i = 0; i < 128; ++i) {
        high_counts[i] = 0;
      }
      const uint16_t* __restrict__ src = kv_side == 0 ? key : value;
      const int64_t stride_dim =
          kv_side == 0 ? key_stride_dim : value_stride_dim;
      for (int source_offset = 0; source_offset < page_token_count;
           ++source_offset) {
        const int64_t token_idx = page_token_start + source_offset;
        const int64_t row_src_base =
            token_idx * (kv_side == 0 ? key_stride_token : value_stride_token) +
            head_idx * (kv_side == 0 ? key_stride_head : value_stride_head);
        for (int dim_offset = 0; dim_offset < kCodecDimBlock; ++dim_offset) {
          const int dim = dim_tile * kCodecDimBlock + dim_offset;
          const uint16_t bits = src[row_src_base + dim * stride_dim];
          ++high_counts[byte_v2_high7(bits)];
        }
      }

      int best_base = 0;
      int best_count = -1;
      for (int base = 0; base <= 120; ++base) {
        int window_count = 0;
        for (int delta = 0; delta < 8; ++delta) {
          window_count += high_counts[base + delta];
        }
        if (window_count > best_count) {
          best_count = window_count;
          best_base = base;
        }
      }
      const int elem_count = page_token_count * kCodecDimBlock;
      const int outlier_count = elem_count - best_count;
      const int fallback =
          outlier_count > ByteV2DefaultLayout::OutlierEntriesPerTileValue;
      const int has_overlay = outlier_count > 0 && !fallback;
      const int token_tile = 0;
      int outlier_pool_index = 0;
      if (has_overlay) {
        const int allocation_entries =
            WriteOutlierHighSideband ? kCodecTileElems : outlier_count;
        outlier_pool_index =
            byte_v2_allocate_outlier_segment<ByteV2DefaultLayout>(
                page, allocation_entries);
      }
      const int tile_idx =
          kv_side == 0
              ? dim_tile * ByteV2DefaultPolicy::CodecTokenTilesPerAllocBlock +
                    token_tile
              : token_tile * kVDimTiles + dim_tile;
      if (kv_side == 0) {
        page[ByteV2DefaultLayout::k_base_offset(
            head_idx, dim_tile, token_tile)] = static_cast<uint8_t>(best_base);
        if (fallback) {
          atomicOr(
              reinterpret_cast<unsigned int*>(
                  page + ByteV2DefaultLayout::k_fallback_mask_offset(head_idx)),
              1u << tile_idx);
        } else if (has_overlay) {
          atomicOr(
              reinterpret_cast<unsigned int*>(
                  page + ByteV2DefaultLayout::k_outlier_mask_offset(head_idx)),
              1u << tile_idx);
          byte_v2_set_outlier_descriptor<ByteV2DefaultLayout>(
              page, kv_side, head_idx, dim_tile, token_tile, outlier_count,
              outlier_pool_index);
        }
      } else {
        page[ByteV2DefaultLayout::v_base_offset(
            head_idx, dim_tile, token_tile)] = static_cast<uint8_t>(best_base);
        if (fallback) {
          atomicOr(
              reinterpret_cast<unsigned int*>(
                  page + ByteV2DefaultLayout::v_fallback_mask_offset(head_idx)),
              1u << tile_idx);
        } else if (has_overlay) {
          atomicOr(
              reinterpret_cast<unsigned int*>(
                  page + ByteV2DefaultLayout::v_outlier_mask_offset(head_idx)),
              1u << tile_idx);
          byte_v2_set_outlier_descriptor<ByteV2DefaultLayout>(
              page, kv_side, head_idx, dim_tile, token_tile, outlier_count,
              outlier_pool_index);
        }
      }
      if (has_overlay) {
        int overlay_entry_idx = 0;
        for (int source_offset = 0; source_offset < page_token_count;
             ++source_offset) {
          const int64_t token_idx = page_token_start + source_offset;
          const int64_t slot_idx = slot_mapping[token_idx];
          const int row_in_tile = static_cast<int>(slot_idx % kBlockSize);
          const int64_t row_src_base =
              token_idx *
                  (kv_side == 0 ? key_stride_token : value_stride_token) +
              head_idx * (kv_side == 0 ? key_stride_head : value_stride_head);
          for (int dim_offset = 0; dim_offset < kCodecDimBlock; ++dim_offset) {
            const int dim = dim_tile * kCodecDimBlock + dim_offset;
            const uint16_t bits = src[row_src_base + dim * stride_dim];
            const int high = static_cast<int>(bits >> 8);
            const int elem_idx = row_in_tile * kCodecDimBlock + dim_offset;
            if constexpr (WriteOutlierHighSideband) {
              const int high_offset =
                  kv_side == 0 ? ByteV2DefaultLayout::k_outlier_payload_offset(
                                     page, head_idx, dim_tile, token_tile)
                               : ByteV2DefaultLayout::v_outlier_payload_offset(
                                     page, head_idx, dim_tile, token_tile);
              page[high_offset + elem_idx] = static_cast<uint8_t>(high);
            } else {
              if (byte_v2_high7_in_window(bits, best_base)) {
                continue;
              }
              const uint16_t entry = static_cast<uint16_t>(
                  ByteV2DefaultLayout::OutlierEntryPolicy::encode(elem_idx,
                                                                  high));
              const int entry_offset =
                  kv_side == 0 ? ByteV2DefaultLayout::k_outlier_payload_offset(
                                     page, head_idx, dim_tile, token_tile,
                                     overlay_entry_idx)
                               : ByteV2DefaultLayout::v_outlier_payload_offset(
                                     page, head_idx, dim_tile, token_tile,
                                     overlay_entry_idx);
              byte_v2_store_u16_bytes(page, entry_offset, entry);
              ++overlay_entry_idx;
            }
          }
        }
      }
      shared_base = best_base;
      shared_fallback = fallback;
    }
    __syncthreads();

    const int pair_idx = threadIdx.x;
    const int source_offset = pair_idx / kPairsPerRow;
    if (pair_idx < kPairsPerCodecTile && source_offset < page_token_count) {
      const int64_t token_idx = page_token_start + source_offset;
      const int64_t slot_idx = slot_mapping[token_idx];
      const int pair_in_dim_tile = pair_idx % kPairsPerRow;
      const int token_offset = static_cast<int>(slot_idx % kBlockSize);
      const int token_tile = token_offset / kCodecTokenBlock;
      const int row_in_tile = token_offset % kCodecTokenBlock;

      const int dim0 = dim_tile * kCodecDimBlock + pair_in_dim_tile * 2;
      const int dim1 = dim0 + 1;
      const uint16_t* __restrict__ src = kv_side == 0 ? key : value;
      const int64_t stride_token =
          kv_side == 0 ? key_stride_token : value_stride_token;
      const int64_t stride_head =
          kv_side == 0 ? key_stride_head : value_stride_head;
      const int64_t stride_dim =
          kv_side == 0 ? key_stride_dim : value_stride_dim;
      const int64_t src_base =
          token_idx * stride_token + head_idx * stride_head;
      const uint16_t bits0 = src[src_base + dim0 * stride_dim];
      const uint16_t bits1 = src[src_base + dim1 * stride_dim];

      int64_t tile_offset;
      if (kv_side == 0) {
        tile_offset =
            ByteV2DefaultLayout::KPayloadBaseBytes +
            head_idx * ByteV2DefaultLayout::AlignedKPayloadBytesPerKvHead +
            (dim_tile * ByteV2DefaultPolicy::CodecTokenTilesPerAllocBlock +
             token_tile) *
                kPayloadBytesPerTile;
      } else {
        tile_offset =
            ByteV2DefaultLayout::VPayloadBaseBytes +
            head_idx * ByteV2DefaultLayout::AlignedVPayloadBytesPerKvHead +
            (token_tile * kVDimTiles + dim_tile) * kPayloadBytesPerTile;
      }

      const int elem_base = row_in_tile * kCodecDimBlock + pair_in_dim_tile * 2;
      page[tile_offset + elem_base] = static_cast<uint8_t>(bits0 & 0xff);
      page[tile_offset + elem_base + 1] = static_cast<uint8_t>(bits1 & 0xff);
      const uint8_t code0 = shared_fallback
                                ? byte_v2_code_nibble(bits0)
                                : byte_v2_delta_code_nibble(bits0, shared_base);
      const uint8_t code1 = shared_fallback
                                ? byte_v2_code_nibble(bits1)
                                : byte_v2_delta_code_nibble(bits1, shared_base);
      page[tile_offset + kCodeBase + elem_base / 2] =
          (code0 & 0x0f) | static_cast<uint8_t>((code1 & 0x0f) << 4);

      if constexpr (ByteV2DefaultLayout::IncludeRawPayloadValue) {
        const int64_t raw_offset0 = kv_side == 0
                                        ? ByteV2DefaultLayout::raw_key_offset(
                                              head_idx, token_offset, dim0)
                                        : ByteV2DefaultLayout::raw_value_offset(
                                              head_idx, token_offset, dim0);
        const int64_t raw_offset1 = kv_side == 0
                                        ? ByteV2DefaultLayout::raw_key_offset(
                                              head_idx, token_offset, dim1)
                                        : ByteV2DefaultLayout::raw_value_offset(
                                              head_idx, token_offset, dim1);
        byte_v2_store_u16_bytes(page, raw_offset0, bits0);
        byte_v2_store_u16_bytes(page, raw_offset1, bits1);
      }
    }
    __syncthreads();
  }
}

__global__ void byte_v2_reshape_and_cache_high_byte_kernel(
    const uint16_t* __restrict__ key, const uint16_t* __restrict__ value,
    uint8_t* __restrict__ kv_cache, const int64_t* __restrict__ slot_mapping,
    int64_t num_tokens, int64_t num_heads, int64_t key_stride_token,
    int64_t key_stride_head, int64_t key_stride_dim, int64_t value_stride_token,
    int64_t value_stride_head, int64_t value_stride_dim,
    int64_t kv_cache_stride_block) {
  constexpr int kCodecDimBlock = ByteV2DefaultPolicy::CodecDimBlock;
  constexpr int kCodecTileElems = ByteV2DefaultPolicy::CodecTileElems;
  constexpr int kPairsPerDimTile = kCodecDimBlock / 2;
  constexpr int kKDimTiles = ByteV2DefaultPolicy::KDimTiles;
  constexpr int kVDimTiles = ByteV2DefaultPolicy::VDimTiles;
  constexpr int kBlockSize = ByteV2DefaultPolicy::AllocBlockTokens;
  constexpr int kPayloadBytesPerTile =
      ByteV2HighByteLayout::CodecPayloadBytesPerTile;
  constexpr int kHighBase = kCodecTileElems;

  const int64_t work_items =
      num_tokens * 2 * num_heads * kKDimTiles * kPairsPerDimTile;
  for (int64_t work_idx = blockIdx.x * blockDim.x + threadIdx.x;
       work_idx < work_items;
       work_idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    int64_t tmp = work_idx;
    const int pair_in_dim_tile = tmp % kPairsPerDimTile;
    tmp /= kPairsPerDimTile;
    const int dim_tile = tmp % kKDimTiles;
    tmp /= kKDimTiles;
    const int head_idx = tmp % num_heads;
    tmp /= num_heads;
    const int kv_side = tmp & 1;
    const int64_t token_idx = tmp >> 1;

    const int64_t slot_idx = slot_mapping[token_idx];
    if (slot_idx < 0) {
      continue;
    }

    const int64_t physical_block = slot_idx / kBlockSize;
    const int token_offset = static_cast<int>(slot_idx % kBlockSize);
    const int token_tile = token_offset / ByteV2DefaultPolicy::CodecTokenBlock;
    const int row_in_tile = token_offset % ByteV2DefaultPolicy::CodecTokenBlock;
    uint8_t* __restrict__ page =
        kv_cache + physical_block * kv_cache_stride_block;

    const int dim0 = dim_tile * kCodecDimBlock + pair_in_dim_tile * 2;
    const int dim1 = dim0 + 1;
    const uint16_t* __restrict__ src = kv_side == 0 ? key : value;
    const int64_t stride_token =
        kv_side == 0 ? key_stride_token : value_stride_token;
    const int64_t stride_head =
        kv_side == 0 ? key_stride_head : value_stride_head;
    const int64_t stride_dim = kv_side == 0 ? key_stride_dim : value_stride_dim;
    const int64_t src_base = token_idx * stride_token + head_idx * stride_head;
    const uint16_t bits0 = src[src_base + dim0 * stride_dim];
    const uint16_t bits1 = src[src_base + dim1 * stride_dim];

    int64_t tile_offset;
    if (kv_side == 0) {
      tile_offset =
          ByteV2HighByteLayout::KPayloadBaseBytes +
          head_idx * ByteV2HighByteLayout::AlignedKPayloadBytesPerKvHead +
          (dim_tile * ByteV2DefaultPolicy::CodecTokenTilesPerAllocBlock +
           token_tile) *
              kPayloadBytesPerTile;
    } else {
      tile_offset =
          ByteV2HighByteLayout::VPayloadBaseBytes +
          head_idx * ByteV2HighByteLayout::AlignedVPayloadBytesPerKvHead +
          (token_tile * kVDimTiles + dim_tile) * kPayloadBytesPerTile;
    }

    const int elem_base = row_in_tile * kCodecDimBlock + pair_in_dim_tile * 2;
    page[tile_offset + elem_base] = static_cast<uint8_t>(bits0 & 0xff);
    page[tile_offset + elem_base + 1] = static_cast<uint8_t>(bits1 & 0xff);
    page[tile_offset + kHighBase + elem_base] =
        static_cast<uint8_t>(bits0 >> 8);
    page[tile_offset + kHighBase + elem_base + 1] =
        static_cast<uint8_t>(bits1 >> 8);
  }
}

__global__ void byte_v2_append_raw_staging_kernel(
    const uint16_t* __restrict__ key, const uint16_t* __restrict__ value,
    uint8_t* __restrict__ raw_staging, const int64_t* __restrict__ slot_mapping,
    const int32_t* __restrict__ block_to_staging_slot, int64_t num_tokens,
    int64_t num_heads, int64_t key_stride_token, int64_t key_stride_head,
    int64_t key_stride_dim, int64_t value_stride_token,
    int64_t value_stride_head, int64_t value_stride_dim,
    int64_t raw_staging_stride_slot, int64_t raw_staging_slots,
    int64_t block_to_staging_size) {
  constexpr int kBlockSize = ByteV2DefaultPolicy::AllocBlockTokens;
  constexpr int kHeadDim = ByteV2DefaultPolicy::HeadDim;
  constexpr int kHeadDimV = ByteV2DefaultPolicy::HeadDimV;

  const int64_t work_items = num_tokens * 2 * num_heads * kHeadDim;
  for (int64_t work_idx = blockIdx.x * blockDim.x + threadIdx.x;
       work_idx < work_items;
       work_idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    int64_t tmp = work_idx;
    const int dim = tmp % kHeadDim;
    tmp /= kHeadDim;
    const int head_idx = tmp % num_heads;
    tmp /= num_heads;
    const int kv_side = tmp & 1;
    const int64_t token_idx = tmp >> 1;

    const int64_t slot_idx = slot_mapping[token_idx];
    if (slot_idx < 0) {
      continue;
    }
    const int64_t physical_block = slot_idx / kBlockSize;
    if (physical_block < 0 || physical_block >= block_to_staging_size) {
      continue;
    }
    const int32_t staging_slot = block_to_staging_slot[physical_block];
    if (staging_slot < 0 || staging_slot >= raw_staging_slots) {
      continue;
    }

    const int row = static_cast<int>(slot_idx % kBlockSize);
    const uint16_t* __restrict__ src = kv_side == 0 ? key : value;
    const int64_t stride_token =
        kv_side == 0 ? key_stride_token : value_stride_token;
    const int64_t stride_head =
        kv_side == 0 ? key_stride_head : value_stride_head;
    const int64_t stride_dim = kv_side == 0 ? key_stride_dim : value_stride_dim;
    const int64_t src_base = token_idx * stride_token + head_idx * stride_head;
    const uint16_t bits = src[src_base + dim * stride_dim];
    const int64_t elem_offset =
        kv_side == 0
            ? ByteV2DefaultRawStagingLayout::key_offset(head_idx, row, dim)
            : ByteV2DefaultRawStagingLayout::value_offset(head_idx, row,
                                                          dim % kHeadDimV);
    byte_v2_store_u16_bytes(raw_staging + static_cast<int64_t>(staging_slot) *
                                              raw_staging_stride_slot,
                            elem_offset, bits);
  }
}

__global__ void byte_v2_prepare_raw_staging_kernel(
    const int64_t* __restrict__ slot_mapping,
    int32_t* __restrict__ block_to_staging_slot,
    int32_t* __restrict__ staging_to_physical_block,
    int32_t* __restrict__ valid_rows, int32_t* __restrict__ next_staging_slot,
    int32_t* __restrict__ overflow, int64_t num_tokens,
    int64_t num_physical_blocks, int64_t num_staging_slots) {
  constexpr int kBlockSize = ByteV2DefaultPolicy::AllocBlockTokens;

  if (blockIdx.x != 0 || threadIdx.x != 0) {
    return;
  }

  for (int64_t token_idx = 0; token_idx < num_tokens; ++token_idx) {
    const int64_t slot_idx = slot_mapping[token_idx];
    if (slot_idx < 0) {
      continue;
    }

    const int64_t physical_block = slot_idx / kBlockSize;
    if (physical_block < 0 || physical_block >= num_physical_blocks) {
      *overflow = 1;
      continue;
    }
    const int row = static_cast<int>(slot_idx % kBlockSize);
    int32_t staging_slot = block_to_staging_slot[physical_block];
    if (staging_slot < 0) {
      staging_slot = next_staging_slot[0]++;
      if (staging_slot < 0 || staging_slot >= num_staging_slots) {
        *overflow = 1;
        continue;
      }
      block_to_staging_slot[physical_block] = staging_slot;
      staging_to_physical_block[staging_slot] =
          static_cast<int32_t>(physical_block);
      valid_rows[staging_slot] = 0;
    } else if (staging_slot >= num_staging_slots) {
      *overflow = 1;
      continue;
    }
    if (valid_rows[staging_slot] < row + 1) {
      valid_rows[staging_slot] = row + 1;
    }
  }
}

__global__ void byte_v2_hydrate_raw_staging_from_cache_kernel(
    uint8_t* __restrict__ raw_staging, const uint8_t* __restrict__ kv_cache,
    const int32_t* __restrict__ staging_to_physical_block,
    const int32_t* __restrict__ valid_rows, int64_t num_staging_slots,
    int64_t raw_staging_stride_slot, int64_t kv_cache_stride_block) {
  constexpr int kBlockSize = ByteV2DefaultPolicy::AllocBlockTokens;
  constexpr int kHeadDim = ByteV2DefaultPolicy::HeadDim;
  constexpr int kHeadDimV = ByteV2DefaultPolicy::HeadDimV;
  constexpr int kNumKvHeads = ByteV2DefaultLayout::NumKvHeadsValue;

  const int64_t work_items =
      num_staging_slots * 2 * kNumKvHeads * kBlockSize * kHeadDim;
  for (int64_t work_idx = blockIdx.x * blockDim.x + threadIdx.x;
       work_idx < work_items;
       work_idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    int64_t tmp = work_idx;
    const int dim = tmp % kHeadDim;
    tmp /= kHeadDim;
    const int row = tmp % kBlockSize;
    tmp /= kBlockSize;
    const int head_idx = tmp % kNumKvHeads;
    tmp /= kNumKvHeads;
    const int kv_side = tmp & 1;
    const int64_t staging_slot = tmp >> 1;

    const int32_t physical_block = staging_to_physical_block[staging_slot];
    if (physical_block < 0) {
      continue;
    }
    int rows = valid_rows[staging_slot];
    if (rows <= 0) {
      continue;
    }
    if (rows > kBlockSize) {
      rows = kBlockSize;
    }
    if (row >= rows) {
      continue;
    }
    if (kv_side != 0 && dim >= kHeadDimV) {
      continue;
    }

    const uint8_t* __restrict__ page =
        kv_cache + static_cast<int64_t>(physical_block) * kv_cache_stride_block;
    float value;
    int64_t staging_offset;
    if (kv_side == 0) {
      value = byte_v2_load_payload_elem<ByteV2DefaultLayout, false>(
          page,
          ByteV2DefaultLayout::KPayloadBaseBytes +
              head_idx * ByteV2DefaultLayout::AlignedKPayloadBytesPerKvHead,
          head_idx, row, dim);
      staging_offset =
          ByteV2DefaultRawStagingLayout::key_offset(head_idx, row, dim);
    } else {
      value = byte_v2_load_payload_elem<ByteV2DefaultLayout, true>(
          page,
          ByteV2DefaultLayout::VPayloadBaseBytes +
              head_idx * ByteV2DefaultLayout::AlignedVPayloadBytesPerKvHead,
          head_idx, row, dim);
      staging_offset =
          ByteV2DefaultRawStagingLayout::value_offset(head_idx, row, dim);
    }
    byte_v2_store_u16_bytes(
        raw_staging + staging_slot * raw_staging_stride_slot, staging_offset,
        byte_v2_float_to_bf16_bits(value));
  }
}

__global__ void byte_v2_release_raw_staging_kernel(
    int32_t* __restrict__ block_to_staging_slot,
    int32_t* __restrict__ staging_to_physical_block,
    int32_t* __restrict__ valid_rows, int32_t* __restrict__ next_staging_slot,
    int32_t* __restrict__ overflow, int64_t num_staging_slots,
    int64_t num_physical_blocks) {
  for (int64_t staging_slot = blockIdx.x * blockDim.x + threadIdx.x;
       staging_slot < num_staging_slots;
       staging_slot += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int32_t physical_block = staging_to_physical_block[staging_slot];
    if (physical_block >= 0 && physical_block < num_physical_blocks) {
      atomicCAS(block_to_staging_slot + physical_block,
                static_cast<int32_t>(staging_slot), -1);
    }
    staging_to_physical_block[staging_slot] = -1;
    valid_rows[staging_slot] = 0;
  }

  if (blockIdx.x == 0 && threadIdx.x == 0) {
    next_staging_slot[0] = 0;
    overflow[0] = 0;
  }
}

__global__ void byte_v2_release_raw_staging_and_update_flags_kernel(
    int32_t* __restrict__ block_to_staging_slot,
    int32_t* __restrict__ staging_to_physical_block,
    int32_t* __restrict__ valid_rows, int32_t* __restrict__ next_staging_slot,
    int32_t* __restrict__ overflow, int32_t* __restrict__ page_unsafe_flags,
    const uint8_t* __restrict__ kv_cache, int64_t num_staging_slots,
    int64_t num_physical_blocks, int64_t kv_cache_stride_block) {
  for (int64_t staging_slot = blockIdx.x * blockDim.x + threadIdx.x;
       staging_slot < num_staging_slots;
       staging_slot += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int32_t physical_block = staging_to_physical_block[staging_slot];
    if (physical_block >= 0 && physical_block < num_physical_blocks) {
      const uint8_t* __restrict__ page =
          kv_cache +
          static_cast<int64_t>(physical_block) * kv_cache_stride_block;
      page_unsafe_flags[physical_block] = byte_v2_page_unsafe_flag(page);
      atomicCAS(block_to_staging_slot + physical_block,
                static_cast<int32_t>(staging_slot), -1);
    }
    staging_to_physical_block[staging_slot] = -1;
    valid_rows[staging_slot] = 0;
  }

  if (blockIdx.x == 0 && threadIdx.x == 0) {
    next_staging_slot[0] = 0;
    overflow[0] = 0;
  }
}

template <bool FuseMetadataClear, bool BypassSerialMetadata,
          bool WarpParallelHistogram>
__global__ void byte_v2_commit_raw_staging_to_cache_kernel(
    const uint8_t* __restrict__ raw_staging, uint8_t* __restrict__ kv_cache,
    const int32_t* __restrict__ staging_to_physical_block,
    const int32_t* __restrict__ valid_rows, int64_t num_staging_slots,
    int64_t raw_staging_stride_slot, int64_t kv_cache_stride_block) {
  constexpr int kCodecDimBlock = ByteV2DefaultPolicy::CodecDimBlock;
  constexpr int kCodecTileElems = ByteV2DefaultPolicy::CodecTileElems;
  constexpr int kPairsPerRow = kCodecDimBlock / 2;
  constexpr int kPairsPerCodecTile = kCodecTileElems / 2;
  constexpr int kKDimTiles = ByteV2DefaultPolicy::KDimTiles;
  constexpr int kBlockSize = ByteV2DefaultPolicy::AllocBlockTokens;
  constexpr int kPayloadBytesPerTile =
      ByteV2DefaultLayout::CodecPayloadBytesPerTile;
  constexpr int kCodeBase = kCodecTileElems;
  static_assert(!(BypassSerialMetadata && WarpParallelHistogram));

  const int64_t staging_slot = blockIdx.x;
  if (staging_slot >= num_staging_slots) {
    return;
  }
  const int32_t physical_block = staging_to_physical_block[staging_slot];
  if (physical_block < 0) {
    return;
  }
  int rows = valid_rows[staging_slot];
  if (rows <= 0) {
    return;
  }
  if (rows > kBlockSize) {
    rows = kBlockSize;
  }

  int64_t tile_group = blockIdx.y;
  const int dim_tile = tile_group % kKDimTiles;
  tile_group /= kKDimTiles;
  const int head_idx = tile_group % ByteV2DefaultLayout::NumKvHeadsValue;
  tile_group /= ByteV2DefaultLayout::NumKvHeadsValue;
  const int kv_side = tile_group;

  uint8_t* __restrict__ page =
      kv_cache + static_cast<int64_t>(physical_block) * kv_cache_stride_block;

  __shared__ int shared_base;
  __shared__ int shared_fallback;
  extern __shared__ int shared_high_counts[];

  int parallel_best_base = 0;
  int parallel_outlier_count = 0;
  int parallel_fallback = 0;
  if constexpr (WarpParallelHistogram) {
    shared_high_counts[threadIdx.x] = 0;
    __syncthreads();

    const uint8_t* __restrict__ staging =
        raw_staging + staging_slot * raw_staging_stride_slot;
    const int elem_count = rows * kCodecDimBlock;
    for (int elem_idx = threadIdx.x; elem_idx < elem_count;
         elem_idx += blockDim.x) {
      const int row = elem_idx / kCodecDimBlock;
      const int dim_offset = elem_idx % kCodecDimBlock;
      const int dim = dim_tile * kCodecDimBlock + dim_offset;
      const int64_t offset =
          kv_side == 0
              ? ByteV2DefaultRawStagingLayout::key_offset(head_idx, row, dim)
              : ByteV2DefaultRawStagingLayout::value_offset(head_idx, row, dim);
      const uint16_t bits = byte_v2_load_u16_bytes(staging, offset);
      atomicAdd(shared_high_counts + byte_v2_high7(bits), 1);
    }
    __syncthreads();

    if (threadIdx.x < 32) {
      const int lane = threadIdx.x;
      int lane_best_count = -1;
      int lane_best_base = lane;
      for (int base = lane; base <= 120; base += 32) {
        int window_count = 0;
#pragma unroll
        for (int delta = 0; delta < 8; ++delta) {
          window_count += shared_high_counts[base + delta];
        }
        if (window_count > lane_best_count) {
          lane_best_count = window_count;
          lane_best_base = base;
        }
      }
      unsigned int best_score =
          (static_cast<unsigned int>(lane_best_count) << 8) |
          static_cast<unsigned int>(255 - lane_best_base);
#pragma unroll
      for (int offset = 16; offset > 0; offset /= 2) {
        best_score =
            max(best_score, __shfl_down_sync(0xffffffffu, best_score, offset));
      }
      if (lane == 0) {
        const int best_count = static_cast<int>(best_score >> 8);
        parallel_best_base = 255 - static_cast<int>(best_score & 0xffu);
        parallel_outlier_count = elem_count - best_count;
        parallel_fallback = parallel_outlier_count >
                            ByteV2DefaultLayout::OutlierEntriesPerTileValue;
      }
    }
  }

  if (threadIdx.x == 0) {
    int best_base = 0;
    int outlier_count = 0;
    int fallback = 0;
    int has_overlay = 0;
    const uint8_t* __restrict__ staging =
        raw_staging + staging_slot * raw_staging_stride_slot;
    if constexpr (WarpParallelHistogram) {
      best_base = parallel_best_base;
      outlier_count = parallel_outlier_count;
      fallback = parallel_fallback;
      has_overlay = outlier_count > 0 && !fallback;
    } else if constexpr (!BypassSerialMetadata) {
      int high_counts[128];
      for (int i = 0; i < 128; ++i) {
        high_counts[i] = 0;
      }
      int elem_count = 0;
      for (int row = 0; row < rows; ++row) {
        for (int dim_offset = 0; dim_offset < kCodecDimBlock; ++dim_offset) {
          const int dim = dim_tile * kCodecDimBlock + dim_offset;
          const int64_t offset =
              kv_side == 0 ? ByteV2DefaultRawStagingLayout::key_offset(head_idx,
                                                                       row, dim)
                           : ByteV2DefaultRawStagingLayout::value_offset(
                                 head_idx, row, dim);
          const uint16_t bits = byte_v2_load_u16_bytes(staging, offset);
          ++high_counts[byte_v2_high7(bits)];
          ++elem_count;
        }
      }
      int best_count = -1;
      for (int base = 0; base <= 120; ++base) {
        int window_count = 0;
        for (int delta = 0; delta < 8; ++delta) {
          window_count += high_counts[base + delta];
        }
        if (window_count > best_count) {
          best_count = window_count;
          best_base = base;
        }
      }
      outlier_count = elem_count - best_count;
      fallback =
          outlier_count > ByteV2DefaultLayout::OutlierEntriesPerTileValue;
      has_overlay = outlier_count > 0 && !fallback;
    }
    const int token_tile = 0;
    const int tile_idx =
        kv_side == 0
            ? dim_tile * ByteV2DefaultPolicy::CodecTokenTilesPerAllocBlock +
                  token_tile
            : token_tile * ByteV2DefaultPolicy::VDimTiles + dim_tile;
    int outlier_pool_index = 0;
    if (has_overlay) {
      outlier_pool_index =
          byte_v2_allocate_outlier_segment<ByteV2DefaultLayout>(page,
                                                                outlier_count);
    }
    if constexpr (FuseMetadataClear) {
      const unsigned int tile_bit = 1u << tile_idx;
      if (kv_side == 0) {
        atomicAnd(
            reinterpret_cast<unsigned int*>(
                page + ByteV2DefaultLayout::k_fallback_mask_offset(head_idx)),
            ~tile_bit);
        atomicAnd(
            reinterpret_cast<unsigned int*>(
                page + ByteV2DefaultLayout::k_outlier_mask_offset(head_idx)),
            ~tile_bit);
        ByteV2DefaultLayout::set_k_outlier_count(page, head_idx, dim_tile,
                                                 token_tile, 0);
      } else {
        atomicAnd(
            reinterpret_cast<unsigned int*>(
                page + ByteV2DefaultLayout::v_fallback_mask_offset(head_idx)),
            ~tile_bit);
        atomicAnd(
            reinterpret_cast<unsigned int*>(
                page + ByteV2DefaultLayout::v_outlier_mask_offset(head_idx)),
            ~tile_bit);
        ByteV2DefaultLayout::set_v_outlier_count(page, head_idx, dim_tile,
                                                 token_tile, 0);
      }
    }
    if (kv_side == 0) {
      page[ByteV2DefaultLayout::k_base_offset(head_idx, dim_tile, token_tile)] =
          static_cast<uint8_t>(best_base);
      if (fallback) {
        atomicOr(
            reinterpret_cast<unsigned int*>(
                page + ByteV2DefaultLayout::k_fallback_mask_offset(head_idx)),
            1u << tile_idx);
      } else if (has_overlay) {
        atomicOr(
            reinterpret_cast<unsigned int*>(
                page + ByteV2DefaultLayout::k_outlier_mask_offset(head_idx)),
            1u << tile_idx);
        byte_v2_set_outlier_descriptor<ByteV2DefaultLayout>(
            page, kv_side, head_idx, dim_tile, token_tile, outlier_count,
            outlier_pool_index);
      }
    } else {
      page[ByteV2DefaultLayout::v_base_offset(head_idx, dim_tile, token_tile)] =
          static_cast<uint8_t>(best_base);
      if (fallback) {
        atomicOr(
            reinterpret_cast<unsigned int*>(
                page + ByteV2DefaultLayout::v_fallback_mask_offset(head_idx)),
            1u << tile_idx);
      } else if (has_overlay) {
        atomicOr(
            reinterpret_cast<unsigned int*>(
                page + ByteV2DefaultLayout::v_outlier_mask_offset(head_idx)),
            1u << tile_idx);
        byte_v2_set_outlier_descriptor<ByteV2DefaultLayout>(
            page, kv_side, head_idx, dim_tile, token_tile, outlier_count,
            outlier_pool_index);
      }
    }
    if (has_overlay) {
      int overlay_entry_idx = 0;
      for (int row = 0; row < rows; ++row) {
        for (int dim_offset = 0; dim_offset < kCodecDimBlock; ++dim_offset) {
          const int dim = dim_tile * kCodecDimBlock + dim_offset;
          const int64_t offset =
              kv_side == 0 ? ByteV2DefaultRawStagingLayout::key_offset(head_idx,
                                                                       row, dim)
                           : ByteV2DefaultRawStagingLayout::value_offset(
                                 head_idx, row, dim);
          const uint16_t bits = byte_v2_load_u16_bytes(staging, offset);
          const int high = static_cast<int>(bits >> 8);
          if (byte_v2_high7_in_window(bits, best_base)) {
            continue;
          }
          const int elem_idx = row * kCodecDimBlock + dim_offset;
          const uint16_t entry = static_cast<uint16_t>(
              ByteV2DefaultLayout::OutlierEntryPolicy::encode(elem_idx, high));
          const int entry_offset =
              kv_side == 0
                  ? ByteV2DefaultLayout::k_outlier_payload_offset(
                        page, head_idx, dim_tile, token_tile, overlay_entry_idx)
                  : ByteV2DefaultLayout::v_outlier_payload_offset(
                        page, head_idx, dim_tile, token_tile,
                        overlay_entry_idx);
          byte_v2_store_u16_bytes(page, entry_offset, entry);
          ++overlay_entry_idx;
        }
      }
    }
    shared_base = best_base;
    shared_fallback = fallback;
  }
  __syncthreads();

  const int pair_idx = threadIdx.x;
  if (pair_idx >= kPairsPerCodecTile) {
    return;
  }
  const int row = pair_idx / kPairsPerRow;
  if (row >= rows) {
    return;
  }

  const int pair_in_dim_tile = pair_idx % kPairsPerRow;
  const int dim0 = dim_tile * kCodecDimBlock + pair_in_dim_tile * 2;
  const int dim1 = dim0 + 1;
  const uint8_t* __restrict__ staging =
      raw_staging + staging_slot * raw_staging_stride_slot;
  const int64_t offset0 =
      kv_side == 0
          ? ByteV2DefaultRawStagingLayout::key_offset(head_idx, row, dim0)
          : ByteV2DefaultRawStagingLayout::value_offset(head_idx, row, dim0);
  const int64_t offset1 =
      kv_side == 0
          ? ByteV2DefaultRawStagingLayout::key_offset(head_idx, row, dim1)
          : ByteV2DefaultRawStagingLayout::value_offset(head_idx, row, dim1);
  const uint16_t bits0 = byte_v2_load_u16_bytes(staging, offset0);
  const uint16_t bits1 = byte_v2_load_u16_bytes(staging, offset1);

  int64_t tile_offset;
  if (kv_side == 0) {
    tile_offset =
        ByteV2DefaultLayout::KPayloadBaseBytes +
        head_idx * ByteV2DefaultLayout::AlignedKPayloadBytesPerKvHead +
        dim_tile * ByteV2DefaultPolicy::CodecTokenTilesPerAllocBlock *
            kPayloadBytesPerTile;
  } else {
    tile_offset =
        ByteV2DefaultLayout::VPayloadBaseBytes +
        head_idx * ByteV2DefaultLayout::AlignedVPayloadBytesPerKvHead +
        dim_tile * kPayloadBytesPerTile;
  }

  const int elem_base = row * kCodecDimBlock + pair_in_dim_tile * 2;
  page[tile_offset + elem_base] = static_cast<uint8_t>(bits0 & 0xff);
  page[tile_offset + elem_base + 1] = static_cast<uint8_t>(bits1 & 0xff);
  const uint8_t code0 = shared_fallback
                            ? byte_v2_code_nibble(bits0)
                            : byte_v2_delta_code_nibble(bits0, shared_base);
  const uint8_t code1 = shared_fallback
                            ? byte_v2_code_nibble(bits1)
                            : byte_v2_delta_code_nibble(bits1, shared_base);
  page[tile_offset + kCodeBase + elem_base / 2] =
      (code0 & 0x0f) | static_cast<uint8_t>((code1 & 0x0f) << 4);

  if constexpr (ByteV2DefaultLayout::IncludeRawPayloadValue) {
    const int64_t raw_offset0 =
        kv_side == 0
            ? ByteV2DefaultLayout::raw_key_offset(head_idx, row, dim0)
            : ByteV2DefaultLayout::raw_value_offset(head_idx, row, dim0);
    const int64_t raw_offset1 =
        kv_side == 0
            ? ByteV2DefaultLayout::raw_key_offset(head_idx, row, dim1)
            : ByteV2DefaultLayout::raw_value_offset(head_idx, row, dim1);
    byte_v2_store_u16_bytes(page, raw_offset0, bits0);
    byte_v2_store_u16_bytes(page, raw_offset1, bits1);
  }
}

__global__ void byte_v2_collect_cache_stats_kernel(
    int32_t* __restrict__ stats, const uint8_t* __restrict__ kv_cache,
    const int32_t* __restrict__ block_tables,
    const int32_t* __restrict__ seq_lens, int64_t num_seqs,
    int64_t block_table_stride_seq, int64_t block_table_stride_block,
    int64_t max_num_blocks_per_seq, int64_t kv_cache_blocks,
    int64_t kv_cache_stride_block, int64_t max_seq_len) {
  constexpr int kBlockSize = ByteV2DefaultPolicy::AllocBlockTokens;

  const int64_t work_items = num_seqs * max_num_blocks_per_seq;
  for (int64_t work_idx = blockIdx.x * blockDim.x + threadIdx.x;
       work_idx < work_items;
       work_idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int64_t seq_idx = work_idx / max_num_blocks_per_seq;
    const int64_t logical_block = work_idx % max_num_blocks_per_seq;
    int64_t seq_len = static_cast<int64_t>(seq_lens[seq_idx]);
    if (seq_len < 0) {
      seq_len = 0;
    }
    if (seq_len > max_seq_len) {
      seq_len = max_seq_len;
    }
    const int64_t num_blocks = (seq_len + kBlockSize - 1) / kBlockSize;
    if (logical_block >= num_blocks) {
      continue;
    }

    const int32_t physical_block =
        block_tables[seq_idx * block_table_stride_seq +
                     logical_block * block_table_stride_block];
    if (physical_block < 0 || physical_block >= kv_cache_blocks) {
      continue;
    }

    const uint8_t* __restrict__ page =
        kv_cache + static_cast<int64_t>(physical_block) * kv_cache_stride_block;
    const ByteV2PageUnsafeStats page_stats =
        byte_v2_collect_page_unsafe_stats(page);

    if (page_stats.fallback_tiles != 0 || page_stats.outlier_tiles != 0) {
      atomicExch(stats, 1);
    }
    atomicAdd(stats + 1, page_stats.fallback_tiles);
    atomicAdd(stats + 2, page_stats.outlier_tiles);
    atomicAdd(stats + 3, 1);
  }
}

__global__ void byte_v2_update_cache_unsafe_flags_kernel(
    int32_t* __restrict__ page_unsafe_flags,
    const uint8_t* __restrict__ kv_cache,
    const int64_t* __restrict__ slot_mapping, int64_t num_tokens,
    int64_t kv_cache_blocks, int64_t kv_cache_stride_block) {
  constexpr int kBlockSize = ByteV2DefaultPolicy::AllocBlockTokens;

  for (int64_t token_idx = blockIdx.x * blockDim.x + threadIdx.x;
       token_idx < num_tokens;
       token_idx += static_cast<int64_t>(blockDim.x) * gridDim.x) {
    const int64_t slot_idx = slot_mapping[token_idx];
    if (slot_idx < 0) {
      continue;
    }
    const int64_t physical_block = slot_idx / kBlockSize;
    if (physical_block < 0 || physical_block >= kv_cache_blocks) {
      continue;
    }
    const uint8_t* __restrict__ page =
        kv_cache + physical_block * kv_cache_stride_block;
    page_unsafe_flags[physical_block] = byte_v2_page_unsafe_flag(page);
  }
}

__global__ void byte_v2_update_cache_single_token_kernel(
    const uint16_t* __restrict__ key, const uint16_t* __restrict__ value,
    uint8_t* __restrict__ kv_cache, const int64_t* __restrict__ slot_mapping,
    int32_t* __restrict__ page_unsafe_flags, int64_t num_heads,
    int64_t key_stride_token, int64_t key_stride_head, int64_t key_stride_dim,
    int64_t value_stride_token, int64_t value_stride_head,
    int64_t value_stride_dim, int64_t kv_cache_blocks,
    int64_t kv_cache_stride_block) {
  constexpr int kCodecDimBlock = ByteV2DefaultPolicy::CodecDimBlock;
  constexpr int kCodecTileElems = ByteV2DefaultPolicy::CodecTileElems;
  constexpr int kPairsPerRow = kCodecDimBlock / 2;
  constexpr int kKDimTiles = ByteV2DefaultPolicy::KDimTiles;
  constexpr int kBlockSize = ByteV2DefaultPolicy::AllocBlockTokens;
  constexpr int kPayloadBytesPerTile =
      ByteV2DefaultLayout::CodecPayloadBytesPerTile;
  constexpr int kCodeBase = kCodecTileElems;

  const int64_t slot_idx = slot_mapping[0];
  if (slot_idx < 0) {
    return;
  }
  const int64_t physical_block = slot_idx / kBlockSize;
  if (physical_block < 0 || physical_block >= kv_cache_blocks) {
    return;
  }
  const int update_row = static_cast<int>(slot_idx % kBlockSize);

  int64_t tile_group = blockIdx.x;
  const int dim_tile = tile_group % kKDimTiles;
  tile_group /= kKDimTiles;
  const int head_idx = tile_group % num_heads;
  tile_group /= num_heads;
  const int kv_side = tile_group;

  uint8_t* __restrict__ page =
      kv_cache + physical_block * kv_cache_stride_block;

  __shared__ int shared_base;

  if (threadIdx.x == 0) {
    const uint16_t* __restrict__ src = kv_side == 0 ? key : value;
    const int64_t stride_head =
        kv_side == 0 ? key_stride_head : value_stride_head;
    const int64_t stride_dim = kv_side == 0 ? key_stride_dim : value_stride_dim;
    const int64_t src_base = head_idx * stride_head;
    const int token_tile = 0;
    const int tile_idx =
        kv_side == 0
            ? dim_tile * ByteV2DefaultPolicy::CodecTokenTilesPerAllocBlock +
                  token_tile
            : token_tile * ByteV2DefaultPolicy::VDimTiles + dim_tile;
    const uint32_t bit = uint32_t{1} << tile_idx;

    const int fallback_mask_offset =
        kv_side == 0 ? ByteV2DefaultLayout::k_fallback_mask_offset(head_idx)
                     : ByteV2DefaultLayout::v_fallback_mask_offset(head_idx);
    const int outlier_mask_offset =
        kv_side == 0 ? ByteV2DefaultLayout::k_outlier_mask_offset(head_idx)
                     : ByteV2DefaultLayout::v_outlier_mask_offset(head_idx);
    const int base_offset = kv_side == 0 ? ByteV2DefaultLayout::k_base_offset(
                                               head_idx, dim_tile, token_tile)
                                         : ByteV2DefaultLayout::v_base_offset(
                                               head_idx, dim_tile, token_tile);
    int base;
    int outlier_count;
    int outlier_pool_index = 0;
    uint32_t previous_outlier_mask = 0;
    if (update_row == 0) {
      int high_counts[128];
      for (int i = 0; i < 128; ++i) {
        high_counts[i] = 0;
      }
      for (int dim_offset = 0; dim_offset < kCodecDimBlock; ++dim_offset) {
        const int dim = dim_tile * kCodecDimBlock + dim_offset;
        const uint16_t bits = src[src_base + dim * stride_dim];
        ++high_counts[byte_v2_high7(bits)];
      }
      int best_base = 0;
      int best_count = -1;
      for (int candidate_base = 0; candidate_base <= 120; ++candidate_base) {
        int window_count = 0;
        for (int delta = 0; delta < 8; ++delta) {
          window_count += high_counts[candidate_base + delta];
        }
        if (window_count > best_count) {
          best_count = window_count;
          best_base = candidate_base;
        }
      }
      base = best_base;
      outlier_count = 0;
      page[base_offset] = static_cast<uint8_t>(base);
    } else {
      base = static_cast<int>(page[base_offset]);
      previous_outlier_mask = byte_v2_load_u32_bytes(page, outlier_mask_offset);
      if ((previous_outlier_mask & bit) != 0) {
        outlier_count = kv_side == 0
                            ? ByteV2DefaultLayout::k_outlier_count(
                                  page, head_idx, dim_tile, token_tile)
                            : ByteV2DefaultLayout::v_outlier_count(
                                  page, head_idx, dim_tile, token_tile);
        outlier_pool_index = kv_side == 0
                                 ? ByteV2DefaultLayout::k_outlier_pool_index(
                                       page, head_idx, dim_tile, token_tile)
                                 : ByteV2DefaultLayout::v_outlier_pool_index(
                                       page, head_idx, dim_tile, token_tile);
      } else {
        outlier_count = 0;
      }
    }

    atomicAnd(reinterpret_cast<unsigned int*>(page + fallback_mask_offset),
              ~bit);
    atomicAnd(reinterpret_cast<unsigned int*>(page + outlier_mask_offset),
              ~bit);

    int row_highs[kCodecDimBlock];
    for (int dim_offset = 0; dim_offset < kCodecDimBlock; ++dim_offset) {
      const int dim = dim_tile * kCodecDimBlock + dim_offset;
      const uint16_t bits = src[src_base + dim * stride_dim];
      row_highs[dim_offset] = static_cast<int>(bits >> 8);
    }

    int row_outlier_count = 0;
    for (int dim_offset = 0; dim_offset < kCodecDimBlock; ++dim_offset) {
      const int high = row_highs[dim_offset];
      if ((high & 0x7f) < base || (high & 0x7f) >= base + 8) {
        ++row_outlier_count;
      }
    }

    const int previous_outlier_count = outlier_count;
    outlier_count += row_outlier_count;
    if (outlier_count > 0) {
      const int previous_capacity =
          previous_outlier_count > 0
              ? byte_v2_outlier_segment_capacity(previous_outlier_count)
              : 0;
      const int required_capacity =
          byte_v2_outlier_segment_capacity(outlier_count);
      if (required_capacity > previous_capacity) {
        const int new_pool_index =
            byte_v2_allocate_outlier_segment<ByteV2DefaultLayout>(
                page, outlier_count);
        for (int entry_idx = 0; entry_idx < previous_outlier_count;
             ++entry_idx) {
          const int old_offset = ByteV2DefaultLayout::OutlierPoolBaseBytes +
                                 (outlier_pool_index + entry_idx) *
                                     ByteV2DefaultLayout::OutlierEntryBytes;
          const int new_offset = ByteV2DefaultLayout::OutlierPoolBaseBytes +
                                 (new_pool_index + entry_idx) *
                                     ByteV2DefaultLayout::OutlierEntryBytes;
          byte_v2_store_u16_bytes(page, new_offset,
                                  byte_v2_load_u16_bytes(page, old_offset));
        }
        outlier_pool_index = new_pool_index;
      }

      int entry_idx = previous_outlier_count;
      for (int dim_offset = 0; dim_offset < kCodecDimBlock; ++dim_offset) {
        const int high = row_highs[dim_offset];
        if ((high & 0x7f) >= base && (high & 0x7f) < base + 8) {
          continue;
        }
        const int elem_idx = update_row * kCodecDimBlock + dim_offset;
        const uint16_t entry = static_cast<uint16_t>(
            ByteV2DefaultLayout::OutlierEntryPolicy::encode(elem_idx, high));
        const int relocated_offset = ByteV2DefaultLayout::OutlierPoolBaseBytes +
                                     (outlier_pool_index + entry_idx) *
                                         ByteV2DefaultLayout::OutlierEntryBytes;
        byte_v2_store_u16_bytes(page, relocated_offset, entry);
        ++entry_idx;
      }
      byte_v2_set_outlier_descriptor<ByteV2DefaultLayout>(
          page, kv_side, head_idx, dim_tile, token_tile, outlier_count,
          outlier_pool_index);
    }

    if (kv_side == 0) {
      if (outlier_count > 0) {
        atomicOr(
            reinterpret_cast<unsigned int*>(
                page + ByteV2DefaultLayout::k_outlier_mask_offset(head_idx)),
            bit);
      }
    } else {
      if (outlier_count > 0) {
        atomicOr(
            reinterpret_cast<unsigned int*>(
                page + ByteV2DefaultLayout::v_outlier_mask_offset(head_idx)),
            bit);
      }
    }

    if (page_unsafe_flags != nullptr && outlier_count > 0) {
      const int side_flag =
          kv_side == 0 ? kByteV2PageUnsafeK : kByteV2PageUnsafeV;
      atomicOr(page_unsafe_flags + physical_block,
               kByteV2PageUnsafeAny | side_flag);
    }

    shared_base = base;
  }
  __syncthreads();

  const int pair_idx = threadIdx.x;
  if (pair_idx >= kPairsPerRow) {
    return;
  }

  const int pair_in_dim_tile = pair_idx % kPairsPerRow;
  const int dim0 = dim_tile * kCodecDimBlock + pair_in_dim_tile * 2;
  const int dim1 = dim0 + 1;
  const uint16_t* __restrict__ src = kv_side == 0 ? key : value;
  const int64_t stride_head =
      kv_side == 0 ? key_stride_head : value_stride_head;
  const int64_t stride_dim = kv_side == 0 ? key_stride_dim : value_stride_dim;
  const int64_t src_base = head_idx * stride_head;
  const uint16_t bits0 = src[src_base + dim0 * stride_dim];
  const uint16_t bits1 = src[src_base + dim1 * stride_dim];

  int64_t tile_offset;
  if (kv_side == 0) {
    tile_offset =
        ByteV2DefaultLayout::KPayloadBaseBytes +
        head_idx * ByteV2DefaultLayout::AlignedKPayloadBytesPerKvHead +
        dim_tile * ByteV2DefaultPolicy::CodecTokenTilesPerAllocBlock *
            kPayloadBytesPerTile;
  } else {
    tile_offset =
        ByteV2DefaultLayout::VPayloadBaseBytes +
        head_idx * ByteV2DefaultLayout::AlignedVPayloadBytesPerKvHead +
        dim_tile * kPayloadBytesPerTile;
  }

  const int elem_base = update_row * kCodecDimBlock + pair_in_dim_tile * 2;
  page[tile_offset + elem_base] = static_cast<uint8_t>(bits0 & 0xff);
  page[tile_offset + elem_base + 1] = static_cast<uint8_t>(bits1 & 0xff);
  const uint8_t code0 = byte_v2_delta_code_nibble(bits0, shared_base);
  const uint8_t code1 = byte_v2_delta_code_nibble(bits1, shared_base);
  page[tile_offset + kCodeBase + elem_base / 2] =
      (code0 & 0x0f) | static_cast<uint8_t>((code1 & 0x0f) << 4);
}

template <typename Layout, bool UseRawFallback, bool AssumeNoFallbackNoOutlier>
void launch_byte_v2_paged_decode_attention(torch::stable::Tensor& output,
                                           torch::stable::Tensor& query,
                                           torch::stable::Tensor& kv_cache,
                                           torch::stable::Tensor& block_tables,
                                           torch::stable::Tensor& seq_lens,
                                           double scale) {
  constexpr int kThreads = Layout::TilePolicy::HeadDim;
  static_assert(kThreads == 128);

  const int num_seqs = output.size(0);
  const int num_heads = output.size(1);
  if (num_seqs == 0 || num_heads == 0) {
    return;
  }

  const torch::stable::accelerator::DeviceGuard device_guard(
      query.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream(query.get_device_index());

  const dim3 grid(num_heads, num_seqs, 1);
  const dim3 block(kThreads);
  byte_v2_paged_decode_attention_kernel<Layout, UseRawFallback,
                                        AssumeNoFallbackNoOutlier, kThreads>
      <<<grid, block, 0, stream>>>(
          reinterpret_cast<uint16_t*>(output.mutable_data_ptr()),
          reinterpret_cast<const uint16_t*>(query.const_data_ptr()),
          reinterpret_cast<const uint8_t*>(kv_cache.const_data_ptr()),
          block_tables.const_data_ptr<int32_t>(),
          seq_lens.const_data_ptr<int32_t>(), static_cast<float>(scale),
          query.stride(0), query.stride(1), query.stride(2), output.stride(0),
          output.stride(1), output.stride(2), kv_cache.stride(0),
          block_tables.stride(0), block_tables.size(1));

  const cudaError_t err = cudaGetLastError();
  STD_TORCH_CHECK(err == cudaSuccess,
                  "byte_v2_paged_decode_attention kernel launch failed: ",
                  cudaGetErrorString(err));
}

template <typename Layout, bool UseRawFallback, bool AssumeNoFallbackNoOutlier,
          bool UsePageUnsafeFlags>
void launch_byte_v2_paged_decode_attention_split_k(
    torch::stable::Tensor& output, torch::stable::Tensor& exp_sums,
    torch::stable::Tensor& max_logits, torch::stable::Tensor& tmp_out,
    torch::stable::Tensor& query, torch::stable::Tensor& kv_cache,
    const int32_t* page_unsafe_flags, torch::stable::Tensor& block_tables,
    torch::stable::Tensor& seq_lens, double scale, int64_t partition_size) {
  constexpr int kThreads = Layout::TilePolicy::HeadDim;
  static_assert(kThreads == 128);

  const int num_seqs = output.size(0);
  const int num_heads = output.size(1);
  if (num_seqs == 0 || num_heads == 0) {
    return;
  }

  const int max_num_partitions = exp_sums.size(2);
  const torch::stable::accelerator::DeviceGuard device_guard(
      query.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream(query.get_device_index());

  const dim3 split_grid(num_heads, num_seqs, max_num_partitions);
  const dim3 block(kThreads);
  byte_v2_paged_decode_attention_split_k_kernel<Layout, UseRawFallback,
                                                AssumeNoFallbackNoOutlier,
                                                UsePageUnsafeFlags, kThreads>
      <<<split_grid, block, 0, stream>>>(
          reinterpret_cast<float*>(tmp_out.mutable_data_ptr()),
          reinterpret_cast<float*>(exp_sums.mutable_data_ptr()),
          max_logits.numel() == 0
              ? nullptr
              : reinterpret_cast<float*>(max_logits.mutable_data_ptr()),
          reinterpret_cast<const uint16_t*>(query.const_data_ptr()),
          reinterpret_cast<const uint8_t*>(kv_cache.const_data_ptr()),
          page_unsafe_flags, block_tables.const_data_ptr<int32_t>(),
          seq_lens.const_data_ptr<int32_t>(), static_cast<float>(scale),
          query.stride(0), query.stride(1), query.stride(2), kv_cache.stride(0),
          block_tables.stride(0), block_tables.size(1), max_num_partitions,
          static_cast<int>(partition_size));

  cudaError_t err = cudaGetLastError();
  STD_TORCH_CHECK(
      err == cudaSuccess,
      "byte_v2_paged_decode_attention_split_k kernel launch failed: ",
      cudaGetErrorString(err));

  constexpr int kReduceWarpsPerBlock = 4;
  const dim3 reduce_grid(
      num_heads, num_seqs,
      (Layout::TilePolicy::HeadDimV + kReduceWarpsPerBlock - 1) /
          kReduceWarpsPerBlock);
  byte_v2_paged_decode_attention_split_k_reduce_warp_kernel<Layout>
      <<<reduce_grid, block, 0, stream>>>(
          reinterpret_cast<uint16_t*>(output.mutable_data_ptr()),
          reinterpret_cast<const float*>(tmp_out.const_data_ptr()),
          reinterpret_cast<const float*>(exp_sums.const_data_ptr()),
          max_logits.numel() == 0
              ? nullptr
              : reinterpret_cast<const float*>(max_logits.const_data_ptr()),
          seq_lens.const_data_ptr<int32_t>(), output.stride(0),
          output.stride(1), output.stride(2), max_num_partitions,
          static_cast<int>(partition_size));

  err = cudaGetLastError();
  STD_TORCH_CHECK(
      err == cudaSuccess,
      "byte_v2_paged_decode_attention_split_k_reduce kernel launch failed: ",
      cudaGetErrorString(err));
}

template <typename Layout, bool UsePageUnsafeFlags>
void launch_byte_v2_paged_decode_attention_split_k_gqa4_no_fallback_no_outlier(
    torch::stable::Tensor& output, torch::stable::Tensor& exp_sums,
    torch::stable::Tensor& max_logits, torch::stable::Tensor& tmp_out,
    torch::stable::Tensor& query, torch::stable::Tensor& kv_cache,
    const int32_t* page_unsafe_flags, torch::stable::Tensor& block_tables,
    torch::stable::Tensor& seq_lens, double scale, int64_t partition_size) {
  constexpr int kThreads = Layout::TilePolicy::HeadDim;
  static_assert(kThreads == 128);

  const int num_seqs = output.size(0);
  const int num_heads = output.size(1);
  if (num_seqs == 0 || num_heads == 0) {
    return;
  }

  const int max_num_partitions = exp_sums.size(2);
  const torch::stable::accelerator::DeviceGuard device_guard(
      query.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream(query.get_device_index());

  const dim3 split_grid(Layout::NumKvHeadsValue, num_seqs, max_num_partitions);
  const dim3 block(kThreads);
  byte_v2_paged_decode_attention_split_k_gqa4_no_fallback_no_outlier_kernel<
      Layout, UsePageUnsafeFlags, kThreads><<<split_grid, block, 0, stream>>>(
      reinterpret_cast<float*>(tmp_out.mutable_data_ptr()),
      reinterpret_cast<float*>(exp_sums.mutable_data_ptr()),
      max_logits.numel() == 0
          ? nullptr
          : reinterpret_cast<float*>(max_logits.mutable_data_ptr()),
      reinterpret_cast<const uint16_t*>(query.const_data_ptr()),
      reinterpret_cast<const uint8_t*>(kv_cache.const_data_ptr()),
      page_unsafe_flags, block_tables.const_data_ptr<int32_t>(),
      seq_lens.const_data_ptr<int32_t>(), static_cast<float>(scale),
      query.stride(0), query.stride(1), query.stride(2), kv_cache.stride(0),
      block_tables.stride(0), block_tables.size(1), max_num_partitions,
      static_cast<int>(partition_size));

  cudaError_t err = cudaGetLastError();
  STD_TORCH_CHECK(
      err == cudaSuccess,
      "byte_v2_paged_decode_attention_split_k_gqa4 kernel launch failed: ",
      cudaGetErrorString(err));

  constexpr int kReduceWarpsPerBlock = 4;
  const dim3 reduce_grid(
      num_heads, num_seqs,
      (Layout::TilePolicy::HeadDimV + kReduceWarpsPerBlock - 1) /
          kReduceWarpsPerBlock);
  byte_v2_paged_decode_attention_split_k_reduce_warp_kernel<Layout>
      <<<reduce_grid, block, 0, stream>>>(
          reinterpret_cast<uint16_t*>(output.mutable_data_ptr()),
          reinterpret_cast<const float*>(tmp_out.const_data_ptr()),
          reinterpret_cast<const float*>(exp_sums.const_data_ptr()),
          max_logits.numel() == 0
              ? nullptr
              : reinterpret_cast<const float*>(max_logits.const_data_ptr()),
          seq_lens.const_data_ptr<int32_t>(), output.stride(0),
          output.stride(1), output.stride(2), max_num_partitions,
          static_cast<int>(partition_size));

  err = cudaGetLastError();
  STD_TORCH_CHECK(err == cudaSuccess,
                  "byte_v2_paged_decode_attention_split_k_gqa4_reduce kernel "
                  "launch failed: ",
                  cudaGetErrorString(err));
}

template <typename Layout, bool UsePageUnsafeFlags, bool UseQkMma,
          bool UseFa2Mainloop, bool UseFa2Multiwarp, bool UseFa2Direct,
          int QHeadsPerKv = 4, int QGroupTile = 4,
          int DirectDiagnosticMode = kByteV2DirectDiagnosticCurrent,
          int SpeculativeQueryLen = 0, bool UseRaggedSpeculativeQ4 = false>
void launch_byte_v2_paged_decode_attention_split_k_gqa4_fa2_like_no_fallback_no_outlier(
    torch::stable::Tensor& output, torch::stable::Tensor& exp_sums,
    torch::stable::Tensor& max_logits, torch::stable::Tensor& tmp_out,
    torch::stable::Tensor& query, torch::stable::Tensor& kv_cache,
    const int32_t* page_unsafe_flags, torch::stable::Tensor& block_tables,
    torch::stable::Tensor& seq_lens, double scale, int64_t partition_size,
    const int32_t* query_start_locs = nullptr, int num_ragged_requests = 0,
    int num_actual_tokens = 0) {
  using Policy = typename Layout::TilePolicy;
  constexpr int kThreads = Policy::HeadDim;
  constexpr int kGroupsPerKv = (QHeadsPerKv + QGroupTile - 1) / QGroupTile;
  static_assert(kThreads == 128);
  static_assert(Policy::ComputeBlockN == 64);
  static_assert(QHeadsPerKv > 0);
  static_assert(QGroupTile > 0);
  static_assert(QGroupTile <= QHeadsPerKv);
  static_assert(DirectDiagnosticMode == kByteV2DirectDiagnosticCurrent ||
                UseFa2Direct);
  static_assert(!UseRaggedSpeculativeQ4 || SpeculativeQueryLen == 4);

  const int num_seqs =
      UseRaggedSpeculativeQ4
          ? num_ragged_requests
          : (SpeculativeQueryLen > 0
                 ? static_cast<int>(output.size(0) / SpeculativeQueryLen)
                 : static_cast<int>(output.size(0)));
  const int num_heads = SpeculativeQueryLen > 0
                            ? Layout::NumKvHeadsValue * QHeadsPerKv
                            : static_cast<int>(output.size(1));
  if (num_seqs == 0 || num_heads == 0) {
    return;
  }

  const int max_num_partitions = exp_sums.size(2);
  if constexpr (UseRaggedSpeculativeQ4) {
    const size_t row_reduce_smem = (max_num_partitions + 1) * sizeof(float);
    STD_TORCH_CHECK(row_reduce_smem <= 48 * 1024,
                    "ByteV2 ragged Q4 row-packed reduction exceeds 48 KiB");
  }
  if constexpr (DirectDiagnosticMode ==
                kByteV2DirectDiagnosticUnnormalizedPartitionOutput) {
    STD_TORCH_CHECK(
        max_logits.numel() != 0,
        "ByteV2 unnormalized partition output requires max_logits workspace");
  }
  const torch::stable::accelerator::DeviceGuard device_guard(
      query.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream(query.get_device_index());

  const dim3 split_grid(Layout::NumKvHeadsValue * kGroupsPerKv, num_seqs,
                        max_num_partitions);
  const dim3 block(kThreads);
  byte_v2_paged_decode_attention_split_k_gqa4_fa2_like_no_fallback_no_outlier_kernel<
      Layout, UsePageUnsafeFlags, UseQkMma, UseFa2Mainloop, UseFa2Multiwarp,
      UseFa2Direct, QHeadsPerKv, QGroupTile, kThreads, DirectDiagnosticMode,
      SpeculativeQueryLen, UseRaggedSpeculativeQ4>
      <<<split_grid, block, 0, stream>>>(
          reinterpret_cast<float*>(tmp_out.mutable_data_ptr()),
          reinterpret_cast<float*>(exp_sums.mutable_data_ptr()),
          max_logits.numel() == 0
              ? nullptr
              : reinterpret_cast<float*>(max_logits.mutable_data_ptr()),
          reinterpret_cast<const uint16_t*>(query.const_data_ptr()),
          reinterpret_cast<const uint8_t*>(kv_cache.const_data_ptr()),
          page_unsafe_flags, block_tables.const_data_ptr<int32_t>(),
          seq_lens.const_data_ptr<int32_t>(), query_start_locs,
          num_actual_tokens, static_cast<float>(scale), query.stride(0),
          query.stride(1), query.stride(2), kv_cache.stride(0),
          block_tables.stride(0), block_tables.size(1), max_num_partitions,
          static_cast<int>(partition_size));

  cudaError_t err = cudaGetLastError();
  STD_TORCH_CHECK(
      err == cudaSuccess,
      "byte_v2_paged_decode_attention_split_k_gqa4_fa2_like kernel launch "
      "failed: ",
      cudaGetErrorString(err));

  if constexpr (SpeculativeQueryLen > 0 &&
                DirectDiagnosticMode ==
                    kByteV2DirectDiagnosticDecodeStageOnly) {
    return;
  }

  if constexpr (DirectDiagnosticMode ==
                kByteV2DirectDiagnosticKDecodeQkNoOutput) {
    return;
  }

  constexpr int kReduceWarpsPerBlock = 4;
  const dim3 reduce_grid(
      num_heads, num_seqs,
      (Layout::TilePolicy::HeadDimV + kReduceWarpsPerBlock - 1) /
          kReduceWarpsPerBlock);
  constexpr bool kUseUnnormalizedOutput =
      DirectDiagnosticMode ==
      kByteV2DirectDiagnosticUnnormalizedPartitionOutput;
  if constexpr ((SpeculativeQueryLen == 16 || UseRaggedSpeculativeQ4) &&
                !kUseUnnormalizedOutput) {
    const size_t row_reduce_smem = (max_num_partitions + 1) * sizeof(float);
    if (row_reduce_smem <= 48 * 1024) {
      const dim3 row_reduce_grid(num_heads, num_seqs);
      byte_v2_paged_decode_attention_split_k_reduce_speculative_row_kernel<
          Layout, SpeculativeQueryLen, UseRaggedSpeculativeQ4>
          <<<row_reduce_grid, block, row_reduce_smem, stream>>>(
              reinterpret_cast<uint16_t*>(output.mutable_data_ptr()),
              reinterpret_cast<const float*>(tmp_out.const_data_ptr()),
              reinterpret_cast<const float*>(exp_sums.const_data_ptr()),
              seq_lens.const_data_ptr<int32_t>(), query_start_locs,
              num_actual_tokens, output.stride(0), output.stride(1),
              output.stride(2), max_num_partitions,
              static_cast<int>(partition_size));
    } else {
      byte_v2_paged_decode_attention_split_k_reduce_warp_kernel<
          Layout, kUseUnnormalizedOutput, SpeculativeQueryLen>
          <<<reduce_grid, block, 0, stream>>>(
              reinterpret_cast<uint16_t*>(output.mutable_data_ptr()),
              reinterpret_cast<const float*>(tmp_out.const_data_ptr()),
              reinterpret_cast<const float*>(exp_sums.const_data_ptr()),
              nullptr, seq_lens.const_data_ptr<int32_t>(), output.stride(0),
              output.stride(1), output.stride(2), max_num_partitions,
              static_cast<int>(partition_size));
    }
  } else {
    byte_v2_paged_decode_attention_split_k_reduce_warp_kernel<
        Layout, kUseUnnormalizedOutput, SpeculativeQueryLen>
        <<<reduce_grid, block, 0, stream>>>(
            reinterpret_cast<uint16_t*>(output.mutable_data_ptr()),
            reinterpret_cast<const float*>(tmp_out.const_data_ptr()),
            reinterpret_cast<const float*>(exp_sums.const_data_ptr()),
            max_logits.numel() == 0
                ? nullptr
                : reinterpret_cast<const float*>(max_logits.const_data_ptr()),
            seq_lens.const_data_ptr<int32_t>(), output.stride(0),
            output.stride(1), output.stride(2), max_num_partitions,
            static_cast<int>(partition_size));
  }

  err = cudaGetLastError();
  STD_TORCH_CHECK(
      err == cudaSuccess,
      "byte_v2_paged_decode_attention_split_k_gqa4_fa2_like_reduce kernel "
      "launch failed: ",
      cudaGetErrorString(err));
}

template <typename Layout, bool UsePageUnsafeFlags>
void launch_byte_v2_paged_decode_attention_split_k_gqa4_qk_window16_diagnostic(
    torch::stable::Tensor& output, torch::stable::Tensor& exp_sums,
    torch::stable::Tensor& max_logits, torch::stable::Tensor& tmp_out,
    torch::stable::Tensor& query, torch::stable::Tensor& kv_cache,
    const int32_t* page_unsafe_flags, torch::stable::Tensor& block_tables,
    torch::stable::Tensor& seq_lens, double scale, int64_t partition_size) {
  using Policy = typename Layout::TilePolicy;
  constexpr int kThreads = Policy::HeadDim;
  static_assert(kThreads == 128);
  static_assert(Policy::ComputeBlockN == 64);

  const int num_seqs = output.size(0);
  const int num_heads = output.size(1);
  if (num_seqs == 0 || num_heads == 0) {
    return;
  }

  const int max_num_partitions = exp_sums.size(2);
  const torch::stable::accelerator::DeviceGuard device_guard(
      query.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream(query.get_device_index());

  const dim3 split_grid(Layout::NumKvHeadsValue, num_seqs, max_num_partitions);
  const dim3 block(kThreads);
  byte_v2_paged_decode_attention_split_k_gqa4_qk_window16_diagnostic_kernel<
      Layout, UsePageUnsafeFlags, 4, 4, kThreads>
      <<<split_grid, block, 0, stream>>>(
          reinterpret_cast<float*>(tmp_out.mutable_data_ptr()),
          reinterpret_cast<float*>(exp_sums.mutable_data_ptr()),
          max_logits.numel() == 0
              ? nullptr
              : reinterpret_cast<float*>(max_logits.mutable_data_ptr()),
          reinterpret_cast<const uint16_t*>(query.const_data_ptr()),
          reinterpret_cast<const uint8_t*>(kv_cache.const_data_ptr()),
          page_unsafe_flags, block_tables.const_data_ptr<int32_t>(),
          seq_lens.const_data_ptr<int32_t>(), static_cast<float>(scale),
          query.stride(0), query.stride(1), query.stride(2), kv_cache.stride(0),
          block_tables.stride(0), block_tables.size(1), max_num_partitions,
          static_cast<int>(partition_size));

  cudaError_t err = cudaGetLastError();
  STD_TORCH_CHECK(
      err == cudaSuccess,
      "byte_v2_paged_decode_attention_split_k_gqa4_qk_window16_diagnostic "
      "kernel launch failed: ",
      cudaGetErrorString(err));

  constexpr int kReduceWarpsPerBlock = 4;
  const dim3 reduce_grid(
      num_heads, num_seqs,
      (Layout::TilePolicy::HeadDimV + kReduceWarpsPerBlock - 1) /
          kReduceWarpsPerBlock);
  byte_v2_paged_decode_attention_split_k_reduce_warp_kernel<Layout>
      <<<reduce_grid, block, 0, stream>>>(
          reinterpret_cast<uint16_t*>(output.mutable_data_ptr()),
          reinterpret_cast<const float*>(tmp_out.const_data_ptr()),
          reinterpret_cast<const float*>(exp_sums.const_data_ptr()),
          max_logits.numel() == 0
              ? nullptr
              : reinterpret_cast<const float*>(max_logits.const_data_ptr()),
          seq_lens.const_data_ptr<int32_t>(), output.stride(0),
          output.stride(1), output.stride(2), max_num_partitions,
          static_cast<int>(partition_size));

  err = cudaGetLastError();
  STD_TORCH_CHECK(
      err == cudaSuccess,
      "byte_v2_paged_decode_attention_split_k_gqa4_qk_window16_diagnostic "
      "reduce kernel launch failed: ",
      cudaGetErrorString(err));
}

template <typename Layout, bool UsePageUnsafeFlags>
void launch_byte_v2_paged_decode_attention_split_k_fa2_direct_grouped_no_fallback_no_outlier(
    torch::stable::Tensor& output, torch::stable::Tensor& exp_sums,
    torch::stable::Tensor& max_logits, torch::stable::Tensor& tmp_out,
    torch::stable::Tensor& query, torch::stable::Tensor& kv_cache,
    const int32_t* page_unsafe_flags, torch::stable::Tensor& block_tables,
    torch::stable::Tensor& seq_lens, double scale, int64_t partition_size,
    int q_heads_per_kv, int direct_diagnostic_mode) {
  if (direct_diagnostic_mode != kByteV2DirectDiagnosticCurrent) {
    if (direct_diagnostic_mode == kByteV2DirectDiagnosticEffectiveM16Profile) {
      STD_TORCH_CHECK(
          q_heads_per_kv == 16,
          "ByteV2 effective-m16 diagnostic mode requires q_per_kv=16");
    } else {
      STD_TORCH_CHECK(
          q_heads_per_kv == 4,
          "ByteV2 FA2-direct diagnostic modes currently require q_per_kv=4");
    }
    switch (direct_diagnostic_mode) {
      case kByteV2DirectDiagnosticDecodeStageOnly:
        launch_byte_v2_paged_decode_attention_split_k_gqa4_fa2_like_no_fallback_no_outlier<
            Layout, UsePageUnsafeFlags, true, false, false, true, 4, 4,
            kByteV2DirectDiagnosticDecodeStageOnly>(
            output, exp_sums, max_logits, tmp_out, query, kv_cache,
            page_unsafe_flags, block_tables, seq_lens, scale, partition_size);
        return;
      case kByteV2DirectDiagnosticFakeDecodeZero:
        launch_byte_v2_paged_decode_attention_split_k_gqa4_fa2_like_no_fallback_no_outlier<
            Layout, UsePageUnsafeFlags, true, false, false, true, 4, 4,
            kByteV2DirectDiagnosticFakeDecodeZero>(
            output, exp_sums, max_logits, tmp_out, query, kv_cache,
            page_unsafe_flags, block_tables, seq_lens, scale, partition_size);
        return;
      case kByteV2DirectDiagnosticRawSameSkeleton:
        launch_byte_v2_paged_decode_attention_split_k_gqa4_fa2_like_no_fallback_no_outlier<
            Layout, UsePageUnsafeFlags, true, false, false, true, 4, 4,
            kByteV2DirectDiagnosticRawSameSkeleton>(
            output, exp_sums, max_logits, tmp_out, query, kv_cache,
            page_unsafe_flags, block_tables, seq_lens, scale, partition_size);
        return;
      case kByteV2DirectDiagnosticKDecodeStageOnly:
        launch_byte_v2_paged_decode_attention_split_k_gqa4_fa2_like_no_fallback_no_outlier<
            Layout, UsePageUnsafeFlags, true, false, false, true, 4, 4,
            kByteV2DirectDiagnosticKDecodeStageOnly>(
            output, exp_sums, max_logits, tmp_out, query, kv_cache,
            page_unsafe_flags, block_tables, seq_lens, scale, partition_size);
        return;
      case kByteV2DirectDiagnosticVDecodeStageOnly:
        launch_byte_v2_paged_decode_attention_split_k_gqa4_fa2_like_no_fallback_no_outlier<
            Layout, UsePageUnsafeFlags, true, false, false, true, 4, 4,
            kByteV2DirectDiagnosticVDecodeStageOnly>(
            output, exp_sums, max_logits, tmp_out, query, kv_cache,
            page_unsafe_flags, block_tables, seq_lens, scale, partition_size);
        return;
      case kByteV2DirectDiagnosticKDecodeQkOnly:
        launch_byte_v2_paged_decode_attention_split_k_gqa4_fa2_like_no_fallback_no_outlier<
            Layout, UsePageUnsafeFlags, true, false, false, true, 4, 4,
            kByteV2DirectDiagnosticKDecodeQkOnly>(
            output, exp_sums, max_logits, tmp_out, query, kv_cache,
            page_unsafe_flags, block_tables, seq_lens, scale, partition_size);
        return;
      case kByteV2DirectDiagnosticVDecodePvOnly:
        launch_byte_v2_paged_decode_attention_split_k_gqa4_fa2_like_no_fallback_no_outlier<
            Layout, UsePageUnsafeFlags, true, false, false, true, 4, 4,
            kByteV2DirectDiagnosticVDecodePvOnly>(
            output, exp_sums, max_logits, tmp_out, query, kv_cache,
            page_unsafe_flags, block_tables, seq_lens, scale, partition_size);
        return;
      case kByteV2DirectDiagnosticKDecodeProducer3StageOnly:
        launch_byte_v2_paged_decode_attention_split_k_gqa4_fa2_like_no_fallback_no_outlier<
            Layout, UsePageUnsafeFlags, true, false, false, true, 4, 4,
            kByteV2DirectDiagnosticKDecodeProducer3StageOnly>(
            output, exp_sums, max_logits, tmp_out, query, kv_cache,
            page_unsafe_flags, block_tables, seq_lens, scale, partition_size);
        return;
      case kByteV2DirectDiagnosticVDecodeProducer3StageOnly:
        launch_byte_v2_paged_decode_attention_split_k_gqa4_fa2_like_no_fallback_no_outlier<
            Layout, UsePageUnsafeFlags, true, false, false, true, 4, 4,
            kByteV2DirectDiagnosticVDecodeProducer3StageOnly>(
            output, exp_sums, max_logits, tmp_out, query, kv_cache,
            page_unsafe_flags, block_tables, seq_lens, scale, partition_size);
        return;
      case kByteV2DirectDiagnosticKDecodeProducer2StageOnly:
        launch_byte_v2_paged_decode_attention_split_k_gqa4_fa2_like_no_fallback_no_outlier<
            Layout, UsePageUnsafeFlags, true, false, false, true, 4, 4,
            kByteV2DirectDiagnosticKDecodeProducer2StageOnly>(
            output, exp_sums, max_logits, tmp_out, query, kv_cache,
            page_unsafe_flags, block_tables, seq_lens, scale, partition_size);
        return;
      case kByteV2DirectDiagnosticVDecodeProducer2StageOnly:
        launch_byte_v2_paged_decode_attention_split_k_gqa4_fa2_like_no_fallback_no_outlier<
            Layout, UsePageUnsafeFlags, true, false, false, true, 4, 4,
            kByteV2DirectDiagnosticVDecodeProducer2StageOnly>(
            output, exp_sums, max_logits, tmp_out, query, kv_cache,
            page_unsafe_flags, block_tables, seq_lens, scale, partition_size);
        return;
      case kByteV2DirectDiagnosticKDecodeNoStoreStageOnly:
        launch_byte_v2_paged_decode_attention_split_k_gqa4_fa2_like_no_fallback_no_outlier<
            Layout, UsePageUnsafeFlags, true, false, false, true, 4, 4,
            kByteV2DirectDiagnosticKDecodeNoStoreStageOnly>(
            output, exp_sums, max_logits, tmp_out, query, kv_cache,
            page_unsafe_flags, block_tables, seq_lens, scale, partition_size);
        return;
      case kByteV2DirectDiagnosticVDecodeNoStoreStageOnly:
        launch_byte_v2_paged_decode_attention_split_k_gqa4_fa2_like_no_fallback_no_outlier<
            Layout, UsePageUnsafeFlags, true, false, false, true, 4, 4,
            kByteV2DirectDiagnosticVDecodeNoStoreStageOnly>(
            output, exp_sums, max_logits, tmp_out, query, kv_cache,
            page_unsafe_flags, block_tables, seq_lens, scale, partition_size);
        return;
      case kByteV2DirectDiagnosticKDecodeLowOnlyStageOnly:
        launch_byte_v2_paged_decode_attention_split_k_gqa4_fa2_like_no_fallback_no_outlier<
            Layout, UsePageUnsafeFlags, true, false, false, true, 4, 4,
            kByteV2DirectDiagnosticKDecodeLowOnlyStageOnly>(
            output, exp_sums, max_logits, tmp_out, query, kv_cache,
            page_unsafe_flags, block_tables, seq_lens, scale, partition_size);
        return;
      case kByteV2DirectDiagnosticVDecodeLowOnlyStageOnly:
        launch_byte_v2_paged_decode_attention_split_k_gqa4_fa2_like_no_fallback_no_outlier<
            Layout, UsePageUnsafeFlags, true, false, false, true, 4, 4,
            kByteV2DirectDiagnosticVDecodeLowOnlyStageOnly>(
            output, exp_sums, max_logits, tmp_out, query, kv_cache,
            page_unsafe_flags, block_tables, seq_lens, scale, partition_size);
        return;
      case kByteV2DirectDiagnosticKDecodeHighOnlyStageOnly:
        launch_byte_v2_paged_decode_attention_split_k_gqa4_fa2_like_no_fallback_no_outlier<
            Layout, UsePageUnsafeFlags, true, false, false, true, 4, 4,
            kByteV2DirectDiagnosticKDecodeHighOnlyStageOnly>(
            output, exp_sums, max_logits, tmp_out, query, kv_cache,
            page_unsafe_flags, block_tables, seq_lens, scale, partition_size);
        return;
      case kByteV2DirectDiagnosticVDecodeHighOnlyStageOnly:
        launch_byte_v2_paged_decode_attention_split_k_gqa4_fa2_like_no_fallback_no_outlier<
            Layout, UsePageUnsafeFlags, true, false, false, true, 4, 4,
            kByteV2DirectDiagnosticVDecodeHighOnlyStageOnly>(
            output, exp_sums, max_logits, tmp_out, query, kv_cache,
            page_unsafe_flags, block_tables, seq_lens, scale, partition_size);
        return;
      case kByteV2DirectDiagnosticPhaseProfile:
        launch_byte_v2_paged_decode_attention_split_k_gqa4_fa2_like_no_fallback_no_outlier<
            Layout, UsePageUnsafeFlags, true, false, false, true, 4, 4,
            kByteV2DirectDiagnosticPhaseProfile>(
            output, exp_sums, max_logits, tmp_out, query, kv_cache,
            page_unsafe_flags, block_tables, seq_lens, scale, partition_size);
        return;
      case kByteV2DirectDiagnosticStageWindow16:
        launch_byte_v2_paged_decode_attention_split_k_gqa4_fa2_like_no_fallback_no_outlier<
            Layout, UsePageUnsafeFlags, true, false, false, true, 4, 4,
            kByteV2DirectDiagnosticStageWindow16>(
            output, exp_sums, max_logits, tmp_out, query, kv_cache,
            page_unsafe_flags, block_tables, seq_lens, scale, partition_size);
        return;
      case kByteV2DirectDiagnosticStageWindow32:
        launch_byte_v2_paged_decode_attention_split_k_gqa4_fa2_like_no_fallback_no_outlier<
            Layout, UsePageUnsafeFlags, true, false, false, true, 4, 4,
            kByteV2DirectDiagnosticStageWindow32>(
            output, exp_sums, max_logits, tmp_out, query, kv_cache,
            page_unsafe_flags, block_tables, seq_lens, scale, partition_size);
        return;
      case kByteV2DirectDiagnosticStageWindow16Specialized:
        launch_byte_v2_paged_decode_attention_split_k_gqa4_fa2_like_no_fallback_no_outlier<
            Layout, UsePageUnsafeFlags, true, false, false, true, 4, 4,
            kByteV2DirectDiagnosticStageWindow16Specialized>(
            output, exp_sums, max_logits, tmp_out, query, kv_cache,
            page_unsafe_flags, block_tables, seq_lens, scale, partition_size);
        return;
      case kByteV2DirectDiagnosticQkWindow16Profile:
        launch_byte_v2_paged_decode_attention_split_k_gqa4_qk_window16_diagnostic<
            Layout, UsePageUnsafeFlags>(
            output, exp_sums, max_logits, tmp_out, query, kv_cache,
            page_unsafe_flags, block_tables, seq_lens, scale, partition_size);
        return;
      case kByteV2DirectDiagnosticEffectiveM16Profile:
        launch_byte_v2_paged_decode_attention_split_k_gqa4_fa2_like_no_fallback_no_outlier<
            Layout, UsePageUnsafeFlags, true, false, false, true, 16, 16,
            kByteV2DirectDiagnosticEffectiveM16Profile>(
            output, exp_sums, max_logits, tmp_out, query, kv_cache,
            page_unsafe_flags, block_tables, seq_lens, scale, partition_size);
        return;
      case kByteV2DirectDiagnosticVDecodePvNoGemmOnly:
        launch_byte_v2_paged_decode_attention_split_k_gqa4_fa2_like_no_fallback_no_outlier<
            Layout, UsePageUnsafeFlags, true, false, false, true, 4, 4,
            kByteV2DirectDiagnosticVDecodePvNoGemmOnly>(
            output, exp_sums, max_logits, tmp_out, query, kv_cache,
            page_unsafe_flags, block_tables, seq_lens, scale, partition_size);
        return;
      case kByteV2DirectDiagnosticVDecodePvNoAccumOnly:
        launch_byte_v2_paged_decode_attention_split_k_gqa4_fa2_like_no_fallback_no_outlier<
            Layout, UsePageUnsafeFlags, true, false, false, true, 4, 4,
            kByteV2DirectDiagnosticVDecodePvNoAccumOnly>(
            output, exp_sums, max_logits, tmp_out, query, kv_cache,
            page_unsafe_flags, block_tables, seq_lens, scale, partition_size);
        return;
      case kByteV2DirectDiagnosticKDecodeQkNoOutput:
        launch_byte_v2_paged_decode_attention_split_k_gqa4_fa2_like_no_fallback_no_outlier<
            Layout, UsePageUnsafeFlags, true, false, false, true, 4, 4,
            kByteV2DirectDiagnosticKDecodeQkNoOutput>(
            output, exp_sums, max_logits, tmp_out, query, kv_cache,
            page_unsafe_flags, block_tables, seq_lens, scale, partition_size);
        return;
      case kByteV2DirectDiagnosticUnnormalizedPartitionOutput:
        launch_byte_v2_paged_decode_attention_split_k_gqa4_fa2_like_no_fallback_no_outlier<
            Layout, UsePageUnsafeFlags, true, false, false, true, 4, 4,
            kByteV2DirectDiagnosticUnnormalizedPartitionOutput>(
            output, exp_sums, max_logits, tmp_out, query, kv_cache,
            page_unsafe_flags, block_tables, seq_lens, scale, partition_size);
        return;
      default:
        STD_TORCH_CHECK(false, "unknown ByteV2 FA2-direct diagnostic mode");
    }
  }

  switch (q_heads_per_kv) {
    case 1:
      launch_byte_v2_paged_decode_attention_split_k_gqa4_fa2_like_no_fallback_no_outlier<
          Layout, UsePageUnsafeFlags, true, false, false, true, 1, 1>(
          output, exp_sums, max_logits, tmp_out, query, kv_cache,
          page_unsafe_flags, block_tables, seq_lens, scale, partition_size);
      return;
    case 2:
      launch_byte_v2_paged_decode_attention_split_k_gqa4_fa2_like_no_fallback_no_outlier<
          Layout, UsePageUnsafeFlags, true, false, false, true, 2, 2>(
          output, exp_sums, max_logits, tmp_out, query, kv_cache,
          page_unsafe_flags, block_tables, seq_lens, scale, partition_size);
      return;
    case 4:
      launch_byte_v2_paged_decode_attention_split_k_gqa4_fa2_like_no_fallback_no_outlier<
          Layout, UsePageUnsafeFlags, true, false, false, true, 4, 4>(
          output, exp_sums, max_logits, tmp_out, query, kv_cache,
          page_unsafe_flags, block_tables, seq_lens, scale, partition_size);
      return;
    case 8:
      launch_byte_v2_paged_decode_attention_split_k_gqa4_fa2_like_no_fallback_no_outlier<
          Layout, UsePageUnsafeFlags, true, false, false, true, 8, 8>(
          output, exp_sums, max_logits, tmp_out, query, kv_cache,
          page_unsafe_flags, block_tables, seq_lens, scale, partition_size);
      return;
    case 16:
      launch_byte_v2_paged_decode_attention_split_k_gqa4_fa2_like_no_fallback_no_outlier<
          Layout, UsePageUnsafeFlags, true, false, false, true, 16, 16>(
          output, exp_sums, max_logits, tmp_out, query, kv_cache,
          page_unsafe_flags, block_tables, seq_lens, scale, partition_size);
      return;
    case 32:
      launch_byte_v2_paged_decode_attention_split_k_gqa4_fa2_like_no_fallback_no_outlier<
          Layout, UsePageUnsafeFlags, true, false, false, true, 32, 16>(
          output, exp_sums, max_logits, tmp_out, query, kv_cache,
          page_unsafe_flags, block_tables, seq_lens, scale, partition_size);
      return;
    default:
      STD_TORCH_CHECK(false,
                      "ByteV2 FA2-direct grouped decode supports q_per_kv in "
                      "{1, 2, 4, 8, 16, 32}");
  }
}
}  // namespace

void byte_v2_reshape_and_cache(torch::stable::Tensor& key,
                               torch::stable::Tensor& value,
                               torch::stable::Tensor& kv_cache,
                               torch::stable::Tensor& slot_mapping,
                               int64_t codec_token_block,
                               int64_t codec_dim_block,
                               int64_t alloc_block_tokens) {
  using torch::headeronly::ScalarType;

  STD_TORCH_CHECK(key.device().is_cuda(), "key must be a CUDA tensor");
  STD_TORCH_CHECK(key.device() == value.device(),
                  "key and value must be on the same device");
  STD_TORCH_CHECK(key.device() == kv_cache.device(),
                  "key and kv_cache must be on the same device");
  STD_TORCH_CHECK(key.device() == slot_mapping.device(),
                  "key and slot_mapping must be on the same device");
  STD_TORCH_CHECK(key.dim() == 3, "key must be [num_tokens, heads, head_dim]");
  STD_TORCH_CHECK(value.dim() == 3,
                  "value must be [num_tokens, heads, head_dim_v]");
  STD_TORCH_CHECK(key.size(0) >= slot_mapping.size(0),
                  "key must contain all tokens in slot_mapping");
  STD_TORCH_CHECK(value.size(0) >= slot_mapping.size(0),
                  "value must contain all tokens in slot_mapping");
  check_byte_v2_tile_policy({codec_token_block, codec_dim_block,
                             alloc_block_tokens, alloc_block_tokens,
                             key.size(2), value.size(2)});

  STD_TORCH_CHECK(key.scalar_type() == ScalarType::BFloat16,
                  "ByteV2 cache update currently supports bf16 key tensors");
  STD_TORCH_CHECK(value.scalar_type() == key.scalar_type(),
                  "key and value dtypes must match");
  STD_TORCH_CHECK(kv_cache.scalar_type() == ScalarType::Byte,
                  "kv_cache must use uint8 storage");
  STD_TORCH_CHECK(slot_mapping.scalar_type() == ScalarType::Long,
                  "slot_mapping must be int64");
  STD_TORCH_CHECK(kv_cache.dim() == 2,
                  "kv_cache must be [num_blocks, page_size_bytes]");
  STD_TORCH_CHECK(kv_cache.stride(1) == 1,
                  "kv_cache page dimension must be contiguous");
  STD_TORCH_CHECK(key.size(1) == ByteV2DefaultLayout::NumKvHeadsValue,
                  "ByteV2 cache update currently supports 8 KV heads");
  STD_TORCH_CHECK(value.size(1) == key.size(1),
                  "key and value head counts must match");
  STD_TORCH_CHECK(key.size(2) == ByteV2DefaultPolicy::HeadDim,
                  "ByteV2 cache update currently supports head_dim=128");
  STD_TORCH_CHECK(value.size(2) == ByteV2DefaultPolicy::HeadDimV,
                  "ByteV2 cache update currently supports head_dim_v=128");
  STD_TORCH_CHECK(
      codec_token_block == ByteV2DefaultPolicy::CodecTokenBlock &&
          codec_dim_block == ByteV2DefaultPolicy::CodecDimBlock &&
          alloc_block_tokens == ByteV2DefaultPolicy::AllocBlockTokens,
      "ByteV2 cache update currently supports 16x16 codec tiles "
      "and block_size=16");
  STD_TORCH_CHECK(kv_cache.size(1) >= ByteV2DefaultLayout::PageSizeBytes,
                  "kv_cache page size is smaller than ByteV2 V5 layout");
  STD_TORCH_CHECK(kv_cache.stride(0) >= ByteV2DefaultLayout::PageSizeBytes,
                  "kv_cache block stride is smaller than ByteV2 V5 layout");

  const int64_t num_tokens = slot_mapping.size(0);
  if (num_tokens == 0) {
    return;
  }

  const torch::stable::accelerator::DeviceGuard device_guard(
      key.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream(key.get_device_index());

  constexpr int kClearThreads = 256;
  const int64_t clear_work_items =
      num_tokens * ByteV2DefaultLayout::AlignedMetadataBytes;
  const int clear_blocks =
      static_cast<int>((clear_work_items + kClearThreads - 1) / kClearThreads);
  byte_v2_clear_page_metadata_from_slots_kernel<<<clear_blocks, kClearThreads,
                                                  0, stream>>>(
      reinterpret_cast<uint8_t*>(kv_cache.mutable_data_ptr()),
      slot_mapping.const_data_ptr<int64_t>(), num_tokens, kv_cache.stride(0));

  constexpr int kThreads = ByteV2DefaultPolicy::CodecPackedElems;
  const int64_t token_blocks =
      (num_tokens + ByteV2DefaultPolicy::CodecTokenBlock - 1) /
      ByteV2DefaultPolicy::CodecTokenBlock;
  const dim3 grid(
      static_cast<unsigned int>(token_blocks),
      static_cast<unsigned int>(2 * ByteV2DefaultLayout::NumKvHeadsValue *
                                ByteV2DefaultPolicy::KDimTiles),
      1);
  byte_v2_reshape_and_cache_block_direct_kernel<false>
      <<<grid, kThreads, 0, stream>>>(
          reinterpret_cast<const uint16_t*>(key.const_data_ptr()),
          reinterpret_cast<const uint16_t*>(value.const_data_ptr()),
          reinterpret_cast<uint8_t*>(kv_cache.mutable_data_ptr()),
          slot_mapping.const_data_ptr<int64_t>(), num_tokens, key.size(1),
          key.stride(0), key.stride(1), key.stride(2), value.stride(0),
          value.stride(1), value.stride(2), kv_cache.stride(0));
  const cudaError_t err = cudaGetLastError();
  STD_TORCH_CHECK(err == cudaSuccess,
                  "byte_v2_reshape_and_cache kernel launch failed: ",
                  cudaGetErrorString(err));
}

void byte_v2_reshape_and_cache_sideband_high(
    torch::stable::Tensor& key, torch::stable::Tensor& value,
    torch::stable::Tensor& kv_cache, torch::stable::Tensor& slot_mapping,
    int64_t codec_token_block, int64_t codec_dim_block,
    int64_t alloc_block_tokens) {
  using torch::headeronly::ScalarType;

  STD_TORCH_CHECK(key.device().is_cuda(), "key must be a CUDA tensor");
  STD_TORCH_CHECK(key.device() == value.device(),
                  "key and value must be on the same device");
  STD_TORCH_CHECK(key.device() == kv_cache.device(),
                  "key and kv_cache must be on the same device");
  STD_TORCH_CHECK(key.device() == slot_mapping.device(),
                  "key and slot_mapping must be on the same device");
  STD_TORCH_CHECK(key.dim() == 3, "key must be [num_tokens, heads, head_dim]");
  STD_TORCH_CHECK(value.dim() == 3,
                  "value must be [num_tokens, heads, head_dim_v]");
  STD_TORCH_CHECK(key.size(0) >= slot_mapping.size(0),
                  "key must contain all tokens in slot_mapping");
  STD_TORCH_CHECK(value.size(0) >= slot_mapping.size(0),
                  "value must contain all tokens in slot_mapping");
  check_byte_v2_tile_policy({codec_token_block, codec_dim_block,
                             alloc_block_tokens, alloc_block_tokens,
                             key.size(2), value.size(2)});

  STD_TORCH_CHECK(key.scalar_type() == ScalarType::BFloat16,
                  "ByteV2 sideband-high cache update supports bf16 key "
                  "tensors");
  STD_TORCH_CHECK(value.scalar_type() == key.scalar_type(),
                  "key and value dtypes must match");
  STD_TORCH_CHECK(kv_cache.scalar_type() == ScalarType::Byte,
                  "kv_cache must use uint8 storage");
  STD_TORCH_CHECK(slot_mapping.scalar_type() == ScalarType::Long,
                  "slot_mapping must be int64");
  STD_TORCH_CHECK(kv_cache.dim() == 2,
                  "kv_cache must be [num_blocks, page_size_bytes]");
  STD_TORCH_CHECK(kv_cache.stride(1) == 1,
                  "kv_cache page dimension must be contiguous");
  STD_TORCH_CHECK(key.size(1) == ByteV2SidebandHighLayout::NumKvHeadsValue,
                  "ByteV2 sideband-high cache update currently supports 8 KV "
                  "heads");
  STD_TORCH_CHECK(value.size(1) == key.size(1),
                  "key and value head counts must match");
  STD_TORCH_CHECK(key.size(2) == ByteV2DefaultPolicy::HeadDim,
                  "ByteV2 sideband-high cache update supports head_dim=128");
  STD_TORCH_CHECK(value.size(2) == ByteV2DefaultPolicy::HeadDimV,
                  "ByteV2 sideband-high cache update supports head_dim_v=128");
  STD_TORCH_CHECK(
      codec_token_block == ByteV2DefaultPolicy::CodecTokenBlock &&
          codec_dim_block == ByteV2DefaultPolicy::CodecDimBlock &&
          alloc_block_tokens == ByteV2DefaultPolicy::AllocBlockTokens,
      "ByteV2 sideband-high cache update currently supports 16x16 codec "
      "tiles and block_size=16");
  STD_TORCH_CHECK(kv_cache.size(1) >= ByteV2SidebandHighLayout::PageSizeBytes,
                  "kv_cache page size is smaller than ByteV2 sideband-high "
                  "layout");
  STD_TORCH_CHECK(kv_cache.stride(0) >= ByteV2SidebandHighLayout::PageSizeBytes,
                  "kv_cache block stride is smaller than ByteV2 sideband-high "
                  "layout");

  const int64_t num_tokens = slot_mapping.size(0);
  if (num_tokens == 0) {
    return;
  }

  const torch::stable::accelerator::DeviceGuard device_guard(
      key.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream(key.get_device_index());

  constexpr int kClearThreads = 256;
  const int64_t clear_work_items =
      num_tokens * ByteV2SidebandHighLayout::AlignedMetadataBytes;
  const int clear_blocks =
      static_cast<int>((clear_work_items + kClearThreads - 1) / kClearThreads);
  byte_v2_clear_page_metadata_from_slots_kernel<<<clear_blocks, kClearThreads,
                                                  0, stream>>>(
      reinterpret_cast<uint8_t*>(kv_cache.mutable_data_ptr()),
      slot_mapping.const_data_ptr<int64_t>(), num_tokens, kv_cache.stride(0));

  constexpr int kThreads = ByteV2DefaultPolicy::CodecPackedElems;
  const int64_t token_blocks =
      (num_tokens + ByteV2DefaultPolicy::CodecTokenBlock - 1) /
      ByteV2DefaultPolicy::CodecTokenBlock;
  const dim3 grid(
      static_cast<unsigned int>(token_blocks),
      static_cast<unsigned int>(2 * ByteV2SidebandHighLayout::NumKvHeadsValue *
                                ByteV2DefaultPolicy::KDimTiles),
      1);
  byte_v2_reshape_and_cache_block_direct_kernel<true>
      <<<grid, kThreads, 0, stream>>>(
          reinterpret_cast<const uint16_t*>(key.const_data_ptr()),
          reinterpret_cast<const uint16_t*>(value.const_data_ptr()),
          reinterpret_cast<uint8_t*>(kv_cache.mutable_data_ptr()),
          slot_mapping.const_data_ptr<int64_t>(), num_tokens, key.size(1),
          key.stride(0), key.stride(1), key.stride(2), value.stride(0),
          value.stride(1), value.stride(2), kv_cache.stride(0));
  const cudaError_t err = cudaGetLastError();
  STD_TORCH_CHECK(err == cudaSuccess,
                  "byte_v2_reshape_and_cache_sideband_high kernel launch "
                  "failed: ",
                  cudaGetErrorString(err));
}

void byte_v2_reshape_and_cache_high_byte(torch::stable::Tensor& key,
                                         torch::stable::Tensor& value,
                                         torch::stable::Tensor& kv_cache,
                                         torch::stable::Tensor& slot_mapping,
                                         int64_t codec_token_block,
                                         int64_t codec_dim_block,
                                         int64_t alloc_block_tokens) {
  using torch::headeronly::ScalarType;

  STD_TORCH_CHECK(key.device().is_cuda(), "key must be a CUDA tensor");
  STD_TORCH_CHECK(key.device() == value.device(),
                  "key and value must be on the same device");
  STD_TORCH_CHECK(key.device() == kv_cache.device(),
                  "key and kv_cache must be on the same device");
  STD_TORCH_CHECK(key.device() == slot_mapping.device(),
                  "key and slot_mapping must be on the same device");
  STD_TORCH_CHECK(key.dim() == 3, "key must be [num_tokens, heads, head_dim]");
  STD_TORCH_CHECK(value.dim() == 3,
                  "value must be [num_tokens, heads, head_dim_v]");
  STD_TORCH_CHECK(key.size(0) >= slot_mapping.size(0),
                  "key must contain all tokens in slot_mapping");
  STD_TORCH_CHECK(value.size(0) >= slot_mapping.size(0),
                  "value must contain all tokens in slot_mapping");
  check_byte_v2_tile_policy({codec_token_block, codec_dim_block,
                             alloc_block_tokens, alloc_block_tokens,
                             key.size(2), value.size(2)});

  STD_TORCH_CHECK(key.scalar_type() == ScalarType::BFloat16,
                  "ByteV2 high-byte cache update supports bf16 key tensors");
  STD_TORCH_CHECK(value.scalar_type() == key.scalar_type(),
                  "key and value dtypes must match");
  STD_TORCH_CHECK(kv_cache.scalar_type() == ScalarType::Byte,
                  "kv_cache must use uint8 storage");
  STD_TORCH_CHECK(slot_mapping.scalar_type() == ScalarType::Long,
                  "slot_mapping must be int64");
  STD_TORCH_CHECK(kv_cache.dim() == 2,
                  "kv_cache must be [num_blocks, page_size_bytes]");
  STD_TORCH_CHECK(kv_cache.stride(1) == 1,
                  "kv_cache page dimension must be contiguous");
  STD_TORCH_CHECK(key.size(1) == ByteV2HighByteLayout::NumKvHeadsValue,
                  "ByteV2 high-byte cache update currently supports 8 KV "
                  "heads");
  STD_TORCH_CHECK(value.size(1) == key.size(1),
                  "key and value head counts must match");
  STD_TORCH_CHECK(key.size(2) == ByteV2DefaultPolicy::HeadDim,
                  "ByteV2 high-byte cache update supports head_dim=128");
  STD_TORCH_CHECK(value.size(2) == ByteV2DefaultPolicy::HeadDimV,
                  "ByteV2 high-byte cache update supports head_dim_v=128");
  STD_TORCH_CHECK(
      codec_token_block == ByteV2DefaultPolicy::CodecTokenBlock &&
          codec_dim_block == ByteV2DefaultPolicy::CodecDimBlock &&
          alloc_block_tokens == ByteV2DefaultPolicy::AllocBlockTokens,
      "ByteV2 high-byte cache update currently supports 16x16 codec tiles "
      "and block_size=16");
  STD_TORCH_CHECK(kv_cache.size(1) >= ByteV2HighByteLayout::PageSizeBytes,
                  "kv_cache page size is smaller than ByteV2 high-byte "
                  "layout");
  STD_TORCH_CHECK(kv_cache.stride(0) >= ByteV2HighByteLayout::PageSizeBytes,
                  "kv_cache block stride is smaller than ByteV2 high-byte "
                  "layout");

  const int64_t num_tokens = slot_mapping.size(0);
  if (num_tokens == 0) {
    return;
  }

  const torch::stable::accelerator::DeviceGuard device_guard(
      key.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream(key.get_device_index());

  constexpr int kClearThreads = 256;
  const int64_t clear_work_items =
      num_tokens * ByteV2HighByteLayout::AlignedMetadataBytes;
  const int clear_blocks =
      static_cast<int>((clear_work_items + kClearThreads - 1) / kClearThreads);
  byte_v2_clear_page_metadata_from_slots_kernel<<<clear_blocks, kClearThreads,
                                                  0, stream>>>(
      reinterpret_cast<uint8_t*>(kv_cache.mutable_data_ptr()),
      slot_mapping.const_data_ptr<int64_t>(), num_tokens, kv_cache.stride(0));

  constexpr int kThreads = ByteV2DefaultPolicy::CodecPackedElems;
  const int64_t token_blocks =
      (num_tokens + ByteV2DefaultPolicy::CodecTokenBlock - 1) /
      ByteV2DefaultPolicy::CodecTokenBlock;
  const dim3 grid(
      static_cast<unsigned int>(token_blocks),
      static_cast<unsigned int>(2 * ByteV2HighByteLayout::NumKvHeadsValue *
                                ByteV2DefaultPolicy::KDimTiles),
      1);
  byte_v2_reshape_and_cache_high_byte_kernel<<<grid, kThreads, 0, stream>>>(
      reinterpret_cast<const uint16_t*>(key.const_data_ptr()),
      reinterpret_cast<const uint16_t*>(value.const_data_ptr()),
      reinterpret_cast<uint8_t*>(kv_cache.mutable_data_ptr()),
      slot_mapping.const_data_ptr<int64_t>(), num_tokens, key.size(1),
      key.stride(0), key.stride(1), key.stride(2), value.stride(0),
      value.stride(1), value.stride(2), kv_cache.stride(0));
  const cudaError_t err = cudaGetLastError();
  STD_TORCH_CHECK(err == cudaSuccess,
                  "byte_v2_reshape_and_cache_high_byte kernel launch failed: ",
                  cudaGetErrorString(err));
}

void byte_v2_update_cache_single_token(
    torch::stable::Tensor& key, torch::stable::Tensor& value,
    torch::stable::Tensor& kv_cache, torch::stable::Tensor& slot_mapping,
    torch::stable::Tensor& page_unsafe_flags, int64_t codec_token_block,
    int64_t codec_dim_block, int64_t alloc_block_tokens) {
  using torch::headeronly::ScalarType;

  STD_TORCH_CHECK(key.device().is_cuda(), "key must be a CUDA tensor");
  STD_TORCH_CHECK(key.device() == value.device(),
                  "key and value must be on the same device");
  STD_TORCH_CHECK(key.device() == kv_cache.device(),
                  "key and kv_cache must be on the same device");
  STD_TORCH_CHECK(key.device() == slot_mapping.device(),
                  "key and slot_mapping must be on the same device");
  STD_TORCH_CHECK(key.device() == page_unsafe_flags.device(),
                  "key and page_unsafe_flags must be on the same device");
  STD_TORCH_CHECK(key.dim() == 3, "key must be [1, heads, head_dim]");
  STD_TORCH_CHECK(value.dim() == 3, "value must be [1, heads, head_dim_v]");
  STD_TORCH_CHECK(slot_mapping.dim() == 1, "slot_mapping must be [num_tokens]");
  STD_TORCH_CHECK(slot_mapping.size(0) == 1,
                  "single-token update requires one slot mapping");
  STD_TORCH_CHECK(key.size(0) >= 1, "key must contain one token");
  STD_TORCH_CHECK(value.size(0) >= 1, "value must contain one token");
  check_byte_v2_tile_policy({codec_token_block, codec_dim_block,
                             alloc_block_tokens, alloc_block_tokens,
                             key.size(2), value.size(2)});

  STD_TORCH_CHECK(key.scalar_type() == ScalarType::BFloat16,
                  "ByteV2 single-token update supports bf16 key tensors");
  STD_TORCH_CHECK(value.scalar_type() == key.scalar_type(),
                  "key and value dtypes must match");
  STD_TORCH_CHECK(kv_cache.scalar_type() == ScalarType::Byte,
                  "kv_cache must use uint8 storage");
  STD_TORCH_CHECK(slot_mapping.scalar_type() == ScalarType::Long,
                  "slot_mapping must be int64");
  STD_TORCH_CHECK(page_unsafe_flags.scalar_type() == ScalarType::Int,
                  "page_unsafe_flags must be int32");
  STD_TORCH_CHECK(page_unsafe_flags.dim() == 1,
                  "page_unsafe_flags must be a 1D tensor");
  STD_TORCH_CHECK(kv_cache.dim() == 2,
                  "kv_cache must be [num_blocks, page_size_bytes]");
  STD_TORCH_CHECK(kv_cache.stride(1) == 1,
                  "kv_cache page dimension must be contiguous");
  STD_TORCH_CHECK(key.size(1) == ByteV2DefaultLayout::NumKvHeadsValue,
                  "ByteV2 single-token update currently supports 8 KV heads");
  STD_TORCH_CHECK(value.size(1) == key.size(1),
                  "key and value head counts must match");
  STD_TORCH_CHECK(key.size(2) == ByteV2DefaultPolicy::HeadDim,
                  "ByteV2 single-token update currently supports head_dim=128");
  STD_TORCH_CHECK(value.size(2) == ByteV2DefaultPolicy::HeadDimV,
                  "ByteV2 single-token update supports head_dim_v=128");
  STD_TORCH_CHECK(
      codec_token_block == ByteV2DefaultPolicy::CodecTokenBlock &&
          codec_dim_block == ByteV2DefaultPolicy::CodecDimBlock &&
          alloc_block_tokens == ByteV2DefaultPolicy::AllocBlockTokens,
      "ByteV2 single-token update currently supports 16x16 codec tiles "
      "and block_size=16");
  STD_TORCH_CHECK(kv_cache.size(1) >= ByteV2DefaultLayout::PageSizeBytes,
                  "kv_cache page size is smaller than ByteV2 V5 layout");
  STD_TORCH_CHECK(kv_cache.stride(0) >= ByteV2DefaultLayout::PageSizeBytes,
                  "kv_cache block stride is smaller than ByteV2 V5 layout");
  STD_TORCH_CHECK(page_unsafe_flags.numel() == 0 ||
                      page_unsafe_flags.size(0) >= kv_cache.size(0),
                  "page_unsafe_flags must be empty or cover every cache block");

  const torch::stable::accelerator::DeviceGuard device_guard(
      key.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream(key.get_device_index());

  constexpr int kThreads = ByteV2DefaultPolicy::CodecPackedElems;
  const dim3 grid(
      static_cast<unsigned int>(2 * ByteV2DefaultLayout::NumKvHeadsValue *
                                ByteV2DefaultPolicy::KDimTiles),
      1, 1);
  int32_t* page_unsafe_flags_ptr =
      page_unsafe_flags.numel() == 0
          ? nullptr
          : page_unsafe_flags.mutable_data_ptr<int32_t>();
  byte_v2_prepare_single_token_page_kernel<<<1, 256, 0, stream>>>(
      reinterpret_cast<uint8_t*>(kv_cache.mutable_data_ptr()),
      slot_mapping.const_data_ptr<int64_t>(), page_unsafe_flags_ptr,
      kv_cache.size(0), kv_cache.stride(0));
  byte_v2_update_cache_single_token_kernel<<<grid, kThreads, 0, stream>>>(
      reinterpret_cast<const uint16_t*>(key.const_data_ptr()),
      reinterpret_cast<const uint16_t*>(value.const_data_ptr()),
      reinterpret_cast<uint8_t*>(kv_cache.mutable_data_ptr()),
      slot_mapping.const_data_ptr<int64_t>(), page_unsafe_flags_ptr,
      key.size(1), key.stride(0), key.stride(1), key.stride(2), value.stride(0),
      value.stride(1), value.stride(2), kv_cache.size(0), kv_cache.stride(0));
  const cudaError_t err = cudaGetLastError();
  STD_TORCH_CHECK(err == cudaSuccess,
                  "byte_v2_update_cache_single_token kernel launch failed: ",
                  cudaGetErrorString(err));
}

void byte_v2_append_raw_staging(
    torch::stable::Tensor& key, torch::stable::Tensor& value,
    torch::stable::Tensor& raw_staging, torch::stable::Tensor& slot_mapping,
    torch::stable::Tensor& block_to_staging_slot, int64_t codec_token_block,
    int64_t codec_dim_block, int64_t alloc_block_tokens) {
  using torch::headeronly::ScalarType;

  STD_TORCH_CHECK(key.device().is_cuda(), "key must be a CUDA tensor");
  STD_TORCH_CHECK(key.device() == value.device(),
                  "key and value must be on the same device");
  STD_TORCH_CHECK(key.device() == raw_staging.device(),
                  "key and raw_staging must be on the same device");
  STD_TORCH_CHECK(key.device() == slot_mapping.device(),
                  "key and slot_mapping must be on the same device");
  STD_TORCH_CHECK(key.device() == block_to_staging_slot.device(),
                  "key and block_to_staging_slot must be on the same device");
  STD_TORCH_CHECK(key.dim() == 3, "key must be [num_tokens, heads, head_dim]");
  STD_TORCH_CHECK(value.dim() == 3,
                  "value must be [num_tokens, heads, head_dim_v]");
  STD_TORCH_CHECK(key.size(0) >= slot_mapping.size(0),
                  "key must contain all tokens in slot_mapping");
  STD_TORCH_CHECK(value.size(0) >= slot_mapping.size(0),
                  "value must contain all tokens in slot_mapping");
  check_byte_v2_tile_policy({codec_token_block, codec_dim_block,
                             alloc_block_tokens, alloc_block_tokens,
                             key.size(2), value.size(2)});

  STD_TORCH_CHECK(key.scalar_type() == ScalarType::BFloat16,
                  "ByteV2 raw staging currently supports bf16 key tensors");
  STD_TORCH_CHECK(value.scalar_type() == key.scalar_type(),
                  "key and value dtypes must match");
  STD_TORCH_CHECK(raw_staging.scalar_type() == ScalarType::Byte,
                  "raw_staging must use uint8 storage");
  STD_TORCH_CHECK(slot_mapping.scalar_type() == ScalarType::Long,
                  "slot_mapping must be int64");
  STD_TORCH_CHECK(block_to_staging_slot.scalar_type() == ScalarType::Int,
                  "block_to_staging_slot must be int32");
  STD_TORCH_CHECK(raw_staging.dim() == 2,
                  "raw_staging must be [num_slots, slot_size_bytes]");
  STD_TORCH_CHECK(raw_staging.stride(1) == 1,
                  "raw_staging slot dimension must be contiguous");
  STD_TORCH_CHECK(block_to_staging_slot.dim() == 1,
                  "block_to_staging_slot must be [num_blocks]");
  STD_TORCH_CHECK(key.size(1) == ByteV2DefaultRawStagingLayout::NumKvHeadsValue,
                  "ByteV2 raw staging currently supports 8 KV heads");
  STD_TORCH_CHECK(value.size(1) == key.size(1),
                  "key and value head counts must match");
  STD_TORCH_CHECK(key.size(2) == ByteV2DefaultPolicy::HeadDim,
                  "ByteV2 raw staging currently supports head_dim=128");
  STD_TORCH_CHECK(value.size(2) == ByteV2DefaultPolicy::HeadDimV,
                  "ByteV2 raw staging currently supports head_dim_v=128");
  STD_TORCH_CHECK(
      codec_token_block == ByteV2DefaultPolicy::CodecTokenBlock &&
          codec_dim_block == ByteV2DefaultPolicy::CodecDimBlock &&
          alloc_block_tokens == ByteV2DefaultPolicy::AllocBlockTokens,
      "ByteV2 raw staging currently supports 16x16 codec tiles "
      "and block_size=16");
  STD_TORCH_CHECK(
      raw_staging.size(1) >= ByteV2DefaultRawStagingLayout::SlotSizeBytes,
      "raw_staging slot size is smaller than ByteV2 raw staging layout");
  STD_TORCH_CHECK(
      raw_staging.stride(0) >= ByteV2DefaultRawStagingLayout::SlotSizeBytes,
      "raw_staging slot stride is smaller than ByteV2 raw staging layout");

  const int64_t num_tokens = slot_mapping.size(0);
  if (num_tokens == 0 || raw_staging.size(0) == 0 ||
      block_to_staging_slot.size(0) == 0) {
    return;
  }

  constexpr int kThreads = 256;
  constexpr int kWorkPerToken = 2 *
                                ByteV2DefaultRawStagingLayout::NumKvHeadsValue *
                                ByteV2DefaultPolicy::HeadDim;
  const int64_t work_items = num_tokens * kWorkPerToken;
  const int64_t blocks64 = (work_items + kThreads - 1) / kThreads;
  const int blocks = static_cast<int>(blocks64 > 65535 ? 65535 : blocks64);

  const torch::stable::accelerator::DeviceGuard device_guard(
      key.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream(key.get_device_index());

  byte_v2_append_raw_staging_kernel<<<blocks, kThreads, 0, stream>>>(
      reinterpret_cast<const uint16_t*>(key.const_data_ptr()),
      reinterpret_cast<const uint16_t*>(value.const_data_ptr()),
      reinterpret_cast<uint8_t*>(raw_staging.mutable_data_ptr()),
      slot_mapping.const_data_ptr<int64_t>(),
      block_to_staging_slot.const_data_ptr<int32_t>(), num_tokens, key.size(1),
      key.stride(0), key.stride(1), key.stride(2), value.stride(0),
      value.stride(1), value.stride(2), raw_staging.stride(0),
      raw_staging.size(0), block_to_staging_slot.size(0));
  const cudaError_t err = cudaGetLastError();
  STD_TORCH_CHECK(err == cudaSuccess,
                  "byte_v2_append_raw_staging kernel launch failed: ",
                  cudaGetErrorString(err));
}

void byte_v2_prepare_raw_staging(
    torch::stable::Tensor& slot_mapping,
    torch::stable::Tensor& block_to_staging_slot,
    torch::stable::Tensor& staging_to_physical_block,
    torch::stable::Tensor& valid_rows, torch::stable::Tensor& next_staging_slot,
    torch::stable::Tensor& overflow, int64_t alloc_block_tokens) {
  using torch::headeronly::ScalarType;

  STD_TORCH_CHECK(slot_mapping.device().is_cuda(),
                  "slot_mapping must be a CUDA tensor");
  STD_TORCH_CHECK(slot_mapping.device() == block_to_staging_slot.device(),
                  "slot_mapping and block_to_staging_slot must be on the same "
                  "device");
  STD_TORCH_CHECK(slot_mapping.device() == staging_to_physical_block.device(),
                  "slot_mapping and staging_to_physical_block must be on the "
                  "same device");
  STD_TORCH_CHECK(slot_mapping.device() == valid_rows.device(),
                  "slot_mapping and valid_rows must be on the same device");
  STD_TORCH_CHECK(slot_mapping.device() == next_staging_slot.device(),
                  "slot_mapping and next_staging_slot must be on the same "
                  "device");
  STD_TORCH_CHECK(slot_mapping.device() == overflow.device(),
                  "slot_mapping and overflow must be on the same device");
  STD_TORCH_CHECK(slot_mapping.scalar_type() == ScalarType::Long,
                  "slot_mapping must be int64");
  STD_TORCH_CHECK(block_to_staging_slot.scalar_type() == ScalarType::Int,
                  "block_to_staging_slot must be int32");
  STD_TORCH_CHECK(staging_to_physical_block.scalar_type() == ScalarType::Int,
                  "staging_to_physical_block must be int32");
  STD_TORCH_CHECK(valid_rows.scalar_type() == ScalarType::Int,
                  "valid_rows must be int32");
  STD_TORCH_CHECK(next_staging_slot.scalar_type() == ScalarType::Int,
                  "next_staging_slot must be int32");
  STD_TORCH_CHECK(overflow.scalar_type() == ScalarType::Int,
                  "overflow must be int32");
  STD_TORCH_CHECK(slot_mapping.dim() == 1, "slot_mapping must be [num_tokens]");
  STD_TORCH_CHECK(block_to_staging_slot.dim() == 1,
                  "block_to_staging_slot must be [num_blocks]");
  STD_TORCH_CHECK(staging_to_physical_block.dim() == 1,
                  "staging_to_physical_block must be [num_staging_slots]");
  STD_TORCH_CHECK(valid_rows.dim() == 1, "valid_rows must be [num_slots]");
  STD_TORCH_CHECK(next_staging_slot.dim() == 1,
                  "next_staging_slot must be [1]");
  STD_TORCH_CHECK(overflow.dim() == 1, "overflow must be [1]");
  STD_TORCH_CHECK(valid_rows.size(0) >= staging_to_physical_block.size(0),
                  "valid_rows must contain all staging slots");
  STD_TORCH_CHECK(next_staging_slot.size(0) >= 1,
                  "next_staging_slot must contain one counter");
  STD_TORCH_CHECK(overflow.size(0) >= 1, "overflow must contain one flag");
  STD_TORCH_CHECK(alloc_block_tokens == ByteV2DefaultPolicy::AllocBlockTokens,
                  "ByteV2 raw staging prepare currently supports "
                  "block_size=16");

  const int64_t num_tokens = slot_mapping.size(0);
  if (num_tokens == 0 || block_to_staging_slot.size(0) == 0 ||
      staging_to_physical_block.size(0) == 0) {
    return;
  }

  const torch::stable::accelerator::DeviceGuard device_guard(
      slot_mapping.get_device_index());
  const cudaStream_t stream =
      get_current_cuda_stream(slot_mapping.get_device_index());
  cudaMemsetAsync(overflow.mutable_data_ptr<int32_t>(), 0, sizeof(int32_t),
                  stream);

  constexpr int kThreads = 256;
  const int64_t blocks64 = (num_tokens + kThreads - 1) / kThreads;
  const int blocks = static_cast<int>(blocks64 > 65535 ? 65535 : blocks64);
  byte_v2_prepare_raw_staging_kernel<<<blocks, kThreads, 0, stream>>>(
      slot_mapping.const_data_ptr<int64_t>(),
      block_to_staging_slot.mutable_data_ptr<int32_t>(),
      staging_to_physical_block.mutable_data_ptr<int32_t>(),
      valid_rows.mutable_data_ptr<int32_t>(),
      next_staging_slot.mutable_data_ptr<int32_t>(),
      overflow.mutable_data_ptr<int32_t>(), num_tokens,
      block_to_staging_slot.size(0), staging_to_physical_block.size(0));
  const cudaError_t err = cudaGetLastError();
  STD_TORCH_CHECK(err == cudaSuccess,
                  "byte_v2_prepare_raw_staging kernel launch failed: ",
                  cudaGetErrorString(err));
}

void byte_v2_hydrate_raw_staging_from_cache(
    torch::stable::Tensor& raw_staging, torch::stable::Tensor& kv_cache,
    torch::stable::Tensor& staging_to_physical_block,
    torch::stable::Tensor& valid_rows, int64_t codec_token_block,
    int64_t codec_dim_block, int64_t alloc_block_tokens) {
  using torch::headeronly::ScalarType;

  STD_TORCH_CHECK(raw_staging.device().is_cuda(),
                  "raw_staging must be a CUDA tensor");
  STD_TORCH_CHECK(raw_staging.device() == kv_cache.device(),
                  "raw_staging and kv_cache must be on the same device");
  STD_TORCH_CHECK(raw_staging.device() == staging_to_physical_block.device(),
                  "raw_staging and staging_to_physical_block must be on the "
                  "same device");
  STD_TORCH_CHECK(raw_staging.device() == valid_rows.device(),
                  "raw_staging and valid_rows must be on the same device");
  check_byte_v2_tile_policy({codec_token_block, codec_dim_block,
                             alloc_block_tokens, alloc_block_tokens,
                             ByteV2DefaultPolicy::HeadDim,
                             ByteV2DefaultPolicy::HeadDimV});
  STD_TORCH_CHECK(raw_staging.scalar_type() == ScalarType::Byte,
                  "raw_staging must use uint8 storage");
  STD_TORCH_CHECK(kv_cache.scalar_type() == ScalarType::Byte,
                  "kv_cache must use uint8 storage");
  STD_TORCH_CHECK(staging_to_physical_block.scalar_type() == ScalarType::Int,
                  "staging_to_physical_block must be int32");
  STD_TORCH_CHECK(valid_rows.scalar_type() == ScalarType::Int,
                  "valid_rows must be int32");
  STD_TORCH_CHECK(raw_staging.dim() == 2,
                  "raw_staging must be [num_slots, slot_size_bytes]");
  STD_TORCH_CHECK(kv_cache.dim() == 2,
                  "kv_cache must be [num_blocks, page_size_bytes]");
  STD_TORCH_CHECK(staging_to_physical_block.dim() == 1,
                  "staging_to_physical_block must be [num_slots]");
  STD_TORCH_CHECK(valid_rows.dim() == 1, "valid_rows must be [num_slots]");
  STD_TORCH_CHECK(valid_rows.size(0) >= staging_to_physical_block.size(0),
                  "valid_rows must contain all staging slots");
  STD_TORCH_CHECK(raw_staging.size(0) >= staging_to_physical_block.size(0),
                  "raw_staging must contain all staging slots");
  STD_TORCH_CHECK(raw_staging.stride(1) == 1,
                  "raw_staging slot dimension must be contiguous");
  STD_TORCH_CHECK(kv_cache.stride(1) == 1,
                  "kv_cache page dimension must be contiguous");
  STD_TORCH_CHECK(
      codec_token_block == ByteV2DefaultPolicy::CodecTokenBlock &&
          codec_dim_block == ByteV2DefaultPolicy::CodecDimBlock &&
          alloc_block_tokens == ByteV2DefaultPolicy::AllocBlockTokens,
      "ByteV2 raw staging hydrate currently supports 16x16 codec tiles "
      "and block_size=16");
  STD_TORCH_CHECK(
      raw_staging.size(1) >= ByteV2DefaultRawStagingLayout::SlotSizeBytes,
      "raw_staging slot size is smaller than ByteV2 raw staging layout");
  STD_TORCH_CHECK(
      raw_staging.stride(0) >= ByteV2DefaultRawStagingLayout::SlotSizeBytes,
      "raw_staging slot stride is smaller than ByteV2 raw staging layout");
  STD_TORCH_CHECK(kv_cache.size(1) >= ByteV2DefaultLayout::PageSizeBytes,
                  "kv_cache page size is smaller than ByteV2 V5 layout");
  STD_TORCH_CHECK(kv_cache.stride(0) >= ByteV2DefaultLayout::PageSizeBytes,
                  "kv_cache block stride is smaller than ByteV2 V5 layout");

  const int64_t num_staging_slots = staging_to_physical_block.size(0);
  if (num_staging_slots == 0) {
    return;
  }

  constexpr int kThreads = 256;
  const int64_t work_items =
      num_staging_slots * 2 * ByteV2DefaultLayout::NumKvHeadsValue *
      ByteV2DefaultPolicy::AllocBlockTokens * ByteV2DefaultPolicy::HeadDim;
  const int64_t blocks64 = (work_items + kThreads - 1) / kThreads;
  const int blocks = static_cast<int>(blocks64 > 65535 ? 65535 : blocks64);

  const torch::stable::accelerator::DeviceGuard device_guard(
      raw_staging.get_device_index());
  const cudaStream_t stream =
      get_current_cuda_stream(raw_staging.get_device_index());
  byte_v2_hydrate_raw_staging_from_cache_kernel<<<blocks, kThreads, 0,
                                                  stream>>>(
      reinterpret_cast<uint8_t*>(raw_staging.mutable_data_ptr()),
      reinterpret_cast<const uint8_t*>(kv_cache.const_data_ptr()),
      staging_to_physical_block.const_data_ptr<int32_t>(),
      valid_rows.const_data_ptr<int32_t>(), num_staging_slots,
      raw_staging.stride(0), kv_cache.stride(0));
  const cudaError_t err = cudaGetLastError();
  STD_TORCH_CHECK(err == cudaSuccess,
                  "byte_v2_hydrate_raw_staging_from_cache kernel launch "
                  "failed: ",
                  cudaGetErrorString(err));
}

void byte_v2_release_raw_staging(
    torch::stable::Tensor& block_to_staging_slot,
    torch::stable::Tensor& staging_to_physical_block,
    torch::stable::Tensor& valid_rows, torch::stable::Tensor& next_staging_slot,
    torch::stable::Tensor& overflow) {
  using torch::headeronly::ScalarType;

  STD_TORCH_CHECK(block_to_staging_slot.device().is_cuda(),
                  "block_to_staging_slot must be a CUDA tensor");
  STD_TORCH_CHECK(
      block_to_staging_slot.device() == staging_to_physical_block.device(),
      "block_to_staging_slot and staging_to_physical_block must "
      "be on the same device");
  STD_TORCH_CHECK(block_to_staging_slot.device() == valid_rows.device(),
                  "block_to_staging_slot and valid_rows must be on the same "
                  "device");
  STD_TORCH_CHECK(block_to_staging_slot.device() == next_staging_slot.device(),
                  "block_to_staging_slot and next_staging_slot must be on the "
                  "same device");
  STD_TORCH_CHECK(block_to_staging_slot.device() == overflow.device(),
                  "block_to_staging_slot and overflow must be on the same "
                  "device");
  STD_TORCH_CHECK(block_to_staging_slot.scalar_type() == ScalarType::Int,
                  "block_to_staging_slot must be int32");
  STD_TORCH_CHECK(staging_to_physical_block.scalar_type() == ScalarType::Int,
                  "staging_to_physical_block must be int32");
  STD_TORCH_CHECK(valid_rows.scalar_type() == ScalarType::Int,
                  "valid_rows must be int32");
  STD_TORCH_CHECK(next_staging_slot.scalar_type() == ScalarType::Int,
                  "next_staging_slot must be int32");
  STD_TORCH_CHECK(overflow.scalar_type() == ScalarType::Int,
                  "overflow must be int32");
  STD_TORCH_CHECK(block_to_staging_slot.dim() == 1,
                  "block_to_staging_slot must be [num_blocks]");
  STD_TORCH_CHECK(staging_to_physical_block.dim() == 1,
                  "staging_to_physical_block must be [num_staging_slots]");
  STD_TORCH_CHECK(valid_rows.dim() == 1, "valid_rows must be [num_slots]");
  STD_TORCH_CHECK(next_staging_slot.dim() == 1,
                  "next_staging_slot must be [1]");
  STD_TORCH_CHECK(overflow.dim() == 1, "overflow must be [1]");
  STD_TORCH_CHECK(valid_rows.size(0) >= staging_to_physical_block.size(0),
                  "valid_rows must contain all staging slots");
  STD_TORCH_CHECK(next_staging_slot.size(0) >= 1,
                  "next_staging_slot must contain one counter");
  STD_TORCH_CHECK(overflow.size(0) >= 1, "overflow must contain one flag");

  const torch::stable::accelerator::DeviceGuard device_guard(
      block_to_staging_slot.get_device_index());
  const cudaStream_t stream =
      get_current_cuda_stream(block_to_staging_slot.get_device_index());

  const int64_t num_staging_slots = staging_to_physical_block.size(0);
  if (num_staging_slots == 0) {
    cudaMemsetAsync(next_staging_slot.mutable_data_ptr<int32_t>(), 0,
                    sizeof(int32_t), stream);
    cudaMemsetAsync(overflow.mutable_data_ptr<int32_t>(), 0, sizeof(int32_t),
                    stream);
    return;
  }

  constexpr int kThreads = 256;
  const int64_t blocks64 = (num_staging_slots + kThreads - 1) / kThreads;
  const int blocks = static_cast<int>(blocks64 > 65535 ? 65535 : blocks64);
  byte_v2_release_raw_staging_kernel<<<blocks, kThreads, 0, stream>>>(
      block_to_staging_slot.mutable_data_ptr<int32_t>(),
      staging_to_physical_block.mutable_data_ptr<int32_t>(),
      valid_rows.mutable_data_ptr<int32_t>(),
      next_staging_slot.mutable_data_ptr<int32_t>(),
      overflow.mutable_data_ptr<int32_t>(), num_staging_slots,
      block_to_staging_slot.size(0));
  const cudaError_t err = cudaGetLastError();
  STD_TORCH_CHECK(err == cudaSuccess,
                  "byte_v2_release_raw_staging kernel launch failed: ",
                  cudaGetErrorString(err));
}

void byte_v2_release_raw_staging_and_update_flags(
    torch::stable::Tensor& block_to_staging_slot,
    torch::stable::Tensor& staging_to_physical_block,
    torch::stable::Tensor& valid_rows, torch::stable::Tensor& next_staging_slot,
    torch::stable::Tensor& overflow, torch::stable::Tensor& page_unsafe_flags,
    torch::stable::Tensor& kv_cache, const std::vector<int64_t>& tile_policy) {
  using torch::headeronly::ScalarType;

  STD_TORCH_CHECK(block_to_staging_slot.device().is_cuda(),
                  "block_to_staging_slot must be a CUDA tensor");
  STD_TORCH_CHECK(
      block_to_staging_slot.device() == staging_to_physical_block.device(),
      "block_to_staging_slot and staging_to_physical_block must "
      "be on the same device");
  STD_TORCH_CHECK(block_to_staging_slot.device() == valid_rows.device(),
                  "block_to_staging_slot and valid_rows must be on the same "
                  "device");
  STD_TORCH_CHECK(block_to_staging_slot.device() == next_staging_slot.device(),
                  "block_to_staging_slot and next_staging_slot must be on the "
                  "same device");
  STD_TORCH_CHECK(block_to_staging_slot.device() == overflow.device(),
                  "block_to_staging_slot and overflow must be on the same "
                  "device");
  STD_TORCH_CHECK(block_to_staging_slot.device() == page_unsafe_flags.device(),
                  "block_to_staging_slot and page_unsafe_flags must be on the "
                  "same device");
  STD_TORCH_CHECK(block_to_staging_slot.device() == kv_cache.device(),
                  "block_to_staging_slot and kv_cache must be on the same "
                  "device");
  check_byte_v2_tile_policy(tile_policy);
  STD_TORCH_CHECK(block_to_staging_slot.scalar_type() == ScalarType::Int,
                  "block_to_staging_slot must be int32");
  STD_TORCH_CHECK(staging_to_physical_block.scalar_type() == ScalarType::Int,
                  "staging_to_physical_block must be int32");
  STD_TORCH_CHECK(valid_rows.scalar_type() == ScalarType::Int,
                  "valid_rows must be int32");
  STD_TORCH_CHECK(next_staging_slot.scalar_type() == ScalarType::Int,
                  "next_staging_slot must be int32");
  STD_TORCH_CHECK(overflow.scalar_type() == ScalarType::Int,
                  "overflow must be int32");
  STD_TORCH_CHECK(page_unsafe_flags.scalar_type() == ScalarType::Int,
                  "page_unsafe_flags must be int32");
  STD_TORCH_CHECK(kv_cache.scalar_type() == ScalarType::Byte,
                  "kv_cache must use uint8 storage");
  STD_TORCH_CHECK(block_to_staging_slot.dim() == 1,
                  "block_to_staging_slot must be [num_blocks]");
  STD_TORCH_CHECK(staging_to_physical_block.dim() == 1,
                  "staging_to_physical_block must be [num_staging_slots]");
  STD_TORCH_CHECK(valid_rows.dim() == 1, "valid_rows must be [num_slots]");
  STD_TORCH_CHECK(next_staging_slot.dim() == 1,
                  "next_staging_slot must be [1]");
  STD_TORCH_CHECK(overflow.dim() == 1, "overflow must be [1]");
  STD_TORCH_CHECK(page_unsafe_flags.dim() == 1,
                  "page_unsafe_flags must be [num_blocks]");
  STD_TORCH_CHECK(kv_cache.dim() == 2,
                  "kv_cache must be [num_blocks, page_size_bytes]");
  STD_TORCH_CHECK(valid_rows.size(0) >= staging_to_physical_block.size(0),
                  "valid_rows must contain all staging slots");
  STD_TORCH_CHECK(next_staging_slot.size(0) >= 1,
                  "next_staging_slot must contain one counter");
  STD_TORCH_CHECK(overflow.size(0) >= 1, "overflow must contain one flag");
  STD_TORCH_CHECK(block_to_staging_slot.size(0) == kv_cache.size(0),
                  "block_to_staging_slot must cover every cache block");
  STD_TORCH_CHECK(page_unsafe_flags.size(0) >= kv_cache.size(0),
                  "page_unsafe_flags must cover every cache block");
  STD_TORCH_CHECK(page_unsafe_flags.stride(0) == 1,
                  "page_unsafe_flags must be contiguous");
  STD_TORCH_CHECK(kv_cache.stride(1) == 1,
                  "kv_cache page dimension must be contiguous");
  STD_TORCH_CHECK(kv_cache.size(1) >= ByteV2DefaultLayout::PageSizeBytes,
                  "kv_cache page size is smaller than ByteV2 V5 layout");
  STD_TORCH_CHECK(kv_cache.stride(0) >= ByteV2DefaultLayout::PageSizeBytes,
                  "kv_cache block stride is smaller than ByteV2 V5 layout");
  STD_TORCH_CHECK(tile_policy[0] == ByteV2DefaultPolicy::CodecTokenBlock &&
                      tile_policy[1] == ByteV2DefaultPolicy::CodecDimBlock &&
                      tile_policy[2] == ByteV2DefaultPolicy::AllocBlockTokens &&
                      tile_policy[4] == ByteV2DefaultPolicy::HeadDim &&
                      tile_policy[5] == ByteV2DefaultPolicy::HeadDimV,
                  "ByteV2 fused staging release currently supports the "
                  "default V4 page layout");

  const torch::stable::accelerator::DeviceGuard device_guard(
      block_to_staging_slot.get_device_index());
  const cudaStream_t stream =
      get_current_cuda_stream(block_to_staging_slot.get_device_index());
  const int64_t num_staging_slots = staging_to_physical_block.size(0);
  if (num_staging_slots == 0) {
    cudaMemsetAsync(next_staging_slot.mutable_data_ptr<int32_t>(), 0,
                    sizeof(int32_t), stream);
    cudaMemsetAsync(overflow.mutable_data_ptr<int32_t>(), 0, sizeof(int32_t),
                    stream);
    return;
  }

  constexpr int kThreads = 256;
  const int64_t blocks64 = (num_staging_slots + kThreads - 1) / kThreads;
  const int blocks = static_cast<int>(blocks64 > 65535 ? 65535 : blocks64);
  byte_v2_release_raw_staging_and_update_flags_kernel<<<blocks, kThreads, 0,
                                                        stream>>>(
      block_to_staging_slot.mutable_data_ptr<int32_t>(),
      staging_to_physical_block.mutable_data_ptr<int32_t>(),
      valid_rows.mutable_data_ptr<int32_t>(),
      next_staging_slot.mutable_data_ptr<int32_t>(),
      overflow.mutable_data_ptr<int32_t>(),
      page_unsafe_flags.mutable_data_ptr<int32_t>(),
      reinterpret_cast<const uint8_t*>(kv_cache.const_data_ptr()),
      num_staging_slots, kv_cache.size(0), kv_cache.stride(0));
  const cudaError_t err = cudaGetLastError();
  STD_TORCH_CHECK(
      err == cudaSuccess,
      "byte_v2_release_raw_staging_and_update_flags kernel launch failed: ",
      cudaGetErrorString(err));
}

static void byte_v2_commit_raw_staging_to_cache_impl(
    torch::stable::Tensor& raw_staging, torch::stable::Tensor& kv_cache,
    torch::stable::Tensor& staging_to_physical_block,
    torch::stable::Tensor& valid_rows, int64_t codec_token_block,
    int64_t codec_dim_block, int64_t alloc_block_tokens,
    bool fuse_metadata_clear, bool bypass_serial_metadata,
    bool warp_parallel_histogram) {
  using torch::headeronly::ScalarType;

  STD_TORCH_CHECK(raw_staging.device().is_cuda(),
                  "raw_staging must be a CUDA tensor");
  STD_TORCH_CHECK(raw_staging.device() == kv_cache.device(),
                  "raw_staging and kv_cache must be on the same device");
  STD_TORCH_CHECK(raw_staging.device() == staging_to_physical_block.device(),
                  "raw_staging and staging_to_physical_block must be on the "
                  "same device");
  STD_TORCH_CHECK(raw_staging.device() == valid_rows.device(),
                  "raw_staging and valid_rows must be on the same device");
  check_byte_v2_tile_policy({codec_token_block, codec_dim_block,
                             alloc_block_tokens, alloc_block_tokens,
                             ByteV2DefaultPolicy::HeadDim,
                             ByteV2DefaultPolicy::HeadDimV});
  STD_TORCH_CHECK(raw_staging.scalar_type() == ScalarType::Byte,
                  "raw_staging must use uint8 storage");
  STD_TORCH_CHECK(kv_cache.scalar_type() == ScalarType::Byte,
                  "kv_cache must use uint8 storage");
  STD_TORCH_CHECK(staging_to_physical_block.scalar_type() == ScalarType::Int,
                  "staging_to_physical_block must be int32");
  STD_TORCH_CHECK(valid_rows.scalar_type() == ScalarType::Int,
                  "valid_rows must be int32");
  STD_TORCH_CHECK(raw_staging.dim() == 2,
                  "raw_staging must be [num_slots, slot_size_bytes]");
  STD_TORCH_CHECK(kv_cache.dim() == 2,
                  "kv_cache must be [num_blocks, page_size_bytes]");
  STD_TORCH_CHECK(staging_to_physical_block.dim() == 1,
                  "staging_to_physical_block must be [num_slots]");
  STD_TORCH_CHECK(valid_rows.dim() == 1, "valid_rows must be [num_slots]");
  STD_TORCH_CHECK(valid_rows.size(0) >= staging_to_physical_block.size(0),
                  "valid_rows must contain all staging slots");
  STD_TORCH_CHECK(raw_staging.size(0) >= staging_to_physical_block.size(0),
                  "raw_staging must contain all staging slots");
  STD_TORCH_CHECK(raw_staging.stride(1) == 1,
                  "raw_staging slot dimension must be contiguous");
  STD_TORCH_CHECK(kv_cache.stride(1) == 1,
                  "kv_cache page dimension must be contiguous");
  STD_TORCH_CHECK(
      codec_token_block == ByteV2DefaultPolicy::CodecTokenBlock &&
          codec_dim_block == ByteV2DefaultPolicy::CodecDimBlock &&
          alloc_block_tokens == ByteV2DefaultPolicy::AllocBlockTokens,
      "ByteV2 raw staging commit currently supports 16x16 codec tiles "
      "and block_size=16");
  STD_TORCH_CHECK(
      raw_staging.size(1) >= ByteV2DefaultRawStagingLayout::SlotSizeBytes,
      "raw_staging slot size is smaller than ByteV2 raw staging layout");
  STD_TORCH_CHECK(
      raw_staging.stride(0) >= ByteV2DefaultRawStagingLayout::SlotSizeBytes,
      "raw_staging slot stride is smaller than ByteV2 raw staging layout");
  STD_TORCH_CHECK(kv_cache.size(1) >= ByteV2DefaultLayout::PageSizeBytes,
                  "kv_cache page size is smaller than ByteV2 V5 layout");
  STD_TORCH_CHECK(kv_cache.stride(0) >= ByteV2DefaultLayout::PageSizeBytes,
                  "kv_cache block stride is smaller than ByteV2 V5 layout");

  const int64_t num_staging_slots = staging_to_physical_block.size(0);
  if (num_staging_slots == 0) {
    return;
  }

  constexpr int kThreads = ByteV2DefaultPolicy::CodecPackedElems;
  const dim3 grid(
      static_cast<unsigned int>(num_staging_slots),
      static_cast<unsigned int>(2 * ByteV2DefaultLayout::NumKvHeadsValue *
                                ByteV2DefaultPolicy::KDimTiles),
      1);

  const torch::stable::accelerator::DeviceGuard device_guard(
      raw_staging.get_device_index());
  const cudaStream_t stream =
      get_current_cuda_stream(raw_staging.get_device_index());

  STD_TORCH_CHECK(!bypass_serial_metadata || fuse_metadata_clear,
                  "ByteV2 serial-metadata bypass requires fused metadata "
                  "clear");
  STD_TORCH_CHECK(!warp_parallel_histogram || fuse_metadata_clear,
                  "ByteV2 warp-parallel histogram requires fused metadata "
                  "clear");
  STD_TORCH_CHECK(!bypass_serial_metadata || !warp_parallel_histogram,
                  "ByteV2 serial-metadata bypass and warp-parallel histogram "
                  "are mutually exclusive");
  if constexpr (ByteV2DefaultLayout::PagePooledOutliersValue) {
    if (fuse_metadata_clear) {
      constexpr int kClearThreads = 256;
      const int64_t clear_work_items =
          num_staging_slots * ByteV2DefaultLayout::AlignedMetadataBytes;
      const int clear_blocks = static_cast<int>(
          (clear_work_items + kClearThreads - 1) / kClearThreads);
      byte_v2_clear_page_metadata_from_staging_kernel<<<
          clear_blocks, kClearThreads, 0, stream>>>(
          reinterpret_cast<uint8_t*>(kv_cache.mutable_data_ptr()),
          staging_to_physical_block.const_data_ptr<int32_t>(),
          num_staging_slots, kv_cache.stride(0));
    }
  }
  if (fuse_metadata_clear) {
    if (bypass_serial_metadata) {
      byte_v2_commit_raw_staging_to_cache_kernel<true, true, false>
          <<<grid, kThreads, 0, stream>>>(
              reinterpret_cast<const uint8_t*>(raw_staging.const_data_ptr()),
              reinterpret_cast<uint8_t*>(kv_cache.mutable_data_ptr()),
              staging_to_physical_block.const_data_ptr<int32_t>(),
              valid_rows.const_data_ptr<int32_t>(), num_staging_slots,
              raw_staging.stride(0), kv_cache.stride(0));
    } else if (warp_parallel_histogram) {
      constexpr int kHistogramBytes = 128 * sizeof(int);
      byte_v2_commit_raw_staging_to_cache_kernel<true, false, true>
          <<<grid, kThreads, kHistogramBytes, stream>>>(
              reinterpret_cast<const uint8_t*>(raw_staging.const_data_ptr()),
              reinterpret_cast<uint8_t*>(kv_cache.mutable_data_ptr()),
              staging_to_physical_block.const_data_ptr<int32_t>(),
              valid_rows.const_data_ptr<int32_t>(), num_staging_slots,
              raw_staging.stride(0), kv_cache.stride(0));
    } else {
      byte_v2_commit_raw_staging_to_cache_kernel<true, false, false>
          <<<grid, kThreads, 0, stream>>>(
              reinterpret_cast<const uint8_t*>(raw_staging.const_data_ptr()),
              reinterpret_cast<uint8_t*>(kv_cache.mutable_data_ptr()),
              staging_to_physical_block.const_data_ptr<int32_t>(),
              valid_rows.const_data_ptr<int32_t>(), num_staging_slots,
              raw_staging.stride(0), kv_cache.stride(0));
    }
  } else {
    constexpr int kClearThreads = 256;
    const int64_t clear_work_items =
        num_staging_slots * ByteV2DefaultLayout::AlignedMetadataBytes;
    const int clear_blocks = static_cast<int>(
        (clear_work_items + kClearThreads - 1) / kClearThreads);
    byte_v2_clear_page_metadata_from_staging_kernel<<<
        clear_blocks, kClearThreads, 0, stream>>>(
        reinterpret_cast<uint8_t*>(kv_cache.mutable_data_ptr()),
        staging_to_physical_block.const_data_ptr<int32_t>(), num_staging_slots,
        kv_cache.stride(0));
    byte_v2_commit_raw_staging_to_cache_kernel<false, false, false>
        <<<grid, kThreads, 0, stream>>>(
            reinterpret_cast<const uint8_t*>(raw_staging.const_data_ptr()),
            reinterpret_cast<uint8_t*>(kv_cache.mutable_data_ptr()),
            staging_to_physical_block.const_data_ptr<int32_t>(),
            valid_rows.const_data_ptr<int32_t>(), num_staging_slots,
            raw_staging.stride(0), kv_cache.stride(0));
  }
  const cudaError_t err = cudaGetLastError();
  STD_TORCH_CHECK(err == cudaSuccess,
                  "byte_v2_commit_raw_staging_to_cache kernel launch failed: ",
                  cudaGetErrorString(err));
}

void byte_v2_commit_raw_staging_to_cache(
    torch::stable::Tensor& raw_staging, torch::stable::Tensor& kv_cache,
    torch::stable::Tensor& staging_to_physical_block,
    torch::stable::Tensor& valid_rows, int64_t codec_token_block,
    int64_t codec_dim_block, int64_t alloc_block_tokens) {
  byte_v2_commit_raw_staging_to_cache_impl(
      raw_staging, kv_cache, staging_to_physical_block, valid_rows,
      codec_token_block, codec_dim_block, alloc_block_tokens, false, false,
      false);
}

void byte_v2_update_cache_raw_staging(
    torch::stable::Tensor& key, torch::stable::Tensor& value,
    torch::stable::Tensor& raw_staging, torch::stable::Tensor& kv_cache,
    torch::stable::Tensor& slot_mapping,
    torch::stable::Tensor& block_to_staging_slot,
    torch::stable::Tensor& staging_to_physical_block,
    torch::stable::Tensor& valid_rows, torch::stable::Tensor& next_staging_slot,
    torch::stable::Tensor& overflow, torch::stable::Tensor& page_unsafe_flags,
    const std::vector<int64_t>& tile_policy, bool fuse_metadata_clear,
    bool bypass_serial_metadata, bool warp_parallel_histogram) {
  check_byte_v2_tile_policy(tile_policy);
  STD_TORCH_CHECK(page_unsafe_flags.dim() == 1,
                  "page_unsafe_flags must be a 1D tensor");
  STD_TORCH_CHECK(page_unsafe_flags.size(0) > 0,
                  "page_unsafe_flags must not be empty");

  const int64_t codec_token_block = tile_policy[0];
  const int64_t codec_dim_block = tile_policy[1];
  const int64_t alloc_block_tokens = tile_policy[2];
  byte_v2_prepare_raw_staging(slot_mapping, block_to_staging_slot,
                              staging_to_physical_block, valid_rows,
                              next_staging_slot, overflow, alloc_block_tokens);
  byte_v2_hydrate_raw_staging_from_cache(
      raw_staging, kv_cache, staging_to_physical_block, valid_rows,
      codec_token_block, codec_dim_block, alloc_block_tokens);
  byte_v2_append_raw_staging(key, value, raw_staging, slot_mapping,
                             block_to_staging_slot, codec_token_block,
                             codec_dim_block, alloc_block_tokens);
  byte_v2_commit_raw_staging_to_cache_impl(
      raw_staging, kv_cache, staging_to_physical_block, valid_rows,
      codec_token_block, codec_dim_block, alloc_block_tokens,
      fuse_metadata_clear, bypass_serial_metadata, warp_parallel_histogram);
  byte_v2_release_raw_staging_and_update_flags(
      block_to_staging_slot, staging_to_physical_block, valid_rows,
      next_staging_slot, overflow, page_unsafe_flags, kv_cache, tile_policy);
}

void byte_v2_collect_cache_stats(torch::stable::Tensor& stats,
                                 torch::stable::Tensor& kv_cache,
                                 torch::stable::Tensor& block_tables,
                                 torch::stable::Tensor& seq_lens,
                                 int64_t max_seq_len,
                                 const std::vector<int64_t>& tile_policy) {
  using torch::headeronly::ScalarType;

  STD_TORCH_CHECK(stats.device().is_cuda(), "stats must be a CUDA tensor");
  STD_TORCH_CHECK(stats.device() == kv_cache.device(),
                  "stats and kv_cache must be on the same device");
  STD_TORCH_CHECK(stats.device() == block_tables.device(),
                  "stats and block_tables must be on the same device");
  STD_TORCH_CHECK(stats.device() == seq_lens.device(),
                  "stats and seq_lens must be on the same device");
  check_byte_v2_tile_policy(tile_policy);
  STD_TORCH_CHECK(stats.scalar_type() == ScalarType::Int,
                  "stats must use int32 storage");
  STD_TORCH_CHECK(kv_cache.scalar_type() == ScalarType::Byte,
                  "kv_cache must use uint8 storage");
  STD_TORCH_CHECK(block_tables.scalar_type() == ScalarType::Int,
                  "block_tables must be int32");
  STD_TORCH_CHECK(seq_lens.scalar_type() == ScalarType::Int,
                  "seq_lens must be int32");
  STD_TORCH_CHECK(stats.dim() == 1, "stats must be a 1D tensor");
  STD_TORCH_CHECK(stats.size(0) >= 4,
                  "stats must have at least 4 int32 values");
  STD_TORCH_CHECK(stats.stride(0) == 1, "stats must be contiguous");
  STD_TORCH_CHECK(kv_cache.dim() == 2,
                  "kv_cache must be [num_blocks, page_size_bytes]");
  STD_TORCH_CHECK(block_tables.dim() == 2,
                  "block_tables must be [num_seqs, max_num_blocks]");
  STD_TORCH_CHECK(seq_lens.dim() == 1, "seq_lens must be [num_seqs]");
  STD_TORCH_CHECK(seq_lens.size(0) == block_tables.size(0),
                  "seq_lens length must match block_tables batch size");
  STD_TORCH_CHECK(kv_cache.stride(1) == 1,
                  "kv_cache page dimension must be contiguous");
  STD_TORCH_CHECK(kv_cache.size(1) >= ByteV2DefaultLayout::PageSizeBytes,
                  "kv_cache page size is smaller than ByteV2 V5 layout");
  STD_TORCH_CHECK(kv_cache.stride(0) >= ByteV2DefaultLayout::PageSizeBytes,
                  "kv_cache block stride is smaller than ByteV2 V5 layout");
  STD_TORCH_CHECK(max_seq_len >= 0, "max_seq_len must be non-negative");
  STD_TORCH_CHECK(tile_policy[0] == ByteV2DefaultPolicy::CodecTokenBlock &&
                      tile_policy[1] == ByteV2DefaultPolicy::CodecDimBlock &&
                      tile_policy[2] == ByteV2DefaultPolicy::AllocBlockTokens &&
                      tile_policy[4] == ByteV2DefaultPolicy::HeadDim &&
                      tile_policy[5] == ByteV2DefaultPolicy::HeadDimV,
                  "ByteV2 cache stats currently supports the default V4 "
                  "page layout");

  const torch::stable::accelerator::DeviceGuard device_guard(
      stats.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream(stats.get_device_index());
  cudaMemsetAsync(stats.mutable_data_ptr<int32_t>(), 0,
                  4 * static_cast<int64_t>(sizeof(int32_t)), stream);

  if (block_tables.size(0) == 0 || block_tables.size(1) == 0 ||
      max_seq_len == 0) {
    return;
  }

  constexpr int kThreads = 256;
  const int64_t work_items = block_tables.size(0) * block_tables.size(1);
  const int blocks = static_cast<int>((work_items + kThreads - 1) / kThreads);
  byte_v2_collect_cache_stats_kernel<<<blocks, kThreads, 0, stream>>>(
      stats.mutable_data_ptr<int32_t>(),
      reinterpret_cast<const uint8_t*>(kv_cache.const_data_ptr()),
      block_tables.const_data_ptr<int32_t>(),
      seq_lens.const_data_ptr<int32_t>(), block_tables.size(0),
      block_tables.stride(0), block_tables.stride(1), block_tables.size(1),
      kv_cache.size(0), kv_cache.stride(0), max_seq_len);
  const cudaError_t err = cudaGetLastError();
  STD_TORCH_CHECK(err == cudaSuccess,
                  "byte_v2_collect_cache_stats kernel launch failed: ",
                  cudaGetErrorString(err));
}

void byte_v2_update_cache_unsafe_flags(
    torch::stable::Tensor& page_unsafe_flags, torch::stable::Tensor& kv_cache,
    torch::stable::Tensor& slot_mapping,
    const std::vector<int64_t>& tile_policy) {
  using torch::headeronly::ScalarType;

  STD_TORCH_CHECK(page_unsafe_flags.device().is_cuda(),
                  "page_unsafe_flags must be a CUDA tensor");
  STD_TORCH_CHECK(page_unsafe_flags.device() == kv_cache.device(),
                  "page_unsafe_flags and kv_cache must be on the same device");
  STD_TORCH_CHECK(page_unsafe_flags.device() == slot_mapping.device(),
                  "page_unsafe_flags and slot_mapping must be on the same "
                  "device");
  check_byte_v2_tile_policy(tile_policy);
  STD_TORCH_CHECK(page_unsafe_flags.scalar_type() == ScalarType::Int,
                  "page_unsafe_flags must use int32 storage");
  STD_TORCH_CHECK(kv_cache.scalar_type() == ScalarType::Byte,
                  "kv_cache must use uint8 storage");
  STD_TORCH_CHECK(slot_mapping.scalar_type() == ScalarType::Long,
                  "slot_mapping must be int64");
  STD_TORCH_CHECK(page_unsafe_flags.dim() == 1,
                  "page_unsafe_flags must be a 1D tensor");
  STD_TORCH_CHECK(page_unsafe_flags.stride(0) == 1,
                  "page_unsafe_flags must be contiguous");
  STD_TORCH_CHECK(kv_cache.dim() == 2,
                  "kv_cache must be [num_blocks, page_size_bytes]");
  STD_TORCH_CHECK(slot_mapping.dim() == 1, "slot_mapping must be a 1D tensor");
  STD_TORCH_CHECK(page_unsafe_flags.size(0) >= kv_cache.size(0),
                  "page_unsafe_flags must cover every kv_cache block");
  STD_TORCH_CHECK(kv_cache.stride(1) == 1,
                  "kv_cache page dimension must be contiguous");
  STD_TORCH_CHECK(kv_cache.size(1) >= ByteV2DefaultLayout::PageSizeBytes,
                  "kv_cache page size is smaller than ByteV2 V5 layout");
  STD_TORCH_CHECK(kv_cache.stride(0) >= ByteV2DefaultLayout::PageSizeBytes,
                  "kv_cache block stride is smaller than ByteV2 V5 layout");
  STD_TORCH_CHECK(tile_policy[0] == ByteV2DefaultPolicy::CodecTokenBlock &&
                      tile_policy[1] == ByteV2DefaultPolicy::CodecDimBlock &&
                      tile_policy[2] == ByteV2DefaultPolicy::AllocBlockTokens &&
                      tile_policy[4] == ByteV2DefaultPolicy::HeadDim &&
                      tile_policy[5] == ByteV2DefaultPolicy::HeadDimV,
                  "ByteV2 unsafe page flags currently support the default V4 "
                  "page layout");

  if (slot_mapping.size(0) == 0 || kv_cache.size(0) == 0) {
    return;
  }

  const torch::stable::accelerator::DeviceGuard device_guard(
      page_unsafe_flags.get_device_index());
  const cudaStream_t stream =
      get_current_cuda_stream(page_unsafe_flags.get_device_index());
  constexpr int kThreads = 256;
  const int blocks =
      static_cast<int>((slot_mapping.size(0) + kThreads - 1) / kThreads);
  byte_v2_update_cache_unsafe_flags_kernel<<<blocks, kThreads, 0, stream>>>(
      page_unsafe_flags.mutable_data_ptr<int32_t>(),
      reinterpret_cast<const uint8_t*>(kv_cache.const_data_ptr()),
      slot_mapping.const_data_ptr<int64_t>(), slot_mapping.size(0),
      kv_cache.size(0), kv_cache.stride(0));
  const cudaError_t err = cudaGetLastError();
  STD_TORCH_CHECK(err == cudaSuccess,
                  "byte_v2_update_cache_unsafe_flags kernel launch failed: ",
                  cudaGetErrorString(err));
}

void byte_v2_paged_decode_attention(torch::stable::Tensor& output,
                                    torch::stable::Tensor& query,
                                    torch::stable::Tensor& kv_cache,
                                    torch::stable::Tensor& block_tables,
                                    torch::stable::Tensor& seq_lens,
                                    double scale, int64_t num_kv_heads,
                                    int64_t block_size, int64_t max_seq_len,
                                    const std::vector<int64_t>& tile_policy) {
  using torch::headeronly::ScalarType;

  STD_TORCH_CHECK(query.device().is_cuda(), "query must be a CUDA tensor");
  STD_TORCH_CHECK(output.device() == query.device(),
                  "output and query must be on the same device");
  STD_TORCH_CHECK(query.device() == kv_cache.device(),
                  "query and kv_cache must be on the same device");
  STD_TORCH_CHECK(query.device() == block_tables.device(),
                  "query and block_tables must be on the same device");
  STD_TORCH_CHECK(query.device() == seq_lens.device(),
                  "query and seq_lens must be on the same device");
  STD_TORCH_CHECK(output.dim() == 3,
                  "output must be [num_tokens, num_heads, head_dim]");
  STD_TORCH_CHECK(query.sizes().equals(output.sizes()),
                  "query and output must have the same shape");
  STD_TORCH_CHECK(kv_cache.dim() == 2,
                  "kv_cache must be [num_blocks, page_size_bytes]");
  STD_TORCH_CHECK(block_tables.dim() == 2,
                  "block_tables must be [num_tokens, max_num_blocks]");
  STD_TORCH_CHECK(seq_lens.dim() == 1, "seq_lens must be [num_tokens]");
  STD_TORCH_CHECK(block_tables.size(0) >= output.size(0),
                  "block_tables must contain all query tokens");
  STD_TORCH_CHECK(seq_lens.size(0) >= output.size(0),
                  "seq_lens must contain all query tokens");
  STD_TORCH_CHECK(num_kv_heads > 0, "num_kv_heads must be positive");
  STD_TORCH_CHECK(block_size > 0, "block_size must be positive");
  STD_TORCH_CHECK(max_seq_len >= 0, "max_seq_len must be non-negative");
  STD_TORCH_CHECK(scale > 0.0, "scale must be positive");
  check_byte_v2_tile_policy(tile_policy);

  STD_TORCH_CHECK(query.scalar_type() == ScalarType::BFloat16,
                  "ByteV2 decode currently supports bf16 query tensors");
  STD_TORCH_CHECK(output.scalar_type() == query.scalar_type(),
                  "output and query dtypes must match");
  STD_TORCH_CHECK(kv_cache.scalar_type() == ScalarType::Byte,
                  "kv_cache must use uint8 storage");
  STD_TORCH_CHECK(block_tables.scalar_type() == ScalarType::Int,
                  "block_tables must be int32");
  STD_TORCH_CHECK(seq_lens.scalar_type() == ScalarType::Int,
                  "seq_lens must be int32");
  STD_TORCH_CHECK(kv_cache.stride(1) == 1,
                  "kv_cache page dimension must be contiguous");
  STD_TORCH_CHECK(block_tables.stride(1) == 1,
                  "block_tables block dimension must be contiguous");
  STD_TORCH_CHECK(output.size(1) % num_kv_heads == 0,
                  "num_heads must be divisible by num_kv_heads");
  STD_TORCH_CHECK(num_kv_heads == ByteV2DefaultLayout::NumKvHeadsValue,
                  "ByteV2 decode currently supports 8 KV heads");
  STD_TORCH_CHECK(output.size(2) == ByteV2DefaultPolicy::HeadDim,
                  "ByteV2 decode currently supports head_dim=128");
  STD_TORCH_CHECK(block_size == ByteV2DefaultPolicy::AllocBlockTokens,
                  "ByteV2 decode currently supports block_size=16");
  STD_TORCH_CHECK(tile_policy[0] == ByteV2DefaultPolicy::CodecTokenBlock &&
                      tile_policy[1] == ByteV2DefaultPolicy::CodecDimBlock &&
                      tile_policy[2] == ByteV2DefaultPolicy::AllocBlockTokens &&
                      tile_policy[4] == ByteV2DefaultPolicy::HeadDim &&
                      tile_policy[5] == ByteV2DefaultPolicy::HeadDimV,
                  "ByteV2 decode currently supports 16x16 codec tiles, "
                  "block_size=16, head_dim=128, and head_dim_v=128");
  STD_TORCH_CHECK(kv_cache.size(1) >= ByteV2DefaultLayout::PageSizeBytes,
                  "kv_cache page size is smaller than ByteV2 V5 layout");
  STD_TORCH_CHECK(kv_cache.stride(0) >= ByteV2DefaultLayout::PageSizeBytes,
                  "kv_cache block stride is smaller than ByteV2 V5 layout");
  STD_TORCH_CHECK(
      block_tables.size(1) >=
          (max_seq_len + ByteV2DefaultPolicy::AllocBlockTokens - 1) /
              ByteV2DefaultPolicy::AllocBlockTokens,
      "block_tables is too small for max_seq_len");

  const bool use_raw_fallback = byte_v2_use_raw_fallback(tile_policy);
  const bool assume_no_fallback_no_outlier =
      byte_v2_assume_no_fallback_no_outlier(tile_policy);
  STD_TORCH_CHECK(
      !use_raw_fallback || ByteV2DefaultLayout::IncludeRawPayloadValue,
      "ByteV2 raw decode fallback requires a page-local raw payload");
  STD_TORCH_CHECK(!use_raw_fallback || !assume_no_fallback_no_outlier,
                  "ByteV2 raw decode fallback and no-outlier fast path are "
                  "mutually exclusive");
  if (tile_policy[3] == ByteV2DefaultPolicy::ComputeBlockN) {
    if (use_raw_fallback) {
      launch_byte_v2_paged_decode_attention<ByteV2DefaultLayout, true, false>(
          output, query, kv_cache, block_tables, seq_lens, scale);
    } else if (assume_no_fallback_no_outlier) {
      launch_byte_v2_paged_decode_attention<ByteV2DefaultLayout, false, true>(
          output, query, kv_cache, block_tables, seq_lens, scale);
    } else {
      launch_byte_v2_paged_decode_attention<ByteV2DefaultLayout, false, false>(
          output, query, kv_cache, block_tables, seq_lens, scale);
    }
  } else if (tile_policy[3] == ByteV2BN128Policy::ComputeBlockN) {
    if (use_raw_fallback) {
      launch_byte_v2_paged_decode_attention<ByteV2BN128Layout, true, false>(
          output, query, kv_cache, block_tables, seq_lens, scale);
    } else if (assume_no_fallback_no_outlier) {
      launch_byte_v2_paged_decode_attention<ByteV2BN128Layout, false, true>(
          output, query, kv_cache, block_tables, seq_lens, scale);
    } else {
      launch_byte_v2_paged_decode_attention<ByteV2BN128Layout, false, false>(
          output, query, kv_cache, block_tables, seq_lens, scale);
    }
  } else {
    STD_TORCH_CHECK(false,
                    "ByteV2 decode currently supports compute_block_n=64 or "
                    "compute_block_n=128");
  }
}

void byte_v2_paged_decode_attention_split_k(
    torch::stable::Tensor& output, torch::stable::Tensor& exp_sums,
    torch::stable::Tensor& max_logits, torch::stable::Tensor& tmp_out,
    torch::stable::Tensor& query, torch::stable::Tensor& kv_cache,
    torch::stable::Tensor& block_tables, torch::stable::Tensor& seq_lens,
    double scale, int64_t num_kv_heads, int64_t block_size, int64_t max_seq_len,
    int64_t partition_size, const std::vector<int64_t>& tile_policy) {
  using torch::headeronly::ScalarType;

  STD_TORCH_CHECK(query.device().is_cuda(), "query must be a CUDA tensor");
  STD_TORCH_CHECK(output.device() == query.device(),
                  "output and query must be on the same device");
  STD_TORCH_CHECK(query.device() == kv_cache.device(),
                  "query and kv_cache must be on the same device");
  STD_TORCH_CHECK(query.device() == block_tables.device(),
                  "query and block_tables must be on the same device");
  STD_TORCH_CHECK(query.device() == seq_lens.device(),
                  "query and seq_lens must be on the same device");
  STD_TORCH_CHECK(query.device() == exp_sums.device(),
                  "query and exp_sums must be on the same device");
  const bool has_max_logits = max_logits.numel() > 0;
  if (has_max_logits) {
    STD_TORCH_CHECK(query.device() == max_logits.device(),
                    "query and max_logits must be on the same device");
  }
  STD_TORCH_CHECK(query.device() == tmp_out.device(),
                  "query and tmp_out must be on the same device");
  STD_TORCH_CHECK(output.dim() == 3,
                  "output must be [num_tokens, num_heads, head_dim]");
  STD_TORCH_CHECK(query.sizes().equals(output.sizes()),
                  "query and output must have the same shape");
  STD_TORCH_CHECK(kv_cache.dim() == 2,
                  "kv_cache must be [num_blocks, page_size_bytes]");
  STD_TORCH_CHECK(block_tables.dim() == 2,
                  "block_tables must be [num_tokens, max_num_blocks]");
  STD_TORCH_CHECK(seq_lens.dim() == 1, "seq_lens must be [num_tokens]");
  STD_TORCH_CHECK(exp_sums.dim() == 3,
                  "exp_sums must be [num_tokens, num_heads, partitions]");
  if (has_max_logits) {
    STD_TORCH_CHECK(max_logits.dim() == 3,
                    "max_logits must be [num_tokens, num_heads, partitions]");
  }
  STD_TORCH_CHECK(
      tmp_out.dim() == 4,
      "tmp_out must be [num_tokens, num_heads, partitions, head_dim]");
  STD_TORCH_CHECK(block_tables.size(0) >= output.size(0),
                  "block_tables must contain all query tokens");
  STD_TORCH_CHECK(seq_lens.size(0) >= output.size(0),
                  "seq_lens must contain all query tokens");
  STD_TORCH_CHECK(num_kv_heads > 0, "num_kv_heads must be positive");
  STD_TORCH_CHECK(block_size > 0, "block_size must be positive");
  STD_TORCH_CHECK(max_seq_len >= 0, "max_seq_len must be non-negative");
  STD_TORCH_CHECK(partition_size > 0, "partition_size must be positive");
  STD_TORCH_CHECK(partition_size % block_size == 0,
                  "partition_size must be divisible by block_size");
  STD_TORCH_CHECK(scale > 0.0, "scale must be positive");
  check_byte_v2_tile_policy(tile_policy);

  STD_TORCH_CHECK(query.scalar_type() == ScalarType::BFloat16,
                  "ByteV2 decode currently supports bf16 query tensors");
  STD_TORCH_CHECK(output.scalar_type() == query.scalar_type(),
                  "output and query dtypes must match");
  STD_TORCH_CHECK(kv_cache.scalar_type() == ScalarType::Byte,
                  "kv_cache must use uint8 storage");
  STD_TORCH_CHECK(block_tables.scalar_type() == ScalarType::Int,
                  "block_tables must be int32");
  STD_TORCH_CHECK(seq_lens.scalar_type() == ScalarType::Int,
                  "seq_lens must be int32");
  STD_TORCH_CHECK(exp_sums.scalar_type() == ScalarType::Float,
                  "exp_sums must be float32");
  if (has_max_logits) {
    STD_TORCH_CHECK(max_logits.scalar_type() == ScalarType::Float,
                    "max_logits must be float32");
  }
  STD_TORCH_CHECK(tmp_out.scalar_type() == ScalarType::Float,
                  "tmp_out must be float32");
  STD_TORCH_CHECK(exp_sums.is_contiguous(), "exp_sums must be contiguous");
  if (has_max_logits) {
    STD_TORCH_CHECK(max_logits.is_contiguous(),
                    "max_logits must be contiguous");
  }
  STD_TORCH_CHECK(tmp_out.is_contiguous(), "tmp_out must be contiguous");
  STD_TORCH_CHECK(kv_cache.stride(1) == 1,
                  "kv_cache page dimension must be contiguous");
  STD_TORCH_CHECK(block_tables.stride(1) == 1,
                  "block_tables block dimension must be contiguous");
  STD_TORCH_CHECK(output.size(1) % num_kv_heads == 0,
                  "num_heads must be divisible by num_kv_heads");
  STD_TORCH_CHECK(num_kv_heads == ByteV2DefaultLayout::NumKvHeadsValue,
                  "ByteV2 decode currently supports 8 KV heads");
  STD_TORCH_CHECK(output.size(2) == ByteV2DefaultPolicy::HeadDim,
                  "ByteV2 decode currently supports head_dim=128");
  STD_TORCH_CHECK(block_size == ByteV2DefaultPolicy::AllocBlockTokens,
                  "ByteV2 decode currently supports block_size=16");
  STD_TORCH_CHECK(tile_policy[0] == ByteV2DefaultPolicy::CodecTokenBlock &&
                      tile_policy[1] == ByteV2DefaultPolicy::CodecDimBlock &&
                      tile_policy[2] == ByteV2DefaultPolicy::AllocBlockTokens &&
                      tile_policy[4] == ByteV2DefaultPolicy::HeadDim &&
                      tile_policy[5] == ByteV2DefaultPolicy::HeadDimV,
                  "ByteV2 decode currently supports 16x16 codec tiles, "
                  "block_size=16, head_dim=128, and head_dim_v=128");
  STD_TORCH_CHECK(kv_cache.size(1) >= ByteV2DefaultLayout::PageSizeBytes,
                  "kv_cache page size is smaller than ByteV2 V5 layout");
  STD_TORCH_CHECK(kv_cache.stride(0) >= ByteV2DefaultLayout::PageSizeBytes,
                  "kv_cache block stride is smaller than ByteV2 V5 layout");
  STD_TORCH_CHECK(
      block_tables.size(1) >=
          (max_seq_len + ByteV2DefaultPolicy::AllocBlockTokens - 1) /
              ByteV2DefaultPolicy::AllocBlockTokens,
      "block_tables is too small for max_seq_len");

  const int64_t required_partitions =
      (max_seq_len + partition_size - 1) / partition_size;
  STD_TORCH_CHECK(exp_sums.size(0) >= output.size(0) &&
                      exp_sums.size(1) >= output.size(1) &&
                      exp_sums.size(2) >= required_partitions &&
                      exp_sums.size(2) > 0,
                  "exp_sums workspace is too small");
  if (has_max_logits) {
    STD_TORCH_CHECK(max_logits.sizes().equals(exp_sums.sizes()),
                    "max_logits must have the same shape as exp_sums");
  }
  STD_TORCH_CHECK(tmp_out.size(0) >= output.size(0) &&
                      tmp_out.size(1) >= output.size(1) &&
                      tmp_out.size(2) >= required_partitions &&
                      tmp_out.size(2) == exp_sums.size(2) &&
                      tmp_out.size(3) == output.size(2),
                  "tmp_out workspace is too small");

  const bool use_raw_fallback = byte_v2_use_raw_fallback(tile_policy);
  const bool assume_no_fallback_no_outlier =
      byte_v2_assume_no_fallback_no_outlier(tile_policy);
  const bool use_gqa_packed = byte_v2_use_gqa_packed(tile_policy);
  const bool use_gqa_fa2_like = byte_v2_use_gqa_fa2_like(tile_policy);
  const bool use_gqa_fa2_qk_mma = byte_v2_use_gqa_fa2_qk_mma(tile_policy);
  const bool use_gqa_fa2_mainloop = byte_v2_use_gqa_fa2_mainloop(tile_policy);
  const bool use_gqa_fa2_multiwarp = byte_v2_use_gqa_fa2_multiwarp(tile_policy);
  const bool use_gqa_fa2_direct = byte_v2_use_gqa_fa2_direct(tile_policy);
  const int direct_diagnostic_mode =
      byte_v2_gqa_fa2_direct_diagnostic_mode(tile_policy);
  const int payload_format = byte_v2_payload_format(tile_policy);
  const int q_heads_per_kv = output.size(1) / num_kv_heads;
  if (payload_format == kByteV2PayloadFormatHighByte) {
    STD_TORCH_CHECK(kv_cache.size(1) >= ByteV2HighByteLayout::PageSizeBytes,
                    "kv_cache page size is smaller than ByteV2 high-byte "
                    "layout");
    STD_TORCH_CHECK(kv_cache.stride(0) >= ByteV2HighByteLayout::PageSizeBytes,
                    "kv_cache block stride is smaller than ByteV2 high-byte "
                    "layout");
  }
  STD_TORCH_CHECK(
      !use_raw_fallback || ByteV2DefaultLayout::IncludeRawPayloadValue,
      "ByteV2 raw decode fallback requires a page-local raw payload");
  STD_TORCH_CHECK(!use_raw_fallback || !assume_no_fallback_no_outlier,
                  "ByteV2 raw decode fallback and no-outlier fast path are "
                  "mutually exclusive");
  STD_TORCH_CHECK(!use_gqa_packed || assume_no_fallback_no_outlier,
                  "ByteV2 GQA-packed decode requires the no-outlier fast path");
  STD_TORCH_CHECK(!use_gqa_fa2_like || use_gqa_packed,
                  "ByteV2 GQA FA2-like decode requires GQA-packed decode");
  STD_TORCH_CHECK(!use_gqa_fa2_qk_mma || use_gqa_fa2_like,
                  "ByteV2 GQA FA2 QK-MMA decode requires FA2-like decode");
  STD_TORCH_CHECK(!use_gqa_fa2_mainloop || use_gqa_fa2_qk_mma,
                  "ByteV2 GQA FA2 mainloop decode requires QK-MMA decode");
  STD_TORCH_CHECK(!use_gqa_fa2_multiwarp || use_gqa_fa2_qk_mma,
                  "ByteV2 GQA FA2 multi-warp decode requires QK-MMA decode");
  STD_TORCH_CHECK(!use_gqa_fa2_direct || use_gqa_fa2_qk_mma,
                  "ByteV2 GQA FA2 direct decode requires QK-MMA decode");
  STD_TORCH_CHECK(direct_diagnostic_mode == kByteV2DirectDiagnosticCurrent ||
                      use_gqa_fa2_direct,
                  "ByteV2 FA2-direct diagnostic modes require direct decode");
  STD_TORCH_CHECK(!(use_gqa_fa2_direct && use_gqa_fa2_mainloop),
                  "ByteV2 GQA FA2 direct and mainloop decode are mutually "
                  "exclusive");
  STD_TORCH_CHECK(!(use_gqa_fa2_direct && use_gqa_fa2_multiwarp),
                  "ByteV2 GQA FA2 direct and multi-warp decode are mutually "
                  "exclusive");
  STD_TORCH_CHECK(
      !use_gqa_fa2_like || tile_policy[3] == ByteV2DefaultPolicy::ComputeBlockN,
      "ByteV2 GQA FA2-like decode currently supports "
      "compute_block_n=64 only");
  STD_TORCH_CHECK(!use_gqa_fa2_like || use_gqa_fa2_direct ||
                      partition_size == tile_policy[3],
                  "ByteV2 GQA FA2-like decode currently requires "
                  "partition_size == compute_block_n");
  STD_TORCH_CHECK(!use_gqa_packed || use_gqa_fa2_direct || q_heads_per_kv == 4,
                  "ByteV2 non-direct GQA-packed decode currently requires "
                  "q_per_kv=4");
  STD_TORCH_CHECK(payload_format == kByteV2PayloadFormatDefault ||
                      (assume_no_fallback_no_outlier && use_gqa_fa2_direct),
                  "ByteV2 non-default payload formats currently require "
                  "assume-no-outlier FA2-direct decode");
  STD_TORCH_CHECK(payload_format == kByteV2PayloadFormatDefault ||
                      direct_diagnostic_mode == kByteV2DirectDiagnosticCurrent,
                  "ByteV2 non-default payload formats currently support only "
                  "the current FA2-direct mode");
  STD_TORCH_CHECK(payload_format != kByteV2PayloadFormatSidebandHigh,
                  "ByteV2 sideband-high payload format requires guarded "
                  "split-k decode");
  if (tile_policy[3] == ByteV2DefaultPolicy::ComputeBlockN) {
    if (use_gqa_packed) {
      if (use_gqa_fa2_like) {
        if (use_gqa_fa2_multiwarp) {
          launch_byte_v2_paged_decode_attention_split_k_gqa4_fa2_like_no_fallback_no_outlier<
              ByteV2DefaultLayout, false, true, false, true, false>(
              output, exp_sums, max_logits, tmp_out, query, kv_cache, nullptr,
              block_tables, seq_lens, scale, partition_size);
        } else if (use_gqa_fa2_mainloop) {
          launch_byte_v2_paged_decode_attention_split_k_gqa4_fa2_like_no_fallback_no_outlier<
              ByteV2DefaultLayout, false, true, true, false, false>(
              output, exp_sums, max_logits, tmp_out, query, kv_cache, nullptr,
              block_tables, seq_lens, scale, partition_size);
        } else if (use_gqa_fa2_direct) {
          if (payload_format == kByteV2PayloadFormatHighByte) {
            launch_byte_v2_paged_decode_attention_split_k_fa2_direct_grouped_no_fallback_no_outlier<
                ByteV2HighByteLayout, false>(
                output, exp_sums, max_logits, tmp_out, query, kv_cache, nullptr,
                block_tables, seq_lens, scale, partition_size, q_heads_per_kv,
                direct_diagnostic_mode);
          } else if (payload_format == kByteV2PayloadFormatSidebandHigh) {
            launch_byte_v2_paged_decode_attention_split_k_fa2_direct_grouped_no_fallback_no_outlier<
                ByteV2SidebandHighLayout, false>(
                output, exp_sums, max_logits, tmp_out, query, kv_cache, nullptr,
                block_tables, seq_lens, scale, partition_size, q_heads_per_kv,
                direct_diagnostic_mode);
          } else {
            launch_byte_v2_paged_decode_attention_split_k_fa2_direct_grouped_no_fallback_no_outlier<
                ByteV2DefaultLayout, false>(
                output, exp_sums, max_logits, tmp_out, query, kv_cache, nullptr,
                block_tables, seq_lens, scale, partition_size, q_heads_per_kv,
                direct_diagnostic_mode);
          }
        } else if (use_gqa_fa2_qk_mma) {
          launch_byte_v2_paged_decode_attention_split_k_gqa4_fa2_like_no_fallback_no_outlier<
              ByteV2DefaultLayout, false, true, false, false, false>(
              output, exp_sums, max_logits, tmp_out, query, kv_cache, nullptr,
              block_tables, seq_lens, scale, partition_size);
        } else {
          launch_byte_v2_paged_decode_attention_split_k_gqa4_fa2_like_no_fallback_no_outlier<
              ByteV2DefaultLayout, false, false, false, false, false>(
              output, exp_sums, max_logits, tmp_out, query, kv_cache, nullptr,
              block_tables, seq_lens, scale, partition_size);
        }
      } else {
        launch_byte_v2_paged_decode_attention_split_k_gqa4_no_fallback_no_outlier<
            ByteV2DefaultLayout, false>(output, exp_sums, max_logits, tmp_out,
                                        query, kv_cache, nullptr, block_tables,
                                        seq_lens, scale, partition_size);
      }
    } else if (use_raw_fallback) {
      launch_byte_v2_paged_decode_attention_split_k<ByteV2DefaultLayout, true,
                                                    false, false>(
          output, exp_sums, max_logits, tmp_out, query, kv_cache, nullptr,
          block_tables, seq_lens, scale, partition_size);
    } else if (assume_no_fallback_no_outlier) {
      launch_byte_v2_paged_decode_attention_split_k<ByteV2DefaultLayout, false,
                                                    true, false>(
          output, exp_sums, max_logits, tmp_out, query, kv_cache, nullptr,
          block_tables, seq_lens, scale, partition_size);
    } else {
      launch_byte_v2_paged_decode_attention_split_k<ByteV2DefaultLayout, false,
                                                    false, false>(
          output, exp_sums, max_logits, tmp_out, query, kv_cache, nullptr,
          block_tables, seq_lens, scale, partition_size);
    }
  } else if (tile_policy[3] == ByteV2BN128Policy::ComputeBlockN) {
    if (use_gqa_packed) {
      launch_byte_v2_paged_decode_attention_split_k_gqa4_no_fallback_no_outlier<
          ByteV2BN128Layout, false>(output, exp_sums, max_logits, tmp_out,
                                    query, kv_cache, nullptr, block_tables,
                                    seq_lens, scale, partition_size);
    } else if (use_raw_fallback) {
      launch_byte_v2_paged_decode_attention_split_k<ByteV2BN128Layout, true,
                                                    false, false>(
          output, exp_sums, max_logits, tmp_out, query, kv_cache, nullptr,
          block_tables, seq_lens, scale, partition_size);
    } else if (assume_no_fallback_no_outlier) {
      launch_byte_v2_paged_decode_attention_split_k<ByteV2BN128Layout, false,
                                                    true, false>(
          output, exp_sums, max_logits, tmp_out, query, kv_cache, nullptr,
          block_tables, seq_lens, scale, partition_size);
    } else {
      launch_byte_v2_paged_decode_attention_split_k<ByteV2BN128Layout, false,
                                                    false, false>(
          output, exp_sums, max_logits, tmp_out, query, kv_cache, nullptr,
          block_tables, seq_lens, scale, partition_size);
    }
  } else {
    STD_TORCH_CHECK(false,
                    "ByteV2 decode currently supports compute_block_n=64 or "
                    "compute_block_n=128");
  }
}

void byte_v2_paged_decode_attention_split_k_guarded(
    torch::stable::Tensor& output, torch::stable::Tensor& exp_sums,
    torch::stable::Tensor& max_logits, torch::stable::Tensor& tmp_out,
    torch::stable::Tensor& query, torch::stable::Tensor& kv_cache,
    torch::stable::Tensor& page_unsafe_flags,
    torch::stable::Tensor& block_tables, torch::stable::Tensor& seq_lens,
    double scale, int64_t num_kv_heads, int64_t block_size, int64_t max_seq_len,
    int64_t partition_size, const std::vector<int64_t>& tile_policy) {
  using torch::headeronly::ScalarType;

  STD_TORCH_CHECK(query.device().is_cuda(), "query must be a CUDA tensor");
  STD_TORCH_CHECK(output.device() == query.device(),
                  "output and query must be on the same device");
  STD_TORCH_CHECK(query.device() == kv_cache.device(),
                  "query and kv_cache must be on the same device");
  STD_TORCH_CHECK(query.device() == page_unsafe_flags.device(),
                  "query and page_unsafe_flags must be on the same device");
  STD_TORCH_CHECK(query.device() == block_tables.device(),
                  "query and block_tables must be on the same device");
  STD_TORCH_CHECK(query.device() == seq_lens.device(),
                  "query and seq_lens must be on the same device");
  STD_TORCH_CHECK(query.device() == exp_sums.device(),
                  "query and exp_sums must be on the same device");
  const bool has_max_logits = max_logits.numel() > 0;
  if (has_max_logits) {
    STD_TORCH_CHECK(query.device() == max_logits.device(),
                    "query and max_logits must be on the same device");
  }
  STD_TORCH_CHECK(query.device() == tmp_out.device(),
                  "query and tmp_out must be on the same device");
  STD_TORCH_CHECK(output.dim() == 3,
                  "output must be [num_tokens, num_heads, head_dim]");
  STD_TORCH_CHECK(query.sizes().equals(output.sizes()),
                  "query and output must have the same shape");
  STD_TORCH_CHECK(kv_cache.dim() == 2,
                  "kv_cache must be [num_blocks, page_size_bytes]");
  STD_TORCH_CHECK(page_unsafe_flags.dim() == 1,
                  "page_unsafe_flags must be a 1D tensor");
  STD_TORCH_CHECK(block_tables.dim() == 2,
                  "block_tables must be [num_tokens, max_num_blocks]");
  STD_TORCH_CHECK(seq_lens.dim() == 1, "seq_lens must be [num_tokens]");
  STD_TORCH_CHECK(exp_sums.dim() == 3,
                  "exp_sums must be [num_tokens, num_heads, partitions]");
  if (has_max_logits) {
    STD_TORCH_CHECK(max_logits.dim() == 3,
                    "max_logits must be [num_tokens, num_heads, partitions]");
  }
  STD_TORCH_CHECK(
      tmp_out.dim() == 4,
      "tmp_out must be [num_tokens, num_heads, partitions, head_dim]");
  STD_TORCH_CHECK(block_tables.size(0) >= output.size(0),
                  "block_tables must contain all query tokens");
  STD_TORCH_CHECK(seq_lens.size(0) >= output.size(0),
                  "seq_lens must contain all query tokens");
  STD_TORCH_CHECK(page_unsafe_flags.size(0) >= kv_cache.size(0),
                  "page_unsafe_flags must cover every kv_cache block");
  STD_TORCH_CHECK(num_kv_heads > 0, "num_kv_heads must be positive");
  STD_TORCH_CHECK(block_size > 0, "block_size must be positive");
  STD_TORCH_CHECK(max_seq_len >= 0, "max_seq_len must be non-negative");
  STD_TORCH_CHECK(partition_size > 0, "partition_size must be positive");
  STD_TORCH_CHECK(partition_size % block_size == 0,
                  "partition_size must be divisible by block_size");
  STD_TORCH_CHECK(scale > 0.0, "scale must be positive");
  check_byte_v2_tile_policy(tile_policy);

  STD_TORCH_CHECK(query.scalar_type() == ScalarType::BFloat16,
                  "ByteV2 decode currently supports bf16 query tensors");
  STD_TORCH_CHECK(output.scalar_type() == query.scalar_type(),
                  "output and query dtypes must match");
  STD_TORCH_CHECK(kv_cache.scalar_type() == ScalarType::Byte,
                  "kv_cache must use uint8 storage");
  STD_TORCH_CHECK(page_unsafe_flags.scalar_type() == ScalarType::Int,
                  "page_unsafe_flags must use int32 storage");
  STD_TORCH_CHECK(block_tables.scalar_type() == ScalarType::Int,
                  "block_tables must be int32");
  STD_TORCH_CHECK(seq_lens.scalar_type() == ScalarType::Int,
                  "seq_lens must be int32");
  STD_TORCH_CHECK(exp_sums.scalar_type() == ScalarType::Float,
                  "exp_sums must be float32");
  if (has_max_logits) {
    STD_TORCH_CHECK(max_logits.scalar_type() == ScalarType::Float,
                    "max_logits must be float32");
  }
  STD_TORCH_CHECK(tmp_out.scalar_type() == ScalarType::Float,
                  "tmp_out must be float32");
  STD_TORCH_CHECK(page_unsafe_flags.stride(0) == 1,
                  "page_unsafe_flags must be contiguous");
  STD_TORCH_CHECK(exp_sums.is_contiguous(), "exp_sums must be contiguous");
  if (has_max_logits) {
    STD_TORCH_CHECK(max_logits.is_contiguous(),
                    "max_logits must be contiguous");
  }
  STD_TORCH_CHECK(tmp_out.is_contiguous(), "tmp_out must be contiguous");
  STD_TORCH_CHECK(kv_cache.stride(1) == 1,
                  "kv_cache page dimension must be contiguous");
  STD_TORCH_CHECK(block_tables.stride(1) == 1,
                  "block_tables block dimension must be contiguous");
  STD_TORCH_CHECK(output.size(1) % num_kv_heads == 0,
                  "num_heads must be divisible by num_kv_heads");
  STD_TORCH_CHECK(num_kv_heads == ByteV2DefaultLayout::NumKvHeadsValue,
                  "ByteV2 guarded decode currently supports 8 KV heads");
  STD_TORCH_CHECK(output.size(2) == ByteV2DefaultPolicy::HeadDim,
                  "ByteV2 guarded decode currently supports head_dim=128");
  STD_TORCH_CHECK(block_size == ByteV2DefaultPolicy::AllocBlockTokens,
                  "ByteV2 guarded decode currently supports block_size=16");
  STD_TORCH_CHECK(tile_policy[0] == ByteV2DefaultPolicy::CodecTokenBlock &&
                      tile_policy[1] == ByteV2DefaultPolicy::CodecDimBlock &&
                      tile_policy[2] == ByteV2DefaultPolicy::AllocBlockTokens &&
                      tile_policy[4] == ByteV2DefaultPolicy::HeadDim &&
                      tile_policy[5] == ByteV2DefaultPolicy::HeadDimV,
                  "ByteV2 guarded decode currently supports 16x16 codec "
                  "tiles, block_size=16, head_dim=128, and head_dim_v=128");
  STD_TORCH_CHECK(kv_cache.size(1) >= ByteV2DefaultLayout::PageSizeBytes,
                  "kv_cache page size is smaller than ByteV2 V5 layout");
  STD_TORCH_CHECK(kv_cache.stride(0) >= ByteV2DefaultLayout::PageSizeBytes,
                  "kv_cache block stride is smaller than ByteV2 V5 layout");
  STD_TORCH_CHECK(
      block_tables.size(1) >=
          (max_seq_len + ByteV2DefaultPolicy::AllocBlockTokens - 1) /
              ByteV2DefaultPolicy::AllocBlockTokens,
      "block_tables is too small for max_seq_len");

  const int64_t required_partitions =
      (max_seq_len + partition_size - 1) / partition_size;
  STD_TORCH_CHECK(exp_sums.size(0) >= output.size(0) &&
                      exp_sums.size(1) >= output.size(1) &&
                      exp_sums.size(2) >= required_partitions &&
                      exp_sums.size(2) > 0,
                  "exp_sums workspace is too small");
  if (has_max_logits) {
    STD_TORCH_CHECK(max_logits.sizes().equals(exp_sums.sizes()),
                    "max_logits must have the same shape as exp_sums");
  }
  STD_TORCH_CHECK(tmp_out.size(0) >= output.size(0) &&
                      tmp_out.size(1) >= output.size(1) &&
                      tmp_out.size(2) >= required_partitions &&
                      tmp_out.size(2) == exp_sums.size(2) &&
                      tmp_out.size(3) == output.size(2),
                  "tmp_out workspace is too small");

  const bool use_raw_fallback = byte_v2_use_raw_fallback(tile_policy);
  const bool assume_no_fallback_no_outlier =
      byte_v2_assume_no_fallback_no_outlier(tile_policy);
  const bool use_gqa_packed = byte_v2_use_gqa_packed(tile_policy);
  const bool use_gqa_fa2_like = byte_v2_use_gqa_fa2_like(tile_policy);
  const bool use_gqa_fa2_qk_mma = byte_v2_use_gqa_fa2_qk_mma(tile_policy);
  const bool use_gqa_fa2_mainloop = byte_v2_use_gqa_fa2_mainloop(tile_policy);
  const bool use_gqa_fa2_multiwarp = byte_v2_use_gqa_fa2_multiwarp(tile_policy);
  const bool use_gqa_fa2_direct = byte_v2_use_gqa_fa2_direct(tile_policy);
  const int direct_diagnostic_mode =
      byte_v2_gqa_fa2_direct_diagnostic_mode(tile_policy);
  const int payload_format = byte_v2_payload_format(tile_policy);
  const int q_heads_per_kv = output.size(1) / num_kv_heads;
  if (payload_format == kByteV2PayloadFormatHighByte) {
    STD_TORCH_CHECK(kv_cache.size(1) >= ByteV2HighByteLayout::PageSizeBytes,
                    "kv_cache page size is smaller than ByteV2 high-byte "
                    "layout");
    STD_TORCH_CHECK(kv_cache.stride(0) >= ByteV2HighByteLayout::PageSizeBytes,
                    "kv_cache block stride is smaller than ByteV2 high-byte "
                    "layout");
  }
  STD_TORCH_CHECK(!use_raw_fallback,
                  "ByteV2 guarded split-k decode does not support raw "
                  "fallback");
  STD_TORCH_CHECK(assume_no_fallback_no_outlier,
                  "ByteV2 guarded split-k decode requires the no-outlier fast "
                  "path");
  STD_TORCH_CHECK(!use_gqa_packed || assume_no_fallback_no_outlier,
                  "ByteV2 GQA-packed decode requires the no-outlier fast path");
  STD_TORCH_CHECK(!use_gqa_fa2_like || use_gqa_packed,
                  "ByteV2 GQA FA2-like decode requires GQA-packed decode");
  STD_TORCH_CHECK(!use_gqa_fa2_qk_mma || use_gqa_fa2_like,
                  "ByteV2 GQA FA2 QK-MMA decode requires FA2-like decode");
  STD_TORCH_CHECK(!use_gqa_fa2_mainloop || use_gqa_fa2_qk_mma,
                  "ByteV2 GQA FA2 mainloop decode requires QK-MMA decode");
  STD_TORCH_CHECK(!use_gqa_fa2_multiwarp || use_gqa_fa2_qk_mma,
                  "ByteV2 GQA FA2 multi-warp decode requires QK-MMA decode");
  STD_TORCH_CHECK(!use_gqa_fa2_direct || use_gqa_fa2_qk_mma,
                  "ByteV2 GQA FA2 direct decode requires QK-MMA decode");
  STD_TORCH_CHECK(direct_diagnostic_mode == kByteV2DirectDiagnosticCurrent ||
                      use_gqa_fa2_direct,
                  "ByteV2 FA2-direct diagnostic modes require direct decode");
  STD_TORCH_CHECK(!(use_gqa_fa2_direct && use_gqa_fa2_mainloop),
                  "ByteV2 GQA FA2 direct and mainloop decode are mutually "
                  "exclusive");
  STD_TORCH_CHECK(!(use_gqa_fa2_direct && use_gqa_fa2_multiwarp),
                  "ByteV2 GQA FA2 direct and multi-warp decode are mutually "
                  "exclusive");
  STD_TORCH_CHECK(
      !use_gqa_fa2_like || tile_policy[3] == ByteV2DefaultPolicy::ComputeBlockN,
      "ByteV2 GQA FA2-like decode currently supports "
      "compute_block_n=64 only");
  STD_TORCH_CHECK(!use_gqa_fa2_like || use_gqa_fa2_direct ||
                      partition_size == tile_policy[3],
                  "ByteV2 GQA FA2-like decode currently requires "
                  "partition_size == compute_block_n");
  STD_TORCH_CHECK(!use_gqa_packed || use_gqa_fa2_direct || q_heads_per_kv == 4,
                  "ByteV2 non-direct GQA-packed decode currently requires "
                  "q_per_kv=4");
  STD_TORCH_CHECK(payload_format == kByteV2PayloadFormatDefault ||
                      (assume_no_fallback_no_outlier && use_gqa_fa2_direct),
                  "ByteV2 non-default payload formats currently require "
                  "assume-no-outlier FA2-direct decode");
  STD_TORCH_CHECK(payload_format == kByteV2PayloadFormatDefault ||
                      direct_diagnostic_mode == kByteV2DirectDiagnosticCurrent,
                  "ByteV2 non-default payload formats currently support only "
                  "the current FA2-direct mode");

  const int32_t* page_unsafe_flags_ptr =
      page_unsafe_flags.const_data_ptr<int32_t>();
  if (tile_policy[3] == ByteV2DefaultPolicy::ComputeBlockN) {
    if (use_gqa_packed) {
      if (use_gqa_fa2_like) {
        if (use_gqa_fa2_multiwarp) {
          launch_byte_v2_paged_decode_attention_split_k_gqa4_fa2_like_no_fallback_no_outlier<
              ByteV2DefaultLayout, true, true, false, true, false>(
              output, exp_sums, max_logits, tmp_out, query, kv_cache,
              page_unsafe_flags_ptr, block_tables, seq_lens, scale,
              partition_size);
        } else if (use_gqa_fa2_mainloop) {
          launch_byte_v2_paged_decode_attention_split_k_gqa4_fa2_like_no_fallback_no_outlier<
              ByteV2DefaultLayout, true, true, true, false, false>(
              output, exp_sums, max_logits, tmp_out, query, kv_cache,
              page_unsafe_flags_ptr, block_tables, seq_lens, scale,
              partition_size);
        } else if (use_gqa_fa2_direct) {
          if (payload_format == kByteV2PayloadFormatHighByte) {
            launch_byte_v2_paged_decode_attention_split_k_fa2_direct_grouped_no_fallback_no_outlier<
                ByteV2HighByteLayout, true>(
                output, exp_sums, max_logits, tmp_out, query, kv_cache,
                page_unsafe_flags_ptr, block_tables, seq_lens, scale,
                partition_size, q_heads_per_kv, direct_diagnostic_mode);
          } else if (payload_format == kByteV2PayloadFormatSidebandHigh) {
            launch_byte_v2_paged_decode_attention_split_k_fa2_direct_grouped_no_fallback_no_outlier<
                ByteV2SidebandHighLayout, true>(
                output, exp_sums, max_logits, tmp_out, query, kv_cache,
                page_unsafe_flags_ptr, block_tables, seq_lens, scale,
                partition_size, q_heads_per_kv, direct_diagnostic_mode);
          } else {
            launch_byte_v2_paged_decode_attention_split_k_fa2_direct_grouped_no_fallback_no_outlier<
                ByteV2DefaultLayout, true>(
                output, exp_sums, max_logits, tmp_out, query, kv_cache,
                page_unsafe_flags_ptr, block_tables, seq_lens, scale,
                partition_size, q_heads_per_kv, direct_diagnostic_mode);
          }
        } else if (use_gqa_fa2_qk_mma) {
          launch_byte_v2_paged_decode_attention_split_k_gqa4_fa2_like_no_fallback_no_outlier<
              ByteV2DefaultLayout, true, true, false, false, false>(
              output, exp_sums, max_logits, tmp_out, query, kv_cache,
              page_unsafe_flags_ptr, block_tables, seq_lens, scale,
              partition_size);
        } else {
          launch_byte_v2_paged_decode_attention_split_k_gqa4_fa2_like_no_fallback_no_outlier<
              ByteV2DefaultLayout, true, false, false, false, false>(
              output, exp_sums, max_logits, tmp_out, query, kv_cache,
              page_unsafe_flags_ptr, block_tables, seq_lens, scale,
              partition_size);
        }
      } else {
        launch_byte_v2_paged_decode_attention_split_k_gqa4_no_fallback_no_outlier<
            ByteV2DefaultLayout, true>(output, exp_sums, max_logits, tmp_out,
                                       query, kv_cache, page_unsafe_flags_ptr,
                                       block_tables, seq_lens, scale,
                                       partition_size);
      }
    } else {
      launch_byte_v2_paged_decode_attention_split_k<ByteV2DefaultLayout, false,
                                                    true, true>(
          output, exp_sums, max_logits, tmp_out, query, kv_cache,
          page_unsafe_flags_ptr, block_tables, seq_lens, scale, partition_size);
    }
  } else if (tile_policy[3] == ByteV2BN128Policy::ComputeBlockN) {
    if (use_gqa_packed) {
      launch_byte_v2_paged_decode_attention_split_k_gqa4_no_fallback_no_outlier<
          ByteV2BN128Layout, true>(
          output, exp_sums, max_logits, tmp_out, query, kv_cache,
          page_unsafe_flags_ptr, block_tables, seq_lens, scale, partition_size);
    } else {
      launch_byte_v2_paged_decode_attention_split_k<ByteV2BN128Layout, false,
                                                    true, true>(
          output, exp_sums, max_logits, tmp_out, query, kv_cache,
          page_unsafe_flags_ptr, block_tables, seq_lens, scale, partition_size);
    }
  } else {
    STD_TORCH_CHECK(false,
                    "ByteV2 guarded decode currently supports "
                    "compute_block_n=64 or compute_block_n=128");
  }
}

void byte_v2_speculative_verify_gqa(
    torch::stable::Tensor& output, torch::stable::Tensor& exp_sums,
    torch::stable::Tensor& max_logits, torch::stable::Tensor& tmp_out,
    torch::stable::Tensor& query, torch::stable::Tensor& kv_cache,
    torch::stable::Tensor& page_unsafe_flags,
    torch::stable::Tensor& block_tables, torch::stable::Tensor& seq_lens,
    int64_t speculative_query_len, double scale, int64_t num_kv_heads,
    int64_t block_size, int64_t max_seq_len, int64_t partition_size,
    const std::vector<int64_t>& tile_policy) {
  using torch::headeronly::ScalarType;

  constexpr int64_t kQueryHeadsPerKv = 4;
  const int64_t virtual_rows_per_kv = speculative_query_len * kQueryHeadsPerKv;
  const int64_t virtual_heads =
      ByteV2DefaultLayout::NumKvHeadsValue * virtual_rows_per_kv;
  const int direct_diagnostic_mode =
      byte_v2_gqa_fa2_direct_diagnostic_mode(tile_policy);

  STD_TORCH_CHECK(query.device().is_cuda(), "query must be a CUDA tensor");
  STD_TORCH_CHECK(output.device() == query.device(),
                  "output and query must be on the same device");
  STD_TORCH_CHECK(kv_cache.device() == query.device(),
                  "kv_cache and query must be on the same device");
  STD_TORCH_CHECK(page_unsafe_flags.device() == query.device(),
                  "page_unsafe_flags and query must be on the same device");
  STD_TORCH_CHECK(block_tables.device() == query.device(),
                  "block_tables and query must be on the same device");
  STD_TORCH_CHECK(seq_lens.device() == query.device(),
                  "seq_lens and query must be on the same device");
  STD_TORCH_CHECK(exp_sums.device() == query.device(),
                  "exp_sums and query must be on the same device");
  STD_TORCH_CHECK(tmp_out.device() == query.device(),
                  "tmp_out and query must be on the same device");
  STD_TORCH_CHECK(
      direct_diagnostic_mode == kByteV2DirectDiagnosticPhaseProfile
          ? max_logits.numel() >= 20
          : max_logits.numel() == 0,
      "ByteV2 GQA verify only uses max_logits for phase-profile diagnostics");
  STD_TORCH_CHECK(speculative_query_len == 2 || speculative_query_len == 4 ||
                      speculative_query_len == 8 || speculative_query_len == 16,
                  "ByteV2 GQA verify currently supports Q2, Q4, Q8, or Q16");
  STD_TORCH_CHECK(direct_diagnostic_mode == kByteV2DirectDiagnosticCurrent ||
                      speculative_query_len == 16,
                  "ByteV2 speculative diagnostics currently require Q16");
  STD_TORCH_CHECK(
      direct_diagnostic_mode == kByteV2DirectDiagnosticCurrent ||
          direct_diagnostic_mode == kByteV2DirectDiagnosticDecodeStageOnly ||
          direct_diagnostic_mode == kByteV2DirectDiagnosticFakeDecodeZero ||
          direct_diagnostic_mode == kByteV2DirectDiagnosticRawSameSkeleton ||
          direct_diagnostic_mode == kByteV2DirectDiagnosticPhaseProfile ||
          direct_diagnostic_mode == kByteV2DirectDiagnosticCodeReady,
      "unsupported ByteV2 Q16 speculative diagnostic mode");

  STD_TORCH_CHECK(output.dim() == 3 && query.sizes().equals(output.sizes()),
                  "query and output must both be [requests * Q, 32, 128]");
  STD_TORCH_CHECK(output.size(0) % speculative_query_len == 0,
                  "ByteV2 GQA verify requires uniform queries per request");
  STD_TORCH_CHECK(
      output.size(1) == 32 && output.size(2) == ByteV2DefaultPolicy::HeadDim,
      "ByteV2 GQA verify currently supports 32 query heads and "
      "head_dim=128");
  const int64_t num_requests = output.size(0) / speculative_query_len;

  STD_TORCH_CHECK(kv_cache.dim() == 2,
                  "kv_cache must be [num_blocks, page_size_bytes]");
  STD_TORCH_CHECK(page_unsafe_flags.dim() == 1,
                  "page_unsafe_flags must be a 1D tensor");
  STD_TORCH_CHECK(block_tables.dim() == 2,
                  "block_tables must be [num_requests, max_num_blocks]");
  STD_TORCH_CHECK(seq_lens.dim() == 1, "seq_lens must be [num_requests]");
  STD_TORCH_CHECK(block_tables.size(0) >= num_requests,
                  "block_tables must contain every request");
  STD_TORCH_CHECK(seq_lens.size(0) >= num_requests,
                  "seq_lens must contain every request");
  STD_TORCH_CHECK(page_unsafe_flags.size(0) >= kv_cache.size(0),
                  "page_unsafe_flags must cover every kv_cache block");

  STD_TORCH_CHECK(query.scalar_type() == ScalarType::BFloat16,
                  "ByteV2 GQA verify requires bf16 query tensors");
  STD_TORCH_CHECK(output.scalar_type() == query.scalar_type(),
                  "output and query dtypes must match");
  STD_TORCH_CHECK(kv_cache.scalar_type() == ScalarType::Byte,
                  "kv_cache must use uint8 storage");
  STD_TORCH_CHECK(page_unsafe_flags.scalar_type() == ScalarType::Int,
                  "page_unsafe_flags must use int32 storage");
  STD_TORCH_CHECK(block_tables.scalar_type() == ScalarType::Int,
                  "block_tables must use int32 storage");
  STD_TORCH_CHECK(seq_lens.scalar_type() == ScalarType::Int,
                  "seq_lens must use int32 storage");
  STD_TORCH_CHECK(exp_sums.scalar_type() == ScalarType::Float,
                  "exp_sums must use float32 storage");
  STD_TORCH_CHECK(tmp_out.scalar_type() == ScalarType::Float,
                  "tmp_out must use float32 storage");
  STD_TORCH_CHECK(exp_sums.is_contiguous(), "exp_sums must be contiguous");
  STD_TORCH_CHECK(tmp_out.is_contiguous(), "tmp_out must be contiguous");
  STD_TORCH_CHECK(page_unsafe_flags.stride(0) == 1,
                  "page_unsafe_flags must be contiguous");
  STD_TORCH_CHECK(kv_cache.stride(1) == 1,
                  "kv_cache page dimension must be contiguous");
  STD_TORCH_CHECK(block_tables.stride(1) == 1,
                  "block_tables block dimension must be contiguous");

  STD_TORCH_CHECK(num_kv_heads == ByteV2DefaultLayout::NumKvHeadsValue,
                  "ByteV2 GQA verify currently supports 8 KV heads");
  STD_TORCH_CHECK(block_size == ByteV2DefaultPolicy::AllocBlockTokens,
                  "ByteV2 GQA verify currently supports block_size=16");
  STD_TORCH_CHECK(max_seq_len >= speculative_query_len,
                  "max_seq_len must include all speculative tokens");
  STD_TORCH_CHECK(partition_size > 0 && partition_size % block_size == 0,
                  "partition_size must be positive and divisible by 16");
  STD_TORCH_CHECK(scale > 0.0, "scale must be positive");
  check_byte_v2_tile_policy(tile_policy);
  STD_TORCH_CHECK(
      tile_policy[0] == ByteV2DefaultPolicy::CodecTokenBlock &&
          tile_policy[1] == ByteV2DefaultPolicy::CodecDimBlock &&
          tile_policy[2] == ByteV2DefaultPolicy::AllocBlockTokens &&
          tile_policy[3] == ByteV2DefaultPolicy::ComputeBlockN &&
          tile_policy[4] == ByteV2DefaultPolicy::HeadDim &&
          tile_policy[5] == ByteV2DefaultPolicy::HeadDimV,
      "ByteV2 GQA verify requires the default 16x16/BN64/H128 policy");
  const int64_t required_cache_stride =
      direct_diagnostic_mode == kByteV2DirectDiagnosticRawSameSkeleton
          ? ByteV2DefaultRawStagingLayout::SlotSizeBytes
          : ByteV2DefaultLayout::PageSizeBytes;
  STD_TORCH_CHECK(kv_cache.size(1) >= required_cache_stride &&
                      kv_cache.stride(0) >= required_cache_stride,
                  "kv_cache page is smaller than the selected Q16 diagnostic "
                  "layout");
  STD_TORCH_CHECK(
      block_tables.size(1) >=
          (max_seq_len + ByteV2DefaultPolicy::AllocBlockTokens - 1) /
              ByteV2DefaultPolicy::AllocBlockTokens,
      "block_tables is too small for max_seq_len");

  const int64_t required_partitions =
      (max_seq_len + partition_size - 1) / partition_size;
  STD_TORCH_CHECK(exp_sums.dim() == 3 && exp_sums.size(0) >= num_requests &&
                      exp_sums.size(1) >= virtual_heads &&
                      exp_sums.size(2) >= required_partitions &&
                      exp_sums.size(2) > 0,
                  "exp_sums workspace must be [requests, 32 * Q, partitions]");
  STD_TORCH_CHECK(tmp_out.dim() == 4 && tmp_out.size(0) >= num_requests &&
                      tmp_out.size(1) >= virtual_heads &&
                      tmp_out.size(2) == exp_sums.size(2) &&
                      tmp_out.size(2) >= required_partitions &&
                      tmp_out.size(3) == ByteV2DefaultPolicy::HeadDimV,
                  "tmp_out workspace must be [requests, 32 * Q, partitions, "
                  "128]");

  if (speculative_query_len == 2) {
    launch_byte_v2_paged_decode_attention_split_k_gqa4_fa2_like_no_fallback_no_outlier<
        ByteV2DefaultLayout, true, true, false, false, true, 8, 8,
        kByteV2DirectDiagnosticCurrent, 2>(
        output, exp_sums, max_logits, tmp_out, query, kv_cache,
        page_unsafe_flags.const_data_ptr<int32_t>(), block_tables, seq_lens,
        scale, partition_size);
  } else if (speculative_query_len == 4) {
    launch_byte_v2_paged_decode_attention_split_k_gqa4_fa2_like_no_fallback_no_outlier<
        ByteV2DefaultLayout, true, true, false, false, true, 16, 16,
        kByteV2DirectDiagnosticCurrent, 4>(
        output, exp_sums, max_logits, tmp_out, query, kv_cache,
        page_unsafe_flags.const_data_ptr<int32_t>(), block_tables, seq_lens,
        scale, partition_size);
  } else if (speculative_query_len == 8) {
    launch_byte_v2_paged_decode_attention_split_k_gqa4_fa2_like_no_fallback_no_outlier<
        ByteV2DefaultLayout, true, true, false, false, true, 32, 32,
        kByteV2DirectDiagnosticCurrent, 8>(
        output, exp_sums, max_logits, tmp_out, query, kv_cache,
        page_unsafe_flags.const_data_ptr<int32_t>(), block_tables, seq_lens,
        scale, partition_size);
  } else {
#define BYTE_V2_LAUNCH_Q16_DIAGNOSTIC(mode)                                           \
  launch_byte_v2_paged_decode_attention_split_k_gqa4_fa2_like_no_fallback_no_outlier< \
      ByteV2DefaultLayout, true, true, false, false, true, 64, 64, mode, 16>(         \
      output, exp_sums, max_logits, tmp_out, query, kv_cache,                         \
      page_unsafe_flags.const_data_ptr<int32_t>(), block_tables, seq_lens,            \
      scale, partition_size)
    switch (direct_diagnostic_mode) {
      case kByteV2DirectDiagnosticCurrent:
        BYTE_V2_LAUNCH_Q16_DIAGNOSTIC(kByteV2DirectDiagnosticCurrent);
        break;
      case kByteV2DirectDiagnosticDecodeStageOnly:
        BYTE_V2_LAUNCH_Q16_DIAGNOSTIC(kByteV2DirectDiagnosticDecodeStageOnly);
        break;
      case kByteV2DirectDiagnosticFakeDecodeZero:
        BYTE_V2_LAUNCH_Q16_DIAGNOSTIC(kByteV2DirectDiagnosticFakeDecodeZero);
        break;
      case kByteV2DirectDiagnosticRawSameSkeleton:
        BYTE_V2_LAUNCH_Q16_DIAGNOSTIC(kByteV2DirectDiagnosticRawSameSkeleton);
        break;
      case kByteV2DirectDiagnosticPhaseProfile:
        BYTE_V2_LAUNCH_Q16_DIAGNOSTIC(kByteV2DirectDiagnosticPhaseProfile);
        break;
      case kByteV2DirectDiagnosticCodeReady:
        BYTE_V2_LAUNCH_Q16_DIAGNOSTIC(kByteV2DirectDiagnosticCodeReady);
        break;
      default:
        STD_TORCH_CHECK(false,
                        "unsupported ByteV2 Q16 speculative diagnostic mode");
    }
#undef BYTE_V2_LAUNCH_Q16_DIAGNOSTIC
  }
}

void byte_v2_speculative_verify_ragged_q4(
    torch::stable::Tensor& output, torch::stable::Tensor& exp_sums,
    torch::stable::Tensor& max_logits, torch::stable::Tensor& tmp_out,
    torch::stable::Tensor& query, torch::stable::Tensor& kv_cache,
    torch::stable::Tensor& page_unsafe_flags,
    torch::stable::Tensor& block_tables, torch::stable::Tensor& seq_lens,
    torch::stable::Tensor& query_start_locs, int64_t num_actual_tokens,
    double scale, int64_t num_kv_heads, int64_t block_size, int64_t max_seq_len,
    int64_t partition_size, const std::vector<int64_t>& tile_policy) {
  using torch::headeronly::ScalarType;

  constexpr int64_t kSpeculativeQueryLen = 4;
  constexpr int64_t kQueryHeadsPerKv = 4;
  constexpr int64_t kVirtualRowsPerKv = kSpeculativeQueryLen * kQueryHeadsPerKv;
  constexpr int64_t kVirtualHeads =
      ByteV2DefaultLayout::NumKvHeadsValue * kVirtualRowsPerKv;

  STD_TORCH_CHECK(query.device().is_cuda(), "query must be a CUDA tensor");
  STD_TORCH_CHECK(output.device() == query.device(),
                  "output and query must be on the same device");
  STD_TORCH_CHECK(kv_cache.device() == query.device(),
                  "kv_cache and query must be on the same device");
  STD_TORCH_CHECK(page_unsafe_flags.device() == query.device(),
                  "page_unsafe_flags and query must be on the same device");
  STD_TORCH_CHECK(block_tables.device() == query.device(),
                  "block_tables and query must be on the same device");
  STD_TORCH_CHECK(seq_lens.device() == query.device(),
                  "seq_lens and query must be on the same device");
  STD_TORCH_CHECK(query_start_locs.device() == query.device(),
                  "query_start_locs and query must be on the same device");
  STD_TORCH_CHECK(exp_sums.device() == query.device(),
                  "exp_sums and query must be on the same device");
  STD_TORCH_CHECK(tmp_out.device() == query.device(),
                  "tmp_out and query must be on the same device");
  STD_TORCH_CHECK(max_logits.device() == query.device(),
                  "max_logits and query must be on the same device");
  STD_TORCH_CHECK(max_logits.numel() == 0,
                  "ByteV2 ragged Q4 does not use max_logits");

  STD_TORCH_CHECK(output.dim() == 3 && query.sizes().equals(output.sizes()),
                  "query and output must both be [packed_tokens, 32, 128]");
  STD_TORCH_CHECK(
      output.size(1) == 32 && output.size(2) == ByteV2DefaultPolicy::HeadDim,
      "ByteV2 ragged Q4 currently supports 32 query heads and head_dim=128");
  STD_TORCH_CHECK(num_actual_tokens >= 0 && num_actual_tokens <= output.size(0),
                  "num_actual_tokens must fit in the packed output");
  STD_TORCH_CHECK(query_start_locs.dim() == 1 && query_start_locs.size(0) >= 2,
                  "query_start_locs must contain at least one request");
  const int64_t num_requests = query_start_locs.size(0) - 1;
  STD_TORCH_CHECK(num_actual_tokens <= num_requests * kSpeculativeQueryLen,
                  "ragged Q4 supports at most four tokens per request");

  STD_TORCH_CHECK(kv_cache.dim() == 2,
                  "kv_cache must be [num_blocks, page_size_bytes]");
  STD_TORCH_CHECK(page_unsafe_flags.dim() == 1,
                  "page_unsafe_flags must be a 1D tensor");
  STD_TORCH_CHECK(block_tables.dim() == 2,
                  "block_tables must be [num_requests, max_num_blocks]");
  STD_TORCH_CHECK(seq_lens.dim() == 1, "seq_lens must be [num_requests]");
  STD_TORCH_CHECK(block_tables.size(0) >= num_requests,
                  "block_tables must contain every request");
  STD_TORCH_CHECK(seq_lens.size(0) >= num_requests,
                  "seq_lens must contain every request");
  STD_TORCH_CHECK(page_unsafe_flags.size(0) >= kv_cache.size(0),
                  "page_unsafe_flags must cover every kv_cache block");

  STD_TORCH_CHECK(query.scalar_type() == ScalarType::BFloat16,
                  "ByteV2 ragged Q4 requires bf16 query tensors");
  STD_TORCH_CHECK(output.scalar_type() == query.scalar_type(),
                  "output and query dtypes must match");
  STD_TORCH_CHECK(kv_cache.scalar_type() == ScalarType::Byte,
                  "kv_cache must use uint8 storage");
  STD_TORCH_CHECK(page_unsafe_flags.scalar_type() == ScalarType::Int,
                  "page_unsafe_flags must use int32 storage");
  STD_TORCH_CHECK(block_tables.scalar_type() == ScalarType::Int,
                  "block_tables must use int32 storage");
  STD_TORCH_CHECK(seq_lens.scalar_type() == ScalarType::Int,
                  "seq_lens must use int32 storage");
  STD_TORCH_CHECK(query_start_locs.scalar_type() == ScalarType::Int,
                  "query_start_locs must use int32 storage");
  STD_TORCH_CHECK(exp_sums.scalar_type() == ScalarType::Float,
                  "exp_sums must use float32 storage");
  STD_TORCH_CHECK(tmp_out.scalar_type() == ScalarType::Float,
                  "tmp_out must use float32 storage");
  STD_TORCH_CHECK(exp_sums.is_contiguous(), "exp_sums must be contiguous");
  STD_TORCH_CHECK(tmp_out.is_contiguous(), "tmp_out must be contiguous");
  STD_TORCH_CHECK(query_start_locs.stride(0) == 1,
                  "query_start_locs must be contiguous");
  STD_TORCH_CHECK(seq_lens.stride(0) == 1, "seq_lens must be contiguous");
  STD_TORCH_CHECK(page_unsafe_flags.stride(0) == 1,
                  "page_unsafe_flags must be contiguous");
  STD_TORCH_CHECK(kv_cache.stride(1) == 1,
                  "kv_cache page dimension must be contiguous");
  STD_TORCH_CHECK(block_tables.stride(1) == 1,
                  "block_tables block dimension must be contiguous");

  STD_TORCH_CHECK(num_kv_heads == ByteV2DefaultLayout::NumKvHeadsValue,
                  "ByteV2 ragged Q4 currently supports 8 KV heads");
  STD_TORCH_CHECK(block_size == ByteV2DefaultPolicy::AllocBlockTokens,
                  "ByteV2 ragged Q4 currently supports block_size=16");
  STD_TORCH_CHECK(
      max_seq_len > 0 && max_seq_len <= std::numeric_limits<int32_t>::max(),
      "max_seq_len must fit in int32");
  STD_TORCH_CHECK(partition_size > 0 && partition_size % block_size == 0,
                  "partition_size must be positive and divisible by 16");
  STD_TORCH_CHECK(partition_size <= std::numeric_limits<int>::max(),
                  "partition_size must fit in int32");
  STD_TORCH_CHECK(scale > 0.0, "scale must be positive");
  check_byte_v2_tile_policy(tile_policy);
  STD_TORCH_CHECK(
      tile_policy[0] == ByteV2DefaultPolicy::CodecTokenBlock &&
          tile_policy[1] == ByteV2DefaultPolicy::CodecDimBlock &&
          tile_policy[2] == ByteV2DefaultPolicy::AllocBlockTokens &&
          tile_policy[3] == ByteV2DefaultPolicy::ComputeBlockN &&
          tile_policy[4] == ByteV2DefaultPolicy::HeadDim &&
          tile_policy[5] == ByteV2DefaultPolicy::HeadDimV,
      "ByteV2 ragged Q4 requires the default 16x16/BN64/H128 policy");
  STD_TORCH_CHECK(kv_cache.size(1) >= ByteV2DefaultLayout::PageSizeBytes &&
                      kv_cache.stride(0) >= ByteV2DefaultLayout::PageSizeBytes,
                  "kv_cache page is smaller than the ByteV2 V5 layout");
  STD_TORCH_CHECK(
      block_tables.size(1) >=
          (max_seq_len + ByteV2DefaultPolicy::AllocBlockTokens - 1) /
              ByteV2DefaultPolicy::AllocBlockTokens,
      "block_tables is too small for max_seq_len");

  const int64_t required_partitions =
      (max_seq_len + partition_size - 1) / partition_size;
  STD_TORCH_CHECK(exp_sums.dim() == 3 && exp_sums.size(0) >= num_requests &&
                      exp_sums.size(1) >= kVirtualHeads &&
                      exp_sums.size(2) >= required_partitions &&
                      exp_sums.size(2) > 0,
                  "exp_sums workspace must be [requests, 128, partitions]");
  STD_TORCH_CHECK(tmp_out.dim() == 4 && tmp_out.size(0) >= num_requests &&
                      tmp_out.size(1) >= kVirtualHeads &&
                      tmp_out.size(2) == exp_sums.size(2) &&
                      tmp_out.size(2) >= required_partitions &&
                      tmp_out.size(3) == ByteV2DefaultPolicy::HeadDimV,
                  "tmp_out workspace must be "
                  "[requests, 128, partitions, 128]");

  launch_byte_v2_paged_decode_attention_split_k_gqa4_fa2_like_no_fallback_no_outlier<
      ByteV2DefaultLayout, true, true, false, false, true, 16, 16,
      kByteV2DirectDiagnosticCurrent, 4, true>(
      output, exp_sums, max_logits, tmp_out, query, kv_cache,
      page_unsafe_flags.const_data_ptr<int32_t>(), block_tables, seq_lens,
      scale, partition_size, query_start_locs.const_data_ptr<int32_t>(),
      static_cast<int>(num_requests), static_cast<int>(num_actual_tokens));
}

void byte_v2_speculative_verify_q4(
    torch::stable::Tensor& output, torch::stable::Tensor& exp_sums,
    torch::stable::Tensor& max_logits, torch::stable::Tensor& tmp_out,
    torch::stable::Tensor& query, torch::stable::Tensor& kv_cache,
    torch::stable::Tensor& page_unsafe_flags,
    torch::stable::Tensor& block_tables, torch::stable::Tensor& seq_lens,
    double scale, int64_t num_kv_heads, int64_t block_size, int64_t max_seq_len,
    int64_t partition_size, const std::vector<int64_t>& tile_policy) {
  byte_v2_speculative_verify_gqa(output, exp_sums, max_logits, tmp_out, query,
                                 kv_cache, page_unsafe_flags, block_tables,
                                 seq_lens, 4, scale, num_kv_heads, block_size,
                                 max_seq_len, partition_size, tile_policy);
}

void byte_v2_prefill_attention(torch::stable::Tensor& output,
                               torch::stable::Tensor& query,
                               torch::stable::Tensor& key,
                               torch::stable::Tensor& value,
                               torch::stable::Tensor& query_start_loc,
                               int64_t max_query_len, double scale,
                               int64_t num_kv_heads, bool causal,
                               const std::vector<int64_t>& tile_policy) {
  using torch::headeronly::ScalarType;

  check_byte_v2_tile_policy(tile_policy);
  STD_TORCH_CHECK(query.device().is_cuda(), "query must be a CUDA tensor");
  STD_TORCH_CHECK(output.device() == query.device(),
                  "output and query must be on the same device");
  STD_TORCH_CHECK(query.device() == key.device(),
                  "query and key must be on the same device");
  STD_TORCH_CHECK(query.device() == value.device(),
                  "query and value must be on the same device");
  STD_TORCH_CHECK(query.device() == query_start_loc.device(),
                  "query and query_start_loc must be on the same device");
  STD_TORCH_CHECK(output.dim() == 3,
                  "output must be [num_tokens, num_heads, head_dim]");
  STD_TORCH_CHECK(query.sizes().equals(output.sizes()),
                  "query and output must have the same shape");
  STD_TORCH_CHECK(key.dim() == 3,
                  "key must be [num_tokens, num_kv_heads, head_dim]");
  STD_TORCH_CHECK(value.dim() == 3,
                  "value must be [num_tokens, num_kv_heads, head_dim_v]");
  STD_TORCH_CHECK(query_start_loc.dim() == 1,
                  "query_start_loc must be [num_reqs + 1]");
  STD_TORCH_CHECK(query_start_loc.size(0) >= 2,
                  "query_start_loc must contain at least one request");
  STD_TORCH_CHECK(key.size(0) >= output.size(0),
                  "key must contain all query tokens");
  STD_TORCH_CHECK(value.size(0) >= output.size(0),
                  "value must contain all query tokens");
  STD_TORCH_CHECK(num_kv_heads > 0, "num_kv_heads must be positive");
  STD_TORCH_CHECK(scale > 0.0, "scale must be positive");
  STD_TORCH_CHECK(output.size(1) % num_kv_heads == 0,
                  "num_heads must be divisible by num_kv_heads");
  STD_TORCH_CHECK(key.size(1) == num_kv_heads,
                  "key head count must match num_kv_heads");
  STD_TORCH_CHECK(value.size(1) == num_kv_heads,
                  "value head count must match num_kv_heads");
  STD_TORCH_CHECK(max_query_len > 0, "max_query_len must be positive");

  STD_TORCH_CHECK(query.scalar_type() == ScalarType::BFloat16,
                  "ByteV2 prefill currently supports bf16 query tensors");
  STD_TORCH_CHECK(output.scalar_type() == query.scalar_type(),
                  "output and query dtypes must match");
  STD_TORCH_CHECK(key.scalar_type() == query.scalar_type(),
                  "key and query dtypes must match");
  STD_TORCH_CHECK(value.scalar_type() == query.scalar_type(),
                  "value and query dtypes must match");
  STD_TORCH_CHECK(query_start_loc.scalar_type() == ScalarType::Int,
                  "query_start_loc must be int32");
  STD_TORCH_CHECK(output.size(2) == ByteV2DefaultPolicy::HeadDim,
                  "ByteV2 prefill currently supports head_dim=128");
  STD_TORCH_CHECK(key.size(2) == ByteV2DefaultPolicy::HeadDim,
                  "ByteV2 prefill currently supports key head_dim=128");
  STD_TORCH_CHECK(value.size(2) == ByteV2DefaultPolicy::HeadDimV,
                  "ByteV2 prefill currently supports value head_dim_v=128");
  STD_TORCH_CHECK(query_start_loc.stride(0) == 1,
                  "query_start_loc must be contiguous");
  STD_TORCH_CHECK(tile_policy[0] == ByteV2DefaultPolicy::CodecTokenBlock &&
                      tile_policy[1] == ByteV2DefaultPolicy::CodecDimBlock &&
                      tile_policy[2] == ByteV2DefaultPolicy::AllocBlockTokens &&
                      tile_policy[4] == ByteV2DefaultPolicy::HeadDim &&
                      tile_policy[5] == ByteV2DefaultPolicy::HeadDimV,
                  "ByteV2 prefill currently supports 16x16 codec tiles, "
                  "block_size=16, head_dim=128, and head_dim_v=128");

  const int64_t num_tokens = output.size(0);
  if (num_tokens == 0 || output.size(1) == 0) {
    return;
  }

  const torch::stable::accelerator::DeviceGuard device_guard(
      query.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream(query.get_device_index());

  constexpr int kThreads = ByteV2DefaultPolicy::HeadDim;
  const int64_t num_reqs = query_start_loc.size(0) - 1;
  const int64_t grid_y = num_reqs * max_query_len;
  STD_TORCH_CHECK(grid_y <= std::numeric_limits<unsigned int>::max(),
                  "ByteV2 prefill grid is too large for CUDA grid.y");
  const dim3 grid(static_cast<unsigned int>(output.size(1)),
                  static_cast<unsigned int>(grid_y), 1);

  if (tile_policy[3] == ByteV2DefaultPolicy::ComputeBlockN) {
    byte_v2_prefill_attention_kernel<ByteV2DefaultPolicy::ComputeBlockN,
                                     kThreads><<<grid, kThreads, 0, stream>>>(
        reinterpret_cast<uint16_t*>(output.mutable_data_ptr()),
        reinterpret_cast<const uint16_t*>(query.const_data_ptr()),
        reinterpret_cast<const uint16_t*>(key.const_data_ptr()),
        reinterpret_cast<const uint16_t*>(value.const_data_ptr()),
        query_start_loc.const_data_ptr<int32_t>(), num_tokens, num_reqs,
        max_query_len, output.size(1), num_kv_heads, causal,
        static_cast<float>(scale), query.stride(0), query.stride(1),
        query.stride(2), key.stride(0), key.stride(1), key.stride(2),
        value.stride(0), value.stride(1), value.stride(2), output.stride(0),
        output.stride(1), output.stride(2));
  } else if (tile_policy[3] == ByteV2BN128Policy::ComputeBlockN) {
    byte_v2_prefill_attention_kernel<ByteV2BN128Policy::ComputeBlockN, kThreads>
        <<<grid, kThreads, 0, stream>>>(
            reinterpret_cast<uint16_t*>(output.mutable_data_ptr()),
            reinterpret_cast<const uint16_t*>(query.const_data_ptr()),
            reinterpret_cast<const uint16_t*>(key.const_data_ptr()),
            reinterpret_cast<const uint16_t*>(value.const_data_ptr()),
            query_start_loc.const_data_ptr<int32_t>(), num_tokens, num_reqs,
            max_query_len, output.size(1), num_kv_heads, causal,
            static_cast<float>(scale), query.stride(0), query.stride(1),
            query.stride(2), key.stride(0), key.stride(1), key.stride(2),
            value.stride(0), value.stride(1), value.stride(2), output.stride(0),
            output.stride(1), output.stride(2));
  } else {
    STD_TORCH_CHECK(false,
                    "ByteV2 prefill currently supports compute_block_n=64 or "
                    "compute_block_n=128");
  }
  const cudaError_t err = cudaGetLastError();
  STD_TORCH_CHECK(err == cudaSuccess,
                  "byte_v2_prefill_attention kernel launch failed: ",
                  cudaGetErrorString(err));
}
