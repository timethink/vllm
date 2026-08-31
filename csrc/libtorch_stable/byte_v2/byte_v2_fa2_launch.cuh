// Copyright (c) 2026, ByteV2 contributors.
//
// ByteV2/SplitZip-specific FA2 launch path.  The attention mainloop is the
// original FA2 split-KV implementation; only its global-to-shared K/V load
// policy changes.

#pragma once

#include <cstdlib>
#include <cstring>
#include <type_traits>

#include "flash_fwd_launch_template.h"
#include "byte_v2_fa2_loader.cuh"

namespace FLASH_NAMESPACE {

DEFINE_FLASH_FORWARD_KERNEL(flash_fwd_splitkv_byte_v2_kernel, bool Is_causal,
                            bool Split, bool ReuseKvSmemNonsplit,
                            vllm::byte_v2::fa2::ExternalKvFormat Format) {
#if defined(ARCH_SUPPORTS_FLASH)
  constexpr bool IsSplitZip =
      Format == vllm::byte_v2::fa2::ExternalKvFormat::SplitZip;
  constexpr bool IsStaticW16Canonical =
      Format == vllm::byte_v2::fa2::ExternalKvFormat::StaticW16Canonical;
  constexpr bool IsStaticW16 =
      Format == vllm::byte_v2::fa2::ExternalKvFormat::StaticW16 ||
      IsStaticW16Canonical;
  using LegacyByteV2Loader =
      std::conditional_t<ReuseKvSmemNonsplit,
                         vllm::byte_v2::fa2::LoaderReuseKvSmemNonsplit,
                         vllm::byte_v2::fa2::Loader>;
  using LegacySplitZipLoader =
      std::conditional_t<ReuseKvSmemNonsplit,
                         vllm::byte_v2::fa2::SplitZipLoaderReuseKvSmemNonsplit,
                         vllm::byte_v2::fa2::SplitZipLoader>;
  using LegacyStaticW16CanonicalLoader = std::conditional_t<
      ReuseKvSmemNonsplit,
      vllm::byte_v2::fa2::StaticW16CanonicalLoaderReuseKvSmemNonsplit,
      vllm::byte_v2::fa2::StaticW16CanonicalLoader>;
  using LegacyStaticW16Loader = std::conditional_t<
      IsStaticW16Canonical, LegacyStaticW16CanonicalLoader,
      std::conditional_t<ReuseKvSmemNonsplit,
                         vllm::byte_v2::fa2::StaticW16LoaderReuseKvSmemNonsplit,
                         vllm::byte_v2::fa2::StaticW16Loader>>;
  using LegacyExternalKvLoader =
      std::conditional_t<IsSplitZip, LegacySplitZipLoader,
                         std::conditional_t<IsStaticW16, LegacyStaticW16Loader,
                                            LegacyByteV2Loader>>;
  constexpr bool UseSharedPageDescriptor =
      vllm::byte_v2::fa2::kReuseKvSmem && vllm::byte_v2::fa2::kStageMode != 0 &&
      (Split || ReuseKvSmemNonsplit);
  using StaticW16SharedPageLoader = std::conditional_t<
      IsStaticW16Canonical,
      vllm::byte_v2::fa2::StaticW16CanonicalLoaderSharedPageDescriptor,
      vllm::byte_v2::fa2::StaticW16LoaderSharedPageDescriptor>;
  using SharedPageLoader = std::conditional_t<
      IsSplitZip, vllm::byte_v2::fa2::SplitZipLoaderSharedPageDescriptor,
      std::conditional_t<IsStaticW16, StaticW16SharedPageLoader,
                         vllm::byte_v2::fa2::LoaderSharedPageDescriptor>>;
  using ExternalKvLoader =
      std::conditional_t<UseSharedPageDescriptor, SharedPageLoader,
                         LegacyExternalKvLoader>;
  FLASH_NAMESPACE::compute_attn_splitkv<
      Kernel_traits, Is_causal, /*Is_local=*/false, /*Has_alibi=*/false,
      /*Is_even_MN=*/false, /*Is_even_K=*/true, /*Is_softcap=*/false, Split,
      /*Append_KV=*/false, ExternalKvLoader>(params);
#else
  FLASH_UNSUPPORTED_ARCH
#endif
}

// Profile-only Q1 feasibility kernel.  It preserves the cooperative W16
// stage/decode path but skips QK and PV MMA in query warps whose 16-row slice
// is entirely outside the valid query extent.  This is the correctness and
// resource gate for assigning those otherwise idle warps to a later
// producer/consumer decode pipeline.
DEFINE_FLASH_FORWARD_KERNEL(
    flash_fwd_splitkv_byte_v2_active_query_warp_profile_kernel, bool Is_causal,
    bool Split, typename ExternalKvLoader) {
#if defined(ARCH_SUPPORTS_FLASH)
  FLASH_NAMESPACE::compute_attn_splitkv<
      Kernel_traits, Is_causal, /*Is_local=*/false, /*Has_alibi=*/false,
      /*Is_even_MN=*/false, /*Is_even_K=*/true, /*Is_softcap=*/false, Split,
      /*Append_KV=*/false, ExternalKvLoader, /*Raw_kv_smem_alias=*/false,
      /*Active_query_warp_gate=*/true>(params);
#else
  FLASH_UNSUPPORTED_ARCH
#endif
}

// Profile-only one-CTA producer/consumer prototype.  The first four warps
// stage and decode W16 K/V while the second four warps execute the original
// logical 128-thread FA2 consumer mapping.
DEFINE_FLASH_FORWARD_KERNEL(
    flash_fwd_splitkv_byte_v2_one_cta_overlap_profile_kernel, bool Is_causal) {
#if defined(ARCH_SUPPORTS_FLASH)
  using ExternalKvLoader =
      vllm::byte_v2::fa2::StaticW16CanonicalLoaderOneCtaOverlap;
  FLASH_NAMESPACE::compute_attn_splitkv<
      Kernel_traits, Is_causal, /*Is_local=*/false, /*Has_alibi=*/false,
      /*Is_even_MN=*/false, /*Is_even_K=*/true, /*Is_softcap=*/false,
      /*Split=*/true, /*Append_KV=*/false, ExternalKvLoader,
      /*Raw_kv_smem_alias=*/false, /*Active_query_warp_gate=*/true,
      /*Warp_specialized_overlap=*/true>(params);
#else
  FLASH_UNSUPPORTED_ARCH
#endif
}

// Instrumented twin of the one-CTA prototype.  Keeping it as a separate
// template instantiation guarantees that the timing stores cannot perturb the
// SASS or benchmark result of the uninstrumented candidate.
DEFINE_FLASH_FORWARD_KERNEL(
    flash_fwd_splitkv_byte_v2_one_cta_overlap_trace_profile_kernel,
    bool Is_causal) {
#if defined(ARCH_SUPPORTS_FLASH)
  using ExternalKvLoader =
      vllm::byte_v2::fa2::StaticW16CanonicalLoaderOneCtaOverlap;
  FLASH_NAMESPACE::compute_attn_splitkv<
      Kernel_traits, Is_causal, /*Is_local=*/false, /*Has_alibi=*/false,
      /*Is_even_MN=*/false, /*Is_even_K=*/true, /*Is_softcap=*/false,
      /*Split=*/true, /*Append_KV=*/false, ExternalKvLoader,
      /*Raw_kv_smem_alias=*/false, /*Active_query_warp_gate=*/true,
      /*Warp_specialized_overlap=*/true,
      /*Warp_specialized_overlap_trace=*/true>(params);
#else
  FLASH_UNSUPPORTED_ARCH
#endif
}

// Q1-only structural follow-up: four warps retain the existing cp.async
// staging map, three additional producer warps split the arithmetic decode
// work, and one logical FA2 warp computes QK/PV.  All four logical consumer
// warps regroup for the epilogue because its vectorized output copy still
// partitions a Q1 result across the full 128-thread FA2 mapping.
DEFINE_FLASH_FORWARD_KERNEL(
    flash_fwd_splitkv_byte_v2_one_cta_7p1c_profile_kernel, bool Is_causal) {
#if defined(ARCH_SUPPORTS_FLASH)
  using ExternalKvLoader =
      vllm::byte_v2::fa2::StaticW16CanonicalLoaderOneCtaOverlap;
  FLASH_NAMESPACE::compute_attn_splitkv<
      Kernel_traits, Is_causal, /*Is_local=*/false, /*Has_alibi=*/false,
      /*Is_even_MN=*/false, /*Is_even_K=*/true, /*Is_softcap=*/false,
      /*Split=*/true, /*Append_KV=*/false, ExternalKvLoader,
      /*Raw_kv_smem_alias=*/false, /*Active_query_warp_gate=*/true,
      /*Warp_specialized_overlap=*/true,
      /*Warp_specialized_overlap_trace=*/false,
      /*Warp_specialized_q1_7p1c=*/true>(params);
#else
  FLASH_UNSUPPORTED_ARCH
#endif
}

DEFINE_FLASH_FORWARD_KERNEL(
    flash_fwd_splitkv_byte_v2_one_cta_7p1c_trace_profile_kernel,
    bool Is_causal) {
#if defined(ARCH_SUPPORTS_FLASH)
  using ExternalKvLoader =
      vllm::byte_v2::fa2::StaticW16CanonicalLoaderOneCtaOverlap;
  FLASH_NAMESPACE::compute_attn_splitkv<
      Kernel_traits, Is_causal, /*Is_local=*/false, /*Has_alibi=*/false,
      /*Is_even_MN=*/false, /*Is_even_K=*/true, /*Is_softcap=*/false,
      /*Split=*/true, /*Append_KV=*/false, ExternalKvLoader,
      /*Raw_kv_smem_alias=*/false, /*Active_query_warp_gate=*/true,
      /*Warp_specialized_overlap=*/true,
      /*Warp_specialized_overlap_trace=*/true,
      /*Warp_specialized_q1_7p1c=*/true>(params);
#else
  FLASH_UNSUPPORTED_ARCH
#endif
}

inline bool byte_v2_fa2_reuse_kv_smem_nonsplit_enabled() {
  static const bool enabled = [] {
    const char* value = std::getenv("BYTE_V2_FA2_REUSE_KV_SMEM_NONSPLIT");
    if (value == nullptr || std::strcmp(value, "1") == 0) {
      return true;
    }
    TORCH_CHECK(
        std::strcmp(value, "0") == 0,
        "BYTE_V2_FA2_REUSE_KV_SMEM_NONSPLIT must be unset, 0, or 1; got '",
        value, "'");
    return false;
  }();
  return enabled;
}

inline size_t byte_v2_fa2_profile_static_w16_smem_padding_bytes() {
  static const size_t padding_bytes = [] {
    const char* value =
        std::getenv("BYTE_V2_FA2_PROFILE_STATIC_W16_SMEM_PADDING_BYTES");
    if (value == nullptr || std::strcmp(value, "0") == 0) {
      return size_t{0};
    }
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(value, &end, 10);
    TORCH_CHECK(end != value && *end == '\0' && parsed <= 64 * 1024,
                "BYTE_V2_FA2_PROFILE_STATIC_W16_SMEM_PADDING_BYTES must be an "
                "integer in [0, 65536]; got '",
                value, "'");
    return static_cast<size_t>(parsed);
  }();
  return padding_bytes;
}

inline bool byte_v2_fa2_profile_static_w16_active_query_warp_enabled() {
  static const bool enabled = [] {
    const char* value =
        std::getenv("BYTE_V2_FA2_PROFILE_STATIC_W16_ACTIVE_QUERY_WARP");
    if (value == nullptr || std::strcmp(value, "0") == 0) {
      return false;
    }
    TORCH_CHECK(
        std::strcmp(value, "1") == 0,
        "BYTE_V2_FA2_PROFILE_STATIC_W16_ACTIVE_QUERY_WARP must be unset, 0, "
        "or 1; got '",
        value, "'");
    return true;
  }();
  return enabled;
}

inline bool byte_v2_fa2_profile_static_w16_one_cta_overlap_enabled() {
  static const bool enabled = [] {
    const char* value =
        std::getenv("BYTE_V2_FA2_PROFILE_STATIC_W16_ONE_CTA_OVERLAP");
    if (value == nullptr || std::strcmp(value, "0") == 0) {
      return false;
    }
    TORCH_CHECK(
        std::strcmp(value, "1") == 0,
        "BYTE_V2_FA2_PROFILE_STATIC_W16_ONE_CTA_OVERLAP must be unset, 0, "
        "or 1; got '",
        value, "'");
    return true;
  }();
  return enabled;
}

inline bool byte_v2_fa2_profile_static_w16_one_cta_trace_enabled() {
  static const bool enabled = [] {
    const char* value =
        std::getenv("BYTE_V2_FA2_PROFILE_STATIC_W16_ONE_CTA_TRACE");
    if (value == nullptr || std::strcmp(value, "0") == 0) {
      return false;
    }
    TORCH_CHECK(
        std::strcmp(value, "1") == 0,
        "BYTE_V2_FA2_PROFILE_STATIC_W16_ONE_CTA_TRACE must be unset, 0, "
        "or 1; got '",
        value, "'");
    return true;
  }();
  return enabled;
}

inline bool byte_v2_fa2_profile_static_w16_one_cta_7p1c_enabled() {
  static const bool enabled = [] {
    const char* value =
        std::getenv("BYTE_V2_FA2_PROFILE_STATIC_W16_ONE_CTA_7P1C");
    if (value == nullptr || std::strcmp(value, "0") == 0) {
      return false;
    }
    TORCH_CHECK(
        std::strcmp(value, "1") == 0,
        "BYTE_V2_FA2_PROFILE_STATIC_W16_ONE_CTA_7P1C must be unset, 0, or "
        "1; got '",
        value, "'");
    return true;
  }();
  return enabled;
}

template <typename Kernel_traits, bool Is_causal,
          vllm::byte_v2::fa2::ExternalKvFormat Format>
void run_flash_byte_v2_splitkv_fwd(Flash_fwd_params& params,
                                   cudaStream_t stream) {
  static_assert(!Kernel_traits::Is_Q_in_regs,
                "SplitKV implementation does not support Is_Q_in_regs");
  static_assert(!Kernel_traits::Share_Q_K_smem,
                "SplitKV implementation does not support Share_Q_K_smem");
  static_assert(Kernel_traits::kHeadDim == 128);
  static_assert(Kernel_traits::kBlockN == vllm::byte_v2::fa2::Loader::BlockN);
  static_assert(Kernel_traits::kNThreads ==
                vllm::byte_v2::fa2::Loader::Threads);

  const bool is_even_MN = params.cu_seqlens_q == nullptr &&
                          params.cu_seqlens_k == nullptr &&
                          params.seqlen_k % Kernel_traits::kBlockN == 0 &&
                          params.seqlen_q % Kernel_traits::kBlockM == 0;
  const bool is_even_K = params.d == Kernel_traits::kHeadDim;

  constexpr bool IsSplitZip =
      Format == vllm::byte_v2::fa2::ExternalKvFormat::SplitZip;
  constexpr bool IsStaticW16Canonical =
      Format == vllm::byte_v2::fa2::ExternalKvFormat::StaticW16Canonical;
  constexpr bool IsStaticW16 =
      Format == vllm::byte_v2::fa2::ExternalKvFormat::StaticW16 ||
      IsStaticW16Canonical;
  constexpr const char* format_name =
      IsSplitZip
          ? "SplitZip"
          : (IsStaticW16Canonical ? "Static-W16-canonical"
                                  : (IsStaticW16 ? "Static-W16" : "ByteV2"));
  TORCH_CHECK(params.blockmask != nullptr, format_name,
              " FA2 cache pointer is missing");
  TORCH_CHECK(params.block_table != nullptr, format_name,
              " FA2 requires a paged block table");
  TORCH_CHECK(!is_even_MN, format_name, " FA2 expects varlen/paged dispatch");
  TORCH_CHECK(
      params.page_block_size == vllm::byte_v2::fa2::Policy::AllocBlockTokens,
      format_name, " FA2 page block size must be ",
      vllm::byte_v2::fa2::Policy::AllocBlockTokens);
  TORCH_CHECK(params.k_batch_stride > 0, format_name,
              " FA2 cache must contain at least one page");
  TORCH_CHECK(is_even_K, format_name, " FA2 requires exact head dim 128");
  if constexpr (IsSplitZip) {
    TORCH_CHECK(params.knew_ptr == params.blockmask, format_name,
                " FA2 format marker mismatch");
    TORCH_CHECK(params.vnew_ptr == nullptr,
                "SplitZip reader prototype does not support raw sidecars");
  } else if constexpr (IsStaticW16Canonical) {
    TORCH_CHECK(
        params.vnew_ptr != nullptr && params.knew_ptr == params.blockmask,
        format_name, " FA2 format marker mismatch");
    TORCH_CHECK(params.k_ptr != nullptr && params.v_ptr != nullptr &&
                    params.v_batch_stride > 0,
                "Static-W16-canonical requires authoritative raw staging");
  } else if constexpr (IsStaticW16) {
    TORCH_CHECK(
        params.vnew_ptr != nullptr && params.knew_ptr == params.vnew_ptr,
        format_name, " FA2 format marker mismatch");
    TORCH_CHECK(params.k_ptr != nullptr && params.v_ptr != nullptr &&
                    params.v_batch_stride > 0,
                "Static-W16 requires authoritative raw staging");
  } else {
    TORCH_CHECK(params.knew_ptr == nullptr, format_name,
                " FA2 format marker mismatch");
  }
  if (params.vnew_ptr != nullptr) {
    TORCH_CHECK(params.k_ptr != nullptr && params.v_ptr != nullptr,
                "ByteV2 hybrid FA2 raw staging pointers are missing");
    TORCH_CHECK(params.v_batch_stride > 0,
                "ByteV2 hybrid FA2 raw staging must contain a slot");
  }
  TORCH_CHECK(params.alibi_slopes_ptr == nullptr, format_name,
              " FA2 does not support ALiBi");
  TORCH_CHECK(params.softcap <= 0.0f, format_name,
              " FA2 does not support softcap");
  TORCH_CHECK(Is_causal ||
                  (params.window_size_left < 0 && params.window_size_right < 0),
              format_name, " FA2 does not support local attention");

  const int num_m_block =
      (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
  dim3 grid(num_m_block, params.num_splits > 1 ? params.num_splits : params.b,
            params.num_splits > 1 ? params.b * params.h : params.h);

  BOOL_SWITCH(params.num_splits > 1, Split, [&] {
    auto launch = [&](auto reuse_nonsplit_tag) {
      constexpr bool ReuseKvSmemNonsplit = decltype(reuse_nonsplit_tag)::value;
      using LegacyByteV2Loader =
          std::conditional_t<ReuseKvSmemNonsplit,
                             vllm::byte_v2::fa2::LoaderReuseKvSmemNonsplit,
                             vllm::byte_v2::fa2::Loader>;
      using LegacySplitZipLoader = std::conditional_t<
          ReuseKvSmemNonsplit,
          vllm::byte_v2::fa2::SplitZipLoaderReuseKvSmemNonsplit,
          vllm::byte_v2::fa2::SplitZipLoader>;
      using LegacyStaticW16CanonicalLoader = std::conditional_t<
          ReuseKvSmemNonsplit,
          vllm::byte_v2::fa2::StaticW16CanonicalLoaderReuseKvSmemNonsplit,
          vllm::byte_v2::fa2::StaticW16CanonicalLoader>;
      using LegacyStaticW16Loader = std::conditional_t<
          IsStaticW16Canonical, LegacyStaticW16CanonicalLoader,
          std::conditional_t<
              ReuseKvSmemNonsplit,
              vllm::byte_v2::fa2::StaticW16LoaderReuseKvSmemNonsplit,
              vllm::byte_v2::fa2::StaticW16Loader>>;
      using LegacyExternalKvLoader = std::conditional_t<
          IsSplitZip, LegacySplitZipLoader,
          std::conditional_t<IsStaticW16, LegacyStaticW16Loader,
                             LegacyByteV2Loader>>;
      constexpr bool UseSharedPageDescriptor =
          vllm::byte_v2::fa2::kReuseKvSmem &&
          vllm::byte_v2::fa2::kStageMode != 0 && (Split || ReuseKvSmemNonsplit);
      using StaticW16SharedPageLoader = std::conditional_t<
          IsStaticW16Canonical,
          vllm::byte_v2::fa2::StaticW16CanonicalLoaderSharedPageDescriptor,
          vllm::byte_v2::fa2::StaticW16LoaderSharedPageDescriptor>;
      using SharedPageLoader = std::conditional_t<
          IsSplitZip, vllm::byte_v2::fa2::SplitZipLoaderSharedPageDescriptor,
          std::conditional_t<IsStaticW16, StaticW16SharedPageLoader,
                             vllm::byte_v2::fa2::LoaderSharedPageDescriptor>>;
      using ExternalKvLoader =
          std::conditional_t<UseSharedPageDescriptor, SharedPageLoader,
                             LegacyExternalKvLoader>;
      constexpr bool reuse_kv_smem =
          ExternalKvLoader::ReuseKvSmem &&
          (Split || ExternalKvLoader::ReuseKvSmemNonsplit);
      constexpr size_t attention_smem_size =
          reuse_kv_smem
              ? Kernel_traits::kSmemQSize + Kernel_traits::kSmemKVSize / 2
              : Kernel_traits::kSmemSize;
      constexpr size_t smem_size =
          attention_smem_size + ExternalKvLoader::SharedStorageBytes;
      const size_t smem_padding_bytes =
          IsStaticW16 ? byte_v2_fa2_profile_static_w16_smem_padding_bytes()
                      : size_t{0};
      const size_t launch_smem_size = smem_size + smem_padding_bytes;
      static_assert(!reuse_kv_smem ||
                    attention_smem_size ==
                        vllm::byte_v2::fa2::kFa2ReuseSmemBytes);
      static_assert(!UseSharedPageDescriptor ||
                    ExternalKvLoader::SharedStorageBytes > 0);
      static_assert(UseSharedPageDescriptor ||
                    ExternalKvLoader::SharedStorageBytes == 0);
      const bool active_query_warp_profile =
          IsStaticW16 &&
          byte_v2_fa2_profile_static_w16_active_query_warp_enabled();
      const bool one_cta_overlap_profile =
          IsStaticW16Canonical &&
          byte_v2_fa2_profile_static_w16_one_cta_overlap_enabled();
      const bool one_cta_trace_profile =
          IsStaticW16Canonical &&
          byte_v2_fa2_profile_static_w16_one_cta_trace_enabled();
      const bool one_cta_7p1c_profile =
          IsStaticW16Canonical &&
          byte_v2_fa2_profile_static_w16_one_cta_7p1c_enabled();
      TORCH_CHECK(!(active_query_warp_profile && one_cta_overlap_profile),
                  "Static-W16 active-query-warp and one-CTA overlap profiles "
                  "are mutually exclusive");
      TORCH_CHECK(!one_cta_trace_profile || one_cta_overlap_profile,
                  "Static-W16 one-CTA trace requires the one-CTA overlap "
                  "profile");
      TORCH_CHECK(!one_cta_7p1c_profile || one_cta_overlap_profile,
                  "Static-W16 one-CTA 7P1C requires the one-CTA overlap "
                  "profile");
      if (active_query_warp_profile) {
        TORCH_CHECK(Split,
                    "Static-W16 active-query-warp profile requires split-K");
        TORCH_CHECK(
            params.seqlen_q <= 16,
            "Static-W16 active-query-warp profile is restricted to Q1/GQA "
            "dispatches with at most 16 FA2 query rows; got ",
            params.seqlen_q);
      }
      if (one_cta_overlap_profile) {
        TORCH_CHECK(Split,
                    "Static-W16 one-CTA overlap profile requires split-K");
        TORCH_CHECK(
            params.seqlen_q <= 16,
            "Static-W16 one-CTA overlap profile is restricted to Q1/GQA "
            "dispatches with at most 16 FA2 query rows; got ",
            params.seqlen_q);
        TORCH_CHECK(smem_padding_bytes == 0,
                    "Static-W16 one-CTA overlap profile owns its shared-memory "
                    "footprint and cannot be combined with profile padding");
      }
      auto kernel =
          one_cta_trace_profile
              ? (one_cta_7p1c_profile
                     ? &flash_fwd_splitkv_byte_v2_one_cta_7p1c_trace_profile_kernel<
                           Kernel_traits, Is_causal>
                     : &flash_fwd_splitkv_byte_v2_one_cta_overlap_trace_profile_kernel<
                           Kernel_traits, Is_causal>)
          : one_cta_overlap_profile
              ? (one_cta_7p1c_profile
                     ? &flash_fwd_splitkv_byte_v2_one_cta_7p1c_profile_kernel<
                           Kernel_traits, Is_causal>
                     : &flash_fwd_splitkv_byte_v2_one_cta_overlap_profile_kernel<
                           Kernel_traits, Is_causal>)
          : active_query_warp_profile
              ? &flash_fwd_splitkv_byte_v2_active_query_warp_profile_kernel<
                    Kernel_traits, Is_causal, Split, ExternalKvLoader>
              : &flash_fwd_splitkv_byte_v2_kernel<Kernel_traits, Is_causal,
                                                  Split, ReuseKvSmemNonsplit,
                                                  Format>;
      constexpr size_t one_cta_overlap_smem_size =
          Kernel_traits::kSmemSize +
          vllm::byte_v2::fa2::StaticW16CanonicalLoaderOneCtaOverlap::
              SharedStorageBytes;
      const size_t selected_launch_smem_size = one_cta_overlap_profile
                                                   ? one_cta_overlap_smem_size
                                                   : launch_smem_size;
      if (selected_launch_smem_size >= 48 * 1024) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
            selected_launch_smem_size));
      }
      if (smem_padding_bytes > 0 || one_cta_overlap_profile) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(
            kernel, cudaFuncAttributePreferredSharedMemoryCarveout,
            cudaSharedmemCarveoutMaxShared));
      }
      const int launch_threads = one_cta_overlap_profile
                                     ? 2 * Kernel_traits::kNThreads
                                     : Kernel_traits::kNThreads;
      kernel<<<grid, launch_threads, selected_launch_smem_size, stream>>>(
          params);
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    };
    if constexpr (Split) {
      launch(std::false_type{});
    } else if (byte_v2_fa2_reuse_kv_smem_nonsplit_enabled()) {
      launch(std::true_type{});
    } else {
      launch(std::false_type{});
    }
  });

  if (params.num_splits > 1) {
    // This is the original FA2 split-K combine launch, intentionally kept
    // byte-for-byte equivalent to run_flash_splitkv_fwd.
    constexpr static int kBlockM =
        Kernel_traits::kHeadDim % 128 == 0
            ? 4
            : (Kernel_traits::kHeadDim % 64 == 0 ? 8 : 16);
    dim3 grid_combine((params.b * params.h * params.seqlen_q + kBlockM - 1) /
                      kBlockM);
    EVENK_SWITCH(is_even_K, IsEvenKConst, [&] {
      if (params.num_splits <= 2) {
        flash_fwd_splitkv_combine_kernel<Kernel_traits, kBlockM, 1,
                                         IsEvenKConst>
            <<<grid_combine, Kernel_traits::kNThreads, 0, stream>>>(params);
      } else if (params.num_splits <= 4) {
        flash_fwd_splitkv_combine_kernel<Kernel_traits, kBlockM, 2,
                                         IsEvenKConst>
            <<<grid_combine, Kernel_traits::kNThreads, 0, stream>>>(params);
      } else if (params.num_splits <= 8) {
        flash_fwd_splitkv_combine_kernel<Kernel_traits, kBlockM, 3,
                                         IsEvenKConst>
            <<<grid_combine, Kernel_traits::kNThreads, 0, stream>>>(params);
      } else if (params.num_splits <= 16) {
        flash_fwd_splitkv_combine_kernel<Kernel_traits, kBlockM, 4,
                                         IsEvenKConst>
            <<<grid_combine, Kernel_traits::kNThreads, 0, stream>>>(params);
      } else if (params.num_splits <= 32) {
        flash_fwd_splitkv_combine_kernel<Kernel_traits, kBlockM, 5,
                                         IsEvenKConst>
            <<<grid_combine, Kernel_traits::kNThreads, 0, stream>>>(params);
      } else if (params.num_splits <= 64) {
        flash_fwd_splitkv_combine_kernel<Kernel_traits, kBlockM, 6,
                                         IsEvenKConst>
            <<<grid_combine, Kernel_traits::kNThreads, 0, stream>>>(params);
      } else if (params.num_splits <= 128) {
        flash_fwd_splitkv_combine_kernel<Kernel_traits, kBlockM, 7,
                                         IsEvenKConst>
            <<<grid_combine, Kernel_traits::kNThreads, 0, stream>>>(params);
      }
      C10_CUDA_KERNEL_LAUNCH_CHECK();
    });
  }
}

template <typename T, int Headdim, bool Is_causal>
void run_mha_byte_v2_fwd_splitkv_dispatch(Flash_fwd_params& params,
                                          cudaStream_t stream) {
  static_assert(Headdim == 128);
  static_assert(std::is_same_v<T, cutlass::bfloat16_t>);
  constexpr static int kBlockM = 64;
  constexpr static int kBlockN = 128;
  using KernelTraits =
      Flash_fwd_kernel_traits<Headdim, kBlockM, kBlockN, 4, false, false, T>;
  if (params.vnew_ptr != nullptr && params.knew_ptr == params.blockmask) {
    run_flash_byte_v2_splitkv_fwd<
        KernelTraits, Is_causal,
        vllm::byte_v2::fa2::ExternalKvFormat::StaticW16Canonical>(params,
                                                                  stream);
  } else if (params.knew_ptr == params.blockmask) {
    run_flash_byte_v2_splitkv_fwd<
        KernelTraits, Is_causal,
        vllm::byte_v2::fa2::ExternalKvFormat::SplitZip>(params, stream);
  } else if (params.vnew_ptr != nullptr && params.knew_ptr == params.vnew_ptr) {
    run_flash_byte_v2_splitkv_fwd<
        KernelTraits, Is_causal,
        vllm::byte_v2::fa2::ExternalKvFormat::StaticW16>(params, stream);
  } else {
    run_flash_byte_v2_splitkv_fwd<KernelTraits, Is_causal,
                                  vllm::byte_v2::fa2::ExternalKvFormat::ByteV2>(
        params, stream);
  }
}

}  // namespace FLASH_NAMESPACE
