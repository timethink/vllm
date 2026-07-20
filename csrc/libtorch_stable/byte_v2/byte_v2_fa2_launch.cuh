// Copyright (c) 2026, ByteV2 contributors.
//
// ByteV2-specific FA2 launch path.  The attention mainloop is the original
// FA2 split-KV implementation; only its global-to-shared K/V load policy is
// selected as ByteV2 decode.

#pragma once

#include <type_traits>

#include "flash_fwd_launch_template.h"
#include "byte_v2_fa2_loader.cuh"

namespace FLASH_NAMESPACE {

DEFINE_FLASH_FORWARD_KERNEL(flash_fwd_splitkv_byte_v2_kernel, bool Is_causal,
                            bool Split) {
#if defined(ARCH_SUPPORTS_FLASH)
  FLASH_NAMESPACE::compute_attn_splitkv<
      Kernel_traits, Is_causal, /*Is_local=*/false, /*Has_alibi=*/false,
      /*Is_even_MN=*/false, /*Is_even_K=*/true, /*Is_softcap=*/false, Split,
      /*Append_KV=*/false, vllm::byte_v2::fa2::Loader>(params);
#else
  FLASH_UNSUPPORTED_ARCH
#endif
}

template <typename Kernel_traits, bool Is_causal>
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

  TORCH_CHECK(params.blockmask != nullptr,
              "ByteV2 FA2 cache pointer is missing");
  TORCH_CHECK(params.block_table != nullptr,
              "ByteV2 FA2 requires a paged block table");
  TORCH_CHECK(!is_even_MN, "ByteV2 FA2 expects varlen/paged dispatch");
  TORCH_CHECK(
      params.page_block_size == vllm::byte_v2::fa2::Policy::AllocBlockTokens,
      "ByteV2 FA2 page block size must be ",
      vllm::byte_v2::fa2::Policy::AllocBlockTokens);
  TORCH_CHECK(params.k_batch_stride > 0,
              "ByteV2 FA2 cache must contain at least one page");
  TORCH_CHECK(is_even_K, "ByteV2 FA2 requires exact head dim 128");
  TORCH_CHECK(params.knew_ptr == nullptr,
              "ByteV2 FA2 does not append KV inside attention");
  if (params.vnew_ptr != nullptr) {
    TORCH_CHECK(params.k_ptr != nullptr && params.v_ptr != nullptr,
                "ByteV2 hybrid FA2 raw staging pointers are missing");
    TORCH_CHECK(params.v_batch_stride > 0,
                "ByteV2 hybrid FA2 raw staging must contain a slot");
  }
  TORCH_CHECK(params.alibi_slopes_ptr == nullptr,
              "ByteV2 FA2 does not support ALiBi");
  TORCH_CHECK(params.softcap <= 0.0f, "ByteV2 FA2 does not support softcap");
  TORCH_CHECK(Is_causal ||
                  (params.window_size_left < 0 && params.window_size_right < 0),
              "ByteV2 FA2 does not support local attention");

  const int num_m_block =
      (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
  dim3 grid(num_m_block, params.num_splits > 1 ? params.num_splits : params.b,
            params.num_splits > 1 ? params.b * params.h : params.h);

  BOOL_SWITCH(params.num_splits > 1, Split, [&] {
    constexpr size_t smem_size =
        Split && vllm::byte_v2::fa2::Loader::ReuseKvSmem
            ? Kernel_traits::kSmemQSize + Kernel_traits::kSmemKVSize / 2
            : Kernel_traits::kSmemSize;
    static_assert(!Split || !vllm::byte_v2::fa2::Loader::ReuseKvSmem ||
                  smem_size == 48 * 1024);
    auto kernel =
        &flash_fwd_splitkv_byte_v2_kernel<Kernel_traits, Is_causal, Split>;
    if (smem_size >= 48 * 1024) {
      C10_CUDA_CHECK(cudaFuncSetAttribute(
          kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    }
    kernel<<<grid, Kernel_traits::kNThreads, smem_size, stream>>>(params);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
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
  run_flash_byte_v2_splitkv_fwd<
      Flash_fwd_kernel_traits<Headdim, kBlockM, kBlockN, 4, false, false, T>,
      Is_causal>(params, stream);
}

}  // namespace FLASH_NAMESPACE
