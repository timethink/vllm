#include "torch_utils.h"
#include "dispatch_utils.h"

#include "../cuda_utils.h"
#include "../cuda_compat.h"

#include "quantization/vectorization_utils.cuh"
#include "concat_mla_q.cuh"

#ifdef USE_ROCM
  #include "../quantization/w8a8/fp8/amd/quant_utils.cuh"
#else
  #include "../quantization/w8a8/fp8/nvidia/quant_utils.cuh"
#endif

#include <algorithm>
#include <cassert>
#include <cfloat>
#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <string>
#include <vector>

#ifdef USE_ROCM
  #include <hip/hip_bf16.h>
typedef __hip_bfloat16 __nv_bfloat16;
#else
  #include <cuda.h>
  #include <cuda_bf16.h>
  #include <mma.h>
#endif

#define BYTE_V2_CUDA_CHECK(cmd)                                             \
  do {                                                                      \
    cudaError_t err = cmd;                                                  \
    STD_TORCH_CHECK(err == cudaSuccess, "CUDA error: ", cudaGetErrorString(err)); \
  } while (0)

#ifndef USE_ROCM
bool byte_v2_device_supports_bf16_wmma(const int device_index) {
  constexpr int kMaxCachedCudaDevices = 64;
  if (device_index < 0 || device_index >= kMaxCachedCudaDevices) {
    cudaDeviceProp device_prop;
    BYTE_V2_CUDA_CHECK(cudaGetDeviceProperties(&device_prop, device_index));
    return device_prop.major >= 8;
  }

  static std::once_flag init_flags[kMaxCachedCudaDevices];
  static bool supports_bf16_wmma[kMaxCachedCudaDevices] = {};
  std::call_once(init_flags[device_index], [device_index]() {
    cudaDeviceProp device_prop;
    BYTE_V2_CUDA_CHECK(cudaGetDeviceProperties(&device_prop, device_index));
    supports_bf16_wmma[device_index] = device_prop.major >= 8;
  });
  return supports_bf16_wmma[device_index];
}
#endif

#if defined(__gfx942__)
constexpr float kFp8ScaleDivisor = 224.f;
#else
constexpr float kFp8ScaleDivisor = 448.f;
#endif

void swap_blocks(torch::stable::Tensor& src, torch::stable::Tensor& dst,
                 int64_t block_size_in_bytes,
                 const torch::stable::Tensor& block_mapping) {
  torch::stable::Device src_device = src.device();
  torch::stable::Device dst_device = dst.device();
  cudaMemcpyKind memcpy_type;
  if (src_device.is_cuda() && dst_device.is_cuda()) {
    STD_TORCH_CHECK(src_device.index() == dst_device.index(),
                    "src and dst must be on the same GPU");
    memcpy_type = cudaMemcpyDeviceToDevice;
  } else if (src_device.is_cuda() && dst_device.is_cpu()) {
    memcpy_type = cudaMemcpyDeviceToHost;
  } else if (src_device.is_cpu() && dst_device.is_cuda()) {
    memcpy_type = cudaMemcpyHostToDevice;
  } else {
    STD_TORCH_CHECK(false, "Invalid device combination");
  }

  // NOTE(youkaichao): keep in mind that `block_mapping` should be
  // a cpu tensor, otherwise every `item` call will require a gpu-cpu
  // synchronization.
  STD_TORCH_CHECK(block_mapping.device().is_cpu(),
                  "block_mapping must be on CPU");

  char* src_ptr = static_cast<char*>(src.data_ptr());
  char* dst_ptr = static_cast<char*>(dst.data_ptr());

  auto guard_device = src_device.is_cuda() ? src_device : dst_device;
  const torch::stable::accelerator::DeviceGuard device_guard(
      guard_device.index());
  const cudaStream_t stream = get_current_cuda_stream();
  // NOTE(woosuk): This can be slow if the number of blocks is large.
  const int64_t num_blocks = block_mapping.size(0);
  const int64_t* bm_ptr = block_mapping.const_data_ptr<int64_t>();
  const int64_t bm_stride0 = block_mapping.stride(0);
  const int64_t bm_stride1 = block_mapping.stride(1);
  for (size_t i = 0; i < num_blocks; i++) {
    int64_t src_block_number = bm_ptr[i * bm_stride0];
    int64_t dst_block_number = bm_ptr[i * bm_stride0 + bm_stride1];
    int64_t src_offset = src_block_number * block_size_in_bytes;
    int64_t dst_offset = dst_block_number * block_size_in_bytes;
    cudaMemcpyAsync(dst_ptr + dst_offset, src_ptr + src_offset,
                    block_size_in_bytes, memcpy_type, stream);
  }
}

void swap_blocks_batch(const torch::stable::Tensor& src_ptrs,
                       const torch::stable::Tensor& dst_ptrs,
                       const torch::stable::Tensor& sizes,
                       bool is_src_access_order_any) {
  STD_TORCH_CHECK(src_ptrs.device().is_cpu(), "src_ptrs must be on CPU");
  STD_TORCH_CHECK(dst_ptrs.device().is_cpu(), "dst_ptrs must be on CPU");
  STD_TORCH_CHECK(sizes.device().is_cpu(), "sizes must be on CPU");
  STD_TORCH_CHECK(src_ptrs.scalar_type() == torch::headeronly::ScalarType::Long,
                  "src_ptrs must be int64");
  STD_TORCH_CHECK(dst_ptrs.scalar_type() == torch::headeronly::ScalarType::Long,
                  "dst_ptrs must be int64");
  STD_TORCH_CHECK(sizes.scalar_type() == torch::headeronly::ScalarType::Long,
                  "sizes must be int64");

  const int64_t n = src_ptrs.size(0);
  STD_TORCH_CHECK(dst_ptrs.size(0) == n, "dst_ptrs length must match src_ptrs");
  STD_TORCH_CHECK(sizes.size(0) == n, "sizes length must match src_ptrs");

  if (n == 0) return;

  int64_t* src_data = src_ptrs.mutable_data_ptr<int64_t>();
  int64_t* dst_data = dst_ptrs.mutable_data_ptr<int64_t>();
  int64_t* size_data = sizes.mutable_data_ptr<int64_t>();

  const cudaStream_t stream = get_current_cuda_stream();

  // Use cuMemcpyBatchAsync / hipMemcpyBatchAsync to submit all copies in a
  // single driver call, amortizing per-copy submission overhead. int64_t
  // and CUdeviceptr/void*/size_t are all 8 bytes on 64-bit platforms, so we
  // reinterpret_cast the tensor data directly to avoid copies.
  static_assert(sizeof(size_t) == sizeof(int64_t));
#if !defined(USE_ROCM) && defined(CUDA_VERSION) && CUDA_VERSION >= 12080
  static_assert(sizeof(CUdeviceptr) == sizeof(int64_t));
  // Resolve cuMemcpyBatchAsync at runtime via cuGetProcAddress so that
  // binaries compiled with CUDA 12.8+ still work on older drivers, and
  // we avoid the CUDA 13.0 header remapping (#define to _v2 signature).
  // The function pointer is cached after the first call.
  using BatchFn =
      CUresult (*)(CUdeviceptr*, CUdeviceptr*, size_t*, size_t,
                   CUmemcpyAttributes*, size_t*, size_t, size_t*, CUstream);
  static BatchFn batch_fn = []() -> BatchFn {
    CUdriverProcAddressQueryResult sym_status;
    void* fn_ptr = nullptr;
    CUresult res = cuGetProcAddress("cuMemcpyBatchAsync", &fn_ptr, 12080,
                                    CU_GET_PROC_ADDRESS_DEFAULT, &sym_status);
    if (res != CUDA_SUCCESS || fn_ptr == nullptr) {
      return nullptr;
    }
    return reinterpret_cast<BatchFn>(fn_ptr);
  }();

  if (batch_fn != nullptr) {
    CUmemcpyAttributes attr = {};
    // ANY lets the DMA engine prefetch source bytes out of stream order,
    // which is only safe when no GPU stream is concurrently writing the
    // source.
    attr.srcAccessOrder = is_src_access_order_any
                              ? CU_MEMCPY_SRC_ACCESS_ORDER_ANY
                              : CU_MEMCPY_SRC_ACCESS_ORDER_STREAM;
    size_t attrs_idx = 0;
    size_t fail_idx = 0;
    CUresult result = batch_fn(reinterpret_cast<CUdeviceptr*>(dst_data),
                               reinterpret_cast<CUdeviceptr*>(src_data),
                               reinterpret_cast<size_t*>(size_data),
                               static_cast<size_t>(n), &attr, &attrs_idx, 1,
                               &fail_idx, static_cast<CUstream>(stream));
    STD_TORCH_CHECK(result == CUDA_SUCCESS,
                    "cuMemcpyBatchAsync failed at index ", fail_idx,
                    " with error ", result);
    return;
  }
#elif defined(USE_ROCM) && defined(HIP_VERSION) && HIP_VERSION >= 70100000
  // ROCm 7.1+ exposes hipMemcpyBatchAsync. The 7.2.1 implementation early-
  // returns hipErrorNotSupported whenever numAttrs > 0 (see ROCm/clr @
  // rocm-7.2.1 hipamd/src/hip_memory.cpp:2819-2822), so call with
  // numAttrs=0.
  {
    hipMemcpyAttributes attr = {};
    size_t attrs_idx = 0;
    size_t fail_idx = 0;
    hipError_t result = hipMemcpyBatchAsync(
        reinterpret_cast<void**>(dst_data), reinterpret_cast<void**>(src_data),
        reinterpret_cast<size_t*>(size_data), static_cast<size_t>(n), &attr,
        &attrs_idx, 0, &fail_idx, static_cast<hipStream_t>(stream));
    STD_TORCH_CHECK(result == hipSuccess,
                    "hipMemcpyBatchAsync failed at index ", fail_idx,
                    " with error ", result);
    return;
  }
#endif
  {
    // Fallback for CUDA < 12.8, older CUDA drivers, and ROCm < 7.1:
    // individual async copies. cudaMemcpyDefault lets the driver infer
    // direction from pointer types.
    for (int64_t i = 0; i < n; i++) {
      cudaMemcpyAsync(reinterpret_cast<void*>(dst_data[i]),
                      reinterpret_cast<void*>(src_data[i]),
                      static_cast<size_t>(size_data[i]), cudaMemcpyDefault,
                      stream);
    }
  }
}

namespace vllm {

// Grid: (num_layers, num_pairs)
template <typename scalar_t>
__global__ void copy_blocks_kernel(int64_t* key_cache_ptrs,
                                   int64_t* value_cache_ptrs,
                                   const int64_t* __restrict__ block_mapping,
                                   const int numel_per_block) {
  const int layer_idx = blockIdx.x;
  const int pair_idx = blockIdx.y;

  scalar_t* key_cache = reinterpret_cast<scalar_t*>(key_cache_ptrs[layer_idx]);
  scalar_t* value_cache =
      reinterpret_cast<scalar_t*>(value_cache_ptrs[layer_idx]);
  int64_t src_block_number = block_mapping[2 * pair_idx];
  int64_t dst_block_number = block_mapping[2 * pair_idx + 1];

  const int64_t src_block_offset = src_block_number * numel_per_block;
  const int64_t dst_block_offset = dst_block_number * numel_per_block;
  for (int i = threadIdx.x; i < numel_per_block; i += blockDim.x) {
    int64_t src_offset = src_block_offset + i;
    int64_t dst_offset = dst_block_offset + i;
    key_cache[dst_offset] = key_cache[src_offset];
  }
  for (int i = threadIdx.x; i < numel_per_block; i += blockDim.x) {
    int64_t src_offset = src_block_offset + i;
    int64_t dst_offset = dst_block_offset + i;
    value_cache[dst_offset] = value_cache[src_offset];
  }
}

// Kernel for MLA, which works on a single joint kv_cache
// Grid: (num_layers, num_pairs)
template <typename scalar_t>
__global__ void copy_blocks_mla_kernel(
    int64_t* cache_ptrs, const int64_t* __restrict__ block_mapping,
    const int mem_footprint_per_block) {
  const int layer_idx = blockIdx.x;
  const int pair_idx = blockIdx.y;
  scalar_t* cache = reinterpret_cast<scalar_t*>(cache_ptrs[layer_idx]);
  int64_t src_block = block_mapping[2 * pair_idx];
  int64_t dst_block = block_mapping[2 * pair_idx + 1];
  int64_t src_offset = src_block * mem_footprint_per_block;
  int64_t dst_offset = dst_block * mem_footprint_per_block;
  for (int i = threadIdx.x; i < mem_footprint_per_block; i += blockDim.x) {
    cache[dst_offset + i] = cache[src_offset + i];
  }
}

}  // namespace vllm

namespace vllm {

// Used to copy/convert one element
template <typename OutT, typename InT, Fp8KVCacheDataType kv_dt>
struct CopyWithScaleOp {
  float scale;

  __device__ __forceinline__ void operator()(OutT& dst, const InT src) const {
    if constexpr (kv_dt == Fp8KVCacheDataType::kAuto) {
      dst = static_cast<OutT>(src);
    } else {
      dst = fp8::scaled_convert<OutT, InT, kv_dt>(src, scale);
    }
  }
};

template <typename scalar_t, typename cache_t, Fp8KVCacheDataType kv_dt>
__global__ void reshape_and_cache_kernel(
    const scalar_t* __restrict__ key,    // [num_tokens, num_heads, head_size]
    const scalar_t* __restrict__ value,  // [num_tokens, num_heads, head_size]
    cache_t* __restrict__ key_cache,     // [num_blocks, num_heads, head_size/x,
                                         // block_size, x]
    cache_t* __restrict__ value_cache,   // [num_blocks, num_heads, head_size,
                                         // block_size]
    const int64_t* __restrict__ slot_mapping,  // [num_tokens]
    const int key_stride, const int value_stride, const int num_heads,
    const int head_size, const int block_size, const int x,
    const float* k_scale, const float* v_scale) {
  const int64_t token_idx = blockIdx.x;
  const int64_t slot_idx = slot_mapping[token_idx];
  if (slot_idx < 0) {
    return;
  }

  const int64_t block_idx = slot_idx / block_size;
  const int64_t block_offset = slot_idx % block_size;
  const int h_block_count = head_size / x;  // head_size//x

  const int h_block_idx = threadIdx.x;
  if (h_block_idx >= num_heads * h_block_count) {
    return;
  }

  const int head_idx = h_block_idx / h_block_count;
  const int h_block = h_block_idx % h_block_count;

  const scalar_t* __restrict__ key_src =
      key + token_idx * key_stride + head_idx * head_size + h_block * x;
  const int64_t src_value_start =
      token_idx * value_stride + head_idx * head_size + h_block * x;

  cache_t* __restrict__ key_dst =
      key_cache + block_idx * num_heads * h_block_count * block_size * x +
      head_idx * h_block_count * block_size * x + h_block * block_size * x +
      block_offset * x;
  const int64_t tgt_value_start =
      block_idx * num_heads * h_block_count * x * block_size +
      head_idx * h_block_count * x * block_size + h_block * x * block_size +
      block_offset;

  constexpr int VEC_SIZE = (sizeof(scalar_t) == 2) ? 8 : 4;
  float k_scale_val = (kv_dt == Fp8KVCacheDataType::kAuto) ? 0.f : *k_scale;
  CopyWithScaleOp<cache_t, scalar_t, kv_dt> k_op{k_scale_val};
  float v_scale_val = (kv_dt == Fp8KVCacheDataType::kAuto) ? 0.f : *v_scale;
  CopyWithScaleOp<cache_t, scalar_t, kv_dt> v_op{v_scale_val};

  vectorize_with_alignment<VEC_SIZE>(key_src, key_dst, x, 0, 1, k_op);

  const scalar_t* __restrict__ value_src = value + src_value_start;
  cache_t* __restrict__ value_dst = value_cache + tgt_value_start;
#pragma unroll
  for (int i = 0; i < x; i++) {
    v_op(value_dst[i * block_size], value_src[i]);
  }
}

template <typename scalar_t, typename cache_t, Fp8KVCacheDataType kv_dt>
__global__ void reshape_and_cache_flash_kernel(
    const scalar_t* __restrict__ key,    // [num_tokens, num_heads, head_size]
    const scalar_t* __restrict__ value,  // [num_tokens, num_heads, head_size]
    cache_t* __restrict__ key_cache,     // NHD or HND, shape see comments below
    cache_t* __restrict__ value_cache,   // same above
    const int64_t* __restrict__ slot_mapping,  // [num_tokens]
    const int64_t block_stride, const int64_t page_stride,
    const int64_t head_stride, const int64_t key_stride,
    const int64_t value_stride, const int num_heads, const int head_size,
    const int block_size, const float* k_scale, const float* v_scale,
    const int kv_scale_stride) {
  const int64_t token_idx = blockIdx.x;
  const int64_t slot_idx = slot_mapping[token_idx];
  // NOTE: slot_idx can be -1 if the token is padded
  if (slot_idx < 0) {
    return;
  }
  const int64_t block_idx = slot_idx / block_size;
  const int64_t block_offset = slot_idx % block_size;
  const int n_elems = num_heads * head_size;

  // pointers to the beginning of the source row for this token.
  const scalar_t* __restrict__ key_src = key + token_idx * key_stride;
  const scalar_t* __restrict__ value_src = value + token_idx * value_stride;

  // find the start position inside the kv-cache for this token.
  cache_t* __restrict__ key_dst =
      key_cache + block_idx * block_stride + block_offset * page_stride;
  cache_t* __restrict__ value_dst =
      value_cache + block_idx * block_stride + block_offset * page_stride;

  // this is true for the NHD layout where `head_stride == head_size`
  const bool is_contiguous_heads = (head_stride == head_size);

  constexpr int VEC_SIZE = (sizeof(scalar_t) == 2) ? 8 : 4;

  if (is_contiguous_heads && kv_scale_stride == 0) {
    // NHD layout and k/v_scales are [1] (i.e. single scale for all heads)
    // kv cache: [num_blocks, block_size, num_heads, head_size]
    float k_scale_val = (kv_dt == Fp8KVCacheDataType::kAuto) ? 0.f : *k_scale;
    float v_scale_val = (kv_dt == Fp8KVCacheDataType::kAuto) ? 0.f : *v_scale;

    CopyWithScaleOp<cache_t, scalar_t, kv_dt> k_op{k_scale_val};
    CopyWithScaleOp<cache_t, scalar_t, kv_dt> v_op{v_scale_val};

    vectorize_with_alignment<VEC_SIZE>(key_src, key_dst, n_elems, threadIdx.x,
                                       blockDim.x, k_op);
    vectorize_with_alignment<VEC_SIZE>(value_src, value_dst, n_elems,
                                       threadIdx.x, blockDim.x, v_op);
  } else {
    // HND layout OR k/v_scales are [num_heads] (i.e. per-attn-head)
    // HND layout: heads are strided, but each head_size segment is contiguous
    // kv cache: [num_blocks, num_heads, block_size, head_size]
    const int lane = threadIdx.x & 31;     // 0..31 within warp
    const int warp_id = threadIdx.x >> 5;  // warp index within block
    const int warps_per_block = blockDim.x >> 5;

    for (int head = warp_id; head < num_heads; head += warps_per_block) {
      const scalar_t* __restrict__ k_src_h = key_src + head * head_size;
      const scalar_t* __restrict__ v_src_h = value_src + head * head_size;

      cache_t* __restrict__ k_dst_h =
          key_dst + static_cast<int64_t>(head) * head_stride;
      cache_t* __restrict__ v_dst_h =
          value_dst + static_cast<int64_t>(head) * head_stride;

      float k_scale_val = (kv_dt == Fp8KVCacheDataType::kAuto)
                              ? 0.f
                              : k_scale[head * kv_scale_stride];
      float v_scale_val = (kv_dt == Fp8KVCacheDataType::kAuto)
                              ? 0.f
                              : v_scale[head * kv_scale_stride];

      CopyWithScaleOp<cache_t, scalar_t, kv_dt> k_op{k_scale_val};
      CopyWithScaleOp<cache_t, scalar_t, kv_dt> v_op{v_scale_val};

      // within each head, let the 32 threads of the warp perform the vector
      // copy
      vectorize_with_alignment<VEC_SIZE>(k_src_h, k_dst_h, head_size, lane, 32,
                                         k_op);

      vectorize_with_alignment<VEC_SIZE>(v_src_h, v_dst_h, head_size, lane, 32,
                                         v_op);
    }
  }
}

template <typename scalar_t, typename cache_t, Fp8KVCacheDataType kv_dt>
__global__ void concat_and_cache_mla_kernel(
    const scalar_t* __restrict__ kv_c,  // [num_tokens, kv_lora_rank]
    const scalar_t* __restrict__ k_pe,  // [num_tokens, pe_dim]
    cache_t* __restrict__ kv_cache,  // [num_blocks, block_size, (kv_lora_rank
                                     // + pe_dim)]
    const int64_t* __restrict__ slot_mapping,  // [num_tokens]
    const int block_stride,                    //
    const int entry_stride,                    //
    const int kv_c_stride,                     //
    const int k_pe_stride,                     //
    const int kv_lora_rank,                    //
    const int pe_dim,                          //
    const int block_size,                      //
    const float* scale                         //
) {
  const int64_t token_idx = blockIdx.x;
  const int64_t slot_idx = slot_mapping[token_idx];
  // NOTE: slot_idx can be -1 if the token is padded
  if (slot_idx < 0) {
    return;
  }
  const int64_t block_idx = slot_idx / block_size;
  const int64_t block_offset = slot_idx % block_size;

  auto copy = [&](const scalar_t* __restrict__ src, cache_t* __restrict__ dst,
                  int src_stride, int dst_stride, int size, int offset) {
    for (int i = threadIdx.x; i < size; i += blockDim.x) {
      const int64_t src_idx = token_idx * src_stride + i;
      const int64_t dst_idx =
          block_idx * block_stride + block_offset * entry_stride + i + offset;
      if constexpr (kv_dt == Fp8KVCacheDataType::kAuto) {
        dst[dst_idx] = src[src_idx];
      } else {
        dst[dst_idx] =
            fp8::scaled_convert<cache_t, scalar_t, kv_dt>(src[src_idx], *scale);
      }
    }
  };

  copy(kv_c, kv_cache, kv_c_stride, block_stride, kv_lora_rank, 0);
  copy(k_pe, kv_cache, k_pe_stride, block_stride, pe_dim, kv_lora_rank);
}

template <typename scalar_t, typename cache_t, Fp8KVCacheDataType kv_dt>
__global__ void concat_and_cache_ds_mla_kernel(
    const scalar_t* __restrict__ kv_c,  // [num_tokens, kv_lora_rank]
    const scalar_t* __restrict__ k_pe,  // [num_tokens, pe_dim]
    cache_t* __restrict__ kv_cache,  // [num_blocks, block_size, (kv_lora_rank
                                     // + pe_dim)]
    const int64_t* __restrict__ slot_mapping,  // [num_tokens]
    const int block_stride,                    //
    const int entry_stride,                    //
    const int kv_c_stride,                     //
    const int k_pe_stride,                     //
    const int kv_lora_rank,                    //
    const int pe_dim,                          //
    const int block_size,                      //
    const float* scale                         //
) {
  const int64_t token_idx = blockIdx.x;
  const int64_t slot_idx = slot_mapping[token_idx];
  // NOTE: slot_idx can be -1 if the token is padded
  if (slot_idx < 0) {
    return;
  }
  const int64_t block_idx = slot_idx / block_size;
  const int64_t block_offset = slot_idx % block_size;
  const int64_t dst_idx_start =
      block_idx * block_stride + block_offset * entry_stride;

  // For the NoPE part, each tile of 128 elements is handled by half of one warp
  // (16 threads). There are 4 total tiles, so 2 warps (64 threads).
  // Lanes 0 and 16 of each warp write the scale values for that warp's tiles.
  // The RoPE part (last 64 elements) is handled by another 1 warp (32 threads).
  // So in total, we use 3 warps (96 threads) per block.

  // Cast kv_cache to 16_bit for RoPE values
  scalar_t* kv_cache_16bit =
      reinterpret_cast<scalar_t*>(&kv_cache[dst_idx_start]);

  // The last warp handles the RoPE part
  if (threadIdx.x >= 64) {
    // Each thread handles two elements of RoPE
    const int8_t pe_idx_start = (threadIdx.x - 64) * 2;
    const int64_t src_idx = token_idx * k_pe_stride + pe_idx_start;
    // Vectorized load of two 16-bit values, performed as one 32-bit load
    const int32_t vals = *reinterpret_cast<const int32_t*>(&k_pe[src_idx]);
    // RoPE values start after the packed 8-bit NoPE values and the
    // 32-bit scales
    const int64_t dst_idx = kv_lora_rank / 2 + 8 + pe_idx_start;
    // Vectorized store of two 16-bit values, performed as one 32-bit store
    *reinterpret_cast<int32_t*>(&kv_cache_16bit[dst_idx]) = vals;
    return;
  }

  // The first two warps handle the NoPE part
  const int8_t warp_idx = threadIdx.x >> 5;
  const int8_t lane_idx = threadIdx.x & 31;
  const int8_t tile_idx = warp_idx * 2 + (lane_idx >> 4);

  // Each thread handles 8 elements of NoPE
  // Load the NoPE elements for this thread into registers
  const int64_t src_idx_start = token_idx * kv_c_stride + (threadIdx.x * 8);
  // Vectorized load of eight 16-bit values, performed as an int4 load
  const int4 vals_i4 = *reinterpret_cast<const int4*>(&kv_c[src_idx_start]);
  const scalar_t* vals = reinterpret_cast<const scalar_t*>(&vals_i4);

  // Max absolute value of this thread's elements
  float max_abs = fmaxf(fmaxf(fmaxf(fabsf(vals[0]), fabsf(vals[1])),
                              fmaxf(fabsf(vals[2]), fabsf(vals[3]))),
                        fmaxf(fmaxf(fabsf(vals[4]), fabsf(vals[5])),
                              fmaxf(fabsf(vals[6]), fabsf(vals[7]))));

  // Warp-level reduction to find the max absolute value in each half-warp
#pragma unroll
  for (int offset = 8; offset > 0; offset /= 2) {
    max_abs = fmaxf(max_abs, VLLM_SHFL_XOR_SYNC_WIDTH(max_abs, offset, 16));
  }

  // Compute the scale for the tile
  float tile_scale = fmaxf(max_abs / kFp8ScaleDivisor, FLT_MIN);

  // The first lane of each half-warp writes the scale to kv_cache
  if ((lane_idx == 0) || (lane_idx == 16)) {
    float* kv_cache_32bit = reinterpret_cast<float*>(&kv_cache[dst_idx_start]);
    const uint64_t dst_idx = kv_lora_rank / 4 + tile_idx;
    kv_cache_32bit[dst_idx] = tile_scale;
  }

  // Now all threads in the block scale and write their elements
  // NoPE data is packed in the first kv_lora_rank/2 bytes (first 256 bytes)
  const int64_t dst_idx_base = dst_idx_start + (threadIdx.x * 8);

  uint8_t result[8];
#pragma unroll
  for (int i = 0; i < 8; i++) {
    result[i] =
        fp8::scaled_convert<uint8_t, scalar_t, Fp8KVCacheDataType::kFp8E4M3>(
            vals[i], tile_scale);
  }

  // Store as aligned 64-bit writes
  *reinterpret_cast<uint64_t*>(&kv_cache[dst_idx_base]) =
      *reinterpret_cast<const uint64_t*>(result);
}

template <typename scalar_t, typename cache_t, Fp8KVCacheDataType kv_dt>
__global__ void indexer_k_quant_and_cache_kernel(
    const scalar_t* __restrict__ k,  // [num_tokens, head_dim]
    cache_t* __restrict__ kv_cache,  // [num_blocks, block_size, cache_stride]
    const int64_t* __restrict__ slot_mapping,  // [num_tokens]
    const int head_dim,                        // dimension of each head
    const int quant_block_size,                // quantization block size
    const int cache_block_size,                // cache block size
    const int cache_stride,  // stride for each token in kv_cache

    const bool use_ue8m0  // use ue8m0 scale format
) {
  constexpr int VEC_SIZE = 4;
  const int64_t token_idx = blockIdx.x;
  const int64_t head_dim_idx = (blockIdx.y * blockDim.y * blockDim.x +
                                threadIdx.y * blockDim.x + threadIdx.x) *
                               VEC_SIZE;
  const int64_t slot_idx = slot_mapping[token_idx];
  const int64_t block_idx = slot_idx / cache_block_size;
  const int64_t block_offset = slot_idx % cache_block_size;

  // NOTE: slot_idx can be -1 if the token is padded
  if (slot_idx < 0 || (head_dim_idx >= head_dim)) {
    return;
  }

  float2 k_val = (reinterpret_cast<const float2*>(
      k))[(token_idx * head_dim + head_dim_idx) / VEC_SIZE];
  scalar_t* k_val_ptr = reinterpret_cast<scalar_t*>(&k_val);
  float amax = 0.0f;
  for (int i = 0; i < VEC_SIZE; i++) {
    amax = fmaxf(amax, fabsf(float(k_val_ptr[i])));
  }

  // Reduced amax
  for (int mask = 16; mask > 0; mask /= 2) {
#ifdef USE_ROCM
    amax = fmaxf(amax, __shfl_xor_sync(uint64_t(-1), amax, mask));
#else
    amax = fmaxf(amax, __shfl_xor_sync(unsigned(-1), amax, mask));
#endif
  }

  float scale = fmaxf(amax, 1e-4) / kFp8ScaleDivisor;

  if (use_ue8m0) {
    scale = exp2f(ceilf(log2f(scale)));
  }

  const int64_t dst_offset = block_idx * cache_block_size * cache_stride +
                             block_offset * head_dim + head_dim_idx;
  for (int i = 0; i < VEC_SIZE; i++) {
    kv_cache[dst_offset + i] =
        fp8::scaled_convert<cache_t, scalar_t, kv_dt>(k_val_ptr[i], scale);
  }
  if (threadIdx.x == 0) {
    const int64_t dst_scale_idx =
        block_idx * cache_block_size * cache_stride +
        cache_block_size * head_dim +
        (block_offset * head_dim + head_dim_idx) * 4 / quant_block_size;
    reinterpret_cast<float*>(kv_cache)[dst_scale_idx / 4] = scale;
  }
}

template <int BLOCK_Y_SIZE>
__global__ void cp_gather_indexer_k_quant_cache_kernel(
    const char* __restrict__ kv_cache,  // [num_blocks, block_size,
                                        // cache_stride]
    char* __restrict__ dst_k,           // [num_tokens, head_dim]
    char* __restrict__ dst_scale,  // [num_tokens, head_dim / quant_block_size *
                                   // 4]
    const int* __restrict__ block_table,  // [batch_size, num_blocks]
    const int* __restrict__ cu_seq_lens,  // [batch_size + 1]
    const int batch_size,                 // batch size
    const int64_t token_stride,           // stride for each token in dst_k
    const int64_t head_dim,               // dimension of each head
    const int64_t block_stride,           // stride for each block in kv_cache
    const int64_t cache_token_stride,     // stride for each token in kv_cache
    const int64_t cache_block_size,  // num_tokens for each block in kv_cache
    const int num_blocks,            // number of blocks
    const int num_tokens,            // number of tokens
    const int quant_block_size       // quantization block size
) {
  constexpr int VEC_SIZE = sizeof(float4) / sizeof(char);
  const int token_idx = blockIdx.x * blockDim.y + threadIdx.y;
  const int head_idx = (blockIdx.y * blockDim.x + threadIdx.x) * VEC_SIZE;
  // Find batch index within a block
  __shared__ int batch_idx[BLOCK_Y_SIZE];
  if (threadIdx.x == 0) {
    batch_idx[threadIdx.y] = -1;
  }
  __syncthreads();

  for (int iter = 0; iter < cuda_utils::ceil_div(batch_size, int(blockDim.x));
       iter++) {
    int tid = iter * blockDim.x + threadIdx.x;
    if (tid < batch_size) {
      const int seq_start = cu_seq_lens[tid];
      const int seq_end = cu_seq_lens[tid + 1];
      if (token_idx >= seq_start && token_idx < seq_end) {
        batch_idx[threadIdx.y] = tid;
      }
    }
  }

  __syncthreads();

  // num_tokens may be an allocation upper bound when Python avoids a D2H sync.
  // Only tokens covered by the exact device-side cu_seq_lens are valid to
  // gather.
  const int batch = batch_idx[threadIdx.y];
  if (head_idx >= head_dim || token_idx >= num_tokens || batch < 0) {
    return;
  }
  const int inbatch_seq_idx = token_idx - cu_seq_lens[batch];
  const int block_idx =
      block_table[batch * num_blocks + inbatch_seq_idx / cache_block_size];
  const int64_t src_block_offset = block_idx * block_stride;
  const int64_t cache_inblock_offset =
      (inbatch_seq_idx % cache_block_size) * head_dim + head_idx;
  const int64_t src_inblock_offset = src_block_offset + cache_inblock_offset;
  const int64_t dst_inblock_offset = token_idx * token_stride + head_idx;

  reinterpret_cast<float4*>(dst_k)[dst_inblock_offset / VEC_SIZE] =
      reinterpret_cast<const float4*>(kv_cache)[src_inblock_offset / VEC_SIZE];
  ;
  if (threadIdx.x == 0) {
    const int64_t src_scale_offset =
        src_block_offset + cache_block_size * head_dim +
        cache_inblock_offset * 4 / quant_block_size;
    reinterpret_cast<float*>(dst_scale)[dst_inblock_offset / quant_block_size] =
        reinterpret_cast<const float*>(kv_cache)[src_scale_offset / 4];
  }
}

}  // namespace vllm

// KV_T is the data type of key and value tensors.
// CACHE_T is the stored data type of kv-cache.
// KV_DTYPE is the real data type of kv-cache.
#define CALL_RESHAPE_AND_CACHE(KV_T, CACHE_T, KV_DTYPE)                     \
  vllm::reshape_and_cache_kernel<KV_T, CACHE_T, KV_DTYPE>                   \
      <<<grid, block, 0, stream>>>(                                         \
          reinterpret_cast<KV_T*>(key.data_ptr()),                          \
          reinterpret_cast<KV_T*>(value.data_ptr()),                        \
          reinterpret_cast<CACHE_T*>(key_cache.data_ptr()),                 \
          reinterpret_cast<CACHE_T*>(value_cache.data_ptr()),               \
          slot_mapping.const_data_ptr<int64_t>(), key_stride, value_stride, \
          num_heads, head_size, block_size, x,                              \
          reinterpret_cast<const float*>(k_scale.data_ptr()),               \
          reinterpret_cast<const float*>(v_scale.data_ptr()));

void reshape_and_cache(
    torch::stable::Tensor& key,    // [num_tokens, num_heads, head_size]
    torch::stable::Tensor& value,  // [num_tokens, num_heads, head_size]
    torch::stable::Tensor&
        key_cache,  // [num_blocks, num_heads, head_size/x, block_size, x]
    torch::stable::Tensor&
        value_cache,  // [num_blocks, num_heads, head_size, block_size]
    torch::stable::Tensor& slot_mapping,  // [num_tokens]
    const std::string& kv_cache_dtype, torch::stable::Tensor& k_scale,
    torch::stable::Tensor& v_scale) {
  int num_tokens = slot_mapping.size(0);
  int num_heads = key.size(1);
  int head_size = key.size(2);
  int block_size = key_cache.size(3);
  int x = key_cache.size(4);

  int key_stride = key.stride(0);
  int value_stride = value.stride(0);
  int head_div_x = head_size / x;

  dim3 grid(num_tokens);
  dim3 block(std::min(num_heads * head_div_x, 512));
  const torch::stable::accelerator::DeviceGuard device_guard(
      key.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream();

  DISPATCH_BY_KV_CACHE_DTYPE(key.scalar_type(), kv_cache_dtype,
                             CALL_RESHAPE_AND_CACHE);
}

// KV_T is the data type of key and value tensors.
// CACHE_T is the stored data type of kv-cache.
// KV_DTYPE is the real data type of kv-cache.
#define CALL_RESHAPE_AND_CACHE_FLASH(KV_T, CACHE_T, KV_DTYPE)                \
  vllm::reshape_and_cache_flash_kernel<KV_T, CACHE_T, KV_DTYPE>              \
      <<<grid, block, 0, stream>>>(                                          \
          reinterpret_cast<KV_T*>(key.data_ptr()),                           \
          reinterpret_cast<KV_T*>(value.data_ptr()),                         \
          reinterpret_cast<CACHE_T*>(key_cache.data_ptr()),                  \
          reinterpret_cast<CACHE_T*>(value_cache.data_ptr()),                \
          slot_mapping.const_data_ptr<int64_t>(), block_stride, page_stride, \
          head_stride, key_stride, value_stride, num_heads, head_size,       \
          block_size, reinterpret_cast<const float*>(k_scale.data_ptr()),    \
          reinterpret_cast<const float*>(v_scale.data_ptr()),                \
          kv_scale_stride);

void reshape_and_cache_flash(
    torch::stable::Tensor& key,    // [num_tokens, num_heads, head_size]
    torch::stable::Tensor& value,  // [num_tokens, num_heads, head_size]
    torch::stable::Tensor&
        key_cache,  // [num_blocks, block_size, num_heads, head_size]
    torch::stable::Tensor&
        value_cache,  // [num_blocks, block_size, num_heads, head_size]
    torch::stable::Tensor& slot_mapping,  // [num_tokens] or [num_actual_tokens]
    const std::string& kv_cache_dtype,
    torch::stable::Tensor& k_scale,    // [1] or [num_heads]
    torch::stable::Tensor& v_scale) {  // [1] or [num_heads]
  // NOTE(woosuk): In vLLM V1, key.size(0) can be different from
  // slot_mapping.size(0) because of padding for CUDA graphs.
  // In vLLM V0, key.size(0) is always equal to slot_mapping.size(0) because
  // both include padding.
  // In vLLM V1, however, key.size(0) can be larger than slot_mapping.size(0)
  // since key includes padding for CUDA graphs, while slot_mapping does not.
  // In this case, slot_mapping.size(0) represents the actual number of tokens
  // before padding.
  // For compatibility with both cases, we use slot_mapping.size(0) as the
  // number of tokens.
  int num_tokens = slot_mapping.size(0);
  int num_heads = key.size(1);
  int head_size = key.size(2);

  const torch::stable::accelerator::DeviceGuard device_guard(
      key.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream();

  if (kv_cache_dtype == "nvfp4") {
#if defined(ENABLE_NVFP4_SM100) || defined(ENABLE_NVFP4_SM120)
    // NVFP4 dispatch is compiled separately for SM100+.
    extern void reshape_and_cache_nvfp4_dispatch(
        torch::stable::Tensor & key, torch::stable::Tensor & value,
        torch::stable::Tensor & key_cache, torch::stable::Tensor & value_cache,
        torch::stable::Tensor & slot_mapping, torch::stable::Tensor & k_scale,
        torch::stable::Tensor & v_scale);
    reshape_and_cache_nvfp4_dispatch(key, value, key_cache, value_cache,
                                     slot_mapping, k_scale, v_scale);
    return;
#else
    STD_TORCH_CHECK(
        false,
        "NVFP4 KV cache requires SM100+ (Blackwell). "
        "Please rebuild vllm with a Blackwell-compatible CUDA target.");
#endif
  }

  // Original FP8/auto path.
  int block_size = key_cache.size(1);

  int64_t key_stride = key.stride(0);
  int64_t value_stride = value.stride(0);
  int64_t block_stride = key_cache.stride(0);
  int64_t page_stride = key_cache.stride(1);
  int64_t head_stride = key_cache.stride(2);
  STD_TORCH_CHECK(key_cache.stride(0) == value_cache.stride(0));

  STD_TORCH_CHECK(k_scale.sizes().equals(v_scale.sizes()),
                  "k_scale and v_scale must have the same shape");
  STD_TORCH_CHECK(k_scale.numel() == 1 || k_scale.numel() == num_heads,
                  "k_scale and v_scale must be of shape [1] or [num_heads]");
  int kv_scale_stride = (k_scale.numel() > 1) ? 1 : 0;

  dim3 grid(num_tokens);
  dim3 block(std::min(num_heads * head_size, 512));

  DISPATCH_BY_KV_CACHE_DTYPE(key.scalar_type(), kv_cache_dtype,
                             CALL_RESHAPE_AND_CACHE_FLASH);
}

namespace vllm {
namespace {

constexpr int kByteV2TileSize = 16;
constexpr int kByteV2TileElems = kByteV2TileSize * kByteV2TileSize;
constexpr int kByteV2PackedTileElems = kByteV2TileElems / 2;
constexpr int kByteV2RawTileBytes = kByteV2TileElems * 2;
constexpr int kByteV2FastTilePayloadBytes =
    1 + kByteV2TileElems + kByteV2PackedTileElems + 1;
constexpr int kByteV2PageHeaderBytes = 16;
constexpr int kByteV2PageStatusOffset = 0;
constexpr int kByteV2PageValidRowsOffset = 1;
constexpr int kByteV2PageLayoutVersionOffset = 2;
constexpr uint8_t kByteV2PayloadLayoutVersionV3 = 3;
constexpr int kByteV2PageHeaderBytesV3 = 128;
constexpr int kByteV2KvHeadMetaBytesV3 = 32;
constexpr int kByteV2TilePayloadBytesV3 = 384;
constexpr int kByteV2TilePayloadStripeBytesV3 = 96;
constexpr int kByteV2TilePayloadStripePairsV3 = 32;
constexpr uint8_t kByteV2PageStatusEmpty = 0;
constexpr uint8_t kByteV2PageStatusCompressed = 1;
constexpr uint8_t kByteV2PageStatusRawFallback = 2;
constexpr int kByteV2MaxTilesPerBlock = 2048;
constexpr int kByteV2MaxPrefillDirectWarps = 8;

constexpr int kByteV2CacheUpdateErrorInvalidPageState = 1;
constexpr int kByteV2CacheUpdateErrorInvalidSlot = 2;
constexpr int kByteV2CacheUpdateErrorDuplicateSlot = 3;
constexpr int kByteV2CacheUpdateErrorFinalizedBlockUpdate = 4;
constexpr int kByteV2CacheUpdateErrorFallbackPoolMissing = 5;
constexpr int kByteV2CacheUpdateErrorFallbackPoolInvalidSlot = 6;
constexpr int kByteV2CacheUpdateErrorFallbackPoolExhausted = 7;
constexpr int kByteV2CacheUpdateErrorInvalidValidRows = 8;
constexpr int kByteV2CacheUpdateErrorNoTouchedToken = 9;
constexpr int kByteV2CacheUpdateErrorOutlierArenaExhausted = 10;
constexpr int kByteV2OutlierMetaCountBits = 8;
constexpr int kByteV2OutlierMetaCountMask =
    (1 << kByteV2OutlierMetaCountBits) - 1;
constexpr int kByteV2MaxOutlierArenaOffset =
    (1 << (31 - kByteV2OutlierMetaCountBits)) - 1;

__device__ __forceinline__ int byte_v2_outlier_meta_count(
    const int32_t meta) {
  return meta & kByteV2OutlierMetaCountMask;
}

__device__ __forceinline__ int byte_v2_outlier_meta_offset(
    const int32_t meta) {
  return static_cast<int>(static_cast<uint32_t>(meta) >>
                          kByteV2OutlierMetaCountBits);
}

__device__ __forceinline__ int byte_v2_outlier_entry_elem(
    const int32_t entry) {
  return entry & 0xff;
}

__device__ __forceinline__ uint16_t byte_v2_outlier_entry_bits(
    const int32_t entry) {
  return static_cast<uint16_t>((static_cast<uint32_t>(entry) >> 8) & 0xffffU);
}

__device__ __forceinline__ bool byte_v2_may_have_block_outliers(
    const int32_t* __restrict__ outlier_block_flags, const int physical_block) {
  return outlier_block_flags == nullptr ||
         outlier_block_flags[physical_block] != 0;
}

__host__ __device__ __forceinline__ int byte_v2_outlier_tile_bitmap_words(
    const int total_tiles) {
  return (total_tiles + 31) / 32;
}

__device__ __forceinline__ bool byte_v2_may_have_tile_outliers(
    const int32_t* __restrict__ outlier_tile_bitmap, const int physical_block,
    const int bitmap_words, const int tile_idx) {
  if (outlier_tile_bitmap == nullptr) {
    return true;
  }
  const uint32_t word = static_cast<uint32_t>(
      outlier_tile_bitmap[physical_block * bitmap_words + tile_idx / 32]);
  return (word & (1u << (tile_idx & 31))) != 0;
}

__device__ __forceinline__ void byte_v2_set_outlier_tile_bitmap(
    int32_t* __restrict__ outlier_tile_bitmap, const int physical_block,
    const int bitmap_words, const int tile_idx) {
  if (outlier_tile_bitmap == nullptr) {
    return;
  }
  atomicOr(
      reinterpret_cast<int*>(outlier_tile_bitmap +
                             physical_block * bitmap_words + tile_idx / 32),
      static_cast<int>(1u << (tile_idx & 31)));
}

__device__ __forceinline__ void byte_v2_record_cache_update_error(
    int32_t* __restrict__ error, const int code) {
  atomicCAS(reinterpret_cast<int*>(error), 0, code);
}

__device__ __forceinline__ void byte_v2_record_cache_update_error_detail(
    int32_t* __restrict__ error, const int code, const int detail) {
  const int old = atomicCAS(reinterpret_cast<int*>(error), 0, code);
  if (old == 0) {
    error[1] = detail;
  }
}

const char* byte_v2_cache_update_error_message(const int code) {
  switch (code) {
    case kByteV2CacheUpdateErrorInvalidPageState:
      return "invalid page state";
    case kByteV2CacheUpdateErrorInvalidSlot:
      return "invalid slot mapping";
    case kByteV2CacheUpdateErrorDuplicateSlot:
      return "duplicate token slot in one cache update";
    case kByteV2CacheUpdateErrorFinalizedBlockUpdate:
      return "attempted to update a finalized KV block";
    case kByteV2CacheUpdateErrorFallbackPoolMissing:
      return "sparse fallback pool is required but missing";
    case kByteV2CacheUpdateErrorFallbackPoolInvalidSlot:
      return "sparse fallback block id is invalid";
    case kByteV2CacheUpdateErrorFallbackPoolExhausted:
      return "sparse fallback pool exhausted";
    case kByteV2CacheUpdateErrorInvalidValidRows:
      return "invalid Byte-v2 valid row count";
    case kByteV2CacheUpdateErrorNoTouchedToken:
      return "touched block has no source token";
    case kByteV2CacheUpdateErrorOutlierArenaExhausted:
      return "outlier arena exhausted";
    default:
      return "unknown cache update error";
  }
}
constexpr int kByteV2MaxHeadSize = 128;
constexpr int kByteV2MaxQPerKv = 8;

__device__ __forceinline__ float byte_v2_bf16_bits_to_float(
    const uint16_t bits) {
  return __uint_as_float(static_cast<uint32_t>(bits) << 16);
}

#ifndef USE_ROCM
__device__ __forceinline__ __nv_bfloat16 byte_v2_bf16_bits_to_wmma(
    const uint16_t bits) {
  return __ushort_as_bfloat16(bits);
}
#endif

__device__ __forceinline__ uint16_t byte_v2_float_to_bf16_bits(
    const float value) {
  const uint32_t bits = __float_as_uint(value);
  const uint32_t lsb = (bits >> 16) & 1U;
  return static_cast<uint16_t>((bits + 0x7fffU + lsb) >> 16);
}

__device__ __forceinline__ float byte_v2_warp_reduce_max(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value = max(value, __shfl_down_sync(0xffffffffU, value, offset));
  }
  return value;
}

__device__ __forceinline__ float byte_v2_warp_reduce_sum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffU, value, offset);
  }
  return value;
}

#ifndef USE_ROCM
__device__ __forceinline__ float byte_v2_wmma_to_float(
    const __nv_bfloat16 value) {
  return __bfloat162float(value);
}
#endif

__device__ __forceinline__ int byte_v2_tile_index(
    const bool is_value, const int kv_head, const int dim_tile,
    const int k_dim_tiles, const int v_dim_tiles) {
  const int kind_offset = is_value ? k_dim_tiles : 0;
  return kv_head * (k_dim_tiles + v_dim_tiles) + kind_offset + dim_tile;
}

__device__ __forceinline__ int byte_v2_tile_start(
    const bool is_value, const int kv_head, const int dim_tile,
    const int k_dim_tiles, const int v_dim_tiles) {
  return kByteV2PageHeaderBytes +
         byte_v2_tile_index(is_value, kv_head, dim_tile, k_dim_tiles,
                            v_dim_tiles) *
             kByteV2FastTilePayloadBytes;
}

__host__ __device__ __forceinline__ int byte_v2_align_up(const int value,
                                                         const int alignment) {
  return ((value + alignment - 1) / alignment) * alignment;
}

__host__ __device__ __forceinline__ int
byte_v2_v3_kv_head_meta_region_bytes(const int num_kv_heads) {
  return byte_v2_align_up(num_kv_heads * kByteV2KvHeadMetaBytesV3, 128);
}

__host__ __device__ __forceinline__ int
byte_v2_v3_k_payload_offset(const int num_kv_heads) {
  return kByteV2PageHeaderBytesV3 +
         byte_v2_v3_kv_head_meta_region_bytes(num_kv_heads);
}

__host__ __device__ __forceinline__ int byte_v2_v3_v_payload_offset(
    const int num_kv_heads, const int k_dim_tiles) {
  return byte_v2_v3_k_payload_offset(num_kv_heads) +
         num_kv_heads * k_dim_tiles * kByteV2TilePayloadBytesV3;
}

__host__ __device__ __forceinline__ int byte_v2_v3_page_size_bytes(
    const int num_kv_heads, const int k_dim_tiles, const int v_dim_tiles) {
  return byte_v2_v3_v_payload_offset(num_kv_heads, k_dim_tiles) +
         num_kv_heads * v_dim_tiles * kByteV2TilePayloadBytesV3;
}

__device__ __forceinline__ int byte_v2_v3_kv_head_meta_start(
    const int kv_head) {
  return kByteV2PageHeaderBytesV3 + kv_head * kByteV2KvHeadMetaBytesV3;
}

__device__ __forceinline__ int byte_v2_v3_tile_payload_start(
    const bool is_value, const int kv_head, const int dim_tile,
    const int num_kv_heads, const int k_dim_tiles, const int v_dim_tiles) {
  if (is_value) {
    return byte_v2_v3_v_payload_offset(num_kv_heads, k_dim_tiles) +
           (kv_head * v_dim_tiles + dim_tile) * kByteV2TilePayloadBytesV3;
  }
  return byte_v2_v3_k_payload_offset(num_kv_heads) +
         (kv_head * k_dim_tiles + dim_tile) * kByteV2TilePayloadBytesV3;
}

__device__ __forceinline__ int byte_v2_total_tiles(
    const int num_kv_heads, const int head_size, const int head_size_v) {
  return num_kv_heads * (head_size / kByteV2TileSize +
                         head_size_v / kByteV2TileSize);
}

__device__ __forceinline__ int byte_v2_raw_key_bytes(
    const int num_kv_heads, const int head_size) {
  return kByteV2TileSize * num_kv_heads * head_size * 2;
}

__device__ __forceinline__ int byte_v2_raw_value_bytes(
    const int num_kv_heads, const int head_size_v) {
  return kByteV2TileSize * num_kv_heads * head_size_v * 2;
}

__device__ __forceinline__ uint16_t byte_v2_load_u16(
    const uint8_t* __restrict__ ptr) {
  return static_cast<uint16_t>(ptr[0]) |
         (static_cast<uint16_t>(ptr[1]) << 8);
}

__device__ __forceinline__ uint16_t byte_v2_load_aligned_u16(
    const uint8_t* __restrict__ ptr) {
  return *reinterpret_cast<const uint16_t*>(ptr);
}

__device__ __forceinline__ uint32_t byte_v2_load_u32(
    const uint8_t* __restrict__ ptr) {
  return static_cast<uint32_t>(ptr[0]) |
         (static_cast<uint32_t>(ptr[1]) << 8) |
         (static_cast<uint32_t>(ptr[2]) << 16) |
         (static_cast<uint32_t>(ptr[3]) << 24);
}

__device__ __forceinline__ uint32_t byte_v2_load_aligned_u32(
    const uint8_t* __restrict__ ptr) {
  return *reinterpret_cast<const uint32_t*>(ptr);
}

__device__ __forceinline__ uint16_t byte_v2_load_payload_low_pair(
    const uint8_t* __restrict__ ptr, const bool use_aligned_u16) {
  return use_aligned_u16 ? byte_v2_load_aligned_u16(ptr)
                         : byte_v2_load_u16(ptr);
}

__device__ __forceinline__ void byte_v2_load_v3_stripe_pair(
    const uint8_t* __restrict__ stripe_ptr, const int lane,
    const bool use_warp_stripe_load, int& low0, int& low1, int& packed) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
  if (use_warp_stripe_load) {
    uint32_t word = 0;
    if (lane < 24) {
      word = byte_v2_load_aligned_u32(stripe_ptr + lane * 4);
    }
    const int word_group = lane >> 2;
    const int byte_shift = (lane & 3) * 8;
    const uint32_t low0_word =
        __shfl_sync(0xffffffffu, word, word_group);
    const uint32_t low1_word =
        __shfl_sync(0xffffffffu, word, 8 + word_group);
    const uint32_t code_word =
        __shfl_sync(0xffffffffu, word, 16 + word_group);
    low0 = static_cast<int>((low0_word >> byte_shift) & 0xff);
    low1 = static_cast<int>((low1_word >> byte_shift) & 0xff);
    packed = static_cast<int>((code_word >> byte_shift) & 0xff);
    return;
  }
#else
  (void)use_warp_stripe_load;
#endif
  low0 = static_cast<int>(stripe_ptr[lane]);
  low1 = static_cast<int>(stripe_ptr[32 + lane]);
  packed = static_cast<int>(stripe_ptr[64 + lane]);
}

__device__ __forceinline__ void byte_v2_cp_async_16(
    uint8_t* __restrict__ dst_shared, const uint8_t* __restrict__ src_global) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  const uint32_t dst =
      static_cast<uint32_t>(__cvta_generic_to_shared(dst_shared));
  asm volatile("cp.async.ca.shared.global [%0], [%1], 16;\n" ::"r"(dst),
               "l"(src_global));
#else
  if (threadIdx.x == 0) {
    for (int i = 0; i < 16; ++i) {
      dst_shared[i] = src_global[i];
    }
  }
#endif
}

__device__ __forceinline__ void byte_v2_cp_async_commit_group() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.commit_group;\n" ::);
#endif
}

__device__ __forceinline__ void byte_v2_cp_async_wait_all() {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("cp.async.wait_group 0;\n" ::);
#endif
}

__device__ __forceinline__ void byte_v2_cp_async_copy_v3_tile_payload(
    uint8_t* __restrict__ staged_tile, const uint8_t* __restrict__ page,
    const bool is_value, const int tid, const int kv_head,
    const int dim_tile, const int num_kv_heads, const int k_dim_tiles,
    const int v_dim_tiles) {
  const int tile_start = byte_v2_v3_tile_payload_start(
      is_value, kv_head, dim_tile, num_kv_heads, k_dim_tiles, v_dim_tiles);
  constexpr int chunks = kByteV2TilePayloadBytesV3 / 16;
  for (int chunk = tid; chunk < chunks; chunk += blockDim.x) {
    byte_v2_cp_async_16(staged_tile + chunk * 16,
                        page + tile_start + chunk * 16);
  }
}

__device__ __forceinline__ void byte_v2_store_u16(uint8_t* __restrict__ ptr,
                                                  const uint16_t value) {
  ptr[0] = static_cast<uint8_t>(value & 0xff);
  ptr[1] = static_cast<uint8_t>(value >> 8);
}

__device__ __forceinline__ void byte_v2_store_u32(uint8_t* __restrict__ ptr,
                                                  const uint32_t value) {
  ptr[0] = static_cast<uint8_t>(value & 0xff);
  ptr[1] = static_cast<uint8_t>((value >> 8) & 0xff);
  ptr[2] = static_cast<uint8_t>((value >> 16) & 0xff);
  ptr[3] = static_cast<uint8_t>((value >> 24) & 0xff);
}

__host__ __device__ __forceinline__ bool byte_v2_page_uses_v3_layout(
    const int page_size_bytes, const int num_kv_heads, const int head_size,
    const int head_size_v) {
  return page_size_bytes == byte_v2_v3_page_size_bytes(
                                num_kv_heads, head_size / kByteV2TileSize,
                                head_size_v / kByteV2TileSize);
}

__device__ __forceinline__ int64_t byte_v2_tile_fallback_capacity(
    const int fallback_pool_blocks, const int raw_block_bytes) {
  return (static_cast<int64_t>(fallback_pool_blocks) * raw_block_bytes) /
         kByteV2RawTileBytes;
}

__device__ __forceinline__ int byte_v2_allocate_tile_fallback_slot(
    int32_t* __restrict__ fallback_tile_next_slot,
    const int64_t tile_capacity) {
  const int tile_ordinal = atomicAdd(fallback_tile_next_slot, 1);
  if (static_cast<int64_t>(tile_ordinal) >= tile_capacity) {
    return -1;
  }
  // Full raw-block fallback slots grow from the beginning of fallback_pool.
  // Raw tile slots grow from the end so the two granularities do not overwrite
  // each other during mixed prefill/decode runs.
  return static_cast<int>(tile_capacity - 1 - tile_ordinal);
}

__device__ __forceinline__ uint16_t byte_v2_load_raw_bits_from_tile_pool(
    const uint8_t* __restrict__ fallback_pool, const int tile_slot,
    const int row, const int dim_in_tile) {
  const int elem_offset = (row * kByteV2TileSize + dim_in_tile) * 2;
  return byte_v2_load_u16(
      fallback_pool + static_cast<int64_t>(tile_slot) * kByteV2RawTileBytes +
      elem_offset);
}

__device__ __forceinline__ void byte_v2_store_raw_bits_to_tile_pool(
    uint8_t* __restrict__ fallback_pool, const int tile_slot, const int row,
    const int dim_in_tile, const uint16_t bits) {
  const int elem_offset = (row * kByteV2TileSize + dim_in_tile) * 2;
  byte_v2_store_u16(
      fallback_pool + static_cast<int64_t>(tile_slot) * kByteV2RawTileBytes +
          elem_offset,
      bits);
}

__device__ __forceinline__ uint16_t byte_v2_load_raw_bits_from_block(
    const uint8_t* __restrict__ raw_block, const bool is_value, const int row,
    const int kv_head, const int dim, const int num_kv_heads,
    const int head_size, const int head_size_v) {
  const int raw_kind_offset =
      is_value ? byte_v2_raw_key_bytes(num_kv_heads, head_size) : 0;
  const int kind_head_size = is_value ? head_size_v : head_size;
  const int elem_offset =
      ((row * num_kv_heads + kv_head) * kind_head_size + dim) * 2;
  return byte_v2_load_u16(raw_block + raw_kind_offset + elem_offset);
}

__device__ __forceinline__ void byte_v2_store_raw_bits_to_block(
    uint8_t* __restrict__ raw_block, const bool is_value, const int row,
    const int kv_head, const int dim, const int num_kv_heads,
    const int head_size, const int head_size_v, const uint16_t bits) {
  const int raw_kind_offset =
      is_value ? byte_v2_raw_key_bytes(num_kv_heads, head_size) : 0;
  const int kind_head_size = is_value ? head_size_v : head_size;
  const int elem_offset =
      ((row * num_kv_heads + kv_head) * kind_head_size + dim) * 2;
  byte_v2_store_u16(raw_block + raw_kind_offset + elem_offset, bits);
}

__device__ __forceinline__ uint16_t byte_v2_load_raw_bits_from_page(
    const uint8_t* __restrict__ page, const bool is_value, const int row,
    const int kv_head, const int dim, const int num_kv_heads,
    const int head_size, const int head_size_v) {
  return byte_v2_load_raw_bits_from_block(page + kByteV2PageHeaderBytes,
                                          is_value, row, kv_head, dim,
                                          num_kv_heads, head_size, head_size_v);
}

__device__ __forceinline__ uint16_t byte_v2_load_compressed_bits_v3(
    const uint8_t* __restrict__ page, const bool is_value, const int row,
    const int kv_head, const int dim, const int k_dim_tiles,
    const int v_dim_tiles) {
  const int dim_tile = dim / kByteV2TileSize;
  const int dim_in_tile = dim % kByteV2TileSize;
  const int elem = is_value ? row * kByteV2TileSize + dim_in_tile
                            : dim_in_tile * kByteV2TileSize + row;
  const int pair_idx = elem / 2;
  const int stripe = pair_idx / kByteV2TilePayloadStripePairsV3;
  const int lane = pair_idx % kByteV2TilePayloadStripePairsV3;
  const int meta_start = byte_v2_v3_kv_head_meta_start(kv_head);
  const int base = static_cast<int>(
      page[meta_start + (is_value ? 8 + dim_tile : dim_tile)]);
  const int payload_offset = static_cast<int>(
      byte_v2_load_u32(page + (is_value ? 16 : 12)));
  const int dim_tiles = is_value ? v_dim_tiles : k_dim_tiles;
  const int tile_start =
      payload_offset +
      (kv_head * dim_tiles + dim_tile) * kByteV2TilePayloadBytesV3;
  const uint8_t* __restrict__ stripe_ptr =
      page + tile_start + stripe * kByteV2TilePayloadStripeBytesV3;
  const int low = static_cast<int>(
      stripe_ptr[(elem & 1) ? 32 + lane : lane]);
  const int packed = static_cast<int>(stripe_ptr[64 + lane]);
  const int code = (elem & 1) ? ((packed >> 4) & 0x0f) : (packed & 0x0f);
  const int low_exp_lsb = low >> 7;
  const int delta_hi = code & 0x07;
  const int exp_hi =
      (base >> 1) + delta_hi + ((base & 1) & (low_exp_lsb ^ 1));
  const int high = ((code & 0x08) << 4) | exp_hi;
  return static_cast<uint16_t>((high << 8) | low);
}

__device__ __forceinline__ uint16_t byte_v2_load_compressed_bits(
    const uint8_t* __restrict__ page, const bool is_value, const int row,
    const int kv_head, const int dim, const int k_dim_tiles,
    const int v_dim_tiles) {
  if (page[kByteV2PageLayoutVersionOffset] == kByteV2PayloadLayoutVersionV3) {
    return byte_v2_load_compressed_bits_v3(
        page, is_value, row, kv_head, dim, k_dim_tiles, v_dim_tiles);
  }
  const int dim_tile = dim / kByteV2TileSize;
  const int dim_in_tile = dim % kByteV2TileSize;
  const int elem = is_value ? row * kByteV2TileSize + dim_in_tile
                            : dim_in_tile * kByteV2TileSize + row;
  const int tile_start =
      byte_v2_tile_start(is_value, kv_head, dim_tile, k_dim_tiles, v_dim_tiles);
  const int base = static_cast<int>(page[tile_start]);
  const int low = static_cast<int>(page[tile_start + 2 + elem]);
  const int packed =
      static_cast<int>(page[tile_start + 2 + kByteV2TileElems + elem / 2]);
  const int code = (elem & 1) ? ((packed >> 4) & 0x0f) : (packed & 0x0f);
  const int low_exp_lsb = low >> 7;
  const int delta_hi = code & 0x07;
  const int exp_hi =
      (base >> 1) + delta_hi + ((base & 1) & (low_exp_lsb ^ 1));
  const int high = ((code & 0x08) << 4) | exp_hi;
  return static_cast<uint16_t>((high << 8) | low);
}

__device__ __forceinline__ bool byte_v2_page_has_v3_payload(
    const uint8_t* __restrict__ page) {
  return page[kByteV2PageLayoutVersionOffset] ==
         kByteV2PayloadLayoutVersionV3;
}

__device__ __forceinline__ uint8_t byte_v2_tile_fallback_flag(
    const uint8_t* __restrict__ page, const bool is_value, const int kv_head,
    const int dim_tile, const int k_dim_tiles, const int v_dim_tiles) {
  if (byte_v2_page_has_v3_payload(page)) {
    const int meta_start = byte_v2_v3_kv_head_meta_start(kv_head);
    const int mask_offset = meta_start + (is_value ? 18 : 16);
    const uint16_t fallback_mask = byte_v2_load_u16(page + mask_offset);
    return static_cast<uint8_t>((fallback_mask >> dim_tile) & 1);
  }
  const int tile_start =
      byte_v2_tile_start(is_value, kv_head, dim_tile, k_dim_tiles, v_dim_tiles);
  return page[tile_start + 1];
}

__device__ __forceinline__ uint16_t byte_v2_overlay_outlier_bits(
    const uint16_t compressed_bits, const int32_t* __restrict__ outlier_arena,
    const int32_t* __restrict__ outlier_tile_bitmap,
    const int32_t* __restrict__ outlier_tile_meta, const int physical_block,
    const int total_tiles, const int bitmap_words, const int tile_idx,
    const int elem) {
  if (outlier_arena == nullptr || outlier_tile_meta == nullptr) {
    return compressed_bits;
  }
  if (!byte_v2_may_have_tile_outliers(outlier_tile_bitmap, physical_block,
                                      bitmap_words, tile_idx)) {
    return compressed_bits;
  }
  const int32_t meta =
      outlier_tile_meta[physical_block * total_tiles + tile_idx];
  if (meta < 0) {
    return compressed_bits;
  }
  const int count = byte_v2_outlier_meta_count(meta);
  const int offset = byte_v2_outlier_meta_offset(meta);
  for (int i = 0; i < count; ++i) {
    const int32_t entry = outlier_arena[offset + i];
    if (byte_v2_outlier_entry_elem(entry) == elem) {
      return byte_v2_outlier_entry_bits(entry);
    }
  }
  return compressed_bits;
}

__device__ __forceinline__ uint16_t byte_v2_make_bf16_bits_from_fast_code(
    const int base, const int low, const int code) {
  const int low_exp_lsb = low >> 7;
  const int delta_hi = code & 0x07;
  const int exp_hi =
      (base >> 1) + delta_hi + ((base & 1) & (low_exp_lsb ^ 1));
  const int high = ((code & 0x08) << 4) | exp_hi;
  return static_cast<uint16_t>((high << 8) | low);
}

__device__ __forceinline__ uint32_t byte_v2_profile_mix_u32(
    const uint32_t acc, const uint32_t value) {
  return acc ^ value;
}

__device__ __forceinline__ uint32_t byte_v2_profile_compressed_tile(
    const uint8_t* __restrict__ page, const bool is_value, const int tid,
    const int kv_head, const int dim_tile, const int k_dim_tiles,
    const int v_dim_tiles, const int profile_mode,
    const bool use_aligned_u16_payload_load) {
  const int tile_start =
      byte_v2_tile_start(is_value, kv_head, dim_tile, k_dim_tiles, v_dim_tiles);
  uint32_t acc = static_cast<uint32_t>(page[tile_start]);
  acc = byte_v2_profile_mix_u32(
      acc, static_cast<uint32_t>(page[tile_start + 1]));
  if (profile_mode == 5) {
    return acc;
  }

  const int base = static_cast<int>(page[tile_start]);
  const uint8_t* __restrict__ low_ptr = page + tile_start + 2;
  const uint8_t* __restrict__ packed_ptr =
      low_ptr + kByteV2TileElems;
  for (int pair_idx = tid; pair_idx < kByteV2PackedTileElems;
       pair_idx += blockDim.x) {
    const int elem0 = pair_idx * 2;
    const uint16_t low_pair = byte_v2_load_payload_low_pair(
        low_ptr + elem0, use_aligned_u16_payload_load);
    const int packed = static_cast<int>(packed_ptr[pair_idx]);
    if (profile_mode == 6) {
      acc = byte_v2_profile_mix_u32(
          acc, static_cast<uint32_t>(low_pair) ^
                   (static_cast<uint32_t>(packed) << 16));
    } else {
      const uint16_t bits0 = byte_v2_make_bf16_bits_from_fast_code(
          base, static_cast<int>(low_pair & 0xff), packed & 0x0f);
      const uint16_t bits1 = byte_v2_make_bf16_bits_from_fast_code(
          base, static_cast<int>(low_pair >> 8), (packed >> 4) & 0x0f);
      acc = byte_v2_profile_mix_u32(
          acc, static_cast<uint32_t>(bits0) ^
                   (static_cast<uint32_t>(bits1) << 16));
    }
  }
  return acc;
}

__device__ __forceinline__ uint32_t byte_v2_profile_compressed_tile_v3(
    const uint8_t* __restrict__ page, const bool is_value, const int tid,
    const int kv_head, const int dim_tile, const int num_kv_heads,
    const int k_dim_tiles, const int v_dim_tiles, const int profile_mode,
    const bool use_v3_warp_stripe_load) {
  const int meta_start = byte_v2_v3_kv_head_meta_start(kv_head);
  const int base = static_cast<int>(
      page[meta_start + (is_value ? 8 + dim_tile : dim_tile)]);
  const uint16_t fallback_mask =
      byte_v2_load_u16(page + meta_start + (is_value ? 18 : 16));
  uint32_t acc = static_cast<uint32_t>(base) ^
                 (static_cast<uint32_t>(fallback_mask) << 8);
  if (profile_mode == 5) {
    return acc;
  }

  const int tile_start = byte_v2_v3_tile_payload_start(
      is_value, kv_head, dim_tile, num_kv_heads, k_dim_tiles, v_dim_tiles);
  for (int pair_idx = tid; pair_idx < kByteV2PackedTileElems;
       pair_idx += blockDim.x) {
    const int stripe = pair_idx / kByteV2TilePayloadStripePairsV3;
    const int lane = pair_idx % kByteV2TilePayloadStripePairsV3;
    const uint8_t* __restrict__ stripe_ptr =
        page + tile_start + stripe * kByteV2TilePayloadStripeBytesV3;
    int low0 = 0;
    int low1 = 0;
    int packed = 0;
    byte_v2_load_v3_stripe_pair(stripe_ptr, lane, use_v3_warp_stripe_load,
                                low0, low1, packed);
    if (profile_mode == 6) {
      acc = byte_v2_profile_mix_u32(
          acc, static_cast<uint32_t>(low0) ^
                   (static_cast<uint32_t>(low1) << 8) ^
                   (static_cast<uint32_t>(packed) << 16));
    } else {
      const uint16_t bits0 =
          byte_v2_make_bf16_bits_from_fast_code(base, low0, packed & 0x0f);
      const uint16_t bits1 = byte_v2_make_bf16_bits_from_fast_code(
          base, low1, (packed >> 4) & 0x0f);
      acc = byte_v2_profile_mix_u32(
          acc, static_cast<uint32_t>(bits0) ^
                   (static_cast<uint32_t>(bits1) << 16));
    }
  }
  return acc;
}

__device__ __forceinline__ uint32_t byte_v2_profile_compressed_tile_layout(
    const uint8_t* __restrict__ page, const bool is_value, const int tid,
    const int kv_head, const int dim_tile, const int num_kv_heads,
    const int k_dim_tiles, const int v_dim_tiles, const int profile_mode,
    const bool use_aligned_u16_payload_load,
    const bool use_v3_warp_stripe_load = false) {
  if (byte_v2_page_has_v3_payload(page)) {
    return byte_v2_profile_compressed_tile_v3(
        page, is_value, tid, kv_head, dim_tile, num_kv_heads, k_dim_tiles,
        v_dim_tiles, profile_mode, use_v3_warp_stripe_load);
  }
  return byte_v2_profile_compressed_tile(
      page, is_value, tid, kv_head, dim_tile, k_dim_tiles, v_dim_tiles,
      profile_mode, use_aligned_u16_payload_load);
}

__device__ __forceinline__ uint32_t byte_v2_profile_raw_tile(
    const uint8_t* __restrict__ raw_block, const bool is_value, const int tid,
    const int kv_head, const int dim_tile, const int num_kv_heads,
    const int head_size, const int head_size_v, const int valid_rows,
    const int profile_mode) {
  uint32_t acc = static_cast<uint32_t>(dim_tile) ^
                 (static_cast<uint32_t>(is_value) << 8);
  if (profile_mode == 5) {
    return acc;
  }

  const int d0 = dim_tile * kByteV2TileSize;
  for (int idx = tid; idx < kByteV2TileElems; idx += blockDim.x) {
    const int row = is_value ? (idx / kByteV2TileSize)
                             : (idx % kByteV2TileSize);
    const int dim_in_tile = is_value ? (idx % kByteV2TileSize)
                                     : (idx / kByteV2TileSize);
    if (row < valid_rows) {
      const uint16_t bits = byte_v2_load_raw_bits_from_block(
          raw_block, is_value, row, kv_head, d0 + dim_in_tile, num_kv_heads,
          head_size, head_size_v);
      acc = byte_v2_profile_mix_u32(acc, static_cast<uint32_t>(bits));
    }
  }
  return acc;
}

__device__ __forceinline__ uint32_t byte_v2_profile_tile_fallback(
    const uint8_t* __restrict__ fallback_pool,
    const int32_t* __restrict__ fallback_tile_ids, const bool is_value,
    const int tid, const int physical_block, const int kv_head,
    const int dim_tile, const int k_dim_tiles, const int v_dim_tiles,
    const int valid_rows, const int total_tiles, const int profile_mode) {
  const int tile_idx =
      byte_v2_tile_index(is_value, kv_head, dim_tile, k_dim_tiles, v_dim_tiles);
  const int tile_slot =
      (fallback_pool != nullptr && fallback_tile_ids != nullptr)
          ? fallback_tile_ids[physical_block * total_tiles + tile_idx]
          : -1;
  uint32_t acc = static_cast<uint32_t>(tile_idx) ^
                 (static_cast<uint32_t>(tile_slot) << 16);
  if (profile_mode == 5 || tile_slot < 0) {
    return acc;
  }

  for (int idx = tid; idx < kByteV2TileElems; idx += blockDim.x) {
    const int row = is_value ? (idx / kByteV2TileSize)
                             : (idx % kByteV2TileSize);
    const int dim_in_tile = is_value ? (idx % kByteV2TileSize)
                                     : (idx / kByteV2TileSize);
    if (row < valid_rows) {
      const uint16_t bits = byte_v2_load_raw_bits_from_tile_pool(
          fallback_pool, tile_slot, row, dim_in_tile);
      acc = byte_v2_profile_mix_u32(acc, static_cast<uint32_t>(bits));
    }
  }
  return acc;
}

__device__ __forceinline__ uint16_t byte_v2_load_compressed_bits(
    const uint8_t* __restrict__ page, const bool is_value, const int row,
    const int kv_head, const int dim, const int k_dim_tiles,
    const int v_dim_tiles, const uint8_t* __restrict__ fallback_pool,
    const int32_t* __restrict__ fallback_tile_ids,
    const int32_t* __restrict__ outlier_arena,
    const int32_t* __restrict__ outlier_tile_bitmap,
    const int32_t* __restrict__ outlier_tile_meta, const int physical_block,
    const int total_tiles, const int bitmap_words) {
  const int dim_tile = dim / kByteV2TileSize;
  const int dim_in_tile = dim % kByteV2TileSize;
  const int elem = is_value ? row * kByteV2TileSize + dim_in_tile
                            : dim_in_tile * kByteV2TileSize + row;
  const int tile_idx =
      byte_v2_tile_index(is_value, kv_head, dim_tile, k_dim_tiles, v_dim_tiles);
  uint8_t fallback = 0;
  if (page[kByteV2PageLayoutVersionOffset] == kByteV2PayloadLayoutVersionV3) {
    const int meta_start = byte_v2_v3_kv_head_meta_start(kv_head);
    const int mask_offset = meta_start + (is_value ? 18 : 16);
    const uint16_t fallback_mask = byte_v2_load_u16(page + mask_offset);
    fallback = ((fallback_mask >> dim_tile) & 1) != 0;
  } else {
    const int tile_start =
        byte_v2_tile_start(is_value, kv_head, dim_tile, k_dim_tiles,
                           v_dim_tiles);
    fallback = page[tile_start + 1];
  }
  if (fallback != 0) {
    if (fallback_pool != nullptr && fallback_tile_ids != nullptr) {
      const int tile_slot =
          fallback_tile_ids[physical_block * total_tiles + tile_idx];
      if (tile_slot >= 0) {
        return byte_v2_load_raw_bits_from_tile_pool(
            fallback_pool, tile_slot, row, dim_in_tile);
      }
    }
    return 0;
  }

  const uint16_t compressed_bits = byte_v2_load_compressed_bits(
      page, is_value, row, kv_head, dim, k_dim_tiles, v_dim_tiles);
  return byte_v2_overlay_outlier_bits(compressed_bits, outlier_arena,
                                      outlier_tile_bitmap, outlier_tile_meta,
                                      physical_block, total_tiles,
                                      bitmap_words, tile_idx, elem);
}

#ifndef USE_ROCM
__device__ __forceinline__ void
byte_v2_decode_k_transposed_tile_to_shared_no_fallback(
    const uint8_t* __restrict__ page, __nv_bfloat16* __restrict__ k_shared,
    const int tid, const int kv_head, const int dim_tile,
    const int k_dim_tiles, const int v_dim_tiles, const int valid_rows,
    const bool use_aligned_u16_payload_load = false) {
  const int tile_start =
      byte_v2_tile_start(false, kv_head, dim_tile, k_dim_tiles, v_dim_tiles);
  const int base = static_cast<int>(page[tile_start]);
  const int d0 = dim_tile * kByteV2TileSize;
  const uint8_t* __restrict__ low_ptr = page + tile_start + 2;
  const uint8_t* __restrict__ packed_ptr =
      low_ptr + kByteV2TileElems;

  for (int pair_idx = tid; pair_idx < kByteV2PackedTileElems;
       pair_idx += blockDim.x) {
    const int elem0 = pair_idx * 2;
    const int dim_in_tile = elem0 / kByteV2TileSize;
    const int row0 = elem0 % kByteV2TileSize;
    const int shared_offset = (d0 + dim_in_tile) * kByteV2TileSize + row0;
    const uint16_t low_pair = byte_v2_load_payload_low_pair(
        low_ptr + elem0, use_aligned_u16_payload_load);
    const int packed = static_cast<int>(packed_ptr[pair_idx]);

    uint16_t bits0 = 0;
    uint16_t bits1 = 0;
    if (row0 < valid_rows) {
      bits0 = byte_v2_make_bf16_bits_from_fast_code(
          base, static_cast<int>(low_pair & 0xff), packed & 0x0f);
    }
    if (row0 + 1 < valid_rows) {
      bits1 = byte_v2_make_bf16_bits_from_fast_code(
          base, static_cast<int>(low_pair >> 8), (packed >> 4) & 0x0f);
    }
    k_shared[shared_offset] = byte_v2_bf16_bits_to_wmma(bits0);
    k_shared[shared_offset + 1] = byte_v2_bf16_bits_to_wmma(bits1);
  }
}

__device__ __forceinline__ void
byte_v2_decode_k_transposed_tile_to_shared_no_fallback_full_rows(
    const uint8_t* __restrict__ page, __nv_bfloat16* __restrict__ k_shared,
    const int tid, const int kv_head, const int dim_tile,
    const int k_dim_tiles, const int v_dim_tiles,
    const bool use_aligned_u16_payload_load = false) {
  const int tile_start =
      byte_v2_tile_start(false, kv_head, dim_tile, k_dim_tiles, v_dim_tiles);
  const int base = static_cast<int>(page[tile_start]);
  const int d0 = dim_tile * kByteV2TileSize;
  const uint8_t* __restrict__ low_ptr = page + tile_start + 2;
  const uint8_t* __restrict__ packed_ptr =
      low_ptr + kByteV2TileElems;

  for (int pair_idx = tid; pair_idx < kByteV2PackedTileElems;
       pair_idx += blockDim.x) {
    const int elem0 = pair_idx * 2;
    const int dim_in_tile = elem0 / kByteV2TileSize;
    const int row0 = elem0 % kByteV2TileSize;
    const int shared_offset = (d0 + dim_in_tile) * kByteV2TileSize + row0;
    const uint16_t low_pair = byte_v2_load_payload_low_pair(
        low_ptr + elem0, use_aligned_u16_payload_load);
    const int packed = static_cast<int>(packed_ptr[pair_idx]);

    const uint16_t bits0 = byte_v2_make_bf16_bits_from_fast_code(
        base, static_cast<int>(low_pair & 0xff), packed & 0x0f);
    const uint16_t bits1 = byte_v2_make_bf16_bits_from_fast_code(
        base, static_cast<int>(low_pair >> 8), (packed >> 4) & 0x0f);
    k_shared[shared_offset] = byte_v2_bf16_bits_to_wmma(bits0);
    k_shared[shared_offset + 1] = byte_v2_bf16_bits_to_wmma(bits1);
  }
}

__device__ __forceinline__ void
byte_v2_decode_k_transposed_tile_to_shared_no_fallback_v3(
    const uint8_t* __restrict__ page, __nv_bfloat16* __restrict__ k_shared,
    const int tid, const int kv_head, const int dim_tile,
    const int num_kv_heads, const int k_dim_tiles, const int v_dim_tiles,
    const int valid_rows, const bool use_v3_warp_stripe_load = false) {
  const int meta_start = byte_v2_v3_kv_head_meta_start(kv_head);
  const int base = static_cast<int>(page[meta_start + dim_tile]);
  const int tile_start = byte_v2_v3_tile_payload_start(
      false, kv_head, dim_tile, num_kv_heads, k_dim_tiles, v_dim_tiles);
  const int d0 = dim_tile * kByteV2TileSize;

  for (int pair_idx = tid; pair_idx < kByteV2PackedTileElems;
       pair_idx += blockDim.x) {
    const int elem0 = pair_idx * 2;
    const int dim_in_tile = elem0 / kByteV2TileSize;
    const int row0 = elem0 % kByteV2TileSize;
    const int shared_offset = (d0 + dim_in_tile) * kByteV2TileSize + row0;
    const int stripe = pair_idx / kByteV2TilePayloadStripePairsV3;
    const int lane = pair_idx % kByteV2TilePayloadStripePairsV3;
    const uint8_t* __restrict__ stripe_ptr =
        page + tile_start + stripe * kByteV2TilePayloadStripeBytesV3;
    int low0 = 0;
    int low1 = 0;
    int packed = 0;
    byte_v2_load_v3_stripe_pair(stripe_ptr, lane, use_v3_warp_stripe_load,
                                low0, low1, packed);

    uint16_t bits0 = 0;
    uint16_t bits1 = 0;
    if (row0 < valid_rows) {
      bits0 = byte_v2_make_bf16_bits_from_fast_code(base, low0, packed & 0x0f);
    }
    if (row0 + 1 < valid_rows) {
      bits1 = byte_v2_make_bf16_bits_from_fast_code(
          base, low1, (packed >> 4) & 0x0f);
    }
    k_shared[shared_offset] = byte_v2_bf16_bits_to_wmma(bits0);
    k_shared[shared_offset + 1] = byte_v2_bf16_bits_to_wmma(bits1);
  }
}

__device__ __forceinline__ void
byte_v2_decode_v_rowmajor_tile_to_shared_no_fallback(
    const uint8_t* __restrict__ page, __nv_bfloat16* __restrict__ v_shared,
    const int tid, const int kv_head, const int dim_tile,
    const int k_dim_tiles, const int v_dim_tiles, const int valid_rows,
    const int head_size_v, const bool use_aligned_u16_payload_load = false) {
  const int tile_start =
      byte_v2_tile_start(true, kv_head, dim_tile, k_dim_tiles, v_dim_tiles);
  const int base = static_cast<int>(page[tile_start]);
  const int d0 = dim_tile * kByteV2TileSize;
  const uint8_t* __restrict__ low_ptr = page + tile_start + 2;
  const uint8_t* __restrict__ packed_ptr =
      low_ptr + kByteV2TileElems;

  for (int pair_idx = tid; pair_idx < kByteV2PackedTileElems;
       pair_idx += blockDim.x) {
    const int elem0 = pair_idx * 2;
    const int row = elem0 / kByteV2TileSize;
    const int dim_in_tile = elem0 % kByteV2TileSize;
    const int shared_offset = row * head_size_v + d0 + dim_in_tile;
    const uint16_t low_pair = byte_v2_load_payload_low_pair(
        low_ptr + elem0, use_aligned_u16_payload_load);
    const int packed = static_cast<int>(packed_ptr[pair_idx]);

    uint16_t bits0 = 0;
    uint16_t bits1 = 0;
    if (row < valid_rows) {
      bits0 = byte_v2_make_bf16_bits_from_fast_code(
          base, static_cast<int>(low_pair & 0xff), packed & 0x0f);
      bits1 = byte_v2_make_bf16_bits_from_fast_code(
          base, static_cast<int>(low_pair >> 8), (packed >> 4) & 0x0f);
    }
    v_shared[shared_offset] = byte_v2_bf16_bits_to_wmma(bits0);
    v_shared[shared_offset + 1] = byte_v2_bf16_bits_to_wmma(bits1);
  }
}

__device__ __forceinline__ void
byte_v2_decode_v_rowmajor_tile_to_shared_no_fallback_v3(
    const uint8_t* __restrict__ page, __nv_bfloat16* __restrict__ v_shared,
    const int tid, const int kv_head, const int dim_tile,
    const int num_kv_heads, const int k_dim_tiles, const int v_dim_tiles,
    const int valid_rows, const int head_size_v,
    const bool use_v3_warp_stripe_load = false) {
  const int meta_start = byte_v2_v3_kv_head_meta_start(kv_head);
  const int base = static_cast<int>(page[meta_start + 8 + dim_tile]);
  const int tile_start = byte_v2_v3_tile_payload_start(
      true, kv_head, dim_tile, num_kv_heads, k_dim_tiles, v_dim_tiles);
  const int d0 = dim_tile * kByteV2TileSize;

  for (int pair_idx = tid; pair_idx < kByteV2PackedTileElems;
       pair_idx += blockDim.x) {
    const int elem0 = pair_idx * 2;
    const int row = elem0 / kByteV2TileSize;
    const int dim_in_tile = elem0 % kByteV2TileSize;
    const int shared_offset = row * head_size_v + d0 + dim_in_tile;
    const int stripe = pair_idx / kByteV2TilePayloadStripePairsV3;
    const int lane = pair_idx % kByteV2TilePayloadStripePairsV3;
    const uint8_t* __restrict__ stripe_ptr =
        page + tile_start + stripe * kByteV2TilePayloadStripeBytesV3;
    int low0 = 0;
    int low1 = 0;
    int packed = 0;
    byte_v2_load_v3_stripe_pair(stripe_ptr, lane, use_v3_warp_stripe_load,
                                low0, low1, packed);

    uint16_t bits0 = 0;
    uint16_t bits1 = 0;
    if (row < valid_rows) {
      bits0 = byte_v2_make_bf16_bits_from_fast_code(base, low0, packed & 0x0f);
      bits1 = byte_v2_make_bf16_bits_from_fast_code(
          base, low1, (packed >> 4) & 0x0f);
    }
    v_shared[shared_offset] = byte_v2_bf16_bits_to_wmma(bits0);
    v_shared[shared_offset + 1] = byte_v2_bf16_bits_to_wmma(bits1);
  }
}

__device__ __forceinline__ void
byte_v2_decode_k_transposed_staged_tile_to_shared_v3(
    const uint8_t* __restrict__ staged_tile,
    __nv_bfloat16* __restrict__ k_shared, const int tid, const int base,
    const int dim_tile, const int valid_rows) {
  const int d0 = dim_tile * kByteV2TileSize;
  for (int pair_idx = tid; pair_idx < kByteV2PackedTileElems;
       pair_idx += blockDim.x) {
    const int elem0 = pair_idx * 2;
    const int dim_in_tile = elem0 / kByteV2TileSize;
    const int row0 = elem0 % kByteV2TileSize;
    const int shared_offset = (d0 + dim_in_tile) * kByteV2TileSize + row0;
    const int stripe = pair_idx / kByteV2TilePayloadStripePairsV3;
    const int lane = pair_idx % kByteV2TilePayloadStripePairsV3;
    const uint8_t* __restrict__ stripe_ptr =
        staged_tile + stripe * kByteV2TilePayloadStripeBytesV3;
    const int low0 = static_cast<int>(stripe_ptr[lane]);
    const int low1 = static_cast<int>(stripe_ptr[32 + lane]);
    const int packed = static_cast<int>(stripe_ptr[64 + lane]);

    uint16_t bits0 = 0;
    uint16_t bits1 = 0;
    if (row0 < valid_rows) {
      bits0 = byte_v2_make_bf16_bits_from_fast_code(base, low0, packed & 0x0f);
    }
    if (row0 + 1 < valid_rows) {
      bits1 = byte_v2_make_bf16_bits_from_fast_code(
          base, low1, (packed >> 4) & 0x0f);
    }
    k_shared[shared_offset] = byte_v2_bf16_bits_to_wmma(bits0);
    k_shared[shared_offset + 1] = byte_v2_bf16_bits_to_wmma(bits1);
  }
}

__device__ __forceinline__ void
byte_v2_decode_v_rowmajor_staged_tile_to_shared_v3(
    const uint8_t* __restrict__ staged_tile,
    __nv_bfloat16* __restrict__ v_shared, const int tid, const int base,
    const int dim_tile, const int valid_rows, const int head_size_v) {
  const int d0 = dim_tile * kByteV2TileSize;
  for (int pair_idx = tid; pair_idx < kByteV2PackedTileElems;
       pair_idx += blockDim.x) {
    const int elem0 = pair_idx * 2;
    const int row = elem0 / kByteV2TileSize;
    const int dim_in_tile = elem0 % kByteV2TileSize;
    const int shared_offset = row * head_size_v + d0 + dim_in_tile;
    const int stripe = pair_idx / kByteV2TilePayloadStripePairsV3;
    const int lane = pair_idx % kByteV2TilePayloadStripePairsV3;
    const uint8_t* __restrict__ stripe_ptr =
        staged_tile + stripe * kByteV2TilePayloadStripeBytesV3;
    const int low0 = static_cast<int>(stripe_ptr[lane]);
    const int low1 = static_cast<int>(stripe_ptr[32 + lane]);
    const int packed = static_cast<int>(stripe_ptr[64 + lane]);

    uint16_t bits0 = 0;
    uint16_t bits1 = 0;
    if (row < valid_rows) {
      bits0 = byte_v2_make_bf16_bits_from_fast_code(base, low0, packed & 0x0f);
      bits1 = byte_v2_make_bf16_bits_from_fast_code(
          base, low1, (packed >> 4) & 0x0f);
    }
    v_shared[shared_offset] = byte_v2_bf16_bits_to_wmma(bits0);
    v_shared[shared_offset + 1] = byte_v2_bf16_bits_to_wmma(bits1);
  }
}

__device__ __forceinline__ void
byte_v2_decode_k_tiles_to_shared_v3_cp_async(
    const uint8_t* __restrict__ page, uint8_t* __restrict__ stage,
    __nv_bfloat16* __restrict__ k_shared, const int tid, const int kv_head,
    const int num_kv_heads, const int k_dim_tiles, const int v_dim_tiles,
    const int valid_rows) {
  if (k_dim_tiles <= 0) {
    return;
  }
  const int meta_start = byte_v2_v3_kv_head_meta_start(kv_head);
  int current_buffer = 0;
  byte_v2_cp_async_copy_v3_tile_payload(
      stage, page, false, tid, kv_head, 0, num_kv_heads, k_dim_tiles,
      v_dim_tiles);
  byte_v2_cp_async_commit_group();
  byte_v2_cp_async_wait_all();
  __syncthreads();

  for (int dim_tile = 0; dim_tile < k_dim_tiles; ++dim_tile) {
    const int next_dim_tile = dim_tile + 1;
    const int next_buffer = current_buffer ^ 1;
    if (next_dim_tile < k_dim_tiles) {
      byte_v2_cp_async_copy_v3_tile_payload(
          stage + next_buffer * kByteV2TilePayloadBytesV3, page, false, tid,
          kv_head, next_dim_tile, num_kv_heads, k_dim_tiles, v_dim_tiles);
      byte_v2_cp_async_commit_group();
    }
    const int base = static_cast<int>(page[meta_start + dim_tile]);
    byte_v2_decode_k_transposed_staged_tile_to_shared_v3(
        stage + current_buffer * kByteV2TilePayloadBytesV3, k_shared, tid,
        base, dim_tile, valid_rows);
    if (next_dim_tile < k_dim_tiles) {
      byte_v2_cp_async_wait_all();
      __syncthreads();
    }
    current_buffer = next_buffer;
  }
  __syncthreads();
}

__device__ __forceinline__ void
byte_v2_decode_v_tiles_to_shared_v3_cp_async(
    const uint8_t* __restrict__ page, uint8_t* __restrict__ stage,
    __nv_bfloat16* __restrict__ v_shared, const int tid, const int kv_head,
    const int num_kv_heads, const int k_dim_tiles, const int v_dim_tiles,
    const int valid_rows, const int head_size_v) {
  if (v_dim_tiles <= 0) {
    return;
  }
  const int meta_start = byte_v2_v3_kv_head_meta_start(kv_head);
  int current_buffer = 0;
  byte_v2_cp_async_copy_v3_tile_payload(
      stage, page, true, tid, kv_head, 0, num_kv_heads, k_dim_tiles,
      v_dim_tiles);
  byte_v2_cp_async_commit_group();
  byte_v2_cp_async_wait_all();
  __syncthreads();

  for (int dim_tile = 0; dim_tile < v_dim_tiles; ++dim_tile) {
    const int next_dim_tile = dim_tile + 1;
    const int next_buffer = current_buffer ^ 1;
    if (next_dim_tile < v_dim_tiles) {
      byte_v2_cp_async_copy_v3_tile_payload(
          stage + next_buffer * kByteV2TilePayloadBytesV3, page, true, tid,
          kv_head, next_dim_tile, num_kv_heads, k_dim_tiles, v_dim_tiles);
      byte_v2_cp_async_commit_group();
    }
    const int base = static_cast<int>(page[meta_start + 8 + dim_tile]);
    byte_v2_decode_v_rowmajor_staged_tile_to_shared_v3(
        stage + current_buffer * kByteV2TilePayloadBytesV3, v_shared, tid,
        base, dim_tile, valid_rows, head_size_v);
    if (next_dim_tile < v_dim_tiles) {
      byte_v2_cp_async_wait_all();
      __syncthreads();
    }
    current_buffer = next_buffer;
  }
  __syncthreads();
}

template <int MacroPages, typename block_table_t>
__device__ __forceinline__ void byte_v2_v4_load_macro_descriptor(
    const uint8_t* __restrict__ kv_cache,
    const block_table_t* __restrict__ block_table, const int req_idx,
    const int seq_len, const int macro_begin_block, const int end_block,
    const int page_size_bytes, const int64_t block_table_stride0,
    const int64_t block_table_stride1,
    int* __restrict__ macro_physical_blocks,
    int* __restrict__ macro_valid_rows,
    int* __restrict__ macro_compressed_v3) {
  const int tid = threadIdx.x;
  if (tid < MacroPages) {
    const int logical_block = macro_begin_block + tid;
    int physical_block = 0;
    int valid_rows = 0;
    int compressed_v3 = 0;
    if (logical_block < end_block) {
      const int token_base = logical_block * kByteV2TileSize;
      const int remaining_rows = seq_len - token_base;
      if (remaining_rows > 0) {
        physical_block = static_cast<int>(
            block_table[req_idx * block_table_stride0 +
                        logical_block * block_table_stride1]);
        const uint8_t* page = kv_cache + physical_block * page_size_bytes;
        const int page_valid_rows =
            static_cast<int>(page[kByteV2PageValidRowsOffset]);
        valid_rows = min(min(kByteV2TileSize, remaining_rows),
                         page_valid_rows);
        compressed_v3 =
            valid_rows > 0 &&
            page[kByteV2PageStatusOffset] == kByteV2PageStatusCompressed &&
            page[kByteV2PageLayoutVersionOffset] ==
                kByteV2PayloadLayoutVersionV3;
      }
    }
    macro_physical_blocks[tid] = physical_block;
    macro_valid_rows[tid] = valid_rows;
    macro_compressed_v3[tid] = compressed_v3;
  }
  __syncthreads();
}

__device__ __forceinline__ void
byte_v2_decode_k_transposed_tile_to_shared_no_fallback_layout(
    const uint8_t* __restrict__ page, __nv_bfloat16* __restrict__ k_shared,
    const int tid, const int kv_head, const int dim_tile,
    const int num_kv_heads, const int k_dim_tiles, const int v_dim_tiles,
    const int valid_rows, const bool use_aligned_u16_payload_load = false,
    const bool use_v3_warp_stripe_load = false) {
  if (byte_v2_page_has_v3_payload(page)) {
    byte_v2_decode_k_transposed_tile_to_shared_no_fallback_v3(
        page, k_shared, tid, kv_head, dim_tile, num_kv_heads, k_dim_tiles,
        v_dim_tiles, valid_rows, use_v3_warp_stripe_load);
    return;
  }
  byte_v2_decode_k_transposed_tile_to_shared_no_fallback(
      page, k_shared, tid, kv_head, dim_tile, k_dim_tiles, v_dim_tiles,
      valid_rows, use_aligned_u16_payload_load);
}

__device__ __forceinline__ void
byte_v2_decode_v_rowmajor_tile_to_shared_no_fallback_layout(
    const uint8_t* __restrict__ page, __nv_bfloat16* __restrict__ v_shared,
    const int tid, const int kv_head, const int dim_tile,
    const int num_kv_heads, const int k_dim_tiles, const int v_dim_tiles,
    const int valid_rows, const int head_size_v,
    const bool use_aligned_u16_payload_load = false,
    const bool use_v3_warp_stripe_load = false) {
  if (byte_v2_page_has_v3_payload(page)) {
    byte_v2_decode_v_rowmajor_tile_to_shared_no_fallback_v3(
        page, v_shared, tid, kv_head, dim_tile, num_kv_heads, k_dim_tiles,
        v_dim_tiles, valid_rows, head_size_v, use_v3_warp_stripe_load);
    return;
  }
  byte_v2_decode_v_rowmajor_tile_to_shared_no_fallback(
      page, v_shared, tid, kv_head, dim_tile, k_dim_tiles, v_dim_tiles,
      valid_rows, head_size_v, use_aligned_u16_payload_load);
}

__device__ __forceinline__ void
byte_v2_decode_v_rowmajor_tile_to_shared_no_fallback_full_rows(
    const uint8_t* __restrict__ page, __nv_bfloat16* __restrict__ v_shared,
    const int tid, const int kv_head, const int dim_tile,
    const int k_dim_tiles, const int v_dim_tiles, const int head_size_v,
    const bool use_aligned_u16_payload_load = false) {
  const int tile_start =
      byte_v2_tile_start(true, kv_head, dim_tile, k_dim_tiles, v_dim_tiles);
  const int base = static_cast<int>(page[tile_start]);
  const int d0 = dim_tile * kByteV2TileSize;
  const uint8_t* __restrict__ low_ptr = page + tile_start + 2;
  const uint8_t* __restrict__ packed_ptr =
      low_ptr + kByteV2TileElems;

  for (int pair_idx = tid; pair_idx < kByteV2PackedTileElems;
       pair_idx += blockDim.x) {
    const int elem0 = pair_idx * 2;
    const int row = elem0 / kByteV2TileSize;
    const int dim_in_tile = elem0 % kByteV2TileSize;
    const int shared_offset = row * head_size_v + d0 + dim_in_tile;
    const uint16_t low_pair = byte_v2_load_payload_low_pair(
        low_ptr + elem0, use_aligned_u16_payload_load);
    const int packed = static_cast<int>(packed_ptr[pair_idx]);

    const uint16_t bits0 = byte_v2_make_bf16_bits_from_fast_code(
        base, static_cast<int>(low_pair & 0xff), packed & 0x0f);
    const uint16_t bits1 = byte_v2_make_bf16_bits_from_fast_code(
        base, static_cast<int>(low_pair >> 8), (packed >> 4) & 0x0f);
    v_shared[shared_offset] = byte_v2_bf16_bits_to_wmma(bits0);
    v_shared[shared_offset + 1] = byte_v2_bf16_bits_to_wmma(bits1);
  }
}

__device__ __forceinline__ void
byte_v2_decode_k_transposed_tile_to_shared_tile_fallback(
    const uint8_t* __restrict__ fallback_pool,
    const int32_t* __restrict__ fallback_tile_ids,
    __nv_bfloat16* __restrict__ k_shared, const int tid,
    const int physical_block, const int kv_head, const int dim_tile,
    const int k_dim_tiles, const int v_dim_tiles, const int valid_rows,
    const int total_tiles) {
  const int tile_idx = byte_v2_tile_index(false, kv_head, dim_tile,
                                          k_dim_tiles, v_dim_tiles);
  const int tile_slot =
      (fallback_pool != nullptr && fallback_tile_ids != nullptr)
          ? fallback_tile_ids[physical_block * total_tiles + tile_idx]
          : -1;
  const int d0 = dim_tile * kByteV2TileSize;

  for (int idx = tid; idx < kByteV2TileElems; idx += blockDim.x) {
    const int dim_in_tile = idx / kByteV2TileSize;
    const int row = idx % kByteV2TileSize;
    uint16_t bits = 0;
    if (row < valid_rows && tile_slot >= 0) {
      bits = byte_v2_load_raw_bits_from_tile_pool(fallback_pool, tile_slot,
                                                  row, dim_in_tile);
    }
    k_shared[(d0 + dim_in_tile) * kByteV2TileSize + row] =
        byte_v2_bf16_bits_to_wmma(bits);
  }
}

__device__ __forceinline__ void
byte_v2_decode_v_rowmajor_tile_to_shared_tile_fallback(
    const uint8_t* __restrict__ fallback_pool,
    const int32_t* __restrict__ fallback_tile_ids,
    __nv_bfloat16* __restrict__ v_shared, const int tid,
    const int physical_block, const int kv_head, const int dim_tile,
    const int k_dim_tiles, const int v_dim_tiles, const int valid_rows,
    const int head_size_v, const int total_tiles) {
  const int tile_idx = byte_v2_tile_index(true, kv_head, dim_tile, k_dim_tiles,
                                          v_dim_tiles);
  const int tile_slot =
      (fallback_pool != nullptr && fallback_tile_ids != nullptr)
          ? fallback_tile_ids[physical_block * total_tiles + tile_idx]
          : -1;
  const int d0 = dim_tile * kByteV2TileSize;

  for (int idx = tid; idx < kByteV2TileElems; idx += blockDim.x) {
    const int row = idx / kByteV2TileSize;
    const int dim_in_tile = idx % kByteV2TileSize;
    uint16_t bits = 0;
    if (row < valid_rows && tile_slot >= 0) {
      bits = byte_v2_load_raw_bits_from_tile_pool(fallback_pool, tile_slot,
                                                  row, dim_in_tile);
    }
    v_shared[row * head_size_v + d0 + dim_in_tile] =
        byte_v2_bf16_bits_to_wmma(bits);
  }
}

__device__ __forceinline__ void byte_v2_overlay_k_tile_outliers_to_shared(
    const int32_t* __restrict__ outlier_arena,
    const int32_t* __restrict__ outlier_tile_bitmap,
    const int32_t* __restrict__ outlier_tile_meta,
    __nv_bfloat16* __restrict__ k_shared, const int tid,
    const int physical_block, const int kv_head, const int dim_tile,
    const int k_dim_tiles, const int v_dim_tiles, const int valid_rows,
    const int total_tiles, const int bitmap_words) {
  if (outlier_arena == nullptr || outlier_tile_meta == nullptr) {
    return;
  }
  const int tile_idx =
      byte_v2_tile_index(false, kv_head, dim_tile, k_dim_tiles, v_dim_tiles);
  if (!byte_v2_may_have_tile_outliers(outlier_tile_bitmap, physical_block,
                                      bitmap_words, tile_idx)) {
    return;
  }
  const int32_t meta =
      outlier_tile_meta[physical_block * total_tiles + tile_idx];
  if (meta < 0) {
    return;
  }
  const int count = byte_v2_outlier_meta_count(meta);
  const int offset = byte_v2_outlier_meta_offset(meta);
  const int d0 = dim_tile * kByteV2TileSize;
  for (int i = tid; i < count; i += blockDim.x) {
    const int32_t entry = outlier_arena[offset + i];
    const int elem = byte_v2_outlier_entry_elem(entry);
    const int dim_in_tile = elem / kByteV2TileSize;
    const int row = elem % kByteV2TileSize;
    if (row < valid_rows) {
      k_shared[(d0 + dim_in_tile) * kByteV2TileSize + row] =
          byte_v2_bf16_bits_to_wmma(byte_v2_outlier_entry_bits(entry));
    }
  }
}

__device__ __forceinline__ void byte_v2_overlay_v_tile_outliers_to_shared(
    const int32_t* __restrict__ outlier_arena,
    const int32_t* __restrict__ outlier_tile_bitmap,
    const int32_t* __restrict__ outlier_tile_meta,
    __nv_bfloat16* __restrict__ v_shared, const int tid,
    const int physical_block, const int kv_head, const int dim_tile,
    const int k_dim_tiles, const int v_dim_tiles, const int valid_rows,
    const int head_size_v, const int total_tiles, const int bitmap_words) {
  if (outlier_arena == nullptr || outlier_tile_meta == nullptr) {
    return;
  }
  const int tile_idx =
      byte_v2_tile_index(true, kv_head, dim_tile, k_dim_tiles, v_dim_tiles);
  if (!byte_v2_may_have_tile_outliers(outlier_tile_bitmap, physical_block,
                                      bitmap_words, tile_idx)) {
    return;
  }
  const int32_t meta =
      outlier_tile_meta[physical_block * total_tiles + tile_idx];
  if (meta < 0) {
    return;
  }
  const int count = byte_v2_outlier_meta_count(meta);
  const int offset = byte_v2_outlier_meta_offset(meta);
  const int d0 = dim_tile * kByteV2TileSize;
  for (int i = tid; i < count; i += blockDim.x) {
    const int32_t entry = outlier_arena[offset + i];
    const int elem = byte_v2_outlier_entry_elem(entry);
    const int row = elem / kByteV2TileSize;
    const int dim_in_tile = elem % kByteV2TileSize;
    if (row < valid_rows) {
      v_shared[row * head_size_v + d0 + dim_in_tile] =
          byte_v2_bf16_bits_to_wmma(byte_v2_outlier_entry_bits(entry));
    }
  }
}

__device__ __forceinline__ void byte_v2_decode_k_raw_block_tile_to_shared(
    const uint8_t* __restrict__ raw_block,
    __nv_bfloat16* __restrict__ k_shared, const int tid, const int kv_head,
    const int dim_tile, const int num_kv_heads, const int head_size,
    const int head_size_v, const int valid_rows) {
  const int d0 = dim_tile * kByteV2TileSize;
  for (int idx = tid; idx < kByteV2TileElems; idx += blockDim.x) {
    const int dim_in_tile = idx / kByteV2TileSize;
    const int row = idx % kByteV2TileSize;
    uint16_t bits = 0;
    if (raw_block != nullptr && row < valid_rows) {
      bits = byte_v2_load_raw_bits_from_block(
          raw_block, false, row, kv_head, d0 + dim_in_tile, num_kv_heads,
          head_size, head_size_v);
    }
    k_shared[(d0 + dim_in_tile) * kByteV2TileSize + row] =
        byte_v2_bf16_bits_to_wmma(bits);
  }
}

__device__ __forceinline__ void byte_v2_decode_v_raw_block_tile_to_shared(
    const uint8_t* __restrict__ raw_block,
    __nv_bfloat16* __restrict__ v_shared, const int tid, const int kv_head,
    const int dim_tile, const int num_kv_heads, const int head_size,
    const int head_size_v, const int valid_rows) {
  const int d0 = dim_tile * kByteV2TileSize;
  for (int idx = tid; idx < kByteV2TileElems; idx += blockDim.x) {
    const int row = idx / kByteV2TileSize;
    const int dim_in_tile = idx % kByteV2TileSize;
    uint16_t bits = 0;
    if (raw_block != nullptr && row < valid_rows) {
      bits = byte_v2_load_raw_bits_from_block(
          raw_block, true, row, kv_head, d0 + dim_in_tile, num_kv_heads,
          head_size, head_size_v);
    }
    v_shared[row * head_size_v + d0 + dim_in_tile] =
        byte_v2_bf16_bits_to_wmma(bits);
  }
}

#endif

__device__ __forceinline__ uint16_t byte_v2_load_kv_bits(
    const uint8_t* __restrict__ page,
    const uint8_t* __restrict__ fallback_pool,
    const int32_t* __restrict__ fallback_block_ids,
    const int32_t* __restrict__ fallback_tile_ids,
    const int32_t* __restrict__ outlier_arena,
    const int32_t* __restrict__ outlier_tile_bitmap,
    const int32_t* __restrict__ outlier_tile_meta, const int physical_block,
    const int raw_block_bytes, const bool is_value, const int row,
    const int kv_head, const int dim, const int num_kv_heads,
    const int head_size, const int head_size_v) {
  const uint8_t status = page[kByteV2PageStatusOffset];
  if (status == kByteV2PageStatusRawFallback) {
    if (fallback_pool != nullptr && fallback_block_ids != nullptr) {
      const int fallback_slot = fallback_block_ids[physical_block];
      if (fallback_slot >= 0) {
        return byte_v2_load_raw_bits_from_block(
            fallback_pool + fallback_slot * raw_block_bytes, is_value, row,
            kv_head, dim, num_kv_heads, head_size, head_size_v);
      }
    }
    return byte_v2_load_raw_bits_from_page(page, is_value, row, kv_head, dim,
                                           num_kv_heads, head_size,
                                           head_size_v);
  }
  if (status == kByteV2PageStatusCompressed) {
    const int k_dim_tiles = head_size / kByteV2TileSize;
    const int v_dim_tiles = head_size_v / kByteV2TileSize;
    const int total_tiles =
        byte_v2_total_tiles(num_kv_heads, head_size, head_size_v);
    const int bitmap_words = byte_v2_outlier_tile_bitmap_words(total_tiles);
    return byte_v2_load_compressed_bits(
        page, is_value, row, kv_head, dim, k_dim_tiles, v_dim_tiles,
        fallback_pool, fallback_tile_ids, outlier_arena, outlier_tile_bitmap,
        outlier_tile_meta, physical_block, total_tiles, bitmap_words);
  }
  return 0;
}

__global__ void byte_v2_decompress_cache_to_bf16_kernel(
    const uint8_t* __restrict__ kv_cache,
    uint16_t* __restrict__ key_cache,
    uint16_t* __restrict__ value_cache,
    const uint8_t* __restrict__ fallback_pool,
    const int32_t* __restrict__ fallback_block_ids,
    const int32_t* __restrict__ fallback_tile_ids,
    const int32_t* __restrict__ outlier_arena,
    const int32_t* __restrict__ outlier_tile_bitmap,
    const int32_t* __restrict__ outlier_tile_meta, const int num_blocks,
    const int page_size_bytes, const int raw_block_bytes,
    const int num_kv_heads, const int head_size, const int head_size_v) {
  const int physical_block = static_cast<int>(blockIdx.x);
  const int kv_head = static_cast<int>(blockIdx.y);
  const bool is_value = blockIdx.z != 0;
  if (physical_block >= num_blocks || kv_head >= num_kv_heads) {
    return;
  }

  const uint8_t* __restrict__ page =
      kv_cache + static_cast<int64_t>(physical_block) * page_size_bytes;
  const int dim = is_value ? head_size_v : head_size;
  const int valid_rows =
      min(max(static_cast<int>(page[kByteV2PageValidRowsOffset]), 0),
          kByteV2TileSize);
  const int numel = kByteV2TileSize * dim;
  for (int idx = threadIdx.x; idx < numel; idx += blockDim.x) {
    const int row = idx / dim;
    const int dim_idx = idx - row * dim;
    uint16_t bits = 0;
    if (row < valid_rows) {
      bits = byte_v2_load_kv_bits(
          page, fallback_pool, fallback_block_ids, fallback_tile_ids,
          outlier_arena, outlier_tile_bitmap, outlier_tile_meta,
          physical_block, raw_block_bytes, is_value, row, kv_head, dim_idx,
          num_kv_heads, head_size, head_size_v);
    }
    const int64_t offset =
        ((static_cast<int64_t>(physical_block) * kByteV2TileSize + row) *
             num_kv_heads +
         kv_head) *
            dim +
        dim_idx;
    if (is_value) {
      value_cache[offset] = bits;
    } else {
      key_cache[offset] = bits;
    }
  }
}

__device__ int byte_v2_best_window_base_from_hist(
    const int* __restrict__ hist, int* __restrict__ covered_out) {
  int window = 0;
  for (int i = 0; i < 16; ++i) {
    window += hist[i];
  }
  int best = window;
  int best_start = 0;
  for (int start = 1; start <= 240; ++start) {
    window += hist[start + 15] - hist[start - 1];
    if (window > best) {
      best = window;
      best_start = start;
    }
  }
  *covered_out = best;
  return best_start;
}

__device__ int byte_v2_best_window_base_from_raw_tile(
    const uint8_t* __restrict__ raw_block, const bool is_value,
    const int kv_head, const int dim_tile, const int num_kv_heads,
    const int head_size, const int head_size_v, const int valid_rows,
    int* __restrict__ covered_out) {
  int hist[256];
#pragma unroll
  for (int i = 0; i < 256; ++i) {
    hist[i] = 0;
  }

  const int d0 = dim_tile * kByteV2TileSize;
  for (int row = 0; row < valid_rows; ++row) {
    for (int d = 0; d < kByteV2TileSize; ++d) {
      const uint16_t bits = byte_v2_load_raw_bits_from_block(
          raw_block, is_value, row, kv_head, d0 + d, num_kv_heads, head_size,
          head_size_v);
      const int exp = (bits >> 7) & 0xff;
      hist[exp] += 1;
    }
  }

  return byte_v2_best_window_base_from_hist(hist, covered_out);
}

__device__ __forceinline__ int byte_v2_clamp_exp_to_window(
    const int exp, const int base) {
  return min(max(exp, base), base + 15);
}

__device__ __forceinline__ bool byte_v2_exp_in_window(
    const int exp, const int base) {
  return exp >= base && exp <= base + 15;
}

__device__ __forceinline__ int32_t byte_v2_pack_outlier_meta(
    const int offset, const int count) {
  return (offset << kByteV2OutlierMetaCountBits) |
         (count & kByteV2OutlierMetaCountMask);
}

__device__ __forceinline__ int32_t byte_v2_pack_outlier_entry(
    const int elem, const uint16_t bits) {
  return (static_cast<int32_t>(bits) << 8) | (elem & 0xff);
}

__device__ bool byte_v2_raw_block_is_compressible(
    const uint8_t* __restrict__ raw_block, const int num_kv_heads,
    const int head_size, const int head_size_v, const int valid_rows,
    const int lossy_max_misses_per_tile) {
  const int k_dim_tiles = head_size / kByteV2TileSize;
  const int v_dim_tiles = head_size_v / kByteV2TileSize;
  const int required_covered = valid_rows * kByteV2TileSize;
  int covered = 0;
  for (int kv_head = 0; kv_head < num_kv_heads; ++kv_head) {
    for (int dim_tile = 0; dim_tile < k_dim_tiles; ++dim_tile) {
      byte_v2_best_window_base_from_raw_tile(
          raw_block, false, kv_head, dim_tile, num_kv_heads, head_size,
          head_size_v, valid_rows, &covered);
      if (required_covered - covered > lossy_max_misses_per_tile) {
        return false;
      }
    }
    for (int dim_tile = 0; dim_tile < v_dim_tiles; ++dim_tile) {
      byte_v2_best_window_base_from_raw_tile(
          raw_block, true, kv_head, dim_tile, num_kv_heads, head_size,
          head_size_v, valid_rows, &covered);
      if (required_covered - covered > lossy_max_misses_per_tile) {
        return false;
      }
    }
  }
  return true;
}

__device__ void byte_v2_store_compressed_tile(
    const uint8_t* __restrict__ raw_block, uint8_t* __restrict__ page,
    const bool is_value, const int kv_head, const int dim_tile,
    const int num_kv_heads, const int head_size, const int head_size_v,
    const int valid_rows, const int lossy_max_misses_per_tile) {
  int covered = 0;
  const int base = byte_v2_best_window_base_from_raw_tile(
      raw_block, is_value, kv_head, dim_tile, num_kv_heads, head_size,
      head_size_v, valid_rows, &covered);
  const int tile_start =
      byte_v2_tile_start(is_value, kv_head, dim_tile,
                         head_size / kByteV2TileSize,
                         head_size_v / kByteV2TileSize);
  page[tile_start] = static_cast<uint8_t>(base);
  page[tile_start + 1] = 0;

  const int d0 = dim_tile * kByteV2TileSize;
  for (int elem = 0; elem < kByteV2TileElems; ++elem) {
    const int row =
        is_value ? elem / kByteV2TileSize : elem % kByteV2TileSize;
    const int d =
        is_value ? elem % kByteV2TileSize : elem / kByteV2TileSize;
    const uint16_t bits =
        row < valid_rows ? byte_v2_load_raw_bits_from_block(
                               raw_block, is_value, row, kv_head, d0 + d,
                               num_kv_heads, head_size, head_size_v)
                         : 0;
    int low = 0;
    if (row < valid_rows) {
      const int exp = (bits >> 7) & 0xff;
      const int stored_exp =
          lossy_max_misses_per_tile > 0
              ? byte_v2_clamp_exp_to_window(exp, base)
              : exp;
      low = (bits & 0x7f) | ((stored_exp & 1) << 7);
    }
    page[tile_start + 2 + elem] = static_cast<uint8_t>(low);
  }
  for (int packed_idx = 0; packed_idx < kByteV2PackedTileElems;
       ++packed_idx) {
    uint8_t packed = 0;
    for (int lane = 0; lane < 2; ++lane) {
      const int elem = packed_idx * 2 + lane;
      const int row =
          is_value ? elem / kByteV2TileSize : elem % kByteV2TileSize;
      const int d =
          is_value ? elem % kByteV2TileSize : elem / kByteV2TileSize;
      if (row < valid_rows) {
        const uint16_t bits = byte_v2_load_raw_bits_from_block(
            raw_block, is_value, row, kv_head, d0 + d, num_kv_heads, head_size,
            head_size_v);
        const int exp = (bits >> 7) & 0xff;
        const int stored_exp =
            lossy_max_misses_per_tile > 0
                ? byte_v2_clamp_exp_to_window(exp, base)
                : exp;
        const int delta = stored_exp - base;
        const int sign = (bits >> 15) & 1;
        const int delta_hi = delta >> 1;
        const uint8_t code =
            static_cast<uint8_t>((sign << 3) | (delta_hi & 0x07));
        packed |= static_cast<uint8_t>(code << (lane * 4));
      }
    }
    page[tile_start + 2 + kByteV2TileElems + packed_idx] = packed;
  }
}

__device__ void byte_v2_init_v3_compressed_page_header(
    uint8_t* __restrict__ page, const int num_kv_heads,
    const int head_size, const int head_size_v) {
  const int k_dim_tiles = head_size / kByteV2TileSize;
  const int v_dim_tiles = head_size_v / kByteV2TileSize;
  const int meta_bytes = byte_v2_v3_kv_head_meta_region_bytes(num_kv_heads);
  const int k_payload_offset = byte_v2_v3_k_payload_offset(num_kv_heads);
  const int v_payload_offset =
      byte_v2_v3_v_payload_offset(num_kv_heads, k_dim_tiles);
  for (int i = 0; i < kByteV2PageHeaderBytesV3 + meta_bytes; ++i) {
    page[i] = 0;
  }
  page[kByteV2PageLayoutVersionOffset] = kByteV2PayloadLayoutVersionV3;
  page[3] = 0x05;  // has tile fallback metadata + compressed-only.
  byte_v2_store_u32(page + 8, static_cast<uint32_t>(k_payload_offset));
  byte_v2_store_u32(page + 12, static_cast<uint32_t>(k_payload_offset));
  byte_v2_store_u32(page + 16, static_cast<uint32_t>(v_payload_offset));
  byte_v2_store_u32(page + 20,
                    static_cast<uint32_t>(kByteV2TilePayloadBytesV3));
  byte_v2_store_u32(page + 24,
                    static_cast<uint32_t>(kByteV2PageHeaderBytesV3));
  byte_v2_store_u32(page + 28, static_cast<uint32_t>(meta_bytes));
  (void)v_dim_tiles;
}

__device__ void byte_v2_store_compressed_tile_v3(
    const uint8_t* __restrict__ raw_block, uint8_t* __restrict__ page,
    const bool is_value, const int kv_head, const int dim_tile,
    const int num_kv_heads, const int head_size, const int head_size_v,
    const int valid_rows, const int lossy_max_misses_per_tile) {
  int covered = 0;
  const int base = byte_v2_best_window_base_from_raw_tile(
      raw_block, is_value, kv_head, dim_tile, num_kv_heads, head_size,
      head_size_v, valid_rows, &covered);
  const int k_dim_tiles = head_size / kByteV2TileSize;
  const int v_dim_tiles = head_size_v / kByteV2TileSize;
  const int meta_start = byte_v2_v3_kv_head_meta_start(kv_head);
  page[meta_start + (is_value ? 8 + dim_tile : dim_tile)] =
      static_cast<uint8_t>(base);
  const int tile_start = byte_v2_v3_tile_payload_start(
      is_value, kv_head, dim_tile, num_kv_heads, k_dim_tiles, v_dim_tiles);
  const int d0 = dim_tile * kByteV2TileSize;

  for (int pair_idx = 0; pair_idx < kByteV2PackedTileElems; ++pair_idx) {
    uint8_t low_values[2] = {0, 0};
    uint8_t packed = 0;
    for (int lane = 0; lane < 2; ++lane) {
      const int elem = pair_idx * 2 + lane;
      const int row =
          is_value ? elem / kByteV2TileSize : elem % kByteV2TileSize;
      const int dim_in_tile =
          is_value ? elem % kByteV2TileSize : elem / kByteV2TileSize;
      if (row < valid_rows) {
        const uint16_t bits = byte_v2_load_raw_bits_from_block(
            raw_block, is_value, row, kv_head, d0 + dim_in_tile,
            num_kv_heads, head_size, head_size_v);
        const int exp = (bits >> 7) & 0xff;
        const int stored_exp =
            lossy_max_misses_per_tile > 0
                ? byte_v2_clamp_exp_to_window(exp, base)
                : exp;
        low_values[lane] =
            static_cast<uint8_t>((bits & 0x7f) | ((stored_exp & 1) << 7));
        const int delta = stored_exp - base;
        const int sign = (bits >> 15) & 1;
        const int delta_hi = delta >> 1;
        const uint8_t code =
            static_cast<uint8_t>((sign << 3) | (delta_hi & 0x07));
        packed |= static_cast<uint8_t>(code << (lane * 4));
      }
    }
    const int stripe = pair_idx / kByteV2TilePayloadStripePairsV3;
    const int stripe_lane = pair_idx % kByteV2TilePayloadStripePairsV3;
    uint8_t* __restrict__ stripe_ptr =
        page + tile_start + stripe * kByteV2TilePayloadStripeBytesV3;
    stripe_ptr[stripe_lane] = low_values[0];
    stripe_ptr[32 + stripe_lane] = low_values[1];
    stripe_ptr[64 + stripe_lane] = packed;
  }
}

__device__ void byte_v2_store_compressed_block(
    const uint8_t* __restrict__ raw_block, uint8_t* __restrict__ page,
    const int num_kv_heads, const int head_size, const int head_size_v,
    const int valid_rows, const int lossy_max_misses_per_tile) {
  const int k_dim_tiles = head_size / kByteV2TileSize;
  const int v_dim_tiles = head_size_v / kByteV2TileSize;
  for (int kv_head = 0; kv_head < num_kv_heads; ++kv_head) {
    for (int dim_tile = 0; dim_tile < k_dim_tiles; ++dim_tile) {
      byte_v2_store_compressed_tile(raw_block, page, false, kv_head, dim_tile,
                                    num_kv_heads, head_size, head_size_v,
                                    valid_rows, lossy_max_misses_per_tile);
    }
    for (int dim_tile = 0; dim_tile < v_dim_tiles; ++dim_tile) {
      byte_v2_store_compressed_tile(raw_block, page, true, kv_head, dim_tile,
                                    num_kv_heads, head_size, head_size_v,
                                    valid_rows, lossy_max_misses_per_tile);
    }
  }
}

__device__ void byte_v2_store_compressed_block_v3(
    const uint8_t* __restrict__ raw_block, uint8_t* __restrict__ page,
    const int num_kv_heads, const int head_size, const int head_size_v,
    const int valid_rows, const int lossy_max_misses_per_tile) {
  byte_v2_init_v3_compressed_page_header(page, num_kv_heads, head_size,
                                         head_size_v);
  const int k_dim_tiles = head_size / kByteV2TileSize;
  const int v_dim_tiles = head_size_v / kByteV2TileSize;
  for (int kv_head = 0; kv_head < num_kv_heads; ++kv_head) {
    for (int dim_tile = 0; dim_tile < k_dim_tiles; ++dim_tile) {
      byte_v2_store_compressed_tile_v3(
          raw_block, page, false, kv_head, dim_tile, num_kv_heads, head_size,
          head_size_v, valid_rows, lossy_max_misses_per_tile);
    }
    for (int dim_tile = 0; dim_tile < v_dim_tiles; ++dim_tile) {
      byte_v2_store_compressed_tile_v3(
          raw_block, page, true, kv_head, dim_tile, num_kv_heads, head_size,
          head_size_v, valid_rows, lossy_max_misses_per_tile);
    }
  }
}

__device__ void byte_v2_store_raw_tile_from_block(
    const uint8_t* __restrict__ raw_block, uint8_t* __restrict__ fallback_pool,
    const int tile_slot, const bool is_value, const int kv_head,
    const int dim_tile, const int num_kv_heads, const int head_size,
    const int head_size_v, const int valid_rows) {
  const int d0 = dim_tile * kByteV2TileSize;
  for (int elem = 0; elem < kByteV2TileElems; ++elem) {
    const int row = elem / kByteV2TileSize;
    const int dim_in_tile = elem % kByteV2TileSize;
    const uint16_t bits =
        row < valid_rows ? byte_v2_load_raw_bits_from_block(
                               raw_block, is_value, row, kv_head,
                               d0 + dim_in_tile, num_kv_heads, head_size,
                               head_size_v)
                         : 0;
    byte_v2_store_raw_bits_to_tile_pool(fallback_pool, tile_slot, row,
                                        dim_in_tile, bits);
  }
}

__device__ bool byte_v2_store_compressed_or_tile_fallback_block(
    const uint8_t* __restrict__ raw_block, uint8_t* __restrict__ page,
    uint8_t* __restrict__ fallback_pool,
    int32_t* __restrict__ fallback_tile_ids,
    int32_t* __restrict__ fallback_tile_next_slot,
    int32_t* __restrict__ error, const int block_id, const int num_kv_heads,
    const int head_size, const int head_size_v, const int valid_rows,
    const int raw_block_bytes, const int fallback_pool_blocks,
    const int lossy_max_misses_per_tile) {
  const int k_dim_tiles = head_size / kByteV2TileSize;
  const int v_dim_tiles = head_size_v / kByteV2TileSize;
  const int tiles_per_head = k_dim_tiles + v_dim_tiles;
  const int total_tiles = num_kv_heads * tiles_per_head;
  const int required_covered = valid_rows * kByteV2TileSize;
  const int64_t tile_capacity =
      byte_v2_tile_fallback_capacity(fallback_pool_blocks, raw_block_bytes);

  for (int kv_head = 0; kv_head < num_kv_heads; ++kv_head) {
    for (int tile_in_head = 0; tile_in_head < tiles_per_head; ++tile_in_head) {
      const bool is_value = tile_in_head >= k_dim_tiles;
      const int dim_tile =
          is_value ? tile_in_head - k_dim_tiles : tile_in_head;
      const int tile_idx = byte_v2_tile_index(is_value, kv_head, dim_tile,
                                              k_dim_tiles, v_dim_tiles);
      int covered = 0;
      byte_v2_best_window_base_from_raw_tile(
          raw_block, is_value, kv_head, dim_tile, num_kv_heads, head_size,
          head_size_v, valid_rows, &covered);
      const bool use_tile_fallback =
          required_covered - covered > lossy_max_misses_per_tile;

      if (use_tile_fallback) {
        if (fallback_pool == nullptr || fallback_tile_ids == nullptr ||
            fallback_tile_next_slot == nullptr) {
          byte_v2_record_cache_update_error(
              error, kByteV2CacheUpdateErrorFallbackPoolMissing);
          return false;
        }
        if (tile_capacity <= 0) {
          byte_v2_record_cache_update_error(
              error, kByteV2CacheUpdateErrorFallbackPoolExhausted);
          return false;
        }
        const int tile_slot = byte_v2_allocate_tile_fallback_slot(
            fallback_tile_next_slot, tile_capacity);
        if (tile_slot < 0) {
          byte_v2_record_cache_update_error(
              error, kByteV2CacheUpdateErrorFallbackPoolExhausted);
          return false;
        }
        fallback_tile_ids[block_id * total_tiles + tile_idx] = tile_slot;
        byte_v2_store_raw_tile_from_block(raw_block, fallback_pool, tile_slot,
                                          is_value, kv_head, dim_tile,
                                          num_kv_heads, head_size, head_size_v,
                                          valid_rows);
        const int tile_start =
            byte_v2_tile_start(is_value, kv_head, dim_tile, k_dim_tiles,
                               v_dim_tiles);
        page[tile_start] = 0;
        page[tile_start + 1] = 1;
      } else {
        if (fallback_tile_ids != nullptr) {
          fallback_tile_ids[block_id * total_tiles + tile_idx] = -1;
        }
        byte_v2_store_compressed_tile(raw_block, page, is_value, kv_head,
                                      dim_tile, num_kv_heads, head_size,
                                      head_size_v, valid_rows,
                                      lossy_max_misses_per_tile);
      }
    }
  }
  page[kByteV2PageStatusOffset] = kByteV2PageStatusCompressed;
  page[kByteV2PageValidRowsOffset] = static_cast<uint8_t>(valid_rows);
  return true;
}

__device__ bool byte_v2_store_compressed_or_tile_fallback_block_v3(
    const uint8_t* __restrict__ raw_block, uint8_t* __restrict__ page,
    uint8_t* __restrict__ fallback_pool,
    int32_t* __restrict__ fallback_tile_ids,
    int32_t* __restrict__ fallback_tile_next_slot,
    int32_t* __restrict__ error, const int block_id, const int num_kv_heads,
    const int head_size, const int head_size_v, const int valid_rows,
    const int raw_block_bytes, const int fallback_pool_blocks,
    const int lossy_max_misses_per_tile) {
  byte_v2_init_v3_compressed_page_header(page, num_kv_heads, head_size,
                                         head_size_v);
  const int k_dim_tiles = head_size / kByteV2TileSize;
  const int v_dim_tiles = head_size_v / kByteV2TileSize;
  const int tiles_per_head = k_dim_tiles + v_dim_tiles;
  const int total_tiles = num_kv_heads * tiles_per_head;
  const int required_covered = valid_rows * kByteV2TileSize;
  const int64_t tile_capacity =
      byte_v2_tile_fallback_capacity(fallback_pool_blocks, raw_block_bytes);

  for (int kv_head = 0; kv_head < num_kv_heads; ++kv_head) {
    uint16_t k_fallback_mask = 0;
    uint16_t v_fallback_mask = 0;
    for (int tile_in_head = 0; tile_in_head < tiles_per_head; ++tile_in_head) {
      const bool is_value = tile_in_head >= k_dim_tiles;
      const int dim_tile =
          is_value ? tile_in_head - k_dim_tiles : tile_in_head;
      const int tile_idx = byte_v2_tile_index(is_value, kv_head, dim_tile,
                                              k_dim_tiles, v_dim_tiles);
      int covered = 0;
      byte_v2_best_window_base_from_raw_tile(
          raw_block, is_value, kv_head, dim_tile, num_kv_heads, head_size,
          head_size_v, valid_rows, &covered);
      const bool use_tile_fallback =
          required_covered - covered > lossy_max_misses_per_tile;

      if (use_tile_fallback) {
        if (fallback_pool == nullptr || fallback_tile_ids == nullptr ||
            fallback_tile_next_slot == nullptr) {
          byte_v2_record_cache_update_error(
              error, kByteV2CacheUpdateErrorFallbackPoolMissing);
          return false;
        }
        if (tile_capacity <= 0) {
          byte_v2_record_cache_update_error(
              error, kByteV2CacheUpdateErrorFallbackPoolExhausted);
          return false;
        }
        const int tile_slot = byte_v2_allocate_tile_fallback_slot(
            fallback_tile_next_slot, tile_capacity);
        if (tile_slot < 0) {
          byte_v2_record_cache_update_error(
              error, kByteV2CacheUpdateErrorFallbackPoolExhausted);
          return false;
        }
        fallback_tile_ids[block_id * total_tiles + tile_idx] = tile_slot;
        byte_v2_store_raw_tile_from_block(raw_block, fallback_pool, tile_slot,
                                          is_value, kv_head, dim_tile,
                                          num_kv_heads, head_size, head_size_v,
                                          valid_rows);
        if (is_value) {
          v_fallback_mask |= static_cast<uint16_t>(1u << dim_tile);
        } else {
          k_fallback_mask |= static_cast<uint16_t>(1u << dim_tile);
        }
      } else {
        if (fallback_tile_ids != nullptr) {
          fallback_tile_ids[block_id * total_tiles + tile_idx] = -1;
        }
        byte_v2_store_compressed_tile_v3(
            raw_block, page, is_value, kv_head, dim_tile, num_kv_heads,
            head_size, head_size_v, valid_rows, lossy_max_misses_per_tile);
      }
    }
    const int meta_start = byte_v2_v3_kv_head_meta_start(kv_head);
    byte_v2_store_u16(page + meta_start + 16, k_fallback_mask);
    byte_v2_store_u16(page + meta_start + 18, v_fallback_mask);
  }
  page[kByteV2PageStatusOffset] = kByteV2PageStatusCompressed;
  page[kByteV2PageValidRowsOffset] = static_cast<uint8_t>(valid_rows);
  return true;
}

__device__ bool byte_v2_finalize_raw_block_parallel(
    const uint8_t* __restrict__ raw_block, uint8_t* __restrict__ page,
    uint8_t* __restrict__ fallback_pool,
    int32_t* __restrict__ fallback_block_ids,
    int32_t* __restrict__ fallback_tile_ids,
    int32_t* __restrict__ fallback_tile_next_slot,
    int32_t* __restrict__ outlier_arena,
    int32_t* __restrict__ outlier_block_flags,
    int32_t* __restrict__ outlier_tile_bitmap,
    int32_t* __restrict__ outlier_tile_meta,
    int32_t* __restrict__ outlier_next_entry,
    int32_t* __restrict__ result, const int block_id,
    const int num_kv_heads, const int head_size, const int head_size_v,
	    const int raw_block_bytes, const int fallback_pool_blocks,
	    const int lossy_max_misses_per_tile, const int outlier_arena_entries,
	    const int outlier_max_per_tile,
	    uint8_t* __restrict__ tile_bases,
	    uint8_t* __restrict__ tile_needs_fallback,
	    uint8_t* __restrict__ tile_outlier_counts,
	    int* __restrict__ tile_slots, int* __restrict__ tile_hist,
	    int* __restrict__ block_compressible, int* __restrict__ block_has_outlier,
	    const bool use_v3_payload_layout,
	    const bool v3_outlier_only_no_fallback) {
  const int tid = threadIdx.x;
  const int warp_id = tid / 32;
  const int lane = tid & 31;
  const int num_warps = blockDim.x / 32;
  const int k_dim_tiles = head_size / kByteV2TileSize;
  const int v_dim_tiles = head_size_v / kByteV2TileSize;
  const int tiles_per_head = k_dim_tiles + v_dim_tiles;
  const int total_tiles = num_kv_heads * tiles_per_head;
	  const bool has_tile_fallback =
	      fallback_pool != nullptr && fallback_tile_ids != nullptr &&
	      fallback_tile_next_slot != nullptr &&
	      !v3_outlier_only_no_fallback;
  const bool has_outlier_arena =
      outlier_arena != nullptr && outlier_tile_meta != nullptr &&
      outlier_next_entry != nullptr && outlier_arena_entries > 0 &&
      outlier_max_per_tile > 0;
  const bool use_single_outlier_fastpath =
      has_outlier_arena && lossy_max_misses_per_tile == 0 &&
      outlier_max_per_tile == 1;
  const int bitmap_words = byte_v2_outlier_tile_bitmap_words(total_tiles);
  const int64_t tile_capacity =
      byte_v2_tile_fallback_capacity(fallback_pool_blocks, raw_block_bytes);

  if (tid == 0) {
    *block_compressible = 1;
    *block_has_outlier = 0;
  }
  __syncthreads();

  if (total_tiles > kByteV2MaxTilesPerBlock) {
    if (tid == 0) {
      byte_v2_record_cache_update_error(
          result, kByteV2CacheUpdateErrorInvalidSlot);
      page[kByteV2PageStatusOffset] = kByteV2PageStatusRawFallback;
      page[kByteV2PageValidRowsOffset] = kByteV2TileSize;
    }
    return false;
  }
  if (use_v3_payload_layout && tid == 0) {
    byte_v2_init_v3_compressed_page_header(page, num_kv_heads, head_size,
                                           head_size_v);
  }
  __syncthreads();

  for (int tile_idx = warp_id; tile_idx < total_tiles;
       tile_idx += num_warps) {
    const int kv_head = tile_idx / tiles_per_head;
    const int tile_in_head = tile_idx % tiles_per_head;
    const bool is_value = tile_in_head >= k_dim_tiles;
    const int dim_tile =
        is_value ? tile_in_head - k_dim_tiles : tile_in_head;
    const int d0 = dim_tile * kByteV2TileSize;

    if (lossy_max_misses_per_tile > 0 || has_outlier_arena) {
      int local_min = 255;
      int local_max = 0;
      for (int elem = lane; elem < kByteV2TileElems; elem += 32) {
        const int row = elem / kByteV2TileSize;
        const int dim_in_tile = elem % kByteV2TileSize;
        const uint16_t bits = byte_v2_load_raw_bits_from_block(
            raw_block, is_value, row, kv_head, d0 + dim_in_tile,
            num_kv_heads, head_size, head_size_v);
        const int exp = (bits >> 7) & 0xff;
        local_min = min(local_min, exp);
        local_max = max(local_max, exp);
      }
      for (int offset = 16; offset > 0; offset >>= 1) {
        local_min = min(local_min,
                        __shfl_down_sync(0xffffffff, local_min, offset));
        local_max = max(local_max,
                        __shfl_down_sync(0xffffffff, local_max, offset));
      }
      const int tile_min = __shfl_sync(0xffffffff, local_min, 0);
      const int tile_max = __shfl_sync(0xffffffff, local_max, 0);
      if (tile_max - tile_min <= 15) {
        if (lane == 0) {
          tile_bases[tile_idx] = static_cast<uint8_t>(tile_min);
          tile_needs_fallback[tile_idx] = 0;
          tile_outlier_counts[tile_idx] = 0;
          tile_slots[tile_idx] = -1;
        }
      } else if (use_single_outlier_fastpath) {
        const int high_base = tile_min;
        const int low_base = max(0, tile_max - 15);
        int high_misses = 0;
        int low_misses = 0;
        for (int elem = lane; elem < kByteV2TileElems; elem += 32) {
          const int row =
              is_value ? elem / kByteV2TileSize : elem % kByteV2TileSize;
          const int dim_in_tile =
              is_value ? elem % kByteV2TileSize : elem / kByteV2TileSize;
          const uint16_t bits = byte_v2_load_raw_bits_from_block(
              raw_block, is_value, row, kv_head, d0 + dim_in_tile,
              num_kv_heads, head_size, head_size_v);
          const int exp = (bits >> 7) & 0xff;
          high_misses += exp > high_base + 15 ? 1 : 0;
          low_misses += exp < low_base ? 1 : 0;
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
          high_misses += __shfl_down_sync(0xffffffff, high_misses, offset);
          low_misses += __shfl_down_sync(0xffffffff, low_misses, offset);
        }
        if (lane == 0) {
          int base = high_base;
          int miss_count = high_misses;
          if (high_misses > 1 && low_misses <= 1) {
            base = low_base;
            miss_count = low_misses;
          }
          const bool use_outlier = miss_count == 1;
          const bool needs_fallback = miss_count > 1;
          tile_bases[tile_idx] = static_cast<uint8_t>(base);
          tile_needs_fallback[tile_idx] = needs_fallback ? 1 : 0;
          tile_outlier_counts[tile_idx] =
              static_cast<uint8_t>(use_outlier ? 1 : 0);
          tile_slots[tile_idx] = -1;
          if (needs_fallback) {
            atomicExch(block_compressible, 0);
          }
        }
      } else {
      int* hist = tile_hist + warp_id * 256;
      for (int exp = lane; exp < 256; exp += 32) {
        hist[exp] = 0;
      }
      __syncwarp();
      for (int elem = lane; elem < kByteV2TileElems; elem += 32) {
        const int row = elem / kByteV2TileSize;
        const int dim_in_tile = elem % kByteV2TileSize;
        const uint16_t bits = byte_v2_load_raw_bits_from_block(
            raw_block, is_value, row, kv_head, d0 + dim_in_tile,
            num_kv_heads, head_size, head_size_v);
        const int exp = (bits >> 7) & 0xff;
        atomicAdd(hist + exp, 1);
      }
      __syncwarp();
      if (lane == 0) {
        int covered = 0;
        const int base = byte_v2_best_window_base_from_hist(hist, &covered);
          const int miss_count = kByteV2TileElems - covered;
          const bool use_outlier =
              has_outlier_arena && miss_count > lossy_max_misses_per_tile &&
              miss_count <= outlier_max_per_tile;
          const bool needs_fallback =
              miss_count > lossy_max_misses_per_tile && !use_outlier;
        tile_bases[tile_idx] = static_cast<uint8_t>(base);
        tile_needs_fallback[tile_idx] = needs_fallback ? 1 : 0;
          tile_outlier_counts[tile_idx] =
              static_cast<uint8_t>(use_outlier ? miss_count : 0);
        tile_slots[tile_idx] = -1;
        if (needs_fallback) {
          atomicExch(block_compressible, 0);
        }
      }
      }
    } else {
      int local_min = 255;
      int local_max = 0;
      for (int elem = lane; elem < kByteV2TileElems; elem += 32) {
        const int row = elem / kByteV2TileSize;
        const int dim_in_tile = elem % kByteV2TileSize;
        const uint16_t bits = byte_v2_load_raw_bits_from_block(
            raw_block, is_value, row, kv_head, d0 + dim_in_tile,
            num_kv_heads, head_size, head_size_v);
        const int exp = (bits >> 7) & 0xff;
        local_min = min(local_min, exp);
        local_max = max(local_max, exp);
      }
      for (int offset = 16; offset > 0; offset >>= 1) {
        local_min = min(local_min,
                        __shfl_down_sync(0xffffffff, local_min, offset));
        local_max = max(local_max,
                        __shfl_down_sync(0xffffffff, local_max, offset));
      }
      if (lane == 0) {
        const bool needs_fallback = local_max - local_min > 15;
        tile_bases[tile_idx] = static_cast<uint8_t>(local_min);
        tile_needs_fallback[tile_idx] = needs_fallback ? 1 : 0;
        tile_outlier_counts[tile_idx] = 0;
        tile_slots[tile_idx] = -1;
        if (needs_fallback) {
          atomicExch(block_compressible, 0);
        }
      }
    }
  }
  __syncthreads();

	  if (*block_compressible == 0 && !has_tile_fallback) {
	    if (tid == 0) {
	      page[kByteV2PageStatusOffset] = kByteV2PageStatusRawFallback;
	      page[kByteV2PageValidRowsOffset] = kByteV2TileSize;
	      if (v3_outlier_only_no_fallback) {
	        byte_v2_record_cache_update_error(
	            result, kByteV2CacheUpdateErrorOutlierArenaExhausted);
	      }
	    }
	    return false;
	  }

  for (int tile_idx = warp_id; tile_idx < total_tiles;
       tile_idx += num_warps) {
    const int kv_head = tile_idx / tiles_per_head;
    const int tile_in_head = tile_idx % tiles_per_head;
    const bool is_value = tile_in_head >= k_dim_tiles;
    const int dim_tile =
        is_value ? tile_in_head - k_dim_tiles : tile_in_head;
    const int d0 = dim_tile * kByteV2TileSize;
    const int tile_start =
        byte_v2_tile_start(is_value, kv_head, dim_tile, k_dim_tiles,
                           v_dim_tiles);
    const int tile_start_v3 =
        use_v3_payload_layout
            ? byte_v2_v3_tile_payload_start(is_value, kv_head, dim_tile,
                                            num_kv_heads, k_dim_tiles,
                                            v_dim_tiles)
            : 0;

    if (tile_outlier_counts[tile_idx] != 0) {
      if (lane == 0) {
        const int count = static_cast<int>(tile_outlier_counts[tile_idx]);
        const int offset = atomicAdd(outlier_next_entry, count);
	        if (offset < 0 || offset + count > outlier_arena_entries ||
	            offset > kByteV2MaxOutlierArenaOffset) {
	          tile_needs_fallback[tile_idx] = 1;
	          tile_slots[tile_idx] = -1;
	          outlier_tile_meta[block_id * total_tiles + tile_idx] = -1;
	          if (v3_outlier_only_no_fallback) {
	            atomicExch(block_compressible, 0);
	            byte_v2_record_cache_update_error(
	                result, kByteV2CacheUpdateErrorOutlierArenaExhausted);
	          }
	        } else {
          tile_slots[tile_idx] = offset;
          outlier_tile_meta[block_id * total_tiles + tile_idx] =
              byte_v2_pack_outlier_meta(offset, count);
          byte_v2_set_outlier_tile_bitmap(outlier_tile_bitmap, block_id,
                                          bitmap_words, tile_idx);
        }
      }
      __syncwarp();
    }
	    if (tile_needs_fallback[tile_idx] != 0) {
	      if (!has_tile_fallback) {
	        if (lane == 0) {
	          if (outlier_tile_meta != nullptr) {
	            outlier_tile_meta[block_id * total_tiles + tile_idx] = -1;
	          }
	          byte_v2_record_cache_update_error(
	              result,
	              v3_outlier_only_no_fallback
	                  ? kByteV2CacheUpdateErrorOutlierArenaExhausted
	                  : kByteV2CacheUpdateErrorFallbackPoolMissing);
	          atomicExch(block_compressible, 0);
	        }
	        continue;
	      }
	      if (lane == 0) {
        if (outlier_tile_meta != nullptr) {
          outlier_tile_meta[block_id * total_tiles + tile_idx] = -1;
        }
        if (tile_capacity <= 0) {
          byte_v2_record_cache_update_error(
              result, kByteV2CacheUpdateErrorFallbackPoolExhausted);
        } else {
          const int tile_slot = byte_v2_allocate_tile_fallback_slot(
              fallback_tile_next_slot, tile_capacity);
          if (tile_slot < 0) {
            byte_v2_record_cache_update_error(
                result, kByteV2CacheUpdateErrorFallbackPoolExhausted);
          } else {
            tile_slots[tile_idx] = tile_slot;
            fallback_tile_ids[block_id * total_tiles + tile_idx] = tile_slot;
            if (!use_v3_payload_layout) {
              page[tile_start] = 0;
              page[tile_start + 1] = 1;
            }
          }
        }
      }
      __syncwarp();
      const int tile_slot = tile_slots[tile_idx];
      if (tile_slot >= 0) {
        for (int elem = lane; elem < kByteV2TileElems; elem += 32) {
          const int row = elem / kByteV2TileSize;
          const int dim_in_tile = elem % kByteV2TileSize;
          const uint16_t bits = byte_v2_load_raw_bits_from_block(
              raw_block, is_value, row, kv_head, d0 + dim_in_tile,
              num_kv_heads, head_size, head_size_v);
          byte_v2_store_raw_bits_to_tile_pool(fallback_pool, tile_slot, row,
                                              dim_in_tile, bits);
        }
      }
      continue;
    }

    const int base = static_cast<int>(tile_bases[tile_idx]);
    const bool use_outlier = tile_outlier_counts[tile_idx] != 0;
    const int outlier_offset = use_outlier ? tile_slots[tile_idx] : -1;
    if (lane == 0) {
      if (fallback_tile_ids != nullptr) {
        fallback_tile_ids[block_id * total_tiles + tile_idx] = -1;
      }
      if (!use_outlier && outlier_tile_meta != nullptr) {
        outlier_tile_meta[block_id * total_tiles + tile_idx] = -1;
      }
      if (use_v3_payload_layout) {
        const int meta_start = byte_v2_v3_kv_head_meta_start(kv_head);
        page[meta_start + (is_value ? 8 + dim_tile : dim_tile)] =
            static_cast<uint8_t>(base);
      } else {
        page[tile_start] = static_cast<uint8_t>(base);
        page[tile_start + 1] = 0;
      }
      if (use_outlier && !use_single_outlier_fastpath) {
        atomicExch(block_has_outlier, 1);
        int written = 0;
        for (int elem = 0; elem < kByteV2TileElems; ++elem) {
          const int row =
              is_value ? elem / kByteV2TileSize : elem % kByteV2TileSize;
          const int dim_in_tile =
              is_value ? elem % kByteV2TileSize : elem / kByteV2TileSize;
          const uint16_t bits = byte_v2_load_raw_bits_from_block(
              raw_block, is_value, row, kv_head, d0 + dim_in_tile,
              num_kv_heads, head_size, head_size_v);
          const int exp = (bits >> 7) & 0xff;
          if (!byte_v2_exp_in_window(exp, base)) {
            outlier_arena[outlier_offset + written] =
                byte_v2_pack_outlier_entry(elem, bits);
            ++written;
          }
        }
      }
    }
    if (use_outlier && use_single_outlier_fastpath) {
      if (lane == 0) {
        atomicExch(block_has_outlier, 1);
      }
      int found_elem = kByteV2TileElems;
      int found_bits = 0;
      for (int elem = lane; elem < kByteV2TileElems; elem += 32) {
        const int row =
            is_value ? elem / kByteV2TileSize : elem % kByteV2TileSize;
        const int dim_in_tile =
            is_value ? elem % kByteV2TileSize : elem / kByteV2TileSize;
        const uint16_t bits = byte_v2_load_raw_bits_from_block(
            raw_block, is_value, row, kv_head, d0 + dim_in_tile,
            num_kv_heads, head_size, head_size_v);
        const int exp = (bits >> 7) & 0xff;
        if (!byte_v2_exp_in_window(exp, base) && elem < found_elem) {
          found_elem = elem;
          found_bits = static_cast<int>(bits);
        }
      }
      for (int offset = 16; offset > 0; offset >>= 1) {
        const int other_elem =
            __shfl_down_sync(0xffffffff, found_elem, offset);
        const int other_bits =
            __shfl_down_sync(0xffffffff, found_bits, offset);
        if (other_elem < found_elem) {
          found_elem = other_elem;
          found_bits = other_bits;
        }
      }
      if (lane == 0 && found_elem < kByteV2TileElems) {
        outlier_arena[outlier_offset] = byte_v2_pack_outlier_entry(
            found_elem, static_cast<uint16_t>(found_bits));
      }
    }

    for (int packed_idx = lane; packed_idx < kByteV2PackedTileElems;
         packed_idx += 32) {
      const int elem0 = packed_idx * 2;
      const int row0 =
          is_value ? elem0 / kByteV2TileSize : elem0 % kByteV2TileSize;
      const int dim_in_tile0 =
          is_value ? elem0 % kByteV2TileSize : elem0 / kByteV2TileSize;
      const int elem1 = elem0 + 1;
      const int row1 =
          is_value ? elem1 / kByteV2TileSize : elem1 % kByteV2TileSize;
      const int dim_in_tile1 =
          is_value ? elem1 % kByteV2TileSize : elem1 / kByteV2TileSize;
      const uint16_t bits0 = byte_v2_load_raw_bits_from_block(
          raw_block, is_value, row0, kv_head, d0 + dim_in_tile0,
          num_kv_heads, head_size, head_size_v);
      const uint16_t bits1 = byte_v2_load_raw_bits_from_block(
          raw_block, is_value, row1, kv_head, d0 + dim_in_tile1,
          num_kv_heads, head_size, head_size_v);
      const int exp0 = (bits0 >> 7) & 0xff;
      const int exp1 = (bits1 >> 7) & 0xff;
      const int stored_exp0 =
          (lossy_max_misses_per_tile > 0 || use_outlier)
              ? byte_v2_clamp_exp_to_window(exp0, base)
              : exp0;
      const int stored_exp1 =
          (lossy_max_misses_per_tile > 0 || use_outlier)
              ? byte_v2_clamp_exp_to_window(exp1, base)
              : exp1;
      const uint8_t low0 =
          static_cast<uint8_t>((bits0 & 0x7f) | ((stored_exp0 & 1) << 7));
      const uint8_t low1 =
          static_cast<uint8_t>((bits1 & 0x7f) | ((stored_exp1 & 1) << 7));
      uint8_t packed = 0;
      const int delta0 = stored_exp0 - base;
      const int sign0 = (bits0 >> 15) & 1;
      packed |= static_cast<uint8_t>(
          ((sign0 << 3) | ((delta0 >> 1) & 0x07)) & 0x0f);
      const int delta1 = stored_exp1 - base;
      const int sign1 = (bits1 >> 15) & 1;
      packed |= static_cast<uint8_t>(
          (((sign1 << 3) | ((delta1 >> 1) & 0x07)) & 0x0f) << 4);

      if (use_v3_payload_layout) {
        const int stripe = packed_idx / kByteV2TilePayloadStripePairsV3;
        const int stripe_lane = packed_idx % kByteV2TilePayloadStripePairsV3;
        uint8_t* __restrict__ stripe_ptr =
            page + tile_start_v3 + stripe * kByteV2TilePayloadStripeBytesV3;
        stripe_ptr[stripe_lane] = low0;
        stripe_ptr[32 + stripe_lane] = low1;
        stripe_ptr[64 + stripe_lane] = packed;
      } else {
        page[tile_start + 2 + elem0] = low0;
        page[tile_start + 2 + elem1] = low1;
        page[tile_start + 2 + kByteV2TileElems + packed_idx] = packed;
      }
    }
	  }
	  __syncthreads();

	  if (*block_compressible == 0 && !has_tile_fallback) {
	    if (tid == 0) {
	      page[kByteV2PageStatusOffset] = kByteV2PageStatusRawFallback;
	      page[kByteV2PageValidRowsOffset] = kByteV2TileSize;
	    }
	    return false;
	  }

	  if (tid == 0) {
    if (use_v3_payload_layout) {
      for (int kv_head = 0; kv_head < num_kv_heads; ++kv_head) {
        uint16_t k_fallback_mask = 0;
        uint16_t v_fallback_mask = 0;
        for (int dim_tile = 0; dim_tile < k_dim_tiles; ++dim_tile) {
          const int tile_idx = byte_v2_tile_index(
              false, kv_head, dim_tile, k_dim_tiles, v_dim_tiles);
          if (tile_needs_fallback[tile_idx] != 0) {
            k_fallback_mask |= static_cast<uint16_t>(1u << dim_tile);
          }
        }
        for (int dim_tile = 0; dim_tile < v_dim_tiles; ++dim_tile) {
          const int tile_idx = byte_v2_tile_index(
              true, kv_head, dim_tile, k_dim_tiles, v_dim_tiles);
          if (tile_needs_fallback[tile_idx] != 0) {
            v_fallback_mask |= static_cast<uint16_t>(1u << dim_tile);
          }
        }
        const int meta_start = byte_v2_v3_kv_head_meta_start(kv_head);
        byte_v2_store_u16(page + meta_start + 16, k_fallback_mask);
        byte_v2_store_u16(page + meta_start + 18, v_fallback_mask);
      }
    }
    page[kByteV2PageStatusOffset] = kByteV2PageStatusCompressed;
    page[kByteV2PageValidRowsOffset] = kByteV2TileSize;
    if (fallback_block_ids != nullptr) {
      fallback_block_ids[block_id] = -1;
    }
    if (outlier_block_flags != nullptr) {
      outlier_block_flags[block_id] = *block_has_outlier;
    }
  }
  return true;
}

__global__ void byte_v2_init_cache_update_kernel(
    const uint8_t* __restrict__ kv_cache, int32_t* __restrict__ valid_rows,
    uint8_t* __restrict__ touched_flags, uint8_t* __restrict__ packed_flags,
    int32_t* __restrict__ error, const int num_blocks,
    const int page_size_bytes) {
  const int block_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (block_id >= num_blocks) {
    return;
  }
  const uint8_t* page = kv_cache + block_id * page_size_bytes;
  const uint8_t status = page[kByteV2PageStatusOffset];
  const uint8_t page_valid_rows = page[kByteV2PageValidRowsOffset];
  if (status == kByteV2PageStatusEmpty) {
    valid_rows[block_id] = 0;
  } else if (status == kByteV2PageStatusRawFallback) {
    valid_rows[block_id] = static_cast<int32_t>(page_valid_rows);
  } else if (status == kByteV2PageStatusCompressed) {
    if (page_valid_rows == 0 || page_valid_rows > kByteV2TileSize) {
      valid_rows[block_id] = 0;
      byte_v2_record_cache_update_error(
          error, kByteV2CacheUpdateErrorInvalidValidRows);
    } else {
      valid_rows[block_id] = static_cast<int32_t>(page_valid_rows);
    }
  } else {
    valid_rows[block_id] = 0;
    byte_v2_record_cache_update_error(
        error, kByteV2CacheUpdateErrorInvalidPageState);
  }
  touched_flags[block_id] = 0;
  packed_flags[block_id] = 0;
}

__global__ void byte_v2_write_raw_tokens_kernel(
    const uint8_t* __restrict__ key, const uint8_t* __restrict__ value,
    uint8_t* __restrict__ kv_cache, const int64_t* __restrict__ slot_mapping,
    int32_t* __restrict__ valid_rows, uint8_t* __restrict__ touched_flags,
    int32_t* __restrict__ error, const int num_tokens, const int num_blocks,
    const int num_kv_heads, const int head_size, const int head_size_v,
    const int page_size_bytes, const int64_t key_stride0,
    const int64_t key_stride1, const int64_t key_stride2,
    const int64_t value_stride0, const int64_t value_stride1,
    const int64_t value_stride2) {
  const int token_idx = blockIdx.x;
  if (token_idx >= num_tokens) {
    return;
  }
  const int64_t slot = slot_mapping[token_idx];
  if (slot < 0) {
    return;
  }
  const int block_id = static_cast<int>(slot / kByteV2TileSize);
  const int block_offset = static_cast<int>(slot % kByteV2TileSize);
  if (block_id < 0 || block_id >= num_blocks) {
    if (threadIdx.x == 0) {
      byte_v2_record_cache_update_error(
          error, kByteV2CacheUpdateErrorInvalidSlot);
    }
    return;
  }

  uint8_t* page = kv_cache + block_id * page_size_bytes;
  __shared__ int skip_block;
  if (threadIdx.x == 0) {
    skip_block = 0;
    const uint8_t status = page[kByteV2PageStatusOffset];
    if (status == kByteV2PageStatusCompressed) {
      byte_v2_record_cache_update_error(
          error, kByteV2CacheUpdateErrorFinalizedBlockUpdate);
      skip_block = 1;
    } else {
      page[kByteV2PageStatusOffset] = kByteV2PageStatusRawFallback;
      touched_flags[block_id] = 1;
      atomicMax(valid_rows + block_id, block_offset + 1);
    }
  }
  __syncthreads();
  if (skip_block) {
    return;
  }

  const int key_elems = num_kv_heads * head_size;
  const int value_elems = num_kv_heads * head_size_v;
  const int raw_key_bytes = byte_v2_raw_key_bytes(num_kv_heads, head_size);
  for (int elem = threadIdx.x; elem < key_elems; elem += blockDim.x) {
    const int kv_head = elem / head_size;
    const int dim = elem % head_size;
    const int raw_elem =
        ((block_offset * num_kv_heads + kv_head) * head_size + dim) * 2;
    const int64_t src_elem =
        (token_idx * key_stride0 + kv_head * key_stride1 + dim * key_stride2) *
        2;
    page[kByteV2PageHeaderBytes + raw_elem] = key[src_elem];
    page[kByteV2PageHeaderBytes + raw_elem + 1] = key[src_elem + 1];
  }
  for (int elem = threadIdx.x; elem < value_elems; elem += blockDim.x) {
    const int kv_head = elem / head_size_v;
    const int dim = elem % head_size_v;
    const int raw_elem =
        raw_key_bytes +
        ((block_offset * num_kv_heads + kv_head) * head_size_v + dim) * 2;
    const int64_t src_elem = (token_idx * value_stride0 +
                              kv_head * value_stride1 + dim * value_stride2) *
                             2;
    page[kByteV2PageHeaderBytes + raw_elem] = value[src_elem];
    page[kByteV2PageHeaderBytes + raw_elem + 1] = value[src_elem + 1];
  }
}

__global__ void byte_v2_finalize_partial_pages_kernel(
    uint8_t* __restrict__ kv_cache, const int32_t* __restrict__ valid_rows,
    const uint8_t* __restrict__ touched_flags, const int num_blocks,
    const int page_size_bytes) {
  const int block_id = blockIdx.x * blockDim.x + threadIdx.x;
  if (block_id >= num_blocks || touched_flags[block_id] == 0) {
    return;
  }
  const int valid = valid_rows[block_id];
  if (valid > 0 && valid < kByteV2TileSize) {
    uint8_t* page = kv_cache + block_id * page_size_bytes;
    page[kByteV2PageStatusOffset] = kByteV2PageStatusRawFallback;
    page[kByteV2PageValidRowsOffset] = static_cast<uint8_t>(valid);
  }
}

__global__ void byte_v2_compress_full_pages_kernel(
    uint8_t* __restrict__ kv_cache, const int64_t* __restrict__ slot_mapping,
    const int32_t* __restrict__ valid_rows,
    const uint8_t* __restrict__ touched_flags, uint8_t* __restrict__ packed_flags,
    uint8_t* __restrict__ raw_staging, const int num_tokens,
    const int num_blocks, const int num_kv_heads, const int head_size,
    const int head_size_v, const int page_size_bytes,
    const int raw_block_bytes, const int lossy_max_misses_per_tile) {
  const int token_idx = blockIdx.x;
  if (token_idx >= num_tokens) {
    return;
  }
  const int64_t slot = slot_mapping[token_idx];
  if (slot < 0 || (slot % kByteV2TileSize) != (kByteV2TileSize - 1)) {
    return;
  }
  const int block_id = static_cast<int>(slot / kByteV2TileSize);
  if (block_id < 0 || block_id >= num_blocks || touched_flags[block_id] == 0 ||
      valid_rows[block_id] != kByteV2TileSize) {
    return;
  }

  uint8_t* page = kv_cache + block_id * page_size_bytes;
  uint8_t* raw_block = raw_staging + token_idx * raw_block_bytes;
  for (int byte_idx = threadIdx.x; byte_idx < raw_block_bytes;
       byte_idx += blockDim.x) {
    raw_block[byte_idx] = page[kByteV2PageHeaderBytes + byte_idx];
  }
  __syncthreads();

  if (threadIdx.x != 0) {
    return;
  }
  if (!byte_v2_raw_block_is_compressible(raw_block, num_kv_heads, head_size,
                                         head_size_v, kByteV2TileSize,
                                         lossy_max_misses_per_tile)) {
    page[kByteV2PageStatusOffset] = kByteV2PageStatusRawFallback;
    page[kByteV2PageValidRowsOffset] = kByteV2TileSize;
    return;
  }

  byte_v2_store_compressed_block(raw_block, page, num_kv_heads, head_size,
                                 head_size_v, kByteV2TileSize,
                                 lossy_max_misses_per_tile);
  page[kByteV2PageStatusOffset] = kByteV2PageStatusCompressed;
  page[kByteV2PageValidRowsOffset] = kByteV2TileSize;
  packed_flags[block_id] = 1;
}

__global__ void byte_v2_mark_touched_tokens_kernel(
    const uint8_t* __restrict__ kv_cache,
    const int64_t* __restrict__ slot_mapping,
    int32_t* __restrict__ valid_rows, uint8_t* __restrict__ touched_flags,
    uint8_t* __restrict__ overwrite_flags,
    int32_t* __restrict__ block_token_indices, int32_t* __restrict__ error,
    const int num_tokens, const int num_blocks, const int page_size_bytes) {
  const int token_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (token_idx >= num_tokens) {
    return;
  }
  const int64_t slot = slot_mapping[token_idx];
  if (slot < 0) {
    return;
  }
  const int block_id = static_cast<int>(slot / kByteV2TileSize);
  const int block_offset = static_cast<int>(slot % kByteV2TileSize);
  if (block_id < 0 || block_id >= num_blocks) {
    byte_v2_record_cache_update_error(
        error, kByteV2CacheUpdateErrorInvalidSlot);
    return;
  }
  const uint8_t* page = kv_cache + block_id * page_size_bytes;
  const uint8_t status = page[kByteV2PageStatusOffset];
  if (status != kByteV2PageStatusEmpty &&
      status != kByteV2PageStatusCompressed &&
      status != kByteV2PageStatusRawFallback) {
    byte_v2_record_cache_update_error(
        error, kByteV2CacheUpdateErrorInvalidPageState);
    return;
  }
  touched_flags[block_id] = 1;
  if (block_offset == 0) {
    overwrite_flags[block_id] = 1;
  }
  atomicMax(valid_rows + block_id, block_offset + 1);
  int32_t* token_slot =
      block_token_indices + block_id * kByteV2TileSize + block_offset;
  const int32_t old = atomicCAS(token_slot, -1, token_idx);
  if (old != -1) {
    byte_v2_record_cache_update_error(
        error, kByteV2CacheUpdateErrorDuplicateSlot);
  }
}

__device__ void byte_v2_decompress_page_to_raw_block(
    const uint8_t* __restrict__ page, uint8_t* __restrict__ raw_block,
    const uint8_t* __restrict__ fallback_pool,
    const int32_t* __restrict__ fallback_tile_ids,
    const int32_t* __restrict__ outlier_arena,
    const int32_t* __restrict__ outlier_tile_bitmap,
    const int32_t* __restrict__ outlier_tile_meta, const int physical_block,
    const int valid_rows, const int num_kv_heads, const int head_size,
    const int head_size_v) {
  const int key_elems = kByteV2TileSize * num_kv_heads * head_size;
  const int value_elems = kByteV2TileSize * num_kv_heads * head_size_v;
  const int total_tiles =
      byte_v2_total_tiles(num_kv_heads, head_size, head_size_v);
  const int bitmap_words = byte_v2_outlier_tile_bitmap_words(total_tiles);
  for (int elem = threadIdx.x; elem < key_elems; elem += blockDim.x) {
    const int row = elem / (num_kv_heads * head_size);
    const int rem = elem % (num_kv_heads * head_size);
    const int kv_head = rem / head_size;
    const int dim = rem % head_size;
    const uint16_t bits =
        row < valid_rows ? byte_v2_load_compressed_bits(
                               page, false, row, kv_head, dim,
                               head_size / kByteV2TileSize,
                               head_size_v / kByteV2TileSize, fallback_pool,
                               fallback_tile_ids, outlier_arena,
                               outlier_tile_bitmap, outlier_tile_meta,
                               physical_block, total_tiles, bitmap_words)
                         : 0;
    byte_v2_store_raw_bits_to_block(raw_block, false, row, kv_head, dim,
                                    num_kv_heads, head_size, head_size_v, bits);
  }
  for (int elem = threadIdx.x; elem < value_elems; elem += blockDim.x) {
    const int row = elem / (num_kv_heads * head_size_v);
    const int rem = elem % (num_kv_heads * head_size_v);
    const int kv_head = rem / head_size_v;
    const int dim = rem % head_size_v;
    const uint16_t bits =
        row < valid_rows ? byte_v2_load_compressed_bits(
                               page, true, row, kv_head, dim,
                               head_size / kByteV2TileSize,
                               head_size_v / kByteV2TileSize, fallback_pool,
                               fallback_tile_ids, outlier_arena,
                               outlier_tile_bitmap, outlier_tile_meta,
                               physical_block, total_tiles, bitmap_words)
                         : 0;
    byte_v2_store_raw_bits_to_block(raw_block, true, row, kv_head, dim,
                                    num_kv_heads, head_size, head_size_v, bits);
  }
}

__device__ void byte_v2_decompress_page_to_raw_block(
    const uint8_t* __restrict__ page, uint8_t* __restrict__ raw_block,
    const int valid_rows, const int num_kv_heads, const int head_size,
    const int head_size_v) {
  byte_v2_decompress_page_to_raw_block(
      page, raw_block, nullptr, nullptr, nullptr, nullptr, nullptr, 0,
      valid_rows, num_kv_heads, head_size, head_size_v);
}

__device__ __forceinline__ uint16_t byte_v2_load_prefill_direct_bits(
    const uint8_t* __restrict__ key, const uint8_t* __restrict__ value,
    const bool is_value, const int token_idx, const int kv_head, const int dim,
    const int64_t key_stride0, const int64_t key_stride1,
    const int64_t key_stride2, const int64_t value_stride0,
    const int64_t value_stride1, const int64_t value_stride2);

__global__ void byte_v2_decode_append_cache_kernel(
    const uint8_t* __restrict__ key, const uint8_t* __restrict__ value,
    uint8_t* __restrict__ kv_cache, const int64_t* __restrict__ slot_mapping,
    uint8_t* __restrict__ fallback_pool,
    int32_t* __restrict__ fallback_block_ids,
    int32_t* __restrict__ fallback_next_slot,
    int32_t* __restrict__ fallback_tile_ids,
    int32_t* __restrict__ fallback_tile_next_slot,
    int32_t* __restrict__ outlier_arena,
    int32_t* __restrict__ outlier_block_flags,
    int32_t* __restrict__ outlier_tile_bitmap,
    int32_t* __restrict__ outlier_tile_meta,
    int32_t* __restrict__ outlier_next_entry,
    int32_t* __restrict__ result,
    const int num_tokens, const int num_blocks, const int num_kv_heads,
    const int head_size, const int head_size_v, const int page_size_bytes,
	    const int raw_block_bytes, const int fallback_pool_blocks,
	    const int lossy_max_misses_per_tile, const int outlier_arena_entries,
	    const int outlier_max_per_tile,
	    const bool v3_outlier_only_no_fallback,
	    const int64_t key_stride0, const int64_t key_stride1,
	    const int64_t key_stride2, const int64_t value_stride0,
	    const int64_t value_stride1, const int64_t value_stride2) {
  const int token_idx = blockIdx.x;
  if (token_idx >= num_tokens) {
    return;
  }
  const int tid = threadIdx.x;

  __shared__ int block_id;
  __shared__ int block_offset;
  __shared__ int fallback_slot;
  __shared__ int existing_valid_rows;
  __shared__ int page_status;
  __shared__ int skip;
  __shared__ int zero_raw_block;
  __shared__ int decompress_existing;
  __shared__ uint8_t tile_bases[kByteV2MaxTilesPerBlock];
  __shared__ uint8_t tile_needs_fallback[kByteV2MaxTilesPerBlock];
  __shared__ uint8_t tile_outlier_counts[kByteV2MaxTilesPerBlock];
  __shared__ int tile_slots[kByteV2MaxTilesPerBlock];
  __shared__ int tile_hist[kByteV2MaxPrefillDirectWarps][256];
  __shared__ int block_compressible;
  __shared__ int block_has_outlier;

  if (num_tokens == 1 && tid == 0) {
    result[0] = 0;
    result[1] = -1;
    result[2] = -1;
  }
  __syncthreads();

  if (tid == 0) {
    skip = 0;
    zero_raw_block = 0;
    decompress_existing = 0;
    fallback_slot = -1;

    const int64_t slot = slot_mapping[token_idx];
    if (slot < 0) {
      skip = 1;
    } else {
      block_id = static_cast<int>(slot / kByteV2TileSize);
      block_offset = static_cast<int>(slot % kByteV2TileSize);
      if (block_id < 0 || block_id >= num_blocks) {
        byte_v2_record_cache_update_error_detail(
            result, kByteV2CacheUpdateErrorInvalidSlot, token_idx);
        skip = 1;
      } else {
        for (int other = 0; other < num_tokens; ++other) {
          if (other == token_idx) {
            continue;
          }
          const int64_t other_slot = slot_mapping[other];
          if (other_slot < 0) {
            continue;
          }
          const int other_block =
              static_cast<int>(other_slot / kByteV2TileSize);
          if (other_block == block_id) {
            byte_v2_record_cache_update_error_detail(
                result, kByteV2CacheUpdateErrorDuplicateSlot, block_id);
            skip = 1;
            break;
          }
        }
      }
    }
  }
  __syncthreads();
  if (skip) {
    return;
  }

  uint8_t* page = kv_cache + block_id * page_size_bytes;
  const int total_tiles =
      byte_v2_total_tiles(num_kv_heads, head_size, head_size_v);
  const int bitmap_words = byte_v2_outlier_tile_bitmap_words(total_tiles);
  if (tid == 0) {
    page_status = static_cast<int>(page[kByteV2PageStatusOffset]);
    existing_valid_rows =
        static_cast<int>(page[kByteV2PageValidRowsOffset]);
    const bool overwrite_block = block_offset == 0;

    if (page_status == kByteV2PageStatusEmpty) {
      existing_valid_rows = 0;
      if (!overwrite_block) {
        byte_v2_record_cache_update_error_detail(
            result, kByteV2CacheUpdateErrorInvalidValidRows, block_id);
        skip = 1;
      } else {
        zero_raw_block = 1;
      }
    } else if (page_status == kByteV2PageStatusCompressed ||
               page_status == kByteV2PageStatusRawFallback) {
      if (existing_valid_rows <= 0 ||
          existing_valid_rows > kByteV2TileSize) {
        byte_v2_record_cache_update_error_detail(
            result, kByteV2CacheUpdateErrorInvalidValidRows, block_id);
        skip = 1;
      } else if (!overwrite_block &&
                 existing_valid_rows == kByteV2TileSize) {
        byte_v2_record_cache_update_error_detail(
            result, kByteV2CacheUpdateErrorFinalizedBlockUpdate, block_id);
        skip = 1;
      } else if (!overwrite_block &&
                 existing_valid_rows != block_offset) {
        byte_v2_record_cache_update_error_detail(
            result, kByteV2CacheUpdateErrorInvalidValidRows, block_id);
        skip = 1;
      } else if (overwrite_block) {
        zero_raw_block = 1;
      } else if (page_status == kByteV2PageStatusCompressed) {
        decompress_existing = 1;
      }
    } else {
      byte_v2_record_cache_update_error_detail(
          result, kByteV2CacheUpdateErrorInvalidPageState, block_id);
      skip = 1;
    }

    if (!skip) {
      if (fallback_block_ids == nullptr || fallback_next_slot == nullptr) {
        byte_v2_record_cache_update_error_detail(
            result, kByteV2CacheUpdateErrorFallbackPoolMissing, block_id);
        skip = 1;
      } else if (fallback_pool_blocks <= 0) {
        byte_v2_record_cache_update_error_detail(
            result, kByteV2CacheUpdateErrorFallbackPoolExhausted, block_id);
        skip = 1;
      } else if (fallback_pool == nullptr) {
        byte_v2_record_cache_update_error_detail(
            result, kByteV2CacheUpdateErrorFallbackPoolMissing, block_id);
        skip = 1;
      }
    }

    if (!skip) {
      fallback_slot = fallback_block_ids[block_id];
      if (page_status != kByteV2PageStatusRawFallback || overwrite_block) {
        fallback_slot = -1;
      }
      if (fallback_slot < 0) {
        fallback_slot = atomicAdd(fallback_next_slot, 1);
        if (fallback_slot >= fallback_pool_blocks) {
          fallback_block_ids[block_id] = -1;
          byte_v2_record_cache_update_error_detail(
              result, kByteV2CacheUpdateErrorFallbackPoolExhausted, block_id);
          skip = 1;
        } else {
          fallback_block_ids[block_id] = fallback_slot;
        }
      } else if (fallback_slot >= fallback_pool_blocks) {
        byte_v2_record_cache_update_error_detail(
            result, kByteV2CacheUpdateErrorFallbackPoolInvalidSlot, block_id);
        skip = 1;
      }
    }
  }
  __syncthreads();
  if (skip) {
    return;
  }

  uint8_t* raw_block = fallback_pool + fallback_slot * raw_block_bytes;
  if (zero_raw_block) {
    for (int byte_idx = tid; byte_idx < raw_block_bytes;
         byte_idx += blockDim.x) {
      raw_block[byte_idx] = 0;
    }
  }
  __syncthreads();

  if (decompress_existing) {
    byte_v2_decompress_page_to_raw_block(
        page, raw_block, fallback_pool, fallback_tile_ids, outlier_arena,
        outlier_tile_bitmap, outlier_tile_meta, block_id,
        existing_valid_rows, num_kv_heads, head_size, head_size_v);
  }
  __syncthreads();

  if (tid == 0 && outlier_block_flags != nullptr) {
    outlier_block_flags[block_id] = 0;
  }
  if (outlier_tile_bitmap != nullptr) {
    for (int word = tid; word < bitmap_words; word += blockDim.x) {
      outlier_tile_bitmap[block_id * bitmap_words + word] = 0;
    }
  }
  if (outlier_tile_meta != nullptr) {
    for (int tile_idx = tid; tile_idx < total_tiles; tile_idx += blockDim.x) {
      outlier_tile_meta[block_id * total_tiles + tile_idx] = -1;
    }
  }
  __syncthreads();

  const int key_elems = num_kv_heads * head_size;
  const int value_elems = num_kv_heads * head_size_v;
  for (int elem = tid; elem < key_elems; elem += blockDim.x) {
    const int kv_head = elem / head_size;
    const int dim = elem % head_size;
    const uint16_t bits = byte_v2_load_prefill_direct_bits(
        key, value, false, token_idx, kv_head, dim, key_stride0, key_stride1,
        key_stride2, value_stride0, value_stride1, value_stride2);
    byte_v2_store_raw_bits_to_block(raw_block, false, block_offset, kv_head,
                                    dim, num_kv_heads, head_size, head_size_v,
                                    bits);
  }
  for (int elem = tid; elem < value_elems; elem += blockDim.x) {
    const int kv_head = elem / head_size_v;
    const int dim = elem % head_size_v;
    const uint16_t bits = byte_v2_load_prefill_direct_bits(
        key, value, true, token_idx, kv_head, dim, key_stride0, key_stride1,
        key_stride2, value_stride0, value_stride1, value_stride2);
    byte_v2_store_raw_bits_to_block(raw_block, true, block_offset, kv_head,
                                    dim, num_kv_heads, head_size, head_size_v,
                                    bits);
  }
  __syncthreads();

  const int valid_rows = block_offset + 1;
  if (valid_rows < kByteV2TileSize) {
    if (tid == 0) {
      page[kByteV2PageStatusOffset] = kByteV2PageStatusRawFallback;
      page[kByteV2PageValidRowsOffset] = static_cast<uint8_t>(valid_rows);
    }
    return;
  }

  if (byte_v2_finalize_raw_block_parallel(
          raw_block, page, fallback_pool, fallback_block_ids,
          fallback_tile_ids, fallback_tile_next_slot, outlier_arena,
          outlier_block_flags, outlier_tile_bitmap, outlier_tile_meta,
          outlier_next_entry, result, block_id,
          num_kv_heads, head_size, head_size_v, raw_block_bytes,
          fallback_pool_blocks, lossy_max_misses_per_tile,
          outlier_arena_entries, outlier_max_per_tile, tile_bases,
          tile_needs_fallback, tile_outlier_counts, tile_slots,
          &tile_hist[0][0], &block_compressible, &block_has_outlier,
	          byte_v2_page_uses_v3_layout(page_size_bytes, num_kv_heads,
	                                      head_size, head_size_v),
	          v3_outlier_only_no_fallback)) {
    if (tid == 0) {
      result[2 + token_idx] = block_id;
    }
  } else if (tid == 0 && result[0] != 0) {
    atomicCAS(reinterpret_cast<int*>(result + 1), -1, block_id);
  }
}

__global__ void byte_v2_init_decode_append_result_kernel(
    int32_t* __restrict__ result, const int num_tokens) {
  const int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx == 0) {
    result[0] = 0;
  }
  if (idx < num_tokens + 1) {
    result[idx + 1] = -1;
  }
}

__global__ void byte_v2_validate_decode_append_slots_kernel(
    const uint8_t* __restrict__ kv_cache,
    const int64_t* __restrict__ slot_mapping, int32_t* __restrict__ result,
    const int num_tokens, const int num_blocks, const int page_size_bytes) {
  const int token_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (token_idx >= num_tokens) {
    return;
  }

  const int64_t slot = slot_mapping[token_idx];
  if (slot < 0) {
    return;
  }
  const int block_id = static_cast<int>(slot / kByteV2TileSize);
  const int block_offset = static_cast<int>(slot % kByteV2TileSize);
  if (block_id < 0 || block_id >= num_blocks) {
    byte_v2_record_cache_update_error_detail(
        result, kByteV2CacheUpdateErrorInvalidSlot, token_idx);
    return;
  }

  for (int other = token_idx + 1; other < num_tokens; ++other) {
    const int64_t other_slot = slot_mapping[other];
    if (other_slot < 0) {
      continue;
    }
    if (static_cast<int>(other_slot / kByteV2TileSize) == block_id) {
      byte_v2_record_cache_update_error_detail(
          result, kByteV2CacheUpdateErrorDuplicateSlot, block_id);
      return;
    }
  }

  const uint8_t* page = kv_cache + block_id * page_size_bytes;
  const int page_status = static_cast<int>(page[kByteV2PageStatusOffset]);
  const int existing_valid_rows =
      static_cast<int>(page[kByteV2PageValidRowsOffset]);
  const bool overwrite_block = block_offset == 0;
  if (page_status == kByteV2PageStatusEmpty) {
    if (!overwrite_block) {
      byte_v2_record_cache_update_error_detail(
          result, kByteV2CacheUpdateErrorInvalidValidRows, block_id);
    }
    return;
  }
  if (page_status == kByteV2PageStatusCompressed ||
      page_status == kByteV2PageStatusRawFallback) {
    if (existing_valid_rows <= 0 ||
        existing_valid_rows > kByteV2TileSize) {
      byte_v2_record_cache_update_error_detail(
          result, kByteV2CacheUpdateErrorInvalidValidRows, block_id);
    } else if (!overwrite_block &&
               existing_valid_rows == kByteV2TileSize) {
      byte_v2_record_cache_update_error_detail(
          result, kByteV2CacheUpdateErrorFinalizedBlockUpdate, block_id);
    } else if (!overwrite_block &&
               existing_valid_rows != block_offset) {
      byte_v2_record_cache_update_error_detail(
          result, kByteV2CacheUpdateErrorInvalidValidRows, block_id);
    }
    return;
  }
  byte_v2_record_cache_update_error_detail(
      result, kByteV2CacheUpdateErrorInvalidPageState, block_id);
}

__global__ void byte_v2_record_deferred_cache_update_error_kernel(
    const int32_t* __restrict__ result,
    const int32_t* __restrict__ fallback_next_slot,
    int32_t* __restrict__ deferred_error,
    const int fallback_pool_blocks) {
  const int code = result[0];
  if (code == 0) {
    return;
  }
  const int old =
      atomicCAS(reinterpret_cast<int*>(deferred_error), 0, code);
  if (old == 0) {
    deferred_error[1] = result[1];
    deferred_error[2] =
        fallback_next_slot == nullptr ? -1 : fallback_next_slot[0];
    deferred_error[3] = fallback_pool_blocks;
  }
}

__global__ void byte_v2_validate_prefill_direct_blocks_kernel(
    const int64_t* __restrict__ slot_mapping,
    int32_t* __restrict__ group_block_ids, int32_t* __restrict__ block_claims,
    int32_t* __restrict__ direct_ineligible, const int num_groups,
    const int num_blocks) {
  const int group_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (group_idx >= num_groups) {
    return;
  }

  const int token_base = group_idx * kByteV2TileSize;
  const int64_t first_slot = slot_mapping[token_base];
  if (first_slot < 0 || (first_slot % kByteV2TileSize) != 0) {
    group_block_ids[group_idx] = -1;
    byte_v2_record_cache_update_error(direct_ineligible, 1);
    return;
  }
  const int block_id = static_cast<int>(first_slot / kByteV2TileSize);
  if (block_id < 0 || block_id >= num_blocks) {
    group_block_ids[group_idx] = -1;
    byte_v2_record_cache_update_error(direct_ineligible, 1);
    return;
  }

  for (int row = 1; row < kByteV2TileSize; ++row) {
    const int64_t slot = slot_mapping[token_base + row];
    if (slot < 0 ||
        static_cast<int>(slot / kByteV2TileSize) != block_id ||
        static_cast<int>(slot % kByteV2TileSize) != row) {
      group_block_ids[group_idx] = -1;
      byte_v2_record_cache_update_error(direct_ineligible, 1);
      return;
    }
  }
  const int32_t old_claim = atomicCAS(block_claims + block_id, -1, group_idx);
  if (old_claim != -1) {
    group_block_ids[group_idx] = -1;
    byte_v2_record_cache_update_error(direct_ineligible, 1);
    return;
  }
  group_block_ids[group_idx] = block_id;
}

__device__ __forceinline__ uint16_t byte_v2_load_prefill_direct_bits(
    const uint8_t* __restrict__ key, const uint8_t* __restrict__ value,
    const bool is_value, const int token_idx, const int kv_head, const int dim,
    const int64_t key_stride0, const int64_t key_stride1,
    const int64_t key_stride2, const int64_t value_stride0,
    const int64_t value_stride1, const int64_t value_stride2) {
  const uint8_t* src = is_value ? value : key;
  const int64_t stride0 = is_value ? value_stride0 : key_stride0;
  const int64_t stride1 = is_value ? value_stride1 : key_stride1;
  const int64_t stride2 = is_value ? value_stride2 : key_stride2;
  const int64_t src_elem =
      (token_idx * stride0 + kv_head * stride1 + dim * stride2) * 2;
  return byte_v2_load_u16(src + src_elem);
}

__global__ void byte_v2_prefill_direct_encode_blocks_kernel(
    const uint8_t* __restrict__ key, const uint8_t* __restrict__ value,
    uint8_t* __restrict__ kv_cache,
    const int32_t* __restrict__ group_block_ids,
    uint8_t* __restrict__ fallback_pool, int32_t* __restrict__ fallback_block_ids,
    int32_t* __restrict__ fallback_next_slot,
    int32_t* __restrict__ fallback_tile_ids,
    int32_t* __restrict__ fallback_tile_next_slot,
    int32_t* __restrict__ outlier_arena,
    int32_t* __restrict__ outlier_block_flags,
    int32_t* __restrict__ outlier_tile_bitmap,
    int32_t* __restrict__ outlier_tile_meta,
    int32_t* __restrict__ outlier_next_entry,
    int32_t* __restrict__ result,
    const int num_groups, const int num_blocks, const int num_kv_heads,
	    const int head_size, const int head_size_v, const int page_size_bytes,
	    const int raw_block_bytes, const int fallback_pool_blocks,
	    const int lossy_max_misses_per_tile, const int outlier_arena_entries,
	    const int outlier_max_per_tile,
	    const bool v3_outlier_only_no_fallback,
	    const int64_t key_stride0, const int64_t key_stride1,
    const int64_t key_stride2, const int64_t value_stride0,
    const int64_t value_stride1, const int64_t value_stride2) {
  const int group_idx = blockIdx.x;
  const int tid = threadIdx.x;
  if (group_idx >= num_groups) {
    return;
  }

  const int block_id = group_block_ids[group_idx];
  if (block_id < 0 || block_id >= num_blocks) {
    if (tid == 0) {
      result[group_idx + 1] = -1;
      byte_v2_record_cache_update_error(
          result, kByteV2CacheUpdateErrorInvalidSlot);
    }
    return;
  }

  uint8_t* page = kv_cache + block_id * page_size_bytes;
  __shared__ uint8_t tile_bases[kByteV2MaxTilesPerBlock];
  __shared__ uint8_t tile_needs_fallback[kByteV2MaxTilesPerBlock];
  __shared__ uint8_t tile_outlier_counts[kByteV2MaxTilesPerBlock];
  __shared__ int tile_slots[kByteV2MaxTilesPerBlock];
  __shared__ int tile_hist[kByteV2MaxPrefillDirectWarps][256];
  __shared__ int block_compressible;
  __shared__ int block_has_outlier;
  __shared__ int block_skip;
  __shared__ int fallback_slot_shared;

  if (tid == 0) {
    result[group_idx + 1] = -1;
    block_compressible = 1;
    block_has_outlier = 0;
    block_skip = 0;
    fallback_slot_shared = -1;
    if (outlier_block_flags != nullptr) {
      outlier_block_flags[block_id] = 0;
    }
    const uint8_t previous_status = page[kByteV2PageStatusOffset];
    if (previous_status != kByteV2PageStatusEmpty &&
        previous_status != kByteV2PageStatusCompressed &&
        previous_status != kByteV2PageStatusRawFallback) {
      byte_v2_record_cache_update_error(
          result, kByteV2CacheUpdateErrorInvalidPageState);
      block_skip = 1;
    }
  }
  __syncthreads();
  if (block_skip) {
    return;
  }

  const int k_dim_tiles = head_size / kByteV2TileSize;
  const int v_dim_tiles = head_size_v / kByteV2TileSize;
  const int tiles_per_head = k_dim_tiles + v_dim_tiles;
  const int total_tiles = num_kv_heads * tiles_per_head;
  const bool use_v3_payload_layout = byte_v2_page_uses_v3_layout(
      page_size_bytes, num_kv_heads, head_size, head_size_v);
  if (total_tiles > kByteV2MaxTilesPerBlock) {
    if (tid == 0) {
      byte_v2_record_cache_update_error(
          result, kByteV2CacheUpdateErrorInvalidSlot);
    }
    return;
  }
  if (use_v3_payload_layout && tid == 0) {
    byte_v2_init_v3_compressed_page_header(page, num_kv_heads, head_size,
                                           head_size_v);
  }
  const int bitmap_words = byte_v2_outlier_tile_bitmap_words(total_tiles);
  if (outlier_tile_bitmap != nullptr) {
    for (int word = tid; word < bitmap_words; word += blockDim.x) {
      outlier_tile_bitmap[block_id * bitmap_words + word] = 0;
    }
  }
  __syncthreads();

  const int warp_id = tid / 32;
  const int lane = tid & 31;
  const int num_warps = blockDim.x / 32;
  const int token_base = group_idx * kByteV2TileSize;
	  const bool has_tile_fallback =
	      fallback_pool != nullptr && fallback_tile_ids != nullptr &&
	      fallback_tile_next_slot != nullptr &&
	      !v3_outlier_only_no_fallback;
  const bool has_outlier_arena =
      outlier_arena != nullptr && outlier_tile_meta != nullptr &&
      outlier_next_entry != nullptr && outlier_arena_entries > 0 &&
      outlier_max_per_tile > 0;
  const bool use_single_outlier_fastpath =
      has_outlier_arena && lossy_max_misses_per_tile == 0 &&
      outlier_max_per_tile == 1;
  const int64_t tile_capacity =
      byte_v2_tile_fallback_capacity(fallback_pool_blocks, raw_block_bytes);

  for (int tile_idx = warp_id; tile_idx < total_tiles; tile_idx += num_warps) {
    const int kv_head = tile_idx / tiles_per_head;
    const int tile_in_head = tile_idx % tiles_per_head;
    const bool is_value = tile_in_head >= k_dim_tiles;
    const int dim_tile =
        is_value ? tile_in_head - k_dim_tiles : tile_in_head;
    const int d0 = dim_tile * kByteV2TileSize;

    if (lossy_max_misses_per_tile > 0 || has_outlier_arena) {
      int local_min = 255;
      int local_max = 0;
      for (int elem = lane; elem < kByteV2TileElems; elem += 32) {
        const int row = elem / kByteV2TileSize;
        const int dim = d0 + (elem % kByteV2TileSize);
        const uint16_t bits = byte_v2_load_prefill_direct_bits(
            key, value, is_value, token_base + row, kv_head, dim, key_stride0,
            key_stride1, key_stride2, value_stride0, value_stride1,
            value_stride2);
        const int exp = (bits >> 7) & 0xff;
        local_min = min(local_min, exp);
        local_max = max(local_max, exp);
      }
      for (int offset = 16; offset > 0; offset >>= 1) {
        local_min = min(local_min,
                        __shfl_down_sync(0xffffffff, local_min, offset));
        local_max = max(local_max,
                        __shfl_down_sync(0xffffffff, local_max, offset));
      }
      const int tile_min = __shfl_sync(0xffffffff, local_min, 0);
      const int tile_max = __shfl_sync(0xffffffff, local_max, 0);
      if (tile_max - tile_min <= 15) {
        if (lane == 0) {
          tile_bases[tile_idx] = static_cast<uint8_t>(tile_min);
          tile_needs_fallback[tile_idx] = 0;
          tile_outlier_counts[tile_idx] = 0;
          tile_slots[tile_idx] = -1;
        }
      } else if (use_single_outlier_fastpath) {
        const int high_base = tile_min;
        const int low_base = max(0, tile_max - 15);
        int high_misses = 0;
        int low_misses = 0;
        for (int elem = lane; elem < kByteV2TileElems; elem += 32) {
          const int row =
              is_value ? elem / kByteV2TileSize : elem % kByteV2TileSize;
          const int dim_in_tile =
              is_value ? elem % kByteV2TileSize : elem / kByteV2TileSize;
          const int dim = d0 + dim_in_tile;
          const uint16_t bits = byte_v2_load_prefill_direct_bits(
              key, value, is_value, token_base + row, kv_head, dim,
              key_stride0, key_stride1, key_stride2, value_stride0,
              value_stride1, value_stride2);
          const int exp = (bits >> 7) & 0xff;
          high_misses += exp > high_base + 15 ? 1 : 0;
          low_misses += exp < low_base ? 1 : 0;
        }
        for (int offset = 16; offset > 0; offset >>= 1) {
          high_misses +=
              __shfl_down_sync(0xffffffff, high_misses, offset);
          low_misses += __shfl_down_sync(0xffffffff, low_misses, offset);
        }
        if (lane == 0) {
          int base = high_base;
          int miss_count = high_misses;
          if (high_misses > 1 && low_misses <= 1) {
            base = low_base;
            miss_count = low_misses;
          }
          const bool use_outlier = miss_count == 1;
          const bool needs_fallback = miss_count > 1;
          tile_bases[tile_idx] = static_cast<uint8_t>(base);
          tile_needs_fallback[tile_idx] = needs_fallback ? 1 : 0;
          tile_outlier_counts[tile_idx] =
              static_cast<uint8_t>(use_outlier ? 1 : 0);
          tile_slots[tile_idx] = -1;
          if (needs_fallback) {
            atomicExch(&block_compressible, 0);
          }
        }
      } else {
        int* hist = tile_hist[warp_id];
        for (int exp = lane; exp < 256; exp += 32) {
          hist[exp] = 0;
        }
        __syncwarp();
        for (int elem = lane; elem < kByteV2TileElems; elem += 32) {
          const int row =
              is_value ? elem / kByteV2TileSize : elem % kByteV2TileSize;
          const int dim_in_tile =
              is_value ? elem % kByteV2TileSize : elem / kByteV2TileSize;
          const int dim = d0 + dim_in_tile;
          const uint16_t bits = byte_v2_load_prefill_direct_bits(
              key, value, is_value, token_base + row, kv_head, dim,
              key_stride0, key_stride1, key_stride2, value_stride0,
              value_stride1, value_stride2);
          const int exp = (bits >> 7) & 0xff;
          atomicAdd(hist + exp, 1);
        }
        __syncwarp();
        if (lane == 0) {
          int covered = 0;
          const int base = byte_v2_best_window_base_from_hist(hist, &covered);
          const int miss_count = kByteV2TileElems - covered;
          const bool use_outlier =
              has_outlier_arena && miss_count > lossy_max_misses_per_tile &&
              miss_count <= outlier_max_per_tile;
          tile_bases[tile_idx] = static_cast<uint8_t>(base);
          const bool needs_fallback =
              miss_count > lossy_max_misses_per_tile && !use_outlier;
          tile_needs_fallback[tile_idx] = needs_fallback ? 1 : 0;
          tile_outlier_counts[tile_idx] =
              static_cast<uint8_t>(use_outlier ? miss_count : 0);
          tile_slots[tile_idx] = -1;
          if (needs_fallback) {
            atomicExch(&block_compressible, 0);
          }
        }
      }
    } else {
      int local_min = 255;
      int local_max = 0;
      for (int elem = lane; elem < kByteV2TileElems; elem += 32) {
        const int row = elem / kByteV2TileSize;
        const int dim = d0 + (elem % kByteV2TileSize);
        const uint16_t bits = byte_v2_load_prefill_direct_bits(
            key, value, is_value, token_base + row, kv_head, dim, key_stride0,
            key_stride1, key_stride2, value_stride0, value_stride1,
            value_stride2);
        const int exp = (bits >> 7) & 0xff;
        local_min = min(local_min, exp);
        local_max = max(local_max, exp);
      }
      for (int offset = 16; offset > 0; offset >>= 1) {
        local_min = min(local_min,
                        __shfl_down_sync(0xffffffff, local_min, offset));
        local_max = max(local_max,
                        __shfl_down_sync(0xffffffff, local_max, offset));
      }
      if (lane == 0) {
        tile_bases[tile_idx] = static_cast<uint8_t>(local_min);
        const bool needs_fallback = local_max - local_min > 15;
        tile_needs_fallback[tile_idx] = needs_fallback ? 1 : 0;
        tile_outlier_counts[tile_idx] = 0;
        tile_slots[tile_idx] = -1;
        if (needs_fallback) {
          atomicExch(&block_compressible, 0);
        }
      }
    }
  }
  __syncthreads();

	  if (!block_compressible && v3_outlier_only_no_fallback) {
	    if (tid == 0) {
	      byte_v2_record_cache_update_error(
	          result, kByteV2CacheUpdateErrorOutlierArenaExhausted);
	    }
	    return;
	  }

	  if (!block_compressible && !has_tile_fallback) {
    if (tid == 0) {
      if (fallback_block_ids == nullptr || fallback_next_slot == nullptr) {
        byte_v2_record_cache_update_error(
            result, kByteV2CacheUpdateErrorFallbackPoolMissing);
      } else if (fallback_pool_blocks <= 0) {
        byte_v2_record_cache_update_error(
            result, kByteV2CacheUpdateErrorFallbackPoolExhausted);
      } else if (fallback_pool == nullptr) {
        byte_v2_record_cache_update_error(
            result, kByteV2CacheUpdateErrorFallbackPoolMissing);
      } else {
        int fallback_slot = fallback_block_ids[block_id];
        if (fallback_slot < 0) {
          fallback_slot = atomicAdd(fallback_next_slot, 1);
          if (fallback_slot >= fallback_pool_blocks) {
            fallback_block_ids[block_id] = -1;
            byte_v2_record_cache_update_error(
                result, kByteV2CacheUpdateErrorFallbackPoolExhausted);
            fallback_slot = -1;
          } else {
            fallback_block_ids[block_id] = fallback_slot;
          }
        }
        if (fallback_slot >= fallback_pool_blocks) {
          byte_v2_record_cache_update_error(
              result, kByteV2CacheUpdateErrorFallbackPoolInvalidSlot);
          fallback_slot = -1;
        }
        fallback_slot_shared = fallback_slot;
      }
    }
    __syncthreads();
    if (fallback_slot_shared < 0) {
      return;
    }

    uint8_t* fallback_block =
        fallback_pool + fallback_slot_shared * raw_block_bytes;
    const int key_elems = kByteV2TileSize * num_kv_heads * head_size;
    const int value_elems = kByteV2TileSize * num_kv_heads * head_size_v;
    for (int elem = tid; elem < key_elems; elem += blockDim.x) {
      const int row = elem / (num_kv_heads * head_size);
      const int rem = elem % (num_kv_heads * head_size);
      const int kv_head = rem / head_size;
      const int dim = rem % head_size;
      const uint16_t bits = byte_v2_load_prefill_direct_bits(
          key, value, false, token_base + row, kv_head, dim, key_stride0,
          key_stride1, key_stride2, value_stride0, value_stride1,
          value_stride2);
      byte_v2_store_raw_bits_to_block(fallback_block, false, row, kv_head, dim,
                                      num_kv_heads, head_size, head_size_v,
                                      bits);
    }
    for (int elem = tid; elem < value_elems; elem += blockDim.x) {
      const int row = elem / (num_kv_heads * head_size_v);
      const int rem = elem % (num_kv_heads * head_size_v);
      const int kv_head = rem / head_size_v;
      const int dim = rem % head_size_v;
      const uint16_t bits = byte_v2_load_prefill_direct_bits(
          key, value, true, token_base + row, kv_head, dim, key_stride0,
          key_stride1, key_stride2, value_stride0, value_stride1,
          value_stride2);
      byte_v2_store_raw_bits_to_block(fallback_block, true, row, kv_head, dim,
                                      num_kv_heads, head_size, head_size_v,
                                      bits);
    }
    __syncthreads();
    if (tid == 0) {
      page[kByteV2PageStatusOffset] = kByteV2PageStatusRawFallback;
      page[kByteV2PageValidRowsOffset] = kByteV2TileSize;
    }
    return;
  }

  for (int tile_idx = warp_id; tile_idx < total_tiles; tile_idx += num_warps) {
    const int kv_head = tile_idx / tiles_per_head;
    const int tile_in_head = tile_idx % tiles_per_head;
    const bool is_value = tile_in_head >= k_dim_tiles;
    const int dim_tile =
        is_value ? tile_in_head - k_dim_tiles : tile_in_head;
    const int d0 = dim_tile * kByteV2TileSize;
    const int tile_start =
        byte_v2_tile_start(is_value, kv_head, dim_tile, k_dim_tiles,
                           v_dim_tiles);
    const int tile_start_v3 =
        use_v3_payload_layout
            ? byte_v2_v3_tile_payload_start(is_value, kv_head, dim_tile,
                                            num_kv_heads, k_dim_tiles,
                                            v_dim_tiles)
            : 0;
    if (tile_outlier_counts[tile_idx] != 0) {
      if (lane == 0) {
        const int count = static_cast<int>(tile_outlier_counts[tile_idx]);
        const int offset = atomicAdd(outlier_next_entry, count);
	        if (offset < 0 || offset + count > outlier_arena_entries ||
	            offset > kByteV2MaxOutlierArenaOffset) {
	          tile_needs_fallback[tile_idx] = 1;
	          tile_slots[tile_idx] = -1;
	          outlier_tile_meta[block_id * total_tiles + tile_idx] = -1;
	          if (v3_outlier_only_no_fallback) {
	            atomicExch(&block_compressible, 0);
	            byte_v2_record_cache_update_error(
	                result, kByteV2CacheUpdateErrorOutlierArenaExhausted);
	          }
	        } else {
          tile_slots[tile_idx] = offset;
          outlier_tile_meta[block_id * total_tiles + tile_idx] =
              byte_v2_pack_outlier_meta(offset, count);
          byte_v2_set_outlier_tile_bitmap(outlier_tile_bitmap, block_id,
                                          bitmap_words, tile_idx);
        }
      }
      __syncwarp();
	    }
	    if (tile_needs_fallback[tile_idx] != 0) {
	      if (!has_tile_fallback) {
	        if (lane == 0) {
	          if (outlier_tile_meta != nullptr) {
	            outlier_tile_meta[block_id * total_tiles + tile_idx] = -1;
	          }
	          byte_v2_record_cache_update_error(
	              result,
	              v3_outlier_only_no_fallback
	                  ? kByteV2CacheUpdateErrorOutlierArenaExhausted
	                  : kByteV2CacheUpdateErrorFallbackPoolMissing);
	          atomicExch(&block_compressible, 0);
	        }
	        continue;
	      }
	      if (lane == 0) {
        if (outlier_tile_meta != nullptr) {
          outlier_tile_meta[block_id * total_tiles + tile_idx] = -1;
        }
        if (tile_capacity <= 0) {
          byte_v2_record_cache_update_error(
              result, kByteV2CacheUpdateErrorFallbackPoolExhausted);
        } else {
          const int tile_slot = byte_v2_allocate_tile_fallback_slot(
              fallback_tile_next_slot, tile_capacity);
          if (tile_slot < 0) {
            byte_v2_record_cache_update_error(
                result, kByteV2CacheUpdateErrorFallbackPoolExhausted);
          } else {
            tile_slots[tile_idx] = tile_slot;
            fallback_tile_ids[block_id * total_tiles + tile_idx] = tile_slot;
            if (!use_v3_payload_layout) {
              page[tile_start] = 0;
              page[tile_start + 1] = 1;
            }
          }
        }
      }
      __syncwarp();
      const int tile_slot = tile_slots[tile_idx];
      if (tile_slot >= 0) {
        for (int elem = lane; elem < kByteV2TileElems; elem += 32) {
          const int row = elem / kByteV2TileSize;
          const int dim_in_tile = elem % kByteV2TileSize;
          const uint16_t bits = byte_v2_load_prefill_direct_bits(
              key, value, is_value, token_base + row, kv_head,
              d0 + dim_in_tile, key_stride0, key_stride1, key_stride2,
              value_stride0, value_stride1, value_stride2);
          byte_v2_store_raw_bits_to_tile_pool(fallback_pool, tile_slot, row,
                                              dim_in_tile, bits);
        }
      }
      continue;
    }
    const int base = static_cast<int>(tile_bases[tile_idx]);
    const bool use_outlier = tile_outlier_counts[tile_idx] != 0;
    const int outlier_offset = use_outlier ? tile_slots[tile_idx] : -1;
    if (lane == 0) {
      if (fallback_tile_ids != nullptr) {
        fallback_tile_ids[block_id * total_tiles + tile_idx] = -1;
      }
      if (!use_outlier && outlier_tile_meta != nullptr) {
        outlier_tile_meta[block_id * total_tiles + tile_idx] = -1;
      }
      if (use_v3_payload_layout) {
        const int meta_start = byte_v2_v3_kv_head_meta_start(kv_head);
        page[meta_start + (is_value ? 8 + dim_tile : dim_tile)] =
            static_cast<uint8_t>(base);
      } else {
        page[tile_start] = static_cast<uint8_t>(base);
        page[tile_start + 1] = 0;
      }
      if (use_outlier && !use_single_outlier_fastpath) {
        atomicExch(&block_has_outlier, 1);
        int written = 0;
        for (int elem = 0; elem < kByteV2TileElems; ++elem) {
          const int row =
              is_value ? elem / kByteV2TileSize : elem % kByteV2TileSize;
          const int dim_in_tile =
              is_value ? elem % kByteV2TileSize : elem / kByteV2TileSize;
          const int dim = d0 + dim_in_tile;
          const uint16_t bits = byte_v2_load_prefill_direct_bits(
              key, value, is_value, token_base + row, kv_head, dim,
              key_stride0, key_stride1, key_stride2, value_stride0,
              value_stride1, value_stride2);
          const int exp = (bits >> 7) & 0xff;
          if (!byte_v2_exp_in_window(exp, base)) {
            outlier_arena[outlier_offset + written] =
                byte_v2_pack_outlier_entry(elem, bits);
            ++written;
          }
        }
      }
    }
    if (use_outlier && use_single_outlier_fastpath) {
      if (lane == 0) {
        atomicExch(&block_has_outlier, 1);
      }
      int found_elem = kByteV2TileElems;
      int found_bits = 0;
      for (int elem = lane; elem < kByteV2TileElems; elem += 32) {
        const int row =
            is_value ? elem / kByteV2TileSize : elem % kByteV2TileSize;
        const int dim_in_tile =
            is_value ? elem % kByteV2TileSize : elem / kByteV2TileSize;
        const int dim = d0 + dim_in_tile;
        const uint16_t bits = byte_v2_load_prefill_direct_bits(
            key, value, is_value, token_base + row, kv_head, dim,
            key_stride0, key_stride1, key_stride2, value_stride0,
            value_stride1, value_stride2);
        const int exp = (bits >> 7) & 0xff;
        if (!byte_v2_exp_in_window(exp, base) && elem < found_elem) {
          found_elem = elem;
          found_bits = static_cast<int>(bits);
        }
      }
      for (int offset = 16; offset > 0; offset >>= 1) {
        const int other_elem =
            __shfl_down_sync(0xffffffff, found_elem, offset);
        const int other_bits =
            __shfl_down_sync(0xffffffff, found_bits, offset);
        if (other_elem < found_elem) {
          found_elem = other_elem;
          found_bits = other_bits;
        }
      }
      if (lane == 0 && found_elem < kByteV2TileElems) {
        outlier_arena[outlier_offset] =
            byte_v2_pack_outlier_entry(
                found_elem, static_cast<uint16_t>(found_bits));
      }
    }

    for (int packed_idx = lane; packed_idx < kByteV2PackedTileElems;
         packed_idx += 32) {
      const int elem0 = packed_idx * 2;
      const int row0 =
          is_value ? elem0 / kByteV2TileSize : elem0 % kByteV2TileSize;
      const int dim_in_tile0 =
          is_value ? elem0 % kByteV2TileSize : elem0 / kByteV2TileSize;
      const int elem1 = elem0 + 1;
      const int row1 =
          is_value ? elem1 / kByteV2TileSize : elem1 % kByteV2TileSize;
      const int dim_in_tile1 =
          is_value ? elem1 % kByteV2TileSize : elem1 / kByteV2TileSize;
      const uint16_t bits0 = byte_v2_load_prefill_direct_bits(
          key, value, is_value, token_base + row0, kv_head,
          d0 + dim_in_tile0, key_stride0, key_stride1, key_stride2,
          value_stride0, value_stride1, value_stride2);
      const uint16_t bits1 = byte_v2_load_prefill_direct_bits(
          key, value, is_value, token_base + row1, kv_head,
          d0 + dim_in_tile1, key_stride0, key_stride1, key_stride2,
          value_stride0, value_stride1, value_stride2);
      const int exp0 = (bits0 >> 7) & 0xff;
      const int exp1 = (bits1 >> 7) & 0xff;
      const int stored_exp0 =
          (lossy_max_misses_per_tile > 0 || use_outlier)
              ? byte_v2_clamp_exp_to_window(exp0, base)
              : exp0;
      const int stored_exp1 =
          (lossy_max_misses_per_tile > 0 || use_outlier)
              ? byte_v2_clamp_exp_to_window(exp1, base)
              : exp1;
      const uint8_t low0 =
          static_cast<uint8_t>((bits0 & 0x7f) | ((stored_exp0 & 1) << 7));
      const uint8_t low1 =
          static_cast<uint8_t>((bits1 & 0x7f) | ((stored_exp1 & 1) << 7));
      uint8_t packed = 0;
      const int delta0 = stored_exp0 - base;
      const int sign0 = (bits0 >> 15) & 1;
      packed |= static_cast<uint8_t>(
          ((sign0 << 3) | ((delta0 >> 1) & 0x07)) & 0x0f);
      const int delta1 = stored_exp1 - base;
      const int sign1 = (bits1 >> 15) & 1;
      packed |= static_cast<uint8_t>(
          (((sign1 << 3) | ((delta1 >> 1) & 0x07)) & 0x0f) << 4);

      if (use_v3_payload_layout) {
        const int stripe = packed_idx / kByteV2TilePayloadStripePairsV3;
        const int stripe_lane = packed_idx % kByteV2TilePayloadStripePairsV3;
        uint8_t* __restrict__ stripe_ptr =
            page + tile_start_v3 + stripe * kByteV2TilePayloadStripeBytesV3;
        stripe_ptr[stripe_lane] = low0;
        stripe_ptr[32 + stripe_lane] = low1;
        stripe_ptr[64 + stripe_lane] = packed;
      } else {
        page[tile_start + 2 + elem0] = low0;
        page[tile_start + 2 + elem1] = low1;
        page[tile_start + 2 + kByteV2TileElems + packed_idx] = packed;
      }
    }
	  }
	  __syncthreads();

	  if (!block_compressible && !has_tile_fallback) {
	    if (tid == 0) {
	      page[kByteV2PageStatusOffset] = kByteV2PageStatusRawFallback;
	      page[kByteV2PageValidRowsOffset] = kByteV2TileSize;
	    }
	    return;
	  }

	  if (tid == 0) {
    if (use_v3_payload_layout) {
      for (int kv_head = 0; kv_head < num_kv_heads; ++kv_head) {
        uint16_t k_fallback_mask = 0;
        uint16_t v_fallback_mask = 0;
        for (int dim_tile = 0; dim_tile < k_dim_tiles; ++dim_tile) {
          const int tile_idx = byte_v2_tile_index(
              false, kv_head, dim_tile, k_dim_tiles, v_dim_tiles);
          if (tile_needs_fallback[tile_idx] != 0) {
            k_fallback_mask |= static_cast<uint16_t>(1u << dim_tile);
          }
        }
        for (int dim_tile = 0; dim_tile < v_dim_tiles; ++dim_tile) {
          const int tile_idx = byte_v2_tile_index(
              true, kv_head, dim_tile, k_dim_tiles, v_dim_tiles);
          if (tile_needs_fallback[tile_idx] != 0) {
            v_fallback_mask |= static_cast<uint16_t>(1u << dim_tile);
          }
        }
        const int meta_start = byte_v2_v3_kv_head_meta_start(kv_head);
        byte_v2_store_u16(page + meta_start + 16, k_fallback_mask);
        byte_v2_store_u16(page + meta_start + 18, v_fallback_mask);
      }
    }
    page[kByteV2PageStatusOffset] = kByteV2PageStatusCompressed;
    page[kByteV2PageValidRowsOffset] = kByteV2TileSize;
    if (fallback_block_ids != nullptr) {
      fallback_block_ids[block_id] = -1;
    }
    if (outlier_block_flags != nullptr) {
      outlier_block_flags[block_id] = block_has_outlier;
    }
    result[group_idx + 1] = block_id;
  }
}

__global__ void byte_v2_compress_touched_pages_kernel(
    const uint8_t* __restrict__ key, const uint8_t* __restrict__ value,
    uint8_t* __restrict__ kv_cache, const int64_t* __restrict__ slot_mapping,
    const int32_t* __restrict__ valid_rows,
    const uint8_t* __restrict__ touched_flags, uint8_t* __restrict__ packed_flags,
    uint8_t* __restrict__ raw_staging, uint8_t* __restrict__ fallback_pool,
    int32_t* __restrict__ fallback_block_ids,
    int32_t* __restrict__ fallback_next_slot, int32_t* __restrict__ error,
    const int num_tokens, const int num_blocks, const int num_kv_heads,
    const int head_size, const int head_size_v, const int page_size_bytes,
    const int raw_block_bytes, const int fallback_pool_blocks,
    const int lossy_max_misses_per_tile,
    const int64_t key_stride0, const int64_t key_stride1, const int64_t key_stride2,
    const int64_t value_stride0, const int64_t value_stride1,
    const int64_t value_stride2) {
  const int token_idx = blockIdx.x;
  if (token_idx >= num_tokens) {
    return;
  }
  const int64_t slot = slot_mapping[token_idx];
  if (slot < 0) {
    return;
  }
  const int block_id = static_cast<int>(slot / kByteV2TileSize);
  if (block_id < 0 || block_id >= num_blocks || touched_flags[block_id] == 0) {
    return;
  }
  for (int prev = 0; prev < token_idx; ++prev) {
    const int64_t prev_slot = slot_mapping[prev];
    if (prev_slot >= 0 &&
        static_cast<int>(prev_slot / kByteV2TileSize) == block_id) {
      return;
    }
  }

  uint8_t* page = kv_cache + block_id * page_size_bytes;
  uint8_t* raw_block = raw_staging + token_idx * raw_block_bytes;
  for (int byte_idx = threadIdx.x; byte_idx < raw_block_bytes;
       byte_idx += blockDim.x) {
    raw_block[byte_idx] = 0;
  }
  __syncthreads();

  const uint8_t status = page[kByteV2PageStatusOffset];
  const int existing_valid_rows =
      static_cast<int>(page[kByteV2PageValidRowsOffset]);
  if (status == kByteV2PageStatusCompressed) {
    if (existing_valid_rows <= 0 || existing_valid_rows > kByteV2TileSize) {
      if (threadIdx.x == 0) {
        byte_v2_record_cache_update_error(
            error, kByteV2CacheUpdateErrorInvalidValidRows);
      }
      return;
    }
    if (existing_valid_rows == kByteV2TileSize) {
      if (threadIdx.x == 0) {
        byte_v2_record_cache_update_error(
            error, kByteV2CacheUpdateErrorFinalizedBlockUpdate);
      }
      return;
    }
    byte_v2_decompress_page_to_raw_block(page, raw_block, existing_valid_rows,
                                         num_kv_heads, head_size, head_size_v);
  } else if (status == kByteV2PageStatusRawFallback) {
    if (existing_valid_rows <= 0 || existing_valid_rows > kByteV2TileSize) {
      if (threadIdx.x == 0) {
        byte_v2_record_cache_update_error(
            error, kByteV2CacheUpdateErrorInvalidValidRows);
      }
      return;
    }
    if (existing_valid_rows == kByteV2TileSize) {
      if (threadIdx.x == 0) {
        byte_v2_record_cache_update_error(
            error, kByteV2CacheUpdateErrorFinalizedBlockUpdate);
      }
      return;
    }
    if (fallback_pool == nullptr || fallback_block_ids == nullptr) {
      if (threadIdx.x == 0) {
        byte_v2_record_cache_update_error(
            error, kByteV2CacheUpdateErrorFallbackPoolMissing);
      }
      return;
    }
    const int fallback_slot = fallback_block_ids[block_id];
    if (fallback_slot < 0 || fallback_slot >= fallback_pool_blocks) {
      if (threadIdx.x == 0) {
        byte_v2_record_cache_update_error(
            error, kByteV2CacheUpdateErrorFallbackPoolInvalidSlot);
      }
      return;
    }
    const uint8_t* fallback_block =
        fallback_pool + fallback_slot * raw_block_bytes;
    for (int byte_idx = threadIdx.x; byte_idx < raw_block_bytes;
         byte_idx += blockDim.x) {
      raw_block[byte_idx] = fallback_block[byte_idx];
    }
  } else if (status != kByteV2PageStatusEmpty) {
    if (threadIdx.x == 0) {
      byte_v2_record_cache_update_error(
          error, kByteV2CacheUpdateErrorInvalidPageState);
    }
    return;
  }
  __syncthreads();

  const int key_elems = num_kv_heads * head_size;
  const int value_elems = num_kv_heads * head_size_v;
  for (int src_idx = 0; src_idx < num_tokens; ++src_idx) {
    const int64_t src_slot = slot_mapping[src_idx];
    if (src_slot < 0 ||
        static_cast<int>(src_slot / kByteV2TileSize) != block_id) {
      continue;
    }
    const int block_offset = static_cast<int>(src_slot % kByteV2TileSize);
    for (int elem = threadIdx.x; elem < key_elems; elem += blockDim.x) {
      const int kv_head = elem / head_size;
      const int dim = elem % head_size;
      const int64_t src_elem =
          (src_idx * key_stride0 + kv_head * key_stride1 + dim * key_stride2) *
          2;
      const uint16_t bits = byte_v2_load_u16(key + src_elem);
      byte_v2_store_raw_bits_to_block(raw_block, false, block_offset, kv_head,
                                      dim, num_kv_heads, head_size,
                                      head_size_v, bits);
    }
    for (int elem = threadIdx.x; elem < value_elems; elem += blockDim.x) {
      const int kv_head = elem / head_size_v;
      const int dim = elem % head_size_v;
      const int64_t src_elem =
          (src_idx * value_stride0 + kv_head * value_stride1 +
           dim * value_stride2) *
          2;
      const uint16_t bits = byte_v2_load_u16(value + src_elem);
      byte_v2_store_raw_bits_to_block(raw_block, true, block_offset, kv_head,
                                      dim, num_kv_heads, head_size,
                                      head_size_v, bits);
    }
  }
  __syncthreads();

  if (threadIdx.x != 0) {
    return;
  }
  const int valid = valid_rows[block_id];
  if (valid <= 0 || valid > kByteV2TileSize) {
    byte_v2_record_cache_update_error(
        error, kByteV2CacheUpdateErrorInvalidValidRows);
    return;
  }
  if (!byte_v2_raw_block_is_compressible(raw_block, num_kv_heads, head_size,
                                         head_size_v, valid,
                                         lossy_max_misses_per_tile)) {
    if (fallback_block_ids == nullptr || fallback_next_slot == nullptr) {
      byte_v2_record_cache_update_error(
          error, kByteV2CacheUpdateErrorFallbackPoolMissing);
      return;
    }
    if (fallback_pool_blocks <= 0) {
      byte_v2_record_cache_update_error(
          error, kByteV2CacheUpdateErrorFallbackPoolExhausted);
      return;
    }
    if (fallback_pool == nullptr) {
      byte_v2_record_cache_update_error(
          error, kByteV2CacheUpdateErrorFallbackPoolMissing);
      return;
    }
    int fallback_slot = fallback_block_ids[block_id];
    if (fallback_slot < 0) {
      fallback_slot = atomicAdd(fallback_next_slot, 1);
      if (fallback_slot >= fallback_pool_blocks) {
        fallback_block_ids[block_id] = -1;
        byte_v2_record_cache_update_error(
            error, kByteV2CacheUpdateErrorFallbackPoolExhausted);
        return;
      }
      fallback_block_ids[block_id] = fallback_slot;
    }
    if (fallback_slot >= fallback_pool_blocks) {
      byte_v2_record_cache_update_error(
          error, kByteV2CacheUpdateErrorFallbackPoolInvalidSlot);
      return;
    }
    uint8_t* fallback_block = fallback_pool + fallback_slot * raw_block_bytes;
    for (int byte_idx = 0; byte_idx < raw_block_bytes; ++byte_idx) {
      fallback_block[byte_idx] = raw_block[byte_idx];
    }
    page[kByteV2PageStatusOffset] = kByteV2PageStatusRawFallback;
    page[kByteV2PageValidRowsOffset] = static_cast<uint8_t>(valid);
    return;
  }
  byte_v2_store_compressed_block(raw_block, page, num_kv_heads, head_size,
                                 head_size_v, valid,
                                 lossy_max_misses_per_tile);
  page[kByteV2PageStatusOffset] = kByteV2PageStatusCompressed;
  page[kByteV2PageValidRowsOffset] = static_cast<uint8_t>(valid);
  if (fallback_block_ids != nullptr) {
    fallback_block_ids[block_id] = -1;
  }
  if (valid == kByteV2TileSize) {
    packed_flags[block_id] = 1;
  }
}

__global__ void byte_v2_compress_touched_blocks_kernel(
    const uint8_t* __restrict__ key, const uint8_t* __restrict__ value,
    uint8_t* __restrict__ kv_cache,
    const int32_t* __restrict__ block_token_indices,
    const int32_t* __restrict__ valid_rows,
    const uint8_t* __restrict__ touched_flags,
    const uint8_t* __restrict__ overwrite_flags,
    uint8_t* __restrict__ packed_flags,
    uint8_t* __restrict__ raw_staging, uint8_t* __restrict__ fallback_pool,
    int32_t* __restrict__ fallback_block_ids,
    int32_t* __restrict__ fallback_next_slot,
    int32_t* __restrict__ fallback_tile_ids,
    int32_t* __restrict__ fallback_tile_next_slot,
    int32_t* __restrict__ outlier_block_flags,
    int32_t* __restrict__ outlier_tile_bitmap,
    int32_t* __restrict__ error,
    const int num_tokens, const int num_blocks, const int num_kv_heads,
    const int head_size, const int head_size_v, const int page_size_bytes,
    const int raw_block_bytes, const int fallback_pool_blocks,
    const int lossy_max_misses_per_tile,
    const int64_t key_stride0, const int64_t key_stride1, const int64_t key_stride2,
    const int64_t value_stride0, const int64_t value_stride1,
    const int64_t value_stride2) {
  const int block_id = blockIdx.x;
  if (block_id >= num_blocks || touched_flags[block_id] == 0) {
    return;
  }

  const int32_t* block_tokens =
      block_token_indices + block_id * kByteV2TileSize;
  int first_token_idx = -1;
  int max_touched_rows = 0;
  for (int offset = 0; offset < kByteV2TileSize; ++offset) {
    const int token_idx = block_tokens[offset];
    if (token_idx >= 0) {
      max_touched_rows = offset + 1;
      first_token_idx = token_idx;
      break;
    }
  }
  for (int offset = max_touched_rows; offset < kByteV2TileSize; ++offset) {
    if (block_tokens[offset] >= 0) {
      max_touched_rows = offset + 1;
    }
  }
  if (first_token_idx < 0 || first_token_idx >= num_tokens) {
    if (threadIdx.x == 0) {
      byte_v2_record_cache_update_error(
          error, kByteV2CacheUpdateErrorNoTouchedToken);
    }
    return;
  }

  uint8_t* page = kv_cache + block_id * page_size_bytes;
  uint8_t* raw_block = raw_staging + first_token_idx * raw_block_bytes;
  for (int byte_idx = threadIdx.x; byte_idx < raw_block_bytes;
       byte_idx += blockDim.x) {
    raw_block[byte_idx] = 0;
  }
  __syncthreads();

  const uint8_t status = page[kByteV2PageStatusOffset];
  const int existing_valid_rows =
      static_cast<int>(page[kByteV2PageValidRowsOffset]);
  const bool overwrite_block = overwrite_flags[block_id] != 0;
  if (threadIdx.x == 0 && outlier_block_flags != nullptr) {
    outlier_block_flags[block_id] = 0;
  }
  const int total_tiles =
      byte_v2_total_tiles(num_kv_heads, head_size, head_size_v);
  const int bitmap_words = byte_v2_outlier_tile_bitmap_words(total_tiles);
  if (outlier_tile_bitmap != nullptr) {
    for (int word = threadIdx.x; word < bitmap_words; word += blockDim.x) {
      outlier_tile_bitmap[block_id * bitmap_words + word] = 0;
    }
  }
  if (status == kByteV2PageStatusCompressed) {
    if (!overwrite_block &&
        (existing_valid_rows <= 0 || existing_valid_rows > kByteV2TileSize)) {
      if (threadIdx.x == 0) {
        byte_v2_record_cache_update_error(
            error, kByteV2CacheUpdateErrorInvalidValidRows);
      }
      return;
    }
    if (!overwrite_block && existing_valid_rows == kByteV2TileSize) {
      if (threadIdx.x == 0) {
        byte_v2_record_cache_update_error(
            error, kByteV2CacheUpdateErrorFinalizedBlockUpdate);
      }
      return;
    }
    if (!overwrite_block) {
      byte_v2_decompress_page_to_raw_block(
          page, raw_block, fallback_pool, fallback_tile_ids, nullptr, nullptr,
          nullptr, block_id, existing_valid_rows, num_kv_heads, head_size,
          head_size_v);
    }
  } else if (status == kByteV2PageStatusRawFallback) {
    if (!overwrite_block &&
        (existing_valid_rows <= 0 || existing_valid_rows > kByteV2TileSize)) {
      if (threadIdx.x == 0) {
        byte_v2_record_cache_update_error(
            error, kByteV2CacheUpdateErrorInvalidValidRows);
      }
      return;
    }
    if (!overwrite_block && existing_valid_rows == kByteV2TileSize) {
      if (threadIdx.x == 0) {
        byte_v2_record_cache_update_error(
            error, kByteV2CacheUpdateErrorFinalizedBlockUpdate);
      }
      return;
    }
    if (!overwrite_block &&
        (fallback_pool == nullptr || fallback_block_ids == nullptr)) {
      if (threadIdx.x == 0) {
        byte_v2_record_cache_update_error(
            error, kByteV2CacheUpdateErrorFallbackPoolMissing);
      }
      return;
    }
    if (!overwrite_block) {
      const int fallback_slot = fallback_block_ids[block_id];
      if (fallback_slot < 0 || fallback_slot >= fallback_pool_blocks) {
        if (threadIdx.x == 0) {
          byte_v2_record_cache_update_error(
              error, kByteV2CacheUpdateErrorFallbackPoolInvalidSlot);
        }
        return;
      }
      const uint8_t* fallback_block =
          fallback_pool + fallback_slot * raw_block_bytes;
      for (int byte_idx = threadIdx.x; byte_idx < raw_block_bytes;
           byte_idx += blockDim.x) {
        raw_block[byte_idx] = fallback_block[byte_idx];
      }
    }
  } else if (status != kByteV2PageStatusEmpty) {
    if (threadIdx.x == 0) {
      byte_v2_record_cache_update_error(
          error, kByteV2CacheUpdateErrorInvalidPageState);
    }
    return;
  }
  __syncthreads();

  const int key_elems = num_kv_heads * head_size;
  const int value_elems = num_kv_heads * head_size_v;
  for (int block_offset = 0; block_offset < kByteV2TileSize; ++block_offset) {
    const int token_idx = block_tokens[block_offset];
    if (token_idx < 0) {
      continue;
    }
    for (int elem = threadIdx.x; elem < key_elems; elem += blockDim.x) {
      const int kv_head = elem / head_size;
      const int dim = elem % head_size;
      const int64_t src_elem =
          (token_idx * key_stride0 + kv_head * key_stride1 + dim * key_stride2) *
          2;
      const uint16_t bits = byte_v2_load_u16(key + src_elem);
      byte_v2_store_raw_bits_to_block(raw_block, false, block_offset, kv_head,
                                      dim, num_kv_heads, head_size,
                                      head_size_v, bits);
    }
    for (int elem = threadIdx.x; elem < value_elems; elem += blockDim.x) {
      const int kv_head = elem / head_size_v;
      const int dim = elem % head_size_v;
      const int64_t src_elem =
          (token_idx * value_stride0 + kv_head * value_stride1 +
           dim * value_stride2) *
          2;
      const uint16_t bits = byte_v2_load_u16(value + src_elem);
      byte_v2_store_raw_bits_to_block(raw_block, true, block_offset, kv_head,
                                      dim, num_kv_heads, head_size,
                                      head_size_v, bits);
    }
  }
  __syncthreads();

  if (threadIdx.x != 0) {
    return;
  }
  const int valid = overwrite_block ? max_touched_rows : valid_rows[block_id];
  if (valid <= 0 || valid > kByteV2TileSize) {
    byte_v2_record_cache_update_error(
        error, kByteV2CacheUpdateErrorInvalidValidRows);
    return;
  }
  const bool use_v3_payload_layout = byte_v2_page_uses_v3_layout(
      page_size_bytes, num_kv_heads, head_size, head_size_v);
  if (!byte_v2_raw_block_is_compressible(raw_block, num_kv_heads, head_size,
                                         head_size_v, valid,
                                         lossy_max_misses_per_tile)) {
    if (fallback_tile_ids != nullptr && fallback_tile_next_slot != nullptr) {
      const bool stored = use_v3_payload_layout
                              ? byte_v2_store_compressed_or_tile_fallback_block_v3(
                                    raw_block, page, fallback_pool,
                                    fallback_tile_ids,
                                    fallback_tile_next_slot, error, block_id,
                                    num_kv_heads, head_size, head_size_v,
                                    valid, raw_block_bytes,
                                    fallback_pool_blocks,
                                    lossy_max_misses_per_tile)
                              : byte_v2_store_compressed_or_tile_fallback_block(
                                    raw_block, page, fallback_pool,
                                    fallback_tile_ids,
                                    fallback_tile_next_slot, error, block_id,
                                    num_kv_heads, head_size, head_size_v,
                                    valid, raw_block_bytes,
                                    fallback_pool_blocks,
                                    lossy_max_misses_per_tile);
      if (stored) {
        if (fallback_block_ids != nullptr) {
          fallback_block_ids[block_id] = -1;
        }
        if (valid == kByteV2TileSize) {
          packed_flags[block_id] = 1;
        }
      }
      return;
    }
    if (fallback_block_ids == nullptr || fallback_next_slot == nullptr) {
      byte_v2_record_cache_update_error(
          error, kByteV2CacheUpdateErrorFallbackPoolMissing);
      return;
    }
    if (fallback_pool_blocks <= 0) {
      byte_v2_record_cache_update_error(
          error, kByteV2CacheUpdateErrorFallbackPoolExhausted);
      return;
    }
    if (fallback_pool == nullptr) {
      byte_v2_record_cache_update_error(
          error, kByteV2CacheUpdateErrorFallbackPoolMissing);
      return;
    }
    int fallback_slot = fallback_block_ids[block_id];
    if (fallback_slot < 0) {
      fallback_slot = atomicAdd(fallback_next_slot, 1);
      if (fallback_slot >= fallback_pool_blocks) {
        fallback_block_ids[block_id] = -1;
        byte_v2_record_cache_update_error(
            error, kByteV2CacheUpdateErrorFallbackPoolExhausted);
        return;
      }
      fallback_block_ids[block_id] = fallback_slot;
    }
    if (fallback_slot >= fallback_pool_blocks) {
      byte_v2_record_cache_update_error(
          error, kByteV2CacheUpdateErrorFallbackPoolInvalidSlot);
      return;
    }
    uint8_t* fallback_block = fallback_pool + fallback_slot * raw_block_bytes;
    for (int byte_idx = 0; byte_idx < raw_block_bytes; ++byte_idx) {
      fallback_block[byte_idx] = raw_block[byte_idx];
    }
    page[kByteV2PageStatusOffset] = kByteV2PageStatusRawFallback;
    page[kByteV2PageValidRowsOffset] = static_cast<uint8_t>(valid);
    return;
  }
  if (fallback_tile_ids != nullptr) {
    const bool stored = use_v3_payload_layout
                            ? byte_v2_store_compressed_or_tile_fallback_block_v3(
                                  raw_block, page, fallback_pool,
                                  fallback_tile_ids, fallback_tile_next_slot,
                                  error, block_id, num_kv_heads, head_size,
                                  head_size_v, valid, raw_block_bytes,
                                  fallback_pool_blocks,
                                  lossy_max_misses_per_tile)
                            : byte_v2_store_compressed_or_tile_fallback_block(
                                  raw_block, page, fallback_pool,
                                  fallback_tile_ids, fallback_tile_next_slot,
                                  error, block_id, num_kv_heads, head_size,
                                  head_size_v, valid, raw_block_bytes,
                                  fallback_pool_blocks,
                                  lossy_max_misses_per_tile);
    if (!stored) {
      return;
    }
  } else {
    if (use_v3_payload_layout) {
      byte_v2_store_compressed_block_v3(raw_block, page, num_kv_heads,
                                        head_size, head_size_v, valid,
                                        lossy_max_misses_per_tile);
    } else {
      byte_v2_store_compressed_block(raw_block, page, num_kv_heads, head_size,
                                     head_size_v, valid,
                                     lossy_max_misses_per_tile);
    }
    page[kByteV2PageStatusOffset] = kByteV2PageStatusCompressed;
    page[kByteV2PageValidRowsOffset] = static_cast<uint8_t>(valid);
  }
  if (fallback_block_ids != nullptr) {
    fallback_block_ids[block_id] = -1;
  }
  if (valid == kByteV2TileSize) {
    packed_flags[block_id] = 1;
  }
}

template <typename block_table_t, typename seq_lens_t>
__global__ void byte_v2_paged_decode_attention_kernel(
    const uint16_t* __restrict__ query, const uint8_t* __restrict__ kv_cache,
    const block_table_t* __restrict__ block_table,
    const seq_lens_t* __restrict__ seq_lens, uint16_t* __restrict__ output,
    const uint8_t* __restrict__ fallback_pool,
    const int32_t* __restrict__ fallback_block_ids,
    const int32_t* __restrict__ fallback_tile_ids,
    const int32_t* __restrict__ outlier_arena,
    const int32_t* __restrict__ outlier_block_flags,
    const int32_t* __restrict__ outlier_tile_bitmap,
    const int32_t* __restrict__ outlier_tile_meta,
    const float scale, const int num_decode_tokens, const int num_heads,
    const int num_kv_heads, const int head_size, const int head_size_v,
    const int page_size_bytes, const int raw_block_bytes,
    const int64_t block_table_stride0, const int64_t block_table_stride1,
    const int64_t seq_lens_stride0) {
  const int req_idx = blockIdx.x;
  const int q_head = blockIdx.y;
  const int tid = threadIdx.x;
  if (req_idx >= num_decode_tokens || q_head >= num_heads) {
    return;
  }

  __shared__ float score_sums[128];
  __shared__ float softmax_factor;
  __shared__ float softmax_weight;
  __shared__ float inv_denom;

  float acc = 0.0f;
  const int seq_len = static_cast<int>(seq_lens[req_idx * seq_lens_stride0]);
  if (seq_len <= 0) {
    for (int d = tid; d < head_size_v; d += blockDim.x) {
      output[(req_idx * num_heads + q_head) * head_size_v + d] =
          byte_v2_float_to_bf16_bits(0.0f);
    }
    return;
  }

  const int q_per_kv = num_heads / num_kv_heads;
  const int kv_head = q_head / q_per_kv;
  float running_max = -FLT_MAX;
  float denom = 0.0f;

  for (int token = 0; token < seq_len; ++token) {
    const int logical_block = token / kByteV2TileSize;
    const int row = token % kByteV2TileSize;
    const int physical_block = static_cast<int>(
        block_table[req_idx * block_table_stride0 +
                    logical_block * block_table_stride1]);
    const uint8_t* page = kv_cache + physical_block * page_size_bytes;
    if (row >= static_cast<int>(page[kByteV2PageValidRowsOffset])) {
      continue;
    }
    const bool block_may_have_outliers =
        byte_v2_may_have_block_outliers(outlier_block_flags, physical_block);
    const int32_t* block_outlier_arena =
        block_may_have_outliers ? outlier_arena : nullptr;
    const int32_t* block_outlier_tile_bitmap =
        block_may_have_outliers ? outlier_tile_bitmap : nullptr;
    const int32_t* block_outlier_tile_meta =
        block_may_have_outliers ? outlier_tile_meta : nullptr;

    float score_part = 0.0f;
    for (int d = tid; d < head_size; d += blockDim.x) {
      const uint16_t q_bits =
          query[(req_idx * num_heads + q_head) * head_size + d];
      const uint16_t k_bits = byte_v2_load_kv_bits(
          page, fallback_pool, fallback_block_ids, fallback_tile_ids,
          block_outlier_arena, block_outlier_tile_bitmap,
          block_outlier_tile_meta,
          physical_block,
          raw_block_bytes, false, row, kv_head, d, num_kv_heads, head_size,
          head_size_v);
      score_part +=
          byte_v2_bf16_bits_to_float(q_bits) * byte_v2_bf16_bits_to_float(k_bits);
    }
    score_sums[tid] = score_part;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
        score_sums[tid] += score_sums[tid + stride];
      }
      __syncthreads();
    }

    if (tid == 0) {
      const float score = score_sums[0] * scale;
      float factor = 1.0f;
      if (score > running_max) {
        factor = expf(running_max - score);
        denom *= factor;
        running_max = score;
      }
      softmax_factor = factor;
      softmax_weight = expf(score - running_max);
      denom += softmax_weight;
    }
    __syncthreads();

    if (tid < head_size_v) {
      const uint16_t v_bits = byte_v2_load_kv_bits(
          page, fallback_pool, fallback_block_ids, fallback_tile_ids,
          block_outlier_arena, block_outlier_tile_bitmap,
          block_outlier_tile_meta,
          physical_block,
          raw_block_bytes, true, row, kv_head, tid, num_kv_heads, head_size,
          head_size_v);
      acc = acc * softmax_factor +
            softmax_weight * byte_v2_bf16_bits_to_float(v_bits);
    }
    __syncthreads();
  }

  if (tid == 0) {
    inv_denom = denom > 0.0f ? (1.0f / denom) : 0.0f;
  }
  __syncthreads();
  if (tid < head_size_v) {
    output[(req_idx * num_heads + q_head) * head_size_v + tid] =
        byte_v2_float_to_bf16_bits(acc * inv_denom);
  }
}

template <typename block_table_t, typename seq_lens_t>
__global__ void byte_v2_paged_decode_attention_gqa_shared_kernel(
    const uint16_t* __restrict__ query, const uint8_t* __restrict__ kv_cache,
    const block_table_t* __restrict__ block_table,
    const seq_lens_t* __restrict__ seq_lens, uint16_t* __restrict__ output,
    const uint8_t* __restrict__ fallback_pool,
    const int32_t* __restrict__ fallback_block_ids,
    const int32_t* __restrict__ fallback_tile_ids,
    const int32_t* __restrict__ outlier_arena,
    const int32_t* __restrict__ outlier_block_flags,
    const int32_t* __restrict__ outlier_tile_bitmap,
    const int32_t* __restrict__ outlier_tile_meta,
    const float scale, const int num_decode_tokens, const int num_heads,
    const int num_kv_heads, const int head_size, const int head_size_v,
    const int page_size_bytes, const int raw_block_bytes,
    const int64_t block_table_stride0, const int64_t block_table_stride1,
    const int64_t seq_lens_stride0) {
  const int req_idx = blockIdx.x;
  const int kv_head = blockIdx.y;
  const int tid = threadIdx.x;
  if (req_idx >= num_decode_tokens || kv_head >= num_kv_heads) {
    return;
  }

  const int q_per_kv = num_heads / num_kv_heads;
  if (q_per_kv <= 0 || q_per_kv > kByteV2MaxQPerKv) {
    return;
  }

  __shared__ float k_shared[kByteV2MaxHeadSize];
  __shared__ float v_shared[kByteV2MaxHeadSize];
  __shared__ float score_sums[kByteV2MaxQPerKv][128];
  __shared__ float softmax_factor[kByteV2MaxQPerKv];
  __shared__ float softmax_weight[kByteV2MaxQPerKv];
  __shared__ float inv_denom[kByteV2MaxQPerKv];

  float acc[kByteV2MaxQPerKv];
#pragma unroll
  for (int q = 0; q < kByteV2MaxQPerKv; ++q) {
    acc[q] = 0.0f;
  }

  const int seq_len = static_cast<int>(seq_lens[req_idx * seq_lens_stride0]);
  const int first_q_head = kv_head * q_per_kv;
  if (seq_len <= 0) {
    for (int q = 0; q < q_per_kv; ++q) {
      const int q_head = first_q_head + q;
      if (q_head >= num_heads) {
        continue;
      }
      for (int d = tid; d < head_size_v; d += blockDim.x) {
        output[(req_idx * num_heads + q_head) * head_size_v + d] =
            byte_v2_float_to_bf16_bits(0.0f);
      }
    }
    return;
  }

  float running_max[kByteV2MaxQPerKv];
  float denom[kByteV2MaxQPerKv];
#pragma unroll
  for (int q = 0; q < kByteV2MaxQPerKv; ++q) {
    running_max[q] = -FLT_MAX;
    denom[q] = 0.0f;
  }

  for (int token = 0; token < seq_len; ++token) {
    const int logical_block = token / kByteV2TileSize;
    const int row = token % kByteV2TileSize;
    const int physical_block = static_cast<int>(
        block_table[req_idx * block_table_stride0 +
                    logical_block * block_table_stride1]);
    const uint8_t* page = kv_cache + physical_block * page_size_bytes;
    if (row >= static_cast<int>(page[kByteV2PageValidRowsOffset])) {
      continue;
    }
    const bool block_may_have_outliers =
        byte_v2_may_have_block_outliers(outlier_block_flags, physical_block);
    const int32_t* block_outlier_arena =
        block_may_have_outliers ? outlier_arena : nullptr;
    const int32_t* block_outlier_tile_bitmap =
        block_may_have_outliers ? outlier_tile_bitmap : nullptr;
    const int32_t* block_outlier_tile_meta =
        block_may_have_outliers ? outlier_tile_meta : nullptr;

    for (int d = tid; d < head_size; d += blockDim.x) {
      const uint16_t k_bits = byte_v2_load_kv_bits(
          page, fallback_pool, fallback_block_ids, fallback_tile_ids,
          block_outlier_arena, block_outlier_tile_bitmap,
          block_outlier_tile_meta,
          physical_block,
          raw_block_bytes, false, row, kv_head, d, num_kv_heads, head_size,
          head_size_v);
      k_shared[d] = byte_v2_bf16_bits_to_float(k_bits);
    }
    for (int d = tid; d < head_size_v; d += blockDim.x) {
      const uint16_t v_bits = byte_v2_load_kv_bits(
          page, fallback_pool, fallback_block_ids, fallback_tile_ids,
          block_outlier_arena, block_outlier_tile_bitmap,
          block_outlier_tile_meta,
          physical_block,
          raw_block_bytes, true, row, kv_head, d, num_kv_heads, head_size,
          head_size_v);
      v_shared[d] = byte_v2_bf16_bits_to_float(v_bits);
    }
    __syncthreads();

    for (int q = 0; q < q_per_kv; ++q) {
      const int q_head = first_q_head + q;
      float score_part = 0.0f;
      if (q_head < num_heads) {
        for (int d = tid; d < head_size; d += blockDim.x) {
          const uint16_t q_bits =
              query[(req_idx * num_heads + q_head) * head_size + d];
          score_part += byte_v2_bf16_bits_to_float(q_bits) * k_shared[d];
        }
      }
      score_sums[q][tid] = score_part;
    }
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
        for (int q = 0; q < q_per_kv; ++q) {
          score_sums[q][tid] += score_sums[q][tid + stride];
        }
      }
      __syncthreads();
    }

    if (tid == 0) {
      for (int q = 0; q < q_per_kv; ++q) {
        const float score = score_sums[q][0] * scale;
        float factor = 1.0f;
        if (score > running_max[q]) {
          factor = expf(running_max[q] - score);
          denom[q] *= factor;
          running_max[q] = score;
        }
        softmax_factor[q] = factor;
        softmax_weight[q] = expf(score - running_max[q]);
        denom[q] += softmax_weight[q];
      }
    }
    __syncthreads();

    if (tid < head_size_v) {
      const float v = v_shared[tid];
      for (int q = 0; q < q_per_kv; ++q) {
        acc[q] = acc[q] * softmax_factor[q] + softmax_weight[q] * v;
      }
    }
    __syncthreads();
  }

  if (tid == 0) {
    for (int q = 0; q < q_per_kv; ++q) {
      inv_denom[q] = denom[q] > 0.0f ? (1.0f / denom[q]) : 0.0f;
    }
  }
  __syncthreads();

  if (tid < head_size_v) {
    for (int q = 0; q < q_per_kv; ++q) {
      const int q_head = first_q_head + q;
      if (q_head < num_heads) {
        output[(req_idx * num_heads + q_head) * head_size_v + tid] =
            byte_v2_float_to_bf16_bits(acc[q] * inv_denom[q]);
      }
    }
  }
}

#ifndef USE_ROCM
template <typename block_table_t, typename seq_lens_t>
__global__ void byte_v2_paged_decode_attention_gqa_wmma_kernel(
    const uint16_t* __restrict__ query, const uint8_t* __restrict__ kv_cache,
    const block_table_t* __restrict__ block_table,
    const seq_lens_t* __restrict__ seq_lens, uint16_t* __restrict__ output,
    const uint8_t* __restrict__ fallback_pool,
    const int32_t* __restrict__ fallback_block_ids,
    const int32_t* __restrict__ fallback_tile_ids,
    const int32_t* __restrict__ outlier_arena,
    const int32_t* __restrict__ outlier_block_flags,
    const int32_t* __restrict__ outlier_tile_bitmap,
    const int32_t* __restrict__ outlier_tile_meta,
    const float scale, const int num_decode_tokens, const int num_heads,
    const int num_kv_heads, const int head_size, const int head_size_v,
    const int page_size_bytes, const int raw_block_bytes,
    const int64_t block_table_stride0, const int64_t block_table_stride1,
    const int64_t seq_lens_stride0) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  const int req_idx = blockIdx.x;
  const int kv_head = blockIdx.y;
  const int tid = threadIdx.x;
  if (req_idx >= num_decode_tokens || kv_head >= num_kv_heads) {
    return;
  }

  const int q_per_kv = num_heads / num_kv_heads;
  if (q_per_kv <= 0 || q_per_kv > kByteV2MaxQPerKv) {
    return;
  }

  __shared__ __align__(16) __nv_bfloat16 q_shared[kByteV2TileSize *
                                                  kByteV2MaxHeadSize];
  __shared__ __align__(16) __nv_bfloat16 k_shared[kByteV2MaxHeadSize *
                                                  kByteV2TileSize];
  __shared__ __align__(16) __nv_bfloat16 v_shared[kByteV2TileSize *
                                                  kByteV2MaxHeadSize];
  __shared__ __align__(16) __nv_bfloat16 p_shared[kByteV2TileSize *
                                                  kByteV2TileSize];
  __shared__ float scores[kByteV2TileSize * kByteV2TileSize];
  __shared__ float pv_shared[kByteV2TileSize * kByteV2TileSize];
  __shared__ float tile_acc_factor[kByteV2MaxQPerKv];
  __shared__ float inv_denom[kByteV2MaxQPerKv];

  const int seq_len = static_cast<int>(seq_lens[req_idx * seq_lens_stride0]);
  const int first_q_head = kv_head * q_per_kv;

  float acc[kByteV2MaxQPerKv];
  float running_max[kByteV2MaxQPerKv];
  float denom[kByteV2MaxQPerKv];
#pragma unroll
  for (int q = 0; q < kByteV2MaxQPerKv; ++q) {
    acc[q] = 0.0f;
    running_max[q] = -FLT_MAX;
    denom[q] = 0.0f;
  }

  if (seq_len <= 0) {
    for (int q = 0; q < q_per_kv; ++q) {
      const int q_head = first_q_head + q;
      if (q_head >= num_heads) {
        continue;
      }
      for (int d = tid; d < head_size_v; d += blockDim.x) {
        output[(req_idx * num_heads + q_head) * head_size_v + d] =
            byte_v2_float_to_bf16_bits(0.0f);
      }
    }
    return;
  }

  for (int idx = tid; idx < kByteV2TileSize * head_size; idx += blockDim.x) {
    const int q_row = idx / head_size;
    const int dim = idx % head_size;
    uint16_t q_bits = 0;
    if (q_row < q_per_kv) {
      const int q_head = first_q_head + q_row;
      if (q_head < num_heads) {
        q_bits = query[(req_idx * num_heads + q_head) * head_size + dim];
      }
    }
    q_shared[idx] = byte_v2_bf16_bits_to_wmma(q_bits);
  }
  __syncthreads();

  for (int token_base = 0; token_base < seq_len;
       token_base += kByteV2TileSize) {
    const int logical_block = token_base / kByteV2TileSize;
    const int physical_block = static_cast<int>(
        block_table[req_idx * block_table_stride0 +
                    logical_block * block_table_stride1]);
    const uint8_t* page = kv_cache + physical_block * page_size_bytes;
    const int page_valid_rows =
        static_cast<int>(page[kByteV2PageValidRowsOffset]);
    const int seq_rows = min(kByteV2TileSize, seq_len - token_base);
    const int valid_rows = min(seq_rows, page_valid_rows);
    if (valid_rows <= 0) {
      continue;
    }
    const bool block_may_have_outliers =
        byte_v2_may_have_block_outliers(outlier_block_flags, physical_block);
    const int32_t* block_outlier_arena =
        block_may_have_outliers ? outlier_arena : nullptr;
    const int32_t* block_outlier_tile_bitmap =
        block_may_have_outliers ? outlier_tile_bitmap : nullptr;
    const int32_t* block_outlier_tile_meta =
        block_may_have_outliers ? outlier_tile_meta : nullptr;

    for (int idx = tid; idx < head_size * kByteV2TileSize;
         idx += blockDim.x) {
      const int dim = idx / kByteV2TileSize;
      const int row = idx % kByteV2TileSize;
      uint16_t k_bits = 0;
      if (row < valid_rows) {
        k_bits = byte_v2_load_kv_bits(
            page, fallback_pool, fallback_block_ids, fallback_tile_ids,
            block_outlier_arena, block_outlier_tile_bitmap,
            block_outlier_tile_meta,
            physical_block,
            raw_block_bytes, false, row, kv_head, dim, num_kv_heads, head_size,
            head_size_v);
      }
      k_shared[idx] = byte_v2_bf16_bits_to_wmma(k_bits);
    }

    for (int idx = tid; idx < kByteV2TileSize * head_size_v;
         idx += blockDim.x) {
      const int row = idx / head_size_v;
      const int dim = idx % head_size_v;
      uint16_t v_bits = 0;
      if (row < valid_rows) {
        v_bits = byte_v2_load_kv_bits(
            page, fallback_pool, fallback_block_ids, fallback_tile_ids,
            block_outlier_arena, block_outlier_tile_bitmap,
            block_outlier_tile_meta,
            physical_block,
            raw_block_bytes, true, row, kv_head, dim, num_kv_heads, head_size,
            head_size_v);
      }
      v_shared[idx] = byte_v2_bf16_bits_to_wmma(v_bits);
    }
    __syncthreads();

    if (tid < warpSize) {
      nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16,
                             __nv_bfloat16, nvcuda::wmma::row_major>
          a_frag;
      nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16,
                             __nv_bfloat16, nvcuda::wmma::row_major>
          b_frag;
      nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float>
          c_frag;
      nvcuda::wmma::fill_fragment(c_frag, 0.0f);
      for (int dim_base = 0; dim_base < head_size;
           dim_base += kByteV2TileSize) {
        nvcuda::wmma::load_matrix_sync(a_frag, q_shared + dim_base,
                                       head_size);
        nvcuda::wmma::load_matrix_sync(
            b_frag, k_shared + dim_base * kByteV2TileSize, kByteV2TileSize);
        nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
      }
      nvcuda::wmma::store_matrix_sync(scores, c_frag, kByteV2TileSize,
                                      nvcuda::wmma::mem_row_major);
    }
    __syncthreads();

    if (tid == 0) {
      const __nv_bfloat16 zero = byte_v2_bf16_bits_to_wmma(0);
      for (int idx = 0; idx < kByteV2TileSize * kByteV2TileSize; ++idx) {
        p_shared[idx] = zero;
      }
      for (int q = 0; q < q_per_kv; ++q) {
        float tile_max = -FLT_MAX;
        for (int row = 0; row < valid_rows; ++row) {
          tile_max =
              max(tile_max, scores[q * kByteV2TileSize + row] * scale);
        }
        const float new_max = max(running_max[q], tile_max);
        const float old_scale = expf(running_max[q] - new_max);
        float new_denom = denom[q] * old_scale;
        for (int row = 0; row < valid_rows; ++row) {
          const float weight =
              expf(scores[q * kByteV2TileSize + row] * scale - new_max);
          p_shared[q * kByteV2TileSize + row] =
              byte_v2_bf16_bits_to_wmma(byte_v2_float_to_bf16_bits(weight));
          new_denom += weight;
        }
        tile_acc_factor[q] = old_scale;
        running_max[q] = new_max;
        denom[q] = new_denom;
      }
    }
    __syncthreads();

    if (tid < head_size_v) {
      for (int q = 0; q < q_per_kv; ++q) {
        acc[q] *= tile_acc_factor[q];
      }
    }
    __syncthreads();

    for (int dim_base = 0; dim_base < head_size_v;
         dim_base += kByteV2TileSize) {
      if (tid < warpSize) {
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16,
                               __nv_bfloat16, nvcuda::wmma::row_major>
            p_frag;
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16,
                               __nv_bfloat16, nvcuda::wmma::row_major>
            v_frag;
        nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float>
            pv_frag;
        nvcuda::wmma::fill_fragment(pv_frag, 0.0f);
        nvcuda::wmma::load_matrix_sync(p_frag, p_shared, kByteV2TileSize);
        nvcuda::wmma::load_matrix_sync(v_frag, v_shared + dim_base,
                                       head_size_v);
        nvcuda::wmma::mma_sync(pv_frag, p_frag, v_frag, pv_frag);
        nvcuda::wmma::store_matrix_sync(pv_shared, pv_frag, kByteV2TileSize,
                                        nvcuda::wmma::mem_row_major);
      }
      __syncthreads();

      if (tid >= dim_base && tid < dim_base + kByteV2TileSize &&
          tid < head_size_v) {
        const int dim_in_tile = tid - dim_base;
        for (int q = 0; q < q_per_kv; ++q) {
          acc[q] += pv_shared[q * kByteV2TileSize + dim_in_tile];
        }
      }
      __syncthreads();
    }
  }

  if (tid == 0) {
    for (int q = 0; q < q_per_kv; ++q) {
      inv_denom[q] = denom[q] > 0.0f ? (1.0f / denom[q]) : 0.0f;
    }
  }
  __syncthreads();

  if (tid < head_size_v) {
    for (int q = 0; q < q_per_kv; ++q) {
      const int q_head = first_q_head + q;
      if (q_head < num_heads) {
        output[(req_idx * num_heads + q_head) * head_size_v + tid] =
            byte_v2_float_to_bf16_bits(acc[q] * inv_denom[q]);
      }
    }
  }
#endif
}

template <typename block_table_t, typename seq_lens_t>
__global__ void byte_v2_paged_decode_attention_gqa_wmma_split_stage1_kernel(
    const uint16_t* __restrict__ query, const uint8_t* __restrict__ kv_cache,
    const block_table_t* __restrict__ block_table,
    const seq_lens_t* __restrict__ seq_lens, float* __restrict__ partial_output,
    const uint8_t* __restrict__ fallback_pool,
    const int32_t* __restrict__ fallback_block_ids,
    const int32_t* __restrict__ fallback_tile_ids,
    const int32_t* __restrict__ outlier_arena,
    const int32_t* __restrict__ outlier_block_flags,
    const int32_t* __restrict__ outlier_tile_bitmap,
    const int32_t* __restrict__ outlier_tile_meta,
    const float scale, const int num_decode_tokens, const int num_heads,
    const int num_kv_heads, const int head_size, const int head_size_v,
    const int page_size_bytes, const int raw_block_bytes,
    const int num_kv_splits, const int64_t block_table_stride0,
    const int64_t block_table_stride1, const int64_t seq_lens_stride0,
    const bool use_page_fastpath, const bool use_tile_fastpath) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  const int req_idx = blockIdx.x;
  const int kv_head = blockIdx.y;
  const int split_idx = blockIdx.z;
  const int tid = threadIdx.x;
  if (req_idx >= num_decode_tokens || kv_head >= num_kv_heads ||
      split_idx >= num_kv_splits) {
    return;
  }

  const int q_per_kv = num_heads / num_kv_heads;
  if (q_per_kv <= 0 || q_per_kv > kByteV2MaxQPerKv) {
    return;
  }

  __shared__ __align__(16) __nv_bfloat16 q_shared[kByteV2TileSize *
                                                  kByteV2MaxHeadSize];
  __shared__ __align__(16) __nv_bfloat16 k_shared[kByteV2MaxHeadSize *
                                                  kByteV2TileSize];
  __shared__ __align__(16) __nv_bfloat16 v_shared[kByteV2TileSize *
                                                  kByteV2MaxHeadSize];
  __shared__ __align__(16) __nv_bfloat16 p_shared[kByteV2TileSize *
                                                  kByteV2TileSize];
  __shared__ float scores[kByteV2TileSize * kByteV2TileSize];
  __shared__ float pv_shared[kByteV2TileSize * kByteV2TileSize];
  __shared__ float tile_acc_factor[kByteV2MaxQPerKv];
  __shared__ float lse_shared[kByteV2MaxQPerKv];
  __shared__ float inv_denom[kByteV2MaxQPerKv];

  const int seq_len = static_cast<int>(seq_lens[req_idx * seq_lens_stride0]);
  const int first_q_head = kv_head * q_per_kv;

  float acc[kByteV2MaxQPerKv];
  float running_max[kByteV2MaxQPerKv];
  float denom[kByteV2MaxQPerKv];
#pragma unroll
  for (int q = 0; q < kByteV2MaxQPerKv; ++q) {
    acc[q] = 0.0f;
    running_max[q] = -FLT_MAX;
    denom[q] = 0.0f;
  }

  const int partial_stride = head_size_v + 1;
  if (seq_len <= 0) {
    for (int q = 0; q < q_per_kv; ++q) {
      const int q_head = first_q_head + q;
      if (q_head >= num_heads) {
        continue;
      }
      float* partial =
          partial_output +
          (((req_idx * num_heads + q_head) * num_kv_splits + split_idx) *
           partial_stride);
      for (int d = tid; d < head_size_v; d += blockDim.x) {
        partial[d] = 0.0f;
      }
      if (tid == 0) {
        partial[head_size_v] = -FLT_MAX;
      }
    }
    return;
  }

  const int logical_blocks =
      (seq_len + kByteV2TileSize - 1) / kByteV2TileSize;
  const int blocks_per_split =
      (logical_blocks + num_kv_splits - 1) / num_kv_splits;
  const int start_block = split_idx * blocks_per_split;
  const int end_block = min(logical_blocks, start_block + blocks_per_split);
  if (start_block >= end_block) {
    for (int q = 0; q < q_per_kv; ++q) {
      const int q_head = first_q_head + q;
      if (q_head >= num_heads) {
        continue;
      }
      float* partial =
          partial_output +
          (((req_idx * num_heads + q_head) * num_kv_splits + split_idx) *
           partial_stride);
      for (int d = tid; d < head_size_v; d += blockDim.x) {
        partial[d] = 0.0f;
      }
      if (tid == 0) {
        partial[head_size_v] = -FLT_MAX;
      }
    }
    return;
  }

  for (int idx = tid; idx < kByteV2TileSize * head_size; idx += blockDim.x) {
    const int q_row = idx / head_size;
    const int dim = idx % head_size;
    uint16_t q_bits = 0;
    if (q_row < q_per_kv) {
      const int q_head = first_q_head + q_row;
      if (q_head < num_heads) {
        q_bits = query[(req_idx * num_heads + q_head) * head_size + dim];
      }
    }
    q_shared[idx] = byte_v2_bf16_bits_to_wmma(q_bits);
  }
  __syncthreads();

  const int k_dim_tiles = head_size / kByteV2TileSize;
  const int v_dim_tiles = head_size_v / kByteV2TileSize;
  const int total_tiles = num_kv_heads * (k_dim_tiles + v_dim_tiles);
  const int bitmap_words = byte_v2_outlier_tile_bitmap_words(total_tiles);

  for (int logical_block = start_block; logical_block < end_block;
       ++logical_block) {
    const int token_base = logical_block * kByteV2TileSize;
    const int physical_block = static_cast<int>(
        block_table[req_idx * block_table_stride0 +
                    logical_block * block_table_stride1]);
    const uint8_t* page = kv_cache + physical_block * page_size_bytes;
    const int page_valid_rows =
        static_cast<int>(page[kByteV2PageValidRowsOffset]);
    const int seq_rows = min(kByteV2TileSize, seq_len - token_base);
    const int valid_rows = min(seq_rows, page_valid_rows);
    if (valid_rows <= 0) {
      continue;
    }
    const uint8_t page_status = page[kByteV2PageStatusOffset];
    const bool block_may_have_outliers =
        byte_v2_may_have_block_outliers(outlier_block_flags, physical_block);
    const int32_t* block_outlier_arena =
        block_may_have_outliers ? outlier_arena : nullptr;
    const int32_t* block_outlier_tile_bitmap =
        block_may_have_outliers ? outlier_tile_bitmap : nullptr;
    const int32_t* block_outlier_tile_meta =
        block_may_have_outliers ? outlier_tile_meta : nullptr;

    const bool compressed_fastpath =
        use_page_fastpath && page_status == kByteV2PageStatusCompressed;
    const bool compressed_no_fallback_fastpath =
        compressed_fastpath && fallback_pool == nullptr &&
        fallback_tile_ids == nullptr && block_outlier_arena == nullptr &&
        block_outlier_tile_meta == nullptr;
    const bool raw_fallback_fastpath = false;

    if (compressed_no_fallback_fastpath) {
        for (int dim_tile = 0; dim_tile < k_dim_tiles; ++dim_tile) {
          byte_v2_decode_k_transposed_tile_to_shared_no_fallback_layout(
              page, k_shared, tid, kv_head, dim_tile, num_kv_heads,
              k_dim_tiles, v_dim_tiles, valid_rows);
          byte_v2_overlay_k_tile_outliers_to_shared(
              block_outlier_arena, block_outlier_tile_bitmap,
              block_outlier_tile_meta, k_shared, tid, physical_block, kv_head,
              dim_tile, k_dim_tiles, v_dim_tiles, valid_rows, total_tiles,
              bitmap_words);
        }
        for (int dim_tile = 0; dim_tile < v_dim_tiles; ++dim_tile) {
          byte_v2_decode_v_rowmajor_tile_to_shared_no_fallback_layout(
              page, v_shared, tid, kv_head, dim_tile, num_kv_heads,
              k_dim_tiles, v_dim_tiles, valid_rows, head_size_v);
          byte_v2_overlay_v_tile_outliers_to_shared(
              block_outlier_arena, block_outlier_tile_bitmap,
              block_outlier_tile_meta, v_shared, tid, physical_block, kv_head,
              dim_tile, k_dim_tiles, v_dim_tiles, valid_rows, head_size_v,
              total_tiles, bitmap_words);
        }
      } else if (compressed_fastpath && use_tile_fastpath) {
        for (int dim_tile = 0; dim_tile < k_dim_tiles; ++dim_tile) {
          if (byte_v2_tile_fallback_flag(page, false, kv_head, dim_tile,
                                         k_dim_tiles, v_dim_tiles) == 0) {
            byte_v2_decode_k_transposed_tile_to_shared_no_fallback_layout(
                page, k_shared, tid, kv_head, dim_tile, num_kv_heads,
                k_dim_tiles, v_dim_tiles, valid_rows);
            byte_v2_overlay_k_tile_outliers_to_shared(
                block_outlier_arena, block_outlier_tile_bitmap,
                block_outlier_tile_meta, k_shared, tid, physical_block,
                kv_head, dim_tile, k_dim_tiles, v_dim_tiles, valid_rows,
                total_tiles, bitmap_words);
          } else {
            byte_v2_decode_k_transposed_tile_to_shared_tile_fallback(
                fallback_pool, fallback_tile_ids, k_shared, tid,
                physical_block, kv_head, dim_tile, k_dim_tiles, v_dim_tiles,
                valid_rows, total_tiles);
          }
        }
        for (int dim_tile = 0; dim_tile < v_dim_tiles; ++dim_tile) {
          if (byte_v2_tile_fallback_flag(page, true, kv_head, dim_tile,
                                         k_dim_tiles, v_dim_tiles) == 0) {
            byte_v2_decode_v_rowmajor_tile_to_shared_no_fallback_layout(
                page, v_shared, tid, kv_head, dim_tile, num_kv_heads,
                k_dim_tiles, v_dim_tiles, valid_rows, head_size_v);
            byte_v2_overlay_v_tile_outliers_to_shared(
                block_outlier_arena, block_outlier_tile_bitmap,
                block_outlier_tile_meta, v_shared, tid, physical_block,
                kv_head, dim_tile, k_dim_tiles, v_dim_tiles, valid_rows,
                head_size_v, total_tiles, bitmap_words);
          } else {
            byte_v2_decode_v_rowmajor_tile_to_shared_tile_fallback(
                fallback_pool, fallback_tile_ids, v_shared, tid,
                physical_block, kv_head, dim_tile, k_dim_tiles, v_dim_tiles,
                valid_rows, head_size_v, total_tiles);
          }
        }
      } else if (compressed_fastpath) {
        for (int idx = tid; idx < head_size * kByteV2TileSize;
             idx += blockDim.x) {
          const int dim = idx / kByteV2TileSize;
          const int row = idx % kByteV2TileSize;
          uint16_t k_bits = 0;
          if (row < valid_rows) {
            k_bits = byte_v2_load_compressed_bits(
                page, false, row, kv_head, dim, k_dim_tiles, v_dim_tiles,
                fallback_pool, fallback_tile_ids, block_outlier_arena,
                block_outlier_tile_bitmap, block_outlier_tile_meta,
                physical_block, total_tiles, bitmap_words);
          }
          k_shared[idx] = byte_v2_bf16_bits_to_wmma(k_bits);
        }

        for (int idx = tid; idx < kByteV2TileSize * head_size_v;
             idx += blockDim.x) {
          const int row = idx / head_size_v;
          const int dim = idx % head_size_v;
          uint16_t v_bits = 0;
          if (row < valid_rows) {
            v_bits = byte_v2_load_compressed_bits(
                page, true, row, kv_head, dim, k_dim_tiles, v_dim_tiles,
                fallback_pool, fallback_tile_ids, block_outlier_arena,
                block_outlier_tile_bitmap, block_outlier_tile_meta,
                physical_block, total_tiles, bitmap_words);
          }
          v_shared[idx] = byte_v2_bf16_bits_to_wmma(v_bits);
        }
      } else if (raw_fallback_fastpath) {
        const uint8_t* raw_block = page + kByteV2PageHeaderBytes;
        if (fallback_pool != nullptr && fallback_block_ids != nullptr) {
          const int fallback_slot = fallback_block_ids[physical_block];
          if (fallback_slot >= 0) {
            raw_block = fallback_pool + fallback_slot * raw_block_bytes;
          }
        }

        for (int idx = tid; idx < head_size * kByteV2TileSize;
             idx += blockDim.x) {
          const int dim = idx / kByteV2TileSize;
          const int row = idx % kByteV2TileSize;
          uint16_t k_bits = 0;
          if (row < valid_rows) {
            k_bits = byte_v2_load_raw_bits_from_block(
                raw_block, false, row, kv_head, dim, num_kv_heads, head_size,
                head_size_v);
          }
          k_shared[idx] = byte_v2_bf16_bits_to_wmma(k_bits);
        }

        for (int idx = tid; idx < kByteV2TileSize * head_size_v;
             idx += blockDim.x) {
          const int row = idx / head_size_v;
          const int dim = idx % head_size_v;
          uint16_t v_bits = 0;
          if (row < valid_rows) {
            v_bits = byte_v2_load_raw_bits_from_block(
                raw_block, true, row, kv_head, dim, num_kv_heads, head_size,
                head_size_v);
          }
          v_shared[idx] = byte_v2_bf16_bits_to_wmma(v_bits);
        }
      } else {
        for (int idx = tid; idx < head_size * kByteV2TileSize;
             idx += blockDim.x) {
          const int dim = idx / kByteV2TileSize;
          const int row = idx % kByteV2TileSize;
          uint16_t k_bits = 0;
          if (row < valid_rows) {
            k_bits = byte_v2_load_kv_bits(
                page, fallback_pool, fallback_block_ids, fallback_tile_ids,
                block_outlier_arena, block_outlier_tile_bitmap,
                block_outlier_tile_meta, physical_block, raw_block_bytes,
                false, row, kv_head, dim, num_kv_heads, head_size,
                head_size_v);
          }
          k_shared[idx] = byte_v2_bf16_bits_to_wmma(k_bits);
        }

        for (int idx = tid; idx < kByteV2TileSize * head_size_v;
             idx += blockDim.x) {
          const int row = idx / head_size_v;
          const int dim = idx % head_size_v;
          uint16_t v_bits = 0;
          if (row < valid_rows) {
            v_bits = byte_v2_load_kv_bits(
                page, fallback_pool, fallback_block_ids, fallback_tile_ids,
                block_outlier_arena, block_outlier_tile_bitmap,
                block_outlier_tile_meta, physical_block, raw_block_bytes, true,
                row, kv_head, dim, num_kv_heads, head_size, head_size_v);
          }
          v_shared[idx] = byte_v2_bf16_bits_to_wmma(v_bits);
        }
    }
    __syncthreads();

    if (tid < warpSize) {
      nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16,
                             __nv_bfloat16, nvcuda::wmma::row_major>
          a_frag;
      nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16,
                             __nv_bfloat16, nvcuda::wmma::row_major>
          b_frag;
      nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float>
          c_frag;
      nvcuda::wmma::fill_fragment(c_frag, 0.0f);
      for (int dim_base = 0; dim_base < head_size;
           dim_base += kByteV2TileSize) {
        nvcuda::wmma::load_matrix_sync(a_frag, q_shared + dim_base,
                                       head_size);
        nvcuda::wmma::load_matrix_sync(
            b_frag, k_shared + dim_base * kByteV2TileSize, kByteV2TileSize);
        nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
      }
      nvcuda::wmma::store_matrix_sync(scores, c_frag, kByteV2TileSize,
                                      nvcuda::wmma::mem_row_major);
    }
    __syncthreads();

    if (tid == 0) {
      const __nv_bfloat16 zero = byte_v2_bf16_bits_to_wmma(0);
      for (int idx = 0; idx < kByteV2TileSize * kByteV2TileSize; ++idx) {
        p_shared[idx] = zero;
      }
      for (int q = 0; q < q_per_kv; ++q) {
        float tile_max = -FLT_MAX;
        for (int row = 0; row < valid_rows; ++row) {
          tile_max =
              max(tile_max, scores[q * kByteV2TileSize + row] * scale);
        }
        const float new_max = max(running_max[q], tile_max);
        const float old_scale = expf(running_max[q] - new_max);
        float new_denom = denom[q] * old_scale;
        for (int row = 0; row < valid_rows; ++row) {
          const float weight =
              expf(scores[q * kByteV2TileSize + row] * scale - new_max);
          p_shared[q * kByteV2TileSize + row] =
              byte_v2_bf16_bits_to_wmma(byte_v2_float_to_bf16_bits(weight));
          new_denom += weight;
        }
        tile_acc_factor[q] = old_scale;
        running_max[q] = new_max;
        denom[q] = new_denom;
      }
    }
    __syncthreads();

    if (tid < head_size_v) {
      for (int q = 0; q < q_per_kv; ++q) {
        acc[q] *= tile_acc_factor[q];
      }
    }
    __syncthreads();

    for (int dim_base = 0; dim_base < head_size_v;
         dim_base += kByteV2TileSize) {
      if (tid < warpSize) {
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16,
                               __nv_bfloat16, nvcuda::wmma::row_major>
            p_frag;
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16,
                               __nv_bfloat16, nvcuda::wmma::row_major>
            v_frag;
        nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float>
            pv_frag;
        nvcuda::wmma::fill_fragment(pv_frag, 0.0f);
        nvcuda::wmma::load_matrix_sync(p_frag, p_shared, kByteV2TileSize);
        nvcuda::wmma::load_matrix_sync(v_frag, v_shared + dim_base,
                                       head_size_v);
        nvcuda::wmma::mma_sync(pv_frag, p_frag, v_frag, pv_frag);
        nvcuda::wmma::store_matrix_sync(pv_shared, pv_frag, kByteV2TileSize,
                                        nvcuda::wmma::mem_row_major);
      }
      __syncthreads();

      if (tid >= dim_base && tid < dim_base + kByteV2TileSize &&
          tid < head_size_v) {
        const int dim_in_tile = tid - dim_base;
        for (int q = 0; q < q_per_kv; ++q) {
          acc[q] += pv_shared[q * kByteV2TileSize + dim_in_tile];
        }
      }
      __syncthreads();
    }
  }

  if (tid == 0) {
    for (int q = 0; q < q_per_kv; ++q) {
      inv_denom[q] = denom[q] > 0.0f ? (1.0f / denom[q]) : 0.0f;
      lse_shared[q] =
          denom[q] > 0.0f ? running_max[q] + logf(denom[q]) : -FLT_MAX;
    }
  }
  __syncthreads();

  for (int q = 0; q < q_per_kv; ++q) {
    const int q_head = first_q_head + q;
    if (q_head >= num_heads) {
      continue;
    }
    float* partial =
        partial_output +
        (((req_idx * num_heads + q_head) * num_kv_splits + split_idx) *
         partial_stride);
    if (tid < head_size_v) {
      partial[tid] = acc[q] * inv_denom[q];
    }
    if (tid == 0) {
      partial[head_size_v] = lse_shared[q];
    }
  }
#endif
}

template <typename block_table_t, typename seq_lens_t>
__global__ void
byte_v2_paged_decode_attention_gqa4_h128_fast_split_stage1_kernel(
    const uint16_t* __restrict__ query, const uint8_t* __restrict__ kv_cache,
    const block_table_t* __restrict__ block_table,
    const seq_lens_t* __restrict__ seq_lens, float* __restrict__ partial_output,
    const float scale, const int num_decode_tokens,
    const int page_size_bytes, const int num_kv_splits,
    const int64_t block_table_stride0, const int64_t block_table_stride1,
    const int64_t seq_lens_stride0) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  constexpr int num_heads = 32;
  constexpr int num_kv_heads = 8;
  constexpr int head_size = 128;
  constexpr int head_size_v = 128;
  constexpr int q_per_kv = 4;
  constexpr int k_dim_tiles = head_size / kByteV2TileSize;
  constexpr int v_dim_tiles = head_size_v / kByteV2TileSize;
  constexpr int partial_stride = head_size_v + 1;

  const int req_idx = blockIdx.x;
  const int kv_head = blockIdx.y;
  const int split_idx = blockIdx.z;
  const int tid = threadIdx.x;
  if (req_idx >= num_decode_tokens || kv_head >= num_kv_heads ||
      split_idx >= num_kv_splits) {
    return;
  }

  __shared__ __align__(16) __nv_bfloat16 q_shared[kByteV2TileSize *
                                                  head_size];
  __shared__ __align__(16) __nv_bfloat16 k_shared[head_size *
                                                  kByteV2TileSize];
  __shared__ __align__(16) __nv_bfloat16 v_shared[kByteV2TileSize *
                                                  head_size_v];
  __shared__ __align__(16) __nv_bfloat16 p_shared[kByteV2TileSize *
                                                  kByteV2TileSize];
  __shared__ float scores[kByteV2TileSize * kByteV2TileSize];
  __shared__ float pv_shared[kByteV2TileSize * kByteV2TileSize];
  __shared__ float tile_acc_factor[q_per_kv];
  __shared__ float lse_shared[q_per_kv];
  __shared__ float inv_denom[q_per_kv];
  __shared__ float running_max_shared[q_per_kv];
  __shared__ float denom_shared[q_per_kv];

  const int seq_len = static_cast<int>(seq_lens[req_idx * seq_lens_stride0]);
  const int first_q_head = kv_head * q_per_kv;

  float acc[q_per_kv];
#pragma unroll
  for (int q = 0; q < q_per_kv; ++q) {
    acc[q] = 0.0f;
  }

  if (seq_len <= 0) {
#pragma unroll
    for (int q = 0; q < q_per_kv; ++q) {
      const int q_head = first_q_head + q;
      float* partial =
          partial_output +
          (((req_idx * num_heads + q_head) * num_kv_splits + split_idx) *
           partial_stride);
      for (int d = tid; d < head_size_v; d += blockDim.x) {
        partial[d] = 0.0f;
      }
      if (tid == 0) {
        partial[head_size_v] = -FLT_MAX;
      }
    }
    return;
  }

  const int logical_blocks =
      (seq_len + kByteV2TileSize - 1) / kByteV2TileSize;
  const int blocks_per_split =
      (logical_blocks + num_kv_splits - 1) / num_kv_splits;
  const int start_block = split_idx * blocks_per_split;
  const int end_block = min(logical_blocks, start_block + blocks_per_split);
  if (start_block >= end_block) {
#pragma unroll
    for (int q = 0; q < q_per_kv; ++q) {
      const int q_head = first_q_head + q;
      float* partial =
          partial_output +
          (((req_idx * num_heads + q_head) * num_kv_splits + split_idx) *
           partial_stride);
      for (int d = tid; d < head_size_v; d += blockDim.x) {
        partial[d] = 0.0f;
      }
      if (tid == 0) {
        partial[head_size_v] = -FLT_MAX;
      }
    }
    return;
  }

  if (tid < q_per_kv) {
    running_max_shared[tid] = -FLT_MAX;
    denom_shared[tid] = 0.0f;
  }
  __syncthreads();

  for (int idx = tid; idx < kByteV2TileSize * head_size; idx += blockDim.x) {
    const int q_row = idx / head_size;
    const int dim = idx % head_size;
    uint16_t q_bits = 0;
    if (q_row < q_per_kv) {
      const int q_head = first_q_head + q_row;
      q_bits = query[(req_idx * num_heads + q_head) * head_size + dim];
    }
    q_shared[idx] = byte_v2_bf16_bits_to_wmma(q_bits);
  }
  __syncthreads();

  for (int logical_block = start_block; logical_block < end_block;
       ++logical_block) {
    const int token_base = logical_block * kByteV2TileSize;
    const int physical_block = static_cast<int>(
        block_table[req_idx * block_table_stride0 +
                    logical_block * block_table_stride1]);
    const uint8_t* page = kv_cache + physical_block * page_size_bytes;
    const int remaining_rows = seq_len - token_base;
    int valid_rows = kByteV2TileSize;
    if (remaining_rows < kByteV2TileSize) {
      const int page_valid_rows =
          static_cast<int>(page[kByteV2PageValidRowsOffset]);
      valid_rows = min(max(remaining_rows, 0), page_valid_rows);
    }
    if (valid_rows <= 0) {
      continue;
    }

    if (valid_rows == kByteV2TileSize) {
#pragma unroll
      for (int dim_tile = 0; dim_tile < k_dim_tiles; ++dim_tile) {
        byte_v2_decode_k_transposed_tile_to_shared_no_fallback_full_rows(
            page, k_shared, tid, kv_head, dim_tile, k_dim_tiles, v_dim_tiles);
      }
#pragma unroll
      for (int dim_tile = 0; dim_tile < v_dim_tiles; ++dim_tile) {
        byte_v2_decode_v_rowmajor_tile_to_shared_no_fallback_full_rows(
            page, v_shared, tid, kv_head, dim_tile, k_dim_tiles, v_dim_tiles,
            head_size_v);
      }
    } else {
#pragma unroll
      for (int dim_tile = 0; dim_tile < k_dim_tiles; ++dim_tile) {
        byte_v2_decode_k_transposed_tile_to_shared_no_fallback(
            page, k_shared, tid, kv_head, dim_tile, k_dim_tiles, v_dim_tiles,
            valid_rows);
      }
#pragma unroll
      for (int dim_tile = 0; dim_tile < v_dim_tiles; ++dim_tile) {
        byte_v2_decode_v_rowmajor_tile_to_shared_no_fallback(
            page, v_shared, tid, kv_head, dim_tile, k_dim_tiles, v_dim_tiles,
            valid_rows, head_size_v);
      }
    }
    __syncthreads();

    if (tid < warpSize) {
      nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16,
                             __nv_bfloat16, nvcuda::wmma::row_major>
          a_frag;
      nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16,
                             __nv_bfloat16, nvcuda::wmma::row_major>
          b_frag;
      nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float>
          c_frag;
      nvcuda::wmma::fill_fragment(c_frag, 0.0f);
#pragma unroll
      for (int dim_base = 0; dim_base < head_size;
           dim_base += kByteV2TileSize) {
        nvcuda::wmma::load_matrix_sync(a_frag, q_shared + dim_base,
                                       head_size);
        nvcuda::wmma::load_matrix_sync(
            b_frag, k_shared + dim_base * kByteV2TileSize, kByteV2TileSize);
        nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
      }
      nvcuda::wmma::store_matrix_sync(scores, c_frag, kByteV2TileSize,
                                      nvcuda::wmma::mem_row_major);
    }
    __syncthreads();

    const __nv_bfloat16 zero = byte_v2_bf16_bits_to_wmma(0);
    for (int idx = tid; idx < kByteV2TileSize * kByteV2TileSize;
         idx += blockDim.x) {
      p_shared[idx] = zero;
    }
    __syncthreads();

    const int warp_id = tid / warpSize;
    const int lane_id = tid % warpSize;
    if (warp_id < q_per_kv) {
      const float score =
          lane_id < valid_rows
              ? scores[warp_id * kByteV2TileSize + lane_id] * scale
              : -FLT_MAX;
      const float tile_max = byte_v2_warp_reduce_max(score);
      const float old_max = running_max_shared[warp_id];
      const float new_max = max(old_max, tile_max);
      const float old_scale = expf(old_max - new_max);
      const float weight =
          lane_id < valid_rows ? expf(score - new_max) : 0.0f;
      const float tile_denom = byte_v2_warp_reduce_sum(weight);
      if (lane_id < kByteV2TileSize) {
        p_shared[warp_id * kByteV2TileSize + lane_id] =
            lane_id < valid_rows
                ? byte_v2_bf16_bits_to_wmma(
                      byte_v2_float_to_bf16_bits(weight))
                : zero;
      }
      if (lane_id == 0) {
        tile_acc_factor[warp_id] = old_scale;
        denom_shared[warp_id] = denom_shared[warp_id] * old_scale + tile_denom;
        running_max_shared[warp_id] = new_max;
      }
    }
    __syncthreads();

    if (tid < head_size_v) {
#pragma unroll
      for (int q = 0; q < q_per_kv; ++q) {
        acc[q] *= tile_acc_factor[q];
      }
    }
    __syncthreads();

#pragma unroll
    for (int dim_base = 0; dim_base < head_size_v;
         dim_base += kByteV2TileSize) {
      if (tid < warpSize) {
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16,
                               __nv_bfloat16, nvcuda::wmma::row_major>
            p_frag;
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16,
                               __nv_bfloat16, nvcuda::wmma::row_major>
            v_frag;
        nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float>
            pv_frag;
        nvcuda::wmma::fill_fragment(pv_frag, 0.0f);
        nvcuda::wmma::load_matrix_sync(p_frag, p_shared, kByteV2TileSize);
        nvcuda::wmma::load_matrix_sync(v_frag, v_shared + dim_base,
                                       head_size_v);
        nvcuda::wmma::mma_sync(pv_frag, p_frag, v_frag, pv_frag);
        nvcuda::wmma::store_matrix_sync(pv_shared, pv_frag, kByteV2TileSize,
                                        nvcuda::wmma::mem_row_major);
      }
      __syncthreads();

      if (tid >= dim_base && tid < dim_base + kByteV2TileSize &&
          tid < head_size_v) {
        const int dim_in_tile = tid - dim_base;
#pragma unroll
        for (int q = 0; q < q_per_kv; ++q) {
          acc[q] += pv_shared[q * kByteV2TileSize + dim_in_tile];
        }
      }
      __syncthreads();
    }
  }

  if (tid == 0) {
#pragma unroll
    for (int q = 0; q < q_per_kv; ++q) {
      inv_denom[q] =
          denom_shared[q] > 0.0f ? (1.0f / denom_shared[q]) : 0.0f;
      lse_shared[q] = denom_shared[q] > 0.0f
                          ? running_max_shared[q] + logf(denom_shared[q])
                          : -FLT_MAX;
    }
  }
  __syncthreads();

#pragma unroll
  for (int q = 0; q < q_per_kv; ++q) {
    const int q_head = first_q_head + q;
    float* partial =
        partial_output +
        (((req_idx * num_heads + q_head) * num_kv_splits + split_idx) *
         partial_stride);
    if (tid < head_size_v) {
      partial[tid] = acc[q] * inv_denom[q];
    }
    if (tid == 0) {
      partial[head_size_v] = lse_shared[q];
    }
  }
#endif
}

template <typename block_table_t, typename seq_lens_t,
          bool metadata_fastpath>
__global__ void
byte_v2_paged_decode_attention_gqa4_h128_flash_split_stage1_kernel(
    const uint16_t* __restrict__ query, const uint8_t* __restrict__ kv_cache,
    const block_table_t* __restrict__ block_table,
    const seq_lens_t* __restrict__ seq_lens, float* __restrict__ partial_output,
    const uint8_t* __restrict__ fallback_pool,
    const int32_t* __restrict__ fallback_block_ids,
    const int32_t* __restrict__ fallback_tile_ids, const float scale,
    const int num_decode_tokens, const int page_size_bytes,
    const int raw_block_bytes, const int num_kv_splits,
    const int64_t block_table_stride0, const int64_t block_table_stride1,
    const int64_t seq_lens_stride0, const bool use_tile_fastpath) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  constexpr int num_heads = 32;
  constexpr int num_kv_heads = 8;
  constexpr int head_size = 128;
  constexpr int head_size_v = 128;
  constexpr int q_per_kv = 4;
  constexpr int k_dim_tiles = head_size / kByteV2TileSize;
  constexpr int v_dim_tiles = head_size_v / kByteV2TileSize;
  constexpr int partial_stride = head_size_v + 1;

  const int req_idx = blockIdx.x;
  const int kv_head = blockIdx.y;
  const int split_idx = blockIdx.z;
  const int tid = threadIdx.x;
  const int warp_id = tid / warpSize;
  const int lane_id = tid % warpSize;
  if (req_idx >= num_decode_tokens || kv_head >= num_kv_heads ||
      split_idx >= num_kv_splits) {
    return;
  }

  __shared__ __align__(16) __nv_bfloat16 q_shared[q_per_kv * head_size];
  __shared__ __align__(16) __nv_bfloat16 k_shared[head_size *
                                                  kByteV2TileSize];
  __shared__ __align__(16) __nv_bfloat16 v_shared[kByteV2TileSize *
                                                  head_size_v];

  const int seq_len = static_cast<int>(seq_lens[req_idx * seq_lens_stride0]);
  const int first_q_head = kv_head * q_per_kv;

  if (seq_len <= 0) {
#pragma unroll
    for (int q = 0; q < q_per_kv; ++q) {
      const int q_head = first_q_head + q;
      float* partial =
          partial_output +
          (((req_idx * num_heads + q_head) * num_kv_splits + split_idx) *
           partial_stride);
      for (int d = tid; d < head_size_v; d += blockDim.x) {
        partial[d] = 0.0f;
      }
      if (tid == 0) {
        partial[head_size_v] = -FLT_MAX;
      }
    }
    return;
  }

  const int logical_blocks =
      (seq_len + kByteV2TileSize - 1) / kByteV2TileSize;
  const int blocks_per_split =
      (logical_blocks + num_kv_splits - 1) / num_kv_splits;
  const int start_block = split_idx * blocks_per_split;
  const int end_block = min(logical_blocks, start_block + blocks_per_split);
  if (start_block >= end_block) {
#pragma unroll
    for (int q = 0; q < q_per_kv; ++q) {
      const int q_head = first_q_head + q;
      float* partial =
          partial_output +
          (((req_idx * num_heads + q_head) * num_kv_splits + split_idx) *
           partial_stride);
      for (int d = tid; d < head_size_v; d += blockDim.x) {
        partial[d] = 0.0f;
      }
      if (tid == 0) {
        partial[head_size_v] = -FLT_MAX;
      }
    }
    return;
  }

  for (int idx = tid; idx < q_per_kv * head_size; idx += blockDim.x) {
    const int q_row = idx / head_size;
    const int dim = idx % head_size;
    const int q_head = first_q_head + q_row;
    const uint16_t q_bits =
        query[(req_idx * num_heads + q_head) * head_size + dim];
    q_shared[idx] = byte_v2_bf16_bits_to_wmma(q_bits);
  }
  __syncthreads();

  float running_max = -FLT_MAX;
  float denom = 0.0f;
  float acc0 = 0.0f;
  float acc1 = 0.0f;
  float acc2 = 0.0f;
  float acc3 = 0.0f;

  for (int logical_block = start_block; logical_block < end_block;
       ++logical_block) {
    const int token_base = logical_block * kByteV2TileSize;
    const int physical_block = static_cast<int>(
        block_table[req_idx * block_table_stride0 +
                    logical_block * block_table_stride1]);
    const uint8_t* page = kv_cache + physical_block * page_size_bytes;
    const int remaining_rows = seq_len - token_base;
    int valid_rows = kByteV2TileSize;
    if (remaining_rows < kByteV2TileSize) {
      const int page_valid_rows =
          static_cast<int>(page[kByteV2PageValidRowsOffset]);
      valid_rows = min(max(remaining_rows, 0), page_valid_rows);
    }
    if (valid_rows <= 0) {
      continue;
    }
    const uint8_t page_status = page[kByteV2PageStatusOffset];
    if constexpr (!metadata_fastpath) {
      if (page_status != kByteV2PageStatusCompressed) {
        continue;
      }
      if (valid_rows == kByteV2TileSize) {
#pragma unroll
        for (int dim_tile = 0; dim_tile < k_dim_tiles; ++dim_tile) {
          byte_v2_decode_k_transposed_tile_to_shared_no_fallback_full_rows(
              page, k_shared, tid, kv_head, dim_tile, k_dim_tiles,
              v_dim_tiles);
        }
#pragma unroll
        for (int dim_tile = 0; dim_tile < v_dim_tiles; ++dim_tile) {
          byte_v2_decode_v_rowmajor_tile_to_shared_no_fallback_full_rows(
              page, v_shared, tid, kv_head, dim_tile, k_dim_tiles, v_dim_tiles,
              head_size_v);
        }
      } else {
#pragma unroll
        for (int dim_tile = 0; dim_tile < k_dim_tiles; ++dim_tile) {
          byte_v2_decode_k_transposed_tile_to_shared_no_fallback(
              page, k_shared, tid, kv_head, dim_tile, k_dim_tiles, v_dim_tiles,
              valid_rows);
        }
#pragma unroll
        for (int dim_tile = 0; dim_tile < v_dim_tiles; ++dim_tile) {
          byte_v2_decode_v_rowmajor_tile_to_shared_no_fallback(
              page, v_shared, tid, kv_head, dim_tile, k_dim_tiles, v_dim_tiles,
              valid_rows, head_size_v);
        }
      }
    } else {
      constexpr int total_tiles = num_kv_heads * (k_dim_tiles + v_dim_tiles);
      if (page_status == kByteV2PageStatusCompressed && use_tile_fastpath) {
#pragma unroll
        for (int dim_tile = 0; dim_tile < k_dim_tiles; ++dim_tile) {
          const int tile_start =
              byte_v2_tile_start(false, kv_head, dim_tile, k_dim_tiles,
                                 v_dim_tiles);
          if (page[tile_start + 1] == 0) {
            if (valid_rows == kByteV2TileSize) {
              byte_v2_decode_k_transposed_tile_to_shared_no_fallback_full_rows(
                  page, k_shared, tid, kv_head, dim_tile, k_dim_tiles,
                  v_dim_tiles);
            } else {
              byte_v2_decode_k_transposed_tile_to_shared_no_fallback(
                  page, k_shared, tid, kv_head, dim_tile, k_dim_tiles,
                  v_dim_tiles, valid_rows);
            }
          } else {
            byte_v2_decode_k_transposed_tile_to_shared_tile_fallback(
                fallback_pool, fallback_tile_ids, k_shared, tid,
                physical_block, kv_head, dim_tile, k_dim_tiles, v_dim_tiles,
                valid_rows, total_tiles);
          }
        }
#pragma unroll
        for (int dim_tile = 0; dim_tile < v_dim_tiles; ++dim_tile) {
          const int tile_start =
              byte_v2_tile_start(true, kv_head, dim_tile, k_dim_tiles,
                                 v_dim_tiles);
          if (page[tile_start + 1] == 0) {
            if (valid_rows == kByteV2TileSize) {
              byte_v2_decode_v_rowmajor_tile_to_shared_no_fallback_full_rows(
                  page, v_shared, tid, kv_head, dim_tile, k_dim_tiles,
                  v_dim_tiles, head_size_v);
            } else {
              byte_v2_decode_v_rowmajor_tile_to_shared_no_fallback(
                  page, v_shared, tid, kv_head, dim_tile, k_dim_tiles,
                  v_dim_tiles, valid_rows, head_size_v);
            }
          } else {
            byte_v2_decode_v_rowmajor_tile_to_shared_tile_fallback(
                fallback_pool, fallback_tile_ids, v_shared, tid,
                physical_block, kv_head, dim_tile, k_dim_tiles, v_dim_tiles,
                valid_rows, head_size_v, total_tiles);
          }
        }
      } else if (page_status == kByteV2PageStatusCompressed) {
        for (int idx = tid; idx < head_size * kByteV2TileSize;
             idx += blockDim.x) {
          const int dim = idx / kByteV2TileSize;
          const int row = idx % kByteV2TileSize;
          uint16_t k_bits = 0;
          if (row < valid_rows) {
            k_bits = byte_v2_load_compressed_bits(
                page, false, row, kv_head, dim, k_dim_tiles, v_dim_tiles,
                fallback_pool, fallback_tile_ids, nullptr, nullptr, nullptr,
                physical_block, total_tiles, 0);
          }
          k_shared[idx] = byte_v2_bf16_bits_to_wmma(k_bits);
        }
        for (int idx = tid; idx < kByteV2TileSize * head_size_v;
             idx += blockDim.x) {
          const int row = idx / head_size_v;
          const int dim = idx % head_size_v;
          uint16_t v_bits = 0;
          if (row < valid_rows) {
            v_bits = byte_v2_load_compressed_bits(
                page, true, row, kv_head, dim, k_dim_tiles, v_dim_tiles,
                fallback_pool, fallback_tile_ids, nullptr, nullptr, nullptr,
                physical_block, total_tiles, 0);
          }
          v_shared[idx] = byte_v2_bf16_bits_to_wmma(v_bits);
        }
      } else if (page_status == kByteV2PageStatusRawFallback) {
        const uint8_t* raw_block = page + kByteV2PageHeaderBytes;
        if (fallback_pool != nullptr && fallback_block_ids != nullptr) {
          const int fallback_slot = fallback_block_ids[physical_block];
          if (fallback_slot >= 0) {
            raw_block = fallback_pool + fallback_slot * raw_block_bytes;
          }
        }
        for (int idx = tid; idx < head_size * kByteV2TileSize;
             idx += blockDim.x) {
          const int dim = idx / kByteV2TileSize;
          const int row = idx % kByteV2TileSize;
          uint16_t k_bits = 0;
          if (row < valid_rows) {
            k_bits = byte_v2_load_raw_bits_from_block(
                raw_block, false, row, kv_head, dim, num_kv_heads, head_size,
                head_size_v);
          }
          k_shared[idx] = byte_v2_bf16_bits_to_wmma(k_bits);
        }
        for (int idx = tid; idx < kByteV2TileSize * head_size_v;
             idx += blockDim.x) {
          const int row = idx / head_size_v;
          const int dim = idx % head_size_v;
          uint16_t v_bits = 0;
          if (row < valid_rows) {
            v_bits = byte_v2_load_raw_bits_from_block(
                raw_block, true, row, kv_head, dim, num_kv_heads, head_size,
                head_size_v);
          }
          v_shared[idx] = byte_v2_bf16_bits_to_wmma(v_bits);
        }
      } else {
        continue;
      }
    }
    __syncthreads();

    if (warp_id < q_per_kv) {
      const int lane_row = lane_id & (kByteV2TileSize - 1);
      const int dim_begin = (lane_id >> 4) * (head_size / 2);
      const int dim_end = dim_begin + (head_size / 2);
      float qk = 0.0f;
#pragma unroll
      for (int dim = dim_begin; dim < dim_end; ++dim) {
        const float q_value =
            byte_v2_wmma_to_float(q_shared[warp_id * head_size + dim]);
        const float k_value =
            byte_v2_wmma_to_float(k_shared[dim * kByteV2TileSize + lane_row]);
        qk += q_value * k_value;
      }
      qk += __shfl_xor_sync(0xffffffffU, qk, kByteV2TileSize);
      const float score =
          (lane_id < kByteV2TileSize && lane_row < valid_rows) ? qk * scale
                                                               : -FLT_MAX;
      float tile_max = byte_v2_warp_reduce_max(score);
      tile_max = __shfl_sync(0xffffffffU, tile_max, 0);
      const float new_max = max(running_max, tile_max);
      const float old_scale = expf(running_max - new_max);
      const float weight =
          (lane_id < kByteV2TileSize && lane_row < valid_rows)
              ? expf(score - new_max)
              : 0.0f;
      float tile_denom = byte_v2_warp_reduce_sum(weight);
      tile_denom = __shfl_sync(0xffffffffU, tile_denom, 0);

      float pv0 = 0.0f;
      float pv1 = 0.0f;
      float pv2 = 0.0f;
      float pv3 = 0.0f;
#pragma unroll
      for (int row = 0; row < kByteV2TileSize; ++row) {
        const float row_weight = __shfl_sync(0xffffffffU, weight, row);
        if (row < valid_rows) {
          pv0 += row_weight *
                 byte_v2_wmma_to_float(v_shared[row * head_size_v + lane_id]);
          pv1 += row_weight *
                 byte_v2_wmma_to_float(
                     v_shared[row * head_size_v + lane_id + 32]);
          pv2 += row_weight *
                 byte_v2_wmma_to_float(
                     v_shared[row * head_size_v + lane_id + 64]);
          pv3 += row_weight *
                 byte_v2_wmma_to_float(
                     v_shared[row * head_size_v + lane_id + 96]);
        }
      }
      acc0 = acc0 * old_scale + pv0;
      acc1 = acc1 * old_scale + pv1;
      acc2 = acc2 * old_scale + pv2;
      acc3 = acc3 * old_scale + pv3;
      denom = denom * old_scale + tile_denom;
      running_max = new_max;
    }
    __syncthreads();
  }

  if (warp_id < q_per_kv) {
    const int q_head = first_q_head + warp_id;
    float* partial =
        partial_output +
        (((req_idx * num_heads + q_head) * num_kv_splits + split_idx) *
         partial_stride);
    const float inv_denom = denom > 0.0f ? (1.0f / denom) : 0.0f;
    partial[lane_id] = acc0 * inv_denom;
    partial[lane_id + 32] = acc1 * inv_denom;
    partial[lane_id + 64] = acc2 * inv_denom;
    partial[lane_id + 96] = acc3 * inv_denom;
    if (lane_id == 0) {
      partial[head_size_v] =
          denom > 0.0f ? running_max + logf(denom) : -FLT_MAX;
    }
  }
#endif
}

template <typename block_table_t, typename seq_lens_t, int MacroPages,
          bool Specialized, bool OutlierOverlay>
__global__ void
byte_v2_paged_decode_attention_gqa4_h128_v4_split_stage1_kernel(
    const uint16_t* __restrict__ query, const uint8_t* __restrict__ kv_cache,
    const block_table_t* __restrict__ block_table,
    const seq_lens_t* __restrict__ seq_lens, float* __restrict__ partial_output,
    const uint8_t* __restrict__ fallback_pool,
    const int32_t* __restrict__ fallback_block_ids,
    const int32_t* __restrict__ outlier_arena,
    const int32_t* __restrict__ outlier_tile_bitmap,
    const int32_t* __restrict__ outlier_tile_meta,
    const float scale, const int num_decode_tokens,
    const int page_size_bytes, const int raw_block_bytes,
    const int num_kv_splits,
    const int64_t block_table_stride0, const int64_t block_table_stride1,
    const int64_t seq_lens_stride0) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  static_assert(MacroPages == 4 || MacroPages == 8,
                "ByteV2 V4 supports 4 or 8 pages per macro");
  constexpr int num_heads = 32;
  constexpr int num_kv_heads = 8;
  constexpr int head_size = 128;
  constexpr int head_size_v = 128;
  constexpr int q_per_kv = 4;
  constexpr int k_dim_tiles = head_size / kByteV2TileSize;
  constexpr int v_dim_tiles = head_size_v / kByteV2TileSize;
  constexpr int total_tiles = num_kv_heads * (k_dim_tiles + v_dim_tiles);
  constexpr int bitmap_words = (total_tiles + 31) / 32;
  constexpr int partial_stride = head_size_v + 1;

  const int req_idx = blockIdx.x;
  const int kv_head = blockIdx.y;
  const int split_idx = blockIdx.z;
  const int tid = threadIdx.x;
  const int warp_id = tid / warpSize;
  const int lane_id = tid % warpSize;
  if (req_idx >= num_decode_tokens || kv_head >= num_kv_heads ||
      split_idx >= num_kv_splits) {
    return;
  }

  __shared__ __align__(16) __nv_bfloat16 q_shared[q_per_kv * head_size];
  __shared__ __align__(16) __nv_bfloat16 k_shared[head_size *
                                                  kByteV2TileSize];
  __shared__ __align__(16) __nv_bfloat16 v_shared[kByteV2TileSize *
                                                  head_size_v];
  __shared__ int macro_physical_blocks[MacroPages];
  __shared__ int macro_valid_rows[MacroPages];
  __shared__ int macro_compressed_v3[MacroPages];

  const int seq_len = static_cast<int>(seq_lens[req_idx * seq_lens_stride0]);
  const int first_q_head = kv_head * q_per_kv;

  if (seq_len <= 0) {
#pragma unroll
    for (int q = 0; q < q_per_kv; ++q) {
      const int q_head = first_q_head + q;
      float* partial =
          partial_output +
          (((req_idx * num_heads + q_head) * num_kv_splits + split_idx) *
           partial_stride);
      for (int d = tid; d < head_size_v; d += blockDim.x) {
        partial[d] = 0.0f;
      }
      if (tid == 0) {
        partial[head_size_v] = -FLT_MAX;
      }
    }
    return;
  }

  const int logical_blocks =
      (seq_len + kByteV2TileSize - 1) / kByteV2TileSize;
  const int blocks_per_split =
      (logical_blocks + num_kv_splits - 1) / num_kv_splits;
  const int start_block = split_idx * blocks_per_split;
  const int end_block = min(logical_blocks, start_block + blocks_per_split);
  if (start_block >= end_block) {
#pragma unroll
    for (int q = 0; q < q_per_kv; ++q) {
      const int q_head = first_q_head + q;
      float* partial =
          partial_output +
          (((req_idx * num_heads + q_head) * num_kv_splits + split_idx) *
           partial_stride);
      for (int d = tid; d < head_size_v; d += blockDim.x) {
        partial[d] = 0.0f;
      }
      if (tid == 0) {
        partial[head_size_v] = -FLT_MAX;
      }
    }
    return;
  }

  for (int idx = tid; idx < q_per_kv * head_size; idx += blockDim.x) {
    const int q_row = idx / head_size;
    const int dim = idx % head_size;
    const int q_head = first_q_head + q_row;
    const uint16_t q_bits =
        query[(req_idx * num_heads + q_head) * head_size + dim];
    q_shared[idx] = byte_v2_bf16_bits_to_wmma(q_bits);
  }
  __syncthreads();

  float running_max = -FLT_MAX;
  float denom = 0.0f;
  float acc0 = 0.0f;
  float acc1 = 0.0f;
  float acc2 = 0.0f;
  float acc3 = 0.0f;

  for (int macro_begin = start_block; macro_begin < end_block;
       macro_begin += MacroPages) {
    byte_v2_v4_load_macro_descriptor<MacroPages>(
        kv_cache, block_table, req_idx, seq_len, macro_begin, end_block,
        page_size_bytes, block_table_stride0, block_table_stride1,
        macro_physical_blocks, macro_valid_rows, macro_compressed_v3);

#pragma unroll
    for (int macro_idx = 0; macro_idx < MacroPages; ++macro_idx) {
      const int valid_rows = macro_valid_rows[macro_idx];
      if (valid_rows <= 0) {
        continue;
      }
      const int physical_block = macro_physical_blocks[macro_idx];
      const uint8_t* page = kv_cache + physical_block * page_size_bytes;
      if constexpr (Specialized) {
        if (macro_compressed_v3[macro_idx] == 0) {
          continue;
        }

#pragma unroll
        for (int dim_tile = 0; dim_tile < k_dim_tiles; ++dim_tile) {
          byte_v2_decode_k_transposed_tile_to_shared_no_fallback_v3(
              page, k_shared, tid, kv_head, dim_tile, num_kv_heads,
              k_dim_tiles, v_dim_tiles, valid_rows, true);
        }
#pragma unroll
        for (int dim_tile = 0; dim_tile < v_dim_tiles; ++dim_tile) {
          byte_v2_decode_v_rowmajor_tile_to_shared_no_fallback_v3(
              page, v_shared, tid, kv_head, dim_tile, num_kv_heads,
              k_dim_tiles, v_dim_tiles, valid_rows, head_size_v, true);
        }
        if constexpr (OutlierOverlay) {
#pragma unroll
          for (int dim_tile = 0; dim_tile < k_dim_tiles; ++dim_tile) {
            byte_v2_overlay_k_tile_outliers_to_shared(
                outlier_arena, outlier_tile_bitmap, outlier_tile_meta,
                k_shared, tid, physical_block, kv_head, dim_tile, k_dim_tiles,
                v_dim_tiles, valid_rows, total_tiles, bitmap_words);
          }
#pragma unroll
          for (int dim_tile = 0; dim_tile < v_dim_tiles; ++dim_tile) {
            byte_v2_overlay_v_tile_outliers_to_shared(
                outlier_arena, outlier_tile_bitmap, outlier_tile_meta,
                v_shared, tid, physical_block, kv_head, dim_tile, k_dim_tiles,
                v_dim_tiles, valid_rows, head_size_v, total_tiles,
                bitmap_words);
          }
        }
      } else {
        const bool is_compressed_v3 = macro_compressed_v3[macro_idx] != 0;
        const uint8_t* raw_block = nullptr;
        if (!is_compressed_v3 &&
            page[kByteV2PageStatusOffset] == kByteV2PageStatusRawFallback &&
            fallback_pool != nullptr && fallback_block_ids != nullptr) {
          const int fallback_slot = fallback_block_ids[physical_block];
          if (fallback_slot >= 0) {
            raw_block = fallback_pool +
                        static_cast<int64_t>(fallback_slot) * raw_block_bytes;
          }
        }
        if (!is_compressed_v3 && raw_block == nullptr) {
          continue;
        }

        if (is_compressed_v3) {
#pragma unroll
          for (int dim_tile = 0; dim_tile < k_dim_tiles; ++dim_tile) {
            byte_v2_decode_k_transposed_tile_to_shared_no_fallback_v3(
                page, k_shared, tid, kv_head, dim_tile, num_kv_heads,
                k_dim_tiles, v_dim_tiles, valid_rows, true);
          }
#pragma unroll
          for (int dim_tile = 0; dim_tile < v_dim_tiles; ++dim_tile) {
            byte_v2_decode_v_rowmajor_tile_to_shared_no_fallback_v3(
                page, v_shared, tid, kv_head, dim_tile, num_kv_heads,
                k_dim_tiles, v_dim_tiles, valid_rows, head_size_v, true);
          }
        } else {
#pragma unroll
          for (int dim_tile = 0; dim_tile < k_dim_tiles; ++dim_tile) {
            byte_v2_decode_k_raw_block_tile_to_shared(
                raw_block, k_shared, tid, kv_head, dim_tile, num_kv_heads,
                head_size, head_size_v, valid_rows);
          }
#pragma unroll
          for (int dim_tile = 0; dim_tile < v_dim_tiles; ++dim_tile) {
            byte_v2_decode_v_raw_block_tile_to_shared(
                raw_block, v_shared, tid, kv_head, dim_tile, num_kv_heads,
                head_size, head_size_v, valid_rows);
          }
        }
        if (is_compressed_v3 && outlier_arena != nullptr &&
            outlier_tile_meta != nullptr) {
#pragma unroll
          for (int dim_tile = 0; dim_tile < k_dim_tiles; ++dim_tile) {
            byte_v2_overlay_k_tile_outliers_to_shared(
                outlier_arena, outlier_tile_bitmap, outlier_tile_meta,
                k_shared, tid, physical_block, kv_head, dim_tile, k_dim_tiles,
                v_dim_tiles, valid_rows, total_tiles, bitmap_words);
          }
#pragma unroll
          for (int dim_tile = 0; dim_tile < v_dim_tiles; ++dim_tile) {
            byte_v2_overlay_v_tile_outliers_to_shared(
                outlier_arena, outlier_tile_bitmap, outlier_tile_meta,
                v_shared, tid, physical_block, kv_head, dim_tile, k_dim_tiles,
                v_dim_tiles, valid_rows, head_size_v, total_tiles,
                bitmap_words);
          }
        }
      }
      __syncthreads();

      if (warp_id < q_per_kv) {
        const int lane_row = lane_id & (kByteV2TileSize - 1);
        const int dim_begin = (lane_id >> 4) * (head_size / 2);
        const int dim_end = dim_begin + (head_size / 2);
        float qk = 0.0f;
#pragma unroll
        for (int dim = dim_begin; dim < dim_end; ++dim) {
          const float q_value =
              byte_v2_wmma_to_float(q_shared[warp_id * head_size + dim]);
          const float k_value = byte_v2_wmma_to_float(
              k_shared[dim * kByteV2TileSize + lane_row]);
          qk += q_value * k_value;
        }
        qk += __shfl_xor_sync(0xffffffffU, qk, kByteV2TileSize);
        const float score =
            (lane_id < kByteV2TileSize && lane_row < valid_rows) ? qk * scale
                                                                 : -FLT_MAX;
        float tile_max = byte_v2_warp_reduce_max(score);
        tile_max = __shfl_sync(0xffffffffU, tile_max, 0);
        const float new_max = max(running_max, tile_max);
        const float old_scale = expf(running_max - new_max);
        const float weight =
            (lane_id < kByteV2TileSize && lane_row < valid_rows)
                ? expf(score - new_max)
                : 0.0f;
        float tile_denom = byte_v2_warp_reduce_sum(weight);
        tile_denom = __shfl_sync(0xffffffffU, tile_denom, 0);

        float pv0 = 0.0f;
        float pv1 = 0.0f;
        float pv2 = 0.0f;
        float pv3 = 0.0f;
#pragma unroll
        for (int row = 0; row < kByteV2TileSize; ++row) {
          const float row_weight = __shfl_sync(0xffffffffU, weight, row);
          if (row < valid_rows) {
            pv0 += row_weight *
                   byte_v2_wmma_to_float(v_shared[row * head_size_v +
                                                  lane_id]);
            pv1 += row_weight *
                   byte_v2_wmma_to_float(v_shared[row * head_size_v +
                                                  lane_id + 32]);
            pv2 += row_weight *
                   byte_v2_wmma_to_float(v_shared[row * head_size_v +
                                                  lane_id + 64]);
            pv3 += row_weight *
                   byte_v2_wmma_to_float(v_shared[row * head_size_v +
                                                  lane_id + 96]);
          }
        }
        acc0 = acc0 * old_scale + pv0;
        acc1 = acc1 * old_scale + pv1;
        acc2 = acc2 * old_scale + pv2;
        acc3 = acc3 * old_scale + pv3;
        denom = denom * old_scale + tile_denom;
        running_max = new_max;
      }
      __syncthreads();
    }
  }

  if (warp_id < q_per_kv) {
    const int q_head = first_q_head + warp_id;
    float* partial =
        partial_output +
        (((req_idx * num_heads + q_head) * num_kv_splits + split_idx) *
         partial_stride);
    const float inv_denom = denom > 0.0f ? (1.0f / denom) : 0.0f;
    partial[lane_id] = acc0 * inv_denom;
    partial[lane_id + 32] = acc1 * inv_denom;
    partial[lane_id + 64] = acc2 * inv_denom;
    partial[lane_id + 96] = acc3 * inv_denom;
    if (lane_id == 0) {
      partial[head_size_v] =
          denom > 0.0f ? running_max + logf(denom) : -FLT_MAX;
    }
  }
#endif
}

template <typename block_table_t, typename seq_lens_t, bool OutlierOverlay>
__global__ void
byte_v2_paged_decode_attention_gqa4_h128_v4_block128_split_stage1_kernel(
    const uint16_t* __restrict__ query, const uint8_t* __restrict__ kv_cache,
    const block_table_t* __restrict__ block_table,
    const seq_lens_t* __restrict__ seq_lens, float* __restrict__ partial_output,
    const int32_t* __restrict__ outlier_arena,
    const int32_t* __restrict__ outlier_tile_bitmap,
    const int32_t* __restrict__ outlier_tile_meta, const float scale,
    const int num_decode_tokens, const int page_size_bytes,
    const int num_kv_splits, const int64_t block_table_stride0,
    const int64_t block_table_stride1, const int64_t seq_lens_stride0) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  constexpr int macro_pages = 8;
  constexpr int macro_tokens = macro_pages * kByteV2TileSize;
  constexpr int num_heads = 32;
  constexpr int num_kv_heads = 8;
  constexpr int head_size = 128;
  constexpr int head_size_v = 128;
  constexpr int q_per_kv = 4;
  constexpr int k_dim_tiles = head_size / kByteV2TileSize;
  constexpr int v_dim_tiles = head_size_v / kByteV2TileSize;
  constexpr int total_tiles = num_kv_heads * (k_dim_tiles + v_dim_tiles);
  constexpr int bitmap_words = (total_tiles + 31) / 32;
  constexpr int partial_stride = head_size_v + 1;

  const int req_idx = blockIdx.x;
  const int kv_head = blockIdx.y;
  const int split_idx = blockIdx.z;
  const int tid = threadIdx.x;
  const int warp_id = tid / warpSize;
  const int lane_id = tid % warpSize;
  if (req_idx >= num_decode_tokens || kv_head >= num_kv_heads ||
      split_idx >= num_kv_splits) {
    return;
  }

  __shared__ __align__(16) __nv_bfloat16 q_shared[q_per_kv * head_size];
  __shared__ __align__(16) __nv_bfloat16 k_shared[head_size *
                                                  kByteV2TileSize];
  __shared__ __align__(16) __nv_bfloat16 v_shared[kByteV2TileSize *
                                                  head_size_v];
  __shared__ float macro_scores[q_per_kv * macro_tokens];
  __shared__ int macro_physical_blocks[macro_pages];
  __shared__ int macro_valid_rows[macro_pages];
  __shared__ int macro_compressed_v3[macro_pages];

  const int seq_len = static_cast<int>(seq_lens[req_idx * seq_lens_stride0]);
  const int first_q_head = kv_head * q_per_kv;

  if (seq_len <= 0) {
#pragma unroll
    for (int q = 0; q < q_per_kv; ++q) {
      const int q_head = first_q_head + q;
      float* partial =
          partial_output +
          (((req_idx * num_heads + q_head) * num_kv_splits + split_idx) *
           partial_stride);
      for (int d = tid; d < head_size_v; d += blockDim.x) {
        partial[d] = 0.0f;
      }
      if (tid == 0) {
        partial[head_size_v] = -FLT_MAX;
      }
    }
    return;
  }

  const int logical_blocks =
      (seq_len + kByteV2TileSize - 1) / kByteV2TileSize;
  const int blocks_per_split =
      (logical_blocks + num_kv_splits - 1) / num_kv_splits;
  const int start_block = split_idx * blocks_per_split;
  const int end_block = min(logical_blocks, start_block + blocks_per_split);
  if (start_block >= end_block) {
#pragma unroll
    for (int q = 0; q < q_per_kv; ++q) {
      const int q_head = first_q_head + q;
      float* partial =
          partial_output +
          (((req_idx * num_heads + q_head) * num_kv_splits + split_idx) *
           partial_stride);
      for (int d = tid; d < head_size_v; d += blockDim.x) {
        partial[d] = 0.0f;
      }
      if (tid == 0) {
        partial[head_size_v] = -FLT_MAX;
      }
    }
    return;
  }

  for (int idx = tid; idx < q_per_kv * head_size; idx += blockDim.x) {
    const int q_row = idx / head_size;
    const int dim = idx % head_size;
    const int q_head = first_q_head + q_row;
    const uint16_t q_bits =
        query[(req_idx * num_heads + q_head) * head_size + dim];
    q_shared[idx] = byte_v2_bf16_bits_to_wmma(q_bits);
  }
  __syncthreads();

  float running_max = -FLT_MAX;
  float denom = 0.0f;
  float acc0 = 0.0f;
  float acc1 = 0.0f;
  float acc2 = 0.0f;
  float acc3 = 0.0f;

  for (int macro_begin = start_block; macro_begin < end_block;
       macro_begin += macro_pages) {
    for (int idx = tid; idx < q_per_kv * macro_tokens; idx += blockDim.x) {
      macro_scores[idx] = -FLT_MAX;
    }
    byte_v2_v4_load_macro_descriptor<macro_pages>(
        kv_cache, block_table, req_idx, seq_len, macro_begin, end_block,
        page_size_bytes, block_table_stride0, block_table_stride1,
        macro_physical_blocks, macro_valid_rows, macro_compressed_v3);

#pragma unroll
    for (int macro_idx = 0; macro_idx < macro_pages; ++macro_idx) {
      const int valid_rows = macro_valid_rows[macro_idx];
      if (valid_rows <= 0 || macro_compressed_v3[macro_idx] == 0) {
        continue;
      }
      const int physical_block = macro_physical_blocks[macro_idx];
      const uint8_t* page = kv_cache + physical_block * page_size_bytes;

#pragma unroll
      for (int dim_tile = 0; dim_tile < k_dim_tiles; ++dim_tile) {
        byte_v2_decode_k_transposed_tile_to_shared_no_fallback_v3(
            page, k_shared, tid, kv_head, dim_tile, num_kv_heads, k_dim_tiles,
            v_dim_tiles, valid_rows, true);
      }
      if constexpr (OutlierOverlay) {
#pragma unroll
        for (int dim_tile = 0; dim_tile < k_dim_tiles; ++dim_tile) {
          byte_v2_overlay_k_tile_outliers_to_shared(
              outlier_arena, outlier_tile_bitmap, outlier_tile_meta, k_shared,
              tid, physical_block, kv_head, dim_tile, k_dim_tiles, v_dim_tiles,
              valid_rows, total_tiles, bitmap_words);
        }
      }
      __syncthreads();

      if (warp_id < q_per_kv) {
        const int lane_row = lane_id & (kByteV2TileSize - 1);
        const int dim_begin = (lane_id >> 4) * (head_size / 2);
        const int dim_end = dim_begin + (head_size / 2);
        float qk = 0.0f;
#pragma unroll
        for (int dim = dim_begin; dim < dim_end; ++dim) {
          const float q_value =
              byte_v2_wmma_to_float(q_shared[warp_id * head_size + dim]);
          const float k_value = byte_v2_wmma_to_float(
              k_shared[dim * kByteV2TileSize + lane_row]);
          qk += q_value * k_value;
        }
        qk += __shfl_xor_sync(0xffffffffU, qk, kByteV2TileSize);
        if (lane_id < kByteV2TileSize && lane_row < valid_rows) {
          macro_scores[warp_id * macro_tokens +
                       macro_idx * kByteV2TileSize + lane_row] = qk * scale;
        }
      }
      __syncthreads();
    }

    if (warp_id < q_per_kv) {
      float local_max = -FLT_MAX;
      for (int token = lane_id; token < macro_tokens; token += warpSize) {
        local_max =
            max(local_max, macro_scores[warp_id * macro_tokens + token]);
      }
      float macro_max = byte_v2_warp_reduce_max(local_max);
      macro_max = __shfl_sync(0xffffffffU, macro_max, 0);
      const float new_max = max(running_max, macro_max);
      const float old_scale = expf(running_max - new_max);
      float local_denom = 0.0f;
      for (int token = lane_id; token < macro_tokens; token += warpSize) {
        const int offset = warp_id * macro_tokens + token;
        const float score = macro_scores[offset];
        const float weight = score > -FLT_MAX ? expf(score - new_max) : 0.0f;
        macro_scores[offset] = weight;
        local_denom += weight;
      }
      float macro_denom = byte_v2_warp_reduce_sum(local_denom);
      macro_denom = __shfl_sync(0xffffffffU, macro_denom, 0);
      acc0 *= old_scale;
      acc1 *= old_scale;
      acc2 *= old_scale;
      acc3 *= old_scale;
      denom = denom * old_scale + macro_denom;
      running_max = new_max;
    }
    __syncthreads();

#pragma unroll
    for (int macro_idx = 0; macro_idx < macro_pages; ++macro_idx) {
      const int valid_rows = macro_valid_rows[macro_idx];
      if (valid_rows <= 0 || macro_compressed_v3[macro_idx] == 0) {
        continue;
      }
      const int physical_block = macro_physical_blocks[macro_idx];
      const uint8_t* page = kv_cache + physical_block * page_size_bytes;

#pragma unroll
      for (int dim_tile = 0; dim_tile < v_dim_tiles; ++dim_tile) {
        byte_v2_decode_v_rowmajor_tile_to_shared_no_fallback_v3(
            page, v_shared, tid, kv_head, dim_tile, num_kv_heads, k_dim_tiles,
            v_dim_tiles, valid_rows, head_size_v, true);
      }
      if constexpr (OutlierOverlay) {
#pragma unroll
        for (int dim_tile = 0; dim_tile < v_dim_tiles; ++dim_tile) {
          byte_v2_overlay_v_tile_outliers_to_shared(
              outlier_arena, outlier_tile_bitmap, outlier_tile_meta, v_shared,
              tid, physical_block, kv_head, dim_tile, k_dim_tiles, v_dim_tiles,
              valid_rows, head_size_v, total_tiles, bitmap_words);
        }
      }
      __syncthreads();

      if (warp_id < q_per_kv) {
        float pv0 = 0.0f;
        float pv1 = 0.0f;
        float pv2 = 0.0f;
        float pv3 = 0.0f;
#pragma unroll
        for (int row = 0; row < kByteV2TileSize; ++row) {
          if (row < valid_rows) {
            const float row_weight =
                macro_scores[warp_id * macro_tokens +
                             macro_idx * kByteV2TileSize + row];
            pv0 += row_weight *
                   byte_v2_wmma_to_float(v_shared[row * head_size_v +
                                                  lane_id]);
            pv1 += row_weight *
                   byte_v2_wmma_to_float(v_shared[row * head_size_v +
                                                  lane_id + 32]);
            pv2 += row_weight *
                   byte_v2_wmma_to_float(v_shared[row * head_size_v +
                                                  lane_id + 64]);
            pv3 += row_weight *
                   byte_v2_wmma_to_float(v_shared[row * head_size_v +
                                                  lane_id + 96]);
          }
        }
        acc0 += pv0;
        acc1 += pv1;
        acc2 += pv2;
        acc3 += pv3;
      }
      __syncthreads();
    }
  }

  if (warp_id < q_per_kv) {
    const int q_head = first_q_head + warp_id;
    float* partial =
        partial_output +
        (((req_idx * num_heads + q_head) * num_kv_splits + split_idx) *
         partial_stride);
    const float inv_denom = denom > 0.0f ? (1.0f / denom) : 0.0f;
    partial[lane_id] = acc0 * inv_denom;
    partial[lane_id + 32] = acc1 * inv_denom;
    partial[lane_id + 64] = acc2 * inv_denom;
    partial[lane_id + 96] = acc3 * inv_denom;
    if (lane_id == 0) {
      partial[head_size_v] =
          denom > 0.0f ? running_max + logf(denom) : -FLT_MAX;
    }
  }
#endif
}

	template <typename block_table_t, typename seq_lens_t,
	          bool metadata_fastpath, bool no_outlier_metadata,
	          bool no_fallback_metadata>
__global__ void byte_v2_paged_decode_attention_gqa_cute_split_stage1_kernel(
    const uint16_t* __restrict__ query, const uint8_t* __restrict__ kv_cache,
    const block_table_t* __restrict__ block_table,
    const seq_lens_t* __restrict__ seq_lens, float* __restrict__ partial_output,
    const uint8_t* __restrict__ fallback_pool,
    const int32_t* __restrict__ fallback_block_ids,
    const int32_t* __restrict__ fallback_tile_ids,
    const int32_t* __restrict__ outlier_arena,
    const int32_t* __restrict__ outlier_block_flags,
    const int32_t* __restrict__ outlier_tile_bitmap,
    const int32_t* __restrict__ outlier_tile_meta,
    const float scale, const int num_decode_tokens, const int num_heads,
    const int num_kv_heads, const int page_size_bytes,
    const int raw_block_bytes, const int num_kv_splits,
    const int64_t block_table_stride0, const int64_t block_table_stride1,
    const int64_t seq_lens_stride0, const bool use_tile_fastpath,
    const bool use_aligned_u16_payload_load,
    const bool use_v3_warp_stripe_load,
    const bool use_v3_cp_async_stage, const int early_exit_mode) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  constexpr int head_size = kByteV2MaxHeadSize;
  constexpr int head_size_v = kByteV2MaxHeadSize;
  constexpr int q_per_kv = 4;
  constexpr int k_dim_tiles = head_size / kByteV2TileSize;
  constexpr int v_dim_tiles = head_size_v / kByteV2TileSize;

  const int req_idx = blockIdx.x;
  const int kv_head = blockIdx.y;
  const int split_idx = blockIdx.z;
  const int tid = threadIdx.x;
  if (req_idx >= num_decode_tokens || kv_head >= num_kv_heads ||
      split_idx >= num_kv_splits) {
    return;
  }

  __shared__ __align__(16) __nv_bfloat16 q_shared[kByteV2TileSize *
                                                  head_size];
  __shared__ __align__(16) __nv_bfloat16 k_shared[head_size *
                                                  kByteV2TileSize];
  __shared__ __align__(16) __nv_bfloat16 v_shared[kByteV2TileSize *
                                                  head_size_v];
  __shared__ __align__(16) __nv_bfloat16 p_shared[kByteV2TileSize *
                                                  kByteV2TileSize];
  __shared__ float scores[kByteV2TileSize * kByteV2TileSize];
  __shared__ float pv_shared[kByteV2TileSize * kByteV2TileSize];
  __shared__ __align__(16) uint8_t v3_cp_stage[2 *
                                               kByteV2TilePayloadBytesV3];
  __shared__ float tile_acc_factor[q_per_kv];
  __shared__ float lse_shared[q_per_kv];
  __shared__ float inv_denom[q_per_kv];
  __shared__ float running_max_shared[q_per_kv];
  __shared__ float denom_shared[q_per_kv];

  const int seq_len = static_cast<int>(seq_lens[req_idx * seq_lens_stride0]);
  const int first_q_head = kv_head * q_per_kv;
  const int partial_stride = head_size_v + 1;

  float acc[q_per_kv];
#pragma unroll
  for (int q = 0; q < q_per_kv; ++q) {
    acc[q] = 0.0f;
  }
  uint32_t profile_checksum =
      static_cast<uint32_t>(tid) ^
      (static_cast<uint32_t>(req_idx) << 8) ^
      (static_cast<uint32_t>(kv_head) << 16) ^
      (static_cast<uint32_t>(split_idx) << 24);
  const bool fine_profile_mode =
      early_exit_mode >= 5 && early_exit_mode <= 7;

  if (seq_len <= 0) {
    for (int q = 0; q < q_per_kv; ++q) {
      const int q_head = first_q_head + q;
      if (q_head >= num_heads) {
        continue;
      }
      float* partial =
          partial_output +
          (((req_idx * num_heads + q_head) * num_kv_splits + split_idx) *
           partial_stride);
      for (int d = tid; d < head_size_v; d += blockDim.x) {
        partial[d] = 0.0f;
      }
      if (tid == 0) {
        partial[head_size_v] = -FLT_MAX;
      }
    }
    return;
  }

  const int logical_blocks =
      (seq_len + kByteV2TileSize - 1) / kByteV2TileSize;
  const int blocks_per_split =
      (logical_blocks + num_kv_splits - 1) / num_kv_splits;
  const int start_block = split_idx * blocks_per_split;
  const int end_block = min(logical_blocks, start_block + blocks_per_split);
  if (start_block >= end_block) {
    for (int q = 0; q < q_per_kv; ++q) {
      const int q_head = first_q_head + q;
      if (q_head >= num_heads) {
        continue;
      }
      float* partial =
          partial_output +
          (((req_idx * num_heads + q_head) * num_kv_splits + split_idx) *
           partial_stride);
      for (int d = tid; d < head_size_v; d += blockDim.x) {
        partial[d] = 0.0f;
      }
      if (tid == 0) {
        partial[head_size_v] = -FLT_MAX;
      }
    }
    return;
  }

  if (tid < q_per_kv) {
    running_max_shared[tid] = -FLT_MAX;
    denom_shared[tid] = 0.0f;
  }
  __syncthreads();

  for (int idx = tid; idx < kByteV2TileSize * head_size; idx += blockDim.x) {
    const int q_row = idx / head_size;
    const int dim = idx % head_size;
    uint16_t q_bits = 0;
    if (q_row < q_per_kv) {
      const int q_head = first_q_head + q_row;
      if (q_head < num_heads) {
        q_bits = query[(req_idx * num_heads + q_head) * head_size + dim];
      }
    }
    q_shared[idx] = byte_v2_bf16_bits_to_wmma(q_bits);
  }
  __syncthreads();

  for (int logical_block = start_block; logical_block < end_block;
       ++logical_block) {
    const int token_base = logical_block * kByteV2TileSize;
    const int physical_block = static_cast<int>(
        block_table[req_idx * block_table_stride0 +
                    logical_block * block_table_stride1]);
    const uint8_t* page = kv_cache + physical_block * page_size_bytes;
    const int page_valid_rows =
        static_cast<int>(page[kByteV2PageValidRowsOffset]);
    const int seq_rows = min(kByteV2TileSize, seq_len - token_base);
    const int valid_rows = min(seq_rows, page_valid_rows);
    if (valid_rows <= 0) {
      continue;
    }

	    const uint8_t page_status = page[kByteV2PageStatusOffset];
	    if (fine_profile_mode) {
      if constexpr (!metadata_fastpath) {
        if (page_status == kByteV2PageStatusCompressed) {
#pragma unroll
          for (int dim_tile = 0; dim_tile < k_dim_tiles; ++dim_tile) {
            profile_checksum = byte_v2_profile_mix_u32(
                profile_checksum,
                byte_v2_profile_compressed_tile_layout(
                    page, false, tid, kv_head, dim_tile, num_kv_heads,
                    k_dim_tiles, v_dim_tiles, early_exit_mode,
                    use_aligned_u16_payload_load,
                    use_v3_warp_stripe_load));
          }
#pragma unroll
          for (int dim_tile = 0; dim_tile < v_dim_tiles; ++dim_tile) {
            profile_checksum = byte_v2_profile_mix_u32(
                profile_checksum,
                byte_v2_profile_compressed_tile_layout(
                    page, true, tid, kv_head, dim_tile, num_kv_heads,
                    k_dim_tiles, v_dim_tiles, early_exit_mode,
                    use_aligned_u16_payload_load,
                    use_v3_warp_stripe_load));
          }
        }
      } else {
        const int total_tiles = num_kv_heads * (k_dim_tiles + v_dim_tiles);
        int bitmap_words = 0;
        const int32_t* block_outlier_arena = nullptr;
        const int32_t* block_outlier_tile_bitmap = nullptr;
        const int32_t* block_outlier_tile_meta = nullptr;
        if constexpr (!no_outlier_metadata) {
          bitmap_words = byte_v2_outlier_tile_bitmap_words(total_tiles);
          const bool block_may_have_outliers =
              byte_v2_may_have_block_outliers(outlier_block_flags,
                                              physical_block);
          block_outlier_arena =
              block_may_have_outliers ? outlier_arena : nullptr;
          block_outlier_tile_bitmap =
              block_may_have_outliers ? outlier_tile_bitmap : nullptr;
          block_outlier_tile_meta =
              block_may_have_outliers ? outlier_tile_meta : nullptr;
        }

        if (page_status == kByteV2PageStatusCompressed) {
#pragma unroll
          for (int dim_tile = 0; dim_tile < k_dim_tiles; ++dim_tile) {
            const int tile_idx = byte_v2_tile_index(
                false, kv_head, dim_tile, k_dim_tiles, v_dim_tiles);
	            const uint8_t tile_fallback =
	                no_fallback_metadata
	                    ? 0
	                    : byte_v2_tile_fallback_flag(page, false, kv_head,
	                                                 dim_tile, k_dim_tiles,
	                                                 v_dim_tiles);
            profile_checksum = byte_v2_profile_mix_u32(
                profile_checksum,
                static_cast<uint32_t>(tile_fallback) ^
                    static_cast<uint32_t>(tile_idx));
            if (tile_fallback == 0) {
              profile_checksum = byte_v2_profile_mix_u32(
                  profile_checksum,
                  byte_v2_profile_compressed_tile_layout(
                      page, false, tid, kv_head, dim_tile, num_kv_heads,
                      k_dim_tiles, v_dim_tiles, early_exit_mode,
                      use_aligned_u16_payload_load,
                      use_v3_warp_stripe_load));
              if constexpr (!no_outlier_metadata) {
                if (block_outlier_arena != nullptr &&
                    block_outlier_tile_meta != nullptr &&
                    byte_v2_may_have_tile_outliers(block_outlier_tile_bitmap,
                                                   physical_block,
                                                   bitmap_words, tile_idx)) {
                  const int32_t meta =
                      block_outlier_tile_meta[physical_block * total_tiles +
                                              tile_idx];
                  profile_checksum = byte_v2_profile_mix_u32(
                      profile_checksum, static_cast<uint32_t>(meta));
                  if (early_exit_mode >= 6 && meta >= 0) {
                    const int count = byte_v2_outlier_meta_count(meta);
                    const int offset = byte_v2_outlier_meta_offset(meta);
                    for (int i = tid; i < count; i += blockDim.x) {
                      profile_checksum = byte_v2_profile_mix_u32(
                          profile_checksum,
                          static_cast<uint32_t>(
                              block_outlier_arena[offset + i]));
                    }
                  }
                }
              }
            } else {
              profile_checksum = byte_v2_profile_mix_u32(
                  profile_checksum,
                  byte_v2_profile_tile_fallback(
                      fallback_pool, fallback_tile_ids, false, tid,
                      physical_block, kv_head, dim_tile, k_dim_tiles,
                      v_dim_tiles, valid_rows, total_tiles,
                      early_exit_mode));
            }
          }
#pragma unroll
          for (int dim_tile = 0; dim_tile < v_dim_tiles; ++dim_tile) {
            const int tile_idx = byte_v2_tile_index(
                true, kv_head, dim_tile, k_dim_tiles, v_dim_tiles);
		            const uint8_t tile_fallback =
		                no_fallback_metadata
		                    ? 0
		                    : byte_v2_tile_fallback_flag(page, true, kv_head,
		                                                 dim_tile, k_dim_tiles,
		                                                 v_dim_tiles);
            profile_checksum = byte_v2_profile_mix_u32(
                profile_checksum,
                static_cast<uint32_t>(tile_fallback) ^
                    static_cast<uint32_t>(tile_idx));
            if (tile_fallback == 0) {
              profile_checksum = byte_v2_profile_mix_u32(
                  profile_checksum,
                  byte_v2_profile_compressed_tile_layout(
                      page, true, tid, kv_head, dim_tile, num_kv_heads,
                      k_dim_tiles, v_dim_tiles, early_exit_mode,
                      use_aligned_u16_payload_load,
                      use_v3_warp_stripe_load));
              if constexpr (!no_outlier_metadata) {
                if (block_outlier_arena != nullptr &&
                    block_outlier_tile_meta != nullptr &&
                    byte_v2_may_have_tile_outliers(block_outlier_tile_bitmap,
                                                   physical_block,
                                                   bitmap_words, tile_idx)) {
                  const int32_t meta =
                      block_outlier_tile_meta[physical_block * total_tiles +
                                              tile_idx];
                  profile_checksum = byte_v2_profile_mix_u32(
                      profile_checksum, static_cast<uint32_t>(meta));
                  if (early_exit_mode >= 6 && meta >= 0) {
                    const int count = byte_v2_outlier_meta_count(meta);
                    const int offset = byte_v2_outlier_meta_offset(meta);
                    for (int i = tid; i < count; i += blockDim.x) {
                      profile_checksum = byte_v2_profile_mix_u32(
                          profile_checksum,
                          static_cast<uint32_t>(
                              block_outlier_arena[offset + i]));
                    }
                  }
                }
              }
            } else {
              profile_checksum = byte_v2_profile_mix_u32(
                  profile_checksum,
                  byte_v2_profile_tile_fallback(
                      fallback_pool, fallback_tile_ids, true, tid,
                      physical_block, kv_head, dim_tile, k_dim_tiles,
                      v_dim_tiles, valid_rows, total_tiles,
                      early_exit_mode));
            }
          }
        } else if (page_status == kByteV2PageStatusRawFallback) {
          const uint8_t* raw_block = page + kByteV2PageHeaderBytes;
          if (fallback_pool != nullptr && fallback_block_ids != nullptr) {
            const int fallback_slot = fallback_block_ids[physical_block];
            if (fallback_slot >= 0) {
              raw_block = fallback_pool + fallback_slot * raw_block_bytes;
            }
            profile_checksum = byte_v2_profile_mix_u32(
                profile_checksum, static_cast<uint32_t>(fallback_slot));
          }
#pragma unroll
          for (int dim_tile = 0; dim_tile < k_dim_tiles; ++dim_tile) {
            profile_checksum = byte_v2_profile_mix_u32(
                profile_checksum,
                byte_v2_profile_raw_tile(
                    raw_block, false, tid, kv_head, dim_tile, num_kv_heads,
                    head_size, head_size_v, valid_rows, early_exit_mode));
          }
#pragma unroll
          for (int dim_tile = 0; dim_tile < v_dim_tiles; ++dim_tile) {
            profile_checksum = byte_v2_profile_mix_u32(
                profile_checksum,
                byte_v2_profile_raw_tile(
                    raw_block, true, tid, kv_head, dim_tile, num_kv_heads,
                    head_size, head_size_v, valid_rows, early_exit_mode));
          }
        }
      }
      continue;
    }
    if constexpr (!metadata_fastpath) {
      if (page_status != kByteV2PageStatusCompressed) {
        continue;
      }
      if (use_v3_cp_async_stage && byte_v2_page_has_v3_payload(page)) {
        byte_v2_decode_k_tiles_to_shared_v3_cp_async(
            page, v3_cp_stage, k_shared, tid, kv_head, num_kv_heads,
            k_dim_tiles, v_dim_tiles, valid_rows);
        byte_v2_decode_v_tiles_to_shared_v3_cp_async(
            page, v3_cp_stage, v_shared, tid, kv_head, num_kv_heads,
            k_dim_tiles, v_dim_tiles, valid_rows, head_size_v);
      } else {
        for (int dim_tile = 0; dim_tile < k_dim_tiles; ++dim_tile) {
          byte_v2_decode_k_transposed_tile_to_shared_no_fallback_layout(
              page, k_shared, tid, kv_head, dim_tile, num_kv_heads,
              k_dim_tiles, v_dim_tiles, valid_rows,
              use_aligned_u16_payload_load, use_v3_warp_stripe_load);
        }
#pragma unroll
        for (int dim_tile = 0; dim_tile < v_dim_tiles; ++dim_tile) {
          byte_v2_decode_v_rowmajor_tile_to_shared_no_fallback_layout(
              page, v_shared, tid, kv_head, dim_tile, num_kv_heads,
              k_dim_tiles, v_dim_tiles, valid_rows, head_size_v,
              use_aligned_u16_payload_load, use_v3_warp_stripe_load);
        }
      }
    } else {
      const int total_tiles = num_kv_heads * (k_dim_tiles + v_dim_tiles);
      int bitmap_words = 0;
      const int32_t* block_outlier_arena = nullptr;
      const int32_t* block_outlier_tile_bitmap = nullptr;
      const int32_t* block_outlier_tile_meta = nullptr;
      if constexpr (!no_outlier_metadata) {
        bitmap_words = byte_v2_outlier_tile_bitmap_words(total_tiles);
        const bool block_may_have_outliers =
            byte_v2_may_have_block_outliers(outlier_block_flags,
                                            physical_block);
        block_outlier_arena =
            block_may_have_outliers ? outlier_arena : nullptr;
        block_outlier_tile_bitmap =
            block_may_have_outliers ? outlier_tile_bitmap : nullptr;
        block_outlier_tile_meta =
            block_may_have_outliers ? outlier_tile_meta : nullptr;
      }

      if (page_status == kByteV2PageStatusCompressed && use_tile_fastpath) {
        bool use_v3_cp_async_page = false;
		        if (use_v3_cp_async_stage && byte_v2_page_has_v3_payload(page)) {
		          if constexpr (no_fallback_metadata) {
		            use_v3_cp_async_page = true;
		          } else {
		          const int meta_start = byte_v2_v3_kv_head_meta_start(kv_head);
		          const uint16_t k_fallback_mask =
		              byte_v2_load_u16(page + meta_start + 16);
		          const uint16_t v_fallback_mask =
		              byte_v2_load_u16(page + meta_start + 18);
		          use_v3_cp_async_page =
		              (k_fallback_mask == 0) && (v_fallback_mask == 0);
		          }
		        }
        if (use_v3_cp_async_page) {
          byte_v2_decode_k_tiles_to_shared_v3_cp_async(
              page, v3_cp_stage, k_shared, tid, kv_head, num_kv_heads,
              k_dim_tiles, v_dim_tiles, valid_rows);
          byte_v2_decode_v_tiles_to_shared_v3_cp_async(
              page, v3_cp_stage, v_shared, tid, kv_head, num_kv_heads,
              k_dim_tiles, v_dim_tiles, valid_rows, head_size_v);
          if constexpr (!no_outlier_metadata) {
#pragma unroll
            for (int dim_tile = 0; dim_tile < k_dim_tiles; ++dim_tile) {
              byte_v2_overlay_k_tile_outliers_to_shared(
                  block_outlier_arena, block_outlier_tile_bitmap,
                  block_outlier_tile_meta, k_shared, tid, physical_block,
                  kv_head, dim_tile, k_dim_tiles, v_dim_tiles, valid_rows,
                  total_tiles, bitmap_words);
            }
#pragma unroll
            for (int dim_tile = 0; dim_tile < v_dim_tiles; ++dim_tile) {
              byte_v2_overlay_v_tile_outliers_to_shared(
                  block_outlier_arena, block_outlier_tile_bitmap,
                  block_outlier_tile_meta, v_shared, tid, physical_block,
                  kv_head, dim_tile, k_dim_tiles, v_dim_tiles, valid_rows,
                  head_size_v, total_tiles, bitmap_words);
            }
          }
        } else {
#pragma unroll
          for (int dim_tile = 0; dim_tile < k_dim_tiles; ++dim_tile) {
		            const int tile_fallback =
		                no_fallback_metadata
		                    ? 0
		                    : byte_v2_tile_fallback_flag(page, false, kv_head,
		                                                 dim_tile, k_dim_tiles,
		                                                 v_dim_tiles);
            if (tile_fallback == 0) {
              byte_v2_decode_k_transposed_tile_to_shared_no_fallback_layout(
                  page, k_shared, tid, kv_head, dim_tile, num_kv_heads,
                  k_dim_tiles, v_dim_tiles, valid_rows,
                  use_aligned_u16_payload_load, use_v3_warp_stripe_load);
              if constexpr (!no_outlier_metadata) {
                byte_v2_overlay_k_tile_outliers_to_shared(
                    block_outlier_arena, block_outlier_tile_bitmap,
                    block_outlier_tile_meta, k_shared, tid, physical_block,
                    kv_head, dim_tile, k_dim_tiles, v_dim_tiles, valid_rows,
                    total_tiles, bitmap_words);
              }
            } else {
              byte_v2_decode_k_transposed_tile_to_shared_tile_fallback(
                  fallback_pool, fallback_tile_ids, k_shared, tid,
                  physical_block, kv_head, dim_tile, k_dim_tiles, v_dim_tiles,
                  valid_rows, total_tiles);
            }
          }
#pragma unroll
          for (int dim_tile = 0; dim_tile < v_dim_tiles; ++dim_tile) {
		            const int tile_fallback =
		                no_fallback_metadata
		                    ? 0
		                    : byte_v2_tile_fallback_flag(page, true, kv_head,
		                                                 dim_tile, k_dim_tiles,
		                                                 v_dim_tiles);
            if (tile_fallback == 0) {
              byte_v2_decode_v_rowmajor_tile_to_shared_no_fallback_layout(
                  page, v_shared, tid, kv_head, dim_tile, num_kv_heads,
                  k_dim_tiles, v_dim_tiles, valid_rows, head_size_v,
                  use_aligned_u16_payload_load, use_v3_warp_stripe_load);
              if constexpr (!no_outlier_metadata) {
                byte_v2_overlay_v_tile_outliers_to_shared(
                    block_outlier_arena, block_outlier_tile_bitmap,
                    block_outlier_tile_meta, v_shared, tid, physical_block,
                    kv_head, dim_tile, k_dim_tiles, v_dim_tiles, valid_rows,
                    head_size_v, total_tiles, bitmap_words);
              }
            } else {
              byte_v2_decode_v_rowmajor_tile_to_shared_tile_fallback(
                  fallback_pool, fallback_tile_ids, v_shared, tid,
                  physical_block, kv_head, dim_tile, k_dim_tiles, v_dim_tiles,
                  valid_rows, head_size_v, total_tiles);
            }
          }
        }
      } else if (page_status == kByteV2PageStatusCompressed) {
        for (int idx = tid; idx < head_size * kByteV2TileSize;
             idx += blockDim.x) {
          const int dim = idx / kByteV2TileSize;
          const int row = idx % kByteV2TileSize;
          uint16_t k_bits = 0;
          if (row < valid_rows) {
            k_bits = byte_v2_load_compressed_bits(
                page, false, row, kv_head, dim, k_dim_tiles, v_dim_tiles,
                fallback_pool, fallback_tile_ids, block_outlier_arena,
                block_outlier_tile_bitmap, block_outlier_tile_meta,
                physical_block, total_tiles, bitmap_words);
          }
          k_shared[idx] = byte_v2_bf16_bits_to_wmma(k_bits);
        }
        for (int idx = tid; idx < kByteV2TileSize * head_size_v;
             idx += blockDim.x) {
          const int row = idx / head_size_v;
          const int dim = idx % head_size_v;
          uint16_t v_bits = 0;
          if (row < valid_rows) {
            v_bits = byte_v2_load_compressed_bits(
                page, true, row, kv_head, dim, k_dim_tiles, v_dim_tiles,
                fallback_pool, fallback_tile_ids, block_outlier_arena,
                block_outlier_tile_bitmap, block_outlier_tile_meta,
                physical_block, total_tiles, bitmap_words);
          }
          v_shared[idx] = byte_v2_bf16_bits_to_wmma(v_bits);
        }
      } else if (page_status == kByteV2PageStatusRawFallback) {
        const uint8_t* raw_block = page + kByteV2PageHeaderBytes;
        if (fallback_pool != nullptr && fallback_block_ids != nullptr) {
          const int fallback_slot = fallback_block_ids[physical_block];
          if (fallback_slot >= 0) {
            raw_block = fallback_pool + fallback_slot * raw_block_bytes;
          }
        }
        for (int idx = tid; idx < head_size * kByteV2TileSize;
             idx += blockDim.x) {
          const int dim = idx / kByteV2TileSize;
          const int row = idx % kByteV2TileSize;
          uint16_t k_bits = 0;
          if (row < valid_rows) {
            k_bits = byte_v2_load_raw_bits_from_block(
                raw_block, false, row, kv_head, dim, num_kv_heads, head_size,
                head_size_v);
          }
          k_shared[idx] = byte_v2_bf16_bits_to_wmma(k_bits);
        }
        for (int idx = tid; idx < kByteV2TileSize * head_size_v;
             idx += blockDim.x) {
          const int row = idx / head_size_v;
          const int dim = idx % head_size_v;
          uint16_t v_bits = 0;
          if (row < valid_rows) {
            v_bits = byte_v2_load_raw_bits_from_block(
                raw_block, true, row, kv_head, dim, num_kv_heads, head_size,
                head_size_v);
          }
          v_shared[idx] = byte_v2_bf16_bits_to_wmma(v_bits);
        }
      } else {
        continue;
      }
    }
    __syncthreads();
    if (early_exit_mode == 1) {
      continue;
    }

    if (tid < warpSize) {
      nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16,
                             __nv_bfloat16, nvcuda::wmma::row_major>
          a_frag;
      nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16,
                             __nv_bfloat16, nvcuda::wmma::row_major>
          b_frag;
      nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float>
          c_frag;
      nvcuda::wmma::fill_fragment(c_frag, 0.0f);
#pragma unroll
      for (int dim_base = 0; dim_base < head_size;
           dim_base += kByteV2TileSize) {
        nvcuda::wmma::load_matrix_sync(a_frag, q_shared + dim_base,
                                       head_size);
        nvcuda::wmma::load_matrix_sync(
            b_frag, k_shared + dim_base * kByteV2TileSize, kByteV2TileSize);
        nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
      }
      nvcuda::wmma::store_matrix_sync(scores, c_frag, kByteV2TileSize,
                                      nvcuda::wmma::mem_row_major);
    }
    __syncthreads();
    if (early_exit_mode == 2) {
      continue;
    }

    const __nv_bfloat16 zero = byte_v2_bf16_bits_to_wmma(0);
    for (int idx = tid; idx < kByteV2TileSize * kByteV2TileSize;
         idx += blockDim.x) {
      p_shared[idx] = zero;
    }
    __syncthreads();

    const int warp_id = tid / warpSize;
    const int lane_id = tid % warpSize;
    if (warp_id < q_per_kv) {
      const float score =
          lane_id < valid_rows
              ? scores[warp_id * kByteV2TileSize + lane_id] * scale
              : -FLT_MAX;
      const float tile_max = byte_v2_warp_reduce_max(score);
      const float old_max = running_max_shared[warp_id];
      const float new_max = max(old_max, tile_max);
      const float old_scale = expf(old_max - new_max);
      const float weight =
          lane_id < valid_rows ? expf(score - new_max) : 0.0f;
      const float tile_denom = byte_v2_warp_reduce_sum(weight);
      if (lane_id < kByteV2TileSize) {
        p_shared[warp_id * kByteV2TileSize + lane_id] =
            lane_id < valid_rows
                ? byte_v2_bf16_bits_to_wmma(
                      byte_v2_float_to_bf16_bits(weight))
                : zero;
      }
      if (lane_id == 0) {
        tile_acc_factor[warp_id] = old_scale;
        denom_shared[warp_id] = denom_shared[warp_id] * old_scale + tile_denom;
        running_max_shared[warp_id] = new_max;
      }
    }
    __syncthreads();
    if (early_exit_mode == 3) {
      continue;
    }

    if (tid < head_size_v) {
#pragma unroll
      for (int q = 0; q < q_per_kv; ++q) {
        acc[q] *= tile_acc_factor[q];
      }
    }
    __syncthreads();

#pragma unroll
    for (int dim_base = 0; dim_base < head_size_v;
         dim_base += kByteV2TileSize) {
      if (tid < warpSize) {
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16,
                               __nv_bfloat16, nvcuda::wmma::row_major>
            p_frag;
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16,
                               __nv_bfloat16, nvcuda::wmma::row_major>
            v_frag;
        nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float>
            pv_frag;
        nvcuda::wmma::fill_fragment(pv_frag, 0.0f);
        nvcuda::wmma::load_matrix_sync(p_frag, p_shared, kByteV2TileSize);
        nvcuda::wmma::load_matrix_sync(v_frag, v_shared + dim_base,
                                       head_size_v);
        nvcuda::wmma::mma_sync(pv_frag, p_frag, v_frag, pv_frag);
        nvcuda::wmma::store_matrix_sync(pv_shared, pv_frag, kByteV2TileSize,
                                        nvcuda::wmma::mem_row_major);
      }
      __syncthreads();

      if (tid >= dim_base && tid < dim_base + kByteV2TileSize &&
          tid < head_size_v) {
        const int dim_in_tile = tid - dim_base;
#pragma unroll
        for (int q = 0; q < q_per_kv; ++q) {
          acc[q] += pv_shared[q * kByteV2TileSize + dim_in_tile];
        }
      }
      __syncthreads();
    }
  }

  if (fine_profile_mode) {
#pragma unroll
    for (int q = 0; q < q_per_kv; ++q) {
      const int q_head = first_q_head + q;
      if (q_head >= num_heads) {
        continue;
      }
      float* partial =
          partial_output +
          (((req_idx * num_heads + q_head) * num_kv_splits + split_idx) *
           partial_stride);
      if (tid < head_size_v) {
        partial[tid] = static_cast<float>(
            profile_checksum ^ static_cast<uint32_t>(q * 0x45d9f3bU));
      }
      if (tid == 0) {
        partial[head_size_v] = static_cast<float>(profile_checksum);
      }
    }
    return;
  }

  if (tid == 0) {
#pragma unroll
    for (int q = 0; q < q_per_kv; ++q) {
      inv_denom[q] =
          denom_shared[q] > 0.0f ? (1.0f / denom_shared[q]) : 0.0f;
      lse_shared[q] = denom_shared[q] > 0.0f
                          ? running_max_shared[q] + logf(denom_shared[q])
                          : -FLT_MAX;
    }
  }
  __syncthreads();

#pragma unroll
  for (int q = 0; q < q_per_kv; ++q) {
    const int q_head = first_q_head + q;
    if (q_head >= num_heads) {
      continue;
    }
    float* partial =
        partial_output +
        (((req_idx * num_heads + q_head) * num_kv_splits + split_idx) *
         partial_stride);
    if (tid < head_size_v) {
      partial[tid] = acc[q] * inv_denom[q];
    }
    if (tid == 0) {
      partial[head_size_v] = lse_shared[q];
    }
  }
#endif
}

__global__ void byte_v2_paged_decode_attention_split_reduce_kernel(
    const float* __restrict__ partial_output, uint16_t* __restrict__ output,
    const int num_decode_tokens, const int num_heads, const int head_size_v,
    const int num_kv_splits) {
  const int req_idx = blockIdx.x;
  const int q_head = blockIdx.y;
  const int tid = threadIdx.x;
  if (req_idx >= num_decode_tokens || q_head >= num_heads) {
    return;
  }

  __shared__ float global_max;
  __shared__ float global_denom;
  const int partial_stride = head_size_v + 1;

  if (tid == 0) {
    float max_lse = -FLT_MAX;
    for (int split = 0; split < num_kv_splits; ++split) {
      const float* partial =
          partial_output +
          (((req_idx * num_heads + q_head) * num_kv_splits + split) *
           partial_stride);
      max_lse = max(max_lse, partial[head_size_v]);
    }
    float denom = 0.0f;
    if (max_lse > -FLT_MAX) {
      for (int split = 0; split < num_kv_splits; ++split) {
        const float* partial =
            partial_output +
            (((req_idx * num_heads + q_head) * num_kv_splits + split) *
             partial_stride);
        const float lse = partial[head_size_v];
        if (lse > -FLT_MAX) {
          denom += expf(lse - max_lse);
        }
      }
    }
    global_max = max_lse;
    global_denom = denom;
  }
  __syncthreads();

  for (int d = tid; d < head_size_v; d += blockDim.x) {
    float acc = 0.0f;
    if (global_denom > 0.0f) {
      for (int split = 0; split < num_kv_splits; ++split) {
        const float* partial =
            partial_output +
            (((req_idx * num_heads + q_head) * num_kv_splits + split) *
             partial_stride);
        const float lse = partial[head_size_v];
        if (lse > -FLT_MAX) {
          acc += expf(lse - global_max) * partial[d];
        }
      }
      acc /= global_denom;
    }
    output[(req_idx * num_heads + q_head) * head_size_v + d] =
        byte_v2_float_to_bf16_bits(acc);
  }
}

__global__ void byte_v2_paged_decode_attention_split_reduce_parallel_kernel(
    const float* __restrict__ partial_output, uint16_t* __restrict__ output,
    const int num_decode_tokens, const int num_heads, const int head_size_v,
    const int num_kv_splits) {
  const int req_idx = blockIdx.x;
  const int q_head = blockIdx.y;
  const int tid = threadIdx.x;
  if (req_idx >= num_decode_tokens || q_head >= num_heads) {
    return;
  }

  __shared__ float reduce_storage[kByteV2MaxHeadSize];
  __shared__ float global_max;
  __shared__ float global_denom;
  const int partial_stride = head_size_v + 1;
  const int64_t partial_base =
      (static_cast<int64_t>(req_idx) * num_heads + q_head) * num_kv_splits *
      partial_stride;

  float local_max = -FLT_MAX;
  for (int split = tid; split < num_kv_splits; split += blockDim.x) {
    const float* partial =
        partial_output + partial_base + split * partial_stride;
    local_max = max(local_max, partial[head_size_v]);
  }
  reduce_storage[tid] = local_max;
  __syncthreads();

  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      reduce_storage[tid] =
          max(reduce_storage[tid], reduce_storage[tid + offset]);
    }
    __syncthreads();
  }
  if (tid == 0) {
    global_max = reduce_storage[0];
  }
  __syncthreads();

  float local_denom = 0.0f;
  if (global_max > -FLT_MAX) {
    for (int split = tid; split < num_kv_splits; split += blockDim.x) {
      const float* partial =
          partial_output + partial_base + split * partial_stride;
      const float lse = partial[head_size_v];
      if (lse > -FLT_MAX) {
        local_denom += expf(lse - global_max);
      }
    }
  }
  reduce_storage[tid] = local_denom;
  __syncthreads();

  for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
    if (tid < offset) {
      reduce_storage[tid] += reduce_storage[tid + offset];
    }
    __syncthreads();
  }
  if (tid == 0) {
    global_denom = reduce_storage[0];
  }
  __syncthreads();

  for (int d = tid; d < head_size_v; d += blockDim.x) {
    float acc = 0.0f;
    if (global_denom > 0.0f) {
      for (int split = 0; split < num_kv_splits; ++split) {
        const float* partial =
            partial_output + partial_base + split * partial_stride;
        const float lse = partial[head_size_v];
        if (lse > -FLT_MAX) {
          acc += expf(lse - global_max) * partial[d];
        }
      }
      acc /= global_denom;
    }
    output[(req_idx * num_heads + q_head) * head_size_v + d] =
        byte_v2_float_to_bf16_bits(acc);
  }
}

__global__ void byte_v2_wmma_layout_microbench_kernel(
    const uint16_t* __restrict__ query, const uint16_t* __restrict__ key,
    const uint16_t* __restrict__ value, uint16_t* __restrict__ output,
    const int num_tiles, const int repeat_count) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  constexpr int tile_size = kByteV2TileSize;
  constexpr int head_size = kByteV2MaxHeadSize;
  const int tile_idx = blockIdx.x;
  const int tid = threadIdx.x;
  if (tile_idx >= num_tiles) {
    return;
  }

  __shared__ __align__(16) __nv_bfloat16 q_shared[tile_size * head_size];
  __shared__ __align__(16) __nv_bfloat16 k_shared[head_size * tile_size];
  __shared__ __align__(16) __nv_bfloat16 v_shared[tile_size * head_size];
  __shared__ __align__(16) __nv_bfloat16 p_shared[tile_size * tile_size];
  __shared__ float scores[tile_size * tile_size];
  __shared__ float pv_shared[tile_size * tile_size];

  const int64_t tile_offset =
      static_cast<int64_t>(tile_idx) * tile_size * head_size;
  for (int idx = tid; idx < tile_size * head_size; idx += blockDim.x) {
    const int row = idx / head_size;
    const int dim = idx % head_size;
    q_shared[idx] = byte_v2_bf16_bits_to_wmma(query[tile_offset + idx]);
    k_shared[dim * tile_size + row] =
        byte_v2_bf16_bits_to_wmma(key[tile_offset + idx]);
    v_shared[idx] = byte_v2_bf16_bits_to_wmma(value[tile_offset + idx]);
  }
  __syncthreads();

  for (int repeat = 0; repeat < repeat_count; ++repeat) {
    if (tid < warpSize) {
      nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16,
                             __nv_bfloat16, nvcuda::wmma::row_major>
          a_frag;
      nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16,
                             __nv_bfloat16, nvcuda::wmma::row_major>
          b_frag;
      nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float>
          c_frag;
      nvcuda::wmma::fill_fragment(c_frag, 0.0f);
      for (int dim_base = 0; dim_base < head_size; dim_base += tile_size) {
        nvcuda::wmma::load_matrix_sync(a_frag, q_shared + dim_base,
                                       head_size);
        nvcuda::wmma::load_matrix_sync(b_frag,
                                       k_shared + dim_base * tile_size,
                                       tile_size);
        nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
      }
      nvcuda::wmma::store_matrix_sync(scores, c_frag, tile_size,
                                      nvcuda::wmma::mem_row_major);
    }
    __syncthreads();

    for (int idx = tid; idx < tile_size * tile_size; idx += blockDim.x) {
      p_shared[idx] =
          byte_v2_bf16_bits_to_wmma(byte_v2_float_to_bf16_bits(scores[idx]));
    }
    __syncthreads();

    for (int dim_base = 0; dim_base < head_size; dim_base += tile_size) {
      if (tid < warpSize) {
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16,
                               __nv_bfloat16, nvcuda::wmma::row_major>
            p_frag;
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16,
                               __nv_bfloat16, nvcuda::wmma::row_major>
            v_frag;
        nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float>
            pv_frag;
        nvcuda::wmma::fill_fragment(pv_frag, 0.0f);
        nvcuda::wmma::load_matrix_sync(p_frag, p_shared, tile_size);
        nvcuda::wmma::load_matrix_sync(v_frag, v_shared + dim_base, head_size);
        nvcuda::wmma::mma_sync(pv_frag, p_frag, v_frag, pv_frag);
        nvcuda::wmma::store_matrix_sync(pv_shared, pv_frag, tile_size,
                                        nvcuda::wmma::mem_row_major);
      }
      __syncthreads();

      for (int idx = tid; idx < tile_size * tile_size; idx += blockDim.x) {
        const int row = idx / tile_size;
        const int dim = idx % tile_size;
        output[tile_offset + row * head_size + dim_base + dim] =
            byte_v2_float_to_bf16_bits(pv_shared[idx]);
      }
      __syncthreads();
    }
  }
#endif
}

__global__ void byte_v2_decode_page_wmma_microbench_kernel(
    const uint16_t* __restrict__ query, const uint8_t* __restrict__ kv_cache,
    uint16_t* __restrict__ output, const int num_pages,
    const int num_kv_heads, const int kv_head, const int page_size_bytes,
    const int repeat_count, const bool use_v3_warp_stripe_load,
    const bool use_v3_cp_async_stage) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  constexpr int tile_size = kByteV2TileSize;
  constexpr int head_size = kByteV2MaxHeadSize;
  constexpr int head_size_v = kByteV2MaxHeadSize;
  const int page_idx = blockIdx.x;
  const int tid = threadIdx.x;
  if (page_idx >= num_pages) {
    return;
  }

  __shared__ __align__(16) __nv_bfloat16 q_shared[tile_size * head_size];
  __shared__ __align__(16) __nv_bfloat16 k_shared[head_size * tile_size];
  __shared__ __align__(16) __nv_bfloat16 v_shared[tile_size * head_size_v];
  __shared__ __align__(16) __nv_bfloat16 p_shared[tile_size * tile_size];
  __shared__ float scores[tile_size * tile_size];
  __shared__ float pv_shared[tile_size * tile_size];
  __shared__ __align__(16) uint8_t v3_cp_stage[2 *
                                               kByteV2TilePayloadBytesV3];

  const int64_t tile_offset =
      static_cast<int64_t>(page_idx) * tile_size * head_size;
  for (int idx = tid; idx < tile_size * head_size; idx += blockDim.x) {
    q_shared[idx] = byte_v2_bf16_bits_to_wmma(query[tile_offset + idx]);
  }
  __syncthreads();

  const uint8_t* page =
      kv_cache + static_cast<int64_t>(page_idx) * page_size_bytes;
  const uint8_t page_status = page[kByteV2PageStatusOffset];
  const int page_valid_rows =
      static_cast<int>(page[kByteV2PageValidRowsOffset]);
  if (page_status != kByteV2PageStatusCompressed || page_valid_rows <= 0) {
    for (int idx = tid; idx < tile_size * head_size_v; idx += blockDim.x) {
      output[tile_offset + idx] = byte_v2_float_to_bf16_bits(0.0f);
    }
    return;
  }

  const int valid_rows = min(page_valid_rows, tile_size);
  constexpr int k_dim_tiles = head_size / tile_size;
  constexpr int v_dim_tiles = head_size_v / tile_size;
  const bool use_v3_payload =
      page[kByteV2PageLayoutVersionOffset] == kByteV2PayloadLayoutVersionV3;
  for (int repeat = 0; repeat < repeat_count; ++repeat) {
    if (use_v3_payload && use_v3_cp_async_stage) {
      byte_v2_decode_k_tiles_to_shared_v3_cp_async(
          page, v3_cp_stage, k_shared, tid, kv_head, num_kv_heads,
          k_dim_tiles, v_dim_tiles, valid_rows);
      byte_v2_decode_v_tiles_to_shared_v3_cp_async(
          page, v3_cp_stage, v_shared, tid, kv_head, num_kv_heads,
          k_dim_tiles, v_dim_tiles, valid_rows, head_size_v);
    } else {
      for (int dim_tile = 0; dim_tile < k_dim_tiles; ++dim_tile) {
        if (use_v3_payload) {
        byte_v2_decode_k_transposed_tile_to_shared_no_fallback_v3(
            page, k_shared, tid, kv_head, dim_tile, num_kv_heads,
            k_dim_tiles, v_dim_tiles, valid_rows, use_v3_warp_stripe_load);
        } else {
          byte_v2_decode_k_transposed_tile_to_shared_no_fallback(
              page, k_shared, tid, kv_head, dim_tile, k_dim_tiles, v_dim_tiles,
              valid_rows);
        }
      }
      for (int dim_tile = 0; dim_tile < v_dim_tiles; ++dim_tile) {
        if (use_v3_payload) {
          byte_v2_decode_v_rowmajor_tile_to_shared_no_fallback_v3(
              page, v_shared, tid, kv_head, dim_tile, num_kv_heads,
              k_dim_tiles, v_dim_tiles, valid_rows, head_size_v,
              use_v3_warp_stripe_load);
        } else {
          byte_v2_decode_v_rowmajor_tile_to_shared_no_fallback(
              page, v_shared, tid, kv_head, dim_tile, k_dim_tiles, v_dim_tiles,
              valid_rows, head_size_v);
        }
      }
    }
    __syncthreads();

    if (tid < warpSize) {
      nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16,
                             __nv_bfloat16, nvcuda::wmma::row_major>
          a_frag;
      nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16,
                             __nv_bfloat16, nvcuda::wmma::row_major>
          b_frag;
      nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float>
          c_frag;
      nvcuda::wmma::fill_fragment(c_frag, 0.0f);
      for (int dim_base = 0; dim_base < head_size; dim_base += tile_size) {
        nvcuda::wmma::load_matrix_sync(a_frag, q_shared + dim_base,
                                       head_size);
        nvcuda::wmma::load_matrix_sync(b_frag,
                                       k_shared + dim_base * tile_size,
                                       tile_size);
        nvcuda::wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
      }
      nvcuda::wmma::store_matrix_sync(scores, c_frag, tile_size,
                                      nvcuda::wmma::mem_row_major);
    }
    __syncthreads();

    for (int idx = tid; idx < tile_size * tile_size; idx += blockDim.x) {
      p_shared[idx] =
          byte_v2_bf16_bits_to_wmma(byte_v2_float_to_bf16_bits(scores[idx]));
    }
    __syncthreads();

    for (int dim_base = 0; dim_base < head_size_v; dim_base += tile_size) {
      if (tid < warpSize) {
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_a, 16, 16, 16,
                               __nv_bfloat16, nvcuda::wmma::row_major>
            p_frag;
        nvcuda::wmma::fragment<nvcuda::wmma::matrix_b, 16, 16, 16,
                               __nv_bfloat16, nvcuda::wmma::row_major>
            v_frag;
        nvcuda::wmma::fragment<nvcuda::wmma::accumulator, 16, 16, 16, float>
            pv_frag;
        nvcuda::wmma::fill_fragment(pv_frag, 0.0f);
        nvcuda::wmma::load_matrix_sync(p_frag, p_shared, tile_size);
        nvcuda::wmma::load_matrix_sync(v_frag, v_shared + dim_base,
                                       head_size_v);
        nvcuda::wmma::mma_sync(pv_frag, p_frag, v_frag, pv_frag);
        nvcuda::wmma::store_matrix_sync(pv_shared, pv_frag, tile_size,
                                        nvcuda::wmma::mem_row_major);
      }
      __syncthreads();

      for (int idx = tid; idx < tile_size * tile_size; idx += blockDim.x) {
        const int row = idx / tile_size;
        const int dim = idx % tile_size;
        output[tile_offset + row * head_size_v + dim_base + dim] =
            byte_v2_float_to_bf16_bits(pv_shared[idx]);
      }
      __syncthreads();
    }
  }
#else
  (void)num_kv_heads;
#endif
}

#else
template <typename block_table_t, typename seq_lens_t>
__global__ void byte_v2_paged_decode_attention_gqa_wmma_kernel(
    const uint16_t* __restrict__, const uint8_t* __restrict__,
    const block_table_t* __restrict__, const seq_lens_t* __restrict__,
    uint16_t* __restrict__, const uint8_t* __restrict__,
    const int32_t* __restrict__, const int32_t* __restrict__,
    const int32_t* __restrict__, const int32_t* __restrict__,
    const int32_t* __restrict__, const int32_t* __restrict__, const float,
    const int, const int, const int, const int, const int, const int, const int,
    const int64_t, const int64_t, const int64_t) {}

template <typename block_table_t, typename seq_lens_t>
__global__ void byte_v2_paged_decode_attention_gqa_wmma_split_stage1_kernel(
    const uint16_t* __restrict__, const uint8_t* __restrict__,
    const block_table_t* __restrict__, const seq_lens_t* __restrict__,
    float* __restrict__, const uint8_t* __restrict__,
    const int32_t* __restrict__, const int32_t* __restrict__,
    const int32_t* __restrict__, const int32_t* __restrict__,
    const int32_t* __restrict__, const int32_t* __restrict__, const float,
    const int, const int, const int, const int, const int, const int, const int,
    const int, const int64_t, const int64_t, const int64_t, const bool,
    const bool) {}

	template <typename block_table_t, typename seq_lens_t,
	          bool metadata_fastpath = false, bool no_outlier_metadata = false,
	          bool no_fallback_metadata = false>
__global__ void byte_v2_paged_decode_attention_gqa_cute_split_stage1_kernel(
    const uint16_t* __restrict__, const uint8_t* __restrict__,
    const block_table_t* __restrict__, const seq_lens_t* __restrict__,
    float* __restrict__, const uint8_t* __restrict__,
    const int32_t* __restrict__, const int32_t* __restrict__,
    const int32_t* __restrict__, const int32_t* __restrict__,
    const int32_t* __restrict__, const int32_t* __restrict__, const float,
    const int, const int, const int, const int, const int, const int,
    const int64_t, const int64_t, const int64_t, const bool, const bool,
    const bool, const bool, const int) {}

template <typename block_table_t, typename seq_lens_t>
__global__ void
byte_v2_paged_decode_attention_gqa4_h128_fast_split_stage1_kernel(
    const uint16_t* __restrict__, const uint8_t* __restrict__,
    const block_table_t* __restrict__, const seq_lens_t* __restrict__,
    float* __restrict__, const float, const int, const int, const int,
    const int64_t, const int64_t, const int64_t) {}

template <typename block_table_t, typename seq_lens_t, int MacroPages,
          bool Specialized, bool OutlierOverlay>
__global__ void
byte_v2_paged_decode_attention_gqa4_h128_v4_split_stage1_kernel(
    const uint16_t* __restrict__, const uint8_t* __restrict__,
    const block_table_t* __restrict__, const seq_lens_t* __restrict__,
    float* __restrict__, const uint8_t* __restrict__,
    const int32_t* __restrict__, const int32_t* __restrict__,
    const int32_t* __restrict__, const int32_t* __restrict__, const float,
    const int, const int, const int, const int, const int64_t,
    const int64_t, const int64_t) {}

template <typename block_table_t, typename seq_lens_t, bool OutlierOverlay>
__global__ void
byte_v2_paged_decode_attention_gqa4_h128_v4_block128_split_stage1_kernel(
    const uint16_t* __restrict__, const uint8_t* __restrict__,
    const block_table_t* __restrict__, const seq_lens_t* __restrict__,
    float* __restrict__, const int32_t* __restrict__,
    const int32_t* __restrict__, const int32_t* __restrict__, const float,
    const int, const int, const int, const int64_t, const int64_t,
    const int64_t) {}

__global__ void byte_v2_paged_decode_attention_split_reduce_kernel(
    const float* __restrict__, uint16_t* __restrict__, const int, const int,
    const int, const int) {}

__global__ void byte_v2_paged_decode_attention_split_reduce_parallel_kernel(
    const float* __restrict__, uint16_t* __restrict__, const int, const int,
    const int, const int) {}

__global__ void byte_v2_wmma_layout_microbench_kernel(
    const uint16_t* __restrict__, const uint16_t* __restrict__,
    const uint16_t* __restrict__, uint16_t* __restrict__, const int,
    const int) {}

__global__ void byte_v2_decode_page_wmma_microbench_kernel(
    const uint16_t* __restrict__, const uint8_t* __restrict__,
    uint16_t* __restrict__, const int, const int, const int, const int,
    const int, const bool, const bool) {}

#endif

}  // namespace
}  // namespace vllm

torch::stable::Tensor byte_v2_reshape_and_cache(
    torch::stable::Tensor& key,    // [num_tokens, num_kv_heads, head_size]
    torch::stable::Tensor& value,  // [num_tokens, num_kv_heads, head_size_v]
    torch::stable::Tensor& kv_cache,      // [num_blocks, page_size_bytes]
    torch::stable::Tensor& slot_mapping,  // [num_tokens]
    int64_t block_size, int64_t num_kv_heads, int64_t head_size,
    int64_t head_size_v, int64_t page_size_bytes,
    std::optional<torch::stable::Tensor> fallback_pool,
    std::optional<torch::stable::Tensor> fallback_block_ids,
    std::optional<torch::stable::Tensor> fallback_next_slot,
    std::optional<torch::stable::Tensor> fallback_tile_ids,
    std::optional<torch::stable::Tensor> fallback_tile_next_slot,
    std::optional<torch::stable::Tensor> deferred_error,
    std::optional<torch::stable::Tensor> outlier_arena,
    std::optional<torch::stable::Tensor> outlier_block_flags,
    std::optional<torch::stable::Tensor> outlier_tile_bitmap,
    std::optional<torch::stable::Tensor> outlier_tile_meta,
    std::optional<torch::stable::Tensor> outlier_next_entry,
    bool decode_append_fast_path_safe) {
  STD_TORCH_CHECK(key.device().is_cuda() && value.device().is_cuda() &&
                      kv_cache.device().is_cuda() &&
                      slot_mapping.device().is_cuda(),
                  "Byte-v2 cache update expects CUDA tensors");
  STD_TORCH_CHECK(key.device().index() == kv_cache.device().index() &&
                      value.device().index() == kv_cache.device().index() &&
                      slot_mapping.device().index() == kv_cache.device().index(),
                  "Byte-v2 cache update tensors must be on the same GPU");
  STD_TORCH_CHECK(key.scalar_type() ==
                      torch::headeronly::ScalarType::BFloat16 &&
                      value.scalar_type() ==
                          torch::headeronly::ScalarType::BFloat16,
                  "Byte-v2 cache update expects BF16 key/value tensors");
  STD_TORCH_CHECK(kv_cache.scalar_type() ==
                      torch::headeronly::ScalarType::Byte,
                  "Byte-v2 kv_cache must use uint8 storage");
  STD_TORCH_CHECK(slot_mapping.scalar_type() ==
                      torch::headeronly::ScalarType::Long,
                  "Byte-v2 slot_mapping must be int64");
  STD_TORCH_CHECK(kv_cache.is_contiguous() && slot_mapping.is_contiguous(),
                  "Byte-v2 cache update expects contiguous kv_cache and "
                  "slot_mapping tensors");
  STD_TORCH_CHECK(block_size == vllm::kByteV2TileSize,
                  "Byte-v2 cache update requires block_size=16");
  STD_TORCH_CHECK(head_size > 0 && head_size <= vllm::kByteV2MaxHeadSize &&
                      head_size_v > 0 &&
                          head_size_v <= vllm::kByteV2MaxHeadSize,
                  "Byte-v2 cache update requires head sizes in [1, 128]");
  STD_TORCH_CHECK(head_size % vllm::kByteV2TileSize == 0 &&
                      head_size_v % vllm::kByteV2TileSize == 0,
                  "Byte-v2 cache update head sizes must be multiples of 16");
  STD_TORCH_CHECK(key.dim() == 3 && value.dim() == 3 && kv_cache.dim() == 2 &&
                      slot_mapping.dim() == 1,
                  "Byte-v2 cache update tensor ranks are invalid");
  STD_TORCH_CHECK(key.size(1) == num_kv_heads && value.size(1) == num_kv_heads &&
                      key.size(2) == head_size &&
                      value.size(2) == head_size_v,
                  "Byte-v2 cache update tensor shapes do not match metadata");

  const int64_t compressed_payload_bytes =
      num_kv_heads * (head_size / vllm::kByteV2TileSize +
                      head_size_v / vllm::kByteV2TileSize) *
      vllm::kByteV2FastTilePayloadBytes;
  const int64_t raw_block_bytes =
      vllm::kByteV2TileSize * num_kv_heads * (head_size + head_size_v) * 2;
  const int64_t compressed_page_size =
      vllm::kByteV2PageHeaderBytes + compressed_payload_bytes;
  const int64_t compressed_page_size_v3 = vllm::byte_v2_v3_page_size_bytes(
      static_cast<int>(num_kv_heads),
      static_cast<int>(head_size / vllm::kByteV2TileSize),
      static_cast<int>(head_size_v / vllm::kByteV2TileSize));
  const int64_t raw_overlay_page_size =
      vllm::kByteV2PageHeaderBytes +
      std::max(compressed_payload_bytes, raw_block_bytes);
  const int64_t total_tiles64 =
      num_kv_heads * (head_size / vllm::kByteV2TileSize +
                      head_size_v / vllm::kByteV2TileSize);
  const int64_t outlier_tile_bitmap_words64 = (total_tiles64 + 31) / 32;
  const bool use_v3_payload_layout = page_size_bytes == compressed_page_size_v3;
  bool v3_outlier_only_no_fallback = false;
  if (const char* outlier_only_env =
          std::getenv("VLLM_BYTE_V2_V3_OUTLIER_ONLY_NO_FALLBACK")) {
    v3_outlier_only_no_fallback = std::atoi(outlier_only_env) != 0;
  }
  v3_outlier_only_no_fallback =
      v3_outlier_only_no_fallback && use_v3_payload_layout;
  const bool compressed_only_pages =
      page_size_bytes == compressed_page_size || use_v3_payload_layout;
  STD_TORCH_CHECK((compressed_only_pages ||
                   page_size_bytes == raw_overlay_page_size) &&
                      kv_cache.size(1) == page_size_bytes,
                  "Byte-v2 cache update page size mismatch");
  const bool has_sparse_fallback = fallback_pool.has_value() &&
                                   fallback_block_ids.has_value() &&
                                   fallback_next_slot.has_value();
  STD_TORCH_CHECK(
      has_sparse_fallback ||
          (!fallback_pool.has_value() && !fallback_block_ids.has_value() &&
           !fallback_next_slot.has_value()),
      "Byte-v2 sparse fallback pool arguments must be provided together");
  const bool has_tile_fallback = fallback_pool.has_value() &&
                                 fallback_tile_ids.has_value() &&
                                 fallback_tile_next_slot.has_value();
  STD_TORCH_CHECK(
      has_tile_fallback ||
          (!fallback_tile_ids.has_value() &&
           !fallback_tile_next_slot.has_value()),
      "Byte-v2 tile fallback arguments must be provided together with "
      "fallback_pool");
  const bool has_outlier_arena =
      outlier_arena.has_value() || outlier_tile_meta.has_value() ||
      outlier_next_entry.has_value();
  STD_TORCH_CHECK(
      (outlier_arena.has_value() == outlier_tile_meta.has_value()) &&
          (outlier_arena.has_value() == outlier_next_entry.has_value()),
      "Byte-v2 outlier arena arguments must be provided together");
  STD_TORCH_CHECK(
      !has_outlier_arena || has_tile_fallback || v3_outlier_only_no_fallback,
      "Byte-v2 outlier arena requires tile fallback metadata unless "
      "V3 outlier-only no-fallback mode is enabled");
  const bool has_outlier_block_flags = outlier_block_flags.has_value();
  STD_TORCH_CHECK(!has_outlier_block_flags || has_outlier_arena,
                  "Byte-v2 outlier block flags require outlier arena metadata");
  const bool has_outlier_tile_bitmap = outlier_tile_bitmap.has_value();
  STD_TORCH_CHECK(!has_outlier_tile_bitmap || has_outlier_arena,
                  "Byte-v2 outlier tile bitmap requires outlier arena metadata");
  int64_t fallback_pool_blocks64 = 0;
  int64_t outlier_arena_entries64 = 0;
  if (has_sparse_fallback) {
    STD_TORCH_CHECK(fallback_pool->device().is_cuda() &&
                        fallback_block_ids->device().is_cuda() &&
                        fallback_next_slot->device().is_cuda(),
                    "Byte-v2 sparse fallback tensors must be CUDA tensors");
    STD_TORCH_CHECK(fallback_pool->device().index() == kv_cache.device().index() &&
                        fallback_block_ids->device().index() ==
                            kv_cache.device().index() &&
                        fallback_next_slot->device().index() ==
                            kv_cache.device().index(),
                    "Byte-v2 sparse fallback tensors must be on the same GPU");
    STD_TORCH_CHECK(fallback_pool->scalar_type() ==
                            torch::headeronly::ScalarType::Byte &&
                        fallback_block_ids->scalar_type() ==
                            torch::headeronly::ScalarType::Int &&
                        fallback_next_slot->scalar_type() ==
                            torch::headeronly::ScalarType::Int,
                    "Byte-v2 sparse fallback tensors have invalid dtypes");
    STD_TORCH_CHECK(fallback_pool->is_contiguous() &&
                        fallback_block_ids->is_contiguous() &&
                        fallback_next_slot->is_contiguous(),
                    "Byte-v2 sparse fallback tensors must be contiguous");
    STD_TORCH_CHECK(fallback_pool->dim() == 2 &&
                        fallback_pool->size(1) == raw_block_bytes,
                    "Byte-v2 fallback_pool must have shape "
                    "[pool_blocks, raw_block_bytes]");
    STD_TORCH_CHECK(fallback_block_ids->dim() == 1 &&
                        fallback_block_ids->size(0) == kv_cache.size(0),
                    "Byte-v2 fallback_block_ids must have one entry per KV "
                    "block");
    STD_TORCH_CHECK(fallback_next_slot->numel() == 1,
                    "Byte-v2 fallback_next_slot must be a scalar tensor");
    fallback_pool_blocks64 = fallback_pool->size(0);
    STD_TORCH_CHECK(fallback_pool_blocks64 <= INT_MAX,
                    "Byte-v2 fallback pool supports at most INT_MAX slots");
  }
  if (has_tile_fallback) {
    STD_TORCH_CHECK(fallback_tile_ids->device().is_cuda() &&
                        fallback_tile_next_slot->device().is_cuda(),
                    "Byte-v2 tile fallback tensors must be CUDA tensors");
    STD_TORCH_CHECK(fallback_tile_ids->device().index() ==
                            kv_cache.device().index() &&
                        fallback_tile_next_slot->device().index() ==
                            kv_cache.device().index(),
                    "Byte-v2 tile fallback tensors must be on the same GPU");
    STD_TORCH_CHECK(fallback_tile_ids->scalar_type() ==
                            torch::headeronly::ScalarType::Int &&
                        fallback_tile_next_slot->scalar_type() ==
                            torch::headeronly::ScalarType::Int,
                    "Byte-v2 tile fallback tensors have invalid dtypes");
    STD_TORCH_CHECK(fallback_tile_ids->is_contiguous() &&
                        fallback_tile_next_slot->is_contiguous(),
                    "Byte-v2 tile fallback tensors must be contiguous");
    STD_TORCH_CHECK(fallback_tile_ids->dim() == 2 &&
                        fallback_tile_ids->size(0) == kv_cache.size(0) &&
                        fallback_tile_ids->size(1) == total_tiles64,
                    "Byte-v2 fallback_tile_ids must have shape "
                    "[num_blocks, total_tiles_per_block]");
    STD_TORCH_CHECK(fallback_tile_next_slot->numel() == 1,
                    "Byte-v2 fallback_tile_next_slot must be a scalar tensor");
  }
  if (has_outlier_arena) {
    STD_TORCH_CHECK(outlier_arena->device().is_cuda() &&
                        outlier_tile_meta->device().is_cuda() &&
                        outlier_next_entry->device().is_cuda(),
                    "Byte-v2 outlier arena tensors must be CUDA tensors");
    STD_TORCH_CHECK(outlier_arena->device().index() ==
                            kv_cache.device().index() &&
                        outlier_tile_meta->device().index() ==
                            kv_cache.device().index() &&
                        outlier_next_entry->device().index() ==
                            kv_cache.device().index(),
                    "Byte-v2 outlier arena tensors must be on the same GPU");
    STD_TORCH_CHECK(outlier_arena->scalar_type() ==
                            torch::headeronly::ScalarType::Int &&
                        outlier_tile_meta->scalar_type() ==
                            torch::headeronly::ScalarType::Int &&
                        outlier_next_entry->scalar_type() ==
                            torch::headeronly::ScalarType::Int,
                    "Byte-v2 outlier arena tensors must use int32 dtype");
    STD_TORCH_CHECK(outlier_arena->is_contiguous() &&
                        outlier_tile_meta->is_contiguous() &&
                        outlier_next_entry->is_contiguous(),
                    "Byte-v2 outlier arena tensors must be contiguous");
    STD_TORCH_CHECK(outlier_arena->dim() == 1,
                    "Byte-v2 outlier_arena must be a 1-D tensor");
    STD_TORCH_CHECK(outlier_tile_meta->dim() == 2 &&
                        outlier_tile_meta->size(0) == kv_cache.size(0) &&
                        outlier_tile_meta->size(1) == total_tiles64,
                    "Byte-v2 outlier_tile_meta must have shape "
                    "[num_blocks, total_tiles_per_block]");
    STD_TORCH_CHECK(outlier_next_entry->numel() == 1,
                    "Byte-v2 outlier_next_entry must be a scalar tensor");
    outlier_arena_entries64 = outlier_arena->numel();
    STD_TORCH_CHECK(outlier_arena_entries64 <=
                        vllm::kByteV2MaxOutlierArenaOffset,
                    "Byte-v2 outlier_arena supports at most 2^23-1 entries");
  }
  if (has_outlier_block_flags) {
    STD_TORCH_CHECK(outlier_block_flags->device().is_cuda(),
                    "Byte-v2 outlier_block_flags must be a CUDA tensor");
    STD_TORCH_CHECK(outlier_block_flags->device().index() ==
                        kv_cache.device().index(),
                    "Byte-v2 outlier_block_flags must be on the same GPU");
    STD_TORCH_CHECK(outlier_block_flags->scalar_type() ==
                        torch::headeronly::ScalarType::Int,
                    "Byte-v2 outlier_block_flags must use int32 dtype");
    STD_TORCH_CHECK(outlier_block_flags->is_contiguous(),
                    "Byte-v2 outlier_block_flags must be contiguous");
    STD_TORCH_CHECK(outlier_block_flags->dim() == 1 &&
                        outlier_block_flags->size(0) == kv_cache.size(0),
                    "Byte-v2 outlier_block_flags must have one entry per KV "
                    "block");
  }
  if (has_outlier_tile_bitmap) {
    STD_TORCH_CHECK(outlier_tile_bitmap->device().is_cuda(),
                    "Byte-v2 outlier_tile_bitmap must be a CUDA tensor");
    STD_TORCH_CHECK(outlier_tile_bitmap->device().index() ==
                        kv_cache.device().index(),
                    "Byte-v2 outlier_tile_bitmap must be on the same GPU");
    STD_TORCH_CHECK(outlier_tile_bitmap->scalar_type() ==
                        torch::headeronly::ScalarType::Int,
                    "Byte-v2 outlier_tile_bitmap must use int32 dtype");
    STD_TORCH_CHECK(outlier_tile_bitmap->is_contiguous(),
                    "Byte-v2 outlier_tile_bitmap must be contiguous");
    STD_TORCH_CHECK(outlier_tile_bitmap->dim() == 2 &&
                        outlier_tile_bitmap->size(0) == kv_cache.size(0) &&
                        outlier_tile_bitmap->size(1) ==
                            outlier_tile_bitmap_words64,
                    "Byte-v2 outlier_tile_bitmap must have shape "
                    "[num_blocks, ceil(total_tiles_per_block / 32)]");
  }
  const bool has_deferred_error = deferred_error.has_value();
  if (has_deferred_error) {
    STD_TORCH_CHECK(deferred_error->device().is_cuda(),
                    "Byte-v2 deferred_error must be a CUDA tensor");
    STD_TORCH_CHECK(
        deferred_error->device().index() == kv_cache.device().index(),
        "Byte-v2 deferred_error must be on the same GPU");
    STD_TORCH_CHECK(deferred_error->scalar_type() ==
                        torch::headeronly::ScalarType::Int,
                    "Byte-v2 deferred_error must use int32 dtype");
    STD_TORCH_CHECK(deferred_error->is_contiguous(),
                    "Byte-v2 deferred_error must be contiguous");
    STD_TORCH_CHECK(deferred_error->numel() >= 4,
                    "Byte-v2 deferred_error must have at least 4 elements");
  }

  const int64_t num_tokens64 = slot_mapping.size(0);
  STD_TORCH_CHECK(key.size(0) >= num_tokens64 && value.size(0) >= num_tokens64,
                  "Byte-v2 key/value must contain one row per slot");
  const int64_t num_blocks64 = kv_cache.size(0);
  STD_TORCH_CHECK(num_tokens64 <= INT_MAX && num_blocks64 <= INT_MAX,
                  "Byte-v2 native cache update supports at most INT_MAX tokens "
                  "and blocks");
  const int num_tokens = static_cast<int>(num_tokens64);
  const int num_blocks = static_cast<int>(num_blocks64);
  const int fallback_pool_blocks = static_cast<int>(fallback_pool_blocks64);
  const int outlier_arena_entries = static_cast<int>(outlier_arena_entries64);
  uint8_t* fallback_pool_ptr =
      has_sparse_fallback
          ? reinterpret_cast<uint8_t*>(fallback_pool->mutable_data_ptr())
          : nullptr;
  int32_t* fallback_block_ids_ptr =
      has_sparse_fallback
          ? reinterpret_cast<int32_t*>(fallback_block_ids->mutable_data_ptr())
          : nullptr;
  int32_t* fallback_next_slot_ptr =
      has_sparse_fallback
          ? reinterpret_cast<int32_t*>(fallback_next_slot->mutable_data_ptr())
          : nullptr;
  int32_t* fallback_tile_ids_ptr =
      has_tile_fallback
          ? reinterpret_cast<int32_t*>(fallback_tile_ids->mutable_data_ptr())
          : nullptr;
  int32_t* fallback_tile_next_slot_ptr =
      has_tile_fallback ? reinterpret_cast<int32_t*>(
                              fallback_tile_next_slot->mutable_data_ptr())
                        : nullptr;
  int32_t* deferred_error_ptr =
      has_deferred_error
          ? reinterpret_cast<int32_t*>(deferred_error->mutable_data_ptr())
          : nullptr;
  int32_t* outlier_arena_ptr =
      has_outlier_arena
          ? reinterpret_cast<int32_t*>(outlier_arena->mutable_data_ptr())
          : nullptr;
  int32_t* outlier_block_flags_ptr =
      has_outlier_block_flags ? reinterpret_cast<int32_t*>(
                                    outlier_block_flags->mutable_data_ptr())
                              : nullptr;
  int32_t* outlier_tile_bitmap_ptr =
      has_outlier_tile_bitmap ? reinterpret_cast<int32_t*>(
                                    outlier_tile_bitmap->mutable_data_ptr())
                              : nullptr;
  int32_t* outlier_tile_meta_ptr =
      has_outlier_arena
          ? reinterpret_cast<int32_t*>(outlier_tile_meta->mutable_data_ptr())
          : nullptr;
  int32_t* outlier_next_entry_ptr =
      has_outlier_arena
          ? reinterpret_cast<int32_t*>(outlier_next_entry->mutable_data_ptr())
          : nullptr;
  const torch::stable::accelerator::DeviceGuard device_guard(
      key.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream(key.get_device_index());
  int lossy_max_misses_per_tile = 0;
  if (const char* lossy_env =
          std::getenv("VLLM_BYTE_V2_LOSSY_MAX_MISSES_PER_TILE")) {
    lossy_max_misses_per_tile = std::max(0, std::atoi(lossy_env));
  }
  int outlier_max_per_tile = 0;
  if (has_outlier_arena) {
    if (const char* outlier_env =
            std::getenv("VLLM_BYTE_V2_OUTLIER_MAX_PER_TILE")) {
      outlier_max_per_tile = std::max(0, std::atoi(outlier_env));
    }
    outlier_max_per_tile =
        std::min(outlier_max_per_tile, vllm::kByteV2OutlierMetaCountMask);
  }
  STD_TORCH_CHECK(
      !v3_outlier_only_no_fallback ||
          (has_outlier_arena && outlier_max_per_tile > 0),
      "V3 outlier-only no-fallback mode requires outlier arena and "
      "VLLM_BYTE_V2_OUTLIER_MAX_PER_TILE > 0");
  if (num_tokens == 0) {
    return torch::stable::empty({0}, torch::headeronly::ScalarType::Long,
                                std::nullopt, slot_mapping.device());
  }

  cudaStreamCaptureStatus prefill_capture_status =
      cudaStreamCaptureStatusNone;
  BYTE_V2_CUDA_CHECK(cudaStreamIsCapturing(stream, &prefill_capture_status));
  if (prefill_capture_status == cudaStreamCaptureStatusNone &&
      compressed_only_pages &&
      (has_sparse_fallback || v3_outlier_only_no_fallback) &&
      num_tokens >= vllm::kByteV2TileSize &&
      (num_tokens % vllm::kByteV2TileSize) == 0 &&
      total_tiles64 <= vllm::kByteV2MaxTilesPerBlock) {
    const int num_groups = num_tokens / vllm::kByteV2TileSize;
    auto direct_ineligible =
        torch::stable::empty({1}, torch::headeronly::ScalarType::Int,
                             std::nullopt, kv_cache.device());
    auto group_block_ids =
        torch::stable::empty({num_groups}, torch::headeronly::ScalarType::Int,
                             std::nullopt, kv_cache.device());
    auto block_claims =
        torch::stable::empty({num_blocks64}, torch::headeronly::ScalarType::Int,
                             std::nullopt, kv_cache.device());
    BYTE_V2_CUDA_CHECK(cudaMemsetAsync(direct_ineligible.mutable_data_ptr(), 0,
                                       sizeof(int32_t), stream));
    BYTE_V2_CUDA_CHECK(cudaMemsetAsync(block_claims.mutable_data_ptr(), 0xFF,
                                       num_blocks64 * sizeof(int32_t),
                                       stream));

    constexpr int validation_threads = 256;
    const int validation_blocks =
        (num_groups + validation_threads - 1) / validation_threads;
    vllm::byte_v2_validate_prefill_direct_blocks_kernel<<<
        validation_blocks, validation_threads, 0, stream>>>(
        reinterpret_cast<const int64_t*>(slot_mapping.const_data_ptr()),
        reinterpret_cast<int32_t*>(group_block_ids.mutable_data_ptr()),
        reinterpret_cast<int32_t*>(block_claims.mutable_data_ptr()),
        reinterpret_cast<int32_t*>(direct_ineligible.mutable_data_ptr()),
        num_groups, num_blocks);
    BYTE_V2_CUDA_CHECK(cudaGetLastError());

    int32_t host_direct_ineligible = 0;
    BYTE_V2_CUDA_CHECK(cudaMemcpyAsync(
        &host_direct_ineligible, direct_ineligible.const_data_ptr(),
        sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    BYTE_V2_CUDA_CHECK(cudaStreamSynchronize(stream));

    if (host_direct_ineligible == 0) {
      auto direct_result =
          torch::stable::empty({num_groups + 1},
                               torch::headeronly::ScalarType::Int,
                               std::nullopt, kv_cache.device());
      BYTE_V2_CUDA_CHECK(cudaMemsetAsync(direct_result.mutable_data_ptr(), 0,
                                         sizeof(int32_t), stream));

      constexpr int update_threads = 256;
      vllm::byte_v2_prefill_direct_encode_blocks_kernel<<<
          num_groups, update_threads, 0, stream>>>(
          reinterpret_cast<const uint8_t*>(key.const_data_ptr()),
          reinterpret_cast<const uint8_t*>(value.const_data_ptr()),
          reinterpret_cast<uint8_t*>(kv_cache.mutable_data_ptr()),
          reinterpret_cast<const int32_t*>(group_block_ids.const_data_ptr()),
          fallback_pool_ptr, fallback_block_ids_ptr, fallback_next_slot_ptr,
          fallback_tile_ids_ptr, fallback_tile_next_slot_ptr,
          outlier_arena_ptr, outlier_block_flags_ptr, outlier_tile_bitmap_ptr,
          outlier_tile_meta_ptr, outlier_next_entry_ptr,
          reinterpret_cast<int32_t*>(direct_result.mutable_data_ptr()),
          num_groups, num_blocks, static_cast<int>(num_kv_heads),
          static_cast<int>(head_size), static_cast<int>(head_size_v),
	          static_cast<int>(page_size_bytes), static_cast<int>(raw_block_bytes),
	          fallback_pool_blocks, lossy_max_misses_per_tile,
	          outlier_arena_entries, outlier_max_per_tile,
	          v3_outlier_only_no_fallback, key.stride(0), key.stride(1),
	          key.stride(2), value.stride(0), value.stride(1), value.stride(2));
      BYTE_V2_CUDA_CHECK(cudaGetLastError());

      std::vector<int32_t> host_direct_result(num_groups + 1);
      BYTE_V2_CUDA_CHECK(cudaMemcpyAsync(
          host_direct_result.data(), direct_result.const_data_ptr(),
          host_direct_result.size() * sizeof(int32_t),
          cudaMemcpyDeviceToHost, stream));
      int32_t host_fallback_next_slot = 0;
      if (has_sparse_fallback) {
        BYTE_V2_CUDA_CHECK(cudaMemcpyAsync(
            &host_fallback_next_slot, fallback_next_slot->const_data_ptr(),
            sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
      }
      BYTE_V2_CUDA_CHECK(cudaStreamSynchronize(stream));

      if (host_direct_result[0] != 0) {
        std::string message =
            "Byte-v2 native prefill direct cache update failed: ";
        message +=
            vllm::byte_v2_cache_update_error_message(host_direct_result[0]);
        message += " (error_code=" + std::to_string(host_direct_result[0]);
        if (has_sparse_fallback) {
          message += ", fallback_pool_used=" +
                     std::to_string(host_fallback_next_slot);
          message += ", fallback_pool_capacity=" +
                     std::to_string(fallback_pool_blocks);
        }
        message += ")";
        STD_TORCH_CHECK(false, message);
      }

      std::vector<int64_t> packed_block_ids;
      packed_block_ids.reserve(num_groups);
      for (int group_idx = 0; group_idx < num_groups; ++group_idx) {
        const int32_t block_id = host_direct_result[group_idx + 1];
        if (block_id >= 0) {
          packed_block_ids.push_back(static_cast<int64_t>(block_id));
        }
      }
      auto packed = torch::stable::empty(
          {static_cast<int64_t>(packed_block_ids.size())},
          torch::headeronly::ScalarType::Long, std::nullopt,
          slot_mapping.device());
      if (!packed_block_ids.empty()) {
        BYTE_V2_CUDA_CHECK(cudaMemcpyAsync(
            packed.mutable_data_ptr(), packed_block_ids.data(),
            packed_block_ids.size() * sizeof(int64_t),
            cudaMemcpyHostToDevice, stream));
      }
      return packed;
    }
  }

  const char* batch_decode_append_env =
      std::getenv("VLLM_BYTE_V2_DECODE_APPEND_BATCH_FASTPATH");
  const bool force_batch_decode_append =
      batch_decode_append_env != nullptr &&
      std::atoi(batch_decode_append_env) != 0;
  const bool decode_append_supports_outlier_arena =
      !has_outlier_arena || use_v3_payload_layout;
  bool use_decode_append_fast_path =
      compressed_only_pages && has_sparse_fallback &&
      decode_append_supports_outlier_arena &&
      (num_tokens == 1 ||
       (num_tokens > 1 &&
        (prefill_capture_status != cudaStreamCaptureStatusNone ||
         force_batch_decode_append ||
         (decode_append_fast_path_safe && has_deferred_error))));
  if (!use_decode_append_fast_path && compressed_only_pages &&
      has_sparse_fallback && !has_outlier_arena && !has_deferred_error &&
      num_tokens > 1 &&
      prefill_capture_status == cudaStreamCaptureStatusNone) {
    auto validation_result =
        torch::stable::empty({2}, torch::headeronly::ScalarType::Int,
                             std::nullopt, kv_cache.device());
    BYTE_V2_CUDA_CHECK(cudaMemsetAsync(validation_result.mutable_data_ptr(), 0,
                                       2 * sizeof(int32_t), stream));
    constexpr int validation_threads = 256;
    const int validation_blocks =
        (num_tokens + validation_threads - 1) / validation_threads;
    vllm::byte_v2_validate_decode_append_slots_kernel<<<
        validation_blocks, validation_threads, 0, stream>>>(
        reinterpret_cast<const uint8_t*>(kv_cache.const_data_ptr()),
        reinterpret_cast<const int64_t*>(slot_mapping.const_data_ptr()),
        reinterpret_cast<int32_t*>(validation_result.mutable_data_ptr()),
        num_tokens, num_blocks, static_cast<int>(page_size_bytes));
    BYTE_V2_CUDA_CHECK(cudaGetLastError());

    int32_t host_validation_result[2] = {0, -1};
    BYTE_V2_CUDA_CHECK(cudaMemcpyAsync(
        host_validation_result, validation_result.const_data_ptr(),
        sizeof(host_validation_result), cudaMemcpyDeviceToHost, stream));
    BYTE_V2_CUDA_CHECK(cudaStreamSynchronize(stream));
    use_decode_append_fast_path = host_validation_result[0] == 0;
  }
  if (use_decode_append_fast_path) {
    auto fast_result =
        torch::stable::empty({num_tokens64 + 2},
                             torch::headeronly::ScalarType::Int,
                             std::nullopt, kv_cache.device());
    constexpr int update_threads = 256;
    if (num_tokens > 1) {
      const int result_init_blocks =
          (num_tokens + 1 + update_threads - 1) / update_threads;
      vllm::byte_v2_init_decode_append_result_kernel<<<
          result_init_blocks, update_threads, 0, stream>>>(
          reinterpret_cast<int32_t*>(fast_result.mutable_data_ptr()),
          num_tokens);
    }
    vllm::byte_v2_decode_append_cache_kernel<<<num_tokens, update_threads, 0,
                                                stream>>>(
        reinterpret_cast<const uint8_t*>(key.const_data_ptr()),
        reinterpret_cast<const uint8_t*>(value.const_data_ptr()),
        reinterpret_cast<uint8_t*>(kv_cache.mutable_data_ptr()),
        reinterpret_cast<const int64_t*>(slot_mapping.const_data_ptr()),
        fallback_pool_ptr, fallback_block_ids_ptr, fallback_next_slot_ptr,
        fallback_tile_ids_ptr, fallback_tile_next_slot_ptr,
        outlier_arena_ptr, outlier_block_flags_ptr, outlier_tile_bitmap_ptr,
        outlier_tile_meta_ptr, outlier_next_entry_ptr,
        reinterpret_cast<int32_t*>(fast_result.mutable_data_ptr()),
        num_tokens, num_blocks, static_cast<int>(num_kv_heads),
        static_cast<int>(head_size), static_cast<int>(head_size_v),
	        static_cast<int>(page_size_bytes), static_cast<int>(raw_block_bytes),
	        fallback_pool_blocks, lossy_max_misses_per_tile,
	        outlier_arena_entries, outlier_max_per_tile,
	        v3_outlier_only_no_fallback, key.stride(0),
	        key.stride(1), key.stride(2), value.stride(0),
	        value.stride(1), value.stride(2));
    BYTE_V2_CUDA_CHECK(cudaGetLastError());

    if (has_deferred_error) {
      vllm::byte_v2_record_deferred_cache_update_error_kernel<<<1, 1, 0,
                                                               stream>>>(
          reinterpret_cast<const int32_t*>(fast_result.const_data_ptr()),
          fallback_next_slot_ptr, deferred_error_ptr, fallback_pool_blocks);
      BYTE_V2_CUDA_CHECK(cudaGetLastError());
      return torch::stable::empty({0}, torch::headeronly::ScalarType::Long,
                                  std::nullopt, slot_mapping.device());
    }

    const char* skip_decode_append_sync_env =
        std::getenv("VLLM_BYTE_V2_DECODE_APPEND_SKIP_VALIDATION_SYNC");
    const bool skip_decode_append_sync =
        skip_decode_append_sync_env != nullptr &&
        std::atoi(skip_decode_append_sync_env) != 0;
    if (skip_decode_append_sync) {
      return torch::stable::empty({0}, torch::headeronly::ScalarType::Long,
                                  std::nullopt, slot_mapping.device());
    }

    cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
    BYTE_V2_CUDA_CHECK(cudaStreamIsCapturing(stream, &capture_status));
    if (capture_status != cudaStreamCaptureStatusNone) {
      return torch::stable::empty({0}, torch::headeronly::ScalarType::Long,
                                  std::nullopt, slot_mapping.device());
    }

    std::vector<int32_t> host_result(num_tokens + 2);
    BYTE_V2_CUDA_CHECK(cudaMemcpyAsync(
        host_result.data(), fast_result.const_data_ptr(),
        host_result.size() * sizeof(int32_t),
        cudaMemcpyDeviceToHost, stream));
    int32_t host_fallback_next_slot = 0;
    BYTE_V2_CUDA_CHECK(cudaMemcpyAsync(
        &host_fallback_next_slot, fallback_next_slot->const_data_ptr(),
        sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
    BYTE_V2_CUDA_CHECK(cudaStreamSynchronize(stream));
    if (host_result[0] != 0) {
      std::string message = "Byte-v2 native decode cache append failed: ";
      message += vllm::byte_v2_cache_update_error_message(host_result[0]);
      message += " (error_code=" + std::to_string(host_result[0]);
      message += ", detail=" + std::to_string(host_result[1]);
      message +=
          ", fallback_pool_used=" + std::to_string(host_fallback_next_slot);
      message +=
          ", fallback_pool_capacity=" + std::to_string(fallback_pool_blocks);
      message += ")";
      STD_TORCH_CHECK(false, message);
    }

    std::vector<int64_t> packed_block_ids;
    packed_block_ids.reserve(num_tokens);
    for (int token_idx = 0; token_idx < num_tokens; ++token_idx) {
      const int32_t block_id = host_result[2 + token_idx];
      if (block_id >= 0) {
        packed_block_ids.push_back(static_cast<int64_t>(block_id));
      }
    }
    if (packed_block_ids.empty()) {
      return torch::stable::empty({0}, torch::headeronly::ScalarType::Long,
                                  std::nullopt, slot_mapping.device());
    }

    auto packed = torch::stable::empty(
        {static_cast<int64_t>(packed_block_ids.size())},
        torch::headeronly::ScalarType::Long, std::nullopt,
        slot_mapping.device());
    BYTE_V2_CUDA_CHECK(cudaMemcpyAsync(
        packed.mutable_data_ptr(), packed_block_ids.data(),
        packed_block_ids.size() * sizeof(int64_t),
        cudaMemcpyHostToDevice, stream));
    return packed;
  }

  auto valid_rows =
      torch::stable::empty({num_blocks64}, torch::headeronly::ScalarType::Int,
                           std::nullopt, kv_cache.device());
  auto touched_flags =
      torch::stable::empty({num_blocks64}, torch::headeronly::ScalarType::Byte,
                           std::nullopt, kv_cache.device());
  auto packed_flags =
      torch::stable::empty({num_blocks64}, torch::headeronly::ScalarType::Byte,
                           std::nullopt, kv_cache.device());
  auto overwrite_flags =
      torch::stable::empty({num_blocks64}, torch::headeronly::ScalarType::Byte,
                           std::nullopt, kv_cache.device());
  auto error =
      torch::stable::empty({1}, torch::headeronly::ScalarType::Int,
                           std::nullopt, kv_cache.device());
  auto raw_staging = torch::stable::empty(
      {num_tokens64 * raw_block_bytes}, torch::headeronly::ScalarType::Byte,
      std::nullopt, kv_cache.device());
  auto block_token_indices =
      torch::stable::empty({num_blocks64 * vllm::kByteV2TileSize},
                           torch::headeronly::ScalarType::Int, std::nullopt,
                           kv_cache.device());
  BYTE_V2_CUDA_CHECK(cudaMemsetAsync(error.mutable_data_ptr(), 0, sizeof(int32_t),
                                 stream));
  BYTE_V2_CUDA_CHECK(cudaMemsetAsync(block_token_indices.mutable_data_ptr(), 0xFF,
                                 num_blocks64 * vllm::kByteV2TileSize *
                                     sizeof(int32_t),
                                 stream));
  BYTE_V2_CUDA_CHECK(cudaMemsetAsync(overwrite_flags.mutable_data_ptr(), 0,
                                 num_blocks64 * sizeof(uint8_t), stream));

  constexpr int init_threads = 256;
  const int init_blocks = (num_blocks + init_threads - 1) / init_threads;
  vllm::byte_v2_init_cache_update_kernel<<<init_blocks, init_threads, 0,
                                           stream>>>(
      reinterpret_cast<const uint8_t*>(kv_cache.const_data_ptr()),
      reinterpret_cast<int32_t*>(valid_rows.mutable_data_ptr()),
      reinterpret_cast<uint8_t*>(touched_flags.mutable_data_ptr()),
      reinterpret_cast<uint8_t*>(packed_flags.mutable_data_ptr()),
      reinterpret_cast<int32_t*>(error.mutable_data_ptr()), num_blocks,
      static_cast<int>(page_size_bytes));

  constexpr int update_threads = 256;
  if (compressed_only_pages) {
    const int mark_blocks = (num_tokens + update_threads - 1) / update_threads;
    vllm::byte_v2_mark_touched_tokens_kernel<<<mark_blocks, update_threads, 0,
                                               stream>>>(
        reinterpret_cast<const uint8_t*>(kv_cache.const_data_ptr()),
        reinterpret_cast<const int64_t*>(slot_mapping.const_data_ptr()),
        reinterpret_cast<int32_t*>(valid_rows.mutable_data_ptr()),
        reinterpret_cast<uint8_t*>(touched_flags.mutable_data_ptr()),
        reinterpret_cast<uint8_t*>(overwrite_flags.mutable_data_ptr()),
        reinterpret_cast<int32_t*>(block_token_indices.mutable_data_ptr()),
        reinterpret_cast<int32_t*>(error.mutable_data_ptr()), num_tokens,
        num_blocks, static_cast<int>(page_size_bytes));

    vllm::byte_v2_compress_touched_blocks_kernel<<<num_blocks, update_threads, 0,
                                                   stream>>>(
        reinterpret_cast<const uint8_t*>(key.const_data_ptr()),
        reinterpret_cast<const uint8_t*>(value.const_data_ptr()),
        reinterpret_cast<uint8_t*>(kv_cache.mutable_data_ptr()),
        reinterpret_cast<const int32_t*>(block_token_indices.const_data_ptr()),
        reinterpret_cast<const int32_t*>(valid_rows.const_data_ptr()),
        reinterpret_cast<const uint8_t*>(touched_flags.const_data_ptr()),
        reinterpret_cast<const uint8_t*>(overwrite_flags.const_data_ptr()),
        reinterpret_cast<uint8_t*>(packed_flags.mutable_data_ptr()),
        reinterpret_cast<uint8_t*>(raw_staging.mutable_data_ptr()),
        fallback_pool_ptr, fallback_block_ids_ptr, fallback_next_slot_ptr,
        fallback_tile_ids_ptr, fallback_tile_next_slot_ptr,
        outlier_block_flags_ptr, outlier_tile_bitmap_ptr,
        reinterpret_cast<int32_t*>(error.mutable_data_ptr()), num_tokens,
        num_blocks, static_cast<int>(num_kv_heads), static_cast<int>(head_size),
        static_cast<int>(head_size_v), static_cast<int>(page_size_bytes),
        static_cast<int>(raw_block_bytes), fallback_pool_blocks,
        lossy_max_misses_per_tile, key.stride(0), key.stride(1), key.stride(2),
        value.stride(0), value.stride(1), value.stride(2));
  } else {
    vllm::byte_v2_write_raw_tokens_kernel<<<num_tokens, update_threads, 0,
                                            stream>>>(
        reinterpret_cast<const uint8_t*>(key.const_data_ptr()),
        reinterpret_cast<const uint8_t*>(value.const_data_ptr()),
        reinterpret_cast<uint8_t*>(kv_cache.mutable_data_ptr()),
        reinterpret_cast<const int64_t*>(slot_mapping.const_data_ptr()),
        reinterpret_cast<int32_t*>(valid_rows.mutable_data_ptr()),
        reinterpret_cast<uint8_t*>(touched_flags.mutable_data_ptr()),
        reinterpret_cast<int32_t*>(error.mutable_data_ptr()), num_tokens,
        num_blocks, static_cast<int>(num_kv_heads), static_cast<int>(head_size),
        static_cast<int>(head_size_v), static_cast<int>(page_size_bytes),
        key.stride(0), key.stride(1), key.stride(2), value.stride(0),
        value.stride(1), value.stride(2));

    vllm::byte_v2_finalize_partial_pages_kernel<<<init_blocks, init_threads, 0,
                                                  stream>>>(
        reinterpret_cast<uint8_t*>(kv_cache.mutable_data_ptr()),
        reinterpret_cast<const int32_t*>(valid_rows.const_data_ptr()),
        reinterpret_cast<const uint8_t*>(touched_flags.const_data_ptr()),
        num_blocks, static_cast<int>(page_size_bytes));

    vllm::byte_v2_compress_full_pages_kernel<<<num_tokens, update_threads, 0,
                                               stream>>>(
        reinterpret_cast<uint8_t*>(kv_cache.mutable_data_ptr()),
        reinterpret_cast<const int64_t*>(slot_mapping.const_data_ptr()),
        reinterpret_cast<const int32_t*>(valid_rows.const_data_ptr()),
        reinterpret_cast<const uint8_t*>(touched_flags.const_data_ptr()),
        reinterpret_cast<uint8_t*>(packed_flags.mutable_data_ptr()),
        reinterpret_cast<uint8_t*>(raw_staging.mutable_data_ptr()), num_tokens,
        num_blocks, static_cast<int>(num_kv_heads), static_cast<int>(head_size),
        static_cast<int>(head_size_v), static_cast<int>(page_size_bytes),
        static_cast<int>(raw_block_bytes), lossy_max_misses_per_tile);
  }
  BYTE_V2_CUDA_CHECK(cudaGetLastError());

  cudaStreamCaptureStatus capture_status = cudaStreamCaptureStatusNone;
  BYTE_V2_CUDA_CHECK(cudaStreamIsCapturing(stream, &capture_status));
  if (capture_status != cudaStreamCaptureStatusNone) {
    return torch::stable::empty({0}, torch::headeronly::ScalarType::Long,
                                std::nullopt, slot_mapping.device());
  }

  int32_t host_error = 0;
  BYTE_V2_CUDA_CHECK(cudaMemcpyAsync(&host_error, error.const_data_ptr(),
                                 sizeof(int32_t), cudaMemcpyDeviceToHost,
                                 stream));
  int32_t host_fallback_next_slot = 0;
  if (has_sparse_fallback) {
    BYTE_V2_CUDA_CHECK(cudaMemcpyAsync(
        &host_fallback_next_slot, fallback_next_slot->const_data_ptr(),
        sizeof(int32_t), cudaMemcpyDeviceToHost, stream));
  }
  std::vector<uint8_t> host_packed_flags(num_blocks);
  BYTE_V2_CUDA_CHECK(cudaMemcpyAsync(host_packed_flags.data(),
                                 packed_flags.const_data_ptr(), num_blocks,
                                 cudaMemcpyDeviceToHost, stream));
  BYTE_V2_CUDA_CHECK(cudaStreamSynchronize(stream));
  if (host_error != 0) {
    std::string message = "Byte-v2 native cache update failed: ";
    message += vllm::byte_v2_cache_update_error_message(host_error);
    message += " (error_code=" + std::to_string(host_error);
    if (has_sparse_fallback) {
      message += ", fallback_pool_used=" +
                 std::to_string(host_fallback_next_slot);
      message += ", fallback_pool_capacity=" +
                 std::to_string(fallback_pool_blocks);
    }
    message += ")";
    STD_TORCH_CHECK(false, message);
  }

  std::vector<int64_t> packed_block_ids;
  packed_block_ids.reserve(num_blocks);
  for (int block_id = 0; block_id < num_blocks; ++block_id) {
    if (host_packed_flags[block_id] != 0) {
      packed_block_ids.push_back(block_id);
    }
  }

  auto packed = torch::stable::empty(
      {static_cast<int64_t>(packed_block_ids.size())},
      torch::headeronly::ScalarType::Long, std::nullopt, slot_mapping.device());
  if (!packed_block_ids.empty()) {
    BYTE_V2_CUDA_CHECK(cudaMemcpyAsync(packed.mutable_data_ptr(),
                                   packed_block_ids.data(),
                                   packed_block_ids.size() * sizeof(int64_t),
                                   cudaMemcpyHostToDevice, stream));
  }
  return packed;
}

torch::stable::Tensor byte_v2_paged_decode_attention(
    torch::stable::Tensor& query,     // [num_decode_tokens, num_heads, D]
    torch::stable::Tensor& kv_cache,  // [num_blocks, page_size_bytes]
    torch::stable::Tensor& block_table,
    torch::stable::Tensor& seq_lens, double scale, int64_t block_size,
    int64_t num_kv_heads, int64_t head_size, int64_t head_size_v,
    int64_t page_size_bytes,
    std::optional<torch::stable::Tensor> fallback_pool,
    std::optional<torch::stable::Tensor> fallback_block_ids,
    std::optional<torch::stable::Tensor> fallback_tile_ids,
    std::optional<torch::stable::Tensor> partial_workspace,
    std::optional<torch::stable::Tensor> outlier_arena,
    std::optional<torch::stable::Tensor> outlier_block_flags,
    std::optional<torch::stable::Tensor> outlier_tile_bitmap,
    std::optional<torch::stable::Tensor> outlier_tile_meta) {
  STD_TORCH_CHECK(query.device().is_cuda() && kv_cache.device().is_cuda() &&
                      block_table.device().is_cuda() &&
                      seq_lens.device().is_cuda(),
                  "Byte-v2 paged decode expects CUDA tensors");
  STD_TORCH_CHECK(query.device().index() == kv_cache.device().index() &&
                      block_table.device().index() == kv_cache.device().index() &&
                      seq_lens.device().index() == kv_cache.device().index(),
                  "Byte-v2 paged decode tensors must be on the same GPU");
  STD_TORCH_CHECK(query.scalar_type() ==
                      torch::headeronly::ScalarType::BFloat16,
                  "Byte-v2 paged decode expects BF16 query");
  STD_TORCH_CHECK(kv_cache.scalar_type() ==
                      torch::headeronly::ScalarType::Byte,
                  "Byte-v2 kv_cache must use uint8 storage");
  STD_TORCH_CHECK(block_table.scalar_type() ==
                          torch::headeronly::ScalarType::Int ||
                      block_table.scalar_type() ==
                          torch::headeronly::ScalarType::Long,
                  "Byte-v2 block_table must be int32 or int64");
  STD_TORCH_CHECK(seq_lens.scalar_type() ==
                          torch::headeronly::ScalarType::Int ||
                      seq_lens.scalar_type() ==
                          torch::headeronly::ScalarType::Long,
                  "Byte-v2 seq_lens must be int32 or int64");
  STD_TORCH_CHECK(query.is_contiguous() && kv_cache.is_contiguous(),
                  "Byte-v2 paged decode expects contiguous query/kv_cache");
  STD_TORCH_CHECK(block_size == vllm::kByteV2TileSize,
                  "Byte-v2 paged decode requires block_size=16");
  STD_TORCH_CHECK(head_size > 0 && head_size <= vllm::kByteV2MaxHeadSize &&
                      head_size_v > 0 &&
                          head_size_v <= vllm::kByteV2MaxHeadSize,
                  "Byte-v2 paged decode requires head sizes in [1, 128]");
  STD_TORCH_CHECK(head_size % vllm::kByteV2TileSize == 0 &&
                      head_size_v % vllm::kByteV2TileSize == 0,
                  "Byte-v2 paged decode head sizes must be multiples of 16");
  STD_TORCH_CHECK(query.dim() == 3 && kv_cache.dim() == 2 &&
                      block_table.dim() == 2 && seq_lens.dim() == 1,
                  "Byte-v2 paged decode tensor ranks are invalid");
  STD_TORCH_CHECK(query.size(2) == head_size &&
                      query.size(1) % num_kv_heads == 0,
                  "Byte-v2 paged decode query shape does not match metadata");

  const int64_t compressed_payload_bytes =
      num_kv_heads * (head_size / vllm::kByteV2TileSize +
                      head_size_v / vllm::kByteV2TileSize) *
      vllm::kByteV2FastTilePayloadBytes;
  const int64_t raw_block_bytes =
      vllm::kByteV2TileSize * num_kv_heads * (head_size + head_size_v) * 2;
  const int64_t compressed_page_size =
      vllm::kByteV2PageHeaderBytes + compressed_payload_bytes;
  const int64_t compressed_page_size_v3 =
      vllm::byte_v2_v3_page_size_bytes(
          static_cast<int>(num_kv_heads),
          static_cast<int>(head_size / vllm::kByteV2TileSize),
          static_cast<int>(head_size_v / vllm::kByteV2TileSize));
  const int64_t raw_overlay_page_size =
      vllm::kByteV2PageHeaderBytes +
      std::max(compressed_payload_bytes, raw_block_bytes);
  const int64_t total_tiles64 =
      num_kv_heads * (head_size / vllm::kByteV2TileSize +
                      head_size_v / vllm::kByteV2TileSize);
  const int64_t outlier_tile_bitmap_words64 = (total_tiles64 + 31) / 32;
  STD_TORCH_CHECK((page_size_bytes == compressed_page_size ||
                   page_size_bytes == compressed_page_size_v3 ||
                   page_size_bytes == raw_overlay_page_size) &&
                      kv_cache.size(1) == page_size_bytes,
                  "Byte-v2 paged decode page size mismatch");
  const bool use_v3_payload_layout = page_size_bytes == compressed_page_size_v3;
  const bool has_sparse_fallback =
      fallback_pool.has_value() && fallback_block_ids.has_value();
  STD_TORCH_CHECK(
      has_sparse_fallback ||
          (!fallback_pool.has_value() && !fallback_block_ids.has_value() &&
           !fallback_tile_ids.has_value()),
      "Byte-v2 sparse fallback decode arguments must be provided together");
  const bool has_tile_fallback =
      fallback_pool.has_value() && fallback_tile_ids.has_value();
  const bool has_outlier_arena =
      outlier_arena.has_value() || outlier_tile_meta.has_value();
  STD_TORCH_CHECK(
      outlier_arena.has_value() == outlier_tile_meta.has_value(),
      "Byte-v2 outlier decode arguments must be provided together");
  const bool has_outlier_block_flags = outlier_block_flags.has_value();
  STD_TORCH_CHECK(!has_outlier_block_flags || has_outlier_arena,
                  "Byte-v2 outlier block flags require outlier arena metadata");
  const bool has_outlier_tile_bitmap = outlier_tile_bitmap.has_value();
  STD_TORCH_CHECK(!has_outlier_tile_bitmap || has_outlier_arena,
                  "Byte-v2 outlier tile bitmap requires outlier arena metadata");
  if (has_sparse_fallback) {
    STD_TORCH_CHECK(fallback_pool->device().is_cuda() &&
                        fallback_block_ids->device().is_cuda(),
                    "Byte-v2 sparse fallback decode tensors must be CUDA "
                    "tensors");
    STD_TORCH_CHECK(fallback_pool->device().index() ==
                            kv_cache.device().index() &&
                        fallback_block_ids->device().index() ==
                            kv_cache.device().index(),
                    "Byte-v2 sparse fallback decode tensors must be on the "
                    "same GPU");
    STD_TORCH_CHECK(fallback_pool->scalar_type() ==
                            torch::headeronly::ScalarType::Byte &&
                        fallback_block_ids->scalar_type() ==
                            torch::headeronly::ScalarType::Int,
                    "Byte-v2 sparse fallback decode tensors have invalid "
                    "dtypes");
    STD_TORCH_CHECK(fallback_pool->is_contiguous() &&
                        fallback_block_ids->is_contiguous(),
                    "Byte-v2 sparse fallback decode tensors must be "
                    "contiguous");
    STD_TORCH_CHECK(fallback_pool->dim() == 2 &&
                        fallback_pool->size(1) == raw_block_bytes,
                    "Byte-v2 fallback_pool must have shape "
                    "[pool_blocks, raw_block_bytes]");
	    STD_TORCH_CHECK(fallback_block_ids->dim() == 1 &&
                        fallback_block_ids->size(0) == kv_cache.size(0),
                    "Byte-v2 fallback_block_ids must have one entry per KV "
                    "block");
    if (has_tile_fallback) {
      STD_TORCH_CHECK(fallback_tile_ids->device().is_cuda(),
                      "Byte-v2 tile fallback decode tensor must be CUDA");
      STD_TORCH_CHECK(fallback_tile_ids->device().index() ==
                          kv_cache.device().index(),
                      "Byte-v2 tile fallback decode tensor must be on the "
                      "same GPU");
      STD_TORCH_CHECK(fallback_tile_ids->scalar_type() ==
                          torch::headeronly::ScalarType::Int,
                      "Byte-v2 tile fallback decode tensor must be int32");
      STD_TORCH_CHECK(fallback_tile_ids->is_contiguous(),
                      "Byte-v2 tile fallback decode tensor must be contiguous");
      STD_TORCH_CHECK(fallback_tile_ids->dim() == 2 &&
                          fallback_tile_ids->size(0) == kv_cache.size(0) &&
                          fallback_tile_ids->size(1) == total_tiles64,
                      "Byte-v2 fallback_tile_ids must have shape "
                      "[num_blocks, total_tiles_per_block]");
    }
  }
  if (has_outlier_arena) {
    STD_TORCH_CHECK(outlier_arena->device().is_cuda() &&
                        outlier_tile_meta->device().is_cuda(),
                    "Byte-v2 outlier decode tensors must be CUDA tensors");
    STD_TORCH_CHECK(outlier_arena->device().index() ==
                            kv_cache.device().index() &&
                        outlier_tile_meta->device().index() ==
                            kv_cache.device().index(),
                    "Byte-v2 outlier decode tensors must be on the same GPU");
    STD_TORCH_CHECK(outlier_arena->scalar_type() ==
                            torch::headeronly::ScalarType::Int &&
                        outlier_tile_meta->scalar_type() ==
                            torch::headeronly::ScalarType::Int,
                    "Byte-v2 outlier decode tensors must use int32 dtype");
    STD_TORCH_CHECK(outlier_arena->is_contiguous() &&
                        outlier_tile_meta->is_contiguous(),
                    "Byte-v2 outlier decode tensors must be contiguous");
    STD_TORCH_CHECK(outlier_arena->dim() == 1,
                    "Byte-v2 outlier_arena must be a 1-D tensor");
    STD_TORCH_CHECK(outlier_tile_meta->dim() == 2 &&
                        outlier_tile_meta->size(0) == kv_cache.size(0) &&
                        outlier_tile_meta->size(1) == total_tiles64,
                    "Byte-v2 outlier_tile_meta must have shape "
                    "[num_blocks, total_tiles_per_block]");
  }
  if (has_outlier_block_flags) {
    STD_TORCH_CHECK(outlier_block_flags->device().is_cuda(),
                    "Byte-v2 outlier_block_flags must be a CUDA tensor");
    STD_TORCH_CHECK(outlier_block_flags->device().index() ==
                        kv_cache.device().index(),
                    "Byte-v2 outlier_block_flags must be on the same GPU");
    STD_TORCH_CHECK(outlier_block_flags->scalar_type() ==
                        torch::headeronly::ScalarType::Int,
                    "Byte-v2 outlier_block_flags must use int32 dtype");
    STD_TORCH_CHECK(outlier_block_flags->is_contiguous(),
                    "Byte-v2 outlier_block_flags must be contiguous");
    STD_TORCH_CHECK(outlier_block_flags->dim() == 1 &&
                        outlier_block_flags->size(0) == kv_cache.size(0),
                    "Byte-v2 outlier_block_flags must have one entry per KV "
                    "block");
  }
  if (has_outlier_tile_bitmap) {
    STD_TORCH_CHECK(outlier_tile_bitmap->device().is_cuda(),
                    "Byte-v2 outlier_tile_bitmap must be a CUDA tensor");
    STD_TORCH_CHECK(outlier_tile_bitmap->device().index() ==
                        kv_cache.device().index(),
                    "Byte-v2 outlier_tile_bitmap must be on the same GPU");
    STD_TORCH_CHECK(outlier_tile_bitmap->scalar_type() ==
                        torch::headeronly::ScalarType::Int,
                    "Byte-v2 outlier_tile_bitmap must use int32 dtype");
    STD_TORCH_CHECK(outlier_tile_bitmap->is_contiguous(),
                    "Byte-v2 outlier_tile_bitmap must be contiguous");
    STD_TORCH_CHECK(outlier_tile_bitmap->dim() == 2 &&
                        outlier_tile_bitmap->size(0) == kv_cache.size(0) &&
                        outlier_tile_bitmap->size(1) ==
                            outlier_tile_bitmap_words64,
                    "Byte-v2 outlier_tile_bitmap must have shape "
                    "[num_blocks, ceil(total_tiles_per_block / 32)]");
  }

  const int64_t num_decode_tokens64 = query.size(0);
  const int64_t num_heads64 = query.size(1);
  STD_TORCH_CHECK(num_decode_tokens64 <= INT_MAX && num_heads64 <= INT_MAX,
                  "Byte-v2 native paged decode supports at most INT_MAX "
                  "decode tokens and heads");
  auto output =
      torch::stable::empty({num_decode_tokens64, num_heads64, head_size_v},
                           query.scalar_type(), std::nullopt, query.device());
  if (num_decode_tokens64 == 0) {
    return output;
  }

  const torch::stable::accelerator::DeviceGuard device_guard(
      query.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream(query.get_device_index());
  const int max_head_size = std::max(head_size, head_size_v);
  const dim3 block(max_head_size <= 32 ? 32 : (max_head_size <= 64 ? 64 : 128));
  const uint8_t* fallback_pool_ptr =
      has_sparse_fallback
          ? reinterpret_cast<const uint8_t*>(fallback_pool->const_data_ptr())
          : nullptr;
  const int32_t* fallback_block_ids_ptr =
      has_sparse_fallback ? reinterpret_cast<const int32_t*>(
                                fallback_block_ids->const_data_ptr())
                          : nullptr;
  const int32_t* fallback_tile_ids_ptr =
      has_tile_fallback ? reinterpret_cast<const int32_t*>(
                              fallback_tile_ids->const_data_ptr())
                        : nullptr;
  const int32_t* outlier_arena_ptr =
      has_outlier_arena ? reinterpret_cast<const int32_t*>(
                              outlier_arena->const_data_ptr())
                        : nullptr;
  const int32_t* outlier_block_flags_ptr =
      has_outlier_block_flags
          ? reinterpret_cast<const int32_t*>(
                outlier_block_flags->const_data_ptr())
          : nullptr;
  const int32_t* outlier_tile_bitmap_ptr =
      has_outlier_tile_bitmap
          ? reinterpret_cast<const int32_t*>(
                outlier_tile_bitmap->const_data_ptr())
          : nullptr;
  const int32_t* outlier_tile_meta_ptr =
      has_outlier_arena ? reinterpret_cast<const int32_t*>(
                              outlier_tile_meta->const_data_ptr())
                        : nullptr;
  const int q_per_kv = static_cast<int>(num_heads64 / num_kv_heads);
  bool device_supports_bf16_wmma = false;
#ifndef USE_ROCM
  device_supports_bf16_wmma =
      byte_v2_device_supports_bf16_wmma(query.get_device_index());
#endif
  const bool use_gqa_wmma_kernel =
      device_supports_bf16_wmma && q_per_kv > 1 &&
      q_per_kv <= vllm::kByteV2MaxQPerKv;
  const bool use_gqa_shared_kernel =
      q_per_kv > 1 && q_per_kv <= vllm::kByteV2MaxQPerKv;
  const int num_logical_pages = static_cast<int>(block_table.size(1));
  const int active_head_groups =
      static_cast<int>(num_decode_tokens64) * static_cast<int>(num_kv_heads);
  int max_split_k = 1;
  if (num_logical_pages >= 256) {
    max_split_k = 128;
  } else if (num_logical_pages >= 128) {
    max_split_k = 64;
  } else if (num_logical_pages >= 96) {
    max_split_k = 32;
  } else if (num_logical_pages >= 16) {
    max_split_k = 16;
  }
  if (active_head_groups >= 64) {
    max_split_k = std::min(max_split_k, 8);
  } else if (active_head_groups >= 32) {
    max_split_k = std::min(max_split_k, 16);
  }
  if (const char* split_env = std::getenv("VLLM_BYTE_V2_DECODE_SPLIT_K")) {
    const int requested_split_k = std::atoi(split_env);
    if (requested_split_k > 0) {
      max_split_k = std::max(1, requested_split_k);
    }
  }
  int num_kv_splits = 1;
  if (use_gqa_wmma_kernel && max_split_k > 1 && num_logical_pages >= 16) {
    num_kv_splits = std::min(max_split_k, num_logical_pages);
  }
  const bool use_gqa_wmma_split_kernel = num_kv_splits > 1;
  bool use_decode_page_fastpath = false;
  if (const char* page_fastpath_env =
          std::getenv("VLLM_BYTE_V2_DECODE_PAGE_FASTPATH")) {
    use_decode_page_fastpath = std::atoi(page_fastpath_env) != 0;
  }
  bool use_decode_tile_fastpath = true;
  if (const char* tile_fastpath_env =
          std::getenv("VLLM_BYTE_V2_DECODE_TILE_FASTPATH")) {
    use_decode_tile_fastpath = std::atoi(tile_fastpath_env) != 0;
  }
  bool use_cute_stage1_env = false;
  if (const char* cute_stage1_env =
          std::getenv("VLLM_BYTE_V2_DECODE_CUTE_STAGE1")) {
    use_cute_stage1_env = std::atoi(cute_stage1_env) != 0;
  }
  bool use_cute_stage1_auto_env = false;
  if (const char* cute_stage1_auto_env =
          std::getenv("VLLM_BYTE_V2_DECODE_CUTE_STAGE1_AUTO")) {
    use_cute_stage1_auto_env = std::atoi(cute_stage1_auto_env) != 0;
  }
  bool use_fast_stage1_env = false;
  if (const char* fast_stage1_env =
          std::getenv("VLLM_BYTE_V2_DECODE_FAST_STAGE1")) {
    use_fast_stage1_env = std::atoi(fast_stage1_env) != 0;
  }
  bool use_flash_stage1_env = false;
  if (const char* flash_stage1_env =
          std::getenv("VLLM_BYTE_V2_DECODE_FLASH_STAGE1")) {
    use_flash_stage1_env = std::atoi(flash_stage1_env) != 0;
  }
  int cute_stage1_early_exit_mode = 0;
  if (const char* early_exit_env =
          std::getenv("VLLM_BYTE_V2_DECODE_CUTE_STAGE1_EARLY_EXIT")) {
    cute_stage1_early_exit_mode = std::atoi(early_exit_env);
    cute_stage1_early_exit_mode =
        std::max(0, std::min(7, cute_stage1_early_exit_mode));
  }
  bool use_aligned_u16_payload_load = false;
  if (const char* aligned_u16_env =
          std::getenv("VLLM_BYTE_V2_DECODE_ALIGNED_U16_PAYLOAD_LOAD")) {
    use_aligned_u16_payload_load = std::atoi(aligned_u16_env) != 0;
  }
  bool use_v3_warp_stripe_load = false;
  if (const char* v3_warp_stripe_env =
          std::getenv("VLLM_BYTE_V2_DECODE_V3_WARP_STRIPE_LOAD")) {
    use_v3_warp_stripe_load = std::atoi(v3_warp_stripe_env) != 0;
  }
  bool use_v3_cp_async_stage = false;
  if (const char* v3_cp_async_env =
          std::getenv("VLLM_BYTE_V2_DECODE_V3_CP_ASYNC_STAGE")) {
    use_v3_cp_async_stage = std::atoi(v3_cp_async_env) != 0;
  }
  bool use_v4_stage1_env = false;
  if (const char* v4_stage1_env =
          std::getenv("VLLM_BYTE_V2_DECODE_V4_STAGE1")) {
    use_v4_stage1_env = std::atoi(v4_stage1_env) != 0;
  }
  int v4_macro_pages = 8;
  if (const char* v4_macro_pages_env =
          std::getenv("VLLM_BYTE_V2_DECODE_V4_MACRO_PAGES")) {
    const int requested_macro_pages = std::atoi(v4_macro_pages_env);
    if (requested_macro_pages == 4 || requested_macro_pages == 8) {
      v4_macro_pages = requested_macro_pages;
    }
  }
  bool use_v4_specialized_stage1 = true;
  if (const char* v4_specialized_env =
          std::getenv("VLLM_BYTE_V2_DECODE_V4_SPECIALIZED")) {
    use_v4_specialized_stage1 = std::atoi(v4_specialized_env) != 0;
  }
  bool use_v4_block128_stage1 = false;
  if (const char* v4_block128_env =
          std::getenv("VLLM_BYTE_V2_DECODE_V4_BLOCK128")) {
    use_v4_block128_stage1 = std::atoi(v4_block128_env) != 0;
  }
	  bool use_v3_outlier_only_no_fallback = false;
	  if (const char* outlier_only_env =
	          std::getenv("VLLM_BYTE_V2_V3_OUTLIER_ONLY_NO_FALLBACK")) {
	    use_v3_outlier_only_no_fallback = std::atoi(outlier_only_env) != 0;
	  }
	  use_v3_outlier_only_no_fallback =
	      use_v3_outlier_only_no_fallback && use_v3_payload_layout &&
	      has_outlier_arena && !has_tile_fallback;
	  const bool use_cute_stage1_shape =
      use_gqa_wmma_split_kernel && use_decode_page_fastpath &&
      (page_size_bytes == compressed_page_size || use_v3_payload_layout) &&
      head_size == 128 && head_size_v == 128 && num_kv_heads == 8 &&
      num_heads64 == 32 && q_per_kv == 4;
  const bool has_cute_stage1_metadata =
      has_sparse_fallback || has_outlier_arena;
  const bool use_cute_stage1_auto_scale =
      active_head_groups >= 32 || num_logical_pages >= 128;
	  const bool use_cute_stage1_auto =
	      use_cute_stage1_auto_env && use_cute_stage1_shape &&
	      use_cute_stage1_auto_scale && has_cute_stage1_metadata &&
	      ((!has_outlier_arena && !has_outlier_block_flags &&
	        !has_outlier_tile_bitmap) ||
	       use_v3_outlier_only_no_fallback);
  const bool use_cute_stage1 =
      use_cute_stage1_shape && (use_cute_stage1_env || use_cute_stage1_auto);
  const bool use_cute_stage1_metadata =
      use_cute_stage1 && has_cute_stage1_metadata;
	  const bool use_cute_stage1_metadata_no_outlier =
	      use_cute_stage1_metadata && !has_outlier_arena &&
	      !has_outlier_block_flags && !has_outlier_tile_bitmap;
	  const bool use_cute_stage1_metadata_no_fallback_outlier =
	      use_cute_stage1_metadata && use_v3_outlier_only_no_fallback;
  const bool use_fast_stage1 =
      use_fast_stage1_env && use_gqa_wmma_split_kernel &&
      use_decode_page_fastpath && page_size_bytes == compressed_page_size &&
      head_size == 128 && head_size_v == 128 && num_kv_heads == 8 &&
      num_heads64 == 32 && q_per_kv == 4 && !has_sparse_fallback &&
      !has_tile_fallback && !has_outlier_arena &&
      !has_outlier_block_flags && !has_outlier_tile_bitmap;
  const bool use_flash_stage1 =
      use_flash_stage1_env && use_gqa_wmma_split_kernel &&
      use_decode_page_fastpath && page_size_bytes == compressed_page_size &&
      head_size == 128 && head_size_v == 128 && num_kv_heads == 8 &&
      num_heads64 == 32 && q_per_kv == 4 && !has_outlier_arena &&
      !has_outlier_block_flags && !has_outlier_tile_bitmap;
  const bool use_flash_stage1_metadata =
      use_flash_stage1 && (has_sparse_fallback || has_tile_fallback);
  const bool use_v4_stage1_no_metadata =
      !has_sparse_fallback && !has_tile_fallback && !has_outlier_arena &&
      !has_outlier_block_flags && !has_outlier_tile_bitmap;
  const bool use_v4_stage1_outlier_only =
      use_v3_outlier_only_no_fallback && has_outlier_arena &&
      !has_tile_fallback;
  const bool use_v4_stage1 =
      use_v4_stage1_env && use_gqa_wmma_split_kernel &&
      use_decode_page_fastpath && use_v3_payload_layout && head_size == 128 &&
      head_size_v == 128 && num_kv_heads == 8 && num_heads64 == 32 &&
      q_per_kv == 4 &&
      (use_v4_stage1_no_metadata || use_v4_stage1_outlier_only);
  use_v4_block128_stage1 =
      use_v4_block128_stage1 && use_v4_stage1 && use_v4_specialized_stage1 &&
      v4_macro_pages == 8;
  bool use_parallel_reduce = num_kv_splits >= 64;
  if (const char* parallel_reduce_env =
          std::getenv("VLLM_BYTE_V2_DECODE_PARALLEL_REDUCE")) {
    use_parallel_reduce = std::atoi(parallel_reduce_env) != 0;
  }
  const int64_t partial_stride = head_size_v + 1;
  const int64_t partial_numel =
      use_gqa_wmma_split_kernel
          ? num_decode_tokens64 * num_heads64 * num_kv_splits * partial_stride
          : 0;
  std::optional<torch::stable::Tensor> owned_split_partial_output;
  torch::stable::Tensor* split_partial_output = nullptr;
  if (use_gqa_wmma_split_kernel && partial_workspace.has_value()) {
    auto& workspace = partial_workspace.value();
    STD_TORCH_CHECK(workspace.device().is_cuda(),
                    "Byte-v2 partial_workspace must be a CUDA tensor");
    STD_TORCH_CHECK(workspace.device().index() == query.device().index(),
                    "Byte-v2 partial_workspace must be on the same GPU");
    STD_TORCH_CHECK(
        workspace.scalar_type() == torch::headeronly::ScalarType::Float,
        "Byte-v2 partial_workspace must be float32");
    STD_TORCH_CHECK(workspace.is_contiguous(),
                    "Byte-v2 partial_workspace must be contiguous");
    STD_TORCH_CHECK(workspace.dim() == 1 &&
                        workspace.size(0) >= partial_numel,
                    "Byte-v2 partial_workspace is too small");
    split_partial_output = &workspace;
  } else {
    owned_split_partial_output.emplace(torch::stable::empty(
        {partial_numel}, torch::headeronly::ScalarType::Float, std::nullopt,
        query.device()));
    split_partial_output = &owned_split_partial_output.value();
  }
#define CALL_BYTE_V2_DECODE(BLOCK_T, SEQ_T)                                  \
  do {                                                                        \
    if (use_gqa_wmma_split_kernel) {                                          \
      const dim3 split_grid(static_cast<unsigned int>(num_decode_tokens64),   \
                            static_cast<unsigned int>(num_kv_heads),          \
                            static_cast<unsigned int>(num_kv_splits));        \
      if (use_v4_stage1) {                                                    \
        if (use_v4_block128_stage1) {                                         \
          if (use_v4_stage1_outlier_only) {                                   \
            vllm::byte_v2_paged_decode_attention_gqa4_h128_v4_block128_split_stage1_kernel< \
                BLOCK_T, SEQ_T, true>                                         \
                <<<split_grid, block, 0, stream>>>(                           \
                    reinterpret_cast<const uint16_t*>(query.const_data_ptr()),\
                    reinterpret_cast<const uint8_t*>(kv_cache.const_data_ptr()),\
                    reinterpret_cast<const BLOCK_T*>(                         \
                        block_table.const_data_ptr()),                        \
                    reinterpret_cast<const SEQ_T*>(seq_lens.const_data_ptr()),\
                    reinterpret_cast<float*>(                                 \
                        split_partial_output->mutable_data_ptr()),            \
                    outlier_arena_ptr, outlier_tile_bitmap_ptr,               \
                    outlier_tile_meta_ptr, static_cast<float>(scale),         \
                    static_cast<int>(num_decode_tokens64),                    \
                    static_cast<int>(page_size_bytes), num_kv_splits,         \
                    block_table.stride(0), block_table.stride(1),             \
                    seq_lens.stride(0));                                      \
          } else {                                                            \
            vllm::byte_v2_paged_decode_attention_gqa4_h128_v4_block128_split_stage1_kernel< \
                BLOCK_T, SEQ_T, false>                                        \
                <<<split_grid, block, 0, stream>>>(                           \
                    reinterpret_cast<const uint16_t*>(query.const_data_ptr()),\
                    reinterpret_cast<const uint8_t*>(kv_cache.const_data_ptr()),\
                    reinterpret_cast<const BLOCK_T*>(                         \
                        block_table.const_data_ptr()),                        \
                    reinterpret_cast<const SEQ_T*>(seq_lens.const_data_ptr()),\
                    reinterpret_cast<float*>(                                 \
                        split_partial_output->mutable_data_ptr()),            \
                    nullptr, nullptr, nullptr, static_cast<float>(scale),     \
                    static_cast<int>(num_decode_tokens64),                    \
                    static_cast<int>(page_size_bytes), num_kv_splits,         \
                    block_table.stride(0), block_table.stride(1),             \
                    seq_lens.stride(0));                                      \
          }                                                                   \
        } else if (v4_macro_pages == 8) {                                     \
          if (!use_v4_specialized_stage1) {                                   \
            vllm::byte_v2_paged_decode_attention_gqa4_h128_v4_split_stage1_kernel< \
                BLOCK_T, SEQ_T, 8, false, false>                              \
                <<<split_grid, block, 0, stream>>>(                           \
                    reinterpret_cast<const uint16_t*>(query.const_data_ptr()),\
                    reinterpret_cast<const uint8_t*>(kv_cache.const_data_ptr()),\
                    reinterpret_cast<const BLOCK_T*>(                         \
                        block_table.const_data_ptr()),                        \
                    reinterpret_cast<const SEQ_T*>(seq_lens.const_data_ptr()),\
                    reinterpret_cast<float*>(                                 \
                        split_partial_output->mutable_data_ptr()),            \
                    fallback_pool_ptr, fallback_block_ids_ptr,                \
                    outlier_arena_ptr, outlier_tile_bitmap_ptr,               \
                    outlier_tile_meta_ptr,                                    \
                    static_cast<float>(scale),                                \
                    static_cast<int>(num_decode_tokens64),                    \
                    static_cast<int>(page_size_bytes),                        \
                    static_cast<int>(raw_block_bytes), num_kv_splits,         \
                    block_table.stride(0), block_table.stride(1),             \
                    seq_lens.stride(0));                                      \
          } else if (use_v4_stage1_outlier_only) {                            \
            vllm::byte_v2_paged_decode_attention_gqa4_h128_v4_split_stage1_kernel< \
                BLOCK_T, SEQ_T, 8, true, true>                                \
                <<<split_grid, block, 0, stream>>>(                           \
                    reinterpret_cast<const uint16_t*>(query.const_data_ptr()),\
                    reinterpret_cast<const uint8_t*>(kv_cache.const_data_ptr()),\
                    reinterpret_cast<const BLOCK_T*>(                         \
                        block_table.const_data_ptr()),                        \
                    reinterpret_cast<const SEQ_T*>(seq_lens.const_data_ptr()),\
                    reinterpret_cast<float*>(                                 \
                        split_partial_output->mutable_data_ptr()),            \
                    fallback_pool_ptr, fallback_block_ids_ptr,                \
                    outlier_arena_ptr, outlier_tile_bitmap_ptr,               \
                    outlier_tile_meta_ptr,                                    \
                    static_cast<float>(scale),                                \
                    static_cast<int>(num_decode_tokens64),                    \
                    static_cast<int>(page_size_bytes),                        \
                    static_cast<int>(raw_block_bytes), num_kv_splits,         \
                    block_table.stride(0), block_table.stride(1),             \
                    seq_lens.stride(0));                                      \
          } else {                                                            \
            vllm::byte_v2_paged_decode_attention_gqa4_h128_v4_split_stage1_kernel< \
                BLOCK_T, SEQ_T, 8, true, false>                               \
                <<<split_grid, block, 0, stream>>>(                           \
                    reinterpret_cast<const uint16_t*>(query.const_data_ptr()),\
                    reinterpret_cast<const uint8_t*>(kv_cache.const_data_ptr()),\
                    reinterpret_cast<const BLOCK_T*>(                         \
                        block_table.const_data_ptr()),                        \
                    reinterpret_cast<const SEQ_T*>(seq_lens.const_data_ptr()),\
                    reinterpret_cast<float*>(                                 \
                        split_partial_output->mutable_data_ptr()),            \
                    fallback_pool_ptr, fallback_block_ids_ptr, nullptr,       \
                    nullptr, nullptr, static_cast<float>(scale),              \
                    static_cast<int>(num_decode_tokens64),                    \
                    static_cast<int>(page_size_bytes),                        \
                    static_cast<int>(raw_block_bytes), num_kv_splits,         \
                    block_table.stride(0), block_table.stride(1),             \
                    seq_lens.stride(0));                                      \
          }                                                                   \
        } else {                                                              \
          if (!use_v4_specialized_stage1) {                                   \
            vllm::byte_v2_paged_decode_attention_gqa4_h128_v4_split_stage1_kernel< \
                BLOCK_T, SEQ_T, 4, false, false>                              \
                <<<split_grid, block, 0, stream>>>(                           \
                    reinterpret_cast<const uint16_t*>(query.const_data_ptr()),\
                    reinterpret_cast<const uint8_t*>(kv_cache.const_data_ptr()),\
                    reinterpret_cast<const BLOCK_T*>(                         \
                        block_table.const_data_ptr()),                        \
                    reinterpret_cast<const SEQ_T*>(seq_lens.const_data_ptr()),\
                    reinterpret_cast<float*>(                                 \
                        split_partial_output->mutable_data_ptr()),            \
                    fallback_pool_ptr, fallback_block_ids_ptr,                \
                    outlier_arena_ptr, outlier_tile_bitmap_ptr,               \
                    outlier_tile_meta_ptr,                                    \
                    static_cast<float>(scale),                                \
                    static_cast<int>(num_decode_tokens64),                    \
                    static_cast<int>(page_size_bytes),                        \
                    static_cast<int>(raw_block_bytes), num_kv_splits,         \
                    block_table.stride(0), block_table.stride(1),             \
                    seq_lens.stride(0));                                      \
          } else if (use_v4_stage1_outlier_only) {                            \
            vllm::byte_v2_paged_decode_attention_gqa4_h128_v4_split_stage1_kernel< \
                BLOCK_T, SEQ_T, 4, true, true>                                \
                <<<split_grid, block, 0, stream>>>(                           \
                    reinterpret_cast<const uint16_t*>(query.const_data_ptr()),\
                    reinterpret_cast<const uint8_t*>(kv_cache.const_data_ptr()),\
                    reinterpret_cast<const BLOCK_T*>(                         \
                        block_table.const_data_ptr()),                        \
                    reinterpret_cast<const SEQ_T*>(seq_lens.const_data_ptr()),\
                    reinterpret_cast<float*>(                                 \
                        split_partial_output->mutable_data_ptr()),            \
                    fallback_pool_ptr, fallback_block_ids_ptr,                \
                    outlier_arena_ptr, outlier_tile_bitmap_ptr,               \
                    outlier_tile_meta_ptr,                                    \
                    static_cast<float>(scale),                                \
                    static_cast<int>(num_decode_tokens64),                    \
                    static_cast<int>(page_size_bytes),                        \
                    static_cast<int>(raw_block_bytes), num_kv_splits,         \
                    block_table.stride(0), block_table.stride(1),             \
                    seq_lens.stride(0));                                      \
          } else {                                                            \
            vllm::byte_v2_paged_decode_attention_gqa4_h128_v4_split_stage1_kernel< \
                BLOCK_T, SEQ_T, 4, true, false>                               \
                <<<split_grid, block, 0, stream>>>(                           \
                    reinterpret_cast<const uint16_t*>(query.const_data_ptr()),\
                    reinterpret_cast<const uint8_t*>(kv_cache.const_data_ptr()),\
                    reinterpret_cast<const BLOCK_T*>(                         \
                        block_table.const_data_ptr()),                        \
                    reinterpret_cast<const SEQ_T*>(seq_lens.const_data_ptr()),\
                    reinterpret_cast<float*>(                                 \
                        split_partial_output->mutable_data_ptr()),            \
                    fallback_pool_ptr, fallback_block_ids_ptr, nullptr,       \
                    nullptr, nullptr, static_cast<float>(scale),              \
                    static_cast<int>(num_decode_tokens64),                    \
                    static_cast<int>(page_size_bytes),                        \
                    static_cast<int>(raw_block_bytes), num_kv_splits,         \
                    block_table.stride(0), block_table.stride(1),             \
                    seq_lens.stride(0));                                      \
          }                                                                   \
        }                                                                     \
      } else if (use_fast_stage1) {                                           \
        vllm::byte_v2_paged_decode_attention_gqa4_h128_fast_split_stage1_kernel< \
            BLOCK_T, SEQ_T>                                                   \
            <<<split_grid, block, 0, stream>>>(                               \
                reinterpret_cast<const uint16_t*>(query.const_data_ptr()),    \
                reinterpret_cast<const uint8_t*>(kv_cache.const_data_ptr()),  \
                reinterpret_cast<const BLOCK_T*>(                             \
                    block_table.const_data_ptr()),                            \
                reinterpret_cast<const SEQ_T*>(seq_lens.const_data_ptr()),    \
                reinterpret_cast<float*>(                                     \
                    split_partial_output->mutable_data_ptr()),                \
                static_cast<float>(scale),                                    \
                static_cast<int>(num_decode_tokens64),                        \
                static_cast<int>(page_size_bytes), num_kv_splits,             \
                block_table.stride(0), block_table.stride(1),                 \
                seq_lens.stride(0));                                          \
      } else if (use_flash_stage1 && !use_flash_stage1_metadata) {            \
        vllm::byte_v2_paged_decode_attention_gqa4_h128_flash_split_stage1_kernel< \
            BLOCK_T, SEQ_T, false>                                            \
            <<<split_grid, block, 0, stream>>>(                               \
                reinterpret_cast<const uint16_t*>(query.const_data_ptr()),    \
                reinterpret_cast<const uint8_t*>(kv_cache.const_data_ptr()),  \
                reinterpret_cast<const BLOCK_T*>(                             \
                    block_table.const_data_ptr()),                            \
                reinterpret_cast<const SEQ_T*>(seq_lens.const_data_ptr()),    \
                reinterpret_cast<float*>(                                     \
                    split_partial_output->mutable_data_ptr()),                \
                nullptr, nullptr, nullptr, static_cast<float>(scale),         \
                static_cast<int>(num_decode_tokens64),                        \
                static_cast<int>(page_size_bytes),                            \
                static_cast<int>(raw_block_bytes), num_kv_splits,             \
                block_table.stride(0), block_table.stride(1),                 \
                seq_lens.stride(0), use_decode_tile_fastpath);                \
      } else if (use_flash_stage1_metadata) {                                 \
        vllm::byte_v2_paged_decode_attention_gqa4_h128_flash_split_stage1_kernel< \
            BLOCK_T, SEQ_T, true>                                             \
            <<<split_grid, block, 0, stream>>>(                               \
                reinterpret_cast<const uint16_t*>(query.const_data_ptr()),    \
                reinterpret_cast<const uint8_t*>(kv_cache.const_data_ptr()),  \
                reinterpret_cast<const BLOCK_T*>(                             \
                    block_table.const_data_ptr()),                            \
                reinterpret_cast<const SEQ_T*>(seq_lens.const_data_ptr()),    \
                reinterpret_cast<float*>(                                     \
                    split_partial_output->mutable_data_ptr()),                \
                fallback_pool_ptr, fallback_block_ids_ptr,                    \
                fallback_tile_ids_ptr, static_cast<float>(scale),             \
                static_cast<int>(num_decode_tokens64),                        \
                static_cast<int>(page_size_bytes),                            \
                static_cast<int>(raw_block_bytes), num_kv_splits,             \
                block_table.stride(0), block_table.stride(1),                 \
                seq_lens.stride(0), use_decode_tile_fastpath);                \
	      } else if (use_cute_stage1 && !use_cute_stage1_metadata) {              \
	        vllm::byte_v2_paged_decode_attention_gqa_cute_split_stage1_kernel<   \
	            BLOCK_T, SEQ_T, false, true, true>                                \
            <<<split_grid, block, 0, stream>>>(                               \
                reinterpret_cast<const uint16_t*>(query.const_data_ptr()),    \
                reinterpret_cast<const uint8_t*>(kv_cache.const_data_ptr()),  \
                reinterpret_cast<const BLOCK_T*>(                             \
                    block_table.const_data_ptr()),                            \
                reinterpret_cast<const SEQ_T*>(seq_lens.const_data_ptr()),    \
                reinterpret_cast<float*>(                                     \
                    split_partial_output->mutable_data_ptr()),                \
                nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,         \
                nullptr,                                                      \
                static_cast<float>(scale),                                    \
                static_cast<int>(num_decode_tokens64),                        \
                static_cast<int>(num_heads64),                                \
                static_cast<int>(num_kv_heads),                               \
                static_cast<int>(page_size_bytes),                            \
                static_cast<int>(raw_block_bytes), num_kv_splits,             \
                block_table.stride(0), block_table.stride(1),                 \
                seq_lens.stride(0), use_decode_tile_fastpath,                 \
                use_aligned_u16_payload_load,                                 \
                use_v3_warp_stripe_load,                                      \
	                use_v3_cp_async_stage,                                        \
	                cute_stage1_early_exit_mode);                                 \
		      } else if (use_cute_stage1_metadata_no_outlier) {                       \
		        vllm::byte_v2_paged_decode_attention_gqa_cute_split_stage1_kernel<   \
		            BLOCK_T, SEQ_T, true, true, false>                                \
	            <<<split_grid, block, 0, stream>>>(                               \
	                reinterpret_cast<const uint16_t*>(query.const_data_ptr()),    \
	                reinterpret_cast<const uint8_t*>(kv_cache.const_data_ptr()),  \
	                reinterpret_cast<const BLOCK_T*>(                             \
	                    block_table.const_data_ptr()),                            \
	                reinterpret_cast<const SEQ_T*>(seq_lens.const_data_ptr()),    \
	                reinterpret_cast<float*>(                                     \
	                    split_partial_output->mutable_data_ptr()),                \
	                fallback_pool_ptr, fallback_block_ids_ptr,                    \
	                fallback_tile_ids_ptr, nullptr, nullptr, nullptr, nullptr,    \
	                static_cast<float>(scale),                                    \
	                static_cast<int>(num_decode_tokens64),                        \
	                static_cast<int>(num_heads64),                                \
	                static_cast<int>(num_kv_heads),                               \
	                static_cast<int>(page_size_bytes),                            \
	                static_cast<int>(raw_block_bytes), num_kv_splits,             \
	                block_table.stride(0), block_table.stride(1),                 \
	                seq_lens.stride(0), use_decode_tile_fastpath,                 \
	                use_aligned_u16_payload_load,                                 \
	                use_v3_warp_stripe_load,                                      \
	                use_v3_cp_async_stage,                                        \
	                cute_stage1_early_exit_mode);                                 \
		      } else if (use_cute_stage1_metadata_no_fallback_outlier) {              \
		        vllm::byte_v2_paged_decode_attention_gqa_cute_split_stage1_kernel<   \
		            BLOCK_T, SEQ_T, true, false, true>                                \
		            <<<split_grid, block, 0, stream>>>(                               \
		                reinterpret_cast<const uint16_t*>(query.const_data_ptr()),    \
		                reinterpret_cast<const uint8_t*>(kv_cache.const_data_ptr()),  \
		                reinterpret_cast<const BLOCK_T*>(                             \
		                    block_table.const_data_ptr()),                            \
		                reinterpret_cast<const SEQ_T*>(seq_lens.const_data_ptr()),    \
		                reinterpret_cast<float*>(                                     \
		                    split_partial_output->mutable_data_ptr()),                \
		                fallback_pool_ptr, fallback_block_ids_ptr, nullptr,           \
		                outlier_arena_ptr, outlier_block_flags_ptr,                   \
		                outlier_tile_bitmap_ptr, outlier_tile_meta_ptr,               \
		                static_cast<float>(scale),                                    \
		                static_cast<int>(num_decode_tokens64),                        \
		                static_cast<int>(num_heads64),                                \
		                static_cast<int>(num_kv_heads),                               \
		                static_cast<int>(page_size_bytes),                            \
		                static_cast<int>(raw_block_bytes), num_kv_splits,             \
		                block_table.stride(0), block_table.stride(1),                 \
		                seq_lens.stride(0), true,                                     \
		                use_aligned_u16_payload_load,                                 \
		                use_v3_warp_stripe_load,                                      \
		                use_v3_cp_async_stage,                                        \
		                cute_stage1_early_exit_mode);                                 \
		      } else if (use_cute_stage1_metadata) {                                  \
		        vllm::byte_v2_paged_decode_attention_gqa_cute_split_stage1_kernel<   \
		            BLOCK_T, SEQ_T, true, false, false>                               \
	            <<<split_grid, block, 0, stream>>>(                               \
	                reinterpret_cast<const uint16_t*>(query.const_data_ptr()),    \
	                reinterpret_cast<const uint8_t*>(kv_cache.const_data_ptr()),  \
	                reinterpret_cast<const BLOCK_T*>(                             \
	                    block_table.const_data_ptr()),                            \
	                reinterpret_cast<const SEQ_T*>(seq_lens.const_data_ptr()),    \
	                reinterpret_cast<float*>(                                     \
	                    split_partial_output->mutable_data_ptr()),                \
	                fallback_pool_ptr, fallback_block_ids_ptr,                    \
	                fallback_tile_ids_ptr, outlier_arena_ptr,                     \
	                outlier_block_flags_ptr, outlier_tile_bitmap_ptr,             \
	                outlier_tile_meta_ptr, static_cast<float>(scale),             \
	                static_cast<int>(num_decode_tokens64),                        \
	                static_cast<int>(num_heads64),                                \
	                static_cast<int>(num_kv_heads),                               \
	                static_cast<int>(page_size_bytes),                            \
	                static_cast<int>(raw_block_bytes), num_kv_splits,             \
	                block_table.stride(0), block_table.stride(1),                 \
	                seq_lens.stride(0), use_decode_tile_fastpath,                 \
	                use_aligned_u16_payload_load,                                 \
	                use_v3_warp_stripe_load,                                      \
	                use_v3_cp_async_stage,                                        \
	                cute_stage1_early_exit_mode);                                 \
	      } else {                                                                \
        vllm::byte_v2_paged_decode_attention_gqa_wmma_split_stage1_kernel<   \
            BLOCK_T, SEQ_T>                                                   \
            <<<split_grid, block, 0, stream>>>(                               \
                reinterpret_cast<const uint16_t*>(query.const_data_ptr()),    \
                reinterpret_cast<const uint8_t*>(kv_cache.const_data_ptr()),  \
                reinterpret_cast<const BLOCK_T*>(                             \
                    block_table.const_data_ptr()),                            \
                reinterpret_cast<const SEQ_T*>(seq_lens.const_data_ptr()),    \
                reinterpret_cast<float*>(                                     \
                    split_partial_output->mutable_data_ptr()),                \
                fallback_pool_ptr, fallback_block_ids_ptr,                    \
                fallback_tile_ids_ptr,                                        \
                outlier_arena_ptr, outlier_block_flags_ptr,                   \
                outlier_tile_bitmap_ptr, outlier_tile_meta_ptr,               \
                static_cast<float>(scale),                                    \
                static_cast<int>(num_decode_tokens64),                        \
                static_cast<int>(num_heads64),                                \
                static_cast<int>(num_kv_heads),                               \
                static_cast<int>(head_size), static_cast<int>(head_size_v),   \
                static_cast<int>(page_size_bytes),                            \
                static_cast<int>(raw_block_bytes), num_kv_splits,             \
                block_table.stride(0), block_table.stride(1),                 \
                seq_lens.stride(0), use_decode_page_fastpath,                 \
                use_decode_tile_fastpath);                                    \
      }                                                                       \
      const dim3 reduce_grid(static_cast<unsigned int>(num_decode_tokens64),  \
                             static_cast<unsigned int>(num_heads64));         \
      if (use_parallel_reduce) {                                              \
        vllm::byte_v2_paged_decode_attention_split_reduce_parallel_kernel     \
            <<<reduce_grid, block, 0, stream>>>(                              \
                reinterpret_cast<const float*>(                               \
                    split_partial_output->const_data_ptr()),                  \
                reinterpret_cast<uint16_t*>(output.mutable_data_ptr()),       \
                static_cast<int>(num_decode_tokens64),                        \
                static_cast<int>(num_heads64), static_cast<int>(head_size_v), \
                num_kv_splits);                                               \
      } else {                                                                \
        vllm::byte_v2_paged_decode_attention_split_reduce_kernel              \
            <<<reduce_grid, block, 0, stream>>>(                              \
                reinterpret_cast<const float*>(                               \
                    split_partial_output->const_data_ptr()),                  \
                reinterpret_cast<uint16_t*>(output.mutable_data_ptr()),       \
                static_cast<int>(num_decode_tokens64),                        \
                static_cast<int>(num_heads64), static_cast<int>(head_size_v), \
                num_kv_splits);                                               \
      }                                                                       \
    } else if (use_gqa_wmma_kernel) {                                         \
      const dim3 gqa_grid(static_cast<unsigned int>(num_decode_tokens64),     \
                          static_cast<unsigned int>(num_kv_heads));           \
      vllm::byte_v2_paged_decode_attention_gqa_wmma_kernel<BLOCK_T, SEQ_T>   \
          <<<gqa_grid, block, 0, stream>>>(                                   \
              reinterpret_cast<const uint16_t*>(query.const_data_ptr()),      \
              reinterpret_cast<const uint8_t*>(kv_cache.const_data_ptr()),    \
              reinterpret_cast<const BLOCK_T*>(block_table.const_data_ptr()), \
              reinterpret_cast<const SEQ_T*>(seq_lens.const_data_ptr()),      \
              reinterpret_cast<uint16_t*>(output.mutable_data_ptr()),         \
              fallback_pool_ptr, fallback_block_ids_ptr,                      \
              fallback_tile_ids_ptr,                                          \
              outlier_arena_ptr, outlier_block_flags_ptr,                     \
              outlier_tile_bitmap_ptr, outlier_tile_meta_ptr,                 \
              static_cast<float>(scale),                                      \
              static_cast<int>(num_decode_tokens64),                          \
              static_cast<int>(num_heads64), static_cast<int>(num_kv_heads),  \
              static_cast<int>(head_size), static_cast<int>(head_size_v),     \
              static_cast<int>(page_size_bytes),                              \
              static_cast<int>(raw_block_bytes), block_table.stride(0),       \
              block_table.stride(1), seq_lens.stride(0));                     \
    } else if (use_gqa_shared_kernel) {                                       \
      const dim3 gqa_grid(static_cast<unsigned int>(num_decode_tokens64),     \
                          static_cast<unsigned int>(num_kv_heads));           \
      vllm::byte_v2_paged_decode_attention_gqa_shared_kernel<BLOCK_T, SEQ_T> \
          <<<gqa_grid, block, 0, stream>>>(                                   \
              reinterpret_cast<const uint16_t*>(query.const_data_ptr()),      \
              reinterpret_cast<const uint8_t*>(kv_cache.const_data_ptr()),    \
              reinterpret_cast<const BLOCK_T*>(block_table.const_data_ptr()), \
              reinterpret_cast<const SEQ_T*>(seq_lens.const_data_ptr()),      \
              reinterpret_cast<uint16_t*>(output.mutable_data_ptr()),         \
              fallback_pool_ptr, fallback_block_ids_ptr,                      \
              fallback_tile_ids_ptr,                                          \
              outlier_arena_ptr, outlier_block_flags_ptr,                     \
              outlier_tile_bitmap_ptr, outlier_tile_meta_ptr,                 \
              static_cast<float>(scale),                                      \
              static_cast<int>(num_decode_tokens64),                          \
              static_cast<int>(num_heads64), static_cast<int>(num_kv_heads),  \
              static_cast<int>(head_size), static_cast<int>(head_size_v),     \
              static_cast<int>(page_size_bytes),                              \
              static_cast<int>(raw_block_bytes), block_table.stride(0),       \
              block_table.stride(1), seq_lens.stride(0));                     \
    } else {                                                                  \
      const dim3 grid(static_cast<unsigned int>(num_decode_tokens64),         \
                      static_cast<unsigned int>(num_heads64));                \
      vllm::byte_v2_paged_decode_attention_kernel<BLOCK_T, SEQ_T>            \
          <<<grid, block, 0, stream>>>(                                       \
              reinterpret_cast<const uint16_t*>(query.const_data_ptr()),      \
              reinterpret_cast<const uint8_t*>(kv_cache.const_data_ptr()),    \
              reinterpret_cast<const BLOCK_T*>(block_table.const_data_ptr()), \
              reinterpret_cast<const SEQ_T*>(seq_lens.const_data_ptr()),      \
              reinterpret_cast<uint16_t*>(output.mutable_data_ptr()),         \
              fallback_pool_ptr, fallback_block_ids_ptr,                      \
              fallback_tile_ids_ptr,                                          \
              outlier_arena_ptr, outlier_block_flags_ptr,                     \
              outlier_tile_bitmap_ptr, outlier_tile_meta_ptr,                 \
              static_cast<float>(scale),                                      \
              static_cast<int>(num_decode_tokens64),                          \
              static_cast<int>(num_heads64), static_cast<int>(num_kv_heads),  \
              static_cast<int>(head_size), static_cast<int>(head_size_v),     \
              static_cast<int>(page_size_bytes),                              \
              static_cast<int>(raw_block_bytes), block_table.stride(0),       \
              block_table.stride(1), seq_lens.stride(0));                     \
    }                                                                         \
  } while (0)

  if (block_table.scalar_type() == torch::headeronly::ScalarType::Int &&
      seq_lens.scalar_type() == torch::headeronly::ScalarType::Int) {
    CALL_BYTE_V2_DECODE(int32_t, int32_t);
  } else if (block_table.scalar_type() ==
                 torch::headeronly::ScalarType::Int &&
             seq_lens.scalar_type() == torch::headeronly::ScalarType::Long) {
    CALL_BYTE_V2_DECODE(int32_t, int64_t);
  } else if (block_table.scalar_type() ==
                 torch::headeronly::ScalarType::Long &&
             seq_lens.scalar_type() == torch::headeronly::ScalarType::Int) {
    CALL_BYTE_V2_DECODE(int64_t, int32_t);
  } else {
    CALL_BYTE_V2_DECODE(int64_t, int64_t);
  }
#undef CALL_BYTE_V2_DECODE

  BYTE_V2_CUDA_CHECK(cudaGetLastError());
  return output;
}

void byte_v2_decompress_cache_to_bf16(
    torch::stable::Tensor& kv_cache, torch::stable::Tensor& key_cache,
    torch::stable::Tensor& value_cache, int64_t block_size,
    int64_t num_kv_heads, int64_t head_size, int64_t head_size_v,
    int64_t page_size_bytes,
    std::optional<torch::stable::Tensor> fallback_pool,
    std::optional<torch::stable::Tensor> fallback_block_ids,
    std::optional<torch::stable::Tensor> fallback_tile_ids,
    std::optional<torch::stable::Tensor> outlier_arena,
    std::optional<torch::stable::Tensor> outlier_tile_bitmap,
    std::optional<torch::stable::Tensor> outlier_tile_meta) {
  STD_TORCH_CHECK(kv_cache.device().is_cuda() &&
                      key_cache.device().is_cuda() &&
                      value_cache.device().is_cuda(),
                  "Byte-v2 decompress expects CUDA tensors");
  STD_TORCH_CHECK(kv_cache.device().index() == key_cache.device().index() &&
                      kv_cache.device().index() == value_cache.device().index(),
                  "Byte-v2 decompress tensors must be on the same GPU");
  STD_TORCH_CHECK(kv_cache.scalar_type() ==
                      torch::headeronly::ScalarType::Byte,
                  "Byte-v2 kv_cache must use uint8 storage");
  STD_TORCH_CHECK(key_cache.scalar_type() ==
                          torch::headeronly::ScalarType::BFloat16 &&
                      value_cache.scalar_type() ==
                          torch::headeronly::ScalarType::BFloat16,
                  "Byte-v2 decompress workspaces must be BF16");
  STD_TORCH_CHECK(kv_cache.is_contiguous() && key_cache.is_contiguous() &&
                      value_cache.is_contiguous(),
                  "Byte-v2 decompress expects contiguous tensors");
  STD_TORCH_CHECK(block_size == vllm::kByteV2TileSize,
                  "Byte-v2 decompress requires block_size=16");
  STD_TORCH_CHECK(head_size > 0 && head_size <= vllm::kByteV2MaxHeadSize &&
                      head_size_v > 0 &&
                          head_size_v <= vllm::kByteV2MaxHeadSize,
                  "Byte-v2 decompress requires head sizes in [1, 128]");
  STD_TORCH_CHECK(head_size % vllm::kByteV2TileSize == 0 &&
                      head_size_v % vllm::kByteV2TileSize == 0,
                  "Byte-v2 decompress head sizes must be multiples of 16");
  STD_TORCH_CHECK(kv_cache.dim() == 2 && key_cache.dim() == 4 &&
                      value_cache.dim() == 4,
                  "Byte-v2 decompress tensor ranks are invalid");
  const int64_t num_blocks64 = kv_cache.size(0);
  STD_TORCH_CHECK(num_blocks64 <= INT_MAX,
                  "Byte-v2 decompress supports at most INT_MAX KV blocks");
  STD_TORCH_CHECK(
      key_cache.size(0) == num_blocks64 &&
          key_cache.size(1) == block_size &&
          key_cache.size(2) == num_kv_heads &&
          key_cache.size(3) == head_size &&
          value_cache.size(0) == num_blocks64 &&
          value_cache.size(1) == block_size &&
          value_cache.size(2) == num_kv_heads &&
          value_cache.size(3) == head_size_v,
      "Byte-v2 decompress workspaces must have shapes "
      "[num_blocks, block_size, num_kv_heads, head_size(_v)]");

  const int64_t compressed_payload_bytes =
      num_kv_heads * (head_size / vllm::kByteV2TileSize +
                      head_size_v / vllm::kByteV2TileSize) *
      vllm::kByteV2FastTilePayloadBytes;
  const int64_t raw_block_bytes =
      vllm::kByteV2TileSize * num_kv_heads * (head_size + head_size_v) * 2;
  const int64_t compressed_page_size =
      vllm::kByteV2PageHeaderBytes + compressed_payload_bytes;
  const int64_t compressed_page_size_v3 =
      vllm::byte_v2_v3_page_size_bytes(
          static_cast<int>(num_kv_heads),
          static_cast<int>(head_size / vllm::kByteV2TileSize),
          static_cast<int>(head_size_v / vllm::kByteV2TileSize));
  const int64_t raw_overlay_page_size =
      vllm::kByteV2PageHeaderBytes +
      std::max(compressed_payload_bytes, raw_block_bytes);
  STD_TORCH_CHECK((page_size_bytes == compressed_page_size ||
                   page_size_bytes == compressed_page_size_v3 ||
                   page_size_bytes == raw_overlay_page_size) &&
                      kv_cache.size(1) == page_size_bytes,
                  "Byte-v2 decompress page size mismatch");

  const int64_t total_tiles64 =
      num_kv_heads * (head_size / vllm::kByteV2TileSize +
                      head_size_v / vllm::kByteV2TileSize);
  const int64_t outlier_tile_bitmap_words64 = (total_tiles64 + 31) / 32;
  const bool has_sparse_fallback =
      fallback_pool.has_value() && fallback_block_ids.has_value();
  STD_TORCH_CHECK(
      has_sparse_fallback ||
          (!fallback_pool.has_value() && !fallback_block_ids.has_value() &&
           !fallback_tile_ids.has_value()),
      "Byte-v2 decompress sparse fallback arguments must be provided together");
  const bool has_tile_fallback =
      fallback_pool.has_value() && fallback_tile_ids.has_value();
  if (has_sparse_fallback) {
    STD_TORCH_CHECK(fallback_pool->device().is_cuda() &&
                        fallback_block_ids->device().is_cuda(),
                    "Byte-v2 decompress fallback tensors must be CUDA");
    STD_TORCH_CHECK(fallback_pool->device().index() ==
                            kv_cache.device().index() &&
                        fallback_block_ids->device().index() ==
                            kv_cache.device().index(),
                    "Byte-v2 decompress fallback tensors must be on the same "
                    "GPU");
    STD_TORCH_CHECK(fallback_pool->scalar_type() ==
                            torch::headeronly::ScalarType::Byte &&
                        fallback_block_ids->scalar_type() ==
                            torch::headeronly::ScalarType::Int,
                    "Byte-v2 decompress fallback tensors have invalid dtypes");
    STD_TORCH_CHECK(fallback_pool->is_contiguous() &&
                        fallback_block_ids->is_contiguous(),
                    "Byte-v2 decompress fallback tensors must be contiguous");
    STD_TORCH_CHECK(fallback_pool->dim() == 2 &&
                        fallback_pool->size(1) == raw_block_bytes,
                    "Byte-v2 fallback_pool shape is invalid");
    STD_TORCH_CHECK(fallback_block_ids->dim() == 1 &&
                        fallback_block_ids->size(0) == num_blocks64,
                    "Byte-v2 fallback_block_ids shape is invalid");
    if (has_tile_fallback) {
      STD_TORCH_CHECK(fallback_tile_ids->device().is_cuda() &&
                          fallback_tile_ids->device().index() ==
                              kv_cache.device().index(),
                      "Byte-v2 tile fallback ids must be on the same GPU");
      STD_TORCH_CHECK(fallback_tile_ids->scalar_type() ==
                          torch::headeronly::ScalarType::Int,
                      "Byte-v2 tile fallback ids must be int32");
      STD_TORCH_CHECK(fallback_tile_ids->is_contiguous(),
                      "Byte-v2 tile fallback ids must be contiguous");
      STD_TORCH_CHECK(fallback_tile_ids->dim() == 2 &&
                          fallback_tile_ids->size(0) == num_blocks64 &&
                          fallback_tile_ids->size(1) == total_tiles64,
                      "Byte-v2 fallback_tile_ids shape is invalid");
    }
  }

  const bool has_outlier_arena =
      outlier_arena.has_value() || outlier_tile_meta.has_value();
  STD_TORCH_CHECK(
      outlier_arena.has_value() == outlier_tile_meta.has_value(),
      "Byte-v2 decompress outlier arena arguments must be provided together");
  const bool has_outlier_tile_bitmap = outlier_tile_bitmap.has_value();
  STD_TORCH_CHECK(!has_outlier_tile_bitmap || has_outlier_arena,
                  "Byte-v2 outlier tile bitmap requires arena metadata");
  if (has_outlier_arena) {
    STD_TORCH_CHECK(outlier_arena->device().is_cuda() &&
                        outlier_tile_meta->device().is_cuda(),
                    "Byte-v2 outlier tensors must be CUDA");
    STD_TORCH_CHECK(outlier_arena->device().index() ==
                            kv_cache.device().index() &&
                        outlier_tile_meta->device().index() ==
                            kv_cache.device().index(),
                    "Byte-v2 outlier tensors must be on the same GPU");
    STD_TORCH_CHECK(outlier_arena->scalar_type() ==
                            torch::headeronly::ScalarType::Int &&
                        outlier_tile_meta->scalar_type() ==
                            torch::headeronly::ScalarType::Int,
                    "Byte-v2 outlier tensors must be int32");
    STD_TORCH_CHECK(outlier_arena->is_contiguous() &&
                        outlier_tile_meta->is_contiguous(),
                    "Byte-v2 outlier tensors must be contiguous");
    STD_TORCH_CHECK(outlier_arena->dim() == 1 &&
                        outlier_tile_meta->dim() == 2 &&
                        outlier_tile_meta->size(0) == num_blocks64 &&
                        outlier_tile_meta->size(1) == total_tiles64,
                    "Byte-v2 outlier tensor shapes are invalid");
  }
  if (has_outlier_tile_bitmap) {
    STD_TORCH_CHECK(outlier_tile_bitmap->device().is_cuda() &&
                        outlier_tile_bitmap->device().index() ==
                            kv_cache.device().index(),
                    "Byte-v2 outlier tile bitmap must be on the same GPU");
    STD_TORCH_CHECK(outlier_tile_bitmap->scalar_type() ==
                        torch::headeronly::ScalarType::Int,
                    "Byte-v2 outlier tile bitmap must be int32");
    STD_TORCH_CHECK(outlier_tile_bitmap->is_contiguous(),
                    "Byte-v2 outlier tile bitmap must be contiguous");
    STD_TORCH_CHECK(outlier_tile_bitmap->dim() == 2 &&
                        outlier_tile_bitmap->size(0) == num_blocks64 &&
                        outlier_tile_bitmap->size(1) ==
                            outlier_tile_bitmap_words64,
                    "Byte-v2 outlier tile bitmap shape is invalid");
  }

  if (num_blocks64 == 0) {
    return;
  }
  const torch::stable::accelerator::DeviceGuard device_guard(
      kv_cache.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream(kv_cache.get_device_index());
  const uint8_t* fallback_pool_ptr =
      has_sparse_fallback
          ? reinterpret_cast<const uint8_t*>(fallback_pool->const_data_ptr())
          : nullptr;
  const int32_t* fallback_block_ids_ptr =
      has_sparse_fallback ? reinterpret_cast<const int32_t*>(
                                fallback_block_ids->const_data_ptr())
                          : nullptr;
  const int32_t* fallback_tile_ids_ptr =
      has_tile_fallback ? reinterpret_cast<const int32_t*>(
                              fallback_tile_ids->const_data_ptr())
                        : nullptr;
  const int32_t* outlier_arena_ptr =
      has_outlier_arena ? reinterpret_cast<const int32_t*>(
                              outlier_arena->const_data_ptr())
                        : nullptr;
  const int32_t* outlier_tile_bitmap_ptr =
      has_outlier_tile_bitmap
          ? reinterpret_cast<const int32_t*>(
                outlier_tile_bitmap->const_data_ptr())
          : nullptr;
  const int32_t* outlier_tile_meta_ptr =
      has_outlier_arena ? reinterpret_cast<const int32_t*>(
                              outlier_tile_meta->const_data_ptr())
                        : nullptr;
  const dim3 grid(static_cast<unsigned int>(num_blocks64),
                  static_cast<unsigned int>(num_kv_heads), 2);
  constexpr int threads = 256;
  vllm::byte_v2_decompress_cache_to_bf16_kernel<<<grid, threads, 0, stream>>>(
      reinterpret_cast<const uint8_t*>(kv_cache.const_data_ptr()),
      reinterpret_cast<uint16_t*>(key_cache.mutable_data_ptr()),
      reinterpret_cast<uint16_t*>(value_cache.mutable_data_ptr()),
      fallback_pool_ptr, fallback_block_ids_ptr, fallback_tile_ids_ptr,
      outlier_arena_ptr, outlier_tile_bitmap_ptr, outlier_tile_meta_ptr,
      static_cast<int>(num_blocks64), static_cast<int>(page_size_bytes),
      static_cast<int>(raw_block_bytes), static_cast<int>(num_kv_heads),
      static_cast<int>(head_size), static_cast<int>(head_size_v));
  BYTE_V2_CUDA_CHECK(cudaGetLastError());
}

torch::stable::Tensor byte_v2_wmma_layout_microbench(
    torch::stable::Tensor& query, torch::stable::Tensor& key,
    torch::stable::Tensor& value, int64_t variant, int64_t repeat_count) {
  STD_TORCH_CHECK(query.device().is_cuda() && key.device().is_cuda() &&
                      value.device().is_cuda(),
                  "Byte-v2 WMMA microbench expects CUDA tensors");
  STD_TORCH_CHECK(query.device().index() == key.device().index() &&
                      query.device().index() == value.device().index(),
                  "Byte-v2 WMMA microbench tensors must be on the same GPU");
  STD_TORCH_CHECK(query.scalar_type() ==
                          torch::headeronly::ScalarType::BFloat16 &&
                      key.scalar_type() ==
                          torch::headeronly::ScalarType::BFloat16 &&
                      value.scalar_type() ==
                          torch::headeronly::ScalarType::BFloat16,
                  "Byte-v2 WMMA microbench expects BF16 tensors");
  STD_TORCH_CHECK(query.is_contiguous() && key.is_contiguous() &&
                      value.is_contiguous(),
                  "Byte-v2 WMMA microbench expects contiguous tensors");
  STD_TORCH_CHECK(query.dim() == 3 && key.dim() == 3 && value.dim() == 3,
                  "Byte-v2 WMMA microbench expects rank-3 tensors");
  STD_TORCH_CHECK(query.size(1) == vllm::kByteV2TileSize &&
                      key.size(1) == vllm::kByteV2TileSize &&
                      value.size(1) == vllm::kByteV2TileSize &&
                      query.size(2) == vllm::kByteV2MaxHeadSize &&
                      key.size(2) == vllm::kByteV2MaxHeadSize &&
                      value.size(2) == vllm::kByteV2MaxHeadSize,
                  "Byte-v2 WMMA microbench expects [num_tiles, 16, 128]");
  STD_TORCH_CHECK(query.size(0) == key.size(0) &&
                      query.size(0) == value.size(0),
                  "Byte-v2 WMMA microbench tensors must have matching "
                  "num_tiles");
  STD_TORCH_CHECK(variant == 0,
                  "Byte-v2 WMMA microbench currently supports variant=0");
  STD_TORCH_CHECK(repeat_count > 0 && repeat_count <= INT_MAX,
                  "Byte-v2 WMMA microbench repeat_count must fit in int32");
  STD_TORCH_CHECK(query.size(0) <= INT_MAX,
                  "Byte-v2 WMMA microbench num_tiles must fit in int32");

  bool device_supports_bf16_wmma = false;
#ifndef USE_ROCM
  device_supports_bf16_wmma =
      byte_v2_device_supports_bf16_wmma(query.get_device_index());
#endif
  STD_TORCH_CHECK(device_supports_bf16_wmma,
                  "Byte-v2 WMMA microbench requires Ampere or newer");

  auto output = torch::stable::empty(
      {query.size(0), query.size(1), query.size(2)}, query.scalar_type(),
      std::nullopt, query.device());
  if (query.size(0) == 0) {
    return output;
  }

  const torch::stable::accelerator::DeviceGuard device_guard(
      query.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream(query.get_device_index());
  constexpr int threads = 128;
  vllm::byte_v2_wmma_layout_microbench_kernel<<<query.size(0), threads, 0,
                                                stream>>>(
      reinterpret_cast<const uint16_t*>(query.const_data_ptr()),
      reinterpret_cast<const uint16_t*>(key.const_data_ptr()),
      reinterpret_cast<const uint16_t*>(value.const_data_ptr()),
      reinterpret_cast<uint16_t*>(output.mutable_data_ptr()),
      static_cast<int>(query.size(0)), static_cast<int>(repeat_count));
  BYTE_V2_CUDA_CHECK(cudaGetLastError());
  return output;
}

torch::stable::Tensor byte_v2_decode_page_wmma_microbench(
    torch::stable::Tensor& query, torch::stable::Tensor& kv_cache,
    int64_t num_kv_heads, int64_t kv_head, int64_t page_size_bytes,
    int64_t repeat_count) {
  STD_TORCH_CHECK(query.device().is_cuda() && kv_cache.device().is_cuda(),
                  "Byte-v2 decode page microbench expects CUDA tensors");
  STD_TORCH_CHECK(query.device().index() == kv_cache.device().index(),
                  "Byte-v2 decode page microbench tensors must be on the "
                  "same GPU");
  STD_TORCH_CHECK(query.scalar_type() ==
                      torch::headeronly::ScalarType::BFloat16,
                  "Byte-v2 decode page microbench expects BF16 query");
  STD_TORCH_CHECK(kv_cache.scalar_type() ==
                      torch::headeronly::ScalarType::Byte,
                  "Byte-v2 decode page microbench expects uint8 kv_cache");
  STD_TORCH_CHECK(query.is_contiguous() && kv_cache.is_contiguous(),
                  "Byte-v2 decode page microbench expects contiguous tensors");
  STD_TORCH_CHECK(query.dim() == 3 && kv_cache.dim() == 2,
                  "Byte-v2 decode page microbench expects query rank 3 and "
                  "kv_cache rank 2");
  STD_TORCH_CHECK(query.size(1) == vllm::kByteV2TileSize &&
                      query.size(2) == vllm::kByteV2MaxHeadSize,
                  "Byte-v2 decode page microbench expects query shape "
                  "[num_pages, 16, 128]");
  STD_TORCH_CHECK(query.size(0) == kv_cache.size(0),
                  "Byte-v2 decode page microbench expects one query tile per "
                  "KV page");
  STD_TORCH_CHECK(num_kv_heads > 0 && num_kv_heads <= INT_MAX &&
                      kv_head >= 0 && kv_head < num_kv_heads,
                  "Byte-v2 decode page microbench has invalid KV head");
  STD_TORCH_CHECK(repeat_count > 0 && repeat_count <= INT_MAX,
                  "Byte-v2 decode page microbench repeat_count must fit in "
                  "int32");
  STD_TORCH_CHECK(query.size(0) <= INT_MAX,
                  "Byte-v2 decode page microbench num_pages must fit in "
                  "int32");
  const int64_t compressed_payload_bytes =
      num_kv_heads * ((vllm::kByteV2MaxHeadSize / vllm::kByteV2TileSize) * 2) *
      vllm::kByteV2FastTilePayloadBytes;
  const int64_t compressed_page_size =
      vllm::kByteV2PageHeaderBytes + compressed_payload_bytes;
  constexpr int k_dim_tiles =
      vllm::kByteV2MaxHeadSize / vllm::kByteV2TileSize;
  constexpr int v_dim_tiles =
      vllm::kByteV2MaxHeadSize / vllm::kByteV2TileSize;
  const int64_t compressed_page_size_v3 =
      vllm::byte_v2_v3_page_size_bytes(static_cast<int>(num_kv_heads),
                                       k_dim_tiles, v_dim_tiles);
  STD_TORCH_CHECK((page_size_bytes == compressed_page_size ||
                   page_size_bytes == compressed_page_size_v3) &&
                      kv_cache.size(1) == page_size_bytes,
                  "Byte-v2 decode page microbench requires compressed-only "
                  "V1 or V3 page size for head_size=head_size_v=128");

  bool device_supports_bf16_wmma = false;
#ifndef USE_ROCM
  device_supports_bf16_wmma =
      byte_v2_device_supports_bf16_wmma(query.get_device_index());
#endif
  STD_TORCH_CHECK(device_supports_bf16_wmma,
                  "Byte-v2 decode page microbench requires Ampere or newer");

  auto output = torch::stable::empty(
      {query.size(0), query.size(1), query.size(2)}, query.scalar_type(),
      std::nullopt, query.device());
  if (query.size(0) == 0) {
    return output;
  }

  const torch::stable::accelerator::DeviceGuard device_guard(
      query.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream(query.get_device_index());
  bool use_v3_warp_stripe_load = false;
  if (const char* v3_warp_stripe_env =
          std::getenv("VLLM_BYTE_V2_DECODE_V3_WARP_STRIPE_LOAD")) {
    use_v3_warp_stripe_load = std::atoi(v3_warp_stripe_env) != 0;
  }
  bool use_v3_cp_async_stage = false;
  if (const char* v3_cp_async_env =
          std::getenv("VLLM_BYTE_V2_DECODE_V3_CP_ASYNC_STAGE")) {
    use_v3_cp_async_stage = std::atoi(v3_cp_async_env) != 0;
  }
  constexpr int threads = 128;
  vllm::byte_v2_decode_page_wmma_microbench_kernel<<<query.size(0), threads, 0,
                                                     stream>>>(
      reinterpret_cast<const uint16_t*>(query.const_data_ptr()),
      reinterpret_cast<const uint8_t*>(kv_cache.const_data_ptr()),
      reinterpret_cast<uint16_t*>(output.mutable_data_ptr()),
      static_cast<int>(query.size(0)), static_cast<int>(num_kv_heads),
      static_cast<int>(kv_head), static_cast<int>(page_size_bytes),
      static_cast<int>(repeat_count), use_v3_warp_stripe_load,
      use_v3_cp_async_stage);
  BYTE_V2_CUDA_CHECK(cudaGetLastError());
  return output;
}

// KV_T is the data type of key and value tensors.
// CACHE_T is the stored data type of kv-cache.
// KV_DTYPE is the real data type of kv-cache.
#define CALL_CONCAT_AND_CACHE_MLA(KV_T, CACHE_T, KV_DTYPE)                    \
  vllm::concat_and_cache_mla_kernel<KV_T, CACHE_T, KV_DTYPE>                  \
      <<<grid, block, 0, stream>>>(                                           \
          reinterpret_cast<KV_T*>(kv_c.data_ptr()),                           \
          reinterpret_cast<KV_T*>(k_pe.data_ptr()),                           \
          reinterpret_cast<CACHE_T*>(kv_cache.data_ptr()),                    \
          slot_mapping.const_data_ptr<int64_t>(), block_stride, entry_stride, \
          kv_c_stride, k_pe_stride, kv_lora_rank, pe_dim, block_size,         \
          reinterpret_cast<const float*>(scale.data_ptr()));

// KV_T is the data type of key and value tensors.
// CACHE_T is the stored data type of kv-cache.
#define CALL_CONCAT_AND_CACHE_DS_MLA(KV_T, CACHE_T, KV_DTYPE)                 \
  vllm::concat_and_cache_ds_mla_kernel<KV_T, CACHE_T, KV_DTYPE>               \
      <<<grid, block, 0, stream>>>(                                           \
          reinterpret_cast<KV_T*>(kv_c.data_ptr()),                           \
          reinterpret_cast<KV_T*>(k_pe.data_ptr()),                           \
          reinterpret_cast<CACHE_T*>(kv_cache.data_ptr()),                    \
          slot_mapping.const_data_ptr<int64_t>(), block_stride, entry_stride, \
          kv_c_stride, k_pe_stride, kv_lora_rank, pe_dim, block_size,         \
          reinterpret_cast<const float*>(scale.data_ptr()));

void concat_and_cache_mla(
    torch::stable::Tensor& kv_c,      // [num_tokens, kv_lora_rank]
    torch::stable::Tensor& k_pe,      // [num_tokens, pe_dim]
    torch::stable::Tensor& kv_cache,  // [num_blocks, block_size, (kv_lora_rank
                                      // + pe_dim)]
    torch::stable::Tensor& slot_mapping,  // [num_tokens] or [num_actual_tokens]
    const std::string& kv_cache_dtype, torch::stable::Tensor& scale) {
  // NOTE(woosuk): In vLLM V1, key.size(0) can be different from
  // slot_mapping.size(0) because of padding for CUDA graphs.
  // In vLLM V0, key.size(0) is always equal to slot_mapping.size(0) because
  // both include padding.
  // In vLLM V1, however, key.size(0) can be larger than slot_mapping.size(0)
  // since key includes padding for CUDA graphs, while slot_mapping does not.
  // In this case, slot_mapping.size(0) represents the actual number of tokens
  // before padding.
  // For compatibility with both cases, we use slot_mapping.size(0) as the
  // number of tokens.
  int num_tokens = slot_mapping.size(0);
  int kv_lora_rank = kv_c.size(1);
  int pe_dim = k_pe.size(1);
  int block_size = kv_cache.size(1);

  if (kv_cache_dtype == "fp8_ds_mla") {
    STD_TORCH_CHECK(kv_lora_rank == 512,
                    "kv_lora_rank must be 512 for fp8_ds_mla");
    STD_TORCH_CHECK(pe_dim == 64, "pe_dim must be 64 for fp8_ds_mla");
    STD_TORCH_CHECK(kv_cache.size(2) == 656 / kv_cache.element_size(),
                    "kv_cache.size(2) must be 656 bytes for fp8_ds_mla");
    STD_TORCH_CHECK(kv_c.element_size() == 2,
                    "kv_c.element_size() must be 2 for fp8_ds_mla");
    STD_TORCH_CHECK(k_pe.element_size() == 2,
                    "k_pe.element_size() must be 2 for fp8_ds_mla");
  } else {
    STD_TORCH_CHECK(kv_cache.size(2) == kv_lora_rank + pe_dim);
  }

  int kv_c_stride = kv_c.stride(0);
  int k_pe_stride = k_pe.stride(0);
  int block_stride = kv_cache.stride(0);
  int entry_stride = kv_cache.stride(1);

  const torch::stable::accelerator::DeviceGuard device_guard(
      kv_c.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream();

  if (kv_cache_dtype == "fp8_ds_mla") {
    dim3 grid(num_tokens);
    // For the NoPE part, each tile of 128 elements is handled by half of one
    // warp (16 threads). There are 4 total tiles, so 2 warps (64 threads).
    // Lanes 0 and 16 of each warp write the scale values for that warp's tiles.
    // The RoPE part (last 64 elements) is handled by another 1 warp (32
    // threads). So in total, we use 3 warps (96 threads) per block.
    dim3 block(96);
    DISPATCH_BY_KV_CACHE_DTYPE(kv_c.scalar_type(), kv_cache_dtype,
                               CALL_CONCAT_AND_CACHE_DS_MLA);
  } else {
    dim3 grid(num_tokens);
    dim3 block(std::min(kv_lora_rank, 512));
    DISPATCH_BY_KV_CACHE_DTYPE(kv_c.scalar_type(), kv_cache_dtype,
                               CALL_CONCAT_AND_CACHE_MLA);
  }
}

namespace vllm {

template <typename Tout, typename Tin, Fp8KVCacheDataType kv_dt>
__global__ void convert_fp8_kernel(const Tin* __restrict__ src_cache,
                                   Tout* __restrict__ dst_cache,
                                   const float scale,
                                   const int64_t block_stride) {
  const int64_t block_idx = blockIdx.x;
  for (int i = threadIdx.x; i < block_stride; i += blockDim.x) {
    int64_t idx = block_idx * block_stride + i;
    dst_cache[idx] =
        fp8::scaled_convert<Tout, Tin, kv_dt>(src_cache[idx], scale);
  }
}

}  // namespace vllm

#define CALL_CONVERT_FP8(Tout, Tin, KV_DTYPE)                                \
  vllm::convert_fp8_kernel<Tout, Tin, KV_DTYPE><<<grid, block, 0, stream>>>( \
      reinterpret_cast<Tin*>(src_cache.data_ptr()),                          \
      reinterpret_cast<Tout*>(dst_cache.data_ptr()), scale, block_stride);

// Only for testing.
void convert_fp8(torch::stable::Tensor& dst_cache,
                 torch::stable::Tensor& src_cache, const double scale,
                 const std::string& kv_cache_dtype) {
  torch::stable::Device src_device = src_cache.device();
  torch::stable::Device dst_device = dst_cache.device();
  STD_TORCH_CHECK(src_device.is_cuda(), "src must be on a GPU")
  STD_TORCH_CHECK(dst_device.is_cuda(), "dst must be on a GPU")
  STD_TORCH_CHECK(src_device.index() == dst_device.index(),
                  "src and dst must be on the same GPU");
  torch::stable::accelerator::DeviceGuard device_guard(src_device.index());

  int64_t num_blocks = src_cache.size(0);
  int64_t block_stride = src_cache.stride(0);

  dim3 grid(num_blocks);
  dim3 block(std::min(block_stride, int64_t(512)));
  const cudaStream_t stream = get_current_cuda_stream();

  if (kv_cache_dtype == "auto") {
    if (src_cache.scalar_type() == torch::headeronly::ScalarType::Float) {
      CALL_CONVERT_FP8(uint8_t, float, vllm::Fp8KVCacheDataType::kAuto);
    } else if (src_cache.scalar_type() == torch::headeronly::ScalarType::Half) {
      CALL_CONVERT_FP8(uint8_t, uint16_t, vllm::Fp8KVCacheDataType::kAuto);
    } else if (src_cache.scalar_type() ==
               torch::headeronly::ScalarType::BFloat16) {
      CALL_CONVERT_FP8(uint8_t, __nv_bfloat16, vllm::Fp8KVCacheDataType::kAuto);
    } else if (dst_cache.scalar_type() ==
               torch::headeronly::ScalarType::Float) {
      CALL_CONVERT_FP8(float, uint8_t, vllm::Fp8KVCacheDataType::kAuto);
    } else if (dst_cache.scalar_type() == torch::headeronly::ScalarType::Half) {
      CALL_CONVERT_FP8(uint16_t, uint8_t, vllm::Fp8KVCacheDataType::kAuto);
    } else if (dst_cache.scalar_type() ==
               torch::headeronly::ScalarType::BFloat16) {
      CALL_CONVERT_FP8(__nv_bfloat16, uint8_t, vllm::Fp8KVCacheDataType::kAuto);
    }
  } else if (kv_cache_dtype == "fp8" || kv_cache_dtype == "fp8_e4m3") {
    if (src_cache.scalar_type() == torch::headeronly::ScalarType::Float) {
      CALL_CONVERT_FP8(uint8_t, float, vllm::Fp8KVCacheDataType::kFp8E4M3);
    } else if (src_cache.scalar_type() == torch::headeronly::ScalarType::Half) {
      CALL_CONVERT_FP8(uint8_t, uint16_t, vllm::Fp8KVCacheDataType::kFp8E4M3);
    } else if (src_cache.scalar_type() ==
               torch::headeronly::ScalarType::BFloat16) {
      CALL_CONVERT_FP8(uint8_t, __nv_bfloat16,
                       vllm::Fp8KVCacheDataType::kFp8E4M3);
    } else if (dst_cache.scalar_type() ==
               torch::headeronly::ScalarType::Float) {
      CALL_CONVERT_FP8(float, uint8_t, vllm::Fp8KVCacheDataType::kFp8E4M3);
    } else if (dst_cache.scalar_type() == torch::headeronly::ScalarType::Half) {
      CALL_CONVERT_FP8(uint16_t, uint8_t, vllm::Fp8KVCacheDataType::kFp8E4M3);
    } else if (dst_cache.scalar_type() ==
               torch::headeronly::ScalarType::BFloat16) {
      CALL_CONVERT_FP8(__nv_bfloat16, uint8_t,
                       vllm::Fp8KVCacheDataType::kFp8E4M3);
    }
  } else {
    STD_TORCH_CHECK(false, "Unsupported data type: ", kv_cache_dtype);
  }
}

namespace vllm {

// grid is launched with dimensions (batch, num_splits)
template <typename scalar_t, typename cache_t, Fp8KVCacheDataType kv_dt,
          int ENTRY_SIZE, int CTA_SIZE>
__global__ void gather_and_maybe_dequant_cache(
    const cache_t* __restrict__ src_cache,     // [NUM_BLOCKS, BLOCK_SIZE,
                                               // ENTRIES...]
    scalar_t* __restrict__ dst,                // [TOT_TOKENS, ENTRIES...]
    const int32_t* __restrict__ block_table,   // [BATCH, BLOCK_INDICES]
    const int32_t* __restrict__ cu_seq_lens,   // [BATCH+1]
    const int32_t* __restrict__ token_to_seq,  // [MAX_TOKEN_ACROSS_CHUNK]
    const int32_t num_tokens, const int32_t block_size,
    const int64_t block_table_stride, const int64_t cache_block_stride,
    const int64_t cache_entry_stride, const int64_t dst_entry_stride,
    const float* __restrict__ scale,
    const int32_t* __restrict__ seq_starts) {  // Optional: starting offsets per
                                               // batch
  constexpr int vec_size = sizeof(float4) / sizeof(scalar_t);
  using ltype = vllm::vec_n_t<cache_t, vec_size>;
  using stype = vllm::vec_n_t<scalar_t, vec_size>;
  // We are adding this for code readability which will be optimized out when
  // build in release.
  assert(CTA_SIZE == blockDim.x);

#pragma unroll
  for (int token_id = blockIdx.x; token_id < num_tokens;
       token_id += gridDim.x) {
    int64_t batch_id = token_to_seq[token_id];
    int64_t batch_start = cu_seq_lens[batch_id];
    int64_t batch_end = cu_seq_lens[batch_id + 1];
    int32_t batch_offset = token_id - batch_start;

    if (token_id >= batch_end) return;
    int32_t offset = 0;
    if (seq_starts != nullptr) {
      offset = seq_starts[batch_id];
    }
    batch_offset += offset;
    int32_t block_table_id = batch_offset / block_size;
    int32_t slot_id = batch_offset % block_size;
    int32_t block_table_offset = batch_id * block_table_stride + block_table_id;
    int32_t block_id = block_table[block_table_offset];
    int64_t cache_offset =
        block_id * cache_block_stride + slot_id * cache_entry_stride;
    constexpr int32_t vec_iter_cnt = ENTRY_SIZE / vec_size;
    scalar_t* dst_ = dst + token_id * dst_entry_stride;
    cache_t* src_ = const_cast<cache_t*>(src_cache) + cache_offset;

#pragma unroll
    for (int idx = threadIdx.x; idx < vec_iter_cnt; idx += CTA_SIZE) {
      if constexpr (kv_dt == Fp8KVCacheDataType::kAuto) {
        reinterpret_cast<stype*>(dst_)[idx] =
            static_cast<stype>(reinterpret_cast<ltype*>(src_)[idx]);
      } else {
        ltype loaded_val = reinterpret_cast<ltype*>(src_)[idx];
        stype store_val;
#pragma unroll
        for (int j = 0; j < vec_size; ++j) {
          store_val.val[j] = fp8::scaled_convert<scalar_t, cache_t, kv_dt>(
              loaded_val.val[j], *scale);
        }
        reinterpret_cast<stype*>(dst_)[idx] = store_val;
      }
    }
    // process tail
    constexpr int32_t tail_cnt = ENTRY_SIZE % vec_size;
    dst_ = dst_ + ENTRY_SIZE - tail_cnt;
    src_ = src_ + ENTRY_SIZE - tail_cnt;
#pragma unroll
    for (int idx = threadIdx.x; idx < tail_cnt; idx += CTA_SIZE) {
      if constexpr (kv_dt == Fp8KVCacheDataType::kAuto) {
        dst_[idx] = static_cast<scalar_t>(src_[idx]);
      } else {
        dst_[idx] =
            fp8::scaled_convert<scalar_t, cache_t, kv_dt>(src_[idx], *scale);
      }
    }
  }
}

}  // namespace vllm

// Macro to dispatch the kernel based on the data type.
// SCALAR_T is the data type of the destination tensor.
// CACHE_T is the stored data type of kv-cache.
// KV_DTYPE is the real data type of kv-cache.
#define CALL_GATHER_CACHE(SCALAR_T, CACHE_T, KV_DTYPE, ENTRY_SZ)              \
  vllm::gather_and_maybe_dequant_cache<SCALAR_T, CACHE_T, KV_DTYPE, ENTRY_SZ, \
                                       thread_block_size>                     \
      <<<grid, block, 0, stream>>>(                                           \
          reinterpret_cast<CACHE_T*>(src_cache.data_ptr()),                   \
          reinterpret_cast<SCALAR_T*>(dst.data_ptr()),                        \
          block_table.const_data_ptr<int32_t>(),                              \
          cu_seq_lens.const_data_ptr<int32_t>(),                              \
          token_to_seq.const_data_ptr<int32_t>(), num_tokens, block_size,     \
          block_table_stride, cache_block_stride, cache_entry_stride,         \
          dst_entry_stride, reinterpret_cast<const float*>(scale.data_ptr()), \
          seq_starts_ptr);

#define CALL_GATHER_CACHE_576(SCALAR_T, CACHE_T, KV_DTYPE) \
  CALL_GATHER_CACHE(SCALAR_T, CACHE_T, KV_DTYPE, 576)

#define CALL_GATHER_CACHE_320(SCALAR_T, CACHE_T, KV_DTYPE) \
  CALL_GATHER_CACHE(SCALAR_T, CACHE_T, KV_DTYPE, 320)

// Gather sequences from the cache into the destination tensor.
//  - cu_seq_lens contains the cumulative sequence lengths for each batch
//  - block_table contains the cache block indices for each sequence
//  - token_to_seq contains the back mapping from token_id to batch_id
//  - Optionally, seq_starts (if provided) offsets the starting block index by
//  (seq_starts[bid] / page_size)
void gather_and_maybe_dequant_cache(
    torch::stable::Tensor const&
        src_cache,                     // [NUM_BLOCKS, BLOCK_SIZE, ENTRIES...]
    torch::stable::Tensor const& dst,  // [TOT_TOKENS, ENTRIES...]
    torch::stable::Tensor const& block_table,   // [BATCH, BLOCK_INDICES]
    torch::stable::Tensor const& cu_seq_lens,   // [BATCH+1]
    torch::stable::Tensor const& token_to_seq,  // [MAX_TOKEN_ACROSS_CHUNKS]
    int64_t num_tokens, const std::string& kv_cache_dtype,
    torch::stable::Tensor const& scale,
    std::optional<torch::stable::Tensor> seq_starts = std::nullopt) {
  torch::stable::accelerator::DeviceGuard device_guard(
      src_cache.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream();

  int32_t block_size = src_cache.size(1);
  int32_t head_dim = dst.size(-1);

  STD_TORCH_CHECK(
      block_table.scalar_type() == torch::headeronly::ScalarType::Int,
      "block_table must be int32");
  STD_TORCH_CHECK(
      cu_seq_lens.scalar_type() == torch::headeronly::ScalarType::Int,
      "cu_seq_lens must be int32");
  if (seq_starts.has_value()) {
    STD_TORCH_CHECK(
        seq_starts.value().scalar_type() == torch::headeronly::ScalarType::Int,
        "seq_starts must be int32");
  }
  STD_TORCH_CHECK(
      head_dim == 320 || head_dim == 576,
      "gather_and_maybe_dequant_cache only support the head_dim to 320 or 576 "
      "for better performance")

  STD_TORCH_CHECK(src_cache.device() == dst.device(),
                  "src_cache and dst must be on the same device");
  STD_TORCH_CHECK(src_cache.device() == block_table.device(),
                  "src_cache and block_table must be on the same device");
  STD_TORCH_CHECK(src_cache.device() == cu_seq_lens.device(),
                  "src_cache and cu_seq_lens must be on the same device");
  if (seq_starts.has_value()) {
    STD_TORCH_CHECK(src_cache.device() == seq_starts.value().device(),
                    "src_cache and seq_starts must be on the same device");
  }

  int64_t block_table_stride = block_table.stride(0);
  int64_t cache_block_stride = src_cache.stride(0);
  int64_t cache_entry_stride = src_cache.stride(1);
  int64_t dst_entry_stride = dst.stride(0);

  constexpr int32_t thread_block_size = 64;
  dim3 grid(num_tokens);
  dim3 block(thread_block_size);

  const int32_t* seq_starts_ptr =
      seq_starts.has_value() ? seq_starts.value().const_data_ptr<int32_t>()
                             : nullptr;

  if (head_dim == 576) {
    DISPATCH_BY_KV_CACHE_DTYPE(dst.scalar_type(), kv_cache_dtype,
                               CALL_GATHER_CACHE_576);
  } else {
    DISPATCH_BY_KV_CACHE_DTYPE(dst.scalar_type(), kv_cache_dtype,
                               CALL_GATHER_CACHE_320);
  }
}

namespace vllm {

// Gather and upconvert FP8 KV cache tokens to BF16 workspace
// Similar to cp_gather_cache but specifically for FP8->BF16 conversion
__global__ void cp_gather_and_upconvert_fp8_kv_cache(
    const uint8_t* __restrict__ src_cache,    // [NUM_BLOCKS, BLOCK_SIZE, 656]
    __nv_bfloat16* __restrict__ dst,          // [total_tokens, 576]
    const int32_t* __restrict__ block_table,  // [num_reqs, BLOCK_INDICES]
    const int32_t* __restrict__ workspace_starts,  // [num_reqs]
    const int32_t num_reqs, const int32_t block_size,
    const int32_t total_tokens, const int64_t block_table_stride,
    const int64_t cache_block_stride, const int64_t cache_entry_stride,
    const int64_t dst_entry_stride) {
  const int flat_warp_id = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  if (flat_warp_id >= total_tokens) return;
  const int lane_id = threadIdx.x & 31;

  // Binary search to find which request owns this output token
  int lo = 0, hi = num_reqs - 1;
  while (lo < hi) {
    int mid = (lo + hi + 1) >> 1;
    if (workspace_starts[mid] <= flat_warp_id)
      lo = mid;
    else
      hi = mid - 1;
  }
  const int req_id = lo;

  // Compute physical token address via block table
  const int out_token_id = flat_warp_id;
  const int token_offset = out_token_id - workspace_starts[req_id];
  const int cache_block_idx = token_offset / block_size;
  const int offset_in_block = token_offset % block_size;
  const int physical_block =
      block_table[req_id * block_table_stride + cache_block_idx];

  const uint8_t* token_ptr = src_cache + physical_block * cache_block_stride +
                             offset_in_block * cache_entry_stride;

  const int4* nope_src = reinterpret_cast<const int4*>(token_ptr);
  const int4 fp8_data = nope_src[lane_id];

  const float* scales_ptr = reinterpret_cast<const float*>(token_ptr + 512);
  const float scale = scales_ptr[lane_id >> 3];

  const uint2 fp8_lo = make_uint2(fp8_data.x, fp8_data.y);
  const uint2 fp8_hi = make_uint2(fp8_data.z, fp8_data.w);
#ifdef USE_ROCM
  const bf16_8_t bf16_lo =
      fp8::scaled_vec_conversion<bf16_8_t, uint2>(fp8_lo, scale);
  const bf16_8_t bf16_hi =
      fp8::scaled_vec_conversion<bf16_8_t, uint2>(fp8_hi, scale);
#else
  const bf16_8_t bf16_lo =
      fp8::scaled_vec_conversion<bf16_8_t, uint2>(fp8_lo, scale, __NV_E4M3);
  const bf16_8_t bf16_hi =
      fp8::scaled_vec_conversion<bf16_8_t, uint2>(fp8_hi, scale, __NV_E4M3);
#endif

  __nv_bfloat16* dst_ptr = dst + out_token_id * dst_entry_stride;
  int4* nope_dst = reinterpret_cast<int4*>(dst_ptr) + lane_id * 2;
  nope_dst[0] = *reinterpret_cast<const int4*>(&bf16_lo);
  nope_dst[1] = *reinterpret_cast<const int4*>(&bf16_hi);

  const int* rope_src = reinterpret_cast<const int*>(token_ptr + 528);
  int* rope_dst = reinterpret_cast<int*>(dst_ptr + 512);
  rope_dst[lane_id] = rope_src[lane_id];
}

template <typename scalar_t>
// Note(hc): The cp_gather_cache allows seq_starts to no longer be divisible by
// block_size.
__global__ void cp_gather_cache(
    const scalar_t* __restrict__ src_cache,   // [NUM_BLOCKS, BLOCK_SIZE,
                                              // ENTRY_SIZE]
    scalar_t* __restrict__ dst,               // [TOT_TOKENS, ENTRY_SIZE]
    const int32_t* __restrict__ block_table,  // [BATCH, BLOCK_INDICES]
    const int32_t* __restrict__ cu_seq_lens,  // [BATCH+1]
    const int32_t block_size, const int32_t entry_size,
    const int64_t block_table_stride, const int64_t cache_block_stride,
    const int64_t cache_entry_stride, const int64_t dst_entry_stride,
    const int32_t* __restrict__ seq_starts  // Optional: starting offsets per
                                            // batch
) {
  const int64_t bid = blockIdx.x;  // Batch ID
  const int32_t num_splits = gridDim.y;
  const int32_t split = blockIdx.y;
  const int32_t seq_start = cu_seq_lens[bid];
  const int32_t seq_end = cu_seq_lens[bid + 1];
  const int32_t seq_len = seq_end - seq_start;
  const int32_t tot_slots = seq_len;
  const int32_t split_slots = cuda_utils::ceil_div(tot_slots, num_splits);

  const int32_t split_start = split * split_slots;
  const int32_t split_end = min((split + 1) * split_slots, tot_slots);

  const bool is_active_split = (split_start < tot_slots);

  if (!is_active_split) return;

  // Adjust the pointer for the block_table for this batch.
  // If seq_starts is provided, compute an offset based on it
  const int32_t batch_offset = bid * block_table_stride;
  int32_t offset = split_start;
  if (seq_starts != nullptr) {
    offset += seq_starts[bid];
  }
  int32_t offset_div = offset / block_size;
  offset = offset % block_size;
  const int32_t* batch_block_table = block_table + batch_offset;

  // Adjust dst pointer based on the cumulative sequence lengths.
  dst += seq_start * dst_entry_stride;

  auto copy_entry = [&](const scalar_t* __restrict__ _src,
                        scalar_t* __restrict__ _dst) {
    for (int i = threadIdx.x; i < entry_size; i += blockDim.x)
      _dst[i] = _src[i];
  };

  for (int pid = split_start; pid < split_end; ++pid) {
    auto block_id = batch_block_table[offset_div];
    auto block_start_ptr = src_cache + block_id * cache_block_stride;
    auto block_dst_ptr = dst + pid * dst_entry_stride;
    copy_entry(block_start_ptr + offset * cache_entry_stride, block_dst_ptr);
    offset += 1;
    // bump to next block
    if (offset == block_size) {
      offset_div += 1;
      offset = 0;
    }
  }
}
}  // namespace vllm

// Macro to dispatch the kernel based on the data type.
#define CALL_CP_GATHER_CACHE(CPY_DTYPE)                              \
  vllm::cp_gather_cache<CPY_DTYPE><<<grid, block, 0, stream>>>(      \
      reinterpret_cast<CPY_DTYPE*>(src_cache.data_ptr()),            \
      reinterpret_cast<CPY_DTYPE*>(dst.data_ptr()),                  \
      block_table.const_data_ptr<int32_t>(),                         \
      cu_seq_lens.const_data_ptr<int32_t>(), block_size, entry_size, \
      block_table_stride, cache_block_stride, cache_entry_stride,    \
      dst_entry_stride, seq_starts_ptr);

// Gather sequences from the cache into the destination tensor.
//  - cu_seq_lens contains the cumulative sequence lengths for each batch
//  - block_table contains the cache block indices for each sequence
//  - Optionally, seq_starts (if provided) offsets the starting slot index by
//  seq_starts[bid]
void cp_gather_cache(
    torch::stable::Tensor const&
        src_cache,                     // [NUM_BLOCKS, BLOCK_SIZE, ENTRIES...]
    torch::stable::Tensor const& dst,  // [TOT_TOKENS, ENTRIES...]
    torch::stable::Tensor const& block_table,  // [BATCH, BLOCK_INDICES]
    torch::stable::Tensor const& cu_seq_lens,  // [BATCH+1]
    int64_t batch_size,
    std::optional<torch::stable::Tensor> seq_starts = std::nullopt) {
  torch::stable::accelerator::DeviceGuard device_guard(
      src_cache.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream();

  int32_t block_size = src_cache.size(1);
  int32_t entry_size = torch::stable::flatten(src_cache, 2, -1).size(2);

  STD_TORCH_CHECK(
      block_table.scalar_type() == torch::headeronly::ScalarType::Int,
      "block_table must be int32");
  STD_TORCH_CHECK(
      cu_seq_lens.scalar_type() == torch::headeronly::ScalarType::Int,
      "cu_seq_lens must be int32");
  if (seq_starts.has_value()) {
    STD_TORCH_CHECK(
        seq_starts.value().scalar_type() == torch::headeronly::ScalarType::Int,
        "seq_starts must be int32");
  }

  STD_TORCH_CHECK(src_cache.device() == dst.device(),
                  "src_cache and dst must be on the same device");
  STD_TORCH_CHECK(src_cache.device() == block_table.device(),
                  "src_cache and block_table must be on the same device");
  STD_TORCH_CHECK(src_cache.device() == cu_seq_lens.device(),
                  "src_cache and cu_seq_lens must be on the same device");
  if (seq_starts.has_value()) {
    STD_TORCH_CHECK(src_cache.device() == seq_starts.value().device(),
                    "src_cache and seq_starts must be on the same device");
  }

  int64_t block_table_stride = block_table.stride(0);
  int64_t cache_block_stride = src_cache.stride(0);
  int64_t cache_entry_stride = src_cache.stride(1);
  int64_t dst_entry_stride = dst.stride(0);

  // Decide on the number of splits based on the batch size.
  int num_splits = batch_size > 128 ? 2 : batch_size > 64 ? 4 : 16;
  dim3 grid(batch_size, num_splits);
  dim3 block(1024);

  STD_TORCH_CHECK(src_cache.scalar_type() == dst.scalar_type(),
                  "src_cache and dst must have the same dtype");

  const int dtype_bits = src_cache.element_size() * 8;
  const int32_t* seq_starts_ptr =
      seq_starts.has_value() ? seq_starts.value().const_data_ptr<int32_t>()
                             : nullptr;

  if (dtype_bits == 32) {
    CALL_CP_GATHER_CACHE(uint32_t);
  } else if (dtype_bits == 16) {
    CALL_CP_GATHER_CACHE(uint16_t);
  } else if (dtype_bits == 8) {
    CALL_CP_GATHER_CACHE(uint8_t);
  } else {
    STD_TORCH_CHECK(false, "Unsupported data type width: ", dtype_bits);
  }
}

void cp_gather_and_upconvert_fp8_kv_cache(
    torch::stable::Tensor const& src_cache,    // [NUM_BLOCKS, BLOCK_SIZE, 656]
    torch::stable::Tensor const& dst,          // [TOT_TOKENS, 576]
    torch::stable::Tensor const& block_table,  // [BATCH, BLOCK_INDICES]
    torch::stable::Tensor const& seq_lens,     // [BATCH]
    torch::stable::Tensor const& workspace_starts,  // [BATCH]
    int64_t batch_size) {
  torch::stable::accelerator::DeviceGuard device_guard(
      src_cache.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream();

  int32_t block_size = src_cache.size(1);
  int32_t head_dim = dst.size(1);

  STD_TORCH_CHECK(
      block_table.scalar_type() == torch::headeronly::ScalarType::Int,
      "block_table must be int32");
  STD_TORCH_CHECK(seq_lens.scalar_type() == torch::headeronly::ScalarType::Int,
                  "seq_lens must be int32");
  STD_TORCH_CHECK(
      workspace_starts.scalar_type() == torch::headeronly::ScalarType::Int,
      "workspace_starts must be int32");

  STD_TORCH_CHECK(src_cache.device() == dst.device(),
                  "src_cache and dst must be on the same device");
  STD_TORCH_CHECK(src_cache.device() == block_table.device(),
                  "src_cache and block_table must be on the same device");
  STD_TORCH_CHECK(src_cache.device() == seq_lens.device(),
                  "src_cache and seq_lens must be on the same device");
  STD_TORCH_CHECK(src_cache.device() == workspace_starts.device(),
                  "src_cache and workspace_starts must be on the same device");
  auto dtype = src_cache.scalar_type();
  STD_TORCH_CHECK(
      dtype == torch::headeronly::ScalarType::Byte ||               // uint8
          dtype == torch::headeronly::ScalarType::Float8_e4m3fn ||  // fp8 e4m3
          dtype == torch::headeronly::ScalarType::Float8_e5m2,      // fp8 e5m2
      "src_cache must be uint8, float8_e4m3fn, or float8_e5m2, but got ",
      src_cache.scalar_type());
  STD_TORCH_CHECK(dst.scalar_type() == torch::headeronly::ScalarType::BFloat16,
                  "dst must be bfloat16");
  STD_TORCH_CHECK(head_dim == 576, "head_dim must be 576 for MLA");

  int64_t block_table_stride = block_table.stride(0);
  int64_t cache_block_stride = src_cache.stride(0);
  int64_t cache_entry_stride = src_cache.stride(1);
  int64_t dst_entry_stride = dst.stride(0);

  const uint8_t* src_ptr = nullptr;
  if (dtype == torch::headeronly::ScalarType::Byte) {
    src_ptr = src_cache.const_data_ptr<uint8_t>();
  } else {
    // float8_e4m3fn or float8_e5m2
    src_ptr = reinterpret_cast<const uint8_t*>(src_cache.data_ptr());
  }

  const int total_tokens = dst.size(0);
  constexpr int warps_per_block = 8;
  const int grid_size = (total_tokens + warps_per_block - 1) / warps_per_block;
  const int block_size_threads = warps_per_block * 32;  // 256 threads

  vllm::cp_gather_and_upconvert_fp8_kv_cache<<<grid_size, block_size_threads, 0,
                                               stream>>>(
      src_ptr, reinterpret_cast<__nv_bfloat16*>(dst.data_ptr()),
      block_table.const_data_ptr<int32_t>(),
      workspace_starts.const_data_ptr<int32_t>(),
      static_cast<int32_t>(batch_size), block_size, total_tokens,
      block_table_stride, cache_block_stride, cache_entry_stride,
      dst_entry_stride);
}

// Macro to dispatch the kernel based on the data type.
#define CALL_INDEXER_K_QUANT_AND_CACHE(KV_T, CACHE_T, KV_DTYPE)               \
  vllm::indexer_k_quant_and_cache_kernel<KV_T, CACHE_T, KV_DTYPE>             \
      <<<grid, block, 0, stream>>>(                                           \
          reinterpret_cast<KV_T*>(k.data_ptr()),                              \
          reinterpret_cast<CACHE_T*>(kv_cache.data_ptr()),                    \
          slot_mapping.const_data_ptr<int64_t>(), head_dim, quant_block_size, \
          cache_block_size, cache_stride, use_ue8m0);

void indexer_k_quant_and_cache(
    torch::stable::Tensor& k,         // [num_tokens, head_dim]
    torch::stable::Tensor& kv_cache,  // [num_blocks, block_size, cache_stride]
    torch::stable::Tensor& slot_mapping,  // [num_tokens]
    int64_t quant_block_size,             // quantization block size
    const std::string& scale_fmt) {
  int num_tokens = k.size(0);
  int head_dim = k.size(1);
  int cache_block_size = kv_cache.size(1);
  int cache_stride = kv_cache.size(2);
  bool use_ue8m0 = scale_fmt == "ue8m0";

  STD_TORCH_CHECK(k.device() == kv_cache.device(),
                  "k and kv_cache must be on the same device");
  STD_TORCH_CHECK(k.device() == slot_mapping.device(),
                  "k and slot_mapping must be on the same device");
  STD_TORCH_CHECK(head_dim % quant_block_size == 0,
                  "head_dim must be divisible by quant_block_size");

  constexpr int vec_size = 4;
  dim3 grid(num_tokens, (head_dim + quant_block_size * vec_size - 1) /
                            (quant_block_size * vec_size));
  dim3 block(32, vec_size);
  const torch::stable::accelerator::DeviceGuard device_guard(
      k.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream();

  static const std::string kv_cache_dtype = "fp8_e4m3";
  DISPATCH_BY_KV_CACHE_DTYPE(k.scalar_type(), kv_cache_dtype,
                             CALL_INDEXER_K_QUANT_AND_CACHE);
}

// Macro to dispatch the kernel based on the data amount.
#define CALL_CP_GATHER_INDEXER_K_QUANT_CACHE(BLOCK_Y_SIZE)                    \
  vllm::cp_gather_indexer_k_quant_cache_kernel<BLOCK_Y_SIZE>                  \
      <<<dim3((num_tokens + BLOCK_Y_SIZE - 1) / BLOCK_Y_SIZE,                 \
              (head_dim + 8 * vec_size - 1) / (8 * vec_size)),                \
         dim3(8, BLOCK_Y_SIZE), 0, stream>>>(                                 \
          reinterpret_cast<char*>(kv_cache.data_ptr()),                       \
          reinterpret_cast<char*>(dst_k.data_ptr()),                          \
          reinterpret_cast<char*>(dst_scale.data_ptr()),                      \
          block_table.const_data_ptr<int32_t>(),                              \
          cu_seq_lens.const_data_ptr<int32_t>(), batch_size, dst_k.stride(0), \
          dst_k.size(1), kv_cache.stride(0), kv_cache.stride(1),              \
          kv_cache.size(1), block_table.size(1), num_tokens,                  \
          quant_block_size);

void cp_gather_indexer_k_quant_cache(
    const torch::stable::Tensor&
        kv_cache,                  // [num_blocks, block_size, cache_stride]
    torch::stable::Tensor& dst_k,  // [num_tokens, head_dim]
    torch::stable::Tensor&
        dst_scale,  // [num_tokens, head_dim / quant_block_size * 4]
    const torch::stable::Tensor& block_table,  // [batch_size, num_blocks]
    const torch::stable::Tensor& cu_seq_lens   // [batch_size + 1]
) {
  int batch_size = block_table.size(0);
  int num_tokens = dst_k.size(0);
  int head_dim = dst_k.size(1);
  int quant_block_size = head_dim * 4 / dst_scale.size(1);

  STD_TORCH_CHECK(kv_cache.device() == dst_k.device(),
                  "kv_cache and dst_k must be on the same device");
  STD_TORCH_CHECK(kv_cache.device() == dst_scale.device(),
                  "kv_cache and dst_scale must be on the same device");
  STD_TORCH_CHECK(kv_cache.device() == block_table.device(),
                  "kv_cache and block_table must be on the same device");
  STD_TORCH_CHECK(kv_cache.device() == cu_seq_lens.device(),
                  "kv_cache and cu_seq_lens must be on the same device");
  STD_TORCH_CHECK(head_dim % quant_block_size == 0,
                  "head_dim must be divisible by quant_block_size");

  constexpr int vec_size = 16;
  const torch::stable::accelerator::DeviceGuard device_guard(
      kv_cache.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream();

  if (num_tokens < 32) {
    CALL_CP_GATHER_INDEXER_K_QUANT_CACHE(1);
  } else if (num_tokens < 64) {
    CALL_CP_GATHER_INDEXER_K_QUANT_CACHE(2);
  } else if (num_tokens < 128) {
    CALL_CP_GATHER_INDEXER_K_QUANT_CACHE(4);
  } else if (num_tokens < 256) {
    CALL_CP_GATHER_INDEXER_K_QUANT_CACHE(8);
  } else if (num_tokens < 512) {
    CALL_CP_GATHER_INDEXER_K_QUANT_CACHE(16);
  } else {
    CALL_CP_GATHER_INDEXER_K_QUANT_CACHE(32);
  }
}

// Concatenate ql_nope and q_pe into a contiguous q_out tensor for MLA/DSA.
// Replaces torch.cat((ql_nope, q_pe), dim=-1).
void concat_mla_q(
    torch::stable::Tensor& ql_nope,  // [num_tokens, num_heads, nope_dim]
    torch::stable::Tensor& q_pe,     // [num_tokens, num_heads, rope_dim]
    torch::stable::Tensor& q_out     // [num_tokens, num_heads, nope_dim +
                                     // rope_dim]
) {
  const int num_tokens = ql_nope.size(0);
  const int num_heads = ql_nope.size(1);
  const int nope_dim = ql_nope.size(2);
  const int rope_dim = q_pe.size(2);

  STD_TORCH_CHECK(nope_dim % 512 == 0,
                  "nope_dim must be a multiple of 512, got ", nope_dim);
  STD_TORCH_CHECK(rope_dim == 64, "rope_dim must be 64, got ", rope_dim);
  STD_TORCH_CHECK(q_out.size(2) == nope_dim + rope_dim);

  STD_TORCH_CHECK(ql_nope.stride(2) == 1,
                  "ql_nope must have stride 1 in dim 2");
  STD_TORCH_CHECK(q_pe.stride(2) == 1, "q_pe must have stride 1 in dim 2");
  STD_TORCH_CHECK(q_out.stride(2) == 1, "q_out must have stride 1 in dim 2");
  STD_TORCH_CHECK(
      ql_nope.scalar_type() == torch::headeronly::ScalarType::Half ||
          ql_nope.scalar_type() == torch::headeronly::ScalarType::BFloat16,
      "ql_nope must be float16 or bfloat16 dtype");

  if (num_tokens == 0) return;

  constexpr int warps_per_block = 8;
  const int total_warps = num_tokens * num_heads;
  const int grid_size = (total_warps + warps_per_block - 1) / warps_per_block;
  const int block_size = warps_per_block * 32;

  const torch::stable::accelerator::DeviceGuard device_guard(
      ql_nope.get_device_index());
  const cudaStream_t stream = get_current_cuda_stream();

  VLLM_STABLE_DISPATCH_HALF_TYPES(ql_nope.scalar_type(), "concat_mla_q", [&] {
    vllm::ConcatMLAQKernel<scalar_t, 512><<<grid_size, block_size, 0, stream>>>(
        q_out.mutable_data_ptr<scalar_t>(), ql_nope.const_data_ptr<scalar_t>(),
        q_pe.const_data_ptr<scalar_t>(), num_tokens, num_heads, q_out.stride(0),
        q_out.stride(1), ql_nope.stride(0), ql_nope.stride(1), q_pe.stride(0),
        q_pe.stride(1));
  });
}
