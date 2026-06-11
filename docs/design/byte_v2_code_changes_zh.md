# ByteV2 vLLM 主要代码改动阅读文档

本文档基于当前工作区的 `git status` / `git diff` 结果，整理 ByteV2
集成到 vLLM 后保留下来的主要代码改动。它的目标不是重新描述算法设计，而是给后续
阅读、review、继续优化代码时提供一份代码地图。

相关设计和实验记录见：

- `docs/design/byte_v2_kv_cache_zh.md`
- `docs/design/byte_v2_bottleneck_resolution_zh.md`
- `docs/design/byte_v2_kernel_optimization_experiments_zh.md`
- `docs/design/byte_v2_outperform_raw_zh.md`

## 1. 快速阅读顺序

建议按下面顺序读代码：

1. `vllm/config/cache.py`、`vllm/envs.py`、`vllm/platforms/cuda.py`、
   `vllm/v1/attention/backends/registry.py`：看 ByteV2 如何作为一种
   `kv_cache_dtype` 和 attention backend 被选中。
2. `vllm/v1/kv_cache_interface.py` 中的 `ByteV2FullAttentionSpec`：看
   ByteV2 page、raw fallback pool、metadata 如何计入 KV cache 分配。
3. `vllm/v1/core/kv_cache_utils.py`、`vllm/v1/core/single_type_kv_cache_manager.py`、
   `vllm/v1/worker/gpu_model_runner.py`：看 allocator 如何为 compressed page
   和 sparse fallback pool 分配并切分内存。
4. `vllm/v1/attention/backends/byte_v2_codec.py` 和
   `vllm/v1/attention/backends/byte_v2_layout.py`：看 ByteV2 tile/page 的格式。
5. `vllm/v1/attention/backends/byte_v2_attn.py`：看 vLLM attention backend 主路径，
   包括 metadata、KV cache update、prefill、decode、prefix cache 行为。
6. `vllm/_custom_ops.py`、`csrc/libtorch_stable/ops.h`、
   `csrc/libtorch_stable/torch_bindings.cpp`、`csrc/libtorch_stable/cache_kernels.cu`：
   看 Python op 到 native CUDA kernel 的桥接。
7. `tests/v1/attention/test_byte_v2_*.py`、`benchmarks/benchmark_byte_v2_decode_e2e.py`、
   `benchmarks/kernels/benchmark_byte_v2_*.py`：看 correctness、E2E、kernel microbench
   如何验证。

## 2. 当前改动范围

`git diff --stat` 中 tracked 代码改动主要集中在 18 个文件，约 6k 行新增。
新增文件主要是 ByteV2 backend、测试、benchmark 和设计文档。

### 2.1 tracked 集成改动

- `csrc/libtorch_stable/cache_kernels.cu`
- `csrc/libtorch_stable/ops.h`
- `csrc/libtorch_stable/torch_bindings.cpp`
- `vllm/_custom_ops.py`
- `vllm/config/cache.py`
- `vllm/envs.py`
- `vllm/model_executor/layers/attention/attention.py`
- `vllm/platforms/cuda.py`
- `vllm/utils/torch_utils.py`
- `vllm/v1/attention/backends/registry.py`
- `vllm/v1/core/kv_cache_utils.py`
- `vllm/v1/core/single_type_kv_cache_manager.py`
- `vllm/v1/kv_cache_interface.py`
- `vllm/v1/worker/gpu/attn_utils.py`
- `vllm/v1/worker/gpu/model_runner.py`
- `vllm/v1/worker/gpu_model_runner.py`
- `vllm/v1/worker/worker_base.py`
- `tests/v1/test_kv_cache_spec_registry.py`

### 2.2 新增 ByteV2 backend 文件

- `vllm/v1/attention/backends/byte_v2_attn.py`
- `vllm/v1/attention/backends/byte_v2_codec.py`
- `vllm/v1/attention/backends/byte_v2_decode.py`
- `vllm/v1/attention/backends/byte_v2_layout.py`
- `vllm/v1/attention/backends/byte_v2_ops.py`
- `vllm/v1/attention/backends/byte_v2_torch.py`

### 2.3 新增测试和 benchmark

- `tests/config/test_cache_config.py`
- `tests/v1/test_byte_v2_kv_cache_spec.py`
- `tests/v1/attention/test_byte_v2_backend.py`
- `tests/v1/attention/test_byte_v2_codec.py`
- `tests/v1/attention/test_byte_v2_decode.py`
- `tests/v1/attention/test_byte_v2_e2e.py`
- `tests/v1/attention/test_byte_v2_layout.py`
- `tests/v1/attention/test_byte_v2_metadata.py`
- `tests/v1/attention/test_byte_v2_ops.py`
- `tests/v1/attention/test_byte_v2_wmma_microbench.py`
- `benchmarks/benchmark_byte_v2_decode_e2e.py`
- `benchmarks/byte_v2_tile_fallback_stats.py`
- `benchmarks/kernels/benchmark_byte_v2_cache_update_kernel.py`
- `benchmarks/kernels/benchmark_byte_v2_decode_kernel.py`
- `benchmarks/kernels/benchmark_byte_v2_decode_page_wmma_microbench.py`
- `benchmarks/kernels/benchmark_byte_v2_kv_read_microbench.py`
- `benchmarks/kernels/benchmark_byte_v2_wmma_microbench.py`

## 3. 用户入口和配置

### 3.1 `kv_cache_dtype=byte_v2`

`vllm/config/cache.py` 将 `CacheDType` 扩展为支持 `"byte_v2"`。配置校验中把
ByteV2 标记为 experimental compressed KV format。

`vllm/utils/torch_utils.py` 做了两件事：

- `STR_DTYPE_TO_TORCH_DTYPE["byte_v2"] = torch.uint8`，因为 ByteV2 page 以
  byte payload 形式存储。
- 新增 `is_byte_v2_kv_cache()`，并让 `is_compressed_kv_cache()` 把 ByteV2
  视为 compressed KV cache，但不把它视为普通 quantized KV cache。

### 3.2 backend 选择

`vllm/platforms/cuda.py` 在 `kv_cache_dtype` 是 `byte_v2` 时返回
`AttentionBackendEnum.BYTE_V2`。

`vllm/v1/attention/backends/registry.py` 注册：

```python
BYTE_V2 = "vllm.v1.attention.backends.byte_v2_attn.ByteV2AttentionBackend"
```

因此用户侧只要设置 `kv_cache_dtype="byte_v2"`，CUDA 平台会选择 ByteV2 backend。

### 3.3 环境变量

`vllm/envs.py` 新增了正式注册的 ByteV2 环境变量：

| 环境变量 | 默认值 | 作用 |
| --- | --- | --- |
| `VLLM_BYTE_V2_USE_NATIVE_KERNELS` | `1` | 是否使用 native CUDA ops；设为 `0` 时走 Python/PyTorch reference fallback。 |
| `VLLM_BYTE_V2_COMPRESSED_ONLY_CACHE` | `0` | 是否只在主 page 中保留 compressed payload；设为 `1` 才能体现主要容量收益。 |
| `VLLM_BYTE_V2_ENABLE_SPARSE_FALLBACK_POOL` | `1` | compressed-only 时是否启用外置 sparse fallback pool。 |
| `VLLM_BYTE_V2_SPARSE_FALLBACK_POOL_RATIO` | `0.03` | fallback pool block 数占 physical block 数的比例。 |
| `VLLM_BYTE_V2_SPARSE_FALLBACK_POOL_MIN_BLOCKS` | `512` | fallback pool 最小 block 数；压力测试时可设为 `0` 只用 ratio。 |
| `VLLM_BYTE_V2_DECODE_SPLIT_K` | `0` | decode split-K 强制值；`0` 表示走 host 侧 auto heuristic。 |
| `VLLM_BYTE_V2_DECODE_PAGE_FASTPATH` | `0` | split-K stage1 的 compressed page fast path 开关。 |
| `VLLM_BYTE_V2_DECODE_TILE_FASTPATH` | `1` | page fast path 中 tile-level fast decoder 开关。 |
| `VLLM_BYTE_V2_DECODE_PARALLEL_REDUCE` | `-1` | split-K reduce kernel 选择；`-1` 表示自动。 |
| `VLLM_BYTE_V2_LOSSY_MAX_MISSES_PER_TILE` | `0` | 每个 tile 允许多少个 exponent window miss；`0` 是 lossless。 |
| `VLLM_BYTE_V2_PREFILL_DIRECT_SKIP_VALIDATION_SYNC` | `0` | prefill direct encode 是否跳过 host validation sync。 |
| `VLLM_BYTE_V2_DECODE_APPEND_SKIP_VALIDATION_SYNC` | `0` | decode append 是否跳过 host validation sync。 |
| `VLLM_BYTE_V2_DEFERRED_CACHE_UPDATE_ERROR_CHECK` | `0` | 用 device-side sticky error flag 延迟报告 cache update 错误。 |
| `VLLM_BYTE_V2_PERSISTENT_PARTIAL_WORKSPACE` | `0` | 是否复用 split-K partial workspace。 |

`csrc/libtorch_stable/cache_kernels.cu` 中还存在直接通过 `std::getenv()` 读取的实验开关，
主要用于 kernel A/B：

- `VLLM_BYTE_V2_DECODE_APPEND_BATCH_FASTPATH`
- `VLLM_BYTE_V2_DECODE_CUTE_STAGE1`

这些不在 `envs.py` 的正式表里，阅读或复现实验时需要特别注意。

## 4. ByteV2 cache 格式

### 4.1 tile 格式

核心格式定义在：

- `vllm/v1/attention/backends/byte_v2_codec.py`
- `vllm/v1/attention/backends/byte_v2_layout.py`
- `csrc/libtorch_stable/cache_kernels.cu`

当前 ByteV2 fast tile 固定为 `16 tokens x 16 dims`：

```text
raw BF16 tile: 16 * 16 * 2 = 512 bytes
ByteV2 fast tile:
  base exponent: 1 byte
  fallback flag: 1 byte
  low bytes: 256 bytes
  packed exponent code: 128 bytes
  total: 386 bytes
```

`byte_v2_codec.py` 提供 CPU/PyTorch reference codec：

- `compress_byte_v2_tensor()`
- `decompress_byte_v2_tensor()`
- `ByteV2TensorPayload`

这个文件适合先看算法正确性，不适合看生产性能路径。

### 4.2 page 格式

`byte_v2_layout.py` 中的 `ByteV2PageLayout` 统一管理 page 内 offset：

- page header 固定 `16` bytes。
- `BYTE_V2_PAGE_STATUS_EMPTY = 0`
- `BYTE_V2_PAGE_STATUS_COMPRESSED = 1`
- `BYTE_V2_PAGE_STATUS_RAW_FALLBACK = 2`
- `BYTE_V2_PAGE_STATUS_OFFSET = 0`
- `BYTE_V2_PAGE_VALID_ROWS_OFFSET = 1`

一个 page 对应一个 physical KV block。默认 block size 是 16，也就是一个 page
对应一个完整 tile-token 范围。

主要 helper：

- `pack_byte_v2_kv_block_to_page()`
- `pack_byte_v2_raw_kv_block_to_page()`
- `unpack_byte_v2_kv_block_from_page()`
- `unpack_byte_v2_raw_kv_block_from_page()`
- `count_byte_v2_page_statuses()`
- `byte_v2_reshape_and_cache_ref()`

### 4.3 overlay mode 与 compressed-only mode

`ByteV2FullAttentionSpec.raw_tail_bytes` 决定主 page 是否保留 raw tail：

- `raw_tail_bytes=None`：默认 overlay mode。`__post_init__()` 会把 page size 补到接近
  raw block，主要用于早期 correctness 和兼容 raw fallback。
- `raw_tail_bytes=0`：compressed-only mode。主 page 只保存 compressed payload 和 header，
  raw fallback 放到外置 sparse fallback pool。

要验证容量收益，应使用：

```text
VLLM_BYTE_V2_COMPRESSED_ONLY_CACHE=1
VLLM_BYTE_V2_ENABLE_SPARSE_FALLBACK_POOL=1
```

## 5. KV cache spec 和 allocator 改动

### 5.1 `ByteV2FullAttentionSpec`

`vllm/v1/kv_cache_interface.py` 新增 `ByteV2FullAttentionSpec`。它继承
`FullAttentionSpec`，但重写了 page size 和 allocation size 的计算。

关键字段：

- `tile_token_size=16`
- `tile_head_size=16`
- `fast_tile_payload_bytes=386`
- `page_header_bytes=16`
- `raw_tail_bytes`
- `fallback_pool_bytes`
- `sparse_fallback_pool_ratio`
- `sparse_fallback_pool_min_blocks`
- `sparse_fallback_block_id_bytes`
- `sparse_fallback_next_slot_bytes`
- `sparse_fallback_tile_id_bytes`
- `sparse_fallback_tile_next_slot_bytes`

关键计算：

```text
compressed_payload_bytes =
  (block_size / 16)
  * num_kv_heads
  * (head_size / 16 + head_size_v / 16)
  * 386

raw_block_bytes =
  block_size * num_kv_heads * (head_size + head_size_v) * sizeof(bfloat16)

main_page_size_bytes =
  page_header_bytes + compressed_payload_bytes + raw_tail_bytes

sparse_fallback_pool_blocks(num_blocks) =
  min(num_blocks, max(1, min_blocks, ceil(num_blocks * ratio)))

allocation_size_bytes(num_blocks) =
  main_page_bytes
  + sparse raw fallback pool bytes
  + fallback metadata bytes
```

因为 `allocation_size_bytes(num_blocks)` 中包含 `ceil(num_blocks * ratio)` 和
`min_blocks`，它不再是简单线性 page-size 乘法。

### 5.2 allocator 特殊路径

`vllm/v1/core/kv_cache_utils.py` 新增 ByteV2 sparse fallback 相关 helper：

- `_get_byte_v2_sparse_fallback_spec()`
- `_get_num_blocks_byte_v2_sparse_fallback()`
- `_pool_bytes_for_num_blocks()`
- `_estimate_num_blocks_from_memory()`

主要作用：

- allocator 用二分搜索计算在给定 HBM 下能放多少 ByteV2 block。
- `num_gpu_blocks_override` 时按 `allocation_size_bytes(override)` 重新估算显存。
- max concurrency 和 max memory usage 都使用 ByteV2 allocation size，而不是 raw
  `page_size_bytes * num_blocks`。
- 多 KV cache group 或非 ByteV2 spec 时仍走原 vLLM 路径。

`vllm/v1/core/single_type_kv_cache_manager.py` 把 `ByteV2FullAttentionSpec`
注册到 `FullAttentionManager`，让普通 full attention block manager 能管理 ByteV2
block。

### 5.3 worker 中的内存切分

`vllm/v1/worker/gpu_model_runner.py` 和 `vllm/v1/worker/gpu/attn_utils.py`
都增加了 ByteV2 sparse fallback allocator 切分逻辑。不同 runner 路径里代码近似重复。

一个 layer 的 raw allocation 被切成：

```text
[main compressed pages]
[fallback_pool: pool_blocks * raw_block_bytes]
[padding to int32 alignment]
[fallback_block_ids: num_blocks int32]
[fallback_next_slot: 1 int32]
[fallback_tile_ids: num_blocks * total_tiles int32]
[fallback_tile_next_slot: 1 int32]
```

初始化规则：

- `fallback_block_ids.fill_(-1)`
- `fallback_next_slot.zero_()`
- `fallback_tile_ids.fill_(-1)`
- `fallback_tile_next_slot.zero_()`

之后调用 attention impl 的 `register_sparse_fallback_pool()`，把这些 tensor 注册到
`ByteV2AttentionImpl`，decode 和 cache update 时都会使用同一组 metadata。

## 6. Attention backend 主路径

### 6.1 backend 类

`vllm/v1/attention/backends/byte_v2_attn.py` 新增：

- `ByteV2AttentionBackend`
- `ByteV2Metadata`
- `ByteV2SparseFallbackPool`
- `ByteV2MetadataBuilder`
- `ByteV2AttentionImpl`

`ByteV2AttentionBackend` 声明：

- 支持 `kv_cache_dtype="byte_v2"`。
- 支持 block size `16`。
- native decode head size 需要 16 对齐，当前 CUDA 路径限制在较小固定范围内，主要覆盖
  Llama-3 8B 这类 `head_size=128`、GQA 模型。

### 6.2 metadata 与 prefix cache

`ByteV2MetadataBuilder` 负责把 common attention metadata 转成 ByteV2 metadata：

- `seq_lens`
- `slot_mapping`
- `block_table`
- `query_start_loc`
- decode/prefill token 数
- `page_size_bytes`
- CPU side seq len / query start location cache

当前 prefix cache 支持方式是普通 paged block table 复用：

- `common_prefix_len` 被忽略。
- ByteV2 不使用 vLLM cascade attention 的 common-prefix split。
- prefix cache block 命中后，ByteV2 decode 通过 `block_table` 读取复用的 compressed page。

测试覆盖：

- `tests/config/test_cache_config.py`
- `tests/v1/attention/test_byte_v2_e2e.py::test_byte_v2_prefix_cache_hit_e2e_smoke`
- `tests/v1/attention/test_byte_v2_metadata.py`

### 6.3 KV cache update

`ByteV2AttentionImpl.do_kv_cache_update()` 是 Python backend 到 cache update op 的主入口：

```python
ops.byte_v2_reshape_and_cache(
    key,
    value,
    kv_cache,
    slot_mapping,
    block_size=16,
    num_kv_heads=self.num_kv_heads,
    head_size=self.head_size,
    head_size_v=self.head_size_v,
    page_size_bytes=kv_cache.shape[1],
    fallback_pool=...,
    fallback_block_ids=...,
    fallback_next_slot=...,
    fallback_tile_ids=...,
    fallback_tile_next_slot=...,
    deferred_error=...,
)
```

它不在 Python 层压缩 K/V，真正的生产路径在 native CUDA op 里。

### 6.4 forward 中的 attention 计算

`ByteV2AttentionImpl.forward()` 当前行为：

1. 先处理 decode tokens：
   - 调用 `ops.byte_v2_paged_decode_attention()`。
   - 传入 `kv_cache`、`block_table`、`seq_lens`、fallback pool metadata、
     `partial_workspace`。
2. 再处理 prefill tokens：
   - raw-only prompt prefill 优先使用 `_raw_prefill_attention_flash()`，即 raw vLLM
     FlashAttention fast path。
   - 如果 FlashAttention 不可用，并且是 raw-only prefill，会回退到
     `byte_v2_raw_prefill_attention_torch()`。这只是 correctness fallback，不应作为主性能路径。
   - continuation prefill 会使用 `_paged_prefill_attention_decode_like()`，把 prefill
     query 展开成 decode-like query 调 native paged decode attention。
   - CPU 或特殊 fallback 才走 `byte_v2_paged_prefill_attention_ref()` /
     `byte_v2_paged_prefill_attention_torch()`。

### 6.5 error reporting 与 workspace

`ByteV2AttentionImpl` 还维护：

- sparse fallback pool stats：
  - `get_sparse_fallback_pool_stats()`
  - `get_tile_fallback_stats()`
- deferred cache update sticky error：
  - `reset_deferred_cache_update_error()`
  - `check_deferred_cache_update_error()`
- decode split-K partial workspace：
  - `_get_decode_partial_workspace()`
  - 可通过 `VLLM_BYTE_V2_PERSISTENT_PARTIAL_WORKSPACE=1` 走 persistent workspace。

`vllm/v1/worker/gpu_model_runner.py` 和 `vllm/v1/worker/gpu/model_runner.py`
在执行前后调用 reset/check。打开
`VLLM_BYTE_V2_DEFERRED_CACHE_UPDATE_ERROR_CHECK=1` 后，cache update kernel 可以少做
每 token/layer 的 host sync，而是在安全边界集中检查 device-side sticky flag。

`vllm/v1/worker/worker_base.py` 新增 RPC helper：

- `get_byte_v2_sparse_fallback_stats()`
- `get_byte_v2_tile_fallback_stats()`

benchmark 通过这些 RPC 统计 pool 使用率和 tile fallback 估算。

## 7. Python op 与 native CUDA op 桥接

### 7.1 Python wrapper

`vllm/_custom_ops.py` 新增：

- `byte_v2_reshape_and_cache()`
- `byte_v2_paged_decode_attention()`
- `byte_v2_wmma_layout_microbench()`
- `byte_v2_decode_page_wmma_microbench()`

当 tensor 在 CUDA 上且 `VLLM_BYTE_V2_USE_NATIVE_KERNELS=1` 时，wrapper 调
`torch.ops._C_cache_ops.*`。否则回到 `vllm/v1/attention/backends/byte_v2_ops.py`
中的 Python custom op / reference path。

### 7.2 Torch op schema

`csrc/libtorch_stable/ops.h` 声明 native op：

- `byte_v2_reshape_and_cache`
- `byte_v2_paged_decode_attention`
- `byte_v2_wmma_layout_microbench`
- `byte_v2_decode_page_wmma_microbench`

`csrc/libtorch_stable/torch_bindings.cpp` 注册 schema 和 CUDA impl：

- `STABLE_TORCH_LIBRARY_FRAGMENT(_C_cache_ops, ops)`
- `STABLE_TORCH_LIBRARY_IMPL(_C_cache_ops, CUDA, ops)`

这些 schema 是 Python wrapper 调 native CUDA 的 ABI 边界。改参数时必须同步改
`ops.h`、`torch_bindings.cpp`、`_custom_ops.py`、`byte_v2_ops.py` 和对应 tests。

## 8. Native CUDA 实现

大部分 native 实现都在 `csrc/libtorch_stable/cache_kernels.cu`。

### 8.1 通用 helper

ByteV2 constants 和 helper 从约 `kByteV2TileSize` 开始：

- tile/page constants。
- cache update error code。
- BF16 bits load/store helper。
- compressed tile load/decode helper。
- raw block/tile fallback load/store helper。
- exponent window 选择和 compressibility 判断。

错误码对应 Python 侧 RuntimeError 文案：

- invalid page state
- invalid slot
- duplicate slot
- finalized block update
- fallback pool missing
- invalid fallback slot
- fallback pool exhausted
- invalid valid rows
- no touched token

### 8.2 cache update kernels

cache update host 函数：

```cpp
torch::stable::Tensor byte_v2_reshape_and_cache(...)
```

主要 kernel/路径：

- `byte_v2_validate_prefill_direct_blocks_kernel`
- `byte_v2_prefill_direct_encode_blocks_kernel`
- `byte_v2_decode_append_cache_kernel`
- `byte_v2_init_decode_append_result_kernel`
- `byte_v2_record_deferred_cache_update_error_kernel`
- `byte_v2_init_cache_update_kernel`
- `byte_v2_mark_touched_tokens_kernel`
- `byte_v2_compress_touched_blocks_kernel`
- `byte_v2_write_raw_tokens_kernel`
- `byte_v2_finalize_partial_pages_kernel`
- `byte_v2_compress_full_pages_kernel`

主优化点：

- full prompt prefill 在 compressed-only + sparse fallback 场景下走 block-parallel
  direct encode，避免老路径按 token 扫描。
- decode append 有单 token/批量 append fast path，支持 deferred error reporting。
- fallback 可按 block raw fallback，也保留 tile fallback metadata：
  `fallback_tile_ids` 和 `fallback_tile_next_slot`。
- `VLLM_BYTE_V2_LOSSY_MAX_MISSES_PER_TILE=0` 时保持 lossless；大于 0 时允许有限 miss，
  需要单独评估精度。

### 8.3 decode attention kernels

decode host 函数：

```cpp
torch::stable::Tensor byte_v2_paged_decode_attention(...)
```

它负责：

- 校验 query、kv_cache、block_table、seq_lens、fallback metadata。
- 根据 GPU BF16 WMMA 能力、`q_per_kv`、page 数、env 开关选择 kernel。
- 计算 split-K 数量。
- 分配或使用传入的 `partial_workspace`。

当前保留的主要 decode kernels：

- `byte_v2_paged_decode_attention_kernel`
- `byte_v2_paged_decode_attention_gqa_shared_kernel`
- `byte_v2_paged_decode_attention_gqa_wmma_kernel`
- `byte_v2_paged_decode_attention_gqa_wmma_split_stage1_kernel`
- `byte_v2_paged_decode_attention_gqa_cute_split_stage1_kernel`
- `byte_v2_paged_decode_attention_split_reduce_kernel`
- `byte_v2_paged_decode_attention_split_reduce_parallel_kernel`

当前默认主线是 GQA WMMA + split-K/page-parallel + reduce。CUTE-style stage1 是
实验 opt-in 路径，通过 `VLLM_BYTE_V2_DECODE_CUTE_STAGE1=1` 启用。

### 8.4 microbench kernels

为了隔离问题，CUDA 里还保留了两个 microbench op：

- `byte_v2_wmma_layout_microbench`
- `byte_v2_decode_page_wmma_microbench`

对应 benchmark：

- `benchmarks/kernels/benchmark_byte_v2_wmma_microbench.py`
- `benchmarks/kernels/benchmark_byte_v2_decode_page_wmma_microbench.py`

它们分别用于测纯 WMMA data path，以及 compressed page decode + WMMA QK/PV，
不包含完整 softmax、split-K reduce、scheduler 和模型其他层。

## 9. reference / fallback 实现

新增 ByteV2 Python backend 文件中，reference 路径分工如下：

- `byte_v2_codec.py`：纯 tensor codec reference，验证压缩/解压格式。
- `byte_v2_layout.py`：CPU page pack/unpack/reference reshape_and_cache。
- `byte_v2_torch.py`：GPU/CPU PyTorch eager fallback，支持 sparse fallback pool 和
  tile fallback 读取。
- `byte_v2_decode.py`：CPU paged decode/prefill reference attention。
- `byte_v2_ops.py`：用 `direct_register_custom_op()` 注册 Python op fallback 和 fake impl。

这些文件的价值主要是 correctness、单测和 native op 失效时的降级，不是性能目标。

## 10. 测试覆盖

### 10.1 配置和 spec

- `tests/config/test_cache_config.py`
  - `cache_dtype="byte_v2"` 配置可用。
  - prefix caching 保持开启。
  - dtype helper 把 ByteV2 识别为 compressed KV cache。
- `tests/v1/test_byte_v2_kv_cache_spec.py`
  - `Attention.get_kv_cache_spec()` 返回 `ByteV2FullAttentionSpec`。
  - sliding window 被拒绝。
  - page size、raw tail、compressed-only page size 正确。
  - sparse fallback pool allocation size 正确。
  - full attention manager 注册正确。
- `tests/v1/test_kv_cache_spec_registry.py`
  - `ByteV2FullAttentionSpec` 注册到 `FullAttentionManager` 和 uniform spec 体系。

### 10.2 backend、metadata、codec、layout

- `tests/v1/attention/test_byte_v2_backend.py`
  - backend registry、shape、head size、sparse pool、workspace、deferred error。
- `tests/v1/attention/test_byte_v2_metadata.py`
  - decode/prefill metadata build、block table update、cudagraph capture metadata。
- `tests/v1/attention/test_byte_v2_codec.py`
  - ByteV2 tile codec correctness。
- `tests/v1/attention/test_byte_v2_layout.py`
  - page layout、pack/unpack、status count。

### 10.3 ops、decode、E2E

- `tests/v1/attention/test_byte_v2_ops.py`
  - reshape/cache update reference 和 CUDA native。
  - partial raw block、full block finalize、overwrite reused block。
  - compressed-only cache update、sparse fallback pool exhaustion。
  - native prefill direct encode、decode append fast path。
- `tests/v1/attention/test_byte_v2_decode.py`
  - paged decode reference 对齐 raw attention。
  - raw tail、sparse block fallback、tile fallback。
  - GQA WMMA、split-K、parallel reduce、page fast path、tile fast path、CUTE stage1。
  - raw prefill FlashAttention fast path。
  - continuation prefill decode-like path。
- `tests/v1/attention/test_byte_v2_e2e.py`
  - LLM generate smoke。
  - ByteV2 和 raw vLLM token 输出 smoke 对齐。
  - prefix cache hit smoke。
- `tests/v1/attention/test_byte_v2_wmma_microbench.py`
  - microbench op 基本可用性。

推荐局部测试命令：

```bash
.venv/bin/python -m pytest tests/config/test_cache_config.py -q
.venv/bin/python -m pytest tests/v1/test_byte_v2_kv_cache_spec.py -q
.venv/bin/python -m pytest tests/v1/attention/test_byte_v2_codec.py tests/v1/attention/test_byte_v2_layout.py -q
.venv/bin/python -m pytest tests/v1/attention/test_byte_v2_ops.py -q
.venv/bin/python -m pytest tests/v1/attention/test_byte_v2_decode.py -q
```

涉及 CUDA native op 的测试需要已编译当前 vLLM 扩展，并且机器有可用 GPU。

## 11. benchmark 和 profile 文件

### 11.1 E2E benchmark

`benchmarks/benchmark_byte_v2_decode_e2e.py` 比较：

- `raw`
- `byte_v2_overlay`
- `byte_v2_compressed_only`

它会在不同 decode len 下跑 LLM.generate，并采集：

- latency
- output tok/s
- sparse fallback pool stats
- mode/env 配置

默认模型路径是：

```text
/mnt/sda1/yxz/byte_v2/Meta-Llama-3-8B-Instruct
```

### 11.2 kernel benchmark

- `benchmarks/kernels/benchmark_byte_v2_cache_update_kernel.py`
  - 隔离 `byte_v2_reshape_and_cache`，测 prefill direct encode 和 decode append。
- `benchmarks/kernels/benchmark_byte_v2_decode_kernel.py`
  - 隔离 `byte_v2_paged_decode_attention`，扫 seq len、split-K、fallback ratio。
- `benchmarks/kernels/benchmark_byte_v2_kv_read_microbench.py`
  - 只测 raw KV block 和 ByteV2 compressed page 的 HBM streaming read。
  - 用来证明压缩 page 在纯读场景下确实减少 global load time。
- `benchmarks/kernels/benchmark_byte_v2_wmma_microbench.py`
  - 测纯 16x128 WMMA QK/PV data path。
- `benchmarks/kernels/benchmark_byte_v2_decode_page_wmma_microbench.py`
  - 测 compressed page decode + WMMA QK/PV，不含 softmax/reduce。

### 11.3 profile artifacts

`benchmarks/profiles/` 下的 `.nsys-rep`、`.sqlite`、`.csv`、`.json` 是之前实验结果。
它们不是运行时代码，但用于复盘性能瓶颈。当前实验文档中已记录关键结论：

- compressed page 纯 HBM read 约为 raw 的 `0.754x` bytes，时间约 `0.758x`。
- E2E 没有直接超过 raw，主要因为 decode stage1 的解码指令、metadata 检查、
  shared/register 搬运和 split-K reduce 开销抵消了读带宽收益。

## 12. 当前主路径总结

运行时主流程可以简化成：

```text
用户设置 kv_cache_dtype=byte_v2
  -> CacheConfig 接受 byte_v2
  -> CUDA platform 选择 BYTE_V2 backend
  -> Attention.get_kv_cache_spec() 返回 ByteV2FullAttentionSpec
  -> KV allocator 按 ByteV2 allocation_size_bytes() 分配内存
  -> model runner 把 raw allocation 切成 main pages + fallback pool + metadata
  -> ByteV2AttentionImpl 注册 fallback pool
  -> KV cache update 调 byte_v2_reshape_and_cache native CUDA
  -> raw-only prefill attention 优先走 raw FlashAttention
  -> decode attention 调 byte_v2_paged_decode_attention native CUDA
```

## 13. 当前限制和注意事项

- ByteV2 只支持 decoder full attention，不支持 sliding window attention。
- native CUDA 路径要求 block size 为 16，head size/head size v 为 16 对齐。
- 当前主要验证场景是 BF16 CUDA，CPU/PyTorch 路径主要用于 correctness fallback。
- `ByteV2AttentionImpl.forward()` 不支持 speculative decode。
- 不支持 fused output quantization。
- prefix cache 可用，但通过普通 block table 复用实现，没有使用 cascade attention split。
- 默认 `VLLM_BYTE_V2_COMPRESSED_ONLY_CACHE=0`，此时容量收益不明显。做容量实验必须打开
  compressed-only。
- `VLLM_BYTE_V2_SPARSE_FALLBACK_POOL_RATIO=0.03` 在部分真实长 prompt lossless 场景会耗尽。
  压力测试可调高 ratio 或继续优化 tile/outlier fallback 表达。
- CUTE-style stage1、skip validation sync、persistent workspace 等是 opt-in 实验路径；
  默认主路径应优先保持 correctness 和可复现。
- 当前性能主要瓶颈仍是 decode attention stage1，而不是单纯 HBM read bytes。

## 14. 后续修改时的同步点

修改 ByteV2 op 参数时，必须同步检查：

- `csrc/libtorch_stable/ops.h`
- `csrc/libtorch_stable/torch_bindings.cpp`
- `vllm/_custom_ops.py`
- `vllm/v1/attention/backends/byte_v2_ops.py`
- `vllm/v1/attention/backends/byte_v2_attn.py`
- `tests/v1/attention/test_byte_v2_ops.py`
- `tests/v1/attention/test_byte_v2_decode.py`

修改 page 格式时，必须同步检查：

- `vllm/v1/attention/backends/byte_v2_codec.py`
- `vllm/v1/attention/backends/byte_v2_layout.py`
- `vllm/v1/attention/backends/byte_v2_torch.py`
- `csrc/libtorch_stable/cache_kernels.cu`
- `vllm/v1/kv_cache_interface.py`
- allocator 切分逻辑
- codec/layout/decode tests

修改 allocator 或 fallback pool 时，必须同步检查：

- `ByteV2FullAttentionSpec.allocation_size_bytes()`
- `vllm/v1/core/kv_cache_utils.py`
- `vllm/v1/worker/gpu_model_runner.py`
- `vllm/v1/worker/gpu/attn_utils.py`
- `ByteV2AttentionImpl.register_sparse_fallback_pool()`
- sparse/tile fallback stats RPC
- E2E benchmark 的 fallback stats 输出

修改 decode kernel 时，建议至少跑：

```bash
.venv/bin/python -m pytest \
  tests/v1/attention/test_byte_v2_decode.py::test_byte_v2_paged_decode_attention_op_gqa_wmma_split_k_cuda \
  tests/v1/attention/test_byte_v2_decode.py::test_byte_v2_paged_decode_attention_op_page_fastpath_split_k_cuda \
  tests/v1/attention/test_byte_v2_decode.py::test_byte_v2_paged_decode_attention_op_tile_fastpath_cuda \
  -q
```

修改 cache update kernel 时，建议至少跑：

```bash
.venv/bin/python -m pytest \
  tests/v1/attention/test_byte_v2_ops.py::test_byte_v2_native_prefill_direct_encode_full_blocks_cuda \
  tests/v1/attention/test_byte_v2_ops.py::test_byte_v2_native_compressed_only_cache_update_and_decode_cuda \
  tests/v1/attention/test_byte_v2_ops.py::test_byte_v2_native_decode_append_fallback_pool_exhaustion_cuda \
  -q
```
