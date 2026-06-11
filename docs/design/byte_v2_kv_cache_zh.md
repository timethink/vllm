# Byte-v2 压缩 KV Cache 集成设计

本文档说明如何把 Byte-v2 压缩 KV cache 原型整合到 vLLM 中。目标是让
KV cache 以压缩形式常驻 HBM，在 decode 阶段直接读取压缩页，将 K/V tile
解压到寄存器或 shared memory 中，并立即参与 attention 计算。预期收益是降低
KV cache 显存占用、减少 KV cache 读取带宽，同时尽量让 decode 性能接近 raw
vLLM 路径。

原型代码位于：

```text
/mnt/sda1/yxz/byte_v2/vllm_integration/byte_v2_attention_kernel
```

原型对于 Byte-v2 tile codec 和 fused decode + attention dataflow 很有价值，
但不能直接作为 vLLM attention backend 使用。原型假设 K/V 是连续的
`[G, N, D]` tensor，query rows 共享同一段 K/V，并使用 benchmark 友好的 payload
tensor。vLLM decode 使用 paged KV cache、`block_table`、变长序列、
prefill/decode 混合调度以及多套 runtime metadata。因此集成时需要新增一个
vLLM-native attention backend，并重写 paged Byte-v2 decode kernel。

## 目标

- 增加新的 `kv_cache_dtype=byte_v2` 模式。
- 让大部分 KV cache block 以 Byte-v2 压缩格式存放在 HBM 中。
- decode attention kernel 直接读取压缩 K/V page。
- 在 attention kernel 内部按 tile 解压 K/V，而不是先全量 dequant/gather。
- 保留 vLLM paged KV cache 生命周期、scheduler 和 block table 模型。
- 优先高效支持 GQA decode，尤其是常见模型中的 `q_per_kv=2/4`。
- 显存收益必须基于实际分配字节数，而不是 benchmark 中的逻辑压缩字节数。

## 第一版非目标

- 不追求与所有 vLLM attention backend 完全特性等价。
- 不实现 FP16-lossless Byte-v2 压缩。
- 不支持 sliding window、attention sinks、ALiBi、encoder-decoder attention 或 MLA。
- 不支持 CPU/offload KV cache transfer。
- 不改变 vLLM scheduler 语义。
- 第一阶段不替换 prefill FlashAttention。

## 原型总结

Byte-v2 原型以 `16x16` BF16 tile 为压缩单位。fast-path tile payload 为：

```text
base:          1 byte
low_bytes:   256 bytes
code_packed: 128 bytes
fallback:      1 byte
```

这里的 `fallback` 是 tile-level raw fallback 标志位，不是已经实现的
per-element outlier escape。原型 benchmark 中可以配套 dense `fallback_raw`
保存不可压缩 tile 的完整 BF16 数据，但 vLLM 主路径不能为每个 tile 预留 dense
raw storage，否则显存收益会被抵消。因此当前 vLLM 集成只在 compressed page 中保留
该标志位，正常 compressed tile 的值为 0；遇到无法无损表示的 tile 时，不会写
tile-level payload，而是按当前实现的 block/page 级 raw fallback 处理。

解码公式如下：

```text
low_exp_lsb = low >> 7
delta_hi = code & 0x07
sign = code >> 3
exp_hi = (base >> 1) + delta_hi + ((base & 1) & (low_exp_lsb ^ 1))
high = (sign << 7) | exp_hi
bf16 = (high << 8) | low
```

原型中推荐的 attention 路径：

- `byte_v2_attention_grouped_tiles`：普通 grouped fused decode。
- `byte_v2_attention_grouped_gqa_tiles`：GQA decode reuse，支持 `q_per_kv=2/4`。

核心 dataflow：

```text
compressed K/V payload
  -> 将 16x16 K tile 解压到 shared memory
  -> BF16 WMMA QK
  -> online softmax
  -> 将 16x16 V tile 解压到 shared memory
  -> BF16 WMMA PV
  -> segment partial output
  -> segment reduction
```

重要约束：

- `N` 和 `D` 必须是 16 的倍数。
- `head_dim <= 128`。
- 当前 codec 面向 BF16。
- benchmark 中为了方便使用了 dense `fallback_raw` tensor。这个布局不能直接搬到
  vLLM 中，否则实际分配可能比 raw BF16 cache 更大。
- 当前 kernel 不理解 vLLM paged cache 的 `block_table` 寻址。

## vLLM 需要修改的模块

### Cache dtype 与配置入口

需要修改：

- `vllm/config/cache.py`
  - 在 `CacheDType` 中加入 `byte_v2`。
  - 校验 Byte-v2 只能显式选择。
  - 记录实验性压缩 KV cache 日志。
- `vllm/utils/torch_utils.py`
  - 将 `byte_v2` 映射为 `torch.uint8`。
  - 增加 Byte-v2/compressed KV cache 判断 helper。

`kv_cache_dtype=auto` 不应该自动选择 Byte-v2。第一版必须由用户显式设置。

### Attention backend 注册

需要修改：

- `vllm/v1/attention/backends/registry.py`
  - 新增 `BYTE_V2`。
  - 注册 `ByteV2AttentionBackend`。
- `vllm/platforms/cuda.py`
  - 只有当 `kv_cache_dtype == "byte_v2"` 或用户显式选择时才启用 `BYTE_V2`。
  - 不应把 `BYTE_V2` 放到普通 raw KV cache 的默认优先级前面。

backend 对不支持的配置必须 fail-closed，不能静默回退到 raw backend。

### KV cache spec 与显存预算

需要新增 Byte-v2 专用 spec：

- `vllm/v1/kv_cache_interface.py`
  - 新增 `ByteV2FullAttentionSpec`，作用类似 `TQFullAttentionSpec`。
  - 重写 page size 计算，使用真实 Byte-v2 page layout。
- `vllm/model_executor/layers/attention/attention.py`
  - 在 `Attention.get_kv_cache_spec()` 中，当 `kv_cache_dtype == "byte_v2"` 时返回
    `ByteV2FullAttentionSpec`。

spec 必须预算：

- 每个 full block 的固定压缩 payload 字节。
- block status metadata。
- raw tail storage，如果 raw tail 存在于 page 内。
- fallback metadata。
- sparse fallback pool 预留，如果 fallback pool 属于该层 cache 分配。

关键要求：vLLM block budget 必须基于实际分配字节数，而不是 benchmark 里的逻辑压缩字节数。

### KV cache 分配与 reshape

vLLM GPU worker 目前通过 backend 提供的 cache shape 分配和 reshape KV cache：

- `vllm/v1/worker/gpu/attn_utils.py`
- `vllm/v1/worker/gpu_model_runner.py`

Byte-v2 backend 的 `get_kv_cache_shape()` 应返回单个 `uint8` tensor 形状，例如：

```text
[num_blocks, page_bytes]
```

也可以返回结构化但仍然连续的 view，例如：

```text
[num_blocks, num_kv_heads, num_d_tiles, payload_bytes]
```

推荐第一版使用单个 flat `uint8` page，因为 vLLM KV cache allocator、sleep/wake
路径和 block 管理已经默认每个 cache group 对应一个 storage object。Byte-v2 backend
可以在内部派生具体 offsets 和 typed views。

当前实现对 compressed-only + sparse fallback pool 采用 allocator-aware 布局：

```text
per-layer KVCacheTensor:
  main compressed pages:
    [num_blocks, main_page_size_bytes]

  sparse fallback raw pool:
    [ceil(num_blocks * pool_ratio), raw_block_bytes]

  4-byte aligned metadata:
    fallback_block_ids: [num_blocks] int32
    fallback_next_slot: [1] int32
    fallback_tile_ids: [num_blocks, total_tiles_per_block] int32
    fallback_tile_next_slot: [1] int32
```

因此 `KVCacheTensor.size` 不再是简单的 `num_blocks * page_size_bytes`，而是：

```text
allocation_size(num_blocks) =
  num_blocks * main_page_size_bytes
  + ceil(num_blocks * pool_ratio) * raw_block_bytes
  + alignment padding
  + num_blocks * sizeof(int32)
  + sizeof(int32)
  + num_blocks * total_tiles_per_block * sizeof(int32)
  + sizeof(int32)
```

`kv_cache_utils` 为该布局使用二分搜索反推最大 `num_blocks`，避免用线性 page
size 估算时高估可分配 block 数。`gpu_model_runner` 在创建 KV cache 后，把同一个
raw tensor 拆成主 compressed cache、fallback pool、`fallback_block_ids`、
`fallback_next_slot`、`fallback_tile_ids` 和 `fallback_tile_next_slot`，再注册给 Byte-v2 backend。这样 sparse fallback pool
  会进入 vLLM 的 GPU KV cache size 日志和 block budget，而不是 backend 额外偷偷分配。

当前 sparse fallback pool 同时支持两种粒度：

- `fallback_block_ids[physical_block]`：指向完整 raw BF16 K/V block。该路径仍用于
  decode append 的未满 partial block，以及未传 tile metadata 的兼容路径。
- `fallback_tile_ids[physical_block, tile_idx]`：指向一个 `16x16` BF16 raw tile。
  lossless full block 编码时，只有超出 ByteV2 16-exponent window 的 tile 写入
  tile fallback；其余 tile 继续写 compressed payload，page status 仍为
  `COMPRESSED`。

raw tile pool 复用 `fallback_pool` 的字节空间，但以 512B 为一个 tile slot 寻址。
完整 block fallback 从 pool 头部按 raw block slot 递增分配，tile fallback 从 pool
尾部按 512B tile slot 递减分配；`fallback_next_slot` 和
`fallback_tile_next_slot` 是两个独立计数器。当前还没有 element-level outlier list。

### Cache update 路径

需要新增压缩写 cache 的 GPU op：

- `csrc/libtorch_stable/cache_kernels.cu`
- `csrc/libtorch_stable/torch_bindings.cpp`
- `vllm/_custom_ops.py`

这个 op 的逻辑角色类似 `reshape_and_cache_flash`，但写入 Byte-v2 page：

```text
byte_v2_reshape_and_cache(
    key,
    value,
    kv_cache_uint8,
    slot_mapping,
    block_status,
    fallback_pool,
    ...
)
```

推荐 block 生命周期：

1. 新 token 写入当前 physical block 的 raw tail storage。
2. 当 block 填满 16 行后，对所有 KV head 和 `head_dim` tile 进行压缩。
3. fast-path Byte-v2 payload 写入压缩 page 区域。
4. fallback raw tile 写入 sparse fallback pool，或者将整个 block 标成 raw fallback。
5. 将 physical block 标记为 finalized/compressed。

这样可以避免每 decode 一个 token 就重新压缩整个 16-row tile。

### Byte-v2 attention backend

新增文件：

```text
vllm/v1/attention/backends/byte_v2_attn.py
```

结构上参考：

- `turboquant_attn.py`：最接近压缩 KV cache backend。
- `flash_attn.py`：raw CUDA attention backend 参考。

backend 应包含：

- `ByteV2AttentionBackend`
- `ByteV2Metadata`
- `ByteV2MetadataBuilder`
- `ByteV2AttentionImpl`

第一版校验要求：

- CUDA platform。
- decoder attention。
- `kv_cache_dtype == "byte_v2"`。
- `block_size == 16`。
- `head_dim % 16 == 0`。
- `head_dim <= 128`。
- BF16 K/V cache。
- 不支持 sliding window、attention sinks、ALiBi、MLA。

第一版保持：

```text
forward_includes_kv_cache_update = False
```

这样 `Attention.forward()` 仍然可以先调用 backend cache update，再调用 attention forward。

### Metadata builder

Byte-v2 metadata builder 需要向 kernel 提供：

- decode token 数量。
- decode row 的 request-to-token mapping。
- `seq_lens`。
- `block_table`。
- query start locations。
- cache update 使用的 slot mappings。
- CUDA graph 兼容的静态 buffer。
- raw tail 或 raw fallback block 标志。

与 TurboQuant 类似，mixed prefill/decode batch 中应优先把 decode 排在前面。这样
decode kernel path 更简单，prefill 部分仍可以走现有 FlashAttention。

## Byte-v2 Page Layout

第一版使用 `block_size=16`，与 Byte-v2 token tile 高度一致。

推荐逻辑 page layout：

```text
physical block page:
  header:
    status
    valid_rows
    fallback counters or pool offsets

  K fast payload:
    [kv_head][d_tile]:
      base
      fallback metadata
      low_bytes
      code_packed

  V fast payload:
    [kv_head][d_tile]:
      base
      fallback metadata
      low_bytes
      code_packed

  optional raw tail storage:
    [K/V][kv_head][valid_tail_rows][head_dim]
```

不能为每个 tile dense 分配 `fallback_raw`。应该选择：

1. 每层一个 sparse fallback pool。
2. 每个 cache group 一个 sparse fallback pool。
3. block-level raw fallback。

sparse fallback 显存比最好，但需要容量管理。block-level raw fallback 简单一些，更适合作为第一版 correctness 路径。

## Decode Kernel 设计

原型里的 tile decode helper 可以复用思路，但 kernel 调度和寻址必须重写。

需要的输入：

```text
q:              [num_decode_tokens, num_heads, head_dim]
kv_cache:       uint8 compressed pages
block_table:    [num_reqs, max_blocks_per_req]
seq_lens:       [num_reqs]
out:            [num_decode_tokens, num_heads, head_dim]
scale:          attention scale
```

每个逻辑 K/V tile 的寻址：

```text
logical_block = n_tile
physical_block = block_table[request_id, logical_block]
page = kv_cache[physical_block]
tile = page[k_or_v, kv_head, d_tile]
```

如果 block 已 finalized，kernel 从压缩 payload 解码到 shared memory 或寄存器。如果
block 是 raw tail 或 raw fallback，kernel 直接读取 BF16。

原型假设 `M=16` query rows 共享同一段 K/V。vLLM decode 通常是很多 request
各一个 query row，而且每个 row 的 `block_table` 不同。因此生产 kernel 不能简单地把
16 个任意 request 塞进原型的 `M` 维度。

推荐 kernel 里程碑：

1. 正确的 paged decode kernel：
   - 一个 CTA 处理一个 request 的一个 KV head 或 GQA group。
   - 支持压缩 full block、raw tail block、fallback block。
   - `q_per_kv=1` 时可能 tensor core 利用率不足，但能验证完整 runtime 路径。
2. GQA reuse kernel：
   - 特化 `q_per_kv=2` 和 `q_per_kv=4`。
   - K/V 解压一次，服务多个 Q head。
3. Split-K 或 segmented decode：
   - 长上下文按 block range 并行。
   - reduce partial softmax/output segments。
   - 保留原型 online softmax reduction 思路，同时支持 vLLM block-table 寻址。
4. 常见 shape 特化：
   - `head_dim=128`
   - `block_size=16`
   - `q_per_kv=4`
   - BF16 query/KV

## Prefill 与 Mixed Batch

第一版不替换 prefill attention。

推荐行为：

- First-chunk prefill：
  - 用 raw FlashAttention 计算 attention。
  - 将 K/V 写入 Byte-v2 cache。
  - cache update 后压缩所有 full blocks。
- Decode：
  - 使用 Byte-v2 compressed paged decode。
- Mixed prefill/decode：
  - decode slice 走 Byte-v2。
  - prefill slice 走 FlashAttention。
- Continuation prefill：
  - 初期使用保守 fallback path。
  - 后续再优化为专用 compressed prefill 或 chunked decode path。

## 构建与 Op 注册

kernel 代码应整合进 vLLM native extension build，而不是继续作为 standalone benchmark extension 加载。

可能涉及文件：

- `csrc/libtorch_stable/cache_kernels.cu`
  - 增加压缩 cache update/finalization kernel，或者拆成单独 Byte-v2 source。
- `csrc/libtorch_stable/torch_bindings.cpp`
  - 注册 Byte-v2 cache update 和 attention op。
- `vllm/_custom_ops.py`
  - 增加 Python wrapper。
- `vllm/v1/attention/backends/byte_v2_attn.py`
  - 在 `do_kv_cache_update()` 和 `forward()` 中调用 wrapper。

原型中的 benchmark-only CPU compression 不能作为 runtime 路径使用。

## 测试计划

Codec 与 cache update 测试：

- 单个 `16x16` BF16 tile round-trip。
- 多 heads、多 `head_dim` tiles round-trip。
- 强制 fallback tile。
- fallback pool overflow 或 block-level raw fallback。
- 随机 `slot_mapping`。
- partial tail block。
- full block finalization。

Attention 正确性测试：

- Byte-v2 decode output 与 raw FlashAttention/PagedAttention 对比。
- 随机 `block_table`。
- 随机 `seq_lens`。
- tail length 从 1 到 15。
- `q_per_kv=1/2/4`。
- `head_dim=64/128`。
- prefix-cache 复用 full blocks。
- mixed prefill/decode batches。

显存预算测试：

- 校验 `ByteV2FullAttentionSpec.page_size_bytes`。
- 启用压缩后，分配 block 数应相对 raw BF16 增加。
- 实际分配字节数必须包含 fallback storage。

性能测试：

- Byte-v2 decode latency 与 raw BF16 vLLM 对比。
- 长上下文 tokens/s。
- 使用 Nsight Compute 测 HBM read bandwidth。
- compression ratio 必须基于实际 allocation，而不是逻辑 payload。
- 在代表性模型层上统计 fallback rate。

运行测试时遵守 vLLM 环境规则，例如：

```bash
.venv/bin/python -m pytest tests/path/to/test_file.py -v
```

## 实施阶段

### 当前实现状态

截至当前实现，已经完成：

- `kv_cache_dtype=byte_v2` 配置入口、dtype helper、backend 注册和配置校验。
- `ByteV2FullAttentionSpec` 与 flat uint8 page cache shape。
- Byte-v2 BF16 tile codec、page layout、partial raw tail、block-level raw fallback。
  默认布局仍是 raw-overlay page，即每个 page 预留 compressed payload 和 raw block
  二者中的较大空间，保证不可压缩 block 可以无损 raw fallback。
- opt-in compressed-only cache layout 已接入：设置
  `VLLM_BYTE_V2_COMPRESSED_ONLY_CACHE=1` 后，Byte-v2 page 只按
  `header + compressed_payload` 分配，不再为每个 page 预留 raw tail/raw fallback
  空间。该模式默认启用 sparse fallback pool：大多数 block 仍以 compressed page
  存放，少数超出 Byte-v2 exponent window 的不可压缩 block 会写入单独的 raw BF16
  fallback pool，主 page header 只保留 `RAW_FALLBACK` 状态和有效行数。
- sparse fallback pool 已接入 native cache update 和 native paged decode：
  `VLLM_BYTE_V2_ENABLE_SPARSE_FALLBACK_POOL=1` 默认打开，
  `VLLM_BYTE_V2_SPARSE_FALLBACK_POOL_RATIO` 控制 pool slot 数占 physical block
  数的比例，默认 `0.03`；`VLLM_BYTE_V2_SPARSE_FALLBACK_POOL_MIN_BLOCKS`
  提供每层最小 pool slot 下限，默认 `512`。实际 pool 容量为
  `min(num_blocks, max(ceil(num_blocks * ratio), min_blocks))`。pool 容量不足时
  当前实现 fail closed，并在 native cache update 报错中打印明确的 exhausted/missing
  诊断，不再对不可压缩值做饱和有损编码。
- sparse fallback pool 已接入 vLLM KV cache allocator：compressed-only 模式下，
  `ByteV2FullAttentionSpec` 会把主 compressed page、raw fallback pool 和 metadata
  一起计入每层 `KVCacheTensor.size`；`kv_cache_utils` 用该非线性分配公式计算
  `num_blocks`；`gpu_model_runner` 从同一个 raw allocation 中切分出 pool 和 metadata
  并注册给 backend。`GPU KV cache size` 日志现在反映 fallback pool 预留后的真实容量。
- CPU reference cache update，支持 partial block、full block finalization、不可压缩 tile
  的 raw fallback。
- CPU reference paged decode attention，支持 compressed page、raw tail/raw fallback page、
  GQA head mapping。
- CPU reference mixed decode/prefill correctness path：decode slice 走 Byte-v2 paged
  decode，prefill slice 从 Byte-v2 pages gather K/V 后做 causal attention。
- 默认 runtime 使用 PyTorch eager CUDA correctness fallback：cache update 可在 CUDA 上写 Byte-v2 page，
  full block 会尝试压缩，decode 可在 CUDA 上从 compressed/raw Byte-v2 page gather
  并计算 attention。
- 原生 CUDA correctness 路径默认打开；设置 `VLLM_BYTE_V2_USE_NATIVE_KERNELS=0`
  可强制回退到 PyTorch fallback，便于对照和调试。
- CUDA `byte_v2_reshape_and_cache` 已实现第一版 native cache update：支持
  non-contiguous model K/V 输入、partial raw page 写入、existing raw page finalization、
  full page 压缩、block-level raw fallback 以及 tile-level fallback。
- compressed-only CUDA cache update 已实现：支持 partial compressed page、后续 token
  追加、full block finalization，并且不在 page 内写入 raw fallback 数据；不可压缩
  partial block 会进入 block-level sparse fallback pool；full block 现在优先以
  tile 粒度 fallback，只把不可压缩 `16x16` BF16 tile 写入 raw tile pool。
- 当前已实现 tile-level fallback payload，但还没有实现 element-level outlier list。
  native decode 会在 compressed page 内读取 per-tile `fallback` byte，并通过
  `fallback_tile_ids` 从 tile pool 读取 raw tile。
- CUDA `byte_v2_paged_decode_attention` 已实现第一版 native paged decode attention：
  通过 `block_table` 直接读取 Byte-v2 page，在 kernel 内读取 compressed/raw K/V，
  并完成 online softmax attention。当前实现包含三条 decode dispatch：
  `q_per_kv=1` 或不支持的 GQA shape 走每个 request/head 一个 CUDA block 的
  scalar baseline；`q_per_kv=2..8` 且设备支持 SM80+ BF16 tensor core 时，走
  GQA shared-memory/WMMA fused kernel；没有 BF16 WMMA 支持时，走 GQA shared-memory
  scalar fallback。
- GQA shared-memory/WMMA fused decode kernel 已接入 native op：一个 CTA 处理一个
  request 的一个 KV head，K/V Byte-v2 page 解压一次并服务同组多个 Q head；QK 使用
  BF16 WMMA 计算 16-token page 的 score tile，softmax 使用 tile-wise online 更新，
  PV 也使用 BF16 WMMA 做 `P @ V_tile`，最终 accumulator 保持 FP32 后写回 BF16。
  该路径复用同一套 compressed/raw/sparse fallback load helper，支持 raw-overlay 和
  compressed-only page。
- native paged decode 已支持 raw-overlay page size 和 compressed-only page size，
  并修复了 batch decode 中 query 非 contiguous 时 native op 拒绝运行的问题。
- backend 已允许受支持配置进入 `BYTE_V2`，但 cudagraph support 暂时设为 `NEVER`。
- prefill 优先使用当前 forward 中的 raw K/V 做 causal attention；只有 continuation
  prefill 等已有历史上下文的情况才从 Byte-v2 page gather。
- opt-in vLLM e2e smoke test 已加入，设置 `BYTE_V2_RUN_E2E=1` 后可用本地小模型
  验证 `LLM.generate()`，并包含 raw vLLM 与 Byte-v2 贪心生成 token 对齐检查。
- native e2e 已通过：`BYTE_V2_RUN_E2E=1` 可跑通本地 Pythia-14m smoke 测试；
  `VLLM_BYTE_V2_USE_NATIVE_KERNELS=0` 可切回 PyTorch fallback 做对照。
- compressed-only native e2e 已通过：在
  `VLLM_BYTE_V2_COMPRESSED_ONLY_CACHE=1 BYTE_V2_RUN_E2E=1` 下可跑通本地
  Pythia-14m smoke 测试，并通过 raw vLLM 与 Byte-v2 贪心生成 token 对齐检查。
- native CUDA sparse fallback pool 单测已覆盖不可压缩 block：cache update 会写
  fallback pool，paged decode 会从 pool 读取 raw K/V，并与 raw attention 输出对齐。
- Python custom op 调用点和 native op schema/stub，显式传递并校验
  `page_size_bytes`。
- page header 状态统计 helper 支持 CPU/CUDA tensor，用于观测
  empty/compressed/partial compressed/partial raw/full raw fallback/invalid page 数量。

当前 Llama-3-8B eager e2e benchmark 结果：

- 测试配置：`Meta-Llama-3-8B-Instruct`，A40，batch=1，prompt_len=1024，
  decode_len=16/64/128/256，`enforce_eager=True`，`gpu_memory_utilization=0.80`。
- KV 容量：raw vLLM 为 166,768 tokens；Byte-v2 raw-overlay 为 166,736 tokens。
  在旧默认 `pool_ratio=0.10` 下，Byte-v2 compressed-only + sparse fallback pool
  为 195,216 tokens，约为 raw 的 117.1%；将默认 `pool_ratio` 降到 `0.03`
  后，实测 compressed-only 容量为 212,656 tokens，约为 raw 的 127.5%。
  同配置 pool stats run 没有出现 pool exhausted；最终总 fallback `next_slot`
  为 2,901 / 12,768，最紧张单层为 95 / 399，约 23.8% pool 使用率。
- decode_len=256 output tok/s：raw vLLM 为 34.90；Byte-v2 raw-overlay 为 6.55，
  约为 raw 的 18.8%；Byte-v2 compressed-only 在 3% pool 下为 4.52，约为 raw
  的 13.0%。
- 对比 GQA WMMA 接入前的同配置 baseline，decode_len=256 下 raw-overlay 从
  7.80 tok/s 降到 6.55 tok/s，compressed-only 从 5.06 tok/s 降到 4.52 tok/s。
- `prompt_len=1024, decode_len=64` profile 见
  `benchmarks/byte_v2_profile_llama3_8b_p1024_d64_summary.md`。当前主要瓶颈不是
  sparse fallback pool，也不是 3% pool 容量，而是 ByteV2 prefill/continuation
  prefill 仍走 PyTorch fallback，以及 compressed-only cache update compression
  kernel 还没有 block-parallel fast path。nsys 中 ByteV2 native paged decode kernel
  只占 ByteV2 GPU kernel time 的约 1.7%，而 PyTorch GEMV/softmax/elementwise/copy
  合计超过 89%。

这些数字说明当前代码已经具备 e2e 可用性和实际压缩 page allocation，但当前
GQA shared-memory/WMMA fused decode 的 e2e 吞吐没有改善，反而低于接入前的
conservative baseline。下一步应该做 kernel/pipeline 分解 profile，分别测
cache update、prefill、decode attention、per-token kernel latency，而不是继续直接堆功能。

当前 e2e 测试会触发 vLLM GPU memory profiling。为了避免其他 CUDA 测试进程释放显存
造成 profiler 断言失败，建议单独运行 e2e 命令。

仍未完成：

- split-K 或 segmented long-context decode。当前 GQA WMMA kernel 仍是一个
  request/KV head 一个 CTA，长上下文下不能跨 block range 并行，需要后续增加
  partial softmax/output segment reduction。
- CUDA graph capture 和更完整的 shape autotune。当前 native e2e 已不依赖 PyTorch
  fallback，但 cache update 仍存在多 kernel pipeline 和 host sync，decode kernel
  也还没有针对 `head_dim=128`、`q_per_kv=4`、不同 batch/decode length 做系统调优。
- sparse fallback pool 的 slot 回收和更复杂 group 场景。当前 worker 路径已经把 pool
  纳入 vLLM KV cache block budget，但 pool slot 仍采用单调分配，block 重新变为
  compressed 时会清除映射但暂不回收 slot；tile slot 也采用单调分配，暂不回收。
  Byte-v2 sparse allocator 路径目前覆盖
  单一 Byte-v2 full-attention cache group，混合 Byte-v2/hybrid group 还没有泛化。
  backend 里仍保留未注册 pool 时的懒分配兼容路径，主要用于独立单测和调试，不作为
  正常 e2e 的容量模型。
- element-level outlier list 或 multi-window tile。tile-level fallback 已能避免少数
  exponent outlier 放大成整块 raw BF16 fallback；如果真实 E2E/profile 显示 raw tile
  仍然过多，再继续评估 element-level outlier list，但它会增加 decode 中的
  scatter/patch 开销，需要单独 benchmark 后决定是否保留。
- cache update 当前为一个 native op 内的多 kernel pipeline，并且为了返回精确
  `packed_block_ids` 会做一次 host sync；后续需要改为 graph-friendly 的 device
  side 状态输出或静态 buffer。
- 更完整的 serving 场景、CUDA graph 验证和性能优化。

### Phase 1：Correctness backend

- 已完成：增加 `kv_cache_dtype=byte_v2`。
- 已完成：增加 `ByteV2FullAttentionSpec`。
- 已完成：注册 Byte-v2 backend。
- 已完成：增加压缩 cache update op，包括 PyTorch fallback 和 opt-in native CUDA 路径。
- 已完成：为 correctness 增加保守 decode fallback。
- 已完成：验证基础 vLLM scheduling、block allocation 和 cache lifecycle。

### Phase 2：直接压缩 paged decode

- 已完成：增加第一个 vLLM-native Byte-v2 paged decode correctness kernel。
- 已完成：通过 `block_table` 直接读取压缩 page。
- 已完成：在 attention kernel 内读取并解码 compressed/raw K/V。
- 已完成：支持 raw tail blocks。
- 已完成：支持 block-level raw fallback。
- 已完成：支持 compressed-only allocation 和 allocator-aware 无损 sparse fallback
  pool MVP。
- 已完成：支持 tile-level fallback pool；lossless full block 只把不可压缩 tile 写入
  raw tile pool，block-level raw fallback 主要保留给未满 partial block 和兼容路径。
- 待完善：slot 回收、hybrid group 泛化和 CUDA graph 友好状态输出。
- 已完成：与 raw vLLM output 对比，并已测量 compressed-only page allocation 带来的
  KV token capacity 增加；尚未达到生产性能目标。

### Phase 3：性能路径

- 已完成：增加 GQA reuse shared-memory/WMMA fused decode kernel，覆盖 `q_per_kv=2..8`
  的 SM80+ BF16 tensor core 路径，并保留 shared scalar fallback。
- 增加 split-K 或 segmented long-context decode。
- 特化常见 shapes。
- 调优 shared memory 使用和 segment size。
- 确保 CUDA graph 兼容。
- 优化 fallback metadata layout，避免额外 global reads；当前 tile id 使用 dense
  `[num_blocks, total_tiles_per_block]` int32 表，后续可继续压缩成 compact list 或
  bitset + index。
- 已完成：设计并验证 tile-level fallback pool，避免少数 outlier 触发整块 raw fallback。
- 评估 element-level outlier list 或 multi-window tile；只有在真实 E2E/profile
  有收益时才保留。

### Phase 4：特性扩展

- FP16 支持。
- Sliding window。
- Prefix cache 压力测试与复用。
- KV transfer/offload 兼容。
- Speculative decode 兼容。
- Sleep/wake 支持。

## 主要风险

### Dense fallback storage 会抵消显存收益

原型里的 dense `fallback_raw` tensor 适合 benchmark，但不适合以节省显存为目标的
vLLM 集成。vLLM 必须使用 sparse fallback storage 或 block-level fallback accounting。

### 原型 query layout 不匹配 vLLM decode

原型的 `M=16` query tile 假设 16 行共享同一段 K/V。vLLM decode 通常是很多 request
各一个 query，且各自有不同 `block_table`。生产 decode kernel 必须是 paged 且 request-aware。

### Partial block 必须有明确策略

Byte-v2 压缩单位是 16-token tile，而 decode 每次通常追加一个 token。推荐策略是
block 未满时保留 raw tail storage，block 填满后进行 finalization。

### 端到端性能可能不同于 prototype benchmark

原型测的是连续 tensor。vLLM serving 包含 scheduler metadata、page-table indirection、
CUDA graph 约束、prefill/decode 混合和 fallback 管理。因此需要同时 benchmark
kernel-level latency 和 end-to-end latency。

## 建议的第一批 PR 边界

第一批可上游的修改应保持范围较窄：

- 增加 `byte_v2` 配置入口。
- 增加 cache spec。
- 增加 backend 注册和配置校验。
- 增加 config/spec 相关测试。

不要提交单纯 cosmetic 或低价值 PR。提交者需要人工 review 每一行改动，运行相关测试，
并在 PR 描述中说明使用了 AI assistance。
