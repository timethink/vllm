# ByteV2 两个主要性能瓶颈解决方案

本文档用于指导后续代码修改，目标是解决当前 ByteV2 相比 raw vLLM 明显落后的两个主要瓶颈：

1. `byte_v2_compress_touched_blocks_kernel` cache update/compress 太慢。
2. `byte_v2_paged_decode_attention_gqa_wmma_split_stage1_kernel` decode attention 太慢。

相关 profile 报告：

- `benchmarks/profiles/current_bytev2_vs_raw_p2048_d32_profile_report.md`

相关背景文档：

- `docs/design/byte_v2_kv_cache_zh.md`
- `docs/design/byte_v2_outperform_raw_zh.md`
- `docs/design/byte_v2_kernel_optimization_experiments_zh.md`

## 当前 profile 结论

测试配置：

```text
model: /mnt/sda1/yxz/byte_v2/Meta-Llama-3-8B-Instruct
GPU: CUDA_VISIBLE_DEVICES=2
batch_size: 1
prompt_len: 2048
decode_len: 32
dtype: bfloat16
block_size: 16
ByteV2: compressed-only + 3% sparse fallback pool + min 512 fallback blocks
```

E2E measured:

| 模式 | elapsed | output tok/s |
|---|---:|---:|
| raw prefix-on | 0.925 s | 34.58 |
| raw prefix-off | 1.180 s | 27.13 |
| ByteV2 compressed-only | 3.557 s | 9.00 |

Nsight kernel 分类：

| 环节 | raw prefix-off | ByteV2 |
|---|---:|---:|
| Linear/GEMM/MLP | 1911.487 ms | 1884.480 ms |
| KV cache update | 5.886 ms | 3087.972 ms |
| decode attention | 67.893 ms | 155.151 ms |

结论：

- 普通模型计算不是瓶颈，ByteV2 和 raw 基本一致。
- 第一瓶颈是 ByteV2 cache update/compress，约占 ByteV2 profiled GPU kernel time 的 58.46%。
- 第二瓶颈是 ByteV2 decode attention，当前约为 raw attention 的 2.29x。
- prefix cache 缺失会影响默认 E2E 公平性，但在当前 p2048 d32 中只解释约 0.25 s 差距，不是最大问题。

## 当前实现位置

核心 CUDA 实现：

- `csrc/libtorch_stable/cache_kernels.cu`
  - `byte_v2_reshape_and_cache`
  - `byte_v2_init_cache_update_kernel`
  - `byte_v2_mark_touched_tokens_kernel`
  - `byte_v2_compress_touched_blocks_kernel`
  - `byte_v2_paged_decode_attention_gqa_wmma_split_stage1_kernel`
  - `byte_v2_paged_decode_attention_split_reduce_kernel`

Python backend：

- `vllm/_custom_ops.py`
- `vllm/v1/attention/backends/byte_v2_attn.py`
- `vllm/v1/attention/backends/byte_v2_ops.py`

测试：

- `tests/v1/attention/test_byte_v2_ops.py`
- `tests/v1/attention/test_byte_v2_decode.py`
- `tests/v1/attention/test_byte_v2_e2e.py`

benchmark：

- `benchmarks/benchmark_byte_v2_decode_e2e.py`
- `benchmarks/kernels/benchmark_byte_v2_decode_kernel.py`

## 瓶颈一：cache update/compress

### 根因

当前 compressed-only cache update 的主路径是：

```text
byte_v2_reshape_and_cache
  allocate valid_rows[num_blocks]
  allocate touched_flags[num_blocks]
  allocate packed_flags[num_blocks]
  allocate overwrite_flags[num_blocks]
  allocate block_token_indices[num_blocks * 16]
  allocate raw_staging[num_tokens * raw_block_bytes]
  memset error/block_token_indices/overwrite_flags
  byte_v2_init_cache_update_kernel over all num_blocks
  byte_v2_mark_touched_tokens_kernel over num_tokens
  byte_v2_compress_touched_blocks_kernel over all num_blocks
  optional D2H validation/copy packed_flags outside CUDA graph capture
```

这个结构的问题：

- decode 每次通常只更新 1 个 token，但仍然初始化和扫描当前 layer 的所有 physical blocks。
- `byte_v2_compress_touched_blocks_kernel` 是 one CTA per physical block，虽然已经比 token scan 好，但 grid 仍是 `num_blocks`，大部分 CTA 只是检查 `touched_flags` 后返回。
- partial block 每追加一个 token 都会重建 raw block 并重新压缩一次；对于一个 16-token block，前 15 次 decode 更新都在反复做小 block 的解压/重压。
- `raw_staging` 按 `num_tokens * raw_block_bytes` 分配，prefill 时偏大，decode 时也造成 allocator/runtime 压力。
- 非 graph capture 路径有 D2H error/packed_flags/fallback_next_slot copy 和 stream synchronize。

当前 profile 中，该 kernel：

```text
byte_v2_compress_touched_blocks_kernel
total: 3087.972 ms
calls: 1504
avg:   2053.17 us
```

raw 对应 KV 写入：

```text
reshape_and_cache_flash_kernel
total: 5.886 ms
calls: 1504
avg:   3.913 us
```

### 目标

近期目标：

- p2048 d32 profile 中，ByteV2 cache update/compress 总时间从 3.09 s 降到 0.30 s 以下。
- 单次 decode cache update 从约 2 ms 降到 50 us 以内。
- 保持 compressed-only page layout 和 sparse fallback pool 兼容。
- 不破坏 cudagraph capture/replay。

理想目标：

- decode 阶段 cache update 接近 raw `reshape_and_cache_flash_kernel` 同量级。
- prefill 阶段 cache update 只按 touched physical blocks 工作，不按全 cache `num_blocks` 工作。

### 方案 1：decode partial block 延迟压缩

这是最高优先级方案。

核心思路：

- decode 追加 token 时，如果当前 physical block 还没有满 16 rows，不立即压缩成 ByteV2 page。
- 对 partial block 使用 raw fallback pool 存放原始 BF16 K/V。
- 只有当 `block_offset == 15` 时，才把 16-token raw block 一次性压缩成 ByteV2 page。
- decode attention 已经支持 `kByteV2PageStatusRawFallback`，因此只让当前 active partial block 走 raw fallback 不影响正确性。

数据状态：

| page status | valid rows | 存储位置 | 含义 |
|---|---:|---|---|
| empty | 0 | none | 未写入 |
| raw fallback | 1..15 | fallback pool | active partial block |
| compressed | 16 | kv_cache page | finalized compressed block |
| raw fallback | 16 | fallback pool | 不可压缩 finalized block |

decode fast path kernel：

```text
byte_v2_decode_append_cache_kernel
grid: one CTA per token

for each token:
  slot = slot_mapping[token_idx]
  block_id = slot / 16
  row = slot % 16
  page = kv_cache[block_id]

  if row < 15:
    ensure fallback slot for block
    if page is compressed partial legacy:
      decompress existing rows once into fallback block
    write current K/V row into fallback block
    page.status = raw_fallback
    page.valid_rows = row + 1
    return

  if row == 15:
    ensure raw block exists:
      if page raw_fallback: use fallback block
      if page empty or compressed partial legacy: materialize to fallback/raw staging
    write row 15
    if block compressible:
      compress raw block into page
      page.status = compressed
      page.valid_rows = 16
      fallback_block_ids[block_id] = -1
    else:
      keep raw_fallback
      page.valid_rows = 16
```

预期收益：

- 一个 decode token update 只处理当前 block，不再扫描 `num_blocks`。
- partial block 前 15 个 token 只做 raw write，不做 tile exponent/window 选择和 pack。
- 每 16 个 decode token 才做一次 full block compress。
- 对 decode E2E 最有直接收益。

风险：

- fallback pool 需要额外容纳 active partial blocks。
- 对 batch/concurrency 较高时，需要确认 pool 不因为 partial blocks 被占满。
- 如果 raw fallback finalized blocks 和 active partial blocks 共享同一个 pool，需要避免 active partial 占用导致不可压缩 block 没有空间。

风险控制：

- 增加 fallback pool 统计，区分 `partial_raw_blocks` 和 `final_raw_blocks`。
- 默认仍保持 `VLLM_BYTE_V2_SPARSE_FALLBACK_POOL_MIN_BLOCKS=512`。
- 如果 pool 耗尽，先回退到旧 generic path 或报出明确错误，不 silent corruption。

### 方案 2：prefill block-direct encode

用于首段 prompt prefill。

核心思路：

- prefill attention 已经复用 raw FlashAttention fast path，cache update 应直接从 raw K/V 输入按 physical block 编码，不经过 full-cache init/mark/compress。
- 对连续 slot 的 full block，one CTA per touched physical block，直接从 `key/value[token_idx]` 读取 16 rows 并编码到 ByteV2 page。
- 对 tail partial block，按方案 1 写入 raw fallback pool，等待后续 decode 补满后再压缩。

prefill fast path 条件：

```text
num_tokens >= 16
slot_mapping 基本按 token 顺序覆盖连续 slot
block_size == 16
compressed-only page layout
sparse fallback pool available
```

如果条件不满足，先回退旧 generic path，保证 correctness。

kernel 设计：

```text
byte_v2_prefill_encode_blocks_kernel
grid: one CTA per candidate 16-token group

group_start = group_idx * 16
inspect slot_mapping[group_start : group_start + 16]
if 16 rows all valid and map to same physical block offsets 0..15:
  encode full block directly from key/value tensors
else:
  mark as partial/irregular
```

partial/irregular 处理：

- 第一版可以回退旧 generic path。
- 第二版增加 `byte_v2_prefill_write_partial_kernel`，把 partial rows 写入 fallback pool。

预期收益：

- prefill 阶段压缩工作量从 `num_blocks` 降到 `touched_blocks`。
- 删除 `valid_rows/touched_flags/block_token_indices/raw_staging` 大部分临时分配。
- 避免 block_token_indices + second pass。

风险：

- 多请求混合 prefill、chunked prefill、prefix cache continuation 的 slot_mapping 可能不是简单连续。
- 需要严格检测 fast path 条件，不满足就回退。

### 方案 3：压缩函数并行化

当前 `byte_v2_store_compressed_block()` 和相关 helper 里，有大量 per tile 串行逻辑。后续如果方案 1/2 后 cache update 仍然偏慢，再做此项。

优化方向：

- 一个 CTA 处理一个 physical block。
- 一个 warp 或 half-warp 处理一个 16x16 tile。
- 用 warp-level reduction 计算 exponent min/max/base。
- 并行写 low bytes 和 packed sign/delta code。
- `byte_v2_raw_block_is_compressible()` 和 `byte_v2_store_compressed_block()` 合并，避免两次扫描 raw block。

保留标准：

- decode fast path 已经上线后，full block compress kernel 仍是 profile top 项。
- 对 prefill block encode 或每 16-token finalize 有稳定收益。

## 瓶颈二：decode attention

### 根因

当前最优 decode 路径通常是：

```text
byte_v2_paged_decode_attention
  allocate output [num_decode_tokens, num_heads, head_size_v]
  maybe allocate split_partial_output
  byte_v2_paged_decode_attention_gqa_wmma_split_stage1_kernel
  byte_v2_paged_decode_attention_split_reduce_kernel
  Python backend copy decode_out -> vLLM output
```

stage1 kernel 内部：

```text
one CTA per (request, kv_head, split)
load Q group into shared
for each logical page in split:
  load/decode K tile into shared
  load/decode V tile into shared
  WMMA QK
  tid==0 serial online softmax + p_shared fill
  WMMA PV
write FP32 partial output + LSE
```

主要问题：

- Llama-3-8B GQA 常见 `q_per_kv=4`，但 WMMA 计算 16 Q rows，只有前 4 行有效，tensor core 算力利用率低。
- `byte_v2_load_kv_bits()` 在每个 K/V element 上检查 page status 和 fallback metadata，分支在主循环中重复执行。
- split-K stage1 写 FP32 partial workspace，reduce kernel 再读回来，增加 global memory traffic。
- `p_shared` 和 online softmax 主要由 `tid == 0` 串行完成。
- custom op 内部分配 output，backend 再 copy 到 vLLM output buffer，有额外分配和 copy。

当前 profile：

```text
stage1: 148.582 ms
reduce:   6.569 ms
total:  155.151 ms
raw attention reference: 67.893 ms
```

### 目标

近期目标：

- p2048 d32 profile 中，ByteV2 decode attention 从 155 ms 降到 90 ms 以下。
- decode-only p2048 benchmark 中，ByteV2 attention kernel 至少提升 30%。

理想目标：

- 当前 12-bit format 下，decode attention 接近 raw attention 1.0x-1.3x。
- 后续如果引入更高压缩率 format，再争取超过 raw。

### 方案 1：decode op direct-output

状态：已实验，不保留。Step 27 p2048/d32 A/B 中 direct-output off/on 分别为
29.149/29.148 output tok/s，收益在噪声内。后续不再优先做 output copy 级优化，
应集中在 decode attention stage1 的 K/V decode、shared layout、split-K partial
和 fallback metadata 成本。

先做低风险优化。

新增 native op：

```text
byte_v2_paged_decode_attention_out(
    query,
    kv_cache,
    block_table,
    seq_lens,
    output,
    scale,
    block_size,
    num_kv_heads,
    head_size,
    head_size_v,
    page_size_bytes,
    fallback_pool,
    fallback_block_ids,
    optional partial_workspace,
)
```

改造后 backend：

```text
if output.ndim == 3:
  out_view = output[:num_decode_tokens]
else:
  out_view = output[:num_decode_tokens].view(num_decode_tokens, num_heads, head_size_v)
ops.byte_v2_paged_decode_attention_out(..., out_view, ...)
```

收益：

- 删除 op 内部 `output = torch::stable::empty(...)`。
- 删除 Python backend 的 `output.copy_(decode_out)`。
- 更适合 cudagraph，因为输出 buffer 由 vLLM 管理。

风险：

- output 可能是 flattened `[tokens, hidden]`，需要在 Python 侧传 contiguous/view 兼容的 tensor，或 native op 支持 stride。
- 需要保留旧 op 供 CPU/reference tests 使用。

保留标准：

- 单测通过。
- e2e 至少不回退。
- Nsight 中 decode output copy 消失。

### 方案 2：hoist page status/fallback 分支，拆 compressed/raw loader

当前每个 element 都调用通用 `byte_v2_load_kv_bits()`，会重复判断：

```text
page.status == raw_fallback?
page.status == compressed?
fallback_block_ids?
tile fallback?
```

优化：

```text
for each page:
  status = page[status]
  valid_rows = page[valid_rows]

  if status == compressed:
    load compressed K/V with byte_v2_load_compressed_bits_fast()
  else if status == raw_fallback:
    fallback_slot = fallback_block_ids[physical_block]
    load raw K/V with byte_v2_load_raw_bits_from_fallback()
  else:
    skip/error-safe zero
```

新增 helper：

```text
byte_v2_load_compressed_k_bits_fast(page, row, kv_head, dim, ...)
byte_v2_load_compressed_v_bits_fast(page, row, kv_head, dim, ...)
byte_v2_load_raw_fallback_k_bits_fast(fallback_block, row, kv_head, dim, ...)
byte_v2_load_raw_fallback_v_bits_fast(fallback_block, row, kv_head, dim, ...)
```

收益：

- 主循环 element-level 分支减少。
- 对绝大多数 compressed page 使用专门 fast path。
- 对 partial raw fallback block 只在少数 page 走 raw path。

保留标准：

- decode-only p2048/p4096 benchmark 有稳定收益。
- fallback pool 非空和空两种情况正确。

### 方案 3：q_per_kv=4 专用 decode kernel

这是中等风险、高潜在收益方案。

问题：

- 当前 WMMA kernel 固定 16 Q rows，但 Llama-3-8B `q_per_kv=4`，实际只用 4 rows。
- QK/PV 都做了 16-row WMMA，计算和 shared memory 写回都有浪费。

候选实现 A：SIMT q4 kernel

```text
grid: one CTA per (request, kv_head, split)
threads: 128
for each page:
  decompress K/V tile once into shared
  four warp groups分别处理 q0/q1/q2/q3
  warp-level dot(Q,K)
  warp-level online softmax
  parallel PV accumulation
```

优点：

- 只计算 4 个 Q rows，不浪费 12 行。
- softmax 更容易按 q row 并行。
- 适合 q_per_kv=4 这种主模型场景。

缺点：

- 失去 tensor core QK/PV，是否更快需要实测。
- head_size=128 的 dot 需要设计好 lane 分工和 reduction。

候选实现 B：hybrid q4 WMMA

```text
仍用 WMMA，但把多个 request/kv_head/q group 打包进 16 rows
例如 4 个 head-group 拼成一个 16-row Q tile
```

优点：

- 保持 tensor core。
- 理论算力利用率更高。

缺点：

- block_table/seq_lens 不同，softmax state 不同，packing 复杂。
- batch=1 时难以凑满 4 个同长度 group。
- 对 vLLM 动态 batch 不友好。

建议先做候选 A，并通过 benchmark 决定是否保留。

保留标准：

- Llama-3-8B shape (`num_heads=32`, `num_kv_heads=8`, `head_size=128`) 下，decode-only p2048/p4096 至少提升 15%。
- p512 不回退超过 5%，否则用 seq_len heuristic 只在长 context 启用。

### 方案 4：split-K heuristic 和 reduce 优化

当前 split-K 的收益依赖：

- context length
- batch size
- active head groups
- q_per_kv
- head_size
- fallback ratio

先做 benchmark sweep：

```bash
CUDA_VISIBLE_DEVICES=2 .venv/bin/python benchmarks/kernels/benchmark_byte_v2_decode_kernel.py \
  --seq-lens 512,1024,2048,4096,8192 \
  --batch-sizes 1,2,4,8 \
  --split-k 1,2,4,8,16,32 \
  --num-runs 50 \
  --warmup-runs 10 \
  --output-json benchmarks/profiles/bytev2_decode_splitk_sweep_after_cache_fix.json
```

优化方向：

- 按 seq_len/active_head_groups 自动选择 split-K。
- 对 short context 禁用 split-K，避免 workspace/reduce overhead。
- 对 long context 保留 page-parallel，提高 occupancy。
- 如果 direct-output op 加入 partial workspace 参数，则让 backend 管理持久 partial workspace，减少 op 内部分配。

保留标准：

- heuristic 在 p512/p1024/p2048/p4096 都不明显回退。
- e2e p2048 d32 至少提升 3%。

### 方案 5：parallel softmax/p_shared fill

当前 tile softmax 由 `tid == 0` 串行处理：

```text
for q in q_per_kv:
  tile_max over valid_rows
  exp per row
  fill p_shared
```

优化：

- 用一个 warp 处理一个 q row。
- lane 0..15 对应 16 rows，做 tile max 和 denom reduction。
- lane 写 `p_shared[q, row]`。

收益预期：

- 单 tile softmax latency 降低。
- 对 q_per_kv=4、长 context 更明显。

保留标准：

- decode-only benchmark 有可测收益。
- 数值误差与当前 kernel 相当。

## 分阶段执行计划

### 阶段 0：补 cache update microbenchmark

新增：

- `benchmarks/kernels/benchmark_byte_v2_cache_update_kernel.py`

指标：

```text
mode: generic/current, decode_append, prefill_direct
num_blocks
num_tokens
prompt_len
decode_steps
fallback_pool_ratio
mean_us
median_us
p50/p90/p99_us
packed_blocks
fallback_used
max_abs_diff_decode_after_update
```

必须覆盖：

- prefill 2048 tokens。
- decode append 1 token。
- decode append 16 steps，确认第 16 步 finalize compress。
- fallback pool 3% + min 512。

验收：

- benchmark 本身不改变生产路径。
- 能对比当前 generic path 和新 fast path。

### 阶段 1：实现 decode partial block 延迟压缩

修改：

- `csrc/libtorch_stable/cache_kernels.cu`
  - 新增 `byte_v2_decode_append_cache_kernel`
  - 在 `byte_v2_reshape_and_cache` 中检测 decode fast path
- `tests/v1/attention/test_byte_v2_ops.py`
  - 新增 partial raw fallback append 测试
  - 新增 16-step finalize compress 测试
  - 新增 pool exhausted error 测试

fast path 条件第一版：

```text
compressed_only_pages == true
has_sparse_fallback == true
num_tokens <= 8
slot_mapping contiguous enough for one token per active request
```

如果不满足，回退旧 generic path。

测试命令：

```bash
CUDA_VISIBLE_DEVICES=2 .venv/bin/python -m pytest \
  tests/v1/attention/test_byte_v2_ops.py \
  tests/v1/attention/test_byte_v2_decode.py \
  -q
```

benchmark：

```bash
CUDA_VISIBLE_DEVICES=2 .venv/bin/python benchmarks/kernels/benchmark_byte_v2_cache_update_kernel.py \
  --mode decode_append \
  --num-blocks 8192 \
  --decode-steps 32 \
  --fallback-ratio 0.03 \
  --output-json benchmarks/profiles/bytev2_cache_update_decode_append_v1.json
```

E2E：

```bash
CUDA_VISIBLE_DEVICES=2 .venv/bin/python benchmarks/benchmark_byte_v2_decode_e2e.py \
  --modes byte_v2_compressed_only \
  --prompt-len 2048 \
  --decode-lens 16,32 \
  --batch-size 1 \
  --num-runs 3 \
  --warmup-decode-len 8 \
  --output-json benchmarks/profiles/bytev2_e2e_after_decode_append_cache_v1.json
```

保留标准：

- 单测通过。
- decode append cache update median <= 50 us。
- E2E p2048 d32 ByteV2 至少提升 25%。
- fallback pool 不耗尽。

### 阶段 2：实现 prefill block-direct encode

修改：

- `csrc/libtorch_stable/cache_kernels.cu`
  - 新增 `byte_v2_prefill_encode_blocks_kernel`
  - 新增 fast path 检测和 fallback
- `tests/v1/attention/test_byte_v2_ops.py`
  - full block prefill direct encode
  - partial tail raw fallback
  - irregular slot_mapping fallback correctness

测试命令同阶段 1。

benchmark：

```bash
CUDA_VISIBLE_DEVICES=2 .venv/bin/python benchmarks/kernels/benchmark_byte_v2_cache_update_kernel.py \
  --mode prefill \
  --prompt-lens 512,1024,2048,4096 \
  --fallback-ratio 0.03 \
  --output-json benchmarks/profiles/bytev2_cache_update_prefill_direct_v1.json
```

保留标准：

- prefill cache update 相比当前 generic path 至少提升 5x。
- E2E p2048 d32 ByteV2 继续提升。
- 不破坏 raw FlashAttention prefill attention fast path。

### 阶段 3：decode direct-output op

状态：已实验，不保留。

修改：

- `csrc/libtorch_stable/cache_kernels.cu`
  - 新增 native out variant，内部复用 launch 逻辑
- `vllm/_custom_ops.py`
  - 新增 `byte_v2_paged_decode_attention_out`
- `vllm/v1/attention/backends/byte_v2_attn.py`
  - decode 直接写入 vLLM output buffer
- `tests/v1/attention/test_byte_v2_decode.py`
  - out op correctness
  - flattened output view correctness

测试命令：

```bash
CUDA_VISIBLE_DEVICES=2 .venv/bin/python -m pytest \
  tests/v1/attention/test_byte_v2_decode.py \
  tests/v1/attention/test_byte_v2_backend.py \
  -q
```

保留标准：

- e2e 不回退。
- Nsight 中 backend copy 消失。
- cudagraph capture/replay 正常。

### 阶段 4：decode compressed fast loader

修改：

- `csrc/libtorch_stable/cache_kernels.cu`
  - 增加 compressed/raw fallback 分离 loader
  - 在 WMMA split stage1 和 non-split kernel 中 hoist page status

测试：

```bash
CUDA_VISIBLE_DEVICES=2 .venv/bin/python -m pytest \
  tests/v1/attention/test_byte_v2_decode.py \
  tests/v1/attention/test_byte_v2_ops.py \
  -q
```

benchmark：

```bash
CUDA_VISIBLE_DEVICES=2 .venv/bin/python benchmarks/kernels/benchmark_byte_v2_decode_kernel.py \
  --seq-lens 512,1024,2048,4096 \
  --batch-sizes 1 \
  --split-k 0,1,2,4,8,16 \
  --num-runs 50 \
  --warmup-runs 10 \
  --output-json benchmarks/profiles/bytev2_decode_fast_loader_v1.json
```

保留标准：

- p2048/p4096 decode-only 至少提升 8%。
- fallback ratio 0.03 下正确。

### 阶段 5：q_per_kv=4 专用 kernel 实验

修改：

- `csrc/libtorch_stable/cache_kernels.cu`
  - 新增 `byte_v2_paged_decode_attention_gqa_q4_kernel`
  - 通过 env 或 shape heuristic 启用

建议 env：

```text
VLLM_BYTE_V2_DECODE_VARIANT=auto|wmma|q4_simt
```

测试和 benchmark 同阶段 4。

保留标准：

- Llama-3-8B shape 下 p2048/p4096 decode-only 至少提升 15%。
- p512 回退 <= 5%，否则只在 `seq_len >= 2048` 启用。
- E2E p2048 d32 至少提升 5%。

### 阶段 6：split-K heuristic 和 softmax 并行化

执行顺序：

1. 先基于阶段 1-5 后的新 kernel 跑 split-K sweep。
2. 更新 auto split heuristic。
3. 再实验 parallel softmax/p_shared fill。

保留标准：

- split-K heuristic 对短/长 context 都稳定。
- parallel softmax 对 decode-only 有可测收益，否则不保留。

## Nsight profile 验收

每个阶段完成后至少跑一次 p2048 d32 profile。

raw prefix-off 参考：

```bash
CUDA_VISIBLE_DEVICES=2 VLLM_ENABLE_V1_MULTIPROCESSING=0 \
nsys profile --trace=cuda,nvtx --cuda-graph-trace=node \
  --cuda-event-trace=false --force-overwrite=true --stats=false \
  --output benchmarks/profiles/raw_p2048_d32_reference_singleproc_node \
  .venv/bin/python benchmarks/benchmark_byte_v2_decode_e2e.py \
  --modes raw \
  --prompt-len 2048 \
  --decode-lens 32 \
  --batch-size 1 \
  --num-runs 1 \
  --warmup-decode-len 8 \
  --disable-prefix-caching \
  --output-json benchmarks/profiles/raw_p2048_d32_reference_singleproc_node.json
```

ByteV2：

```bash
CUDA_VISIBLE_DEVICES=2 VLLM_ENABLE_V1_MULTIPROCESSING=0 \
VLLM_BYTE_V2_USE_NATIVE_KERNELS=1 \
VLLM_BYTE_V2_SPARSE_FALLBACK_POOL_RATIO=0.03 \
VLLM_BYTE_V2_SPARSE_FALLBACK_POOL_MIN_BLOCKS=512 \
VLLM_BYTE_V2_DECODE_SPLIT_K=0 \
nsys profile --trace=cuda,nvtx --cuda-graph-trace=node \
  --cuda-event-trace=false --force-overwrite=true --stats=false \
  --output benchmarks/profiles/bytev2_p2048_d32_after_change_singleproc_node \
  .venv/bin/python benchmarks/benchmark_byte_v2_decode_e2e.py \
  --modes byte_v2_compressed_only \
  --prompt-len 2048 \
  --decode-lens 32 \
  --batch-size 1 \
  --num-runs 1 \
  --warmup-decode-len 8 \
  --output-json benchmarks/profiles/bytev2_p2048_d32_after_change_singleproc_node.json
```

导出：

```bash
nsys stats --force-export=true --report cuda_gpu_kern_sum --format csv \
  --output benchmarks/profiles/bytev2_p2048_d32_after_change_cuda_kernels \
  benchmarks/profiles/bytev2_p2048_d32_after_change_singleproc_node.nsys-rep
```

重点检查：

- `byte_v2_compress_touched_blocks_kernel` 是否从 top1 消失或明显下降。
- 新 decode append kernel 的 total/avg 时间。
- `byte_v2_paged_decode_attention_gqa_wmma_split_stage1_kernel` 是否下降。
- `byte_v2_paged_decode_attention_split_reduce_kernel` 是否下降或保持很小。
- `cudaMemcpyAsync`/`cudaEventSynchronize` 调用数是否异常增加。

## Step 27 后的 decode stage1 优化方向

Step 27 已确认 direct-output decode op 对 p2048/d32 没有可测收益，实验代码已删除。
当前 ByteV2 自身主要瓶颈重新收敛到 decode attention stage1：

```text
decode attention stage1 + reduce = 143.96 ms
decode append cache update       =   6.38 ms
deferred error record            =   1.21 ms
prefill direct encode            =   1.21 ms
```

因此后续不应继续投入 output copy 级优化，而应按
`docs/design/byte_v2_kernel_optimization_experiments_zh.md` 的 Step 28 执行：

1. page-level no-fallback bitset，减少 compressed hot path 的 fallback metadata 成本。
2. compressed-only K/V decoder v2，继续降低 K/V decode 指令数。
3. persistent partial workspace，降低 split-K partial allocator/runtime 成本。
4. 独立 WMMA fragment swizzle microkernel，先证明 shared/WMMA layout 有收益。
5. 若 microkernel 有收益，再做独立 CUTE/CUTLASS-style stage1 variant。

明确不再重复的方向：

- direct-output decode op。
- 简单 K shared stride padding。
- row-major K shared + WMMA B col-major。
- 单独 raw-fallback-only stage1 kernel。

## 最终阶段验收目标

短期目标：

| 指标 | 当前 | 阶段 1-2 目标 | 阶段 3-6 目标 |
|---|---:|---:|---:|
| ByteV2 p2048 d32 output tok/s | 9.00 | >= 15 | >= 20 |
| ByteV2 / raw prefix-off | 33.2% | >= 55% | >= 75% |
| cache update profiled time | 3088 ms | <= 300 ms | <= 150 ms |
| decode attention profiled time | 155 ms | <= 155 ms | <= 90 ms |
| fallback pool exhausted | false | false | false |

中期目标：

- 当前 12-bit format 下 ByteV2 接近 raw prefix-off 的 70-90%。
- 如果要超过 raw 单请求性能，需要结合 `byte_v2_outperform_raw_zh.md` 中的高压缩 performance format，否则 1.28x 左右的有效压缩率很难覆盖解压和调度开销。

## 回滚规则

每个阶段都必须能独立回滚。

不保留的条件：

- correctness 单测失败。
- cudagraph capture/replay 失败。
- p2048 d32 E2E 没有稳定提升，且 decode-only/cache-update microbenchmark 也没有对应收益。
- fallback pool 更容易耗尽。
- 引入额外 D2H sync。
- 代码复杂度显著增加，但收益小于 2%。

建议每个阶段提交前保留：

- 单测输出。
- microbenchmark JSON。
- E2E JSON。
- 如涉及 kernel，至少一份 Nsight kernel CSV。
