# ByteV2 Payload Layout V3 设计方案

## 1. 背景

当前 ByteV2 的主要性能瓶颈已经不在 prefill，也不在 reduce，而在 decode
attention stage1 的 K/V load/decode 路径。已有 profile 显示，当前 CUTE metadata
stage1 中 `load/decode K/V` 占 full stage1 的大部分时间；进一步 early-exit
拆分显示，payload read 本身明显高于 register bit decode。

当前 compressed tile 格式是：

```text
ByteV2 V1 tile:
  base exponent:       1B
  fallback/status:     1B
  low bytes:         256B
  packed exp code:   128B
  total:             386B
```

这个格式容量效率较高，但对 GPU decode kernel 不友好：

1. 每个 tile 是 `386B`，不是 `128B/256B/512B` 对齐大小。
2. `low[256]` 和 `code[128]` 是两段分离 stream。
3. `base/status` 嵌在每个 tile 内，metadata 读取随 tile 分散。
4. K 和 V 使用同一物理 tile 格式，但 decode 时 K/V 的消费顺序不同。
5. 当前计算推进粒度过于贴近 `16-token page`，没有充分模仿
   FlashAttention-style 的大 KV compute tile 和 pipeline。

V3 的目标是只改变 compressed page 的物理排布，不改变 ByteV2 的数值编码语义。

## 2. 设计目标

V3 payload layout 需要满足以下目标：

1. **保留 16x16 codec tile**
   - `16 tokens x 16 dims` 继续作为压缩和 fallback 粒度。
   - 该粒度匹配 vLLM `block_size=16`，也匹配 WMMA `m16n16k16`。

2. **让 tile payload 128B 对齐**
   - 单 tile compressed payload 固定为 `384B = 3 * 128B`。
   - 避免当前 `386B` stride 造成 tile 起点不断偏移 cache sector 边界。

3. **metadata SoA 化**
   - `base/status/fallback mask/outlier mask` 从 tile payload 中移出。
   - 对每个 `kv_head` 聚合成连续 metadata block。

4. **lane-striped payload**
   - 按 warp lane 访问顺序组织 `low0/low1/code`。
   - 让一个 warp 对一个 stripe 执行连续 32B load。

5. **K/V 分开物理顺序**
   - K 按 QK/WMMA-B 消费顺序存储。
   - V 按 PV/row-major 消费顺序存储。

6. **为后续 FlashAttention-style compute tile 做准备**
   - codec tile 仍是 `16x16`。
   - compute tile 后续可以聚合 `2` 或 `4` 个 page，即 `BLOCK_N=32/64`。

## 3. 当前 V1 和 V3 对比

以 Llama-3 8B 典型形状为例：

```text
block_size = 16
num_kv_heads = 8
head_size = 128
head_size_v = 128
k_dim_tiles = 8
v_dim_tiles = 8
total_tiles = 8 * (8 + 8) = 128
```

V1 page 大小：

```text
page_header = 16B
tile_payload = 386B
total = 16 + 128 * 386 = 49424B
```

V3 page 大小：

```text
page_header_v3 = 128B
kv_head_meta = 8 * 32B = 256B
K payload = 8 * 8 * 384B = 24576B
V payload = 8 * 8 * 384B = 24576B
total = 128 + 256 + 24576 + 24576 = 49536B
```

相比 V1，V3 增加：

```text
49536 - 49424 = 112B / page
```

raw BF16 block 大小：

```text
16 * 8 * (128 + 128) * 2 = 65536B
```

V3 压缩比：

```text
49536 / 65536 = 75.59%
```

因此 V3 基本保持当前压缩率，但显著改善 payload 读取形态。

## 4. Page V3 总体布局

推荐 page layout：

```text
ByteV2 Page V3:

[0, 128)
  PageHeaderV3, 128B

[128, 384)
  KvHeadMetaV3[8], 8 * 32B

[384, 24960)
  K payload: [kv_head][k_dim_tile][384B]

[24960, 49536)
  V payload: [kv_head][v_dim_tile][384B]
```

通用公式：

```text
header_bytes = 128
kv_head_meta_bytes = align_up(num_kv_heads * 32, 128)
tile_payload_bytes_v3 = 384

k_payload_offset = header_bytes + kv_head_meta_bytes
k_payload_bytes = num_kv_heads * k_dim_tiles * tile_payload_bytes_v3

v_payload_offset = k_payload_offset + k_payload_bytes
v_payload_bytes = num_kv_heads * v_dim_tiles * tile_payload_bytes_v3

page_size_v3 = v_payload_offset + v_payload_bytes
```

首版实现可以只支持 Llama-3 8B 形状：

```text
num_kv_heads = 8
head_size = 128
head_size_v = 128
block_size = 16
```

后续再把公式推广到更多模型。

## 5. PageHeaderV3

V3 header 固定为 128B，前两个字段保持和旧版本兼容：

```text
offset 0:
  page_status

offset 1:
  valid_rows

offset 2:
  layout_version = 3

offset 3:
  flags

offset 4..7:
  reserved

offset 8..11:
  payload_base_offset = 384

offset 12..15:
  k_payload_offset = 384

offset 16..19:
  v_payload_offset = 24960

offset 20..23:
  tile_payload_bytes = 384

offset 24..27:
  kv_head_meta_offset = 128

offset 28..31:
  kv_head_meta_bytes = 256

offset 32..127:
  reserved
```

建议 flags：

```text
bit 0: has_tile_fallback_metadata
bit 1: has_outlier_metadata
bit 2: compressed_only
bit 3: reserved
```

旧 kernel 看到 `layout_version != 3` 时继续走 V1 path。V3 kernel 要求
`layout_version == 3`，否则直接回退或报 deferred error。

## 6. KvHeadMetaV3

每个 `kv_head` 使用 32B metadata：

```text
KvHeadMetaV3, 32B:

offset 0..7:
  k_base[8]

offset 8..15:
  v_base[8]

offset 16..17:
  k_fallback_mask

offset 18..19:
  v_fallback_mask

offset 20..21:
  k_outlier_mask

offset 22..23:
  v_outlier_mask

offset 24..27:
  reserved0

offset 28..31:
  reserved1
```

其中：

```text
k_base[i] / v_base[i]:
  第 i 个 dim tile 的 base exponent

k_fallback_mask bit i:
  K dim_tile i 是否使用 tile-level raw fallback

v_fallback_mask bit i:
  V dim_tile i 是否使用 tile-level raw fallback

k_outlier_mask / v_outlier_mask:
  后续 outlier arena 或 tile outlier bitmap 使用
```

这样 decode stage1 对一个 page 和一个 kv_head 只需要读取一个连续 32B metadata
block，而不是对每个 tile 分散读取 `base/status`。

## 7. TilePayloadV3

V3 的 tile payload 固定为 384B，不再包含 base/status。

一个 tile 仍包含 256 个元素，两个元素组成一个 pair，因此共有 128 个 pair。
V3 将 128 个 pair 分成 4 个 stripe，每个 stripe 覆盖 32 个 pair。

```text
TilePayloadV3, 384B:

stripe0, pair 0..31:
  low0[32]
  low1[32]
  code[32]

stripe1, pair 32..63:
  low0[32]
  low1[32]
  code[32]

stripe2, pair 64..95:
  low0[32]
  low1[32]
  code[32]

stripe3, pair 96..127:
  low0[32]
  low1[32]
  code[32]
```

每个 stripe：

```text
32B + 32B + 32B = 96B
```

每个 tile：

```text
4 * 96B = 384B
```

warp lane 访问方式：

```cpp
const int stripe = pair_idx >> 5;
const int lane = pair_idx & 31;
const uint8_t* stripe_ptr = tile_ptr + stripe * 96;

uint8_t low0 = stripe_ptr[lane];
uint8_t low1 = stripe_ptr[32 + lane];
uint8_t code = stripe_ptr[64 + lane];
```

这样一个 warp 可以对 `low0/low1/code` 分别发起连续 32B 读取，避免 V1 中
`low_ptr + elem0` 和 `packed_ptr + pair_idx` 两个分离 stream。

## 8. K/V 物理顺序

V3 明确区分 K 和 V 的 tile element order。

### 8.1 K tile order

K 用于 QK，当前 CUTE/WMMA path 消费的是 transposed shared layout：

```text
K logical tile:
  row = token row, 0..15
  dim = dim_in_tile, 0..15

K physical elem order:
  elem = dim * 16 + row
```

也就是 dim-major，匹配 WMMA-B / QK 消费顺序。

### 8.2 V tile order

V 用于 PV，更自然的消费顺序是 row-major：

```text
V logical tile:
  row = token row, 0..15
  dim = dim_in_tile, 0..15

V physical elem order:
  elem = row * 16 + dim
```

这样 V decode 后更接近 PV 阶段需要的 shared layout。

## 9. 与 FlashAttention Tile 的关系

V3 不建议把 ByteV2 codec tile 直接改成 FlashAttention 的 compute tile。

FlashAttention-style kernel 的高效点通常来自：

```text
更大的 compute tile
online softmax
shared/register blocking
load/compute pipeline
减少中间 attention matrix 落地
```

而 ByteV2 的 codec tile 还承担压缩语义：

```text
base exponent 粒度
fallback 粒度
outlier 统计粒度
```

因此 V3 的原则是：

```text
codec tile:
  16 tokens x 16 dims，保持不变

compute tile:
  聚合多个 codec tile/page
```

推荐下一阶段 compute tile：

```text
BLOCK_N = 32:
  一次处理 2 个 ByteV2 page

BLOCK_N = 64:
  一次处理 4 个 ByteV2 page
```

首版建议先做 `BLOCK_N=32`，如果 shared memory、register 和 occupancy 可接受，
再尝试 `BLOCK_N=64`。

## 10. Decode Loader V3 伪代码

以 no-outlier compressed tile fast path 为例：

```cpp
uint8_t* page = kv_cache + physical_block * page_size_v3;

PageHeaderV3 header = load_header(page);
KvHeadMetaV3 meta = load_kv_head_meta(page, kv_head);

uint8_t k_base = meta.k_base[dim_tile];
bool k_fallback = meta.k_fallback_mask & (1u << dim_tile);

if (!k_fallback) {
  const uint8_t* tile =
      page + k_payload_offset
      + (kv_head * k_dim_tiles + dim_tile) * 384;

  for each stripe:
    low0_vec = load 32B contiguous
    low1_vec = load 32B contiguous
    code_vec = load 32B contiguous
    decode pairs
    write K shared in WMMA-B order
} else {
  load raw fallback tile
}
```

V loader 相同，但使用 `v_payload_offset` 和 V row-major element order。

## 11. Encode Path V3

V3 encode 需要从 raw K/V block 写出：

```text
PageHeaderV3
KvHeadMetaV3
K payload V3
V payload V3
```

编码顺序建议：

1. 一个 CTA 负责一个 physical block 或一个 block 内的 kv_head。
2. 先扫描每个 K/V dim tile，计算 base exponent 和 fallback/outlier 状态。
3. 写 `KvHeadMetaV3`。
4. 写 K payload，使用 K physical element order。
5. 写 V payload，使用 V physical element order。

首版实现应限制在：

```text
block_size = 16
head_size = 128
head_size_v = 128
num_kv_heads = 8
compressed-only
tile fallback supported
no outlier arena
```

## 12. 兼容和开关

建议新增实验开关：

```text
VLLM_BYTE_V2_PAYLOAD_LAYOUT=v1
VLLM_BYTE_V2_PAYLOAD_LAYOUT=v3
```

默认仍然使用 V1。只有满足以下条件时才允许 V3：

```text
kv_cache_dtype = byte_v2
block_size = 16
head_size = 128
head_size_v = 128
num_kv_heads = 8
compressed_only = true
outlier_arena = false
```

如果 `VLLM_BYTE_V2_PAYLOAD_LAYOUT=v3` 但形状不支持，应 fail closed，避免静默走错
layout。

## 13. 测试计划

### 13.1 Python reference 测试

新增或扩展测试：

```text
tests/v1/attention/test_byte_v2_codec.py
tests/v1/attention/test_byte_v2_layout.py
```

覆盖：

```text
V1 encode/decode 结果不变
V3 encode/decode 与 raw BF16 bits 对齐
K physical order 正确
V physical order 正确
fallback mask 正确
page size 和 offset 正确
```

### 13.2 CUDA correctness 测试

新增 decode op 测试：

```text
test_byte_v2_paged_decode_attention_op_payload_v3_cuda
test_byte_v2_paged_decode_attention_op_payload_v3_tile_fallback_cuda
```

对比：

```text
V3 CUDA decode output
V1 CUDA decode output
raw paged attention reference
```

### 13.3 Microbench

先不接 E2E，先跑 decode-only early-exit：

```text
mode5: metadata/status traversal
mode6: metadata + payload read
mode7: metadata + payload read + register decode
mode0: full stage1
```

workload：

```text
p512/b4
p2048/b1
p4096/b4
```

### 13.4 NCU 指标

至少对比：

```text
duration
integer instructions
memory instructions
long scoreboard
global sectors/request
uncoalesced global sectors
shared excessive wavefronts
registers/thread
eligible warps/scheduler
```

## 14. 保留门槛

V3 只有满足以下门槛才接入 production path：

```text
decode-only mode6 - mode5 payload read 下降 >= 8%
mode0 full stage1 下降 >= 5%
p4096/b4 E2E 提升 >= 3%
p512/b4 E2E 不退化超过 1%
NCU long scoreboard 或 global sector 指标有同步改善
```

如果只看到 tok/s 小幅波动，但 NCU 没有对应改善，不保留。

## 15. 风险

1. **page size 增加**
   - Llama-3 8B 下增加 112B/page，容量影响很小。

2. **encode 更复杂**
   - 需要额外写 `KvHeadMetaV3`，并把 low/code 写成 stripe layout。

3. **V3 对更多模型泛化不足**
   - 首版只面向 h128/q_per_kv=4/num_kv_heads=8。

4. **只改 payload layout 未必足够**
   - 如果 stage1 主要瓶颈来自 compute skeleton，而不是 payload read，V3 收益可能低于预期。

5. **tile fallback/outlier 支持要分阶段**
   - 首版支持 tile fallback mask。
   - outlier arena 不进入首版性能默认路径。

## 16. 分阶段实施计划

### Phase 1: Reference layout

实现 Python reference：

```text
ByteV2PageLayoutV3
pack_byte_v2_kv_block_to_page_v3
unpack_byte_v2_kv_block_from_page_v3
```

目标：

```text
pytest CPU/reference 通过
V3 page offset 和 V1 raw value 对齐
```

### Phase 1/2 实验记录 2026-06-15

本轮在保留 V1 默认布局的前提下，新增了 V3 reference layout 和
decode-page WMMA microbench 的 V3 读取路径。V3 尚未接入 production cache
update / production decode stage1。

已完成：

```text
ByteV2PageLayoutV3
pack_byte_v2_kv_block_to_page_v3
unpack_byte_v2_kv_block_from_page_v3
byte_v2_decode_page_wmma_microbench 对 V3 page header 的自动识别
benchmark_byte_v2_decode_page_wmma_microbench.py --payload-layout v1|v3
```

正确性：

```text
.venv/bin/python -m pytest tests/v1/attention/test_byte_v2_layout.py -q
结果：14 passed

V1/V3 decode-page WMMA microbench 输出：
bit_equal=True
max_abs_diff=0.0
```

实验环境：

```text
GPU: NVIDIA A40
workload:
  num_pages=256
  num_kv_heads=8
  kv_head=0
  head_size=head_size_v=128
  repeat_count=32
  warmup=20
  iterations=100
```

结果：

| layout | page size | median | per page-repeat | 对比 V1 |
| --- | ---: | ---: | ---: | ---: |
| V1 | 49424B | 316.42us | 0.038625us | baseline |
| V3 | 49536B | 337.92us | 0.041250us | -6.8% |

结论：

```text
当前 V3 lane-striped payload 在 decode-page WMMA microbench 中退化，
没有达到 mode0/full microbench 提升 >= 5% 的保留门槛。
```

初步原因：

1. 当前 V3 decoder 仍是每个线程标量读取 `low0/low1/code`，没有真正使用
   warp-level vectorized 32B load 或 cp.async。
2. V1 当前等价于每 pair 读取一个 low u16 加一个 packed byte；V3 当前变成三个
   scalar byte stream，虽然 stream 更连续，但指令数没有减少。
3. microbench 的 compute skeleton 仍是原来的 16-page 粒度 WMMA 路径，V3 只换
   payload physical layout，没有引入 BLOCK_N=32/64 聚合或 producer/consumer
   pipeline。

保留策略：

```text
保留 V3 reference layout 和 microbench 实验入口；
不接入 production cache update / production decode 默认路径；
后续若继续尝试，必须先做真正的 warp-vectorized stripe loader 或 cp.async
loader，再重新评估。
```

### Phase 2: CUDA encode

实现 V3 direct encode：

```text
byte_v2_store_compressed_block_v3
byte_v2_direct_encode_pages_v3_kernel
```

目标：

```text
CUDA encode/decode reference 正确
不接 E2E
```

### Phase 3: Decode-only V3 loader

实现 V3 no-outlier compressed loader：

```text
byte_v2_decode_k_tile_v3_to_shared
byte_v2_decode_v_tile_v3_to_shared
```

目标：

```text
decode-only correctness 通过
early-exit mode5/6/7/mode0 可以对比 V1
```

### Phase 4: CUTE metadata stage1 V3 path

在当前最优 CUTE metadata kernel 上增加 V3 opt-in：

```text
VLLM_BYTE_V2_PAYLOAD_LAYOUT=v3
VLLM_BYTE_V2_DECODE_CUTE_STAGE1_AUTO=1
```

目标：

```text
mode6-mode5 下降 >= 8%
mode0 下降 >= 5%
```

### Phase 5: E2E A/B

在 Llama-3 8B 上对比：

```text
raw vLLM
ByteV2 V1 CUTE metadata
ByteV2 V3 CUTE metadata
```

workload：

```text
p512/b4/d128
p512/b4/d256
p4096/b4/d64
p4096/b8/d128
```

目标：

```text
p4096/b4 E2E >= +3%
p512/b4 不退化超过 1%
pool 不耗尽
输出 correctness smoke 通过
```

## 17. 当前实现状态和实验结果（2026-06-15）

### 17.1 已完成的 V3 opt-in production 路径

当前代码已经在保留 V1 默认路径的前提下，完成了 V3 的 opt-in 接入：

```text
VLLM_BYTE_V2_PAYLOAD_LAYOUT=v3
```

已接入部分：

1. Python runtime/spec/allocator：
   - `VLLM_BYTE_V2_PAYLOAD_LAYOUT=v1|v3`
   - `ByteV2FullAttentionSpec.payload_layout`
   - V3 page size 计算和 cache allocation
   - V3 fail-closed：只允许 compressed-only、block_size=16、num_kv_heads=8、
     head_size=head_size_v=128、无 outlier arena。

2. CPU/reference layout：
   - `ByteV2PageLayoutV3`
   - V3 pack/unpack reference
   - V3 layout roundtrip tests

3. Native CUDA cache update：
   - generic touched-block V3 encode
   - prefill direct encode V3 fastpath
   - decode append V3 fastpath
   - V3 tile fallback mask 写入

4. Native CUDA decode/decompress：
   - generic scalar compressed loader 支持 V3
   - CUTE metadata split-stage1 支持 V3 no-outlier/tile-fallback metadata
   - `byte_v2_decompress_cache_to_bf16` 支持 V3 page size

5. Benchmark/test：
   - decode page WMMA microbench 支持 `--payload-layout v1|v3`
   - E2E benchmark JSON 记录 `VLLM_BYTE_V2_PAYLOAD_LAYOUT`
   - CUDA correctness 覆盖：
     - V3 full prefill-direct encode -> decompress -> CUTE decode
     - V3 partial 15+1 decode-append finalize -> decompress

### 17.2 Correctness

已运行：

```bash
.venv/bin/python -m pytest tests/v1/test_byte_v2_kv_cache_spec.py \
  tests/v1/attention/test_byte_v2_layout.py -q

CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest \
  tests/v1/attention/test_byte_v2_ops.py::test_byte_v2_v3_native_cache_update_decompress_and_cute_decode_cuda \
  tests/v1/attention/test_byte_v2_ops.py::test_byte_v2_v3_decode_append_finalizes_partial_block_cuda \
  -q -s
```

结果：

```text
29 passed
2 passed
```

### 17.3 Decode page WMMA microbench

workload：

```text
num_pages=256
num_kv_heads=8
kv_head=0
repeat_count=32
warmup=20
iterations=100
GPU=NVIDIA A40
```

结果：

| layout | page bytes | median us | per page-repeat us | vs V1 |
| --- | ---: | ---: | ---: | ---: |
| V1 | 49424 | 316.42 | 0.03862 | 1.000x |
| V3 | 49536 | 337.92 | 0.04125 | 1.068x slower |

结论：

```text
V3 当前 loader 在 microbench 中仍比 V1 慢约 6.8%，没有达到 +5% 的保留门槛。
原因仍是 V3 只是改变 physical layout，decoder 仍是 scalar byte load，
没有真正做 warp-vectorized stripe load / cp.async / producer-consumer pipeline。
```

### 17.4 E2E A/B

workload：

```text
model=/mnt/sda1/yxz/byte_v2/Meta-Llama-3-8B-Instruct
prompt_len=512
batch_size=4
decode_len=64
num_runs=1
gpu_memory_utilization=0.75
VLLM_BYTE_V2_DECODE_PAGE_FASTPATH=1
VLLM_BYTE_V2_DECODE_TILE_FASTPATH=1
VLLM_BYTE_V2_DECODE_CUTE_STAGE1_AUTO=1
VLLM_BYTE_V2_SPARSE_FALLBACK_POOL_RATIO=0.03
VLLM_BYTE_V2_ENABLE_OUTLIER_ARENA=0
```

结果：

| mode | layout | output tok/s | elapsed s | pool exhausted |
| --- | --- | ---: | ---: | --- |
| raw vLLM | raw | 131.43 | 1.948 | false |
| ByteV2 compressed-only | V1 | 110.39 | 2.319 | false |
| ByteV2 compressed-only | V3, cache update slow path | 23.89 | 10.716 | false |
| ByteV2 compressed-only | V3, fast cache update | 110.23 | 2.322 | false |

关键结论：

```text
1. V3 初始 E2E 退化到 23.89 tok/s，不是 payload decode 本身导致，
   而是 prefill direct / decode append fast cache update 没有接 V3，
   导致 E2E 走 generic cache update slow path。

2. 接入 V3 prefill direct encode 和 decode append fastpath 后，
   V3 E2E 恢复到 110.23 tok/s，和 V1 的 110.39 tok/s 基本持平。

3. V3 当前没有超过 V1；microbench 已经显示 V3 loader 本身仍慢约 6.8%。
   因此 V3 可以作为 opt-in 实验路径保留，但不应设为默认性能路径。
```

### 17.5 保留/回滚策略

保留：

```text
V3 reference layout
V3 page-size/spec/allocator
V3 native cache update/decode/decompress opt-in path
V3 microbench and correctness tests
```

不默认启用：

```text
VLLM_BYTE_V2_PAYLOAD_LAYOUT 仍默认 v1
V3 不支持 outlier arena
V3 不作为 performance default
```

后续继续优化 V3 的必要门槛：

```text
decode-page WMMA microbench V3 >= V1 + 5%
或 detailed profile 显示 V3 stage1 load/decode 明显下降
否则不继续扩大 V3 production 面
```

### 17.6 Warp-vectorized stripe load 实验记录

本轮实现了 V3-only 的 warp-vectorized stripe loader：

```text
VLLM_BYTE_V2_DECODE_V3_WARP_STRIPE_LOAD=1
```

实现方式：

```text
每个 96B stripe:
  原路径：每个 lane 各自 scalar 读取 low0/low1/code 三个 byte
  新路径：lane 0..23 各读取一个 aligned uint32，共 24 个 u32 load，
          然后通过 __shfl_sync 把 low0/low1/code 分发到对应 lane
```

正确性：

```bash
CUDA_VISIBLE_DEVICES=0 VLLM_BYTE_V2_DECODE_V3_WARP_STRIPE_LOAD=1 \
  .venv/bin/python -m pytest \
  tests/v1/attention/test_byte_v2_ops.py::test_byte_v2_v3_native_cache_update_decompress_and_cute_decode_cuda \
  -q -s

结果：1 passed
```

decode-page WMMA microbench：

```text
workload:
  payload_layout=v3
  num_pages=256
  num_kv_heads=8
  kv_head=0
  repeat_count=32
  warmup=20
  iterations=100
```

| V3 loader | median us | per page-repeat us | 对比 scalar |
| --- | ---: | ---: | ---: |
| scalar byte load | 332.80 | 0.04063 | baseline |
| warp-stripe u32+shuffle | 377.86 | 0.04612 | -13.5% |

CUTE stage1 early-exit profile：

```text
workload:
  payload_layout=v3
  batch_size=4
  seq_len=512
  split_k=16
  fallback=0
  cute_stage1=on
```

| mode | 含义 | scalar median us | warp-stripe median us |
| --- | --- | ---: | ---: |
| mode5 | metadata/status traversal | 35.84 | 35.84 |
| mode6 | metadata + payload read | 60.42 | 73.73 |
| mode6 rerun | metadata + payload read | 63.49 | 75.78 |
| mode7 | metadata + payload read + register decode | 77.81 | 76.80 |
| mode0 | full decode kernel | 72.70 | 71.68 |

结论：

```text
1. warp-stripe loader 没有减少 mode6 的 payload read 时间；
   mode6-mode5 从约 24.6us/27.7us 退化到约 37.9us/39.9us。

2. decode-page full microbench 也明显退化，说明当前 u32+shuffle 方案
   并没有改善 V3 payload loader 的关键瓶颈。

3. full decode kernel mode0 约 1.4% 的小幅改善不应视为有效收益，
   因为它没有对应的 payload-read 指标改善，且可能来自调度噪声。

4. 该路径保持 opt-in/default-off，不进入默认 V3 performance path。
```

### 17.7 cp.async staging/double-buffer 实验记录

本轮实现了 V3-only 的 `cp.async` staging 路径：

```text
VLLM_BYTE_V2_DECODE_V3_CP_ASYNC_STAGE=1
```

实现范围：

```text
1. decode-page WMMA microbench 支持 V3 tile payload cp.async staging。
2. CUTE metadata split-stage1 支持 V3 cp.async staging。
3. 每个 CTA 使用两个 384B shared staging buffer。
4. 处理 K 时预取下一个 K dim-tile，同时解码当前 dim-tile。
5. 处理 V 时预取下一个 V dim-tile，同时解码当前 dim-tile。
6. fallback tile / outlier arena / 非 V3 layout 仍走旧路径。
```

该实现仍不是完整 FlashAttention producer/consumer pipeline。它只把
global-to-shared payload load 改成 `cp.async` 并做 dim-tile 级双缓冲，
后续 decode、shared 写入、WMMA/softmax/PV 仍保持当前 CUTE metadata
stage1 结构。

正确性 smoke：

```bash
CUDA_VISIBLE_DEVICES=0 \
  .venv/bin/python benchmarks/kernels/benchmark_byte_v2_decode_kernel.py \
  --payload-layout v3 --batch-size 2 --seq-len 256 --split-k 16 \
  --fallback-ratio 0.0 --cute-stage1 --v3-cp-async-stage \
  --num-runs 5 --warmup-runs 2

结果：max_abs_diff=0.0
```

decode-page WMMA microbench：

```text
workload:
  payload_layout=v3
  num_pages=256
  num_kv_heads=8
  kv_head=0
  repeat_count=32
  warmup=20
  iterations=100
```

| V3 loader | median us | per page-repeat us | 对比 scalar |
| --- | ---: | ---: | ---: |
| scalar byte load | 335.87 | 0.04100 | baseline |
| cp.async staging | 302.08 | 0.03688 | +10.1% |

CUTE split-stage1 decode-only：

```text
workload:
  payload_layout=v3
  batch_size=4
  seq_len=512
  split_k=16
  cute_stage1=on
  skip_correctness=on
  num_runs=200
  warmup_runs=40
```

| fallback | V3 loader | median us | output tok/s | 对比 scalar |
| ---: | --- | ---: | ---: | ---: |
| 0.00 | scalar | 71.68 | 55803.57 | baseline |
| 0.00 | cp.async staging | 65.54 | 61035.16 | +8.6% |
| 0.03 + tile fallback pool | scalar | 86.02 | 46502.98 | baseline |
| 0.03 + tile fallback pool | cp.async staging | 73.73 | 54253.47 | +14.3% |

E2E 状态：

```text
clean GPU0, gpu_memory_utilization=0.75, num_runs=1:
  raw:             131.41 output tok/s
  V3 scalar:       109.06 output tok/s
  V3 cp.async:      78.73 / 110.68 output tok/s

clean GPU0, gpu_memory_utilization=0.75, num_runs=3 median:
  raw:             131.37 output tok/s
  V3 scalar:       109.33 output tok/s
  V3 cp.async:     110.66 output tok/s

busy GPU0, gpu_memory_utilization=0.45:
  V3 scalar:        50.08 output tok/s
  V3 cp.async:      52.24 output tok/s
```

`num_runs=1` 的第一轮 cp.async 出现过明显异常低值，单次结果不适合
做保留/回滚判断。后续以 `num_runs=3` median 为准：cp.async staging
相比 V3 scalar E2E 约 `+1.2%`，相比 raw 约为 `84.2%`。busy GPU0 的
结果只作为功能 smoke 参考，不作为性能结论。

结论：

```text
1. cp.async staging 是目前 V3 payload 读取方向里第一个稳定降低
   decode-page 和 CUTE decode-only 时间的方案。

2. 相比 warp-stripe u32+shuffle，cp.async 明确改善了 payload load
   相关 microbench，因此值得保留为 opt-in 实验路径。

3. 该路径暂不默认开启。原因是 decode-only 收益明显，但 p512/b4/d64
   E2E 只有约 `+1.2%`，还不足以证明生产默认收益。默认前必须补齐：
   - 更多 decode_len / batch_size 的 raw / V3 scalar / V3 cp.async E2E A/B；
   - NCU 对比 global sectors、long scoreboard、eligible warps；
   - fallback tile 非零场景的正确性和性能复测。

4. 下一步如果继续做 FlashAttention-style pipeline，应在此基础上进一步
   拆成真正 producer/consumer 结构，而不是只增加语句级预取。
```

### 17.8 compressed-first outlier overlay 验证记录

本轮尝试把 V3 CUTE split-stage1 的 cp.async path 改成：

```text
1. 只要 page 是 V3 compressed payload 且没有 tile fallback mask，
   即使传入 outlier metadata，也先走 compressed payload cp.async decode。
2. 如果存在 outlier metadata，再在 shared memory 中对 K/V tile 做 overlay。
3. overlay 必须发生在 QK / PV 之前，避免 attention 看到被 clamp 的占位值。
```

实现后重新编译 CUDA 扩展：

```bash
uv pip install -e . --torch-backend=auto
```

编译成功。基础 outlier 单测通过：

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest \
  tests/v1/attention/test_byte_v2_ops.py::test_byte_v2_native_outlier_arena_prefill_direct_cuda \
  tests/v1/attention/test_byte_v2_ops.py::test_byte_v2_native_outlier_arena_decode_overlay_cuda \
  -q -s

结果：2 passed
```

decode kernel smoke / A-B：

```text
workload:
  payload_layout=v3
  batch_size=2
  seq_len=512
  split_k=16
  fallback_ratio=0.25
  fallback_pattern=single_outlier
  tile_fallback_pool=on
  outlier_arena allocated
  CUTE stage1=on
```

| variant | median us | output tok/s | tile fallback | outlier entries |
| --- | ---: | ---: | ---: | ---: |
| V3 CUTE scalar metadata | 86.02 | 23251.49 | 16 | 0 |
| V3 CUTE cp.async metadata | 72.70 | 27508.80 | 16 | 0 |
| V3 CUTE cp.async no-arena/no-fallback | 54.27 | 36851.42 | 0 | 0 |
| V3 CUTE cp.async outlier-arena encode | 58.37 | 34265.35 | 0 | 8 |

E2E：

```text
model=/mnt/sda1/yxz/byte_v2/Meta-Llama-3-8B-Instruct
prompt_len=512
batch_size=4
decode_len=64
num_runs=1
gpu_memory_utilization=0.75
payload_layout=v3
outlier_arena=off
```

| variant | raw output tok/s | ByteV2 output tok/s | ByteV2/raw |
| --- | ---: | ---: | ---: |
| V3 scalar | 131.41 | 109.23 | 83.1% |
| V3 cp.async | 131.39 | 110.35 | 84.0% |
| V3 cp.async + outlier_arena | 131.44 | 24.38 | 18.5% |

结论：

```text
1. 本轮 patch 让 V3 cp.async 在“metadata tensor 存在”的 CUTE stage1
   中不再被禁用；decode kernel microbench 的 metadata/tile-fallback case
   从 86.02 us 降到 72.70 us。

2. p512/b4/d64 E2E 中 V3 cp.async 比 V3 scalar 小幅提升
   109.23 -> 110.35 output tok/s，约 +1.0%，且 fallback pool 未耗尽。

3. 后续补齐了 V3 outlier arena encode 的最小路径：
   - Python spec / backend shape 不再拒绝 `V3 + outlier_arena`；
   - native prefill-direct host dispatch 不再排除 `V3 + outlier_arena`；
   - V3 prefill direct encode 可以把少量 window miss 写入 compact
     outlier arena，并把 compressed payload 中的 exponent clamp 到窗口内。

4. 编译后 kernel smoke 证明 encode 已打通：
   `outlier_entries=8`、`fallback_tiles=0`、`max_abs_diff=0.03125`。
   小 E2E smoke 也能完成，final stats 中
   `total_outlier_next_entry=855`、`any_outlier_exhausted=false`。

5. 但 p512/b4/d64 E2E 性能明显退化：
   no-arena V3 cp.async 约 `110.35 output tok/s`，outlier_arena on 只有
   `24.38 output tok/s`。因此 outlier arena 目前只能作为 correctness /
   capacity 实验路径，不能作为性能默认路径。

6. 已补做 `outlier_arena on/off` profile，结论见 17.9。退化主因不是
   decode attention stage1，而是 cache update 被 `V3 + outlier_arena`
   排除出 decode append fast path，落到 generic touched-block compression。

### 17.9 outlier_arena on 退化 profile

profile workload：

```text
model=/mnt/sda1/yxz/byte_v2/Meta-Llama-3-8B-Instruct
prompt_len=512
batch_size=4
decode_len=64
payload_layout=v3
CUTE stage1=on
V3 cp.async stage=on
sparse fallback ratio=0.03
```

E2E 复现：

| variant | output tok/s | elapsed s | assigned block fallback | outlier entries | outlier exhausted |
| --- | ---: | ---: | ---: | ---: | --- |
| V3 no-arena | 107.84 | 2.374 | 256 | 0 | false |
| V3 outlier_arena | 24.29 | 10.539 | 0 | 2692 | false |

为了让 NVTX range 和 CUDA kernels 在同一进程，额外用
`VLLM_ENABLE_V1_MULTIPROCESSING=0` 跑 nsys。measured range 的 kernel
汇总如下：

| group | no-arena total ms | arena-on total ms | delta |
| --- | ---: | ---: | ---: |
| GEMM | 1837.44 | 1842.30 | +4.86 |
| ByteV2 decode stage1 | 175.95 | 194.43 | +18.48 |
| ByteV2 decode reduce | 7.41 | 7.36 | -0.05 |
| ByteV2 decode append cache | 15.01 | 0.00 | -15.01 |
| ByteV2 init/mark cache update | 0.00 | 11.33 | +11.33 |
| ByteV2 compress touched blocks | 0.00 | 7697.90 | +7697.90 |

关键 kernel：

```text
no-arena:
  byte_v2_decode_append_cache_kernel
    2016 launches, 15.01 ms total, 7.45 us/launch

arena-on:
  byte_v2_compress_touched_blocks_kernel
    2016 launches, 7697.90 ms total, 3818.40 us/launch
```

因此 `outlier_arena on` 的主要退化来自 cache update 路径切换：

```cpp
use_decode_append_fast_path =
    compressed_only_pages && has_sparse_fallback &&
    (!use_v3_payload_layout || !has_outlier_arena) &&
    ...
```

当 `V3 + outlier_arena` 同时开启时，上述条件会禁用 decode append
fast path，后续走 generic slow path：

```text
byte_v2_init_cache_update_kernel
byte_v2_mark_touched_tokens_kernel
byte_v2_compress_touched_blocks_kernel
```

这个 slow path 每 token/layer 需要扫描 touched block、准备 raw staging、
必要时解压旧 block，再重新压缩整个 16-token block；在 decode 阶段它比
decode append fast path 慢约 `3818.40 / 7.45 = 512x` 每 launch。

stage1 early-exit microbench 也做了 arena on/off 对照：

| mode | 含义 | no-arena us | arena-on us | ratio |
| --- | --- | ---: | ---: | ---: |
| 5 | metadata/status traversal | 53.25 | 65.54 | 1.23x |
| 6 | metadata + payload read | 82.94 | 110.59 | 1.33x |
| 7 | metadata + payload read + register decode | 83.97 | 113.66 | 1.35x |
| 0 | full stage1 | 105.47 | 110.59 | 1.05x |

stage1 确实有 metadata / payload 额外成本，但 full stage1 只退化约 5%；
在 E2E 中 stage1 只增加约 18.5 ms，而 cache update 增加约 7.7 s。

后续修复优先级：

```text
P0:
  实现 V3 + outlier_arena decode append fast path。
  目标是在 byte_v2_decode_append_cache_kernel 内支持 outlier arena
  metadata / arena entry 写入，不能回退到 touched-block compression。

P1:
  对 decode append fast path 做 production-safe deferred error reporting，
  避免恢复每 token/layer host sync。

P2:
  再考虑优化 arena-on decode stage1 的 metadata/payload early-exit 成本。
  该项只有在 cache update 修复后才值得继续投入。
```

### 17.10 decode append fast path 支持 outlier_arena 的修复结果

本轮已补齐 `V3 + outlier_arena` 的 decode append fast path，不再让它
落到 generic touched-block compression。

主要代码变化：

```text
csrc/libtorch_stable/cache_kernels.cu
  byte_v2_finalize_raw_block_parallel()
    增加 outlier_arena / outlier_tile_meta / outlier_next_entry 参数。
    复用 prefill-direct 的 tile outlier 判定、arena entry 分配、meta/bitmap
    写入逻辑。

  byte_v2_decode_append_cache_kernel()
    增加 outlier arena 参数。
    finalizing 16-token block 时直接写 compressed V3 page + outlier metadata。
    清理旧 outlier meta/bitmap，避免 block 复用时读到 stale metadata。
    解压已有 compressed partial page 时先保留旧 metadata 并带 outlier
    overlay，解压完成后再清理 metadata，避免丢 outlier。

  host dispatch
    允许 V3 + outlier_arena 进入 decode append fast path。
    batched append 仍保留 capture / force / deferred-safe 条件。
```

新增单测：

```text
tests/v1/attention/test_byte_v2_ops.py
  test_byte_v2_v3_decode_append_writes_outlier_arena_cuda
  test_byte_v2_v3_decode_append_preserves_partial_outliers_cuda
```

验证命令：

```bash
CUDA_VISIBLE_DEVICES=0 VLLM_BYTE_V2_USE_NATIVE_KERNELS=1 \
  .venv/bin/python -m pytest \
  tests/v1/attention/test_byte_v2_ops.py::test_byte_v2_v3_decode_append_writes_outlier_arena_cuda \
  tests/v1/attention/test_byte_v2_ops.py::test_byte_v2_v3_decode_append_preserves_partial_outliers_cuda \
  -q -s

CUDA_VISIBLE_DEVICES=0 VLLM_BYTE_V2_USE_NATIVE_KERNELS=1 \
  .venv/bin/python -m pytest \
  tests/v1/attention/test_byte_v2_ops.py::test_byte_v2_v3_decode_append_finalizes_partial_block_cuda \
  tests/v1/attention/test_byte_v2_ops.py::test_byte_v2_native_outlier_arena_prefill_direct_cuda \
  tests/v1/attention/test_byte_v2_ops.py::test_byte_v2_native_outlier_arena_decode_overlay_cuda \
  -q -s
```

结果：

```text
2 passed
3 passed
```

持久化 CUDA 构建目录：

```bash
cmake --build build/temp.linux-x86_64-cpython-312 \
  -j=24 --target _C_stable_libtorch

cmake --install build/temp.linux-x86_64-cpython-312 \
  --prefix /mnt/sda1/yxz/byte_v2/vllm \
  --component _C_stable_libtorch
```

E2E 对比 workload：

```text
model=/mnt/sda1/yxz/byte_v2/Meta-Llama-3-8B-Instruct
prompt_len=512
batch_size=4
decode_len=64
payload_layout=v3
outlier_arena=on
sparse fallback ratio=0.03
```

| variant | output tok/s | elapsed s | pool exhausted |
| --- | ---: | ---: | --- |
| 修复前 V3 outlier_arena touched path | 24.29 | 10.539 | false |
| 修复后 V3 outlier_arena fast path | 110.67 | 2.313 | false |
| 修复后 V3 outlier_arena fast path under nsys | 108.94 | 2.350 | false |
| 修复后 + partial outlier clear-order fix | 110.70 | 2.313 | false |

Nsight Systems measured range 结论：

```text
byte_v2_compress_touched_blocks_kernel: 0 launches
byte_v2_init_cache_update_kernel:      0 launches
byte_v2_mark_touched_tokens_kernel:    0 launches

byte_v2_decode_append_cache_kernel:
  2016 launches, 15.52 ms total, 7.70 us/launch

byte_v2_init_decode_append_result_kernel:
  2016 launches, 2.03 ms total, 1.01 us/launch
```

profile artifact：

```text
benchmarks/profiles/v3_arena_decode_append_fastpath_p512_b4_d64.json
benchmarks/profiles/v3_arena_decode_append_fastpath_p512_b4_d64_profile.json
benchmarks/profiles/v3_arena_decode_append_fastpath_p512_b4_d64_profile.nsys-rep
benchmarks/profiles/v3_arena_decode_append_fastpath_p512_b4_d64_kern_cuda_gpu_kern_sum_nvtx=byte_v2_bench_measured.csv
benchmarks/profiles/v3_arena_decode_append_fastpath_p512_b4_d64_after_clear_fix.json
```

### 17.11 BLOCK_N=32/64 FlashAttention-style stage1 实验记录

本轮尝试实现了一个 opt-in 的 CUTE/WMMA block-n split-stage1 kernel：

```text
VLLM_BYTE_V2_DECODE_CUTE_BLOCK_N=32  # 一次处理 2 个 16-token page
VLLM_BYTE_V2_DECODE_CUTE_BLOCK_N=64  # 一次处理 4 个 16-token page
```

设计目标：

```text
1. shared memory 中同时放 2/4 个 page 的 K/V。
2. QK 产生 16x32 或 16x64 scores。
3. 在更大的 BLOCK_N 上做一次 online softmax。
4. PV 阶段用一个 accumulator 累加多个 16-row V tile。
5. 显式打开 BLOCK_N 时同步限制 split-k：
   BLOCK_N=32 时每个 split 至少 2 个 page；
   BLOCK_N=64 时每个 split 至少 4 个 page。
```

正确性临时测试：

```text
tests/v1/attention/test_byte_v2_decode.py::
  test_byte_v2_paged_decode_attention_op_cute_block_n_stage1_cuda
tests/v1/attention/test_byte_v2_decode.py::
  test_byte_v2_paged_decode_attention_op_cute_block_n_metadata_cuda

结果：3 passed
```

这些测试只用于验证本轮实验 kernel，因性能未达标，已随实验代码一起撤回。

decode-only A/B：

```text
CUDA_VISIBLE_DEVICES=0 VLLM_BYTE_V2_USE_NATIVE_KERNELS=1 \
  .venv/bin/python benchmarks/kernels/benchmark_byte_v2_decode_kernel.py \
  --device cuda --batch-size <B> --seq-len <S> \
  --num-heads 32 --num-kv-heads 8 --head-size 128 --head-size-v 128 \
  --split-k <K> --variant page_fastpath --payload-layout v3 \
  --tile-fastpath-mode on --num-runs 50 --warmup-runs 10
```

| workload | variant | median us | tok/s | 变化 |
| --- | --- | ---: | ---: | ---: |
| p512/b4/split16 | CUTE baseline | 91.14 | 43890.45 | baseline |
| p512/b4/split16 | BLOCK_N=32 | 98.30 | 40690.10 | -7.9% |
| p512/b4/split16 | BLOCK_N=64 | 143.90 | 27796.31 | -58.0% |
| p2048/b1/split64 | CUTE baseline | 72.70 | 13754.40 | baseline |
| p2048/b1/split64 | BLOCK_N=32 | 98.30 | 10172.53 | -35.2% |
| p2048/b1/split64 | BLOCK_N=64 | 148.48 | 6734.91 | -104.2% |

结论：

```text
该 naive BLOCK_N=32/64 聚合没有达到保留门槛，代码不保留。
```

主要原因判断：

1. 当前 CUTE split-stage1 已经通过 split-k 让一个 CTA 处理多个 page；
   naive BLOCK_N 只把多个 16-row page 合到一个 shared score/p 矩阵，并没有减少
   compressed payload load/decode 的主体开销。
2. `p_shared` 和 `scores` 的 leading dimension 从 16 变成 32/64 后，WMMA shared
   load/store 形态变差，可能增加 shared wavefront/bank conflict。
3. `BLOCK_N=64` 会降低 split 数和 CTA 数，在 p2048/b1 这类低 batch 场景下进一步
   降低并行度，收益被 occupancy/latency 损失抵消。
4. 若未来继续做大 BLOCK_N，应重做 shared swizzle、ldmatrix/WMMA layout 和
   producer/consumer pipeline，而不是在现有 CUTE kernel 上简单扩大 score tile。

### 17.12 macro_N=64 warp-per-query compute macro-tile 实验记录

本轮继续尝试了一个更接近 compute macro-tile 的临时 kernel：

```text
VLLM_BYTE_V2_DECODE_MACRO64_STAGE1=1
```

设计约束：

```text
shape: Llama-3 8B decode
q_per_kv = 4
head_dim = 128
block_size = 16
macro_N = 64，也就是每个 CTA 一次聚合最多 4 个 16-token page
支持 compressed page + tile fallback/raw fallback
不支持 outlier arena overlay
```

实现方式：

```text
1. 保持 ByteV2 codec tile 为 16 tokens x 16 dims。
2. 一个 CTA 一次把最多 4 个 page 的 K/V 解到 shared memory。
3. 4 个 warp 分别处理同一个 KV head 下的 4 个 query head。
4. 每个 warp 跨最多 64 个 token 做一次 online softmax 更新。
5. PV 阶段一次累加 4 个 page 的 V。
```

正确性 smoke：

```text
p512/b4/split16/fallback=0.03/payload_layout=v3
max_abs_diff = 0.0
```

decode-only A/B：

```text
CUDA_VISIBLE_DEVICES=0 VLLM_BYTE_V2_USE_NATIVE_KERNELS=1 \
  .venv/bin/python benchmarks/kernels/benchmark_byte_v2_decode_kernel.py \
  --device cuda:0 --batch-size <B> --seq-len <S> \
  --num-heads 32 --num-kv-heads 8 --head-size 128 --head-size-v 128 \
  --split-k <K> --fallback-ratio 0.03 \
  --variant page_fastpath --payload-layout v3 \
  --num-runs 20 --warmup-runs 5 --skip-correctness
```

| workload | variant | op median us | tok/s | stage1 avg us | reduce avg us |
| --- | --- | ---: | ---: | ---: | ---: |
| p512/b4/split8 | CUTE metadata baseline | 93.18 | 42925.82 | 60.02 | 3.34 |
| p512/b4/split8 | macro_N=64 | 140.80 | 28409.09 | 105.92 | 3.33 |
| p2048/b1/split32 | CUTE metadata baseline | 100.35 | 9964.92 | 62.06 | 5.43 |
| p2048/b1/split32 | macro_N=64 | 144.38 | 6925.98 | 107.57 | 5.61 |

profile artifact：

```text
benchmarks/profiles/macro64_p512_b4_s8_cute.json
benchmarks/profiles/macro64_p512_b4_s8_macro64.json
benchmarks/profiles/macro64_p512_b4_s8_cute_profile.nsys-rep
benchmarks/profiles/macro64_p512_b4_s8_macro64_profile.nsys-rep
benchmarks/profiles/macro64_p2048_b1_s32_cute.json
benchmarks/profiles/macro64_p2048_b1_s32_macro64.json
benchmarks/profiles/macro64_p2048_b1_s32_cute_profile.nsys-rep
benchmarks/profiles/macro64_p2048_b1_s32_macro64_profile.nsys-rep
```

结论：

```text
macro_N=64 warp-per-query compute macro-tile 没有达到保留门槛，代码不保留。
```

主要原因判断：

1. 退化几乎全部来自 stage1。reduce 时间基本不变，说明问题不是 split reduce。
2. macro64 虽然减少了 online softmax 更新次数，但没有减少 K/V payload
   read/decode 的总工作量。
3. shared memory 从单 page K/V 变成 4 page K/V 后，shared footprint 变大，
   loader 和 compute 的寄存器/shared 压力增加。
4. warp-per-query macro64 使用标量 dot/PV，不是 ldmatrix/WMMA macro-tile；
   因此没有获得 FlashAttention/CUTLASS 风格大 tile 的 tensor-core 数据复用收益。
5. 未来如果继续做 macro-tile，不能沿用 warp-per-query 标量结构，应转向真正的
   ldmatrix/WMMA/CUTLASS-style macro kernel，并同时重做 shared swizzle 和
   producer/consumer pipeline。

### 17.13 page metadata descriptor Step1 实验记录

本轮尝试了一个 in-kernel page metadata descriptor：

```text
VLLM_BYTE_V2_DECODE_PAGE_META_DESC=1
```

设计目标：

```text
不改变 compute tile / shared K/V tile
不新增全局 descriptor tensor
只在 CUTE metadata stage1 内部提前读取每个 page 的：
  page_status
  valid_rows
  V3 k_fallback_mask
  V3 v_fallback_mask

后续 K/V dim-tile 循环中不再反复调用 byte_v2_tile_fallback_flag()
读取 V3 fallback mask，而是使用寄存器中的 page-level mask 做 bit test。
```

临时实现：

```text
template<bool page_meta_desc>
byte_v2_tile_fallback_flag_cached(...)

byte_v2_paged_decode_attention_gqa_cute_split_stage1_kernel<
  ..., metadata_fastpath=true, page_meta_desc=true>
```

correctness smoke：

```text
p512/b4/split8/fallback=0.03/payload_layout=v3
max_abs_diff = 0.0078125
```

early-exit microbench：

| workload | mode | baseline us | page-meta-desc us | 变化 |
| --- | ---: | ---: | ---: | ---: |
| p512/b4/split8 | 5 metadata/status | 53.73 | 50.18 | +6.6% |
| p512/b4/split8 | 6 metadata+payload | 79.87 | 81.92 | -2.6% |
| p512/b4/split8 | 7 metadata+payload+decode | 78.85 | 85.50 | -8.4% |
| p512/b4/split8 | 0 full | 93.68 | 93.70 | ~0% |
| p2048/b1/split32 | 5 metadata/status | 49.15 | 54.27 | -10.4% |
| p2048/b1/split32 | 6 metadata+payload | 80.90 | 84.99 | -5.1% |
| p2048/b1/split32 | 7 metadata+payload+decode | 85.50 | 88.05 | -3.0% |
| p2048/b1/split32 | 0 full | 99.33 | 98.82 | +0.5% |

Nsight Systems full stage1：

| workload | variant | op median us | stage1 avg us | reduce avg us |
| --- | --- | ---: | ---: | ---: |
| p512/b4/split8 | baseline | 108.11 | 59.33 | 3.33 |
| p512/b4/split8 | page-meta-desc | 108.29 | 60.29 | 3.33 |
| p2048/b1/split32 | baseline | 111.60 | 61.32 | 5.43 |
| p2048/b1/split32 | page-meta-desc | 110.93 | 62.57 | 5.40 |

profile artifact：

```text
benchmarks/profiles/page_meta_desc_*_m{0,5,6,7}.json
benchmarks/profiles/page_meta_desc_p512_b4_s8_{off,on}_profile.nsys-rep
benchmarks/profiles/page_meta_desc_p2048_b1_s32_{off,on}_profile.nsys-rep
benchmarks/profiles/page_meta_desc_summary.json
```

结论：

```text
page metadata descriptor Step1 没有达到保留门槛，代码不保留。
```

主要原因判断：

1. 该实验只减少了 V3 fallback mask 的重复读取，不能减少 payload bytes 或
   register decode 指令。
2. 在 p512/b4 上 mode5 有改善，但 mode6/mode7/full stage1 没有改善，说明节省的
   metadata load 被额外寄存器、模板路径和调度差异抵消。
3. 在 p2048/b1 上 stage1 avg 反而略升，说明当前瓶颈更偏向 payload read/decode 和
   shared/compute pipeline，而不是 fallback mask 重读。
4. 如果继续做 descriptor，下一步不应只做 register cache，而应考虑真正的
   per-physical-block metadata array 或 request-level macro descriptor，前提是
   descriptor 构建/维护成本不能进入每 token hot path。

### 17.14 V3 compressed-first outlier overlay 当前验证结果

本轮重新确认了下面这个方案是否已经整合到 V3：

```text
主路径：
  所有 tile 先按 compressed payload decode 到 shared memory

然后：
  如果该 tile 有 outlier metadata，则从 outlier arena 读取原始 BF16 bits
  并 overlay 到 shared memory 中对应元素

最后：
  QK / softmax / PV 正常使用 shared memory 中的 K/V
```

源码状态：

```text
csrc/libtorch_stable/cache_kernels.cu:
  byte_v2_overlay_k_tile_outliers_to_shared()
  byte_v2_overlay_v_tile_outliers_to_shared()

V3 CUTE stage1:
  decode compressed K/V tile -> shared
  overlay K/V outlier entries -> shared
  QK / PV consume shared K/V
```

也就是说，该方案已经是当前 V3 CUTE stage1 的实现方式之一，不需要再新增
一个功能重复的 kernel。注意 K outlier 不能在 attention output 之后再修正，
因为 K 会改变 QK score 和 softmax 分布，所以 overlay 必须发生在 QK 之前。

功能验证：

```text
CUDA_VISIBLE_DEVICES=0 VLLM_BYTE_V2_USE_NATIVE_KERNELS=1 \
  .venv/bin/python -m pytest \
  tests/v1/attention/test_byte_v2_ops.py::test_byte_v2_v3_decode_append_writes_outlier_arena_cuda \
  tests/v1/attention/test_byte_v2_ops.py::test_byte_v2_v3_decode_append_preserves_partial_outliers_cuda \
  tests/v1/attention/test_byte_v2_ops.py::test_byte_v2_native_outlier_arena_prefill_direct_cuda \
  tests/v1/attention/test_byte_v2_ops.py::test_byte_v2_native_outlier_arena_decode_overlay_cuda \
  -q -s
```

结果：

```text
4 passed
```

decode-only A/B workload：

```text
device: GPU0
batch_size = 2
seq_len = 512
num_heads = 32
num_kv_heads = 8
head_size = 128
head_size_v = 128
split_k = 16
payload_layout = v3
variant = page_fastpath
CUTE stage1 auto = on
V3 cp.async stage = on
```

| variant | fallback ratio | tile fallback | outlier entries | median us | tok/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| clean compressed | 0.00 | 0 | 0 | 75.78 | 26393.58 |
| single-outlier + tile fallback | 0.25 | 16 | 0 | 75.78 | 26393.58 |
| single-outlier + outlier arena overlay | 0.25 | 0 | 16 | 82.43 | 24262.42 |
| outlier arena overlay + block/tile bitmap | 0.25 | 0 | 16 | 79.87 | 25040.06 |

profile artifacts：

```text
benchmarks/profiles/v3_overlay_ab_clean.json
benchmarks/profiles/v3_overlay_ab_tile_fallback.json
benchmarks/profiles/v3_overlay_ab_outlier_arena.json
benchmarks/profiles/v3_overlay_ab_outlier_arena_flags.json
benchmarks/profiles/v3_overlay_ab_*_correctness.json
```

结论：

```text
compressed-first outlier overlay 在 V3 中功能可用，但当前不应作为性能默认路径。
```

主要原因：

1. 在本次 single-outlier workload 下，outlier tile 只有 `16 / 4096`，比例很低。
2. outlier arena overlay 避免了 tile fallback，但引入了额外的
   outlier metadata / arena 读取和 overlay 指令。
3. block/tile bitmap gating 可以把 median 从 `82.43 us` 拉回到 `79.87 us`，
   但仍慢于 clean compressed / tile fallback 的 `75.78 us`。
4. 因此 outlier arena 目前更适合作为 capacity/correctness 路径，而不是
   decode 性能默认路径。

后续如果继续优化该方向，应优先做：

```text
1. 只在 outlier tile 比例较高、tile fallback pool 压力明显时启用 arena overlay。
2. 将 outlier_tile_meta / bitmap 做成更紧凑的 per-page coalesced descriptor。
3. 为 no-outlier hot path 保留 no_outlier_metadata=true 的专用 CUTE kernel，
   让编译器彻底删除 outlier 相关分支。
```

### 17.15 V3 outlier-only / no-fallback metadata 实验记录

本轮实现了一个更激进的 V3 实验模式：

```text
VLLM_BYTE_V2_V3_OUTLIER_ONLY_NO_FALLBACK=1
```

目标是：

```text
1. 不再把 outlier arena 绑定到 tile fallback metadata。
2. full-block prefill direct encode 可以在没有 fallback_pool /
   fallback_tile_ids 的情况下直接写 V3 compressed page。
3. 对 exponent window miss，encode 先把 exponent clamp 到当前 window 内，
   同时把真实 BF16 bits 写入 outlier arena。
4. decode stage1 先统一走 compressed decode，再根据 outlier metadata overlay
   K/V outlier。
5. 如果 outlier 数量超过 VLLM_BYTE_V2_OUTLIER_MAX_PER_TILE 或 arena 容量不足，
   strict mode fail-closed，报 outlier arena exhausted，不再静默退回 fallback。
```

代码改动：

```text
vllm/envs.py
  增加 VLLM_BYTE_V2_V3_OUTLIER_ONLY_NO_FALLBACK。

csrc/libtorch_stable/cache_kernels.cu
  encode host 参数校验允许 V3 outlier arena 不传 tile fallback metadata。
  prefill direct fast path 在 strict mode 下不再要求 sparse fallback pool。
  CUTE metadata stage1 增加 no_fallback_metadata template 参数，
  hot path 不读取 page/tile fallback mask。

vllm/v1/attention/backends/byte_v2_attn.py
  strict mode 下不向 cache update / decode 传 fallback_tile_ids。

benchmarks/kernels/benchmark_byte_v2_decode_kernel.py
  增加 --v3-outlier-only-no-fallback。
```

验证：

```bash
CUDA_VISIBLE_DEVICES=0 VLLM_BYTE_V2_USE_NATIVE_KERNELS=1 \
.venv/bin/python -m pytest \
  tests/v1/attention/test_byte_v2_ops.py::test_byte_v2_v3_decode_append_writes_outlier_arena_cuda \
  tests/v1/attention/test_byte_v2_ops.py::test_byte_v2_v3_decode_append_preserves_partial_outliers_cuda \
  tests/v1/attention/test_byte_v2_ops.py::test_byte_v2_v3_outlier_only_no_fallback_cuda \
  tests/v1/attention/test_byte_v2_ops.py::test_byte_v2_native_outlier_arena_prefill_direct_cuda \
  tests/v1/attention/test_byte_v2_ops.py::test_byte_v2_native_outlier_arena_decode_overlay_cuda \
  -q -s
```

结果：

```text
5 passed
```

decode-only A/B：

```text
device: GPU0
batch_size = 2
num_heads = 32
num_kv_heads = 8
head_size = 128
head_size_v = 128
split_k = 16
payload_layout = v3
fallback_pattern = single_outlier
fallback_ratio = 0.25
CUTE stage1 auto = on
V3 cp.async stage = on
outlier_max_per_tile = 1
```

| workload | variant | fallback tile metadata | median us | tok/s |
| --- | --- | ---: | ---: | ---: |
| seq512/b2 | V3 outlier arena + tile fallback metadata | 4096 | 79.87 | 25040.06 |
| seq512/b2 | V3 outlier-only no-fallback metadata | 0 | 80.88 | 24727.99 |
| seq2048/b2 | V3 outlier arena + tile fallback metadata | 16384 | 191.49 | 10444.52 |
| seq2048/b2 | V3 outlier-only no-fallback metadata | 0 | 209.92 | 9527.44 |

profile artifacts：

```text
benchmarks/profiles/v3_outlier_only_baseline_tile_fallback_s512_gpu0_rerun.json
benchmarks/profiles/v3_outlier_only_no_fallback_s512_gpu0_rerun.json
benchmarks/profiles/v3_outlier_only_baseline_tile_fallback_s2048_gpu0_rerun.json
benchmarks/profiles/v3_outlier_only_no_fallback_s2048_gpu0_rerun.json
benchmarks/profiles/v3_outlier_only_no_fallback_correctness.json
benchmarks/profiles/v3_outlier_only_baseline_tile_fallback_correctness.json
```

结论：

```text
V3 outlier-only/no-fallback metadata 功能可用，但当前不应作为性能默认路径。
```

原因：

1. 它确实把 fallback tile metadata 从 decode 输入中移除了，`fallback_tiles`
   从 `4096/16384` 降到 `0`。
2. 但 single-outlier 场景下 outlier tile 比例很低，移除 fallback metadata
   节省的读取不足以抵消 outlier overlay 路径的额外指令和调度差异。
3. 长上下文 `seq2048/b2` 下退化更明显，说明当前 no-fallback outlier overlay
   stage1 template 还没有形成更好的 memory/compute pipeline。
4. 因此该模式保留为 off-by-default correctness/capacity 实验路径。后续只有在
   tile fallback pool 压力很高，或者 no-fallback template 的 NCU 指标显示
   integer/memory instruction、long scoreboard 和 shared/global sectors 同时下降时，
   才考虑启用 heuristic。

## 18. 结论

当前 `16x16` tile 作为 ByteV2 的压缩粒度是合理的，不建议直接改成
FlashAttention 的大 compute tile。V3 的核心是：

```text
压缩粒度保持 16x16
tile payload 改成 384B 对齐
metadata 从 tile 中移出并 SoA 化
low/code 改成 lane-striped 读取
K/V 使用不同物理顺序
compute kernel 后续聚合多个 16-token page
```

这个方案优先解决当前 profile 中最明确的问题：compressed payload 读取和
metadata traversal 的 latency/issue overhead。只有 V3 loader 在 decode-only 和
NCU 指标上达到门槛后，才继续接入 E2E 默认路径。
