# ByteV2 FlashAttention-like 大改设计

## 1. 目标

目标是把 ByteV2 从“压缩 KV + 自研 paged decode attention”改成更接近
FlashAttention / FlashInfer 的结构：

```text
压缩 KV 继续常驻 HBM
decode 时按 FlashAttention-style macro tile 流式读取 K/V
在 kernel 内解压到寄存器或 shared memory
QK / softmax / PV 使用接近 FlashAttention 的 mainloop
减少 metadata、payload read/decode、shared layout、split-K/reduce 开销
```

这次大改不应该只改 ByteV2 的 `16x16 tile`。如果只把 payload tile 换个排布，
但 attention skeleton、metadata traversal、cache update、allocator、prefix cache 和
cudagraph 仍然沿用旧路径，性能大概率仍会被 stage1 和 host/API 开销吃掉。

## 2. 关键判断：FlashAttention 的 tile 不是 ByteV2 当前的 tile

ByteV2 现在的 tile 是 **压缩编码 tile**：

```text
16 tokens x 16 dims
每个 tile 一个 exponent window / base
low bytes + packed exponent code
```

FlashAttention 的 tile 主要是 **attention compute tile**：

```text
BlockM x BlockN x HeadDim
Q tile 常驻寄存器/shared
K/V 按 BlockN 分段流式进入 shared
QK -> online softmax -> PV
```

因此不要把两者简单等同。建议引入三层概念，并且把每一层做成独立参数：

| 层级 | 建议大小 | 作用 |
| ---- | -------- | ---- |
| codec subtile | 默认 `16 tokens x 16 dims` | ByteV2 lossless/lossy 编码和 outlier 记录粒度 |
| storage macro tile | 默认 `64/128 tokens x head_dim` | HBM 中按 FlashAttention 消费顺序组织 payload |
| compute tile | decode: `q_per_kv x 64/128`，prefill: `64/128 x 64/128` | QK/softmax/PV 的 kernel 计算粒度 |

不建议第一步把 exponent window 扩大到 `128 tokens x 128 dims`。这个范围太大，
真实 Llama KV 很容易出现 outlier，lossless 场景下会显著增加 outlier metadata 或
fallback 压力。更稳妥的做法是：

```text
数值编码仍按 16x16 codec subtile
物理存储按 64/128-token FlashAttention macro tile 重排
kernel 以 64/128-token compute tile 消费
```

### 2.1 参数化 tile policy

新 V4 设计不应继续用一个 `TileSize=16` 同时表示 token 维、dim 维、vLLM block
大小、payload 元素数和 WMMA tile。建议显式拆成：

```cpp
struct ByteV2TilePolicy {
  static constexpr int CodecTokenBlock = 16;
  static constexpr int CodecDimBlock = 16;
  static constexpr int AllocBlockTokens = 16;
  static constexpr int ComputeBlockN = 64;   // 或 128
  static constexpr int HeadDim = 128;
  static constexpr int HeadDimV = 128;
};
```

含义：

```text
CodecTokenBlock:
  一个 codec subtile 覆盖多少 token。默认 16。

CodecDimBlock:
  一个 codec subtile 覆盖多少 hidden dim。默认 16。

AllocBlockTokens:
  vLLM allocator / block_table / prefix cache 的 token block。第一阶段继续 16。

ComputeBlockN:
  attention kernel 一次处理多少 KV token。默认实验 64 或 128。

HeadDim / HeadDimV:
  模型 head size。首版只特化 128。
```

需要保持的约束：

```text
AllocBlockTokens % CodecTokenBlock == 0
HeadDim % CodecDimBlock == 0
HeadDimV % CodecDimBlock == 0
ComputeBlockN % AllocBlockTokens == 0
```

第一阶段建议：

```text
CodecTokenBlock = 16
CodecDimBlock = 16
AllocBlockTokens = 16
ComputeBlockN = 64 或 128
```

这样可以先把代码结构参数化，但不同时冲击 allocator、prefix cache 和 outlier
encoding。后续如果要实验更大的 codec tile，优先尝试：

```text
16 tokens x 32 dims
```

不建议第一步尝试：

```text
32 tokens x 32 dims
```

因为 `32 tokens` 会改变 append/finalize、valid_rows、block_table、prefix cache 和
allocator 假设，风险远大于只改 dim 维。

## 3. 需要模仿 FlashAttention 的核心结构

除了 tile layout，还需要同步改以下部分。

### 3.1 PagedKVManager 风格的 metadata/pointer 管理

参考本地：

```text
.deps/vllm-flash-attn-src/hopper/paged_kv.h
```

ByteV2 应新增类似：

```cpp
ByteV2PagedKVManager
```

职责：

```text
加载 block_table / page_table
生成 64/128-token macro descriptor
集中读取 page_status / valid_rows / layout_version
集中读取 base / fallback / outlier metadata
计算 K/V payload 指针
用 warp broadcast 减少重复 int64 地址计算
给 mainloop 提供 load_K_tile / load_V_tile 接口
```

当前 ByteV2 的 page status、valid rows、fallback mask、outlier bitmap、outlier
meta、fallback slot 分散在 hot loop 内。大改版必须把这些收拢，否则即使 payload
变成 FlashAttention-like，也仍会在 metadata traversal 上浪费时间。

### 3.2 FlashAttention-style mainloop

目标 mainloop：

```text
load Q -> Q 常驻寄存器或固定 shared layout
prefetch/decode K[stage 0]
for n_block in KV macro tiles:
  wait K
  QK
  同时推进 V 或下一段 K 的 load/decode
  online softmax
  wait V
  PV
  推进下一段 K
写 partial output / lse
```

之前简单调语句顺序、double-buffer、cp.async 没有效果，是因为没有形成真正的
producer/consumer mainloop：很多 load/decode 后立刻 wait，payload decode 又依赖
数据到达，无法和 QK/PV 稳定 overlap。

大改版应把 load/decode 和 compute 作为一个整体设计，而不是在旧 kernel 上继续加
环境变量分支。

### 3.3 更大的 KV compute tile

decode 场景中 Q 很小：

```text
Llama-3 8B:
num_heads = 32
num_kv_heads = 8
q_per_kv = 4
head_dim = 128
```

不能照搬 prefill 的 `BlockM=128`，但应该照搬更大的 `BlockN`：

```text
BN64:  4 个 16-token page
BN128: 8 个 16-token page
```

现有 V4 macro8 只是 macro descriptor 变大，计算仍基本逐 page。真正的新版需要让
QK/softmax/PV 的计算粒度也围绕 BN64/BN128 设计。

### 3.4 K/V shared layout 与 MMA/ldmatrix 消费顺序一致

需要分别设计 K 和 V 的 physical order：

```text
K:
  面向 QK，按 WMMA-B / ldmatrix 消费顺序存入 shared
  目标是减少 shared bank conflict 和 shared transpose

V:
  面向 PV，按 row-major 或 PV MMA 需要的 layout 存入 shared
  目标是让 softmax weights x V 的访问连续
```

V3 只解决了 payload 对齐和 stripe 化，还没有完整解决 shared swizzle /
ldmatrix / MMA atom 的组合问题。大改版需要把 payload layout、shared layout、
MMA load pattern 一起设计。

### 3.5 no-fallback/no-outlier 专用 hot path

性能主路径必须是独立 kernel 或 compile-time specialization：

```text
compressed-only
no raw fallback
no tile fallback
no outlier overlay
Llama-3 8B h128 q_per_kv=4
```

不要在 hot path 中保留：

```text
if page fallback
if tile fallback
if outlier arena
if bitmap exists
if layout v1/v3/v4
```

fallback/outlier 应该走单独路径：

```text
Path A: compressed no-outlier fast path
Path B: compressed + outlier overlay path
Path C: rare raw/tile fallback path
```

Path A 才是要逼近或超过 raw vLLM 的路径。

## 4. 新 payload / page layout 方向

建议新增 `ByteV2LayoutV4`，不要直接覆盖 V3。

### 4.1 参数化 codec subtile，默认 16x16

V4 首版默认仍使用：

```text
codec subtile = 16 tokens x 16 dims
每个 subtile 一个 base/exponent window
每个 subtile low/code payload
```

原因：

```text
维持当前压缩率和 lossless/outlier 语义
避免 exponent window 变大导致 outlier 爆炸
兼容 block_size=16 的 vLLM allocator/prefix cache
```

但代码结构必须按参数写，不能再隐式假设：

```text
tile_elems = 16 * 16
packed_code_elems = tile_elems / 2
outlier elem index 永远是 0..255
dim_tiles 永远是 head_dim / 16
page valid_rows 永远等于 codec token block
```

应改成：

```cpp
constexpr int codec_token_block = Policy::CodecTokenBlock;
constexpr int codec_dim_block = Policy::CodecDimBlock;
constexpr int codec_tile_elems = codec_token_block * codec_dim_block;
constexpr int codec_packed_elems = codec_tile_elems / 2;
constexpr int k_dim_tiles = head_size / codec_dim_block;
constexpr int v_dim_tiles = head_size_v / codec_dim_block;
```

outlier entry 也要为未来扩展预留：

```text
当前 16x16:
  elem index 8 bit 足够，范围 0..255

未来 16x32:
  elem index 9 bit，范围 0..511

未来 32x32:
  elem index 10 bit，范围 0..1023
```

因此 V4 outlier sideband 不应继续硬编码 `elem & 0xff`。建议定义：

```cpp
struct ByteV2OutlierEntryPolicy {
  int elem_bits;
  int value_bits;
};
```

首版可以仍用当前格式，但文档和新 V4 helper 要把 entry encode/decode 封装起来，
后续改 codec tile 时只改 helper，不改所有 kernel。

### 4.1.1 后续 codec tile 变更顺序

如果要实验更大 codec tile，建议顺序是：

```text
Step A:
  16x16 默认参数化，不改变行为。

Step B:
  只改 dim 维：16x16 -> 16x32。
  allocator/prefix cache 不动，valid_rows 仍是 16。

Step C:
  如果 16x32 有收益，再考虑 16x64。

Step D:
  最后才考虑 token 维：16x16 -> 32x16 或 32x32。
  这一步需要 allocator/block_table/prefix cache/append 路径一起改。
```

保留门槛：

```text
codec tile 变大后，outlier entry 数不能显著增加
压缩率不能明显下降
payload read/decode microbench 至少 +5%
E2E 不能退化
```

### 4.2 增加 storage macro tile

把 4 或 8 个 physical page 组合成 macro tile：

```text
MacroTile64:
  4 pages x 16 tokens = 64 tokens

MacroTile128:
  8 pages x 16 tokens = 128 tokens
```

有两种实现选择。

#### 选择 A：保留 page=16，新增 macro descriptor

这是推荐第一步。

```text
allocator 仍以 16-token page 分配
block_table 仍指向 16-token physical block
kernel 内按 4/8 page 构造 macro descriptor
payload 可以继续 page-local，但 metadata/pointer 聚合
```

优点：

```text
对 vLLM allocator、prefix cache、block_table 侵入最小
容易和现有 V3/V4 做 A/B
不会引入 128-token page 的内部碎片
```

缺点：

```text
payload 仍跨多个 physical page
HBM 访问不能做到完全连续
需要 descriptor 把小 page 指针聚合
```

#### 选择 B：真正的 64/128-token physical macro page

这是后续更激进版本。

```text
ByteV2 physical block = 64/128 tokens
一个 macro page 内保存 K/V payload 和 metadata
block_table 粒度变大
```

优点：

```text
metadata 连续
payload 连续
最接近 FlashAttention paged KV 的消费方式
```

缺点：

```text
需要改 allocator / KV cache spec / block_table / prefix cache
decode append partial page 复杂
短序列和动态 batch 可能有内部碎片
和 raw vLLM block_size=16 的行为差异变大
```

第一阶段不要直接做选择 B，除非选择 A 已证明 metadata/pointer scatter 是主要瓶颈。

### 4.3 推荐 V4 page-local + macro descriptor layout

保留 V3 的 page-local payload 思路，但重排为更适合 macro tile 的 SoA：

```text
PageHeaderV4, 128B
KvHeadMetaV4[8], 8 * 64B
K payload:
  [kv_head][dim_tile][token_subtile][stripe]
V payload:
  [kv_head][token_subtile][dim_tile][stripe]
Outlier sideband:
  独立 arena / per-tile meta
```

其中：

```text
token_subtile = 16 tokens
dim_tile = 16 dims
stripe = 128B-aligned low/code group
```

K 和 V 的顺序可以不同：

```text
K 按 dim_tile-major，方便构造 D x BN 的 QK operand
V 按 token-major 或 PV 需要的 layout，方便权重连续乘 V
```

### 4.4 Macro descriptor

新增 128B 或 256B 对齐 descriptor：

```cpp
struct ByteV2MacroDesc {
  int32_t physical_blocks[8];
  uint16_t valid_rows[8];
  uint16_t compressed_mask;
  uint16_t outlier_page_mask;
  uint16_t reserved0;
  uint16_t reserved1;
  uint32_t k_payload_offsets[8];
  uint32_t v_payload_offsets[8];
};
```

如果后续证明每 page descriptor 仍太散，可以改成 host/cache-update 阶段预生成
macro descriptor table：

```text
cache update / scheduler 生成 macro_desc_cache
decode kernel 直接连续读取 macro_desc_cache
```

但第一版可以先在 kernel 内构造，避免改 scheduler 太多。

## 5. Decode kernel 重写方案

新增独立 symbol，不继续在现有 kernel 里堆分支：

```cpp
byte_v2_paged_decode_attention_gqa4_h128_fa_tile_stage1_kernel
```

首版只支持：

```text
num_heads = 32
num_kv_heads = 8
q_per_kv = 4
head_size = 128
head_size_v = 128
dtype = bf16
layout = ByteV2LayoutV4 或 V3+macro descriptor
fallback = off
outlier = off
split_k = 8/16/32 可调
```

### 5.1 CTA / warp 分工

建议第一版：

```text
一个 CTA: one request x one kv_head x one split-K chunk
4 个 consumer warp: 每个 warp 处理一个 q head
1-2 个 producer warp: 负责 K/V payload load/decode 到 shared
```

如果寄存器或 occupancy 不理想，再退回：

```text
4 warp CTA: 每 warp 自己 load/decode + compute
```

但真正想超过 raw，需要 producer/consumer 分工，否则 payload decode 很难和 QK/PV
overlap。

### 5.2 BN64 / BN128 两个版本

先实现两个 compile-time 版本：

```cpp
template<int BlockN>
// BlockN = 64 或 128
```

保留门槛：

```text
BN64/BN128 至少一个 decode-only stage1 >= 当前 V4 macro8 +5%
E2E 不退化
NCU 中 long scoreboard、global sectors、shared wavefront 至少一项改善
```

### 5.3 QK / softmax / PV

可选实现路线：

#### 路线 1：warp-per-query scalar dot

优点：

```text
实现简单
和当前 V4 接近
容易验证 correctness
```

缺点：

```text
不是真正 FlashAttention/CUTLASS-style
compute pipe 利用率低
很难大幅超过当前 V4
```

#### 路线 2：ldmatrix/MMA tile

优点：

```text
更接近 FlashAttention
可以复用 CUTLASS/CUTE layout 思路
QK/PV 更可能进入 tensor core fast path
```

缺点：

```text
需要重做 shared swizzle
需要处理 q_per_kv=4 导致 M 很小的问题
可能需要把多个 request/head-group 合并到一个 CTA/warpgroup
```

建议：

```text
P1 先做 warp-per-query BN64/BN128，验证 macro/pipeline 方向
P2 再做 ldmatrix/MMA 版本
```

## 6. Prefill 也需要同步设计

prompt prefill 不应该走 ByteV2 自研 PyTorch attention fallback。

推荐：

```text
prefill attention:
  继续复用 raw vLLM FlashAttention/Triton fast path

prefill cache write:
  raw K/V 在同一个 forward 中生成
  直接 encode 到 ByteV2 compressed cache
  不为了 attention 读取 compressed cache
```

continuation prefill / prefix cache 命中后的 paged prefill 需要两种选择：

```text
短期:
  尽量规避 PyTorch gather+softmax fallback
  没有 native path 时保守回 raw 或禁用该组合

长期:
  实现 ByteV2 native paged prefill
  BlockM=64/128, BlockN=64/128
  和 decode 共享 ByteV2PagedKVManager / payload decoder
```

## 7. Cache update / encoder 也要改

如果 payload layout 改成 FA-like，但 cache update 仍按旧 tile 逐 token 扫描，E2E
仍会被 cache update 吃掉。

需要改：

```text
byte_v2_reshape_and_cache
decode append fast path
prefill block-direct encode
outlier arena writer
fallback/error deferred reporting
```

目标：

```text
prefill:
  one CTA per physical block 或 per macro page
  raw K/V -> compressed V4 layout

decode append:
  token append 写入 page-local V4 subtile
  当 16-token page 完整时 finalize metadata
  如果使用 macro descriptor cache，则更新受影响 descriptor
```

必须避免：

```text
每 token/layer host sync
按 touched token 全局扫描
为了新 layout 重新 staging raw K/V
```

## 8. Allocator / block table / prefix cache 需要检查

如果第一阶段选择“page=16 + macro descriptor”，allocator 改动较少：

```text
KV cache spec 增加 layout_v4 page_size
block_table 仍是 16-token block
prefix cache hash 仍以 16-token block 为单位
macro descriptor 在 kernel 内或辅助 buffer 中构造
```

如果后续选择“physical macro page=64/128”，需要大改：

```text
KV cache block_size
block_table logical indexing
free list / allocator page size
prefix cache block hash 粒度
partial block valid rows
decode append slot_mapping
raw fallback pool 单位
swap/copy block 逻辑
```

因此建议先不要改 allocator block granularity。

## 9. CUDAGraph / workspace / scheduling

FlashAttention-like 性能依赖稳定 launch 和静态 workspace。

需要同步做：

```text
ByteV2 backend 恢复 cudagraph-safe 路径
partial workspace 预分配
macro descriptor workspace 预分配
split-K reduce workspace 预分配
sticky error flag 替代 host sync
避免根据 outlier/fallback 每 token 改变 kernel signature
```

调度层可以加 conservative heuristic：

```text
if layout_v4
and no fallback
and no outlier
and h128 q_per_kv=4
and decode-only
and context_len >= threshold:
    use fa_tile_stage1
else:
    fallback current V4/CUTE path
```

## 10. 文件改动范围

主要文件：

```text
csrc/libtorch_stable/cache_kernels.cu
csrc/libtorch_stable/ops.h
csrc/libtorch_stable/torch_bindings.cpp
vllm/_custom_ops.py
vllm/v1/attention/backends/byte_v2_layout.py
vllm/v1/attention/backends/byte_v2_ops.py
vllm/v1/attention/backends/byte_v2_attn.py
vllm/v1/attention/backends/byte_v2_outliers.py
vllm/v1/kv_cache_interface.py
vllm/v1/worker/gpu_model_runner.py
vllm/v1/worker/gpu/attn_utils.py
benchmarks/kernels/benchmark_byte_v2_decode_kernel.py
benchmarks/kernels/benchmark_byte_v2_decode_page_wmma_microbench.py
benchmarks/benchmark_byte_v2_decode_e2e.py
tests/v1/attention/test_byte_v2_layout.py
tests/v1/attention/test_byte_v2_ops.py
tests/v1/attention/test_byte_v2_decode.py
tests/v1/attention/test_byte_v2_e2e.py
```

参数化 tile policy 建议优先落在：

```text
Python:
  vllm/v1/attention/backends/byte_v2_layout.py
    ByteV2TilePolicy
    ByteV2PageLayoutV4
    payload size / tile count / metadata size 统一从 policy 计算

CUDA:
  csrc/libtorch_stable/cache_kernels.cu
    ByteV2TilePolicy template
    byte_v2_codec_tile_elems(policy)
    byte_v2_outlier_entry_encode/decode(policy)
    byte_v2_v4_tile_payload_start(policy)
    byte_v2_v4_decode_k/v_tile(policy)
```

第一步只允许：

```text
把 16x16 常量替换成 policy 计算
行为保持 bit-equivalent
测试覆盖 page_size、tile_count、outlier entry、encode/decode
```

不要在同一个 patch 里既参数化又改变 codec tile 大小，否则很难定位性能和正确性问题。

不要一开始修改 vendored FlashAttention 源码。先在 ByteV2 自己的 kernel 中复刻
必要结构，等 lower-bound 证明有效后，再考虑 FlashInfer/FlashAttention loader
集成。

## 11. 分阶段执行计划

### Phase 0：固定基线

目的：避免重构后不知道收益来自哪里。

固定 benchmark：

```text
p512/b4/d128
p512/b4/d256
p2048/b1/d32
p4096/b1/d128
p8192/b1/d128
```

记录：

```text
raw vLLM
current ByteV2 V4 macro8 no-arena
current ByteV2 V4 outlier-only
decode kernel microbench
E2E tok/s
NSYS stage time
NCU: integer inst, memory inst, long scoreboard, sectors, shared wavefronts
```

### Phase 1：TilePolicy 参数化，不改变行为

目的：

```text
把 CodecTokenBlock / CodecDimBlock / ComputeBlockN 解耦
默认仍是 16x16 codec subtile
确保当前 V3/V4 行为不变
```

需要完成：

```text
Python layout policy
CUDA policy/helper
outlier entry helper 封装
payload size 由 policy 推导
测试补齐 16x16 默认行为
```

保留门槛：

```text
现有 ByteV2 tests 通过
decode kernel correctness 不变
page size / compression ratio 不变
E2E 不退化
```

### Phase 2：LayoutV4 microbench

只做 payload layout 和 decode microbench，不接 E2E。

目标：

```text
V4 payload read/decode microbench >= V3 +5%
global sectors/request 下降
long scoreboard 不上升
```

如果没有收益，不进入下一阶段。

本轮执行记录：

```text
已完成：
  - Python 侧新增 ByteV2PageLayoutV4，作为当前 V4 stage1 的显式 layout view。
  - 当前 V4 仍消费 V3 physical page format，不改变 page bytes / allocator block size。
  - 新增 V4 macro4、macro8、block128 CUDA correctness tests。

测试：
  .venv/bin/python -m pytest tests/v1/attention/test_byte_v2_layout.py -q
  CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m pytest \
    tests/v1/attention/test_byte_v2_decode.py -q \
    -k "v4_stage1_cuda or v4_block128_stage1_cuda"

结果：
  layout: 17 passed
  V4 decode correctness: 3 passed
```

decode-only A/B，GPU0，`batch=1`、`split_k=16`、`fallback=0`、
`partial_workspace`、`parallel_reduce=off`：

| workload | baseline CUTE no-metadata | V4 macro4 | V4 macro8 | V4 block128 |
| --- | ---: | ---: | ---: | ---: |
| p1024/b1 | 69.63 us | 64.51 us | 63.49 us | 64.51 us |
| p4096/b1 | 225.76 us | 未测 | 201.73 us | 202.75 us |

结论：

```text
V4 macro8 decode-only 达到 Phase 2 的 +5% 保留门槛：
  p1024: 约 +8.8%
  p4096: 约 +10.6%

block128 没有超过 macro8，暂不作为默认候选。
下一步可以进入 Phase 3，但只针对 no-fallback/no-outlier specialized path。
```

### Phase 3：FA-style no-outlier decode kernel

实现：

```text
ByteV2PagedKVManager
BN64/BN128 stage1
no fallback/no outlier specialized kernel
split-K reduce 复用现有路径
```

保留门槛：

```text
decode-only stage1 >= 当前 V4 macro8 +5%
E2E >= 当前 ByteV2 +2%
correctness 不退化
```

本轮执行记录：

```text
现有代码中已经有 V4 macro descriptor stage1：
  VLLM_BYTE_V2_DECODE_V4_STAGE1=1
  VLLM_BYTE_V2_DECODE_V4_MACRO_PAGES=8

该路径只在以下条件下启用：
  payload_layout=v3
  head shape = Llama-3 8B GQA: H=32, KVH=8, D=128
  split-K enabled
  no sparse fallback metadata
  no tile fallback metadata
  no outlier arena / bitmap metadata

为了验证 no-fallback/no-outlier hot path，E2E 使用性能实验配置：
  VLLM_BYTE_V2_ENABLE_SPARSE_FALLBACK_POOL=0
  VLLM_BYTE_V2_ENABLE_OUTLIER_ARENA=0
  VLLM_BYTE_V2_LOSSY_MAX_MISSES_PER_TILE=256

注意：这不是 lossless production 配置，只用于隔离 Phase 3 hot path。
```

correctness gate：

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m pytest \
  tests/v1/attention/test_byte_v2_decode.py -q \
  -k "v4_stage1_cuda or v4_block128_stage1_cuda"
```

结果：

```text
3 passed
```

E2E A/B，GPU0，Llama-3 8B，`prompt_len=512`、`batch=1`、
`decode_len=128,256`、`num_runs=2`、CUDA graph on：

| decode len | CUTE no-metadata baseline | V4 macro8 | delta |
| --- | ---: | ---: | ---: |
| 128 | 6.902 tok/s | 6.916 tok/s | +0.21% |
| 256 | 6.904 tok/s | 6.920 tok/s | +0.23% |

结论：

```text
V4 macro8 在 decode-only microbench 中有 +8%~10% 收益，
但在完整 E2E 中只兑现约 +0.2%，未达到 Phase 3 的 +2% 保留门槛。

因此：
  - V4 macro8 继续保留为实验开关和后续 profile 对象。
  - 不默认启用 V4 macro8。
  - 不进入 Phase 4 默认开发，除非先通过 profile 找到 E2E 中吞掉 kernel 收益的环节。
```

Phase 3 profile，GPU0，Llama-3 8B，`prompt_len=512`、`batch=1`、
`decode_len=128`、`num_runs=1`、`warmup_decode_len=8`、CUDA graph on，
profile 区间为 `byte_v2_bench_measured`：

| stage | CUTE no-metadata | V4 macro8 | delta |
| --- | ---: | ---: | ---: |
| total GPU kernels | 18413.8 ms | 18395.8 ms | -18.0 ms |
| ByteV2 cache update/compress | 14715.8 ms | 14720.3 ms | +4.4 ms |
| Linear/GEMM | 3451.4 ms | 3451.7 ms | +0.3 ms |
| ByteV2 stage1 | 168.1 ms | 143.0 ms | -25.1 ms |
| ByteV2 reduce | 13.0 ms | 13.0 ms | -0.0 ms |
| RMSNorm | 23.2 ms | 24.8 ms | +1.6 ms |

top kernel：

| kernel | CUTE no-metadata | V4 macro8 |
| --- | ---: | ---: |
| `byte_v2_compress_touched_blocks_kernel` | 14696.0 ms / 4096 calls | 14700.8 ms / 4096 calls |
| decode stage1 | 168.1 ms / 4096 calls | 143.0 ms / 4096 calls |
| split reduce | 13.0 ms / 4096 calls | 13.0 ms / 4096 calls |

CUDA API summary 中 `cudaMemcpyAsync` 占比很高，但 GPU MemOps summary 显示显式
memcpy/memset 不是主因：

| mem op | CUTE no-metadata | V4 macro8 |
| --- | ---: | ---: |
| CUDA memset time | 27.9 ms | 28.9 ms |
| D2D memcpy time | 5.0 ms | 5.0 ms |
| H2D/D2H memcpy time | 0.8 ms | 0.8 ms |

profile 结论：

```text
V4 macro8 的 stage1 优化是有效的：
  stage1 168.1 ms -> 143.0 ms，下降约 14.9%。

但在该 E2E workload 中，stage1 只占 GPU kernel 总时间的 0.9% 左右；
byte_v2_compress_touched_blocks_kernel 占约 80%，并且 V4 不改变该部分。

因此 decode-only microbench 的 +8%~10% 收益无法在 E2E 中兑现。
下一步不能继续直接进入 Phase 4 producer/consumer pipeline，
应先优化 decode append/cache update/compress 路径，否则后续 stage1 优化仍会被稀释。
```

#### cache update 优化尝试：no-fallback temp staging append

尝试内容：

```text
问题：
  no sparse fallback pool + no outlier arena + V3 compressed-only + lossy=256
  时，decode append fast path 原来不能启用，因为现有 append kernel 依赖
  fallback_pool 作为 partial raw staging。

尝试：
  给 byte_v2_decode_append_cache_kernel 增加可选 raw_staging 参数。
  当没有 fallback_pool 时，每个 append token 使用临时 raw_staging：
    1. 解压已有 compressed partial page 到 raw_staging。
    2. 写入当前 token 的 raw K/V。
    3. partial block 重新写回 compressed partial page。
    4. full block 走现有 finalize/compress。

安全边界：
  只允许 V3 compressed-only、无 sparse fallback、无 tile fallback、
  无 outlier arena/block/tile metadata、lossy_max_misses_per_tile >= 256。
  最终默认关闭，只保留实验开关：
    VLLM_BYTE_V2_DECODE_APPEND_TEMP_STAGING=1
```

correctness：

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest \
  tests/v1/attention/test_byte_v2_ops.py -q \
  -k "native_decode_append_uses_partial_raw_fallback_cuda \
      or native_decode_append_finalizes_partial_raw_fallback_cuda \
      or native_batched_decode_append_finalizes_blocks_cuda \
      or native_compressed_only_cache_update_and_decode_cuda \
      or temp_staging_no_fallback"
```

结果：

```text
5 passed
```

microbench，GPU0 当时被其他任务占用，只作方向判断：

| path | condition | median per append |
| --- | --- | ---: |
| touched path | no fallback pool, lossy unset | ~5.05 ms |
| temp-staging append | no fallback pool, lossy=256, temp staging on | ~3.75 ms |
| existing fallback-pool append | fallback_pool on | ~0.166 ms |

Nsight Systems kernel summary，16 次 append：

| path | main kernel | total / calls |
| --- | --- | ---: |
| touched path | `byte_v2_compress_touched_blocks_kernel` | 67.7 ms / 16 |
| temp-staging append | `byte_v2_decode_append_cache_kernel` | 74.4 ms / 16 |

结论：

```text
temp-staging append 成功把路径从 touched kernel 切到 append kernel，
但没有解决核心问题：partial block 仍然每 token 解压并重压整个 compressed page。

因此该方案不作为默认性能路径，仅保留显式实验开关。
真正有效的是 existing fallback-pool append path，因为它把 partial block
保持为 raw staging，直到 16-token block 满后才压缩。
```

下一步应改为：

```text
1. 设计 partial raw staging pool，和 raw fallback capacity 解耦。
2. 或者在现有 fallback_pool 上增加 finalized partial slot reuse/free-list，
   让 decode append 只需要 O(active_partial_blocks) raw staging 容量，
   而不是 O(total_generated_blocks) 单调消耗 fallback_next_slot。
3. 在 no-fallback/no-outlier 模式下默认启用该 staging pool，
   目标是保留 fallback-pool append 的 ~0.1ms 级 cache update，
   同时避免 3%/512-block sparse fallback pool 的常驻容量成本。
```

### Phase 4：producer/consumer pipeline

在 FA-style no-outlier decode kernel 有收益后再做：

```text
producer warp load/decode K/V
consumer warp QK/softmax/PV
double-buffer shared
cp.async 只用于 raw payload bytes staging，不直接替代 decode
```

保留门槛：

```text
NCU long scoreboard 下降
eligible warps/scheduler 上升
stage1 >= Phase 2 +3%
```

### Phase 5：outlier overlay path

只在 no-outlier fast path 稳定后再加：

```text
compressed decode 一律执行
outlier sideband overlay 到 shared 或寄存器
不在 hot path 扫 fallback metadata
```

保留门槛：

```text
outlier_ratio 真实场景下 E2E 不低于当前 V4
no-outlier fast path 不受影响
```

### Phase 6：cache update / prefill / prefix cache

接入：

```text
LayoutV4 encode
prefill block-direct encode
decode append fast path
prefix cache correctness
cudagraph-safe workspace
```

保留门槛：

```text
prefill 不慢于当前 ByteV2
decode append cache update 不显著退化
prefix cache 可用
cudagraph 不禁用
```

### Phase 7：E2E 默认 heuristic

只在完整 E2E 有稳定收益后默认启用：

```text
if no-outlier/no-fallback and shape supported:
    LayoutV4 + FA tile kernel
else:
    current stable V4/CUTE path
```

## 12. 实验命令模板

decode kernel A/B：

```bash
CUDA_VISIBLE_DEVICES=<gpu> \
VLLM_BYTE_V2_DECODE_SPLIT_K=16 \
.venv/bin/python benchmarks/kernels/benchmark_byte_v2_decode_kernel.py \
  --batch-size 1 \
  --seq-len 4096,8192,12288 \
  --num-heads 32 \
  --num-kv-heads 8 \
  --head-size 128 \
  --head-size-v 128 \
  --block-size 16 \
  --fallback-ratio 0.0 \
  --v4-stage1 \
  --v4-macro-pages 8 \
  --partial-workspace \
  --parallel-reduce-mode off \
  --num-runs 100 \
  --warmup-runs 25
```

E2E：

```bash
CUDA_VISIBLE_DEVICES=<gpu> \
VLLM_ENABLE_V1_MULTIPROCESSING=0 \
VLLM_BYTE_V2_USE_NATIVE_KERNELS=1 \
VLLM_BYTE_V2_COMPRESSED_ONLY_CACHE=1 \
VLLM_BYTE_V2_PAYLOAD_LAYOUT=v4 \
VLLM_BYTE_V2_DECODE_PAGE_FASTPATH=1 \
VLLM_BYTE_V2_DECODE_APPEND_SKIP_VALIDATION_SYNC=1 \
VLLM_BYTE_V2_DEFERRED_CACHE_UPDATE_ERROR_CHECK=1 \
.venv/bin/python benchmarks/benchmark_byte_v2_decode_e2e.py \
  --modes raw,byte_v2_compressed_only \
  --model /mnt/sda1/yxz/byte_v2/Meta-Llama-3-8B-Instruct \
  --prompt-len 512 \
  --decode-lens 128,256,512 \
  --batch-size 1,4 \
  --num-runs 3 \
  --warmup-decode-len 8
```

NCU/NSYS 需要至少对比：

```text
current V4 macro8
new LayoutV4 + FA tile stage1
raw vLLM
```

## 13. 成功标准

不能只看单个 microbench。

最低保留标准：

```text
payload read/decode microbench >= +5%
decode stage1 >= +5%
E2E tok/s >= +2%
no correctness regression
no prefix cache regression
no cudagraph regression
no fallback/outlier path accidental default
```

如果只有：

```text
microbench +1% 到 +2%
E2E 无收益
NCU 指标无改善
```

则不保留默认启用，只保留文档记录或实验开关。

## 14. 最大风险

1. **机械照搬 FlashAttention tile**
   - FlashAttention 的 prefill tile 和 ByteV2 decode GQA 场景不同。
   - decode 的 M 很小，直接上大 MMA tile 可能 occupancy/利用率不佳。

2. **把 block_size 从 16 改到 128**
   - 可能破坏 allocator/prefix cache，并引入内部碎片。
   - 只有在 macro descriptor 方案证明不够时才考虑。

3. **layout 改了但 encoder/cache update 没跟上**
   - E2E 会被 cache update 吃掉。

4. **outlier/fallback 分支污染 hot path**
   - 必须用独立 kernel 或 compile-time specialization。

5. **cp.async 没有真正 overlap**
   - cp.async 只是工具，必须和 producer/consumer mainloop 一起设计。

## 15. 最核心的执行原则

这次大改的核心不是“把 16x16 tile 改成 FlashAttention 的 tile”，而是：

```text
16x16 作为默认 ByteV2 codec subtile，但通过 TilePolicy 参数化
CodecTokenBlock / CodecDimBlock / ComputeBlockN 解耦
64/128-token macro tile 作为 storage/compute 单位
用 FlashAttention-style PagedKVManager + mainloop 消费 compressed KV
把 metadata/fallback/outlier 从主路径中拿掉
让 cache update、prefill、prefix cache、cudagraph 同步适配
```

只有这些一起改，ByteV2 才有机会把“压缩 KV 减少 HBM 读取”的理论优势转化成
E2E 性能优势。
