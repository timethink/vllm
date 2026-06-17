# ByteV2 FlashAttention-style Decode Stage1 优化方案

## 1. 目标

本文档记录下一阶段 ByteV2 decode attention stage1 的结构性优化方案。
核心思路是参考本地 vendored FlashAttention 的 SM80 paged KV 实现，把 ByteV2
当前 page/tile 级 load-decode-compute 路径改造成更接近 FlashAttention 的
macro tile、paged KV manager、cp.async pipeline、QK/PV overlap 结构。

当前结论：

1. ByteV2 compressed KV 的纯 HBM 读取量低于 raw KV，读带宽收益是存在的。
2. 真实 E2E 中收益被 decode attention stage1 的额外开销吃掉，包括 payload
   read/decode、metadata traversal、shared layout、WMMA load pattern、partial
   workspace/reduce。
3. 后续不应继续优先做 split-K、bitmap、outlier 小分支这类外围优化，而应先建立
   一个独立的 FlashAttention-style stage1 kernel，验证结构性上限。

本文档只设计优化路线，不要求修改 vendored FlashAttention 源码。所有 ByteV2
实现应继续放在 `csrc` 或 vLLM ByteV2 backend 中。

## 2. 本地 FlashAttention 参考点

本地源码位于：

```text
.deps/vllm-flash-attn-src/
```

重点参考文件：

```text
.deps/vllm-flash-attn-src/hopper/paged_kv.h
.deps/vllm-flash-attn-src/hopper/mainloop_fwd_sm80.hpp
.deps/vllm-flash-attn-src/hopper/tile_size.h
.deps/vllm-flash-attn-src/hopper/flash_fwd_kernel_sm80.h
.deps/vllm-flash-attn-src/csrc/flash_attn/src/flash_fwd_kernel.h
.deps/vllm-flash-attn-src/csrc/flash_attn/src/kernel_traits.h
.deps/vllm-flash-attn-src/csrc/flash_attn/src/softmax.h
```

### 2.1 PagedKVManager

`hopper/paged_kv.h` 中的 `PagedKVManager` 对 ByteV2 最有参考价值。

它做了几件事：

1. 以 `kBlockN x head_dim` 为单位处理 paged KV，而不是每 16-token page 单独做
   一个完整 attention 小循环。
2. 用 `uint128_t` / 16B 向量化 load，并通过 CUTE tiled copy 组织 cp.async。
3. 对 paged KV 指针计算做 warp 内分工，再用 `__shfl_sync` 广播给同一行的线程，
   避免每个线程重复做 int64 地址计算。
4. 将 `load_page_table()`、`compute_K_ptr()`、`compute_V_ptr()`、`load_K()`、
   `load_V()` 封装成 manager，mainloop 只关心加载哪个 `n_block`。

ByteV2 当前问题是 page status、valid rows、tile fallback、outlier metadata 和
payload 指针计算分散在 decode hot loop 中。后续应仿照 `PagedKVManager` 做
`ByteV2PagedKVManager`，把 metadata/pointer traversal 从 attention mainloop
中收拢出来。

### 2.2 SM80 mainloop pipeline

`hopper/mainloop_fwd_sm80.hpp` 的 SM80 mainloop 值得重点参考：

```text
load_Q -> cp_async_fence
load_K stage(s) -> cp_async_fence
load_V stage(s) -> cp_async_fence
preprocess_Q / Q in registers
for each n_block:
  wait/sync current K
  QK GEMM
    overlap load_V_next callback
  online softmax
  wait/sync current V
  PV GEMM
  load_K_next
```

关键点不是简单使用 `cp.async`，而是：

1. K/V load 与 QK/PV compute 之间形成固定 mainloop。
2. `load_V_next` 作为 QK GEMM 的 callback，在计算 QK 时推进下一段 V load。
3. `load_K_next` 在 PV 后推进下一段 K load。
4. Q 可以常驻寄存器，减少 shared 往返。
5. shared memory 被 mainloop 和 epilogue 复用，避免额外 buffer。

ByteV2 之前的 double-buffer / cp.async 实验收益不明显，主要原因是没有形成类似
FlashAttention 的 producer/consumer mainloop：很多 load/decode 之后很快 wait，
payload decode 又依赖数据到达，实际 overlap 很弱。

### 2.3 Tile policy

`hopper/tile_size.h` 中 SM80 的 hdim=128 策略大致是：

```text
element_size = 2
head_dim <= 128
sm86_or_89 = true
kBlockM = 128
kBlockN = 128 或 96/112
kNWarps = 8
kStages = 1
Q_in_regs = true
```

ByteV2 decode 场景和 FlashAttention prefill 不完全一样：

```text
raw FlashAttention prefill:
  Q tile 通常有较大的 BlockM，例如 128。

ByteV2 decode GQA:
  每个 request / kv head group 只有 q_per_kv=4 个 Q row。
```

因此 ByteV2 不能机械照搬 `BlockM=128`。但可以照搬另一个关键思想：

```text
KV 方向使用更大的 macro tile，例如 BLOCK_N=64 或 BLOCK_N=128，
而不是让一个 CTA 只覆盖 16-token page。
```

这样可以摊薄 page metadata、partial workspace、reduce 和 launch 内固定开销，并为
load/decode 与 QK/PV overlap 留出空间。

## 3. 当前 ByteV2 stage1 的主要差距

当前 ByteV2 主要 stage1 实现在：

```text
csrc/libtorch_stable/cache_kernels.cu
```

现有变体包括：

```text
byte_v2_paged_decode_attention_gqa_wmma_split_stage1_kernel
byte_v2_paged_decode_attention_gqa4_h128_fast_split_stage1_kernel
byte_v2_paged_decode_attention_gqa4_h128_flash_split_stage1_kernel
byte_v2_paged_decode_attention_gqa_cute_split_stage1_kernel
```

现有开关包括：

```text
VLLM_BYTE_V2_DECODE_CUTE_STAGE1
VLLM_BYTE_V2_DECODE_CUTE_STAGE1_AUTO
VLLM_BYTE_V2_DECODE_FAST_STAGE1
VLLM_BYTE_V2_DECODE_FLASH_STAGE1
VLLM_BYTE_V2_DECODE_V3_CP_ASYNC_STAGE
```

从已有 profile 和实验看，差距主要来自：

1. **计算粒度过小**
   - 逻辑上仍很接近 16-token page。
   - 每 page 都要处理 metadata/status/payload pointer。
   - partial output 和 reduce 压力被放大。

2. **load/decode 没有形成真正 pipeline**
   - cp.async 或语句重排不能自动带来 overlap。
   - ByteV2 payload 必须先 decode 成 BF16-like 值才能参与 QK/PV。
   - 当前很多路径是 load 后马上 decode/wait，计算期间没有稳定推进下一段 K/V。

3. **metadata traversal 分散**
   - page_status、valid_rows、fallback mask、outlier flag、outlier bitmap、
     outlier meta、fallback slot 等检查散落在 hot loop。
   - 对 compressed-only/no-arena/no-fallback 这种性能主路径，很多判断本应被编译期
     或独立 kernel 删除。

4. **payload layout 与 WMMA 消费顺序不完全匹配**
   - V3 已经把 tile payload 做到 384B 和 stripe 化，但当前 stage1 仍未完全按
     ldmatrix/WMMA-B/PV 消费顺序重建 shared layout。

5. **自研 attention skeleton 与 FlashAttention/FlashInfer 有结构差距**
   - 即使 compressed read bytes 更少，当前 stage1 仍可能因为 softmax/PV/reduce
     结构不够高效而落后 raw。

## 4. 总体方案

新增一个独立 kernel symbol，不继续在现有 CUTE/WMMA kernel 中堆运行时分支：

```cpp
byte_v2_paged_decode_attention_gqa4_h128_fa_style_stage1_kernel
```

第一版只支持最重要的生产形状：

```text
model: Llama-3 8B class
dtype: BF16
num_heads = 32
num_kv_heads = 8
q_per_kv = 4
head_size = 128
head_size_v = 128
block_size = 16
cache layout = ByteV2 V3 compressed page
fallback pool = off 或本 batch 无 raw fallback page
outlier arena = off
tile fallback overlay = off
```

不满足条件时继续回退当前稳定路径。这样可以让编译器删除 fallback/outlier 相关分支，
也避免实验模板污染当前最优 kernel。

### 4.1 Compute macro tile

首选尝试两个 macro tile：

```text
BN64:
  一个 CTA 处理 64 tokens x head_dim，即 4 个 16-token page。

BN128:
  一个 CTA 处理 128 tokens x head_dim，即 8 个 16-token page。
```

预期收益：

1. page metadata 和 pointer traversal 从每 16 token 一次摊薄到每 64/128 token 一次。
2. split-K partial 个数减少，reduce 压力下降。
3. 单 CTA 工作量更接近 FlashAttention mainloop，load/compute overlap 更容易成立。

风险：

1. q_per_kv=4，Q row 很少，BN 太大可能增加寄存器和 shared 压力。
2. 长上下文下 BN128 可能好，短上下文或小 batch 可能 occupancy 不足。
3. outlier/tile fallback 支持会更复杂，所以第一版必须 no-arena/no-fallback。

### 4.2 ByteV2PagedKVManager

仿照 FlashAttention 的 `PagedKVManager`，新增 ByteV2 专用 manager/helper。

职责：

```text
load_macro_descriptor()
load_page_table()
compute_payload_ptrs()
load_K_payload()
load_V_payload()
decode_K_to_shared()
decode_V_to_shared()
```

第一版 manager 只支持 no-arena/no-fallback V3 page：

```text
input:
  block_table
  seq_lens
  page base pointer
  V3 page size / offsets
  kv_head id
  macro_n block id

output:
  K shared tile in WMMA-B friendly layout
  V shared tile in PV friendly layout
```

实现要点：

1. page table / page pointer 由少量 lane 计算，warp 内 broadcast。
2. metadata 尽量一次性连续加载到寄存器或 shared，不在每个 16x16 tile 中反复读。
3. K loader 和 V loader 使用独立模板，不传 runtime layout bool。
4. K decode 输出直接面向 QK 的 ldmatrix/shared layout。
5. V decode 输出直接面向 PV 的 shared layout。

### 4.3 Mainloop pipeline

目标 mainloop：

```text
prologue:
  load Q -> registers
  load/decode K[0] -> shared_k[0]
  load/decode V[0] -> shared_v[0] 或推迟到 QK overlap

for macro block n:
  wait K[n]
  QK(n)
    overlap load/decode V[n]
    overlap prefetch K[n+1] payload when possible
  online softmax update
  wait V[n]
  PV(n)
    overlap load/decode K[n+1]
  rotate buffers

epilogue:
  write partial output / lse
```

和 raw FlashAttention 的差别是：

```text
raw:
  cp.async -> shared BF16 K/V -> ldmatrix -> mma

ByteV2:
  cp.async compressed payload -> registers/shared decode -> shared BF16-like K/V
  -> ldmatrix -> mma
```

因此 ByteV2 的 pipeline 需要两层：

1. **payload prefetch pipeline**
   - 负责把 compressed low/code/base 连续搬到 shared/register staging。

2. **decode-to-MMA-layout pipeline**
   - 负责把 compressed payload 转成 ldmatrix 可消费的 shared layout。

如果 decode 指令仍然串在 QK/PV 前面，cp.async 本身不会解决问题。必须让
`load/decode next K/V` 与当前 QK/PV 的执行重叠。

### 4.4 Shared layout

首版不要再只做局部 padding 或 row-major/col-major 切换，而应按两个 consumer
分别设计：

```text
K shared:
  面向 QK 的 WMMA-B / ldmatrix 消费顺序。
  目标是减少 shared bank conflict 和 ldmatrix 前的重排。

V shared:
  面向 PV 的 row/value 消费顺序。
  目标是让 P x V 的 B fragment load 连续。
```

V3 payload 已经把 K/V 物理 payload 分开，这是后续 shared layout 专门化的前提。
如果 BN64/BN128 新 kernel 仍沿用旧 shared 写法，很可能只能获得有限收益。

## 5. 分阶段实施计划

### Phase 0：固定基线

先固定当前 no-arena 稳定基线，后续所有实验都与它比较：

```bash
CUDA_VISIBLE_DEVICES=<gpu> \
VLLM_ENABLE_V1_MULTIPROCESSING=0 \
VLLM_BYTE_V2_USE_NATIVE_KERNELS=1 \
VLLM_BYTE_V2_COMPRESSED_ONLY_CACHE=1 \
VLLM_BYTE_V2_ENABLE_SPARSE_FALLBACK_POOL=1 \
VLLM_BYTE_V2_SPARSE_FALLBACK_POOL_RATIO=0.03 \
VLLM_BYTE_V2_DEFERRED_CACHE_UPDATE_ERROR_CHECK=1 \
VLLM_BYTE_V2_DECODE_PAGE_FASTPATH=1 \
VLLM_BYTE_V2_DECODE_TILE_FASTPATH=1 \
VLLM_BYTE_V2_ENABLE_OUTLIER_ARENA=0 \
.venv/bin/python benchmarks/benchmark_byte_v2_decode_e2e.py \
  --modes raw,byte_v2_compressed_only \
  --model /mnt/sda1/yxz/byte_v2/Meta-Llama-3-8B-Instruct \
  --prompt-len 512 \
  --decode-lens 128,256 \
  --batch-size 4 \
  --num-runs 3 \
  --warmup-decode-len 8
```

必须记录：

```text
raw output tok/s
ByteV2 output tok/s
stage1 total ms
reduce total ms
cache update total ms
host/API/sampling/GEMM 时间
```

### Phase 1：ByteV2PagedKVManager load/decode microbench

先不要接 E2E，只做 load/decode microbench。

实验对象：

```text
V3 current loader
V3 warp-vectorized stripe loader
V3 ByteV2PagedKVManager BN64 loader
V3 ByteV2PagedKVManager BN128 loader
```

early-exit mode 建议：

```text
mode5: metadata/status traversal + payload read
mode6: metadata/status traversal + payload read + bit decode
mode7: metadata/status traversal + payload read + bit decode + shared store
```

保留门槛：

```text
mode5 或 mode6 median time 下降 >= 15%
global sectors/request 不上升
integer instructions 下降
memory instructions 下降
long scoreboard 不上升
```

如果 load/decode microbench 没有明显收益，不进入 attention stage1。

### Phase 2：BN64 no-arena/no-fallback stage1

新增独立 kernel：

```text
byte_v2_paged_decode_attention_gqa4_h128_fa_style_stage1_kernel<BN=64>
```

dispatch 开关：

```text
VLLM_BYTE_V2_DECODE_FA_STYLE_STAGE1=1
VLLM_BYTE_V2_DECODE_FA_STYLE_BLOCK_N=64
```

只在以下条件启用：

```text
compressed_only = true
outlier_arena = off
layout_version = 3
head_size = 128
head_size_v = 128
q_per_kv = 4
block_size = 16
batch 中没有 raw fallback page
```

验证：

```text
decode-only:
  seq_len = 512, 1024, 2048, 4096
  batch = 1, 2, 4, 8
  split_k = 0, 8, 16, 32

E2E:
  p512/b4/d128
  p512/b4/d256
  p2048/b1/d32
  p4096/b1/d128
```

保留门槛：

```text
decode-only stage1 median >= +8%
E2E output tok/s >= +3%
correctness max error 与当前 ByteV2 baseline 一致
短上下文回退 <= 2%
```

### Phase 3：BN128 stage1

在 BN64 有收益后再尝试 BN128：

```text
VLLM_BYTE_V2_DECODE_FA_STYLE_BLOCK_N=128
```

预期 BN128 更适合长上下文：

```text
prompt_len >= 2048
decode_len >= 128
batch 较小但 context 较长
```

风险是 register/shared 压力和 occupancy。保留门槛：

```text
p2048/p4096 长上下文 E2E >= +5%
p512 短上下文不退化超过 3%
register/thread 不导致 occupancy 明显下降
eligible warps/scheduler 不下降
```

### Phase 4：outlier/tile fallback 支持

不要把 outlier/tile fallback 逻辑加回 no-arena hot kernel。应使用独立 kernel 或独立模板：

```text
fa_style_stage1_no_fallback
fa_style_stage1_tile_fallback
fa_style_stage1_outlier_overlay
```

推荐策略：

1. 默认性能路径只处理 compressed clean tile。
2. 有 outlier arena 时，先 compressed decode，再根据 compact outlier metadata overlay。
3. 有 raw tile fallback 时，单独走 fallback-aware kernel 或回退旧路径。

这样做的目标是让最常用 no-arena/no-fallback 路径在编译期删除 outlier/fallback 分支。

### Phase 5：E2E 和 NCU 四路 A/B

最终对比至少包含四路：

```text
raw vLLM
ByteV2 current CUTE metadata baseline
ByteV2 V3 current cp.async baseline
ByteV2 FA-style BN64/BN128
```

固定 workload：

```text
p512/b4/d128
p512/b4/d256
p2048/b1/d32
p4096/b1/d128
```

必须记录：

```text
stage1 total ms
reduce total ms
cache update total ms
stage1 avg us/launch
integer instructions
memory instructions
long scoreboard
uncoalesced global sectors
uncoalesced shared wavefronts
registers/thread
eligible warps/scheduler
```

如果 E2E 变快但 NCU 的 stage1 duration、instruction、stall 指标没有对应改善，
视为噪声或调度差异，不能直接作为默认路径。

## 6. 代码落点建议

第一版可以先放在当前文件中，便于快速实验：

```text
csrc/libtorch_stable/cache_kernels.cu
```

如果代码量继续增大，应拆成：

```text
csrc/libtorch_stable/byte_v2/byte_v2_fa_style_stage1.cuh
csrc/libtorch_stable/byte_v2/byte_v2_paged_kv_manager.cuh
csrc/libtorch_stable/byte_v2/byte_v2_v3_layout.cuh
```

Python/backend 侧只增加严格 guarded dispatch，不改变默认路径：

```text
VLLM_BYTE_V2_DECODE_FA_STYLE_STAGE1
VLLM_BYTE_V2_DECODE_FA_STYLE_BLOCK_N
```

不建议修改：

```text
.deps/vllm-flash-attn-src/
```

除非后续明确要做 upstream FlashAttention/FlashInfer integration。

## 7. 不建议继续优先做的方向

基于已有实验，下列方向暂不作为主线：

| 方向 | 原因 |
|---|---|
| 单纯调 split-K | decode-only 有时有效，但 E2E 不稳定，且 reduce 不是当前最大瓶颈。 |
| 继续加 outlier bitmap | 真实 E2E 没有稳定收益，且 hot path metadata 更复杂。 |
| 在当前 WMMA kernel 内继续加 runtime 分支 | 容易污染编译形态，难以让编译器删除无关逻辑。 |
| 只做语句重排 | 没有 producer/consumer pipeline，通常不会形成真正 overlap。 |
| 只做 cp.async 包装 | 如果 wait 太早或 decode 串行，cp.async 不会自动降低 stage1。 |
| 修改 vendored FlashAttention 源码 | 不利于维护，也不能直接解决 ByteV2 compressed payload decode。 |

## 8. 成功标准

阶段性成功：

```text
load/decode microbench:
  mode5/mode6 >= +15%

decode-only stage1:
  BN64/BN128 >= +8%

E2E:
  典型 workload output tok/s >= +3%
  长上下文 workload output tok/s >= +5%
```

最终目标：

```text
ByteV2 attention stage1+reduce 接近 raw attention 的 1.1x-1.3x，
并用更小 KV read bytes 在长上下文或显存受限场景中超过 raw E2E。
```

如果 FA-style stage1 仍无法缩小差距，下一步应转向：

```text
1. FlashInfer-compatible compressed loader
2. decompress-to-BF16 staging + raw FlashAttention oracle
3. 更低 decode 指令数的 ByteV2-Fast 编码格式
```

## 9. 下一步执行顺序

建议下一轮按以下顺序执行：

1. 固定当前 no-arena baseline，保存 E2E 和 stage-level profile。
2. 实现 `ByteV2PagedKVManager` 的 BN64/BN128 load/decode microbench。
3. 如果 microbench 达标，接入 `fa_style_stage1<BN=64>` no-arena kernel。
4. 跑 decode-only 和 E2E；如果 BN64 有收益，再做 BN128。
5. 只有在 no-arena hot path 证明有效后，再设计 outlier/tile fallback overlay。

核心判断标准：

```text
先证明结构性 mainloop 能降低 stage1，再讨论如何把 fallback/outlier 加回来。
```

## 10. V4 首版实现记录

V4 首版采用独立 kernel symbol，不替换默认路径：

```cpp
byte_v2_paged_decode_attention_gqa4_h128_v4_split_stage1_kernel
```

启用开关：

```bash
VLLM_BYTE_V2_DECODE_V4_STAGE1=1
VLLM_BYTE_V2_DECODE_V4_MACRO_PAGES=4   # 可选 4 或 8
VLLM_BYTE_V2_DECODE_PAGE_FASTPATH=1
VLLM_BYTE_V2_PAYLOAD_LAYOUT=v3
```

首版支持范围：

```text
num_heads = 32
num_kv_heads = 8
q_per_kv = 4
head_size = 128
head_size_v = 128
block_size = 16
payload layout = V3
outlier arena = off
sparse fallback / tile fallback = off
split-K stage1 path = on
```

首版实现了两个独立点：

1. **macro descriptor**
   - 每个 CTA 按 `MacroPages=4/8` 聚合多个 16-token page。
   - 由少量线程集中读取 `block_table`、`valid_rows`、`page_status`、
     `layout_version`。
   - attention 主循环复用 shared descriptor，减少每个线程重复 metadata
     traversal。

2. **V3-only no-fallback hot path**
   - 只处理 compressed V3 page。
   - 删除 fallback/outlier runtime 分支。
   - K/V decode 直接调用 V3 no-fallback decoder，并默认使用 warp stripe load。
   - 计算 skeleton 复用当前 flash-style warp-per-query QK/softmax/PV 路径。

这不是最终版 CUTLASS/CUTE producer/consumer mainloop。它的定位是：

```text
先建立一个独立、可 A/B、可回滚的 V4 production-shape kernel，
验证 V3-only + macro metadata 合并 + no-fallback 分支删除是否有收益。
```

### 10.1 Decode-only 验证命令

编译 CUDA 扩展后，建议先跑：

```bash
CUDA_VISIBLE_DEVICES=<gpu> \
.venv/bin/python benchmarks/kernels/benchmark_byte_v2_decode_kernel.py \
  --batch-size 4 \
  --seq-lens 512,1024,2048,4096 \
  --num-heads 32 \
  --num-kv-heads 8 \
  --head-size 128 \
  --head-size-v 128 \
  --payload-layout v3 \
  --variant page_fastpath \
  --split-ks 8,16,32 \
  --v4-stage1 \
  --v4-macro-pages 4 \
  --tile-fastpath-mode off \
  --skip-correctness
```

再对比当前 CUTE/V3 baseline：

```bash
CUDA_VISIBLE_DEVICES=<gpu> \
.venv/bin/python benchmarks/kernels/benchmark_byte_v2_decode_kernel.py \
  --batch-size 4 \
  --seq-lens 512,1024,2048,4096 \
  --num-heads 32 \
  --num-kv-heads 8 \
  --head-size 128 \
  --head-size-v 128 \
  --payload-layout v3 \
  --variant page_fastpath \
  --split-ks 8,16,32 \
  --cute-stage1 \
  --tile-fastpath-mode off \
  --skip-correctness
```

如果 correctness 需要验证，去掉 `--skip-correctness` 并先缩小到：

```text
batch_size = 1
seq_len = 128 或 256
split_k = 8
```

### 10.2 E2E 验证命令

```bash
CUDA_VISIBLE_DEVICES=<gpu> \
VLLM_ENABLE_V1_MULTIPROCESSING=0 \
VLLM_BYTE_V2_USE_NATIVE_KERNELS=1 \
VLLM_BYTE_V2_COMPRESSED_ONLY_CACHE=1 \
VLLM_BYTE_V2_PAYLOAD_LAYOUT=v3 \
VLLM_BYTE_V2_ENABLE_SPARSE_FALLBACK_POOL=0 \
VLLM_BYTE_V2_ENABLE_OUTLIER_ARENA=0 \
VLLM_BYTE_V2_DECODE_PAGE_FASTPATH=1 \
VLLM_BYTE_V2_DECODE_TILE_FASTPATH=0 \
VLLM_BYTE_V2_DECODE_SPLIT_K=16 \
VLLM_BYTE_V2_DECODE_V4_STAGE1=1 \
VLLM_BYTE_V2_DECODE_V4_MACRO_PAGES=4 \
VLLM_BYTE_V2_DECODE_APPEND_SKIP_VALIDATION_SYNC=1 \
VLLM_BYTE_V2_DEFERRED_CACHE_UPDATE_ERROR_CHECK=1 \
.venv/bin/python benchmarks/benchmark_byte_v2_decode_e2e.py \
  --modes raw,byte_v2_compressed_only \
  --model /mnt/sda1/yxz/byte_v2/Meta-Llama-3-8B-Instruct \
  --prompt-len 512 \
  --decode-lens 128,256 \
  --batch-size 4 \
  --num-runs 3 \
  --warmup-decode-len 8
```

注意：V4 首版是 no-fallback/no-outlier 专用实验路径。真实 E2E 如果编码阶段产生
raw fallback page、tile fallback 或 outlier arena，host dispatch 会回退当前稳定
stage1，而不是进入 V4。做 V4 E2E 时必须同时检查 profile 里的 stage1 kernel name，
避免把 fallback baseline 误认为 V4 结果。

### 10.3 保留门槛

V4 首版只有在满足以下条件时才继续扩展：

```text
decode-only stage1 median >= 当前 CUTE/V3 baseline +5%
E2E output tok/s 不退化
BN4 与 BN8 至少一个在长上下文稳定收益
correctness 与当前 ByteV2 baseline 一致
```

如果 V4 首版没有收益，应保留文档记录，但不默认启用。后续再进入真正的
CUTLASS/CUTE shared swizzle、ldmatrix、producer/consumer pipeline。

## 11. Block128 Score-Buffer / Two-Pass Stage1 实验

### 11.1 目的

上一版 V4 虽然把 macro descriptor 扩大到 8 个 16-token page，但 attention 计算
仍然是逐 page 执行：

```text
decode K/V page
QK
online softmax update
PV
进入下一个 page
```

这不是严格意义上的 128-token compute tile。block128 实验把一个 macro 内 8 个
page 作为一个 128-token block 处理：

```text
pass 1:
  逐 page decode K
  计算 4 个 query head 对 128 token 的 QK score
  将 scores[4][128] 写入 shared memory

macro softmax:
  每个 query warp 对 128 scores 做 max/sum
  将 scores shared 原地改写为 softmax weight
  只做一次 macro-level running max / denom 更新

pass 2:
  逐 page decode V
  读取 scores/weights[4][128]
  累加 PV 到寄存器
```

该路径通过显式开关启用：

```bash
VLLM_BYTE_V2_DECODE_V4_BLOCK128=1
```

benchmark 脚本增加：

```bash
--v4-block128
```

约束：

```text
只支持 V4 + V3 payload
只支持 macro_pages=8
只支持 specialized no-fallback 路径
支持 no-arena 和 V3 outlier-only overlay
不作为默认路径
```

### 11.2 实现位置

核心 kernel：

```text
byte_v2_paged_decode_attention_gqa4_h128_v4_block128_split_stage1_kernel
```

dispatch 条件：

```text
VLLM_BYTE_V2_DECODE_V4_STAGE1=1
VLLM_BYTE_V2_DECODE_V4_MACRO_PAGES=8
VLLM_BYTE_V2_DECODE_V4_SPECIALIZED=1
VLLM_BYTE_V2_DECODE_V4_BLOCK128=1
```

### 11.3 结果

GPU 空闲后对同一 kernel benchmark 做 A/B。workload：

```text
batch_size = 1
num_heads = 32
num_kv_heads = 8
head_size = 128
head_size_v = 128
block_size = 16
split_k = 16
parallel_reduce = off
```

no-arena：

| seq_len | V4 macro8 | block128 | 结论 |
| ------- | --------: | -------: | ---- |
| 4096    | 199.68 us | 199.68 us | 持平 |
| 7168    | 321.54 us | 322.56 us | 轻微退化 |
| 8192    | 363.52 us | 361.47 us | +0.6% |
| 12288   | 526.34 us | 524.29 us | +0.4% |

outlier-only, fallback_ratio=0.04：

| seq_len | V4 macro8 | block128 | 结论 |
| ------- | --------: | -------: | ---- |
| 4096    | 232.45 us | 218.11 us | +6.2%，但需复测 |
| 7168    | 352.26 us | 351.17 us | 基本持平 |
| 8192    | 396.29 us | 397.28 us | 轻微退化 |
| 12288   | 572.42 us | 574.46 us | 轻微退化 |

correctness：

```text
seq512 no-arena: max_abs_diff = 0.0
seq512 outlier-only: max_abs_diff = 0.0
seq4096 no-arena: max_abs_diff = 0.0
seq4096 outlier-only: max_abs_diff = 0.015625
```

seq4096 outlier-only 的 `0.015625` 与当前 V4 macro8 baseline 相同，不是
block128 新增误差；来源是 outlier 场景下 BF16 输出和当前参考路径的归约/舍入差异。

### 11.4 结论

block128 score-buffer/two-pass 没有达到默认启用门槛。原因是：

```text
收益:
  macro 内只做一次 running max / denom 更新
  no-arena 长上下文有 0.4%-0.6% 小幅改善

成本:
  scores[4][128] shared memory 写入和读取
  pass1/pass2 之间额外同步
  V decode 仍然逐 16-token page 执行
  没有减少 K/V payload decode 次数
```

因此当前策略：

```text
保留为实验开关
不设为默认
不用于 E2E 默认 benchmark
后续若继续优化，需要减少 score-buffer shared 往返，或进入真正的
CUTLASS/CUTE block-N pipeline
```
