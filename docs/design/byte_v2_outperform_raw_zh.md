# ByteV2 性能超过 raw vLLM 的优化设计

本文档讨论如何让 ByteV2 KV cache 后端在端到端 decode 性能上超过 raw vLLM。
这里的 raw vLLM 指当前生产默认路径：BF16/FP16 KV cache、FlashAttention 后端、
torch.compile 和 CUDA graph 均开启。

当前 ByteV2 已经具备 e2e 可用性：

- prefill attention 复用 raw FlashAttention fast path。
- continuation prefill 的 PyTorch fallback 已被规避。
- compressed-only cache update 已改成 block-parallel native CUDA。
- decode 已有 GQA WMMA + split-K/page-parallel kernel。
- ByteV2 backend 已启用 single-token decode CUDA graph。

但最新同机 benchmark 表明，当前实现距离 raw vLLM 仍有明显差距：

```text
模型: /mnt/sda1/yxz/byte_v2/Meta-Llama-3-8B-Instruct
配置: batch=1, prompt_len=512, dtype=bf16, block_size=16, cudagraph on
raw:    kv_cache_dtype=auto, FLASH_ATTN backend
ByteV2: compressed-only + 3% sparse fallback pool + split-K=8
结果: benchmarks/profiles/raw_vs_bytev2_best_p512_d16_d32_cudagraph.json
```

| decode len | raw output tok/s | ByteV2 output tok/s | ByteV2 / raw |
|---:|---:|---:|---:|
| 16 | 35.13 | 9.37 | 26.7% |
| 32 | 35.43 | 9.72 | 27.4% |

同一轮初始化日志中，在约 8.41 GiB 可用 KV cache 内存下：

| 模式 | KV capacity | 相对 raw |
|---|---:|---:|
| raw | 68,864 tokens | 1.00x |
| ByteV2 compressed-only | 87,808 tokens | 1.28x |

结论是：当前 ByteV2 已经换到了更多 KV token capacity，但单请求 decode
吞吐只有 raw 的约 27%。要超过 raw，不能只做小修补，必须同时提升压缩收益和
decode kernel 效率。

## 性能超过 raw 的必要条件

端到端 decode 单步可以粗略分成：

```text
T_raw  = T_non_attn + T_raw_attn
T_bv2  = T_non_attn + T_bv2_attn + T_bv2_update + T_bv2_runtime
```

`T_non_attn` 包括 RMSNorm、QKV projection、MLP、sampling 等，ByteV2 和 raw
基本相同。ByteV2 只能主要优化 `T_raw_attn` 中的 KV cache 读取部分，同时会引入：

- K/V 解压计算。
- page status、fallback metadata 读取。
- sparse fallback raw block/tile 分支。
- split-K partial workspace 写回和 reduce。
- cache update 压缩开销。
- 更复杂的 CUDA graph capture 约束。

如果 ByteV2 的有效压缩率是 `C_eff`，理想情况下 attention 读取 KV 的成本最多
降低到 `1 / C_eff`。要比 raw 更快，新增开销必须满足：

```text
ByteV2 extra overhead < T_raw_attn * (1 - 1 / C_eff)
```

当前 fast tile 格式是 16x16 BF16 tile：

```text
raw tile:      16 * 16 * 2 = 512 bytes
ByteV2 tile:  base 1 + low 256 + packed code 128 + fallback 1 = 386 bytes
tile 压缩率:  512 / 386 = 1.326x
```

以 Llama-3-8B 的常见布局估算：

```text
num_kv_heads = 8
head_size = 128
block_size = 16

raw block bytes =
  16 tokens * 8 kv heads * (128 K + 128 V) * 2 bytes
  = 65,536 bytes

compressed page bytes =
  header 16 + 8 kv heads * (8 K tiles + 8 V tiles) * 386
  = 49,424 bytes

block 压缩率 = 65,536 / 49,424 = 1.326x
```

加上 3% sparse fallback pool 和 metadata 后，实际分配收益接近 benchmark 中看到的
1.28x。这个压缩率太低：即使 decode attention 完全 memory-bound，理论上能节省的
attention 时间也只有约 25%。而当前 ByteV2 decode kernel 还额外付出了显著解压和
runtime 开销，所以很难靠现有 12-bit 格式在单请求 latency 上超过 raw FlashAttention。

因此，超过 raw 有两条路线：

1. **同格式极致优化路线**：保持当前 12-bit ByteV2 格式，把 decode kernel 和
   cache update 优化到接近 raw，目标是接近 raw 性能，同时保留 1.25-1.33x
   KV capacity。
2. **性能格式升级路线**：新增更高压缩率的 ByteV2 performance format，例如
   K12/V8、K8/V8、tile-level fallback、多窗口 tile，使有效 KV 读取减少 1.6-2.5x。
   只有这条路线才更现实地支持单请求 decode 性能超过 raw。

## 当前主要瓶颈判断

当前 e2e 结果是 ByteV2 约 9.7 tok/s，raw 约 35.4 tok/s。ByteV2 要追平 raw，
需要约 3.6x 的总提升。结合当前实现和 raw 的成熟度，瓶颈主要在以下几类。

### 1. 当前压缩率不够高

12-bit tile 只减少约 25% KV 字节。raw vLLM 的 paged decode kernel 已经非常成熟，
如果 ByteV2 只少读 25%，但解压、fallback、split reduce 多做一堆工作，很难赢。

### 2. block-level sparse fallback 太粗

当前 compressed-only sparse fallback 是 block-level：只要一个 16-token block 中
有任意 K/V tile 无法被 ByteV2 exponent window 无损表示，就把整个 block 放进 raw
fallback pool。这样会产生两个问题：

- 实际压缩率下降。
- decode 中遇到 raw fallback block 时会读完整 raw BF16 block，ByteV2 的带宽优势
  在该 block 上消失。

更合理的是 tile-level fallback：只有不可压缩的 16x16 tile 走 raw fallback，其余
tile 仍保持压缩。

### 3. decode kernel 仍不够接近 raw 的数据流效率

当前 split-K GQA WMMA kernel 已经比单 CTA decode 好，但仍有明显开销：

- K/V 解压先写 shared memory，再用 WMMA 读取，load/decode/store 路径较重。
- page status 和 fallback 分支在 decode 主循环里执行。
- split-K stage1 写 FP32 partial workspace，stage2 再读回来 reduce。
- long context 通过更多 CTA 增加并行度，但也增加 launch、workspace 和 reduce 成本。
- 未根据 `seq_len`、`q_per_kv`、`head_size` 做 autotune。

### 4. cache update 压缩仍在主路径上

prefill attention 已经复用 raw FlashAttention，但 K/V 写入 ByteV2 cache 需要压缩。
当前 block-parallel update 比之前更合理，但仍可能在大 prompt、多层、长 batch 下占用
明显时间。raw vLLM 写 KV cache 只是 layout copy，ByteV2 还要计算 exponent window、
写 packed payload、处理 fallback。

### 5. benchmark 场景对 ByteV2 不够有利

batch=1、prompt=512/1024、decode 16/32 时，raw vLLM 的 CUDA graph + FlashAttention
启动开销很低，且整个模型非 attention 部分占比高。ByteV2 的优势更可能出现在：

- context 很长，KV read 成为主瓶颈。
- batch/concurrency 高，raw 被 KV capacity 限制。
- serving 场景中 raw 因 KV capacity 不能维持同样并发，而 ByteV2 可以。

但是当前 ByteV2 单 token 性能只有 raw 27%，即使 capacity 多 28%，端到端吞吐仍不足。
因此必须先把单 token decode 至少提高到 raw 的 70-90% 区间，capacity 收益才有机会
转化为总吞吐超过 raw。

## 总体目标

建议把目标拆成三个层级。

### 目标 A：当前 12-bit ByteV2 接近 raw

适合作为近期目标。

验收标准：

- Llama-3-8B, batch=1, prompt=512/1024, decode=32/64：
  - ByteV2 output tok/s 达到 raw 的 70% 以上。
- KV capacity 保持 raw 的 1.25x 以上。
- 3% fallback pool 不耗尽。
- cudagraph 默认可用。

这个目标不保证超过 raw，但可以证明 runtime 路径足够健康。

### 目标 B：高压缩 performance format 超过 raw 单请求 decode

适合作为核心研发目标。

验收标准：

- 有效 KV read bytes 降到 raw 的 50-60% 或更低。
- 单请求长 context decode 中，ByteV2 output tok/s 超过 raw 1.05x。
- accuracy 回归可接受，例如短集 perplexity、固定 prompt 输出一致性或任务集指标
  在可控范围内。

### 目标 C：serving 吞吐超过 raw

适合作为最终产品目标。

验收标准：

- 在固定 GPU memory、固定 latency SLO 下，ByteV2 可承载更多并发请求。
- ShareGPT/long-context 合成数据上，request throughput 或 output tok/s 超过 raw。
- raw 因 KV cache capacity 需要更低并发或更高 preemption 时，ByteV2 保持稳定。

## 优化路线一：把现有 12-bit ByteV2 做到接近 raw

### 1. 建立 decode-only microbenchmark

当前 e2e benchmark 混合了 prefill、cache update、decode、sampling、engine 调度。
下一步需要一个只测 attention decode kernel 的 benchmark：

```text
输入:
  query: [B, num_heads, head_dim]
  ByteV2 kv_cache: [num_blocks, page_bytes]
  block_table: [B, max_blocks]
  seq_lens: [B]

输出:
  attention output

对照:
  raw paged attention kernel
  current ByteV2 decode kernel
  ByteV2 candidate kernels
```

必须记录：

- kernel elapsed time。
- HBM read/write bytes 估算和 Nsight 实测。
- achieved occupancy。
- tensor core utilization。
- memory throughput。
- fallback block/tile ratio。
- split-K reduce 时间占比。

没有这个 benchmark，很难判断优化是否真的作用在 decode kernel 上。

### 2. decode kernel v2：减少 shared memory 往返

当前路径：

```text
global compressed bytes
  -> unpack to shared BF16 K tile
  -> WMMA QK
  -> unpack to shared BF16 V tile
  -> WMMA PV
```

优化方向：

- 使用 vectorized load 读取 `low_bytes` 和 packed code，例如按 16B/32B 对齐读取。
- 在 warp 内直接 unpack 成 WMMA 需要的 shared-memory swizzled layout，减少中间格式。
- K/V tile 双缓冲：

```text
buffer 0: 当前 tile 做 QK/PV
buffer 1: 下一 tile cp.async/load + unpack
```

- 对 `head_size=128` 固化 fast path，避免通用 head size 分支。
- 对 Llama GQA 固化 `q_per_kv=4` fast path，减少循环和边界判断。
- page status 在 block 开头一次性读到 shared/register，尽量让 warp 内分支一致。
- fallback 分支拆 kernel：
  - 全 compressed pages 使用 pure compressed kernel。
  - 有 fallback 的 batch 使用 fallback-aware kernel。

预期收益：

- 当前 ByteV2 9.7 tok/s 到 15-20 tok/s。
- 如果 fallback 很少，pure compressed kernel 应明显优于统一分支 kernel。

### 3. split-K reduce 融合或轻量化

当前 split-K 是两阶段：

```text
stage1: 每个 page chunk 写 partial output + LSE
stage2: reduce partial output
```

这提升了长 context 并行度，但每层每步多一次 kernel launch 和 FP32 workspace 读写。
优化选项：

- 小 `seq_len` 不启用 split-K，继续单 CTA。
- 中等 `seq_len` 使用 2/4 split。
- 长 `seq_len` 使用 8/16 split。
- 根据 `seq_len` 动态选择 split，而不是只看 `block_table.size(1)`。
- stage2 用 warp-level reduce，固定 `head_size_v=128` fast path。
- 对 batch=1 长 context，尝试 persistent CTA 聚合：同一 request/head 的多个 page
  chunk 在同一个 cooperative group 内归约，减少 global partial write。
- 在 Hopper/Blackwell 上尝试 cluster-level cooperative reduction。

预期收益：

- 避免短 decode 被 split-K overhead 拖慢。
- 长 context 保留并行度。

### 4. cache update 与 prefill 重叠

raw vLLM 的 KV update 是轻量 copy；ByteV2 需要压缩。要超过 raw，不能让 prefill
cache update 长时间占主路径。

优化方案：

1. 继续保留 prefill attention raw FlashAttention。
2. KV projection 产生 raw K/V 后，ByteV2 compression 在独立 CUDA stream 上异步执行。
3. 当前层后续 attention output/MLP 与该层 KV compression 尽可能 overlap。
4. decode 使用该 KV 前，通过 event 保证 compression 完成。

示意：

```text
main stream:
  layer L qkv projection
  raw FlashAttention prefill
  output projection
  MLP
  layer L+1 ...

kv stream:
  wait K/V ready for layer L
  compress layer L K/V blocks
  record layer L cache-ready event
```

需要修改：

- `ByteV2AttentionImpl.do_kv_cache_update()` 支持 async stream。
- scheduler/model runner 在 decode 前等待相关 cache-ready events。
- cudagraph capture 时使用固定 stream/event 或在 capture 外完成 prefill compression。

风险：

- stream/event 管理复杂。
- CUDA graph capture 下跨 stream side effect 需要严格控制。
- 对短 prompt 可能得不偿失，需要阈值。

### 5. fallback 统计和自动策略

当前 fallback pool stats 是粗粒度累计值。需要更细的 runtime 指标：

- 每层 compressed block 数。
- 每层 raw fallback block 数。
- decode 时实际访问 fallback 的比例。
- fallback 对输出 token latency 的影响。
- pool slot 复用情况。

如果某层 fallback 比例过高，可以自动切换策略：

- 对该层使用 raw KV cache。
- 对该层使用更宽 exponent window 格式。
- 对该层使用 lossy clamp/performance mode。

## 优化路线二：新增更高压缩率的 ByteV2 performance format

当前 12-bit 格式最大问题是压缩率只有 1.33x。要真正超过 raw，建议新增
`byte_v2_perf` 或 `byte_v2_v3` 格式，而不是只优化 12-bit codec。

### 1. K12/V8 混合格式

观察：

- K 影响 attention scores，误差会被 softmax 放大。
- V 影响 weighted sum，通常对量化更宽容。

第一阶段可保持 K 使用当前 12-bit lossless-window 格式，V 改为 8-bit tile quant。

估算：

```text
K raw: 16 bits/value -> 12 bits/value
V raw: 16 bits/value -> 8 bits/value

整体 K+V: 32 bits -> 20 bits
压缩率: 1.6x
```

优点：

- 比全 8-bit 更稳。
- V decode 可以更轻，PV 读带宽明显下降。
- 对当前结构改动相对可控。

需要新增：

- V8 tile codec。
- V8 -> BF16 shared/fragment unpack。
- accuracy benchmark。
- fallback 策略：V8 可允许 lossy，不一定需要 raw fallback。

### 2. K8/V8 全 8-bit performance format

目标是 2x KV read reduction。

可选格式：

```text
per 16x16 tile:
  scale/base metadata: 2-8 bytes
  data: 256 bytes
```

候选编码：

- FP8 E4M3/E5M2 + per-tile scale。
- sign + exponent delta + truncated mantissa。
- int8 symmetric/asymmetric per-tile scale。
- per-row scale + int8，以改善 score 稳定性。

硬件路径：

- Ampere：解码到 BF16 后用 BF16 WMMA。
- Hopper/Blackwell：如果使用 FP8，可尝试 FP8 Tensor Core，进一步降低解码成本。

风险：

- K8 可能影响 attention scores，需要严格 accuracy 验证。
- fallback 或 mixed precision 规则会更复杂。

### 3. tile-level fallback

不论 12-bit 还是 8-bit，fallback 都应从 block-level 改为 tile-level。

当前 block-level fallback：

```text
if any tile in block is not compressible:
  whole block -> raw fallback pool
```

目标 tile-level fallback：

```text
for each tile in block:
  if tile compressible:
    store compressed tile
  else:
    store raw fallback tile
```

新的 page metadata：

```text
page header:
  status
  valid_rows
  tile_fallback_bitmap_offset

tile metadata:
  fallback bitmap: num_kv_heads * (k_tiles + v_tiles) bits
  fallback slot id per fallback tile, or compact index into tile fallback pool
```

fallback pool：

```text
tile_fallback_pool: [num_fallback_tiles, 16, 16, bf16]
tile_fallback_ids:  [num_blocks, total_tiles] int32 or compact sparse metadata
```

收益：

- 避免一个坏 tile 让整个 block 退回 raw。
- 提高有效压缩率。
- decode 中大部分 tile 仍走 compressed fast path。

代价：

- metadata 变多。
- cache update 更复杂。
- decode 需要 tile-level branch。

工程上可以先做两版：

1. 简单版：`fallback_tile_ids[num_blocks, total_tiles] int32`，实现快，但 metadata 较大。
2. 优化版：bitmap + compact ids，降低 metadata。

### 4. 多窗口 tile，减少 fallback

当前 ByteV2 12-bit tile 只能覆盖一个 16-wide exponent window。不可压缩 tile 可能只是
少量 outlier 破坏窗口。

可以新增 two-window tile：

```text
tile header:
  base0
  base1
  selector bits
  low/code payload
```

或者 outlier-aware tile：

```text
main window compressed values
small outlier list:
  positions + raw bf16 values
```

目标是让绝大多数 tile 不用 raw fallback，同时保持比 raw 小很多。

风险是 header 和 selector 可能吃掉压缩收益，所以必须用真实模型 KV 分布评估。

## 优化路线三：用 capacity 优势超过 raw serving 吞吐

单请求 latency 超过 raw 最难，因为 raw kernel 太成熟。更现实的产品目标是：

```text
固定 GPU memory + 固定 latency SLO 下，ByteV2 serving throughput > raw serving throughput
```

需要让 ByteV2 scheduler 利用更大的 KV capacity：

- 同样 `gpu_memory_utilization` 下允许更大 `max_num_seqs`。
- 长 context workload 中减少 preemption。
- prefix cache 命中时避免 raw 无法承载更多 blocks。
- 对 raw 会 OOM 的长上下文 batch，ByteV2 能保持 batch 内并发。

但是 capacity 优势只有 1.28x 时，如果单 token 速度只有 0.27x raw，总吞吐不会赢。
所以 serving 超过 raw 的前提仍然是 ByteV2 单 token 性能至少达到 raw 的 70-80%。

## 推荐实施阶段

### Phase 1：测清楚 decode kernel 上限

目标：把优化从 e2e 黑盒转成可解释的 kernel 指标。

任务：

- 新增 ByteV2 decode-only benchmark。
- 增加 raw paged attention 对照。
- Nsight 指标自动导出：
  - kernel time
  - dram bytes
  - tensor core utilization
  - occupancy
  - split reduce 占比
  - fallback 访问比例
- 对 `seq_len = 512, 1024, 2048, 4096, 8192` 和 `batch = 1, 4, 8` 扫描。

完成标准：

- 能说明 ByteV2 9.7 tok/s 中 decode kernel、cache update、runtime 分别占多少。
- 能确认当前 12-bit kernel 是否 memory-bound 还是 compute/unpack-bound。

### Phase 2：现有格式 decode kernel v2

目标：不改压缩格式，把当前 ByteV2 拉到 raw 的 50-70%。

任务：

- Llama GQA fast path：固定 `head_size=128`、`q_per_kv=4`。
- pure compressed kernel 与 fallback-aware kernel 分离。
- vectorized compressed payload load。
- K/V unpack 双缓冲。
- split-K autotune。
- stage2 reduce 轻量化。
- capture-safe preallocated workspace。

完成标准：

- prompt=512/1024, decode=32/64 下 ByteV2 >= raw 50%。
- 长 context 下 ByteV2 decode kernel 时间随压缩读字节下降而下降。

### Phase 3：tile-level fallback

目标：提高有效压缩率，并减少 raw fallback 对 decode 的破坏。

任务：

- 设计 tile fallback metadata。
- allocator 预算 tile fallback pool。
- cache update 支持 tile fallback。
- decode 支持 per-tile compressed/raw 读取。
- stats 输出 per-layer fallback tile ratio。

完成标准：

- 相比 block-level fallback，KV capacity 明显提升。
- fallback-heavy 层 decode 时间下降。
- pool 不耗尽，且 fallback metadata 开销可接受。

### Phase 4：K12/V8 performance format

目标：将有效压缩率从 1.3x 提高到约 1.6x。

任务：

- 实现 V8 tile codec。
- cache update 压缩 V8。
- decode PV 路径支持 V8 -> BF16/FP32。
- accuracy benchmark：
  - perplexity smoke
  - fixed prompt output drift
  - small downstream task 或内部评估集
- 增加配置：

```text
kv_cache_dtype=byte_v2
VLLM_BYTE_V2_FORMAT=fast12 | k12v8
```

完成标准：

- ByteV2 >= raw 70-90%。
- KV capacity >= raw 1.5x。
- accuracy 无明显退化。

### Phase 5：K8/V8 或 FP8 Tensor Core path

目标：单请求长 context decode 超过 raw。

任务：

- 实现 K8/V8 format。
- Hopper/Blackwell 上尝试 FP8 MMA。
- Ampere 上评估 int8/FP8 decode-to-BF16 的开销。
- 增加 layer/head 自适应策略：
  - 敏感层用 K12/V8。
  - 非敏感层用 K8/V8。
  - 高 fallback 层用 raw 或 multi-window。

完成标准：

- 有效 KV read bytes <= raw 55%。
- 长 context decode ByteV2 > raw 1.05x。
- serving throughput 在固定 SLO 下超过 raw。

### Phase 6：serving 策略和自动选择

目标：把 ByteV2 从实验模式变成可自动选择的生产策略。

任务：

- 根据 model、context、batch、GPU capability 自动选择：
  - raw
  - ByteV2 fast12
  - ByteV2 k12v8
  - ByteV2 k8v8
- 对短 context 保持 raw。
- 对长 context 或 raw KV capacity 不足时启用 ByteV2。
- 支持 prefix cache native paged prefill，避免禁用 prefix cache 造成产品缺口。

完成标准：

- 默认策略不会让短请求变慢。
- 长 context/high concurrency workload 自动超过 raw。

## 关键实现细节

### decode kernel 数据流建议

目标 kernel 应尽量接近以下结构：

```text
for request/head group:
  load Q for q_per_kv heads once
  for page chunk assigned to CTA/warpgroup:
    async load compressed K tile metadata + payload
    unpack K tile to WMMA-ready shared layout
    BF16/FP8 MMA QK
    online softmax update
    async load compressed V tile payload
    unpack V tile
    MMA PV
  write partial output + LSE
reduce partial outputs
```

需要避免：

- 每个 value 元素单独分支 fallback。
- 每个 tile 重复计算复杂 offsets。
- 对 short context 也启用 split-K。
- stage1/stage2 workspace 动态分配。
- capture 中任何 D2H sync。

### page layout 建议

当前 flat page 适合集成，但 performance format 应考虑 decode 访问顺序：

```text
page:
  header
  compressed K tiles, grouped by kv_head then dim_tile
  compressed V tiles, grouped by kv_head then dim_tile
  metadata/fallback bitmap
```

为了 coalesced load，可考虑把 low bytes 和 packed code 分成独立连续区域：

```text
tile:
  base/fallback metadata
  low_bytes[256]
  code_packed[128]
```

如果 decode kernel 中 `low_bytes` 和 `code_packed` 总是一起读，则保持当前布局。
如果 unpack 需要分 warp 处理，可能独立 SOA 更好：

```text
page:
  tile_bases[]
  tile_fallback_flags[]
  all_low_bytes[]
  all_code_packed[]
```

需要用 microbenchmark 选择，而不是凭直觉。

### cache update 压缩建议

当前 one CTA per physical block 是正确方向，但还可以继续优化：

- one CTA per block 改为 one CTA per block + multiple warps per tile group。
- parallel histogram/window selection，而不是单线程扫描所有 tile。
- 对 prompt 连续 slot 使用 specialized contiguous path。
- 对 decode append token，只写 partial raw/compressed tail；full block 才 finalize。
- 对 full prompt block 使用 batched compression，不经过 token-level mark。

### CUDA graph 约束

ByteV2 所有 production path 必须 graph-safe：

- 不在 capture 中 `cudaMemcpyDeviceToHost`。
- 不在 capture 中 `cudaStreamSynchronize`。
- 不在 capture 中临时创建 fallback pool。
- 不在 attention forward 中分配动态 workspace。
- split-K workspace 由 backend/model runner 预分配并复用。

当前 cache update 已在 capture 中跳过 host validation，这是最低限度。后续需要把
decode partial workspace 也改成持久 workspace，否则 graph capture 虽然可跑，但仍可能
有隐藏 allocator 开销或 replay 限制。

## 风险和取舍

| 风险 | 影响 | 应对 |
|---|---|---|
| 12-bit 格式压缩率太低 | 单请求 latency 难超过 raw | 新增 K12/V8、K8/V8 format |
| K8 影响 attention score | 输出质量下降 | 分层/分头策略，K 保持 12-bit，先做 V8 |
| tile fallback metadata 太大 | capacity 被 metadata 吃掉 | 先简单实现验证，再 compact bitmap |
| split-K reduce 开销过大 | short context 变慢 | seq_len autotune，短 context 走单 CTA |
| cache update 压缩占主路径 | prefill 慢 | 异步 stream overlap，contiguous prompt fast path |
| cudagraph 被 side effect 破坏 | production 性能落后 raw | 所有 op 做 capture-safe 检查和持久 workspace |
| 只在特定 workload 超过 raw | 产品收益不稳定 | 自动策略：短 context raw，长 context ByteV2 |

## 推荐下一步

建议不要直接继续堆 e2e benchmark，而是按以下顺序推进：

1. **写 decode-only microbenchmark。**
   先证明当前 ByteV2 decode kernel 距 raw paged attention kernel差多少，并拆出
   unpack、fallback、split reduce 的时间。

2. **实现 pure compressed fast kernel。**
   当 batch 中没有 fallback block/tile 时，完全去掉 fallback 分支。当前 benchmark 的
   fallback pool 没耗尽，但不代表 decode 热路径没有被 fallback-aware 分支拖慢。

3. **实现 tile-level fallback。**
   这是继续提高有效压缩率的关键，不然 block-level fallback 会持续侵蚀收益。

4. **做 K12/V8 format。**
   这是最现实的第一版 performance format。目标不是一步到位超过 raw，而是把
   ByteV2 从 raw 的 27% 拉到 70-90%。

5. **再做 K8/V8 或 FP8 path。**
   真正要单请求超过 raw，大概率需要 2x 级别的 KV 读取压缩，而不是当前 1.33x。

6. **最终做 serving benchmark。**
   在 ShareGPT/long-context/high-concurrency 场景中验证固定 GPU memory 下是否超过 raw。

## 判断是否值得继续的里程碑

为了避免长期优化但看不到超过 raw 的希望，建议设置硬性里程碑：

| 阶段 | 必须达到的指标 |
|---|---|
| decode-only benchmark 完成 | 能解释 ByteV2 kernel 时间组成 |
| 现有 12-bit kernel v2 | e2e >= raw 50% |
| tile-level fallback | KV capacity >= raw 1.4x，fallback 访问明显下降 |
| K12/V8 | e2e >= raw 70%，capacity >= raw 1.5x |
| K8/V8 或 FP8 | 长 context decode > raw 1.05x |
| serving 策略 | 固定 SLO 下 request throughput > raw |

如果 12-bit kernel v2 仍达不到 raw 50%，说明当前 kernel/dataflow 还有根本问题，应优先
重写 decode kernel，而不是进入 K12/V8。  
如果 K12/V8 达不到 raw 70%，说明解压/format 设计仍太重，应考虑 FP8 Tensor Core
路径或更激进的量化格式。

## 简短结论

当前 ByteV2 只有 raw vLLM 约 27% 的 e2e decode 性能，主要原因不是 CUDA graph 或
prefill fallback 了，而是：

- 当前 12-bit ByteV2 格式压缩率只有约 1.33x。
- block-level fallback 会进一步降低有效压缩收益。
- decode kernel 的 unpack、fallback branch、split reduce 还没有达到 raw
  FlashAttention 级别的效率。

要超过 raw，最现实的路线是：

```text
decode-only profile
  -> 12-bit pure compressed fast kernel
  -> tile-level fallback
  -> K12/V8 performance format
  -> K8/V8 或 FP8 Tensor Core path
  -> serving 自动策略
```

只优化当前 12-bit 格式有机会接近 raw，但单请求性能超过 raw 的概率不高。真正超过 raw
需要更高有效压缩率，至少 1.6x，最好接近 2x，同时 decode kernel 必须把解压开销压到
raw attention 节省时间以内。
