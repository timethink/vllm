# ByteV2 当前 Kernel 优化实验设计

本文档只讨论**当前 ByteV2 kernel 和 runtime 路径**的优化实验，不讨论大幅改动
压缩格式，例如 K12/V8、K8/V8 或 tile-level fallback。目标是为后续代码修改提供
一套逐项实验方案：每次只实现一个候选优化，跑 correctness、kernel benchmark 和
e2e benchmark；如果没有可测收益，则不保留该改动。

相关背景文档：

- `docs/design/byte_v2_kv_cache_zh.md`
- `docs/design/byte_v2_outperform_raw_zh.md`

当前核心实现位于：

- `csrc/libtorch_stable/cache_kernels.cu`
- `vllm/v1/attention/backends/byte_v2_attn.py`
- `vllm/_custom_ops.py`

## 当前基线

最新同机 e2e 对照：

```text
模型: /mnt/sda1/yxz/byte_v2/Meta-Llama-3-8B-Instruct
配置: batch=1, prompt_len=512, dtype=bf16, block_size=16, cudagraph on
raw:    kv_cache_dtype=auto, FLASH_ATTN backend
ByteV2: compressed-only + 3% sparse fallback pool + split-K=8 + cudagraph
结果: benchmarks/profiles/raw_vs_bytev2_best_p512_d16_d32_cudagraph.json
```

| decode len | raw output tok/s | ByteV2 output tok/s | ByteV2 / raw |
|---:|---:|---:|---:|
| 16 | 35.13 | 9.37 | 26.7% |
| 32 | 35.43 | 9.72 | 27.4% |

ByteV2 的 KV capacity 有收益：

| 模式 | KV capacity | 相对 raw |
|---|---:|---:|
| raw | 68,864 tokens | 1.00x |
| ByteV2 compressed-only | 87,808 tokens | 1.28x |

因此当前优化重点不是“是否能跑通”，而是把 ByteV2 decode/cache-update 路径的
额外开销压下去。

## 当前 kernel 结构

### Decode kernel

当前 native decode 入口：

```text
torch::stable::Tensor byte_v2_paged_decode_attention(...)
```

主要 CUDA kernel：

```text
byte_v2_paged_decode_attention_kernel
byte_v2_paged_decode_attention_gqa_shared_kernel
byte_v2_paged_decode_attention_gqa_wmma_kernel
byte_v2_paged_decode_attention_gqa_wmma_split_stage1_kernel
byte_v2_paged_decode_attention_split_reduce_kernel
```

当前最优路径通常是：

```text
GQA + BF16 WMMA + split-K stage1 + split reduce
```

核心流程：

```text
load query -> q_shared
for logical page chunk:
  read block_table
  page = kv_cache + physical_block * page_size_bytes
  read/decode K tile -> k_shared
  read/decode V tile -> v_shared
  WMMA QK -> scores
  scalar online softmax -> p_shared
  WMMA PV -> acc
write partial output + LSE
stage2 reduce partials
```

已知问题：

- `byte_v2_load_kv_bits()` 在每个元素上重复检查 page status、fallback metadata。
- compressed page 的 tile fallback flag 实际上通常为 0，但当前 decode 仍按通用路径。
- softmax 和 `p_shared` 填充主要由 `tid == 0` 串行完成。
- `q_per_kv=4` 时 WMMA 仍按 16 个 Q row 计算，只有前 4 行有效。
- split-K 会多写/读 FP32 partial workspace，并多一次 reduce kernel。
- decode op 先分配 `decode_out`，再由 Python backend copy 到 vLLM output buffer。
- split-K partial workspace 当前在 op 内部创建，尚未作为 backend 持久 workspace 管理。

### Cache update kernel

当前 compressed-only cache update 路径：

```text
byte_v2_init_cache_update_kernel
byte_v2_mark_touched_tokens_kernel
byte_v2_compress_touched_blocks_kernel
```

当前已经比早期 token-scan 版本更合理：`byte_v2_compress_touched_blocks_kernel`
是 one CTA per physical block。但仍有问题：

- 每次 update 都初始化 `valid_rows/touched_flags/packed_flags/overwrite_flags`。
- 对 prompt 连续 slot 仍先 mark token，再按 block 查 token index。
- `raw_staging` 按 `num_tokens * raw_block_bytes` 分配，可能偏大。
- capture 中已经跳过 host-side validation，但非 capture 仍有 D2H sync。
- exponent window 选择和 compress store 在 CTA 内仍有大量串行逻辑。

## 实验原则

后续按本文档修改时，必须遵守以下原则。

### 单实验最小化

每次只改一个明确候选方案，例如：

```text
只改 split-K heuristic
只加 pure compressed decode kernel
只改 softmax 并行化
只加 direct-output op
```

不要把多个候选方案混在一个 patch 里，否则无法判断收益来源。

### 全部候选方案必须可回退

每个实验都需要开关，建议使用 env：

```text
VLLM_BYTE_V2_KERNEL_VARIANT=baseline|...
VLLM_BYTE_V2_DECODE_SPLIT_K=...
VLLM_BYTE_V2_DISABLE_DIRECT_OUTPUT=...
```

如果实验无收益，删除实验代码和开关，不保留长期 dead code。

### 保留标准

一个实验必须同时满足：

1. correctness 单测通过。
2. decode-only microbenchmark 有稳定收益，或 e2e 有稳定收益。
3. 不破坏 cudagraph。
4. 不降低 KV capacity。
5. 不让 fallback pool 更容易耗尽。

建议阈值：

| 改动类型 | 保留门槛 |
|---|---:|
| 低风险、小代码量 | e2e output tok/s >= +2%，且无显著回退 |
| 中等复杂度 | e2e output tok/s >= +5%，或 decode kernel time >= -8% |
| 大 kernel 重写 | e2e output tok/s >= +10%，或为后续关键实验提供必要基础 |
| 只改善特定长 context | 对目标 `seq_len` 段 >= +10%，短 context 回退 <= 2% |

如果 3 次重复 benchmark 的中位数收益小于阈值，则回滚。

### 不保留标准

满足以下任一条件就不保留：

- 只在单次 benchmark 中提升，重复后不稳定。
- e2e 提升来自噪声，decode-only benchmark 没有对应改善。
- 短 context 明显变慢，且无法通过 heuristic 避开。
- cudagraph capture/replay 失败。
- 代码复杂度明显增加但收益低于阈值。
- 引入额外 allocator/D2H sync/runtime 分支。

## 必做基础设施：decode-only benchmark

在继续改 kernel 前，必须先补一个 decode-only benchmark。当前 e2e 包含模型前后处理、
prefill、cache update、sampling、scheduler 和 CUDA graph，不能准确判断单个 kernel
的收益。

### 设计

新增脚本建议：

```text
benchmarks/kernels/benchmark_byte_v2_decode_kernel.py
```

输入参数：

```text
--batch-size
--seq-len
--num-heads
--num-kv-heads
--head-size
--head-size-v
--block-size
--num-runs
--warmup-runs
--fallback-ratio
--split-k
--variant
```

构造数据：

- 随机 raw K/V。
- 用现有 `byte_v2_reshape_and_cache` 或 Python pack helper 填充 ByteV2 cache。
- 构造 block_table 和 seq_lens。
- query 使用 BF16。

输出指标：

```text
variant
batch_size
seq_len
num_heads
num_kv_heads
q_per_kv
head_size
split_k
fallback_ratio
mean_us
median_us
p50/p90/p99_us
effective_output_tok_s
max_abs_diff
max_rel_diff
```

对照：

- 当前 ByteV2 baseline。
- `VLLM_BYTE_V2_DECODE_SPLIT_K=1` 单 CTA。
- `VLLM_BYTE_V2_DECODE_SPLIT_K=2/4/8/16`。
- raw reference 或 raw paged attention kernel，如果接入方便。

扫描矩阵：

```text
seq_len: 128, 256, 512, 1024, 2048, 4096, 8192
batch:   1, 2, 4, 8
split_k: 1, 2, 4, 8, 16
fallback_ratio: 0, 0.01, 0.03, 0.10
```

保留标准：

- benchmark 本身不改变生产路径。
- 输出 JSON/CSV，方便后续比较。
- 能作为每个 kernel 实验的固定回归命令。

## 候选方案 1：direct output，去掉 decode_out 分配和 copy

### 问题

当前 Python backend 中：

```python
decode_out = ops.byte_v2_paged_decode_attention(...)
output[:num_decode_tokens].copy_(decode_out)
```

也就是说 native op 内部分配 output tensor，返回后再 copy 到 vLLM 的 output buffer。
raw backend 通常直接写入调用方传入的 output。

### 方案

新增 inplace op：

```text
byte_v2_paged_decode_attention_out(
    query,
    kv_cache,
    block_table,
    seq_lens,
    output,
    ...
) -> None
```

内部 kernel 直接写 `output`，不再 `torch::stable::empty` 返回 decode_out。

Python backend：

```python
ops.byte_v2_paged_decode_attention_out(..., output=output[:num_decode_tokens])
```

保留旧 returning op 作为测试/调试 fallback，或在实验确认后把旧路径删除。

### 预期收益

- eager 下减少一次 allocation 和一次 copy。
- cudagraph 下减少 captured copy node。
- 对短 decode 更可能有效。

### 实验命令

```bash
CUDA_VISIBLE_DEVICES=2 \
VLLM_BYTE_V2_DECODE_SPLIT_K=8 \
.venv/bin/python benchmarks/benchmark_byte_v2_decode_e2e.py \
  --model /mnt/sda1/yxz/byte_v2/Meta-Llama-3-8B-Instruct \
  --modes byte_v2_compressed_only \
  --prompt-len 512 \
  --decode-lens 16,32 \
  --batch-size 1 \
  --num-runs 3 \
  --warmup-decode-len 4 \
  --dtype bfloat16 \
  --block-size 16 \
  --gpu-memory-utilization 0.55
```

保留标准：

- e2e >= +2%。
- cudagraph capture 成功。
- 单测覆盖 returning op 和 out op 数值一致。

不保留标准：

- e2e 无提升。
- 为兼容 output shape 引入复杂 Python 分支但收益小于 2%。

## 候选方案 2：split-K autotune heuristic

### 问题

当前 split-K 选择逻辑较粗：

```text
if use_gqa_wmma_kernel && max_split_k > 1 && block_table.size(1) >= 16:
    num_kv_splits = min(max_split_k, block_table.size(1))
```

这会让 512 token 左右的 context 也可能使用 split=8。split-K 提高长 context 并行度，
但会增加：

- stage1 partial output 写入。
- stage2 reduce 读取。
- 额外 kernel launch。

短 context 下 split 过多可能不划算。

### 方案

新增 heuristic：

```text
logical_blocks = ceil(max_seq_len_or_block_table_width)

if logical_blocks < 16:
    split = 1
elif logical_blocks < 64:
    split = 2 or 4
elif logical_blocks < 128:
    split = 4 or 8
else:
    split = 8 or 16
```

由于 host 侧没有无同步的真实 max seq_len，可先使用 `block_table.size(1)` 作为上界。
后续 decode-only benchmark 中可以直接传真实 seq_len 或用 metadata 预计算。

新增 env：

```text
VLLM_BYTE_V2_SPLIT_K_HEURISTIC=static|table_width|disabled
```

### 实验矩阵

```text
split_k: 1, 2, 4, 8, 16
prompt_len: 512, 1024, 2048, 4096
decode_len: 16, 32, 64
batch: 1, 4
```

保留标准：

- 至少一个主要区间提升 >= 5%。
- 任何短 context 回退 <= 2%。
- 逻辑简单，不引入 D2H sync。

不保留标准：

- heuristic 只对单个点有效，整体不稳定。
- 需要 CPU sync 获取真实 seq_len。

## 候选方案 3：pure compressed decode kernel

### 问题

当前 `byte_v2_load_kv_bits()` 每次读取 K/V 元素都会检查：

```text
page status
fallback_pool pointer
fallback_block_ids
compressed/raw path
tile fallback byte
```

但 compressed-only 模式中，大多数 page 是 `kByteV2PageStatusCompressed`，且当前
block-level fallback 保证 compressed page 中 tile fallback 通常为 0。统一 fallback-aware
路径让热路径承担了不必要的分支和 metadata load。

### 方案

新增 pure compressed kernel：

```text
byte_v2_paged_decode_attention_gqa_wmma_compressed_kernel
byte_v2_paged_decode_attention_gqa_wmma_compressed_split_stage1_kernel
```

特点：

- 不传 `fallback_pool` 和 `fallback_block_ids`。
- 不调用 `byte_v2_load_kv_bits()`。
- 使用 `byte_v2_load_compressed_bits_fast()`：

```text
base = page[tile_start]
low = page[low_offset + elem]
packed = page[packed_offset + elem / 2]
decode bits
```

- 不读 tile fallback flag。
- 不处理 raw fallback status。

dispatch 策略：

第一版可以用用户显式开关：

```text
VLLM_BYTE_V2_DECODE_ASSUME_ALL_COMPRESSED=1
```

只用于受控 benchmark。后续再做安全自动选择：

- 如果 sparse fallback stats 显示该 batch 没有 fallback block，走 pure compressed。
- 或者在 cache update 维护 per-block compressed bitmap，decode 前可无同步地选择 batch kernel。

### 风险

如果实际 block 是 raw fallback，pure compressed kernel 会读错。因此第一版只能作为
实验开关，不默认启用。

### 保留标准

- fallback_ratio=0 的 decode-only benchmark 提升 >= 8%。
- e2e 在 fallback 很少的场景提升 >= 5%。
- 默认路径保持安全 fallback-aware kernel。

不保留标准：

- fallback_ratio=0 下提升不明显。
- 需要复杂 batch 预扫描才安全，且预扫描成本抵消收益。

## 候选方案 4：并行化 tile softmax 和 p_shared 填充

### 问题

当前 WMMA kernel 中，QK 后的 tile softmax 由 `tid == 0` 串行执行：

```text
for all 16x16 p_shared:
    zero
for q in q_per_kv:
    tile_max = max(scores[q, row])
    update denom/running_max
    for row:
        p_shared[q, row] = exp(...)
```

这对每个 page 都执行一次。`q_per_kv=4` 时数据量不大，但串行 exp 和 shared 写入会拖慢
每个 page 的固定开销。

### 方案 A：warp-level softmax

使用一个 warp 处理一个 `q` row，或者一个 warp 处理多个 q row：

```text
lane 0..15: valid row scores
warp reduce max
warp reduce sum exp
lane 0..15: write p_shared[q, row]
```

`q_per_kv <= 8`，最多 8 个 q row。可用一个 CTA 的前几个 warp 处理。

优点：

- 减少 `tid==0` 串行瓶颈。
- softmax 数据规模固定，适合手写 warp primitive。

风险：

- shared memory 和同步逻辑更复杂。
- expf 仍然较重，收益需要实测。

### 方案 B：跳过 p_shared zero 全矩阵

当前 `p_shared` 初始化 256 个元素，但实际只需要 `q_per_kv * valid_rows`。

优化：

- 只写有效 q rows。
- 对无效 rows，在 PV WMMA 前确保对应位置为 0。
- 可在 q_shared 初始化时或专门小循环清理 q_per_kv 之外区域。

风险：

- 如果未清零的 p_shared 被 WMMA PV 使用，会产生错误。
- 需要测试 q_per_kv=2/4/8 和 partial valid_rows。

### 保留标准

- decode-only kernel time >= -8%。
- e2e >= +5%。
- 数值误差与 baseline 一致或更小。

不保留标准：

- 代码复杂但提升小于 5%。
- 只在某个 `q_per_kv` 有效，但其他常见模型回退明显。

## 候选方案 5：Llama fast path specialization

### 问题

当前 kernel 支持通用：

```text
head_size <= 128
head_size_v <= 128
q_per_kv <= 8
block_table dtype int32/int64
seq_lens dtype int32/int64
```

通用性带来循环和边界判断。当前主要 benchmark 是 Llama-3-8B：

```text
head_size = 128
head_size_v = 128
q_per_kv = 4
num_kv_heads = 8
block_size = 16
```

### 方案

新增 specialized kernel：

```text
byte_v2_paged_decode_attention_llama_gqa4_h128_kernel
byte_v2_paged_decode_attention_llama_gqa4_h128_split_stage1_kernel
```

固定：

- `head_size = 128`
- `head_size_v = 128`
- `q_per_kv = 4`
- `block_size = 16`
- BF16 WMMA

优化点：

- 移除 `q_per_kv` 循环的动态边界。
- 固化 dim tile 数：K/V 各 8 个 16-dim tile。
- 固化 shared memory layout。
- 对 Q load、K load、V load、PV loop 完全 unroll。
- 可直接写 4 个 q head 的 output。

dispatch：

```text
if head_size == 128 and head_size_v == 128 and q_per_kv == 4:
    use specialized kernel
else:
    generic kernel
```

### 保留标准

- Llama-3-8B decode-only >= +10%。
- e2e >= +5%。
- generic path 不回退。

不保留标准：

- specialized kernel 代码过大但收益低于 5%。
- nvcc compile time 或 binary size 增长明显且收益不足。

## 候选方案 6：compressed payload vectorized load/unpack

### 问题

当前 `byte_v2_load_compressed_bits()` 每个元素读取：

```text
base
fallback
low byte
packed nibble
```

并逐元素计算 tile offsets。对于一个 16x16 tile，`base` 和 tile offsets 是相同的，
却被重复读取/计算 256 次。

### 方案

为 compressed tile 增加 bulk unpack helper：

```text
byte_v2_unpack_tile_to_wmma_shared(
    page,
    is_value,
    kv_head,
    dim_tile,
    valid_rows,
    dst_shared
)
```

优化方式：

- CTA 或 warp 先读取一次 `base`。
- 用 vectorized load 读取 low bytes，例如 `uint4`/`ulonglong2`。
- packed code 每 byte 包含两个元素，按连续线程解包。
- 写入 WMMA 需要的 shared memory layout。
- 对 K tile 和 V tile 分开 specialized helper。

第一版不要引入 `cp.async`，先只做 vectorized global load + 减少重复 offset。
如果有效，再做双缓冲。

### 预期收益

- 减少全局内存指令数量。
- 减少整数 offset 计算。
- 减少每元素重复读取 base/fallback。

### 保留标准

- decode-only fallback_ratio=0 提升 >= 8%。
- e2e >= +5%。

不保留标准：

- vectorized load 对齐处理复杂且收益不稳定。
- 不同 page layout 下引入未对齐访问问题。

## 候选方案 7：K/V load 顺序和双缓冲

### 问题

当前每个 page 内先加载 K 和 V 到 shared，然后做 QK 和 PV：

```text
load K
load V
WMMA QK
softmax
WMMA PV
```

这会在 QK 前就支付 V 解压成本，也会占用较多 shared memory。

### 方案 A：延迟 V 解压

改为：

```text
load K
WMMA QK
softmax
load V
WMMA PV
```

可能收益：

- QK 更早开始。
- 减少 K/V 同时驻留的 shared memory 压力。

风险：

- 没有 overlap，可能只是移动开销。
- 需要更多同步。

### 方案 B：page 双缓冲

对 page chunk 做双缓冲：

```text
buffer 0: 当前 page QK/PV
buffer 1: 下一 page load/unpack
```

理论上可以隐藏部分 unpack/global load latency。

风险：

- 代码复杂度高。
- shared memory 翻倍，occupancy 可能下降。
- 当前每 CTA 已经使用多个 shared arrays，可能不适合 Ampere。

### 实验顺序

先做方案 A。只有方案 A 有收益或 Nsight 显示 load/unpack latency 明显时，再考虑方案 B。

保留标准：

- 方案 A：e2e >= +3%。
- 方案 B：decode-only >= +10%，且 occupancy 不明显下降。

## 候选方案 8：PV 路径替代实验

### 问题

当前 PV 使用 BF16 WMMA：

```text
p_shared[16x16] * v_shared[16x128]
```

但 `q_per_kv=4` 时，只有前 4 个 q row 有效，WMMA 仍计算 16 行，理论上浪费 75%
row compute。虽然 tensor core 很快，但 `p_shared` 构造和 `pv_shared` store/read
也有开销。

### 方案 A：保留 WMMA PV，优化 p_shared

这是候选方案 4 的延伸，优先级更高。

### 方案 B：CUDA core scalar PV

每个 thread 负责若干 output dim，对 `q_per_kv` 做：

```text
for row in valid_rows:
    acc[q, dim] += weight[q, row] * V[row, dim]
```

优点：

- 只计算有效 `q_per_kv` 行。
- 不需要构造完整 16x16 `p_shared`。
- 不需要 `pv_shared` 中间矩阵。

风险：

- CUDA core FMA 可能远慢于 Tensor Core。
- 对 `head_size_v=128` 和多 q row，thread/register 压力可能较高。

实验方式：

- 只做 Llama `q_per_kv=4, head_size_v=128` specialized variant。
- 和 WMMA PV 在 decode-only benchmark 中比较。

保留标准：

- decode-only >= +8%。
- e2e >= +5%。

不保留标准：

- 只减少理论计算但实际更慢。

## 候选方案 9：persistent split workspace

### 问题

当前 native decode op 中：

```cpp
auto output = torch::stable::empty(...)
auto split_partial_output = torch::stable::empty(...)
```

候选方案 1 会处理 output，但 split partial workspace 仍在 op 内部创建。CUDA graph
capture 可以把 allocator 行为捕获进 graph pool，但 eager 和 capture 初始化仍有开销。

### 方案

在 ByteV2 backend 或 model runner 中为每层预分配：

```text
partial_output_workspace:
  [max_num_seqs, num_heads, max_split_k, head_size_v + 1] float32
```

decode op 改为接收 workspace：

```text
byte_v2_paged_decode_attention_out(..., output, partial_workspace)
```

优点：

- 去掉 op 内动态 allocation。
- cudagraph replay 更稳定。
- 为后续 persistent reduction 做准备。

风险：

- workspace 占用额外 HBM。
- shape 需要匹配 max capture size、num heads、head size。
- backend 生命周期管理复杂。

保留标准：

- e2e >= +2%，或者是 direct-output/cudagraph 稳定性的必要前置。
- workspace 内存开销可解释，并计入文档。

不保留标准：

- 只增加内存，没有性能收益。

## 候选方案 10：cache update contiguous prompt fast path

### 问题

当前 compressed-only update 对所有情况统一：

```text
init all block flags
mark touched tokens
compress touched blocks
```

对 prefill prompt，slot_mapping 往往是连续或接近连续的 16-token blocks。此时无需先
按 token mark，再按 block 查 `block_token_indices`。

### 方案

新增 contiguous fast path：

```text
if slot_mapping is contiguous block-aligned:
    one CTA per prompt block
    token_base = block_id * 16
    directly load key/value[token_base + row]
    compress block
else:
    existing generic path
```

难点是如何无 D2H sync 判断 contiguous：

- 第一版可以由 Python/metadata 在 prefill raw-only path 中传入标志。
- 或者 kernel 内检查每个 block 的 slot pattern。
- decode append 仍走 generic path。

优化点：

- 去掉 mark kernel。
- 减少 `block_token_indices` workspace。
- raw_staging 可以按 touched blocks 分配，而不是 num_tokens。

保留标准：

- prompt=512/1024/2048 e2e prefill+decode 总时间提升 >= 5%。
- decode-only 不回退。
- 不破坏 block reuse 和 partial block append。

不保留标准：

- contiguous 判断复杂且容易错。
- 只在 synthetic prompt 有效，真实 scheduler 场景无收益。

## 候选方案 11：cache update 压缩内部并行化

### 问题

`byte_v2_raw_block_is_compressible()` 和 `byte_v2_store_compressed_block()`
内部对 tile/window 的处理仍有较多串行循环。当前 one CTA per block，但 CTA 内不是所有
步骤都充分并行。

### 方案

将 compression 拆成 tile-parallel：

```text
grid:
  blockIdx.x = physical block
  blockIdx.y = tile id within K/V block

per tile:
  compute exponent histogram/window
  decide compressible
  write compressed tile or mark fallback
```

两种实现：

1. 单 CTA per block 内多 warp 处理多个 tile。
2. one CTA per tile，另一个 kernel 汇总 block status。

风险：

- 多 kernel 可能增加 launch overhead。
- fallback pool allocation 需要 block-level coordination。
- store order 和 page status 需要谨慎。

保留标准：

- cache update kernel time >= -15%。
- prompt e2e >= +5%。
- fallback 行为完全一致。

不保留标准：

- kernel time 下降但 e2e 无改善，且代码复杂。

## 候选方案 12：raw fallback 访问路径分离

### 问题

fallback-aware decode 在每个元素上检查是否 raw fallback。当前 block-level fallback
比例不为 0，且 fallback block 会让该 page 读 raw BF16 pool。统一 kernel 同时处理
compressed/raw，热路径分支多。

### 方案

按 request/head/page chunk 分类：

- pure compressed chunks：走 pure compressed kernel。
- fallback chunks：走 fallback-aware kernel。

第一版不做复杂调度，只增加一个 `fallback_block_ids` bitmap scan kernel，生成：

```text
has_fallback_per_request: [B]
```

如果 request 没有 fallback block，走 pure compressed；否则走 fallback-aware。

风险：

- scan kernel 本身有开销。
- 如果大多数 request 都有少量 fallback，分类收益不明显。
- cudagraph 下动态选择 kernel 需要固定图或 capture 多 variant。

保留标准：

- fallback_ratio <= 3% 场景 e2e >= +5%。
- fallback_ratio 高场景不回退超过 2%。

不保留标准：

- scan overhead 抵消收益。
- 破坏 cudagraph 简洁性。

## 推荐实验顺序

建议按低风险到高风险顺序推进。

### Step 0：decode-only benchmark

必须先做。没有这个 benchmark，后续 kernel 优化无法准确判断。

保留：无条件保留，因为是实验基础设施。

### Step 1：direct output

低风险，主要减少 allocation/copy。

保留门槛：e2e >= +2%。

### Step 2：split-K autotune

低风险，只改 dispatch heuristic。

保留门槛：主要 benchmark 不回退，至少一个常用区间 >= +5%。

### Step 3：pure compressed decode fast path

中风险，但很可能暴露统一 fallback-aware path 的真实成本。

保留门槛：fallback_ratio=0 decode-only >= +8%，e2e >= +5%。

### Step 4：parallel tile softmax

中风险，针对当前 `tid==0` 串行瓶颈。

保留门槛：decode-only >= +8%，e2e >= +5%。

### Step 5：Llama fast path specialization

中高风险，代码量会增加，但当前主要模型就是 Llama-3-8B。

保留门槛：Llama e2e >= +5%，decode-only >= +10%。

### Step 6：vectorized unpack

中高风险，需要细致处理 alignment 和 shared layout。

保留门槛：fallback_ratio=0 decode-only >= +8%，e2e >= +5%。

### Step 7：cache update contiguous fast path

中风险，主要改善 prompt/prefill。

保留门槛：prompt e2e >= +5%。

### Step 8：cache update tile-parallel compression

高风险，放在 decode 热路径优化之后。

保留门槛：cache update kernel time >= -15%，e2e >= +5%。

## 标准验证命令

### 单测

每个实验至少跑：

```bash
CUDA_VISIBLE_DEVICES=2 \
.venv/bin/python -m pytest -q \
  tests/v1/attention/test_byte_v2_ops.py \
  tests/v1/attention/test_byte_v2_decode.py \
  tests/v1/attention/test_byte_v2_metadata.py \
  tests/config/test_cache_config.py
```

### e2e 小实验

```bash
CUDA_VISIBLE_DEVICES=2 \
VLLM_BYTE_V2_SPARSE_FALLBACK_POOL_RATIO=0.03 \
VLLM_BYTE_V2_DECODE_SPLIT_K=8 \
.venv/bin/python benchmarks/benchmark_byte_v2_decode_e2e.py \
  --model /mnt/sda1/yxz/byte_v2/Meta-Llama-3-8B-Instruct \
  --modes raw,byte_v2_compressed_only \
  --prompt-len 512 \
  --decode-lens 16,32 \
  --batch-size 1 \
  --num-runs 3 \
  --warmup-decode-len 4 \
  --dtype bfloat16 \
  --block-size 16 \
  --gpu-memory-utilization 0.55 \
  --output-json benchmarks/profiles/<experiment_name>_p512_d16_d32.json
```

### 长 context 实验

```bash
CUDA_VISIBLE_DEVICES=2 \
VLLM_BYTE_V2_SPARSE_FALLBACK_POOL_RATIO=0.03 \
VLLM_BYTE_V2_DECODE_SPLIT_K=8 \
.venv/bin/python benchmarks/benchmark_byte_v2_decode_e2e.py \
  --model /mnt/sda1/yxz/byte_v2/Meta-Llama-3-8B-Instruct \
  --modes raw,byte_v2_compressed_only \
  --prompt-len 2048 \
  --decode-lens 32,64 \
  --batch-size 1 \
  --num-runs 3 \
  --warmup-decode-len 4 \
  --dtype bfloat16 \
  --block-size 16 \
  --gpu-memory-utilization 0.55 \
  --output-json benchmarks/profiles/<experiment_name>_p2048_d32_d64.json
```

如果 3% fallback pool 在长 context 下耗尽，可临时用 5% 或 10% 只为测试 kernel 性能，
但结果必须单独标注，不能和 3% 默认配置混在一起比较。

### Nsight profile

对通过 e2e 初筛的实验再跑 Nsight：

```bash
CUDA_VISIBLE_DEVICES=2 \
VLLM_BYTE_V2_SPARSE_FALLBACK_POOL_RATIO=0.03 \
VLLM_BYTE_V2_DECODE_SPLIT_K=8 \
nsys profile \
  --trace=cuda,nvtx \
  --force-overwrite=true \
  --output benchmarks/profiles/<experiment_name> \
  .venv/bin/python benchmarks/benchmark_byte_v2_decode_e2e.py \
    --model /mnt/sda1/yxz/byte_v2/Meta-Llama-3-8B-Instruct \
    --modes byte_v2_compressed_only \
    --prompt-len 1024 \
    --decode-lens 64 \
    --batch-size 1 \
    --num-runs 1 \
    --warmup-decode-len 4 \
    --dtype bfloat16 \
    --block-size 16 \
    --gpu-memory-utilization 0.55 \
    --enforce-eager
```

## 结果记录模板

每个实验完成后，在对应 benchmark summary 或实验 issue 中记录：

```text
实验名:
代码开关:
commit/patch:
GPU:
模型:
配置:

correctness:
  pytest:
  max_abs_diff:
  max_rel_diff:

decode-only:
  baseline median_us:
  experiment median_us:
  speedup:

e2e:
  raw output tok/s:
  baseline ByteV2 output tok/s:
  experiment ByteV2 output tok/s:
  ByteV2/raw:
  experiment/baseline:

capacity:
  raw KV tokens:
  ByteV2 KV tokens:
  sparse fallback max_next_slot/capacity:
  pool exhausted:

profile:
  main kernel time:
  split reduce time:
  cache update time:
  dram throughput:
  occupancy:

结论:
  keep / revert
  原因:
```

## 最重要的判断

当前 ByteV2 只有 raw 的约 27%。如果只优化当前 kernel，短期合理目标是：

```text
27% -> 40% -> 50% -> 70%
```

不要期望一个小 patch 直接超过 raw。每个实验都应回答一个具体问题：

- direct output 能否减少 runtime/copy 开销？
- split-K 是否选错了？
- fallback-aware 分支成本有多高？
- `tid==0` softmax 是否是 tile 固定开销瓶颈？
- Llama specialization 能否明显减少通用分支和循环？
- vectorized unpack 是否真的减少 decode kernel time？
- cache update 是否仍拖慢 prompt e2e？

如果某个实验不能清楚回答问题，或者回答是“没有稳定收益”，就删除该实验代码。

## 后续修改建议

建议下一轮从 Step 0 和 Step 1 开始：

1. 新增 decode-only benchmark。
2. 实现 `byte_v2_paged_decode_attention_out()`，直接写 vLLM output buffer。
3. 跑 baseline vs out-op：pytest、p512/d16,d32 e2e、必要时 Nsight。
4. 如果收益低于 2%，删除 out-op 改动，仅保留 benchmark。

这样可以建立后续所有 kernel 实验的执行纪律：每个优化都必须用数据证明自己值得保留。

## 实验记录

### Step 0: decode-only benchmark

状态：保留。

新增脚本：

```text
benchmarks/kernels/benchmark_byte_v2_decode_kernel.py
```

验证：

```text
.venv/bin/python -m ruff check benchmarks/kernels/benchmark_byte_v2_decode_kernel.py
CUDA_VISIBLE_DEVICES=2 .venv/bin/python benchmarks/kernels/benchmark_byte_v2_decode_kernel.py \
  --batch-size 1 --seq-len 64 --num-heads 8 --num-kv-heads 2 \
  --head-size 32 --head-size-v 32 --split-k 1,4 \
  --num-runs 10 --warmup-runs 3
```

smoke 结果：

| seq_len | split_k | median_us | max_abs_diff |
|---:|---:|---:|---:|
| 64 | 1 | 970.24 | 0.0 |
| 64 | 4 | 970.75 | 0.0 |

Llama3-8B 形状基线：

```text
CUDA_VISIBLE_DEVICES=2 .venv/bin/python benchmarks/kernels/benchmark_byte_v2_decode_kernel.py \
  --batch-size 1 --seq-len 512,1024 --num-heads 32 --num-kv-heads 8 \
  --head-size 128 --head-size-v 128 --split-k 1,2,4,8,16 \
  --num-runs 50 --warmup-runs 10 \
  --output-json benchmarks/profiles/bytev2_decode_kernel_baseline_p512_p1024_b1_split_sweep.json
```

| seq_len | split_k=1 | split_k=2 | split_k=4 | split_k=8 | split_k=16 |
|---:|---:|---:|---:|---:|---:|
| 512 median_us | 1470.98 | 1226.24 | 1100.80 | 1037.82 | 1013.76 |
| 1024 median_us | 1980.93 | 1481.73 | 1228.80 | 1105.92 | 1057.79 |

3% sparse fallback：

```text
benchmarks/profiles/bytev2_decode_kernel_baseline_p512_p1024_b1_fallback003.json
```

| seq_len | split_k | median_us | fallback blocks | max_abs_diff |
|---:|---:|---:|---:|---:|
| 512 | 8 | 1034.24 | 1/5 | 0.0 |
| 512 | 16 | 1008.13 | 1/5 | 0.0 |
| 1024 | 8 | 1105.92 | 2/6 | 0.0 |
| 1024 | 16 | 1060.86 | 2/6 | 0.0 |

结论：

- decode-only benchmark 能稳定复现当前 native decode latency。
- 在该 microbenchmark 上，`split_k=16` 比当前默认 `split_k=8` 略快。
- 3% sparse fallback 对 decode-only median latency 影响很小；当前 benchmark 能覆盖 fallback pool 路径。

### Step 1: direct-output op

状态：不保留，实验代码已删除。

实验内容：

```text
byte_v2_paged_decode_attention_out(output, ...)
```

目标是让 decode kernel 直接写入 vLLM attention output buffer，避免旧路径中的
`decode_out` 分配和 Python `copy_`。

correctness：

```text
CUDA_VISIBLE_DEVICES=2 .venv/bin/python -m pytest -q \
  tests/v1/attention/test_byte_v2_decode.py::test_byte_v2_paged_decode_attention_out_op_matches_returning_op_cuda
```

结果：通过。

decode-only 对照：

| seq_len | split_k | returning-op median_us | direct-output median_us | direct / returning |
|---:|---:|---:|---:|---:|
| 512 | 8 | 1024.00 | 1026.05 | 100.2% |
| 512 | 16 | 1000.45 | 998.40 | 99.8% |
| 1024 | 8 | 1097.73 | 1099.78 | 100.2% |
| 1024 | 16 | 1051.65 | 1054.21 | 100.2% |

e2e 对照：

```text
prompt_len=512, decode_lens=16,32, batch=1, split_k=8, cudagraph on
```

| decode_len | direct off tok/s | direct on tok/s | direct / off |
|---:|---:|---:|---:|
| 16 | 9.32 | 9.38 | 100.7% |
| 32 | 9.74 | 9.72 | 99.9% |

结论：

- direct-output 对 decode-only 没有稳定收益。
- e2e 结果低于本文档设定的“小改动至少 +2%”保留门槛。
- 说明当前瓶颈主要不在 Python copy 或 output tensor 分配，而在 decode kernel 内部和整体模型路径。
- 因此删除 `byte_v2_paged_decode_attention_out()`、Python wrapper、env 开关和专属测试，只保留 benchmark 结果作为负例记录。

回归验证：

```text
CUDA_VISIBLE_DEVICES=2 .venv/bin/python - <<'PY'
import torch
import vllm._custom_ops  # noqa: F401
print('byte_v2_paged_decode_attention_out' in dir(torch.ops._C_cache_ops))
PY
# False

CUDA_VISIBLE_DEVICES=2 .venv/bin/python -m pytest -q \
  tests/v1/attention/test_byte_v2_decode.py::test_byte_v2_paged_decode_attention_op_gqa_wmma_split_k_cuda \
  tests/v1/attention/test_byte_v2_decode.py::test_byte_v2_paged_decode_attention_op_gqa_wmma_q_per_kv_4_cuda
# 2 passed
```

### Step 2: split-K 默认值初步实验

状态：保留 opt-in auto heuristic；不改默认值。

背景：

Step 0 decode-only benchmark 显示 `split_k=16` 在 `seq_len=512/1024` 上比当前默认
`split_k=8` 略快：

```text
seq_len=512:  1037.82 us -> 1013.76 us, +2.4%
seq_len=1024: 1105.92 us -> 1057.79 us, +4.6%
```

但 e2e 对照没有形成足够收益：

```text
prompt_len=512, decode_lens=16,32, batch=1, cudagraph on
split_k=8  结果: benchmarks/profiles/bytev2_e2e_direct_output_off_p512_d16_d32.json
split_k=16 结果: benchmarks/profiles/bytev2_e2e_split16_p512_d16_d32.json
```

| decode_len | split_k=8 tok/s | split_k=16 tok/s | split16 / split8 |
|---:|---:|---:|---:|
| 16 | 9.32 | 9.33 | 100.1% |
| 32 | 9.74 | 9.79 | 100.6% |

结论：

- 对 `prompt_len=512` 的 e2e，简单把默认 split-K 从 8 改成 16 不值得保留。
- decode-only 的改善说明 split-K 选择仍可能影响长 context kernel latency。
- 后续如果继续 Step 2，应做按 `block_table.size(1)` 和 batch/head 并行度选择的 heuristic，
  并重点测试 `seq_len >= 2048/4096`，而不是直接改全局默认值。

### Step 2b: split-K auto heuristic

状态：保留。

实现：

```text
VLLM_BYTE_V2_DECODE_SPLIT_K=0
```

显式设置为 0 时启用 page-count heuristic；默认值仍是 8，手工设置正整数仍按原有
方式强制最大 split 数。

当前规则：

```text
num_pages >= 128 -> max_split_k=32
num_pages >=  64 -> max_split_k=16
num_pages >=  16 -> max_split_k=8
otherwise       -> max_split_k=1

active_head_groups = num_decode_tokens * num_kv_heads
active_head_groups >= 64 -> cap to 8
active_head_groups >= 32 -> cap to 16
```

这样做的目的：

- 单 batch / 长 context 时提高 page-parallelism。
- batch 或 active KV-head group 已经足够多时避免过度拆分导致 partial workspace 和 reduce 开销过大。
- 保持默认 `split_k=8` 不变，降低短 context 回退风险。

代码位置：

```text
csrc/libtorch_stable/cache_kernels.cu
vllm/envs.py
tests/v1/attention/test_byte_v2_decode.py
```

correctness：

```text
CUDA_VISIBLE_DEVICES=2 .venv/bin/python -m pytest -q \
  tests/v1/attention/test_byte_v2_decode.py::test_byte_v2_paged_decode_attention_op_gqa_wmma_auto_split_k_cuda
# 1 passed
```

decode-only 对照：

```text
benchmarks/profiles/bytev2_decode_kernel_auto_split_p512_p2048_p4096_b1.json
```

| seq_len | split_k=8 median_us | auto median_us | split_k=32 median_us | auto / split8 |
|---:|---:|---:|---:|---:|
| 512 | 1042.94 | 1034.24 | 1018.37 | 99.2% |
| 2048 | 1391.62 | 1166.85 | 1169.41 | 83.9% |
| 4096 | 1726.46 | 1272.32 | 1267.71 | 73.7% |

e2e 对照：

修复前，`prompt_len=2048` 使用 3% fallback pool 时，split=8 在 prefill cache
update 阶段失败：

```text
byte_v2_reshape_and_cache: Byte-v2 native cache update received an invalid slot,
page state, or finalized block update
```

同一配置用 10% fallback pool 可跑通，说明 p2048 的真实 Llama KV 分布下纯 3%
pool 偏紧。

10% pool 下的 split-K 对照：

```text
split_k=8:
  benchmarks/profiles/bytev2_e2e_split8_p2048_d16_d32_pool010.json
auto:
  benchmarks/profiles/bytev2_e2e_autosplit_p2048_d16_d32_pool010.json
```

| prompt_len | decode_len | split_k=8 tok/s | auto tok/s | auto / split8 |
|---:|---:|---:|---:|---:|
| 2048 | 16 | 7.78 | 8.04 | 103.4% |
| 2048 | 32 | 8.46 | 8.87 | 104.8% |

结论：

- auto split-K 在 decode-only 长 context 下收益明显。
- p2048 e2e 在 10% pool 下有 3-5% output tok/s 提升，达到保留门槛。
- 当前仍不把默认值改为 auto；需要用户或 benchmark 显式设置
  `VLLM_BYTE_V2_DECODE_SPLIT_K=0`。
- p2048 + 3% pool 失败暴露出另一个问题：长 prompt 下 fallback pool 需要更稳健的
  sizing 或更明确的 pool-exhaustion diagnostics。

已完成的修复：

- 新增 `VLLM_BYTE_V2_SPARSE_FALLBACK_POOL_MIN_BLOCKS`，默认 `512`。effective
  pool slots 为
  `min(num_blocks, max(ceil(num_blocks * ratio), min_blocks))`。
- `ByteV2FullAttentionSpec`、runtime backend 私有 pool、KV cache allocator 统一使用
  同一容量公式，避免 allocator 仍按纯 ratio 预留。
- native cache update 从单一 `error=1` 改成具体错误码，pool 耗尽时会报告
  `sparse fallback pool exhausted`，并打印 `fallback_pool_used` 和
  `fallback_pool_capacity`。

修复后验证：

```text
CUDA_VISIBLE_DEVICES=2 \
VLLM_BYTE_V2_USE_NATIVE_KERNELS=1 \
VLLM_BYTE_V2_SPARSE_FALLBACK_POOL_RATIO=0.03 \
VLLM_BYTE_V2_SPARSE_FALLBACK_POOL_MIN_BLOCKS=512 \
VLLM_BYTE_V2_DECODE_SPLIT_K=0 \
.venv/bin/python benchmarks/benchmark_byte_v2_decode_e2e.py \
  --modes byte_v2_compressed_only \
  --prompt-len 2048 \
  --decode-lens 16,32 \
  --batch-size 1 \
  --num-runs 1 \
  --warmup-decode-len 8 \
  --output-json benchmarks/profiles/bytev2_e2e_autosplit_p2048_d16_d32_pool003_min512.json
```

结果：

| prompt_len | decode_len | output tok/s | max pool used/capacity | exhausted |
|---:|---:|---:|---:|:---:|
| 2048 | 16 | 8.03 | 258 / 512 | no |
| 2048 | 32 | 8.88 | 388 / 512 | no |

本轮 `GPU KV cache size` 为 200,816 tokens；相比 10% pool 的约 80k tokens，固定
512 floor 保留了更高容量，同时 p2048 不再耗尽。

### Step 3: decode append cache fast path

状态：保留。

实现：

- 新增 `byte_v2_decode_append_cache_kernel`。
- compressed-only + sparse fallback pool + `num_tokens == 1` 时走 fast path。
- partial decode block 保持 raw fallback，只有 row 15 才尝试 full-block compress。

correctness：

```text
CUDA_VISIBLE_DEVICES=2 .venv/bin/python -m pytest \
  tests/v1/attention/test_byte_v2_ops.py \
  tests/v1/attention/test_byte_v2_decode.py -q
# 25 passed
```

cache-update microbenchmark：

```text
benchmarks/profiles/bytev2_cache_update_after_decode_append_fastpath_p2048_d32.json
```

| case | before median CUDA | after median CUDA | after per-step |
|---|---:|---:|---:|
| prefill p2048 | 6359.04 us | 6333.44 us | n/a |
| decode append d32 | 129637.37 us | 10602.50 us | 331.33 us |

e2e：

```text
benchmarks/profiles/bytev2_e2e_after_decode_append_fastpath_p2048_d16_d32.json
```

| prompt_len | decode_len | output tok/s | exhausted |
|---:|---:|---:|:---:|
| 2048 | 16 | 16.86 | no |
| 2048 | 32 | 21.77 | no |

结论：

- decode append cache update 从约 4.05 ms/step 降到约 0.33 ms/step。
- Nsight 显示 cache compress 总时间从约 3088 ms 降到约 443 ms，decode append
  kernel 本身约 13.7 ms / 1344 calls。
- 仍未达到 raw `reshape_and_cache_flash_kernel` 的 us 级开销，后续还需要减少
  host validation、allocator 压力和 full-block compress 串行逻辑。

### Step 4: prefill block-direct encode v1

状态：保留第一版；未达到最终阶段目标。

实现：

- 新增 `byte_v2_validate_prefill_direct_blocks_kernel`。
- 新增 `byte_v2_prefill_direct_encode_blocks_kernel`。
- `byte_v2_reshape_and_cache()` 在 compressed-only + sparse fallback pool +
  `num_tokens` 为 16 的倍数 + 非 CUDA graph capture 时尝试 direct path。
- validation 要求每个 16-token group 映射到同一个 physical block 的 offsets 0..15，
  并用 per-block claim 检测重复 physical block。
- validation 失败不写 KV cache，直接回退旧 generic path；direct encode 中遇到
  不可压缩 full block 时写入 sparse fallback pool。

correctness：

```text
CUDA_VISIBLE_DEVICES=2 .venv/bin/python -m pytest \
  tests/v1/attention/test_byte_v2_ops.py::test_byte_v2_native_prefill_direct_encode_full_blocks_cuda -q
# 1 passed

CUDA_VISIBLE_DEVICES=2 .venv/bin/python -m pytest \
  tests/v1/attention/test_byte_v2_ops.py \
  tests/v1/attention/test_byte_v2_decode.py -q
# 26 passed
```

cache-update microbenchmark：

```text
benchmarks/profiles/bytev2_cache_update_after_prefill_direct_p2048_d32_repeat.json
```

| case | previous median CUDA | after median CUDA | delta |
|---|---:|---:|---:|
| prefill p2048 | 6333.44 us | 5485.57 us | -13.4% |
| decode append d32 | 10602.50 us | 10638.34 us | +0.3% |
| decode append per step | 331.33 us | 332.45 us | +0.3% |

e2e：

```text
benchmarks/profiles/bytev2_e2e_after_prefill_direct_p2048_d16_d32.json
```

| prompt_len | decode_len | before tok/s | after tok/s | delta | max pool used/capacity | exhausted |
|---:|---:|---:|---:|---:|---:|:---:|
| 2048 | 16 | 16.86 | 17.36 | +3.0% | 258 / 512 | no |
| 2048 | 32 | 21.77 | 22.23 | +2.1% | 388 / 512 | no |

结论：

- v1 direct path 去掉了 full-cache init/mark/compress 的一部分开销，isolated
  prefill cache update 有稳定收益。
- 收益远低于文档中“prefill 5x”的最终目标，原因是 full-block compression 内部仍然
  主要由 `threadIdx.x == 0` 串行完成，并且 host validation 仍有一次 D2H sync。
- 后续如果继续优化 cache update，应优先做 tile-parallel compression 和 capture-safe
  metadata validation；否则 prefill block-direct encode 的上限有限。

### Step 5: prefill tile-parallel encode v2

状态：保留。

实现：

- `byte_v2_prefill_direct_encode_blocks_kernel` 从 v1 的 thread0 串行
  `byte_v2_store_compressed_block()` 改为 CTA 内 tile-parallel。
- one CTA per 16-token physical block，8 warps/CTA；每个 warp 处理一个或多个
  16x16 K/V tile。
- 第一阶段每个 warp 计算 tile exponent min/max，用 `max - min <= 15` 判断
  是否能被当前 ByteV2 12-bit window 无损表示。
- 第二阶段各 warp 并行写 compressed tile 的 base、low bytes 和 packed code。
- 不可压缩 block 仍写 sparse fallback pool，但 raw fallback copy 也改成 CTA 内并行。
- direct prefill fast path 不再分配 `direct_raw_staging`。

correctness：

```text
CUDA_VISIBLE_DEVICES=2 .venv/bin/python -m pytest \
  tests/v1/attention/test_byte_v2_ops.py::test_byte_v2_native_prefill_direct_encode_full_blocks_cuda \
  tests/v1/attention/test_byte_v2_ops.py::test_byte_v2_native_sparse_fallback_pool_cuda \
  tests/v1/attention/test_byte_v2_ops.py::test_byte_v2_native_sparse_fallback_pool_exhaustion_cuda -q
# 3 passed

CUDA_VISIBLE_DEVICES=2 .venv/bin/python -m pytest \
  tests/v1/attention/test_byte_v2_ops.py \
  tests/v1/attention/test_byte_v2_decode.py -q
# 26 passed
```

cache-update microbenchmark：

```text
benchmarks/profiles/bytev2_cache_update_after_prefill_tile_parallel_p2048_d32.json
benchmarks/profiles/bytev2_cache_update_after_prefill_tile_parallel_sweep.json
```

| prompt_len | v1 median CUDA | v2 median CUDA | delta |
|---:|---:|---:|---:|
| 512 | n/a | 212.99 us | n/a |
| 1024 | n/a | 209.92 us | n/a |
| 2048 | 5485.57 us | 251.90 us | -95.4% |
| 4096 | n/a | 283.65 us | n/a |

同 shape raw `reshape_and_cache_flash`：

```text
benchmarks/profiles/raw_reshape_and_cache_flash_prefill_p2048.json
```

| case | median CUDA |
|---|---:|
| raw p2048 cache write | 46.08 us |
| ByteV2 v1 p2048 prefill encode | 5485.57 us |
| ByteV2 v2 p2048 prefill encode | 251.90 us |

E2E p2048 prefix-off `decode_len=1`：

```text
benchmarks/profiles/raw_vs_bytev2_prefill_approx_p2048_d1_prefixoff_current.json
benchmarks/profiles/bytev2_e2e_after_prefill_tile_parallel_p2048_d1_prefixoff.json
```

| mode | elapsed | total tok/s |
|---|---:|---:|
| raw prefix-off | 0.290 s | 7075 tok/s |
| ByteV2 v1 | 0.449 s | 4559 tok/s |
| ByteV2 v2 | 0.334 s | 6132 tok/s |

E2E p2048 d16/d32：

```text
benchmarks/profiles/bytev2_e2e_after_prefill_direct_p2048_d16_d32.json
benchmarks/profiles/bytev2_e2e_after_prefill_tile_parallel_p2048_d16_d32.json
```

| decode_len | ByteV2 v1 tok/s | ByteV2 v2 tok/s | delta | max pool used/capacity | exhausted |
|---:|---:|---:|---:|---:|:---:|
| 16 | 17.36 | 19.35 | +11.5% | 258 / 512 | no |
| 32 | 22.23 | 24.40 | +9.7% | 388 / 512 | no |

结论：

- tile-parallel encode 基本解决了首段 prefill cache compression 的最大串行瓶颈。
- 单层 p2048 cache update 从 raw 的约 119x 慢缩小到约 5.5x 慢。
- p2048 d1 prefix-off E2E 中，ByteV2 从 raw 的 1.55x 慢缩小到 1.15x 慢。
- 下一步应继续去掉 production path 的 host validation/D2H sync，或把 prefill
  compression 放到异步 stream 与后续层计算 overlap。

### Step 6: prefill validation sync 实验

状态：保留安全版；不保留完全 no-host-sync 和 async overlap。

目标：

- 减少 prefill direct path 中 validation kernel 后的 D2H sync。
- 保持 fail-closed：fallback pool 耗尽、invalid slot、duplicate slot 等错误仍必须
  返回 host 并抛错，不能静默损坏 KV cache。

实现：

- 新增实验开关 `VLLM_BYTE_V2_PREFILL_DIRECT_SKIP_VALIDATION_SYNC=1`。
- 开关打开时，`byte_v2_reshape_and_cache()` 不再把
  `validation_error` 拷回 host 之后再决定是否进入 direct encode。
- validation kernel 和 direct encode kernel 仍在同一 CUDA stream 上串行执行；
  validation 失败时，`group_block_ids[group] = -1`，direct encode 会把错误写入
  `direct_result[0]`。
- `direct_result` 和 `fallback_next_slot` 仍然 D2H 并 `cudaStreamSynchronize()`；
  因此 fallback pool exhaustion 等 kernel 错误仍然 fail-closed。
- 完全 no-host-sync 版本曾经试过，但因为会跳过 `direct_result` host 检查，在
  sparse fallback pool 接近耗尽时可能静默吞掉错误，已移除。

correctness：

```text
CUDA_VISIBLE_DEVICES=2 .venv/bin/python -m pytest \
  tests/v1/attention/test_byte_v2_ops.py::test_byte_v2_native_prefill_direct_encode_full_blocks_cuda \
  tests/v1/attention/test_byte_v2_ops.py::test_byte_v2_native_prefill_direct_skip_validation_sync_cuda -q
# 2 passed

CUDA_VISIBLE_DEVICES=2 .venv/bin/python -m pytest \
  tests/v1/attention/test_byte_v2_backend.py \
  tests/v1/attention/test_byte_v2_ops.py \
  tests/v1/attention/test_byte_v2_decode.py -q
# 35 passed
```

cache-update microbenchmark：

```text
benchmarks/profiles/bytev2_cache_update_prefill_skip_validation_sync_sweep.json
```

| prompt_len | tile-parallel v2 median CUDA | skip validation sync median CUDA | delta |
|---:|---:|---:|---:|
| 512 | 212.99 us | 141.31 us | -33.7% |
| 1024 | 209.92 us | 145.41 us | -30.7% |
| 2048 | 251.90 us | 166.91 us | -33.7% |
| 4096 | 283.65 us | 212.99 us | -24.9% |

decode append 基本不变：

| case | median CUDA |
|---|---:|
| decode append d32 total | 10631.17 us |
| decode append per step | 332.22 us |

E2E p2048 prefix-off：

```text
benchmarks/profiles/bytev2_e2e_prefill_skip_validation_sync_p2048_d1_d16_prefixoff.json
benchmarks/profiles/bytev2_e2e_prefill_skip_validation_sync_p2048_d32_prefixoff.json
```

| decode_len | elapsed | output tok/s | pool exhausted |
|---:|---:|---:|:---:|
| 1 | 0.298 s | 3.35 | no |
| 16 | 0.776 s | 20.61 | no |
| 32 | 1.291 s | 24.79 | no |

与 Step 5 的 p2048/d1 prefix-off 对比：

| mode | elapsed | total tok/s |
|---|---:|---:|
| raw prefix-off | 0.290 s | 7075 tok/s |
| ByteV2 tile-parallel v2 | 0.334 s | 6132 tok/s |
| ByteV2 skip validation sync | 0.298 s | 6874 tok/s |

完全 no-host-sync 的拒绝记录：

```text
benchmarks/profiles/bytev2_cache_update_prefill_no_host_sync_sweep.json
benchmarks/profiles/bytev2_e2e_prefill_no_host_sync_p2048_d1_d16_d32_prefixoff.json
```

| case | result |
|---|---:|
| p2048 cache-update median CUDA | 114.69 us |
| p2048/d1 elapsed | 0.294 s |
| p2048/d16 output tok/s | 20.80 |
| p2048/d32 output tok/s | 24.96 |

该版本虽然更快，但 `direct_result` 不回 host，遇到 sparse fallback pool exhaustion
时不会立刻抛错；同一 engine 连续 warmup + d1 + d16 + d32 的实验中已经观察到
`max_next_slot=515 / capacity=512`。因此该版本不满足 fail-closed 要求，不保留。

async overlap 的拒绝记录：

```text
benchmarks/profiles/bytev2_e2e_prefill_no_host_sync_async_p2048_d1_d16_d32_prefixoff.json
```

| decode_len | no-host-sync | no-host-sync + async |
|---:|---:|---:|
| 1 | 0.294 s | 0.293 s |
| 16 | 20.80 tok/s | 20.81 tok/s |
| 32 | 24.96 tok/s | 24.95 tok/s |

结论：

- 保留 `VLLM_BYTE_V2_PREFILL_DIRECT_SKIP_VALIDATION_SYNC` 作为默认关闭的安全实验开关。
- 不保留完全 no-host-sync，因为会吞掉 fallback pool exhaustion。
- 不保留 Python 侧 side-stream async overlap，因为当前 per-layer wait 和 residual
  host result sync 让收益不可测。
- 当前 p2048/d1 prefix-off 已接近 raw vLLM：0.298 s vs raw 0.290 s。

### Step 8: decode loader 分支提升和默认 auto split-K

状态：不保留 page branch hoist；保留默认 auto split-K。

日期：2026-06-08。

目标：

- 继续降低 ByteV2 decode attention 的长 context 延迟。
- 遵守实验策略：每个小改动先做 microbenchmark/E2E，对收益不足的改动回滚。

实验 A：page-level loader branch hoist

- 尝试在 decode kernel 中把 `page.status == compressed` 判断提升到 page 级，
  对 compressed page 调用不检查 tile fallback flag 的 fast loader。
- 该实验只针对大多数 page 都 compressed 的场景，目的是减少每个 K/V 元素 load
  时的分支和 page status 重复读取。
- correctness 通过：

```text
CUDA_VISIBLE_DEVICES=2 .venv/bin/python -m pytest \
  tests/v1/attention/test_byte_v2_decode.py::test_byte_v2_paged_decode_attention_op_gqa_wmma_split_k_cuda \
  tests/v1/attention/test_byte_v2_decode.py::test_byte_v2_paged_decode_attention_op_gqa_wmma_q_per_kv_4_cuda -q
# 2 passed
```

decode-only 对比：

```text
benchmarks/profiles/current4_decode_before_page_branch_hoist.json
benchmarks/profiles/current4_decode_after_page_branch_hoist.json
benchmarks/profiles/current4_decode_after_page_branch_hoist_fallback003.json
```

| seq_len | before median_us | after median_us | delta |
|---:|---:|---:|---:|
| 1024 | 1069.06 | 1057.79 | -1.1% |
| 2048 | 1167.36 | 1166.34 | -0.1% |
| 4096 | 1268.74 | 1267.71 | -0.1% |

结论：收益远低于保留门槛，已回滚，不保留该 fast loader。

实验 B：默认启用 auto split-K

- 之前只有显式设置 `VLLM_BYTE_V2_DECODE_SPLIT_K=0` 才启用 page-count
  heuristic；未设置环境变量时默认 `max_split_k=8`。
- 本轮把未设置环境变量的默认行为改为同一套 auto heuristic：

```text
num_pages >= 128 -> max_split_k=32
num_pages >=  64 -> max_split_k=16
num_pages >=  16 -> max_split_k=8
otherwise       -> max_split_k=1

active_head_groups >= 64 -> cap to 8
active_head_groups >= 32 -> cap to 16
```

- 正整数 `VLLM_BYTE_V2_DECODE_SPLIT_K=N` 仍然强制覆盖。
- `VLLM_BYTE_V2_DECODE_SPLIT_K=0` 或负数仍走 auto。
- benchmark JSON 记录项补充 `VLLM_BYTE_V2_DECODE_SPLIT_K`，避免后续实验无法区分
  split=8 和 auto。

decode-only split-K 重扫：

```text
benchmarks/profiles/current4_decode_splitk_resweep.json
```

| seq_len | best split_k | best median_us | split_k=8 median_us | best / split8 |
|---:|---:|---:|---:|---:|
| 512 | 16 | 1013.76 | 1037.31 | 97.7% |
| 1024 | 16 | 1062.91 | 1131.52 | 93.9% |
| 2048 | 32 | 1170.43 | 1390.08 | 84.2% |
| 4096 | 32 | 1264.64 | 1722.88 | 73.4% |

3% sparse fallback 下趋势一致：

| seq_len | best split_k | best median_us | split_k=8 median_us | best / split8 |
|---:|---:|---:|---:|---:|
| 512 | 16 | 1004.54 | 1029.63 | 97.6% |
| 1024 | 16 | 1058.30 | 1096.70 | 96.5% |
| 2048 | 32 | 1167.87 | 1387.52 | 84.2% |
| 4096 | 32 | 1265.15 | 1727.49 | 73.2% |

E2E p2048/d16,d32，Llama-3-8B，compressed-only，3% fallback pool：

```text
benchmarks/profiles/current4_e2e_bytev2_split8_p2048_d16_d32_pool003.json
benchmarks/profiles/current4_e2e_bytev2_autosplit_p2048_d16_d32_pool003.json
benchmarks/profiles/current4_e2e_bytev2_default_autosplit_after_rebuild_p2048_d16_d32_pool003.json
```

| decode_len | split_k=8 tok/s | explicit auto tok/s | default auto tok/s | default / split8 |
|---:|---:|---:|---:|---:|
| 16 | 22.78 | 27.06 | 27.15 | 119.2% |
| 32 | 23.93 | 28.93 | 28.93 | 120.9% |

fallback pool 状态：

- 3% fallback pool 未耗尽。
- `max_next_slot=134`，`max_capacity=512`。
- `cached_tokens=2032`，说明 prefix cache 仍可用。

验证：

```text
CUDA_VISIBLE_DEVICES=2 .venv/bin/python -m pytest \
  tests/v1/attention/test_byte_v2_decode.py::test_byte_v2_paged_decode_attention_op_gqa_wmma_split_k_cuda \
  tests/v1/attention/test_byte_v2_decode.py::test_byte_v2_paged_decode_attention_op_gqa_wmma_auto_split_k_cuda \
  tests/v1/attention/test_byte_v2_decode.py::test_byte_v2_paged_decode_attention_op_gqa_wmma_q_per_kv_4_cuda -q
# 3 passed
```

构建备注：

- 修改 C++ 后需要重建 native extension；本轮执行
  `.venv/bin/python setup.py build_ext --inplace`。
- `_C_stable_libtorch.abi3.so` 已成功编译并复制到 `vllm/`。
- 该命令最后进入 `build_rust` 时由于本环境没有 Rust compiler 报错；这不影响本次
  CUDA extension 的产物，但后续若要完整 build 需要安装 Rust 或关闭 rust build。

### Step 9: profile 后的小步实验和长 context split-K 调整

状态：保留长 context 默认 split-K 调整；不保留 prefill validation sync 默认启用；
不保留 `p_shared` full-page 清零跳过实验。

日期：2026-06-08。

fresh profile，p2048/d32，Llama-3-8B，compressed-only，3% fallback pool：

| mode | output tok/s | GPU kernel total | attention/cache 主要差距 |
|---|---:|---:|---:|
| raw prefix-on | 34.59 | 1847 ms | raw attention/cache 58 ms |
| raw prefix-off | 27.18 | 2101 ms | raw attention/cache 73 ms |
| ByteV2 prefix-on | 28.63 | 2075 ms | decode stage1 182 ms，compress 59 ms，append 17 ms |

结论：

- ByteV2 与 raw 的 model GEMM 基本一致。
- 当前差距仍主要来自 ByteV2 decode attention，其次是 cache update/compress。
- ByteV2 prefix-on 已经快于 raw prefix-off，但仍落后 raw prefix-on。

实验 A：prefill validation sync 默认启用

先用 env 验证收益，不直接改默认：

```text
benchmarks/profiles/current6_bytev2_p2048_d32_prefixoff_default.json
benchmarks/profiles/current6_bytev2_p2048_d32_prefixoff_skipval.json
```

| case | output tok/s | elapsed | pool exhausted |
|---|---:|---:|:---:|
| default | 24.87 | 1.2866 s | no |
| `VLLM_BYTE_V2_PREFILL_DIRECT_SKIP_VALIDATION_SYNC=1` | 24.91 | 1.2848 s | no |

结论：E2E 收益只有约 +0.14%，不满足默认启用门槛；继续保留为显式实验开关。

实验 B：跳过 full page 的 `p_shared` 清零

尝试在 decode WMMA split stage1 中，当 `valid_rows == 16` 时跳过
`tid==0` 对 `p_shared[16x16]` 的全量清零，只在 tail page 清零有效 Q 行。

correctness：

```text
CUDA_VISIBLE_DEVICES=2 .venv/bin/python -m pytest \
  tests/v1/attention/test_byte_v2_decode.py::test_byte_v2_paged_decode_attention_op_gqa_wmma_split_k_cuda \
  tests/v1/attention/test_byte_v2_decode.py::test_byte_v2_paged_decode_attention_op_gqa_wmma_q_per_kv_4_cuda -q
# 2 passed
```

decode-only：

```text
benchmarks/profiles/current6_decode_before_skip_full_pzero.json
benchmarks/profiles/current6_decode_after_skip_full_pzero.json
```

| seq_len | before auto median_us | after auto median_us | delta |
|---:|---:|---:|---:|
| 2048 | 1141.76 | 1166.34 | +2.2% |
| 4096 | 1215.49 | 1261.06 | +3.7% |

结论：性能回退，已回滚，不保留。

实验 C：长 context split-K 上限重扫

decode-only：

```text
benchmarks/profiles/current6_decode_splitk_16_128_sweep.json
benchmarks/profiles/current6_decode_splitk_32_128_fallback003_sweep.json
```

3% fallback pool：

| seq_len | split32 median_us | split64 median_us | split128 median_us | best |
|---:|---:|---:|---:|---:|
| 2048 | 1163.26 | 1159.68 | 1171.46 | split64，小于 1% |
| 4096 | 1255.42 | 1258.50 | 1245.70 | split128，约 +0.8% |
| 8192 | 1436.67 | 1445.89 | 1382.91 | split128，约 +3.7% |

p4096 E2E，prefix cache on，gpu_memory_utilization=0.5：

```text
benchmarks/profiles/current6_bytev2_p4096_d32_default_split32_gpu05.json
benchmarks/profiles/current6_bytev2_p4096_d32_forced_split128_gpu05.json
```

| case | output tok/s | elapsed | cached_tokens | pool exhausted |
|---|---:|---:|---:|:---:|
| default split32 | 25.73 | 1.2439 s | 4080 | no |
| forced split128 | 26.54 | 1.2059 s | 4080 | no |

结论：

- p2048 不改，避免短/中 context 受 split 过多影响。
- `num_pages >= 256` 时默认 `max_split_k=128`，p4096 E2E 有约 +3.1%。
- `VLLM_BYTE_V2_DECODE_SPLIT_K=N` 正整数仍然强制覆盖；未设置或设置为 0 都走 auto。
- Python env 默认值同步为 0，避免和 native 默认 auto 语义不一致。

最终 heuristic：

```text
num_pages >= 256 -> max_split_k=128
num_pages >= 128 -> max_split_k=32
num_pages >=  64 -> max_split_k=16
num_pages >=  16 -> max_split_k=8
otherwise        -> max_split_k=1

active_head_groups >= 64 -> cap to 8
active_head_groups >= 32 -> cap to 16
```

验证：

```text
CUDA_VISIBLE_DEVICES=2 .venv/bin/python -m pytest \
  tests/v1/attention/test_byte_v2_decode.py::test_byte_v2_paged_decode_attention_op_gqa_wmma_split_k_cuda \
  tests/v1/attention/test_byte_v2_decode.py::test_byte_v2_paged_decode_attention_op_gqa_wmma_auto_split_k_cuda \
  tests/v1/attention/test_byte_v2_decode.py::test_byte_v2_paged_decode_attention_op_gqa_wmma_q_per_kv_4_cuda -q
# 3 passed

benchmarks/profiles/current6_decode_after_long_context_split128_heuristic.json
```

备注：

- 改默认后尝试直接复跑 p4096 E2E，但 GPU2 上出现外部进程占用约 28GB 显存，
  KV cache 初始化失败，因此未得到新的默认 E2E JSON。
- 这不影响上述 forced split128 的可比性：改默认前的 forced split128 和改默认后的
  `num_pages >= 256` auto 使用同一 native split path。

GPU0 复测：

日期：2026-06-08。

GPU0 空闲后重新跑 E2E，`gpu_memory_utilization=0.8`，prefix cache on，
compressed-only，3% fallback pool：

```text
benchmarks/profiles/current7_gpu0_bytev2_p4096_d32_default_auto.json
benchmarks/profiles/current7_gpu0_bytev2_p4096_d32_forced_split32.json
benchmarks/profiles/current7_gpu0_raw_p4096_d32_prefixon.json
benchmarks/profiles/current7_gpu0_raw_bytev2_p2048_d32_prefixon.json
```

| prompt_len | mode | output tok/s | elapsed | cached_tokens | pool exhausted |
|---:|---|---:|---:|---:|:---:|
| 2048 | raw prefix-on | 34.77 | 0.9204 s | 2032 | n/a |
| 2048 | ByteV2 default auto | 28.88 | 1.1082 s | 2032 | no |
| 4096 | raw prefix-on | 34.14 | 0.9372 s | 4080 | n/a |
| 4096 | ByteV2 default auto | 26.36 | 1.2140 s | 4080 | no |
| 4096 | ByteV2 forced split32 | 26.24 | 1.2194 s | 4080 | no |

结论：

- GPU0 上 p4096 新默认 auto 相比 forced split32 为 +0.45%，小于此前 GPU2
  forced split128 对照的 +3.1%，说明该 E2E 对 split-K 的敏感度和运行噪声都较高。
- 新默认没有回退，pool 未耗尽，prefix cache 正常生效。
- 当前 ByteV2 / raw：p2048 为 83.0%，p4096 为 77.2%。

## Step 10：q_per_kv=4 SIMT split stage1 实验

日期：2026-06-08。

目标：验证一个只面向 Llama3 形状的 `q_per_kv=4, head_size=128` 专用
split decode stage1 是否能比当前 WMMA split stage1 更快。

实验改动：

- 新增 opt-in kernel：`byte_v2_paged_decode_attention_gqa_q4_simt_split_stage1_kernel`
- 只在 `VLLM_BYTE_V2_DECODE_VARIANT=q4_simt`、`q_per_kv=4`、
  `head_size=head_size_v=128`、split-K path 下启用。
- stage2 reduce 沿用现有 `byte_v2_paged_decode_attention_split_reduce_kernel`。
- benchmark 增加临时 variant 注入，测试增加 q4 split 正确性用例。

正确性：

```text
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest \
  tests/v1/attention/test_byte_v2_decode.py::test_byte_v2_paged_decode_attention_op_q4_simt_split_cuda \
  tests/v1/attention/test_byte_v2_decode.py::test_byte_v2_paged_decode_attention_op_gqa_wmma_split_k_cuda \
  tests/v1/attention/test_byte_v2_decode.py::test_byte_v2_paged_decode_attention_op_gqa_wmma_q_per_kv_4_cuda -q
# 3 passed
```

decode-only：

```text
benchmarks/profiles/current8_decode_q4_baseline.json
benchmarks/profiles/current8_decode_q4_simt.json
```

| seq_len | split_k | baseline median_us | q4_simt median_us | delta |
|---:|---:|---:|---:|---:|
| 2048 | 32 | 1187.33 | 1181.70 | -0.5% |
| 2048 | 128 | 1196.54 | 1180.16 | -1.4% |
| 4096 | 32 | 1278.98 | 1302.53 | +1.8% |
| 4096 | 128 | 1257.98 | 1277.95 | +1.6% |

结论：

- q4 SIMT 在 p2048 只有小幅波动级收益，p4096 明确回退。
- 原因判断：虽然 q4 SIMT 避免了 WMMA 对 16 个 Q row 的空算，但 QK/PV 都退回
  FP32 SIMT，无法抵消 tensor core 的吞吐优势；当前瓶颈不是单纯的无效 Q row。
- 未达到保留阈值，实验代码已回滚，不进入默认路径，也不保留
  `VLLM_BYTE_V2_DECODE_VARIANT=q4_simt` 开关。

回滚后验证：

```text
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest \
  tests/v1/attention/test_byte_v2_decode.py::test_byte_v2_paged_decode_attention_op_gqa_wmma_split_k_cuda \
  tests/v1/attention/test_byte_v2_decode.py::test_byte_v2_paged_decode_attention_op_gqa_wmma_q_per_kv_4_cuda \
  tests/v1/attention/test_byte_v2_decode.py::test_byte_v2_paged_decode_attention_op_gqa_wmma_auto_split_k_cuda -q
# 3 passed
```

后续方向：

- 不再优先做 q4 SIMT 替代 WMMA。
- 下一步更值得尝试的是减少 split workspace/reduce 开销，或在当前 WMMA stage1 内
  优化 page metadata/fallback 分支和 softmax 串行部分。

## Step 11：WMMA softmax/p_shared q-row 并行化实验

日期：2026-06-08。

目标：减少当前 WMMA decode kernel 中 `tid == 0` 串行清理 `p_shared` 并计算
各 q row softmax 权重的开销。

实验改动：

- 在 `byte_v2_paged_decode_attention_gqa_wmma_kernel` 和
  `byte_v2_paged_decode_attention_gqa_wmma_split_stage1_kernel` 中，把
  `tid == 0` 串行循环改为 `tid < q_per_kv` 时每个线程负责一个 q row。
- 每个 q row 只清理本 row 的 16 个 `p_shared` 槽位，不再清理完整 16x16。
- QK/PV WMMA、online softmax 公式、split reduce 均不变。

正确性：

```text
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest \
  tests/v1/attention/test_byte_v2_decode.py::test_byte_v2_paged_decode_attention_op_gqa_wmma_split_k_cuda \
  tests/v1/attention/test_byte_v2_decode.py::test_byte_v2_paged_decode_attention_op_gqa_wmma_q_per_kv_4_cuda \
  tests/v1/attention/test_byte_v2_decode.py::test_byte_v2_paged_decode_attention_op_gqa_wmma_auto_split_k_cuda -q
# 3 passed
```

decode-only：

```text
baseline: benchmarks/profiles/current8_decode_q4_baseline.json
after:    benchmarks/profiles/current8_decode_softmax_qrow.json
```

| seq_len | split_k | baseline median_us | q-row median_us | delta |
|---:|---:|---:|---:|---:|
| 2048 | 32 | 1187.33 | 1181.18 | -0.5% |
| 2048 | 128 | 1196.54 | 1215.49 | +1.6% |
| 4096 | 32 | 1278.98 | 1278.98 | 0.0% |
| 4096 | 128 | 1257.98 | 1265.66 | +0.6% |

结论：

- 该改动在 p2048/split32 只有约 +0.5% 噪声级收益。
- p4096 默认 auto 对应 split128，结果回退约 0.6%。
- 不满足保留阈值，实验代码已回滚。

最终回滚后验证：

```text
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest \
  tests/v1/attention/test_byte_v2_decode.py::test_byte_v2_paged_decode_attention_op_gqa_wmma_split_k_cuda \
  tests/v1/attention/test_byte_v2_decode.py::test_byte_v2_paged_decode_attention_op_gqa_wmma_q_per_kv_4_cuda \
  tests/v1/attention/test_byte_v2_decode.py::test_byte_v2_paged_decode_attention_op_gqa_wmma_auto_split_k_cuda -q
# 3 passed
```

## Step 12：split decode compressed page fastpath 实验

日期：2026-06-09。

目标：减少 `byte_v2_paged_decode_attention_gqa_wmma_split_stage1_kernel`
里每个 K/V 元素重复执行的 page status/fallback metadata 判断。

实验设计：

- 新增 opt-in 开关：`VLLM_BYTE_V2_DECODE_PAGE_FASTPATH=1`。
- 只作用于 split-K WMMA stage1。
- 在每个 logical page 入口读取一次 `page_status` 和 `valid_rows`。
- compressed page 直接调用 `byte_v2_load_compressed_bits()`，避免每个元素都走
  `byte_v2_load_kv_bits()` 的 status/fallback 分支。
- raw fallback page 最初也尝试 page-level raw direct load，但 GPU2 benchmark 显示
  在 3% fallback 场景明显回退，因此最终 raw fallback page 仍走旧的 generic loader。
- 默认关闭，不改变当前生产路径。

正确性：

```text
CUDA_VISIBLE_DEVICES=2 .venv/bin/python -m pytest \
  tests/v1/attention/test_byte_v2_decode.py::test_byte_v2_paged_decode_attention_op_page_fastpath_split_k_cuda \
  tests/v1/attention/test_byte_v2_decode.py::test_byte_v2_paged_decode_attention_op_gqa_wmma_split_k_cuda \
  tests/v1/attention/test_byte_v2_decode.py::test_byte_v2_paged_decode_attention_op_gqa_wmma_auto_split_k_cuda \
  tests/v1/attention/test_byte_v2_decode.py::test_byte_v2_paged_decode_attention_op_gqa_wmma_q_per_kv_4_cuda -q
# 4 passed
```

环境备注：

- GPU0 当时被 `sglang::scheduler` 占用约 37GB 且利用率 100%，早期 GPU0 benchmark
  出现 120us/2.5ms 双峰，不可信。
- 可信数据使用空闲 GPU2 重跑。

decode-only：

```text
baseline: benchmarks/profiles/current10_gpu2_decode_page_fastpath_baseline.json
final:    benchmarks/profiles/current10_gpu2_decode_page_fastpath_compressed_only_enabled.json
```

| seq_len | split_k | fallback_ratio | baseline median_us | fastpath median_us | delta |
|---:|---:|---:|---:|---:|---:|
| 2048 | 32 | 0.00 | 1157.63 | 1160.70 | +0.27% |
| 2048 | 128 | 0.00 | 1174.53 | 1172.48 | -0.17% |
| 4096 | 32 | 0.00 | 1659.39 | 1253.38 | -24.47% |
| 4096 | 128 | 0.00 | 1758.72 | 1247.74 | -29.05% |
| 2048 | 32 | 0.03 | 1157.63 | 1159.17 | +0.13% |
| 2048 | 128 | 0.03 | 1176.58 | 1171.46 | -0.44% |
| 4096 | 32 | 0.03 | 1258.50 | 1258.50 | 0.00% |
| 4096 | 128 | 0.03 | 1249.79 | 1247.23 | -0.20% |

raw fallback direct-load 负结果：

```text
benchmarks/profiles/current10_gpu2_decode_page_fastpath_enabled.json
```

在同时启用 compressed page 和 raw fallback page fastpath 时，3% fallback 场景回退：

| seq_len | split_k | fallback_ratio | baseline median_us | raw+compressed fastpath median_us |
|---:|---:|---:|---:|---:|
| 2048 | 32 | 0.03 | 1157.63 | 1418.75 |
| 2048 | 128 | 0.03 | 1176.58 | 1432.58 |
| 4096 | 32 | 0.03 | 1258.50 | 1501.18 |
| 4096 | 128 | 0.03 | 1249.79 | 1507.33 |

因此最终实现只保留 compressed page fastpath，raw fallback page 继续使用
`byte_v2_load_kv_bits()`。

E2E：

```text
benchmarks/profiles/current10_gpu2_e2e_bytev2_p4096_d32_baseline.json
benchmarks/profiles/current10_gpu2_e2e_bytev2_p4096_d32_page_fastpath.json
```

配置：GPU2，Llama-3-8B-Instruct，prompt_len=4096，decode_len=32，batch=1，
prefix cache on，cudagraph on，compressed-only，3% sparse fallback pool。

| mode | output tok/s | elapsed | cached_tokens | fallback exhausted |
|---|---:|---:|---:|:---:|
| baseline | 26.627 | 1.20177 s | 4080 | no |
| compressed page fastpath | 26.626 | 1.20183 s | 4080 | no |

结论：

- 对纯 compressed long context，decode-only kernel 有明显收益。
- 对当前真实 Llama E2E，3% sparse fallback pool 下基本持平，没有可见 E2E 提升。
- 因为默认关闭、正确性通过、E2E 不回退，保留为实验 opt-in 开关：
  `VLLM_BYTE_V2_DECODE_PAGE_FASTPATH=1`。
- 不作为默认启用；后续如果能降低 raw fallback block 数量或单独优化 raw fallback
  page load，可重新评估默认开启。

## Step 13：raw fallback page decode 路径实验

日期：2026-06-09。

目标：解释并尝试优化 `fallback_ratio=0.03` 下 page fastpath 收益消失的问题。
当前 sparse fallback page 存在两个不利点：

- raw fallback page 不走 compressed page fastpath，仍需查 `fallback_block_ids` 并从
  `fallback_pool` 读取 raw BF16 block。
- raw K block 的物理布局是 row-major，而 WMMA QK 使用的 shared K layout 是
  dim-major；直接按 shared layout 读 raw K 会造成 global load 大步长访问。

实验 A：按 page type 拆分 stage1

- 在 native op 内将 split-K stage1 拆成 compressed-only 和 raw-fallback-only
  两个 kernel。
- 两个 stage1 写入两组 partial softmax/LSE，再复用现有 reduce kernel 合并。
- 这样 raw fallback 专用路径不会污染 compressed page kernel，但会增加一次 stage1
  launch，并让 reduce 的 partial 数量翻倍。

实验 B：raw fallback K load coalescing

- 在 raw-fallback-only kernel 中，将 K load 改成按 raw block row-major 连续读取，
  再转置写入 shared memory。
- 目标是减少 raw fallback page 的 global memory stride。

实验 C：单 kernel raw fallback fastpath

- 不拆 page type，只在 split stage1 内为 raw fallback page 使用单独模板实例。
- raw K 同样使用 coalesced global read + shared transpose。
- 目标是避免实验 A 的额外 launch/reduce 开销。

正确性：

实验期间曾临时加入 split-page-types 专用测试，并通过 mixed compressed/raw fallback
场景。由于实验代码最终回滚，该临时测试函数也已删除。最终保留路径的回归测试如下：

```text
CUDA_VISIBLE_DEVICES=2 .venv/bin/python -m pytest \
  tests/v1/attention/test_byte_v2_decode.py::test_byte_v2_paged_decode_attention_op_page_fastpath_split_k_cuda -q
# 1 passed
```

最终完整回归：

```text
CUDA_VISIBLE_DEVICES=2 .venv/bin/python -m pytest \
  tests/v1/attention/test_byte_v2_decode.py::test_byte_v2_paged_decode_attention_op_page_fastpath_split_k_cuda \
  tests/v1/attention/test_byte_v2_decode.py::test_byte_v2_paged_decode_attention_op_gqa_wmma_split_k_cuda \
  tests/v1/attention/test_byte_v2_decode.py::test_byte_v2_paged_decode_attention_op_gqa_wmma_auto_split_k_cuda \
  tests/v1/attention/test_byte_v2_decode.py::test_byte_v2_paged_decode_attention_op_gqa_wmma_q_per_kv_4_cuda -q
# 4 passed
```

decode-only 结果：

```text
page_fastpath:
  benchmarks/profiles/current11_gpu2_decode_split_page_types_page_fastpath.json
split page types:
  benchmarks/profiles/current11_gpu2_decode_split_page_types_enabled.json
split page types + coalesced raw K:
  benchmarks/profiles/current11_gpu2_decode_split_page_types_coalesced_rawk.json
single-kernel raw fallback fastpath:
  benchmarks/profiles/current11_gpu2_decode_raw_fallback_fastpath_enabled.json
final retained page_fastpath after rollback:
  benchmarks/profiles/current11_gpu2_decode_after_raw_fallback_experiment_final_page_fastpath.json
```

| seq_len | split_k | page_fastpath median_us | split page types | split + coalesced raw K | single raw fastpath |
|---:|---:|---:|---:|---:|---:|
| 2048 | 32 | 1163.26 | 1209.86 | 1204.22 | 1171.46 |
| 2048 | 128 | 1177.60 | 1243.14 | 1234.43 | 1183.74 |
| 4096 | 32 | 1264.13 | 1324.03 | 1318.40 | 1265.66 |
| 4096 | 128 | 1253.38 | 1315.33 | 1305.60 | 1258.50 |

结论：

- page-type 拆分方向不适合当前 3% fallback 场景。raw page 太少，额外 stage1
  launch、扫描 page status 和更多 partial reduce 的成本超过了 raw page 专用路径收益。
- coalesced raw K load 确实让 split page-types 略快，但仍明显慢于当前 page_fastpath。
- 单 kernel raw fallback fastpath 基本持平或轻微回退，说明 raw fallback page 数量太少时，
  为其增加额外模板路径/指令体积没有稳定收益。
- 本轮实验代码已回滚，不保留 `VLLM_BYTE_V2_DECODE_SPLIT_PAGE_TYPES` 和
  `VLLM_BYTE_V2_DECODE_RAW_FALLBACK_FASTPATH`。
- 当前保留路径仍是 Step 12 的 `VLLM_BYTE_V2_DECODE_PAGE_FASTPATH=1`。

后续如果继续处理 raw fallback 性能，应该优先降低 fallback block 产生率，或在 cache
update 阶段生成 compact fallback page list，避免 decode 时让 raw-only kernel 扫描全部
logical page。

## Step 14：编码侧降低 sparse fallback block 产生率

日期：2026-06-09。

目标：先尝试“从编码侧降低 fallback block 产生率”。实现一个默认关闭的实验开关：

```text
VLLM_BYTE_V2_LOSSY_MAX_MISSES_PER_TILE=N
```

当 `N > 0` 时，ByteV2 cache update 允许一个 16x16 tile 中最多 `N` 个 BF16
值落在最佳 16-exponent window 之外。这些 outlier 的 exponent 会被夹到
`[base, base + 15]` 边界内，然后继续写 compressed page。默认 `N=0`，仍保持
lossless：只要 tile 覆盖不了全部元素，整个 block 走 sparse raw fallback。

实现范围：

- generic compressed-only cache update、raw-overlay full-page compress、decode append
  cache update 均接入 `lossy_max_misses_per_tile`。
- prefill direct encode fast path 也接入该参数：默认 `N=0` 仍使用原来的 warp
  min/max；`N>0` 时每个 warp 为当前 tile 建 256-bin exponent histogram，选择覆盖
  最多元素的 16-exponent window，并对 outlier 做边界 clamp。
- `benchmarks/kernels/benchmark_byte_v2_cache_update_kernel.py` 增加
  `--outlier-block-ratio`，用于构造每个选中 block 只有 1 个 exponent outlier 的
  cache update 场景。
- `benchmarks/kernels/benchmark_byte_v2_decode_kernel.py` 增加
  `--fallback-pattern single_outlier`，用于 decode-only 评估这种稀疏 outlier。

正确性测试：

```text
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest \
  tests/v1/attention/test_byte_v2_ops.py::test_byte_v2_native_lossy_outlier_threshold_cuda \
  tests/v1/attention/test_byte_v2_ops.py::test_byte_v2_native_sparse_fallback_pool_cuda \
  tests/v1/attention/test_byte_v2_ops.py::test_byte_v2_native_sparse_fallback_pool_exhaustion_cuda \
  tests/v1/attention/test_byte_v2_ops.py::test_byte_v2_native_decode_append_finalizes_partial_raw_fallback_cuda -q
# 4 passed
```

cache update microbenchmark，Llama-3-8B 形状，`prompt_len=2048`：

```text
benchmarks/profiles/current12_cache_update_lossy0_no_outliers.json
benchmarks/profiles/current12_cache_update_lossy1_no_outliers.json
benchmarks/profiles/current12_cache_update_lossy0_outliers_all.json
benchmarks/profiles/current12_cache_update_lossy1_outliers_all.json
```

| 场景 | lossy misses/tile | median CUDA us | packed blocks | fallback used |
|---|---:|---:|---:|---:|
| no outliers | 0 | 4706.30 | 128 | 0/512 |
| no outliers | 1 | 4818.43 | 128 | 0/512 |
| one outlier per block | 0 | 2508.35 | 0 | 128/512 |
| one outlier per block | 1 | 4788.22 | 128 | 0/512 |

结论：该开关能显著降低 fallback pool 占用，但 cache update 本身不一定更快。
原因是写 raw fallback pool 比完整压缩 compressed page 更便宜；lossy histogram 也会
带来少量额外开销。

decode-only，`seq_len=2048`，`fallback_ratio=0.03`，
`--fallback-pattern single_outlier`，低值 outlier `2^-20`：

```text
benchmarks/profiles/current12_decode_single_outlier_lossy0_fallback003.json
benchmarks/profiles/current12_decode_single_outlier_lossy1_fallback003.json
```

| split_k | lossy misses/tile | median us | fallback blocks | max_abs_diff |
|---:|---:|---:|---:|---:|
| 32 | 0 | 2418.69 | 4/8 | 0.0 |
| 32 | 1 | 2408.96 | 0/8 | 0.0 |
| 128 | 0 | 128.00 | 4/8 | 0.0 |
| 128 | 1 | 129.02 | 0/8 | 0.0 |

极端上界，`fallback_ratio=1.0`：

```text
benchmarks/profiles/current12_decode_single_outlier_lossy0_fallback100.json
benchmarks/profiles/current12_decode_single_outlier_lossy1_fallback100.json
```

| split_k | lossy misses/tile | median us | fallback blocks | max_abs_diff |
|---:|---:|---:|---:|---:|
| 32 | 0 | 121.86 | 128/128 | 0.0 |
| 32 | 1 | 2420.74 | 0/128 | 0.0 |
| 128 | 0 | 138.24 | 128/128 | 0.0 |
| 128 | 1 | 128.00 | 0/128 | 0.0 |

最终判断：

- 保留为实验开关，默认关闭。它主要用于降低 compressed-only sparse fallback pool
  耗尽风险、提高压缩率，而不是默认性能优化。
- 不把 `VLLM_BYTE_V2_LOSSY_MAX_MISSES_PER_TILE=1` 作为推荐默认值；真实模型精度还
  需要单独评估。
- 对性能目标而言，下一步不应继续押注“把 raw fallback page 变 compressed page”。
  当前 decode kernel 中 raw fallback page 省掉了解压，有些 split 配置反而更快。
- 如果后续继续处理 raw fallback 路径，应优先做 compact fallback metadata/list，
  让调度层只对真实 fallback page 做额外处理，而不是扫描全部 logical page。

## Step 15：当前基线刷新与小粒度 hot-loop 实验

日期：2026-06-09。

目标：按照当前文档建议，先刷新当前最优路径基线，再尝试一个小粒度 no-fallback
hot-loop 优化；如果没有效果则不保留。

### 当前 decode-only 基线

配置：GPU0，Llama-3-8B 形状，`VLLM_BYTE_V2_DECODE_PAGE_FASTPATH=1`，
`VLLM_BYTE_V2_LOSSY_MAX_MISSES_PER_TILE=0`。

```text
benchmarks/profiles/current13_decode_sweep_page_fastpath.json
```

| seq_len | fallback_ratio | best split_k | best median_us |
|---:|---:|---:|---:|
| 1024 | 0.00 | 16 | 1044.99 |
| 1024 | 0.03 | 16 | 1043.94 |
| 2048 | 0.00 | 64 | 1151.47 |
| 2048 | 0.03 | 64 | 1154.05 |

观察：

- p1024 最优 split 仍是 16。
- p2048 split 32/64 基本持平，当前 auto split=32 不需要改。
- 3% synthetic fallback 对 decode-only median 几乎没有影响，说明当前 benchmark 的
  fallback 注入和真实 E2E 的 fallback 分布/数据流不是同一个瓶颈。

### 当前 E2E 基线

配置：GPU0，`prompt_len=1024`，`decode_len=16,32`，`batch=1`，
`VLLM_BYTE_V2_DECODE_PAGE_FASTPATH=1`，3% sparse fallback pool。

```text
benchmarks/profiles/current13_e2e_raw_bytev2_p1024_d16_d32.json
```

| mode | d16 output tok/s | d32 output tok/s | d32 fallback assigned blocks |
|---|---:|---:|---:|
| raw | 34.92 | 35.26 | 0 |
| ByteV2 compressed-only | 28.04 | 29.65 | 2048 |

观察：

- 当前 p1024 ByteV2 已经达到 raw 的约 80%-84%，比早期 27% 明显改善。
- 真实 Llama KV 下，p1024 prompt 每层约 64 个 block，但 d32 后总 assigned fallback
  blocks 为 2048，几乎等于 32 层全部 prompt block 都进入 block-level fallback。
- 因此真实 E2E 的主要问题不是少量 fallback 分支，而是当前 block-level fallback
  太粗，导致大部分真实 KV 实际按 raw fallback 读。

### 实验 A：跳过 compressed tile fallback byte 检查

实现：

- 临时新增 `byte_v2_load_compressed_bits_no_tile_fallback()`。
- 只在 `VLLM_BYTE_V2_DECODE_PAGE_FASTPATH=1` 的 compressed page 分支使用。
- 理由：当前 block-level fallback 保证 compressed page 内 tile fallback byte 通常为
  0，可以尝试去掉每个 K/V element 的 tile fallback 检查。

正确性：

```text
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest \
  tests/v1/attention/test_byte_v2_decode.py::test_byte_v2_paged_decode_attention_op_page_fastpath_split_k_cuda \
  tests/v1/attention/test_byte_v2_decode.py::test_byte_v2_paged_decode_attention_op_gqa_wmma_split_k_cuda \
  tests/v1/attention/test_byte_v2_decode.py::test_byte_v2_paged_decode_attention_op_gqa_wmma_auto_split_k_cuda \
  tests/v1/attention/test_byte_v2_decode.py::test_byte_v2_paged_decode_attention_op_gqa_wmma_q_per_kv_4_cuda -q
# 4 passed
```

benchmark：

```text
benchmarks/profiles/current13_decode_sweep_page_fastpath_unchecked_tile.json
```

| seq_len | fallback_ratio | split_k | baseline median_us | unchecked median_us | delta |
|---:|---:|---:|---:|---:|---:|
| 1024 | 0.00 | 16 | 1044.99 | 1057.79 | -1.2% |
| 1024 | 0.03 | 16 | 1043.94 | 1049.60 | -0.5% |
| 2048 | 0.00 | 32 | 1152.00 | 1153.50 | -0.1% |
| 2048 | 0.03 | 32 | 1157.12 | 1153.54 | +0.3% |

结论：

- 没有稳定收益，p1024 还回退。
- 实验代码已回滚，不保留 unchecked compressed tile loader。
- 这说明当前瓶颈不在 tile fallback byte 这一处小分支上。

### 实验 B：真实 E2E 下启用 lossy=1

目标：验证 Step 14 的“降低 fallback block 产生率”在真实 Llama KV 上是否能提升
E2E，而不是只在 synthetic single-outlier benchmark 上有效。

配置：

```text
VLLM_BYTE_V2_LOSSY_MAX_MISSES_PER_TILE=1
benchmarks/profiles/current13_e2e_bytev2_p1024_d16_d32_lossy1.json
```

| mode | d16 output tok/s | d32 output tok/s | d32 fallback assigned blocks |
|---|---:|---:|---:|
| ByteV2 lossless | 28.04 | 29.65 | 2048 |
| ByteV2 lossy=1 | 28.39 | 26.55 | 218 |

结论：

- `lossy=1` 确实把 fallback assigned blocks 从 2048 降到 218，说明真实 Llama KV 中
  很多 block 是少量 exponent outlier 触发 fallback。
- 但 d32 E2E 反而回退，从 29.65 tok/s 降到 26.55 tok/s。原因是更多 page 走
  compressed decode 后需要解压；raw fallback page 在当前 kernel 中省掉了解压，
  对短/中 context 不一定更慢。
- 因此 `VLLM_BYTE_V2_LOSSY_MAX_MISSES_PER_TILE=1` 继续保持默认关闭，只作为容量和
  fallback pool 压力实验开关。

### 下一步判断

本轮两个实验都说明，继续做小分支优化或单纯减少 block-level fallback 不足以稳定提升
性能。下一步应转向两类更结构性的改动：

1. **tile-level fallback**：不要让一个坏 tile 把整个 16-token block 变成 raw fallback。
   只有不可压缩 tile 走 raw/tile fallback，其余 tile 继续 compressed。这样才能同时
   保留压缩率和避免全 block 解压成本失衡。
2. **更高压缩 performance format**：例如 K12/V8 或 K8/V8。当前 12-bit tile 只有
   1.326x 理论压缩率，真实 E2E 即使 kernel 优化后也很难稳定超过 raw。

短期如果继续做 compact fallback metadata/list，应该只作为 tile-level fallback 的配套
元数据实验，而不是再拆一个 raw-fallback-only stage1 kernel；之前 Step 13 已证明单独
拆 page type 在 3% fallback 场景不划算。

## Step 16：真实 Llama KV 的 tile-level fallback 统计

目标：验证 block-level fallback 是否真的被少数 tile/outlier 放大，并判断后续应该做
tile-level fallback 还是 element-level outlier list。

实现：

- 新增只读诊断接口 `get_byte_v2_tile_fallback_stats()`。
- 新增脚本 `benchmarks/byte_v2_tile_fallback_stats.py`。
- 统计方式：
  - 真实跑一次 vLLM ByteV2 compressed-only generate。
  - 读取每层当前 `kv_cache` 和 sparse fallback pool。
  - 对 full raw fallback block 中的 K/V 按 `16x16` tile 重新检查 lossless
    ByteV2 16-exponent window。
  - compressed page 视为 0 个 bad tile；raw fallback block 只把真正不可压缩的
    tile 计为 tile-level fallback。
  - 对 bad tile 进一步统计最少 miss/outlier 数量，用来判断 element-level
    outlier list 是否值得做。

sanity check：

```text
.venv/bin/python -m py_compile \
  vllm/v1/attention/backends/byte_v2_attn.py \
  vllm/v1/worker/worker_base.py \
  benchmarks/byte_v2_tile_fallback_stats.py

.venv/bin/python -m ruff check \
  vllm/v1/attention/backends/byte_v2_attn.py \
  vllm/v1/worker/worker_base.py \
  benchmarks/byte_v2_tile_fallback_stats.py
# All checks passed
```

合成 sanity test 构造 1 个 raw fallback block，其中只有 1 个 K tile 含单个
exponent outlier，统计结果为 `full_bad_tiles=1`，符合预期。

真实 Llama-3-8B-Instruct，batch=1，decode_len=1，lossless：

```text
VLLM_BYTE_V2_COMPRESSED_ONLY_CACHE=1
VLLM_BYTE_V2_ENABLE_SPARSE_FALLBACK_POOL=1
VLLM_BYTE_V2_SPARSE_FALLBACK_POOL_RATIO=0.03
VLLM_BYTE_V2_SPARSE_FALLBACK_POOL_MIN_BLOCKS=512
VLLM_BYTE_V2_USE_NATIVE_KERNELS=1
VLLM_BYTE_V2_LOSSY_MAX_MISSES_PER_TILE=0
```

输出：

```text
benchmarks/profiles/current14_tile_fallback_stats_p1024_d1.json
benchmarks/profiles/current14_tile_fallback_stats_p2048_d1.json
```

| prompt_len | full blocks | block raw fallback | block fallback ratio | total tiles | bad tiles if tile-level | tile fallback ratio | bad tile ratio inside raw blocks |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 2048 | 1843 | 89.99% | 262144 | 4941 | 1.8848% | 2.0945% |
| 2048 | 4096 | 3715 | 90.70% | 524288 | 9925 | 1.8930% | 2.0872% |

bad tile 的 miss/outlier 分布：

| prompt_len | bad tiles | misses <= 1 | misses <= 2 | misses > 8 | mean misses | max misses |
|---:|---:|---:|---:|---:|---:|---:|
| 1024 | 4941 | 4868 | 4941 | 0 | 1.0148 | 2 |
| 2048 | 9925 | 9756 | 9925 | 0 | 1.0170 | 2 |

结论：

- 当前 block-level fallback 确实严重放大了 raw fallback：约 90% full blocks 进入
  raw fallback，但如果按 tile 粒度，真正不可压缩 tile 只有约 1.9%。
- 在 raw fallback block 内，也只有约 2.1% tile 真正需要 fallback；其余约 97.9%
  tile 其实可以继续用 compressed payload。
- bad tile 几乎都是单 outlier：p1024 下 4868/4941 个 bad tile 只缺 1 个元素；
  p2048 下 9756/9925 个 bad tile 只缺 1 个元素，全部 bad tile 都在 2 个 miss
  以内。

后续判断：

1. **必须优先做 tile-level fallback pool。** 这是当前压缩率和 raw fallback pool
   压力的最大结构性问题；它可以把 raw fallback 读写从整块 64 KiB 降到少数
   512B raw tile。
2. **element-level outlier list 有潜力，但不应该第一步直接做。** 统计说明 bad
   tile 大多只有 1 个 outlier，因此 element-level outlier list 的压缩率会更好；
   但 decode 需要对 tile 做 patch/scatter，可能破坏 fused attention 的热路径。
   更稳妥的路线是先实现 tile-level fallback，验证容量和性能收益，再对 bad tile
   继续实验 small outlier list。
3. `lossy=1` 能降低 block fallback 的原因也被解释清楚：它等价于把这些 1-2 个
   outlier clamp 到窗口内。但这是有损路径，默认不能启用；lossless 方向应该用
   tile fallback 或 outlier list 保存这些值。

## Step 17：实现 tile-level fallback pool MVP

目标：把 Step 16 统计结论落到 native cache update/decode 中，避免单个坏 tile 把
整个 16-token block 放大成 raw BF16 block fallback。

实现内容：

- allocator-aware metadata 增加：
  - `fallback_tile_ids: [num_blocks, total_tiles_per_block] int32`
  - `fallback_tile_next_slot: [1] int32`
- raw tile pool 复用已有 `fallback_pool` 字节空间，但以 512B 为一个 tile slot 寻址。
  `fallback_next_slot` 仍用于完整 raw block slot，`fallback_tile_next_slot` 独立用于
  raw tile slot。修复后 block slot 从 pool 头部递增分配，tile slot 从 pool 尾部
  递减分配，避免两种粒度互相覆盖。
- `byte_v2_reshape_and_cache()` 新增可选 tile fallback 参数。
  - direct prefill encode kernel 支持 per-tile 判断。
  - generic compressed-only cache update 支持 per-tile fallback。
  - decode append 的 partial block 仍先使用 block-level raw fallback；当 block 满 16
    token 后，若有 tile metadata，则 finalization 写 compressed page + raw tile
    fallback。
- `byte_v2_paged_decode_attention()` 新增可选 `fallback_tile_ids`。
  - compressed page fastpath 和 fallback-aware loader 都能在 tile fallback byte 为 1
    时从 raw tile pool 读取 `16x16` BF16 tile。
  - block-level raw fallback page 仍按 `fallback_block_ids` 读取完整 raw block。
- PyTorch correctness fallback 增加 tile fallback 读取支持，主要用于 smoke test 和
  关闭 native kernel 的调试路径。
- ByteV2 backend 的 lazy fallback pool 和 vLLM allocator 注册路径都已传递 tile
  metadata；正常 E2E 不需要 backend 额外偷偷分配 metadata。

当前限制：

- tile slot 和 block slot 都是单调分配，block/page 重新压回 compressed 时只清除
  mapping，不回收 slot。
- `fallback_tile_ids` 是 dense int32 表。Llama-3 8B
  `num_kv_heads=8, head_size=128, head_size_v=128` 时每 block 128 个 tile，metadata
  为 `num_blocks * 128 * 4B`。后续如果 profile 显示 metadata load 或容量明显影响
  性能，再实验 compact list、bitset + prefix index 或按 request 的 bad tile list。
- 还没有实现 element-level outlier list；它仍是 tile fallback 之后的候选压缩方案。

验证：

```text
.venv/bin/python -m py_compile \
  vllm/_custom_ops.py \
  vllm/v1/attention/backends/byte_v2_ops.py \
  vllm/v1/attention/backends/byte_v2_attn.py \
  vllm/v1/attention/backends/byte_v2_torch.py \
  vllm/v1/kv_cache_interface.py \
  vllm/v1/worker/gpu/attn_utils.py \
  vllm/v1/worker/gpu_model_runner.py \
  tests/v1/attention/test_byte_v2_ops.py

.venv/bin/python setup.py build_ext --inplace

CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest \
  tests/v1/attention/test_byte_v2_ops.py::test_byte_v2_native_tile_fallback_pool_cuda \
  tests/v1/attention/test_byte_v2_ops.py::test_byte_v2_native_sparse_fallback_pool_cuda \
  tests/v1/attention/test_byte_v2_ops.py::test_byte_v2_native_decode_append_finalizes_partial_raw_fallback_cuda \
  tests/v1/attention/test_byte_v2_decode.py::test_byte_v2_paged_decode_attention_op_page_fastpath_split_k_cuda -q
# 4 passed

CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest \
  tests/v1/attention/test_byte_v2_backend.py \
  tests/v1/attention/test_byte_v2_ops.py \
  tests/v1/attention/test_byte_v2_decode.py -q
# 40 passed

.venv/bin/python -m ruff check \
  vllm/_custom_ops.py \
  vllm/v1/attention/backends/byte_v2_ops.py \
  vllm/v1/attention/backends/byte_v2_attn.py \
  vllm/v1/attention/backends/byte_v2_torch.py \
  vllm/v1/kv_cache_interface.py \
  vllm/v1/worker/gpu/attn_utils.py \
  vllm/v1/worker/gpu_model_runner.py \
  tests/v1/attention/test_byte_v2_ops.py
# All checks passed
```

小模型 E2E smoke：

```text
CUDA_VISIBLE_DEVICES=0
VLLM_BYTE_V2_COMPRESSED_ONLY_CACHE=1
VLLM_BYTE_V2_ENABLE_SPARSE_FALLBACK_POOL=1
VLLM_BYTE_V2_SPARSE_FALLBACK_POOL_RATIO=0.03
VLLM_BYTE_V2_SPARSE_FALLBACK_POOL_MIN_BLOCKS=16
VLLM_BYTE_V2_USE_NATIVE_KERNELS=1
.venv/bin/python - <<'PY'
# LLM(model="/mnt/sda1/yxz/new_idea/models/pythia-14m",
#     kv_cache_dtype="byte_v2", gpu_memory_utilization=0.1, ...)
# generate max_tokens=2
# collect get_byte_v2_sparse_fallback_stats
PY

prompt_len=24
output_len=2
enabled_layers=6
total_capacity=21420
total_next_slot=6
total_assigned_blocks=6
tile_capacity_per_layer=57120
total_assigned_tiles=1
any_exhausted=False
```

项目内 `tests/v1/attention/test_byte_v2_e2e.py` 也尝试运行过，但该测试固定
`gpu_memory_utilization=0.2`；当前 GPU0 启动时只有约 7.29/44.42 GiB 空闲，低于
vLLM 要求的 8.88 GiB，因此失败原因是显存环境不足，不是 tile fallback 路径错误。

新增关键单测：

- 构造 1 个 full block，其中 K tile 不可压缩、V tile 可压缩。
- cache update 后 page status 必须是 `COMPRESSED`。
- `fallback_block_ids == [-1]`，`fallback_next_slot == 0`。
- `fallback_tile_next_slot == 1`，只有 K tile 的 `fallback` byte 为 1。
- native paged decode 输出与 raw attention 对齐。

下一步：

1. 重新跑真实 Llama-3 8B E2E，重点看：
   - block fallback assigned blocks 是否显著下降。
   - tile fallback assigned tiles 是否接近 Step 16 预测的约 1.9%。
   - 3% pool 是否仍会耗尽。
2. profile cache update 和 decode：
   - direct prefill encode 中 per-tile fallback 的额外 histogram/metadata 写成本。
   - decode stage1 是否因为 dense `fallback_tile_ids` 读取和 tile fallback branch 回退。
3. 若压缩率改善但性能回退，再按 profile 决定是否做 compact tile metadata 或
   element-level outlier list。

## Step 18：tile-level fallback pool 真实 E2E 复测

目标：在 Step 17 实现后，重新跑真实 Llama-3 8B E2E，确认 correctness、pool
容量和性能影响。

### 修复 correctness 问题

首次 E2E 后补做 Llama-3 8B correctness smoke，发现 ByteV2 贪心输出与 raw 不一致。
原因是同一个 `fallback_pool` 内，block-level raw fallback 和 tile-level raw fallback
都从 pool 头部开始分配：

- decode append 的 partial block 会分配完整 raw block slot。
- prompt full block 中的 bad tile 会分配 raw tile slot。
- 两者同时存在时，block slot 会覆盖已经写入的 raw tile payload。

修复：

- `fallback_block_ids` 仍使用 raw block slot，从 `fallback_pool` 头部递增。
- `fallback_tile_ids` 存 absolute raw tile slot，但 slot 从 `fallback_pool` 尾部递减
  分配。
- 新增单测
  `test_byte_v2_native_tile_and_block_fallback_pool_do_not_overlap_cuda`：
  先写一个需要 K tile fallback 的 full block，再 decode-append 一个 partial raw
  fallback block，最后 decode 第一个 block 并与 raw attention 对齐。

验证：

```text
.venv/bin/python setup.py build_ext --inplace

CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest \
  tests/v1/attention/test_byte_v2_ops.py::test_byte_v2_native_tile_and_block_fallback_pool_do_not_overlap_cuda \
  tests/v1/attention/test_byte_v2_ops.py::test_byte_v2_native_tile_fallback_pool_cuda -q
# 2 passed

CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest \
  tests/v1/attention/test_byte_v2_backend.py \
  tests/v1/attention/test_byte_v2_ops.py \
  tests/v1/attention/test_byte_v2_decode.py -q
# 41 passed
```

Llama-3 8B correctness smoke：

```text
model=/mnt/sda1/yxz/byte_v2/Meta-Llama-3-8B-Instruct
prompt_len=1024
decode_len=32
temperature=0.0
gpu_memory_utilization=0.45
VLLM_BYTE_V2_COMPRESSED_ONLY_CACHE=1
VLLM_BYTE_V2_ENABLE_SPARSE_FALLBACK_POOL=1
VLLM_BYTE_V2_SPARSE_FALLBACK_POOL_RATIO=0.03
VLLM_BYTE_V2_SPARSE_FALLBACK_POOL_MIN_BLOCKS=512
VLLM_BYTE_V2_USE_NATIVE_KERNELS=1
VLLM_BYTE_V2_DECODE_PAGE_FASTPATH=1
VLLM_BYTE_V2_LOSSY_MAX_MISSES_PER_TILE=0

raw_tokens == byte_v2_tokens: True
```

### E2E benchmark

配置：

```text
GPU: A40
model: /mnt/sda1/yxz/byte_v2/Meta-Llama-3-8B-Instruct
batch_size=1
prompt_len=1024
decode_lens=16,32,64,128,256
num_runs=1
enforce_eager=True
gpu_memory_utilization=0.80
prefix_cache=True
VLLM_BYTE_V2_COMPRESSED_ONLY_CACHE=1
VLLM_BYTE_V2_ENABLE_SPARSE_FALLBACK_POOL=1
VLLM_BYTE_V2_SPARSE_FALLBACK_POOL_RATIO=0.03
VLLM_BYTE_V2_SPARSE_FALLBACK_POOL_MIN_BLOCKS=512
VLLM_BYTE_V2_USE_NATIVE_KERNELS=1
VLLM_BYTE_V2_DECODE_PAGE_FASTPATH=1
VLLM_BYTE_V2_LOSSY_MAX_MISSES_PER_TILE=0
VLLM_BYTE_V2_PREFILL_DIRECT_SKIP_VALIDATION_SYNC=1
```

输出：

```text
benchmarks/profiles/current19_e2e_tile_fallback_pool_fixed_raw_bytev2_p1024_d16_32_64_128_256.json
```

容量：

| mode | GPU KV cache size |
| --- | ---: |
| raw | 166,768 tokens |
| ByteV2 compressed-only + tile fallback | 208,112 tokens |

ByteV2 容量为 raw 的约 124.8%。相比早期 3% sparse fallback 的约 127.5%，tile
metadata 的 dense `fallback_tile_ids` 消耗了一部分容量。

吞吐：

| decode_len | raw tok/s | ByteV2 tok/s | ByteV2/raw |
| ---: | ---: | ---: | ---: |
| 16 | 34.39 | 14.01 | 40.7% |
| 32 | 34.69 | 13.31 | 38.4% |
| 64 | 34.82 | 13.04 | 37.4% |
| 128 | 34.87 | 12.89 | 37.0% |
| 256 | 34.89 | 12.80 | 36.7% |

fallback 统计：

| decode_len | assigned block fallback | total block slots used | max block slot/layer | assigned tile fallback | tile slots used | exhausted |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 16 | 64 | 96 | 3 | 5,007 | 5,172 | False |
| 32 | 96 | 160 | 5 | 5,283 | 5,448 | False |
| 64 | 128 | 288 | 9 | 5,959 | 6,124 | False |
| 128 | 160 | 544 | 17 | 7,557 | 7,722 | False |
| 256 | 192 | 1,056 | 33 | 10,976 | 11,141 | False |

最终每层 block pool 容量为 512，最多只用到 33；tile pool capacity 为 65,536，
最多只用到 409。3% pool 没有耗尽。

### 结论

- tile-level fallback pool 解决了 correctness 和容量压力：真实 Llama KV 不再因为
  少数 bad tile 把大多数 prompt block 放大成完整 raw fallback block。
- 当前性能没有提升，反而明显低于之前约 80% raw 的 block-level fallback 路径；修复
  pool 覆盖后仍只有约 37%-41% raw。
- 这说明当前瓶颈在 tile fallback decode 热路径：compressed page 中每个 element
  都会检查 tile fallback byte，并可能通过 dense `fallback_tile_ids` 间接加载 raw tile。
  对真实 Llama KV，bad tile 数量虽然少，但 metadata/branch 已进入所有 compressed
  tile 的热路径。

下一步优化方向：

1. page/block 级 `has_tile_fallback` bit 已在 Step 19 尝试并回退。E2E 无稳定收益，
   后续不要重复该方向。
2. 把 `fallback_tile_ids` dense 表改成 compact bad-tile list 或 page-local small index，
   避免 decode 对所有 tile 付 dense metadata 成本。
3. profile 当前 E2E，分离 cache update、decode attention、tile fallback load 三部分
   成本，再决定是否继续保留当前 dense tile metadata 方案。

## Step 19：page-level has_tile_fallback header fastpath

目标：落实 Step 18 的第一条后续方向。compressed page 中大多数 tile 可能没有
tile fallback；如果 page 级别能确认没有 raw tile，就让 decode loader 跳过
`fallback_tile_ids` dense metadata 和 per-tile fallback byte 检查。

实现：

- page header offset 2 增加 `has_tile_fallback` 字节。
- full compressed block 写入时置 0。
- tile-level fallback 写入路径只要任一 K/V tile 进入 raw tile pool，就把该 page
  header 置 1。
- block-level raw fallback、partial raw fallback 和 raw-overlay 兼容路径置 0。
- split-K WMMA page fastpath 中：
  - `status == COMPRESSED && has_tile_fallback == 0`：调用无 tile metadata 的
    compressed loader。
  - `status == COMPRESSED && has_tile_fallback != 0`：调用 fallback-aware loader，
    通过 `fallback_tile_ids` 读取 raw tile。

正确性验证：

```text
.venv/bin/python -m py_compile \
  vllm/v1/attention/backends/byte_v2_layout.py \
  tests/v1/attention/test_byte_v2_ops.py

.venv/bin/python -m ruff check \
  vllm/v1/attention/backends/byte_v2_layout.py \
  tests/v1/attention/test_byte_v2_ops.py

.venv/bin/python setup.py build_ext --inplace

CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest \
  tests/v1/attention/test_byte_v2_ops.py::test_byte_v2_native_tile_fallback_pool_cuda \
  tests/v1/attention/test_byte_v2_ops.py::test_byte_v2_native_tile_and_block_fallback_pool_do_not_overlap_cuda \
  tests/v1/attention/test_byte_v2_decode.py::test_byte_v2_paged_decode_attention_op_page_fastpath_split_k_cuda \
  tests/v1/attention/test_byte_v2_decode.py::test_byte_v2_paged_decode_attention_op_gqa_wmma_split_k_cuda \
  tests/v1/attention/test_byte_v2_decode.py::test_byte_v2_paged_decode_attention_op_gqa_wmma_auto_split_k_cuda -q
# 5 passed

CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest \
  tests/v1/attention/test_byte_v2_backend.py \
  tests/v1/attention/test_byte_v2_ops.py \
  tests/v1/attention/test_byte_v2_decode.py -q
# 41 passed

CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest \
  tests/v1/attention/test_byte_v2_layout.py -q
# 11 passed
```

测试覆盖：

- 普通 compressed full block 的 `has_tile_fallback == 0`。
- tile fallback block 的 `has_tile_fallback == 1`。
- tile fallback slot 和 block fallback slot 不重叠时，tile fallback page 标 1，
  block-level raw fallback page 标 0。

E2E 结果：

```text
benchmarks/profiles/current20_e2e_page_has_tile_fallback_fastpath_raw_bytev2_p1024_d16_32_64_128_256.json
```

配置：GPU0，Llama-3-8B-Instruct，prompt_len=1024，decode_len=16/32/64/128/256，
batch=1，enforce_eager=True，compressed-only，3% sparse fallback pool，
`VLLM_BYTE_V2_DECODE_PAGE_FASTPATH=1`。

容量保持不变：

| mode | GPU KV cache size |
| --- | ---: |
| raw | 166,768 tokens |
| ByteV2 | 208,112 tokens |

吞吐对比 Step 18 `current19`：

| decode_len | raw current20 tok/s | ByteV2 current19 tok/s | ByteV2 current20 tok/s | current20 / current19 |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 34.42 | 14.01 | 14.00 | 99.96% |
| 32 | 34.68 | 13.31 | 13.31 | 100.02% |
| 64 | 34.81 | 13.04 | 13.00 | 99.70% |
| 128 | 34.87 | 12.89 | 12.87 | 99.85% |
| 256 | 34.89 | 12.80 | 12.79 | 99.92% |

current20 ByteV2/raw：

| decode_len | ByteV2/raw |
| ---: | ---: |
| 16 | 40.7% |
| 32 | 38.4% |
| 64 | 37.3% |
| 128 | 36.9% |
| 256 | 36.7% |

fallback pool 未耗尽：

| decode_len | assigned block fallback | max block slot/layer | exhausted |
| ---: | ---: | ---: | :---: |
| 16 | 64 | 3 / 512 | no |
| 32 | 96 | 5 / 512 | no |
| 64 | 128 | 9 / 512 | no |
| 128 | 160 | 17 / 512 | no |
| 256 | 192 | 33 / 512 | no |

临时 decode-only 数据仍保留为负例参考：

```text
benchmarks/profiles/current20_decode_kernel_page_has_tile_fallback_fastpath.json
```

这组数据是在三张 GPU 都被外部 workload 高利用率占用时跑的，只能作为 sanity
参考，不能作为保留/回滚依据。部分点出现明显异常的 split-K 非单调现象，例如
`seq_len=2048, split_k=16, fallback=0` 为约 171us 且 correctness diff 为 0；
同一环境下其它点仍在 2ms 量级，说明当前 GPU 共用状态不适合做性能结论。

最终结论：

- 不保留，实验代码已回退。
- E2E 与 Step 18 基本持平，长 decode 还轻微回退，没有达到保留门槛。
- 该 fastpath 只优化“compressed page 且无 tile fallback”的热路径；真实 Llama
  当前更大的问题仍是 dense tile fallback metadata layout、decode attention kernel
  解压/softmax/reduce 成本，以及 compressed page 解压本身。
- 后续不再继续 page-level flag 方向。下一步应优先 profile 当前 tile-level fallback
  E2E，并尝试 compact tile metadata 或更低解压成本的 format。

回退后验证：

```text
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest \
  tests/v1/attention/test_byte_v2_backend.py \
  tests/v1/attention/test_byte_v2_ops.py \
  tests/v1/attention/test_byte_v2_decode.py \
  tests/v1/attention/test_byte_v2_layout.py -q
# 52 passed

benchmarks/profiles/current20_after_revert_bytev2_p1024_d16_smoke.json
# ByteV2 d16 output tok/s = 14.02, pool exhausted = false
```

## Step 21：profile 当前 tile-level decode

目标：

- 确认当前 tile-level decode 中，解压、softmax、split reduce 分别占多少。
- 判断后续 kernel 优化应优先做哪一段，避免继续优化非瓶颈。

profile 产物：

```text
benchmarks/profiles/current21_bytev2_tile_level_p1024_d64_e2e.nsys-rep
benchmarks/profiles/current21_bytev2_tile_level_p1024_d64_e2e.json
benchmarks/profiles/current21_bytev2_tile_level_p1024_d64_e2e_cuda_gpu_kern_sum_cuda_gpu_kern_sum.csv

benchmarks/profiles/current21_ncu_decode_stage1_tile_level_p1024_s16_fb003.csv
benchmarks/profiles/current21_ncu_stage1_mode0_full.csv
benchmarks/profiles/current21_ncu_stage1_mode1_load.csv
benchmarks/profiles/current21_ncu_stage1_mode2_load_qk.csv
benchmarks/profiles/current21_ncu_stage1_mode3_load_qk_softmax.csv
```

E2E 配置：

```text
Llama-3-8B-Instruct
prompt_len = 1024
decode_len = 64
batch_size = 1
enforce_eager = true
VLLM_BYTE_V2_SPARSE_FALLBACK_POOL_RATIO = 0.03
VLLM_BYTE_V2_DECODE_PAGE_FASTPATH = 1
VLLM_BYTE_V2_LOSSY_MAX_MISSES_PER_TILE = 0
```

E2E kernel 汇总：

| kernel | total | launches | avg |
| --- | ---: | ---: | ---: |
| `byte_v2_decode_append_cache_kernel` | 427.035 ms | 1841 | 231.96 us |
| `byte_v2_paged_decode_attention_gqa_wmma_split_stage1_kernel` | 236.146 ms | 1873 | 126.08 us |
| `byte_v2_compress_touched_blocks_kernel` | 84.363 ms | 32 | 2636.34 us |
| `byte_v2_paged_decode_attention_split_reduce_kernel` | 6.319 ms | 1873 | 3.37 us |
| `byte_v2_prefill_direct_encode_blocks_kernel` | 2.699 ms | 64 | 42.17 us |

只看 ByteV2 attention：

| phase | total | share |
| --- | ---: | ---: |
| split stage1 | 236.146 ms | 97.4% |
| split reduce | 6.319 ms | 2.6% |

结论：split reduce 不是瓶颈。当前 E2E 中 ByteV2-specific 的最大项反而是
decode append cache update，其次才是 decode attention stage1。

stage1 单 kernel 内部拆分：

为了定位 stage1 内部瓶颈，临时加了 profile-only early-exit mode：

- mode0：完整 stage1。
- mode1：读 page/tile、处理 fallback、解压 K/V 到 shared 后退出。
- mode2：mode1 + QK WMMA。
- mode3：mode2 + online softmax。

该临时 hook 已在 profile 后移除，不保留在主路径。

NCU SpeedOfLight 结果：

| segment | duration | share of full |
| --- | ---: | ---: |
| full stage1 | 112.03 us | 100.0% |
| load/decode K/V | 76.90 us | 68.6% |
| QK increment | 4.22 us | 3.8% |
| softmax increment | 16.45 us | 14.7% |
| PV/write partial increment | 14.46 us | 12.9% |

结合 split reduce：

| phase | representative avg |
| --- | ---: |
| stage1 load/decode | 76.90 us |
| stage1 softmax | 16.45 us |
| split reduce kernel | 3.37 us |

NCU full profile 指标：

| metric | value |
| --- | ---: |
| Duration | 110.82 us |
| Memory Throughput | 9.96% |
| DRAM Throughput | 6.71% |
| Compute SM Throughput | 13.28% |
| Issue Slots Busy | 13.30% |
| No Eligible | 86.70% |
| Eligible Warps Per Scheduler | 0.14 |
| Issued Warp Per Scheduler | 0.13 |
| Theoretical Occupancy | 50.00% |
| Achieved Occupancy | 12.73% |
| Waves Per SM | 0.25 |
| Branch Efficiency | 98.77% |
| Executed Instructions | 5,973,764 |

NCU warnings：

- L1TEX scoreboard stall 约 4.2 cycles，占 37.0% warp issue interval。
- uncoalesced global access：558,208 excessive sectors，占 837,760 sectors 的
  67%。
- uncoalesced shared access：299,008 excessive wavefronts，占 550,400
  wavefronts 的 54%。
- grid 只有 `(1, 8, 16)`，约 0.25 waves/SM，小 batch 下并行度不足。

结论：

- 当前 tile-level decode 的主瓶颈是 compressed tile 的读取/metadata/fallback
  检查/解压到 shared memory，约占 stage1 的 69%。
- softmax 只占约 15%，split reduce 只占 ByteV2 attention 的约 2.6%，都不是首要
  优化目标。
- kernel 不是 DRAM bandwidth bound，也不是 branch divergence bound；主要问题是
  低并行度、低 eligible warp、L1TEX scoreboard stall，以及 global/shared 访问不规整。
- 下一步优先方向应是降低 load/decode 段成本：压缩页 layout 更 coalesced、tile
  metadata compact 化、减少每元素解码指令数、减少 shared memory bank conflict；
  同时单独优化 decode append cache update，因为它在 E2E 中比 attention stage1 更大。

验证：

```text
.venv/bin/python setup.py build_ext --inplace
# exit code 0

CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest \
  tests/v1/attention/test_byte_v2_ops.py::test_byte_v2_native_tile_fallback_pool_cuda \
  tests/v1/attention/test_byte_v2_decode.py::test_byte_v2_paged_decode_attention_op_page_fastpath_split_k_cuda \
  tests/v1/attention/test_byte_v2_decode.py::test_byte_v2_paged_decode_attention_op_gqa_wmma_split_k_cuda -q
# 3 passed
```

## Step 22：tile-level decode latency/issue bottleneck 解决方案

Step 21 的 profile 显示，当前 stage1 主要瓶颈不是 softmax 或 split reduce，而是
compressed tile 的读取、metadata/fallback 检查、K/V 解压到 shared memory：

```text
full stage1      = 112.03 us
load/decode K/V  =  76.90 us, 68.6%
QK               =   4.22 us,  3.8%
softmax          =  16.45 us, 14.7%
PV/write partial =  14.46 us, 12.9%
split reduce     =   3.37 us avg launch
```

因此后续实验优先解决四类问题：

- global/shared 访问不规整。
- 每个 element 反复执行 tile metadata/fallback/address 计算。
- shared memory layout 和 WMMA 输入 layout 不匹配。
- 小 batch、长 context 时 grid 太小，eligible warp 太少。

### 方案 A：tile-scope compressed loader

现状问题：

- `byte_v2_load_compressed_bits()` 是 element 粒度 helper。
- 每个 element 都会计算 `dim_tile`、`dim_in_tile`、`tile_start`。
- 每个 element 都会读 tile fallback flag。
- compressed fastpath 里连续线程对 K 的访问近似是同一 dim 的不同 row，
  对 `low[256]` 是 stride=16 访问，不利于 coalescing。

实验改法：

- 新增 tile 粒度 helper，例如：

```text
byte_v2_decode_compressed_tile_to_shared(...)
byte_v2_decode_compressed_k_tile_to_shared(...)
byte_v2_decode_compressed_v_tile_to_shared(...)
```

- 一个 16x16 tile 只计算一次 `tile_start`。
- 一个 16x16 tile 只读取一次 `base` 和 `fallback flag`。
- compressed tile 和 tile fallback 在 tile 粒度分流，不在 element 粒度分流。
- 对 compressed tile，优先让连续线程读取连续 payload：

```text
low[row * 16 + dim]
packed[(row * 16 + dim) / 2]
```

保留标准：

- CUDA 单测通过。
- NCU 中 stage1 `load/decode K/V` 下降。
- uncoalesced global excessive sectors 或 L1TEX scoreboard stall 下降。
- E2E `p1024 d64` ByteV2 吞吐不低于修改前。

回滚标准：

- stage1 duration 无下降。
- E2E 无改善或回退超过噪声。
- shared/global uncoalesced 指标没有改善。

### 方案 B：K shared layout + WMMA layout 重排

现状问题：

- compressed payload 自然是 row-major tile。
- 当前 K shared 为了喂 WMMA，逻辑上更接近 `[dim][row]`。
- 这会在 global coalescing 和 shared layout 之间产生冲突。

实验 B1：

- global 读取按 row-major 连续读取。
- 写入时转置到当前 `k_shared` layout。
- 不改 WMMA `matrix_b` 的读取方式。

实验 B2：

- K 也按 row-major 写入 shared。
- 尝试把 WMMA `matrix_b` 改为 `col_major` 或调整 `load_matrix_sync`
  stride，使 K 不再需要 shared transpose。

保留标准：

- B1/B2 分别和 baseline 比较。
- 如果 B2 降低 shared excessive wavefronts，并且 QK correctness 不变，则保留。
- 如果 B2 破坏 WMMA layout 或没有收益，只保留 B1 或全部回滚。

### 方案 C：减少解码指令数

现状问题：

- 当前每个 element 都独立完成 metadata/address/decode。
- `base >> 1`、`base & 1`、`tile_start` 等可 tile-scope 复用。
- packed nibble 每两个 element 共用一个 byte，但当前逻辑按 element 读取。

实验改法：

- 在 tile-scope loader 中缓存：

```text
base_hi = base >> 1
base_lsb = base & 1
tile_start
payload pointers
```

- 每个线程尽量一次处理两个相邻 element，共用同一个 packed byte。
- 尝试 `uint32_t` 或 `uint64_t` vector load 读取 low/packed payload，再在寄存器中拆分。

保留标准：

- NCU executed instructions 下降。
- load/decode 段 duration 下降。
- correctness diff 保持在现有阈值内。

### 方案 D：shared memory bank conflict 优化

现状问题：

- NCU 显示 shared excessive wavefronts 约 54%。
- 当前 `k_shared`、`v_shared`、`p_shared` 都是紧密 16 对齐 layout，
  可能在 WMMA load 或后续 per-dim accumulation 中出现 bank conflict。

实验改法：

- 给 shared tile 增加 padding stride，例如 16 -> 16/17/32 的可选布局。
- 只先对 K/V shared 做实验，避免同时影响 softmax/PV。
- 结合 B1/B2 比较不同 layout 对 WMMA load 和 shared excessive wavefronts 的影响。

保留标准：

- shared excessive wavefronts 下降。
- stage1 duration 下降。
- shared memory 增加不能导致 occupancy 明显恶化。

### 方案 E：split-K/page parallel autotune

现状问题：

- `batch=1, seq_len=1024, num_kv_heads=8, split_k=16` 时 grid 约
  `(1, 8, 16)`，只有 128 CTAs。
- NCU 显示约 `0.25 waves/SM`，小 batch 下并行度不足。

实验改法：

- 对 decode-only 和 E2E 分别扫描：

```text
split_k = 8, 16, 32, 64
seq_len = 1024, 2048, 4096
decode_len = 16, 64, 128
```

- 对小 batch 长 context 允许更高 split-K。
- 监控 split reduce 是否从 2.6% 变成新瓶颈。

保留标准：

- E2E 吞吐提升。
- stage1 并行度改善。
- split reduce 增长可控。

回滚标准：

- decode-only 变快但 E2E 不变或变慢。
- split reduce 或 partial output 写入成本吞掉收益。

### 方案 F：decode append cache update 单独优化

虽然 Step 21 的问题聚焦 tile-level decode attention，但 E2E 中最大的
ByteV2-specific kernel 是 cache update：

```text
byte_v2_decode_append_cache_kernel = 427.035 ms
decode attention stage1           = 236.146 ms
```

后续 attention stage1 优化后，如果不处理 append cache update，整体 E2E 仍会被
cache update 限制。

实验改法：

- profile `byte_v2_decode_append_cache_kernel` 内部阶段。
- 检查 decode append 是否存在 token 粒度重复 metadata/fallback 检查。
- 尝试 block/tile 粒度 append encode，减少 per-token scatter。
- 如果 decode append 只追加 1 token，考虑专门的 single-token fastpath。

保留标准：

- E2E ByteV2 output tok/s 提升。
- cache update total time 下降。
- prefill/direct encode 路径不受影响。

### 执行顺序

优先顺序：

1. 方案 A：tile-scope compressed loader。
2. 方案 B：K shared layout + WMMA layout 重排。
3. 方案 C/D：减少解码指令数和 shared bank conflict。
4. 方案 E：split-K/page parallel autotune。
5. 方案 F：decode append cache update。

每一步都必须单独实验：

```text
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest \
  tests/v1/attention/test_byte_v2_ops.py::test_byte_v2_native_tile_fallback_pool_cuda \
  tests/v1/attention/test_byte_v2_decode.py::test_byte_v2_paged_decode_attention_op_page_fastpath_split_k_cuda \
  tests/v1/attention/test_byte_v2_decode.py::test_byte_v2_paged_decode_attention_op_gqa_wmma_split_k_cuda -q

CUDA_VISIBLE_DEVICES=0 .venv/bin/python benchmarks/kernels/benchmark_byte_v2_decode_kernel.py \
  --variant page_fastpath --seq-len 1024 --split-k 16 \
  --fallback-ratio 0.03 --fallback-pattern single_outlier \
  --batch-size 1 --num-runs 20 --warmup-runs 5 --skip-correctness

CUDA_VISIBLE_DEVICES=0 .venv/bin/python benchmarks/benchmark_byte_v2_decode_e2e.py \
  --modes raw,byte_v2_compressed_only \
  --model /mnt/sda1/yxz/byte_v2/Meta-Llama-3-8B-Instruct \
  --prompt-len 1024 --decode-lens 64 \
  --batch-size 1 --num-runs 3 --warmup-decode-len 8 \
  --gpu-memory-utilization 0.80 --enforce-eager
```

保留/回滚原则：

- 单测失败：立即修复或回滚。
- microbench 变快但 E2E 不变：默认不保留，除非 NCU 明确证明它是后续优化的必要前置。
- E2E 变快但 profile 指标没改善：保留前需要复跑确认，避免 GPU 噪声误判。
- 任一实验都不和下一项混在一起提交，确保可以定位收益来源。

## Step 23：按 Step 22 尝试 A/B/E 多个方案

本轮目标：

- 优先尝试直接降低 stage1 load/decode 成本的方案 A/B。
- 如果 load/decode layout 方向无收益，再尝试方案 E 的 split-K/page parallel
  autotune。
- 无收益的 kernel 改动不保留。

### Baseline

配置：

```text
GPU0
variant = page_fastpath
seq_len = 1024
split_k = 16
fallback_ratio = 0.03
fallback_pattern = single_outlier
batch_size = 1
num_runs = 20
warmup_runs = 5
```

baseline：

```text
benchmarks/profiles/current22_baseline_decode_p1024_s16_fb003.json
median_us = 1055.23
tok/s = 947.66
```

### 方案 A：tile-scope compressed loader

实验 A：

- 新增 tile-scope element decode helper。
- compressed fastpath 的 K/V 都改为按 tile 遍历。
- tile header、fallback flag、tile_start 在 tile 粒度复用。

结果：

```text
benchmarks/profiles/current22_scheme_a_tile_loader_decode_p1024_s16_fb003.json
median_us = 1058.82
tok/s = 944.45
```

结论：

- 比 baseline 慢约 0.3%。
- 不保留。
- 推测 V 本来就是 row-major 连续读取，改成 tile loop 后增加了 tile/header
  循环开销，抵消了 K 侧收益。

实验 A1：

- 只保留 K tile-scope loader。
- V 恢复原始 row-major element helper 路径。

结果：

```text
benchmarks/profiles/current22_scheme_a1_k_tile_loader_decode_p1024_s16_fb003.json
median_us = 1058.82
tok/s = 944.45
```

结论：

- 仍比 baseline 慢约 0.3%。
- 不保留，相关代码已回滚。
- 说明 K global coalescing 的潜在收益被 shared transpose 写入、tile loop
  开销或 WMMA load 后续成本抵消。

验证：

```text
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest \
  tests/v1/attention/test_byte_v2_ops.py::test_byte_v2_native_tile_fallback_pool_cuda \
  tests/v1/attention/test_byte_v2_decode.py::test_byte_v2_paged_decode_attention_op_page_fastpath_split_k_cuda \
  tests/v1/attention/test_byte_v2_decode.py::test_byte_v2_paged_decode_attention_op_gqa_wmma_split_k_cuda -q
# 3 passed
```

### 方案 B2：K row-major shared + WMMA B col-major

实验改法：

- split stage1 中，K 按 `[row][dim]` 写入 `k_shared`。
- QK 的 WMMA `matrix_b` 改为 `col_major`。
- `load_matrix_sync(b_frag, k_shared + dim_base, head_size)`。

正确性：

```text
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest \
  tests/v1/attention/test_byte_v2_ops.py::test_byte_v2_native_tile_fallback_pool_cuda \
  tests/v1/attention/test_byte_v2_decode.py::test_byte_v2_paged_decode_attention_op_page_fastpath_split_k_cuda \
  tests/v1/attention/test_byte_v2_decode.py::test_byte_v2_paged_decode_attention_op_gqa_wmma_split_k_cuda -q
# 3 passed
```

性能：

```text
benchmarks/profiles/current22_scheme_b2_k_colmajor_decode_p1024_s16_fb003.json
median_us = 1063.94
tok/s = 939.91
```

结论：

- 比 baseline 慢约 0.8%。
- 不保留，代码已回滚。
- 虽然 K load 顺序更自然，但 WMMA col-major load 和新的 shared layout
  成本更高。

### 方案 E：split-K/page parallel autotune

decode-only 扫描：

```text
fallback_ratio = 0.03
fallback_pattern = single_outlier
batch_size = 1
```

| seq_len | split_k | median_us | tok/s |
| ---: | ---: | ---: | ---: |
| 1024 | 8 | 1095.17 | 913.10 |
| 1024 | 16 | 1052.67 | 949.96 |
| 1024 | 32 | 1088.51 | 918.69 |
| 1024 | 64 | 1109.50 | 901.30 |
| 2048 | 8 | 1392.64 | 718.06 |
| 2048 | 16 | 1238.02 | 807.74 |
| 2048 | 32 | 1170.43 | 854.39 |
| 2048 | 64 | 1163.26 | 859.65 |
| 4096 | 8 | 1700.86 | 587.94 |
| 4096 | 16 | 1414.14 | 707.14 |
| 4096 | 32 | 1253.38 | 797.85 |
| 4096 | 64 | 1265.66 | 790.10 |
| 4096 | 128 | 1248.26 | 801.12 |

结论：

- 1024 长度下 split-K=16 仍最优，不能提高。
- 2048 长度下 split-K=64 略优于当前默认 32。
- 4096 长度下当前默认 128 仍略优。

E2E 验证：

```text
Llama-3-8B-Instruct
prompt_len = 2048
decode_len = 64
batch_size = 1
num_runs = 1
warmup_decode_len = 8
enforce_eager = true
```

| config | output tok/s | elapsed |
| --- | ---: | ---: |
| default split heuristic | 12.84 | 4.985 s |
| `VLLM_BYTE_V2_DECODE_SPLIT_K=64` | 13.06 | 4.899 s |

结果：

- split-K=64 比当前默认约 +1.8% output tok/s。
- sparse fallback pool 未耗尽。

保留改动：

```cpp
if (num_logical_pages >= 256) {
  max_split_k = 128;
} else if (num_logical_pages >= 128) {
  max_split_k = 64;
} else if (num_logical_pages >= 64) {
  max_split_k = 16;
}
```

也就是只把 `128 <= num_logical_pages < 256` 的默认 split-K 从 32 提高到 64。
这覆盖约 2048-token context；1024-token context 仍保持 16，4096-token context
仍保持 128。

最终验证：

```text
.venv/bin/python setup.py build_ext --inplace
# exit code 0

CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest \
  tests/v1/attention/test_byte_v2_ops.py::test_byte_v2_native_tile_fallback_pool_cuda \
  tests/v1/attention/test_byte_v2_decode.py::test_byte_v2_paged_decode_attention_op_page_fastpath_split_k_cuda \
  tests/v1/attention/test_byte_v2_decode.py::test_byte_v2_paged_decode_attention_op_gqa_wmma_split_k_cuda -q
# 3 passed

benchmarks/profiles/current22_final_p1024_s16_fb003.json
# seq_len=1024 split_k=16 median_us=1063.42
```

说明：

- `current22_final_p1024_s16_fb003.json` 使用显式 `split_k=16`，路径未受本轮
  heuristic 改动影响。该点比最早 baseline 慢约 0.8%，判断为运行噪声/重编译后
  波动；保留改动只影响默认 heuristic 下的 2048-token context。
- A/A1/B2 的源码改动均已回滚，只保留方案 E 的 split-K heuristic。

下一步：

- 继续从 Step 21 的 E2E profile 出发，优先 profile 并优化
  `byte_v2_decode_append_cache_kernel`，因为它在 E2E 中比 decode attention stage1
  更大。
- 如果继续优化 attention stage1，应避免简单改变 K layout；需要更底层的压缩
  format/layout 调整，单纯把读取顺序变 coalesced 没有带来收益。

## Step 24：Load / Decode K/V 专项解决方案

### 当前状态

Step 21 的分段 profile 已确认 stage1 内部最大开销是：

```text
load/decode K/V = 76.90 us, 68.6% of stage1
```

Step 23 已经验证以下浅层方案无效：

- A：在现有格式上做 tile-scope compressed loader，K/V 都按 tile 读取。
- A1：只对 K 做 tile-scope loader，V 保持原路径。
- B2：K 按 row-major 写 shared，WMMA B 改成 col-major。

结论：

- 仅在 decode kernel 里调整 loader 或 shared layout，无法真正解决问题。
- 根因是当前 compressed tile 的物理 payload 顺序和 K 的 WMMA 输入顺序不匹配。
- K 和 V 在 attention 中需要的布局不同，不应继续强行共用同一种 tile payload
  语义。

当前 K 路径：

```text
compressed K payload: [row][dim]
k_shared / WMMA B:    [dim][row]
```

为了喂当前 WMMA row-major B，decode 必须把 K 从 `[row][dim]` 变成
`[dim][row]`。这会导致：

- K global load 对 `low[256]` 近似 stride=16。
- 每个 element 都要做 tile metadata/address/decode。
- 即使尝试 row-major shared + col-major WMMA，WMMA/shared 成本也会抵消收益。

因此下一步必须做 format/layout 级改动：**encode 阶段就把 K tile 存成
WMMA-friendly 的转置顺序，V 仍保持 row-major。**

### 目标

优先目标：

- 降低 stage1 的 `load/decode K/V` 时间。
- 降低 NCU 中的 uncoalesced global/shared access。
- 降低 L1TEX scoreboard stall 和 executed instructions。

硬性保留标准：

```text
seq_len=1024, split_k=16, fallback=0.03:
  load/decode K/V 需要明显低于 76.90 us
  stage1 duration 需要低于 112 us
  decode microbench median_us 需要优于 baseline 1055 us

E2E:
  p1024/d64 ByteV2 output tok/s 不能回退
  p2048/d64 ByteV2 output tok/s 不能回退
```

如果只改善 NCU 指标但 E2E 没有改善，默认不保留，除非它是后续格式改动的必要
前置。

### 方案 L1：K-transposed compressed tile layout

核心改动：

- K tile payload 改为 physical order `[dim_in_tile][row]`。
- V tile payload 保持 physical order `[row][dim_in_tile]`。
- payload 大小不变，page size 不变，只改变 K tile 内部元素顺序。

现有格式：

```text
K low elem = row * 16 + dim_in_tile
V low elem = row * 16 + dim_in_tile
```

新格式：

```text
K low elem = dim_in_tile * 16 + row
V low elem = row * 16 + dim_in_tile
```

这样 K decode 时可以直接：

```text
global payload order == k_shared order == WMMA B row-major order
```

预期收益：

- K global load 从 stride-like 访问变成 contiguous 访问。
- K shared store 不需要转置。
- WMMA B 保持当前 row-major 路径，不使用已验证变慢的 col-major B。
- 不改变 V 的高效 row-major 路径。

需要修改的位置：

- `byte_v2_store_compressed_tile()`：
  - `is_value == false` 时按 `[dim][row]` 写 low/packed。
  - `is_value == true` 时保持 `[row][dim]`。
- `byte_v2_load_compressed_bits()` 或新增 K/V 专用 loader：
  - K 用 `elem = dim_in_tile * 16 + row`。
  - V 用 `elem = row * 16 + dim_in_tile`。
- split stage1 compressed fastpath：
  - K load loop 保持当前 `idx = dim * 16 + row` 的 `k_shared` 写法。
  - loader 内部使用 K-transposed payload。
- Python/CPU codec 参考实现和 layout 测试：
  - 更新 K tile decode reference。
  - V reference 不变。

兼容策略：

- ByteV2 仍是实验格式，可以直接更新格式语义。
- 为了降低调试风险，先用 env guard 做实验：

```text
VLLM_BYTE_V2_K_TRANSPOSED_TILE_LAYOUT=1
```

- 实验保留后，再决定是否变成默认格式。

第一步实验范围：

- 只改 compressed tile。
- block-level raw fallback 保持 row-major raw block。
- tile-level fallback 先保持现有 raw tile pool 语义，作为 fallback 慢路径。

保留标准：

- CUDA 单测通过。
- decode-only p1024/split16 比 baseline `1055 us` 明显下降。
- NCU 中 `uncoalesced global excessive sectors` 下降。
- `load/decode K/V` 早退 profile 下降。

回滚标准：

- correctness diff 变大。
- p1024 decode-only 不改善。
- E2E p1024/d64 或 p2048/d64 回退。

### 方案 L2：K tile fallback pool 也改成转置布局

背景：

- L1 只优化 compressed K tile。
- 真实 Llama-3 8B 下仍存在 tile fallback。
- 如果 tile fallback 比例高，K fallback raw tile 仍可能保持 row-major，从而继续
  带来 K load/decode 开销。

改动：

- tile fallback pool 中 K raw tile 也按 `[dim][row]` 存。
- V raw tile 仍按 `[row][dim]` 存。
- `byte_v2_store_raw_bits_to_tile_pool()` / `byte_v2_load_raw_bits_from_tile_pool()`
  增加 `is_value` 或 layout 参数。

注意：

- block-level raw fallback pool 可以暂时不改，因为它是完整 raw block，语义更接近
  vLLM raw cache。
- tile-level fallback 是 ByteV2 自己的补充结构，更适合跟随 compressed tile layout。

保留标准：

- 在真实 Llama-3 p1024/p2048 E2E 中 tile fallback 不少时，decode attention
  stage1 继续下降。
- tile fallback correctness 测试通过。

回滚标准：

- tile fallback 测试复杂度显著上升但性能无收益。
- fallback pool 分配或统计逻辑变得不稳定。

### 方案 L3：K/V 专用 vectorized tile decoder

L1 改格式后，再继续降低 per-element decode 指令数。

当前每个元素都要：

```text
load low byte
load packed nibble byte
extract nibble
reconstruct bf16 high byte
store shared
```

改动：

- 新增 K/V 专用 tile decoder：

```text
byte_v2_decode_k_transposed_tile_to_shared(...)
byte_v2_decode_v_rowmajor_tile_to_shared(...)
```

- 每个线程处理两个连续 element，共用一个 packed byte。
- 尝试 `uint32_t` / `uint64_t` vector load：
  - low payload 连续读取。
  - packed payload 连续读取。
- `base_hi = base >> 1`、`base_lsb = base & 1` 在 tile scope 复用。

执行顺序：

1. 先在 L1 正确且有收益后做。
2. 先只优化 no-fallback compressed tile。
3. 再处理 tile fallback path。

保留标准：

- NCU `Executed Instructions` 下降。
- `load/decode K/V` 早退时间下降。
- decode-only 和 E2E 同时不回退。

### 方案 L4：warp-specialized load/decode 与 WMMA pipeline

如果 L1/L3 后仍受 latency/eligible warp 限制，再考虑 pipeline。

思路：

- 一个 CTA 内分工：
  - warp0/warp1 负责下一 tile 的 K/V decode。
  - warp2 负责当前 tile QK/PV WMMA。
  - warp3 负责 softmax/PV accumulation 或辅助 load。
- 使用双 buffer：

```text
k_shared[2][...]
v_shared[2][...]
```

风险：

- 当前 blockDim=128、shared 已不少，双 buffer 会进一步降低 occupancy。
- softmax 依赖在线 running max/denom，tile 间存在顺序依赖，pipeline 粒度要谨慎。

优先级：

- 低于 L1/L3。
- 只有在 L1/L3 后 NCU 仍显示 L1TEX scoreboard 和 no eligible 极高时才做。

保留标准：

- stage1 duration 明显下降。
- occupancy 不显著恶化。
- split reduce 不成为新瓶颈。

### 方案 L5：减少 metadata/fallback 热路径检查

当前 compressed loader 仍可能每 element 检查 tile fallback flag。

L1 格式改动后，同时可以做 metadata 分层：

- page status 在 page 粒度检查。
- tile fallback flag 在 tile 粒度检查。
- compressed no-fallback tile 进入完全无 fallback 检查的 loader。

可能实现：

```text
byte_v2_decode_k_tile_no_fallback(...)
byte_v2_decode_k_tile_with_fallback(...)
byte_v2_decode_v_tile_no_fallback(...)
byte_v2_decode_v_tile_with_fallback(...)
```

但 Step 23 说明“只做 tile-scope 分流”本身没有收益，所以 L5 必须依赖 L1 的新
payload layout，不再单独作为第一步实验。

保留标准：

- L1 已经带来收益后，L5 进一步降低 executed instructions。
- 如果 L5 单独无收益，不保留。

### 执行计划

第一轮只做 L1：

1. 加 env guard：

```text
VLLM_BYTE_V2_K_TRANSPOSED_TILE_LAYOUT=1
```

2. 修改 compressed K tile encode/decode。
3. 更新 Python reference codec 和单测。
4. 跑 correctness：

```text
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest \
  tests/v1/attention/test_byte_v2_ops.py \
  tests/v1/attention/test_byte_v2_decode.py \
  tests/v1/attention/test_byte_v2_layout.py -q
```

5. 跑 decode microbench：

```text
CUDA_VISIBLE_DEVICES=0 VLLM_BYTE_V2_K_TRANSPOSED_TILE_LAYOUT=1 \
  .venv/bin/python benchmarks/kernels/benchmark_byte_v2_decode_kernel.py \
  --variant page_fastpath --seq-len 1024 --split-k 16 \
  --fallback-ratio 0.03 --fallback-pattern single_outlier \
  --batch-size 1 --num-runs 20 --warmup-runs 5 --skip-correctness
```

6. 跑 NCU：

```text
CUDA_VISIBLE_DEVICES=0 VLLM_BYTE_V2_K_TRANSPOSED_TILE_LAYOUT=1 \
  ncu --target-processes all \
  --kernel-name regex:byte_v2_paged_decode_attention_gqa_wmma_split_stage1_kernel \
  --launch-skip 2 --launch-count 1 --set full --csv \
  --log-file benchmarks/profiles/current24_ncu_k_transposed_stage1.csv \
  .venv/bin/python benchmarks/kernels/benchmark_byte_v2_decode_kernel.py \
  --variant page_fastpath --seq-len 1024 --split-k 16 \
  --fallback-ratio 0.03 --fallback-pattern single_outlier \
  --batch-size 1 --num-runs 3 --warmup-runs 2 --skip-correctness
```

7. 跑 E2E：

```text
CUDA_VISIBLE_DEVICES=0 VLLM_BYTE_V2_K_TRANSPOSED_TILE_LAYOUT=1 \
  VLLM_BYTE_V2_SPARSE_FALLBACK_POOL_RATIO=0.03 \
  VLLM_BYTE_V2_SPARSE_FALLBACK_POOL_MIN_BLOCKS=512 \
  VLLM_BYTE_V2_USE_NATIVE_KERNELS=1 \
  VLLM_BYTE_V2_DECODE_PAGE_FASTPATH=1 \
  VLLM_BYTE_V2_LOSSY_MAX_MISSES_PER_TILE=0 \
  VLLM_BYTE_V2_PREFILL_DIRECT_SKIP_VALIDATION_SYNC=1 \
  .venv/bin/python benchmarks/benchmark_byte_v2_decode_e2e.py \
  --modes byte_v2_compressed_only \
  --model /mnt/sda1/yxz/byte_v2/Meta-Llama-3-8B-Instruct \
  --prompt-len 1024 --decode-lens 64 \
  --batch-size 1 --num-runs 3 --warmup-decode-len 8 \
  --gpu-memory-utilization 0.80 --enforce-eager
```

### 决策

优先做 L1。原因：

- 它直接解决 K physical payload 和 WMMA B layout 不匹配的问题。
- 它不增加 page size，不降低压缩率。
- 它避免了 Step 23 已验证变慢的 WMMA col-major 路径。
- 它对 V 没有负面影响，因为 V 继续保持 row-major。

如果 L1 无收益：

- 不继续在现有 ByteV2 tile 格式上做小修。
- 转向更激进格式，例如按 WMMA fragment 存储 K packed payload，或者把 K/V
  分成完全不同的 page sections。

如果 L1 有收益：

- 继续 L2，处理 K tile fallback pool。
- 然后做 L3，减少 decode 指令数。

### L1 实验结果：不保留

已尝试实现 `VLLM_BYTE_V2_K_TRANSPOSED_TILE_LAYOUT=1`：

- K compressed tile payload 物理顺序改为 `[dim_in_tile][row]`。
- V compressed tile 和 raw/tile fallback pool 保持原布局。
- native prefill direct encode、compressed-only update、native decode 和 Python
  reference helper 均同步支持该开关。

正确性验证：

```text
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest \
  tests/v1/attention/test_byte_v2_ops.py::test_byte_v2_native_prefill_direct_k_transposed_layout_cuda \
  tests/v1/attention/test_byte_v2_decode.py::test_byte_v2_paged_decode_page_fastpath_k_transposed_layout_cuda -q

结果：2 passed

CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest \
  tests/v1/attention/test_byte_v2_ops.py::test_byte_v2_native_prefill_direct_encode_full_blocks_cuda \
  tests/v1/attention/test_byte_v2_decode.py::test_byte_v2_paged_decode_attention_op_page_fastpath_split_k_cuda -q

结果：2 passed
```

性能验证：

```text
baseline:
seq_len=1024 split_k=16 fallback=0.030 median_us=1102.85
tok/s=906.74 fallback_blocks=2/6

K-transposed:
seq_len=1024 split_k=16 fallback=0.030 median_us=2406.91
tok/s=415.47 fallback_blocks=2/6

rollback 后默认路径曾出现一次异常低值：
seq_len=1024 split_k=16 fallback=0.030 median_us=98.30
tok/s=10172.53 fallback_blocks=2/6

后续重复复测恢复到约 2.4 ms，因此 98.30 us 不作为可靠 baseline。
```

结论：

- L1 正确但显著退化，暂不保留代码。
- 该实现即使关闭 env，也可能因为 runtime layout 参数进入 hot path 而拉低
  baseline；后续复测显示默认路径仍有较大波动，需要用多次重复和 E2E 判断。
- 可能原因是 per-element runtime layout branch、K low/code 的新读取顺序、以及
  shared-memory 写入/WMMA 消费之间没有形成预期收益，反而降低 issue efficiency。
- 后续不要继续做“仅改变 tile 内元素顺序”的小改动；应转向 L3/L5 的 fused
  tile decoder，目标是减少 load/decode 指令和 per-element metadata 检查，而不是
  只调整 payload 顺序。

### L3 实验：pair decode loader，不保留

在 L1 回滚后，继续尝试 L3 的一个更小变体：

- 只改 split-K stage1 的 compressed page fastpath。
- K/V compressed tile 内每个线程处理相邻两个 element。
- 两个 element 共用同一个 packed nibble byte，减少 packed load 和 nibble decode。
- tile fallback 在 tile 粒度分流，保持 correctness。

正确性：

```text
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest \
  tests/v1/attention/test_byte_v2_ops.py::test_byte_v2_native_tile_fallback_pool_cuda \
  tests/v1/attention/test_byte_v2_decode.py::test_byte_v2_paged_decode_attention_op_page_fastpath_split_k_cuda \
  tests/v1/attention/test_byte_v2_decode.py::test_byte_v2_paged_decode_attention_op_gqa_wmma_split_k_cuda -q

结果：3 passed
```

性能：

```text
baseline rollback repeat:
seq_len=1024 split_k=16 fallback=0.030 median_us=2414.08
tok/s=414.24 fallback_blocks=2/6

pair decode loader:
seq_len=1024 split_k=16 fallback=0.030 median_us=2393.09
tok/s=417.87 fallback_blocks=2/6

default fallback-aware repeat:
seq_len=1024 split_k=16 fallback=0.030 median_us=2391.04
tok/s=418.23 fallback_blocks=2/6
```

结论：

- pair decode 正确，但没有稳定提升，代码已回滚。
- 原因很可能是 tile loop、额外分流和 shared 写入顺序破坏了当前编译器对简单
  element loop 的优化；减少 packed byte 读取并没有抵消这些成本。
- 后续不要再在当前 row-major tile payload 上做线程级 pair/tile loop 微调。
- 下一步应转向更大粒度的方案：
  - 独立 K/V page section 或 WMMA-friendly payload format。
  - 或先处理 E2E 中更重的 `byte_v2_decode_append_cache_kernel`。

### L1B + L3 实验：K-transposed format + K/V 专用 no-fallback decoder，保留

上一轮 L1 失败的主要问题是 runtime layout 参数进入 hot path，并且只改了
payload 顺序，没有减少 per-element decode/fallback metadata 开销。本轮重新做一个
更干净的版本：

- K compressed tile payload 物理顺序改为 `[dim_in_tile][row]`。
- V compressed tile payload 仍保持 `[row][dim_in_tile]`。
- `byte_v2_load_compressed_bits()` 的通用路径同步理解新 K 布局，保证 fallback-aware
  和非 split decode 仍正确。
- split-K WMMA stage1 中，`fallback_pool == nullptr && fallback_tile_ids == nullptr`
  的 compressed page 走新的 no-fallback fastpath：
  - `byte_v2_decode_k_transposed_tile_to_shared_no_fallback()`
  - `byte_v2_decode_v_rowmajor_tile_to_shared_no_fallback()`
- 新 fastpath 不传 `is_value`，K/V loader 完全分离。
- K/V decoder 每个线程处理一对相邻 element，共用一个 packed byte，并用
  `byte_v2_load_u16()` 一次读取两个 low byte。
- Python layout reference 同步更新：K tile 存储时转置，读取时转回原始 K block
  语义。

#### 正确性

```text
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest \
  tests/v1/attention/test_byte_v2_layout.py \
  tests/v1/attention/test_byte_v2_ops.py::test_byte_v2_native_prefill_direct_encode_full_blocks_cuda \
  tests/v1/attention/test_byte_v2_decode.py::test_byte_v2_paged_decode_attention_op_page_fastpath_split_k_cuda \
  tests/v1/attention/test_byte_v2_decode.py::test_byte_v2_paged_decode_attention_op_gqa_wmma_split_k_cuda -q

结果：14 passed

CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest \
  tests/v1/attention/test_byte_v2_backend.py \
  tests/v1/attention/test_byte_v2_codec.py \
  tests/v1/attention/test_byte_v2_decode.py \
  tests/v1/attention/test_byte_v2_layout.py \
  tests/v1/attention/test_byte_v2_metadata.py \
  tests/v1/attention/test_byte_v2_ops.py -q

结果：61 passed
```

#### Decode-only benchmark

fallback=0，隔离 compressed no-fallback path：

```text
baseline:
seq_len=1024 split_k=16 fallback=0.000 median_us=1059.33
tok/s=943.99 fallback_blocks=0/0

L1B+L3:
seq_len=1024 split_k=16 fallback=0.000 median_us=1030.14
tok/s=970.74 fallback_blocks=0/0

L1B+L3 repeat:
seq_len=1024 split_k=16 fallback=0.000 median_us=1027.07
tok/s=973.64 fallback_blocks=0/0

L1B+L3 correctness short run:
seq_len=1024 split_k=16 fallback=0.000 median_us=1054.72
tok/s=948.12 fallback_blocks=0/0 max_abs_diff=0.0
```

fallback=0.03，确认 fallback-aware path 没有功能性回退：

```text
seq_len=1024 split_k=16 fallback=0.030 median_us=1061.89
tok/s=941.72 fallback_blocks=2/6

correctness short run:
seq_len=1024 split_k=16 fallback=0.030 median_us=1076.22
tok/s=929.17 fallback_blocks=2/6 max_abs_diff=2.015625
```

`max_abs_diff=2.015625` 来自 injected single-outlier fallback stress case；native
tile fallback 单测已覆盖正确性，benchmark 脚本只记录 diff，不做阈值判定。

#### NCU

命令：

```text
CUDA_VISIBLE_DEVICES=0 VLLM_BYTE_V2_LOSSY_MAX_MISSES_PER_TILE=0 \
  ncu --target-processes all \
  --kernel-name regex:byte_v2_paged_decode_attention_gqa_wmma_split_stage1_kernel \
  --launch-skip 2 --launch-count 1 --set full --csv \
  --log-file benchmarks/profiles/current25_ncu_vec_loader_stage1_p1024_s16_fb0.csv \
  .venv/bin/python benchmarks/kernels/benchmark_byte_v2_decode_kernel.py \
  --variant page_fastpath --seq-len 1024 --split-k 16 \
  --fallback-ratio 0.0 --batch-size 1 --num-runs 3 --warmup-runs 2 \
  --skip-correctness
```

结果：

```text
stage1 duration: 86.688 us
historical Speed-of-Light baseline duration: 112.032 us

executed instructions: 4,435,328
registers/thread: 48
eligible warps/scheduler: 0.13
no eligible: 87.34%
achieved occupancy: 12.73%
L1/TEX hit rate: 68.64%
L2 hit rate: 60.78%
uncoalesced global excessive sectors: 159,488 / 310,016, 51%
uncoalesced shared excessive wavefronts: 397,312 / 624,128, 64%
```

解读：

- stage1 duration 相比历史 Speed-of-Light baseline 有明显下降。
- global uncoalesced 仍高，说明 K-transposed + pair decoder 只缓解了一部分问题。
- shared excessive wavefronts 仍是主要残留问题，后续仍需要处理 shared layout /
  WMMA load layout。
- eligible warp 仍很低，decode stage1 仍受 latency/issue 限制。

#### E2E

```text
CUDA_VISIBLE_DEVICES=0 \
  VLLM_BYTE_V2_SPARSE_FALLBACK_POOL_RATIO=0.03 \
  VLLM_BYTE_V2_SPARSE_FALLBACK_POOL_MIN_BLOCKS=512 \
  VLLM_BYTE_V2_USE_NATIVE_KERNELS=1 \
  VLLM_BYTE_V2_DECODE_PAGE_FASTPATH=1 \
  VLLM_BYTE_V2_LOSSY_MAX_MISSES_PER_TILE=0 \
  VLLM_BYTE_V2_PREFILL_DIRECT_SKIP_VALIDATION_SYNC=1 \
  .venv/bin/python benchmarks/benchmark_byte_v2_decode_e2e.py \
  --modes byte_v2_compressed_only \
  --model /mnt/sda1/yxz/byte_v2/Meta-Llama-3-8B-Instruct \
  --prompt-len 1024 --decode-lens 64 --batch-size 1 --num-runs 1 \
  --warmup-decode-len 8 --gpu-memory-utilization 0.80 --enforce-eager \
  --max-model-len 1296 --max-num-batched-tokens 1280 \
  --output-json benchmarks/profiles/current25_vec_loader_e2e_p1024_d64_b1.json
```

结果：

```text
decode_len=64 median_output_tps=12.65 median_elapsed_s=5.060
sparse fallback exhausted=false
total_next_slot / total_capacity = 192 / 16384
max layer next_slot / capacity = 6 / 512
```

注意：这个 E2E run 命中了 prefix cache，且当前可用 KV cache capacity 与旧报告不同，
因此不能把 `12.65 tok/s` 直接当作严格 apples-to-apples 的历史吞吐提升；它只用于
确认新格式没有导致真实 Llama-3 compressed-only 路径功能性回退。

#### 决策

保留本轮代码。理由：

- fallback=0 decode-only 稳定小幅提升，repeat median 从 1059.33 us 降到
  1027.07 us，约 3.0%。
- NCU stage1 duration 从历史 112.032 us 降到 86.688 us。
- fallback=0.03 和 tile fallback 单测均通过，没有破坏 sparse/tile fallback。
- E2E compressed-only p1024/d64 跑通，fallback pool 未耗尽。

限制：

- 该改动仍没有解决 shared excessive wavefronts 和 low eligible warp。
- 对真实 E2E 的收益会被 cache update、fallback pages、prefix cache 命中情况和
  scheduler 开销稀释。
- 下一步不应继续只改 K physical order，而应处理 shared layout/WMMA load 或
  `byte_v2_decode_append_cache_kernel`。

### S2 实验：K shared stride padding 到 32，不保留

目标：

- 只优化 QK WMMA 的 K shared layout。
- 不改 V shared。
- 不改 compressed payload 格式。
- 把 K shared 从逻辑 `[head_size][16]` / `ldm=16` 改成
  `[head_size][32]` / `ldm=32`，尝试降低 WMMA B shared bank conflict。

实现范围：

- 新增临时常量 `kByteV2KSharedStride = 32`。
- `k_shared` 从 `kByteV2MaxHeadSize * 16` 扩成
  `kByteV2MaxHeadSize * 32`。
- K decoder 写入：

```text
k_shared[dim * 32 + row]
```

- QK WMMA B load：

```text
load_matrix_sync(b_frag, k_shared + dim_base * 32, 32)
```

正确性：

```text
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest \
  tests/v1/attention/test_byte_v2_ops.py::test_byte_v2_native_tile_fallback_pool_cuda \
  tests/v1/attention/test_byte_v2_decode.py::test_byte_v2_paged_decode_attention_op_page_fastpath_split_k_cuda \
  tests/v1/attention/test_byte_v2_decode.py::test_byte_v2_paged_decode_attention_op_gqa_wmma_split_k_cuda \
  tests/v1/attention/test_byte_v2_decode.py::test_byte_v2_paged_decode_attention_op_gqa_wmma_q_per_kv_4_cuda -q

结果：4 passed
```

decode-only：

```text
上一轮保留版本 fallback=0 repeat:
seq_len=1024 split_k=16 fallback=0.000 median_us=1027.07

S2 K shared pad32 fallback=0:
seq_len=1024 split_k=16 fallback=0.000 median_us=1126.91

S2 K shared pad32 fallback=0 repeat:
seq_len=1024 split_k=16 fallback=0.000 median_us=1105.92

S2 K shared pad32 fallback=0.03:
seq_len=1024 split_k=16 fallback=0.030 median_us=1034.24
```

NCU：

```text
上一轮保留版本:
stage1 duration: 86.688 us
static shared memory/block: 14,944 B
theoretical occupancy: 50%
block limit shared mem: 6
shared excessive wavefronts: 397,312 / 624,128, 64%

S2 K shared pad32:
stage1 duration: 86.944 us
static shared memory/block: 19,040 B
theoretical occupancy: 41.67%
block limit shared mem: 5
shared excessive wavefronts: 446,464 / 673,280, 66%
```

结论：

- S2 正确但不保留。
- padding 没有降低 stage1 duration，反而让 fallback=0 decode-only 明显退化。
- shared excessive wavefronts 从 397k 增到 446k，说明简单把 K leading dimension
  变成 32 并没有改善 WMMA shared access，反而增加了 shared footprint。
- theoretical occupancy 从 50% 降到 41.67%，shared memory 成本更高。
- 代码已回滚；回滚后 fallback=0 复测恢复到：

```text
seq_len=1024 split_k=16 fallback=0.000 median_us=1034.75
```

后续不要再做单纯 K shared stride padding。下一步如果继续 shared/WMMA 方向，应做
更明确的 WMMA fragment swizzle 或直接切换到 CUTLASS/CUTE 风格 fragment layout；
否则优先回到 E2E 更重的 `byte_v2_decode_append_cache_kernel`。

### S3 实验：CUTE cooperative_gemm swizzled QK，不保留

目标：

- 尝试把 split-K decode stage1 的 QK 从 `nvcuda::wmma::load_matrix_sync`
  替换为 CUTE/CUTLASS 风格 fragment layout。
- 只动 QK，PV、softmax、split reduce 保持不变。
- 默认路径不变，实验路径通过临时 env `VLLM_BYTE_V2_DECODE_CUTE_QK=1`
  启用。

实现尝试：

- 本地 CUTLASS/CUTE 头文件可用，`cache_kernels.cu` 可以包含
  `cute/tensor.hpp` 和 `cute/algorithm/cooperative_gemm.hpp`。
- 第一版尝试使用 CUTE 教程中的 `Swizzle<3,3,3>` + 8x64 swizzle atom，
  编译失败：

```text
tile_to_shape: block shape does not divide the target shape
```

- 原因：教程 atom 面向较大 GEMM tile，block shape 近似 8x64，而 ByteV2 QK
  micro tile 是 16x16。
- 第二版改为对 16x16 row-major offset 直接做 CUTE XOR swizzle：

```text
composition(Swizzle<3,3,3>, Layout<Shape<_16,_16>, Stride<_16,_1>>)
```

- `SM80_16x8x16_F32BF16BF16F32_TN` 的 B fragment 只需要 2 个 u32，
  所以 A 使用 `SM75_U32x4_LDSM_N`，B 使用 `SM75_U32x2_LDSM_N`。
- CUTE 路径编译通过，并且以下正确性测试通过：

```text
CUDA_VISIBLE_DEVICES=0 VLLM_BYTE_V2_DECODE_CUTE_QK=1 \
  .venv/bin/python -m pytest \
  tests/v1/attention/test_byte_v2_decode.py::test_byte_v2_paged_decode_attention_op_gqa_wmma_split_k_cuda \
  tests/v1/attention/test_byte_v2_decode.py::test_byte_v2_paged_decode_attention_op_page_fastpath_split_k_cuda \
  tests/v1/attention/test_byte_v2_decode.py::test_byte_v2_paged_decode_attention_op_gqa_wmma_q_per_kv_4_cuda -q

结果：3 passed
```

性能：

```text
临时 CUTE 代码存在时，默认 WMMA path:
seq_len=1024 split_k=16 fallback=0.000 median_us=2380.29

临时 CUTE QK path:
seq_len=1024 split_k=16 fallback=0.000 median_us=2393.60

回滚临时代码后恢复版 repeat:
seq_len=1024 split_k=16 fallback=0.000 median_us=1037.33
```

结论：

- 不保留本轮 CUTE 实验代码。
- 正确性可做通，但第一版实现需要先从当前 `q_shared/k_shared` 重排到
  CUTE swizzled 16x16 tile，再调用 `cooperative_gemm`。这个额外 shared-to-shared
  重排和 64-thread tiled MMA 形态抵消了所有潜在收益。
- 更重要的是，即使 env 不启用 CUTE，临时把 split-stage1 模板化为
  `UseCuteQk` 也改变了默认 WMMA kernel 的编译形态，导致默认 decode-only
  退化到约 2.38 ms。因此不能把该实验以 hidden env 形式留在主代码里。

后续如果继续 CUTE 方向，不能再从现有 WMMA shared layout 临时重排，而应新建
独立 experimental kernel，并满足以下条件后再接入主路径：

- cache encode 阶段直接写入 CUTE/ldmatrix 友好的 K physical layout，decode 不再
  做 shared-to-shared 重排。
- Q/K/V shared layout、ldmatrix copy atom、MMA atom 和 epilogue store 作为一个
  完整 tiled MMA design 一起生成，避免混用 WMMA layout。
- 先用独立 microbenchmark 验证 QK-only latency、shared wavefront、eligible warps
  确实优于现有 WMMA，再迁移到 paged decode。
- 默认 WMMA kernel 必须保持二进制形态不受实验模板影响；实验 kernel 用独立
  symbol/launcher，失败可直接移除。

## Step 25：decode append cache update hot path，保留

### 背景

前面 profile 显示 E2E 中 `byte_v2_decode_append_cache_kernel` 比 decode
attention stage1 更重。因此本轮回到方案 F，单独优化 decode append cache update。

本轮先用 cache update microbenchmark 隔离 decode append：

```text
CUDA_VISIBLE_DEVICES=0 VLLM_BYTE_V2_USE_NATIVE_KERNELS=1 \
VLLM_BYTE_V2_SPARSE_FALLBACK_POOL_RATIO=0.03 \
VLLM_BYTE_V2_SPARSE_FALLBACK_POOL_MIN_BLOCKS=1 \
.venv/bin/python benchmarks/kernels/benchmark_byte_v2_cache_update_kernel.py \
  --mode decode_append \
  --decode-steps 16,64 \
  --num-blocks 8192 \
  --fallback-pool-ratio 0.03 \
  --fallback-pool-min-blocks 1 \
  --num-runs 20 \
  --warmup-runs 5
```

基线：

```text
steps=16 per_step_cuda_us=2608.22
steps=64 per_step_cuda_us=2346.62
```

构建后默认基线复测：

```text
steps=16 per_step_cuda_us=2676.61
steps=64 per_step_cuda_us=2300.15
```

### 实验 25.1：跳过 decode append host validation sync，opt-in 保留

问题：

- `byte_v2_reshape_and_cache()` 的 decode append fast path 每 token 都会把
  `fast_result` 和 `fallback_next_slot` D2H 拷回 host，并
  `cudaStreamSynchronize()`。
- 主路径 `ByteV2AttentionImpl.do_kv_cache_update()` 不消费 packed block id。
- 这个同步会把每 token cache update 固定放大到毫秒级。

实现：

- 新增 env `VLLM_BYTE_V2_DECODE_APPEND_SKIP_VALIDATION_SYNC=1`。
- 只影响 `compressed_only_pages && has_sparse_fallback && num_tokens == 1`
  的 decode append fast path。
- 打开后 kernel launch 后直接返回空 packed list，不做 D2H result/fallback
  检查。
- 默认值为 0，现有 fail-closed 行为不变。

结果：

```text
默认路径:
steps=16 per_step_cuda_us=2676.61
steps=64 per_step_cuda_us=2300.15

skip validation sync:
steps=16 per_step_cuda_us=589.02
steps=64 per_step_cuda_us=631.34
```

结论：

- host sync 是 decode append 的第一层固定成本。
- 该开关有明显 microbenchmark 收益，但不是默认安全语义：pool exhaustion 等
  错误会延迟暴露。
- 因此作为实验/benchmark opt-in 保留，暂不默认启用。

### 实验 25.2：拆 full-block finalize，确认压缩串行瓶颈

打开 skip-sync 后继续扫不同 decode steps：

```text
steps=1  per_step_cuda_us=14.34
steps=15 per_step_cuda_us=6.69
steps=16 per_step_cuda_us=588.61
steps=31 per_step_cuda_us=309.28
steps=32 per_step_cuda_us=596.54
steps=63 per_step_cuda_us=495.56
steps=64 per_step_cuda_us=619.60
```

结论：

- `steps=15` 总 GPU 时间只有约 100 us。
- `steps=16` 突增到约 9.4 ms。
- 第 16 个 token 的 full-block finalize/compress 是第二个主要瓶颈。
- 原因是 decode append full-block 分支由 `tid0` 串行执行
  `byte_v2_raw_block_is_compressible()` 和 `byte_v2_store_compressed_block()`，
  需要串行扫描所有 K/V tile。

### 实验 25.3：CTA 内 tile-parallel full-block finalize，保留

实现：

- 新增 `byte_v2_finalize_raw_block_parallel()` device helper。
- 满 16 行后，CTA 内每个 warp 负责若干 16x16 K/V tile：
  - 并行扫描 exponent min/max 或 lossy histogram。
  - 并行判断 tile 是否需要 fallback。
  - 并行写 compressed payload。
  - 有 `fallback_tile_ids` 时，bad tile 写入 tile fallback pool。
  - 无 tile fallback 且 block 不可压缩时，保持 full raw fallback page。
- 修改 `byte_v2_decode_append_cache_kernel`：
  - partial append 仍只更新 raw fallback block 和 page header。
  - full append 不再让非 tid0 提前 return，而是整个 CTA 调用 parallel finalize。

正确性：

```text
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest \
  tests/v1/attention/test_byte_v2_ops.py::test_byte_v2_native_decode_append_uses_partial_raw_fallback_cuda \
  tests/v1/attention/test_byte_v2_ops.py::test_byte_v2_native_decode_append_finalizes_partial_raw_fallback_cuda \
  tests/v1/attention/test_byte_v2_ops.py::test_byte_v2_native_decode_append_fallback_pool_exhaustion_cuda \
  tests/v1/attention/test_byte_v2_ops.py::test_byte_v2_native_tile_fallback_pool_cuda -q

结果：4 passed
```

skip-sync correctness smoke：

```text
VLLM_BYTE_V2_DECODE_APPEND_SKIP_VALIDATION_SYNC=1
skip decode append tile metadata smoke passed
```

性能：

```text
parallel finalize + skip validation sync:
steps=1  per_step_cuda_us=16.38
steps=15 per_step_cuda_us=7.30
steps=16 per_step_cuda_us=9.47
steps=31 per_step_cuda_us=7.89
steps=32 per_step_cuda_us=9.06
steps=63 per_step_cuda_us=8.37
steps=64 per_step_cuda_us=8.94
```

对比实验 25.2：

- `steps=16` 从约 9.4 ms 降到 0.15 ms。
- `steps=64` 从约 39.7 ms 降到 0.57 ms。
- full-block finalize 已不再是毫秒级瓶颈。
- partial append 从约 6.69 us/step 到 7.30 us/step，退化很小。

带 E2E 实际使用的 `fallback_tile_ids` metadata 复测：

```text
steps=16 per_step_cuda_us=9.15, tile_fallback=0
steps=64 per_step_cuda_us=8.94, tile_fallback=0
```

默认同步路径复测：

```text
parallel finalize, default validation sync:
steps=16 per_step_cuda_us=1976.96
steps=64 per_step_cuda_us=1732.41
```

默认路径也比基线下降，但仍被 per-token host sync 限制。

raw `reshape_and_cache_flash` 参考：

```text
raw steps=16 per_step_cuda_us=3.90
raw steps=64 per_step_cuda_us=3.63
```

当前结论：

- ByteV2 decode append 在 skip-sync + parallel finalize 下约 8.9 us/step。
- raw vLLM cache update 约 3.6-3.9 us/step。
- ByteV2 cache update 已从毫秒级缩小到 raw 的约 2.3-2.5x。
- 剩余差距主要来自 ByteV2 仍要写 raw partial fallback pool，并在 full block
  时执行压缩编码；raw 只做 reshape/cache store。

### E2E 状态

本轮尝试跑 Llama-3-8B compressed-only E2E：

```text
CUDA_VISIBLE_DEVICES=0 ... benchmark_byte_v2_decode_e2e.py \
  --modes byte_v2_compressed_only \
  --prompt-len 1024 \
  --decode-lens 16,64 \
  --batch-size 1 \
  --enforce-eager
```

失败原因不是代码错误，而是 GPU0 显存不足：

```text
Free memory on device cuda:0 (6.78/44.42 GiB) is less than desired
gpu_memory_utilization 0.8
```

`nvidia-smi` 显示三张 A40 均只有约 7-8 GiB 空闲，因此本轮无法完成
Llama-3-8B E2E 对比。需要等至少 25-35 GiB 空闲显存后复测：

```text
VLLM_BYTE_V2_DECODE_APPEND_SKIP_VALIDATION_SYNC=0/1
decode_lens=16,64,128,256
raw vs byte_v2_compressed_only
```

### 保留决策

保留：

- `byte_v2_finalize_raw_block_parallel()`。
- decode append full-block 分支接入 parallel finalize。
- `VLLM_BYTE_V2_DECODE_APPEND_SKIP_VALIDATION_SYNC` opt-in 开关。

理由：

- 正确性单测通过。
- 默认路径不改变安全语义，且 cache update microbenchmark 下降。
- opt-in skip-sync 在 benchmark 中有显著收益，可用于后续验证“去 host sync 后”
  ByteV2 cache update 的真实 kernel 下限。

后续：

- 等 GPU 空闲后做 E2E 复测。
- 如果 E2E 使用 skip-sync 有收益，需要设计 production-safe 的 deferred error
  reporting：例如 device-side sticky error flag，在 batch/step 边界统一检查，而不是
  每层每 token 同步。
- decode append kernel 进一步逼近 raw 的方向是减少 partial raw fallback pool 写入：
  对 compressed page 直接维护 partial compressed/tile state，或把 16-token block 的
  encode 与下一步 attention 调度 overlap。

## Step 26：Step 25 后 Llama-3-8B E2E 复测

### 配置

GPU2 空闲后，使用同一组 p1024/b1/eager 配置复测：

```text
CUDA_VISIBLE_DEVICES=2
model=/mnt/sda1/yxz/byte_v2/Meta-Llama-3-8B-Instruct
prompt_len=1024
decode_lens=16,64,128,256
batch_size=1
num_runs=1
warmup_decode_len=8
block_size=16
gpu_memory_utilization=0.80
max_model_len=1296
max_num_batched_tokens=1280
enforce_eager=True
prefix cache=True
VLLM_BYTE_V2_SPARSE_FALLBACK_POOL_RATIO=0.03
VLLM_BYTE_V2_SPARSE_FALLBACK_POOL_MIN_BLOCKS=0
```

原始结果：

```text
benchmarks/profiles/bytev2_e2e_step25_default_sync_p1024_b1_20260609.json
```

### KV 容量

日志中 KV cache size：

```text
raw vLLM:                  166,768 tokens
ByteV2 compressed-only:    210,560 tokens
```

ByteV2 compressed-only 容量是 raw 的约 126.3%。

### 吞吐

| Mode | Decode len | Elapsed (s) | Output tok/s | ByteV2 / raw |
|---|---:|---:|---:|---:|
| raw | 16 | 0.465 | 34.38 | 100.0% |
| raw | 64 | 1.837 | 34.83 | 100.0% |
| raw | 128 | 3.669 | 34.89 | 100.0% |
| raw | 256 | 7.335 | 34.90 | 100.0% |
| ByteV2 compressed-only | 16 | 1.148 | 13.93 | 40.5% |
| ByteV2 compressed-only | 64 | 4.515 | 14.17 | 40.7% |
| ByteV2 compressed-only | 128 | 8.991 | 14.24 | 40.8% |
| ByteV2 compressed-only | 256 | 18.135 | 14.12 | 40.4% |

对比旧 p1024/b1 结果，ByteV2 compressed-only 的 decode_len=256 从约 4.52
tok/s 提升到 14.12 tok/s，主要来自 Step 25 中 decode append full-block
parallel finalize。

### Sparse fallback pool

最终 compressed-only stats：

```text
any_exhausted=false
total_next_slot=992
total_capacity=12640
max_next_slot=31
max_capacity=395
```

3% pool 没有耗尽。最终最紧张 layer 的 full raw fallback slot 使用为 31/395，
约 7.8%。tile fallback 使用量最高约 399/50560，也远低于 tile fallback 容量。

### skip decode-append sync E2E

GPU2 再次空闲后，补跑：

```text
CUDA_VISIBLE_DEVICES=2 \
VLLM_BYTE_V2_USE_NATIVE_KERNELS=1 \
VLLM_BYTE_V2_SPARSE_FALLBACK_POOL_RATIO=0.03 \
VLLM_BYTE_V2_SPARSE_FALLBACK_POOL_MIN_BLOCKS=0 \
VLLM_BYTE_V2_DECODE_APPEND_SKIP_VALIDATION_SYNC=1 \
.venv/bin/python benchmarks/benchmark_byte_v2_decode_e2e.py \
  --modes byte_v2_compressed_only \
  --prompt-len 1024 \
  --decode-lens 16,64,128,256 \
  --batch-size 1 \
  --num-runs 1 \
  --warmup-decode-len 8 \
  --block-size 16 \
  --gpu-memory-utilization 0.80 \
  --max-model-len 1296 \
  --max-num-batched-tokens 1280 \
  --enforce-eager \
  --output-json benchmarks/profiles/bytev2_e2e_step25_skip_decode_append_sync_p1024_b1_20260609.json
```

原始结果：

```text
benchmarks/profiles/bytev2_e2e_step25_skip_decode_append_sync_p1024_b1_20260609.json
```

结果：

| Decode len | raw tok/s | ByteV2 default tok/s | ByteV2 skip-sync tok/s | default/raw | skip/raw | skip vs default |
|---:|---:|---:|---:|---:|---:|---:|
| 16 | 34.38 | 13.93 | 15.58 | 40.5% | 45.3% | +11.8% |
| 64 | 34.83 | 14.17 | 15.91 | 40.7% | 45.7% | +12.3% |
| 128 | 34.89 | 14.24 | 16.04 | 40.8% | 46.0% | +12.7% |
| 256 | 34.90 | 14.12 | 16.08 | 40.4% | 46.1% | +13.9% |

skip-sync pool 状态：

```text
any_exhausted=false
total_next_slot=992
total_capacity=12640
max_next_slot=31
max_capacity=395
```

结论：

- `VLLM_BYTE_V2_DECODE_APPEND_SKIP_VALIDATION_SYNC=1` 在真实 E2E 中有效。
- 相比默认安全同步路径，输出吞吐提升约 11.8%-13.9%。
- 相比 raw vLLM，ByteV2 compressed-only 从约 40.5% raw 提高到约 45.3%-46.1%
  raw。
- pool 使用与默认同步路径一致，没有出现 pool exhausted。
- 该开关仍然不是 production-safe 默认路径，因为它跳过 per-token host-side error
  check；后续需要 device-side sticky error flag，在安全边界统一检查。

## Step 26: production-safe deferred error reporting

目标：保留 Step 25 skip-sync 的性能收益方向，但不静默吞掉 cache update 错误。

### 实现方案

- 新增 `VLLM_BYTE_V2_DEFERRED_CACHE_UPDATE_ERROR_CHECK`。
- ByteV2 attention backend 为同一 GPU 上的所有 ByteV2 layer 共享一个
  `int32[4]` sticky error tensor：
  - `[0]`: error code
  - `[1]`: block id/packed block id detail
  - `[2]`: fallback pool used
  - `[3]`: fallback pool capacity
- `byte_v2_reshape_and_cache(..., deferred_error=...)` 在 decode append fast path
  中不再把 `fast_result` 拷回 host，而是在 kernel 之后追加一个
  `byte_v2_record_deferred_cache_update_error_kernel`。
- 该记录 kernel 正常路径只读 `fast_result[0]` 并直接返回；出错时用
  `atomicCAS` 抢占 sticky flag，只记录第一个错误。
- v1 GPU model runner 在 model forward 前异步清零 sticky flag，在 model
  forward 后、logits/sampling 前统一读取并检查一次。

### 修改范围

- `vllm/envs.py`
- `vllm/v1/attention/backends/byte_v2_attn.py`
- `vllm/v1/attention/backends/byte_v2_ops.py`
- `vllm/_custom_ops.py`
- `vllm/v1/worker/gpu/model_runner.py`
- `vllm/v1/worker/gpu_model_runner.py`
- `csrc/libtorch_stable/cache_kernels.cu`
- `csrc/libtorch_stable/ops.h`
- `csrc/libtorch_stable/torch_bindings.cpp`
- `tests/v1/attention/test_byte_v2_backend.py`

### 验证

已完成：

```bash
.venv/bin/python -m py_compile \
  vllm/envs.py \
  vllm/v1/attention/backends/byte_v2_attn.py \
  vllm/v1/attention/backends/byte_v2_ops.py \
  vllm/_custom_ops.py \
  vllm/v1/worker/gpu/model_runner.py \
  vllm/v1/worker/gpu_model_runner.py

.venv/bin/python -m pytest \
  tests/v1/attention/test_byte_v2_backend.py \
  tests/v1/attention/test_byte_v2_decode.py \
  -q

.venv/bin/python -m ruff check \
  vllm/envs.py \
  vllm/v1/attention/backends/byte_v2_attn.py \
  vllm/v1/attention/backends/byte_v2_ops.py \
  vllm/_custom_ops.py \
  vllm/v1/worker/gpu/model_runner.py \
  vllm/v1/worker/gpu_model_runner.py \
  tests/v1/attention/test_byte_v2_backend.py
```

结果：

```text
tests/v1/attention/test_byte_v2_backend.py: 9 passed
tests/v1/attention/test_byte_v2_decode.py: 15 passed
combined run: 24 passed
ruff: All checks passed
```

native build 说明：

- `uv pip install -e . --torch-backend=auto` 被 CMake 外部依赖 fetch 阻塞在
  `deepgemm`/`triton_kernels` clone 阶段。
- 为了验证本次改动，复用已有 `build/temp.linux-x86_64-cpython-312` 的 Ninja
  command，手动重编译：
  - `cache_kernels.cu.o`
  - `torch_bindings.cpp.o`
  - `_C_stable_libtorch.abi3.so`
- 新 so 已同步到 `vllm/_C_stable_libtorch.abi3.so` 和 build/lib。
- schema 检查确认 `_C_cache_ops::byte_v2_reshape_and_cache` 已包含
  `Tensor!? deferred_error=None`。

E2E 已补测：

```bash
CUDA_VISIBLE_DEVICES=2 \
VLLM_BYTE_V2_USE_NATIVE_KERNELS=1 \
VLLM_BYTE_V2_SPARSE_FALLBACK_POOL_RATIO=0.03 \
VLLM_BYTE_V2_SPARSE_FALLBACK_POOL_MIN_BLOCKS=0 \
VLLM_BYTE_V2_DEFERRED_CACHE_UPDATE_ERROR_CHECK=1 \
.venv/bin/python benchmarks/benchmark_byte_v2_decode_e2e.py \
  --modes byte_v2_compressed_only \
  --prompt-len 1024 \
  --decode-lens 16,64,128,256 \
  --batch-size 1 \
  --num-runs 1 \
  --warmup-decode-len 8 \
  --block-size 16 \
  --gpu-memory-utilization 0.80 \
  --max-model-len 1296 \
  --max-num-batched-tokens 1280 \
  --enforce-eager \
  --output-json benchmarks/profiles/bytev2_e2e_step26_deferred_error_p1024_b1_20260609.json
```

原始结果：

```text
benchmarks/profiles/bytev2_e2e_step26_deferred_error_p1024_b1_20260609.json
```

结果：

| Decode len | raw tok/s | ByteV2 default tok/s | ByteV2 skip-sync tok/s | ByteV2 deferred-safe tok/s | deferred/raw | deferred/default | deferred/skip |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 16 | 34.38 | 13.93 | 15.58 | 15.45 | 44.9% | 110.9% | 99.2% |
| 64 | 34.83 | 14.17 | 15.91 | 15.80 | 45.4% | 111.5% | 99.3% |
| 128 | 34.89 | 14.24 | 16.04 | 15.94 | 45.7% | 111.9% | 99.3% |
| 256 | 34.90 | 14.12 | 16.08 | 15.99 | 45.8% | 113.3% | 99.4% |

Sparse fallback pool 状态：

```text
any_exhausted=false
total_next_slot=992
total_capacity=12640
max_next_slot=31
max_capacity=395
```

结论：

- deferred-safe 性能基本等于 unsafe skip-sync，达到 skip-sync 的 99.2%-99.4%。
- 相比 default safe sync，deferred-safe 提升约 10.9%-13.3%。
- 相比 raw vLLM，deferred-safe 约为 44.9%-45.8%。
- pool 使用量与 skip-sync 完全一致，没有出现 pool exhausted。
- 因 benchmark 脚本当时未记录新 env，JSON 的 `env` 字段缺少
  `VLLM_BYTE_V2_DEFERRED_CACHE_UPDATE_ERROR_CHECK`；本轮已补上脚本记录字段，
  后续重新跑会正确写入。

### 当前结论

- Step 25 的 parallel finalize 对 E2E 有显著收益。
- 默认安全同步路径下，ByteV2 compressed-only 已从旧的约 13% raw 提升到约
  40.5% raw。
- skip decode-append host sync 后，ByteV2 compressed-only 进一步提升到约
  45.3%-46.1% raw。
- 当前 E2E 剩余差距不再主要是 decode append full-block 压缩，而是 decode
  attention、production-safe deferred error reporting、runtime 调度和 ByteV2
  partial/fallback metadata 路径的组合。

## Step 27: deferred-safe profile 与 direct-output decode 实验

### 目的

在 Step 26 的 production-safe deferred error reporting 之后，重新确认 p2048/d32
生产路径瓶颈，并尝试一个低风险优化：让 native ByteV2 decode attention 直接写入
vLLM attention output buffer，避免临时 `decode_out` tensor 和 Python backend
`copy_`。

### Profile

使用 GPU0、cudagraph on、single-process、`prompt_len=2048`、`decode_len=32`：

```text
benchmarks/profiles/bytev2_p2048_d32_step27_deferred_singleproc_node.nsys-rep
benchmarks/profiles/bytev2_p2048_d32_step27_deferred_singleproc_node.json
```

E2E:

```text
output tok/s: 28.80
elapsed: 1.111 s
sparse fallback exhausted: false
```

Nsight measured range 内 ByteV2 相关 kernel 时间：

| 组件 | 时间 |
|---|---:|
| decode attention stage1 + reduce | 143.96 ms |
| decode append cache update | 6.38 ms |
| deferred error record | 1.21 ms |
| prefill direct encode | 1.21 ms |
| validation | 0.10 ms |

Top kernel 仍是模型 GEMM，但 ByteV2 自身剩余主瓶颈是
`byte_v2_paged_decode_attention_gqa_wmma_split_stage1_kernel`，而不是 cache update
或 host sync。

### Direct-output A/B

实现过的候选改动：

- C++ schema 增加 optional mutable output tensor。
- Python wrapper 增加 `byte_v2_paged_decode_attention_out()`。
- backend decode 分支直接传入 output view。
- CUDA 单测验证 output tensor 被原地写入。

同一份代码内用实验 env 做 A/B：

```text
VLLM_BYTE_V2_DECODE_DIRECT_OUTPUT=0  # 旧路径: 临时输出 + copy_
VLLM_BYTE_V2_DECODE_DIRECT_OUTPUT=1  # direct-output 路径
```

结果：

| 模式 | median output tok/s | median elapsed |
|---|---:|---:|
| direct-output off | 29.149 | 1.098 s |
| direct-output on | 29.148 | 1.098 s |

结论：收益在噪声内，未达到保留门槛。因此 direct-output schema/wrapper/backend
实验代码已删除，不保留。

保留的相关修正：

- backend 旧路径 copy 兼容 2D/3D output；否则 cudagraph capture 下 output 为 3D
  时会出现 shape mismatch。

验证：

```bash
.venv/bin/python -m py_compile \
  vllm/envs.py \
  vllm/_custom_ops.py \
  vllm/v1/attention/backends/byte_v2_attn.py \
  tests/v1/attention/test_byte_v2_decode.py

.venv/bin/python -m ruff check \
  vllm/envs.py \
  vllm/_custom_ops.py \
  vllm/v1/attention/backends/byte_v2_attn.py \
  tests/v1/attention/test_byte_v2_decode.py

.venv/bin/python -m pytest tests/v1/attention/test_byte_v2_decode.py -q
.venv/bin/python -m pytest tests/v1/attention/test_byte_v2_backend.py -q
```

结果：

```text
test_byte_v2_decode.py: 15 passed
test_byte_v2_backend.py: 9 passed
ruff: All checks passed
native schema: byte_v2_paged_decode_attention 无 output 参数
```

还原后 E2E smoke：

```text
benchmarks/profiles/bytev2_p2048_d32_step27_reverted_direct_output_smoke.json
output tok/s: 29.154
elapsed: 1.098 s
sparse fallback exhausted: false
```

下一步优化方向不应继续做 output copy 级别优化，而应集中在 decode attention
stage1：

- compressed-only fast decode path 中进一步减少 K/V decode 指令数。
- page/tile loader 的 shared layout 和 WMMA load pattern。
- split-K stage1 的 page chunk 并行度、partial 写回和 reduce 开销。
- 对 fallback tile/page 做更精确的分离，避免 compressed hot path 承担 fallback
  元数据检查。

## Step 28: decode attention stage1 后续候选方案

### 目标

Step 27 之后，短期优化目标只看
`byte_v2_paged_decode_attention_gqa_wmma_split_stage1_kernel` 及其 reduce：

```text
p2048/d32 measured range:
  decode attention stage1 + reduce = 143.96 ms
```

下一阶段目标：

| 指标 | 当前 | 短期目标 | 保留门槛 |
|---|---:|---:|---:|
| p2048/d32 stage1+reduce | 143.96 ms | <= 120 ms | 至少下降 8% |
| decode-only p1024/p2048 median_us | 现有基线 | 稳定下降 | 至少下降 5% |
| p2048/d32 E2E output tok/s | 约 29.15 | 稳定上升 | 至少 +2% |
| fallback pool exhausted | false | false | 不允许回退 |

如果只改善 NCU 指标但 decode-only 和 E2E 没有对应改善，默认不保留。所有实验
必须有 env guard 或独立 kernel variant，验证无效后删除代码。

### 方向 A：减少 K/V decode 指令数

当前判断：

- Step 21 分段 profile 显示 `load/decode K/V` 曾占 stage1 约 68.6%。
- Step 24/25 已验证 K-transposed + no-fallback vectorized decoder 有收益；
  NCU stage1 duration 从历史 112.03 us 降到 86.69 us。
- 仍然存在 executed instructions 高、eligible warp 低、latency/issue 受限的问题。

#### A1：专用 compressed-only K/V tile decoder v2

只针对 hot path：

```text
status == COMPRESSED
fallback_pool == nullptr 或 fallback_tile_ids == nullptr
tile_fallback flag == 0
head_size == head_size_v == 128
q_per_kv == 4
```

实现要点：

- K/V loader 完全分离，不传 `is_value` runtime bool。
- K 使用当前已保留的 transposed physical order，V 使用 row-major order。
- 每个线程处理 2 个连续 element，共用 1 个 packed byte。
- 进一步尝试每线程处理 4 个 element：
  - `low` 使用 `uint32_t` load。
  - `packed` 使用 `uint16_t`/`uint32_t` load。
  - 一次生成 4 个 BF16 bits 后写 shared。
- tile-scope 常量预计算：
  - `base_hi = base >> 1`
  - `base_lsb = base & 1`
  - `tile_start`
  - `low_ptr`
  - `packed_ptr`

验证指标：

- NCU `smsp__sass_thread_inst_executed_op_integer` 下降。
- NCU `smsp__sass_thread_inst_executed_op_memory` 不上升。
- `load/decode K/V` early-exit profile 下降。
- decode-only p1024/p2048 至少 +5%。

回滚条件：

- 寄存器数增加导致 occupancy/eligible warp 下降，抵消指令减少。
- E2E p2048/d32 低于 baseline。

实验记录，2026-06-10：

- 尝试了两个 env-guarded decoder 变体，均只作用于
  `page_fastpath && no fallback` 的 split-K WMMA stage1：
  - `vec4`：每线程处理 4 个连续 element，使用两个 `low` pair 和两个 packed byte。
  - `pair_v2`：保持每线程 2 个 element 和原有 128-thread/tile 并行度，只 hoist
    `base_hi/base_lsb`，并尝试用 32-bit pair store 写 shared。
- correctness smoke 通过，但 decode-only p1024/p2048 未达到 +5% 保留门槛：

| seq_len | split_k | baseline median_us | pair_v2 median_us | pair_v2 delta | vec4 median_us | vec4 delta |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 16 | 1031.17 | 1033.73 | -0.25% | 1033.22 | -0.20% |
| 1024 | 64 | 1102.85 | 1101.30 | +0.14% | 1108.45 | -0.51% |
| 2048 | 16 | 1200.13 | 1197.06 | +0.26% | 1204.24 | -0.34% |
| 2048 | 64 | 1135.62 | 1131.52 | +0.36% | 1138.69 | -0.27% |

- 结论：不保留代码，已回滚 `VLLM_BYTE_V2_DECODE_VEC4_FASTPATH`、
  `VLLM_BYTE_V2_DECODE_PAIR_V2_FASTPATH`、新增 tests 和 benchmark variant。
  benchmark JSON 保留在：
  `benchmarks/profiles/bytev2_step28_a1_decode_page_fastpath_vec4.json`、
  `benchmarks/profiles/bytev2_step28_a1_pair_decode_page_fastpath_baseline.json`、
  `benchmarks/profiles/bytev2_step28_a1_decode_page_fastpath_pair_v2.json`、
  `benchmarks/profiles/bytev2_step28_a1_pair_decode_page_fastpath_vec4.json`。
- 判断：
  - `vec4` 虽然减少循环次数，但每 tile active lane 从 128 降到 64，latency/issue
    并行度损失抵消了指令减少。
  - `pair_v2` 的算术 hoist 和 32-bit shared store 过小，编译器可能已优化了大部分
    表达式，真实瓶颈不在这几个 per-element integer op。
  - A1 这种局部 decoder 微调不再优先；后续若继续减少 K/V decode 成本，应做更大
    粒度的格式/layout 改动，例如 encode-side MMA-friendly layout 或独立 CUTE/CUTLASS
    stage1 variant。

#### A2：tile descriptor hoist

当前 stage1 内每个 tile 反复计算：

```text
tile_start
low_ptr
packed_ptr
base
tile fallback id
valid_rows mask
```

实验方案：

- 每个 CTA/page/tile 开始时由少数线程把 descriptor 写入寄存器或小 shared struct。
- K/V decoder 只接收 descriptor，不再重复地址计算。
- descriptor 不跨 page 保存，避免全局 metadata workspace。

适用场景：

- `head_size=128`、每 page K/V 各 8 个 dim tile，descriptor 数量固定。
- 对 compressed hot path 优先，不处理 raw fallback。

风险：

- descriptor 写入 shared 可能引入新的 shared load/store。
- 编译器可能已经做了部分 CSE，收益不一定明显。

保留标准：

- 指令数下降并体现在 stage1 duration。
- 不能增加 shared bank conflict。

#### A3：decode LUT 小表实验

ByteV2 decode 公式中 high byte 由 `base/low/code` 得出。可以尝试 tile 内小 LUT：

```text
lut[16][2] 或 lut[32]
输入: nibble code + low_exp_lsb
输出: high byte contribution
```

目的：

- 用少量 shared/register LUT 替代每 element 的 bit arithmetic。

风险：

- LUT load 可能比整数指令更慢。
- shared LUT 可能增加 bank conflict。

执行顺序：

1. 先用 microbenchmark/NCU 做单 kernel 对照。
2. 只有 instruction 明显下降且 stage1 下降才进 E2E。

默认优先级低于 A1/A2。

#### A4：CTA 内 K/V shared double-buffer

实验目标：

- 使用两套 K/V shared buffer。
- 当前 page 的 QK WMMA 由 warp0 执行时，warp1-3 预解码下一 page 到另一套
  shared buffer。
- 下一轮 page loop 直接使用已解码的 K/V，尝试隐藏 compressed K/V load/decode
  latency。

实验限制：

- 只在 `page_fastpath && fallback_pool == nullptr && fallback_tile_ids == nullptr`
  的 compressed-only 路径启用。
- 只支持 `head_size == head_size_v == 128` 且每个 split 至少 2 个 page；否则回到
  原路径。
- 使用 env-guarded 实验开关，验证失败后删除。

实验记录，2026-06-10：

- correctness smoke 通过。
- decode-only p1024/p2048 均退化：

| seq_len | split_k | baseline median_us | double-buffer median_us | delta |
| ---: | ---: | ---: | ---: | ---: |
| 1024 | 16 | 1029.63 | 1035.26 | -0.54% |
| 1024 | 64 | 1097.71 | 1103.36 | -0.51% |
| 2048 | 16 | 1187.84 | 1240.06 | -4.21% |
| 2048 | 64 | 1123.82 | 1152.00 | -2.45% |

- 结论：不保留代码，已回滚 `VLLM_BYTE_V2_DECODE_DOUBLE_BUFFER_FASTPATH`、
  range decoder helper、double shared storage、benchmark variant 和新增测试。
  benchmark JSON 保留在：
  `benchmarks/profiles/bytev2_step28_db_decode_page_fastpath_baseline.json`、
  `benchmarks/profiles/bytev2_step28_db_decode_page_fastpath_double_buffer.json`。
- 判断：
  - warp1-3 预解码下一页降低了每个 tile decode 的并行度，decode 本身变慢。
  - double K/V shared 增加 shared footprint 和同步压力。
  - 当前 kernel 的 QK/PV/softmax barrier 很密，CTA 内手工 overlap 很难形成有效
    pipeline。
  - 如果后续还要做 pipeline，应转向独立 CUTE/CUTLASS-style stage1，用更完整的
    producer/consumer pipeline 和异步 copy 设计，而不是在当前 WMMA kernel 中局部
    插入 double-buffer。

### 方向 B：shared layout / WMMA load pattern

当前判断：

- Step 23 的简单 K row-major shared + WMMA B col-major 退化，已回滚。
- Step 25 的 K shared stride padding 退化，已回滚。
- Step 25 的临时 CUTE 混合路径污染默认 WMMA 编译形态，也已回滚。
- 因此后续不能再做“局部改 shared stride”或“在现有 WMMA kernel 里硬塞 CUTE”。

#### B1：独立 microkernel 验证 WMMA fragment swizzle

先不接入 paged decode，单独写一个小 benchmark kernel：

```text
输入: synthetic decoded K/V shared tile
操作: QK WMMA + PV WMMA
比较:
  baseline WMMA shared layout
  swizzled shared layout
  CUTLASS/CUTE-like ldmatrix layout
```

要求：

- 独立文件或 profile-only kernel，不改默认 decode kernel 模板。
- 只测 QK/PV shared load，不包含 ByteV2 decode。
- 输出 NCU 指标：
  - shared load excessive wavefronts
  - shared bank conflict
  - eligible warps
  - register count
  - kernel duration

保留标准：

- microkernel 中 QK/PV shared load 明显优于 baseline。
- 然后再迁移到 paged decode。

回滚条件：

- 只改变编译形态但无指标收益。
- 默认 WMMA kernel 二进制形态被实验模板影响。

#### B2：完整独立 CUTE/CUTLASS-style decode stage1 variant

如果 B1 证明 swizzle 有收益，再新建独立 stage1 kernel：

```text
byte_v2_paged_decode_attention_gqa_cute_split_stage1_kernel
```

设计原则：

- Q/K/V shared layout、ldmatrix copy atom、MMA atom、epilogue store 作为一个
  完整 tiled MMA design 一起定义。
- 不复用当前 WMMA shared layout。
- 通过 env opt-in，不影响默认 WMMA path 编译。
- 首版只支持 Llama-3 8B 固定形状：
  - `num_heads=32`
  - `num_kv_heads=8`
  - `head_size=head_size_v=128`
  - `q_per_kv=4`
  - `block_size=16`

和当前 WMMA kernel 的区别：

- 当前路径是在 `byte_v2_paged_decode_attention_gqa_wmma_split_stage1_kernel`
  里把 ByteV2 K/V 解码到普通 shared layout，再用 `wmma::load_matrix_sync`
  读取。局部调换代码顺序、局部 double-buffer 或临时 CUTE swizzle 已经证明收益
  不稳定或退化。
- B2 必须新建独立 kernel variant，不在当前 kernel 里继续加 `if/env` 大分支。
  目标是让 data movement、shared layout、MMA fragment 和 pipeline 从一开始就是
  一个整体设计。

首版 kernel 边界：

```text
byte_v2_paged_decode_attention_gqa_cute_split_stage1_kernel(
    query,
    kv_cache,
    block_table,
    seq_lens,
    partial_output,
    scale,
    page_size_bytes,
    num_kv_splits,
    ...
)
```

- 只接 compressed-only no-fallback path：
  `fallback_pool == nullptr && fallback_tile_ids == nullptr`。
- 不支持 raw fallback、tile fallback、非 128 head size、非 BF16。
- reduce kernel 先复用现有
  `byte_v2_paged_decode_attention_split_reduce_kernel`，避免同时改两个变量。
- host 侧用独立 env，例如 `VLLM_BYTE_V2_DECODE_CUTE_STAGE1=1`，默认关闭。

核心设计点：

1. **MMA-friendly shared layout**
   - K/V 不再解码到当前 row-major/transposed shared 后再让 WMMA 读。
   - 解码阶段直接写入适合 `ldmatrix`/MMA atom 的 swizzled shared layout。
   - 目标是减少 shared bank conflict、excessive wavefronts 和 WMMA shared load
     stall。

2. **producer/consumer pipeline**
   - producer 负责从 compressed page 读取 ByteV2 payload、解码 BF16 K/V tile。
   - consumer 负责 QK MMA、softmax/PV MMA。
   - 采用 2-stage 或 3-stage shared buffer，但必须保持每个 tile 足够 decode
     并行度；A4 的 warp1-3 预解码下一页已经证明简单减少 decode warp 数会退化。

3. **固定形状专用展开**
   - 首版直接假设 `head_size=head_size_v=128`，K/V 各 8 个 dim tile。
   - Q/K/V tile loop 可以手写展开或模板展开，减少动态索引和分支。
   - 不做通用 shape 支持，通用路径继续走当前 WMMA kernel。

4. **独立寄存器和 shared footprint 预算**
   - 新 kernel 需要先记录 ptxas register count、shared bytes、occupancy。
   - 如果 shared footprint 导致 occupancy 明显下降，必须在 decode-only 阶段回滚。

5. **ByteV2 decode 和 MMA layout 共同设计**
   - 如果 encode-side layout 可改，优先考虑把 K compressed physical order 设计成
     解码后天然落到 MMA-friendly layout。
   - 如果 encode-side layout 暂时不改，stage1 内允许做一次专用 swizzle write，
     但不能像 Step 25 那样先写普通 shared 再 shared-to-shared 重排。

执行顺序：

1. **B2a microkernel skeleton**
   - 不接真实 ByteV2 cache，只用 synthetic BF16 K/V tile。
   - 比较当前 WMMA layout 与 CUTE/CUTLASS-style swizzled layout 的 QK/PV cost。
   - 只看 shared load wavefront、bank conflict、eligible warps、register、duration。
   - 当前已完成第一版 baseline skeleton：
     - native op：`byte_v2_wmma_layout_microbench`
     - CUDA kernel：`byte_v2_wmma_layout_microbench_kernel`
     - 测试：
       `tests/v1/attention/test_byte_v2_wmma_microbench.py`
     - benchmark：
       `benchmarks/kernels/benchmark_byte_v2_wmma_microbench.py`
   - 该版本只实现当前 row-major/transposed WMMA shared layout 的 QK/PV 基线，
     不包含真实 ByteV2 解码、softmax、split-K、fallback metadata，也不接入默认
     decode path。
   - A40 baseline 结果：
     - 命令：
       `.venv/bin/python benchmarks/kernels/benchmark_byte_v2_wmma_microbench.py --num-tiles 256 --repeat-count 32 --warmup 20 --iterations 80 --output benchmarks/profiles/bytev2_b2a_wmma_microbench_baseline.json`
     - median：159.744 us/launch
     - per tile/repeat median：0.0195 us
     - p90 抖动明显，本结果只作为同机同参数后续 A/B 对照。
   - 下一步如果继续 B2a，应新增独立 variant，不在该 kernel 内传 runtime layout
     bool；host 侧按 variant 分发到不同 kernel，避免再次污染默认 WMMA 编译形态。
   - 已尝试独立 variant：K row-major shared + QK B fragment `wmma::col_major`。
     - 正确性通过，但 A40/GPU2 decode microbench 无收益：
       - `repeat_count=32`：variant 0 median 1082.368 us，variant 1 median
         1089.024 us，约慢 0.6%。
       - `repeat_count=128`：variant 0 median 1461.248 us，variant 1 median
         1519.616 us，约慢 4.0%。
     - 结论：该 variant 不保留，源码已回滚。
     - 这进一步说明单纯改变 K shared row/col-major 并不能解决当前瓶颈；后续
       B2a 如果继续，需要真正的 ldmatrix-friendly fragment swizzle 或 inline MMA，
       而不是继续尝试 `wmma::load_matrix_sync` 的布局小改。

2. **B2b decode-to-swizzled-shared**
   - 接真实 compressed page，但只做 decode K/V 到 swizzled shared，再做最小 QK MMA
     correctness。
   - 不接 softmax/PV/reduce，先确认 decode+ldmatrix layout 不退化。
   - 当前已完成 B2b baseline：
     - native op：`byte_v2_decode_page_wmma_microbench`
     - CUDA kernel：`byte_v2_decode_page_wmma_microbench_kernel`
     - 测试：
       `tests/v1/attention/test_byte_v2_wmma_microbench.py::test_byte_v2_decode_page_wmma_microbench_matches_reference_cuda`
     - benchmark：
       `benchmarks/kernels/benchmark_byte_v2_decode_page_wmma_microbench.py`
   - 该 baseline 读取真实 ByteV2 compressed page，并把 K/V 解码到当前
     row-major/transposed shared layout，然后跑和 B2a 相同的 QK/PV math。
     还不是 swizzled shared，也不接生产 decode。
   - A40/GPU2 结果：
     - synthetic BF16 shared baseline：
       `benchmarks/profiles/bytev2_b2b_synthetic_wmma_baseline_gpu2.json`
       - `num_tiles=256, repeat_count=32`
       - median：1064.960 us
       - per tile/repeat：0.1300 us
     - compressed page decode + WMMA，`num_kv_heads=1`：
       `benchmarks/profiles/bytev2_b2b_decode_page_wmma_gpu2.json`
       - median：1239.040 us
       - per page/repeat：0.15125 us
       - 相对 synthetic baseline 约 +16.3%
     - compressed page decode + WMMA，`num_kv_heads=8, kv_head=0`：
       `benchmarks/profiles/bytev2_b2b_decode_page_wmma_gqa8_gpu2.json`
       - median：1243.648 us
       - per page/repeat：0.15181 us
       - 和 `num_kv_heads=1` 基本一致，说明该 microbench 主要量到单 KV head 的
         ByteV2 tile decode + WMMA 成本，page 总大小不是主要变量。
   - 结论：
     - 当前 ByteV2 compressed tile decode 在最小 QK/PV 路径上约带来 16% 额外成本。
     - 下一步不能只改 page layout 参数；应在这个 B2b op 上新增真正的
       decode-to-ldmatrix-friendly shared variant，或者 inline MMA variant，并要求
       per page/repeat 低于 0.151 us，最好逼近 0.130 us synthetic baseline。
   - 已尝试 B2b 独立 variant：K row-major decode 到 shared + QK B fragment
     `wmma::col_major`。
     - 正确性通过，但 A40/GPU2 microbench 无收益：
       - `repeat_count=32`：baseline median 1233.920 us，variant median
         1247.744 us，约慢 1.1%。
       - `repeat_count=128`：baseline median 2083.840 us，variant median
         2131.968 us，约慢 2.3%。
     - 结论：该 variant 不保留，源码已回滚。
     - 这和 B2a synthetic 结果一致：`wmma::load_matrix_sync` 的 row/col-major
       组合无法解决 shared/issue bottleneck；后续若继续 B2b，应实现真正的
       ldmatrix/inline MMA path，或者先用 NCU 确认当前 decode 指令与 shared load
       stall 的占比。
   - 已完成 NCU 对照 profile：
     - synthetic BF16 shared baseline：
       `benchmarks/profiles/bytev2_b2a_ncu_wmma_synthetic_baseline.csv`
     - compressed page decode + WMMA baseline：
       `benchmarks/profiles/bytev2_b2b_ncu_decode_page_wmma_baseline.csv`
       和
       `benchmarks/profiles/bytev2_b2b_ncu_decode_page_wmma_stalls.csv`
     - 同为 `num_pages/tiles=256, repeat_count=32`，关键指标：

       | 指标 | synthetic BF16 | compressed page | 结论 |
       |---|---:|---:|---|
       | `gpu__time_duration.sum` | 184.224 us | 393.408 us | compressed path 约 2.1x |
       | integer thread inst | 471.335M | 1230.340M | ByteV2 decode integer work 是主要新增成本 |
       | memory thread inst | 83.886M | 174.129M | compressed payload load/decode 明显增加 memory 指令 |
       | HMMA inst | 262144 | 262144 | tensor core work 相同 |
       | shared LD wavefronts | 5.833M | 5.833M | WMMA/shared-load 形态相同 |
       | shared LD bank conflicts | 4.194M | 4.194M | bank conflict 不是 compressed path 相对退化来源 |
       | LDSM inst | 0 | 0 | 当前路径没有使用 ldmatrix |
       | register/thread | 40.29 | 40.00 | register footprint 基本相同 |
       | long scoreboard stall | 6.95% | 30.69% | compressed payload/global load dependency 明显更重 |
       | barrier stall | 47.09% | 21.71% | synthetic 更受同步占比影响，compressed 更受 load/decode 影响 |

     - 结论：
       - B2b 当前相对 synthetic 的差距不是 HMMA 数量、shared wavefront 或
         shared bank conflict 导致的，而是 ByteV2 tile decode 的 integer/memory
         指令和 load dependency。
       - 下一步优先级应从“改 WMMA row/col-major layout”转为“减少/流水化 ByteV2
         decode/load 指令”。真正的 ldmatrix/inline MMA 仍可能有价值，但只有在
         decode/load 成本降下来后才可能体现。
       - 近期小实验应优先尝试：
         1. 用更宽的 packed load 合并 low/code 读取，减少 byte/halfword load 指令。
         2. 把 K/V decoder 与 WMMA loop 分离成更少分支、更强 unroll 的固定 128 维
            专用 helper。
         3. 对 compressed payload 做 encode-side MMA-friendly/interleaved 布局，
            让 decode 写 shared 的地址计算更少，并减少 global load scoreboard。
   - 已尝试 B2b `vec4 decode` variant：
     - 实现思路：每个线程一次处理 4 个 tile 元素，两个 packed-code byte 合并成
       一次 `u16` 读取，减少循环次数、packed byte load 和部分地址计算。
     - 正确性通过，但 A40/GPU2 microbench 无收益：
       - `repeat_count=32`：baseline median 1241.088 us，variant median
         1244.160 us，约慢 0.25%。
       - `repeat_count=128`：baseline median 2083.328 us，variant median
         2133.504 us，约慢 2.4%。
     - 结论：该 variant 不保留，源码已回滚。
     - 推断：简单把每线程粒度从 2 元素扩大到 4 元素，会降低参与 decode 的线程数
       并增加每线程寄存器/依赖链；省掉的少量 packed byte load 不足以抵消并行度和
       dependency 退化。下一步不应继续做“更粗粒度 per-thread decode”，而应考虑：
       - encode-side payload 改为更连续的 lane-friendly layout；
       - 或者 warp-level cooperative decode，把同一 tile 的 base/packed/low 读取和
         写 shared 的地址计算在 warp 内重排，而不是减少 active lanes。
   - 已尝试 B2b `tile base prefetch` variant：
     - 实现思路：每个 CTA 先把当前 KV head 的 K/V 16 个 tile base byte 读入
       shared memory，decode helper 从 shared 读 base，避免每个线程在每个 tile
       上重复从 compressed page 读取同一个 base byte。
     - 正确性通过，但 A40/GPU2 microbench 收益太小，低于保留门槛：
       - `repeat_count=32`：baseline median 1236.992 us，variant median
         1225.728 us，约快 0.9%。
       - `repeat_count=128`：baseline median 2074.624 us，variant median
         2045.440 us，约快 1.4%，且 variant 有一次明显 max outlier。
     - 记录文件：
       `benchmarks/profiles/bytev2_b2b_base_prefetch_v0_r32_gpu2.json`、
       `benchmarks/profiles/bytev2_b2b_base_prefetch_v1_r32_gpu2.json`、
       `benchmarks/profiles/bytev2_b2b_base_prefetch_v0_r128_gpu2.json`、
       `benchmarks/profiles/bytev2_b2b_base_prefetch_v1_r128_gpu2.json`。
     - 结论：该 variant 不保留，源码已回滚。base byte 的重复 global load 不是
       当前主要瓶颈；下一步不应单独优化 base metadata，而应转向 payload
       `low/code` load、decode 指令链和 encode-side lane-friendly/interleaved
       layout。

   - 已实现第一版独立 `byte_v2_paged_decode_attention_gqa_cute_split_stage1_kernel`
     实验路径：
     - host 侧通过 `VLLM_BYTE_V2_DECODE_CUTE_STAGE1=1` 启用，默认关闭。
     - 只在 split-K、`VLLM_BYTE_V2_DECODE_PAGE_FASTPATH=1`、compressed-only、
       无 sparse/tile fallback、`num_heads=32`、`num_kv_heads=8`、
       `head_size=head_size_v=128`、`q_per_kv=4` 时分发到独立 kernel。
     - reduce 复用现有 `byte_v2_paged_decode_attention_split_reduce_kernel`。
     - 该版本避免污染默认 WMMA split-stage1 编译形态，但仍是固定形状
       no-fallback stage1；尚未真正完成 ldmatrix/inline MMA、swizzled shared
       copy atom 和 producer/consumer pipeline。
   - Correctness：
     - `tests/v1/attention/test_byte_v2_decode.py::test_byte_v2_paged_decode_attention_op_cute_stage1_cuda`
       通过。
     - ByteV2 attention 小测试集 56 项通过。
     - decode benchmark raw reference：p1024/split64/fallback0 `max_abs_diff=0.0`。
   - Decode-only A40/GPU2 结果：

     | seq_len | split_k | baseline median_us | cute-stage1 median_us | delta |
     |---:|---:|---:|---:|---:|
     | 1024 | 16 | 1042.94 | 1016.80 | +2.51% |
     | 2048 | 16 | 1188.86 | 1200.64 | -0.99% |
     | 1024 | 64 | 1108.48 | 1006.08 | +9.24% |
     | 2048 | 64 | 1157.63 | 1114.11 | +3.76% |
     | 4096 | 64 | 1217.54 | 1199.10 | +1.51% |

   - 记录文件：
     `benchmarks/profiles/bytev2_cute_stage1_baseline_p1024_s16_fb0.json`、
     `benchmarks/profiles/bytev2_cute_stage1_v1_p1024_s16_fb0.json`、
     `benchmarks/profiles/bytev2_cute_stage1_baseline_p2048_s16_fb0.json`、
     `benchmarks/profiles/bytev2_cute_stage1_v1_p2048_s16_fb0.json`、
     `benchmarks/profiles/bytev2_cute_stage1_baseline_p1024_p2048_s64_fb0.json`、
     `benchmarks/profiles/bytev2_cute_stage1_v1_p1024_p2048_s64_fb0.json`、
     `benchmarks/profiles/bytev2_cute_stage1_baseline_p4096_s64_fb0.json`、
     `benchmarks/profiles/bytev2_cute_stage1_v1_p4096_s64_fb0.json`。
   - 当前判断：
     - 该独立 kernel 在 split64 下有稳定正向趋势，split16 长 context 有小幅回退。
     - 因为默认关闭、分发条件严格、且不影响 fallback 路径，暂时可作为后续
       ldmatrix/producer-consumer 实验的独立载体保留。
     - 不能把它视为最终 CUTE/CUTLASS-style 完整版本；下一步必须在这个独立
       symbol 内继续实现真正的 swizzled shared + ldmatrix/inline MMA，并用 NCU
       确认 `LDSM`/shared wavefront/eligible warp 指标改善。

   - 后续小步实验：warp-parallel softmax / `p_shared` fill：
     - 背景：NCU 显示第一版独立 stage1 的 `No Eligible` 约 66.56%，
       `Eligible Warps Per Scheduler` 约 0.49，并且存在明显 CTA barrier stall。
       其中 `tid == 0` 串行清零 `p_shared`、计算 4 个 Q row 的 online softmax，
       会让其他 warp 在 softmax 后的 barrier 等待。
     - 改动：
       - 每个 CTA 内 4 个 warp 分别处理 `q_per_kv=4` 的一个 Q row。
       - 用 warp reduction 计算 `tile_max` 和 `tile_denom`。
       - 用 shared `running_max_shared/denom_shared` 保存 online softmax 状态。
       - `p_shared` 清零由 128 threads 并行完成。
     - Correctness：
       - `test_byte_v2_paged_decode_attention_op_cute_stage1_cuda` 通过。
       - ByteV2 attention 小测试集 56 项通过。
       - decode benchmark p1024/split64/fallback0 raw reference：
         `max_abs_diff=0.0`。
     - 串行 benchmark，A40/GPU2：

       | seq_len | split_k | baseline median_us | cute+warp-softmax median_us | delta |
       |---:|---:|---:|---:|---:|
       | 1024 | 16 | 1041.92 | 1003.52 | +3.69% |
       | 2048 | 16 | 1190.40 | 1128.96 | +5.16% |
       | 1024 | 64 | 1112.58 | 994.82 | +10.58% |
       | 2048 | 64 | 1137.66 | 1112.06 | +2.25% |
       | 4096 | 64 | 1220.61 | 1160.19 | +4.95% |

     - NCU，p1024/split64/fallback0：

       | 指标 | cute-stage1 v1 | warp-softmax | 变化 |
       |---|---:|---:|---:|
       | duration | 49.248 us | 40.224 us | +18.3% |
       | executed instructions | 4.187M | 3.559M | -15.0% |
       | registers/thread | 62 | 56 | -9.7% |
       | static shared/block | 14,896 B | 14,928 B | 基本不变 |
       | eligible warps/scheduler | 0.49 | 0.54 | 小幅改善 |
       | no eligible | 66.56% | 64.99% | 小幅改善 |
       | active threads/warp | 25.42 | 30.89 | 明显改善 |

     - 记录文件：
       `benchmarks/profiles/bytev2_cute_stage1_warp_softmax_seq_baseline_p1024_s16_fb0.json`、
       `benchmarks/profiles/bytev2_cute_stage1_warp_softmax_seq_v1_p1024_s16_fb0.json`、
       `benchmarks/profiles/bytev2_cute_stage1_warp_softmax_seq_baseline_p2048_s16_fb0.json`、
       `benchmarks/profiles/bytev2_cute_stage1_warp_softmax_seq_v1_p2048_s16_fb0.json`、
       `benchmarks/profiles/bytev2_cute_stage1_warp_softmax_seq_baseline_p1024_s64_fb0.json`、
       `benchmarks/profiles/bytev2_cute_stage1_warp_softmax_seq_v1_p1024_s64_fb0.json`、
       `benchmarks/profiles/bytev2_cute_stage1_warp_softmax_seq_baseline_p2048_s64_fb0.json`、
       `benchmarks/profiles/bytev2_cute_stage1_warp_softmax_seq_v1_p2048_s64_fb0.json`、
       `benchmarks/profiles/bytev2_cute_stage1_warp_softmax_seq_baseline_p4096_s64_fb0.json`、
       `benchmarks/profiles/bytev2_cute_stage1_warp_softmax_seq_v1_p4096_s64_fb0.json`、
       `benchmarks/profiles/bytev2_cute_stage1_warp_softmax_ncu_p1024_s64_fb0.csv`。
     - 当前判断：
       - 该改动保留在独立 `VLLM_BYTE_V2_DECODE_CUTE_STAGE1=1` 路径中。
       - 收益来自减少串行 softmax/p_shared fill 和降低寄存器/指令数，不是来自
         ldmatrix；NCU 仍显示 L1TEX scoreboard 是主要 stall。
       - 下一步再做 ldmatrix/inline MMA 时，应继续保持独立 symbol，先替换 QK
         或 PV 中的一段，并用 NCU 确认 `LDSM` 指标出现且 duration 下降。

   - 已尝试 full-page packed shared-store decode，不保留：
     - 背景：warp-softmax 后 NCU 的主要瓶颈转为 `L1TEX scoreboard`。尝试为
       `valid_rows == 16` 的 full compressed page 增加专用 K/V decode helper，
       用 aligned 16-bit load 读取 low bytes，用 32-bit store 一次写两个 BF16
       到 shared，并去掉每 pair 的 valid-row 分支。
     - Correctness：
       - `test_byte_v2_paged_decode_attention_op_cute_stage1_cuda` 通过。
       - 默认 WMMA/split-K/page-fastpath 相关 decode 单测 3 项通过。
     - 串行 benchmark，A40/GPU2，fallback=0：

       | seq_len | split_k | 默认 baseline median_us | 实验路径 median_us | 对上一轮保留路径 |
       |---:|---:|---:|---:|---:|
       | 1024 | 64 | 1127.42 | 1015.81 | 慢于 994.82 |
       | 2048 | 64 | 1203.20 | 1152.00 | 慢于 1112.06 |
       | 1024 | 16 | 1061.89 | 1025.02 | 慢于 1003.52 |

     - 结论：
       - 该实验虽然仍快于默认 baseline，但没有超过已经保留的 warp-softmax
         independent path，因此已回滚，不保留代码。
       - 可能原因是 packed 32-bit shared store 和 `reinterpret_cast` 没有减少
         真正关键的 global dependency，反而引入了更差的 store/alias/codegen。
       - 后续不再单独做“只换 2 元素 packed store”的局部 decode helper；
         如果继续处理 Load/Decode K/V，应优先做 payload layout 或真正的
         per-tile producer/consumer 管线，而不是只改 store 粒度。

   - 已尝试 tile-base warp broadcast，不保留：
     - 背景：no-fallback K/V decode helper 中每个线程都会读取同一个 tile
       `base` byte。尝试改成每个 warp 只由 lane0 读取，再用 `__shfl_sync`
       广播，减少重复 global byte load。
     - Correctness：
       - `test_byte_v2_paged_decode_attention_op_cute_stage1_cuda` 通过。
       - 默认 WMMA/split-K/page-fastpath 相关 decode 单测 3 项通过。
     - 串行 benchmark，A40/GPU2，fallback=0：

       | seq_len | split_k | 默认 baseline median_us | 实验路径 median_us | 对上一轮保留路径 |
       |---:|---:|---:|---:|---:|
       | 1024 | 64 | 1108.99 | 1030.14 | 慢于 994.82 |
       | 2048 | 64 | 1140.74 | 1152.00 | 慢于 1112.06 |
       | 1024 | 16 | 1073.15 | 1035.26 | 慢于 1003.52 |

     - 结论：
       - 对默认 baseline 的个别配置有轻微正向，但幅度接近噪声，并且没有改善
         independent env path。
       - 推测编译器或 L1 已经较好处理 uniform `base` load；额外 `shfl`
         没有降低关键路径，反而增加指令依赖。
       - 该实验已回滚，不保留代码。

   - 已尝试 K-first / V-late page schedule，不保留：
     - 背景：当前 independent stage1 对每个 page 先解压 K 和 V，然后再执行
       QK、online softmax 和 PV。尝试改成先只解压 K，完成 QK/softmax/acc
       rescale 后，再解压 V 做 PV，目标是让 QK 更早开始，避免 QK 前等待无关
       V load/decode。
     - Correctness：
       - `test_byte_v2_paged_decode_attention_op_cute_stage1_cuda` 通过。
       - 默认 WMMA/split-K/page-fastpath 相关 decode 单测 3 项通过。
     - 串行 benchmark，A40/GPU2，`VLLM_BYTE_V2_DECODE_CUTE_STAGE1=1`，
       fallback=0：

       | seq_len | split_k | K-first/V-late median_us | 上一轮保留路径 |
       |---:|---:|---:|---:|
       | 1024 | 64 | 1013.76 | 994.82 |
       | 2048 | 64 | 1117.18 | 1112.06 |
       | 1024 | 16 | 1014.78 | 1003.52 |

     - 结论：
       - 没有稳定超过已保留的 warp-softmax independent path，已回滚。
       - 说明单纯把 V decode 延后不能解决 issue/scoreboard bottleneck；同一 CTA
         内仍然串行执行 K decode、QK、V decode、PV，没有形成真正的
         producer/consumer overlap。
       - 后续如果继续做 schedule/pipeline，应改成双 buffer + warp specialization
         或者跨 page producer/consumer，而不是仅调整语句顺序。

   - 已尝试 warp-specialized double-buffer page prefetch，不保留：
     - 设计：
       - 新增临时 env `VLLM_BYTE_V2_DECODE_CUTE_STAGE1_WS=1`。
       - 新增独立 WS stage1 kernel，不替换默认路径。
       - 使用两套 K/V shared buffer。
       - 第一页由全 CTA 同步解压。
       - 后续每页计算时，warp0 执行当前页 QK；warp1-3 使用 worker-subset
         decode helper 把下一页 K/V 解压到另一个 shared buffer。
       - QK 和 producer prefetch 后用 CTA barrier 汇合，再执行 warp-parallel
         softmax、acc rescale 和 PV。
     - Correctness：
       - 原 `test_byte_v2_paged_decode_attention_op_cute_stage1_cuda` 通过。
       - WS decode benchmark raw reference `max_abs_diff=0.0`。
     - 串行 benchmark，A40/GPU2，fallback=0：

       | seq_len | split_k | 保留路径 median_us | WS median_us | 结论 |
       |---:|---:|---:|---:|---|
       | 1024 | 64 | 1011.71 | 1025.02 | 退化 |
       | 2048 | 64 | 1103.87 | 1118.21 | 退化 |
       | 1024 | 16 | 1038.34 | 1017.86 | 单点略快但低于历史最好 |
       | 2048 | 16 | 1153.02 | 1203.20 | 明显退化 |

     - 结论：
       - 该实验已回滚，不保留 kernel、worker-subset helper 和 env 开关。
       - 主要原因是当前 QK 阶段太短，warp1-3 解压下一页完整 K/V 通常比 warp0
         的 QK 更慢，barrier 仍然等待 producer，实际 overlap 不足。
       - 双 K/V shared buffer 增加 shared footprint，可能降低 occupancy；同时
         producer 只有 96 个线程，decode throughput 低于原全 CTA decode。
       - 当前 PV/acc 结构需要所有线程参与 per-dim accumulator 更新，无法简单让
         producer warp 在 PV 阶段继续预取下一页 V。
       - 后续如果继续 overlap 方向，需要更彻底地改 PV accumulator ownership，
         或转向跨 CTA/page 的 producer-consumer，而不是在当前 CTA 内只让
         warp1-3 预取下一页。

3. **B2c full split-stage1**
   - 接完整 QK、online softmax、PV、partial output。
   - reduce 复用当前 kernel。
   - 跑 decode-only p1024/p2048/p4096，再跑 p2048/d32 E2E。

4. **B2d E2E guarded rollout**
   - 只在 `kv_cache_dtype=byte_v2`、compressed-only、Llama-3 8B 固定形状下启用。
   - 其他形状和 fallback path 自动回当前 WMMA kernel。

最小验证：

```bash
.venv/bin/python -m pytest \
  tests/v1/attention/test_byte_v2_decode.py::test_byte_v2_paged_decode_attention_op_page_fastpath_split_k_cuda \
  tests/v1/attention/test_byte_v2_backend.py -q

CUDA_VISIBLE_DEVICES=<gpu> \
.venv/bin/python benchmarks/kernels/benchmark_byte_v2_decode_kernel.py \
  --batch-size 1 \
  --seq-len 1024,2048,4096 \
  --num-heads 32 \
  --num-kv-heads 8 \
  --head-size 128 \
  --head-size-v 128 \
  --fallback-ratio 0 \
  --split-k 16,64,128 \
  --variant page_fastpath
```

NCU 必看指标：

- `smsp__sass_thread_inst_executed_op_memory`
- `smsp__sass_thread_inst_executed_op_integer`
- `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum`
- shared load excessive wavefronts
- global load sectors / excessive sectors
- register count、shared memory bytes、achieved occupancy、eligible warps
- stage1 duration 和 reduce duration

风险：

- 开发成本高。
- 如果 register/shared footprint 太大，E2E 可能无收益。
- 新 kernel 可能只改善 QK/PV shared load，但 ByteV2 decode 或 softmax 仍然主导；
  这种情况下不能保留。
- CUTE/CUTLASS 头文件、模板实例和当前 extension build 可能显著增加编译时间；
  实验代码必须保持独立，不能污染默认 WMMA path 编译形态。

保留标准：

- decode-only p1024/p2048 >= +10%。
- p2048/d32 E2E >= +5%。
- 若只在 microkernel 里有收益，但 full stage1/E2E 无收益，则不保留。
- correctness 误差不超过当前 ByteV2 compressed-only decode path。

### 方向 C：split-K partial/reduce 开销

当前判断：

- split-K 解决长 context 并行度，但引入：
  - stage1 写 FP32 partial output + LSE。
  - reduce kernel 再读 partial workspace。
  - 多一次 kernel launch。
- Step 27 profile 中 reduce 约 8.79 ms，小于 stage1，但不是 0。

#### C1：persistent partial workspace

当前 partial workspace 在 native op 内部创建：

```text
torch::stable::empty({partial_numel}, Float, ...)
```

实验方案：

- backend/model runner 为 ByteV2 decode 持久分配 partial workspace。
- native op 接收 workspace tensor。
- cudagraph capture 时 workspace 地址稳定。

预期收益：

- 减少 per-token/layer allocator 开销。
- 让后续 reduce fusion 更容易。

风险：

- workspace size 依赖 `num_decode_tokens * num_heads * num_kv_splits * (head_size_v+1)`。
- vLLM runner 需要按最大 capture shape 管理 workspace。

保留标准：

- E2E p2048/d32 >= +2%。
- cudagraph capture/replay 正常。
- 不增加峰值显存到影响 KV capacity。

实验记录，2026-06-10：

- 已实现可选 external partial workspace：
  - native `byte_v2_paged_decode_attention` 增加 optional
    `partial_workspace` 参数。
  - Python op wrapper 和 ByteV2 attention backend 增加传参。
  - backend 在 `VLLM_BYTE_V2_PERSISTENT_PARTIAL_WORKSPACE=1` 时复用
    vLLM workspace manager；workspace manager 未初始化时复用 backend 私有 tensor。
  - E2E benchmark JSON 记录
    `VLLM_BYTE_V2_PERSISTENT_PARTIAL_WORKSPACE`。
- 正确性/构建验证：
  - `_C_stable_libtorch` rebuild 通过。
  - `test_byte_v2_native_paged_decode_external_partial_workspace_cuda` 通过。
  - `test_byte_v2_decode_partial_workspace_reuses_buffer` 通过。
  - ByteV2 attention 小测试集 58 项通过。
- decode-only 有小幅正向信号，且 `max_abs_diff=0.0`：

| seq_len | split_k | baseline median_us | external workspace median_us | delta |
| ---: | ---: | ---: | ---: | ---: |
| 1024 | 64 | 1020.93 | 1001.47 | +1.94% |
| 2048 | 64 | 1120.26 | 1099.78 | +1.86% |
| 1024 | 16 | 1013.76 | 1002.50 | +1.12% |

- E2E 未达到保留标准：

| prompt_len | decode_len | workspace off | workspace on | delta |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 16 | 30.120 tok/s | 30.125 tok/s | +0.016% |
| 2048 | 32 | 29.528 tok/s | 29.512 tok/s | -0.055% |

- 两组 E2E 均完成 CUDA graph capture/replay，fallback pool 未耗尽。
- 结论：C1 不进入默认主路径，`VLLM_BYTE_V2_PERSISTENT_PARTIAL_WORKSPACE`
  默认改为 `0`。保留 optional 代码只作为后续 C2/C3 reduce workspace 实验基础；
  若后续实验也不能利用该接口取得 E2E 收益，应删除这一可选路径。
- 结果文件：
  `benchmarks/profiles/bytev2_persistent_partial_workspace_base_p1024_s64_fb0.json`、
  `benchmarks/profiles/bytev2_persistent_partial_workspace_ext_p1024_s64_fb0.json`、
  `benchmarks/profiles/bytev2_persistent_partial_workspace_base_p2048_s64_fb0.json`、
  `benchmarks/profiles/bytev2_persistent_partial_workspace_ext_p2048_s64_fb0.json`、
  `benchmarks/profiles/bytev2_persistent_partial_workspace_base_p1024_s16_fb0.json`、
  `benchmarks/profiles/bytev2_persistent_partial_workspace_ext_p1024_s16_fb0.json`、
  `benchmarks/profiles/bytev2_e2e_workspace_off_p512_d16.json`、
  `benchmarks/profiles/bytev2_e2e_workspace_on_p512_d16.json`、
  `benchmarks/profiles/bytev2_e2e_workspace_off_p2048_d32.json`、
  `benchmarks/profiles/bytev2_e2e_workspace_on_p2048_d32.json`。

#### C2：stage1 partial output 压缩/半精度化实验

当前 partial accumulator 以 FP32 写出：

```text
partial_out[head_size_v] + partial_lse
```

实验方案：

- partial output 用 BF16 写出，LSE 仍 FP32。
- reduce 时读 BF16 partial 并转换 FP32。
- 或者对 partial output 做 per-row scale + BF16。

预期收益：

- partial global write/read 带宽约减半。

风险：

- 数值误差可能影响 logits。
- reduce 需要额外转换，短 context 可能无收益。

验证：

- decode correctness diff 与 raw/当前 ByteV2 对齐。
- E2E 输出一致性 smoke。
- 仅对 `num_kv_splits >= 16` 启用。

保留标准：

- p2048/p4096 decode-only 有收益。
- 输出误差在现有 ByteV2 lossless 路径允许范围内。

实验记录，2026-06-10：

- 尝试实现：
  - split-stage1 增加 env-gated BF16 partial value buffer。
  - partial value 以 BF16 写出，LSE 保持 FP32。
  - reduce kernel 读取 BF16 partial 后转 FP32 参与最终合并。
  - decode-only benchmark 增加 `--partial-bf16`。
- correctness smoke：
  - 新增 BF16 partial native split-K test 通过。
  - `max_abs_diff` 在 fallback=0 的 decode-only 场景仍为 `0.0`。
- decode-only A/B，A40/GPU2，`page_fastpath`，`split_k=64/128`：

| seq_len | split_k | fallback_ratio | FP32 partial median_us | BF16 partial median_us | speedup |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 64 | 0 | 1120.26 | 1113.09 | +0.64% |
| 1024 | 64 | 0.03 | 1124.35 | 1122.82 | +0.14% |
| 2048 | 64 | 0 | 1137.15 | 1147.39 | -0.89% |
| 2048 | 64 | 0.03 | 1171.46 | 1175.55 | -0.35% |
| 4096 | 64 | 0 | 1212.93 | 1224.70 | -0.96% |
| 4096 | 128 | 0 | 1217.54 | 1238.02 | -1.65% |

- 结论：不满足保留标准，C2 代码已删除，不进入默认路径，也不保留
  hidden env。原因判断：
  - partial workspace 带宽不是当前 decode-only 主瓶颈。
  - BF16 value 写出/读回需要额外转换，并且 values 与 LSE 拆成两个 buffer
    后增加地址计算和指针压力。
  - p4096/s128 退化说明 split-K partial/reduce 的瓶颈更可能在 kernel
    launch、softmax/reduce 指令和整体 stage1 并行结构，而不是 partial value
    字节数。
- 保留结果文件：
  `benchmarks/profiles/bytev2_partial_bf16_baseline_p1024_p2048_s64.json`、
  `benchmarks/profiles/bytev2_partial_bf16_p1024_p2048_s64.json`、
  `benchmarks/profiles/bytev2_partial_bf16_baseline_p4096_s64_s128.json`、
  `benchmarks/profiles/bytev2_partial_bf16_p4096_s64_s128.json`。

#### C3：reduce fusion / two-level reduce

实验方案：

- 当 `num_kv_splits <= 16` 时，在 stage1 中把同一 head 的多个 page chunk 映射到
  同一 cooperative group，尝试 CTA cluster 或 block-level two-pass reduce。
- 或新增 stage1b：每个 request/head 先把多个 split reduce 成较少 partial，再由
  final reduce 合并。

现实约束：

- CUDA block 间同步不可用，单 kernel 完成全 split reduce 很难。
- 多一个 stage1b kernel 可能不比现有 reduce 快。

优先级：

- 低于 C1。
- 只在 p4096/p8192 长 context 目标下尝试。

保留标准：

- reduce kernel time 下降，并且总 stage1+reduce 下降。

实验记录，2026-06-10：

- 实现 `byte_v2_paged_decode_attention_split_reduce_parallel_kernel`：
  - 保持 stage1 和 partial workspace 格式不变。
  - 只把 reduce kernel 中的 LSE max/denom 从 `tid==0` 串行循环改成 CTA
    内并行规约。
  - value accumulation 仍保持每个 output dim 由一个 thread 串行遍历 split。
- 增加 `VLLM_BYTE_V2_DECODE_PARALLEL_REDUCE` 三态控制：
  - unset / `-1`：native heuristic，`num_kv_splits >= 64` 时启用。
  - `0`：强制旧 serial-LSE reduce。
  - `1`：强制 parallel-LSE reduce。
- decode-only，A40/GPU2，fallback=0：

| seq_len | split_k | serial reduce median_us | parallel reduce median_us | speedup |
| ---: | ---: | ---: | ---: | ---: |
| 1024 | 64 | 1113.60 | 1100.80 | +1.16% |
| 1024 | 128 | 1109.50 | 1106.43 | +0.28% |
| 2048 | 64 | 1131.52 | 1125.38 | +0.55% |
| 2048 | 128 | 1159.17 | 1139.71 | +1.71% |
| 4096 | 64 | 1209.34 | 1198.08 | +0.94% |
| 4096 | 128 | 1219.58 | 1201.15 | +1.53% |

- fallback=0.03 下也没有稳定退化：

| seq_len | split_k | serial reduce median_us | parallel reduce median_us | speedup |
| ---: | ---: | ---: | ---: | ---: |
| 1024 | 128 | 1125.89 | 1121.28 | +0.41% |
| 2048 | 64 | 1174.53 | 1167.36 | +0.61% |
| 2048 | 128 | 1188.86 | 1177.60 | +0.96% |

注：p1024/s64/fallback=0.03 的 serial reduce run 出现明显 outlier
`1726.46 us`，不作为稳定 speedup 依据。

- short split 验证：
  - `split_k=16` 强制 parallel reduce 会退化。
  - 因此 auto heuristic 只在 `num_kv_splits >= 64` 启用。
  - `split_k=16` auto 不进入新 kernel，短场景保持旧路径。
- E2E：

| prompt_len | decode_len | serial reduce | parallel reduce | delta |
| ---: | ---: | ---: | ---: | ---: |
| 2048 | 32 | 29.433 tok/s | 29.584 tok/s | +0.51% |

- 结论：保留 C3 parallel-LSE reduce kernel，并默认使用 heuristic 启用。
  它不是主要瓶颈的根治，但对 `split_k>=64` 的长 context decode 有稳定小收益；
  对短 split 通过 heuristic 避免退化。
- 结果文件：
  `benchmarks/profiles/bytev2_parallel_reduce_baseline_p1024_p2048_p4096.json`、
  `benchmarks/profiles/bytev2_parallel_reduce_p1024_p2048_p4096.json`、
  `benchmarks/profiles/bytev2_parallel_reduce_baseline_fb003_p1024_p2048.json`、
  `benchmarks/profiles/bytev2_parallel_reduce_fb003_p1024_p2048.json`、
  `benchmarks/profiles/bytev2_parallel_reduce_heuristic_off_s16_short.json`、
  `benchmarks/profiles/bytev2_parallel_reduce_heuristic_auto_s16_short.json`、
  `benchmarks/profiles/bytev2_parallel_reduce_heuristic_off_s64_long.json`、
  `benchmarks/profiles/bytev2_parallel_reduce_heuristic_auto_s64_long.json`、
  `benchmarks/profiles/bytev2_parallel_reduce_e2e_off_p2048_d32.json`、
  `benchmarks/profiles/bytev2_parallel_reduce_e2e_on_p2048_d32.json`。

#### C4：split-K heuristic 继续 autotune

当前已保留部分 heuristic：

```text
16 <= pages < 96   -> split 16
96 <= pages < 128  -> split 32
128 <= pages < 256 -> split 64
pages >= 256       -> split 128
```

后续实验：

- 加入 `active_head_groups`、batch size、fallback tile ratio。
- 对 `num_decode_tokens > 1` 降低 split，避免过度并行导致 partial workspace 过大。
- 对 fallback tile ratio 高的 page chunk 降低 split，因为 raw/tile fallback 会降低
  compressed fastpath 效率。

保留标准：

- p1024/p2048/p4096 decode-only sweep 中没有明显退化点。
- E2E 至少一个目标区间提升，其他区间回退 <= 2%。

实验记录，2026-06-10：

- 目的：重新校准默认 split-K，而不是改 kernel 结构。此前 decode-only 显示
  `p512` 的旧默认 `split8` 和 `p1536` 的旧默认 `split16` 都偏低；但 E2E
  显示 `p2048` 不能按 decode-only 单点结果从 `split64` 降到 `split32`。
- 实现：
  - native `byte_v2_paged_decode_attention()` 与 Python backend
    `_decode_num_kv_splits()` 同步更新。
  - `VLLM_BYTE_V2_DECODE_SPLIT_K > 0` 仍可强制覆盖，用于实验和回退。
  - `active_head_groups` cap 保持不变：`>=64` cap 到 8，`>=32` cap 到 16。
- 新默认规则：

```text
num_pages >= 256 -> max_split_k=128
num_pages >= 128 -> max_split_k=64
num_pages >=  96 -> max_split_k=32
num_pages >=  16 -> max_split_k=16
otherwise        -> max_split_k=1
```

decode-only，小矩阵复测，A40/GPU2，`page_fastpath`，parallel reduce auto：

| seq_len | fallback | auto median_us | old/default 对照 | 对照 median_us | 结论 |
| ---: | ---: | ---: | ---: | ---: | --- |
| 512 | 0 | 1011.20 | split8 | 1021.95 | auto 更快 |
| 1536 | 0 | 1110.02 | split16 | 1135.62 | auto 更快 |
| 2048 | 0 | 1143.81 | split64 | 1144.32 | 保持 split64 附近 |
| 512 | 0.03 | 1021.82 | split8 | 1038.34 | auto 更快 |
| 1536 | 0.03 | 1127.42 | split16 | 1174.53 | auto 更快 |
| 2048 | 0.03 | 1163.78 | split64 | 1164.29 | 保持 split64 附近 |

关键 E2E 对照，Llama-3-8B-Instruct，batch=1，cudagraph on，
compressed-only，fallback pool ratio=0.03：

| prompt_len | decode_len | split 策略 | output tok/s | 结论 |
| ---: | ---: | --- | ---: | --- |
| 2048 | 32 | explicit split32 | 28.63 | 不采用 |
| 2048 | 32 | explicit split64 | 29.58 | 保持 128 pages 起用 split64 |
| 512 | 32 | explicit split8 | 29.56 | 旧短 context 对照 |
| 512 | 32 | new auto | 30.72 | +3.9%，保留 |

结果文件：

- `benchmarks/profiles/bytev2_splitk_autotune_fb0_p512_p4096.json`
- `benchmarks/profiles/bytev2_splitk_autotune_fb0_p256_p768_p8192.json`
- `benchmarks/profiles/bytev2_splitk_autotune_fb003_p512_p4096.json`
- `benchmarks/profiles/bytev2_splitk_autotune_after_heuristic_fb0_p512_p1536_p2048.json`
- `benchmarks/profiles/bytev2_splitk_autotune_after_heuristic_fb003_p512_p1536_p2048.json`
- `benchmarks/profiles/bytev2_splitk_e2e_p2048_d32_s32.json`
- `benchmarks/profiles/bytev2_splitk_e2e_p2048_d32_s64.json`
- `benchmarks/profiles/bytev2_splitk_e2e_p512_d32_auto_after_heuristic.json`
- `benchmarks/profiles/bytev2_splitk_e2e_p512_d32_s8_after_heuristic.json`

correctness/build：

```bash
cmake --build build/temp.linux-x86_64-cpython-312 \
  --target _C_stable_libtorch -j 16

CUDA_VISIBLE_DEVICES=2 .venv/bin/python -m pytest \
  tests/v1/attention/test_byte_v2_decode.py::test_byte_v2_paged_decode_attention_op_gqa_wmma_auto_split_k_cuda \
  tests/v1/attention/test_byte_v2_decode.py::test_byte_v2_paged_decode_attention_op_gqa_wmma_split_k_cuda \
  tests/v1/attention/test_byte_v2_decode.py::test_byte_v2_paged_decode_attention_op_page_fastpath_split_k_cuda \
  tests/v1/attention/test_byte_v2_decode.py::test_byte_v2_native_paged_decode_parallel_reduce_cuda -q
# 4 passed
```

结论：

- 保留 C4 heuristic 调整。
- 不把 `p2048` 下调到 `split32`，因为真实 E2E 明显慢于 `split64`。
- 当前收益主要在 p512/p1536 这类原默认 split 偏低区间；不是核心瓶颈的根治，
  但属于低风险调度修正。

### 方向 D：compressed hot path 的 fallback metadata 成本

当前判断：

- Step 13 证明 raw-fallback-only stage1 拆分不适合当前 3% fallback 场景：
  raw page 太少，额外 launch/partial 合并会抵消收益。
- Step 14/15 显示 compact fallback metadata/list 方向只有在 tile-level fallback
  metadata 成本成为明确瓶颈时才值得做。
- 当前 hot path 仍可能为每个 element 或每个 tile 检查 fallback metadata。

#### D1：page-level no-fallback bitset

cache update 阶段为每个 physical block 生成一个 bit：

```text
page_has_any_tile_fallback[physical_block] = 0/1
```

decode stage1 中：

```text
if status == COMPRESSED && page_has_any_tile_fallback == 0:
    use no-tile-fallback decoder
else:
    use generic tile fallback decoder
```

预期收益：

- 绝大多数 compressed page 可完全跳过 dense `fallback_tile_ids` 读取。
- 比 raw/compressed page 拆 kernel 少一次 launch。

风险：

- 每 page 增加 1 bit/byte metadata。
- cache update 要维护该 bit，不能引入 host sync。

保留标准：

- NCU 中 metadata global load 下降。
- p2048/d32 E2E >= +2%。
- fallback pool exhausted 状态不变。

实验记录，2026-06-09：

- 尝试实现：复用 ByteV2 page header 中未定义的 byte 作为
  `page_has_any_tile_fallback`，cache update/finalize 阶段写入，split-K
  WMMA stage1 在 `VLLM_BYTE_V2_DECODE_PAGE_FASTPATH=1` 时让无 tile fallback
  page 跳过 `fallback_tile_ids` 检查。
- decode-only microbench 有正向信号：

| seq_len | split_k | fallback_ratio | baseline median_us | D1 median_us | speedup |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1024 | 16 | 0 | 1078.26 | 1057.28 | +1.98% |
| 1024 | 64 | 0 | 1142.78 | 1131.01 | +1.04% |
| 2048 | 16 | 0 | 1281.54 | 1214.98 | +5.48% |
| 2048 | 64 | 0 | 1186.82 | 1156.10 | +2.66% |
| 1024 | 16 | 0.03 | 1082.88 | 1059.33 | +2.22% |
| 1024 | 64 | 0.03 | 1139.74 | 1120.26 | +1.74% |
| 2048 | 16 | 0.03 | 1287.17 | 1226.24 | +4.97% |
| 2048 | 64 | 0.03 | 1188.86 | 1163.26 | +2.20% |

- E2E 未达到保留标准：

| prompt_len | decode_len | page_fastpath=0 | page_fastpath=1 | delta |
| ---: | ---: | ---: | ---: | ---: |
| 2048 | 32 | 29.06 tok/s | 29.17 tok/s | +0.35% |
| 2048 | 128 | 30.95 tok/s | 30.91 tok/s | -0.14% |

- fallback pool 未耗尽，但 E2E 收益低于 +2%，且把 header byte 引入为隐式格式
  语义会增加维护成本。
- 结论：不保留 D1 代码，已回滚实验实现和新增测试；只保留 benchmark JSON：
  `benchmarks/profiles/bytev2_step28_d1_decode_baseline.json`、
  `benchmarks/profiles/bytev2_step28_d1_decode_page_fastpath.json`、
  `benchmarks/profiles/bytev2_step28_d1_e2e_page_fastpath_off_p2048_d32.json`、
  `benchmarks/profiles/bytev2_step28_d1_e2e_page_fastpath_on_p2048_d32.json`、
  `benchmarks/profiles/bytev2_step28_d1_e2e_page_fastpath_off_p2048_d128.json`、
  `benchmarks/profiles/bytev2_step28_d1_e2e_page_fastpath_on_p2048_d128.json`。
- 后续不再优先做 page-level no-fallback metadata，除非 NCU 明确证明
  `fallback_tile_ids` global load 已成为主瓶颈。

#### D2：tile fallback compact list + per-page tile mask

替代 dense：

```text
fallback_tile_ids[physical_block][total_tiles]
```

改为：

```text
page_tile_fallback_mask[physical_block][num_words]
compact_tile_slots[...]
```

decode 时：

- 先读 tile mask。
- mask=0 时走 compressed decoder。
- mask=1 时通过 compact offset 找 raw tile slot。

预期收益：

- 降低 dense fallback metadata 的 global memory footprint。
- 对 tile fallback 稀疏场景更合适。

风险：

- 需要 compact offset 或 prefix/count，cache update 更复杂。
- tile fallback 比例高时 mask/list 查找未必更快。

执行顺序：

1. 先实现 D1，确认 page no-fallback 比例和收益。
2. 只有 D1 显示 metadata 仍是瓶颈时再做 D2。

保留标准：

- 真实 Llama-3 p1024/p2048 下 E2E 有稳定收益。
- cache update 不明显变慢。

#### D3：decode kernel 内 page chunk 分类

在单个 stage1 kernel 内按 page chunk 分类：

```text
all compressed no fallback
compressed with tile fallback
raw block fallback
```

实现方式：

- 不新增 raw-only kernel launch。
- 在 page chunk 入口处选择函数对象/模板分支。
- 分支粒度是 page/chunk，不是 element。

目的：

- 保留单 launch，同时避免 element-level fallback 分支。

风险：

- CUDA 模板展开可能增大代码体积和寄存器。
- chunk 内如果混合 page type，仍需要 fallback 到 generic path。

保留标准：

- compressed-only/fallback=0 不退化。
- 真实 fallback=0.03 有收益。

实验记录，2026-06-10：

- 实现 tile-level compressed fastpath：
  - 在 split-K WMMA stage1 的 compressed page 分支中，每个 K/V tile 只读取一次
    `page[tile_start + 1]` fallback flag。
  - fallback flag 为 0 时直接调用已有
    `byte_v2_decode_*_tile_to_shared_no_fallback()`，避免每个 element 重复检查
    tile fallback metadata。
  - fallback flag 为 1 时调用新增
    `byte_v2_decode_*_tile_to_shared_tile_fallback()`，从 tile fallback pool 读取
    16x16 raw BF16 tile。
  - 不新增 kernel launch，不新增 cache metadata。
- 控制开关：
  - `VLLM_BYTE_V2_DECODE_TILE_FASTPATH=1`：启用。
  - `VLLM_BYTE_V2_DECODE_TILE_FASTPATH=0`：回退到旧 element-level loader。
  - 默认启用；它只在 `VLLM_BYTE_V2_DECODE_PAGE_FASTPATH=1` 的 compressed page
    stage1 路径生效。
- correctness：

```bash
CUDA_VISIBLE_DEVICES=2 .venv/bin/python -m pytest \
  tests/v1/attention/test_byte_v2_decode.py::test_byte_v2_paged_decode_attention_op_tile_fastpath_cuda \
  tests/v1/attention/test_byte_v2_decode.py::test_byte_v2_paged_decode_attention_op_page_fastpath_split_k_cuda \
  tests/v1/attention/test_byte_v2_decode.py::test_byte_v2_paged_decode_attention_op_gqa_wmma_split_k_cuda \
  tests/v1/attention/test_byte_v2_decode.py::test_byte_v2_native_paged_decode_parallel_reduce_cuda -q
# 4 passed

CUDA_VISIBLE_DEVICES=2 .venv/bin/python -m pytest \
  tests/v1/attention/test_byte_v2_decode.py -q
# 19 passed
```

decode-only sanity，A40/GPU2，`page_fastpath`，parallel reduce auto：

| seq_len | split_k | fallback | tile fastpath off median_us | tile fastpath on median_us | delta |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 16 | 0.03 | 1014.78 | 997.89 | +1.7% |
| 512 | 16 | 0.03 | 1014.78 | auto 1012.74 | +0.2% |

注：decode-only benchmark 当前不传真实 `fallback_tile_ids`，因此它只能作为
compressed no-fallback / block fallback sanity；真实收益以 E2E 为准。

E2E，Llama-3-8B-Instruct，batch=1，cudagraph on，compressed-only，
fallback pool ratio=0.03：

| prompt_len | decode_len | runs | tile fastpath off | tile fastpath on | delta |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 32 | 3 | 30.73 tok/s | 31.65 tok/s | +3.0% |
| 2048 | 32 | 3 | 29.41 tok/s | 30.54 tok/s | +3.8% |

单次 p2048/d32 预检也显示 29.45 -> 30.56 tok/s，方向一致。
所有 E2E run 的 sparse fallback pool `any_exhausted=false`。

结果文件：

- `benchmarks/profiles/bytev2_tile_fastpath_off_decode_sanity.json`
- `benchmarks/profiles/bytev2_tile_fastpath_on_decode_sanity.json`
- `benchmarks/profiles/bytev2_tile_fastpath_recheck_off_decode_p512_s16_fb003.json`
- `benchmarks/profiles/bytev2_tile_fastpath_recheck_auto_decode_p512_s16_fb003.json`
- `benchmarks/profiles/bytev2_tile_fastpath_recheck_on_decode_p512_s16_fb003.json`
- `benchmarks/profiles/bytev2_tile_fastpath_off_e2e_p2048_d32_r3.json`
- `benchmarks/profiles/bytev2_tile_fastpath_on_e2e_p2048_d32_r3.json`
- `benchmarks/profiles/bytev2_tile_fastpath_off_e2e_p512_d32_r3.json`
- `benchmarks/profiles/bytev2_tile_fastpath_on_e2e_p512_d32_r3.json`

结论：保留 D3，并默认启用。该改动解决的是 compressed hot path 中
element-level fallback metadata 检查成本的一部分，对 p512/p2048 的 E2E 都有
稳定收益。它不是对 raw fallback page 的调度层拆分，因此不会增加 launch/reduce
开销。

## Step 29: batch>1 E2E 退化诊断与修复计划

### 现象

2026-06-10 重新扫 `prompt_len=512`、`decode_len=16/32/64/128`、
`batch_size=1/2/4`，同机 GPU2，CUDA graph on，raw 与 ByteV2 compressed-only
对照。表中 tok/s 是整个 batch 的 output token 吞吐：

| batch | decode_len | raw tok/s | ByteV2 tok/s | ByteV2/raw |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 16 | 35.26 | 29.03 | 82.3% |
| 1 | 32 | 35.51 | 30.61 | 86.2% |
| 1 | 64 | 35.62 | 31.48 | 88.4% |
| 1 | 128 | 35.67 | 31.86 | 89.3% |
| 2 | 16 | 67.13 | 12.19 | 18.2% |
| 2 | 32 | 67.49 | 11.88 | 17.6% |
| 2 | 64 | 67.63 | 11.82 | 17.5% |
| 2 | 128 | 67.68 | 11.75 | 17.4% |
| 4 | 16 | 132.01 | 24.21 | 18.3% |
| 4 | 32 | 133.15 | 23.77 | 17.9% |
| 4 | 64 | 133.60 | 23.52 | 17.6% |

raw vLLM 随 batch 基本线性扩展：

```text
batch=2: raw ~= 1.90x batch=1
batch=4: raw ~= 3.75x batch=1
```

ByteV2 当前没有正常扩展：

```text
batch=2: ByteV2 ~= 0.37x-0.42x batch=1
batch=4: ByteV2 ~= 0.75x-0.83x batch=1
```

所有 run 的 sparse fallback pool 均未耗尽：

```text
batch=1: any_exhausted=false, max_next_slot=32, max_capacity=379
batch=2: any_exhausted=false, max_next_slot=0,  max_capacity=391
batch=4: any_exhausted=false, max_next_slot=0,  max_capacity=378
```

结果文件：

- `benchmarks/profiles/byv2_vs_raw_matrix_p512_b1_d16_32_64_128_r2.json`
- `benchmarks/profiles/byv2_vs_raw_matrix_p512_b2_d16_32_64_128_r2.json`
- `benchmarks/profiles/byv2_vs_raw_matrix_p512_b4_d16_32_64_r2_clean.json`

### 根因判断

主要问题是 decode append cache update 只优化了 `num_tokens == 1`。

当前 fast path：

```cpp
if (compressed_only_pages && has_sparse_fallback && num_tokens == 1) {
  byte_v2_decode_append_cache_kernel<<<1, 256>>>(...);
}
```

当 `batch_size=1` 时，每步 decode 只有一个 token，能走该路径。  
当 `batch_size=2/4` 时，每步 decode 的 `num_tokens=batch_size`，fast path 不成立，
会落回 generic compressed-only update：

```text
valid_rows[num_blocks]
touched_flags[num_blocks]
packed_flags[num_blocks]
overwrite_flags[num_blocks]
block_token_indices[num_blocks * 16]
raw_staging[num_tokens * raw_block_bytes]

byte_v2_init_cache_update_kernel<<<num_blocks>>>
byte_v2_mark_touched_tokens_kernel<<<num_tokens>>>
byte_v2_compress_touched_blocks_kernel<<<num_blocks>>>
```

这条 generic path 是为通用 prefill/update 设计的。对 decode 每步只追加几个
token 的场景，它会引入 `num_blocks` 级别扫描、临时 staging allocation、block-token
索引初始化和更重的压缩流程，因此 batch>1 时把 E2E 吞吐压到 raw 的约 17%-18%。

次要问题是 split-K heuristic 在多 decode token 下会降低 split：

```python
active_head_groups = num_decode_tokens * num_kv_heads
if active_head_groups >= 64:
    max_split_k = min(max_split_k, 8)
elif active_head_groups >= 32:
    max_split_k = min(max_split_k, 16)
```

这个 heuristic 原本是为了避免 batch/head group 已足够多时过度 split。但在 cache
update 修复前，attention split-K 不是首要瓶颈；应先修 cache update，再重新 sweep
split-K。

### 修复方案 A：batched decode append fast path

新增或改造 `byte_v2_decode_append_cache_kernel` 为 batched decode append kernel：

```text
grid.x = num_tokens
one CTA per decode token
blockDim.x = 256
```

每个 CTA 独立处理一个 decode token：

1. 读取 `slot_mapping[token_idx]`，计算 `block_id` 和 `block_offset`。
2. 读取当前 page status 和 valid rows。
3. 若需要 partial raw block：
   - 复用已有 block fallback slot，或从 `fallback_next_slot` 分配一个 slot。
   - 如果 page 是 compressed 且要继续 append，先把已有 compressed page 解压到
     raw partial block。
4. 从 `key[token_idx]`、`value[token_idx]` 写入 raw partial block 对应 row。
5. 若 `block_offset + 1 < 16`：
   - page 标为 raw fallback partial。
   - `valid_rows = block_offset + 1`。
6. 若 `block_offset == 15`：
   - 复用 `byte_v2_finalize_raw_block_parallel()` 做 CTA 内 tile-parallel finalize。
   - full block 可压缩时写 compressed page。
   - 少量不可压缩 tile 写 tile fallback pool。
   - 必要时返回 finalized block id。

正确性约束：

- batch 内通常每个 request 写不同 physical block，但 kernel 必须 fail-closed。
- 增加轻量 duplicate block 检查：
  - 小 batch 可在 batched kernel 内做 `O(num_tokens^2)` block id 检查。
  - 或新增一个 small validation kernel，在进入 append kernel 前检查同一步是否有
    多个 token 写同一 `(block_id, block_offset)` 或同一 block 的不合法 offset。
- 如发现 duplicate/race，写 error code 到 device-side result/sticky flag。

### 修复方案 B：batched deferred error reporting

当前 single-token fast path 的 `fast_result` 只有两个字段：

```text
fast_result[0] = error code
fast_result[1] = finalized block id or -1
```

batched 版本应改成 device-side result buffer：

```text
result[0] = first_error_code
result[1] = first_error_token_or_block
result[2 + token_idx] = finalized block id or -1
```

production-safe 路径继续使用 Step 26 的 deferred sticky error flag：

```text
forward 前清零 sticky flag
cache update kernel 出错时 atomicCAS 写 sticky flag
forward 后统一检查一次
```

不允许恢复每 token/layer host sync。benchmark 路径可以继续返回空 packed list；当前
ByteV2 主路径不消费 packed block id。

### 修复方案 C：cache update 修复后再调 split-K

batched decode append 修复后，重新 sweep attention split-K：

```text
batch_size = 1,2,4
decode_len = 16,32,64,128
prompt_len = 512,2048
split_k = auto,8,16,32,64
```

保留标准：

- batch=1 不回退超过 1%。
- batch=2/4 ByteV2 总 output tok/s 至少随 batch 正向扩展。
- batch=2/4 ByteV2/raw 明显高于当前 17%-18%，第一阶段目标至少恢复到 60% raw。
- fallback pool `any_exhausted=false`。
- greedy correctness smoke 与 raw 对齐。

### 本轮实现与实验结果

2026-06-10 已完成 batched decode append fast path，并保留代码。

实现要点：

- `byte_v2_decode_append_cache_kernel` 从固定 `grid=1` 改为
  `grid.x=num_tokens`，每个 decode token 一个 CTA。
- K/V 读取改为使用 `token_idx * stride0`，修复 batch>1 时不能直接复用
  single-token source offset 的问题。
- result layout 扩展为：

```text
result[0] = first_error_code
result[1] = first_error_detail
result[2 + token_idx] = finalized block id or -1
```

- batch>1 下增加同一步 duplicate physical block 检查；发现多个 token 写同一
  block 时 fail-closed，避免并发 CTA 写同一个 partial raw block。
- single-token fast path 不额外 launch result init kernel；`num_tokens==1` 时仍在
  append CTA 内初始化 result，避免 b1 回退。
- batch>1 fast path 只在以下场景启用：
  - CUDA graph capture 中自动启用，用于生产 decode graph；
  - 或显式设置 `VLLM_BYTE_V2_DECODE_APPEND_BATCH_FASTPATH=1`，用于单测/实验。
  这样普通 non-graph 小段 continuation prefill 仍走 generic path，避免被误判成
  decode append。
- deferred error reporting 继续复用 device-side sticky error flag；graph replay 中不
  引入每 token/layer host sync。

新增/更新测试：

```bash
CUDA_VISIBLE_DEVICES=2 .venv/bin/python -m pytest \
  tests/v1/attention/test_byte_v2_ops.py::test_byte_v2_native_decode_append_uses_partial_raw_fallback_cuda \
  tests/v1/attention/test_byte_v2_ops.py::test_byte_v2_native_batched_decode_append_uses_partial_raw_fallback_cuda \
  tests/v1/attention/test_byte_v2_ops.py::test_byte_v2_native_decode_append_finalizes_partial_raw_fallback_cuda \
  tests/v1/attention/test_byte_v2_ops.py::test_byte_v2_native_batched_decode_append_finalizes_blocks_cuda \
  tests/v1/attention/test_byte_v2_ops.py::test_byte_v2_native_batched_decode_append_duplicate_block_fails_cuda \
  tests/v1/attention/test_byte_v2_ops.py::test_byte_v2_native_decode_append_fallback_pool_exhaustion_cuda \
  -q
```

结果：`6 passed`。

额外 smoke：

```bash
CUDA_VISIBLE_DEVICES=2 .venv/bin/python -m pytest \
  tests/v1/attention/test_byte_v2_ops.py::test_byte_v2_native_tile_fallback_pool_cuda \
  tests/v1/attention/test_byte_v2_ops.py::test_byte_v2_native_tile_and_block_fallback_pool_do_not_overlap_cuda \
  tests/v1/attention/test_byte_v2_ops.py::test_byte_v2_reshape_and_cache_op_packs_full_block_cuda \
  tests/v1/attention/test_byte_v2_ops.py::test_byte_v2_reshape_and_cache_op_finalizes_existing_raw_block_cuda \
  tests/v1/attention/test_byte_v2_backend.py::test_byte_v2_deferred_cache_update_records_sticky_error \
  -q
```

结果：`5 passed`。

E2E 配置：GPU2，Llama-3-8B-Instruct，`prompt_len=512`，CUDA graph on，
`VLLM_BYTE_V2_DEFERRED_CACHE_UPDATE_ERROR_CHECK=1`，3% sparse fallback pool。

| batch | decode_len | raw tok/s | ByteV2 tok/s | ByteV2/raw | 修复前 ByteV2 tok/s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 16 | 35.26 | 29.15 | 82.7% | 29.03 |
| 1 | 32 | 35.52 | 30.68 | 86.4% | 30.61 |
| 2 | 16 | 67.18 | 55.32 | 82.3% | 12.19 |
| 2 | 32 | 67.52 | 58.40 | 86.5% | 11.88 |
| 2 | 64 | 67.63 | 60.00 | 88.7% | 11.82 |
| 2 | 128 | 67.68 | 60.90 | 90.0% | 11.75 |
| 4 | 16 | 132.05 | 106.26 | 80.5% | 24.21 |
| 4 | 32 | 133.11 | 113.59 | 85.3% | 23.77 |
| 4 | 64 | 133.60 | 117.75 | 88.1% | 23.52 |

结果文件：

- `benchmarks/profiles/byv2_batched_append_fix_p512_b1_d16_32_r2.json`
- `benchmarks/profiles/byv2_batched_append_fix_p512_b2_d16_32_r2.json`
- `benchmarks/profiles/byv2_batched_append_fix_p512_b2_d64_128_r2_byte_only.json`
- `benchmarks/profiles/byv2_batched_append_fix_p512_b4_d16_32_r2.json`
- `benchmarks/profiles/byv2_batched_append_fix_p512_b4_d64_r2_byte_only.json`

sparse fallback pool：上述 ByteV2 run 均 `any_exhausted=false`。

结论：

- batch>1 的主要 E2E 退化已确认来自 decode cache update 误走 generic
  touched-block path。
- batched append 修复后，b2/b4 从 raw 的约 17%-18% 恢复到约 80%-90%。
- b1 基本不变，因此本改动保留。
- 下一步主要瓶颈应重新转向 decode attention stage1/reduce 和 batch-aware
  split-K，而不是 cache update。

### 最小验证

correctness：

```bash
CUDA_VISIBLE_DEVICES=<gpu> .venv/bin/python -m pytest \
  tests/v1/attention/test_byte_v2_ops.py::test_byte_v2_native_decode_append_uses_partial_raw_fallback_cuda \
  tests/v1/attention/test_byte_v2_ops.py::test_byte_v2_native_decode_append_finalizes_partial_raw_fallback_cuda \
  tests/v1/attention/test_byte_v2_ops.py::test_byte_v2_native_decode_append_fallback_pool_exhaustion_cuda \
  tests/v1/attention/test_byte_v2_ops.py::test_byte_v2_native_tile_fallback_pool_cuda \
  tests/v1/attention/test_byte_v2_decode.py -q
```

新增测试：

- `test_byte_v2_native_batched_decode_append_cuda`
- `test_byte_v2_native_batched_decode_append_finalizes_blocks_cuda`
- `test_byte_v2_native_batched_decode_append_duplicate_block_fails_cuda`
- `test_byte_v2_e2e_batch_gt_one_cuda`

E2E 复测：

```bash
CUDA_VISIBLE_DEVICES=<gpu> \
VLLM_ENABLE_V1_MULTIPROCESSING=0 \
VLLM_BYTE_V2_USE_NATIVE_KERNELS=1 \
VLLM_BYTE_V2_SPARSE_FALLBACK_POOL_RATIO=0.03 \
VLLM_BYTE_V2_SPARSE_FALLBACK_POOL_MIN_BLOCKS=0 \
VLLM_BYTE_V2_DEFERRED_CACHE_UPDATE_ERROR_CHECK=1 \
.venv/bin/python benchmarks/benchmark_byte_v2_decode_e2e.py \
  --modes raw,byte_v2_compressed_only \
  --model /mnt/sda1/yxz/byte_v2/Meta-Llama-3-8B-Instruct \
  --prompt-len 512 \
  --decode-lens 16,32,64,128 \
  --batch-size 1 \
  --num-runs 2 \
  --warmup-decode-len 8 \
  --gpu-memory-utilization 0.80

# 再分别跑 --batch-size 2 和 --batch-size 4。
```

profile 必看：

- `byte_v2_decode_append_cache_kernel` 或新 batched append kernel 的 total/avg time。
- generic `byte_v2_compress_touched_blocks_kernel` 是否从 decode measured range 消失。
- `byte_v2_paged_decode_attention_gqa_wmma_split_stage1_kernel` 与 reduce 是否成为新的
  主瓶颈。
- fallback pool `any_exhausted`、`max_next_slot/max_capacity`、tile fallback slot 使用。

### 推荐执行顺序

下一轮按以下顺序做，每步无收益就删除。D1 已在 2026-06-09 尝试且未保留，
A1 已在 2026-06-10 尝试且未保留，D3 已在 2026-06-10 尝试且保留。
Step 29 发现 batch>1 是当前新的主要 E2E 缺口，因此下一步优先从 batched decode
append 继续：

1. **Batched decode append fast path**：把 `num_tokens > 1` 的 decode cache update
   从 generic touched-block path 拉回 one-CTA-per-token fast path。
2. **Batched deferred error reporting**：保持 production-safe，但不引入 per-token
   host sync。
3. **batch-aware split-K sweep**：cache update 修复后，再重新调
   `active_head_groups` cap。
4. **C1 persistent partial workspace**：处理 allocator/partial workspace runtime 成本，
   要特别验证 cudagraph 和 KV capacity。
5. **B1 WMMA fragment swizzle microkernel**：先单独验证 shared/WMMA layout，不直接
   改 paged decode。
6. 如果 B1 有明确收益，再做 **B2 独立 CUTE/CUTLASS-style stage1 variant**。

暂时不要做：

- 再次实现 direct-output decode op。Step 27 已证明无收益。
- 再次做简单 K shared stride padding。Step 25 已证明退化。
- 再次做 row-major K shared + WMMA B col-major。Step 23 已证明退化。
- 单独 raw-fallback-only stage1 kernel。Step 13 已证明额外 launch/partial 合并不划算。
- 再次做局部 compressed decoder 微调。A1 的 vec4 和 pair_v2 都没有达到
  decode-only 保留门槛。
- 再次在当前 WMMA kernel 内做 CTA-local K/V shared double-buffer。A4 已证明
  warp-specialized 预解码下一页会退化。

### 每个实验的最小验证集

每完成一个候选实验，至少运行：

```bash
.venv/bin/python -m py_compile \
  vllm/_custom_ops.py \
  vllm/v1/attention/backends/byte_v2_attn.py \
  tests/v1/attention/test_byte_v2_decode.py

.venv/bin/python -m ruff check \
  vllm/_custom_ops.py \
  vllm/v1/attention/backends/byte_v2_attn.py \
  tests/v1/attention/test_byte_v2_decode.py

.venv/bin/python -m pytest tests/v1/attention/test_byte_v2_decode.py -q
.venv/bin/python -m pytest tests/v1/attention/test_byte_v2_backend.py -q
```

性能验证：

```bash
# decode-only
CUDA_VISIBLE_DEVICES=<gpu> \
.venv/bin/python benchmarks/kernels/benchmark_byte_v2_decode_kernel.py \
  --batch-size 1 \
  --seq-len 1024,2048 \
  --num-heads 32 \
  --num-kv-heads 8 \
  --head-size 128 \
  --head-size-v 128 \
  --split-k 16,64 \
  --fallback-ratio 0,0.03 \
  --num-runs 50 \
  --warmup-runs 10

# E2E
CUDA_VISIBLE_DEVICES=<gpu> \
VLLM_ENABLE_V1_MULTIPROCESSING=0 \
VLLM_BYTE_V2_USE_NATIVE_KERNELS=1 \
VLLM_BYTE_V2_SPARSE_FALLBACK_POOL_RATIO=0.03 \
VLLM_BYTE_V2_SPARSE_FALLBACK_POOL_MIN_BLOCKS=0 \
VLLM_BYTE_V2_DEFERRED_CACHE_UPDATE_ERROR_CHECK=1 \
.venv/bin/python benchmarks/benchmark_byte_v2_decode_e2e.py \
  --modes byte_v2_compressed_only \
  --prompt-len 2048 \
  --decode-lens 32 \
  --batch-size 1 \
  --num-runs 3 \
  --warmup-decode-len 8
```

必要时补 NCU：

```bash
ncu --set full \
  --kernel-name regex:byte_v2_paged_decode_attention_gqa_wmma_split_stage1_kernel \
  --target-processes all \
  --log-file benchmarks/profiles/<experiment>_stage1.csv \
  ...
```

记录项：

- stage1 duration。
- split reduce duration。
- global load sectors / excessive sectors。
- shared bank conflict / excessive wavefronts。
- executed instructions。
- register count。
- achieved occupancy / eligible warps。
- fallback pool `any_exhausted`、`total_next_slot`、`tile_next_slot`。

## Step 30：长 decode split-K 与 persistent workspace 复测

### 背景

Step 29 修复 batched decode append 后，`batch_size=2/4` 的 E2E 已恢复到 raw
vLLM 的约 80%-90%。随后把 decode len 拉长到 256/512/1024/2048，发现
ByteV2 仍没有超过 raw：

| batch | decode_len | raw tok/s | ByteV2 tok/s | ByteV2/raw | 备注 |
| ---: | ---: | ---: | ---: | ---: | --- |
| 2 | 256 | 67.48 | 62.21 | 92.2% | 3% fallback pool |
| 2 | 512 | 67.36 | 62.31 | 92.5% | 3% fallback pool |
| 2 | 1024 | 67.09 | 61.54 | 91.7% | 3% fallback pool |
| 4 | 256 | 133.55 | 120.36 | 90.1% | 10% fallback pool |
| 4 | 512 | 133.18 | 119.11 | 89.4% | 10% fallback pool |
| 4 | 1024 | 132.25 | 115.90 | 87.6% | 10% fallback pool |
| 4 | 2048 | 130.25 | 109.39 | 84.0% | 10% fallback pool |

3% fallback pool 在 `batch=4, decode_len=1024` 的累计 run 中会耗尽：

```text
fallback_pool_used=380, fallback_pool_capacity=377
```

因此本轮性能实验使用 10% fallback pool 只作为隔离容量限制的性能上界，不代表
默认配置应改回 10%。

### 实验 A：decode-only split-K sweep

命令：

```bash
CUDA_VISIBLE_DEVICES=2 \
.venv/bin/python benchmarks/kernels/benchmark_byte_v2_decode_kernel.py \
  --batch-size 4 \
  --seq-len 1024,2048,4096 \
  --num-heads 32 \
  --num-kv-heads 8 \
  --head-size 128 \
  --head-size-v 128 \
  --split-k 8,16,32,64,128 \
  --fallback-ratio 0,0.03 \
  --num-runs 50 \
  --warmup-runs 10 \
  --skip-correctness \
  --output-json benchmarks/profiles/byv2_splitk_sweep_b4_s1024_2048_4096_fb0_003.json
```

decode-only 结果显示长上下文下更大的 split-K 有收益：

| seq_len | fallback | best split_k | best median_us | split_k=16 median_us | delta |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 0 | 32 | 1256.96 | 1281.02 | 1.9% faster |
| 2048 | 0 | 64 | 1384.45 | 1467.39 | 5.7% faster |
| 4096 | 0 | 64 | 1630.21 | 1835.52 | 11.2% faster |
| 1024 | 0.03 | 32 | 1259.01 | 1285.12 | 2.0% faster |
| 2048 | 0.03 | 64 | 1384.45 | 1467.39 | 5.7% faster |
| 4096 | 0.03 | 64 | 1633.28 | 1837.06 | 11.1% faster |

这说明 isolated attention kernel 里，当前 `active_head_groups` cap 可能对长上下文
过于保守。

### 实验 B：E2E split-K override

E2E 配置：A40/GPU2，Llama-3-8B-Instruct，`prompt_len=512`，`batch_size=4`，
CUDA graph on，10% fallback pool。

| decode_len | auto tok/s | split_k=32 tok/s | split_k=64 tok/s |
| ---: | ---: | ---: | ---: |
| 512 | 119.11 | 117.45 | 116.67 |
| 1024 | 115.90 | 114.67 | 113.78 |

补充 `split_k=8` 也更差：

| decode_len | auto tok/s | split_k=8 tok/s |
| ---: | ---: | ---: |
| 512 | 119.11 | 113.07 |
| 1024 | 115.90 | 108.44 |

结果文件：

- `benchmarks/profiles/byv2_long_decode_p512_b4_d512_1024_r1_fb010_split8.json`
- `benchmarks/profiles/byv2_long_decode_p512_b4_d512_1024_r1_fb010_split32.json`
- `benchmarks/profiles/byv2_long_decode_p512_b4_d512_1024_r1_fb010_split64.json`

结论：decode-only 的 split-K 收益没有转化为 E2E 收益。原因很可能是更大 split
增加了 reduce 和 graph 内 kernel/runtime 开销，而真实生成每一步的有效 seq_len
从 prompt_len 逐步增长，并不总处在 decode-only 固定长上下文的最佳点。暂不修改
`_decode_num_kv_splits()` 默认 heuristic。

### 实验 B2：NCU stage1/reduce 对比

固定形状：`batch=4, seq_len=2048, fallback=0.03`。使用 NCU 只采一个 matched
kernel launch：

```bash
CUDA_VISIBLE_DEVICES=2 ncu --set full --csv --force-overwrite \
  --target-processes all \
  --kernel-name regex:byte_v2_paged_decode_attention_gqa_wmma_split_stage1_kernel \
  --launch-skip 2 --launch-count 1 \
  --log-file benchmarks/profiles/byv2_ncu_decode_stage1_b4_s2048_fb003_split16.csv \
  .venv/bin/python benchmarks/kernels/benchmark_byte_v2_decode_kernel.py \
    --batch-size 4 --seq-len 2048 --num-heads 32 --num-kv-heads 8 \
    --head-size 128 --head-size-v 128 --split-k 16 \
    --fallback-ratio 0.03 --num-runs 3 --warmup-runs 2 --skip-correctness
```

同样命令替换 `split-k=64` 和 reduce kernel regex：

```text
regex:byte_v2_paged_decode_attention_split_reduce
```

关键 NCU 指标：

| kernel | split_k | grid | duration | DRAM throughput | SM throughput | no eligible | executed inst |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| stage1 | 16 | 512 | 483.424 us | 16.03% | 24.25% | 62.01% | 48.89M |
| stage1 | 64 | 2048 | 361.248 us | 33.77% | 35.71% | 61.56% | 54.01M |
| reduce | 16 | 128 | 7.360 us | 26.57% | 10.71% | 84.70% | 0.315M |
| reduce parallel | 64 | 128 | 16.576 us | 50.19% | 14.53% | 82.57% | 0.950M |

结果文件：

- `benchmarks/profiles/byv2_ncu_decode_stage1_b4_s2048_fb003_split16.csv`
- `benchmarks/profiles/byv2_ncu_decode_stage1_b4_s2048_fb003_split64.csv`
- `benchmarks/profiles/byv2_ncu_decode_reduce_b4_s2048_fb003_split16.csv`
- `benchmarks/profiles/byv2_ncu_decode_reduce_b4_s2048_fb003_split64.csv`

解释：

- 固定 `seq_len=2048` 时，`split_k=64` 的 stage1 确实更快，且 DRAM/SM throughput
  明显更高；reduce 从 7.36us 增到 16.58us，但不足以抵消 stage1 收益。
- E2E 中 `split_k=64` 仍退化，说明问题不是单个固定长上下文 kernel，而是生成过程
  的动态上下文长度和 CUDA graph 固定 launch 形状：较大 split 在早期 token 会产生
  更多空 page chunk 和 partial/reduce 工作。
- `split_k=8` 明显退化，说明当前 auto=16 是这组 `batch=4, prompt=512,
  decode<=1024` E2E 的局部最优。
- 后续如果要利用长上下文 split=64 的收益，需要做分段/多图 split 策略，或让
  stage1/reduce 在固定大 grid 下更便宜地跳过无效 split；单纯改默认 split-K 不可取。

### 实验 C：persistent partial workspace

E2E 配置同上，只增加：

```text
VLLM_BYTE_V2_PERSISTENT_PARTIAL_WORKSPACE=1
```

结果：

| decode_len | auto tok/s | persistent workspace tok/s | delta |
| ---: | ---: | ---: | ---: |
| 512 | 119.11 | 119.22 | +0.09% |
| 1024 | 115.90 | 115.92 | +0.02% |
| 2048 | 109.39 | 109.46 | +0.06% |

结果文件：

- `benchmarks/profiles/byv2_long_decode_p512_b4_d512_1024_r1_fb010_persistent_workspace.json`
- `benchmarks/profiles/byv2_long_decode_p512_b4_d2048_r1_fb010_persistent_workspace.json`

结论：persistent workspace 可正常 CUDA graph capture，但收益在噪声内。partial
workspace 分配不是当前 E2E 主要瓶颈，暂不默认开启。

### 结论与下一步

本轮没有代码改动需要保留。

后续不要优先做：

- 单纯放大默认 split-K。
- 默认开启 persistent partial workspace。

下一步应回到真正的 stage1 kernel 结构问题：

1. 用 NCU 对比 auto、split_k=32、split_k=64 的 stage1/reduce，确认 E2E 中更大
   split 为什么不能兑现 decode-only 收益。
2. 在默认 WMMA split-stage1 上继续拆解 `load/decode K/V`、QK、softmax、PV、
   reduce 的占比，避免只看端到端 tok/s。
3. 如果 `load/decode K/V` 仍是主项，应设计新的 encode-side MMA-friendly layout
   或独立 stage1，而不是继续做局部 decoder 微调。
4. 如果 reduce 占比变大，应尝试 split reduce 合并/减少 partial 写回，而不是继续
   增加 split-K。

### 实验 D：空 split 只写 LSE，跳过 partial value 清零

动机：

- 当前 split stage1 对 `start_block >= end_block` 或 `seq_len <= 0` 的 split 会写
  `partial[d]=0` 和 `partial[head_size_v]=-FLT_MAX`。
- reduce 在 LSE 为 `-FLT_MAX` 时不会读取该 split 的 partial value。
- 因此尝试让空 split 只写 LSE，不再清零整段 partial value，降低强制大 split 时
  的空 split 成本。

实现范围：

- 默认 `byte_v2_paged_decode_attention_gqa_wmma_split_stage1_kernel`。
- 独立 `byte_v2_paged_decode_attention_gqa_cute_split_stage1_kernel`。
- 非 split kernel 不适用，因为它直接写最终 output。

正确性：

```bash
CUDA_VISIBLE_DEVICES=2 .venv/bin/python -m pytest \
  tests/v1/attention/test_byte_v2_decode.py::test_byte_v2_paged_decode_attention_op_gqa_wmma_split_k_cuda \
  tests/v1/attention/test_byte_v2_decode.py::test_byte_v2_paged_decode_attention_op_page_fastpath_split_k_cuda \
  tests/v1/attention/test_byte_v2_decode.py::test_byte_v2_native_paged_decode_parallel_reduce_cuda \
  tests/v1/attention/test_byte_v2_decode.py::test_byte_v2_paged_decode_attention_op_cute_stage1_cuda \
  -q
```

结果：`4 passed`。

E2E 对比，`prompt_len=512, batch=4, fallback=0.10`：

| case | decode_len | 修改前 tok/s | 修改后 tok/s | 结论 |
| --- | ---: | ---: | ---: | --- |
| auto | 512 | 119.11 | 119.07 | 无收益 |
| auto | 1024 | 115.90 | 115.83 | 无收益 |
| split_k=64 | 512 | 116.67 | 116.52 | 无收益 |
| split_k=64 | 1024 | 113.78 | 113.68 | 无收益 |

结果文件：

- `benchmarks/profiles/byv2_empty_split_lse_only_p512_b4_d512_1024_r1_fb010_auto.json`
- `benchmarks/profiles/byv2_empty_split_lse_only_p512_b4_d512_1024_r1_fb010_split64.json`

结论：该实验不保留，代码已回滚。空 split partial value 清零不是当前 E2E 可测
瓶颈；真正成本仍在有效 split 的 K/V decode、WMMA/softmax/PV，以及 reduce 需要
扫描固定 `num_kv_splits` 的整体结构。

### 实验 E：更长 prompt/decode 下是否可能自然超过 raw

目标：

- 验证 “decode 越长，ByteV2 读带宽优势越大，因此可能自然超过 raw” 是否成立。
- 验证 decode-only 中 `split_k=64` 的长上下文收益，是否能在更长 E2E 场景中出现。

公共配置：

- A40/GPU2，Llama-3-8B-Instruct，`batch_size=4`，CUDA graph on。
- ByteV2 使用 compressed-only cache，10% sparse fallback pool。
- `VLLM_BYTE_V2_DEFERRED_CACHE_UPDATE_ERROR_CHECK=1`。
- 每组 `num_runs=1`，因此只用于方向判断，不作为稳定发布数值。

#### 长 prompt：`prompt_len=2048`

命令：

```bash
CUDA_VISIBLE_DEVICES=2 VLLM_ENABLE_V1_MULTIPROCESSING=0 \
VLLM_BYTE_V2_USE_NATIVE_KERNELS=1 \
VLLM_BYTE_V2_SPARSE_FALLBACK_POOL_RATIO=0.10 \
VLLM_BYTE_V2_SPARSE_FALLBACK_POOL_MIN_BLOCKS=0 \
VLLM_BYTE_V2_DEFERRED_CACHE_UPDATE_ERROR_CHECK=1 \
.venv/bin/python benchmarks/benchmark_byte_v2_decode_e2e.py \
  --modes raw,byte_v2_compressed_only \
  --model /mnt/sda1/yxz/byte_v2/Meta-Llama-3-8B-Instruct \
  --prompt-len 2048 --decode-lens 512,1024 \
  --batch-size 4 --num-runs 1 --warmup-decode-len 8 \
  --gpu-memory-utilization 0.80 \
  --output-json benchmarks/profiles/byv2_long_prompt_p2048_b4_d512_1024_r1_fb010_auto_vs_raw.json
```

`split_k=64` 只跑 ByteV2，对照 raw 沿用同配置：

```bash
CUDA_VISIBLE_DEVICES=2 VLLM_ENABLE_V1_MULTIPROCESSING=0 \
VLLM_BYTE_V2_USE_NATIVE_KERNELS=1 \
VLLM_BYTE_V2_SPARSE_FALLBACK_POOL_RATIO=0.10 \
VLLM_BYTE_V2_SPARSE_FALLBACK_POOL_MIN_BLOCKS=0 \
VLLM_BYTE_V2_DEFERRED_CACHE_UPDATE_ERROR_CHECK=1 \
VLLM_BYTE_V2_DECODE_SPLIT_K=64 \
.venv/bin/python benchmarks/benchmark_byte_v2_decode_e2e.py \
  --modes byte_v2_compressed_only \
  --model /mnt/sda1/yxz/byte_v2/Meta-Llama-3-8B-Instruct \
  --prompt-len 2048 --decode-lens 512,1024 \
  --batch-size 4 --num-runs 1 --warmup-decode-len 8 \
  --gpu-memory-utilization 0.80 \
  --output-json benchmarks/profiles/byv2_long_prompt_p2048_b4_d512_1024_r1_fb010_split64.json
```

结果：

| decode_len | raw tok/s | ByteV2 auto tok/s | auto/raw | ByteV2 split64 tok/s | split64/raw |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 512 | 129.79 | 100.77 | 77.6% | 100.44 | 77.4% |
| 1024 | 128.79 | 98.30 | 76.3% | 98.87 | 76.8% |

fallback pool 状态：

```text
auto/split64 final max_next_slot=392, max_capacity=1142, exhausted=false
```

结论：即使 decode 一开始就处在 2048+ 的长 context，ByteV2 也没有自然接近 raw；
`split_k=64` 只有噪声级变化。

#### 更长 decode：`prompt_len=512, decode_len=4096`

命令：

```bash
CUDA_VISIBLE_DEVICES=2 VLLM_ENABLE_V1_MULTIPROCESSING=0 \
VLLM_BYTE_V2_USE_NATIVE_KERNELS=1 \
VLLM_BYTE_V2_SPARSE_FALLBACK_POOL_RATIO=0.10 \
VLLM_BYTE_V2_SPARSE_FALLBACK_POOL_MIN_BLOCKS=0 \
VLLM_BYTE_V2_DEFERRED_CACHE_UPDATE_ERROR_CHECK=1 \
.venv/bin/python benchmarks/benchmark_byte_v2_decode_e2e.py \
  --modes raw,byte_v2_compressed_only \
  --model /mnt/sda1/yxz/byte_v2/Meta-Llama-3-8B-Instruct \
  --prompt-len 512 --decode-lens 4096 \
  --batch-size 4 --num-runs 1 --warmup-decode-len 8 \
  --gpu-memory-utilization 0.80 \
  --output-json benchmarks/profiles/byv2_long_decode_p512_b4_d4096_r1_fb010_auto_vs_raw.json
```

`split_k=64`：

```bash
CUDA_VISIBLE_DEVICES=2 VLLM_ENABLE_V1_MULTIPROCESSING=0 \
VLLM_BYTE_V2_USE_NATIVE_KERNELS=1 \
VLLM_BYTE_V2_SPARSE_FALLBACK_POOL_RATIO=0.10 \
VLLM_BYTE_V2_SPARSE_FALLBACK_POOL_MIN_BLOCKS=0 \
VLLM_BYTE_V2_DEFERRED_CACHE_UPDATE_ERROR_CHECK=1 \
VLLM_BYTE_V2_DECODE_SPLIT_K=64 \
.venv/bin/python benchmarks/benchmark_byte_v2_decode_e2e.py \
  --modes byte_v2_compressed_only \
  --model /mnt/sda1/yxz/byte_v2/Meta-Llama-3-8B-Instruct \
  --prompt-len 512 --decode-lens 4096 \
  --batch-size 4 --num-runs 1 --warmup-decode-len 8 \
  --gpu-memory-utilization 0.80 \
  --output-json benchmarks/profiles/byv2_long_decode_p512_b4_d4096_r1_fb010_split64.json
```

结果：

| decode_len | raw tok/s | ByteV2 auto tok/s | auto/raw | ByteV2 split64 tok/s | split64/raw |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4096 | 126.31 | 97.39 | 77.1% | 97.73 | 77.4% |

fallback pool 状态：

```text
auto/split64 final max_next_slot=1032, max_capacity=1104, exhausted=false
```

结论：

- 10% fallback pool 在这组极长 decode 中接近满载，但没有耗尽，因此性能差距不是
  fallback pool exhaustion 造成的。
- 更长 decode 没有让 ByteV2 逼近 raw，反而相对比例低于 `decode_len=2048` 的
  约 84%。
- 固定 `split_k=64` 对 d4096 只有约 0.34% 提升，仍在噪声/微小收益范围内，不值得
  保留为默认策略。

#### 本实验结论

单纯增加 decode length，或在 E2E 中强制更大 split-K，不能让当前 ByteV2 超过 raw
vLLM。后续优化应集中在有效 split 的 stage1 kernel 结构，而不是继续调 split-K
heuristic：

1. 降低 K/V compressed decode 指令数和 metadata 检查成本。
2. 重新设计 shared layout / WMMA load pattern，减少 shared bank conflict 和
   register pressure。
3. 降低 split partial 写回和 reduce 扫描固定 `num_kv_splits` 的成本。
4. 若继续使用大 split-K，需要做真正的分段 graph/split profile，而不是固定覆盖
   `VLLM_BYTE_V2_DECODE_SPLIT_K=64`。

## Step 31：当前版本 vs raw vLLM 的 measured decode profile

时间：2026-06-11。

目的：

- 在继续设计 kernel 结构优化前，重新 profile 当前版本和 raw vLLM，确认差距主要
  落在哪个环节。
- 避免基于旧版本 profile 继续优化已经不再是瓶颈的 prefill/cache-update 路径。

### 环境与注意事项

`/home/yxz/.cache/vllm/torch_compile_cache` 所在文件系统已满，因此本轮 profile 将
cache 临时切到工作盘：

```text
VLLM_CACHE_ROOT=/mnt/sda1/yxz/byte_v2/vllm/.cache/vllm_profile
XDG_CACHE_HOME=/mnt/sda1/yxz/byte_v2/vllm/.cache/xdg
CUDA_CACHE_PATH=/mnt/sda1/yxz/byte_v2/vllm/.cache/cuda
VLLM_NO_USAGE_STATS=1
```

公共配置：

- GPU0，A40。
- 模型：`/mnt/sda1/yxz/byte_v2/Meta-Llama-3-8B-Instruct`。
- `prompt_len=512`，`batch_size=4`。
- prefix cache on，warmup 后重复同一 prompt，因此 measured range 代表
  prefix cache 命中后的 decode 主路径。
- raw：`kv_cache_dtype=auto`。
- ByteV2：`kv_cache_dtype=byte_v2`，compressed-only，10% fallback pool。
- CUDA graph on。

### E2E 结果

`decode_len=256`，普通 CUDA graph trace：

| mode | output tok/s | elapsed |
| --- | ---: | ---: |
| raw | 133.30 | 7.682s |
| ByteV2 compressed-only | 119.93 | 8.538s |

`decode_len=128`，`--cuda-graph-trace=node` 展开 CUDA graph 节点：

| mode | output tok/s | elapsed |
| --- | ---: | ---: |
| raw | 133.02 | 3.849s |
| ByteV2 compressed-only | 117.37 | 4.362s |

### Profile 命令

普通 graph trace：

```bash
CUDA_VISIBLE_DEVICES=0 VLLM_ENABLE_V1_MULTIPROCESSING=0 \
VLLM_NO_USAGE_STATS=1 \
VLLM_CACHE_ROOT=/mnt/sda1/yxz/byte_v2/vllm/.cache/vllm_profile \
XDG_CACHE_HOME=/mnt/sda1/yxz/byte_v2/vllm/.cache/xdg \
CUDA_CACHE_PATH=/mnt/sda1/yxz/byte_v2/vllm/.cache/cuda \
VLLM_BYTE_V2_USE_NATIVE_KERNELS=1 \
VLLM_BYTE_V2_SPARSE_FALLBACK_POOL_RATIO=0.10 \
VLLM_BYTE_V2_SPARSE_FALLBACK_POOL_MIN_BLOCKS=0 \
VLLM_BYTE_V2_DEFERRED_CACHE_UPDATE_ERROR_CHECK=1 \
nsys profile --force-overwrite=true --trace=cuda,nvtx,cublas \
  --sample=none --cpuctxsw=none --cuda-event-trace=false \
  --output benchmarks/profiles/nsys_current_raw_p512_b4_d256 \
  .venv/bin/python benchmarks/benchmark_byte_v2_decode_e2e.py \
    --child-mode raw \
    --model /mnt/sda1/yxz/byte_v2/Meta-Llama-3-8B-Instruct \
    --prompt-len 512 --decode-lens 256 \
    --batch-size 4 --num-runs 1 --warmup-decode-len 8 \
    --gpu-memory-utilization 0.80 \
    --output-json benchmarks/profiles/nsys_current_raw_p512_b4_d256.json
```

ByteV2 版本增加：

```text
VLLM_BYTE_V2_COMPRESSED_ONLY_CACHE=1
VLLM_BYTE_V2_ENABLE_SPARSE_FALLBACK_POOL=1
--child-mode byte_v2_compressed_only
```

node 级 graph trace 只把 `--decode-lens` 改为 `128`，并增加：

```text
--cuda-graph-trace=node
```

stats 导出：

```bash
nsys stats --force-overwrite=true \
  --filter-nvtx byte_v2_bench_measured \
  --report cuda_gpu_kern_sum:base,cuda_gpu_sum,cuda_api_sum,\
nvtx_gpu_proj_sum,cuda_kern_exec_sum:base \
  --format csv \
  --output benchmarks/profiles/<prefix> \
  benchmarks/profiles/<prefix>.nsys-rep
```

### node 级 kernel 分类结果

`decode_len=128`，measured range 内 GPU kernel time：

| category | raw time | raw share | ByteV2 time | ByteV2 share | delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| weight GEMM/GEMV | 3669.8 ms | 95.5% | 3687.2 ms | 87.5% | +17.4 ms |
| raw FlashAttention decode | 93.3 ms | 2.4% | N/A | N/A | N/A |
| ByteV2 decode stage1 | N/A | N/A | 419.1 ms | 9.9% | +419.1 ms |
| ByteV2 decode reduce | N/A | N/A | 15.5 ms | 0.4% | +15.5 ms |
| raw KV cache update | 9.8 ms | 0.3% | N/A | N/A | N/A |
| ByteV2 decode cache update | N/A | N/A | 40.2 ms | 1.0% | +40.2 ms |
| Triton model fused kernels | 41.0 ms | 1.1% | 39.5 ms | 0.9% | -1.5 ms |
| sampling/logits | 3.6 ms | 0.1% | 3.7 ms | 0.1% | +0.1 ms |
| small torch/scheduler kernels | 23.6 ms | 0.6% | 7.9 ms | 0.2% | -15.7 ms |
| total GPU kernel time | 3841.2 ms | 100% | 4214.5 ms | 100% | +373.2 ms |

更直观地看 attention/cache update 对比：

| component | raw | ByteV2 | delta |
| --- | ---: | ---: | ---: |
| decode attention | FlashAttention `93.3 ms` | stage1 + reduce `434.6 ms` | +341.3 ms |
| decode KV append/cache update | `reshape_and_cache_flash` `9.8 ms` | append/init/error kernels `40.2 ms` | +30.4 ms |

因此 `decode_len=128` 下，ByteV2 比 raw 多出的 GPU kernel time 约 373ms，其中
decode attention 本身解释了约 91% 的 GPU 差距；cache update 解释了约 8%。

### 普通 graph trace 交叉验证

`decode_len=256`，不展开 CUDA graph node 时，profile 只显示 graph-level 关键节点，
但结论一致：

| category | raw time | ByteV2 time | delta |
| --- | ---: | ---: | ---: |
| weight GEMM/GEMV | 496.2 ms | 496.2 ms | ~0 |
| ByteV2 decode stage1 | N/A | 28.9 ms | +28.9 ms |
| ByteV2 decode reduce | N/A | 0.7 ms | +0.7 ms |
| raw FlashAttention decode | 1.3 ms | N/A | N/A |

该 trace 因默认 CUDA graph 粒度是 graph-level，不能用于精确拆每层耗时，只作为
交叉验证。真正的逐层判断以 node 级 trace 为准。

### 结论

当前版本已经不再是旧 profile 中的 PyTorch prefill fallback 主导。prefix-cache 命中
后的 decode 主路径中：

1. **最大差距是 ByteV2 decode attention stage1。**
   raw FlashAttention decode 在 d128 measured range 中约 93ms，而 ByteV2
   stage1+reduce 约 435ms，慢约 4.7x。
2. **decode cache update 是第二瓶颈，但量级小很多。**
   raw `reshape_and_cache_flash` 约 9.8ms，ByteV2 append/cache-update 约 40.2ms。
3. **权重 GEMM/GEMV 基本相同。**
   两边都是约 3.67s，是总时间大头，但不是 ByteV2 相对 raw 的主要差距。
4. **sampling、scheduler metadata、小 PyTorch kernel 不是主要差距。**

### 后续优化优先级

1. 优先优化 `byte_v2_paged_decode_attention_gqa_wmma_split_stage1_kernel`：
   - compressed no-fallback hot kernel。
   - 降低 K/V decode 指令数。
   - encode-side MMA-friendly payload layout。
   - shared layout / WMMA load pattern。
   - 真正 warp-specialized producer/consumer double buffer。
2. 第二优先级优化 decode append/cache update：
   - 合并 `init_decode_append_result`、`decode_append_cache`、
     `record_deferred_cache_update_error` 的 per-layer/per-token overhead。
   - 减少 validation/error 相关 D2H/D2D 小操作。
3. 暂不优先优化：
   - 权重 GEMM/GEMV。
   - sampling/logits。
   - prefill direct encode，当前 measured decode 主路径里只约 1.3ms。

结果文件：

- `benchmarks/profiles/nsys_current_raw_p512_b4_d256.nsys-rep`
- `benchmarks/profiles/nsys_current_bytev2_p512_b4_d256.nsys-rep`
- `benchmarks/profiles/nsys_node_raw_p512_b4_d128.nsys-rep`
- `benchmarks/profiles/nsys_node_bytev2_p512_b4_d128.nsys-rep`
- `benchmarks/profiles/nsys_node_raw_p512_b4_d128_cuda_gpu_kern_sum_nvtx=byte_v2_bench_measured_base.csv`
- `benchmarks/profiles/nsys_node_bytev2_p512_b4_d128_cuda_gpu_kern_sum_nvtx=byte_v2_bench_measured_base.csv`

## Step 32：验证压缩 KV 读取收益能否显现

时间：2026-06-11。

目的：

- 单独验证 ByteV2 page 更小是否真的减少 HBM/global load bytes。
- 用 read-only microbenchmark 排除 ByteV2 decode、softmax、reduce 和 scheduler
  开销，只看 KV bytes 变小带来的读取时间变化。
- 用 E2E capacity pressure 看压缩 cache 的容量收益，以及当前 sparse fallback pool
  是否会先成为瓶颈。

### 32.1 NCU：只读 raw block vs compressed page

新增实验脚本：

```text
benchmarks/kernels/benchmark_byte_v2_kv_read_microbench.py
```

该脚本只做 int32 streaming read + per-program sum，不做 ByteV2 解码，也不做
attention。Llama-3 8B GQA 形状：

| item | bytes |
| --- | ---: |
| raw K/V block | 65,536 |
| ByteV2 compressed page | 49,424 |
| compressed/raw | 0.754 |

NCU 命令：

```bash
CUDA_VISIBLE_DEVICES=0 VLLM_NO_USAGE_STATS=1 \
VLLM_CACHE_ROOT=/mnt/sda1/yxz/byte_v2/vllm/.cache/vllm_profile \
XDG_CACHE_HOME=/mnt/sda1/yxz/byte_v2/vllm/.cache/xdg \
CUDA_CACHE_PATH=/mnt/sda1/yxz/byte_v2/vllm/.cache/cuda \
ncu --csv --force-overwrite --target-processes all \
  --kernel-name regex:stream_i32_sum --launch-skip 0 --launch-count 2 \
  --metrics gpu__time_duration.sum,dram__bytes_read.sum,\
l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum \
  --log-file benchmarks/profiles/ncu_kv_read_microbench_b8192.csv \
  .venv/bin/python benchmarks/kernels/benchmark_byte_v2_kv_read_microbench.py \
    --num-blocks 8192 --read-block-bytes 4096 \
    --warmup 0 --iters 1 \
    --output-json benchmarks/profiles/ncu_kv_read_microbench_b8192.json
```

NCU 结果：

| launch | global load bytes | DRAM read bytes | GPU time |
| --- | ---: | ---: | ---: |
| raw | 536,870,912 | 573,124,992 | 850,720 ns |
| compressed | 404,881,408 | 432,226,944 | 642,976 ns |
| compressed/raw | 0.754 | 0.754 | 0.756 |

结论：

- 在只读 streaming 场景下，compressed page 的 global load bytes 与格式大小一致，
  约为 raw 的 75.4%。
- NCU 看到的 DRAM read bytes 和 GPU time 也约为 raw 的 75.4%-75.6%。
- 这证明“压缩 KV 可以减少 HBM 读取时间”这个前提成立，但只在读取本身成为瓶颈、
  且解码/attention 额外指令没有盖过收益时成立。

### 32.2 Read-only microbenchmark：多 read tile 大小

命令：

```bash
CUDA_VISIBLE_DEVICES=0 VLLM_NO_USAGE_STATS=1 \
VLLM_CACHE_ROOT=/mnt/sda1/yxz/byte_v2/vllm/.cache/vllm_profile \
XDG_CACHE_HOME=/mnt/sda1/yxz/byte_v2/vllm/.cache/xdg \
CUDA_CACHE_PATH=/mnt/sda1/yxz/byte_v2/vllm/.cache/cuda \
.venv/bin/python benchmarks/kernels/benchmark_byte_v2_kv_read_microbench.py \
  --num-blocks 8192 --read-block-bytes 1024,2048,4096 \
  --warmup 20 --iters 100 \
  --output-json benchmarks/profiles/bytev2_kv_read_microbench_b8192.json
```

结果：

| read block bytes | raw time | compressed time | time ratio | raw GB/s | compressed GB/s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 0.8796 ms | 0.6666 ms | 0.758 | 610.3 | 607.4 |
| 2048 | 0.8643 ms | 0.6564 ms | 0.759 | 621.2 | 616.9 |
| 4096 | 0.8632 ms | 0.6543 ms | 0.758 | 622.0 | 618.8 |

结论：

- 只读 microbenchmark 的 compressed/raw time ratio 稳定在 0.758 左右，
  基本等于 0.754 的 bytes ratio。
- 因此，ByteV2 的容量/带宽收益在纯读场景中是可测的；当前 E2E 没体现为
  吞吐优势，是因为真实 decode kernel 还包含大量 ByteV2 解码、metadata 检查、
  split-K partial/reduce 和 shared/register 数据搬运。

### 32.3 E2E capacity pressure

#### 短 decode，高 max_model_len

配置：

```text
model=/mnt/sda1/yxz/byte_v2/Meta-Llama-3-8B-Instruct
prompt_len=512
decode_len=16
batch_size=20
max_model_len=8192
max_num_batched_tokens=10560
gpu_memory_utilization=0.80
prefix cache on
```

结果：

| mode | KV cache size | max concurrency | elapsed | output tok/s |
| --- | ---: | ---: | ---: | ---: |
| raw | 165,984 tokens | 20.26x | 0.566 s | 564.9 |
| ByteV2 3% pool | 200,224 tokens | 24.44x | 3.518 s | 91.0 |

结论：

- ByteV2 在相同 GPU memory utilization 下提供约 1.206x token capacity。
- 该配置的 measured run prefix-cache 命中，且 decode 很短，因此吞吐主要反映
  batch=20 短 decode 下 ByteV2 attention/cache-update overhead，不能证明容量优势。

#### 长 prompt，高 batch，prefix cache off

配置：

```text
prompt_len=2048
decode_len=1
batch_size=82
max_model_len=2065
max_num_batched_tokens=8192
prefix cache off
```

结果：

| mode | KV cache size | max concurrency | status | elapsed | total tok/s |
| --- | ---: | ---: | --- | ---: | ---: |
| raw | 159,783 tokens | 77.38x | pass | 23.378 s | 7187.1 |
| ByteV2 3% pool | 199,145 tokens | 96.44x | fail | N/A | N/A |
| ByteV2 10% pool | 182,990 tokens | 88.62x | pass | 23.638 s | 7107.8 |

ByteV2 3% pool 失败信息：

```text
Byte-v2 native prefill direct cache update failed:
sparse fallback pool exhausted
error_code=7, fallback_pool_capacity=377
```

结论：

- 主 compressed KV cache 的 capacity 收益存在：3% pool 下约 1.246x，
  10% pool 下约 1.145x。
- 3% sparse fallback pool 在真实 Llama-3 8B、p2048/b82/prefix-off 的
  lossless prefill 下会先耗尽，导致压缩主 cache 的容量收益无法转化为可用 E2E
  容量。
- 10% pool 可以跑通该压力配置，但容量收益下降到约 1.145x，且总吞吐与 raw
  基本持平略低。这说明该场景仍主要由 prefill 计算和 ByteV2 encode/fallback
  维护开销决定，不是单纯 HBM 读带宽瓶颈。
- raw 在 `batch_size > max concurrency` 时仍能跑通，是因为 chunked prefill
  和 `decode_len=1` 允许 scheduler 分批处理并释放请求；它不是“所有请求都以
  2048+decode 长度长期 resident”的硬容量测试。若要做严格 resident capacity
  测试，需要更长 decode、禁用 prefix cache，并控制 scheduler 让更多请求同时
  进入 decode resident set。

### 本轮总判断

压缩 KV 的 HBM 读取收益是存在且可测的：

```text
pure read compressed/raw time ~= 0.758
pure read compressed/raw bytes ~= 0.754
```

但当前 E2E 尚不能靠这部分超过 raw vLLM，主要原因是：

1. ByteV2 decode attention stage1 的解码指令、metadata 检查、shared layout 和
   split-K reduce 开销仍显著大于 raw FlashAttention 的读带宽节省。
2. 高并发长 prompt 下，3% sparse fallback pool 会先成为 lossless ByteV2 的
   capacity 瓶颈。
3. 10% fallback pool 能提升可用性，但会吃掉一部分压缩 capacity，且不能解决
   decode kernel 本身落后 raw 的问题。

后续要让 KV 读取收益体现在 E2E，需要优先做两件事：

1. 继续压缩 decode hot path 的非 HBM 开销，使 kernel 更接近 memory-bound：
   fewer decode instructions、MMA-friendly payload layout、shared swizzle、
   更低成本的 split-K reduce。
2. 改 fallback 表达方式，让 3% 甚至更低 pool 在真实 Llama-3 KV 上不耗尽：
   element-level outlier、better tile fallback packing、或允许有界 lossy/outlier
   residual，而不是扩大 block/raw fallback pool。

## Step 33：下一轮结构性实验计划

时间：2026-06-11。

背景：

- Step 31 显示当前 decode measured range 中，ByteV2 多出的 GPU kernel time
  主要来自 decode attention stage1：raw FlashAttention decode 约 `93.3 ms`，
  ByteV2 stage1+reduce 约 `434.6 ms`。
- Step 32 证明 compressed KV 的纯 HBM read 收益存在：
  compressed/raw read time 约 `0.758x`，与 bytes ratio `0.754x` 基本一致。
- 因此当前问题不是“压缩是否能省读带宽”，而是 ByteV2 stage1 的
  decode/load 指令、metadata、shared/register 搬运和 split/reduce 开销把
  读带宽收益吃掉了。

### 已经不应重复优先尝试的方向

以下方向已有 A/B 结果，除非新的 NCU profile 给出相反证据，否则不再作为下一步
优先级：

- 单纯放大或微调默认 split-K：decode-only 有收益，但 E2E 不兑现。
- 默认开启 persistent partial workspace：E2E 收益在噪声内。
- 局部 decoder 微调：`vec4`、`pair_v2`、base prefetch、base warp broadcast、
  packed shared store 都未超过保留门槛。
- 当前 WMMA kernel 内的 CTA-local K/V double-buffer 或 warp-specialized prefetch：
  已退化。
- row-major K shared、K shared stride padding、raw-fallback-only 单独 kernel：
  已证明不划算。
- 仅调整语句顺序，例如 K-first/V-late：没有形成真正 overlap。

### 方案 1：payload/layout-v2 no-fallback microbench

目标：

- 只针对 compressed no-fallback hot path，验证新的 payload 物理布局是否能减少
  ByteV2 decode/load 指令和 long scoreboard stall。
- 首轮不改默认 ByteV2 page format，不进入生产路径。

实验边界：

```text
head_size = head_size_v = 128
num_kv_heads = 8
block_size = 16
fallback_pool = nullptr
fallback_tile_ids = nullptr
page status = compressed
```

实现方式：

1. 新增独立 microbench variant，不改现有默认 decoder：
   - 可复用 `byte_v2_decode_page_wmma_microbench`。
   - 增加 `variant` 参数，`variant=0` 为当前 layout，`variant=1` 为
     layout-v2。
2. layout-v2 先作为实验用 packed tensor，不要求与生产 `ByteV2PageLayout`
   完全一致：
   - header 保持 16B，方便复用状态/valid rows。
   - 每个 tile 仍保持 lossless `base + low + packed code` 信息量。
   - low/code 按 warp/lane 读取顺序重排，目标是让同一 warp 的 global load
     更连续，减少 byte/halfword load 指令和地址计算。
3. 只在 B2b microbench 中比较：
   - compressed page decode + current WMMA QK/PV。
   - 不接 softmax、split-K、fallback metadata。

保留标准：

- B2b `decode_page_wmma_microbench` 至少快 `5%`。
- NCU 中 integer/memory 指令和 long scoreboard stall 有同步下降。
- correctness 与 `variant=0` 对齐。

不保留标准：

- 只减少 bytes/read 但 kernel time 无收益。
- 指令下降但 occupancy/eligible warps 下降抵消收益。
- 需要污染默认 production page format 或默认 stage1 编译形态。

实验记录，2026-06-11：

- 尝试了 `pair-interleaved 3B` layout-v2：
  - 每个 tile 的总 payload 仍保持 386B，不改变压缩率。
  - 原 layout：`base, fallback, low[256], code[128]`。
  - 实验 layout：`base, fallback, {low0, low1, code}[128]`。
  - 目标是让每个 decoded pair 的 low/code 读取更邻近，减少 long scoreboard 或
    scattered load 成本。
- 实现范围：
  - 只接入 `byte_v2_decode_page_wmma_microbench` 的临时 `variant=1`。
  - 不接 production page format，不接默认 decode stage1。
  - correctness 通过后运行 B2b microbenchmark。
- 结果，A40/GPU0，`num_pages=256, num_kv_heads=8, kv_head=0`：

| repeat | current layout | pair-interleaved | delta |
| ---: | ---: | ---: | ---: |
| 32 | 304.13 us | 299.01 us | +1.68% |
| 128 | 1164.29 us | 1192.96 us | -2.46% |

- 记录文件：
  - `benchmarks/profiles/bytev2_step33_layoutv2_b2b_variant0_r32.json`
  - `benchmarks/profiles/bytev2_step33_layoutv2_b2b_variant1_r32.json`
  - `benchmarks/profiles/bytev2_step33_layoutv2_b2b_variant0_r128.json`
  - `benchmarks/profiles/bytev2_step33_layoutv2_b2b_variant1_r128.json`
- 结论：
  - 未达到 B2b `+5%` 保留门槛。
  - r128 退化说明 pair-interleaved 的 `pair_idx * 3` 地址计算和非 2/4 字节
    对齐访问没有稳定改善 decode/load 瓶颈。
  - 代码已回滚，不保留 op schema、decoder helper、benchmark variant 和新增测试。
  - 因 microbench 未达门槛，本轮不继续接入 independent cute stage1，也不跑 E2E。
  - 后续如果继续 payload layout，不能只做 3B 邻近重排；应考虑真正的
    warp/lane-aligned 4B group 或 encode-side 更高压缩率格式，但必须先接受
    bytes ratio 与 decode 指令之间的 tradeoff。

### 方案 2：independent cute stage1 接入 layout-v2

只有方案 1 在 microbench 中达到保留门槛后才做。

实现方式：

- 在现有独立 `byte_v2_paged_decode_attention_gqa_cute_split_stage1_kernel`
  中增加 layout-v2 decoder 分支。
- 通过独立 env 启用，例如：

```text
VLLM_BYTE_V2_DECODE_CUTE_STAGE1=1
VLLM_BYTE_V2_EXPERIMENTAL_PAYLOAD_LAYOUT=1
```

- 默认 `byte_v2_paged_decode_attention_gqa_wmma_split_stage1_kernel` 不受影响。
- 首版仍只支持 compressed-only no-fallback 固定形状。
- reduce kernel 继续复用现有实现，避免同时改两个变量。

验证：

```bash
CUDA_VISIBLE_DEVICES=<gpu> \
.venv/bin/python benchmarks/kernels/benchmark_byte_v2_decode_kernel.py \
  --batch-size 1 \
  --seq-len 1024,2048,4096 \
  --num-heads 32 --num-kv-heads 8 \
  --head-size 128 --head-size-v 128 \
  --split-k 16,64 \
  --fallback-ratio 0 \
  --num-runs 50 --warmup-runs 10 --skip-correctness
```

保留标准：

- decode-only 至少 `+8%`。
- E2E p512/p2048 至少 `+3%`。
- fallback=0 默认路径不退化。

### 方案 3：fallback 表达方式优化

Step 32 的 capacity pressure 显示，3% sparse fallback pool 在真实 Llama-3 8B
`p2048/b82/prefix-off` 下会耗尽。该问题和主 compressed cache 容量是两个独立约束。

下一轮不应简单把默认 pool 改回 10%，因为它会吃掉一部分 capacity。更合理的方向：

1. **element-level outlier list**
   - 大多数 tile 仍用 compressed payload。
   - 只有超出 16-exponent window 的少量元素写 outlier list。
   - decode 时先按 compressed payload 解码，再用 outlier 覆盖。
2. **better tile fallback packing**
   - tile fallback pool 不按 raw block 预算。
   - 用 compact tile slots + per-page/tile mask，降低 fallback pool 颗粒度。
3. **bounded lossy/residual format**
   - 在可接受精度范围内允许极少数 outlier 走 residual 或饱和近似。
   - 需要单独 accuracy/perplexity 验证，不能直接进入默认 lossless path。

验收目标：

- `prompt_len=2048, batch_size=82, prefix cache off, pool=3%` 跑通。
- KV capacity 仍保持 raw 的 `1.20x+`。
- p512/p2048 常规 E2E 不明显退化。

### 本轮执行顺序

1. 先做方案 1 的 microbench variant。
2. 若方案 1 无收益，立即回滚，不做方案 2。
3. 若方案 1 有收益，再接入 independent cute stage1，跑 decode-only 和 E2E。
4. fallback 表达方式优化作为下一条主线，不和 layout-v2 同时改，避免变量混在一起。

## Step 34：tile fallback outlier 分布统计

### 目的

Step 32/33 后，下一条主线转向 fallback 表达方式。这里先不改 production
CUDA decode，而是做 reference codec 和真实 KV 分布统计，回答两个问题：

1. 真实 Llama-3 8B 的 tile fallback 是否大多只是少数元素超出 exponent window。
2. 如果改成 element-level outlier list，理论上能把 raw tile fallback pool 降到什么量级。

### 代码改动

- 新增 `vllm/v1/attention/backends/byte_v2_outliers.py`：
  - `compress_byte_v2_tile_with_outliers()` /
    `decompress_byte_v2_tile_with_outliers()`，用于单 tile reference codec。
  - `byte_v2_tile_exponent_miss_counts()`，统计每个 tile 超出 16-exponent
    window 的元素数。
  - `estimate_byte_v2_outlier_storage_from_misses()`，估算不同
    `max_outliers_per_tile` 下的 outlier list bytes。
- `ByteV2AttentionImpl.get_tile_fallback_stats()` 不再只看 raw block fallback。
  它现在也会读取 `fallback_tile_ids` 指向的 tile fallback pool，统计真实 tile
  fallback 的 exponent miss 分布。
- `benchmarks/byte_v2_tile_fallback_stats.py` 合并 worker/layer 的
  `outlier_storage_estimates`，输出全局估算。
- 新增单测覆盖 reference codec、storage estimate、以及 stats 从 tile pool 读取
  raw tile。

### 验证命令

```bash
.venv/bin/python -m pytest \
  tests/v1/attention/test_byte_v2_outliers.py \
  tests/v1/attention/test_byte_v2_backend.py::test_byte_v2_tile_fallback_stats_read_tile_pool \
  -q

.venv/bin/python -m pytest \
  tests/v1/attention/test_byte_v2_backend.py::test_byte_v2_attention_impl_sparse_fallback_pool_min_blocks \
  tests/v1/attention/test_byte_v2_metadata.py \
  -q

.venv/bin/python -m ruff check \
  vllm/v1/attention/backends/byte_v2_outliers.py \
  vllm/v1/attention/backends/byte_v2_attn.py \
  vllm/v1/worker/worker_base.py \
  benchmarks/byte_v2_tile_fallback_stats.py \
  tests/v1/attention/test_byte_v2_outliers.py \
  tests/v1/attention/test_byte_v2_backend.py
```

结果：

- `6 passed`
- `6 passed`
- `All checks passed`

### 真实模型统计

公共配置：

```bash
CUDA_VISIBLE_DEVICES=2 \
VLLM_NO_USAGE_STATS=1 \
VLLM_BYTE_V2_COMPRESSED_ONLY_CACHE=1 \
VLLM_BYTE_V2_ENABLE_SPARSE_FALLBACK_POOL=1 \
VLLM_BYTE_V2_SPARSE_FALLBACK_POOL_RATIO=0.03 \
VLLM_BYTE_V2_SPARSE_FALLBACK_POOL_MIN_BLOCKS=0 \
VLLM_BYTE_V2_DEFERRED_CACHE_UPDATE_ERROR_CHECK=1 \
.venv/bin/python benchmarks/byte_v2_tile_fallback_stats.py \
  --model /mnt/sda1/yxz/byte_v2/Meta-Llama-3-8B-Instruct \
  --decode-len 1 \
  --batch-size 1 \
  --gpu-memory-utilization 0.45 \
  --disable-prefix-caching
```

记录文件：

- `benchmarks/profiles/bytev2_step34_outlier_stats_p1024_b1_d1_fixed.json`
- `benchmarks/profiles/bytev2_step34_outlier_stats_p2048_b1_d1_fixed.json`

| prompt_len | active layer-blocks | tile fallback tiles | tile fallback ratio | miss sum | mean miss/tile | max miss/tile | pool exhausted |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 1024 | 2048 | 4941 | 1.8848% | 5014 | 1.0148 | 2 | false |
| 2048 | 4096 | 9925 | 1.8930% | 10094 | 1.0170 | 2 | false |

miss 分布：

| prompt_len | bad tiles <= 1 miss | bad tiles <= 2 miss | bad tiles > 8 miss |
| ---: | ---: | ---: | ---: |
| 1024 | 4868 | 4941 | 0 |
| 2048 | 9756 | 9925 | 0 |

storage estimate：

| prompt_len | raw tile fallback bytes | max_outliers=1 bytes | max_outliers=1 / raw tile | max_outliers=2 bytes | max_outliers=2 / raw tile |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 2,529,792 | 56,848 | 2.25% | 19,983 | 0.79% |
| 2048 | 5,081,600 | 125,552 | 2.47% | 40,207 | 0.79% |

当前 estimator 仍按“每 layer 至少 1 个 raw-block slot 等价容量”向上取整，
所以 `equivalent_raw_block_slots` 对全局 outlier bytes 不敏感：

- p1024：`max_outliers=2` 等价 32 个 raw-block slot，pool ratio 1.5625%。
- p2048：`max_outliers=2` 等价 32 个 raw-block slot，pool ratio 0.78125%。

这不是 outlier list 的真实下限，而是当前 allocator 仍按 layer/block slot
分配的结果。若后续实现真正 byte-addressed outlier arena，理论额外容量应接近
`additional_bytes`，不应被 raw-block slot 粒度放大。

### 结论

1. 真实 Llama-3 8B p1024/p2048 下，当前 lossless ByteV2 的 fallback 已经主要是
   tile-level fallback，而不是 block-level raw fallback。
2. 发生 fallback 的 tile 中，几乎都是 1 个 outlier 元素，最多 2 个；没有发现
   `>8` misses 的 tile。
3. raw tile fallback 的 512B 粒度过粗。以 `max_outliers_per_tile=2` 估算，
   outlier list 只需要当前 raw tile fallback bytes 的约 0.79%。
4. 下一步不应继续扩大 `VLLM_BYTE_V2_SPARSE_FALLBACK_POOL_RATIO`，而应实现 compact
   outlier pool：
   - encode 阶段为 bad tile 写 compressed payload + outlier entries；
   - metadata 从 `fallback_tile_ids` 转为 per-page outlier offset/count 或 compact
     tile outlier directory；
   - decode 阶段 compressed hot path 先正常解码 tile，再按 outlier list 覆盖少量
     BF16 元素。

### 后续实现顺序

1. 增加 outlier arena allocator，仅作为 opt-in 实验，不替换当前 tile fallback
   默认路径。
2. 先做 cache update encode 端：bad tile 不写 raw tile fallback，而是写
   outlier entries；超过阈值的 tile 仍回退 raw tile。
3. 再做 decode 端：在 tile-level fastpath 中补 outlier overlay。第一版只支持
   `max_outliers_per_tile <= 2`。
4. 单测覆盖 encode/decode bit exact；E2E 先跑 p1024/p2048/b1，再跑 Step 32 中
   触发 3% pool exhaustion 的高并发长 prompt。
5. 保留门槛：
   - p2048/b1 correctness 不退化；
   - 3% pool 在原先耗尽场景跑通；
   - 常规 p512/p2048 E2E 不慢于当前 tile fallback 路径 3% 以上。

## Step 35：compact outlier arena allocator 预埋

### 本轮目标

先完成 Step 34 后续实现顺序中的第 1 步：增加 opt-in 的 compact outlier
arena allocation 和注册通路。当前不改 native CUDA cache update/decode 主路径，
因此默认推理行为不变。

### 新增 env

```text
VLLM_BYTE_V2_ENABLE_OUTLIER_ARENA=0
VLLM_BYTE_V2_OUTLIER_ARENA_ENTRIES_PER_BLOCK=0.0
VLLM_BYTE_V2_OUTLIER_ARENA_MIN_ENTRIES=0
VLLM_BYTE_V2_OUTLIER_MAX_PER_TILE=0
```

只有同时满足以下条件时，`Attention.get_kv_cache_spec()` 才会把 outlier arena
容量写入 `ByteV2FullAttentionSpec`：

- `VLLM_BYTE_V2_COMPRESSED_ONLY_CACHE=1`
- `VLLM_BYTE_V2_ENABLE_SPARSE_FALLBACK_POOL=1`
- `VLLM_BYTE_V2_ENABLE_OUTLIER_ARENA=1`
- `VLLM_BYTE_V2_OUTLIER_ARENA_ENTRIES_PER_BLOCK > 0`

### allocation layout

每个 layer 的 raw KV allocation 现在支持如下布局：

```text
[main ByteV2 pages]
[raw fallback pool]
[padding to int32 alignment]
[outlier arena: int32 entries]
[padding to int32 alignment]
[fallback_block_ids: int32[num_blocks]]
[fallback_next_slot: int32[1]]
[fallback_tile_ids: int32[num_blocks, tiles_per_block]]
[fallback_tile_next_slot: int32[1]]
[outlier_tile_meta: int32[num_blocks, tiles_per_block]]
[outlier_next_entry: int32[1]]
```

outlier entry 第一版按 4B 预留，后续 CUDA encode 可 pack 成：

```text
bits[7:0]   = tile 内元素 index
bits[23:8]  = raw BF16 bits
bits[31:24] = reserved
```

`outlier_tile_meta` 第一版只作为 `int32` slot，建议后续 pack：

```text
bits[7:0]   = outlier count
bits[30:8]  = arena offset
bit[31]     = 0
```

这样每个 tile 只需要一个 int32 metadata。无 outlier tile 使用 `-1`。

### 已改代码

- `vllm/envs.py`
  - 新增 outlier arena 三个 opt-in env。
- `vllm/v1/kv_cache_interface.py`
  - `ByteV2FullAttentionSpec` 新增：
    - `outlier_arena_entries_per_block`
    - `outlier_arena_min_entries`
    - `outlier_arena_entry_bytes`
    - `outlier_tile_meta_bytes`
    - `outlier_next_entry_bytes`
  - 新增 offset helper：
    - `sparse_fallback_pool_bytes()`
    - `outlier_arena_entries()`
    - `outlier_arena_bytes()`
    - `outlier_arena_start_bytes()`
    - `metadata_start_bytes()`
    - `outlier_metadata_bytes()`
- `vllm/model_executor/layers/attention/attention.py`
  - 将 env 传入 `ByteV2FullAttentionSpec`。
- `vllm/v1/worker/gpu/attn_utils.py`
- `vllm/v1/worker/gpu_model_runner.py`
  - 同步 raw allocation 切分：
    `main + raw fallback + outlier arena + metadata`。
  - 初始化 `outlier_tile_meta.fill_(-1)` 和 `outlier_next_entry.zero_()`。
- `vllm/v1/attention/backends/byte_v2_attn.py`
  - `ByteV2SparseFallbackPool` 增加可选 outlier tensors。
  - `_get_sparse_fallback_pool()` 支持 runtime allocation fallback。
  - `register_sparse_fallback_pool()` 支持 model runner 注册 arena。
  - `get_sparse_fallback_pool_stats()` 输出 outlier capacity/usage。
- `vllm/v1/worker/worker_base.py`
  - sparse fallback RPC 汇总 outlier capacity/usage。

### 验证

```bash
.venv/bin/python -m pytest \
  tests/v1/test_byte_v2_kv_cache_spec.py \
  tests/v1/attention/test_byte_v2_backend.py::test_byte_v2_attention_impl_sparse_fallback_pool_min_blocks \
  tests/v1/attention/test_byte_v2_backend.py::test_byte_v2_attention_impl_outlier_arena_pool \
  -q
```

结果：`15 passed`

```bash
.venv/bin/python -m pytest \
  tests/v1/attention/test_byte_v2_backend.py::test_byte_v2_tile_fallback_stats_read_tile_pool \
  tests/v1/attention/test_byte_v2_metadata.py \
  -q
```

结果：`6 passed`

```bash
.venv/bin/python -m ruff check \
  vllm/envs.py \
  vllm/v1/kv_cache_interface.py \
  vllm/model_executor/layers/attention/attention.py \
  vllm/v1/attention/backends/byte_v2_attn.py \
  vllm/v1/worker/gpu/attn_utils.py \
  vllm/v1/worker/gpu_model_runner.py \
  vllm/v1/worker/worker_base.py \
  benchmarks/byte_v2_tile_fallback_stats.py \
  tests/v1/test_byte_v2_kv_cache_spec.py \
  tests/v1/attention/test_byte_v2_backend.py
```

结果：`All checks passed`

### 当前状态

当前 Step 35 完成 allocator/metadata 预埋。Step 36 已进一步实现
prefill-direct native encode 的 opt-in 写入路径。Step 37 已完成 decode 主路径
overlay，因此在 `VLLM_BYTE_V2_OUTLIER_MAX_PER_TILE > 0` 且 arena tensor 存在时，
prefill-direct 写出的 compressed+outlier tile 可以在 native paged decode 中还原。

### 下一步

1. 补 continuation prefill / decode append / general touched-block compress 的
   outlier arena 写入或保真解压。
2. 增加 outlier arena exhausted 的显式 CUDA 单测。
3. 继续 profile outlier overlay 对 decode stage1 的额外开销。

## Step 36：compact outlier arena native prefill-direct encode

### 本轮目标

把 Step 35 预埋的 outlier arena 接入 native cache update 的 prefill-direct
fast path。目标是先减少真实 Llama KV 中“只有少量 outlier 的 tile”对 raw tile
fallback pool 的占用，但保持默认关闭，避免 decode overlay 未完成前影响 E2E
正确性。

### 设计

新增开关：

```text
VLLM_BYTE_V2_OUTLIER_MAX_PER_TILE=0
```

只有同时满足以下条件才启用：

- compressed-only cache；
- sparse fallback pool 和 tile fallback metadata 已启用；
- outlier arena tensors 已分配并传入 native op；
- `VLLM_BYTE_V2_OUTLIER_MAX_PER_TILE > 0`；
- 当前 cache update 命中 prefill-direct path。

prefill-direct encode 的 tile 分类改为三类：

1. `miss_count <= VLLM_BYTE_V2_LOSSY_MAX_MISSES_PER_TILE`
   - 仍走普通 compressed tile。
2. `lossy_max < miss_count <= VLLM_BYTE_V2_OUTLIER_MAX_PER_TILE`
   - 写 compressed payload；
   - window 外元素在 payload 中 clamp 到 base window；
   - raw BF16 bits + tile element index 写入 `outlier_arena`；
   - `outlier_tile_meta[block_id, tile_idx]` 记录 offset/count；
   - 不占 raw tile fallback pool。
3. `miss_count > VLLM_BYTE_V2_OUTLIER_MAX_PER_TILE` 或 arena 容量不足
   - 回退当前 raw tile fallback。

metadata pack：

```text
outlier entry int32:
bits[7:0]   = tile element index
bits[23:8]  = raw BF16 bits
bits[31:24] = reserved

outlier tile meta int32:
bits[7:0]   = outlier count
bits[30:8]  = arena offset
bit[31]     = 0
```

### 已改代码

- `vllm/envs.py`
  - 新增 `VLLM_BYTE_V2_OUTLIER_MAX_PER_TILE`。
- `vllm/_custom_ops.py`
- `vllm/v1/attention/backends/byte_v2_ops.py`
- `vllm/v1/attention/backends/byte_v2_attn.py`
- `csrc/libtorch_stable/ops.h`
- `csrc/libtorch_stable/torch_bindings.cpp`
  - `byte_v2_reshape_and_cache()` 参数链路新增：
    - `outlier_arena`
    - `outlier_tile_meta`
    - `outlier_next_entry`
- `csrc/libtorch_stable/cache_kernels.cu`
  - `byte_v2_prefill_direct_encode_blocks_kernel()` 支持 outlier arena。
  - native wrapper 校验 outlier tensor dtype/device/shape。
  - 新增 error code 10：`outlier arena exhausted`，用于后续 fail-closed
    reporting；当前 encode 分支优先回退 raw tile fallback。
- `tests/v1/attention/test_byte_v2_ops.py`
  - 新增 `test_byte_v2_native_outlier_arena_prefill_direct_cuda`。

### 验证

由于完整 `uv pip install -e . --torch-backend=auto` 会触发 flash-attn/MoE 等大量
无关 target，本轮手动重编并 relink 了 `_C_stable_libtorch.abi3.so`：

```bash
# 重新编译 C++ schema binding
ccache /usr/bin/c++ ... -c csrc/libtorch_stable/torch_bindings.cpp

# 重新编译 ByteV2 cache update CUDA TU
ccache /usr/local/cuda/bin/nvcc ... -c csrc/libtorch_stable/cache_kernels.cu

# 重新链接并复制到 vllm/_C_stable_libtorch.abi3.so
/usr/bin/c++ ... -shared -o _C_stable_libtorch.abi3.so ...
cp build/temp.linux-x86_64-cpython-312/_C_stable_libtorch.abi3.so \
  vllm/_C_stable_libtorch.abi3.so
```

CUDA 单测：

```bash
CUDA_VISIBLE_DEVICES=2 .venv/bin/python -m pytest \
  tests/v1/attention/test_byte_v2_ops.py::test_byte_v2_native_tile_fallback_pool_cuda \
  tests/v1/attention/test_byte_v2_ops.py::test_byte_v2_native_lossy_outlier_threshold_cuda \
  tests/v1/attention/test_byte_v2_ops.py::test_byte_v2_native_outlier_arena_prefill_direct_cuda \
  -q
```

结果：`3 passed`

### 当前状态

prefill-direct encode 已可把 small-outlier tile 写入 compact arena，验证中
`fallback_tile_next_slot` 保持 0，说明没有占 raw tile fallback pool。Step 37
进一步补上 native paged decode overlay。

未完成：

- continuation prefill / decode append / general touched-block compress path 仍未写
  outlier arena；涉及已带 outlier 的 partial page append 时仍需要额外保真处理。
- arena exhausted 当前会走 raw tile fallback；后续需要补显式 exhaustion 单测和
  stats 观测。

## Step 37：compact outlier arena decode overlay 和 E2E smoke

### 本轮目标

把 Step 36 写入的 compact outlier arena 接入 native paged decode，使
compressed payload 解码后能用 arena 中的 raw BF16 bits 覆盖 window 外元素，
从而可以打开 `VLLM_BYTE_V2_OUTLIER_MAX_PER_TILE` 做 E2E correctness smoke。

### 设计

- `byte_v2_paged_decode_attention()` 新增可选参数：
  - `outlier_arena: int32[num_entries]`
  - `outlier_tile_meta: int32[num_blocks, total_tiles]`
- native wrapper 校验两个 tensor 必须一起提供，且 dtype/device/shape/contiguous
  合法。
- generic element loader 在 compressed tile 路径中：
  - 先按 compressed payload 解码 BF16 bits；
  - 若 `outlier_tile_meta[physical_block, tile_idx] >= 0`，扫描该 tile 的少量
    outlier entry；
  - entry 的 `tile element index` 命中当前元素时返回 raw BF16 bits。
- split-K GQA WMMA 主路径中：
  - K/V tile 先解压到 shared memory；
  - 对 compressed tile 调 `byte_v2_overlay_*_tile_outliers_to_shared()`；
  - raw tile fallback 不做 overlay；
  - CUTE stage1 暂未支持 overlay，outlier arena 存在时自动禁用 CUTE stage1。

当前 overlay 只保证 native paged decode 主路径正确。`byte_v2_decompress_page_to_raw_block()`
仍没有读取 outlier arena，所以带 outlier 的 compressed page 若后续进入 partial-page
decode append 解压，仍可能丢失 outlier；这需要下一步单独处理。

### 已改代码

- `vllm/_custom_ops.py`
- `vllm/v1/attention/backends/byte_v2_ops.py`
- `vllm/v1/attention/backends/byte_v2_attn.py`
  - decode 参数链路新增 `outlier_arena` / `outlier_tile_meta`。
- `csrc/libtorch_stable/ops.h`
- `csrc/libtorch_stable/torch_bindings.cpp`
  - `_C_cache_ops.byte_v2_paged_decode_attention` schema 新增两个可选参数。
- `csrc/libtorch_stable/cache_kernels.cu`
  - 新增 outlier meta/entry unpack helper。
  - generic `byte_v2_load_kv_bits()` 支持 element-level overlay。
  - split-K WMMA stage1 的 tile fastpath 支持 shared-memory overlay。
- `tests/v1/attention/test_byte_v2_ops.py`
  - 新增 `test_byte_v2_native_outlier_arena_decode_overlay_cuda`。
- `tests/v1/attention/test_byte_v2_e2e.py`
  - `_generate_token_ids()` 支持通过
    `BYTE_V2_E2E_GPU_MEMORY_UTILIZATION` 覆盖 8B smoke 需要的显存比例。
- `benchmarks/benchmark_byte_v2_decode_e2e.py`
  - E2E benchmark JSON 记录 outlier arena 相关 env。

### 验证

native schema：

```bash
.venv/bin/python - <<'PY'
import torch
import vllm._C_stable_libtorch
print(torch.ops._C_cache_ops.byte_v2_paged_decode_attention._schemas)
PY
```

确认 schema 包含：

```text
Tensor? outlier_arena=None, Tensor? outlier_tile_meta=None
```

CUDA 单测：

```bash
CUDA_VISIBLE_DEVICES=2 .venv/bin/python -m pytest \
  tests/v1/attention/test_byte_v2_ops.py::test_byte_v2_native_outlier_arena_decode_overlay_cuda \
  -q
```

结果：`1 passed`

完整 ByteV2 ops/decode 回归：

```bash
CUDA_VISIBLE_DEVICES=2 .venv/bin/python -m pytest \
  tests/v1/attention/test_byte_v2_ops.py \
  tests/v1/attention/test_byte_v2_decode.py \
  -q
```

结果：`42 passed`

8B E2E correctness smoke：

```bash
CUDA_VISIBLE_DEVICES=2 \
BYTE_V2_RUN_E2E=1 \
BYTE_V2_E2E_MODEL=/mnt/sda1/yxz/byte_v2/Meta-Llama-3-8B-Instruct \
BYTE_V2_E2E_GPU_MEMORY_UTILIZATION=0.8 \
VLLM_BYTE_V2_USE_NATIVE_KERNELS=1 \
VLLM_BYTE_V2_COMPRESSED_ONLY_CACHE=1 \
VLLM_BYTE_V2_ENABLE_SPARSE_FALLBACK_POOL=1 \
VLLM_BYTE_V2_SPARSE_FALLBACK_POOL_RATIO=0.03 \
VLLM_BYTE_V2_SPARSE_FALLBACK_POOL_MIN_BLOCKS=512 \
VLLM_BYTE_V2_ENABLE_OUTLIER_ARENA=1 \
VLLM_BYTE_V2_OUTLIER_ARENA_ENTRIES_PER_BLOCK=4 \
VLLM_BYTE_V2_OUTLIER_ARENA_MIN_ENTRIES=4096 \
VLLM_BYTE_V2_OUTLIER_MAX_PER_TILE=1 \
VLLM_BYTE_V2_DECODE_PAGE_FASTPATH=1 \
VLLM_BYTE_V2_DECODE_TILE_FASTPATH=1 \
VLLM_BYTE_V2_DECODE_SPLIT_K=0 \
VLLM_BYTE_V2_PREFILL_DIRECT_SKIP_VALIDATION_SYNC=1 \
VLLM_BYTE_V2_DECODE_APPEND_SKIP_VALIDATION_SYNC=1 \
VLLM_BYTE_V2_DEFERRED_CACHE_UPDATE_ERROR_CHECK=1 \
.venv/bin/python -m pytest \
  tests/v1/attention/test_byte_v2_e2e.py::test_byte_v2_matches_raw_vllm_e2e_smoke \
  -q -s
```

结果：`1 passed`。raw 和 ByteV2 生成的 prompt/output token ids 一致。

8B E2E performance smoke：

```bash
CUDA_VISIBLE_DEVICES=2 \
VLLM_BYTE_V2_USE_NATIVE_KERNELS=1 \
VLLM_BYTE_V2_ENABLE_SPARSE_FALLBACK_POOL=1 \
VLLM_BYTE_V2_SPARSE_FALLBACK_POOL_RATIO=0.03 \
VLLM_BYTE_V2_SPARSE_FALLBACK_POOL_MIN_BLOCKS=512 \
VLLM_BYTE_V2_ENABLE_OUTLIER_ARENA=1 \
VLLM_BYTE_V2_OUTLIER_ARENA_ENTRIES_PER_BLOCK=4 \
VLLM_BYTE_V2_OUTLIER_ARENA_MIN_ENTRIES=4096 \
VLLM_BYTE_V2_OUTLIER_MAX_PER_TILE=1 \
VLLM_BYTE_V2_DECODE_PAGE_FASTPATH=1 \
VLLM_BYTE_V2_DECODE_TILE_FASTPATH=1 \
VLLM_BYTE_V2_DECODE_SPLIT_K=0 \
VLLM_BYTE_V2_PREFILL_DIRECT_SKIP_VALIDATION_SYNC=1 \
VLLM_BYTE_V2_DECODE_APPEND_SKIP_VALIDATION_SYNC=1 \
VLLM_BYTE_V2_DEFERRED_CACHE_UPDATE_ERROR_CHECK=1 \
.venv/bin/python benchmarks/benchmark_byte_v2_decode_e2e.py \
  --model /mnt/sda1/yxz/byte_v2/Meta-Llama-3-8B-Instruct \
  --modes raw,byte_v2_compressed_only \
  --prompt-len 1024 \
  --decode-lens 16,64 \
  --batch-size 1 \
  --num-runs 1 \
  --warmup-decode-len 4 \
  --gpu-memory-utilization 0.8 \
  --enforce-eager \
  --disable-prefix-caching \
  --output-json benchmarks/byte_v2_decode_e2e_llama3_8b_p1024_b1_outlier_arena_smoke.json
```

结果：

| mode | decode_len | output tok/s | elapsed |
| --- | ---: | ---: | ---: |
| raw | 16 | 27.72 | 0.577s |
| raw | 64 | 32.82 | 1.950s |
| ByteV2 compressed-only + outlier arena | 16 | 14.74 | 1.086s |
| ByteV2 compressed-only + outlier arena | 64 | 15.79 | 4.054s |

sparse/outlier stats：

- `workers_with_pool=1`
- `any_exhausted=false`
- `total_next_slot=224`
- `total_assigned_blocks=96`
- `total_outlier_next_entry=14604`
- `total_assigned_outlier_tiles=14604`
- `total_outlier_capacity=1649024`
- `any_outlier_exhausted=false`

### 当前结论

1. compact outlier arena 现在可以跑通 native prefill-direct encode + native paged
   decode overlay + 8B E2E correctness smoke。
2. p1024/b1/d16,d64 下没有 raw fallback pool 或 outlier arena exhaustion。
3. 性能仍明显低于 raw，主要因为 decode stage1 额外 overlay metadata 检查和 ByteV2
   解码开销叠加；该实验目标是 correctness/容量表达跑通，不是性能优化完成。

### 未完成

- outlier-aware `byte_v2_decompress_page_to_raw_block()`，用于 partial-page append
  或未来 continuation update 保真。
- continuation prefill / decode append / general touched-block compress path 写入
  compact outlier arena。
- outlier arena exhausted 的显式单测。
- overlay 开销 profile，以及只在存在 outlier tile 的 block/page 上启用 overlay 的
  compact metadata fast path。

## Step 38：compact outlier arena E2E 开销隔离

状态：已实验，暂不作为默认性能路径；保持 opt-in。

### 目的

Step 37 已证明 compact outlier arena 的 native prefill-direct encode、native paged
decode overlay 和 8B E2E correctness 可以跑通。但 E2E 性能仍低于 raw，因此本轮隔离：

1. 不开 outlier arena 的 ByteV2 compressed-only 基线；
2. 打开 arena 但 `VLLM_BYTE_V2_OUTLIER_MAX_PER_TILE=0`，测纯 metadata/路径切换开销；
3. 打开 arena 且 `VLLM_BYTE_V2_OUTLIER_MAX_PER_TILE=1`，测 raw tile fallback
   被 outlier overlay 替代后的收益。

### 实验配置

共同配置：

```bash
CUDA_VISIBLE_DEVICES=2 \
VLLM_BYTE_V2_USE_NATIVE_KERNELS=1 \
VLLM_BYTE_V2_ENABLE_SPARSE_FALLBACK_POOL=1 \
VLLM_BYTE_V2_SPARSE_FALLBACK_POOL_RATIO=0.03 \
VLLM_BYTE_V2_SPARSE_FALLBACK_POOL_MIN_BLOCKS=512 \
VLLM_BYTE_V2_DECODE_PAGE_FASTPATH=1 \
VLLM_BYTE_V2_DECODE_TILE_FASTPATH=1 \
VLLM_BYTE_V2_DECODE_SPLIT_K=0 \
VLLM_BYTE_V2_PREFILL_DIRECT_SKIP_VALIDATION_SYNC=1 \
VLLM_BYTE_V2_DECODE_APPEND_SKIP_VALIDATION_SYNC=1 \
VLLM_BYTE_V2_DEFERRED_CACHE_UPDATE_ERROR_CHECK=1 \
.venv/bin/python benchmarks/benchmark_byte_v2_decode_e2e.py \
  --model /mnt/sda1/yxz/byte_v2/Meta-Llama-3-8B-Instruct \
  --prompt-len 1024 \
  --decode-lens 16,64 \
  --batch-size 1 \
  --num-runs 1 \
  --warmup-decode-len 4 \
  --gpu-memory-utilization 0.8 \
  --enforce-eager \
  --disable-prefix-caching
```

输出文件：

- `benchmarks/byte_v2_decode_e2e_llama3_8b_p1024_b1_no_arena_compare.json`
- `benchmarks/byte_v2_decode_e2e_llama3_8b_p1024_b1_arena_max0_compare.json`
- `benchmarks/byte_v2_decode_e2e_llama3_8b_p1024_b1_arena_max1_compare.json`

### 结果

| mode | arena | max outlier/tile | decode_len | output tok/s | elapsed |
| --- | ---: | ---: | ---: | ---: | ---: |
| raw | 0 | 0 | 16 | 27.69 | 0.578s |
| raw | 0 | 0 | 64 | 32.81 | 1.951s |
| ByteV2 compressed-only | 0 | 0 | 16 | 14.63 | 1.094s |
| ByteV2 compressed-only | 0 | 0 | 64 | 15.63 | 4.094s |
| ByteV2 compressed-only | 1 | 0 | 16 | 14.55 | 1.100s |
| ByteV2 compressed-only | 1 | 0 | 64 | 7.60 | 8.420s |
| ByteV2 compressed-only | 1 | 1 | 16 | 14.56 | 1.099s |
| ByteV2 compressed-only | 1 | 1 | 64 | 14.75 | 4.338s |

fallback/outlier stats：

| config | decode_len | raw tile fallbacks | outlier tiles | outlier entries | exhausted |
| --- | ---: | ---: | ---: | ---: | --- |
| no arena | 16 | 9882 | 0 | 0 | false |
| no arena | 64 | 15491 | 0 | 0 | false |
| arena, max=0 | 16 | 9882 | 0 | 0 | false |
| arena, max=0 | 64 | 15529 | 0 | 0 | false |
| arena, max=1 | 16 | 146 | 9736 | 9736 | false |
| arena, max=1 | 64 | 903 | 14604 | 14604 | false |

### 结论

1. `max_outlier_per_tile=1` 能显著减少 raw tile fallback：
   p1024/b1/d64 从约 1.55 万个 raw tile fallback 降到 903 个。
2. 这个容量表达收益没有转化为 E2E 性能收益。d64 从 no-arena 的 15.63 tok/s
   变成 arena max=1 的 14.75 tok/s，略有退化。
3. `arena=1,max=0` 的 d64 掉到 7.60 tok/s，说明 arena 打开后走到的
   decode overlay/metadata 路径本身存在明显额外开销；即使没有 outlier entry，也会影响
   长 decode。
4. 当前不应默认启用 compact outlier arena 作为性能优化。它仍可作为 opt-in 的
   correctness/capacity 实验路径保留。

### 下一步

1. 对 arena decode overlay 做 Nsight profile，重点看：
   - outlier metadata load/check 指令数；
   - CUTE stage1 被禁用后的 stage1 kernel 差距；
   - raw tile fallback path 和 compressed+overlay path 的 global load sectors；
   - register、shared bank conflict、eligible warps。
2. 设计只在存在 outlier tile 的 block/page 上启用 overlay 的 fast path：
   - encode 阶段生成 per-block/per-page `has_outlier` compact bitmap；
   - decode hot path 先走 no-overlay kernel；
   - 只有 `has_outlier` 的 page/tile 走 overlay variant，避免每个 tile 都查
     `outlier_tile_meta`。
3. CUTE/independent stage1 若要支持 outlier arena，必须把 overlay 融入 shared
   layout，而不是在通用 loader 中逐元素检查。
4. 如果 profile 显示 overlay/check 成本无法低于 raw tile fallback 成本，则停止
   性能方向，只保留 outlier arena 作为容量实验，不继续扩大这条优化线。

## Step 39：compact outlier arena encode 开销优化

状态：已实现并保留。

### 背景

Step 38 的 E2E 结果显示 compact outlier arena 能显著减少 raw tile fallback，
但没有转化为稳定性能收益。本轮先用 decode-only benchmark + Nsight Systems
隔离 arena 路径的 kernel 时间，避免直接从高方差 E2E 判断。

### profile 结果

命令：

```bash
CUDA_VISIBLE_DEVICES=2 nsys profile --force-overwrite=true \
  --trace=cuda,nvtx \
  --output benchmarks/profiles/nsys_bytev2_decode_kernel_p1024_arena_max1 \
  .venv/bin/python benchmarks/kernels/benchmark_byte_v2_decode_kernel.py \
    --batch-size 1 \
    --seq-len 1024 \
    --split-k 16 \
    --fallback-ratio 0.125 \
    --fallback-pattern single_outlier_per_k_tile \
    --tile-fallback-pool \
    --outlier-arena-entries-per-block 4 \
    --outlier-arena-min-entries 4096 \
    --outlier-max-per-tile 1 \
    --num-runs 20 \
    --warmup-runs 5 \
    --skip-correctness \
    --variant page_fastpath \
    --tile-fastpath-mode on
```

关键 kernel 时间：

| config | stage1 avg | split-reduce avg | prefill-direct encode |
| --- | ---: | ---: | ---: |
| no arena | 63.0 us | 2.93 us | 58.0 us |
| arena max=1, 优化前 | 65.0 us | 2.92 us | 257.1 us |

结论：

1. decode overlay 本身只让 stage1 增加约 2 us/launch，不是主要瓶颈。
2. 主要退化在 `byte_v2_prefill_direct_encode_blocks_kernel()`：打开 arena 后
   encode 从 58 us 增加到 257 us。
3. 原因是 arena encode 对每个 tile 使用 256-bin histogram 找最佳 exponent
   window；即使绝大多数 tile 本来可以直接压缩，也会多做 histogram/atomic。

### 修改 1：compressible tile fast filter

在 arena/lossy 路径先做 warp-level exponent min/max：

- 如果 `tile_max - tile_min <= 15`，直接写 `tile_bases`，不建 histogram。
- 只有超出当前 ByteV2 16-exponent window 的 tile 才进入旧 histogram
  best-window/outlier 判断。

结果：

| config | prefill-direct encode |
| --- | ---: |
| arena max=1, 优化前 | 257.1 us |
| min/max fast filter | 236.2 us |

收益只有约 8%。说明 outlier tile 上的 histogram、lane0 串行 outlier entry
写入仍然很重。

### 修改 2：single-outlier tile fast path

真实统计和 synthetic case 中，大多数 arena tile 都是 1 个 outlier。因此新增
仅覆盖以下条件的 fast path：

```text
has_outlier_arena == true
lossy_max_misses_per_tile == 0
outlier_max_per_tile == 1
```

做法：

1. min/max 判定发现 tile 超出 16-exponent window 后，不建 histogram。
2. 分别测试两个候选 window：
   - `base = tile_min`，统计高端 miss 数；
   - `base = tile_max - 15`，统计低端 miss 数。
3. 如果任一候选 window 只有 1 个 miss，则把该 tile 记录为 outlier tile；
   如果都超过 1 个 miss，仍走 raw tile fallback。
4. outlier entry 写入从 lane0 串行扫描 256 个元素，改为 warp 并行查找唯一
   outlier，再由 lane0 写 compact arena entry。
5. 其他情况仍走旧 histogram 路径，保持 `outlier_max_per_tile > 1` 和 lossy
   实验的通用性。

### 验证

CUDA 单测：

```bash
CUDA_VISIBLE_DEVICES=2 .venv/bin/python -m pytest \
  tests/v1/attention/test_byte_v2_ops.py::test_byte_v2_native_outlier_arena_prefill_direct_cuda \
  tests/v1/attention/test_byte_v2_ops.py::test_byte_v2_native_outlier_arena_decode_overlay_cuda \
  tests/v1/attention/test_byte_v2_ops.py::test_byte_v2_native_tile_fallback_pool_cuda \
  -q -s
```

结果：`3 passed`。

decode-only benchmark：

```bash
CUDA_VISIBLE_DEVICES=2 \
.venv/bin/python benchmarks/kernels/benchmark_byte_v2_decode_kernel.py \
  --batch-size 1 \
  --seq-len 1024 \
  --split-k 16 \
  --fallback-ratio 0.125 \
  --fallback-pattern single_outlier_per_k_tile \
  --tile-fallback-pool \
  --outlier-arena-entries-per-block 4 \
  --outlier-arena-min-entries 4096 \
  --outlier-max-per-tile 1 \
  --num-runs 100 \
  --warmup-runs 20 \
  --skip-correctness \
  --variant page_fastpath \
  --tile-fastpath-mode on \
  --output-json benchmarks/profiles/bytev2_decode_kernel_p1024_b1_single_outlier_tiles_arena_max1_after_single_outlier_fastpath.json
```

结果：

```text
median_us=1043.97
p90_us=1072.13
fallback_tiles=0/2048
outlier_tiles=512
outlier_entries=512/4096
```

Nsight Systems：

| config | stage1 avg | split-reduce avg | prefill-direct encode |
| --- | ---: | ---: | ---: |
| no arena | 63.0 us | 2.93 us | 58.0 us |
| arena max=1, 优化前 | 65.0 us | 2.92 us | 257.1 us |
| min/max fast filter | 65.2 us | 2.93 us | 236.2 us |
| single-outlier fast path | 64.9 us | 2.93 us | 75.8 us |

single-outlier fast path 把 arena encode 开销从 257.1 us 降到 75.8 us，
接近 no-arena 的 58.0 us；decode stage1 没有明显退化。

8B E2E smoke：

```bash
CUDA_VISIBLE_DEVICES=2 \
VLLM_BYTE_V2_USE_NATIVE_KERNELS=1 \
VLLM_BYTE_V2_DECODE_PAGE_FASTPATH=1 \
VLLM_BYTE_V2_DECODE_TILE_FASTPATH=1 \
VLLM_BYTE_V2_DECODE_SPLIT_K=16 \
VLLM_BYTE_V2_DECODE_PARALLEL_REDUCE=1 \
VLLM_BYTE_V2_ENABLE_SPARSE_FALLBACK_POOL=1 \
VLLM_BYTE_V2_SPARSE_FALLBACK_POOL_RATIO=0.03 \
VLLM_BYTE_V2_ENABLE_OUTLIER_ARENA=1 \
VLLM_BYTE_V2_OUTLIER_ARENA_ENTRIES_PER_BLOCK=4 \
VLLM_BYTE_V2_OUTLIER_ARENA_MIN_ENTRIES=4096 \
VLLM_BYTE_V2_OUTLIER_MAX_PER_TILE=1 \
VLLM_BYTE_V2_PREFILL_DIRECT_SKIP_VALIDATION_SYNC=1 \
VLLM_BYTE_V2_DECODE_APPEND_SKIP_VALIDATION_SYNC=1 \
VLLM_BYTE_V2_DEFERRED_CACHE_UPDATE_ERROR_CHECK=1 \
.venv/bin/python benchmarks/benchmark_byte_v2_decode_e2e.py \
  --modes byte_v2_compressed_only \
  --model /mnt/sda1/yxz/byte_v2/Meta-Llama-3-8B-Instruct \
  --prompt-len 1024 \
  --decode-lens 64 \
  --batch-size 1 \
  --num-runs 2 \
  --warmup-decode-len 8 \
  --gpu-memory-utilization 0.80 \
  --enforce-eager \
  --output-json benchmarks/profiles/bytev2_e2e_p1024_b1_d64_arena_max1_after_single_outlier_fastpath.json
```

结果：

| config | decode_len | output tok/s | elapsed |
| --- | ---: | ---: | ---: |
| ByteV2 compressed-only + arena max=1 | 64 | 15.99 | 4.003s |

pool 统计：

- `any_exhausted=false`
- `any_outlier_exhausted=false`
- `total_assigned_outlier_tiles=4999`
- `total_outlier_next_entry=4999`
- `total_outlier_capacity=1649024`
- `total_next_slot=320`

E2E 相比 Step 38 的 arena max=1 d64（约 14.75 tok/s，另一次 rerun 约
15.36 tok/s）有改善，但 E2E 方差较大；本轮保留修改的主要依据是 nsys 中
prefill-direct encode 的稳定下降。

### 结论

1. compact outlier arena 的主要新增瓶颈不是 decode overlay，而是 encode 端
   outlier tile 分类和 entry 写入。
2. 对 `outlier_max_per_tile=1` 做专用快路径有效，能把 arena encode 开销从
   257.1 us 降到 75.8 us。
3. 当前仍保留通用 histogram 路径，避免影响 `outlier_max_per_tile > 1` 和
   lossy 实验。
4. 后续若继续优化 arena，优先看 decode stage1 的 overlay metadata check 和
   CUTE/independent stage1 对 outlier overlay 的支持，而不是继续扩大 encode
   端复杂度。

## Step 40: per-block has-outlier flag 实验

### 背景

compact outlier arena 的 decode overlay 需要检查 `outlier_tile_meta`。如果一个
physical block 没有任何 outlier tile，则理论上可以在 block 粒度直接跳过所有 tile
metadata 检查，减少 compressed hot path 的 metadata 读和分支。

### 实现

新增 per-block int32 flag：

```text
outlier_block_flags[physical_block] == 0: 该 block 没有 compact outlier entry
outlier_block_flags[physical_block] != 0: 该 block 可能有 compact outlier entry
```

改动点：

1. `ByteV2FullAttentionSpec` 增加 `outlier_block_flag_bytes=4`，并把
   `num_blocks * 4` 计入 `outlier_metadata_bytes()`。
2. allocator 在 sparse fallback metadata 之后切出 `outlier_block_flags`：
   - `vllm/v1/worker/gpu/attn_utils.py`
   - `vllm/v1/worker/gpu_model_runner.py`
3. native cache update 支持写 flag：
   - prefill-direct encode：block 开始清 0；写入 compact outlier entry 后置 1。
   - decode append / touched-block compress：当前不写 outlier arena，清 0，避免旧
     metadata 被误用。
4. native decode 支持读取 flag：
   - flag 为 0 时，把 block-local `outlier_arena/outlier_tile_meta` 置空，跳过
     tile metadata overlay。
   - flag 参数为 `nullptr` 时保持旧行为。
5. Python/backend 增加 opt-in 开关：
   - `VLLM_BYTE_V2_USE_OUTLIER_BLOCK_FLAGS=1` 时 backend 才传 flag。
   - 默认关闭，因为真实 Llama-3 8B prompt 下多数 block 都含至少一个 outlier，
     block 级 flag 很少能跳过 metadata。
6. decode kernel benchmark 增加 `--use-outlier-block-flags`，用于显式测该方案。

### 验证

CUDA 单测：

```bash
CUDA_VISIBLE_DEVICES=2 .venv/bin/python -m pytest \
  tests/v1/attention/test_byte_v2_ops.py::test_byte_v2_native_outlier_arena_prefill_direct_cuda \
  tests/v1/attention/test_byte_v2_ops.py::test_byte_v2_native_outlier_arena_decode_overlay_cuda \
  tests/v1/attention/test_byte_v2_ops.py::test_byte_v2_native_tile_fallback_pool_cuda \
  tests/v1/attention/test_byte_v2_backend.py::test_byte_v2_attention_impl_outlier_arena_pool \
  tests/v1/test_byte_v2_kv_cache_spec.py::test_byte_v2_full_attention_spec_outlier_arena_allocation_size \
  -q -s
```

结果：`5 passed`。

构建：

```bash
uv pip install --python .venv/bin/python -e . --torch-backend=auto
```

结果：成功，native rebuild 耗时约 17 分钟。

### decode-only 结果

合成低 outlier-block 场景：

```bash
CUDA_VISIBLE_DEVICES=2 .venv/bin/python \
  benchmarks/kernels/benchmark_byte_v2_decode_kernel.py \
  --batch-size 1 \
  --seq-len 1024 \
  --split-k 16 \
  --fallback-ratio 0.125 \
  --fallback-pattern single_outlier_per_k_tile \
  --tile-fallback-pool \
  --outlier-arena-entries-per-block 4 \
  --outlier-arena-min-entries 4096 \
  --outlier-max-per-tile 1 \
  --num-runs 100 \
  --warmup-runs 20 \
  --skip-correctness \
  --variant page_fastpath \
  --tile-fastpath-mode on
```

结果：

| config | median | p90 | outlier blocks | outlier tiles |
| --- | ---: | ---: | ---: | ---: |
| flags off | 1049.60 us | 1144.83 us | 0 | 512 |
| flags on | 1044.99 us | 1126.40 us | 8/64 | 512 |

Nsight Systems 对比上一版：

| config | stage1 avg | split-reduce avg | prefill-direct encode |
| --- | ---: | ---: | ---: |
| single-outlier fast path | 64.89 us | 2.93 us | 75.78 us |
| block flags on | 61.40 us | 2.93 us | 76.29 us |

在合成场景中，stage1 kernel 约有 5.4% 改善；整体 decode-only median 只有小幅
改善，属于低 outlier-block 分布下的局部收益。

### E2E 结果

8B E2E smoke，prompt=1024、decode=64、batch=1、enforce eager：

| config | output tok/s | elapsed | pool exhausted |
| --- | ---: | ---: | --- |
| block flags on | 14.05-14.15 | 4.52-4.56s | false |
| block flags off after gate | 14.09 | 4.54s | false |

真实 Llama-3 8B prompt 的统计显示，每层约 48-65 个 prompt block 被标记为
`assigned_outlier_blocks`，而总 prompt block 约 64 个；也就是说大多数 block 都有
至少一个 compact outlier entry，block 级 flag 几乎不能跳过 metadata。

### 结论

1. per-block has-outlier flag 在“多数 block 无 outlier”的合成场景下能降低 stage1
   kernel 时间。
2. 真实 Llama-3 8B prompt 下多数 block 都有 outlier，E2E 没有可观收益。
3. 该能力保留为 opt-in 实验路径，默认关闭：

```bash
VLLM_BYTE_V2_USE_OUTLIER_BLOCK_FLAGS=1
```

4. 后续如果要继续优化真实场景，应做 per-tile / warp-local metadata fast path，而不是
   只依赖 per-block flag。

## Step 41: tile-level has-outlier bitmap 实验

### 背景

Step 40 的 per-block flag 只能跳过完全没有 compact outlier entry 的 block。
真实 Llama-3 8B prompt 中多数 block 都至少有一个 outlier，但之前 tile 统计显示，
一个 block 内真正含 outlier 的 tile 比例仍然很低。因此继续增加 tile 粒度 bitmap：

```text
outlier_tile_bitmap[physical_block, word] bit tile_idx == 0:
  该 tile 没有 compact outlier entry，可跳过 outlier_tile_meta 读取

outlier_tile_bitmap[physical_block, word] bit tile_idx == 1:
  该 tile 可能有 compact outlier entry，需要读取 outlier_tile_meta 并 overlay
```

当 bitmap 参数为 `nullptr` 时保持旧行为：只要 outlier arena/meta 存在，就按旧路径查
`outlier_tile_meta`。

### 实现

1. `ByteV2FullAttentionSpec` 增加 `outlier_tile_bitmap_word_bytes=4` 和
   `outlier_tile_bitmap_words_per_block = ceil(total_tiles / 32)`。
2. allocator metadata layout 更新为：

```text
fallback_block_ids
fallback_next_slot
fallback_tile_ids
fallback_tile_next_slot
outlier_block_flags
outlier_tile_bitmap
outlier_tile_meta
outlier_next_entry
```

3. backend pool 增加 `outlier_tile_bitmap`，并在 stats 中输出
   `assigned_outlier_bitmap_tiles`。
4. native op/schema/Python wrapper 增加可选参数：
   - cache update：`outlier_block_flags -> outlier_tile_bitmap -> outlier_tile_meta`
   - decode：`outlier_block_flags -> outlier_tile_bitmap -> outlier_tile_meta`
5. cache update kernel 行为：
   - prefill-direct encode：每个 physical block 开始时清零 bitmap words；只有成功写入
     compact outlier arena 的 tile 才置位。
   - decode append / touched-block compress：当前不写 outlier arena，清零该 block 的
     block flag 和 tile bitmap，避免旧 metadata 被误用。
6. decode kernel 行为：
   - block flag 为 0 时仍先跳过整个 block 的 arena/meta。
   - block 可能有 outlier 时，再按 tile bitmap 跳过无 outlier tile 的
     `outlier_tile_meta` 读取。
   - GQA shared、GQA WMMA、split-K stage1 以及 compressed/tile fast path 都接入了
     bitmap gate。
7. 新增 opt-in 开关，默认关闭：

```bash
VLLM_BYTE_V2_USE_OUTLIER_TILE_BITMAP=1
```

decode-only benchmark 增加：

```bash
--use-outlier-tile-bitmap
```

### 验证

Python/ruff：

```bash
.venv/bin/python -m py_compile \
  vllm/envs.py \
  vllm/v1/kv_cache_interface.py \
  vllm/v1/attention/backends/byte_v2_attn.py \
  vllm/_custom_ops.py \
  vllm/v1/attention/backends/byte_v2_ops.py \
  vllm/v1/worker/gpu/attn_utils.py \
  vllm/v1/worker/gpu_model_runner.py \
  benchmarks/kernels/benchmark_byte_v2_decode_kernel.py \
  tests/v1/attention/test_byte_v2_ops.py \
  tests/v1/attention/test_byte_v2_backend.py \
  tests/v1/test_byte_v2_kv_cache_spec.py

.venv/bin/python -m ruff check \
  vllm/envs.py \
  vllm/v1/kv_cache_interface.py \
  vllm/v1/attention/backends/byte_v2_attn.py \
  vllm/_custom_ops.py \
  vllm/v1/attention/backends/byte_v2_ops.py \
  vllm/v1/worker/gpu/attn_utils.py \
  vllm/v1/worker/gpu_model_runner.py \
  benchmarks/kernels/benchmark_byte_v2_decode_kernel.py \
  tests/v1/attention/test_byte_v2_ops.py \
  tests/v1/attention/test_byte_v2_backend.py \
  tests/v1/test_byte_v2_kv_cache_spec.py
```

结果：通过。

native rebuild：

```bash
uv pip install --python .venv/bin/python -e . --torch-backend=auto
```

结果：成功，native rebuild 耗时约 17 分 29 秒。

CUDA 单测：

```bash
CUDA_VISIBLE_DEVICES=2 .venv/bin/python -m pytest \
  tests/v1/test_byte_v2_kv_cache_spec.py::test_byte_v2_full_attention_spec_outlier_arena_allocation_size \
  tests/v1/attention/test_byte_v2_backend.py::test_byte_v2_attention_impl_outlier_arena_pool \
  tests/v1/attention/test_byte_v2_ops.py::test_byte_v2_native_outlier_arena_prefill_direct_cuda \
  tests/v1/attention/test_byte_v2_ops.py::test_byte_v2_native_outlier_arena_decode_overlay_cuda \
  tests/v1/attention/test_byte_v2_ops.py::test_byte_v2_native_tile_fallback_pool_cuda \
  -q -s
```

结果：`5 passed`。

其中 `test_byte_v2_native_outlier_arena_prefill_direct_cuda` 额外覆盖了一个 stale
`outlier_tile_meta` 场景：手动把没有 outlier 的 V tile meta 写成旧值，如果 tile
bitmap 不生效，decode 会错误 overlay；当前结果与 raw reference 一致。

E2E smoke：

```bash
CUDA_VISIBLE_DEVICES=2 \
VLLM_BYTE_V2_ENABLE_OUTLIER_ARENA=1 \
VLLM_BYTE_V2_OUTLIER_MAX_PER_TILE=1 \
VLLM_BYTE_V2_OUTLIER_ARENA_ENTRIES_PER_BLOCK=4 \
VLLM_BYTE_V2_OUTLIER_ARENA_MIN_ENTRIES=4096 \
VLLM_BYTE_V2_USE_OUTLIER_TILE_BITMAP=1 \
VLLM_BYTE_V2_USE_OUTLIER_BLOCK_FLAGS=1 \
.venv/bin/python benchmarks/benchmark_byte_v2_decode_e2e.py \
  --model /mnt/sda1/yxz/byte_v2/Meta-Llama-3-8B-Instruct \
  --modes byte_v2_compressed_only \
  --prompt-len 128 \
  --decode-lens 8 \
  --batch-size 1 \
  --num-runs 1 \
  --warmup-decode-len 1 \
  --max-model-len 256 \
  --max-num-batched-tokens 256 \
  --gpu-memory-utilization 0.85 \
  --enforce-eager \
  --disable-prefix-caching
```

结果：

- `decode_len=8`，median output throughput `14.72 tok/s`，median elapsed
  `0.543s`。
- `any_exhausted=false`，`any_outlier_exhausted=false`。
- final stats 中 `total_assigned_outlier_tiles=1574`，
  `total_outlier_next_entry=1574`，并且各层
  `assigned_outlier_bitmap_tiles == assigned_outlier_tiles`，说明 bitmap 在真实
  backend 路径中被写入。

### decode-only 结果

场景 A：每个 block 只有 1 个 tile 有 outlier，且所有 block 都有 outlier。

```bash
CUDA_VISIBLE_DEVICES=2 .venv/bin/python \
  benchmarks/kernels/benchmark_byte_v2_decode_kernel.py \
  --batch-size 1 \
  --seq-len 1024 \
  --split-k 16 \
  --fallback-ratio 1.0 \
  --fallback-pattern single_outlier \
  --tile-fallback-pool \
  --outlier-max-per-tile 1 \
  --outlier-arena-entries-per-block 2 \
  --outlier-arena-min-entries 4096 \
  --use-outlier-block-flags \
  --num-runs 50 \
  --warmup-runs 10 \
  --skip-correctness \
  --variant page_fastpath \
  --tile-fastpath-mode on \
  --partial-workspace
```

| config | median | p90 | outlier bitmap tiles | outlier tiles |
| --- | ---: | ---: | ---: | ---: |
| tile bitmap off | 1043.97 us | 1096.70 us | 0 | 64 |
| tile bitmap on | 1050.62 us | 1065.98 us | 64 | 64 |

场景 B：只有 12.5% block 有 1 个 outlier tile，不传 block flag，只看 tile bitmap
是否能单独降低 meta 读取。

| config | median | p90 | outlier bitmap tiles | outlier tiles |
| --- | ---: | ---: | ---: | ---: |
| tile bitmap off | 1038.85 us | 1062.91 us | 0 | 8 |
| tile bitmap on | 1040.90 us | 1120.26 us | 8 | 8 |

### 结论

1. tile-level bitmap 功能已打通，并能正确防止 stale `outlier_tile_meta` 被误用。
2. 当前 decode-only microbenchmark 没有观察到稳定性能提升；bitmap 读、bit test 和
   现有 stage1 pipeline 的其它成本抵消了减少 `outlier_tile_meta` 读取的收益。
3. 因为 E2E 可能受方差影响，当前保留实现但默认关闭，只作为后续 profiling 和
   CUTE/independent stage1 overlay 改造的实验开关。
4. 如果后续要让该 bitmap 产生收益，需要把它和 stage1 tile loader 融合得更彻底：
   - warp/group 级一次读取 bitmap word，避免每个 overlay helper 重复算 word；
   - 在 shared loader 调度中把 no-outlier tile 走完全独立的 no-overlay path；
   - CUTE/independent stage1 中把 tile bitmap 作为 producer 分支，而不是 consumer
     元素级 overlay 分支。

## Step 42：CUTE stage1 支持 fallback/outlier metadata

### 目标

前一轮 profile 表明 decode stage1 的主要差距不在单纯的 fallback/outlier 分支检查。
本轮先尝试了一个 generic compressed-only hotpath：在 split stage1 中通过编译期模板
去掉 fallback/outlier 参数和 overlay helper。该方案在 fallback=0 decode-only 下没有
收益，因此不保留。

随后把已有 CUTE-style fixed-shape stage1 扩展到 metadata 场景，使其可以在
`VLLM_BYTE_V2_DECODE_CUTE_STAGE1=1` 且 Llama-3 8B 形状下处理：

- sparse fallback pool / block raw fallback；
- tile fallback pool；
- compact outlier arena；
- outlier block flag / tile bitmap gate。

没有 metadata 时仍走 `metadata_fastpath=false` 编译期路径，保持原 CUTE no-metadata
fast path。

### 代码改动

- `csrc/libtorch_stable/cache_kernels.cu`
  - `byte_v2_paged_decode_attention_gqa_cute_split_stage1_kernel` 增加
    `metadata_fastpath` 模板参数。
  - metadata 变体在 compressed page 下复用 no-fallback tile decoder、tile fallback
    decoder 和 outlier overlay helper。
  - metadata 变体支持 raw fallback page，从 sparse fallback pool 或 page raw tail
    读取 raw K/V。
  - host launch 中当 `VLLM_BYTE_V2_DECODE_CUTE_STAGE1=1`、shape 为
    `num_heads=32,num_kv_heads=8,head_size=128,head_size_v=128`、page size 为
    compressed-only 时启用 CUTE；有 fallback/outlier metadata 时实例化
    `metadata_fastpath=true`。
- `benchmarks/kernels/benchmark_byte_v2_decode_kernel.py`
  - 新增 `--cute-stage1`，并自动启用 page fastpath。
- `vllm/envs.py`
  - 增加 `VLLM_BYTE_V2_DECODE_CUTE_STAGE1` 环境变量定义。
- `tests/v1/attention/test_byte_v2_decode.py`
  - 新增 CUTE metadata correctness 测试，覆盖 compressed-only page size +
    fallback pool raw block。

### 验证

```bash
.venv/bin/python -m py_compile \
  benchmarks/kernels/benchmark_byte_v2_decode_kernel.py \
  tests/v1/attention/test_byte_v2_decode.py \
  vllm/envs.py

.venv/bin/python -m ruff check \
  benchmarks/kernels/benchmark_byte_v2_decode_kernel.py \
  tests/v1/attention/test_byte_v2_decode.py \
  vllm/envs.py

git diff --check -- \
  csrc/libtorch_stable/cache_kernels.cu \
  benchmarks/kernels/benchmark_byte_v2_decode_kernel.py \
  tests/v1/attention/test_byte_v2_decode.py \
  vllm/envs.py

uv pip install --python .venv/bin/python -e . --torch-backend=auto

CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m pytest \
  tests/v1/attention/test_byte_v2_decode.py::test_byte_v2_paged_decode_attention_op_cute_stage1_cuda \
  tests/v1/attention/test_byte_v2_decode.py::test_byte_v2_paged_decode_attention_op_cute_stage1_metadata_cuda \
  tests/v1/attention/test_byte_v2_decode.py::test_byte_v2_paged_decode_attention_op_gqa_wmma_split_k_cuda \
  tests/v1/attention/test_byte_v2_decode.py::test_byte_v2_paged_decode_attention_op_page_fastpath_split_k_cuda \
  tests/v1/attention/test_byte_v2_decode.py::test_byte_v2_paged_decode_attention_op_tile_fastpath_cuda \
  -q -s
```

结果：Python/ruff/diff-check 通过，native rebuild 成功，CUDA 单测 `5 passed`。

E2E smoke：

```bash
CUDA_VISIBLE_DEVICES=1 \
VLLM_BYTE_V2_DECODE_PAGE_FASTPATH=1 \
VLLM_BYTE_V2_DECODE_CUTE_STAGE1=1 \
VLLM_BYTE_V2_ENABLE_OUTLIER_ARENA=1 \
VLLM_BYTE_V2_OUTLIER_MAX_PER_TILE=1 \
VLLM_BYTE_V2_OUTLIER_ARENA_ENTRIES_PER_BLOCK=4 \
VLLM_BYTE_V2_OUTLIER_ARENA_MIN_ENTRIES=4096 \
.venv/bin/python benchmarks/benchmark_byte_v2_decode_e2e.py \
  --model /mnt/sda1/yxz/byte_v2/Meta-Llama-3-8B-Instruct \
  --modes byte_v2_compressed_only \
  --prompt-len 128 \
  --decode-lens 8 \
  --batch-size 1 \
  --num-runs 1 \
  --warmup-decode-len 1 \
  --max-model-len 256 \
  --max-num-batched-tokens 256 \
  --gpu-memory-utilization 0.85 \
  --enforce-eager \
  --disable-prefix-caching
```

结果：`decode_len=8`，median output throughput `14.83 tok/s`，median elapsed
`0.539s`；`any_exhausted=false`，`any_outlier_exhausted=false`；
`total_assigned_outlier_tiles=1574`，`total_outlier_next_entry=1574`。

### decode-only A/B

配置：

```bash
CUDA_VISIBLE_DEVICES=1 .venv/bin/python \
  benchmarks/kernels/benchmark_byte_v2_decode_kernel.py \
  --batch-size 1 \
  --seq-len 1024,2048 \
  --split-k 16,32 \
  --num-runs 100 \
  --warmup-runs 20 \
  --variant page_fastpath \
  --tile-fastpath-mode on \
  --partial-workspace \
  --fallback-ratio 0.03 \
  --fallback-pattern single_outlier_per_k_tile \
  --tile-fallback-pool \
  --outlier-arena-entries-per-block 8 \
  --outlier-arena-min-entries 4096 \
  --outlier-max-per-tile 1 \
  --skip-correctness
```

同配置启用 CUTE：

```bash
--cute-stage1
```

| seq_len | split_k | page-fastpath | CUTE metadata | latency delta |
| ---: | ---: | ---: | ---: | ---: |
| 1024 | 16 | 1046.53 us | 1015.81 us | -2.9% |
| 1024 | 32 | 1084.42 us | 992.26 us | -8.5% |
| 2048 | 16 | 1231.87 us | 1171.46 us | -4.9% |
| 2048 | 32 | 1155.07 us | 1114.59 us | -3.5% |

额外 fallback=0/no-metadata 场景下，CUTE stage1 也保持收益：

| seq_len | split_k | CUTE median |
| ---: | ---: | ---: |
| 1024 | 16 | 999.94 us |
| 1024 | 32 | 987.14 us |
| 2048 | 16 | 1122.30 us |
| 2048 | 32 | 1101.82 us |

### 结论

1. 单纯去掉 generic stage1 的 fallback/outlier 分支没有收益，说明瓶颈主要来自
   stage1 的 softmax/reduce 组织、shared layout 和 WMMA 消费方式，而不是 metadata
   pointer check。
2. CUTE-style stage1 的 warp-level softmax 和固定 Llama GQA/head geometry 对 decode
   stage1 有稳定收益。
3. metadata 支持后，CUTE stage1 可以进入 outlier arena / tile fallback 实验路径，
   下一步应做 E2E 对比，并考虑把 CUTE stage1 作为 ByteV2 Llama-3 8B 形状的默认
   opt-in 性能路径。

## Step 43：batched decode append 默认化与 host device-property cache

### 目标

Step 42 之后重新 profile p512/b4/d128，发现 E2E 中还有两个容易误判的问题：

1. `benchmark_byte_v2_decode_e2e.py --child-mode byte_v2_compressed_only`
   没有在 child 进程内应用 `_mode_env()`，单独 nsys child profile 会漏掉
   `VLLM_BYTE_V2_COMPRESSED_ONLY_CACHE=1`。
2. 在修正 child-mode 以后，decode append fast path 已经把 cache update GPU
   时间压到几十毫秒级，但 Nsight API 表里出现 4096 次
   `cudaGetDeviceProperties`，总计约 4.5s，对应 128 decode step * 32 layers。

因此本轮目标不是继续改 stage1，而是：

- 让 child-mode profile 与 parent-mode 环境一致；
- 让 eager/profile 下安全地默认走 batched decode append fast path；
- 缓存 ByteV2 attention wrapper 里的 BF16 WMMA device capability 查询。

### 代码改动

1. `benchmarks/benchmark_byte_v2_decode_e2e.py`
   - `run_child()` 开头执行 `os.environ.update(_mode_env(mode))`。
   - 修正后，直接跑 `--child-mode byte_v2_compressed_only` 会设置
     `VLLM_BYTE_V2_COMPRESSED_ONLY_CACHE=1`，并创建 sparse fallback pool。

2. `csrc/libtorch_stable/cache_kernels.cu`
   - 新增 `byte_v2_validate_decode_append_slots_kernel`。
   - eager multi-token decode update 默认先做无副作用 validation：
     - slot/block 合法；
     - active tokens 不重复写同一个 physical block；
     - page status/valid rows 合法；
     - 空 block 只允许 offset 0；
     - non-overwrite append 只允许 `existing_valid_rows == block_offset`；
     - finalized block 或 same-block continuation 自动回退 generic path。
   - validation 通过时走 `byte_v2_decode_append_cache_kernel`；
     失败时走 generic touched-block path。
   - 单 token、CUDA graph capture、显式
     `VLLM_BYTE_V2_DECODE_APPEND_BATCH_FASTPATH=1` 保持原 fast path 行为。
   - 新增 `byte_v2_device_supports_bf16_wmma()`，用 `std::once_flag` 按 device
     缓存 BF16 WMMA capability，替换每次 attention launch 前的
     `cudaGetDeviceProperties`。

3. `tests/v1/attention/test_byte_v2_ops.py`
   - batched decode append 测试不再依赖显式 env，覆盖默认 safe path。
   - 新增 same-block batched update fallback 测试，验证 continuation/same-block
     场景不会被 fast path 误写。

### 验证

native rebuild：

```bash
uv pip install -e . --torch-backend=auto
```

结果：成功，约 `22m46s`。

targeted tests：

```bash
CUDA_VISIBLE_DEVICES=2 .venv/bin/python -m pytest \
  tests/v1/attention/test_byte_v2_ops.py::test_byte_v2_native_batched_decode_append_uses_partial_raw_fallback_cuda \
  tests/v1/attention/test_byte_v2_ops.py::test_byte_v2_native_batched_same_block_update_falls_back_cuda \
  tests/v1/attention/test_byte_v2_ops.py::test_byte_v2_native_decode_append_finalizes_partial_raw_fallback_cuda \
  tests/v1/attention/test_byte_v2_ops.py::test_byte_v2_native_batched_decode_append_finalizes_blocks_cuda \
  tests/v1/attention/test_byte_v2_ops.py::test_byte_v2_native_batched_decode_append_duplicate_block_fails_cuda \
  tests/v1/attention/test_byte_v2_ops.py::test_byte_v2_native_decode_append_fallback_pool_exhaustion_cuda \
  -q
```

结果：`6 passed`。

decode/backend smoke：

```bash
CUDA_VISIBLE_DEVICES=2 .venv/bin/python -m pytest \
  tests/v1/attention/test_byte_v2_decode.py \
  tests/v1/attention/test_byte_v2_backend.py::test_byte_v2_deferred_cache_update_records_sticky_error \
  -q
```

结果：`27 passed`。

### E2E

workload：

```text
model=/mnt/sda1/yxz/byte_v2/Meta-Llama-3-8B-Instruct
prompt_len=512
batch_size=4
decode_len=128
enforce_eager=true
fallback_pool_ratio=0.03
outlier_arena=off
CUTE metadata auto=on
split_k=8
```

结果：

| mode | tok/s | elapsed | pool exhausted |
| --- | ---: | ---: | --- |
| raw | 131.15 | 3.904s | false |
| ByteV2 compressed-only | 102.66 | 4.988s | false |

ByteV2 达到 raw 的约 `78.3%`。

### nsys

有效 compressed-only profile：

```bash
CUDA_VISIBLE_DEVICES=2 \
VLLM_ENABLE_V1_MULTIPROCESSING=0 \
VLLM_BYTE_V2_USE_NATIVE_KERNELS=1 \
VLLM_BYTE_V2_ENABLE_SPARSE_FALLBACK_POOL=1 \
VLLM_BYTE_V2_SPARSE_FALLBACK_POOL_RATIO=0.03 \
VLLM_BYTE_V2_ENABLE_OUTLIER_ARENA=0 \
VLLM_BYTE_V2_DECODE_PAGE_FASTPATH=1 \
VLLM_BYTE_V2_DECODE_TILE_FASTPATH=1 \
VLLM_BYTE_V2_DECODE_SPLIT_K=8 \
VLLM_BYTE_V2_DECODE_PARALLEL_REDUCE=1 \
VLLM_BYTE_V2_DECODE_CUTE_STAGE1_AUTO=1 \
nsys profile --trace=cuda,nvtx,cublas \
  --output benchmarks/profiles/nsys_cached_props_bytev2_cute_p512_b4_d128 \
  .venv/bin/python benchmarks/benchmark_byte_v2_decode_e2e.py \
    --child-mode byte_v2_compressed_only \
    --model /mnt/sda1/yxz/byte_v2/Meta-Llama-3-8B-Instruct \
    --prompt-len 512 --decode-lens 128 --batch-size 4 \
    --num-runs 1 --warmup-decode-len 8 \
    --gpu-memory-utilization 0.80 --enforce-eager
```

ByteV2 profile run：`104.56 tok/s`。

关键 GPU kernel：

| kernel/category | total |
| --- | ---: |
| measured projected GPU time | 5455.5 ms |
| `Kernel2` | 2248.4 ms |
| GEMM `64x64_sliced1x2` | 1144.4 ms |
| `byte_v2_paged_decode_attention_gqa_cute_split_stage1_kernel` | 292.6 ms |
| `byte_v2_paged_decode_attention_split_reduce_parallel_kernel` | 13.5 ms |
| `byte_v2_decode_append_cache_kernel` | 33.3 ms |
| `byte_v2_validate_decode_append_slots_kernel` | 10.2 ms |
| `byte_v2_init_decode_append_result_kernel` | 4.6 ms |

关键 API 对比：

| API | 修复前 | 修复后 |
| --- | ---: | ---: |
| `cudaGetDeviceProperties_v2_v12000` | 4096 calls / 4506.2 ms | 不再出现在 measured API top list |

### 结论

本轮改动保留。

主要收益来自 host wrapper 的 `cudaGetDeviceProperties` cache，其次是 batched
decode append 默认 safe path。p512/b4/d128 已经不再被
`byte_v2_compress_touched_blocks_kernel` 主导，cache update GPU 时间下降到：

```text
decode append: 33.3 ms
validation:    10.2 ms
result init:    4.6 ms
```

剩余差距主要是：

1. ByteV2 CUTE stage1 仍有 292.6 ms，raw attention 之前约 93 ms 量级；
2. eager 路径还有大量 `cudaMemcpyAsync` 和 `cudaStreamSynchronize`；
3. ByteV2 production cudagraph 还未重新启用和验证。

下一步优先级：

1. 减少 validation/result host copy，同步到 deferred sticky error 或调度侧 metadata；
2. 重新回到 stage1：CUTE/CUTLASS-style loader、shared layout、PV/write partial；
3. 做 non-eager/cudagraph profile，确认 production 路径和 raw 的真实差距。

## Step 44：metadata-gated deferred batched decode append

### 背景

Step 43 之后，p512/b4/d128 的 ByteV2 compressed-only 已经恢复到
`102.66 tok/s`，但 nsys 仍显示 eager multi-token decode append 有明显的 host
validation/result copy：

| item | Step 43 no-deferred |
| --- | ---: |
| `byte_v2_validate_decode_append_slots_kernel` | 4064 calls / 10.24 ms |
| `cudaMemcpyAsync` | 17293 calls / 2026.35 ms |
| `cudaStreamSynchronize` | 8320 calls / 28.17 ms |

因此本轮目标是把 Step 43 的 validation/result host copy 转为
production-safe 的 deferred sticky error 路径。

### 反例：不能只看 `deferred_error`

第一版尝试在 native 层用：

```text
has_deferred_error && !has_outlier_arena
```

直接打开 `num_tokens > 1` 的 decode append fast path。targeted unit test 能过，
但真实 E2E warmup 失败：

```text
Byte-v2 deferred decode cache append failed:
duplicate token slot in one cache update (error_code=3, block_id=1, ...)
```

原因是 `deferred_error` 只表示“错误可以延后上报”，不能证明当前 update 是
pure decode append。warmup/prefill 同样会携带 deferred error tensor，因此会被误放进
batched decode append fast path。

### 修复方案

本轮改成 metadata-gated：

1. `unified_kv_cache_update()` 从 `get_attention_context()` 取
   `attn_metadata`。
2. 对实现了 `do_kv_cache_update_with_metadata()` 的 backend 传入 metadata；
   其他 backend 仍走旧的 `do_kv_cache_update()`，不改变 raw vLLM 路径。
3. `ByteV2AttentionImpl` 增加 `_is_pure_decode_cache_update()`：
   - metadata 是 `ByteV2Metadata`；
   - `num_prefills == 0`；
   - `num_prefill_tokens == 0`；
   - `max_query_len == 1`；
   - `num_decodes == num_decode_tokens`；
   - `num_decode_tokens == num_actual_tokens == slot_mapping.numel()`；
   - `slot_mapping.numel() > 1`。
4. native op 新增默认参数：

```text
decode_append_fast_path_safe=False
```

只有上层 metadata 判定为 pure decode，且 native 层确认
`has_deferred_error && !has_outlier_arena` 时，eager multi-token update 才跳过
validation/result host copy。

单 token fast path、CUDA graph capture fast path、显式
`VLLM_BYTE_V2_DECODE_APPEND_BATCH_FASTPATH=1` 保持 Step 43 行为。

### 代码改动

1. `vllm/model_executor/layers/attention/attention.py`
   - `unified_kv_cache_update()` 增加 metadata-aware 分发。

2. `vllm/v1/attention/backends/byte_v2_attn.py`
   - 增加 `_is_pure_decode_cache_update()`。
   - 增加 `do_kv_cache_update_with_metadata()`。
   - 透传 `decode_append_fast_path_safe` 到 ByteV2 cache update op。

3. `vllm/_custom_ops.py`、
   `vllm/v1/attention/backends/byte_v2_ops.py`、
   `csrc/libtorch_stable/ops.h`、
   `csrc/libtorch_stable/torch_bindings.cpp`、
   `csrc/libtorch_stable/cache_kernels.cu`
   - 为 `byte_v2_reshape_and_cache` 增加
     `decode_append_fast_path_safe` 参数。

4. `tests/v1/attention/test_byte_v2_backend.py`
   - 增加 direct native op 的 deferred batched append fast path 测试；
   - 增加 duplicate-block sticky error 测试。

### 验证

静态检查：

```bash
.venv/bin/python -m py_compile \
  vllm/_custom_ops.py \
  vllm/v1/attention/backends/byte_v2_ops.py \
  vllm/v1/attention/backends/byte_v2_attn.py \
  vllm/model_executor/layers/attention/attention.py \
  tests/v1/attention/test_byte_v2_backend.py

.venv/bin/python -m ruff check \
  vllm/_custom_ops.py \
  vllm/v1/attention/backends/byte_v2_ops.py \
  vllm/v1/attention/backends/byte_v2_attn.py \
  vllm/model_executor/layers/attention/attention.py \
  tests/v1/attention/test_byte_v2_backend.py
```

结果：通过。

native rebuild：

```bash
uv pip install -e . --torch-backend=auto
```

结果：成功，约 `16m05s`。

targeted CUDA tests：

```bash
CUDA_VISIBLE_DEVICES=2 .venv/bin/python -m pytest \
  tests/v1/attention/test_byte_v2_backend.py::test_byte_v2_deferred_batched_decode_append_fast_path_cuda \
  tests/v1/attention/test_byte_v2_backend.py::test_byte_v2_deferred_batched_duplicate_block_records_error_cuda \
  tests/v1/attention/test_byte_v2_backend.py::test_byte_v2_deferred_cache_update_records_sticky_error \
  tests/v1/attention/test_byte_v2_ops.py::test_byte_v2_native_batched_decode_append_uses_partial_raw_fallback_cuda \
  tests/v1/attention/test_byte_v2_ops.py::test_byte_v2_native_batched_same_block_update_falls_back_cuda \
  -q
```

结果：`5 passed`。

decode/backend smoke：

```bash
CUDA_VISIBLE_DEVICES=2 .venv/bin/python -m pytest \
  tests/v1/attention/test_byte_v2_decode.py \
  tests/v1/attention/test_byte_v2_backend.py::test_byte_v2_deferred_cache_update_records_sticky_error \
  tests/v1/attention/test_byte_v2_backend.py::test_byte_v2_deferred_batched_decode_append_fast_path_cuda \
  tests/v1/attention/test_byte_v2_backend.py::test_byte_v2_deferred_batched_duplicate_block_records_error_cuda \
  -q
```

结果：`29 passed`。

### E2E

workload：

```text
model=/mnt/sda1/yxz/byte_v2/Meta-Llama-3-8B-Instruct
prompt_len=512
batch_size=4
decode_len=128
fallback_pool_ratio=0.03
outlier_arena=off
CUTE metadata auto=on
split_k=8
enforce_eager=true
```

结果：

| variant | tok/s | elapsed | raw 占比 |
| --- | ---: | ---: | ---: |
| raw | 131.09 | 3.906s | 100% |
| ByteV2 no-deferred | 102.60 | 4.990s | 78.3% |
| ByteV2 deferred metadata gate | 122.24 | 4.189s | 93.3% |

相对 no-deferred 提升约 `+19.1%`。本轮改动保留。

### nsys

本轮 deferred metadata-gated profile：

```bash
CUDA_VISIBLE_DEVICES=2 \
VLLM_ENABLE_V1_MULTIPROCESSING=0 \
VLLM_BYTE_V2_USE_NATIVE_KERNELS=1 \
VLLM_BYTE_V2_ENABLE_SPARSE_FALLBACK_POOL=1 \
VLLM_BYTE_V2_SPARSE_FALLBACK_POOL_RATIO=0.03 \
VLLM_BYTE_V2_ENABLE_OUTLIER_ARENA=0 \
VLLM_BYTE_V2_OUTLIER_MAX_PER_TILE=0 \
VLLM_BYTE_V2_DECODE_PAGE_FASTPATH=1 \
VLLM_BYTE_V2_DECODE_TILE_FASTPATH=1 \
VLLM_BYTE_V2_DECODE_SPLIT_K=8 \
VLLM_BYTE_V2_DECODE_PARALLEL_REDUCE=1 \
VLLM_BYTE_V2_DECODE_CUTE_STAGE1_AUTO=1 \
VLLM_BYTE_V2_DECODE_FLASH_STAGE1=0 \
VLLM_BYTE_V2_DEFERRED_CACHE_UPDATE_ERROR_CHECK=1 \
nsys profile --force-overwrite=true --trace=cuda,nvtx,cublas \
  --cuda-graph-trace=node --sample=none --cpuctxsw=none \
  --cuda-event-trace=false \
  --output benchmarks/profiles/nsys_deferred_append_bytev2_cute_p512_b4_d128 \
  .venv/bin/python benchmarks/benchmark_byte_v2_decode_e2e.py \
    --child-mode byte_v2_compressed_only \
    --model /mnt/sda1/yxz/byte_v2/Meta-Llama-3-8B-Instruct \
    --prompt-len 512 --decode-lens 128 --batch-size 4 \
    --num-runs 1 --warmup-decode-len 8 \
    --gpu-memory-utilization 0.80 --enforce-eager
```

profile run：`122.27 tok/s`。

对比 Step 43：

| item | Step 43 no-deferred | Step 44 deferred gate |
| --- | ---: | ---: |
| `byte_v2_validate_decode_append_slots_kernel` | 4064 calls / 10.24 ms | 0 |
| `byte_v2_decode_append_cache_kernel` | 4064 calls / 33.29 ms | 4064 calls / 34.49 ms |
| `byte_v2_init_decode_append_result_kernel` | 4064 calls / 4.63 ms | 4064 calls / 6.21 ms |
| `byte_v2_record_deferred_cache_update_error_kernel` | 0 | 4064 calls / 6.08 ms |
| `cudaMemcpyAsync` | 17293 calls / 2026.35 ms | 5005 calls / 1056.50 ms |
| `cudaStreamSynchronize` | 8320 calls / 28.17 ms | 320 calls / 2.81 ms |

解释：

- validation kernel 和 host-side validation result copy 被移除；
- sticky error record 增加约 `6.08 ms`，但它是 device-side 小 kernel；
- host API 层 `cudaMemcpyAsync` 和 `cudaStreamSynchronize` 大幅下降，是 E2E
  提升的主要原因；
- `byte_v2_decode_append_cache_kernel` 本身略有波动，不是本轮主要收益来源。

### 结论与下一步

Step 44 达到保留门槛，并且修复了“只看 deferred_error 会误把 warmup/prefill
当 pure decode”的正确性问题。

当前 p512/b4/d128 eager 下，ByteV2 compressed-only 约为 raw 的 `93.3%`。下一步优先：

1. 做 raw vs ByteV2 的最新 detailed profile，确认剩余 `6.7%` 差距来自
   attention stage1、host/API、sampling/GEMM 还是 cudagraph 缺失；
2. 尝试 non-eager/cudagraph 路径。raw production 会开启 cudagraph，如果 ByteV2
   仍只在 eager 对齐，真实 production 差距会重新变大；
3. 如果继续 cache update，小心评估
   `init_decode_append_result + record_deferred_cache_update_error` 融入
   `byte_v2_decode_append_cache_kernel`。该方向预计收益较小，必须用 E2E 保留。

## Step 45：最新 raw vs ByteV2 detailed profile

### 目标

Step 44 后，p512/b4/d128 eager 下 ByteV2 compressed-only 达到 raw 的约
`93.3%`。本轮目标是确认剩余差距主要来自：

1. attention stage1；
2. host/API；
3. sampling/GEMM；
4. cudagraph 缺失。

### 环境限制

本轮尝试重新跑最新 raw nsys：

```bash
CUDA_VISIBLE_DEVICES=2 VLLM_ENABLE_V1_MULTIPROCESSING=0 \
nsys profile --force-overwrite=true --trace=cuda,nvtx,cublas \
  --cuda-graph-trace=node --sample=none --cpuctxsw=none \
  --cuda-event-trace=false \
  --output benchmarks/profiles/nsys_step45_raw_eager_p512_b4_d128 \
  .venv/bin/python benchmarks/benchmark_byte_v2_decode_e2e.py \
    --child-mode raw \
    --model /mnt/sda1/yxz/byte_v2/Meta-Llama-3-8B-Instruct \
    --prompt-len 512 --decode-lens 128 --batch-size 4 \
    --num-runs 1 --warmup-decode-len 8 \
    --gpu-memory-utilization 0.80 --enforce-eager
```

但 GPU2 在启动时已被其他进程占用，vLLM 报：

```text
Free memory on device cuda:0 (14.78/44.42 GiB) on startup is less than
desired GPU memory utilization (0.8, 35.54 GiB).
```

随后尝试在 GPU0 用较低 `gpu_memory_utilization=0.32` 跑 non-eager/cudagraph
sanity check，但外部进程占用约 `29.10 GiB`，Inductor 编译阶段 OOM：

```text
torch._inductor.exc.InductorError: OutOfMemoryError
```

因此本轮 detailed profile 使用：

- raw eager：已有同 workload 的有效 profile
  `benchmarks/profiles/nsys_next_sp_raw_p512_b4_d128_*`。raw 路径没有被 Step 44
  代码改动影响。
- ByteV2 eager：Step 44 最新 deferred metadata-gated profile
  `benchmarks/profiles/nsys_deferred_append_bytev2_cute_p512_b4_d128_*`。
- cudagraph 判断：引用已有干净 production baseline
  `benchmarks/profiles/bytev2_no_arena_baseline_p512_b4_d128_256_r3.json`。

### E2E 对照

Step 44 最新 eager E2E：

| mode | tok/s | elapsed | ByteV2/raw |
| --- | ---: | ---: | ---: |
| raw | 131.09 | 3.906s | 100% |
| ByteV2 compressed-only | 122.24 | 4.189s | 93.3% |

已有 production/cudagraph baseline：

| mode | decode_len | tok/s | elapsed | ByteV2/raw |
| --- | ---: | ---: | ---: | ---: |
| raw | 128 | 133.60 | 3.832s | 100% |
| ByteV2 no-arena | 128 | 123.19 | 4.156s | 92.2% |
| raw | 256 | 133.45 | 7.673s | 100% |
| ByteV2 no-arena | 256 | 123.78 | 8.273s | 92.8% |

结论：production/cudagraph 下 ByteV2 与 raw 的差距没有明显重新扩大，仍约
`7%-8%`。因此当前主要问题不是“ByteV2 完全缺 cudagraph”，而是 graph 内部的
attention/cache-update kernel 差距。

### eager nsys kernel 分桶

workload：

```text
model=/mnt/sda1/yxz/byte_v2/Meta-Llama-3-8B-Instruct
prompt_len=512
batch_size=4
decode_len=128
prefix cache on
enforce_eager=true
ByteV2: compressed-only, fallback_pool_ratio=0.03, no arena,
        split_k=8, CUTE metadata auto, deferred cache update error on
```

分类脚本按 kernel 名称分桶：

- raw attention：
  `flash_fwd_splitkv_kernel` + `flash_fwd_splitkv_combine_kernel`
- ByteV2 attention：
  `byte_v2_paged_decode_attention_gqa_cute_split_stage1_kernel` +
  `byte_v2_paged_decode_attention_split_reduce_parallel_kernel`
- raw cache update：
  `reshape_and_cache_flash_kernel`
- ByteV2 cache update：
  `byte_v2_decode_append_cache_kernel` +
  `byte_v2_init_decode_append_result_kernel` +
  `byte_v2_record_deferred_cache_update_error_kernel` +
  `byte_v2_prefill_direct_encode_blocks_kernel`

结果：

| category | raw eager | ByteV2 eager | delta |
| --- | ---: | ---: | ---: |
| attention | 93.37 ms | 305.29 ms | +211.92 ms |
| cache update | 12.25 ms | 48.25 ms | +36.00 ms |
| GEMM/MLP/linear | 3677.46 ms | 3679.25 ms | +1.78 ms |
| norm/rope/activation/elementwise | 84.87 ms | 76.17 ms | -8.70 ms |
| sampling/scheduler misc | 4.81 ms | 7.11 ms | +2.30 ms |
| other | 0.91 ms | 1.26 ms | +0.35 ms |
| total GPU kernel | 3873.67 ms | 4117.32 ms | +243.65 ms |

注意：E2E elapsed delta 是 `4.189s - 3.906s = 283 ms`，与 GPU kernel delta
`244 ms` 同量级。差额主要来自 CPU/API 调度、测量同步和运行噪声。

### top kernels

raw eager：

| kernel | total |
| --- | ---: |
| `Kernel2` | 2248.26 ms |
| GEMM `64x64_sliced1x2` | 1142.51 ms |
| GEMM `64x64_ldg8` | 233.55 ms |
| raw FlashAttention split-k | 72.56 ms |
| raw FlashAttention combine | 20.81 ms |
| `reshape_and_cache_flash_kernel` | 12.25 ms |

ByteV2 eager：

| kernel | total |
| --- | ---: |
| `Kernel2` | 2248.25 ms |
| GEMM `64x64_sliced1x2` | 1144.41 ms |
| ByteV2 CUTE stage1 | 292.14 ms |
| GEMM `64x64_ldg8` | 233.55 ms |
| `byte_v2_decode_append_cache_kernel` | 34.49 ms |
| ByteV2 split reduce | 13.15 ms |
| `byte_v2_init_decode_append_result_kernel` | 6.21 ms |
| `byte_v2_record_deferred_cache_update_error_kernel` | 6.08 ms |

### API 对比

raw eager API：

| API | calls | total |
| --- | ---: | ---: |
| `cudaEventSynchronize` | 128 | 1004.81 ms |
| `cudaLaunchKernel` | 46692 | 314.42 ms |
| `cuLaunchKernel` | 8128 | 50.77 ms |
| `cudaMemcpyAsync` | 525 | 4.92 ms |

ByteV2 eager API：

| API | calls | total |
| --- | ---: | ---: |
| `cudaMemcpyAsync` | 5005 | 1056.50 ms |
| `cudaLaunchKernel` | 51396 | 348.60 ms |
| `cuLaunchKernel` | 8128 | 52.47 ms |
| `cudaStreamSynchronize` | 320 | 2.81 ms |
| `cudaEventSynchronize` | 128 | 1.45 ms |

解释：

- API 表中 raw 的 `cudaEventSynchronize` 和 ByteV2 的 `cudaMemcpyAsync` 是 host
  侧等待/同步开销表现，不应直接相加到 GPU kernel delta 上。
- Step 44 已经把 ByteV2 的 per-token validation sync 大幅减少：
  `cudaStreamSynchronize` 从 8320 次降到 320 次。
- 当前 ByteV2 API 仍有更多 `cudaMemcpyAsync`，但从 GPU kernel delta 看，
  E2E 剩余差距主要不是 API，而是 attention stage1。

### 结论

当前剩余 `6%-8%` 差距的来源排序：

1. **attention stage1 是第一瓶颈。**
   ByteV2 attention `305.29 ms`，raw FlashAttention `93.37 ms`，多
   `211.92 ms`，解释了绝大多数 GPU kernel delta。
2. **cache update 是第二瓶颈，但量级明显小。**
   ByteV2 cache update `48.25 ms`，raw `12.25 ms`，多 `36.00 ms`。
   其中 deferred sticky error record 和 result init 约 `12.29 ms`。
3. **GEMM/MLP 不是差距来源。**
   两边只差 `+1.78 ms`，基本持平。
4. **sampling/scheduler 不是主要差距。**
   只差 `+2.30 ms`。
5. **cudagraph 缺失不是当前主要解释。**
   已有 production/cudagraph baseline 中 ByteV2 仍为 raw 的 `92%-93%`，与本轮
   eager 的 `93.3%` 接近。ByteV2 backend 当前也声明
   `UNIFORM_SINGLE_TOKEN_DECODE` cudagraph support。

### 下一步

不要把下一轮重点放在 sampling、GEMM 或单纯 cudagraph 开关上。建议：

1. 继续做 ByteV2 attention stage1 的结构性优化：
   - K/V decode 指令数；
   - shared layout / WMMA load pattern；
   - PV/write partial；
   - split reduce partial 写回和扫描成本。
2. cache update 可以做一个小步 fusion：
   把 `init_decode_append_result` 和
   `record_deferred_cache_update_error` 融入
   `byte_v2_decode_append_cache_kernel`，但预期上限只有约 `12 ms / 4.1s`
   量级，必须用 E2E 保留。
3. 等 GPU 空闲后，重跑干净的 p512/b4/d128 production nsys，确认 cudagraph
   baseline 是否仍是 `92%-93% raw`。

### GPU0 空闲后的 production/cudagraph 复测

GPU0 空闲后重新跑了干净的 non-eager/cudagraph E2E 和 Nsight Systems profile。
workload 固定为：

```text
model=/mnt/sda1/yxz/byte_v2/Meta-Llama-3-8B-Instruct
prompt_len=512
batch_size=4
decode_len=128
gpu_memory_utilization=0.80
enforce_eager=false
ByteV2: compressed-only, fallback_pool_ratio=0.03, no arena,
        split_k=8, CUTE metadata auto, deferred cache update error on
```

输出文件：

- E2E:
  `benchmarks/profiles/step45_cudagraph_clean_p512_b4_d128_gpu0.json`
- raw nsys:
  `benchmarks/profiles/nsys_step45_cudagraph_raw_clean_p512_b4_d128.nsys-rep`
- ByteV2 nsys:
  `benchmarks/profiles/nsys_step45_cudagraph_bytev2_clean_p512_b4_d128.nsys-rep`

E2E 结果：

| mode | tok/s | elapsed | ByteV2/raw |
| --- | ---: | ---: | ---: |
| raw production/cudagraph | 133.55 | 3.834s | 100% |
| ByteV2 compressed-only production/cudagraph | 125.15 | 4.091s | 93.7% |

ByteV2 sparse fallback pool 未耗尽：

```text
any_exhausted=false
total_assigned_blocks=256
total_capacity=16384
max_next_slot=40
```

raw 和 ByteV2 日志中都完成了 cudagraph capture，包括 `PIECEWISE` 和 `FULL`
graph。因此本轮确认：当前 `6%-7%` E2E 差距不是因为 ByteV2 没有进入
cudagraph production 路径。

production/cudagraph nsys kernel 分桶：

| category | raw | ByteV2 | delta |
| --- | ---: | ---: | ---: |
| attention | 93.30 ms | 304.19 ms | +210.89 ms |
| cache update | 9.99 ms | 42.20 ms | +32.20 ms |
| GEMM/MLP/linear | 3673.36 ms | 3674.40 ms | +1.03 ms |
| norm/rope/activation/elementwise | 62.02 ms | 43.96 ms | -18.07 ms |
| sampling/scheduler misc | 5.25 ms | 6.59 ms | +1.34 ms |
| other | 0.63 ms | 0.73 ms | +0.10 ms |
| total GPU kernel | 3844.55 ms | 4072.06 ms | +227.51 ms |

top attention/cache kernels：

| path | kernel | total | launches | avg |
| --- | --- | ---: | ---: | ---: |
| raw attention | `flash_fwd_splitkv_kernel` | 76.09 ms | 4096 | 18.58 us |
| raw attention | `flash_fwd_splitkv_combine_kernel` | 17.21 ms | 4064 | 4.23 us |
| raw cache update | `reshape_and_cache_flash_kernel` | 9.99 ms | 4096 | 2.44 us |
| ByteV2 attention | `byte_v2_paged_decode_attention_gqa_cute_split_stage1_kernel` | 291.01 ms | 4096 | 71.05 us |
| ByteV2 attention | `byte_v2_paged_decode_attention_split_reduce_parallel_kernel` | 13.18 ms | 4096 | 3.22 us |
| ByteV2 cache update | `byte_v2_decode_append_cache_kernel` | 31.70 ms | 4064 | 7.80 us |
| ByteV2 cache update | `byte_v2_record_deferred_cache_update_error_kernel` | 4.85 ms | 4064 | 1.19 us |
| ByteV2 cache update | `byte_v2_init_decode_append_result_kernel` | 4.08 ms | 4064 | 1.00 us |
| ByteV2 prefill encode | `byte_v2_prefill_direct_encode_blocks_kernel` | 1.46 ms | 32 | 45.63 us |

production API summary 的主要差异：

| API | raw | ByteV2 |
| --- | ---: | ---: |
| `cudaGraphLaunch_v10000` | 79.97 ms / 127 calls | 84.02 ms / 127 calls |
| `cudaLaunchKernel` | 14.05 ms / 1476 calls | 16.76 ms / 2084 calls |
| `cudaMemcpyAsync` | 4.86 ms / 525 calls | 3818.86 ms / 941 calls |
| `cudaEventSynchronize` | 3519.65 ms / 128 calls | 1.48 ms / 128 calls |
| `cudaStreamSynchronize` | not top | 2.79 ms / 320 calls |

API 表中的 raw `cudaEventSynchronize` 和 ByteV2 `cudaMemcpyAsync` 都包含 host
等待行为，不能简单与 GPU kernel 时间相加。不过 `cudaGraphLaunch` 两边接近，
GPU kernel delta 又主要集中在 attention/cache update，因此 host/API 或
cudagraph 缺失不是当前主因。

更新后的结论：

1. **attention stage1 仍是绝对主瓶颈。** production 下 ByteV2 attention 比
   raw 多 `210.89 ms`，解释了大部分 `227.51 ms` GPU kernel delta。
2. **cache update 是第二瓶颈。** production 下 ByteV2 比 raw 多 `32.20 ms`。
   其中 append kernel 是主要项，init/error record 的 launch/node 成本仍可继续
   fusion。
3. **GEMM/MLP、sampling 和 cudagraph 不是主要差距来源。** GEMM/MLP 只差
   `1.03 ms`，两边都完成 cudagraph capture，E2E ratio 提升到 `93.7% raw`。
4. 下一步仍应优先优化 `byte_v2_paged_decode_attention_gqa_cute_split_stage1_kernel`
   的 K/V decode、shared/WMMA load、PV/write partial；cache update fusion
   作为第二优先级的小步实验。

## Step 46：CUTE metadata stage1 early-exit + NCU 复测

### 目标

Step 45 的 production profile 已确认剩余差距主要来自
`byte_v2_paged_decode_attention_gqa_cute_split_stage1_kernel`。本轮不先改代码，
而是重新拆分当前最优 CUTE metadata stage1 内部耗时，确认下一步优化应该落在
loader、QK、softmax 还是 PV/write。

### 配置

固定当前 production 相同的核心配置：

```text
GPU=0
num_heads=32
num_kv_heads=8
head_size=128
head_size_v=128
block_size=16
split_k=8
fallback_ratio=0.03
fallback_pattern=single_outlier
tile_fallback_pool=on
cute_stage1_auto=on
tile_fastpath=on
parallel_reduce=on
outlier_arena=off
```

decode-only early-exit 输出：

```text
benchmarks/profiles/step46_cute_early_summary.json
benchmarks/profiles/step46_cute_early_p512_b4_d128_m{0,1,2,3}.json
benchmarks/profiles/step46_cute_early_p512_b4_d256_m{0,1,2,3}.json
benchmarks/profiles/step46_cute_early_p2048_b1_d32_m{0,1,2,3}.json
```

NCU 输出：

```text
benchmarks/profiles/step46_ncu_cute_stage1_p512_b4_d128_m{0,1,2,3}.csv
benchmarks/profiles/step46_ncu_cute_stage1_p2048_b1_d32_m{0,1,2,3}.csv
benchmarks/profiles/step46_ncu_cute_stage1_p512_b4_d128_m0.ncu-rep
benchmarks/profiles/step46_ncu_cute_stage1_p512_b4_d128_m0_source.txt
benchmarks/profiles/step46_ncu_cute_stage1_p512_b4_d128_m0_details.txt
```

### decode-only early-exit timing

注意：decode-only benchmark 调用完整 op，因此 mode 0-3 的 median latency 包含
stage1 之外的 reduce/op overhead。这里主要看差分趋势；更准确的单 kernel duration
见下面 NCU。

| workload | mode0 full | mode1 load/decode | QK inc | softmax inc | PV/write inc |
| --- | ---: | ---: | ---: | ---: | ---: |
| p512/b4/d128 | 89.09 us | 73.73 us | 5.12 us | 2.05 us | 8.19 us |
| p512/b4/d256 | 90.11 us | 71.68 us | 4.10 us | 2.05 us | 12.29 us |
| p2048/b1/d32 | 202.75 us | 155.65 us | 10.24 us | 9.22 us | 27.65 us |

粗略结论：load/decode 仍是最大段；长 context / batch=1 下 PV/write 和 QK/softmax
增量会变大，但仍小于 load/decode。

### NCU stage1 duration

NCU 直接过滤
`byte_v2_paged_decode_attention_gqa_cute_split_stage1_kernel`，每个 mode 采一个
stage1 launch。

| workload | full | load/decode | QK inc | softmax inc | PV/write inc |
| --- | ---: | ---: | ---: | ---: | ---: |
| p512/b4/d128 | 73.22 us | 54.27 us | 4.42 us | 2.75 us | 11.78 us |
| p2048/b1/d32 | 234.50 us | 178.85 us | 12.93 us | 10.62 us | 32.10 us |

占比：

| workload | load/decode | QK | softmax | PV/write |
| --- | ---: | ---: | ---: | ---: |
| p512/b4/d128 | 74.1% | 6.0% | 3.8% | 16.1% |
| p2048/b1/d32 | 76.3% | 5.5% | 4.5% | 13.7% |

结论很明确：当前 stage1 的主瓶颈仍是 compressed K/V loader + decode +
shared 写入路径，而不是 QK、softmax 或 split reduce。

### NCU 关键指标

| workload | mode | duration | executed inst | memory throughput | DRAM | L1 hit | L2 hit | regs/thread | waves/SM | achieved occ |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| p512/b4/d128 | full | 73.22 us | 6.43M | 115.6 GB/s | 17.30% | 68.19% | 15.41% | 96 | 0.61 | 25.22% |
| p512/b4/d128 | load/decode | 54.27 us | 4.96M | 155.6 GB/s | 23.19% | 68.21% | 15.14% | 96 | 0.61 | 25.13% |
| p2048/b1/d32 | full | 234.50 us | 6.00M | 36.4 GB/s | 5.32% | 69.73% | 9.53% | 96 | 0.15 | 8.33% |
| p2048/b1/d32 | load/decode | 178.85 us | 4.56M | 50.5 GB/s | 7.38% | 69.73% | 9.49% | 96 | 0.15 | 8.33% |

NCU warnings：

| workload | warning | value |
| --- | --- | ---: |
| p512/b4/d128 full | uncoalesced global excessive sectors | 319856 / 686464 = 47% |
| p512/b4/d128 full | uncoalesced shared excessive wavefronts | 696320 / 1070608 = 65% |
| p2048/b1/d32 full | uncoalesced global excessive sectors | 317168 / 663808 = 48% |
| p2048/b1/d32 full | uncoalesced shared excessive wavefronts | 696320 / 1051024 = 66% |

其他重要信号：

- p512/b4/d128 full 的 long scoreboard warning：平均每 warp 约 `6.4 cycles`
  等待 L1TEX scoreboard，占 issue 间隔约 `47.6%`。
- p512/b4/d128 load/decode 也有同样的 uncoalesced global excessive sectors，
  说明大量问题发生在 loader 本身，而不是后面的 QK/PV。
- p2048/b1/d32 只有 `0.15 waves/SM`、`8.33% achieved occupancy`，长 context
  batch=1 下并行度明显不足；这会放大单 CTA 处理过多 page 的 latency。
- 当前 `.ncu-rep` 的 source page 只能看到 SASS 地址，没有 CUDA 源码行映射。
  这说明当前扩展构建缺少可用 lineinfo；后续若要做真正源码行级 NCU，需要重新
  用 lineinfo/debug 编译 CUDA 扩展。

### 本轮结论

1. **下一步应优先优化 compressed K/V loader。**
   load/decode 在两个代表 workload 中占 stage1 的 `74%-76%`。
2. **shared layout 仍是明确问题。**
   full stage1 有 `65%-66%` shared excessive wavefronts，PV/write 和 WMMA load
   都可能受 shared layout 影响。
3. **global access pattern 仍是明确问题。**
   full 与 load/decode mode 都有约 `47%-48%` excessive global sectors，说明
   compressed payload/tile metadata 读取没有充分合并。
4. **低 batch/长 context 的并行度不足仍存在。**
   p2048/b1 只有 `0.15 waves/SM`，继续优化单 CTA 内部指令虽有意义，但后续也
   需要考虑 page-chunk 粒度或更高 CTA 并行度。

上一轮曾尝试一个最小 no-arena overlay-skip patch：在 CUTE metadata
tile-fastpath 中，当没有 outlier arena 时跳过
`byte_v2_overlay_*_tile_outliers_to_shared()`。Step 47 已完成编译后验证：
decode-only 有小幅正向信号，但 NCU stage1 指标几乎完全不变，E2E 略退化。
因此该 patch 已按保留门槛回滚，不保留代码入口。

### 下一步实验

下一轮只做一个小 patch，目标是 loader/data-movement：

1. 为 CUTE metadata stage1 增加一个 opt-in 的 no-arena compressed tile loader
   fast path：
   - 当 `outlier_arena == nullptr` 且 tile header 无 fallback 时，完全跳过
     outlier bitmap/meta 指针准备和 overlay helper 调用；
   - K/V loader 分离为专用 helper，避免 runtime `is_value`/overlay 分支；
   - 继续保留 tile fallback path，不能破坏 correctness。
2. 保留门槛：
   - p512/b4/d128 NCU load/decode duration 下降；
   - full stage1 duration 下降；
   - uncoalesced global sectors 或 executed instructions 至少一项下降；
   - p512/b4/d128 E2E 不退化。
3. 如果该 fast path 没有明确收益，回滚，不继续在这一层堆分支。

## Step 47：no-arena overlay-skip 实验结果

### 实验目标

验证一个最小 opt-in patch：在 CUTE metadata stage1 的 no-arena production
路径中，如果 tile 没有 fallback，则跳过
`byte_v2_overlay_*_tile_outliers_to_shared()` helper 调用。

### correctness

编译后通过了最相关的 CUDA smoke：

```text
tests/v1/attention/test_byte_v2_decode.py::test_byte_v2_paged_decode_attention_op_cute_stage1_metadata_cuda PASSED
tests/v1/attention/test_byte_v2_decode.py::test_byte_v2_paged_decode_attention_op_cute_stage1_auto_metadata_cuda PASSED
tests/v1/attention/test_byte_v2_decode.py::test_byte_v2_paged_decode_attention_op_cute_stage1_auto_long_context_cuda PASSED
3 passed in 34.24s
```

### decode-only A/B

workload：`p512/b4/d128`，`split_k=8`，`fallback_ratio=0.03`，
`single_outlier`，tile fallback pool，CUTE auto，parallel reduce on。

| variant | median us | p90 us | tok/s |
| --- | ---: | ---: | ---: |
| baseline | 94.21 | 102.40 | 42,459 |
| overlay-skip | 89.09 | 91.14 | 44,899 |
| baseline repeat 1 | 91.14 | 93.18 | 43,890 |
| overlay-skip repeat 1 | 89.09 | 92.16 | 44,899 |
| baseline repeat 2 | 90.11 | 91.14 | 44,389 |
| overlay-skip repeat 2 | 89.09 | 90.11 | 44,899 |

decode-only 有小幅正向信号，但幅度接近短 kernel benchmark 的抖动区间。

### NCU A/B

同一 workload 抓取一个 CUTE stage1 launch：

| metric | baseline | overlay-skip |
| --- | ---: | ---: |
| Duration | 74.240 us | 74.016 us |
| Executed Instructions | 6,514,096 | 6,514,096 |
| Issued Instructions | 6,522,481 | 6,522,506 |
| Branch Instructions | 715,824 | 715,824 |
| Registers / thread | 96 | 96 |
| Waves / SM | 0.61 | 0.61 |
| Achieved occupancy | 25.22% | 25.23% |
| Excessive global sectors | 319,856 / 686,464, 47% | 319,856 / 686,464, 47% |
| Excessive shared wavefronts | 696,320 / 1,070,608, 65% | 696,320 / 1,070,608, 65% |

NCU 没有支持该 patch 的证据：stage1 duration 基本不变，执行指令数、分支数、
global/shared uncoalesced 指标完全不变。

### E2E

workload：Llama-3 8B，`p512/b4/d128`，ByteV2 compressed-only，
`fallback_ratio=0.03`，CUTE auto，cudagraph enabled。

| variant | elapsed s | output tok/s | fallback exhausted |
| --- | ---: | ---: | --- |
| baseline | 4.09556 | 125.013 | false |
| overlay-skip | 4.09848 | 124.924 | false |

E2E 没有收益，且略低于 baseline。

### 结论

不保留该 patch。虽然 decode-only median 有小幅改善，但 NCU stage1 指标不变，
E2E 没有兑现。按保留门槛，已删除：

- `VLLM_BYTE_V2_DECODE_CUTE_NO_ARENA_OVERLAY_SKIP`
- `--cute-no-arena-overlay-skip`
- CUTE stage1 kernel 的 `no_arena_overlay_skip` 参数
- tile-fastpath 中的 overlay helper 条件跳过逻辑

后续不要继续在“空 outlier arena helper 早退”这一层做优化；主要矛盾仍是
K/V load-decode 访问模式、shared layout/WMMA load pattern 和低并行度。

## Step 48：CUTE stage1 lineinfo NCU 源码行级 profile

### 目标

上一轮 Step 46 的 early-exit profile 已经确认 stage1 的主要时间在
compressed K/V load/decode，但当时 `.ncu-rep` 没有 CUDA source line 映射，
只能看到 SASS 地址。本轮用带 `-lineinfo` 的 `_C_stable_libtorch` 临时构建产物
重新 profile，目标是把 global/shared/stall 热点映射回
`csrc/libtorch_stable/cache_kernels.cu` 的具体源码行。

### 构建和验证

全局 `CMAKE_CUDA_FLAGS=-lineinfo uv pip install -e . --torch-backend=auto`
会触发大量 CUDA target 重编，成本很高；本轮最终复用临时 CMake build dir，
只链接 `_C_stable_libtorch`：

```text
cmake --build /tmp/tmpvfglk_3g.build-temp --target _C_stable_libtorch -j 8
cp /tmp/tmpvfglk_3g.build-temp/_C_stable_libtorch.abi3.so vllm/_C_stable_libtorch.abi3.so
```

smoke：

```text
p512/b4/d128 decode-only:
median_us=88.06, p90_us=92.16, tok/s=45421.51
fallback_blocks=0/8, fallback_tiles=4/1024
```

说明 lineinfo 临时构建产物可以正常加载和执行。

### NCU workload

均抓取一个 CUTE metadata stage1 launch：

```text
p512/b4/d128, split_k=8, fallback_ratio=0.03, single_outlier
p2048/b1/d32, split_k=8, fallback_ratio=0.03, single_outlier
```

报告文件：

```text
benchmarks/profiles/step48_lineinfo_cute_stage1_p512_b4_d128.ncu-rep
benchmarks/profiles/step48_lineinfo_cute_stage1_p2048_b1_d32.ncu-rep
benchmarks/profiles/step48_lineinfo_source_p512_b4_d128.txt
benchmarks/profiles/step48_lineinfo_source_p2048_b1_d32.txt
```

### stage1 总体指标

| workload | duration | executed inst | long scoreboard | issue active | regs/thread | waves/SM | achieved occ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| p512/b4/d128 | 73.376 us | 6.431M | 6.35 cycle/issue | 0.23 | 96 | 0.61 | 25.23% |
| p2048/b1/d32 | 233.728 us | 6.004M | 6.35 cycle/issue | 0.08 | 96 | 0.15 | 8.33% |

| workload | DRAM throughput | L1/TEX hit | L2 hit | global excessive sectors | shared excessive wavefronts |
| --- | ---: | ---: | ---: | ---: | ---: |
| p512/b4/d128 | 17.15% | 68.20% | 15.50% | 319,856 / 686,464 = 47% | 696,320 / 1,070,608 = 65% |
| p2048/b1/d32 | 5.36% | 69.74% | 7.39% | 317,168 / 663,808 = 48% | 696,320 / 1,051,024 = 66% |

NCU 的规则提示和 Step 46 一致：

- `CPIStall`：每 warp 约 `6.3-6.4 cycles` 在等待 L1TEX/global/local
  scoreboard，占总 issue 间隔约 `47.5%-51.2%`。
- `UncoalescedGlobalAccess`：global excessive sectors 约 `47%-48%`。
- `UncoalescedSharedAccess`：shared excessive wavefronts 约 `65%-66%`。
- `SOLBottleneck`：grid 太小，p512 只有 `0.6 waves/SM`，p2048/b1 只有
  `0.2 waves/SM`。

### 源码行热点

1. compressed payload 的 `byte_v2_load_u16()` 是主要 global excessive 来源：

```cpp
// cache_kernels.cu:1051-1054
__device__ __forceinline__ uint16_t byte_v2_load_u16(
    const uint8_t* __restrict__ ptr) {
  return static_cast<uint16_t>(ptr[0]) |
         (static_cast<uint16_t>(ptr[1]) << 8);
}
```

source/SASS 显示 line 1053/1054 对应大量 `LDG.E.U8`，并贡献主要
global excessive sectors：

| source line | 含义 | p512 signal | p2048 signal |
| --- | --- | ---: | ---: |
| 1053 | `ptr[0]` U8 load | global excessive 约 193K | global excessive 约 193K |
| 1054 | `ptr[1]` U8 load | global excessive 约 127K | global excessive 约 127K |

这说明当前 2 个 byte 分别读取再 `PRMT/LOP3` 拼接的方式没有被合并成理想的
16/32-bit 连续读取；compressed payload 的低字节读取本身就是主要 global
访问模式问题，而不是 outlier metadata 分支造成的。

2. K/V tile decode helper 仍有明显指令成本：

```cpp
// K tile: cache_kernels.cu:1248-1268
const uint16_t low_pair = byte_v2_load_u16(low_ptr + elem0);
const int packed = static_cast<int>(packed_ptr[pair_idx]);
bits0 = byte_v2_make_bf16_bits_from_fast_code(...);
bits1 = byte_v2_make_bf16_bits_from_fast_code(...);
k_shared[shared_offset] = ...
k_shared[shared_offset + 1] = ...

// V tile: cache_kernels.cu:1317-1335
const uint16_t low_pair = byte_v2_load_u16(low_ptr + elem0);
const int packed = static_cast<int>(packed_ptr[pair_idx]);
...
v_shared[shared_offset] = ...
v_shared[shared_offset + 1] = ...
```

high-level line instruction counts 中，`low >> 7` / `code & 0x07`
等 decode 行仍有约 `131K` 级别执行次数；说明 decode 逻辑仍是 stage1 的
重要非带宽开销。

3. shared layout 问题主要出现在 V shared 写入和 WMMA/PV shared 读取：

```cpp
// cache_kernels.cu:1334-1335
v_shared[shared_offset] = ...
v_shared[shared_offset + 1] = ...

// cache_kernels.cu:5718-5719
nvcuda::wmma::load_matrix_sync(
    b_frag, k_shared + dim_base * kByteV2TileSize, kByteV2TileSize);

// cache_kernels.cu:5790-5792
nvcuda::wmma::load_matrix_sync(p_frag, p_shared, kByteV2TileSize);
nvcuda::wmma::load_matrix_sync(v_frag, v_shared + dim_base, head_size_v);
```

特别是 V shared store 的 SASS 行显示单条 store 有 `98,304`
级别 shared excessive wavefront；PV 的 C++ `load_matrix_sync` 展开到大量
shared `LD.E`，每条约 `7,168 / 8,192` wavefront，ideal 只有 `1,024`。
这与之前“row-major K + WMMA col-major / stride padding 退化”的结果一致：
简单改 stride 不够，必须让 shared layout 和实际 `ldmatrix/load_matrix_sync`
消费顺序一起设计。

4. softmax/PV 标量 shared 数组也有开销，但不是首要矛盾：

```cpp
// cache_kernels.cu:5759-5761
tile_acc_factor[warp_id] = old_scale;
denom_shared[warp_id] = ...
running_max_shared[warp_id] = new_max;

// cache_kernels.cu:5772
acc[q] *= tile_acc_factor[q];
```

这些行有 shared excessive wavefront，但绝对值远小于总 shared excessive；
优化它们可能有收益，但不应优先于 payload load 与 WMMA shared layout。

5. p2048/b1 的低并行度非常明确：

```text
p512/b4/d128 grid_size = 256, waves/SM = 0.61, achieved occ = 25.23%
p2048/b1/d32 grid_size = 64,  waves/SM = 0.15, achieved occ = 8.33%
```

长 context/低 batch 下，即使单 CTA 内部优化有效，也会被并行度不足放大。
因此后续优化不能只做单 CTA 内 micro-op 调整，还需要引入更细的 page chunk
并行或 persistent/page-parallel stage1。

### 结论

本轮 lineinfo profile 明确推翻了“主要是 outlier metadata 检查成本”的假设：
no-arena overlay-skip 上轮已经没有 NCU/E2E 收益，而本轮源码行显示最大 global
问题来自 compressed payload 的 `LDG.E.U8` 低字节读取本身。

后续优先级应调整为：

1. **payload load v3**：避免两次 U8 global load，尝试对齐的 U16/U32/warp-stripe
   payload 读取；先做 decode-only microbench，门槛必须看到 global excessive
   sectors 和 long scoreboard 下降。
2. **V shared / WMMA layout v3**：围绕实际 `load_matrix_sync` 展开的 shared
   `LD.E` pattern 设计 swizzle，而不是只改 stride。
3. **低 batch 并行度**：p2048/b1 需要更细 page-chunk 或 persistent/page-parallel
   stage1；否则 waves/SM 只有 `0.15`，单 CTA latency 再低也难接近 raw。
4. **secondary**：softmax/PV 标量 shared 数组可做 register 化或 warp shuffle 化，
   但它不是当前第一瓶颈。

下一轮建议先做一个很小的下限实验：只改 compressed tile loader 的 `low_pair`
读取方式，增加 opt-in `u16-aligned load` variant；如果 NCU 中 line 1053/1054 的
global excessive sectors 和 long scoreboard 不下降，就不接 E2E，不保留。

## Step 49：payload load v3 lower-bound：aligned U16 low payload load

### 目标

Step 48 的 lineinfo profile 显示 CUTE metadata stage1 的最大 global
uncoalesced 来源是 `byte_v2_load_u16()` 内部两次 `LDG.E.U8`：

```cpp
return static_cast<uint16_t>(ptr[0]) |
       (static_cast<uint16_t>(ptr[1]) << 8);
```

本轮做一个很小的下限实验：只在 CUTE stage1 compressed tile fastpath 中，把
`low` payload 的两字节读取改成 opt-in aligned `uint16_t` load。ByteV2 tile
layout 中：

```text
page header = 16B
tile payload = 386B
low payload start = page + 16 + tile_id * 386 + 2
```

因此 `low_ptr + elem0` 在当前格式下始终 2 字节对齐。实验实现为模板参数和
benchmark 开关，不在 hot loop 传 runtime layout bool。

保留门槛：

1. NCU 中 global excessive sectors 下降。
2. long scoreboard 下降。
3. stage1 duration 或 decode-only median 至少有稳定收益。

如果只降低 sector 但 stage1 duration 不降，则不保留。

### 构建和 correctness

实验实现后复用 lineinfo build dir 只重编 `_C_stable_libtorch`：

```text
cmake --build /tmp/tmpvfglk_3g.build-temp --target _C_stable_libtorch -j 8
cp /tmp/tmpvfglk_3g.build-temp/_C_stable_libtorch.abi3.so vllm/_C_stable_libtorch.abi3.so
```

correctness smoke：

```text
CUDA_VISIBLE_DEVICES=0 \
VLLM_BYTE_V2_DECODE_ALIGNED_U16_PAYLOAD_LOAD=1 \
.venv/bin/python -m pytest \
  tests/v1/attention/test_byte_v2_decode.py::test_byte_v2_paged_decode_attention_op_cute_stage1_metadata_cuda -q

结果：1 passed
```

说明 aligned loader 版本功能上可执行。

### decode-only A/B

GPU 当时有外部负载，因此 decode-only 只作为初筛，最终判断以 NCU 为准。

| workload | baseline median | aligned median | 结论 |
| --- | ---: | ---: | --- |
| p512/b4/split-k8，首轮 | 90.11 us | 73.73 us | 看似大幅提升，但疑似运行顺序/负载噪声 |
| p512/b4/split-k8，反向复跑 | 72.70 us | 73.73 us | aligned 略慢，p512 不稳定 |
| p2048/b1/split-k64，首轮 | 81.92 us | 72.70 us | aligned 较快 |
| p2048/b1/split-k64，反向复跑 | 79.87 us | 74.75 us | aligned 仍较快，但该 split-k 与 Step 48 profile 不一致 |
| p2048/b1/split-k8 | 194.56 us | 212.99 us | aligned 明显退化 |

decode-only 的信号不一致：p2048/split-k64 有正向，但和 Step 48 的
p2048/split-k8 lineinfo workload 不一致；p512 和 p2048/split-k8 都没有稳定收益。

### NCU A/B

捕获同一个 CUTE metadata stage1 kernel：

```text
p512/b4/d128, split_k=8, fallback_ratio=0.03, single_outlier
p2048/b1/d32, split_k=8, fallback_ratio=0.03, single_outlier
```

报告：

```text
benchmarks/profiles/step49_payload_u16_ncu_baseline_p512_b4.ncu-rep
benchmarks/profiles/step49_payload_u16_ncu_aligned_p512_b4.ncu-rep
benchmarks/profiles/step49_payload_u16_ncu_baseline_p2048_b1_s8.ncu-rep
benchmarks/profiles/step49_payload_u16_ncu_aligned_p2048_b1_s8.ncu-rep
```

关键指标：

| workload | variant | stage1 duration | inst | long scoreboard | issue active | waves/SM | global sectors | global excessive | shared excessive |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| p512/b4 | baseline | 72.960 us | 6.431M | 6.33 | 20.83% | 0.61 | 686,464 | 319,856 | 696,320 |
| p512/b4 | aligned U16 | 74.240 us | 6.693M | 6.04 | 21.51% | 0.61 | 494,000 | 127,392 | 696,320 |
| p2048/b1 | baseline | 234.976 us | 6.004M | 6.53 | 5.80% | 0.15 | 663,808 | 317,168 | 696,320 |
| p2048/b1 | aligned U16 | 238.560 us | 6.266M | 6.18 | 5.99% | 0.15 | 471,344 | 124,704 | 696,320 |

观察：

1. aligned U16 确实把 global sectors 降低约 `28%-29%`，把 global excessive
   sectors 降低约 `60%`。
2. long scoreboard 从 `6.33/6.53` 降到 `6.04/6.18`，方向正确但幅度有限。
3. executed instructions 增加约 `4.1%-4.4%`。
4. shared excessive wavefronts 完全不变，仍是 `696,320`。
5. stage1 duration 没有下降：p512 `+1.8%`，p2048 `+1.5%`。

source/SASS 确认 aligned 版本把主要 low payload 读取从成对 `LDG.E.U8` 变成
`LDG.E.U16`，但额外的数据整理/依赖成本抵消了 sector 改善。

### 处理结果

不保留本轮代码改动。

原因是该实验没有满足门槛：global excessive sectors 和 long scoreboard 改善了，
但 stage1 duration 没有改善，decode-only A/B 也不稳定。代码已回滚到默认
`byte_v2_load_u16()` 路径，重新编译并通过：

```text
CUDA_VISIBLE_DEVICES=0 \
.venv/bin/python -m pytest \
  tests/v1/attention/test_byte_v2_decode.py::test_byte_v2_paged_decode_attention_op_cute_stage1_metadata_cuda -q

结果：1 passed
```

### 结论和下一步

这个下限实验说明：

1. Step 48 对 global uncoalesced 来源的判断是正确的，`low` payload 读取确实是
   sector 热点。
2. 但只把两个 U8 load 换成一个 U16 load 不够，因为 stage1 当前还受 shared
   layout/WMMA load pattern、decode 指令依赖、低 waves/SM 共同限制。
3. 后续不要继续做单点 scalar loader 替换；如果要解决 payload load，应做真正的
   layout-v3/warp-stripe payload，使 load、decode、shared store 和 WMMA 消费顺序
   一起变好。

下一步优先级：

1. **shared/WMMA layout v3 lower-bound**：在现有 payload 格式不变的情况下，
   先验证能否降低 `696,320` shared excessive wavefronts；如果不能降低 shared
   指标，不接 E2E。
2. **payload layout-v3**：不是 `reinterpret_cast<uint16_t*>` 这种局部替换，而是
   重新排布 tile payload 为 warp/lane-stripe 读取格式，并同时设计 K/V 不同物理顺序。
3. **p2048/b1 page-parallel stage1**：当前 p2048 split-k8 只有 `0.15 waves/SM`，
   需要更细粒度 page chunk 并行；单 CTA 内 loader 优化无法解决并行度不足。

## Step 50：shared/WMMA lower-bound 复测

### 实验目标

Step 49 证明 aligned U16 payload loader 可以降低 global sectors，但不能降低
stage1 duration，且 `shared excessive wavefronts` 固定在 `696,320`。因此本轮不再
继续做单点 payload loader，而是先验证 shared/WMMA 结构是否真是下一层瓶颈。

代码中已经存在一个显式实验路径：

```text
VLLM_BYTE_V2_DECODE_FLASH_STAGE1=1
```

该路径使用 `byte_v2_paged_decode_attention_gqa4_h128_flash_split_stage1_kernel`，
仍然读取真实 compressed page metadata/fallback，但把 CUTE metadata kernel 中的
`scores/p_shared/pv_shared` 中间共享内存路径替换为 4 个 warp 的直接 online
softmax/PV。它可以作为 shared/WMMA lower-bound：如果它能降低 shared wavefronts
和 stage1 duration，说明 Step 49 后续应该围绕 shared/WMMA pipeline，而不是继续
做普通 scalar loader 微调。

本轮没有新增代码，只复用已有显式路径并做 A/B profile。当前 `_C_stable_libtorch`
仍是 lineinfo 构建，适合 NCU/source profile；正式性能数值仍需用普通 production
构建复测。

### correctness

```text
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest \
  tests/v1/attention/test_byte_v2_decode.py::test_byte_v2_paged_decode_attention_op_flash_stage1_raw_fallback_cuda \
  tests/v1/attention/test_byte_v2_decode.py::test_byte_v2_paged_decode_attention_op_flash_stage1_tile_fallback_cuda -q

结果：2 passed, 16 warnings
```

说明 flash-stage1 的 raw fallback 与 tile fallback 功能 smoke test 仍然可用。

### decode-only A/B

固定：

```text
fallback_ratio=0.03
fallback_pattern=single_outlier
tile_fallback_pool=on
outlier_arena=off
tile_fastpath=on
parallel_reduce=on
skip_correctness
```

结果：

| workload | CUTE metadata | flash-stage1 | 变化 |
| --- | ---: | ---: | ---: |
| p512/b4/split-k8 | 87.04 us / 45.96K tok/s | 79.87 us / 50.08K tok/s | +8.9% tok/s |
| p512/b4/split-k8，反向复跑 | 89.09 us / 44.90K tok/s | 64.51 us / 62.00K tok/s | +38.1% tok/s |
| p2048/b1/split-k8 | 204.80 us / 4.88K tok/s | 171.97 us / 5.82K tok/s | +19.1% tok/s |
| p2048/b1/split-k8，反向复跑 | 210.94 us / 4.74K tok/s | 171.01 us / 5.85K tok/s | +23.4% tok/s |
| p2048/b1/split-k64 | 93.18 us / 10.73K tok/s | 86.02 us / 11.63K tok/s | +8.3% tok/s |

decode-only 显示 flash-stage1 对 stage1 kernel 本身有稳定收益，尤其是
p2048/b1/split-k8 这种低并行度场景。

### NCU A/B

捕获：

```text
byte_v2_paged_decode_attention_gqa4_h128_cute_split_stage1_metadata_kernel
byte_v2_paged_decode_attention_gqa4_h128_flash_split_stage1_kernel
```

报告：

```text
benchmarks/profiles/step50_shared_wmma_ncu_cute_p512_b4_s8.ncu-rep
benchmarks/profiles/step50_shared_wmma_ncu_flash_p512_b4_s8.ncu-rep
benchmarks/profiles/step50_shared_wmma_ncu_cute_p2048_b1_s8.ncu-rep
benchmarks/profiles/step50_shared_wmma_ncu_flash_p2048_b1_s8.ncu-rep
```

关键指标：

| workload | variant | duration | inst | long scoreboard | barrier | issue active | regs | shmem | waves/SM | global excessive | shared excessive |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| p512/b4/s8 | CUTE | 73.440 us | 6.431M | 6.37 | 2.09 | 20.08% | 96 | 14,928 | 0.61 | 319,856 | 696,320 |
| p512/b4/s8 | flash | 62.912 us | 6.521M | 5.97 | 0.14 | 23.51% | 116 | 9,216 | 0.76 | 319,856 | 360,448 |
| p2048/b1/s8 | CUTE | 233.760 us | 6.004M | 6.38 | 1.70 | 5.94% | 96 | 14,928 | 0.15 | 317,168 | 696,320 |
| p2048/b1/s8 | flash | 194.336 us | 6.257M | 5.80 | 0.07 | 7.52% | 116 | 9,216 | 0.19 | 317,168 | 360,448 |

观察：

1. flash-stage1 把 shared excessive 从 `696,320` 降到 `360,448`，下降约 48%。
2. stage1 duration 下降：p512 `73.44 -> 62.91 us`，p2048 `233.76 -> 194.34 us`。
3. barrier stall 大幅下降，说明去掉 `p_shared/pv_shared` 和额外同步是有效的。
4. global excessive 完全不变，说明本轮解决的是 shared/WMMA 路径，不是 payload
   global load。
5. executed instructions 和 registers/thread 上升，flash-stage1 不是免费收益；
   它用更多寄存器和指令换掉 shared 往返。

这个 NCU 结果支持 Step 49 的判断：下一层瓶颈确实包含 shared/WMMA pipeline。

### E2E 复测

固定：

```text
model=/mnt/sda1/yxz/byte_v2/Meta-Llama-3-8B-Instruct
p512/b4/d64
gpu_memory_utilization=0.45
enforce_eager
split_k=8
sparse_fallback_pool_ratio=0.03
outlier_arena=off
```

结果：

| variant | run1 | run2 | fallback pool |
| --- | ---: | ---: | --- |
| CUTE metadata | 102.91 tok/s | 99.72 tok/s | not exhausted |
| flash-stage1 | 61.73 tok/s | 89.16 tok/s | not exhausted |

E2E 与 kernel/NCU 结论不一致：flash-stage1 kernel 本身更快，但整机吞吐低于 CUTE
metadata，且第一次 run 退化更明显。复跑后退化收敛，但仍低约 10%。因此不能把
flash-stage1 设为默认，也不能仅凭 stage1 microbench 判断 production 方向。

### 处理结果

本轮不修改 production default。保留现有显式实验开关
`VLLM_BYTE_V2_DECODE_FLASH_STAGE1=1`，不启用 auto heuristic。

原因：

1. correctness 与 kernel profile 都支持 flash-stage1 是有效 lower-bound。
2. 但 E2E 没有兑现 kernel 收益，说明还有跨层交互或调度/host/API/cache update
   影响。
3. 在解释 E2E 退化前，继续做 shared layout 大改风险较高。

### 下一步

下一步不应直接把 flash-stage1 默认打开，而应做 `CUTE metadata vs flash-stage1`
的 detailed E2E profile：

1. 用同一 production build 复测，排除 lineinfo build 对 E2E 的影响。
2. 对同一 workload 做 nsys/NVTX 分段：attention stage1、reduce、cache update、
   sampling/GEMM、host/API 间隔。
3. 检查 flash-stage1 是否因为 `regs/thread=116` 降低跨层并发或改变调度节奏。
4. 如果 E2E 退化来自非 attention 环节，则继续优化调度/launch/cudagraph；如果
   退化来自 flash-stage1 本身，则以 CUTE metadata 为 production baseline，只把
   flash-stage1 作为独立 lower-bound 参考。

## Step 51：CUTE vs flash-stage1 的 detailed nsys profile

### 实验目标

Step 50 出现了矛盾结果：

1. decode-only/NCU 中 flash-stage1 更快，shared excessive wavefronts 明显下降。
2. E2E 中 flash-stage1 没有兑现收益，p512/b4/d64 反而低于 CUTE metadata。

本轮目标是做 detailed profile，确认 E2E 差距到底来自 attention stage1、reduce、
cache update、GEMM/sampling，还是 host/API 或 cudagraph 缺失。

### profiling 方式和限制

先尝试用 NVTX capture range：

```text
nsys profile \
  --trace=cuda,nvtx,osrt,cublas \
  --capture-range=nvtx \
  --nvtx-capture=byte_v2_bench_measured \
  --capture-range-end=stop
```

该方式没有生成 report。改为全程抓取 parent/child 后，sqlite 里仍只有少量
CUDA runtime API，没有 `CUPTI_ACTIVITY_KIND_KERNEL`，说明 nsys 没有注入到 vLLM
的 EngineCore worker 进程。

继续尝试：

```text
nsys profile --trace-fork-before-exec=true ...
```

仍然没有捕获 worker 的 CUDA kernel 表。因此本轮为了得到 kernel-level 分解，
临时使用：

```text
VLLM_ENABLE_V1_MULTIPROCESSING=0
```

让 V1 EngineCore 在当前进程中运行。这个 profile 只用于 GPU kernel 分解，不作为
production E2E 吞吐结论；真实 production 默认仍是 multiprocessing。

当前 `_C_stable_libtorch` 仍是 lineinfo build，因此本轮结论用于定位，不用于正式
性能报告。若要做正式 production profile，需要先重新安装 extension：

```text
cd /mnt/sda1/yxz/byte_v2/vllm
uv pip install -e . --torch-backend=auto
```

### workload

```text
model=/mnt/sda1/yxz/byte_v2/Meta-Llama-3-8B-Instruct
p512/b4/d64
split_k=8
sparse_fallback_pool_ratio=0.03
outlier_arena=off
enforce_eager
gpu_memory_utilization=0.45
VLLM_ENABLE_V1_MULTIPROCESSING=0
```

生成的主要报告：

```text
benchmarks/profiles/step51_nsys_uniproc_cute_p512_b4_d64.nsys-rep
benchmarks/profiles/step51_nsys_uniproc_cute_p512_b4_d64.sqlite
benchmarks/profiles/step51_nsys_uniproc_flash_p512_b4_d64.nsys-rep
benchmarks/profiles/step51_nsys_uniproc_flash_p512_b4_d64.sqlite
```

### measured 区间总览

用 `byte_v2_bench_decode_len_64_run_0` NVTX range 切 measured 区间：

| variant | measured wall | summed GPU kernel | 说明 |
| --- | ---: | ---: | --- |
| CUTE metadata | 2408.8 ms | 2294.9 ms | 约 256 output tokens |
| flash-stage1 | 3506.4 ms | 3259.9 ms | 同 workload |

注意：这是 uniproc + lineinfo build + nsys 的 profiling-only 数值，绝对吞吐不作为
最终性能；这里主要看分类比例和两条路径差异。

### kernel 分类

| category | CUTE metadata | flash-stage1 | 差值 |
| --- | ---: | ---: | ---: |
| GEMM/linear | 2041.4 ms | 2868.9 ms | +827.4 ms |
| attention stage1 | 180.9 ms | 313.4 ms | +132.5 ms |
| other model kernels | 38.1 ms | 38.5 ms | +0.4 ms |
| decode cache update | 23.2 ms | 27.5 ms | +4.4 ms |
| attention reduce | 6.6 ms | 6.8 ms | +0.2 ms |
| sampling/scheduler kernels | 2.3 ms | 2.4 ms | +0.1 ms |
| prefill cache update | 1.5 ms | 1.5 ms | ~0 |

Top kernels：

| variant | kernel | calls | total | avg |
| --- | --- | ---: | ---: | ---: |
| CUTE | `Kernel2` | 4032 | 1259.0 ms | 312.3 us |
| CUTE | `ampere_bf16_s16816gemm_bf16_64x64_sliced1x2...` | 4032 | 622.3 ms | 154.4 us |
| CUTE | `byte_v2_paged_decode_attention_gqa_cute_split_stage1_kernel` | 2048 | 180.9 ms | 88.3 us |
| CUTE | `byte_v2_decode_append_cache_kernel` | 2016 | 17.1 ms | 8.5 us |
| CUTE | `byte_v2_paged_decode_attention_split_reduce_parallel_kernel` | 2048 | 6.6 ms | 3.2 us |
| flash | `Kernel2` | 4032 | 1824.0 ms | 452.4 us |
| flash | `ampere_bf16_s16816gemm_bf16_64x64_sliced1x2...` | 4032 | 879.6 ms | 218.2 us |
| flash | `byte_v2_paged_decode_attention_gqa4_h128_flash_split_stage1_kernel` | 2048 | 313.4 ms | 153.0 us |
| flash | `byte_v2_decode_append_cache_kernel` | 2016 | 21.5 ms | 10.7 us |
| flash | `byte_v2_paged_decode_attention_split_reduce_parallel_kernel` | 2048 | 6.8 ms | 3.3 us |

### 结论

1. E2E 路径中 flash-stage1 没有复现 decode-only microbench 的收益。相反，
   measured run 里 flash stage1 是 `313.4 ms`，CUTE stage1 是 `180.9 ms`。
2. flash run 的 GEMM/linear 时间也显著变慢，说明该次 E2E/profile 仍受运行环境、
   profiler 或调度噪声影响；但即使只看 ByteV2 attention stage1，flash 也不是
   production 可启用路径。
3. cache update 不是本轮主要问题：CUTE 约 `23.2 ms`，flash 约 `27.5 ms`。
4. attention reduce 更不是主因：两者都约 `6.6-6.8 ms`。
5. 这解释了 Step 50 的 E2E 反转：flash-stage1 的 decode-only lower-bound 不能
   代表真实 E2E path。当前生产 baseline 应继续使用 CUTE metadata。

### 处理结果

不启用 `VLLM_BYTE_V2_DECODE_FLASH_STAGE1` 默认值，也不增加 auto heuristic。
flash-stage1 继续只作为显式实验开关保留。

### 下一步

1. **先切 production build 后复测**：当前 lineinfo build 适合定位，不适合最终性能。
   需要用户手动执行：

   ```text
   cd /mnt/sda1/yxz/byte_v2/vllm
   uv pip install -e . --torch-backend=auto
   ```

2. **继续优化 CUTE metadata stage1，而不是 flash-stage1**：下一轮应围绕 CUTE
   path 做更小的 source-guided 优化，重点是降低 `180.9 ms` stage1，而不是改
   reduce/cache update。
3. **补一个不依赖 nsys worker 注入的 timing hook**：在 benchmark 或 ByteV2
   backend 中增加可选 CUDA event timing，直接记录 attention stage1/reduce/cache
   update 的 per-token/per-layer 时间。这样 production multiprocessing 下也能拿到
   分段时间，不依赖 nsys 是否成功注入 worker。

## Step 52: memory-bound 长上下文 E2E 对比 CUTE metadata split-stage1 与 raw

### 目的

验证在更偏 memory-bound 的长上下文 decode 场景中，ByteV2 压缩 KV cache 的读带宽
优势是否能在 E2E 吞吐中体现出来。

本轮不使用 `--enforce-eager`，保留 vLLM production CUDA graph 路径。ByteV2 使用当前
稳定最快的 CUTE metadata split-stage1，而不是 experimental flash-stage1：

```text
VLLM_BYTE_V2_ENABLE_SPARSE_FALLBACK_POOL=1
VLLM_BYTE_V2_SPARSE_FALLBACK_POOL_RATIO=0.03
VLLM_BYTE_V2_SPARSE_FALLBACK_POOL_MIN_BLOCKS=512
VLLM_BYTE_V2_USE_NATIVE_KERNELS=1
VLLM_BYTE_V2_DECODE_PAGE_FASTPATH=1
VLLM_BYTE_V2_DECODE_TILE_FASTPATH=1
VLLM_BYTE_V2_DECODE_CUTE_STAGE1_AUTO=1
VLLM_BYTE_V2_DECODE_FLASH_STAGE1=0
VLLM_BYTE_V2_DECODE_SPLIT_K=8
VLLM_BYTE_V2_DECODE_PARALLEL_REDUCE=1
VLLM_BYTE_V2_ENABLE_OUTLIER_ARENA=0
VLLM_BYTE_V2_OUTLIER_MAX_PER_TILE=0
VLLM_BYTE_V2_LOSSY_MAX_MISSES_PER_TILE=0
VLLM_BYTE_V2_PREFILL_DIRECT_SKIP_VALIDATION_SYNC=1
VLLM_BYTE_V2_DECODE_APPEND_SKIP_VALIDATION_SYNC=1
VLLM_BYTE_V2_DEFERRED_CACHE_UPDATE_ERROR_CHECK=1
```

### workload

模型：

```text
/mnt/sda1/yxz/byte_v2/Meta-Llama-3-8B-Instruct
```

workload：

```text
p4096/b1/d64
p4096/b1/d128
p8064/b1/d64
```

由于该 Llama-3 配置 `max_position_embeddings=8192`，不能直接运行
`prompt_len=8192 + decode_len=64`。因此 8K 场景使用 `prompt_len=8064`，
`decode_len=64`，总长度 `8128`，不超过模型最大位置。

输出文件：

```text
benchmarks/profiles/step52_memory_bound_raw_bytev2_cute_p4096_b1_d64_d128.json
benchmarks/profiles/step52_memory_bound_raw_bytev2_cute_p8064_b1_d64.json
```

### E2E 结果

| workload | raw tok/s | ByteV2 CUTE tok/s | ByteV2 / raw | raw elapsed | ByteV2 elapsed | pool exhausted |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| p4096/b1/d64 | 34.34 | 24.36 | 70.9% | 1.864 s | 2.627 s | false |
| p4096/b1/d128 | 34.45 | 24.57 | 71.3% | 3.715 s | 5.209 s | false |
| p8064/b1/d64 | 33.20 | 18.94 | 57.1% | 1.928 s | 3.379 s | false |

### 结论

1. 在本轮更偏 memory-bound 的长上下文 E2E 场景中，当前 ByteV2 CUTE metadata
   split-stage1 没有超过 raw vLLM。
2. 4K context 下 ByteV2 约为 raw 的 `71%`；接近 8K context 时下降到约 `57%`。
   如果压缩 KV 读带宽收益已经主导，长 context 应该缩小差距；实际结果相反。
3. sparse fallback pool 没有耗尽，因此本轮退化不是 pool exhaustion 导致。
4. 这进一步说明当前主要瓶颈仍在 ByteV2 decode attention stage1 的额外成本：
   compressed payload 解码、metadata 访问、shared/WMMA load pattern、split-stage
   调度和 partial reduce 共同吃掉了 KV 读带宽降低带来的收益。

### 下一步

不要把长上下文 memory-bound 作为已经能体现 ByteV2 优势的证据。下一步仍应优先：

1. 增加 production path 内的 CUDA event timing hook，直接在 E2E 中拆出
   attention stage1、reduce、decode cache update、GEMM/sampling 时间。
2. 围绕 `byte_v2_paged_decode_attention_gqa_cute_split_stage1_kernel` 做针对性优化，
   尤其是 load/decode pipeline 与 shared/WMMA layout。
3. 只有当 stage1 的 per-token/layer 时间接近 raw attention 后，再重新扩大到更长
   context、更多 batch 和更高并发验证压缩 KV 的读带宽收益。

## Step 53: 加大 decode_len 的长 decode E2E 测试

### 目的

Step 52 主要看长 context、短 decode。本轮进一步加大 decode_len，检查较长 decode
是否能摊薄调度/初始化开销，并让 ByteV2 压缩 KV 的读带宽优势在 steady-state 中体现。

测试仍然使用 production CUDA graph 路径，ByteV2 使用当前稳定 baseline：

```text
CUTE metadata split-stage1
VLLM_BYTE_V2_DECODE_CUTE_STAGE1_AUTO=1
VLLM_BYTE_V2_DECODE_FLASH_STAGE1=0
VLLM_BYTE_V2_DECODE_SPLIT_K=8
VLLM_BYTE_V2_DECODE_PARALLEL_REDUCE=1
outlier_arena=off
sparse_fallback_pool_ratio=0.03
```

输出文件：

```text
benchmarks/profiles/step53_long_decode_raw_bytev2_cute_p4096_b1_d256_d512_d1024.json
benchmarks/profiles/step53_long_decode_raw_bytev2_cute_p7168_b1_d256_d512.json
```

### E2E 结果

| workload | raw tok/s | ByteV2 CUTE tok/s | ByteV2 / raw | raw elapsed | ByteV2 elapsed | pool exhausted |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| p4096/b1/d256 | 34.24 | 24.54 | 71.7% | 7.477 s | 10.431 s | false |
| p4096/b1/d512 | 34.26 | 24.36 | 71.1% | 14.943 s | 21.019 s | false |
| p4096/b1/d1024 | 34.27 | 23.94 | 69.9% | 29.879 s | 42.779 s | false |
| p7168/b1/d256 | 33.62 | 20.14 | 59.9% | 7.615 s | 12.710 s | false |
| p7168/b1/d512 | 33.61 | 20.04 | 59.6% | 15.234 s | 25.549 s | false |

### 结论

1. 长 decode 没有让 ByteV2 追近 raw。`p4096/b1` 下，ByteV2 从 `d256` 的
   `71.7% raw` 轻微下降到 `d1024` 的 `69.9% raw`。
2. 更长 context 下差距进一步扩大。`p7168/b1` 下，ByteV2 只有约 `59.6%-59.9% raw`。
3. raw 在长 decode 下非常稳定：`p4096` 约 `34.24-34.27 tok/s`，`p7168` 约
   `33.61-33.62 tok/s`。ByteV2 随 context 变长明显下降，说明当前瓶颈仍随 KV
   context 扫描成本增长。
4. sparse fallback pool 没有耗尽，因此本轮不是 fallback pool capacity 问题。

### 对后续优化的影响

较长 decode 进一步排除了“短 decode 调度噪声掩盖 ByteV2 优势”的解释。当前差距更像是
每个 decode step 内的稳定额外成本：

```text
compressed K/V payload decode
metadata/fallback check
shared layout / WMMA load
split-stage partial write + reduce
```

下一步仍应优先做 production path 内的 CUDA event timing hook，把 E2E 中的
attention stage1、reduce、decode cache update、GEMM/sampling 分开计时，然后继续针对
CUTE metadata stage1 的 load/decode/shared/WMMA pipeline 做优化。

## Step 54: 增大 batch_size 的容量/并发压力 E2E 测试

### 目的

确认前面 `b1` 长 context/长 decode 场景是否已经是显存容量受限，并测试增大 batch 后
ByteV2 是否能因为压缩 KV cache 而体现容量或带宽优势。

需要区分两个概念：

```text
显存容量受限: KV cache tokens 接近 GPU cache capacity，raw 无法继续增大 batch/context。
HBM 带宽/attention 受限: attention 每步扫描 KV 的读带宽或解码流水线成为瓶颈。
```

本轮仍使用 production CUDA graph 路径，ByteV2 使用 CUTE metadata split-stage1。

### workload

```text
p4096/b4/d128,d256
p4096/b8/d128
```

输出文件：

```text
benchmarks/profiles/step54_batch_memory_raw_bytev2_cute_p4096_b4_d128_d256.json
benchmarks/profiles/step54_batch_memory_raw_bytev2_cute_p4096_b8_d128.json
benchmarks/profiles/step54_batch_memory_raw_bytev2_cute_p4096_b8_d128_eager.json
```

### b4 E2E 结果

| workload | raw tok/s | ByteV2 CUTE tok/s | ByteV2 / raw | raw elapsed | ByteV2 elapsed | pool exhausted |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| p4096/b4/d128 | 123.93 | 87.87 | 70.9% | 4.131 s | 5.827 s | false |
| p4096/b4/d256 | 125.17 | 90.17 | 72.0% | 8.181 s | 11.357 s | false |

raw 在 `b4` 下从 `b1` 的约 `34 tok/s` 扩展到约 `124-125 tok/s`，说明该点不是显存
容量受限。ByteV2 也能扩展到约 `88-90 tok/s`，但相对 raw 仍然只有约 `71%-72%`，
和 `b1` 基本一致。

初始化日志里的 KV capacity 也支持这一点：

```text
b4 raw:    GPU KV cache size 153,984 tokens, max concurrency for 8192 tokens/request 18.80x
b4 ByteV2: GPU KV cache size 191,328 tokens, max concurrency for 8192 tokens/request 23.36x
```

实际 `p4096/b4/d256` 只需要约 `4 * (4096 + 256) = 17,408` tokens，远低于容量上限。

### b8 结果

raw production 可以跑通：

| workload | mode | tok/s | elapsed | 备注 |
| --- | --- | ---: | ---: | --- |
| p4096/b8/d128 | raw production | 227.41 | 4.503 s | 跑通 |
| p4096/b8/d128 | raw eager | 223.79 | 4.576 s | 跑通 |

ByteV2 在 `b8` 下失败，production 和 eager 都失败在同一个位置：

```text
Byte-v2 native prefill direct cache update failed: invalid slot mapping
error_code=2
fallback_pool_used=0
fallback_pool_capacity=512
```

production 路径中失败发生在 CUDA graph capture；eager 路径中失败发生在 warmup/真实
prefill 执行。因此这不是单纯 cudagraph 问题，也不是 OOM。日志里 ByteV2 在 `b8`
仍报告：

```text
GPU KV cache size 173,248 tokens
Maximum concurrency for 8,192 tokens per request: 21.15x
```

实际 `p4096/b8/d128` 约 `8 * (4096 + 128) = 33,792` tokens，仍明显小于 KV cache
capacity。失败更像是 ByteV2 native prefill direct cache update 对大 batch /
`max_num_batched_tokens=32768` 的 slot mapping 兼容性 bug。

### 结论

1. 当前已测试的 `b1`、`b4`、`b8 raw` 场景都不是显存容量受限。raw 和 ByteV2 初始化
   日志里的 KV cache capacity 都显著高于实际请求 token 数。
2. 增大到 `b4` 后，ByteV2 没有相对 raw 变好，仍约 `71%-72% raw`。这说明当前主要
   差距不是 batch 太小导致的调度摊销问题。
3. `b8` 暴露了一个新的功能问题：ByteV2 native prefill direct cache update 的 slot
   mapping 在更大的 batched prefill 下会失败。这个问题需要先修复，否则无法继续
   验证更高 batch 的 ByteV2 E2E。
4. ByteV2 的 KV capacity 优势确实存在，例如 `b4` 下 ByteV2 报告的 cache tokens
   高于 raw；但在当前 batch/context 下还没有转化为吞吐优势。

### 下一步

1. 先修复 `p4096/b8` 的 ByteV2 prefill direct cache update `invalid slot mapping`。
   修复后至少验证：

   ```text
   p4096/b8/d128 production
   p4096/b8/d128 eager
   ```

2. 修复后再跑更接近显存容量边界的 workload，例如：

   ```text
   p4096/b16/d128
   p7168/b8/d128
   ```

3. 如果目标是证明压缩 KV 的容量优势，需要选择 raw 接近或达到 KV capacity 的场景；
   当前 `b4` 仍远离容量上限，只能说明高 batch 并没有自动解决 stage1 性能差距。

## Step 55: 长上下文优化执行计划

### 问题重新定性

当前 `90%+ raw` 与 `70% raw` 的差异主要来自 workload 变化，而不是已经确认的代码
退化：

| workload | raw tok/s | ByteV2 tok/s | ByteV2 / raw |
| --- | ---: | ---: | ---: |
| p512/b4/d128 | 133.55 | 125.15 | 93.7% |
| p4096/b1/d128 | 34.45 | 24.57 | 71.3% |
| p4096/b4/d128 | 123.93 | 87.87 | 70.9% |
| p7168/b1/d512 | 33.61 | 20.04 | 59.6% |

短 context 下，ByteV2 attention/cache-update 的额外开销被 GEMM、MLP、sampling 和
调度部分掩盖；长 context 下，每个 decode step 扫描更多 KV page，ByteV2 stage1 的
额外成本被放大：

```text
compressed K/V payload decode
metadata/fallback check
shared layout / WMMA load
split-stage partial write + reduce
decode append/cache update
```

因此长上下文优化不能继续依赖“加 batch 或加 decode_len 自动追近 raw”。下一步需要
按下面的顺序做。

### 执行顺序

#### 0. 短 context 哨兵复测

每轮长 context 优化前先跑：

```text
p512/b4/d128 production/cudagraph
p512/b4/d256 production/cudagraph
```

保留门槛：

```text
p512/b4/d128 ByteV2 >= 90% raw
p512/b4/d256 ByteV2 >= 90% raw
```

如果短 context 也掉到 80% 左右，先查回归，不继续判断长 context 优化。

#### 1. 修复 p4096/b8 prefill slot mapping

当前 `p4096/b8/d128` 下 raw 能跑通，但 ByteV2 production/eager 都失败：

```text
Byte-v2 native prefill direct cache update failed: invalid slot mapping
error_code=2
fallback_pool_used=0
fallback_pool_capacity=512
```

这不是 OOM，因为 ByteV2 仍报告 `GPU KV cache size 173,248 tokens`，实际 workload
约 `33,792 tokens`。需要修复 ByteV2 native prefill direct cache update 对大 batch /
chunked prefill / padding slot 的兼容性。

修复点：

```text
slot < 0: skip
slot >= num_gpu_blocks * block_size: sticky error
valid slot: 根据 slot_mapping[token_idx] 更新对应 physical block
不能假设 slot 连续、单调或无 padding
```

验证：

```text
unit: 连续 slot / 非连续 slot / -1 padding / 跨 block / batch=8
E2E: p4096/b8/d128 eager
E2E: p4096/b8/d128 production/cudagraph
```

#### 2. 增加 production CUDA event timing

nsys 对 multiprocessing worker 注入不稳定，因此需要默认关闭的 event timing hook：

```text
VLLM_BYTE_V2_PROFILE_EVENTS=1
```

输出到 benchmark JSON：

```text
prefill cache update total/avg
decode cache update total/avg
decode attention stage1 total/avg
decode attention reduce total/avg
sampling/output copy total
```

先跑：

```text
p512/b4/d128
p4096/b1/d128
p4096/b4/d128
p7168/b1/d128
```

保留门槛：

```text
profile off 零开销
profile on E2E 额外开销 < 3%
能稳定输出每段时间
```

#### 3. 长 context CUTE stage1 early-exit + NCU

在当前 CUTE metadata split-stage1 上复测：

```text
mode0: full stage1
mode1: load/decode K/V 后退出
mode2: load/decode K/V + QK 后退出
mode3: load/decode K/V + QK + softmax 后退出
mode4: full stage1 + PV/write partial
```

workload：

```text
p4096/b1
p4096/b4
p7168/b1
```

必须记录：

```text
stage1 duration
integer instructions
memory instructions
long scoreboard
global excessive sectors
shared excessive wavefronts
registers/thread
eligible warps/scheduler
```

这个结果决定后续优化分支。

#### 4. adaptive split/page parallel sweep

当前固定 `split_k=8` 不一定适合长 context。下一步做 sweep：

```text
p4096/b1/d128: split_k=8,16,32,64,128
p4096/b4/d128: split_k=4,8,16,32,64
p7168/b1/d128: split_k=16,32,64,128
```

目标是找到 stage1 并行度与 reduce 开销的平衡点，而不是盲目增大 split_k。

保留门槛：

```text
stage1 + reduce total 下降 >= 5%
p4096/b1 或 p4096/b4 E2E 提升 >= 3%
p512/b4 不退化超过 1%
```

#### 5. 按 profile 选择 kernel 优化方向

如果 `load/decode K/V` 最大：

```text
做 layout-v3 / vectorized decoder
microbench >= +5%
global excessive sectors 或 long scoreboard 必须下降
E2E >= +3%
```

如果 `partial write/reduce` 最大：

```text
做 Stream-K / 多 page chunk 内部 online softmax 累积
减少 partial output 写回和 reduce 输入规模
p4096/b1 stage1+reduce >= -8%
p7168/b1 stage1+reduce >= -10%
```

如果 metadata/fallback 检查最大：

```text
做 coarse kernel selection:
  compressed-only no-fallback kernel
  metadata/fallback kernel
不要重新默认启用 tile bitmap
```

如果 fused compressed stage1 仍然明显慢：

```text
做 active raw staging oracle:
  compressed KV 常驻 HBM
  active decode batch 临时解压到 raw BF16 staging
  decode attention 走 raw FlashAttention/FlashInfer
```

### active raw staging oracle

这不是立刻替换 production 的方案，而是长 decode 场景的系统级 oracle。它利用长
decode 中 prefix pages 会被重复读取的特点，把一次解压成本摊到多个 output token。

估算 Llama-3 8B raw KV 全层每 token：

```text
32 layers * 2(K,V) * 8 kv_heads * 128 dim * 2 bytes = 131,072 bytes/token
```

active staging 额外显存：

```text
p4096/b1 ~= 512 MiB
p4096/b4 ~= 2 GiB
p7168/b4 ~= 3.5 GiB
```

oracle workload：

```text
p4096/b1/d128,d512,d1024
p4096/b4/d128,d256
p7168/b1/d128,d512
```

判断：

```text
staging >= 90% raw: 可以作为长 decode hybrid 路线
staging 只在 d512/d1024 有效: 需要 decode_len threshold
staging 仍慢: 继续 fused compressed stage1
```

### 阶段目标

| 阶段 | 目标 |
| --- | --- |
| P0 | `p512/b4/d128` 保持 `>=90% raw` |
| P1 | `p4096/b8/d128` ByteV2 production/eager 跑通 |
| P2 | `p4096/b1,b4` 有 production event timing 分解 |
| P3 | `p4096/b4/d128` 从 `70.9% raw` 提升到 `>=80% raw` |
| P4 | `p7168/b1/d128/d512` 从 `~60% raw` 提升到 `>=70% raw` |
| P5 | active staging oracle 在长 decode 达到 `>=90% raw` 或明确失败 |

### 不优先继续做的方向

基于已有实验，下面方向暂时不作为长 context 主线：

```text
盲目继续调 split_k 单点参数
默认启用 flash-stage1
默认启用 tile bitmap
pair-interleaved 3B payload layout
只看 microbench 不看 E2E/NCU 的 decoder 小改
```

每个 patch 必须有明确保留门槛；如果 E2E 没收益且 NCU 指标没有对应改善，就不保留为
默认路径。

## Step 56：b8 prefill slot mapping 修复与 E2E 结果

时间：2026-06-14

### 修改内容

修复 `p4096/b8` ByteV2 在 prefill cache update 阶段报
`invalid slot mapping` 的问题。

根因是 `byte_v2_validate_prefill_direct_blocks_kernel` 把 direct prefill
fast path 的资格检查当成硬错误处理。该 fast path 假设每 16 个输入 token
正好映射到一个完整、连续、从 offset 0 开始的物理 block；但大 batch / 长
prompt 下 vLLM 的 `slot_mapping` 可能不是这种布局。旧实现即使
`VLLM_BYTE_V2_PREFILL_DIRECT_SKIP_VALIDATION_SYNC=1` 打开，也会强行进入
direct encode，最终在 direct kernel 中触发 `invalid slot mapping`。

本轮改动：

```text
csrc/libtorch_stable/cache_kernels.cu
  byte_v2_validate_prefill_direct_blocks_kernel:
    非 direct-friendly mapping 只标记 direct_ineligible
    不再写 cache update error

  host prefill direct branch:
    必须读取 direct_ineligible
    direct_ineligible == 0 才走 direct encode
    direct_ineligible != 0 自动落到已有 compressed-only generic path

tests/v1/attention/test_byte_v2_ops.py
  新增 split direct group CUDA 回归测试
```

注意：这个修复使 `VLLM_BYTE_V2_PREFILL_DIRECT_SKIP_VALIDATION_SYNC=1` 不再跳过
direct 资格判定同步。该同步只用于选择 direct/generic 路径；direct path 原本也会
同步读取 direct encode 结果，所以这是 correctness 优先的低风险代价。

### 验证

重编译：

```bash
uv pip install -e . --torch-backend=auto
```

结果：

```text
成功，构建耗时约 16m04s
```

单测：

```bash
CUDA_VISIBLE_DEVICES=0 \
.venv/bin/python -m pytest \
  tests/v1/attention/test_byte_v2_ops.py::test_byte_v2_reshape_and_cache_op_falls_back_for_split_direct_group_cuda \
  -q
```

结果：

```text
1 passed
```

### E2E 结果

#### p512/b4 哨兵，production/cudagraph

输出文件：

```text
benchmarks/profiles/step55_sentinel_after_fix_p512_b4_d128_d256.json
```

| workload | raw tok/s | ByteV2 tok/s | ByteV2/raw | raw elapsed | ByteV2 elapsed |
| --- | ---: | ---: | ---: | ---: | ---: |
| p512/b4/d128 | 133.354 | 118.225 | 88.66% | 3.839s | 4.331s |
| p512/b4/d256 | 132.657 | 121.061 | 91.26% | 7.719s | 8.459s |

结论：d256 仍超过 90%，d128 降到 88.66%，没有功能回归但 P0 的严格
`p512/b4/d128 >= 90% raw` 门槛未完全满足。后续需要确认这是否来自新增
direct 资格同步、运行噪声，还是当前 ByteV2 decode stage1 本身波动。

#### p4096/b8/d16 smoke，production/cudagraph

输出文件：

```text
benchmarks/profiles/step55_b8_smoke_p4096_b8_d16.json
```

结果：

```text
ByteV2 d16: 93.397 tok/s, elapsed 1.370s
fallback any_exhausted=false
fallback max_capacity=512
fallback max_next_slot=24
fallback total_capacity=16384
fallback total_next_slot=768
```

结论：之前的 `invalid slot mapping` 已消失，`p4096/b8` ByteV2 production 能跑通。

#### p4096/b8/d128，production/cudagraph

输出文件：

```text
benchmarks/profiles/step55_b8_production_p4096_b8_d128.json
```

| workload | raw tok/s | ByteV2 tok/s | ByteV2/raw | raw elapsed | ByteV2 elapsed |
| --- | ---: | ---: | ---: | ---: | ---: |
| p4096/b8/d128 | 227.251 | 136.637 | 60.13% | 4.506s | 7.494s |

fallback：

```text
any_exhausted=false
max_capacity=512
max_next_slot=80
total_capacity=16384
total_next_slot=2560
```

#### p4096/b8/d128，eager

输出文件：

```text
benchmarks/profiles/step55_b8_eager_p4096_b8_d128.json
```

| workload | raw tok/s | ByteV2 tok/s | ByteV2/raw | raw elapsed | ByteV2 elapsed |
| --- | ---: | ---: | ---: | ---: | ---: |
| p4096/b8/d128 | 222.119 | 134.664 | 60.63% | 4.610s | 7.604s |

fallback：

```text
any_exhausted=false
max_capacity=512
max_next_slot=80
total_capacity=16384
total_next_slot=2560
```

### 结论

P1 的功能目标已完成：`p4096/b8/d128` ByteV2 production/eager 都能跑通，
且 sparse fallback pool 没有耗尽。

性能目标仍未达成：b8 长 context 下 ByteV2 只有 raw 的约 60%。这说明当前
主要问题不再是 b8 slot mapping correctness，而是长 context / 大 batch 下
ByteV2 decode attention stage1、metadata/tile fallback 访问和 prefill/cache update
额外开销的组合。下一步应按长 context 路线继续做 production event timing：

```text
attention stage1
reduce
cache update
host/API
sampling/GEMM
```

优先确认 `p4096/b8` 的 40% 差距中有多少来自 stage1，有多少来自 b8 下
prefill generic fallback 路径代替 direct path。
