# ByteV2 decode vs FlashAttention kernel profile, 2026-06-23

## Scope

本报告只比较 decode attention kernel 相关指标，不比较 tokenizer、scheduler、
sampling、prefill 或端到端服务开销。

测试环境：

- GPU: NVIDIA A40, `CUDA_VISIBLE_DEVICES=4`
- head config: `num_heads=32`, `num_kv_heads=8`, `head_size=128`,
  `block_size=16`
- dtype: BF16 query/raw KV
- ByteV2 tile policy: `(16, 16, 16, 64, 128, 128)`
- FlashAttention baseline: vLLM FA2 paged decode
  `flash::flash_fwd_splitkv_kernel`
- ByteV2 当前重点路径：
    - generic split-k
    - GQA4 page-guarded / no-outlier lower-bound split-k

注意：GQA4 guarded microbench 使用 no-outlier 输入，是当前正确生产路径的
lower-bound。真实 E2E cache 中会有 unsafe/outlier page，kernel 时间会更高。

## Artifacts

microbench:

- `profiles/byte_v2_vs_flash_microbench_generic_1x_1k16k.jsonl`
- `profiles/byte_v2_vs_flash_microbench_gqa_guarded_1x_1k16k.jsonl`
- `profiles/byte_v2_vs_flash_microbench_generic_b4_4k16k.jsonl`
- `profiles/byte_v2_vs_flash_microbench_gqa_guarded_b4_4k16k.jsonl`

Nsight Compute:

- `profiles/byte_v2_ncu_generic_8192_p64_thread0sum_warp.ncu-rep`
- `profiles/byte_v2_ncu_generic_16384_p128_thread0sum_warp.ncu-rep`
- `profiles/byte_v2_ncu_gqa_guarded_8192_p128_thread0sum_warp.ncu-rep`
- `profiles/byte_v2_ncu_gqa_guarded_16384_p64_thread0sum_warp.ncu-rep`
- `profiles/byte_v2_ncu_gqa_guarded_16384_p64_fused_qreduce_warp.ncu-rep`
- `profiles/byte_v2_current_vs_raw_gqa_fused_qreduce_b1_1k16k.jsonl`
- `profiles/byte_v2_current_vs_raw_gqa_fused_qreduce_b4_4k16k.jsonl`
- `profiles/byte_v2_ncu_gqa_guarded_16384_p64_warp_softmax.ncu-rep`
- `profiles/byte_v2_current_vs_raw_gqa_warp_softmax_b1_1k16k.jsonl`
- `profiles/byte_v2_current_vs_raw_gqa_warp_softmax_b4_4k16k.jsonl`
- `profiles/byte_v2_ncu_gqa_guarded_16384_p64_warp_qk.ncu-rep`
- `profiles/byte_v2_current_vs_raw_gqa_warp_qk_b1_1k16k.jsonl`
- `profiles/byte_v2_current_vs_raw_gqa_warp_qk_b4_4k16k.jsonl`
- `profiles/byte_v2_ncu_gqa_guarded_16384_p64_warp_reduce.ncu-rep`
- `profiles/byte_v2_current_vs_raw_gqa_warp_reduce_b1_1k16k.jsonl`
- `profiles/byte_v2_current_vs_raw_gqa_warp_reduce_b4_4k16k.jsonl`
- `profiles/flash_fa2_paged_8192_warp.ncu-rep`
- `profiles/flash_fa2_paged_16384_warp.ncu-rep`

## 1. Kernel Time Sweep

### Batch 1, Generic ByteV2 vs FA2

| seq_len | ByteV2 best | ByteV2 median | FA2 median | gap | workspace |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1024 | p16 | 0.0973 ms | 0.0625 ms | 1.56x | 1.01 MiB |
| 2048 | p32 | 0.1864 ms | 0.0635 ms | 2.94x | 1.01 MiB |
| 4096 | p32 | 0.3308 ms | 0.0635 ms | 5.21x | 2.02 MiB |
| 8192 | p64 | 0.6062 ms | 0.0727 ms | 8.34x | 2.02 MiB |
| 16384 | p128 | 1.1658 ms | 0.1239 ms | 9.41x | 2.02 MiB |

结论：generic ByteV2 的时间基本随上下文线性增长；FA2 在 1k-4k 基本固定，
8k/16k 才明显增长。长上下文下 generic 差距稳定接近 9x。

### Batch 1, GQA4 Guarded Lower-bound vs FA2

| seq_len | ByteV2 best | ByteV2 median | FA2 median | gap | workspace |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1024 | p16 | 0.0358 ms | 0.0625 ms | 0.57x | 1.01 MiB |
| 2048 | p32 | 0.0584 ms | 0.0645 ms | 0.90x | 1.01 MiB |
| 4096 | p64 | 0.1024 ms | 0.0645 ms | 1.59x | 1.01 MiB |
| 8192 | p32 | 0.1772 ms | 0.0727 ms | 2.44x | 4.03 MiB |
| 16384 | p64 | 0.3226 ms | 0.1239 ms | 2.60x | 4.03 MiB |

结论：GQA packing 消除了大部分 repeated KV work，把 8k/16k 的差距从
8-9x 降到约 2.4-2.6x。但长上下文仍没有接近 FA2。

### Batch 4

| path | seq_len | ByteV2 best | ByteV2 median | FA2 median | gap |
| --- | ---: | --- | ---: | ---: | ---: |
| generic | 4096 | p64 | 1.1315 ms | 0.1239 ms | 9.13x |
| generic | 8192 | p64 | 2.0511 ms | 0.2232 ms | 9.19x |
| generic | 16384 | p128 | 3.8308 ms | 0.4209 ms | 9.10x |
| GQA4 guarded | 4096 | p32 | 0.3195 ms | 0.1249 ms | 2.56x |
| GQA4 guarded | 8192 | p64 | 0.6072 ms | 0.2232 ms | 2.72x |
| GQA4 guarded | 16384 | p128 | 1.1776 ms | 0.4219 ms | 2.79x |

结论：batch=4 后差距没有显著收敛，说明问题不是 batch=1 CTA 数不足，而是
ByteV2 每个 CTA 内部的 scan/decode/reduction 结构效率不足。

## 2. Nsight Compute Summary

下面使用 ncu replay 下的单 kernel 指标，绝对时间只用于辅助判断；性能结论以
microbench event 时间为准。

| kernel | case | ncu duration | DRAM GB/s | mem req %peak | issue active | active warps | eligible warps | regs | waves/SM |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ByteV2 generic | 8k p64 | 714 us | 64.9 | 58.7% | 58.3% | 9.51 | 1.13 | 44 | 4.88 |
| ByteV2 generic | 16k p128 | 1417 us | 64.2 | 58.7% | 58.0% | 9.50 | 1.13 | 44 | 4.88 |
| ByteV2 GQA4 | 8k p128 | 402 us | 63.5 | 60.2% | 38.1% | 5.97 | 0.53 | 40 | 0.51 |
| ByteV2 GQA4 main | 16k p64 | 372 us | 179.4 | 67.2% | 72.2% | 10.04 | 3.26 | 40 | 2.03 |
| ByteV2 GQA4 reduce | 16k p64 | 20 us | 206.5 | n/a | 26.3% | n/a | 0.51 | 24 | n/a |
| FA2 paged | 8k | 65.9 us | 514.0 | 77.9% | 11.2% | 0.98 | 0.11 | 252 | 1.52 |
| FA2 paged | 16k | 115.1 us | 585.5 | 88.5% | 10.7% | 1.00 | 0.11 | 252 | 1.81 |

Interpretation:

1. FA2 is a high-throughput streaming kernel. It uses far fewer active warps,
   much higher registers per thread, almost no L1/L2 hits, and reaches
   514-586 GB/s DRAM throughput on A40.
2. ByteV2 is not close to DRAM bandwidth saturation. Even generic reads only
   about 64 GB/s from DRAM, while the SM side is busy enough to show high
   issue active and many active warps.
3. ByteV2's high L1/L2 hit rate is not a win by itself. It means many accesses
   are cache-resident metadata/payload dependent accesses, but the kernel still
   cannot stream at high bandwidth because decode logic and dependency chains
   dominate.

## 3. Stall Breakdown

| kernel | case | top stalls |
| --- | --- | --- |
| ByteV2 generic | 8k p64 | long_scoreboard 5.95, barrier 3.36, wait 2.49, short_scoreboard 1.27 |
| ByteV2 generic | 16k p128 | long_scoreboard 6.01, barrier 3.38, wait 2.49, short_scoreboard 1.25 |
| ByteV2 GQA4 | 8k p128 | barrier 5.74, short_scoreboard 3.62, long_scoreboard 2.15, wait 2.00 |
| ByteV2 GQA4 main | 16k p64 | long_scoreboard 3.31, wait 1.84, short_scoreboard 0.59, barrier 0.07 |
| ByteV2 GQA4 reduce | 16k p64 | long_scoreboard 32.17, barrier 0.00 |
| FA2 paged | 8k | long_scoreboard 2.95, wait 1.98, math_pipe_throttle 1.65, barrier 0.41 |
| FA2 paged | 16k | long_scoreboard 3.23, wait 2.07, math_pipe_throttle 1.78, barrier 0.39 |

Interpretation:

1. Generic ByteV2 的第一瓶颈是 `long_scoreboard`。这对应 payload/page metadata
   地址依赖、fallback/outlier loader、block table/page pointer traversal。
2. GQA4 ByteV2 的第一瓶颈已经不再是 `barrier`。q_group fused reduction、
   warp softmax 和 warp QK 后，16k/p64 的 barrier stall 约为 0.07 cycles；
   当前主要看 `long_scoreboard`，更接近 payload decode / memory dependency。
3. FA2 也有 scoreboard，但 barrier 几乎不是问题；它的 mainloop 更接近
   vectorized load + tiled compute pipeline，而不是每 token 做 CTA-wide scalar
   reduction。

## 4. Main Gaps

### Gap A: KV Scan Structure

ByteV2 当前仍是：

```text
CTA = one sequence, one q_head or kv_head group, one partition
for token:
  each dim thread loads one K element
  CTA reduction -> one score
thread0 softmax over tile
for token:
  each dim thread loads one V element
  acc[dim] += prob * V
```

FA2 是 tiled splitkv mainloop，数据访问和 QK/PV 计算按 tile 组织，能高效 streaming
raw KV。ByteV2 的 token-by-token CTA-wide score reduction 天然 barrier 多、指令链长。

### Gap B: Payload Decode And Metadata Dependency

generic ByteV2 每个元素 load 都经过 codec/payload helper。即使 page-loop 已把
block table 查找从逐 token 降到逐 page，元素路径仍有：

- payload base/offset 计算
- fallback/outlier mask 检查
- outlier overlay 分支和 entry 查找
- K loop 和 V loop 之间缺少完整 descriptor 复用

ncu 的 generic `long_scoreboard` 说明这个方向优先级最高。

### Gap C: GQA CTA Barrier

GQA4 packed 已经减少重复 KV scan，但当前还是在一个 CTA 内串行处理 4 个 q group
的 reduction。GQA4 16k/p64 的 top stall 是 barrier 8.78 cycles，比 generic 更严重。

### Gap D: Split-k Workspace

ByteV2 split-k workspace 是 fp32 `tmp_out`：

```text
[num_seqs, num_heads, num_partitions, head_dim]
```

长上下文和 batch>1 时 workspace 很快增大。例如 batch=4、16k：

- generic p128: 8.06 MiB
- GQA4 p64: 16.13 MiB

目前主瓶颈仍在 main scan kernel，但 workspace 写/read reduce 会限制后续扩展。

## 5. Recommended Next Steps

### P0: Macro Descriptor / Paged KV Manager

目标是降低 generic `long_scoreboard`。

先在 kernel 内做 local descriptor，不急着引入持久 workspace：

```text
for macro page/tile:
  load block_table entry once
  compute physical page pointer once
  compute row valid mask once
  compute K payload base once
  compute V payload base once
  K loop and V loop share descriptor
```

具体改动：

1. 把 current split-k page-loop 再提升一级，显式构造 `PageDescriptor` /
   `TileDescriptor` struct。
2. descriptor 中缓存 page pointer、row range、K/V payload bases、page unsafe bit。
3. 让 K path 和 V path 共享 descriptor，减少重复地址计算和 page metadata load。
4. 对 generic unsafe loader 增加 no-fallback raw branch 的 fast inline path，
   outlier overlay 只在 mask 命中后进入慢 helper。

验收：

- generic 8k/p64 和 16k/p128 microbench 至少下降 10%。
- ncu generic top stall 中 `long_scoreboard` 明显下降，或 DRAM throughput 明显上升。

2026-06-23 已完成第一版 kernel-local payload tile descriptor：

- 只保留在 generic split-k kernel。
- 每个 physical page + 当前 dim 构造一次 descriptor，缓存 payload tile offset、
  base byte、fallback/outlier bit、outlier count 和 outlier payload offset。
- row loop 内不再重复读取 base/mask 或重复计算 outlier/payload tile offset。
- 不引入新的 CTA barrier，也不引入持久 descriptor workspace。
- GQA4 kernel 曾尝试同样 descriptor，但 batch=4 16k/p64 从约 1.758 ms 回退到
  约 1.799 ms，因此已回退；GQA4 保持原 loader。

结果：

| path | case | before | after | delta |
| --- | --- | ---: | ---: | ---: |
| generic | batch=1, 8192, p64 | 0.6062 ms | 0.5478 ms | -9.6% |
| generic | batch=1, 16384, p128 | 1.1658 ms | 1.0414 ms | -10.7% |
| generic | batch=4, 8192, p64 | 2.0511 ms | 1.8258 ms | -11.0% |
| generic | batch=4, 16384, best | 3.8308 ms | 3.3843 ms | -11.7% |
| GQA4 guarded | batch=4, 16384, p64 | 1.7582 ms | 1.7705 ms | ~flat |

generic 16k/p128 ncu:

| metric | before | after |
| --- | ---: | ---: |
| duration | 1416.6 us | 1248.3 us |
| DRAM throughput | 64.2 GB/s | 72.8 GB/s |
| registers/thread | 44 | 48 |
| issue active | 58.0% | 51.7% |
| eligible warps/scheduler | 1.13 | 0.86 |
| top stall | long_scoreboard 6.0 | long_scoreboard 7.5 |

解释：descriptor 减少了重复 metadata/base/mask work，总 duration 明显下降，
DRAM throughput 上升；但剩余指令里 memory dependency 占比更高，所以
`long_scoreboard` 的 per-issued-instruction stall 反而上升。这说明方向有效，但
下一步要继续减少真正的数据依赖，而不是只减少外围指令。

2026-06-23 继续尝试了 tile 内 shared page pointer cache：

- QK 阶段把当前 compute tile 的 page pointer 和 page unsafe flag 写入 shared
  memory，PV 阶段复用，避免再次读取 block table 和 page unsafe flag。
- 不增加新的 CTA barrier，利用 QK 和 softmax 后已有的 `__syncthreads()`。
- 实测收益未超过 5%，按保留门槛已回退，不保留代码。

microbench，GPU4，bn64，unguarded generic：

| case | descriptor-only | page cache experiment | delta |
| --- | ---: | ---: | ---: |
| batch=1, 8192, p64 | 0.5478 ms | 0.5356 ms | -2.2% |
| batch=1, 16384, p128 | 1.0414 ms | 1.0158 ms | -2.5% |
| batch=4, 8192, p64 | 1.8258 ms | 1.7777 ms | -2.6% |
| batch=4, 16384, p64 | 3.3843 ms | 3.2819 ms | -3.0% |
| batch=4, 16384, p128 | 3.3987 ms | 3.2824 ms | -3.4% |

结论：单纯复用 page pointer / page flag 不是当前主要瓶颈。后续如果继续做
macro descriptor，需要把 payload/base/mask 或 K/V decode work 本身合并复用，
而不是只减少 block table 外围读取。

### P1: GQA4 CTA Structure

目标是降低 GQA4 `barrier` / `short_scoreboard`。

已完成第一步：GQA4 QK score path 把 4 个 q_group 的 block-wide sum 合并为一次
CTA reduction，一次写入 4 路 shared scores。它不是完整 warp-per-q 重写，但先把最热
row loop 中的 q_group 串行 reduction barrier 打掉。

microbench，GPU4，bn64，guarded no-outlier lower-bound：

| case | previous | fused qreduce | delta |
| --- | ---: | ---: | ---: |
| batch=1, 8192 | 0.3205 ms | 0.2775 ms | -13.4% |
| batch=1, 16384 | 0.5908 ms | 0.4946 ms | -16.3% |
| batch=4, 8192 | 0.9370 ms | 0.8069 ms | -13.9% |
| batch=4, 16384 | 1.7582 ms | 1.5698 ms | -10.7% |

16k/p64 ncu 显示 duration 从约 662 us 降到约 540 us，DRAM throughput 从
81.9 GB/s 升到 100.4 GB/s，eligible warps 从 0.89 升到 1.07；
`short_scoreboard` 从 5.07 降到 1.77。`barrier` 的 per-issued-inst 统计仍约
9.15 cycles，说明剩余同步在总指令减少后更突出，不能只看 stall ratio 判断失败。

已完成第二步：tile softmax 从 thread0 串行改成 warp-per-q_group。每个 q_group
由一个 warp 负责 tile max、prob 写回和 tile sum，PV 仍保留当前一次 V decode 复用
4 个 q_group 的结构。

microbench，GPU4，bn64，guarded no-outlier lower-bound：

| case | fused qreduce | warp softmax | delta |
| --- | ---: | ---: | ---: |
| batch=1, 8192 | 0.2775 ms | 0.2509 ms | -9.6% |
| batch=1, 16384 | 0.4946 ms | 0.4495 ms | -9.1% |
| batch=4, 8192 | 0.8069 ms | 0.7516 ms | -6.9% |
| batch=4, 16384 | 1.5698 ms | 1.4469 ms | -7.8% |

16k/p64 ncu 继续改善：duration 从约 540 us 降到约 493 us，DRAM throughput 从
100.4 GB/s 升到 136.1 GB/s，eligible warps 从 1.07 升到 1.27，`barrier`
从 9.15 降到 7.06，`short_scoreboard` 从 1.77 降到 1.22。这个改动超过 5%
保留线，已保留。

已完成第三步：QK row-level reduction 从 CTA-wide reduction 改为 warp-per-q_group。
每个 warp 负责一个 q_group，lane 预加载该 q_group 的 4 个 query dim，并在每个 row
加载对应 4 个 K dim 做 warp 内 dot/reduce。这样每个 row 不再需要跨 warp shared
reduction 和 CTA barrier；代价是 K decode 按 q_group 重复，但实测收益显著。

microbench，GPU4，bn64，guarded no-outlier lower-bound：

| case | warp softmax | warp QK | delta |
| --- | ---: | ---: | ---: |
| batch=1, 8192 | 0.2509 ms | 0.1976 ms | -21.2% |
| batch=1, 16384 | 0.4495 ms | 0.3748 ms | -16.6% |
| batch=4, 8192 | 0.7516 ms | 0.6216 ms | -17.3% |
| batch=4, 16384 | 1.4469 ms | 1.1837 ms | -18.2% |

16k/p64 ncu：duration 从约 493 us 降到约 372 us，eligible warps 从 1.27 升到
3.26，`barrier` 从 7.06 降到 0.07，`short_scoreboard` 从 1.22 降到 0.59。
这说明当前 GQA4 的 barrier 阶段基本被打掉，下一步应重新看 payload decode /
memory dependency，而不是继续围绕 `__syncthreads()` 做小改。

已完成第四步：GQA4 split-k reduce 从 one-CTA-per-head 改为 warp-per-dim reduce。
旧 reduce 在 batch=1 只有 32 个 CTA，16k/p64 ncu duration 约 106 us，issue active
约 2%。新 reduce 每个 warp 负责一个 output dim，增加 CTA 并行度，放弃 shared
weights staging，直接在 warp 内做 LSE max/sum 和 weighted tmp_out reduction。

microbench，GPU4，bn64，guarded no-outlier lower-bound：

| case | warp QK | warp reduce | delta |
| --- | ---: | ---: | ---: |
| batch=1, 8192 | 0.1976 ms | 0.1772 ms | -10.4% |
| batch=1, 16384 | 0.3748 ms | 0.3226 ms | -13.9% |
| batch=4, 8192 | 0.6216 ms | 0.6072 ms | -2.3% |
| batch=4, 16384 | 1.1837 ms | 1.1776 ms | -0.5% |

16k/p64 reduce ncu：duration 从约 106 us 降到约 20 us，issue active 从约 2% 升到
26%，eligible warps 从 0.02 升到 0.51。该改动对 batch=1 明显，对 batch=4 只有小幅
改善但没有回退，因此保留在 GQA4 路径中。

已完成第五步：GQA4 PV safe-page descriptor。PV 阶段按 page 分 safe/unsafe path，
safe page 使用 tile descriptor 复用 V payload/base 地址计算，unsafe page 仍走通用
loader 保持 page flags fallback 语义。

microbench，GPU4，bn64，guarded no-outlier lower-bound：

| case | warp reduce | PV descriptor | delta |
| --- | ---: | ---: | ---: |
| batch=1, 8192 | 0.1772 ms | 0.1638 ms | -7.5% |
| batch=1, 16384 | 0.3226 ms | 0.3062 ms | -5.1% |
| batch=4, 8192 | 0.6072 ms | 0.5765 ms | -5.1% |
| batch=4, 16384 | 1.1776 ms | 1.1274 ms | -4.3% |

16k/p64 main kernel ncu：duration 从约 372 us 降到约 354 us，`long_scoreboard`
从 3.31 降到 2.84，DRAM throughput 从 179.4 GB/s 升到 186.8 GB/s。代价是
registers/thread 从 40 增到 48，eligible warps 从 3.26 降到 2.92。

下一步需要围绕 payload 和寄存器压力做结构性改变：

1. 已尝试 QK path 的 per-thread lightweight descriptor，正确但几何平均只有
   1.024x，低于 5% 保留线，已回退。不要继续做同形态的 `payload_offset/base`
   小缓存。
2. 若继续优化 QK，应改为更结构化的 macro descriptor / tile staging，减少寄存器
   压力和 per-thread 私有元数据。
3. 评估 PV tile staging/vectorized decode：先验证 V decode/PV 是否能把 scalar
   load/decode 变成更连续的 per-page/per-tile work。
4. 对新增 descriptor 监控 registers/thread；如果继续升高，需要把 descriptor
   字段压缩或分阶段计算，避免 occupancy 继续下降。
5. 已尝试把 GQA4 split-k `tmp_out` intermediate 改为 BF16 bits 读写，正确但
   几何平均只有 1.001x，低于 5% 保留线，已回退。不要继续只压缩 `tmp_out`
   element size；若做 workspace，需要减少 partition 写入次数或 fuse main+reduce
   的特化路径。

验收：

- microbench GQA4 8k/16k 再下降 10% 以上：已达成。
- GQA4 16k/p64 total duration 降低、eligible warps 提升、short_scoreboard 下降：
  barrier 轮次已达成；PV descriptor 轮次 duration 继续下降，但 eligible warps
  因寄存器增加而下降，需要后续控制。
- GQA4 16k/p64 barrier stall 从 8.8 cycles 降到 6 以下：已达成，当前约 0.07。

### P2: FA-style Split Heuristic

当前 partition heuristic 已经避免明显坏点，但仍是 partition-size 驱动，不是
FA-style `num_splits` 驱动。

下一步可以把 heuristic 改成：

```text
num_splits = f(num_seqs, num_heads or num_kv_heads, seq_len, sm_count)
partition_size = ceil_div(seq_len, num_splits)
partition_size = align_up(partition_size, block_size)
```

不过 ncu 显示这不是第一瓶颈，应排在 descriptor 和 GQA CTA 结构之后。

### P3: Reduce Workspace Pressure

如果 P0/P1 后 main kernel 接近 FA2，再做：

- GQA packed workspace 以 kv_head/q_group 组织，减少 tmp_out 写放大。
- fuse split-k reduce for small `num_partitions` 或对 batch=1 做 specialized reduce。
- 对 LSE/tmp_out 做更紧凑 layout，减少 reduce kernel HBM 读写。

## 6. Current Bottom Line

1. generic ByteV2 当前长上下文 decode kernel 约比 FA2 慢 8-9x。
2. GQA4 guarded lower-bound 把差距缩到约 2.25-2.67x，说明 repeated KV work、
   CTA barrier 和一部分 PV payload dependency 已经被解决。
3. 剩余差距主要不是单纯带宽，而是 ByteV2 主 kernel 的 payload decode、metadata
   address dependency、寄存器压力和 split-k workspace 结构。
4. 下一轮 GQA4 最值得做的是 macro descriptor / tile staging、PV vectorized decode
   或减少 split-k partition 写入次数；generic 路径继续做 macro descriptor /
   paged KV manager。不要继续只调 partition size、做同形态 QK 小 descriptor、
   压缩 `tmp_out` element size，或只做零散 reduction/barrier 微优化。
