# ByteV2 decode kernel vs FlashAttention kernel 对比和优化计划

## 1. 当前结论

当前 ByteV2 decode kernel 和 FlashAttention paged decode 的差距，主要不在
split-k reduce 的数学形式上。ByteV2 已经改成 partial output + LSE 的合并方式，
和 FlashAttention 的 split combine 方向一致。

真正可能拉开差距的是主 kernel：

1. ByteV2 按 `q_head` 独立扫描同一份 KV，GQA 场景下会对同一个 `kv_head`
   重复读取、重复解压 K/V。当前模型配置是 `num_heads=32`、`num_kv_heads=8`，
   `q_per_kv=4`，这里天然有最高接近 4x 的重复 KV 工作。
2. ByteV2 每个 `(seq, q_head, partition)` CTA 用 128 个线程对应 128 个 head dim，
   逐 token 做一次 block reduction 得到 score，再由 thread 0 串行做 tile 内
   softmax。FlashAttention 是 tile/MMA/向量化 copy/pipeline 的整体设计。
3. ByteV2 的 payload decode 是逐元素 helper：每个 K/V 元素都会重新计算 tile
   offset、读 fallback/outlier mask、读 base/code/low byte，且 outlier/fallback
   通用逻辑在 no-outlier 热路径也存在。
4. ByteV2 split-k 现在按固定 `partition_size` 切分；FlashAttention 是按 SM
   occupancy、KV block 数、batch/head tile 数选择 `num_splits`，尽量避免过多
   partial output HBM 读写。
5. ByteV2 split-k workspace 的 `tmp_out` 是
   `[num_tokens, num_heads, num_partitions, head_dim]` fp32。4096 长度、p32 时约
   2 MiB；LSE 单统计量优化后性能几乎没变，说明瓶颈更偏向主 scan/decode 和
   repeated KV work，而不是 max/sum 统计量本身。

基于现有 profile，4096 decode-only microbench 中：

| kernel | median |
| --- | ---: |
| ByteV2 split p32/bn64 | 0.3799 ms |
| vLLM FlashAttention paged FA2 | 0.0666 ms |

差距约 5.7x。E2E 4096 cached profile 中，ByteV2 split-k CUDA 总时间约
558.9 ms / 1024 calls，平均约 545.8 us/call。prefill 已经不是当前最主要矛盾。

### 1.1 2026-06-20 已执行结果

已经完成 Phase B 和 Phase C 的最小闭环：

- `BYTE_V2_DECODE_ASSUME_NO_OUTLIER=1`：接入 no-fallback/no-outlier payload
  fast loader，默认关闭。
- `BYTE_V2_DECODE_GQA_PACKED=1`：接入 `gqa4_h128` split-k main kernel，
  默认关闭，且 Python 层要求同时开启
  `BYTE_V2_DECODE_ASSUME_NO_OUTLIER=1`。
- `BYTE_V2_DECODE_GQA_PACKED_MIN_SEQ_LEN`：默认 2048，避免短上下文启用
  GQA-packed 后退化。
- `BYTE_V2_DECODE_GQA_PACKED_PARTITION_SIZE`：显式 override；未设置时走
  batch/seq_len auto heuristic。

4096 decode-only microbench 结果：

| kernel | median |
| --- | ---: |
| 原始 ByteV2 split p32/bn64，普通合成输入 | 0.3799 ms |
| no-outlier 输入，通用 loader | 0.3226 ms |
| no-outlier fast loader，p32/bn64 | 0.2611 ms |
| GQA4 packed + no-outlier fast loader，p32/bn64 | 0.2079 ms |
| GQA4 packed + no-outlier fast loader，p64/bn64 | 0.1925 ms |

长上下文 sweep，p32/bn64：

| seq_len | no-outlier fast loader | GQA4 packed no-outlier |
| ---: | ---: | ---: |
| 256 | 0.0379 ms | 0.0655 ms |
| 512 | 0.0481 ms | 0.0696 ms |
| 1024 | 0.0829 ms | 0.0788 ms |
| 2048 | 0.1423 ms | 0.1065 ms |
| 4096 | 0.2621 ms | 0.2058 ms |

4096 GQA4 partition sweep：

| partition_size | median |
| ---: | ---: |
| 16 | 0.2212 ms |
| 32 | 0.2099 ms |
| 64 | 0.1925 ms |
| 128 | 0.2949 ms |
| 256 | 0.5335 ms |

结论：

1. no-outlier fast loader 在相同 no-outlier 输入上把 4096 median 从
   0.3226 ms 降到 0.2611 ms，约 19%。
2. GQA4 packed 在 4096 p32 上进一步降到 0.2079 ms；p64 最好，为
   0.1925 ms。
3. 相比原始普通合成输入的 0.3799 ms，当前 lower-bound 路径约快 49%。
4. 距离 vLLM FlashAttention paged FA2 的 0.0666 ms 仍约 2.9x，说明只消除
   通用 payload 分支和重复 GQA KV 解压还不够。
5. GQA-packed 对短上下文有固定开销，256/512 明显更慢；因此接入层必须做
   长上下文门控。

注意：no-outlier/GQA-packed 路径是假设页内没有 fallback/outlier 的 lower-bound
路径。如果实际 cache 内存在 fallback/outlier tile，输出会不正确。生产默认仍应
关闭，除非先证明当前数据路径满足这个假设，或实现安全 fallback/分派。

### 1.2 2026-06-20 安全门控进展

已经增加一个显式验证式安全门控：

- 新增 CUDA cache metadata stats op：
  `byte_v2_collect_cache_stats(stats, kv_cache, block_tables, seq_lens, ...)`。
- stats 输出 4 个 int32：
  `[unsafe, fallback_tiles, outlier_tiles, referenced_pages]`。
- 新增环境变量 `BYTE_V2_DECODE_VALIDATE_NO_OUTLIER=1`。
- 当同时开启
  `BYTE_V2_DECODE_ASSUME_NO_OUTLIER=1` 和
  `BYTE_V2_DECODE_VALIDATE_NO_OUTLIER=1` 时，decode 前会扫描当前
  `block_tables/seq_lens` 引用的 page metadata。
- 如果发现任意 fallback/outlier mask，Python 层会清掉 no-outlier 和
  GQA-packed tile policy flags，自动回退到通用 decode。
- 如果 `kv_cache/block_table/seq_lens` 任何一个不是 CUDA tensor，也会保守回退。

这个门控当前会通过 `stats[0].item()` 做一次 host 同步，因此它是正确性验证和
调试用闭环，不是最终性能方案。下一步应把 unsafe 状态维护在 cache manager 或
page/block metadata 状态里，避免 decode 每步扫描和同步。

已验证：

```bash
TRITON_KERNELS_SRC_DIR=/mnt/sdb/yxz/ByteV2/vllm/.deps/triton_kernels-src/python/triton_kernels/triton_kernels \
VLLM_CUTLASS_SRC_DIR=/mnt/sdb/yxz/ByteV2/vllm/.deps/cutlass-src \
uv pip install -e . --torch-backend=auto

.venv/bin/python -m pytest tests/v1/attention/test_byte_v2_layout.py -q \
  -k "decode_validate_no_outlier_defaults_off or decode_validation_disables_unsafe_fast_path or gqa_packed_uses_long_context_partition or gqa_packed_requires_no_outlier"

CUDA_VISIBLE_DEVICES=4 .venv/bin/python -m pytest tests/v1/attention/test_byte_v2_layout.py -q \
  -k "collect_cache_stats_cuda_detects_masks"

CUDA_VISIBLE_DEVICES=4 .venv/bin/python -m pytest tests/v1/attention/test_byte_v2_layout.py -q \
  -k "split_k_gqa_packed_cuda_matches_raw_reference or split_k_cuda_matches_raw_reference"

.venv/bin/pre-commit run --files \
  csrc/libtorch_stable/byte_v2/byte_v2_ops.cu \
  csrc/libtorch_stable/ops.h \
  csrc/libtorch_stable/torch_bindings.cpp \
  vllm/v1/attention/backends/byte_v2_attn.py \
  vllm/v1/attention/backends/byte_v2_ops.py \
  tests/v1/attention/test_byte_v2_layout.py \
  scripts/byte_v2_decode_microbench.py
```

### 1.3 2026-06-21 低开销 unsafe 状态维护

已经完成第一版低开销 page unsafe flags：

- 新增 cache op：
  `byte_v2_update_cache_unsafe_flags(page_unsafe_flags, kv_cache, slot_mapping, ...)`。
  cache update / raw staging commit / single-token update 后，只扫描本次
  `slot_mapping` 触达的 physical pages，并在 GPU tensor 中维护每页 unsafe bit。
- 新增 decode op：
  `byte_v2_paged_decode_attention_split_k_guarded(...)`。
  split-k main kernel 在每个 physical page 上读取 `page_unsafe_flags`：
  safe page 走 no-fallback/no-outlier fast loader，unsafe page 走通用 payload
  loader。GQA4 packed 分支保留，只在 loader 层做 page guard。
- Python backend 新增 `BYTE_V2_DECODE_PAGE_UNSAFE_FLAGS`，默认开启，但只在
  `BYTE_V2_DECODE_ASSUME_NO_OUTLIER=1` 时生效。
- 如果没有可用 flags，且没有开启显式
  `BYTE_V2_DECODE_VALIDATE_NO_OUTLIER=1`，backend 会保守清掉 no-outlier/GQA
  fast-path flags，回到通用 decode。
- 显式验证门控仍保留，用于调试和 correctness audit；page flags 路径避免了
  每步 stats kernel + `.item()` host sync。

限制：

- guarded path 能安全处理 outlier overlay page；如果 metadata 里出现 fallback
  mask 且当前 V4 page 没有 raw payload，通用 loader 也无法恢复 raw bf16 值。
  现阶段默认布局要求 outlier overlay 容量覆盖 full tile，正常路径不应产生 raw
  fallback。
- no-split decode 还没有 guarded 版本；实际长上下文路径使用 split-k，短上下文
  默认不启用 GQA-packed。

4096 decode-only microbench，GQA4 p64/bn64，safe page：

| kernel | median |
| --- | ---: |
| GQA4 packed + no-outlier fast loader，无 guard | 0.1915 ms |
| GQA4 packed + no-outlier fast loader，page-guarded | 0.1961 ms |

page flag guard 在 safe-page lower-bound 上增加约 2.4% kernel 时间，换来每步
decode 无 host sync 的安全分派能力。

4096 prompt / 32 decode tokens E2E cached profile，GPU4，enforce eager：

| path | cached wall | custom CUDA total | split-k CUDA avg | output |
| --- | ---: | ---: | ---: | --- |
| generic split-k，正确基线 | 1348.4 ms | 567.5 ms | 545.6 us | matches guarded |
| GQA4 no-outlier，unguarded lower-bound | 1037.9 ms | 259.7 ms | 245.1 us | mismatches generic |
| GQA4 no-outlier，page-guarded | 1121.8 ms | 340.7 ms | 319.6 us | matches generic |

结论：

1. page-guarded 路径没有出现 `byte_v2_collect_cache_stats`，steady-state 已经不再
   经过 validation stats kernel + `.item()` host sync。
2. 真实 E2E cache 中确实存在 unsafe/outlier page；unguarded no-outlier
   lower-bound token 输出和 generic/guarded 不一致，不能作为正确路径。
3. guarded GQA4 在正确输出下把 cached split-k 平均从 545.6 us 降到 319.6 us，
   约快 41%；custom CUDA total 从 567.5 ms 降到 340.7 ms。
4. `byte_v2_update_cache_unsafe_flags` 在 cached profile 中 1024 次总计约 4.55 ms，
   平均 4.44 us/call，当前不是主要瓶颈。

已验证：

```bash
TRITON_KERNELS_SRC_DIR=/mnt/sdb/yxz/ByteV2/vllm/.deps/triton_kernels-src/python/triton_kernels/triton_kernels \
VLLM_CUTLASS_SRC_DIR=/mnt/sdb/yxz/ByteV2/vllm/.deps/cutlass-src \
uv pip install -e . --torch-backend=auto

.venv/bin/python -m pytest tests/v1/attention/test_byte_v2_layout.py -v \
  -k "page_unsafe_flags or update_cache_unsafe_flags or split_k_guarded or validation_disables_unsafe_fast_path"

.venv/bin/python -m pytest tests/v1/attention/test_byte_v2_layout.py -v \
  -k "decode_partition_size or gqa_packed or ops_fail_cleanly_without_registered_kernels or collect_cache_stats_cuda_detects_masks or paged_decode_attention_split_k_cuda_matches_raw_reference or paged_decode_attention_split_k_gqa_packed_cuda_matches_raw_reference"

.venv/bin/pre-commit run --files \
  csrc/libtorch_stable/byte_v2/byte_v2_ops.cu \
  csrc/libtorch_stable/ops.h \
  csrc/libtorch_stable/torch_bindings.cpp \
  vllm/v1/attention/backends/byte_v2_attn.py \
  vllm/v1/attention/backends/byte_v2_ops.py \
  tests/v1/attention/test_byte_v2_layout.py \
  scripts/byte_v2_decode_microbench.py
```

### 1.4 2026-06-21 tile-level guard 负结果

尝试过在 unsafe page 内进一步按 codec tile 判断是否真的含
fallback/outlier mask：只有当前 tile unsafe 时才走通用 payload loader，否则继续走
no-outlier fast loader。正确性用例通过，但 E2E cached profile 退化：

| path | cached wall | custom CUDA total | guarded split-k avg | output |
| --- | ---: | ---: | ---: | --- |
| page-level guard | 1121.8 ms | 340.7 ms | 319.6 us | matches generic |
| tile-level guard inside unsafe page | 1148.6 ms | 363.4 ms | 341.8 us | matches generic |

已回退到 page-level guard。主要原因是 tile-level guard 在元素加载热路径里增加了
mask offset 和 mask bit 检查；在真实 unsafe page 上，这个额外判断的成本超过了
少量 safe tile 继续使用 fast loader 的收益。下一步不应继续沿着“每元素更细粒度
guard”优化，而应优先减少 unsafe/outlier page 产生，或优化通用 outlier overlay
loader 本身。

### 1.5 2026-06-21 warp-broadcast mask 读取负结果

尝试过把通用 payload loader 中每个元素重复读取的 fallback/outlier 32-bit mask
改成 warp 内 lane0 读取后 `__shfl_sync` 广播。正确性用例通过，但 4096 generic
split-k microbench 没有收益：

| path | partition | median |
| --- | ---: | ---: |
| 原始 generic split-k 基线 | p32 | 0.3799 ms |
| warp-broadcast mask generic split-k | p32 | 0.3820 ms |
| warp-broadcast mask generic split-k | p64 | 0.4086 ms |

已回退。结论是当前通用 unsafe/outlier loader 的主要成本不在同一 warp 内重复读取
两组 32-bit mask；`__shfl_sync` 和额外 helper 拆分没有带来可见收益。下一步应
聚焦 outlier overlay 命中后的 entry 查找/加载、unsafe page 产生比例，以及
page/macro descriptor 级别复用，而不是只优化 mask load。

### 1.6 2026-06-21 split-k page-loop 复用

已把 split-k main kernel 的 K/V 扫描从逐 token 查 `block_table` / page flag 改为
按 logical block/page 分组：

- 每个 `ComputeBlockN` tile 先枚举覆盖的 logical blocks。
- 每个 physical page 只读取一次 `block_table` entry、page pointer 和
  `page_unsafe_flags`。
- 页内再遍历 `AllocBlockTokens=16` 个 row，softmax/reduce 数学保持不变。
- generic split-k 和 GQA4 packed split-k 都已接入；no-split decode 暂未改。

4096 decode microbench，GPU4，bn64：

| path | partition | median |
| --- | ---: | ---: |
| 原始 generic split-k 基线 | p32 | 0.3799 ms |
| page-loop generic split-k | p32 | 0.3400 ms |
| page-loop generic split-k | p64 | 0.3523 ms |
| 原始 GQA4 page-guarded safe-page | p64 | 0.1961 ms |
| page-loop GQA4 page-guarded safe-page | p64 | 0.1823 ms |

结论：

1. generic unsafe 输入 p32 约快 10.5%，说明 page/block metadata 重复计算确实是
   主 scan 的可见成本。
2. GQA4 page-guarded safe-page lower bound 从 0.1961 ms 降到 0.1823 ms，约快
   7.0%。
3. 这是 Phase E macro descriptor / paged KV manager 的最小闭环：先在 kernel
   内把 page metadata 复用起来，还没有引入独立 descriptor tensor。

4096 prompt / 32 decode tokens E2E cached profile，GPU4，enforce eager，
GQA4 + page unsafe flags：

| path | cached wall | custom CUDA total | guarded split-k avg | output |
| --- | ---: | ---: | ---: | --- |
| page-level guard baseline | 1121.8 ms | 340.7 ms | 319.6 us | reference |
| page-loop page-level guard | 1102.0 ms | 321.2 ms | 300.8 us | matches baseline |

E2E 正确路径的 guarded split-k 平均再降约 5.9%，custom CUDA total 降约 19.5 ms。
`byte_v2_update_cache_unsafe_flags` 仍约 4.3 us/call，不是主要瓶颈。

### 1.7 2026-06-21 split/partition heuristic

page-loop 后重新跑了 partition sweep，不能继续完全沿用旧 kernel 的经验点。
下面只采用串行运行的 sweep 结果；并行跑多个 microbench 会引入 GPU contention，
不作为 heuristic 依据。

generic split-k，unsafe 输入，batch=1，bn64：

| seq_len | best partition | median |
| ---: | ---: | ---: |
| 512 | p16 / p32 | 0.0543 ms |
| 1024 | p16 | 0.1004 ms |
| 2048 | p16 | 0.1915 ms |
| 4096 | p32 | 0.3379 ms |
| 8192 | p64 | 0.6164 ms |
| 16384 | p128 | 1.1889 ms |

GQA4 packed + page-guarded，safe-page lower bound，batch=1，bn64：

| seq_len | best partition | median |
| ---: | ---: | ---: |
| 1024 | p16 | 0.0584 ms |
| 2048 | p32 | 0.1019 ms |
| 4096 | p64 | 0.1802 ms |
| 8192 | p128 | 0.3441 ms |
| 16384 | p64 | 0.6175 ms |

batch>1 时，decode 本身已经提供更多 CTA，最佳 partition 会改变：

| path | batch | seq_len | best partition | median |
| --- | ---: | ---: | ---: | ---: |
| generic split-k | 2 | 4096 | p64 | 0.5990 ms |
| generic split-k | 2 | 8192 | p64 | 1.1663 ms |
| generic split-k | 2 | 16384 | p64 | 2.1258 ms |
| generic split-k | 4 | 4096 | p64 | 1.1530 ms |
| generic split-k | 4 | 8192 | p64 | 2.0961 ms |
| generic split-k | 4 | 16384 | p128 | 3.8881 ms |
| GQA4 page-guarded | 2 | 4096 | p32 | 0.3144 ms |
| GQA4 page-guarded | 2 | 8192 | p32 | 0.5755 ms |
| GQA4 page-guarded | 2 | 16384 | p64 | 1.0435 ms |
| GQA4 page-guarded | 4 | 4096 | p32 | 0.5325 ms |
| GQA4 page-guarded | 4 | 8192 | p64 | 1.0004 ms |
| GQA4 page-guarded | 4 | 16384 | p64 | 1.9108 ms |

已接入默认 auto heuristic：

- generic split-k batch=1：`seq_len < 4096` 用 p16，4096 用 p32，
  8192 用 p64，16384+ 用 p128。
- generic split-k batch=2/3：`seq_len >= 4096` 用 p64，更短上下文保持 p16。
- generic split-k batch>=4：4096/8192 用 p64，16384+ 用 p128，更短上下文保持 p16。
- GQA4 packed batch=1：`seq_len < 2048` 用 p16，2048 用 p32，4096 用 p64，
  8192 用 p128，16384+ 回到 p64。这个档位是测量表驱动，不是单调递增规则。
- GQA4 packed batch=2/3：2048/4096/8192 用 p32，16384+ 用 p64。
- GQA4 packed batch>=4：2048/4096 用 p32，8192+ 用 p64。
- `BYTE_V2_DECODE_SPLIT_K_PARTITION_SIZE`、
  `BYTE_V2_DECODE_SPLIT_K_LONG_PARTITION_SIZE` 和
  `BYTE_V2_DECODE_GQA_PACKED_PARTITION_SIZE` 仍作为显式 override；只要用户设置，
  backend 就不使用对应 auto 档位。

代码上 `_decode_partition_size` 现在接收 `num_decode_tokens`，并由
`_run_paged_decode` 用 `output.shape[0]` 传入。workspace 大小和 split-k enable
判断都使用同一个 batch-aware partition，避免 partition 和临时 buffer 不一致。

8192/16384 E2E cached profile 也已跑通。GQA pageflags 和 generic 输出 token IDs
完全一致：

| prompt_len | path | auto partition | cached decode avg | cached wall | token IDs |
| ---: | --- | ---: | ---: | ---: | --- |
| 8192 | generic split-k | p64 | 893.0 us | 1701.3 ms | baseline |
| 8192 | GQA4 page-guarded | p128 | 573.9 us | 1383.8 ms | match |
| 16384 | generic split-k | p128 | 1743.8 us | 2575.7 ms | baseline |
| 16384 | GQA4 page-guarded | p64 | 1056.7 us | 1879.6 ms | match |

相对 generic split-k，GQA pageflags 在 cached decode kernel 上约快 1.56x
（8192）和 1.65x（16384）。

### 1.8 2026-06-23 Nsight Compute 和 block reduction 优化

对 8192/16384 的主 decode kernel 做了 Nsight Compute。结论是当前瓶颈仍在主
scan/decode kernel，不在 split-k reduce kernel：

完整 kernel-level 对比见：
`/mnt/sdb/yxz/ByteV2/Doc/ByteV2_decode_vs_flashattention_profile_20260623.md`。

| path | seq_len / partition | duration | issue | active warps | eligible warps | L1 hit | L2 hit | top stalls |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| generic | 8192 / p64 | 730.9 us | 55.7% | 38.14 | 1.06 | 76.0% | 69.2% | long_scoreboard 5.98, barrier 3.58 |
| generic | 16384 / p128 | 1439.6 us | 55.8% | 38.07 | 1.07 | 79.0% | 59.5% | long_scoreboard 6.02, barrier 3.60 |
| GQA4 guarded | 8192 / p128 | 427.8 us | 33.3% | 23.82 | 0.49 | 69.2% | 7.8% | barrier 6.29, short_scoreboard 3.91 |
| GQA4 guarded | 16384 / p64 | 697.4 us | 41.2% | 41.76 | 0.82 | 67.8% | 12.7% | barrier 9.90, short_scoreboard 5.17 |

解读：

1. generic 不是 DRAM bandwidth bound。DRAM 占用只有约 9% 到 11%，但
   L1/SM 和 `long_scoreboard` 明显，说明通用 payload/page metadata 的地址依赖、
   fallback/outlier loader、block table/page pointer 仍是主要方向。
2. GQA4 guarded 的主要 stall 是 CTA barrier。代码上每个 QK score 都调用一次
   `byte_v2_block_sum`，而原实现有 3 次 `__syncthreads()`；GQA4 每个 row 会做
   4 个 q-head reduction，因此 barrier stall 被放大。

已完成一个小优化：新增 `byte_v2_block_sum_thread0`，只保证 thread 0 拿到 block
sum，减少一次 CTA barrier；decode 主 kernel 的 QK reduction 改用该 helper。
split-k reduce kernel 没改，因为那里所有 thread 都需要 reduce 后的值。

验证结果：

```bash
PATH="$PWD/.venv/bin:$PATH" TORCH_CUDA_ARCH_LIST=8.6 \
VLLM_FLASH_ATTN_SRC_DIR="$PWD/.deps/vllm-flash-attn-src" \
cmake -S . -B build-bytev2-stable -G Ninja ...

PATH="$PWD/.venv/bin:$PATH" cmake --build build-bytev2-stable -j=32 \
  --target _C_stable_libtorch
PATH="$PWD/.venv/bin:$PATH" cmake --install build-bytev2-stable \
  --component _C_stable_libtorch

CUDA_VISIBLE_DEVICES=4 .venv/bin/python -m pytest \
  tests/v1/attention/test_byte_v2_layout.py -v \
  -k "paged_decode_attention_split_k_cuda_matches_raw_reference or paged_decode_attention_split_k_guarded_cuda_matches_raw_reference or paged_decode_attention_split_k_gqa_packed_cuda_matches_raw_reference"

CUDA_VISIBLE_DEVICES=4 .venv/bin/python -m pytest \
  tests/v1/attention/test_byte_v2_layout.py -v \
  -k "paged_decode_attention_cuda_matches_raw_reference or paged_decode_attention_cuda_matches_current_codec_reference"

.venv/bin/pre-commit run --files \
  csrc/libtorch_stable/byte_v2/byte_v2_ops.cu
```

CUDA correctness split-k/guarded/GQA 16 个 selected tests 全部通过；no-split
decode 26 个 selected tests 全部通过；pre-commit 通过。普通
`uv pip install -e . --torch-backend=auto` 在本机 CUDA 13.1 + Torch 2.11 环境会被
无关的 `vllm-flash-attn` FA3/Hopper 目标卡住：
`sm90_get_smem_store_op_for_accumulator` template deduction 失败；因此本轮采用
手动 CMake 只构建 `_C_stable_libtorch`。

isolated decode microbench，GPU4，bn64：

| path | seq_len / partition | before | after | delta |
| --- | --- | ---: | ---: | ---: |
| generic split-k | 8192 / p64 | 0.6164 ms | 0.6093 ms | -1.2% |
| generic split-k | 16384 / p128 | 1.1889 ms | 1.1674 ms | -1.8% |
| GQA4 guarded | 8192 / p128 | 0.3441 ms | 0.3205 ms | -6.9% |
| GQA4 guarded | 16384 / p64 | 0.6175 ms | 0.5898 ms | -4.5% |

GQA4 16384/p64 的 ncu 对比：

| metric | before | after |
| --- | ---: | ---: |
| issued warp / scheduler | 0.44 | 0.47 |
| eligible warps / scheduler | 0.82 | 0.89 |
| warp cycles / issued inst | 23.98 | 22.25 |
| barrier stall | 9.9 cycles | 8.8 cycles |

这个优化值得保留，但收益上限有限。下一步不应继续只抠 block reduction，而应：

1. generic 路径优先处理 `long_scoreboard`：减少 payload/page metadata 地址依赖，
   做 macro descriptor，把 block table、valid row、payload base 在 K/V 路径间复用。
2. GQA4 路径如果继续优化 reduction，应改成 warp-per-q 或更细的 q-head 内并行结构，
   而不是在当前 CTA 结构里继续小改 `__syncthreads()`。
3. 继续评估 FA-style `num_splits` heuristic，但它不是当前 ncu 显示的第一瓶颈。

### 1.9 2026-06-23 generic payload tile descriptor

已完成第一版 kernel-local descriptor，只保留在 generic split-k kernel：

- 每个 physical page + 当前 dim 构造一次 payload tile descriptor。
- descriptor 缓存 payload tile offset、base byte、fallback/outlier bit、
  outlier count 和 outlier payload offset。
- row loop 内不再重复读取 base/mask 或重复计算 outlier/payload tile offset。
- 不新增 CTA barrier，也不新增持久 workspace。

microbench，GPU4，bn64：

| path | case | before | after | delta |
| --- | --- | ---: | ---: | ---: |
| generic | batch=1, 8192, p64 | 0.6062 ms | 0.5478 ms | -9.6% |
| generic | batch=1, 16384, p128 | 1.1658 ms | 1.0414 ms | -10.7% |
| generic | batch=4, 8192, p64 | 2.0511 ms | 1.8258 ms | -11.0% |
| generic | batch=4, 16384, best | 3.8308 ms | 3.3843 ms | -11.7% |

generic 16k/p128 ncu duration 从 1416.6 us 降到 1248.3 us，DRAM throughput 从
64.2 GB/s 升到 72.8 GB/s。寄存器从 44 增到 48；`long_scoreboard` 按
per-issued-instruction 统计从 6.0 升到 7.5，说明外围重复指令减少后，剩余主依赖
更突出。

同样 descriptor 曾尝试用于 GQA4 guarded，但 batch=4 16k/p64 从约 1.758 ms 回退
到约 1.799 ms，已回退；GQA4 仍保留原 loader。下一步 generic 可继续沿这个方向做
page-level K/V descriptor 复用；GQA4 下一步应做 CTA/q_group 结构调整。

继续尝试了一个更轻的 page-level 复用实验：QK 阶段把当前 compute tile 的 page
pointer 和 page unsafe flag 写入 shared memory，PV 阶段复用。这个版本没有新增
CTA barrier，但只减少 block table/page flag 这类外围读取，实测未达到 5% 保留线，
因此已回退。

microbench，GPU4，bn64，unguarded generic：

| case | descriptor-only | page cache experiment | delta |
| --- | ---: | ---: | ---: |
| batch=1, 8192, p64 | 0.5478 ms | 0.5356 ms | -2.2% |
| batch=1, 16384, p128 | 1.0414 ms | 1.0158 ms | -2.5% |
| batch=4, 8192, p64 | 1.8258 ms | 1.7777 ms | -2.6% |
| batch=4, 16384, p64 | 3.3843 ms | 3.2819 ms | -3.0% |
| batch=4, 16384, p128 | 3.3987 ms | 3.2824 ms | -3.4% |

结论：下一步的 macro descriptor 不能只缓存 page pointer，而要继续靠近
FlashAttention 的 paged-KV 内核组织，把 payload/base/mask 或 K/V decode 本体的
依赖压缩掉。

### 1.10 2026-06-23 GQA4 fused q_group reduction

这是 GQA4 QK score path 的第一步 CTA/q_group 结构调整，作为历史增量记录：

- 新增 `byte_v2_block_sum4_thread0_store`，一次 CTA reduction 同时处理 4 个
  q_group 的 QK partial。
- GQA4 split-k kernel 的 row loop 不再对 q_group 逐个调用 block sum，而是一次写入
  `shared_scores[4][tile_offset]`。
- 该改动只影响 GQA4 guarded/no-outlier lower-bound 路径；generic split-k、no-split
  和 split-k reduce kernel 不变。
- shared reduction buffer 从 `NumThreads / 32` 扩为 `4 * NumThreads / 32`，寄存器数
  仍为 40。

后续 1.12 已把 QK score path 改成 warp-per-q_group QK；这个 helper 已从当前代码中
删除。

正确性和构建：

```bash
PATH="$PWD/.venv/bin:$PATH" cmake --build build-bytev2-stable -j=32 \
  --target _C_stable_libtorch
PATH="$PWD/.venv/bin:$PATH" cmake --install build-bytev2-stable \
  --component _C_stable_libtorch

CUDA_VISIBLE_DEVICES=4 .venv/bin/python -m pytest \
  tests/v1/attention/test_byte_v2_layout.py -v \
  -k "paged_decode_attention_split_k_gqa_packed_cuda_matches_raw_reference or paged_decode_attention_split_k_guarded_cuda_matches_raw_reference or paged_decode_attention_split_k_cuda_matches_raw_reference"
```

结果：16 个 selected tests 通过。

isolated decode microbench，GPU4，bn64，guarded no-outlier lower-bound：

| case | previous | fused qreduce | delta |
| --- | ---: | ---: | ---: |
| batch=1, 8192 | 0.3205 ms | 0.2775 ms | -13.4% |
| batch=1, 16384 | 0.5908 ms | 0.4946 ms | -16.3% |
| batch=4, 8192 | 0.9370 ms | 0.8069 ms | -13.9% |
| batch=4, 16384 | 1.7582 ms | 1.5698 ms | -10.7% |

同 raw FA2 的当前差距：

| case | ByteV2 fused qreduce | raw FA2 | gap |
| --- | ---: | ---: | ---: |
| batch=1, 4096 | 0.1577 ms | 0.0645 ms | 2.44x |
| batch=1, 8192 | 0.2775 ms | 0.0727 ms | 3.82x |
| batch=1, 16384 | 0.4946 ms | 0.1239 ms | 3.99x |
| batch=4, 4096 | 0.4352 ms | 0.1239 ms | 3.51x |
| batch=4, 8192 | 0.8069 ms | 0.2222 ms | 3.63x |
| batch=4, 16384 | 1.5698 ms | 0.4209 ms | 3.73x |

GQA4 16384/p64 ncu 对比：

| metric | thread0 block sum | fused qreduce |
| --- | ---: | ---: |
| duration | 662 us | 540 us |
| DRAM throughput | 81.9 GB/s | 100.4 GB/s |
| issue active | 47.2% | 45.9% |
| eligible warps / scheduler | 0.89 | 1.07 |
| short_scoreboard | 5.07 cycles | 1.77 cycles |
| barrier stall | 8.78 cycles | 9.15 cycles |

结论：

1. 该改动达到保留线：GQA4 8k/16k 和 batch=4 长上下文都有 10% 以上收益。
2. 这轮主要减少的是 q_group 串行 reduction 带来的同步和依赖链，总时间下降明显。
   `barrier` 的 per-issued-inst 指标没有下降，原因是指令结构改变后剩余同步占比更高。
3. 这条后续已由 1.11 warp softmax 和 1.12 warp QK 继续推进；当前代码不再使用
   这个 block-sum helper。

### 1.11 2026-06-23 GQA4 warp-per-q softmax

已完成 GQA4 tile softmax 的 q_group 并行化：

- 当时 QK score reduction 仍使用上一节的 fused qreduce，保留单次 K decode 复用。
  后续 1.12 已把 QK score path 改为 warp-per-q_group。
- tile softmax 从 thread0 串行处理 4 个 q_group，改为 4 个 warp 分别处理 4 个
  q_group。
- 每个 warp 内用 shuffle 做 tile max 和 tile sum，lane 分摊 `shared_probs` 写回。
- PV 路径暂不改，仍一次 V decode 后累加 4 个 q_group，避免重复解码 V。

正确性：

```bash
PATH="$PWD/.venv/bin:$PATH" cmake --build build-bytev2-stable -j=32 \
  --target _C_stable_libtorch
PATH="$PWD/.venv/bin:$PATH" cmake --install build-bytev2-stable \
  --component _C_stable_libtorch

CUDA_VISIBLE_DEVICES=4 .venv/bin/python -m pytest \
  tests/v1/attention/test_byte_v2_layout.py -v \
  -k "paged_decode_attention_split_k_gqa_packed_cuda_matches_raw_reference or paged_decode_attention_split_k_guarded_cuda_matches_raw_reference or paged_decode_attention_split_k_cuda_matches_raw_reference"
```

结果：16 个 selected tests 通过。

microbench，GPU4，bn64，guarded no-outlier lower-bound：

| case | fused qreduce | warp softmax | delta |
| --- | ---: | ---: | ---: |
| batch=1, 8192 | 0.2775 ms | 0.2509 ms | -9.6% |
| batch=1, 16384 | 0.4946 ms | 0.4495 ms | -9.1% |
| batch=4, 8192 | 0.8069 ms | 0.7516 ms | -6.9% |
| batch=4, 16384 | 1.5698 ms | 1.4469 ms | -7.8% |

同 raw FA2 的当前差距：

| case | ByteV2 warp softmax | raw FA2 | gap |
| --- | ---: | ---: | ---: |
| batch=1, 4096 | 0.1423 ms | 0.0645 ms | 2.21x |
| batch=1, 8192 | 0.2509 ms | 0.0727 ms | 3.45x |
| batch=1, 16384 | 0.4495 ms | 0.1239 ms | 3.63x |
| batch=4, 4096 | 0.4004 ms | 0.1239 ms | 3.23x |
| batch=4, 8192 | 0.7516 ms | 0.2222 ms | 3.38x |
| batch=4, 16384 | 1.4469 ms | 0.4219 ms | 3.43x |

GQA4 16384/p64 ncu 对比：

| metric | fused qreduce | warp softmax |
| --- | ---: | ---: |
| duration | 540 us | 493 us |
| DRAM throughput | 100.4 GB/s | 136.1 GB/s |
| issue active | 45.9% | 46.6% |
| eligible warps / scheduler | 1.07 | 1.27 |
| registers / thread | 40 | 39 |
| barrier stall | 9.15 cycles | 7.06 cycles |
| short_scoreboard | 1.77 cycles | 1.22 cycles |

结论：

1. warp softmax 达到保留线，且没有牺牲 GQA 的 K/V decode 复用。
2. 这轮真正降低了 `barrier` stall，但 16k/p64 仍有约 7.06 cycles，说明剩余主要来自
   QK row-level CTA reduction。
3. 下一步如果继续 GQA4，应专门评估 QK reduction 的两种方案：shared-K staging
   减少跨 warp reduction barrier，或 warp-per-q QK 直接放弃部分 K decode 复用并用
   microbench 判断是否值得。

### 1.12 2026-06-23 GQA4 warp-per-q QK

已完成 GQA4 QK row-level reduction 的 warp-per-q_group 重写：

- 每个 CTA 仍处理一个 `kv_head`、一个 sequence、一个 partition。
- `threadIdx.x >> 5` 映射到 q_group，4 个 warp 分别处理 4 个 q_group。
- 每个 lane 预加载该 q_group 的 4 个 query dim：`lane + {0,32,64,96}`。
- 每个 row 内，warp 加载对应 4 个 K dim，做 lane-local dot 后用 warp shuffle
  reduce；lane0 写 `shared_scores[q_group][tile_offset]`。
- 去掉了 QK row 内的 shared-memory cross-warp reduction 和 CTA barrier。
- 代价是 K decode 按 q_group 重复，但实测重复 K decode 比 CTA barrier 更便宜。

正确性：

```bash
PATH="$PWD/.venv/bin:$PATH" cmake --build build-bytev2-stable -j=32 \
  --target _C_stable_libtorch
PATH="$PWD/.venv/bin:$PATH" cmake --install build-bytev2-stable \
  --component _C_stable_libtorch

CUDA_VISIBLE_DEVICES=4 .venv/bin/python -m pytest \
  tests/v1/attention/test_byte_v2_layout.py -v \
  -k "paged_decode_attention_split_k_gqa_packed_cuda_matches_raw_reference or paged_decode_attention_split_k_guarded_cuda_matches_raw_reference or paged_decode_attention_split_k_cuda_matches_raw_reference"
```

结果：16 个 selected tests 通过。

microbench，GPU4，bn64，guarded no-outlier lower-bound：

| case | warp softmax | warp QK | delta |
| --- | ---: | ---: | ---: |
| batch=1, 8192 | 0.2509 ms | 0.1976 ms | -21.2% |
| batch=1, 16384 | 0.4495 ms | 0.3748 ms | -16.6% |
| batch=4, 8192 | 0.7516 ms | 0.6216 ms | -17.3% |
| batch=4, 16384 | 1.4469 ms | 1.1837 ms | -18.2% |

同 raw FA2 的当前差距：

| case | ByteV2 warp QK | raw FA2 | gap |
| --- | ---: | ---: | ---: |
| batch=1, 4096 | 0.1106 ms | 0.0645 ms | 1.71x |
| batch=1, 8192 | 0.1976 ms | 0.0737 ms | 2.68x |
| batch=1, 16384 | 0.3748 ms | 0.1239 ms | 3.02x |
| batch=4, 4096 | 0.3282 ms | 0.1249 ms | 2.63x |
| batch=4, 8192 | 0.6216 ms | 0.2232 ms | 2.78x |
| batch=4, 16384 | 1.1837 ms | 0.4219 ms | 2.81x |

GQA4 16384/p64 ncu 对比：

| metric | warp softmax | warp QK |
| --- | ---: | ---: |
| duration | 493 us | 372 us |
| DRAM throughput | 136.1 GB/s | 179.4 GB/s |
| issue active | 46.6% | 72.2% |
| eligible warps / scheduler | 1.27 | 3.26 |
| registers / thread | 39 | 40 |
| barrier stall | 7.06 cycles | 0.07 cycles |
| short_scoreboard | 1.22 cycles | 0.59 cycles |
| long_scoreboard | 3.13 cycles | 3.31 cycles |

结论：

1. warp-per-q QK 达到保留线，且是目前 GQA4 中单轮收益最大的 barrier 改动。
2. `barrier` 已经基本被打掉；继续围绕 `__syncthreads()` 做微优化不再是高优先级。
3. 当前 GQA4 的主差距转向 payload decode / memory dependency、PV 路径和
   split-k workspace。下一步应重新 profile 这些项，而不是继续扩大 q_group
   reduction 改动。

### 1.13 2026-06-23 GQA4 warp split-k reduce

已完成 GQA4 split-k reduce 的专用 warp reduce：

- generic split-k 仍使用原 reduce kernel。
- GQA4 reduce 从 one-CTA-per-head 改为 warp-per-dim。
- 每个 warp 负责一个 output dim，lane 以 stride-32 遍历 partitions。
- warp 内直接计算 LSE max、LSE sum 和 weighted tmp_out sum。
- 不再使用 shared `weights` staging，不再需要 block-wide reduction。
- 代价是每个 dim 重复读取 `exp_sums`；但 batch=1 下并行度提升远大于这部分重复读。

正确性：

```bash
PATH="$PWD/.venv/bin:$PATH" cmake --build build-bytev2-stable -j=32 \
  --target _C_stable_libtorch
PATH="$PWD/.venv/bin:$PATH" cmake --install build-bytev2-stable \
  --component _C_stable_libtorch

CUDA_VISIBLE_DEVICES=4 .venv/bin/python -m pytest \
  tests/v1/attention/test_byte_v2_layout.py -v \
  -k "paged_decode_attention_split_k_gqa_packed_cuda_matches_raw_reference or paged_decode_attention_split_k_guarded_cuda_matches_raw_reference or paged_decode_attention_split_k_cuda_matches_raw_reference"
```

结果：16 个 selected tests 通过。

microbench，GPU4，bn64，guarded no-outlier lower-bound：

| case | warp QK | warp reduce | delta |
| --- | ---: | ---: | ---: |
| batch=1, 8192 | 0.1976 ms | 0.1772 ms | -10.4% |
| batch=1, 16384 | 0.3748 ms | 0.3226 ms | -13.9% |
| batch=4, 8192 | 0.6216 ms | 0.6072 ms | -2.3% |
| batch=4, 16384 | 1.1837 ms | 1.1776 ms | -0.5% |

同 raw FA2 的当前差距：

| case | ByteV2 warp reduce | raw FA2 | gap |
| --- | ---: | ---: | ---: |
| batch=1, 4096 | 0.1024 ms | 0.0645 ms | 1.59x |
| batch=1, 8192 | 0.1772 ms | 0.0727 ms | 2.44x |
| batch=1, 16384 | 0.3226 ms | 0.1239 ms | 2.60x |
| batch=4, 4096 | 0.3195 ms | 0.1249 ms | 2.56x |
| batch=4, 8192 | 0.6072 ms | 0.2232 ms | 2.72x |
| batch=4, 16384 | 1.1776 ms | 0.4219 ms | 2.79x |

GQA4 16384/p64 reduce ncu 对比：

| metric | old reduce | warp reduce |
| --- | ---: | ---: |
| duration | 106 us | 20 us |
| DRAM throughput | 39.9 GB/s | 206.5 GB/s |
| issue active | 2.0% | 26.3% |
| eligible warps / scheduler | 0.02 | 0.51 |
| registers / thread | 25 | 24 |
| barrier stall | 0.15 cycles | 0.00 cycles |

结论：

1. warp reduce 对 batch=1 明显超过保留线；batch=4 只有小幅改善但无明显回退。
2. reduce kernel 自身已从低并行度/低 issue active 状态改善到可接受水平。
3. 当前 GQA4 的下一步不应继续优先 reduce kernel，而应回到主 kernel 的 payload
   decode / memory dependency 和 PV 路径。

### 1.14 2026-06-23 GQA4 PV safe-page descriptor

已完成 GQA4 主 kernel 的 PV safe-page descriptor 优化：

- 只改 GQA4 no-fallback/no-outlier guarded 路径。
- PV 阶段在 page 级别分 safe/unsafe path。
- safe page 使用 `byte_v2_make_payload_tile_descriptor`，把 V dim tile、
  base/payload tile 指针移出 per-row element loader。
- unsafe page 仍走通用 `byte_v2_load_payload_elem`，保持 page flags fallback 语义。
- 代价是主 kernel registers/thread 从 40 增到 48，需要后续继续盯寄存器压力。

正确性：

```bash
PATH="$PWD/.venv/bin:$PATH" cmake --build build-bytev2-stable -j=32 \
  --target _C_stable_libtorch
PATH="$PWD/.venv/bin:$PATH" cmake --install build-bytev2-stable \
  --component _C_stable_libtorch

.venv/bin/python -m pytest tests/v1/attention/test_byte_v2_layout.py -v \
  -k "paged_decode_attention_split_k_gqa_packed_cuda_matches_raw_reference or paged_decode_attention_split_k_guarded_cuda_matches_raw_reference or paged_decode_attention_split_k_cuda_matches_raw_reference"
```

结果：16 个 selected tests 通过。

microbench，GPU4，bn64，guarded no-outlier lower-bound：

| case | warp reduce baseline | PV descriptor | delta |
| --- | ---: | ---: | ---: |
| batch=1, 8192 | 0.1772 ms | 0.1638 ms | -7.5% |
| batch=1, 16384 | 0.3226 ms | 0.3062 ms | -5.1% |
| batch=4, 8192 | 0.6072 ms | 0.5765 ms | -5.1% |
| batch=4, 16384 | 1.1776 ms | 1.1274 ms | -4.3% |

几何平均 speedup 约 1.058x，超过 5% 保留线。

GQA4 16384/p64 主 kernel ncu 对比：

| metric | before PV descriptor | PV descriptor |
| --- | ---: | ---: |
| duration | 372.4 us | 353.8 us |
| DRAM throughput | 179.4 GB/s | 186.8 GB/s |
| issue active | 72.2% | 72.4% |
| eligible warps / scheduler | 3.26 | 2.92 |
| registers / thread | 40 | 48 |
| barrier stall | 0.07 cycles | 0.07 cycles |
| short_scoreboard | 0.59 cycles | 0.70 cycles |
| long_scoreboard | 3.31 cycles | 2.84 cycles |

结论：

1. PV descriptor 确实降低了 payload/PV path 的 memory dependency，`long_scoreboard`
   从 3.31 降到 2.84。
2. 收益主要来自 safe page 上减少 per-row 地址/metadata 重算，和预期一致。
3. 新瓶颈不是 barrier，而是 payload decode 依赖、寄存器压力和 split-k workspace。
   下一步应优先做 QK payload descriptor / macro descriptor，或把 V decode/PV 做更
   接近 FA-style tile staging/vectorization 的结构验证。

### 1.15 2026-06-23 GQA4 QK lightweight descriptor experiment

已尝试 GQA4 QK safe-page lightweight descriptor，但未保留：

- 实现方式：只在 safe page 上缓存 K tile 的 `payload_offset/base`，unsafe page
  继续走通用 loader。
- 正确性：同一组 16 个 selected tests 通过。
- 性能：相对 PV descriptor baseline 的几何平均只有 1.024x，低于 5% 保留线。
- 状态：代码已回退，保留 benchmark artifact 供后续参考。

microbench，GPU4，bn64，guarded no-outlier lower-bound：

| case | PV descriptor baseline | QK lightweight desc | delta |
| --- | ---: | ---: | ---: |
| batch=1, 8192 | 0.1638 ms | 0.1679 ms | +2.5% |
| batch=1, 16384 | 0.3062 ms | 0.2918 ms | -4.7% |
| batch=4, 8192 | 0.5765 ms | 0.5581 ms | -3.2% |
| batch=4, 16384 | 1.1274 ms | 1.0854 ms | -3.7% |

结论：

1. 简单 per-thread QK descriptor 的边际收益不足，8k batch=1 还会回退。
2. 后续不应继续做同形态的 `payload_offset/base` 小缓存；如果继续优化 QK，需要
   做更结构化的 macro descriptor / tile staging，或者直接减少 split-k workspace
   和主 kernel 写放大。
3. PV descriptor 仍保留；本实验只回退 QK path。

### 1.16 2026-06-24 GQA4 BF16 split-k tmp_out experiment

已尝试只在 GQA4 专用路径把 split-k `tmp_out` 从 float32 intermediate 改为 BF16
bits 读写，但未保留：

- 外部 Python workspace 仍保持 float32 tensor，不改接口。
- GQA4 main kernel 内部把 `tmp_out` storage reinterpret 为 `uint16_t*` 写入 BF16 bits。
- GQA4 warp reduce 再按 BF16 bits 读回并转 float 做 weighted reduction。
- 正确性：同一组 16 个 selected tests 通过。
- 性能：相对 PV descriptor baseline 的几何平均只有 1.001x，低于 5% 保留线。
- 状态：代码已回退，保留 benchmark artifact 供后续参考。

microbench，GPU4，bn64，guarded no-outlier lower-bound：

| case | PV descriptor baseline | BF16 tmp_out | delta |
| --- | ---: | ---: | ---: |
| batch=1, 8192 | 0.1638 ms | 0.1618 ms | -1.3% |
| batch=1, 16384 | 0.3062 ms | 0.3082 ms | +0.7% |
| batch=4, 8192 | 0.5765 ms | 0.5786 ms | +0.4% |
| batch=4, 16384 | 1.1274 ms | 1.1274 ms | 0.0% |

结论：

1. GQA4 当前主耗时仍不在 split-k `tmp_out` 读写字节数上；把 intermediate 压成
   BF16 没有明显收益。
2. 该实验会引入内部 storage reinterpret 语义，收益不足以承担复杂度，因此回退。
3. 后续 split-k workspace 方向如果继续做，应考虑减少 partition 数/写入次数、
   fuse main+reduce 的特化路径，或改变工作划分，而不是只压缩 `tmp_out` element size。

### 1.17 2026-06-24 GQA4 K tile staging

已完成并保留 GQA4 safe-page K tile staging：

- 只改 GQA4 no-fallback/no-outlier guarded 路径。
- safe page 上先把一个 16-token page 的 K `[row, dim]` decode 到 shared memory。
- 4 个 q_group/warp 在 QK 阶段复用同一份 staged K tile。
- unsafe page 仍走通用 payload loader。
- 代价是 static shared memory 从约 2.1 KiB 增到约 10.3 KiB，并引入 page 级
  `__syncthreads()`；收益是 K payload global load/decode 不再按 4 个 q_group 重复。

正确性：

```bash
PATH="$PWD/.venv/bin:$PATH" cmake --build build-bytev2-stable -j=32 \
  --target _C_stable_libtorch
PATH="$PWD/.venv/bin:$PATH" cmake --install build-bytev2-stable \
  --component _C_stable_libtorch

.venv/bin/python -m pytest tests/v1/attention/test_byte_v2_layout.py -v \
  -k "paged_decode_attention_split_k_gqa_packed_cuda_matches_raw_reference or paged_decode_attention_split_k_guarded_cuda_matches_raw_reference or paged_decode_attention_split_k_cuda_matches_raw_reference"
```

结果：16 个 selected tests 通过。

microbench，GPU4，bn64，guarded no-outlier lower-bound：

| case | PV descriptor baseline | K staging | delta |
| --- | ---: | ---: | ---: |
| batch=1, 8192 | 0.1638 ms | 0.1413 ms | -13.8% |
| batch=1, 16384 | 0.3062 ms | 0.2570 ms | -16.1% |
| batch=4, 8192 | 0.5765 ms | 0.4741 ms | -17.8% |
| batch=4, 16384 | 1.1274 ms | 0.9124 ms | -19.1% |

几何平均 speedup 约 1.20x，明显超过 5% 保留线。

同 FA2 的当前 gap：

| case | ByteV2 K staging | FA2 | gap |
| --- | ---: | ---: | ---: |
| batch=1, 4096 | 0.0809 ms | 0.0640 ms | 1.26x |
| batch=1, 8192 | 0.1403 ms | 0.0717 ms | 1.96x |
| batch=1, 16384 | 0.2591 ms | 0.1239 ms | 2.09x |
| batch=4, 4096 | 0.2611 ms | 0.1249 ms | 2.09x |
| batch=4, 8192 | 0.4772 ms | 0.2222 ms | 2.15x |
| batch=4, 16384 | 0.9083 ms | 0.4214 ms | 2.16x |

GQA4 16384/p64 main kernel ncu：

| metric | PV descriptor baseline | K staging |
| --- | ---: | ---: |
| duration | 354.0 us | 289.9 us |
| DRAM throughput | 186.7 GB/s | 224.7 GB/s |
| issue active | 72.4% | 57.9% |
| eligible warps / scheduler | 2.92 | 1.12 |
| registers / thread | 48 | 40 |
| static shared memory | 2.1 KiB | 10.3 KiB |
| barrier stall | 0.07 cycles | 0.48 cycles |
| short_scoreboard | 0.70 cycles | 1.73 cycles |
| long_scoreboard | 2.84 cycles | 5.22 cycles |
| global load inst | 7.54M | 2.83M |
| shared load wavefronts | 2.19M | 4.29M |
| shared store wavefronts | 0.57M | 1.10M |

结论：

1. K staging 是当前最有效的结构性改动之一，直接验证了“减少重复 K payload
   decode/load”是正确方向。
2. 它把 global load 指令从 7.54M 降到 2.83M，虽然增加了 shared traffic 和 barrier，
   但主 kernel 仍从 354 us 降到 290 us。
3. 后续不应回到 per-thread 小 descriptor；应继续做 tile/macro 级结构化复用。
   下一步可以评估更细的 double-buffer/staged K layout、PV vectorized/tile decode，
   或减少 split-k partition 写入次数。

### 1.18 2026-06-24 GQA4 BF16 shared K staging

已完成并保留 GQA4 BF16 shared K staging：

- 在 1.17 的 K tile staging 基础上，把 staged K 从 `float` shared memory 改为
  BF16 bits (`uint16_t`) shared memory。
- staging 时直接从 ByteV2 payload 生成 BF16 bits，避免先转 float 再写 shared。
- QK 使用时从 shared 读 BF16 bits，再转 float 参与 dot。
- unsafe page 仍走原通用 payload loader。
- 目标是降低 K staging 引入的 shared memory footprint。

正确性：

```bash
PATH="$PWD/.venv/bin:$PATH" cmake --build build-bytev2-stable -j=32 \
  --target _C_stable_libtorch
PATH="$PWD/.venv/bin:$PATH" cmake --install build-bytev2-stable \
  --component _C_stable_libtorch

.venv/bin/python -m pytest tests/v1/attention/test_byte_v2_layout.py -v \
  -k "paged_decode_attention_split_k_gqa_packed_cuda_matches_raw_reference or paged_decode_attention_split_k_guarded_cuda_matches_raw_reference or paged_decode_attention_split_k_cuda_matches_raw_reference"
```

结果：16 个 selected tests 通过。

microbench，GPU4，bn64，guarded no-outlier lower-bound：

| case | float K staging | BF16 shared K | delta |
| --- | ---: | ---: | ---: |
| batch=1, 8192 | 0.1413 ms | 0.1362 ms | -3.6% |
| batch=1, 16384 | 0.2570 ms | 0.2376 ms | -7.6% |
| batch=4, 8192 | 0.4741 ms | 0.4342 ms | -8.4% |
| batch=4, 16384 | 0.9124 ms | 0.8325 ms | -8.8% |

几何平均 speedup 约 1.077x，超过 5% 保留线。

同 FA2 的当前 gap：

| case | ByteV2 BF16 shared K | FA2 | gap |
| --- | ---: | ---: | ---: |
| batch=1, 4096 | 0.0809 ms | 0.0645 ms | 1.25x |
| batch=1, 8192 | 0.1362 ms | 0.0717 ms | 1.90x |
| batch=1, 16384 | 0.2386 ms | 0.1239 ms | 1.93x |
| batch=4, 4096 | 0.2386 ms | 0.1249 ms | 1.91x |
| batch=4, 8192 | 0.4372 ms | 0.2232 ms | 1.96x |
| batch=4, 16384 | 0.8315 ms | 0.4219 ms | 1.97x |

GQA4 16384/p64 main kernel ncu：

| metric | float K staging | BF16 shared K |
| --- | ---: | ---: |
| duration | 289.9 us | 261.1 us |
| DRAM throughput | 224.7 GB/s | 255.3 GB/s |
| issue active | 57.9% | 65.0% |
| eligible warps / scheduler | 1.12 | 1.94 |
| registers / thread | 40 | 40 |
| static shared memory | 10.3 KiB | 6.2 KiB |
| barrier stall | 0.48 cycles | 0.71 cycles |
| short_scoreboard | 1.73 cycles | 2.06 cycles |
| long_scoreboard | 5.22 cycles | 5.67 cycles |
| global load inst | 2.83M | 2.83M |
| shared load wavefronts | 4.29M | 4.29M |
| shared store wavefronts | 1.10M | 1.11M |

结论：

1. 主要收益来自降低 static shared footprint，使 eligible warps 从 1.12 回升到 1.94。
2. shared load/store wavefront 数没有明显下降，说明 16-bit shared access 没有减少
   NCU 统计到的 wavefront 数，但 occupancy/调度改善足以带来收益。
3. 后续 K staging 优化要围绕 shared layout、bank behavior 和 barrier placement，
   不是再简单压 element size。

## 2. 当前 ByteV2 kernel 结构

代码位置：

- `csrc/libtorch_stable/byte_v2/byte_v2_ops.cu`
    - `byte_v2_paged_decode_attention_kernel`
    - `byte_v2_paged_decode_attention_split_k_kernel`
    - `byte_v2_paged_decode_attention_split_k_reduce_kernel`
- `vllm/v1/attention/backends/byte_v2_attn.py`
    - `_decode_partition_size`
    - `_get_split_k_workspace`
    - `_run_paged_decode`

当前 no-split 和 split-k 主 kernel 的执行模型基本一致：

```text
grid = (num_heads, num_seqs, num_partitions)
block = 128 threads

threadIdx.x = dim
CTA = one q_head, one sequence, one partition

for token tile in partition:
  for token in tile:
    each dim thread loads one K element
    block reduce 128 dim partials -> score[token]

  thread 0 computes tile max / exp / softmax denominator

  for token in tile:
    each dim thread loads one V element
    acc[dim] += prob[token] * V[token, dim]

write tmp_out[seq, head, partition, dim]
write lse[seq, head, partition]
```

几个直接能看到的成本点：

- `head_idx` 是 grid.x，`kv_head_idx = head_idx / q_per_kv`。同一个 `kv_head`
  对应的 4 个 query heads 会分别 launch CTA，并分别走完整 K/V scan。
- `byte_v2_load_payload_elem` 在每个元素上检查 fallback mask、outlier mask、
  outlier count，并执行 payload offset 计算。这个 helper 是正确的通用路径，
  但不是 fast path。
- K path 和 V path 都按 token 循环访问 page/block table，不能复用 page
  descriptor。
- tile 内 softmax 只有 thread 0 做 exp 和归一化，其他 127 个线程等待。
- split-k reduce 已经改成 LSE combine，但仍要写/读 `tmp_out`。

## 3. FlashAttention 对应结构

本地源码位置：`/mnt/sdb/yxz/ByteV2/flash-attention`。

关键参考点：

- `hopper/heuristics.h`
    - `num_splits_heuristic` 根据 `total_mblocks`、SM 数、KV N blocks、KV head
    size 选择 split 数。它不是固定 token partition。
- `hopper/flash_api.cpp`
    - `get_num_splits` 根据 tile size、paged KV、varlen、local/causal 情况生成
    split 上限。
    - split 时申请 `out_accum` 和 `softmax_lse_accum`。
- `csrc/flash_attn/src/flash_fwd_kernel.h`
    - combine 逻辑对 split 维度做 LSE logsumexp，再用
    `exp(lse_i - global_lse)` 加权 partial output。
- `hopper/flash_fwd_combine_kernel.h`
    - Hopper combine 用 tiled copy / shared staging / cp.async 预取 partial output，
    并支持 dynamic split。
- `hopper/pack_gqa.h`
    - PackGQA 把小 query 维度的 GQA case 打包，减少 page table/pointer 计算和
    paged KV 的重复工作。
- `hopper/flash_fwd_kernel_sm80.h`
    - 主 kernel 通过 scheduler 分发 work tile，mainloop 做 MMA/softmax/epilogue，
    而不是一 token 一 reduction。

和 ByteV2 最接近的点是 split combine：二者都写 partial output + LSE，再用 LSE
权重合并。因此继续只微调 reduce kernel，收益大概率有限。

## 4. 可能导致差距的核心差异

### 4.1 GQA 没有打包，重复 KV 读/解压

这是最高优先级。

当前 ByteV2 每个 q head 单独处理。GQA 下 4 个 q heads 共用一个 kv head，但
ByteV2 会对同一个 compressed K/V payload 扫描 4 次、解压 4 次。FlashAttention
在 paged/split 场景会倾向启用 PackGQA，源码里也明确提到 paged KV 的 pointer
计算昂贵，需要让同一行的线程共享 page table entry 和 pointer work。

ByteV2 的压缩优势依赖“少读 HBM”。如果同一份 KV 被 4 个 q heads 重复解压，
压缩带来的 HBM 节省会被重复工作抵消。

建议新增专门 kernel：

```text
byte_v2_paged_decode_attention_gqa4_h128_split_k_kernel

grid = (num_kv_heads, num_seqs, partitions)
CTA handles one kv_head and q_per_kv=4 query heads

for token tile:
  load/decode K once
  compute 4 scores for 4 q heads
  compute 4 softmax states
  load/decode V once
  accumulate 4 outputs
```

首版不一定照搬 CUTE/MMA。先做 plain CUDA 的 GQA-packed no-outlier fast path，
只要 K/V decode 从 4 次降到 1 次，就能验证主方向。

### 4.2 payload decode 逐元素、通用分支太重

`byte_v2_load_payload_elem` 当前每个元素都会做：

- dim/token tile index 计算
- fallback mask load/check
- outlier mask load/check
- base load
- low byte / packed code load
- 可能的 outlier payload scan

即使真实页面没有 fallback/outlier，也会执行 mask load 和分支。FlashAttention
的快路径大量依赖 compile-time specialization 和 vectorized copy；ByteV2 也应该
把 no-fallback/no-outlier 变成独立 hot kernel，而不是在通用 helper 里判断。

建议拆出：

```text
byte_v2_load_payload_elem_no_outlier_no_fallback
byte_v2_decode_codec_tile_no_outlier_no_fallback
```

首版目标：

- 每个 codec tile 的 base/mask 只读一次或少量线程协作读取。
- 对 `16 x 16` codec tile 做向量化读，至少用 `uint32_t`/`uint4` 级别搬运
  low/code payload，再在寄存器中还原 bf16。
- no-outlier kernel 完全不访问 outlier payload 和 outlier count。

### 4.3 page/block metadata 没有 FA-style manager

当前 kernel 在 K loop 和 V loop 中都根据 token 计算 logical block、读
`block_table`、计算 physical page pointer。`ByteV2MacroDesc` 已经在 layout
header 里存在，但 decode kernel 还没使用它。

FlashAttention paged KV 路径的核心不是简单“有 block table”，而是把 paged KV
访问封装成 manager/scheduler，让 pointer 计算、page table entry、tile shape
和 copy pattern 一起优化。

建议下一步把 64/128-token macro tile 的 page 信息提前整理为 descriptor：

```text
ByteV2MacroDesc {
  physical_blocks[8]
  valid_rows[8]
  compressed_mask
  outlier_page_mask
  k_payload_offsets[8]
  v_payload_offsets[8]
}
```

短期可以先在 kernel 内为当前 partition 构造 descriptor；验证收益后再考虑
Python/CUDA 辅助 buffer。

### 4.4 split-k 策略仍是固定 partition size

ByteV2 当前：

```text
short default partition_size = 16
long context >= 4096 default partition_size = 32
num_partitions = ceil(max_seq_len / partition_size)
```

FlashAttention 当前思路：

```text
先确定 kBlockN
num_n_blocks = ceil(seqlen_k_loaded / kBlockN)
total_mblocks = batch_or_dynamic * kv_heads * q_mblocks
num_splits = occupancy heuristic(total_mblocks, num_SMs, num_n_blocks, ...)
```

ByteV2 为 batch=1/long context 用很多 partitions 可以提高 occupancy，但它也会：

- 放大 `tmp_out` HBM 写读。
- 放大 reduce kernel 工作。
- 增加 launch/grid overhead 和 scheduler overhead。
- 在 batch 或 head 数已经足够填满 SM 时继续过度切分。

建议把 API 从“partition_size first”改成“num_splits first”：

```text
num_splits = byte_v2_num_splits_heuristic(
    batch_size,
    num_kv_heads,
    q_per_kv,
    max_seq_len,
    compute_block_n,
    num_sms,
    max_splits,
)
partition_size = ceil_div(max_seq_len, num_splits)
partition_size = round_up(partition_size, alloc_block_tokens)
```

保留当前 env override，但默认走 heuristic。

### 4.5 softmax 和 PV 组织方式仍偏串行

ByteV2 的 tile softmax 由 thread 0 完成，对 `ComputeBlockN=64/128` 做 max/exp/sum，
然后其他线程用 shared_probs 做 PV。这个结构简单且正确，但和 FlashAttention 的
row-wise softmax fragment/reduction 不是同一个级别。

在 decode M 很小的场景，完全照搬 prefill MMA tile 可能不划算；但 GQA-packed 后
M 变成 `q_per_kv`，至少有机会把 4 个 query heads 合在同一 CTA 中并行处理，
减少 thread 0 的串行比例。

## 5. 推荐优化顺序

### Phase A: profile 主 kernel 内部瓶颈

先不要继续猜 reduce。

用 Nsight Compute 或临时 instrumentation 分开确认：

- split main kernel vs reduce kernel 时间占比。
- global load bytes / dram throughput。
- branch efficiency。
- achieved occupancy。
- integer instruction 和 fp instruction 比例。
- `byte_v2_load_payload_elem` 周边的 source line hotspot。

接受标准：

- 能证明主 scan/decode 占主要时间。
- 能区分是 repeated GQA、payload decode、metadata traversal、还是 softmax/PV。

### Phase B: no-outlier/no-fallback fast path

新增独立 loader 和独立 kernel specialization：

```text
byte_v2_paged_decode_attention_split_k_no_outlier_kernel
```

先保持原 grid `(num_heads, num_seqs, partitions)`，只替换 payload loader。
这样风险最低，容易和当前 kernel A/B。

测试：

```bash
CUDA_VISIBLE_DEVICES=4 .venv/bin/python -m pytest \
  tests/v1/attention/test_byte_v2_layout.py -q \
  -k "split_k_cuda_matches_raw_reference"

CUDA_VISIBLE_DEVICES=4 .venv/bin/python scripts/byte_v2_decode_microbench.py \
  --seq-lens 256,512,1024,2048,4096 \
  --partition-sizes 16,32,64 \
  --include-vllm-flash \
  --output-jsonl profiles/byte_v2_decode_microbench_no_outlier_fastpath.jsonl
```

### Phase C: GQA-packed split-k kernel

新增 `gqa4_h128` specialization，先限制：

- `head_dim=128`
- `num_heads=32`
- `num_kv_heads=8`
- `q_per_kv=4`
- no fallback / no outlier fast path
- block size 16
- compute block N 64 或 128

这个 kernel 是最可能缩小差距的点。不要一开始泛化所有模型配置。

建议输出 workspace 形状仍保持 `[seq, q_head, partition, dim]`，先保证接口和
reduce kernel 不变。等 GQA-packed main kernel 证明收益后，再考虑 packed
workspace 或 fused reduce。

### Phase D: FA-style split heuristic

在 GQA-packed 后再做 heuristic，否则 split 参数会被当前重复 KV 工作扭曲。

新增：

```text
BYTE_V2_DECODE_SPLIT_K_NUM_SPLITS
BYTE_V2_DECODE_SPLIT_K_MAX_SPLITS
BYTE_V2_DECODE_SPLIT_K_HEURISTIC=fa_like|partition
```

默认策略：

- batch/head work 已经能填满 SM 时，减少 split。
- batch=1、长上下文时，允许 split，但选择最小的高效率 split 数。
- 仍保证 partition 对齐 `alloc_block_tokens`。

### Phase E: macro descriptor / paged KV manager

把 block table、valid rows、page pointer、payload offset 组织为一个 local
manager，减少 K/V loop 中重复的 page traversal。

可以先做 kernel 内构造：

```text
for macro tile:
  load physical blocks once
  compute row valid mask once
  compute payload base once
  K loop and V loop share descriptor
```

后续再改成持久 workspace。

### Phase F: pipeline/vectorized decode

最后再靠近 FlashAttention mainloop：

- raw payload bytes 用 vectorized copy/cp.async staging。
- decode K/V tile 到 shared 或 registers。
- K tile decode、QK、V tile decode、PV 做双缓冲。
- Hopper 上再评估 TMA/GMMA 是否值得。

这个阶段工程量大，应在 Phase B/C 证明方向后再做。

## 6. 不建议优先做的事

1. 不要继续只优化 split reduce 的 max/sum/LSE 统计量。已经改成 LSE combine，
   4096 p32 decode-only median 基本没有变化。
2. 不要先把所有参数泛化。当前瓶颈在专用热路径，不在 API 泛化。
3. 不要一开始改 vendored FlashAttention。先在 ByteV2 自己的 kernel 里复刻
   必要结构。
4. 不要直接用大型 CUTE 重写整个 kernel。先用 plain CUDA 做 no-outlier 和
   GQA-packed lower bound，确认收益后再决定是否引入 CUTE。

## 7. 推荐下一步

Phase B + Phase C 的最小闭环、显式验证式安全分派，以及 page-level unsafe flags
已经完成。下一步不要直接把 `BYTE_V2_DECODE_ASSUME_NO_OUTLIER=1` 当成生产默认，
也不要继续在元素热路径里加更细粒度 guard，而是优化真实 unsafe/outlier page 的
成本并继续缩小主 kernel 差距：

1. 已完成第一版显式验证式安全分派：
   `BYTE_V2_DECODE_VALIDATE_NO_OUTLIER=1` 会扫描当前 referenced pages 的
   fallback/outlier metadata，并在 unsafe 时回退到通用 decode。
2. 已完成第一版低开销状态维护：在 cache update / commit 路径维护
   page 级 unsafe flags，split-k decode 通过 guarded kernel 查 GPU flags，不做
   每步 stats kernel + `.item()` 同步。
3. 已跑开启 GQA-packed + page flags 的 4096 E2E cached profile：
   guarded GQA4 正确路径把 split-k 平均从 545.6 us 降到 319.6 us，且
   steady-state 不再出现 validation stats kernel + `.item()` 同步。
4. 已尝试 tile-level guard inside unsafe page，正确但更慢，已回退。结论是不要在
   每元素 load 上继续加 mask 细分判断。
5. Nsight Compute 已确认 generic 主 kernel 更像是 `long_scoreboard` /
   payload-page metadata 地址依赖瓶颈，不是 DRAM bandwidth bound。下一步应优化
   unsafe/outlier page 的通用 payload loader：减少 mask/payload offset 重算、
   批量读取 outlier overlay，或调整 cache update 让真实 decode 中 unsafe page 更少。
6. 已尝试 warp-broadcast 读取 fallback/outlier mask，正确但没有性能收益，已回退。
   后续不要单独围绕 mask load 做微优化。
7. 已完成 split-k page-loop 复用，把 page/block metadata 查找从逐 token 降到
   逐 physical page；generic p32 4096 从 0.3799 ms 降到 0.3400 ms，GQA4
   page-guarded safe-page p64 从 0.1961 ms 降到 0.1823 ms。
8. 已跑 4096 E2E cached profile：page-loop 对真实 unsafe/outlier page 的
   guarded GQA4 路径有效，split-k CUDA avg 从 319.6 us 降到 300.8 us，输出
   token IDs 和 page-level guard baseline 一致。
9. 已完成 batch-aware split/partition heuristic：generic 覆盖 batch=1/2/4
   的 4096/8192/16384 档位；GQA4 覆盖 batch=1/2/4 的
   2048/4096/8192/16384 档位；显式 env partition 仍优先。
10. 已完成 8192/16384 E2E cached profile：GQA pageflags token IDs 和 generic
   一致，cached decode kernel 分别快约 1.56x 和 1.65x。
11. 已完成 8192/16384 的 Nsight Compute：generic 主要看 `long_scoreboard`，
   GQA4 主要看 barrier/short_scoreboard；二者都不是单纯的 HBM 带宽瓶颈。
12. 已完成一个小的 decode QK reduction 优化：`byte_v2_block_sum_thread0`
   减少 CTA barrier，GQA4 8192/p128 从 0.3441 ms 到 0.3205 ms，
   16384/p64 从 0.6175 ms 到 0.5898 ms；正确性和 pre-commit 通过。
13. 已完成 GQA4 fused q_group reduction：4 个 q_group 的 QK score block sum 合并
   为一次 CTA reduction。GQA4 batch=1 8192 从 0.3205 ms 到 0.2775 ms，
   16384 从 0.5908 ms 到 0.4946 ms；batch=4 8192/16384 也有 10% 以上收益。
14. 已完成 GQA4 warp-per-q softmax：tile softmax 从 thread0 串行改成 4 个 warp
   分别处理 4 个 q_group。batch=1 8192/16384 分别到 0.2509/0.4495 ms；
   batch=4 8192/16384 分别到 0.7516/1.4469 ms。
15. 已完成 GQA4 warp-per-q QK：row-level QK reduction 从 CTA-wide reduction 改为
   warp 内 reduction。batch=1 8192/16384 分别到 0.1976/0.3748 ms；
   batch=4 8192/16384 分别到 0.6216/1.1837 ms。
16. 已完成 GQA4 warp split-k reduce：batch=1 8192/16384 分别到
   0.1772/0.3226 ms；batch=4 8192/16384 分别到 0.6072/1.1776 ms。
17. 已完成 GQA4 PV safe-page descriptor：batch=1 8192/16384 分别到
   0.1638/0.3062 ms；batch=4 8192/16384 分别到 0.5765/1.1274 ms。
   16k/p64 主 kernel ncu duration 从 372.4 us 降到 353.8 us，`long_scoreboard`
   从 3.31 降到 2.84；代价是 registers/thread 从 40 增到 48。
18. 已尝试 GQA4 QK lightweight descriptor，正确但几何平均只有 1.024x，低于
   5% 保留线，已回退。
19. 已尝试 GQA4 BF16 split-k `tmp_out`，正确但几何平均只有 1.001x，低于
   5% 保留线，已回退。
20. 已完成 GQA4 K tile staging：batch=1 8192/16384 分别到 0.1413/0.2570 ms；
   batch=4 8192/16384 分别到 0.4741/0.9124 ms；16k/p64 main ncu duration
   从 354 us 降到 290 us。
21. 已完成 GQA4 BF16 shared K staging：batch=1 8192/16384 分别到
   0.1362/0.2376 ms；batch=4 8192/16384 分别到 0.4342/0.8325 ms；
   16k/p64 main ncu duration 从 290 us 降到 261 us。
22. 已尝试 GQA4 PV pair-vectorized decode：让相邻两个 dim 共享 low-pair/code-byte
   读取，正确性通过，但 batch=1 8192/16384 从 0.1362/0.2376 ms 退化到
   0.1444/0.2550 ms，未达到 5% 保留线，已回退。结论是不要在当前 PV 标量
   descriptor 路径上加入 shfl 型 pair sharing。
23. 已尝试 GQA4 K staging final score barrier 条件化：正确性通过，但
   batch=1/4 的 8192 基本持平，16384 轻微退化，未达到 5% 保留线，已回退。
   结论是当前 barrier 主要成本不在 tile 末尾这一次全 CTA 同步。
24. 已尝试 GQA4 shared-weight reduce：每 head 一个 CTA 共享 partition 权重，
   正确性通过，但 batch=1 8192/16384 最优退化到 0.1516/0.2934 ms，已回退。
   结论是减少 `exp_sums` 重复读取不值得牺牲 partition 维并行度。
25. 已尝试 GQA4 warp4 grouped reduce：一个 warp 同时处理 4 个 dim、每 dim 8 lanes，
   并在 dim 之间共享 stats 读取；GQA correctness 通过，但 batch=1 8192/16384
   最优退化到 0.1567/0.3021 ms，已回退。结论是现有 warp-per-dim reduce 仍是
   当前最佳 reduce 结构。
26. 已尝试 GQA4 PV in-place accumulation：去掉每 tile 的 `pv[4]` 临时数组，改为
   `acc *= alpha` 后在 PV row loop 中直接累加到 `acc`。GQA correctness 通过，但
   同 GPU 恢复 baseline 对比没有稳定收益：batch=1 8192/16384 基本相同，
   batch=4 8192 仅噪声级波动、16384 持平，低于 5% 保留线，已回退。结论是当前
   `pv[4]` 临时累加不是主要瓶颈，单纯移动累加目标不会改善 V 路径。
27. 已尝试 GQA4 V base half-warp broadcast：每个 16-wide V codec tile 只由
   leader lane 读取 `base`，再通过 `shfl` 广播给同 tile 其它 lane。GQA
   correctness 通过，但同 GPU 对比只有 0-1% 噪声级收益，低于 5% 保留线，已回退。
   结论是 V base metadata load 不是当前主要瓶颈。
28. 下一步 GQA4 不再做同形态的小 descriptor、单纯 `tmp_out` element-size 压缩、
   当前 PV pair sharing、tile-final barrier 条件化，或降低 reduce 并行度的 grouped
   reduce，也不要只把 PV 累加从 `pv[4]` 改成 in-place 或只广播 V base；优先评估
   真正改变 V decode/compute 复用的 PV mapping。generic 方向继续是
   paged KV manager / macro descriptor，处理 `long_scoreboard`。

当前 GQA pageflags 路径已在 4096/8192/16384 E2E 中和 generic token IDs 对齐；
BF16 shared K staging 后 isolated GQA4 decode 在 8192/16384 约 0.1362/0.2376 ms
（batch=1）。因此下一轮优先级应是 GQA4 更结构化的 PV tile decode / mapping、
generic metadata/macro descriptor、真实 unsafe/outlier loader；
`num_splits` heuristic 仍可做，但不是 ncu 显示的第一瓶颈。不要继续只改 split
reduce、降低 reduce 并行度、tile 末尾 barrier 条件化、PV in-place 累加、V base
broadcast，或继续在元素级 guard 上加分支。
