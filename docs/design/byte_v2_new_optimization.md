According to a document from 2026-06-12, 我的建议是：**下一步不要再围绕 outlier arena / split-K / metadata 小分支继续打补丁，主线应回到 decode attention stage1，做一个“独立 fast stage1 kernel + 新 payload/layout microbench”的结构性实验；outlier arena 只作为 capacity/correctness 支线继续。**

## 1. 当前最重要的判断

现在 ByteV2 的核心瓶颈已经很明确：**压缩 KV 的纯读取收益是存在的，但 decode attention stage1 的额外计算和搬运把收益吃掉了。**

文档里 Step 32 证明了 compressed page 的纯 streaming read 时间约为 raw 的 `0.758x`，bytes 约为 raw 的 `0.754x`；也就是说“压缩 KV 能减少 HBM 读取”这个前提成立。但同一段总结也指出，E2E 没超过 raw 的原因是 ByteV2 stage1 的解码指令、metadata 检查、shared layout、split-K reduce 等开销仍大于读带宽收益。

Step 31 的测量也支持这个结论：当前 decode measured range 里，ByteV2 额外 GPU kernel time 主要来自 decode attention stage1；文档总结为 raw FlashAttention decode 约 `93.3 ms`，ByteV2 stage1+reduce 约 `434.6 ms`。因此，**真正该攻的是 `byte_v2_paged_decode_attention_gqa_wmma_split_stage1_kernel`，不是 cache update 或 split-K heuristic。**

## 2. 下一步优先级

### P0：先固定一个“性能基线”，不要默认打开 outlier arena

我建议当前默认性能基线保持：

```bash
VLLM_BYTE_V2_USE_NATIVE_KERNELS=1
VLLM_BYTE_V2_COMPRESSED_ONLY_CACHE=1
VLLM_BYTE_V2_ENABLE_SPARSE_FALLBACK_POOL=1
VLLM_BYTE_V2_DEFERRED_CACHE_UPDATE_ERROR_CHECK=1
VLLM_BYTE_V2_DECODE_PAGE_FASTPATH=1
VLLM_BYTE_V2_DECODE_TILE_FASTPATH=1
VLLM_BYTE_V2_DECODE_SPLIT_K=0
VLLM_BYTE_V2_ENABLE_OUTLIER_ARENA=0
```

原因是 outlier arena 虽然能显著减少 raw tile fallback，例如 p1024/b1/d64 下 raw tile fallback 从约 1.55 万降到 903，但 E2E 没有变快，反而从 no-arena 的 15.63 tok/s 降到 arena max=1 的 14.75 tok/s；arena max=0 的长 decode 还出现明显退化。文档明确判断 compact outlier arena 不应作为默认性能路径，只适合作为 opt-in correctness/capacity 实验。

Step 39 已经把 arena encode 端做了很大优化：single-outlier fast path 把 arena encode 从 257.1 us 降到 75.8 us，但文档也指出后续如果继续 arena，重点应看 decode stage1 overlay metadata check 和 CUTE/independent stage1 的 overlay 融合，而不是继续加 encode 端复杂度。

### P1：主线做“独立 no-fallback fast stage1 kernel”

下一步最值得做的是一个**独立 kernel symbol**，不要继续往当前 WMMA kernel 里塞 env 分支。目标不是马上支持所有 fallback/outlier，而是先把最干净的 compressed no-fallback hot path 做快。

建议新增：

```cpp
byte_v2_paged_decode_attention_gqa4_h128_fast_split_stage1_kernel
```

第一版只支持：

```text
Llama-3-8B shape:
  head_size = 128
  head_size_v = 128
  q_per_kv = 4
  num_heads = 32
  num_kv_heads = 8
  block_size = 16
  dtype = bf16

runtime:
  compressed-only
  no outlier arena
  no tile fallback overlay
  split-K path
```

fallback、outlier、非 Llama shape 全部回退当前稳定路径。这样能避免之前 CUTE/WMMA 实验污染默认 kernel 编译形态的问题。文档里也明确给过类似原则：如果继续 CUTE/CUTLASS-style stage1，应该使用独立 symbol/launcher，并且默认 WMMA kernel 不能被实验模板影响。

这个 kernel 的重点不是再做 pair/vec4 小修，而是整体重写 hot path：

```text
page loop:
  load/decode K tile -> MMA-friendly shared
  QK
  warp-parallel online softmax
  load/decode V tile -> MMA-friendly shared
  PV
  write partial
```

保留已有成功经验：独立 stage1 里的 warp-parallel softmax 曾经带来正向收益；但不要重复已经失败的 packed store、base broadcast、K-first/V-late、CTA-local double-buffer 等小改动。

验收门槛建议设高一点：

```text
decode-only:
  p1024/p2048, fallback=0, split_k=16/64
  median_us 至少提升 8%-10%

E2E:
  p512/b4/d128 或 p2048/b1/d32
  output tok/s 至少提升 3%-5%

NCU:
  stage1 duration 下降
  integer/memory instructions 下降
  long scoreboard stall 下降
  eligible warps 不下降
```

如果 decode-only 有收益但 E2E 没收益，默认不保留。

### P2：在 microbench 中继续研究 payload/layout-v2，但不要直接改生产格式

Step 33 已经试过 `pair-interleaved 3B` layout-v2，没有达到保留门槛；r32 有 1.68% 正向，r128 反而退化 2.46%，说明简单把 low/code 变成 3B 邻近重排不是方向。文档也总结：后续 payload layout 不能只做 3B 邻近重排，应考虑 warp/lane-aligned 4B group 或更高压缩率格式，但要接受 bytes ratio 和 decode 指令之间的 tradeoff。

下一轮 layout 实验建议只放在：

```text
byte_v2_decode_page_wmma_microbench
```

不要接入 production decode。目标是先回答一个问题：

> 能不能在不显著增加 page bytes 的前提下，把 ByteV2 decode 的 integer/memory instruction 和 long scoreboard stall 降下来？

可以尝试两个方向：

**方向 A：warp/lane-major layout**

不要按 tile 的自然 `[row][dim]` 或简单 pair interleave 存，而是按当前 stage1 中线程读取顺序重排 payload，例如每 32 lane 对应一个连续 stripe：

```text
tile payload:
  base
  low_stripe[warp_id][lane-local continuous]
  code_stripe[warp_id][lane-local continuous]
```

K 和 V 可以采用不同物理顺序：

```text
K: 面向 QK 的 MMA-B/shared layout
V: 面向 PV 的 row-major / output-dim layout
```

**方向 B：小幅牺牲压缩率换 decode latency**

当前 ByteV2 page 大小约 raw 的 0.754x。可以设计一个 `ByteV2-Fast` variant，比如把部分 3B pair group pad 到 4B 或做 group-of-8 16B 对齐。压缩率可能从 1.326x 降到比如 1.20x，但如果 decode stage1 能明显下降，E2E 可能更好。文档里 Step 32 已经证明纯读收益存在，但现在的问题是 decode 开销盖过了读收益；所以“少压一点、解得快很多”可能比“压得紧、解得慢”更值得。

这个方向的保留门槛应是：

```text
decode_page_wmma_microbench >= +5%
NCU integer/memory instructions 下降
long scoreboard stall 下降
page bytes 仍至少比 raw 小 15%-20%
```

否则不要进入 E2E。

## 3. outlier arena 下一步怎么做

outlier arena 不建议作为性能主线，但它对 capacity 很有价值。文档里 Step 34 的统计很关键：真实 Llama-3 8B p1024/p2048 下 bad tile 几乎都是 1-2 个 outlier，`max_outliers=2` 的估算只需要 raw tile fallback bytes 的约 0.79%。这说明 element-level outlier list 是正确的 capacity 方向。

下一步 outlier 支线应该补完整性，而不是追 E2E 性能：

1. 补 outlier-aware `byte_v2_decompress_page_to_raw_block()`，否则带 outlier 的 compressed page 在 partial append / continuation update 时可能丢 outlier。
2. 补 continuation prefill、decode append、general touched-block compress 的 outlier arena 写入或保真回退。
3. 增加 outlier arena exhausted 的 CUDA 单测。
4. 重新跑高并发长 prompt capacity pressure，例如之前 3% pool 会耗尽的 p2048/b82/prefix-off 场景。

验收标准：

```text
3% pool 原先耗尽的高并发场景能跑通
KV capacity 仍保持 raw 的 1.20x 左右
常规 p512/p2048 E2E 不比 no-arena 慢超过 3%
```

如果做不到，就把 outlier arena 明确定位成“capacity experimental mode”，默认关闭。

## 4. cache update 还有没有必要继续做？

有必要，但不是第一优先级。Step 25/26/29 已经把 decode append 从毫秒级修到了比较合理的程度；文档里 raw cache update 约 3.6-3.9 us/step，ByteV2 在 skip-sync + parallel finalize 下约 8.9 us/step，差距还在，但已经不是最大瓶颈。

下一步 cache update 可以做一个低风险 fusion：

```text
把 init_decode_append_result
+ byte_v2_decode_append_cache_kernel
+ byte_v2_record_deferred_cache_update_error_kernel

融合成一个 append kernel 内部写 sticky error。
```

目标是减少每层每 token 的小 kernel/node。Step 31 里 decode cache update 的差距大约是 ByteV2 40.2 ms vs raw 9.8 ms，虽然只解释总体差距的一小部分，但如果 fusion 简单，值得做。

保留门槛：

```text
E2E p512/b4/d128 >= +1.5%
node-level profile 中 cache update kernel time 明显下降
不能破坏 deferred-safe error reporting
```

如果只是 microbench 好看，E2E 没变，就不要默认开启。

## 5. 明确不要再优先做的方向

这些方向文档里已经有负结果，下一步不要再花时间重复：

| 方向                                 | 原因                                                                   |
| ---------------------------------- | -------------------------------------------------------------------- |
| 继续调大 split-K                       | decode-only 有时变快，但 E2E 不兑现；长 decode 强制 split64 只有噪声级收益。              |
| persistent partial workspace 默认开启  | E2E 收益在噪声内。                                                          |
| direct-output decode               | 已做过，收益在噪声内。                                                          |
| page/block has-outlier flag 默认开启   | 合成场景有用，真实 Llama 多数 block 都有 outlier，E2E 无可观收益。                       |
| tile-level outlier bitmap 作为默认性能路径 | 功能打通，但 decode-only 没有稳定性能提升，默认关闭更合理。                                 |
| lossy=1 默认开启                       | 能降 fallback，但性能不一定更好，而且需要精度评估；文档明确建议默认关闭。                            |
| raw-fallback-only 独立 stage1        | 额外 launch/reduce 成本抵消收益。                                             |
| 当前 WMMA kernel 内的小 decoder 微调      | vec4、pair_v2、base prefetch、base broadcast、packed shared store都没达到门槛。 |

## 6. 我建议的下一轮实验顺序

### Step A：刷新“无 arena 性能基线”

先跑一组固定 benchmark，作为后面所有实验对照：

```bash
# E2E
CUDA_VISIBLE_DEVICES=2 \
VLLM_ENABLE_V1_MULTIPROCESSING=0 \
VLLM_BYTE_V2_USE_NATIVE_KERNELS=1 \
VLLM_BYTE_V2_COMPRESSED_ONLY_CACHE=1 \
VLLM_BYTE_V2_ENABLE_SPARSE_FALLBACK_POOL=1 \
VLLM_BYTE_V2_SPARSE_FALLBACK_POOL_RATIO=0.03 \
VLLM_BYTE_V2_DEFERRED_CACHE_UPDATE_ERROR_CHECK=1 \
VLLM_BYTE_V2_DECODE_PAGE_FASTPATH=1 \
VLLM_BYTE_V2_DECODE_TILE_FASTPATH=1 \
VLLM_BYTE_V2_DECODE_SPLIT_K=0 \
VLLM_BYTE_V2_ENABLE_OUTLIER_ARENA=0 \
.venv/bin/python benchmarks/benchmark_byte_v2_decode_e2e.py \
  --modes raw,byte_v2_compressed_only \
  --model /mnt/sda1/yxz/byte_v2/Meta-Llama-3-8B-Instruct \
  --prompt-len 512 \
  --decode-lens 128,256 \
  --batch-size 4 \
  --num-runs 3 \
  --warmup-decode-len 8 \
  --gpu-memory-utilization 0.80
```

同时跑 node-level Nsight，确认 stage1/cache update 分布没有变。

### Step B：实现独立 `gqa4_h128_fast_stage1` no-fallback kernel

只支持 fixed shape，不支持 outlier/tile fallback。dispatch 条件严格：

```text
if llama_shape &&
   compressed_only &&
   outlier_arena == nullptr &&
   fallback_tile_ids == nullptr or no tile fallback in batch:
    use fast_stage1
else:
    old path
```

第一版甚至可以只接 decode-only benchmark，不接 E2E。通过后再接 E2E。

### Step C：在 microbench 中做 layout-v3

复用 `byte_v2_decode_page_wmma_microbench`。只要 microbench 达不到 +5%，不要接入 stage1。

### Step D：cache update kernel fusion

在 stage1 没有快速突破之前，可以并行做一个小 patch：把 deferred error record 融进 append kernel。这个 patch 风险低，但收益预期也低，不要把它当主线。

### Step E：outlier arena 补 correctness/capacity 闭环

补全 `decompress_page_to_raw_block()`、continuation/decode append/general touched-block outlier 路径和 exhaustion 单测。完成后再跑 p2048/b82/prefix-off 的 capacity pressure。

## 7. 一句话结论

**下一步应该把 outlier arena 暂时从性能主线移开，固定 no-arena 的最快稳定配置，然后集中做一个独立的 compressed no-fallback fast stage1 kernel；同时在 microbench 里探索新的 lane/MMA-friendly payload layout。** 只有当 stage1 的 decode/load 指令和 long scoreboard stall 明显下降，ByteV2 的 0.754x 纯读取优势才有机会在 E2E 里体现出来。

## 8. 实验记录：no-arena baseline 和 fast stage1 v1

日期：2026-06-12。

### Step A：no-arena production baseline

命令按本文 Step A 执行，改用空闲的 `CUDA_VISIBLE_DEVICES=1`，输出：

```text
benchmarks/profiles/bytev2_no_arena_baseline_p512_b4_d128_256_r3.json
```

配置：

```text
model=/mnt/sda1/yxz/byte_v2/Meta-Llama-3-8B-Instruct
prompt_len=512
decode_lens=128,256
batch_size=4
num_runs=3
production mode, cudagraph enabled
VLLM_BYTE_V2_ENABLE_OUTLIER_ARENA=0
VLLM_BYTE_V2_DECODE_SPLIT_K=0
VLLM_BYTE_V2_DECODE_PAGE_FASTPATH=1
VLLM_BYTE_V2_DECODE_TILE_FASTPATH=1
VLLM_BYTE_V2_SPARSE_FALLBACK_POOL_RATIO=0.03
```

结果：

| mode | decode_len | median output tok/s | median elapsed |
| --- | ---: | ---: | ---: |
| raw | 128 | 133.60 | 3.832 s |
| raw | 256 | 133.45 | 7.673 s |
| ByteV2 no-arena | 128 | 123.19 | 4.156 s |
| ByteV2 no-arena | 256 | 123.78 | 8.273 s |

ByteV2 no-arena 相对 raw：

| decode_len | ByteV2/raw |
| ---: | ---: |
| 128 | 92.21% |
| 256 | 92.75% |

sparse fallback pool 未耗尽：

```text
any_exhausted=false
total_assigned_blocks=896
total_assigned_tiles=24273
```

结论：在 batch4 + cudagraph production 场景，no-arena ByteV2 已经接近 raw，
明显好于 batch1/eager 长 decode 的结果。后续优化需要同时看 batch/head-group
并行度，不能只用 batch1/eager 判断。

### Step B：独立 no-fallback fast stage1 v1

实现内容：

- 新增 opt-in env：`VLLM_BYTE_V2_DECODE_FAST_STAGE1=1`。
- 新增独立 kernel：
  `byte_v2_paged_decode_attention_gqa4_h128_fast_split_stage1_kernel`。
- 第一版只支持：
  - Llama-3 8B shape：`num_heads=32`、`num_kv_heads=8`、
    `head_size=head_size_v=128`、`q_per_kv=4`。
  - compressed-only page size。
  - split-K path。
  - 无 sparse fallback metadata、无 tile fallback metadata、无 outlier arena。
- 增加 full-row K/V decode helper，完整 16-token block 避免每个元素的
  `row < valid_rows` 分支。
- 第二轮尝试在完整 block 跳过 `page_valid_rows` header load，只在尾块读取。
- 新增 decode-only benchmark 参数：`--fast-stage1`。
- 新增 CUDA 正确性测试：
  `test_byte_v2_paged_decode_attention_op_fast_stage1_cuda`。

验证：

```bash
.venv/bin/python -m py_compile \
  benchmarks/kernels/benchmark_byte_v2_decode_kernel.py \
  benchmarks/benchmark_byte_v2_decode_e2e.py \
  tests/v1/attention/test_byte_v2_decode.py \
  vllm/envs.py

.venv/bin/python -m ruff check \
  benchmarks/kernels/benchmark_byte_v2_decode_kernel.py \
  benchmarks/benchmark_byte_v2_decode_e2e.py \
  tests/v1/attention/test_byte_v2_decode.py \
  vllm/envs.py

git diff --check -- \
  csrc/libtorch_stable/cache_kernels.cu \
  benchmarks/kernels/benchmark_byte_v2_decode_kernel.py \
  benchmarks/benchmark_byte_v2_decode_e2e.py \
  tests/v1/attention/test_byte_v2_decode.py \
  vllm/envs.py

uv pip install --python .venv/bin/python -e . --torch-backend=auto

CUDA_VISIBLE_DEVICES=1 .venv/bin/python -m pytest \
  tests/v1/attention/test_byte_v2_decode.py::test_byte_v2_paged_decode_attention_op_fast_stage1_cuda \
  tests/v1/attention/test_byte_v2_decode.py::test_byte_v2_paged_decode_attention_op_cute_stage1_cuda \
  -q -s
```

结果：检查通过，native rebuild 通过，CUDA 测试 `2 passed`。

decode-only A/B，输出：

```text
benchmarks/profiles/bytev2_fast_stage1_base_p1024_p2048_s16_s64_fb0.json
benchmarks/profiles/bytev2_fast_stage1_cute_p1024_p2048_s16_s64_fb0.json
benchmarks/profiles/bytev2_fast_stage1_fast_after_validrows_p1024_p2048_s16_s64_fb0.json
```

配置：

```text
batch_size=1
seq_len=1024,2048
split_k=16,64
fallback_ratio=0
partial_workspace=true
correctness skipped for timing; correctness covered by CUDA unit test
```

结果：

| seq_len | split_k | page-fastpath | CUTE no-metadata | fast stage1 v1 |
| ---: | ---: | ---: | ---: | ---: |
| 1024 | 16 | 1038.26 us | 1009.15 us | 1004.54 us |
| 1024 | 64 | 1116.16 us | 994.30 us | 994.30 us |
| 2048 | 16 | 1201.15 us | 1131.52 us | 1125.38 us |
| 2048 | 64 | 1137.66 us | 1107.44 us | 1104.90 us |

相对 page-fastpath：

| seq_len | split_k | CUTE | fast stage1 v1 |
| ---: | ---: | ---: | ---: |
| 1024 | 16 | +2.88% | +3.36% |
| 1024 | 64 | +12.26% | +12.26% |
| 2048 | 16 | +6.15% | +6.73% |
| 2048 | 64 | +2.73% | +2.97% |

相对已有 CUTE no-metadata，fast stage1 v1 只有 `0.0%-0.55%`，属于噪声级
到极小收益。

决策：

- `VLLM_BYTE_V2_DECODE_FAST_STAGE1` 保持 opt-in，不进入默认路径。
- 该 v1 没达到“相对已有 best stage1 提升 8%-10%”的门槛。
- 保留它只作为后续 layout-v3 / lane-major loader 实验 scaffold；如果后续
  layout 实验仍不能显著拉开与 CUTE 的差距，应删除该 opt-in kernel。

### CUTE metadata E2E 补测

虽然 fast stage1 v1 不适用于带 sparse fallback metadata 的真实 no-arena
E2E，本轮顺手补测了现有 CUTE metadata stage1：

```text
benchmarks/profiles/bytev2_no_arena_cute_p512_b4_d128_256_r3.json
```

配置同 Step A，但额外设置：

```text
VLLM_BYTE_V2_DECODE_CUTE_STAGE1=1
```

结果：

| mode | decode_len | median output tok/s | median elapsed |
| --- | ---: | ---: | ---: |
| ByteV2 no-arena + CUTE metadata | 128 | 126.29 | 4.054 s |
| ByteV2 no-arena + CUTE metadata | 256 | 125.35 | 8.169 s |

相对 no-CUTE ByteV2 baseline：

| decode_len | delta |
| ---: | ---: |
| 128 | +2.52% |
| 256 | +1.27% |

相对 raw：

| decode_len | CUTE metadata ByteV2/raw |
| ---: | ---: |
| 128 | 94.54% |
| 256 | 93.93% |

sparse fallback pool 未耗尽：

```text
any_exhausted=false
total_assigned_blocks=896
total_assigned_tiles=24226
```

结论：

- batch4/no-arena production 场景下，CUTE metadata stage1 有 E2E 正收益。
- 但之前 batch1/eager/arena 场景 CUTE 没有收益甚至略退化，所以不能简单默认
  全局开启。
- 下一步更合理的是做一个保守 heuristic，例如只在 `active_head_groups >= 32`
  且 Llama-3 8B shape 时自动启用 CUTE metadata；需要再补 batch1、batch4、
  不同 decode_len 的 A/B 后才能改默认。

### 固定 no-arena 稳定基线与 CUTE metadata auto

为了避免 outlier arena 实验噪声影响后续判断，当前把以下配置固定为
production-safe no-arena baseline：

```text
VLLM_BYTE_V2_ENABLE_OUTLIER_ARENA=0
VLLM_BYTE_V2_ENABLE_SPARSE_FALLBACK_POOL=1
VLLM_BYTE_V2_SPARSE_FALLBACK_POOL_RATIO=0.03
VLLM_BYTE_V2_DECODE_PAGE_FASTPATH=1
VLLM_BYTE_V2_DECODE_TILE_FASTPATH=1
VLLM_BYTE_V2_DECODE_SPLIT_K=0
VLLM_BYTE_V2_DEFERRED_CACHE_UPDATE_ERROR_CHECK=1
```

固定对照结果：

```text
benchmarks/profiles/bytev2_no_arena_baseline_p512_b4_d128_256_r3.json
```

| mode | decode_len | median output tok/s | ByteV2/raw |
| --- | ---: | ---: | ---: |
| raw | 128 | 133.60 | 100.00% |
| raw | 256 | 133.45 | 100.00% |
| ByteV2 no-arena baseline | 128 | 123.19 | 92.21% |
| ByteV2 no-arena baseline | 256 | 123.78 | 92.75% |

新增保守 auto 开关：

```text
VLLM_BYTE_V2_DECODE_CUTE_STAGE1_AUTO=1
```

auto 只在以下条件全部满足时启用 CUTE metadata stage1：

```text
split-K GQA WMMA path
compressed-only page size
VLLM_BYTE_V2_DECODE_PAGE_FASTPATH=1
num_heads=32
num_kv_heads=8
head_size=head_size_v=128
q_per_kv=4
active_head_groups = num_decode_tokens * num_kv_heads >= 32
or num_logical_pages >= 128
has sparse fallback metadata or outlier arena metadata
no outlier arena
no outlier block flags
no outlier tile bitmap
```

含义：

- batch1 短上下文通常只有 `active_head_groups=8` 且 pages 较少，auto 不触发，
  避免小 batch 退化。
- batch1 长上下文在 `num_logical_pages >= 128` 时触发 auto。实验 B 显示
  p2048/b1 下 oracle 快于当前 fused，根因之一是 CUTE auto 未启用；强制
  CUTE metadata 的 nsys stage1+reduce 从约 `82.76 us/op` 降到
  `55.55 us/op`，所以 long-context batch1 需要单独放行。
- batch4、Llama-3 8B、no-arena sparse fallback metadata 触发 auto；这是当前
  E2E 观测到 CUTE metadata 有收益的最小生产形状。
- `VLLM_BYTE_V2_DECODE_CUTE_STAGE1=1` 仍保留强制实验语义，不受 auto 限制。

auto E2E 结果：

```text
benchmarks/profiles/bytev2_no_arena_cute_auto_p512_b4_d128_256_r3.json
```

| mode | decode_len | median output tok/s | vs no-arena baseline | ByteV2/raw |
| --- | ---: | ---: | ---: | ---: |
| ByteV2 no-arena + CUTE auto | 128 | 126.27 | +2.50% | 94.51% |
| ByteV2 no-arena + CUTE auto | 256 | 125.36 | +1.27% | 93.93% |

该结果与强制 `VLLM_BYTE_V2_DECODE_CUTE_STAGE1=1` 基本一致，说明 auto 在
batch4/no-arena metadata 场景触发了预期路径。sparse fallback pool 未耗尽：

```text
any_exhausted=false
total_assigned_blocks=896
total_assigned_tiles=22892
```

batch1 guard decode-only 复测：

```text
benchmarks/profiles/bytev2_cute_auto_b1_guard_base_repeat_p1024_s16_fb003.json
benchmarks/profiles/bytev2_cute_auto_b1_guard_auto_repeat_p1024_s16_fb003.json
```

| batch_size | seq_len | split_k | fallback_ratio | auto | median latency |
| ---: | ---: | ---: | ---: | --- | ---: |
| 1 | 1024 | 16 | 0.03 | off | 1034.24 us |
| 1 | 1024 | 16 | 0.03 | on | 1031.17 us |

batch1 的 `active_head_groups=8`，不满足 auto 门槛；repeat 结果持平，说明保守
heuristic 没有把小 batch 自动切到 CUTE metadata 路径。

#### long-context batch1 auto 放行实验

实验 B 后追加 A/B：

```text
p2048/b1, seq_len=2064, split_k=64, fallback_ratio=0.03
```

| mode | stage1 | reduce | stage1+reduce |
| --- | ---: | ---: | ---: |
| old auto, not triggered | 75.79 us | 6.96 us | 82.76 us |
| forced CUTE metadata | 48.91 us | 6.65 us | 55.55 us |
| new auto, long-context triggered | 49.27 us | 6.79 us | 56.07 us |

因此 auto heuristic 扩展为：

```text
active_head_groups >= 32 || num_logical_pages >= 128
```

该变更只影响同样的 fixed Llama-3 8B/no-arena metadata shape；短 batch1 仍不触发。

验证文件：

```text
benchmarks/profiles/nsys_cute_auto_long_b1_before_p2048_s64.nsys-rep
benchmarks/profiles/nsys_cute_force_long_b1_before_p2048_s64.nsys-rep
benchmarks/profiles/nsys_cute_auto_long_b1_after_p2048_s64.nsys-rep
benchmarks/profiles/cute_auto_short_b1_after_p1024_s64.json
```

## Step C：decode attention stage1 下限实验

四路 A/B profile 已经把问题收敛到 decode attention stage1：当前
ByteV2 no-arena + CUTE metadata 虽然已经是较好的版本，但 attention 仍明显慢于
raw FlashAttention/FlashInfer decode。

| workload | raw attention | ByteV2 stage1+reduce | ByteV2 / raw |
| --- | ---: | ---: | ---: |
| p512/b4/d128 | 93.1 ms | 224.7 ms | 2.41x |
| p512/b4/d256 | 203.5 ms | 565.6 ms | 2.78x |
| p2048/b1/d32 | 29.8 ms | 72.1 ms | 2.42x |

同时，read-only microbench 已经证明 compressed page 纯读时间约为 raw 的
`0.758x`，bytes ratio 约为 `0.754x`。因此 ByteV2 的真实 stage1 不是 HBM
读带宽占优后接近 raw，而是被 decode 指令、metadata 检查、shared 搬运、WMMA
layout、softmax/PV 结构和 partial 写回等额外成本吃掉。

### C.1 目标

下一步不继续优先投入 split-K、reduce、output copy、outlier bitmap 等外围点，
而是建立下限 / oracle 实验，回答：

```text
ByteV2 慢，是因为自研 attention skeleton 慢，
还是 ByteV2 compressed decode/load 慢，
还是 metadata/fallback 慢？
```

### C.2 实验 A：custom attention skeleton lower bound

固定 workload：

```text
p512/b4/d128
p512/b4/d256
p2048/b1/d32
```

构造四路：

| variant | K/V 来源 | attention skeleton | 目的 |
| --- | --- | --- | --- |
| R0 | raw BF16 KV | raw FlashAttention/FlashInfer | raw baseline |
| R1 | raw BF16 KV | ByteV2 WMMA split-stage1 | 自研 attention skeleton 下限 |
| R2 | ByteV2 compressed KV，无 fallback/meta | ByteV2 WMMA split-stage1 | 纯 ByteV2 decode/load 成本 |
| R3 | ByteV2 compressed KV，真实 fallback/meta | ByteV2 WMMA split-stage1 | 当前真实 metadata 路径 |

关键差值：

```text
R1 - R0 = 自研 attention skeleton 相比 raw FlashAttention 的差距
R2 - R1 = ByteV2 decode/load 格式本身的差距
R3 - R2 = metadata / fallback / overlay 的差距
```

当前第一轮先用已有 `--raw-overlay-pages` 近似 R1：raw BF16 K/V 存在
ByteV2 raw-overlay page 内，仍经过 ByteV2 paged decode skeleton。该近似仍带有
generic raw page loader 的分支/地址开销；如果结果显示 R1 已显著慢于 R0，再新增
专门 raw-BF16 skeleton kernel 去去除这部分误差。

判断：

- 如果 R1 已经比 R0 慢 `>1.5x`，主因是自研 attention skeleton；下一步转向
  FlashInfer 集成或独立 CUTE/CUTLASS-style stage1。
- 如果 R1 接近 R0，但 R2 明显慢，主因是 ByteV2 payload decode/load/layout。
- 如果 R2 接近 R1，但 R3 明显慢，主因是 metadata/fallback。

### C.3 实验 B：decompress-to-BF16 + raw FlashAttention oracle

构造：

```text
ByteV2 compressed pages
-> 临时 BF16 K/V workspace
-> raw FlashAttention/FlashInfer decode
```

这不是 production 路径，只用来判断 raw attention kernel 结构优势是否足够大。

| 结果 | 说明 | 下一步 |
| --- | --- | --- |
| decompress + raw FA 快于当前 ByteV2 fused stage1 | 自研 stage1 是主因 | 考虑 FlashInfer 集成 / FA-compatible compressed loader |
| decompress + raw FA 更慢 | fused ByteV2 attention 有价值 | 继续优化 ByteV2 decode/load format |
| decompress 成本低且 raw FA 很快 | staging 策略可能可行 | 做分层 staging prototype |
| decompress 成本高 | 必须 fused decode + attention | 继续 fused kernel |

### C.4 实验 C：CUTE metadata early-exit profile

在当前最优 CUTE metadata 路径上重做 early-exit：

```text
mode0: full stage1
mode1: load/decode K/V 后退出
mode2: load/decode K/V + QK 后退出
mode3: load/decode K/V + QK + softmax 后退出
mode4: full stage1 + PV/write partial
```

输出：

| workload | load/decode | QK | softmax | PV/write | total |
| --- | ---: | ---: | ---: | ---: | ---: |
| p512/b4/d128 | 37.73 us | 2.20 us | 2.12 us | 6.24 us | 48.30 us |
| p512/b4/d256 | 43.15 us | 2.95 us | 3.11 us | 8.49 us | 57.70 us |
| p2048/b1/d32 | 36.29 us | 2.16 us | 1.96 us | 6.41 us | 46.82 us |

实现方式：

```text
VLLM_BYTE_V2_DECODE_CUTE_STAGE1_EARLY_EXIT=0  # full/default
VLLM_BYTE_V2_DECODE_CUTE_STAGE1_EARLY_EXIT=1  # load/decode K/V 后退出
VLLM_BYTE_V2_DECODE_CUTE_STAGE1_EARLY_EXIT=2  # load/decode + QK 后退出
VLLM_BYTE_V2_DECODE_CUTE_STAGE1_EARLY_EXIT=3  # load/decode + QK + softmax 后退出
VLLM_BYTE_V2_DECODE_CUTE_STAGE1_EARLY_EXIT=4  # full，等价于 0
```

该开关只用于 profile，mode 1-3 不产生有效 attention output；默认值为 0，
不会改变 production 路径。benchmark 参数为：

```text
--cute-stage1-early-exit-mode {0,1,2,3,4}
```

结果文件：

```text
benchmarks/profiles/nsys_cute_early_p512_b4_d128_m{0,1,2,3}.nsys-rep
benchmarks/profiles/nsys_cute_early_p512_b4_d256_m{0,1,2,3}.nsys-rep
benchmarks/profiles/nsys_cute_early_p2048_b1_d32_m{0,1,2,3}.nsys-rep
benchmarks/profiles/cute_stage1_early_exit_report.md
```

结论：

- 当前 CUTE metadata stage1 的最大段是 K/V load/decode，约占 full stage1 的
  `74.8%` 到 `78.3%`。
- QK WMMA 和 softmax 本身都只有约 `2-3 us`，不是当前主瓶颈。
- PV/write 是第二大项，约 `6.2-8.5 us`，但仍显著小于 load/decode。
- 因此下一步优化不应继续优先调 split-K/reduce/output copy，而应先降低
  compressed K/V loader 的指令数和依赖链，或者做 layout-v3 microbench
  验证 payload 物理布局是否能减少 long scoreboard 和全局访问开销。

关注指标：

```text
load/decode K/V 是否仍是最大段
long scoreboard 对应源码行
global sectors/request 是否仍约 2.06-2.09
shared bank conflicts / excessive wavefronts 是否仍高
register/thread 是否压住 occupancy
eligible warps/scheduler 是否上不去
```

### C.5 layout-v3 仅作为 microbench 支线

如果实验 A/C 显示 decode/load 是主因，再做 layout-v3。不要继续投入
`{low0, low1, code}` 的 3B pair-interleaved layout-v2；它引入 `pair_idx * 3`
地址计算和非对齐访问，之前已经退化。

候选 layout-v3：

```text
一个 tile = 128 pair
每 16 pair 一个 stripe

stripe:
  low[32]    // 16 pair * 2 low bytes
  code[16]   // 16 pair * 1 packed code byte

8 stripes * 48B = 384B
```

K/V 可使用不同 physical order：

```text
K: WMMA-B / QK lane order
V: row-major / PV lane order
```

只接 `byte_v2_decode_page_wmma_microbench`，保留门槛：

```text
B2b decode_page_wmma_microbench >= +5%
integer inst 下降
memory inst 下降
long scoreboard stall 下降
global excessive sectors 下降
```

低于该门槛不接 E2E，避免增加格式复杂度。

#### C.5.1 2026-06-13 16-pair stripe layout-v3 结果

已在 `byte_v2_decode_page_wmma_microbench` 中做过一版仅限
microbench 的 16-pair stripe layout-v3：

```text
baseline:
  num_pages=256, num_kv_heads=8, repeat_count=32
  median = 1233.41 us
  per_page_repeat = 0.1506 us

layout-v3 stripe:
  num_pages=256, num_kv_heads=8, repeat_count=32
  median = 1247.23 us
  per_page_repeat = 0.1522 us
  max_abs_diff = 0.0

reverse-order rerun:
  layout-v3 median = 1257.46 us
  baseline median = 1258.50 us
```

结论：

```text
layout-v3 stripe 输出正确，但没有达到 +5% 保留门槛；
首轮还比 baseline 慢约 1.1%，复测基本打平。
```

因此该方案不接入 E2E，也不保留 microbench/ABI 分支。后续如果继续做
payload layout，应换成更激进的 lane-major / 4B aligned ByteV2-Fast
格式，而不是继续在当前 3B 信息量上做 48B stripe 重排。

### C.6 如果 skeleton 本身慢，转向独立 stage1 / FlashInfer-style

如果 R1 已明显慢于 R0，继续微调 decoder 意义有限。下一步选择：

1. 新建独立 CUTE/CUTLASS-style stage1 symbol，例如：

```cpp
byte_v2_paged_decode_attention_gqa4_h128_cute_split_stage1_kernel
```

首版只支持 Llama-3-8B：

```text
num_heads=32
num_kv_heads=8
q_per_kv=4
head_size=head_size_v=128
block_size=16
dtype=bf16
compressed-only
no outlier arena
no block/tile fallback overlay
```

2. 做 FlashInfer-compatible compressed loader / decode staging oracle。

### C.7 cache update fusion 是第二优先级

cache update 仍慢于 raw：

| workload | raw cache update | ByteV2 cache update | delta |
| --- | ---: | ---: | ---: |
| p512/b4/d128 | 10.0 ms | 42.0 ms | +32.0 ms |
| p512/b4/d256 | 19.8 ms | 84.3 ms | +64.5 ms |
| p2048/b1/d32 | 2.6 ms | 9.4 ms | +6.8 ms |

但 p512/b4/d256 的 attention delta 是 `+362 ms`，cache update delta 是
`+64.5 ms`。所以 cache update 是第二瓶颈，不应阻塞 stage1 主线。

低风险 fusion 候选：

```text
init_decode_append_result
+ byte_v2_decode_append_cache_kernel
+ byte_v2_record_deferred_cache_update_error_kernel
```

融合到 `byte_v2_decode_append_cache_kernel` 内部：

```text
初始化 result
执行 append/finalize
出错时 atomicCAS sticky error
```

保留门槛：

```text
cache update total time 下降 >= 20%
p512/b4/d128 E2E >= +0.8% 到 +1.5%
不能破坏 deferred-safe error reporting
不能恢复 per-token host sync
```

#### C.7.1 2026-06-13 deferred error record fusion 结果

尝试过一版 fusion：

```text
deferred_error 路径下：
  旧路径 = init_decode_append_result + decode_append_cache + record_deferred_error
  新路径 = decode_append_cache 内直接 atomicCAS 写 sticky deferred_error

非 deferred 调试/校验路径保持旧 result 逻辑。
```

正确性：

```text
test_byte_v2_native_decode_append_finalizes_partial_raw_fallback_cuda
test_byte_v2_native_batched_decode_append_finalizes_blocks_cuda
test_byte_v2_deferred_cache_update_records_sticky_error

结果：通过。
```

cache update microbench：

```text
decode_append, 128 steps, deferred_error enabled

batch=1:
  old = 29.16 us/step
  fused = 27.88 us/step
  delta = +4.4%

batch=4 with VLLM_BYTE_V2_DECODE_APPEND_BATCH_FASTPATH=1:
  old = 31.20 us/step
  fused = 27.80 us/step
  delta = +10.9%

batch=16 with VLLM_BYTE_V2_DECODE_APPEND_BATCH_FASTPATH=1:
  old = 1603.64 us/step
  fused = 1464.64 us/step
  delta = +8.7%
```

E2E：

```text
Llama-3 8B, p512/b4/d128, num_runs=2, no-arena, CUTE auto

old deferred record path:
  median_output_tps = 125.91
  median_elapsed_s = 4.066

fused deferred record path:
  median_output_tps = 126.35
  median_elapsed_s = 4.052

E2E delta = +0.35%
```

结论：

```text
局部 cache update microbench 有 4%-11% 收益，但 E2E 只有 +0.35%，
低于 +0.8%-1.5% 保留门槛；cache update 总体也没有达到 20% 下降。
```

因此该 fusion 不保留。保留 benchmark 扩展项 `--deferred-error` 和
`--decode-batch-size`，用于后续 cache update 实验复测。

### C.8 暂停优先投入的方向

| 方向 | 原因 |
| --- | --- |
| 继续调 split-K | E2E 不稳定兑现；reduce 不是主瓶颈 |
| persistent partial workspace 默认开启 | E2E 基本噪声级 |
| direct-output decode | 已证明没收益 |
| pair/vec4/base prefetch/base broadcast/packed store | 局部 decoder 微调，未达门槛 |
| K shared stride padding / row-major K + WMMA col-major | 已退化 |
| raw-fallback-only 独立 stage1 | 额外 launch/reduce 抵消收益 |
| outlier block flag / tile bitmap 默认开启 | E2E 没稳定收益 |
| outlier arena 作为性能默认路径 | 当前主要是 capacity/correctness 支线 |

核心原则：

```text
先用 R1/R2/R3 和 decompress+raw-FA oracle 找 stage1 下限。
再决定追 FlashAttention-style skeleton、ByteV2 payload layout，还是 metadata fastpath。
```

### C.9 实验 A 第一轮结果：raw-overlay skeleton lower bound

结果文件：

```text
benchmarks/profiles/lower_bound_nsys_p512_b4_d128_*.nsys-rep
benchmarks/profiles/lower_bound_nsys_p512_b4_d256_*.nsys-rep
benchmarks/profiles/lower_bound_nsys_p2048_b1_d32_*.nsys-rep
benchmarks/profiles/ncu_lower_bound_*_{R1_raw_overlay,R2_compressed_nometa}.csv
benchmarks/profiles/ncu_fourab_*_{base,cute}.csv
benchmarks/profiles/lower_bound_stage1_report.md
```

注意：R1 使用 `--raw-overlay-pages` 近似 raw BF16 KV + ByteV2 skeleton。
这仍包含 generic raw page loader 的分支和地址计算，不是最终专用 raw-BF16
skeleton kernel；但已经足够判断数量级。

#### nsys stage1/reduce

单位是每个 decode op / layer / token 的平均时间。

| workload | R0 raw FA | R1 raw-overlay skeleton | R2 compressed no-meta | R3 compressed metadata | R3 CUTE metadata |
| --- | ---: | ---: | ---: | ---: | ---: |
| p512/b4/d128 | 22.73 us | 73.19 us | 72.45 us | 80.40 us | 52.54 us |
| p512/b4/d256 | 24.84 us | 74.17 us | 73.26 us | 81.31 us | 61.97 us |
| p2048/b1/d32 | 29.10 us | 74.68 us | 74.42 us | 78.37 us | 54.01 us |

相对 raw FA：

| workload | R1 / R0 | R2 / R0 | R3 / R0 | R3 CUTE / R0 |
| --- | ---: | ---: | ---: | ---: |
| p512/b4/d128 | 3.22x | 3.19x | 3.54x | 2.31x |
| p512/b4/d256 | 2.99x | 2.95x | 3.27x | 2.49x |
| p2048/b1/d32 | 2.57x | 2.56x | 2.69x | 1.86x |

#### NCU stage1 指标

| workload | variant | integer inst | memory inst | long scoreboard | global sectors/request | shared bank conflicts | registers/thread | eligible warps |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| p512/b4/d128 | R1 raw-overlay | 168.9M | 23.2M | 40.2% | 2.12 | 0.90M | 50.9 | 0.39 |
| p512/b4/d128 | R2 compressed no-meta | 168.9M | 23.2M | 39.9% | 2.12 | 0.90M | 51.0 | 0.39 |
| p512/b4/d128 | R3 metadata | 179.2M | 25.6M | 38.5% | 2.08 | 0.89M | 50.8 | 0.39 |
| p512/b4/d128 | R3 CUTE metadata | 137.3M | 23.5M | 44.5% | 2.09 | 0.78M | 100.2 | 0.42 |
| p512/b4/d256 | R1 raw-overlay | 187.5M | 25.7M | 39.5% | 2.12 | 1.00M | 50.3 | 0.44 |
| p512/b4/d256 | R2 compressed no-meta | 187.5M | 25.7M | 39.0% | 2.12 | 1.00M | 50.5 | 0.44 |
| p512/b4/d256 | R3 metadata | 199.0M | 28.5M | 37.9% | 2.07 | 0.99M | 50.4 | 0.44 |
| p512/b4/d256 | R3 CUTE metadata | 152.5M | 26.2M | 44.2% | 2.08 | 0.87M | 99.8 | 0.44 |
| p2048/b1/d32 | R1 raw-overlay | 152.3M | 20.8M | 40.7% | 2.12 | 0.80M | 51.0 | 0.35 |
| p2048/b1/d32 | R2 compressed no-meta | 152.3M | 20.8M | 40.2% | 2.12 | 0.80M | 50.8 | 0.35 |
| p2048/b1/d32 | R3 metadata | 161.6M | 23.0M | 38.8% | 2.06 | 0.80M | 51.1 | 0.36 |
| p2048/b1/d32 | R3 CUTE metadata | 123.9M | 21.1M | 45.2% | 2.07 | 0.70M | 100.2 | 0.37 |

#### 结论

1. `R1` 已经比 `R0` 慢 `2.57x-3.22x`，超过 `>1.5x` 门槛。
   这说明主要问题首先是当前 ByteV2 自研 attention skeleton 距离 raw
   FlashAttention/FlashInfer 太远。
2. `R2 - R1` 基本为零，第一轮没有证据表明 compressed payload decode/load
   是最大主因。该结论受 R1 raw-overlay 近似影响，后续可用专用 raw-BF16
   skeleton kernel 复核。
3. `R3 - R2` 约 `4-9 us/op`，metadata/fallback 有成本，但不是 `2.4x-2.8x`
   差距的根因。
4. CUTE metadata 明显降低 stage1，但仍为 raw FA 的 `1.86x-2.49x`。它应继续
   作为 guarded default 候选，但不能替代 stage1 skeleton / FlashInfer-style
   路线。
5. 下一步优先做实验 B：decompress-to-BF16 + raw FlashAttention oracle。如果
   该 oracle 快于当前 fused CUTE stage1，应优先转向 FlashInfer-compatible
   compressed loader 或 staging，而不是继续微调当前 skeleton。

### C.10 实验 B 结果：decompress-to-BF16 + raw FlashInfer oracle

结果文件：

```text
benchmarks/kernels/benchmark_byte_v2_decompress_raw_attention_oracle.py
benchmarks/profiles/experiment_b_decompress_raw_attention_oracle_report.md
benchmarks/profiles/experiment_b_nsys_p512_b4_d128.nsys-rep
benchmarks/profiles/experiment_b_nsys_p512_b4_d256.nsys-rep
benchmarks/profiles/experiment_b_nsys_p2048_b1_d32.nsys-rep
```

本轮新增了一个实验用 native op：

```text
byte_v2_decompress_cache_to_bf16
```

它把 ByteV2 compressed/fallback/outlier page 解压到 NHD BF16 paged KV
workspace，然后调用 FlashInfer：

```text
BatchDecodeWithPagedKVCacheWrapper(kv_layout="NHD", use_tensor_cores=True)
```

单测：

```text
.venv/bin/python -m pytest \
  tests/v1/attention/test_byte_v2_ops.py::test_byte_v2_decompress_cache_to_bf16_op_cuda -q
```

通过。

#### 正确性

三组 workload 的 correctness run 均为：

```text
decompress_key_max_abs_diff = 0.0
decompress_value_max_abs_diff = 0.0
oracle_max_abs_diff = 0.0
```

#### nsys kernel-time 结果

注意：decode-only CUDA event 会把 `byte_v2_paged_decode_attention` 的
custom-op/wrapper 固定开销放大到约 `1 ms`，这和实验 A 的观察一致。因此下面
只采用 nsys kernel summary，不采用 event median 比较 fused ByteV2。

单位是每个 decode op / layer / token：

| workload | ByteV2 decompress | raw FlashInfer | oracle total | current fused ByteV2 | raw FA R0 | oracle / fused | oracle / raw |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| p512/b4/d128 | 45.28 us | 24.10 us | 69.38 us | 52.40 us | 22.73 us | 1.32x | 3.05x |
| p512/b4/d256 | 49.47 us | 26.59 us | 76.06 us | 62.15 us | 24.84 us | 1.22x | 3.06x |
| p2048/b1/d32 | 41.13 us | 22.39 us | 63.52 us | 78.68 us | 29.10 us | 0.81x | 2.18x |

current fused ByteV2 为同一 nsys run 中的 `stage1 + reduce`：

| workload | stage1 | reduce | fused total |
| --- | ---: | ---: | ---: |
| p512/b4/d128 | 48.26 us | 4.14 us | 52.40 us |
| p512/b4/d256 | 57.94 us | 4.21 us | 62.15 us |
| p2048/b1/d32 | 72.00 us | 6.69 us | 78.68 us |

#### 结论

1. raw FlashInfer attention skeleton 很强，oracle 中的 raw attention 本体只有
   `22-27 us/op`，接近实验 A 的 R0 raw FA。
2. 但是 full BF16 workspace materialization 太贵，单独解压已经需要
   `41-49 us/op`，比 raw FlashInfer attention 本身更贵。
3. 对 batch=4，`decompress + FlashInfer` 比当前 fused ByteV2 慢 `22%-32%`，
   不应作为 production 默认路径。
4. 对 batch=1 长上下文，oracle 比当前 fused 快 `19%` 左右，原因是
   `active_head_groups=8` 时 CUTE auto 未启用，当前路径仍走较慢的 WMMA split
   stage1；但 oracle 仍是 raw FA 的 `2.18x`。
5. 这个实验支持的路线不是“两步解压到完整 BF16 KV 再调用 FlashInfer”，而是：

```text
把 FlashInfer/CUTLASS-style attention skeleton 和 ByteV2 compressed tile loader 融合
```

也就是只在 attention kernel 内按 tile 解码需要的数据，直接进入 shared/register /
MMA pipeline，避免写出完整 BF16 KV workspace。

后续优先级调整：

```text
主线：FlashInfer/CUTLASS-style compressed loader 或独立 stage1 skeleton
支线：batch=1 长上下文可以考虑 staging fallback，但不作为默认路径
暂停：继续优化 full-workspace decompress staging，除非需要作为 oracle/调试工具
```

### C.11 p_shared 全矩阵清零移除实验

#### 实验动机

fast stage1 和 CUTE stage1 在每个 16-token tile 的 PV 前都会把
`p_shared[16,16]` 全部清零，然后只写入实际使用的 `q_per_kv=4` 行。
理论上 PV 结果中只读取 q 行，因此未使用的 4..15 行不应影响输出。

本轮尝试：

```text
去掉 fast/CUTE stage1 中每 tile 的 p_shared 全 16x16 清零
保留 q 行 softmax weight 写入
去掉对应的一次 __syncthreads()
```

目标是降低 shared store 和同步开销。

#### 正确性

修改后通过：

```bash
CUDA_VISIBLE_DEVICES=2 .venv/bin/python -m pytest \
  tests/v1/attention/test_byte_v2_decode.py::test_byte_v2_paged_decode_attention_op_cute_stage1_cuda \
  tests/v1/attention/test_byte_v2_decode.py::test_byte_v2_paged_decode_attention_op_fast_stage1_cuda \
  tests/v1/attention/test_byte_v2_decode.py::test_byte_v2_paged_decode_attention_op_cute_stage1_metadata_cuda \
  -q
```

结果：`3 passed`。

#### decode-only 性能

workload：

```text
p512/b4/split_k=16
```

| variant | baseline | remove p_shared zero | delta |
| --- | ---: | ---: | ---: |
| CUTE metadata, tile fallback 3% | 1111.04 us | 1105.92 us | +0.46% |
| fast stage1, fallback 0 | 1099.78 us | 1101.82 us | -0.19% |

#### 结论

该修改正确，但收益低于保留门槛，且 fast no-fallback 路径略退化、p90 变差。
因此代码已回滚，不保留该优化。

这个结果说明当前 stage1 主要瓶颈不在 `p_shared` 清零本身。后续不应继续围绕
单个 shared 初始化语句做局部微调，而应继续按 C.10 结论推进：

```text
FlashInfer/CUTLASS-style attention skeleton + ByteV2 compressed tile loader
```

### C.12 CUTE loader full-row helper 实验

#### 实验动机

`byte_v2_paged_decode_attention_gqa4_h128_fast_split_stage1_kernel` 在满
16 行 page 上会使用：

```text
byte_v2_decode_k_transposed_tile_to_shared_no_fallback_full_rows
byte_v2_decode_v_rowmajor_tile_to_shared_no_fallback_full_rows
```

这两个 helper 去掉了 `row < valid_rows` 判断。相比之下，CUTE stage1 和
CUTE metadata stage1 一直使用带 `valid_rows` 判断的通用 no-fallback helper。
真实 decode 中除尾页外绝大多数 page 都是满 16 行，因此本轮尝试在 CUTE loader
中加入：

```text
if valid_rows == 16:
    compressed tile 使用 full_rows helper
else:
    保持原通用 helper
```

metadata/tile-fallback 路径中，只有 compressed non-fallback tile 使用
full_rows helper；raw tile fallback 和 tail page 仍走原逻辑。

#### 正确性

修改后通过：

```bash
CUDA_VISIBLE_DEVICES=2 .venv/bin/python -m pytest \
  tests/v1/attention/test_byte_v2_decode.py::test_byte_v2_paged_decode_attention_op_cute_stage1_cuda \
  tests/v1/attention/test_byte_v2_decode.py::test_byte_v2_paged_decode_attention_op_cute_stage1_metadata_cuda \
  tests/v1/attention/test_byte_v2_decode.py::test_byte_v2_paged_decode_attention_op_cute_stage1_auto_metadata_cuda \
  -q
```

结果：`3 passed`。

#### decode-only 性能

主测 workload：

```text
p512/b4/split_k=16/fallback=0.03/tile_fallback_pool/CUTE metadata
```

| variant | median | p90 | tok/s |
| --- | ---: | ---: | ---: |
| baseline | 1116.67 us | 1144.83 us | 3582.07 |
| CUTE full-row helper | 1135.62 us | 1208.32 us | 3522.32 |

额外 no-fallback sanity run：

```text
p512/b4/split_k=16/fallback=0/CUTE stage1
median = 1098.75 us
p90 = 1132.54 us
```

#### 结论

该修改正确，但主目标 CUTE metadata 路径退化约 `1.7%`，p90 也明显变差。
原因推测是新增 `valid_rows == 16` runtime 分支和更大的代码路径增加了控制流 /
寄存器 / instruction-cache 压力，抵消甚至超过了 helper 内少掉的 `row <
valid_rows` 判断。

因此该修改已回滚，不保留。

这个结果再次说明：当前 CUTE stage1 的瓶颈不能靠在现有 loader 上继续加局部
分支解决。后续应避免再在同一个 WMMA/CUTE kernel 里叠加 runtime 分支，转向：

```text
独立 symbol 的 compressed hot-path kernel
或真正 FlashInfer/CUTLASS-style skeleton + ByteV2 tile loader
```

### C.13 fast stage1 v2 compact-Q 实验

#### 实验动机

按 P1 的要求，本轮实现了一个独立 opt-in stage1 v2 kernel symbol，用于严格
no-fallback compressed hot path：

```text
VLLM_BYTE_V2_DECODE_FAST_STAGE1_V2=1
```

支持范围与 v1 fast stage1 相同：

```text
Llama-3 8B shape:
  num_heads = 32
  num_kv_heads = 8
  q_per_kv = 4
  head_size = 128
  head_size_v = 128
  block_size = 16
  dtype = bf16

runtime:
  compressed-only page
  fallback = 0
  no tile fallback
  no outlier arena
  split-K
```

v2 的唯一结构性变化是 compact Q shared load：

```text
v1:
  q_shared[16, 128] 全部写入，只有前 q_per_kv=4 行来自 query，
  其余 12 行写 0

v2:
  只写 q_shared[0:4, 128]
  q_shared[4:16, :] 不初始化
```

这个假设基于 WMMA QK 的行独立性：后续只读取 q0-q3 对应的 score/PV 输出，
未使用的 q4-q15 行不应影响 q0-q3。

#### 正确性

修改后将 fast stage1 测试参数化为 v1/v2 两个 env，包含 tail page：

```bash
CUDA_VISIBLE_DEVICES=2 .venv/bin/python -m pytest \
  tests/v1/attention/test_byte_v2_decode.py::test_byte_v2_paged_decode_attention_op_fast_stage1_cuda \
  -q
```

结果：`2 passed`。

#### decode-only 性能

重构后在同一张 GPU 上重新跑 v1/v2 A/B：

| workload | v1 median | v1 p90 | v2 median | v2 p90 | delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| p512/b4/split16/fallback0 | 1097.73 us | 1121.28 us | 1116.67 us | 1220.61 us | -1.73% |
| p2048/b1/split64/fallback0 | 1103.87 us | 1133.57 us | 1123.33 us | 1238.02 us | -1.76% |

#### 结论

compact-Q v2 正确，但两个目标 workload 都退化约 `1.7%-1.8%`，且 p90 明显
变差。因此该 v2 kernel/env/benchmark/test 接入已全部回滚，不保留。

这个结果说明：当前 fast stage1 的 Q shared 初始化不是主要瓶颈；减少这部分
写入反而可能导致更差的 shared/WMMA 行为或编译器调度。后续不要继续围绕
`q_shared` / `p_shared` 未使用行做局部削减。

下一步应从更大的结构切入：

```text
1. 做真正独立的 FlashInfer/CUTLASS-style stage1 skeleton，
   重新设计 K/V loader、shared swizzle、MMA atom、softmax/PV pipeline。

2. 或只在 byte_v2_decode_page_wmma_microbench 中做 payload layout 实验，
   比如 warp/lane-major 4B group，先证明 K/V load-decode 段能明显下降。
```

### C.14 Fast4B aligned-pair payload microbench

#### 实验动机

为了验证“牺牲部分压缩率换更低 decode latency”是否可行，本轮只在
`byte_v2_decode_page_wmma_microbench` 中实现了一个 microbench-only layout：

```text
variant 0:
  production ByteV2 tile layout
  base + fallback + low[256] + code[128]

variant 1:
  Fast4B aligned-pair layout
  tile header 4B
  pair[pair_idx] = {low0, low1, packed_code, pad}
```

Fast4B 的目的不是作为 production 格式，而是测试：

```text
一次 aligned 32-bit load 取代分离的 low/code load
```

能否明显降低 K/V decode + WMMA microbench 时间。

#### 正确性

修改后将 decode-page WMMA microbench correctness 参数化为 variant 0/1：

```bash
CUDA_VISIBLE_DEVICES=2 .venv/bin/python -m pytest \
  tests/v1/attention/test_byte_v2_wmma_microbench.py::test_byte_v2_decode_page_wmma_microbench_matches_reference_cuda \
  -q
```

结果：`2 passed`。

#### 性能结果

workload A：

```text
num_pages=256
num_kv_heads=1
repeat_count=64
```

| variant | page bytes | bytes/raw | median | p90 | delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| production layout | 6192 | 0.7559x | 1513.47 us | 1544.19 us | baseline |
| Fast4B | 8272 | 1.0098x | 1529.86 us | 1547.26 us | -1.08% |

workload B：

```text
num_pages=128
num_kv_heads=8
repeat_count=32
```

| variant | page bytes | bytes/raw | median | p90 | delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| production layout | 49424 | 0.7542x | 1097.22 us | 1105.92 us | baseline |
| Fast4B | 66064 | 1.0081x | 1160.19 us | 1240.06 us | -5.74% |

#### 结论

Fast4B aligned-pair layout 正确，但性能退化，且 page bytes 已经略大于 raw BF16
KV，完全不满足：

```text
decode_page_wmma_microbench >= +5%
page bytes 仍至少比 raw 小 15%-20%
```

因此该 microbench variant/ABI/benchmark/test 接入已全部回滚，不保留。

这个结果说明：简单把 pair pad 到 4B 并用 aligned 32-bit load，不能解决当前
load/decode 瓶颈；额外 HBM bytes 和更差的 cache behavior 会抵消甚至超过
指令侧收益。

后续 layout 实验不应再做 4B-per-pair。若继续走 payload-layout 方向，应保持
接近当前 `~0.75x raw` 的 bytes ratio，例如：

```text
1. 仍用 3B/pair，但做 warp/lane-major stripe，减少地址发散。
2. 做 group-of-16-pair 的低字节/code stripe，但避免之前 layout-v3 的额外地址计算。
3. 先用 NCU 对比 global sectors、integer/memory inst、long scoreboard；
   microbench 没有 >=5% 收益就不接 stage1/E2E。
```

### C.15 FlashInfer-style warp-per-query stage1 v2

#### 实验动机

C.9/C.10 的 lower-bound/oracle 结果表明，当前差距主要不是单独的 ByteV2
payload bytes，而是自研 split-stage1 skeleton 和 raw FlashInfer/FlashAttention
decode 之间的结构差距。CUTE metadata 虽然降低了 stage1，但仍保留了：

```text
decode K/V -> shared
WMMA QK -> scores shared
softmax -> p_shared
WMMA PV -> pv_shared
partial write
```

其中 `scores/p_shared/pv_shared` 多次 shared 往返和只有一个 warp 做 WMMA 的结构，
与 FlashInfer decode 的 warp-level online softmax/PV 风格仍有明显差距。

本轮新增一个独立 opt-in kernel：

```cpp
byte_v2_paged_decode_attention_gqa4_h128_flash_split_stage1_kernel
```

运行条件严格限制为：

```text
Llama-3 8B shape:
  num_heads=32
  num_kv_heads=8
  q_per_kv=4
  head_size=head_size_v=128
  block_size=16

cache:
  compressed-only page size
  no sparse fallback pool
  no tile fallback
  no outlier arena/block flags/tile bitmap

runtime:
  split-K decode
  page fastpath enabled
  VLLM_BYTE_V2_DECODE_FLASH_STAGE1=1
```

#### Kernel 结构

该 v2 kernel 仍复用现有 ByteV2 compressed tile decoder，把 K/V 每个 physical
block 解压到 shared 一次，但 attention skeleton 改为 FlashInfer-style：

```text
1 CTA = 1 request x 1 KV head x 1 split
4 warps = 4 query heads under the same KV head

每个 q-head 一个 warp:
  lane 0..15 对应 16 个 KV row
  lane 16..31 和 lane 0..15 配对完成 head_dim=128 的 QK dot
  warp-level max/sum 做 online softmax
  每个 lane 维护 4 个 output dim accumulator
  PV 直接从 v_shared 读出并在寄存器累加
```

相对 CUTE/fast v1，v2 去掉：

```text
scores shared store/load
p_shared softmax matrix
PV WMMA fragment
pv_shared store/load
多次中间 __syncthreads()
```

同时保留：

```text
K/V compressed page decode once per block
split-K partial output + existing reduce kernel
```

注意：warp reduce helper 只有 lane0 拿到完整归约值，因此 v2 kernel 在
online softmax 中显式把 `tile_max` 和 `tile_denom` 从 lane0 broadcast 回整个
warp，避免不同 lane 使用不同 softmax 基准。

#### 代码改动

```text
csrc/libtorch_stable/cache_kernels.cu
  - 新增 byte_v2_wmma_to_float()
  - 新增 byte_v2_paged_decode_attention_gqa4_h128_flash_split_stage1_kernel
  - native decode launcher 新增 VLLM_BYTE_V2_DECODE_FLASH_STAGE1 分支

vllm/envs.py
  - 新增 VLLM_BYTE_V2_DECODE_FLASH_STAGE1

benchmarks/kernels/benchmark_byte_v2_decode_kernel.py
  - 新增 --flash-stage1

benchmarks/benchmark_byte_v2_decode_e2e.py
  - 输出环境中记录 VLLM_BYTE_V2_DECODE_FLASH_STAGE1

tests/v1/attention/test_byte_v2_decode.py
  - 新增 test_byte_v2_paged_decode_attention_op_flash_stage1_cuda
```

#### 正确性

```bash
CUDA_VISIBLE_DEVICES=2 .venv/bin/python -m pytest \
  tests/v1/attention/test_byte_v2_decode.py::test_byte_v2_paged_decode_attention_op_flash_stage1_cuda \
  -q
```

结果：

```text
1 passed
```

回归：

```bash
CUDA_VISIBLE_DEVICES=2 .venv/bin/python -m pytest \
  tests/v1/attention/test_byte_v2_decode.py::test_byte_v2_paged_decode_attention_op_gqa_wmma_split_k_cuda \
  tests/v1/attention/test_byte_v2_decode.py::test_byte_v2_paged_decode_attention_op_cute_stage1_cuda \
  tests/v1/attention/test_byte_v2_decode.py::test_byte_v2_paged_decode_attention_op_fast_stage1_cuda \
  tests/v1/attention/test_byte_v2_decode.py::test_byte_v2_paged_decode_attention_op_flash_stage1_cuda \
  -q
```

结果：

```text
4 passed
```

#### decode-only event 结果

文件：

```text
benchmarks/profiles/bytev2_flash_stage1_p512_b4_{base,cute,fast,flash}.json
benchmarks/profiles/bytev2_flash_stage1_p2048_b1_{base,cute,fast,flash}.json
```

p512/b4/split8/fallback0：

| variant | median | p90 | tok/s | max_abs_diff |
| --- | ---: | ---: | ---: | ---: |
| page fastpath baseline | 1138.18 us | 1229.82 us | 3514.40 | 0.0 |
| CUTE stage1 | 1104.90 us | 1120.26 us | 3620.25 | 0.0 |
| fast stage1 v1 | 1100.80 us | 1128.45 us | 3633.72 | 0.0 |
| flash stage1 v2 | 1100.80 us | 1208.32 us | 3633.72 | 0.0 |

p2048/b1/split64/fallback0：

| variant | median | p90 | tok/s | max_abs_diff |
| --- | ---: | ---: | ---: | ---: |
| page fastpath baseline | 1141.76 us | 1156.10 us | 875.84 | 0.0 |
| CUTE stage1 | 1102.85 us | 1117.18 us | 906.74 | 0.0 |
| fast stage1 v1 | 1100.80 us | 1125.38 us | 908.43 | 0.0 |
| flash stage1 v2 | 1094.66 us | 1171.46 us | 913.53 | 0.0 |

event median 中，v2 相对 fast v1：

```text
p512/b4: 基本打平
p2048/b1: +0.56%
```

该 benchmark 的 event 总时间包含 cache update/setup，其中
`byte_v2_compress_touched_blocks_kernel` 单次约 `11.2 ms`，会显著稀释 stage1
差异。因此需要看 nsys kernel summary。

#### nsys stage1 kernel summary

文件：

```text
benchmarks/profiles/nsys_flash_stage1_p512_b4_{fast,flash}.nsys-rep
benchmarks/profiles/nsys_flash_stage1_p2048_b1_{fast,flash}.nsys-rep
```

p512/b4/split8/fallback0：

| variant | stage1 total | launches | avg/launch | median/launch | delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| fast stage1 v1 | 1.215 ms | 25 | 48.60 us | 48.51 us | baseline |
| flash stage1 v2 | 1.085 ms | 25 | 43.42 us | 43.39 us | +10.7% |

p2048/b1/split64/fallback0：

| variant | stage1 total | launches | avg/launch | median/launch | delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| fast stage1 v1 | 1.238 ms | 25 | 49.51 us | 49.92 us | baseline |
| flash stage1 v2 | 0.792 ms | 25 | 31.67 us | 31.81 us | +36.0% |

reduce kernel 基本不变：

```text
p512/b4: ~3.06 us/launch
p2048/b1: ~6.83 us/launch
```

#### 结论

1. v2 的 stage1 本体收益明确，尤其在 p2048/b1 长上下文低 batch 场景下，
   warp-per-query online softmax/PV skeleton 明显优于 fast v1 的 WMMA
   `scores/p_shared/pv_shared` 路径。
2. p512/b4 中 stage1 本体也有约 `10.7%` 改善，但 decode-only event 基本打平，
   因为 benchmark 总时间被 cache update/setup 和 wrapper 固定成本稀释。
3. 该 kernel 当前只支持 compressed-only/no-metadata，因此真实 E2E 默认路径如果
   分配 sparse fallback pool 或 outlier arena，不会自动启用 v2。
4. 保留该 kernel 作为 opt-in 实验路径是合理的，因为它证明了“真正
   FlashInfer-style attention skeleton”可以显著降低 stage1 本体。

#### 下一步

优先把 v2 扩展到真实 no-arena production metadata：

```text
compressed page + tile fallback
raw fallback page
outlier arena overlay 可后置
```

具体做法是复用 CUTE metadata 的 K/V loader 分支，继续使用 v2 的 warp-per-query
online softmax/PV skeleton。只有 metadata 版本也能保持 stage1 收益后，再接：

```text
VLLM_BYTE_V2_DECODE_FLASH_STAGE1_AUTO
E2E p512/b4/d128,d256
E2E p2048/b1/d32,d128
```

### C.16 FlashInfer-style stage1 v2 metadata 接入实验

#### 目标

C.15 的 v2 skeleton 只支持 compressed-only/no-metadata，因此真实 no-arena
production 路径一旦启用 sparse fallback pool 或 tile fallback metadata，就会回到
CUTE metadata/旧路径。本轮目标是把 v2 skeleton 接到 CUTE metadata loader 语义上：

```text
compressed page + no-fallback tile fastpath
compressed page + tile fallback metadata
raw fallback page metadata
no outlier arena
```

outlier arena overlay 暂不接入 v2 metadata 路径。原因是 arena 当前主要是
capacity/correctness 支线，之前 E2E 已证明它不是性能默认路径；本轮只验证真实
no-arena production 的 fallback/tile metadata。

#### 实现

核心改动：

```text
csrc/libtorch_stable/cache_kernels.cu
  byte_v2_paged_decode_attention_gqa4_h128_flash_split_stage1_kernel
    template <..., bool metadata_fastpath>
```

`metadata_fastpath=false` 保留 C.15 的 no-metadata hot path。
`metadata_fastpath=true` 新增：

1. compressed page + tile fastpath：
   - tile fallback flag 为 0 时使用 no-fallback decode；
   - tile fallback flag 非 0 时通过 `fallback_tile_ids` 从 `fallback_pool`
     读取 raw tile。
2. compressed page + 非 tile-fastpath：
   - 走通用 `byte_v2_load_compressed_bits(...)` metadata helper。
3. raw fallback page：
   - 通过 `fallback_block_ids[physical_block]` 找 raw block slot；
   - 从 `fallback_pool + slot * raw_block_bytes` 读取 raw K/V。

launcher 条件从“必须没有 fallback metadata”放宽为：

```text
VLLM_BYTE_V2_DECODE_FLASH_STAGE1=1
and Llama-3-8B fixed shape
and compressed page size matches
and page_fastpath enabled
and no outlier arena/block flags/tile bitmap
```

如果存在 sparse fallback 或 tile fallback，则实例化
`metadata_fastpath=true`；否则实例化 `metadata_fastpath=false`。

#### correctness

新增测试：

```text
tests/v1/attention/test_byte_v2_decode.py
  test_byte_v2_paged_decode_attention_op_flash_stage1_raw_fallback_cuda
  test_byte_v2_paged_decode_attention_op_flash_stage1_tile_fallback_cuda
```

回归命令：

```bash
CUDA_VISIBLE_DEVICES=2 .venv/bin/python -m pytest \
  tests/v1/attention/test_byte_v2_decode.py::test_byte_v2_paged_decode_attention_op_flash_stage1_cuda \
  tests/v1/attention/test_byte_v2_decode.py::test_byte_v2_paged_decode_attention_op_flash_stage1_raw_fallback_cuda \
  tests/v1/attention/test_byte_v2_decode.py::test_byte_v2_paged_decode_attention_op_flash_stage1_tile_fallback_cuda \
  -q
```

结果：

```text
3 passed, 16 warnings in 12.65s
```

metadata 回归：

```bash
CUDA_VISIBLE_DEVICES=2 .venv/bin/python -m pytest \
  tests/v1/attention/test_byte_v2_decode.py::test_byte_v2_paged_decode_attention_op_cute_stage1_metadata_cuda \
  tests/v1/attention/test_byte_v2_decode.py::test_byte_v2_paged_decode_attention_op_cute_stage1_auto_metadata_cuda \
  tests/v1/attention/test_byte_v2_decode.py::test_byte_v2_paged_decode_attention_op_flash_stage1_raw_fallback_cuda \
  tests/v1/attention/test_byte_v2_decode.py::test_byte_v2_paged_decode_attention_op_flash_stage1_tile_fallback_cuda \
  -q
```

结果：

```text
4 passed, 16 warnings in 17.20s
```

#### decode-only event 结果

文件：

```text
benchmarks/profiles/bytev2_flash_metadata_p512_b4_block_{cute,flash}.json
benchmarks/profiles/bytev2_flash_metadata_p512_b4_tile_{cute,flash}.json
benchmarks/profiles/bytev2_flash_metadata_p2048_b1_block_{cute,flash}.json
```

p512/b4/split8/fallback0.03/block fallback：

| variant | median | p90 | tok/s | fallback blocks | max_abs_diff |
| --- | ---: | ---: | ---: | ---: | ---: |
| CUTE metadata | 1112.06 us | 1131.52 us | 3596.92 | 4/8 | 0.0078125 |
| flash v2 metadata | 1101.31 us | 1124.35 us | 3632.03 | 4/8 | 0.0 |

p512/b4/split8/fallback0.03/tile fallback：

| variant | median | p90 | tok/s | fallback tiles | max_abs_diff |
| --- | ---: | ---: | ---: | ---: | ---: |
| CUTE metadata | 1136.13 us | 1225.73 us | 3520.73 | 256/1024 | 0.0078125 |
| flash v2 metadata | 1119.23 us | 1261.57 us | 3573.88 | 256/1024 | 0.0 |

p2048/b1/split64/fallback0.03/block fallback：

| variant | median | p90 | tok/s | fallback blocks | max_abs_diff |
| --- | ---: | ---: | ---: | ---: | ---: |
| CUTE metadata | 1111.04 us | 1132.54 us | 900.06 | 4/8 | 0.0 |
| flash v2 metadata | 1124.35 us | 1232.90 us | 889.40 | 4/8 | 0.0 |

结论：

1. v2 metadata correctness 成立。
2. 在 synthetic metadata decode-only 中，p512/b4 有 `~1.0%-1.5%` 小收益，
   p2048/b1/block fallback 退化 `~1.2%`。
3. 这和 C.15 的 no-metadata nsys 结论不同：metadata loader 接入后，v2
   skeleton 的 stage1 本体优势没有稳定转化到 event latency。

#### E2E 结果

workload：

```text
model=/mnt/sda1/yxz/byte_v2/Meta-Llama-3-8B-Instruct
prompt_len=512
batch_size=4
decode_lens=128,256
enforce_eager
VLLM_BYTE_V2_ENABLE_SPARSE_FALLBACK_POOL=1
VLLM_BYTE_V2_SPARSE_FALLBACK_POOL_RATIO=0.03
VLLM_BYTE_V2_ENABLE_OUTLIER_ARENA=0
VLLM_BYTE_V2_OUTLIER_MAX_PER_TILE=0
VLLM_BYTE_V2_DECODE_PAGE_FASTPATH=1
VLLM_BYTE_V2_DECODE_TILE_FASTPATH=1
VLLM_BYTE_V2_DECODE_SPLIT_K=8
```

文件：

```text
benchmarks/profiles/bytev2_e2e_flash_metadata_p512_b4_d128_256.json
benchmarks/profiles/bytev2_e2e_cute_metadata_p512_b4_d128_256.json
```

| mode | stage1 path | d128 tok/s | d128 elapsed | d256 tok/s | d256 elapsed |
| --- | --- | ---: | ---: | ---: | ---: |
| raw vLLM | raw FA, eager | 130.06 | 3.937 s | 130.34 | 7.857 s |
| ByteV2 no-arena | flash v2 metadata | 18.07 | 28.341 s | 18.15 | 56.423 s |
| ByteV2 no-arena | CUTE metadata auto | 18.10 | 28.290 s | 18.15 | 56.408 s |

fallback stats：

| run | exhausted | block fallbacks | tile fallback assigned sum | max tile_next_slot |
| --- | --- | ---: | ---: | ---: |
| flash v2 metadata | false | 0 | 12350 | 2765 |
| CUTE metadata auto | false | 0 | 11629 | 2276 |

结论：

1. 真实 no-arena production E2E 跑通，没有 fallback pool exhaustion。
2. 实际 fallback 主要是 tile-level fallback，而不是 block raw fallback；
   `total_assigned_blocks=0`，但每层有大量 `assigned_tiles`。
3. flash v2 metadata 与 CUTE metadata auto 的 E2E 基本打平，甚至略慢/略快都在
   单次运行噪声内。
4. 因此本轮实现应保留为显式实验开关
   `VLLM_BYTE_V2_DECODE_FLASH_STAGE1=1`，但不应升级为 production default 或
   `AUTO` heuristic。

#### 后续判断

v2 skeleton 的 no-metadata stage1 本体曾有明确收益，但真实 metadata 后没有
兑现，说明剩余瓶颈大概率不在 online softmax/PV skeleton 本身，而在：

```text
metadata/tile fallback loader 的指令和访存依赖
真实 tile fallback 比例与 tile slot 访问模式
cache update / fallback tile encoding 成本
wrapper/reduce/setup 固定成本
```

下一步不建议继续围绕当前 v2 metadata skeleton 做默认化。更高价值的实验是：

1. 用 nsys/ncu 对真实 no-arena E2E 拆出 stage1、reduce、cache update、
   tile fallback encode 的时间占比。
2. 对 production tile fallback 做 loader 专项优化，而不是只优化 no-fallback
   compressed tile skeleton。
3. 若 profile 证明 tile fallback loader 主导，则优先做 tile fallback compact
   layout/slot locality，而不是继续调 split-K 或 v2 skeleton。

### C.17 单进程 nsys profile：当前瓶颈分成两类

#### 实验目的

C.16 的 E2E 只能说明 flash v2 metadata 没有稳定兑现收益，但不能说明时间花在
哪里。本轮用 Nsight Systems 重新拆分真实 no-arena production 路径：

```text
raw
ByteV2 no-arena + CUTE metadata auto
ByteV2 no-arena + flash v2 metadata
```

workload：

```text
p512/b4/d128
p2048/b1/d32
```

重要设置：

```text
VLLM_ENABLE_V1_MULTIPROCESSING=0
enforce_eager
VLLM_BYTE_V2_ENABLE_SPARSE_FALLBACK_POOL=1
VLLM_BYTE_V2_SPARSE_FALLBACK_POOL_RATIO=0.03
VLLM_BYTE_V2_ENABLE_OUTLIER_ARENA=0
VLLM_BYTE_V2_OUTLIER_MAX_PER_TILE=0
VLLM_BYTE_V2_DECODE_PAGE_FASTPATH=1
VLLM_BYTE_V2_DECODE_TILE_FASTPATH=1
VLLM_BYTE_V2_DECODE_SPLIT_K=8
VLLM_BYTE_V2_DECODE_PARALLEL_REDUCE=1
```

这里必须禁用 V1 multiprocessing。原因是 parent benchmark 里的
`byte_v2_bench_measured` NVTX range 不会覆盖 EngineCore 子进程里的 kernel；
用 multiprocess profile 时，`nsys stats --filter-nvtx byte_v2_bench_measured`
会漏掉真实 kernel。单进程模式下 NVTX 过滤和 kernel 统计才是可信的。

#### 产物

```text
benchmarks/profiles/nsys_next_sp_raw_p512_b4_d128.nsys-rep
benchmarks/profiles/nsys_next_sp_bytev2_cute_p512_b4_d128.nsys-rep
benchmarks/profiles/nsys_next_sp_bytev2_flash_p512_b4_d128.nsys-rep
benchmarks/profiles/nsys_next_sp_raw_p2048_b1_d32.nsys-rep
benchmarks/profiles/nsys_next_sp_bytev2_cute_p2048_b1_d32.nsys-rep
benchmarks/profiles/nsys_next_sp_bytev2_flash_p2048_b1_d32.nsys-rep

benchmarks/profiles/*_cuda_gpu_kern_sum_nvtx=byte_v2_bench_measured_base.csv
benchmarks/profiles/bytev2_next_sp_e2e_cute_p2048_b1_d32.json
benchmarks/profiles/bytev2_next_sp_e2e_flash_p2048_b1_d32.json
```

#### p512/b4/d128：cache update 是第一瓶颈

E2E 单次结果：

| mode | tok/s | elapsed |
| --- | ---: | ---: |
| raw | 130.45 | 3.925 s |
| ByteV2 CUTE metadata | 18.43 | 27.776 s |
| ByteV2 flash v2 metadata | 18.47 | 27.725 s |

按 kernel 分类：

| category | raw | ByteV2 CUTE | ByteV2 flash |
| --- | ---: | ---: | ---: |
| total measured GPU kernels | 3873.7 ms | 21848.1 ms | 21738.9 ms |
| model GEMM/GEMV | 3677.5 ms | 3685.3 ms | 3685.0 ms |
| raw decode attention | 93.4 ms | - | - |
| raw cache update | 12.2 ms | - | - |
| ByteV2 stage1 | - | 292.4 ms | 249.2 ms |
| ByteV2 reduce | - | 11.7 ms | 11.7 ms |
| ByteV2 cache compress | - | 17748.8 ms | 17682.6 ms |
| ByteV2 cache bookkeeping | - | 25.2 ms | 25.1 ms |
| ByteV2 prefill encode tail | - | 1.6 ms | 1.6 ms |

关键观察：

1. `byte_v2_compress_touched_blocks_kernel` 在 p512/b4/d128 下启动 4064 次，
   总时间约 `17.7 s`，平均约 `4.35 ms/launch`。
2. 4064 次基本等于 `128 decode tokens * 32 layers - tail`，说明当前 decode
   cache update 仍然近似每 token/layer 触发一次重压缩 touched block。
3. flash v2 metadata 把 stage1 从 `292.4 ms` 降到 `249.2 ms`，stage1 本身
   有 `~14.8%` 收益；但它只节省 `43 ms`，完全被 `17.7 s` 的 cache compress
   吞没，所以 E2E 看不到收益。
4. p512/b4/d128 的第一瓶颈不是 stage1，而是 decode cache update/encode。

因此，这个 workload 下继续调 stage1 skeleton、split-K、metadata bitmap 都不会
改变 E2E。必须先把 decode cache update 从“每步重压缩 touched block”改成
增量 append 或 block-close encode。

#### p2048/b1/d32：stage1 差距更清楚

E2E 单次结果：

| mode | tok/s | elapsed | block fallback | tile fallback sum | exhausted |
| --- | ---: | ---: | ---: | ---: | --- |
| raw | 34.31 | 0.933 s | 0 | 0 | false |
| ByteV2 CUTE metadata | 13.50 | 2.371 s | 64 | 10195 | false |
| ByteV2 flash v2 metadata | 13.72 | 2.332 s | 64 | 10248 | false |

按 kernel 分类：

| category | raw | ByteV2 CUTE | ByteV2 flash |
| --- | ---: | ---: | ---: |
| total measured GPU kernels | 919.2 ms | 970.1 ms | 987.4 ms |
| model GEMM/GEMV | 867.0 ms | 868.3 ms | 868.7 ms |
| raw decode attention | 31.8 ms | - | - |
| raw cache update | 2.8 ms | - | - |
| ByteV2 stage1 | - | 64.5 ms | 81.6 ms |
| ByteV2 reduce | - | 8.0 ms | 7.7 ms |
| ByteV2 decode append cache | - | 7.3 ms | 7.3 ms |
| ByteV2 prefill encode tail | - | 1.6 ms | 1.6 ms |

关键观察：

1. p2048/b1/d32 没有 p512 那种 `compress_touched_blocks` 巨额成本，因为 decode
   token 少、batch 小，路径主要体现 attention stage1 差距。
2. CUTE metadata 的 `stage1 + reduce` 约 `72.5 ms`，raw decode attention
   约 `31.8 ms`，仍慢 `~2.3x`。
3. flash v2 metadata 在 Nsight kernel 时间上比 CUTE 更慢：
   `stage1 81.6 ms` vs `64.5 ms`。E2E 里 flash 略高于 CUTE 的 `13.72` vs
   `13.50 tok/s` 不能作为默认化依据，单次 E2E 噪声和调度差异足以解释。
4. 这个 workload 说明 stage1 仍然是长期必须解决的问题，但不是所有场景的第一
   瓶颈。

#### 本轮结论

当前 ByteV2 的瓶颈不能再用一个单一答案概括：

| 场景 | 第一瓶颈 | 说明 |
| --- | --- | --- |
| p512/b4/d128 | decode cache update/encode | 每 token/layer 触发 `compress_touched_blocks`，总计 `17.7 s` |
| p2048/b1/d32 | decode attention stage1 | `stage1 + reduce` 仍约 raw attention 的 `2.3x` |

更重要的是，p512 的结果说明：即使 stage1 继续优化 `10%-20%`，E2E 也几乎不动。
下一轮必须先做 cache update 结构性修复，然后再回到 stage1。

#### 下一步设计：decode cache update 增量化

目标是移除 p512/b4/d128 下的 `byte_v2_compress_touched_blocks_kernel`
主路径，避免每 token/layer 重压缩 touched block。

建议实现一个新的 decode append 方案：

1. prefill 阶段继续一次性 direct encode 完整 16-token block。
2. decode 阶段对当前未满 block 不立即重压缩整块。
3. 为每个序列/layer/head 保留 open-block 状态：
   `physical_block_id`、`filled_tokens`、必要的 raw tail K/V 或 partial metadata。
4. 当 block 未满时，attention 对 closed compressed blocks 走 ByteV2 compressed
   path；最后一个 open block 走 raw-tail overlay 或小型 raw append path。
5. 当 block 填满 16 token 时，只对这个 block 做一次 block-close encode，
   同时生成 tile fallback metadata。
6. fallback/tile metadata 只在 block-close encode 时稳定写入，避免每 token 重扫
   整个 16-token block。

这个方案的预期收益：

```text
p512/b4/d128:
  当前 compress launches: ~4064
  block-close encode launches: 约 ceil(128/16) * 32 = 256
  理论 launch 数下降约 16x
```

如果 raw-tail overlay 只覆盖最多 15 个 token，它对 attention 读带宽影响很小，
但能避免 open block 每步重压缩。

#### 验收门槛

先不要改 stage1 default。下一轮 cache update patch 的保留门槛：

```text
p512/b4/d128:
  byte_v2_compress_touched_blocks_kernel total time 下降 >= 8x
  ByteV2 E2E tok/s 至少从 ~18 tok/s 提升到 >= 40 tok/s
  correctness: byte_v2 vs raw decode test 通过
  prefix cache: ByteV2 prefix cache smoke test 通过
  fallback pool: any_exhausted=false

p2048/b1/d32:
  不允许明显退化
  stage1 路径保持 CUTE metadata auto 默认，不启用 flash v2 metadata 默认
```

如果增量 append 需要临时 raw tail storage，它只允许保存当前 open block 的尾部
token，不允许退回到长期 raw KV cache；否则会破坏 ByteV2 常驻压缩 cache 的目标。

### C.18 decode append 默认 fast path 与 host device-property 缓存

#### 背景

C.17 的 p512/b4/d128 profile 把 `byte_v2_compress_touched_blocks_kernel`
定位为第一瓶颈，但后续复查发现有两个问题需要先修正：

1. `benchmark_byte_v2_decode_e2e.py --child-mode byte_v2_compressed_only`
   没有在 child 进程里应用 `_mode_env()`，因此单独 nsys child-mode profile
   实际可能没有设置 `VLLM_BYTE_V2_COMPRESSED_ONLY_CACHE=1`，会误走 overlay/generic
   full-page path。
2. 在有效 compressed-only profile 中，decode append fast path 生效后，GPU 侧
   cache update 已经降到几十毫秒级，但 API 表显示 attention host wrapper 每层每
   token 调用一次 `cudaGetDeviceProperties`，p512/b4/d128 下共 4096 次，总计约
   `4.5 s`。

#### 本轮改动

1. 修复 benchmark child-mode：
   `run_child()` 开头执行 `os.environ.update(_mode_env(mode))`，保证直接 profile
   child-mode 与 parent-mode 的环境一致。
2. 增加 side-effect-free 的
   `byte_v2_validate_decode_append_slots_kernel`，默认在 eager/profile 的
   multi-token decode update 前验证：
   - slot 合法；
   - active token 不重复写同一个 physical block；
   - page status/valid rows 合法；
   - 只允许 append 到空 block offset 0，或 append 到当前 `valid_rows` 位置；
   - finalized block、overwrite、same-block continuation 自动回退 generic path。
3. 默认启用 safe batched decode append fast path：
   - 单 token、CUDA graph capture、显式
     `VLLM_BYTE_V2_DECODE_APPEND_BATCH_FASTPATH=1` 维持原逻辑；
   - eager multi-token 在 validation 通过后走
     `byte_v2_decode_append_cache_kernel`；
   - validation 失败时回退 generic touched-block path，不写坏 cache。
4. 增加 `byte_v2_device_supports_bf16_wmma()` host helper，用 `std::once_flag`
   按 device 缓存 BF16 WMMA capability，替换 attention wrapper 和 WMMA
   microbench 中每次 launch 前的 `cudaGetDeviceProperties`。

#### 正确性验证

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

```bash
CUDA_VISIBLE_DEVICES=2 .venv/bin/python -m pytest \
  tests/v1/attention/test_byte_v2_decode.py \
  tests/v1/attention/test_byte_v2_backend.py::test_byte_v2_deferred_cache_update_records_sticky_error \
  -q
```

结果：`27 passed`。

native rebuild：

```bash
uv pip install -e . --torch-backend=auto
```

结果：成功，用时约 `22m46s`。

#### E2E 结果

配置：

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
.venv/bin/python benchmarks/benchmark_byte_v2_decode_e2e.py \
  --modes raw,byte_v2_compressed_only \
  --model /mnt/sda1/yxz/byte_v2/Meta-Llama-3-8B-Instruct \
  --prompt-len 512 \
  --decode-lens 128 \
  --batch-size 4 \
  --num-runs 1 \
  --warmup-decode-len 8 \
  --gpu-memory-utilization 0.80 \
  --enforce-eager \
  --output-json benchmarks/profiles/bytev2_cached_props_p512_b4_d128_e2e.json
```

| mode | tok/s | elapsed |
| --- | ---: | ---: |
| raw | 131.15 | 3.904 s |
| ByteV2 compressed-only | 102.66 | 4.988 s |

对比本轮中间 profile：

| 版本 | ByteV2 tok/s | 说明 |
| --- | ---: | --- |
| child-mode 未修正/overlay-generic profile | 50.37 | 未设置 `COMPRESSED_ONLY_CACHE=1`，不作为有效 compressed-only 结论 |
| safe append + device-property cache | 102.66 | 有效 compressed-only no-arena CUTE metadata 路径 |

ByteV2 当前约为 raw 的 `78.3%`。

#### nsys 结果

有效 compressed-only nsys：

```bash
nsys profile --trace=cuda,nvtx,cublas \
  --output benchmarks/profiles/nsys_cached_props_bytev2_cute_p512_b4_d128 \
  .venv/bin/python benchmarks/benchmark_byte_v2_decode_e2e.py \
    --child-mode byte_v2_compressed_only \
    --prompt-len 512 --decode-lens 128 --batch-size 4 --num-runs 1 \
    --warmup-decode-len 8 --gpu-memory-utilization 0.80 --enforce-eager
```

关键 kernel/API 汇总：

| item | total |
| --- | ---: |
| measured projected GPU time | 5455.5 ms |
| `Kernel2` | 2248.4 ms |
| GEMM `64x64_sliced1x2` | 1144.4 ms |
| ByteV2 CUTE stage1 | 292.6 ms |
| ByteV2 split reduce | 13.5 ms |
| `byte_v2_decode_append_cache_kernel` | 33.3 ms |
| `byte_v2_validate_decode_append_slots_kernel` | 10.2 ms |
| `byte_v2_init_decode_append_result_kernel` | 4.6 ms |

API 表中：

| API | before cache | after cache |
| --- | ---: | ---: |
| `cudaGetDeviceProperties_v2_v12000` | 4096 calls / 4506.2 ms | not present |

这个结果说明本轮性能主要来自两个点：

1. batched decode append 避免了 p512/b4/d128 下每 token/layer 的
   `compress_touched_blocks` 重路径；
2. device-property 缓存移除了 attention wrapper 的每层每 token host 查询。

#### 结论

本轮改动达到保留门槛，保留。

当前 p512/b4/d128 的 ByteV2 已从之前的约 `18-50 tok/s` 区间恢复到
`102.66 tok/s`，距离 raw `131.15 tok/s` 还差约 `21.7%`。剩余差距已经不再是
decode cache update 的毫秒级重压缩，而主要来自：

1. attention stage1 仍比 raw FlashAttention/FlashInfer decode 慢；
2. eager/profile 路径仍有大量小 kernel launch、`cudaMemcpyAsync` 和
   `cudaStreamSynchronize`；
3. ByteV2 还未启用 cudagraph production path。

下一步不应再做 open-block raw-tail 大改，除非新的 profile 再次显示
`compress_touched_blocks` 回到主路径。更合理的下一步是：

1. 先尝试把 validation/result host copy 转为 deferred-safe sticky error，减少
   eager 多 token decode append 的 `cudaMemcpyAsync`/sync；
2. 重新 profile raw vs ByteV2 的 attention stage1，继续推进 CUTE/CUTLASS-style
   stage1；
3. 在 non-eager/cudagraph 路径验证 ByteV2 是否能进一步接近 raw production。

### C.19 metadata-gated deferred batched decode append

C.18 的下一步是减少 eager 多 token decode append 的 validation/result host copy。
第一版尝试直接用 `has_deferred_error && !has_outlier_arena` 打开
multi-token append fast path，但真实 E2E warmup 会失败：

```text
Byte-v2 deferred decode cache append failed:
duplicate token slot in one cache update (error_code=3, block_id=1, ...)
```

原因是 `deferred_error` 只说明错误可以延后上报，不能证明本次 cache update 是
pure decode append。warmup/prefill 的多 token cache update 也会携带
`deferred_error`，直接跳过 validation 会把 prefill 当成 batched decode append。

#### 设计

本轮改成 metadata-gated：

1. `unified_kv_cache_update()` 从 `get_attention_context()` 取出
   `attn_metadata`。
2. 如果 backend 实现了 `do_kv_cache_update_with_metadata()`，则把 metadata 传给
   backend；其他 backend 仍走旧的 `do_kv_cache_update()`。
3. `ByteV2AttentionImpl` 只有在以下条件同时满足时传
   `decode_append_fast_path_safe=True`：
   - metadata 类型是 `ByteV2Metadata`；
   - `num_prefills == 0`；
   - `num_prefill_tokens == 0`；
   - `max_query_len == 1`；
   - `num_decodes == num_decode_tokens`；
   - `num_decode_tokens == num_actual_tokens == slot_mapping.numel()`；
   - `slot_mapping.numel() > 1`。
4. native op 新增默认参数
   `decode_append_fast_path_safe=False`。只有
   `decode_append_fast_path_safe && has_deferred_error && !has_outlier_arena`
   时，eager multi-token update 才跳过 validation/result host copy，直接走
   decode append fast path 和 device-side sticky error。

该方案保留单 token fast path、CUDA graph capture fast path、显式
`VLLM_BYTE_V2_DECODE_APPEND_BATCH_FASTPATH=1` 的原行为。

#### 验证

native rebuild：

```bash
uv pip install -e . --torch-backend=auto
```

结果：成功，用时约 `16m05s`。

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

decode smoke：

```bash
CUDA_VISIBLE_DEVICES=2 .venv/bin/python -m pytest \
  tests/v1/attention/test_byte_v2_decode.py \
  tests/v1/attention/test_byte_v2_backend.py::test_byte_v2_deferred_cache_update_records_sticky_error \
  tests/v1/attention/test_byte_v2_backend.py::test_byte_v2_deferred_batched_decode_append_fast_path_cuda \
  tests/v1/attention/test_byte_v2_backend.py::test_byte_v2_deferred_batched_duplicate_block_records_error_cuda \
  -q
```

结果：`29 passed`。

#### E2E

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

| variant | tok/s | elapsed | 对比 |
| --- | ---: | ---: | ---: |
| raw | 131.09 | 3.906s | 100% |
| ByteV2 no-deferred | 102.60 | 4.990s | 78.3% raw |
| ByteV2 deferred metadata gate | 122.24 | 4.189s | 93.3% raw |

本轮相对 no-deferred 提升约 `+19.1%`，保留。

#### nsys 对比

对比 C.18 cached-props profile 与本轮 deferred metadata-gated profile：

| item | C.18 no-deferred | C.19 deferred gate |
| --- | ---: | ---: |
| `byte_v2_validate_decode_append_slots_kernel` | 4064 calls / 10.24 ms | 0 |
| `byte_v2_decode_append_cache_kernel` | 4064 calls / 33.29 ms | 4064 calls / 34.49 ms |
| `byte_v2_init_decode_append_result_kernel` | 4064 calls / 4.63 ms | 4064 calls / 6.21 ms |
| `byte_v2_record_deferred_cache_update_error_kernel` | 0 | 4064 calls / 6.08 ms |
| `cudaMemcpyAsync` | 17293 calls / 2026.35 ms | 5005 calls / 1056.50 ms |
| `cudaStreamSynchronize` | 8320 calls / 28.17 ms | 320 calls / 2.81 ms |

GPU kernel 时间上，deferred path 多了约 `6.08 ms` 的 sticky error record，
但移除了 validation kernel 的约 `10.24 ms`，更重要的是大幅减少 host API 同步和
host/device copy 次数。

#### 结论

本轮改动达到保留门槛。p512/b4/d128 eager 下，ByteV2 compressed-only 已从
C.18 的 `102.66 tok/s` 提升到 `122.24 tok/s`，与 raw `131.09 tok/s` 的差距缩小到
约 `6.7%`。

下一步重点不再是 decode append validation host copy，而是：

1. 继续 profile attention stage1 与 raw FlashAttention/FlashInfer 的差距；
2. 尝试 non-eager/cudagraph 路径，避免 raw production cudagraph 重新拉开差距；
3. 如果 cache update 继续优化，优先考虑把
   `init_decode_append_result + record_deferred_cache_update_error` 融进
   `byte_v2_decode_append_cache_kernel`，但门槛应看 E2E，而不是 microbench。

### C.20 latest raw vs ByteV2 profile after deferred append

Step C.19 后重新做 p512/b4/d128 detailed profile。由于本轮 GPU2/GPU0 被其他进程
占用，raw nsys 不能干净重跑；这里使用已有同 workload 的 raw eager 有效 profile
`nsys_next_sp_raw_p512_b4_d128`，并与最新 ByteV2 deferred metadata-gated profile
`nsys_deferred_append_bytev2_cute_p512_b4_d128` 对比。raw 路径未被 C.19 改动影响。

E2E：

| mode | tok/s | elapsed | ByteV2/raw |
| --- | ---: | ---: | ---: |
| raw eager | 131.09 | 3.906s | 100% |
| ByteV2 eager | 122.24 | 4.189s | 93.3% |

GPU kernel 分桶：

| category | raw | ByteV2 | delta |
| --- | ---: | ---: | ---: |
| attention | 93.37 ms | 305.29 ms | +211.92 ms |
| cache update | 12.25 ms | 48.25 ms | +36.00 ms |
| GEMM/MLP/linear | 3677.46 ms | 3679.25 ms | +1.78 ms |
| norm/rope/activation | 84.87 ms | 76.17 ms | -8.70 ms |
| sampling/scheduler misc | 4.81 ms | 7.11 ms | +2.30 ms |
| total GPU kernel | 3873.67 ms | 4117.32 ms | +243.65 ms |

结论：

1. 剩余差距的主因仍是 attention stage1。ByteV2 CUTE stage1 + reduce 约
   `305.29 ms`，raw FlashAttention decode 约 `93.37 ms`。
2. cache update 是第二项，约 `+36 ms`；其中
   `init_decode_append_result + record_deferred_cache_update_error` 约 `12.29 ms`。
3. GEMM/MLP 和 sampling 不是差距来源。
4. cudagraph 缺失不是当前主要解释。已有干净 production/cudagraph baseline：
   raw `133.60 tok/s`，ByteV2 no-arena `123.19 tok/s`，ByteV2 仍约为 raw 的
   `92.2%`，与 eager 的 `93.3%` 接近。ByteV2 backend 也已声明
   `UNIFORM_SINGLE_TOKEN_DECODE` cudagraph support。

本轮尝试在 GPU0 以 `gpu_memory_utilization=0.32` 重跑 non-eager sanity check，但
外部进程占用约 `29.10 GiB`，Inductor 编译阶段 OOM。因此需要等 GPU 空闲后再做
干净 production nsys 复测。

GPU0 空闲后已完成干净 production/cudagraph 复测：

```text
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

E2E：

| mode | tok/s | elapsed | ByteV2/raw |
| --- | ---: | ---: | ---: |
| raw production/cudagraph | 133.55 | 3.834s | 100% |
| ByteV2 production/cudagraph | 125.15 | 4.091s | 93.7% |

ByteV2 fallback pool 未耗尽：

```text
any_exhausted=false
total_assigned_blocks=256
total_capacity=16384
max_next_slot=40
```

两边日志都显示完成 `PIECEWISE` 和 `FULL` cudagraph capture，因此 cudagraph
缺失不是当前主要解释。

production/cudagraph GPU kernel 分桶：

| category | raw | ByteV2 | delta |
| --- | ---: | ---: | ---: |
| attention | 93.30 ms | 304.19 ms | +210.89 ms |
| cache update | 9.99 ms | 42.20 ms | +32.20 ms |
| GEMM/MLP/linear | 3673.36 ms | 3674.40 ms | +1.03 ms |
| norm/rope/activation | 62.02 ms | 43.96 ms | -18.07 ms |
| sampling/scheduler misc | 5.25 ms | 6.59 ms | +1.34 ms |
| total GPU kernel | 3844.55 ms | 4072.06 ms | +227.51 ms |

top kernels：

| path | kernel | total | launches | avg |
| --- | --- | ---: | ---: | ---: |
| raw attention | `flash_fwd_splitkv_kernel` | 76.09 ms | 4096 | 18.58 us |
| raw attention | `flash_fwd_splitkv_combine_kernel` | 17.21 ms | 4064 | 4.23 us |
| ByteV2 attention | `byte_v2_paged_decode_attention_gqa_cute_split_stage1_kernel` | 291.01 ms | 4096 | 71.05 us |
| ByteV2 attention | `byte_v2_paged_decode_attention_split_reduce_parallel_kernel` | 13.18 ms | 4096 | 3.22 us |
| raw cache update | `reshape_and_cache_flash_kernel` | 9.99 ms | 4096 | 2.44 us |
| ByteV2 cache update | `byte_v2_decode_append_cache_kernel` | 31.70 ms | 4064 | 7.80 us |

更新结论：

1. production 路径下 ByteV2 已达到 raw 的 `93.7%`，比之前受环境限制时引用的
   `92%-93%` baseline 略好。
2. 剩余差距仍主要来自 attention stage1。ByteV2 attention 多 `210.89 ms`，
   cache update 多 `32.20 ms`，GEMM/MLP 只多 `1.03 ms`。
3. 因此下一步不应优先追 sampling、GEMM 或 cudagraph 开关，而应继续做
   stage1 K/V decode、shared/WMMA load、PV/write partial，以及小步 cache
   update fusion。

下一步主线：

1. 继续优化 ByteV2 attention stage1：K/V decode 指令、shared/WMMA layout、
   PV/write partial、split reduce。
2. cache update 只做低风险 fusion 实验，预期上限较小，必须用 E2E 决定是否保留。

### C.21 CUTE metadata stage1 early-exit + NCU refresh

按 C.20 结论，本轮先不改代码，而是重新 profile 当前最优 production-like
CUTE metadata stage1。配置：

```text
GPU=0
split_k=8
fallback_ratio=0.03
fallback_pattern=single_outlier
tile_fallback_pool=on
cute_stage1_auto=on
tile_fastpath=on
parallel_reduce=on
outlier_arena=off
```

输出文件：

```text
benchmarks/profiles/step46_cute_early_summary.json
benchmarks/profiles/step46_ncu_cute_stage1_p512_b4_d128_m{0,1,2,3}.csv
benchmarks/profiles/step46_ncu_cute_stage1_p2048_b1_d32_m{0,1,2,3}.csv
benchmarks/profiles/step46_ncu_cute_stage1_p512_b4_d128_m0.ncu-rep
```

decode-only early-exit median latency：

| workload | full | load/decode | QK inc | softmax inc | PV/write inc |
| --- | ---: | ---: | ---: | ---: | ---: |
| p512/b4/d128 | 89.09 us | 73.73 us | 5.12 us | 2.05 us | 8.19 us |
| p512/b4/d256 | 90.11 us | 71.68 us | 4.10 us | 2.05 us | 12.29 us |
| p2048/b1/d32 | 202.75 us | 155.65 us | 10.24 us | 9.22 us | 27.65 us |

NCU stage1-only duration：

| workload | full | load/decode | QK inc | softmax inc | PV/write inc |
| --- | ---: | ---: | ---: | ---: | ---: |
| p512/b4/d128 | 73.22 us | 54.27 us | 4.42 us | 2.75 us | 11.78 us |
| p2048/b1/d32 | 234.50 us | 178.85 us | 12.93 us | 10.62 us | 32.10 us |

NCU 关键信号：

| workload | executed inst | global excessive sectors | shared excessive wavefronts | regs/thread | waves/SM | achieved occ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| p512/b4/d128 full | 6.43M | 319856 / 686464 = 47% | 696320 / 1070608 = 65% | 96 | 0.61 | 25.22% |
| p2048/b1/d32 full | 6.00M | 317168 / 663808 = 48% | 696320 / 1051024 = 66% | 96 | 0.15 | 8.33% |

source page 只能看到 SASS 地址，没有 CUDA source line 映射；当前构建缺少
可用 lineinfo。若后续需要真正 line-level NCU，需要用 lineinfo/debug 重新编译
CUDA 扩展。

结论：

1. stage1 第一瓶颈仍是 compressed K/V load/decode，占 `74%-76%`。
2. QK 和 softmax 都不是当前首要方向。
3. shared layout 和 global access pattern 都有硬指标问题：shared excessive
   wavefronts 约 `65%-66%`，global excessive sectors 约 `47%-48%`。
4. p2048/b1 只有 `0.15 waves/SM`，长 context/低 batch 下并行度不足会放大
   单 CTA page chunk 的 latency。

本轮还尝试过一个最小 no-arena overlay-skip patch：没有 outlier arena 时跳过
`byte_v2_overlay_*_tile_outliers_to_shared()` 调用。Step C.22 已完成编译后
验证：decode-only 有小幅正向信号，但 NCU stage1 指标不变，E2E 略退化。
因此该 patch 已回滚，不作为结果保留。

下一轮建议：

1. 做一个 opt-in no-arena compressed tile loader fast path：当没有 outlier arena
   且 tile 无 fallback 时，跳过 overlay metadata 准备和 overlay helper 调用。
2. K/V loader 拆成专用 helper，避免 runtime `is_value`/overlay 分支。
3. 保留门槛看 NCU 而不是只看 E2E：load/decode duration、full stage1 duration、
   executed instructions 或 excessive global sectors 必须下降；E2E 不能退化。

### C.22 no-arena overlay-skip 实验结果

已按 C.21 的建议做最小 opt-in 实验，但最终不保留。

实验内容：在 CUTE metadata stage1 的 no-arena production 路径中，如果 tile
没有 fallback，则跳过 `byte_v2_overlay_*_tile_outliers_to_shared()` helper
调用。

correctness：

```text
3 个 CUTE metadata CUDA smoke 测试全部通过，耗时 34.24s。
```

decode-only，`p512/b4/d128`：

| variant | median us | p90 us | tok/s |
| --- | ---: | ---: | ---: |
| baseline | 94.21 | 102.40 | 42,459 |
| overlay-skip | 89.09 | 91.14 | 44,899 |
| baseline repeat 1 | 91.14 | 93.18 | 43,890 |
| overlay-skip repeat 1 | 89.09 | 92.16 | 44,899 |
| baseline repeat 2 | 90.11 | 91.14 | 44,389 |
| overlay-skip repeat 2 | 89.09 | 90.11 | 44,899 |

NCU stage1：

| metric | baseline | overlay-skip |
| --- | ---: | ---: |
| Duration | 74.240 us | 74.016 us |
| Executed Instructions | 6,514,096 | 6,514,096 |
| Branch Instructions | 715,824 | 715,824 |
| Registers / thread | 96 | 96 |
| Waves / SM | 0.61 | 0.61 |
| Excessive global sectors | 319,856 / 686,464, 47% | 319,856 / 686,464, 47% |
| Excessive shared wavefronts | 696,320 / 1,070,608, 65% | 696,320 / 1,070,608, 65% |

E2E，Llama-3 8B，`p512/b4/d128`，ByteV2 compressed-only，cudagraph enabled：

| variant | elapsed s | output tok/s |
| --- | ---: | ---: |
| baseline | 4.09556 | 125.013 |
| overlay-skip | 4.09848 | 124.924 |

结论：decode-only 的小幅正向没有被 NCU 和 E2E 支持，因此已回滚该 patch。
代码中不再保留 `VLLM_BYTE_V2_DECODE_CUTE_NO_ARENA_OVERLAY_SKIP`、
`--cute-no-arena-overlay-skip` 或 kernel 参数。后续不继续优化空 outlier
overlay helper 早退路径，优先回到 K/V load-decode 访问模式、shared
layout/WMMA load pattern 和低并行度问题。

### C.23 lineinfo NCU：stage1 源码行级瓶颈

本轮为 `_C_stable_libtorch` 临时构建带 `-lineinfo` 的 profile 版本，并重新跑
CUTE metadata stage1 NCU。目的不是改代码，而是把 C.21 的早退 profile 进一步
映射到 CUDA 源码行。

构建和 smoke：

```text
cmake --build /tmp/tmpvfglk_3g.build-temp --target _C_stable_libtorch -j 8
cp /tmp/tmpvfglk_3g.build-temp/_C_stable_libtorch.abi3.so vllm/_C_stable_libtorch.abi3.so

p512/b4/d128 decode-only smoke:
median_us=88.06, p90_us=92.16, tok/s=45421.51
```

NCU workload：

```text
p512/b4/d128, split_k=8, fallback_ratio=0.03, single_outlier
p2048/b1/d32, split_k=8, fallback_ratio=0.03, single_outlier
```

总体指标：

| workload | duration | executed inst | long scoreboard | issue active | waves/SM | achieved occ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| p512/b4/d128 | 73.376 us | 6.431M | 6.35 cycle/issue | 0.23 | 0.61 | 25.23% |
| p2048/b1/d32 | 233.728 us | 6.004M | 6.35 cycle/issue | 0.08 | 0.15 | 8.33% |

访存指标：

| workload | global excessive sectors | shared excessive wavefronts | L1/TEX hit | L2 hit |
| --- | ---: | ---: | ---: | ---: |
| p512/b4/d128 | 319,856 / 686,464 = 47% | 696,320 / 1,070,608 = 65% | 68.20% | 15.50% |
| p2048/b1/d32 | 317,168 / 663,808 = 48% | 696,320 / 1,051,024 = 66% | 69.74% | 7.39% |

源码行级结论：

1. 最大 global excessive sectors 来自 compressed payload 的 `byte_v2_load_u16()`：

```cpp
// cache_kernels.cu:1051-1054
return static_cast<uint16_t>(ptr[0]) |
       (static_cast<uint16_t>(ptr[1]) << 8);
```

line 1053/1054 映射到大量 `LDG.E.U8`，两行合计基本解释了本轮
`317K-320K` excessive global sectors 中的主要部分。这说明主要问题不是
outlier metadata 分支，而是 low payload 的两个 U8 load 和当前 tile layout
没有形成理想合并访问。

2. decode 指令仍然显著：

```cpp
// cache_kernels.cu:1188-1193
const int low_exp_lsb = low >> 7;
const int delta_hi = code & 0x07;
...
return static_cast<uint16_t>((high << 8) | low);
```

`low >> 7`、`code & 0x07` 等行仍有约 `131K` 级别执行次数；payload load
优化必须同时关注 decode 指令数，不只是 bytes ratio。

3. shared excessive 主要来自 V shared 写入和 `load_matrix_sync` 展开的 shared
load pattern：

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

单纯 stride padding 或 row/col-major 切换此前已经退化；下一次如果继续做 shared
layout，必须按实际 SASS `LD.E/ldmatrix` 的 lane access pattern 设计 swizzle。

4. p2048/b1 的主要额外问题是并行度：

```text
p2048/b1/d32 grid_size=64, waves/SM=0.15, achieved occupancy=8.33%
```

这不是单个 load 指令顺序能解决的问题；长 context/低 batch 需要 page-chunk
更细粒度的 CTA 并行，或 persistent/page-parallel stage1。

更新后的下一步顺序：

1. **payload load v3 lower-bound**：先做 opt-in aligned U16/U32/warp-stripe
   low payload loader，只接 decode-only microbench 和 NCU。保留门槛：
   global excessive sectors、long scoreboard、stage1 duration 至少两项下降。
2. **shared/WMMA layout v3**：只在 payload loader 有明确收益或 source profile
   继续指向 shared 后再做；必须围绕 SASS shared load pattern 设计，而不是再做
   普通 stride 改动。
3. **p2048/b1 page-parallel**：如果 payload loader 对 p512 有效但 p2048 仍慢，
   下一步转向更细 page chunk 并行，提升 waves/SM。
4. **不要继续做** no-arena overlay metadata skip、outlier bitmap 默认开启、
   split-K 微调、direct-output 或 persistent partial workspace 这类外围优化；
   这些方向之前已经没有稳定 E2E 收益。

## C.24 Step49：aligned U16 payload loader 下限实验结果

根据 C.23 的建议，本轮尝试了一个很小的 payload load v3 lower-bound：
在 CUTE metadata stage1 的 compressed tile fastpath 中，把 `low` payload 的
两个 U8 load 替换为 opt-in aligned `uint16_t` load。该实验只用于 decode-only
和 NCU A/B，不接默认生产路径。

功能验证：

```text
CUDA_VISIBLE_DEVICES=0 \
VLLM_BYTE_V2_DECODE_ALIGNED_U16_PAYLOAD_LOAD=1 \
.venv/bin/python -m pytest \
  tests/v1/attention/test_byte_v2_decode.py::test_byte_v2_paged_decode_attention_op_cute_stage1_metadata_cuda -q

结果：1 passed
```

NCU 结果：

| workload | variant | stage1 duration | inst | long scoreboard | global sectors | global excessive | shared excessive |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| p512/b4/split-k8 | baseline | 72.960 us | 6.431M | 6.33 | 686,464 | 319,856 | 696,320 |
| p512/b4/split-k8 | aligned U16 | 74.240 us | 6.693M | 6.04 | 494,000 | 127,392 | 696,320 |
| p2048/b1/split-k8 | baseline | 234.976 us | 6.004M | 6.53 | 663,808 | 317,168 | 696,320 |
| p2048/b1/split-k8 | aligned U16 | 238.560 us | 6.266M | 6.18 | 471,344 | 124,704 | 696,320 |

结论：

1. aligned U16 load 能显著降低 global excessive sectors，说明 Step48 的热点定位
   是正确的。
2. 但 stage1 duration 没有下降，p512/p2048 均略慢；executed instructions 增加
   约 4%，shared excessive 完全不变。
3. 因此本实验不满足保留门槛，代码已回滚，只保留 profile 和文档记录。

后续路线调整：

1. 不再继续做单点 scalar loader 替换。
2. 如果继续优化 payload load，应做真正的 layout-v3/warp-lane stripe：让 global
   load、decode、shared store、WMMA 消费顺序一起变化。
3. 更直接的下一步是 shared/WMMA layout lower-bound，因为当前
   `696,320` shared excessive wavefronts 没被 aligned U16 影响。

## C.25 Step50：shared/WMMA lower-bound profile

本轮根据 C.24 的结论，没有继续做 scalar payload loader，而是复测已有显式
`flash-stage1` 实验路径作为 shared/WMMA lower-bound。

该路径通过：

```text
VLLM_BYTE_V2_DECODE_FLASH_STAGE1=1
```

启用 `byte_v2_paged_decode_attention_gqa4_h128_flash_split_stage1_kernel`。它仍然
使用真实 compressed page metadata/fallback，但移除了 CUTE metadata kernel 中的
`scores/p_shared/pv_shared` 中间共享内存路径，改为 4 个 warp 直接做 online
softmax/PV。目标是验证 shared/WMMA 往返是否确实是 Step49 之后的下一层瓶颈。

功能验证：

```text
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest \
  tests/v1/attention/test_byte_v2_decode.py::test_byte_v2_paged_decode_attention_op_flash_stage1_raw_fallback_cuda \
  tests/v1/attention/test_byte_v2_decode.py::test_byte_v2_paged_decode_attention_op_flash_stage1_tile_fallback_cuda -q

结果：2 passed, 16 warnings
```

decode-only A/B：

| workload | CUTE metadata | flash-stage1 | 结论 |
| --- | ---: | ---: | --- |
| p512/b4/split-k8 | 87.04 us | 79.87 us | flash 更快 |
| p512/b4/split-k8，反向复跑 | 89.09 us | 64.51 us | flash 更快 |
| p2048/b1/split-k8 | 204.80 us | 171.97 us | flash 更快 |
| p2048/b1/split-k8，反向复跑 | 210.94 us | 171.01 us | flash 更快 |
| p2048/b1/split-k64 | 93.18 us | 86.02 us | flash 更快 |

NCU 关键指标：

| workload | variant | duration | inst | long scoreboard | barrier | issue active | regs/thread | shmem | waves/SM | global excessive | shared excessive |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| p512/b4/s8 | CUTE | 73.440 us | 6.431M | 6.37 | 2.09 | 20.08% | 96 | 14,928 | 0.61 | 319,856 | 696,320 |
| p512/b4/s8 | flash | 62.912 us | 6.521M | 5.97 | 0.14 | 23.51% | 116 | 9,216 | 0.76 | 319,856 | 360,448 |
| p2048/b1/s8 | CUTE | 233.760 us | 6.004M | 6.38 | 1.70 | 5.94% | 96 | 14,928 | 0.15 | 317,168 | 696,320 |
| p2048/b1/s8 | flash | 194.336 us | 6.257M | 5.80 | 0.07 | 7.52% | 116 | 9,216 | 0.19 | 317,168 | 360,448 |

结论：

1. flash-stage1 把 shared excessive 从 `696,320` 降到 `360,448`，下降约 48%。
2. stage1 duration 对 p512/p2048 都下降，说明 shared/WMMA lower-bound 成立。
3. global excessive 不变，说明该路径没有解决 payload global load。
4. flash-stage1 的指令数和 registers/thread 增加，因此它不是直接可默认启用的
   免费优化。

E2E 复测：

```text
model=/mnt/sda1/yxz/byte_v2/Meta-Llama-3-8B-Instruct
p512/b4/d64
split_k=8
sparse_fallback_pool_ratio=0.03
outlier_arena=off
enforce_eager
gpu_memory_utilization=0.45
```

| variant | run1 | run2 |
| --- | ---: | ---: |
| CUTE metadata | 102.91 tok/s | 99.72 tok/s |
| flash-stage1 | 61.73 tok/s | 89.16 tok/s |

两轮 fallback pool 均未耗尽。E2E 没有兑现 kernel/NCU 的收益，复跑后 flash 仍低于
CUTE metadata 约 10%。因此本轮不启用 `flash-stage1` auto/default，只保留显式
实验开关。

后续决策：

1. 先用 production build 重新做同 workload 的 CUTE vs flash E2E，排除当前
   lineinfo build 对整机吞吐的影响。
2. 做 detailed nsys/NVTX profile，确认差距来自 attention stage1、reduce、
   cache update、sampling/GEMM、host/API gap，还是 launch/cudagraph 缺失。
3. 如果 flash-stage1 的 E2E 退化来自高 register 或跨层调度影响，则不要继续以它
   作为 production kernel，而是回到 CUTE metadata baseline 做局部 shared layout
   优化。
4. 如果退化来自非 attention 环节，则优先优化 launch/cudagraph/cache update，
   因为 stage1 下限已经证明还有约 14%-17% kernel 空间。

## C.26 Step51：CUTE vs flash-stage1 detailed profile

本轮继续验证 C.25 的矛盾：flash-stage1 在 decode-only/NCU 中更快，但 E2E 不快。

### profiling 限制

尝试过三种 nsys 方式：

```text
--capture-range=nvtx --nvtx-capture=byte_v2_bench_measured
full trace parent/child
full trace + --trace-fork-before-exec=true
```

前两种没有得到可用 report 或没有 `CUPTI_ACTIVITY_KIND_KERNEL`；第三种也没有捕获
EngineCore worker 的 CUDA kernel 表。原因是 GPU 工作在 vLLM multiprocessing
EngineCore worker 中，当前 nsys 注入方式没有跟到 worker。

因此本轮为了得到 kernel-level 分解，使用 profiling-only 配置：

```text
VLLM_ENABLE_V1_MULTIPROCESSING=0
```

这只用于分析 GPU kernel 分类，不作为 production E2E 结论。当前
`_C_stable_libtorch` 仍是 lineinfo build，也不作为正式性能报告。

主要 profile 文件：

```text
benchmarks/profiles/step51_nsys_uniproc_cute_p512_b4_d64.sqlite
benchmarks/profiles/step51_nsys_uniproc_flash_p512_b4_d64.sqlite
```

### workload

```text
p512/b4/d64
split_k=8
sparse_fallback_pool_ratio=0.03
outlier_arena=off
enforce_eager
gpu_memory_utilization=0.45
```

### profile 结果

按 `byte_v2_bench_decode_len_64_run_0` NVTX range 统计：

| category | CUTE metadata | flash-stage1 | 差值 |
| --- | ---: | ---: | ---: |
| measured wall | 2408.8 ms | 3506.4 ms | +1097.6 ms |
| summed GPU kernel | 2294.9 ms | 3259.9 ms | +965.0 ms |
| GEMM/linear | 2041.4 ms | 2868.9 ms | +827.4 ms |
| attention stage1 | 180.9 ms | 313.4 ms | +132.5 ms |
| decode cache update | 23.2 ms | 27.5 ms | +4.4 ms |
| attention reduce | 6.6 ms | 6.8 ms | +0.2 ms |
| sampling/scheduler | 2.3 ms | 2.4 ms | +0.1 ms |

实际命中的 attention kernel：

```text
CUTE:
byte_v2_paged_decode_attention_gqa_cute_split_stage1_kernel
2048 calls, 180.9 ms total, 88.3 us avg

flash:
byte_v2_paged_decode_attention_gqa4_h128_flash_split_stage1_kernel
2048 calls, 313.4 ms total, 153.0 us avg
```

### 结论

1. E2E path 中 flash-stage1 没有复现 decode-only microbench 的优势；在
   profiling-only E2E measured range 里，flash stage1 明显慢于 CUTE。
2. flash run 的 GEMM/linear 也明显变慢，说明这组 E2E/profile 仍存在运行环境、
   profiler 或调度噪声；但即使只看 ByteV2 attention stage1，flash 也不应默认启用。
3. cache update 和 reduce 都不是这轮差距主因。CUTE 的 decode cache update 约
   `23 ms`，attention reduce 约 `6.6 ms`，相对 measured wall 很小。
4. 当前 production baseline 应继续保留 CUTE metadata path，flash-stage1 只作为
   explicit lower-bound/实验开关。

### 下一步

1. 先恢复 production build，再复测 E2E：

   ```text
   cd /mnt/sda1/yxz/byte_v2/vllm
   uv pip install -e . --torch-backend=auto
   ```

2. 不继续把 flash-stage1 往 default 推。后续 kernel 优化应回到 CUTE metadata
   stage1，重点降低 `byte_v2_paged_decode_attention_gqa_cute_split_stage1_kernel`
   的 load/decode/shared/WMMA 成本。
3. 增加一个不依赖 nsys worker 注入的可选 CUDA event timing hook，在
   multiprocessing production path 内直接统计 attention stage1、reduce、cache update
   的 per-layer/per-token 时间。这样后续比较 raw/CUTE/新 kernel 不会受 nsys 注入限制。

## C.27 memory-bound 长上下文 E2E: CUTE metadata split-stage1 vs raw

### 目的

用更长 context 的 decode workload 检查 ByteV2 压缩 KV cache 的读带宽优势是否能在
E2E 中体现。测试保持 production CUDA graph 路径，不使用 `--enforce-eager`。

ByteV2 使用当前稳定 baseline：

```text
CUTE metadata split-stage1
VLLM_BYTE_V2_DECODE_CUTE_STAGE1_AUTO=1
VLLM_BYTE_V2_DECODE_FLASH_STAGE1=0
split_k=8
outlier_arena=off
sparse_fallback_pool_ratio=0.03
```

### 结果

输出文件：

```text
benchmarks/profiles/step52_memory_bound_raw_bytev2_cute_p4096_b1_d64_d128.json
benchmarks/profiles/step52_memory_bound_raw_bytev2_cute_p8064_b1_d64.json
```

| workload | raw tok/s | ByteV2 CUTE tok/s | ByteV2 / raw | pool exhausted |
| --- | ---: | ---: | ---: | --- |
| p4096/b1/d64 | 34.34 | 24.36 | 70.9% | false |
| p4096/b1/d128 | 34.45 | 24.57 | 71.3% | false |
| p8064/b1/d64 | 33.20 | 18.94 | 57.1% | false |

`p8064/b1/d64` 用来近似 8K context，因为 Llama-3 配置的
`max_position_embeddings=8192`，不能运行 `prompt_len=8192 + decode_len=64`。

### 结论

当前 ByteV2 CUTE metadata split-stage1 在长上下文 memory-bound E2E 场景中没有超过
raw vLLM。context 从 4K 增加到接近 8K 后，ByteV2 相对 raw 的比例从约 `71%`
下降到约 `57%`，说明压缩 KV 的纯读带宽优势仍被 stage1 里的解码、metadata、
shared/WMMA load pattern 和 split-stage 调度成本抵消。

sparse fallback pool 没有耗尽，所以这轮退化不是 capacity fallback 造成。

### 对优化路线的影响

下一步不要继续假设“更 memory-bound 就会自然追上 raw”。需要先把 production E2E
中的分段时间打清楚：

1. 增加 CUDA event timing hook，直接统计 ByteV2 attention stage1、reduce、
   decode cache update 的 per-layer/per-token 时间。
2. 继续围绕 `byte_v2_paged_decode_attention_gqa_cute_split_stage1_kernel` 优化
   load/decode pipeline、shared layout、WMMA load pattern。
3. 当 stage1 接近 raw attention 后，再重新验证更长 context 或更高并发是否能体现
   KV 压缩读带宽优势。

## C.28 长 decode_len E2E: CUTE metadata split-stage1 vs raw

### 目的

在 Step 52 的长 context、短 decode 基础上，加大 decode_len，检查 steady-state
较长 decode 是否能摊薄调度开销，并让压缩 KV 的读带宽收益体现出来。

### 结果

输出文件：

```text
benchmarks/profiles/step53_long_decode_raw_bytev2_cute_p4096_b1_d256_d512_d1024.json
benchmarks/profiles/step53_long_decode_raw_bytev2_cute_p7168_b1_d256_d512.json
```

| workload | raw tok/s | ByteV2 CUTE tok/s | ByteV2 / raw | pool exhausted |
| --- | ---: | ---: | ---: | --- |
| p4096/b1/d256 | 34.24 | 24.54 | 71.7% | false |
| p4096/b1/d512 | 34.26 | 24.36 | 71.1% | false |
| p4096/b1/d1024 | 34.27 | 23.94 | 69.9% | false |
| p7168/b1/d256 | 33.62 | 20.14 | 59.9% | false |
| p7168/b1/d512 | 33.61 | 20.04 | 59.6% | false |

### 结论

长 decode 没有让 ByteV2 追近 raw。`p4096/b1` 下 ByteV2 维持在约 `70%-72% raw`，
并随 decode_len 变长略降；`p7168/b1` 下只有约 `60% raw`。raw 在两个 context 下都
非常稳定，而 ByteV2 随 context 变长明显下降。

这说明当前差距不是短 decode 的调度噪声或初始化摊销问题，而是每个 decode step 内的
稳定额外成本。后续仍应优先定位和优化 CUTE metadata stage1 的：

```text
compressed K/V payload decode
metadata/fallback check
shared layout / WMMA load
split-stage partial write + reduce
```

下一步建议先做 production path 内 CUDA event timing hook，再针对 stage1 的
load/decode/shared/WMMA pipeline 做优化。

## C.29 batch_size 扩展与显存容量判断

### 目的

确认当前长 context/long decode 测试是否已经是显存容量受限，并测试增大 batch 后
ByteV2 是否能体现压缩 KV cache 的容量或带宽优势。

### 结果

输出文件：

```text
benchmarks/profiles/step54_batch_memory_raw_bytev2_cute_p4096_b4_d128_d256.json
benchmarks/profiles/step54_batch_memory_raw_bytev2_cute_p4096_b8_d128.json
benchmarks/profiles/step54_batch_memory_raw_bytev2_cute_p4096_b8_d128_eager.json
```

`p4096/b4` production E2E：

| workload | raw tok/s | ByteV2 CUTE tok/s | ByteV2 / raw | pool exhausted |
| --- | ---: | ---: | ---: | --- |
| p4096/b4/d128 | 123.93 | 87.87 | 70.9% | false |
| p4096/b4/d256 | 125.17 | 90.17 | 72.0% | false |

`p4096/b8/d128`：

| mode | result |
| --- | --- |
| raw production | 227.41 tok/s, passed |
| raw eager | 223.79 tok/s, passed |
| ByteV2 production | failed |
| ByteV2 eager | failed |

ByteV2 `b8` 的失败不是 OOM，而是：

```text
Byte-v2 native prefill direct cache update failed: invalid slot mapping
error_code=2
fallback_pool_used=0
fallback_pool_capacity=512
```

production 失败发生在 CUDA graph capture，eager 失败发生在 warmup/真实 prefill
执行，所以不是单纯 cudagraph 问题。

### 显存容量判断

当前已测试场景不是显存容量受限：

```text
b4 raw:    GPU KV cache size 153,984 tokens
b4 ByteV2: GPU KV cache size 191,328 tokens
b8 raw:    GPU KV cache size 140,160 tokens
b8 ByteV2: GPU KV cache size 173,248 tokens
```

而实际请求 token 数约为：

```text
p4096/b4/d256: 17,408 tokens
p4096/b8/d128: 33,792 tokens
```

这些都明显低于 KV cache capacity。ByteV2 的容量优势存在，但当前测试点还没有逼近
capacity 边界，因此不会自然体现为吞吐反超。

### 结论

1. 增大到 `b4` 后 ByteV2 仍只有约 `71%-72% raw`，和 `b1` 基本一致。
2. raw 能继续扩展到 `b8`，说明当前 raw 也没有被显存容量卡住。
3. ByteV2 在 `b8` 暴露出 prefill direct cache update 的 slot mapping bug，需要先
   修复，否则无法继续测试更高 batch。
4. 后续如果要验证压缩 KV 的容量优势，应选择 raw 接近 KV capacity 的 workload；
   但在此之前必须先修复 `p4096/b8` 的 ByteV2 prefill slot mapping 问题。

### 下一步

先修复：

```text
ByteV2 native prefill direct cache update invalid slot mapping at p4096/b8
```

修复后再跑：

```text
p4096/b8/d128 production/eager
p4096/b16/d128
p7168/b8/d128
```

## C.30 长上下文优化路线

### 背景

当前性能不是简单的“代码退化”，而是 workload 从短 context 变成长 context 后，
ByteV2 的 attention stage1 成本被放大：

| workload | raw tok/s | ByteV2 tok/s | ByteV2 / raw |
| --- | ---: | ---: | ---: |
| p512/b4/d128 | 133.55 | 125.15 | 93.7% |
| p4096/b1/d128 | 34.45 | 24.57 | 71.3% |
| p4096/b4/d128 | 123.93 | 87.87 | 70.9% |
| p7168/b1/d512 | 33.61 | 20.04 | 59.6% |

raw 从 `p512` 到 `p4096` 下降不多，但 ByteV2 明显下降。这说明当前差距主要来自
长 context 下随 KV pages 增长的 per-step 成本：

```text
compressed K/V payload decode
metadata/fallback check
shared layout / WMMA load
split-stage partial write + reduce
decode append/cache update 的小步开销
```

当前 `p4096/b4` 也不是显存容量受限；raw 和 ByteV2 的 KV cache capacity 都远高于
实际请求 token 数。因此短期优化目标不是“靠容量优势自然超过 raw”，而是先降低长
context stage1 的结构性额外成本。

### 总目标

分两条线推进：

1. **fused compressed attention 路线**：继续让 KV 以 ByteV2 compressed 形式常驻
   HBM，在 decode stage1 中直接解码并参与 attention。目标是把长 context E2E 从
   `70% raw` 提升到 `80%+ raw`，再继续逼近 `90% raw`。
2. **hybrid active raw staging 路线**：compressed KV 作为常驻格式，只对当前活跃
   decode batch 的长 prefix 临时解压到 raw BF16 staging，然后复用 raw attention
   fast path。目标是验证“压缩常驻 + 活跃窗口 raw 化”是否能同时获得较高 capacity
   和接近 raw 的 decode 性能。

这两条路线不互斥。fused compressed attention 是最终低显存读带宽路线；active raw
staging 是长 decode 场景的 oracle/系统折中路线。

### Track 0: 先确认没有短 context 回归

每轮长 context 优化前，先固定短 context 哨兵：

```text
p512/b4/d128 production/cudagraph
p512/b4/d256 production/cudagraph
```

保留门槛：

```text
p512/b4/d128 ByteV2 >= 90% raw
p512/b4/d256 ByteV2 >= 90% raw
```

如果短 context 先掉到 80% 左右，说明存在新代码回归，先修回归，不进入长 context
优化判断。

### Track 1: 修复 b8 prefill slot mapping

当前 `p4096/b8/d128` 下 raw production/eager 均可运行，但 ByteV2 production/eager
都失败：

```text
Byte-v2 native prefill direct cache update failed: invalid slot mapping
error_code=2
fallback_pool_used=0
fallback_pool_capacity=512
```

这不是 OOM，也不是单纯 cudagraph 问题。需要先修复，否则无法测试更高 batch 和更接近
capacity 的场景。

#### 可能原因

ByteV2 native prefill direct cache update 可能假设 slot mapping 满足更强条件：

```text
所有 token 都有有效 slot
slot mapping 连续或单调
没有 padding / graph warmup fake slots
batch 内 chunked prefill 的 slot block 边界简单
```

`max_num_batched_tokens=32768`、`batch=8` 和 warmup/cudagraph 捕获会产生更复杂的
slot mapping，触发 kernel 侧 error_code=2。

#### 修复方案

1. 在 Python/backend 层保存失败时的 slot mapping 统计，而不是打印全量 tensor：

   ```text
   num_tokens
   min/max slot
   negative slot count
   out-of-range slot count
   monotonic breaks
   block index min/max
   first/last 64 slot values
   ```

2. 在 native prefill direct kernel 中支持合法的 padding/无效 slot：

   ```text
   slot < 0: skip
   slot >= num_gpu_blocks * block_size: set sticky error
   valid slot: encode/update ByteV2 page
   ```

3. 对 chunked prefill/batched prefill 明确使用 `slot_mapping[token_idx]` 计算 physical
   block 和 block offset，不依赖 token 在 batch 内连续。
4. cudagraph warmup 或 fake input 的无效 slot 不能污染真实 fallback pool 计数。
5. 保留 deferred sticky error，不恢复每 token/layer host sync。

#### 验证

先做小测试：

```text
tests/v1/attention/test_byte_v2_decode.py::test_byte_v2_reshape_and_cache_prefill_slot_mapping_cuda
```

覆盖：

```text
连续 slot
非连续 slot
带 -1 padding slot
跨 block slot
batch=8/chunked prefill slot
```

再跑 E2E：

```text
p4096/b8/d128 eager
p4096/b8/d128 production/cudagraph
```

保留门槛：

```text
b8 不再 invalid slot mapping
p4096/b4/d128 不退化超过 1%
p512/b4/d128 仍 >= 90% raw
```

### Track 2: production path CUDA event timing

在继续改 kernel 前，必须把长 context 的 E2E 分段时间打清楚。nsys 对
multiprocessing worker 注入不稳定，下一步应增加低开销、默认关闭的 CUDA event timing。

#### 开关

```text
VLLM_BYTE_V2_PROFILE_EVENTS=1
VLLM_BYTE_V2_PROFILE_EVENTS_MAX_STEPS=...
```

#### 需要统计的 ByteV2 分段

```text
prefill direct cache update
decode append/cache update
decode attention stage1
decode attention reduce
sampling/output copy
```

统计粒度：

```text
per layer
per decode step
total / avg_us_per_launch / p50 / p95
```

benchmark JSON 中追加：

```json
"byte_v2_event_profile": {
  "decode_stage1_ms": ...,
  "decode_reduce_ms": ...,
  "decode_cache_update_ms": ...,
  "prefill_cache_update_ms": ...,
  "stage1_avg_us": ...,
  "reduce_avg_us": ...
}
```

#### 需要跑的 workload

```text
p512/b4/d128
p4096/b1/d128
p4096/b4/d128
p7168/b1/d128
```

保留门槛：

```text
profile 开关关闭时零开销
profile 开关开启时 E2E 额外开销 < 3%
JSON 能稳定输出 stage1/reduce/cache update
```

### Track 3: 长 context CUTE stage1 early-exit + NCU 复测

之前 early-exit 主要覆盖 p512/p2048。长 context 需要重新测当前 production best path。

#### early-exit 模式

```text
mode0: full stage1
mode1: load/decode K/V 后退出
mode2: load/decode K/V + QK 后退出
mode3: load/decode K/V + QK + softmax 后退出
mode4: full stage1 + PV/write partial
```

#### workload

```text
p4096/b1/split8
p4096/b4/split8
p7168/b1/split8
```

#### NCU 指标

```text
stage1 duration
integer instructions
memory instructions
long scoreboard
global sectors/request
global excessive sectors
shared excessive wavefronts
registers/thread
eligible warps/scheduler
achieved occupancy
```

输出表：

| workload | load/decode | QK | softmax | PV/write | total |
| --- | ---: | ---: | ---: | ---: | ---: |
| p4096/b1 | | | | | |
| p4096/b4 | | | | | |
| p7168/b1 | | | | | |

只有知道 `load/decode`、`PV/write partial`、`reduce` 谁随 context 增长最快，后续
kernel patch 才有明确目标。

### Track 4: fused compressed attention 的长 context 优化

#### 4.1 自适应 page parallel / split-K policy

不要再盲目固定 `split_k=8` 或 `split_k=128`。长 context 需要根据 batch、context、
kv heads 和 SM 数选择 page chunk 并行度。

建议策略：

```text
target_ctas = SM_count * waves_per_sm
base_work = batch * num_kv_heads
pages = ceil(context_len / block_size)
split_k = clamp_power2(ceil(target_ctas / base_work), min=1, max=128)
page_chunk = ceil(pages / split_k)
```

需要同时约束 reduce 开销：

```text
如果 split_k 增加导致 reduce_ms / stage1_ms > 15%，降低 split_k。
如果 eligible warps/scheduler 很低且 reduce 占比低，提高 split_k。
```

实验矩阵：

```text
p4096/b1/d128: split_k=8,16,32,64,128
p4096/b4/d128: split_k=4,8,16,32,64
p7168/b1/d128: split_k=16,32,64,128
```

保留门槛：

```text
stage1 + reduce total 下降 >= 5%
p4096/b1 或 p4096/b4 E2E 提升 >= 3%
p512/b4 不退化超过 1%
```

#### 4.2 Stream-K / 多 page chunk 内部累积，减少 partial 写回

如果 Track 3/4.1 显示 split_k 增大后 reduce 和 partial write 变重，则不要继续提高
split_k，而是让一个 CTA 或 CTA group 在内部流式处理多个 page chunk，使用 online
softmax 累积后只写一次 partial。

目标：

```text
减少 partial output bytes
减少 reduce launch 输入规模
保持足够 page-level parallelism
```

首版只支持：

```text
Llama-3 8B shape
head_size=128
q_per_kv=4
block_size=16
compressed-only/no-arena
```

保留门槛：

```text
p4096/b1 stage1+reduce 下降 >= 8%
p7168/b1 stage1+reduce 下降 >= 10%
E2E 至少 +5%
```

#### 4.3 长 context 专用 compressed payload layout-v3

当前 ByteV2 payload 在长 context 下暴露出 load/decode 和 uncoalesced sectors 问题。
如果 Track 3 显示 load/decode 仍占 stage1 的最大比例，就做只接 microbench 的
layout-v3：

```text
每 tile 256 elements = 128 pairs
每 16 pairs 一个 stripe:
  low[32]
  code[16]
  pad/alignment
```

K/V 使用不同物理顺序：

```text
K: 按 QK/WMMA-B 消费顺序存
V: 按 PV row-major 消费顺序存
```

实验顺序：

```text
decode_page_wmma_microbench
decode-only p4096/p7168
E2E p4096/b1,b4
```

保留门槛：

```text
microbench >= +5%
NCU: global excessive sectors 下降
NCU: memory inst 或 long scoreboard 下降
E2E >= +3%
```

如果只在 microbench 有 1%-2%，不接 production。

#### 4.4 metadata/fallback hot path 分离

当前 `outlier_arena=off`，但 stage1 仍可能保留 fallback/tile metadata 检查。对长
context，metadata 检查随 page 数放大。

保守方案：

```text
kernel A: compressed-only, no raw fallback, no tile fallback
kernel B: compressed + sparse raw/tile fallback metadata
```

调度层根据每层/每 request 的 compact metadata 判断：

```text
if no fallback pages and no tile fallback:
    launch kernel A
else:
    launch kernel B
```

注意：之前 tile bitmap 默认没有 E2E 收益，所以本轮不重新启用 bitmap。这里只做
coarse kernel selection，避免 hot path 内反复检查。

保留门槛：

```text
p4096/b1 stage1 下降 >= 3%
p4096/b4 E2E 下降/持平不能接受
额外 launch/reduce 成本不能抵消收益
```

### Track 5: hybrid active raw staging oracle

如果 fused compressed stage1 的优化收益不足，需要做一个长 decode 专用 oracle：

```text
compressed KV 常驻 HBM
active decode batch 的长 prefix pages 临时解压到 raw BF16 staging
decode attention 走 raw FlashAttention/FlashInfer fast path
decode 新 token 同时写 compressed cache 和 active raw staging
inactive sequences 只保留 compressed cache
```

这条路线牺牲一部分活跃 batch 的临时显存，但保留“非活跃 KV 常驻压缩”的 capacity
优势。它适合长 decode，因为同一批 prefix pages 会被后续很多 decode step 重复读取，
一次 staging 可以被多个 output token 摊销。

#### staging 显存估算

Llama-3 8B raw KV 全层每 token 约：

```text
32 layers * 2(K,V) * 8 kv_heads * 128 dim * 2 bytes = 131,072 bytes/token
```

因此：

```text
p4096/b1 active raw staging ~= 512 MiB
p4096/b4 active raw staging ~= 2 GiB
p7168/b4 active raw staging ~= 3.5 GiB
```

这对单个 active batch 可能可接受，但不能无条件默认开启。

#### oracle 实验

先不接 production，只做 oracle：

```text
ByteV2 compressed prefill
decompress active KV pages -> BF16 staging
decode 调 raw attention
```

对比：

```text
current fused ByteV2 CUTE stage1
raw vLLM
ByteV2 staging oracle
```

workload：

```text
p4096/b1/d128,d512,d1024
p4096/b4/d128,d256
p7168/b1/d128,d512
```

判断：

| 结果 | 结论 |
| --- | --- |
| staging oracle 接近 raw | 当前自研 fused stage1 是主瓶颈，staging 可作为长 decode 系统路线 |
| staging oracle 仍慢 | 解压/staging 成本太高，必须继续 fused compressed stage1 |
| staging 在 d512/d1024 才快 | 需要 decode_len 阈值策略 |

#### production 策略

只有 oracle 成立后再考虑：

```text
if context_len >= 4096
and expected_decode_len >= threshold
and free_memory >= active_staging_bytes * safety_factor
and batch_size <= staging_batch_limit:
    use active raw staging
else:
    use fused compressed attention
```

保留门槛：

```text
p4096/b1/d512 ByteV2 staging >= 90% raw
p4096/b4/d256 ByteV2 staging >= 90% raw
staging memory 有明确上限和回收路径
prefix cache / eviction 不破坏 correctness
```

### Track 6: capacity-bound 证明实验

在修复 `b8` 后，才做真正 capacity-bound 实验。目标不是只看 tok/s，而是证明 raw 接近
或达到 KV capacity，而 ByteV2 能容纳更多 resident tokens。

候选 workload：

```text
p4096/b16/d128
p7168/b8/d128
p2048/b32/d64
```

记录：

```text
raw 是否 OOM / preemption / lower max concurrency
ByteV2 是否能运行
resident tokens
output tok/s
per-request latency
fallback pool exhaustion
```

如果 raw 不能运行而 ByteV2 能运行，报告 capacity advantage；如果 raw 能运行，则仍按
tok/s 和 stage timing 比较。

### 执行顺序

后续按这个顺序执行：

1. 复跑 `p512/b4/d128`，确认短 context 仍有 `90%+ raw`。
2. 修复 `p4096/b8` 的 ByteV2 prefill slot mapping。
3. 增加 CUDA event timing hook，得到 production E2E 分段时间。
4. 做长 context CUTE stage1 early-exit + NCU。
5. 做 adaptive split/page parallel sweep。
6. 根据 profile 结果选择：

   ```text
   load/decode 最大 -> layout-v3 / decoder layout
   reduce/partial 最大 -> Stream-K / partial write 优化
   metadata 最大 -> hot path kernel 分离
   fused stage1 仍远慢 -> active raw staging oracle
   ```

7. 修复 b8 后再跑 capacity-bound batch/context 实验。

### 阶段目标

| 阶段 | 目标 |
| --- | --- |
| P0 | `p512/b4/d128` 保持 `>=90% raw` |
| P1 | `p4096/b8/d128` ByteV2 production/eager 跑通 |
| P2 | `p4096/b1,b4` 有 production event timing 分解 |
| P3 | `p4096/b4/d128` 从 `70.9% raw` 提升到 `>=80% raw` |
| P4 | `p7168/b1/d128/d512` 从 `~60% raw` 提升到 `>=70% raw` |
| P5 | active staging oracle 在长 decode 达到 `>=90% raw` 或明确失败 |

任何 patch 如果没有达到对应门槛，或只在 microbench 改善但 E2E/NCU 不支持，都不作为
默认路径保留。

## C.31 b8 Slot-Mapping Fix And Latest E2E Results

Date: 2026-06-14

### Patch

The `p4096/b8` ByteV2 failure was caused by the prefill direct cache update
validator treating a direct-fast-path eligibility miss as a hard cache update
error. The direct prefill path only supports groups where each 16-token input
chunk maps to one complete physical block with offsets `0..15`. Larger batch
or longer prompt runs can produce valid `slot_mapping` layouts that do not
match this direct-path assumption.

Changes:

```text
csrc/libtorch_stable/cache_kernels.cu
  byte_v2_validate_prefill_direct_blocks_kernel:
    records direct_ineligible instead of invalid slot mapping

  host prefill direct branch:
    runs direct encode only when direct_ineligible == 0
    otherwise falls back to the existing compressed-only generic update path

tests/v1/attention/test_byte_v2_ops.py
  adds a CUDA regression test for split direct groups
```

`VLLM_BYTE_V2_PREFILL_DIRECT_SKIP_VALIDATION_SYNC=1` no longer skips the direct
eligibility decision. This is intentional: skipping the decision was unsafe for
valid but non-direct-friendly mappings.

Validation:

```text
uv pip install -e . --torch-backend=auto
  success, build time about 16m04s

CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest \
  tests/v1/attention/test_byte_v2_ops.py::test_byte_v2_reshape_and_cache_op_falls_back_for_split_direct_group_cuda \
  -q
  1 passed
```

### Results

Short-context sentinel after the fix:

```text
benchmarks/profiles/step55_sentinel_after_fix_p512_b4_d128_d256.json
```

| workload | raw tok/s | ByteV2 tok/s | ByteV2/raw |
| --- | ---: | ---: | ---: |
| p512/b4/d128 | 133.354 | 118.225 | 88.66% |
| p512/b4/d256 | 132.657 | 121.061 | 91.26% |

b8 smoke:

```text
benchmarks/profiles/step55_b8_smoke_p4096_b8_d16.json
p4096/b8/d16 ByteV2 production: 93.397 tok/s
fallback any_exhausted=false, max_next_slot=24 / 512
```

b8 production:

```text
benchmarks/profiles/step55_b8_production_p4096_b8_d128.json
```

| workload | raw tok/s | ByteV2 tok/s | ByteV2/raw |
| --- | ---: | ---: | ---: |
| p4096/b8/d128 production | 227.251 | 136.637 | 60.13% |

b8 eager:

```text
benchmarks/profiles/step55_b8_eager_p4096_b8_d128.json
```

| workload | raw tok/s | ByteV2 tok/s | ByteV2/raw |
| --- | ---: | ---: | ---: |
| p4096/b8/d128 eager | 222.119 | 134.664 | 60.63% |

Fallback pool was not exhausted in either b8 run:

```text
max_capacity=512
max_next_slot=80
total_capacity=16384
total_next_slot=2560
any_exhausted=false
```

### Conclusion

P1 correctness is now satisfied: `p4096/b8/d128` runs in both production and
eager modes without the previous `invalid slot mapping` failure.

Performance is still not acceptable for long-context b8: ByteV2 is about
`60%` of raw. The next step should not be another blind kernel tweak. It should
be production event timing for:

```text
attention stage1
reduce
cache update
host/API
sampling/GEMM
```

The key question is how much of the b8 gap comes from decode attention stage1
versus the generic prefill/cache update path used when direct prefill is not
eligible.

## C.32 K/V load/decode 细粒度拆分实验

### 目的

上一轮 profile 只能看到 `load/decode K/V` 是 CUTE metadata stage1 的最大段，
但没有拆开 compressed payload 读取、ByteV2 bit decode、shared/layout 各自的
成本。本轮新增 profile-only early-exit mode：

```text
mode5: metadata/status traversal
mode6: metadata + compressed/raw payload read
mode7: metadata + payload read + ByteV2 bit decode to registers
mode1: existing load/decode + shared layout
mode0: full stage1
```

实现注意点：

- mode5/6/7 只在 `VLLM_BYTE_V2_DECODE_CUTE_STAGE1_EARLY_EXIT` 显式设置时
  生效，不影响 production 默认路径。
- 为避免编译器消掉读取，mode5/6/7 使用轻量 XOR sink 写入 partial output。
- 第一版 rotate+xor sink 会把较多整数指令混入 mode6/7，已改为纯 XOR 后重测。

### 命令

重编译：

```bash
uv pip install -e . --torch-backend=auto
```

profile：

```bash
CUDA_VISIBLE_DEVICES=0 nsys profile --force-overwrite=true --trace=cuda,nvtx \
  --cuda-graph-trace=node --cuda-event-trace=false \
  --output benchmarks/profiles/fine_kv_split_xor_p512_b4_d128_m5 \
  .venv/bin/python benchmarks/kernels/benchmark_byte_v2_decode_kernel.py \
    --device cuda:0 --batch-size 4 --seq-len 512 --split-k 8 \
    --fallback-ratio 0.03 --num-runs 20 --warmup-runs 5 \
    --skip-correctness --cute-stage1-auto \
    --cute-stage1-early-exit-mode 5
```

输出文件：

```text
benchmarks/profiles/fine_kv_load_decode_split_report.md
benchmarks/profiles/fine_kv_load_decode_split_summary.json
benchmarks/profiles/fine_kv_split_xor_{workload}_m{0,1,5,6,7}.nsys-rep
benchmarks/profiles/fine_kv_split_xor_{workload}_m{0,1,5,6,7}_kern_cuda_gpu_kern_sum.csv
```

### Stage1 avg time

| workload | mode5 metadata | mode6 +payload read | mode7 +reg decode | mode1 +shared | mode0 full | reduce avg(mode0) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| p512_b4_d128 | 19.54 us | 43.33 us | 44.10 us | 43.94 us | 56.97 us | 3.25 us |
| p512_b4_d256 | 19.55 us | 42.90 us | 44.21 us | 43.96 us | 56.92 us | 3.24 us |
| p2048_b1_d32 | 65.15 us | 124.57 us | 127.95 us | 128.93 us | 172.87 us | 2.94 us |

### Derived incremental time

| workload | metadata/q/traversal | payload read | register decode | shared/layout estimate | rest of attention |
| --- | ---: | ---: | ---: | ---: | ---: |
| p512_b4_d128 | 19.54 us | 23.79 us | 0.77 us | -0.17 us | 13.03 us |
| p512_b4_d256 | 19.55 us | 23.35 us | 1.31 us | -0.25 us | 12.96 us |
| p2048_b1_d32 | 65.15 us | 59.41 us | 3.39 us | 0.98 us | 43.94 us |

### 验证

```bash
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest \
  tests/v1/attention/test_byte_v2_decode.py::test_byte_v2_paged_decode_attention_op_cute_stage1_metadata_cuda \
  -q
```

结果：

```text
1 passed
```

mode0 correctness smoke：

```text
seq_len=512 split_k=8 fallback=0.030 median_us=92.16
max_abs_diff=0.0
```

### 结论

1. 真实瓶颈不是 ByteV2 exponent/low/code 的寄存器 bit reconstruction。register
   decode 增量只有约 `0.8-1.3 us`（p512/b4）和 `3.4 us`（p2048/b1）。
2. 最大的可见增量是 compressed/raw payload read：p512/b4 约 `23-24 us`，
   p2048/b1 约 `59 us`。这说明当前 stage1 仍主要卡在 K/V payload 访问模式、
   memory dependency 和 coalescing，而不是纯解码算术。
3. mode1-mode7 的 shared/layout estimate 接近 0，p512 下甚至为轻微负值。
   这不是说明 shared 完全免费，而是说明当前 profile-only mode7 的 sink/控制流
   和 mode1 不完全同构；只能把 shared/layout 解读为“没有 payload read 明显大”。
4. full-mode0 减 mode1 的 rest attention 仍明显：p512/b4 约 `13 us`，
   p2048/b1 约 `44 us`。所以后续不能只优化 payload layout，也要继续看 full
   attention skeleton 的 QK/softmax/PV 随 context 增长的开销。

### 下一步

优先方向应从“减少解码指令数”转为：

```text
1. payload layout / lane-stripe coalescing
2. page/chunk scheduling 降低 long scoreboard
3. full stage1 attention skeleton 中 QK/softmax/PV 的长 context scaling
4. 如果 layout 改动不能显著降低 mode6-mode5，则回到 FlashInfer-compatible loader/staging 路线
```

## C.33 compressed payload 读取优化计划

### 背景

C.32 已经把 CUTE metadata stage1 的 K/V 路径拆开：

```text
p512/b4:   payload read ≈ 23-24 us, register decode ≈ 1 us
p2048/b1:  payload read ≈ 59 us,    register decode ≈ 3.4 us
```

因此下一步不要继续优先减少 bit decode 指令，而应直接降低 compressed payload
读取成本。当前格式的主要问题：

```text
tile payload = base 1B + fallback 1B + low[256] + code[128] = 386B
tile_start = page_header(16B) + tile_idx * 386B
```

`386B` 不是 32/64/128B 对齐步长，导致每个 tile 的 low/code 起点相对 sector
持续漂移。当前 low 读取还通过两个 U8 load 组成一个 U16：

```cpp
return ptr[0] | (ptr[1] << 8);
```

Step49 曾经显示 aligned U16 可以显著减少 global excessive sectors，但当时没有
端到端收益。C.32 的 fine split mode 可以更直接判断它是否真的降低 payload read
段，因此重新做一个更窄的 A/B。

### 实验 A1：opt-in aligned U16 low payload loader

目标：只替换 compressed tile fastpath 中的 `low` pair 读取：

```text
baseline: two U8 loads -> combine U16
variant:  reinterpret_cast<const uint16_t*> 直接读 U16
```

约束：

- 只由 `VLLM_BYTE_V2_DECODE_ALIGNED_U16_PAYLOAD_LOAD=1` 开启。
- 默认 production 路径不变。
- 只先看 CUTE stage1 的 compressed fastpath 和 mode5/6/7。
- 保留门槛：

```text
payload read = mode6 - mode5 下降 >= 5%
mode7 - mode6 不明显变差
mode0 full stage1 不变差
correctness max_abs_diff = 0
```

如果 mode6-mode5 下降但 mode0 不下降，说明 payload read 已不是唯一瓶颈，需要
转向 layout/WMMA 或 page-parallel，而不是保留这个开关作为默认。

### 实验 A2：32B/64B aligned tile-stride layout microbench

如果 A1 无法降低 mode6-mode5，说明不改物理格式很难继续提升读取。下一步只在
microbench 中做 padded tile stride：

```text
tile_stride = 416B 或 448B
tile payload 内部保持 base/fallback/low/code
tile_start 对齐到 32B/64B 边界
```

这会牺牲压缩率：

```text
416B tile: page ≈ 16 + 128 * 416 = 53264B, ratio ≈ 0.813
448B tile: page ≈ 57360B, ratio ≈ 0.875
```

所以必须先用 microbench/NCU 证明 payload read 和 long scoreboard 明显下降，
否则不能进入 production。

### 实验 A3：lane-stripe layout-v3

如果 padded stride 有效，再做真正 layout-v3，而不是简单 padding：

```text
按 warp/lane group 存 low/code stripe
K 使用 QK/WMMA-B 消费顺序
V 使用 PV row-major 消费顺序
尽量让每个 warp 的 low/code 读取变成连续 32B/64B sector
```

这一步需要同时改 encode、decode、decompress/reference 和 allocator page size，
风险较高。只有 A1/A2 证明 payload read 确实可被布局改善后再做。

### A1 结果：aligned U16 payload loader

本轮实现了 `VLLM_BYTE_V2_DECODE_ALIGNED_U16_PAYLOAD_LOAD=1` /
`--aligned-u16-payload-load`，只影响 CUTE stage1 compressed tile fastpath，
默认不开启。

profile 文件：

```text
benchmarks/profiles/aligned_u16_payload_a1_report.md
benchmarks/profiles/aligned_u16_payload_a1_summary.json
benchmarks/profiles/aligned_u16_payload_a1_{baseline,aligned}_{p512_b4,p2048_b1}_m{0,1,5,6,7}.nsys-rep
```

#### Stage1 avg time

| workload | variant | mode5 metadata | mode6 +payload | mode7 +reg decode | mode1 +shared | mode0 full | reduce avg |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| p512_b4 | baseline | 19.73 us | 42.64 us | 44.11 us | 44.59 us | 57.38 us | 3.30 us |
| p512_b4 | aligned | 19.76 us | 42.26 us | 44.40 us | 44.63 us | 57.53 us | 3.32 us |
| p2048_b1 | baseline | 65.48 us | 130.32 us | 133.99 us | 137.90 us | 182.25 us | 2.62 us |
| p2048_b1 | aligned | 65.47 us | 127.98 us | 135.30 us | 137.91 us | 182.19 us | 2.61 us |

#### Derived incremental time

| workload | variant | payload read | register decode | shared/layout est. | rest attention | full stage1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| p512_b4 | baseline | 22.91 us | 1.48 us | 0.48 us | 12.79 us | 57.38 us |
| p512_b4 | aligned | 22.50 us | 2.15 us | 0.23 us | 12.90 us | 57.53 us |
| p2048_b1 | baseline | 64.84 us | 3.67 us | 3.90 us | 44.35 us | 182.25 us |
| p2048_b1 | aligned | 62.51 us | 7.32 us | 2.61 us | 44.28 us | 182.19 us |

#### Delta

| workload | payload read delta | payload read ratio | register decode delta | full stage1 delta | full stage1 ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| p512_b4 | -0.41 us | 0.982x | +0.67 us | +0.15 us | 1.003x |
| p2048_b1 | -2.33 us | 0.964x | +3.64 us | -0.06 us | 1.000x |

验证：

```text
aligned mode0 correctness: max_abs_diff=0.0
test_byte_v2_paged_decode_attention_op_cute_stage1_metadata_cuda: 1 passed
```

结论：

1. aligned U16 对 payload read 有小幅改善：p512/b4 `-1.8%`，p2048/b1
   `-3.6%`。
2. 没达到 A1 设定的 `>=5%` 保留门槛。
3. register decode 段上升，抵消了 payload read 的小收益。
4. full stage1 基本不变，p512/b4 略退化，p2048/b1 基本持平。
5. 因此 aligned U16 不能作为默认优化。后续如果继续优化读取，应做 A2
   physical layout/padded stride 或 A3 lane-stripe layout，而不是继续替换
   scalar low-byte loader。

## C.34 CUTE metadata: no-outlier specialization 与 warp broadcast 实验

本轮针对长上下文 `p4096/b4/d64` 中最大的 `metadata/status traversal`
成本，尝试两项优化：

1. `base/status` warp broadcast：每个 warp 只让 lane0 读取 tile/page header，
   再用 `__shfl_sync` 广播。
2. tile-fallback-only no-outlier CUTE specialization：当 CUTE metadata stage1
   打开、存在 sparse/tile fallback、但没有 outlier arena/block flag/tile bitmap
   时，实例化 `no_outlier_metadata=true` 的专用 kernel，用 `if constexpr`
   删除 outlier overlay 和 bitmap/meta 检查。

实验文件：

```text
benchmarks/profiles/no_outlier_specialized_metadata_report.md
benchmarks/profiles/warp_broadcast_no_outlier_p4096_b4_fb003_m{0,5,6}.json
benchmarks/profiles/no_outlier_specialized_p4096_b4_fb003_m{0,5,6}.json
benchmarks/profiles/no_outlier_specialized_e2e_bytev2_p4096_b4_d64_measured.nsys-rep
benchmarks/profiles/no_outlier_specialized_e2e_bytev2_p4096_b4_d64_measured_kern_cuda_gpu_kern_sum_nvtx=byte_v2_bench_measured.csv
```

### 结果 1：base/status warp broadcast 不保留

| mode | baseline | warp broadcast | 结果 |
| --- | ---: | ---: | ---: |
| mode5 metadata/status | 250.86 us | 384.00 us | 退化 53.1% |
| mode6 + payload read | 317.44 us | 475.14 us | 退化 49.7% |
| mode0 full decode op | 450.56 us | 641.02 us | 退化 42.3% |

结论：不保留。当前 header/status 读取虽然重复，但改成 lane0 load +
`__shfl_sync` 后引入了额外 shuffle dependency 和 warp 内串行化，收益远小于开销。
后续除非 tile header 物理布局改变，否则不要继续做这个方向。

### 结果 2：no-outlier CUTE metadata specialization 保留

| mode | baseline | no-outlier specialized | delta |
| --- | ---: | ---: | ---: |
| mode5 metadata/status | 250.86 us | 243.71 us | +2.9% |
| mode6 + payload read | 317.44 us | 309.25 us | +2.6% |
| mode0 full decode op | 450.56 us | 436.22 us | +3.2% |

E2E `p4096/b4/d64`，Nsight measured range：

| metric | baseline | no-outlier specialized | delta |
| --- | ---: | ---: | ---: |
| measured NVTX wall time | 5.1249 s | 5.0753 s | +1.0% |
| benchmark elapsed | 5.1175 s | 5.0679 s | +1.0% |
| output tokens/s | 50.02 | 50.51 | +1.0% |
| stage1 kernel total | 894.5 ms | 860.2 ms | +3.8% |
| stage1 avg launch | 443.7 us | 426.7 us | +3.8% |
| total measured kernel time | 5036.0 ms | 4994.7 ms | +0.8% |

验证：

```text
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m pytest \
  tests/v1/attention/test_byte_v2_decode.py::test_byte_v2_paged_decode_attention_op_cute_stage1_metadata_cuda \
  -q

1 passed
```

结论：保留 no-outlier specialization。收益不大，但满足保留门槛：
`fallback=0.03` 下 mode5/mode0 均下降，E2E 没有退化，并且 stage1 total
下降约 `34 ms`。
