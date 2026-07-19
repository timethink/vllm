# ByteV2 Speculative Decoding Profile

Date: 2026-07-13  
GPU: NVIDIA A40 (SM86)  
Model: local Llama-3.1-8B-Instruct, BF16  
Backends: ByteV2 current kernel and raw vLLM FlashAttention 2

## Goal

Determine whether the current ByteV2 paged-decode kernel can accelerate
speculative decoding without changing the kernel.

The benchmark uses CPU n-gram drafting so no draft model is involved. The
periodic token prompt gives 100% draft acceptance, making this an optimistic
upper bound for speculative decoding. All primary comparisons force eager mode
so CUDA graph support does not confound the attention-path comparison.

Artifacts:

- Driver: `scripts/byte_v2_speculative_profile.py`
- Correctness: `profile/bytev2_spec_correctness_a40_20260713/results.jsonl`
- Context sweep: `profile/bytev2_spec_context_sweep_a40_20260713/results.jsonl`
- Batch 4: `profile/bytev2_spec_batch4_a40_20260713/results.jsonl`
- Batch 16: `profile/bytev2_spec_batch16_a40_20260713/results.jsonl`

## Current Dispatch

For normal decode, `max_query_len == 1` enters `_run_paged_decode()` directly.
Speculative verification has `max_query_len == 1 + num_speculative_tokens`, so
it enters `_forward_prefill_from_cache()` instead. That method expands one
request with query length `Q` into `Q` virtual sequences with the same block
table and increasing sequence lengths.

This is causally correct, but every virtual sequence independently reads and
decodes the same compressed prefix. The current kernel therefore does not turn
GQA4 with `Q=4` into a shared-KV M16 tile. It executes four independent M4
paths.

## Correctness

At context 256 and greedy sampling, ByteV2 and raw FA2 each produced exactly
the same output tokens as their own non-speculative reference for draft-token
counts 1, 3, and 7. All cases had 100% acceptance.

This validates the tested cached-context path. It does not cover arbitrary
rejection patterns or partial-prefix/chunked-prefill behavior.

## Context Sweep

Output throughput in tokens/s, batch 1, 64 output tokens:

| Context | Byte Q1 | Byte Q2 | Byte Q4 | Byte Q8 | FA2 Q1 | FA2 Q2 | FA2 Q4 | FA2 Q8 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1K | 30.48 | 42.60 | 76.54 | 128.15 | 34.83 | 59.65 | 115.24 | 215.79 |
| 4K | 27.25 | 35.29 | 56.43 | 81.52 | 33.84 | 54.86 | 105.75 | 198.00 |
| 8K | 23.12 | 28.61 | 41.36 | 53.24 | 32.67 | 49.51 | 95.30 | 178.04 |
| 16K | 17.50 | 20.54 | 26.86 | 31.71 | 30.74 | 41.41 | 79.57 | 147.63 |

ByteV2 throughput divided by raw FA2 throughput:

| Context | Q1 | Q2 | Q4 | Q8 |
| ---: | ---: | ---: | ---: | ---: |
| 1K | 0.875 | 0.714 | 0.664 | 0.594 |
| 4K | 0.805 | 0.643 | 0.534 | 0.412 |
| 8K | 0.708 | 0.578 | 0.434 | 0.299 |
| 16K | 0.569 | 0.496 | 0.337 | 0.215 |

Relative to ByteV2 normal decode, speculative decoding still helps at batch 1
under perfect acceptance. Q4 speedup falls from 2.51x at 1K to 1.53x at 16K;
Q8 speedup falls from 4.20x to 1.81x. Raw FA2 preserves much more of the ideal
gain: Q4 is 3.31x to 2.59x and Q8 is 6.20x to 4.80x.

The widening gap with both context length and query length is direct evidence
that the current ByteV2 path repeats compressed-prefix decode for every verify
token.

## Per-Layer Attribution

CUDA-event time per attention layer at context 16K, batch 1:

| Query length | Byte attention | FA2 attention | Ratio | Byte cache update | FA2 cache update |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 765.4 us | 141.5 us | 5.41x | 10.4 us | 9.6 us |
| 2 | 1524.7 us | 483.6 us | 3.15x | 231.2 us | 11.3 us |
| 4 | 2763.9 us | 482.3 us | 5.73x | 233.8 us | 11.4 us |
| 8 | 5262.4 us | 482.2 us | 10.91x | 240.5 us | 11.4 us |

For FA2, the Q2/Q4/Q8 verify kernel time is almost constant. For ByteV2 it is
approximately linear in Q. At 16K Q4, attention is about 12 times larger than
the ByteV2 cache-update cost, so shared-prefix attention is the first target.

At 1K Q4, ByteV2 attention is 363.4 us and cache update is 229.1 us. A fused
multi-token cache update matters more at short context, but it cannot solve the
long-context scaling problem.

The ByteV2 multi-token update uses prepare, hydrate, append, commit, and release
staging operations. It costs about 230 us/layer for batch-1 verification,
versus about 11 us/layer for raw `reshape_and_cache_flash`. Single-token
ByteV2 update remains close to raw because it uses the dedicated fused update.

## Host Path

ByteV2 Q4 `forward()` CPU call time grows from 828 us at 1K to 3167 us at 16K;
raw FA2 remains about 98 us. The ByteV2 cached-context dispatch contains Python
request loops, tensor allocation, and GPU tensor `.item()` checks. In
particular, `_prefill_has_cached_context()` synchronizes before the cached
prefill path and prevents the host from enqueueing layers efficiently.

This host path should be rewritten even before a production speculative
kernel is enabled. It is not the main long-context CUDA cost, but it makes the
current fallback unsuitable for serving.

## Batch Sweep

Context 4K, Q4 versus Q1 within each batch. Batch 1 uses 64 output tokens;
batches 4 and 16 use 32 output tokens per request.

| Batch | Byte Q1 | Byte Q4 | Byte speedup | FA2 Q1 | FA2 Q4 | FA2 speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 27.25 | 56.43 | 2.07x | 33.84 | 105.75 | 3.13x |
| 4 | 62.42 | 89.28 | 1.43x | 116.68 | 337.65 | 2.89x |
| 16 | 94.40 | 103.28 | 1.09x | 333.79 | 776.07 | 2.33x |

At batch 16, ByteV2 Q4 attention is 12.70 ms/layer, about 4.39 times its Q1
attention. FA2 Q4 is 1.00 ms/layer, about 2.03 times its Q1 attention while
producing four times as many query outputs. Sequence-level parallelism does not
hide ByteV2 repeated decode.

## Prefix-Cache Failures

The initial sweep also found a separate robustness issue:

- A 4K request with about 1K cached prefix tried to allocate a 2.95 GiB
  split-K workspace.
- An 8K request with about 1K cached prefix tried to allocate 13.91 GiB.
- Four concurrent 4K chunked prefills tried to allocate another 3.96 GiB and
  failed on the A40.

`_forward_prefill_from_cache()` expands every uncached query token into a
paged-decode row, while the split-K workspace scales with query rows,
partitions, heads, and head dimension. The benchmark now isolates cases with a
unique first block and fills batch prefixes one request at a time, but the
production limitation remains.

## Decision

The current kernel can execute speculative verification and can accelerate
ByteV2 relative to ByteV2 normal decode at batch 1 with perfect acceptance. It
does not provide a competitive speculative path against raw FA2. The gap gets
worse exactly where compression should help most: long context, larger verify
Q, and larger batch.

Do not tune speculative dispatch around the current virtual-sequence path.
Keep the current kernel for Q1 decode and use a separate specialized path:

1. Make cached-context metadata construction GPU-resident and batched. Remove
   Python request loops, `.item()` synchronization, and unbounded workspace.
2. Add `byte_v2_spec_verify_attention` for Q2/Q4, with Q4 as the first target.
   Decode each old-prefix K/V tile once and consume it with
   `Q * q_heads_per_kv` query rows. Only the short speculative tail needs
   per-row causal masking.
3. After shared-prefix attention works, add a fused multi-token cache update.
4. Preserve the current Q1 kernel and dispatch speculative verification only
   for uniform cached batches with supported Q.

For a Q4 prototype at 16K, require at least a 2x reduction from the current
2.76 ms/layer verify attention and at least a 20% E2E improvement over the
current ByteV2 speculative path. If it misses either threshold, do not retain
the specialized kernel. Even a successful prototype must still be compared to
the raw FA2 reference of about 0.48 ms/layer.

## Q4 Prototype Result (2026-07-14)

The dedicated Q4 prototype is implemented and remains disabled by default. Set
`BYTE_V2_SPECULATIVE_VERIFY_Q4=1` to enable it. The existing Q1 kernel and the
generic cached-prefill path are unchanged.

The new `byte_v2_speculative_verify_q4` op reuses the FA2-direct M16 body. Its
internal workspace is `[request, 128 virtual_heads, partitions]`, which has the
same number of elements as `[request * 4, 32 heads, partitions]`. For each KV
head, the 16 M rows map as:

```text
token = row / 4
query_head_in_group = row % 4
causal_seq_len = final_seq_len - 3 + token
```

Compressed K/V staging, QK MMA, online softmax, and PV MMA are shared across
all 16 rows. Only query loading, the per-row causal bound, and final output
mapping differ from the Q1 FA2-direct kernel. Unsafe-page flags remain active,
so pages containing outliers or fallback metadata retain the existing guarded
decode behavior.

The backend dispatch is fail-closed. It requires uniform Q4 requests, causal
attention, 32 query heads, 8 KV heads, head dimension 128, the default ByteV2
layout, a complete four-token batch, and valid unsafe-page flags. Unsupported
or mixed batches continue through `_forward_prefill_from_cache()`.

### Correctness

The direct CUDA test covers an unsafe/outlier cache, two split-K partitions,
and four distinct causal sequence lengths. It matches the raw BF16 reference
within the existing ByteV2 tolerance. Existing guarded and Q1 FA2-direct tests
also pass.

At E2E level, old ByteV2, Q4-specialized ByteV2, and raw FA2 produced identical
token IDs for batch 1 and batch 4 at context 4K. All runs retained 100% draft
acceptance and mean acceptance length 4.

### Partition Selection

Reusing the generic FA2-direct heuristic initially selected partition 1024 at
4K and made the new op take 266.1 us. A dedicated 4K sweep measured:

| Partition | Q4 op time |
| ---: | ---: |
| 64 | 134.9 us |
| 128 | 111.0 us |
| 256 | 92.2 us |
| 512 | 142.2 us |

The Q4-specific defaults are now 64 below 2K, 128 at 2K, 256 at 4K-8K, and
512 at 16K and above. `BYTE_V2_DECODE_GQA_PACKED_PARTITION_SIZE` still provides
an explicit override.

### Batch-1 Sweep

These runs use Q4, 32 output tokens, eager mode, and 100% acceptance. Kernel
time is the CUDA-event average for the cached verify attention op.

| Context | Old Byte op | New Byte op | Raw FA2 op | Kernel speedup | Old Byte tok/s | New Byte tok/s | Raw FA2 tok/s | E2E gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1K | 202.7 us | 49.2 us | 42.9 us | 4.12x | 76.64 | 84.06 | 107.18 | 9.7% |
| 4K | 684.8 us | 91.7 us | 130.0 us | 7.46x | 58.02 | 79.36 | 98.30 | 36.8% |
| 8K | 1315.9 us | 193.5 us | 245.7 us | 6.80x | 38.03 | 57.26 | 88.95 | 50.6% |
| 16K | 2583.4 us | 317.9 us | 476.4 us | 8.13x | 24.03 | 42.42 | 73.86 | 76.6% |

The prototype passes both retention gates at 16K: attention improves by 8.13x
and E2E throughput improves by 76.6%. It also passes both gates at 4K and 8K.
At 1K, kernel time improves by more than 2x but E2E gain is only 9.7%, because
fixed update and dispatch costs are a larger share of runtime.

The new compressed verify op is faster than raw FA2 attention at 4K, 8K, and
16K in this benchmark. This does not make ByteV2 E2E faster than raw vLLM: the
remaining non-Q4 work still dominates.

### Batch 4

At context 4K, Q4, and 32 output tokens per request:

| Metric | Old ByteV2 | New ByteV2 | Raw FA2 |
| --- | ---: | ---: | ---: |
| Verify op | 2505.8 us | 334.4 us | 249.6 us |
| Throughput | 95.23 tok/s | 174.99 tok/s | 338.27 tok/s |

The dedicated op is 7.49x faster than the old ByteV2 path and raises E2E
throughput by 83.8%. It is still 1.34x slower than raw FA2 at batch 4, and E2E
ByteV2 reaches only 51.7% of raw FA2 throughput.

### Remaining Bottlenecks

The Q4 shared-prefix attention problem is no longer the primary gap. The next
two measured targets are:

1. The cached-prefix Q16 fallback still costs about 2.46 ms/layer at batch 1,
   versus about 0.13 ms/layer for raw FA2. The benchmark triggers this path
   once per layer while reusing the prompt prefix.
2. ByteV2 multi-token cache update remains about 67 us/layer for Q4 at 4K,
   versus about 11 us/layer for raw FA2. Maintaining unsafe-page flags adds
   another small kernel launch.

The retained next step is therefore not another Q4 mainloop rewrite. It is a
bounded batched cached-prefill path for larger Q, followed by a fused
multi-token compressed-cache update. The Q4 kernel should remain behind its
experimental environment switch until rejection-pattern and mixed-batch tests
are added.

New artifacts:

- `profiles/byte_v2_speculative_q4_ab/old.jsonl`
- `profiles/byte_v2_speculative_q4_ab/new_tuned.jsonl`
- `profiles/byte_v2_speculative_q4_ab/raw_fa2.jsonl`
- `profiles/byte_v2_speculative_q4_ab/old_sweep.jsonl`
- `profiles/byte_v2_speculative_q4_ab/new_sweep.jsonl`
- `profiles/byte_v2_speculative_q4_ab/raw_fa2_sweep.jsonl`
- `profiles/byte_v2_speculative_q4_ab/old_batch4.jsonl`
- `profiles/byte_v2_speculative_q4_ab/new_batch4.jsonl`
- `profiles/byte_v2_speculative_q4_ab/raw_fa2_batch4.jsonl`

## GQA Kernel Family (2026-07-14)

The Q4-only compile-time layout has been generalized to a GQA4 kernel family.
`SpeculativeQueryLen` is a template parameter, so Q2, Q4, and Q8 remain
separate compiled kernels without a runtime Q branch in the hot loop.

The virtual row mapping is now:

```text
virtual_row = query_group_base + row
token = virtual_row / 4
query_head_in_group = virtual_row % 4
causal_seq_len = final_seq_len - (Q - 1) + token
```

The instances are:

| Verify Q | Virtual rows per KV head | CTA groups per KV head |
| ---: | ---: | ---: |
| 2 | 8 | one M16 group with 8 valid rows |
| 4 | 16 | one full M16 group |
| 8 | 32 | two M16 groups |

Q2 and Q4 decode each compressed KV tile once per KV head. Q8 reduces the old
eight independent decode rows to two M16 groups, but still decodes the tile
twice. A true M32 implementation is therefore the next Q8-specific option.

Set `BYTE_V2_SPECULATIVE_VERIFY_GQA=1` to enable Q2/Q4/Q8 dispatch. The older
`BYTE_V2_SPECULATIVE_VERIFY_Q4=1` switch remains compatible and enables only
Q4. Both switches default to off.

### GQA Family Results

Context 4K, batch 1, 32 output tokens, eager mode, and 100% acceptance:

| Q | Old Byte op | New GQA op | Raw FA2 op | Kernel speedup | Old Byte tok/s | New Byte tok/s | Raw FA2 tok/s | E2E gain |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 366.2 us | 71.8 us | 130.4 us | 5.10x | 39.07 | 47.60 | 52.70 | 21.8% |
| 4 | 684.7 us | 91.8 us | 130.1 us | 7.46x | 57.96 | 79.46 | 98.41 | 37.1% |
| 8 | 1285.0 us | 184.2 us | 129.8 us | 6.97x | 77.48 | 116.16 | 170.53 | 49.9% |

Q2 and Q4 kernel time is lower than raw FA2 in this isolated attention timing.
Q8 remains 1.42x slower than raw FA2 because it stages and decodes each KV tile
for two M16 groups. All three variants exceed the 5% retention threshold and
improve E2E throughput by more than 20% over the old ByteV2 path at 4K.

Old ByteV2, the new GQA family, and raw FA2 produced identical output token IDs
for every Q in this sweep. The CUDA tests independently cover Q2/Q4/Q8 on an
unsafe/outlier cache with split-K and row-specific causal bounds.

Artifacts:

- `profiles/byte_v2_speculative_gqa_family_20260714/old.jsonl`
- `profiles/byte_v2_speculative_gqa_family_20260714/new.jsonl`
- `profiles/byte_v2_speculative_gqa_family_20260714/raw_fa2.jsonl`

## Steady-State Step Profile (2026-07-14)

The original E2E number mixed two different workloads: one cached-prefix Q16
step and the repeated Q2/Q4/Q8 speculative verify steps. The profiler now
times every `llm_engine.step()` and reports them separately. The speculative
Q label includes the target token, so prompt lookup with 1/3/7 draft tokens
produces verify Q2/Q4/Q8 respectively.

At context 4K, batch 1, 32 output tokens, eager mode, and 100% acceptance:

| Verify Q | ByteV2 steady tok/s | Raw steady tok/s | Steady gap | ByteV2 Q16 step | Raw Q16 step |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 55.70 | 55.89 | -0.3% | 115.8 ms | 38.6 ms |
| 4 | 108.52 | 111.49 | -2.7% | 116.2 ms | 38.7 ms |
| 8 | 200.34 | 221.62 | -9.6% | 115.7 ms | 38.6 ms |

The dedicated Q2/Q4 attention kernels are therefore not the main steady-state
gap. Q2 is effectively tied with raw and Q4 is within 3%. Q8 remains limited
by two M16 groups. The much larger whole-generate gap is dominated by the one
Q16 cached-prefix step, where ByteV2 still uses the generic paged-decode path.

Artifacts:

- `profiles/byte_v2_speculative_steady_20260714/byte_q2_q8.jsonl`
- `profiles/byte_v2_speculative_steady_20260714/byte_q4.jsonl`
- `profiles/byte_v2_speculative_steady_20260714/raw_q2_q8.jsonl`
- `profiles/byte_v2_speculative_steady_20260714/raw_q4.jsonl`

## Fused Staging Release and Unsafe Flags (2026-07-14)

Multi-token ByteV2 cache update previously launched
`byte_v2_release_raw_staging` and then scanned unsafe metadata with
`byte_v2_update_cache_unsafe_flags`. The latter iterated over token slots, so
multiple Q tokens in the same physical page rescanned that page. The new
`byte_v2_release_raw_staging_and_update_flags` kernel iterates over unique
staging slots instead, scans each touched page once, writes the exact K/V
unsafe bits, and releases the allocator state in the same launch.

This is an exact post-commit scan. It does not infer flags from the current
update because commit metadata uses monotonic mask updates and may contain
older unsafe bits. The old release path remains available by setting
`BYTE_V2_FUSED_STAGING_RELEASE_FLAGS=0`; fused release defaults to on when
unsafe-page flags are present.

Same-binary Q4 A/B at context 4K:

| Metric | Old release + scan | Fused release | Change |
| --- | ---: | ---: | ---: |
| Release/flag sub-ops | 9.49 us/layer | 5.06 us/layer | -46.7% |
| Cache update | 66.99 us/layer | 60.04 us/layer | -10.4% |
| Verify attention op | 91.78 us/layer | 91.82 us/layer | unchanged |
| Steady throughput | 108.75 tok/s | 109.34 tok/s | +0.54% |
| Whole-generate throughput | 79.31 tok/s | 79.60 tok/s | within noise |

The cache-update target passes the 5% local retention threshold and saves
about 1.6-1.8 ms across the eight Q4 verify steps. Model compute and attention
still dominate each step, so the E2E gain is below 1% and is not statistically
separated from run-to-run noise. Q2 and Q8 whole-generate throughput changes
are also negligible (about +0.1% and +0.05%).

For Q4, raw FA2 remains at 111.49 steady tok/s. Fused ByteV2 is about 2% behind
in steady state, even though its verify attention is faster (91.82 versus
130.07 us/layer), because compressed cache update remains 60.04 versus 11.30
us/layer. Whole-generate ByteV2 remains about 19% slower due primarily to the
Q16 prefix step, not the Q4 verify mainloop.

Correctness coverage includes separate K-unsafe and V-unsafe pages, allocator
reset, partial-page staging, direct-cache equivalence, unsafe/outlier Q2/Q4/Q8
verify, and the complete ByteV2 test file: `146 passed, 1 skipped`.

Artifacts:

- `profiles/byte_v2_speculative_steady_20260714/byte_q4_release_old_ab.jsonl`
- `profiles/byte_v2_speculative_steady_20260714/byte_q4_release_fused_ab.jsonl`
- `profiles/byte_v2_speculative_steady_20260714/byte_q2_q8_release_fused.jsonl`

## Cached-Prefix Q16 Kernel (2026-07-14)

The one Q16 step in this benchmark is a prefix-cache scheduling artifact, not
a speculative draft choice. It previously fell through to the generic
cached-prefill paged-decode path. The new Q16 instance reuses the same
FA2-direct M16 body as the Q2/Q4/Q8 family and is independently gated by
`BYTE_V2_CACHED_PREFIX_Q16=1`, which defaults to off.

For GQA4, Q16 produces 64 virtual rows per KV head:

```text
virtual_rows_per_kv = 16 query tokens * 4 query heads = 64
cta_groups_per_kv = 64 / M16 = 4
```

Each CTA still has exactly the existing M16 shared-memory and register shape.
The new instance increases the number of independent groups; it does not build
an M64 CTA. The virtual-row mapping and reduce kernel map each row back to its
token/head output and use 16 distinct causal sequence lengths.

### Q16 Partition Sweep

Q16 CUDA-event time on A40, batch 1:

| Context | P64 | P128 | P256 | P512 | Selected |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1K | 153.5 us | 133.0 us | 116.9 us | 171.2 us | 256 |
| 2K | 235.2 us | 213.7 us | 193.5 us | 177.3 us | 512 |
| 4K | 407.4 us | 341.6 us | 334.0 us | 313.0 us | 512 |
| 8K | 784.2 us | 603.3 us | 555.6 us | 569.5 us | 512 |
| 16K | 1559.7 us | 1163.3 us | 1001.7 us | 968.4 us | 512 |

At 8K, repeated P256/P512 averages were 553.3/566.2 us. The 2.3% P256
advantage is below the 5% threshold, so the retained heuristic stays simple:
P256 at 1K, P512 at 2K and above, and P64 below 1K. Q2/Q4/Q8 keep their existing
partition rules. An explicit `BYTE_V2_DECODE_GQA_PACKED_PARTITION_SIZE` still
overrides both paths.

### Q16 and E2E Results

Context 4K, batch 1, 32 output tokens, eager mode, 100% acceptance:

| Metric | Old ByteV2 | Q16 ByteV2 | Raw FA2 |
| --- | ---: | ---: | ---: |
| Q16 attention | 2466.3 us/layer | 310.6 us/layer | 126.2 us/layer |
| Q16 scheduler step | 115.6 ms | 46.7 ms | 38.7 ms |
| Q4 steady throughput | 109.34 tok/s | 109.34 tok/s | 111.35 tok/s |

The Q16 kernel is 7.9x faster than the old ByteV2 Q16 path. It remains 2.46x
slower than raw FA2 attention, but the whole Q16 step is now within about 8 ms
of raw because model compute is shared by both backends.

Whole-generate results:

| Verify Q | ByteV2 before Q16 | ByteV2 with Q16 | Raw FA2 | ByteV2 gain | Gap to raw |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 47.73 tok/s | 53.20 tok/s | 52.72 tok/s | +11.5% | +0.9% |
| 4 | 79.60 tok/s | 95.79 tok/s | 97.12 tok/s | +20.3% | -1.4% |
| 8 | 116.71 tok/s | 155.76 tok/s | 172.97 tok/s | +33.5% | -9.9% |

ByteV2 and raw FA2 token IDs match exactly for Q2/Q4/Q8, and all runs retain
100% draft acceptance. The Q16 path passes the retention threshold for all
three complete workloads. Q2 and Q4 are now effectively at raw E2E parity on
this 4K batch-1 benchmark. Q8 remains about 10% behind because its steady
verify kernel uses two M16 groups and takes about 184.5 us/layer versus raw
FA2's approximately 130 us/layer.

The direct CUDA test now covers Q2/Q4/Q8/Q16 on an unsafe/outlier cache with
row-specific causal bounds. The full ByteV2 test file reports
`149 passed, 1 skipped`.

Artifacts:

- `profiles/byte_v2_speculative_q16_20260714/p64.jsonl`
- `profiles/byte_v2_speculative_q16_20260714/p128.jsonl`
- `profiles/byte_v2_speculative_q16_20260714/p256.jsonl`
- `profiles/byte_v2_speculative_q16_20260714/p512.jsonl`
- `profiles/byte_v2_speculative_q16_20260714/final_q2_q8_4k.jsonl`

## Q8 CTA Decode-Once M32 (2026-07-14)

The previous Q8 instance maps 32 virtual rows per KV head to two independent
M16 CTAs. Both CTAs read and decode the same compressed K/V partition. The new
instance changes only Q8 to one CTA with two independent M16 compute warps:

```text
stage:  4 warps cooperatively decode one K64x128 and V64x128 tile
MMA:    warp 0 computes virtual rows 0..15
        warp 1 computes virtual rows 16..31
state:  each compute warp retains only 16 softmax rows
```

This is an effective CTA M32 tile, but it deliberately keeps the existing M16
CuTe MMA and per-warp state shape. It therefore reuses decoded K/V without
doubling each thread's softmax arrays or accumulator fragments. Q2, Q4, and
Q16 continue to use their previous M16 instances.

The compiled A40 resource usage is:

| Q8 variant | Registers/thread | Shared/CTA | Stack/thread |
| --- | ---: | ---: | ---: |
| Two M16 CTAs | 254 | 36 KB | 320 B |
| One CTA, two M16 warps | 184 | 40 KB | 320 B |

The extra 4 KB is the second Q tile. K and V shared-memory tiles remain single
copies. The pre-existing 320-byte stack allocation is unchanged, so this
experiment does not solve the direct kernel's spill problem.

At context 4K, batch 1, 32 output tokens, the fresh old-path baseline and three
new-path repetitions are:

| Metric | Old M16x2 | New run 1 | New run 2 | New run 3 | New average |
| --- | ---: | ---: | ---: | ---: | ---: |
| Q8 native op | 184.78 us | 110.19 us | 110.27 us | 110.62 us | 110.36 us |
| Steady throughput | 200.67 tok/s | 213.76 tok/s | 213.53 tok/s | 213.18 tok/s | 213.49 tok/s |
| Whole generate | 153.03 tok/s | 162.84 tok/s | 162.59 tok/s | 160.24 tok/s | 161.89 tok/s |

The native Q8 op is 40.3% faster. Steady throughput improves by 6.4%, and
whole-generate throughput improves by 5.8%, so the change passes the 5%
retention threshold. All runs retain 100% acceptance and matching profile
tokens. The direct unsafe/outlier Q2/Q4/Q8/Q16 tests and the complete ByteV2
test file pass (`149 passed, 1 skipped`).

Fresh raw FA2 comparison shows that Q8 attention is no longer the long-context
bottleneck:

| Context | ByteV2 Q8 op | Raw FA2 Q8 op | ByteV2 whole generate | Raw whole generate |
| ---: | ---: | ---: | ---: | ---: |
| 1K | 61.82 us | 43.05 us | 153.94 tok/s | 189.26 tok/s |
| 4K | 110.36 us | 130.84 us | 161.89 tok/s | 175.64 tok/s |
| 8K | 208.72 us | 247.12 us | 144.94 tok/s | 156.39 tok/s |

At 4K and 8K, ByteV2 Q8 attention is about 16% faster than raw FA2, while E2E
remains slower. Profiling attributes the remaining gap to cache update and the
single cached-prefix Q16 step.

The context sweep also exposes a separate staging high-water-mark problem. A
1K prefill satisfies `max_tokens_per_update=1024`, so the reusable staging
manager allocates 1024 slots. Hydrate, commit, and release subsequently launch
over `staging_to_physical_block.size(0)`, even when a Q8 decode update uses only
a few active slots. This raises `cache_update.n8` from about 63 us to about
232 us. The same process then carries this allocation into the 8K case and
reports only 131.19 tok/s; a fresh 8K process reports 144.94 tok/s and restores
the 63 us update. Restricting these kernels to the active slot prefix, rather
than retained capacity, is the next optimization target. Further Q8 attention
changes are lower priority.

Artifacts:

- `profiles/byte_v2_speculative_q8_m32_20260714/baseline_m16x2.jsonl`
- `profiles/byte_v2_speculative_q8_m32_20260714/m32_run1.jsonl`
- `profiles/byte_v2_speculative_q8_m32_20260714/m32_run2.jsonl`
- `profiles/byte_v2_speculative_q8_m32_20260714/m32_run3.jsonl`
- `profiles/byte_v2_speculative_q8_m32_20260714/m32_context_sweep.jsonl`
- `profiles/byte_v2_speculative_q8_m32_20260714/m32_8k_fresh.jsonl`
- `profiles/byte_v2_speculative_q8_m32_20260714/raw_context_sweep.jsonl`

## Raw Staging Active-Slot Prefix (2026-07-14)

The high-water-mark issue above is fixed without changing the native op ABI.
For each update, the manager computes a conservative active-slot capacity:

```text
active_slot_capacity = min(num_kv_cache_blocks, num_update_tokens)
```

The number of unique physical pages touched by an update cannot exceed either
bound. Prepare, hydrate, append, commit, and release now receive prefix views of
the retained staging tensors with this capacity. Allocated buffers still keep
their high-water size for reuse, but native kernels no longer launch over that
retained capacity. This requires no readback of the GPU-side active counter and
introduces no synchronization or reallocation in the decode path.

The exact 1K then 8K mixed-context sweep was repeated before and after the
change:

| Context/metric | Retained-capacity scan | Active prefix | Improvement |
| --- | ---: | ---: | ---: |
| 1K `cache_update.n8` | 231.82 us | 63.15 us | 72.8% |
| 1K hydrate | 92.40 us | 6.30 us | 93.2% |
| 1K commit | 108.13 us | 23.26 us | 78.5% |
| 1K steady throughput | 194.06 tok/s | 222.54 tok/s | 14.7% |
| 1K whole generate | 153.94 tok/s | 176.16 tok/s | 14.4% |
| 8K `cache_update.n8` | 231.78 us | 62.65 us | 73.0% |
| 8K steady throughput | 174.30 tok/s | 197.08 tok/s | 13.1% |
| 8K whole generate | 131.19 tok/s | 147.73 tok/s | 12.6% |

A fresh 4K run remains at 163.13 tok/s, with Q8 attention at 110.17 us and
`cache_update.n8` at 63.51 us. This is consistent with the prior M32 results,
so the optimization removes retained-capacity work without regressing the
normal small-capacity path.

The same prompt list was also run with raw FA2:

| Context | ByteV2 | Raw FA2 | ByteV2 gap | ByteV2 steady gap |
| ---: | ---: | ---: | ---: | ---: |
| 1K | 176.16 tok/s | 187.20 tok/s | -5.9% | -7.2% |
| 8K | 147.73 tok/s | 156.80 tok/s | -5.8% | -1.8% |

ByteV2 and raw FA2 token IDs match exactly at both contexts, and acceptance is
100%. The complete ByteV2 test file passes (`149 passed, 1 skipped`). The E2E
gain exceeds the 5% retention threshold in both affected contexts, so the
active-prefix change is retained.

Artifacts:

- `profiles/byte_v2_speculative_q8_m32_20260714/active_prefix_context_sweep.jsonl`
- `profiles/byte_v2_speculative_q8_m32_20260714/active_prefix_4k.jsonl`
- `profiles/byte_v2_speculative_q8_m32_20260714/active_prefix_raw_context_sweep.jsonl`

## Q16 CTA Decode-Once M64 (2026-07-14)

Q16 produces 64 virtual rows per KV head. The previous path launched four M16
CTAs for every KV-head partition, so compressed K/V was read and decoded four
times. The new path launches one CTA and maps one M16 tile to each warp:

```text
stage:  all 4 warps cooperatively decode one K64x128 and V64x128 tile
MMA:    warp 0 computes virtual rows  0..15
        warp 1 computes virtual rows 16..31
        warp 2 computes virtual rows 32..47
        warp 3 computes virtual rows 48..63
state:  each warp retains only 16 softmax rows
```

The K/V shared-memory tiles remain single copies. Four Q tiles occupy 16 KB,
so total static shared memory grows to 48 KB. Per-warp state remains M16:

| Q16 variant | Registers/thread | Shared/CTA | Stack/thread |
| --- | ---: | ---: | ---: |
| Four M16 CTAs | 254 | 36 KB | 320 B |
| One M64 CTA | 179 | 48 KB | 320 B |

The larger CTA does not add spill and reduces register allocation. Because it
also reduces independent CTA count by four, the old Q16 partition heuristic is
not retained. The measured Q16 native-op sweep is:

| Context | P64 | P128 | P256 | P512 | P1024 | Selected |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1K | 90.02 us | 95.52 us | 123.84 us | - | - | 64 |
| 2K | 162.69 us | 114.98 us | 127.55 us | - | - | 128 |
| 4K | - | 209.76 us | 165.57 us | 193.79 us | 322.02 us | 256 |
| 8K | - | 389.89 us | 308.45 us | 269.41 us | 332.93 us | 512 |

The retained heuristic therefore doubles partition size at each 2x context
step from 1K through 8K. Q8 M32 independently switches from P256 to P512 at 8K;
Q2 and Q4 retain their previous rules.

With the default heuristic, Q16 changes as follows:

| Context | Four M16 CTAs | M64 CTA | Improvement | Raw FA2 |
| ---: | ---: | ---: | ---: | ---: |
| 1K | 116.96 us | 90.37 us | 22.7% | 39.55 us |
| 4K | 312.26 us | 166.82 us | 46.6% | 127.04 us |
| 8K | 569.89 us | 270.53 us | 52.5% | 243.36 us |

Final batch-1, 32-output-token E2E results using the same prompts are:

| Context | Verify Q | ByteV2 | Raw FA2 | ByteV2 gap |
| ---: | ---: | ---: | ---: | ---: |
| 4K | Q2 | 53.53 tok/s | 52.54 tok/s | +1.9% |
| 4K | Q4 | 97.27 tok/s | 96.98 tok/s | +0.3% |
| 4K | Q8 | 166.08 tok/s | 171.05 tok/s | -2.9% |
| 8K | Q2 | 50.52 tok/s | 47.59 tok/s | +6.2% |
| 8K | Q4 | 90.68 tok/s | 88.64 tok/s | +2.3% |
| 8K | Q8 | 157.38 tok/s | 156.12 tok/s | +0.8% |

All ByteV2/raw token IDs match exactly and acceptance is 100%. The direct
unsafe/outlier test covers Q2/Q4/Q8/Q16 with row-specific causal bounds, and
the complete ByteV2 test file passes (`150 passed, 1 skipped`). Q16 native time
improves by 23% to 53%, and the affected 8K Q8 E2E path improves by more than
5%, so M64 is retained.

Artifacts:

- `profiles/byte_v2_speculative_q16_m64_20260714/short_p64.jsonl`
- `profiles/byte_v2_speculative_q16_m64_20260714/short_p128.jsonl`
- `profiles/byte_v2_speculative_q16_m64_20260714/short_p256.jsonl`
- `profiles/byte_v2_speculative_q16_m64_20260714/p128.jsonl`
- `profiles/byte_v2_speculative_q16_m64_20260714/p256.jsonl`
- `profiles/byte_v2_speculative_q16_m64_20260714/p512.jsonl`
- `profiles/byte_v2_speculative_q16_m64_20260714/p1024.jsonl`
- `profiles/byte_v2_speculative_q16_m64_20260714/final_q8_context_sweep.jsonl`
- `profiles/byte_v2_speculative_q16_m64_20260714/final_raw_q8_context_sweep.jsonl`
- `profiles/byte_v2_speculative_q16_m64_20260714/final_byte_all_q.jsonl`
- `profiles/byte_v2_speculative_q16_m64_20260714/final_raw_all_q.jsonl`
- `profiles/byte_v2_speculative_q16_20260714/final_q4_4k.jsonl`
- `profiles/byte_v2_speculative_q16_20260714/final_raw_4k.jsonl`

## Compact Staging CTA Experiment (2026-07-14)

After active-prefix slicing, an N-token speculative update still launches
hydrate and commit grids using N staging slots even when the tokens touch only
one or two physical pages. A default-off experiment changed hydrate, metadata
clear, and commit to launch one page worth of CTAs and loop over active staging
slots inside each CTA. Prepare, append, and release were unchanged.

The candidate passed byte-exact cache comparisons for 3/16-token single-page
updates and an 8-token update crossing from row 12 into the next page. The
allocator routing and fused page-unsafe flag tests also passed. Stable A40
microbenchmark p50 results for the complete five-stage update were:

| Tokens | Start row | Active pages | Baseline | Compact | Change |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 0 | 1 | 67.58 us | 68.61 us | -1.5% |
| 2 | 12 | 1 | 68.61 us | 68.61 us | 0.0% |
| 4 | 0 | 1 | 67.58 us | 68.61 us | -1.5% |
| 4 | 12 | 1 | 68.61 us | 68.61 us | 0.0% |
| 8 | 0 | 1 | 67.62 us | 68.61 us | -1.5% |
| 8 | 12 | 2 | 74.75 us | 72.70 us | +2.7% |
| 16 | 0 | 1 | 68.61 us | 68.61 us | 0.0% |
| 16 | 12 | 2 | 74.75 us | 74.75 us | 0.0% |

The Q8 cross-page commit itself improved from 39.94 to 37.89 us, but the
complete update improved only 2.7%. Inactive CTAs return cheaply enough that
compacting their launch grid does not address the dominant cost. The remaining
cost is the five-op launch/dispatch chain plus per-tile exponent-window scan,
outlier construction, and payload re-encoding in commit.

Because no complete-update case reached the 5% retention threshold, the
compact kernels, op-schema changes, and environment switch were removed. E2E
testing was intentionally skipped after the microbenchmark gate failed. The
next cache-update experiment should reduce launch count or commit work, rather
than only compact inactive CTA slots.

Artifact:

- `profiles/byte_v2_compact_staging_update_20260714.jsonl`

## Single-Dispatch Raw-Staging Update (2026-07-14)

Nsight Systems showed that the speculative raw-staging update spent more time
between kernels than in its helper kernels. A retained native orchestration op,
`byte_v2_update_cache_raw_staging`, now enters C++ once and invokes the existing
prepare, hydrate, append, commit, and fused release functions in the same order.
It does not change any CUDA kernel or reduce the seven GPU operations.

On A40, synchronized complete-update latency changed as follows:

| Tokens | Start row | Old chain | Native entry | Latency reduction |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 0 | 55.49 us | 34.16 us | 38.4% |
| 4 | 0 | 55.86 us | 33.94 us | 39.2% |
| 8 | 0 | 55.50 us | 34.44 us | 37.9% |
| 8 | 12 | 56.48 us | 45.03 us | 20.3% |
| 16 | 0 | 55.60 us | 39.04 us | 29.8% |
| 16 | 12 | 55.90 us | 48.57 us | 13.1% |

All measured shapes exceed the 5% retention threshold. In the 4K Q8 engine
profile, Q8 cache update decreased from 62.38 to 42.81 us and cached-prefix Q16
decreased from 67.90 to 48.22 us. Steady speculative throughput increased by
1.8%; one-shot generation throughput varied by -1.3%, so no standalone E2E
claim is made. Token IDs match exactly and acceptance remains 100%.

The default speculative path uses the native entry when page-unsafe flags are
available. Set `BYTE_V2_NATIVE_RAW_STAGING_UPDATE=0` to restore the old chain.
The complete ByteV2 test file passes (`155 passed, 1 skipped`).

Artifact:

- `profile/byte-v2-native-cache-update-a40-20260714/REPORT.md`

## Q16 Row-Packed Split-K Reduction (2026-07-14)

The Q16 M64 main kernel still used the generic split-K reducer. That reducer
launches one CTA per virtual head and four output dimensions, producing 16,384
CTAs at Q16. Every output dimension independently recomputed the same
partition LSE reduction.

The retained Q16 reducer instead launches one CTA per virtual row (512 CTAs).
Warp 0 computes partition weights once into dynamic shared memory, then all
four warps reduce the 128 output dimensions in parallel. Inputs, main
attention, normalized partial-output format, and partition heuristic are
unchanged. Dispatch falls back to the old reducer if the weight workspace
would exceed 48 KiB.

On the real 4K/P256 Q16 capture:

| Metric | Original | Row-packed | Change |
| --- | ---: | ---: | ---: |
| Reduction kernel | 32.07 us | 8.26 us | -74.2% |
| Complete verify op | 174.08 us | 151.55 us | -12.9% |
| Active warps in reducer | 12.5% | 46.9% | +34.4 pp |

At E2E level, the median of three independent 4K/Q16 runs changes from
165.708 to 142.827 us for verify (-13.81%), from 268.006 to 273.168 tok/s for
steady throughput (+1.93%), and from 261.001 to 267.095 tok/s for one-shot
throughput (+2.33%). Token IDs and acceptance are unchanged.

Q8's four-row candidate improved the complete op by only 0.9%, so it was
removed. Q2/Q4/Q8 retain the old reducer. Q16 clears the 5% verify threshold
and is retained.

Artifact:

- `profile/byte-v2-spec-packed-reduce-a40-20260714/REPORT.md`

## FA2 Thread-Row Softmax State (2026-07-14)

Source-level NCU attribution showed that the M16 direct kernel gave every
thread dynamically indexed state for all 16 rows, although the tensor-core
fragment assigns only two logical row fragments to each thread. The old
layout spilled online-softmax and output state to local memory and performed
16 full-warp max/sum reductions per tile.

The retained rewrite now uses the same accumulator row/column transformation
as FA2:

```text
(MMA=4, MMA_M, MMA_N)
    -> thread rows (2, MMA_M) x thread columns (2, MMA_N)
```

Each thread keeps two compile-time-indexed `softmax_m/l`, `row_m/l`,
`new_m`, and `alpha` values. Each logical row is reduced by its four owning
lanes with two `shfl_xor` steps. PV rescaling, accumulation, normalization,
and stat writes use the same transformed layout. The cache format, fixed
decoder, MMA shapes, partition heuristic, and Q16 row reducer are unchanged.

Real 4K/P256 results on A40 are:

| Metric | Before | Thread-row | Change |
| --- | ---: | ---: | ---: |
| Q8 complete verify | 113.664 us | 86.016 us | -24.3% |
| Q16 complete verify | 152.576 us | 90.112 us | -40.9% |
| Q16 main kernel, NSYS | 139.24 us | 80.48 us | -42.2% |
| Q16 local load/store instructions | 207K / 255K | 0 / 0 | eliminated |
| Q16 steady E2E throughput | 273.168 tok/s | 287.054 tok/s | +5.08% |

The Q16 verify-op median in the three E2E engine processes falls from 142.827
to 81.291 us. Token IDs remain exact, acceptance is 100%, and mean acceptance
length remains 16. The candidate also measures 29.6% faster than the 128.000 us
raw FA2 Q16 event result for this workload, despite still executing about 2.2x
as many instructions.

This result supersedes the rejected scalar/switch and shared-output-state
experiments. The important change is fragment state ownership, not merely
moving the same dynamically indexed arrays to a different storage space.
Remaining source hotspots are fixed-payload exponent reconstruction, Q
staging/global dependency, and final partition-output division.

Artifact:

- `profile/byte-v2-q16-fa2-thread-row-softmax-a40-20260714/REPORT.md`

## Asynchronous Q Staging (2026-07-14)

After the thread-row softmax rewrite, source counters identified synchronous Q
staging as the largest long-scoreboard hotspot. The retained direct-path fast
path now emits aligned 16-byte SM80 `cp.async.cg` copies, commits them before
the tile loop, overlaps them with first-tile compressed K/V decode, and waits
immediately before the first compute barrier. Non-contiguous Q strides retain
the scalar implementation.

On the real 4K/P256 capture, Q16 complete verify falls from 90.112 to
69.632 us (-22.7%) and Q8 falls from 86.016 to 76.800 us (-10.7%). Q16 main
kernel time falls from 80.480 to 60.416 us. NCU true thread instructions fall
from 281.355M to 225.204M, long-scoreboard samples fall from 4453 to 2664,
and the old Q staging hotspot disappears. Register use remains 173 per thread.

Three Q16 E2E processes measure verify-op time at 70.529 us, steady throughput
at 289.861 tok/s, and one-shot throughput at 284.075 tok/s. Relative to the
thread-row baseline these are -13.2%, +0.98%, and +1.53%. Token IDs and
acceptance remain unchanged.

The candidate clears the 5% complete-kernel threshold and is retained. Fixed
4-bit exponent-code reconstruction is now the leading source hotspot.

Artifact:

- `profile/byte-v2-q-async-staging-a40-20260714/REPORT.md`

## Packed Exponent Reconstruction Experiment (2026-07-14)

The next experiment rewrote fixed 4-bit exponent-code reconstruction with
packed 32-bit nibble operations and `PRMT`, without changing payload loads,
cache format, outlier handling, or MMA. On Q16 this reduced true thread
instructions from 225.204M to 182.745M (-18.9%), but complete verify improved
only from 69.632 to 68.608 us (-1.47%). Q8 improved by 1.33%.

NCU explains the weak conversion from instruction reduction to latency:
global-load instructions were unchanged and the long-scoreboard ratio worsened
from 2.117 to 2.809. The rewrite shortened independent ALU work around the same
payload load-to-use chain instead of hiding or removing that dependency.

The candidate failed the 5% retention threshold and was removed. Future
fixed-decode work should change scheduling by preloading or pipelining payload
data while controlling register pressure. Rewriting the same dependency chain
with fewer scalar integer instructions is not a sufficient optimization target.

Artifact:

- `profile/byte-v2-packed-exponent-reconstruct-a40-20260714/REPORT.md`

## K/V Fixed-Word Preload Experiment (2026-07-15)

The fixed staging mapping assigns one physical-page row and one 16-dimension
hex to each of 128 threads. Because a page has exactly 16 rows, there is no
next-row inner-loop iteration to preload. The latest-kernel dependency trial
instead loaded the three K fixed words and three V fixed words before decoding
either side, extending the first load-to-use distance without changing format,
MMA, outlier handling, or CTA shape.

The candidate reduced the Q16 main kernel's long-scoreboard ratio from 2.141
to 1.726 (-19.4%) and increased eligible warps per cycle from 0.3164 to 0.3442.
Registers remained 173 per thread and local loads/stores remained zero. The
split helpers and longer live ranges added 0.6% dynamic instructions, however,
and Q16 complete verify improved only from 69.632 to 68.608 us (-1.47%). Q8
improved from 75.776 to 74.752 us (-1.35%).

The candidate failed the 5% threshold and was removed. This closes another
same-thread scheduling variant: the payload dependency is measurable, but a
single K/V fragment does not provide a large enough overlap window. A future
experiment should first test cooperative compact-word staging and separate
producer/consumer ownership on one codec tile while preserving the current
two-CTA-per-SM shared-memory limit.

Artifact:

- `profile/byte-v2-kv-word-preload-a40-20260715/REPORT.md`

## Single-Codec-Tile Producer/Consumer Pipeline (2026-07-15)

The cooperative-staging follow-up isolated one 384-byte K codec tile and used
a two-slot shared-memory ring. Warp 0 issued `cp.async.cg` payload copies and
warp 1 decoded shared payload into the existing staged-K layout. Named
ready/free barriers pipelined four physical pages in an N64 tile. A dedicated
K-only direct-load diagnostic provided an equal-work baseline; earlier K+V
versus K-only measurements were discarded.

The pipeline reduced explicit global-load instructions from 16.9K to 6.656K,
but it did not reduce DRAM bytes. Its K-only NCU kernel time increased from
17.34 to 20.67 us (+19.2%), instrumented total cycles increased 30.5%, and
stage service cycles increased 32.9%. Registers rose from 39 to 48, while the
shared-memory occupancy limit remained four CTAs per SM. Long-scoreboard and
barrier-stall ratios worsened by 42.2% and 46.9%, respectively.

The pipeline's consumer decode service measured 13,604 cycles versus 5,510
copy cycles. The single consumer therefore remained serial, while idle warps
converted its latency into final CTA barrier stalls. The candidate fails the
retention threshold and was removed. Fine-grained one-tile warp specialization
is not useful with the current format and effective M; subsequent work should
reduce consumer decode service through format/kernel co-design or increase the
useful compute window before revisiting staging.

Artifact:

- `profile/byte-v2-codec-tile-pipeline-a40-20260715/REPORT.md`

## Speculative Row-Fast Staging (2026-07-15)

After rejecting producer/consumer staging, the next experiment changed only
the speculative direct kernel's thread ownership for fixed-payload decode and
shared staging. The previous dim-fast map assigned neighboring threads to
eight dimension hexes of the same row. The retained row-fast map assigns one
half warp to all 16 rows of a single dimension hex. Cache format, decode
arithmetic, unsafe/outlier handling, shared layout, and QK/PV are unchanged.

Real 4K capture medians improve from 59.392 to 55.296 us at Q2 (-6.90%),
67.584 to 63.488 us at Q4 (-6.06%), 76.800 to 72.704 us at Q8 (-5.33%), and
69.632 to 65.536 us at Q16 (-5.88%). Q16 NCU main-kernel time falls 4.92%,
long-scoreboard ratio falls 18.6%, and eligible warps per cycle rises 6.88%.
Registers increase from 173 to 174 without changing the two-CTA-per-SM limit.

The key mechanism is shared-memory layout alignment. Shared-store bank
conflicts fall from 135,692 to 2,329, shared-store wavefronts fall 49.2%, and
all 131,584 excess shared wavefronts are eliminated. L1 global-load sectors
fall only 0.33%, so the retained gain should be attributed primarily to shared
store conflict removal rather than HBM traffic reduction.

An E2E baseline/candidate pair preserves exact token IDs, 100% acceptance, and
mean acceptance length 16. Overall throughput is unchanged because the kernel
saving is small relative to a complete model step. The candidate clears the
complete-operation threshold and is retained for speculative kernels only.

Artifact:

- `profile/byte-v2-spec-row-fast-staging-a40-20260715/REPORT.md`

## Generic Row-Fast Staging Follow-Up (2026-07-15)

The speculative ownership map was next tested as an isolated generic
FA2-direct diagnostic across MHA and GQA2/4/8/16. Full row-fast ownership
improved all-safe kernels by 2.33-4.88%, below the 5% threshold. On fully
unsafe/outlier pages it improved every shape by 5.48-6.17%.

Forced GQA4 NCU confirms the same shared-layout mechanism as speculative
verify: shared-store bank conflicts fall from 132,250 to 754 and all 131,072
excess shared wavefronts are removed. Global-load requests also fall 7.0% and
long-scoreboard samples fall 12.9%, although barrier samples rise 27.4%.

A deployable follow-up kept dim-fast ownership for safe pages and enabled
row-fast only in the guarded generic unsafe branch. This combined current
kernel improved forced MHA by 4.23%, GQA4 by 4.17%, and GQA16 by 4.94%. It did
not preserve the specialized diagnostic's full gain and failed the retention
threshold. The generic change and temporary mode were removed; speculative
row-fast remains active and still measures 65.536 us at Q16.

Artifact:

- `profile/byte-v2-generic-row-fast-staging-a40-20260715/REPORT.md`

## Packed-Code Word Preload Follow-Up (2026-07-15)

The row-fast kernel next preloaded only the K and V 64-bit packed-code words,
leaving the low-byte loads in the original decode helper. This avoided the six
live 64-bit words of the earlier full K/V preload while targeting the remaining
packed-code load-to-use hotspot.

The candidate regressed Q8 by 5.71% and Q16 by 4.69%. Q16 NCU duration rose
5.49%, and long-scoreboard samples rose from 2117 to 2561 even though registers,
global-load instructions, and L1 sectors were unchanged. Pulling both code words
forward lengthened one thread's combined K/V dependency window without adding
independent work. The candidate was removed, closing same-thread code-word
rescheduling.

Artifact:

- `profile/byte-v2-code-word-preload-a40-20260715/REPORT.md`

## Row-Fast Split-Low-Plane Format Probe (2026-07-15)

A same-size 12-bit format probe rearranged the 256-byte low plane from
`row[16]` into two 128-byte `row[8]` planes. Under row-fast ownership this made
all three per-row 64-bit loads contiguous across a half warp without changing
the 384-byte tile size or code representation.

NCU confirmed that total L1 load sectors fell 39.0%, but L1 misses and DRAM
bytes were unchanged. The default layout's extra sectors were cache hits caused
by the second low-word load reusing the first load's sectors. Both full-output
and shared-stage event medians were unchanged, while the shared-stage NCU
kernel was 0.2% slower and long-scoreboard exposure increased. The format was
rejected before production integration.

Artifact:

- `profile/byte-v2-row-fast-split-low-planes-a40-20260715/REPORT.md`

## Real ShareGPT Acceptance Follow-Up (2026-07-17)

The earlier periodic-token prompt is an optimistic 100%-acceptance workload.
A follow-up benchmark now reads first-turn user prompts from 1,000 rows of
`ShareGPT_Vicuna_unfiltered`, applies the Llama-3.1 chat template, disables
prefix caching, and uses deterministic prompt sampling. The model, GPU, BF16
dtype, CPU n-gram drafter, and greedy sampling are unchanged. Q2/Q4/Q8 mean
one, three, and seven draft tokens respectively.

For seed 20260717, the four actual prompt lengths were 81, 60, 87, and 384
tokens. Each request generated 64 tokens. This workload completed on both
backends, but rejection made the scheduled batches ragged rather than a
uniform fixed-Q verification batch.

| Q | Byte acceptance | Raw acceptance | Byte tok/s | Raw tok/s | Byte vs raw |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 57.14% | 61.54% | 87.640 | 95.861 | -8.58% |
| 4 | 47.62% | 41.67% | 90.687 | 98.661 | -8.08% |
| 8 | 28.57% | 21.43% | 87.058 | 98.094 | -11.25% |

The 64-token trajectories diverged between backends in this first sample.
The earliest divergence was token 6-9 for the longest request, while some
requests remained exact. This is a valid end-to-end serving observation, but
the differing acceptance counts make it unsuitable as a strictly paired
kernel comparison.

A second Q4 sample, seed 20260718, used prompt lengths 40, 41, 288, and 465.
Both backends produced exactly the same 64 output tokens for every request and
both measured 41.11% acceptance. ByteV2 reached 116.424 tok/s versus 129.163
tok/s for raw FA2, a 9.86% deficit. This paired result confirms that the first
sample's 8-11% gap is not explained only by token divergence.

### Dispatch Attribution

No `byte_v2_speculative_verify_gqa` event appears in either real trace. Partial
rejection produces mixed Q1/Q2/Q4 work, which fails the specialized kernel's
uniform-complete-batch dispatch guards and falls back to generic cached
prefill plus split-K paged decode.

For seed 20260717 Q4, per-layer CUDA-event averages were:

| Operation | ByteV2 | Raw FA2 | Ratio |
| --- | ---: | ---: | ---: |
| Cached Q4 attention wrapper | 196.6 us | 38.1 us | 5.16x |
| Main cache-update op | 37.5 us | 5.7 us | 6.58x |

The ByteV2 trace also contains substantial Q1 and Q2 fallback work between
verification steps. Therefore optimizing only the uniform fixed-Q kernel
cannot preserve the ideal benchmark gain under real rejection. Production
dispatch needs a ragged or masked verification kernel that retains shared KV
decode for the still-active requests, plus a cheaper multi-token cache update.

### Q1 Cache-Update Failure

A stratified four-request test selected ShareGPT prompts nearest 256, 512,
1024, and 2048 tokens; actual lengths were 257, 512, 1024, and 2072. Raw FA2
completed two-token generation. ByteV2 completed prefill and the first output
token, then failed on the next cache commit.

A benchmark-only pre-commit diagnostic reproduced the kernel's exact high7
window and power-of-two outlier allocation. The failing physical page had nine
valid rows and required 2,048 pool entries, exceeding the V5 page capacity of
1,024 and triggering the allocator's device `__trap()`. All 128 K/V codec
tiles had exactly 16 outliers. Row attribution was:

```text
row_outliers = [0, 0, 0, 0, 0, 0, 0, 0, 2048]
```

The new decode row was entirely outside the exponent windows selected by the
eight hydrated rows. None of its values were zero. This regular pattern points
to an incompatibility between hydrated compressed rows and the appended raw
row, rather than a natural small-capacity overflow. The exact hydrate/append
root cause still needs to be fixed before ByteV2 can be considered robust on
real continuous decode.

Artifacts:

- `profile/byte-v2-real-acceptance-a40-20260717/sharegpt_q2_q4_q8.jsonl`
- `profile/byte-v2-real-acceptance-a40-20260717/sharegpt_seed20260718_q4.jsonl`

## Direct-Prefill Page Boundary Fix (2026-07-17)

The Q1 failure was traced to the direct prefill cache writer rather than the
hydrate/append path. The kernel grouped every 16 contiguous source tokens to
select one exponent base. Batched vLLM input concatenates requests without
padding each request to 16 tokens, so one source group can cross physical KV
pages. Payload threads wrote each physical page, but the group wrote its base
metadata only to the first page. A later cache update decoded the remaining
pages with an invalid zero base and classified the appended row as 2,048
outliers.

The direct writer now identifies every physical-page run inside each 16-token
source group and independently writes that page's base, outlier metadata, and
payload. The launch geometry and compressed format are unchanged. A regression
test uses unaligned request lengths `(1, 15, 8, 9)` and verifies exact BF16 K/V
round trips for four separate physical pages.

The stratified Q1 workload with prompt lengths 257, 512, 1024, and 2072 then
completed 32 tokens per request without a device trap. Across 2,048 active page
observations, no page overflowed and the largest outlier-pool allocation was
14 entries instead of 2,048. In eager paired timing, ByteV2 reached 72.676
tok/s versus raw FA2 at 83.994 tok/s, a 13.47% deficit. This is a correctness
fix, not a performance optimization.

Artifact:

- `profile/byte-v2-real-acceptance-a40-20260717/sharegpt_long_q1_page_group_fix_eager.jsonl`

## Per-Request Ragged Dispatch Probe (2026-07-17)

A default-off Python dispatch probe sent each cached-prefill request with
actual Q2/Q4/Q8 work to the existing fixed-Q GQA verification kernel. Q1 and
unsupported requests retained the generic split-K path. This tested whether
reusing the existing kernel was sufficient before implementing a new compacted
or masked ragged CUDA kernel.

On the paired ShareGPT seed 20260718 Q4 workload, output token IDs and the
41.11% draft acceptance rate were unchanged. Dedicated GQA operator calls rose
from 64 to 960, while generic split-K calls fell from 4,512 to 3,616. Generic
split-K CUDA time fell from 205.51 ms to 137.57 ms, but dedicated GQA operator
time rose from 2.37 ms to 32.75 ms because work was split into many single-
request launches.

| Mode | Throughput | Change |
| --- | ---: | ---: |
| Per-request dispatch off | 116.021 tok/s | baseline |
| Per-request dispatch on, run 1 | 118.268 tok/s | +1.94% |
| Per-request dispatch on, paired repeat | 117.975 tok/s | +1.68% |

The experiment was removed because its repeatable E2E gain was below the 5%
retention threshold. The useful follow-up is a single batched masked/ragged
kernel with compact request descriptors and one workspace, not Python-level
per-request launches.

Artifacts:

- `profile/byte-v2-real-acceptance-a40-20260717/sharegpt_seed20260718_q4_ragged_ab_off.jsonl`
- `profile/byte-v2-real-acceptance-a40-20260717/sharegpt_seed20260718_q4_ragged_per_request.jsonl`
- `profile/byte-v2-real-acceptance-a40-20260717/sharegpt_seed20260718_q4_ragged_ab_on_repeat.jsonl`

## Batched Mixed-Q Ragged Probe (2026-07-17)

The profiler was extended to record non-empty per-request query lengths from
`query_start_loc_cpu`. Dividing the per-layer counts by 32 layers gives the
following 32 cached-prefill steps for the paired seed 20260718 workload:

| Active request Q pattern | Steps |
| --- | ---: |
| `1,1,1` | 9 |
| `1,1,1,4` | 5 |
| `1,1,4` | 1 |
| `1,1,4,1` | 6 |
| `1,4,1,1` | 1 |
| `1,4,1,4` | 2 |
| `4,1,1` | 4 |
| `4,1,1,1` | 1 |
| `4,1,1,4` | 3 |

This confirmed that the useful CUDA interface must process mixed Q1/Q4 in one
launch. A temporary implementation instantiated the fixed Q4 M16 kernel with
dynamic valid rows, used GPU `query_start_loc` to map packed Q/O, skipped empty
requests, and launched one main kernel plus one reduction for the whole batch.
A `[Q1,Q4,Q0,Q1]` CUDA test matched the causal raw-BF16 reference, and the real
workload preserved token IDs and the 41.11% acceptance rate.

The mixed kernel reduced generic split-K calls from 4,512 to 1,728 and reduced
their profile CUDA time from 205.51 ms to 80.13 ms. The new mixed operator cost
32.82 ms. End-to-end results were:

| Pair | Disabled | Enabled | Gain |
| --- | ---: | ---: | ---: |
| Run 1 | 116.220 tok/s | 121.945 tok/s | +4.93% |
| Reverse-order repeat | 116.218 tok/s | 121.642 tok/s | +4.67% |

Partition sizes 128 and 256 reached only 121.360 and 120.918 tok/s,
respectively, versus 121.642-121.945 tok/s for partition 64. The implementation
was removed because neither paired run exceeded the 5% retention threshold.
The shape attribution remains in the benchmark. A future attempt needs to
remove additional cache-update or reduction cost rather than only consolidate
attention launches.

Artifacts:

- `profile/byte-v2-real-acceptance-a40-20260717/sharegpt_seed20260718_q4_pattern_attribution.jsonl`
- `profile/byte-v2-real-acceptance-a40-20260717/sharegpt_seed20260718_q4_batched_ragged_off.jsonl`
- `profile/byte-v2-real-acceptance-a40-20260717/sharegpt_seed20260718_q4_batched_ragged_on.jsonl`
- `profile/byte-v2-real-acceptance-a40-20260717/sharegpt_seed20260718_q4_batched_ragged_on_repeat.jsonl`
- `profile/byte-v2-real-acceptance-a40-20260717/sharegpt_seed20260718_q4_batched_ragged_off_repeat.jsonl`
- `profile/byte-v2-real-acceptance-a40-20260717/sharegpt_seed20260718_q4_batched_ragged_p128.jsonl`
- `profile/byte-v2-real-acceptance-a40-20260717/sharegpt_seed20260718_q4_batched_ragged_p256.jsonl`

## CUDA Graph Workspace Fix (2026-07-17)

Compiled Q1 initially failed with an illegal memory access on the first FULL
CUDA Graph replay. Capture itself completed for batch sizes 4, 2, and 1. The
split-K workspace allocator used exact tensor shapes, however, so each smaller
capture replaced the workspace referenced by the previously captured graph.
The first batch-4 replay consequently dereferenced freed device storage.

The workspace now uses stable one-dimensional backing tensors sized on the
first descending capture for all decode partition heuristic buckets. Each
launch takes only the required contiguous prefix and reshapes it to the exact
`[tokens, heads, partitions]` or `[tokens, heads, partitions, head_dim]`
shape. This preserves the captured base address without launching or reducing
unused partitions. Empty and populated max-logit workspaces also use separate
storage so changing output mode cannot release a captured pointer.

Both a 2-token smoke test and a 16-token repeated-replay test completed on the
A40 with `CUDA_LAUNCH_BLOCKING=1`. The full ByteV2 attention test file passed
with 172 tests and one skip.

Artifacts:

- `profile/byte-v2-real-acceptance-a40-20260717/sharegpt_q1_compiled_flat_workspace.jsonl`
- `profile/byte-v2-real-acceptance-a40-20260717/sharegpt_q1_compiled_flat_workspace_long.jsonl`

## Real Compiled Q1/Q4 Comparison (2026-07-17)

The corrected compiled path was measured with four distinct ShareGPT prompts
of 257, 512, 1,024, and 2,072 tokens, 64 generated tokens per request, and
prefix caching disabled. Prompt-lookup speculative decoding used three draft
tokens, so the verification query had a maximum length of four. ByteV2 and raw
FA2 had nearly identical draft acceptance, approximately 42%.

| Backend | Q1 | Q4 speculative | Q4 versus Q1 |
| --- | ---: | ---: | ---: |
| ByteV2 | 87.65 tok/s | 83.34 tok/s | -4.91% |
| Raw FA2 | 103.09 tok/s | 105.59 tok/s | +2.42% |

ByteV2 was 14.98% behind raw FA2 for Q1 and 21.07% behind for realistic Q4.
This workload therefore does not yet preserve raw throughput despite saving KV
capacity. The result also explains why the earlier 116 tok/s number was not
comparable: that run selected prompts of only 40, 41, 288, and 465 tokens.

The dominant recorded ByteV2 operator was the generic split-K decode path:
3,872 layer calls consumed 526.49 ms. Raw FA2's Q1 and Q4 varlen attention calls
consumed 123.99 ms in total. The dedicated ByteV2 Q4 operator consumed only
2.36 ms across 64 calls because most real steps contained mixed Q1/Q4 requests
and fell back to generic split-K. Raw-staging cache updates added 61.34 ms,
versus 17.31 ms for raw FA2's measured incremental cache updates.

The next optimization target is therefore a batched mixed-Q kernel or an
equivalent FA2-style varlen dispatch that removes generic split-K work. Further
tuning of the already-small uniform-Q4 kernel cannot materially close the E2E
gap.

Artifacts:

- `profile/byte-v2-real-acceptance-a40-20260717/sharegpt_seed20260718_q1_compiled_flat_workspace_ab.jsonl`
- `profile/byte-v2-real-acceptance-a40-20260717/sharegpt_seed20260718_q4_compiled_workspace_fix_ab.jsonl`
- `profile/byte-v2-real-acceptance-a40-20260717/sharegpt_seed20260718_q4_compiled_flat_workspace.jsonl`

## Batched Mixed-Q Row-Packed Reduction (2026-07-18)

The rejected batched ragged probe used the fixed-Q reduction geometry, which
launched many CTAs for each output dimension. The retained implementation uses
the Q16 row-packed reducer: one reduction CTA owns one virtual
`(query token, query head)` row and reduces all partitions and 128 output
dimensions together. The main kernel still shares each KV decode across the
four GQA heads and accepts per-request Q0 through Q4 from GPU
`query_start_loc` metadata.

The production compiled path exposed an additional dispatch requirement:
vLLM pads the packed query and output tensor to a compiled size bucket. For
example, a mixed `[1,4,1,1]` step has seven actual tokens but query/output
shape eight; other steps in this workload also map actual sizes 5/6/7 to 8
and 10/13 to 16. The native and Python guards now accept an actual-token
prefix, clamp every request segment to that prefix, and never read or write
the padded tail. Direct CUDA tests cover `[1,4,0,1]` and `[0,1,2,3,4]`, compare
against a causal raw-BF16 reference, and verify that a padded output sentinel
is unchanged.

The ragged path uses an independent geometrically growing workspace, so a
larger four-row-per-request allocation cannot replace backing storage captured
by Q1 decode graphs. It also fails closed when an exact CPU sequence-length
mirror is unavailable; async speculative decoding therefore keeps the generic
path instead of introducing a GPU-to-CPU synchronization in every layer.

On the same four-prompt compiled ShareGPT workload used by the previous real
comparison, paired order `off -> on -> on -> off` produced the following clean
timings. The measured generation ran before any profiler monkeypatch was
installed; operator events were collected only during a separate replay.

| Pair | Disabled | Enabled | Gain |
| --- | ---: | ---: | ---: |
| Run 1 | 84.707 tok/s | 91.357 tok/s | +7.851% |
| Reverse-order repeat | 84.558 tok/s | 91.285 tok/s | +7.955% |

All four ByteV2 runs had identical token IDs, 41.935% acceptance, and mean
accepted length 2.258. Both profile replays also recorded identical mixed-Q
patterns and compiled bucket shapes between arms. The generic split-K op fell
from 3,872 calls / 526.124 ms to 1,504 calls / 251.760 ms in the first pair.
The new ragged op handled 640 calls in 51.026 ms. The reverse pair measured
525.961 ms, 251.668 ms, and 51.010 ms, respectively. The clean off/on means
were 84.632 and 91.321 tok/s, a 7.903% gain. This passes the two-run 5%
retention threshold, so the implementation is kept behind the default-off
`BYTE_V2_SPECULATIVE_VERIFY_RAGGED_Q4` experiment flag.

Raw FA2 reached 105.667 tok/s on the same-input clean reference run, leaving
the optimized ByteV2 mean of 91.321 tok/s 13.576% behind in E2E serving throughput.
Raw FA2 followed a different token trajectory and had 42.157% acceptance, so
this is not a token-for-token paired comparison. The dominant remaining
operator is Q1/generic split-K decode (251.760 ms), followed by incremental
raw-staging cache updates (60.026 ms). Those paths have a larger remaining
end-to-end ceiling than further tuning the 51 ms mixed-Q op.

Artifacts:

- `profile/byte-v2-ragged-row-reduce-a40-20260718/REPORT.md`
- `profile/byte-v2-ragged-row-reduce-a40-20260718/raw/off_clean_pair1.jsonl`
- `profile/byte-v2-ragged-row-reduce-a40-20260718/raw/on_clean_pair1.jsonl`
- `profile/byte-v2-ragged-row-reduce-a40-20260718/raw/on_clean_pair2.jsonl`
- `profile/byte-v2-ragged-row-reduce-a40-20260718/raw/off_clean_pair2.jsonl`
- `profile/byte-v2-ragged-row-reduce-a40-20260718/raw/raw_fa2_clean_reference.jsonl`

## FA2 Template ByteV2 Loader (2026-07-18)

The Q1 decode path now has an experimental implementation based directly on
the pinned FA2 split-KV template. It replaces only the global-to-shared K/V
load policy with V5 ByteV2 decode. The FA2 QK MMA, online softmax, PV MMA,
split-selection heuristic, output layout, and split-K combine kernel are left
unchanged. The implementation is currently restricted to causal BF16 decode
with local `Hq=32`, `Hkv=8`, `D=128`, the default V5 page layout, and no ALiBi,
local window, or positive softcap. Set `BYTE_V2_DECODE_KERNEL=fa2` to select it
and fail closed if the request is incompatible; `auto` uses it when compatible
and otherwise keeps the legacy ByteV2 path. The default remains `legacy` while
overflow/fallback pre-routing and broader model coverage are still experimental.

The first scalar-copy prototype preserved the FA2 math but decoded all 128
elements independently. It generated 255 registers plus 344 bytes of stack
and took 143.968 us at sequence length 64. The retained loader instead maps
each FA2 8-element vector to one ByteV2 page row/dimension tile, hoists the
page and sideband metadata, loads packed low bytes and nibbles, reconstructs
eight BF16 values with packed integer operations, and writes shared memory
with `STS.128`. Its nonsplit Q1 kernel uses 248 registers and no stack or local
memory.

| A40 kernel-only case | ByteV2 FA2 | Raw FA2 | Byte/raw |
| --- | ---: | ---: | ---: |
| Q1, context 64, main | 12.704 us | 7.232 us | 1.757x |
| Q1, context 4,099, split main | 68.482 us | 38.081 us | 1.798x |
| Q1, context 4,099, original combine | 8.032 us | 8.256 us | 0.973x |

The vectorized loader is 11.33x faster than the scalar prototype at context
64. The long-context trace launches the original
`flash_fwd_splitkv_combine_kernel`; its timing is statistically equivalent to
raw FA2, confirming that the combine path was not reimplemented. The remaining
approximately 1.8x main-kernel gap is in synchronous compressed-data loads,
integer reconstruction, sideband handling, and shared stores, while raw FA2
can use its native asynchronous copy pipeline.

Correctness checks compared the new kernel with a raw-BF16 FA2 oracle after
decoding the same V5 cache. Output and softmax LSE were bitwise identical for
Q1/Q2/Q4 probes, a 67-token partial page, forced outliers, a three-request
batch, and split-K contexts including 4,099 tokens. The packed eight-lane BF16
reconstruction was also exhaustively checked for every single-lane encoded
value plus 1,006,400 randomized full words. CUDA memcheck reported zero errors
for the partial-page forced-outlier case.

Two separate-engine end-to-end comparisons also produced identical token IDs:

| E2E case | ByteV2 FA2 | Raw FA2 | Gap | Token match |
| --- | ---: | ---: | ---: | --- |
| Eager, context 128, 8 output tokens | 31.965 tok/s | 34.226 tok/s | -6.61% | 8/8 exact |
| Compiled/CUDA graph, context 4,096, 32 output tokens | 21.390 tok/s | 22.753 tok/s | -5.99% | 32/32 exact |

This establishes a bitwise-aligned decode path without duplicating FA2's
attention or combine math. The next useful optimization target is the ByteV2
load policy itself: reduce reconstruction instructions and overlap its global
loads with decode/shared-memory production without changing FA2's downstream
math order. Production promotion also requires a safe pre-route for V5 pages
marked overflow/fallback; the experimental loader currently traps on those
pages because V5 contains no raw payload from which to recover them.

## FA2 In-Place Async Staging (2026-07-19)

The retained FA2 loader now restores the original mainloop's global-load
overlap without allocating a second K/V shared-memory buffer. Each FA2 thread
already owns sixteen independent 16-byte destination vectors. For every
vector, the loader uses `cp.async.ca` to stage the ByteV2 eight-byte low plane
at bytes 0--7 and the four-byte exponent-code plane at bytes 8--11. Bytes
12--15 hold packed base/outlier metadata. After the existing FA2
`wait_group 0`, the same thread expands the staged 12 bytes in place to eight
BF16 values and the existing CTA barrier publishes them to the MMA consumers.

The resulting schedule is:

1. stage K and commit it with the unchanged Q prologue;
2. wait, decode K in place, and enter the existing QK phase;
3. stage V before QK so its global copies overlap QK and masking;
4. wait and decode V before the existing PV barrier;
5. stage next-K before softmax/PV so its global copies overlap that work.

QK MMA, masking, online softmax, PV MMA, split selection, output layout, and
the original FA2 combine launch are otherwise unchanged. The raw FA2 branches
inside the template are also unchanged.

A metadata leader/broadcast probe was rejected before this change. Replacing
same-address metadata loads with half-warp shuffles changed sequence-64 time
from 12.640 to 12.768 us and sequence-4,099 time from 68.288 to 70.081 us.
Those metadata loads are already coalesced/broadcast by the memory system, so
the added shuffle and control instructions were pure overhead.

Fresh Nsight Systems traces on the A40 show:

| Q1 kernel-only case | Sync ByteV2 | Sync-run raw | Staged ByteV2 | Staged-run raw | Byte improvement | Staged byte/raw |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Context 64, main | 12.640 us | 7.296 us | 11.649 us | 7.200 us | 7.84% | 1.618x |
| Context 4,099, split main | 68.288 us | 34.880 us | 56.513 us | 35.104 us | 17.24% | 1.610x |

The raw control shifts by only -1.32% at context 64 and +0.64% at context
4,099 across the two processes. Normalizing each ByteV2 result by its paired
raw control changes the relative improvement to 6.61% and 17.77%
respectively.

Each 4,099-token oracle process also contains two launches of the same
`flash_fwd_splitkv_combine_kernel` specialization, one after ByteV2 and one
after raw FA2. Their aggregate average changes from 8.752 to 8.528 us, which is
noise-level and confirms that the combine path did not change.

A second trace with 20 interleaved ByteV2/raw calls removes most one-launch
noise. At context 64, the staged ByteV2 median is 10.945 us versus 6.720 us for
raw FA2. At context 4,099, the medians are 51.520 and 35.009 us respectively,
reducing the warm repeated main-kernel gap to 1.472x. All 20 comparisons remain
bitwise exact. This repeated trace contains only the staged implementation and
raw control; the 17.24% synchronous-to-staged result above remains a
single-launch comparison, supported independently by the NCU replay trend.

Nsight Compute replay also moves in the expected direction. Its absolute
duration changes from 80.10 to 69.95 us, active IPC from 0.42 to 0.60, eligible
warps per scheduler from 0.11 to 0.14, and the dominant long-scoreboard stall
from about 4.95 to 2.6 cycles per issued instruction. Executed instructions
increase from 2.72 million to 3.46 million because each compressed vector uses
separate 8-byte and 4-byte async copies, but hiding their latency more than
offsets the extra issue work.

The four generated ByteV2 kernels use 250--255 registers per thread. The
noncausal nonsplit specialization reaches the architectural 255-register
limit, but all four retain zero stack and zero local memory. Dynamic shared
memory remains the original FA2 80.00 KiB (81.92 kB).

Bitwise comparison against raw paged FA2 remains exact for Q1/Q2/Q4, partial
pages with forced outliers, a three-request batch, and a split-K 4,099-token
case: both output and LSE report zero mismatches. Compute Sanitizer memcheck
and synccheck report zero errors for both the partial-page forced-outlier case
and the 4,099-token split case. The upper-half partial-page boundary is also
covered explicitly: sequence lengths 73 and 79 with forced outliers remain
bitwise exact for three runs each, and sequence 73 reports zero memcheck and
synccheck errors. A fresh eager E2E run at context 128 produces
the same eight token IDs, with 32.040 tok/s for ByteV2 and 34.238 tok/s for raw
FA2. The approximately 0.23% gain over the prior 31.965 tok/s ByteV2 run is
within E2E noise; the measurable benefit is concentrated in long-context Q1
attention. A fresh compiled/CUDA-graph run at context 4,096 also preserves all
32 token IDs and measures 21.563 tok/s for ByteV2 versus 22.718 tok/s for raw
FA2. That is 0.81% above the prior 21.390 tok/s ByteV2 result, still too small
for a standalone E2E retention claim but consistent with the kernel-level
improvement.

## Cooperative 12-Copy FA2 Staging and Fatal V5 Pages (2026-07-19)

The retained in-place loader now uses `VLLM_BYTE_V2_FA2_STAGE_MODE=2`.
Mode 0 preserves the per-vector 32-copy baseline and mode 1 preserves the
first cooperative 16-copy experiment for controlled A/B testing. For mode 2,
thread `t = 16 * page + 8 * row_half + column` owns one 8-row by
16-dimension ByteV2 quadrant. It issues eight 16-byte low-plane copies and
four 16-byte paired-row code-plane copies, or 12 `cp.async` operations per
thread. Across 128 threads this transfers exactly the 24 KiB compressed
payload of an FA2 N128 K/V tile with 16-byte operations, so payload staging
has reached its byte-count lower bound. Sideband metadata is separate from
that bound.

The code bytes temporarily occupy the adjacent `+/-16` byte partner vector
in FA2's existing `Swizzle<3,3,3>` shared-memory layout. No second shared
buffer or additional CTA barrier is introduced. Each row pair shares a
16-byte code slot, so the decoder expands the odd row first and the even row
second; the even write overwrites the temporary slot only after both rows
have consumed it. The same worker reconstructs both 8-dimension halves and
writes the final FA2 vectors. QK, mask, online softmax, PV, split handling,
output layout, and the original FA2 combine kernel remain unchanged. Outlier
entries are scanned in reverse order so duplicate entries retain the scalar
decoder's first-entry-wins behavior.

Twenty interleaved ByteV2/raw Nsight Systems calls show the following warm
medians:

| Context | 32-copy | 16-copy | 12-copy | Current raw | 12-copy/raw | 12-copy versus 32-copy |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 10.9445 us | 9.984 us | 9.792 us | 6.816 us | 1.437x | -10.53% |
| 4,099 | 51.520 us | 49.376 us | 47.3285 us | 35.200 us | 1.345x | -8.13% |

All 20 output/LSE comparisons are bitwise exact. A same-profiler NCU A/B at
context 4,099 explains the retained gain:

| Metric | 32-copy staged | 12-copy staged | Change |
| --- | ---: | ---: | ---: |
| Replay duration | 69.952 us | 63.424 us | -9.33% |
| Executed instructions | 3.455008 M | 2.761608 M | -20.07% |
| `LDGSTS` instructions | 70,400 | 29,120 | -58.64% |
| Registers per thread | 251 | 246 | -5 registers |
| Dynamic shared memory | 81.920 kB | 81.920 kB | unchanged |
| DRAM throughput | 36.903% | 36.214% | -0.689 pp |

Active IPC decreases from 0.5972 to 0.5422, eligible warps per scheduler from
0.1445 to 0.1352, and the long-scoreboard ratio increases from 2.6019 to
3.0559. The improvement therefore comes from removing copy/issue work, not
from higher occupancy or more latency hiding. The causal nonsplit/split and
noncausal nonsplit/split specializations use 247/255/246/246 registers. All
four retain zero stack and zero local memory, and the original FA2 80 KiB
dynamic shared-memory allocation is unchanged.

A fresh NCU 2025.4.1 full/source run on an otherwise idle A40 measures the
current context-4,099 split main at 60.800 us. The 136-CTA grid has 1.619 waves
over 84 SMs. The 80 KiB shared-memory footprint limits occupancy to one CTA
per SM and 8.33% theoretical occupancy. DRAM read throughput reaches only
30.22% of sustained peak, while 964 of 2,528 source samples (38.1%) are
long-scoreboard stalls. Of those, 586 map to the authoritative page-overflow
load in `byte_v2_fa2_loader.cuh`; another 92 map to the fallback-mask check.
The next loader-local experiment is therefore to prefetch/issue these
sideband reads before the 12 payload copies and consume them afterward,
preserving the same fail-closed checks while testing whether the dependency
latency can overlap. The larger structural ceiling is reducing the FA2 shared
footprint enough for two CTAs per SM, but that would violate the current
constraint of changing only the raw-copy policy.

Correctness coverage now includes Q1 lengths 64, 73, 79, and 4,099;
forced-outlier Q2/Q4; batch 3; and ragged `[73, 128, 4105]` with non-monotonic
physical pages, four shared-prefix blocks, forced outliers, and explicit
split 4. Output and LSE are bitwise exact in every case. Memcheck and
synccheck report zero errors for both the 73-token forced-outlier case and the
4,099-token split case. The complete attention test file reports 194 passed
and one skipped.

Overflow/fallback handling is deliberately fail-closed:

- an injected page-overflow marker traps in the FA2 reader;
- an injected tile fallback bit traps in the FA2 reader;
- a real writer stress case that needs more than the 1,024-entry page outlier
  pool sets the overflow marker and traps during allocation;
- ordinary outlier pages are not fatal and remain bitwise exact.

V5 sets `IncludeRawPayloadValue=false`. An overflow/fallback page therefore
contains no authoritative BF16 payload from which either the FA2 reader or
the legacy kernel could recover. Dispatching such a page to legacy would only
reinterpret the same incomplete compressed data and could silently corrupt
the result. A recoverable design requires a persistent raw sidecar/shadow or
a new page-local raw payload plus an authoritative page mode. The existing
`page_unsafe_flags` is not that mode: it mixes ordinary outliers with fallback,
may be stale, and does not include the page overflow marker. Also, a valid V5
writer cannot reach the current single-tile fallback threshold because a tile
has at most 256 elements while fallback requires more than 256 outliers; the
real reachable resource failure is exhaustion of the page-wide 1,024-entry
pool.

Finally, three independent E2E repetitions compare the current mode-2 loader
with raw FA2. Every pair produces identical tokens, and the token sequence is
also stable across repetitions:

| E2E case | Median ByteV2 | Median raw | Median paired ByteV2/raw | Gap | Token match |
| --- | ---: | ---: | ---: | ---: | --- |
| Eager, context 128, 8 tokens | 36.791 tok/s | 39.800 tok/s | 0.9310 | -6.90% | 8/8 exact in all runs |
| Compiled/CUDA graph, context 4,096, 32 tokens | 22.879 tok/s | 24.034 tok/s | 0.9504 | -4.96% | 32/32 exact in all runs |

These paired ratios are more stable than absolute throughput across the two
idle A40s. They include cache update, prefill, independent-engine, and graph
overheads and are therefore not isolated decode-kernel slowdown ratios. The
default production dispatch remains `legacy`; mode-2 FA2 selection still
requires `BYTE_V2_DECODE_KERNEL=fa2` (or the compatible experimental `auto`
path).

Artifacts:

- `profile/bytev2-fa2-coop12-a40-20260719/REPORT.md`
- `profile/bytev2-fa2-coop12-a40-20260719/reports/full_seq4099_q1.ncu-rep`
- `profile/bytev2-fa2-coop12-a40-20260719/reports/source_seq4099_q1.ncu-rep`
- `profile/bytev2-fa2-coop12-a40-20260719/analysis/stall_hotspots_seq4099_q1.txt`
- `profile/bytev2-fa2-coop12-a40-20260719/analysis/e2e_three_run_summary.txt`

## FA2 Sideband Demand-Load Overlap (2026-07-19)

The retained loader now also uses
`VLLM_BYTE_V2_FA2_SIDEBAND_PREFETCH_MODE=2`. After validating the active
thread's physical page, it issues ordinary `ld.global.u32` demand loads for
the page-global overflow marker and the K/V-specific fallback mask. It then
issues the existing twelve payload `cp.async` operations before consuming the
sideband values. The loads are explicit inline PTX with a compiler memory
clobber, and final SASS is an acceptance condition: both `LDG.E` instructions
must precede the payload `LDGSTS`, while the trap branches remain afterward.
This is a verified scheduling property of the current toolchain, not a new
concurrent-writer synchronization primitive.

Twenty interleaved ByteV2/raw calls on an idle A40 give the following final
warm medians:

| Q1 main kernel | Post-copy sideband | Prefetch mode 2 | Change | Paired raw |
| --- | ---: | ---: | ---: | ---: |
| Context 64 | 9.824 us | 9.264 us | -5.70% | 6.848 us |
| Context 4,099 | 47.7125 us | 44.704 us | -6.31% | 35.184 us |

The long-context paired-raw-normalized improvement is 6.18%. NCU 2025.4.1
independently moves from 60.800 to 57.952 us (-4.68%). Global-load instruction
count remains 11,952 and DRAM reads remain 12.77 MB, while long-scoreboard
samples fall from 1,009 to 804 (-20.3%), eligible warps per scheduler rise
5.3%, and active IPC rises 4.7%. This establishes latency overlap rather than
reduced memory traffic as the cause.

The controlled modes were:

- mode 0: all sideband loads after payload copies;
- mode 1: early overflow marker only;
- mode 2: early overflow plus fallback, retained;
- mode 3: also early outlier mask, rejected because its paired ByteV2/raw
  ratio regressed 0.33% at context 64 and 0.31% at context 4,099 versus mode 2.

The four final specializations use 247/255/245/244 registers for causal
nonsplit/split and noncausal nonsplit/split. All retain zero stack and zero
local memory, the shared footprint remains 80 KiB, and QK/softmax/PV/split and
the original combine kernel are unchanged.

The final correctness matrix remains bitwise exact for Q1 lengths 64, 73, 79,
and 4,099; forced-outlier Q2/Q4; batch 3; and ragged
`[73, 128, 4105]` with permuted pages, four shared-prefix blocks, forced
outliers, and split 4. Overflow, fallback, and real pool-exhaustion probes all
fail closed. Memcheck, synccheck, and racecheck report zero issues for the
73-token forced-outlier and 4,099-token split cases. The complete attention
test file reports 194 passed and one skipped, and an eager independent-engine
smoke keeps all eight generated token IDs exact.

Source sampling shows the old overflow/fallback hotspots are no longer
dominant. The remaining largest loader-local dependency is outlier-mask
consumption, but mode 3 proves that another scalar live-range extension does
not pay for itself. The next large ceiling is therefore structural: the
original FA2 80 KiB shared allocation still limits the A40 to one CTA per SM.
Changing that would exceed the current loader-only constraint. A separate
format-level alternative is an authoritative, race-safe committed-page mode
that lets common pages bypass fatal checks.

Artifacts:

- `profile/bytev2-fa2-sideband-prefetch-a40-20260719/REPORT.md`
- `profile/bytev2-fa2-sideband-prefetch-a40-20260719/reports/full-prefetch2-seq4099-q1.ncu-rep`
- `profile/bytev2-fa2-sideband-prefetch-a40-20260719/reports/source-prefetch2-seq4099-q1.ncu-rep`
- `profile/bytev2-fa2-sideband-prefetch-a40-20260719/analysis/nsys_ab_summary.txt`
- `profile/bytev2-fa2-sideband-prefetch-a40-20260719/analysis/stall_hotspots_prefetch2.txt`

## Split-Only FA2 K/V Shared-Memory Alias (2026-07-19)

The structural follow-up reuses one FA2 32 KiB KV shared tile for both K and
V, but only when `Split=true`. The original D128/M64/N128/4-warp layout uses
16 KiB for Q, 32 KiB for K, and 32 KiB for V, or 80 KiB dynamic shared memory.
Together with NCU's 1 KiB driver allocation, that limited the A40 to one CTA
per SM. The alias uses 48 KiB dynamic and 50,176 B total per CTA; two CTAs use
100,352 B and fit under the A40's 102,400 B per-SM limit. Nonsplit retains the
original 80 KiB pipeline because its small grid cannot benefit from a second
resident CTA.

The split mainloop now owns the shared tile in the order K decode, QK, V
stage/decode, softmax/PV, then next K. CTA-uniform barriers prevent V from
overwriting K before QK and prevent the next K from overwriting V before PV.
V staging overlaps the existing mask/softcap register work. N128, QK/PV MMA,
masking, online-softmax arithmetic, split boundaries, output layout, and the
original FA2 combine kernel are unchanged. The generic/raw loader advertises
no alias capability and compiles through the original branch.

Twenty interleaved ByteV2/raw calls on the final full rebuild produce:

| Q1 main kernel | Sideband mode-2 baseline | K/V alias | Change |
| --- | ---: | ---: | ---: |
| Context 4,099 mean | 44.711 us | 36.163 us | -19.12% |
| Context 4,099 median | 44.704 us | 36.017 us | -19.43% |
| Mean paired ByteV2/raw ratio | 1.270x | 1.015x | -20.10% |
| Context 64 mean | 9.267 us | 9.283 us | +0.17% |
| Context 64 median | 9.216 us | 9.216 us | unchanged |

At context 4,099, ByteV2 main plus the unchanged combine averages 44.432 us;
raw main plus combine averages 44.709 us in the same candidate process. All
20 output and LSE comparisons are bitwise exact.

NCU 2025.4.1 independently reduces replay duration from 57.952 to 44.832 us
(-22.64%). The shared-memory occupancy limit rises from one to two CTAs,
theoretical occupancy from 8.33% to 16.67%, achieved active warps from 8.53%
to 13.73%, and eligible warps per scheduler from 0.1461 to 0.2112. Registers
fall from 244 to 240 for the active noncausal split specialization. DRAM bytes
and executed work are essentially unchanged, while read bandwidth rises from
220.4 to 285.0 GB/s; the improvement is therefore latency hiding from restored
concurrency. The extra alias barriers increase barrier and MIO stall pressure,
but not enough to offset the occupancy gain.

Final causal nonsplit/split specializations use 247/244 registers and
noncausal nonsplit/split use 245/240. All four have zero stack/local memory;
both split SASS images contain no `LDL` or `STL`. SASS also confirms that the
temporary paired-row code/lows values are loaded before their shared slots are
overwritten. This ordering remains an acceptance check for future toolchain
upgrades.

The complete bitwise matrix passes Q1 lengths 64, 73, 79, and 4,099;
forced-outlier Q2/Q4; batch 3; and ragged `[73, 128, 4105]` with permuted
pages, four shared-prefix blocks, forced outliers, and explicit split 4.
Overflow, fallback, and page-pool-exhaustion probes fail closed. Memcheck,
synccheck, and racecheck report zero issues for both the 73-token
forced-outlier and 4,099-token split cases. The full attention test reports
194 passed and one skipped.

Serving-path validation explicitly sets `BYTE_V2_DECODE_KERNEL=fa2` and
`BYTE_V2_DECODE_RAW_FALLBACK=0`. An Nsys engine trace records 64 split alias
launches at 49,152 B dynamic shared and no legacy decode kernel. Three fresh
compiled/CUDA-graph context-4,096 pairs, ordered Byte/raw, raw/Byte, and
Byte/raw, preserve all 32 token IDs. Their paired throughput ratios are
0.953047, 0.955579, and 0.957142; median ByteV2/raw throughput is
21.7809/22.7921 tok/s, a 4.442% gap. The previous sideband mode-2 median ratio
was 0.950436, so the isolated 19% kernel gain closes about 0.514 percentage
point of E2E gap; other model compute, cache update, and scheduling work
dominates wall time. An eager context-128 control also preserves all eight
tokens and measures 32.151 versus 34.233 tok/s on the unchanged nonsplit path.

The next evidence-backed experiment is to move only the unchanged online
softmax call before the aliased V wait/decode, overlapping its register work
with the V asynchronous trip. It preserves floating-point order but may
extend register live ranges, so any spill or less than 5% isolated gain should
reject it. N64 remains excluded because it would change softmax/split
reduction order and lose the current raw-FA2 bitwise oracle.

Artifacts:

- `profile/bytev2-fa2-kv-alias-a40-20260719/REPORT.md`
- `profile/bytev2-fa2-kv-alias-a40-20260719/reports/final-seq4099-q1.nsys-rep`
- `profile/bytev2-fa2-kv-alias-a40-20260719/reports/full-kv-alias-seq4099-q1.ncu-rep`
- `profile/bytev2-fa2-kv-alias-a40-20260719/reports/source-kv-alias-seq4099-q1.ncu-rep`
- `profile/bytev2-fa2-kv-alias-a40-20260719/analysis/nsys_ab_summary.txt`
- `profile/bytev2-fa2-kv-alias-a40-20260719/analysis/validation_summary.txt`
- `profile/bytev2-fa2-kv-alias-a40-20260719/analysis/e2e_fa2_summary.txt`
