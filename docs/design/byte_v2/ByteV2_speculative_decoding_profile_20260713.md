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

## Safe Fused Single-Token Staging Update (2026-07-19)

The old in-place n=1 updater is now disabled by default. It freezes each
tile's base from row 0 and allocates replacement pooled-outlier segments
without reclaiming the old segment during growth. In the full-FA2 long-output
workload this eventually exceeds the 1,024-entry page pool and traps after
roughly 75 generated tokens.

The retained replacement keeps the safe full-page rebase/compaction semantics
but reduces its launch chain. One CUDA kernel reads the device-side slot,
initializes staging slot zero, decodes the existing page prefix, and copies the
new BF16 row. It then reuses the existing metadata clear, warp-histogram
commit, and release-plus-flags kernels. QK, softmax, PV, split selection, and
the original FA2 combine remain unchanged.

The complete n=1 update changes from seven traced GPU operations to four.
Across target rows 0, 1, 4, 8, 12, and 15, a same-process alternating A/B
reduces CUDA-event medians from 34.99--35.82 us to 25.32--25.83 us, or
27.0%--28.3%. Nsight Systems row 8 reduces projected traced time from 59.21 to
40.02 us/update (-32.42%). The unchanged commit remains the largest active
kernel at about 8.38 us; the new combined hydrate/append stage takes about
2.15 us.

NCU confirms that the commit is a tiny-grid latency kernel rather than a
bandwidth or spill bottleneck: 128 CTAs, 0.127 waves/SM, 12.15% achieved
occupancy, 2.14% SM throughput, 0.754% peak DRAM-read throughput, 40
registers/thread, and no local spill. This supports removing orchestration
rather than another histogram rewrite.

Generic Q1 decode does not otherwise consume page-unsafe flags, so an initial
E2E attempt did not hit the native operation: an engine trace showed 192
prepare/hydrate/append launches and zero fused launches. When the new safe
mode is enabled, the cache-update caller now provides the existing internal
flags buffer for n=1. A second trace records the fused stage,
warp-histogram commit, and release-with-flags path; padded compile-warmup
shapes continue to use the general implementation.

Compiled/CUDA-graph E2E at context 4,096, batch 1, and no speculation gives:

| Outputs | Mode | ByteV2 | Raw FA2 | Paired throughput gap | Token result |
| ---: | --- | ---: | ---: | ---: | --- |
| 128 | Previous native-safe | 3.9955 s | 3.9032 s | -2.31% | exact |
| 128 | Default safe fused staging | 3.9721 s | 3.9029 s | -1.74% | exact |
| 256 | Fresh generic-safe median | 34.0737 tok/s | 35.1655 tok/s | -3.10% | exact |
| 256 | Safe fused staging median | 34.6984 tok/s | 35.1675 tok/s | -1.33% | exact |

The three 256-output candidate walls are 7.3779, 7.3684, and 7.4006 seconds.
Compared with fresh generic-safe runs, median E2E throughput improves about
1.83%. The previously recorded native-safe control is stricter and faster
than those fresh generic-safe runs; against it, the candidate still reduces
wall time by 0.68% and closes about 0.73 percentage point of the raw gap.

Correctness covers random, low-outlier, overlay, and non-contiguous K/V
inputs; rows 0/1/8/15; invalid negative and positive slots; old Torch schema
arity; preflight fail-before-mutation; and 32 CUDA Graph replays spanning two
physical pages. Memcheck, synccheck, and racecheck report zero errors or
hazards. The complete attention test reports 217 passed and one skipped. All
saved 128/256 ByteV2 and raw FA2 output tokens match exactly.

The safe fused staging mode is therefore enabled by default through
`BYTE_V2_FUSED_SINGLE_TOKEN_STAGING`, with `0` retaining the general path for
controlled comparison. `BYTE_V2_NATIVE_SINGLE_TOKEN_UPDATE` remains
default-off.

Artifacts:

- `profile/byte-v2-safe-n1-update-baseline-a40-20260719/REPORT.md`
- `profile/byte-v2-safe-n1-update-baseline-a40-20260719/reports/full_commit_row8.ncu-rep`
- `profile/byte-v2-safe-n1-update-baseline-a40-20260719/reports/source_commit_row8.ncu-rep`
- `profile/byte-v2-safe-n1-update-baseline-a40-20260719/reports/nsys_fused_row8.nsys-rep`
- `profile/byte-v2-safe-n1-update-baseline-a40-20260719/reports/nsys_engine_fused_hit_out2.nsys-rep`

## Safe Single-Token Two-Kernel Update Experiment (2026-07-20)

The next launch-reduction experiment folds the four-kernel safe n=1 update
into two kernels while preserving the full-page V5 rebase/compaction
semantics. It does not restore the rejected in-place updater.

The first fusion moves page-unsafe-flag aggregation and allocator release into
the existing warp-histogram commit. The 128 tile CTAs use the existing
`overflow[0]` word as a completion counter, atomically OR exact K/V unsafe
bits, and let the last CTA release staging state. Invalid padding and
out-of-range slots participate in the completion protocol but preserve valid
page flags. This changes four launches to three.

The second fusion moves metadata clear into the hydrate/append stage. Its 128
CTAs temporarily use `next_staging_slot[0]` as a completion counter. Once all
old-page reads finish, the last CTA clears the destination metadata and
publishes the staging slot. Commit then runs unchanged on the same stream and
also performs the fused release. This changes three launches to two.

At target row 8, same-process alternating tests use 21 trials of 500 updates:

| Complete n=1 update | Baseline event | Candidate event | Change | Baseline wall | Candidate wall | Change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Commit/release fusion, 4 to 3 | 27.1176 us | 21.9566 us | -19.03% | 27.1500 us | 21.9866 us | -19.02% |
| Stage/clear fusion, 3 to 2 | 22.3949 us | 19.4744 us | -13.04% | 22.4259 us | 19.5052 us | -13.02% |

The final isolated Nsys trace contains only the fused stage/clear and
commit/release kernels, averaging 2.650 and 8.846 us over 300 updates. No
standalone clear or release remains.

An output-8 engine trace initially appears to contain both fused and generic
chains. Timestamp and CUDA Graph attribution resolves the ambiguity: the 96
generic instances are three 32-layer passes during compilation, per-layer
graph construction and validation. Their final launch precedes real prefill
by 1.618 seconds. Real 4,096-token prefill uses the direct writer, while
measured and profiler decode each execute seven replays of full-model Graph
199 captured immediately after a fused n=1 pass. The serving path therefore
does hit the candidate; generic kernels are not residual steady-decode work.

The decisive same-native-binary E2E comparison uses context 4,096, batch 1,
256 output tokens, no speculation and compiled CUDA Graph execution:

| Mode | Median ByteV2 | Median paired gap to raw | Token result |
| --- | ---: | ---: | --- |
| Existing four-kernel safe update | 34.6127 tok/s | -1.4619% | 256 exact in all three runs |
| Two-kernel candidate | 34.7125 tok/s | -1.2012% | 256 exact in all three runs |

The candidate reduces median ByteV2 wall time by 21.262 ms, improves
throughput by 0.2883%, and closes 0.2606 percentage point of the paired raw
gap. That is below the declared 0.5% E2E retention gate. The two new controls
therefore default to off; `BYTE_V2_FUSED_SINGLE_TOKEN_STAGING` remains on, so
the production default is the already-validated four-kernel path. The tested
three- and two-kernel implementations remain available for controlled
experiments through:

```text
BYTE_V2_FUSED_SINGLE_TOKEN_COMMIT_RELEASE=1
BYTE_V2_FUSED_SINGLE_TOKEN_STAGE_METADATA_CLEAR=1
```

Correctness covers both three- and two-kernel combinations; rows 0/1/8/15;
random, low-outlier and strided inputs; exact flags 0/3/5/7 against an
independent scan; invalid slots; preflight failure; and 32 cross-page CUDA
Graph replays. The focused final selection reports 33 passed. Two-kernel
memcheck, synccheck, racecheck and graph memcheck all report zero issues. All
six saved E2E token sequences are mutually exact. The complete attention test
reports 236 passed and one skipped.

The result places a data-backed ceiling on further local update-tail work:
removing half its graph nodes moves complete E2E by less than 0.3%. Attention
main plus the original combine is already approximately tied with raw (44.432
versus 44.709 us). The next optimization should first attribute the remaining
step-level raw/ByteV2 residual outside attention and then target the dominant
graph, cache-update or scheduler interval rather than another small update
kernel rewrite.

Artifacts:

- `profile/byte-v2-safe-n1-commit-release-a40-20260720/REPORT.md`
- `profile/byte-v2-safe-n1-stage-clear-a40-20260720/REPORT.md`
- `profile/byte-v2-safe-n1-stage-clear-a40-20260720/analysis/e2e_summary.json`
- `profile/byte-v2-safe-n1-stage-clear-a40-20260720/reports/nsys_engine_out8.nsys-rep`
- `profile/byte-v2-safe-n1-stage-clear-a40-20260720/reports/nsys_fused_stage_clear_row8.nsys-rep`

## Short-Context Two-Kernel E2E Retention (2026-07-20)

The existing two-kernel n=1 update candidate was retested at contexts 128,
512, and 1,024 using a short-capacity compiled engine, batch 1, no
speculation, and 256 generated tokens. Four-kernel, two-kernel, and raw FA2
engines were ordered as a three-round cyclic Latin square. Both ByteV2 modes
used the same native binary and FA2-template decode; only the two optional
single-token fusion controls differed.

The paired two-kernel/four-kernel throughput changes are:

| Context | Round 1 | Round 2 | Round 3 | Median | +0.5% gate |
| ---: | ---: | ---: | ---: | ---: | :---: |
| 128 | -1.0256% | +0.3699% | -0.6062% | -0.6062% | Fail |
| 512 | -0.9391% | +0.2957% | -0.3174% | -0.3174% | Fail |
| 1,024 | -0.2695% | +0.1969% | +0.0762% | +0.0762% | Fail |

None of the nine individual pairs reaches +0.5%; the largest is +0.3699%.
The 128/512 signs change across rounds, so the negative medians are not treated
as precise regressions, but the absence of a retainable improvement is clear.
The four-kernel median paired gaps to raw are -1.0778%, -1.3578%, and -2.0363%
at 128/512/1,024; the two-kernel gaps are -1.5841%, -1.6708%, and -2.1322%.

All 27 measured requests produce 256 tokens. At each context, all four-kernel,
two-kernel, and raw token lists are elementwise identical across every round;
all instrumented replays also report exact agreement with their measured
request. The candidate remains available for diagnostics but defaults stay
off, leaving the safe four-kernel path in production.

This short-context result strengthens the ceiling on update-tail launch work:
removing two Graph nodes does not recover the E2E gap. The next trace should
re-attribute the short-shape 1K Graph residual. If cache update is still the
largest component, the next structural design should reduce complete active-
page hydrate/re-encode work, not remove another clear/release launch. A
persistent authoritative raw tail page with compression on page closure is a
candidate, but it requires a real BF16 sidecar and explicit page mode; V5's
current overflow/unsafe markers are not recoverable storage.

Artifacts:

- `profile/byte-v2-short-context-two-kernel-e2e-a40-20260720/REPORT.md`
- `profile/byte-v2-short-context-two-kernel-e2e-a40-20260720/analysis/e2e_ab_summary.json`
- `profile/byte-v2-short-context-two-kernel-e2e-a40-20260720/analysis/e2e_ab_summary.csv`
- `profile/byte-v2-short-context-two-kernel-e2e-a40-20260720/reports/run{1,2,3}_{four_kernel,two_kernel,raw}.jsonl`

## Exact Hybrid Raw-Fallback Checkpoint (2026-07-20)

Checkpoint `7489f5f17` (`Add exact ByteV2 FA2 raw fallback`) is the current
reproducible baseline for the compact/raw hybrid design. The feature is
experimental and defaults to off. It is enabled with:

```text
BYTE_V2_FA2_HYBRID_RAW_FALLBACK=1
```

The checkpoint preserves the existing FA2 template and changes only the KV
load/store boundary:

- `page_to_raw_slot[page] == -1` selects the 52,096-byte compact V5 page;
  a non-negative entry selects an authoritative 65,536-byte raw BF16 sidecar
  page.
- The FA2 loader decodes compact pages or copies raw BF16 pages into the
  original shared-memory tile. QK, online softmax, PV, split selection, output
  layout, and the original FA2 combine kernel remain unchanged.
- The writer keeps ordinary representable outliers in V5. It allocates a raw
  sidecar only for an already-raw page, a true compact-pool overflow, or an
  explicit fallback result. Raw data is copied and fenced before the page map
  is published.
- Raw-slot allocation is race-safe and fail-closed. Invalid maps, missing
  extension operations, and raw-pool exhaustion do not silently read or write
  a compact page.
- Q1 decode and generic causal cached-context Q greater than one both use the
  hybrid FA2 reader. Prefix-cache block death/reuse and global prefix reset
  reclaim raw slots. One fixed-address staging workspace is shared across all
  ByteV2 layers and is allocated before CUDA Graph capture.

Direct CUDA coverage at this checkpoint includes compact-only, raw-only,
mixed, ragged, permuted, shared-prefix, split, real compact overflow, update
of an existing raw page, reset/reuse, and allocator exhaustion. Compact and
raw hybrid reads preserve the original FA2 output/LSE bit pattern in the
covered cases. The serving sweep below adds full-model token-exact evidence
for the common path, but it did not naturally allocate a raw page.

### Current Support Boundary

The hybrid mode deliberately rejects configurations whose sidecar lifetime or
FA2 arithmetic has not been made exact:

- BF16 causal decoder self-attention, compact V5 pages, 16 tokens per page,
  local `(Hq, Hkv, D) = (32, 8, 128)`, and the default V5 tile policy are
  required.
- ALiBi, sliding-window/local attention, positive logit softcap, non-decoder
  attention, and cross-layer KV-cache sharing are rejected.
- KV connectors, KV-cache offload, sleep mode, DCP/PCP, ubatching, and DBO are
  rejected because they do not yet transfer, restore, or isolate the raw
  sidecar and its single-lane staging workspace.
- The tested serving scope is one A40, batch 1, one model process, compiled
  CUDA Graph execution, and no speculation. Prefix caching and CUDA Graphs
  have lifecycle integration, but distributed and offloaded serving are not
  implied by this checkpoint.
- Large writes are divided into page-aware waves bounded by the shared staging
  capacity. The default is 128 staging pages. This bounds memory and keeps
  captured addresses stable, at the cost of additional writer launches for
  long prompts.

These checks are fail-closed at configuration or dispatch time. They should
not be relaxed until the corresponding sidecar ownership and bitwise tests
exist.

### Complete Memory Budget

Let:

- \(L\) be the number of ByteV2 layers;
- \(P\) be the number of physical cache blocks;
- \(R\) be persistent raw fallback slots per layer, defaulting to
  \(\max(1, \lfloor P / 256 \rfloor)\);
- \(S\) be runner-owned shared staging slots, defaulting to 128.

For the tested 32/8/128 layout, one compact page is 52,096 bytes and one raw
BF16 page is 65,536 bytes. Persistent state per ByteV2 layer is:

```text
page map                 = 4 * P
raw pages + free stack   = (65,536 + 4) * R
free/fatal counters      = 8
```

The single cross-layer staging workspace is:

```text
raw staging pages        = 65,536 * S
block-to-slot map        = 4 * P
slot maps/valid rows     = 8 * S
allocator counters       = 8
```

The full planned hybrid allocation is therefore:

```text
M_hybrid(P) =
    52,096 * L * P
  + L * (4 * P + 65,540 * R + 8)
  + (65,536 * S + 4 * P + 8 * S + 8)

M_raw(P) = 65,536 * L * P
```

This accounting includes the persistent raw sidecars and the transient
fixed-address workspace; neither is hidden outside the KV-cache budget. The
planner uses a monotonic search for the largest \(P\) that fits the profiled
budget.

For round 1, \(L=32\), \(S=128\), and the paired approximately 20.11 GB
planner budget produced:

| Plan | Blocks | Compact tensor | Sidecar | Workspace | Total planned |
| --- | ---: | ---: | ---: | ---: | ---: |
| Compact V5 | 12,065 | 20,113,223,680 B | 0 | 0 | 20,113,223,680 B |
| Hybrid V5/raw | 12,001 | 20,006,531,072 B | 98,011,264 B | 8,437,644 B | 20,112,979,980 B |
| Raw BF16 | 9,591 | 20,113,784,832 B | 0 | 0 | 20,113,784,832 B |

The hybrid plan has \(R=46\) slots per layer. At the same 12,001-block
capacity, raw BF16 would require 25,167,921,152 bytes, so the complete hybrid
allocation saves 20.08486%, after sidecars and workspace. Under the fixed
planner budget, hybrid provides 12,001 versus 9,591 raw blocks, a 25.12772%
capacity increase. The sidecar/workspace cost reduces capacity by only 64
blocks, or about 0.53%, relative to compact-only V5 in this configuration.

### Round-1 Compiled E2E Baseline

The first sweep leg uses batch 1, no speculation, 256 output tokens, compiled
CUDA Graph execution, and contexts 64 through 16K. The order is compact,
hybrid, then raw. All three variants use independent engine processes. The
hybrid run collects its allocator state after timed generation and profiler
replay.

This is one round, not a statistically complete comparison:

| Context | Compact tok/s | Hybrid tok/s | Raw tok/s | Hybrid/compact | Hybrid/raw |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 37.5357 | 35.3362 | 38.3386 | -5.8596% | -7.8312% |
| 128 | 37.4488 | 36.1792 | 38.2672 | -3.3902% | -5.4563% |
| 512 | 36.9426 | 35.9428 | 37.8799 | -2.7063% | -5.1138% |
| 1,024 | 36.3695 | 35.3709 | 37.2673 | -2.7456% | -5.0886% |
| 2,048 | 35.2776 | 34.1900 | 36.1002 | -3.0829% | -5.2916% |
| 4,096 | 33.3708 | 32.2708 | 33.9679 | -3.2963% | -4.9962% |
| 8,192 | 29.7428 | 28.6129 | 30.2574 | -3.7990% | -5.4354% |
| 16,384 | 23.7262 | 22.7219 | 24.0206 | -4.2327% | -5.4065% |

Correctness is stronger than the single-round timing:

- All 24 measured compact/hybrid/raw requests produced 256 tokens.
- At every context, the three token lists are elementwise identical.
- Every instrumented replay reproduced its corresponding measured token list.
- All eight hybrid state observations report `fatal=0`,
  `free_count=slot_count`, and zero raw pages.

The zero raw-page count means this sweep validates common compact-page
serving, planner accounting, and sidecar lifecycle. Exact overflow-to-raw
behavior is covered by direct CUDA tests, not yet by a forced-fallback
full-model E2E run.

### Round-1 Bottleneck and Next Measurement

The incremental hybrid gap is concentrated in the unfused writer and its
page-aware waves, not in changed QK/softmax/PV arithmetic. At context 4,096,
the profiled hybrid prepare/hydrate/append/commit/release chain executes 96
operations of each type, or three waves per layer, and totals about 102.0 ms.
Compact direct `reshape_and_cache` totals 14.79 ms. At context 16,384, hybrid
executes nine waves per layer and the chain totals about 417.2 ms, versus
59.63 ms for compact direct write. Raw-slot reset contributes another roughly
9 ms per profiled request. This launch and full-page hydrate/re-encode work
tracks the growing 4K--16K wall-time gap.

The 64-token result is a single-order anomaly: its wall gap is larger even
though the attributed CUDA operations do not explain that magnitude. It must
not be treated as a stable short-context regression without the remaining
Latin-square rounds.

Most importantly, round 1 was captured at checkpoint `7489f5f17`, before the
new fused hybrid Q1 writer currently under evaluation. It is the pre-fusion
baseline and cannot establish the fused path's retained E2E performance.
Formal publication evidence still requires the full three-round cyclic
compact/hybrid/raw sweep on the post-fusion binary, cross-round token
exactness, the same allocator-state checks, and renewed CUDA attribution. The
retention decision should use the three-round paired medians, not the numbers
above.

Artifacts:

- `profile/byte-v2-hybrid-workspace-e2e-a40-20260720/round1_compact.jsonl`
- `profile/byte-v2-hybrid-workspace-e2e-a40-20260720/round1_hybrid.jsonl`
- `profile/byte-v2-hybrid-workspace-e2e-a40-20260720/round1_raw.jsonl`
- `profile/byte-v2-hybrid-workspace-e2e-a40-20260720/analysis/summary.json`
- `profile/byte-v2-hybrid-workspace-e2e-a40-20260720/analysis/comparison.csv`

### Post-Q1-Fusion Preliminary Result

Checkpoint `56e155936` (`Fuse exact ByteV2 hybrid Q1 updates`) replaces the
decode-time hybrid prepare/hydrate/append/commit/persist/release chain with one
host op that launches three ordered CUDA kernels:

1. compact/raw hydrate, append, and compact-metadata clear;
2. the existing warp-histogram compact commit;
3. raw persistence, map-last publication, page-flag update, and transient-slot
   release.

The original FA2 reader and combine kernels are unchanged. Invalid raw maps,
raw-pool exhaustion, and an externally corrupted free-count fail closed. A
CUDA Graph reset/reuse test covers two raw pages, selective reset, raw-slot
reuse, and dummy replay. On GPU 6, the combined layout and analysis regression
run completed with 276 passed and one skipped test. Compute Sanitizer reported
zero errors for `memcheck` and `synccheck`, and zero hazards for `racecheck`,
on the real compact-overflow and CUDA Graph reset/reuse cases.

At context 1,024 in eager mode with 64 output tokens, the visible Q1 writer
cost changed from the five-op hybrid chain's 57.246 us/layer to 20.299
us/layer for the fused three-kernel op. The compact-only fused writer measured
21.204 us/layer in the same profiler setup. E2E throughput changed from 31.035
to 31.770 token/s, versus 31.754 token/s for compact-only V5. The hybrid FA2
reader itself remained effectively unchanged; the measured gain came from the
writer path.

A post-fusion compiled pre-sweep then reran the same eight contexts and 256
output tokens. The compact and raw columns below are retained round-1 runs;
only the hybrid column was rerun, so these paired ratios are directional and
must not be substituted for the final cyclic three-round result.

| Context | Fused hybrid tok/s | Versus round-1 compact | Versus round-1 raw |
| ---: | ---: | ---: | ---: |
| 64 | 37.530 | -0.015% | -2.109% |
| 128 | 37.407 | -0.112% | -2.248% |
| 512 | 36.923 | -0.052% | -2.525% |
| 1,024 | 36.319 | -0.140% | -2.546% |
| 2,048 | 34.963 | -0.892% | -3.151% |
| 4,096 | 32.991 | -1.139% | -2.877% |
| 8,192 | 29.146 | -2.008% | -3.675% |
| 16,384 | 23.030 | -2.932% | -4.122% |

All token lists match the corresponding compact and raw round-1 lists, every
profiler replay is exact, and all hybrid observations report zero raw pages,
`fatal=0`, and complete slot return. Across contexts, the directional median
hybrid/compact TPS gap is -0.516%; before Q1 fusion it was -3.343%. The
directional median hybrid/raw gap is -2.711%.

The remaining long-context gap is now attributable to the generic prefill
writer rather than Q1 decode. At 8K and 16K it executes five and nine waves per
layer. Its prepare/hydrate/append/commit/release CUDA totals are 205.988 and
416.515 ms/request, while compact `byte_v2_reshape_and_cache` totals 29.530 and
59.628 ms. The excesses, 176.458 and 356.888 ms, closely track the observed
wall-time gaps. Therefore the next performance target is a page-centric
three-kernel generic/prefill writer; further Q1 or FA2-reader tuning is not the
largest E2E opportunity at these lengths.

The pre-sweep preceded a defensive check that is taken only when an allocator
free-count is externally corrupted; the normal measured path allocated no raw
page. Formal publication numbers still require all variants to run on the
final binary in three cyclic orders, followed by strict automated analysis.

Additional artifacts:

- `profile/byte-v2-hybrid-workspace-e2e-a40-20260720/eager_hybrid_q1_fused_ctx1024.jsonl`
- `profile/byte-v2-hybrid-workspace-e2e-a40-20260720/cg_hybrid_q1_fused_ctx1024.jsonl`
- `profile/byte-v2-hybrid-workspace-e2e-a40-20260720/prefinal_hybrid_q1_fused.jsonl`

### Forced-Raw Serving Lifecycle Diagnostic

The normal E2E sweeps did not naturally exhaust the compact outlier pool, so
zero raw pages at request completion could not prove that a full model request
had traversed the raw reader, existing-raw writer, and scheduler reset paths.
An additive test-only diagnostic closes this evidence gap. It is enabled only
by the explicit profile flag:

```text
--diagnose-forced-raw-lifecycle
```

The worker then sets `BYTE_V2_TEST_FORCE_RAW_PROMOTION=1`. Default serving does
not allocate diagnostic state, require the diagnostic op, or launch it. In
diagnostic mode each layer adds three int32 values, or 12 bytes, and the
planner includes those bytes in the sidecar budget.

After ordinary warmup and CUDA Graph capture, the profile harness clears and
arms a device one-shot latch. Immediately after the next valid fused Q1
update, one 256-thread diagnostic CTA copies the transient 65,536-byte raw
page to a persistent raw slot. Each copy thread fences its own stores, a CTA
barrier follows, and thread 0 publishes `page_to_raw_slot` last. For a partial
page, only the prefix through the appended row is semantically valid; unused
tail rows remain unspecified and are masked by sequence length. Later Q1
updates hydrate only that valid prefix and overwrite the new target row.

The diagnostic keeps two witnesses after request teardown:

- `promotion_count` proves that the latch was consumed and a raw map was
  published;
- `mapped_page_visit_count` proves that a later Q1 completed while the page
  was already raw.

After each measured and profiler-replay request, the harness explicitly calls
the model runner's real `_zero_block_ids` path for the mapped physical block.
That path invokes raw-sidecar reset before compact zero. The JSON records this
as `runner_zero_block_ids_after_request`; this is an explicit block-reuse
lifecycle event rather than an assertion that request completion immediately
zeros all physical blocks.

On A40, context 1,024 and three output tokens passed in both eager and compiled
CUDA Graph modes. In each mode, all 32 layers reported the following for both
the measured request and its independent profiler replay:

```text
promotion_count          = 32
mapped_page_visit_count  = 32
request_reset_observed   = true
```

The final state had zero mapped raw pages, `free_count == slot_count`, and
`fatal == 0`. Unforced hybrid, forced-raw hybrid, and independent raw FA2 all
produced the same three tokens, `[13, 2579, 6307]`; both replay token lists
were also exact. The diagnostic JSON sets `performance_valid_for_tps=false`
because it deliberately adds one 64 KiB copy and one CUDA launch per layer.

Low-level eager and CUDA Graph tests additionally cover row-0 partial-page
promotion, same-step hybrid/raw FA2 bitwise equality, the next existing-raw
update, reset and slot reuse, dummy replay, and pool-exhaustion fail-closed
behavior. Compute Sanitizer reported zero memcheck/synccheck errors and zero
racecheck hazards for the eager and graph lifecycle cases. This diagnostic is
fault injection: natural compact overflow remains covered by separate direct
CUDA tests and must not be inferred from the forced E2E run.

Artifacts:

- `profile/byte-v2-hybrid-workspace-e2e-a40-20260720/forced_raw_eager_ctx1024.jsonl`
- `profile/byte-v2-hybrid-workspace-e2e-a40-20260720/forced_raw_compiled_ctx1024.jsonl`
- `profile/byte-v2-hybrid-workspace-e2e-a40-20260720/forced_raw_reference_fa2_ctx1024.jsonl`

### Generic/Prefill Three-Kernel Writer

Checkpoint `58dc9ff3f` (`Fuse exact ByteV2 hybrid multi-token updates`)
removes the remaining Q>1 writer chain. For every page-aware wave, the hybrid
path now launches exactly three kernels:

1. a page-owner kernel claims transient slots, hydrates each valid prefix from
   either compact V5 or its authoritative raw sidecar, overlays incoming BF16
   K/V, and clears compact metadata after all reads of that page complete;
2. the unchanged warp-histogram compact commit kernel;
3. the existing raw persist/map-last protocol, page-flag update, and transient
   release in one launch.

Q1 continues to use its specialized three-kernel implementation. If the new
multi-token op is absent from an older extension, Python falls back to the
previous generic chain; normal hybrid availability does not depend on the new
symbol. The implementation retains the existing append-prefix contract:
`valid_rows` is the largest row touched by the current wave plus one. Normal
vLLM KV append satisfies this contract; rewriting only low rows of a previously
complete page is outside both the old and new writer interfaces.

Direct differential tests use a 34-entry mapping that interleaves two pages
across three owner chunks, permutes non-contiguous rows, and includes dummy
slots. They cover compact and existing-raw hydration, all-dummy input, CUDA
Graph capture/reset/reuse, and subprocess fail-closed behavior for an invalid
raw map, a duplicated physical slot, insufficient transient capacity, and raw
pool exhaustion. Persistent sidecars are compared canonically by physical
page because concurrent allocation need not assign the same raw-slot number.

An initial Compute Sanitizer racecheck found that thread 0 could reuse a
shared status word for the first candidate page while slower threads were
still reading its initialization value. A CTA barrier before the candidate
loop removes that race. On the final binary, compact/existing-raw Q34 and CUDA
Graph replay report zero memcheck and synccheck errors and zero racecheck
hazards. The combined layout and strict-analysis regression reports 295 passed
and one skipped test before this barrier-only correction; the affected
targeted functional and sanitizer tests were rerun after rebuilding.

The direct A40 writer comparison below uses complete compact pages, no raw
promotions, six alternating-order samples, and the median CUDA-event time. It
times only the writer operations, excluding common input gathering:

| Tokens/pages per wave | Generic eager | Fused eager | Speedup | Generic graph | Fused graph | Speedup |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 / 8 | 0.13233 ms | 0.06263 ms | 2.11x | 0.12665 ms | 0.06121 ms | 2.07x |
| 512 / 32 | 0.40174 ms | 0.08267 ms | 4.86x | 0.39422 ms | 0.08118 ms | 4.86x |
| 2,048 / 128 | 1.43977 ms | 0.14351 ms | 10.03x | 1.43123 ms | 0.14052 ms | 10.19x |

The increasing gain is expected: the old chain repeatedly launches serial
prepare/hydrate/append work over the active-page prefix, while the owner
kernel scans the wave once per claimed page and performs one cooperative page
hydrate. The 2,048-token case matches the full waves that dominated the prior
8K and 16K E2E attribution.

### Final Three-Round Compiled E2E

The final comparison uses the post-race-fix binary from checkpoint
`58dc9ff3f`. All runs use one otherwise idle A40
(GPU 0), batch 1, 256 output tokens, prefix caching disabled, compiled CUDA
Graphs with size specialization, and contexts 64 through 16K. Each engine is
an independent process. The cyclic order is:

```text
round 1: compact -> hybrid -> raw
round 2: hybrid  -> raw    -> compact
round 3: raw     -> compact -> hybrid
```

The table reports the median TPS for each variant. Ratio columns are medians
of the three within-round paired ratios, so they need not equal ratios of the
displayed median TPS values.

| Context | Compact tok/s | Hybrid tok/s | Raw tok/s | Hybrid/compact | Hybrid/raw |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 64 | 37.3429 | 37.3158 | 38.2106 | -0.2424% | -2.3013% |
| 128 | 37.1058 | 37.0536 | 38.0485 | -0.0699% | -2.6409% |
| 512 | 36.6589 | 36.6042 | 37.6528 | +0.0853% | -2.6790% |
| 1,024 | 36.1423 | 36.0988 | 37.0975 | +0.0813% | -2.6647% |
| 2,048 | 35.1754 | 34.9763 | 36.0787 | -0.4870% | -2.9496% |
| 4,096 | 33.2925 | 33.2010 | 33.9645 | -0.2884% | -2.2479% |
| 8,192 | 29.6486 | 29.6782 | 30.2860 | +0.0930% | -2.0215% |
| 16,384 | 23.6997 | 23.7174 | 24.0324 | +0.0748% | -1.3107% |

Across the eight contexts, the median of the paired hybrid/compact gaps is
+0.0024%, with a range from -0.4870% to +0.0930%. The corresponding
hybrid/raw median is -2.4711%, ranging from -2.9496% to -1.3107%. Compact V5
itself has a -2.3344% median gap to raw FA2, so the persistent raw-fallback
machinery adds no measurable median throughput cost over compact serving in
this protocol. The remaining raw gap is predominantly the compact ByteV2
load/decode cost inside the otherwise unchanged FA2 template.

Correctness and state validation pass before aggregation:

- all 72 measured rows return exactly 256 tokens;
- compact, hybrid, and raw token IDs are elementwise identical at every
  context in every round and remain identical across rounds;
- all 72 instrumented profiler replays reproduce their measured token IDs;
- all 24 hybrid observations report `fatal=0`, zero mapped raw pages, and
  `free_count=slot_count=1472` across the 32 layers.

The three rounds reproduce the same complete memory plan. Hybrid provides
12,001 blocks versus 9,591 for raw BF16 under the fixed approximately 20.11 GB
KV budget, a 25.1277% capacity increase. At the same 12,001-block capacity,
the complete hybrid allocation, including 98,011,264 bytes of persistent raw
sidecars and the 8,437,644-byte shared workspace, saves 20.0849% versus raw
BF16.

These results support a configuration-specific claim of exact generated
tokens, about 20.1% complete KV-memory reduction at equal capacity, and at
most 3.0% paired TPS regression to raw FA2 over the tested context sweep. They
do not by themselves establish task-level accuracy on an evaluation suite,
model-family generality, multi-request serving scalability, or a universal
mathematical losslessness claim. The forced-raw lifecycle result above is
correctness evidence and remains excluded from TPS aggregation.

Artifacts:

- `profile/byte-v2-hybrid-final-e2e-a40-20260720/round{1,2,3}_{compact,hybrid,raw}.jsonl`
- `profile/byte-v2-hybrid-final-e2e-a40-20260720/analysis/summary.json`
- `profile/byte-v2-hybrid-final-e2e-a40-20260720/analysis/comparison.csv`

### Batch-8 Cooperative Decode Writer

The exact initial-prefill path exposed a different bottleneck under continuous
batching. With eight requests, decode supplies eight Q1 cache updates per
layer. The multi-token writer originally launched one page-owner CTA for the
entire 16-token owner chunk, so that CTA hydrated as many as eight independent
pages serially. In a CUDA Graph microbenchmark its median time was 198.656 us,
versus 13.312 us for raw `reshape_and_cache_flash`.

A first page-parallel version launched one owner candidate per token and
reduced the batch-8 median to 70.656 us. The retained implementation uses a
more cooperative four-kernel protocol for `2 <= N <= 16` when the active
staging capacity is also at most 16 pages:

1. one thread validates the complete wave before allocator mutation, assigns
   dense staging slots to unique physical pages, rejects duplicate
   `(page,row)` slots, and packs an eight-CTA completion count above
   `valid_rows`;
2. eight CTAs per active page partition compact/raw hydration and BF16 overlay;
3. the unchanged warp-histogram commit re-encodes the compact page;
4. the existing persist/map-last/release kernel handles real overflow pages and
   clears transient state.

Each hydrate CTA writes a disjoint `work_idx` partition and decrements the
packed completion count once. The last CTA clears compact metadata and
publishes `block_to_staging_slot`. There is no spin loop or grid-wide barrier;
the following commit runs on the same stream after a kernel boundary. Larger
waves retain the generic owner kernel. A legal native call with more than 16
staging pages also retains the generic path rather than trapping in the small
path.

The final CUDA Graph microbenchmark used complete compact pages, distinct
physical pages, 100 warmups, 500 alternating-order samples, and direct hybrid
FA2-versus-raw output/LSE checks:

| Batch/pages | Hybrid writer | Raw writer | Extra / layer |
| ---: | ---: | ---: | ---: |
| 1 | 13.312 us | 13.312 us | 0.000 us |
| 2 | 16.384 us | 13.312 us | 3.072 us |
| 4 | 24.576 us | 13.312 us | 11.264 us |
| 8 | 30.720 us | 13.312 us | 17.408 us |

At batch 8 this is 6.47x faster than the original generic writer and 2.30x
faster than the first page-parallel version. It removes 90.61% of the original
writer excess over raw. The remaining writer upper bound for 32 layers and 61
decode steps is about 34.0 ms/request.

The full-model E2E comparison uses the same eight ShareGPT prompts with actual
lengths `[512, 768, 1024, 1279, 1506, 2072, 2925, 3796]`, 64 output tokens per
request, prefix caching disabled, a 1,042-page exact-prefill staging workspace,
compiled CUDA Graphs, and separate engines. Round 2 reverses raw/hybrid order.

| Round | Hybrid tok/s | Raw tok/s | Hybrid/raw | Wall-time excess |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 117.0285 | 121.4614 | -3.6496% | 150.3 ms |
| 2 | 116.0896 | 121.9079 | -4.7728% | 198.2 ms |
| 3 | 116.1214 | 120.7801 | -3.8572% | 160.1 ms |

The paired median is -3.8572%, versus -19.9269% for the original generic
batch-8 writer. Median hybrid TPS improves by 20.08%; the raw-relative gap is
16.07 percentage points smaller. All hybrid observations finish with
`fatal=0`, zero mapped raw pages, and `free_count=slot_count=1472`.

This configuration plans 11,979 hybrid blocks versus 9,601 raw blocks, a
24.7683% capacity increase under the approximately fixed KV budget. At the
same 11,979-block capacity, the complete hybrid allocation is 4,985,575,100
bytes smaller than raw BF16, a 19.8456% saving. This includes the 98,008,448
byte persistent sidecar and 68,344,772 byte shared workspace.

The compiled token result needs a stricter qualification than the earlier
single-backend profiler field implied. Hybrid is internally repeatable across
all three rounds and raw is internally repeatable across all three rounds, but
only seven of eight requests match across backends. Request 5 first diverges at
generated-token index 28:

- hybrid reports tokens 1687 and 2181 tied at logprob `-0.770114` and greedy
  selection returns 1687;
- raw reports token 2181 at `-0.705454` and token 1687 at `-0.830454`, a 0.125
  margin.

The trajectories differ for two tokens and then rejoin. A diagnostic eager
run on the identical prompts produces elementwise-identical token IDs and
elementwise-identical serialized top-5 logprobs at every generation step.
Direct cache round trips and hybrid/raw FA2 comparisons also remain bitwise.
The discrepancy is therefore specific to the separately compiled backend
graphs, which already produce slightly different logits at the first
generation step; it is not evidence of a race in the cooperative writer.
Conversely, these results cannot support a compiled E2E bitwise-exact claim.

The profile script now records both `same_backend_nonspec_match` and the
independent `raw_fa2_reference_match`, plus mismatching request indices. The
old `reference_match` column compared a backend's spec-0 row to itself and was
not a cross-backend correctness check.

Correctness coverage includes compact and existing-raw sources, shared and
permuted pages, dummy entries, exact duplicate rejection, the N=16/N=17
dispatch boundary, N=16 CUDA Graph capture/reset/replay, compact-to-raw
promotion, raw-pool exhaustion, and transient allocator cleanup. Compute
Sanitizer reports zero memcheck errors, zero racecheck hazards, and zero
synccheck errors on the new small path.

Nsight Systems capture-time samples attribute as much as 10.336 us to the
serial prepare kernel and 14.880 us to cooperative hydrate at batch 8; commit
and persist remain about 3 us and 2 us. Eliminating the separate prepare launch
is the next writer-specific opportunity. At full E2E scope, however, the
remaining paired median wall gap is 160.1 ms, so Q1 compact loading, prefill
lifecycle/reset work, and CUDA Graph/host overhead must be measured alongside
any further writer change. The current data support neither "unchanged TPS"
nor a universal lossless-serving claim; broader task accuracy, model-family,
concurrency, and natural-overflow studies are still required.

Additional artifacts:

- `profile/byte-v2-retained-prefill-a40-20260720/sharegpt_batch8_cooperative_writer_compiled.jsonl`
- `profile/byte-v2-retained-prefill-a40-20260720/sharegpt_batch8_cooperative_writer_round2_{hybrid,raw}.jsonl`
- `profile/byte-v2-retained-prefill-a40-20260720/sharegpt_batch8_cooperative_writer_round3.jsonl`
- `profile/byte-v2-sharegpt-exactness-a40-20260720/batch8_cooperative_writer_{compiled,eager}_logprobs.jsonl`

### Adaptive Small-Batch Cooperative Sharding

The next experiment audited whether the serial prepare launch could be folded
into the cooperative hydrate kernel. The direct design was rejected before
implementation. Without a grid barrier or a generation counter, a late CTA
cannot distinguish stale `valid_rows` state from completion state written by
an earlier CTA. Letting each page publish independently also allows one page
to mutate maps while another page is still validating the clean allocator
state. Accepting that protocol would weaken the existing fail-closed
guarantee. A conditionally safe design would need another manager-owned
completion scalar and a global-last publisher, but the maximum E2E upside from
removing prepare is less than about 0.5 percentage point. The production path
therefore retains the separate prepare boundary.

A lower-risk follow-up keeps the allocator, completion, commit, persist, and
release protocols unchanged and only adapts the number of hydrate shards per
page. The prior eight-CTA setting underfills the GPU for two to four active
staging slots. The retained policy fills approximately one 64-CTA wave for
`N <= 4`, where `N=min(num_tokens, staging_capacity)`:

| Small-stage slots | CTAs per page |
| ---: | ---: |
| 2 | 32 |
| 3 | 21 |
| 4 | 16 |
| 5--16 | 8 |

The CUDA Graph A/B used complete compact pages, a raw FA2 oracle, and
alternating execution order. A paired `N=2..16` sweep with identical slot/row
inputs produced:

| Batch/pages | Fixed 8 CTAs | 64-wave policy | Change |
| ---: | ---: | ---: | ---: |
| 2 | 16.384 us | 16.384 us | 0.0% |
| 3 | 18.432 us | 17.408 us | -5.6% |
| 4 | 19.456 us | 18.432 us | -5.3% |
| 8 | 26.624 us | 26.624 us | 0.0% |

Trying to keep 64 total CTAs by reducing the per-page shard count above batch
eight regressed `N=9..16` by 3.6% to 17.1%. The final policy consequently
retains eight shards for every `N >= 5`; in particular the established batch-8
and batch-16 paths are unchanged. A separate representative ShareGPT-prefix
run with 100 warmups and 500 samples improves batch 4 from 24.576 us to
21.504 us (-12.5%), versus 13.312 us for raw. The unchanged batch-8 result is
30.720 us versus 13.312 us for raw.

New regression coverage compares generic and adaptive writers bitwise for two,
three, and four distinct compact pages at the append-prefix row. Output, LSE,
compact cache bytes, allocator state, and page flags match. An eager update
followed by three CUDA Graph replays at `N=4` reports zero memcheck errors,
zero racecheck hazards, and zero synccheck errors. This optimization targets
small or shrinking dynamic batches and does not change the earlier batch-8
E2E result by construction.

A three-round batch-4 compiled E2E comparison used ShareGPT prompt lengths
`[512, 768, 1024, 1279]`, 64 output tokens, speculative decoding disabled,
prefix caching disabled, and alternating backend order:

| Round | Hybrid tok/s | Raw tok/s | Hybrid/raw | Wall excess |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 93.0588 | 106.2078 | -12.3805% | +340.58 ms |
| 2 | 102.1808 | 106.0990 | -3.6929% | +92.52 ms |
| 3 | 102.1571 | 105.8642 | -3.5017% | +87.75 ms |

Round 1 is a visible slow outlier, while rounds 2 and 3 are stable. Retaining
all rounds, the paired median is -3.6929% and +92.52 ms. All four requests
produce elementwise-identical token IDs across backends in every round, both
backends are internally repeatable, and all profiler replays reproduce the
measured tokens. Hybrid state remains `fatal=0`, with zero raw pages and
`free_count=slot_count=1472` in every round.

The stable memory plan remains 11,979 hybrid blocks versus 9,601 raw blocks,
or 24.7683% more block capacity under the fixed budget. At equal 11,979-block
capacity, the complete hybrid allocation is 4,985,575,100 bytes, or 19.8456%,
smaller than raw BF16. This E2E result validates the current small-batch path
against raw FA2; it does not isolate the adaptive-shard gain from the prior
fixed-eight-CTA implementation.

### Fixed Decode-Length and Batch Matrix

The earlier ShareGPT runs treated `max_tokens` as an upper bound. In
particular, the batch-8, 64-token workload generated
`[64, 64, 64, 64, 64, 64, 34, 64]` tokens because one request stopped at EOS.
That result measures a real stopping workload, but it also includes a dynamic
batch shrink and is not a fixed decode-length comparison. The profiler now
has an explicit `--ignore-eos` mode that fails closed unless every measured
and replayed request produces exactly `max_tokens` tokens.

The fixed-length matrix used compiled CUDA Graph execution, speculation
disabled, prefix caching disabled, separate hybrid/raw engines, and three
paired rounds ordered hybrid/raw, raw/hybrid, and hybrid/raw. The batch sizes
were 1, 2, 4, and 8, and the decode lengths were 16, 64, and 256. Every batch
uses a prefix of the same ShareGPT prompt set with lengths
`[512, 768, 1024, 1279, 1506, 2072, 2925, 3796]`.

All cells keep `max_model_len=max_num_batched_tokens=16,456`. Because the
profiler derives this value as `context_len + max_tokens + 8`, the context
caps are 16,432, 16,384, and 16,192 for decode lengths 16, 64, and 256. The
caps only select the fixed prompts; they are not the actual prompt lengths.
This avoids changing engine capacity, staging planning, or chunked-prefill
behavior as decode length changes. The four batch subsets ran concurrently on
four otherwise idle A40 GPUs, while every paired hybrid/raw cell stayed on the
same GPU and executed back to back.

The main result uses the median of the three per-round hybrid/raw TPS ratios;
it does not divide two independently computed TPS medians:

| Batch | Decode | Hybrid tok/s | Raw tok/s | Paired gap | Wall excess | Exact requests |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 16 | 32.720 | 34.064 | -3.9576% | +19.33 ms | 1/1 |
| 1 | 64 | 36.115 | 37.235 | -2.8440% | +50.20 ms | 1/1 |
| 1 | 256 | 36.933 | 37.839 | -2.3570% | +163.38 ms | 1/1 |
| 2 | 16 | 49.627 | 51.496 | -3.6293% | +23.40 ms | 2/2 |
| 2 | 64 | 60.568 | 62.342 | -2.8399% | +60.01 ms | 2/2 |
| 2 | 256 | 64.042 | 65.707 | -2.5344% | +202.62 ms | 2/2 |
| 4 | 16 | 70.405 | 74.726 | -5.7369% | +52.11 ms | 4/4 |
| 4 | 64 | 116.806 | 122.738 | -4.8335% | +105.94 ms | 4/4 |
| 4 | 256 | 139.687 | 146.173 | -4.4345% | +325.08 ms | 2/4 |
| 8 | 16 | 48.279 | 52.357 | -7.5624% | +200.50 ms | 8/8 |
| 8 | 64 | 118.138 | 126.745 | -6.7913% | +294.33 ms | 7/8 |
| 8 | 256 | 184.493 | 196.298 | -5.9733% | +663.20 ms | 2/8 |

The B2/D16 first round is a cold slow sample at -9.6531%; the other two
rounds are -3.6293% and -3.5483%, so the all-round median remains stable.
No sample was removed. All other cells have a per-round gap range narrower
than 1.08 percentage points.

An isolated fourth D64 round ran every batch sequentially on GPU 0 with raw
before hybrid. It reports -2.9332%, -2.8650%, -4.2551%, and -6.9189% for
batches 1, 2, 4, and 8. This same-GPU calibration reproduces the matrix trend
and completes an ABBA backend-order check at the representative decode length.

The absolute wall gap grows nearly linearly with decode length. A linear fit
to wall excess versus requested decode steps gives incremental hybrid overhead
of 0.597, 0.746, 1.139, and 1.926 ms per additional decode step for batches
1, 2, 4, and 8. The independently measured writer excess accounts for about
0%, 13%, 23%, and 29% of those slopes, respectively. The relative TPS gap
narrows for longer generation because fixed prefill and graph overhead is
amortized, but the absolute decode cost continues to accumulate. The writer
therefore remains relevant at larger batches, but it is not the majority of
the remaining E2E gap; Q1 compact decode/loading and the batch-8 fixed
prefill/graph component remain the primary attribution targets.

Every one of the 72 main-matrix rows and eight isolated-calibration rows has
the requested output length, matching measured/profile-replay tokens, a valid
TPS flag, and stable prompt hashes. All hybrid runs finish with `fatal=0`, no
raw pages, and every sidecar slot free. At the measured plans, the complete
hybrid allocation saves 20.085% to 20.088% at equal block capacity and exposes
25.141% to 25.143% more blocks under the approximately fixed budget.

Cross-backend token exactness has a separate qualification. Decode lengths 16
and 64 are exact through batch 4. At batch 8 and length 64, request index 5
first differs at generated-token offset 28, reproducing the previously known
compiled near-tie. At length 256, batch 4 differs for request indices 1 and 2,
first at offsets 232 and 104. Batch 8 differs for indices 1, 2, 4, 5, 6, and
7, first at offsets 232, 104, 124, 28, 86, and 90. Each backend is internally
identical across all three rounds, and each shorter output is an exact prefix
of its longer output for the same backend. These are deterministic compiled
cross-backend trajectory splits, not replay failure or run-to-run allocator
nondeterminism. The longer continuation creates more opportunities for a
small logits difference to cross a greedy decision boundary and then amplify
autoregressively. The new data do not by themselves prove that every first
split is a near-tie, so top-logprob or eager diagnostics are still required
before attributing the later offsets as precisely as the known offset-28
case.

For performance, forcing generation beyond EOS is useful because both arms do
identical work. It is not a claim about user-visible generation quality. The
compiled token mismatches also mean this matrix cannot establish bitwise E2E
losslessness, even though the direct cache and attention comparisons remain
bitwise and the fixed-length performance samples are structurally valid.

Artifacts:

- `profile/byte-v2-e2e-batch-decode-matrix-a40-20260721/b{1,2,4,8}/r{1,2,3}_*.jsonl`
- `profile/byte-v2-e2e-batch-decode-matrix-a40-20260721/b{1,2,4,8}/r4_*_d64_*.jsonl`

### Fixed-Budget Memory-Pressure Decode

The fixed-length matrix above did not exercise the capacity advantage: its
largest live request set was still well below the raw BF16 cache limit. The
memory-pressure experiment therefore decouples the engine limit from each
workload length and fixes the usable KV-cache budget directly. All runs use
one A40, Llama-3.1-8B-Instruct in BF16, compiled CUDA Graph execution,
speculation disabled, prefix caching disabled, exact decode lengths, and
`max_model_len=max_num_batched_tokens=16,384`. The cache budget is exactly
20,000,000,000 bytes for both backends. Long screening cells skip the
instrumented replay so the clean workload is not executed twice.

The resulting plans are:

| Backend | Planned bytes | Blocks | Capacity tokens | Bytes/capacity token |
| --- | ---: | ---: | ---: | ---: |
| ByteV2 hybrid | 19,999,610,108 | 11,933 | 190,928 | 104,749.5 |
| Raw BF16 FA2 | 19,998,441,472 | 9,536 | 152,576 | 131,072.0 |

At the same approximately 20 GB budget, ByteV2 exposes 25.1363% more token
capacity. Equivalently, its complete allocation, including the persistent raw
sidecar and shared staging workspace, uses 20.0825% fewer bytes per unit of
capacity. The peak demand below is computed per request as
`ceil((prompt + output - 1) / 16)` blocks. The final sampled token has no
subsequent forward pass and is therefore not resident in KV cache.

The first sweep fixes batch 32 and a 4,096-token prompt, then increases the
decode length across the raw-only and both-backends pressure boundaries:

| Decode | Peak blocks | ByteV2 pressure / preempt | Raw pressure / preempt | ByteV2 tok/s | Raw tok/s | ByteV2/raw |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 8,704 | 0.7294 / 0 | 0.9128 / 0 | 173.742 | 233.394 | -25.5587% |
| 1,024 | 10,240 | 0.8581 / 0 | 1.0738 / 3 | 256.594 | 334.570 | -23.3065% |
| 1,536 | 11,264 | 0.9439 / 0 | 1.1812 / 5 | 279.684 | 375.256 | -25.4680% |
| 2,048 | 12,288 | 1.0297 / 1 | 1.2886 / 8 | 264.089 | 319.630 | -17.3765% |

Raw preemption increases monotonically and the relative gap narrows once both
backends cross their capacity limits, but the current batch-32 ByteV2 path
does not overtake raw even after avoiding three to five recomputations. The
batch-32/decode-1,024 point was repeated in reverse backend order on the same
GPU. It reports -23.9823%, versus -23.3065% in the forward order; ByteV2 and
raw TPS individually change by only -0.5805% and +0.3033%. The large negative
result is consequently not a backend-order artifact.

The second sweep fixes total prompt tokens at 131,072, total output tokens at
32,768, and padded peak demand at 10,240 blocks, while exchanging batch width
for sequence length:

| Batch | Prompt | Decode | ByteV2 tok/s / preempt | Raw tok/s / preempt | ByteV2/raw |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 8,192 | 2,048 | 220.610 / 0 | 190.705 / 2 | +15.6812% |
| 32 | 4,096 | 1,024 | 256.594 / 0 | 334.570 / 3 | -23.3065% |
| 64 | 2,048 | 512 | 437.289 / 0 | 554.604 / 5 | -21.1529% |
| 128 | 1,024 | 256 | 689.543 / 0 | 833.568 / 9 | -17.2781% |

The positive batch-16 result is repeatable. A same-GPU reverse-order run gives
218.978 tok/s for ByteV2 and 190.778 tok/s for raw, or +14.7817%, while the
wall time falls by 22.1195 seconds. Raw has two preemptions and ByteV2 has zero
in both orders. Across the two observations, raw TPS changes by 0.0380% and
ByteV2 TPS by -0.7399%. A same-shape batch-16/prompt-8,192/decode-256 control
keeps both backends below capacity and reports 101.500 versus 113.463 tok/s,
or -10.5434%. The transition from -10.54% without pressure to approximately
+15% under raw-only pressure demonstrates that the win comes from avoiding
the raw capacity/recompute cliff, not from a generally faster ByteV2 kernel.

One raw batch-32/prompt-4,096/decode-1,024 diagnostic retains the replay and
samples scheduler state after every step. The measured counter, replay
counter, and direct scheduler wrapper each record exactly three preemptions.
They discard 14,732 already-computed KV tokens in total, reach 100% peak KV
usage, and leave as many as 28 requests waiting for capacity. The replay
records seven chunked-prefill steps and reproduces all measured tokens. This
is direct evidence of KV release and recomputation rather than a conclusion
drawn only from a static pressure ratio.

The batch-shape discontinuity coincides with two code-level path changes that
are the leading attribution candidates. The steady attention reader at batch
32 is still the hybrid Q1 FA2 kernel; it does not fall back to the legacy
paged-decode implementation. The cache writer, however, uses the cooperative
small path only for at most 16 tokens/staging pages. Batch 16 launches 128
hydrate CTAs, whereas batch 32 falls back to two generic owner CTAs that
serially process roughly 16 candidate pages each. Independently, the
inherited FA2 split heuristic crosses its A40 occupancy threshold between
batch 16 and 17. Batch 16 normally uses a split kernel with the 48 KiB K/V
alias and two resident CTAs per SM; batch 32 uses the approximately 80 KiB
nonsplit kernel and one resident CTA per SM. The raw path sees the same split
decision, but the ByteV2 loader's decode and metadata work makes lower
residency a plausible backend-specific cost. Dedicated A/B experiments are
still required to quantify each contribution.

Every screening and confirmation request produces its exact requested length.
All 16, 32, 64, and 128-request cross-backend comparisons are elementwise
token-identical. Every hybrid run ends with all 32 layers initialized,
`fatal=0`, no mapped raw pages, and all 1,472 sidecar slots free. These
synthetic greedy results establish structural correctness for this matrix;
they do not replace broad task-accuracy evaluation.

The E2E conclusion is therefore conditional but useful: under the same memory
budget, ByteV2's 25.14% capacity increase can become a stable approximately
15% throughput win when raw crosses its KV limit, as demonstrated at batch 16
with long sequences. It is not yet a general win at larger batches because
the avoided recomputation is smaller than the observed ByteV2 overhead there.
The next optimization should first measure a two-wave cooperative writer for
batch 32, then test a nonsplit 48 KiB K/V-alias specialization that preserves
the original FA2 reduction order. QK, softmax, PV, split reduction, and
combine should remain unchanged.

Artifacts:

- `profile/byte-v2-memory-pressure-a40-20260721/capacity/`
- `profile/byte-v2-memory-pressure-a40-20260721/screening/`
- `profile/byte-v2-memory-pressure-a40-20260721/screening_confirm/`
- `profile/byte-v2-memory-pressure-a40-20260721/diagnostic/`

### Cooperative Batch-32 Writer and Realtime E2E Path

The batch-32 writer cliff identified above is now removed on the realtime
path. The implementation does not invoke two independent 16-token writers.
Instead, one thread first validates all 17--32 input tokens, merges rows by
physical page, rejects duplicate `(page, row)` entries, checks every raw map
and staging bound, and only then claims the unique pages. A single
`grid=(8, unique_pages)` hydrate launch lets the GPU schedule the page CTAs in
hardware waves. The existing compact commit, raw persist/map-last publication,
transient release, QK, softmax, PV, split reduction, and FA2 combine code are
unchanged.

The cooperative prepare limit is 32 instead of 16. The default dispatch uses
the new limit; setting
`BYTE_V2_HYBRID_COOPERATIVE_WRITER_LIMIT=16` restores the old generic dispatch
inside the same compiled extension for paired A/B measurements. The only
accepted values are 16 and 32, and an invalid value raises before launching a
kernel. This switch changes no custom-op schema and is read once per process,
so CUDA Graph replay contains only the captured kernel launches.

An interleaved CUDA-event microbenchmark used one append per request, distinct
physical pages, 100 warmups, 1,000 measured Graph replays, and the actual
hybrid FA2 reader as a bitwise oracle:

| Batch | Old limit 16 | Cooperative limit 32 | Change |
| ---: | ---: | ---: | ---: |
| 8 | 30.656 us | 29.696 us | -3.13% |
| 16 | 42.928 us | 41.984 us | -2.20% |
| 17 | 388.096 us | 44.032 us | -88.65% |
| 32 | 391.136 us | 75.776 us | -80.63% |

All four candidate cases have zero hybrid/raw output and LSE mismatches, clean
transient state, and a fully returned persistent allocator. The N8/N16
differences are within the approximately 1-us event quantization and show no
regression. N17 and N32 no longer fall into the two-owner-CTA generic path.

Correctness coverage now includes the old and new dispatch boundaries
`N={16,17,32,33}`, an N32 CUDA Graph reset/replay/dummy cycle, a duplicate
slot split across token indices 0 and 31, and 32 unique pages with only 31
staging slots. The latter two cases trap before successful publication. The
compact and existing-raw source cases, page permutations, adaptive CTA cases,
logical BF16 cache reads, FA2 output/LSE, page flags, and allocator partition
remain exact against the generic reference. Compute Sanitizer reports zero
memcheck errors, zero synccheck errors on the N32 Graph replay, and zero
racecheck hazards.

The clean fixed-budget E2E gate used one A40, BF16 Llama-3.1-8B-Instruct,
batch 32, a 4,096-token prompt per request, 256 forced output tokens, compiled
CUDA Graph execution, no speculation, no prefix caching, and the exact
20,000,000,000-byte KV budget. Three control/candidate runs were ordered
control/candidate, candidate/control, and control/candidate:

| Round | Old writer seconds / tok/s | B32 writer seconds / tok/s |
| ---: | ---: | ---: |
| 1 | 47.131210 / 173.813 | 40.817264 / 200.699 |
| 2 | 47.719661 / 171.669 | 40.961616 / 199.992 |
| 3 | 47.388192 / 172.870 | 40.725993 / 201.149 |

The medians are 47.388192 versus 40.817264 seconds and 172.870 versus
200.699 tok/s. The new writer therefore reduces wall time by 13.8662% and
raises TPS by 16.0984%. Control varies by 1.25% across the three runs and the
candidate by 0.58%, so the signal is much larger than run-to-run noise. Every
candidate and control request matches the raw FA2 token sequence exactly,
both schedulers report zero preemptions, and all candidate runs finish with
32 initialized layers, `fatal=0`, no authoritative raw pages, and all 1,472
sidecar slots free.

Two adjacent raw FA2 references have a 35.268312-second and 232.277-tok/s
median. Relative to that reference, the old ByteV2 path was 34.3648% slower in
wall time and 25.5760% lower in TPS. The new path is 15.7335% slower and
13.5949% lower in TPS. It removes 54.2161% of the absolute ByteV2/raw wall gap
without using the compression capacity advantage or inducing raw preemption.

Request-level medians show where the improvement occurs:

| Metric | Old writer | B32 writer | Raw FA2 |
| --- | ---: | ---: | ---: |
| Queue | 9.336711 s | 9.290619 s | 5.909442 s |
| First-token latency | 16.203130 s | 16.142097 s | 11.887259 s |
| Decode span | 30.983789 s | 24.473965 s | 23.232500 s |
| Engine E2E | 47.186614 s | 40.615751 s | 35.119429 s |

The writer change reduces the median decode span by 21.01%, while first-token
latency is effectively unchanged. After this optimization, only about 1.24
seconds of the median request gap to raw is in decode, whereas about 4.25
seconds is present before the first token. For the E2E objective, initial and
mixed prefill consequently become a higher-priority target than another Q1
decode micro-optimization.

An instrumented B32/P4096/D1 replay localizes that prefill gap. With the
default 128-slot cross-layer workspace, each four-request 16,384-token
prefill step is split into 12 staging waves and cannot retain one complete
exact-BF16 lease. The hybrid Q4096 cache-reader call accumulates 4,334.596 ms,
versus 1,926.284 ms for raw FA2; the multi-token writer accumulates 439.502 ms.
These are nested event scopes and must not be added together, but the
approximately 2.41-second attention difference matches the observed D1 wall
gap.

The existing `BYTE_V2_FA2_RAW_STAGING_SLOTS` switch provides a no-code upper
bound. A conservative 1,028-slot workspace fits all four 4,096-token requests
in one staging wave. At D1 it changes ByteV2 from 21.167044 to 19.046222
seconds, while the paired raw result is 18.819670 seconds; tokens remain exact.
However, the full D256 result is only 40.343576 seconds and 203.056 tok/s,
about 1.16% faster than the 128-slot candidate median. It also consumes about
59 MB more shared transient storage and reduces the fixed-budget plan from
11,933 to 11,897 compact blocks.

The small D256 gain has a structural explanation. Once the first prompt group
starts decoding, later prompt groups share a scheduler step with requests that
already have cached context. The current `has_cached_context` decision routes
the entire mixed batch through the hybrid cache prefill reader, so the larger
workspace only helps the first pure-prefill group. Increasing the default
workspace is therefore not retained as the next main-path optimization.

The mixed-prefill route remains the structural TTFT target, but a direct
implementation review found that splitting cached Q1 requests out of a mixed
FA2 batch changes the GQA/split dispatch unless it uses a special dispatch
sentinel. The lower-risk nonsplit 48-KiB K/V shared-memory alias was therefore
executed first; its result is recorded below.

Artifacts:

- `profile/byte-v2-realtime-writer-b32-a40-20260721/control_r{1,2,3}.jsonl`
- `profile/byte-v2-realtime-writer-b32-a40-20260721/candidate_r{1,2,3}.jsonl`
- `profile/byte-v2-realtime-writer-b32-a40-20260721/prefill_profile_candidate.jsonl`
- `profile/byte-v2-realtime-writer-b32-a40-20260721/prefill_slots1028_d1.jsonl`
- `profile/byte-v2-realtime-writer-b32-a40-20260721/slots1028_d256_r1.jsonl`

### Realtime Nonsplit FA2 K/V Alias

The B17--32 Q1 attention cliff is now removed without replacing the FA2
attention implementation. The ByteV2 external loader has a dedicated
nonsplit capability. When enabled, `sK` and `sV` alias the same 32-KiB shared
tile and use the already validated serialized schedule
`K -> QK -> V -> PV -> next K`. Q, QK, masking, softmax, PV, output stores,
split reduction, and the original combine kernel are unchanged. Raw FA2 is
unchanged at the specialization, code-path, and launch-behavior level.

`BYTE_V2_FA2_REUSE_KV_SMEM_NONSPLIT` accepts only `0` or `1`. It is read once
per process before the first nonsplit launch, so CUDA Graph replay contains a
fixed kernel. The validated realtime path is now the default when the variable
is unset; `0` restores the prior 80-KiB nonsplit kernel in the same binary.
Split-K always uses its established 48-KiB specialization and does not inspect
the nonsplit selector.

The shared-memory lifetime is safe for both output types. Nonsplit BF16 output
uses the original 16-KiB Q region and cannot overwrite the aliased K/V region.
Split-K FP32 output is larger and can overlap it, so the original split
epilogue barrier remains intact. The launch and device-side reuse predicates
are identical, preventing a 48/80-KiB layout mismatch.

NCU on A40 at B17/C4096 confirms the intended resource transition:

| Metric | Prior nonsplit | K/V alias |
| --- | ---: | ---: |
| Shared memory per CTA | 82.94 KiB | 50.18 KiB |
| Shared-memory occupancy limit | 1 CTA/SM | 2 CTAs/SM |
| Registers per thread | 243 | 241 |
| Grid / waves per SM | 136 / 1.62 | 136 / 0.81 |
| Instrumented kernel time | 703.97 us | 502.66 us |

There is no local-memory allocation in either nonsplit specialization. The
instrumented duration includes NCU overhead; the interleaved CUDA-event result
below is the performance gate.

The isolated Graph benchmark uses 100 warmups, 500 measured replays, three
independent inputs, and interleaved hybrid/raw events. The key long-context
medians are:

| Context / batch | Prior ByteV2 | K/V alias | Change | Raw FA2 |
| --- | ---: | ---: | ---: | ---: |
| 4096 / 16 | 411.648 us | 411.648 us | 0.00% | 483.328 us |
| 4096 / 17 | 594.944 us | 432.128 us | -27.37% | 492.544 us |
| 4096 / 32 | 1168.384 us | 860.160 us | -26.38% | 1006.592 us |

The B16-to-B17 ByteV2 jump falls from 44.53% to 4.98%. At context 1024, B16,
B17, and B32 improve by 23.2--23.9%. A separate small-active-batch gate covers
`B={1,2,4,8}` and contexts 128, 1024, and 4096. Eight of twelve candidate
medians are identical to control; the other four differ by only
0.512--1.024 us with no consistent regression. A host batch threshold is
therefore not retained.

Correctness covers one tile, multiple tiles, the masking-to-unmasked boundary,
ragged lengths, permuted pages, shared prefixes, outliers, compact pages,
authoritative raw-sidecar pages, explicit split 4, CUDA Graph replay, and both
selector values. All direct output and LSE comparisons are bitwise exact.
The candidate/control timing matrices add 78 exact output/LSE checks. Invalid
selector values fail before launch. Compute Sanitizer reports zero memcheck
errors, zero synccheck errors, and zero racecheck hazards.

The fixed-20-GB B32/P4096/D256 realtime E2E gate uses the same model and flags
as the cooperative-writer experiment. Runs are ordered ByteV2/raw,
raw/ByteV2, and ByteV2/raw:

| Round | ByteV2 seconds / tok/s | Raw FA2 seconds / tok/s |
| ---: | ---: | ---: |
| 1 | 36.842101 / 222.354 | 35.254802 / 232.366 |
| 2 | 37.145369 / 220.539 | 35.301522 / 232.058 |
| 3 | 37.157137 / 220.469 | 35.320769 / 231.932 |

The medians are 37.145369 seconds and 220.539 tok/s for ByteV2, versus
35.301522 seconds and 232.058 tok/s for raw. The remaining ByteV2 gap is
5.2231% in wall time and 4.9639% in TPS. Relative to the preceding B32 writer
checkpoint, the alias reduces ByteV2 wall time by 8.9959% and raises TPS by
9.8852%. Relative to the original generic-writer checkpoint, the combined
writer and alias changes remove 84.51% of the absolute wall gap to raw.
This percentage uses the original 35.268312-second raw reference as a fixed
denominator. If each checkpoint instead uses its contemporaneous raw median,
the removed gap is 84.79%.

All six E2E token arrays have the same SHA-256 digest and are elementwise
identical. All runs generate exactly 8,192 output tokens with zero
preemptions. Every ByteV2 run ends with `fatal=0`, no authoritative raw pages,
and all 1,472 raw-sidecar slots free.

Request-level medians show that the remaining realtime issue has moved out of
steady decode. Each table entry is the median of the three per-run request
medians, where each run contains 32 requests; the 96 requests are not pooled:

| Metric | B32 writer | K/V alias | Raw FA2 |
| --- | ---: | ---: | ---: |
| Queue | 9.290619 s | 8.888568 s | 5.915375 s |
| First-token latency | 16.142097 s | 15.403290 s | 11.889541 s |
| Decode span | 24.473965 s | 21.540014 s | 23.254867 s |
| Engine E2E | 40.615751 s | 36.943001 s | 35.151962 s |

Dividing those median decode spans by the 255 post-first-token intervals gives
a derived per-request mean ITL of 84.47 ms for ByteV2 versus 91.20 ms for raw.
ByteV2 therefore has 7.37% lower median decode-span and derived-ITL latency in
this saturated workload, equivalent to a 7.96% reciprocal speedup. Its 5.22%
E2E deficit is before the first token, including queue time; queue and TTFT are
not additive. These request metrics localize the gap to pre-first-token work,
while the preceding D1 replay identifies initial and mixed prefill routing as
the concrete source. The next realtime optimization should address that route
while preserving the exact paged-FA2 reduction path, not another Q1 reduction
kernel.

Artifacts:

- `profile/byte-v2-realtime-nonsplit-alias-a40-20260721/pair_r1.jsonl`
- `profile/byte-v2-realtime-nonsplit-alias-a40-20260721/byte_r2.jsonl`
- `profile/byte-v2-realtime-nonsplit-alias-a40-20260721/raw_r2.jsonl`
- `profile/byte-v2-realtime-nonsplit-alias-a40-20260721/pair_r3.jsonl`
- `profile/byte-v2-realtime-nonsplit-alias-a40-20260721/README.md`

### Direct Raw-Paged Initial and Mixed Prefill

The realtime pre-first-token gap is now reduced by preserving the original
FA2 attention implementation while changing only where its K/V pages come
from. For every scheduler step, the metadata builder recognizes only the
strict request layout

```text
[cached context] * N + [initial context] * M
```

and constructs one plan shared by all 32 layers. The cached prefix continues
to use the ByteV2 hybrid FA2 reader. The initial suffix views the current BF16
K/V projections directly as 16-token raw pages and calls the original paged
FA2 `varlen_fwd`. There is no K/V copy and no change to QK, masking, softmax,
PV, split reduction, or the FA2 combine implementation.

Initial requests are grouped so every internal request boundary is page
aligned. A ragged group may round its final page up into rows belonging to the
following group, but those borrowed rows are beyond that request's exact
sequence length and are masked by FA2. All group shapes, devices, dtypes,
strides, metadata extents, and rounded backing-storage bounds are checked
before either the cached or direct side launches. An interleaved request
layout, missing metadata, an unsupported projection layout, or insufficient
padded storage returns to the existing whole-batch hybrid reader.

The implementation preserves fused-QKV views rather than calling
`contiguous()`. FA2 receives K/V page strides such as
`(98,304, 6,144, 128, 1)` as well as dense
`(16,384, 1,024, 128, 1)`. K and V are validated independently because the
production projection can make K dense after rotary embedding while V keeps
the fused 6,144-element row stride.

Splitting a cached Q1 prefix out of a mixed batch would normally let FA2
rewrite causal Q1 into its noncausal GQA-swapped decode dispatch. The cached
sub-batch therefore passes `max_seqlen_q=2` as a legal dispatch upper bound
while its real `cu_seqlens_q` still describes one query. This retains causal
QK, Hq=32/Hkv=8, and `num_splits=0`, matching the original mixed-batch
topology. The same sentinel is applied to a direct group whose actual maximum
query length is one.

`seq_lens_cpu_upper_bound` supplies the plan when the deprecated exact CPU
copy is absent. It is exact for prefill rows; an optimistic async-spec decode
row remains classified as cached and never becomes a direct initial row. The
fast path is enabled by default only when hybrid raw fallback is enabled.
`BYTE_V2_FA2_DIRECT_PREFILL=0` restores the whole-batch hybrid route.
Explicitly setting it to `1` without hybrid fallback is an error.

The checked-in CUDA oracle covers a cached Q1 request followed by ragged Q13
and aligned Q16 initial requests, permuted compact pages, an authoritative raw
sidecar page, forced outliers, different K/V row strides, and the ragged
group borrowing three masked rows from the following request. Split output
and LSE are bitwise identical to one whole-batch hybrid FA2 call. CPU tests
cover the production `[Q1*4, 4092, 4096*3]` plan, pure initial batches,
interleaved rejection, missing metadata, padded and unpadded final ragged
groups, zero-copy pointers, dispatch sentinels, and the rollback selector.
Compute Sanitizer memcheck on the ragged CUDA oracle reports zero errors.

The fixed-budget E2E gate is unchanged: one A40, BF16
Llama-3.1-8B-Instruct, batch 32, a 4,096-token synthetic prompt per request,
compiled CUDA Graph execution, no speculation, no prefix caching,
`max_model_len=max_num_batched_tokens=16,384`, and an exact
20,000,000,000-byte KV budget. D1 and D8 are single paired structural gates;
D256 uses three runs:

| Decode | ByteV2 seconds / tok/s | Raw FA2 seconds / tok/s | Wall / TPS delta |
| ---: | ---: | ---: | ---: |
| 1 | 18.889139 / 1.694 | 18.728313 / 1.709 | +0.8587% / -0.8514% |
| 8 | 19.954998 / 12.829 | 19.378501 / 13.211 | +2.9749% / -2.8890% |
| 256, round 1 | 35.758325 / 229.094 | 35.471358 / 230.947 | +0.8090% / -0.8025% |
| 256, round 2 | 35.969210 / 227.750 | 35.240496 / 232.460 | +2.0678% / -2.0259% |
| 256, round 3 | 35.882546 / 228.300 | 35.200043 / 232.727 | +1.9389% / -1.9020% |

Using the same independent per-backend three-run median convention as the
preceding checkpoint, D256 is 35.882546 seconds and 228.300 tok/s for ByteV2,
versus 35.240496 seconds and 232.460 tok/s for raw. The remaining gap is
1.8219% in wall time and 1.7893% in TPS. Relative to the preceding ByteV2
median, wall time improves by 3.3997% and TPS by 3.5193%; the wall gap to raw
falls from 5.2231% to 1.8219%, removing 65.12% of that remaining gap. The
contemporaneous raw median moves by only 0.17%.

The instrumented replay is a separate run from the clean measurement and its
scopes are nested, so its CUDA totals must not be summed as a wall-time
breakdown. It nevertheless provides an unambiguous route witness:

| Decode | Direct raw-paged calls | Cached hybrid calls |
| ---: | ---: | ---: |
| 1 | 256 | 0 |
| 8 | 480 | 256 |
| 256 | 480 | 256 |

The D8/D256 counts match one pure-initial step, seven mixed steps with two
direct groups and one cached group, one cached tail, and 32 layers. With the
selector unset, D8 independently reproduces 480/256 calls and the raw token
hash, proving that the validated route is the default rather than an
explicit-selector-only result.

All paired D1, D8, and D256 token lists are elementwise equal to raw. The
compact JSON SHA-256 values are respectively
`d41b66ef7a1b5f0a92306a18b9589adf90082343bcb3cc1ab3dc5eecef467caa`,
`17fa1308b01be4604311dc0093e6cfdcd77897edb91ec82502588056d6622dcb`,
and `e37702d6c177d8be3406057a2aec2a06bcbd88a66a332134faee92ffada126f5`.
Every ByteV2 run has zero preemptions, `fatal=0`, 32 initialized layers, no
remaining authoritative raw pages, and all 1,472 sidecar slots returned.

D256 request metrics are the median of the three per-run request medians:

| Metric | ByteV2 | Raw FA2 | Difference |
| --- | ---: | ---: | ---: |
| Queue | 8.362678 s | 5.901312 s | +2.461367 s |
| First-token latency | 14.496855 s | 11.864798 s | +2.632056 s |
| Decode span | 21.183672 s | 23.227210 s | -2.043538 s |
| Engine E2E | 35.680226 s | 35.091697 s | +0.588528 s |

The realtime decode span is 8.80% lower than raw, while queue and TTFT remain
higher. In the profiled prefill replay, ByteV2 direct plus cached attention is
about 145 ms above the corresponding raw attention scopes, and the ByteV2
multi-token writer is about 445 ms above raw reshape/cache. These totals are
not additive due to nesting, but their scale agrees with the remaining
approximately 0.59-second request-E2E difference. The next realtime target is
therefore the prefill writer and per-layer Python/metadata launch overhead,
not the steady Q1 attention kernel.

The memory result is unchanged. Under the same 20-GB budget, ByteV2 exposes
190,928 tokens versus 152,576 for raw, a 25.1363% capacity increase and a
20.0825% reduction in complete bytes per capacity token, including sidecar
and staging workspace. The current checkpoint therefore supports the narrow
claim of approximately 20% complete KV-memory reduction with less than a 2%
median TPS loss on this one realtime workload; broader models, prompt mixes,
sampling modes, and task-accuracy evaluation remain required for a paper
claim.

Artifacts:

- `profile/byte-v2-realtime-direct-prefill-a40-20260721/README.md`
- `profile/byte-v2-realtime-direct-prefill-a40-20260721/profile_d1_v2.jsonl`
- `profile/byte-v2-realtime-direct-prefill-a40-20260721/profile_d8_v1.jsonl`
- `profile/byte-v2-realtime-direct-prefill-a40-20260721/profile_d256_v1.jsonl`
- `profile/byte-v2-realtime-direct-prefill-a40-20260721/e2e_d256_r{2,3}.jsonl`
- `profile/byte-v2-realtime-direct-prefill-a40-20260721/default_unset_d8.jsonl`

The final ByteV2 layout suite reports 331 passed and one skipped test. The
profile-script suite reports 21 passed; Ruff and Mypy 3.12 pass on the changed
Python sources.

### Current-Stage Attribution, Host Sync, Writer Waves, and Page Reset

The 2026-07-22 B32/P4096/D8 attribution run uses the same one-A40, compiled,
no-speculation, no-prefix-cache, fixed-20-GB configuration. Its clean baseline
is 20.076101 seconds and 12.751 tok/s for ByteV2 versus 19.759393 seconds and
12.956 tok/s for raw, a +1.6028% latency / -1.5775% TPS gap with exact tokens.
The trace separates the ByteV2-specific work as follows:

| Category | ByteV2 | Raw | Active-time delta |
| --- | ---: | ---: | ---: |
| Attention main | 2508.352 ms | 2363.151 ms | +145.201 ms |
| KV writer | 500.114 ms | 71.154 ms | +428.960 ms |
| Whole compact-page zero | 162.023 ms | 0 | +162.023 ms |
| Raw-sidecar reset | 1.045 ms | 0 | +1.045 ms |
| Scheduler metadata | 0.281 ms | 0.267 ms | +0.013 ms |
| FA2 combine | 0.889 ms | 0.921 ms | -0.032 ms |

The host trace also contains 576 pageable device-to-host copies of sequence
lengths, 44,032 bytes total, and 591 stream synchronizations. Their correlated
host API duration is approximately 18.93 seconds because each small copy
drains queued GPU work. This time is a wait-attribution signal, not additive
work.

The first optimization keeps `seq_lens_cpu_upper_bound` distinct from exact
`seq_lens_cpu`, uses it only for fail-closed prefill routing, and derives group
query offsets from the existing GPU `query_start_loc`. Pageable sequence-length
D2H falls from 576 calls to zero, explicit stream synchronization from 591 to
zero, and pageable H2D from 47 to 32 calls, removing the 15 direct-plan copies.
The nine direct-plan builds fall from 84.812 to 4.121 ms, while 576 cached-route
checks fall from 19,011.896 to 9.113 ms. A six-run same-binary comparison gives
20.360816 seconds for the emulated legacy path and 20.305072 seconds for the
new path at the median. The approximately 0.27% difference is within adjacent
pair noise, so the conclusion is mechanism cleanup and enqueue-ahead behavior,
not a material E2E speedup.

The second optimization targets actual writer work. A request segment formerly
assumed an unknown first row and could use only `16 * 128 - 15 = 2033` tokens
per staging wave. For exact sequence length `S`, CPU upper bound `U`, and query
length `q`, the metadata contract gives `U >= S >= q`; therefore `U == q`
strictly proves zero context even in async speculative mode. Such a request is
page aligned and Q4096 can use `2048 + 2048` rather than
`2033 + 2033 + 30`. Cached and untrusted rows keep the conservative formula.
Writer launches fall from 11,456 to 7,872 and writer active time from 498.024
to 463.264 ms. The 864 traced three-page prepare/hydrate/commit/persist tail
chains disappear, removing 3,584 launches and 34.760 ms of measured GPU work.
The isolated clean pair remains noisy at a +2.737% ByteV2/raw gap, so only the
kernel-work reduction, not a separately resolved E2E improvement, is claimed.

The third optimization fixes the page-lifecycle zeroing scope. Scheduler block
reporting and raw-sidecar reset remain unchanged, but a reclaimed ByteV2 compact
page now clears only its aligned 896-byte metadata prefix rather than all
52,096 bytes. The Triton kernel carries separate logical-page stride and reset
length constants; this distinction is required so block `N` remains addressed
at `N * 52,096` while only 896 bytes are written. Ordinary FullAttention pages
still receive a full reset. A poison-page CUDA test verifies zero metadata,
bitwise-preserved compact payload, untouched neighboring pages, and unchanged
ordinary FullAttention behavior.

Nsight Systems reports 17 page-reset launches in both modes. Their active time
falls from 161.538 to 2.910 ms, a 158.629-ms or 98.20% reduction. Raw-sidecar
reset remains 544 launches / 1.044 ms, and the aligned writer remains 7,872
launches / 462.355 ms. A same-binary ABBA clean comparison gives:

| Reset mode | Runs | Median |
| --- | --- | ---: |
| Full 52,096-byte page | 20.288808 s, 20.305449 s | 20.297128 s |
| 896-byte metadata prefix | 20.169632 s, 20.203821 s | 20.186727 s |

Metadata-only reset therefore saves 110.402 ms E2E, reducing latency by
0.5439% and increasing TPS by 0.5469%. All A/B, clean paired, and Nsys replay
tokens are elementwise equal to raw. The clean paired structural gate is
19.944278 seconds for ByteV2 versus 19.717589 seconds for raw, a +1.1497%
latency gap; two additional ByteV2 runs are 20.165657 and 20.163319 seconds,
so a new multi-run ByteV2/raw headline is intentionally not inferred from one
raw sample.

The complete ByteV2 layout suite now reports 334 passed and one skipped test.
Focused staging, page-reset, profile-script, runner-reset, and cache-config
tests report 71 passed; worker utility tests report three passed. Ruff,
Mypy 3.12, and `git diff --check` pass. The remaining trace-visible targets
are the approximately +391-ms writer and +137-ms attention-main active-time
deltas; the FA2 combine path remains unchanged.

Artifacts:

- `profile/byte-v2-current-stage-attribution-a40-20260722/REPORT.md`
- `profile/byte-v2-host-sync-elision-a40-20260722/REPORT.md`
- `profile/byte-v2-initial-prefill-aligned-waves-v2-a40-20260722/REPORT.md`
- `profile/byte-v2-metadata-only-page-reset-a40-20260722/REPORT.md`

### Full-Wave Hybrid Writer Vectorization and Parallel Outlier Commit

The next 2026-07-22 iteration targets the largest remaining stage-attribution
delta: the ByteV2 writer. In the metadata-only-reset trace it accounts for
7,872 launches and 462.355 ms, versus 512 launches and 71.154 ms for raw FA2.
Hydrate plus commit contributes 444.721 ms, or 96.19% of the ByteV2 total.

For a strictly verified, aligned 16-row page with unit K/V inner stride and
16-byte-compatible token/head strides, the multi-token hydrate kernel now
copies token-major K/V into the existing raw-staging layout with aligned
16-byte operations. It retains the complete global slot scan,
duplicate detection, allocator claims, metadata clear, persist, release, and
fallback lifecycle. Partial or cached-page hydration keeps the existing
element-wise source selection and decode, and Q<=32 keeps its small hydrate
kernel. Aligned staging and commit may still use 16-bit accesses on those
paths; non-unit-inner-stride or unaligned K/V selects the complete control
specialization. The commit specialization emits outlier entries
cooperatively with a deterministic CTA prefix rank, preserving the existing
row-major payload.

The isolated 2,048-token/128-page writer median falls from 161.165 to
97.043 us, a 39.79% reduction. Nsight Compute attributes this to:

| Kernel | Control | Candidate | Change |
| --- | ---: | ---: | ---: |
| Hydrate | 112.608 us | 44.000 us | -60.93% |
| Hydrate global-load instructions | 141,440 | 26,752 | -81.09% |
| Hydrate global-store instructions | 266,112 | 20,352 | -92.35% |
| Commit | 84.672 us | 63.264 us | -25.28% |
| Commit global-load instructions | 627,242 | 397,218 | -36.67% |
| Commit barrier-stall samples | 1,656 | 949 | -42.69% |

In the production B32/P4096/D8 profile window, multi-token hydrate falls from
275.831 to 155.589 ms and commit from 168.890 to 128.626 ms. Total writer
active time is 301.924 ms, down 160.431 ms or 34.70%, with the same 7,872
launches. This removes 41.01% of the historical ByteV2-versus-raw writer gap.

The same-binary GPU 0 ABBA E2E medians improve from 20.183409 to 20.138438
seconds (-0.223%), below the 0.5% noise gate. Three clean repetitions on GPU 3
give 20.071428 seconds / 12.754 tok/s for control and 19.863649 seconds /
12.888 tok/s for the candidate, a -1.035% latency / +1.046% TPS change. A
candidate/raw pair is 19.594585 versus 19.684155 seconds, but separate raw
samples drift by 67.089 ms; this is therefore an observed positive ByteV2
direction, not a general claim that ByteV2 is faster than raw.

All six GPU 3 ByteV2 outputs share the same 32-by-8 token array, and both raw
pairs are elementwise exact. The complete layout suite reports 334 passed and
one skipped test. Compute Sanitizer reports zero memcheck errors, synccheck
errors, and racecheck hazards. The validated specialization is default-on;
`BYTE_V2_HYBRID_FULL_WAVE_WRITER_CANDIDATE=0` is the rollback and same-binary
control switch, while invalid values fail closed.

The writer remains 301.924 ms versus 71.154 ms for raw. The next profile-led
target is the cached-tail generic hydrate: `grid.x=3,4,5,7,9` consumes
61.592 ms in 160 launches because it must reconstruct pre-existing compact
rows. A cooperative page decode followed by appended-row overlay is the next
candidate; commit histogram/global-load latency follows after that.

Artifacts:

- `profile/byte-v2-prefill-writer-full-wave-a40-20260722/REPORT.md`
- `profile/byte-v2-prefill-writer-full-wave-a40-20260722/analysis/summary.json`
- `profile/byte-v2-prefill-writer-full-wave-a40-20260722/reports/`

### Cached-Tail Cooperative Hydrate

The next 2026-07-22 iteration expands the existing cooperative small-writer
route to the residual production cached-tail waves. The preceding generic
hydrate assigns one CTA to each 16-token input chunk. A wave can contain many
different partially populated pages, so its few CTAs process pages serially
even though the page reconstructions are independent.

The cooperative route first runs a fail-closed prepare kernel. It validates
every
slot, duplicate row, page index, staging capacity, and persistent-raw mapping
before changing allocator state, then builds a deterministic unique-page list
and row mask. Hydration launches eight 256-thread CTA shards for every active
staging slot. The shards reconstruct disjoint portions of one compact or
authoritative-raw page and overlay the new BF16 rows. A per-slot completion
count lets only the final shard clear compact metadata and publish the staging
mapping. Commit, persistent-raw fallback, release, the full-wave writer, and
all attention kernels are unchanged.

The specialization remains bounded. Both the input-token count and active
staging capacity must be at most 144. Retained transient staging above the
previous 32-token boundary continues through the generic route. For ordinary
persist-and-release writer calls, `BYTE_V2_HYBRID_COOPERATIVE_WRITER_LIMIT`
selects the process-wide boundary:

- unset or `144`: validated candidate and new default;
- `32`: same-binary rollback to the preceding dispatch;
- `16`: legacy boundary experiment.

Any other value fails before launch. Both timing arms keep
`BYTE_V2_HYBRID_FULL_WAVE_WRITER_CANDIDATE=1`, so this A/B changes only the
cached-tail dispatch.

The final default-on extension installed for the closing smoke test has
SHA-256
`7d98f48b6074c38e4188241b160592d4524094f290331d9e689851377c3f3a91`.
Recording the binary digest prevents the final selector result from being
confused with an earlier in-place build.

The formal micro, NCU, Nsys, and E2E artifacts predate this final default-only
rebuild, and their collection-binary hash was not archived. Those A/B runs set
the selector explicitly to `32` or `144`; the rebuild changed only the unset
host default, not either selected route or its device kernel. Accordingly, the
final hash identifies the closing smoke, focused tests, and sanitizer runs,
while source equivalence links those checks to the earlier formal performance
evidence.

#### Production Shapes

A diagnostic-only shape probe recorded the real B32/P4096/D8 writer inputs.
It copied slot mappings to the CPU only in the separate replay and was not
used for E2E timing. The five residual waves occur once per layer, or 32 times
each in the profile window:

| Shape | Unique pages / capacity | Rows contributed per page | Control hydrate grid | Candidate hydrate grid |
| --- | ---: | --- | ---: | ---: |
| `n37.capacity18` | 17 / 18 | 15x1, 1x6, 1x16 | 3 | `(8,18,1)` = 144 CTAs |
| `n56.capacity23` | 22 / 23 | 19x1, 1x5, 2x16 | 4 | `(8,23,1)` = 184 CTAs |
| `n79.capacity28` | 27 / 28 | 23x1, 1x8, 3x16 | 5 | `(8,28,1)` = 224 CTAs |
| `n106.capacity33` | 32 / 33 | 27x1, 1x15, 4x16 | 7 | `(8,33,1)` = 264 CTAs |
| `n133.capacity35` | 34 / 35 | 27x1, 1x10, 6x16 | 9 | `(8,35,1)` = 280 CTAs |

The shapes are not synthetic one-token-per-page cases. Each combines many
cached one-row request tails with a partial prompt page and zero or more full
pages. The spare staging slot is intentionally harmless: its CTA shards see
no prepared physical page and return.

#### Isolated Writer Results

The shape probe preserves aggregate row/page histograms rather than the full
slot array. The harness therefore rebuilds mappings consistent with those
histograms in production dispatch order, seeds the existing compact prefix,
and times the complete fused operation, including prepare/hydrate, commit,
persist, and release. Each number below is the mean of two independent
21-trial medians; each trial contains 500 CUDA-event iterations after 20
warmups.

| Shape | 32-token control | 144-token candidate | Change | Speedup |
| --- | ---: | ---: | ---: | ---: |
| `n37.capacity18` | 156.808 us | 55.432 us | -64.65% | 2.83x |
| `n56.capacity23` | 209.867 us | 80.935 us | -61.44% | 2.59x |
| `n79.capacity28` | 266.398 us | 117.538 us | -55.88% | 2.27x |
| `n106.capacity33` | 319.762 us | 164.097 us | -48.68% | 1.95x |
| `n133.capacity35` | 317.756 us | 206.602 us | -34.98% | 1.54x |

Every run is canonically BF16-identical to the independent generic
prepare/hydrate/append/commit reference. The observed production shapes use
compact source pages and allocate no authoritative raw-sidecar page.

After the final rebuild, an additional `n106.capacity33` smoke test left the
selector unset and measured 164.710407 us, versus 320.624657 us with the
explicit `32` rollback, a 48.628% reduction. Both arms produced the same
canonical BF16 SHA-256,
`6f0052ec5665aa46499cb086b718b4bca286c100f6810f48984f0473bb4bf4f5`.
This five-trial, 100-iteration smoke is not substituted for the full micro
table; it verifies that the final binary really resolves unset to the expanded
144-token non-retained path.

#### Nsight Compute Attribution

On the representative `n106.capacity33` shape, the full NCU replay changes
the generic seven-CTA hydrate into one serialized prepare plus a 264-CTA
hydrate:

| Kernel | Grid / block | Waves per SM | Duration | Elapsed SM throughput |
| --- | --- | ---: | ---: | ---: |
| Generic hydrate | `7x1x1 / 256` | 0.0139 | 535.168 us | 0.589% |
| Candidate prepare | `1x1x1 / 1` | 0.0007 | 191.040 us | 0.035% |
| Candidate hydrate | `8x33x1 / 256` | 0.5238 | 21.888 us | 15.406% |

Prepare plus hydrate is 212.928 us, 60.21% below the replayed control. The
hydrate body alone is 95.91% lower. This is a parallelism result rather than a
memory-work elimination result: control DRAM reads/writes are 1.234/0.482 MB,
while candidate prepare plus hydrate reads/writes 1.150/0.476 MB. The candidate
hydrate also has higher active long-scoreboard and barrier exposure, but
spreading independent page work across the A40 converts that exposure into a
much shorter elapsed kernel.

#### Production Nsight Systems Attribution

The production comparison reconstructs the profile window by reverse-pairing
the last 2,304 persist launches, excluding warmup ambiguity. Across the five
target shapes, the control generic hydrate consumes 60.201464 ms. The
candidate hydrate consumes 2.819711 ms, but that number is not the complete
replacement: routing these shapes also adds 20.184436 ms of serialized
prepare. The valid comparison is therefore 60.201464 versus 23.004147 ms, a
37.197317-ms or 61.79% reduction. Prepare is already 87.74% of the candidate
target path.

| Production writer component | Control | Candidate | Change |
| --- | ---: | ---: | ---: |
| Five cached-tail hydrate paths, including incremental prepare | 60.201464 ms | 23.004147 ms | -61.79% |
| Commit | 122.887201 ms | 122.611493 ms | -0.22% |
| Persist/release | 7.895231 ms | 7.893451 ms | unchanged |
| Complete writer | 277.973809 ms | 241.675473 ms | -36.298336 ms / -13.06% |

An independent CUDA-event profile of all 2,304 multi-token writer operations
measures 290.263200 versus 254.075904 ms, a 36.187296-ms or 12.47% reduction.
The agreement in absolute saving confirms that the Nsys window reconstruction
captures the production effect. Commit and persist do not materially move;
the reduction comes from the intended cached-tail route.

#### E2E, Raw FA2, and Tokens

The clean gate keeps the same one-A40, compiled B32/P4096/D8 workload, exact
20,000,000,000-byte KV budget, and 256 total output tokens. Two ByteV2 arms
and the paired raw references measure:

| Run | 32-token control | 144-token candidate | Candidate minus control |
| ---: | ---: | ---: | ---: |
| 1 | 19.540829 s / 13.100775 tok/s | 19.285086 s / 13.274507 tok/s | -255.743 ms |
| 2 | 19.418547 s / 13.183272 tok/s | 19.438668 s / 13.169627 tok/s | +20.121 ms |

The two-run means are 19.479688 seconds for control and 19.361877 seconds for
the candidate, nominally -0.605% latency and +0.608% TPS. The second comparison
reverses direction, and the within-arm spans are 122.3 and 153.6 ms. This is
therefore not a resolved E2E speedup claim; the production kernel reduction is
the retained evidence.

The candidate's paired raw reference is 19.199749 seconds / 13.333508 tok/s,
leaving only +0.444% wall time / -0.443% TPS in that sample. The other raw
reference is 19.187286 seconds / 13.342168 tok/s. All six output arrays are
elementwise identical, all runs emit 256 tokens, and every run has zero
preemptions. The optimization changes writer scheduling, not model numerics.

#### Validation and Next Step

The complete ByteV2 layout suite reports 337 passed and one skipped test. New
coverage checks unset/16/32/144 selector behavior, invalid-value failure,
canonical production `n106.capacity33` output, the retained-staging boundary,
CUDA Graph workspace reuse, cross-boundary duplicate rows, and staging-capacity
overflow. Compute Sanitizer reports zero memcheck errors, zero synccheck
errors, and zero racecheck hazards, errors, or warnings.

The final focused selector, production-n106, retained-staging, and maximum
boundary run reports eight passed tests. Its explicit edge cases include
retained-transient N32/N33 and persist-and-release token/capacity pairs
`144/144`, `144/145`, and `145/145`. The complete-suite result remains the
337-passed, one-skipped result above.

The cooperative hydrate is adopted and default-on because it removes 61.79%
of the measured target path with exact output and a strict rollback. Its new
bottleneck is explicit: the one-thread prepare consumes 20.184436 of the
candidate path's 23.004147 ms. The next experiment should parallelize or
hierarchically compact the bounded 144-token page/row descriptors while
preserving deterministic staging-slot order and the current validate-before-
mutation contract. Only after prepare falls should the broader writer return
to commit histogram and global-load latency, which now dominate total writer
time.

Artifacts:

- `profile/byte-v2-cached-tail-cooperative-hydrate-a40-20260722/harness/`
- `profile/byte-v2-cached-tail-cooperative-hydrate-a40-20260722/reports/`
- `profile/byte-v2-cached-tail-cooperative-hydrate-a40-20260722/analysis/`

### B1 Cached-Prefill Full-Prefix Hydrate-to-Raw FA2 (2026-07-24)

#### Motivation and Result

The direct raw-paged initial-prefill route above already preserves the
original FA2 path for the first scheduler chunk. The remaining long-prefill
regression came from later cached chunks: the hybrid reader reconstructed the
same compressed KV tiles inside many Q-tile consumers. At Q16K/S32K, the
balanced hybrid shape measured 115.083 ms versus 73.558 ms for raw, a 56.45%
regression despite reading 48.19% fewer DRAM bytes. In the corresponding
production sweep, the accumulated TTFT gap reached 19.90% at 32K and 33.36%
at 65K.

The new default-off experiment moves reconstruction out of the attention
reader. For each layer and cached-prefill chunk, it materializes the complete
KV prefix once into the existing BF16 raw-staging workspace, constructs a
dense local page table, and invokes the original raw FA2 `varlen_fwd`.
QK, causal masking, softmax, PV, split selection, and combine remain
unchanged. The operation count therefore changes from repeated decode work
near O(QK) to one O(K) hydrate before the unchanged O(QK) attention.

On A40, 200 balanced iterations at Q16K/S32K measured:

| Distribution | Raw FA2 | Hydrate + raw FA2 | Production mirror |
| --- | ---: | ---: | ---: |
| Compact-safe | 78.6212 ms | 79.0738 ms (+0.576%) | 79.0170 ms (+0.503%) |
| Forced in-page outlier | 78.6017 ms | 79.0047 ms (+0.513%) | 79.0738 ms (+0.601%) |

The production mirror includes descriptor initialization and release. Both
candidate outputs and FP32 LSE tensors are bitwise identical to raw FA2 in
both distributions. The sub-0.1% inversion between the safe core and mirror
medians is timing noise, not negative bookkeeping cost.

Five clean, alternating-order E2E pairs give:

| Context | Median paired TTFT gap | MAD | Range | Candidate wins |
| ---: | ---: | ---: | ---: | ---: |
| 32,768 | +0.057% | 1.070% | -1.624% to +2.085% | 2/5 |
| 65,536 | -0.226% | 0.140% | -0.365% to +0.374% | 3/5 |

Within this measured A40/B1/16K-chunk/32K--65K boundary, the correct
interpretation is raw-equivalent performance within observed run-to-run
variability and removal of the old structural regression, not a stable
ByteV2 speedup. No predefined statistical equivalence margin/test was run.
At 32K, the candidate-first subgroup median is -0.111% while the raw-first
subgroup is +1.606%, directly exposing inter-process/order drift. The 65K
range also crosses zero.

#### Code Route and Lifecycle

`BYTE_V2_FA2_CACHED_PREFILL_HYDRATE_TO_RAW` controls the route. It accepts
only unset, `0`, or `1` and defaults to off. Explicit enablement requires
`BYTE_V2_FA2_HYBRID_RAW_FALLBACK=1`; a contradictory configuration fails at
initialization.

`ByteV2RawStagingManager.stage_cached_prefill()` accepts the effective-B1
scheduler block-table row and target sequence length. For
`P = ceil(seq_len / 16)`, it:

1. validates that all P physical pages fit the block table, raw workspace,
   and descriptors;
2. sets exact valid rows, including a partial final page;
3. calls the existing hybrid hydrate op, which selects compact reconstruction
   or the authoritative raw-sidecar copy per physical page;
4. reuses the fixed `staging_to_physical_block[:P]` descriptor as local page
   IDs `[0, ..., P - 1]`.

`byte_v2_fa2_raw_staging_prefill_attention()` now accepts
`block_tables_are_staging_slots=True`. In this mode, it passes those local
page IDs directly to the original raw FA2 and avoids the advanced-index
temporary `block_to_staging_slot[block_tables]`.

The forward route wraps FA2 in `try/finally` and always calls
`release_cached_prefill()`. Hydrate failures also trigger release. Descriptor
initialization, hydrate, raw FA2, and release all use the current CUDA stream.
Safety depends on the existing single-lane shared-workspace invariant:
`block_to_staging_slot` stays quiescent throughout this full-prefix route.
The route must not be generalized to concurrent u-batching or another stream
without replacing that lifecycle.

No new attention CUDA kernel was added for this experiment. It reuses
`byte_v2_hydrate_raw_staging_from_hybrid_cache()` and the installed original
FA2 binary.

#### Selector and Fallback

The route requires:

- the feature selector and hybrid raw-fallback store;
- causal attention and `max_query_len > 1`;
- `max_query_len == num_actual_tokens`;
- CPU query starts of `[0, Q, Q, ...]`, so only the first effective row is
  nonempty while trailing padded zero-length rows remain legal;
- `seq_len == max_seq_len > Q`, proving a cached prefix is present;
- a CPU sequence upper bound equal to `max_seq_len`; it is exact for the
  supported prefill case, but the selector does not independently prove
  exactness in asynchronous speculative decode;
- valid CUDA int32 device sequence lengths and block table; a production
  padded table is supported as long as its last-dimension stride is one;
- enough block-table columns and staging slots for the complete prefix.

An ordinary shape, layout, or capacity miss has no side effects and returns to
the generic hybrid reader. In this experiment, with direct prefill enabled,
the first no-cache scheduler chunk continues to use the existing direct
raw-paged initial-prefill route. An active initial-prefill staging lease is an
invariant violation and fails closed rather than taking the ordinary fallback.

The selector currently infers prefill from shape rather than consuming an
explicit `is_prefilling` flag. An effective-B1 Q>1 asynchronous speculative
decode could therefore enter this route. Exact device sequence lengths still
make the raw FA2 mask numerically safe, but the hydrate may be unnecessary.
Token padding can conversely cause a conservative fallback. A production
extension should add explicit semantics and route counters.

Physical page IDs retain the existing scheduler block-table trusted-input
boundary. The hydrate kernel rejects negative IDs but does not independently
check an ID against `kv_cache.shape[0]`.

#### Production Evidence and Attribution

The formal workload uses one A40, Llama-3.1-8B-Instruct BF16, B1, a 16,384
scheduler chunk, a 131,072 model limit, 16 output tokens, compiled/CUDA-Graph
Q1 decode, no prefix caching, and an exact 20,000,000,000-byte KV budget.
Candidate runs use 4,096 staging slots.

Only uniform Q1 decode is graph-captured. The Q>1 cached-prefill route in this
experiment executes eagerly even though the full E2E run is compiled.

All 20 clean JSON records pass the protocol gates:

- candidate/raw prompt SHA and all 16 token IDs match at each context;
- instrumented replay tokens match their clean request;
- `performance_valid_for_tps=true`;
- clean and replay preemptions are zero;
- candidate `fatal=0`, `raw_page_count=0`, and
  `free_count=slot_count=1472`.

Route counts prove that the experiment did not silently retain the hybrid
reader. At 32K, 32 hydrate calls cover one cached chunk across 32 layers. At
65K, 96 calls cover three cached chunks. The raw-staging FA2 counts are 64 and
128 because they include the initial chunk as well. The existing profile name
`initial_raw` also labels the new cached calls; the analysis identifies them
by scheduler step and Q/S shape rather than that stale label.

The first instrumented pair illustrates the new residual:

| Context | Hydrate + raw-FA2 attention vs raw | Writer delta | Net of these leaves |
| ---: | ---: | ---: | ---: |
| 32K | -52.344 ms | +33.656 ms | -18.688 ms |
| 65K | -76.025 ms | +67.895 ms | -8.130 ms |

Only non-overlapping attention and writer leaves are combined; parent
`forward`, `prefill_from_cache`, shape, and op scopes are nested and must not
be summed. This one replay is mechanism evidence, not a clean speedup claim.
It shows that the attention regression is gone and the multi-token writer now
consumes most of the remaining margin.

#### Memory Tradeoff

The 4,096-slot workspace covers 65,536 tokens and occupies 268,515,340 bytes
(256.076 MiB). It is shared serially across all 32 layers, not allocated per
layer. The same plan also contains a 97,982,592-byte (93.443-MiB) hybrid raw
sidecar.

Under the fixed 20-GB budget:

| Plan | Blocks | Token capacity | Total planned bytes |
| --- | ---: | ---: | ---: |
| ByteV2 candidate | 11,777 | 188,432 | 19,999,604,876 |
| Raw FA2 | 9,536 | 152,576 | 19,998,441,472 |

The candidate provides 35,856 more theoretical allocator token slots, or
23.500% higher capacity. At the same 188,432-token capacity, raw requires
24,698,159,104 bytes, so the complete candidate KV plan reservation is
4,698,554,228 bytes, or 19.024%, lower. The ByteV2 compact KV page/tensor
footprint, including metadata and outlier storage, is 20.508% lower per block.
These are KV-planner figures, not total GPU memory or a measured allocator
peak.

A 65,536-token occupied-block working-set model needs its own denominator.
For 4,096 logical prompt blocks, compact KV plus the 256.076-MiB scratch is
17.382% lower than raw occupied blocks; charging the full 93.443-MiB sidecar
pool as well gives 16.241%. Using the observed 4,097 peak-resident blocks
changes them only to approximately 17.383% and 16.242%. These are attribution
estimates, not measured active-peak savings: vLLM has already reserved the
complete KV plan and does not release it with a smaller request working set.

The workspace is runner-owned, preallocated, and resident for the process
lifetime; only its request contents are temporary. Its capacity grows
linearly with the configured slot ceiling, making this a deliberate prefill
latency/capacity tradeoff rather than a zero-copy design. The default 128
slots cover only 2,048 tokens. The 4,096-slot experiment covers 65K; 131K
would require approximately 8,192 slots, or 512 MiB. A longer sequence
currently falls back safely to hybrid and can recover the old regression.

At 65K, observed peak KV pressure is only 34.79% for the candidate and 42.96%
for raw. Both allocator capacities exceed the 131,072 engine model limit, so
the 23.5% slot increase does not raise this experiment's single-request
maximum context. It is a concurrency-capacity proposition, while this fast
path is currently validated only at B1; memory-saturated multi-request TPS
has not been demonstrated.

#### Validation, Scope, and Next Step

Focused CPU/static tests report 24 passed. CUDA bitwise source/path tests
report three passed and cover compact-safe, compact-outlier, authoritative
raw-sidecar pages, page permutation, and a partial final page. The latter
three boundaries have correctness coverage but not isolated/E2E performance
coverage. The complete attention-layout plus staging-workspace suite reports
370 passed and one skipped test. Related Python files also pass `py_compile`,
Ruff, and `git diff --check`.

The validated production-performance boundary is one A40, TP1/PP1,
Llama-3.1-8B, BF16, 32 Q heads, 8 KV heads, head size 128, causal B1,
speculation disabled, temperature zero, synthetic-repeat prompts, prefix
caching disabled, 16K chunked cached-prefill, and 16 generated tokens. It
does not yet cover ragged B>1, dynamic batching, DBO/u-batching, multi-stream
workspace users, KV connectors/offload, DCP, another GPU, or another head
shape. It inherits the complete Current Support Boundary above, including its
ALiBi, sliding/local attention, softcap, non-decoder, cache-sharing, sleep, and
parallel-context exclusions. Q1 decode intentionally continues through the
existing ByteV2 decode route and must not perform full-prefix hydrate per
token.

The next prefill optimization should target the multi-token writer, which now
dominates the controllable residual. After that, the route needs adaptive
workspace planning, explicit prefill/route counters, and a batched local-page
namespace with a concurrency-safe lifecycle before B>1 testing.

Artifacts:

- `profile/byte-v2-cached-prefill-hydrate-to-raw-a40-20260724/REPORT.md`
- `profile/byte-v2-cached-prefill-hydrate-to-raw-a40-20260724/harness/`
- `profile/byte-v2-cached-prefill-hydrate-to-raw-a40-20260724/reports/`

### A40 Long-Q1 Shared Split-20 Selector (2026-07-25)

#### Motivation and Isolated Evidence

The final long-decode study used one request, a 65,536-token context, BF16
Llama-3.1-8B, 32 query heads, eight KV heads, head size 128, 16-token pages,
compiled Q1 decode, no speculation or prefix caching, and an exact
20,000,000,000-byte KV budget. Before changing split selection, clean ABBA
runs on three physical A40s showed ByteV2 TPOT 1.57% to 3.74% below raw for
both 256- and 512-token generations.

An Nsight Systems replay over 255 decode steps attributed the per-token
difference as follows:

| Leaf/category | Raw | ByteV2 | ByteV2 minus raw |
| --- | ---: | ---: | ---: |
| Main attention | 13.0717 ms | 11.9993 ms | -1.0723 ms |
| Original FA2 combine | 0.3453 ms | 0.3167 ms | -0.0286 ms |
| KV writer | 0.0833 ms | 0.4572 ms | +0.3739 ms |
| Common model/runtime | 22.8873 ms | 22.9267 ms | +0.0394 ms |
| GPU envelope | 37.4172 ms | 36.9033 ms | -0.5138 ms |

The stock FA2 heuristic selected 18 splits at 65K. With eight effective heads
this launches 144 CTAs, while the measured ByteV2 specialization can keep two
CTAs resident on each of the A40's 84 SMs. An explicit split sweep therefore
tested 16, 18, 20, 21, and 24 with the complete main-plus-original-combine
call, 12 warmups, 200 CUDA-event iterations, and three independent
repetitions.

At 65K, split 20 reduced ByteV2 attention by 3.57% for compact-safe pages and
8.50% for the forced-outlier stress distribution relative to split 18.
Split 21 was faster for the outlier stress case but slower for safe data;
split 24 regressed sharply from tail-wave and combine overhead. Split 20 was
therefore selected as the distribution-independent compromise.

The multi-length sweep produced:

| Sequence length | Safe split-20 delta | Forced-outlier delta |
| ---: | ---: | ---: |
| 4,099 | -1.59% | -3.13% |
| 16,384 | +0.93% | 0.00% |
| 32,768 | -5.67% | -10.00% |
| 65,536 | -3.57% | -8.50% |
| 131,072 | -2.18% | -5.38% |
| 188,416 | -2.13% | -6.27% |

The production interval is conservatively limited to inclusive
[32,768, 131,072]. At 16K there is no stable gain. At 188K raw FA2 regresses
0.705%, the length exceeds the current model/E2E validation limit, and fixed
20-GB raw capacity cannot support an equal-length production comparison.

#### Shared Selector and Bitwise Contract

The change is in vendored FA2 `set_params_splitkv()`, represented by
`cmake/patches/vllm_flash_attn_byte_v2.patch`. An automatic call selects 20
only when all of the following hold:

- the cached device name is exactly `NVIDIA A40` and it reports 84 SMs;
- dtype is BF16 and softcap is disabled;
- attention is paged with a 16-token page;
- the FA2 Q1/GQA transpose is active;
- the post-transpose shape is B1, eight effective heads, eight KV heads,
  `max_seqlen_q=4`, and head size 128;
- `32768 <= max_seqlen_k <= 131072`.

Every miss uses the original FA2 heuristic. Any explicit positive split count
bypasses the new selector.

The selector is deliberately shared by raw and ByteV2 rather than placed in a
ByteV2-only Python wrapper. Both calls continue through the same split tree
and the original `flash_fwd_splitkv_combine_kernel`; QK, masking, softmax, PV,
and combine code are unchanged. Same-split ByteV2/raw BF16 output and FP32 LSE
were bitwise identical at every tested length and distribution. Different
split trees are not required to be bitwise identical, which is why changing
only ByteV2 would violate the raw-bitwise contract.

Boundary tests compare auto, explicit 18, and explicit 20 at 32,767, 32,768,
65,536 safe/outlier, 131,072, and 131,073. Auto matches explicit 20 only
inside the intended interval and explicit 18 remains distinct. Additional
GQA2 and softcap near-miss tests match the stock explicit-18 result.
The final complete attention-layout test file reports 371 passed and one
skipped test on the validated A40 build.

#### Diagnostic and Final Production E2E

Before installing the selector, a palindromic diagnostic run compared ByteV2
auto/split-20 and raw auto/split-20 on three A40s. Split 20 improved ByteV2
TPOT by 1.22%, 1.23%, and 2.33%, while raw changed by -0.17%, -0.16%, and
+0.03%. All 24 diagnostic requests had identical prompts/tokens and passed
preemption, fallback, fatal, and staging-resource gates.

After the exact-device and softcap gates were installed, a fresh production
raw/Byte/Byte/raw ABBA run gave:

| Physical A40 | Raw TPOT | ByteV2 TPOT | ByteV2 vs raw |
| ---: | ---: | ---: | ---: |
| 0 | 40.1751 ms | 38.2970 ms | -4.675% |
| 1 | 40.2138 ms | 38.3139 ms | -4.724% |
| 4 | 36.2226 ms | 34.8600 ms | -3.762% |

All six raw/Byte pairs matched all 256 output tokens. Every run was
performance-valid with zero preemptions; ByteV2 reported zero fatal events,
zero authoritative raw pages, and all 1,472 raw-fallback sidecar slots free. Trace
records show auto requests, zero diagnostic forced calls, and observed Q1
calls. A separate production run before the final scope gates gave
-4.615%, -4.480%, and -3.899%, confirming the result is repeatable and that
the cached device-property check has no visible TPOT cost.

The fixed-budget allocator capacities remain 188,432 tokens for ByteV2 and
152,576 for raw, a 23.50% increase. Within the validated 65K B1 shape this is
therefore a simultaneous capacity and latency result, not merely a
capacity-for-latency trade.

#### Scope and Next Step

This is a narrow A40 shape specialization, not a generic FA2 heuristic. It
does not claim results for another GPU, B>1, another GQA ratio, page size,
dtype, head dimension, softcap, local/sliding attention, or sequence length
outside the gated interval. Forced outliers are a route stress test rather
than a production distribution estimate.

The largest remaining ByteV2-specific positive decode cost is the writer:
approximately +0.3739 ms/token in the pre-selector Nsight Systems replay. The
next optimization should reduce append/commit/persist launches and metadata
work while preserving this shared split selector, original FA2 combine, and
same-shape bitwise gate. It must be retained only after another clean
multi-GPU ABBA E2E comparison.

Artifacts:

- `profile/bytev2-final-long-decode-e2e-a40-20260725/REPORT.md`
- `profile/bytev2-final-long-decode-e2e-a40-20260725/analysis/`
- `profile/bytev2-final-long-decode-e2e-a40-20260725/reports/`
- `profile/bytev2-fa2-split-sweep-a40-20260725/analysis/`
- `profile/bytev2-fa2-split-sweep-a40-20260725/reports/`

### A40 Q1 Raw-Tail Dynamic Demotion (2026-07-26)

#### Motivation and Lifecycle

The fused Q1 raw-tail path removed the largest remaining ByteV2-specific
decode-writer cost, but its initial lifecycle could leave a page authoritative
in the persistent BF16 sidecar after dynamic scheduling returned from Q1 to
Q>1. That was correct but could steadily consume raw slots and erode the
capacity benefit.

The non-retained Q>1 update now runs compact commit, safe raw-page demotion,
and raw fallback persistence on the same CUDA stream. The new one-CTA,
128-thread demotion kernel scans touched staging descriptors and reclaims a
mapped raw slot only when the page is full, compact outlier-pool overflow is
zero, and no compact tile requires fallback. Partial and unsafe pages retain
their raw mapping. The retained prefill update is unchanged.

This does not alter QK, masking, softmax, PV, split selection, or the original
FA2 combine kernel. It is a cache-writer lifecycle change.

#### Dynamic-Batching Correctness

A two-request witness uses a 257-token scheduler budget. Request A completes a
4,096-token prefill and executes one pure Q1 step, allocating one raw tail in
each of 32 layers. Request B then arrives with a 3,840-token prompt. Fifteen
mixed steps schedule A Q1 plus B Q256, so A's page becomes full while both
requests remain live.

Eager and compiled ByteV2 runs match eager and compiled raw FA2 in every token
and in the normalized 33-step scheduler trace. Request A produces 18 tokens
and request B produces two. In the final default-on compiled lifecycle run:

| Checkpoint | Slots | Free | Raw pages | Fatal | Live requests |
| --- | ---: | ---: | ---: | ---: | ---: |
| after pure Q1 | 1,536 | 1,504 | 32 | 0 | 1 |
| after mixed step 15 | 1,536 | 1,536 | 0 | 0 | 2 |
| after finish | 1,536 | 1,536 | 0 | 0 | 0 |

All layers have the same 48-slot state for this two-request configuration.
The captured auto plan has 11,930 compact blocks, 128 staging slots, and
`max_num_seqs=2`; therefore `11930 // 256 + 2 == 48`, with no explicit slot
override. Reclamation occurs online rather than as a side effect of request
completion or cache reset.

Focused CUDA tests cover safe and overlay demotion, partial continuation,
unsafe retention, same-wave reuse, multiple pages, CUDA Graph replay, and
dummy no-op behavior. Memcheck, racecheck, and synccheck report zero errors
for the focused witness.

#### Cost Attribution

ABBA CUDA-event measurements of the complete Q>1 update give:

| Staging capacity | Demotion off | Demotion on | Delta |
| ---: | ---: | ---: | ---: |
| 33 | 164.721 us | 166.745 us | +2.024 us |
| 128 | 95.503 us | 97.792 us | +2.289 us |

Nsight Systems isolates the new kernel at 1.565 us per layer. Prepare,
hydrate, compact commit, and persist each move by no more than about 0.03 us.
The kernel alone contributes about 0.050 ms across 32 layers; the complete
measured update delta of 2.024 us per layer corresponds to about 0.065 ms per
Q>1 engine step. This is about 1% of the measured 6--7 ms ByteV2/raw
difference in the dynamic mixed-prefill trace, so demotion is not the dominant
residual.

A fresh compiled B8 off/on/on/off process bracket has identical prompts and
tokens. Off averages 7.312474 s and on averages 7.222381 s. The -1.232%
difference is treated only as a no-regression result because it is below
whole-process variation, not as a raw-tail speedup.

#### Capacity-Safe Default

Raw-tail Q1 is now implicit only when the existing experimental hybrid mode
is enabled. It remains globally off when
`BYTE_V2_FA2_HYBRID_RAW_FALLBACK` is off, and
`BYTE_V2_HYBRID_RAW_MUTABLE_TAIL_Q1=0` is the rollback.

For ordinary requests, the planner reserves:

```text
raw slots per ByteV2 layer =
    max(1, num_blocks // 256) + max_num_seqs
```

`max_num_seqs` bounds one mutable partial tail per running request. The first
term is the pre-existing empirical reserve for full unsafe pages.
`BYTE_V2_FA2_RAW_FALLBACK_SLOTS` keeps its historical total-slot meaning and
must be at least the sum above when raw-tail Q1 is active. Override planning,
automatic block search, final multi-rank shrink, and the minimal CUDA Graph
cache all account for the resolved requirement.

The installed native ABI must contain both the Q1
`fuse_commit_finalize` schema and the Q>1 `demote_safe_raw_pages` schema. An
implicit request falls back to the legacy writer if they are missing; an
explicit raw-tail opt-in fails closed. The resolved capability is carried in
`KVCacheConfig`, keeping the planner, scheduler, CUDA Graph cache, and
attention path consistent. Worker workspace binding also compares the planned
bit with every active attention implementation and rejects a mismatch, so
heterogeneous extension installs fail closed.

Resumable or streaming sessions retain KV outside the ordinary running set,
so `max_num_seqs` cannot bound their tails. They are rejected in
`EngineCore.preprocess_add_request`, whose multiprocess input path returns a
request-scoped error. The scheduler repeats the check as defense in depth.

At the common 32-layer, `max_num_seqs=256` default, the tail term alone costs
536,903,680 bytes, or 512.031 MiB, of persistent sidecar memory. This is
charged to the KV planner rather than hidden as a temporary peak.

#### Final 65K Check

The final default-on B1 check uses compiled execution, a 65,536-token prompt,
257 generated tokens, auto split selection, and a 20,000,000,000-byte KV
budget. All output token IDs exactly match the raw FA2 bracket. Using
`decode_seconds / 256`, the fresh raw/Byte/Byte/raw ABBA gives a ByteV2 mean
TPOT of 38.131789 ms versus a raw mean of 40.199216 ms, or -5.143%.
Relative to the earlier `reports/e2e-long-default.jsonl` single run
(38.150527 ms), the fresh ByteV2 mean changes by -0.049%; relative to
`reports/e2e-long-capacity.jsonl` (38.176678 ms), it changes by -0.118%.
Both are within process variation. Both ByteV2 plans explicitly report
`raw_mutable_tail_q1=true`; all four runs are performance-valid, have zero
preemptions, and use the same prompt and 257 output tokens.

The fixed-budget plan contains 11,775 ByteV2 blocks or 188,400 token slots,
versus 9,536 raw blocks or 152,576 token slots. That is 23.479% more token
capacity. At the same 188,400-token capacity, the complete ByteV2 plan is
19.024% smaller than raw. Its final allocator has all 1,472 slots free, zero
raw pages, and no fatal state.

The B1 plan still has 46 slots per layer because the compact block count falls
from 11,777 to 11,775 and crosses the integer unsafe-reserve boundary:
`11775 // 256 + 1 == 46`. The tail reserve is active even though the final
slot count is unchanged.

The scope remains A40/SM86, one writer stream, the established Llama-3.1-8B
BF16 ByteV2 shape, and non-resumable requests. Prefix sharing does not create
additional mutable tails because partial blocks are not shared. The unsafe
reserve is still empirical rather than a guarantee for arbitrary unsafe-page
density. Wider preemption, connector/offload, PP/DCP, and multi-stream
coverage remains future work.

All workers must use the same extension ABI and initialization environment,
and selectors must remain unchanged after initialization. Unsupported
resumable requests currently use the general request-error response and
therefore surface as an HTTP 500 rather than a dedicated configuration 4xx.

The measured demotion cost is too small to be the next optimization target.
For Q>1/mixed prefill, the remaining work stays in the existing prepare and
multi-token writer path. For long Q1 decode, the raw-tail writer plus shared
split-20 selector already preserves the measured ByteV2 latency advantage.

Final regression results are 394 passed and one skipped for the complete
attention-layout file, 88 passed for the complete KV-planner file, 35 passed
for the staging-workspace plus profile-script files, and one passed for the
multiprocess request-scoped rejection. Related Python files pass Ruff format,
Ruff check, and `git diff --check`.

Artifacts:

- `profile/bytev2-q1-raw-tail-dynamic-demotion-a40-20260726/REPORT.md`
- `profile/bytev2-q1-raw-tail-dynamic-demotion-a40-20260726/analysis/`
- `profile/bytev2-q1-raw-tail-dynamic-demotion-a40-20260726/harness/`
- `profile/bytev2-q1-raw-tail-dynamic-demotion-a40-20260726/reports/`

### A40 Q>1 Bounded Parallel Prepare (2026-07-26)

The remaining cooperative-writer bottleneck for non-retained waves with at
most 144 input tokens was the single-thread
`prepare_small_multi_token_hybrid_staging` kernel. On the representative
`n106.capacity33` shape, Nsight Systems measured 137.460 us in prepare,
12.460 us in hydrate/append, 11.296 us in compact commit, 2.319 us in persist,
and 1.562 us in safe demotion. Prepare was therefore the only material target
in that route.

A new single-CTA, 256-thread kernel parallelizes clean-descriptor checks,
token page/row parsing, bounded prefix scans, duplicate-row detection, and
per-page validation. Stable first-occurrence ranks preserve the serialized
kernel's staging-slot order. Thread zero publishes the already validated
descriptors in that order, retaining the existing fail-closed behavior.
Hydrate, compact commit, persist, demotion, attention, split selection, and
the original FA2 combine kernel are unchanged.

The candidate applies only to non-retained cooperative waves with `N<=144`.
Retained staging stays on the serialized control and `N>144` stays on the
generic writer. `BYTE_V2_HYBRID_PARALLEL_PREPARE=0` restores the serialized
kernel; unset or `1` selects the validated default, and any other value fails
closed. Publication retains the established single-stream, quiescent
allocator contract.

ABBA CUDA-event measurements of the complete update give:

| Production-reconstructed shape | Serialized | Parallel | Delta |
| --- | ---: | ---: | ---: |
| `n37.capacity18` | 57.488 us | 40.507 us | -29.538% |
| `n56.capacity23` | 83.032 us | 46.978 us | -43.422% |
| `n79.capacity28` | 119.427 us | 53.875 us | -54.889% |
| `n106.capacity33` | 166.245 us | 59.348 us | -64.301% |
| `n133.capacity35` | 209.016 us | 63.958 us | -69.400% |

The dedicated `n106` bracket confirms 166.690 us versus 59.651 us. Nsight
Systems measures prepare itself at 137.460 us versus 30.080 us (-78.12%);
hydrate, commit, persist, and demote each change by less than 0.10 us. A
generic `n2048.capacity128` non-routing guardrail is unchanged at 97.761 us
versus 97.756 us.

All candidate/control cache outputs are canonical BF16 exact. Focused tests
cover selector values, invalid selector handling, retained and maximum
boundaries, CUDA Graph replay, invalid maps, nearby and far duplicate rows,
capacity overflow, dirty descriptors, and pool exhaustion. Candidate
memcheck and synccheck report zero errors; racecheck reports zero hazards,
errors, or warnings.

The complete attention-layout regression reports 400 passed and one skipped.
The focused optimization gate reports 17 passed with 384 deselected; after
pinning the fatal subprocesses explicitly to the candidate, all eight
fail-closed probes pass again.

The E2E result is deliberately narrower than the kernel result. A compiled
B32/P4096/D8 ByteV2-only ABBA bracket is 20.182256 seconds for control versus
20.253688 seconds for the candidate, a nominal +0.354% inside the 0.5%
no-regression threshold and below the observed process drift. Tokens and KV
plans are exact and preemptions are zero. The same tokens match the prior
same-prompt raw FA2 reference, but this round has no fresh raw performance
arm and therefore makes no claim about changing the raw gap.

The dynamic `Q1 + Q256` negative control is +0.112% and the B1/65K
long-context negative control is +0.009% in wall time. Both are exact and
performance-valid. They do not exercise this optimization: `N=257` and the
long prefill chunks exceed the 144-token limit. The local improvement closes
the profiled prepare bottleneck. Applying the five measured writer-wave
savings once per layer gives only 11.857 ms, or 0.0588% of the 20.182-second
B32 control E2E time, consistent with the absence of a resolvable E2E gain.

The next E2E target is the generic `N>144` mixed-wave writer. For the observed
`Q1 + Q256` structure, the preferred experiment is page-centric: keep the 16
fresh full pages on the existing aligned vector writer and cooperatively
hydrate only the cached partial tail. A simpler cooperative-limit extension
to at least 257 tokens can first serve as an upper-bound experiment, but must
not displace the optimized full-page path without data.

Artifacts:

- `profile/bytev2-qgt1-parallel-prepare-a40-20260726/REPORT.md`
- `profile/bytev2-qgt1-parallel-prepare-a40-20260726/analysis/`
- `profile/bytev2-qgt1-parallel-prepare-a40-20260726/reports/`

### A40 SplitZip Fixed-Page FA2 Reader Prototype (2026-07-26)

A reader-only experiment replaces ByteV2's dense reconstruction with
SplitZip sign/mantissa plus contiguous-16 exponent-window decode inside the
current external FA2 loader. It deliberately reuses the 52,096-byte V5 page,
cp.async staging, 896-byte shared page descriptor, QK/softmax/PV mainloop,
split reduction, and original FA2 combine kernel. Separate calibrated windows
are K 115--130 and V 110--125. The offline page packer stores even codes in
the low nibble, pooled entries as `(true_exp << 8) | local_pos`, and now masks
invalid partial-tail rows from escape-pool accounting.

On A40, batch-1 Q1 single-call median-of-three latencies for
raw/ByteV2/SplitZip are 57.344/61.440/62.464 us at 4K,
137.216/120.832/121.856 us at 16K, and
466.944/407.552/411.648 us at 65K. SplitZip is therefore 8.93% slower than raw
at 4K but 11.19% and 11.84% faster at 16K and 65K. A 20-call/event explicit
same-split confirmation gives SplitZip gains of 10.25% and 11.44% at 16K and
65K. At long lengths it remains 1%--2% slower than ByteV2 despite identical
resident bytes, isolating dense exponent reconstruction as the next format
optimization target.

All nine main repetitions and nine batched confirmations are output/LSE
bitwise exact against same-split raw FA2. Independent fixed-page decoding,
an exponent 0--254 escape stress page, and a ragged/permuted/shared-prefix
case with partial tails also pass bitwise. Real-capture SplitZip escape rate
is 0.024426%, maximum demand is 15 entries/page and two entries/tile, and no
page overflows.

This prototype proves direct realtime consumption but not a production vLLM
cache format. It has no online writer or raw overflow fallback, does not
support arbitrary Top-16 LUT pages, and uses the same fixed allocation as
ByteV2. Its capacity saving is therefore 20.508%, not the public SplitZip
variable-payload result of 23.341%. The next gate is a 65K ByteV2/SplitZip NCU
comparison followed by packed-integer reconstruction work.

Artifacts:

- `profile/bytev2-splitzip-fa2-reader-a40-20260726/reports/REPORT.md`
- `profile/bytev2-splitzip-fa2-reader-a40-20260726/reports/aggregate.json`
- `profile/bytev2-splitzip-fa2-reader-a40-20260726/plots/`
- `profile/bytev2-splitzip-fa2-reader-a40-20260726/raw/`

### A40 SplitZip FA2 Reconstruction NCU Optimization (2026-07-26)

The follow-up 65K NCU comparison confirms that SplitZip's residual ByteV2
gap is reconstruction arithmetic, not metadata or DRAM traffic. With Q1 and
explicit split 20, the strict same-GPU serial profiles execute 41,528,425
instructions for ByteV2 and 44,963,623 for SplitZip (+8.27%). SplitZip uses
255 rather than 250 registers/thread and raises ALU-pipe activity from 19.27%
to 23.18%, while DRAM bytes differ by only 0.10%. Both paths keep the same
160-CTA grid, 51,072-byte shared allocation, two-CTA/SM limit, tensor-core
mainloop, split output, and original combine kernel.

The retained change expresses reconstruction of each BF16 byte as an explicit
single `lop3.b32` select with the per-byte `0x80808080` mask. It replaces the
compiler's separate mask and merge instructions without changing the
SplitZip wire or escape patch. SASS contains one LOP3 for each selection and
the dynamic instruction count falls exactly by 2,097,152 to 42,866,471
(-4.66%). This removes 61.0% of the baseline SplitZip-vs-ByteV2 instruction
excess. ALU activity falls to 21.02%; registers remain 255 and local
loads/stores remain zero.

The final installed binary's NCU main-kernel duration improves from 451.424
to 448.352 us (-0.68%). An earlier build of the identical retained source
measures 444.928 us (-1.44%), exposing replay/run variation despite identical
instructions. Final three-round event timing moves the mean of SplitZip
repetition medians from 411.204 to 410.897 us (-0.075%), while raw-normalized
latency moves from 0.88547x to 0.88489x. The first retained build measures
410.428 us, so no event-level speedup is claimed. The result is retained for
its exact structural improvement and lack of regression. At 77.61% of peak
DRAM-read throughput, with long scoreboard the largest stall ratio, the
removed ALU instructions are mostly hidden behind the memory critical path.

A second attempt reordered the same reconstruction into two lexical scopes
to shorten live ranges. It retained 255 registers and the identical
42,866,471 instructions. Its 410.445-us event result is indistinguishable
from both retained builds, so it was rejected.

The final bitwise gates cover the repeated real capture at 65K, every finite
BF16 exponent 0--254 in one pressure tile with 240 escape entries, and the
ragged `[4099,2051]` topology with 64 shared-prefix blocks, physical-page
permutation, and two three-row tails. Output and LSE mismatch counts are zero
throughout, with no page overflow.

This establishes the format-preserving optimization limit more clearly.
The retained kernel remains 1,338,046 instructions and five registers/thread
above ByteV2; two retained-source NCU runs put its duration gap at
0.55%--1.32%, while final CUDA events put it at 1.42%. Eliminating that
remaining reconstruction requires changing the resident wire to retain the
exponent low bit, which is effectively ByteV2's layout. For a realtime
product path, the preferred architecture is therefore SplitZip as an
archive/transport representation with background conversion into ByteV2
resident pages, rather than repeatedly paying canonical SplitZip
reconstruction in attention.

Artifacts:

- `profile/bytev2-splitzip-fa2-ncu-a40-20260726/REPORT.md`
- `profile/bytev2-splitzip-fa2-ncu-a40-20260726/reports/`
- `profile/bytev2-splitzip-fa2-ncu-a40-20260726/analysis/`
- `profile/bytev2-splitzip-fa2-ncu-a40-20260726/harness/`

### A40 ByteV2 V6-256 Fixed-Page Integration (2026-07-27)

The production ByteV2 layout now retains the V5 metadata and dense payload
ABI while reducing the page-wide outlier pool from 1,024 to 256 entries. The
page therefore changes from 52,096 to 50,560 bytes: fixed storage saving
against the 65,536-byte raw BF16 page increases from 20.5078% to 22.8516%.
SplitZip and the diagnostic sideband-high format intentionally retain their
V5 52,096-byte envelope; both require the larger pool for their existing
reader or fixed-256-entry-per-overlay contract.

The pool choice is supported by an exact production-allocation scan of a
32-layer Llama-3.1-8B natural-text capture. Calibration has 1,024 pages with
mean/P99/max demand 3.980/15/29 entries; evaluation has 2,048 pages with
3.514/9/27. Capacity 256 has zero observed overflow for full pages and every
`valid_rows=1..16` prefix. This is not a distribution-wide safety proof:
the capture is one model, batch one, short natural-text prefill KV, and does
not cover code, multilingual input, long-context distributions, multiple
batches, or decode tails.

For that reason V6 production is explicitly tied to the authoritative hybrid
raw sidecar. The cache planner and engine workspace binder reject compact-only
V6 configurations unless `BYTE_V2_FA2_HYBRID_RAW_FALLBACK=1` is set. The
low-level legacy decoder also traps on a fallback tile when the compact page
has no embedded raw payload, preventing standalone diagnostic calls from
silently continuing with invalid compressed values. The direct compact-only
writer remains available only as a fail-closed low-level diagnostic path.

Same-run Q1 CUDA-event medians show 50.688 versus 49.101 us at 4K
(V6 +3.15%), 118.630 versus 133.734 us at 16K (-11.29%), and 406.579 versus
465.856 us at 65K (-12.71%). All ByteV2 output and LSE comparisons are
bitwise exact against same-split raw FA2. A 65K-context, 128-output-token
production ABBA run also produces the same 128 tokens in all four requests.
Its median TPOT is 38.162 ms for V6 versus 40.201 ms for raw (-5.07%), while
median TTFT is 1.13% higher; a single ABBA block does not establish a stable
prefill regression or improvement.

With a 20 GB KV budget, the measured planner capacity is 152,576 tokens for
raw and 194,112 for V6 (+27.22%), including the hybrid raw sidecar and shared
staging workspace. Historical V5 capacity is 188,432 tokens, so V6 adds
5,680 tokens (+3.01%) over that earlier format.

NCU explains the long-sequence result structurally. Raw/V6 DRAM read is
299.619/241.886 MB (-19.27%); achieved occupancy is 8.44%/15.55%; and the
long-scoreboard ratio is 4.578/2.487. V6 shifts some pressure to decode and
shared-memory dependencies: its short-scoreboard ratio rises from 0.149 to
0.794, and NCU flags uncoalesced global loads and shared-store bank
conflicts. NCU replay duration is diagnostic only and is not substituted for
CUDA-event or E2E latency.

Shrinking the unused pool tail is primarily a capacity improvement, not a
reader-traffic optimization. Historical V5 NCU reads 242.148 MB and current
V6 reads 241.886 MB, only 0.108% less, because unused pool bytes were never
loaded on the hot path. The historical V5 result also comes from a different
binary and is not used as a strict causal latency A/B.

The focused sideband, page-size, legacy fallback, hybrid overflow-promotion,
planner, and binder gates all pass. The complete attention-layout regression
reports 403 passed and one skipped.

Artifacts:

- `profile/bytev2-v6-256-fa2-a40-20260727/REPORT.md`
- `profile/bytev2-v6-256-fa2-a40-20260727/analysis/`
- `profile/bytev2-v6-256-fa2-a40-20260727/reports/`
- `profile/bytev2-v6-pool-demand-a40-20260727/REPORT.md`
