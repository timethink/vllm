# ByteV2 Current GQA4 vs FA2 Paged Decode Profile - 2026-06-24

## 1. Setup

Repo: `/mnt/sdb/yxz/ByteV2/vllm`

GPU: NVIDIA A40, `CUDA_VISIBLE_DEVICES=4`

ByteV2 path:

- GQA4 guarded no-fallback/no-outlier split-k path.
- Current retained optimization includes PV safe-page descriptor.
- QK lightweight descriptor and BF16 `tmp_out` experiments are not retained.

Artifacts:

- `profiles/byte_v2_vs_fa2_current_gqa_pv_b1_4k16k_20260624.jsonl`
- `profiles/byte_v2_vs_fa2_current_gqa_pv_b4_4k16k_20260624.jsonl`
- `profiles/byte_v2_ncu_gqa_current_16384_p64_20260624.ncu-rep`
- `profiles/byte_v2_ncu_gqa_current_16384_p64_20260624_raw.csv`
- `profiles/byte_v2_ncu_gqa_reduce_current_16384_p64_20260624.ncu-rep`
- `profiles/byte_v2_ncu_gqa_reduce_current_16384_p64_20260624_raw.csv`
- `profiles/fa2_ncu_paged_16384_20260624.ncu-rep`
- `profiles/fa2_ncu_paged_16384_20260624_raw.csv`

## 2. Kernel Time

### Batch = 1

| seq_len | ByteV2 best | ByteV2 median | FA2 median | gap | workspace |
| ---: | --- | ---: | ---: | ---: | ---: |
| 4096 | p32 | 0.0983 ms | 0.0645 ms | 1.52x | 2.02 MiB |
| 8192 | p32 | 0.1618 ms | 0.0717 ms | 2.26x | 4.03 MiB |
| 16384 | p64 | 0.3082 ms | 0.1239 ms | 2.49x | 4.03 MiB |

### Batch = 4

| seq_len | ByteV2 best | ByteV2 median | FA2 median | gap | workspace |
| ---: | --- | ---: | ---: | ---: | ---: |
| 4096 | p32 / p64 | 0.3092 ms | 0.1249 ms | 2.48x | 8.06 / 4.03 MiB |
| 8192 | p64 | 0.5806 ms | 0.2232 ms | 2.60x | 8.06 MiB |
| 16384 | p128 | 1.1284 ms | 0.4219 ms | 2.67x | 8.06 MiB |

Interpretation:

1. Current GQA4 ByteV2 is still about 2.3-2.7x slower than FA2 at 8k-16k.
2. Batch=4 does not close the gap, so this is not primarily a CTA-count problem.
3. ByteV2 needs split-k workspace; FA2 reported workspace is 0 in this isolated benchmark.

## 3. Nsight Compute - 16k

NCU absolute duration can differ from CUDA event timing because of replay and profiling
settings. Use this table mostly for structural comparison.

| metric | ByteV2 main p64 | ByteV2 reduce p64 | FA2 splitkv |
| --- | ---: | ---: | ---: |
| NCU duration | 354.0 us | 20.9 us | 115.7 us |
| DRAM throughput | 186.7 GB/s | 202.1 GB/s | 582.7 GB/s |
| compute/memory throughput | 73.6% | 55.0% | 86.6% |
| DRAM throughput pct | 27.0% | 30.3% | 86.6% |
| L1/TEX throughput | 75.9% | 70.1% | 23.4% |
| L2 hit rate | 31.9% | 58.9% | 1.1% |
| issue active | 72.4% | 26.4% | 10.7% |
| eligible warps / scheduler | 2.92 | 0.51 | 0.11 |
| active warps / scheduler | 8.73 | 10.55 | 0.99 |
| registers / thread | 48 | 24 | 252 |
| static shared memory | 2.096 KiB | 0 | 0 |
| barrier stall | 0.07 | 0.00 | 0.40 |
| long scoreboard | 2.84 | 32.44 | 3.30 |
| short scoreboard | 0.70 | 1.37 | 0.18 |
| MIO throttle | 0.11 | 0.22 | 0.25 |
| wait stall | 1.47 | 2.26 | 2.07 |
| global load inst | 7,544,832 | 102,400 | 9,408 |
| global store inst | 40,960 | 4,096 | 1,064 |

## 4. Gap Analysis

FA2 is behaving like a streaming raw-KV kernel:

- It reaches about 583 GB/s DRAM throughput on this profile.
- L2 hit rate is low because it is streaming through KV.
- Issue active is low, but that is expected for a high-register, memory-streaming kernel.

ByteV2 main is not HBM-bandwidth bound:

- DRAM throughput is only about 187 GB/s, far below FA2.
- L1/TEX throughput and issue active are high.
- Global load instruction count is much higher than FA2.
- The remaining main-kernel cost is dominated by payload decode, address/metadata dependency,
  and scalarized K/V element loading, not by raw DRAM bytes.

ByteV2 reduce is not the primary wall-time bottleneck after warp reduce:

- It is about 21 us in NCU for 16k/p64.
- Its `long_scoreboard` is high because it reads partition stats/tmp_out, but the absolute
  duration is much smaller than the main kernel.

## 5. Recommended Next Step

Do not continue with:

- Same-shape QK lightweight descriptor.
- BF16 `tmp_out` element-size compression.
- Small barrier/reduction tweaks.

Next useful experiments should be structural:

1. Macro descriptor / tile staging for payload decode:
   reduce scalar per-element metadata/address work rather than caching a few fields per thread.
2. PV vectorized/tile decode staging:
   make V decode and PV more contiguous and closer to FA-style tile processing.
3. Split-k partition write-count reduction:
   reduce how often main writes `tmp_out`, or specialize/fuse main+reduce for small partition
   counts, instead of only shrinking element size.

## 6. K Staging Update

After the initial profile above, GQA4 safe-page K tile staging was implemented and retained.
The new path decodes each safe 16-token K page into shared memory once and lets the four
q_group warps reuse it.

Additional artifacts:

- `profiles/byte_v2_decode_microbench_gqa4_k_staging_b1_8k16k_20260624.jsonl`
- `profiles/byte_v2_decode_microbench_gqa4_k_staging_b4_8k16k_20260624.jsonl`
- `profiles/byte_v2_vs_fa2_current_gqa_k_staging_b1_4k16k_20260624.jsonl`
- `profiles/byte_v2_vs_fa2_current_gqa_k_staging_b4_4k16k_20260624.jsonl`
- `profiles/byte_v2_ncu_gqa_k_staging_16384_p64_20260624.ncu-rep`
- `profiles/byte_v2_ncu_gqa_k_staging_16384_p64_20260624_raw.csv`

### New Kernel Time

Batch = 1:

| seq_len | ByteV2 best | ByteV2 median | FA2 median | gap | workspace |
| ---: | --- | ---: | ---: | ---: | ---: |
| 4096 | p64 | 0.0809 ms | 0.0640 ms | 1.26x | 1.01 MiB |
| 8192 | p128 | 0.1403 ms | 0.0717 ms | 1.96x | 1.01 MiB |
| 16384 | p64 | 0.2591 ms | 0.1239 ms | 2.09x | 4.03 MiB |

Batch = 4:

| seq_len | ByteV2 best | ByteV2 median | FA2 median | gap | workspace |
| ---: | --- | ---: | ---: | ---: | ---: |
| 4096 | p64 | 0.2611 ms | 0.1249 ms | 2.09x | 4.03 MiB |
| 8192 | p64 | 0.4772 ms | 0.2222 ms | 2.15x | 8.06 MiB |
| 16384 | p128 | 0.9083 ms | 0.4214 ms | 2.16x | 8.06 MiB |

### New NCU Main-Kernel Comparison

| metric | PV descriptor baseline | K staging | FA2 splitkv |
| --- | ---: | ---: | ---: |
| NCU duration | 354.0 us | 289.9 us | 115.7 us |
| DRAM throughput | 186.7 GB/s | 224.7 GB/s | 582.7 GB/s |
| compute/memory throughput | 73.6% | 69.5% | 86.6% |
| DRAM throughput pct | 27.0% | 32.4% | 86.6% |
| L1/TEX throughput | 75.9% | 74.6% | 23.4% |
| L2 hit rate | 31.9% | 35.1% | 1.1% |
| issue active | 72.4% | 57.9% | 10.7% |
| eligible warps / scheduler | 2.92 | 1.12 | 0.11 |
| registers / thread | 48 | 40 | 252 |
| static shared memory | 2.1 KiB | 10.3 KiB | 0 |
| barrier stall | 0.07 | 0.48 | 0.40 |
| long scoreboard | 2.84 | 5.22 | 3.30 |
| global load inst | 7.54M | 2.83M | 9.4K |
| shared load wavefronts | 2.19M | 4.29M | 2.27M |
| shared store wavefronts | 0.57M | 1.10M | 0.04M |

Interpretation:

1. K staging cuts ByteV2 main-kernel global load instructions by about 2.7x and lowers NCU
   main duration by about 18%.
2. The tradeoff is visible: more static shared memory, more shared traffic, and higher
   barrier/scoreboard stalls. The tradeoff is still clearly positive.
3. The remaining FA2 gap is now roughly 2.0-2.2x at 8k-16k. ByteV2 is still not close to
   FA2's streaming DRAM throughput, but the largest repeated-K-load issue has been reduced.

Updated next steps:

1. Refine K staging layout or buffering to reduce the new shared/barrier cost.
2. Evaluate PV tile/vectorized decode staging.
3. Reduce split-k partition writes or specialize main+reduce for favorable partition counts.

## 7. BF16 Shared K Staging Update

K staging was refined again by storing staged K as BF16 bits in shared memory instead of
float. This keeps the staged payload semantically identical to decoded BF16 K while reducing
static shared memory footprint.

Additional artifacts:

- `profiles/byte_v2_decode_microbench_gqa4_k_staging_bf16_smem_b1_8k16k_20260624.jsonl`
- `profiles/byte_v2_decode_microbench_gqa4_k_staging_bf16_smem_b4_8k16k_20260624.jsonl`
- `profiles/byte_v2_vs_fa2_current_gqa_k_bf16_smem_b1_4k16k_20260624.jsonl`
- `profiles/byte_v2_vs_fa2_current_gqa_k_bf16_smem_b4_4k16k_20260624.jsonl`
- `profiles/byte_v2_ncu_gqa_k_staging_bf16_smem_16384_p64_20260624.ncu-rep`
- `profiles/byte_v2_ncu_gqa_k_staging_bf16_smem_16384_p64_20260624_raw.csv`

### New Kernel Time

Batch = 1:

| seq_len | ByteV2 best | ByteV2 median | FA2 median | gap | workspace |
| ---: | --- | ---: | ---: | ---: | ---: |
| 4096 | p64 | 0.0809 ms | 0.0645 ms | 1.25x | 1.01 MiB |
| 8192 | p32 | 0.1362 ms | 0.0717 ms | 1.90x | 4.03 MiB |
| 16384 | p64 | 0.2386 ms | 0.1239 ms | 1.93x | 4.03 MiB |

Batch = 4:

| seq_len | ByteV2 best | ByteV2 median | FA2 median | gap | workspace |
| ---: | --- | ---: | ---: | ---: | ---: |
| 4096 | p32 | 0.2386 ms | 0.1249 ms | 1.91x | 8.06 MiB |
| 8192 | p64 | 0.4372 ms | 0.2232 ms | 1.96x | 8.06 MiB |
| 16384 | p128 | 0.8315 ms | 0.4219 ms | 1.97x | 8.06 MiB |

### New NCU Main-Kernel Comparison

| metric | float K staging | BF16 shared K | FA2 splitkv |
| --- | ---: | ---: | ---: |
| NCU duration | 289.9 us | 261.1 us | 115.7 us |
| DRAM throughput | 224.7 GB/s | 255.3 GB/s | 582.7 GB/s |
| compute/memory throughput | 69.5% | 77.7% | 86.6% |
| DRAM throughput pct | 32.4% | 37.0% | 86.6% |
| L1/TEX throughput | 74.6% | 81.7% | 23.4% |
| issue active | 57.9% | 65.0% | 10.7% |
| eligible warps / scheduler | 1.12 | 1.94 | 0.11 |
| active warps / scheduler | 7.24 | 10.26 | 0.99 |
| registers / thread | 40 | 40 | 252 |
| static shared memory | 10.3 KiB | 6.2 KiB | 0 |
| barrier stall | 0.48 | 0.71 | 0.40 |
| long scoreboard | 5.22 | 5.67 | 3.30 |
| global load inst | 2.83M | 2.83M | 9.4K |
| shared load wavefronts | 4.29M | 4.29M | 2.27M |
| shared store wavefronts | 1.10M | 1.11M | 0.04M |

Interpretation:

1. BF16 shared K improves the float K staging path by about 7.7% geometric mean on the
   8k/16k batch=1/4 sweep.
2. The main improvement is reduced static shared memory and better eligible/active warp
   counts, not fewer shared wavefronts.
3. ByteV2's current 8k-16k gap to FA2 is now about 1.9-2.0x in this isolated benchmark.

Updated next steps:

1. Avoid more small descriptor, pair-sharing, or tile-final barrier changes; they have not met the
   retention threshold.
2. Avoid PV in-place accumulation; it did not reduce the dominant dependency.
3. Evaluate larger PV tile/vectorized decode mapping changes that actually change V decode or
   compute reuse.
4. Avoid grouped reduce variants that reduce partition-lane parallelism; current warp-per-dim
   reduce remains faster.

## 8. PV Pair-Vectorized Decode Attempt

I tested a narrower PV vectorization attempt after BF16 shared K staging: adjacent dims in each
16-wide codec tile shared one low-pair load and one packed-code load through a warp shuffle.
The unsafe/fallback path was unchanged, and the change only touched the no-mask GQA4 PV fast
path.

Artifact:

- `profiles/byte_v2_decode_microbench_gqa4_v_pair_b1_8k16k_20260624.jsonl`

Batch = 1:

| seq_len | BF16 shared K baseline | pair-vectorized PV | result |
| ---: | ---: | ---: | ---: |
| 8192 | 0.1362 ms | 0.1444 ms | -6.0% |
| 16384 | 0.2376 ms | 0.2550 ms | -7.3% |

The experiment was reverted. The likely issue is that the reduced payload byte loads are smaller
than the added shuffle dependency in the current one-output-dim-per-thread PV loop. Future PV work
should use a larger structural change, such as staging a V tile only if it also changes compute
reuse, or changing the PV mapping so each thread owns more useful contiguous V work.

## 9. K Staging Final-Barrier Attempt

I also tested conditionally skipping the tile-final `__syncthreads()` after the QK page loop when
the last page already used the safe-page K staging path, which has its own trailing barrier.

Artifacts:

- `profiles/byte_v2_decode_microbench_gqa4_k_barrier_b1_8k16k_20260624.jsonl`
- `profiles/byte_v2_decode_microbench_gqa4_k_barrier_b4_8k16k_20260624.jsonl`

Batch = 1:

| seq_len | BF16 shared K baseline | conditional barrier | result |
| ---: | ---: | ---: | ---: |
| 8192 | 0.1362 ms | 0.1362 ms | 0.0% |
| 16384 | 0.2376 ms | 0.2386 ms | -0.4% |

Batch = 4:

| seq_len | BF16 shared K baseline | conditional barrier | result |
| ---: | ---: | ---: | ---: |
| 8192 | 0.4342 ms | 0.4342 ms | 0.0% |
| 16384 | 0.8325 ms | 0.8335 ms | -0.1% |

This experiment was reverted. The result suggests the extra tile-final CTA barrier is not the
dominant barrier cost after BF16 shared K staging; the remaining cost is more likely inside the
per-page staging/QK/PV dependency chain or from split-k partition traffic.

## 10. Reduce Variants

I tested two GQA4 reduce alternatives after BF16 shared K staging:

1. Shared-weight reduce: reuse the existing one-CTA-per-head reduce kernel so partition weights are
   computed once per head instead of once per output dim.
2. Warp4 grouped reduce: one warp handles 4 output dims with 8 lanes per dim, sharing stats loads
   across the 4 dims.

Artifacts:

- `profiles/byte_v2_decode_microbench_gqa4_shared_reduce_b1_8k16k_20260624.jsonl`
- `profiles/byte_v2_decode_microbench_gqa4_reduce_warp4_b1_8k16k_20260624.jsonl`

Batch = 1:

| seq_len | BF16 shared K baseline | shared-weight reduce | warp4 grouped reduce |
| ---: | ---: | ---: | ---: |
| 8192 | 0.1362 ms | 0.1516 ms | 0.1567 ms |
| 16384 | 0.2376 ms | 0.2934 ms | 0.3021 ms |

Both experiments were reverted. The result is clear enough despite the warp4 run using GPU5 due
to GPU4 contention: reducing stats duplication is less important than preserving partition
parallelism in the reduce stage. Keep the current warp-per-dim reduce unless a future change also
reduces the number of intermediate partitions written by the main kernel.

## 11. PV In-Place Accumulation Attempt

I tested removing the per-tile `pv[4]` temporary accumulator in the GQA4 main kernel. The experiment
scaled `acc` by `alpha` before the PV loop and accumulated `shared_probs * V` directly into `acc`.
This reduced one explicit temporary array and one tile-end combine loop, but it did not change V
decode reuse.

Artifacts:

- `profiles/byte_v2_decode_microbench_gqa4_pv_inplace_b1_8k16k_20260624.jsonl`
- `profiles/byte_v2_decode_microbench_gqa4_pv_inplace_b4_8k16k_20260624.jsonl`
- `profiles/byte_v2_decode_microbench_gqa4_restored_baseline_b1_8k16k_20260624.jsonl`
- `profiles/byte_v2_decode_microbench_gqa4_restored_baseline_b4_8k16k_20260624.jsonl`

Batch = 1, same GPU sanity check:

| seq_len | restored baseline | PV in-place | result |
| ---: | ---: | ---: | ---: |
| 8192 | 0.1434 ms | 0.1434 ms | 0.0% |
| 16384 | 0.2447 ms | 0.2447 ms | 0.0% |

Batch = 4, same GPU sanity check:

| seq_len | restored baseline | PV in-place | result |
| ---: | ---: | ---: | ---: |
| 8192 | 0.4500 ms | 0.4485 ms | +0.3% |
| 16384 | 0.8591 ms | 0.8591 ms | 0.0% |

This experiment was reverted. The result suggests the `pv[4]` temporary is not the bottleneck.
Future PV work should change V decode or compute reuse, not just move the accumulation
destination.

## 12. V Base Broadcast Attempt

I tested a narrower V decode reuse change: within each 16-wide V codec tile, only the leader lane
loaded the tile `base` metadata byte and then broadcast it with `shfl` to the other lanes in that
half warp. The safe-page PV path then decoded V directly from the broadcast base.

Artifacts:

- `profiles/byte_v2_decode_microbench_gqa4_v_base_broadcast_b1_8k16k_20260624.jsonl`
- `profiles/byte_v2_decode_microbench_gqa4_v_base_broadcast_b4_8k16k_20260624.jsonl`

Batch = 1, same GPU comparison to restored baseline:

| seq_len | restored baseline | V base broadcast | result |
| ---: | ---: | ---: | ---: |
| 8192 | 0.1434 ms | 0.1434 ms | 0.0% |
| 16384 | 0.2447 ms | 0.2437 ms | +0.4% |

Batch = 4, same GPU comparison to restored baseline:

| seq_len | restored baseline | V base broadcast | result |
| ---: | ---: | ---: | ---: |
| 8192 | 0.4500 ms | 0.4454 ms | +1.0% |
| 16384 | 0.8591 ms | 0.8571 ms | +0.2% |

This experiment was reverted. V base metadata reuse is too small to justify replacing the existing
descriptor helper with shuffle-based manual decode. Future PV work needs to change the amount of V
payload work or how it is reused, not just the metadata base load.

## 13. Fresh Current-vs-Raw Profile on GPU5

This rerun uses the retained BF16 shared-K GQA4 kernel and profiles ByteV2 main, ByteV2 reduce,
and raw paged FA2 on the same A40 (`CUDA_VISIBLE_DEVICES=5`) with the same 16k/p64 setup.

Artifacts:

- `profiles/byte_v2_vs_raw_gqa_current_16384_p64_gpu5_20260624.jsonl`
- `profiles/byte_v2_ncu_gqa_main_retained_16384_p64_gpu5_20260624.ncu-rep`
- `profiles/byte_v2_ncu_gqa_main_retained_16384_p64_gpu5_20260624_raw.csv`
- `profiles/byte_v2_ncu_gqa_reduce_retained_16384_p64_gpu5_20260624.ncu-rep`
- `profiles/byte_v2_ncu_gqa_reduce_retained_16384_p64_gpu5_20260624_raw.csv`
- `profiles/raw_fa2_ncu_paged_16384_p64_gpu5_20260624.ncu-rep`
- `profiles/raw_fa2_ncu_paged_16384_p64_gpu5_20260624_raw.csv`

Stable CUDA-event microbench, no NCU instrumentation:

| kernel | median | mean | min | p90 |
| --- | ---: | ---: | ---: | ---: |
| ByteV2 GQA4 p64 | 0.2417 ms | 0.2411 ms | 0.2365 ms | 0.2427 ms |
| raw paged FA2 | 0.1362 ms | 0.1366 ms | 0.1352 ms | 0.1372 ms |

ByteV2 is currently about 1.77x slower end-to-end in this isolated single-token decode case.
NCU duration below is not meant to be added back into the CUDA-event timing because replay and
counter collection can perturb absolute kernel time; use it for structural comparison.

| metric | ByteV2 main | ByteV2 reduce | raw FA2 | main/raw |
| --- | ---: | ---: | ---: | ---: |
| NCU duration | 265.3 us | 20.7 us | 126.9 us | 2.09x |
| DRAM throughput | 411.6 GB/s | 294.1 GB/s | 596.9 GB/s | 0.69x |
| DRAM read | 91.6 MB | 6.1 MB | 75.4 MB | 1.21x |
| DRAM write | 17.6 MB | 0.004 MB | 0.365 MB | 48.3x |
| memory throughput | 77.1% | 54.1% | 89.5% | 0.86x |
| DRAM throughput pct | 60.1% | 42.6% | 89.5% | 0.67x |
| L1/TEX throughput | 80.8% | 69.6% | 21.2% | 3.80x |
| L2 throughput pct | 17.6% | 26.3% | 29.5% | 0.60x |
| issue active | 61.3% | 20.3% | 8.4% | 7.31x |
| eligible warps / scheduler | 1.87 | 0.51 | 0.10 | 19.3x |
| active warps / scheduler | 10.30 | 10.82 | 0.99 | 10.4x |
| registers / thread | 40 | 24 | 252 | 0.16x |
| shared memory / block | 7.2 KB | 1.0 KB | 82.9 KB | 0.09x |
| executed instructions | 69.7M | 1.78M | 4.44M | 15.7x |
| global load requests | 2.83M | 0.10M | 0.14M | 20.1x |
| global load sectors | 4.91M | 1.31M | 2.11M | 2.32x |
| `LDGSTS` inst | 0 | 0 | 0.136M | 0x |
| `LDGSTS` sectors | 0 | 0 | 2.10M | 0x |
| shared wavefronts | 8.14M | 0.07M | 2.40M | 3.39x |
| local-store traffic | 16.78 MB | 0 | 0 | n/a |
| warp cycles / issued inst | 16.0 | 41.4 | 10.2 | 1.57x |
| long scoreboard | 6.02 | 32.99 | 3.99 | 1.51x |
| barrier | 0.75 | 0 | 0.55 | 1.37x |
| short scoreboard | 2.04 | 1.41 | 0.19 | 10.9x |
| MIO throttle | 0.86 | 0.27 | 0.25 | 3.45x |

Main findings:

1. The biggest new actionable signal is local memory traffic in ByteV2 main. NCU reports
   `local_st` sectors equivalent to about 16.78 MB, matching almost all of the extra DRAM write.
   The likely source is dynamically indexed local arrays in the GQA4 main kernel, especially
   `acc[4]`, `softmax_m[4]`, `softmax_l[4]`, and per-tile `pv[4]`.
2. ByteV2 still executes far more instructions than raw FA2: 69.7M vs 4.44M. The remaining gap is
   dominated by compressed payload decode, address arithmetic, and scalar load dependencies rather
   than raw HBM bytes alone.
3. FA2's raw path uses a high-register, high-shared-memory streaming structure and moves KV through
   `LDGSTS`. ByteV2 currently uses much lower register/shared footprint but much higher issue
   activity, global load request count, and L1/TEX pressure.
4. ByteV2 shared traffic is high, but shared bank conflicts are not the main issue: current main has
   about 8.14M shared wavefronts and only about 28k shared bank conflicts. The problem is amount of
   shared traffic plus dependency, not a large bank-conflict multiplier.
5. ByteV2 reduce is still secondary in absolute time. Its long-scoreboard stall is high because it
   streams `exp_sums/tmp_out`, but it is only about 20 us in this setup.

Recommended next experiment:

1. First scalarize the GQA4 main local state to eliminate spills. Replace dynamically indexed local
   arrays with explicit scalar state (`acc0..acc3`, `m0..m3`, `l0..l3`, `pv0..pv3`) and use small
   helper/switch logic for the warp-owned softmax state. Keep only if CUDA-event median improves by
   at least 5% and NCU local traffic drops near zero.
2. After local spills are removed, re-profile the same 16k/p64 case. If DRAM write drops but time
   barely moves, the next bottleneck is payload decode and shared/L1 dependency.
3. Then focus on reducing global load requests and instruction count: coarser payload decode,
   page/tile macro descriptors, or a larger FA-style staging path that changes the amount of scalar
   metadata/address work. Small V metadata broadcasts and accumulation-only rewrites have already
   failed the retention threshold.

## 14. GQA4 Main Scalarization Attempt

I tested the scalarization recommended above by replacing the GQA4 main kernel's local
`acc[4]`, `softmax_m[4]`, `softmax_l[4]`, and `pv[4]` arrays with explicit scalar state. The goal
was to remove the local-memory spill seen in section 13.

Artifacts:

- `profiles/byte_v2_decode_microbench_gqa4_scalarized_smoke_1k4k_20260624.jsonl`
- `profiles/byte_v2_vs_raw_gqa_scalarized_16384_p64_gpu5_20260624.jsonl`
- `profiles/byte_v2_vs_raw_gqa_scalarized_16384_p64_gpu5_repeat_20260624.jsonl`
- `profiles/byte_v2_vs_raw_gqa_scalarized_b1_8k16k_p64_gpu5_20260624.jsonl`
- `profiles/byte_v2_vs_raw_gqa_scalarized_b4_8k16k_p64_gpu5_20260624.jsonl`
- `profiles/byte_v2_ncu_gqa_main_scalarized_16384_p64_gpu5_20260624.ncu-rep`
- `profiles/byte_v2_ncu_gqa_main_scalarized_16384_p64_gpu5_20260624_raw.csv`

Correctness:

- `CUDA_VISIBLE_DEVICES=5 .venv/bin/python -m pytest tests/v1/attention/test_byte_v2_layout.py -v`
  passed: 111 passed, 1 skipped.
- Smoke microbench kept the same raw FA2 comparison tolerance: max abs diff `0.0009765625`.

Performance, 16k/p64, GPU5:

| run | ByteV2 scalarized | previous retained baseline | change |
| --- | ---: | ---: | ---: |
| single 16k run | 0.2324 ms | 0.2417 ms | +3.8% |
| repeat 16k run | 0.2314 ms | 0.2417 ms | +4.2% |
| 8k/16k sweep 16k row | 0.2273 ms | 0.2417 ms | +5.9% |

Additional fixed-p64 sweep:

| case | scalarized ByteV2 | raw FA2 |
| --- | ---: | ---: |
| batch=1, 8k | 0.1372 ms | 0.0799 ms |
| batch=1, 16k | 0.2273 ms | 0.1372 ms |
| batch=4, 8k | 0.4280 ms | 0.2447 ms |
| batch=4, 16k | 0.8417 ms | 0.4628 ms |

NCU main-kernel counters vs the section 13 baseline:

| metric | baseline | scalarized | delta |
| --- | ---: | ---: | ---: |
| NCU duration | 265.3 us | 258.1 us | -2.7% |
| DRAM read | 91.6 MB | 90.2 MB | -1.6% |
| DRAM write | 17.6 MB | 4.06 MB | -76.9% |
| executed instructions | 69.7M | 73.3M | +5.1% |
| local load traffic | 1.31 MB | 0 | -100% |
| local store traffic | 16.78 MB | 0 | -100% |
| long scoreboard | 6.02 | 4.76 | -21.0% |
| MIO throttle | 0.86 | 0.71 | -17.6% |

Conclusion:

The experiment succeeded at removing the local-memory spill, but the median speedup was not
stable enough to exceed the 5% retention threshold. The extra scalarized control/data movement
raised executed instructions by about 5.1%, offsetting most of the saved local-memory traffic.
This source change was reverted. The useful conclusion is that local spills are real but not the
dominant remaining gap; the next optimization should target payload decode instruction count and
global load requests instead of only scalarizing accumulator state.

## 15. Single-Point Experiment 1: Specialized K Staging Decode

Purpose:

- Test one isolated instruction-count reduction in the GQA4 main kernel.
- Only changed the safe-page K staging loop.
- V path, softmax, reduce, split-k workspace, and fallback/unsafe paths were unchanged.

Change:

- Replaced the generic `byte_v2_load_payload_elem_no_fallback_no_outlier_bits` call inside K
  staging with a hand-written safe-page K decoder.
- Hoisted `dim_tile`, `dim_in_tile`, `tile_offset`, and K `base` out of the per-row load loop.
- Kept the same BF16 shared-K staging layout and the same output values.

Artifacts:

- `profiles/byte_v2_vs_raw_gqa_baseline_restored_16384_p64_gpu5_20260624.jsonl`
- `profiles/byte_v2_vs_raw_gqa_k_staging_specialized_16384_p64_gpu5_20260624.jsonl`
- `profiles/byte_v2_vs_raw_gqa_k_staging_specialized_b1_8k16k_p64_gpu5_20260624.jsonl`
- `profiles/byte_v2_vs_raw_gqa_k_staging_specialized_b4_8k16k_p64_gpu5_20260624.jsonl`
- `profiles/byte_v2_ncu_gqa_main_k_staging_specialized_16384_p64_gpu5_20260624.ncu-rep`
- `profiles/byte_v2_ncu_gqa_main_k_staging_specialized_16384_p64_gpu5_20260624_raw.csv`

Correctness:

- `CUDA_VISIBLE_DEVICES=5 .venv/bin/python -m pytest tests/v1/attention/test_byte_v2_layout.py -k "gqa_packed_cuda_matches_raw_reference" -v`
  passed: 2 passed, 110 deselected.
- Microbench max abs diff vs raw FA2 stayed at `0.0009765625`.

CUDA-event result:

| case | baseline | specialized K staging | change |
| --- | ---: | ---: | ---: |
| batch=1, 16k, p64, same fresh baseline | 0.2396 ms | 0.2386 ms | +0.4% |
| batch=1, 16k, p64, sweep run | 0.2396 ms | 0.2355 ms | +1.7% |
| batch=4, 16k, p64, previous baseline | 0.8591 ms | 0.8438 ms | +1.8% |
| batch=1, 8k, p64, previous baseline | 0.1495 ms | 0.1444 ms | +3.4% |
| batch=4, 8k, p64, previous baseline | 0.4500 ms | 0.4280 ms | +4.9% |

NCU main-kernel counters, 16k/p64:

| metric | baseline | specialized K staging | delta |
| --- | ---: | ---: | ---: |
| NCU duration | 265.3 us | 258.4 us | -2.6% |
| executed instructions | 69.7M | 59.4M | -14.8% |
| issued instructions | 69.7M | 59.5M | -14.8% |
| global load requests | 2.83M | 2.34M | -17.4% |
| global load sectors | 4.91M | 4.44M | -9.6% |
| registers / thread | 40 | 46 | +15.0% |
| shared wavefronts | 8.14M | 8.13M | -0.1% |
| local store traffic | 16.78 MB | 16.75 MB | -0.2% |
| long scoreboard | 6.02 | 6.93 | +15.1% |
| short scoreboard | 2.04 | 2.32 | +13.4% |
| MIO throttle | 0.86 | 0.69 | -19.9% |

Conclusion:

This single-point change worked for the intended local metric: it reduced main-kernel executed
instructions by about 14.8% and global load requests by about 17.4%. However, it did not translate
to a stable wall-time gain above the retention threshold. The likely reason is that the new
per-thread dim-owned staging form increases register count and exposes higher long/short
scoreboard stalls, while shared traffic and local spill remain essentially unchanged.

The source change was reverted. The result is still useful: K staging helper overhead is real, but
instruction-count reduction alone is insufficient if it increases dependency stalls. The next
single-point experiment should target dependency and load coalescing in the V/PV path or remove
safe-page branch/control overhead, not another K-staging arithmetic hoist.

## 16. Single-Point Experiment 2: Safe-Page Branch/Control Removal

Purpose:

- Test whether guarded GQA4 loses measurable time to `page_unsafe_flags` load, safe/unsafe
  branch selection, and the extra dead unsafe path in the no-outlier benchmark case.
- Keep the experiment correctness-contained: default guarded behavior remained unchanged, and the
  safe-only path was enabled only through a temporary host-side env switch.

Temporary change:

- Added `BYTE_V2_EXPERIMENT_GQA_SAFE_ONLY=1`.
- Under `--guarded-split --gqa-packed`, the env switch launched the same GQA4 kernel with
  `UsePageUnsafeFlags=false` and passed `nullptr` for `page_unsafe_flags`.
- The normal guarded path still launched `UsePageUnsafeFlags=true`.

Artifacts:

- `profiles/byte_v2_vs_raw_gqa_safe_only_16384_p64_gpu5_20260624.jsonl`
- `profiles/byte_v2_vs_raw_gqa_after_safe_only_revert_16384_p64_gpu5_20260624.jsonl`
- `profiles/byte_v2_ncu_gqa_main_safe_only_16384_p64_gpu5_20260624.ncu-rep`
- `profiles/byte_v2_ncu_gqa_main_safe_only_16384_p64_gpu5_20260624_raw.csv`

Correctness:

- `BYTE_V2_EXPERIMENT_GQA_SAFE_ONLY=1 CUDA_VISIBLE_DEVICES=5 .venv/bin/python -m pytest tests/v1/attention/test_byte_v2_layout.py -k "gqa_packed_cuda_matches_raw_reference" -v`
  passed: 2 passed, 110 deselected.
- This test uses no unsafe pages, so it is the right scope for the safe-only assumption. The env
  switch was not used for guarded unsafe-page correctness.

CUDA-event result, batch=1, seq=16k, partition=64:

| run | ByteV2 | raw FA2 | note |
| --- | ---: | ---: | --- |
| safe-only env | 0.2437 ms | 0.1362 ms | no stable improvement |
| restored default after revert | 0.2437 ms | 0.1362 ms | same median in this run |
| previous restored baseline | 0.2406 ms | 0.1362 ms | shows run-to-run noise scale |

NCU main-kernel counters, safe-only vs section 13 baseline:

| metric | baseline | safe-only | delta |
| --- | ---: | ---: | ---: |
| NCU duration | 265.3 us | 264.5 us | -0.3% |
| executed instructions | 69.7M | 70.5M | +1.2% |
| issued instructions | 69.7M | 70.5M | +1.2% |
| global load requests | 2.83M | 2.76M | -2.3% |
| global load sectors | 4.91M | 4.84M | -1.4% |
| registers / thread | 40 | 38 | -5.0% |
| shared wavefronts | 8.14M | 8.14M | 0.0% |
| local store sectors | 524k | 523k | -0.2% |
| long scoreboard | 6.02 | 5.82 | -3.4% |
| short scoreboard | 2.04 | 2.07 | +1.3% |
| MIO throttle | 0.86 | 0.78 | -9.7% |

Conclusion:

The safe-page branch/control overhead is not a material bottleneck in the current main kernel.
Removing it produced only a sub-1% NCU-duration change, no CUDA-event speedup, and even a small
increase in executed instructions. The temporary env switch and source change were reverted, and
the default guarded GQA binary was rebuilt.

This point should not be revisited until the larger payload decode/PV dependency costs are reduced.
The next single-point experiment should target the PV path directly, because this is where every
output lane repeatedly decodes V payload bytes and consumes `shared_probs` with long dependency
chains.

## 17. PV Path Structural Analysis

Scope:

- Analyze the current retained GQA4 main kernel without changing source.
- Focus on V payload decode, `shared_probs` fanout, and PV accumulation pressure.
- Source locations:
    - GQA4 shared state: `csrc/libtorch_stable/byte_v2/byte_v2_ops.cu:1021`.
    - softmax probability materialization: `byte_v2_ops.cu:1163`.
    - PV loop: `byte_v2_ops.cu:1182`.
    - V descriptor/decode helper: `byte_v2_ops.cu:369`.

Current PV structure:

- One CTA has 128 threads, split as 4 warps for GQA4.
- During QK, warp `0..3` owns query group `0..3`.
- During PV, every thread owns one output `dim`, decodes one `V(row, dim)`, then applies it to all
  four query groups:
  `pv[q_group] += shared_probs[q_group][tile_offset] * v_value`.
- Therefore V decode is not repeated four times per q group inside the same thread. The decoded
  `v_value` is already reused across the four GQA groups.

V decode cost:

- Safe-page V descriptor hoists page pointer, payload offset, and base per page/dim tile, but each
  row still performs:
    - one low-byte payload load,
    - one packed 4-bit-code load,
    - nibble extract,
    - `base + code`,
    - BF16-bit reconstruction.
- Per 16k/p64 run, ByteV2 main has about 2.83M global load instructions vs raw FA2's 0.009M
  SASS global-load instructions in the raw CSV. This is not an apples-to-apples instruction model
  because FA2 uses `LDGSTS`, but it still shows ByteV2 is issuing many scalar byte loads.
- Prior local V optimizations were too small:
    - pair-vectorized packed-code sharing was slower by 6-7%;
    - V base broadcast improved only 0-1%;
    - so the remaining V opportunity is structural, not another one-byte metadata reuse.

`shared_probs` fanout:

- Probabilities are written to `shared_probs[q_group][tile_offset]` once per q group/tile offset.
- The PV loop then has all 128 output-dim threads read those values. At the warp-instruction level,
  this is roughly `rows * q_groups * warps = 64 * 4 * 4 = 1024` shared-load warp instructions per
  CTA just for `shared_probs`.
- For 2048 CTAs in the 16k/p64 setup, that is about 2.1M shared-load instructions, roughly half of
  the measured 4.25M shared-load instructions in ByteV2 main. The other large half is mostly the
  staged-K QK shared loads.
- This explains why ByteV2 has much higher short-scoreboard and MIO pressure than raw FA2:
    - shared-load SASS instructions: 4.25M vs 0.0097M;
    - shared-store SASS instructions: 1.09M vs 0.019M;
    - short scoreboard: 2.04 vs 0.19;
    - MIO throttle: 0.86 vs 0.25.

PV accumulation dependency:

- Each output thread maintains four independent scalar accumulators, one per q group.
- For each row, the loop performs four dependent FMA chains:
  `pv0 += p0 * v`, `pv1 += p1 * v`, `pv2 += p2 * v`, `pv3 += p3 * v`.
- Moving the destination from temporary `pv[4]` directly into `acc[4]` was previously neutral, so
  the bottleneck is not the final combine loop. The pressure comes from the row-by-row shared-prob
  load plus scalar FMA chain.
- The measured long-scoreboard gap is smaller than the shared-memory gap but still real:
  ByteV2 main `6.02` vs raw FA2 `3.99`. This is consistent with serialized scalar V payload loads
  feeding the PV loop.

Comparison with FA2:

- FA2 keeps softmax probabilities in register fragments (`rP`) and feeds them directly into a tiled
  `gemm_rs` with staged/transposed V. It does not materialize scalar probabilities to shared and
  then have every output dimension reload them.
- ByteV2 cannot directly copy FA2's tensor-core path because V is compressed and must be decoded,
  but the useful design lesson is clear: reduce the `P` fanout through shared memory, or stage V in
  a form that makes a tiled `P x V` loop possible.

Most plausible next experiments:

1. V tile staging micro-experiment:
   decode safe-page V for the whole 64x128 compute tile into BF16 shared memory, then run PV from
   staged V. Keep `shared_probs` unchanged. This isolates whether decoupling V decode/global-load
   latency from PV improves long-scoreboard enough to pay for extra shared traffic and barriers.
2. Warp-local `P` consumption experiment:
   restructure PV so each q-group warp consumes its own probabilities without reading all four
   q-groups from shared. This likely requires V staging, otherwise V loads are repeated per q group.
3. Smaller diagnostic first:
   add NCU section/source profile around PV-only variants or temporarily skip PV accumulation to
   estimate an upper bound. This is not correctness-preserving, so it should be benchmark-only and
   immediately reverted.

Working conclusion:

The current PV path's biggest actionable problem is not that each thread decodes the same V value
four times; it does not. The bigger issue is that `shared_probs` is a cross-warp broadcast fabric:
small probability scalars are produced by one q-group warp, materialized to shared, and then fanned
out to every output-dim warp. That creates high shared-load instruction count, short-scoreboard
stall, and MIO pressure. V payload decode contributes long-scoreboard and global-load request
pressure, but prior byte-level reuse attempts show that the next useful change needs to alter the
tile structure, not just hoist one more scalar.

## 18. Single-Point Experiment 3: V Tile Staging

Purpose:

- Test whether decoupling safe-page V payload decode/global loads from the PV accumulation loop
  reduces long-scoreboard enough to offset extra shared memory traffic.
- Keep `shared_probs` unchanged, so this experiment isolates V staging rather than changing the
  probability fanout structure.

Temporary change:

- Added a no-fallback/no-outlier descriptor helper that returns BF16 bits instead of float.
- Added `shared_v_tile[Policy::ComputeBlockN][Policy::HeadDimV]` to the GQA4 main kernel.
- After softmax/probability materialization, decoded all safe-page V values for the current compute
  tile into `shared_v_tile`.
- The PV loop then loaded `V(row, dim)` from shared memory instead of decoding directly from the
  ByteV2 page payload.
- Unsafe pages continued to use the original direct generic decode path.

Artifacts:

- `profiles/byte_v2_vs_raw_gqa_v_tile_staging_16384_p64_gpu5_20260624.jsonl`
- `profiles/byte_v2_vs_raw_gqa_after_v_tile_staging_revert_16384_p64_gpu5_20260624.jsonl`
- `profiles/byte_v2_ncu_gqa_main_v_tile_staging_16384_p64_gpu5_20260624.ncu-rep`
- `profiles/byte_v2_ncu_gqa_main_v_tile_staging_16384_p64_gpu5_20260624_raw.csv`

Correctness:

- `CUDA_VISIBLE_DEVICES=5 .venv/bin/python -m pytest tests/v1/attention/test_byte_v2_layout.py -k "gqa_packed_cuda_matches_raw_reference" -v`
  passed: 2 passed, 110 deselected.

CUDA-event result, batch=1, seq=16k, partition=64:

| run | ByteV2 | raw FA2 | result |
| --- | ---: | ---: | --- |
| V tile staging | 0.3922 ms | 0.1362 ms | -61.6% vs ~0.242 ms baseline |
| restored default after revert | 0.2427 ms | 0.1362 ms | back to baseline range |

NCU main-kernel counters, V staging vs section 13 baseline:

| metric | baseline | V staging | delta |
| --- | ---: | ---: | ---: |
| NCU duration | 265.3 us | 456.7 us | +72.2% |
| DRAM read | 91.6 MB | 90.0 MB | -1.8% |
| DRAM write | 17.6 MB | 13.6 MB | -22.7% |
| executed instructions | 69.7M | 78.9M | +13.2% |
| global load requests | 2.83M | 2.89M | +2.3% |
| global load sectors | 4.91M | 5.02M | +2.2% |
| shared-load instructions | 4.25M | 4.78M | +12.3% |
| shared-store instructions | 1.09M | 1.61M | +48.1% |
| static shared memory / block | 6.2 KB | 22.6 KB | +264.6% |
| shared-memory occupancy limit | 14 blocks | 4 blocks | -71.4% |
| eligible warps / scheduler | 1.87 | 0.58 | -69.2% |
| long scoreboard | 6.02 | 3.87 | -35.7% |
| short scoreboard | 2.04 | 1.31 | -35.8% |
| barrier | 0.75 | 0.28 | -63.2% |
| MIO throttle | 0.86 | 0.06 | -93.3% |

Conclusion:

The hypothesis was partially right but not useful: V staging reduced several stall ratios, including
long scoreboard, short scoreboard, barrier, and MIO throttle. However, it also increased executed
instructions, shared load/store traffic, and static shared memory so much that occupancy collapsed
from a shared-memory limit of 14 blocks/SM to 4 blocks/SM. Wall time regressed by about 62% in the
CUDA-event benchmark.

The source change was reverted and the default guarded GQA binary was rebuilt. Future PV work
should not stage the entire 64x128 V tile in shared memory unless it also removes the
`shared_probs` fanout or changes PV into a more FA-style tiled `P x V` computation. The next
candidate should be a smaller structural change around probability consumption, not full V staging.

## 19. Single-Point Experiment 4: Warp-Local P Broadcast

Purpose:

- Test a smaller `shared_probs` fanout change without staging V.
- Keep V decode, softmax writeback, split-k workspace, and q-group ownership unchanged.
- In each PV row, lanes `0..3` of every warp loaded `shared_probs[0..3][tile_offset]`, then
  broadcast those four probabilities to the rest of the warp with `__shfl_sync`.

Temporary change:

- Replaced the PV inner loop:
  `pv[q_group] += shared_probs[q_group][tile_offset] * v_value`
  with a warp-local probability load/broadcast sequence.
- Applied to both unsafe and safe-page PV paths.

Artifacts:

- `profiles/byte_v2_vs_raw_gqa_p_broadcast_16384_p64_gpu5_20260624.jsonl`
- `profiles/byte_v2_vs_raw_gqa_after_p_broadcast_revert_16384_p64_gpu5_20260624.jsonl`
- `profiles/byte_v2_ncu_gqa_main_p_broadcast_16384_p64_gpu5_20260624.ncu-rep`
- `profiles/byte_v2_ncu_gqa_main_p_broadcast_16384_p64_gpu5_20260624_raw.csv`

Correctness:

- `CUDA_VISIBLE_DEVICES=5 .venv/bin/python -m pytest tests/v1/attention/test_byte_v2_layout.py -k "gqa_packed_cuda_matches_raw_reference" -v`
  passed: 2 passed, 110 deselected.

CUDA-event result, batch=1, seq=16k, partition=64:

| run | ByteV2 | raw FA2 | result |
| --- | ---: | ---: | --- |
| P broadcast | 0.2478 ms | 0.1362 ms | -2.1% vs ~0.2427 ms baseline |
| restored default after revert | 0.2427 ms | 0.1362 ms | back to baseline range |

NCU main-kernel counters, P broadcast vs section 13 baseline:

| metric | baseline | P broadcast | delta |
| --- | ---: | ---: | ---: |
| NCU duration | 265.3 us | 271.0 us | +2.1% |
| executed instructions | 69.7M | 74.0M | +6.1% |
| issued instructions | 69.7M | 74.0M | +6.2% |
| shared-load instructions | 4.25M | 2.68M | -37.0% |
| shared-store instructions | 1.09M | 1.09M | 0.0% |
| shared wavefronts | 8.14M | 10.23M | +25.8% |
| global load requests | 2.83M | 2.83M | 0.0% |
| registers / thread | 40 | 40 | 0.0% |
| static shared memory / block | 6.2 KB | 6.2 KB | 0.0% |
| eligible warps / scheduler | 1.87 | 1.99 | +6.5% |
| long scoreboard | 6.02 | 5.14 | -14.7% |
| short scoreboard | 2.04 | 2.16 | +5.8% |
| barrier | 0.75 | 0.71 | -4.7% |
| MIO throttle | 0.86 | 1.05 | +21.5% |

Conclusion:

The experiment proved that the probability fanout can reduce shared-load instruction count: SASS
shared-load instructions dropped by about 37%. However, the required shuffle sequence and dynamic
per-lane shared load increased total instructions by about 6%, increased shared wavefronts, and
raised MIO throttle. Wall time regressed slightly, so the change was reverted.

This result narrows the direction: `shared_probs` traffic is real, but replacing shared loads with
warp shuffles in the current scalar PV loop is not enough. A useful next step would need to remove
the probability materialization/fanout at a higher level, or change the PV mapping so fewer lanes
need the same probability scalar, rather than adding shuffle work on top of the existing scalar
row loop.

## 20. Validation Experiment 1: Expanded 8-bit Code Payload

Purpose:

- Test whether the 4-bit exponent-code packing itself is a primary decode bottleneck.
- Temporarily expand each codec tile from low-byte plane + 4-bit code plane
  (`256 + 128 = 384B`) to low-byte plane + 8-bit code plane (`256 + 256 = 512B`).
- Keep CTA structure, q-group mapping, split-k, and safe-page assumptions unchanged.

Temporary change:

- C++/Python default payload code bits were set to 8 for the experiment.
- The CUDA payload load/store boundary was made compile-time generic for 4-bit and 8-bit code
  fields.
- After the experiment, the default was restored to 4-bit. The temporary generic helper changes
  were later removed after the pair-interleaved validation also failed.

Artifacts:

- `profiles/byte_v2_vs_raw_gqa_code8_16384_p64_gpu5_20260624.jsonl`
- `profiles/byte_v2_decode_microbench_ncu_gqa_main_code8_16384_p64_gpu5_20260624.jsonl`
- `profiles/byte_v2_ncu_gqa_main_code8_16384_p64_gpu5_20260624.ncu-rep`
- `profiles/byte_v2_ncu_gqa_main_code8_16384_p64_gpu5_20260624_raw.csv`
- `profiles/byte_v2_vs_raw_gqa_after_code8_revert_16384_p64_gpu5_20260624.jsonl`

Correctness:

- 8-bit temporary build:
  `CUDA_VISIBLE_DEVICES=5 .venv/bin/python -m pytest tests/v1/attention/test_byte_v2_layout.py -k gqa_packed_cuda_matches_raw_reference -q`
  passed: 2 passed, 110 deselected.
- Restored 4-bit build: same command passed again.

Layout size:

| layout | page size | codec payload / tile | code bytes / tile |
| --- | ---: | ---: | ---: |
| 4-bit baseline | 115328 B | 384 B | 128 B |
| 8-bit temporary | 131712 B | 512 B | 256 B |

CUDA-event result, batch=1, seq=16k, partition=64:

| run | ByteV2 | raw FA2 | result |
| --- | ---: | ---: | --- |
| 4-bit baseline after P-broadcast revert | 0.2427 ms | 0.1362 ms | baseline |
| 8-bit expanded code | 0.2703 ms | 0.1362 ms | -11.4% |
| restored 4-bit after revert | 0.2437 ms | 0.1362 ms | back to baseline range |

NCU main-kernel counters, 8-bit vs section 13 baseline:

| metric | 4-bit baseline | 8-bit expanded | delta |
| --- | ---: | ---: | ---: |
| NCU duration | 265.3 us | 281.9 us | +6.3% |
| DRAM throughput | 411.6 GB/s | 505.9 GB/s | +22.9% |
| DRAM read | 91.6 MB | 123.2 MB | +34.5% |
| DRAM write | 17.6 MB | 19.5 MB | +10.6% |
| executed instructions | 69.7M | 63.6M | -8.8% |
| global load requests | 2.83M | 2.83M | 0.0% |
| global load sectors | 4.91M | 4.89M | -0.4% |
| shared wavefronts | 8.14M | 8.13M | -0.1% |
| long scoreboard | 6.02 | 7.90 | +31.1% |
| short scoreboard | 2.04 | 2.06 | +0.8% |
| MIO throttle | 0.86 | 0.55 | -35.9% |

Conclusion:

Expanding the code field did remove some decode instructions, but the larger payload increased
DRAM read by about 34.5% and long-scoreboard stall by about 31.1%. Wall time regressed by about
11.4%, so this is not a viable direction for the main format.

The useful signal is that pure instruction removal at the nibble extraction level is not enough
when it increases the payload footprint. The next validation should keep the same 12-bit/element
footprint and change only the physical placement of low bytes and packed codes, e.g. a
pair-interleaved 3-byte layout, to test whether address/coalescing/locality can improve without
increasing HBM bytes.

## 21. Validation Experiment 2: 12-bit Pair-Interleaved Payload

Purpose:

- Test a warp-friendlier physical placement without increasing payload size.
- Keep each codec tile at `384B` and each element at `12 bits` total.
- Temporarily change the 4-bit payload from split planes
  `[low[256], packed_code[128]]` to pair-interleaved triples
  `[low0, low1, packed_code] * 128`.

Temporary change:

- The CUDA payload load/store helper for 4-bit code read and wrote 3-byte element pairs.
- Python reference decode was updated to read the same pair-interleaved format.
- Page size and metadata layout were unchanged.
- The change was reverted after measurement.

Artifacts:

- `profiles/byte_v2_vs_raw_gqa_pair_interleaved12_16384_p64_gpu5_20260624.jsonl`
- `profiles/byte_v2_decode_microbench_ncu_gqa_main_pair_interleaved12_16384_p64_gpu5_20260624.jsonl`
- `profiles/byte_v2_ncu_gqa_main_pair_interleaved12_16384_p64_gpu5_20260624.ncu-rep`
- `profiles/byte_v2_ncu_gqa_main_pair_interleaved12_16384_p64_gpu5_20260624_raw.csv`
- `profiles/byte_v2_vs_raw_gqa_after_pair_interleaved12_revert_16384_p64_gpu5_20260624.jsonl`
- `profiles/byte_v2_vs_raw_gqa_final_restored_after_format_experiments_16384_p64_gpu5_20260624.jsonl`

Correctness:

- Pair-interleaved temporary build:
  `CUDA_VISIBLE_DEVICES=5 .venv/bin/python -m pytest tests/v1/attention/test_byte_v2_layout.py -k gqa_packed_cuda_matches_raw_reference -q`
  passed: 2 passed, 110 deselected.
- Restored split-plane build: same command passed again.

CUDA-event result, batch=1, seq=16k, partition=64:

| run | ByteV2 | raw FA2 | result |
| --- | ---: | ---: | --- |
| split-plane 4-bit before this experiment | 0.2437 ms | 0.1362 ms | baseline range |
| 12-bit pair-interleaved | 0.2796 ms | 0.1362 ms | -14.7% |
| restored split-plane after revert | 0.2458 ms | 0.1362 ms | back to baseline range |
| final restored after removing temporary helpers | 0.2427 ms | 0.1362 ms | back to original baseline |

NCU main-kernel counters, pair-interleaved vs section 13 split-plane baseline:

| metric | split-plane | pair-interleaved | delta |
| --- | ---: | ---: | ---: |
| NCU duration | 265.3 us | 292.7 us | +10.3% |
| DRAM throughput | 411.6 GB/s | 395.6 GB/s | -3.9% |
| DRAM read | 91.6 MB | 96.1 MB | +4.9% |
| DRAM write | 17.6 MB | 19.7 MB | +11.9% |
| executed instructions | 69.7M | 71.7M | +2.9% |
| global load requests | 2.83M | 2.83M | 0.0% |
| global load sectors | 4.91M | 7.05M | +43.6% |
| shared wavefronts | 8.14M | 8.13M | -0.1% |
| long scoreboard | 6.02 | 8.18 | +35.8% |
| short scoreboard | 2.04 | 2.02 | -1.2% |
| MIO throttle | 0.86 | 0.66 | -24.1% |

Conclusion:

The same-size 3-byte pair layout is worse than the current split-plane layout. It keeps global
load request count unchanged, but the 3-byte stride/coalescing pattern increases global load
sectors by about 43.6% and long-scoreboard stall by about 35.8%. Wall time regresses by about
14.7%, so this physical placement should not be used.

Together with the 8-bit expanded-code result, the current data says that simple payload format
changes are not the next performance lever. The 4-bit split-plane layout is still the best of the
tested options. The next useful optimization should target how many scalar payload decodes are
performed and how PV consumes decoded V/probabilities, not just where the existing 12-bit fields are
placed.

## 22. PV Diagnostic: Skip-PV and Probability-Only PV

Purpose:

- Estimate how much of the current GQA4 main-kernel time is spent below the QK/softmax stage.
- Separate the approximate cost of `shared_probs` fanout/FMA from safe-page V payload decode.
- This was a diagnostic only. Both temporary modes produce incorrect output and were reverted.

Temporary modes:

- `skip-PV`: initialize `pv[4]` to zero and skip the entire PV logical-block loop.
- `prob-only PV`: keep the row/q-group PV loop and consume `shared_probs`, but do not decode V.

Artifacts:

- `profiles/byte_v2_vs_raw_gqa_diag_skip_pv_16384_p64_gpu5_20260625.jsonl`
- `profiles/byte_v2_vs_raw_gqa_diag_prob_only_pv_16384_p64_gpu5_20260625.jsonl`
- `profiles/byte_v2_vs_raw_gqa_after_pv_diag_restore_16384_p64_gpu5_20260625.jsonl`

Correctness after restoring normal code:

- `CUDA_VISIBLE_DEVICES=5 .venv/bin/python -m pytest tests/v1/attention/test_byte_v2_layout.py -k gqa_packed_cuda_matches_raw_reference -q`
  passed: 2 passed, 110 deselected.

CUDA-event result, batch=1, seq=16k, partition=64:

| run | ByteV2 | raw FA2 | note |
| --- | ---: | ---: | --- |
| skip-PV diagnostic | 0.1761 ms | 0.1362 ms | wrong output |
| prob-only PV diagnostic | 0.1935 ms | 0.1362 ms | wrong output |
| restored normal kernel | 0.2447 ms | 0.1372 ms | correctness restored |

Approximate decomposition, using restored normal as the reference:

| component | estimate |
| --- | ---: |
| QK + softmax + loop/reduce-side main work | 0.1761 ms |
| `shared_probs` fanout + scalar PV accumulation | 0.0174 ms |
| safe-page V payload decode/load inside PV | 0.0512 ms |
| total PV-side cost | 0.0686 ms |

Conclusion:

PV is a meaningful part of the remaining ByteV2 main-kernel cost, but the larger share is V
payload decode/load rather than just probability fanout. This matches earlier negative results:
warp-local probability broadcast reduced shared-load instructions but did not improve wall time,
while byte-level V metadata/pair sharing also failed.

The next useful experiment should therefore change the V decode/PV mapping structurally. A small
probability-only tweak is unlikely to reach the 5% retention bar unless it also avoids repeated
scalar V payload loads or converts PV into a more tiled `P x V` computation.

## 23. Current Format and Current Kernel Performance

Purpose:

- Re-measure the current checked-out ByteV2 format and kernel after the sparse outlier overlay
  writer changes.
- Use the production-like fast path that is currently the main optimized decode path:
  split-plane 12-bit payload, guarded no-outlier path, GQA4 packed decode, partition size 64.
- Also measure the generic outlier-capable split kernel to quantify the gap between the fast path
  and the fully general path.

Build and correctness:

- `cmake --build build-bytev2-stable --target _C_stable_libtorch -j 32`
  reported no rebuild needed.
- `cp build-bytev2-stable/_C_stable_libtorch.abi3.so vllm/_C_stable_libtorch.abi3.so`
- `CUDA_VISIBLE_DEVICES=5 .venv/bin/python -m pytest tests/v1/attention/test_byte_v2_layout.py -k gqa_packed_cuda_matches_raw_reference -q`
  passed: 2 passed, 110 deselected.

Artifacts:

- `profiles/byte_v2_current_format_kernel_sweep_gqa_p64_gpu5_20260625.jsonl`
- `profiles/byte_v2_current_format_kernel_partition_sweep_16384_gpu5_20260625.jsonl`
- `profiles/byte_v2_current_format_generic_nooutlier_inputs_p64_gpu5_20260625.jsonl`
- `profiles/byte_v2_current_format_generic_default_inputs_p64_gpu5_20260625.jsonl`
- `profiles/byte_v2_ncu_gqa_main_current_format_kernel_16384_p64_gpu5_20260625.ncu-rep`
- `profiles/byte_v2_ncu_gqa_main_current_format_kernel_16384_p64_gpu5_20260625_raw.csv`
- `profiles/raw_fa2_ncu_current_format_compare_16384_p64_gpu5_20260625.ncu-rep`
- `profiles/raw_fa2_ncu_current_format_compare_16384_p64_gpu5_20260625_raw.csv`

CUDA-event context-length sweep, batch=1, p64, bn64, no-outlier GQA4 packed fast path:

| seq len | ByteV2 | raw FA2 | ByteV2 / raw |
| ---: | ---: | ---: | ---: |
| 256 | 0.0369 ms | 0.0635 ms | 0.58x |
| 512 | 0.0369 ms | 0.0614 ms | 0.60x |
| 1024 | 0.0471 ms | 0.0625 ms | 0.75x |
| 2048 | 0.0614 ms | 0.0645 ms | 0.95x |
| 4096 | 0.0809 ms | 0.0645 ms | 1.25x |
| 8192 | 0.1464 ms | 0.0809 ms | 1.81x |
| 16384 | 0.2427 ms | 0.1362 ms | 1.78x |

Partition sweep at seq=16k, no-outlier GQA4 packed fast path:

| partition | ByteV2 | workspace |
| ---: | ---: | ---: |
| 32 | 0.2601 ms | 8.06 MiB |
| 64 | 0.2458 ms | 4.03 MiB |
| 128 | 0.2765 ms | 2.02 MiB |
| 256 | 0.2785 ms | 1.01 MiB |

Generic outlier-capable split kernel, p64/bn64:

| input | seq len | ByteV2 generic | raw FA2 | ByteV2 / raw |
| --- | ---: | ---: | ---: | ---: |
| no-outlier inputs | 4096 | 0.2038 ms | 0.0891 ms | 2.29x |
| no-outlier inputs | 16384 | 0.6902 ms | 0.1362 ms | 5.07x |
| default inputs | 4096 | 0.2683 ms | 0.0645 ms | 4.16x |
| default inputs | 16384 | 0.9196 ms | 0.1362 ms | 6.75x |

NCU counters, seq=16k, p64 fast path vs raw FA2 `flash_fwd_splitkv_kernel`:

| metric | ByteV2 | raw FA2 | ByteV2 / raw |
| --- | ---: | ---: | ---: |
| NCU duration | 266.6 us | 127.9 us | 2.08x |
| DRAM read | 91.6 MB | 75.6 MB | 1.21x |
| DRAM write | 20.0 MB | 2.5 MB | 7.85x |
| DRAM throughput | 418.7 GB/s | 611.3 GB/s | 0.69x |
| executed instructions | 69.7M | 4.4M | 15.69x |
| global load requests | 2.83M | 0.14M | 20.08x |
| global load sectors | 4.91M | 2.11M | 2.32x |
| shared-memory wavefronts | 8.14M | 2.40M | 3.40x |
| long scoreboard stall | 6.05 | 4.04 | 1.50x |
| short scoreboard stall | 2.05 | 0.19 | 10.96x |
| MIO throttle stall | 0.86 | 0.25 | 3.45x |
| registers / thread | 40 | 252 | 0.16x |
| shared memory / block | 7.2 KB | 82.9 KB | 0.09x |
| launched threads | 262144 | 19456 | 13.47x |

Conclusion:

The current optimized fast path is correct and useful at short context lengths. It is faster than
raw FA2 up to 1k tokens and roughly tied at 2k. The crossover is between 2k and 4k; at 16k it is
about 1.78x slower than raw FA2 by CUDA-event timing.

Partition size 64 remains the best measured setting for 16k in this kernel. Smaller partitions
increase reduction/workspace overhead; larger partitions reduce workspace but slow the main kernel.

The generic outlier-capable path is not competitive yet. Even on no-outlier inputs it is about
5.1x slower than raw FA2 at 16k, and default inputs are about 6.8x slower. Current high-performance
decode therefore depends on routing safe pages through the no-outlier GQA4 packed fast path.

The main gap to raw FA2 is instruction/control count rather than only compressed-format bandwidth.
ByteV2 reads about 21% more DRAM at 16k, but executes about 15.7x more instructions and launches
13.5x more threads. The next optimization should reduce scalar per-element decode/control work and
PV path load fanout, or change the CTA/PV mapping to consume decoded V in a more FA2-like tiled
structure.

## 24. Instruction-Reduction Experiment: K Pair Staging and Fixed-Dim V Decode

Purpose:

- Try small, isolated instruction-count reductions in the current GQA4 no-outlier fast path.
- Keep the format unchanged: split-plane 12-bit payload and sparse outlier overlay.
- Retain only changes that improve CUDA-event timing by about 5% or more.

Final retained code changes:

- Add an aligned `uint16_t` payload load helper for known-aligned low-byte pairs.
- Decode K staging in adjacent dim pairs for safe pages.
- Hoist each staging thread's fixed K dim tile, tile offset, dim-in-tile, and base byte out of the
  per-row pair loop.
- Use a fixed-dim safe V decode helper in the no-outlier V path, avoiding the generic descriptor
  helper's fallback/outlier checks and extra parameters.

Artifacts:

- `profiles/byte_v2_exp_pair_k_staging_16384_p64_gpu5_20260625.jsonl`
- `profiles/byte_v2_exp_pair_k_staging_sweep_repeat_gpu5_20260625.jsonl`
- `profiles/byte_v2_ncu_exp_pair_k_staging_16384_p64_gpu5_20260625_raw.csv`
- `profiles/byte_v2_exp_pair_k_plus_v_fixed_dim_sweep_gpu5_20260625.jsonl`
- `profiles/byte_v2_ncu_exp_pair_k_plus_v_fixed_dim_16384_p64_gpu5_20260625_raw.csv`
- `profiles/byte_v2_exp_pair_k_aligned_low_plus_v_fixed_dim_sweep_gpu5_20260625.jsonl`
- `profiles/byte_v2_ncu_exp_pair_k_aligned_low_plus_v_fixed_dim_16384_p64_gpu5_20260625_raw.csv`
- `profiles/byte_v2_exp_fixed_tile_k_pair_plus_v_fixed_dim_sweep_gpu5_20260625.jsonl`
- `profiles/byte_v2_ncu_exp_fixed_tile_k_pair_plus_v_fixed_dim_16384_p64_gpu5_20260625_raw.csv`
- `profiles/byte_v2_final_fixed_tile_decode_sanity_16384_p64_gpu5_20260625.jsonl`

Correctness:

- `CUDA_VISIBLE_DEVICES=5 .venv/bin/python -m pytest tests/v1/attention/test_byte_v2_layout.py -k "reshape_and_cache_cuda_writes_outlier_overlay or reshape_and_cache_cuda_full_tile_overlay_avoids_raw_fallback or raw_staging_commit_matches_direct_cache or raw_staging_incremental_partial_block_matches_direct_cache or single_token_cache_update_cuda_matches_direct_cache or gqa_packed_cuda_matches_raw_reference" -q`
  passed: 10 passed, 102 deselected.

CUDA-event result, p64/bn64, no-outlier GQA4 packed fast path:

| seq len | baseline | final | delta |
| ---: | ---: | ---: | ---: |
| 4096 | 0.0809 ms | 0.0768 ms | -5.1% |
| 8192 | 0.1464 ms | 0.1393 ms | -4.9% |
| 16384 | 0.2427 ms | 0.2294 ms | -5.5% |

Incremental 16k results:

| experiment | ByteV2 16k | note |
| --- | ---: | --- |
| baseline current format/kernel | 0.2427 ms | section 23 baseline |
| K pair staging only | 0.2340 ms | -3.6%; not enough alone |
| K pair + V fixed-dim | 0.2324 ms | -4.2%; still below retention line |
| K pair + aligned low load + V fixed-dim | 0.2314 ms | -4.6%; still marginal |
| fixed-tile K pair + aligned low load + V fixed-dim | 0.2294 ms | -5.5%; retained |

NCU counters, seq=16k, final retained version vs section 23 baseline:

| metric | baseline | final | delta |
| --- | ---: | ---: | ---: |
| NCU duration | 266.6 us | 243.3 us | -8.8% |
| DRAM read | 91.6 MB | 89.5 MB | -2.3% |
| DRAM write | 20.0 MB | 19.7 MB | -1.5% |
| DRAM throughput | 418.7 GB/s | 448.9 GB/s | +7.2% |
| executed instructions | 69.7M | 51.7M | -25.8% |
| global load requests | 2.83M | 1.81M | -35.9% |
| global load sectors | 4.91M | 4.43M | -9.8% |
| shared-memory wavefronts | 8.14M | 7.87M | -3.3% |
| long scoreboard stall | 6.05 | 8.91 | +47.2% |
| short scoreboard stall | 2.05 | 3.19 | +55.6% |
| MIO throttle stall | 0.86 | 1.50 | +74.4% |
| registers / thread | 40 | 40 | 0.0% |

Conclusion:

The retained changes reduce the main-kernel instruction count by about 25.8% and improve 16k
CUDA-event timing by about 5.5%. This confirms that a meaningful part of the ByteV2 gap is still
avoidable scalar decode/control work inside the current CTA structure.

The stall ratios rise after the instruction-count reduction. That does not mean the change is bad:
absolute time still improves, but the remaining instructions are now more dominated by memory
dependency and PV/shared-prob consumption. The next optimization should therefore target the
remaining V/PV path structure or reduce the number of split-k CTAs/workspace writes; another small
decode helper tweak is less likely to produce the next large gain.

## 25. Split-K Retune and V Pair Shuffle Experiment

Purpose:

- After the section 24 instruction reduction, re-check whether the best split-k partition size
  changed.
- Try one PV-side structure experiment: adjacent V dims share one safe payload decode, with even
  lanes decoding a pair and odd lanes receiving the second value through warp shuffle.

Artifacts:

- `profiles/byte_v2_post_instr_partition_sweep_8192_16384_gpu5_20260625.jsonl`
- `profiles/byte_v2_exp_v_pair_shuffle_sweep_gpu5_20260625.jsonl`
- `profiles/byte_v2_ncu_exp_v_pair_shuffle_16384_p64_gpu5_20260625_raw.csv`
- `profiles/byte_v2_after_v_pair_revert_sanity_16384_p64_gpu5_20260625.jsonl`

Partition sweep after section 24 retained changes:

| seq len | p32 | p64 | p128 | p256 | best |
| ---: | ---: | ---: | ---: | ---: | --- |
| 8192 | 0.1382 ms | 0.1393 ms | 0.1393 ms | 0.1966 ms | p32/p64/p128 tied |
| 16384 | 0.2468 ms | 0.2304 ms | 0.2652 ms | 0.2580 ms | p64 |

V pair shuffle experiment:

| seq len | retained baseline | V pair shuffle | result |
| ---: | ---: | ---: | --- |
| 4096 | 0.0768 ms | 0.0788 ms | -2.7% regression |
| 8192 | 0.1393 ms | 0.1444 ms | -3.7% regression |
| 16384 | 0.2294 ms | 0.2365 ms | -3.1% regression |

NCU counters, seq=16k, V pair shuffle vs section 24 retained baseline:

| metric | retained baseline | V pair shuffle | delta |
| --- | ---: | ---: | ---: |
| NCU duration | 243.3 us | 256.5 us | +5.5% |
| DRAM read | 89.5 MB | 89.8 MB | +0.3% |
| DRAM write | 19.7 MB | 19.3 MB | -2.1% |
| DRAM throughput | 448.9 GB/s | 425.0 GB/s | -5.3% |
| executed instructions | 51.7M | 55.3M | +6.9% |
| global load requests | 1.81M | 1.81M | 0.0% |
| global load sectors | 4.43M | 4.43M | 0.0% |
| shared-memory wavefronts | 7.87M | 8.39M | +6.6% |

Conclusion:

Partition size 64 is still the best 16k setting after the instruction reduction. Larger partitions
reduce workspace but slow the main kernel more than they save in reduction/workspace overhead.

The V pair shuffle experiment was reverted. It looked attractive because adjacent dims share the
same packed 4-bit code byte, but in the current PV mapping it does not reduce global load requests
and adds shuffle/control overhead. It also increases instruction count and shared-memory pressure,
so this is not the right way to optimize PV.

The next useful PV-side attempt should avoid per-row `shared_probs` fanout or change the CTA/PV
mapping more structurally. Pair-sharing within the current one-thread-per-output-dim mapping is too
small and adds dependency overhead.

## 26. FA2-Style Split Count / CTA Reduction Attempt

Purpose:

- Compare ByteV2's fixed partition-per-CTA split-k scheme with FA2's `num_splits`-based splitkv
  launcher.
- Try reducing ByteV2 main CTA count with larger partitions and `ComputeBlockN=128`.
- Keep only changes that reduce wall time; lower CTA count alone is not enough if latency regresses.

FA2 reference:

- FA2 splitkv uses `num_splits`, not a fixed 64-token partition.
- The FA2 launcher grid for the same raw 16k benchmark was `1 x 19 x 8 = 152` CTAs.
- ByteV2 p64 uses `8 x 1 x 256 = 2048` main CTAs at 16k.
- FA2 then runs a combine kernel specialized by `num_splits`; ByteV2 uses a generic warp reduce
  over `tmp_out/exp_sums`.

Artifacts:

- `profiles/byte_v2_fa2_style_large_partition_sweep_gpu5_20260625.jsonl`
- `profiles/byte_v2_fa2_style_partition_bn128_sweep_gpu5_20260625.jsonl`
- `profiles/byte_v2_ncu_fa2_style_p64_bn128_16384_gpu5_20260625_raw.csv`
- `profiles/byte_v2_ncu_fa2_style_p256_bn128_16384_gpu5_20260625_raw.csv`

Large partition sweep, no-outlier GQA4 packed fast path:

| seq len | p64 bn64 | p128 bn64 | p256 bn64 | p512 bn64 | p1024 bn64 | p2048 bn64 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8192 | 0.1393 ms | 0.1382 ms | 0.1966 ms | 0.3308 ms | 0.6287 ms | 1.2339 ms |
| 16384 | 0.2304 ms | 0.2652 ms | 0.2570 ms | 0.3727 ms | 0.6482 ms | 1.2411 ms |

`ComputeBlockN=128` sweep:

| seq len | p64 bn128 | p128 bn128 | p256 bn128 | p512 bn128 |
| ---: | ---: | ---: | ---: | ---: |
| 8192 | 0.1393 ms | 0.1362 ms | 0.1946 ms | 0.3308 ms |
| 16384 | 0.2294 ms | 0.2529 ms | 0.2458 ms | 0.3666 ms |

NCU counters, seq=16k, retained p64/bn64 vs lower-CTA alternatives:

| metric | p64 bn64 retained | p64 bn128 | p256 bn128 |
| --- | ---: | ---: | ---: |
| main CTA grid size | 2048 | 2048 | 512 |
| launched threads | 262144 | 262144 | 65536 |
| NCU duration | 243.3 us | 245.3 us | 287.4 us |
| executed instructions | 51.7M | 51.7M | 49.9M |
| global load requests | 1.81M | 1.81M | 1.78M |
| global load sectors | 4.43M | 4.44M | 4.40M |
| DRAM read | 89.5 MB | 89.4 MB | 89.5 MB |
| DRAM write | 19.7 MB | 19.2 MB | 8.4 MB |

Conclusion:

Lowering CTA count with larger ByteV2 partitions does not improve performance in the current kernel.
`p256_bn128` cuts main CTAs and launched threads by 75% and reduces workspace writes, but it only
reduces executed instructions by about 3.5% and regresses NCU duration by about 18%. The reason is
that most instructions are real per-token decode/QK/PV work; larger partitions serialize more of
that work inside fewer CTAs.

`ComputeBlockN=128` is slightly useful at 8k with p128, but it is not a clear win at 16k. The current
16k best setting remains p64. FA2's smaller CTA count is not directly transferable unless ByteV2
also changes the CTA's internal tiled QK/PV work distribution. A simple partition-size or
`ComputeBlockN` retune cannot reproduce FA2's low instruction count.

## 27. FA2-Style CTA Internal V Staging Attempt

Purpose:

- Test whether one visible piece of FA2's CTA-internal tiled PV structure, staging V into shared
  memory before the PV step, reduces ByteV2 instruction count or scoreboard stalls.
- Keep the existing ByteV2 scalar PV mapping unchanged so this is a single-point experiment.

FA2 reference:

- In the SM80 FA2 forward path, softmax probabilities remain in register fragments (`rP`), are
  converted to `tOrP`, and feed `gemm_rs(acc_o, tOrP, tOrVt, ...)`.
- In the Hopper path, the same idea appears as tiled `tOrP/tOrV` fragments passed to
  `flash::gemm`.
- Therefore FA2's V shared-memory staging is coupled to tiled MMA PV. It is not just an isolated
  cache of scalar V values.

Temporary ByteV2 change:

- Added `shared_v_tile[ComputeBlockN][HeadDimV]` to the GQA4 no-outlier main kernel.
- After softmax probability materialization, decoded every safe-page V element in the current
  compute tile into `shared_v_tile`.
- The PV loop then loaded `V(row, dim)` from shared memory instead of directly decoding the fixed
  V dim from the ByteV2 page payload.
- Unsafe pages still used the original direct generic decode path.

Artifacts:

- `profiles/byte_v2_exp_fa2_style_v_staging_current_sweep_gpu5_20260625.jsonl`
- `profiles/byte_v2_ncu_exp_fa2_style_v_staging_current_16384_p64_gpu5_20260625_raw.csv`
- `profiles/byte_v2_after_fa2_style_v_staging_revert_sanity_16384_p64_gpu5_20260625.jsonl`

Correctness:

- `CUDA_VISIBLE_DEVICES=5 .venv/bin/python -m pytest tests/v1/attention/test_byte_v2_layout.py -k gqa_packed_cuda_matches_raw_reference -q`
  passed: 2 passed, 110 deselected.

CUDA-event result, no-outlier GQA4 packed fast path:

| seq len | retained baseline | V staging experiment | raw FA2 |
| ---: | ---: | ---: | ---: |
| 4096 | 0.0768 ms | 0.1208 ms | 0.0645 ms |
| 8192 | 0.1393 ms | 0.2161 ms | 0.0799 ms |
| 16384 | 0.2294 ms | 0.3809 ms | 0.1362 ms |

Revert sanity:

- After removing the temporary V staging code and rebuilding, 16k p64/bn64 returned to
  0.2284 ms. This confirms the negative experiment was not kept in the active binary.

NCU main-kernel counters, seq=16k, V staging vs section 24 retained baseline:

| metric | retained baseline | V staging | delta |
| --- | ---: | ---: | ---: |
| NCU duration | 243.3 us | 443.9 us | +82.5% |
| executed instructions | 51.7M | 75.0M | +44.9% |
| global load requests | 1.81M | 2.37M | +30.8% |
| global load sectors | 4.43M | 5.02M | +13.3% |
| shared-memory wavefronts | 7.87M | 8.93M | +13.4% |
| DRAM read | 89.5 MB | 89.2 MB | -0.4% |
| DRAM write | 19.7 MB | 16.0 MB | -18.6% |
| shared memory / block | 7.2 KB | 23.6 KB | +227.1% |

Conclusion:

This FA2-style fragment is a clear regression in ByteV2's current scalar PV structure. It reduces
some dependency stalls, but it pays for that by adding a full-tile V decode pass, extra shared
stores/loads, another barrier, and much larger static shared memory per block. The result is higher
instruction count and much worse wall time.

The useful lesson from FA2 is not "stage V in shared memory" by itself. The useful lesson is the
combined tiled dataflow: QK and PV are both expressed as fragment-level matrix products, softmax
probabilities are consumed from register/MMA-friendly layouts, and V staging feeds tensor-core PV.
For ByteV2, reducing instruction count in the same style would require one of these deeper changes:

1. Decode compressed K/V into a tile layout that can feed MMA PV, then replace scalar `P * V`
   accumulation with a tiled PV kernel.
2. Keep scalar V decode but change probability consumption so `shared_probs` is not fanned out to
   every output-dim thread for every row.
3. Specialize a larger GQA/CTA tile where decoded V is reused across q groups without materializing
   the entire `ComputeBlockN x HeadDimV` tile as plain BF16 shared memory.

No code from this V staging experiment was retained. The active kernel remains the section 24
version: fixed-tile K pair staging, aligned payload low loads, and fixed-dim safe V decode.

## 28. FA2-Like GQA4 Warp-Local PV Prototype

Purpose:

- Test the first small step toward FA2-like CTA-internal work distribution without introducing
  MMA yet.
- Keep QK and softmax mostly unchanged.
- Change PV so each warp owns one GQA q-group, lane 0 reads that q-group's probability, broadcasts
  it to the warp, and the warp computes all 128 output dims for that q-group.
- This removes most `shared_probs` fanout, but intentionally repeats V decode across q groups.

Temporary change:

- Added a tenth `tile_policy` flag and a `--gqa-tiled-pv` microbench option.
- Added a separate GQA4 no-outlier split-k kernel prototype.
- The prototype kept the same split-k workspace/reduce format as the retained kernel.
- The experiment was reverted after measurement because it did not meet the retention threshold.

Artifacts:

- `profiles/byte_v2_tiled_pv_baseline_sweep_gpu5_20260625.jsonl`
- `profiles/byte_v2_exp_gqa_tiled_pv_sweep_gpu5_20260625.jsonl`
- `profiles/byte_v2_ncu_exp_gqa_tiled_pv_16384_p64_gpu5_20260625.jsonl`
- `profiles/byte_v2_ncu_exp_gqa_tiled_pv_16384_p64_gpu5_20260625_raw.csv`
- `profiles/byte_v2_after_gqa_tiled_pv_revert_sanity_16384_p64_gpu5_20260625.jsonl`

Correctness:

- With the temporary prototype present,
  `CUDA_VISIBLE_DEVICES=5 .venv/bin/python -m pytest tests/v1/attention/test_byte_v2_layout.py -k "gqa_packed_cuda_matches_raw_reference or split_k_guarded_cuda_matches_raw_reference" -q`
  passed: 7 passed, 108 deselected.
- After reverting the temporary prototype, the same command passed: 4 passed, 108 deselected.

CUDA-event result, p64/bn64, no-outlier GQA4 guarded split-k:

| seq len | retained baseline | warp-local PV prototype | result |
| ---: | ---: | ---: | --- |
| 4096 | 0.0758 ms | 0.0901 ms | +18.9% slower |
| 8192 | 0.1393 ms | 0.1587 ms | +14.0% slower |
| 16384 | 0.2294 ms | 0.2847 ms | +24.1% slower |

Revert sanity:

- After removing the temporary prototype and rebuilding, 16k p64/bn64 returned to 0.2284 ms.

NCU main-kernel counters, seq=16k, prototype vs section 24 retained baseline:

| metric | retained baseline | warp-local PV prototype | delta |
| --- | ---: | ---: | ---: |
| NCU duration | 243.3 us | 329.6 us | +35.5% |
| executed instructions | 51.7M | 79.3M | +53.2% |
| global load requests | 1.81M | 5.05M | +179.2% |
| global load sectors | 4.43M | 10.85M | +145.0% |
| shared-memory wavefronts | 7.87M | 6.73M | -14.4% |
| DRAM read | 89.5 MB | 89.7 MB | +0.2% |
| DRAM write | 19.7 MB | 6.2 MB | -68.4% |
| registers / thread | 40 | 63 | +57.5% |
| shared memory / block | 7.2 KB | 6.1 KB | -14.9% |

Conclusion:

The prototype confirms that `shared_probs` fanout is real but not the dominant standalone problem.
Reducing probability fanout without reusing V across q groups makes the kernel much worse: V decode
and global-load requests increase by roughly 2.8x, register pressure rises from 40 to 63 registers
per thread, and executed instructions rise by about 53%.

This rules out a simple "one warp per q-group computes all dims" PV rewrite. The next FA2-like
attempt must preserve V reuse while changing probability consumption. In practice that means one of
two directions:

1. Decode a compact V tile once per CTA and feed several q groups from it, but only if the PV step
   is changed into a real tiled computation so the extra shared-memory traffic pays for itself.
2. Keep the current one-thread-per-output-dim V reuse, but reduce `shared_probs` traffic with a
   smaller broadcast/cache structure that does not duplicate V decode.

No code from this prototype was retained. The active kernel remains the section 24 retained version.

## 29. Decode-Once Shared-V + Tiled-PV Prototype

Purpose:

- Test the combined version of the two previous negative experiments.
- Decode each safe V tile once per CTA into shared memory.
- Consume that shared V tile with warp-local GQA PV: each warp owns one q group, broadcasts that
  q group's probability, and computes all 128 output dims for the q group.
- This is closer to FA2's dataflow than either isolated experiment: V is reused across q groups,
  and probability fanout is reduced. It still does not use tensor-core MMA.

FA2 reference:

- FA2's SM80 path stages V and feeds `gemm_rs(acc_o, tOrP, tOrVt, ...)`.
- FA2's Hopper path similarly passes `tOrP/tOrV` fragments into `flash::gemm`.
- The ByteV2 prototype copied the "decode/stage V once, then tiled PV consume it" shape, but kept
  scalar CUDA-core FMA instead of MMA fragments.

Temporary change:

- Added a tenth `tile_policy` flag and a `--gqa-decode-once-tiled-pv` microbench option.
- Added a separate GQA4 no-outlier split-k kernel prototype.
- The prototype staged safe-page `ComputeBlockN x HeadDimV` V bits into shared memory.
- The PV loop used one warp per q group and four output dims per lane.
- The experiment was reverted after measurement because it regressed clearly.

Artifacts:

- `profiles/byte_v2_decode_once_tiled_pv_baseline_sweep_gpu5_20260625.jsonl`
- `profiles/byte_v2_exp_decode_once_tiled_pv_sweep_gpu5_20260625.jsonl`
- `profiles/byte_v2_ncu_exp_decode_once_tiled_pv_16384_p64_gpu5_20260625.jsonl`
- `profiles/byte_v2_ncu_exp_decode_once_tiled_pv_16384_p64_gpu5_20260625_raw.csv`
- `profiles/byte_v2_after_decode_once_tiled_pv_revert_sanity_16384_p64_gpu5_20260625.jsonl`

Correctness:

- With the temporary prototype present,
  `CUDA_VISIBLE_DEVICES=5 .venv/bin/python -m pytest tests/v1/attention/test_byte_v2_layout.py -k "gqa_packed_cuda_matches_raw_reference or split_k_guarded_cuda_matches_raw_reference" -q`
  passed: 7 passed, 108 deselected.
- After reverting the temporary prototype, the same command passed: 4 passed, 108 deselected.

CUDA-event result, p64/bn64, no-outlier GQA4 guarded split-k:

| seq len | retained baseline | decode-once tiled-PV prototype | result |
| ---: | ---: | ---: | --- |
| 4096 | 0.0799 ms | 0.1229 ms | +53.8% slower |
| 8192 | 0.1413 ms | 0.2171 ms | +53.6% slower |
| 16384 | 0.2314 ms | 0.3871 ms | +67.3% slower |

Revert sanity:

- After removing the temporary prototype and rebuilding, 16k p64/bn64 returned to 0.2314 ms.

NCU main-kernel counters, seq=16k, prototype vs section 24 retained baseline:

| metric | retained baseline | decode-once tiled-PV | delta |
| --- | ---: | ---: | ---: |
| NCU duration | 243.3 us | 447.2 us | +83.8% |
| executed instructions | 51.7M | 75.0M | +45.0% |
| global load requests | 1.81M | 2.37M | +30.8% |
| global load sectors | 4.43M | 5.02M | +13.3% |
| shared-memory wavefronts | 7.87M | 9.37M | +19.1% |
| DRAM read | 89.5 MB | 88.9 MB | -0.7% |
| DRAM write | 19.7 MB | 7.5 MB | -61.7% |
| registers / thread | 40 | 47 | +17.5% |
| shared memory / block | 7.2 KB | 22.5 KB | +212.2% |
| shared-memory occupancy limit | 12 blocks | 4 blocks | -66.7% |

Conclusion:

The combined decode-once + tiled-PV prototype still regresses badly. It fixes the V reuse problem of
the warp-local PV experiment, but it reintroduces the full shared-V tile cost from the V-staging
experiment. The extra V staging pass, larger shared memory footprint, lower occupancy, and scalar
PV instruction count overwhelm the saved probability fanout and workspace writes.

This is the strongest evidence so far that ByteV2 cannot approach FA2 just by rearranging scalar
CUDA-core PV around shared memory. FA2's advantage comes from the full fragment/MMA path: `P` is
kept in a tensor-core-friendly layout, `V` is staged in the matching layout, and `P x V` is a tiled
matrix multiply. ByteV2 needs either:

1. a true MMA PV prototype that converts probabilities and decoded V into tensor-core fragments, or
2. a different compression/cache format that can decode directly into an MMA-friendly shared layout
   with much lower staging overhead.

No code from this prototype was retained. The active kernel remains the section 24 retained version.

## 30. PV-Only WMMA Prototype, Reverted

Question:

Can ByteV2 keep the current GQA4 QK/softmax path but replace scalar PV with a
small FA2-like tensor-core PV path?

Temporary prototype:

- Added a tenth `tile_policy` flag and a `--gqa-packed-mma-pv` microbench
  option.
- Restricted the experiment to p64/bn64:
  `partition_size == compute_block_n == 64`.
- Kept the current GQA4 QK and online softmax code unchanged.
- Converted `shared_probs[4][64]` into a padded BF16 `P[16][64]` shared tile.
- Decoded V into shared BF16 `V[64][128]`.
- Used warp-level WMMA `m16n16k16` to compute `P x V`, with rows 0..3 valid.

Correctness:

- Existing scalar GQA4 split-k test still passed after adding the temporary
  code:
  `.venv/bin/python -m pytest tests/v1/attention/test_byte_v2_layout.py::test_byte_v2_paged_decode_attention_split_k_gqa_packed_cuda_matches_raw_reference -q`
  passed: 2 passed.
- A direct p64 check with explicit
  `tile_policy=(16, 16, 16, 64, 128, 128, 0, 1, 1, 1)` showed:
    - scalar GQA4 max abs vs raw reference: `4.88e-4`
    - WMMA PV max abs vs raw reference: `5.62e-4`
    - WMMA PV max abs vs scalar GQA4: `9.77e-4`

CUDA-event result, p64/bn64, no-outlier GQA4 split-k:

| seq len | retained scalar GQA4 | PV-only WMMA prototype | result |
| ---: | ---: | ---: | --- |
| 4096 | 0.0758 ms | 0.1505 ms | +98.6% slower |
| 8192 | 0.1434 ms | 0.2509 ms | +75.0% slower |
| 16384 | 0.2294 ms | 0.4516 ms | +96.9% slower |

Artifacts:

- `profiles/byte_v2_decode_microbench_gqa4_scalar_p64_bn64_20260625.jsonl`
- `profiles/byte_v2_decode_microbench_gqa4_mma_pv_p64_bn64_20260625.jsonl`
- `profiles/byte_v2_after_mma_pv_revert_sanity_16384_p64_20260625.jsonl`

Conclusion:

The prototype is numerically usable but performance-negative. It is worse than
the retained scalar kernel because it stages a full `64 x 128` V tile through
shared memory, writes/reads padded P, burns MMA work on 12 invalid M rows, and
still carries ByteV2 decode overhead. This is not the same structure as FA2's
CUTE mainloop, where P and V are already in fragment-friendly layouts and the M
tile is useful.

No code from this replacement-style prototype was retained in the main kernel.
The active default kernel remains the section 24 retained version. Revert
sanity at 16k p64/bn64 measured `0.2294 ms`.

Follow-up:

The same coarse PV-only WMMA structure was later reintroduced as a separate
non-default experiment kernel:
`byte_v2_paged_decode_attention_split_k_gqa4_fa2_like_no_fallback_no_outlier_kernel`.
It is selected only by the tenth `tile_policy` flag or by
`BYTE_V2_DECODE_GQA_FA2_LIKE=1`, so the retained scalar GQA4 kernel remains
unchanged.

Sanity artifact:

- `profiles/byte_v2_gqa4_fa2_like_sanity_4096_p64_20260625.jsonl`

Sanity result:

- 4096 p64/bn64 FA2-like experiment median: `0.1536 ms`

## 31. FA2-Like PV N-Tile Staging Update

Question:

Can the dedicated FA2-like experiment kernel avoid full row-major
`V[64,128]` shared staging and instead stage/consume one 16-column V tile at a
time?

Change:

- Only the dedicated FA2-like experiment kernel was modified.
- `shared_v_mma` changed from `V[64][128]` to `V[4][64][16]`.
- Each warp now stages the current `V[64][16]` N tile and immediately feeds it
  to WMMA.
- The default retained scalar GQA4 kernel was not modified.

Correctness:

- Existing default GQA4 split-k test passed:
  `.venv/bin/python -m pytest tests/v1/attention/test_byte_v2_layout.py::test_byte_v2_paged_decode_attention_split_k_gqa_packed_cuda_matches_raw_reference -q`
  passed: 2 passed.
- Direct p64 check:
    - scalar GQA4 max abs vs raw reference: `4.88e-4`
    - FA2-like N-tile max abs vs raw reference: `5.62e-4`
    - FA2-like N-tile max abs vs scalar GQA4: `9.77e-4`

CUDA-event sanity result:

| seq len | FA2-like full-V shared staging | FA2-like N-tile staging |
| ---: | ---: | ---: |
| 4096 | 0.1536 ms | 0.1347 ms |

Artifact:

- `profiles/byte_v2_gqa4_fa2_like_ntile_sanity_4096_p64_20260625.jsonl`

Conclusion:

This moves the experiment kernel closer to FA2's tiled PV consumption: V is no
longer decoded into a full `64 x 128` shared tile before PV. It is still not
FA2-equivalent, because QK remains scalar and V is decoded into a simple WMMA
row-major tile rather than a CUTE/FA2 fragment layout. The next useful step is
to make the per-N tile loader decode directly into a fragment-friendly layout
or replace the WMMA wrapper with CUTE-style tiled MMA.

## 32. P-Direct Staging Experiment Not Retained

Question:

Is the `shared_probs -> shared_p_mma` materialization loop a meaningful part of
the current FA2-like experiment-kernel gap?

Experiment:

- Only the dedicated FA2-like experiment kernel was modified.
- Softmax wrote BF16 probabilities directly into `shared_p_mma`.
- The explicit `shared_probs[4][64]` write plus later
  `shared_p_mma[16][64]` materialization loop was removed.
- Padded rows were not treated as a retained optimization because only output
  rows 0..3 are used, and the experiment was evaluated purely as a local
  performance probe.

Correctness:

- Default retained GQA4 split-k test still passed after rebuilding the final
  retained code:
  `.venv/bin/python -m pytest tests/v1/attention/test_byte_v2_layout.py::test_byte_v2_paged_decode_attention_split_k_gqa_packed_cuda_matches_raw_reference -q`
  passed: 2 passed.
- P-direct probe:
    - scalar GQA4 max abs vs simple raw BF16 reference: `0.0`
    - FA2-like P-direct max abs vs simple raw BF16 reference: `9.77e-4`
    - FA2-like P-direct max abs vs scalar GQA4: `9.77e-4`

CUDA-event sanity result:

| seq len | Retained N-tile FA2-like | P-direct probe |
| ---: | ---: | ---: |
| 4096 | 0.134656 ms | 0.134144 ms |

Artifact:

- `profiles/byte_v2_gqa4_fa2_like_p_direct_sanity_4096_p64_20260625.jsonl`

Conclusion:

The observed change is only about `0.4%`, so this is not the main gap and was
not retained. The final code was rebuilt back to the previous N-tile PV staging
version. The next optimization should spend effort on the larger FA2
differences: V decode directly into a fragment-friendly tile, CUTE-style MMA
layout, or QK/PV work distribution.

## 33. FA2 Source Recheck and Revised Migration Order

Files checked:

- `/mnt/sdb/yxz/ByteV2/flash-attention/csrc/flash_attn/src/flash_fwd_kernel.h`
- `/mnt/sdb/yxz/ByteV2/flash-attention/csrc/flash_attn/src/kernel_traits.h`
- `/mnt/sdb/yxz/ByteV2/flash-attention/csrc/flash_attn/src/softmax.h`
- `/mnt/sdb/yxz/ByteV2/flash-attention/csrc/flash_attn/src/utils.h`

Confirmed FA2 behavior:

- QK is computed through CUTE tiled MMA into `acc_s`.
- Online softmax is applied directly over the accumulator layout and rescales
  `acc_o` when the running max changes.
- P is not materialized through shared memory. FA2 converts `acc_s` to a
  register P fragment and retile it with `convert_layout_acc_Aregs`.
- PV uses `gemm_rs`, where P is in registers and V is loaded from the
  CUTE-compatible shared-memory layout.
- V shared memory is not a simple row-major `V[block_n][head_dim]`; it uses
  `SmemLayoutKV` plus the transposed V view used by the MMA copy atom.

Implication for ByteV2:

The P-direct experiment in section 32 was expected to be low impact because it
only removed one materialization loop. It did not reproduce FA2's real P path.
The next retained-performance opportunity is to move the experiment kernel from
`nvcuda::wmma` plus row-major staging toward the CUTE dataflow.

Revised execution order:

1. Compile a minimal CUTE/TiledMMA PV skeleton inside the ByteV2 stable
   extension.
2. Replace the FA2-like experiment kernel's PV `nvcuda::wmma` wrapper with a
   CUTE-style tiled MMA path.
3. Make ByteV2 V decode write directly into the CUTE-compatible shared layout.
4. Replace shared P materialization with a register P fragment.
5. Move scalar QK to tiled MMA.
6. Re-sweep block sizes. FA2 hdim128 split-KV can use `kBlockN=128`, while the
   single-split standard-aligned path uses `kBlockN=64`, so ByteV2 p64/bn64 is
   only the current experiment point.

Direct line-for-line migration is not appropriate because ByteV2 decodes a
compressed KV cache and groups GQA4 rows in the current CTA shape. The useful
FA2 target is the dataflow and layout discipline, not the exact kernel body.

## 34. CUTE PV Skeleton Compile Step

Change:

- Added CUTLASS/CUTE includes to `byte_v2_ops.cu`.
- Added `ByteV2Fa2LikeCutePvTraits` with:
    - BF16 element type.
    - CUTE `TiledMMA`.
    - `SmemLayoutKV`.
    - `SmemLayoutVtransposed`.
    - `SmemLayoutVtransposedNoSwizzle`.
- Bound that traits type to the FA2-like experiment kernel using the current
  `Policy::ComputeBlockN` and `Policy::HeadDimV`.
- No runtime path was switched yet. The retained scalar GQA4 kernel and the
  current FA2-like experiment kernel math are unchanged.

Build architecture:

- Local GPU: NVIDIA A40, capability `8.6`.
- The previous build cache was `CMAKE_CUDA_ARCHITECTURES=75`, which is not
  suitable for enabling an actual SM80+ BF16 CUTE MMA path.
- Reconfigured `build-bytev2-stable` to `86` and clean-rebuilt
  `_C_stable_libtorch`.
- `build.ninja` now contains `-gencode arch=compute_86,code=sm_86`.

Verification:

- Build passed.
- Default GQA4 split-k test passed:
  `.venv/bin/python -m pytest tests/v1/attention/test_byte_v2_layout.py::test_byte_v2_paged_decode_attention_split_k_gqa_packed_cuda_matches_raw_reference -q`
  passed: 2 passed.

FA2-like sanity:

| seq len | route | median |
| ---: | --- | ---: |
| 4096 | CUTE skeleton, runtime still N-tile WMMA | 0.134656 ms |

Artifact:

- `profiles/byte_v2_gqa4_fa2_like_cute_probe_sm86_sanity_4096_p64_20260625.jsonl`

Conclusion:

This is a compile and environment-preparation step, not a performance step.
The important result is that ByteV2 can now compile FA2-like CUTE PV traits in
the stable extension under `sm_86`. The next performance-affecting experiment
should replace the current `nvcuda::wmma` PV wrapper with a CUTE `gemm_rs`-style
path inside the non-default FA2-like kernel.

## 35. CUTE PV, Swizzled V, and Register-P Results

Implementation status:

- The non-default FA2-like kernel now uses a CUTE `TiledMMA` PV path instead of
  `nvcuda::wmma`.
- V decode writes directly into a CUTE-compatible swizzled shared-memory layout.
- PV consumes V through the transposed/no-swizzle view and LDSM copy atom.
- P no longer materializes through shared memory in the FA2-like route. It is
  built as a register fragment, retiled with `convert_layout_acc_Aregs`, and
  passed to a local `gemm_rs` helper.
- The default GQA4 packed kernel is unchanged.

Build and test:

```bash
cmake --build build-bytev2-stable --target _C_stable_libtorch -j 32
cp build-bytev2-stable/_C_stable_libtorch.abi3.so vllm/_C_stable_libtorch.abi3.so
.venv/bin/python -m pytest tests/v1/attention/test_byte_v2_layout.py -v
```

Result:

- Build passed under `sm_86`.
- Test result: `111 passed, 1 skipped, 16 warnings`.
- Only the pre-existing unused `byte_v2_reshape_and_cache_kernel` CUDA warning
  remains in the ByteV2 build.

4096 p64/bn64 FA2-like measurements:

| variant | median | max diff |
| --- | ---: | ---: |
| CUTE skeleton, runtime still N-tile WMMA | 0.134656 ms | 0.0 |
| CUTE PV, row-major shared path | 0.140288 ms | 0.0 |
| CUTE PV, swizzled/LDSM V path | 0.138240 ms | 0.0 |
| CUTE PV + register P + `gemm_rs` | 0.133120 ms | 0.0 |

Artifacts:

- `profiles/byte_v2_gqa4_fa2_like_cute_pv_rowmajor_4096_p64_20260626.jsonl`
- `profiles/byte_v2_gqa4_fa2_like_cute_pv_swizzled_4096_p64_20260626.jsonl`
- `profiles/byte_v2_gqa4_fa2_like_cute_preg_clean_4096_p64_20260626.jsonl`

Context sweep, p64/bn64:

| seq len | default GQA4 packed | FA2-like CUTE reg-P |
| ---: | ---: | ---: |
| 256 | 0.033792 ms | 0.041984 ms |
| 512 | 0.033792 ms | 0.041984 ms |
| 1024 | 0.038912 ms | 0.049152 ms |
| 2048 | 0.055296 ms | 0.070656 ms |
| 4096 | 0.075264 ms | 0.133120 ms |

Artifacts:

- `profiles/byte_v2_gqa4_default_sweep_p64_bn64_20260626.jsonl`
- `profiles/byte_v2_gqa4_fa2_like_cute_preg_sweep_p64_bn64_20260626.jsonl`

Conclusion:

The register-P CUTE PV path is the best FA2-like variant so far and slightly
beats the previous FA2-like WMMA N-tile baseline. It still should not replace
the default GQA4 packed kernel. The current FA2-like route pays for an M=16
tensor-core PV shape while only four GQA rows are active, and it still inherits
the scalar QK/softmax staging that reconstructs P from `shared_probs`.

The remaining performance work is therefore not another local PV copy-layout
tweak. The next experiment should either:

1. prototype QK MMA so `acc_s` is produced in the same accumulator layout FA2
   expects; or
2. combine QK/PV into a single online mainloop, so P stays in registers from
   score MMA through PV.

Both should stay behind the FA2-like experiment route until they beat the
default GQA4 packed path on correctness and timing.

## 36. QK-MMA Prototype Attempt

Attempted but not retained.

Prototype intent:

- Keep the change isolated to the FA2-like experiment route.
- Decode/stage a padded 16x128 Q tile and a 64x128 K tile into CUTE shared
  layouts.
- Use CUTE MMA to produce a 16x64 score fragment.
- Write rows 0..3 back to the existing `shared_scores`, then reuse the current
  scalar softmax and CUTE/reg-P PV path.

Observed compile failures:

- The one-warp 16x64 QK version failed CUTE LDSM vectorization and B/C fragment
  shape checks.
- The four-warps-along-N version still failed:
    - `SM75_U32x4_LDSM_N` source layout was incompatible with the staged K view;
    - `cute::gemm` asserted `size<1>(B) == size<2>(C)`.

Decision:

- The QK-MMA prototype was fully reverted.
- The retained FA2-like kernel remains the CUTE PV + register-P version from
  section 35.

Post-revert sanity:

| route | median | max diff |
| --- | ---: | ---: |
| FA2-like CUTE PV + register P after QK revert | 0.133120 ms | 0.0 |

Artifact:

- `profiles/byte_v2_gqa4_fa2_like_cute_preg_after_qk_revert_4096_p64_20260626.jsonl`

Next QK action:

Do not retry QK-MMA as another direct scalar-loop replacement. Build a small
compile probe that exactly mirrors FA2's `TiledMma`, `SmemLayoutQdO`,
`SmemLayoutKV`, `SmemLayoutKtransposedNoSwizzle`, and `SmemCopyAtom` choices,
then integrate that proven layout into ByteV2's decode-to-K-tile staging.

## 37. FA2-Style QK-MMA Experiment Integrated

Change:

- Replaced the failed ad hoc QK-MMA attempts with a FA2-style Q/K layout:
  `ByteV2Fa2LikeCuteQkTraits<16, 64, 128, 1>`.
- Integrated the QK-MMA path into the isolated FA2-like GQA4 decode kernel.
- Kept it behind an explicit flag:
    - C++ tile policy: `tile_policy[10]`
    - microbench CLI: `--gqa-fa2-qk-mma`
- Added dispatch coverage for both normal split-k and guarded split-k.
- Added a CUDA correctness test covering FA2-like with and without QK-MMA.

Current QK-MMA structure:

```text
decode Q/K -> FA2-like CUTE shared Q/K tiles
QK CUTE MMA -> acc_s(16x64)
valid rows 0..3 -> shared_scores
shared_scores -> shared_probs
CUTE register-P PV path -> tmp_out
```

Build/test:

```bash
cmake --build build-bytev2-stable --target _C_stable_libtorch -j 32
cp build-bytev2-stable/_C_stable_libtorch.abi3.so vllm/_C_stable_libtorch.abi3.so
.venv/bin/python -m pytest tests/v1/attention/test_byte_v2_layout.py -v
```

Result:

- Build passed under `sm_86`.
- Only the pre-existing unused `byte_v2_reshape_and_cache_kernel` CUDA warning
  remains.
- Test result: `113 passed, 1 skipped, 16 warnings`.

4096 p64/bn64 CUDA-event sanity:

| variant | median | max diff |
| --- | ---: | ---: |
| FA2-like CUTE PV + register P | 0.133120 ms | 0.0 |
| FA2-like + QK-MMA | 0.179200 ms | 0.0 |
| guarded FA2-like + QK-MMA | 0.181776 ms | 0.0 |

Artifacts:

- `profiles/byte_v2_gqa4_fa2_like_final_4096_p64_20260626.jsonl`
- `profiles/byte_v2_gqa4_fa2_like_qk_mma_final_restored_4096_p64_20260626.jsonl`
- `profiles/byte_v2_gqa4_fa2_like_qk_mma_guarded_4096_p64_20260626.jsonl`

Negative sub-experiment:

- A direct-softmax variant computed softmax from the QK-MMA `acc_s` fragment
  and skipped the `shared_scores` read/write.
- It stayed correct, but regressed to `0.187392 ms`.
- The change was reverted and not retained.

Interpretation:

The QK-MMA path is now correct and test-covered, but it is not a performance
win. The main remaining difference from FA2 is CTA/work partitioning:

- FA2 keeps QK accumulator, online softmax, register P, and PV in a coherent
  tiled mainloop.
- Current ByteV2 QK-MMA uses warp 0 for QK, then writes scores to shared memory
  so the four PV warps can consume P for different output-dimension tiles.
- Avoiding shared P/scores without losing the four-warp PV mapping requires a
  different CTA/MMA partition, not a local copy-layout change.

Next optimization direction:

1. Treat the current QK-MMA path as a correctness/layout baseline only.
2. Design a new CTA shape where the same warp group owns a QK accumulator row
   group and its PV output tiles, or where P broadcast is explicitly amortized.
3. Do not promote the FA2-like route until it beats the default GQA4 packed
   kernel; the current default remains the faster path.

## 38. FA2-Style Single-Warp Mainloop Redesign

Change:

- Added a second experimental FA2-like subpath behind `tile_policy[11]` and
  microbench flag `--gqa-fa2-mainloop`.
- The flag requires the QK-MMA subpath.
- The new path keeps the existing split-k/reduce interface but changes the
  partition-internal flow to:

```text
decode/stage Q/K/V -> CUTE shared tiles
warp0 QK CUTE MMA -> acc_s
softmax directly on acc_s
acc_s -> BF16 register P
warp0 full-head PV gemm_rs -> acc_o(16x128)
write rows 0..3 to tmp_out and LSE to exp_sums
```

This is closer to FA2 than the previous QK-MMA path because P no longer goes
through `shared_scores/shared_probs` before PV.

Build/test:

```bash
cmake --build build-bytev2-stable --target _C_stable_libtorch -j 32
cp build-bytev2-stable/_C_stable_libtorch.abi3.so vllm/_C_stable_libtorch.abi3.so
.venv/bin/python -m pytest tests/v1/attention/test_byte_v2_layout.py -v
.venv/bin/pre-commit run ruff-check --files \
  tests/v1/attention/test_byte_v2_layout.py \
  scripts/byte_v2_decode_microbench.py
```

Result:

- Build passed under `sm_86`.
- Only the pre-existing unused `byte_v2_reshape_and_cache_kernel` CUDA warning
  remains.
- Test result: `114 passed, 1 skipped, 16 warnings`.
- Ruff passed.

4096 p64/bn64 CUDA-event sanity:

| variant | median | max diff |
| --- | ---: | ---: |
| default GQA4 packed | 0.076800 ms | 0.0 |
| FA2-like CUTE PV + register P | 0.133120 ms | 0.0 |
| FA2-like + QK-MMA shared-score path | 0.179200 ms | 0.0 |
| FA2-like + QK-MMA + single-warp mainloop | 0.145408 ms | 0.0 |
| guarded FA2-like + QK-MMA + single-warp mainloop | 0.143360 ms | 0.0 |

Artifacts:

- `profiles/byte_v2_gqa4_default_after_mainloop_4096_p64_20260626.jsonl`
- `profiles/byte_v2_gqa4_fa2_like_after_mainloop_base_4096_p64_20260626.jsonl`
- `profiles/byte_v2_gqa4_fa2_like_qk_mma_after_mainloop_4096_p64_20260626.jsonl`
- `profiles/byte_v2_gqa4_fa2_like_mainloop_4096_p64_20260626.jsonl`
- `profiles/byte_v2_gqa4_fa2_like_mainloop_guarded_4096_p64_20260626.jsonl`

Interpretation:

The FA2-style mainloop improves the previous QK-MMA experiment substantially:

```text
0.179200 ms -> 0.145408 ms
```

This confirms that keeping score/P in the accumulator/register path is the
right structural direction. It is still not a retained fast path because it is
slower than the existing FA2-like CUTE PV/register-P route and much slower than
the default GQA4 packed kernel.

Current main gap:

- The new path uses only warp0 for full 128-dim PV, so it gives up the
  four-warp output-dimension parallelism used by the older FA2-like path.
- The older path keeps PV parallelism but pays shared-score/shared-P fanout.
- The next useful design needs both properties: FA2-like P/register dataflow
  and multi-warp PV parallelism.

Next profile target:

Run NCU on:

1. default GQA4 packed;
2. FA2-like CUTE PV/register-P;
3. FA2-like QK-MMA shared-score path;
4. FA2-like single-warp mainloop.

The key metrics should be instruction count, tensor-core utilization, shared
load/store wavefronts, barriers, and eligible warps. This will show whether the
single-warp mainloop is mainly limited by lost PV parallelism, full V staging,
or register/scoreboard pressure.

## 39. Multi-Warp Register-P Mainloop Attempt

Change:

- Added another FA2-like experiment behind `tile_policy[12]` and microbench flag
  `--gqa-fa2-multiwarp`.
- The path requires `--gqa-fa2-qk-mma` and is mutually exclusive with
  `--gqa-fa2-mainloop`.
- The new path keeps P in registers for PV, but restores multi-warp output-dim
  parallelism by having each warp recompute QK/softmax and then compute its own
  32 output dimensions.

Structure:

```text
decode/stage shared Q/K once
each warp:
  QK CUTE MMA -> local acc_s
  softmax on acc_s
  acc_s -> register P
  decode two 16-dim V tiles
  PV gemm_rs -> two output-dim tiles
warp0 writes partition LSE
```

Build/test:

```bash
cmake --build build-bytev2-stable --target _C_stable_libtorch -j 32
cp build-bytev2-stable/_C_stable_libtorch.abi3.so vllm/_C_stable_libtorch.abi3.so
.venv/bin/python -m pytest tests/v1/attention/test_byte_v2_layout.py -v
.venv/bin/pre-commit run ruff-check --files \
  tests/v1/attention/test_byte_v2_layout.py \
  scripts/byte_v2_decode_microbench.py
```

Result:

- Build passed under `sm_86`.
- Only the pre-existing unused `byte_v2_reshape_and_cache_kernel` CUDA warning
  remains.
- Test result: `115 passed, 1 skipped, 16 warnings`.
- Ruff passed.

4096 p64/bn64 CUDA-event sanity:

| variant | median | max diff |
| --- | ---: | ---: |
| FA2-like CUTE PV + register P | 0.133120 ms | 0.0 |
| FA2-like + QK-MMA shared-score path | 0.179200 ms | 0.0 |
| FA2-like + QK-MMA + single-warp mainloop | 0.145408 ms | 0.0 |
| FA2-like + QK-MMA + multi-warp register-P | 0.186368 ms | 0.0 |
| guarded FA2-like + QK-MMA + multi-warp register-P | 0.185344 ms | 0.0 |

Artifacts:

- `profiles/byte_v2_gqa4_fa2_like_multiwarp_4096_p64_20260626.jsonl`
- `profiles/byte_v2_gqa4_fa2_like_multiwarp_guarded_4096_p64_20260626.jsonl`

Interpretation:

The experiment is correct but slower than both the single-warp mainloop and the
older QK-MMA shared-score path. The added QK/softmax recomputation in four
warps dominates the benefit from restoring PV output-dim parallelism.

Updated conclusion:

- The useful signal from section 38 still holds: `acc_s -> register P -> PV`
  is better than shared-score fanout when QK/P is computed once.
- Recomputing QK/P per PV warp is not viable.
- The next multi-warp design needs a one-producer/multi-consumer P delivery
  mechanism that is cheaper than recomputing QK and cheaper than the old
  float `shared_scores/shared_probs` path.

## 40. FA2-Direct Payload Pair-Load Optimization

Question:

Can the retained FA2-direct no-outlier path reduce instruction count by loading
two adjacent fixed 12-bit payload elements at once?

Change:

- Added a pair loader for the fixed split-plane payload:
    - one aligned `u16` low-byte load for two adjacent dimensions;
    - one packed-code byte load for the two 4-bit high codes;
    - two BF16 bit results.
- Updated the FA2-direct safe staging path so each staging thread writes two
  adjacent K dims and two adjacent V dims.
- Kept the current compressed format unchanged.
- Kept the change limited to the direct no-fallback/no-outlier experiment path
  behind `--gqa-fa2-direct` / `BYTE_V2_DECODE_GQA_FA2_DIRECT=1`.

Build/test:

```bash
cmake --build build-bytev2-stable --target _C_stable_libtorch -j 32
cp build-bytev2-stable/_C_stable_libtorch.abi3.so vllm/_C_stable_libtorch.abi3.so
CUDA_VISIBLE_DEVICES=6 .venv/bin/python -m pytest \
  tests/v1/attention/test_byte_v2_layout.py::test_byte_v2_paged_decode_attention_split_k_gqa_fa2_like_cuda_matches_raw_reference -v
.venv/bin/python -m py_compile \
  vllm/v1/attention/backends/byte_v2_attn.py \
  scripts/byte_v2_decode_microbench.py \
  tests/v1/attention/test_byte_v2_layout.py
.venv/bin/pre-commit run ruff-check --files \
  vllm/v1/attention/backends/byte_v2_attn.py \
  scripts/byte_v2_decode_microbench.py \
  tests/v1/attention/test_byte_v2_layout.py
```

Result:

- Build passed under `sm_86`.
- Targeted CUDA correctness test: `7 passed, 16 warnings`.
- Python compile passed.
- Ruff passed.

4096 p256 CUDA-event sanity:

| variant | median |
| --- | ---: |
| FA2-direct after partition loop | 0.138240 ms |
| FA2-direct after descriptor hoist | 0.129024 ms |
| FA2-direct after skip full-tile clear | 0.113664 ms |
| FA2-direct after pair payload load | 0.101376-0.103424 ms |
| raw vLLM FA2 paged | 0.064000-0.064512 ms |

Context-length sweep after pair payload load:

| seq len | best ByteV2 direct | best partition | raw FA2 |
| ---: | ---: | ---: | ---: |
| 256 | 0.026624 ms | p64 | 0.062464 ms |
| 512 | 0.026624 ms | p64 | 0.060416 ms |
| 1024 | 0.035840 ms | p64 | 0.060416 ms |
| 2048 | 0.060416 ms | p128 | 0.063488 ms |
| 4096 | 0.101376-0.103424 ms | p256 | 0.064000-0.064512 ms |

NCU main-kernel comparison at 4096/p256:

| metric | skip-clear | pair payload load | raw FA2 splitkv |
| --- | ---: | ---: | ---: |
| NCU duration | 131.168 us | 117.312 us | 65.9 us |
| total instructions | 11.140M | 8.351M | 1.408M |
| global load inst | 551.4K | 289.3K | n/a |
| integer thread inst | 243.8M | 174.9M | n/a |
| DRAM read | 22.57 MB | 21.71 MB | 19.25 MB |
| shared store inst | 294.9K | 294.9K | n/a |
| HMMA inst | 262.1K | 262.1K | n/a |
| registers/thread | 177 | 175 | 252 |
| static shared/block | 49.152 KB | 49.152 KB | n/a |

Artifacts:

- `profiles/byte_v2_fa2_direct_pairvec_partition_sweep_vs_raw_fa2_4096_20260626.jsonl`
- `profiles/byte_v2_fa2_direct_pairvec_sweep_vs_raw_fa2_20260626.jsonl`
- `profiles/ncu_byte_v2_fa2_direct_main_default_4096_p256_pairvec_20260626.txt`
- `profiles/ncu_byte_v2_fa2_direct_main_counters_4096_p256_pairvec_20260626.txt`
- `profiles/ncu_raw_fa2_splitkv_default_4096_20260626.txt`
- `profiles/ncu_raw_fa2_splitkv_counters_4096_20260626.txt`

Interpretation:

This is a retained optimization. It improves the best 4096/p256 direct path by
about 9-11% relative to the previous skip-clear state, and it cuts the main
kernel instruction count by about 25%. The improvement matches the expected
effect: fewer payload loads and less per-element integer decode work. The HMMA
and shared-store counts are unchanged, which confirms this did not change the
QK/PV math shape.

Remaining gap:

- The direct path still executes about 5.9x more main-kernel instructions than
  raw FA2 splitkv at the same 4096 decode point.
- DRAM read is now close to raw FA2, so the dominant remaining gap is not raw
  byte traffic. It is the compressed-payload decode and staging instruction
  stream.
- The next local optimization should continue reducing staging/decode
  instructions before attempting another CTA-level rewrite.

## 41. Current ByteV2 vs FA2 Profile Refresh

Setup:

- GPU: `CUDA_VISIBLE_DEVICES=6`
- Decode shape: batch 1, 32 Q heads, 8 KV heads, head dim 128, block size 16
- ByteV2 path: direct no-fallback/no-outlier GQA4 FA2-like QK-MMA path
- ByteV2 partitions swept: p64, p128, p256
- FA2 baseline: vLLM paged FA2

Commands:

```bash
CUDA_VISIBLE_DEVICES=6 .venv/bin/python scripts/byte_v2_decode_microbench.py \
  --seq-lens 256 512 1024 2048 4096 \
  --partition-sizes 64 128 256 \
  --compute-block-ns 64 \
  --warmup 30 --iters 80 \
  --no-outlier-inputs --assume-no-outlier \
  --gqa-packed --gqa-fa2-like --gqa-fa2-qk-mma --gqa-fa2-direct \
  --include-flash \
  --output-jsonl profiles/byte_v2_current_vs_fa2_sweep_20260626.jsonl
```

CUDA-event sweep:

| seq len | best ByteV2 direct | partition | raw FA2 | ByteV2 / FA2 |
| ---: | ---: | ---: | ---: | ---: |
| 256 | 0.026624 ms | p64 | 0.061440 ms | 0.433x |
| 512 | 0.026624 ms | p64 | 0.060416 ms | 0.441x |
| 1024 | 0.035840 ms | p64 | 0.063488 ms | 0.565x |
| 2048 | 0.060416 ms | p128 | 0.062464 ms | 0.967x |
| 4096 | 0.103424 ms | p256 | 0.063488 ms | 1.629x |

4096 NCU kernel-level comparison:

| metric | ByteV2 main | ByteV2 reduce | FA2 splitkv | FA2 combine |
| --- | ---: | ---: | ---: | ---: |
| NCU duration | 116.320 us | 6.496 us | 44.192 us | 13.376 us |
| grid | 128 CTA | 1024 CTA | 128 CTA | 8 CTA |
| registers/thread | 175 | 24 | 252 | 52 |
| shared/block | 49.152 KB static | 0 | 81.920 KB dynamic | 320 B static |
| active warps | 5.99 | 35.65 | 3.96 | 3.97 |
| active warp pct | 12.48% | 74.27% | 8.25% | 8.28% |
| memory throughput | 28.25% | 27.34% | 70.46% | 3.29% |
| SM throughput | 16.71% | 30.15% | 24.52% | 0.50% |
| total instructions | 8.351M | 0.782M | 1.408M | 0.028M |
| HMMA instructions | 262.1K | 0 | 262.1K | 0 |
| global load inst | 289.3K | 16.4K | 3.1K | 0.5K |
| global store inst | 4.6K | 4.1K | 0.9K | 0.05K |
| shared load inst | 0 | 0 | 8.2K | 0.5K |
| shared store inst | 294.9K | 0 | 16.4K | 0.03K |
| integer thread inst | 174.9M | 10.4M | 11.7M | 0.42M |
| DRAM read | 21.695 MB | 370.6 KB | 19.275 MB | 323.6 KB |
| DRAM write | 3.122 MB | 2.0 KB | 2.441 MB | 18.2 KB |

Aggregate 4096 NCU view:

| metric | ByteV2 total | FA2 total | ByteV2 / FA2 |
| --- | ---: | ---: | ---: |
| NCU duration | 122.816 us | 57.568 us | 2.13x |
| total instructions | 9.133M | 1.436M | 6.36x |
| integer thread inst | 185.4M | 12.1M | 15.33x |
| DRAM read | 22.065 MB | 19.599 MB | 1.13x |
| DRAM write | 3.124 MB | 2.459 MB | 1.27x |
| global load inst | 305.7K | 3.6K | 84.9x |
| shared store inst | 294.9K | 16.4K | 18.0x |

Artifacts:

- `profiles/byte_v2_current_vs_fa2_sweep_20260626.jsonl`
- `profiles/ncu_byte_v2_current_main_default_4096_p256_20260626.txt`
- `profiles/ncu_byte_v2_current_main_counters_4096_p256_20260626.txt`
- `profiles/ncu_byte_v2_current_reduce_default_4096_p256_20260626.txt`
- `profiles/ncu_byte_v2_current_reduce_counters_4096_p256_20260626.txt`
- `profiles/ncu_fa2_current_splitkv_default_4096_20260626.txt`
- `profiles/ncu_fa2_current_splitkv_counters_4096_20260626.txt`
- `profiles/ncu_fa2_current_combine_default_4096_20260626.txt`
- `profiles/ncu_fa2_current_combine_counters_4096_20260626.txt`

Conclusion:

- ByteV2 direct is faster than FA2 at 256-1024 tokens and roughly tied at 2048.
- At 4096, ByteV2 is still slower: `0.103424 ms` vs FA2 `0.063488 ms`.
- The 4096 gap is not explained by DRAM bytes. ByteV2 reads only about 13%
  more DRAM than FA2.
- The gap is dominated by instruction shape:
    - about 6.4x more total instructions;
    - about 15.3x more integer thread instructions;
    - about 85x more global load instructions;
    - about 18x more shared store instructions.
- Both main kernels execute the same HMMA count, so the extra work is around
  payload decode/staging and not the tensor-core QK/PV math itself.

Next optimization target:

Focus on reducing direct-path payload decode/staging instructions:

1. Further vectorize payload decode across more dimensions or rows.
2. Reduce shared K/V staging stores by writing directly into the exact fragment
   layout consumed by QK/PV.
3. Avoid per-row/per-dim address arithmetic in the hot staging loop by hoisting
   more descriptor/offset math or using warp-uniform row scheduling.

## 42. FA2-Direct Quad Payload Load Experiment

Question:

Can the pair payload loader be widened again so each staging thread decodes
four adjacent dimensions?

Change:

- Added an aligned `u32` low-byte loader.
- Added a quad payload decode helper for the fixed split-plane 12-bit payload:
    - one aligned `u32` load for four low bytes;
    - one aligned `u16` load for four packed 4-bit high codes;
    - four BF16 bit results.
- Changed the FA2-direct safe staging path from `HeadDim / 2` dim groups to
  `HeadDim / 4` dim groups.
- Kept the cache format unchanged.
- Kept unsafe/outlier fallback paths unchanged.

Build/test:

```bash
cmake --build build-bytev2-stable --target _C_stable_libtorch -j 32
cp build-bytev2-stable/_C_stable_libtorch.abi3.so vllm/_C_stable_libtorch.abi3.so
CUDA_VISIBLE_DEVICES=6 .venv/bin/python -m pytest \
  tests/v1/attention/test_byte_v2_layout.py::test_byte_v2_paged_decode_attention_split_k_gqa_fa2_like_cuda_matches_raw_reference -v
```

Result:

- Build passed under `sm_86`.
- Targeted CUDA correctness test: `7 passed, 16 warnings`.
- Only the pre-existing unused CUDA warnings remain.

4096 p256 first check:

| variant | median |
| --- | ---: |
| pair payload load | 0.103424 ms |
| quad payload load | 0.078848 ms |
| raw FA2 in same run | 0.065536 ms |

Context sweep after quad payload load:

| seq len | best ByteV2 direct | best partition | raw FA2 | ByteV2 / FA2 |
| ---: | ---: | ---: | ---: | ---: |
| 256 | 0.024576 ms | p64 | 0.063488 ms | 0.387x |
| 512 | 0.024576 ms | p64 | 0.061440 ms | 0.400x |
| 1024 | 0.030720 ms | p64 | 0.061440 ms | 0.500x |
| 2048 | 0.048128 ms | p128 | 0.063488 ms | 0.758x |
| 4096 | 0.077824 ms | p256 | 0.064512 ms | 1.206x |

4096 p256 main-kernel NCU:

| metric | pair payload load | quad payload load | delta |
| --- | ---: | ---: | ---: |
| NCU duration | 116.320 us | 89.664 us | -22.9% |
| total instructions | 8.351M | 7.481M | -10.4% |
| global load inst | 289.3K | 158.2K | -45.3% |
| integer thread inst | 174.9M | 150.3M | -14.1% |
| DRAM read | 21.695 MB | 19.057 MB | -12.2% |
| shared store inst | 294.9K | 294.9K | 0.0% |
| HMMA inst | 262.1K | 262.1K | 0.0% |
| registers/thread | 175 | 175 | 0 |

Artifacts:

- `profiles/byte_v2_fa2_direct_quadvec_4096_p256_vs_fa2_20260626.jsonl`
- `profiles/byte_v2_fa2_direct_quadvec_sweep_vs_fa2_20260626.jsonl`
- `profiles/ncu_byte_v2_fa2_direct_main_default_4096_p256_quadvec_20260626.txt`
- `profiles/ncu_byte_v2_fa2_direct_main_counters_4096_p256_quadvec_20260626.txt`

Interpretation:

This is a retained optimization. It beats the 5% threshold by a wide margin.
The effect is exactly where expected: global load instruction count drops
almost in half, integer decode work drops, and register count does not increase.
Shared-store and HMMA counts are unchanged, so the remaining gap is now even
more clearly the staging path after decode rather than tensor-core math.

Updated remaining gap:

- At 4096, ByteV2 direct is now only about 20% slower than raw FA2 in the
  CUDA-event sweep.
- The main kernel still has far more global load instructions than FA2 because
  it explicitly decodes compressed payload into shared memory.
- The next single-point experiment should test `8 dims/thread` only if it does
  not inflate registers or hurt coalescing; otherwise the next target should be
  reducing the unchanged `294.9K` shared stores by writing into a more direct
  FA2/CUTE consumer layout.

## 43. 8/16/32 Dims-Per-Thread Payload Experiments

Question:

After quad payload load, can the direct no-outlier staging path continue to
benefit from larger per-thread decode granularity?

Experiment policy:

- Try one granularity at a time.
- Keep a step only if it improves the previous retained version by more than
  5% on 4096/p256.
- Stop once a larger granularity does not clear the threshold.

Changes tested:

1. `8 dims/thread`
   - one `u64` low-plane load;
   - one `u32` code-plane load;
   - 8 BF16 outputs per K/V descriptor.
2. `16 dims/thread`
   - two `u64` low-plane loads;
   - one `u64` code-plane load;
   - 16 BF16 outputs per K/V descriptor.
3. `32 dims/thread`
   - two 16-dim codec tiles per thread;
   - correct, but crosses codec tile descriptors and leaves fewer staging
     threads active.

4096/p256 CUDA-event results:

| variant | median | vs previous retained | decision |
| --- | ---: | ---: | --- |
| 4 dims/thread | 0.077824 ms | baseline | previous retained |
| 8 dims/thread | 0.066560 ms | +14.5% | retained for next test |
| 16 dims/thread | 0.061440 ms | +7.7% | retained |
| 32 dims/thread | 0.063488 ms | -3.3% vs 16 dims | reverted |

Final 16-dim context sweep:

| seq len | best ByteV2 direct | best partition | raw FA2 | ByteV2 / FA2 |
| ---: | ---: | ---: | ---: | ---: |
| 256 | 0.024576 ms | p64 | 0.062464 ms | 0.393x |
| 512 | 0.023552 ms | p64 | 0.061440 ms | 0.383x |
| 1024 | 0.027648 ms | p64 | 0.061440 ms | 0.450x |
| 2048 | 0.039936 ms | p128 | 0.063488 ms | 0.629x |
| 4096 | 0.061440 ms | p256 | 0.064512 ms | 0.952x |

Final 16-dim main-kernel NCU at 4096/p256:

| metric | 4 dims/thread | 8 dims/thread | final 16 dims/thread |
| --- | ---: | ---: | ---: |
| NCU duration | 89.664 us | 75.616 us | 69.120 us |
| total instructions | 7.481M | 6.605M | 6.564M |
| global load inst | 158.2K | 92.7K | 76.3K |
| integer thread inst | 150.3M | 125.2M | 126.8M |
| DRAM read | 19.057 MB | 17.624 MB | 15.909 MB |
| shared store inst | 294.9K | 294.9K | 294.9K |
| HMMA inst | 262.1K | 262.1K | 262.1K |
| registers/thread | 175 | 174 | 173 |

Build/test:

```bash
cmake --build build-bytev2-stable --target _C_stable_libtorch -j 32
cp build-bytev2-stable/_C_stable_libtorch.abi3.so vllm/_C_stable_libtorch.abi3.so
CUDA_VISIBLE_DEVICES=6 .venv/bin/python -m pytest \
  tests/v1/attention/test_byte_v2_layout.py::test_byte_v2_paged_decode_attention_split_k_gqa_fa2_like_cuda_matches_raw_reference -v
```

Result:

- Final retained version is `16 dims/thread`.
- Correctness: `7 passed, 16 warnings`.
- 32 dims/thread was correct but not retained.
- The final cleanup removed the unretained 4/8-dim helper code from the active
  source; only the 16-dim `u64` path remains in the direct safe staging path.

Artifacts:

- `profiles/byte_v2_fa2_direct_octvec_4096_p256_vs_fa2_20260626.jsonl`
- `profiles/byte_v2_fa2_direct_hexvec_4096_p256_vs_fa2_20260626.jsonl`
- `profiles/byte_v2_fa2_direct_32dims_4096_p256_vs_fa2_20260626.jsonl`
- `profiles/byte_v2_fa2_direct_hexvec_final_sweep_vs_fa2_20260626.jsonl`
- `profiles/byte_v2_fa2_direct_hexvec_final_clean_4096_p256_vs_fa2_20260626.jsonl`
- `profiles/ncu_byte_v2_fa2_direct_main_default_4096_p256_hexvec_final_20260626.txt`
- `profiles/ncu_byte_v2_fa2_direct_main_counters_4096_p256_hexvec_final_20260626.txt`

Interpretation:

The 16-dim path is the best granularity found in this sequence. It is large
enough to reduce descriptor/address/global-load overhead, but not so large that
staging parallelism collapses. The failed 32-dim result suggests that going to
64 dims/thread is unlikely to help: it would further reduce active staging
threads and cross more codec descriptors without reducing the fixed shared
store count.

Updated next target:

The decode/load part is now close enough that the unchanged `294.9K` shared
stores are the next obvious local bottleneck. The next useful single-point
experiment should target shared staging layout/store count, not wider payload
decode granularity.

## 44. Vectorized Shared Store Experiment

Question:

Can the retained 16-dim direct staging path reduce shared-store instruction
count without changing the FA2-direct QK/PV dataflow?

Layout check:

- `CuteDirect{QK,PV}Traits::SmemLayoutKV` stores each row in 8-BF16 contiguous
  physical groups.
- A logical 16-dim slice is two 8-BF16 physical groups.
- Each 8-BF16 group is 16-byte aligned.

Change:

- Replaced sixteen scalar `sK/sV(row, col + i)` stores in the hot direct safe
  staging path with two raw `uint4` shared-memory stores.
- The helper packs eight BF16 bit patterns into one 16-byte `uint4`.
- K and V still land in the exact same CUTE shared layout.
- QK/PV math, payload format, split-K interface, and unsafe/outlier fallback
  paths are unchanged.

Build/test:

```bash
cmake --build build-bytev2-stable --target _C_stable_libtorch -j 32
cp build-bytev2-stable/_C_stable_libtorch.abi3.so vllm/_C_stable_libtorch.abi3.so
CUDA_VISIBLE_DEVICES=6 .venv/bin/python -m pytest \
  tests/v1/attention/test_byte_v2_layout.py::test_byte_v2_paged_decode_attention_split_k_gqa_fa2_like_cuda_matches_raw_reference -v
```

Result:

- Build passed under `sm_86`.
- Correctness: `7 passed, 16 warnings`.

4096/p256 CUDA-event check:

| variant | median | decision |
| --- | ---: | --- |
| 16 dims/thread scalar CUTE store | 0.061440 ms | previous retained |
| 16 dims/thread vector shared store | 0.057344 ms | retained |
| raw FA2 in same run | 0.065536 ms | reference |

Context sweep after vector shared store:

| seq len | best ByteV2 direct | best partition | raw FA2 | ByteV2 / FA2 |
| ---: | ---: | ---: | ---: | ---: |
| 256 | 0.024576 ms | p64 | 0.063488 ms | 0.387x |
| 512 | 0.023552 ms | p64 | 0.062464 ms | 0.377x |
| 1024 | 0.025600 ms | p64 | 0.062464 ms | 0.410x |
| 2048 | 0.038912 ms | p128 | 0.064512 ms | 0.603x |
| 4096 | 0.057344 ms | p256 | 0.065536 ms | 0.875x |

4096/p256 main-kernel NCU:

| metric | 16-dim scalar store | vector shared store | delta |
| --- | ---: | ---: | ---: |
| NCU duration | 69.120 us | 64.576 us | -6.6% |
| total instructions | 6.564M | 5.893M | -10.2% |
| global load inst | 76.3K | 76.3K | 0.0% |
| shared store inst | 294.9K | 65.5K | -77.8% |
| integer thread inst | 126.8M | 116.4M | -8.2% |
| DRAM read | 15.909 MB | 15.211 MB | -4.4% |
| HMMA inst | 262.1K | 262.1K | 0.0% |
| registers/thread | 173 | 175 | +2 |

Artifacts:

- `profiles/byte_v2_fa2_direct_vector_shared_store_4096_p256_vs_fa2_20260626.jsonl`
- `profiles/byte_v2_fa2_direct_vector_shared_store_sweep_vs_fa2_20260626.jsonl`
- `profiles/ncu_byte_v2_fa2_direct_main_default_4096_p256_vector_shared_store_20260626.txt`
- `profiles/ncu_byte_v2_fa2_direct_main_counters_4096_p256_vector_shared_store_20260626.txt`

Interpretation:

This experiment directly improved the target metric and clears the 5% retention
threshold. The large shared-store instruction drop confirms that the previous
`sK/sV(row, col)` sequence was still leaving expensive shared-store instruction
pressure in the staging path. The new version keeps the CUTE consumer layout
unchanged while feeding it with wider stores.

Updated next target:

The main remaining local differences are no longer full payload staging stores
or HBM bytes. The next profile should compare the retained vector-store ByteV2
against FA2 again, then decide between:

1. reducing residual integer/address instructions in the 16-dim decode helper;
2. reducing split/reduce overhead now that the main kernel is faster than the
   raw FA2 event baseline for the tested 4096 case;
3. trying a guarded vector-store version for tail/invalid tiles if shorter or
   irregular contexts show regressions.

## 45. Retained Vector-Store ByteV2 vs FA2 Total Profile

Question:

After vectorizing shared stores, is the next bottleneck the main kernel's
residual decode/address work, or split/reduce overhead?

Setup:

- Same 4096 decode shape as previous sections.
- ByteV2 path: direct no-fallback/no-outlier, 16 dims/thread, vector shared
  stores, p256.
- FA2 reference: vLLM paged FA2 splitkv + combine.

CUDA-event check:

| path | median |
| --- | ---: |
| ByteV2 direct p256 | 0.057344 ms |
| raw FA2 paged | 0.065536 ms |

NCU kernel-level comparison:

| metric | ByteV2 main | ByteV2 reduce | FA2 splitkv | FA2 combine |
| --- | ---: | ---: | ---: | ---: |
| NCU duration | 64.576 us | 6.592 us | 44.864 us | 13.440 us |
| grid | 128 CTA | 1024 CTA | 128 CTA | 8 CTA |
| registers/thread | 175 | 24 | 252 | 52 |
| total instructions | 5.893M | 0.782M | 1.408M | 0.028M |
| HMMA instructions | 262.1K | 0 | 262.1K | 0 |
| global load inst | 76.3K | 16.4K | 3.1K | 0.5K |
| global store inst | 4.6K | 4.1K | 0.9K | 0.05K |
| shared load inst | 0 | 0 | 8.2K | 0.5K |
| shared store inst | 65.5K | 0 | 16.4K | 0.03K |
| integer thread inst | 116.4M | 10.4M | 11.7M | 0.42M |
| DRAM read | 15.211 MB | 0.369 MB | 19.279 MB | 0.323 MB |
| DRAM write | 3.023 MB | 0.002 MB | 2.321 MB | 0.002 MB |

Aggregate NCU view:

| metric | ByteV2 total | FA2 total | ByteV2 / FA2 |
| --- | ---: | ---: | ---: |
| NCU duration | 71.168 us | 58.304 us | 1.22x |
| total instructions | 6.675M | 1.436M | 4.65x |
| integer thread inst | 126.8M | 12.1M | 10.49x |
| global load inst | 92.7K | 3.6K | 25.74x |
| shared store inst | 65.5K | 16.4K | 3.99x |
| DRAM read | 15.580 MB | 19.602 MB | 0.79x |
| DRAM write | 3.025 MB | 2.323 MB | 1.30x |

Artifacts:

- `profiles/byte_v2_fa2_direct_vector_shared_store_total_profile_4096_p256_vs_fa2_20260626.jsonl`
- `profiles/ncu_byte_v2_vector_shared_store_reduce_default_4096_p256_20260626.txt`
- `profiles/ncu_byte_v2_vector_shared_store_reduce_counters_4096_p256_20260626.txt`
- `profiles/ncu_fa2_vector_store_compare_splitkv_default_4096_20260626.txt`
- `profiles/ncu_fa2_vector_store_compare_splitkv_counters_4096_20260626.txt`
- `profiles/ncu_fa2_vector_store_compare_combine_default_4096_20260626.txt`
- `profiles/ncu_fa2_vector_store_compare_combine_counters_4096_20260626.txt`

Interpretation:

- CUDA-event timing shows the retained ByteV2 direct path is faster than raw
  FA2 for this 4096/p256 benchmark.
- NCU replay timings should be used mainly for kernel breakdown and counters;
  under NCU, ByteV2 main is still longer than FA2 splitkv.
- ByteV2 reduce is not the next bottleneck. It is only `6.592 us` and is already
  smaller than FA2 combine.
- The remaining local gap is in the main kernel:
    - about 10x more integer thread instructions than FA2 splitkv;
    - about 25x more global load instructions in the aggregate view;
    - about 4x more shared store instructions even after vectorizing stores.

Decision:

Do not spend the next step on split/reduce. The next single-point experiment
should target residual integer/address instructions in the 16-dim payload decode
helper. In particular, the current retained path decodes into 16 `uint16_t`
temporaries and then packs them into two `uint4` stores; a direct decode-to-`uint4`
helper may remove some of that scalar unpack/repack work.

## 46. Direct Decode-to-uint4 Helper Experiment

Question:

Can the retained 16-dim path reduce residual integer/repack instructions by
decoding fixed-format payload directly into two `uint4` values, instead of first
materializing 16 `uint16_t` bf16 bit values and then packing them for vector
shared stores?

Change tested:

- Added a direct helper that loads `low[16]` and packed codes once and emits two
  `uint4` values.
- Replaced the retained safe-page staging path with direct `uint4` decode/store.
- Kept the same FA2-direct no-fallback/no-outlier path, p256 partitioning, and
  vector shared-store consumer layout.

Correctness:

- `tests/v1/attention/test_byte_v2_layout.py::test_byte_v2_paged_decode_attention_split_k_gqa_fa2_like_cuda_matches_raw_reference`
  passed: 7 passed, 16 warnings.

CUDA-event result, 4096/p256:

| path | median |
| --- | ---: |
| retained vector-store baseline | 0.057344 ms |
| decode-to-uint4 experiment | 0.057344 ms |
| raw FA2 paged | 0.065536 ms |

Artifacts:

- `profiles/byte_v2_fa2_direct_decode_to_uint4_4096_p256_vs_fa2_20260626.jsonl`
- `profiles/byte_v2_fa2_direct_vector_store_restored_4096_p256_vs_fa2_20260626.jsonl`

Decision:

Rejected and reverted. This change did not clear the 5% retention threshold and
did not move the CUDA-event median. The retained implementation remains the
previous 16-dim/thread path with vectorized shared stores:

- decode fixed payload into 16 bf16 bit values;
- pack those values into two `uint4` stores;
- feed the unchanged CUTE QK/PV shared-memory layout.

Next target:

The no-op result suggests the compiler was already handling most of the scalar
pack/repack work efficiently, or the remaining time is dominated by address
generation, metadata loads, softmax/PV bookkeeping, and FA2-style mainloop
overheads rather than this final local packing step. The next single-point
optimization should target one of these remaining instruction sources with an
NCU counter check before/after:

1. reduce per-row descriptor/address arithmetic inside the safe-page staging
   loop;
2. reduce softmax/shared-prob fanout instructions;
3. reduce PV accumulator and output writeback bookkeeping.

## 47. K/V Offset Reuse in Safe Staging

Question:

Does the retained direct safe staging path still pay visible integer/address
cost for computing the same row offsets separately in the K and V 16-dim decode
helpers?

Change tested:

- Added an offsets-based 16-dim fixed payload loader.
- Computed `low_offset` and `code_offset` once per staged row in the hot direct
  safe-page loop.
- Reused those offsets for both K and V payload decode.
- Kept vector shared stores and the FA2-direct QK/PV path unchanged.

Correctness:

- `tests/v1/attention/test_byte_v2_layout.py::test_byte_v2_paged_decode_attention_split_k_gqa_fa2_like_cuda_matches_raw_reference`
  passed: 7 passed, 16 warnings.

CUDA-event result, 4096/p256:

| path | median |
| --- | ---: |
| retained vector-store baseline | 0.057344 ms |
| K/V offset reuse experiment | 0.057344 ms |
| restored retained vector-store binary | 0.057344 ms |

Artifacts:

- `profiles/byte_v2_fa2_direct_reuse_offsets_4096_p256_vs_fa2_20260626.jsonl`
- `profiles/byte_v2_fa2_direct_vector_store_restored_after_offsets_4096_p256_vs_fa2_20260626.jsonl`

Decision:

Rejected and reverted. The event median did not move, so this did not clear the
5% retention threshold. This likely means nvcc already common-subexpressed or
strength-reduced the row-offset arithmetic, or the remaining integer pressure is
not from these two offset expressions.

Current retained state:

- direct no-fallback/no-outlier path;
- 16 dims/thread fixed payload decode;
- two `u64` low-plane loads plus one `u64` code-plane load per K/V tile slice;
- vectorized shared stores into the unchanged CUTE K/V layout;
- no direct decode-to-`uint4` helper;
- no explicit K/V offset reuse helper.

Next target:

Move away from these tiny local decode arithmetic changes. The next experiment
should inspect a larger instruction source, preferably with NCU before/after:

1. softmax/shared-prob fanout and synchronization;
2. PV accumulator/writeback bookkeeping;
3. descriptor construction/base metadata load sharing at warp or CTA scope.

## 48. Direct-Path log2 Softmax Experiment

Question:

Can the FA2-direct path reduce softmax instruction cost by matching FA2's common
log2-domain softmax style: pre-scale QK scores by `log2(e)`, use `exp2f`, and
convert the final LSE back to natural-log units?

Change tested:

- Added `scale_log2 = scale * log2(e)` inside `UseFa2Direct`.
- Computed row maxima in log2 score units.
- Replaced the direct-path `__expf` calls for online softmax with `exp2f`.
- Converted final partition LSE back with `softmax_m * ln(2)`.
- Kept output normalization, PV MMA, vector shared stores, and split/reduce
  interface unchanged.

Correctness:

- `tests/v1/attention/test_byte_v2_layout.py::test_byte_v2_paged_decode_attention_split_k_gqa_fa2_like_cuda_matches_raw_reference`
  passed: 7 passed, 16 warnings.

CUDA-event result, 4096/p256:

| path | median |
| --- | ---: |
| retained vector-store baseline | 0.057344 ms |
| log2 softmax experiment | 0.057344 ms |
| restored retained vector-store binary | 0.057344 ms |

Artifacts:

- `profiles/byte_v2_fa2_direct_log2_softmax_4096_p256_vs_fa2_20260626.jsonl`
- `profiles/byte_v2_fa2_direct_vector_store_restored_after_log2_4096_p256_vs_fa2_20260626.jsonl`

Decision:

Rejected and reverted. The change was correct, but it did not move the 4096/p256
event median and therefore did not clear the 5% retention threshold.

Interpretation:

For this direct decode shape, softmax exponent choice is not the dominant local
bottleneck. The compiler/hardware cost difference between `__expf` and `exp2f`
is hidden by other work, or the remaining time is dominated by the CUTE QK/PV
mainloop, payload staging, and output/reduce interface rather than exponent
throughput.

Current retained state remains unchanged:

- 16 dims/thread payload decode;
- vector shared stores;
- natural-log online softmax in the direct path;
- no direct decode-to-`uint4`;
- no explicit K/V offset-reuse helper.

Next target:

Further tiny arithmetic rewrites are unlikely to help. The next useful step is
not another scalar expression tweak, but a structural profile/experiment around
the direct mainloop:

1. count instruction contribution from QK/PV CUTE plumbing versus payload
   staging using source-level NCU sections or SASS correlation;
2. test a smaller `kBlockM`/effective-row direct path to reduce inactive rows
   from the current 64-row CUTE tile when only `kQPerKv == 4` rows are useful;
3. test whether tmp_out / exp_sums writeback and reduce shape can be collapsed
   for the single-token decode benchmark.

## 49. Current ByteV2 vs FA2 Reprofile

Question:

After reverting the rejected micro-experiments, what is the current performance
and counter gap between the retained ByteV2 direct kernel and raw FA2?

Setup:

- GPU: `CUDA_VISIBLE_DEVICES=6`
- Shape: `num_seqs=1`, `num_heads=32`, `num_kv_heads=8`, `head_dim=128`,
  `block_size=16`, BF16.
- ByteV2 path: direct no-fallback/no-outlier, GQA4 packed, QK MMA, 16
  dims/thread payload decode, vector shared stores.
- FA2 path: vLLM paged FA2.
- Event sweep uses best ByteV2 partition among p64/p128/p256 for each sequence
  length.
- NCU breakdown uses 4096 tokens with ByteV2 p256.

CUDA-event sweep:

| seq len | best ByteV2 | best p | raw FA2 | ByteV2 / FA2 |
| ---: | ---: | ---: | ---: | ---: |
| 256 | 0.023552 ms | 64 | 0.062464 ms | 0.377x |
| 512 | 0.022528 ms | 64 | 0.060416 ms | 0.373x |
| 1024 | 0.025600 ms | 64 | 0.060416 ms | 0.424x |
| 2048 | 0.038912 ms | 128 | 0.062464 ms | 0.623x |
| 4096 | 0.057344 ms | 256 | 0.063488 ms | 0.903x |

4096/p256 NCU kernel-level comparison:

| metric | ByteV2 main | ByteV2 reduce | FA2 splitkv | FA2 combine |
| --- | ---: | ---: | ---: | ---: |
| NCU duration | 64.320 us | 6.624 us | 43.232 us | 13.472 us |
| grid | 128 CTA | 1024 CTA | 128 CTA | 8 CTA |
| registers/thread | 175 | 24 | 252 | 52 |
| shared/block | 49.152 KB static | 0 | 81.920 KB dynamic | 320 B static |
| active warps | 5.90 | 35.73 | 4.01 | 3.92 |
| active warp pct | 12.28% | 74.44% | 8.36% | 8.17% |
| memory throughput | 36.91% | 27.27% | 70.20% | 3.28% |
| SM throughput | 21.78% | 30.07% | 24.40% | 0.50% |
| total instructions | 5.893M | 0.782M | 1.408M | 0.028M |
| HMMA instructions | 262.1K | 0 | 262.1K | 0 |
| global load inst | 76.3K | 16.4K | 3.1K | 0.5K |
| global store inst | 4.6K | 4.1K | 0.9K | 0.05K |
| shared load inst | 0 | 0 | 8.2K | 0.5K |
| shared store inst | 65.5K | 0 | 16.4K | 0.03K |
| integer thread inst | 116.4M | 10.4M | 11.7M | 0.42M |
| DRAM read | 15.180 MB | 359.424 KB | 19.301 MB | 322.176 KB |
| DRAM write | 2.953 MB | 15.488 KB | 2.411 MB | 128 B |

Aggregate NCU view:

| metric | ByteV2 total | FA2 total | ByteV2 / FA2 |
| --- | ---: | ---: | ---: |
| NCU duration | 70.944 us | 56.704 us | 1.25x |
| total instructions | 6.675M | 1.436M | 4.65x |
| integer thread inst | 126.8M | 12.1M | 10.49x |
| global load inst | 92.7K | 3.6K | 25.74x |
| global store inst | 8.7K | 0.9K | 9.22x |
| shared store inst | 65.5K | 16.4K | 3.99x |
| HMMA instructions | 262.1K | 262.1K | 1.00x |
| DRAM read | 15.539 MB | 19.624 MB | 0.79x |
| DRAM write | 2.968 MB | 2.411 MB | 1.23x |

Artifacts:

- `profiles/byte_v2_current_vs_fa2_event_sweep_20260626.jsonl`
- `profiles/ncu_byte_v2_current_main_default_4096_p256_reprofile_20260626.txt`
- `profiles/ncu_byte_v2_current_main_counters_4096_p256_reprofile_20260626.txt`
- `profiles/ncu_byte_v2_current_reduce_default_4096_p256_reprofile_20260626.txt`
- `profiles/ncu_byte_v2_current_reduce_counters_4096_p256_reprofile_20260626.txt`
- `profiles/ncu_fa2_current_splitkv_default_4096_reprofile_20260626.txt`
- `profiles/ncu_fa2_current_splitkv_counters_4096_reprofile_20260626.txt`
- `profiles/ncu_fa2_current_combine_default_4096_reprofile_20260626.txt`
- `profiles/ncu_fa2_current_combine_counters_4096_reprofile_20260626.txt`

Interpretation:

- CUDA event remains favorable for ByteV2 across the tested 256-4096 range.
  The advantage is largest at short context and narrows at 4096.
- NCU replay duration should not be treated as the same quantity as CUDA event
  timing. Under NCU replay, ByteV2 total is still longer than FA2 total, but the
  counter shape is the more useful result.
- Both main kernels execute the same HMMA count, so the core tensor-math work is
  aligned.
- ByteV2's remaining gap is instruction-side, not HBM byte-side:
    - about 4.65x total instructions;
    - about 10.49x integer thread instructions;
    - about 25.74x global load instructions;
    - about 3.99x shared store instructions.
- ByteV2 reads less DRAM than FA2 because the cache is compressed, but spends
  many more instructions unpacking/staging it.

Next target:

The next optimization should be structural, not another scalar arithmetic tweak:

1. reduce inactive work from the direct 64-row CUTE tile when only 4 GQA rows
   carry useful data;
2. split the NCU/SASS contribution of payload staging vs QK/PV CUTE plumbing;
3. consider a specialized single-token decode output path that reduces the
   ByteV2 reduce/global writeback overhead without changing correctness.

## 50. Direct16 Effective-M Tile Experiment

Question:

Can the FA2-direct path reduce inactive QK/PV work by using a 16-row CUTE tile
instead of the previous 64-row tile, since only `kQPerKv == 4` rows carry useful
GQA data?

Change:

- Changed the direct CUTE QK/PV traits from 4-warp / 64M to 1-warp / 16M.
- Replaced direct-path `Shape<64, ...>` fragments with `kDirectBlockM == 16`.
- Kept 128 CTA threads for payload staging, but restricted CUTE QK/PV compute
  and direct writeback to warp 0.
- Kept the compressed cache format, 16 dims/thread payload decode, vector
  shared stores, split/reduce interface, and backend partition heuristic
  unchanged.

Why the backend heuristic was left unchanged:

- The real backend only enables GQA packed direct by default at
  `max_seq_len >= 2048`.
- The existing direct heuristic already chooses the best measured direct16
  partitions for that default range:
    - 2048 -> p128;
    - 4096 -> p256.
- Extra p16/p32/p64 probing was only used to verify that short-context behavior
  is not inherently worse if direct is forced below the default min length.

Build/test:

```bash
cmake --build build-bytev2-stable --target _C_stable_libtorch -j 32
cp build-bytev2-stable/_C_stable_libtorch.abi3.so vllm/_C_stable_libtorch.abi3.so
CUDA_VISIBLE_DEVICES=6 .venv/bin/python -m pytest \
  tests/v1/attention/test_byte_v2_layout.py::test_byte_v2_paged_decode_attention_split_k_gqa_fa2_like_cuda_matches_raw_reference -v
```

Result:

- Build passed under `sm_86`.
- Targeted CUDA correctness test: `7 passed, 16 warnings`.
- The only CUDA compile warning in this build was the pre-existing unused
  `byte_v2_reshape_and_cache_kernel` warning.

4096/p256 CUDA-event check:

| variant | median | vs previous retained |
| --- | ---: | ---: |
| previous retained 64M direct | 0.057344 ms | baseline |
| direct16 effective-M tile | 0.052224 ms | +8.9% |
| raw FA2 in same direct16 run | 0.064512 ms | reference |

Context sweep after direct16:

| seq len | best ByteV2 direct16 | best p | raw FA2 | ByteV2 / FA2 |
| ---: | ---: | ---: | ---: | ---: |
| 256 | 0.025600 ms | 128 | 0.065536 ms | 0.391x |
| 512 | 0.025600 ms | 64/128 | 0.063488 ms | 0.403x |
| 1024 | 0.025600 ms | 64/128 | 0.064512 ms | 0.397x |
| 2048 | 0.032768 ms | 128 | 0.066560 ms | 0.492x |
| 4096 | 0.052224 ms | 256 | 0.067584 ms | 0.773x |

Short-context partition probe with direct forced below the backend default:

| seq len | best direct16 with p16/p32/p64/p128 | best p |
| ---: | ---: | ---: |
| 256 | 0.022528 ms | 32 |
| 512 | 0.023552 ms | 32/64 |
| 1024 | 0.022528 ms | 64 |

4096/p256 main-kernel NCU:

| metric | 64M direct | direct16 | delta |
| --- | ---: | ---: | ---: |
| NCU duration | 64.320 us | 56.960 us | -11.4% |
| registers/thread | 175 | 186 | +11 |
| shared/block | 49.152 KB | 36.864 KB | -25.0% |
| active warps | 5.90 | 5.76 | -2.3% |
| total instructions | 5.893M | 3.607M | -38.8% |
| HMMA instructions | 262.1K | 65.5K | -75.0% |
| global load inst | 76.3K | 76.3K | 0.0% |
| global store inst | 4.6K | 4.6K | 0.0% |
| shared store inst | 65.5K | 41.0K | -37.5% |
| integer thread inst | 116.4M | 84.0M | -27.8% |
| DRAM read | 15.180 MB | 14.830 MB | -2.3% |
| DRAM write | 2.953 MB | 2.698 MB | -8.6% |

Artifacts:

- `profiles/byte_v2_fa2_direct16_4096_p256_vs_fa2_20260626.jsonl`
- `profiles/byte_v2_fa2_direct16_sweep_vs_fa2_20260626.jsonl`
- `profiles/byte_v2_fa2_direct16_short_partition_sweep_vs_fa2_20260626.jsonl`
- `profiles/ncu_byte_v2_fa2_direct16_main_default_4096_p256_20260626.txt`
- `profiles/ncu_byte_v2_fa2_direct16_main_counters_4096_p256_20260626.txt`

Decision:

Retained. This is the first structural direct-mainloop change after vector
shared stores that clears the 5% threshold. It directly validates the hypothesis
that the previous 64-row direct CUTE tile was doing substantial inactive row
work for GQA4.

Interpretation:

- The payload staging cost is still present: global load/store instruction
  counts did not change.
- The QK/PV tensor-core work now matches the actually useful row count much
  better: HMMA count dropped by 75%.
- Total instructions and integer instructions dropped substantially, but
  integer work is still high because payload decode/staging and CUTE plumbing
  remain instruction-heavy.
- Register pressure increased from 175 to 186 regs/thread. This did not hurt
  the tested 4096/p256 event time, but it should be watched in future changes.

Updated retained state:

- direct no-fallback/no-outlier path;
- 16 dims/thread fixed payload decode;
- vectorized shared stores;
- direct16 1-warp / 16M effective CUTE QK/PV tile;
- existing partition heuristic unchanged.

Next target:

The next useful single-point experiment should target the still-unchanged
payload staging side:

1. reduce descriptor/base/payload address work or global load instruction count;
2. reduce the remaining `41.0K` shared store instructions;
3. investigate whether the direct16 path can lower the new 186-register pressure
   without undoing the instruction reduction.

## 51. Grouped-Q FA2-Direct Kernel Family (2026-06-26)

Change:

- Converted the retained GQA4 FA2-direct split-k kernel into a grouped-Q
  template family.
- Added compile-time specializations for `q_per_kv` in
  `{1, 2, 4, 8, 16, 32}`.
- Kept the current ByteV2 V4 page layout fixed at 8 KV heads. This change does
  not yet add arbitrary `num_kv_heads` page-layout instances.
- For `q_per_kv <= 16`, one CTA group covers all Q rows for one KV head.
- For `q_per_kv == 32`, one KV head is split into two grouped-Q CTAs with
  `QGroupTile=16`.
- Non-direct GQA-packed paths remain restricted to the previous GQA4 behavior.

Files changed:

- `csrc/libtorch_stable/byte_v2/byte_v2_ops.cu`
- `vllm/v1/attention/backends/byte_v2_attn.py`
- `tests/v1/attention/test_byte_v2_layout.py`

Implementation notes:

- `QHeadsPerKv` is now the model ratio.
- `QGroupTile` is the number of Q rows processed by one CTA group.
- The direct kernel's grid-x dimension is now
  `num_kv_heads * ceil_div(QHeadsPerKv, QGroupTile)`.
- `stats_base` and `tmp_out` offsets are based on
  `kv_head * QHeadsPerKv + q_group_base`, so split/reduce still sees the normal
  `[num_heads]` workspace layout.
- Invalid rows in a partial group are guarded with `valid_q_rows`, although the
  currently enabled set only needs a partial-group guard for future non-power
  ratios.

Build/test:

```bash
cmake --build build-bytev2-stable --target _C_stable_libtorch -j 32
cp build-bytev2-stable/_C_stable_libtorch.abi3.so vllm/_C_stable_libtorch.abi3.so
.venv/bin/pre-commit run ruff-check --files \
  vllm/v1/attention/backends/byte_v2_attn.py \
  tests/v1/attention/test_byte_v2_layout.py
CUDA_VISIBLE_DEVICES=7 .venv/bin/python -m pytest \
  tests/v1/attention/test_byte_v2_layout.py::test_byte_v2_gqa_fa2_direct_allows_grouped_q_per_kv \
  tests/v1/attention/test_byte_v2_layout.py::test_byte_v2_split_k_fa2_direct_grouped_q_cuda_matches_raw_reference \
  tests/v1/attention/test_byte_v2_layout.py::test_byte_v2_paged_decode_attention_split_k_gqa_fa2_like_cuda_matches_raw_reference \
  -v
```

Result:

- Build passed under `sm_86`.
- Ruff check passed.
- Targeted pytest: `19 passed, 16 warnings`.
- Covered CUDA correctness for `q_per_kv=1/2/4/8/16/32` on the FA2-direct
  grouped-Q path.
- Existing GQA4 FA2-like regression cases still pass.

GQA4 4096/p256 performance sanity after template conversion:

| variant | median | min | p90 |
| --- | ---: | ---: | ---: |
| ByteV2 grouped-Q direct template | 0.051200 ms | 0.050176 ms | 0.052224 ms |
| raw FA2 in same run | 0.065536 ms | 0.062464 ms | 0.071680 ms |

Artifact:

- `profiles/byte_v2_grouped_direct_template_sanity_4096_p256_20260626.jsonl`

Decision:

Retained. The grouped-Q template family generalizes the direct path without a
GQA4 performance regression. This is still not a fully arbitrary MHA/GQA page
layout implementation: the first retained scope is generic `q_per_kv` on the
existing 8-KV ByteV2 layout.

Next target:

Add a separate page-layout/reshape/cache dispatch layer if true MHA with
`num_kv_heads == num_heads` and non-8 KV-head layouts are required. Within the
current 8-KV layout, the next useful benchmark is a ratio sweep over
`q_per_kv=1/2/4/8/16/32` to decide which specializations should be enabled by
default beyond GQA4.

## 52. Grouped-Q Ratio Sweep (2026-06-26)

Scope:

- Existing ByteV2 V4 layout with `num_kv_heads=8`.
- `q_per_kv` sweep: `1/2/4/8/16/32`.
- Context sweep: `256/512/1024/2048/4096`.
- Grouped-Q FA2-direct partition sweep: `p32/p64/p128/p256`.
- Generic no-outlier split-k baseline partition sweep:
  `p16/p32/p64/p128/p256`.
- Raw FA2 was measured in the grouped-Q direct runs as a raw-KV reference.

Commands:

```bash
for q in 1 2 4 8 16 32; do
  heads=$((8*q))
  CUDA_VISIBLE_DEVICES=7 .venv/bin/python scripts/byte_v2_decode_microbench.py \
    --seq-lens 256 512 1024 2048 4096 \
    --num-heads ${heads} --num-kv-heads 8 \
    --partition-sizes 32 64 128 256 --compute-block-ns 64 \
    --warmup 20 --iters 80 --no-outlier-inputs --assume-no-outlier \
    --gqa-packed --gqa-fa2-like --gqa-fa2-qk-mma --gqa-fa2-direct \
    --include-flash \
    --output-jsonl profiles/byte_v2_grouped_direct_q${q}_sweep_20260626.jsonl
done

for q in 1 2 4 8 16 32; do
  heads=$((8*q))
  CUDA_VISIBLE_DEVICES=7 .venv/bin/python scripts/byte_v2_decode_microbench.py \
    --seq-lens 256 512 1024 2048 4096 \
    --num-heads ${heads} --num-kv-heads 8 \
    --partition-sizes 16 32 64 128 256 --compute-block-ns 64 \
    --warmup 20 --iters 80 --no-outlier-inputs --assume-no-outlier \
    --output-jsonl profiles/byte_v2_generic_nooutlier_q${q}_sweep_20260626.jsonl
done
```

4096-token summary, best partition per variant:

| q_per_kv | best direct | direct p | best generic | generic p | raw FA2 | direct/generic speedup | direct/FA2 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 0.043008 ms | 256 | 0.067584 ms | 16 | 0.122880 ms | 1.57x | 0.35x |
| 2 | 0.049152 ms | 256 | 0.100352 ms | 16 | 0.064512 ms | 2.04x | 0.76x |
| 4 | 0.052224 ms | 256 | 0.173056 ms | 16 | 0.063488 ms | 3.31x | 0.82x |
| 8 | 0.054272 ms | 256 | 0.313344 ms | 32 | 0.063488 ms | 5.77x | 0.85x |
| 16 | 0.074256 ms | 256 | 0.587776 ms | 64 | 0.063488 ms | 7.92x | 1.17x |
| 32 | 0.135168 ms | 256 | 1.119232 ms | 64 | 0.063488 ms | 8.28x | 2.13x |

Important context points:

| q_per_kv | 1024 best direct | 2048 best direct | 4096 best direct |
| ---: | ---: | ---: | ---: |
| 1 | 0.022528 ms | 0.025600 ms | 0.043008 ms |
| 2 | 0.022528 ms | 0.029696 ms | 0.049152 ms |
| 4 | 0.022528 ms | 0.031744 ms | 0.052224 ms |
| 8 | 0.022528 ms | 0.035840 ms | 0.054272 ms |
| 16 | 0.035840 ms | 0.051200 ms | 0.074256 ms |
| 32 | 0.053248 ms | 0.077824 ms | 0.135168 ms |

Interpretation:

- Grouped-Q FA2-direct is consistently better than the current generic
  no-outlier split-k baseline across the tested ratios and contexts.
- In the backend's default long-context range, the direct/generic speedup grows
  with `q_per_kv` because direct reuses decoded K/V across grouped Q rows.
- `q_per_kv <= 8` remains faster than raw FA2 at 4096 tokens in this sweep.
- `q_per_kv=16` and `q_per_kv=32` are still much faster than ByteV2 generic,
  but fall behind raw FA2 at long context. This points to an intra-CTA compute
  scaling limit in the current `1-warp / 16M` direct tile.
- The best direct partition is generally `p256` at 4096 and `p128` around 2048.
  Short contexts still prefer `p32/p64`.

Decision:

- Keep grouped-Q direct as the preferred ByteV2 path for the existing 8-KV
  layout when the no-fallback/no-outlier precondition holds.
- It is reasonable to enable `q_per_kv in {1, 2, 4, 8, 16, 32}` for ByteV2
  direct relative to the current ByteV2 generic baseline.
- For raw-FA2 parity work, prioritize `q_per_kv >= 16`: try `2-warp / 32M`,
  `4-warp / 64M`, or a high-q split that increases CTA-level parallelism while
  retaining decode reuse.

Artifacts:

- `profiles/byte_v2_grouped_direct_q1_sweep_20260626.jsonl`
- `profiles/byte_v2_grouped_direct_q2_sweep_20260626.jsonl`
- `profiles/byte_v2_grouped_direct_q4_sweep_20260626.jsonl`
- `profiles/byte_v2_grouped_direct_q8_sweep_20260626.jsonl`
- `profiles/byte_v2_grouped_direct_q16_sweep_20260626.jsonl`
- `profiles/byte_v2_grouped_direct_q32_sweep_20260626.jsonl`
- `profiles/byte_v2_generic_nooutlier_q1_sweep_20260626.jsonl`
- `profiles/byte_v2_generic_nooutlier_q2_sweep_20260626.jsonl`
- `profiles/byte_v2_generic_nooutlier_q4_sweep_20260626.jsonl`
- `profiles/byte_v2_generic_nooutlier_q8_sweep_20260626.jsonl`
- `profiles/byte_v2_generic_nooutlier_q16_sweep_20260626.jsonl`
- `profiles/byte_v2_generic_nooutlier_q32_sweep_20260626.jsonl`

## 53. q32 32M/2-Warp Direct Experiment (2026-06-26)

Scope:

- Only tested the high-`q_per_kv` direct path idea from section 52.
- Experimental change: `q_per_kv=32` used one `QGroupTile=32`,
  `BlockM=32`, `NumWarps=2` direct CTA instead of the retained two
  `QGroupTile=16`, `BlockM=16`, `NumWarps=1` CTA groups.
- `q_per_kv=4/16` were included as sanity runs.

Commands:

```bash
for q in 4 16 32; do
  heads=$((8*q))
  CUDA_VISIBLE_DEVICES=7 .venv/bin/python scripts/byte_v2_decode_microbench.py \
    --seq-lens 2048 4096 \
    --num-heads ${heads} --num-kv-heads 8 \
    --partition-sizes 64 128 256 --compute-block-ns 64 \
    --warmup 20 --iters 120 --no-outlier-inputs --assume-no-outlier \
    --gqa-packed --gqa-fa2-like --gqa-fa2-qk-mma --gqa-fa2-direct \
    --include-flash \
    --output-jsonl profiles/byte_v2_direct_highq_32m_q${q}_sanity_20260626.jsonl
done
```

Key results:

| q_per_kv | context | retained direct best | 32M/2-warp experiment best | result |
| ---: | ---: | ---: | ---: | --- |
| 4 | 4096 | 0.052224 ms | 0.051200 ms | noise-level improvement |
| 16 | 2048 | 0.051200 ms | 0.050176 ms | noise-level improvement |
| 16 | 4096 | 0.074256 ms | 0.073728 ms | noise-level improvement |
| 32 | 2048 | 0.077824 ms | 0.089088 ms | regression |
| 32 | 4096 | 0.135168 ms | 0.132096 ms | +2.3%, below 5% threshold |

Decision:

- Do not retain the 32M/2-warp q32 experiment.
- The long-context q32 gain is below the 5% keep threshold and the 2048-token
  q32 case regresses.
- Reverted q32 dispatch to the retained two-group shape:
  `QHeadsPerKv=32`, `QGroupTile=16`, direct `BlockM=16`, `NumWarps=1`.

Post-revert validation:

```bash
cmake --build build-bytev2-stable --target _C_stable_libtorch -j 32
cp build-bytev2-stable/_C_stable_libtorch.abi3.so \
  vllm/_C_stable_libtorch.abi3.so
CUDA_VISIBLE_DEVICES=7 .venv/bin/python -m pytest \
  tests/v1/attention/test_byte_v2_layout.py::test_byte_v2_split_k_fa2_direct_grouped_q_cuda_matches_raw_reference \
  tests/v1/attention/test_byte_v2_layout.py::test_byte_v2_paged_decode_attention_split_k_gqa_fa2_like_cuda_matches_raw_reference \
  -v
CUDA_VISIBLE_DEVICES=7 .venv/bin/python scripts/byte_v2_decode_microbench.py \
  --seq-lens 4096 --num-heads 256 --num-kv-heads 8 \
  --partition-sizes 256 --compute-block-ns 64 \
  --warmup 20 --iters 80 --no-outlier-inputs --assume-no-outlier \
  --gqa-packed --gqa-fa2-like --gqa-fa2-qk-mma --gqa-fa2-direct \
  --include-flash \
  --output-jsonl profiles/byte_v2_direct_q32_reverted_sanity_20260626.jsonl
```

Validation result:

- Build passed. The only nvcc warning was the existing unused
  `byte_v2_reshape_and_cache_kernel` warning.
- Targeted correctness: `13 passed, 16 warnings`.
- Reverted q32 4096 p256 sanity:
    - ByteV2 direct: `0.135680 ms`
    - Raw FA2: `0.064512 ms`

Artifacts:

- `profiles/byte_v2_direct_highq_32m_q4_sanity_20260626.jsonl`
- `profiles/byte_v2_direct_highq_32m_q16_sanity_20260626.jsonl`
- `profiles/byte_v2_direct_highq_32m_q32_sanity_20260626.jsonl`
- `profiles/byte_v2_direct_q32_reverted_sanity_20260626.jsonl`
