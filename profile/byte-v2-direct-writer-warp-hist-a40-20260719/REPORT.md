# ByteV2 Direct Writer Warp Histogram

Date: 2026-07-19
GPU: NVIDIA A40 (SM86), driver 590.48.01
Source revision: `c08264a54c55aa893144b6387ec8670911230e33`
Candidate extension SHA256:
`5519463e2bdbaae192cca44f2787723efbb3ba5a400da02527031f6c64d05161`

## 1. Outcome

The warp-parallel histogram candidate is retained. It removes the direct
writer's thread-0 local histogram and reduces the complete 4K writer from a
CUDA-event median of 1,757.184 us to 425.472 us (-75.79%). In the serving
profile, `byte_v2_reshape_and_cache` falls from 55.075 ms to 14.572 ms per
request (-73.54%).

The operation-level gain survives end to end:

- synthetic Q1, context 4,096, 32 outputs: 21.7809 to 22.1989 tok/s (+1.92%);
- ShareGPT batch 4, Q4 speculative verification, 64 outputs/request: 94.9059
  to 95.7806 tok/s (+0.92%);
- relative to the legacy ByteV2 decode baseline, the retained stack reaches
  +5.17% on the ShareGPT workload.

The candidate preserves ByteV2 token IDs and speculative acceptance exactly.
It does not close the full raw-FA2 service gap: the retained ShareGPT result is
7.61% below raw and the exact-token Q1 result is 2.38% below raw. The remaining
headroom is therefore real, but direct-writer work is no longer the first E2E
priority. Incremental cache update is both a larger recurring cost and, in a
new full-FA2 long-output test, a correctness blocker due to row-0-frozen bases
and monotonic outlier-segment relocation.

## 2. Implementation

The change is confined to
`csrc/libtorch_stable/byte_v2/byte_v2_ops.cu` in
`byte_v2_reshape_and_cache_block_direct_kernel`:

1. a 128-bin shared-memory histogram replaces thread 0's 128-element local
   array;
2. all 128 threads load source BF16 values and update the histogram;
3. warp 0 evaluates the 121 valid eight-bin windows;
4. a packed max reduction preserves the original ordering: largest count,
   then smallest base on ties;
5. thread 0 retains the original row-major outlier enumeration, descriptor
   writes, pool allocation, overflow trap, fallback handling, and payload
   packing.

The launch ABI, `(256, 128)` grid for 4K, 128-thread CTA, page-run discovery,
cross-request physical-page behavior, and cache format are unchanged.

The candidate compiles with 48 registers/thread and 528 B static shared memory
for the active non-sideband specialization. The previous image used 40
registers/thread and 512 B stack/local storage. The candidate reports zero
stack and local memory in `cuobjdump`.

## 3. Isolated Performance

The saved-input CUDA-event comparison uses identical random BF16 K/V tensors,
slot mapping, and cache shape:

| Complete 4K operation | Baseline | Candidate | Change |
| --- | ---: | ---: | ---: |
| Mean | 1,756.692 us | 426.414 us | -75.73% |
| Median | 1,757.184 us | 425.472 us | -75.79% |
| Minimum | 1,736.704 us | 419.840 us | -75.83% |

An independent Nsight Systems sequence-4,099 witness records the direct kernel
at 1,687.749 us before and 329.892 us after (-80.45%). The difference from the
CUDA-event result is due to workload and timing protocol; both measurements
show the same large effect.

## 4. End-to-End Results

### ShareGPT Q4 speculative workload

Configuration: actual prompt lengths `[257, 512, 1024, 2072]`, batch 4,
three draft tokens, 64 output tokens/request, compiled execution with CUDA
graphs, prefix caching disabled, and approximately 42% draft acceptance.

| Mode | Runs (tok/s) | Median | Relative to candidate |
| --- | --- | ---: | ---: |
| Legacy ByteV2 decode | 91.2162, 90.9216 | 91.0689 | -4.92% |
| FA2 before this change | 95.0948, 94.7171 | 94.9059 | -0.91% |
| FA2 plus warp histogram | 95.7122, 96.0308, 95.7806 | 95.7806 | reference |
| Raw FA2 | 104.0661, 102.7305, 103.6742 | 103.6742 | +8.24% |

All five ByteV2 runs have identical token IDs, acceptance rate
`0.4193548387`, and mean acceptance length `2.2580645`. Raw takes a slightly
different numerical/token path and has acceptance rate `0.4215686275`, so the
ByteV2/raw comparison is a service-level throughput comparison, not a paired
token oracle.

The profiled full-prompt operation changes as follows:

| CUDA total per request | Before | Candidate | Change |
| --- | ---: | ---: | ---: |
| `byte_v2_reshape_and_cache` | 55.075 ms | 14.572 ms | -40.503 ms |
| Incremental `byte_v2_update_cache_raw_staging` | 60.096 ms | 60.330 ms | unchanged |

Raw FA2 spends 12.433 ms on all `reshape_and_cache_flash` calls in the same
workload. Subtracting its 2.340 ms bulk call leaves roughly 10.09 ms for raw
incremental writes, versus 60.33 ms for ByteV2. This stage comparison is an
attribution estimate because raw and ByteV2 accept different speculative
tokens, but it identifies a much larger recurring target than the remaining
approximately 12.23 ms bulk-writer gap.

### Exact-token synthetic Q1 control

Configuration: context 4,096, batch 1, 32 outputs, no speculation, compiled
execution with CUDA graphs, and prefix caching disabled.

| Backend | Runs (tok/s) | Median |
| --- | --- | ---: |
| Candidate ByteV2 | 22.1336, 22.2748, 22.1989 | 22.1989 |
| Raw FA2 | 22.6940, 22.7881 | 22.7410 |

All 32 token IDs match. Candidate ByteV2 is 2.38% below raw. Its bulk writer
uses 14.497 ms versus 2.296 ms for raw, a remaining 12.20 ms fixed difference.
Even eliminating that difference completely would improve the 1.442 s median
request by only about 0.85%; recurring decode/update overhead must be improved
to close the rest.

The full-FA2 output-length controls make the fixed/recurring split explicit:

| Outputs | ByteV2 wall | Raw wall | Wall gap | Token result |
| ---: | ---: | ---: | ---: | --- |
| 2 | 0.6301 s | 0.6211 s | +8.94 ms | 2/2 exact |
| 32 | 1.4415 s | 1.4072 s | +34.36 ms | 32/32 exact |
| 64 | 2.2933 s | 2.2408 s | +52.54 ms | 64/64 exact |

Least-squares fits are
`T_byte = 0.605532 + 0.0268241 * (N - 1)` seconds and
`T_raw = 0.595804 + 0.0261224 * (N - 1)` seconds, both with R-squared above
0.99997. ByteV2 therefore carries approximately 9.73 ms more fixed cost and
0.702 ms more per decode step (+2.69%). The recurring term agrees with the
earlier eager cache-update delta of about 0.61 ms/decode step across 32 layers,
which explains roughly 87% of the fitted step gap. This is the strongest E2E
evidence that the next target should be the single-token/incremental writer
rather than another direct-writer micro-optimization.

## 5. Nsight Compute Diagnosis

The profile commands were:

```bash
/opt/nvidia/nsight-compute/2025.4.1/target/linux-desktop-glibc_2_11_3-x64/ncu \
  --set full --section PmSampling --section PmSampling_WarpStates \
  --kernel-name regex:byte_v2_reshape_and_cache_block_direct_kernel \
  --launch-skip 20 --launch-count 1 --force-overwrite \
  --export profile/byte-v2-direct-writer-warp-hist-a40-20260719/reports/full_candidate_4k \
  .venv/bin/python \
  profile/byte-v2-direct-writer-warp-hist-a40-20260719/harness/direct_writer_profile.py \
  --tokens 4096 --warmup 20

/opt/nvidia/nsight-compute/2025.4.1/target/linux-desktop-glibc_2_11_3-x64/ncu \
  --set source --section SourceCounters \
  --kernel-name regex:byte_v2_reshape_and_cache_block_direct_kernel \
  --launch-skip 20 --launch-count 1 --force-overwrite \
  --export profile/byte-v2-direct-writer-warp-hist-a40-20260719/reports/source_candidate_4k \
  .venv/bin/python \
  profile/byte-v2-direct-writer-warp-hist-a40-20260719/harness/direct_writer_profile.py \
  --tokens 4096 --warmup 20
```

NCU replay duration is 477.600 us. Replay overhead makes this longer than the
CUDA-event median, so it is used for bottleneck attribution rather than the
primary latency result.

| Metric | Candidate |
| --- | ---: |
| Grid / block | 32,768 CTAs / 128 threads |
| Registers / static shared | 48 / 528 B |
| Theoretical / achieved occupancy | 83.33% / 80.01% |
| Active warps per SM | 38.40 |
| Issue active | 78.56% |
| SM throughput | 76.26% |
| DRAM throughput | 15.01% |
| DRAM read / write | 33.909 MB / 15.933 MB |
| L1 / L2 hit rate | 89.68% / 52.39% |
| Local loads / stores | 0 / 0 |
| ALU / LSU pipe active | 56.79% / 43.19% |

The old serial/local-memory failure mode has been removed. The candidate is
now highly occupied and keeps schedulers busy. Sampled stalls are distributed
across wait (23.01%), long scoreboard (20.66%), barrier (15.57%), and
not-selected (15.41%). The largest source hotspots are repeated slot/page-run
loads at lines 5597--5615 and the synchronization before payload packing at
line 5781.

The remaining direct-writer inefficiency is structural rather than another
histogram tweak. NCU reports only 10.8/32 useful bytes per global-load sector
and 18.2/32 per global-store sector; 786,432 sectors are excessive, 9% of the
total. A page-owned encoder or precomputed source-page map could reduce the
duplicated slot-map traversal and improve coalescing. NCU estimates an 8.416%
kernel-level opportunity from excessive global sectors, but even a perfect
direct writer now has less than 1% Q1 E2E headroom at 32 outputs.

## 6. Correctness and Safety

- Full test: `194 passed, 1 skipped` in
  `tests/v1/attention/test_byte_v2_layout.py`.
- Five focused CUDA tests pass for V5 payload, 3/16-token parity, cross-page
  behavior, and unaligned request boundaries.
- Twenty deterministic tie probes always choose base 3 and retain the original
  row-major 128-entry outlier order.
- Compute Sanitizer `synccheck` and `memcheck` both report zero errors on the
  4K harness.
- `clang-format --dry-run --Werror` and `git diff --check` pass.

A raw cache-byte comparison found six differing bytes in one page: two valid
pool segments exchanged allocation indices because independent CTAs race on
the page allocator. This ordering was already nondeterministic and is not a
logical cache difference. Decoding all 32,768 K/V elements on that page gives
zero mismatches. Therefore the supported claim is logical decode parity and
E2E token parity, not whole-cache byte-for-byte identity.

## 7. New Long-Output Blocker

A current full-FA2 4K-prompt, batch-1, non-speculative run completes 64 output
tokens in two independent runs, but the 128-output run fails after 75 outputs.
With `CUDA_LAUNCH_BLOCKING=1` and eager execution, the failure is localized to
`byte_v2_update_cache_single_token_fused` while attempting the next cache
write at scheduler state `num_computed_tokens=4170`, `num_output_tokens=75`.
Raw FA2 completes 128 outputs in 3.9077 and 3.9030 seconds. The ByteV2 and raw
token prefixes are exact through the independently completed 72-token control.

This is not caused by the retained direct-writer change: the failing fused
single-token kernel is a separate path, and newly allocated decode pages were
never touched by the full-prompt direct writer. Static inspection shows the
cause: each new page chooses tile bases from row 0 and never rebases them as
later rows arrive. When a tile's required power-of-two segment grows, the fused
updater also allocates a replacement and copies live entries without reclaiming
the old segment. Both stale-base outlier growth and relocation waste contribute
to exhausting the page allocator.

Two current full-FA2 diagnostic traces quantify both failure mechanisms. The
eager launch-blocking trace reaches logical page 260, row 10, physical block
517, layer 11 with 924 compact live entries. Ten tiles require 357 entries of
replacement allocation, projecting 1,281 transient entries; its frozen-base
final live capacity is also 1,103. A compiled Inductor trace with CUDA-graph
replay disabled reaches page 517, row 12 with 798 compact live entries. Eleven
growing tiles request 260 replacement entries, so transient demand is 1,058,
although final live capacity after discarding old segments would be only 928.
The exact numeric path shifts the boundary, but both exceed the 1,024-entry
pool and reach the allocator's intentional fail-closed trap. A robust safe path
must therefore reselect bases when needed and must not retain superseded
segments during growth.

As a no-source-change control, forcing the manager to skip the fused n1 writer
and use the existing native raw-staging full-page commit completes 80 outputs.
Under the same full-FA2 compiled configuration, its two 64-output wall times
are 2.2866 and 2.2950 seconds, versus 2.2872 and 2.2994 seconds for the fused
writer. The difference is noise and all token IDs are exact. The staging path
therefore provides a measured safe baseline with no detectable compiled E2E
penalty at 64 outputs; it also rebases pages and reclaims dead pool segments.
It also completes 128 outputs in 4.0120 seconds (31.9044 tok/s), with all 128
tokens exact against both raw runs. This is 2.73% longer in wall time and 2.66%
lower in throughput than the 3.9054-second raw median, consistent with the
approximately 2.3% ByteV2/raw gap already measured at 32 and 64 outputs.

The correct next E2E target is therefore the incremental updater:

1. first route n1 updates through the existing full-page raw-staging commit;
2. validate at least 128 generated tokens and compare exact tokens to raw;
3. then restore a fused fast path with device-side rebase and compaction that
   keeps both live and transient segment demand within the page capacity;
4. profile and optimize that path, whose current serving cost is roughly
   50 ms/request above raw in the real Q4 workload.

Further direct-writer restructuring remains possible, but its measured E2E
ceiling is now smaller than both the incremental-update cost and the
long-output correctness requirement.

### 2026-07-19 follow-up

The blocker is resolved without restoring the unsafe in-place algorithm. The
old fused updater is default-off. A new one-token specialization performs
device-side prefix hydrate plus raw BF16 append, then reuses the existing full
metadata clear, warp-histogram commit, and release/flag kernels. This preserves
page rebase and compaction while reducing the safe update from seven traced
GPU operations to four.

The complete update improves by 27.0%--28.3% across page rows 0, 1, 4, 8, 12,
and 15. It completes compiled/CUDA-graph 128- and 256-output E2E with exact raw
FA2 token IDs. Three 256-output candidate runs have a 34.698 tok/s median and
are 1.33% behind paired raw FA2, versus a 3.10% gap for fresh generic-safe
runs. The new safe specialization is default-on; the crashing updater remains
default-off. Full evidence is in
`profile/byte-v2-safe-n1-update-baseline-a40-20260719/REPORT.md`.

## 8. Artifacts

```text
profile/byte-v2-direct-writer-warp-hist-a40-20260719/
├── REPORT.md
├── harness/direct_writer_profile.py
├── reports/
│   ├── full_candidate_4k.ncu-rep
│   ├── source_candidate_4k.ncu-rep
│   ├── direct_writer_{baseline,candidate}_cuda_event.json
│   ├── e2e_real_*.jsonl
│   ├── e2e_q1_*.jsonl
│   └── e2e_q1_out{80,128}_fa2_*_failure.log
└── analysis/
    ├── metrics_{all,key}_candidate_4k.{json,txt}
    ├── details_candidate_4k.txt
    ├── stall_hotspots_candidate_4k.txt
    └── pm_timeline_plots.txt
```
