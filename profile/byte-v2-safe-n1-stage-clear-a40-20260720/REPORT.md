# ByteV2 Safe Single-Token Two-Kernel Update

Date: 2026-07-20

GPU: NVIDIA A40 (SM86), driver 590.48.01

Source revision: `c08264a54c55aa893144b6387ec8670911230e33`

Candidate extension SHA256:
`d04c3cbc3f13b2bf4a73fffc22e1a596af849aa6c0751dbd1c042387e47dd38f`

## 1. Outcome

The safe V5 n=1 staging update can be reduced from four kernels to two without
changing decoded cache values, page-unsafe flags, allocator state, CUDA Graph
replay behavior or generated tokens. The two isolated launch fusions are
large: four-to-three improves row-8 wall time by 19.02%, and three-to-two
improves it by another 13.02%.

The combined candidate is not enabled by default. In the final same-native-
binary context-4096, 256-output E2E comparison, it improves median ByteV2
throughput from 34.6127 to 34.7125 tok/s, or only 0.2883%. That misses the
declared 0.5% E2E retention gate. The production default therefore remains the
previous safe four-kernel update; the tested candidate is retained only as an
explicit diagnostic/experimental path.

Set both variables to reproduce the two-kernel path:

```bash
export BYTE_V2_FUSED_SINGLE_TOKEN_COMMIT_RELEASE=1
export BYTE_V2_FUSED_SINGLE_TOKEN_STAGE_METADATA_CLEAR=1
```

Both variables default to off. The already-retained
`BYTE_V2_FUSED_SINGLE_TOKEN_STAGING` remains default-on.

## 2. Implementation

The prior four-kernel n=1 update was:

1. hydrate the encoded prefix and append the new BF16 row to raw staging;
2. clear destination metadata;
3. rebase and compact the complete page with the warp-histogram commit;
4. update page-unsafe flags and release staging state.

The first experiment folds step 4 into commit. The existing `overflow[0]`
word becomes a 128-CTA completion counter. Tile CTAs atomically aggregate K/V
unsafe bits, and the final CTA resets staging metadata and releases the page.
Invalid slots participate in the same completion protocol so graph padding
cannot leave the counter dirty.

The second experiment folds step 2 into the stage. The stage temporarily uses
`next_staging_slot[0]` as its 128-CTA completion counter. After every CTA has
finished reading the old encoded page, the last CTA cooperatively clears the
destination metadata and publishes one valid staging slot. Commit then starts
on the same stream and performs the existing full-page rebase/compaction.

The final path therefore launches only:

1. fused hydrate/append plus metadata clear;
2. fused commit plus flags plus allocator release.

It does not change QK, online softmax, PV, split selection, the FA2 mainloop or
the original FA2 combine kernel. It also does not weaken the V5 page-pool
fail-closed behavior.

The two optional schema arguments are trailing `False` defaults. The stage-
clear option is effective only when commit/release fusion is also effective;
commit/release fusion additionally requires fused metadata clear,
warp-parallel histogram and no serial-metadata bypass. The manager owns and
reuses the counters on one CUDA stream; callers must not concurrently share
one staging-manager state across streams.

## 3. Isolated Performance

Both comparisons use target row 8, 21 alternating trials and 500 updates per
mode per trial.

| Experiment | Baseline event | Candidate event | Change | Baseline wall | Candidate wall | Change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Four to three kernels | 27.117567 us | 21.956608 us | -19.0318% | 27.149990 us | 21.986646 us | -19.0178% |
| Three to two kernels | 22.394880 us | 19.474432 us | -13.0407% | 22.425938 us | 19.505242 us | -13.0237% |

The final two-kernel Nsys trace contains exactly 300 stage and 300 commit
launches. Their averages are 2.6499 and 8.8462 us. There is no standalone
metadata-clear or release kernel. Stage duration grows because it now runs the
cross-CTA ticket and metadata clear, but eliminating the graph node and gap
still saves about 2.92 us per isolated update.

## 4. Serving-Path Trace Attribution

An output-8 compiled engine trace records both the two-kernel specialization
and 96 instances of the generic prepare/hydrate/append/clear/commit/release
chain. Timestamp and CUDA Graph analysis shows that the generic chain is not
steady decode work:

- its 96 instances form three 32-layer passes during compilation, per-layer
  graph construction and graph validation;
- the last generic launch ends 1.618 seconds before the first real prefill;
- real 4096-token prefill uses the direct-cache writer in three 32-layer
  passes;
- the measured and profiler decode intervals each execute seven replays of
  full-model Graph 199, captured immediately after a 32-layer fused n=1 pass.

The trace uses graph-level replay tracing, so nodes inside Graph 199 are
collapsed. The preceding fused capture sequence and absence of generic work
from the real prefill/decode intervals establish that the candidate does hit
the serving decode path. Its small E2E gain is therefore an end-to-end ceiling,
not a routing miss.

The uncollapsed setup/capture summaries are:

| Kernel | Instances | Average |
| --- | ---: | ---: |
| Fused stage plus clear | 96 | 1.901 us |
| Fused commit plus release | 96 | 1.821 us |
| Generic staging clear | 96 | 1.508 us |
| Generic commit | 96 | 3.350 us |
| Generic release | 96 | 1.413 us |

## 5. End-to-End Results

Configuration: Llama-3.1-8B-Instruct, BF16, context 4,096, batch 1, no
speculation, greedy decoding, 256 output tokens, compiled execution with CUDA
Graphs, prefix caching disabled, `BYTE_V2_DECODE_KERNEL=fa2`, and raw fallback
disabled. Both modes use the same final native extension. The baseline sets
both new fusion controls to `0`; the candidate sets both to `1`.

| Mode | Run | ByteV2 wall | Raw wall | ByteV2 tok/s | Raw tok/s | Paired gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Four kernels | 1 | 7.361272 s | 7.261373 s | 34.7766 | 35.2550 | -1.3571% |
| Four kernels | 2 | 7.396134 s | 7.288012 s | 34.6127 | 35.1262 | -1.4619% |
| Four kernels | 3 | 7.400750 s | 7.284656 s | 34.5911 | 35.1424 | -1.5687% |
| Two kernels | 1 | 7.346305 s | 7.258059 s | 34.8474 | 35.2711 | -1.2012% |
| Two kernels | 2 | 7.374872 s | 7.286765 s | 34.7125 | 35.1322 | -1.1947% |
| Two kernels | 3 | 7.384639 s | 7.291309 s | 34.6666 | 35.1103 | -1.2638% |

Median comparison:

| Metric | Four kernels | Two kernels | Improvement |
| --- | ---: | ---: | ---: |
| ByteV2 wall | 7.396134 s | 7.374872 s | 21.262 ms / 0.2875% |
| ByteV2 throughput | 34.6127 tok/s | 34.7125 tok/s | 0.2883% |
| Median paired gap to raw | -1.4619% | -1.2012% | 0.2606 percentage point |
| Median absolute Byte/raw wall gap | 108.122 ms | 88.246 ms | 19.876 ms |

Every ByteV2/raw pair produces the same 256 token IDs. All six saved output
sequences are also mutually identical; the canonical JSON token-list SHA256 is
`1926b18b058324320c4bbdf574eb9cc3d7a71aba799ed6f4ca5f1d92d7d45c1c`.

## 6. Correctness and Safety

Focused tests cover both the three-kernel and two-kernel opt-in combinations:

- rows 0, 1, 8 and 15 with random, low-outlier and non-contiguous K/V;
- exact clean/K-only/V-only/K+V flags against an independent page scan;
- negative and positive-out-of-range slots with stale flags;
- preflight fail-before-mutation and old Torch schema arity;
- 32 CUDA Graph replays across two physical pages.

The final focused selection reports 33 passed. Compute Sanitizer reports:

- two-kernel row-0 synccheck: zero errors;
- two-kernel row-15 memcheck: zero errors;
- two-kernel row-0 racecheck over 100 updates: zero hazards, errors or
  warnings;
- CUDA Graph target under memcheck: zero errors.

The complete `tests/v1/attention/test_byte_v2_layout.py` run reports 236
passed and one skipped.

## 7. Decision and Next Target

The experiment demonstrates that the remaining update launches are real
isolated overhead but not the main E2E gap. Removing two of four update nodes
improves the isolated operation substantially while moving the complete
serving workload by only 0.288%. Further work on the same update tail has less
than the declared retention margin unless it also removes surrounding graph
or scheduler work.

The next structural target should therefore be the remaining raw-vs-ByteV2
step-level residual outside attention main plus combine. A synchronized
ByteV2/raw engine trace should attribute graph replay, cache-update and host
scheduler intervals before another kernel rewrite. Attention main plus combine
is already approximately tied with raw (44.432 versus 44.709 us), so it is not
the primary E2E target.

## 8. Artifacts

- `analysis/e2e_summary.json`
- `analysis/isolated_ab_summary.json`
- `analysis/nsys_engine_out8_cuda_gpu_kern_sum.csv`
- `analysis/nsys_engine_out8_cuda_gpu_trace_cuda_gpu_trace.csv`
- `analysis/nsys_fused_stage_clear_row8_cuda_gpu_kern_sum.csv`
- `harness/safe_n1_stage_clear_profile.py`
- `reports/e2e_pair_run{1,2,3}.jsonl`
- `reports/e2e_4kernel_pair_run{1,2,3}.jsonl`
- `reports/nsys_engine_out8.nsys-rep`
- `reports/nsys_fused_stage_clear_row8.nsys-rep`
- `reports/memcheck_stage_clear_row15.log`
- `reports/memcheck_stage_clear_cuda_graph.log`
- `reports/synccheck_stage_clear_row0.log`
- `reports/racecheck_stage_clear_row0.log`
