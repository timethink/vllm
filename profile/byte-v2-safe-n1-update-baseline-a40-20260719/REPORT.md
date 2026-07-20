# ByteV2 Safe Single-Token Staging Update

Date: 2026-07-19

GPU: NVIDIA A40 (SM86), driver 590.48.01

Source revision: `c08264a54c55aa893144b6387ec8670911230e33`

Candidate extension SHA256:
`b2adacf547c96fb57c9e6d226fc118c6a433fa3dd13a8adc9f780627aa033e0c`

## 1. Outcome

The safe fused single-token staging path is retained and enabled by default.
The old in-place single-token updater remains disabled by default because it
can exhaust the V5 page allocator after roughly 75 generated tokens.

The retained path does not mutate an encoded page in place. For one appended
token it performs the following sequence:

1. one CUDA kernel reads the device-side slot, initializes staging slot zero,
   decodes the existing page prefix, and copies the new raw BF16 row;
2. the existing metadata-clear kernel resets the destination page metadata;
3. the existing warp-histogram commit fully rebases and compacts the page;
4. the existing release-plus-page-flags kernel returns allocator state to the
   empty state.

This reduces the update from seven traced GPU operations to four. Across six
page rows, the complete same-process update falls from 34.99--35.82 us to
25.32--25.83 us, or 27.0%--28.3%. The compiled 4K-context, 256-output E2E
candidate has a 34.698 tok/s median and remains 1.33% behind raw FA2. Against
fresh generic-safe runs it improves throughput by about 1.83%; against the
previously recorded, faster native-safe run it improves wall time by 0.68%.

All 128- and 256-output ByteV2/raw token IDs are exact. The focused CUDA Graph
test also appends 32 tokens across two physical pages and preserves the full
logical cache exactly.

## 2. Implementation

The CUDA specialization is
`byte_v2_hydrate_append_single_token_raw_staging_kernel` in
`csrc/libtorch_stable/byte_v2/byte_v2_ops.cu`. It is selected only when the
native raw-staging operation receives a one-element slot mapping and
`fuse_single_token_staging=true`.

The specialization is deliberately narrower than the general staging path:

- V5 default policy only: 16-token pages, 16x16 codec tiles, eight K/V heads,
  and D128 K/V;
- exactly one staging slot;
- append-only decode semantics: rows `[0, target_row)` are preserved and the
  target row is replaced by the new BF16 token;
- negative CUDA Graph padding slots and positive out-of-range slots do not
  read or modify the cache;
- all tensor, dtype, device, shape, stride, policy, and commit-mode checks run
  before the first kernel launch.

The Torch schema adds a final
`bool fuse_single_token_staging=False` argument. The default preserves old
direct `torch.ops` call arity. Python enables the new safe specialization by
default through `BYTE_V2_FUSED_SINGLE_TOKEN_STAGING`; setting it to `0`
restores the chained safe path for controlled comparisons.

The final fused-stage image uses 29 registers/thread and reports zero stack,
local memory, or static shared memory in `cuobjdump`.

Generic Q1 decode normally does not consume page-unsafe flags. To let its n=1
update use the native four-kernel operation, `do_kv_cache_update` provides the
existing internal flags buffer when the safe specialization is enabled. The
flags remain exact and the last kernel combines their update with staging
release. The unrelated `BYTE_V2_NATIVE_SINGLE_TOKEN_UPDATE` path remains
default-off.

## 3. Baseline Measurement

The baseline harness profiles one V5 page at representative target rows. Its
safe operation is prepare, hydrate, append, metadata clear, warp-histogram
commit, and release/flag update, all dispatched through one native op.

Initial standalone medians were 35.63, 36.38, 36.81, 35.34, 36.43, and
35.72 us for rows 0, 1, 4, 8, 12, and 15. The final same-binary alternating
A/B results are the primary comparison:

| Target row | Baseline event | Candidate event | Change | Baseline wall | Candidate wall | Change |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 35.105 us | 25.434 us | -27.55% | 35.135 us | 25.467 us | -27.52% |
| 1 | 35.142 us | 25.487 us | -27.47% | 35.175 us | 25.519 us | -27.45% |
| 4 | 34.994 us | 25.479 us | -27.19% | 35.026 us | 25.511 us | -27.17% |
| 8 | 35.389 us | 25.829 us | -27.01% | 35.422 us | 25.859 us | -27.00% |
| 12 | 34.986 us | 25.315 us | -27.64% | 35.013 us | 25.351 us | -27.59% |
| 15 | 35.824 us | 25.690 us | -28.29% | 35.857 us | 25.721 us | -28.27% |

Each row uses 21 trials of 500 updates. Baseline and candidate run in the same
process, and their order reverses on every trial.

## 4. Nsight Systems Attribution

For row 8, a 200-update baseline NVTX range projects 11.842 ms of GPU work,
or 59.212 us/update under tracing, and contains 1,400 GPU operations, or seven
per update. Kernel averages are:

| Baseline stage | Average |
| --- | ---: |
| Commit | 8.379 us |
| Hydrate | 2.194 us |
| Append | 1.973 us |
| Release plus flags | 1.765 us |
| Prepare | 1.712 us |
| Metadata clear | 1.512 us |

The active kernels sum to about 17.53 us. The much larger projected range
shows that launch and inter-kernel gaps are the dominant remaining part of the
complete operation.

The fused candidate projects 8.004 ms for the same 200 updates, or
40.018 us/update, and contains exactly 800 GPU operations, or four per update.
This is a 32.42% traced-range reduction. Its kernel averages are:

| Candidate stage | Average |
| --- | ---: |
| Existing warp-histogram commit | 8.384 us |
| Fused hydrate/append stage | 2.154 us |
| Existing release plus flags | 1.764 us |
| Existing metadata clear | 1.513 us |

The active sum falls to about 13.82 us. Most of the complete-operation gain
therefore comes from removing three launches and their gaps, while the
expensive and already-validated commit remains unchanged.

An engine trace provides the serving-path witness. Before the generic-Q1
routing fix, it recorded 192 prepare/hydrate/append launches and zero fused
stage launches. After the fix, the trace records 96 fused stage, 96
warp-histogram commit, and 96 release-with-flags launches. Separate padded
compile-warmup shapes still use the general path, as intended; actual n=1
decode uses the fused path.

## 5. Nsight Compute Diagnosis

NCU 2025.4.1 profiles the unchanged row-8 commit because it is the largest
active kernel in both baseline and candidate. Full and source reports were
parsed with the report API rather than the CLI text table.

### Grid and occupancy

- Grid: 128 CTAs, block: 128 threads, approximately 0.127 waves/SM.
- Registers: 40/thread; shared memory: 1,552 B/CTA; no local spill.
- Theoretical occupancy: 100%; achieved occupancy: 12.15%.
- Active warps: 5.83/SM; eligible warps: 0.074/scheduler.

The occupancy gap is structural: a one-page update exposes only 128 CTAs on
an A40. It is not caused by registers or shared-memory capacity.

### Compute and memory throughput

- SM throughput: 2.14% of peak.
- DRAM read throughput: 0.754% of peak.
- Global load sectors/request: 6,930/3,346 = 2.07.
- Stores carry about 13.65 useful bytes per 32-byte sector.

Neither arithmetic throughput nor DRAM bandwidth is saturated. NCU's global
coalescing rule estimates only about a 3% kernel opportunity, too small to be
the first E2E target.

### Latency and source stalls

- Long-scoreboard stall ratio: 6.357.
- Barrier stall ratio: 5.868.
- Wait stall ratio: 2.025.
- Source sampling collected only seven samples because the kernel is short;
  five are barrier samples, with four at the commit synchronization near
  source line 6422 and one near line 6233.

The kernel is a tiny-grid latency problem, not a bandwidth or spill problem.
Rewriting its histogram again has limited E2E ceiling. Removing whole stages
from the surrounding update is the higher-confidence optimization, which the
Nsight Systems and same-process A/B results confirm.

## 6. Correctness and Safety

The retained candidate passed the following focused checks:

- rows 0, 1, 8, and 15 against the general safe update, with random,
  low-outlier, and non-contiguous-stride K/V inputs;
- V5 pages containing pooled overlay/outlier entries;
- negative padding slot and positive out-of-range slot with byte-exact cache
  non-mutation;
- old Torch schema call without the new final bool;
- invalid V5 policy and invalid commit mode, both verified to throw before
  changing any input, staging state, or cache byte;
- CUDA Graph capture and 32 replays across two physical pages, followed by a
  `-1` replay; logical K/V and allocator state match the direct reference;
- Compute Sanitizer row-15 `memcheck`: zero errors;
- Compute Sanitizer row-15 `synccheck`: zero errors;
- Compute Sanitizer row-15 `racecheck`: zero hazards, errors, or warnings.
- Complete `tests/v1/attention/test_byte_v2_layout.py`: 217 passed and one
  skipped.

V5 decode followed by BF16 reconstruction is bitwise identity for all BF16
bit patterns, including NaN, infinity, signed zero, and subnormal values: the
loader only expands BF16 bits into an FP32 representation and the staging
writer performs the inverse round-to-nearest-even integer narrowing.

The legal V5 writer cannot generate a tile fallback: a tile contains at most
256 values and fallback requires more than 256 outliers. A page whose final
power-of-two pooled segments require more than 1,024 entries still traps
fail-closed. That is the V5 format's existing representability limit, not a
new transient-relocation failure.

## 7. End-to-End Results

Configuration: Llama-3.1-8B-Instruct, BF16, context 4,096, batch 1, no
speculation, greedy decoding, compiled execution with CUDA graphs, prefix
caching disabled, `BYTE_V2_DECODE_KERNEL=fa2`, and raw fallback disabled.
The old unsafe updater is unset/off.

### 128 output tokens

| Mode | ByteV2 wall | Raw wall | ByteV2 throughput | Raw throughput | Paired gap | Token result |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Previous native-safe | 3.99552 s | 3.90325 s | 32.0359 tok/s | 32.7932 tok/s | -2.31% | 128 exact |
| Default safe fused staging | 3.97211 s | 3.90291 s | 32.2247 tok/s | 32.7961 tok/s | -1.74% | 128 exact |

The default-on candidate reduces ByteV2 wall time by 0.59% and closes 0.57
percentage point of the paired throughput gap in this run. An earlier run with
the same path explicitly enabled measured 3.97575 versus 3.90457 seconds and
also preserved all 128 tokens.

### 256 output tokens

| Mode | ByteV2 runs | Raw runs | Median ByteV2 | Median raw | Median paired throughput gap |
| --- | --- | --- | ---: | ---: | ---: |
| Fresh generic-safe | 7.51255, 7.51372 s | 7.27676, 7.28296 s | 34.0737 tok/s | 35.1655 tok/s | -3.10% |
| Fused candidate | 7.37785, 7.36839, 7.40060 s | 7.28157, 7.27057, 7.27946 s | 34.6984 tok/s | 35.1675 tok/s | -1.33% |

The candidate raises median throughput by about 1.83% over fresh generic-safe
and closes about 1.78 percentage points of its paired raw gap. The previously
recorded native-safe control was faster than the fresh generic-safe runs:
7.42868 s and 34.4611 tok/s versus raw's 7.27565 s and 35.1859 tok/s. Against
that stricter control, the candidate reduces median wall time by 0.68% and
closes about 0.73 percentage point of the raw throughput gap.

All nine saved 128/256 ByteV2/raw comparisons have identical output lengths,
exact token IDs, and `profile_token_ids_match=true` within each engine.

## 8. Decision and Remaining Headroom

The candidate is retained because it satisfies all three gates:

1. it replaces the crashing in-place updater with full-page rebase/compaction;
2. it improves the complete isolated update by at least 27% at every row;
3. it produces a repeatable E2E gain while preserving exact raw-FA2 tokens.

The remaining median 256-token gap to raw is about 1.33%. The commit alone is
8.38 us of the 25.3--25.8 us update, and the new fused stage is only 2.15 us.
Further single-kernel tuning therefore has a small E2E ceiling. The next
evidence-backed targets are either reducing the commit/release launch count or
addressing non-update E2E work; any follow-up should again use paired E2E data
as the retention gate.

## 9. Reproduction

Same-process alternating A/B:

```bash
CUDA_VISIBLE_DEVICES=6 .venv/bin/python \
  profile/byte-v2-safe-n1-update-baseline-a40-20260719/harness/safe_n1_update_profile.py \
  --row 8 --warmup 100 --iterations 500 --trials 21 --comparison
```

Candidate Nsight Systems trace:

```bash
CUDA_VISIBLE_DEVICES=6 nsys profile --trace=cuda,nvtx --sample=none \
  --cpuctxsw=none --force-overwrite=true \
  -o profile/byte-v2-safe-n1-update-baseline-a40-20260719/reports/nsys_fused_row8 \
  .venv/bin/python \
  profile/byte-v2-safe-n1-update-baseline-a40-20260719/harness/safe_n1_update_profile.py \
  --row 8 --warmup 100 --iterations 200 --profile --fused-stage
```

Sanitizer example:

```bash
CUDA_VISIBLE_DEVICES=0 /usr/local/cuda/bin/compute-sanitizer \
  --tool memcheck --error-exitcode=99 .venv/bin/python \
  profile/byte-v2-safe-n1-update-baseline-a40-20260719/harness/safe_n1_update_profile.py \
  --row 15 --warmup 0 --iterations 1 --profile --fused-stage
```

The full reports, parsed metrics, A/B JSON files, engine traces, and sanitizer
logs are stored under this profile directory. E2E JSONL/log artifacts are in
`/tmp/bytev2_e2e_target_20260719/fused_stage_candidate_20260719/`.
