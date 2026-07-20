# ByteV2 FA2 Split-Only K/V Shared-Memory Alias

Date: 2026-07-19
GPU: NVIDIA A40 (SM86)
Representative workload: BF16, Q1, `Hq=32`, `Hkv=8`, `D=128`, sequence
length 4,099, FA2 split count 17

## 1. Outcome

The split-only K/V shared-memory alias is retained. It reduces the ByteV2 FA2
main kernel's dynamic shared memory from 80 KiB to 48 KiB without changing
the N128 attention tile, QK/PV MMA, masking, online softmax, split boundaries,
or the original FA2 combine kernel.

Twenty interleaved ByteV2/raw calls reduce the long-context ByteV2 main mean
from 44.711 us to 36.163 us (-19.12%) and its median from 44.704 us to
36.017 us (-19.43%). The mean paired main-kernel ratio improves from 1.270x
raw to 1.015x raw. Including the original combine launch, the mean ByteV2
main-plus-combine time is 44.432 us versus 44.709 us for raw in the candidate
process. The sequence-64 nonsplit control keeps the original 80 KiB allocation
and its median is unchanged at 9.216 us.

All oracle comparisons remain bitwise exact. Compute Sanitizer reports no
memory, synchronization, or race errors, and the full ByteV2 attention test
file reports 194 passed and one skipped.

## 2. Why 48 KiB Matters

The original FA2 D128/M64/N128/4-warp shared-memory allocation is:

- Q: `64 * 128 * 2 = 16 KiB`;
- K: `128 * 128 * 2 = 32 KiB`;
- V: `128 * 128 * 2 = 32 KiB`;
- total dynamic allocation: 80 KiB (81,920 B).

Nsight Compute includes another 1 KiB driver allocation, so the old total is
82,944 B per CTA. An A40 SM has 102,400 B available, which limits the old
kernel to one CTA per SM even though registers permit two.

For `Split=true`, K and V now alias one 32 KiB shared tile. Dynamic memory is
therefore `16 KiB Q + 32 KiB KV = 48 KiB`; the NCU total is 50,176 B. Two
resident CTAs consume 100,352 B, fitting below the 102,400 B SM limit. The
nonsplit specialization deliberately retains separate K/V tiles: its eight-CTA
small grid cannot benefit from a second resident CTA, while it would lose the
existing load/compute overlap.

## 3. Implementation

`VLLM_BYTE_V2_FA2_REUSE_KV_SMEM=1` exposes a compile-time loader capability.
The generic/raw loader defaults to false, so raw FA2 retains its original
shared layout and schedule. In the ByteV2 split specialization, `sV` points to
`sK` and the mainloop uses the following ownership sequence:

1. wait for and decode K, then synchronize the CTA;
2. execute the original QK MMA;
3. synchronize before V overwrites the shared K tile;
4. stage V asynchronously while masking/softcap executes in registers;
5. wait for and decode V, then synchronize the CTA;
6. execute the original online softmax and PV MMA;
7. synchronize before the next K overwrites V, then stage next K.

The extra barriers protect shared-tile lifetime only. They do not alter the
floating-point operation order. The raw and ByteV2 nonsplit specializations
compile through the original non-alias branch.

The launch template selects shared memory inside the existing compile-time
`Split` switch: 49,152 B for split ByteV2 alias and 81,920 B otherwise.

## 4. Nsight Systems A/B

Each process ran 20 interleaved ByteV2/raw oracle calls on the same idle A40.
Every output and LSE comparison reported zero bitwise mismatches.

| Q1 main kernel | Baseline mode 2 | K/V alias | Change |
| --- | ---: | ---: | ---: |
| Seq 4,099 mean | 44.711 us | 36.163 us | -19.12% |
| Seq 4,099 median | 44.704 us | 36.017 us | -19.43% |
| Paired ByteV2/raw mean ratio | 1.270x | 1.015x | -20.10% |
| Seq 64 mean | 9.267 us | 9.283 us | +0.17% |
| Seq 64 median | 9.216 us | 9.216 us | 0.00% |

At sequence 4,099, ByteV2 main plus its combine launch falls from 53.448 us
to 44.432 us (-16.87%). Candidate raw main plus combine is 44.709 us. The
combine implementation was not modified.

The reports named `candidate-*.nsys-rep` are an excluded build-dependency
diagnostic. Ninja had recorded zero header dependencies for the two ByteV2
CUDA objects, so the first build still launched with 81,920 B. The retained
measurement is `kv-alias-*.nsys-rep`, collected after explicitly cleaning and
recompiling the causal and noncausal BF16/D128 translation units.

## 5. Nsight Compute Diagnosis

NCU 2025.4.1 full/source replays compare the retained sideband-prefetch mode-2
baseline with the K/V alias candidate:

| Metric | Baseline mode 2 | K/V alias | Change |
| --- | ---: | ---: | ---: |
| Replay duration | 57.952 us | 44.832 us | -22.64% |
| Total shared memory/CTA | 82,944 B | 50,176 B | -39.50% |
| Registers/thread | 244 | 240 | -4 |
| Shared-memory occupancy limit | 1 CTA | 2 CTAs | 2x |
| Theoretical maximum active warps | 8.33% | 16.67% | 2x |
| Achieved active warps | 8.53% | 13.73% | +61.0% |
| Eligible warps/scheduler | 0.1461 | 0.2112 | +44.5% |
| DRAM read bytes | 12.773 MB | 12.777 MB | unchanged |
| DRAM read bandwidth | 220.4 GB/s | 285.0 GB/s | +29.3% |
| Local loads/stores | 0 / 0 | 0 / 0 | unchanged |

The grid remains 136 CTAs. `launch__waves_per_multiprocessor` changes from
1.619 to 0.8095 because NCU divides the same grid by the now-two resident CTAs
per SM. Instruction and byte counts are effectively unchanged; the gain comes
from latency hiding and increased concurrent work, not less attention work.

The tradeoff is visible in stall composition. The extra alias barriers and
the higher number of simultaneously active warps increase sampled MIO and
barrier pressure. Absolute runtime nevertheless falls substantially. Long
scoreboard remains the largest class (649 of 1,860 candidate samples), so the
next experiment should recover overlap inside the alias schedule rather than
further changing tile geometry.

## 6. Correctness and Safety

The bitwise oracle matrix covers:

- Q1 sequence lengths 64, 73, 79, and 4,099;
- forced-outlier Q2 and Q4 at sequence 67;
- a three-request forced-outlier batch;
- ragged `[73, 128, 4105]` with non-monotonic physical pages, four shared
  prefix blocks, forced outliers, and explicit split 4.

Every output and LSE mismatch count is zero and every maximum absolute error
is zero. Overflow-marker, fallback-bit, and real page-pool-exhaustion probes
all terminate nonzero, preserving V5's fail-closed contract.

Compute Sanitizer memcheck, synccheck, and racecheck each pass both the
73-token forced-outlier case and the 4,099-token split case. Racecheck reports
zero hazards, errors, and warnings. `cuobjdump` reports zero stack and local
memory for the active ByteV2 specializations, and NCU records zero local
loads/stores. Causal nonsplit/split use 247/244 registers per thread;
noncausal nonsplit/split use 245/240. Both split SASS images contain no
`LDL`/`STL` instructions.

The full test command is:

```bash
.venv/bin/python -m pytest tests/v1/attention/test_byte_v2_layout.py -q
```

Result: 194 passed, one skipped, 16 warnings in 200.25 seconds.

## 7. End-to-End Smoke

An eager independent-engine context-128 smoke keeps all eight token IDs exact:
ByteV2 reaches 32.151 tok/s and raw FA2 34.233 tok/s. This short context uses
the unchanged nonsplit path, so it is a correctness control rather than an
expected performance win.

Three compiled/CUDA-graph context-4,096 runs explicitly set
`BYTE_V2_DECODE_KERNEL=fa2` and `BYTE_V2_DECODE_RAW_FALLBACK=0`. Fresh engines
ran in B-R, R-B, and B-R order:

| Run | ByteV2 | Raw FA2 | Paired ByteV2/raw | Gap | Token match |
| ---: | ---: | ---: | ---: | ---: | --- |
| 1 | 21.7219 tok/s | 22.7921 tok/s | 0.953047 | -4.695% | 32/32 exact |
| 2 | 21.7965 tok/s | 22.8098 tok/s | 0.955579 | -4.442% | 32/32 exact |
| 3 | 21.7809 tok/s | 22.7561 tok/s | 0.957142 | -4.286% | 32/32 exact |
| Median | 21.7809 tok/s | 22.7921 tok/s | 0.955579 | -4.442% | 32/32 exact |

An Nsight Systems serving-path witness records 64 split ByteV2 FA2 launches
at 49,152 B dynamic shared memory and 240 registers, plus 32 intentional
nonsplit launches at 81,920 B and 245 registers. No legacy decode kernel is
present. The previous mode-2 three-run median paired ratio was 0.950436, so
the candidate closes another 0.514 percentage point of E2E gap. This roughly
0.54% paired-ratio gain is much smaller than the isolated 19% main-kernel gain
because model compute, cache updates, scheduling, and other kernels dominate
end-to-end time.

Two preliminary files were collected without explicitly selecting FA2 and
therefore measured the default legacy path. They were renamed
`legacy-control-e2e-*.jsonl` and are excluded from the candidate result.

## 8. Next Step

The highest-value follow-up is to overlap the existing online-softmax register
work with the aliased V tile's asynchronous trip. In the current alias branch,
masking overlaps V staging but softmax runs after the V wait/decode barrier.
Moving only the unchanged softmax call before that wait could hide part of the
remaining V latency while preserving QK, mask, softmax, PV, split, and combine
floating-point order. It must remain a separate A/B because the longer
register live range may change allocation or introduce spills.

If that fails, the profile does not support shrinking N128: N64 would change
online-softmax and split reduction order and would forfeit the current raw FA2
bitwise oracle. The remaining alternatives are format-level metadata changes
or a different committed-page contract, not another copy-count reduction.

## 9. Artifacts

```text
profile/bytev2-fa2-kv-alias-a40-20260719/
├── REPORT.md
├── harness/README.md
├── reports/
│   ├── baseline-*.nsys-rep / .sqlite
│   ├── kv-alias-*.nsys-rep / .sqlite
│   ├── full-kv-alias-seq4099-q1.ncu-rep
│   ├── source-kv-alias-seq4099-q1.ncu-rep
│   └── e2e-*.jsonl
└── analysis/
    ├── nsys_ab_summary.txt
    ├── validation_summary.txt
    ├── e2e_fa2_summary.txt
    ├── compare_baseline-prefetch2_vs_kv-alias.txt
    ├── metrics_key_*.txt / .json
    ├── stall_hotspots_*.txt
    └── pm_timeline_plots.txt
```
