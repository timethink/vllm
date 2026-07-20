# ByteV2 FA2 E2E Length Sweep and Kernel Profile

Date: 2026-07-20

GPU: NVIDIA A40 (SM86), driver 590.48.01

Source revision: `c08264a54c55aa893144b6387ec8670911230e33`

Native extension SHA256:
`d04c3cbc3f13b2bf4a73fffc22e1a596af849aa6c0751dbd1c042387e47dd38f`

## 1. Outcome

The current ByteV2 FA2 path remains slower than raw FA2 at every tested E2E
context length, but the gap does not grow with context. It is largest in the
short-to-medium range and narrows below 1% at 16K:

- context 64--2,048: ByteV2 is 2.27%--2.66% slower;
- context 4,096--8,192: ByteV2 is 1.49%--1.57% slower;
- context 16,384: ByteV2 is 0.94% slower.

All 24 ByteV2/raw pairs generated exactly the same 256 token IDs. The token
lists also match across all three rounds and both profiler replays.

At context 1,024, a serving CUDA Graph replay attributes 214.7 us of the
310.5 us named ByteV2-vs-raw active-kernel gap to the four-kernel ByteV2 cache
update and 97.1 us to attention main plus combine. Thus the update chain is
the largest controllable short-context difference; the ByteV2 loader is the
second target.

No production source was changed in this run. Only profile harness snapshots,
raw reports, analysis scripts and this report were added under this run
directory.

## 2. E2E setup

The formal performance result is the median of three uninstrumented paired
runs. Backend order was Byte/raw in rounds 1 and 3 and raw/Byte in round 2.

- Model: Llama-3.1-8B-Instruct, BF16.
- Batch size: 1.
- Query mode: Q1, no speculative draft tokens.
- Output: 256 greedy tokens.
- Contexts: 64, 128, 512, 1,024, 2,048, 4,096, 8,192 and 16,384.
- Execution: compiled model with CUDA Graphs and size specialization.
- Prefix caching: disabled.
- ByteV2 attention: `BYTE_V2_DECODE_KERNEL=fa2`, raw fallback disabled.
- ByteV2 update: current safe default four-kernel n=1 path. Both experimental
  two-kernel fusion switches were explicitly disabled.

The saved harness hashes are:

- `byte_v2_speculative_profile.py`:
  `c84c6a059815ac390162eb22c892db2f761086a354858e4fdd889384f736cbbe`
- `byte_v2_fa2_oracle.py`:
  `84094237003d52039b826ddf85b124fcac050c8c2ad3e96c859bdeaffc95f83b`

## 3. E2E results

`Paired gap` is the median of each round's paired Byte/raw throughput ratio;
it is intentionally not recomputed from the two independently rounded median
throughputs.

| Context | ByteV2 tok/s | Raw tok/s | Paired gap | Byte/raw wall gap | Tokens |
| ---: | ---: | ---: | ---: | ---: | :---: |
| 64 | 37.1934 | 38.0584 | -2.2727% | 156.429 ms | exact |
| 128 | 36.8901 | 37.8998 | -2.6642% | 184.885 ms | exact |
| 512 | 36.4931 | 37.4694 | -2.5951% | 182.044 ms | exact |
| 1,024 | 36.0237 | 36.9430 | -2.4886% | 176.848 ms | exact |
| 2,048 | 35.0014 | 35.8257 | -2.4083% | 176.471 ms | exact |
| 4,096 | 33.2367 | 33.7839 | -1.5674% | 120.793 ms | exact |
| 8,192 | 29.7073 | 30.1453 | -1.4932% | 128.774 ms | exact |
| 16,384 | 23.7503 | 23.9835 | -0.9436% | 101.935 ms | exact |

The new 4K result is consistent with the preceding current-binary experiment:
the earlier paired gap was -1.4619%, versus -1.5674% here.

### Prefill proxy versus decode residual

Before each measured request, the harness runs the same prompt with one output
token and prefix caching disabled. Its `warmup_seconds` is a useful prefill
proxy, although it is not a separately instrumented TTFT metric. Subtracting
it from the 256-token measured wall time approximates 255 decode steps.

| Context | Prefill-proxy Byte/raw gap | Residual Byte/raw gap | Residual gap / decode step |
| ---: | ---: | ---: | ---: |
| 64 | +9.015 ms | +148.325 ms | +581.7 us |
| 128 | +11.050 ms | +174.712 ms | +685.1 us |
| 512 | +19.089 ms | +162.956 ms | +639.0 us |
| 1,024 | +26.455 ms | +150.393 ms | +589.8 us |
| 2,048 | +7.897 ms | +168.574 ms | +661.1 us |
| 4,096 | +8.598 ms | +112.195 ms | +440.0 us |
| 8,192 | +8.525 ms | +119.760 ms | +469.6 us |
| 16,384 | -27.852 ms | +122.403 ms | +480.0 us |

The dominant gap is therefore in repeated decode work, not one-time prefill.
At 16K the ByteV2 prefill proxy is faster while the decode residual remains
slower.

## 4. Interleaved FA2 attention traces

Nsight Systems 2025.5.2 traced 200 iterations per length in one process with
ByteV2 immediately followed by raw. The oracle uses the same deterministic KV
values, query, block table, split heuristic and original FA2 combine kernel.
The first iteration is excluded from paired means; medians are insensitive to
its lazy-runtime overhead.

| Sequence | Splits | Byte main | Raw main | Byte combine | Raw combine | Byte envelope | Raw envelope | Paired envelope gap |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1,024 | 8 | 14.176 us | 12.352 us | 5.536 us | 5.472 us | 20.512 us | 18.624 us | +1.889 us / +10.17% |
| 4,096 | 16 | 36.049 us | 35.456 us | 7.616 us | 8.497 us | 44.800 us | 44.800 us | -0.001 us / -0.00% |
| 16,384 | 19 | 104.353 us | 119.745 us | 9.761 us | 11.904 us | 115.217 us | 132.481 us | -17.345 us / -13.09% |

For all three lengths and all 200 iterations, `out_mismatch=0`,
`lse_mismatch=0`, and both maximum absolute differences are zero.

This direct oracle deliberately removes model and scheduler noise. Its pages
are deterministic clean/low-outlier synthetic pages, so the 16K advantage is
evidence that compressed KV bandwidth can win at long contexts, not a claim
that every real model page has the same outlier behavior. Formal serving
decisions remain based on the E2E table.

## 5. Serving CUDA Graph attribution at 1K

Separate ByteV2 and raw serving traces used `--cuda-graph-trace=node`, context
1,024 and eight output tokens. Graph 200 contains one validation replay and
two seven-step decode clusters; the table excludes validation and reports the
median of the 14 steady replays.

| Per 32-layer replay | ByteV2 | Raw | ByteV2 - raw |
| --- | ---: | ---: | ---: |
| CUDA Graph nodes | 417 | 321 | +96 |
| Attention main | 496.373 us | 396.915 us | +99.458 us |
| FA2 combine | 172.801 us | 175.457 us | -2.657 us |
| Attention total | 669.365 us | 572.292 us | +97.073 us |
| Cache update total | 295.043 us | 80.305 us | +214.738 us |
| Attention + update | 963.095 us | 652.613 us | +310.482 us |
| All GPU-kernel active time | 24.567 ms | 24.409 ms | +158.867 us |
| Graph envelope | 24.642 ms | 24.477 ms | +164.788 us / +0.673% |

The 96 additional nodes are exactly three extra update nodes per layer. The
current ByteV2 replay contains stage, metadata clear, commit and release;
raw contains one `reshape_and_cache_flash_kernel` per layer. Cache update is
69.2% of the named attention-plus-update gap, and attention is 31.3%.

Common model kernels happened to total 168.5 us less in the Byte trace. That
is cross-process clock/cache variation, not a ByteV2 optimization. Therefore
the graph-envelope comparison is diagnostic; the uninstrumented paired E2E
result is authoritative.

## 6. Nsight Compute diagnosis at 1K

Nsight Compute 2025.4.1 collected one full+PM and one source-level report for
each main kernel. NCU replay duration is diagnostic and is not substituted for
the interleaved Nsys or uninstrumented E2E timing.

| Metric | ByteV2 main | Raw main | Difference |
| --- | ---: | ---: | ---: |
| NCU duration | 20.288 us | 17.120 us | +18.50% |
| Grid / block | 64 / 128 | 64 / 128 | same |
| Registers / thread | 240 | 252 | -12 |
| Shared memory / block | 49 KiB | 81 KiB | -32 KiB |
| DRAM bytes read | 3.594 MiB | 4.630 MiB | -22.38% |
| DRAM read rate | 185.8 GB/s | 283.6 GB/s | -34.50% |
| SM throughput | 11.85% | 14.24% | -2.39 pp |
| Compute-memory throughput | 29.29% | 45.45% | -16.16 pp |
| Instructions / SM partition | 2,233.9 | 1,330.3 | +67.93% |
| Global-load instructions | 3,328 | 1,024 | +225.0% |
| Shared-load instructions | 12,800 | 4,096 | +212.5% |
| Shared-store instructions | 16,896 | 8,192 | +106.3% |
| L1 hit rate | 48.80% | 2.05% | +46.75 pp |
| L2 hit rate | 71.81% | 12.26% | +59.56 pp |

The compressed loader succeeds at reducing DRAM traffic, but 64 CTAs do not
fill all 84 A40 SMs and the kernel is latency/instruction bound at 1K. The
extra decode and staging instructions outweigh the saved bytes.

The dominant ByteV2 source hotspot is
`byte_v2_fa2_loader.cuh:493`: 191 long-scoreboard samples, 69.2% of all
ByteV2 long-scoreboard samples. The source line consumes `outlier_mask` next
to the per-tile `base` load, so the mapped stall is a dependency on sideband
metadata before decode. Other loader long-scoreboard hotspots are lines 395
(40 samples), 453 (18) and 428 (11). Both kernels have zero local-memory
loads/stores.

## 7. Ranked next steps

### 1. Re-evaluate update-node fusion at the short-context E2E worst case

The current four-node ByteV2 update costs 295.0 us per 32-layer replay versus
80.3 us for raw and contributes 69.2% of the named gap. The already-implemented
two-kernel opt-in improved 4K E2E by only 0.288%, so it correctly remains off,
but 128--1K now has a larger fixed-overhead gap. Run the existing two-kernel
candidate at 128, 512 and 1K before designing another update kernel. Retain it
only if the paired E2E gain clears the project threshold and correctness gates.

### 2. Hide the base/outlier sideband dependency in the 1K loader

Move or pack the per-tile base/outlier metadata load early enough to overlap
with the existing `cp.async` payload staging, while preserving the raw FA2
shared-memory layout and bitwise output. Evidence is the 191-sample hotspot at
line 493, +67.9% instructions, and the 1.825 us paired main-kernel gap at 1K.
The upper bound is smaller than update work: attention accounts for about 31%
of the named 1K replay gap.

### 3. Attribute the remaining decode residual before another rewrite

At 1K, the warmup-subtracted E2E residual is about 590 us per decode step,
while named attention+update kernels explain about 310 us in the separate
serving traces. A synchronized timestamp around scheduler work, graph launch
and graph completion is needed to distinguish real host/backend overhead from
cross-process GPU-clock variation. Optimizing an unattributed residual would
be premature.

Long-context QK/softmax/PV/split/combine should not be changed first: the
interleaved direct trace is tied at 4K and ByteV2 is 13.1% faster at 16K on
clean pages.

## 8. Correctness, confidence and caveats

- 24/24 E2E ByteV2/raw pairs: exact 256-token equality.
- All ByteV2 and raw profiler replays: exact token equality.
- All saved output lists across three rounds: exact equality.
- Direct FA2 output and LSE at 1K, 4K and 16K: bitwise exact for 200
  interleaved iterations.
- The ByteV2/raw 1K serving traces: exact eight-token equality, with exact
  profiler replay tokens on both sides.
- `git diff --check`: passed; the pre-existing tracked changes were not
  reverted or altered by this profile-only run.
- All five run-local analysis scripts pass `ruff-format` and `ruff-check`.

Caveats:

- `warmup_seconds` is a prefill proxy, not a dedicated TTFT trace.
- Nsys node tracing and NCU replay perturb clocks and timing; they are used for
  attribution, not the final throughput decision.
- Direct-oracle pages do not reproduce the exact outlier distribution of all
  Llama KV pages.
- The GPU has 84 SMs while the 1K main kernel launches 64 CTAs, making the
  short workload particularly sensitive to latency and clock state.

## 9. Reproduction

Representative E2E command, repeated for both backends and three paired
rounds:

```bash
env CUDA_VISIBLE_DEVICES=0 \
  VLLM_ENABLE_V1_MULTIPROCESSING=0 \
  BYTE_V2_DECODE_KERNEL=fa2 \
  BYTE_V2_DECODE_RAW_FALLBACK=0 \
  BYTE_V2_NATIVE_RAW_STAGING_UPDATE=1 \
  BYTE_V2_FUSED_SINGLE_TOKEN_STAGING=1 \
  BYTE_V2_FUSED_SINGLE_TOKEN_COMMIT_RELEASE=0 \
  BYTE_V2_FUSED_SINGLE_TOKEN_STAGE_METADATA_CLEAR=0 \
  BYTE_V2_FUSED_COMMIT_METADATA_CLEAR=1 \
  BYTE_V2_WARP_PARALLEL_COMMIT_HISTOGRAM=1 \
  BYTE_V2_NATIVE_SINGLE_TOKEN_UPDATE=0 \
  .venv/bin/python scripts/byte_v2_speculative_profile.py \
  --backend byte_v2 \
  --context-lens 64 128 512 1024 2048 4096 8192 16384 \
  --spec-tokens 0 --batch-size 1 --max-tokens 256 \
  --gpu-memory-utilization 0.80 --disable-prefix-caching \
  --no-enforce-eager --compile-size-specialization \
  --output-jsonl profile/byte-v2-e2e-length-sweep-a40-20260720/reports/e2e_run1_byte.jsonl
```

Representative interleaved attention trace:

```bash
CUDA_VISIBLE_DEVICES=0 nsys profile --trace=cuda --sample=none \
  --cpuctxsw=none --force-overwrite=true \
  -o profile/byte-v2-e2e-length-sweep-a40-20260720/reports/nsys_pair_seq1024 \
  .venv/bin/python \
  profile/byte-v2-e2e-length-sweep-a40-20260720/harness/byte_v2_fa2_oracle.py \
  --seq-len 1024 --query-len 1 --num-splits 0 --iterations 200
```

Representative NCU full report:

```bash
CUDA_VISIBLE_DEVICES=0 /opt/nvidia/nsight-compute/2025.4.1/ncu \
  --set full --section PmSampling --section PmSampling_WarpStates \
  --target-processes all \
  --kernel-name 'regex:.*flash_fwd_splitkv_byte_v2_kernel.*' \
  --launch-count 1 --force-overwrite \
  --export profile/byte-v2-e2e-length-sweep-a40-20260720/reports/ncu_full_byte_seq1024 \
  .venv/bin/python \
  profile/byte-v2-e2e-length-sweep-a40-20260720/harness/byte_v2_fa2_oracle.py \
  --seq-len 1024 --query-len 1 --iterations 1 --byte-only
```

## 10. Artifacts

- `analysis/e2e_summary.{json,csv}`
- `analysis/nsys_paired_attention_summary.{json,csv}`
- `analysis/nsys_engine_graph_summary.{json,csv}`
- `analysis/ncu_comparison_summary.{json,csv}`
- `analysis/compare_byte_seq1024_vs_raw_seq1024.txt`
- `analysis/stall_hotspots_{byte,raw}_seq1024.txt`
- `analysis/pm_timeline_plots.txt`
- `reports/e2e_run{1,2,3}_{byte,raw}.jsonl`
- `reports/nsys_pair_seq{1024,4096,16384}.nsys-rep`
- `reports/nsys_engine_{byte,raw}_seq1024_out8.nsys-rep`
- `reports/ncu_{full,source}_{byte,raw}_seq1024.ncu-rep`
