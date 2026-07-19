# ByteV2 FA2 early-softmax experiment (A40, 2026-07-19)

## Decision

Do not retain either early-softmax implementation. Restore the split K/V
shared-memory alias checkpoint at commit `66793e3a19450bc8a26523de23b5bc39cb5de512`.

Moving the source-level online softmax before the V `cp.async.wait_group` did
not improve the generated kernel. Allowing ptxas to schedule freely moved only
masking and the row-max reduction before the wait and regressed the median by
about 1.0% to 1.7%. Forcing the complete softmax to precede the wait with a
row-sum-dependent uniform branch regressed a matched four-split run by 3.3%.

## Scope and setup

- GPU: NVIDIA A40 (SM86)
- Workload: Q1 decode, BF16, head dimension 128, sequence length 4099
- Comparison: ByteV2 external KV loader versus raw paged FA2 in the same process
- Timing: 20 interleaved launches per path, collected with Nsight Systems
- Correctness gate: output and LSE must be bitwise identical
- Main and combine kernels: original FA2 templates; only the ByteV2 V-wait
  scheduling was changed for this experiment

The automatic-split runs selected `gridY=17`. Candidate 2 was collected with
an explicit four-split configuration, so it is compared only with the restored
four-split baseline. Comparing it directly with the 17-split baseline would be
invalid.

## Results

| Variant | Splits | ByteV2 median | Raw median | Result |
| --- | ---: | ---: | ---: | --- |
| Initial stable baseline | auto (17) | 36.080 us | 35.616 us | reference |
| Candidate 1, run 1 | auto (17) | 36.688 us | 35.616 us | +1.69% ByteV2 |
| Candidate 1, run 2 | auto (17) | 36.448 us | 35.600 us | +1.02% ByteV2 |
| Candidate 1, run 3 | auto (17) | 36.496 us | 35.584 us | +1.15% ByteV2 |
| Candidate 2, strict dependency | 4 | 79.170 us | 40.289 us | rejected |
| Restored baseline, matched control | 4 | 76.625 us | 40.544 us | candidate 2 is +3.32% |
| Restored baseline, final check | auto (17) | 36.320 us | 35.585 us | restored |

The raw-normalized candidate-2 regression is about 4.0%; the direct ByteV2
median regression is 3.32%. The final automatic-split check is within 0.7% of
the initial baseline and retains bitwise equality.

The final installed shared object has SHA256
`920cba129ad1167db45cd6f30cca1bf43c371e4a61c6d93e00087dae28c816b6`.
Its noncausal/causal split kernels use 240/244 registers, 49,152 bytes of
dynamic shared memory, and no stack, local-memory, `LDL`, or `STL` traffic.
The experimental row-sum dependency and duplicate wait paths are absent.

## Generated-code findings

### Candidate 1: source reorder without a hard dependency

- Noncausal/causal split kernels used 240/241 registers.
- Stack, local memory, `LDL`, and `STL` were all zero.
- The V async-copy commit was at `0x4530` and the wait at `0x5200` in the
  inspected noncausal path, leaving 204 instructions between them.
- ptxas placed masking and row-max reduction before the wait, but the first
  `MUFU.EX2` remained after it at `0x5290`.

Thus the source reorder became only a partial overlap in final SASS and was
consistently slower across three repetitions.

### Candidate 2: enforced row-sum dependency

Candidate 2 XORed the softmax row sums, broadcast lane 0, and used the result
to select between two identical predicated `cp.async.wait_group 0` paths. This
successfully prevented ptxas from hoisting the wait:

- Noncausal/causal split kernels used 243/244 registers, three more than
  candidate 1.
- Stack, local memory, `LDL`, and `STL` remained zero.
- All `MUFU.EX2` operations and row-sum reductions appeared before the V wait
  in masking and generic paths.
- The emitted XOR, `SHFL`, predicate, `BSSY`/`BRA`/`BSYNC`, and duplicated wait
  control paths added enough scheduling cost to make the kernel slower.

The experiment shows that V-wait placement was not the remaining dominant
bottleneck under this schedule. Forcing a full softmax into the overlap window
increases dependency-chain and control-flow cost more than it hides V traffic.

## Correctness and stopping rule

- Candidate 1 passed bitwise output/LSE checks for Q1, forced-outlier Q1/Q2,
  and a ragged, permuted-page, shared-prefix Q1 case.
- Candidate 2 passed the Q1 sequence-4099 four-split bitwise check.
- The restored implementation passed both four-split and automatic-split Q1
  checks with zero output and LSE mismatches.

No full Nsight Compute collection or sanitizer campaign was run for the
candidates because both failed the Nsight Systems performance gate. The stable
checkpoint had already passed its correctness and sanitizer coverage.

## Artifacts

- `harness/byte_v2_fa2_oracle_baseline.py`: frozen baseline harness
- `analysis/summary.csv`: launch-time summary
- `reports/baseline-seq4099-q1.*`: initial stable baseline
- `reports/candidate{,-repeat2,-repeat3}-seq4099-q1.*`: candidate 1 repeats
- `reports/candidate2-seq4099-q1.*`: candidate 2, explicit four splits
- `reports/restored-baseline-seq4099-q1.*`: matched four-split control
- `reports/restored-baseline-auto-seq4099-q1.*`: final automatic-split check

## Recommended next experiment

Keep the original FA2 mask/softmax/PV order. The next useful target is the
ByteV2 decode itself: reduce instruction and dependency cost in the staged
payload reconstruction while preserving the 48 KiB K/V alias schedule and the
original FA2 combine kernel. Any new candidate should first pass the same
bitwise oracle and interleaved Nsight Systems gate before collecting NCU.
