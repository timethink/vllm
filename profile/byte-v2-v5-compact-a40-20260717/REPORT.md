# ByteV2 V5 Compact KV Cache Results

Date: 2026-07-17

## Scope

- GPU: NVIDIA A40 (SM86)
- Model: Llama-3.1-8B-Instruct, BF16
- Context: 4096 tokens
- Batch: 1
- Speculative verify: Q2, Q4, Q8, and Q16
- Compile mode: static query-size specialization
- Acceptance rate: 100% on the repeated-prompt workload

## Format Change

V5 replaces V4's fixed 256-entry outlier payload for every codec tile with one
1024-entry pool shared by all K/V tiles and KV heads in a page. The fixed
12-bit split-plane payload is unchanged.

| Component | Bytes |
| --- | ---: |
| Header and per-head metadata | 896 |
| Fixed 12-bit K/V payload | 49152 |
| Shared outlier pool | 2048 |
| V5 page | 52096 |
| Raw BF16 page | 65536 |
| Legacy V4 page | 115328 |

V5 saves 20.51% of physical page bytes versus raw BF16. At equal allocatable
memory, this gives a theoretical 25.80% increase in token capacity.

The pool uses per-tile uint16 count and offset descriptors. Writers allocate
power-of-two tile segments. Single-token updates compact the page before
append/relocation. Overflow sets an overflow flag and traps, so the compact
format fails closed instead of dropping outliers.

## Real Outlier Load

The existing layer-0 capture at 4097 tokens contains 257 active pages. Summing
all K/V tile counts per page gives:

| Metric | Entries / page |
| --- | ---: |
| Mean | 4.78 |
| P50 | 5 |
| P95 | 7 |
| P99 | 9 |
| Maximum | 28 |

The observed maximum is 36.6 times smaller than the 1024-entry pool. This is
evidence for the selected capacity on this model/workload, not a proof for all
possible models or adversarial data.

## KV Capacity

| Backend | Available KV memory | Allocated KV tokens | Max 4151-token concurrency |
| --- | ---: | ---: | ---: |
| ByteV2 V5 | 19.87 GiB | 204324 | 49.22x |
| Raw FA2 | 19.96 GiB | 163118 | 39.30x |

Measured token capacity increases by 25.26%. Legacy V4 allocated only 92295
tokens because its 115328-byte page was larger than raw BF16.

## E2E Throughput

| Verify Q | ByteV2 V5 | Raw FA2 | V5 / raw | Previous V4 | V5 / V4 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 59.643 tok/s | 57.618 tok/s | +3.51% | 59.804 tok/s | -0.27% |
| 4 | 109.008 tok/s | 106.986 tok/s | +1.89% | 109.537 tok/s | -0.48% |
| 8 | 189.755 tok/s | 187.194 tok/s | +1.37% | 190.897 tok/s | -0.60% |
| 16 | 314.876 tok/s | 306.845 tok/s | +2.62% | 315.078 tok/s | -0.06% |

V5 is within 0.6% of V4 at every Q width and remains at practical parity with
raw FA2. All paired ByteV2/raw runs produced identical token IDs.

The compact metadata does raise Q16 raw-staging cache-update time from 30.229
us to 36.117 us (+19.5%), but the update is a small part of the complete step,
so no E2E regression is measurable.

## Correctness

```bash
CUDA_VISIBLE_DEVICES=6 .venv/bin/python -m pytest \
  tests/v1/attention/test_byte_v2_layout.py -q \
  --disable-warnings --maxfail=1
```

Result: `170 passed, 1 skipped` in 172.68 seconds.

The full suite covers layout arithmetic, direct/raw-staging writers,
single-token update and pool compaction, generic/direct/speculative decode,
and outlier-heavy cases. Full-model E2E completed without an overflow trap.

## Conclusion

For the tested A40 Llama-3.1-8B speculative workload, ByteV2 V5 does both:

1. It increases usable KV token capacity by 25.26%.
2. It preserves E2E throughput within measurement noise of V4 and raw FA2.

The remaining deployment condition is pool sizing validation on additional
models and distributions. The current 1024-entry ABI is intentionally bounded
and fail-closed, not universally lossless for pages exceeding that bound.

## Artifacts

- `byte_q16_worker.log`: V5 capacity, Q16 E2E, and operation timing.
- `raw_q16_worker.log`: raw FA2 capacity and Q16 E2E.
- `q2_q8_compare.jsonl`: paired Q2/Q8 runs.
- `q4_compare.jsonl`: paired Q4 run.
