# ByteV2 Short-Context Two-Kernel E2E Retention

Date: 2026-07-20

GPU: NVIDIA A40 (SM86), driver 590.48.01

Source revision: `c08264a54c55aa893144b6387ec8670911230e33`

Stable native extension SHA256:
`d04c3cbc3f13b2bf4a73fffc22e1a596af849aa6c0751dbd1c042387e47dd38f`

## 1. Outcome

The optional two-update-kernel n=1 path does not pass the declared 0.5% E2E
retention gate at context 128, 512, or 1,024. Its median paired throughput
changes relative to the current four-update-kernel default are -0.6062%,
-0.3174%, and +0.0762%, respectively. None of the nine individual paired
rounds reaches +0.5%; the best round is only +0.3699%.

All 27 measured requests generate exactly the same 256 token IDs, and every
instrumented replay matches its corresponding uninstrumented request. The
experiment therefore finds no correctness difference, but also no robust E2E
benefit.

The production default remains the safe four-update-kernel path. The two
fusion controls remain off by default:

```text
BYTE_V2_FUSED_SINGLE_TOKEN_COMMIT_RELEASE=0
BYTE_V2_FUSED_SINGLE_TOKEN_STAGE_METADATA_CLEAR=0
```

Here, four-kernel and two-kernel refer only to the ByteV2 single-token cache
update. Both ByteV2 modes use the same FA2-template Q1 attention kernel and
the original FA2 combine kernel.

## 2. Experiment

The workload is Llama-3.1-8B-Instruct in BF16, batch 1, no speculation,
greedy decoding, 256 output tokens, contexts 128/512/1,024, compiled execution
with CUDA Graphs and size-1 specialization, and prefix caching disabled. The
three-context worker sets `max_model_len=1,288`; this is intentionally a
short-context deployment shape rather than the 16K-capacity engine used by
the preceding broad length sweep.

The formal metric is uninstrumented `output_tokens_per_second`, recomputed as
`256 / measured_seconds`. The later profiler replay is used only for token
validation and diagnostic ranges.

Three fresh engines per round are ordered as a cyclic Latin square:

| Round | First | Second | Third |
| ---: | --- | --- | --- |
| 1 | Four-kernel | Two-kernel | Raw FA2 |
| 2 | Two-kernel | Raw FA2 | Four-kernel |
| 3 | Raw FA2 | Four-kernel | Two-kernel |

Common ByteV2 controls are:

```text
BYTE_V2_DECODE_KERNEL=fa2
BYTE_V2_DECODE_RAW_FALLBACK=0
BYTE_V2_NATIVE_RAW_STAGING_UPDATE=1
BYTE_V2_FUSED_STAGING_RELEASE_FLAGS=1
BYTE_V2_FUSED_SINGLE_TOKEN_STAGING=1
BYTE_V2_FUSED_COMMIT_METADATA_CLEAR=1
BYTE_V2_WARP_PARALLEL_COMMIT_HISTOGRAM=1
BYTE_V2_NATIVE_SINGLE_TOKEN_UPDATE=0
```

Both debug-warmup environment variables are explicitly unset. The
four-kernel control sets the two candidate switches to `0`; the two-kernel
mode sets both to `1`. Raw uses the `FLASH_ATTN` backend. File modification
times preserve the declared A-B-R, B-R-A, and R-A-B execution order.

The same native binary and opt-in path were previously validated with an
uncollapsed serving capture: steady Q1 decode executes the fused stage/clear
and commit/release nodes, while generic prepare/hydrate/append ranges belong
to compile and capture work. Python profiler labels in these JSONL files do
not expose nodes inside a replayed CUDA Graph and must not be used to infer
the steady-decode route.

## 3. End-to-End Results

`Two/Four`, `Four/Raw`, and `Two/Raw` are paired throughput ratios from the
same Latin-square round.

| Context | Round | Four tok/s | Two tok/s | Raw tok/s | Two/Four | Four/Raw | Two/Raw |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 1 | 37.825237 | 37.437307 | 38.039902 | -1.025584% | -0.564314% | -1.584111% |
| 128 | 2 | 37.396049 | 37.534394 | 38.024055 | +0.369946% | -1.651602% | -1.287766% |
| 128 | 3 | 37.655678 | 37.427406 | 38.065969 | -0.606210% | -1.077840% | -1.677517% |
| 512 | 1 | 37.306345 | 36.955986 | 37.587627 | -0.939141% | -0.748337% | -1.680450% |
| 512 | 2 | 36.905486 | 37.014608 | 37.559685 | +0.295681% | -1.741761% | -1.451230% |
| 512 | 3 | 37.084310 | 36.966609 | 37.594755 | -0.317388% | -1.357756% | -1.670835% |
| 1,024 | 1 | 36.307472 | 36.209640 | 37.028326 | -0.269454% | -1.946764% | -2.210972% |
| 1,024 | 2 | 36.107199 | 36.178293 | 36.966506 | +0.196897% | -2.324556% | -2.132236% |
| 1,024 | 3 | 36.193870 | 36.221435 | 36.946222 | +0.076158% | -2.036343% | -1.961736% |

Three-round paired summary:

| Context | Median Two/Four | Two/Four range | Median Four/Raw | Median Two/Raw | Rounds at +0.5% | Gate |
| ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| 128 | -0.606210% | [-1.025584%, +0.369946%] | -1.077840% | -1.584111% | 0/3 | Fail |
| 512 | -0.317388% | [-0.939141%, +0.295681%] | -1.357756% | -1.670835% | 0/3 | Fail |
| 1,024 | +0.076158% | [-0.269454%, +0.196897%] | -2.036343% | -2.132236% | 0/3 | Fail |

The median across all nine Two/Four pairs is -0.269454%. At 128 and 512 the
round-to-round spans are 1.3955 and 1.2348 percentage points and the sign
flips, so the small negative medians should not be interpreted as precise
regressions. The retention conclusion is nevertheless unambiguous: every
observed pair is below the +0.5% gate. A reverse Latin square was not added
because the candidate is not borderline with the acceptance threshold.

The short-shape four-kernel gaps to raw are smaller than those in the earlier
64-to-16K sweep. Absolute throughputs across the two studies must not be
combined: their `max_model_len`, cache capacity, and compiled engine shape
sets differ. The paired ratios within this experiment are the decision
metric.

## 4. Token Validation

The analyzer validates backend, context order, batch size, Q1 configuration,
compiled execution, output count, finite timing, and exact recomputation of
TPS for every JSONL row. It then compares all nine saved token lists at each
context element by element.

| Context | Generated tokens | Canonical token-list SHA256 |
| ---: | ---: | --- |
| 128 | 256 | `b065e99e08886d010ec2396707129343eceb03775169c85f5bdd2359bce1edc1` |
| 512 | 256 | `9b10f04ca1bd79c633f3a6fc99f1ebb0b47fec5799926dc21421406859fbd70b` |
| 1,024 | 256 | `8df12ab655390457ff63a217444ebeee4e1539d050765eb77587e5b5a68930e9` |

All 27 `profile_token_ids_match` values are true. The profile replay token
lists are not separately serialized by the harness, so that second check is
represented by the saved boolean rather than an independently recomputed
hash.

## 5. Decision and Next Target

Removing two CUDA Graph update nodes is not a useful short-context E2E
optimization. The result agrees with the prior 4K result, where the same
candidate improved throughput by only 0.2883%. More clear/release launch
fusion should not be pursued unless it also removes substantial device work.

The next measurement should use synchronized short-shape ByteV2/raw Graph
traces at context 1,024 to separate the remaining cache-update, attention, and
host/graph residual. If cache update remains dominant, the structural target
is the work still performed by the two surviving nodes: hydrating and
re-encoding the complete active 16-token page for every new token. A
persistent authoritative BF16 tail-page sidecar, with compression on page
closure and an explicit reader mode, is the highest-ceiling candidate. It
would require new fail-closed storage semantics; the current V5 overflow and
unsafe flags do not provide a recoverable raw payload.

No production source code was changed by this experiment.

## 6. Reproduction

Representative ByteV2 command; set both candidate values to `0` for the
four-kernel control or `1` for the two-kernel candidate:

```bash
env -u BYTE_V2_DEBUG_WARMUP -u VLLM_BYTE_V2_DEBUG_WARMUP \
  CUDA_VISIBLE_DEVICES=0 VLLM_ENABLE_V1_MULTIPROCESSING=0 \
  BYTE_V2_DECODE_KERNEL=fa2 BYTE_V2_DECODE_RAW_FALLBACK=0 \
  BYTE_V2_NATIVE_RAW_STAGING_UPDATE=1 \
  BYTE_V2_FUSED_STAGING_RELEASE_FLAGS=1 \
  BYTE_V2_FUSED_SINGLE_TOKEN_STAGING=1 \
  BYTE_V2_FUSED_SINGLE_TOKEN_COMMIT_RELEASE=<0-or-1> \
  BYTE_V2_FUSED_SINGLE_TOKEN_STAGE_METADATA_CLEAR=<0-or-1> \
  BYTE_V2_FUSED_COMMIT_METADATA_CLEAR=1 \
  BYTE_V2_WARP_PARALLEL_COMMIT_HISTOGRAM=1 \
  BYTE_V2_NATIVE_SINGLE_TOKEN_UPDATE=0 \
  .venv/bin/python scripts/byte_v2_speculative_profile.py \
  --backend byte_v2 --context-lens 128 512 1024 \
  --spec-tokens 0 --batch-size 1 --max-tokens 256 \
  --gpu-memory-utilization 0.80 --disable-prefix-caching \
  --no-enforce-eager --compile-size-specialization \
  --output-jsonl <output.jsonl>
```

Raw uses the same workload with `--backend flash_attn`. Aggregate and validate
all nine files with:

```bash
.venv/bin/python \
  profile/byte-v2-short-context-two-kernel-e2e-a40-20260720/analysis/analyze_e2e_ab.py \
  --run-dir profile/byte-v2-short-context-two-kernel-e2e-a40-20260720
```

## 7. Artifacts

- `analysis/experiment_manifest.json`
- `analysis/e2e_ab_summary.json`
- `analysis/e2e_ab_summary.csv`
- `analysis/analyze_e2e_ab.py`
- `reports/run{1,2,3}_{four_kernel,two_kernel,raw}.jsonl`
