# Static-W16 validation on NVIDIA A100

This directory is a self-contained, fail-closed protocol for deciding whether
the current canonical Static-W16 FA2 reader remains faster than stock Raw-FA2
on an NVIDIA A100. It does not copy an A40 timing conclusion onto A100.

The primary comparison is stock Raw-FA2 versus stock canonical Static-W16,
using one native `sm_80` extension, the same Q/K/V values, the same explicit
split count, and balanced eager/CUDA Graph call order. A split sweep then
compares each backend at its independently best split. Nsight Compute records
the actual register, shared-memory, occupancy, and waves/SM limits on A100.

No A100 result is checked into this branch. The scripts were syntax-checked and
smoke-tested on the available A40 host with the non-formal override; the
managed source was also cross-built with CUDA 13.1 as `sm_80`, linked, and
audited with `cuobjdump`. That proves build compatibility, not A100 runtime
behavior: formal performance and occupancy claims must come from the target
A100.

## What is measured

The timed unit is one Q1, BF16, head-dimension-128, 32-query-head/8-KV-head
paged attention call. A CUDA Graph replay contains the split main kernel and
FA2 split reduction/combine work. It is an attention-operator measurement,
not full-model TPOT or end-to-end serving latency.

Every timing file must first pass all of these gates:

1. unpacking the 49,792-byte canonical W16 page reproduces raw BF16 K/V
   bit-for-bit;
2. eager W16 output and LSE equal stock Raw-FA2 bit-for-bit;
3. CUDA Graph Raw/W16 output and LSE equal eager Raw-FA2 bit-for-bit;
4. the requested compact experiment has zero raw-fallback pages;
5. all A40-only profile controls are unset;
6. the device is an A100 with compute capability 8.0.

`run_smoke.sh` runs both compact pages and pages with deterministic escape
entries. The formal matrix uses compact pages so it measures the compressed
reader instead of mixing in the dense fallback path. A real captured K/V file
can replace synthetic inputs.

## Files

| File | Purpose |
| --- | --- |
| `harness/build_fa2_sm80.sh` | Configure and build a native `sm_80` FA2 extension |
| `harness/benchmark.py` | Bitwise gates, eager timing, CUDA Graph timing, and one-launch profiling |
| `harness/static_w16_fixed_page.py` | Independent canonical W16 pack/unpack oracle |
| `harness/run_smoke.sh` | Fast compact/escape correctness smoke |
| `harness/run_matrix.sh` | Repeated context-length and split-count sweep |
| `harness/collect_ncu.sh` | Stock Raw and W16 main-kernel NCU collection |
| `analysis/analyze.py` | Validate all records and select each backend's best split |
| `analysis/analyze_ncu.py` | Extract comparable launch-resource evidence |

Raw results, profiler reports, model files, captures, and compiled `.so` files
are intentionally ignored. They are large and machine-specific. JSON records
contain the GPU, extension hash, input hash, Torch/CUDA versions, Git revision,
and complete event samples needed for an audit.

## 1. Prepare the environment

Use the repository's `uv` environment; do not use system Python or bare pip.
If the existing checkout is already usable, no reinstall is needed.

```bash
cd /path/to/vllm

uv venv --python 3.12
uv pip install -r requirements/lint.txt
uv pip install -e . --torch-backend=auto
```

The A100 machine needs a CUDA toolkit capable of compiling `sm_80`, a matching
PyTorch CUDA build, CMake, and Nsight Compute for the resource pass.

## 2. Build the exact A100 binary

```bash
profile/bytev2-static-w16-a100-validation/harness/build_fa2_sm80.sh
```

The script configures the ignored, isolated
`profile/bytev2-static-w16-a100-validation/results/build-sm80` directory with
`TORCH_CUDA_ARCH_LIST=8.0` and `CMAKE_CUDA_ARCHITECTURES=80`, builds only
`_vllm_fa2_C`, checks for an `sm_80` image when `cuobjdump` is available, and
prints the binary SHA-256 and the export command. Use the printed path, for
example:

```bash
export BYTEV2_FA2_SO="$PWD/profile/bytev2-static-w16-a100-validation/results/build-sm80/vllm-flash-attn/_vllm_fa2_C.abi3.so"
export BYTEV2_GPU_ID=0
```

Do not reuse the A40 `sm_86` build. Do not set
`VLLM_FLASH_ATTN_SRC_DIR`: this checkout's managed patch must be applied to
the pinned FA2 source.

## 3. Run the correctness smoke

```bash
profile/bytev2-static-w16-a100-validation/harness/run_smoke.sh \
  /absolute/output/a100-smoke
```

The two output JSON files cover no-escape compact pages and 64 total escapes
per page with a non-monotonic block table. Both must finish without a gate
failure. `BYTEV2_ALLOW_NON_A100=1` exists only to test the harness elsewhere;
such output is rejected by the formal analyzer by default.

## 4. Run the performance matrix

The default matrix is five contexts × six split counts × three fresh
processes: 8K, 16K, 32K, 64K, and 128K; splits 8, 12, 16, 20, 24, and 32.
Split order rotates between repetitions to reduce fixed-order bias.

```bash
profile/bytev2-static-w16-a100-validation/harness/run_matrix.sh \
  /absolute/output/a100-formal
```

For a shorter first pass:

```bash
BYTEV2_SEQS="8192 65536" \
BYTEV2_SPLITS="12 16 20 24" \
profile/bytev2-static-w16-a100-validation/harness/run_matrix.sh \
  /absolute/output/a100-first-pass
```

The measurement knobs are explicit:

```text
BYTEV2_REPETITIONS=3
BYTEV2_WARMUP=12
BYTEV2_ITERATIONS=60
BYTEV2_CALLS_PER_SAMPLE=20
```

To use a real capture, the `.pt` file must contain BF16
`evaluation_key`/`evaluation_value` tensors shaped `[tokens, 8, 128]`, or
`[layers, tokens, 8, 128]` with a selected layer:

```bash
BYTEV2_INPUT=capture \
BYTEV2_CAPTURE=/absolute/path/kv_capture.pt \
BYTEV2_CAPTURE_LAYER=0 \
profile/bytev2-static-w16-a100-validation/harness/run_matrix.sh \
  /absolute/output/a100-capture
```

The fixed K/V exponent bases default to 115/110. A capture that produces raw
fallback pages is rejected by this compact-path protocol. Calibrate and freeze
appropriate per-layer bases, or report fallback-path measurements separately;
do not silently include them in the headline compact result.

## 5. Aggregate the timing results

```bash
profile/bytev2-static-w16-a100-validation/harness/analyze_results.sh \
  /absolute/output/a100-formal \
  /absolute/output/a100-analysis
```

The report contains two distinct comparisons:

- same-split Raw versus W16 for every measured point;
- independently tuned best Raw split versus independently tuned best W16
  split for each context.

The second is the fair headline performance comparison. Selecting W16's best
split while holding Raw at a non-optimal split is not accepted. Negative
`W16 vs Raw` means lower W16 latency; positive `W16 speedup` means W16 is
faster.

## 6. Prove the launch-resource explanation

```bash
export BYTEV2_NCU_BIN=/absolute/path/to/ncu

profile/bytev2-static-w16-a100-validation/harness/collect_ncu.sh \
  /absolute/output/a100-ncu

profile/bytev2-static-w16-a100-validation/harness/analyze_ncu.sh \
  /absolute/output/a100-ncu \
  /absolute/output/a100-ncu-analysis
```

The collector profiles one stock main launch per backend after correctness and
warm-up. It deliberately does not enable the A40 attribution controls
(`Raw alias2` or `W16 clamp1`). The resulting table must be read using both
register and shared-memory block limits: the smaller resource limit controls
resident CTAs. The purpose is to verify, rather than assume, whether stock Raw
and W16 have equal residency on A100.

If NCU access is restricted, enable the platform's performance-counter access
and rerun. Do not substitute A40 occupancy numbers in the A100 report.

## Acceptance and interpretation

A defensible A100 conclusion requires:

- every correctness field is zero and every wire gate is true;
- one extension SHA and one input SHA across the complete matrix;
- no profile environment controls and no raw fallback pages;
- three fresh-process repetitions for every context/split pair;
- an independently tuned W16 advantage larger than run-to-run spread at the
  claimed contexts;
- NCU evidence showing the actual stock resource/occupancy relationship.

The earlier A40 result showed W16 ahead by about 5.7% at 8K and 14.7% at 64K
against stock Raw, but stock Raw was one-CTA resident while W16 was two-CTA
resident there. An A100 has a different SM shared-memory budget, so its stock
Raw path may gain residency. The A100 matrix is specifically designed to
separate that architecture effect from the long-context bandwidth benefit.
The frozen human-readable evidence and machine-readable aggregate are in
[`baseline/A40_OCCUPANCY_REPORT.md`](baseline/A40_OCCUPANCY_REPORT.md) and
[`baseline/A40_SUMMARY.json`](baseline/A40_SUMMARY.json); large raw NCU reports
are intentionally not duplicated here.

Even a positive attention-op result is not automatically a TPOT claim. For a
full serving conclusion, follow it with matched model-level Raw/W16 runs using
the same model, batch shape, CUDA Graph policy, prefix/cascade policy, and
output length, then report attention and non-attention time separately.
