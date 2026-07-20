# ByteV2 Safe Single-Token Commit/Release Fusion

Date: 2026-07-20

GPU: NVIDIA A40 (SM86), driver 590.48.01

Source revision: `c08264a54c55aa893144b6387ec8670911230e33`

Candidate extension SHA256:
`d04c3cbc3f13b2bf4a73fffc22e1a596af849aa6c0751dbd1c042387e47dd38f`

## Outcome

The experiment safely reduces the default V5 n=1 staging update from four
kernels to three by folding page-unsafe-flag aggregation and allocator release
into the existing commit. At target row 8, a same-process alternating A/B
reduces the complete update from 27.118 to 21.957 us by CUDA event, or 19.03%.

This implementation remains available as an explicit experiment through
`BYTE_V2_FUSED_SINGLE_TOKEN_COMMIT_RELEASE=1`, but it is not enabled by
default. The subsequent two-kernel candidate, which includes this change,
improves the final same-binary E2E workload by only 0.288%, below the declared
0.5% retention gate.

## Implementation

The specialized commit uses the existing `overflow[0]` word as a completion
counter for its 128 CTAs. Each tile CTA atomically contributes K/V unsafe bits
when its encoded tile uses an overlay or fallback representation. The final
CTA releases `block_to_staging_slot`, resets the slot metadata and counter,
and leaves `page_unsafe_flags` with the same externally visible value as the
old release-plus-flags kernel.

The specialized stage clears the destination page's old unsafe flags before
commit. Invalid negative and positive-out-of-range slots preserve valid-page
flags and still complete all CTA tickets, preventing a stale counter from
poisoning the next CUDA Graph replay. The generic update and direct Torch-op
call arity retain their old behavior because the new schema argument is a
trailing `False` default.

The specialization is selected only when all of the following hold:

- V5 default tile policy, one input token and one staging slot;
- safe fused single-token staging is enabled;
- fused metadata clear and warp-parallel histogram are enabled;
- serial-metadata bypass is disabled;
- `fuse_single_token_commit_release=true`.

## Isolated A/B

The row-8 comparison uses 21 trials of 500 updates in one process and reverses
baseline/candidate order every trial.

| Timer | Four kernels | Three kernels | Change |
| --- | ---: | ---: | ---: |
| CUDA event | 27.117567 us | 21.956608 us | -19.0318% |
| Host wall | 27.149990 us | 21.986646 us | -19.0178% |

Nsight Systems traces 300 updates in each mode:

| Stage | Four-kernel average | Three-kernel average |
| --- | ---: | ---: |
| Fused hydrate/append | 2.1663 us | 2.1535 us |
| Metadata clear | 1.5225 us | 1.5236 us |
| Commit | 8.4417 us | 8.8407 us |
| Release plus flags | 1.7750 us | absent |

The commit grows by about 0.40 us because it performs flag atomics and the
cross-CTA completion protocol. Removing the release launch and its graph gap
still produces a 5.16 us complete-operation reduction.

## Correctness and Safety

The implementation passed cache equivalence for rows 0, 1, 8 and 15 with
random, low-outlier and non-contiguous K/V inputs. Deterministic clean,
K-only, V-only and K+V pages produce exact flags 0, 3, 5 and 7 against an
independent full-page scan. Invalid slots, preflight failure and CUDA Graph
replay preserve cache and allocator state.

Compute Sanitizer reports:

- row-0 synccheck: zero errors;
- row-15 memcheck: zero errors;
- row-0 racecheck: zero hazards, errors or warnings.

## Artifacts

- `harness/safe_n1_commit_release_profile.py`
- `reports/nsys_standalone_release_row8.nsys-rep`
- `reports/nsys_fused_release_row8.nsys-rep`
- `reports/memcheck_fused_release_row15.log`
- `reports/synccheck_fused_release_row0.log`
- `reports/racecheck_fused_release_row0.log`
- `analysis/nsys_standalone_release_row8_cuda_gpu_kern_sum.csv`
- `analysis/nsys_fused_release_row8_cuda_gpu_kern_sum.csv`
