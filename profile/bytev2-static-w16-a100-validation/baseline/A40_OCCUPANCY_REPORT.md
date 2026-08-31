# Static-W16 occupancy portability attribution on A40

## Conclusion

The concern is valid, but it is only part of the explanation.

- On the stock A40 launch, Raw-FA2 is limited to one resident CTA per SM while
  canonical W16 reaches two. At 8K, W16 is 5.72% faster; at 64K it is 14.75%
  faster.
- When a profile-only Raw path is also made two-CTA resident, W16 remains 1.98%
  faster at 8K and 13.20% faster at 64K. The remaining long-context advantage
  therefore cannot be explained by the CTA count alone.
- When the exact same W16 SASS is forced from two CTAs to one by adding unused
  launch-time shared memory, it becomes 32.18% slower than Raw at 8K and
  15.01% slower at 64K. Two-CTA residency is a critical enabler because it
  hides a large part of W16's decode and synchronization latency.

The practical answer is therefore:

1. On another GPU where unmodified Raw-FA2 and W16 have equal residency, the
   short-context W16 gain can shrink substantially and may disappear after
   architecture-specific effects are included.
2. It is not supported by these data to say that all of the current gain comes
   from occupancy. In the 64K A40 proxy, W16 still has a 13.20% advantage after
   both controls are two-CTA resident.
3. No physical cross-architecture claim is made. All eight locally available
   GPUs are A40 (SM 8.6), so an actual equal-residency GPU remains a required
   follow-up.

## Controlled measurements

These are attention-op CUDA Graph replay times. Each call includes the split
main kernel and split reduction/combine kernels; it is not full-model E2E.
Each number is the median of three fresh-process medians. Every process used 12
warm-up rounds, 60 timed samples, and 20 graph replays per sample. The three
conditions were rotated to avoid a fixed execution order.

| Context | Control | Raw (us) | W16 (us) | W16 vs Raw |
| ---: | --- | ---: | ---: | ---: |
| 8K | stock Raw 1 CTA / W16 2 CTA | 69.8368 | 65.8432 | -5.718% |
| 8K | Raw alias 2 CTA / W16 2 CTA | 67.2256 | 65.8944 | -1.980% |
| 8K | stock Raw 1 CTA / same-SASS W16 1 CTA | 69.8368 | 92.3136 | +32.185% |
| 64K | stock Raw 1 CTA / W16 2 CTA | 418.9952 | 357.1968 | -14.749% |
| 64K | Raw alias 2 CTA / W16 2 CTA | 412.4160 | 357.9648 | -13.203% |
| 64K | stock Raw 1 CTA / same-SASS W16 1 CTA | 418.6368 | 481.4848 | +15.013% |

Two useful within-path deltas make the attribution clearer:

- Raw's two-CTA alias control is 3.74% faster than stock Raw at 8K and 1.57%
  faster at 64K.
- Forcing the same W16 kernel from two resident CTAs to one makes W16 40.20%
  slower at 8K and 34.80% slower at 64K.

At 8K, the equal-residency Raw control closes 3.74 percentage points, or about
65% of the observed stock relative gap. At 64K it closes only 1.55 percentage
points, about 10% of the stock gap. These percentages describe this control;
they are not a universal decomposition because the Raw alias path also changes
the K/V staging schedule.

The two controls are intentionally asymmetric, and their deltas must not be
added linearly. Giving Raw a second CTA helps modestly, while taking W16's
second CTA away hurts severely. W16 has extra decode, shared-memory reuse, and
barrier latency for the second CTA to hide; stock Raw does not have the same
latency mix. Thus "two CTAs" is more than a bonus for W16 in this kernel—it is
part of what makes compressed reads profitable—but it is still not the source
of the bytes saved at long context.

## Nsight Compute resource proof

The four 8K/split20 main launches were collected with Nsight Compute 2025.4.1.
The shared-memory carveout is 102.4 KB in every case, so the W16 clamp does not
accidentally trade shared memory for a 64 KB L1/shared configuration.

| Main kernel | Registers/thread | Dynamic shared/block | Driver shared/block | Shared block limit | Theoretical occupancy | Achieved occupancy | Waves/SM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| stock Raw | 250 | 81.92 KB | 1.02 KB | 1 | 8.33% | 8.33% | 1.90 |
| Raw K/V alias | 255 | 49.15 KB | 1.02 KB | 2 | 16.67% | 13.39% | 0.95 |
| W16 default | 254 | 49.41 KB | 1.02 KB | 2 | 16.67% | 13.49% | 0.95 |
| W16 + 1,792-byte clamp | 254 | 51.20 KB | 1.02 KB | 1 | 8.33% | 8.33% | 1.90 |

The W16 clamp adds 1,792 unused bytes only to the dynamic shared-memory launch
size: 49,408 becomes 51,200 bytes. Including the 1,024 driver bytes, two blocks
would require more than the A40's 102,400-byte shared-memory configuration. No
instruction reads or writes the padding.

The default Raw and W16 machine code was hashed before and after adding the
controls:

| Kernel | Pre-control SASS SHA-256 | Final SASS SHA-256 |
| --- | --- | --- |
| stock Raw | `68e2ead97be842f4e5162102826150ac0fe6d5816454cccb5d1322090ba031bf` | same |
| W16 default | `4a384e3b25f8eadf6e3fb0e011566ec4a27044fac768315ec16f4d20644ea6c4` | same |

This makes the W16 2-to-1 CTA experiment a same-SASS launch control. The Raw
two-CTA experiment is complementary but weaker: Raw aliases the K and V shared
tiles and adds the required barriers, so it is not identical to stock Raw
running on a GPU with a larger shared-memory budget. Its register count also
changes from 250 to 255. A physical equal-residency GPU is still needed for the
final portability result. In particular, a stock Raw kernel that gains a
second CTA from hardware capacity would retain its original K/V schedule and
could be faster than this alias proxy. The residual 1.98% 8K W16 margin is too
small to treat as portable proof; the 13.20% 64K margin is stronger evidence,
but remains an A40 result.

## Correctness and default-path gates

- All 18 formal files pass the W16 wire round trip and eager/graph output and
  LSE comparisons bitwise against Raw-FA2.
- The Raw alias output/LSE is also bitwise identical to stock Raw through the
  common W16 reference.
- A separate run with both profile environment variables completely unset
  passes wire, eager, and CUDA Graph output/LSE bitwise gates.
- All formal files use the same FA2 extension SHA-256:
  `d6024ebfd7b13ce2e93c53116f28f0f110c21e8f6fcbfda5bb876d12a6d7ba84`.

The profile knobs are default-off:

- `BYTE_V2_FA2_PROFILE_STATIC_W16_SMEM_PADDING_BYTES=1792` enables the
  same-SASS W16 one-CTA clamp.
- `BYTE_V2_FA2_PROFILE_RAW_KV_SMEM_ALIAS=1` enables the restricted BF16,
  head-128, explicit split-K Raw two-CTA control.

The Raw control rejects append KV, ALiBi, softcap, and local attention rather
than silently changing those paths. Neither profile knob is a production
selection policy.

## Protocol and hardware scope

- Timings: physical GPU 4, NVIDIA A40, SM 8.6, ECC disabled.
- Shape: Q1, head dimension 128, GQA capture, explicit 20 splits, 8K and 64K
  contexts.
- Data: captured Llama layer-0 KV tensor
  `llama_layer0_cal2048_eval4096.pt`, SHA-256
  `ddc6b95feca30eda985dbdcb329d2ddc55d9320dc638c775bb8f4afce5b2d1d3`.
- Local inventory: GPU 0 through 7 are all NVIDIA A40. Seven expose 46,068 MiB
  with ECC enabled; GPU 4 exposes 49,140 MiB with ECC disabled.

The paired protocol is valid for the A40 attribution, but it cannot establish
performance on Ampere variants with different shared-memory limits, Hopper,
Blackwell, or future FA kernels. Those targets require an architecture-native
build and the unmodified Raw path, not an extrapolation from this table.

## Implementation and reproduction

The W16 launch-only padding control is in
`csrc/libtorch_stable/byte_v2/byte_v2_fa2_launch.cuh`. The Raw shared-tile
control is in the managed FA2 changes to `flash_fwd_kernel.h` and
`flash_fwd_launch_template.h`. The managed patch was regenerated from the exact
validated external source, applied to a pristine pinned-commit copy, and all
ten resulting source files matched the build source byte-for-byte. The CMake
idempotence script then reported that the patch was already applied.

Reproduce and analyze with:

```bash
profile/bytev2-static-w16-occupancy-portability-a40-20260830/harness/run_graph_matrix.sh

for variant in raw_default raw_alias2 w16_default w16_clamp1; do
  profile/bytev2-static-w16-occupancy-portability-a40-20260830/harness/collect_launch_ncu.sh "${variant}"
done

.venv/bin/python \
  profile/bytev2-static-w16-occupancy-portability-a40-20260830/analysis/analyze.py
```

The collectors deliberately refuse to overwrite archived results; run them in
a clean copy of this experiment directory when recollecting the matrix.

Principal artifacts:

- `analysis/SUMMARY.json`: machine-readable timing, NCU, correctness, and SASS
  audit.
- `results/`: all 18 formal timing/correctness records plus the unset-default
  smoke record.
- `ncu/`: four raw `.ncu-rep` files and their correctness/provenance records.
- `harness/`: the formal CUDA Graph matrix and NCU collectors.

## Next portability test

The next decisive experiment is not another A40 micro-tune. On at least one
different GPU, rebuild for that architecture and collect stock Raw and W16
resource limits first. If both stock kernels are equally resident, run the same
paired 8K/64K protocol and a split-count sweep. The acceptance criterion should
be a W16 advantage that survives equal stock residency; otherwise the short
context path needs an architecture-aware Raw/W16 dispatch rather than a single
global choice.
