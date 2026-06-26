# ByteV2 FA2-like Decode-to-MMA Kernel Plan

## Goal

在保持 ByteV2 当前 KV cache 压缩格式不变的前提下，把 decode attention
从当前的 scalar QK/PV 数据流逐步改成更接近 FlashAttention-2 的 tiled
MMA 数据流。

目标不是直接复用 FA2 kernel，而是把 FA2 的 dense global load 阶段替换成
ByteV2 的 decode-to-tile 阶段：

```text
FA2:
  dense K/V global tile -> FA2 shared/register tile -> MMA QK/PV

ByteV2 target:
  compressed K/V page -> decode -> FA2-like shared/register tile -> MMA QK/PV
```

## Current State

当前最快路径是 GQA4 no-outlier guarded split-k fast path：

- Source: `csrc/libtorch_stable/byte_v2/byte_v2_ops.cu`
- Kernel:
  `byte_v2_paged_decode_attention_split_k_gqa4_no_fallback_no_outlier_kernel`
- Scope:
  `num_heads=32`, `num_kv_heads=8`, `q_per_kv=4`, `head_dim=128`,
  `block_size=16`, no-outlier safe-page fast path.

当前数据流：

```text
QK:
  compressed K -> decode/stage shared_k_tile[row][dim]
  Q * K scalar dot
  warp shuffle reduction
  score -> shared_scores

Softmax:
  shared_scores -> shared_probs

PV:
  V(row, dim) scalar decode
  pv[q_group] += shared_probs[q_group][token] * V(row, dim)
```

当前主要问题：

- QK/PV 都是 CUDA-core scalar 指令，不是 tensor-core MMA。
- `shared_probs` 被每个 output-dim thread 反复读取，存在 fanout。
- V decode 现在被当前 mapping 复用到了 4 个 q_group，但仍然是 scalar PV。
- 之前的 shared V staging 和 warp-local PV 实验都说明：只改 shared memory
  layout 或只改 q_group mapping，不足以接近 FA2。

重要负例：

- Full V staging: 指令数和 shared memory/block 大幅增加，16k 明显变慢。
- Warp-local PV: 减少 P fanout，但 V decode 按 q_group 重复，16k 变慢。
- Decode-once shared-V + tiled-PV: V 不重复 decode，但 shared V tile 太大，
  occupancy 降低，scalar PV 指令仍然多，16k 变慢。

结论：

```text
ByteV2 不能靠 "shared V + scalar FMA" 接近 FA2。
下一步必须验证 tensor-core MMA 是否能抵消 decode/staging 成本。
```

## Constraints

第一阶段只支持当前性能路径，不处理通用功能：

- Only CUDA.
- Only BF16 query/output.
- Only `head_dim=128`, `head_dim_v=128`.
- Only `block_size=16`.
- Only `num_heads=32`, `num_kv_heads=8`, GQA4.
- Only no-fallback/no-outlier safe pages.
- First prototype can require `compute_block_n=64`, `partition_size=64`.
- Fallback/outlier path stays on existing scalar kernel.
- Do not route into E2E default path until isolated benchmark is positive.

Retention rule:

- Correctness must pass targeted CUDA tests.
- 16k p64/bn64 CUDA-event time should improve by at least 5%.
- NCU should show a real reduction in main-kernel instruction count or wall time.
- If an experiment regresses, revert code and only keep documentation/artifacts.

## FA2 Pieces To Mimic

SM80 FA2 forward path:

- V is partitioned as an MMA B fragment:
  `tOrVt = thr_mma.partition_fragment_B(...)`.
- Softmax P is reshaped to an MMA A fragment:
  `tOrP = make_tensor(rP.data(), convert_layout_acc_Aregs(...))`.
- PV is:
  `gemm_rs(acc_o, tOrP, tOrVt, ...)`.

Hopper FA2 path:

- Uses `tOrP/tOrV` fragments and `flash::gemm(...)`.
- V staging only works because it feeds a tiled MMA PV pipeline.

ByteV2 target:

- Decode compressed K/V into the layout expected by the MMA copy/fragments.
- Keep P in a fragment-friendly form.
- Use MMA for `P x V` and later for `Q x K`.

## Phased Plan

### Phase 0: Experiment Scaffolding

Purpose:

- Make future experiments explicit and easy to revert.
- Avoid changing the current production fast path until a prototype wins.

Changes:

1. Add a temporary explicit experiment flag only in microbench or tile policy.
2. Add a separate kernel name, for example:
   `byte_v2_paged_decode_attention_split_k_gqa4_mma_pv_experiment_kernel`.
3. Keep current scalar GQA4 kernel unchanged.
4. Keep the same `tmp_out/exp_sums` workspace and reduce kernel at first.

Tests:

```bash
cmake --build build-bytev2-stable --target _C_stable_libtorch -j 32
cp build-bytev2-stable/_C_stable_libtorch.abi3.so vllm/_C_stable_libtorch.abi3.so
CUDA_VISIBLE_DEVICES=5 .venv/bin/python -m pytest \
  tests/v1/attention/test_byte_v2_layout.py \
  -k "gqa_packed_cuda_matches_raw_reference or split_k_guarded_cuda_matches_raw_reference" -q
```

Benchmark baseline:

```bash
CUDA_VISIBLE_DEVICES=5 .venv/bin/python scripts/byte_v2_decode_microbench.py \
  --seq-lens 4096 8192 16384 \
  --partition-sizes 64 \
  --compute-block-ns 64 \
  --assume-no-outlier \
  --guarded-split \
  --gqa-packed \
  --include-flash \
  --iters 300 \
  --warmup 80 \
  --output-jsonl profiles/byte_v2_mma_baseline_sweep_gpu5_YYYYMMDD.jsonl
```

Exit criteria:

- Current retained kernel performance is unchanged.
- Experiment flag is off by default.

### Phase 1: Decode-To-MMA Layout Microkernel

Purpose:

- Verify that ByteV2 compressed payload can be decoded directly into an
  MMA-friendly shared layout without first materializing a plain
  `shared_v_tile[row][dim]`.

Scope:

- V only.
- No softmax/PV yet.
- Use no-outlier pages.
- Compare decoded shared tile against dense raw V reference.

Implementation idea:

1. Add a debug-only or test-only CUDA kernel:
   `byte_v2_decode_v_to_mma_layout_debug`.
2. Input:
   compressed `kv_cache`, `block_tables`, `seq_lens`.
3. Output:
   a dense/debug tensor representing the logical V tile after reversing the
   MMA layout, or a checksum per tile.
4. Decode directly from current format:

```text
low[256] + code_packed[128] + base -> bf16 bits
```

1. Store into the same logical layout that the future MMA PV path will read.

Important:

- Do not decode into row-major `[token][dim]` and then transpose. That repeats
  the failed shared-V staging pattern.
- The write pattern should be designed from the MMA copy/fragments backward.

Expected risk:

- The current compressed format groups data by 16-token x 16-dim codec tiles.
  MMA PV wants K dimension along tokens and N dimension along V dims. A direct
  decode-to-MMA layout may require nontrivial address mapping.

Exit criteria:

- Correctness against raw dense V.
- NCU on this debug kernel shows coalesced enough global reads and acceptable
  shared stores.
- No retention decision yet; this phase only proves layout feasibility.

### Phase 2: PV-Only MMA Prototype

Purpose:

- Keep current scalar QK and online softmax.
- Replace only PV scalar accumulation with MMA `P x V`.
- This tests whether tensor-core PV can offset ByteV2 V decode/staging cost.

Data shape:

```text
P:  q_group x token = 4 x 64
V:  token x dim     = 64 x 128
O:  q_group x dim   = 4 x 128
```

MMA issue:

- GQA4 gives M=4, while tensor cores prefer larger M tiles.
- First prototype can pad M to 8 or 16.
- Wasted lanes are acceptable for the first measurement if instruction count
  drops enough.

Implementation sketch:

1. Keep current QK:
   - K pair staging remains.
   - `shared_scores` and online softmax remain initially.
2. Convert current `shared_probs[4][64]` into a BF16/FP16 MMA A tile.
3. Decode V directly into MMA B tile layout.
4. Run MMA PV:

```text
P_tile_bf16(16 padded x 64) x V_tile_bf16(64 x 128)
  -> O_acc_fp32(16 padded x 128)
```

1. Write only valid q_group rows `0..3` to `tmp_out`.

Precision:

- FA2 also converts P to lower precision for MMA paths.
- Accept the existing raw FA2 tolerance target first:
  max abs diff around `1e-3` to `5e-3`, then tighten if possible.

Implementation options:

- Option A: use CUTLASS/CUTE fragments similar to FA2 SM80 code.
- Option B: write a smaller inline WMMA/MMA prototype for fixed BF16 shapes.
- Option A is closer to FA2 but requires more boilerplate and careful layout.
- Option B is faster to validate but may not represent final performance.

Metrics to collect:

- CUDA event sweep: 4k/8k/16k.
- NCU main-kernel:
    - duration
    - executed instructions
    - global load requests/sectors
    - shared wavefronts
    - registers/thread
    - shared memory/block
    - occupancy limits

Retention criteria:

- If 16k improves >= 5%, keep prototype behind experimental flag and continue.
- If it regresses, revert and document. That would imply QK/PV must be
  rewritten together or the current format is too costly for PV-only MMA.

### Phase 3: QK MMA Prototype

Purpose:

- Replace scalar QK dot product with MMA.
- This is harder than PV-only because K decode and Q fragment layout must feed
  the score fragment used by online softmax.

Data shape:

```text
Q: q_group x dim = 4 x 128
K: token x dim   = 64 x 128
S: q_group x token = 4 x 64
```

Implementation sketch:

1. Load Q into a padded M tile, likely `M=8` or `M=16`.
2. Decode K into MMA B layout.
3. Run MMA QK:

```text
Q_tile_bf16(16 padded x 128) x K_tile_bf16^T(128 x 64)
  -> S_acc_fp32(16 padded x 64)
```

1. Feed valid rows `0..3` into online softmax.
2. Keep PV as current scalar path initially to isolate QK.

Risks:

- QK M=4 padding may waste tensor core work.
- K decode-to-layout may be more expensive than current K pair staging.
- Softmax integration must avoid materializing more shared state than today.

Exit criteria:

- Correctness vs raw reference.
- Benchmark decides whether QK MMA is independently useful.

### Phase 4: Combined QK+PV MMA Mainloop

Purpose:

- Combine the winning QK and PV MMA pieces.
- Make the dataflow structurally closer to FA2:

```text
decode K tile -> MMA QK -> score fragment
online softmax -> P fragment
decode V tile -> MMA PV -> O fragment
```

Key requirement:

- Avoid writing `shared_probs` as the primary P transport.
- P should stay in register fragments or be written to shared only in an
  MMA-friendly layout.

What to remove from current path:

- Scalar PV loop over `row`.
- Per-output-dim repeated `shared_probs[q_group][tile_offset]` loads.
- Scalar QK shuffle reduction, if QK MMA is positive.

What may remain:

- Split-k partitioning and existing reduce kernel, initially.
- Page unsafe guard, but only to route unsafe pages away from MMA path.

Exit criteria:

- 16k p64/bn64 wins over section 24 retained kernel.
- Instruction count moves meaningfully toward FA2.
- Occupancy loss from larger shared tiles is acceptable.

### Phase 5: Split-K and Combine Cleanup

Purpose:

- Once main kernel improves, reduce split-k/reduce overhead.

Possible changes:

1. Tune `partition_size` again after MMA changes.
2. Add a specialized combine for fixed GQA4/head_dim=128.
3. Consider FA2-like `num_splits` heuristic instead of fixed partition size.
4. Avoid workspace writes where sequence length has only one partition.

This phase should not start before main-kernel time improves; otherwise it
will hide the real bottleneck.

## Code Touchpoints

Primary ByteV2 files:

- `csrc/libtorch_stable/byte_v2/byte_v2_ops.cu`
    - current decode kernels
    - launchers
    - temporary experiment kernels
- `scripts/byte_v2_decode_microbench.py`
    - experiment flags
    - sweep output
- `tests/v1/attention/test_byte_v2_layout.py`
    - targeted correctness coverage

FA2 reference files:

- `/mnt/sdb/yxz/ByteV2/flash-attention/csrc/flash_attn/src/flash_fwd_kernel.h`
    - SM80 `tOrP`, `tOrVt`, `gemm_rs`
- `/mnt/sdb/yxz/ByteV2/flash-attention/hopper/mainloop_fwd_sm90_tma_gmma_ws.hpp`
    - Hopper `tOrP/tOrV`, `flash::gemm`

Documentation:

- `/mnt/sdb/yxz/ByteV2/Doc/ByteV2_current_vs_fa2_profile_20260624.md`
    - append benchmark/profile results
- This document
    - keep as the implementation roadmap

## Suggested First Concrete Experiment

Do not start with full QK+PV. Start with PV-only MMA:

1. Add an experimental GQA4 kernel:
   `byte_v2_paged_decode_attention_split_k_gqa4_mma_pv_experiment_kernel`.
2. Keep current QK/softmax intact.
3. Convert `shared_probs[4][64]` to a padded BF16 P tile.
4. Decode V directly into the MMA B tile layout.
5. Run fixed-shape BF16 MMA for `P x V`.
6. Write valid q_group rows to `tmp_out`.
7. Compare:

```bash
CUDA_VISIBLE_DEVICES=5 .venv/bin/python scripts/byte_v2_decode_microbench.py \
  --seq-lens 4096 8192 16384 \
  --partition-sizes 64 \
  --compute-block-ns 64 \
  --assume-no-outlier \
  --guarded-split \
  --gqa-packed \
  --include-flash \
  --iters 300 \
  --warmup 80 \
  --output-jsonl profiles/byte_v2_exp_mma_pv_sweep_gpu5_YYYYMMDD.jsonl
```

1. Collect NCU for 16k if event timing is promising or surprising.

Expected result:

- If PV-only MMA does not improve, then preserving the current compression
  format while only changing compute is probably insufficient.
- If PV-only MMA improves, then proceed to QK MMA and combined mainloop.

## Decision Matrix

| Result | Interpretation | Next step |
| --- | --- | --- |
| PV-only MMA improves >= 5% | Tensor-core PV can pay for decode/staging | Keep flag, implement QK MMA |
| PV-only MMA neutral | Need combined QK+PV before deciding | Try QK MMA only if instruction count drops |
| PV-only MMA regresses with high shared memory | Decode-to-layout/shared footprint too expensive | Try smaller V tile or direct fragment decode |
| PV-only MMA regresses with high instructions | MMA wrapper/layout conversion too costly | Revisit format or abandon MMA path |
| QK MMA improves but PV does not | Keep scalar PV reuse; optimize QK path | Combine QK MMA + scalar PV |
| Both improve separately | Build combined FA2-like mainloop | Retune split-k |

## Open Technical Questions

1. Which MMA API should be used first: CUTE/CUTLASS fragments or a smaller
   fixed-shape WMMA/MMA prototype?
2. Can V be decoded directly into fragment/registers for a smaller tile instead
   of staging a full `64 x 128` shared V tile?
3. Is M=4 padding acceptable, or do we need to pack multiple kv heads/sequences
   into one MMA M tile?
4. Can current `low + code_packed + base` layout be decoded coalesced enough
   for MMA staging?
5. Should probabilities be converted to BF16 for PV MMA exactly like FA2, or
   should the first prototype keep P in FP32 and use a different MMA path?

## Working Conclusion

The correct direction is:

```text
compressed page -> decode directly into MMA-friendly tile -> FA2-like MMA QK/PV
```

The incorrect direction is:

```text
compressed page -> decode into plain shared row-major tile -> scalar tiled PV
```

Previous experiments already ruled out the second path. The next useful work is
a PV-only MMA prototype that proves whether tensor-core computation can overcome
ByteV2's decode/staging overhead while keeping the current compression format.

## 2026-06-25 PV-Only WMMA Prototype Result

The first concrete experiment was implemented as a temporary tenth
`tile_policy` flag and a `--gqa-packed-mma-pv` microbench option. It kept the
current GQA4 QK and softmax path, converted `P[4, 64]` into a padded BF16
`16 x 64` tile, decoded `V[64, 128]` into shared BF16, and used WMMA
`m16n16k16` for `P x V`.

Correctness was acceptable for the p64/bn64 no-outlier case:

- scalar GQA4 max abs vs raw reference: `4.88e-4`
- WMMA PV max abs vs raw reference: `5.62e-4`
- WMMA PV max abs vs scalar GQA4: `9.77e-4`

Performance regressed clearly, so the code was not retained:

| seq len | scalar GQA4 p64/bn64 | PV-only WMMA | result |
| ---: | ---: | ---: | --- |
| 4096 | 0.0758 ms | 0.1505 ms | 98.6% slower |
| 8192 | 0.1434 ms | 0.2509 ms | 75.0% slower |
| 16384 | 0.2294 ms | 0.4516 ms | 96.9% slower |

Artifacts:

- `profiles/byte_v2_decode_microbench_gqa4_scalar_p64_bn64_20260625.jsonl`
- `profiles/byte_v2_decode_microbench_gqa4_mma_pv_p64_bn64_20260625.jsonl`
- `profiles/byte_v2_after_mma_pv_revert_sanity_16384_p64_20260625.jsonl`

Conclusion:

This rules out a naive "stage full V tile to row-major shared memory, pad M=4
to M=16, then call WMMA" approach. It pays the full decode/staging cost plus
extra shared memory traffic and only uses 4 valid rows of the MMA M tile. The
next MMA attempt should avoid full row-major V staging and M=4 waste: decode
directly into the fragment/shared layout expected by FA2-style CUTE kernels, or
pack multiple q groups/kv heads/sequences into a fuller MMA M tile.

After reverting the temporary code, the retained 16k p64/bn64 kernel measured
`0.2294 ms`, matching the pre-experiment baseline.

## 2026-06-25 Dedicated FA2-Like Experiment Kernel

The PV-only WMMA structure was reintroduced as a separate non-default kernel so
future FA2-like work can proceed without touching the retained scalar GQA4
kernel.

New kernel:

- `byte_v2_paged_decode_attention_split_k_gqa4_fa2_like_no_fallback_no_outlier_kernel`
- Launched only when the tenth `tile_policy` field is set:
  `(16, 16, 16, 64, 128, 128, 0, 1, 1, 1)`
- Python env route:
  `BYTE_V2_DECODE_GQA_FA2_LIKE=1`
- Microbench route:
  `--gqa-packed --gqa-fa2-like`

Current constraints:

- `compute_block_n == 64`
- `partition_size == 64`
- GQA4 only: 32 query heads / 8 KV heads
- no-fallback/no-outlier fast path
- split-k only
- QK and softmax still reuse the retained scalar implementation
- PV is FA2-like at a coarse level: padded BF16 P tile, decoded BF16 V tile,
  and WMMA `m16n16k16` for `P x V`

Sanity commands:

```bash
cmake --build build-bytev2-stable --target _C_stable_libtorch -j 32
cp build-bytev2-stable/_C_stable_libtorch.abi3.so vllm/_C_stable_libtorch.abi3.so
.venv/bin/python -m pytest \
  tests/v1/attention/test_byte_v2_layout.py::test_byte_v2_paged_decode_attention_split_k_gqa_packed_cuda_matches_raw_reference -q
.venv/bin/python scripts/byte_v2_decode_microbench.py \
  --seq-lens 4096 \
  --partition-sizes 64 \
  --compute-block-ns 64 \
  --assume-no-outlier \
  --gqa-packed \
  --gqa-fa2-like \
  --no-outlier-inputs \
  --warmup 20 \
  --iters 50 \
  --output-jsonl profiles/byte_v2_gqa4_fa2_like_sanity_4096_p64_20260625.jsonl
```

This kernel is not expected to be faster yet. Its purpose is to provide a safe
place to incrementally replace the current scalar pieces with FA2-like tiled
QK/PV work distribution, CUTE fragments, and eventually direct decode-to-MMA
layout.

## 2026-06-25 FA2-Like PV N-Tile Staging Update

The dedicated FA2-like kernel was moved one step closer to FA2's tiled PV
structure.

Previous experiment-kernel PV shape:

```text
decode full V[64, 128] into shared row-major
for each 16-column N tile:
  WMMA(P[16,64], V[64,16])
```

Updated experiment-kernel PV shape:

```text
build shared P[16,64]
for each 16-column N tile:
  decode only V[64,16] for that tile into warp-local shared storage
  WMMA(P[16,64], V[64,16])
  write the tile result
```

Implementation details:

- `shared_v_mma` changed from `V[64][128]` to `V[4][64][16]`.
- Each warp owns one 16-column N tile at a time.
- V tile staging now happens immediately before the corresponding WMMA.
- The retained scalar GQA4 kernel is still unchanged.
- The experiment route remains:
  `--gqa-packed --gqa-fa2-like` or `BYTE_V2_DECODE_GQA_FA2_LIKE=1`.

Sanity:

- Existing default GQA4 CUDA test still passed:
  `.venv/bin/python -m pytest tests/v1/attention/test_byte_v2_layout.py::test_byte_v2_paged_decode_attention_split_k_gqa_packed_cuda_matches_raw_reference -q`
  passed: 2 passed.
- Direct p64 correctness check stayed unchanged:
    - scalar GQA4 max abs vs raw reference: `4.88e-4`
    - FA2-like N-tile max abs vs raw reference: `5.62e-4`
    - FA2-like N-tile max abs vs scalar GQA4: `9.77e-4`
- 4096 p64/bn64 sanity median:
  `0.1347 ms`

Artifact:

- `profiles/byte_v2_gqa4_fa2_like_ntile_sanity_4096_p64_20260625.jsonl`

Interpretation:

This is still not a final FA2 port. QK and softmax remain scalar, P is still
materialized through shared memory, and the V tile is still decoded to a simple
WMMA row-major layout rather than a CUTE/FA2 fragment layout. The useful
progress is structural: the experiment kernel no longer requires full
`64 x 128` V shared staging before PV, so future work can replace the per-N
tile staging with direct decode-to-fragment or CUTE-style shared layout.

## 2026-06-25 P-Direct Staging Experiment Not Retained

After the N-tile PV staging change, one isolated P-path experiment was tried:
write BF16 P directly while computing softmax probabilities, instead of first
writing `shared_probs[4][64]` and then materializing `shared_p_mma[16][64]`.

Tested shape:

```text
softmax:
  shared_p_mma[q_group][tile_offset] = bf16(exp(score - m))
PV:
  WMMA(P[16,64], V[64,16])
```

This removes the explicit `shared_probs -> shared_p_mma` conversion loop and
avoids writing padded P rows in the experiment kernel. The change was built and
tested successfully, but its benchmark result was noise-level:

- Previous retained N-tile median at 4096 p64/bn64: `0.134656 ms`
- P-direct experiment median at 4096 p64/bn64: `0.134144 ms`
- Relative change: about `0.4%`

Correctness stayed within the same practical tolerance:

- scalar GQA4 max abs vs simple raw BF16 reference: `0.0`
- FA2-like P-direct max abs vs simple raw BF16 reference: `9.77e-4`
- FA2-like P-direct max abs vs scalar GQA4: `9.77e-4`

Artifact:

- `profiles/byte_v2_gqa4_fa2_like_p_direct_sanity_4096_p64_20260625.jsonl`

Decision:

The code change was not retained because the speedup was far below the 5%
retention threshold and the larger gap is not the extra P materialization loop.
The retained experiment kernel remains the previous N-tile PV staging version.
Next work should target larger structural differences: direct decode into
fragment-friendly V layout, CUTE-style tiled MMA, or eliminating scalar QK/PV
work distribution rather than only removing the P conversion loop.

## 2026-06-25 FA2 Source Recheck and Updated Execution Order

Local FA2 source files checked:

- `/mnt/sdb/yxz/ByteV2/flash-attention/csrc/flash_attn/src/flash_fwd_kernel.h`
- `/mnt/sdb/yxz/ByteV2/flash-attention/csrc/flash_attn/src/kernel_traits.h`
- `/mnt/sdb/yxz/ByteV2/flash-attention/csrc/flash_attn/src/softmax.h`
- `/mnt/sdb/yxz/ByteV2/flash-attention/csrc/flash_attn/src/utils.h`

Relevant FA2 structure:

```text
Q, K, V are copied into CUTE shared-memory layouts.
QK:
  acc_s = cute tiled MMA(Q, K)
softmax:
  acc_s is normalized in register layout
  acc_o is rescaled when max changes
PV:
  rP = convert_type<Element>(acc_s)
  tOrP = convert_layout_acc_Aregs(rP.layout)
  acc_o = gemm_rs(tOrP, V)
epilogue:
  normalize acc_o and write O/LSE
```

This confirms that the previous P-direct experiment was too small: FA2 does
not optimize by merely skipping a `shared_probs -> shared_p_mma` loop. The key
structural difference is that P is a register fragment derived from `acc_s` and
fed directly into the PV MMA path.

Updated migration order for the dedicated ByteV2 FA2-like experiment kernel:

1. Add a CUTE/TiledMMA PV skeleton that compiles inside the stable libtorch
   extension. Keep it isolated to the non-default FA2-like route.
2. Replace the current `nvcuda::wmma` PV wrapper with a CUTE-style tiled MMA
   path while preserving current scalar QK/softmax behavior.
3. Change ByteV2 V decode so it writes directly into the CUTE-compatible shared
   layout used by PV, instead of row-major `shared_v_mma`.
4. Replace `shared_p_mma` materialization with a register-fragment P path,
   following FA2's `convert_layout_acc_Aregs + gemm_rs` pattern.
5. Move QK from scalar dot/shuffle reduction to tiled MMA.
6. After the kernel is structurally closer to FA2, sweep `kBlockN`/partition
   settings again. FA2 split-KV hdim128 uses `kBlockN=128` when split-KV is
   active and `kBlockN=64` for the single-split standard-aligned path, so
   ByteV2 should not assume p64/bn64 is final.

Directly copying FA2 is not appropriate because ByteV2 has compressed KV cache
decode and currently groups GQA4 rows inside one CTA. The target is therefore
not a line-for-line port. The target is to migrate the same dataflow:
fragment-friendly shared layout, register P, CUTE tiled MMA, and online
softmax/rescale.

## 2026-06-25 CUTE PV Skeleton Compile Step

First execution step completed:

- Added CUTLASS/CUTE includes to the ByteV2 stable libtorch extension.
- Added `ByteV2Fa2LikeCutePvTraits`, modeled after FA2's forward traits:
    - BF16 element type.
    - CUTE `TiledMMA`.
    - `SmemLayoutKV`.
    - `SmemLayoutVtransposed`.
    - `SmemLayoutVtransposedNoSwizzle`.
- Bound the traits to the dedicated FA2-like experiment kernel with:
  `ByteV2Fa2LikeCutePvTraits<16, Policy::ComputeBlockN, Policy::HeadDimV, 1>`.
- The default retained GQA4 kernel still does not use this path.

Build environment correction:

- Local GPU: NVIDIA A40, capability `8.6`.
- Previous `build-bytev2-stable` cache used `CMAKE_CUDA_ARCHITECTURES=75`.
- Reconfigured the build directory with:

```bash
cmake -S . -B build-bytev2-stable -DCMAKE_CUDA_ARCHITECTURES=86
```

- Clean-rebuilt `_C_stable_libtorch` so ByteV2 compiles under `sm_86`:

```bash
cmake --build build-bytev2-stable --target _C_stable_libtorch --clean-first -j 32
cp build-bytev2-stable/_C_stable_libtorch.abi3.so vllm/_C_stable_libtorch.abi3.so
```

Verification:

- Build passed. The only ByteV2 warning left is the pre-existing unused
  `byte_v2_reshape_and_cache_kernel` warning.
- Default GQA4 split-k test:

```bash
.venv/bin/python -m pytest \
  tests/v1/attention/test_byte_v2_layout.py::test_byte_v2_paged_decode_attention_split_k_gqa_packed_cuda_matches_raw_reference -q
```

passed: 2 passed.

- FA2-like experiment sanity:

```bash
.venv/bin/python scripts/byte_v2_decode_microbench.py \
  --seq-lens 4096 \
  --partition-sizes 64 \
  --compute-block-ns 64 \
  --assume-no-outlier \
  --gqa-packed \
  --gqa-fa2-like \
  --no-outlier-inputs \
  --warmup 10 \
  --iters 20 \
  --output-jsonl \
  profiles/byte_v2_gqa4_fa2_like_cute_probe_sm86_sanity_4096_p64_20260625.jsonl
```

Result:

- 4096 p64/bn64 median: `0.134656 ms`
- `max_abs_diff_vs_first`: `0.0`

Interpretation:

This step intentionally should not improve runtime: it only proves that the
ByteV2 stable extension can compile FA2-like CUTE PV traits under the correct
local architecture. The next code step is to replace the current
`nvcuda::wmma` PV wrapper with a CUTE `gemm_rs`-style path in the non-default
FA2-like experiment kernel.

## 2026-06-26 CUTE PV and Register-P Step

Implemented inside the non-default FA2-like experiment route:

- Replaced the FA2-like kernel's `nvcuda::wmma` PV wrapper with a CUTE
  `TiledMMA` path.
- Changed ByteV2 V decode to write directly into a swizzled CUTE-compatible
  shared-memory layout, then consume it through the transposed/no-swizzle view
  used by the MMA copy atom.
- Replaced shared P materialization with a register P fragment:
    - construct a 16x64 CUTE accumulator-shaped fragment;
    - fill it from the current scalar softmax probabilities using the CUTE
    `partition_C` coordinate mapping;
    - convert the layout with `convert_layout_acc_Aregs`;
    - call a local `gemm_rs` helper so PV loads only V from shared memory.
- Removed the old FA2-like shared-P staging and stale `__nv_bfloat16` helper.
- The default retained GQA4 packed kernel remains unchanged.

Verification:

```bash
cmake --build build-bytev2-stable --target _C_stable_libtorch -j 32
cp build-bytev2-stable/_C_stable_libtorch.abi3.so vllm/_C_stable_libtorch.abi3.so
.venv/bin/python -m pytest tests/v1/attention/test_byte_v2_layout.py -v
```

Result:

- Build passed under `sm_86`.
- Test result: `111 passed, 1 skipped, 16 warnings`.
- The remaining CUDA build warning is the pre-existing unused
  `byte_v2_reshape_and_cache_kernel` warning.

4096 p64/bn64 FA2-like progression:

| route | median | diff |
| --- | ---: | ---: |
| CUTE skeleton, runtime still N-tile WMMA | 0.134656 ms | 0.0 |
| CUTE PV, row-major shared path | 0.140288 ms | 0.0 |
| CUTE PV, swizzled/LDSM V path | 0.138240 ms | 0.0 |
| CUTE PV + register P + `gemm_rs` | 0.133120 ms | 0.0 |

Artifacts:

- `profiles/byte_v2_gqa4_fa2_like_cute_pv_rowmajor_4096_p64_20260626.jsonl`
- `profiles/byte_v2_gqa4_fa2_like_cute_pv_swizzled_4096_p64_20260626.jsonl`
- `profiles/byte_v2_gqa4_fa2_like_cute_preg_clean_4096_p64_20260626.jsonl`

Context sweep at p64/bn64:

| seq len | default GQA4 packed | FA2-like CUTE reg-P |
| ---: | ---: | ---: |
| 256 | 0.033792 ms | 0.041984 ms |
| 512 | 0.033792 ms | 0.041984 ms |
| 1024 | 0.038912 ms | 0.049152 ms |
| 2048 | 0.055296 ms | 0.070656 ms |
| 4096 | 0.075264 ms | 0.133120 ms |

Artifacts:

- `profiles/byte_v2_gqa4_default_sweep_p64_bn64_20260626.jsonl`
- `profiles/byte_v2_gqa4_fa2_like_cute_preg_sweep_p64_bn64_20260626.jsonl`

Interpretation:

The CUTE/reg-P step is structurally useful and slightly faster than the prior
FA2-like WMMA N-tile experiment, so it is retained in the isolated FA2-like
route. It is not yet the fastest ByteV2 decode kernel. The default GQA4 packed
path remains much faster because it avoids the FA2-like experiment's current
M=16 padded PV shape and repeated V decode/work for the four GQA rows.

Completed migration-order items:

1. CUTE/TiledMMA PV skeleton.
2. CUTE-style tiled PV path.
3. V decode directly into the CUTE-compatible shared layout.
4. Register-fragment P path with `convert_layout_acc_Aregs + gemm_rs`.

Remaining high-impact items:

1. Move QK from scalar dot/shuffle reduction to tiled MMA.
2. Combine QK and PV into one FA2-like online mainloop so P is produced from an
   MMA accumulator instead of reconstructed from `shared_probs`.
3. Revisit CTA shape so the experiment does not pay M=16 tensor-core work for
   only four active GQA rows unless the work is amortized by a larger combined
   mainloop.

Do not promote the FA2-like route to default based on this step. The next
experiment should be a separate QK-MMA prototype or a combined QK/PV prototype
with an explicit correctness and timing gate.

## 2026-06-26 QK-MMA Prototype Attempt

Attempted but not retained.

Prototype shape:

- Keep the current FA2-like route isolated behind `--gqa-fa2-like`.
- Stage Q as a padded 16x128 BF16 CUTE shared tile.
- Stage the full 64x128 K tile directly from ByteV2 pages into a CUTE shared
  layout.
- Use CUTE MMA to compute a 16x64 score fragment and write only rows 0..3 back
  to the existing `shared_scores`.
- Leave scalar softmax and the already-working CUTE/reg-P PV path unchanged.

Results:

- A one-warp 16x64 QK attempt failed to compile:
    - CUTE LDSM copy reported `src failed to vectorize into registers`;
    - B-fragment N dimension did not match the C fragment.
- A revised four-warps-along-N attempt also failed with the same class of
  constraints:
    - `SM75_U32x4_LDSM_N` source layout was incompatible with the selected K
    shared layout;
    - `cute::gemm` asserted `size<1>(B) == size<2>(C)`.
- The QK-MMA code was fully reverted.
- The retained FA2-like route is still the CUTE PV + register-P version.

Post-revert verification:

```bash
cmake --build build-bytev2-stable --target _C_stable_libtorch -j 32
cp build-bytev2-stable/_C_stable_libtorch.abi3.so vllm/_C_stable_libtorch.abi3.so
.venv/bin/python scripts/byte_v2_decode_microbench.py \
  --seq-lens 4096 \
  --partition-sizes 64 \
  --compute-block-ns 64 \
  --assume-no-outlier \
  --gqa-packed \
  --gqa-fa2-like \
  --no-outlier-inputs \
  --warmup 10 \
  --iters 20 \
  --output-jsonl \
  profiles/byte_v2_gqa4_fa2_like_cute_preg_after_qk_revert_4096_p64_20260626.jsonl
```

Result:

- median: `0.133120 ms`
- `max_abs_diff_vs_first`: `0.0`

Conclusion:

QK-MMA should not be reattempted as an ad hoc local replacement of the scalar
QK loop. The next QK step needs a faithful FA2-style Q/K shared layout and MMA
partition design first, preferably copied from the local FA2 `Kernel_traits`
shape parameters and reduced into a small standalone compile probe before it is
placed back into the ByteV2 kernel.

## 2026-06-26 FA2-Style QK-MMA Integration

Status: implemented as a non-default experiment, not promoted to the retained
fast path.

What changed:

- Added `ByteV2Fa2LikeCuteQkTraits`, mirroring the FA2 forward Q/K shared
  layout and copy atom choices for the fixed ByteV2 experiment shape:
  `M=16`, `N=64`, `K=128`, one warp.
- Added a local `byte_v2_cute_gemm_smem` helper for shared-Q/shared-K CUTE
  GEMM.
- Added `UseQkMma` as a template parameter to the isolated FA2-like GQA4
  kernel.
- Added `tile_policy[10]` as the QK-MMA subpath flag:
    - `tile_policy[8]`: GQA packed
    - `tile_policy[9]`: FA2-like experiment
    - `tile_policy[10]`: FA2-like QK-MMA subpath
- Added `--gqa-fa2-qk-mma` to `scripts/byte_v2_decode_microbench.py`.
- Wired both unguarded and guarded split-k dispatch to instantiate the QK-MMA
  experiment when the flag is set.
- Added a CUDA correctness test that covers FA2-like with and without QK-MMA:
  `test_byte_v2_paged_decode_attention_split_k_gqa_fa2_like_cuda_matches_raw_reference`.

Current QK-MMA dataflow:

```text
Q:
  BF16 query -> padded 16x128 CUTE shared Q tile

K:
  compressed ByteV2 K page -> decode -> 64x128 CUTE shared K tile

QK:
  CUTE GEMM shared-Q x shared-K -> acc_s(16x64)
  rows 0..3 -> shared_scores

Softmax/PV:
  existing shared_scores -> shared_probs path
  existing CUTE PV + register-P path
```

Verification:

```bash
cmake --build build-bytev2-stable --target _C_stable_libtorch -j 32
cp build-bytev2-stable/_C_stable_libtorch.abi3.so vllm/_C_stable_libtorch.abi3.so
.venv/bin/python -m pytest tests/v1/attention/test_byte_v2_layout.py -v
```

Result:

- Build passed under `sm_86`.
- The only CUDA build warning is the pre-existing unused
  `byte_v2_reshape_and_cache_kernel`.
- Test result after adding QK-MMA coverage:
  `113 passed, 1 skipped, 16 warnings`.

4096 p64/bn64 measurements:

| route | median | max diff |
| --- | ---: | ---: |
| FA2-like CUTE PV + register P | 0.133120 ms | 0.0 |
| FA2-like + QK-MMA | 0.179200 ms | 0.0 |
| guarded FA2-like + QK-MMA | 0.181776 ms | 0.0 |

Artifacts:

- `profiles/byte_v2_gqa4_fa2_like_final_4096_p64_20260626.jsonl`
- `profiles/byte_v2_gqa4_fa2_like_qk_mma_final_restored_4096_p64_20260626.jsonl`
- `profiles/byte_v2_gqa4_fa2_like_qk_mma_guarded_4096_p64_20260626.jsonl`

Negative sub-experiment:

- Tried computing softmax directly from the QK-MMA `acc_s` fragment to skip the
  `shared_scores` write/read.
- Correctness still passed with `max_abs_diff_vs_first=0.0`.
- Runtime regressed from `0.179200 ms` to `0.187392 ms`.
- The direct-softmax change was reverted.

Conclusion:

The previously unfinished QK-MMA item is now complete as a correct, explicit,
test-covered experiment path. It is not a retained performance path because it
is slower than the CUTE PV/register-P FA2-like route and much slower than the
default GQA4 packed kernel.

The remaining FA2-like gap is not a missing local patch. The current ByteV2
CTA shape computes QK-MMA in warp 0, while PV uses four warps to cover the
128 output dimensions. In FA2, the score accumulator, online softmax, register
P, and PV are shaped as one coherent tiled mainloop. In the current ByteV2
experiment, keeping P entirely in registers across QK and all PV warps would
require either cross-warp shared materialization or a different CTA/MMA
partition. The former was measured as negative; the latter is a new kernel
design task.

Updated completed migration-order items:

1. CUTE/TiledMMA PV skeleton.
2. CUTE-style tiled PV path.
3. V decode directly into the CUTE-compatible shared layout.
4. Register-fragment P path with `convert_layout_acc_Aregs + gemm_rs`.
5. FA2-style Q/K shared-layout QK-MMA experiment behind an explicit flag.

Remaining structural work:

1. Redesign CTA/MMA partitioning so QK and PV share a coherent FA2-like
   accumulator/register-P pipeline instead of broadcasting P through shared
   memory.
2. Revisit whether the GQA4 decode kernel should use an M=16 padded tensor-core
   shape at all, or use a different grouped-query CTA shape that amortizes the
   padded rows.
3. Only promote a new FA2-like path after it beats the default GQA4 packed
   kernel, not merely the older FA2-like experiment.

## 2026-06-26 FA2-Style Single-Warp Mainloop

Status: implemented as a second non-default experiment, still not promoted to
the retained fast path.

What changed:

- Added `tile_policy[11]` as the FA2-style mainloop subpath flag.
- Added `--gqa-fa2-mainloop` to the microbench script. It requires
  `--gqa-fa2-qk-mma`.
- Extended the isolated FA2-like GQA4 kernel template with
  `UseFa2Mainloop`.
- Added full-head PV support to the existing CUTE PV traits:
    - full `SmemLayoutKV` storage size;
    - V transposed/no-swizzle view;
    - `SM75_U16x8_LDSM_T` copy atom for V.
- Added correctness coverage for three FA2-like modes:
    - CUTE PV + register P, scalar QK;
    - QK-MMA + shared-score softmax/PV;
    - QK-MMA + FA2-style single-warp mainloop.

Mainloop dataflow:

```text
all 128 CTA threads:
  decode/stage Q -> CUTE shared Q
  decode/stage K -> CUTE shared K
  decode/stage full V(64x128) -> CUTE shared V

warp0:
  QK CUTE MMA -> acc_s(16x64)
  row-wise softmax directly on acc_s
  acc_s -> BF16 register P
  full-head PV CUTE gemm_rs -> acc_o(16x128)
  write rows 0..3 to tmp_out
  write partition LSE to exp_sums
```

This is structurally closer to FA2 than the previous QK-MMA path because
scores/P do not round-trip through shared memory before PV.

Verification:

```bash
cmake --build build-bytev2-stable --target _C_stable_libtorch -j 32
cp build-bytev2-stable/_C_stable_libtorch.abi3.so vllm/_C_stable_libtorch.abi3.so
.venv/bin/python -m pytest tests/v1/attention/test_byte_v2_layout.py -v
.venv/bin/pre-commit run ruff-check --files \
  tests/v1/attention/test_byte_v2_layout.py \
  scripts/byte_v2_decode_microbench.py
```

Result:

- Build passed under `sm_86`.
- The only CUDA build warning is the pre-existing unused
  `byte_v2_reshape_and_cache_kernel`.
- Test result: `114 passed, 1 skipped, 16 warnings`.
- Ruff passed.

4096 p64/bn64 measurements:

| route | median | max diff |
| --- | ---: | ---: |
| default GQA4 packed | 0.076800 ms | 0.0 |
| FA2-like CUTE PV + register P | 0.133120 ms | 0.0 |
| FA2-like + QK-MMA shared-score path | 0.179200 ms | 0.0 |
| FA2-like + QK-MMA + single-warp mainloop | 0.145408 ms | 0.0 |
| guarded FA2-like + QK-MMA + single-warp mainloop | 0.143360 ms | 0.0 |

Artifacts:

- `profiles/byte_v2_gqa4_default_after_mainloop_4096_p64_20260626.jsonl`
- `profiles/byte_v2_gqa4_fa2_like_after_mainloop_base_4096_p64_20260626.jsonl`
- `profiles/byte_v2_gqa4_fa2_like_qk_mma_after_mainloop_4096_p64_20260626.jsonl`
- `profiles/byte_v2_gqa4_fa2_like_mainloop_4096_p64_20260626.jsonl`
- `profiles/byte_v2_gqa4_fa2_like_mainloop_guarded_4096_p64_20260626.jsonl`

Interpretation:

The redesign is useful as a structure proof. It recovers most of the regression
introduced by QK-MMA plus shared-score fanout:

```text
0.179200 ms -> 0.145408 ms
```

That means the FA2-style accumulator-to-P-to-PV flow is directionally correct.
However, it is still slower than the earlier FA2-like CUTE PV/register-P
variant and much slower than the default GQA4 packed kernel.

The main reason is work distribution: the new path makes warp0 own the complete
QK/softmax/PV pipeline for all 128 output dimensions, while the older path uses
four warps to split PV output tiles. It removes P broadcast overhead, but loses
PV parallelism. A production-quality FA2-like ByteV2 decode kernel needs a
warp-group/CTA design that preserves FA2's register-P dataflow while also
parallelizing the 128 output dimensions.

Updated next structural target:

1. Keep the single-warp mainloop as the correctness/layout baseline.
2. Design a multi-warp FA2 mainloop where each warp owns a slice of output
   dimensions but receives P without shared-score materialization, likely via a
   deliberate warp-group broadcast/register reconstruction scheme.
3. Profile the single-warp mainloop with NCU before changing it again, because
   it gives a clean reference for the cost of full-head V staging plus
   one-warp PV.

## 2026-06-26 Multi-Warp Register-P Mainloop Attempt

Status: implemented as a non-default experiment and measured as negative.

What changed:

- Added `tile_policy[12]` as a multi-warp FA2-style mainloop flag.
- Added `--gqa-fa2-multiwarp` to the microbench script. It requires
  `--gqa-fa2-qk-mma` and is mutually exclusive with `--gqa-fa2-mainloop`.
- Added a new `UseFa2Multiwarp` template path in the isolated FA2-like GQA4
  kernel.
- Extended the FA2-like CUDA correctness test to cover four modes:
    - scalar QK + CUTE PV/register-P;
    - QK-MMA shared-score path;
    - QK-MMA single-warp mainloop;
    - QK-MMA multi-warp register-P mainloop.

Experiment design:

```text
all 128 CTA threads:
  decode/stage Q -> CUTE shared Q
  decode/stage K -> CUTE shared K

each warp:
  recompute QK CUTE MMA -> acc_s
  softmax directly on local acc_s
  acc_s -> BF16 register P
  decode/stage its 32 V output dims as two 16-dim tiles
  PV gemm_rs for its output-dim slice
```

This preserves the FA2-like register-P path for each PV warp and restores
multi-warp output-dimension parallelism. The tradeoff is that QK/softmax are
recomputed by all four warps.

Verification:

```bash
cmake --build build-bytev2-stable --target _C_stable_libtorch -j 32
cp build-bytev2-stable/_C_stable_libtorch.abi3.so vllm/_C_stable_libtorch.abi3.so
.venv/bin/python -m pytest tests/v1/attention/test_byte_v2_layout.py -v
.venv/bin/pre-commit run ruff-check --files \
  tests/v1/attention/test_byte_v2_layout.py \
  scripts/byte_v2_decode_microbench.py
```

Result:

- Build passed under `sm_86`.
- The only CUDA build warning is the pre-existing unused
  `byte_v2_reshape_and_cache_kernel`.
- Test result: `115 passed, 1 skipped, 16 warnings`.
- Ruff passed.

4096 p64/bn64 measurements:

| route | median | max diff |
| --- | ---: | ---: |
| FA2-like CUTE PV + register P | 0.133120 ms | 0.0 |
| FA2-like + QK-MMA shared-score path | 0.179200 ms | 0.0 |
| FA2-like + QK-MMA + single-warp mainloop | 0.145408 ms | 0.0 |
| FA2-like + QK-MMA + multi-warp register-P | 0.186368 ms | 0.0 |
| guarded FA2-like + QK-MMA + multi-warp register-P | 0.185344 ms | 0.0 |

Artifacts:

- `profiles/byte_v2_gqa4_fa2_like_multiwarp_4096_p64_20260626.jsonl`
- `profiles/byte_v2_gqa4_fa2_like_multiwarp_guarded_4096_p64_20260626.jsonl`

Conclusion:

The multi-warp version is correct but not useful as a performance path. It
restores PV output-dimension parallelism, but the cost of recomputing QK and
softmax in all four warps is larger than the saved single-warp PV bottleneck.

Do not continue with QK-recompute multi-warp variants. The next multi-warp
design must avoid both extremes:

- single-warp mainloop: no P fanout, but insufficient PV parallelism;
- QK-recompute multi-warp: PV parallelism, but repeated QK/softmax.

The next viable design should compute QK/P once and feed multiple PV warps
without the old shared-score/shared-probability path, for example by a compact
BF16 P tile broadcast or a warp-group specific register reconstruction scheme.
