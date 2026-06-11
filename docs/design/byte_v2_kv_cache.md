# Byte-v2 Compressed KV Cache Integration

This document describes how to integrate the Byte-v2 compressed KV cache
prototype into vLLM. The goal is to keep KV cache resident in HBM in a
compressed format, read compressed KV pages during decode, decode each tile
into registers or shared memory, and immediately use the decoded values in
attention. The expected benefit is lower KV cache memory usage and lower KV
cache read bandwidth while keeping decode performance close to the raw vLLM
path.

The prototype referenced by this document lives under:

```text
/mnt/sda1/yxz/byte_v2/vllm_integration/byte_v2_attention_kernel
```

The prototype is useful for the Byte-v2 tile codec and fused decode +
attention dataflow, but it is not a drop-in replacement for vLLM attention.
It assumes contiguous `[G, N, D]` tensors, homogeneous query rows, and
benchmark-oriented payload tensors. vLLM decode uses paged KV cache,
`block_table`, variable sequence lengths, prefill/decode mixing, and several
runtime metadata paths. The integration therefore needs a new vLLM attention
backend and a vLLM-native paged Byte-v2 decode kernel.

## Goals

- Add a new `kv_cache_dtype=byte_v2` mode.
- Store most KV cache blocks in compressed Byte-v2 format in HBM.
- Decode compressed K/V tiles directly inside the decode attention kernel.
- Avoid full-cache dequantization or full-cache gather before attention.
- Preserve the vLLM paged KV cache lifecycle, scheduler, and block table model.
- Support GQA decode efficiently, starting with `q_per_kv` values used by
  common models.
- Make actual allocated KV memory smaller than raw BF16 cache, not just
  logically compressed in benchmarks.

## Non-goals For The First Version

- Full feature parity with every vLLM attention backend.
- FP16-lossless Byte-v2 compression.
- Sliding window, attention sinks, ALiBi, encoder-decoder attention, or MLA.
- CPU/offload KV cache transfer support.
- Changing vLLM scheduler semantics.
- Replacing prefill FlashAttention in the first milestone.

## Prototype Summary

The prototype uses 16-by-16 BF16 tiles. For a fast-path tile, the payload is:

```text
base:          1 byte
low_bytes:   256 bytes
code_packed: 128 bytes
fallback:      1 byte
```

The decode formula reconstructs the BF16 high byte from the tile base and the
4-bit code, then combines it with the stored low byte:

```text
low_exp_lsb = low >> 7
delta_hi = code & 0x07
sign = code >> 3
exp_hi = (base >> 1) + delta_hi + ((base & 1) & (low_exp_lsb ^ 1))
high = (sign << 7) | exp_hi
bf16 = (high << 8) | low
```

The recommended prototype attention paths are:

- `byte_v2_attention_grouped_tiles` for standard grouped fused decode.
- `byte_v2_attention_grouped_gqa_tiles` for GQA reuse.

The dataflow is:

```text
compressed K/V payload
  -> decode 16x16 K tile into shared memory
  -> BF16 WMMA QK
  -> online softmax
  -> decode 16x16 V tile into shared memory
  -> BF16 WMMA PV
  -> segment partial output
  -> segment reduction
```

Important prototype constraints:

- `N` and `D` are multiples of 16.
- `head_dim <= 128`.
- Current codec is BF16-oriented.
- Current benchmark payload includes dense `fallback_raw` tensors for
  convenience. This must not be copied directly into vLLM, because dense
  fallback storage can make the allocated cache larger than raw BF16 cache.
- Current attention kernels assume contiguous K/V and do not understand vLLM
  paged cache addressing.

## Required vLLM Changes

### Cache dtype and CLI/config plumbing

Update the cache dtype configuration to accept Byte-v2:

- `vllm/config/cache.py`
  - Add `byte_v2` to `CacheDType`.
  - Validate that Byte-v2 is explicitly requested.
  - Log that it is an experimental compressed KV cache dtype.
- `vllm/utils/torch_utils.py`
  - Map `byte_v2` to `torch.uint8` for storage.
  - Include `byte_v2` in quantized/compressed KV cache detection where needed.

Byte-v2 should not be selected implicitly from `kv_cache_dtype=auto`.
The first version should require an explicit config value.

### Attention backend registration

Add a new backend enum and CUDA backend selection entry:

- `vllm/v1/attention/backends/registry.py`
  - Add `BYTE_V2`.
  - Register `ByteV2AttentionBackend`.
- `vllm/platforms/cuda.py`
  - Make `BYTE_V2` available only when `kv_cache_dtype == "byte_v2"` or the
    user explicitly selects it.
  - Do not place it ahead of existing backends for normal raw KV cache.

The backend should fail validation for unsupported configurations rather than
silently falling back to a raw backend.

### KV cache spec and memory accounting

Add a Byte-v2-specific cache spec:

- `vllm/v1/kv_cache_interface.py`
  - Add `ByteV2FullAttentionSpec`, similar in purpose to
    `TQFullAttentionSpec`.
  - Override page size accounting to use the actual Byte-v2 page layout.
- `vllm/model_executor/layers/attention/attention.py`
  - In `Attention.get_kv_cache_spec()`, return `ByteV2FullAttentionSpec` when
    `kv_cache_dtype == "byte_v2"`.

The spec must account for:

- Fixed compressed payload bytes per full block.
- Block status metadata.
- Raw tail storage, if stored inside each page.
- Fallback metadata.
- Sparse fallback pool reservation, if the pool is part of the layer cache
  allocation.

The key requirement is that vLLM's block budget must be based on actual
allocated bytes, not benchmark-only logical compressed bytes.

### KV cache allocation and reshape

The vLLM GPU worker currently allocates raw cache pages and reshapes them using
the backend-provided cache shape:

- `vllm/v1/worker/gpu/attn_utils.py`
- `vllm/v1/worker/gpu_model_runner.py`

For Byte-v2, `get_kv_cache_shape()` should return a shape backed by a single
`uint8` tensor per layer, for example:

```text
[num_blocks, page_bytes]
```

or a structured but still contiguous view such as:

```text
[num_blocks, num_kv_heads, num_d_tiles, payload_bytes]
```

A single raw `uint8` tensor is preferred because vLLM's KV cache allocator,
sleep/wake path, and block management already assume one storage object per
cache group. The Byte-v2 backend can derive typed views and offsets internally.

### Cache update path

Add a new GPU op for compressed cache writes:

- `csrc/libtorch_stable/cache_kernels.cu`
- `csrc/libtorch_stable/torch_bindings.cpp`
- `vllm/_custom_ops.py`

The operation should have the same logical role as `reshape_and_cache_flash`,
but it writes Byte-v2 pages:

```text
byte_v2_reshape_and_cache(
    key,
    value,
    kv_cache_uint8,
    slot_mapping,
    block_status,
    fallback_pool,
    ...
)
```

Recommended block lifecycle:

1. Store new rows for the currently open physical block in raw tail storage.
2. When a block becomes full, compress all 16 rows for every KV head and
   `head_dim` tile.
3. Store fast-path Byte-v2 payloads in the compressed page area.
4. Store fallback raw tiles in a sparse fallback pool or mark the whole block
   as raw fallback.
5. Mark the physical block as finalized/compressed.

This avoids recompressing a 16-row tile after every single decode token.

### Byte-v2 attention backend

Create:

```text
vllm/v1/attention/backends/byte_v2_attn.py
```

Use `turboquant_attn.py` as the closest structural reference and
`flash_attn.py` as the raw vLLM CUDA attention reference.

The backend should define:

- `ByteV2AttentionBackend`
- `ByteV2Metadata`
- `ByteV2MetadataBuilder`
- `ByteV2AttentionImpl`

Validation should initially require:

- CUDA platform.
- Decoder attention.
- `kv_cache_dtype == "byte_v2"`.
- `block_size == 16`.
- `head_dim % 16 == 0`.
- `head_dim <= 128`.
- BF16 K/V cache.
- No sliding window, attention sinks, ALiBi, or MLA.

The first version should keep:

```text
forward_includes_kv_cache_update = False
```

Then `Attention.forward()` can continue to call the backend cache update before
calling the backend attention forward path.

### Metadata builder

The metadata builder must provide the Byte-v2 kernel with:

- Number of decode tokens.
- Request-to-token mapping for decode rows.
- `seq_lens`.
- `block_table`.
- Query start locations.
- Slot mappings for cache update.
- CUDA graph compatible static buffers.
- Optional flags for raw tail or raw fallback blocks.

As in TurboQuant, the backend should prefer decode-first ordering when mixed
prefill/decode batches are present. This keeps the decode kernel path simple
and allows prefill to use existing FlashAttention behavior.

## Byte-v2 Page Layout

The first implementation should use `block_size=16`, matching the Byte-v2
token tile height.

Recommended logical page layout:

```text
physical block page:
  header:
    status
    valid_rows
    fallback counters or pool offsets

  K fast payload:
    [kv_head][d_tile]:
      base
      fallback metadata
      low_bytes
      code_packed

  V fast payload:
    [kv_head][d_tile]:
      base
      fallback metadata
      low_bytes
      code_packed

  optional raw tail storage:
    [K/V][kv_head][valid_tail_rows][head_dim]
```

Do not allocate dense `fallback_raw` for every tile. Use one of:

1. Sparse fallback pool per layer.
2. Sparse fallback pool per cache group.
3. Block-level raw fallback.

Sparse fallback gives the best memory ratio but requires capacity management.
Block-level raw fallback is simpler and safer for an initial correctness
implementation.

## Decode Kernel Design

The prototype kernel's tile decode helpers can be reused conceptually, but the
kernel scheduler and addressing must be rewritten for vLLM.

Required inputs:

```text
q:              [num_decode_tokens, num_heads, head_dim]
kv_cache:       uint8 compressed pages
block_table:    [num_reqs, max_blocks_per_req]
seq_lens:       [num_reqs]
out:            [num_decode_tokens, num_heads, head_dim]
scale:          attention scale
```

For each logical K/V tile:

```text
logical_block = n_tile
physical_block = block_table[request_id, logical_block]
page = kv_cache[physical_block]
tile = page[k_or_v, kv_head, d_tile]
```

If the block is finalized, the kernel decodes compressed payload into shared
memory or registers. If the block is a raw tail or raw fallback block, the
kernel loads BF16 values directly.

The prototype assumes `M=16` query rows that share the same K/V sequence. vLLM
decode often has many requests with one query row each, and those rows have
different block tables. Therefore the production kernel cannot simply batch
16 arbitrary requests into the prototype's `M` dimension.

Recommended kernel milestones:

1. Correct paged decode kernel:
   - One CTA handles one request and one KV head or GQA group.
   - Supports compressed full blocks, raw tail blocks, and fallback blocks.
   - May underutilize tensor cores for `q_per_kv=1`, but validates the whole
     runtime path.
2. GQA reuse kernel:
   - Specialize for common `q_per_kv` values such as 2 and 4.
   - Decode K/V once and reuse for multiple Q heads.
   - Use shared-memory tile staging and BF16 WMMA for QK and PV on SM80+,
     with scalar/shared fallback for unsupported devices or shapes.
3. Split-K or segmented decode:
   - Parallelize long context over block ranges.
   - Reduce partial softmax/output segments.
   - Match the prototype's online softmax reduction idea while preserving
     vLLM block-table addressing.
4. Shape specializations:
   - `head_dim=128`
   - `block_size=16`
   - `q_per_kv=4`
   - BF16 query/KV

## Prefill and Mixed Batches

The first implementation should not try to replace prefill attention.

Recommended behavior:

- First-chunk prefill:
  - Compute attention with raw FlashAttention.
  - Write K/V into Byte-v2 cache.
  - Compress all full blocks after the cache update.
- Decode:
  - Use Byte-v2 compressed paged decode.
- Mixed prefill/decode:
  - Run the decode slice with Byte-v2.
  - Run the prefill slice with FlashAttention.
- Continuation prefill:
  - Initially use a conservative fallback path.
  - Later optimize with a dedicated compressed prefill or chunked decode path.

## Build and Op Registration

The kernel code should be integrated into vLLM's native extension build rather
than loaded as a standalone benchmark extension.

Likely files:

- `csrc/libtorch_stable/cache_kernels.cu`
  - Add compressed cache update/finalization kernels, or split them into a new
    Byte-v2-specific source file if the build system supports it cleanly.
- `csrc/libtorch_stable/torch_bindings.cpp`
  - Register Byte-v2 cache update and attention ops.
- `vllm/_custom_ops.py`
  - Add Python wrappers.
- `vllm/v1/attention/backends/byte_v2_attn.py`
  - Call the wrappers from `do_kv_cache_update()` and `forward()`.

The standalone prototype's benchmark-only CPU compression should not be used
in the runtime path.

## Testing Plan

Codec and cache update tests:

- Round-trip one 16-by-16 BF16 tile.
- Round-trip many heads and `head_dim` tiles.
- Forced fallback tile.
- Fallback pool overflow or block-level raw fallback.
- Random `slot_mapping`.
- Partially filled tail block.
- Full block finalization.

Attention correctness tests:

- Compare Byte-v2 decode output against raw FlashAttention/PagedAttention.
- Random `block_table`.
- Random `seq_lens`.
- Tail lengths from 1 to 15.
- `q_per_kv=1`, `q_per_kv=2`, `q_per_kv=4`.
- `head_dim=64` and `head_dim=128`.
- Prefix-cache reused full blocks.
- Mixed prefill/decode batches.

Memory accounting tests:

- Verify `ByteV2FullAttentionSpec.page_size_bytes`.
- Verify the number of allocated blocks increases relative to raw BF16 when
  compression is enabled.
- Verify actual allocated bytes include fallback storage.

Performance tests:

- End-to-end vLLM decode latency versus raw BF16 vLLM.
- Tokens per second at long context lengths.
- HBM read bandwidth using Nsight Compute.
- Compression ratio based on actual allocation, not logical payload only.
- Fallback rate across representative model layers.

Use vLLM's environment rules when running tests, for example:

```bash
.venv/bin/python -m pytest tests/path/to/test_file.py -v
```

## Implementation Phases

### Phase 1: Correctness backend

- Add `kv_cache_dtype=byte_v2`.
- Add `ByteV2FullAttentionSpec`.
- Add Byte-v2 backend registration.
- Add compressed cache update op.
- Add conservative decode fallback for correctness, such as temporary
  dequant/gather for small tests.
- Validate vLLM scheduling, block allocation, and cache lifecycle.

### Phase 2: Direct compressed paged decode

- Add the first vLLM-native Byte-v2 paged decode kernel.
- Read compressed pages directly through `block_table`.
- Decode K/V tiles inside the attention kernel.
- Support raw tail blocks.
- Support fallback blocks or sparse fallback pool.
- Compare against raw vLLM outputs and measure actual KV memory reduction.

### Phase 3: Performance path

- Completed: add GQA reuse shared-memory/WMMA fused decode for SM80+
  `q_per_kv=2..8`, with scalar/shared fallback.
- Add split-K or segmented long-context decode.
- Specialize common shapes.
- Tune shared memory usage and segment sizes.
- Ensure CUDA graph compatibility.
- Optimize fallback metadata layout and avoid extra global reads.

### Phase 4: Feature expansion

- FP16 support if needed.
- Sliding window.
- Prefix cache stress tests and cache reuse.
- KV transfer/offload compatibility.
- Speculative decode compatibility.
- Sleep/wake support.

## Main Risks

### Dense fallback storage can remove the memory benefit

The prototype's dense `fallback_raw` tensor is acceptable for benchmarking but
not for a memory-saving vLLM integration. vLLM must use sparse fallback storage
or block-level fallback accounting.

### The prototype query layout does not match vLLM decode

The prototype's `M=16` query tile assumes rows share the same K/V sequence.
vLLM decode usually has many requests with one query each and different
`block_table` entries. The production decode kernel must be paged and
request-aware.

### Partial blocks need a clear policy

Byte-v2 compresses 16-token tiles. Decode appends one token at a time. The
recommended policy is raw tail storage until the block is full, followed by
block finalization.

### End-to-end performance may differ from prototype benchmarks

The prototype measures contiguous tensors. vLLM serving includes scheduler
metadata, page-table indirection, CUDA graph constraints, prefill/decode
mixing, and fallback management. Benchmark both kernel-level and end-to-end
latency.

## Suggested First PR Boundary

The first upstreamable unit should be narrow:

- Add config plumbing for `byte_v2`.
- Add the cache spec.
- Add backend registration and validation.
- Add tests for config/spec behavior.

Do not include a trivial or cosmetic-only PR. A human submitter must review
and understand all changed lines, run relevant tests, and disclose AI
assistance in the PR description.
