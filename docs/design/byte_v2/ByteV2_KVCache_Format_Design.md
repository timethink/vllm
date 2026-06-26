# ByteV2 KV Cache Format Design

## 1. Scope

This document describes the ByteV2 compressed KV cache storage format used by
the current decode kernel.

The format target is:

```text
fixed-length 12-bit main payload for every K/V element
+ sparse outlier sideband for exact high-byte repair
+ fast no-outlier decode path that does not touch outlier metadata
```

The important rule is that outlier elements are still present in the fixed
12-bit main payload. The outlier sideband is an overlay, not a replacement for
the main payload.

## 2. Default Parameters

Current default policy:

| field | value |
| --- | ---: |
| codec token block | 16 |
| codec dim block | 16 |
| codec tile elems | 256 |
| alloc block tokens | 16 |
| compute block N | 64 |
| head dim K/V | 128 |
| KV heads | 8 |
| low bytes / elem | 1 |
| exponent code bits / elem | 4 |
| outlier value bits | 8 |
| outlier entries / tile | 256 |

Current default page sizing:

| field | value |
| --- | ---: |
| codec payload / tile | 384 B |
| code bytes / tile | 128 B |
| page size | 115328 B |

## 3. Codec Tile Payload

Each codec tile covers:

```text
16 tokens x 16 dims = 256 bf16 elements
```

The main payload stores every element in a fixed 12-bit form:

```text
low byte:      8 bits
exponent code: 4 bits
total:        12 bits / element
```

Physical layout is split-plane:

```text
payload tile:
  low[256]          // 1 byte per element
  code_packed[128]  // 2 x 4-bit codes per byte
```

Element mapping:

```text
elem_idx = row_in_tile * CodecDimBlock + dim_in_tile

low  = payload[tile_offset + elem_idx]
pack = payload[tile_offset + CodecTileElems + elem_idx / 2]
code = (elem_idx & 1) ? (pack >> 4) : (pack & 0x0f)
```

This split-plane layout is retained because profiling showed that alternative
placements are worse:

- 8-bit expanded code reduced some instructions, but increased payload bytes
  and regressed 16k/p64 decode by about 11.4%.
- 12-bit pair-interleaved `[low0, low1, code_pair]` kept the same tile size,
  but increased global load sectors by about 43.6% and regressed by about
  14.7%.

## 4. Base Metadata

Each codec tile has an 8-bit base for K and V:

```text
normal high byte = base + code
bf16 bits        = (high << 8) | low
```

The cache writer chooses a 16-wide high-byte window for each tile:

```text
base <= high <= base + 15
```

Elements inside the selected window are represented exactly by:

```text
code = high - base
```

Elements outside the selected window are still written to the 12-bit payload,
but their 4-bit code is not sufficient to recover the true high byte. Those
elements are repaired by outlier sideband entries.

## 5. Outlier Sideband

Outlier sideband is sparse logical metadata, but its storage area is fixed-size
per tile in the page layout.

Per tile metadata:

```text
outlier mask bit: whether this tile has any outlier overlay
outlier count:    number of valid outlier entries for this tile
outlier entries:  elem_idx + true high byte
```

Current outlier entry layout:

```text
elem index bits: 8   // enough for 0..255
value bits:      8   // true bf16 high byte
entry size:      2 B
entries / tile:  256
```

Logical repair:

```text
low  = main payload low byte
high = base + code

if tile has outlier metadata and elem_idx is present:
    high = outlier_true_high

bf16 = (high << 8) | low
```

The low byte is always read from the fixed main payload, including outliers.
The outlier sideband only overrides the high byte.

## 6. Decode Paths

### 6.1 Fast Safe Path

For pages known to have no fallback and no outliers, the decode kernel uses the
no-fallback/no-outlier path:

```text
read base
read low
read packed 4-bit code
high = base + code
return bf16(high, low)
```

This path must not read outlier count or outlier entries. It is the performance
critical path for current GQA decode benchmarks.

### 6.2 Generic Path

For unsafe pages or generic decode:

```text
check fallback mask
read base
read low
read packed 4-bit code
high = base + code
check outlier mask
if tile has outliers:
    read outlier count / entries
    repair high when elem_idx matches
return bf16(high, low)
```

The implementation should avoid touching outlier entries unless the tile-level
outlier mask is set.

## 7. Writer Requirements

The cache writer must preserve these invariants:

1. Every element is written to the fixed 12-bit main payload.
2. The low byte of every element, including outliers, is stored exactly.
3. For in-window elements, `code = high - base`.
4. For out-of-window elements, an outlier entry stores the true high byte.
5. Outlier metadata is tile-local and each stored entry carries its logical
   `elem_idx`.
6. The page layout remains fixed-size; outlier count changes valid entries, not
   page offsets.

With `OutlierEntriesPerTile = 256`, every element in a 16x16 tile can be
represented by the outlier overlay if needed. The fallback mask remains part of
the layout, but the target compressed format should avoid relying on raw
fallback in the hot decode path.

Current implementation note:

- Full-block cache writer stores only true out-of-window elements in the
  outlier sideband.
- Raw-staging commit uses the same sparse overlay format as the direct writer.
- Single-token fused cache update appends only out-of-window elements for the
  updated row.
- The old dense-prefix overlay behavior is not used by writers. Generic decode
  can still scan entries by `elem_idx`, so the sparse sideband remains exact.

## 8. Why Outliers Also Keep 12-bit Payload

Outliers should not be removed from the main payload.

Keeping fixed 12-bit payload for all elements gives:

- direct address arithmetic for every element;
- no variable-length tile payload;
- no prefix sum or per-tile remapping during decode;
- identical access pattern for normal and outlier elements until the optional
  high-byte repair;
- a fast path that can skip all outlier sideband reads.

If outliers were removed from the payload, decode would need an extra mapping
from logical `elem_idx` to compressed payload position. That would add control
flow and memory dependencies in the hottest path, which is worse than storing
the 12-bit placeholder for outlier elements.

## 9. Current Performance Guidance

The current data says the next optimization should not be a simple payload
format reshuffle.

Known negative results:

| experiment | result |
| --- | --- |
| 8-bit code payload | slower by about 11.4%; DRAM read +34.5% |
| 12-bit pair-interleaved payload | slower by about 14.7%; global load sectors +43.6% |

Therefore the default format should remain:

```text
fixed 12-bit split-plane payload
+ sparse high-byte outlier overlay
```

The next performance work should instead target:

- reducing repeated scalar payload decodes;
- reusing decoded V across the GQA q-group;
- changing how PV consumes probabilities and V values;
- avoiding unnecessary outlier/fallback metadata reads in safe pages.

## 10. Compatibility Checklist

Any future format or kernel change should answer these questions before being
kept:

1. Does every element still have a fixed address in the main payload?
2. Can the no-outlier fast path decode without reading outlier metadata?
3. Does the format keep the hot-path payload at 12 bits / element?
4. Does it avoid 3-byte strided access patterns that increase global sectors?
5. Does it keep page offsets compile-time/layout-policy computable?
6. Does it improve 16k/p64 GQA decode by more than noise, ideally at least 5%?

If the answer to any of the first five questions is no, the change needs a
strong measured speedup to justify the added complexity.

## 11. Verification

Sparse overlay implementation was validated with:

```text
CUDA_VISIBLE_DEVICES=5 .venv/bin/python -m pytest \
  tests/v1/attention/test_byte_v2_layout.py \
  -k "reshape_and_cache_cuda_writes_outlier_overlay or \
      reshape_and_cache_cuda_full_tile_overlay_avoids_raw_fallback or \
      raw_staging_commit_matches_direct_cache or \
      raw_staging_incremental_partial_block_matches_direct_cache or \
      single_token_cache_update_cuda_matches_direct_cache" -q
```

Result:

```text
8 passed, 104 deselected
```

Decode correctness:

```text
CUDA_VISIBLE_DEVICES=5 .venv/bin/python -m pytest \
  tests/v1/attention/test_byte_v2_layout.py \
  -k gqa_packed_cuda_matches_raw_reference -q
```

Result:

```text
2 passed, 110 deselected
```

No-outlier fast-path microbench artifact:

```text
profiles/byte_v2_vs_raw_gqa_sparse_overlay_format_16384_p64_gpu5_20260624.jsonl
```

Result:

| kernel | median |
| --- | ---: |
| ByteV2 GQA4 p64 no-outlier | 0.2417 ms |
| raw FA2 | 0.1362 ms |

This confirms the sparse overlay writer change does not regress the current
safe-page decode fast path.
