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

Current V5 default page sizing:

| field | value |
| --- | ---: |
| codec payload / tile | 384 B |
| code bytes / tile | 128 B |
| page header | 128 B |
| metadata / KV head | 96 B |
| metadata / page | 896 B |
| fixed 12-bit K/V payload | 49152 B |
| shared outlier pool | 2048 B (1024 entries) |
| V5 page size | 52096 B |
| raw BF16 page size | 65536 B |
| physical bytes saved | 20.5% |

The legacy V4 layout reserved 256 outlier entries for every codec tile and
therefore occupied 115328 B per page. V5 keeps the same fixed payload and
logical repair format, but pools sparse outlier storage across the page.

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

V5 stores outliers in one fixed-size pool shared by all K/V tiles and KV heads
in a cache page. It does not reserve a worst-case payload for every tile.

Per tile metadata:

```text
outlier mask bit: whether this tile has any outlier overlay
outlier count:    uint16 number of valid entries
pool offset:      uint16 index of the tile's first pool entry
outlier entries:  elem_idx + true high byte
```

Current outlier entry layout:

```text
elem index bits: 8   // enough for 0..255
value bits:      8   // true bf16 high byte
entry size:      2 B
pool entries:    1024 / page
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

The writer allocates a power-of-two segment for each nonempty tile. The
single-token updater compacts the page pool before appending new entries, then
relocates a segment only when it must grow. Pool exhaustion sets the page
overflow flag and traps the writer. It never silently drops an outlier.

V5 is therefore exact for every page that fits in the 1024-entry pool and
fail-closed otherwise. It is not a claim that arbitrary pathological pages
with more than 1024 pooled entries can be represented by the compact format.

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
5. Each tile descriptor stores its count and page-pool offset, and each entry
   carries its logical `elem_idx`.
6. Pool allocation failure must trap; it must not truncate the overlay.
7. The page layout remains fixed-size; outlier count changes pool use, not page
   size.

The per-tile logical maximum remains 256 entries, so one pathological tile is
representable. The page-wide total is bounded by the 1024-entry V5 pool. The
fallback mask remains part of the metadata, but V5 does not allocate a raw
page-local fallback payload.

Current implementation note:

- Full-block cache writer stores only true out-of-window elements in the
  shared outlier pool.
- Raw-staging commit uses the same sparse overlay format as the direct writer.
- Single-token fused cache update compacts the pool, then appends only
  out-of-window elements for the updated row.
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
| in-band zero escape | forced-outlier faster, but safe path slower by 5.8-17.3%; rejected |
| zero-only bitmap overlay | synthetic forced-outlier -6.0%, but real-cache replay +4.5%; rejected |

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

The V5 implementation was validated on an NVIDIA A40 with:

```text
CUDA_VISIBLE_DEVICES=6 .venv/bin/python -m pytest \
  tests/v1/attention/test_byte_v2_layout.py -q \
  --disable-warnings --maxfail=1
```

Result:

```text
170 passed, 1 skipped
```

The suite covers V5 layout arithmetic, direct and raw-staging writers,
single-token updates, generic/direct/speculative decode, and semantic cache
equality after nondeterministic parallel pool allocation.

Real layer-0 cache data at 4097 tokens contained 257 active pages:

| page outlier entries | value |
| --- | ---: |
| mean | 4.78 |
| p95 | 7 |
| p99 | 9 |
| maximum | 28 |

The observed maximum uses 2.7% of the 1024-entry pool. Full-model E2E also
completed without the fail-closed overflow trap.

Capacity and E2E were measured with Llama-3.1-8B, context 4096, batch 1, static
query-size compilation, and 100% speculative acceptance:

| metric | ByteV2 V5 | raw FA2 | difference |
| --- | ---: | ---: | ---: |
| allocated KV tokens | 204324 | 163118 | +25.26% |
| Q2 throughput | 59.643 tok/s | 57.618 tok/s | +3.51% |
| Q4 throughput | 109.008 tok/s | 106.986 tok/s | +1.89% |
| Q8 throughput | 189.755 tok/s | 187.194 tok/s | +1.37% |
| Q16 throughput | 314.876 tok/s | 306.845 tok/s | +2.62% |

ByteV2 and raw FA2 produced identical output token IDs at every Q width. V5
Q16 throughput differs from the previous V4 measurement by -0.06%, while its
physical page shrinks from 115328 B to 52096 B.

Detailed artifacts are recorded in:

```text
profile/byte-v2-v5-compact-a40-20260717/REPORT.md
```
