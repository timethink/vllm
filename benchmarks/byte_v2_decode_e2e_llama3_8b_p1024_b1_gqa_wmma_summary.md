# Byte-v2 Llama-3-8B Decode E2E Benchmark After GQA WMMA

Model:

```text
/mnt/sda1/yxz/byte_v2/Meta-Llama-3-8B-Instruct
```

Benchmark settings:

- GPU: `CUDA_VISIBLE_DEVICES=2`, NVIDIA A40
- dtype: `bfloat16`
- prompt length: `1024`
- batch size: `1`
- decode lengths: `16, 64, 128, 256`
- timed runs per point: `1`
- warmup decode length: `8`
- `gpu_memory_utilization=0.80`
- `block_size=16`
- `max_model_len=1296`
- `max_num_batched_tokens=1280`
- `enforce_eager=True`
- `VLLM_BYTE_V2_SPARSE_FALLBACK_POOL_RATIO=0.10` for the compressed-only
  capacity row below. The default was later reduced to `0.03`.

Raw result files:

- `benchmarks/byte_v2_decode_e2e_llama3_8b_p1024_b1_gqa_wmma.json`
- `benchmarks/byte_v2_decode_e2e_llama3_8b_p1024_b1_gqa_wmma.log`

## KV Capacity

| Mode | GPU KV cache size | Relative to raw |
|---|---:|---:|
| raw vLLM | 166,768 tokens | 100.0% |
| ByteV2 raw-overlay | 166,736 tokens | 100.0% |
| ByteV2 compressed-only + sparse fallback pool | 195,216 tokens | 117.1% |

The compressed-only row above was measured with the older 10% sparse fallback
pool. With the current 3% default, the same Llama-3-8B geometry is expected to
allocate about 51,394 bytes per block per layer, which gives roughly 212,656 KV
tokens at the same memory budget, or about 127.5% of raw vLLM. This estimate
assumes the actual incompressible-block rate stays below the 3% pool capacity.

## Decode E2E Throughput

Output throughput is generated tokens divided by full `llm.generate()` wall
time. The timing includes prompt processing/cache update, so short decode
lengths are not pure decode-kernel measurements.

| Mode | Decode len | Elapsed (s) | Output tok/s | Total tok/s | Output tok/s vs raw | Slowdown vs raw |
|---|---:|---:|---:|---:|---:|---:|
| raw vLLM | 16 | 0.465 | 34.40 | 2235.91 | 100.0% | 1.00x |
| raw vLLM | 64 | 1.838 | 34.83 | 592.05 | 100.0% | 1.00x |
| raw vLLM | 128 | 3.669 | 34.89 | 314.00 | 100.0% | 1.00x |
| raw vLLM | 256 | 7.336 | 34.90 | 174.48 | 100.0% | 1.00x |
| ByteV2 raw-overlay | 16 | 12.457 | 1.28 | 83.48 | 3.7% | 26.78x |
| ByteV2 raw-overlay | 64 | 17.588 | 3.64 | 61.86 | 10.5% | 9.57x |
| ByteV2 raw-overlay | 128 | 24.578 | 5.21 | 46.87 | 14.9% | 6.70x |
| ByteV2 raw-overlay | 256 | 39.069 | 6.55 | 32.76 | 18.8% | 5.33x |
| ByteV2 compressed-only | 16 | 13.717 | 1.17 | 75.82 | 3.4% | 29.49x |
| ByteV2 compressed-only | 64 | 22.066 | 2.90 | 49.31 | 8.3% | 12.01x |
| ByteV2 compressed-only | 128 | 33.429 | 3.83 | 34.46 | 11.0% | 9.11x |
| ByteV2 compressed-only | 256 | 56.655 | 4.52 | 22.59 | 13.0% | 7.72x |

## Comparison With Pre-WMMA Baseline

The previous Llama-3-8B report used the same prompt length, decode lengths,
batch size, and eager mode, before the GQA shared-memory/WMMA kernel was
connected.

| Mode | Decode len | Pre-WMMA output tok/s | GQA WMMA output tok/s | Change |
|---|---:|---:|---:|---:|
| ByteV2 raw-overlay | 256 | 7.803 | 6.552 | -16.0% |
| ByteV2 compressed-only | 256 | 5.056 | 4.519 | -10.6% |

## Notes

- The measured allocator-visible capacity row uses the old 10% sparse fallback
  pool and improves KV token capacity by about 17.1% on this setup. The current
  3% default was confirmed in
  `benchmarks/byte_v2_decode_e2e_llama3_8b_p1024_b1_pool003_summary.md`: it
  reaches 212,656 KV tokens, about 127.5% of raw vLLM, without exhausting the
  sparse fallback pool on the tested workload.
- The current GQA WMMA e2e path is slower than the previous conservative
  baseline. This points to kernel-level overhead rather than memory capacity:
  the kernel still uses one CTA per request/KV head, has no split-K segmented
  parallelism, uses many CTA-wide synchronizations, and computes tile softmax
  from one thread before PV WMMA.
- The next useful measurement is a kernel/pipeline breakdown: cache update
  time, prefill time, decode attention time, and per-token kernel latency with
  and without the GQA WMMA dispatch.
