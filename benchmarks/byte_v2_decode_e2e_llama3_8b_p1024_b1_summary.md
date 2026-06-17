# Byte-v2 Llama-3-8B Decode E2E Benchmark

Model:

```text
/mnt/sda1/yxz/byte_v2/Meta-Llama-3-8B-Instruct
```

Benchmark settings:

- GPU: `CUDA_VISIBLE_DEVICES=2`
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

Raw result JSON:

- `benchmarks/byte_v2_decode_e2e_llama3_8b_p1024_b1.json`
- `benchmarks/byte_v2_decode_e2e_llama3_8b_p1024_b1_compressed_only.json`

## KV Capacity

| Mode | GPU KV cache size | Relative to raw |
|---|---:|---:|
| raw vLLM | 166,768 tokens | 100.0% |
| ByteV2 raw-overlay | 166,736 tokens | 100.0% |
| ByteV2 compressed-only + sparse fallback pool | 195,216 tokens | 117.1% |

The compressed-only row above was measured with the older 10% sparse fallback
pool. With the current 3% default, the same Llama-3-8B geometry is expected to
reach about 212,656 KV tokens at the same memory budget, or about 127.5% of raw
vLLM, assuming the incompressible-block rate fits in the smaller pool.

## Decode E2E Throughput

Output throughput is measured as generated tokens divided by `llm.generate()`
wall time. Total throughput includes prompt tokens, so it is dominated by prefill
for short decode lengths.

| Mode | Decode len | Elapsed (s) | Output tok/s | Total tok/s | Output tok/s vs raw | Slowdown vs raw |
|---|---:|---:|---:|---:|---:|---:|
| raw vLLM | 16 | 0.465 | 34.394 | 2235.624 | 100.0% | 1.00x |
| raw vLLM | 64 | 1.838 | 34.829 | 592.098 | 100.0% | 1.00x |
| raw vLLM | 128 | 3.669 | 34.883 | 313.950 | 100.0% | 1.00x |
| raw vLLM | 256 | 7.335 | 34.899 | 174.497 | 100.0% | 1.00x |
| ByteV2 raw-overlay | 16 | 12.111 | 1.321 | 85.871 | 3.8% | 26.03x |
| ByteV2 raw-overlay | 64 | 15.974 | 4.007 | 68.111 | 11.5% | 8.69x |
| ByteV2 raw-overlay | 128 | 21.466 | 5.963 | 53.667 | 17.1% | 5.85x |
| ByteV2 raw-overlay | 256 | 32.806 | 7.803 | 39.017 | 22.4% | 4.47x |
| ByteV2 compressed-only | 16 | 13.374 | 1.196 | 77.764 | 3.5% | 28.75x |
| ByteV2 compressed-only | 64 | 20.523 | 3.119 | 53.015 | 9.0% | 11.17x |
| ByteV2 compressed-only | 128 | 30.237 | 4.233 | 38.099 | 12.1% | 8.24x |
| ByteV2 compressed-only | 256 | 50.629 | 5.056 | 25.282 | 14.5% | 6.90x |

## Notes

- Raw vLLM was run in eager mode for apples-to-apples comparison with current
  ByteV2, whose backend currently disables CUDA graph capture.
- With the old 10% sparse fallback pool, ByteV2 compressed-only improves
  allocator-visible KV capacity by about 17.1% on this setup, but it is slower
  than raw-overlay in this benchmark. The current 3% default was confirmed in
  `benchmarks/byte_v2_decode_e2e_llama3_8b_p1024_b1_pool003_summary.md`: it
  reaches 212,656 KV tokens, about 127.5% of raw vLLM, without exhausting the
  sparse fallback pool on the tested workload.
- The main performance bottleneck is still the correctness-oriented ByteV2
  runtime path: decode attention is a conservative CTA baseline, and long
  prompt chunked/continuation prefill can still hit PyTorch gather fallback.
- During this benchmark, Llama-3-8B exposed a V2 Model Runner allocator gap and
  a compressed-only continuation-prefill sparse fallback gather gap; both were
  fixed before collecting the compressed-only numbers above.
