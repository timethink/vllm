# Byte-v2 Llama-3-8B Decode E2E Benchmark With 3% Sparse Fallback Pool

模型：

```text
/mnt/sda1/yxz/byte_v2/Meta-Llama-3-8B-Instruct
```

测试配置：

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
- compressed-only 使用当前默认 `VLLM_BYTE_V2_SPARSE_FALLBACK_POOL_RATIO=0.03`

原始结果文件：

- `benchmarks/byte_v2_decode_e2e_llama3_8b_p1024_b1_pool003.json`
- `benchmarks/byte_v2_decode_e2e_llama3_8b_p1024_b1_pool003.log`
- `benchmarks/byte_v2_decode_e2e_llama3_8b_p1024_b1_pool003_stats.json`
- `benchmarks/byte_v2_decode_e2e_llama3_8b_p1024_b1_pool003_stats.log`

第一份 JSON 包含 raw、raw-overlay、compressed-only 的吞吐对比；其中 sparse
fallback stats 采样被 vLLM 安全序列化开关拒绝。第二份 JSON 只补跑
compressed-only，并启用 `VLLM_ALLOW_INSECURE_SERIALIZATION=1` 采集 pool 使用量。
吞吐计时不包含 stats 采样。随后 benchmark 脚本已改成 worker 字符串 RPC，不再需要
该 env；`benchmarks/byte_v2_decode_e2e_llama3_8b_pool003_rpc_smoke.json`
验证了新采样路径。

## KV 容量

| Mode | GPU KV cache size | Relative to raw |
|---|---:|---:|
| raw vLLM | 166,768 tokens | 100.0% |
| ByteV2 raw-overlay | 166,736 tokens | 100.0% |
| ByteV2 compressed-only + 3% sparse fallback pool | 212,656 tokens | 127.5% |

对比旧 10% pool 的历史结果，compressed-only 容量从 195,216 tokens 提升到
212,656 tokens，增加 17,440 tokens。相对 raw vLLM 的容量收益从约 17.1%
提升到约 27.5%。

## Decode E2E 吞吐

output throughput 是生成 token 数除以完整 `llm.generate()` wall time。短 decode
点仍包含 prompt 处理和 cache update，不是纯 decode kernel latency。

| Mode | Decode len | Elapsed (s) | Output tok/s | Output tok/s vs raw | Slowdown vs raw |
|---|---:|---:|---:|---:|---:|
| raw vLLM | 16 | 0.466 | 34.33 | 100.0% | 1.00x |
| raw vLLM | 64 | 1.837 | 34.83 | 100.0% | 1.00x |
| raw vLLM | 128 | 3.670 | 34.88 | 100.0% | 1.00x |
| raw vLLM | 256 | 7.336 | 34.90 | 100.0% | 1.00x |
| ByteV2 raw-overlay | 16 | 12.407 | 1.29 | 3.8% | 26.62x |
| ByteV2 raw-overlay | 64 | 17.525 | 3.65 | 10.5% | 9.54x |
| ByteV2 raw-overlay | 128 | 24.533 | 5.22 | 15.0% | 6.69x |
| ByteV2 raw-overlay | 256 | 39.015 | 6.56 | 18.8% | 5.32x |
| ByteV2 compressed-only | 16 | 13.609 | 1.18 | 3.4% | 29.20x |
| ByteV2 compressed-only | 64 | 21.942 | 2.92 | 8.4% | 11.94x |
| ByteV2 compressed-only | 128 | 33.340 | 3.84 | 11.0% | 9.08x |
| ByteV2 compressed-only | 256 | 56.636 | 4.52 | 13.0% | 7.72x |

3% pool 对吞吐没有明显改善或恶化；compressed-only 的 decode_len=256 仍约为
raw vLLM 的 13.0%，与旧 10% pool 下的 GQA WMMA 结果基本一致。

## Sparse Fallback Pool 使用量

compressed-only run 完整跑通，没有出现 pool exhausted。每层 pool capacity 为
399 slots，32 层总 capacity 为 12,768 slots。

| Point | Any exhausted | Total next_slot / capacity | Total pool used | Max layer next_slot / capacity | Max layer pool used |
|---|---:|---:|---:|---:|---:|
| warmup | false | 1,857 / 12,768 | 14.5% | 62 / 399 | 15.5% |
| decode_len=16 | false | 1,918 / 12,768 | 15.0% | 64 / 399 | 16.0% |
| decode_len=64 | false | 2,075 / 12,768 | 16.3% | 69 / 399 | 17.3% |
| decode_len=128 | false | 2,360 / 12,768 | 18.5% | 78 / 399 | 19.5% |
| decode_len=256 | false | 2,901 / 12,768 | 22.7% | 95 / 399 | 23.8% |

`next_slot` 是累计分配过的 fallback slots。当前实现没有 slot 回收，所以它是判断
是否接近耗尽的保守指标。最终最紧张的一层是 95/399，约 23.8% pool 使用率；
折算到 physical blocks 是约 0.71%，低于 3% pool 预算。

## 结论

- 在这组 Llama-3-8B、prompt_len=1024、decode_len<=256、batch=1 的 e2e 负载下，
  3% sparse fallback pool 没有耗尽。
- 3% pool 将 allocator-visible KV capacity 提升到 212,656 tokens，约为 raw
  vLLM 的 127.5%。
- 当前性能瓶颈仍在 ByteV2 runtime/decode kernel 路径，不在 sparse fallback pool
  容量本身；3% pool 主要带来容量收益，吞吐基本维持旧结果。
