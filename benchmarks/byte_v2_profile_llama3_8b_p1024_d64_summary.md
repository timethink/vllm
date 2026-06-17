# Byte-v2 Llama-3-8B Profile Summary, prompt=1024 decode=64

模型：

```text
/mnt/sda1/yxz/byte_v2/Meta-Llama-3-8B-Instruct
```

主要配置：

- GPU: `CUDA_VISIBLE_DEVICES=2`, NVIDIA A40
- dtype: `bfloat16`
- prompt length: `1024`
- decode length: `64`
- batch size: `1`
- `block_size=16`
- `max_model_len=1104`
- `max_num_batched_tokens=1088`
- `enforce_eager=True`
- ByteV2 compressed-only 使用当前默认 `VLLM_BYTE_V2_SPARSE_FALLBACK_POOL_RATIO=0.03`

原始文件：

- `benchmarks/profiles/raw_p1024_d64_pool003_fork.json`
- `benchmarks/profiles/byte_v2_compressed_pool003_p1024_d64.json`
- `benchmarks/profiles/byte_v2_compressed_pool003_p1024_d64_fork.json`
- `benchmarks/profiles/byte_v2_compressed_pool003_p1024_d64_fork.nsys-rep`
- `benchmarks/profiles/byte_v2_compressed_pool003_p1024_d64_fork_cuda_kernels_cuda_gpu_kern_sum.csv`
- `benchmarks/profiles/prefix_off_raw_bytev2_p1024_d64.json`

说明：`nsys` 需要 `--trace-fork-before-exec=true --wait=all` 才能捕获 vLLM
EngineCore 子进程。该模式对 ByteV2 有明显额外开销，所以性能数字以普通 JSON
run 为准，kernel 构成以 fork nsys run 为准。

## E2E 结果

| Mode | Prefix cache | Elapsed (s) | Output tok/s |
|---|---:|---:|---:|
| raw vLLM | on | 1.839 | 34.80 |
| ByteV2 compressed-only | on | 22.202 | 2.88 |
| raw vLLM | off | 1.950 | 32.81 |
| ByteV2 compressed-only | off | 116.389 | 0.55 |

关闭 prefix cache 后 raw vLLM 只小幅变慢，但 ByteV2 大幅变慢。这说明当前 ByteV2
瓶颈不是 sparse fallback pool，也不只是 decode kernel，而是 prefill/cache update
路径中仍有大量 PyTorch eager 工作。默认 prefix cache 开启时，重复 prompt 会触发
continuation/paged prefill fallback；关闭 prefix cache 后，每次完整 prompt prefill
都会走 ByteV2 的 PyTorch raw prefill 路径，因此更慢。

## ByteV2 nsys kernel 聚合

`byte_v2_compressed_pool003_p1024_d64_fork_cuda_kernels_cuda_gpu_kern_sum.csv`
中 GPU kernel 总时间约 17.35s。按 kernel 名称粗分：

| Category | GPU kernel time | Share | Instances |
|---|---:|---:|---:|
| GEMV / torch attention or LM-head related | 6.227s | 35.9% | 2,173,607 |
| PyTorch elementwise/copy | 6.129s | 35.3% | 3,928,624 |
| PyTorch softmax | 3.209s | 18.5% | 1,066,321 |
| ByteV2 cache update compress touched pages | 0.876s | 5.1% | 351 |
| GEMM | 0.509s | 2.9% | 1,530 |
| ByteV2 native paged decode attention | 0.294s | 1.7% | 256 |
| Other | 0.103s | 0.6% | 46,135 |

关键观察：

- `byte_v2_paged_decode_attention_gqa_wmma_kernel` 只占 1.7% GPU kernel time，
  且只有 256 次 launch，刚好对应 warmup 的 `8 decode tokens * 32 layers`。
- 测量阶段的大量时间落在 PyTorch `softmax_warp_forward`、GEMV 和 elementwise/copy
  kernel 上，说明 measured run 主要不是 native ByteV2 decode kernel，而是 ByteV2
  prefill/continuation-prefill fallback。
- `byte_v2_compress_touched_pages_kernel` 本身也偏重，单次平均约 2.50ms。这对
  prompt prefill/cache update 是明显瓶颈。
- nsys CUDA API 表显示 fork profile 下有 7M+ `cudaLaunchKernel` 和 1M+
  `cudaMemcpyAsync` 调用，符合 PyTorch eager fallback 产生大量小 kernel 的特征。

## 主要瓶颈

1. **ByteV2 prefill attention 仍是 PyTorch fallback。**
   当前 `ByteV2AttentionImpl.forward()` 在 prefill 分支调用
   `byte_v2_raw_prefill_attention_torch()` 或 `byte_v2_paged_prefill_attention_torch()`。
   它没有复用 raw vLLM 的 FlashAttention prefill，因此 prompt_len=1024 时已经远慢于
   raw vLLM。

2. **Continuation/prefix prefill 会触发 paged-prefill fallback。**
   benchmark 中 warmup 和 measured run 使用相同 prompt。默认 prefix cache 开启后，
   measured run 命中已有 KV blocks，更容易进入从 ByteV2 page gather/decompress 的
   paged prefill fallback。这就是 nsys 中百万级 PyTorch softmax/GEMV/elementwise kernel
   的主要来源。

3. **Cache update compression kernel 还不是 production 级。**
   当前 compressed-only update 会先标记 touched page，再做
   `byte_v2_compress_touched_pages_kernel`。这个 kernel 对 prompt prefill 不够友好：
   它按 token launch/grid 组织、扫描同 block token，并经 raw staging 重建 block。
   对连续 prompt block，应该改成 one CTA per physical block 的直接压缩路径。

4. **Native decode kernel 仍不够快，但不是这次 e2e profile 的最大项。**
   当前 GQA WMMA decode 是一个 CTA 处理一个 request/KV head，循环全部 page，缺少
   split-K/page-parallel 并行和成熟的 online softmax/PV pipeline。即使移除 prefill
   fallback，它大概率仍难超过 raw vLLM 的 FlashAttention/FlashInfer decode。

5. **ByteV2 backend 仍禁用 CUDA graph。**
   这次 benchmark 对 raw 也用了 `enforce_eager=True` 做对照；真实 raw vLLM 开启
   cudagraph 后会更快。ByteV2 想超过 production raw vLLM，最终必须支持 cudagraph。

## 怎样才可能超过 raw vLLM

短期目标不是直接优化当前 kernel 微细节，而是先移除错误的慢路径：

1. **prefill 用 raw FlashAttention，不走 ByteV2 PyTorch prefill。**
   对首段 prompt，forward 中已经有 raw `q/k/v`。应该直接调用 raw vLLM 的
   FlashAttention/Triton prefill kernel 计算 attention，同时 cache update 另行写入
   ByteV2 compressed page。也就是：
   `prefill attention = raw vLLM fast path`，
   `KV storage = ByteV2 compressed cache`。

2. **实现 native ByteV2 paged prefill，或者先禁用会触发 fallback 的 prefix-cache路径。**
   如果需要 prefix caching/continuation prefill，就必须有 native CUDA paged prefill：
   block/page 级并行解压，QK/PV 使用 FlashAttention 风格 tiling，而不是 PyTorch gather
   后 softmax。

3. **重写 cache update compression 为 block-parallel fused kernel。**
   prompt prefill 连续 slot 的常见路径应直接 one CTA per block：
   从 raw K/V 读 16-token x 16-dim tile，计算 ByteV2 exponent window，写 compressed
   page；不可压缩 block 写 sparse fallback pool。避免按 token 扫描和 raw staging。

4. **重写 decode kernel 为 split-K/page-parallel。**
   当前 one CTA per KV head 对 1k+ context 并行度不足。需要多个 CTA 处理同一
   request/KV head 的 page chunks，输出 partial max/sum/accumulator，再做 reduce。
   目标是把每 token 每层 attention latency 降到 raw FlashInfer 同量级。

5. **启用 ByteV2 cudagraph。**
   需要保证 ByteV2 metadata、fallback pool pointer、page size、block table shape 在
   capture/replay 下稳定。否则即使 eager 对照接近 raw，production raw 仍会因为 cudagraph
   领先。

6. **选择更有利的 benchmark 区间。**
   Llama-3-8B batch=1、context=1k 时，decode 很大一部分时间在权重 GEMV/MLP/LM head，
   KV bandwidth 不是唯一主项。ByteV2 的 KV 压缩收益更可能在长 context、高 batch、
   或 KV bandwidth 占比更高的配置下体现。当前 ByteV2 page 对 Llama-3-8B 的 raw KV
   字节压缩约为 `65536 / 49424 = 1.33x`，如果解压和 kernel overhead 没有压到很低，
   这个压缩率不足以抵消慢路径。

## 建议的下一步

优先级：

1. 接入 raw FlashAttention prefill path，保留 ByteV2 cache update。
2. 重写 compressed-only cache update 为 block-parallel fast path。
3. 实现 native paged prefill 或暂时规避 prefix-cache continuation prefill fallback。
4. 再 profile pure decode kernel，针对 split-K/page-parallel 优化。
5. 最后启用 cudagraph，再和 production raw vLLM 比较。
