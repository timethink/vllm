# ByteV2 current decode kernel vs raw BF16 FA2, 2026-06-23

## Scope

本文记录当前 ByteV2 decode kernel 和 raw BF16 KV baseline 的性能差距。

这里的 `raw` 指：

- raw BF16 K/V cache：`raw_key_cache` / `raw_value_cache`
- vLLM FlashAttention FA2 paged decode：
  `flash::flash_fwd_splitkv_kernel`
- 不包含 ByteV2 codec、page metadata、fallback/outlier overlay 或 split-k
  workspace

这里的 `ByteV2 current` 指：

- generic split-k：当前保留的 payload tile descriptor 版本
- GQA4 guarded：no-outlier safe-page lower-bound，当前仍使用原 loader，
  tile softmax、QK row-level reduction 和 split-k reduce 均使用 warp-level
  GQA-specialized kernels
- 上一轮 `shared page pointer cache` 实验已回退，不包含在当前 kernel 中

本报告只比较 isolated decode attention kernel 时间，不比较 tokenizer、scheduler、
sampling、prefill 或 E2E wall time。

## Environment

- GPU: NVIDIA A40, `CUDA_VISIBLE_DEVICES=4`
- dtype: BF16 query / BF16 raw K/V
- heads: `num_heads=32`, `num_kv_heads=8`, `head_size=128`
- block size: `16`
- ByteV2 tile policy: `(16, 16, 16, 64, 128, 128)`
- metric: CUDA event `median_ms`

Artifacts:

- `profiles/byte_v2_current_vs_raw_generic_b1_1k16k.jsonl`
- `profiles/byte_v2_current_vs_raw_gqa_warp_reduce_b1_1k16k.jsonl`
- `profiles/byte_v2_current_vs_raw_generic_b4_4k16k.jsonl`
- `profiles/byte_v2_current_vs_raw_gqa_warp_reduce_b4_4k16k.jsonl`
- `profiles/byte_v2_ncu_gqa_guarded_16384_p64_warp_qk.ncu-rep`
- `profiles/byte_v2_ncu_gqa_guarded_16384_p64_warp_reduce.ncu-rep`
- `profiles/byte_v2_decode_microbench_gqa4_pv_desc_b1_8k16k.jsonl`
- `profiles/byte_v2_decode_microbench_gqa4_pv_desc_b4_8k16k.jsonl`
- `profiles/byte_v2_ncu_gqa_guarded_16384_p64_pv_desc.ncu-rep`

## 1. Kernel Time Gap

### Generic ByteV2, batch=1

| seq_len | ByteV2 best | ByteV2 median | raw FA2 median | gap | workspace |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1024 | p16 | 0.0901 ms | 0.0614 ms | 1.47x | 1.01 MiB |
| 2048 | p32 | 0.1720 ms | 0.0635 ms | 2.71x | 1.01 MiB |
| 4096 | p32 | 0.3021 ms | 0.0625 ms | 4.84x | 2.02 MiB |
| 8192 | p64 | 0.5448 ms | 0.0727 ms | 7.49x | 2.02 MiB |
| 16384 | p128 | 1.0404 ms | 0.1239 ms | 8.40x | 2.02 MiB |

### GQA4 guarded lower-bound, batch=1

| seq_len | ByteV2 best | ByteV2 median | raw FA2 median | gap | workspace |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1024 | p16 | 0.0358 ms | 0.0625 ms | 0.57x | 1.01 MiB |
| 2048 | p32 | 0.0584 ms | 0.0645 ms | 0.90x | 1.01 MiB |
| 4096 | p64 | 0.1024 ms | 0.0645 ms | 1.59x | 1.01 MiB |
| 8192 | p32 | 0.1638 ms | 0.0727 ms | 2.25x | 4.03 MiB |
| 16384 | p64 | 0.3062 ms | 0.1239 ms | 2.47x | 4.03 MiB |

### Generic ByteV2, batch=4

| seq_len | ByteV2 best | ByteV2 median | raw FA2 median | gap | workspace |
| ---: | --- | ---: | ---: | ---: | ---: |
| 4096 | p64 / p128 | 1.0117 ms | 0.1239 ms | 8.17x | 4.03 MiB |
| 8192 | p64 | 1.8350 ms | 0.2232 ms | 8.22x | 8.06 MiB |
| 16384 | p128 | 3.3751 ms | 0.4209 ms | 8.02x | 8.06 MiB |

### GQA4 guarded lower-bound, batch=4

| seq_len | ByteV2 best | ByteV2 median | raw FA2 median | gap | workspace |
| ---: | --- | ---: | ---: | ---: | ---: |
| 4096 | p32 | 0.3195 ms | 0.1249 ms | 2.56x | 8.06 MiB |
| 8192 | p64 | 0.5765 ms | 0.2232 ms | 2.58x | 8.06 MiB |
| 16384 | p128 | 1.1274 ms | 0.4219 ms | 2.67x | 8.06 MiB |

## 2. Nsight Compute Gap

下面是已有 ncu replay 指标。绝对 duration 会受 replay 影响，主要看结构性指标。
generic 行使用 descriptor 后的当前 16k/p128 结果；GQA4 main 行使用 q_group
fused reduction + warp softmax + warp QK + PV descriptor 后的 16k/p64 结果；
GQA4 reduce 行记录 warp reduce 后的 split-k reduce kernel。

| kernel | case | ncu duration | DRAM GB/s | issue active | eligible warps | regs | top stall |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| ByteV2 generic current | 16k p128 | 1248.3 us | 72.8 | 51.7% | 0.86 | 48 | long_scoreboard 7.5 |
| ByteV2 GQA4 main | 16k p64 | 353.8 us | 186.8 | 72.4% | 2.92 | 48 | long_scoreboard 2.84 |
| ByteV2 GQA4 reduce | 16k p64 | 20.5 us | 206.5 | 26.3% | 0.51 | 24 | long_scoreboard 32.17 |
| raw FA2 paged | 16k | 115.1 us | 585.5 | 10.7% | 0.11 | 252 | long_scoreboard 3.23 |

Interpretation:

1. raw FA2 能把 raw BF16 KV 当作 streaming workload 推到约 585 GB/s。
2. ByteV2 generic 当前只有约 73 GB/s，说明不是纯 HBM 带宽上限，而是
   payload decode、metadata address dependency 和 scalar reduction 限制了吞吐。
3. GQA4 把 repeated KV work 降低后，时间差距从 generic 的约 8x 降到约 2.25-2.67x。
   q_group fused reduction + warp softmax + warp QK 基本打掉了 `barrier`；PV descriptor
   进一步降低了主 kernel 的 `long_scoreboard`。当前瓶颈转向剩余 payload decode /
   memory dependency、寄存器压力和 split-k workspace。

## 3. Memory Footprint Note

当前默认 V4 page 还不是存储压缩收益版本：

| layout | bytes per 16-token page |
| --- | ---: |
| raw BF16 K/V, 8 KV heads, head_dim=128 | 65536 B |
| ByteV2 V4 current page | 115328 B |

当前 ByteV2 page 约是 raw BF16 page 的 1.76x。原因是 V4 layout 为 fallback /
outlier overlay 预留了 full outlier payload 容量。所以上面的性能对比不能解释为
"ByteV2 用更少 HBM 但更慢"；当前阶段主要是在验证 kernel 结构和 codec 路径。

## 4. Current Bottom Line

1. generic ByteV2 当前长上下文 isolated decode kernel 仍比 raw FA2 慢约 8x。
2. GQA4 guarded lower-bound 能把长上下文差距压到约 2.25x-2.67x。
3. batch=4 后差距没有明显收敛，说明主要问题不是 CTA 数不足。GQA4 的 barrier
   已基本打掉，PV safe-page descriptor 已带来约 5% 几何平均收益；剩余主要转向
   QK/PV payload decode dependency、寄存器压力和 split-k workspace。
4. 下一步优化重点应放在 GQA QK payload descriptor / macro descriptor、PV tile
   staging/vectorized decode、split-k workspace 和 generic payload dependency，而不是
   只缓存 page pointer 这类外围 metadata。

## 5. Reproduce Commands

```bash
CUDA_VISIBLE_DEVICES=4 .venv/bin/python scripts/byte_v2_decode_microbench.py \
  --seq-lens 1024 2048 4096 8192 16384 \
  --partition-sizes 16 32 64 128 \
  --compute-block-ns 64 \
  --include-flash \
  --iters 200 \
  --warmup 30 \
  --output-jsonl profiles/byte_v2_current_vs_raw_generic_b1_1k16k.jsonl

CUDA_VISIBLE_DEVICES=4 .venv/bin/python scripts/byte_v2_decode_microbench.py \
  --seq-lens 1024 2048 4096 8192 16384 \
  --partition-sizes 16 32 64 128 \
  --compute-block-ns 64 \
  --include-flash \
  --assume-no-outlier \
  --guarded-split \
  --gqa-packed \
  --iters 200 \
  --warmup 50 \
  --output-jsonl profiles/byte_v2_current_vs_raw_gqa_warp_reduce_b1_1k16k.jsonl

CUDA_VISIBLE_DEVICES=4 .venv/bin/python scripts/byte_v2_decode_microbench.py \
  --num-seqs 4 \
  --seq-lens 4096 8192 16384 \
  --partition-sizes 32 64 128 \
  --compute-block-ns 64 \
  --include-flash \
  --iters 150 \
  --warmup 30 \
  --output-jsonl profiles/byte_v2_current_vs_raw_generic_b4_4k16k.jsonl

CUDA_VISIBLE_DEVICES=4 .venv/bin/python scripts/byte_v2_decode_microbench.py \
  --num-seqs 4 \
  --seq-lens 4096 8192 16384 \
  --partition-sizes 32 64 128 \
  --compute-block-ns 64 \
  --include-flash \
  --assume-no-outlier \
  --guarded-split \
  --gqa-packed \
  --iters 150 \
  --warmup 50 \
  --output-jsonl profiles/byte_v2_current_vs_raw_gqa_warp_reduce_b4_4k16k.jsonl
```
