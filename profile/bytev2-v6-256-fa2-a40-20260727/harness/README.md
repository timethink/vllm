# V6-256 FA2 harness

本目录保存 ByteV2 V6-256 实验使用的可复现 harness。所有 Python 命令都应
通过仓库的 `.venv/bin/python` 执行。

## 文件

- `benchmark_three_way.py`：在相同 FA2 参数下轮换测量 raw、ByteV2 V6
  和固定页 SplitZip，使用 CUDA event 计时，并强制 output/LSE 相对 raw
  bitwise 一致。
- `profile_main_kernel.py`：准备 65K Q1 输入，完成 bitwise gate 后只在
  `cudaProfilerStart/Stop` 区间发射一次指定 backend，供 NCU 捕获主 kernel。
- `splitzip_fixed_page.py`：实验用 SplitZip V5-1024 固定页 packer。
- `sitecustomize.py`：确保子进程使用当前仓库源码与已构建 extension。
- `run_split_e2e.sh`：固定 production E2E 环境变量和模型参数，运行一次
  raw 或 ByteV2 请求。
- `run_production_abba.sh`：按 raw → ByteV2 → ByteV2 → raw 执行一个
  ABBA block。

## 前置条件

这些 harness 不负责构建 extension。运行前需保证当前源码对应的
`_C_stable_libtorch` 和 `_vllm_fa2_C` 已构建并安装到当前仓库环境，且
本地模型与 capture 存在：

```text
/mnt/sdb/yxz/ByteV2/Meta-Llama-3.1-8B-Instruct
profile/bytev2-splitzip-codec-a40-20260726/raw/llama_layer0_cal2048_eval4096.pt
```

结果脚本默认拒绝覆盖已有 JSON/JSONL。复现时请换用新的输出目录或新的
repetition 编号，不要覆盖本次报告产物。

## CUDA-event 三路对比

本轮参数为 Q1，4K/16K/65K 分别使用 16/19/20 split。每个 sample 包含
20 次调用；三种 backend 按全部六种排列轮换。

下面是 65K repetition 1 对应的命令：

```bash
cd /mnt/sdb/yxz/ByteV2/vllm
CUDA_VISIBLE_DEVICES=0 .venv/bin/python \
  profile/bytev2-v6-256-fa2-a40-20260727/harness/benchmark_three_way.py \
  --seq-len 65536 \
  --query-len 1 \
  --num-splits 20 \
  --warmup 12 \
  --iterations 60 \
  --calls-per-sample 20 \
  --repetition 1 \
  --output \
  profile/bytev2-v6-256-fa2-a40-20260727/reports/event_s65536_split20_rep1.json
```

4K 和 16K 使用同一命令，分别替换为：

```text
--seq-len 4096  --num-splits 16
--seq-len 16384 --num-splits 19
```

本轮 4K/16K 各有 repetition 1–5，65K 有 repetition 1–10。重复运行时
必须同时更新 `--repetition` 和 `--output`。

每个 JSON 的关键字段：

- `correctness.*.output_bit_mismatch` 和 `lse_bit_mismatch` 必须为 0；
- `timings.*.median_us` 是单次 kernel 调用的 median；
- `ratios.*.latency_gap_percent` 定义为 candidate/raw - 1；
- `storage` 给出三种固定表示的字节数。

## E2E ABBA

本轮 E2E 使用 batch 1、65,536 context、128 output token、20 GB
KV-cache planner，执行顺序为 raw → V6 → V6 → raw：

```bash
cd /mnt/sdb/yxz/ByteV2/vllm
CUDA_VISIBLE_DEVICES=0 \
  profile/bytev2-v6-256-fa2-a40-20260727/harness/run_production_abba.sh \
  1 128 \
  profile/bytev2-v6-256-fa2-a40-20260727/reports/e2e_production
```

`run_split_e2e.sh` 固定了本轮使用的 ByteV2 writer、hybrid raw fallback、
compiled-mode 和 planner 环境变量。E2E 使用 `num_splits=0`，不要把它与
隔离测试的 65K/20-split kernel 时间直接相减。

V6 production planner 会拒绝 compact-only 配置；手工运行 ByteV2 E2E
时必须保留脚本中的 `BYTE_V2_FA2_HYBRID_RAW_FALLBACK=1`。

结果检查：

```bash
jq '{
  backend,
  token_ids,
  performance_valid_for_tps,
  request_metrics,
  kv_cache_plan,
  hybrid_raw_fallback_state
}' \
  profile/bytev2-v6-256-fa2-a40-20260727/reports/e2e_production/*.jsonl
```

TPOT 应按 `request_metrics[0].decode_seconds / (output_tokens - 1)` 计算。
本轮四个请求的 token 序列完全一致，且没有 preemption。

## Nsight Compute

NCU 版本为 `/usr/local/cuda/bin/ncu` 2025.4.1。`profile_main_kernel.py`
在 profiler 区间内只执行一次候选调用，且调用前先与 raw 做 output/LSE
bitwise gate。

### Full + PM sampling：V6

```bash
cd /mnt/sdb/yxz/ByteV2/vllm
/usr/local/cuda/bin/ncu \
  --profile-from-start off \
  --target-processes all \
  --set full \
  --section PmSampling \
  --section PmSampling_WarpStates \
  --kernel-name 'regex:.*flash_fwd_splitkv_byte_v2_kernel.*' \
  --launch-count 1 \
  --export \
  profile/bytev2-v6-256-fa2-a40-20260727/reports/full_v6_bytev2_s65536_q1 \
  .venv/bin/python \
  profile/bytev2-v6-256-fa2-a40-20260727/harness/profile_main_kernel.py \
  --backend bytev2 --seq-len 65536 --num-splits 20 --warmup 8
```

### Full + PM sampling：raw

```bash
cd /mnt/sdb/yxz/ByteV2/vllm
/usr/local/cuda/bin/ncu \
  --profile-from-start off \
  --target-processes all \
  --set full \
  --section PmSampling \
  --section PmSampling_WarpStates \
  --kernel-name 'regex:.*flash_fwd_splitkv_kernel.*' \
  --launch-count 1 \
  --export \
  profile/bytev2-v6-256-fa2-a40-20260727/reports/full_raw_s65536_q1 \
  .venv/bin/python \
  profile/bytev2-v6-256-fa2-a40-20260727/harness/profile_main_kernel.py \
  --backend raw --seq-len 65536 --num-splits 20 --warmup 8
```

### SourceCounters

V6：

```bash
cd /mnt/sdb/yxz/ByteV2/vllm
/usr/local/cuda/bin/ncu \
  --profile-from-start off \
  --target-processes all \
  --set source \
  --section SourceCounters \
  --kernel-name 'regex:.*flash_fwd_splitkv_byte_v2_kernel.*' \
  --launch-count 1 \
  --export \
  profile/bytev2-v6-256-fa2-a40-20260727/reports/source_v6_bytev2_s65536_q1 \
  .venv/bin/python \
  profile/bytev2-v6-256-fa2-a40-20260727/harness/profile_main_kernel.py \
  --backend bytev2 --seq-len 65536 --num-splits 20 --warmup 8
```

raw 使用相同命令，将 regex 改为 `.*flash_fwd_splitkv_kernel.*`，backend
改为 `raw`，输出名改为 `source_raw_s65536_q1`。

NCU report 自带的 duration 可能受 replay 和 profiling 干扰。延迟结论以
CUDA-event repetitions 为准；NCU 用于解释 DRAM、occupancy、stall 和
source hotspot。

## 汇总与图表

汇总脚本只依赖 Python 标准库：

```bash
cd /mnt/sdb/yxz/ByteV2/vllm
.venv/bin/python \
  profile/bytev2-v6-256-fa2-a40-20260727/analysis/analyze_results.py
```

它会重新生成 `summary.json`、两个 CSV 和两张 SVG；不会运行新的 GPU
实验。

## 产物索引

- `../reports/event_*.json`：CUDA-event 原始结果；
- `../reports/e2e_production/*.jsonl`：E2E 请求与 trace；
- `../reports/{full,source}_*.ncu-rep`：可用 NCU UI 重新打开的报告；
- `../analysis/compare_raw_vs_v6.txt`：关键 metrics 并排比较；
- `../analysis/stall_hotspots_{raw,v6}.txt`：逐 source line stall；
- `../analysis/pm_timeline_plots.txt`：PM sampling ASCII timeline。
- `../analysis/{summary.json,event_latency.csv,e2e_summary.csv}`：统一汇总；
- `../analysis/{latency_vs_seq.svg,capacity_storage.svg}`：报告图表。
