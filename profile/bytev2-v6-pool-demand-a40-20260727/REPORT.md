# ByteV2 V6 outlier-pool demand 扫描

**目标：** 判断 V5 的 1,024-entry 固定 outlier pool 是否可以缩小，并为
V6 固定页选择提供数据门

**Capture：** Llama-3.1-8B-Instruct，32 层，BF16，8 KV head，
head dimension 128

**日期：** 2026-07-27

**实验目录：** `profile/bytev2-v6-pool-demand-a40-20260727/`

## 结论

在当前单模型、batch 1、自然文本 capture 中，V6-256 有很大样本内余量：

- calibration 共 1,024 page，pool demand mean/P99/max 为
  3.980/15/29；
- evaluation 共 2,048 page，pool demand mean/P99/max 为
  3.514/9/27；
- 256-entry pool 在完整 page 和所有 `valid_rows=1..16` 前缀上均为
  0 overflow；
- 即使 128-entry pool 在这个样本中也为 0 overflow，但当前 metadata
  ABI 下的 128-entry 方案不能直接落地，且样本范围不足以批准删除
  authoritative raw fallback。

因此本轮选择 V6-256：沿用 V5 的 896 B metadata 和 49,152 B dense
payload，只将 pool 从 1,024 entry 缩到 256 entry。page 从 52,096 B
缩到 50,560 B，相对 raw BF16 page 节省 22.8516%，比 V5 每页再减少
1,536 B。

这份扫描是容量选择的支持证据，不是全分布安全证明。生产路径仍必须在
compact publication 前保留 authoritative raw sidecar；direct
compact-only path 遇到 overflow 或 fallback tile 会 fail-closed/trap。
V6 production planner 和 engine workspace binder 因此要求显式启用
`BYTE_V2_FA2_HYBRID_RAW_FALLBACK=1`。

## 1. Demand 定义

扫描器复现 production writer 的分配口径：

1. 一个 page 含 16 token；K/V 共形成 128 个 tile segment。
2. 每个 tile 在 BF16 high-7 exponent 空间选择宽度为 8 的最优连续窗口。
3. 窗口外元素记为 outlier。
4. 每个 segment 的实际 pool allocation 为：0 个 outlier 分配 0，
   否则向上取整到 2 的幂。
5. page demand 是 128 个 segment allocation 之和；只有
   `page_demand > pool_entries` 才算 overflow。

因此报告中的 `exact_outliers` 与 `page_demand` 不一定相等；后者包含
production power-of-two segment allocation 的内部余量。

扫描对每个完整 page 以及 `valid_rows=1..16` 的每个前缀都重新计算最优
窗口和 demand，覆盖 ragged tail 的 page-level pool 需求。

## 2. 输入与 provenance

Capture 文件：

```text
profile/splitzip-residual-redundancy-a40-20260727/raw/
  llama_all_layers_natural_cal512_eval1024.pt
```

SHA-256：

```text
c722d01849256af1a06fbaea07e86be61e4e8ab024f58dafdb18f7d28a7a4fe0
```

数据形状：

| Split | Tokens/layer | Layers | Pages | BF16 K/V values |
| --- | ---: | ---: | ---: |
| Calibration | 512 | 32 | 1,024 | 33,554,432 |
| Evaluation | 1,024 | 32 | 2,048 | 67,108,864 |

Capture metadata 记录 calibration 使用一个自然文本样本，evaluation
使用两个自然文本样本；具体 record ID 与 token hash 保存在
[`pool_demand.json`](reports/pool_demand.json)。

## 3. Pool demand 结果

### 3.1 完整 page

| Split | Mean | P50 | P90 | P95 | P99 | P99.9 | Max |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Calibration | 3.980 | 4 | 7 | 8 | 15 | 19.977 | 29 |
| Evaluation | 3.514 | 3 | 6 | 7 | 9 | 17.953 | 27 |

256-entry pool 对 observed max 的 headroom 分别为 227 和 229 entry。
128-entry pool 的 headroom 也分别有 99 和 101 entry。

| Split | Capacity 128 overflow | Capacity 256 overflow |
| --- | ---: |
| Calibration，1,024 page | 0 | 0 |
| Evaluation，2,048 page | 0 | 0 |

### 3.2 Outlier 密度

| Split | Exact outliers | Outlier fraction | 单 tile 最大 outliers |
| --- | ---: | ---: |
| Calibration | 4,072 | 0.01214% | 3 |
| Evaluation | 7,191 | 0.01072% | 3 |

这解释了当前样本中 pool demand 很低，但不能外推到 code、多语言、异常
数值分布或其他模型。

### 3.3 Ragged valid-row 前缀

对 `valid_rows=1..16` 的所有前缀，calibration observed max 不超过 29，
evaluation 不超过 27；capacity 128 和 256 均为 0 overflow。完整的逐前缀
quantile、capacity 和 top-page 数据保存在
[`pool_demand.json`](reports/pool_demand.json)，逐层摘要保存在
[`pool_demand_by_layer.csv`](analysis/pool_demand_by_layer.csv)。

## 4. 格式候选

Raw page 为 65,536 B。现有 dense payload 为 49,152 B，每个 pool entry
为 2 B：

| 候选 | Metadata | Pool entries | Page bytes | 相对 raw 节省 | 相对 V5 少 |
| --- | ---: | ---: | ---: | ---: |
| V5-1024 | 896 B | 1,024 | 52,096 | 20.5078% | 0 B |
| V6-512 | 896 B | 512 | 51,072 | 22.0703% | 1,024 B |
| **V6-256** | **896 B** | **256** | **50,560** | **22.8516%** | **1,536 B** |
| V6-128，当前 metadata | 896 B | 128 | 50,304 | 23.2422% | 1,792 B |
| V6-128，u8 metadata 目标 | 640 B | 128 | 50,048 | 23.6328% | 2,048 B |

V6-128 的额外固定页收益相对 V6-256 只有 256–512 B/page，但需要 metadata
ABI 重设计，并且当前 capture 不能证明 128 在更广分布上的安全余量。
因此 V6-256 是这一轮较稳妥的工程点；V6-128 暂不进入 production ABI。

## 5. 限制与安全结论

当前证据只覆盖：

- 一个 Llama-3.1-8B 模型；
- 一次 512-token calibration 和一次 1,024-token evaluation capture；
- batch 1 的自然文本 prefill KV；
- 32 层，但没有独立的长上下文自然分布。

未覆盖 code、多语言、多 batch、decode-tail、其他模型、异常输入或长期运行
中的分布漂移。故“0 observed overflow”只能支持将常规页设为 V6-256，
不能支持：

- 删除 hybrid raw sidecar；
- 将 direct compact-only writer 宣称为 production-safe；
- 把 128-entry pool 直接作为新 ABI；
- 将当前 overflow rate 当成真实线上概率上界。

下一步应在 code/多语言/长上下文/多 batch/不同模型的 capture 上重复同一
扫描，并单独记录真实 production hybrid fallback 的 raw promotion 次数。

## 6. 复现

本报告对应的扫描命令：

```bash
cd /mnt/sdb/yxz/ByteV2/vllm
.venv/bin/python \
  profile/bytev2-v6-pool-demand-a40-20260727/analysis/scan_pool_demand.py \
  --capture \
  profile/splitzip-residual-redundancy-a40-20260727/raw/llama_all_layers_natural_cal512_eval1024.pt \
  --output \
  profile/bytev2-v6-pool-demand-a40-20260727/reports/pool_demand.json \
  --layer-csv \
  profile/bytev2-v6-pool-demand-a40-20260727/analysis/pool_demand_by_layer.csv
```

扫描器使用 CPU 读取 capture，不会发射 CUDA kernel。JSON 保存完整
provenance、demand contract、每个 prefix 的统计和格式候选；CSV 便于按层
检查 demand 与 overflow。
