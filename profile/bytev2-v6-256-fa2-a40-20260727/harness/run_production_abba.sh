#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 <block-id> <max-tokens> <reports-dir>" >&2
  exit 2
fi

block_id=$1
max_tokens=$2
reports_dir=$3
script_dir=/mnt/sdb/yxz/ByteV2/vllm/profile/bytev2-v6-256-fa2-a40-20260727/harness

mkdir -p "$reports_dir"

run_one() {
  local slot=$1
  local backend=$2
  local mode=$3
  local output_jsonl
  output_jsonl="$reports_dir/production_o${max_tokens}_r${block_id}_${slot}_${mode}.jsonl"
  "$script_dir/run_split_e2e.sh" \
    "$backend" \
    0 \
    "$max_tokens" \
    "$output_jsonl"
}

run_one a flash_attn raw
run_one b byte_v2 byte
run_one c byte_v2 byte
run_one d flash_attn raw
