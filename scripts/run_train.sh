#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python3 -m train.train_qlora --config configs/train_qlora.yaml
