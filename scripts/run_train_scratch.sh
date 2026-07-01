#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Downgrade datasets if you already installed 4.x:
#   pip install 'datasets>=3.0.0,<4.0.0'
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python3 -m train.train_scratch --config "${1:-configs/train_scratch_30m.yaml}"
