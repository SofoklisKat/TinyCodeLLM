#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

usage() {
  cat <<'EOF'
Evaluate a scratch-pretrain snapshot on MBPP.

Usage:
  ./scripts/eval_snapshot.sh                 # latest checkpoint
  ./scripts/eval_snapshot.sh 100000          # checkpoint-100000
  ./scripts/eval_snapshot.sh --final         # outputs/.../final
  ./scripts/eval_snapshot.sh --list          # list snapshots
  ./scripts/eval_snapshot.sh 100000 --limit 50

Options are passed through to train.eval_snapshot.
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

python3 -m train.eval_snapshot "$@"
