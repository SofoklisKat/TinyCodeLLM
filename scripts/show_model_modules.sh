#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
python3 -m train.show_model_modules "$@"
