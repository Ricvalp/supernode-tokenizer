#!/bin/bash
set -euo pipefail
CKPT_PATH=${1:?usage: $0 <checkpoint_path>}
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
python scripts/eval_robustness.py \
  --config=configs/eval_robustness.py \
  --config.checkpoint_path="$CKPT_PATH"
