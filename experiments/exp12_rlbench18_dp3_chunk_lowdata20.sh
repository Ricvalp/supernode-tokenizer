#!/bin/bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
python scripts/train_chunk_policy.py \
  --config=configs/train_chunk_policy.py \
  --config.model.encoder_name=dp3 \
  --config.data.low_data_train_demos_per_variation=20 \
  --config.output.run_name=rlbench18_dp3_chunk_lowdata20
