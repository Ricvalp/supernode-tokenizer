#!/bin/bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
python scripts/train_chunk_policy.py \
  --config=configs/train_chunk_policy.py \
  --config.model.encoder_name=supernode \
  --config.data.low_data_train_demos_per_variation=20 \
  --config.output.run_name=rlbench18_supernode_chunk_lowdata20
