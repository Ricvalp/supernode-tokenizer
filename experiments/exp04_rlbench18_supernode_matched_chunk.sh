#!/bin/bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
python scripts/train_chunk_policy.py \
  --config=configs/train_chunk_policy.py \
  --config.model.encoder_name=supernode \
  --config.model.supernode_encoder.num_supernodes=128 \
  --config.model.supernode_nomsg_encoder.num_supernodes=128 \
  --config.output.run_name=rlbench18_supernode_matched_chunk
