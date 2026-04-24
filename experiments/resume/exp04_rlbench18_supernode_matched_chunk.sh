#!/bin/bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"
: "${SUPERNODE_TOKENIZER_CHECKPOINT_ROOT:?source an environment script that sets SUPERNODE_TOKENIZER_CHECKPOINT_ROOT}"
RUN_NAME="rlbench18_supernode_matched_chunk"
RESUME_PATH="${SUPERNODE_TOKENIZER_CHECKPOINT_ROOT}/${RUN_NAME}/step_0000000.pt"
python scripts/train_chunk_policy.py \
  --config=configs/train_chunk_policy.py \
  --config.model.encoder_name=supernode \
  --config.model.supernode_encoder.num_supernodes=128 \
  --config.model.supernode_nomsg_encoder.num_supernodes=128 \
  --config.output.run_name="$RUN_NAME" \
  --config.train.resume_path="$RESUME_PATH"
