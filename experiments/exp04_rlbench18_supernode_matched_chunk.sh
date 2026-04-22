#!/bin/bash
set -euo pipefail
PYTHONPATH=supernode-tokenizer python supernode-tokenizer/scripts/train_chunk_policy.py \
  --config=supernode-tokenizer/configs/train_chunk_policy.py \
  --config.model.encoder_name=supernode \
  --config.model.supernode_encoder.num_supernodes=128 \
  --config.model.supernode_nomsg_encoder.num_supernodes=128 \
  --config.output.run_name=rlbench18_supernode_matched_chunk
