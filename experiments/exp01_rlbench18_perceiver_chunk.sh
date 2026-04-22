#!/bin/bash
set -euo pipefail
PYTHONPATH=supernode-tokenizer python supernode-tokenizer/scripts/train_chunk_policy.py \
  --config=supernode-tokenizer/configs/train_chunk_policy.py \
  --config.model.encoder_name=perceiver \
  --config.output.run_name=rlbench18_perceiver_chunk
