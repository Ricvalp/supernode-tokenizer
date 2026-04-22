#!/bin/bash
set -euo pipefail
PYTHONPATH=supernode-tokenizer python supernode-tokenizer/scripts/train_chunk_policy.py \
  --config=supernode-tokenizer/configs/train_chunk_policy.py \
  --config.model.encoder_name=perceiver \
  --config.data.low_data_train_demos_per_variation=20 \
  --config.output.run_name=rlbench18_perceiver_chunk_lowdata20
