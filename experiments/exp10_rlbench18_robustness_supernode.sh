#!/bin/bash
set -euo pipefail
CKPT_PATH=${1:?usage: $0 <checkpoint_path>}
PYTHONPATH=supernode-tokenizer python supernode-tokenizer/scripts/eval_robustness.py \
  --config=supernode-tokenizer/configs/eval_robustness.py \
  --config.checkpoint_path="$CKPT_PATH"
