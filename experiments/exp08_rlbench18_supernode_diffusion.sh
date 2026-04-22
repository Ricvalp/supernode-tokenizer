#!/bin/bash
set -euo pipefail
PYTHONPATH=supernode-tokenizer python supernode-tokenizer/scripts/train_diffusion_policy.py \
  --config=supernode-tokenizer/configs/train_diffusion_policy.py \
  --config.model.encoder_name=supernode \
  --config.output.run_name=rlbench18_supernode_diffusion
