#!/bin/sh
set -eu

if [ $# -lt 1 ]; then
  printf '%s\n' "usage: $0 <experiment_script> [experiment args...]" >&2
  exit 1
fi

EXPERIMENT_SCRIPT="$1"
shift

REPO_ROOT="/home/valperga/supernode-tokenizer"
. "$REPO_ROOT/experiments/env_das.sh"

mkdir -p \
  "$SUPERNODE_TOKENIZER_OUTPUT_ROOT" \
  "$SUPERNODE_TOKENIZER_CHECKPOINT_ROOT" \
  "$SUPERNODE_TOKENIZER_EVAL_ROOT"

cd "$SUPERNODE_TOKENIZER_REPO_ROOT"

hostname
if MEMORY_LIMIT=$(ulimit -m 2>/dev/null); then
  printf 'memory=%s\n' "$MEMORY_LIMIT"
else
  printf '%s\n' 'memory=unavailable'
fi
printf 'nproc=%s\n' "$(nproc)"
printf 'CUDA_VISIBLE_DEVICES=%s\n' "${CUDA_VISIBLE_DEVICES:-<unset>}"
printf 'CONDA_DEFAULT_ENV=%s\n' "${CONDA_DEFAULT_ENV:-<unset>}"
printf 'CONDA_PREFIX=%s\n' "${CONDA_PREFIX:-<unset>}"
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi
else
  printf '%s\n' '[supernode-tokenizer] nvidia-smi not found in PATH'
fi
printf 'python=%s\n' "$(command -v python)"
python - <<'PY'
import sys
import torch

print(f"torch={torch.__version__}")
print(f"torch.version.cuda={torch.version.cuda}")
print(f"torch.cuda.is_available={torch.cuda.is_available()}")
print(f"torch.cuda.device_count={torch.cuda.device_count()}")
if not torch.cuda.is_available():
    raise SystemExit(
        "[supernode-tokenizer] CUDA is not available inside this Slurm job. "
        "Check the allocated node environment or install a CUDA-enabled PyTorch build."
    )
PY

printf '[supernode-tokenizer] launching %s' "$EXPERIMENT_SCRIPT"
if [ $# -gt 0 ]; then
  printf ' %s' "$@"
fi
printf '\n'

exec bash "$SUPERNODE_TOKENIZER_REPO_ROOT/experiments/$EXPERIMENT_SCRIPT" "$@"
