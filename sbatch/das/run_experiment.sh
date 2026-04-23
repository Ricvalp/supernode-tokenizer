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
nvidia-smi || true

printf '[supernode-tokenizer] launching %s' "$EXPERIMENT_SCRIPT"
if [ $# -gt 0 ]; then
  printf ' %s' "$@"
fi
printf '\n'

exec bash "$SUPERNODE_TOKENIZER_REPO_ROOT/experiments/$EXPERIMENT_SCRIPT" "$@"
