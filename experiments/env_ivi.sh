#!/bin/sh

# Source this file on IvI before running the experiment launchers:
#   source experiments/env_ivi.sh
#
# This variant intentionally keeps Python environment setup minimal.
# If you need module loads or a specific conda env, activate them before
# sourcing this file, or export SUPERNODE_TOKENIZER_CONDA_PREFIX.

_snt_env_fail() {
  printf '%s\n' "$1" >&2
  return 1 2>/dev/null || exit 1
}

SUPERNODE_TOKENIZER_REPO_ROOT="/home/rvalper/supernode-tokenizer"
SUPERNODE_TOKENIZER_CACHE_ROOT="/ivi/zfs/s0/original_homes/rvalper/robotics/rlbench/rlbench18/.rlbench_cache_dense_18"
SUPERNODE_TOKENIZER_OUTPUT_ROOT="$SUPERNODE_TOKENIZER_REPO_ROOT/output"
SUPERNODE_TOKENIZER_CHECKPOINT_ROOT="/ivi/zfs/s0/original_homes/rvalper/robotics/rlbench/rlbench18/checkpoints"
SUPERNODE_TOKENIZER_EVAL_ROOT="$SUPERNODE_TOKENIZER_REPO_ROOT/eval_output"

export SUPERNODE_TOKENIZER_REPO_ROOT
export SUPERNODE_TOKENIZER_CACHE_ROOT
export SUPERNODE_TOKENIZER_OUTPUT_ROOT
export SUPERNODE_TOKENIZER_CHECKPOINT_ROOT
export SUPERNODE_TOKENIZER_EVAL_ROOT

if [ ! -d "$SUPERNODE_TOKENIZER_REPO_ROOT" ]; then
  _snt_env_fail "[supernode-tokenizer] repo root not found: $SUPERNODE_TOKENIZER_REPO_ROOT"
  return 1 2>/dev/null || exit 1
fi

if [ -n "${SUPERNODE_TOKENIZER_CONDA_PREFIX:-}" ]; then
  case ":$PATH:" in
    *":$SUPERNODE_TOKENIZER_CONDA_PREFIX/bin:"*) ;;
    *) PATH="$SUPERNODE_TOKENIZER_CONDA_PREFIX/bin:$PATH" ;;
  esac
  export PATH
  export CONDA_PREFIX="$SUPERNODE_TOKENIZER_CONDA_PREFIX"
  if [ -z "${CONDA_DEFAULT_ENV:-}" ]; then
    export CONDA_DEFAULT_ENV="$(basename "$SUPERNODE_TOKENIZER_CONDA_PREFIX")"
  fi
fi

: "${WANDB_PROJECT:=supernode-tokenizer}"
: "${WANDB_MODE:=online}"
export WANDB_PROJECT
export WANDB_MODE
if [ -n "${WANDB_ENTITY:-}" ]; then
  export WANDB_ENTITY
fi

printf '%s\n' '[supernode-tokenizer] IvI environment configured:'
printf '  %s=%s\n' "SUPERNODE_TOKENIZER_REPO_ROOT" "$SUPERNODE_TOKENIZER_REPO_ROOT"
printf '  %s=%s\n' "SUPERNODE_TOKENIZER_CACHE_ROOT" "$SUPERNODE_TOKENIZER_CACHE_ROOT"
printf '  %s=%s\n' "SUPERNODE_TOKENIZER_OUTPUT_ROOT" "$SUPERNODE_TOKENIZER_OUTPUT_ROOT"
printf '  %s=%s\n' "SUPERNODE_TOKENIZER_CHECKPOINT_ROOT" "$SUPERNODE_TOKENIZER_CHECKPOINT_ROOT"
printf '  %s=%s\n' "SUPERNODE_TOKENIZER_EVAL_ROOT" "$SUPERNODE_TOKENIZER_EVAL_ROOT"
printf '  %s=%s\n' "WANDB_PROJECT" "$WANDB_PROJECT"
printf '  %s=%s\n' "WANDB_MODE" "$WANDB_MODE"
if [ -n "${WANDB_ENTITY:-}" ]; then
  printf '  %s=%s\n' "WANDB_ENTITY" "$WANDB_ENTITY"
fi
if [ -n "${SUPERNODE_TOKENIZER_CONDA_PREFIX:-}" ]; then
  printf '  %s=%s\n' "SUPERNODE_TOKENIZER_CONDA_PREFIX" "$SUPERNODE_TOKENIZER_CONDA_PREFIX"
fi
printf '  %s=%s\n' "PYTHON" "${PYTHON:-$(command -v python 2>/dev/null || printf '<not found>')}"

unset -f _snt_env_fail
