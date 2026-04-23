#!/bin/sh

# Source this file on Snellius before running the experiment launchers:
#   source experiments/env_snellius.sh

_snt_env_fail() {
  printf '%s\n' "$1" >&2
  return 1 2>/dev/null || exit 1
}

if [ -n "${ZSH_VERSION:-}" ]; then
  _SNT_ENV_SOURCE="${(%):-%N}"
elif [ -n "${BASH_VERSION:-}" ]; then
  _SNT_ENV_SOURCE="${BASH_SOURCE[0]}"
else
  _SNT_ENV_SOURCE=""
fi

_SNT_ENV_DIR=""
if [ -n "$_SNT_ENV_SOURCE" ]; then
  _SNT_ENV_DIR=$(CDPATH= cd -- "$(dirname -- "$_SNT_ENV_SOURCE")" 2>/dev/null && pwd)
fi

_SNT_REPO_ROOT=""
if [ -n "$_SNT_ENV_DIR" ]; then
  _SNT_REPO_ROOT=$(CDPATH= cd -- "$_SNT_ENV_DIR/.." 2>/dev/null && pwd)
fi
if [ -z "$_SNT_REPO_ROOT" ] && command -v git >/dev/null 2>&1; then
  _SNT_REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null || true)
fi
if [ -z "$_SNT_REPO_ROOT" ] && [ -f "${PWD:-}/experiments/env_snellius.sh" ]; then
  _SNT_REPO_ROOT="$PWD"
fi
if [ -z "$_SNT_REPO_ROOT" ] && [ -n "${SUPERNODE_TOKENIZER_REPO_ROOT:-}" ]; then
  _SNT_REPO_ROOT="$SUPERNODE_TOKENIZER_REPO_ROOT"
fi
if [ -z "$_SNT_REPO_ROOT" ]; then
  _snt_env_fail "[supernode-tokenizer] unable to determine repo root while sourcing experiments/env_snellius.sh"
  return 1 2>/dev/null || exit 1
fi

export SUPERNODE_TOKENIZER_REPO_ROOT="$_SNT_REPO_ROOT"

SUPERNODE_TOKENIZER_CACHE_ROOT="/projects/prjs1905/robotics/rlbench/rlbench18/.rlbench_cache_dense_18/"
SUPERNODE_TOKENIZER_OUTPUT_ROOT="$SUPERNODE_TOKENIZER_REPO_ROOT/output"
SUPERNODE_TOKENIZER_CHECKPOINT_ROOT="/projects/prjs1905/robotics/rlbench/rlbench18/checkpoints/"
SUPERNODE_TOKENIZER_EVAL_ROOT="$SUPERNODE_TOKENIZER_REPO_ROOT/eval_output"

export SUPERNODE_TOKENIZER_CACHE_ROOT
export SUPERNODE_TOKENIZER_OUTPUT_ROOT
export SUPERNODE_TOKENIZER_CHECKPOINT_ROOT
export SUPERNODE_TOKENIZER_EVAL_ROOT

: "${WANDB_PROJECT:=supernode-tokenizer}"
: "${WANDB_MODE:=online}"
export WANDB_PROJECT
export WANDB_MODE
if [ -n "${WANDB_ENTITY:-}" ]; then
  export WANDB_ENTITY
fi

printf '%s\n' '[supernode-tokenizer] Snellius environment configured:'
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

unset _SNT_ENV_SOURCE
unset _SNT_ENV_DIR
unset _SNT_REPO_ROOT
unset -f _snt_env_fail
