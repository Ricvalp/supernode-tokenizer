#!/bin/sh

# Source this file on DAS before running the experiment launchers:
#   source experiments/env_das.sh

_snt_env_fail() {
  printf '%s\n' "$1" >&2
  return 1 2>/dev/null || exit 1
}

SUPERNODE_TOKENIZER_REPO_ROOT="/home/valperga/supernode-tokenizer"
SUPERNODE_TOKENIZER_CACHE_ROOT="/var/scratch/valperga/robotics/rlbench/rlbench18/.rlbench_cache_dense_18"
SUPERNODE_TOKENIZER_OUTPUT_ROOT="$SUPERNODE_TOKENIZER_REPO_ROOT/output"
SUPERNODE_TOKENIZER_CHECKPOINT_ROOT="/var/scratch/valperga/robotics/rlbench/rlbench18/checkpoints"
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

: "${WANDB_PROJECT:=supernode-tokenizer}"
: "${WANDB_MODE:=online}"
export WANDB_PROJECT
export WANDB_MODE
if [ -n "${WANDB_ENTITY:-}" ]; then
  export WANDB_ENTITY
fi

if [ -z "${COPPELIASIM_ROOT:-}" ] && [ -d "$HOME/CoppeliaSim" ]; then
  COPPELIASIM_ROOT="$HOME/CoppeliaSim"
fi
if [ -n "${COPPELIASIM_ROOT:-}" ] && [ -d "$COPPELIASIM_ROOT" ]; then
  export COPPELIASIM_ROOT
  case ":${LD_LIBRARY_PATH:-}:" in
    *":$COPPELIASIM_ROOT:"*) ;;
    *) LD_LIBRARY_PATH="$COPPELIASIM_ROOT${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" ;;
  esac
  export LD_LIBRARY_PATH
  : "${QT_QPA_PLATFORM_PLUGIN_PATH:=$COPPELIASIM_ROOT}"
  : "${QT_QPA_PLATFORM:=xcb}"
  export QT_QPA_PLATFORM_PLUGIN_PATH
  export QT_QPA_PLATFORM
fi

printf '%s\n' '[supernode-tokenizer] DAS environment configured:'
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
if [ -n "${COPPELIASIM_ROOT:-}" ]; then
  printf '  %s=%s\n' "COPPELIASIM_ROOT" "$COPPELIASIM_ROOT"
fi
if [ -n "${QT_QPA_PLATFORM_PLUGIN_PATH:-}" ]; then
  printf '  %s=%s\n' "QT_QPA_PLATFORM_PLUGIN_PATH" "$QT_QPA_PLATFORM_PLUGIN_PATH"
fi
if [ -n "${QT_QPA_PLATFORM:-}" ]; then
  printf '  %s=%s\n' "QT_QPA_PLATFORM" "$QT_QPA_PLATFORM"
fi
if [ -n "${DISPLAY:-}" ]; then
  printf '  %s=%s\n' "DISPLAY" "$DISPLAY"
fi

unset -f _snt_env_fail
