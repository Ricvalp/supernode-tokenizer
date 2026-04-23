#!/bin/sh

# Source this file before running the experiment launchers:
#   source experiments/env.sh
#
# The defaults below are real values, not placeholders. Edit them here if your
# cache or output locations should live outside the repo checkout.

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
if [ -z "$_SNT_REPO_ROOT" ] && [ -f "${PWD:-}/experiments/env.sh" ]; then
  _SNT_REPO_ROOT="$PWD"
fi
if [ -z "$_SNT_REPO_ROOT" ] && [ -n "${SUPERNODE_TOKENIZER_REPO_ROOT:-}" ]; then
  _SNT_REPO_ROOT="$SUPERNODE_TOKENIZER_REPO_ROOT"
fi
if [ -z "$_SNT_REPO_ROOT" ]; then
  _snt_env_fail "[supernode-tokenizer] unable to determine repo root while sourcing experiments/env.sh"
fi

export SUPERNODE_TOKENIZER_REPO_ROOT="$_SNT_REPO_ROOT"

SUPERNODE_TOKENIZER_CACHE_ROOT="/mnt/external_storage/robotics/rlbench/rlbench18/.rlbench_cache_dense_18"
SUPERNODE_TOKENIZER_OUTPUT_ROOT="$SUPERNODE_TOKENIZER_REPO_ROOT/output"
SUPERNODE_TOKENIZER_CHECKPOINT_ROOT="$SUPERNODE_TOKENIZER_REPO_ROOT/checkpoints"
SUPERNODE_TOKENIZER_EVAL_ROOT="$SUPERNODE_TOKENIZER_REPO_ROOT/eval_output"

export SUPERNODE_TOKENIZER_CACHE_ROOT
export SUPERNODE_TOKENIZER_OUTPUT_ROOT
export SUPERNODE_TOKENIZER_CHECKPOINT_ROOT
export SUPERNODE_TOKENIZER_EVAL_ROOT

# Optional logging defaults. Override these before sourcing if you want
# machine-specific values.
: "${WANDB_PROJECT:=supernode-tokenizer}"
: "${WANDB_MODE:=online}"
export WANDB_PROJECT
export WANDB_MODE
if [ -n "${WANDB_ENTITY:-}" ]; then
  export WANDB_ENTITY
fi

# RLBench / PyRep / CoppeliaSim helpers. These are only needed for raw dataset
# generation and live rollout evaluation / robustness experiments.
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

# Uncomment if your local PyRep/CoppeliaSim setup requires an X server:
# export DISPLAY=:99

printf '%s\n' '[supernode-tokenizer] environment configured:'
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

unset _SNT_ENV_SOURCE
unset _SNT_ENV_DIR
unset _SNT_REPO_ROOT
unset -f _snt_env_fail
