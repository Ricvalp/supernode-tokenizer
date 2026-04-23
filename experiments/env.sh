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
  return 1 2>/dev/null || exit 1
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

# Keep logs unbuffered so long RLBench eval runs flush output promptly.
: "${PYTHONUNBUFFERED:=1}"
export PYTHONUNBUFFERED

if [ -z "${XAUTHORITY:-}" ]; then
  case "${DISPLAY:-}" in
    localhost:*|127.0.0.1:*|*:*.*)
      if [ -f "$HOME/.Xauthority" ]; then
        XAUTHORITY="$HOME/.Xauthority"
      fi
      ;;
    :*)
      _SNT_UID="$(id -u 2>/dev/null || echo '')"
      if [ -n "$_SNT_UID" ] && [ -f "/run/user/$_SNT_UID/gdm/Xauthority" ]; then
        XAUTHORITY="/run/user/$_SNT_UID/gdm/Xauthority"
      elif [ -f "$HOME/.Xauthority" ]; then
        XAUTHORITY="$HOME/.Xauthority"
      fi
      unset _SNT_UID
      ;;
    *)
      if [ -f "$HOME/.Xauthority" ]; then
        XAUTHORITY="$HOME/.Xauthority"
      fi
      ;;
  esac
  if [ -n "${XAUTHORITY:-}" ]; then
    export XAUTHORITY
  fi
fi

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
  QT_QPA_PLATFORM_PLUGIN_PATH="$COPPELIASIM_ROOT"
  QT_QPA_PLATFORM="xcb"
  QT_XCB_GL_INTEGRATION="xcb_glx"
  __GLX_VENDOR_LIBRARY_NAME="nvidia"
  : "${DISPLAY:=:99}"
  export QT_QPA_PLATFORM_PLUGIN_PATH
  export QT_QPA_PLATFORM
  export QT_XCB_GL_INTEGRATION
  export __GLX_VENDOR_LIBRARY_NAME
  export DISPLAY
fi

printf '%s\n' '[supernode-tokenizer] environment configured:'
printf '  %s=%s\n' "SUPERNODE_TOKENIZER_REPO_ROOT" "$SUPERNODE_TOKENIZER_REPO_ROOT"
printf '  %s=%s\n' "SUPERNODE_TOKENIZER_CACHE_ROOT" "$SUPERNODE_TOKENIZER_CACHE_ROOT"
printf '  %s=%s\n' "SUPERNODE_TOKENIZER_OUTPUT_ROOT" "$SUPERNODE_TOKENIZER_OUTPUT_ROOT"
printf '  %s=%s\n' "SUPERNODE_TOKENIZER_CHECKPOINT_ROOT" "$SUPERNODE_TOKENIZER_CHECKPOINT_ROOT"
printf '  %s=%s\n' "SUPERNODE_TOKENIZER_EVAL_ROOT" "$SUPERNODE_TOKENIZER_EVAL_ROOT"
printf '  %s=%s\n' "PYTHONUNBUFFERED" "$PYTHONUNBUFFERED"
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
if [ -n "${QT_XCB_GL_INTEGRATION:-}" ]; then
  printf '  %s=%s\n' "QT_XCB_GL_INTEGRATION" "$QT_XCB_GL_INTEGRATION"
fi
if [ -n "${__GLX_VENDOR_LIBRARY_NAME:-}" ]; then
  printf '  %s=%s\n' "__GLX_VENDOR_LIBRARY_NAME" "$__GLX_VENDOR_LIBRARY_NAME"
fi
if [ -n "${DISPLAY:-}" ]; then
  printf '  %s=%s\n' "DISPLAY" "$DISPLAY"
fi
if [ -n "${XAUTHORITY:-}" ]; then
  printf '  %s=%s\n' "XAUTHORITY" "$XAUTHORITY"
fi

unset _SNT_ENV_SOURCE
unset _SNT_ENV_DIR
unset _SNT_REPO_ROOT
unset -f _snt_env_fail
