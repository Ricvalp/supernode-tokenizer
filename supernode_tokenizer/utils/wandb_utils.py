from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional


def init_wandb(cfg: Any, *, run_dir: Path, config_dict: Dict[str, Any]):
    if not bool(getattr(cfg, "enable", False)):
        return None
    import wandb

    run = wandb.init(
        project=str(getattr(cfg, "project", "supernode-tokenizer")),
        entity=str(getattr(cfg, "entity", "")) or None,
        mode=str(getattr(cfg, "mode", "online")),
        name=str(getattr(cfg, "name", "")) or None,
        dir=str(run_dir),
        config=config_dict,
    )
    return run


def log_wandb(run: Any, payload: Dict[str, Any], step: int) -> None:
    if run is None:
        return
    run.log(payload, step=int(step))


def finish_wandb(run: Any) -> None:
    if run is not None:
        run.finish()
