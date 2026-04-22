from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import torch


def save_checkpoint(
    path: Path,
    *,
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: Optional[torch.cuda.amp.GradScaler],
    config: Dict[str, Any],
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "step": int(step),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": config,
    }
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()
    if extra:
        payload.update(extra)
    torch.save(payload, path)


def load_checkpoint(path: Path, device: torch.device) -> Dict[str, Any]:
    return torch.load(path, map_location=device)


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)
