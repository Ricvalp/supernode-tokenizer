from __future__ import annotations

from typing import Iterable, Tuple

import torch


def count_parameters(module: torch.nn.Module) -> Tuple[int, int]:
    total = sum(int(p.numel()) for p in module.parameters())
    trainable = sum(int(p.numel()) for p in module.parameters() if p.requires_grad)
    return total, trainable


def grad_norm(parameters: Iterable[torch.nn.Parameter]) -> float:
    norms = []
    for param in parameters:
        if param.grad is None:
            continue
        norms.append(param.grad.detach().norm(2))
    if not norms:
        return 0.0
    return float(torch.norm(torch.stack(norms), 2).item())
