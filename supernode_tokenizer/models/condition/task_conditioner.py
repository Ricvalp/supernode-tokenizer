from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass(frozen=True)
class TaskConditionerConfig:
    num_tasks: int
    d_model: int = 512
    n_task_tokens: int = 4
    dropout: float = 0.0


class TaskConditioner(nn.Module):
    def __init__(self, cfg: TaskConditionerConfig):
        super().__init__()
        self.cfg = cfg
        self.task_embed = nn.Embedding(int(cfg.num_tasks), int(cfg.d_model))
        self.task_tokens = nn.Parameter(torch.randn(int(cfg.n_task_tokens), int(cfg.d_model)) * 0.02)
        self.proj = nn.Sequential(
            nn.Linear(int(cfg.d_model), int(cfg.d_model)),
            nn.SiLU(),
            nn.Linear(int(cfg.d_model), int(cfg.d_model)),
        )
        self.drop = nn.Dropout(float(cfg.dropout))

    def forward(self, task_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        task_ids = task_ids.to(torch.long)
        task_emb = self.proj(self.task_embed(task_ids))
        task_tokens = self.task_tokens.unsqueeze(0).expand(int(task_ids.shape[0]), -1, -1)
        task_tokens = self.drop(task_tokens + task_emb.unsqueeze(1))
        return task_emb, task_tokens
