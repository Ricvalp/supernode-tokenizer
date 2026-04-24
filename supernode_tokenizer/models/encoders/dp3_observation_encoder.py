from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn


@dataclass
class DP3ObservationEncoderConfig:
    use_rgb: bool = False
    point_hidden_dim1: int = 64
    point_hidden_dim2: int = 128
    point_hidden_dim3: int = 256
    point_feature_dim: int = 64
    state_hidden_dim: int = 64
    state_feature_dim: int = 64


class DP3ObservationEncoder(nn.Module):
    def __init__(self, *, cfg: DP3ObservationEncoderConfig, state_dim: int):
        super().__init__()
        self.cfg = cfg
        point_channels = 6 if bool(cfg.use_rgb) else 3
        self.point_mlp = nn.Sequential(
            nn.Linear(point_channels, int(cfg.point_hidden_dim1)),
            nn.LayerNorm(int(cfg.point_hidden_dim1)),
            nn.ReLU(),
            nn.Linear(int(cfg.point_hidden_dim1), int(cfg.point_hidden_dim2)),
            nn.LayerNorm(int(cfg.point_hidden_dim2)),
            nn.ReLU(),
            nn.Linear(int(cfg.point_hidden_dim2), int(cfg.point_hidden_dim3)),
            nn.LayerNorm(int(cfg.point_hidden_dim3)),
            nn.ReLU(),
        )
        self.point_proj = nn.Sequential(
            nn.Linear(int(cfg.point_hidden_dim3), int(cfg.point_feature_dim)),
            nn.LayerNorm(int(cfg.point_feature_dim)),
        )
        self.state_mlp = nn.Sequential(
            nn.Linear(int(state_dim), int(cfg.state_hidden_dim)),
            nn.ReLU(),
            nn.Linear(int(cfg.state_hidden_dim), int(cfg.state_feature_dim)),
        )
        self.output_dim = int(cfg.point_feature_dim) + int(cfg.state_feature_dim)

    def forward(
        self,
        *,
        xyz: torch.Tensor,
        state: torch.Tensor,
        rgb: Optional[torch.Tensor] = None,
        valid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if bool(self.cfg.use_rgb):
            if rgb is None:
                raise ValueError("DP3ObservationEncoder is configured with use_rgb=True but rgb was not provided.")
            point_input = torch.cat([xyz, rgb.to(device=xyz.device, dtype=xyz.dtype)], dim=-1)
        else:
            point_input = xyz
        point_feat = self.point_mlp(point_input)
        if valid is not None:
            point_feat = point_feat.masked_fill(~valid.to(torch.bool).unsqueeze(-1), float("-inf"))
        pooled = point_feat.max(dim=1).values
        pooled = torch.where(torch.isfinite(pooled), pooled, torch.zeros_like(pooled))
        pc_feat = self.point_proj(pooled)
        state_feat = self.state_mlp(state)
        return torch.cat([pc_feat, state_feat], dim=-1)
