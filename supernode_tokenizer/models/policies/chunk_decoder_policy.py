from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..condition import TaskConditioner
from ..encoders import ObservationEncoder
from ..common import TaskConditionedCrossAttentionBlock


@dataclass
class ChunkDecoderConfig:
    d_model: int = 512
    n_heads: int = 8
    n_layers: int = 8
    mlp_mult: int = 4
    dropout: float = 0.0
    horizon: int = 16
    loss_type: str = "l1"


class ChunkDecoderPolicy(nn.Module):
    def __init__(
        self,
        *,
        cfg: ChunkDecoderConfig,
        encoder: ObservationEncoder,
        task_conditioner: TaskConditioner,
        action_dim: int,
    ):
        super().__init__()
        self.cfg = cfg
        self.encoder = encoder
        self.task_conditioner = task_conditioner
        self.action_dim = int(action_dim)
        d = int(cfg.d_model)
        self.action_queries = nn.Parameter(torch.randn(int(cfg.horizon), d) * 0.02)
        self.action_slot_embed = nn.Parameter(torch.randn(int(cfg.horizon), d) * 0.02)
        self.decoder = nn.ModuleList(
            [
                TaskConditionedCrossAttentionBlock(
                    d=d,
                    n_heads=int(cfg.n_heads),
                    cond_dim=d,
                    mlp_mult=int(cfg.mlp_mult),
                    dropout=float(cfg.dropout),
                )
                for _ in range(int(cfg.n_layers))
            ]
        )
        self.out = nn.Linear(d, int(action_dim))

    def _build_memory(
        self,
        *,
        task_ids: torch.Tensor,
        obs_xyz: torch.Tensor,
        obs_state: torch.Tensor,
        obs_valid: Optional[torch.Tensor],
        obs_rgb: Optional[torch.Tensor],
        obs_mask_id: Optional[torch.Tensor],
        return_debug: bool = False,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, Optional[Dict[str, Any]]]:
        task_emb, task_tokens = self.task_conditioner(task_ids)
        enc_out = self.encoder(
            obs_xyz=obs_xyz,
            obs_state=obs_state,
            task_emb=task_emb,
            obs_valid=obs_valid,
            obs_rgb=obs_rgb,
            obs_mask_id=obs_mask_id,
            return_debug=return_debug,
        )
        memory = torch.cat([enc_out.tokens, task_tokens], dim=1)
        memory_mask = None
        if enc_out.token_mask is not None:
            task_mask = torch.ones(task_tokens.shape[:2], device=task_tokens.device, dtype=torch.bool)
            memory_mask = torch.cat([enc_out.token_mask.to(torch.bool), task_mask], dim=1)
        return memory, memory_mask, task_emb, enc_out.debug

    def forward(
        self,
        *,
        task_ids: torch.Tensor,
        obs_xyz: torch.Tensor,
        obs_state: torch.Tensor,
        obs_valid: Optional[torch.Tensor] = None,
        obs_rgb: Optional[torch.Tensor] = None,
        obs_mask_id: Optional[torch.Tensor] = None,
        return_debug: bool = False,
    ) -> tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        memory, memory_mask, task_emb, debug = self._build_memory(
            task_ids=task_ids,
            obs_xyz=obs_xyz,
            obs_state=obs_state,
            obs_valid=obs_valid,
            obs_rgb=obs_rgb,
            obs_mask_id=obs_mask_id,
            return_debug=return_debug,
        )
        batch = int(task_ids.shape[0])
        h = self.action_queries.unsqueeze(0).expand(batch, -1, -1) + self.action_slot_embed.unsqueeze(0)
        for blk in self.decoder:
            h = blk(h, task_emb, memory, memory_mask)
        return self.out(h), debug

    def compute_loss(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        pred, _ = self(
            task_ids=batch["task_id"],
            obs_xyz=batch["obs_xyz"],
            obs_state=batch["obs_state"],
            obs_valid=batch.get("obs_valid", None),
            obs_rgb=batch.get("obs_rgb", None),
            obs_mask_id=batch.get("obs_mask_id", None),
            return_debug=False,
        )
        target = batch["target_action"]
        if str(self.cfg.loss_type).lower() == "mse":
            loss = F.mse_loss(pred, target)
        else:
            loss = F.l1_loss(pred, target)
        mse = F.mse_loss(pred, target)
        return {"loss": loss, "l1": F.l1_loss(pred, target), "mse": mse, "pred_action": pred}

    @torch.no_grad()
    def sample_actions(
        self,
        *,
        task_ids: torch.Tensor,
        obs_xyz: torch.Tensor,
        obs_state: torch.Tensor,
        obs_valid: Optional[torch.Tensor] = None,
        obs_rgb: Optional[torch.Tensor] = None,
        obs_mask_id: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        pred, _ = self(
            task_ids=task_ids,
            obs_xyz=obs_xyz,
            obs_state=obs_state,
            obs_valid=obs_valid,
            obs_rgb=obs_rgb,
            obs_mask_id=obs_mask_id,
            return_debug=False,
        )
        return pred
