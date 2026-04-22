from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from ..common import TaskConditionedFramePerceiverTokenizer, TaskConditionedSelfAttentionBlock
from .base import ObservationEncoder, ObservationEncoderOutput


@dataclass
class ObservationPerceiverEncoderConfig:
    d_model: int = 512
    n_heads: int = 8
    m_frame_tokens: int = 128
    frame_tokenizer_layers: int = 2
    post_self_attn_layers: int = 2
    post_self_attn_mlp_mult: int = 4
    dropout: float = 0.0
    attention_backend: str = "manual"
    rgb_alpha_init: float = 1.0
    use_gripper_point_features: bool = True
    gripper_xyz_state_start: int = 0
    gripper_alpha_init: float = 1.0
    tokenize_frames_chunked: bool = True
    chunk_frames: int = 64


class _PointFeatureProjector(nn.Module):
    def __init__(self, *, d_model: int, state_dim: int, cfg: ObservationPerceiverEncoderConfig):
        super().__init__()
        self.xyz_proj = nn.Linear(3, int(d_model), bias=False)
        self.rgb_proj = nn.Linear(3, int(d_model), bias=False)
        self.rgb_alpha = nn.Parameter(torch.tensor(float(cfg.rgb_alpha_init), dtype=torch.float32))
        self.state_proj = nn.Sequential(
            nn.Linear(int(state_dim), int(d_model)),
            nn.SiLU(),
            nn.Linear(int(d_model), int(d_model)),
        )
        self.use_gripper_point_features = bool(cfg.use_gripper_point_features)
        self.gripper_xyz_state_start = int(cfg.gripper_xyz_state_start)
        self.gripper_proj = nn.Linear(4, int(d_model), bias=False) if self.use_gripper_point_features else None
        self.gripper_alpha = (
            nn.Parameter(torch.tensor(float(cfg.gripper_alpha_init), dtype=torch.float32))
            if self.use_gripper_point_features
            else None
        )

    def _gripper_xyz(self, state: torch.Tensor) -> torch.Tensor:
        start = int(self.gripper_xyz_state_start)
        end = start + 3
        if int(state.shape[-1]) < end:
            raise ValueError(
                f"state_dim={int(state.shape[-1])} is too small for gripper_xyz_state_start={start}."
            )
        return state[..., start:end]

    def forward(
        self,
        *,
        xyz: torch.Tensor,
        state: torch.Tensor,
        rgb: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.xyz_proj(xyz)
        if rgb is not None:
            h = h + self.rgb_alpha.to(dtype=h.dtype) * self.rgb_proj(rgb.to(dtype=xyz.dtype))
        if self.use_gripper_point_features:
            if self.gripper_proj is None or self.gripper_alpha is None:
                raise RuntimeError("Gripper point feature modules are not initialized.")
            gripper_xyz = self._gripper_xyz(state).to(device=xyz.device, dtype=xyz.dtype)
            rel = xyz - gripper_xyz.unsqueeze(1)
            dist = torch.norm(rel, dim=-1, keepdim=True)
            grip_feat = torch.cat([rel, dist], dim=-1)
            h = h + self.gripper_alpha.to(dtype=h.dtype) * self.gripper_proj(grip_feat)
        state_token = self.state_proj(state).unsqueeze(1)
        return h, state_token


class ObservationPerceiverEncoder(ObservationEncoder):
    def __init__(self, *, cfg: ObservationPerceiverEncoderConfig, state_dim: int):
        super().__init__()
        self.cfg = cfg
        self.d_model = int(cfg.d_model)
        self.point_projector = _PointFeatureProjector(d_model=self.d_model, state_dim=int(state_dim), cfg=cfg)
        self.frame_tokenizer = TaskConditionedFramePerceiverTokenizer(
            d=int(cfg.d_model),
            m=int(cfg.m_frame_tokens),
            n_heads=int(cfg.n_heads),
            cond_dim=int(cfg.d_model),
            n_layers=int(cfg.frame_tokenizer_layers),
            dropout=float(cfg.dropout),
            attention_backend=str(cfg.attention_backend),
        )
        self.time_embed = nn.Embedding(32, int(cfg.d_model))
        self.post_refiner = nn.ModuleList(
            [
                TaskConditionedSelfAttentionBlock(
                    d=int(cfg.d_model),
                    n_heads=int(cfg.n_heads),
                    cond_dim=int(cfg.d_model),
                    mlp_mult=int(cfg.post_self_attn_mlp_mult),
                    dropout=float(cfg.dropout),
                    attention_backend=str(cfg.attention_backend),
                )
                for _ in range(int(cfg.post_self_attn_layers))
            ]
        )

    def _tokenize_frames(
        self,
        *,
        obs_xyz_f: torch.Tensor,
        obs_state_f: torch.Tensor,
        task_emb_f: torch.Tensor,
        obs_valid_f: Optional[torch.Tensor],
        obs_rgb_f: Optional[torch.Tensor],
    ) -> torch.Tensor:
        point_tokens, state_token = self.point_projector(xyz=obs_xyz_f, state=obs_state_f, rgb=obs_rgb_f)
        mask = obs_valid_f.to(torch.bool) if obs_valid_f is not None else None
        visual_tokens = self.frame_tokenizer(point_tokens, task_emb_f, point_mask=mask)
        return torch.cat([visual_tokens, state_token], dim=1)

    def forward(
        self,
        *,
        obs_xyz: torch.Tensor,
        obs_state: torch.Tensor,
        task_emb: torch.Tensor,
        obs_valid: Optional[torch.Tensor] = None,
        obs_rgb: Optional[torch.Tensor] = None,
        obs_mask_id: Optional[torch.Tensor] = None,
        return_debug: bool = False,
    ) -> ObservationEncoderOutput:
        del obs_mask_id, return_debug
        batch, t_obs, n_points, _ = obs_xyz.shape
        xyz_f = obs_xyz.reshape(batch * t_obs, n_points, 3)
        state_f = obs_state.reshape(batch * t_obs, -1)
        valid_f = obs_valid.reshape(batch * t_obs, n_points).to(torch.bool) if obs_valid is not None else None
        rgb_f = obs_rgb.reshape(batch * t_obs, n_points, 3) if obs_rgb is not None else None
        task_emb_f = task_emb.unsqueeze(1).expand(batch, t_obs, -1).reshape(batch * t_obs, -1)

        if bool(self.cfg.tokenize_frames_chunked):
            chunks = []
            chunk_frames = max(1, int(self.cfg.chunk_frames))
            for start in range(0, batch * t_obs, chunk_frames):
                end = min(batch * t_obs, start + chunk_frames)
                chunks.append(
                    self._tokenize_frames(
                        obs_xyz_f=xyz_f[start:end],
                        obs_state_f=state_f[start:end],
                        task_emb_f=task_emb_f[start:end],
                        obs_valid_f=None if valid_f is None else valid_f[start:end],
                        obs_rgb_f=None if rgb_f is None else rgb_f[start:end],
                    )
                )
            frame_tokens = torch.cat(chunks, dim=0)
        else:
            frame_tokens = self._tokenize_frames(
                obs_xyz_f=xyz_f,
                obs_state_f=state_f,
                task_emb_f=task_emb_f,
                obs_valid_f=valid_f,
                obs_rgb_f=rgb_f,
            )

        tokens_per_frame = int(frame_tokens.shape[1])
        frame_tokens = frame_tokens.reshape(batch, t_obs, tokens_per_frame, self.d_model)
        if t_obs > self.time_embed.num_embeddings:
            raise ValueError(f"T_obs={t_obs} exceeds max time embeddings={self.time_embed.num_embeddings}.")
        time_ids = torch.arange(t_obs, device=obs_xyz.device).clamp_max(self.time_embed.num_embeddings - 1)
        frame_tokens = frame_tokens + self.time_embed(time_ids).view(1, t_obs, 1, self.d_model)
        tokens = frame_tokens.reshape(batch, t_obs * tokens_per_frame, self.d_model)
        for layer in self.post_refiner:
            tokens = layer(tokens, task_emb)
        return ObservationEncoderOutput(tokens=tokens)
