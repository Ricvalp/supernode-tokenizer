from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

from ..common import TaskConditionedSelfAttentionBlock
from .base import ObservationEncoder, ObservationEncoderOutput
from .dp3_observation_encoder import DP3ObservationEncoder, DP3ObservationEncoderConfig


@dataclass
class ObservationDP3EncoderConfig:
    d_model: int = 512
    use_rgb: bool = False
    fps_num_points: int = 1024
    point_hidden_dim1: int = 64
    point_hidden_dim2: int = 128
    point_hidden_dim3: int = 256
    point_feature_dim: int = 64
    state_hidden_dim: int = 64
    state_feature_dim: int = 64
    n_heads: int = 8
    post_self_attn_layers: int = 0
    post_self_attn_mlp_mult: int = 4
    dropout: float = 0.0
    attention_backend: str = "manual"
    time_embed_max: int = 32
    tokenize_frames_chunked: bool = True
    chunk_frames: int = 64


@torch.no_grad()
def _batched_fps_indices(
    xyz: torch.Tensor,
    valid: torch.Tensor,
    *,
    num_samples: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    batch, n_points, _ = xyz.shape
    if batch == 0:
        empty_idx = torch.zeros((0, int(num_samples)), device=xyz.device, dtype=torch.long)
        empty_valid = torch.zeros((0, int(num_samples)), device=xyz.device, dtype=torch.bool)
        return empty_idx, empty_valid
    valid = valid.to(torch.bool)
    if n_points == 0:
        idx = torch.zeros((batch, int(num_samples)), device=xyz.device, dtype=torch.long)
        sampled_valid = torch.zeros((batch, int(num_samples)), device=xyz.device, dtype=torch.bool)
        return idx, sampled_valid

    xyz_f = xyz.float()
    batch_idx = torch.arange(batch, device=xyz.device)
    any_valid = valid.any(dim=1)
    counts = valid.sum(dim=1).clamp_min(1).to(dtype=xyz_f.dtype)
    centroid = (xyz_f * valid.unsqueeze(-1).to(dtype=xyz_f.dtype)).sum(dim=1) / counts.unsqueeze(-1)
    first_dist = ((xyz_f - centroid.unsqueeze(1)) ** 2).sum(dim=-1)
    first_dist = first_dist.masked_fill(~valid, -1.0)
    first = torch.where(any_valid, first_dist.argmax(dim=1), torch.zeros(batch, device=xyz.device, dtype=torch.long))

    idx = torch.zeros((batch, int(num_samples)), device=xyz.device, dtype=torch.long)
    selected = torch.zeros((batch, n_points), device=xyz.device, dtype=torch.bool)
    min_dist = torch.full((batch, n_points), float("inf"), device=xyz.device, dtype=xyz_f.dtype)
    current = first

    for step in range(int(num_samples)):
        idx[:, step] = current
        selected[batch_idx, current] |= any_valid
        sel_xyz = xyz_f[batch_idx, current]
        dist = ((xyz_f - sel_xyz.unsqueeze(1)) ** 2).sum(dim=-1)
        dist = dist.masked_fill(~valid, float("inf"))
        min_dist = torch.minimum(min_dist, dist)
        if step + 1 >= int(num_samples):
            continue
        candidate = min_dist.masked_fill(selected | ~valid, -1.0)
        has_candidate = candidate.amax(dim=1) >= 0.0
        next_idx = candidate.argmax(dim=1)
        current = torch.where(has_candidate, next_idx, first)

    sampled_valid = any_valid.unsqueeze(1).expand(-1, int(num_samples))
    return idx, sampled_valid


def _gather_points(points: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    return points.gather(1, idx.unsqueeze(-1).expand(-1, -1, int(points.shape[-1])))


class ObservationDP3Encoder(ObservationEncoder):
    def __init__(self, *, cfg: ObservationDP3EncoderConfig, state_dim: int):
        super().__init__()
        self.cfg = cfg
        self.d_model = int(cfg.d_model)
        self.frame_encoder = DP3ObservationEncoder(
            cfg=DP3ObservationEncoderConfig(
                use_rgb=bool(cfg.use_rgb),
                point_hidden_dim1=int(cfg.point_hidden_dim1),
                point_hidden_dim2=int(cfg.point_hidden_dim2),
                point_hidden_dim3=int(cfg.point_hidden_dim3),
                point_feature_dim=int(cfg.point_feature_dim),
                state_hidden_dim=int(cfg.state_hidden_dim),
                state_feature_dim=int(cfg.state_feature_dim),
            ),
            state_dim=int(state_dim),
        )
        self.obs_proj = nn.Sequential(
            nn.Linear(int(self.frame_encoder.output_dim), int(cfg.d_model)),
            nn.LayerNorm(int(cfg.d_model)),
        )
        self.time_embed = nn.Embedding(int(cfg.time_embed_max), int(cfg.d_model))
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

    def _maybe_downsample(
        self,
        *,
        xyz: torch.Tensor,
        rgb: Optional[torch.Tensor],
        valid: torch.Tensor,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor]]:
        if int(xyz.shape[1]) <= int(self.cfg.fps_num_points):
            return xyz, rgb, valid, None
        sample_idx, sampled_valid = _batched_fps_indices(
            xyz,
            valid,
            num_samples=int(self.cfg.fps_num_points),
        )
        xyz_out = _gather_points(xyz, sample_idx)
        rgb_out = _gather_points(rgb, sample_idx) if rgb is not None else None
        return xyz_out, rgb_out, sampled_valid, sample_idx

    def _encode_frames(
        self,
        *,
        xyz_f: torch.Tensor,
        state_f: torch.Tensor,
        task_emb_f: torch.Tensor,
        valid_f: torch.Tensor,
        rgb_f: Optional[torch.Tensor],
        return_debug: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[Dict[str, Any]]]:
        xyz_f, rgb_f, valid_f, fps_idx = self._maybe_downsample(
            xyz=xyz_f,
            rgb=rgb_f,
            valid=valid_f,
        )
        obs_feat = self.frame_encoder(
            xyz=xyz_f,
            state=state_f,
            rgb=rgb_f,
            valid=valid_f,
        )
        obs_tokens = self.obs_proj(obs_feat) + task_emb_f
        frame_valid = valid_f.any(dim=1)
        debug = None
        if return_debug:
            debug = {
                "frame_valid": frame_valid.detach(),
            }
            if fps_idx is not None:
                debug["fps_idx"] = fps_idx.detach()
        return obs_tokens, frame_valid, debug

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
        del obs_mask_id
        batch, t_obs, n_points, _ = obs_xyz.shape
        xyz_f = obs_xyz.reshape(batch * t_obs, n_points, 3)
        state_f = obs_state.reshape(batch * t_obs, -1)
        valid_f = obs_valid.reshape(batch * t_obs, n_points).to(torch.bool) if obs_valid is not None else torch.ones(
            batch * t_obs, n_points, device=obs_xyz.device, dtype=torch.bool
        )
        rgb_f = obs_rgb.reshape(batch * t_obs, n_points, 3) if obs_rgb is not None else None
        task_emb_f = task_emb.unsqueeze(1).expand(batch, t_obs, -1).reshape(batch * t_obs, -1)

        if bool(self.cfg.tokenize_frames_chunked):
            chunk_frames = max(1, int(self.cfg.chunk_frames))
            token_chunks = []
            valid_chunks = []
            debug_chunks = []
            for start in range(0, batch * t_obs, chunk_frames):
                end = min(batch * t_obs, start + chunk_frames)
                tokens_chunk, frame_valid_chunk, debug_chunk = self._encode_frames(
                    xyz_f=xyz_f[start:end],
                    state_f=state_f[start:end],
                    task_emb_f=task_emb_f[start:end],
                    valid_f=valid_f[start:end],
                    rgb_f=None if rgb_f is None else rgb_f[start:end],
                    return_debug=return_debug,
                )
                token_chunks.append(tokens_chunk)
                valid_chunks.append(frame_valid_chunk)
                if return_debug and debug_chunk is not None:
                    debug_chunks.append(debug_chunk)
            frame_tokens = torch.cat(token_chunks, dim=0)
            frame_valid = torch.cat(valid_chunks, dim=0)
            debug = None
            if return_debug and debug_chunks:
                debug = {
                    "frame_valid": torch.cat([chunk["frame_valid"] for chunk in debug_chunks], dim=0).reshape(batch, t_obs),
                }
                if "fps_idx" in debug_chunks[0]:
                    debug["fps_idx"] = torch.cat([chunk["fps_idx"] for chunk in debug_chunks], dim=0).reshape(
                        batch, t_obs, -1
                    )
        else:
            frame_tokens, frame_valid, debug = self._encode_frames(
                xyz_f=xyz_f,
                state_f=state_f,
                task_emb_f=task_emb_f,
                valid_f=valid_f,
                rgb_f=rgb_f,
                return_debug=return_debug,
            )
            if debug is not None:
                debug["frame_valid"] = debug["frame_valid"].reshape(batch, t_obs)
                if "fps_idx" in debug:
                    debug["fps_idx"] = debug["fps_idx"].reshape(batch, t_obs, -1)

        frame_tokens = frame_tokens.reshape(batch, t_obs, self.d_model)
        frame_valid = frame_valid.reshape(batch, t_obs)
        if t_obs > self.time_embed.num_embeddings:
            raise ValueError(f"T_obs={t_obs} exceeds max time embeddings={self.time_embed.num_embeddings}.")
        time_ids = torch.arange(t_obs, device=obs_xyz.device).clamp_max(self.time_embed.num_embeddings - 1)
        tokens = frame_tokens + self.time_embed(time_ids).view(1, t_obs, self.d_model)
        for layer in self.post_refiner:
            tokens = layer(tokens, task_emb, mask=frame_valid)
        return ObservationEncoderOutput(tokens=tokens, token_mask=frame_valid, debug=debug)
