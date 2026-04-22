from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from ..common import (
    IdentityTaskFiLM,
    TaskConditionedFramePerceiverTokenizer,
    TaskConditionedSelfAttentionBlock,
    build_knn_neighbors,
    fast_quota_based_supernode_sampling,
    gather_neighbors,
    gather_points,
    quota_based_supernode_sampling,
)
from .base import ObservationEncoder, ObservationEncoderOutput


@dataclass
class ObservationSupernodeEncoderConfig:
    d_model: int = 512
    n_heads: int = 8
    frame_tokens_out: int = 128
    num_supernodes: int = 128
    neighbors_per_supernode: int = 32
    supernode_refine_layers: int = 2
    compress_supernodes: bool = True
    supernode_pool_layers: int = 1
    post_self_attn_layers: int = 2
    post_self_attn_mlp_mult: int = 4
    dropout: float = 0.0
    attention_backend: str = "manual"
    use_mask_id: bool = True
    use_mask_embedding: bool = False
    mask_hash_buckets: int = 2048
    use_mask_instance_quota: bool = True
    supernode_sampling_mode: str = "fast_random"
    min_mask_supernodes: int = 4
    min_gripper_supernodes: int = 2
    gripper_sampling_radius: float = 0.10
    use_gripper_point_features: bool = True
    gripper_xyz_state_start: int = 0
    gripper_alpha_init: float = 1.0
    rgb_alpha_init: float = 1.0
    tokenize_frames_chunked: bool = True
    chunk_frames: int = 64
    use_message_passing: bool = True


class _PointFeatureProjector(nn.Module):
    def __init__(self, *, d_model: int, state_dim: int, cfg: ObservationSupernodeEncoderConfig):
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
        self.mask_embed = (
            nn.Embedding(int(cfg.mask_hash_buckets), int(d_model))
            if bool(cfg.use_mask_embedding)
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

    def _hash_mask_ids(self, mask_id: torch.Tensor) -> torch.Tensor:
        if self.mask_embed is None:
            raise RuntimeError("mask embedding requested but mask_embed is not initialized")
        return torch.remainder(mask_id, self.mask_embed.num_embeddings)

    def forward(
        self,
        *,
        xyz: torch.Tensor,
        state: torch.Tensor,
        rgb: Optional[torch.Tensor],
        mask_id: Optional[torch.Tensor],
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
            h = h + self.gripper_alpha.to(dtype=h.dtype) * self.gripper_proj(torch.cat([rel, dist], dim=-1))
        if self.mask_embed is not None and mask_id is not None:
            h = h + self.mask_embed(self._hash_mask_ids(mask_id.to(torch.long)))
        state_token = self.state_proj(state).unsqueeze(1)
        return h, state_token


class _TaskConditionedPointToSupernodeMessagePassing(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.edge_mlp = nn.Sequential(
            nn.Linear(int(d_model) + 4, int(d_model)),
            nn.SiLU(),
            nn.Linear(int(d_model), int(d_model)),
        )
        self.task_film = IdentityTaskFiLM(int(d_model), int(d_model), use_layernorm=False)
        self.out_norm = nn.LayerNorm(int(d_model))

    def forward(
        self,
        *,
        point_feat: torch.Tensor,
        point_xyz: torch.Tensor,
        supernode_xyz: torch.Tensor,
        neighbor_idx: torch.Tensor,
        neighbor_mask: torch.Tensor,
        supernode_init_feat: torch.Tensor,
        task_emb: torch.Tensor,
    ) -> torch.Tensor:
        neigh_feat = gather_neighbors(point_feat, neighbor_idx)
        neigh_xyz = gather_neighbors(point_xyz, neighbor_idx)
        rel = neigh_xyz - supernode_xyz.unsqueeze(2)
        dist = torch.norm(rel, dim=-1, keepdim=True)
        msg = self.edge_mlp(torch.cat([neigh_feat, rel.to(dtype=neigh_feat.dtype), dist.to(dtype=neigh_feat.dtype)], dim=-1))
        mask_f = neighbor_mask.to(dtype=msg.dtype).unsqueeze(-1)
        agg = (msg * mask_f).sum(dim=2) / mask_f.sum(dim=2).clamp_min(1.0)
        agg = self.task_film(agg, task_emb)
        return self.out_norm(supernode_init_feat + agg)


class _SimpleNeighborhoodPooling(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.out_norm = nn.LayerNorm(int(d_model))
        self.task_film = IdentityTaskFiLM(int(d_model), int(d_model), use_layernorm=False)

    def forward(
        self,
        *,
        point_feat: torch.Tensor,
        neighbor_idx: torch.Tensor,
        neighbor_mask: torch.Tensor,
        supernode_init_feat: torch.Tensor,
        task_emb: torch.Tensor,
    ) -> torch.Tensor:
        neigh_feat = gather_neighbors(point_feat, neighbor_idx)
        mask_f = neighbor_mask.to(dtype=neigh_feat.dtype).unsqueeze(-1)
        agg = (neigh_feat * mask_f).sum(dim=2) / mask_f.sum(dim=2).clamp_min(1.0)
        agg = self.task_film(agg, task_emb)
        return self.out_norm(supernode_init_feat + agg)


class _TaskConditionedSupernodeFrameTokenizer(nn.Module):
    def __init__(self, *, cfg: ObservationSupernodeEncoderConfig, state_dim: int):
        super().__init__()
        self.cfg = cfg
        self.d_model = int(cfg.d_model)
        self.projector = _PointFeatureProjector(d_model=self.d_model, state_dim=int(state_dim), cfg=cfg)
        self.use_message_passing = bool(cfg.use_message_passing)
        self.message_passing = _TaskConditionedPointToSupernodeMessagePassing(self.d_model)
        self.simple_pool = _SimpleNeighborhoodPooling(self.d_model)
        self.refiner = nn.ModuleList(
            [
                TaskConditionedSelfAttentionBlock(
                    d=int(cfg.d_model),
                    n_heads=int(cfg.n_heads),
                    cond_dim=int(cfg.d_model),
                    mlp_mult=4,
                    dropout=float(cfg.dropout),
                    attention_backend=str(cfg.attention_backend),
                )
                for _ in range(int(cfg.supernode_refine_layers))
            ]
        )
        self.pool = (
            TaskConditionedFramePerceiverTokenizer(
                d=int(cfg.d_model),
                m=int(cfg.frame_tokens_out),
                n_heads=int(cfg.n_heads),
                cond_dim=int(cfg.d_model),
                n_layers=int(cfg.supernode_pool_layers),
                dropout=float(cfg.dropout),
                attention_backend=str(cfg.attention_backend),
            )
            if bool(cfg.compress_supernodes)
            else None
        )

    def forward(
        self,
        *,
        xyz: torch.Tensor,
        state: torch.Tensor,
        task_emb: torch.Tensor,
        valid: Optional[torch.Tensor],
        rgb: Optional[torch.Tensor],
        mask_id: Optional[torch.Tensor],
        return_diagnostics: bool = False,
    ) -> tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        valid = valid.to(torch.bool) if valid is not None else torch.ones(xyz.shape[:2], device=xyz.device, dtype=torch.bool)
        point_feat, state_token = self.projector(xyz=xyz, state=state, rgb=rgb, mask_id=mask_id)
        sampler = fast_quota_based_supernode_sampling if str(self.cfg.supernode_sampling_mode).lower() in {"fast", "fast_random"} else quota_based_supernode_sampling
        sample_idx, supernode_mask, sample_bucket = sampler(
            xyz,
            valid,
            num_supernodes=int(self.cfg.num_supernodes),
            state=state,
            mask_id=mask_id if bool(self.cfg.use_mask_id) else None,
            use_mask_instance_quota=bool(self.cfg.use_mask_instance_quota),
            min_mask_supernodes=int(self.cfg.min_mask_supernodes),
            min_gripper_supernodes=int(self.cfg.min_gripper_supernodes),
            gripper_xyz_state_start=int(self.cfg.gripper_xyz_state_start),
            gripper_sampling_radius=float(self.cfg.gripper_sampling_radius),
        )
        supernode_xyz = gather_points(xyz, sample_idx)
        supernode_init_feat = gather_points(point_feat, sample_idx)
        neighbor_idx, neighbor_mask = build_knn_neighbors(xyz, valid, supernode_xyz, k=int(self.cfg.neighbors_per_supernode))
        if self.use_message_passing:
            h_super = self.message_passing(
                point_feat=point_feat,
                point_xyz=xyz,
                supernode_xyz=supernode_xyz,
                neighbor_idx=neighbor_idx,
                neighbor_mask=neighbor_mask,
                supernode_init_feat=supernode_init_feat,
                task_emb=task_emb,
            )
        else:
            h_super = self.simple_pool(
                point_feat=point_feat,
                neighbor_idx=neighbor_idx,
                neighbor_mask=neighbor_mask,
                supernode_init_feat=supernode_init_feat,
                task_emb=task_emb,
            )
        for layer in self.refiner:
            h_super = layer(h_super, task_emb, mask=supernode_mask)
        visual_tokens = self.pool(h_super, task_emb, point_mask=supernode_mask) if self.pool is not None else h_super
        frame_tokens = torch.cat([visual_tokens, state_token], dim=1)
        debug = None
        if return_diagnostics:
            debug = {
                "supernode_idx": sample_idx.detach(),
                "supernode_xyz": supernode_xyz.detach(),
                "supernode_mask": supernode_mask.detach(),
                "sample_bucket": sample_bucket.detach(),
                "neighbor_idx": neighbor_idx.detach(),
                "neighbor_mask": neighbor_mask.detach(),
            }
        return frame_tokens, debug


class ObservationSupernodeEncoder(ObservationEncoder):
    def __init__(self, *, cfg: ObservationSupernodeEncoderConfig, state_dim: int):
        super().__init__()
        self.cfg = cfg
        self.d_model = int(cfg.d_model)
        self.frame_stack = _TaskConditionedSupernodeFrameTokenizer(cfg=cfg, state_dim=int(state_dim))
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
        xyz_f: torch.Tensor,
        state_f: torch.Tensor,
        task_emb_f: torch.Tensor,
        valid_f: Optional[torch.Tensor],
        rgb_f: Optional[torch.Tensor],
        mask_f: Optional[torch.Tensor],
        return_debug: bool,
    ) -> tuple[torch.Tensor, Optional[Dict[str, Any]]]:
        return self.frame_stack(
            xyz=xyz_f,
            state=state_f,
            task_emb=task_emb_f,
            valid=valid_f,
            rgb=rgb_f,
            mask_id=mask_f,
            return_diagnostics=return_debug,
        )

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
        batch, t_obs, n_points, _ = obs_xyz.shape
        xyz_f = obs_xyz.reshape(batch * t_obs, n_points, 3)
        state_f = obs_state.reshape(batch * t_obs, -1)
        valid_f = obs_valid.reshape(batch * t_obs, n_points).to(torch.bool) if obs_valid is not None else None
        rgb_f = obs_rgb.reshape(batch * t_obs, n_points, 3) if obs_rgb is not None else None
        mask_f = obs_mask_id.reshape(batch * t_obs, n_points) if obs_mask_id is not None else None
        task_emb_f = task_emb.unsqueeze(1).expand(batch, t_obs, -1).reshape(batch * t_obs, -1)

        debug = None
        if bool(self.cfg.tokenize_frames_chunked):
            chunks = []
            chunk_frames = max(1, int(self.cfg.chunk_frames))
            debug_chunks = []
            for start in range(0, batch * t_obs, chunk_frames):
                end = min(batch * t_obs, start + chunk_frames)
                tokens_chunk, debug_chunk = self._tokenize_frames(
                    xyz_f=xyz_f[start:end],
                    state_f=state_f[start:end],
                    task_emb_f=task_emb_f[start:end],
                    valid_f=None if valid_f is None else valid_f[start:end],
                    rgb_f=None if rgb_f is None else rgb_f[start:end],
                    mask_f=None if mask_f is None else mask_f[start:end],
                    return_debug=return_debug,
                )
                chunks.append(tokens_chunk)
                if return_debug and debug_chunk is not None:
                    debug_chunks.append(debug_chunk)
            frame_tokens = torch.cat(chunks, dim=0)
            if return_debug and debug_chunks:
                debug = debug_chunks[0]
        else:
            frame_tokens, debug = self._tokenize_frames(
                xyz_f=xyz_f,
                state_f=state_f,
                task_emb_f=task_emb_f,
                valid_f=valid_f,
                rgb_f=rgb_f,
                mask_f=mask_f,
                return_debug=return_debug,
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
        return ObservationEncoderOutput(tokens=tokens, debug=debug)
