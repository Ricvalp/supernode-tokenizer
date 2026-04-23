from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

from ..condition import TaskConditioner
from ..encoders import ObservationEncoder
from ..common import TaskConditionedCrossAttentionBlock, TimeMLP, sinusoidal_position_embedding, sinusoidal_time_embedding


@dataclass
class DiffusionPolicyConfig:
    d_model: int = 512
    n_heads: int = 8
    denoiser_layers: int = 10
    denoiser_mlp_mult: int = 4
    dropout: float = 0.0
    horizon: int = 16
    num_train_timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    beta_schedule: str = "squaredcos_cap_v2"
    prediction_type: str = "v_prediction"
    set_alpha_to_one: bool = True
    steps_offset: int = 0
    num_inference_steps: Optional[int] = None


class ObservationDiffusionPolicy(nn.Module):
    def __init__(
        self,
        *,
        cfg: DiffusionPolicyConfig,
        encoder: ObservationEncoder,
        task_conditioner: TaskConditioner,
        action_dim: int,
    ):
        super().__init__()
        self.cfg = cfg
        self.encoder = encoder
        self.task_conditioner = task_conditioner
        d = int(cfg.d_model)
        self.action_in = nn.Linear(int(action_dim), d)
        self.action_out = nn.Linear(d, int(action_dim))
        self.time_mlp = TimeMLP(emb_dim=d, out_dim=d)
        self.task_cond_proj = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        self.denoiser = nn.ModuleList(
            [
                TaskConditionedCrossAttentionBlock(
                    d=d,
                    n_heads=int(cfg.n_heads),
                    cond_dim=d,
                    mlp_mult=int(cfg.denoiser_mlp_mult),
                    dropout=float(cfg.dropout),
                )
                for _ in range(int(cfg.denoiser_layers))
            ]
        )
        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=int(cfg.num_train_timesteps),
            beta_start=float(cfg.beta_start),
            beta_end=float(cfg.beta_end),
            beta_schedule=str(cfg.beta_schedule),
            clip_sample=False,
            set_alpha_to_one=bool(cfg.set_alpha_to_one),
            steps_offset=int(cfg.steps_offset),
            prediction_type=str(cfg.prediction_type),
        )
        self.num_inference_steps = int(cfg.num_inference_steps) if cfg.num_inference_steps is not None else int(cfg.num_train_timesteps)

    def _build_memory(
        self,
        *,
        task_ids: torch.Tensor,
        obs_xyz: torch.Tensor,
        obs_state: torch.Tensor,
        obs_valid: Optional[torch.Tensor],
        obs_rgb: Optional[torch.Tensor],
        obs_mask_id: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        task_emb, task_tokens = self.task_conditioner(task_ids)
        enc_out = self.encoder(
            obs_xyz=obs_xyz,
            obs_state=obs_state,
            task_emb=task_emb,
            obs_valid=obs_valid,
            obs_rgb=obs_rgb,
            obs_mask_id=obs_mask_id,
            return_debug=False,
        )
        memory = torch.cat([enc_out.tokens, task_tokens], dim=1)
        memory_mask = None
        if enc_out.token_mask is not None:
            task_mask = torch.ones(task_tokens.shape[:2], device=task_tokens.device, dtype=torch.bool)
            memory_mask = torch.cat([enc_out.token_mask.to(torch.bool), task_mask], dim=1)
        return memory, memory_mask, task_emb

    def _time_condition(self, timesteps: torch.Tensor, task_emb: torch.Tensor) -> torch.Tensor:
        t_emb = sinusoidal_time_embedding(timesteps, int(self.cfg.d_model))
        return self.time_mlp(t_emb) + self.task_cond_proj(task_emb)

    def _denoise(
        self,
        noisy_actions: torch.Tensor,
        *,
        timesteps: torch.Tensor,
        memory: torch.Tensor,
        memory_mask: Optional[torch.Tensor],
        task_emb: torch.Tensor,
    ) -> torch.Tensor:
        batch, horizon, _ = noisy_actions.shape
        h = self.action_in(noisy_actions)
        pos = sinusoidal_position_embedding(horizon, int(self.cfg.d_model), noisy_actions.device)
        h = h + pos.unsqueeze(0)
        cond = self._time_condition(timesteps, task_emb)
        for blk in self.denoiser:
            h = blk(h, cond, memory, memory_mask)
        return self.action_out(h)

    def forward_loss(self, batch: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        target = batch["target_action"]
        memory, memory_mask, task_emb = self._build_memory(
            task_ids=batch["task_id"],
            obs_xyz=batch["obs_xyz"],
            obs_state=batch["obs_state"],
            obs_valid=batch.get("obs_valid", None),
            obs_rgb=batch.get("obs_rgb", None),
            obs_mask_id=batch.get("obs_mask_id", None),
        )
        noise = torch.randn_like(target)
        timesteps = torch.randint(
            0,
            int(self.noise_scheduler.config.num_train_timesteps),
            (int(target.shape[0]),),
            device=target.device,
            dtype=torch.long,
        )
        noisy = self.noise_scheduler.add_noise(target, noise, timesteps)
        pred = self._denoise(
            noisy,
            timesteps=timesteps,
            memory=memory,
            memory_mask=memory_mask,
            task_emb=task_emb,
        )
        prediction_type = str(self.noise_scheduler.config.prediction_type)
        if prediction_type == "epsilon":
            gt = noise
        elif prediction_type == "sample":
            gt = target
        elif prediction_type == "v_prediction":
            gt = self.noise_scheduler.get_velocity(target, noise, timesteps)
        else:
            raise ValueError(f"Unsupported prediction_type={prediction_type!r}.")
        loss = torch.nn.functional.mse_loss(pred, gt)
        return {"loss": loss, "diffusion_loss": loss}

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
        inference_steps: Optional[int] = None,
        eta: float = 0.0,
    ) -> torch.Tensor:
        memory, memory_mask, task_emb = self._build_memory(
            task_ids=task_ids,
            obs_xyz=obs_xyz,
            obs_state=obs_state,
            obs_valid=obs_valid,
            obs_rgb=obs_rgb,
            obs_mask_id=obs_mask_id,
        )
        steps = int(self.num_inference_steps if inference_steps is None else inference_steps)
        horizon = int(self.cfg.horizon)
        action_dim = int(self.action_out.out_features)
        sample = torch.randn((int(task_ids.shape[0]), horizon, action_dim), device=obs_xyz.device, dtype=obs_xyz.dtype)
        self.noise_scheduler.set_timesteps(steps, device=obs_xyz.device)
        for t in self.noise_scheduler.timesteps:
            timestep = torch.full((int(task_ids.shape[0]),), int(t), device=obs_xyz.device, dtype=torch.long)
            model_out = self._denoise(
                sample,
                timesteps=timestep,
                memory=memory,
                memory_mask=memory_mask,
                task_emb=task_emb,
            )
            step_out = self.noise_scheduler.step(model_out, t, sample, eta=float(eta))
            sample = step_out.prev_sample
        return sample
