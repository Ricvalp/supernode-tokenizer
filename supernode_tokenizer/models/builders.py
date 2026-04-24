from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple

from .condition import TaskConditioner, TaskConditionerConfig
from .encoders import (
    ObservationDP3Encoder,
    ObservationDP3EncoderConfig,
    ObservationPerceiverEncoder,
    ObservationPerceiverEncoderConfig,
    ObservationSupernodeEncoder,
    ObservationSupernodeEncoderConfig,
    ObservationSupernodeNoMessageEncoder,
    ObservationSupernodeNoMessageEncoderConfig,
)
from .policies.chunk_decoder_policy import ChunkDecoderConfig, ChunkDecoderPolicy
from .policies.diffusion_policy import DiffusionPolicyConfig, ObservationDiffusionPolicy


@dataclass
class ModelConfig:
    encoder_name: str = "perceiver"
    policy_head: str = "chunk"
    task_conditioner: TaskConditionerConfig = field(
        default_factory=lambda: TaskConditionerConfig(num_tasks=18, d_model=512, n_task_tokens=4)
    )
    dp3_encoder: ObservationDP3EncoderConfig = field(default_factory=ObservationDP3EncoderConfig)
    perceiver_encoder: ObservationPerceiverEncoderConfig = field(default_factory=ObservationPerceiverEncoderConfig)
    supernode_encoder: ObservationSupernodeEncoderConfig = field(default_factory=ObservationSupernodeEncoderConfig)
    supernode_nomsg_encoder: ObservationSupernodeNoMessageEncoderConfig = field(default_factory=ObservationSupernodeNoMessageEncoderConfig)
    chunk_decoder: ChunkDecoderConfig = field(default_factory=ChunkDecoderConfig)
    diffusion_policy: DiffusionPolicyConfig = field(default_factory=DiffusionPolicyConfig)


def build_encoder(cfg: ModelConfig, *, state_dim: int):
    name = str(cfg.encoder_name)
    if name == "dp3":
        return ObservationDP3Encoder(cfg=cfg.dp3_encoder, state_dim=state_dim)
    if name == "perceiver":
        return ObservationPerceiverEncoder(cfg=cfg.perceiver_encoder, state_dim=state_dim)
    if name == "supernode":
        return ObservationSupernodeEncoder(cfg=cfg.supernode_encoder, state_dim=state_dim)
    if name == "supernode_nomsg":
        return ObservationSupernodeNoMessageEncoder(cfg=cfg.supernode_nomsg_encoder, state_dim=state_dim)
    raise ValueError(f"Unknown encoder_name={name!r}.")


def build_policy(cfg: ModelConfig, *, state_dim: int, action_dim: int):
    encoder = build_encoder(cfg, state_dim=state_dim)
    task_conditioner = TaskConditioner(cfg.task_conditioner)
    if str(cfg.policy_head) == "chunk":
        return ChunkDecoderPolicy(
            cfg=cfg.chunk_decoder,
            encoder=encoder,
            task_conditioner=task_conditioner,
            action_dim=action_dim,
        )
    if str(cfg.policy_head) == "diffusion":
        return ObservationDiffusionPolicy(
            cfg=cfg.diffusion_policy,
            encoder=encoder,
            task_conditioner=task_conditioner,
            action_dim=action_dim,
        )
    raise ValueError(f"Unknown policy_head={cfg.policy_head!r}.")


def infer_active_d_model(cfg: ModelConfig) -> int:
    if str(cfg.encoder_name) == "dp3":
        return int(cfg.dp3_encoder.d_model)
    if str(cfg.encoder_name) == "perceiver":
        return int(cfg.perceiver_encoder.d_model)
    if str(cfg.encoder_name) == "supernode":
        return int(cfg.supernode_encoder.d_model)
    if str(cfg.encoder_name) == "supernode_nomsg":
        return int(cfg.supernode_nomsg_encoder.d_model)
    raise ValueError(f"Unknown encoder_name={cfg.encoder_name!r}.")


def validate_model_config(cfg: ModelConfig) -> None:
    d_model = int(infer_active_d_model(cfg))
    if int(cfg.task_conditioner.d_model) != d_model:
        raise ValueError(
            f"Task conditioner d_model={cfg.task_conditioner.d_model} must match encoder d_model={d_model}."
        )
    if str(cfg.policy_head) == "chunk":
        if int(cfg.chunk_decoder.d_model) != d_model:
            raise ValueError("chunk_decoder.d_model must match encoder d_model.")
    elif str(cfg.policy_head) == "diffusion":
        if int(cfg.diffusion_policy.d_model) != d_model:
            raise ValueError("diffusion_policy.d_model must match encoder d_model.")
