from __future__ import annotations

from dataclasses import dataclass

from .observation_encoder_supernode import ObservationSupernodeEncoder, ObservationSupernodeEncoderConfig


@dataclass
class ObservationSupernodeNoMessageEncoderConfig(ObservationSupernodeEncoderConfig):
    use_message_passing: bool = False


class ObservationSupernodeNoMessageEncoder(ObservationSupernodeEncoder):
    def __init__(self, *, cfg: ObservationSupernodeNoMessageEncoderConfig, state_dim: int):
        super().__init__(cfg=cfg, state_dim=state_dim)
