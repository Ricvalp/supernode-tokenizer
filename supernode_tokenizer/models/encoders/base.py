from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


@dataclass
class ObservationEncoderOutput:
    tokens: torch.Tensor
    token_mask: Optional[torch.Tensor] = None
    debug: Optional[Dict[str, Any]] = None


class ObservationEncoder(nn.Module):
    d_model: int

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
        raise NotImplementedError
