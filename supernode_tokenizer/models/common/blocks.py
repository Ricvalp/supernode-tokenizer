from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .attention import CrossAttention, SelfAttention


class IdentityTaskFiLM(nn.Module):
    def __init__(self, dim: int, cond_dim: int, *, use_layernorm: bool = False):
        super().__init__()
        self.norm = nn.LayerNorm(dim) if use_layernorm else None
        self.to_scale_shift = nn.Linear(cond_dim, 2 * dim)
        nn.init.zeros_(self.to_scale_shift.weight)
        nn.init.zeros_(self.to_scale_shift.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.norm(x) if self.norm is not None else x
        scale, shift = self.to_scale_shift(cond).unsqueeze(1).chunk(2, dim=-1)
        return h * (1.0 + scale) + shift


class IdentityTaskAdaLN(nn.Module):
    def __init__(self, dim: int, cond_dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.to_scale_shift = nn.Linear(cond_dim, 2 * dim)
        nn.init.zeros_(self.to_scale_shift.weight)
        nn.init.zeros_(self.to_scale_shift.bias)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        scale, shift = self.to_scale_shift(cond).unsqueeze(1).chunk(2, dim=-1)
        return h * (1.0 + scale) + shift


class TaskConditionedSelfAttentionBlock(nn.Module):
    def __init__(
        self,
        *,
        d: int,
        n_heads: int,
        cond_dim: int,
        mlp_mult: int = 4,
        dropout: float = 0.0,
        attention_backend: str = "manual",
    ):
        super().__init__()
        self.adaln1 = IdentityTaskAdaLN(d, cond_dim)
        self.self_attn = SelfAttention(d, n_heads, dropout, attention_backend=attention_backend)
        self.adaln2 = IdentityTaskAdaLN(d, cond_dim)
        self.mlp = nn.Sequential(
            nn.Linear(d, int(mlp_mult) * d),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(mlp_mult) * d, d),
        )
        self.drop = nn.Dropout(float(dropout))

    def forward(self, x: torch.Tensor, cond: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.drop(self.self_attn(self.adaln1(x, cond), mask=mask))
        x = x + self.drop(self.mlp(self.adaln2(x, cond)))
        return x


class TaskConditionedCrossAttentionBlock(nn.Module):
    def __init__(
        self,
        *,
        d: int,
        n_heads: int,
        cond_dim: int,
        mlp_mult: int = 4,
        dropout: float = 0.0,
        attention_backend: str = "manual",
    ):
        super().__init__()
        self.adaln1 = IdentityTaskAdaLN(d, cond_dim)
        self.self_attn = SelfAttention(d, n_heads, dropout, attention_backend=attention_backend)
        self.adaln2 = IdentityTaskAdaLN(d, cond_dim)
        self.cross_attn = CrossAttention(d, n_heads, dropout, attention_backend=attention_backend)
        self.adaln3 = IdentityTaskAdaLN(d, cond_dim)
        self.mlp = nn.Sequential(
            nn.Linear(d, int(mlp_mult) * d),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(mlp_mult) * d, d),
        )
        self.drop = nn.Dropout(float(dropout))

    def forward(
        self,
        x: torch.Tensor,
        cond: torch.Tensor,
        memory: torch.Tensor,
        memory_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.drop(self.self_attn(self.adaln1(x, cond)))
        x = x + self.drop(self.cross_attn(self.adaln2(x, cond), memory, kv_mask=memory_mask))
        x = x + self.drop(self.mlp(self.adaln3(x, cond)))
        return x


class TaskConditionedFramePerceiverTokenizer(nn.Module):
    def __init__(
        self,
        *,
        d: int,
        m: int,
        n_heads: int,
        cond_dim: int,
        n_layers: int = 2,
        dropout: float = 0.0,
        attention_backend: str = "manual",
    ):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(int(m), int(d)) * 0.02)
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "adaln1": IdentityTaskAdaLN(d, cond_dim),
                        "xattn": CrossAttention(d, n_heads, dropout, attention_backend=attention_backend),
                        "adaln2": IdentityTaskAdaLN(d, cond_dim),
                        "mlp": nn.Sequential(
                            nn.Linear(d, 4 * d),
                            nn.GELU(),
                            nn.Dropout(float(dropout)),
                            nn.Linear(4 * d, d),
                        ),
                    }
                )
                for _ in range(int(n_layers))
            ]
        )
        self.drop = nn.Dropout(float(dropout))

    def forward(
        self,
        point_tokens: torch.Tensor,
        cond: torch.Tensor,
        point_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch = int(point_tokens.shape[0])
        z = self.latents.unsqueeze(0).expand(batch, -1, -1)
        for layer in self.layers:
            z = z + self.drop(layer["xattn"](layer["adaln1"](z, cond), point_tokens, kv_mask=point_mask))
            z = z + self.drop(layer["mlp"](layer["adaln2"](z, cond)))
        return z
