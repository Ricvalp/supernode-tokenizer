from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class DDPContext:
    distributed: bool
    rank: int
    world_size: int
    local_rank: int
    device: torch.device

    @property
    def is_main_process(self) -> bool:
        return int(self.rank) == 0


def init_distributed() -> DDPContext:
    if dist.is_available() and "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl")
        device = torch.device(f"cuda:{local_rank}")
        return DDPContext(True, rank, world_size, local_rank, device)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return DDPContext(False, 0, 1, 0, device)


def cleanup_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def barrier() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def all_reduce_mean(value: torch.Tensor) -> torch.Tensor:
    if not (dist.is_available() and dist.is_initialized()):
        return value
    out = value.clone()
    dist.all_reduce(out, op=dist.ReduceOp.SUM)
    out /= float(dist.get_world_size())
    return out


def all_reduce_dict_mean(metrics: Dict[str, float], device: torch.device) -> Dict[str, float]:
    if not metrics:
        return metrics
    if not (dist.is_available() and dist.is_initialized()):
        return metrics
    keys = list(metrics.keys())
    values = torch.tensor([float(metrics[k]) for k in keys], device=device, dtype=torch.float64)
    values = all_reduce_mean(values)
    return {k: float(v) for k, v in zip(keys, values.tolist())}
