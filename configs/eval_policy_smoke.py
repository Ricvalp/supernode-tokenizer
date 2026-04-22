from __future__ import annotations

from configs.eval_policy import get_config as _get_base_config


def get_config():
    cfg = _get_base_config()
    cfg.eval.tasks = ["open_drawer"]
    cfg.eval.episodes_per_task = 1
    cfg.eval.max_env_steps = 32
    cfg.video.enable = False
    return cfg
