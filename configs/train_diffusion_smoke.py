from __future__ import annotations

from configs.train_diffusion_policy import get_config as _get_base_config


def get_config():
    cfg = _get_base_config()
    cfg.data.tasks = ["open_drawer"]
    cfg.train.num_steps = 200
    cfg.train.batch_size = 2
    cfg.train.val_batch_size = 2
    cfg.train.num_workers = 0
    cfg.train.log_every = 5
    cfg.train.eval_every = 50
    cfg.train.ckpt_every = 100
    cfg.train.val_num_samples = 64
    cfg.train.val_num_batches = 4
    cfg.output.run_name = "diffusion_smoke"
    return cfg
