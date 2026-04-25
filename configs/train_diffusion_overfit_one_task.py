from __future__ import annotations

from configs.train_diffusion_policy import get_config as _get_base_config


def get_config():
    cfg = _get_base_config()
    cfg.data.tasks = ["stack_blocks"]
    cfg.data.low_data_train_demos_per_variation = 4
    cfg.train.num_steps = 50000
    cfg.train.batch_size = 8
    cfg.train.val_batch_size = 2
    cfg.train.num_workers = 0
    cfg.train.log_every = 10
    cfg.train.eval_every = 100
    cfg.train.ckpt_every = 500
    cfg.train.val_num_samples = 32
    cfg.train.val_num_batches = 4
    cfg.output.run_name = "diffusion_overfit_one_task"
    return cfg
