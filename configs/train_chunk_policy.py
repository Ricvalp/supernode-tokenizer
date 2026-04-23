from __future__ import annotations

import os
import time
from pathlib import Path

from ml_collections import ConfigDict

from supernode_tokenizer.data import RLBENCH18_TASKS

REPO_ROOT = Path(__file__).resolve().parents[1]


def get_config() -> ConfigDict:
    cfg = ConfigDict()
    cfg.seed = 0

    cfg.data = ConfigDict()
    cfg.data.cache_root = os.environ.get(
        "SUPERNODE_TOKENIZER_CACHE_ROOT",
        os.environ.get("ICIL_CACHE_ROOT", str(REPO_ROOT / ".rlbench_cache_standard_il")),
    )
    cfg.data.tasks = list(RLBENCH18_TASKS)
    cfg.data.task_sampling = "variation_power"
    cfg.data.task_sampling_alpha = 0.5
    cfg.data.low_data_train_demos_per_variation = -1
    cfg.data.use_rgb = False
    cfg.data.use_mask_id = True

    cfg.dataset = ConfigDict()
    cfg.dataset.T_obs = 2
    cfg.dataset.H = 16
    cfg.dataset.stride = 2

    cfg.model = ConfigDict()
    cfg.model.d_model = 512
    cfg.model.n_task_tokens = 4
    cfg.model.encoder_name = "perceiver"

    cfg.model.perceiver_encoder = ConfigDict()
    cfg.model.perceiver_encoder.d_model = 512
    cfg.model.perceiver_encoder.n_heads = 8
    cfg.model.perceiver_encoder.m_frame_tokens = 128
    cfg.model.perceiver_encoder.frame_tokenizer_layers = 2
    cfg.model.perceiver_encoder.post_self_attn_layers = 2
    cfg.model.perceiver_encoder.post_self_attn_mlp_mult = 4
    cfg.model.perceiver_encoder.dropout = 0.0
    cfg.model.perceiver_encoder.attention_backend = "manual"
    cfg.model.perceiver_encoder.rgb_alpha_init = 1.0
    cfg.model.perceiver_encoder.use_gripper_point_features = True
    cfg.model.perceiver_encoder.gripper_xyz_state_start = 0
    cfg.model.perceiver_encoder.gripper_alpha_init = 1.0
    cfg.model.perceiver_encoder.tokenize_frames_chunked = True
    cfg.model.perceiver_encoder.chunk_frames = 64

    cfg.model.supernode_encoder = ConfigDict()
    cfg.model.supernode_encoder.d_model = 512
    cfg.model.supernode_encoder.n_heads = 8
    cfg.model.supernode_encoder.frame_tokens_out = 128
    cfg.model.supernode_encoder.num_supernodes = 192
    cfg.model.supernode_encoder.neighbors_per_supernode = 32
    cfg.model.supernode_encoder.supernode_refine_layers = 2
    cfg.model.supernode_encoder.compress_supernodes = True
    cfg.model.supernode_encoder.supernode_pool_layers = 1
    cfg.model.supernode_encoder.post_self_attn_layers = 2
    cfg.model.supernode_encoder.post_self_attn_mlp_mult = 4
    cfg.model.supernode_encoder.dropout = 0.0
    cfg.model.supernode_encoder.attention_backend = "manual"
    cfg.model.supernode_encoder.use_mask_id = True
    cfg.model.supernode_encoder.use_mask_embedding = False
    cfg.model.supernode_encoder.mask_hash_buckets = 2048
    cfg.model.supernode_encoder.use_mask_instance_quota = True
    cfg.model.supernode_encoder.supernode_sampling_mode = "fast_random"
    cfg.model.supernode_encoder.min_mask_supernodes = 4
    cfg.model.supernode_encoder.min_gripper_supernodes = 2
    cfg.model.supernode_encoder.gripper_sampling_radius = 0.10
    cfg.model.supernode_encoder.use_gripper_point_features = True
    cfg.model.supernode_encoder.gripper_xyz_state_start = 0
    cfg.model.supernode_encoder.gripper_alpha_init = 1.0
    cfg.model.supernode_encoder.rgb_alpha_init = 1.0
    cfg.model.supernode_encoder.tokenize_frames_chunked = True
    cfg.model.supernode_encoder.chunk_frames = 64
    cfg.model.supernode_encoder.use_message_passing = True

    cfg.model.supernode_nomsg_encoder = ConfigDict(cfg.model.supernode_encoder.to_dict())
    cfg.model.supernode_nomsg_encoder.use_message_passing = False

    cfg.model.chunk_decoder = ConfigDict()
    cfg.model.chunk_decoder.d_model = 512
    cfg.model.chunk_decoder.n_heads = 8
    cfg.model.chunk_decoder.n_layers = 8
    cfg.model.chunk_decoder.mlp_mult = 4
    cfg.model.chunk_decoder.dropout = 0.0
    cfg.model.chunk_decoder.horizon = 16
    cfg.model.chunk_decoder.loss_type = "l1"

    cfg.optimizer = ConfigDict()
    cfg.optimizer.lr = 1e-4
    cfg.optimizer.beta1 = 0.9
    cfg.optimizer.beta2 = 0.95
    cfg.optimizer.weight_decay = 1e-4
    cfg.optimizer.grad_clip_norm = 1.0

    cfg.train = ConfigDict()
    cfg.train.num_steps = 300000
    cfg.train.batch_size = 128
    cfg.train.val_batch_size = 32
    cfg.train.grad_accum_steps = 1
    cfg.train.num_workers = 8
    cfg.train.amp = True
    cfg.train.log_every = 20
    cfg.train.eval_every = 5000
    cfg.train.ckpt_every = 10000
    cfg.train.val_num_samples = 1024
    cfg.train.val_num_batches = 32
    cfg.train.resume_path = ""

    default_run = f"chunk_{cfg.model.encoder_name}_{time.strftime('%Y%m%d_%H%M%S')}"
    cfg.output = ConfigDict()
    cfg.output.run_name = default_run
    cfg.output.root_dir = os.environ.get("SUPERNODE_TOKENIZER_OUTPUT_ROOT", str(REPO_ROOT / "output"))
    cfg.output.checkpoint_dir = os.environ.get(
        "SUPERNODE_TOKENIZER_CHECKPOINT_ROOT",
        str(REPO_ROOT / "checkpoints"),
    )

    cfg.wandb = ConfigDict()
    cfg.wandb.enable = True
    cfg.wandb.project = os.environ.get("WANDB_PROJECT", "supernode-tokenizer")
    cfg.wandb.entity = os.environ.get("WANDB_ENTITY", "")
    cfg.wandb.mode = os.environ.get("WANDB_MODE", "online")
    cfg.wandb.name = ""
    return cfg
