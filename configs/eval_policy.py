from __future__ import annotations

import os
from pathlib import Path

from ml_collections import ConfigDict

from supernode_tokenizer.data import GEOMETRY_SENSITIVE_TASKS, RLBENCH18_TASKS

REPO_ROOT = Path(__file__).resolve().parents[1]


def get_config() -> ConfigDict:
    cfg = ConfigDict()
    cfg.seed = 0
    cfg.device = "cuda:0"
    cfg.checkpoint_path = ""

    cfg.dataset = ConfigDict()
    cfg.dataset.T_obs = 2
    cfg.dataset.stride = 2

    cfg.conditioning = ConfigDict()
    cfg.conditioning.num_points = 4096
    cfg.conditioning.use_rgb = False
    cfg.conditioning.use_mask_id = True

    cfg.eval = ConfigDict()
    cfg.eval.tasks = list(RLBENCH18_TASKS)
    cfg.eval.geometry_subset = list(GEOMETRY_SENSITIVE_TASKS)
    cfg.eval.episodes_per_task = 25
    cfg.eval.max_env_steps = 200
    cfg.eval.query_stride_mode = "dataset"
    cfg.eval.variation_ids = []

    cfg.control = ConfigDict()
    cfg.control.execute_actions_per_plan = 8
    cfg.control.normalize_quaternion = True
    cfg.control.discretize_gripper = True

    cfg.inference = ConfigDict()
    cfg.inference.inference_steps = 50
    cfg.inference.eta = 0.0

    cfg.robustness = ConfigDict()
    cfg.robustness.point_dropout = 0.0
    cfg.robustness.point_counts = [4096, 2048, 1024, 512]
    cfg.robustness.point_dropouts = [0.0, 0.25, 0.5]

    cfg.sim = ConfigDict()
    cfg.sim.image_size = (128, 128)
    cfg.sim.renderer = "opengl3"
    cfg.sim.headless = True
    cfg.sim.collision_checking = False
    cfg.sim.arm_max_velocity = 1.0
    cfg.sim.arm_max_acceleration = 4.0

    cfg.video = ConfigDict()
    cfg.video.enable = False
    cfg.video.camera = "front"
    cfg.video.format = "gif"
    cfg.video.fps = 20

    cfg.output = ConfigDict()
    cfg.output.root_dir = os.environ.get("SUPERNODE_TOKENIZER_EVAL_ROOT", str(REPO_ROOT / "eval_output"))
    return cfg
