from __future__ import annotations

import json
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from ml_collections import ConfigDict
from tqdm.auto import tqdm

from supernode_tokenizer.data.cache_variation_h5 import (
    MASK_NAME_SUBSTRINGS_TO_IGNORE,
    MASK_NAMES_TO_IGNORE,
    _build_vector,
    _filter_by_ignore_ids,
    _filter_by_xyz_bounds,
    _subsample_fixedN,
)
from supernode_tokenizer.models import ModelConfig, build_policy
from supernode_tokenizer.utils import dataclass_from_dict

_CAMERAS: Tuple[str, ...] = (
    "left_shoulder",
    "right_shoulder",
    "overhead",
    "wrist",
    "front",
)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _resolve_device(device_str: str) -> torch.device:
    if torch.cuda.is_available() and str(device_str).startswith("cuda"):
        return torch.device(str(device_str))
    return torch.device("cpu")


def _strip_module_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if state_dict and all(k.startswith("module.") for k in state_dict):
        return {k[len("module."):]: v for k, v in state_dict.items()}
    return state_dict


def _load_checkpoint(path: Path, device: torch.device) -> Dict[str, Any]:
    ckpt = torch.load(path, map_location=device)
    if not isinstance(ckpt, dict):
        raise TypeError(f"Unsupported checkpoint type: {type(ckpt).__name__}")
    ckpt["model"] = _strip_module_prefix(ckpt["model"])
    return ckpt


def _model_cfg_from_checkpoint(ckpt: Dict[str, Any]) -> ModelConfig:
    config = (ckpt.get("config", {}) or {}).get("model", {})
    return dataclass_from_dict(ModelConfig(), config)


def _normalize_quaternion_xyzw(q: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(q))
    if norm < 1e-8:
        return np.asarray([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return (q / norm).astype(np.float32)


def _sanitize_action(action: np.ndarray, *, normalize_quaternion: bool, discretize_gripper: bool) -> np.ndarray:
    out = np.asarray(action, dtype=np.float32).copy()
    if out.shape[0] >= 7 and normalize_quaternion:
        out[3:7] = _normalize_quaternion_xyzw(out[3:7])
    if out.shape[0] >= 8:
        out[7] = 1.0 if float(out[7]) > 0.5 else 0.0 if discretize_gripper else float(np.clip(out[7], 0.0, 1.0))
    return out


def _extract_rgb_frame(obs: Any, camera: str) -> np.ndarray:
    frame = getattr(obs, f"{camera}_rgb", None)
    if frame is None:
        frame = getattr(obs, "front_rgb", None)
    if frame is None:
        return np.zeros((128, 128, 3), dtype=np.uint8)
    arr = np.asarray(frame)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        arr = np.repeat(arr[..., None], 3, axis=-1)
    return arr


def _write_video(frames: Sequence[np.ndarray], out_path: Path, fps: int) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ext = out_path.suffix.lower()
    if ext == ".mp4":
        try:
            import imageio.v2 as imageio

            imageio.mimsave(str(out_path), list(frames), fps=int(fps))
            return out_path
        except Exception:
            out_path = out_path.with_suffix(".gif")
            ext = ".gif"
    if ext == ".gif":
        try:
            from PIL import Image

            pil_frames = [Image.fromarray(np.asarray(frame, dtype=np.uint8)) for frame in frames]
            pil_frames[0].save(
                out_path,
                save_all=True,
                append_images=pil_frames[1:],
                duration=max(1, int(round(1000.0 / max(1, int(fps))))),
                loop=0,
            )
            return out_path
        except Exception:
            pass
    fallback = out_path.with_suffix(".npz")
    np.savez_compressed(str(fallback), frames=np.asarray(frames, dtype=np.uint8))
    return fallback


class LiveObservationProcessor:
    def __init__(
        self,
        *,
        task_env: Any,
        num_points: int,
        use_rgb: bool,
        use_mask_id: bool,
        point_dropout: float,
        seed: int,
    ):
        self.task_env = task_env
        self.num_points = int(num_points)
        self.use_rgb = bool(use_rgb)
        self.use_mask_id = bool(use_mask_id)
        self.point_dropout = float(point_dropout)
        self.rng = np.random.default_rng(int(seed))
        self.handle_to_name: Dict[int, str] = {}

    def _update_handle_names(self, masks: Sequence[np.ndarray]) -> None:
        from rlbench.segmentation_utils import build_handle_label_map

        unresolved: set[int] = set()
        for arr in masks:
            flat = np.asarray(arr).reshape(-1).astype(np.int64, copy=False)
            for value in np.unique(flat):
                value_i = int(value)
                if value_i != 0 and value_i not in self.handle_to_name:
                    unresolved.add(value_i)
        if not unresolved:
            return
        mapping, _ = build_handle_label_map(self.task_env, unresolved)
        for handle, name in mapping.items():
            self.handle_to_name[int(handle)] = str(name)

    def _ignore_ids(self) -> Tuple[int, ...]:
        ignore = set()
        for handle, name in self.handle_to_name.items():
            if name in MASK_NAMES_TO_IGNORE:
                ignore.add(int(handle))
                continue
            lname = str(name).lower()
            if any(token in lname for token in MASK_NAME_SUBSTRINGS_TO_IGNORE):
                ignore.add(int(handle))
        return tuple(sorted(ignore))

    def observation_to_frame(self, obs: Any) -> Dict[str, torch.Tensor]:
        merged_points: List[np.ndarray] = []
        merged_colors: List[np.ndarray] = []
        merged_masks: List[np.ndarray] = []
        raw_masks: List[np.ndarray] = []
        for camera in _CAMERAS:
            pc = getattr(obs, f"{camera}_point_cloud", None)
            mask = getattr(obs, f"{camera}_mask", None)
            rgb = getattr(obs, f"{camera}_rgb", None)
            if pc is None or mask is None:
                continue
            pts = np.asarray(pc, dtype=np.float32).reshape(-1, 3)
            masks = np.asarray(mask).reshape(-1).astype(np.int32, copy=False)
            cols = np.asarray(rgb).reshape(-1, 3).astype(np.uint8, copy=False) if (self.use_rgb and rgb is not None) else None
            finite = np.isfinite(pts).all(axis=1)
            pts = pts[finite]
            masks = masks[finite]
            if cols is not None:
                cols = cols[finite]
            if pts.shape[0] == 0:
                continue
            merged_points.append(pts)
            merged_masks.append(masks)
            if cols is not None:
                merged_colors.append(cols)
            raw_masks.append(masks)

        if merged_points:
            pts_all = np.concatenate(merged_points, axis=0).astype(np.float32, copy=False)
            mask_all = np.concatenate(merged_masks, axis=0).astype(np.int32, copy=False)
            rgb_all = np.concatenate(merged_colors, axis=0).astype(np.uint8, copy=False) if (self.use_rgb and merged_colors) else None
        else:
            pts_all = np.zeros((0, 3), dtype=np.float32)
            mask_all = np.zeros((0,), dtype=np.int32)
            rgb_all = np.zeros((0, 3), dtype=np.uint8) if self.use_rgb else None

        self._update_handle_names(raw_masks)
        keep = _filter_by_ignore_ids(mask_all, self._ignore_ids())
        pts_all = pts_all[keep]
        mask_all = mask_all[keep]
        if rgb_all is not None:
            rgb_all = rgb_all[keep]

        keep = _filter_by_xyz_bounds(pts_all)
        pts_all = pts_all[keep]
        mask_all = mask_all[keep]
        if rgb_all is not None:
            rgb_all = rgb_all[keep]

        if self.point_dropout > 0.0 and pts_all.shape[0] > 0:
            keep_n = max(1, int(round((1.0 - self.point_dropout) * float(pts_all.shape[0]))))
            choice = self.rng.choice(pts_all.shape[0], size=keep_n, replace=False)
            pts_all = pts_all[choice]
            mask_all = mask_all[choice]
            if rgb_all is not None:
                rgb_all = rgb_all[choice]

        if pts_all.shape[0] == 0:
            xyz = np.zeros((self.num_points, 3), dtype=np.float32)
            valid = np.zeros((self.num_points,), dtype=bool)
            rgb = np.zeros((self.num_points, 3), dtype=np.uint8) if self.use_rgb else None
            mask_id = np.zeros((self.num_points,), dtype=np.int64) if self.use_mask_id else None
        else:
            idx = _subsample_fixedN(self.rng, int(pts_all.shape[0]), self.num_points)
            xyz = pts_all[idx].astype(np.float32, copy=False)
            valid = np.ones((self.num_points,), dtype=bool)
            rgb = rgb_all[idx].astype(np.uint8, copy=False) if (self.use_rgb and rgb_all is not None) else None
            mask_id = mask_all[idx].astype(np.int64, copy=False) if self.use_mask_id else None

        state = _build_vector(obs, ("gripper_pose", "gripper_open")).astype(np.float32, copy=False)
        out: Dict[str, torch.Tensor] = {
            "xyz": torch.from_numpy(xyz).float(),
            "state": torch.from_numpy(state).float(),
            "valid": torch.from_numpy(valid).bool(),
        }
        if rgb is not None:
            out["rgb"] = torch.from_numpy(rgb).float() / 255.0
        if mask_id is not None:
            out["mask_id"] = torch.from_numpy(mask_id).long()
        return out


def _build_query_window(
    history: Sequence[Dict[str, torch.Tensor]],
    *,
    t_obs: int,
    stride: int,
) -> Dict[str, torch.Tensor]:
    if not history:
        raise RuntimeError("Observation history is empty.")
    last = len(history) - 1
    idx = [max(0, last - (t_obs - 1 - i) * stride) for i in range(t_obs)]
    frames = [history[i] for i in idx]
    out: Dict[str, torch.Tensor] = {
        "obs_xyz": torch.stack([f["xyz"] for f in frames], 0).unsqueeze(0),
        "obs_state": torch.stack([f["state"] for f in frames], 0).unsqueeze(0),
        "obs_valid": torch.stack([f["valid"] for f in frames], 0).unsqueeze(0),
    }
    if all("rgb" in frame for frame in frames):
        out["obs_rgb"] = torch.stack([f["rgb"] for f in frames], 0).unsqueeze(0)
    if all("mask_id" in frame for frame in frames):
        out["obs_mask_id"] = torch.stack([f["mask_id"] for f in frames], 0).unsqueeze(0)
    return out


def _build_rlbench_env(cfg: ConfigDict):
    from pyrep.const import RenderMode
    from rlbench import ObservationConfig
    from rlbench.action_modes.action_mode import MoveArmThenGripper
    from rlbench.action_modes.arm_action_modes import EndEffectorPoseViaPlanning
    from rlbench.action_modes.gripper_action_modes import Discrete
    from rlbench.environment import Environment

    obs_config = ObservationConfig()
    obs_config.set_all(True)
    image_size = tuple(int(x) for x in cfg.sim.image_size)
    renderer_name = str(cfg.sim.renderer).lower()
    render_mode = RenderMode.OPENGL if renderer_name == "opengl" else RenderMode.OPENGL3
    for camera in _CAMERAS:
        cam_cfg = getattr(obs_config, f"{camera}_camera")
        cam_cfg.image_size = image_size
        cam_cfg.depth_in_meters = False
        cam_cfg.masks_as_one_channel = True
        cam_cfg.render_mode = render_mode

    action_mode = MoveArmThenGripper(
        EndEffectorPoseViaPlanning(
            absolute_mode=True,
            collision_checking=bool(cfg.sim.collision_checking),
        ),
        Discrete(),
    )
    env = Environment(
        action_mode=action_mode,
        obs_config=obs_config,
        headless=bool(cfg.sim.headless),
        arm_max_velocity=float(cfg.sim.arm_max_velocity),
        arm_max_acceleration=float(cfg.sim.arm_max_acceleration),
    )
    env.launch()
    return env


def _run_episode(
    *,
    task_env: Any,
    model: Any,
    device: torch.device,
    task_id: int,
    t_obs: int,
    stride: int,
    processor: LiveObservationProcessor,
    execute_actions_per_plan: int,
    max_env_steps: int,
    inference_steps: int,
    eta: float,
    normalize_quaternion: bool,
    discretize_gripper: bool,
    video_cfg: ConfigDict,
    episode_index: int,
    task_run_dir: Path,
) -> Dict[str, Any]:
    from rlbench.backend.exceptions import InvalidActionError

    descriptions, obs = task_env.reset()
    del descriptions
    history = [processor.observation_to_frame(obs)]
    frames: List[np.ndarray] = []
    if bool(video_cfg.enable):
        frames.append(_extract_rgb_frame(obs, str(video_cfg.camera)))

    success = False
    terminated = False
    env_steps = 0
    error: Optional[str] = None

    pbar = tqdm(total=max_env_steps, desc=f"episode {episode_index}", leave=False, unit="step")
    try:
        while env_steps < max_env_steps and not success and not terminated:
            query = _build_query_window(history, t_obs=t_obs, stride=stride)
            batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in query.items()}
            batch["task_ids"] = torch.tensor([int(task_id)], device=device, dtype=torch.long)
            with torch.no_grad():
                if hasattr(model, "cfg") and hasattr(model.cfg, "num_inference_steps"):
                    plan = model.sample_actions(
                        task_ids=batch["task_ids"],
                        obs_xyz=batch["obs_xyz"],
                        obs_state=batch["obs_state"],
                        obs_valid=batch.get("obs_valid", None),
                        obs_rgb=batch.get("obs_rgb", None),
                        obs_mask_id=batch.get("obs_mask_id", None),
                        inference_steps=int(inference_steps),
                        eta=float(eta),
                    )
                else:
                    plan = model.sample_actions(
                        task_ids=batch["task_ids"],
                        obs_xyz=batch["obs_xyz"],
                        obs_state=batch["obs_state"],
                        obs_valid=batch.get("obs_valid", None),
                        obs_rgb=batch.get("obs_rgb", None),
                        obs_mask_id=batch.get("obs_mask_id", None),
                    )
            plan_np = plan[0].detach().cpu().numpy()
            for action_idx in range(min(int(execute_actions_per_plan), int(plan_np.shape[0]), max_env_steps - env_steps)):
                action = _sanitize_action(plan_np[action_idx], normalize_quaternion=normalize_quaternion, discretize_gripper=discretize_gripper)
                try:
                    obs, reward, terminated = task_env.step(action.astype(np.float32))
                except InvalidActionError as exc:
                    error = f"InvalidActionError: {exc}"
                    terminated = True
                    break
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    terminated = True
                    break
                env_steps += 1
                pbar.update(1)
                success = bool(float(reward) > 0.5)
                history.append(processor.observation_to_frame(obs))
                if bool(video_cfg.enable):
                    frames.append(_extract_rgb_frame(obs, str(video_cfg.camera)))
                if success or terminated or env_steps >= max_env_steps:
                    break
    finally:
        pbar.close()

    video_path = None
    if bool(video_cfg.enable) and frames:
        video_name = f"episode_{episode_index:04d}.{str(video_cfg.format).lower()}"
        video_path = str(_write_video(frames, task_run_dir / "videos" / video_name, fps=int(video_cfg.fps)))
    return {
        "episode_index": int(episode_index),
        "success": bool(success),
        "terminated": bool(terminated),
        "env_steps": int(env_steps),
        "error": error,
        "video_path": video_path,
    }


def evaluate_policy(cfg: ConfigDict) -> Dict[str, Any]:
    seed = int(cfg.seed)
    _set_seed(seed)
    device = _resolve_device(str(cfg.device))
    checkpoint_path = Path(str(cfg.checkpoint_path)).expanduser().resolve()
    ckpt = _load_checkpoint(checkpoint_path, device)
    model_cfg = _model_cfg_from_checkpoint(ckpt)
    state_dim = int(ckpt.get("state_dim", 8))
    action_dim = int(ckpt.get("action_dim", 8))
    task_name_to_id = {str(k): int(v) for k, v in (ckpt.get("task_name_to_id", {}) or {}).items()}
    if not task_name_to_id:
        raise RuntimeError("Checkpoint is missing task_name_to_id; cannot run class-conditioned evaluation.")

    model = build_policy(model_cfg, state_dim=state_dim, action_dim=action_dim).to(device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()

    tasks = [str(task) for task in (list(cfg.eval.tasks) if cfg.eval.tasks else list(task_name_to_id.keys()))]
    run_id = f"{time.strftime('%Y%m%d-%H%M%S')}_{time.time_ns() % 1000000:06d}"
    out_dir = Path(str(cfg.output.root_dir)).expanduser().resolve() / f"eval_{run_id}"
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "resolved_eval_config.json").open("w", encoding="utf-8") as file:
        json.dump(cfg.to_dict(), file, indent=2)

    overall_results: Dict[str, Any] = {"checkpoint": str(checkpoint_path), "tasks": {}, "summary": {}}
    task_success_rates: Dict[str, float] = {}
    env = _build_rlbench_env(cfg)
    try:
        from rlbench.backend.utils import task_file_to_task_class

        for task_name in tasks:
            if task_name not in task_name_to_id:
                raise KeyError(f"Task {task_name!r} is not available in the checkpoint task mapping.")
            task_env = env.get_task(task_file_to_task_class(task_name))
            task_dir = out_dir / task_name
            task_dir.mkdir(parents=True, exist_ok=True)
            task_results = []
            processor = LiveObservationProcessor(
                task_env=task_env,
                num_points=int(cfg.conditioning.num_points),
                use_rgb=bool(cfg.conditioning.use_rgb),
                use_mask_id=bool(cfg.conditioning.use_mask_id),
                point_dropout=float(cfg.robustness.point_dropout),
                seed=seed + 17,
            )
            if cfg.eval.variation_ids:
                variation_ids = [int(v) for v in cfg.eval.variation_ids]
            else:
                variation_ids = list(range(int(task_env.variation_count())))
                if not variation_ids:
                    variation_ids = [0]
            for episode_idx in range(int(cfg.eval.episodes_per_task)):
                variation = int(variation_ids[episode_idx % len(variation_ids)])
                task_env.set_variation(variation)
                res = _run_episode(
                    task_env=task_env,
                    model=model,
                    device=device,
                    task_id=int(task_name_to_id[task_name]),
                    t_obs=int(cfg.dataset.T_obs),
                    stride=int(cfg.dataset.stride if cfg.eval.query_stride_mode == "dataset" else 1),
                    processor=processor,
                    execute_actions_per_plan=int(cfg.control.execute_actions_per_plan),
                    max_env_steps=int(cfg.eval.max_env_steps),
                    inference_steps=int(cfg.inference.inference_steps),
                    eta=float(cfg.inference.eta),
                    normalize_quaternion=bool(cfg.control.normalize_quaternion),
                    discretize_gripper=bool(cfg.control.discretize_gripper),
                    video_cfg=cfg.video,
                    episode_index=episode_idx,
                    task_run_dir=task_dir,
                )
                res["variation"] = variation
                task_results.append(res)
            success_rate = float(sum(1 for r in task_results if r["success"]) / max(1, len(task_results)))
            task_success_rates[task_name] = success_rate
            overall_results["tasks"][task_name] = {
                "success_rate": success_rate,
                "episodes": task_results,
            }
    finally:
        env.shutdown()

    all_success = float(sum(task_success_rates.values()) / max(1, len(task_success_rates)))
    geometry_subset = [task for task in cfg.eval.geometry_subset if task in task_success_rates]
    geometry_success = float(sum(task_success_rates[t] for t in geometry_subset) / max(1, len(geometry_subset)))
    overall_results["summary"] = {
        "all_tasks_success_rate": all_success,
        "geometry_subset_success_rate": geometry_success,
        "per_task_success_rate": task_success_rates,
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(overall_results, file, indent=2)
    return overall_results


def evaluate_robustness(cfg: ConfigDict) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"point_count": {}, "point_dropout": {}}
    base_num_points = int(cfg.conditioning.num_points)
    base_dropout = float(cfg.robustness.point_dropout)
    for num_points in cfg.robustness.point_counts:
        cfg.conditioning.num_points = int(num_points)
        cfg.robustness.point_dropout = 0.0
        result = evaluate_policy(cfg)
        summary["point_count"][str(num_points)] = result["summary"]
    cfg.conditioning.num_points = base_num_points
    for dropout in cfg.robustness.point_dropouts:
        cfg.robustness.point_dropout = float(dropout)
        result = evaluate_policy(cfg)
        summary["point_dropout"][str(dropout)] = result["summary"]
    cfg.robustness.point_dropout = base_dropout
    return summary
