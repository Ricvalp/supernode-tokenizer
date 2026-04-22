from __future__ import annotations

import argparse
import os
import pickle
import time
from dataclasses import dataclass
from multiprocessing import Manager, Process
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
from PIL import Image
from pyrep.const import RenderMode
from tqdm.auto import tqdm

import rlbench.backend.task as task_backend
from rlbench import ObservationConfig
from rlbench.action_modes.action_mode import MoveArmThenGripper
from rlbench.action_modes.arm_action_modes import JointVelocity
from rlbench.action_modes.gripper_action_modes import Discrete
from rlbench.backend import utils
from rlbench.backend.const import (
    DEPTH_SCALE,
    EPISODE_FOLDER,
    EPISODES_FOLDER,
    FRONT_DEPTH_FOLDER,
    FRONT_MASK_FOLDER,
    FRONT_RGB_FOLDER,
    IMAGE_FORMAT,
    LEFT_SHOULDER_DEPTH_FOLDER,
    LEFT_SHOULDER_MASK_FOLDER,
    LEFT_SHOULDER_RGB_FOLDER,
    LOW_DIM_PICKLE,
    OVERHEAD_DEPTH_FOLDER,
    OVERHEAD_MASK_FOLDER,
    OVERHEAD_RGB_FOLDER,
    RIGHT_SHOULDER_DEPTH_FOLDER,
    RIGHT_SHOULDER_MASK_FOLDER,
    RIGHT_SHOULDER_RGB_FOLDER,
    VARIATION_DESCRIPTIONS,
    VARIATIONS_FOLDER,
    WRIST_DEPTH_FOLDER,
    WRIST_MASK_FOLDER,
    WRIST_RGB_FOLDER,
)
try:
    from rlbench.backend.const import MERGED_POINT_CLOUD_FOLDER
except ImportError:  # pragma: no cover - backward compatibility
    MERGED_POINT_CLOUD_FOLDER = "merged_point_cloud"
from rlbench.backend.utils import task_file_to_task_class
from rlbench.environment import Environment

from .segmentation_utils import (
    DEFAULT_MAP_FILENAME,
    build_handle_label_map,
    collect_mask_handles,
    write_label_map,
)
from .splits import RLBENCH18_TASKS, resolve_task_names


@dataclass(frozen=True)
class RawGenerationConfig:
    raw_root: Path
    tasks: tuple[str, ...]
    image_size: tuple[int, int] = (128, 128)
    renderer: str = "opengl3"
    processes: int = 1
    episodes_per_variation: int = 150
    variations: int = 1
    arm_max_velocity: float = 1.0
    arm_max_acceleration: float = 4.0


def _build_observation_config(cfg: RawGenerationConfig) -> ObservationConfig:
    obs_config = ObservationConfig()
    obs_config.set_all(True)
    img_size = list(map(int, cfg.image_size))
    obs_config.right_shoulder_camera.image_size = img_size
    obs_config.left_shoulder_camera.image_size = img_size
    obs_config.overhead_camera.image_size = img_size
    obs_config.wrist_camera.image_size = img_size
    obs_config.front_camera.image_size = img_size
    obs_config.right_shoulder_camera.depth_in_meters = False
    obs_config.left_shoulder_camera.depth_in_meters = False
    obs_config.overhead_camera.depth_in_meters = False
    obs_config.wrist_camera.depth_in_meters = False
    obs_config.front_camera.depth_in_meters = False
    obs_config.left_shoulder_camera.masks_as_one_channel = True
    obs_config.right_shoulder_camera.masks_as_one_channel = True
    obs_config.overhead_camera.masks_as_one_channel = True
    obs_config.wrist_camera.masks_as_one_channel = True
    obs_config.front_camera.masks_as_one_channel = True

    if str(cfg.renderer) == "opengl":
        render_mode = RenderMode.OPENGL
    elif str(cfg.renderer) == "opengl3":
        render_mode = RenderMode.OPENGL3
    else:
        raise ValueError(f"Unsupported renderer={cfg.renderer!r}")
    obs_config.right_shoulder_camera.render_mode = render_mode
    obs_config.left_shoulder_camera.render_mode = render_mode
    obs_config.overhead_camera.render_mode = render_mode
    obs_config.wrist_camera.render_mode = render_mode
    obs_config.front_camera.render_mode = render_mode
    return obs_config


def _make_environment(cfg: RawGenerationConfig) -> Environment:
    return Environment(
        action_mode=MoveArmThenGripper(JointVelocity(), Discrete()),
        obs_config=_build_observation_config(cfg),
        arm_max_velocity=float(cfg.arm_max_velocity),
        arm_max_acceleration=float(cfg.arm_max_acceleration),
        headless=True,
    )


def _estimate_total_episode_slots(tasks, cfg: RawGenerationConfig) -> int | None:
    if int(cfg.variations) < 0:
        return None
    return int(len(tasks) * int(cfg.variations) * int(cfg.episodes_per_variation))


def _check_and_make(path: str | Path) -> None:
    os.makedirs(path, exist_ok=True)


def _normalize_task_names(tasks: Sequence[str], *, use_all_tasks: bool) -> List[str]:
    task_files = sorted(
        t.replace(".py", "")
        for t in os.listdir(task_backend.TASKS_PATH)
        if t.endswith(".py") and t != "__init__.py"
    )
    if tasks:
        resolved = resolve_task_names(tasks)
    elif use_all_tasks:
        resolved = task_files
    else:
        resolved = list(RLBENCH18_TASKS)
    missing = [task for task in resolved if task not in task_files]
    if missing:
        raise ValueError(f"Unrecognized RLBench task names: {missing}")
    return resolved


def _save_demo(demo, example_path: str) -> None:
    left_shoulder_rgb_path = os.path.join(example_path, LEFT_SHOULDER_RGB_FOLDER)
    left_shoulder_depth_path = os.path.join(example_path, LEFT_SHOULDER_DEPTH_FOLDER)
    left_shoulder_mask_path = os.path.join(example_path, LEFT_SHOULDER_MASK_FOLDER)
    right_shoulder_rgb_path = os.path.join(example_path, RIGHT_SHOULDER_RGB_FOLDER)
    right_shoulder_depth_path = os.path.join(example_path, RIGHT_SHOULDER_DEPTH_FOLDER)
    right_shoulder_mask_path = os.path.join(example_path, RIGHT_SHOULDER_MASK_FOLDER)
    overhead_rgb_path = os.path.join(example_path, OVERHEAD_RGB_FOLDER)
    overhead_depth_path = os.path.join(example_path, OVERHEAD_DEPTH_FOLDER)
    overhead_mask_path = os.path.join(example_path, OVERHEAD_MASK_FOLDER)
    wrist_rgb_path = os.path.join(example_path, WRIST_RGB_FOLDER)
    wrist_depth_path = os.path.join(example_path, WRIST_DEPTH_FOLDER)
    wrist_mask_path = os.path.join(example_path, WRIST_MASK_FOLDER)
    front_rgb_path = os.path.join(example_path, FRONT_RGB_FOLDER)
    front_depth_path = os.path.join(example_path, FRONT_DEPTH_FOLDER)
    front_mask_path = os.path.join(example_path, FRONT_MASK_FOLDER)
    merged_point_cloud_path = os.path.join(example_path, MERGED_POINT_CLOUD_FOLDER)

    for path in (
        left_shoulder_rgb_path,
        left_shoulder_depth_path,
        left_shoulder_mask_path,
        right_shoulder_rgb_path,
        right_shoulder_depth_path,
        right_shoulder_mask_path,
        overhead_rgb_path,
        overhead_depth_path,
        overhead_mask_path,
        wrist_rgb_path,
        wrist_depth_path,
        wrist_mask_path,
        front_rgb_path,
        front_depth_path,
        front_mask_path,
        merged_point_cloud_path,
    ):
        _check_and_make(path)

    camera_names = ("left_shoulder", "right_shoulder", "overhead", "wrist", "front")

    for i, obs in enumerate(demo):
        left_shoulder_rgb = Image.fromarray(obs.left_shoulder_rgb)
        left_shoulder_depth = utils.float_array_to_rgb_image(obs.left_shoulder_depth, scale_factor=DEPTH_SCALE)
        left_shoulder_mask = Image.fromarray((obs.left_shoulder_mask * 255).astype(np.uint8))
        right_shoulder_rgb = Image.fromarray(obs.right_shoulder_rgb)
        right_shoulder_depth = utils.float_array_to_rgb_image(obs.right_shoulder_depth, scale_factor=DEPTH_SCALE)
        right_shoulder_mask = Image.fromarray((obs.right_shoulder_mask * 255).astype(np.uint8))
        overhead_rgb = Image.fromarray(obs.overhead_rgb)
        overhead_depth = utils.float_array_to_rgb_image(obs.overhead_depth, scale_factor=DEPTH_SCALE)
        overhead_mask = Image.fromarray((obs.overhead_mask * 255).astype(np.uint8))
        wrist_rgb = Image.fromarray(obs.wrist_rgb)
        wrist_depth = utils.float_array_to_rgb_image(obs.wrist_depth, scale_factor=DEPTH_SCALE)
        wrist_mask = Image.fromarray((obs.wrist_mask * 255).astype(np.uint8))
        front_rgb = Image.fromarray(obs.front_rgb)
        front_depth = utils.float_array_to_rgb_image(obs.front_depth, scale_factor=DEPTH_SCALE)
        front_mask = Image.fromarray((obs.front_mask * 255).astype(np.uint8))

        left_shoulder_rgb.save(os.path.join(left_shoulder_rgb_path, IMAGE_FORMAT % i))
        left_shoulder_depth.save(os.path.join(left_shoulder_depth_path, IMAGE_FORMAT % i))
        left_shoulder_mask.save(os.path.join(left_shoulder_mask_path, IMAGE_FORMAT % i))
        right_shoulder_rgb.save(os.path.join(right_shoulder_rgb_path, IMAGE_FORMAT % i))
        right_shoulder_depth.save(os.path.join(right_shoulder_depth_path, IMAGE_FORMAT % i))
        right_shoulder_mask.save(os.path.join(right_shoulder_mask_path, IMAGE_FORMAT % i))
        overhead_rgb.save(os.path.join(overhead_rgb_path, IMAGE_FORMAT % i))
        overhead_depth.save(os.path.join(overhead_depth_path, IMAGE_FORMAT % i))
        overhead_mask.save(os.path.join(overhead_mask_path, IMAGE_FORMAT % i))
        wrist_rgb.save(os.path.join(wrist_rgb_path, IMAGE_FORMAT % i))
        wrist_depth.save(os.path.join(wrist_depth_path, IMAGE_FORMAT % i))
        wrist_mask.save(os.path.join(wrist_mask_path, IMAGE_FORMAT % i))
        front_rgb.save(os.path.join(front_rgb_path, IMAGE_FORMAT % i))
        front_depth.save(os.path.join(front_depth_path, IMAGE_FORMAT % i))
        front_mask.save(os.path.join(front_mask_path, IMAGE_FORMAT % i))

        merged_points = []
        merged_colors = []
        merged_masks = []
        for name in camera_names:
            pc = getattr(obs, f"{name}_point_cloud", None)
            rgb = getattr(obs, f"{name}_rgb", None)
            mask_arr = getattr(obs, f"{name}_mask", None)
            if pc is None or rgb is None or mask_arr is None:
                continue
            pc = np.asarray(pc).reshape(-1, 3)
            rgb = np.asarray(rgb).reshape(-1, 3)
            mask_arr = np.asarray(mask_arr).reshape(-1)
            valid = np.isfinite(pc).all(axis=1)
            if not np.all(valid):
                pc = pc[valid]
                rgb = rgb[valid]
                mask_arr = mask_arr[valid]
            if pc.size == 0:
                continue
            merged_points.append(pc)
            merged_colors.append(rgb)
            merged_masks.append(mask_arr)

        if merged_points:
            merged_points_np = np.concatenate(merged_points, axis=0).astype(np.float32)
            merged_colors_np = np.concatenate(merged_colors, axis=0).astype(np.uint8)
            merged_masks_np = np.concatenate(merged_masks, axis=0).astype(np.int32)
        else:
            merged_points_np = np.empty((0, 3), dtype=np.float32)
            merged_colors_np = np.empty((0, 3), dtype=np.uint8)
            merged_masks_np = np.empty((0,), dtype=np.int32)

        np.savez_compressed(
            os.path.join(merged_point_cloud_path, f"{i}.npz"),
            points=merged_points_np,
            colors=merged_colors_np,
            masks=merged_masks_np,
        )

        obs.left_shoulder_rgb = None
        obs.left_shoulder_depth = None
        obs.left_shoulder_point_cloud = None
        obs.left_shoulder_mask = None
        obs.right_shoulder_rgb = None
        obs.right_shoulder_depth = None
        obs.right_shoulder_point_cloud = None
        obs.right_shoulder_mask = None
        obs.overhead_rgb = None
        obs.overhead_depth = None
        obs.overhead_point_cloud = None
        obs.overhead_mask = None
        obs.wrist_rgb = None
        obs.wrist_depth = None
        obs.wrist_point_cloud = None
        obs.wrist_mask = None
        obs.front_rgb = None
        obs.front_depth = None
        obs.front_point_cloud = None
        obs.front_mask = None

    with open(os.path.join(example_path, LOW_DIM_PICKLE), "wb") as file:
        pickle.dump(demo, file)


def _worker_run(
    worker_id,
    lock,
    task_index,
    variation_count,
    results,
    file_lock,
    progress_lock,
    progress_saved,
    progress_resolved,
    progress_skipped,
    tasks,
    cfg: RawGenerationConfig,
):
    np.random.seed(None)
    num_tasks = len(tasks)

    rlbench_env = _make_environment(cfg)
    rlbench_env.launch()

    task_errors = results[worker_id] = ""
    while True:
        with lock:
            if int(task_index.value) >= num_tasks:
                break

            current_variation = int(variation_count.value)
            current_task = tasks[int(task_index.value)]
            task_env = rlbench_env.get_task(current_task)
            variation_target = int(task_env.variation_count())
            if int(cfg.variations) >= 0:
                variation_target = min(int(cfg.variations), variation_target)
            if current_variation >= variation_target:
                variation_count.value = 0
                task_index.value += 1
                current_variation = 0
            if int(task_index.value) >= num_tasks:
                break
            current_task = tasks[int(task_index.value)]
            variation_count.value += 1

        task_env = rlbench_env.get_task(current_task)
        task_env.set_variation(current_variation)
        descriptions, _ = task_env.reset()

        variation_path = os.path.join(str(cfg.raw_root), task_env.get_name(), VARIATIONS_FOLDER % current_variation)
        _check_and_make(variation_path)
        with open(os.path.join(variation_path, VARIATION_DESCRIPTIONS), "wb") as file:
            pickle.dump(descriptions, file)

        episodes_path = os.path.join(variation_path, EPISODES_FOLDER)
        _check_and_make(episodes_path)

        abort_variation = False
        for ex_idx in range(int(cfg.episodes_per_variation)):
            attempts = 10
            while attempts > 0:
                try:
                    demo, = task_env.get_demos(amount=1, live_demos=True)
                except Exception as exc:
                    attempts -= 1
                    if attempts > 0:
                        continue
                    problem = (
                        f"worker {worker_id} failed collecting task {task_env.get_name()} "
                        f"(variation={current_variation}, example={ex_idx}). Skipping this task/variation.\n{exc}\n"
                    )
                    print(problem)
                    task_errors += problem
                    remaining_slots = int(cfg.episodes_per_variation) - int(ex_idx)
                    with progress_lock:
                        progress_resolved.value += int(remaining_slots)
                        progress_skipped.value += int(remaining_slots)
                    abort_variation = True
                    break

                episode_path = os.path.join(episodes_path, EPISODE_FOLDER % ex_idx)
                with file_lock:
                    _save_demo(demo, episode_path)
                with progress_lock:
                    progress_saved.value += 1
                    progress_resolved.value += 1
                break
            if abort_variation:
                break

        if abort_variation:
            continue

        variation_path_obj = Path(variation_path)
        handles = collect_mask_handles(variation_path_obj)
        if handles:
            try:
                task_env.reset()
            except Exception as exc:  # pragma: no cover - best effort
                print(
                    f"warning: failed to reset task {task_env.get_name()} variation {current_variation} "
                    f"before building label map: {exc}"
                )
            mapping, outstanding = build_handle_label_map(task_env, handles)
            if mapping:
                output_path = variation_path_obj / DEFAULT_MAP_FILENAME
                with file_lock:
                    write_label_map(output_path, mapping, overwrite=True)
            if outstanding:
                preview = ", ".join(str(h) for h in sorted(outstanding)[:10])
                print(
                    f"warning: {len(outstanding)} handles missing names for {task_env.get_name()} variation {current_variation}" +
                    (f": {preview}" if preview else "")
                )
        else:
            print(
                f"warning: no segmentation masks found for {task_env.get_name()} variation {current_variation}; "
                "skipping label map generation."
            )

    results[worker_id] = task_errors
    rlbench_env.shutdown()


def generate_raw_dataset(cfg: RawGenerationConfig) -> None:
    task_files = _normalize_task_names(cfg.tasks, use_all_tasks=False)
    tasks = [task_file_to_task_class(t) for t in task_files]
    total_episode_slots = _estimate_total_episode_slots(tasks, cfg)
    manager = Manager()
    result_dict = manager.dict()
    file_lock = manager.Lock()
    progress_lock = manager.Lock()
    task_index = manager.Value("i", 0)
    variation_count = manager.Value("i", 0)
    progress_saved = manager.Value("i", 0)
    progress_resolved = manager.Value("i", 0)
    progress_skipped = manager.Value("i", 0)
    lock = manager.Lock()

    _check_and_make(cfg.raw_root)
    processes = [
        Process(
            target=_worker_run,
            args=(
                worker_id,
                lock,
                task_index,
                variation_count,
                result_dict,
                file_lock,
                progress_lock,
                progress_saved,
                progress_resolved,
                progress_skipped,
                tasks,
                cfg,
            ),
        )
        for worker_id in range(int(cfg.processes))
    ]
    for proc in processes:
        proc.start()
    last_resolved = 0
    with tqdm(
        total=None if total_episode_slots is None else int(total_episode_slots),
        desc="raw-generate",
        unit="episode",
        dynamic_ncols=True,
        mininterval=0.5,
    ) as pbar:
        while True:
            resolved = int(progress_resolved.value)
            saved = int(progress_saved.value)
            skipped = int(progress_skipped.value)
            if resolved > last_resolved:
                pbar.update(resolved - last_resolved)
                last_resolved = resolved
            pbar.set_postfix(saved=saved, skipped=skipped, refresh=False)
            if not any(proc.is_alive() for proc in processes):
                break
            time.sleep(0.5)
        for proc in processes:
            proc.join()
        resolved = int(progress_resolved.value)
        saved = int(progress_saved.value)
        skipped = int(progress_skipped.value)
        if resolved > last_resolved:
            pbar.update(resolved - last_resolved)
        pbar.set_postfix(saved=saved, skipped=skipped, refresh=True)

    print("data collection done")
    for worker_id in range(int(cfg.processes)):
        print(result_dict[worker_id])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate raw RLBench point-cloud dataset export.")
    parser.add_argument("--raw-root", "--save-path", dest="raw_root", type=Path, required=True)
    parser.add_argument("--tasks", nargs="*", default=[])
    parser.add_argument("--all-tasks", action="store_true", help="Generate all RLBench tasks instead of the repo RLBench-18 default.")
    parser.add_argument("--image-size", nargs=2, type=int, default=[128, 128])
    parser.add_argument("--renderer", type=str, choices=["opengl", "opengl3"], default="opengl3")
    parser.add_argument("--processes", type=int, default=1)
    parser.add_argument("--episodes-per-variation", "--episodes-per-task", dest="episodes_per_variation", type=int, default=150)
    parser.add_argument("--variations", type=int, default=1, help="Number of variations per task. Use -1 for all available variations.")
    parser.add_argument("--arm-max-velocity", type=float, default=1.0)
    parser.add_argument("--arm-max-acceleration", type=float, default=4.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tasks = _normalize_task_names(args.tasks, use_all_tasks=bool(args.all_tasks))
    cfg = RawGenerationConfig(
        raw_root=args.raw_root.expanduser().resolve(),
        tasks=tuple(tasks),
        image_size=(int(args.image_size[0]), int(args.image_size[1])),
        renderer=str(args.renderer),
        processes=int(args.processes),
        episodes_per_variation=int(args.episodes_per_variation),
        variations=int(args.variations),
        arm_max_velocity=float(args.arm_max_velocity),
        arm_max_acceleration=float(args.arm_max_acceleration),
    )
    generate_raw_dataset(cfg)


if __name__ == "__main__":
    main()
