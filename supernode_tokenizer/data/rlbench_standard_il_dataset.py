from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import IterableDataset, get_worker_info

from .splits import SplitSpec, maybe_truncate_train_episode_ids, resolve_task_names, split_episode_ids
from .variation_store import VariationStore, build_variation_keys


@dataclass(frozen=True)
class StandardILConfig:
    T_obs: int
    H: int
    stride: int = 1
    split: str = "train"
    task_sampling: str = "variation_power"
    task_sampling_alpha: float = 0.5
    max_train_episodes_per_variation: int = -1
    train_episodes_per_variation: int = 100
    val_episodes_per_variation: int = 25
    test_episodes_per_variation: int = 25
    use_rgb: bool = False
    use_mask_id: bool = True

    def __post_init__(self) -> None:
        for name in ("T_obs", "H", "stride"):
            value = getattr(self, name)
            if not isinstance(value, Integral):
                raise TypeError(f"{name} must be an integer, got {type(value).__name__}.")
        if int(self.T_obs) < 1 or int(self.H) < 1 or int(self.stride) < 1:
            raise ValueError("T_obs, H, and stride must all be >= 1.")
        if str(self.split) not in {"train", "val", "test"}:
            raise ValueError("split must be one of: train, val, test.")
        if str(self.task_sampling) not in {"variation_power", "variation_uniform", "task_uniform"}:
            raise ValueError("task_sampling must be one of: variation_power, variation_uniform, task_uniform.")
        if float(self.task_sampling_alpha) < 0.0:
            raise ValueError("task_sampling_alpha must be >= 0.")


class RLBenchStandardILDataset(IterableDataset):
    def __init__(
        self,
        store: VariationStore,
        *,
        cfg: StandardILConfig,
        task_name_to_id: Dict[str, int],
        seed: int = 0,
        num_samples: Optional[int] = None,
        num_tries_per_item: int = 50,
    ):
        super().__init__()
        self.store = store
        self.cfg = cfg
        self.task_name_to_id = {str(k): int(v) for k, v in task_name_to_id.items()}
        self.seed = int(seed)
        self.num_samples = None if num_samples is None else int(num_samples)
        self.num_tries_per_item = int(num_tries_per_item)
        if self.num_tries_per_item < 1:
            raise ValueError("num_tries_per_item must be >= 1.")

        self.split_spec = SplitSpec(
            train_episodes=int(cfg.train_episodes_per_variation),
            val_episodes=int(cfg.val_episodes_per_variation),
            test_episodes=int(cfg.test_episodes_per_variation),
        )
        self._iter_counter = 0
        self._task_sampling_task_names: Optional[List[str]] = None
        self._task_sampling_vidx_by_task: Optional[List[np.ndarray]] = None
        self._task_sampling_probs: Optional[np.ndarray] = None
        self._eligible_episode_ids_by_vidx = self._build_episode_index()

    def _rng(self) -> np.random.Generator:
        wi = get_worker_info()
        wid = 0 if wi is None else wi.id
        return np.random.default_rng(self.seed + 10007 * wid + 1000003 * self._iter_counter)

    def _build_episode_index(self) -> List[np.ndarray]:
        out: List[np.ndarray] = []
        split_name = str(self.cfg.split)
        for vidx, key in enumerate(self.store.keys):
            episode_ids = self.store.list_episode_ids(vidx)
            split = split_episode_ids(episode_ids, self.split_spec)
            selected = getattr(split, split_name)
            if split_name == "train":
                selected = maybe_truncate_train_episode_ids(
                    selected,
                    int(self.cfg.max_train_episodes_per_variation),
                )
            out.append(np.asarray(selected, dtype=np.int64))
        return out

    def _build_task_sampling_index(self) -> bool:
        if self._task_sampling_probs is not None:
            return True
        task_to_vidx: Dict[str, List[int]] = {}
        for vidx, key in enumerate(self.store.keys):
            if self._eligible_episode_ids_by_vidx[vidx].size == 0:
                continue
            task = str(getattr(key, "task", ""))
            if task:
                task_to_vidx.setdefault(task, []).append(int(vidx))
        if not task_to_vidx:
            return False
        task_names = sorted(task_to_vidx)
        vidx_by_task = [np.asarray(task_to_vidx[task], dtype=np.int64) for task in task_names]
        mode = str(self.cfg.task_sampling)
        alpha = 0.0 if mode == "task_uniform" else float(self.cfg.task_sampling_alpha)
        counts = np.asarray([len(vidxs) for vidxs in vidx_by_task], dtype=np.float64)
        weights = counts if mode == "variation_uniform" else np.power(counts, alpha)
        total = float(weights.sum())
        if total <= 0.0 or not np.isfinite(total):
            weights = np.ones_like(counts)
            total = float(weights.sum())
        self._task_sampling_task_names = task_names
        self._task_sampling_vidx_by_task = vidx_by_task
        self._task_sampling_probs = weights / total
        return True

    def task_sampling_probabilities(self) -> Dict[str, float]:
        if not self._build_task_sampling_index():
            return {}
        assert self._task_sampling_task_names is not None
        assert self._task_sampling_probs is not None
        return {
            task: float(prob)
            for task, prob in zip(self._task_sampling_task_names, self._task_sampling_probs)
        }

    def _sample_vidx(self, rng: np.random.Generator) -> Optional[int]:
        if len(self.store) == 0:
            return None
        use_task_sampling = (
            str(self.cfg.task_sampling) != "variation_uniform"
            and self._build_task_sampling_index()
        )
        for _ in range(self.num_tries_per_item):
            if use_task_sampling:
                assert self._task_sampling_probs is not None
                assert self._task_sampling_vidx_by_task is not None
                task_idx = int(rng.choice(len(self._task_sampling_probs), p=self._task_sampling_probs))
                vidxs = self._task_sampling_vidx_by_task[task_idx]
                vidx = int(vidxs[int(rng.integers(0, len(vidxs)))])
            else:
                vidx = int(rng.integers(0, len(self.store)))
            if self._eligible_episode_ids_by_vidx[vidx].size > 0:
                return vidx
        return None

    def _sample_t0(self, episode_length: int, rng: np.random.Generator) -> Optional[int]:
        required_obs = 1 + (int(self.cfg.T_obs) - 1) * int(self.cfg.stride)
        max_t0 = int(episode_length) - required_obs
        if max_t0 < 0:
            return None
        return int(rng.integers(0, max_t0 + 1))

    def _build_indices(self, t0: int, episode_length: int) -> Tuple[np.ndarray, np.ndarray]:
        obs_idx = t0 + np.arange(0, int(self.cfg.T_obs) * int(self.cfg.stride), int(self.cfg.stride), dtype=np.int64)
        act_start = int(obs_idx[-1] + int(self.cfg.stride))
        act_idx = act_start + np.arange(0, int(self.cfg.H) * int(self.cfg.stride), int(self.cfg.stride), dtype=np.int64)
        act_idx = np.minimum(act_idx, int(episode_length) - 1)
        return obs_idx, act_idx

    def _load_item(self, vidx: int, episode_id: int, t0: int) -> Dict[str, Any]:
        key = self.store.keys[vidx]
        episode_length = int(self.store.episode_length(vidx, episode_id))
        obs_idx, act_idx = self._build_indices(t0, episode_length)
        obs = self.store.load_episode_slices(
            vidx,
            episode_id,
            obs_idx,
            load_rgb=bool(self.cfg.use_rgb),
            load_mask_id=bool(self.cfg.use_mask_id),
            load_full_traj=False,
        )
        act = self.store.load_episode_slices(
            vidx,
            episode_id,
            act_idx,
            load_rgb=False,
            load_mask_id=False,
            load_full_traj=False,
        )
        task_name = str(key.task)
        item: Dict[str, Any] = {
            "task_id": torch.tensor(self.task_name_to_id[task_name], dtype=torch.long),
            "obs_xyz": obs["xyz"],
            "obs_state": obs["state"],
            "obs_valid": obs["valid"],
            "target_action": act["action"],
            "meta": {
                "task": task_name,
                "variation": int(key.variation),
                "vidx": int(vidx),
                "episode_id": int(episode_id),
                "t0": int(t0),
            },
        }
        if bool(self.cfg.use_rgb) and "rgb" in obs:
            item["obs_rgb"] = obs["rgb"]
        if bool(self.cfg.use_mask_id) and "mask_id" in obs:
            item["obs_mask_id"] = obs["mask_id"]
        return item

    def __iter__(self):
        rng = self._rng()
        produced = 0
        target = self.num_samples
        self._iter_counter += 1
        while target is None or produced < target:
            vidx = self._sample_vidx(rng)
            if vidx is None:
                raise RuntimeError("Unable to sample a valid variation for the requested split.")
            episode_ids = self._eligible_episode_ids_by_vidx[vidx]
            episode_id = int(episode_ids[int(rng.integers(0, len(episode_ids)))])
            episode_length = int(self.store.episode_length(vidx, episode_id))
            t0 = self._sample_t0(episode_length, rng)
            if t0 is None:
                continue
            yield self._load_item(vidx, episode_id, t0)
            produced += 1


def discover_cached_tasks(cache_root: Path) -> List[str]:
    if not cache_root.is_dir():
        return []
    tasks: List[str] = []
    for path in sorted(cache_root.iterdir()):
        if path.is_dir() and any(path.glob("variation*.h5")):
            tasks.append(path.name)
    return tasks


def build_store(
    cache_root: Path,
    *,
    tasks: Optional[Sequence[str]] = None,
    keep_open_per_worker: bool = True,
) -> Tuple[VariationStore, List[str]]:
    if not cache_root.is_dir():
        raise FileNotFoundError(f"Cache root not found: {cache_root}")
    selected_tasks = resolve_task_names(tasks or discover_cached_tasks(cache_root))
    keys = []
    missing: List[str] = []
    for task in selected_tasks:
        task_keys = build_variation_keys(cache_root, task)
        if not task_keys:
            missing.append(task)
            continue
        keys.extend(task_keys)
    if missing:
        raise RuntimeError(f"No variation*.h5 files found for tasks: {', '.join(sorted(missing))}")
    if not keys:
        raise RuntimeError(f"No variation*.h5 files found under {cache_root}")
    return VariationStore(keys, keep_open_per_worker=keep_open_per_worker), selected_tasks


def infer_state_action_dims(store: VariationStore) -> Tuple[int, int]:
    for vidx in range(len(store)):
        episode_ids = store.list_episode_ids(vidx)
        if episode_ids.size == 0:
            continue
        episode_id = int(episode_ids[0])
        if int(store.episode_length(vidx, episode_id)) <= 0:
            continue
        sample = store.load_episode_slices(
            vidx=vidx,
            episode_id=episode_id,
            t_idx=np.asarray([0], dtype=np.int64),
            load_rgb=False,
            load_mask_id=False,
        )
        return int(sample["state"].shape[-1]), int(sample["action"].shape[-1])
    raise RuntimeError("Unable to infer state/action dims from cache.")
