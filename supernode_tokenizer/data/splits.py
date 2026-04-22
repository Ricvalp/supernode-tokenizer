from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

RLBENCH18_TASK_ALIASES: Dict[str, str] = {
    "open_drawer": "open_drawer",
    "slide_block_to_color_target": "slide_block_to_target",
    "sweep_to_dustpan_of_size": "sweep_to_dustpan",
    "meat_off_grill": "meat_off_grill",
    "turn_tap": "turn_tap",
    "put_item_in_drawer": "put_item_in_drawer",
    "close_jar": "close_jar",
    "reach_and_drag": "reach_and_drag",
    "stack_blocks": "stack_blocks",
    "light_bulb_in": "light_bulb_in",
    "put_money_in_safe": "put_money_in_safe",
    "place_wine_at_rack_location": "stack_wine",
    "put_groceries_in_cupboard": "put_groceries_in_cupboard",
    "place_shape_in_shape_sorter": "place_shape_in_shape_sorter",
    "push_buttons": "push_buttons",
    "insert_onto_square_peg": "insert_onto_square_peg",
    "stack_cups": "stack_cups",
    "place_cups": "place_cups",
}

RLBENCH18_TASKS: Tuple[str, ...] = tuple(RLBENCH18_TASK_ALIASES.values())
GEOMETRY_SENSITIVE_TASKS: Tuple[str, ...] = (
    "open_drawer",
    "put_item_in_drawer",
    "close_jar",
    "turn_tap",
    "light_bulb_in",
    "stack_wine",
    "place_shape_in_shape_sorter",
    "insert_onto_square_peg",
)


@dataclass(frozen=True)
class EpisodeSplit:
    train: Tuple[int, ...]
    val: Tuple[int, ...]
    test: Tuple[int, ...]


@dataclass(frozen=True)
class SplitSpec:
    train_episodes: int = 100
    val_episodes: int = 25
    test_episodes: int = 25

    @property
    def total_required(self) -> int:
        return int(self.train_episodes + self.val_episodes + self.test_episodes)


def resolve_task_names(tasks: Sequence[str] | None) -> List[str]:
    if not tasks:
        return list(RLBENCH18_TASKS)
    resolved: List[str] = []
    for task in tasks:
        key = str(task).strip()
        if not key:
            continue
        resolved.append(RLBENCH18_TASK_ALIASES.get(key, key))
    return resolved


def split_episode_ids(episode_ids: Iterable[int], split_spec: SplitSpec) -> EpisodeSplit:
    ordered = tuple(sorted(int(eid) for eid in episode_ids))
    required = split_spec.total_required
    if len(ordered) < required:
        raise ValueError(
            f"Need at least {required} episodes for deterministic split, got {len(ordered)}."
        )
    t = int(split_spec.train_episodes)
    v = int(split_spec.val_episodes)
    s = int(split_spec.test_episodes)
    return EpisodeSplit(
        train=ordered[:t],
        val=ordered[t:t + v],
        test=ordered[t + v:t + v + s],
    )


def maybe_truncate_train_episode_ids(episode_ids: Sequence[int], max_train_episodes: int) -> Tuple[int, ...]:
    if int(max_train_episodes) <= 0:
        return tuple(int(eid) for eid in episode_ids)
    ordered = tuple(int(eid) for eid in episode_ids)
    return ordered[: int(max_train_episodes)]
