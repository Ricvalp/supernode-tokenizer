from __future__ import annotations

from typing import Any

from .build_dense_cache_per_variation import build_cache, build_cache_all_tasks, discover_tasks
from .cache_variation_h5 import CacheSpec
from .collate_standard_il import StandardILCollator
from .mask_utils import MASK_LABEL_MAP_FILENAME, resolve_mask_ids
from .rlbench_standard_il_dataset import RLBenchStandardILDataset, StandardILConfig, build_store, infer_state_action_dims
from .splits import GEOMETRY_SENSITIVE_TASKS, RLBENCH18_TASKS, EpisodeSplit, SplitSpec, resolve_task_names
from .variation_store import VariationKey, VariationStore, build_variation_keys

_RAW_IMPORT_ERROR: ImportError | None = None
try:
    from .generate_rlbench_raw_dataset import RawGenerationConfig, generate_raw_dataset
except ImportError as exc:  # pragma: no cover - depends on optional RLBench/PyRep install
    _RAW_IMPORT_ERROR = exc

    def _raise_raw_generation_import_error() -> None:
        raise ImportError(
            "Raw RLBench generation requires optional RLBench/PyRep dependencies. "
            "Use the cache-training path without importing raw generation helpers, "
            "or install RLBench and PyRep before using generate_raw_dataset."
        ) from _RAW_IMPORT_ERROR

    class RawGenerationConfig:  # type: ignore[no-redef]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs
            _raise_raw_generation_import_error()

    def generate_raw_dataset(*args: Any, **kwargs: Any) -> None:  # type: ignore[no-redef]
        del args, kwargs
        _raise_raw_generation_import_error()

__all__ = [
    "CacheSpec",
    "EpisodeSplit",
    "GEOMETRY_SENSITIVE_TASKS",
    "MASK_LABEL_MAP_FILENAME",
    "RLBENCH18_TASKS",
    "RLBenchStandardILDataset",
    "RawGenerationConfig",
    "SplitSpec",
    "StandardILCollator",
    "StandardILConfig",
    "VariationKey",
    "VariationStore",
    "build_cache",
    "build_cache_all_tasks",
    "build_store",
    "build_variation_keys",
    "discover_tasks",
    "generate_raw_dataset",
    "infer_state_action_dims",
    "resolve_mask_ids",
    "resolve_task_names",
]
