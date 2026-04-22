from __future__ import annotations

from typing import Any, Dict, List

import torch


def _stack_if_tensor(values: List[Any]) -> Any:
    if not values:
        return values
    if torch.is_tensor(values[0]):
        return torch.stack(values, dim=0)
    return values


class StandardILCollator:
    def __call__(self, items: List[Dict[str, Any]]) -> Dict[str, Any]:
        if not items:
            raise ValueError("Cannot collate an empty batch.")
        keys = list(items[0].keys())
        batch: Dict[str, Any] = {}
        for key in keys:
            values = [item[key] for item in items]
            if key == "meta":
                batch[key] = values
            else:
                batch[key] = _stack_if_tensor(values)
        return batch
