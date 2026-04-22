from __future__ import annotations

from dataclasses import fields, is_dataclass
from typing import Any, Dict


def dataclass_from_dict(default_obj: Any, src: Dict[str, Any]) -> Any:
    if not is_dataclass(default_obj):
        return default_obj
    kwargs = {}
    for field in fields(default_obj):
        default_value = getattr(default_obj, field.name)
        if is_dataclass(default_value):
            child_src = src.get(field.name, {}) if isinstance(src, dict) else {}
            if not isinstance(child_src, dict):
                child_src = {}
            kwargs[field.name] = dataclass_from_dict(default_value, child_src)
        else:
            value = src.get(field.name, default_value) if isinstance(src, dict) else default_value
            if isinstance(default_value, tuple) and isinstance(value, (list, tuple)):
                value = tuple(value)
            kwargs[field.name] = value
    return type(default_obj)(**kwargs)
