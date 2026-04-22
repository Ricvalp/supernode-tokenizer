from __future__ import annotations

from .eval_policy import get_config as _get_eval_config


def get_config():
    return _get_eval_config()
