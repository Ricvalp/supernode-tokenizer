from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from absl import app
from ml_collections.config_flags import config_flags

from supernode_tokenizer.trainers import train_chunk
from supernode_tokenizer.utils import cleanup_distributed

_CONFIG = config_flags.DEFINE_config_file(
    "config",
    default=str(_REPO_ROOT / "configs" / "train_chunk_policy.py"),
    help_string="Path to a ml_collections config file.",
)


def main(argv):
    del argv
    try:
        train_chunk(_CONFIG.value)
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    app.run(main)
