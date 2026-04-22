from __future__ import annotations

from absl import app
from ml_collections.config_flags import config_flags

from supernode_tokenizer.trainers import train_chunk
from supernode_tokenizer.utils import cleanup_distributed

_CONFIG = config_flags.DEFINE_config_file(
    "config",
    default="supernode-tokenizer/configs/train_chunk_policy.py",
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
