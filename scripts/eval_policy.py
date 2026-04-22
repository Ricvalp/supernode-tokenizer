from __future__ import annotations

from absl import app
from ml_collections.config_flags import config_flags

from supernode_tokenizer.eval.eval_rlbench import evaluate_policy

_CONFIG = config_flags.DEFINE_config_file(
    "config",
    default="supernode-tokenizer/configs/eval_policy.py",
    help_string="Path to a ml_collections config file.",
)


def main(argv):
    del argv
    evaluate_policy(_CONFIG.value)


if __name__ == "__main__":
    app.run(main)
