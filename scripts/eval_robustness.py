from __future__ import annotations

import json
from pathlib import Path

from absl import app
from ml_collections.config_flags import config_flags

from supernode_tokenizer.eval.eval_rlbench import evaluate_robustness

_CONFIG = config_flags.DEFINE_config_file(
    "config",
    default="supernode-tokenizer/configs/eval_robustness.py",
    help_string="Path to a ml_collections config file.",
)


def main(argv):
    del argv
    cfg = _CONFIG.value
    summary = evaluate_robustness(cfg)
    out_path = Path(str(cfg.output.root_dir)).expanduser().resolve() / "robustness_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)


if __name__ == "__main__":
    app.run(main)
