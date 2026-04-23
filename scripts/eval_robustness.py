from __future__ import annotations

import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from absl import app
from ml_collections.config_flags import config_flags

from supernode_tokenizer.eval.eval_rlbench import evaluate_robustness

_CONFIG = config_flags.DEFINE_config_file(
    "config",
    default=str(_REPO_ROOT / "configs" / "eval_robustness.py"),
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
