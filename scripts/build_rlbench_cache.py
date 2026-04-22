from __future__ import annotations

import argparse
from pathlib import Path

from supernode_tokenizer.data import build_cache_all_tasks, resolve_task_names


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build RLBench point-cloud cache for standard IL.")
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--tasks", type=str, default="")
    parser.add_argument("--num-points", type=int, default=4096)
    parser.add_argument("--num-workers", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tasks = [part.strip() for part in str(args.tasks).split(",") if part.strip()]
    build_cache_all_tasks(
        root_raw=args.raw_root.expanduser().resolve(),
        root_cache=args.cache_root.expanduser().resolve(),
        tasks=resolve_task_names(tasks) if tasks else None,
        N=int(args.num_points),
        num_workers=int(args.num_workers),
    )


if __name__ == "__main__":
    main()
