from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 1: fast league self-play training.")
    parser.add_argument("--max-parallel", type=int, default=6)
    parser.add_argument("--steps", type=int, default=800000)
    parser.add_argument("--trainer-pool", default="league_pool_filtered",
                        help="Opponent pool folder. Default is the balanced 40-model runtime pool. Use league_pool_all_submitted only for the full 211-model member archive.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    cmd = [
        sys.executable, "run_6_learners_parallel.py",
        "--max-parallel", str(args.max_parallel),
        "--trainer-pool", args.trainer_pool,
        "--output-root", "outputs/six_learners_tscale2_800k",
        "--steps", str(args.steps),
        "--chunk-steps", "20000",
        "--monitor",
        "--monitor-interval", "60",
        "--time-scale", "2",
        "--target-frame-rate", "-1",
        "--learning-rate", "0.00015",
        "--n-steps", "2048",
        "--batch-size", "512",
        "--n-epochs", "3",
        "--reset-optimizer",
        "--save-every-chunks", "5",
        "--add-selfplay-to-pool",
    ]
    return subprocess.call(cmd, cwd=str(root))


if __name__ == "__main__":
    raise SystemExit(main())
