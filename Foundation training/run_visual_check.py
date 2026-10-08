# Optional visualisation launcher.
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Optional visual check for Foundation training.")
    parser.add_argument("--mode", choices=["env", "latest-policy"], default="env", help="Visual check mode. env = simple debug controller; latest-policy = watch checkpoints/latest.zip.")
    parser.add_argument("--level", type=int, default=1, help="Wall curriculum level to visualize.")
    parser.add_argument("--steps", type=int, default=2000, help="Number of visual steps.")
    parser.add_argument("--obs-source", type=str, default="state", choices=["state", "opencv"], help="Observation source.")
    parser.add_argument("--model", type=str, default="checkpoints/latest.zip", help="Model path for --mode latest-policy.")
    parser.add_argument("--reload-each-episode", action="store_true", default=True, help="Reload latest model after each episode in latest-policy mode.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent

    if args.mode == "env":
        cmd = [
            sys.executable,
            "-m", "tools.debug_render_env",
            "--level", str(args.level),
            "--steps", str(args.steps),
            "--obs-source", args.obs_source,
        ]
    else:
        cmd = [
            sys.executable,
            "-m", "tools.watch_latest_policy",
            "--model", args.model,
            "--level", str(args.level),
            "--steps", str(args.steps),
            "--obs-source", args.obs_source,
        ]
        if args.reload_each_episode:
            cmd.append("--reload-each-episode")

    print("[Foundation visual check] Terminal 2 optional visual command:")
    print(" ".join(f'\"{x}\"' if " " in x else x for x in cmd))
    return subprocess.call(cmd, cwd=str(root))


if __name__ == "__main__":
    raise SystemExit(main())
