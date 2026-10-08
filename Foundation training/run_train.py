# Foundation training launcher.
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Foundation training for the Python wall warm-up curriculum.")
    parser.add_argument("--num-envs", type=int, default=8, help="Number of parallel Python wall environments.")
    parser.add_argument("--obs-source", type=str, default="state", choices=["state", "opencv"], help="Observation source used by training.")
    parser.add_argument("--fresh", action="store_true", default=True, help="Start a fresh training run. This is the default workflow.")
    parser.add_argument("--continue-training", action="store_true", help="Resume from checkpoints/latest.zip instead of starting fresh.")
    parser.add_argument("--level", type=int, default=None, help="Optional fixed curriculum level.")
    parser.add_argument("--with-inline-preview", action="store_true", help="Optional old V58 behaviour: show a preview before every training interval in the SAME terminal.")
    parser.add_argument("--preview-steps", type=int, default=500, help="Steps for --with-inline-preview.")
    parser.add_argument("--preview-delay-ms", type=int, default=20, help="OpenCV delay for --with-inline-preview.")
    parser.add_argument("--extra", nargs=argparse.REMAINDER, help="Extra args passed after --extra directly to train_python_wall_ppo.py.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    cmd = [
        sys.executable,
        str(root / "train_python_wall_ppo.py"),
        "--num-envs", str(args.num_envs),
        "--obs-source", args.obs_source,
    ]


    if not args.continue_training:
        cmd.append("--fresh")

    if args.level is not None:
        cmd.extend(["--level", str(args.level)])

    if args.with_inline_preview:
        cmd.extend([
            "--preview-each-iteration",
            "--preview-steps", str(args.preview_steps),
            "--preview-delay-ms", str(args.preview_delay_ms),
        ])

    if args.extra:
        cmd.extend(args.extra)

    print("[Foundation training] Terminal 1 training command:")
    print(" ".join(f'\"{x}\"' if " " in x else x for x in cmd))
    print("\n[Tip] Optional visualisation can run in another terminal:")
    print("      python run_visual_check.py --mode env --level 1 --steps 2000")
    print("      python run_visual_check.py --mode latest-policy --level 1 --steps 2000")
    return subprocess.call(cmd, cwd=str(root))


if __name__ == "__main__":
    raise SystemExit(main())
