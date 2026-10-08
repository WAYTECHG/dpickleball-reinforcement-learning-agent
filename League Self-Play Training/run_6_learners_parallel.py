from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from collections import deque

LEARNERS = [
    ("learner_01_zhiling", "learner_starts/learner_01_zhiling_16x9.zip", 2027, 0),
    ("learner_02_haoren", "learner_starts/learner_02_haoren_realft1600_lift16x9.zip", 2028, 0),
    ("learner_03_huayang", "learner_starts/learner_03_huayang_left1600_lift16x9.zip", 2029, 0),
    ("learner_04_jason", "learner_starts/learner_04_jason_16x9.zip", 2030, 0),
    ("learner_05_wilbert", "learner_starts/learner_05_wilbert_blocker_16x9.zip", 2031, 0),
    ("learner_06_best2_actor_init", "learner_starts/learner_06_best2_policy_actor_init_16x9.zip", 2032, 8000),
]

INTERESTING_PATTERNS = (
    "[chunk", "fps", "[done] final saved", "Traceback", "Exception", "UnityTimeOutException",
    "UnityEnvironmentException", "PermissionError", "[FAILED]", "[save]", "[critic warmup]",
    "Killed", "Error", "out of memory", "CUDA", "return code",
)


def _tail_interesting(path: Path, max_lines: int = 10) -> list[str]:
    if not path.exists():
        return ["log not created yet"]
    q: deque[str] = deque(maxlen=max_lines)
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                t = line.rstrip()
                if any(pat in t for pat in INTERESTING_PATTERNS):
                    q.append(t)
    except Exception as exc:
        return [f"could not read log: {exc}"]
    return list(q) or ["no training lines yet"]


def _resolve_start_zip(root: Path, name: str, initial_zip: str, args: argparse.Namespace) -> str:
    if args.start_source == "initial":
        return initial_zip
    if args.start_source == "latest":
        resume_root = Path(args.resume_root)
        if not resume_root.is_absolute():
            resume_root = root / resume_root
        candidate = resume_root / name / "checkpoints" / "ppo_real_latest.zip"
        if not candidate.exists():
            raise FileNotFoundError(
                f"--start-source latest requested, but latest zip not found for {name}: {candidate}"
            )
        return str(candidate)
    raise ValueError(f"Unknown start_source: {args.start_source}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Run six dPickleBall learners in parallel batches.")
    ap.add_argument("--max-parallel", type=int, default=2,
                    help="Recommended: 2 first, 3 if stable. 6 is a stress test and may overload the PC.")
    ap.add_argument("--steps", type=int, default=1_000_000)
    ap.add_argument("--chunk-steps", type=int, default=40_000)
    ap.add_argument("--output-root", default="outputs/six_learners_parallel",
                    help="Root folder where each learner output folder will be created.")
    ap.add_argument("--start-source", choices=["initial", "latest"], default="initial",
                    help="initial = use learner_starts/*.zip. latest = resume from --resume-root/<learner>/checkpoints/ppo_real_latest.zip")
    ap.add_argument("--resume-root", default="outputs/six_learners_tscale2_800k",
                    help="Output root to resume from when --start-source latest is used.")
    ap.add_argument("--no-graphics", action="store_true", help="Do not use for this build unless launch test passes. It crashed on the current Training build.")
    ap.add_argument("--low-res", action="store_true", help="Try Unity low-res args. This may be ignored by this ML-Agents/Unity build.")
    ap.add_argument("--monitor", action="store_true", help="Print summarized progress from each learner log while running.")
    ap.add_argument("--monitor-interval", type=int, default=60)
    ap.add_argument("--build-path", default=r"C:\Users\User\dpickleball-ml-agents\dPickleball BuildFiles\Training\Windows\dp.exe")
    ap.add_argument("--trainer-pool", default="league_pool_filtered",
                    help="Opponent pool folder. Default is the balanced 40-model runtime pool; use league_pool_all_submitted for the full 211-model member archive.")
    ap.add_argument("--time-scale", type=float, default=1.0,
                    help="Unity time_scale. time_scale=2 was stable/fast in your tests; too high may destabilize physics.")
    ap.add_argument("--target-frame-rate", type=int, default=-1,
                    help="Unity target frame rate via EngineConfigurationChannel. -1 = default.")
    ap.add_argument("--base-port", type=int, default=5005)
    ap.add_argument("--n-steps", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--n-epochs", type=int, default=3)
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--gae-lambda", type=float, default=0.95)
    ap.add_argument("--clip-range", type=float, default=0.20)
    ap.add_argument("--ent-coef", type=float, default=0.006)
    ap.add_argument("--vf-coef", type=float, default=0.50)
    ap.add_argument("--learning-rate", type=float, default=0.00015)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"])
    ap.add_argument("--num-envs", type=int, default=1,
                    help="Unity envs per learner. Keep this at 1 when launching many learners.")
    ap.add_argument("--save-every-chunks", type=int, default=5)
    ap.add_argument("--reset-optimizer", action="store_true",
                    help="Use for initial/lifted converted zips. Do NOT use for final fine-tuning from your own latest checkpoint unless desired.")
    ap.add_argument("--add-selfplay-to-pool", action="store_true")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent
    procs: list[tuple[str, subprocess.Popen, object, Path]] = []
    pending = list(LEARNERS)
    finished_ok = True
    next_worker = 0
    last_monitor = 0.0

    def launch(item, worker_id: int):
        name, initial_zip, seed, warmup = item
        start_zip = _resolve_start_zip(root, name, initial_zip, args)
        out_dir = str(Path(args.output_root) / name)
        log_dir = root / out_dir / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / "train_stdout.log"
        cmd = [
            sys.executable, "train_real_league_ppo.py",
            "--build-path", args.build_path,
            "--trainer-pool", args.trainer_pool,
            "--start-zip", start_zip,
            "--output-dir", out_dir,
            "--total-steps", str(args.steps),
            "--chunk-steps", str(args.chunk_steps),
            "--n-steps", str(args.n_steps),
            "--batch-size", str(args.batch_size),
            "--n-epochs", str(args.n_epochs),
            "--gamma", str(args.gamma),
            "--gae-lambda", str(args.gae_lambda),
            "--clip-range", str(args.clip_range),
            "--ent-coef", str(args.ent_coef),
            "--vf-coef", str(args.vf_coef),
            "--learning-rate", str(args.learning_rate),
            "--seed", str(seed),
            "--worker-id", str(worker_id),
            "--base-port", str(args.base_port),
            "--device", args.device,
            "--critic-warmup-steps", str(warmup if args.start_source == "initial" else 0),
            "--save-every-chunks", str(args.save_every_chunks),
            "--opponent-mode", "auto",
            "--timeout-wait", "240",
            "--num-envs", str(args.num_envs),
            "--time-scale", str(args.time_scale),
            "--target-frame-rate", str(args.target_frame_rate),
        ]
        if args.reset_optimizer:
            cmd.append("--reset-optimizer")
        if args.add_selfplay_to_pool:
            cmd.append("--add-selfplay-to-pool")
        if args.no_graphics:
            cmd.append("--no-graphics")
        if args.low_res:
            cmd.append("--low-res")
        f = open(log_path, "w", encoding="utf-8", buffering=1)
        print(f"[launch] {name} worker={worker_id} start={start_zip} log={log_path}", flush=True)
        p = subprocess.Popen(cmd, cwd=str(root), stdout=f, stderr=subprocess.STDOUT)
        return name, p, f, log_path

    while pending or procs:
        while pending and len(procs) < max(1, args.max_parallel):
            item = pending.pop(0)
            procs.append(launch(item, next_worker))
            next_worker += 1
            time.sleep(8)

        now = time.time()
        if args.monitor and now - last_monitor >= args.monitor_interval:
            last_monitor = now
            print("\n[monitor] current learner logs", flush=True)
            for name, p, _f, log_path in procs:
                status = "running" if p.poll() is None else f"exit={p.poll()}"
                print(f"--- {name} ({status}) ---", flush=True)
                for line in _tail_interesting(log_path):
                    print(line, flush=True)

        time.sleep(10)
        still = []
        for name, p, f, log_path in procs:
            ret = p.poll()
            if ret is None:
                still.append((name, p, f, log_path))
            else:
                f.close()
                if ret == 0:
                    print(f"[done] {name}", flush=True)
                else:
                    print(f"[FAILED] {name} exit={ret}; see {log_path}", flush=True)
                    finished_ok = False
        procs = still

    print("[all done]" if finished_ok else "[done with failures]", flush=True)
    return 0 if finished_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
