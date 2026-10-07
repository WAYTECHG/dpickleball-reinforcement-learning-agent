from __future__ import annotations

"""
Robust unattended overnight runner for real-Unity pool fine-tuning.

Fixes the two failure modes that killed previous overnight runs:
  1. Screen lock / display-off / sleep suspends dp.exe rendering -> env.step hangs.
     -> keep_awake() via SetThreadExecutionState (ES_DISPLAY_REQUIRED|ES_SYSTEM_REQUIRED)
        keeps the display on and the system awake for the whole run.
  2. Any hang or crash loses all progress since the last checkpoint.
     -> watchdog: monitors logs/ckpt_log.txt; if no new checkpoint within
        STALL_MIN minutes, kills the training subprocess + dp.exe and relaunches
        from the LATEST checkpoint (weights + num_timesteps continue).

Run this (not realft_train.py directly) for overnight runs.
"""

import ctypes
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable
CKPT_LOG = ROOT / "logs" / "ckpt_log.txt"
CKPT_DIR = ROOT / "checkpoints"
R2_ZIP = CKPT_DIR / "r2" / "realft_00300000.zip"

TARGET_STEPS = 1_600_000   # overnight PFSP run from ~675k (~9h at ~33 fps); self-stops here
SAVE_FREQ = 25_000
STALL_MIN = 22.0          # no new checkpoint in this many minutes => hung
                          # (first checkpoint is ~17 min away at 25k steps / ~26 fps,
                          #  so 22 min avoids false-killing a healthy run)
POLL_SEC = 60.0
OPP_POOL = ROOT / "opp_pool"
PID_FILE = ROOT / "logs" / "watchdog.pid"
DONE_FILE = ROOT / "logs" / "watchdog_done.txt"


def write_status(name: str, msg: str) -> None:
    """Cross-session liveness beacon. Lets ANY later Claude session verify the
    real state (pid alive + heartbeat fresh + ckpt advancing) without relying on
    the harness task list, which is session-scoped and dies on context compaction.
    This watchdog is launched detached from the harness job, so the harness can no
    longer track it -- these files are how we track it instead."""
    try:
        (ROOT / "logs").mkdir(parents=True, exist_ok=True)
        (ROOT / "logs" / name).write_text(
            f"{time.strftime('%Y-%m-%d %H:%M:%S')} pid={os.getpid()} {msg}\n",
            encoding="utf-8")
    except Exception:
        pass


def keep_awake():
    ES_CONTINUOUS = 0x80000000
    ES_SYSTEM_REQUIRED = 0x00000001
    ES_DISPLAY_REQUIRED = 0x00000002
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(
            ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_DISPLAY_REQUIRED)
        print("[watchdog] keep-awake set (display+system required)", flush=True)
    except Exception as e:
        print(f"[watchdog] keep-awake failed: {e}", flush=True)


def best_power_settings():
    for args in (["/change", "monitor-timeout-ac", "0"],
                 ["/change", "standby-timeout-ac", "0"],
                 ["/change", "monitor-timeout-dc", "0"],
                 ["/change", "standby-timeout-dc", "0"]):
        try:
            subprocess.run(["powercfg"] + args, capture_output=True, timeout=15)
        except Exception:
            pass


def latest_resume_zip() -> tuple[str, int]:
    best, beststep = None, -1
    for z in CKPT_DIR.glob("realft_0*.zip"):
        m = re.search(r"realft_(\d+)\.zip", z.name)
        if m:
            s = int(m.group(1))
            if s > beststep:
                beststep, best = s, str(z)
    if best is None:
        return str(R2_ZIP), 300_000
    return best, beststep


def last_ckpt_step() -> int:
    if not CKPT_LOG.exists():
        return -1
    try:
        for ln in reversed(CKPT_LOG.read_text(encoding="utf-8").splitlines()):
            m = re.search(r"steps=(\d+)", ln)
            if m:
                return int(m.group(1))
    except Exception:
        return -1
    return -1


def kill_dp():
    try:
        subprocess.run(["taskkill", "/F", "/IM", "dp.exe"], capture_output=True, timeout=20)
    except Exception:
        pass


def main():
    keep_awake()
    best_power_settings()
    (ROOT / "logs").mkdir(parents=True, exist_ok=True)
    PID_FILE.write_text(str(os.getpid()), encoding="utf-8")
    if DONE_FILE.exists():
        DONE_FILE.unlink()
    write_status("heartbeat.txt", "watchdog started (detached)")
    worker = 63
    attempt = 0
    while True:
        keep_awake()  # re-assert each loop
        cur = last_ckpt_step()
        write_status("heartbeat.txt", f"outer-loop step={cur} attempt={attempt}")
        if cur >= TARGET_STEPS:
            print(f"[watchdog] target {TARGET_STEPS} reached (at {cur}). Done.", flush=True)
            break
        resume_zip, resume_step = latest_resume_zip()
        attempt += 1
        worker += 1
        print(f"[watchdog] launch #{attempt}: resume={Path(resume_zip).name} (~{resume_step}) "
              f"target={TARGET_STEPS} worker={worker}", flush=True)
        proc = subprocess.Popen([
            PY, str(ROOT / "realft_train.py"),
            "--resume-zip", resume_zip,
            "--continue-timesteps",
            "--opponent-pool-dir", str(OPP_POOL),
            "--learner-side", "left",
            "--serve-code", "999993",
            "--total-steps", str(TARGET_STEPS),
            "--save-freq", str(SAVE_FREQ),
            "--critic-warmup", "0",
            "--lr", "5e-5",
            "--worker-id", str(worker),
            "--pfsp",
        ], cwd=str(ROOT))

        # monitor
        last_seen_step = last_ckpt_step()
        last_progress_t = time.time()
        while True:
            time.sleep(POLL_SEC)
            keep_awake()
            write_status("heartbeat.txt", f"training step={last_ckpt_step()} attempt={attempt} worker={worker}")
            if proc.poll() is not None:
                print(f"[watchdog] training process exited (code={proc.returncode}); will relaunch", flush=True)
                break
            step_now = last_ckpt_step()
            if step_now > last_seen_step:
                last_seen_step = step_now
                last_progress_t = time.time()
                if step_now >= TARGET_STEPS:
                    print(f"[watchdog] target reached at {step_now}", flush=True)
                    try:
                        proc.terminate()
                    except Exception:
                        pass
                    break
            elif (time.time() - last_progress_t) > STALL_MIN * 60.0:
                print(f"[watchdog] STALL: no checkpoint in {STALL_MIN} min (stuck at {step_now}). "
                      f"Killing + relaunching from latest.", flush=True)
                try:
                    proc.terminate()
                    time.sleep(3)
                    if proc.poll() is None:
                        proc.kill()
                except Exception:
                    pass
                kill_dp()
                time.sleep(5)
                break

        if last_ckpt_step() >= TARGET_STEPS:
            break
        time.sleep(3)

    write_status("watchdog_done.txt", f"finished at step={last_ckpt_step()}")
    try:
        PID_FILE.unlink()
    except Exception:
        pass
    print("[watchdog] finished.", flush=True)


if __name__ == "__main__":
    main()
