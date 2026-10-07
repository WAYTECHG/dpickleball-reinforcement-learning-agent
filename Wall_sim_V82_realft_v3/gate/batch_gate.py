from __future__ import annotations

"""
Batch real-Unity gate: run a list of candidate .pt models sequentially through
realunity_gate.py (one Unity instance at a time, auto-clicked), collect results,
print a ranking. Unity must be serial (one window / one auto-click at a time),
so this loops subprocess calls rather than parallelizing.

Usage:
  python batch_gate.py --candidates "a.pt,b.pt,..." --opponent trainer_pool_all_pt\\V67_checkpoints_best_2_policy.pt
  python batch_gate.py --glob "current_models\\V54_best2\\snapshots\\*.pt" --opponent ...
"""

import argparse
import glob as globmod
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="", help="comma-separated .pt paths (relative to branch root or absolute)")
    ap.add_argument("--glob", default="", help="glob for candidate .pt files")
    ap.add_argument("--opponent", default=r"trainer_pool_all_pt\V67_checkpoints_best_2_policy.pt")
    ap.add_argument("--opponent-mirror", action="store_true")
    ap.add_argument("--player-no-rotation", action="store_true", help="force candidate rotation to 0 (C rotation-disabled models)")
    ap.add_argument("--env-path", default=r"D:\pickleball\dPickleball BuildFiles\Competition\Windows\dp.exe")
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--target-score", type=float, default=21.0)
    ap.add_argument("--base-worker-id", type=int, default=73)
    ap.add_argument("--out-dir", default=r"logs\batch_gate")
    args = ap.parse_args()

    cands: list[str] = []
    if args.glob:
        cands += sorted(globmod.glob(str((ROOT / args.glob)) if not Path(args.glob).is_absolute() else args.glob))
    if args.candidates:
        for c in args.candidates.split(","):
            c = c.strip().strip('"')
            if c:
                cands.append(c if Path(c).is_absolute() else str(ROOT / c))
    # de-dup preserve order
    seen = set(); ordered = []
    for c in cands:
        if c not in seen:
            seen.add(c); ordered.append(c)
    if not ordered:
        print("no candidates"); return 2

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "summary.json"
    summary = []
    if summary_path.exists():
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            summary = []

    print(f"[batch] {len(ordered)} candidates vs {Path(args.opponent).name}", flush=True)
    for i, cand in enumerate(ordered):
        tag = Path(cand).stem
        out_json = out_dir / f"{tag}.json"
        wid = args.base_worker_id + i
        print(f"\n[batch {i+1}/{len(ordered)}] {tag} (worker {wid})", flush=True)
        cmd = [
            PY, str(ROOT / "realunity_gate.py"),
            "--player-pt", cand,
            "--opponent-pt", args.opponent if Path(args.opponent).is_absolute() else str(ROOT / args.opponent),
            "--env-path", args.env_path,
            "--steps", str(args.steps),
            "--target-score", str(args.target_score),
            "--worker-id", str(wid),
            "--out", str(out_json),
        ]
        if args.opponent_mirror:
            cmd.append("--opponent-mirror")
        if args.player_no_rotation:
            cmd.append("--player-no-rotation")
        rc = subprocess.run(cmd).returncode
        rec = {"candidate": cand, "tag": tag, "rc": rc}
        if out_json.exists():
            try:
                r = json.loads(out_json.read_text(encoding="utf-8"))
                rec.update({
                    "player_score": r.get("player_score"),
                    "opponent_score": r.get("opponent_score"),
                    "won": r.get("player_won"),
                    "rotate_rate": r.get("player_rotate_rate"),
                    "dominant_rotate": r.get("player_dominant_rotate_rate"),
                    "steps": r.get("steps"),
                })
            except Exception as e:
                rec["parse_error"] = repr(e)
        summary = [s for s in summary if s.get("tag") != tag] + [rec]
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        time.sleep(2)

    print("\n==== RANKING (by player_score, then low rotate) ====", flush=True)
    ranked = sorted(
        [s for s in summary if s.get("player_score") is not None],
        key=lambda s: (-(s.get("player_score") or 0), s.get("rotate_rate") or 1.0),
    )
    for s in ranked:
        print(f"  {s['tag']:<28} {s.get('player_score'):>4}:{s.get('opponent_score'):<4} "
              f"rot={s.get('rotate_rate'):.3f} dom={s.get('dominant_rotate'):.3f} won={s.get('won')}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
