"""Multi-match real-Unity verification of the top gate candidates vs router.

serve-code 212 (match-point 21, right serves) == deployment/competition condition;
Unity is not bit-deterministic so repeats give genuine game-to-game variance.
Round-robin order (baseline, 900k, 1600k per round) cancels any time drift.
Auto-retries a game that didn't reach 21 (START click missed / timed out).
"""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable
ENV = r"D:\pickleball\dPickleball BuildFiles\Competition\Windows\dp.exe"
ROUTER = r"D:\pickleball\Wall_sim_V67_FT\pretrained\v54_v67_originals\best_2_policy.pt"
OUT = ROOT / "logs" / "multi_match"
OUT.mkdir(parents=True, exist_ok=True)
SUMMARY = OUT / "running_summary.txt"

CANDS = {
    "realft_baseline": r"D:\pickleball\Smash_Potato_HR_v1_LOCKED\model_left.pt",
    "pfsp_900k": r"D:\pickleball\Wall_sim_V82_realft\checkpoints\realft_00900004_policy.pt",
    "pfsp_1600k": r"D:\pickleball\Wall_sim_V82_realft\checkpoints\realft_01600004_policy.pt",
}
N = 4
SERVE = 212

results = {k: [] for k in CANDS}
wid = 90


def run_game(name, path, rnd):
    global wid
    outj = OUT / f"{name}_r{rnd}.json"
    r = None
    for attempt in range(3):
        cmd = [PY, str(ROOT / "realunity_gate.py"),
               "--player-pt", path, "--opponent-pt", ROUTER, "--env-path", ENV,
               "--serve-code", str(SERVE), "--steps", "40000", "--target-score", "21",
               "--worker-id", str(wid), "--out", str(outj)]
        wid += 1
        subprocess.run(cmd)
        if outj.exists():
            try:
                r = json.loads(outj.read_text(encoding="utf-8"))
            except Exception:
                r = None
        if r and max(r.get("player_score", 0), r.get("opponent_score", 0)) >= 21:
            return r           # completed normally
        print(f"  [retry] {name} r{rnd} did not reach 21 (attempt {attempt+1})", flush=True)
        time.sleep(3)
    return r


def dump_summary():
    lines = ["==== MULTI-MATCH RUNNING SUMMARY (serve 212, vs router) ===="]
    for name in CANDS:
        g = results[name]
        if not g:
            lines.append(f"{name:<18} (no games yet)")
            continue
        wins = sum(1 for s in g if s[0] > s[1])
        ap = sum(s[0] for s in g) / len(g)
        ao = sum(s[1] for s in g) / len(g)
        detail = " ".join(f"{int(s[0])}:{int(s[1])}" for s in g)
        rot = max((s[2] for s in g), default=0.0)
        lines.append(f"{name:<18} wins={wins}/{len(g)}  avg={ap:.1f}:{ao:.1f}  maxrot={rot:.3f}  games=[{detail}]")
    text = "\n".join(lines)
    SUMMARY.write_text(text + "\n", encoding="utf-8")
    print("\n" + text + "\n", flush=True)


for rnd in range(N):
    for name, path in CANDS.items():
        print(f"\n==== round {rnd+1}/{N}  {name} ====", flush=True)
        r = run_game(name, path, rnd)
        if r is not None:
            results[name].append((r.get("player_score", 0), r.get("opponent_score", 0),
                                  r.get("player_rotate_rate", 0.0)))
        dump_summary()
        time.sleep(2)

print("\n==== DONE ====", flush=True)
dump_summary()
