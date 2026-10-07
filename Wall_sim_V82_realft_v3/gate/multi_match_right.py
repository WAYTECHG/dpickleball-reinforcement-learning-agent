"""Right-side confirmation for submission v2: our 1600k MIRROR plays RIGHT vs router LEFT.

realunity_gate convention: player=LEFT, opponent=RIGHT. So here player=router (left),
opponent=our v2 model_right.pt (right, self-contained mirror, no extra mirror flag).
Our mirror WINS when opponent_score > player_score. serve 212 (deployment condition).
Opponent gets NO curve-fix in the gate, which MATCHES teamY deployment (also no curve-fix).
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
# argv[1]=right-mirror .pt to test, argv[2]=tag (default = v2 1600k mirror)
MIRROR = sys.argv[1] if len(sys.argv) > 1 else r"D:\pickleball\Smash_Potato_HR_v2\model_right.pt"
TAG = sys.argv[2] if len(sys.argv) > 2 else "v2"
OUT = ROOT / "logs" / ("multi_match_right_" + TAG)
OUT.mkdir(parents=True, exist_ok=True)
SUMMARY = OUT / "running_summary.txt"

N = 3
SERVE = 212
wid = 120 + (10 if TAG != "v2" else 0)
games = []  # (ours_right_score, router_left_score)


def dump():
    wins = sum(1 for s in games if s[0] > s[1])
    detail = " ".join(f"{int(s[0])}:{int(s[1])}" for s in games)
    avg_us = sum(s[0] for s in games) / len(games) if games else 0
    avg_op = sum(s[1] for s in games) / len(games) if games else 0
    text = (f"RIGHT-SIDE (v2 model_right = 1600k mirror) vs router\n"
            f"wins={wins}/{len(games)}  avg ours:router={avg_us:.1f}:{avg_op:.1f}  "
            f"games(ours:router)=[{detail}]")
    SUMMARY.write_text(text + "\n", encoding="utf-8")
    print("\n" + text + "\n", flush=True)


for g in range(N):
    outj = OUT / f"right_g{g}.json"
    r = None
    for attempt in range(3):
        cmd = [PY, str(ROOT / "realunity_gate.py"),
               "--player-pt", ROUTER, "--opponent-pt", MIRROR, "--env-path", ENV,
               "--serve-code", str(SERVE), "--steps", "40000", "--target-score", "21",
               "--worker-id", str(wid), "--out", str(outj)]
        wid += 1
        print(f"\n==== right game {g+1}/{N} (attempt {attempt+1}) ====", flush=True)
        subprocess.run(cmd)
        if outj.exists():
            try:
                r = json.loads(outj.read_text(encoding="utf-8"))
            except Exception:
                r = None
        if r and max(r.get("player_score", 0), r.get("opponent_score", 0)) >= 21:
            break
        time.sleep(3)
    if r is not None:
        games.append((r.get("opponent_score", 0), r.get("player_score", 0)))
    dump()
    time.sleep(2)

print("\n==== DONE ====", flush=True)
dump()
