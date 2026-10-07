from __future__ import annotations

"""
Phase 1 real-Unity selection gate for WW_SelfPlay_HR.

Our edge over sibling branches: they select checkpoints by sim SFL stats; we
select by REAL dp.exe behaviour. This script runs a candidate player (left,
deployment-faithful via the leader's extractor + curve-fix) against a fixed
benchmark opponent (right) in real Unity, and reports:
  - real match score (candidate left vs opponent right)
  - candidate LEFT rotate_rate / dominant one-way rotation  <- Phase-2 safety metric
  - policy timing

It does NOT modify the training framework; it is a pure external evaluator that
reads an exported player .pt. Best-by-real-Unity replaces deploy-latest.
"""

import argparse
import json
import math
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from envs.opencv_state_extractor import (  # noqa: E402
    OpenCVStateExtractor,
    mirror_features_right_to_left_view,
    mirror_action,
)

from mlagents_envs.environment import UnityEnvironment  # noqa: E402
from mlagents_envs.envs.unity_parallel_env import UnityParallelEnv  # noqa: E402
from mlagents_envs.envs.custom_side_channel import CustomDataChannel, StringSideChannel  # noqa: E402
from mlagents_envs.exception import UnityCommunicatorStoppedException  # noqa: E402


def auto_click_start(delay_s: float, y_ratio: float = 0.875) -> None:
    import ctypes
    import ctypes.wintypes

    user32 = ctypes.windll.user32
    time.sleep(float(delay_s))
    hwnd = 0
    for _ in range(120):
        hwnd = user32.FindWindowW(None, "dp")
        if hwnd:
            break
        time.sleep(0.5)
    if not hwnd:
        print("[auto-start] dp window not found", flush=True)
        return
    user32.ShowWindow(hwnd, 5)
    user32.SetForegroundWindow(hwnd)
    time.sleep(0.25)
    client = ctypes.wintypes.RECT()
    user32.GetClientRect(hwnd, ctypes.byref(client))
    point = ctypes.wintypes.POINT(
        int((client.right - client.left) * 0.50),
        int((client.bottom - client.top) * float(y_ratio)),
    )
    user32.ClientToScreen(hwnd, ctypes.byref(point))
    print(f"[auto-start] clicking START at ({point.x}, {point.y})", flush=True)
    for _ in range(2):
        user32.SetCursorPos(point.x, point.y)
        time.sleep(0.08)
        user32.mouse_event(0x0002, 0, 0, 0, 0)
        user32.mouse_event(0x0004, 0, 0, 0, 0)
        time.sleep(0.25)


class SidePolicy:
    """Deployment-faithful policy wrapper (leader convention).

    side='left'  : native features -> model -> action (curve-fix optional).
    side='right' : if mirror=True, mirror features into the left view, run the
                   model, mirror the action back (for a left-trained model);
                   if mirror=False, feed native right features (for a model that
                   already plays right, e.g. the V67 side-aware router).
    """

    def __init__(self, model_path: str, side: str, mirror: bool, curve_fix: bool, force_no_rotation: bool = False):
        self.side = side
        self.mirror = bool(mirror)
        self.curve_fix = bool(curve_fix)
        self.force_no_rotation = bool(force_no_rotation)
        self.extractor = OpenCVStateExtractor(side=side)
        self.model = torch.jit.load(model_path, map_location="cpu")
        self.model.eval()
        self.action_counts = np.zeros((3, 3), dtype=np.int64)
        with torch.no_grad():
            dummy = torch.zeros(16, dtype=torch.float32)
            dummy[14] = 1.0 if side == "right" else -1.0
            _ = self.model(dummy)

    def reset(self):
        self.extractor.reset()

    def act(self, visual_obs) -> np.ndarray:
        native = self.extractor.features_from_image(visual_obs)
        if self.side == "right" and self.mirror:
            model_feat = mirror_features_right_to_left_view(native)
        else:
            model_feat = native
        with torch.no_grad():
            a = self.model(torch.as_tensor(model_feat, dtype=torch.float32)).cpu().numpy().astype(np.int32)
        a = np.clip(np.asarray(a[:3], dtype=np.int32), 0, 2)
        if self.side == "right" and self.mirror:
            a = np.clip(np.asarray(mirror_action(a), dtype=np.int32), 0, 2)
        # Rotation-disabled models (C line): untrained rotation head -> force 0.
        if self.force_no_rotation:
            a[2] = 0
        # Deployment curve-fix: ball invisible or tiny y-error -> no vertical move.
        if self.curve_fix:
            try:
                if float(native[12]) < 0.5 or abs(float(native[9])) < 0.08:
                    a[0] = 0
            except Exception:
                pass
        for branch, val in enumerate(a):
            if 0 <= int(val) < 3:
                self.action_counts[branch, int(val)] += 1
        return a

    def rotate_rate(self) -> float:
        tot = int(self.action_counts[2].sum())
        return float((self.action_counts[2, 1] + self.action_counts[2, 2]) / max(1, tot))

    def dominant_rotate_rate(self) -> float:
        tot = int(self.action_counts[2].sum())
        return float(max(int(self.action_counts[2, 1]), int(self.action_counts[2, 2])) / max(1, tot))


def infer_lr(agents):
    left = right = None
    for a in agents:
        s = str(a)
        if "PAgent1" in s:
            left = a
        elif "PAgent2" in s:
            right = a
    if left is None or right is None:
        left, right = agents[0], agents[1]
    return left, right


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--player-pt", required=True, help="candidate model, plays LEFT")
    ap.add_argument("--opponent-pt", required=True, help="benchmark opponent model, plays RIGHT")
    ap.add_argument("--opponent-mirror", action="store_true",
                    help="set if the opponent is a left-trained model that needs mirror to play right")
    ap.add_argument("--no-curve-fix", action="store_true", help="disable the deployment vertical curve-fix on the candidate")
    ap.add_argument("--player-no-rotation", action="store_true", help="force candidate (left) rotation action to 0 (for C rotation-disabled models with untrained rotation head)")
    ap.add_argument("--env-path", default=r"D:\pickleball\dPickleball BuildFiles\Competition\Windows\dp.exe")
    ap.add_argument("--serve-code", type=int, default=212)
    ap.add_argument("--target-score", type=float, default=21.0)
    ap.add_argument("--steps", type=int, default=40000)
    ap.add_argument("--worker-id", type=int, default=70)
    ap.add_argument("--base-port", type=int, default=5005)
    ap.add_argument("--no-auto-start", action="store_true")
    ap.add_argument("--start-delay", type=float, default=15.0)
    ap.add_argument("--log-every", type=int, default=250)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    left = SidePolicy(args.player_pt, side="left", mirror=False, curve_fix=not args.no_curve_fix, force_no_rotation=args.player_no_rotation)
    right = SidePolicy(args.opponent_pt, side="right", mirror=args.opponent_mirror, curve_fix=False)

    if not args.no_auto_start:
        threading.Thread(target=auto_click_start, args=(args.start_delay,), daemon=True).start()

    sc = StringSideChannel()
    ch = CustomDataChannel()
    ch.send_data(serve=int(args.serve_code), p1=0, p2=0)
    unity = UnityEnvironment(
        file_name=str(args.env_path),
        worker_id=int(args.worker_id),
        base_port=int(args.base_port),
        side_channels=[sc, ch],
        no_graphics=False,
    )
    env = UnityParallelEnv(unity)

    score = {}
    stopped = False
    steps_done = 0
    tL, tR = [], []
    try:
        obs = env.reset()
        la, ra = infer_lr(env.agents)
        score = {la: 0.0, ra: 0.0}
        print(f"[gate] left={Path(args.player_pt).name} vs right={Path(args.opponent_pt).name} "
              f"(opp_mirror={args.opponent_mirror}) agents=({la},{ra})", flush=True)
        for step in range(int(args.steps)):
            vis = obs[env.agents[0]]["observation"][0]
            t0 = time.perf_counter()
            al = left.act(vis)
            t1 = time.perf_counter()
            ar = right.act(vis)
            t2 = time.perf_counter()
            tL.append((t1 - t0) * 1000.0)
            tR.append((t2 - t1) * 1000.0)
            actions = {la: np.asarray(al, dtype=np.int32), ra: np.asarray(ar, dtype=np.int32)}
            obs, rewards, dones, infos = env.step(actions)
            score[la] += float(rewards[la])
            score[ra] += float(rewards[ra])
            steps_done = step + 1
            if step % args.log_every == 0:
                print(f"[{step:05d}] L={score[la]:.0f} R={score[ra]:.0f}", flush=True)
            if max(score[la], score[ra]) >= args.target_score:
                print("[gate] target score reached", flush=True)
                break
            if dones[env.agents[0]] or dones[env.agents[1]]:
                break
    except UnityCommunicatorStoppedException:
        stopped = True
    finally:
        try:
            env.close()
        except Exception:
            pass

    la_key = [k for k in score if "PAgent1" in str(k)] or list(score)[:1]
    ra_key = [k for k in score if "PAgent2" in str(k)] or list(score)[1:2]
    player_score = float(score.get(la_key[0], 0.0)) if la_key else 0.0
    opp_score = float(score.get(ra_key[0], 0.0)) if ra_key else 0.0

    result = {
        "player_pt": str(args.player_pt),
        "opponent_pt": str(args.opponent_pt),
        "opponent_mirror": bool(args.opponent_mirror),
        "curve_fix": not args.no_curve_fix,
        "player_no_rotation": bool(args.player_no_rotation),
        "steps": int(steps_done),
        "player_score": player_score,
        "opponent_score": opp_score,
        "player_won": bool(player_score > opp_score),
        "player_rotate_rate": left.rotate_rate(),
        "player_dominant_rotate_rate": left.dominant_rotate_rate(),
        "player_action_counts": left.action_counts.tolist(),
        "left_ms_mean": float(np.mean(tL)) if tL else 0.0,
        "left_ms_max": float(np.max(tL)) if tL else 0.0,
        "communicator_stopped": stopped,
    }
    text = json.dumps(result, indent=2)
    print(text, flush=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
