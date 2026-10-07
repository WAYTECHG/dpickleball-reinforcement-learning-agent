from __future__ import annotations

import os
import sys
import threading
import time
from collections import deque
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.exception import UnityCommunicatorStoppedException, UnityTimeOutException
from mlagents_envs.envs.unity_parallel_env import UnityParallelEnv
from mlagents_envs.envs.custom_side_channel import CustomDataChannel, StringSideChannel

V67_FT_DIR = r"D:\pickleball\Wall_sim_V67_FT"
if V67_FT_DIR not in sys.path:
    sys.path.insert(0, V67_FT_DIR)
from envs.opencv_state_extractor import OpenCVStateExtractor  # noqa: E402

DROP_IDX = (6, 7, 11)   # sin, cos, success  -> 13-dim student layout
TIME_IDX_13 = 8         # original index 10 (time_own_side) after dropping


def reduce16to13(feat16: np.ndarray) -> np.ndarray:
    f = np.delete(np.asarray(feat16, dtype=np.float32), list(DROP_IDX))
    f[TIME_IDX_13] = 0.0
    return f


def auto_click_start(delay_s: float, y_ratio: float = 0.875) -> None:
    """dp.exe shows an intro animation and needs a START click in the GUI.
    Proven approach from tools/unity_fight_eval.py (used by the V80 automation):
    find the 'dp' window and click at (50% width, y_ratio height)."""
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
        print("[auto-start] dp window not found")
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
    print(f"[auto-start] clicking START at ({point.x}, {point.y})")
    for _ in range(2):
        user32.SetCursorPos(point.x, point.y)
        time.sleep(0.08)
        user32.mouse_event(0x0002, 0, 0, 0, 0)
        user32.mouse_event(0x0004, 0, 0, 0, 0)
        time.sleep(0.25)


class RealUnityFightEnv(gym.Env):
    """Real-Unity training env (round 3: dual-side learner + opponent pool).

    - learner_side: "left", "right", or "alternate" (re-sampled every point).
      The 13-dim feature layout contains side_sign, so one network plays both
      sides; rotation stays out of the action space entirely.
    - Opponent pool per point: the frozen V67 router (anchor, prob
      1-snapshot_prob) or the learner's own latest exported snapshot
      (self-play, prob snapshot_prob, hot-reloaded when the file changes).
    - Observation: official shared agent0 image -> OpenCVStateExtractor for the
      learner's side -> 16 features -> 13-dim student layout (deployment-exact).
    - Reward: own point - opponent point (the real +-1), plus tiny real-event
      shaping (own hit / ball crossing to the opponent half) detected from the
      image stream with thresholds above measured detection jitter.
    - Lifecycle-agnostic: high match point (serve 9993), every scored point is a
      soft episode; app quit at match end triggers transparent relaunch with an
      automatic START click (the build needs a GUI click after the intro).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        env_path: str = r"D:\pickleball\dPickleball BuildFiles\Competition\Windows\dp.exe",
        opponent_model: str = str(Path(V67_FT_DIR) / "pretrained" / "v54_v67_originals" / "best_2_policy.pt"),
        opponent_pool_dir: str = "",
        snapshot_path: str = "",
        snapshot_prob: float = 0.5,
        learner_side: str = "left",
        serve_code: int = 9993,
        worker_id: int = 60,
        base_port: int = 5005,
        graphics: bool = True,
        max_point_steps: int = 3000,
        auto_start: bool = True,
        start_delay: float = 15.0,
        start_y_ratio: float = 0.875,
        hit_bonus: float = 0.05,
        cross_bonus: float = 0.02,
        seed: int = 0,
        pfsp: bool = False,
        pfsp_floor: float = 0.10,
        pfsp_window: int = 30,
        pfsp_min_samples: int = 6,
    ):
        super().__init__()
        self.env_path = env_path
        self.serve_code = int(serve_code)
        self.worker_id = int(worker_id)
        self.base_port = int(base_port)
        self.graphics = bool(graphics)
        self.max_point_steps = int(max_point_steps)
        self.auto_start = bool(auto_start)
        self.start_delay = float(start_delay)
        self.start_y_ratio = float(start_y_ratio)
        self.hit_bonus = float(hit_bonus)
        self.cross_bonus = float(cross_bonus)
        assert learner_side in ("left", "right", "alternate")
        self.learner_side = learner_side
        self.snapshot_prob = float(snapshot_prob)
        self.snapshot_path = str(snapshot_path)

        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(13,), dtype=np.float32)
        self.action_space = spaces.MultiDiscrete([3, 3])

        self.router = torch.jit.load(opponent_model, map_location="cpu")
        self.router.eval()
        # Opponent pool: rotating set of RIGHT-NATIVE .pt files (router + mirrored
        # diverse opponents). Sampled per match -> principled diversity vs a single
        # frozen opponent (round-2 overfit fix). All must play right from native
        # right features (the env feeds native side features to the opponent).
        self._pool = []  # list of (name, model)
        if opponent_pool_dir:
            for p in sorted(Path(opponent_pool_dir).glob("*.pt")):
                try:
                    m = torch.jit.load(str(p), map_location="cpu"); m.eval()
                    self._pool.append((p.stem, m))
                except Exception as e:
                    print(f"[pool] failed to load {p.name}: {e}")
        if not self._pool:
            self._pool.append(("router", self.router))
        self.pool_play_counts = {name: 0 for name, _ in self._pool}
        print(f"[pool] {len(self._pool)} opponents: {[n for n, _ in self._pool]}")
        # PFSP: sample opponents by recent learnability wr*(1-wr) (peaks at the 0.5
        # even-matchup where the gradient signal is richest), with a per-opponent
        # FLOOR so none is ever eliminated -- only soft de-emphasis, and only for the
        # genuinely mastered (wr->1 => learnability->0 => floor). The router (~0.5) is
        # naturally top-weighted AND floor-protected, so it can never be starved.
        # NOTE: the pool itself is NOT auto-grown -- new opponents are added only
        # manually after passing the real-Unity gate (so additions are always earned).
        self.pfsp = bool(pfsp)
        self.pfsp_floor = float(pfsp_floor)
        self.pfsp_min_samples = int(pfsp_min_samples)
        self._recent = {name: deque(maxlen=int(pfsp_window)) for name, _ in self._pool}
        if self.pfsp:
            print(f"[pfsp] enabled: floor={self.pfsp_floor} window={pfsp_window} "
                  f"min_samples={self.pfsp_min_samples}")
        self._snapshot_model = None
        self._snapshot_mtime = 0.0

        self.ext_left = OpenCVStateExtractor(side="left")
        self.ext_right = OpenCVStateExtractor(side="right")
        self._rng = np.random.default_rng(seed)

        self._env = None
        self._agents = []
        self._visual = None
        self._point_steps = 0
        self._prev_ball = None  # (ball_x, ball_vx) from the left extractor
        self._side = "left"     # learner side for the current point
        self._opp_name = "router"
        self._opp_model = self.router

        self.relaunch_count = 0
        self.points = [0.0, 0.0]      # raw Unity points (left, right agent)
        self.lifetime_steps = 0
        self.stats = {}               # f"{side}|{opp}" -> [learner_won, learner_lost]

    # ---------- Unity lifecycle ----------

    def _launch(self):
        self._close_unity()
        last_err = None
        for attempt in range(3):
            try:
                string_channel = StringSideChannel()
                channel = CustomDataChannel()
                channel.send_data(serve=self.serve_code, p1=0, p2=0)
                if self.auto_start and self.graphics:
                    threading.Thread(
                        target=auto_click_start,
                        args=(self.start_delay, self.start_y_ratio),
                        daemon=True,
                    ).start()
                unity_env = UnityEnvironment(
                    file_name=self.env_path,
                    worker_id=self.worker_id + (self.relaunch_count % 20),
                    base_port=self.base_port,
                    side_channels=[string_channel, channel],
                    no_graphics=not self.graphics,
                )
                self._env = UnityParallelEnv(unity_env)
                obs_dict = self._env.reset()
                self._agents = list(self._env.agents)
                self._visual = obs_dict[self._agents[0]]["observation"][0]
                self.relaunch_count += 1
                if not self._wait_match_started(timeout_s=240.0):
                    raise UnityTimeOutException("match did not start (ball never visible); START click may have missed")
                self.ext_left.reset()
                self.ext_right.reset()
                return
            except (UnityTimeOutException, UnityCommunicatorStoppedException) as e:
                last_err = e
                self.relaunch_count += 1
                self._close_unity()
                time.sleep(3.0)
        raise RuntimeError(f"Unity relaunch failed 3 times: {last_err}")

    def _wait_match_started(self, timeout_s: float = 240.0) -> bool:
        noop = np.zeros(3, dtype=np.int32)
        t0 = time.time()
        consecutive = 0
        while time.time() - t0 < timeout_s:
            obs_dict, _, _, _ = self._env.step({a: noop for a in self._agents})
            self._visual = obs_dict[self._agents[0]]["observation"][0]
            feats = self.ext_left.features_from_image(self._visual)
            if feats[12] > 0.5:
                consecutive += 1
                if consecutive >= 5:
                    print(f"[launch] match running after {time.time() - t0:.1f}s of readiness wait")
                    return True
            else:
                consecutive = 0
        return False

    def _close_unity(self):
        if self._env is not None:
            try:
                self._env.close()
            except Exception:
                pass
            self._env = None

    # ---------- opponent pool ----------

    def _maybe_reload_snapshot(self):
        if not self.snapshot_path:
            return
        try:
            mtime = os.path.getmtime(self.snapshot_path)
        except OSError:
            return
        if mtime > self._snapshot_mtime:
            try:
                model = torch.jit.load(self.snapshot_path, map_location="cpu")
                model.eval()
                self._snapshot_model = model
                self._snapshot_mtime = mtime
                print(f"[pool] snapshot reloaded ({time.strftime('%H:%M:%S')})")
            except Exception as e:
                print(f"[pool] snapshot reload failed: {e}")

    def _opponent_weights(self) -> np.ndarray:
        """PFSP sampling weights. Every opponent keeps prob >= floor (never
        eliminated); the remaining mass is allocated by recent learnability
        wr*(1-wr). Cold-start opponents (< min_samples recent results) are scored
        as wr=0.5 so they get explored before being judged."""
        names = [n for n, _ in self._pool]
        raws = []
        for n in names:
            dq = self._recent.get(n)
            wr = 0.5 if (dq is None or len(dq) < self.pfsp_min_samples) else (sum(dq) / len(dq))
            raws.append(max(wr * (1.0 - wr), 1e-3))
        p_raw = np.asarray(raws, dtype=np.float64)
        p_raw /= p_raw.sum()
        n = len(names)
        floor = min(self.pfsp_floor, 1.0 / n)          # floor can't exceed uniform share
        p = floor + (1.0 - n * floor) * p_raw          # guarantee >= floor for all
        return p / p.sum()

    def _sample_episode_setup(self):
        if self.learner_side == "alternate":
            self._side = "left" if self._rng.uniform() < 0.5 else "right"
        else:
            self._side = self.learner_side
        if self.pfsp and len(self._pool) > 1:
            idx = int(self._rng.choice(len(self._pool), p=self._opponent_weights()))
        else:
            idx = int(self._rng.integers(0, len(self._pool)))
        self._opp_name, self._opp_model = self._pool[idx]
        self.pool_play_counts[self._opp_name] = self.pool_play_counts.get(self._opp_name, 0) + 1

    # ---------- helpers ----------

    def _refresh_features(self):
        """Compute both extractors EXACTLY ONCE per visual frame (velocities are
        position diffs between consecutive calls, so double-calling breaks them)."""
        self._fl = self.ext_left.features_from_image(self._visual)
        self._fr = self.ext_right.features_from_image(self._visual)

    def _obs13(self) -> np.ndarray:
        own = self._fl if self._side == "left" else self._fr
        return reduce16to13(own)

    def _opponent_action(self) -> np.ndarray:
        feats = self._fr if self._side == "left" else self._fl
        with torch.no_grad():
            act = self._opp_model(torch.as_tensor(feats, dtype=torch.float32))
        act = np.asarray(act.cpu().numpy(), dtype=np.int32)[:3]
        return np.clip(act, 0, 2)

    def _shaping(self, fl: np.ndarray) -> float:
        """Tiny real-event bonuses, side-aware. Ball coordinates are shared, so
        the left extractor's ball features serve both sides.
        Thresholds (0.05) sit above measured image-path vx jitter."""
        prev = self._prev_ball
        ball_visible = fl[12] > 0.5
        cur = (float(fl[0]), float(fl[2])) if ball_visible else None
        self._prev_ball = cur
        if prev is None or cur is None:
            return 0.0
        px, pvx = prev
        cx, cvx = cur
        bonus = 0.0
        if self._side == "left":
            if px < 0.0 and pvx < -0.05 and cvx > 0.05:
                bonus += self.hit_bonus
            if px < 0.0 <= cx:
                bonus += self.cross_bonus
        else:
            if px > 0.0 and pvx > 0.05 and cvx < -0.05:
                bonus += self.hit_bonus
            if px > 0.0 >= cx:
                bonus += self.cross_bonus
        return bonus

    # ---------- gym API ----------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self._env is None:
            self._launch()
            self._refresh_features()
        self._sample_episode_setup()
        self._point_steps = 0
        self._prev_ball = None
        return self._obs13(), {}

    def step(self, action):
        act = np.asarray(action).reshape(-1)
        learner_action = np.array([int(act[0]), int(act[1]), 0], dtype=np.int32)
        opp_action = self._opponent_action()

        learner_agent = self._agents[0] if self._side == "left" else self._agents[1]
        opp_agent = self._agents[1] if self._side == "left" else self._agents[0]
        actions = {learner_agent: learner_action, opp_agent: opp_action}

        try:
            obs_dict, rewards, dones, infos = self._env.step(actions)
        except UnityCommunicatorStoppedException:
            self._close_unity()
            return self._obs13(), 0.0, True, False, {"app_exit": True}

        self._visual = obs_dict[self._agents[0]]["observation"][0]
        self.lifetime_steps += 1
        self._point_steps += 1

        r_left = float(rewards[self._agents[0]])
        r_right = float(rewards[self._agents[1]])
        self.points[0] += r_left
        self.points[1] += r_right
        r_own = r_left if self._side == "left" else r_right
        r_opp = r_right if self._side == "left" else r_left

        self._refresh_features()
        obs13 = self._obs13()
        reward = r_own - r_opp + self._shaping(self._fl)

        unity_done = bool(dones[self._agents[0]] or dones[self._agents[1]])
        point_scored = (r_left != 0.0) or (r_right != 0.0)
        truncated = self._point_steps >= self.max_point_steps
        terminated = point_scored or unity_done

        if point_scored:
            key = f"{self._side}|{self._opp_name}"
            won_lost = self.stats.setdefault(key, [0, 0])
            won = r_own > r_opp
            won_lost[0 if won else 1] += 1
            if self._opp_name in self._recent:
                self._recent[self._opp_name].append(1 if won else 0)
        if unity_done:
            self._close_unity()

        info = {"r_left": r_left, "r_right": r_right, "points": list(self.points),
                "side": self._side, "opponent": self._opp_name}
        return obs13, reward, terminated, truncated, info

    def close(self):
        self._close_unity()
