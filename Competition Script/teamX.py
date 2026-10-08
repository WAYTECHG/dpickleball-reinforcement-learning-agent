from __future__ import annotations

import queue
from pathlib import Path
from threading import Thread
from typing import Any, Optional, Tuple
import math

import cv2
import numpy as np
import torch

try:
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    cv2.ocl.setUseOpenCL(False)
except Exception:
    pass


class OpenCVStateExtractor:
    """Extract the 16 input features used by the TorchScript policy."""

    def __init__(self, img_w: int = 168, img_h: int = 84, side: str = "left"):
        self.img_w = img_w
        self.img_h = img_h
        self.side = side
        self.prev_ball_xy: Optional[Tuple[float, float]] = None

    @property
    def side_sign(self) -> float:
        return 1.0 if self.side == "right" else -1.0

    @staticmethod
    def _as_uint8_rgb(img: np.ndarray) -> np.ndarray:
        arr = np.asarray(img)
        if arr.dtype != np.uint8:
            arr = (arr * 255.0 if arr.max() <= 1.0 else arr).clip(0, 255).astype(np.uint8)
        if arr.ndim == 3 and arr.shape[0] == 3 and arr.shape[-1] != 3:
            arr = np.transpose(arr, (1, 2, 0))
        return arr

    @staticmethod
    def _largest_blob_center(mask: np.ndarray, min_area: float, max_area: float = 1e9):
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in sorted(cnts, key=cv2.contourArea, reverse=True):
            area = cv2.contourArea(c)
            if area < min_area or area > max_area:
                continue
            m = cv2.moments(c)
            if m["m00"]:
                return float(m["m10"] / m["m00"]), float(m["m01"] / m["m00"])
        return None

    @staticmethod
    def _largest_oriented_blob(mask: np.ndarray, min_area: float):
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in sorted(cnts, key=cv2.contourArea, reverse=True):
            if cv2.contourArea(c) < min_area:
                continue
            m = cv2.moments(c)
            if not m["m00"]:
                continue
            cx = float(m["m10"] / m["m00"])
            cy = float(m["m01"] / m["m00"])
            angle = 0.0
            if len(c) >= 5:
                angle_deg = cv2.minAreaRect(c)[-1]
                if angle_deg < -45:
                    angle_deg += 90
                angle = math.radians(float(angle_deg))
            return (cx, cy), angle
        return None, 0.0

    def _racket_mask(self, hsv: np.ndarray, roi: np.ndarray) -> np.ndarray:
        orange = cv2.inRange(hsv, np.array([5, 45, 45]), np.array([30, 255, 255]))
        red1 = cv2.inRange(hsv, np.array([0, 45, 45]), np.array([6, 255, 255]))
        red2 = cv2.inRange(hsv, np.array([170, 45, 45]), np.array([179, 255, 255]))
        mask = cv2.bitwise_or(orange, cv2.bitwise_or(red1, red2))
        mask = cv2.bitwise_and(mask, roi)
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))

    def features_from_image(self, img: np.ndarray) -> np.ndarray:
        img_rgb = self._as_uint8_rgb(img)
        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

        roi = np.zeros((img_rgb.shape[0], img_rgb.shape[1]), dtype=np.uint8)
        roi[20:80, 8:164] = 255

        ball_mask = cv2.inRange(hsv, np.array([20, 80, 80]), np.array([45, 255, 255]))
        ball_mask = cv2.bitwise_and(ball_mask, roi)
        ball_mask = cv2.morphologyEx(ball_mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        ball_xy = self._largest_blob_center(ball_mask, min_area=3, max_area=160)

        racket_mask = self._racket_mask(hsv, roi)
        controlled = racket_mask.copy()
        opponent = racket_mask.copy()
        if self.side == "right":
            controlled[:, : self.img_w // 2] = 0
            opponent[:, self.img_w // 2 :] = 0
        else:
            controlled[:, self.img_w // 2 :] = 0
            opponent[:, : self.img_w // 2] = 0

        paddle_xy, paddle_angle = self._largest_oriented_blob(controlled, min_area=15)
        opponent_xy, _ = self._largest_oriented_blob(opponent, min_area=15)

        if ball_xy is None:
            bx, by, ball_visible = 0.5 * self.img_w, 0.5 * self.img_h, 0.0
            bvx = bvy = 0.0
        else:
            bx, by = ball_xy
            ball_visible = 1.0
            if self.prev_ball_xy is None:
                bvx = bvy = 0.0
            else:
                bvx = bx - self.prev_ball_xy[0]
                bvy = by - self.prev_ball_xy[1]
            self.prev_ball_xy = (bx, by)

        if paddle_xy is None:
            px = self.img_w * (0.82 if self.side == "right" else 0.18)
            py = self.img_h * 0.55
            paddle_visible = 0.0
            paddle_angle = math.pi / 2.0
        else:
            px, py = paddle_xy
            paddle_visible = 1.0

        opponent_y = opponent_xy[1] if opponent_xy is not None else self.img_h * 0.5
        return self._pack_features(bx, by, bvx, bvy, px, py, paddle_angle, ball_visible, paddle_visible, opponent_y)

    def _pack_features(self, bx, by, bvx, bvy, px, py, paddle_angle, ball_visible, paddle_visible, opponent_y):
        nbx = (bx / max(1, self.img_w - 1)) * 2.0 - 1.0
        nby = (by / max(1, self.img_h - 1)) * 2.0 - 1.0
        npx = (px / max(1, self.img_w - 1)) * 2.0 - 1.0
        npy = (py / max(1, self.img_h - 1)) * 2.0 - 1.0
        nopp_y = (float(opponent_y) / max(1, self.img_h - 1)) * 2.0 - 1.0
        nbvx = float(np.clip(bvx / 4.0, -1.0, 1.0))
        nbvy = float(np.clip(bvy / 4.0, -1.0, 1.0))
        relx = float(np.clip((bx - px) / self.img_w, -1.0, 1.0))
        rely = float(np.clip((by - py) / self.img_h, -1.0, 1.0))

        return np.array([
            nbx, nby, nbvx, nbvy,
            npx, npy,
            math.sin(paddle_angle), math.cos(paddle_angle),
            relx, rely,
            0.0, 0.0,
            float(ball_visible), float(paddle_visible),
            self.side_sign,
            float(np.clip(nopp_y, -1.0, 1.0)),
        ], dtype=np.float32)


def _is_visual_array(x: Any) -> bool:
    try:
        arr = np.asarray(x)
    except Exception:
        return False
    return arr.ndim == 3 and (
        (arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4))
        or arr.shape[-1] in (1, 3, 4)
    )


def _find_visual_in_agent_obs(agent_obs: Any) -> Optional[np.ndarray]:
    if _is_visual_array(agent_obs):
        return np.asarray(agent_obs)
    if isinstance(agent_obs, dict):
        for obs in agent_obs.get("observation", []):
            if _is_visual_array(obs):
                return np.asarray(obs)
    if isinstance(agent_obs, (list, tuple)):
        for obs in agent_obs:
            if _is_visual_array(obs):
                return np.asarray(obs)
    return None


def extract_visual_observation(observation: Any) -> np.ndarray:
    visual = _find_visual_in_agent_obs(observation)
    if visual is not None:
        return visual
    if isinstance(observation, dict):
        for agent_obs in observation.values():
            visual = _find_visual_in_agent_obs(agent_obs)
            if visual is not None:
                return visual
    raise RuntimeError("No visual observation found.")


def _find_model_file() -> str:
    here = Path(__file__).resolve().parent
    for name in ("teamX.pt", "model_left.pt", "model.pt"):
        path = here / name
        if path.exists():
            return str(path)
    raise FileNotFoundError("Put teamX.pt, model_left.pt, or model.pt beside teamX.py.")


class _PinnedWorker:
    """Runs a callable on a single persistent background thread.

    Competition.py spawns a brand-new native thread every environment step to
    call policy(). Each new OS thread makes torch/numpy's OpenMP/MKL/OpenBLAS
    backends allocate a per-thread scratch buffer that is never freed when the
    thread exits, so RAM grows without bound over a match. Routing the actual
    torch/cv2 work through one long-lived thread keeps those libraries seeing
    a single, stable thread id and eliminates the leak.
    """

    def __init__(self, fn):
        self._fn = fn
        self._in_q: "queue.Queue" = queue.Queue(maxsize=1)
        self._out_q: "queue.Queue" = queue.Queue(maxsize=1)
        Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while True:
            observation, reward = self._in_q.get()
            try:
                self._out_q.put(self._fn(observation, reward))
            except Exception as e:  # keep the worker alive across bad frames
                self._out_q.put(e)

    def __call__(self, observation, reward):
        self._in_q.put((observation, reward))
        result = self._out_q.get()
        if isinstance(result, Exception):
            raise result
        return result


class TeamX:
    """Left-side competition policy."""

    def __init__(self):
        self.extractor = OpenCVStateExtractor(side="left")
        self.model = torch.jit.load(_find_model_file(), map_location="cpu")
        self.model.eval()
        with torch.no_grad():
            dummy = torch.zeros(16, dtype=torch.float32)
            dummy[14] = -1.0
            self.model(dummy)
        self._worker = _PinnedWorker(self._policy_impl)

    def policy(self, observation, reward):
        return self._worker(observation, reward)

    def _policy_impl(self, observation, reward):
        features = self.extractor.features_from_image(extract_visual_observation(observation))
        with torch.no_grad():
            action = self.model(torch.as_tensor(features, dtype=torch.float32)).cpu().numpy()
        action = np.clip(np.asarray(action[:3], dtype=np.int32), 0, 2)
        action[2] = 0
        return action
