from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
import math
import numpy as np
import cv2


@dataclass
class DetectionResult:
    ball_xy: Optional[Tuple[float, float]]
    paddle_xy: Optional[Tuple[float, float]]
    paddle_angle_rad: float
    opponent_xy: Optional[Tuple[float, float]]
    debug: Dict[str, Any]


class OpenCVStateExtractor:
    """Extracts compact features from rendered frames."""

    def __init__(self, img_w: int = 168, img_h: int = 84, side: str = "right"):
        self.img_w = img_w
        self.img_h = img_h
        self.side = side
        self.prev_ball_xy: Optional[Tuple[float, float]] = None

    def reset(self) -> None:
        self.prev_ball_xy = None

    def set_side(self, side: str) -> None:
        self.side = str(side)
        self.reset()

    @property
    def side_sign(self) -> float:
        return 1.0 if self.side == "right" else -1.0

    @staticmethod
    def _as_uint8_rgb(img: np.ndarray) -> np.ndarray:
        arr = np.asarray(img)
        if arr.dtype != np.uint8:
            if arr.max() <= 1.0:
                arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
            else:
                arr = arr.clip(0, 255).astype(np.uint8)
        if arr.ndim == 3 and arr.shape[0] == 3 and arr.shape[-1] != 3:
            arr = np.transpose(arr, (1, 2, 0))
        return arr

    def _racket_mask(self, hsv: np.ndarray, roi_mask: np.ndarray) -> np.ndarray:

        orange = cv2.inRange(hsv, np.array([5, 45, 45]), np.array([30, 255, 255]))

        red1 = cv2.inRange(hsv, np.array([0, 45, 45]), np.array([6, 255, 255]))
        red2 = cv2.inRange(hsv, np.array([170, 45, 45]), np.array([179, 255, 255]))
        mask = cv2.bitwise_or(orange, cv2.bitwise_or(red1, red2))
        mask = cv2.bitwise_and(mask, roi_mask)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        return mask

    def detect(self, img: np.ndarray) -> DetectionResult:
        img_rgb = self._as_uint8_rgb(img)
        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)


        roi_mask = np.zeros((img_rgb.shape[0], img_rgb.shape[1]), dtype=np.uint8)
        roi_mask[20:80, 8:164] = 255


        ball_mask = cv2.inRange(hsv, np.array([20, 80, 80]), np.array([45, 255, 255]))
        ball_mask = cv2.bitwise_and(ball_mask, roi_mask)
        ball_mask = cv2.morphologyEx(ball_mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
        ball_xy = self._largest_blob_center(ball_mask, min_area=3, max_area=160)

        racket_mask = self._racket_mask(hsv, roi_mask)


        controlled_mask = racket_mask.copy()
        opponent_mask = racket_mask.copy()
        if self.side == "right":
            controlled_mask[:, :self.img_w // 2] = 0
            opponent_mask[:, self.img_w // 2:] = 0
        elif self.side == "left":
            controlled_mask[:, self.img_w // 2:] = 0
            opponent_mask[:, :self.img_w // 2] = 0

        paddle_xy, paddle_angle = self._largest_oriented_blob(controlled_mask, min_area=15)
        opponent_xy, _ = self._largest_oriented_blob(opponent_mask, min_area=15)

        return DetectionResult(
            ball_xy=ball_xy,
            paddle_xy=paddle_xy,
            paddle_angle_rad=paddle_angle,
            opponent_xy=opponent_xy,
            debug={
                "ball_found": ball_xy is not None,
                "paddle_found": paddle_xy is not None,
                "opponent_found": opponent_xy is not None,
                "side": self.side,
            },
        )

    @staticmethod
    def _largest_blob_center(mask: np.ndarray, min_area: float = 3, max_area: float = 1e9):
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        for c in cnts:
            area = cv2.contourArea(c)
            if area < min_area or area > max_area:
                continue
            M = cv2.moments(c)
            if M["m00"] == 0:
                continue
            return (float(M["m10"] / M["m00"]), float(M["m01"] / M["m00"]))
        return None

    @staticmethod
    def _largest_oriented_blob(mask: np.ndarray, min_area: float = 15):
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None, 0.0
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        for c in cnts:
            if cv2.contourArea(c) < min_area:
                continue
            M = cv2.moments(c)
            if M["m00"] == 0:
                continue
            cx = float(M["m10"] / M["m00"])
            cy = float(M["m01"] / M["m00"])

            angle_rad = 0.0
            if len(c) >= 5:
                rect = cv2.minAreaRect(c)
                angle_deg = rect[-1]
                if angle_deg < -45:
                    angle_deg += 90
                angle_rad = math.radians(float(angle_deg))
            return (cx, cy), angle_rad
        return None, 0.0

    def features_from_image(
        self,
        img: np.ndarray,
        time_own_side_norm: float = 0.0,
        success_count_norm: float = 0.0,
    ) -> np.ndarray:
        det = self.detect(img)
        if det.ball_xy is None:
            bx, by = (0.5 * self.img_w, 0.5 * self.img_h)
            ball_visible = 0.0
        else:
            bx, by = det.ball_xy
            ball_visible = 1.0

        if self.prev_ball_xy is None or det.ball_xy is None:
            bvx, bvy = 0.0, 0.0
        else:
            bvx = bx - self.prev_ball_xy[0]
            bvy = by - self.prev_ball_xy[1]
        if det.ball_xy is not None:
            self.prev_ball_xy = (bx, by)

        if det.paddle_xy is None:
            px = self.img_w * (0.82 if self.side == "right" else 0.18)
            py = self.img_h * 0.55
            paddle_visible = 0.0
            pang = math.pi / 2.0
        else:
            px, py = det.paddle_xy
            paddle_visible = 1.0
            pang = det.paddle_angle_rad

        opponent_y = det.opponent_xy[1] if det.opponent_xy is not None else self.img_h * 0.5

        return self._pack_features(
            bx=bx, by=by, bvx=bvx, bvy=bvy,
            px=px, py=py, paddle_angle_rad=pang,
            ball_visible=ball_visible, paddle_visible=paddle_visible,
            time_own_side_norm=time_own_side_norm,
            success_count_norm=success_count_norm,
            opponent_y=opponent_y,
        )

    def features_from_state(
        self,
        ball_xy: Tuple[float, float],
        ball_vxy: Tuple[float, float],
        paddle_xy: Tuple[float, float],
        paddle_angle_rad: float,
        time_own_side_norm: float = 0.0,
        success_count_norm: float = 0.0,
        opponent_y: float | None = None,
    ) -> np.ndarray:
        return self._pack_features(
            bx=ball_xy[0], by=ball_xy[1],
            bvx=ball_vxy[0], bvy=ball_vxy[1],
            px=paddle_xy[0], py=paddle_xy[1],
            paddle_angle_rad=paddle_angle_rad,
            ball_visible=1.0, paddle_visible=1.0,
            time_own_side_norm=time_own_side_norm,
            success_count_norm=success_count_norm,
            opponent_y=(self.img_h * 0.5 if opponent_y is None else float(opponent_y)),
        )

    def _pack_features(
        self,
        bx: float, by: float, bvx: float, bvy: float,
        px: float, py: float, paddle_angle_rad: float,
        ball_visible: float, paddle_visible: float,
        time_own_side_norm: float, success_count_norm: float,
        opponent_y: float,
    ) -> np.ndarray:
        nbx = (bx / max(1, self.img_w - 1)) * 2.0 - 1.0
        nby = (by / max(1, self.img_h - 1)) * 2.0 - 1.0
        npx = (px / max(1, self.img_w - 1)) * 2.0 - 1.0
        npy = (py / max(1, self.img_h - 1)) * 2.0 - 1.0
        nopp_y = (float(opponent_y) / max(1, self.img_h - 1)) * 2.0 - 1.0

        nbvx = float(np.clip(bvx / 4.0, -1.0, 1.0))
        nbvy = float(np.clip(bvy / 4.0, -1.0, 1.0))

        relx = float(np.clip((bx - px) / self.img_w, -1.0, 1.0))
        rely = float(np.clip((by - py) / self.img_h, -1.0, 1.0))

        features = np.array([
            nbx, nby, nbvx, nbvy,
            npx, npy,
            math.sin(paddle_angle_rad), math.cos(paddle_angle_rad),
            relx, rely,
            float(np.clip(time_own_side_norm, 0.0, 1.0)),
            float(np.clip(success_count_norm, 0.0, 1.0)),
            float(ball_visible),
            float(paddle_visible),
            self.side_sign,
            float(np.clip(nopp_y, -1.0, 1.0)),
        ], dtype=np.float32)
        return features


def mirror_features_right_to_left_view(features: np.ndarray) -> np.ndarray:
    f = np.array(features, dtype=np.float32).copy()
    f[0] = -f[0]
    f[4] = -f[4]
    f[2] = -f[2]
    f[8] = -f[8]
    if len(f) >= 15:
        f[14] = -f[14]
    return f


def mirror_action(action):
    a = list(map(int, action))
    if len(a) != 3:
        return a
    if a[1] == 1:
        a[1] = 2
    elif a[1] == 2:
        a[1] = 1
    if a[2] == 1:
        a[2] = 2
    elif a[2] == 2:
        a[2] = 1
    return a
