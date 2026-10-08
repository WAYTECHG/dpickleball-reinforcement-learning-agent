from __future__ import annotations

import math
from pathlib import Path
import numpy as np
import cv2


class CourtRenderer:
    """OpenCV renderer for the wall simulator."""

    def __init__(self, w: int = 168, h: int = 84):
        self.w = w
        self.h = h
        self.asset_dir = Path(__file__).resolve().parents[1] / "assets"
        self.left_sprite = self._load_sprite(self.asset_dir / "paddle_left_real.png")
        self.right_sprite = self._load_sprite(self.asset_dir / "paddle_right_real.png")


        self.sprite_height = 13
        self.handle_len = 5

    def _load_sprite(self, path: Path):
        if not path.exists():
            return None
        img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if img is None:
            return None
        if img.ndim == 3 and img.shape[2] == 4:

            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
        elif img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            alpha = np.full(img.shape[:2] + (1,), 255, dtype=np.uint8)
            img = np.concatenate([img, alpha], axis=2)
        return img

    def _resize_sprite(self, sprite: np.ndarray, target_h: int) -> np.ndarray:
        h, w = sprite.shape[:2]
        scale = float(target_h) / max(1.0, float(h))
        target_w = max(4, int(round(w * scale)))
        return cv2.resize(sprite, (target_w, target_h), interpolation=cv2.INTER_AREA)

    def _rotate_sprite(self, sprite: np.ndarray, angle_rad: float) -> np.ndarray:

        angle_deg = math.degrees(angle_rad - math.pi / 2.0)
        h, w = sprite.shape[:2]
        center = (w / 2.0, h / 2.0)
        M = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
        cos = abs(M[0, 0])
        sin = abs(M[0, 1])
        new_w = int(round(h * sin + w * cos))
        new_h = int(round(h * cos + w * sin))
        M[0, 2] += new_w / 2.0 - center[0]
        M[1, 2] += new_h / 2.0 - center[1]
        return cv2.warpAffine(sprite, M, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))

    def _overlay_rgba(self, img_rgb: np.ndarray, sprite_rgba: np.ndarray, cx: float, cy: float) -> None:
        h, w = sprite_rgba.shape[:2]
        x1 = int(round(cx - w / 2.0))
        y1 = int(round(cy - h / 2.0))
        x2 = x1 + w
        y2 = y1 + h

        ix1 = max(0, x1)
        iy1 = max(0, y1)
        ix2 = min(img_rgb.shape[1], x2)
        iy2 = min(img_rgb.shape[0], y2)
        if ix1 >= ix2 or iy1 >= iy2:
            return

        sx1 = ix1 - x1
        sy1 = iy1 - y1
        sx2 = sx1 + (ix2 - ix1)
        sy2 = sy1 + (iy2 - iy1)

        patch = sprite_rgba[sy1:sy2, sx1:sx2]
        alpha = patch[:, :, 3:4].astype(np.float32) / 255.0
        img_rgb[iy1:iy2, ix1:ix2] = (
            patch[:, :, :3].astype(np.float32) * alpha
            + img_rgb[iy1:iy2, ix1:ix2].astype(np.float32) * (1.0 - alpha)
        ).astype(np.uint8)

    def draw_background(self, active_wall_x: float | None = None, home_xy=None) -> np.ndarray:
        img = np.zeros((self.h, self.w, 3), dtype=np.uint8)
        img[:24, :, :] = np.array([25, 75, 35], dtype=np.uint8)
        img[24:, :, :] = np.array([0, 0, 0], dtype=np.uint8)

        x1, x2, y1, y2 = 18, 158, 27, 72

        img[y1:y2, x1:68] = (60, 105, 178)
        img[y1:y2, 108:x2] = (60, 105, 178)
        img[y1:y2, 68:108] = (145, 220, 235)
        img[y1:y2, 82:86] = (135, 210, 228)

        line_col = (235, 245, 255)
        cv2.rectangle(img, (x1, y1), (x2, y2), line_col, 1)
        cv2.line(img, (68, y1), (68, y2), line_col, 1)
        cv2.line(img, (84, y1), (84, y2), line_col, 1)
        cv2.line(img, (108, y1), (108, y2), line_col, 1)
        cv2.line(img, (x1, 50), (68, 50), line_col, 1)
        cv2.line(img, (108, 50), (x2, 50), line_col, 1)

        cv2.rectangle(img, (x1, y1-2), (x2, y1), line_col, -1)
        cv2.rectangle(img, (x1, y2), (x2, y2+2), line_col, -1)

        wall_col = (255, 255, 255)
        cv2.line(img, (68, y1), (108, y1), wall_col, 2)
        cv2.line(img, (68, y2), (108, y2), wall_col, 2)

        if active_wall_x is not None:
            wx = int(round(float(active_wall_x)))
            cv2.line(img, (wx, y1), (wx, y2), (255, 255, 255), 2)

        if home_xy is not None:
            hx, hy = int(round(home_xy[0])), int(round(home_xy[1]))
            cv2.circle(img, (hx, hy), 3, (80, 255, 120), 1)
            cv2.line(img, (hx, max(y1, hy - 7)), (hx, min(y2, hy + 7)), (80, 255, 120), 1)
            cv2.line(img, (max(x1, hx - 7), hy), (min(x2, hx + 7), hy), (80, 255, 120), 1)

        return img

    def draw_ball(self, img: np.ndarray, x: float, y: float, r: float = 3.2) -> None:
        cv2.circle(img, (int(round(x)), int(round(y))), int(round(r)), (235, 230, 0), -1)
        cv2.circle(img, (int(round(x)), int(round(y))), int(round(r)), (255, 255, 80), 1)

    def _draw_poly_paddle_fallback(self, img: np.ndarray, x: float, y: float, angle_rad: float, side: str = "right") -> None:
        length = 13.0
        width = 7.0
        t = np.array([math.cos(angle_rad), math.sin(angle_rad)], dtype=np.float32)
        n = np.array([-math.sin(angle_rad), math.cos(angle_rad)], dtype=np.float32)
        half_len = t * (length / 2)
        half_width = n * (width / 2)
        center = np.array([x, y], dtype=np.float32)
        pts = np.array([
            center - half_len - half_width,
            center + half_len - half_width,
            center + half_len + half_width,
            center - half_len + half_width,
        ], dtype=np.int32)
        fill = (235, 80, 45) if side == "right" else (220, 115, 35)
        edge = (255, 130, 90) if side == "right" else (245, 170, 70)
        cv2.fillConvexPoly(img, pts, fill)
        cv2.polylines(img, [pts], isClosed=True, color=edge, thickness=1)
        n_out = n if ((side == "left" and n[0] < 0) or (side == "right" and n[0] > 0)) else -n
        handle_start = center + n_out * (width / 2)
        handle_end = handle_start + n_out * 7.0
        cv2.line(img, tuple(np.round(handle_start).astype(int)), tuple(np.round(handle_end).astype(int)), (20, 80, 70), 2)

    def _draw_sprite_paddle(self, img: np.ndarray, x: float, y: float, angle_rad: float, side: str) -> bool:
        sprite = self.right_sprite if side == "right" else self.left_sprite
        if sprite is None:
            return False
        small = self._resize_sprite(sprite, self.sprite_height)
        rot = self._rotate_sprite(small, angle_rad)
        self._overlay_rgba(img, rot, x, y)


        t = np.array([math.cos(angle_rad), math.sin(angle_rad)], dtype=np.float32)
        start = np.array([x, y], dtype=np.float32) + t * (self.sprite_height * 0.43)
        end = start + t * float(self.handle_len)
        cv2.line(img, tuple(np.round(start).astype(int)), tuple(np.round(end).astype(int)), (20, 80, 70), 2)
        return True

    def draw_paddle(self, img: np.ndarray, x: float, y: float, angle_rad: float, side: str = "right") -> None:
        if not self._draw_sprite_paddle(img, x, y, angle_rad, side=side):
            self._draw_poly_paddle_fallback(img, x, y, angle_rad, side=side)

    def draw_opponent_racket(self, img: np.ndarray, x: float, y: float, angle_rad: float, side: str = "left") -> None:
        if not self._draw_sprite_paddle(img, x, y, angle_rad, side=side):
            self._draw_poly_paddle_fallback(img, x, y, angle_rad, side=side)

    def render(
        self,
        ball_xy,
        paddle_xy,
        paddle_angle_rad,
        side: str = "right",
        wall_x: float | None = None,
        home_xy=None,
        opponent_xy=None,
        opponent_angle_rad: float = 0.0,
        opponent_side: str | None = None,
    ) -> np.ndarray:
        img = self.draw_background(active_wall_x=wall_x, home_xy=home_xy)
        if opponent_xy is not None:
            opp_side = opponent_side or ("left" if side == "right" else "right")
            self.draw_opponent_racket(img, opponent_xy[0], opponent_xy[1], opponent_angle_rad, side=opp_side)
        self.draw_ball(img, ball_xy[0], ball_xy[1])
        self.draw_paddle(img, paddle_xy[0], paddle_xy[1], paddle_angle_rad, side=side)
        return img
