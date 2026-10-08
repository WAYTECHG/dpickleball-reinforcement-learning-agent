from __future__ import annotations

import argparse
from pathlib import Path
import cv2
import numpy as np

from envs.python_wall_env import PythonWallPickleballEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--level", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--obs-source", type=str, default="state", choices=["state", "opencv"])
    args = parser.parse_args()

    env = PythonWallPickleballEnv(level=args.level, seed=123, obs_source=args.obs_source, side="random")
    obs, info = env.reset()
    total_reward = 0.0

    pause_frames = 0
    last_fail = ""

    for t in range(args.steps):

        vertical = 0
        if env.ball_y < env.paddle_y - 2:
            vertical = 1
        elif env.ball_y > env.paddle_y + 2:
            vertical = 2

        horizontal = 0

        if getattr(env, "active_side", "right") == "right":
            if env.paddle_x < 138:
                horizontal = 1
            elif env.paddle_x > 150:
                horizontal = 2
        else:
            if env.paddle_x < 26:
                horizontal = 1
            elif env.paddle_x > 42:
                horizontal = 2

        rotation = 0
        if env.ball_y < env.paddle_y:
            rotation = 1
        elif env.ball_y > env.paddle_y:
            rotation = 2

        obs, reward, terminated, truncated, info = env.step([vertical, horizontal, rotation])
        total_reward += reward

        img = env.render()
        fail_txt = f" fail={last_fail}" if last_fail else ""
        cv2.putText(img, f"level={args.level} side={info.get('side','?')} t={t} R={total_reward:.1f} returns={info.get('success_count',0)}{fail_txt}",
                    (3, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (255,255,255), 1)
        cv2.putText(img, f"ball=({env.ball_x:.1f},{env.ball_y:.1f}) v=({env.ball_vx:.2f},{env.ball_vy:.2f}) paddle=({env.paddle_x:.1f},{env.paddle_y:.1f})",
                    (3, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255,255,255), 1)
        cv2.imshow("Python Wall Pickleball Env", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        key = cv2.waitKey(30)
        if key == ord("q"):
            break
        if terminated or truncated:
            last_fail = str(info.get("fail_reason", "reset"))


            for _ in range(10):
                cv2.imshow("Python Wall Pickleball Env", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                if cv2.waitKey(30) == ord("q"):
                    break
            obs, info = env.reset()

    cv2.destroyAllWindows()
    env.close()


if __name__ == "__main__":
    main()
