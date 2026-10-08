from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from stable_baselines3 import PPO

from envs.python_wall_env import PythonWallPickleballEnv


def load_model_safely(model_path: Path, retries: int = 8, delay: float = 0.5):
    last_error = None
    for _ in range(retries):
        try:
            if model_path.exists():
                return PPO.load(str(model_path), device="cpu")
        except Exception as exc:
            last_error = exc
            time.sleep(delay)
    if last_error is not None:
        print(f"[watch_latest_policy] Could not load {model_path}: {last_error}")
    else:
        print(f"[watch_latest_policy] Model not found yet: {model_path}")
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch the current Foundation PPO policy in a visual wall environment.")
    parser.add_argument("--model", type=str, default="checkpoints/latest.zip", help="SB3 PPO .zip checkpoint to watch.")
    parser.add_argument("--level", type=int, default=1, help="Wall curriculum level.")
    parser.add_argument("--steps", type=int, default=2000, help="Maximum visual steps.")
    parser.add_argument("--obs-source", type=str, default="state", choices=["state", "opencv"])
    parser.add_argument("--delay-ms", type=int, default=30, help="OpenCV display delay per step.")
    parser.add_argument("--side", type=str, default="random", choices=["random", "left", "right"], help="Which side to visualize.")
    parser.add_argument("--reload-each-episode", action="store_true", help="Reload model after every reset so a running training process can update latest.zip.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = root / model_path

    model = load_model_safely(model_path)
    if model is None:
        print("[watch_latest_policy] No usable model yet. Start training first, or check the model path.")
        return

    env = PythonWallPickleballEnv(level=args.level, seed=321, obs_source=args.obs_source, side=args.side)
    obs, info = env.reset()
    total_reward = 0.0
    episode_returns = 0
    last_load_time = model_path.stat().st_mtime if model_path.exists() else 0.0

    for t in range(args.steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)

        img = env.render()
        text1 = f"latest policy | level={args.level} side={info.get('side','?')} t={t} R={total_reward:.1f} returns={info.get('success_count',0)}"
        text2 = f"model={model_path.name} | press q to quit"
        cv2.putText(img, text1, (3, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (255, 255, 255), 1)
        cv2.putText(img, text2, (3, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 255), 1)
        cv2.imshow("Foundation latest-policy visual check", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        if cv2.waitKey(args.delay_ms) == ord("q"):
            break

        if terminated or truncated:
            episode_returns += int(info.get("success_count", 0))
            obs, info = env.reset()
            if args.reload_each_episode and model_path.exists():
                try:
                    mtime = model_path.stat().st_mtime
                    if mtime != last_load_time:
                        new_model = load_model_safely(model_path)
                        if new_model is not None:
                            model = new_model
                            last_load_time = mtime
                            print(f"[watch_latest_policy] Reloaded updated model: {model_path}")
                except Exception:
                    pass

    cv2.destroyAllWindows()
    env.close()


if __name__ == "__main__":
    main()
