from __future__ import annotations

import argparse
import json
from typing import Dict, Any

import numpy as np

from envs.python_wall_env import PythonWallPickleballEnv
from config import OBS_SOURCE, EVAL_TRIALS, TRIAL_RETURN_TARGET


def evaluate_model(
    model,
    levels,
    n_trials: int = EVAL_TRIALS,
    seed: int = 1000,
    obs_source: str = OBS_SOURCE,
    deterministic: bool = True,
) -> Dict[int, Dict[str, Any]]:
    results: Dict[int, Dict[str, Any]] = {}

    for level in levels:
        successes = 0
        returns = []
        rewards = []
        failures = 0
        front_rates = []
        net_action_rates = []
        avg_paddle_xs = []
        contacts = []
        strong_bounces = []
        recovery_rewards = []
        home_distances = []
        opponent_avoids = []
        opponent_attempts = []
        opponent_rates = []
        opponent_ys = []
        paddle_speed_rewards = []
        slow_hit_counts = []
        fast_hit_counts = []
        net_stick_rates = []
        sides = []
        side_success = {"left": 0, "right": 0}
        side_count = {"left": 0, "right": 0}

        for i in range(n_trials):
            env = PythonWallPickleballEnv(
                level=int(level),
                seed=seed + i,
                obs_source=obs_source,
                domain_randomization=False,
                trial_return_target=TRIAL_RETURN_TARGET,
            )
            obs, info = env.reset(seed=seed + i)
            done = False
            ep_reward = 0.0
            final_info = {}
            while not done:
                action, _ = model.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = env.step(action)
                ep_reward += float(reward)
                done = bool(terminated or truncated)
                final_info = info
            sc = int(final_info.get("success_count", 0))
            returns.append(sc)
            rewards.append(ep_reward)
            front_rates.append(float(final_info.get("front_camp_rate", 0.0)))
            net_action_rates.append(float(final_info.get("action_to_net_rate", 0.0)))
            avg_paddle_xs.append(float(final_info.get("avg_paddle_x", 0.0)))
            contacts.append(float(final_info.get("paddle_contacts", 0.0)))
            strong_bounces.append(float(final_info.get("strong_bounce_count", 0.0)))
            recovery_rewards.append(float(final_info.get("recovery_reward_count", 0.0)))
            home_distances.append(float(final_info.get("avg_home_distance", 0.0)))
            opponent_avoids.append(float(final_info.get("opponent_avoid_count", 0.0)))
            opponent_attempts.append(float(final_info.get("opponent_avoid_attempts", 0.0)))
            opponent_rates.append(float(final_info.get("opponent_avoid_rate", 0.0)))
            opponent_ys.append(float(final_info.get("opponent_y", 0.0)))
            side = str(final_info.get("side", "unknown"))
            sides.append(side)
            if side in side_count:
                side_count[side] += 1
            if sc >= TRIAL_RETURN_TARGET:
                successes += 1
                if side in side_success:
                    side_success[side] += 1
            if "fail_reason" in final_info or ep_reward < 0:
                failures += 1
            env.close()

        results[int(level)] = {
            "success_rate": successes / max(1, n_trials),
            "avg_returns": float(np.mean(returns)) if returns else 0.0,
            "avg_reward": float(np.mean(rewards)) if rewards else 0.0,
            "fail_rate": failures / max(1, n_trials),
            "front_camp_rate": float(np.mean(front_rates)) if front_rates else 0.0,
            "action_to_net_rate": float(np.mean(net_action_rates)) if net_action_rates else 0.0,
            "avg_paddle_x": float(np.mean(avg_paddle_xs)) if avg_paddle_xs else 0.0,
            "avg_contacts": float(np.mean(contacts)) if contacts else 0.0,
            "avg_strong_bounces": float(np.mean(strong_bounces)) if strong_bounces else 0.0,
            "avg_recovery_rewards": float(np.mean(recovery_rewards)) if recovery_rewards else 0.0,
            "avg_home_distance": float(np.mean(home_distances)) if home_distances else 0.0,
            "avg_opponent_avoid_count": float(np.mean(opponent_avoids)) if opponent_avoids else 0.0,
            "avg_opponent_avoid_attempts": float(np.mean(opponent_attempts)) if opponent_attempts else 0.0,
            "avg_opponent_avoid_rate": float(np.mean(opponent_rates)) if opponent_rates else 0.0,
            "avg_opponent_y": float(np.mean(opponent_ys)) if opponent_ys else 0.0,
            "avg_paddle_speed_reward": float(np.mean(paddle_speed_rewards)) if paddle_speed_rewards else 0.0,
            "avg_slow_paddle_hits": float(np.mean(slow_hit_counts)) if slow_hit_counts else 0.0,
            "avg_fast_paddle_hits": float(np.mean(fast_hit_counts)) if fast_hit_counts else 0.0,
            "net_stick_rate": float(np.mean(net_stick_rates)) if net_stick_rates else 0.0,
            "left_success_rate": side_success["left"] / max(1, side_count["left"]),
            "right_success_rate": side_success["right"] / max(1, side_count["right"]),
            "left_trials": side_count["left"],
            "right_trials": side_count["right"],
            "n_trials": n_trials,
        }

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="checkpoints/best_1.zip")
    parser.add_argument("--levels", type=int, nargs="+", default=[1])
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--obs-source", type=str, default=OBS_SOURCE, choices=["state", "opencv"])
    args = parser.parse_args()

    from stable_baselines3 import PPO

    model = PPO.load(args.model_path, device="cpu")
    results = evaluate_model(model, args.levels, args.trials, obs_source=args.obs_source)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
