# PPO training loop for the wall simulator.
from __future__ import annotations

import argparse
import os
import multiprocessing as mp
from pathlib import Path
import json

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

from config import (
    CHECKPOINT_DIR, LOG_DIR, CURRICULUM_STATE_PATH,
    NUM_ENVS, SEED, DEVICE, TRAIN_INTERVAL, MAX_TOTAL_STEPS,
    PASS_SUCCESS_RATE, EVAL_TRIALS, MAX_LEVEL, TOP_K_MODELS, OBS_SOURCE,
    MAX_EPISODE_STEPS, OWN_SIDE_TIMEOUT_STEPS, TRIAL_RETURN_TARGET,
    LEVEL3_MAX_TRAIN_STEPS,
)
from envs.python_wall_env import PythonWallPickleballEnv
from curriculum.curriculum_manager import CurriculumManager
from curriculum.checkpoint_manager import CheckpointManager, compute_validation_score
from evaluate_python_wall import evaluate_model


def make_env(level: int, rank: int, seed: int, obs_source: str):
    def _init():
        env = PythonWallPickleballEnv(
            level=level,
            seed=seed + rank,
            obs_source=obs_source,
            max_episode_steps=MAX_EPISODE_STEPS,
            own_side_timeout_steps=OWN_SIDE_TIMEOUT_STEPS,
            trial_return_target=TRIAL_RETURN_TARGET,
            domain_randomization=True,
        )
        return env
    return _init


def build_vec_env(level: int, num_envs: int, seed: int, obs_source: str, force_dummy: bool = False):
    env_fns = [make_env(level, i, seed, obs_source) for i in range(num_envs)]
    if num_envs <= 1 or force_dummy:
        env = DummyVecEnv(env_fns)
    else:
        env = SubprocVecEnv(env_fns, start_method="spawn")
    return VecMonitor(env)


def reset_curriculum_to_level(curriculum: CurriculumManager, level: int = 1) -> None:
    level = int(max(1, min(MAX_LEVEL, level)))
    curriculum.state = {
        "current_level": level,
        "total_timesteps": 0,
        "level_success_rates": {},
        "level_train_timesteps": {},
        "best_score": None,
    }
    curriculum.save()


def clear_local_top_models(checkpoint_dir: Path, keep_path: Path | None = None) -> None:
    keep_resolved = None
    if keep_path is not None:
        try:
            keep_resolved = Path(keep_path).resolve()
        except Exception:
            keep_resolved = None

    patterns = [
        "latest.zip", "latest_policy.pt",
        "best_*.zip", "best_*_policy.pt",
        "top_models.json",
    ]
    for pat in patterns:
        for p in Path(checkpoint_dir).glob(pat):
            try:
                if keep_resolved is not None and p.resolve() == keep_resolved:
                    continue
                p.unlink()
            except FileNotFoundError:
                pass
            except Exception as exc:
                print(f"[Warn] Could not delete old checkpoint file {p}: {exc}")


def preview_one_policy_rollout(model, level: int, obs_source: str, seed: int, steps: int = 500, delay_ms: int = 20, side: str = "random", title: str = "Policy preview") -> None:
    if steps <= 0:
        return
    try:
        import cv2
    except Exception as exc:
        print(f"[Preview] OpenCV preview skipped because cv2 import failed: {exc}")
        return

    env = PythonWallPickleballEnv(
        level=level,
        seed=seed,
        obs_source=obs_source,
        max_episode_steps=MAX_EPISODE_STEPS,
        own_side_timeout_steps=OWN_SIDE_TIMEOUT_STEPS,
        trial_return_target=TRIAL_RETURN_TARGET,
        domain_randomization=False,
        side=side,
    )
    obs, info = env.reset()
    total_reward = 0.0
    last_fail = ""

    window_name = f"{title} | level {level} | one env only"
    print(f"[Preview] Showing one policy rollout: level={level}, steps={steps}, side={side}. Press q to close early.")
    for t in range(int(steps)):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)

        img = env.render()
        cv2.putText(
            img,
            f"PREVIEW level={level} side={info.get('side','?')} t={t} R={total_reward:.1f} returns={info.get('success_count',0)} rec={info.get('recovery_reward_count',0)}",
            (3, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.28, (255, 255, 255), 1,
        )
        cv2.putText(
            img,
            f"act={list(map(int, action))} ball=({env.ball_x:.1f},{env.ball_y:.1f}) paddle=({env.paddle_x:.1f},{env.paddle_y:.1f}) {last_fail}",
            (3, 82), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 255), 1,
        )
        cv2.imshow(window_name, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        key = cv2.waitKey(max(1, int(delay_ms)))
        if key == ord("q"):
            break
        if terminated or truncated:
            last_fail = f"fail={info.get('fail_reason', 'reset')}"

            for _ in range(5):
                cv2.imshow(window_name, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
                if cv2.waitKey(max(1, int(delay_ms))) == ord("q"):
                    break
            obs, info = env.reset()
            total_reward = 0.0

    cv2.destroyWindow(window_name)
    env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=NUM_ENVS)
    parser.add_argument("--level", type=int, default=None)
    parser.add_argument("--obs-source", type=str, default=OBS_SOURCE, choices=["state", "opencv"])
    parser.add_argument("--train-interval", type=int, default=TRAIN_INTERVAL)
    parser.add_argument("--max-total-steps", type=int, default=MAX_TOTAL_STEPS)
    parser.add_argument("--eval-trials", type=int, default=EVAL_TRIALS)
    parser.add_argument("--level3-max-train-steps", type=int, default=LEVEL3_MAX_TRAIN_STEPS, help="Max training timesteps on level 3 before forcing advance to level 4.")
    parser.add_argument("--device", type=str, default=DEVICE, choices=["auto", "cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--dummy-vec", action="store_true", help="Use DummyVecEnv even with multiple envs.")
    parser.add_argument("--fresh", action="store_true", help="Ignore saved checkpoints and start new PPO.")
    parser.add_argument("--resume-best", action="store_true", help="Resume from best_1 instead of latest.zip. Default V54 behavior resumes latest to continue the current curriculum model.")
    parser.add_argument("--pretrained-zip", type=str, default=None, help="Load an external SB3 .zip as pretrained weights, but restart this curriculum from level 1 by default.")
    parser.add_argument("--restart-from-level1", action="store_true", help="Reset curriculum_state.json to level 1 and total_timesteps 0 before training.")
    parser.add_argument("--pretrained-start-level", type=int, default=1, help="Starting level used when --pretrained-zip or --restart-from-level1 is given.")
    parser.add_argument("--keep-old-top-models", action="store_true", help="Do not clear old best_*.zip/top_models.json when using --pretrained-zip.")
    parser.add_argument("--preview-each-iteration", action="store_true", help="Before every training interval, show one rollout from one standalone env only.")
    parser.add_argument("--preview-steps", type=int, default=500, help="Number of preview steps to show each iteration.")
    parser.add_argument("--preview-delay-ms", type=int, default=20, help="OpenCV waitKey delay for preview window.")
    parser.add_argument("--preview-side", type=str, default="random", choices=["random", "left", "right"], help="Which side to preview.")
    args = parser.parse_args()

    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    curriculum = CurriculumManager(CURRICULUM_STATE_PATH, max_level=MAX_LEVEL)

    pretrained_path = Path(args.pretrained_zip).expanduser() if args.pretrained_zip else None
    if pretrained_path is not None:
        if not pretrained_path.is_absolute():
            pretrained_path = (Path.cwd() / pretrained_path).resolve()
        if not pretrained_path.exists():
            raise FileNotFoundError(f"--pretrained-zip not found: {pretrained_path}")


        reset_curriculum_to_level(curriculum, args.pretrained_start_level)
        if not args.keep_old_top_models:
            clear_local_top_models(CHECKPOINT_DIR, keep_path=pretrained_path)
        print(f"[Pretrained] Will load weights from: {pretrained_path}")
        print(f"[Pretrained] Curriculum restarted at level {curriculum.current_level}, total_timesteps={curriculum.total_timesteps}")

    elif args.restart_from_level1:
        reset_curriculum_to_level(curriculum, args.pretrained_start_level)
        print(f"[Restart] Curriculum restarted at level {curriculum.current_level}, total_timesteps={curriculum.total_timesteps}")

    if args.level is not None:
        curriculum.set_level(args.level)


    if args.fresh and pretrained_path is None:
        reset_curriculum_to_level(curriculum, 1)
        clear_local_top_models(CHECKPOINT_DIR)
        print("[Fresh V57] Cleared old checkpoints/top_models and restarted curriculum at level 1.")

    ckpt = CheckpointManager(CHECKPOINT_DIR, top_k=TOP_K_MODELS)

    current_level = curriculum.current_level
    print(f"[Setup] Python Wall PPO")
    print(f"  current_level = {current_level}")
    print(f"  num_envs      = {args.num_envs}")
    print(f"  obs_source    = {args.obs_source}")
    print(f"  pretrained_zip= {pretrained_path if pretrained_path is not None else 'None'}")
    print(f"  checkpoint_dir= {CHECKPOINT_DIR}")
    print(f"  preview_each_iteration = {args.preview_each_iteration}")

    env = build_vec_env(current_level, args.num_envs, args.seed, args.obs_source, force_dummy=args.dummy_vec)

    if pretrained_path is not None:
        print(f"[Pretrained] Loading external model {pretrained_path}")
        model = PPO.load(str(pretrained_path), env=env, device=args.device)


    elif (not args.fresh) and args.resume_best and ckpt.has_best():
        print(f"[Resume-Best] Loading {ckpt.best_path(1)}")
        model = PPO.load(str(ckpt.best_path(1)), env=env, device=args.device)
    elif (not args.fresh) and ckpt.latest_path().exists():


        print(f"[Resume-Latest] Loading {ckpt.latest_path()}")
        model = PPO.load(str(ckpt.latest_path()), env=env, device=args.device)
    elif (not args.fresh) and ckpt.has_best():
        print(f"[Resume-Fallback-Best] Loading {ckpt.best_path(1)}")
        model = PPO.load(str(ckpt.best_path(1)), env=env, device=args.device)
    else:
        print("[New] Creating PPO MlpPolicy model")
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=1024,
            batch_size=256,
            n_epochs=8,
            gamma=0.995,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            verbose=1,
            tensorboard_log=str(LOG_DIR / "tensorboard"),
            device=args.device,
            seed=args.seed,
            policy_kwargs=dict(net_arch=dict(pi=[128, 128, 64], vf=[128, 128, 64])),
        )

    env_level = current_level

    try:
        while curriculum.total_timesteps < args.max_total_steps:
            current_level = curriculum.current_level
            print(f"\n[Train] level={current_level}, total_done={curriculum.total_timesteps}")


            if current_level != env_level:
                print(f"[Env] Curriculum changed: rebuilding workers for level {current_level}")
                try:
                    model.get_env().close()
                except Exception:
                    pass
                model.set_env(build_vec_env(current_level, args.num_envs, args.seed, args.obs_source, force_dummy=args.dummy_vec))
                env_level = current_level

            if args.preview_each_iteration:
                preview_one_policy_rollout(
                    model=model,
                    level=current_level,
                    obs_source=args.obs_source,
                    seed=args.seed + 90000 + int(curriculum.total_timesteps),
                    steps=args.preview_steps,
                    delay_ms=args.preview_delay_ms,
                    side=args.preview_side,
                    title="Iteration policy preview",
                )


            level_train_steps = curriculum.state.setdefault("level_train_timesteps", {})
            trained_on_current_level = int(level_train_steps.get(str(current_level), 0))
            train_steps_this_iter = int(args.train_interval)

            model.learn(
                total_timesteps=train_steps_this_iter,
                reset_num_timesteps=False,
                tb_log_name="python_wall_ppo",
                progress_bar=False,
            )
            curriculum.add_timesteps(train_steps_this_iter)
            level_train_steps = curriculum.state.setdefault("level_train_timesteps", {})
            level_train_steps[str(current_level)] = int(level_train_steps.get(str(current_level), 0)) + int(train_steps_this_iter)
            curriculum.save()
            ckpt.save_latest(model)


            unlocked_levels = list(range(1, current_level + 1))
            eval_result = evaluate_model(
                model,
                levels=unlocked_levels,
                n_trials=args.eval_trials,
                seed=5000,
                obs_source=args.obs_source,
                deterministic=True,
            )
            curriculum.update_eval(eval_result)
            score = compute_validation_score(eval_result, current_level=current_level)

            current_success = eval_result[current_level]["success_rate"]
            metadata = {
                "total_timesteps": curriculum.total_timesteps,
                "current_level": current_level,
                "rank_level": current_level,
                "score": score,
                "eval_result": eval_result,
            }
            saved = ckpt.update_top_k(model, score, metadata)
            print(f"[Eval] score={score:.2f}, current_success={current_success:.3f}, saved_top={saved}")
            ckpt.print_top_k()
            print(json.dumps(eval_result, indent=2))

            advanced = curriculum.maybe_advance(current_success, PASS_SUCCESS_RATE)
            if advanced:
                print(f"[Curriculum] Passed level {current_level}; advancing to {curriculum.current_level}")
            elif current_level >= MAX_LEVEL and current_success >= PASS_SUCCESS_RATE:
                print(f"[Curriculum] Level {MAX_LEVEL} passed. Continuing Level {MAX_LEVEL} to improve average returns/robustness.")

    except KeyboardInterrupt:
        print("[Stop] Keyboard interrupt. Saving latest checkpoint...")
        ckpt.save_latest(model)
    finally:
        try:
            model.get_env().close()
        except Exception:
            pass
        curriculum.save()
        print("[Done] Training closed safely.")


if __name__ == "__main__":
    mp.freeze_support()
    main()
