from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import List

import numpy as np
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

from real_dpickleball_env import make_real_env_fn
from league_selector import LeagueSelector
from export_sb3_to_torchscript import export_sb3_policy_to_torchscript


class ChunkMetricsCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.rewards: List[float] = []
        self.agent_points = 0
        self.opponent_points = 0
        self.matches = 0

    def _on_step(self) -> bool:
        rewards = self.locals.get('rewards', [])
        try:
            for r in np.asarray(rewards).reshape(-1):
                self.rewards.append(float(r))
        except Exception:
            pass
        infos = self.locals.get('infos', []) or []
        for info in infos:
            if not isinstance(info, dict):
                continue
            if info.get('point_ended'):
                if info.get('point_winner') == 'agent':
                    self.agent_points += 1
                elif info.get('point_winner') == 'opponent':
                    self.opponent_points += 1
            if info.get('match_over'):
                self.matches += 1
        return True

    @property
    def mean_reward(self) -> float:
        return float(np.mean(self.rewards)) if self.rewards else 0.0



class ActorCriticWarmupCallback(BaseCallback):
    """Freeze the actor for a short critic-only warm-up after loading converted actor-only starts.

    This is useful for starts converted from TorchScript deployment actors, where
    the actor is meaningful but the PPO critic/value function is newly initialized
    or only approximate.
    """

    def __init__(self, warmup_steps: int = 0):
        super().__init__()
        self.warmup_steps = int(max(0, warmup_steps))
        self.start_timesteps = None
        self.frozen = False

    def _actor_params(self):
        yield from self.model.policy.mlp_extractor.policy_net.parameters()
        yield from self.model.policy.action_net.parameters()

    def _set_actor_trainable(self, trainable: bool) -> None:
        for p in self._actor_params():
            p.requires_grad = bool(trainable)

    def _on_training_start(self) -> None:
        if self.start_timesteps is None:
            self.start_timesteps = int(self.model.num_timesteps)
            if self.warmup_steps > 0:
                self._set_actor_trainable(False)
                self.frozen = True
                print(f'[critic warmup] actor frozen for first {self.warmup_steps} steps; critic trains first', flush=True)

    def _on_step(self) -> bool:
        if self.frozen and self.start_timesteps is not None:
            progressed = int(self.model.num_timesteps) - int(self.start_timesteps)
            if progressed >= self.warmup_steps:
                self._set_actor_trainable(True)
                self.frozen = False
                print(f'[critic warmup] actor unfrozen after {progressed} steps', flush=True)
        return True

    def _on_training_end(self) -> None:
        if self.frozen:
            self._set_actor_trainable(True)
            self.frozen = False


def reset_policy_optimizer(model, learning_rate: float) -> None:
    """Drop loaded optimizer momentum and create a fresh optimizer for continued PPO training."""
    opt_cls = getattr(model.policy, 'optimizer_class', torch.optim.Adam)
    opt_kwargs = dict(getattr(model.policy, 'optimizer_kwargs', {}) or {})
    opt_kwargs.pop('lr', None)
    model.policy.optimizer = opt_cls(model.policy.parameters(), lr=float(learning_rate), **opt_kwargs)
    print(f'[optimizer] reset policy optimizer with lr={float(learning_rate):g}', flush=True)


def configure_torch_threads(n: int) -> None:
    n = max(1, int(n))
    os.environ.setdefault('OMP_NUM_THREADS', str(n))
    os.environ.setdefault('MKL_NUM_THREADS', str(n))
    os.environ.setdefault('OPENBLAS_NUM_THREADS', str(n))
    os.environ.setdefault('NUMEXPR_NUM_THREADS', str(n))
    try:
        import torch
        torch.set_num_threads(n)
        try:
            torch.set_num_interop_threads(n)
        except Exception:
            pass
        print(f'[torch threads] {n}', flush=True)
    except Exception as e:
        print(f'[torch threads] unable to configure: {e}', flush=True)


def resolve_default_build_path() -> str:
    candidate = Path(r'C:\Users\User\dpickleball-ml-agents\dPickleball BuildFiles\Training\Windows\dp.exe')
    return str(candidate)


def create_model(args, env):
    if args.start_zip:
        start_zip = Path(args.start_zip)
        if not start_zip.is_absolute():
            start_zip = Path(__file__).resolve().parent / start_zip
        if not start_zip.exists():
            raise FileNotFoundError(f'--start-zip not found: {start_zip}')
        args.start_zip = str(start_zip)
    policy_kwargs = dict(net_arch=dict(pi=[128, 128], vf=[128, 128]))
    if args.start_zip:
        print(f'[load] warm-up/start PPO zip: {args.start_zip}', flush=True)
        custom_objects = {
            'observation_space': env.observation_space,
            'action_space': env.action_space,
            'tensorboard_log': None,
        }
        model = PPO.load(
            args.start_zip,
            env=env,
            device=args.device,
            print_system_info=False,
            custom_objects=custom_objects,
        )
        model.tensorboard_log = None
        model._logger = None
        return model
    print('[new] creating fresh PPO model', flush=True)
    return PPO(
        'MlpPolicy',
        env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        policy_kwargs=policy_kwargs,
        verbose=args.verbose,
        device=args.device,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description='Real dPickleBall League-PSRO-style PPO training')
    ap.add_argument('--build-path', default=resolve_default_build_path(), help='Path to Unity build executable, e.g. ...\\dPickleball BuildFiles\\Training\\Windows\\dp.exe')
    ap.add_argument('--trainer-pool', default='league_pool_filtered', help='Folder of TorchScript .pt fixed opponents')
    ap.add_argument('--output-dir', default='real_league_outputs')
    ap.add_argument('--start-zip', default=r'warmup_start\current_model.zip', help='Optional SB3 PPO .zip from virtual-wall warm-up/final warm-up model. Empty string starts fresh.')
    ap.add_argument('--reset-optimizer', action='store_true', help='After PPO.load(), discard loaded optimizer state and start with a fresh Adam optimizer. Recommended for converted/lifted starts.')
    ap.add_argument('--critic-warmup-steps', type=int, default=0, help='Freeze actor for this many steps after loading a converted actor-only start so the critic can adapt first.')
    ap.add_argument('--opponent-mode', default='auto', choices=['auto', 'left_mirror', 'native'], help='How to run fixed right-side opponent .pt files. auto is recommended.')
    ap.add_argument('--total-steps', type=int, default=1_000_000)
    ap.add_argument('--chunk-steps', type=int, default=20_000)
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--device', default='cpu', choices=['cpu', 'cuda', 'auto'])
    ap.add_argument('--base-port', type=int, default=5005)
    ap.add_argument('--worker-id', type=int, default=0)
    ap.add_argument('--no-graphics', action='store_true', default=False)
    ap.add_argument('--show-graphics', action='store_false', dest='no_graphics')
    ap.add_argument('--timeout-wait', type=int, default=180, help='Seconds to wait for Unity ML-Agents handshake. First launch on Windows can be slow.')
    ap.add_argument('--low-res', action='store_true', help='Graphics mode only: try launching Unity in a tiny 320x180 window if ML-Agents supports additional_args. Safer than --no-graphics for this build.')
    ap.add_argument('--time-scale', type=float, default=1.0, help='Unity EngineConfigurationChannel time_scale. Try 2 or 3; too high may destabilize physics/vision.')
    ap.add_argument('--target-frame-rate', type=int, default=-1, help='Unity target_frame_rate via EngineConfigurationChannel. -1 uses platform default; 60/120 may help consistency.')
    ap.add_argument('--num-envs', type=int, default=1, help='Number of Unity instances. Start with 1. Use 2-4 only after single-env launch works.')
    ap.add_argument('--max-episode-steps', type=int, default=30000)
    ap.add_argument('--sfl-peak', type=float, default=0.50)
    ap.add_argument('--epsilon', type=float, default=0.08)
    ap.add_argument('--learning-rate', type=float, default=2.5e-4)
    ap.add_argument('--n-steps', type=int, default=1024)
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--n-epochs', type=int, default=4)
    ap.add_argument('--gamma', type=float, default=0.995)
    ap.add_argument('--gae-lambda', type=float, default=0.95)
    ap.add_argument('--clip-range', type=float, default=0.20)
    ap.add_argument('--ent-coef', type=float, default=0.01)
    ap.add_argument('--vf-coef', type=float, default=0.50)
    ap.add_argument('--max-grad-norm', type=float, default=0.50)
    ap.add_argument('--save-every-chunks', type=int, default=5)
    ap.add_argument('--add-selfplay-to-pool', action='store_true', help='Add exported learner checkpoints back into trainer pool')
    ap.add_argument('--torch-threads', type=int, default=1)
    ap.add_argument('--verbose', type=int, default=1)
    args = ap.parse_args()

    configure_torch_threads(args.torch_threads)

    root = Path(__file__).resolve().parent
    trainer_pool = Path(args.trainer_pool)
    if not trainer_pool.is_absolute():
        trainer_pool = root / trainer_pool
    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        output_dir = root / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'checkpoints').mkdir(parents=True, exist_ok=True)
    (output_dir / 'exported_pt').mkdir(parents=True, exist_ok=True)
    (output_dir / 'reports').mkdir(parents=True, exist_ok=True)

    selector = LeagueSelector(
        trainer_pool=trainer_pool,
        state_path=output_dir / 'league_state.json',
        sfl_peak=args.sfl_peak,
        seed=args.seed,
    )
    opponent = selector.choose(epsilon=args.epsilon)
    print(f'[opponent] initial: {opponent.name}', flush=True)

    env_fns = []
    for env_idx in range(max(1, int(args.num_envs))):
        env_fns.append(make_real_env_fn(
            build_path=args.build_path,
            opponent_model_path=opponent,
            worker_id=args.worker_id + env_idx,
            base_port=args.base_port,
            no_graphics=args.no_graphics,
            seed=args.seed + env_idx * 1000,
            max_steps=args.max_episode_steps,
            opponent_mode=args.opponent_mode,
            timeout_wait=args.timeout_wait,
            low_res=args.low_res,
            time_scale=args.time_scale,
            target_frame_rate=args.target_frame_rate,
        ))
    if len(env_fns) <= 1:
        env = VecMonitor(DummyVecEnv(env_fns))
    else:
        print(f'[vecenv] using SubprocVecEnv with {len(env_fns)} Unity instances', flush=True)
        env = VecMonitor(SubprocVecEnv(env_fns, start_method='spawn'))

    try:
        model = create_model(args, env)
        if args.reset_optimizer:
            reset_policy_optimizer(model, args.learning_rate)
        warmup_cb = ActorCriticWarmupCallback(args.critic_warmup_steps) if args.critic_warmup_steps > 0 else None
        total_done = 0
        chunk_id = int(selector.state.get('global_chunk', 0))
        while total_done < args.total_steps:
            opponent = selector.choose(epsilon=args.epsilon)
            print(f'\n[chunk {chunk_id}] opponent={opponent.name}', flush=True)
            env.env_method('set_opponent', str(opponent), opponent_mode=args.opponent_mode)

            cb = ChunkMetricsCallback()
            before = time.time()
            callbacks = [cb]
            if warmup_cb is not None:
                callbacks.append(warmup_cb)
            model.learn(total_timesteps=args.chunk_steps, reset_num_timesteps=False, callback=callbacks, progress_bar=False)
            elapsed = time.time() - before
            total_done += args.chunk_steps

            try:
                env_metrics = env.env_method('pop_chunk_metrics')[0]
            except Exception:
                env_metrics = {'agent_points': 0, 'opponent_points': 0, 'matches': 0}
            agent_points = max(int(cb.agent_points), int(env_metrics.get('agent_points', 0)))
            opponent_points = max(int(cb.opponent_points), int(env_metrics.get('opponent_points', 0)))
            matches = max(int(cb.matches), int(env_metrics.get('matches', 0)))
            mean_reward = cb.mean_reward
            success = agent_points / max(1, agent_points + opponent_points)

            selector.update_after_chunk(
                opponent.name,
                agent_points=agent_points,
                opponent_points=opponent_points,
                matches=matches,
                mean_reward=mean_reward,
            )
            selector.export_csv(output_dir / 'reports' / 'league_records.csv')

            print(
                f'[chunk {chunk_id}] pts={agent_points}:{opponent_points} '
                f'success={success:.3f} matches={matches} mean_reward={mean_reward:.5f} '
                f'elapsed={elapsed/60:.1f}min',
                flush=True,
            )

            if chunk_id % max(1, args.save_every_chunks) == 0:
                zip_path = output_dir / 'checkpoints' / f'ppo_real_chunk_{chunk_id:05d}.zip'
                pt_path = output_dir / 'exported_pt' / f'ppo_real_chunk_{chunk_id:05d}.pt'
                model.save(str(zip_path))
                export_sb3_policy_to_torchscript(
                    model,
                    pt_path,
                    metadata={
                        'training_env': 'real_dpickleball_unity',
                        'chunk_id': chunk_id,
                        'total_done_this_run': total_done,
                        'last_opponent': opponent.name,
                        'note': 'Left-normalized learner. Use TeamY mirroring for right-side deployment.',
                    },
                )
                print(f'[save] {zip_path.name} and {pt_path.name}', flush=True)
                if args.add_selfplay_to_pool:
                    added = selector.add_checkpoint_to_pool(pt_path, prefix='selfplay')
                    print(f'[pool] added self-play checkpoint: {added.name}', flush=True)

            chunk_id += 1

        final_zip = output_dir / 'checkpoints' / 'ppo_real_latest.zip'
        final_pt = output_dir / 'exported_pt' / 'ppo_real_latest.pt'
        model.save(str(final_zip))
        export_sb3_policy_to_torchscript(model, final_pt, metadata={'training_env': 'real_dpickleball_unity', 'final': True})
        selector.export_csv(output_dir / 'reports' / 'league_records_final.csv')
        print(f'\n[done] final saved: {final_zip} and {final_pt}', flush=True)

    finally:
        env.close()


if __name__ == '__main__':
    main()
