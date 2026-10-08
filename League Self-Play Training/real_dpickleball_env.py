from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception:  # SB3 older stacks may still use gym
    import gym
    from gym import spaces

import torch

from opencv_state_extractor import (
    OpenCVStateExtractor,
    mirror_features_right_to_left_view,
    mirror_action,
)


class TorchScriptPolicy:
    """TorchScript dPickleBall policy wrapper for fixed league opponents.

    During LEFT-agent training, the fixed opponent plays on the RIGHT side.
    Some uploaded .pt files are pure left-side actors, while some are already
    right-side/mirror wrappers. This class supports both:

    - opponent_mode='left_mirror': extract right-side features, mirror into
      the left-side model view, run the model, then mirror the action back.
    - opponent_mode='native': extract right-side features and use the model's
      returned action directly. Use this for right-side / wrapped / universal
      policies.
    - opponent_mode='auto': infer mode from the .pt.json metadata and filename.
    """

    def __init__(
        self,
        model_path: str | Path,
        side: str = 'right',
        opponent_mode: str = 'auto',
    ):
        self.model_path = str(model_path)
        self.side = str(side)
        self.opponent_mode = self._resolve_mode(Path(model_path), opponent_mode)
        self.extractor = OpenCVStateExtractor(side=self.side)
        self.model = torch.jit.load(self.model_path, map_location='cpu')
        self.model.eval()
        try:
            torch.set_num_threads(1)
        except Exception:
            pass
        self.reset()

    @staticmethod
    def _read_metadata(path: Path) -> Dict[str, Any]:
        meta_path = path.with_suffix(path.suffix + '.json')
        if not meta_path.exists():
            return {}
        try:
            return json.loads(meta_path.read_text(encoding='utf-8'))
        except Exception:
            return {}

    @classmethod
    def _resolve_mode(cls, path: Path, requested: str) -> str:
        requested = str(requested or 'auto').lower()
        if requested in {'left_mirror', 'native'}:
            return requested
        if requested != 'auto':
            raise ValueError('opponent_mode must be auto, left_mirror, or native')

        meta = cls._read_metadata(path)
        text = (path.name + ' ' + json.dumps(meta, sort_keys=True)).lower()

        native_keywords = [
            'role": "right',
            'use for right agent',
            'weight-baked mirror',
            'torchscript_left_strong_mirror_right_wrapper',
            'right_only',
            'right_mirror',
            'right_no_',
            'right_v',
            'universal_',
        ]
        if any(k in text for k in native_keywords):
            return 'native'

        return 'left_mirror'

    def reset(self) -> None:
        self.extractor.reset()

    def predict(self, visual_obs: np.ndarray) -> np.ndarray:
        native = self.extractor.features_from_image(visual_obs)
        if self.side == 'right' and self.opponent_mode == 'left_mirror':
            model_features = mirror_features_right_to_left_view(native)
        else:
            model_features = native

        x = torch.as_tensor(model_features, dtype=torch.float32)
        with torch.no_grad():
            a = self.model(x).detach().cpu().numpy().astype(np.int32).reshape(-1)[:3]
        a = np.clip(a, 0, 2).astype(np.int32)

        if self.side == 'right' and self.opponent_mode == 'left_mirror':
            a = np.asarray(mirror_action(a), dtype=np.int32)
        return np.clip(a, 0, 2).astype(np.int32)


class DPickleballUnityPPOEnv(gym.Env):
    """Gym/Gymnasium wrapper for real Unity dPickleBall PPO training.

    This version trains ONE learner only: the LEFT-side paddle. The RIGHT-side
    paddle is a fixed TorchScript league opponent. The returned observation is
    the 16-dimensional OpenCV feature vector used by the warm-up models.
    """

    metadata = {'render_modes': []}

    def __init__(
        self,
        build_path: str | Path,
        opponent_model_path: str | Path,
        *,
        learner_side: str = 'left',
        opponent_mode: str = 'auto',
        worker_id: int = 0,
        base_port: int = 5005,
        no_graphics: bool = True,
        seed: int = 2026,
        timeout_wait: int = 60,
        max_steps: int = 30000,
        serve_code: int = 212,
        low_res: bool = False,
        time_scale: float = 1.0,
        target_frame_rate: int = -1,
    ):
        super().__init__()
        self.build_path = str(build_path)
        self.learner_side = str(learner_side)
        if self.learner_side != 'left':
            raise ValueError('This wrapper intentionally trains learner_side="left" only. Deploy right side by TeamY mirroring.')
        self.opponent_mode = str(opponent_mode)
        self.worker_id = int(worker_id)
        self.base_port = int(base_port)
        self.no_graphics = bool(no_graphics)
        self.seed_value = int(seed)
        self.timeout_wait = int(timeout_wait)
        self.max_steps = int(max_steps)
        self.serve_code = int(serve_code)
        self.low_res = bool(low_res)
        self.time_scale = float(time_scale)
        self.target_frame_rate = int(target_frame_rate)

        self.action_space = spaces.MultiDiscrete([3, 3, 3])
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(16,), dtype=np.float32)

        self.extractor = OpenCVStateExtractor(side='left')
        self.opponent = TorchScriptPolicy(opponent_model_path, side='right', opponent_mode=self.opponent_mode)
        self.opponent_model_path = str(opponent_model_path)

        self.unity_env = None
        self.env = None
        self.agents = []
        self.current_obs_dict = None
        self.current_features = None
        self.steps = 0
        self.agent_points_chunk = 0
        self.opponent_points_chunk = 0
        self.matches_chunk = 0
        self.last_score_left = 0
        self.last_score_right = 0
        self._launch()

    def _launch(self) -> None:
        from mlagents_envs.environment import UnityEnvironment
        from mlagents_envs.envs.unity_parallel_env import UnityParallelEnv
        from mlagents_envs.envs.custom_side_channel import CustomDataChannel, StringSideChannel
        try:
            from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
        except Exception:
            EngineConfigurationChannel = None

        if not Path(self.build_path).exists():
            raise FileNotFoundError(f'Unity build not found: {self.build_path}')

        self.string_channel = StringSideChannel()
        self.custom_channel = CustomDataChannel()
        self.engine_channel = EngineConfigurationChannel() if EngineConfigurationChannel is not None else None
        try:
            self.custom_channel.send_data(serve=self.serve_code, p1=0, p2=0)
        except Exception:
            pass

        unity_kwargs = dict(
            file_name=self.build_path,
            worker_id=self.worker_id,
            base_port=self.base_port,
            side_channels=[c for c in [self.string_channel, self.custom_channel, self.engine_channel] if c is not None],
            no_graphics=self.no_graphics,
            timeout_wait=self.timeout_wait,
        )
        if self.low_res and not self.no_graphics:
            try:
                import inspect
                if 'additional_args' in inspect.signature(UnityEnvironment.__init__).parameters:
                    unity_kwargs['additional_args'] = [
                        '-screen-width', '320',
                        '-screen-height', '180',
                        '-screen-fullscreen', '0',
                        '-window-mode', 'windowed',
                        '-force-d3d11',
                    ]
                    print('[unity] low-res graphics args enabled: 320x180 windowed', flush=True)
                else:
                    print('[unity] low-res requested but this ML-Agents UnityEnvironment has no additional_args support; continuing normal graphics.', flush=True)
            except Exception as exc:
                print(f'[unity] low-res arg detection failed ({exc}); continuing normal graphics.', flush=True)
        self.unity_env = UnityEnvironment(**unity_kwargs)
        if self.engine_channel is not None:
            try:
                self.engine_channel.set_configuration_parameters(
                    time_scale=float(self.time_scale),
                    target_frame_rate=int(self.target_frame_rate),
                )
                print(f'[unity] engine config: time_scale={self.time_scale:g}, target_frame_rate={self.target_frame_rate}', flush=True)
            except Exception as exc:
                print(f'[unity] engine config failed ({exc}); continuing default speed.', flush=True)
        else:
            if abs(float(self.time_scale) - 1.0) > 1e-6 or int(self.target_frame_rate) != -1:
                print('[unity] EngineConfigurationChannel unavailable; continuing default speed.', flush=True)
        self.env = UnityParallelEnv(self.unity_env)

    def set_opponent(self, model_path: str | Path, *, opponent_mode: Optional[str] = None) -> None:
        self.opponent_model_path = str(model_path)
        self.opponent = TorchScriptPolicy(
            model_path,
            side='right',
            opponent_mode=self.opponent_mode if opponent_mode is None else opponent_mode,
        )

    def _extract_visual(self, obs_dict: Dict[str, Any]) -> np.ndarray:
        if not self.agents:
            self.agents = list(self.env.agents)
        first = self.agents[0]
        obs = obs_dict[first]
        if isinstance(obs, dict):
            val = obs.get('observation', obs)
            if isinstance(val, (list, tuple)):
                return np.asarray(val[0])
            return np.asarray(val)
        if isinstance(obs, (list, tuple)):
            return np.asarray(obs[0])
        return np.asarray(obs)

    def _features(self, visual: np.ndarray) -> np.ndarray:
        features = self.extractor.features_from_image(visual).astype(np.float32)
        if features.shape != (16,):
            raise RuntimeError(f'Expected 16-dim feature vector, got shape={features.shape}')
        return features

    @staticmethod
    def _reward_from_pair(learner_reward: float, opponent_reward: float) -> Tuple[float, Optional[str]]:
        lr = float(learner_reward)
        rr = float(opponent_reward)
        if lr > 0:
            return 1.0, 'agent'
        if lr < 0:
            return -1.0, 'opponent'
        if rr > 0:
            return -1.0, 'opponent'
        if rr < 0:
            return 1.0, 'agent'
        return 0.0, None


    @staticmethod
    def _competition_postprocess_left_action(action: np.ndarray, features: Optional[np.ndarray]) -> np.ndarray:
        """Match deploy_template/teamX.py runtime safety postprocess during training.

        TeamX deployment uses a small guard to avoid vertical twitching when the
        ball detector is uncertain or when the paddle is already close enough in
        y. Training with the same guard makes the real Unity experience closer to
        what will happen in the official Competition.py loop.
        """
        a = np.asarray(action, dtype=np.int32).reshape(-1)[:3].copy()
        a = np.clip(a, 0, 2).astype(np.int32)
        if features is None:
            return a
        try:
            if float(features[12]) < 0.5:
                a[0] = 0
            elif abs(float(features[9])) < 0.08:
                a[0] = 0
        except Exception:
            pass
        return np.clip(a, 0, 2).astype(np.int32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        if seed is not None:
            self.seed_value = int(seed)
        self.extractor.reset()
        self.opponent.reset()
        self.steps = 0
        self.agents = []
        self.current_obs_dict = self.env.reset()
        self.agents = list(self.env.agents)
        visual = self._extract_visual(self.current_obs_dict)
        obs = self._features(visual)
        self.current_features = obs.copy()
        info = {
            'opponent_model': Path(self.opponent_model_path).name,
            'opponent_mode': self.opponent.opponent_mode,
        }
        return obs, info

    def step(self, action):
        if self.current_obs_dict is None:
            self.reset()

        visual = self._extract_visual(self.current_obs_dict)
        raw_learner_action = np.asarray(action, dtype=np.int32).reshape(-1)[:3]
        learner_action = self._competition_postprocess_left_action(raw_learner_action, self.current_features)
        opponent_action = self.opponent.predict(visual)

        if not self.agents:
            self.agents = list(self.env.agents)
        left_agent = self.agents[0]
        right_agent = self.agents[1]
        actions = {left_agent: learner_action, right_agent: opponent_action}

        obs_dict, rewards, dones, infos = self.env.step(actions)
        self.current_obs_dict = obs_dict
        self.steps += 1

        lr = float(rewards.get(left_agent, 0.0))
        rr = float(rewards.get(right_agent, 0.0))
        reward, winner = self._reward_from_pair(lr, rr)
        point_ended = winner is not None
        if winner == 'agent':
            self.agent_points_chunk += 1
            self.last_score_left += 1
        elif winner == 'opponent':
            self.opponent_points_chunk += 1
            self.last_score_right += 1

        terminated = bool(dones.get(left_agent, False) or dones.get(right_agent, False))
        truncated = bool(self.steps >= self.max_steps)
        if terminated or truncated:
            self.matches_chunk += 1

        visual_next = self._extract_visual(obs_dict)
        obs_next = self._features(visual_next)
        self.current_features = obs_next.copy()
        info = {
            'opponent_model': Path(self.opponent_model_path).name,
            'opponent_mode': self.opponent.opponent_mode,
            'point_ended': point_ended,
            'point_winner': winner,
            'raw_reward_left': lr,
            'raw_reward_right': rr,
            'raw_learner_action': raw_learner_action.tolist(),
            'executed_learner_action': learner_action.tolist(),
            'agent_score': self.last_score_left,
            'opponent_score': self.last_score_right,
            'match_over': terminated or truncated,
        }
        if isinstance(infos, dict):
            unity_info = infos.get(left_agent, {}) if left_agent in infos else {}
            if isinstance(unity_info, dict):
                for k, v in unity_info.items():
                    info.setdefault(f'unity_{k}', v)
        return obs_next, reward, terminated, truncated, info

    def pop_chunk_metrics(self) -> Dict[str, int]:
        data = {
            'agent_points': int(self.agent_points_chunk),
            'opponent_points': int(self.opponent_points_chunk),
            'matches': int(self.matches_chunk),
        }
        self.agent_points_chunk = 0
        self.opponent_points_chunk = 0
        self.matches_chunk = 0
        return data

    def close(self) -> None:
        try:
            if self.env is not None:
                self.env.close()
        finally:
            self.env = None
            self.unity_env = None


def make_real_env_fn(
    build_path: str | Path,
    opponent_model_path: str | Path,
    *,
    worker_id: int,
    base_port: int,
    no_graphics: bool,
    low_res: bool = False,
    seed: int,
    max_steps: int,
    opponent_mode: str = 'auto',
    timeout_wait: int = 180,
    time_scale: float = 1.0,
    target_frame_rate: int = -1,
):
    def _init():
        return DPickleballUnityPPOEnv(
            build_path=build_path,
            opponent_model_path=opponent_model_path,
            worker_id=worker_id,
            base_port=base_port,
            no_graphics=no_graphics,
            seed=seed + worker_id,
            max_steps=max_steps,
            opponent_mode=opponent_mode,
            timeout_wait=timeout_wait,
            low_res=low_res,
            time_scale=time_scale,
            target_frame_rate=target_frame_rate,
        )
    return _init
