# Wall simulator environment and reward logic.
from __future__ import annotations

import math
import random
from typing import Optional, Tuple, Dict, Any

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception as exc:
    raise ImportError(
        "gymnasium is required. Install with: python -m pip install gymnasium"
    ) from exc

from envs.court_renderer import CourtRenderer
from envs.opencv_state_extractor import OpenCVStateExtractor
from curriculum.wall_levels import get_level


class PythonWallPickleballEnv(gym.Env):
    """Wall simulator used for PPO foundation training."""

    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 30}

    def __init__(
        self,
        level: int = 1,
        seed: int = 0,
        obs_source: str = "state",
        render_mode: Optional[str] = None,
        max_episode_steps: int = 900,
        own_side_timeout_steps: int = 150,
        trial_return_target: int = 3,
        side: str = "random",
        domain_randomization: bool = True,
    ):
        super().__init__()
        self.level = int(level)
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.obs_source = obs_source
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.own_side_timeout_steps = own_side_timeout_steps
        self.trial_return_target = trial_return_target
        self.side_mode = str(side)
        self.active_side = "right"
        self.domain_randomization = domain_randomization

        self.w = 168
        self.h = 84
        self.court_x1 = 18
        self.court_x2 = 158
        self.court_y1 = 27
        self.court_y2 = 72
        self.middle_x1 = 68
        self.middle_x2 = 108


        self.net_x = 84
        self.right_agent_wall_x = self.middle_x1
        self.left_agent_wall_x = self.middle_x2
        self.wall_x = self.net_x


        self.right_paddle_x_min = self.net_x
        self.right_paddle_x_max = 156
        self.left_paddle_x_min = 20
        self.left_paddle_x_max = self.net_x

        self.render_fps = 30


        self.front_camp_grace_steps = int(2.0 * self.render_fps)
        self.front_camp_penalty_per_step = 0.01 / float(self.render_fps)

        self.net_stick_band_px = 2.0
        self.net_stick_grace_steps = int(2.0 * self.render_fps)
        self.net_stick_penalty_per_step = 0.0


        self.passive_touch_reward = 0.05
        self.slow_hit_speed_threshold = 0.45
        self.fast_hit_speed_threshold = 1.20


        self.max_total_paddle_step = 2.00
        self.slow_paddle_hit_reward = 1.0
        self.fast_paddle_hit_reward = 3.0
        self.last_paddle_speed_at_hit = 0.0
        self.last_paddle_speed_reward = 0.0
        self.last_paddle_speed_tier = "none"
        self.min_outgoing_wall_vx_for_hit_reward = 0.12
        self.min_outgoing_speed_for_hit_reward = 0.25


        self.approach_to_ball_reward_per_px = 0.040
        self.approach_to_ball_reward_per_step_cap = 0.090
        self.approach_to_ball_max_reward = 2.80
        self.x_chase_reward_per_px = 0.040
        self.x_chase_reward_per_step_cap = 0.080
        self.y_align_reward_per_step = 0.006
        self.intercept_zone_reward = 0.018
        self.no_chase_penalty_per_step = 0.015


        self.behind_ball_chase_reward_per_px = 0.070
        self.behind_ball_chase_reward_per_step_cap = 0.140
        self.behind_ball_correct_action_reward = 0.020
        self.behind_ball_ignore_penalty = 0.035
        self.approach_to_ball_reward_total = 0.0
        self.approach_to_ball_reward_count = 0


        self.self_start_ball_stopped = True


        self.self_start_behind_probability = 0.35
        self.self_start_official_probability = 0.35
        self.self_start_front_probability = 0.30
        self.self_start_region = "unknown"


        self.recovery_reward_value = 1.0
        self.recovery_home_tolerance_x = 9.0
        self.recovery_home_tolerance_y = 10.0
        self.recovery_min_delay_steps = 3
        self.recovery_window_steps = int(2.0 * self.render_fps)


        self.opponent_avoid_band_px = 30.0
        self.opponent_avoid_reward_value = 1.0


        self.wall_start_probability = 0.50
        self.start_mode = "wall"


        self.self_start_min_speed = 0.15
        self.self_start_max_speed = 2.20


        self.paddle_len = 12.0
        self.paddle_width = 6.5
        self.ball_radius = 3.2


        self.paddle_angle_min = math.radians(45.0)
        self.paddle_angle_max = math.radians(135.0)
        self.long_face_only_collision = True


        self.long_face_end_tolerance = 1.10
        self.long_face_center_reward = 0.35
        self.short_edge_contact_penalty = 0.35
        self.last_contact_local_t = 0.0
        self.last_contact_local_n = 0.0
        self.last_contact_face = "none"


        self.smash_level = 6
        self.smash_speed_threshold = 2.15
        self.smash_speed_target = 3.20
        self.smash_contact_bonus_max = 1.00
        self.smash_wall_success_bonus_max = 1.25
        self.smash_weak_success_penalty = 0.25
        self.last_outgoing_speed_at_hit = 0.0
        self.last_smash_contact_reward = 0.0
        self.last_smash_wall_reward = 0.0

        self.renderer = CourtRenderer(self.w, self.h)
        self.extractor = OpenCVStateExtractor(self.w, self.h, side=self.active_side)

        self.action_space = spaces.MultiDiscrete([3, 3, 3])

        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(16,), dtype=np.float32)

        self._last_img = None
        self.reset()

    def set_level(self, level: int) -> None:
        self.level = int(max(1, min(6, level)))

    def _choose_side(self) -> str:
        if self.side_mode in ("left", "right"):
            return self.side_mode
        return "left" if self.rng.random() < 0.5 else "right"

    def _side_sign(self) -> float:
        return 1.0 if self.active_side == "right" else -1.0

    def _is_right(self) -> bool:
        return self.active_side == "right"

    def _wall_x(self) -> float:
        return float(self.right_agent_wall_x if self._is_right() else self.left_agent_wall_x)

    def _home_x(self) -> float:
        return 133.0 if self._is_right() else 43.0

    def _home_y(self) -> float:
        return 50.0

    def _opponent_side(self) -> str:
        return "left" if self._is_right() else "right"

    def _sample_opponent_racket(self) -> None:
        if self._is_right():

            self.opponent_x = self.rng.uniform(35.0, 50.0)
        else:

            self.opponent_x = self.rng.uniform(126.0, 141.0)
        self.opponent_y = self.rng.uniform(self.court_y1 + 6.0, self.court_y2 - 6.0)
        self.opponent_angle = self.rng.uniform(0.0, math.pi)

    def _ball_moving_away_after_hit(self) -> bool:
        return (self.ball_vx < 0.0) if self._is_right() else (self.ball_vx > 0.0)

    def _near_recovery_home(self) -> bool:
        return (
            abs(self.paddle_x - self._home_x()) <= self.recovery_home_tolerance_x
            and abs(self.paddle_y - self._home_y()) <= self.recovery_home_tolerance_y
        )

    def _self_start_probability_for_level(self) -> float:
        level = int(max(1, min(6, self.level)))
        table = {
            1: 1.00,
            2: 1.00,
            3: 0.95,
            4: 0.70,
            5: 0.50,
            6: 0.35,
        }
        return float(table[level])

    def _sample_self_start_ball_xy(self) -> Tuple[float, float]:
        level = int(max(1, min(6, self.level)))

        p = self.rng.random()
        behind_prob = float(getattr(self, "self_start_behind_probability", 0.03))
        official_prob = float(getattr(self, "self_start_official_probability", 0.57))

        if p < behind_prob:
            region = "behind"
        elif p < behind_prob + official_prob:
            region = "official_like"
        else:
            region = "front"

        self.self_start_region = region


        if level == 1:
            y1, y2 = 38.0, 62.0
        elif level == 2:
            y1, y2 = 32.0, 68.0
        else:
            y1, y2 = 28.5, 70.5

        if self._is_right():
            if region == "behind":

                x1, x2 = 136.0, 156.0
            elif region == "official_like":

                if level == 1:
                    x1, x2 = 126.0, 150.0
                    y1, y2 = 38.0, 64.0
                else:
                    x1, x2 = 124.0, 152.0
            else:

                if level == 1:
                    x1, x2 = 118.0, 134.0
                elif level == 2:
                    x1, x2 = 110.0, 134.0
                else:
                    x1, x2 = 96.0, 134.0
        else:
            if region == "behind":

                x1, x2 = 20.0, 40.0
            elif region == "official_like":

                if level == 1:
                    x1, x2 = 28.0, 52.0
                    y1, y2 = 38.0, 64.0
                else:
                    x1, x2 = 24.0, 56.0
            else:

                if level == 1:
                    x1, x2 = 42.0, 58.0
                elif level == 2:
                    x1, x2 = 42.0, 72.0
                else:
                    x1, x2 = 42.0, 72.0

        return float(self.rng.uniform(x1, x2)), float(self.rng.uniform(y1, y2))

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        if seed is not None:
            self.rng.seed(seed)
            self.np_rng = np.random.default_rng(seed)

        self.active_side = self._choose_side()
        self.extractor.set_side(self.active_side)
        self.extractor.reset()

        self.steps = 0
        self.success_count = 0
        self.time_on_agent_side = 0
        self.last_contact_step = -9999
        self.last_wall_step = -9999
        self.contact_since_wall = False
        self.paddle_contacts = 0
        self.strong_bounce_count = 0
        self.last_hit_angle = 0.0
        self.last_hit_strong = False
        self.last_contact_local_t = 0.0
        self.last_contact_local_n = 0.0
        self.last_contact_face = "none"
        self.front_camp_steps = 0
        self.total_front_camp_steps = 0
        self.action_to_net_steps = 0
        self.net_stick_steps = 0
        self.total_net_stick_steps = 0
        self.last_paddle_speed_at_hit = 0.0
        self.last_paddle_speed_reward = 0.0
        self.last_paddle_speed_tier = "none"
        self.last_outgoing_speed_at_hit = 0.0
        self.last_smash_contact_reward = 0.0
        self.last_smash_wall_reward = 0.0
        self.slow_paddle_hit_count = 0
        self.fast_paddle_hit_count = 0
        self.approach_to_ball_reward_total = 0.0
        self.approach_to_ball_reward_count = 0
        self.total_paddle_x = 0.0
        self.total_home_distance = 0.0
        self.recovery_reward_pending = False
        self.recovery_reward_start_step = 0
        self.recovery_reward_deadline_step = 0
        self.recovery_reward_count = 0
        self.opponent_avoid_count = 0
        self.opponent_avoid_attempts = 0
        self.opponent_x = 43.0
        self.opponent_y = 50.0
        self.opponent_angle = math.pi / 2.0
        self.self_start_region = "unknown"
        self._sample_opponent_racket()


        self.paddle_speed = self.rng.uniform(1.55, 2.25) if self.domain_randomization else 1.90


        self.paddle_rot_speed = math.radians(self.rng.uniform(2.0, 4.0)) if self.domain_randomization else math.radians(3.0)
        self.ball_drag = self.rng.uniform(0.998, 1.002) if self.domain_randomization else 1.0


        if self._is_right():
            self.paddle_x = self.rng.uniform(132, 152)
        else:
            self.paddle_x = self.rng.uniform(24, 44)
        self.paddle_y = self.rng.uniform(38, 64)


        self.paddle_angle = (math.pi / 2.0) + self.rng.uniform(-0.30, 0.30)
        self.paddle_angle = float(np.clip(self.paddle_angle, self.paddle_angle_min, self.paddle_angle_max))
        self.prev_paddle_x = float(self.paddle_x)
        self.prev_paddle_y = float(self.paddle_y)
        self.paddle_vx = 0.0
        self.paddle_vy = 0.0


        self_start_prob = self._self_start_probability_for_level()
        if self.rng.random() >= self_start_prob:
            self.start_mode = "wall"
            self.self_start_region = "wall"
            wx = self._wall_x()
            if self._is_right():
                self.ball_x = wx + self.ball_radius + 1.0
            else:
                self.ball_x = wx - self.ball_radius - 1.0
            self.ball_y = self.rng.uniform(32, 68)
            self.ball_vx, self.ball_vy = self._sample_wall_return_velocity()
        else:
            self.start_mode = "self"
            self.ball_x, self.ball_y = self._sample_self_start_ball_xy()

            self.ball_vx = 0.0
            self.ball_vy = 0.0

        obs = self._get_obs()
        info = self._info()
        return obs, info

    def step(self, action):
        self.steps += 1
        action = np.asarray(action).astype(int).tolist()
        reward = 0.0
        terminated = False
        truncated = False
        info: Dict[str, Any] = {}


        self.prev_paddle_x = float(getattr(self, 'paddle_x', 0.0))
        self.prev_paddle_y = float(getattr(self, 'paddle_y', 0.0))
        prev_ball_dist = math.hypot(float(self.ball_x - self.paddle_x), float(self.ball_y - self.paddle_y))

        self._apply_action(action)

        self.paddle_vx = float(self.paddle_x - self.prev_paddle_x)
        self.paddle_vy = float(self.paddle_y - self.prev_paddle_y)


        ball_on_agent_side_for_approach = (self.ball_x > self._wall_x()) if self._is_right() else (self.ball_x < self._wall_x())
        if ball_on_agent_side_for_approach and self.approach_to_ball_reward_total < self.approach_to_ball_max_reward:
            new_ball_dist = math.hypot(float(self.ball_x - self.paddle_x), float(self.ball_y - self.paddle_y))
            dist_reduced = prev_ball_dist - new_ball_dist
            prev_x_dist = abs(float(self.ball_x - self.prev_paddle_x))
            new_x_dist = abs(float(self.ball_x - self.paddle_x))
            x_reduced = prev_x_dist - new_x_dist

            add_total = 0.0
            if dist_reduced > 0.05:
                add_total += min(
                    self.approach_to_ball_reward_per_step_cap,
                    self.approach_to_ball_reward_per_px * float(dist_reduced),
                )
            if x_reduced > 0.05:
                add_total += min(
                    self.x_chase_reward_per_step_cap,
                    self.x_chase_reward_per_px * float(x_reduced),
                )

            y_err_px = abs(float(self.ball_y - self.paddle_y))
            if y_err_px <= 10.0:
                add_total += self.y_align_reward_per_step * (1.0 - min(1.0, y_err_px / 10.0))

            if new_x_dist <= 8.0 and y_err_px <= 10.0:
                add_total += self.intercept_zone_reward
                info["intercept_zone_reward"] = True


            ball_behind_paddle = (
                (self._is_right() and self.ball_x > self.paddle_x + 4.0)
                or ((not self._is_right()) and self.ball_x < self.paddle_x - 4.0)
            )
            if ball_behind_paddle:
                correct_back_action = (int(action[1]) == 1) if self._is_right() else (int(action[1]) == 2)
                if x_reduced > 0.05:
                    add_total += min(
                        self.behind_ball_chase_reward_per_step_cap,
                        self.behind_ball_chase_reward_per_px * float(x_reduced),
                    )
                    info["behind_ball_chase_reward"] = True
                if correct_back_action:
                    add_total += self.behind_ball_correct_action_reward
                    info["behind_ball_correct_action"] = True
                if (not correct_back_action) and x_reduced <= 0.01 and dist_reduced <= 0.01:
                    reward -= self.behind_ball_ignore_penalty
                    info["behind_ball_ignore_penalty"] = True


            if prev_x_dist > 14.0 and x_reduced <= 0.01 and dist_reduced <= 0.01:
                reward -= self.no_chase_penalty_per_step
                info["no_chase_penalty"] = True

            remaining = self.approach_to_ball_max_reward - self.approach_to_ball_reward_total
            add_total = min(float(add_total), max(0.0, remaining))
            if add_total > 0.0:
                reward += add_total
                self.approach_to_ball_reward_total += add_total
                self.approach_to_ball_reward_count += 1
                info["approach_to_ball_reward"] = float(add_total)


        half_paddle = self.paddle_len * 0.5
        in_front_area = (self.middle_x1 - half_paddle) <= self.paddle_x <= (self.middle_x2 + half_paddle)
        if in_front_area:
            self.front_camp_steps += 1
            self.total_front_camp_steps += 1
            moving_to_net = (int(action[1]) == 2) if self._is_right() else (int(action[1]) == 1)
            if moving_to_net:
                self.action_to_net_steps += 1
            if self.front_camp_steps > self.front_camp_grace_steps:
                reward -= self.front_camp_penalty_per_step
                info["light_blue_soft_penalty"] = True
        else:
            self.front_camp_steps = 0

        near_net = abs(float(self.paddle_x) - float(self.net_x)) <= self.net_stick_band_px
        if near_net:
            self.net_stick_steps += 1
            self.total_net_stick_steps += 1
            if self.net_stick_steps > self.net_stick_grace_steps:
                reward -= self.net_stick_penalty_per_step
                info["net_stick_penalty"] = True
        else:
            self.net_stick_steps = 0

        self.total_paddle_x += float(self.paddle_x)
        self.total_home_distance += abs(self.paddle_x - self._home_x())


        if self.recovery_reward_pending:
            if self.steps > self.recovery_reward_deadline_step:
                self.recovery_reward_pending = False
            elif self.steps >= self.recovery_reward_start_step and self._ball_moving_away_after_hit() and self._near_recovery_home():
                reward += self.recovery_reward_value
                self.recovery_reward_count += 1
                self.recovery_reward_pending = False
                info["recovery_home_reward"] = True


        wx = self._wall_x()
        incoming = (self.ball_vx > 0 and self.ball_x > wx) if self._is_right() else (self.ball_vx < 0 and self.ball_x < wx)
        if incoming:
            y_err = abs(self.paddle_y - self.ball_y) / max(1.0, (self.court_y2 - self.court_y1))
            reward += 0.003 * (1.0 - min(1.0, y_err * 2.0))
            if self._is_right():
                if 128.0 <= self.paddle_x <= 156.0:
                    reward += 0.001
            else:
                if 20.0 <= self.paddle_x <= 50.0:
                    reward += 0.001


        self.ball_x += self.ball_vx
        self.ball_y += self.ball_vy
        self.ball_vx *= self.ball_drag
        self.ball_vy *= self.ball_drag


        in_middle_wall_channel = self.middle_x1 <= self.ball_x <= self.middle_x2
        if in_middle_wall_channel and self.ball_y <= self.court_y1 + self.ball_radius:
            self.ball_y = self.court_y1 + self.ball_radius
            self.ball_vy = abs(self.ball_vy)
        if in_middle_wall_channel and self.ball_y >= self.court_y2 - self.ball_radius:
            self.ball_y = self.court_y2 - self.ball_radius
            self.ball_vy = -abs(self.ball_vy)


        self.last_hit_strong = False
        self.last_paddle_speed_at_hit = 0.0
        self.last_paddle_speed_reward = 0.0
        self.last_paddle_speed_tier = "none"
        hit = self._check_paddle_hit()
        if hit:
            self.contact_since_wall = True
            self.paddle_contacts += 1

            self.recovery_reward_pending = True
            self.recovery_reward_start_step = self.steps + self.recovery_min_delay_steps
            self.recovery_reward_deadline_step = self.steps + self.recovery_window_steps
            info["paddle_contact"] = True
            info["hit_angle_rad"] = self.last_hit_angle
            info["contact_face"] = self.last_contact_face
            info["contact_local_t"] = float(self.last_contact_local_t)


            if self.last_contact_face == "long_face_center":
                reward += self.long_face_center_reward
                info["long_face_center_reward"] = float(self.long_face_center_reward)
            elif self.last_contact_face == "long_face_edge":
                reward -= self.short_edge_contact_penalty
                info["long_face_edge_penalty"] = float(self.short_edge_contact_penalty)


            reward += self.passive_touch_reward
            if self.last_paddle_speed_reward > 0.0:
                reward += self.last_paddle_speed_reward

            if int(self.level) >= int(self.smash_level) and self.last_smash_contact_reward > 0.0:
                reward += self.last_smash_contact_reward
                info["smash_contact_reward"] = float(self.last_smash_contact_reward)
            info["touch_reward"] = True
            info["passive_touch_reward"] = float(self.passive_touch_reward)
            info["paddle_speed_at_hit"] = float(self.last_paddle_speed_at_hit)
            info["paddle_speed_tier"] = self.last_paddle_speed_tier
            info["paddle_speed_reward"] = float(self.last_paddle_speed_reward)
            info["outgoing_speed_at_hit"] = float(self.last_outgoing_speed_at_hit)
            if self.last_paddle_speed_tier == "fast":
                self.fast_paddle_hit_count += 1
            elif self.last_paddle_speed_tier == "slow":
                self.slow_paddle_hit_count += 1
            if self.last_hit_strong:
                self.strong_bounce_count += 1
                info["strong_bounce"] = True


        wx = self._wall_x()
        wall_hit = (self.ball_x <= wx and self.ball_vx < 0) if self._is_right() else (self.ball_x >= wx and self.ball_vx > 0)
        if wall_hit and (self.steps - self.last_contact_step) >= 4:
            if self._is_right():
                self.ball_x = wx + self.ball_radius + 0.5
            else:
                self.ball_x = wx - self.ball_radius - 0.5

            if self.contact_since_wall:
                self.success_count += 1
                reward += 2.0
                info["wall_rebound_success"] = True
                info["success_count"] = self.success_count


                if int(self.level) >= int(self.smash_level):
                    wall_speed = float(math.hypot(self.ball_vx, self.ball_vy))
                    smash_strength = float(np.clip(
                        (wall_speed - self.smash_speed_threshold) / max(1e-6, self.smash_speed_target - self.smash_speed_threshold),
                        0.0,
                        1.0,
                    ))
                    self.last_smash_wall_reward = self.smash_wall_success_bonus_max * smash_strength
                    if self.last_smash_wall_reward > 0.0:
                        reward += self.last_smash_wall_reward
                    elif wall_speed < 1.35:
                        reward -= self.smash_weak_success_penalty
                    info["smash_wall_speed"] = wall_speed
                    info["smash_wall_reward"] = float(self.last_smash_wall_reward)


                if int(self.level) >= 1:
                    self.opponent_avoid_attempts += 1
                    avoid_dist = abs(float(self.ball_y) - float(self.opponent_y))
                    info["opponent_y"] = float(self.opponent_y)
                    info["opponent_avoid_dist"] = float(avoid_dist)
                    if avoid_dist > self.opponent_avoid_band_px:
                        reward += self.opponent_avoid_reward_value
                        self.opponent_avoid_count += 1
                        info["opponent_avoid_reward"] = True
            else:
                info["unearned_wall_rebound"] = True
            self.contact_since_wall = False
            self.time_on_agent_side = 0
            self.last_wall_step = self.steps
            self._sample_opponent_racket()
            self.ball_vx, self.ball_vy = self._sample_wall_return_velocity()


        wx = self._wall_x()
        on_agent_side = (self.ball_x > wx) if self._is_right() else (self.ball_x < wx)
        if on_agent_side:
            self.time_on_agent_side += 1
        else:
            self.time_on_agent_side = 0


        fail_reason = None
        if self.ball_x > self.court_x2 + 8:
            fail_reason = "right_out"
        elif self.ball_x < self.court_x1 - 8:
            fail_reason = "left_out"
        elif self.ball_y < self.court_y1 - 12 or self.ball_y > self.court_y2 + 12:
            fail_reason = "vertical_out"
        elif (self.ball_x > self.middle_x2 or self.ball_x < self.middle_x1) and (
            self.ball_y < self.court_y1 + self.ball_radius or self.ball_y > self.court_y2 - self.ball_radius
        ):
            fail_reason = "side_out_open_court"
        elif self.time_on_agent_side > self.own_side_timeout_steps:
            fail_reason = "own_side_timeout"

        if fail_reason is not None:
            reward -= 3.0
            terminated = True
            info["fail_reason"] = fail_reason

        if self.steps >= self.max_episode_steps:
            truncated = True

        obs = self._get_obs()
        info.update(self._info())
        return obs, float(reward), terminated, truncated, info

    def _sample_wall_return_velocity(self) -> Tuple[float, float]:
        cfg = get_level(self.level)


        if int(self.level) >= 6:
            p = self.rng.random()
            if p < 0.10:
                speed = self.rng.uniform(0.80, 1.35)
            elif p < 0.35:
                speed = self.rng.uniform(1.85, 2.35)
            elif p < 0.85:
                speed = self.rng.uniform(2.35, 3.05)
            else:
                speed = self.rng.uniform(3.05, 3.70)
        elif int(self.level) >= 5:
            p = self.rng.random()
            if p < 0.03:
                speed = self.rng.uniform(0.35, 0.70)
            elif p < 0.40:
                speed = self.rng.uniform(0.80, 1.35)
            elif p < 0.60:
                speed = self.rng.uniform(1.85, 2.35)
            elif p < 0.97:
                speed = self.rng.uniform(2.35, 3.05)
            else:
                speed = self.rng.uniform(3.05, 3.70)
        else:
            speed = self.rng.uniform(cfg.wall_speed_min, cfg.wall_speed_max)

        if self._is_right():
            target_ranges = {
                1: (126.0, 156.0, 30.0, 70.0),
                2: (126.0, 156.0, 30.0, 70.0),
                3: (118.0, 157.0, 28.5, 70.5),
                4: (118.0, 157.0, 28.5, 70.5),
                5: (118.0, 157.0, 28.5, 70.5),
                6: (108.0, 157.0, 28.5, 70.5),
            }
        else:
            target_ranges = {
                1: (20.0, 56.0, 30.0, 70.0),
                2: (20.0, 56.0, 30.0, 70.0),
                3: (19.0, 62.0, 28.5, 70.5),
                4: (19.0, 62.0, 28.5, 70.5),
                5: (19.0, 62.0, 28.5, 70.5),
                6: (19.0, 72.0, 28.5, 70.5),
            }

        tx1, tx2, ty1, ty2 = target_ranges[int(max(1, min(6, self.level)))]
        target_x = self.rng.uniform(tx1, tx2)
        target_y = self.rng.uniform(ty1, ty2)

        dx = target_x - self.ball_x
        dy = target_y - self.ball_y
        norm = max(1e-6, math.hypot(dx, dy))
        vx = dx / norm * speed
        vy = dy / norm * speed

        jitter_max = math.radians(min(20.0, get_level(self.level).wall_angle_deg * 0.25))
        jitter = self.rng.uniform(-jitter_max, jitter_max)
        c, s = math.cos(jitter), math.sin(jitter)
        vx, vy = (vx * c - vy * s), (vx * s + vy * c)


        if self._is_right():
            vx = max(abs(vx), 0.20 * speed)
        else:
            vx = -max(abs(vx), 0.20 * speed)
        return float(vx), float(vy)

    def _apply_action(self, action):


        vertical, horizontal, rotation = action


        dx = 0.0
        dy = 0.0
        if vertical == 1:
            dy -= self.paddle_speed
        elif vertical == 2:
            dy += self.paddle_speed

        if horizontal == 1:
            dx += self.paddle_speed
        elif horizontal == 2:
            dx -= self.paddle_speed

        move_norm = float(math.hypot(dx, dy))
        max_step = float(getattr(self, "max_total_paddle_step", 2.0))
        if move_norm > max_step > 0.0:
            scale = max_step / move_norm
            dx *= scale
            dy *= scale

        self.paddle_x += dx
        self.paddle_y += dy

        if rotation == 1:
            self.paddle_angle -= self.paddle_rot_speed
        elif rotation == 2:
            self.paddle_angle += self.paddle_rot_speed

        if self._is_right():
            self.paddle_x = float(np.clip(self.paddle_x, self.right_paddle_x_min, self.right_paddle_x_max))
        else:
            self.paddle_x = float(np.clip(self.paddle_x, self.left_paddle_x_min, self.left_paddle_x_max))
        self.paddle_y = float(np.clip(self.paddle_y, self.court_y1 + 5, self.court_y2 - 5))

        self.paddle_angle = float(np.clip(self.paddle_angle, self.paddle_angle_min, self.paddle_angle_max))

    def _check_paddle_hit(self) -> bool:


        speed_now = math.hypot(float(self.ball_vx), float(self.ball_vy))
        allow_self_start_push = (self.start_mode == "self" and speed_now < 0.35)
        if not allow_self_start_push:
            if self._is_right():
                if self.ball_vx <= 0:
                    return False
            else:
                if self.ball_vx >= 0:
                    return False


        if (self.steps - self.last_contact_step) < 6:
            return False


        theta = math.pi - float(self.paddle_angle)


        tangent = np.array([math.cos(theta), math.sin(theta)], dtype=np.float64)
        tangent = tangent / max(1e-9, float(np.linalg.norm(tangent)))
        normal = np.array([-math.sin(theta), math.cos(theta)], dtype=np.float64)
        normal = normal / max(1e-9, float(np.linalg.norm(normal)))

        rel = np.array([self.ball_x - self.paddle_x, self.ball_y - self.paddle_y], dtype=np.float64)
        local_t = float(np.dot(rel, tangent))
        local_n = float(np.dot(rel, normal))

        half_len = self.paddle_len / 2.0
        half_width = self.paddle_width / 2.0
        r = float(self.ball_radius)


        if abs(local_t) > half_len + r:
            return False
        if abs(local_n) > half_width + r:
            return False


        self.last_contact_local_t = float(local_t)
        self.last_contact_local_n = float(local_n)
        if self.long_face_only_collision and abs(local_t) > half_len + self.long_face_end_tolerance:
            self.last_contact_face = "short_edge_rejected"
            return False
        core_face = abs(local_t) <= (0.75 * half_len)
        self.last_contact_face = "long_face_center" if core_face else "long_face_edge"

        v_in = np.array([self.ball_vx, self.ball_vy], dtype=np.float64)
        speed_in = float(np.linalg.norm(v_in))
        paddle_v = np.array([
            float(getattr(self, 'paddle_vx', 0.0)),
            float(getattr(self, 'paddle_vy', 0.0)),
        ], dtype=np.float64)
        paddle_speed = float(np.linalg.norm(paddle_v))
        self.last_paddle_speed_at_hit = paddle_speed
        potential_speed_tier = "passive"
        potential_speed_reward = 0.0
        if paddle_speed >= self.fast_hit_speed_threshold:
            potential_speed_tier = "fast"
            potential_speed_reward = float(self.fast_paddle_hit_reward)
        elif paddle_speed >= self.slow_hit_speed_threshold:
            potential_speed_tier = "slow"
            potential_speed_reward = float(self.slow_paddle_hit_reward)

        self.last_paddle_speed_tier = potential_speed_tier
        self.last_paddle_speed_reward = 0.0
        self.last_smash_contact_reward = 0.0
        self.last_outgoing_speed_at_hit = 0.0


        if allow_self_start_push:

            if paddle_speed < 0.05 and speed_in < 0.08:
                return False


            n1 = normal
            n2 = -normal
            desired_sign = -1.0 if self._is_right() else 1.0
            push_dir = n1 if (n1[0] * desired_sign) > (n2[0] * desired_sign) else n2


            push_gain = 1.35 if int(self.level) >= int(self.smash_level) else 1.05
            push_max = 2.80 if int(self.level) >= int(self.smash_level) else self.self_start_max_speed
            speed = 0.55 * speed_in + push_gain * paddle_speed
            speed = float(np.clip(speed, self.self_start_min_speed, push_max))
            v_out = push_dir / max(1e-9, float(np.linalg.norm(push_dir))) * speed
            speed_out = float(np.linalg.norm(v_out))
            dot_in_out = 0.0
            self.last_hit_strong = bool(paddle_speed > 0.60 and speed_out > 0.80)
        else:
            if speed_in < 1e-6:
                return False


            v_out = v_in - 2.0 * float(np.dot(v_in, normal)) * normal


            smash_gain = 0.45 if int(self.level) >= int(self.smash_level) else 0.20
            max_out_speed = 4.20 if int(self.level) >= int(self.smash_level) else 3.45
            speed = speed_in * 0.995 + smash_gain * paddle_speed
            speed = float(np.clip(speed, 0.05, max_out_speed))
            v_out_norm = max(1e-9, float(np.linalg.norm(v_out)))
            v_out = v_out / v_out_norm * speed

            speed_out = float(np.linalg.norm(v_out))
            dot_in_out = float(np.dot(v_in, v_out))
            self.last_hit_strong = bool(dot_in_out < 0.0 and speed_out >= 0.95 * max(speed_in, 1e-6))


        sep_norm = normal if float(np.dot(v_out, normal)) >= 0.0 else -normal
        clamped_t = float(np.clip(local_t, -half_len, half_len))
        contact_point = (
            np.array([self.paddle_x, self.paddle_y], dtype=np.float64)
            + tangent * clamped_t
            + sep_norm * (half_width + r + 0.90)
        )


        outgoing_speed = float(np.linalg.norm(v_out))
        self.last_outgoing_speed_at_hit = outgoing_speed
        correct_outgoing_to_wall = (
            (self._is_right() and float(v_out[0]) <= -self.min_outgoing_wall_vx_for_hit_reward)
            or ((not self._is_right()) and float(v_out[0]) >= self.min_outgoing_wall_vx_for_hit_reward)
        )
        if correct_outgoing_to_wall and outgoing_speed >= self.min_outgoing_speed_for_hit_reward:
            self.last_paddle_speed_reward = potential_speed_reward
            self.last_paddle_speed_tier = potential_speed_tier

            if int(self.level) >= int(self.smash_level):
                smash_strength = float(np.clip(
                    (outgoing_speed - self.smash_speed_threshold) / max(1e-6, self.smash_speed_target - self.smash_speed_threshold),
                    0.0,
                    1.0,
                ))
                self.last_smash_contact_reward = self.smash_contact_bonus_max * smash_strength
        else:
            self.last_paddle_speed_reward = 0.0
            if potential_speed_tier in ("slow", "fast"):
                self.last_paddle_speed_tier = "wrong_direction"
            else:
                self.last_paddle_speed_tier = "passive"

        self.ball_vx = float(v_out[0])
        self.ball_vy = float(v_out[1])
        self.ball_x = float(contact_point[0])
        self.ball_y = float(contact_point[1])

        self.last_contact_step = self.steps
        self.last_hit_angle = float(math.atan2(self.ball_vy, self.ball_vx))
        return True

    def _get_obs(self) -> np.ndarray:
        time_norm = min(1.0, self.time_on_agent_side / max(1, self.own_side_timeout_steps))
        succ_norm = min(1.0, self.success_count / max(1, self.trial_return_target))

        self.extractor.set_side(self.active_side)

        if self.obs_source == "opencv":
            img = self.render()
            return self.extractor.features_from_image(img, time_norm, succ_norm)

        noise = 0.001 if self.domain_randomization else 0.0
        feat = self.extractor.features_from_state(
            ball_xy=(self.ball_x, self.ball_y),
            ball_vxy=(self.ball_vx, self.ball_vy),
            paddle_xy=(self.paddle_x, self.paddle_y),
            paddle_angle_rad=self.paddle_angle,
            time_own_side_norm=time_norm,
            success_count_norm=succ_norm,
            opponent_y=self.opponent_y,
        )
        if noise > 0:
            feat = feat + self.np_rng.normal(0.0, noise, size=feat.shape).astype(np.float32)
            feat = np.clip(feat, -1.0, 1.0)
        return feat.astype(np.float32)

    def _info(self) -> Dict[str, Any]:
        steps = max(1, self.steps)
        return {
            "level": self.level,
            "side": self.active_side,
            "side_sign": self._side_sign(),
            "success_count": self.success_count,
            "trial_success": self.success_count >= self.trial_return_target,
            "time_on_agent_side": self.time_on_agent_side,
            "ball_x": self.ball_x,
            "ball_y": self.ball_y,
            "ball_vx": self.ball_vx,
            "ball_vy": self.ball_vy,
            "paddle_x": self.paddle_x,
            "paddle_y": self.paddle_y,
            "paddle_contacts": self.paddle_contacts,
            "paddle_vx": float(getattr(self, "paddle_vx", 0.0)),
            "paddle_vy": float(getattr(self, "paddle_vy", 0.0)),
            "paddle_angle_deg": math.degrees(float(getattr(self, "paddle_angle", math.pi / 2.0))),
            "last_contact_face": self.last_contact_face,
            "last_contact_local_t": float(getattr(self, "last_contact_local_t", 0.0)),
            "strong_bounce_count": self.strong_bounce_count,
            "last_paddle_speed_at_hit": self.last_paddle_speed_at_hit,
            "last_paddle_speed_reward": self.last_paddle_speed_reward,
            "last_paddle_speed_tier": self.last_paddle_speed_tier,
            "last_outgoing_speed_at_hit": float(getattr(self, "last_outgoing_speed_at_hit", 0.0)),
            "last_smash_contact_reward": float(getattr(self, "last_smash_contact_reward", 0.0)),
            "last_smash_wall_reward": float(getattr(self, "last_smash_wall_reward", 0.0)),
            "slow_paddle_hit_count": self.slow_paddle_hit_count,
            "fast_paddle_hit_count": self.fast_paddle_hit_count,
            "approach_to_ball_reward_total": self.approach_to_ball_reward_total,
            "approach_to_ball_reward_count": self.approach_to_ball_reward_count,
            "front_camp_rate": self.total_front_camp_steps / steps,
            "action_to_net_rate": self.action_to_net_steps / steps,
            "net_stick_rate": self.total_net_stick_steps / steps,
            "avg_paddle_x": self.total_paddle_x / steps,
            "avg_home_distance": self.total_home_distance / steps,
            "wall_x": self._wall_x(),
            "start_mode": self.start_mode,
            "self_start_region": self.self_start_region,
            "home_x": self._home_x(),
            "home_y": self._home_y(),
            "recovery_reward_count": self.recovery_reward_count,
            "opponent_x": self.opponent_x,
            "opponent_y": self.opponent_y,
            "opponent_avoid_count": self.opponent_avoid_count,
            "opponent_avoid_attempts": self.opponent_avoid_attempts,
            "opponent_avoid_rate": self.opponent_avoid_count / max(1, self.opponent_avoid_attempts),
        }

    def render(self):
        self._last_img = self.renderer.render(
            ball_xy=(self.ball_x, self.ball_y),
            paddle_xy=(self.paddle_x, self.paddle_y),
            paddle_angle_rad=self.paddle_angle,
            side=self.active_side,
            wall_x=self._wall_x(),
            home_xy=(self._home_x(), self._home_y()),
            opponent_xy=(self.opponent_x, self.opponent_y),
            opponent_angle_rad=self.opponent_angle,
            opponent_side=self._opponent_side(),
        )
        return self._last_img.copy()

    def close(self):
        pass


def make_python_wall_env(level: int, seed: int, obs_source: str = "state", render_mode: Optional[str] = None, side: str = "random"):
    def _init():
        return PythonWallPickleballEnv(
            level=level,
            seed=seed,
            obs_source=obs_source,
            render_mode=render_mode,
            side=side,
        )
    return _init
