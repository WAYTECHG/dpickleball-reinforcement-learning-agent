# Six-level wall curriculum settings.
from __future__ import annotations

from dataclasses import dataclass
import math
import random


@dataclass(frozen=True)
class WallLevel:
    level: int
    name: str
    wall_speed_min: float
    wall_speed_max: float
    wall_angle_deg: float
    agent_incoming_speed_min: float
    agent_incoming_speed_max: float
    description: str


WALL_LEVELS = {
    1: WallLevel(
        level=1,
        name="slow_90deg",
        wall_speed_min=0.85,
        wall_speed_max=1.35,
        wall_angle_deg=45,
        agent_incoming_speed_min=0.85,
        agent_incoming_speed_max=1.35,
        description="Slow returns with 90 degree total angle range. Teaches slow-ball reading and active movement."
    ),
    2: WallLevel(
        level=2,
        name="fast_90deg",
        wall_speed_min=2.35,
        wall_speed_max=3.05,
        wall_angle_deg=45,
        agent_incoming_speed_min=2.35,
        agent_incoming_speed_max=3.05,
        description="Fast returns with 90 degree total angle range."
    ),
    3: WallLevel(
        level=3,
        name="very_slow_150deg",
        wall_speed_min=0.35,
        wall_speed_max=0.80,
        wall_angle_deg=75,
        agent_incoming_speed_min=0.35,
        agent_incoming_speed_max=0.80,
        description="Very slow returns with 150 degree total angle range. Forces active chase/push rather than only blocking."
    ),
    4: WallLevel(
        level=4,
        name="medium_fast_150deg",
        wall_speed_min=1.85,
        wall_speed_max=2.35,
        wall_angle_deg=75,
        agent_incoming_speed_min=1.85,
        agent_incoming_speed_max=2.35,
        description="Medium-fast returns with 150 degree total angle range."
    ),
    5: WallLevel(
        level=5,
        name="mixed_speed_150deg_final",
        wall_speed_min=0.35,
        wall_speed_max=3.70,
        wall_angle_deg=75,
        agent_incoming_speed_min=0.35,
        agent_incoming_speed_max=3.70,
        description="Mixed speed level: 3% super slow, 37% slow, 20% medium-fast, 37% fast, 3% super fast."
    ),
    6: WallLevel(
        level=6,
        name="level6_smash_finetune_150deg",
        wall_speed_min=0.80,
        wall_speed_max=3.70,
        wall_angle_deg=75,
        agent_incoming_speed_min=0.80,
        agent_incoming_speed_max=3.70,
        description="Smash fine-tuning level: more medium/fast incoming balls and rewards for controlled fast returns."
    ),
}


def get_level(level: int) -> WallLevel:
    level = int(max(1, min(6, level)))
    return WALL_LEVELS[level]


def sample_wall_return_velocity(level: int, toward: str = "right", rng: random.Random | None = None):
    rng = rng or random
    cfg = get_level(level)

    if int(level) >= 6:
        p = rng.random()
        if p < 0.10:
            speed = rng.uniform(0.80, 1.35)
        elif p < 0.35:
            speed = rng.uniform(1.85, 2.35)
        elif p < 0.85:
            speed = rng.uniform(2.35, 3.05)
        else:
            speed = rng.uniform(3.05, 3.70)
    elif int(level) >= 5:
        p = rng.random()
        if p < 0.03:
            speed = rng.uniform(0.35, 0.70)
        elif p < 0.40:
            speed = rng.uniform(0.80, 1.35)
        elif p < 0.60:
            speed = rng.uniform(1.85, 2.35)
        elif p < 0.97:
            speed = rng.uniform(2.35, 3.05)
        else:
            speed = rng.uniform(3.05, 3.70)
    else:
        speed = rng.uniform(cfg.wall_speed_min, cfg.wall_speed_max)

    angle = math.radians(rng.uniform(-cfg.wall_angle_deg, cfg.wall_angle_deg))
    base = 0.0 if toward == "right" else math.pi
    theta = base + angle
    vx = math.cos(theta) * speed
    vy = math.sin(theta) * speed

    if toward == "right":
        vx = max(abs(vx), 0.20 * speed)
    else:
        vx = -max(abs(vx), 0.20 * speed)
    return vx, vy


def sample_initial_ball_velocity(level: int, rng: random.Random | None = None):
    rng = rng or random
    cfg = get_level(level)
    speed = rng.uniform(cfg.agent_incoming_speed_min, cfg.agent_incoming_speed_max)
    angle = math.radians(rng.uniform(-min(45, cfg.wall_angle_deg), min(45, cfg.wall_angle_deg)))
    vx = abs(math.cos(angle) * speed)
    vy = math.sin(angle) * speed
    return vx, vy
