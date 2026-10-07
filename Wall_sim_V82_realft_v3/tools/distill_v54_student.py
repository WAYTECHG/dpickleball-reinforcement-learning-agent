from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Distill the deployed behaviour of the V54 teacher into an angle-blind student.
#
# Teacher input follows the DEPLOYMENT convention, not the training one:
#   sin(angle)=1, cos(angle)=0  (rotation is masked everywhere -> paddle vertical)
#   time_own_side=0, success_count=0  (TeamX_final never fills these at deploy)
# Student input is the V73 13-dim layout: the 16-dim vector with indices
# (6, 7, 11) = sin/cos/success removed, and time_own_side forced to 0.
#
# Rollouts are teacher-driven (round 0) or student-driven (DAgger rounds >= 1),
# always stepping with rotation forced to 0. Epsilon-random actions diversify
# visited states; labels are always the teacher's action at the visited state.

DROP_IDX = (6, 7, 11)
TIME_IDX_13 = 8  # original index 10 after dropping 6,7,11


def load_env(env_dir: Path):
    if str(env_dir) not in sys.path:
        sys.path.insert(0, str(env_dir))
    from envs.python_wall_env import PythonWallPickleballEnv

    return PythonWallPickleballEnv


def teacher_input(obs16: np.ndarray) -> np.ndarray:
    f = np.asarray(obs16, dtype=np.float32).copy()
    f[6] = 1.0   # sin(pi/2): paddle is vertical in the rotation-masked world
    f[7] = 0.0   # cos(pi/2)
    f[10] = 0.0  # time_own_side: deployment never fills it
    f[11] = 0.0  # success_count: deployment never fills it
    return f


def student_input(obs16: np.ndarray) -> np.ndarray:
    f = np.delete(teacher_input(obs16), list(DROP_IDX))
    f[TIME_IDX_13] = 0.0
    return f.astype(np.float32)


class StudentNet(nn.Module):
    def __init__(self, in_dim: int = 13, hidden=(128, 128, 64)):
        super().__init__()
        layers = []
        prev = in_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.Tanh()]
            prev = h
        self.trunk = nn.Sequential(*layers)
        self.head_v = nn.Linear(prev, 3)
        self.head_h = nn.Linear(prev, 3)

    def forward(self, x: torch.Tensor):
        z = self.trunk(x)
        return self.head_v(z), self.head_h(z)


class StudentActor(nn.Module):
    """Deterministic deployment actor. Accepts 13-dim or 16-dim features;
    16-dim input is reduced with the V73 drop rule. Rotation is hard 0."""

    def __init__(self, net: StudentNet):
        super().__init__()
        self.net = net

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.numel() == 16:
            x13 = torch.cat([x[0:6], x[8:11], x[12:16]])
        else:
            x13 = x.clone()
        x13 = x13.clone()
        x13[8] = 0.0
        logits_v, logits_h = self.net(x13.unsqueeze(0))
        v = torch.argmax(logits_v[0])
        h = torch.argmax(logits_h[0])
        r = torch.zeros_like(v)
        return torch.stack([v, h, r])


def collect(
    *,
    env_cls,
    teacher,
    driver,            # None -> teacher drives; else torchscript/StudentActor drives
    episodes: int,
    max_steps: int,
    seed: int,
    behind_fraction: float,
    epsilon: float,
    rng: np.random.Generator,
):
    xs_student, ys_v, ys_h = [], [], []
    sides = ["left", "right"]
    for ep in range(episodes):
        side = sides[ep % 2]
        level = int(rng.integers(1, 7))
        dr = bool(ep % 2 == 0)
        env = env_cls(level=level, side=side, obs_source="state",
                      domain_randomization=dr, seed=seed + ep)
        try:
            if rng.uniform() < behind_fraction:
                env.wall_start_probability = 0.0
                env.self_start_behind_probability = 1.0
                env.self_start_official_probability = 0.0
                env.self_start_front_probability = 0.0
            obs, _ = env.reset(seed=seed + ep)
            for _ in range(max_steps):
                t_in = torch.from_numpy(teacher_input(obs))
                with torch.no_grad():
                    label = teacher(t_in).numpy().astype(np.int64)
                xs_student.append(student_input(obs))
                ys_v.append(int(label[0]))
                ys_h.append(int(label[1]))

                if rng.uniform() < epsilon:
                    act = [int(rng.integers(0, 3)), int(rng.integers(0, 3)), 0]
                elif driver is None:
                    act = [int(label[0]), int(label[1]), 0]
                else:
                    with torch.no_grad():
                        da = driver(torch.from_numpy(student_input(obs)))
                    act = [int(da[0]), int(da[1]), 0]
                obs, _, term, trunc, _ = env.step(act)
                if term or trunc:
                    break
        finally:
            env.close()
    return (np.asarray(xs_student, dtype=np.float32),
            np.asarray(ys_v, dtype=np.int64),
            np.asarray(ys_h, dtype=np.int64))


def train(net: StudentNet, x, yv, yh, *, epochs: int, batch: int, lr: float, seed: int):
    g = torch.Generator().manual_seed(seed)
    n = x.shape[0]
    n_val = max(1, n // 20)
    perm = torch.randperm(n, generator=g)
    xt = torch.from_numpy(x)[perm]
    vt = torch.from_numpy(yv)[perm]
    ht = torch.from_numpy(yh)[perm]
    x_val, v_val, h_val = xt[:n_val], vt[:n_val], ht[:n_val]
    x_tr, v_tr, h_tr = xt[n_val:], vt[n_val:], ht[n_val:]

    opt = torch.optim.Adam(net.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()
    history = []
    for epoch in range(epochs):
        net.train()
        idx = torch.randperm(x_tr.shape[0], generator=g)
        for i in range(0, x_tr.shape[0], batch):
            b = idx[i:i + batch]
            lv, lh = net(x_tr[b])
            loss = ce(lv, v_tr[b]) + ce(lh, h_tr[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            lv, lh = net(x_val)
            acc_v = float((lv.argmax(1) == v_val).float().mean())
            acc_h = float((lh.argmax(1) == h_val).float().mean())
            acc_both = float(((lv.argmax(1) == v_val) & (lh.argmax(1) == h_val)).float().mean())
        history.append({"epoch": epoch, "val_acc_v": acc_v, "val_acc_h": acc_h, "val_acc_both": acc_both})
        print(f"epoch {epoch}: val_acc_v={acc_v:.4f} val_acc_h={acc_h:.4f} both={acc_both:.4f}")
    return history


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-dir", default=r"D:\pickleball\Wall_sim_V67_FT")
    parser.add_argument("--teacher", default=r"D:\pickleball\Wall_sim_V67_FT\pretrained\v54_v67_originals\best_1_policy.pt")
    parser.add_argument("--out-dir", default=r"D:\pickleball\Wall_sim_V81_distill")
    parser.add_argument("--episodes", type=int, default=800)
    parser.add_argument("--dagger-episodes", type=int, default=400)
    parser.add_argument("--rounds", type=int, default=2, help="1 = pure BC, 2+ = add DAgger rounds")
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--behind-fraction", type=float, default=0.3)
    parser.add_argument("--epsilon", type=float, default=0.08)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out_dir / "logs").mkdir(parents=True, exist_ok=True)

    env_cls = load_env(Path(args.env_dir))
    teacher = torch.jit.load(args.teacher, map_location="cpu")
    teacher.eval()
    rng = np.random.default_rng(args.seed)

    t0 = time.time()
    print(f"[round 0] teacher-driven collection: {args.episodes} episodes")
    x, yv, yh = collect(env_cls=env_cls, teacher=teacher, driver=None,
                        episodes=args.episodes, max_steps=args.max_steps,
                        seed=args.seed, behind_fraction=args.behind_fraction,
                        epsilon=args.epsilon, rng=rng)
    print(f"collected {x.shape[0]} samples in {time.time()-t0:.0f}s")

    net = StudentNet()
    all_history = []
    for rnd in range(args.rounds):
        if rnd > 0:
            actor = torch.jit.script(StudentActor(net))
            print(f"[round {rnd}] student-driven DAgger collection: {args.dagger_episodes} episodes")
            x2, yv2, yh2 = collect(env_cls=env_cls, teacher=teacher, driver=actor,
                                   episodes=args.dagger_episodes, max_steps=args.max_steps,
                                   seed=args.seed + 10000 * rnd,
                                   behind_fraction=args.behind_fraction,
                                   epsilon=args.epsilon, rng=rng)
            x = np.concatenate([x, x2]); yv = np.concatenate([yv, yv2]); yh = np.concatenate([yh, yh2])
            print(f"dataset now {x.shape[0]} samples")
        history = train(net, x, yv, yh, epochs=args.epochs, batch=args.batch,
                        lr=args.lr, seed=args.seed + rnd)
        all_history.append(history)

    np.savez_compressed(out_dir / "logs" / "distill_dataset.npz", x=x, yv=yv, yh=yh)
    actor = torch.jit.script(StudentActor(net))
    actor_path = out_dir / "checkpoints" / "student_policy.pt"
    actor.save(str(actor_path))
    torch.save(net.state_dict(), out_dir / "checkpoints" / "student_net_state.pt")

    smoke = actor(torch.zeros(16))
    meta = {
        "format": "torchscript_deterministic_multidiscrete_actor",
        "source": "behavior_cloning_of_v54_deployed_behavior",
        "teacher": str(args.teacher),
        "obs_dim": "13 (also accepts 16, drops sin/cos/success, zeroes time_own_side)",
        "rotation": "hard 0 in export",
        "samples": int(x.shape[0]),
        "final_val": all_history[-1][-1],
        "smoke_action": [int(v) for v in smoke],
    }
    (out_dir / "checkpoints" / "student_policy.pt.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    print(f"saved: {actor_path}")


if __name__ == "__main__":
    main()
