from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

THIS_DIR = Path(__file__).resolve().parent
TOOLS_DIR = r"D:\pickleball\tools"
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

from realft_env import RealUnityFightEnv  # noqa: E402
from distill_v54_student import StudentNet, StudentActor  # noqa: E402


def load_student_net(state_path: str) -> StudentNet:
    net = StudentNet()
    net.load_state_dict(torch.load(state_path, map_location="cpu"))
    net.eval()
    return net


def inject_student_into_policy(model: PPO, net: StudentNet) -> None:
    """Copy distilled student weights into the SB3 actor (policy trunk + action head).

    The value network keeps its fresh initialization; use --critic-warmup so the
    critic learns before the actor is allowed to move.
    """
    policy_net = model.policy.mlp_extractor.policy_net
    trunk_linears = [m for m in net.trunk if isinstance(m, nn.Linear)]
    policy_linears = [m for m in policy_net if isinstance(m, nn.Linear)]
    assert len(trunk_linears) == len(policy_linears), "net_arch mismatch with StudentNet"
    with torch.no_grad():
        for src, dst in zip(trunk_linears, policy_linears):
            dst.weight.copy_(src.weight)
            dst.bias.copy_(src.bias)
        action_net = model.policy.action_net
        action_net.weight[0:3].copy_(net.head_v.weight)
        action_net.bias[0:3].copy_(net.head_v.bias)
        action_net.weight[3:6].copy_(net.head_h.weight)
        action_net.bias[3:6].copy_(net.head_h.bias)


def verify_transfer(model: PPO, net: StudentNet, n: int = 500, seed: int = 0) -> int:
    rng = np.random.default_rng(seed)
    mismatches = 0
    for _ in range(n):
        x = rng.uniform(-1.0, 1.0, size=13).astype(np.float32)
        sb3_action, _ = model.predict(x, deterministic=True)
        with torch.no_grad():
            lv, lh = net(torch.from_numpy(x).unsqueeze(0))
        want = [int(lv.argmax()), int(lh.argmax())]
        if [int(sb3_action[0]), int(sb3_action[1])] != want:
            mismatches += 1
    return mismatches


def export_torchscript(model: PPO, out_path: Path) -> None:
    """Export the current SB3 actor back to the deployable StudentActor format
    (13/16-dim adapter, deterministic argmax, rotation hard 0)."""
    net = StudentNet()
    policy_net = model.policy.mlp_extractor.policy_net
    policy_linears = [m for m in policy_net if isinstance(m, nn.Linear)]
    trunk_linears = [m for m in net.trunk if isinstance(m, nn.Linear)]
    with torch.no_grad():
        for src, dst in zip(policy_linears, trunk_linears):
            dst.weight.copy_(src.weight)
            dst.bias.copy_(src.bias)
        net.head_v.weight.copy_(model.policy.action_net.weight[0:3])
        net.head_v.bias.copy_(model.policy.action_net.bias[0:3])
        net.head_h.weight.copy_(model.policy.action_net.weight[3:6])
        net.head_h.bias.copy_(model.policy.action_net.bias[3:6])
    net.eval()
    actor = torch.jit.script(StudentActor(net))
    actor.save(str(out_path))


class ActorFreezeCallback(BaseCallback):
    def __init__(self, freeze_steps: int):
        super().__init__()
        self.freeze_steps = int(freeze_steps)
        self._frozen = False

    def _actor_params(self):
        yield from self.model.policy.mlp_extractor.policy_net.parameters()
        yield from self.model.policy.action_net.parameters()

    def _on_training_start(self) -> None:
        if self.freeze_steps > 0:
            for p in self._actor_params():
                p.requires_grad_(False)
            self._frozen = True
            print(f"[warmup] actor frozen for first {self.freeze_steps} steps (critic-only)")

    def _on_step(self) -> bool:
        if self._frozen and self.num_timesteps >= self.freeze_steps:
            for p in self._actor_params():
                p.requires_grad_(True)
            self._frozen = False
            print(f"[warmup] actor unfrozen at {self.num_timesteps} steps")
        return True


class CheckpointExportCallback(BaseCallback):
    def __init__(self, save_freq: int, out_dir: Path, raw_env: RealUnityFightEnv,
                 log_path: Path | None = None):
        super().__init__()
        self.save_freq = int(save_freq)
        self.out_dir = out_dir
        self.raw_env = raw_env
        self.log_path = log_path
        self._last_save = 0
        self._last_stats: dict = {}

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_save >= self.save_freq:
            self._last_save = self.num_timesteps
            tag = f"realft_{self.num_timesteps:08d}"
            self.model.save(str(self.out_dir / f"{tag}.zip"))
            export_torchscript(self.model, self.out_dir / f"{tag}_policy.pt")
            export_torchscript(self.model, self.out_dir / "latest_policy.pt")
            parts = []
            won_all = lost_all = 0
            for key in sorted(self.raw_env.stats):
                w, l = self.raw_env.stats[key]
                pw, pl = self._last_stats.get(key, (0, 0))
                dw, dl = w - pw, l - pl
                won_all += dw
                lost_all += dl
                if dw + dl > 0:
                    parts.append(f"{key}={dw}:{dl}({dw / (dw + dl):.2f})")
            self._last_stats = {k: tuple(v) for k, v in self.raw_env.stats.items()}
            total = won_all + lost_all
            wr = (won_all / total) if total > 0 else float("nan")
            line = (f"[ckpt] steps={self.num_timesteps} saved {tag}; window W:L={won_all}:{lost_all} "
                    f"(winrate={wr:.3f}) {' '.join(parts)} relaunches={self.raw_env.relaunch_count}")
            print(line)
            if self.log_path is not None:
                with open(self.log_path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
        return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--student-state", default=r"D:\pickleball\Wall_sim_V81_distill\checkpoints\student_net_state.pt")
    parser.add_argument("--resume-zip", default="", help="resume from an SB3 .zip instead of the distilled student")
    parser.add_argument("--env-path", default=r"D:\pickleball\dPickleball BuildFiles\Competition\Windows\dp.exe")
    parser.add_argument("--opponent", default=str(Path(r"D:\pickleball\Wall_sim_V67_FT") / "pretrained" / "v54_v67_originals" / "best_2_policy.pt"))
    parser.add_argument("--opponent-pool-dir", default="", help="dir of right-native .pt opponents; sampled per match for diversity")
    parser.add_argument("--serve-code", type=int, default=9993,
                        help="match point 999 + random serve: whole run in one app life, one START click")
    parser.add_argument("--worker-id", type=int, default=60)
    parser.add_argument("--no-graphics", action="store_true",
                        help="true headless (this build is known to hang; default keeps graphics)")
    parser.add_argument("--no-auto-start", action="store_true",
                        help="disable the automatic START click (then click manually)")
    parser.add_argument("--start-delay", type=float, default=15.0)
    parser.add_argument("--start-y-ratio", type=float, default=0.875)
    parser.add_argument("--hit-bonus", type=float, default=0.05,
                        help="real-event shaping: own-paddle hit detected from the image stream")
    parser.add_argument("--cross-bonus", type=float, default=0.02,
                        help="real-event shaping: ball crosses into the opponent half")
    parser.add_argument("--learner-side", choices=["left", "right", "alternate"], default="left")
    parser.add_argument("--snapshot-prob", type=float, default=0.0,
                        help=">0 enables self-play: opponent is the latest exported snapshot with this probability")
    parser.add_argument("--total-steps", type=int, default=50000)
    parser.add_argument("--save-freq", type=int, default=10000)
    parser.add_argument("--critic-warmup", type=int, default=8000)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--n-steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=0.999)
    parser.add_argument("--clip-range", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--continue-timesteps", action="store_true", help="resume num_timesteps from the loaded zip (reset_num_timesteps=False) so step tags continue across watchdog restarts")
    parser.add_argument("--pfsp", action="store_true", help="sample pool opponents by recent learnability (floored, never eliminated) instead of uniform")
    args = parser.parse_args()

    out_dir = THIS_DIR / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)

    env = RealUnityFightEnv(
        env_path=args.env_path,
        opponent_model=args.opponent,
        opponent_pool_dir=args.opponent_pool_dir,
        snapshot_path=str(out_dir / "latest_policy.pt") if args.snapshot_prob > 0 else "",
        snapshot_prob=args.snapshot_prob,
        learner_side=args.learner_side,
        serve_code=args.serve_code,
        worker_id=args.worker_id,
        graphics=not args.no_graphics,
        auto_start=not args.no_auto_start,
        start_delay=args.start_delay,
        start_y_ratio=args.start_y_ratio,
        hit_bonus=args.hit_bonus,
        cross_bonus=args.cross_bonus,
        seed=args.seed,
        pfsp=args.pfsp,
    )

    if args.resume_zip:
        model = PPO.load(args.resume_zip, env=env, device="cpu")
        export_torchscript(model, out_dir / "latest_policy.pt")
        print(f"resumed from {args.resume_zip}")
    else:
        model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=args.lr,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=0.95,
            clip_range=args.clip_range,
            ent_coef=0.0,
            policy_kwargs=dict(
                net_arch=dict(pi=[128, 128, 64], vf=[128, 128, 64]),
                activation_fn=nn.Tanh,
            ),
            seed=args.seed,
            device="cpu",
            verbose=1,
        )
        net = load_student_net(args.student_state)
        inject_student_into_policy(model, net)
        mismatches = verify_transfer(model, net)
        print(f"weight transfer verification: {mismatches}/500 mismatches")
        if mismatches:
            raise SystemExit("transfer verification failed; aborting before training")
        export_torchscript(model, out_dir / "realft_00000000_policy.pt")

    callbacks = [
        ActorFreezeCallback(args.critic_warmup),
        CheckpointExportCallback(args.save_freq, out_dir, env,
                                 log_path=THIS_DIR / "logs" / "ckpt_log.txt"),
    ]
    meta = {
        "base": args.resume_zip or args.student_state,
        "opponent": args.opponent,
        "serve_code": args.serve_code,
        "lr": args.lr,
        "gamma": args.gamma,
        "clip_range": args.clip_range,
        "critic_warmup": args.critic_warmup,
        "total_steps": args.total_steps,
    }
    (THIS_DIR / "logs" / "run_config.json").write_text(json.dumps(meta, indent=2))

    try:
        model.learn(total_timesteps=args.total_steps, callback=callbacks, progress_bar=False,
                    reset_num_timesteps=not args.continue_timesteps)
    finally:
        model.save(str(out_dir / "realft_final.zip"))
        export_torchscript(model, out_dir / "realft_final_policy.pt")
        print(f"final points L:R = {env.points[0]:.0f}:{env.points[1]:.0f}, "
              f"relaunches={env.relaunch_count}, unity_steps={env.lifetime_steps}")
        env.close()


if __name__ == "__main__":
    main()
