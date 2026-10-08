from __future__ import annotations

"""
Export a Stable-Baselines3 PPO MultiDiscrete MlpPolicy into a pure PyTorch
TorchScript actor for deployment.

Why this exists:
- SB3 .zip checkpoints are excellent for resume/fine-tuning/self-play training.
- Competition submission should not depend on Stable-Baselines3 being installed.
- The exported .pt file can be loaded with torch.jit.load() and directly returns
  the deterministic discrete action branches.

The exported actor expects the SAME feature vector used during training
(default V23 shape: [16]) and returns an int64 tensor of shape [3]:
    [vertical_action, horizontal_action, rotation_action]
"""

from pathlib import Path
from typing import Sequence, Dict, Any
import json

import torch
import torch.nn as nn


class _DeterministicMultiDiscreteActor(nn.Module):
    """Small wrapper around the SB3 actor-only modules."""

    def __init__(self, policy_net: nn.Module, action_net: nn.Module, action_dims: Sequence[int]):
        super().__init__()
        self.policy_net = policy_net
        self.action_net = action_net
        self.action_dims = list(map(int, action_dims))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        squeeze = False
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
            squeeze = True
        obs = obs.to(torch.float32)
        latent_pi = self.policy_net(obs)
        logits = self.action_net(latent_pi)
        splits = torch.split(logits, self.action_dims, dim=1)
        actions = [torch.argmax(branch_logits, dim=1) for branch_logits in splits]
        out = torch.stack(actions, dim=1).to(torch.int64)
        if squeeze:
            out = out.squeeze(0)
        return out


def _infer_action_dims_from_sb3_model(model) -> list[int]:
    action_space = getattr(model, "action_space", None)
    nvec = getattr(action_space, "nvec", None)
    if nvec is None:
        return [3, 3, 3]
    return [int(x) for x in list(nvec)]


def _infer_obs_dim_from_sb3_model(model) -> int:
    obs_space = getattr(model, "observation_space", None)
    shape = getattr(obs_space, "shape", None)
    if shape is not None and len(shape) == 1:
        return int(shape[0])
    return 16


def export_sb3_policy_to_torchscript(
    model,
    output_path: str | Path,
    *,
    metadata: Dict[str, Any] | None = None,
    verify: bool = True,
) -> Path:
    """
    Export an already-loaded SB3 PPO model to a pure TorchScript .pt actor.

    This does NOT save the value network or optimizer because the .pt file is for
    deterministic deployment/inference. Keep the SB3 .zip for resume/self-play.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    policy = model.policy
    policy.eval()

    action_dims = _infer_action_dims_from_sb3_model(model)
    obs_dim = _infer_obs_dim_from_sb3_model(model)

    actor = _DeterministicMultiDiscreteActor(
        policy_net=policy.mlp_extractor.policy_net,
        action_net=policy.action_net,
        action_dims=action_dims,
    ).eval().cpu()

    example = torch.zeros(obs_dim, dtype=torch.float32)
    traced = torch.jit.trace(actor, example, strict=False)
    traced.save(str(output_path))

    meta = {
        "format": "torchscript_deterministic_multidiscrete_actor",
        "source": "stable_baselines3_ppo_mlp_policy",
        "obs_dim": obs_dim,
        "action_dims": action_dims,
        "returns": "int64 tensor [vertical, horizontal, rotation]",
        "requires_stable_baselines3_for_inference": False,
        "note": "Keep the .zip checkpoint for resume/fine-tuning; use this .pt for lightweight deployment.",
    }
    if metadata:
        meta.update(metadata)
    output_path.with_suffix(output_path.suffix + ".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if verify:
        loaded = torch.jit.load(str(output_path), map_location="cpu")
        with torch.no_grad():
            y = loaded(torch.zeros(obs_dim, dtype=torch.float32))
        if tuple(y.shape) != (len(action_dims),):
            raise RuntimeError(f"Exported policy returned wrong shape: {tuple(y.shape)}")
        if not torch.all((y >= 0) & (y < torch.tensor(action_dims, dtype=torch.int64))).item():
            raise RuntimeError(f"Exported policy returned invalid action values: {y}")

    return output_path


def export_sb3_zip_to_torchscript(
    sb3_zip_path: str | Path,
    output_path: str | Path,
    *,
    device: str = "cpu",
    metadata: Dict[str, Any] | None = None,
) -> Path:
    """Load an SB3 PPO .zip checkpoint and export it to TorchScript .pt."""
    from stable_baselines3 import PPO

    model = PPO.load(str(sb3_zip_path), device=device)
    return export_sb3_policy_to_torchscript(model, output_path, metadata=metadata)
