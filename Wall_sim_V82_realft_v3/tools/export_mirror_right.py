from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# Mirror a LEFT-trained actor into a RIGHT-side player by reflection about the
# court's vertical center line. The game is left-right symmetric (net in the
# middle, symmetric wall segments, no end walls), so a right-side situation is
# exactly a mirrored left-side situation. This is the same principle as the
# V67 router's best_2 (weight-baked mirror of V54), but cleaner: our model is
# angle-blind (no sin/cos to mirror) and rotation is hard 0.
#
# Feature vector is the 16-dim V23 layout fed by OpenCVStateExtractor(side=...).
# Horizontal quantities flip sign under the mirror:
#   0 ball_x, 2 ball_vx, 4 paddle_x, 8 relative_x, 14 side_sign
# Everything vertical (y, vy, opponent_y) is unchanged; angle/success indices
# (6,7,11) are dropped by StudentActor anyway.
#
# Action: vertical (up/down) is unchanged; horizontal (1 right / 2 left) flips
# via (3 - h) % 3 ; rotation stays 0.

FLIP_IDX = (0, 2, 4, 8, 14)


class MirrorRightActor(nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xm = x.clone()
        xm[0] = -x[0]
        xm[2] = -x[2]
        xm[4] = -x[4]
        xm[8] = -x[8]
        xm[14] = -x[14]
        a = self.inner(xm)
        v = a[0]
        h = (3 - a[1]) % 3
        r = torch.zeros_like(v)
        return torch.stack([v, h, r])


def mirror_features(x: np.ndarray) -> np.ndarray:
    xm = np.array(x, dtype=np.float32).copy()
    for i in FLIP_IDX:
        xm[i] = -xm[i]
    return xm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="left-trained TorchScript actor (.pt)")
    parser.add_argument("--out", required=True, help="output right-mirror actor (.pt)")
    parser.add_argument("--tests", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    inner = torch.jit.load(args.model, map_location="cpu")
    inner.eval()
    wrapped = torch.jit.script(MirrorRightActor(inner))

    # Verify the reflection identity: right(x) == mirror_action( left(mirror_feat(x)) ).
    rng = np.random.default_rng(args.seed)
    action_errors = 0
    rotate_nonzero = 0
    for i in range(args.tests):
        x = rng.uniform(-1.0, 1.0, size=16).astype(np.float32)
        x[14] = 1.0  # right-side features
        with torch.no_grad():
            out = wrapped(torch.from_numpy(x))
            inner_on_mirror = inner(torch.from_numpy(mirror_features(x)))
        want_v = int(inner_on_mirror[0])
        want_h = (3 - int(inner_on_mirror[1])) % 3
        if [int(out[0]), int(out[1])] != [want_v, want_h]:
            action_errors += 1
        if int(out[2]) != 0:
            rotate_nonzero += 1

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    wrapped.save(args.out)
    print(f"saved: {args.out}")
    print(f"random_tests={args.tests} action_identity_errors={action_errors} rotate_nonzero={rotate_nonzero}")
    if action_errors or rotate_nonzero:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
