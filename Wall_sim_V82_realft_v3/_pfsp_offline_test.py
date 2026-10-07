"""Offline sanity check of the SHIPPED PFSP weighting (realft_env._opponent_weights).
Calls the real method on a fake `self` so we test the actual code, no Unity/torch init."""
import sys
from collections import deque
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, r"D:\pickleball\Wall_sim_V82_realft")
from realft_env import RealUnityFightEnv  # noqa: E402

NAMES = ["c29_nethug", "realft_self", "router", "v54_leader", "v84_leader"]


def weights(winrates, floor=0.10, min_samples=6, n=20):
    rec = {}
    for nm in NAMES:
        wr = winrates.get(nm, None)
        if wr is None:
            rec[nm] = deque(maxlen=30)                      # cold start: no data
        else:
            ones = int(round(wr * n))
            rec[nm] = deque([1] * ones + [0] * (n - ones), maxlen=30)
    fake = SimpleNamespace(_pool=[(nm, None) for nm in NAMES], _recent=rec,
                           pfsp_floor=floor, pfsp_min_samples=min_samples)
    return RealUnityFightEnv._opponent_weights(fake)


def show(title, winrates):
    p = weights(winrates)
    print(f"\n=== {title} ===")
    for nm, w in zip(NAMES, p):
        wr = winrates.get(nm)
        tag = f"wr={wr:.2f}" if wr is not None else "wr=? (cold)"
        print(f"  {nm:<13} {tag}   sample_prob={w:6.1%}")
    print(f"  sum={p.sum():.3f}  min={p.min():.1%}  (floor guarantee 10%)")
    assert abs(p.sum() - 1.0) < 1e-9, "weights must sum to 1"
    assert p.min() >= 0.10 - 1e-9, "FLOOR VIOLATED: an opponent dropped below 10% (would be near-eliminated)"


# Scenario A: the actual last-window winrates (router/v54/self even, v84/c29 mastered)
show("A  observed (v84 mastered, router contested)",
     {"c29_nethug": 0.78, "realft_self": 0.50, "router": 0.50, "v54_leader": 0.55, "v84_leader": 1.00})

# Scenario B: cold start (no recent data) -> must be ~uniform so we explore first
show("B  cold start (no data yet)", {})

# Scenario C: router later gets mastered too -> weight shifts AWAY but router still >= floor
show("C  router mastered later (0.85)",
     {"c29_nethug": 0.80, "realft_self": 0.55, "router": 0.85, "v54_leader": 0.55, "v84_leader": 0.95})

print("\nALL CHECKS PASSED: sums=1, floor>=10% always (no opponent ever eliminated).")
