from __future__ import annotations

import hashlib
import json
import math
import random
import shutil
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def sfl_score(success: float, peak: float = 0.50) -> float:
    """Learnability score: high for positive but imperfect success.

    peak=0.50 gives classic p(1-p)-like frontier. 0.35 makes training slightly
    harder because it prioritizes opponents the agent is not yet beating.
    """
    p = float(np.clip(success, 0.0, 1.0))
    peak = float(np.clip(peak, 0.05, 0.95))
    if p <= peak:
        return float(p / peak)
    return float((1.0 - p) / (1.0 - peak))


@dataclass
class OpponentRecord:
    name: str
    path: str
    sha256: str
    visits: int = 0
    agent_points: int = 0
    opponent_points: int = 0
    matches: int = 0
    last_seen_chunk: int = -1
    last_success: float = 0.5
    last_mean_reward: float = 0.0

    @property
    def total_points(self) -> int:
        return int(self.agent_points + self.opponent_points)

    @property
    def success(self) -> float:
        if self.total_points <= 0:
            return 0.5
        return float(self.agent_points) / float(self.total_points)


class LeagueSelector:
    """PSRO-style opponent selector for dPickleBall PPO training.

    This is intentionally lightweight and dependency-free. It keeps an opponent
    pool, a running payoff/stat table, and samples opponents using a mixed policy:
      - unvisited coverage first,
      - PSRO-style weakness/meta mixture,
      - SFL/frontier opponents,
      - hard opponents,
      - random/staleness for diversity.

    It is not a full Nash solver. It is a practical PSRO-style selector that
    can run on the user's Windows setup without scipy/cvxpy.
    """

    def __init__(
        self,
        trainer_pool: str | Path,
        state_path: str | Path,
        *,
        sfl_peak: float = 0.50,
        seed: int = 2026,
    ):
        self.trainer_pool = Path(trainer_pool)
        self.state_path = Path(state_path)
        self.sfl_peak = float(sfl_peak)
        self.rng = random.Random(int(seed))
        self.state: Dict = {
            'global_chunk': 0,
            'records': {},
            'history': [],
        }
        self.load()
        self.refresh_pool()

    def load(self) -> None:
        if self.state_path.exists():
            try:
                self.state = json.loads(self.state_path.read_text(encoding='utf-8'))
            except Exception:
                pass

    def save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(self.state, indent=2), encoding='utf-8')

    def refresh_pool(self) -> None:
        self.trainer_pool.mkdir(parents=True, exist_ok=True)
        records = self.state.setdefault('records', {})
        for p in sorted(self.trainer_pool.glob('*.pt')):
            if p.name not in records:
                records[p.name] = asdict(OpponentRecord(
                    name=p.name,
                    path=str(p),
                    sha256=sha256_file(p),
                ))
            else:
                records[p.name]['path'] = str(p)
                if not records[p.name].get('sha256'):
                    records[p.name]['sha256'] = sha256_file(p)
        existing = {p.name for p in self.trainer_pool.glob('*.pt')}
        for name in list(records.keys()):
            if name not in existing:
                records.pop(name, None)
        self.save()

    def records(self) -> List[OpponentRecord]:
        out = []
        for d in self.state.get('records', {}).values():
            out.append(OpponentRecord(**{k: d.get(k) for k in OpponentRecord.__dataclass_fields__.keys()}))
        return sorted(out, key=lambda r: r.name.lower())

    def choose(self, *, epsilon: float = 0.08) -> Path:
        recs = self.records()
        if not recs:
            raise RuntimeError(f'No .pt trainer files found in {self.trainer_pool}')

        unvisited = [r for r in recs if int(r.visits) <= 0]
        if unvisited:
            return Path(self.rng.choice(unvisited).path)

        if self.rng.random() < epsilon:
            return Path(self.rng.choice(recs).path)

        x = self.rng.random()
        if x < 0.40:
            chosen = self._sample_meta_weakness(recs)
        elif x < 0.70:
            chosen = self._sample_sfl_frontier(recs)
        elif x < 0.90:
            chosen = self._sample_hard(recs)
        else:
            chosen = self._sample_stale_or_random(recs)
        return Path(chosen.path)

    def _softmax_sample(self, pairs: List[Tuple[float, OpponentRecord]], temperature: float = 0.20) -> OpponentRecord:
        scores = np.asarray([float(s) for s, _ in pairs], dtype=np.float64)
        scores = np.nan_to_num(scores, nan=0.0, posinf=1.0, neginf=0.0)
        if len(scores) == 1:
            return pairs[0][1]
        probs = np.exp((scores - scores.max()) / max(temperature, 1e-6))
        probs = probs / probs.sum()
        idx = int(np.random.choice(len(pairs), p=probs))
        return pairs[idx][1]

    def _sample_meta_weakness(self, recs: List[OpponentRecord]) -> OpponentRecord:
        pairs = []
        for r in recs:
            p = r.success
            weakness = max(0.0, 0.70 - p) / 0.70
            frontier = sfl_score(p, peak=self.sfl_peak)
            visit_smooth = 1.0 / math.sqrt(max(1, r.visits))
            score = 0.65 * weakness + 0.25 * frontier + 0.10 * visit_smooth
            pairs.append((score, r))
        return self._softmax_sample(pairs, temperature=0.18)

    def _sample_sfl_frontier(self, recs: List[OpponentRecord]) -> OpponentRecord:
        pairs = []
        for r in recs:
            pairs.append((sfl_score(r.success, peak=self.sfl_peak), r))
        return self._softmax_sample(pairs, temperature=0.20)

    def _sample_hard(self, recs: List[OpponentRecord]) -> OpponentRecord:
        pairs = []
        for r in recs:
            p = r.success
            score = 1.0 - p
            pairs.append((score, r))
        return self._softmax_sample(pairs, temperature=0.15)

    def _sample_stale_or_random(self, recs: List[OpponentRecord]) -> OpponentRecord:
        chunk = int(self.state.get('global_chunk', 0))
        pairs = []
        for r in recs:
            stale = 1.0 if r.last_seen_chunk < 0 else min(1.0, max(0.0, (chunk - r.last_seen_chunk) / max(1, len(recs))))
            pairs.append((0.7 * stale + 0.3 * random.random(), r))
        return self._softmax_sample(pairs, temperature=0.25)

    def update_after_chunk(
        self,
        opponent_name: str,
        *,
        agent_points: int,
        opponent_points: int,
        matches: int,
        mean_reward: float,
    ) -> None:
        records = self.state.setdefault('records', {})
        if opponent_name not in records:
            p = self.trainer_pool / opponent_name
            records[opponent_name] = asdict(OpponentRecord(
                name=opponent_name,
                path=str(p),
                sha256=sha256_file(p) if p.exists() else '',
            ))
        r = records[opponent_name]
        r['visits'] = int(r.get('visits', 0)) + 1
        r['agent_points'] = int(r.get('agent_points', 0)) + int(agent_points)
        r['opponent_points'] = int(r.get('opponent_points', 0)) + int(opponent_points)
        r['matches'] = int(r.get('matches', 0)) + int(matches)
        total = int(agent_points + opponent_points)
        r['last_success'] = float(agent_points) / float(total) if total > 0 else 0.5
        r['last_mean_reward'] = float(mean_reward)
        r['last_seen_chunk'] = int(self.state.get('global_chunk', 0))
        self.state.setdefault('history', []).append({
            'chunk': int(self.state.get('global_chunk', 0)),
            'opponent': opponent_name,
            'agent_points': int(agent_points),
            'opponent_points': int(opponent_points),
            'matches': int(matches),
            'mean_reward': float(mean_reward),
        })
        self.state['global_chunk'] = int(self.state.get('global_chunk', 0)) + 1
        self.save()

    def add_checkpoint_to_pool(self, checkpoint_pt: str | Path, prefix: str = 'selfplay') -> Path:
        checkpoint_pt = Path(checkpoint_pt)
        if not checkpoint_pt.exists():
            raise FileNotFoundError(checkpoint_pt)
        digest = sha256_file(checkpoint_pt)[:10]
        dst = self.trainer_pool / f'{prefix}_{checkpoint_pt.stem}_{digest}.pt'
        if not dst.exists():
            shutil.copy2(checkpoint_pt, dst)
        self.refresh_pool()
        return dst

    def export_csv(self, path: str | Path) -> None:
        import csv
        rows = []
        for r in self.records():
            rows.append({
                'name': r.name,
                'visits': r.visits,
                'agent_points': r.agent_points,
                'opponent_points': r.opponent_points,
                'success': r.success,
                'sfl_score': sfl_score(r.success, self.sfl_peak),
                'matches': r.matches,
                'last_seen_chunk': r.last_seen_chunk,
                'sha256': r.sha256,
            })
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', newline='', encoding='utf-8') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ['name'])
            w.writeheader()
            w.writerows(rows)
