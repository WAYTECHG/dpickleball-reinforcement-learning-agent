from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any


class CurriculumManager:
    def __init__(self, state_path: Path, max_level: int = 5):
        self.state_path = Path(state_path)
        self.max_level = int(max_level)
        self.state = {
            "current_level": 1,
            "total_timesteps": 0,
            "level_success_rates": {},
            "best_score": None,
        }
        self.load()

    @property
    def current_level(self) -> int:
        return int(self.state.get("current_level", 1))

    @property
    def total_timesteps(self) -> int:
        return int(self.state.get("total_timesteps", 0))

    def load(self) -> None:
        if self.state_path.exists():
            try:
                self.state.update(json.loads(self.state_path.read_text(encoding="utf-8")))
            except Exception:
                pass

    def save(self) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(self.state, indent=2), encoding="utf-8")

    def add_timesteps(self, n: int) -> None:
        self.state["total_timesteps"] = self.total_timesteps + int(n)
        self.save()

    def update_eval(self, eval_result: Dict[int, Dict[str, Any]]) -> None:
        rates = self.state.setdefault("level_success_rates", {})
        for level, res in eval_result.items():
            rates[str(level)] = float(res.get("success_rate", 0.0))
        self.save()

    def maybe_advance(self, current_success_rate: float, pass_threshold: float = 0.80) -> bool:
        if current_success_rate >= pass_threshold and self.current_level < self.max_level:
            self.state["current_level"] = self.current_level + 1
            self.save()
            return True
        return False

    def set_level(self, level: int) -> None:
        self.state["current_level"] = int(max(1, min(self.max_level, level)))
        self.save()
