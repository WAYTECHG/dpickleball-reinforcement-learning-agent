from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List
import shutil
import time

from curriculum.torch_policy_exporter import export_sb3_policy_to_torchscript, export_sb3_zip_to_torchscript


class CheckpointManager:

    def __init__(self, checkpoint_dir: Path, top_k: int = 3, strict_cleanup: bool = True):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.top_k = int(top_k)
        self.strict_cleanup = bool(strict_cleanup)
        self.index_path = self.checkpoint_dir / "top_models.json"
        self.top_models: List[Dict[str, Any]] = []
        self.load()
        self._sanitize_index()
        self._cleanup_extra_zip_files()

    def load(self) -> None:
        if self.index_path.exists():
            try:
                data = json.loads(self.index_path.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    self.top_models = data
                else:
                    self.top_models = []
            except Exception:
                self.top_models = []
        else:
            self.top_models = []

    def _record_level(self, rec: Dict[str, Any]) -> int:
        meta = rec.get("metadata", {}) or {}
        for key in ("rank_level", "current_level", "level"):
            try:
                if key in meta:
                    return int(meta.get(key, 0))
            except Exception:
                pass

        try:
            eval_result = meta.get("eval_result", {}) or {}
            levels = [int(k) for k in eval_result.keys()]
            if levels:
                return max(levels)
        except Exception:
            pass
        return 0

    def _record_steps(self, rec: Dict[str, Any]) -> int:
        meta = rec.get("metadata", {}) or {}
        try:
            return int(meta.get("total_timesteps", 0))
        except Exception:
            return 0

    def _record_rank_key(self, rec: Dict[str, Any]):
        try:
            score = float(rec.get("score", 0.0))
        except Exception:
            score = 0.0
        return (self._record_level(rec), score, self._record_steps(rec))

    def _sanitize_index(self) -> None:
        cleaned: List[Dict[str, Any]] = []
        for rec in self.top_models:
            if not isinstance(rec, dict):
                continue
            if "score" not in rec:
                continue
            try:
                rec["score"] = float(rec["score"])
            except Exception:
                continue
            cleaned.append(rec)
        cleaned = sorted(cleaned, key=self._record_rank_key, reverse=True)[: self.top_k]

        for i, rec in enumerate(cleaned, start=1):
            rec["path"] = str(self.best_path(i))
            rec["rank"] = i
        self.top_models = cleaned
        self.save_index()

    def save_index(self) -> None:
        self.index_path.write_text(json.dumps(self.top_models, indent=2), encoding="utf-8")

    def latest_path(self) -> Path:
        return self.checkpoint_dir / "latest.zip"

    def best_path(self, rank: int = 1) -> Path:
        return self.checkpoint_dir / f"best_{rank}.zip"

    def latest_policy_path(self) -> Path:
        return self.checkpoint_dir / "latest_policy.pt"

    def best_policy_path(self, rank: int = 1) -> Path:
        return self.checkpoint_dir / f"best_{rank}_policy.pt"

    def has_best(self) -> bool:
        return self.best_path(1).exists()

    def save_latest(self, model) -> None:
        model.save(str(self.latest_path()))
        export_sb3_policy_to_torchscript(
            model,
            self.latest_policy_path(),
            metadata={"checkpoint_rank": "latest"},
        )
        self._cleanup_extra_zip_files()

    def _cleanup_extra_zip_files(self) -> None:
        if not self.strict_cleanup:
            return
        allowed_zip = {self.latest_path().resolve()}
        allowed_pt = {self.latest_policy_path().resolve()}
        allowed_json = {self.index_path.resolve()}
        for i in range(1, self.top_k + 1):
            allowed_zip.add(self.best_path(i).resolve())
            allowed_pt.add(self.best_policy_path(i).resolve())
            allowed_json.add(self.best_policy_path(i).with_suffix(".pt.json").resolve())
        allowed_json.add(self.latest_policy_path().with_suffix(".pt.json").resolve())

        for p in self.checkpoint_dir.glob("*.zip"):
            try:
                if p.resolve() not in allowed_zip:
                    p.unlink(missing_ok=True)
            except Exception:
                pass
        for p in self.checkpoint_dir.glob("*.pt"):
            try:
                if p.resolve() not in allowed_pt:
                    p.unlink(missing_ok=True)
            except Exception:
                pass
        for p in self.checkpoint_dir.glob("*.pt.json"):
            try:
                if p.resolve() not in allowed_json:
                    p.unlink(missing_ok=True)
            except Exception:
                pass
        for p in self.checkpoint_dir.glob("_*"):
            try:
                if p.is_file():
                    p.unlink(missing_ok=True)
            except Exception:
                pass

    def update_top_k(self, model, score: float, metadata: Dict[str, Any]) -> bool:
        self.load()
        self._sanitize_index()

        candidate_id = f"candidate_{int(time.time() * 1000)}"
        candidate = {
            "id": candidate_id,
            "score": float(score),
            "metadata": metadata,
        }


        old_binary_by_id: Dict[str, bytes] = {}
        old_policy_binary_by_id: Dict[str, bytes] = {}
        old_policy_meta_by_id: Dict[str, bytes] = {}
        old_records: List[Dict[str, Any]] = []
        for i, rec in enumerate(self.top_models, start=1):
            rec_id = str(rec.get("id", f"old_rank_{i}"))
            rec["id"] = rec_id
            p = self.best_path(i)
            pt = self.best_policy_path(i)
            pt_meta = pt.with_suffix(".pt.json")
            if p.exists():
                old_binary_by_id[rec_id] = p.read_bytes()
                if pt.exists():
                    old_policy_binary_by_id[rec_id] = pt.read_bytes()
                if pt_meta.exists():
                    old_policy_meta_by_id[rec_id] = pt_meta.read_bytes()
                old_records.append(rec)

        records = old_records + [candidate]
        records = sorted(records, key=self._record_rank_key, reverse=True)[: self.top_k]

        included = any(str(r.get("id")) == candidate_id for r in records)
        if not included:

            self._cleanup_extra_zip_files()
            return False

        candidate_temp = self.checkpoint_dir / "_candidate_temp.zip"
        candidate_policy_temp = self.checkpoint_dir / "_candidate_policy_temp.pt"
        model.save(str(candidate_temp))
        export_sb3_policy_to_torchscript(
            model,
            candidate_policy_temp,
            metadata={"checkpoint_rank": "candidate", "score": float(score)},
        )
        candidate_bytes = candidate_temp.read_bytes()
        candidate_policy_bytes = candidate_policy_temp.read_bytes()
        candidate_policy_meta_path = candidate_policy_temp.with_suffix(".pt.json")
        candidate_policy_meta_bytes = candidate_policy_meta_path.read_bytes() if candidate_policy_meta_path.exists() else b""


        new_top: List[Dict[str, Any]] = []
        for rank, rec in enumerate(records, start=1):
            dest = self.best_path(rank)
            dest_policy = self.best_policy_path(rank)
            dest_policy_meta = dest_policy.with_suffix(".pt.json")
            rec_id = str(rec.get("id"))
            if rec_id == candidate_id:
                dest.write_bytes(candidate_bytes)
                dest_policy.write_bytes(candidate_policy_bytes)
                if candidate_policy_meta_bytes:
                    dest_policy_meta.write_bytes(candidate_policy_meta_bytes)
            else:
                if rec_id in old_binary_by_id:
                    dest.write_bytes(old_binary_by_id[rec_id])
                else:


                    dest.write_bytes(candidate_bytes)

                if rec_id in old_policy_binary_by_id:
                    dest_policy.write_bytes(old_policy_binary_by_id[rec_id])
                    if rec_id in old_policy_meta_by_id:
                        dest_policy_meta.write_bytes(old_policy_meta_by_id[rec_id])
                else:


                    try:
                        export_sb3_zip_to_torchscript(
                            dest,
                            dest_policy,
                            metadata={"checkpoint_rank": f"best_{rank}", "upgraded_from_zip": True},
                        )
                    except Exception:


                        dest_policy.write_bytes(candidate_policy_bytes)
                        if candidate_policy_meta_bytes:
                            dest_policy_meta.write_bytes(candidate_policy_meta_bytes)
            rec = dict(rec)
            rec["rank"] = rank
            rec["path"] = str(dest)
            rec["policy_pt_path"] = str(dest_policy)
            new_top.append(rec)

        for tmp in [candidate_temp, candidate_policy_temp, candidate_policy_meta_path]:
            if tmp.exists():
                tmp.unlink()

        self.top_models = new_top
        self.save_index()
        self._cleanup_extra_zip_files()
        return True

    def print_top_k(self) -> None:
        if not self.top_models:
            print("[TopK] No top models saved yet.")
            return
        print("[TopK] Current best checkpoints:")
        for rec in self.top_models:
            rank = rec.get("rank", "?")
            score = float(rec.get("score", 0.0))
            meta = rec.get("metadata", {}) or {}
            lvl = meta.get("current_level", "?")
            steps = meta.get("total_timesteps", "?")
            pt_exists = self.best_policy_path(int(rank)).exists() if str(rank).isdigit() else False
            pt_status = "pt=yes" if pt_exists else "pt=missing"
            print(f"  best_{rank}.zip + best_{rank}_policy.pt | level={lvl} | score={score:.2f} | steps={steps} | {pt_status}")


def compute_validation_score(eval_result: Dict[int, Dict[str, Any]], current_level: int) -> float:
    levels = sorted(eval_result.keys())
    mastered = 0
    for lvl in levels:
        if eval_result[lvl].get("success_rate", 0.0) >= 0.80:
            mastered = max(mastered, lvl)

    avg_success = sum(eval_result[l].get("success_rate", 0.0) for l in levels) / max(1, len(levels))
    avg_returns = sum(eval_result[l].get("avg_returns", 0.0) for l in levels) / max(1, len(levels))
    fail_rate = sum(eval_result[l].get("fail_rate", 0.0) for l in levels) / max(1, len(levels))
    front_camp_rate = sum(eval_result[l].get("front_camp_rate", 0.0) for l in levels) / max(1, len(levels))
    action_to_net_rate = sum(eval_result[l].get("action_to_net_rate", 0.0) for l in levels) / max(1, len(levels))
    avg_strong_bounces = sum(eval_result[l].get("avg_strong_bounces", 0.0) for l in levels) / max(1, len(levels))
    avg_recovery_rewards = sum(eval_result[l].get("avg_recovery_rewards", 0.0) for l in levels) / max(1, len(levels))
    avg_home_distance = sum(eval_result[l].get("avg_home_distance", 0.0) for l in levels) / max(1, len(levels))
    avg_opponent_avoid_rate = sum(eval_result[l].get("avg_opponent_avoid_rate", 0.0) for l in levels) / max(1, len(levels))


    side_gap = sum(abs(eval_result[l].get("left_success_rate", 0.0) - eval_result[l].get("right_success_rate", 0.0)) for l in levels) / max(1, len(levels))

    score = (
        1000.0 * mastered
        + 500.0 * avg_success
        + 50.0 * avg_returns
        + 8.0 * avg_strong_bounces
        + 20.0 * avg_recovery_rewards
        + 120.0 * avg_opponent_avoid_rate
        - 1.5 * avg_home_distance
        - 100.0 * fail_rate
        - 120.0 * front_camp_rate
        - 80.0 * action_to_net_rate
        - 250.0 * side_gap
        + 10.0 * current_level
    )
    return float(score)
