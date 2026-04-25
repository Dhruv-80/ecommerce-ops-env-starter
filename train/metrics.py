"""Metrics logger for CommerceOps-Env training and evaluation.

Owned by Tendulkar per context.md team split.

Records per-step and per-episode stats to an append-only JSONL file and
keeps an in-memory summary so the training notebook can query running
averages without re-reading the file.

Design goals:
- Zero external dependencies (only stdlib + json).
- Safe for concurrent writes (each line is a complete JSON object).
- Queryable offline by ``eval.py`` and the demo notebook.
"""

from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Data records
# ---------------------------------------------------------------------------


@dataclass
class StepRecord:
    """Logged after every env.step()."""

    run_id: str
    policy: str          # "oracle" | "random" | "grpo_pretrain" | "grpo_trained"
    task_id: str
    seed: int
    episode_idx: int
    step: int
    action_type: str
    reward: float
    cumulative_reward: float
    is_invalid: bool
    is_repeat: bool
    error: Optional[str]
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EpisodeRecord:
    """Logged at the end of every episode."""

    run_id: str
    policy: str
    task_id: str
    seed: int
    episode_idx: int
    score: float
    total_reward: float
    steps: int
    invalid_actions: int
    repeat_actions: int
    breakdown: Dict[str, Any]
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------


class MetricsLogger:
    """Append-only JSONL logger with in-memory running summaries.

    Usage::

        logger = MetricsLogger(run_id="grpo_run_01", policy="oracle",
                               log_dir="./train/logs")
        # … inside training loop …
        logger.log_step(StepRecord(...))
        logger.log_episode(EpisodeRecord(...))

        summary = logger.summary()  # dict with mean scores per task
    """

    STEP_FILE    = "steps.jsonl"
    EPISODE_FILE = "episodes.jsonl"

    def __init__(
        self,
        run_id: str,
        policy: str,
        log_dir: str = "./train/logs",
    ) -> None:
        self.run_id = run_id
        self.policy = policy
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self._step_path    = self.log_dir / self.STEP_FILE
        self._episode_path = self.log_dir / self.EPISODE_FILE

        # In-memory accumulators
        self._episodes: List[EpisodeRecord] = []
        self._steps:    List[StepRecord]    = []

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def log_step(self, record: StepRecord) -> None:
        self._steps.append(record)
        self._append_jsonl(self._step_path, record.to_dict())

    def log_episode(self, record: EpisodeRecord) -> None:
        self._episodes.append(record)
        self._append_jsonl(self._episode_path, record.to_dict())

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def summary(self, task_id: Optional[str] = None) -> Dict[str, Any]:
        """Return aggregate stats for this run_id + policy."""
        eps = [
            e for e in self._episodes
            if e.run_id == self.run_id and e.policy == self.policy
        ]
        if task_id:
            eps = [e for e in eps if e.task_id == task_id]

        if not eps:
            return {"run_id": self.run_id, "policy": self.policy,
                    "task_id": task_id, "n_episodes": 0}

        n = len(eps)
        return {
            "run_id": self.run_id,
            "policy": self.policy,
            "task_id": task_id or "all",
            "n_episodes": n,
            "mean_score":         round(sum(e.score          for e in eps) / n, 4),
            "mean_total_reward":  round(sum(e.total_reward   for e in eps) / n, 4),
            "mean_steps":         round(sum(e.steps          for e in eps) / n, 2),
            "invalid_rate":       round(
                sum(e.invalid_actions for e in eps) /
                max(sum(e.steps for e in eps), 1), 4
            ),
            "repeat_rate":        round(
                sum(e.repeat_actions for e in eps) /
                max(sum(e.steps for e in eps), 1), 4
            ),
            "score_std":          _std([e.score for e in eps]),
        }

    def task_summary_table(self) -> List[Dict[str, Any]]:
        """Return one summary dict per task."""
        tasks = sorted({e.task_id for e in self._episodes})
        return [self.summary(t) for t in tasks]

    def step_rewards(self, task_id: Optional[str] = None) -> List[float]:
        """Return ordered list of per-step rewards for plotting."""
        steps = [
            s for s in self._steps
            if s.run_id == self.run_id and s.policy == self.policy
        ]
        if task_id:
            steps = [s for s in steps if s.task_id == task_id]
        return [s.reward for s in steps]

    def episode_scores(self, task_id: Optional[str] = None) -> List[float]:
        eps = [
            e for e in self._episodes
            if e.run_id == self.run_id and e.policy == self.policy
        ]
        if task_id:
            eps = [e for e in eps if e.task_id == task_id]
        return [e.score for e in eps]

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _append_jsonl(path: Path, record: Dict[str, Any]) -> None:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    @classmethod
    def load_episodes(cls, log_dir: str = "./train/logs") -> List[Dict[str, Any]]:
        """Load all episodes from a log directory (for offline analysis)."""
        path = Path(log_dir) / cls.EPISODE_FILE
        if not path.exists():
            return []
        records = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return records

    @classmethod
    def load_steps(cls, log_dir: str = "./train/logs") -> List[Dict[str, Any]]:
        path = Path(log_dir) / cls.STEP_FILE
        if not path.exists():
            return []
        records = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
        return records


# ---------------------------------------------------------------------------
# GRPO training-loop integration
# ---------------------------------------------------------------------------


class TrainingMetricsTracker:
    """Light wrapper used inside the GRPO training loop.

    Tracks rolling-window average reward and pass-rate so the training
    notebook can print live stats without importing matplotlib.

    Usage::

        tracker = TrainingMetricsTracker(window=20)
        # After each GRPOTrainer step callback:
        tracker.record(step=step, mean_reward=batch_reward,
                       pass_rate=pass_rate, invalid_rate=invalid_rate)
        print(tracker.latest_summary())
    """

    def __init__(self, window: int = 20) -> None:
        self.window = window
        self._history: List[Dict[str, Any]] = []

    def record(
        self,
        *,
        step: int,
        mean_reward: float,
        pass_rate: float = 0.0,
        invalid_rate: float = 0.0,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        entry: Dict[str, Any] = {
            "step": step,
            "mean_reward": mean_reward,
            "pass_rate": pass_rate,
            "invalid_rate": invalid_rate,
            "ts": time.time(),
        }
        if extra:
            entry.update(extra)
        self._history.append(entry)

    def latest_summary(self) -> str:
        if not self._history:
            return "No data yet."
        recent = self._history[-self.window:]
        mean_r = sum(e["mean_reward"] for e in recent) / len(recent)
        mean_p = sum(e["pass_rate"]   for e in recent) / len(recent)
        mean_i = sum(e["invalid_rate"] for e in recent) / len(recent)
        step   = self._history[-1]["step"]
        return (
            f"step={step:4d}  "
            f"mean_reward(w{self.window})={mean_r:+.4f}  "
            f"pass_rate={mean_p:.3f}  "
            f"invalid_rate={mean_i:.3f}"
        )

    def to_lists(self) -> Dict[str, List[Any]]:
        """Return parallel lists for matplotlib."""
        if not self._history:
            return {"steps": [], "mean_reward": [], "pass_rate": [], "invalid_rate": []}
        return {
            "steps":        [e["step"]         for e in self._history],
            "mean_reward":  [e["mean_reward"]   for e in self._history],
            "pass_rate":    [e["pass_rate"]     for e in self._history],
            "invalid_rate": [e["invalid_rate"]  for e in self._history],
        }

    def save_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_lists(), f, indent=2)
        print(f"Training metrics saved to {path}")

    @classmethod
    def from_trainer_log(cls, log_history: List[Dict[str, Any]]) -> "TrainingMetricsTracker":
        """Build a tracker from GRPOTrainer.state.log_history."""
        tracker = cls()
        for entry in log_history:
            if "step" not in entry:
                continue
            tracker.record(
                step=int(entry["step"]),
                mean_reward=float(entry.get("reward", entry.get("rewards/mean", 0.0))),
                pass_rate=float(entry.get("pass_rate", 0.0)),
                invalid_rate=float(entry.get("invalid_rate", 0.0)),
            )
        return tracker


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _std(xs: List[float]) -> float:
    if len(xs) < 2:
        return 0.0
    mean = sum(xs) / len(xs)
    variance = sum((x - mean) ** 2 for x in xs) / (len(xs) - 1)
    return round(variance ** 0.5, 4)


def compare_policies(
    *summaries: Dict[str, Any],
    metrics: Optional[List[str]] = None,
) -> str:
    """Format a comparison table from ``MetricsLogger.summary()`` dicts.

    Example::

        print(compare_policies(
            logger_oracle.summary(),
            logger_random.summary(),
            logger_trained.summary(),
        ))
    """
    if metrics is None:
        metrics = ["mean_score", "mean_total_reward", "invalid_rate", "mean_steps"]

    policies = [s.get("policy", "?") for s in summaries]
    col_w = max(len(p) for p in policies) + 2

    header = f"{'Metric':<24}" + "".join(f"{p:>{col_w}}" for p in policies)
    sep    = "-" * len(header)
    lines  = [sep, header, sep]

    for metric in metrics:
        row = f"{metric:<24}"
        for s in summaries:
            val = s.get(metric, "n/a")
            row += f"{str(val):>{col_w}}"
        lines.append(row)

    lines.append(sep)
    return "\n".join(lines)


__all__ = [
    "EpisodeRecord",
    "MetricsLogger",
    "StepRecord",
    "TrainingMetricsTracker",
    "compare_policies",
]
