"""Tests for Tendulkar's deliverables: metrics.py and eval.py.

Covers:
- MetricsLogger: write / read / summary
- TrainingMetricsTracker: rolling window, to_lists, from_trainer_log
- compare_policies: table formatting
- eval.py oracle + random policies: correctness and score ordering
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

import pytest

# Ensure repo root is importable regardless of pytest cwd.
_ROOT = os.path.join(os.path.dirname(__file__), "..")
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


# ---------------------------------------------------------------------------
# MetricsLogger
# ---------------------------------------------------------------------------


class TestMetricsLogger:
    def test_log_and_summary(self):
        from train.metrics import EpisodeRecord, MetricsLogger, StepRecord

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MetricsLogger(run_id="run01", policy="oracle", log_dir=tmpdir)

            logger.log_step(StepRecord(
                run_id="run01", policy="oracle", task_id="task_1",
                seed=0, episode_idx=0, step=1, action_type="assign_warehouse",
                reward=0.95, cumulative_reward=0.95,
                is_invalid=False, is_repeat=False, error=None,
            ))
            logger.log_episode(EpisodeRecord(
                run_id="run01", policy="oracle", task_id="task_1",
                seed=0, episode_idx=0, score=0.99, total_reward=0.95,
                steps=1, invalid_actions=0, repeat_actions=0,
                breakdown={"label": "best"},
            ))

            s = logger.summary()
            assert s["mean_score"] == 0.99
            assert s["invalid_rate"] == 0.0
            assert s["n_episodes"] == 1

    def test_multiple_episodes_mean(self):
        from train.metrics import EpisodeRecord, MetricsLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MetricsLogger(run_id="r", policy="random", log_dir=tmpdir)
            scores = [0.2, 0.4, 0.6]
            for i, sc in enumerate(scores):
                logger.log_episode(EpisodeRecord(
                    run_id="r", policy="random", task_id="task_2",
                    seed=i, episode_idx=i, score=sc, total_reward=1.0,
                    steps=4, invalid_actions=0, repeat_actions=0, breakdown={},
                ))

            s = logger.summary(task_id="task_2")
            assert abs(s["mean_score"] - sum(scores) / len(scores)) < 1e-5

    def test_load_episodes_from_disk(self):
        from train.metrics import EpisodeRecord, MetricsLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MetricsLogger(run_id="r", policy="oracle", log_dir=tmpdir)
            for i in range(3):
                logger.log_episode(EpisodeRecord(
                    run_id="r", policy="oracle", task_id="task_1",
                    seed=i, episode_idx=i, score=0.99, total_reward=1.0,
                    steps=1, invalid_actions=0, repeat_actions=0, breakdown={},
                ))

            loaded = MetricsLogger.load_episodes(tmpdir)
            assert len(loaded) == 3
            assert all(e["score"] == 0.99 for e in loaded)

    def test_task_filter(self):
        from train.metrics import EpisodeRecord, MetricsLogger

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MetricsLogger(run_id="r", policy="oracle", log_dir=tmpdir)
            for task_id, sc in [("task_1", 0.99), ("task_2", 0.7)]:
                logger.log_episode(EpisodeRecord(
                    run_id="r", policy="oracle", task_id=task_id,
                    seed=0, episode_idx=0, score=sc, total_reward=1.0,
                    steps=1, invalid_actions=0, repeat_actions=0, breakdown={},
                ))

            s1 = logger.summary(task_id="task_1")
            s2 = logger.summary(task_id="task_2")
            assert s1["mean_score"] == 0.99
            assert s2["mean_score"] == 0.7

    def test_step_rewards_list(self):
        from train.metrics import MetricsLogger, StepRecord

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MetricsLogger(run_id="r", policy="p", log_dir=tmpdir)
            for i in range(5):
                logger.log_step(StepRecord(
                    run_id="r", policy="p", task_id="task_1",
                    seed=0, episode_idx=0, step=i + 1,
                    action_type="noop", reward=float(i) * 0.1,
                    cumulative_reward=0.0, is_invalid=False,
                    is_repeat=False, error=None,
                ))

            rewards = logger.step_rewards()
            assert len(rewards) == 5
            assert rewards == pytest.approx([0.0, 0.1, 0.2, 0.3, 0.4])


# ---------------------------------------------------------------------------
# TrainingMetricsTracker
# ---------------------------------------------------------------------------


class TestTrainingMetricsTracker:
    def test_rolling_window(self):
        from train.metrics import TrainingMetricsTracker

        t = TrainingMetricsTracker(window=5)
        for i in range(10):
            t.record(step=i, mean_reward=float(i) * 0.1)

        summary = t.latest_summary()
        # Last 5 rewards: 0.5, 0.6, 0.7, 0.8, 0.9 → mean 0.7
        assert "0.7000" in summary

    def test_to_lists(self):
        from train.metrics import TrainingMetricsTracker

        t = TrainingMetricsTracker()
        t.record(step=1, mean_reward=0.3)
        t.record(step=2, mean_reward=0.5)

        data = t.to_lists()
        assert data["steps"] == [1, 2]
        assert data["mean_reward"] == pytest.approx([0.3, 0.5])

    def test_save_and_load_json(self):
        from train.metrics import TrainingMetricsTracker

        t = TrainingMetricsTracker()
        t.record(step=10, mean_reward=0.42, pass_rate=0.6)

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        try:
            t.save_json(path)
            with open(path) as f:
                loaded = json.load(f)
            assert loaded["steps"] == [10]
            assert loaded["mean_reward"] == pytest.approx([0.42])
        finally:
            os.unlink(path)

    def test_from_trainer_log(self):
        from train.metrics import TrainingMetricsTracker

        fake_log = [
            {"step": 10, "reward": 0.1},
            {"step": 20, "reward": 0.3},
            {"step": 30, "reward": 0.5},
            {"no_step_key": True},   # should be skipped
        ]
        t = TrainingMetricsTracker.from_trainer_log(fake_log)
        data = t.to_lists()
        assert data["steps"] == [10, 20, 30]
        assert data["mean_reward"] == pytest.approx([0.1, 0.3, 0.5])


# ---------------------------------------------------------------------------
# compare_policies
# ---------------------------------------------------------------------------


class TestComparePolicies:
    def test_table_contains_policies(self):
        from train.metrics import compare_policies

        s1 = {"policy": "oracle", "mean_score": 0.99, "mean_total_reward": 5.7,
              "invalid_rate": 0.0, "mean_steps": 3.5}
        s2 = {"policy": "random", "mean_score": 0.41, "mean_total_reward": 2.9,
              "invalid_rate": 0.0, "mean_steps": 5.7}
        table = compare_policies(s1, s2)
        assert "oracle" in table
        assert "random" in table
        assert "mean_score" in table

    def test_custom_metrics(self):
        from train.metrics import compare_policies

        s1 = {"policy": "a", "mean_score": 0.9}
        s2 = {"policy": "b", "mean_score": 0.4}
        table = compare_policies(s1, s2, metrics=["mean_score"])
        assert "0.9" in table
        assert "0.4" in table


# ---------------------------------------------------------------------------
# eval.py policies
# ---------------------------------------------------------------------------


class TestEvalPolicies:
    def test_oracle_t1_perfect_score(self):
        """Oracle on T1 must score 0.99 (best warehouse chosen)."""
        import sys; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        from train.eval import oracle_policy, run_episode_with_policy

        rec = run_episode_with_policy(
            "task_1", seed=0,
            policy_fn=lambda env: oracle_policy(env),
            policy_name="oracle",
            run_id="test",
            episode_idx=0,
        )
        assert rec.score >= 0.95, f"Oracle T1 score too low: {rec.score}"
        assert rec.invalid_actions == 0

    @pytest.mark.parametrize("seed", [0, 1, 2, 3])
    def test_oracle_t2_score_high(self, seed):
        from train.eval import oracle_policy, run_episode_with_policy

        rec = run_episode_with_policy(
            "task_2", seed=seed,
            policy_fn=lambda env: oracle_policy(env),
            policy_name="oracle",
            run_id="test",
            episode_idx=0,
        )
        assert rec.score >= 0.95, f"Oracle T2 seed={seed} score={rec.score}"
        assert rec.invalid_actions == 0

    def test_oracle_beats_random_on_t2(self):
        """Oracle must outperform random on T2 averaged over 4 seeds."""
        import random as _random
        from train.eval import oracle_policy, random_policy, run_episode_with_policy

        oracle_scores, random_scores = [], []
        for seed in range(4):
            rng = _random.Random(seed)
            o = run_episode_with_policy("task_2", seed,
                lambda env: oracle_policy(env), "oracle", "test", seed)
            r = run_episode_with_policy("task_2", seed,
                lambda env, rng=rng: random_policy(env, rng), "random", "test", seed)
            oracle_scores.append(o.score)
            random_scores.append(r.score)

        assert sum(oracle_scores) / 4 > sum(random_scores) / 4, (
            f"oracle mean={sum(oracle_scores)/4:.3f}  "
            f"random mean={sum(random_scores)/4:.3f}"
        )

    def test_random_policy_no_crash(self):
        """Random policy must complete an episode without raising."""
        import random as _random
        from train.eval import random_policy, run_episode_with_policy

        rng = _random.Random(42)
        rec = run_episode_with_policy(
            "task_2", seed=0,
            policy_fn=lambda env, rng=rng: random_policy(env, rng),
            policy_name="random",
            run_id="test",
            episode_idx=0,
        )
        assert 0.0 < rec.score <= 1.0

    def test_episode_record_fields(self):
        from train.eval import oracle_policy, run_episode_with_policy

        rec = run_episode_with_policy(
            "task_1", seed=0,
            policy_fn=lambda env: oracle_policy(env),
            policy_name="oracle",
            run_id="test",
            episode_idx=0,
        )
        assert isinstance(rec.score, float)
        assert isinstance(rec.total_reward, float)
        assert isinstance(rec.steps, int) and rec.steps >= 1
        assert isinstance(rec.breakdown, dict)
