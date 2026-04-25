"""Smoke + correctness tests for the Round-2 CommerceOps-Env.

Covers:
- models: action schema validation and whitelist enforcement
- tasks: episode generation, ground-truth shape, T2 scarcity invariant
- verifier: T1/T2 grading correctness
- environment: reset/step/final_score, anti-hacking checks
- server: HTTP endpoint contract
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class TestEnvAction:
    def test_valid_assign(self):
        from models import EnvAction
        a = EnvAction(action_type="assign_warehouse", order_id="O1", warehouse_id="W1")
        assert a.action_type.value == "assign_warehouse"

    def test_assign_missing_warehouse(self):
        from models import EnvAction
        with pytest.raises(ValidationError):
            EnvAction(action_type="assign_warehouse", order_id="O1")

    def test_split_needs_two_legs(self):
        from models import EnvAction
        with pytest.raises(ValidationError):
            EnvAction(action_type="split_shipment", order_id="O1",
                      allocations=[{"warehouse_id": "W1", "quantity": 1}])

    def test_unknown_action_rejected(self):
        from models import EnvAction
        with pytest.raises(ValidationError):
            EnvAction(action_type="wipe_database")

    def test_extra_field_rejected(self):
        from models import EnvAction
        with pytest.raises(ValidationError):
            EnvAction(action_type="noop", injected_field="hack")

    def test_reason_max_length(self):
        from models import EnvAction
        with pytest.raises(ValidationError):
            EnvAction(action_type="delay_order", order_id="O1", reason="x" * 65)

    def test_valid_split(self):
        from models import EnvAction
        a = EnvAction(
            action_type="split_shipment", order_id="O1",
            allocations=[
                {"warehouse_id": "W1", "quantity": 2},
                {"warehouse_id": "W2", "quantity": 1},
            ],
        )
        assert len(a.allocations) == 2


# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------


class TestTasks:
    def test_catalog_keys(self):
        from tasks import task_catalog
        assert set(task_catalog().keys()) == {"task_1", "task_2", "task_3"}

    def test_unknown_task_raises(self):
        from tasks import get_task_bundle
        with pytest.raises(ValueError):
            get_task_bundle("task_99")

    @pytest.mark.parametrize("seed", [0, 1, 2, 3])
    def test_t1_best_warehouse_is_valid(self, seed):
        from tasks import get_task_bundle
        b = get_task_bundle("task_1", seed=seed)
        gt = b["ground_truth"]
        assert gt["best_warehouse"] in gt["valid_warehouses"]

    @pytest.mark.parametrize("seed", [0, 1, 2, 3])
    def test_t2_demand_exceeds_supply(self, seed):
        from tasks import get_task_bundle
        b = get_task_bundle("task_2", seed=seed)
        assert b["policy_flags"]["demand_exceeds_supply"] is True

    @pytest.mark.parametrize("seed", [0, 1, 2, 3])
    def test_t2_express_orders_never_delayed_by_plan(self, seed):
        """Express-only orders (requiring W2) must always be in the plan as
        assign_warehouse, never as delay_order, because stock is reserved for them."""
        from tasks import get_task_bundle
        from models import ActionType
        b = get_task_bundle("task_2", seed=seed)
        plan = b["ground_truth"]["plan"]
        for oid, p in plan.items():
            if p["action_type"] == ActionType.DELAY_ORDER.value:
                # The delayed order must not be an express-required one.
                order = next(o for o in b["orders"] if o["order_id"] == oid)
                assert order["required_method"] != "express", (
                    f"seed={seed}: express order {oid} was delayed"
                )

    def test_t2_optimal_score_matches_weights(self):
        from tasks import get_task_bundle
        from models import ActionType
        b = get_task_bundle("task_2", seed=0)
        gt = b["ground_truth"]
        plan, weights = gt["plan"], gt["per_order_weights"]
        computed = sum(
            weights.get(oid, 1.0)
            for oid, p in plan.items()
            if p["action_type"] != ActionType.DELAY_ORDER.value
        )
        assert abs(computed - gt["optimal_service_score"]) < 1e-6

    def test_t2_allowed_actions_include_delay(self):
        from tasks import get_task_bundle
        b = get_task_bundle("task_2", seed=0)
        assert "delay_order" in b["allowed_actions"]


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------


class TestVerifier:
    def _make_t1_state(self, assigned_wh: str):
        """Build a minimal state dict as if the env assigned a warehouse."""
        return {
            "ground_truth": {
                "kind": "warehouse_assignment",
                "order_id": "O1",
                "best_warehouse": "W1",
                "valid_warehouses": ["W1", "W2"],
            },
            "orders": [
                {
                    "order_id": "O1",
                    "status": "assigned",
                    "assigned_warehouse": assigned_wh,
                    "customer_tier": "standard",
                    "sku": "SKU-RED",
                    "quantity_requested": 1,
                    "sla_hours_remaining": 36,
                    "destination_region": "north",
                    "distance_buckets": {},
                    "required_method": "standard",
                    "allocations": [],
                    "prioritized": False,
                }
            ],
        }

    def test_t1_perfect(self):
        from verifier import grade_episode
        state = self._make_t1_state("W1")

        class FakeState:
            ground_truth = state["ground_truth"]
            orders = state["orders"]

        result = grade_episode("task_1", FakeState())
        assert result["score"] > 0.9
        assert result["breakdown"]["label"] == "best"

    def test_t1_valid_not_best(self):
        from verifier import grade_episode
        state = self._make_t1_state("W2")

        class FakeState:
            ground_truth = state["ground_truth"]
            orders = state["orders"]

        result = grade_episode("task_1", FakeState())
        assert 0.5 < result["score"] < 0.9
        assert result["breakdown"]["label"] == "valid_not_best"

    def test_t1_invalid_warehouse(self):
        from verifier import grade_episode
        state = self._make_t1_state("W3")  # W3 not in valid_warehouses

        class FakeState:
            ground_truth = state["ground_truth"]
            orders = state["orders"]

        result = grade_episode("task_1", FakeState())
        assert result["score"] == pytest.approx(0.01, abs=0.01)

    def test_t1_unresolved(self):
        from verifier import grade_episode

        class FakeState:
            ground_truth = {
                "kind": "warehouse_assignment",
                "order_id": "O1",
                "best_warehouse": "W1",
                "valid_warehouses": ["W1"],
            }
            orders = [{
                "order_id": "O1", "status": "pending",
                "assigned_warehouse": None,
                "customer_tier": "standard", "sku": "SKU-A",
                "quantity_requested": 1, "sla_hours_remaining": 24,
                "destination_region": "north", "distance_buckets": {},
                "required_method": "standard", "allocations": [],
                "prioritized": False,
            }]

        result = grade_episode("task_1", FakeState())
        assert result["score"] == pytest.approx(0.01, abs=0.01)

    def test_t2_perfect_play_near_max(self):
        """Following the reference plan for T2 should score ≥ 0.95."""
        from environment import CommerceOpsEnv
        env = CommerceOpsEnv()
        env.reset("task_2", seed=0)
        gt = env.state.ground_truth
        for oid, plan in gt["plan"].items():
            action = {"action_type": plan["action_type"], "order_id": oid}
            if plan["action_type"] == "assign_warehouse":
                action["warehouse_id"] = plan["warehouse_id"]
            elif plan["action_type"] == "split_shipment":
                action["allocations"] = plan["allocations"]
            env.step(action)
        score = env.final_score()
        assert score["score"] >= 0.95, f"Expected ≥0.95, got {score['score']}"

    def test_t2_delay_all_gives_low_score(self):
        from environment import CommerceOpsEnv
        env = CommerceOpsEnv()
        env.reset("task_2", seed=0)
        gt = env.state.ground_truth
        for oid in gt["plan"]:
            env.step({"action_type": "delay_order", "order_id": oid, "reason": "test"})
        score = env.final_score()
        assert score["score"] < 0.2, f"Expected <0.2, got {score['score']}"


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


class TestEnvironment:
    def test_reset_returns_observation(self):
        from environment import CommerceOpsEnv
        env = CommerceOpsEnv()
        obs = env.reset("task_1", seed=0)
        assert obs.step == 0
        assert obs.done is False
        assert len(obs.orders) == 1
        assert len(obs.warehouses) == 3

    def test_uninitialized_raises(self):
        from environment import CommerceOpsEnv
        env = CommerceOpsEnv()
        with pytest.raises(RuntimeError):
            _ = env.state

    def test_episode_done_after_correct_t1(self):
        from environment import CommerceOpsEnv
        env = CommerceOpsEnv()
        env.reset("task_1", seed=0)
        obs = env.step({"action_type": "assign_warehouse",
                        "order_id": "O1", "warehouse_id": "W1"})
        assert obs.done is True

    def test_episode_done_after_max_steps(self):
        from environment import CommerceOpsEnv
        env = CommerceOpsEnv()
        env.reset("task_1", seed=0)
        for _ in range(5):  # max_steps=4; exceed by 1
            env.step({"action_type": "noop"})
        assert env.state.episode_done is True

    def test_step_after_done_is_noop(self):
        from environment import CommerceOpsEnv
        env = CommerceOpsEnv()
        env.reset("task_1", seed=0)
        env.step({"action_type": "assign_warehouse", "order_id": "O1", "warehouse_id": "W1"})
        obs = env.step({"action_type": "noop"})
        assert "already finished" in obs.last_action_result.lower()
        assert obs.reward == 0.0

    def test_invalid_action_type_penalised(self):
        from environment import CommerceOpsEnv
        env = CommerceOpsEnv()
        env.reset("task_1", seed=0)
        # split_shipment is not in T1 allowed actions
        obs = env.step({
            "action_type": "split_shipment", "order_id": "O1",
            "allocations": [{"warehouse_id": "W1", "quantity": 1},
                            {"warehouse_id": "W2", "quantity": 1}],
        })
        assert obs.reward < 0
        assert env.state.invalid_action_count == 1

    def test_repeat_action_counted(self):
        from environment import CommerceOpsEnv
        env = CommerceOpsEnv()
        env.reset("task_2", seed=0)
        env.step({"action_type": "noop"})
        env.step({"action_type": "noop"})
        assert env.state.repeat_action_count >= 1

    def test_stock_consumed_on_assign(self):
        from environment import CommerceOpsEnv
        env = CommerceOpsEnv()
        env.reset("task_1", seed=0)  # seed=0: SKU-RED W1 has 3 units
        before = next(
            s.quantity for s in env.state.stock
            if s.sku == "SKU-RED" and s.warehouse_id == "W1"
        )
        env.step({"action_type": "assign_warehouse", "order_id": "O1", "warehouse_id": "W1"})
        after = next(
            (s.quantity for s in env.state.stock
             if s.sku == "SKU-RED" and s.warehouse_id == "W1"),
            None,
        )
        assert after == before - 1

    def test_assign_no_stock_gives_collateral(self):
        """seed=1: W1 has zero SKU-BLUE stock — assigning there is a collateral event."""
        from environment import CommerceOpsEnv
        env = CommerceOpsEnv()
        env.reset("task_1", seed=1)
        obs = env.step({"action_type": "assign_warehouse", "order_id": "O1", "warehouse_id": "W1"})
        # Verifier marks order not-found-in-valid -> low/no state_update reward
        assert obs.reward < 0.7  # can't be perfect

    def test_final_score_structure(self):
        from environment import CommerceOpsEnv
        env = CommerceOpsEnv()
        env.reset("task_1", seed=0)
        score = env.final_score()
        assert "score" in score
        assert "breakdown" in score
        assert "cumulative_reward" in score
        assert "steps" in score
        assert "invalid_actions" in score

    def test_reward_range_valid(self):
        """Per-step reward should stay within [-1, 1]."""
        from environment import CommerceOpsEnv
        env = CommerceOpsEnv()
        env.reset("task_2", seed=0)
        for _ in range(6):
            obs = env.step({"action_type": "noop"})
            assert -1.0 <= obs.reward <= 1.0, f"reward out of range: {obs.reward}"


# ---------------------------------------------------------------------------
# Server (HTTP contract)
# ---------------------------------------------------------------------------


@pytest.fixture
def client():
    from fastapi.testclient import TestClient
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from server.app import app
    return TestClient(app)


class TestServerAPI:
    def test_health(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["ok"] is True

    def test_reset_default(self, client):
        r = client.post("/reset", json={})
        assert r.status_code == 200
        body = r.json()
        assert body["task_id"] == "task_1"
        assert body["done"] is False
        assert isinstance(body["orders"], list)

    def test_reset_task2(self, client):
        r = client.post("/reset", json={"task_id": "task_2", "seed": 0})
        assert r.status_code == 200
        assert r.json()["task_id"] == "task_2"

    def test_step_valid(self, client):
        client.post("/reset", json={"task_id": "task_1", "seed": 0})
        r = client.post("/step", json={"action_type": "assign_warehouse",
                                       "order_id": "O1", "warehouse_id": "W1"})
        assert r.status_code == 200
        body = r.json()
        assert "reward" in body
        assert body["done"] is True

    def test_step_invalid_action(self, client):
        client.post("/reset", json={"task_id": "task_1", "seed": 0})
        r = client.post("/step", json={"action_type": "split_shipment",
                                       "order_id": "O1",
                                       "allocations": [
                                           {"warehouse_id": "W1", "quantity": 1},
                                           {"warehouse_id": "W2", "quantity": 1},
                                       ]})
        assert r.status_code == 200
        body = r.json()
        assert body["reward"] < 0

    def test_state_endpoint(self, client):
        client.post("/reset", json={"task_id": "task_1", "seed": 0})
        r = client.get("/state")
        assert r.status_code == 200
        assert "orders" in r.json()

    def test_tasks_endpoint(self, client):
        r = client.get("/tasks")
        assert r.status_code == 200
        assert "task_1" in r.json()
        assert "task_2" in r.json()

    def test_schema_endpoint(self, client):
        r = client.get("/schema")
        assert r.status_code == 200
        assert "action" in r.json()

    def test_grader_endpoint(self, client):
        client.post("/reset", json={"task_id": "task_1", "seed": 0})
        client.post("/step", json={"action_type": "assign_warehouse",
                                   "order_id": "O1", "warehouse_id": "W1"})
        r = client.post("/grader")
        assert r.status_code == 200
        body = r.json()
        assert "score" in body
        assert 0.0 < body["score"] <= 1.0

    def test_metadata_endpoint(self, client):
        r = client.get("/metadata")
        assert r.status_code == 200
        assert r.json()["version"] == "2.0.0"

    def test_baseline_endpoint(self, client):
        r = client.get("/baseline")
        assert r.status_code == 200
        body = r.json()
        assert "task_1" in body
        assert "task_2" in body
        assert body["task_1"]["score"] > 0
