from types import SimpleNamespace
import pytest

from server.grader import grade_episode, grade_task_1, grade_task_2, grade_task_3
from server.reward import compute_step_reward
from server.tasks import get_task_bundle


def test_task_1_full_credit_when_ticket_statuses_match_ground_truth():
    state = SimpleNamespace(
        ground_truth={
            "ticket_decisions": {
                "T1": "process_refund",
                "T2": "reject_refund",
            }
        },
        tickets=[
            SimpleNamespace(ticket_id="T1", status="REFUNDED"),
            SimpleNamespace(ticket_id="T2", status="REJECTED"),
        ],
    )
    result = grade_task_1(state)
    assert result["score"] == 1.0


def test_task_1_partial_credit():
    state = SimpleNamespace(
        ground_truth={"ticket_decisions": {"T1": "process_refund", "T2": "reject_refund"}},
        tickets=[
            SimpleNamespace(ticket_id="T1", status="REFUNDED"),
            SimpleNamespace(ticket_id="T2", status="REFUNDED"),
        ],
    )
    result = grade_task_1(state)
    assert result["score"] == 0.5


def test_task_2_full_credit_when_inventory_and_routes_match():
    state = SimpleNamespace(
        ground_truth={
            "inventory": {"SKU-A|W1": 2, "SKU-B|W2": 1},
            "routes": {"O1": "W1", "O2": "W2"},
        },
        inventory=[
            SimpleNamespace(sku="SKU-A", warehouse="W1", quantity=2),
            SimpleNamespace(sku="SKU-B", warehouse="W2", quantity=1),
        ],
        orders=[
            SimpleNamespace(order_id="O1", warehouse="W1"),
            SimpleNamespace(order_id="O2", warehouse="W2"),
        ],
    )
    result = grade_task_2(state)
    assert result["score"] == 1.0
    assert result["breakdown"]["inventory_accuracy"] == 1.0
    assert result["breakdown"]["routing_accuracy"] == 1.0


def test_task_2_partial_credit_with_mixed_mistakes():
    state = SimpleNamespace(
        ground_truth={
            "inventory": {"SKU-A|W1": 2, "SKU-B|W2": 1},
            "routes": {"O1": "W1", "O2": "W2"},
        },
        inventory=[
            SimpleNamespace(sku="SKU-A", warehouse="W1", quantity=0),
            SimpleNamespace(sku="SKU-B", warehouse="W2", quantity=1),
        ],
        orders=[
            SimpleNamespace(order_id="O1", warehouse="W2"),
            SimpleNamespace(order_id="O2", warehouse="W2"),
        ],
    )
    result = grade_task_2(state)
    assert 0.0 < result["score"] < 1.0


def test_task_3_collateral_damage_penalized():
    state = SimpleNamespace(
        ground_truth={
            "resolutions": {
                "O301": {"expected_action": "apply_substitute", "substitute_sku": "SKU-X-SUB", "compensation": "coupon_5"},
                "O302": {"expected_action": "cancel_order", "compensation": "coupon_15"},
            },
            "unaffected_orders": ["O999"],
            "tier_weights": {"standard": 1.0, "premium": 1.5, "loyalty": 2.0},
        },
        orders=[
            SimpleNamespace(
                order_id="O301",
                customer_tier="standard",
                status="PENDING",
                items=[SimpleNamespace(substitute_sku="SKU-X-SUB")],
                compensation=["coupon_5"],
                touched=True,
            ),
            SimpleNamespace(
                order_id="O302",
                customer_tier="premium",
                status="CANCELLED",
                items=[],
                compensation=["coupon_15"],
                touched=True,
            ),
            SimpleNamespace(
                order_id="O999",
                customer_tier="standard",
                status="PENDING",
                items=[],
                compensation=[],
                touched=True,
            ),
        ],
        collateral_damage=["O998"],
    )
    result = grade_task_3(state)
    assert result["breakdown"]["weighted_resolution"] == 1.0
    assert result["score"] < 1.0


def test_reward_penalizes_wrong_entity_and_duplicate_action():
    outcome = {
        "valid_action": True,
        "correct_target": False,
        "state_matches_ground_truth": False,
        "collateral_damage": False,
        "within_budget": True,
        "wrong_entity": True,
        "repeat_action": True,
    }
    result = compute_step_reward(action={"action_type": "process_refund"}, outcome=outcome, ground_truth={})
    assert result["reward"] <= 0.0


def test_reward_full_credit_path_positive():
    outcome = {
        "valid_action": True,
        "correct_target": True,
        "state_matches_ground_truth": True,
        "collateral_damage": False,
        "within_budget": True,
    }
    result = compute_step_reward(action={"action_type": "route_order"}, outcome=outcome, ground_truth={})
    assert result["reward"] == 0.85


def test_task_bundle_contract_shape_for_all_tasks():
    for task_id in ("task_1", "task_2", "task_3"):
        bundle = get_task_bundle(task_id)
        assert set(bundle.keys()) == {"task", "initial_state", "ground_truth", "max_steps"}


def test_task_bundle_is_deep_copy_and_deterministic():
    first = get_task_bundle("task_1")
    second = get_task_bundle("task_1")
    first["initial_state"]["tickets"][0]["status"] = "MUTATED"
    assert second["initial_state"]["tickets"][0]["status"] == "OPEN"


def test_task_bundle_unknown_task_raises():
    with pytest.raises(ValueError, match="Unknown task_id"):
        get_task_bundle("task_999")


def test_grade_episode_dispatches_for_task_1():
    state = SimpleNamespace(
        ground_truth={"ticket_decisions": {"T1": "process_refund"}},
        tickets=[SimpleNamespace(ticket_id="T1", status="REFUNDED")],
    )
    result = grade_episode("task_1", state)
    assert result["score"] == 1.0


def test_grade_episode_unknown_task_raises():
    with pytest.raises(ValueError, match="Unknown task_id"):
        grade_episode("unknown", SimpleNamespace())


def test_task_3_exact_weighted_score_without_collateral_damage():
    # Weights: standard=1.0, premium=1.5, loyalty=2.0 => total=4.5
    # Per-order correctness: O301=1.0, O302=0.7, O303=0.3
    # Weighted sum = 1.0*1.0 + 0.7*1.5 + 0.3*2.0 = 2.65
    # Final weighted_resolution = 2.65 / 4.5 = 0.5889
    state = SimpleNamespace(
        ground_truth={
            "resolutions": {
                "O301": {"expected_action": "apply_substitute", "substitute_sku": "SKU-X-SUB", "compensation": "coupon_5"},
                "O302": {"expected_action": "cancel_order", "compensation": "coupon_15"},
                "O303": {"expected_action": "apply_substitute", "substitute_sku": "SKU-X-SUB", "compensation": "priority_support"},
            },
            "unaffected_orders": ["O304", "O305"],
            "tier_weights": {"standard": 1.0, "premium": 1.5, "loyalty": 2.0},
        },
        orders=[
            SimpleNamespace(
                order_id="O301",
                customer_tier="standard",
                status="PENDING",
                items=[SimpleNamespace(substitute_sku="SKU-X-SUB")],
                compensation=["coupon_5"],
                touched=True,
            ),
            SimpleNamespace(
                order_id="O302",
                customer_tier="premium",
                status="CANCELLED",
                items=[],
                compensation=[],
                touched=True,
            ),
            SimpleNamespace(
                order_id="O303",
                customer_tier="loyalty",
                status="PENDING",
                items=[],
                compensation=["priority_support"],
                touched=True,
            ),
            SimpleNamespace(order_id="O304", customer_tier="standard", status="PENDING", items=[], compensation=[], touched=False),
            SimpleNamespace(order_id="O305", customer_tier="premium", status="PENDING", items=[], compensation=[], touched=False),
        ],
        collateral_damage=[],
    )
    result = grade_task_3(state)
    assert result["breakdown"]["weighted_resolution"] == 0.5889
    assert result["score"] == 0.5888888888888889


def test_reward_contract_shape_stable():
    outcome = {
        "valid_action": True,
        "correct_target": True,
        "state_matches_ground_truth": False,
        "collateral_damage": False,
        "within_budget": True,
    }
    result = compute_step_reward(action={"action_type": "inspect_order"}, outcome=outcome, ground_truth={})
    assert set(result.keys()) == {"reward", "breakdown", "error"}
    assert set(result["breakdown"].keys()) == {
        "action_type",
        "target_entity",
        "final_state",
        "collateral_damage",
        "efficiency",
        "penalties",
    }


def test_reward_wrong_entity_penalty_path():
    outcome = {
        "valid_action": True,
        "correct_target": False,
        "state_matches_ground_truth": False,
        "collateral_damage": False,
        "within_budget": True,
        "wrong_entity": True,
    }
    result = compute_step_reward(action={"action_type": "route_order"}, outcome=outcome, ground_truth={})
    assert result["reward"] == 0.10


def test_reward_duplicate_action_penalty_path():
    outcome = {
        "valid_action": True,
        "correct_target": True,
        "state_matches_ground_truth": False,
        "collateral_damage": False,
        "within_budget": True,
        "repeat_action": True,
    }
    result = compute_step_reward(action={"action_type": "route_order"}, outcome=outcome, ground_truth={})
    assert result["reward"] == 0.50


def test_reward_collateral_damage_removes_bonus():
    clean_outcome = {
        "valid_action": True,
        "correct_target": True,
        "state_matches_ground_truth": True,
        "collateral_damage": False,
        "within_budget": True,
    }
    damaged_outcome = {
        "valid_action": True,
        "correct_target": True,
        "state_matches_ground_truth": True,
        "collateral_damage": True,
        "within_budget": True,
    }
    clean = compute_step_reward(action={"action_type": "cancel_order"}, outcome=clean_outcome, ground_truth={})
    damaged = compute_step_reward(action={"action_type": "cancel_order"}, outcome=damaged_outcome, ground_truth={})
    assert clean["reward"] - damaged["reward"] == 0.15


def test_reward_unnecessary_escalation_and_destructive_cancel_penalties():
    outcome = {
        "valid_action": True,
        "correct_target": True,
        "state_matches_ground_truth": False,
        "collateral_damage": False,
        "within_budget": True,
        "unnecessary_escalation": True,
        "destructive_cancel": True,
    }
    result = compute_step_reward(action={"action_type": "cancel_order"}, outcome=outcome, ground_truth={})
    # Base 0.60 minus 0.15 minus 0.25
    assert result["reward"] == 0.20
