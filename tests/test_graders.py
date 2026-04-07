from types import SimpleNamespace

from server.grader import grade_task_1, grade_task_2, grade_task_3
from server.reward import compute_step_reward


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
