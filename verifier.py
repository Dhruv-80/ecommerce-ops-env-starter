"""Verifier for CommerceOps-Env.

The verifier inspects the current ``EnvState`` (or a plain ``dict``
snapshot) and ground-truth payload to produce structured correctness
signals. These signals feed ``reward.py`` *per step* and also drive the
*final episode grade* shown in the /grader endpoint.

Design rules (from context.md):
- Success is checked from environment state and business rules, NOT from
  model explanations.
- Signals must be hard checks — no LLM judge, no "looks good" heuristics.
- Partial credit is allowed in T2/T3 so reward stays smooth enough for RL.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from models import ActionType, TIER_WEIGHT, OrderStatus


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _clamp(x: float, lo: float = 0.01, hi: float = 0.99) -> float:
    """Clamp to (lo, hi) so we never give exactly 0 or 1 (avoids dead RL)."""
    return max(lo, min(hi, x))


def _order_by_id(orders: List[Any], order_id: str) -> Optional[Any]:
    for o in orders:
        if _get(o, "order_id") == order_id:
            return o
    return None


def _get(obj: Any, key: str, default: Any = None) -> Any:
    """Uniform attribute/dict access so verifier works on both dataclasses
    (``EnvState.orders`` which contain ``Order`` instances) and plain dicts
    (used in tests)."""
    if isinstance(obj, dict):
        return obj.get(key, default)
    return getattr(obj, key, default)


def _stock_qty(stock: List[Any], sku: str, warehouse_id: str) -> int:
    for cell in stock:
        if _get(cell, "sku") == sku and _get(cell, "warehouse_id") == warehouse_id:
            return int(_get(cell, "quantity", 0))
    return 0


# ---------------------------------------------------------------------------
# Step-level action signals (fed to reward.py after every step)
# ---------------------------------------------------------------------------


def verify_step(
    *,
    action: Dict[str, Any],
    state_snapshot: Dict[str, Any],
    ground_truth: Dict[str, Any],
) -> Dict[str, Any]:
    """Return a flat signal dict consumed by reward.py.

    Every key is a bool or float so reward.py stays a thin weighted combiner.
    ``valid_action`` is already guaranteed True by the environment before
    this is called (schema rejection happens first), so we check only
    *business-rule correctness* here.
    """
    action_type = action.get("action_type", "")
    order_id = action.get("order_id")
    warehouse_id = action.get("warehouse_id")
    gt_kind = ground_truth.get("kind", "")

    signals: Dict[str, Any] = {
        "correct_entity": False,
        "correct_action_for_entity": False,
        "state_update_correct": False,
        "collateral_damage": False,
        "is_repeat": bool(action.get("_is_repeat", False)),
        "within_budget": bool(action.get("_within_budget", True)),
        "error": None,
    }

    if action_type == ActionType.NOOP.value:
        # A noop is schema-valid but conveys no useful judgment.
        # Give no positive signal and let the efficiency penalty apply.
        return signals

    orders = state_snapshot.get("orders", [])
    order = _order_by_id(orders, order_id) if order_id else None

    if order_id and order is None:
        signals["error"] = "order_not_found"
        return signals

    signals["correct_entity"] = True

    if gt_kind == "warehouse_assignment":
        signals.update(_step_verify_t1(action_type, warehouse_id, ground_truth))
    elif gt_kind == "multi_order_triage":
        signals.update(_step_verify_t2(action_type, order_id, warehouse_id, action, ground_truth, state_snapshot))
    elif gt_kind == "cascade_recovery_stub":
        signals.update(_step_verify_t3(action_type, order_id, warehouse_id, action, ground_truth))
    else:
        # Unknown ground-truth kind — give no bonus, no penalty.
        signals["correct_action_for_entity"] = True
        signals["state_update_correct"] = True

    return signals


def _step_verify_t1(
    action_type: str,
    warehouse_id: Optional[str],
    gt: Dict[str, Any],
) -> Dict[str, Any]:
    best = gt.get("best_warehouse")
    valid = set(gt.get("valid_warehouses", []))

    if action_type != ActionType.ASSIGN_WAREHOUSE.value:
        return {"correct_action_for_entity": False, "state_update_correct": False}

    if warehouse_id not in valid:
        return {"correct_action_for_entity": False, "state_update_correct": False,
                "error": "warehouse_has_no_stock_or_wrong_method"}

    state_ok = warehouse_id == best
    return {
        "correct_action_for_entity": True,
        "state_update_correct": state_ok,
        # Partial credit: valid-but-not-best is better than invalid.
        "partial_credit": 0.5 if (not state_ok) else 1.0,
    }


def _step_verify_t2(
    action_type: str,
    order_id: Optional[str],
    warehouse_id: Optional[str],
    action: Dict[str, Any],
    gt: Dict[str, Any],
    state_snapshot: Dict[str, Any],
) -> Dict[str, Any]:
    plan = gt.get("plan", {})
    expected = plan.get(order_id)

    if expected is None:
        # Order not in plan means it wasn't assigned in the reference plan —
        # the agent acting on it is still allowed, just unexpected.
        return {"correct_action_for_entity": True, "state_update_correct": False,
                "error": "order_not_in_reference_plan"}

    exp_type = expected.get("action_type")
    correct_action = action_type == exp_type

    if not correct_action:
        # Check: delay when expected is delay, or assign when expected is assign.
        return {"correct_action_for_entity": False, "state_update_correct": False}

    if action_type == ActionType.DELAY_ORDER.value:
        return {"correct_action_for_entity": True, "state_update_correct": True}

    if action_type == ActionType.ASSIGN_WAREHOUSE.value:
        exp_wh = expected.get("warehouse_id")
        stock = state_snapshot.get("stock", [])
        orders = state_snapshot.get("orders", [])
        order = _order_by_id(orders, order_id)
        sku = _get(order, "sku") if order else None
        qty = _get(order, "quantity_requested", 1) if order else 1

        if exp_wh and warehouse_id != exp_wh:
            # Wrong warehouse: check at least if the chosen one has stock.
            if sku and _stock_qty(stock, sku, warehouse_id) >= qty:
                return {"correct_action_for_entity": True, "state_update_correct": False,
                        "partial_credit": 0.5}
            return {"correct_action_for_entity": False, "state_update_correct": False}

        return {"correct_action_for_entity": True, "state_update_correct": True}

    if action_type == ActionType.SPLIT_SHIPMENT.value:
        exp_allocs = expected.get("allocations", [])
        act_allocs = action.get("allocations", [])
        if not exp_allocs or not act_allocs:
            return {"correct_action_for_entity": True, "state_update_correct": False}
        # Allocation match: same warehouse set, quantities within +/-1.
        exp_map = {a["warehouse_id"]: a["quantity"] for a in exp_allocs}
        act_map = {a["warehouse_id"]: a["quantity"] for a in (act_allocs if isinstance(act_allocs[0], dict) else [a.dict() for a in act_allocs])}
        match = all(abs(act_map.get(wid, 0) - qty) <= 1 for wid, qty in exp_map.items())
        return {"correct_action_for_entity": True, "state_update_correct": match,
                "partial_credit": 0.8 if match else 0.4}

    return {"correct_action_for_entity": True, "state_update_correct": False}


def _step_verify_t3(
    action_type: str,
    order_id: Optional[str],
    warehouse_id: Optional[str],
    action: Dict[str, Any],
    gt: Dict[str, Any],
) -> Dict[str, Any]:
    expected_actions = gt.get("expected_actions", {})
    expected_sup = gt.get("expected_supplier_escalation")

    if action_type == ActionType.ESCALATE_SUPPLIER.value:
        correct = action.get("supplier_id") == expected_sup
        return {"correct_action_for_entity": True, "state_update_correct": correct}

    if order_id and order_id in expected_actions:
        exp = expected_actions[order_id]
        correct_action = action_type == exp.get("action_type")
        if not correct_action:
            return {"correct_action_for_entity": False, "state_update_correct": False}
        if action_type == ActionType.REROUTE_ORDER.value:
            correct_wh = warehouse_id == exp.get("warehouse_id")
            return {"correct_action_for_entity": True, "state_update_correct": correct_wh}
        if action_type == ActionType.REFUND_OR_COMPENSATE.value:
            correct_comp = action.get("compensation_type") == exp.get("compensation_type")
            return {"correct_action_for_entity": True, "state_update_correct": correct_comp}

    return {"correct_action_for_entity": True, "state_update_correct": False}


# ---------------------------------------------------------------------------
# Episode-level graders (called by /grader and env.final_score())
# ---------------------------------------------------------------------------


def grade_episode(task_id: str, state: Any) -> Dict[str, Any]:
    """Top-level dispatcher. ``state`` may be an ``EnvState`` or a dict."""
    gt = _get(state, "ground_truth", {})
    kind = gt.get("kind", "")

    if kind == "warehouse_assignment":
        return _grade_t1(state, gt)
    if kind == "multi_order_triage":
        return _grade_t2(state, gt)
    if kind == "cascade_recovery_stub":
        return _grade_t3(state, gt)

    # Fallback: task_id-based dispatch for backward compat.
    if task_id == "task_1":
        return _grade_t1(state, gt)
    if task_id == "task_2":
        return _grade_t2(state, gt)
    if task_id == "task_3":
        return _grade_t3(state, gt)

    return {"score": 0.01, "breakdown": {"error": f"unknown task_id: {task_id}"}}


# --- T1 ---


def _grade_t1(state: Any, gt: Dict[str, Any]) -> Dict[str, Any]:
    orders = _get(state, "orders", [])
    order_id = gt.get("order_id", "O1")
    best_wh = gt.get("best_warehouse")
    valid_whs = set(gt.get("valid_warehouses", []))

    order = _order_by_id(orders, order_id)
    if order is None:
        return {"score": 0.01, "breakdown": {"error": "order_not_found"}}

    status = _get(order, "status", "")
    assigned = _get(order, "assigned_warehouse")

    if status == OrderStatus.ASSIGNED.value and assigned:
        if assigned == best_wh:
            score = 1.0
            label = "best"
        elif assigned in valid_whs:
            score = 0.6
            label = "valid_not_best"
        else:
            score = 0.0
            label = "invalid_warehouse"
    else:
        score = 0.0
        label = "not_assigned"

    return {
        "score": _clamp(score),
        "breakdown": {
            "label": label,
            "assigned_warehouse": assigned,
            "best_warehouse": best_wh,
            "valid_warehouses": list(valid_whs),
        },
    }


# --- T2 ---


def _grade_t2(state: Any, gt: Dict[str, Any]) -> Dict[str, Any]:
    """Score = weighted service coverage vs optimal, with a collateral penalty.

    ``per_order_weights`` (tier-weight per order) come from the ground truth
    so that the grader and the task planner use exactly the same scale.
    ``optimal_service_score`` is the sum of weights of orders the reference
    plan assigns/splits (not delays), so achieved/optimal = 1.0 on perfect play.
    """
    orders = _get(state, "orders", [])
    plan = gt.get("plan", {})
    per_order_weights: Dict[str, float] = gt.get("per_order_weights", {})
    optimal_score = float(gt.get("optimal_service_score", 1.0))
    infeasible = set(gt.get("infeasible_orders", []))

    achieved = 0.0
    per_order: Dict[str, str] = {}
    collateral_count = 0

    for order in orders:
        oid = _get(order, "order_id")
        exp = plan.get(oid)
        if exp is None:
            continue

        weight = float(per_order_weights.get(oid, TIER_WEIGHT.get(
            _get(order, "customer_tier", "standard"), 1.0
        )))
        status = _get(order, "status", "")
        assigned_wh = _get(order, "assigned_warehouse")
        exp_type = exp.get("action_type")

        if oid in infeasible:
            if status == OrderStatus.DELAYED.value:
                per_order[oid] = "correct_delay"
                # No contribution to achieved (infeasible → not in optimal)
            elif status in (OrderStatus.ASSIGNED.value, OrderStatus.SPLIT.value):
                per_order[oid] = "incorrect_fulfil_infeasible"
                collateral_count += 1
            else:
                per_order[oid] = "unresolved"
            continue

        if exp_type == ActionType.DELAY_ORDER.value:
            # Expected delay — no contribution to optimal_score; partial credit
            # for correctly delaying keeps gradient non-zero.
            if status == OrderStatus.DELAYED.value:
                per_order[oid] = "correct_delay"
                achieved += weight * 0.2   # small positive signal, not counted in optimal
            elif status in (OrderStatus.ASSIGNED.value, OrderStatus.SPLIT.value):
                # Agent served an order we planned to delay. Risky; not penalised
                # unless another order starved (captured via collateral elsewhere).
                per_order[oid] = "fulfilled_planned_delay"
            else:
                per_order[oid] = "unresolved"
            continue

        if exp_type == ActionType.ASSIGN_WAREHOUSE.value:
            exp_wh = exp.get("warehouse_id")
            if status == OrderStatus.ASSIGNED.value:
                if assigned_wh == exp_wh:
                    per_order[oid] = "perfect"
                    achieved += weight
                else:
                    per_order[oid] = "wrong_warehouse"
                    achieved += weight * 0.5
            elif status == OrderStatus.SPLIT.value:
                per_order[oid] = "split_when_assign_expected"
                achieved += weight * 0.7
            elif status == OrderStatus.DELAYED.value:
                per_order[oid] = "delayed_when_feasible"
            else:
                per_order[oid] = "unresolved"
            continue

        if exp_type == ActionType.SPLIT_SHIPMENT.value:
            if status == OrderStatus.SPLIT.value:
                per_order[oid] = "correct_split"
                achieved += weight
            elif status == OrderStatus.ASSIGNED.value:
                per_order[oid] = "assigned_when_split_expected"
                achieved += weight * 0.5
            elif status == OrderStatus.DELAYED.value:
                per_order[oid] = "delayed_when_feasible"
            else:
                per_order[oid] = "unresolved"

    raw_score = (achieved / optimal_score) if optimal_score > 0 else 0.0
    collateral_penalty = min(collateral_count * 0.1, 0.3)
    final_score = max(0.0, raw_score - collateral_penalty)

    return {
        "score": _clamp(final_score),
        "breakdown": {
            "achieved": round(achieved, 4),
            "optimal_score": round(optimal_score, 4),
            "raw_coverage": round(raw_score, 4),
            "collateral_count": collateral_count,
            "collateral_penalty": round(collateral_penalty, 4),
            "per_order": per_order,
        },
    }


# --- T3 ---


def _grade_t3(state: Any, gt: Dict[str, Any]) -> Dict[str, Any]:
    """Minimal grader for the stretch task stub."""
    orders = _get(state, "orders", [])
    expected_actions = gt.get("expected_actions", {})
    expected_sup = gt.get("expected_supplier_escalation")

    correct = 0
    total = len(expected_actions) + (1 if expected_sup else 0)

    for oid, exp in expected_actions.items():
        order = _order_by_id(orders, oid)
        if order is None:
            continue
        status = _get(order, "status", "")
        exp_type = exp.get("action_type")
        if exp_type == ActionType.REROUTE_ORDER.value:
            exp_wh = exp.get("warehouse_id")
            if status == OrderStatus.ASSIGNED.value and _get(order, "assigned_warehouse") == exp_wh:
                correct += 1
        elif exp_type == ActionType.REFUND_OR_COMPENSATE.value:
            if status == OrderStatus.CANCELLED.value:
                correct += 1

    # Supplier escalation check comes from state policy_flags.
    escalated = _get(state, "policy_flags", {}).get("supplier_escalated")
    if expected_sup and escalated == expected_sup:
        correct += 1

    score = (correct / total) if total else 0.0
    return {
        "score": _clamp(score),
        "breakdown": {"correct": correct, "total": total},
    }


# ---------------------------------------------------------------------------
# Terminal-state check (used by env to decide episode_done early)
# ---------------------------------------------------------------------------


def is_task_resolved(state: Any) -> bool:
    """Return True if the episode reached a terminal correct state early.

    The environment uses this to terminate before max_steps is exhausted so
    the step penalty is minimised for efficient agents.
    """
    gt = _get(state, "ground_truth", {})
    kind = gt.get("kind", "")
    orders = _get(state, "orders", [])

    if kind == "warehouse_assignment":
        order_id = gt.get("order_id", "O1")
        order = _order_by_id(orders, order_id)
        return order is not None and _get(order, "status") == OrderStatus.ASSIGNED.value

    if kind == "multi_order_triage":
        plan = gt.get("plan", {})
        resolved_statuses = {
            OrderStatus.ASSIGNED.value,
            OrderStatus.SPLIT.value,
            OrderStatus.DELAYED.value,
            OrderStatus.CANCELLED.value,
        }
        for oid in plan:
            order = _order_by_id(orders, oid)
            if order is None:
                return False
            if _get(order, "status") not in resolved_statuses:
                return False
        return True

    if kind == "cascade_recovery_stub":
        expected_actions = gt.get("expected_actions", {})
        for oid in expected_actions:
            order = _order_by_id(orders, oid)
            if order is None:
                return False
            if _get(order, "status") == OrderStatus.PENDING.value:
                return False
        return True

    return False


__all__ = [
    "grade_episode",
    "is_task_resolved",
    "verify_step",
]
