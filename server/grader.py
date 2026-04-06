from typing import Any, Dict


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def grade_task_1(state) -> Dict[str, Any]:
    gt = state.ground_truth.get("ticket_decisions", {})
    correct = 0
    total = len(gt)
    for ticket in state.tickets:
        expected = gt.get(ticket.ticket_id)
        actual = ticket.status
        if expected == "process_refund" and actual == "REFUNDED":
            correct += 1
        elif expected == "reject_refund" and actual == "REJECTED":
            correct += 1
    score = correct / total if total else 0.0
    return {"score": _clamp01(score), "breakdown": {"correct": correct, "total": total}}


def grade_task_2(state) -> Dict[str, Any]:
    gt_inventory = state.ground_truth.get("inventory", {})
    gt_routes = state.ground_truth.get("routes", {})

    actual_inventory = {f"{row.sku}|{row.warehouse}": row.quantity for row in state.inventory}
    inventory_total = len(gt_inventory)
    inventory_correct = sum(1 for key, expected_qty in gt_inventory.items() if actual_inventory.get(key) == expected_qty)
    inventory_accuracy = (inventory_correct / inventory_total) if inventory_total else 0.0

    actual_routes = {order.order_id: order.warehouse for order in state.orders}
    route_total = len(gt_routes)
    route_correct = sum(1 for order_id, expected_wh in gt_routes.items() if actual_routes.get(order_id) == expected_wh)
    routing_accuracy = (route_correct / route_total) if route_total else 0.0

    score = (inventory_accuracy + routing_accuracy) / 2.0
    return {
        "score": _clamp01(score),
        "breakdown": {
            "inventory_accuracy": round(inventory_accuracy, 4),
            "routing_accuracy": round(routing_accuracy, 4),
            "inventory_correct": inventory_correct,
            "inventory_total": inventory_total,
            "routes_correct": route_correct,
            "routes_total": route_total,
        },
    }


def _is_substitution_applied(order, expected_substitute: str) -> bool:
    for item in getattr(order, "items", []):
        substitute = getattr(item, "substitute_sku", None)
        if substitute == expected_substitute:
            return True
    return False


def _order_resolution_match(order, expected: Dict[str, Any]) -> float:
    expected_action = expected.get("expected_action")
    expected_compensation = expected.get("compensation")
    expected_substitute = expected.get("substitute_sku")

    resolution_ok = False
    if expected_action == "cancel_order":
        resolution_ok = getattr(order, "status", None) == "CANCELLED"
    elif expected_action == "apply_substitute":
        resolution_ok = _is_substitution_applied(order, expected_substitute)

    compensation_ok = True
    if expected_compensation is not None:
        compensation_ok = expected_compensation in getattr(order, "compensation", [])

    if resolution_ok and compensation_ok:
        return 1.0
    if resolution_ok:
        return 0.7
    if compensation_ok and expected_compensation is not None:
        return 0.3
    return 0.0


def grade_task_3(state) -> Dict[str, Any]:
    gt_resolutions = state.ground_truth.get("resolutions", {})
    tier_weights = state.ground_truth.get("tier_weights", {"standard": 1.0, "premium": 1.5, "loyalty": 2.0})
    unaffected_orders = set(state.ground_truth.get("unaffected_orders", []))

    orders_by_id = {order.order_id: order for order in state.orders}
    weighted_total = 0.0
    weighted_correct = 0.0

    per_order_scores: Dict[str, float] = {}
    for order_id, expected in gt_resolutions.items():
        order = orders_by_id.get(order_id)
        if order is None:
            per_order_scores[order_id] = 0.0
            continue
        weight = tier_weights.get(getattr(order, "customer_tier", "standard"), 1.0)
        correctness = _order_resolution_match(order, expected)
        weighted_total += weight
        weighted_correct += correctness * weight
        per_order_scores[order_id] = round(correctness, 4)

    weighted_resolution = (weighted_correct / weighted_total) if weighted_total else 0.0

    collateral_count = 0
    for order_id in unaffected_orders:
        order = orders_by_id.get(order_id)
        if order is not None and getattr(order, "touched", False):
            collateral_count += 1
    collateral_count += len(getattr(state, "collateral_damage", []))

    normalization = max(len(unaffected_orders), 1)
    collateral_penalty = min(collateral_count / normalization, 1.0) * 0.35
    score = _clamp01(weighted_resolution - collateral_penalty)

    return {
        "score": score,
        "breakdown": {
            "weighted_resolution": round(weighted_resolution, 4),
            "collateral_damage": collateral_count,
            "collateral_penalty": round(collateral_penalty, 4),
            "per_order_scores": per_order_scores,
        },
    }


def grade_episode(task_id: str, state) -> Dict[str, Any]:
    if task_id == "task_1":
        return grade_task_1(state)
    if task_id == "task_2":
        return grade_task_2(state)
    if task_id == "task_3":
        return grade_task_3(state)
    raise ValueError(f"Unknown task_id: {task_id}")
