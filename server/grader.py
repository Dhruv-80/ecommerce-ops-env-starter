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
    # IMPLEMENT: compare reconciled inventory and routes to ground truth
    return {"score": 0.0, "breakdown": {"inventory_accuracy": 0.0, "routing_accuracy": 0.0}}


def grade_task_3(state) -> Dict[str, Any]:
    # IMPLEMENT: weighted per-order score + collateral damage penalty
    return {"score": 0.0, "breakdown": {"weighted_resolution": 0.0, "collateral_damage": len(state.collateral_damage)}}


def grade_episode(task_id: str, state) -> Dict[str, Any]:
    if task_id == "task_1":
        return grade_task_1(state)
    if task_id == "task_2":
        return grade_task_2(state)
    if task_id == "task_3":
        return grade_task_3(state)
    raise ValueError(f"Unknown task_id: {task_id}")
