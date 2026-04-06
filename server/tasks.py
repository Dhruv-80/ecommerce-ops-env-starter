from copy import deepcopy
from typing import Any, Dict


def task_catalog() -> Dict[str, Dict[str, Any]]:
    return {
        "task_1": {
            "name": "Refund Queue Processing",
            "difficulty": "easy",
            "max_steps": 10,
            "description": "Process 5 refund tickets using simple 30-day policy.",
        },
        "task_2": {
            "name": "Inventory Reconciliation",
            "difficulty": "medium",
            "max_steps": 15,
            "description": "Reconcile inventory and route 8 pending orders.",
        },
        "task_3": {
            "name": "Supplier Cancellation Crisis",
            "difficulty": "hard",
            "max_steps": 18,
            "description": "Resolve affected orders with tier-based compensation.",
        },
    }


TASK_BUNDLES: Dict[str, Dict[str, Any]] = {
    "task_1": {
        "task": task_catalog()["task_1"],
        "initial_state": {
            "orders": [],
            "inventory": [],
            "tickets": [
                {"ticket_id": f"T{i}", "order_id": f"O{i}", "customer_id": f"C{i}", "reason": "refund_request", "created_days_ago": days, "status": "OPEN"}
                for i, days in enumerate([5, 12, 18, 32, 41], start=1)
            ],
        },
        "ground_truth": {
            "ticket_decisions": {
                "T1": "process_refund",
                "T2": "process_refund",
                "T3": "process_refund",
                "T4": "reject_refund",
                "T5": "reject_refund",
            }
        },
        "max_steps": 10,
    },
    "task_2": {
        "task": task_catalog()["task_2"],
        "initial_state": {"orders": [], "inventory": [], "tickets": []},
        "ground_truth": {"inventory": {}, "routes": {}},
        "max_steps": 15,
    },
    "task_3": {
        "task": task_catalog()["task_3"],
        "initial_state": {"orders": [], "inventory": [], "tickets": []},
        "ground_truth": {"resolutions": {}, "tier_weights": {"standard": 1.0, "premium": 1.5, "loyalty": 2.0}},
        "max_steps": 18,
    },
}


def get_task_bundle(task_id: str) -> Dict[str, Any]:
    if task_id not in TASK_BUNDLES:
        raise ValueError(f"Unknown task_id: {task_id}")
    return deepcopy(TASK_BUNDLES[task_id])
