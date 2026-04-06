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
            "orders": [
                {
                    "order_id": f"O{i}",
                    "customer_id": f"C{i}",
                    "customer_tier": "standard",
                    "status": "DELIVERED",
                    "items": [{"sku": f"SKU-{i}", "quantity": 1, "status": "DELIVERED"}],
                }
                for i in range(1, 6)
            ],
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
        "initial_state": {
            "orders": [
                {"order_id": "O201", "customer_id": "C201", "customer_tier": "standard", "status": "PENDING", "items": [{"sku": "SKU-A", "quantity": 1}], "warehouse": None},
                {"order_id": "O202", "customer_id": "C202", "customer_tier": "premium", "status": "PENDING", "items": [{"sku": "SKU-B", "quantity": 1}], "warehouse": None},
                {"order_id": "O203", "customer_id": "C203", "customer_tier": "standard", "status": "PENDING", "items": [{"sku": "SKU-C", "quantity": 1}], "warehouse": None},
                {"order_id": "O204", "customer_id": "C204", "customer_tier": "loyalty", "status": "PENDING", "items": [{"sku": "SKU-A", "quantity": 1}], "warehouse": None},
                {"order_id": "O205", "customer_id": "C205", "customer_tier": "standard", "status": "PENDING", "items": [{"sku": "SKU-D", "quantity": 1}], "warehouse": None},
                {"order_id": "O206", "customer_id": "C206", "customer_tier": "premium", "status": "PENDING", "items": [{"sku": "SKU-E", "quantity": 1}], "warehouse": None},
                {"order_id": "O207", "customer_id": "C207", "customer_tier": "standard", "status": "PENDING", "items": [{"sku": "SKU-B", "quantity": 1}], "warehouse": None},
                {"order_id": "O208", "customer_id": "C208", "customer_tier": "loyalty", "status": "PENDING", "items": [{"sku": "SKU-C", "quantity": 1}], "warehouse": None},
            ],
            "inventory": [
                {"sku": "SKU-A", "warehouse": "W1", "quantity": 2},
                {"sku": "SKU-B", "warehouse": "W1", "quantity": 1},
                {"sku": "SKU-C", "warehouse": "W1", "quantity": 0},
                {"sku": "SKU-D", "warehouse": "W1", "quantity": 0},
                {"sku": "SKU-E", "warehouse": "W1", "quantity": 1},
                {"sku": "SKU-A", "warehouse": "W2", "quantity": 1},
                {"sku": "SKU-B", "warehouse": "W2", "quantity": 1},
                {"sku": "SKU-C", "warehouse": "W2", "quantity": 2},
                {"sku": "SKU-D", "warehouse": "W2", "quantity": 1},
                {"sku": "SKU-E", "warehouse": "W2", "quantity": 0},
            ],
            "tickets": [],
        },
        "ground_truth": {
            "inventory": {
                "SKU-A|W1": 2,
                "SKU-B|W1": 1,
                "SKU-C|W1": 0,
                "SKU-D|W1": 0,
                "SKU-E|W1": 1,
                "SKU-A|W2": 1,
                "SKU-B|W2": 1,
                "SKU-C|W2": 2,
                "SKU-D|W2": 1,
                "SKU-E|W2": 0,
            },
            "routes": {
                "O201": "W1",
                "O202": "W1",
                "O203": "W2",
                "O204": "W1",
                "O205": "W2",
                "O206": "W1",
                "O207": "W2",
                "O208": "W2",
            },
        },
        "max_steps": 15,
    },
    "task_3": {
        "task": task_catalog()["task_3"],
        "initial_state": {
            "orders": [
                {
                    "order_id": "O301",
                    "customer_id": "C301",
                    "customer_tier": "standard",
                    "status": "PENDING",
                    "items": [
                        {"sku": "SKU-X", "quantity": 1, "status": "CANCELLED"},
                        {"sku": "SKU-Y", "quantity": 1, "status": "PENDING"},
                    ],
                },
                {
                    "order_id": "O302",
                    "customer_id": "C302",
                    "customer_tier": "premium",
                    "status": "PENDING",
                    "items": [{"sku": "SKU-X", "quantity": 1, "status": "CANCELLED"}],
                },
                {
                    "order_id": "O303",
                    "customer_id": "C303",
                    "customer_tier": "loyalty",
                    "status": "PARTIALLY_SHIPPED",
                    "items": [
                        {"sku": "SKU-Z", "quantity": 1, "status": "SHIPPED"},
                        {"sku": "SKU-X", "quantity": 1, "status": "CANCELLED"},
                    ],
                },
                {
                    "order_id": "O304",
                    "customer_id": "C304",
                    "customer_tier": "standard",
                    "status": "PENDING",
                    "items": [{"sku": "SKU-Y", "quantity": 1, "status": "PENDING"}],
                },
                {
                    "order_id": "O305",
                    "customer_id": "C305",
                    "customer_tier": "premium",
                    "status": "PENDING",
                    "items": [{"sku": "SKU-W", "quantity": 1, "status": "PENDING"}],
                },
            ],
            "inventory": [
                {"sku": "SKU-X", "warehouse": "W1", "quantity": 0},
                {"sku": "SKU-X-SUB", "warehouse": "W1", "quantity": 5},
                {"sku": "SKU-Y", "warehouse": "W1", "quantity": 8},
                {"sku": "SKU-Z", "warehouse": "W1", "quantity": 4},
                {"sku": "SKU-W", "warehouse": "W1", "quantity": 3},
            ],
            "tickets": [],
        },
        "ground_truth": {
            "resolutions": {
                "O301": {"expected_action": "apply_substitute", "substitute_sku": "SKU-X-SUB", "compensation": "coupon_5"},
                "O302": {"expected_action": "cancel_order", "compensation": "coupon_15"},
                "O303": {"expected_action": "apply_substitute", "substitute_sku": "SKU-X-SUB", "compensation": "priority_support"},
            },
            "unaffected_orders": ["O304", "O305"],
            "tier_weights": {"standard": 1.0, "premium": 1.5, "loyalty": 2.0},
        },
        "max_steps": 18,
    },
}


def get_task_bundle(task_id: str) -> Dict[str, Any]:
    if task_id not in TASK_BUNDLES:
        raise ValueError(f"Unknown task_id: {task_id}")
    return deepcopy(TASK_BUNDLES[task_id])
