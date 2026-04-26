"""Episode generators for CommerceOps-Env.

- T1 (warehouse assignment) — single-order assignment.
- T2 (multi-order triage) — headline task: per-warehouse stock scarcity
  forces the agent to assign each order to its NEAR warehouse.
- T3 (cascade recovery) — supplier failure has zeroed stock at one
  warehouse; the agent must reroute the affected order.

Episodes are deterministic given ``(task_id, seed)``. The same call always
returns the same world so verifier and reward stay reproducible.
"""

from __future__ import annotations

import random
from copy import deepcopy
from typing import Any, Dict, List, Optional

from models import (
    ALLOWED_ACTIONS_BY_TASK,
    ActionType,
    CustomerTier,
    DistanceBucket,
    Order,
    OrderStatus,
    ShippingMethod,
    StockCell,
    TIER_WEIGHT,
    TaskType,
    Warehouse,
)


# ---------------------------------------------------------------------------
# Catalog
# ---------------------------------------------------------------------------


TASK_CATALOG: Dict[str, Dict[str, Any]] = {
    "task_1": {
        "task_type": TaskType.T1_WAREHOUSE_ASSIGNMENT.value,
        "name": "Warehouse Assignment",
        "difficulty": "easy",
        "max_steps": 4,
        "description": (
            "One incoming order. Pick the single best warehouse based on "
            "stock availability, distance, and shipping-method support."
        ),
    },
    "task_2": {
        "task_type": TaskType.T2_MULTI_ORDER_TRIAGE.value,
        "name": "Multi-Order Fulfillment Triage",
        "difficulty": "medium",
        "max_steps": 4,
        "description": (
            "Two orders, one SKU, two warehouses with one unit each. "
            "Pick the correct warehouse for each order based on its "
            "destination region (NEAR > MID). Wasting the only stocked "
            "warehouse on the wrong order leaves the other order unfilled."
        ),
    },
    "task_3": {
        "task_type": TaskType.T3_CASCADE_RECOVERY.value,
        "name": "Cascade Recovery",
        "difficulty": "medium",
        "max_steps": 4,
        "description": (
            "A supplier failure has zeroed out stock at one warehouse. "
            "The pending order must be rerouted to the alternative "
            "warehouse that still has stock."
        ),
    },
}


def task_catalog() -> Dict[str, Dict[str, Any]]:
    """Return the public task catalogue (safe to ship in a /tasks endpoint)."""
    return deepcopy(TASK_CATALOG)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def get_task_bundle(task_id: str, seed: int = 0) -> Dict[str, Any]:
    """Return a fully materialised episode bundle for ``task_id``.

    The bundle is a plain ``dict`` so the environment can deepcopy it freely
    and so it serialises cleanly for inspection.
    """
    if task_id not in TASK_CATALOG:
        raise ValueError(f"Unknown task_id: {task_id}")

    if task_id == "task_1":
        return _build_task_1(seed)
    if task_id == "task_2":
        return _build_task_2(seed)
    if task_id == "task_3":
        return _build_task_3(seed)
    raise ValueError(f"Unsupported task_id: {task_id}")  # pragma: no cover


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bundle_skeleton(task_id: str) -> Dict[str, Any]:
    meta = TASK_CATALOG[task_id]
    return {
        "task_id": task_id,
        "task_type": meta["task_type"],
        "task_description": meta["description"],
        "max_steps": int(meta["max_steps"]),
        "allowed_actions": [a.value for a in ALLOWED_ACTIONS_BY_TASK[meta["task_type"]]],
        "warehouses": [],
        "stock": [],
        "orders": [],
        "policy_flags": {},
        "ground_truth": {},
    }


def _wh(warehouse_id: str, region: str, methods: List[str]) -> Dict[str, Any]:
    return Warehouse(warehouse_id=warehouse_id, region=region, supports_methods=methods).to_dict()


def _stock(warehouse_id: str, sku: str, quantity: int) -> Dict[str, Any]:
    return StockCell(warehouse_id=warehouse_id, sku=sku, quantity=quantity).to_dict()


def _order(
    *,
    order_id: str,
    sku: str,
    quantity: int,
    tier: str,
    sla_hours: int,
    region: str,
    distance_buckets: Dict[str, str],
    method: str = ShippingMethod.STANDARD.value,
) -> Dict[str, Any]:
    return Order(
        order_id=order_id,
        customer_id=f"C-{order_id}",
        customer_tier=tier,
        sku=sku,
        quantity_requested=quantity,
        sla_hours_remaining=sla_hours,
        destination_region=region,
        distance_buckets=distance_buckets,
        required_method=method,
    ).to_dict()


_DISTANCE_RANK = {
    DistanceBucket.NEAR.value: 0,
    DistanceBucket.MID.value: 1,
    DistanceBucket.FAR.value: 2,
}


def _stock_qty(stock: List[Dict[str, Any]], sku: str, warehouse_id: str) -> int:
    for cell in stock:
        if cell["sku"] == sku and cell["warehouse_id"] == warehouse_id:
            return int(cell["quantity"])
    return 0


# ---------------------------------------------------------------------------
# Task 1: single-order warehouse assignment
# ---------------------------------------------------------------------------


# Per-seed scenarios, varying which warehouse wins so a model that always
# picks "W1" gets penalised some of the time but not always.
_T1_SCENARIOS: List[Dict[str, Any]] = [
    {
        # W1 near + has stock + supports standard -> W1 wins
        "order": {
            "order_id": "O1",
            "sku": "SKU-RED",
            "quantity": 1,
            "tier": CustomerTier.STANDARD.value,
            "sla_hours": 36,
            "region": "north",
            "method": ShippingMethod.STANDARD.value,
            "distance_buckets": {
                "W1": DistanceBucket.NEAR.value,
                "W2": DistanceBucket.MID.value,
                "W3": DistanceBucket.FAR.value,
            },
        },
        "warehouses": [
            ("W1", "north", [ShippingMethod.STANDARD.value, ShippingMethod.EXPRESS.value]),
            ("W2", "central", [ShippingMethod.STANDARD.value]),
            ("W3", "south", [ShippingMethod.STANDARD.value]),
        ],
        "stock": {"SKU-RED": {"W1": 3, "W2": 2, "W3": 5}},
    },
    {
        # W2 wins: W1 is closest but has zero stock
        "order": {
            "order_id": "O1",
            "sku": "SKU-BLUE",
            "quantity": 1,
            "tier": CustomerTier.PREMIUM.value,
            "sla_hours": 24,
            "region": "central",
            "method": ShippingMethod.EXPRESS.value,
            "distance_buckets": {
                "W1": DistanceBucket.NEAR.value,
                "W2": DistanceBucket.MID.value,
                "W3": DistanceBucket.FAR.value,
            },
        },
        "warehouses": [
            ("W1", "central", [ShippingMethod.STANDARD.value, ShippingMethod.EXPRESS.value]),
            ("W2", "central", [ShippingMethod.STANDARD.value, ShippingMethod.EXPRESS.value]),
            ("W3", "south", [ShippingMethod.STANDARD.value]),
        ],
        "stock": {"SKU-BLUE": {"W1": 0, "W2": 4, "W3": 6}},
    },
    {
        # W3 wins: W1/W2 don't support overnight
        "order": {
            "order_id": "O1",
            "sku": "SKU-GREEN",
            "quantity": 1,
            "tier": CustomerTier.LOYALTY.value,
            "sla_hours": 12,
            "region": "south",
            "method": ShippingMethod.OVERNIGHT.value,
            "distance_buckets": {
                "W1": DistanceBucket.MID.value,
                "W2": DistanceBucket.MID.value,
                "W3": DistanceBucket.FAR.value,
            },
        },
        "warehouses": [
            ("W1", "north", [ShippingMethod.STANDARD.value]),
            ("W2", "central", [ShippingMethod.STANDARD.value, ShippingMethod.EXPRESS.value]),
            ("W3", "south", [ShippingMethod.STANDARD.value, ShippingMethod.EXPRESS.value, ShippingMethod.OVERNIGHT.value]),
        ],
        "stock": {"SKU-GREEN": {"W1": 5, "W2": 5, "W3": 2}},
    },
    {
        # Tie on distance + method, breaks by stock-then-name -> W1 wins
        "order": {
            "order_id": "O1",
            "sku": "SKU-RED",
            "quantity": 2,
            "tier": CustomerTier.STANDARD.value,
            "sla_hours": 48,
            "region": "central",
            "method": ShippingMethod.STANDARD.value,
            "distance_buckets": {
                "W1": DistanceBucket.NEAR.value,
                "W2": DistanceBucket.NEAR.value,
                "W3": DistanceBucket.FAR.value,
            },
        },
        "warehouses": [
            ("W1", "central", [ShippingMethod.STANDARD.value]),
            ("W2", "central", [ShippingMethod.STANDARD.value]),
            ("W3", "south", [ShippingMethod.STANDARD.value]),
        ],
        "stock": {"SKU-RED": {"W1": 4, "W2": 2, "W3": 9}},
    },
]


def _build_task_1(seed: int) -> Dict[str, Any]:
    scenario = _T1_SCENARIOS[seed % len(_T1_SCENARIOS)]
    bundle = _bundle_skeleton("task_1")

    bundle["warehouses"] = [_wh(wid, region, methods) for (wid, region, methods) in scenario["warehouses"]]
    for sku, by_wh in scenario["stock"].items():
        for wid, qty in by_wh.items():
            bundle["stock"].append(_stock(wid, sku, qty))

    order_payload = scenario["order"]
    bundle["orders"].append(_order(**order_payload))

    best = _t1_best_warehouse(order_payload, bundle["warehouses"], bundle["stock"])
    valid = _t1_valid_warehouses(order_payload, bundle["warehouses"], bundle["stock"])

    bundle["ground_truth"] = {
        "kind": "warehouse_assignment",
        "order_id": order_payload["order_id"],
        "best_warehouse": best,
        "valid_warehouses": valid,
    }
    bundle["policy_flags"] = {"single_order": True, "scarcity": False}
    return bundle


def _t1_valid_warehouses(
    order: Dict[str, Any], warehouses: List[Dict[str, Any]], stock: List[Dict[str, Any]]
) -> List[str]:
    """Warehouses that have stock and support the required shipping method."""
    valid: List[str] = []
    for w in warehouses:
        if order["method"] not in w["supports_methods"]:
            continue
        if _stock_qty(stock, order["sku"], w["warehouse_id"]) < order["quantity"]:
            continue
        valid.append(w["warehouse_id"])
    return valid


def _t1_best_warehouse(
    order: Dict[str, Any], warehouses: List[Dict[str, Any]], stock: List[Dict[str, Any]]
) -> Optional[str]:
    """Closest valid warehouse, tie-broken by warehouse_id (deterministic)."""
    candidates = []
    for w in warehouses:
        wid = w["warehouse_id"]
        if order["method"] not in w["supports_methods"]:
            continue
        if _stock_qty(stock, order["sku"], wid) < order["quantity"]:
            continue
        bucket = order["distance_buckets"].get(wid, DistanceBucket.FAR.value)
        candidates.append((_DISTANCE_RANK.get(bucket, 99), wid))
    if not candidates:
        return None
    candidates.sort()
    return candidates[0][1]


# ---------------------------------------------------------------------------
# Task 2: multi-order triage (headline)
# ---------------------------------------------------------------------------


def _build_task_2(seed: int) -> Dict[str, Any]:
    """Multi-order triage: 2 orders, 1 SKU, 2 warehouses.

    Two orders compete for two warehouses with one unit of stock each.
    One order is destined for "north" (W1 is NEAR), the other for
    "central" (W2 is NEAR). Wasting the only stocked warehouse on the
    wrong order leaves the other order with no inventory to serve, so
    the correct play is to assign each order to its NEAR warehouse.

    Tier weights differ (LOYALTY > PREMIUM) so wrong assignments are
    not symmetric in the grader. The seed permutes which order_id
    carries each tier so the agent must read the order, not memorise.
    """
    rng = random.Random(seed)
    bundle = _bundle_skeleton("task_2")

    bundle["warehouses"] = [
        _wh("W1", "north",   [ShippingMethod.STANDARD.value]),
        _wh("W2", "central", [ShippingMethod.STANDARD.value]),
    ]

    # 1 unit of SKU-A at each warehouse. Demand == supply.
    bundle["stock"] = [
        _stock("W1", "SKU-A", 1),
        _stock("W2", "SKU-A", 1),
    ]

    order_ids = ["O1", "O2"]
    rng.shuffle(order_ids)
    o_north_id, o_central_id = order_ids[0], order_ids[1]

    bundle["orders"] = [
        _order(
            order_id=o_north_id,
            sku="SKU-A",
            quantity=1,
            tier=CustomerTier.LOYALTY.value,
            sla_hours=12,
            region="north",
            distance_buckets={
                "W1": DistanceBucket.NEAR.value,
                "W2": DistanceBucket.MID.value,
            },
            method=ShippingMethod.STANDARD.value,
        ),
        _order(
            order_id=o_central_id,
            sku="SKU-A",
            quantity=1,
            tier=CustomerTier.PREMIUM.value,
            sla_hours=24,
            region="central",
            distance_buckets={
                "W1": DistanceBucket.MID.value,
                "W2": DistanceBucket.NEAR.value,
            },
            method=ShippingMethod.STANDARD.value,
        ),
    ]

    plan: Dict[str, Dict[str, Any]] = {
        o_north_id:   {"action_type": ActionType.ASSIGN_WAREHOUSE.value, "warehouse_id": "W1", "quantity": 1},
        o_central_id: {"action_type": ActionType.ASSIGN_WAREHOUSE.value, "warehouse_id": "W2", "quantity": 1},
    }
    per_order_weights: Dict[str, float] = {
        o_north_id:   TIER_WEIGHT[CustomerTier.LOYALTY.value],   # 2.0
        o_central_id: TIER_WEIGHT[CustomerTier.PREMIUM.value],   # 1.5
    }
    optimal_score = sum(per_order_weights.values())  # 3.5

    bundle["ground_truth"] = {
        "kind": "multi_order_triage",
        "plan": plan,
        "per_order_weights": per_order_weights,
        # priority by tier weight: LOYALTY first, then PREMIUM
        "priority_order": [o_north_id, o_central_id],
        "optimal_service_score": round(optimal_score, 4),
        "min_acceptable_service_score": round(optimal_score * 0.6, 4),
        "infeasible_orders": [],
        "tier_weight": dict(TIER_WEIGHT),
    }
    bundle["policy_flags"] = {
        "scarcity": True,            # per-warehouse, not aggregate
        "demand_exceeds_supply": False,  # demand == supply
    }
    return bundle


# ---------------------------------------------------------------------------
# Task 3: cascade recovery
# ---------------------------------------------------------------------------


def _build_task_3(seed: int) -> Dict[str, Any]:
    """Cascade recovery: a single supplier-failure reroute.

    1 SKU, 2 warehouses, 1 active order. One warehouse has had a
    supplier failure (stock = 0); the other still has stock. The
    correct move is to reroute the order to the warehouse that still
    has stock.

    The seed swaps which warehouse failed so the agent must read the
    stock table; "always W2" is wrong on half of the seeds.
    """
    rng = random.Random(seed)
    bundle = _bundle_skeleton("task_3")

    # Pick which warehouse suffered the supplier failure (alternates per seed).
    failed_wh, healthy_wh = ("W1", "W2") if seed % 2 == 0 else ("W2", "W1")

    bundle["warehouses"] = [
        _wh("W1", "north",   [ShippingMethod.STANDARD.value]),
        _wh("W2", "central", [ShippingMethod.STANDARD.value]),
    ]
    bundle["stock"] = [
        _stock(failed_wh,  "SKU-A", 0),
        _stock(healthy_wh, "SKU-A", 2),
    ]

    # Distance buckets: the failed warehouse looks "nearer" so the agent
    # has to reason from stock, not from distance alone.
    if failed_wh == "W1":
        dist = {"W1": DistanceBucket.NEAR.value, "W2": DistanceBucket.MID.value}
    else:
        dist = {"W1": DistanceBucket.MID.value, "W2": DistanceBucket.NEAR.value}

    bundle["orders"] = [
        _order(
            order_id="O1",
            sku="SKU-A",
            quantity=1,
            tier=CustomerTier.PREMIUM.value,
            sla_hours=18,
            region="central",
            distance_buckets=dist,
            method=ShippingMethod.STANDARD.value,
        ),
    ]

    bundle["policy_flags"] = {
        "supplier_failure": True,
        "failed_warehouse": failed_wh,
        "failed_supplier_id": "SUP-A",
    }
    # Use the same "kind" the verifier already understands; the new build is
    # a clean single-decision reroute task, no escalation/refund required.
    bundle["ground_truth"] = {
        "kind": "cascade_recovery",
        "expected_actions": {
            "O1": {"action_type": ActionType.REROUTE_ORDER.value, "warehouse_id": healthy_wh},
        },
        "healthy_warehouse": healthy_wh,
        "failed_warehouse": failed_wh,
    }
    return bundle


__all__ = [
    "TASK_CATALOG",
    "get_task_bundle",
    "task_catalog",
]
