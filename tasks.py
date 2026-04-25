"""Episode generators for CommerceOps-Env.

Per ``context.md`` build order:
- T1 (warehouse assignment) is the bootstrap task and must be easy enough
  for the model to occasionally win early in training.
- T2 (multi-order triage) is the headline task: limited stock, several
  orders competing, the agent must fulfil / split / delay under tier and
  SLA pressure.
- T3 is stretch and only stubbed.

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
        "name": "Warehouse Assignment Bootstrap",
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
        "max_steps": 12,
        "description": (
            "Several orders compete for limited stock across warehouses. "
            "Decide which orders to assign, split, delay or deprioritize "
            "while balancing customer tier and SLA pressure."
        ),
    },
    "task_3": {
        "task_type": TaskType.T3_CASCADE_RECOVERY.value,
        "name": "Cascade Recovery (Stretch)",
        "difficulty": "hard",
        "max_steps": 16,
        "description": (
            "A supplier or shipment failure occurs after earlier "
            "allocation decisions. Recover by rerouting, compensating, "
            "or escalating without causing downstream damage."
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
# Task 1: single-order warehouse assignment (bootstrap)
# ---------------------------------------------------------------------------


# A few hand-crafted seeds, varying which warehouse wins so a model that
# always picks "W1" gets penalised some of the time but not always.
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
    """Generate a triage scenario where stock is intentionally insufficient.

    The episode is parameterised by ``seed`` so we get reproducible variety.
    Concretely: ~6 orders, 2 SKUs, 2 warehouses, total stock < total demand,
    so the agent MUST delay or split some orders. Tier and SLA differ across
    orders so that delaying a loyalty/urgent order is the wrong call.
    """
    rng = random.Random(seed)
    bundle = _bundle_skeleton("task_2")

    # Two warehouses, both standard-capable; only W2 supports express so
    # express-required orders narrow down naturally.
    bundle["warehouses"] = [
        _wh("W1", "north", [ShippingMethod.STANDARD.value]),
        _wh("W2", "central", [ShippingMethod.STANDARD.value, ShippingMethod.EXPRESS.value]),
    ]

    # Constrained stock: total supply (5) < total demand (6) so at least one
    # order must be delayed. W2 always has both SKUs so the two
    # express-required orders (O1 SKU-A, O4 SKU-B) remain feasible.
    sku_a_w1 = rng.choice([1, 2])
    sku_a_w2 = max(1, 3 - sku_a_w1)
    sku_b_w1 = 1
    sku_b_w2 = 1
    bundle["stock"] = [
        _stock("W1", "SKU-A", sku_a_w1),
        _stock("W2", "SKU-A", sku_a_w2),
        _stock("W1", "SKU-B", sku_b_w1),
        _stock("W2", "SKU-B", sku_b_w2),
    ]

    # Six orders. Mix of tiers and urgencies; two require express -> forces W2.
    tiers = [
        CustomerTier.LOYALTY.value,
        CustomerTier.PREMIUM.value,
        CustomerTier.STANDARD.value,
        CustomerTier.PREMIUM.value,
        CustomerTier.STANDARD.value,
        CustomerTier.STANDARD.value,
    ]
    rng.shuffle(tiers)

    base_orders = [
        ("O1", "SKU-A", 1, 8,  "central", ShippingMethod.EXPRESS.value),
        ("O2", "SKU-A", 1, 24, "north",   ShippingMethod.STANDARD.value),
        ("O3", "SKU-A", 1, 36, "south",   ShippingMethod.STANDARD.value),
        ("O4", "SKU-B", 1, 16, "central", ShippingMethod.EXPRESS.value),
        ("O5", "SKU-B", 1, 48, "north",   ShippingMethod.STANDARD.value),
        ("O6", "SKU-B", 1, 30, "south",   ShippingMethod.STANDARD.value),
    ]
    distance_for_region = {
        "north":   {"W1": DistanceBucket.NEAR.value, "W2": DistanceBucket.MID.value},
        "central": {"W1": DistanceBucket.MID.value,  "W2": DistanceBucket.NEAR.value},
        "south":   {"W1": DistanceBucket.FAR.value,  "W2": DistanceBucket.MID.value},
    }
    for (oid, sku, qty, sla, region, method), tier in zip(base_orders, tiers):
        bundle["orders"].append(
            _order(
                order_id=oid,
                sku=sku,
                quantity=qty,
                tier=tier,
                sla_hours=sla,
                region=region,
                distance_buckets=distance_for_region[region],
                method=method,
            )
        )

    plan = _t2_plan(bundle["orders"], bundle["warehouses"], bundle["stock"])
    bundle["ground_truth"] = {
        "kind": "multi_order_triage",
        "plan": plan["per_order"],
        "per_order_weights": plan["per_order_weights"],
        "priority_order": plan["priority_order"],
        "optimal_service_score": plan["optimal_service_score"],
        "min_acceptable_service_score": round(plan["optimal_service_score"] * 0.6, 4),
        "infeasible_orders": plan["infeasible_orders"],
        "tier_weight": dict(TIER_WEIGHT),
    }
    bundle["policy_flags"] = {
        "scarcity": True,
        "demand_exceeds_supply": plan["demand_exceeds_supply"],
    }
    return bundle


def _t2_priority_score(order: Dict[str, Any]) -> float:
    """Higher means more important. Used for ranking, not for reward directly."""
    tier_weight = TIER_WEIGHT.get(order["customer_tier"], 1.0)
    sla = max(int(order["sla_hours_remaining"]), 1)
    urgency = 48.0 / sla
    return tier_weight * urgency


def _t2_plan(
    orders: List[Dict[str, Any]],
    warehouses: List[Dict[str, Any]],
    stock: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute a defensible ground-truth plan for a triage episode.

    Greedy by priority score: walk highest-priority orders first, give each
    its closest valid warehouse with stock; fall back to a 2-warehouse split
    if no single warehouse has the full quantity but combined warehouses do;
    otherwise mark the order as ``delay_order``. The verifier uses this both
    as a target and to compute service-score thresholds for partial credit.
    """
    remaining = {(c["sku"], c["warehouse_id"]): int(c["quantity"]) for c in stock}
    method_by_wh = {w["warehouse_id"]: set(w["supports_methods"]) for w in warehouses}

    def _eligible_count(o: Dict[str, Any]) -> int:
        return sum(1 for w in warehouses if o["required_method"] in method_by_wh[w["warehouse_id"]])

    # Reserve slots for orders with the fewest eligible warehouses first so a
    # premium express-only order isn't starved by greedy standard orders.
    ranked = sorted(orders, key=lambda o: (_eligible_count(o), -_t2_priority_score(o)))

    per_order: Dict[str, Dict[str, Any]] = {}
    optimal_score = 0.0
    infeasible: List[str] = []

    for order in ranked:
        wh_options = sorted(
            warehouses,
            key=lambda w: _DISTANCE_RANK.get(order["distance_buckets"].get(w["warehouse_id"], DistanceBucket.FAR.value), 99),
        )

        eligible = [w for w in wh_options if order["required_method"] in method_by_wh[w["warehouse_id"]]]
        chosen = None
        for w in eligible:
            wid = w["warehouse_id"]
            if remaining.get((order["sku"], wid), 0) >= order["quantity_requested"]:
                chosen = ("assign_warehouse", [(wid, order["quantity_requested"])])
                break

        if chosen is None and len(eligible) >= 2:
            need = order["quantity_requested"]
            legs: List[tuple] = []
            for w in eligible:
                wid = w["warehouse_id"]
                avail = remaining.get((order["sku"], wid), 0)
                if avail <= 0:
                    continue
                take = min(avail, need)
                if take > 0:
                    legs.append((wid, take))
                    need -= take
                if need == 0:
                    break
            if need == 0 and len(legs) >= 2:
                chosen = ("split_shipment", legs)

        if chosen is None:
            per_order[order["order_id"]] = {
                "action_type": ActionType.DELAY_ORDER.value,
                "reason": "stock_insufficient",
            }
            infeasible.append(order["order_id"])
            continue

        action_type, legs = chosen
        for wid, qty in legs:
            remaining[(order["sku"], wid)] = remaining.get((order["sku"], wid), 0) - qty

        if action_type == "assign_warehouse":
            per_order[order["order_id"]] = {
                "action_type": ActionType.ASSIGN_WAREHOUSE.value,
                "warehouse_id": legs[0][0],
                "quantity": legs[0][1],
            }
        else:
            per_order[order["order_id"]] = {
                "action_type": ActionType.SPLIT_SHIPMENT.value,
                "allocations": [{"warehouse_id": wid, "quantity": qty} for wid, qty in legs],
            }
        optimal_score += _t2_priority_score(order)

    total_demand = sum(o["quantity_requested"] for o in orders)
    total_supply = sum(int(c["quantity"]) for c in stock)

    # Per-order weight = tier_weight only (no urgency scaling).
    # Used by the grader so that achieved/optimal ratios are consistent.
    per_order_weights: Dict[str, float] = {}
    for o in orders:
        per_order_weights[o["order_id"]] = TIER_WEIGHT.get(o["customer_tier"], 1.0)

    # Optimal score = sum of tier weights of served (non-delayed) orders.
    optimal_tier_score = sum(
        per_order_weights[oid]
        for oid in per_order
        if per_order[oid]["action_type"] != ActionType.DELAY_ORDER.value
    )

    return {
        "per_order": per_order,
        "per_order_weights": per_order_weights,
        "priority_order": [o["order_id"] for o in ranked],
        "optimal_service_score": round(optimal_tier_score, 4),
        "infeasible_orders": infeasible,
        "demand_exceeds_supply": total_demand > total_supply,
    }


# ---------------------------------------------------------------------------
# Task 3: cascade recovery (stretch — minimal stub)
# ---------------------------------------------------------------------------


def _build_task_3(seed: int) -> Dict[str, Any]:
    """Stretch task: keep the shape valid but logic intentionally minimal.

    The headline of the project is T2; T3 exists so the env exposes the
    right action surface end-to-end. We will flesh this out only if T1 and
    T2 are fully shipped and there is hackathon time left.
    """
    bundle = _bundle_skeleton("task_3")
    bundle["warehouses"] = [
        _wh("W1", "north", [ShippingMethod.STANDARD.value]),
        _wh("W2", "central", [ShippingMethod.STANDARD.value, ShippingMethod.EXPRESS.value]),
    ]
    bundle["stock"] = [
        _stock("W1", "SKU-A", 0),
        _stock("W2", "SKU-A", 2),
        _stock("W1", "SKU-B", 1),
        _stock("W2", "SKU-B", 0),
    ]
    bundle["orders"] = [
        _order(
            order_id="O1",
            sku="SKU-A",
            quantity=1,
            tier=CustomerTier.PREMIUM.value,
            sla_hours=18,
            region="central",
            distance_buckets={"W1": DistanceBucket.MID.value, "W2": DistanceBucket.NEAR.value},
            method=ShippingMethod.STANDARD.value,
        ),
        _order(
            order_id="O2",
            sku="SKU-B",
            quantity=1,
            tier=CustomerTier.LOYALTY.value,
            sla_hours=8,
            region="north",
            distance_buckets={"W1": DistanceBucket.NEAR.value, "W2": DistanceBucket.MID.value},
            method=ShippingMethod.EXPRESS.value,
        ),
    ]
    bundle["policy_flags"] = {
        "supplier_failure": True,
        "stretch": True,
        "failed_supplier_id": "SUP-RED",
    }
    bundle["ground_truth"] = {
        "kind": "cascade_recovery_stub",
        "expected_actions": {
            "O1": {"action_type": ActionType.REROUTE_ORDER.value, "warehouse_id": "W2"},
            "O2": {"action_type": ActionType.REFUND_OR_COMPENSATE.value, "compensation_type": "credit_25"},
        },
        "expected_supplier_escalation": "SUP-RED",
    }
    return bundle


__all__ = [
    "TASK_CATALOG",
    "get_task_bundle",
    "task_catalog",
]
