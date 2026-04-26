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
    """Simplified multi-order triage: 2 orders, 1 SKU, 2 warehouses.

    Design rationale (after the 3-order-with-DELAY version trained to a
    degenerate "delay everything" policy):

      - 2 orders, 1 SKU, 2 warehouses with 1 unit of stock each.
        Total supply (2) == total demand (2) — no order needs to be delayed,
        but the only correct play is to send EACH order to its nearer
        warehouse. Wasting the only-stocked warehouse on the wrong order
        leaves the other order with no inventory to serve.

      - Both expected actions are ASSIGN_WAREHOUSE. There is no order whose
        plan calls for DELAY, so a "delay everything" policy can no longer
        farm step rewards.

      - One order has its destination in "north" (W1 is NEAR), the other in
        "central" (W2 is NEAR). Tier weights still differ (LOYALTY > PREMIUM)
        so wrong assignments are not symmetric.

      - max_steps = 4 leaves a 2-step retry buffer beyond the 2-step
        oracle plan.

    The seed permutes which order_id is which (O1 vs O2 carries each tier),
    so the model cannot memorise "always assign O1 to W1".
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
    """Simplified cascade recovery: a single supplier-failure reroute.

    Design (intentionally trainable for tiny GRPO budgets):
      - 1 SKU, 2 warehouses, 1 active order.
      - W1 has had a supplier failure -> stock = 0.
      - W2 still has stock -> the only correct move is to reroute O1 to W2.
      - max_steps = 4. The model needs ONE correct action to win.

    The seed swaps which warehouse failed so the model can't memorise a
    fixed answer ("always W2"). It must read the stock table.
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
        "kind": "cascade_recovery_simple",
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
