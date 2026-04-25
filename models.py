"""CommerceOps-Env contracts.

Frozen typed contracts for the fulfillment-judgment environment.
Per context.md, this file is the contract layer for T1 (warehouse assignment
bootstrap), T2 (multi-order fulfillment triage, the headline task) and T3
(cascade recovery, stretch). Do not rename keys after this freeze.

Design choices:
- ``EnvAction`` is a single Pydantic model with an action-type discriminator
  validated per ``ActionType``. This gives us cheap schema-compliance checks
  for the reward and a clean FastAPI request body.
- ``EnvObservation`` is a Pydantic model so it serialises trivially over the
  OpenEnv HTTP boundary and is easy for the LLM to read as JSON.
- ``EnvState`` and the inner records (``Order``, ``Warehouse``,
  ``StockCell``) are dataclasses because the environment mutates them many
  times per step; they are never sent to the model directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, model_validator


# ---------------------------------------------------------------------------
# Enums and string literals
# ---------------------------------------------------------------------------


class TaskType(str, Enum):
    """High-level task family. T1 bootstraps action format; T2 is headline."""

    T1_WAREHOUSE_ASSIGNMENT = "t1_warehouse_assignment"
    T2_MULTI_ORDER_TRIAGE = "t2_multi_order_triage"
    T3_CASCADE_RECOVERY = "t3_cascade_recovery"


class CustomerTier(str, Enum):
    STANDARD = "standard"
    PREMIUM = "premium"
    LOYALTY = "loyalty"


class DistanceBucket(str, Enum):
    """Coarse distance bucket between an order destination and a warehouse."""

    NEAR = "near"
    MID = "mid"
    FAR = "far"


class ShippingMethod(str, Enum):
    STANDARD = "standard"
    EXPRESS = "express"
    OVERNIGHT = "overnight"


class OrderStatus(str, Enum):
    PENDING = "pending"
    ASSIGNED = "assigned"
    SPLIT = "split"
    DELAYED = "delayed"
    CANCELLED = "cancelled"
    FULFILLED = "fulfilled"


class ActionType(str, Enum):
    """Whitelisted action verbs. Anything else is an invalid action."""

    ASSIGN_WAREHOUSE = "assign_warehouse"
    SPLIT_SHIPMENT = "split_shipment"
    DELAY_ORDER = "delay_order"
    PRIORITIZE_ORDER = "prioritize_order"
    REROUTE_ORDER = "reroute_order"
    ESCALATE_SUPPLIER = "escalate_supplier"
    REFUND_OR_COMPENSATE = "refund_or_compensate"
    NOOP = "noop"


# Tier weighting used by verifier and reward (loyalty hurts more if missed).
TIER_WEIGHT: Dict[str, float] = {
    CustomerTier.STANDARD.value: 1.0,
    CustomerTier.PREMIUM.value: 1.5,
    CustomerTier.LOYALTY.value: 2.0,
}


# ---------------------------------------------------------------------------
# Action contract (Pydantic, validated)
# ---------------------------------------------------------------------------


class Allocation(BaseModel):
    """A single (warehouse, quantity) leg of a split shipment."""

    model_config = ConfigDict(extra="forbid")

    warehouse_id: str = Field(..., min_length=1)
    quantity: int = Field(..., ge=1)


class EnvAction(BaseModel):
    """Single structured action submitted by the agent.

    Required fields depend on ``action_type``; the validator below enforces
    per-type contracts so the reward layer can trust ``valid_action``.
    """

    model_config = ConfigDict(extra="forbid")

    action_type: ActionType

    # Common targeting fields. None when not applicable.
    order_id: Optional[str] = None
    warehouse_id: Optional[str] = None
    quantity: Optional[int] = Field(default=None, ge=1)

    # split_shipment payload
    allocations: Optional[List[Allocation]] = None

    # cascade-recovery / supplier actions
    supplier_id: Optional[str] = None
    compensation_type: Optional[str] = None

    # Free-form short reason code (not free text; bounded length)
    reason: Optional[str] = Field(default=None, max_length=64)

    @model_validator(mode="after")
    def _validate_per_action_type(self) -> "EnvAction":
        a = self.action_type

        if a == ActionType.ASSIGN_WAREHOUSE:
            self._require("order_id", "warehouse_id")
        elif a == ActionType.SPLIT_SHIPMENT:
            self._require("order_id", "allocations")
            if not self.allocations or len(self.allocations) < 2:
                raise ValueError("split_shipment requires at least 2 allocations")
        elif a == ActionType.DELAY_ORDER:
            self._require("order_id")
        elif a == ActionType.PRIORITIZE_ORDER:
            self._require("order_id")
        elif a == ActionType.REROUTE_ORDER:
            self._require("order_id", "warehouse_id")
        elif a == ActionType.ESCALATE_SUPPLIER:
            self._require("supplier_id")
        elif a == ActionType.REFUND_OR_COMPENSATE:
            self._require("order_id", "compensation_type")
        elif a == ActionType.NOOP:
            pass
        return self

    def _require(self, *names: str) -> None:
        missing = [n for n in names if getattr(self, n) in (None, "", [])]
        if missing:
            raise ValueError(f"{self.action_type.value} requires: {', '.join(missing)}")


# ---------------------------------------------------------------------------
# World records (dataclasses, internal state)
# ---------------------------------------------------------------------------


@dataclass
class Warehouse:
    """A single fulfillment warehouse."""

    warehouse_id: str
    region: str
    supports_methods: List[str] = field(default_factory=lambda: [ShippingMethod.STANDARD.value])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "warehouse_id": self.warehouse_id,
            "region": self.region,
            "supports_methods": list(self.supports_methods),
        }


@dataclass
class StockCell:
    """Per-warehouse stock for one SKU. Mutated by the environment only."""

    warehouse_id: str
    sku: str
    quantity: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "warehouse_id": self.warehouse_id,
            "sku": self.sku,
            "quantity": self.quantity,
        }


@dataclass
class Order:
    """An incoming customer order awaiting a fulfillment decision.

    For T1 there is exactly one such order. For T2 there are several
    competing for the same warehouse stock. SLA is in hours-remaining and
    distance is bucketed (near/mid/far) to keep observations small and the
    decision a judgment call rather than a routing puzzle.
    """

    order_id: str
    customer_id: str
    customer_tier: str
    sku: str
    quantity_requested: int
    sla_hours_remaining: int
    destination_region: str
    distance_buckets: Dict[str, str] = field(default_factory=dict)
    required_method: str = ShippingMethod.STANDARD.value
    status: str = OrderStatus.PENDING.value
    assigned_warehouse: Optional[str] = None
    allocations: List[Dict[str, Any]] = field(default_factory=list)
    reason: Optional[str] = None
    prioritized: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "order_id": self.order_id,
            "customer_id": self.customer_id,
            "customer_tier": self.customer_tier,
            "sku": self.sku,
            "quantity_requested": self.quantity_requested,
            "sla_hours_remaining": self.sla_hours_remaining,
            "destination_region": self.destination_region,
            "distance_buckets": dict(self.distance_buckets),
            "required_method": self.required_method,
            "status": self.status,
            "assigned_warehouse": self.assigned_warehouse,
            "allocations": list(self.allocations),
            "reason": self.reason,
            "prioritized": self.prioritized,
        }


# ---------------------------------------------------------------------------
# Observation (what the model sees)
# ---------------------------------------------------------------------------


class OrderView(BaseModel):
    """Public view of an order sent to the model.

    ``customer_id`` is intentionally excluded — models must reason from
    ``customer_tier``, not from raw customer IDs. ``reason`` is included so
    the model sees why an order was delayed/cancelled.
    """

    model_config = ConfigDict(extra="ignore")   # silently drop unknown fields

    order_id: str
    customer_tier: str
    sku: str
    quantity_requested: int
    sla_hours_remaining: int
    destination_region: str
    distance_buckets: Dict[str, str] = Field(default_factory=dict)
    required_method: str = ShippingMethod.STANDARD.value
    status: str = OrderStatus.PENDING.value
    assigned_warehouse: Optional[str] = None
    allocations: List[Dict[str, Any]] = Field(default_factory=list)
    prioritized: bool = False
    reason: Optional[str] = None


class WarehouseView(BaseModel):
    model_config = ConfigDict(extra="forbid")

    warehouse_id: str
    region: str
    supports_methods: List[str]


class StockView(BaseModel):
    model_config = ConfigDict(extra="forbid")

    warehouse_id: str
    sku: str
    quantity: int


class EnvObservation(BaseModel):
    """Structured observation returned by ``reset`` and ``step``."""

    model_config = ConfigDict(extra="forbid")

    # --- episode meta ---
    task_id: str
    task_type: TaskType
    episode_id: str
    step: int
    max_steps: int
    steps_remaining: int
    done: bool = False
    reward: float = 0.0
    cumulative_reward: float = 0.0

    # --- world view ---
    orders: List[OrderView] = Field(default_factory=list)
    warehouses: List[WarehouseView] = Field(default_factory=list)
    stock: List[StockView] = Field(default_factory=list)

    # --- decision affordances ---
    allowed_actions: List[ActionType] = Field(default_factory=list)
    policy_flags: Dict[str, Any] = Field(default_factory=dict)

    # --- last-step trace ---
    last_action_type: Optional[str] = None
    last_action_result: str = ""
    last_action_error: Optional[str] = None
    reward_breakdown: Dict[str, float] = Field(default_factory=dict)

    # --- task narration (kept short for the LLM) ---
    task_description: str = ""


# ---------------------------------------------------------------------------
# Internal environment state (server-side only)
# ---------------------------------------------------------------------------


@dataclass
class EnvState:
    """Full server-side state. Never returned in observations as-is — the
    environment publishes a curated ``EnvObservation`` instead."""

    episode_id: str = ""
    task_id: str = ""
    task_type: str = TaskType.T1_WAREHOUSE_ASSIGNMENT.value
    task_description: str = ""

    step_count: int = 0
    max_steps: int = 0
    episode_done: bool = False
    cumulative_reward: float = 0.0

    orders: List[Order] = field(default_factory=list)
    warehouses: List[Warehouse] = field(default_factory=list)
    stock: List[StockCell] = field(default_factory=list)

    allowed_actions: List[str] = field(default_factory=list)
    policy_flags: Dict[str, Any] = field(default_factory=dict)

    # Verifier ground-truth payload (per-task). Opaque to the model.
    ground_truth: Dict[str, Any] = field(default_factory=dict)

    # Observability counters (used by anti-hacking checks and metrics)
    invalid_action_count: int = 0
    repeat_action_count: int = 0
    last_action_signature: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "task_id": self.task_id,
            "task_type": self.task_type,
            "task_description": self.task_description,
            "step_count": self.step_count,
            "max_steps": self.max_steps,
            "episode_done": self.episode_done,
            "cumulative_reward": self.cumulative_reward,
            "orders": [o.to_dict() for o in self.orders],
            "warehouses": [w.to_dict() for w in self.warehouses],
            "stock": [s.to_dict() for s in self.stock],
            "allowed_actions": list(self.allowed_actions),
            "policy_flags": dict(self.policy_flags),
            "ground_truth": dict(self.ground_truth),
            "invalid_action_count": self.invalid_action_count,
            "repeat_action_count": self.repeat_action_count,
            "last_action_signature": self.last_action_signature,
        }


# ---------------------------------------------------------------------------
# Convenience: per-task allowed-action sets
# ---------------------------------------------------------------------------


ALLOWED_ACTIONS_BY_TASK: Dict[str, List[ActionType]] = {
    TaskType.T1_WAREHOUSE_ASSIGNMENT.value: [
        ActionType.ASSIGN_WAREHOUSE,
        ActionType.NOOP,
    ],
    TaskType.T2_MULTI_ORDER_TRIAGE.value: [
        ActionType.ASSIGN_WAREHOUSE,
        ActionType.SPLIT_SHIPMENT,
        ActionType.DELAY_ORDER,
        ActionType.PRIORITIZE_ORDER,
        ActionType.NOOP,
    ],
    TaskType.T3_CASCADE_RECOVERY.value: [
        ActionType.REROUTE_ORDER,
        ActionType.ESCALATE_SUPPLIER,
        ActionType.REFUND_OR_COMPENSATE,
        ActionType.DELAY_ORDER,
        ActionType.NOOP,
    ],
}


__all__ = [
    "ActionType",
    "Allocation",
    "ALLOWED_ACTIONS_BY_TASK",
    "CustomerTier",
    "DistanceBucket",
    "EnvAction",
    "EnvObservation",
    "EnvState",
    "Order",
    "OrderStatus",
    "OrderView",
    "ShippingMethod",
    "StockCell",
    "StockView",
    "TaskType",
    "TIER_WEIGHT",
    "Warehouse",
    "WarehouseView",
]
