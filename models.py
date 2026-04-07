from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

try:
    from openenv.core.env_server import Action, Observation, State
except ImportError:
    @dataclass
    class Action:
        pass

    @dataclass
    class Observation:
        done: bool = False
        reward: float = 0.0
        metadata: Dict[str, Any] = field(default_factory=dict)

    @dataclass
    class State:
        episode_id: str = ""
        step_count: int = 0


@dataclass
class OrderItem:
    sku: str
    quantity: int
    status: str = "PENDING"
    substitute_sku: Optional[str] = None

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "OrderItem":
        return cls(
            sku=str(payload.get("sku", "")),
            quantity=int(payload.get("quantity", 0)),
            status=str(payload.get("status", "PENDING")),
            substitute_sku=payload.get("substitute_sku"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class OrderRecord:
    order_id: str
    customer_id: str
    customer_tier: str
    status: str
    items: List[OrderItem] = field(default_factory=list)
    warehouse: Optional[str] = None
    compensation: List[str] = field(default_factory=list)
    touched: bool = False

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "OrderRecord":
        items_payload = payload.get("items", []) or []
        items = [OrderItem.from_dict(item) for item in items_payload]
        compensation = [str(value) for value in (payload.get("compensation", []) or [])]
        return cls(
            order_id=str(payload.get("order_id", "")),
            customer_id=str(payload.get("customer_id", "")),
            customer_tier=str(payload.get("customer_tier", "standard")),
            status=str(payload.get("status", "PENDING")),
            items=items,
            warehouse=payload.get("warehouse"),
            compensation=compensation,
            touched=bool(payload.get("touched", False)),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TicketRecord:
    ticket_id: str
    order_id: str
    customer_id: str
    reason: str
    created_days_ago: int
    status: str = "OPEN"

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "TicketRecord":
        return cls(
            ticket_id=str(payload.get("ticket_id", "")),
            order_id=str(payload.get("order_id", "")),
            customer_id=str(payload.get("customer_id", "")),
            reason=str(payload.get("reason", "")),
            created_days_ago=int(payload.get("created_days_ago", 0)),
            status=str(payload.get("status", "OPEN")),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class InventoryRecord:
    sku: str
    warehouse: str
    quantity: int

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "InventoryRecord":
        return cls(
            sku=str(payload.get("sku", "")),
            warehouse=str(payload.get("warehouse", "")),
            quantity=int(payload.get("quantity", 0)),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EcommerceAction(Action):
    action_type: str = ""
    order_id: Optional[str] = None
    ticket_id: Optional[str] = None
    sku: Optional[str] = None
    warehouse: Optional[str] = None
    quantity: Optional[int] = None
    reason: Optional[str] = None
    compensation_type: Optional[str] = None

    @classmethod
    def from_dict(cls, payload: Dict[str, Any]) -> "EcommerceAction":
        return cls(
            action_type=str(payload.get("action_type", "")),
            order_id=payload.get("order_id"),
            ticket_id=payload.get("ticket_id"),
            sku=payload.get("sku"),
            warehouse=payload.get("warehouse"),
            quantity=payload.get("quantity"),
            reason=payload.get("reason"),
            compensation_type=payload.get("compensation_type"),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EcommerceObservation(Observation):
    open_tickets: List[Dict[str, Any]] = field(default_factory=list)
    orders: List[Dict[str, Any]] = field(default_factory=list)
    inventory: List[Dict[str, Any]] = field(default_factory=list)
    last_action_result: str = ""
    last_action_error: Optional[str] = None
    task_description: str = ""
    task_id: str = ""
    steps_remaining: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EcommerceState(State):
    task_id: str = ""
    max_steps: int = 0
    orders: List[OrderRecord] = field(default_factory=list)
    inventory: List[InventoryRecord] = field(default_factory=list)
    tickets: List[TicketRecord] = field(default_factory=list)
    products: List[Dict[str, Any]] = field(default_factory=list)
    resolved_correctly: List[str] = field(default_factory=list)
    resolved_incorrectly: List[str] = field(default_factory=list)
    collateral_damage: List[str] = field(default_factory=list)
    unnecessary_escalations: int = 0
    cumulative_reward: float = 0.0
    episode_done: bool = False
    ground_truth: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
