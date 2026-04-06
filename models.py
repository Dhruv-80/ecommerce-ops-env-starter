from dataclasses import dataclass, field
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


@dataclass
class TicketRecord:
    ticket_id: str
    order_id: str
    customer_id: str
    reason: str
    created_days_ago: int
    status: str = "OPEN"


@dataclass
class InventoryRecord:
    sku: str
    warehouse: str
    quantity: int


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
