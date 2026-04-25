"""CommerceOps-Env: the stateful OpenEnv-compatible RL environment.

The environment implements the canonical ``reset / step / state`` loop.
It is the *only* component that may mutate world state — models interact
exclusively through ``EnvAction`` objects which are schema-validated before
any mutation occurs.

Anti-hacking rules enforced here (context.md §Anti-Reward-Hacking):
1. Action whitelist — actions outside ``ALLOWED_ACTIONS_BY_TASK`` for the
   current task are rejected as schema-invalid.
2. Strict per-field Pydantic validation before state mutation.
3. Protected inventory: ``StockCell.quantity`` is decremented only by the
   ``_consume_stock`` helper; the model can never write to it directly.
4. Max-step timeout: ``episode_done`` is forced True at ``max_steps``.
5. Repeat-action detection: same (action_type, order_id) on consecutive
   steps carries the ``is_repeat`` penalty signal.
"""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import ValidationError

from models import (
    ALLOWED_ACTIONS_BY_TASK,
    ActionType,
    Allocation,
    EnvAction,
    EnvObservation,
    EnvState,
    Order,
    OrderStatus,
    OrderView,
    StockCell,
    StockView,
    Warehouse,
    WarehouseView,
)
from reward import compute_invalid_action_reward, compute_step_reward
from tasks import get_task_bundle
from verifier import grade_episode, is_task_resolved, verify_step


class CommerceOpsEnv:
    """Stateful fulfillment-judgment RL environment.

    Usage::

        env = CommerceOpsEnv()
        obs = env.reset("task_1", seed=0)
        obs = env.step({"action_type": "assign_warehouse",
                        "order_id": "O1", "warehouse_id": "W1"})
        score = env.final_score()
    """

    def __init__(self) -> None:
        self._state: Optional[EnvState] = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    @property
    def state(self) -> EnvState:
        if self._state is None:
            raise RuntimeError("Environment not initialised — call reset() first.")
        return self._state

    def reset(self, task_id: str = "task_1", seed: int = 0) -> EnvObservation:
        """Start a fresh deterministic episode."""
        bundle = get_task_bundle(task_id, seed=seed)
        task_type = bundle["task_type"]

        self._state = EnvState(
            episode_id=str(uuid4()),
            task_id=task_id,
            task_type=task_type,
            task_description=bundle["task_description"],
            step_count=0,
            max_steps=int(bundle["max_steps"]),
            episode_done=False,
            cumulative_reward=0.0,
            orders=[Order(**o) for o in bundle["orders"]],
            warehouses=[Warehouse(**w) for w in bundle["warehouses"]],
            stock=[StockCell(**s) for s in bundle["stock"]],
            allowed_actions=[a for a in bundle["allowed_actions"]],
            policy_flags=dict(bundle.get("policy_flags", {})),
            ground_truth=dict(bundle["ground_truth"]),
            invalid_action_count=0,
            repeat_action_count=0,
            last_action_signature=None,
        )

        return self._build_obs(
            last_action_result=f"Episode reset for {task_id} (seed={seed}).",
            reward=0.0,
            breakdown={},
        )

    def step(self, action_input: Any) -> EnvObservation:
        """Apply one action and return the next observation.

        ``action_input`` may be an ``EnvAction``, a plain ``dict``, or a
        JSON string — the env normalises it before validation.
        """
        state = self.state

        if state.episode_done:
            return self._build_obs(
                last_action_result="Episode already finished.",
                last_action_error="episode_done",
                reward=0.0,
                breakdown={},
            )

        # ── 1. Parse & schema-validate ──────────────────────────────────
        raw = self._coerce_to_dict(action_input)
        action, parse_error = self._validate_action(raw, state.task_type)

        if action is None:
            state.step_count += 1
            state.invalid_action_count += 1
            self._maybe_end_episode(state)
            ri = compute_invalid_action_reward()
            state.cumulative_reward += ri["reward"]
            return self._build_obs(
                last_action_result=f"Invalid action: {parse_error}",
                last_action_error=parse_error,
                reward=ri["reward"],
                breakdown=ri["breakdown"],
            )

        # ── 2. Repeat detection ─────────────────────────────────────────
        sig = self._action_signature(action)
        is_repeat = sig == state.last_action_signature
        state.last_action_signature = sig
        if is_repeat:
            state.repeat_action_count += 1

        state.step_count += 1

        # ── 3. Apply state transition ───────────────────────────────────
        result_msg, collateral = self._apply(action, state)

        # ── 4. Build snapshot for verifier ─────────────────────────────
        snap = self._state_snapshot(state)
        action_dict = json.loads(action.model_dump_json())
        action_dict["_is_repeat"] = is_repeat
        action_dict["_within_budget"] = state.step_count <= state.max_steps
        action_dict["_collateral_damage"] = collateral

        signals = verify_step(
            action=action_dict,
            state_snapshot=snap,
            ground_truth=state.ground_truth,
        )
        signals["is_repeat"] = is_repeat
        signals["within_budget"] = state.step_count <= state.max_steps
        signals["collateral_damage"] = collateral

        # ── 5. Compute reward ────────────────────────────────────────────
        ri = compute_step_reward(signals=signals)
        state.cumulative_reward += ri["reward"]

        # ── 6. Terminal check ────────────────────────────────────────────
        self._maybe_end_episode(state)

        return self._build_obs(
            last_action_result=result_msg,
            last_action_error=ri.get("error"),
            reward=ri["reward"],
            breakdown=ri["breakdown"],
            last_action_type=action.action_type.value,
        )

    def final_score(self) -> Dict[str, Any]:
        """Grade the completed episode. Safe to call mid-episode too."""
        state = self.state
        graded = grade_episode(state.task_id, state)
        return {
            "score": graded["score"],
            "breakdown": graded["breakdown"],
            "cumulative_reward": round(state.cumulative_reward, 6),
            "steps": state.step_count,
            "invalid_actions": state.invalid_action_count,
            "repeat_actions": state.repeat_action_count,
        }

    # ------------------------------------------------------------------
    # State transition handlers
    # ------------------------------------------------------------------

    def _apply(self, action: EnvAction, state: EnvState) -> tuple[str, bool]:
        """Dispatch to the right handler and return (message, collateral_flag)."""
        a = action.action_type

        if a == ActionType.ASSIGN_WAREHOUSE:
            return self._apply_assign(action, state)
        if a == ActionType.SPLIT_SHIPMENT:
            return self._apply_split(action, state)
        if a == ActionType.DELAY_ORDER:
            return self._apply_delay(action, state)
        if a == ActionType.PRIORITIZE_ORDER:
            return self._apply_prioritize(action, state)
        if a == ActionType.REROUTE_ORDER:
            return self._apply_reroute(action, state)
        if a == ActionType.ESCALATE_SUPPLIER:
            return self._apply_escalate(action, state)
        if a == ActionType.REFUND_OR_COMPENSATE:
            return self._apply_refund(action, state)
        if a == ActionType.NOOP:
            return "No-op.", False

        return f"Unhandled action {a.value}.", False  # pragma: no cover

    def _apply_assign(self, action: EnvAction, state: EnvState) -> tuple[str, bool]:
        order = self._find_order(state, action.order_id)
        if order is None:
            return f"Order {action.order_id} not found.", False

        wh = self._find_warehouse(state, action.warehouse_id)
        if wh is None:
            return f"Warehouse {action.warehouse_id} not found.", False

        qty = action.quantity or order.quantity_requested
        stock_ok = self._consume_stock(state, order.sku, action.warehouse_id, qty)
        if not stock_ok:
            return f"Insufficient stock of {order.sku} at {action.warehouse_id}.", True

        order.assigned_warehouse = action.warehouse_id
        order.status = OrderStatus.ASSIGNED.value
        return f"Order {order.order_id} assigned to {action.warehouse_id}.", False

    def _apply_split(self, action: EnvAction, state: EnvState) -> tuple[str, bool]:
        order = self._find_order(state, action.order_id)
        if order is None:
            return f"Order {action.order_id} not found.", False

        allocs: List[Allocation] = action.allocations or []
        total_qty = sum(a.quantity for a in allocs)
        if total_qty < order.quantity_requested:
            return (
                f"Split allocations total {total_qty} < requested "
                f"{order.quantity_requested}.",
                False,
            )

        collateral = False
        legs_ok: List[Dict[str, Any]] = []
        for alloc in allocs:
            ok = self._consume_stock(state, order.sku, alloc.warehouse_id, alloc.quantity)
            if not ok:
                collateral = True
                continue
            legs_ok.append({"warehouse_id": alloc.warehouse_id, "quantity": alloc.quantity})

        if not legs_ok:
            return "No stock available at any allocation leg.", True

        order.allocations = legs_ok
        order.status = OrderStatus.SPLIT.value
        return (
            f"Order {order.order_id} split across "
            f"{', '.join(l['warehouse_id'] for l in legs_ok)}.",
            collateral,
        )

    def _apply_delay(self, action: EnvAction, state: EnvState) -> tuple[str, bool]:
        order = self._find_order(state, action.order_id)
        if order is None:
            return f"Order {action.order_id} not found.", False
        order.status = OrderStatus.DELAYED.value
        order.reason = action.reason or "agent_decision"
        return f"Order {order.order_id} marked delayed.", False

    def _apply_prioritize(self, action: EnvAction, state: EnvState) -> tuple[str, bool]:
        order = self._find_order(state, action.order_id)
        if order is None:
            return f"Order {action.order_id} not found.", False
        order.prioritized = True
        return f"Order {order.order_id} prioritized.", False

    def _apply_reroute(self, action: EnvAction, state: EnvState) -> tuple[str, bool]:
        order = self._find_order(state, action.order_id)
        if order is None:
            return f"Order {action.order_id} not found.", False

        # Release stock from old warehouse if previously assigned.
        if order.assigned_warehouse and order.status == OrderStatus.ASSIGNED.value:
            self._return_stock(state, order.sku, order.assigned_warehouse, order.quantity_requested)

        qty = order.quantity_requested
        ok = self._consume_stock(state, order.sku, action.warehouse_id, qty)
        if not ok:
            return f"No stock for reroute to {action.warehouse_id}.", True

        order.assigned_warehouse = action.warehouse_id
        order.status = OrderStatus.ASSIGNED.value
        return f"Order {order.order_id} rerouted to {action.warehouse_id}.", False

    def _apply_escalate(self, action: EnvAction, state: EnvState) -> tuple[str, bool]:
        state.policy_flags["supplier_escalated"] = action.supplier_id
        return f"Supplier {action.supplier_id} escalated.", False

    def _apply_refund(self, action: EnvAction, state: EnvState) -> tuple[str, bool]:
        order = self._find_order(state, action.order_id)
        if order is None:
            return f"Order {action.order_id} not found.", False
        order.status = OrderStatus.CANCELLED.value
        order.reason = action.compensation_type
        return (
            f"Order {order.order_id} refunded/compensated "
            f"({action.compensation_type}).",
            False,
        )

    # ------------------------------------------------------------------
    # Stock management (protected — model can never call these)
    # ------------------------------------------------------------------

    def _consume_stock(
        self, state: EnvState, sku: str, warehouse_id: str, qty: int
    ) -> bool:
        """Deduct qty from the stock cell. Returns False if insufficient."""
        cell = self._find_stock(state, sku, warehouse_id)
        if cell is None or cell.quantity < qty:
            return False
        cell.quantity -= qty
        return True

    def _return_stock(
        self, state: EnvState, sku: str, warehouse_id: str, qty: int
    ) -> None:
        """Return qty to stock (used when rerouting cancels a prior assignment)."""
        cell = self._find_stock(state, sku, warehouse_id)
        if cell is not None:
            cell.quantity += qty
        else:
            state.stock.append(StockCell(warehouse_id=warehouse_id, sku=sku, quantity=qty))

    # ------------------------------------------------------------------
    # Lookup helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _find_order(state: EnvState, order_id: Optional[str]) -> Optional[Order]:
        if not order_id:
            return None
        return next((o for o in state.orders if o.order_id == order_id), None)

    @staticmethod
    def _find_warehouse(state: EnvState, warehouse_id: Optional[str]) -> Optional[Warehouse]:
        if not warehouse_id:
            return None
        return next((w for w in state.warehouses if w.warehouse_id == warehouse_id), None)

    @staticmethod
    def _find_stock(state: EnvState, sku: str, warehouse_id: str) -> Optional[StockCell]:
        return next(
            (c for c in state.stock if c.sku == sku and c.warehouse_id == warehouse_id),
            None,
        )

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def _validate_action(
        self, raw: Dict[str, Any], task_type: str
    ) -> tuple[Optional[EnvAction], Optional[str]]:
        """Schema-validate the raw dict and check task whitelist."""
        allowed = {a.value for a in ALLOWED_ACTIONS_BY_TASK.get(task_type, [])}
        action_type = raw.get("action_type", "")

        if action_type not in allowed:
            return None, f"action_type '{action_type}' not allowed for {task_type}"

        try:
            return EnvAction(**raw), None
        except (ValidationError, TypeError) as exc:
            return None, str(exc)

    @staticmethod
    def _coerce_to_dict(action_input: Any) -> Dict[str, Any]:
        if isinstance(action_input, EnvAction):
            return json.loads(action_input.model_dump_json())
        if isinstance(action_input, str):
            try:
                return json.loads(action_input)
            except json.JSONDecodeError:
                return {}
        if isinstance(action_input, dict):
            return action_input
        return {}

    @staticmethod
    def _action_signature(action: EnvAction) -> str:
        return f"{action.action_type.value}|{action.order_id}|{action.warehouse_id}"

    # ------------------------------------------------------------------
    # Terminal condition
    # ------------------------------------------------------------------

    def _maybe_end_episode(self, state: EnvState) -> None:
        if state.step_count >= state.max_steps:
            state.episode_done = True
        elif is_task_resolved(state):
            state.episode_done = True

    # ------------------------------------------------------------------
    # Observation builder
    # ------------------------------------------------------------------

    def _build_obs(
        self,
        last_action_result: str = "",
        last_action_error: Optional[str] = None,
        reward: float = 0.0,
        breakdown: Optional[Dict[str, Any]] = None,
        last_action_type: Optional[str] = None,
    ) -> EnvObservation:
        state = self.state
        return EnvObservation(
            task_id=state.task_id,
            task_type=state.task_type,   # type: ignore[arg-type]
            episode_id=state.episode_id,
            step=state.step_count,
            max_steps=state.max_steps,
            steps_remaining=max(state.max_steps - state.step_count, 0),
            done=state.episode_done,
            reward=round(reward, 6),
            cumulative_reward=round(state.cumulative_reward, 6),
            orders=[OrderView(**o.to_dict()) for o in state.orders],
            warehouses=[WarehouseView(**w.to_dict()) for w in state.warehouses],
            stock=[StockView(**s.to_dict()) for s in state.stock],
            allowed_actions=ALLOWED_ACTIONS_BY_TASK.get(state.task_type, []),
            policy_flags=dict(state.policy_flags),
            last_action_type=last_action_type,
            last_action_result=last_action_result,
            last_action_error=last_action_error,
            reward_breakdown={k: round(v, 6) for k, v in (breakdown or {}).items()},
            task_description=state.task_description,
        )

    # ------------------------------------------------------------------
    # State snapshot (for verifier + debug)
    # ------------------------------------------------------------------

    def _state_snapshot(self, state: EnvState) -> Dict[str, Any]:
        return {
            "orders": [o.to_dict() for o in state.orders],
            "warehouses": [w.to_dict() for w in state.warehouses],
            "stock": [s.to_dict() for s in state.stock],
            "policy_flags": dict(state.policy_flags),
        }

    def state_dict(self) -> Dict[str, Any]:
        """Full serialisable state dump (for /state endpoint)."""
        return self.state.to_dict()
