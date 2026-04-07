from dataclasses import asdict
from typing import Any, Dict, Optional, Tuple
from uuid import uuid4

try:
    from ..models import EcommerceAction, EcommerceObservation, EcommerceState, OrderItem, OrderRecord, TicketRecord, InventoryRecord
    from .tasks import get_task_bundle
    from .reward import compute_step_reward
    from .grader import grade_episode
except ImportError:
    from models import EcommerceAction, EcommerceObservation, EcommerceState, OrderItem, OrderRecord, TicketRecord, InventoryRecord
    from server.tasks import get_task_bundle
    from server.reward import compute_step_reward
    from server.grader import grade_episode

_PLACEHOLDER_ORDER_TIER = "standard"
_PLACEHOLDER_ORDER_STATUS = "PENDING"


def _build_order_record(o: Dict[str, Any]) -> OrderRecord:
    items = [OrderItem(**i) if isinstance(i, dict) else i for i in o.get("items", [])]
    return OrderRecord(**{**o, "items": items})


class EcommerceEnvironment:
    def __init__(self):
        self._state: Optional[EcommerceState] = None
        self._action_history = set()

    @property
    def state(self) -> EcommerceState:
        if self._state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        return self._state

    def reset(self, task_id: str = "task_1") -> EcommerceObservation:
        bundle = get_task_bundle(task_id)
        init = bundle["initial_state"]

        orders = []
        for order in init.get("orders", []):
            mapped = dict(order)
            mapped["items"] = [
                item if isinstance(item, OrderItem) else OrderItem(**item)
                for item in order.get("items", [])
            ]
            orders.append(OrderRecord(**mapped))

        tickets = [TicketRecord(**t) for t in init.get("tickets", [])]

        # Ensure every ticket's order_id has a corresponding OrderRecord so
        # that refund/dispute transitions can succeed even when the task bundle
        # does not pre-populate the orders list.
        existing_order_ids = {o.order_id for o in orders}
        for ticket in tickets:
            if ticket.order_id not in existing_order_ids:
                orders.append(OrderRecord(
                    order_id=ticket.order_id,
                    customer_id=ticket.customer_id,
                    customer_tier=_PLACEHOLDER_ORDER_TIER,
                    status=_PLACEHOLDER_ORDER_STATUS,
                ))
                existing_order_ids.add(ticket.order_id)

        self._state = EcommerceState(
            episode_id=str(uuid4()),
            task_id=task_id,
            max_steps=bundle["max_steps"],
            orders=[_build_order_record(o) for o in init.get("orders", [])],
            orders=orders,
            inventory=[InventoryRecord(**i) for i in init.get("inventory", [])],
            tickets=tickets,
            ground_truth=bundle.get("ground_truth", {}),
        )
        self._action_history = set()
        return self._build_observation(task_description=bundle["task"]["description"])

    def step(self, action: EcommerceAction) -> EcommerceObservation:
        state = self.state
        if state.episode_done:
            return self._build_observation(last_action_result="Episode already complete", last_action_error="episode_done", reward=0.0)

        state.step_count += 1

        outcome = {
            "valid_action": bool(action.action_type),
            "correct_target": False,
            "state_matches_ground_truth": False,
            "collateral_damage": False,
            "within_budget": state.step_count <= state.max_steps,
            "error": None,
        }
        result = f"Received action: {action.action_type}"
        error = None

        required_fields = {
            "process_refund": ["order_id", "ticket_id", "reason"],
            "reject_refund": ["order_id", "ticket_id", "reason"],
            "update_inventory": ["sku", "warehouse", "quantity"],
            "route_order": ["order_id", "warehouse"],
            "apply_substitute": ["order_id", "sku"],
            "cancel_order": ["order_id", "reason"],
            "flag_dispute": ["order_id", "ticket_id"],
            "escalate_to_human": ["ticket_id", "reason"],
            "send_compensation": ["order_id", "compensation_type"],
            "inspect_order": ["order_id"],
        }

        if action.action_type not in required_fields:
            outcome["valid_action"] = False
            error = f"Unknown action_type: {action.action_type}"
            outcome["error"] = error
        else:
            missing = [field for field in required_fields[action.action_type] if getattr(action, field) in (None, "")]
            if missing:
                outcome["valid_action"] = False
                error = f"Missing required fields: {', '.join(missing)}"
                outcome["error"] = error

        action_key = self._action_key(action)
        if action_key in self._action_history:
            outcome["repeat_action"] = True
        self._action_history.add(action_key)

        if error is None:
            result, error = self._apply_transition(action, outcome)

        reward_info = compute_step_reward(action=asdict(action), outcome=outcome, ground_truth=state.ground_truth)
        state.cumulative_reward += reward_info["reward"]
        state.episode_done = state.step_count >= state.max_steps or self._task_complete()

        return self._build_observation(
            last_action_result=result,
            last_action_error=error or reward_info.get("error"),
            reward=reward_info["reward"],
        )

    def final_score(self) -> Dict[str, Any]:
        return grade_episode(self.state.task_id, self.state)

    def _build_observation(self, task_description: str = "", last_action_result: str = "", last_action_error: Optional[str] = None, reward: float = 0.0) -> EcommerceObservation:
        state = self.state
        return EcommerceObservation(
            done=state.episode_done,
            reward=reward,
            metadata={"episode_id": state.episode_id, "step_count": state.step_count},
            open_tickets=[asdict(t) for t in state.tickets if t.status == "OPEN"],
            orders=[{"order_id": o.order_id, "status": o.status, "customer_tier": o.customer_tier} for o in state.orders],
            inventory=[asdict(i) for i in state.inventory],
            last_action_result=last_action_result,
            last_action_error=last_action_error,
            task_description=task_description,
            task_id=state.task_id,
            steps_remaining=max(state.max_steps - state.step_count, 0),
        )

    def _action_key(self, action: EcommerceAction) -> Tuple[Any, ...]:
        return (
            action.action_type,
            action.order_id,
            action.ticket_id,
            action.sku,
            action.warehouse,
            action.quantity,
            action.reason,
            action.compensation_type,
        )

    def _find_order(self, order_id: Optional[str]):
        if not order_id:
            return None
        for order in self.state.orders:
            if order.order_id == order_id:
                return order
        return None

    def _find_ticket(self, ticket_id: Optional[str]):
        if not ticket_id:
            return None
        for ticket in self.state.tickets:
            if ticket.ticket_id == ticket_id:
                return ticket
        return None

    def _find_inventory(self, sku: Optional[str], warehouse: Optional[str]):
        for row in self.state.inventory:
            if row.sku == sku and row.warehouse == warehouse:
                return row
        return None

    def _apply_transition(self, action: EcommerceAction, outcome: Dict[str, Any]) -> Tuple[str, Optional[str]]:
        state = self.state
        order = self._find_order(action.order_id)
        ticket = self._find_ticket(action.ticket_id)

        if action.action_type == "inspect_order":
            if order is None:
                outcome["wrong_entity"] = True
                return "Inspection failed", "order_not_found"
            outcome["correct_target"] = True
            return f"Inspected order {order.order_id}", None

        if action.action_type in {"process_refund", "reject_refund", "flag_dispute"}:
            if order is None or ticket is None or ticket.order_id != order.order_id or ticket.status != "OPEN":
                outcome["wrong_entity"] = True
                return "Ticket operation failed", "invalid_ticket_or_order"

        if action.action_type == "process_refund":
            ticket.status = "REFUNDED"
            order.status = "REFUNDED"
            order.touched = True
            expected = state.ground_truth.get("ticket_decisions", {}).get(ticket.ticket_id)
            outcome["correct_target"] = True
            outcome["state_matches_ground_truth"] = expected == "process_refund"
            if expected != "process_refund":
                state.resolved_incorrectly.append(ticket.ticket_id)
            else:
                state.resolved_correctly.append(ticket.ticket_id)
            return f"Refund processed for {order.order_id}", None

        if action.action_type == "reject_refund":
            ticket.status = "REJECTED"
            order.status = "REFUND_REJECTED"
            order.touched = True
            expected = state.ground_truth.get("ticket_decisions", {}).get(ticket.ticket_id)
            outcome["correct_target"] = True
            outcome["state_matches_ground_truth"] = expected == "reject_refund"
            if expected != "reject_refund":
                state.resolved_incorrectly.append(ticket.ticket_id)
            else:
                state.resolved_correctly.append(ticket.ticket_id)
            return f"Refund rejected for {order.order_id}", None

        if action.action_type == "flag_dispute":
            ticket.status = "DISPUTE"
            order.touched = True
            outcome["correct_target"] = True
            return f"Dispute flagged for {ticket.ticket_id}", None

        if action.action_type == "escalate_to_human":
            if ticket is None:
                outcome["wrong_entity"] = True
                return "Escalation failed", "ticket_not_found"
            state.unnecessary_escalations += 1
            ticket.status = "ESCALATED"
            outcome["correct_target"] = True
            outcome["unnecessary_escalation"] = True
            return f"Escalated ticket {ticket.ticket_id}", None

        if action.action_type == "update_inventory":
            row = self._find_inventory(action.sku, action.warehouse)
            if row is None:
                outcome["wrong_entity"] = True
                return "Inventory update failed", "inventory_row_not_found"
            row.quantity = int(action.quantity)
            outcome["correct_target"] = True
            expected = state.ground_truth.get("inventory", {}).get(f"{row.sku}|{row.warehouse}")
            if expected is not None:
                outcome["state_matches_ground_truth"] = row.quantity == expected
                if row.quantity != expected:
                    outcome["wrong_inventory"] = True
            return f"Inventory updated for {row.sku} at {row.warehouse}", None

        if action.action_type == "route_order":
            if order is None:
                outcome["wrong_entity"] = True
                return "Route failed", "order_not_found"
            order.warehouse = action.warehouse
            order.status = "ROUTED"
            order.touched = True
            outcome["correct_target"] = True
            expected_route = state.ground_truth.get("routes", {}).get(order.order_id)
            if expected_route is not None:
                outcome["state_matches_ground_truth"] = expected_route == action.warehouse
            return f"Order {order.order_id} routed to {action.warehouse}", None

        if action.action_type == "apply_substitute":
            if order is None:
                outcome["wrong_entity"] = True
                return "Substitution failed", "order_not_found"
            substituted = False
            for item in order.items:
                status = getattr(item, "status", None)
                if status == "CANCELLED":
                    setattr(item, "substitute_sku", action.sku)
                    setattr(item, "status", "SUBSTITUTED")
                    substituted = True
                    break
            if not substituted:
                outcome["wrong_entity"] = True
                return "Substitution failed", "no_cancelled_item"
            order.touched = True
            outcome["correct_target"] = True
            expected = state.ground_truth.get("resolutions", {}).get(order.order_id, {})
            if expected.get("expected_action") == "apply_substitute":
                outcome["state_matches_ground_truth"] = expected.get("substitute_sku") == action.sku
            else:
                outcome["destructive_cancel"] = False
            return f"Substitute applied for {order.order_id}", None

        if action.action_type == "cancel_order":
            if order is None:
                outcome["wrong_entity"] = True
                return "Cancel failed", "order_not_found"
            order.status = "CANCELLED"
            order.touched = True
            outcome["correct_target"] = True
            expected = state.ground_truth.get("resolutions", {}).get(order.order_id, {})
            if expected.get("expected_action") == "cancel_order":
                outcome["state_matches_ground_truth"] = True
            elif expected:
                outcome["destructive_cancel"] = True
            return f"Order {order.order_id} cancelled", None

        if action.action_type == "send_compensation":
            if order is None:
                outcome["wrong_entity"] = True
                return "Compensation failed", "order_not_found"
            if action.compensation_type not in order.compensation:
                order.compensation.append(action.compensation_type)
            order.touched = True
            outcome["correct_target"] = True
            expected = state.ground_truth.get("resolutions", {}).get(order.order_id, {})
            expected_comp = expected.get("compensation")
            if expected_comp is not None:
                outcome["state_matches_ground_truth"] = action.compensation_type == expected_comp
            return f"Compensation sent for {order.order_id}", None

        return "No-op", None

    def _task_complete(self) -> bool:
        state = self.state
        if state.task_id == "task_1":
            return all(ticket.status != "OPEN" for ticket in state.tickets)

        if state.task_id == "task_2":
            routes = state.ground_truth.get("routes", {})
            inventory = state.ground_truth.get("inventory", {})
            all_routes_ok = all(
                any(order.order_id == order_id and order.warehouse == warehouse for order in state.orders)
                for order_id, warehouse in routes.items()
            )
            all_inventory_ok = all(
                any(row.sku == sku_wh.split("|")[0] and row.warehouse == sku_wh.split("|")[1] and row.quantity == qty for row in state.inventory)
                for sku_wh, qty in inventory.items()
            )
            return all_routes_ok and all_inventory_ok

        if state.task_id == "task_3":
            resolutions = state.ground_truth.get("resolutions", {})
            if not resolutions:
                return False
            orders_by_id = {order.order_id: order for order in state.orders}
            for order_id, expected in resolutions.items():
                order = orders_by_id.get(order_id)
                if order is None:
                    return False
                if expected.get("expected_action") == "cancel_order" and order.status != "CANCELLED":
                    return False
                if expected.get("expected_action") == "apply_substitute":
                    has_substitute = any(getattr(item, "substitute_sku", None) == expected.get("substitute_sku") for item in order.items)
                    if not has_substitute:
                        return False
                expected_comp = expected.get("compensation")
                if expected_comp and expected_comp not in order.compensation:
                    return False
            return True

        return False
