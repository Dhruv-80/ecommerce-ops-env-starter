import json
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid4

try:
    from ..models import EcommerceAction, EcommerceObservation, EcommerceState, OrderRecord, TicketRecord, InventoryRecord
    from .tasks import get_task_bundle
    from .reward import compute_step_reward
    from .grader import grade_episode
except ImportError:
    from models import EcommerceAction, EcommerceObservation, EcommerceState, OrderRecord, TicketRecord, InventoryRecord
    from server.tasks import get_task_bundle
    from server.reward import compute_step_reward
    from server.grader import grade_episode


class EcommerceEnvironment:
    def __init__(self):
        self._state: Optional[EcommerceState] = None
        self._task_description: str = ""
        self._last_action_signature: Optional[Tuple[Any, ...]] = None
        self._last_inspection: Optional[Dict[str, Any]] = None

    ALLOWED_ACTIONS = {
        "process_refund",
        "reject_refund",
        "update_inventory",
        "route_order",
        "apply_substitute",
        "cancel_order",
        "flag_dispute",
        "escalate_to_human",
        "send_compensation",
        "inspect_order",
    }

    REQUIRED_FIELDS = {
        "process_refund": ("ticket_id", "order_id"),
        "reject_refund": ("ticket_id", "order_id"),
        "update_inventory": ("sku", "warehouse", "quantity"),
        "route_order": ("order_id", "warehouse"),
        "apply_substitute": ("order_id", "sku"),
        "cancel_order": ("order_id",),
        "flag_dispute": ("ticket_id",),
        "escalate_to_human": ("ticket_id",),
        "send_compensation": ("order_id", "compensation_type"),
        "inspect_order": ("order_id",),
    }

    @property
    def state(self) -> EcommerceState:
        if self._state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        return self._state

    def reset(self, task_id: str = "task_1") -> EcommerceObservation:
        bundle = get_task_bundle(task_id)
        init = bundle.get("initial_state", {}) or {}
        self._task_description = str((bundle.get("task", {}) or {}).get("description", ""))
        self._last_action_signature = None
        self._last_inspection = None

        self._state = EcommerceState(
            episode_id=str(uuid4()),
            task_id=task_id,
            max_steps=int(bundle.get("max_steps", 0)),
            orders=[OrderRecord.from_dict(o) for o in init.get("orders", [])],
            inventory=[InventoryRecord.from_dict(i) for i in init.get("inventory", [])],
            tickets=[TicketRecord.from_dict(t) for t in init.get("tickets", [])],
            products=list(init.get("products", []) or []),
            ground_truth=bundle.get("ground_truth", {}),
        )
        self._ensure_ticket_orders_exist()
        self._state.episode_done = False
        return self._build_observation(
            task_description=self._task_description,
            last_action_result=f"Reset episode for {task_id}",
            reward=0.0,
            reward_breakdown={},
        )

    def step(self, action: EcommerceAction) -> EcommerceObservation:
        state = self.state

        if isinstance(action, dict):
            action = EcommerceAction.from_dict(action)

        if state.episode_done:
            return self._build_observation(
                task_description=self._task_description,
                last_action_result="Episode already completed.",
                last_action_error="episode_done",
                reward=0.0,
                reward_breakdown={},
            )

        state.step_count += 1

        outcome = {
            "valid_action": False,
            "correct_target": False,
            "state_matches_ground_truth": False,
            "collateral_damage": False,
            "within_budget": state.step_count <= state.max_steps,
            "error": None,
        }
        result = f"Received action: {action.action_type}"
        error = None

        if action.action_type not in self.ALLOWED_ACTIONS:
            error = f"Unknown action_type: {action.action_type}"
            outcome["error"] = error
        else:
            missing_fields = self._missing_required_fields(action)
            if missing_fields:
                error = f"Missing required fields for {action.action_type}: {', '.join(missing_fields)}"
                outcome["error"] = error
            else:
                outcome["valid_action"] = True
                signature = self._action_signature(action)
                if signature == self._last_action_signature:
                    outcome["repeat_action"] = True
                self._last_action_signature = signature

                if action.action_type == "inspect_order":
                    result, error, details = self._handle_inspect_order(action)
                    outcome.update(details)
                elif action.action_type in {"process_refund", "reject_refund"}:
                    result, error, details = self._handle_refund_action(action)
                    outcome.update(details)
                elif action.action_type == "update_inventory":
                    result, error, details = self._handle_update_inventory(action)
                    outcome.update(details)
                elif action.action_type == "route_order":
                    result, error, details = self._handle_route_order(action)
                    outcome.update(details)
                elif action.action_type == "apply_substitute":
                    result, error, details = self._handle_apply_substitute(action)
                    outcome.update(details)
                elif action.action_type == "cancel_order":
                    result, error, details = self._handle_cancel_order(action)
                    outcome.update(details)
                elif action.action_type == "flag_dispute":
                    result, error, details = self._handle_flag_dispute(action)
                    outcome.update(details)
                elif action.action_type == "escalate_to_human":
                    result, error, details = self._handle_escalate_to_human(action)
                    outcome.update(details)
                elif action.action_type == "send_compensation":
                    result, error, details = self._handle_send_compensation(action)
                    outcome.update(details)
                else:
                    result = f"No transition handler for {action.action_type}"
                    error = "unsupported_action"
                    outcome["error"] = error

        if error:
            outcome["error"] = error
            result = f"Action failed: {error}"

        state.episode_done = state.step_count >= state.max_steps or self._is_task_resolved(state)

        action_payload = asdict(action)
        action_payload["_task_id"] = state.task_id
        action_payload["_step_count"] = state.step_count
        action_payload["_episode_id"] = state.episode_id

        reward_info = compute_step_reward(action=action_payload, outcome=outcome, ground_truth=state.ground_truth)
        reward_value = float(reward_info.get("reward", 0.0))
        state.cumulative_reward += reward_value
        final_error = error or reward_info.get("error") or outcome.get("error")

        return self._build_observation(
            task_description=self._task_description,
            last_action_result=result,
            last_action_error=final_error,
            reward=reward_value,
            reward_breakdown=reward_info.get("breakdown", {}),
        )

    def final_score(self) -> Dict[str, Any]:
        graded = grade_episode(self.state.task_id, self.state)
        score = float(graded.get("score", 0.0))
        score = max(0.0, min(1.0, score))
        return {"score": score, "breakdown": graded.get("breakdown", {})}

    def _build_observation(
        self,
        task_description: str = "",
        last_action_result: str = "",
        last_action_error: Optional[str] = None,
        reward: float = 0.0,
        reward_breakdown: Optional[Dict[str, Any]] = None,
    ) -> EcommerceObservation:
        state = self.state
        metadata: Dict[str, Any] = {
            "episode_id": state.episode_id,
            "step_count": state.step_count,
            "max_steps": state.max_steps,
            "cumulative_reward": state.cumulative_reward,
            "reward_breakdown": reward_breakdown or {},
        }
        if self._last_inspection is not None:
            metadata["inspected_order"] = self._last_inspection

        return EcommerceObservation(
            done=state.episode_done,
            reward=reward,
            metadata=metadata,
            open_tickets=[asdict(t) for t in state.tickets if t.status == "OPEN"],
            orders=[self._order_summary(order) for order in state.orders],
            inventory=[asdict(i) for i in state.inventory],
            last_action_result=last_action_result,
            last_action_error=last_action_error,
            task_description=task_description or self._task_description,
            task_id=state.task_id,
            steps_remaining=max(state.max_steps - state.step_count, 0),
        )

    def _missing_required_fields(self, action: EcommerceAction) -> List[str]:
        required = self.REQUIRED_FIELDS.get(action.action_type, ())
        missing: List[str] = []
        for field_name in required:
            value = getattr(action, field_name, None)
            if value is None:
                missing.append(field_name)
                continue
            if isinstance(value, str) and value.strip() == "":
                missing.append(field_name)
        return missing

    def _action_signature(self, action: EcommerceAction) -> Tuple[Any, ...]:
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

    def _handle_inspect_order(self, action: EcommerceAction) -> Tuple[str, Optional[str], Dict[str, Any]]:
        order = self._find_order(action.order_id)
        if order is None:
            self._last_inspection = None
            return (
                f"Order {action.order_id} not found.",
                "order_not_found",
                {"correct_target": False, "wrong_entity": True},
            )
        self._last_inspection = asdict(order)
        return (
            f"Inspected order {order.order_id}: {json.dumps(self._last_inspection, sort_keys=True)}",
            None,
            {"correct_target": True, "state_matches_ground_truth": True},
        )

    def _handle_refund_action(self, action: EcommerceAction) -> Tuple[str, Optional[str], Dict[str, Any]]:
        state = self.state
        ticket = self._find_ticket(action.ticket_id)
        order = self._find_order(action.order_id)
        if ticket is None:
            return (
                f"Ticket {action.ticket_id} not found.",
                "ticket_not_found",
                {"correct_target": False, "wrong_entity": True},
            )
        if order is None:
            return (
                f"Order {action.order_id} not found.",
                "order_not_found",
                {"correct_target": False, "wrong_entity": True},
            )
        if ticket.order_id != order.order_id:
            return (
                f"Ticket {ticket.ticket_id} is not linked to order {order.order_id}.",
                "ticket_order_mismatch",
                {"correct_target": False, "wrong_entity": True},
            )

        new_status = "REFUNDED" if action.action_type == "process_refund" else "REJECTED"
        ticket.status = new_status
        order.status = new_status
        order.touched = True

        expected = (state.ground_truth.get("ticket_decisions", {}) or {}).get(ticket.ticket_id)
        state_matches_gt = True
        if expected is not None:
            expected_status = "REFUNDED" if expected == "process_refund" else "REJECTED"
            state_matches_gt = ticket.status == expected_status

        return (
            f"{action.action_type} applied to ticket {ticket.ticket_id} and order {order.order_id}.",
            None,
            {"correct_target": True, "state_matches_ground_truth": state_matches_gt},
        )

    def _handle_update_inventory(self, action: EcommerceAction) -> Tuple[str, Optional[str], Dict[str, Any]]:
        state = self.state
        quantity = int(action.quantity if action.quantity is not None else 0)
        record = self._find_inventory(action.sku, action.warehouse)
        if record is None:
            record = InventoryRecord(sku=str(action.sku), warehouse=str(action.warehouse), quantity=quantity)
            state.inventory.append(record)
        else:
            record.quantity = quantity

        expected_quantity = self._expected_inventory_quantity(str(action.sku), str(action.warehouse))
        state_matches_gt = expected_quantity is None or expected_quantity == record.quantity

        details = {
            "correct_target": True,
            "state_matches_ground_truth": state_matches_gt,
        }
        if expected_quantity is not None and expected_quantity != record.quantity:
            details["wrong_inventory"] = True

        return (
            f"Inventory updated for {record.sku}@{record.warehouse} to {record.quantity}.",
            None,
            details,
        )

    def _handle_route_order(self, action: EcommerceAction) -> Tuple[str, Optional[str], Dict[str, Any]]:
        order = self._find_order(action.order_id)
        if order is None:
            return (
                f"Order {action.order_id} not found.",
                "order_not_found",
                {"correct_target": False, "wrong_entity": True},
            )

        order.warehouse = str(action.warehouse)
        order.status = "ROUTED"
        order.touched = True
        expected_warehouse = self._expected_order_route(order.order_id)
        state_matches_gt = expected_warehouse is None or expected_warehouse == order.warehouse
        return (
            f"Order {order.order_id} routed to {order.warehouse}.",
            None,
            {"correct_target": True, "state_matches_ground_truth": state_matches_gt},
        )

    def _handle_apply_substitute(self, action: EcommerceAction) -> Tuple[str, Optional[str], Dict[str, Any]]:
        order = self._find_order(action.order_id)
        if order is None:
            return (
                f"Order {action.order_id} not found.",
                "order_not_found",
                {"correct_target": False, "wrong_entity": True},
            )
        if not order.items:
            return (
                f"Order {order.order_id} has no line items to substitute.",
                "order_has_no_items",
                {"correct_target": True, "state_matches_ground_truth": False},
            )

        target_item = None
        for item in order.items:
            if item.status in {"CANCELLED", "AFFECTED"}:
                target_item = item
                break
        if target_item is None:
            target_item = order.items[0]

        target_item.substitute_sku = str(action.sku)
        target_item.status = "SUBSTITUTED"
        order.status = "RESOLVED"
        order.touched = True

        expected_resolution = self._expected_resolution(order.order_id)
        state_matches_gt = self._resolution_matches(order, expected_resolution, "apply_substitute")

        return (
            f"Applied substitute {action.sku} on order {order.order_id}.",
            None,
            {"correct_target": True, "state_matches_ground_truth": state_matches_gt},
        )

    def _handle_cancel_order(self, action: EcommerceAction) -> Tuple[str, Optional[str], Dict[str, Any]]:
        order = self._find_order(action.order_id)
        if order is None:
            return (
                f"Order {action.order_id} not found.",
                "order_not_found",
                {"correct_target": False, "wrong_entity": True},
            )

        for item in order.items:
            item.status = "CANCELLED"
        order.status = "CANCELLED"
        order.touched = True

        expected_resolution = self._expected_resolution(order.order_id)
        state_matches_gt = self._resolution_matches(order, expected_resolution, "cancel_order")
        destructive_cancel = self._is_destructive_cancel(expected_resolution)

        details: Dict[str, Any] = {
            "correct_target": True,
            "state_matches_ground_truth": state_matches_gt,
        }
        if destructive_cancel:
            details["destructive_cancel"] = True
            details["collateral_damage"] = True
            marker = f"destructive_cancel:{order.order_id}"
            if marker not in self.state.collateral_damage:
                self.state.collateral_damage.append(marker)

        return (
            f"Cancelled order {order.order_id}.",
            None,
            details,
        )

    def _handle_flag_dispute(self, action: EcommerceAction) -> Tuple[str, Optional[str], Dict[str, Any]]:
        ticket = self._find_ticket(action.ticket_id)
        if ticket is None:
            return (
                f"Ticket {action.ticket_id} not found.",
                "ticket_not_found",
                {"correct_target": False, "wrong_entity": True},
            )
        ticket.status = "DISPUTED"

        if action.order_id:
            order = self._find_order(action.order_id)
            if order is not None:
                order.status = "DISPUTED"
                order.touched = True

        return (
            f"Flagged dispute on ticket {ticket.ticket_id}.",
            None,
            {"correct_target": True, "state_matches_ground_truth": True},
        )

    def _handle_escalate_to_human(self, action: EcommerceAction) -> Tuple[str, Optional[str], Dict[str, Any]]:
        ticket = self._find_ticket(action.ticket_id)
        if ticket is None:
            return (
                f"Ticket {action.ticket_id} not found.",
                "ticket_not_found",
                {"correct_target": False, "wrong_entity": True},
            )
        ticket.status = "ESCALATED"

        state = self.state
        expected_escalations = set((state.ground_truth.get("escalations", []) or []))
        unnecessary = bool(expected_escalations) and ticket.ticket_id not in expected_escalations
        if unnecessary:
            state.unnecessary_escalations += 1

        return (
            f"Escalated ticket {ticket.ticket_id} to human.",
            None,
            {
                "correct_target": True,
                "state_matches_ground_truth": not unnecessary,
                "unnecessary_escalation": unnecessary,
            },
        )

    def _handle_send_compensation(self, action: EcommerceAction) -> Tuple[str, Optional[str], Dict[str, Any]]:
        order = self._find_order(action.order_id)
        if order is None:
            return (
                f"Order {action.order_id} not found.",
                "order_not_found",
                {"correct_target": False, "wrong_entity": True},
            )

        compensation = str(action.compensation_type)
        if compensation not in order.compensation:
            order.compensation.append(compensation)
        order.touched = True

        expected = self._expected_compensation(order.order_id)
        state_matches_gt = expected is None or compensation in expected
        return (
            f"Sent compensation {compensation} for order {order.order_id}.",
            None,
            {"correct_target": True, "state_matches_ground_truth": state_matches_gt},
        )

    def _expected_order_route(self, order_id: str) -> Optional[str]:
        routes = (self.state.ground_truth.get("routes", {}) or {})
        value = routes.get(order_id)
        if value is None:
            return None
        return str(value)

    def _expected_inventory_quantity(self, sku: str, warehouse: str) -> Optional[int]:
        inventory_gt = (self.state.ground_truth.get("inventory", {}) or {})
        key = f"{sku}@{warehouse}"
        if key in inventory_gt:
            try:
                return int(inventory_gt[key])
            except (TypeError, ValueError):
                return None
        nested = inventory_gt.get(sku)
        if isinstance(nested, dict) and warehouse in nested:
            try:
                return int(nested[warehouse])
            except (TypeError, ValueError):
                return None
        return None

    def _expected_resolution(self, order_id: str) -> Any:
        resolutions = (self.state.ground_truth.get("resolutions", {}) or {})
        return resolutions.get(order_id)

    def _expected_compensation(self, order_id: str) -> Optional[List[str]]:
        expected = self._expected_resolution(order_id)
        if isinstance(expected, dict):
            value = expected.get("compensation_type")
            if value is None:
                return None
            if isinstance(value, list):
                return [str(v) for v in value]
            return [str(value)]
        return None

    def _resolution_matches(self, order: OrderRecord, expected: Any, action_type: str) -> bool:
        if expected is None:
            return True
        if isinstance(expected, str):
            normalized = expected.strip().lower()
            if normalized in {"cancel", "cancel_order", "cancelled"}:
                return order.status == "CANCELLED"
            if normalized in {"apply_substitute", "substitute"}:
                return any(item.status == "SUBSTITUTED" for item in order.items)
            if normalized in {"send_compensation", "compensate"}:
                return len(order.compensation) > 0
            return normalized == action_type
        if isinstance(expected, dict):
            expected_action = expected.get("action") or expected.get("resolution")
            if action_type and expected_action and str(expected_action) != action_type:
                return False
            expected_status = expected.get("status")
            if expected_status and str(expected_status) != order.status:
                return False
            expected_warehouse = expected.get("warehouse")
            if expected_warehouse and str(expected_warehouse) != (order.warehouse or ""):
                return False
            expected_substitute = expected.get("substitute_sku")
            if expected_substitute and not any(item.substitute_sku == str(expected_substitute) for item in order.items):
                return False
            expected_compensation = expected.get("compensation_type")
            if expected_compensation:
                if isinstance(expected_compensation, list):
                    expected_values = {str(value) for value in expected_compensation}
                else:
                    expected_values = {str(expected_compensation)}
                if not expected_values.issubset(set(order.compensation)):
                    return False
            return True
        return False

    def _is_destructive_cancel(self, expected: Any) -> bool:
        if expected is None:
            return False
        if isinstance(expected, str):
            return expected.strip().lower() in {"apply_substitute", "substitute", "send_compensation", "partial"}
        if isinstance(expected, dict):
            action = str(expected.get("action") or expected.get("resolution") or "").strip().lower()
            status = str(expected.get("status") or "").strip().upper()
            if action in {"apply_substitute", "substitute", "send_compensation"}:
                return True
            if status and status != "CANCELLED":
                return True
        return False

    def _is_task_resolved(self, state: EcommerceState) -> bool:
        if state.task_id == "task_1":
            return all(ticket.status != "OPEN" for ticket in state.tickets)

        if state.task_id == "task_2":
            routes = (state.ground_truth.get("routes", {}) or {})
            inventory_gt = (state.ground_truth.get("inventory", {}) or {})

            if routes:
                for order_id, expected_warehouse in routes.items():
                    order = self._find_order(order_id)
                    if order is None or order.warehouse != str(expected_warehouse):
                        return False
            else:
                for order in state.orders:
                    if not order.warehouse:
                        return False

            if inventory_gt:
                for key, value in inventory_gt.items():
                    if isinstance(value, dict):
                        sku = str(key)
                        for warehouse, qty in value.items():
                            record = self._find_inventory(sku, str(warehouse))
                            if record is None or record.quantity != int(qty):
                                return False
                    else:
                        key_text = str(key)
                        if "@" not in key_text:
                            continue
                        sku, warehouse = key_text.split("@", 1)
                        record = self._find_inventory(sku, warehouse)
                        if record is None or record.quantity != int(value):
                            return False
            return True

        if state.task_id == "task_3":
            resolutions = (state.ground_truth.get("resolutions", {}) or {})
            if resolutions:
                for order_id, expected in resolutions.items():
                    order = self._find_order(order_id)
                    if order is None or not self._resolution_matches(order, expected, action_type=""):
                        return False
                return True

            open_tickets = any(ticket.status == "OPEN" for ticket in state.tickets)
            unresolved_orders = any(order.status in {"PENDING", "OPEN", "AFFECTED"} for order in state.orders)
            return not open_tickets and not unresolved_orders

        return False

    def _order_summary(self, order: OrderRecord) -> Dict[str, Any]:
        return {
            "order_id": order.order_id,
            "status": order.status,
            "customer_tier": order.customer_tier,
        }

    def _find_order(self, order_id: Optional[str]) -> Optional[OrderRecord]:
        if order_id is None:
            return None
        for order in self.state.orders:
            if order.order_id == order_id:
                return order
        return None

    def _find_ticket(self, ticket_id: Optional[str]) -> Optional[TicketRecord]:
        if ticket_id is None:
            return None
        for ticket in self.state.tickets:
            if ticket.ticket_id == ticket_id:
                return ticket
        return None

    def _find_inventory(self, sku: Optional[str], warehouse: Optional[str]) -> Optional[InventoryRecord]:
        if sku is None or warehouse is None:
            return None
        for record in self.state.inventory:
            if record.sku == sku and record.warehouse == warehouse:
                return record
        return None

    def _ensure_ticket_orders_exist(self) -> None:
        existing = {order.order_id for order in self.state.orders}
        for ticket in self.state.tickets:
            if ticket.order_id in existing:
                continue
            self.state.orders.append(
                OrderRecord(
                    order_id=ticket.order_id,
                    customer_id=ticket.customer_id or "",
                    customer_tier="standard",
                    status="PENDING",
                    items=[],
                    warehouse=None,
                    compensation=[],
                    touched=False,
                )
            )
            existing.add(ticket.order_id)

    def _build_state_snapshot(self) -> Dict[str, Any]:
        state = self.state
        return {
            "episode_id": state.episode_id,
            "task_id": state.task_id,
            "step_count": state.step_count,
            "max_steps": state.max_steps,
            "orders": [asdict(order) for order in state.orders],
            "tickets": [asdict(ticket) for ticket in state.tickets],
            "inventory": [asdict(record) for record in state.inventory],
            "collateral_damage": list(state.collateral_damage),
            "cumulative_reward": state.cumulative_reward,
            "episode_done": state.episode_done,
        }
