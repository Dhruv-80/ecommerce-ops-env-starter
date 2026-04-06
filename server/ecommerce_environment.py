from dataclasses import asdict
from typing import Any, Dict, Optional
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

    @property
    def state(self) -> EcommerceState:
        if self._state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        return self._state

    def reset(self, task_id: str = "task_1") -> EcommerceObservation:
        bundle = get_task_bundle(task_id)
        init = bundle["initial_state"]
        self._state = EcommerceState(
            episode_id=str(uuid4()),
            task_id=task_id,
            max_steps=bundle["max_steps"],
            orders=[OrderRecord(**o) for o in init.get("orders", [])],
            inventory=[InventoryRecord(**i) for i in init.get("inventory", [])],
            tickets=[TicketRecord(**t) for t in init.get("tickets", [])],
            ground_truth=bundle.get("ground_truth", {}),
        )
        return self._build_observation(task_description=bundle["task"]["description"])

    def step(self, action: EcommerceAction) -> EcommerceObservation:
        state = self.state
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

        # IMPLEMENT: real state transition logic here.
        if action.action_type == "inspect_order":
            outcome["valid_action"] = True
            outcome["correct_target"] = True
            result = f"Inspected order {action.order_id}"
        else:
            result = f"Stub step executed for {action.action_type}"

        reward_info = compute_step_reward(action=asdict(action), outcome=outcome, ground_truth=state.ground_truth)
        state.cumulative_reward += reward_info["reward"]
        state.episode_done = state.step_count >= state.max_steps

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
