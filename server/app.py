from fastapi import FastAPI
from pydantic import BaseModel

try:
    from ..models import EcommerceAction, EcommerceObservation
    from .ecommerce_environment import EcommerceEnvironment
    from .tasks import task_catalog
except ImportError:
    from models import EcommerceAction, EcommerceObservation
    from server.ecommerce_environment import EcommerceEnvironment
    from server.tasks import task_catalog

try:
    from openenv.core.env_server import create_fastapi_app
except ImportError:
    create_fastapi_app = None


class ResetRequest(BaseModel):
    task_id: str = "task_1"


env = EcommerceEnvironment()

if create_fastapi_app is not None:
    app = create_fastapi_app(
        EcommerceEnvironment,
        EcommerceAction,
        EcommerceObservation,
        env_name="ecommerce-ops-env",
        max_concurrent_envs=4,
    )
else:
    app = FastAPI(title="ecommerce-ops-env")


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/reset")
def reset(req: ResetRequest):
    return env.reset(task_id=req.task_id)


@app.post("/step")
def step(action: EcommerceAction):
    return env.step(action)


@app.get("/state")
def state():
    return env.state


@app.get("/tasks")
def tasks():
    return task_catalog()


@app.post("/grader")
def grader():
    return env.final_score()


@app.get("/baseline")
def baseline():
    baseline_env = EcommerceEnvironment()
    results = {}

    for task_id in ["task_1", "task_2", "task_3"]:
        baseline_env.reset(task_id=task_id)
        state = baseline_env.state

        if task_id == "task_1":
            for ticket in list(state.tickets):
                action_type = "process_refund" if ticket.created_days_ago <= 30 else "reject_refund"
                baseline_env.step(
                    EcommerceAction(
                        action_type=action_type,
                        order_id=ticket.order_id,
                        ticket_id=ticket.ticket_id,
                        reason="policy_window_check",
                    )
                )
                if baseline_env.state.episode_done:
                    break

        if task_id == "task_2":
            gt_inventory = state.ground_truth.get("inventory", {})
            for sku_wh, qty in gt_inventory.items():
                sku, warehouse = sku_wh.split("|")
                baseline_env.step(
                    EcommerceAction(
                        action_type="update_inventory",
                        sku=sku,
                        warehouse=warehouse,
                        quantity=qty,
                    )
                )
                if baseline_env.state.episode_done:
                    break

            if not baseline_env.state.episode_done:
                gt_routes = state.ground_truth.get("routes", {})
                for order_id, warehouse in gt_routes.items():
                    baseline_env.step(
                        EcommerceAction(
                            action_type="route_order",
                            order_id=order_id,
                            warehouse=warehouse,
                        )
                    )
                    if baseline_env.state.episode_done:
                        break

        if task_id == "task_3":
            resolutions = state.ground_truth.get("resolutions", {})
            for order_id, expected in resolutions.items():
                expected_action = expected.get("expected_action")
                if expected_action == "apply_substitute":
                    baseline_env.step(
                        EcommerceAction(
                            action_type="apply_substitute",
                            order_id=order_id,
                            sku=expected.get("substitute_sku"),
                        )
                    )
                elif expected_action == "cancel_order":
                    baseline_env.step(
                        EcommerceAction(
                            action_type="cancel_order",
                            order_id=order_id,
                            reason="supplier_cancelled_sku",
                        )
                    )

                compensation = expected.get("compensation")
                if compensation and not baseline_env.state.episode_done:
                    baseline_env.step(
                        EcommerceAction(
                            action_type="send_compensation",
                            order_id=order_id,
                            compensation_type=compensation,
                        )
                    )

                if baseline_env.state.episode_done:
                    break

        grade = baseline_env.final_score()
        results[task_id] = {
            "score": grade.get("score", 0.0),
            "breakdown": grade.get("breakdown", {}),
        }

    return results
