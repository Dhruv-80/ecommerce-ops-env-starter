from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict

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


class StepRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    action_type: str
    order_id: Optional[str] = None
    ticket_id: Optional[str] = None
    sku: Optional[str] = None
    warehouse: Optional[str] = None
    quantity: Optional[int] = None
    reason: Optional[str] = None
    compensation_type: Optional[str] = None

    def to_action(self) -> EcommerceAction:
        return EcommerceAction(
            action_type=self.action_type,
            order_id=self.order_id,
            ticket_id=self.ticket_id,
            sku=self.sku,
            warehouse=self.warehouse,
            quantity=self.quantity,
            reason=self.reason,
            compensation_type=self.compensation_type,
        )


env = EcommerceEnvironment()

app = FastAPI(
    title="Ecommerce Ops Environment API",
    version="1.0.0",
    description=(
        "Deterministic OpenEnv-style ecommerce operations environment with multi-step tasks for "
        "refund processing, inventory reconciliation, and supplier-cancellation crisis handling."
    ),
    contact={"name": "Ecommerce Ops Env Team"},
)
if create_fastapi_app is not None:
    try:
        openenv_app = create_fastapi_app(
            EcommerceEnvironment,
            EcommerceAction,
            EcommerceObservation,
            env_name="ecommerce-ops-env",
            max_concurrent_envs=4,
        )
        app.mount("/openenv", openenv_app)
    except Exception:
        # Keep local fallback app operational even if OpenEnv wiring fails at import/runtime.
        openenv_app = None


def health() -> Dict[str, Any]:
    return {"ok": True, "status": "healthy"}


def reset(req: Optional[ResetRequest] = None) -> EcommerceObservation:
    task_id = req.task_id if req is not None else "task_1"
    try:
        return env.reset(task_id=task_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def step(action: StepRequest) -> EcommerceObservation:
    try:
        return env.step(action.to_action())
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def state() -> Dict[str, Any]:
    try:
        return env.state.to_dict()
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def tasks() -> Dict[str, Dict[str, Any]]:
    return task_catalog()


def grader() -> Dict[str, Any]:
    try:
        result = env.final_score()
    except RuntimeError:
        env.reset(task_id="task_1")
        result = env.final_score()
    score = max(0.0, min(1.0, float(result.get("score", 0.0))))
    return {"score": score, "breakdown": result.get("breakdown", {})}


def _first_open_ticket(local_env: EcommerceEnvironment):
    return next((ticket for ticket in local_env.state.tickets if ticket.status == "OPEN"), None)


def _first_order_needing_route(local_env: EcommerceEnvironment):
    for order in local_env.state.orders:
        if not order.warehouse or order.status in {"PENDING", "OPEN", "UNROUTED"}:
            return order
    return None


def _best_warehouse(local_env: EcommerceEnvironment) -> str:
    if not local_env.state.inventory:
        return "W1"
    # Deterministic tie-break: highest quantity, then warehouse name.
    sorted_inventory = sorted(local_env.state.inventory, key=lambda row: (-row.quantity, row.warehouse))
    return sorted_inventory[0].warehouse


def _task_policy(local_env: EcommerceEnvironment) -> EcommerceAction:
    task_id = local_env.state.task_id

    if task_id == "task_1":
        ticket = _first_open_ticket(local_env)
        if ticket is not None:
            action_type = "process_refund" if ticket.created_days_ago <= 30 else "reject_refund"
            return EcommerceAction(
                action_type=action_type,
                ticket_id=ticket.ticket_id,
                order_id=ticket.order_id,
                reason=f"policy_{ticket.created_days_ago}_days",
            )
        if local_env.state.orders:
            return EcommerceAction(action_type="inspect_order", order_id=local_env.state.orders[0].order_id)
        return EcommerceAction(action_type="escalate_to_human", ticket_id="UNKNOWN", reason="no_open_tickets")

    if task_id == "task_2":
        order = _first_order_needing_route(local_env)
        if order is not None:
            return EcommerceAction(
                action_type="route_order",
                order_id=order.order_id,
                warehouse=_best_warehouse(local_env),
            )
        if local_env.state.orders:
            return EcommerceAction(action_type="inspect_order", order_id=local_env.state.orders[0].order_id)
        return EcommerceAction(action_type="update_inventory", sku="SKU-DEFAULT", warehouse="W1", quantity=0)

    if task_id == "task_3":
        for order in local_env.state.orders:
            if any(item.status in {"AFFECTED", "CANCELLED"} for item in order.items):
                target_item = next((item for item in order.items if item.status in {"AFFECTED", "CANCELLED"}), order.items[0])
                return EcommerceAction(action_type="apply_substitute", order_id=order.order_id, sku=f"SUB-{target_item.sku}")
            if not order.compensation:
                tier = order.customer_tier.lower()
                comp = "coupon_10"
                if tier == "premium":
                    comp = "coupon_20"
                elif tier == "loyalty":
                    comp = "coupon_30"
                return EcommerceAction(action_type="send_compensation", order_id=order.order_id, compensation_type=comp)
        ticket = _first_open_ticket(local_env)
        if ticket is not None:
            return EcommerceAction(action_type="escalate_to_human", ticket_id=ticket.ticket_id, reason="complex_case")
        if local_env.state.orders:
            return EcommerceAction(action_type="inspect_order", order_id=local_env.state.orders[0].order_id)
        return EcommerceAction(action_type="cancel_order", order_id="UNKNOWN", reason="fallback")

    return EcommerceAction(action_type="inspect_order", order_id="UNKNOWN")


def _run_baseline_for_task(task_id: str) -> Dict[str, Any]:
    local_env = EcommerceEnvironment()
    obs = local_env.reset(task_id=task_id)
    total_reward = 0.0
    steps = 0

    while not obs.done and steps < local_env.state.max_steps:
        action = _task_policy(local_env)
        obs = local_env.step(action)
        total_reward += obs.reward
        steps += 1

    grade = local_env.final_score()
    score = max(0.0, min(1.0, float(grade.get("score", 0.0))))
    return {
        "score": score,
        "breakdown": grade.get("breakdown", {}),
        "steps": steps,
        "total_reward": round(total_reward, 4),
    }


def baseline() -> Dict[str, Dict[str, Any]]:
    return {
        "task_1": _run_baseline_for_task("task_1"),
        "task_2": _run_baseline_for_task("task_2"),
        "task_3": _run_baseline_for_task("task_3"),
    }


def metadata() -> Dict[str, Any]:
    return {
        "name": "ecommerce-ops-env",
        "description": (
            "E-commerce operations environment where an AI agent resolves "
            "refund queues, inventory discrepancies, and supplier cancellation crises."
        ),
        "version": "0.1.0",
        "tasks": list(task_catalog().keys()),
    }


def schema() -> Dict[str, Any]:
    return {
        "action": {
            "type": "object",
            "properties": {
                "action_type": {"type": "string"},
                "order_id": {"type": "string"},
                "ticket_id": {"type": "string"},
                "sku": {"type": "string"},
                "warehouse": {"type": "string"},
                "quantity": {"type": "integer"},
                "reason": {"type": "string"},
                "compensation_type": {"type": "string"},
            },
            "required": ["action_type"],
        },
        "observation": {
            "type": "object",
            "properties": {
                "done": {"type": "boolean"},
                "reward": {"type": "number"},
                "metadata": {"type": "object"},
                "open_tickets": {"type": "array"},
                "orders": {"type": "array"},
                "inventory": {"type": "array"},
                "last_action_result": {"type": "string"},
                "last_action_error": {"type": "string"},
                "task_description": {"type": "string"},
                "task_id": {"type": "string"},
                "steps_remaining": {"type": "integer"},
            },
        },
        "state": {
            "type": "object",
            "properties": {
                "episode_id": {"type": "string"},
                "task_id": {"type": "string"},
                "step_count": {"type": "integer"},
                "max_steps": {"type": "integer"},
                "episode_done": {"type": "boolean"},
                "cumulative_reward": {"type": "number"},
            },
        },
    }


async def mcp_handler(request: Request) -> JSONResponse:
    try:
        body = await request.json()
    except Exception:
        body = {}
    req_id = body.get("id", 1)
    method = body.get("method", "")

    if method == "initialize":
        return JSONResponse({
            "jsonrpc": "2.0", "id": req_id,
            "result": {
                "protocolVersion": "2024-11-05",
                "serverInfo": {"name": "ecommerce-ops-env", "version": "0.1.0"},
                "capabilities": {"tools": {}},
            },
        })

    if method == "tools/list":
        tools: List[Dict[str, Any]] = [
            {
                "name": "reset",
                "description": "Reset the environment to a given task.",
                "inputSchema": {
                    "type": "object",
                    "properties": {"task_id": {"type": "string", "default": "task_1"}},
                },
            },
            {
                "name": "step",
                "description": "Take an action in the environment.",
                "inputSchema": schema()["action"],
            },
        ]
        return JSONResponse({"jsonrpc": "2.0", "id": req_id, "result": {"tools": tools}})

    if method == "tools/call":
        params = body.get("params", {})
        tool_name = params.get("name", "")
        args = params.get("arguments", {})
        try:
            if tool_name == "reset":
                obs = env.reset(task_id=args.get("task_id", "task_1"))
                return JSONResponse({"jsonrpc": "2.0", "id": req_id, "result": {"content": [{"type": "text", "text": str(obs.to_dict())}]}})
            if tool_name == "step":
                action = EcommerceAction(
                    action_type=args.get("action_type", ""),
                    order_id=args.get("order_id"),
                    ticket_id=args.get("ticket_id"),
                    sku=args.get("sku"),
                    warehouse=args.get("warehouse"),
                    quantity=args.get("quantity"),
                    reason=args.get("reason"),
                    compensation_type=args.get("compensation_type"),
                )
                obs = env.step(action)
                return JSONResponse({"jsonrpc": "2.0", "id": req_id, "result": {"content": [{"type": "text", "text": str(obs.to_dict())}]}})
        except Exception as exc:
            return JSONResponse({"jsonrpc": "2.0", "id": req_id, "error": {"code": -32000, "message": str(exc)}})

    return JSONResponse({"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": f"Method not found: {method}"}})


app.add_api_route(
    "/health",
    health,
    methods=["GET"],
    tags=["System"],
    summary="Health check",
    description="Returns service liveness status.",
    response_description="Liveness response.",
)
app.add_api_route(
    "/reset",
    reset,
    methods=["POST"],
    tags=["Episode"],
    summary="Reset environment episode",
    description="Starts a new deterministic episode for the selected task.",
    response_description="Initial observation for the selected task.",
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "example": {"task_id": "task_1"},
                }
            }
        }
    },
)
app.add_api_route(
    "/step",
    step,
    methods=["POST"],
    tags=["Episode"],
    summary="Apply one action",
    description="Applies one environment action and returns the next observation with reward.",
    response_description="Next observation after action execution.",
    openapi_extra={
        "requestBody": {
            "content": {
                "application/json": {
                    "example": {
                        "action_type": "inspect_order",
                        "order_id": "O1",
                    },
                }
            }
        }
    },
)
app.add_api_route(
    "/state",
    state,
    methods=["GET"],
    tags=["Episode"],
    summary="Get internal state",
    description="Returns full internal environment state for debugging and grading inspection.",
    response_description="Current internal state.",
)
app.add_api_route(
    "/tasks",
    tasks,
    methods=["GET"],
    tags=["Catalog"],
    summary="List available tasks",
    description="Returns task metadata and difficulty information.",
    response_description="Task catalog.",
)
app.add_api_route(
    "/grader",
    grader,
    methods=["POST"],
    tags=["Evaluation"],
    summary="Grade current episode",
    description="Computes a normalized score and breakdown for the current episode state.",
    response_description="Episode score and breakdown.",
)
app.add_api_route(
    "/baseline",
    baseline,
    methods=["GET"],
    tags=["Evaluation"],
    summary="Run baseline policy",
    description="Runs deterministic baseline policy on all tasks and returns task-level scores.",
    response_description="Baseline results across task_1, task_2, and task_3.",
)
app.add_api_route(
    "/metadata",
    metadata,
    methods=["GET"],
    tags=["System"],
    summary="Environment metadata",
    description="Returns environment metadata used by integration layers.",
)
app.add_api_route(
    "/schema",
    schema,
    methods=["GET"],
    tags=["System"],
    summary="Action/observation/state schema",
    description="Returns high-level JSON schema for action, observation, and state payloads.",
)
app.add_api_route(
    "/mcp",
    mcp_handler,
    methods=["POST"],
    tags=["MCP"],
    summary="MCP JSON-RPC endpoint",
    description="Implements MCP methods initialize, tools/list, and tools/call for reset and step tools.",
)


def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
