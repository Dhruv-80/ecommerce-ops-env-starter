"""FastAPI / OpenEnv server for CommerceOps-Env.

Exposes the standard OpenEnv endpoints (reset, step, state) plus helpers
used for grading and demo (grader, baseline, tasks, metadata, schema).
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict

try:
    from ..environment import CommerceOpsEnv
    from ..models import EnvAction, EnvObservation
    from ..tasks import task_catalog
except ImportError:
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from environment import CommerceOpsEnv
    from models import EnvAction, EnvObservation
    from tasks import task_catalog

try:
    from openenv.core.env_server import create_fastapi_app as _openenv_create
except ImportError:
    _openenv_create = None


# ---------------------------------------------------------------------------
# Request/response models
# ---------------------------------------------------------------------------


class ResetRequest(BaseModel):
    task_id: str = "task_1"
    seed: int = 0


class StepRequest(BaseModel):
    """Mirrors ``EnvAction`` but with ``extra='ignore'`` so extra HTTP fields
    do not cause 422s on older clients."""

    model_config = ConfigDict(extra="ignore")

    action_type: str
    order_id: Optional[str] = None
    warehouse_id: Optional[str] = None
    quantity: Optional[int] = None
    allocations: Optional[List[Dict[str, Any]]] = None
    supplier_id: Optional[str] = None
    compensation_type: Optional[str] = None
    reason: Optional[str] = None

    def to_action_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.model_dump().items() if v is not None}


# ---------------------------------------------------------------------------
# App setup
# ---------------------------------------------------------------------------

env = CommerceOpsEnv()

app = FastAPI(
    title="CommerceOps-Env",
    version="2.0.0",
    description=(
        "OpenEnv-compatible fulfillment-judgment RL environment. "
        "An agent learns to make warehouse assignment and multi-order "
        "triage decisions under inventory scarcity and SLA pressure."
    ),
    contact={"name": "CommerceOps-Env Team"},
)

if _openenv_create is not None:
    try:
        _openenv_app = _openenv_create(
            CommerceOpsEnv,
            EnvAction,
            EnvObservation,
            env_name="commerce-ops-env",
            max_concurrent_envs=4,
        )
        app.mount("/openenv", _openenv_app)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Endpoint handlers
# ---------------------------------------------------------------------------


def health() -> Dict[str, Any]:
    return {"ok": True, "status": "healthy", "version": "2.0.0"}


def reset(req: Optional[ResetRequest] = None) -> Dict[str, Any]:
    task_id = req.task_id if req else "task_1"
    seed = req.seed if req else 0
    try:
        obs = env.reset(task_id=task_id, seed=seed)
        return obs.model_dump()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def step(action: StepRequest) -> Dict[str, Any]:
    try:
        obs = env.step(action.to_action_dict())
        return obs.model_dump()
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def state() -> Dict[str, Any]:
    try:
        return env.state_dict()
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def tasks() -> Dict[str, Dict[str, Any]]:
    return task_catalog()


def grader() -> Dict[str, Any]:
    try:
        result = env.final_score()
    except RuntimeError:
        env.reset("task_1", seed=0)
        result = env.final_score()
    return result


def metadata() -> Dict[str, Any]:
    return {
        "name": "commerce-ops-env",
        "version": "2.0.0",
        "description": (
            "Fulfillment-judgment RL environment where an LLM agent "
            "learns warehouse assignment and multi-order triage under "
            "inventory scarcity, SLA pressure, and customer-tier conflicts."
        ),
        "tasks": list(task_catalog().keys()),
        "model_recommendation": "Qwen2.5-3B-Instruct",
        "training_algorithm": "GRPO",
    }


def schema() -> Dict[str, Any]:
    return {
        "action": {
            "type": "object",
            "required": ["action_type"],
            "properties": {
                "action_type": {
                    "type": "string",
                    "enum": [
                        "assign_warehouse", "split_shipment", "delay_order",
                        "prioritize_order", "reroute_order", "escalate_supplier",
                        "refund_or_compensate", "noop",
                    ],
                },
                "order_id":          {"type": "string"},
                "warehouse_id":      {"type": "string"},
                "quantity":          {"type": "integer", "minimum": 1},
                "allocations":       {"type": "array",
                                      "items": {"type": "object",
                                                "properties": {
                                                    "warehouse_id": {"type": "string"},
                                                    "quantity": {"type": "integer"},
                                                }}},
                "supplier_id":       {"type": "string"},
                "compensation_type": {"type": "string"},
                "reason":            {"type": "string", "maxLength": 64},
            },
        },
        "observation": {
            "type": "object",
            "properties": {
                "task_id":            {"type": "string"},
                "task_type":          {"type": "string"},
                "episode_id":         {"type": "string"},
                "step":               {"type": "integer"},
                "max_steps":          {"type": "integer"},
                "steps_remaining":    {"type": "integer"},
                "done":               {"type": "boolean"},
                "reward":             {"type": "number"},
                "cumulative_reward":  {"type": "number"},
                "orders":             {"type": "array"},
                "warehouses":         {"type": "array"},
                "stock":              {"type": "array"},
                "allowed_actions":    {"type": "array"},
                "policy_flags":       {"type": "object"},
                "last_action_result": {"type": "string"},
                "last_action_error":  {"type": "string"},
                "reward_breakdown":   {"type": "object"},
                "task_description":   {"type": "string"},
            },
        },
    }


# ---------------------------------------------------------------------------
# Baseline runner — scripted oracle policy for demo and holdout checks
# ---------------------------------------------------------------------------


def _oracle_action_for_state(local_env: CommerceOpsEnv) -> Dict[str, Any]:
    """Return the next oracle action by following the ground-truth plan."""
    state = local_env.state
    gt = state.ground_truth
    plan = gt.get("plan", {})

    for order in state.orders:
        oid = order.order_id
        if order.status != "pending":
            continue
        exp = plan.get(oid)
        if exp is None:
            continue
        action = {"action_type": exp["action_type"], "order_id": oid}
        if exp["action_type"] == "assign_warehouse":
            action["warehouse_id"] = exp["warehouse_id"]
        elif exp["action_type"] == "split_shipment":
            action["allocations"] = exp["allocations"]
        elif exp["action_type"] == "delay_order":
            action["reason"] = "oracle_delay"
        return action

    # T1 fallback
    gt_kind = gt.get("kind", "")
    if gt_kind == "warehouse_assignment":
        best = gt.get("best_warehouse")
        oid = gt.get("order_id", "O1")
        if best:
            return {"action_type": "assign_warehouse", "order_id": oid, "warehouse_id": best}

    return {"action_type": "noop"}


def _run_oracle_for_task(task_id: str, seed: int = 0) -> Dict[str, Any]:
    local_env = CommerceOpsEnv()
    obs = local_env.reset(task_id=task_id, seed=seed)
    steps = 0
    while not obs.done and steps < local_env.state.max_steps:
        action = _oracle_action_for_state(local_env)
        obs = local_env.step(action)
        steps += 1
    result = local_env.final_score()
    return {**result, "steps_taken": steps}


def baseline() -> Dict[str, Any]:
    return {
        "task_1": _run_oracle_for_task("task_1"),
        "task_2": _run_oracle_for_task("task_2"),
        "task_3": _run_oracle_for_task("task_3"),
    }


# ---------------------------------------------------------------------------
# MCP JSON-RPC handler (pass-through for tool integrations)
# ---------------------------------------------------------------------------


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
                "serverInfo": {"name": "commerce-ops-env", "version": "2.0.0"},
                "capabilities": {"tools": {}},
            },
        })

    if method == "tools/list":
        tools: List[Dict[str, Any]] = [
            {"name": "reset", "description": "Reset to a task episode.",
             "inputSchema": {"type": "object",
                             "properties": {"task_id": {"type": "string"},
                                            "seed": {"type": "integer"}}}},
            {"name": "step", "description": "Submit one action.",
             "inputSchema": schema()["action"]},
        ]
        return JSONResponse({"jsonrpc": "2.0", "id": req_id, "result": {"tools": tools}})

    if method == "tools/call":
        params = body.get("params", {})
        tool_name = params.get("name", "")
        args = params.get("arguments", {})
        try:
            if tool_name == "reset":
                obs = env.reset(task_id=args.get("task_id", "task_1"),
                                seed=int(args.get("seed", 0)))
                return JSONResponse({"jsonrpc": "2.0", "id": req_id,
                                     "result": {"content": [{"type": "text",
                                                              "text": obs.model_dump_json()}]}})
            if tool_name == "step":
                obs = env.step(args)
                return JSONResponse({"jsonrpc": "2.0", "id": req_id,
                                     "result": {"content": [{"type": "text",
                                                              "text": obs.model_dump_json()}]}})
        except Exception as exc:
            return JSONResponse({"jsonrpc": "2.0", "id": req_id,
                                 "error": {"code": -32000, "message": str(exc)}})

    return JSONResponse({"jsonrpc": "2.0", "id": req_id,
                         "error": {"code": -32601, "message": f"Method not found: {method}"}})


# ---------------------------------------------------------------------------
# Route registration
# ---------------------------------------------------------------------------


_ROUTES = [
    ("/health",   health,   ["GET"],  "System",      "Health check"),
    ("/reset",    reset,    ["POST"], "Episode",     "Reset environment episode"),
    ("/step",     step,     ["POST"], "Episode",     "Apply one action"),
    ("/state",    state,    ["GET"],  "Episode",     "Get internal state"),
    ("/tasks",    tasks,    ["GET"],  "Catalog",     "List available tasks"),
    ("/grader",   grader,   ["POST"], "Evaluation",  "Grade current episode"),
    ("/baseline", baseline, ["GET"],  "Evaluation",  "Run oracle baseline on all tasks"),
    ("/metadata", metadata, ["GET"],  "System",      "Environment metadata"),
    ("/schema",   schema,   ["GET"],  "System",      "Action/observation schema"),
    ("/mcp",      mcp_handler, ["POST"], "MCP",      "MCP JSON-RPC endpoint"),
]

for _path, _fn, _methods, _tag, _summary in _ROUTES:
    app.add_api_route(_path, _fn, methods=_methods, tags=[_tag], summary=_summary)


def main() -> None:
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)


if __name__ == "__main__":
    main()
