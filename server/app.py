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
    # IMPLEMENT: call local inference workflow or self-loop baseline
    return {"task_1": 0.0, "task_2": 0.0, "task_3": 0.0}
