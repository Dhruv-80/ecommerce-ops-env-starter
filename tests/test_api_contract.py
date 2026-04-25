import importlib

from server.app import app


def _client():
    for module_name in ("fastapi.testclient", "starlette.testclient"):
        try:
            module = importlib.import_module(module_name)
            return module.TestClient(app)
        except ModuleNotFoundError:
            continue
    raise RuntimeError("No compatible TestClient module found")


def test_health_endpoint_contract():
    client = _client()
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["ok"] is True
    assert body["status"] == "healthy"


def test_tasks_endpoint_contract():
    client = _client()
    response = client.get("/tasks")
    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == {"task_1", "task_2", "task_3"}


def test_reset_step_state_grader_flow_contract():
    client = _client()
    reset_response = client.post("/reset", json={"task_id": "task_1"})
    assert reset_response.status_code == 200
    reset_body = reset_response.json()
    assert reset_body["task_id"] == "task_1"
    assert reset_body["done"] is False
    assert "orders" in reset_body
    assert "warehouses" in reset_body

    step_response = client.post(
        "/step",
        json={"action_type": "assign_warehouse", "order_id": "O1", "warehouse_id": "W2"},
    )
    assert step_response.status_code == 200
    step_body = step_response.json()
    assert "reward" in step_body
    assert "last_action_result" in step_body
    assert "steps_remaining" in step_body

    state_response = client.get("/state")
    assert state_response.status_code == 200
    state_body = state_response.json()
    assert state_body["task_id"] == "task_1"
    assert "orders" in state_body
    assert "stock" in state_body

    grader_response = client.post("/grader")
    assert grader_response.status_code == 200
    grader_body = grader_response.json()
    assert "score" in grader_body
    assert "breakdown" in grader_body
    assert 0.0 <= float(grader_body["score"]) <= 1.0


def test_baseline_endpoint_contract():
    client = _client()
    response = client.get("/baseline")
    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == {"task_1", "task_2", "task_3"}
    for task_id in ("task_1", "task_2", "task_3"):
        task_result = body[task_id]
        assert "score" in task_result
        assert "breakdown" in task_result
        assert "steps" in task_result
        assert 0.0 <= float(task_result["score"]) <= 1.0
