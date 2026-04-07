# ecommerce-ops-env

Deterministic OpenEnv-style e-commerce operations environment with 3 tasks:
`task_1` (refund queue), `task_2` (inventory reconciliation), `task_3` (supplier cancellation crisis).

## Project goal
Provide a stable benchmark loop (`reset`/`step`/`state`) that supports structured agent actions, partial observations, and grading via:
- `get_task_bundle(task_id)` from `server/tasks.py`
- `grade_episode(task_id, state)` from `server/grader.py`

## Run locally
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

Smoke checks:
```bash
curl http://localhost:7860/health
curl -X POST http://localhost:7860/reset -H "Content-Type: application/json" -d '{"task_id":"task_1"}'
curl -X POST http://localhost:7860/step -H "Content-Type: application/json" -d '{"action_type":"inspect_order","order_id":"O1"}'
```

## Endpoints
- `GET /health`
- `POST /reset` with body `{"task_id":"task_1|task_2|task_3"}`
- `POST /step` with `EcommerceAction`-shaped payload
- `GET /state`
- `GET /tasks`
- `POST /grader`
- `GET /baseline`

## Action / Observation / State summary
- `EcommerceAction`: `action_type` plus optional fields (`order_id`, `ticket_id`, `sku`, `warehouse`, `quantity`, `reason`, `compensation_type`).
- `EcommerceObservation`: partial by design; includes `done`, `reward`, `metadata`, `open_tickets`, order summaries, inventory snapshot, last action result/error, task info, and `steps_remaining`.
- `EcommerceState`: full internal state with records (`orders`, `tickets`, `inventory`), episode counters, task metadata, reward tracking, and ground truth.
- Full order detail is accessed via `inspect_order` (non-mutating).

## Baseline runner
Run all tasks sequentially with strict JSON action parsing:
```bash
API_BASE_URL=https://router.huggingface.co/v1 MODEL_NAME=gpt-4o-mini HF_TOKEN=your_token ENV_BASE_URL=http://localhost:7860 python inference.py
```

## Quick deployment note
Docker image runs on port `7860`:
```bash
docker build -t ecommerce-ops-env .
docker run --rm -p 7860:7860 ecommerce-ops-env
```

## Start with Docker (recommended)
1. Build image:
```bash
docker build -t ecommerce-ops-env .
```

2. Start container:
```bash
docker run --name ecommerce-ops-env --rm -p 7860:7860 ecommerce-ops-env
```

3. Verify server is up (new terminal):
```bash
curl http://localhost:7860/health
curl http://localhost:7860/tasks
```

4. Stop container:
- If running in foreground: press `Ctrl+C`.
- If running detached, start with:
```bash
docker run -d --name ecommerce-ops-env -p 7860:7860 ecommerce-ops-env
```
Then stop with:
```bash
docker stop ecommerce-ops-env
```

Optional: view logs while running detached:
```bash
docker logs -f ecommerce-ops-env
```
