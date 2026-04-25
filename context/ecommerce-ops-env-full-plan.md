# Ecommerce Ops Env — Full Build Plan and Architecture

This document is the full implementation plan for the `ecommerce-ops-env` project. It is written as the build reference for two teammates working in parallel inside an IDE such as Antigravity or Codex.

## Project goal

Build an OpenEnv-style e-commerce operations environment where an agent solves realistic multi-step business tasks by taking structured actions over several turns. The environment should expose `reset`, `step`, and `state` behavior in the OpenEnv style, and Round 1 success depends primarily on runtime correctness, interface compliance, clear task design, and sensible grading logic.[cite:5][cite:6]

## Core idea

The environment simulates an operations desk inside an e-commerce company. The agent receives partially visible operational problems, decides the next action, changes the environment state, and receives reward based on whether it chose the correct action, targeted the correct entity, and avoided damaging unrelated state.[cite:6][cite:22][cite:76]

The three tasks are intentionally arranged as an easy-to-hard ladder:

1. **Task 1 — Refund Queue Processing**
2. **Task 2 — Inventory Reconciliation**
3. **Task 3 — Supplier Cancellation Crisis**

This makes the benchmark useful because weak agents should score on Task 1, decent agents should partially solve Task 2, and stronger agents should show a meaningful gap on Task 3.[cite:22][cite:76]

## Round 1 build strategy

For Round 1, the priority is not maximum complexity. The priority is a working, compliant, reproducible environment that deploys cleanly and returns valid outputs.[cite:5]

That means this project should be built in the following order:

1. Freeze schemas.
2. Freeze tasks and ground truth.
3. Freeze graders.
4. Implement environment transitions.
5. Expose API endpoints.
6. Make Docker build and deploy.
7. Add the baseline inference loop.

## Final architecture decisions

### 1. Seeds

Use **fixed seeds only**. Deterministic task generation is the safest choice because the grader and baseline should be reproducible from run to run.[cite:5]

### 2. Step budgets

Use these budgets:

| Task | Budget | Why |
|---|---:|---|
| Task 1 | 10 | Enough for 5 tickets plus a little slack. |
| Task 2 | 15 | Gives room to reconcile inventory and route orders without making one mistake fatal. |
| Task 3 | 18 | Leaves some room for inspection and mixed-resolution logic. |

### 3. Observation design

Use **partial observation**. The agent should always see:

- Open tickets
- Summary order list (`order_id`, `status`, `customer_tier`)
- Current inventory snapshot
- Last action result and error
- Task description
- Steps remaining

The agent should **not** automatically see full order-item detail. It must use `inspect_order` to retrieve that detail. Partial observability is more aligned with realistic agent evaluation than exposing the full internal state every turn.[cite:22][cite:76]

## File structure

Use this repository layout:

```text
ecommerce-ops-env/
├── inference.py
├── README.md
├── TEAM_SPLIT.md
├── __init__.py
├── models.py
├── client.py
├── openenv.yaml
├── pyproject.toml
├── requirements.txt
├── Dockerfile
├── .env.example
├── server/
│   ├── __init__.py
│   ├── app.py
│   ├── ecommerce_environment.py
│   ├── tasks.py
│   ├── reward.py
│   └── grader.py
└── tests/
    └── test_graders.py
```

## What each file does

| File | Purpose |
|---|---|
| `models.py` | Dataclasses for action, observation, state, and record types. |
| `client.py` | Simple HTTP client for local testing. |
| `inference.py` | Baseline script that runs all tasks and prints step traces. |
| `openenv.yaml` | OpenEnv metadata and app entry reference. |
| `Dockerfile` | Builds the runnable environment image. |
| `server/app.py` | FastAPI app plus custom endpoints. |
| `server/ecommerce_environment.py` | `reset`, `step`, and state-handling logic. |
| `server/tasks.py` | Fixed task scenarios and ground truth. |
| `server/reward.py` | Step-level reward computation. |
| `server/grader.py` | Final task scoring logic. |
| `tests/test_graders.py` | Smoke tests and grader correctness tests. |

## Domain model

### Core record types

The internal state is built from these record types:

- `OrderItem`
- `OrderRecord`
- `TicketRecord`
- `InventoryRecord`

### EcommerceAction

The action model should support these fields:

- `action_type`
- `order_id`
- `ticket_id`
- `sku`
- `warehouse`
- `quantity`
- `reason`
- `compensation_type`

### EcommerceObservation

The observation should contain:

- `done`
- `reward`
- `metadata`
- `open_tickets`
- `orders`
- `inventory`
- `last_action_result`
- `last_action_error`
- `task_description`
- `task_id`
- `steps_remaining`

### EcommerceState

The full internal state should contain:

- `episode_id`
- `step_count`
- `task_id`
- `max_steps`
- `orders`
- `inventory`
- `tickets`
- `products`
- `resolved_correctly`
- `resolved_incorrectly`
- `collateral_damage`
- `unnecessary_escalations`
- `cumulative_reward`
- `episode_done`
- `ground_truth`

## Action space

The environment supports the following actions:

| Action | Required fields | Effect |
|---|---|---|
| `process_refund` | `order_id`, `ticket_id`, `reason` | Refunds the order and resolves the ticket. |
| `reject_refund` | `order_id`, `ticket_id`, `reason` | Rejects the refund request and resolves the ticket. |
| `update_inventory` | `sku`, `warehouse`, `quantity` | Updates reconciled inventory. |
| `route_order` | `order_id`, `warehouse` | Assigns an order to a warehouse and moves processing forward. |
| `apply_substitute` | `order_id`, `sku` | Replaces a cancelled item with an approved substitute. |
| `cancel_order` | `order_id`, `reason` | Cancels the full order. |
| `flag_dispute` | `order_id`, `ticket_id` | Marks a dispute case. |
| `escalate_to_human` | `ticket_id`, `reason` | Sends unresolved cases to a human. |
| `send_compensation` | `order_id`, `compensation_type` | Logs compensation for the customer. |
| `inspect_order` | `order_id` | Returns full order details without mutating state. |

## Task designs

### Task 1 — Refund Queue Processing

**Goal:** process 5 refund tickets using a simple explicit policy.

**Policy:**
- Within 30 days → approve refund
- Outside 30 days → reject refund

**Why it is easy:** every ticket is independent and has a one-to-one correct decision.

**Grader:** compare each ticket decision against ground truth.

**Expected difficulty:** a competent baseline should do well because the logic is explicit and local.

### Task 2 — Inventory Reconciliation

**Goal:** reconcile conflicting inventory across warehouses, then route 8 orders correctly.

**Required sequence:**
1. Reconcile true inventory.
2. Update the listing state.
3. Route each order to a warehouse with stock.

**Why it is medium:** errors in reconciliation cascade into routing mistakes.

**Grader:** average of:
- inventory accuracy score
- routing accuracy score

### Task 3 — Supplier Cancellation Crisis

**Goal:** resolve orders affected by a cancelled SKU while preserving unaffected order state.

**Rules:**
- Some orders are partially shipped.
- Customer tiers affect compensation.
- Some orders contain mixed SKUs, so not every affected order should be fully cancelled.
- Touching unaffected orders should be penalized.

**Why it is hard:** it requires state tracking, conditional logic, selective action, and collateral-damage avoidance.

**Tier weights:**

| Tier | Weight |
|---|---:|
| standard | 1.0 |
| premium | 1.5 |
| loyalty | 2.0 |

**Grader:** weighted per-order resolution score minus collateral-damage penalty.

## Reward design

Use a simple but asymmetric reward structure so random guessing cannot score well.

### Positive rewards

| Condition | Reward |
|---|---:|
| Correct action type | +0.15 |
| Correct target entity | +0.20 |
| Final state matches ground truth | +0.25 |
| No collateral damage | +0.15 |
| Within step budget | +0.10 |

### Penalties

| Condition | Penalty |
|---|---:|
| Wrong entity targeted | -0.30 |
| Wrong inventory update | -0.20 |
| Unnecessary escalation | -0.15 |
| Destructive cancel when substitute was correct | -0.25 |
| Repeated identical action on same entity | -0.10 |

### Episode score

Use normalized episode scoring:

```text
score = sum(step_rewards) / max_possible_reward
score = clamp(score, 0.0, 1.0)
```

This kind of shaped reward is consistent with how multi-step agent environments use intermediate signals to guide behavior, as long as the shaping still points toward the same final objective.[cite:22][cite:76]

## State transition design

### reset(task_id)

`reset` should:

1. Load the fixed task bundle.
2. Create a new `episode_id`.
3. Convert bundle records into state objects.
4. Set `step_count = 0`.
5. Set `episode_done = False`.
6. Return the first observation.

### step(action)

`step` should:

1. Validate the action structure.
2. Validate required fields for the action type.
3. Apply the state transition.
4. Compute the immediate reward and breakdown.
5. Increment `step_count`.
6. Mark `episode_done` if terminal conditions are met.
7. Return the new observation.

### Terminal conditions

An episode should end when either:

- `step_count >= max_steps`, or
- all target entities for the task are resolved.

## API design

### Required practical endpoints

For this project, expose these endpoints:

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Basic liveness check. |
| `POST` | `/reset` | Start a new task episode. |
| `POST` | `/step` | Apply one action. |
| `GET` | `/state` | Return full internal state. |
| `GET` | `/tasks` | Return task metadata and action schema. |
| `POST` | `/grader` | Return final score and breakdown. |
| `GET` | `/baseline` | Run or trigger baseline scoring for all tasks. |

### `/grader` response contract

Use a stable response shape:

```json
{
  "score": 0.82,
  "breakdown": {
    "correct": 4,
    "total": 5
  }
}
```

For Task 2 and Task 3, the `breakdown` object can differ, but every response must include a top-level `score` in `[0.0, 1.0]`.

## Baseline inference design

The baseline script should:

1. Read `API_BASE_URL`, `MODEL_NAME`, and `HF_TOKEN` from environment variables.
2. Run all 3 tasks sequentially.
3. Print logs in the required pattern:
   - `[START]`
   - `[STEP]`
   - `[END]`
4. Use a strict system prompt.
5. Parse model output robustly.

### System prompt behavior

The prompt should instruct the model to:

- output exactly one JSON action
- never output markdown
- use `inspect_order` when details are missing
- avoid guessing because wrong entity targeting is heavily penalized

### Parser behavior

The parser should try, in this order:

1. direct JSON parse
2. XML-wrapped action extraction
3. first JSON object found in raw text
4. fallback to invalid-action handling

Robust action parsing matters in multi-step agent systems because failures often come from malformed tool calls rather than purely wrong reasoning.[cite:22][cite:76]

## Teammate split

## Teammate A — Task, reward, grader side

**Own these files:**

- `server/tasks.py`
- `server/reward.py`
- `server/grader.py`
- `tests/test_graders.py`

**Responsibilities:**

1. Define fixed scenarios for all three tasks.
2. Encode ground truth.
3. Implement deterministic reward logic.
4. Implement final graders.
5. Add tests for:
   - wrong target
   - duplicate action
   - collateral damage
   - full-credit path
   - partial-credit path

**Must not change without coordination:**

- task IDs
- grader return keys
- ground-truth field names used by environment logic

## Teammate B — Model, environment, API, deployment side

**Own these files:**

- `models.py`
- `client.py`
- `server/ecommerce_environment.py`
- `server/app.py`
- `inference.py`
- `openenv.yaml`
- `Dockerfile`
- `requirements.txt`
- `pyproject.toml`
- `README.md`

**Responsibilities:**

1. Freeze dataclasses.
2. Implement `reset`, `step`, and `state` plumbing.
3. Expose HTTP endpoints.
4. Implement baseline inference loop.
5. Make local run and deployment work.

**Must not change without coordination:**

- dataclass field names
- endpoint names
- expected request/response schema for `/grader`

## Shared merge contract

Both teammates must agree on these interfaces before diverging:

### Contract A — task bundle

`get_task_bundle(task_id)` should return:

```python
{
  "task": {...},
  "initial_state": {...},
  "ground_truth": {...},
  "max_steps": 10
}
```

### Contract B — reward call

`compute_step_reward(...)` should return:

```python
{
  "reward": 0.15,
  "breakdown": {...},
  "error": None
}
```

### Contract C — final grading

`grade_episode(task_id, state)` should return:

```python
{
  "score": 0.75,
  "breakdown": {...}
}
```

If these three contracts remain stable, both teammates can work independently with minimal merge pain.

## Development order

### Phase 1 — Schema freeze

- Lock `models.py`
- Lock task IDs
- Lock action names
- Lock endpoint names

### Phase 2 — Ground truth freeze

- Finish `server/tasks.py`
- Finish all task bundles
- Finish all grader expectations

### Phase 3 — Logic build

- Implement reward logic
- Implement graders
- Implement environment state transitions

### Phase 4 — API and baseline

- Finish app endpoints
- Finish inference script
- Run local smoke tests

### Phase 5 — Deployment and validation

- Build Docker image
- Run local server
- Deploy a skeleton early
- Confirm `/health`, `/reset`, `/step`, `/state`, `/tasks`, `/grader`, `/baseline`

## Local development commands

Use a minimal local flow:

```bash
pip install -r requirements.txt
pytest
uvicorn server.app:app --reload --host 0.0.0.0 --port 7860
```

Useful smoke checks:

```bash
curl http://localhost:7860/health
curl http://localhost:7860/tasks
curl -X POST http://localhost:7860/reset -H "Content-Type: application/json" -d '{"task_id":"task_1"}'
```

## Definition of done

The project is ready for submission when all of these are true:

- Docker image builds.
- App runs without errors.
- `/reset` returns an observation.
- `/step` accepts a valid action.
- `/state` returns internal state.
- `/tasks` returns all 3 tasks.
- `/grader` returns a score in `[0.0, 1.0]`.
- `/baseline` returns all three task scores.
- Baseline script runs end to end.
- All grader tests pass.

## Main risks and how to avoid them

### Risk 1 — Overengineering Round 1

Avoid adding too much complexity before the core loop works. A simpler live environment is more valuable than a sophisticated broken one for Round 1.[cite:5]

### Risk 2 — Interface drift between teammates

Avoid changing shared field names once coding begins. Freeze schemas early and treat them like API contracts.

### Risk 3 — Hidden deployment failures

Deploy a skeleton early instead of waiting until the end. Docker and import-path issues are common failure points in environment projects.[cite:6][cite:8]

### Risk 4 — Broken baseline due to malformed model output

Write the parser before polishing the prompt. A decent parser prevents baseline collapse from formatting drift.[cite:22][cite:76]

## Recommended immediate next steps

1. Open the starter scaffold in the IDE.
2. Freeze `models.py` first.
3. Split work according to the teammate ownership above.
4. Finish `task_1` end to end before making `task_2` and `task_3` sophisticated.
5. Get a server skeleton running as early as possible.

## Bottom-line build principle

This project should be built as a reliable benchmark first and an elegant benchmark second. In practice, that means deterministic tasks, stable contracts, simple graders, fast deployment, and only then extra complexity.[cite:5][cite:6]
