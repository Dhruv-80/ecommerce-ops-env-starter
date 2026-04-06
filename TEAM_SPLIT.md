# Team split

## Shared rules
- Do not change field names in `models.py` without telling the other person.
- All scenario IDs, task IDs, and score field names must stay stable.
- Merge only after both sides pass local smoke tests.

## Teammate A — environment design + grading
Own these files:
- `server/tasks.py`
- `server/reward.py`
- `server/grader.py`
- `tests/test_graders.py`

### Deliverables
1. Hardcode fixed-seed scenarios for task_1, task_2, task_3.
2. Add ground truth annotations for each scenario.
3. Implement deterministic grading functions returning scores in [0.0, 1.0].
4. Implement reward breakdown logic.
5. Add adversarial tests: wrong entity, duplicate action, collateral damage.

### Contract with Teammate B
- `get_task_bundle(task_id)` returns a dict with `task`, `initial_state`, `ground_truth`, `max_steps`.
- `compute_step_reward(...)` returns `{reward, breakdown, error}`.
- `grade_episode(task_id, state)` returns `{score, breakdown}`.

## Teammate B — execution + API + baseline
Own these files:
- `models.py`
- `client.py`
- `server/ecommerce_environment.py`
- `server/app.py`
- `inference.py`
- `Dockerfile`, `requirements.txt`, `pyproject.toml`, `openenv.yaml`

### Deliverables
1. Lock dataclasses and request/response contracts.
2. Implement `reset()`, `step()`, and `state` plumbing.
3. Add `/tasks`, `/grader`, `/baseline` endpoints.
4. Implement baseline inference loop and action parser.
5. Make Docker + HF Space deployment work.

### Contract with Teammate A
- Environment consumes `get_task_bundle(task_id)` exactly as returned.
- Environment passes final state to `grade_episode(task_id, state)`.
- No renaming of task IDs or reward/grader keys.

## Suggested branch plan
- `feat/core-grading` → Teammate A
- `feat/env-server` → Teammate B
- Merge into `main` only after interface check

## Fastest build order
Day 1 morning: models locked
Day 1 afternoon: tasks + grader locked
Day 1 evening: environment wiring done
Day 2 morning: server + deploy skeleton
Day 2 afternoon: inference + readme + validation
