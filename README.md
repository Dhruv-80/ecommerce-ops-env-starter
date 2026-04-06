# ecommerce-ops-env starter

Starter scaffold for an OpenEnv-style e-commerce operations environment.

## Goal
Build a 3-task environment where an agent resolves:
1. Refund queue processing
2. Inventory reconciliation
3. Supplier cancellation crisis

## Recommended work split
- Teammate A: `server/tasks.py`, `server/reward.py`, `server/grader.py`, `tests/test_graders.py`
- Teammate B: `models.py`, `server/ecommerce_environment.py`, `server/app.py`, `inference.py`, Docker/deployment/docs

## Merge order
1. Lock schemas in `models.py`
2. Lock task ground truth in `server/tasks.py`
3. Lock reward + grader contracts
4. Wire environment step logic
5. Expose API + baseline inference

## First run checklist
- Confirm the exact OpenEnv package version you install
- Verify import path for `openenv.core.env_server`
- Fill in the TODOs marked `IMPLEMENT:`
- Deploy a skeleton to HF Spaces early
