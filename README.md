---
title: CommerceOps-Env
emoji: 🏭
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
---

# CommerceOps-Env

> **CommerceOps-Env trains an LLM to make human-like fulfillment decisions under inventory scarcity, SLA pressure, and competing customer priorities inside a verifiable OpenEnv business workflow environment.**

Meta PyTorch OpenEnv Hackathon — Grand Finale submission.

---

## What this is

A stateful, OpenEnv-compatible RL environment where a language model acts as a fulfillment operations manager. The agent receives structured observations about orders, warehouses, stock levels, SLAs, and customer tiers, and must choose structured actions to resolve the situation.

The core claim: human fulfillment managers make judgment calls that cannot be covered cleanly with a rules engine. This environment captures that judgment, and GRPO training shows measurable improvement.

## The three tasks

| Task | Name | Difficulty | Purpose |
|------|------|-----------|---------|
| `task_1` | Warehouse Assignment | Easy | Bootstrap — single order, pick the best warehouse. Gets the model format-compliant early in training. |
| `task_2` | Multi-Order Fulfillment Triage | Medium | **Headline task.** Several orders compete for limited stock. Agent must assign, split, delay, or deprioritize while balancing tier and SLA. |
| `task_3` | Cascade Recovery | Hard | Stretch. Supplier/shipment failure; agent recovers by rerouting, compensating, or escalating. |

## Action whitelist (T2)

```json
{"action_type": "assign_warehouse", "order_id": "O1", "warehouse_id": "W2"}
{"action_type": "split_shipment",   "order_id": "O3", "allocations": [{"warehouse_id": "W1", "quantity": 1}, {"warehouse_id": "W2", "quantity": 1}]}
{"action_type": "delay_order",      "order_id": "O5", "reason": "stock_insufficient"}
{"action_type": "prioritize_order", "order_id": "O1"}
{"action_type": "noop"}
```

## Reward design

| Component | Value |
|-----------|-------|
| Schema compliance | +0.10 |
| Correct entity targeted | +0.20 |
| Correct action type | +0.30 |
| State update matches ground truth | +0.40 |
| Repeat action penalty | −0.10 |
| Step penalty | −0.05 |
| Collateral damage | −0.20 |
| Invalid action | −0.35 |

## Stack

- **Environment**: Python, Pydantic v2, FastAPI, OpenEnv
- **RL algorithm**: GRPO via Hugging Face TRL
- **Efficiency**: Unsloth 4-bit
- **Model**: Qwen2.5-3B-Instruct (fallback: Gemma 3 1B)
- **Deployment**: Hugging Face Spaces (Docker)

---

## Run locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### Quick smoke test

```bash
# Health
curl http://localhost:7860/health

# Reset to T2 (headline task)
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_id": "task_2", "seed": 0}'

# Take a step
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "assign_warehouse", "order_id": "O1", "warehouse_id": "W2"}'

# Grade the current episode
curl -X POST http://localhost:7860/grader

# Run oracle baseline on all tasks
curl http://localhost:7860/baseline
```

### Run tests

```bash
pytest tests/ -q          # 92 tests
```

### Run evaluation (oracle vs random, produces reward_curves.png)

```bash
python train/eval.py                            # full 8-seed run
python train/eval.py --fast-dev --no-plot       # quick smoke
python train/eval.py --model ./grpo_output/final  # include trained model
```

---

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness check |
| POST | `/reset` | Start a new episode `{"task_id": "task_2", "seed": 0}` |
| POST | `/step` | Submit one action, get next observation + reward |
| GET | `/state` | Full internal state (debug/grading inspection) |
| GET | `/tasks` | Task catalogue |
| POST | `/grader` | Episode score + breakdown |
| GET | `/baseline` | Oracle baseline on all tasks |
| GET | `/schema` | Action / observation JSON schema |
| GET | `/metadata` | Environment metadata |
| POST | `/mcp` | MCP JSON-RPC pass-through |

---

## Training

Open `train/grpo_train.ipynb` in Colab (T4 GPU).

1. Set `FAST_DEV_RUN = True` for a 5-step smoke test.
2. Cell 7 measures **baseline** (frozen Qwen2.5-3B-Instruct).
3. Cells 8–11 run **GRPO fine-tuning** (~200 steps, ~30–60 min on T4).
4. Cell 12 measures **post-training** performance.
5. Cell 13 prints the before/after comparison table.
6. Cell 14 saves `grpo_curves.png`.
7. Cell 15 inspects one T2 rollout to verify the reward increase is from better decisions, not format gaming.

### Expected before/after (reference numbers from eval.py)

| Policy | T1 score | T2 score |
|--------|----------|----------|
| Oracle (upper bound) | 0.99 | 0.99 |
| Random (lower bound) | 0.53 | 0.41 |
| Frozen Qwen2.5-3B | *run cell 7* | *run cell 7* |
| GRPO trained | *run cell 12* | *run cell 12* |

---

## File structure

```
├── models.py            # Frozen typed contracts (EnvAction, EnvObservation, EnvState, …)
├── tasks.py             # Episode generators for T1, T2, T3
├── verifier.py          # Step-level and episode-level correctness checks
├── reward.py            # Reward combiner (schema + entity + action + state components)
├── environment.py       # reset / step / final_score + anti-hacking enforcement
├── server/
│   └── app.py           # FastAPI / OpenEnv server
├── train/
│   ├── grpo_train.ipynb # GRPO training notebook (Colab)
│   ├── metrics.py       # MetricsLogger + TrainingMetricsTracker
│   └── eval.py          # Oracle vs random vs model evaluation + reward_curves.png
├── tests/
│   ├── test_env.py      # 51 env + server tests
│   ├── test_training.py # 19 metrics + eval tests
│   └── …               # legacy R1 tests (still pass)
├── openenv.yaml
├── Dockerfile
└── requirements.txt
```

---

## Deployment

```bash
# Build
docker build -t commerce-ops-env .

# Run
docker run --rm -p 7860:7860 commerce-ops-env
```

Live Space: https://huggingface.co/spaces/Gloomytarsier3/my-env  
API base: https://gloomytarsier3-my-env.hf.space  
Swagger: https://gloomytarsier3-my-env.hf.space/docs
