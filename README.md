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

## Project Links

| Resource | Link |
|----------|------|
| **GitHub repo** | [github.com/Dhruv-80/ecommerce-ops-env-starter](https://github.com/Dhruv-80/ecommerce-ops-env-starter) |
| **Live OpenEnv Space (Docker)** | [huggingface.co/spaces/YOUR_SPACE](https://huggingface.co/spaces/YOUR_SPACE) |
| **Trained model (GRPO LoRA)** | [huggingface.co/TenduL/ecommerce-ops-grpo](https://huggingface.co/TenduL/ecommerce-ops-grpo) |
| **Training logs / results dataset** | [huggingface.co/datasets/TenduL/ecommerce-ops-results](https://huggingface.co/datasets/TenduL/ecommerce-ops-results) |
| **Blog post** | [blog.md](blog.md) |
| **Training script** | [train/hf_train.py](train/hf_train.py) |
| **Training notebook** | [train/hf_train.ipynb](train/hf_train.ipynb) |

The training pipeline pushes `results.json` (per-task baseline + trained mean score / mean reward / invalid rate, full GRPO log history) and `reward_curves.png` to the results dataset above. The trained LoRA adapters and tokenizer are pushed to the model repo above.

---

## What This Is

A stateful, OpenEnv-compatible RL environment where a language model acts as a fulfillment operations manager. The agent receives structured observations about orders, warehouses, stock levels, SLAs, and customer tiers, and must choose structured actions to resolve the situation.

**Core claim**: Human fulfillment managers make judgment calls that cannot be covered cleanly with a rules engine. This environment captures that judgment, and GRPO training shows measurable improvement.

---

## The Three Tasks

| Task | Name | Difficulty | Description |
|------|------|----------|---------|
| `task_1` | Warehouse Assignment | Easy | One incoming order; pick the closest valid warehouse with stock and the right shipping method. |
| `task_2` | Multi-Order Fulfillment Triage | Medium | **Headline task.** Two orders share two warehouses with one unit of stock each. Each order has a clear NEAR warehouse based on its destination region. Wasting the only stocked warehouse on the wrong order leaves the other order unfilled. |
| `task_3` | Cascade Recovery | Medium | A supplier failure has zeroed stock at one warehouse. The pending order must be rerouted to the warehouse that still has stock. |

The scenarios are deterministic given `(task_id, seed)`. Seeds permute which order receives each tier (T2) and which warehouse failed (T3) so the agent can't memorize fixed answers.

---

## Action Whitelist (Task 2)

```json
{"action_type": "assign_warehouse", "order_id": "O1", "warehouse_id": "W2"}
{"action_type": "split_shipment",   "order_id": "O3", "allocations": [{"warehouse_id": "W1", "quantity": 1}, {"warehouse_id": "W2", "quantity": 1}]}
{"action_type": "delay_order",      "order_id": "O5", "reason": "stock_insufficient"}
{"action_type": "prioritize_order", "order_id": "O1"}
{"action_type": "noop"}
```

---

## Reward Design

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

---

## Stack

- **Environment**: Python, Pydantic v2, FastAPI, OpenEnv
- **RL algorithm**: GRPO via Hugging Face TRL
- **Efficiency**: Unsloth 4-bit quantization
- **Model**: Qwen2.5-1.5B-Instruct
- **Deployment**: Hugging Face Spaces (Docker)

---

## Run Locally

```bash
python -m venv .venv && source .venv/bin/activate  # Linux/Mac
# or: python -m venv .venv && .venv\Scripts\activate  # Windows

pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 7860
```

### Quick Smoke Test

```bash
# Health check
curl http://localhost:7860/health

# Reset to Task 2 (headline task)
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

### Run Tests

```bash
pytest tests/ -q
```

### Run Evaluation

```bash
python train/eval.py                            # Full multi-seed run
python train/eval.py --fast-dev --no-plot       # Quick smoke test
python train/eval.py --model ./grpo_output/final  # Include trained model
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness check |
| POST | `/reset` | Start a new episode `{"task_id": "task_2", "seed": 0}` |
| POST | `/step` | Submit one action, get next observation + reward |
| GET | `/state` | Full internal state (debug / grading inspection) |
| GET | `/tasks` | Task catalogue |
| POST | `/grader` | Episode score + breakdown |
| GET | `/baseline` | Oracle baseline on all tasks |
| GET | `/schema` | Action / observation JSON schema |
| GET | `/metadata` | Environment metadata |
| POST | `/mcp` | MCP JSON-RPC pass-through |

---

## Training

### Option 1: Google Colab

Open [`train/hf_train.ipynb`](train/hf_train.ipynb) in Colab:

1. Click "Open in Colab" or upload the notebook.
2. Select GPU runtime: Runtime → Change runtime type → T4 / L4 GPU.
3. Set your HF token in the env-vars cell.
4. Run all cells — every output (baseline eval, training logs, post-train eval, before/after table, reward curve) is captured in the notebook for review.

### Option 2: Hugging Face Jobs (One Command)

Set the training values in `.env`, then run:

```bash
bash train/run_hf_job.sh
```

Required `.env` keys:

```bash
HF_TOKEN=hf_xxx
ENV_REPO_URL=https://github.com/Dhruv-80/ecommerce-ops-env-starter.git
HUB_MODEL_REPO=YOUR_USERNAME/ecommerce-ops-grpo
HUB_RESULTS_REPO=YOUR_USERNAME/ecommerce-ops-results
MODEL_NAME=Qwen/Qwen2.5-1.5B-Instruct
TRAIN_STEPS=80
FAST_DEV=0
HF_FLAVOR=l4x1
HF_TIMEOUT=2h
```

---

## Results

### Oracle (Upper Bound, Perfect Play)

| Task | Mean Score | Cumulative Reward | Steps |
|------|------------|-------------------|-------|
| task_1 | 0.99 | +0.95 | 1 |
| task_2 | 0.99 | +1.90 | 2 |
| task_3 | 0.99 | +0.95 | 1 |

### Trivial All-Noop Policy (Lower Bound)

| Task | Mean Score | Cumulative Reward |
|------|------------|-------------------|
| task_1 | 0.01 | -0.10 |
| task_2 | 0.01 | -0.10 |
| task_3 | 0.01 | -0.10 |

Live before/after numbers from each training run (mean score, mean reward, invalid rate per task) are pushed to [huggingface.co/datasets/TenduL/ecommerce-ops-results](https://huggingface.co/datasets/TenduL/ecommerce-ops-results) along with `reward_curves.png` and the full GRPO log history.

---

## File Structure

```
├── models.py            # Frozen typed contracts (EnvAction, EnvObservation, EnvState)
├── tasks.py             # Episode generators for T1, T2, T3
├── verifier.py          # Step-level and episode-level correctness checks
├── reward.py            # Reward combiner (schema + entity + action + state)
├── environment.py       # reset / step / final_score + anti-hacking enforcement
├── server/
│   └── app.py           # FastAPI / OpenEnv server
├── train/
│   ├── hf_train.py      # GRPO training script for HF Jobs
│   ├── hf_train.ipynb   # Same training logic as Jupyter notebook
│   ├── metrics.py       # MetricsLogger + TrainingMetricsTracker
│   └── eval.py          # Oracle vs random vs model evaluation
├── tests/
│   ├── test_env.py      # Env + server tests
│   └── test_training.py # Metrics + eval tests
├── openenv.yaml
├── Dockerfile
├── requirements.txt
├── blog.md              # Detailed writeup on training approach
└── README.md            # This file
```

---

## Deployment

```bash
# Build
docker build -t commerce-ops-env .

# Run
docker run --rm -p 7860:7860 commerce-ops-env
```

**Live Space**: https://huggingface.co/spaces/YOUR_SPACE  
**API base**: https://YOUR_SPACE.hf.space  
**Swagger docs**: https://YOUR_SPACE.hf.space/docs

---

## Blog Post

See [`blog.md`](blog.md) for a detailed writeup on the problem, environment design, training approach, results, and reproducibility.

---

## License

MIT License — built for the Meta PyTorch OpenEnv Hackathon 2026.
