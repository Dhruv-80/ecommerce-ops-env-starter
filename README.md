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

## Quick Links

| Resource | Link |
|----------|------|
| **Live Demo Space** | [huggingface.co/spaces/YOUR_SPACE](https://huggingface.co/spaces/YOUR_SPACE) |
| **Trained Model** | [huggingface.co/TenduL/commerce-ops-grpo](https://huggingface.co/TenduL/commerce-ops-grpo) |
| **Training Results** | [huggingface.co/datasets/TenduL/commerce-ops-results](https://huggingface.co/datasets/TenduL/commerce-ops-results) |
| **Blog Post** | [blog.md](blog.md) |
| **Training Notebook** | [train/hf_train.ipynb](train/hf_train.ipynb) |

---

## What This Is

A stateful, OpenEnv-compatible RL environment where a language model acts as a fulfillment operations manager. The agent receives structured observations about orders, warehouses, stock levels, SLAs, and customer tiers, and must choose structured actions to resolve the situation.

**Core claim**: Human fulfillment managers make judgment calls that cannot be covered cleanly with a rules engine. This environment captures that judgment, and GRPO training shows measurable improvement.

---

## The Three Tasks

| Task | Name | Difficulty | Purpose |
|------|------|----------|---------|
| `task_1` | Warehouse Assignment | Easy | Bootstrap — single order, pick the best warehouse. Gets the model format-compliant early in training. |
| `task_2` | Multi-Order Fulfillment Triage | Medium | **Headline task.** Three orders, one SKU, two warehouses, total stock < total demand. Assign the loyalty/premium orders to their nearest stocked warehouse and delay the standard-tier order. |
| `task_3` | Cascade Recovery | Medium | A supplier failure has zeroed stock at one warehouse. Reroute the pending order to the warehouse that still has stock. |

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
- **Model**: Qwen2.5-1.5B-Instruct (cost-tuned for HF Jobs `l4x1`)
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
pytest tests/ -q          # 92 tests
```

### Run Evaluation

```bash
python train/eval.py                            # Full 8-seed run
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
| GET | `/state` | Full internal state (debug/grading inspection) |
| GET | `/tasks` | Task catalogue |
| POST | `/grader` | Episode score + breakdown |
| GET | `/baseline` | Oracle baseline on all tasks |
| GET | `/schema` | Action / observation JSON schema |
| GET | `/metadata` | Environment metadata |
| POST | `/mcp` | MCP JSON-RPC pass-through |

---

## Training

### Option 1: Google Colab (Recommended for Judges)

Open [`train/hf_train.ipynb`](train/hf_train.ipynb) in Colab:

1. Click "Open in Colab" or upload the notebook
2. Select GPU runtime: Runtime → Change runtime type → T4 GPU
3. Set your HF token in Cell 4
4. Run all cells (~45-60 minutes on T4)

### Option 2: Hugging Face Jobs (One Command)

Set the training values in `.env`, then run:

```bash
bash train/run_hf_job.sh
```

Required `.env` keys:

```bash
HF_TOKEN=hf_xxx
ENV_REPO_URL=https://github.com/Dhruv-80/ecommerce-ops-env-starter.git
HUB_MODEL_REPO=YOUR_USERNAME/commerce-ops-grpo
HUB_RESULTS_REPO=YOUR_USERNAME/commerce-ops-results
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
| task_2 | 0.99 | +2.85 | 3 |
| task_3 | 0.99 | +0.95 | 1 |

### Trivial All-Noop Policy (Lower Bound)

| Task | Mean Score | Cumulative Reward |
|------|------------|-------------------|
| task_1 | 0.01 | -0.10 |
| task_2 | 0.01 | -0.15 |
| task_3 | 0.01 | -0.10 |

The reward delta between optimal and noop is **~3.0 on T2** and **~1.05 on T1/T3** — large enough that GRPO group-relative advantages are non-degenerate.

*Live before/after numbers from each training run (mean score, mean reward, invalid rate per task) are pushed to [huggingface.co/datasets/TenduL/commerce-ops-results-3](https://huggingface.co/datasets/TenduL/commerce-ops-results-3) along with `reward_curves.png`.*

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
│   ├── test_env.py      # 51 env + server tests
│   ├── test_training.py # 19 metrics + eval tests
│   └── …               # Legacy R1 tests
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

See [`blog.md`](blog.md) for a detailed writeup on:

- The problem and why it matters
- Training approach (GRPO with fast-forwarded states)
- What worked and what didn't
- Key learnings and surprising findings
- Future work

---

## License

MIT License — built for the Meta PyTorch OpenEnv Hackathon 2026.
