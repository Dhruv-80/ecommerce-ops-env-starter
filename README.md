---
title: CommerceOps-Env
emoji: 🏭
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 7860
---

# CommerceOps-Env

OpenEnv-compatible RL environment for fulfillment-decision LLMs.
GRPO fine-tuning of `Qwen2.5-1.5B-Instruct` on three structured tasks: warehouse assignment, multi-order triage, cascade recovery.

| Resource | Link |
|---|---|
| Trained model | [huggingface.co/Gloomytarsier3/ecommerce-ops-grpo](https://huggingface.co/Gloomytarsier3/ecommerce-ops-grpo) |
| Training results / logs | [huggingface.co/datasets/Gloomytarsier3/ecommerce-ops-results](https://huggingface.co/datasets/Gloomytarsier3/ecommerce-ops-results) |
| Live OpenEnv Space | [huggingface.co/spaces/Gloomytarsier3/e-com-r2](https://huggingface.co/spaces/Gloomytarsier3/e-com-r2) |
| Training notebook (Colab) | [colab.research.google.com/drive/1zsdtNfhN8_pstougqh66K8bMmSRaeZrZ](https://colab.research.google.com/drive/1zsdtNfhN8_pstougqh66K8bMmSRaeZrZ?usp=sharing) |
| GitHub repo | [github.com/Dhruv-80/ecommerce-ops-env-starter](https://github.com/Dhruv-80/ecommerce-ops-env-starter) |
| Blog post | [blog.md](blog.md) |

---

## Stack

- **Environment** — Python, Pydantic v2, FastAPI, OpenEnv
- **RL algorithm** — GRPO via Hugging Face TRL
- **Adapters** — LoRA (r=16, α=32) via Unsloth 4-bit
- **Base model** — `Qwen/Qwen2.5-1.5B-Instruct`
- **Deployment** — Hugging Face Spaces (Docker)

---

## Tasks

| ID | Task | Action surface | Episode shape |
|---|---|---|---|
| `task_1` | Warehouse Assignment | `assign_warehouse`, `noop` | 1 order, 3 warehouses, `max_steps=4` |
| `task_2` | Multi-Order Triage | `assign_warehouse`, `split_shipment`, `delay_order`, `prioritize_order`, `noop` | 2 orders, 1 SKU, 2 warehouses, `max_steps=4` |
| `task_3` | Cascade Recovery | `reroute_order`, `escalate_supplier`, `refund_or_compensate`, `delay_order`, `noop` | 1 order, 2 warehouses, supplier failure, `max_steps=4` |

Episodes are deterministic given `(task_id, seed)`.

### Action schema

```json
{"action_type": "assign_warehouse", "order_id": "O1", "warehouse_id": "W2"}
{"action_type": "split_shipment",   "order_id": "O3", "allocations": [{"warehouse_id": "W1", "quantity": 1}, {"warehouse_id": "W2", "quantity": 1}]}
{"action_type": "delay_order",      "order_id": "O5", "reason": "stock_insufficient"}
{"action_type": "reroute_order",    "order_id": "O1", "warehouse_id": "W2"}
{"action_type": "noop"}
```

Validated by Pydantic v2 inside `EnvAction`. Schema-invalid actions are rejected before any state mutation.

---

## Reward Components

| Signal | Weight |
|---|---:|
| Schema compliance | +0.10 |
| Correct entity targeted | +0.20 |
| Correct action type for entity | +0.30 |
| State update matches ground truth | +0.40 (× partial credit) |
| Repeat action | −0.10 |
| Step penalty | −0.05 |
| Collateral damage | −0.20 |
| Invalid action (failed schema) | −0.35 |

Anti-hacking enforcement: action whitelist per task, protected stock counters, max-step timeout, repeat-action detection, collateral-damage flag for `assign_warehouse` to a depleted warehouse.

---

## API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness check |
| `POST` | `/reset` | `{"task_id": "task_2", "seed": 0}` → fresh episode |
| `POST` | `/step` | Submit one `EnvAction`, get next observation + reward |
| `GET` | `/state` | Full internal state (debug / grading inspection) |
| `GET` | `/tasks` | Task catalogue |
| `POST` | `/grader` | Episode score + breakdown |
| `GET` | `/baseline` | Oracle baseline on all tasks |
| `GET` | `/schema` | Action / observation JSON schema |
| `GET` | `/metadata` | Environment metadata |
| `POST` | `/mcp` | MCP JSON-RPC pass-through |

---

## Run Locally

```bash
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 7860
pytest tests/ -q
```

```bash
# Smoke
curl http://localhost:7860/health
curl -X POST http://localhost:7860/reset -H "Content-Type: application/json" -d '{"task_id": "task_2", "seed": 0}'
curl -X POST http://localhost:7860/step  -H "Content-Type: application/json" -d '{"action_type": "assign_warehouse", "order_id": "O1", "warehouse_id": "W1"}'
curl -X POST http://localhost:7860/grader
```

---

## Training

### Hugging Face Jobs (one command)

Set values in `.env`, then:

```bash
bash train/run_hf_job.sh
```

`.env` keys:

```bash
HF_TOKEN=hf_xxx
ENV_REPO_URL=https://github.com/Dhruv-80/ecommerce-ops-env-starter.git
HUB_MODEL_REPO=Gloomytarsier3/ecommerce-ops-grpo
HUB_RESULTS_REPO=Gloomytarsier3/ecommerce-ops-results
MODEL_NAME=Qwen/Qwen2.5-1.5B-Instruct
TRAIN_STEPS=80
FAST_DEV=0
HF_FLAVOR=l4x1
HF_TIMEOUT=2h
```

The run pushes the LoRA adapters + tokenizer to `HUB_MODEL_REPO` and `results.json` + `reward_curves.png` (full GRPO log history per step) to `HUB_RESULTS_REPO`.

### Colab / Jupyter

Open the [Colab notebook](https://colab.research.google.com/drive/1zsdtNfhN8_pstougqh66K8bMmSRaeZrZ?usp=sharing) on a T4 / L4 GPU and run all cells. Each cell saves its output, so the executed notebook is itself a record of the training run. The same notebook lives in the repo at `train/hf_train.ipynb`.

---

## Results

Baseline (frozen `Qwen2.5-1.5B-Instruct`) vs. GRPO-trained, 4 eval seeds per task. Pulled from [`results.json`](https://huggingface.co/datasets/Gloomytarsier3/ecommerce-ops-results/blob/main/results.json):

| Task | Metric | Baseline | Trained | Δ |
|---|---|---:|---:|---:|
| `task_1` | mean_score | 0.5500 | 0.7950 | **+0.2450** |
| `task_1` | mean_reward | +0.2125 | +0.8000 | **+0.5875** |
| `task_1` | invalid_rate | 0.5714 | 0.0000 | **−0.5714** |
| `task_2` | mean_score | 0.8139 | 0.8139 | 0.0000 |
| `task_2` | mean_reward | +2.0750 | +2.0750 | 0.0000 |
| `task_2` | invalid_rate | 0.0000 | 0.0000 | 0.0000 |
| `task_3` | mean_score | 0.0100 | 0.5000 | **+0.4900** |
| `task_3` | mean_reward | −1.4000 | +1.2250 | **+2.6250** |
| `task_3` | invalid_rate | 1.0000 | 0.0000 | **−1.0000** |

Training ran for 80 GRPO steps with `train_loss = 0.0121`. Full log history (per-step loss, KL, completion length, reward variance, etc.) is in the [results dataset](https://huggingface.co/datasets/Gloomytarsier3/ecommerce-ops-results).

### Bounds

| Task | Oracle score | Oracle cum. reward | All-noop score | All-noop cum. reward |
|---|---:|---:|---:|---:|
| `task_1` | 0.99 | +0.95 | 0.01 | −0.10 |
| `task_2` | 0.99 | +1.90 | 0.01 | −0.10 |
| `task_3` | 0.99 | +0.95 | 0.01 | −0.10 |

```bash
python train/eval.py                                # full multi-seed run
python train/eval.py --fast-dev --no-plot           # quick smoke test
python train/eval.py --model ./grpo_output/final    # include trained model
```

---

## File Layout

```
├── models.py            # EnvAction, EnvObservation, EnvState contracts
├── tasks.py             # Episode generators for T1, T2, T3
├── verifier.py          # Step-level + episode-level correctness checks
├── reward.py            # Reward combiner
├── environment.py       # reset / step / final_score
├── server/app.py        # FastAPI / OpenEnv server
├── train/
│   ├── hf_train.py      # GRPO training script (HF Jobs UV)
│   ├── hf_train.ipynb   # Same logic, runnable as a notebook
│   ├── eval.py          # Oracle / random / model evaluation
│   └── metrics.py       # MetricsLogger + TrainingMetricsTracker
├── tests/               # pytest suite
├── openenv.yaml
├── Dockerfile
├── requirements.txt
├── blog.md
└── README.md
```

---

## Deployment

```bash
docker build -t commerce-ops-env .
docker run --rm -p 7860:7860 commerce-ops-env
```

Live Space: [huggingface.co/spaces/Gloomytarsier3/e-com-r2](https://huggingface.co/spaces/Gloomytarsier3/e-com-r2)
API base: `https://gloomytarsier3-e-com-r2.hf.space`
Swagger docs: `https://gloomytarsier3-e-com-r2.hf.space/docs`

---

## License

MIT.
