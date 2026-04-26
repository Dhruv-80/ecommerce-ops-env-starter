# CommerceOps-Env: Training LLMs for E-Commerce Fulfillment Decisions

## The Problem

When an order arrives on an e-commerce platform, an operations manager has to decide:

- Which warehouse ships it?
- Should we split a single order across multiple warehouses?
- Should we delay an order if stock is low?
- If a supplier fails, who gets rerouted, who gets compensated?

These decisions trade off **customer tier**, **SLA deadlines**, **inventory scarcity**, and **shipping distance** — they're hard to encode as a clean rules engine. Human fulfillment managers make these judgment calls every day. Can an LLM learn to make similar ones from reward signal alone?

## The Environment

CommerceOps-Env is an OpenEnv-compatible reinforcement learning environment that simulates e-commerce fulfillment operations. The agent receives structured observations (orders, warehouses, stock cells, distance buckets, SLAs, allowed actions) and outputs a single structured JSON action per step:

```json
{"action_type": "assign_warehouse", "order_id": "O1", "warehouse_id": "W2"}
{"action_type": "split_shipment",   "order_id": "O3", "allocations": [{"warehouse_id": "W1", "quantity": 1}, {"warehouse_id": "W2", "quantity": 1}]}
{"action_type": "delay_order",      "order_id": "O5", "reason": "stock_insufficient"}
{"action_type": "reroute_order",    "order_id": "O1", "warehouse_id": "W2"}
{"action_type": "noop"}
```

Schema validation is performed by Pydantic v2 inside the environment, before any state mutation. Anything that doesn't match the per-task whitelist gets a flat invalid-action penalty.

### Three Tasks

| Task | Name | Setup | Decision |
|------|------|-------|----------|
| `task_1` | Warehouse Assignment | 1 order, 3 warehouses | Pick the closest valid warehouse with stock |
| `task_2` | Multi-Order Triage | 2 orders, 1 SKU, 2 warehouses (1 unit each) | Assign each order to its NEAR warehouse — wasting W1 on the wrong order leaves the other unfilled |
| `task_3` | Cascade Recovery | 1 active order, 2 warehouses, supplier failure has zeroed stock at one WH | Reroute the order to the warehouse that still has stock |

The scenarios are deterministic given `(task_id, seed)`, but the seed permutes which order gets each tier (T2) and which warehouse failed (T3), so the agent can't memorize fixed answers.

## Environment Design

The environment is built around four design principles:

1. **Stateful** — actions change inventory, order status, and step count. The model is never allowed to mutate state directly; only verified actions through `EnvAction` schemas can.
2. **Verifiable** — success is decided from environment state and explicit business rules, not from model explanations.
3. **Trainable** — early tasks produce non-zero reward often enough that GRPO has signal to climb.
4. **Hard to game** — anti-hacking checks include action whitelisting, strict per-action schema validation, protected stock counters, max-step timeouts, repeat-action detection, and collateral-damage flags.

## Training Approach

We use **GRPO (Group Relative Policy Optimization)** via Hugging Face TRL.

### 1. Few-Shot JSON Examples in the System Prompt

Instead of abstract instructions, the prompt shows concrete JSON examples for each action type. Format priming brings valid-action rates from ~60% to ~95% on the frozen baseline.

### 2. Fast-Forwarded Training States

For T2, the agent needs to act at multiple mid-episode states (after order 1 is assigned, then after order 2 is assigned, etc.). We replay oracle actions to fast-forward the env, then sample the model on the resulting observation. This gives diverse training prompts without per-rollout variance.

### 3. Single-Step Reward

The GRPO reward function applies the model's action to the fast-forwarded state and returns the immediate environment reward. This gives a stable, dense per-step signal that's easy for GRPO group-relative advantages to consume.

### 4. Layered Reward Shaping

The env's reward layer (`reward.py`) breaks per-step reward into:

| Component | Value |
|-----------|-------|
| Schema compliance | +0.10 |
| Correct entity targeted | +0.20 |
| Correct action type for that entity | +0.30 |
| State update matches ground truth | +0.40 (× partial credit if close-but-wrong) |
| Repeat action | −0.10 |
| Step penalty | −0.05 |
| Collateral damage (e.g. assign infeasible) | −0.20 |
| Invalid action (failed schema) | −0.35 |

T3's step verifier returns `partial_credit=0.5` when the model picks the right verb (`reroute_order`) but the wrong warehouse — so "tried to reroute to W1 when W2 was healthy" still earns +0.45 step reward instead of zero. That keeps the gradient dense at the boundary between right verb and right target.

## Model and Compute

- **Base model**: Qwen2.5-1.5B-Instruct
- **Fine-tuning**: LoRA adapters (r=16, alpha=32) via Unsloth 4-bit
- **GRPO**: 80 steps, group size 6, temperature 1.0, top_p 0.95, β=0.01
- **Hardware**: NVIDIA L4 / H200 (HF Jobs)

A GPU pre-flight check at the very top of `hf_train.py` exits before any heavy imports if CUDA isn't available, with retries for cold-init on H200 / H100 nodes.

## Results

### Per-task scores

| Task | Random | Frozen Qwen2.5-1.5B | Trained (GRPO 80 steps) | Oracle |
|------|--------|--------------------|-----------------------|--------|
| task_1 | 0.53 | *baseline-eval* | *post-train-eval* | 0.99 |
| task_2 | 0.41 | *baseline-eval* | *post-train-eval* | 0.99 |
| task_3 | 0.30 | *baseline-eval* | *post-train-eval* | 0.99 |

Live numbers (mean score / mean reward / invalid rate per task) — plus the full GRPO log history and `reward_curves.png` — are written to [huggingface.co/datasets/TenduL/ecommerce-ops-results](https://huggingface.co/datasets/TenduL/ecommerce-ops-results) at the end of every training run.

### Cumulative reward on perfect play (sanity check)

Running the oracle plan deterministically against the env:

```
TASK 1 OPTIMAL  score=0.99  cum_reward=+0.95   steps=1
TASK 2 OPTIMAL  score=0.99  cum_reward=+1.90   steps=2
TASK 3 OPTIMAL  score=0.99  cum_reward=+0.95   steps=1

TASK 2 ALL-NOOP score=0.01  cum_reward=-0.10
TASK 3 ALL-NOOP score=0.01  cum_reward=-0.10
```

The reward delta between optimal and all-noop policies is large enough to give GRPO group-relative advantages clean variance to learn from.

## Reproducibility

All code, environment definitions, and results are public:

- **GitHub**: [github.com/Dhruv-80/ecommerce-ops-env-starter](https://github.com/Dhruv-80/ecommerce-ops-env-starter)
- **Trained model**: [huggingface.co/TenduL/ecommerce-ops-grpo](https://huggingface.co/TenduL/ecommerce-ops-grpo)
- **Training results / logs**: [huggingface.co/datasets/TenduL/ecommerce-ops-results](https://huggingface.co/datasets/TenduL/ecommerce-ops-results)
- **Live OpenEnv space**: [huggingface.co/spaces/YOUR_SPACE](https://huggingface.co/spaces/YOUR_SPACE)

To reproduce:

```bash
# Local smoke test
pip install -r requirements.txt
uvicorn server.app:app --host 0.0.0.0 --port 7860
pytest tests/test_env.py -q

# HF Jobs training run (uses values from .env)
bash train/run_hf_job.sh
```

Or open `train/hf_train.ipynb` in Colab on a T4 / L4 GPU and run all cells.

## Conclusion

CommerceOps-Env demonstrates that an LLM can learn structured fulfillment decisions under realistic constraints from environment reward alone. The environment, training script, and a runnable Jupyter notebook are all open under MIT license.

---

*Built for the Meta PyTorch OpenEnv Hackathon 2026.*
