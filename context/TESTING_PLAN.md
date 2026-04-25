# CommerceOps-Env — Testing Plan ($30 HF Credit)

Split between Dhruv and Tendulkar.  
**Rule**: finish your part, then share outputs (Space URL, model repo URL, results JSON) with the other before moving on.

---

## Credit budget

| Item | Owner | Hardware | Est. time | Est. cost |
|------|-------|----------|-----------|-----------|
| Deploy HF Space (env API) | Dhruv | cpu-basic (free tier) | — | **$0** |
| API smoke test job (verify live Space) | Dhruv | cpu-basic | 5 min | ~$0.01 |
| Baseline eval — frozen Qwen2.5-3B | Tendulkar | `l4x1` | ~45 min | ~$2 |
| GRPO fast-dev smoke (5 steps) | Tendulkar | `l4x1` | ~20 min | ~$1 |
| GRPO full training run (200 steps) | Tendulkar | `l4x1` | ~90 min | ~$4 |
| Post-training eval | Tendulkar | `l4x1` | ~45 min | ~$2 |
| Re-run buffer / debugging | Both | `l4x1` | — | ~$5 |
| **Total** | | | | **~$14 of $30** |

The remaining ~$16 is buffer for a second training run if the first shows reward hacking, or for trying `a10g-small` if `l4x1` is too slow.

---

## Prerequisites (both)

```bash
pip install huggingface_hub
hf auth login          # paste your write token
hf whoami              # confirm login
```

Make sure your HF account has the $30 credit loaded before submitting any GPU job.

---

## Dhruv's tasks

### D1 — Deploy the HF Space

Update `README.md` with your actual Space URL once deployed.

```bash
# Push the repo to HF Spaces as a Docker Space
# (repo already has Dockerfile + openenv.yaml)
# Create Space at https://huggingface.co/new-space
#   SDK: Docker  |  hardware: cpu-basic (free)  |  port: 7860
# Then link your GitHub repo or push directly:

git remote add space https://huggingface.co/spaces/YOUR_USERNAME/commerce-ops-env
git push space main
```

**Success criteria**: `curl https://YOUR_USERNAME-commerce-ops-env.hf.space/health` returns `{"ok": true}`.

---

### D2 — Smoke-test the live API (HF Jobs, cpu-basic, ~$0.01)

Verifies the deployed Space handles all endpoints correctly before training starts.

```bash
hf jobs uv run train/smoke_test_api.py \
  --flavor cpu-basic \
  --timeout 10m \
  --env ENV_BASE_URL=https://YOUR_USERNAME-commerce-ops-env.hf.space
```

**What it checks**: `/health`, `/reset` T1+T2, `/step` valid+invalid, `/grader`, `/baseline`, `/schema`.

**Success criteria**: All assertions pass, no 500s. Job finishes in <5 min.

---

### D3 — Share with Tendulkar

After D1+D2: post the Space URL so Tendulkar can use `ENV_BASE_URL` in training.

---

## Tendulkar's tasks

### T1 — Baseline eval (frozen Qwen2.5-3B, HF Jobs, l4x1, ~$2)

This produces the "before" numbers for the demo.

```bash
# First update train/hf_train.py: replace YOUR_USERNAME with your HF username
# and set ENV_REPO_URL to your GitHub repo URL.

hf jobs uv run train/hf_train.py \
  --flavor l4x1 \
  --timeout 2h \
  --secret HF_TOKEN=$HF_TOKEN \
  --env TRAIN_STEPS=0 \
  --env HUB_MODEL_REPO=YOUR_USERNAME/commerce-ops-grpo \
  --env HUB_RESULTS_REPO=YOUR_USERNAME/commerce-ops-results \
  --env ENV_REPO_URL=https://github.com/YOUR_USERNAME/ecommerce-ops-env-starter.git
```

> **Tip**: set `TRAIN_STEPS=0` to skip training and only run the baseline eval. Edit `hf_train.py` main() to exit after baseline if you want just the numbers. Or run the fast-dev first (T2 below).

**Success criteria**: `results.json` pushed to `HUB_RESULTS_REPO` with `baseline.by_task.task_2.mean_score` somewhere between 0.1–0.7 (confirms the frozen model is imperfect, giving RL room to improve).

---

### T2 — GRPO fast-dev smoke (5 steps, HF Jobs, l4x1, ~$1)

Verifies the full pipeline runs end-to-end before spending $4 on a full run.

```bash
hf jobs uv run train/hf_train.py \
  --flavor l4x1 \
  --timeout 45m \
  --secret HF_TOKEN=$HF_TOKEN \
  --env FAST_DEV=1 \
  --env HUB_MODEL_REPO=YOUR_USERNAME/commerce-ops-grpo-dev \
  --env HUB_RESULTS_REPO=YOUR_USERNAME/commerce-ops-results \
  --env ENV_REPO_URL=https://github.com/YOUR_USERNAME/ecommerce-ops-env-starter.git
```

**Success criteria**: Job completes, `results.json` is pushed, no OOM, training loss is finite.

---

### T3 — Full GRPO training (200 steps, HF Jobs, l4x1, ~$4)

The real training run. Only submit after T2 passes.

```bash
hf jobs uv run train/hf_train.py \
  --flavor l4x1 \
  --timeout 3h \
  --secret HF_TOKEN=$HF_TOKEN \
  --env TRAIN_STEPS=200 \
  --env HUB_MODEL_REPO=YOUR_USERNAME/commerce-ops-grpo \
  --env HUB_RESULTS_REPO=YOUR_USERNAME/commerce-ops-results \
  --env ENV_REPO_URL=https://github.com/YOUR_USERNAME/ecommerce-ops-env-starter.git
```

**Monitor**:
```bash
hf jobs ps                       # check status
hf jobs logs <job-id>            # tail logs
```

**Success criteria**:
- `trained.by_task.task_2.mean_score` > `baseline.by_task.task_2.mean_score`
- Training loss goes down over steps
- `reward_curves.png` is pushed and clearly shows improvement

---

### T4 — Anti-hacking inspection

After T3, pull `results.json` from the Hub and check the trajectories:

```python
from huggingface_hub import hf_hub_download
import json

path = hf_hub_download("YOUR_USERNAME/commerce-ops-results", "results.json",
                        repo_type="dataset")
with open(path) as f:
    data = json.load(f)

# Spot-check trained rollouts on T2
for ep in data["trained"]["all"]:
    if ep["task_id"] == "task_2":
        print(f"seed={ep['seed']}  score={ep['score']:.3f}  "
              f"invalid={ep['invalid_actions']}  steps={ep['steps']}")
```

**Success criteria**:
- Invalid action rate stays low (model isn't spamming noops or gibberish)
- Score improvement is consistent across seeds, not just 1-2 cherry-picked ones
- Step count is reasonable (model isn't stalling)

---

### T5 — Pull reward_curves.png and share with Dhruv

```python
from huggingface_hub import hf_hub_download
path = hf_hub_download("YOUR_USERNAME/commerce-ops-results",
                        "reward_curves.png", repo_type="dataset")
print("Chart at:", path)
```

This chart goes into the HF mini-blog / demo.

---

## Shared: local eval with trained model (optional, no HF cost)

After the model is pushed, either of you can pull it locally and run:

```bash
python train/eval.py \
  --model YOUR_USERNAME/commerce-ops-grpo \
  --model-label grpo_trained \
  --tasks task_1 task_2 \
  --seeds 0 1 2 3
```

This produces `train/logs/reward_curves.png` locally, comparing oracle / random / trained.

---

## If training shows reward hacking

Signs: reward goes up but score stays flat, or model outputs only `{"action_type": "noop"}` repeatedly.

**Fixes** (in order of effort):
1. **Increase KL penalty**: change `kl_coef=0.01` → `kl_coef=0.05` in `hf_train.py`
2. **Reduce GRPO rollouts**: `GRPO_N_SAMPLES=4` → `2`
3. **Add format penalty**: in `grpo_reward_fn`, subtract 0.1 if `action_type == "noop"` and it wasn't the expected action
4. **Switch to T1 only**: remove `task_2` from `TRAIN_TASKS` — get solid T1 improvement first

---

## What "done" looks like

| Artifact | Where |
|----------|-------|
| Live Space API | `https://YOUR_USERNAME-commerce-ops-env.hf.space/docs` |
| Trained model | `https://huggingface.co/YOUR_USERNAME/commerce-ops-grpo` |
| Results JSON | `https://huggingface.co/datasets/YOUR_USERNAME/commerce-ops-results` |
| Reward curve PNG | Same dataset repo → `reward_curves.png` |
| Before/after table | Printed in HF Jobs logs, copied to mini-blog |

Once all five rows are filled in, you have everything needed for the demo and the HF mini-blog.
