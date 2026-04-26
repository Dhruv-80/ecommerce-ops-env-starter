# /// script
# dependencies = [
#   "unsloth",
#   "trl>=0.12",
#   "peft",
#   "accelerate",
#   "bitsandbytes",
#   "datasets",
#   "huggingface-hub",
#   "fastapi>=0.115.0",
#   "uvicorn[standard]>=0.30.0",
#   "pydantic>=2.8.0",
#   "numpy>=1.26.0",
#   "matplotlib>=3.9.0",
# ]
# ///
"""[DEPRECATED — do NOT run this for the hackathon submission]

This file targeted the *original* task_3 design (multi-action cascade with
``escalate_supplier`` + ``refund_or_compensate``). Task 3 has since been
re-scoped to a single-decision reroute under supplier failure, and the
current canonical training script trains all three tasks (T1, T2, T3)
together with the simplified reward signal.

▶ Use ``train/hf_train.py`` (or ``train/hf_train.ipynb``) instead.
   Submission repos:
     model:   https://huggingface.co/TenduL/ecommerce-ops-grpo
     results: https://huggingface.co/datasets/TenduL/ecommerce-ops-results

This file is kept only for historical reference; it is unused by the
submission pipeline.
"""

from __future__ import annotations

import io
import json
import os
import random
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from huggingface_hub import HfApi, login, whoami

# ---------------------------------------------------------------------------
# Config from env vars
# ---------------------------------------------------------------------------

HF_TOKEN        = (
    os.environ.get("HF_TOKEN_PUSH", "")
    or os.environ.get("HUGGING_FACE_HUB_TOKEN", "")
    or os.environ.get("HF_TOKEN", "")
)
FAST_DEV        = os.environ.get("FAST_DEV", "0") == "1"
TRAIN_STEPS     = int(os.environ.get("TRAIN_STEPS", "40" if FAST_DEV else "400"))
EVAL_SEEDS      = [0, 1] if FAST_DEV else list(range(8))
TRAIN_SEEDS     = list(range(8)) if FAST_DEV else list(range(8))
TRAIN_TASKS     = ["task_3"]
EVAL_TASKS      = ["task_3"]
# Default to base Qwen — task_3 uses a *different action set* from task_1/2
# (reroute / escalate / refund), so warm-starting from a task_2 LoRA would
# actively hurt rather than help.
MODEL_NAME      = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-3B-Instruct")
MAX_SEQ_LEN     = 2048
LORA_R          = 16
LORA_ALPHA      = 32
LR              = 1e-5
GRPO_N_SAMPLES  = 8
BATCH_SIZE      = 2
GRPO_TEMP       = 1.2
GRPO_TOP_P      = 0.95

HUB_MODEL_REPO   = os.environ.get("HUB_MODEL_REPO")
HUB_RESULTS_REPO = os.environ.get("HUB_RESULTS_REPO")

if not HUB_MODEL_REPO or not HUB_RESULTS_REPO:
    try:
        _me = whoami(token=HF_TOKEN, cache=True)["name"]
    except Exception:
        _me = "user"
    HUB_MODEL_REPO   = HUB_MODEL_REPO   or f"{_me}/commerce-ops-grpo-task3"
    HUB_RESULTS_REPO = HUB_RESULTS_REPO or f"{_me}/commerce-ops-results-task3"

print(f"{'='*62}")
print(f"  CommerceOps-Env GRPO Training — Task 3 (cascade recovery)")
print(f"  model      : {MODEL_NAME}")
print(f"  steps      : {TRAIN_STEPS}  fast_dev: {FAST_DEV}")
print(f"  hub_model  : {HUB_MODEL_REPO}")
print(f"  hub_results: {HUB_RESULTS_REPO}")
print(f"  device     : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}")
print(f"  token      : {'SET (len=' + str(len(HF_TOKEN)) + ')' if HF_TOKEN else 'EMPTY ❌ — Hub push will be skipped'}")
print(f"  token_src  : "
      f"HF_TOKEN_PUSH={'y' if os.environ.get('HF_TOKEN_PUSH') else 'n'} "
      f"HUGGING_FACE_HUB_TOKEN={'y' if os.environ.get('HUGGING_FACE_HUB_TOKEN') else 'n'} "
      f"HF_TOKEN={'y' if os.environ.get('HF_TOKEN') else 'n'}")
print(f"{'='*62}\n")

if HF_TOKEN:
    login(token=HF_TOKEN)

# ---------------------------------------------------------------------------
# Locate the env package — clone on HF Jobs, use local files otherwise
# ---------------------------------------------------------------------------

REPO_URL = os.environ.get("ENV_REPO_URL", "")
REPO_DIR = "/tmp/commerce-ops-env"

_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)

if REPO_URL:
    if not os.path.exists(REPO_DIR):
        os.system(f"git clone --depth 1 {REPO_URL} {REPO_DIR}")
    else:
        os.system(
            f"git -C {REPO_DIR} fetch --depth 1 origin main "
            f"&& git -C {REPO_DIR} reset --hard origin/main"
        )
    sys.path.insert(0, REPO_DIR)
else:
    sys.path.insert(0, _PROJECT_DIR)

from environment import CommerceOpsEnv  # noqa: E402
from tasks import get_task_bundle       # noqa: E402

import environment as _env_module
print(f"[CHECKPOINT] environment.py loaded from: {_env_module.__file__}")
print(f"[CHECKPOINT] FAST_DEV={FAST_DEV}, TRAIN_STEPS={TRAIN_STEPS}, TRAIN_TASKS={TRAIN_TASKS}")

# ---------------------------------------------------------------------------
# Prompt / action utilities
# ---------------------------------------------------------------------------

# Few-shot prompt tailored for task_3. Two non-obvious things the model
# needs to learn from this prompt alone (the rest comes from GRPO):
#   1. The action *shapes* for reroute / escalate / refund — these are
#      different from task_1/2 (e.g. escalate_supplier has NO order_id).
#   2. The sequencing rule: episodes terminate as soon as every order is
#      resolved, so escalation has to land BEFORE the last order finishes.
#      Oracle confirmed: escalate-after-orders → 0.67, escalate-first → 0.99.
_SYSTEM_PROMPT = """\
You are a fulfillment operations manager handling a supplier-cascade incident.
Output ONLY a JSON action — no prose, no markdown.

Sequencing rule: when a supplier has failed (cascade recovery), escalate the
failed supplier FIRST. Episodes end the moment every order reaches a final
status, so escalating after the last order resolves is too late and forfeits
the escalation credit.

Examples of valid outputs:

Example 1 — Escalate the failed supplier (do this first when one is flagged):
{"action_type": "escalate_supplier", "supplier_id": "SUP-RED"}

Example 2 — Reroute an order to a warehouse that still has stock:
{"action_type": "reroute_order", "order_id": "O1", "warehouse_id": "W2"}

Example 3 — Refund / compensate when no warehouse can fulfill the order:
{"action_type": "refund_or_compensate", "order_id": "O2", "compensation_type": "credit_25"}

Example 4 — Delay an order temporarily:
{"action_type": "delay_order", "order_id": "O1", "reason": "stock_unavailable"}

Example 5 — Do nothing this step:
{"action_type": "noop"}

Output a single JSON object on one line. No code fences. No reasoning text.
"""


def _obs_to_text(obs) -> str:
    snap = {
        "task": obs.task_id,
        "step": obs.step,
        "steps_remaining": obs.steps_remaining,
        "description": obs.task_description,
        "allowed_actions": [
            a.value if hasattr(a, "value") else a for a in obs.allowed_actions
        ],
        "orders":     [o.model_dump() for o in obs.orders],
        "warehouses": [w.model_dump() for w in obs.warehouses],
        "stock":      [s.model_dump() for s in obs.stock],
    }
    return json.dumps(snap, indent=2)


def _extract_action(text: str) -> Dict[str, Any]:
    """Extract JSON action from model output, handling nested braces and wrappers."""
    text = text.strip()
    text = re.sub(r"```(?:json)?", "", text)
    text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    start = text.find("{")
    if start != -1:
        depth = 0
        for i, c in enumerate(text[start:], start):
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start:i+1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        break

    return {"action_type": "noop"}


def _run_episode(model, tokenizer, task_id: str, seed: int, debug: bool = False) -> Dict[str, Any]:
    env = CommerceOpsEnv()
    obs = env.reset(task_id=task_id, seed=seed)
    total_r = 0.0
    steps = 0
    while not obs.done:
        msgs = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": _obs_to_text(obs)},
        ]
        enc = tokenizer.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt",
            return_dict=True,
        )
        enc = {k: v.to(model.device) for k, v in enc.items()}
        input_len = enc["input_ids"].shape[-1]
        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=256, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        raw = tokenizer.decode(out[0][input_len:], skip_special_tokens=True)
        action = _extract_action(raw)

        if debug and steps == 0:
            print(f"\n[DEBUG {task_id} seed={seed}]")
            print(f"  raw output (first 300 chars): {raw[:300]!r}")
            print(f"  parsed action: {action}")

        obs = env.step(action)

        if debug and steps == 0:
            print(f"  env response: reward={obs.reward}, error={obs.last_action_error}")

        total_r += obs.reward
        steps += 1
    result = env.final_score()
    return {
        "task_id": task_id,
        "seed": seed,
        "score": result["score"],
        "total_reward": round(total_r, 4),
        "steps": steps,
        "invalid_actions": result["invalid_actions"],
    }


def _evaluate(model, tokenizer, label: str) -> Dict[str, Any]:
    print(f"\n--- Evaluating [{label}] ---")
    records = []
    first_episode = True
    for task_id in EVAL_TASKS:
        for seed in EVAL_SEEDS:
            r = _run_episode(model, tokenizer, task_id=task_id, seed=seed, debug=first_episode)
            first_episode = False
            records.append(r)
            print(f"  {label}  {task_id} seed={seed}  score={r['score']:.3f}  "
                  f"reward={r['total_reward']:+.3f}  invalid={r['invalid_actions']}")

    by_task: Dict[str, Any] = {}
    for task_id in EVAL_TASKS:
        task_recs = [r for r in records if r["task_id"] == task_id]
        if not task_recs:
            continue
        n = len(task_recs)
        by_task[task_id] = {
            "mean_score":  round(sum(r["score"]        for r in task_recs) / n, 4),
            "mean_reward": round(sum(r["total_reward"] for r in task_recs) / n, 4),
            "invalid_rate": round(
                sum(r["invalid_actions"] for r in task_recs) /
                max(sum(r["steps"] for r in task_recs), 1), 4
            ),
        }
    return {"label": label, "by_task": by_task, "all": records}


# ---------------------------------------------------------------------------
# GRPO reward function — multi-step rollout (same as hf_train.py)
# ---------------------------------------------------------------------------

_GLOBAL_MODEL = None
_GLOBAL_TOKENIZER = None


@torch.no_grad()
def _greedy_action(model, tokenizer, obs) -> Dict[str, Any]:
    msgs = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user",   "content": _obs_to_text(obs)},
    ]
    enc = tokenizer.apply_chat_template(
        msgs, tokenize=True, add_generation_prompt=True,
        return_tensors="pt", return_dict=True,
    )
    enc = {k: v.to(model.device) for k, v in enc.items()}
    input_len = enc["input_ids"].shape[-1]
    out = model.generate(
        **enc, max_new_tokens=128, do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    raw = tokenizer.decode(out[0][input_len:], skip_special_tokens=True)
    return _extract_action(raw)


def grpo_reward_fn(
    prompts: List[str],
    completions: List[str],
    **kwargs,
) -> List[float]:
    """Multi-step rollout reward (see hf_train.py for full rationale)."""
    task_ids = kwargs.get("task_id", [TRAIN_TASKS[0]] * len(completions))
    seeds    = kwargs.get("seed",    [0]              * len(completions))
    model     = _GLOBAL_MODEL
    tokenizer = _GLOBAL_TOKENIZER

    rewards: List[float] = []
    for completion, task_id, seed in zip(completions, task_ids, seeds):
        first_action = _extract_action(completion)
        try:
            env = CommerceOpsEnv()
            env.reset(task_id=task_id, seed=int(seed))
            obs = env.step(first_action)
            steps = 1
            if model is not None and tokenizer is not None:
                while not obs.done and steps < env.state.max_steps:
                    next_action = _greedy_action(model, tokenizer, obs)
                    obs = env.step(next_action)
                    steps += 1
            final = env.final_score()
            rewards.append(float(final.get("score", 0.0)))
        except Exception:
            rewards.append(-0.5)
    return rewards


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


def _make_prompt(task_id: str, seed: int, tokenizer) -> str:
    env = CommerceOpsEnv()
    obs = env.reset(task_id=task_id, seed=seed)
    msgs = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user",   "content": _obs_to_text(obs)},
    ]
    return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


# ---------------------------------------------------------------------------
# Plotting (single-task: just task_3)
# ---------------------------------------------------------------------------


def _plot(baseline: Dict[str, Any], trained: Dict[str, Any], path: str) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not available — skipping plot.")
        return

    tasks = EVAL_TASKS
    x = np.arange(len(tasks))
    w = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    ax = axes[0]
    b_scores = [baseline["by_task"].get(t, {}).get("mean_score", 0.0) for t in tasks]
    t_scores = [trained["by_task"].get(t,  {}).get("mean_score", 0.0) for t in tasks]
    ax.bar(x - w/2, b_scores, w, label="Baseline",     color="#e07b54", alpha=0.85)
    ax.bar(x + w/2, t_scores, w, label="GRPO trained", color="#5b9bd5", alpha=0.85)
    ax.axhline(0.99, color="green", linestyle="--", alpha=0.4, label="Oracle (0.99)")
    ax.set_xticks(x); ax.set_xticklabels(tasks)
    ax.set_ylim(0, 1.1); ax.set_ylabel("Mean Score"); ax.set_title("Episode Score: Baseline vs Trained")
    ax.legend()

    ax2 = axes[1]
    b_rew = [baseline["by_task"].get(t, {}).get("mean_reward", 0.0) for t in tasks]
    t_rew = [trained["by_task"].get(t,  {}).get("mean_reward", 0.0) for t in tasks]
    ax2.bar(x - w/2, b_rew, w, label="Baseline",     color="#e07b54", alpha=0.85)
    ax2.bar(x + w/2, t_rew, w, label="GRPO trained", color="#5b9bd5", alpha=0.85)
    ax2.set_xticks(x); ax2.set_xticklabels(tasks)
    ax2.set_ylabel("Mean Total Reward"); ax2.set_title("Total Reward: Baseline vs Trained")
    ax2.legend()

    plt.suptitle("CommerceOps-Env Task 3 — GRPO Before / After", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Plot saved: {path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    from unsloth import FastLanguageModel
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer

    api = HfApi(token=HF_TOKEN or None)

    print(f"\nLoading {MODEL_NAME} …")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LEN,
        dtype=None,
        load_in_4bit=True,
        token=HF_TOKEN or None,
    )

    FastLanguageModel.for_inference(model)
    baseline_result = _evaluate(model, tokenizer, label="baseline")

    FastLanguageModel.for_training(model)

    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R, target_modules=["q_proj","k_proj","v_proj","o_proj",
                                   "gate_proj","up_proj","down_proj"],
        lora_alpha=LORA_ALPHA, lora_dropout=0, bias="none",
        use_gradient_checkpointing="unsloth", random_state=42,
    )

    prompts = [
        {"prompt": _make_prompt(tid, seed, tokenizer), "task_id": tid, "seed": seed}
        for tid in TRAIN_TASKS
        for seed in TRAIN_SEEDS
    ]
    train_dataset = Dataset.from_list(prompts)
    print(f"\nTraining dataset: {len(train_dataset)} prompts  steps={TRAIN_STEPS}")

    training_args = GRPOConfig(
        output_dir="/tmp/grpo_output_task3",
        max_steps=TRAIN_STEPS,
        per_device_train_batch_size=BATCH_SIZE,
        num_generations=GRPO_N_SAMPLES,
        learning_rate=LR,
        max_prompt_length=1024,
        max_completion_length=256,
        logging_steps=10,
        save_steps=TRAIN_STEPS,
        warmup_steps=10,
        beta=0.01,
        temperature=GRPO_TEMP,
        top_p=GRPO_TOP_P,
        report_to="none",
        seed=42,
    )
    model.train()

    _lora_before = None
    for name, p in model.named_parameters():
        if "lora_A" in name and p.requires_grad:
            _lora_before = (name, p.detach().clone())
            break

    global _GLOBAL_MODEL, _GLOBAL_TOKENIZER
    _GLOBAL_MODEL = model
    _GLOBAL_TOKENIZER = tokenizer

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        reward_funcs=grpo_reward_fn,
        processing_class=tokenizer,
    )
    print(f"\nStarting GRPO (task_3, multi-step rollout reward) — {TRAIN_STEPS} steps …")
    train_result = trainer.train()
    print(f"Training done.  loss={round(train_result.training_loss, 4)}")

    if _lora_before is not None:
        name, before = _lora_before
        after = dict(model.named_parameters())[name].detach()
        delta = (after - before).norm().item()
        rel   = delta / (before.norm().item() + 1e-9)
        print(f"[SANITY] LoRA weight delta on {name}: ||Δ||={delta:.4e}  rel={rel:.4%}")
        print(f"         (rel ≈ 0% → adapters didn't learn; > 0.1% → real movement)")

    FastLanguageModel.for_inference(model)
    trained_result = _evaluate(model, tokenizer, label="trained")

    print(f"\n{'='*54}")
    print(f"  BEFORE vs AFTER (task_3)")
    print(f"{'='*54}")
    print(f"{'Task':<12} {'Metric':<20} {'Baseline':>10} {'Trained':>10} {'Delta':>8}")
    print("-" * 62)
    for task_id in EVAL_TASKS:
        b = baseline_result["by_task"].get(task_id, {})
        t = trained_result["by_task"].get(task_id, {})
        for metric in ("mean_score", "mean_reward", "invalid_rate"):
            bv, tv = b.get(metric, 0.0), t.get(metric, 0.0)
            delta  = tv - bv
            print(f"{task_id:<12} {metric:<20} {bv:>10.4f} {tv:>10.4f} "
                  f"{'+'if delta>=0 else ''}{delta:>7.4f}")
    print("=" * 62)

    with tempfile.TemporaryDirectory() as tmpdir:
        summary = {
            "model": MODEL_NAME,
            "train_tasks": TRAIN_TASKS,
            "train_steps": TRAIN_STEPS,
            "eval_seeds": EVAL_SEEDS,
            "baseline": baseline_result,
            "trained":  trained_result,
            "train_loss": round(train_result.training_loss, 4),
            "log_history": trainer.state.log_history,
            "ts": int(time.time()),
        }
        json_path = os.path.join(tmpdir, "results.json")
        with open(json_path, "w") as f:
            json.dump(summary, f, indent=2)

        plot_path = os.path.join(tmpdir, "reward_curves.png")
        _plot(baseline_result, trained_result, path=plot_path)

        if HF_TOKEN:
            api.create_repo(HUB_RESULTS_REPO, repo_type="dataset", exist_ok=True)
            api.upload_file(path_or_fileobj=json_path,
                            path_in_repo="results.json",
                            repo_id=HUB_RESULTS_REPO, repo_type="dataset")
            if os.path.exists(plot_path):
                api.upload_file(path_or_fileobj=plot_path,
                                path_in_repo="reward_curves.png",
                                repo_id=HUB_RESULTS_REPO, repo_type="dataset")
            print(f"\nResults pushed → https://huggingface.co/datasets/{HUB_RESULTS_REPO}")

            trainer.save_model("/tmp/grpo_output_task3/final")
            tokenizer.save_pretrained("/tmp/grpo_output_task3/final")
            api.create_repo(HUB_MODEL_REPO, exist_ok=True)
            api.upload_folder(folder_path="/tmp/grpo_output_task3/final",
                              repo_id=HUB_MODEL_REPO)
            print(f"Model pushed    → https://huggingface.co/{HUB_MODEL_REPO}")
        else:
            print("\nHF_TOKEN not set — skipping Hub push. Results in /tmp/grpo_output_task3/")
            print(json.dumps({
                "baseline": {t: baseline_result["by_task"].get(t) for t in EVAL_TASKS},
                "trained":  {t: trained_result["by_task"].get(t)  for t in EVAL_TASKS},
            }, indent=2))

    print("\nDone.")


if __name__ == "__main__":
    main()
