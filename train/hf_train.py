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
"""CommerceOps-Env — GRPO training script for HF Jobs.

Designed to run as a HF Jobs UV script (PEP 723 format).
Pushes the trained model, eval results JSON, and reward curve PNG to Hub.

Env vars (pass as secrets / env in hf_jobs()):
  HF_TOKEN          required — write token to push results
  HUB_MODEL_REPO    e.g. "YOUR_USERNAME/commerce-ops-grpo"  (default: derive from whoami)
  HUB_RESULTS_REPO  e.g. "YOUR_USERNAME/commerce-ops-results" (default: derive from whoami)
  TRAIN_STEPS       int, default 200
  FAST_DEV          "1" to run 5 steps + 2 seeds only (smoke test)

Usage (from CLI after hf login):
  hf jobs uv run train/hf_train.py \\
    --flavor l4x1 \\
    --timeout 2h \\
    --env TRAIN_STEPS=200 \\
    --secret HF_TOKEN=$HF_TOKEN \\
    --env HUB_MODEL_REPO=YOUR_USERNAME/commerce-ops-grpo \\
    --env HUB_RESULTS_REPO=YOUR_USERNAME/commerce-ops-results
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
TRAIN_STEPS     = int(os.environ.get("TRAIN_STEPS", "80" if FAST_DEV else "200"))
EVAL_SEEDS      = [0, 1] if FAST_DEV else list(range(8))
TRAIN_SEEDS     = list(range(8)) if FAST_DEV else list(range(8))
TRAIN_TASKS     = ["task_1", "task_2"] if FAST_DEV else ["task_1", "task_2"]
MODEL_NAME      = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-3B-Instruct")
MAX_SEQ_LEN     = 2048
LORA_R          = 16
LORA_ALPHA      = 32
LR              = 1e-5         # 2× from 5e-6 — KL was 0.012, way too low
GRPO_N_SAMPLES  = 8            # 2× from 4 — give each group more chances at variance
BATCH_SIZE      = 2
GRPO_TEMP       = 1.2          # > 1.0 forces variance among the N samples per prompt
GRPO_TOP_P      = 0.95

# Hub targets — only call whoami when the env vars aren't already set
HUB_MODEL_REPO   = os.environ.get("HUB_MODEL_REPO")
HUB_RESULTS_REPO = os.environ.get("HUB_RESULTS_REPO")

if not HUB_MODEL_REPO or not HUB_RESULTS_REPO:
    try:
        _me = whoami(token=HF_TOKEN, cache=True)["name"]
    except Exception:
        _me = "user"
    HUB_MODEL_REPO   = HUB_MODEL_REPO   or f"{_me}/commerce-ops-grpo"
    HUB_RESULTS_REPO = HUB_RESULTS_REPO or f"{_me}/commerce-ops-results"

print(f"{'='*62}")
print(f"  CommerceOps-Env GRPO Training")
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

# Derive the project root relative to this script (train/hf_train.py → ../)
_SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_SCRIPT_DIR)

if REPO_URL:
    # HF Jobs path: clone or hard-reset to latest so fixes are always picked up
    if not os.path.exists(REPO_DIR):
        os.system(f"git clone --depth 1 {REPO_URL} {REPO_DIR}")
    else:
        os.system(
            f"git -C {REPO_DIR} fetch --depth 1 origin main "
            f"&& git -C {REPO_DIR} reset --hard origin/main"
        )
    sys.path.insert(0, REPO_DIR)
else:
    # Local / Colab path: use the project files already on disk (no clone needed)
    sys.path.insert(0, _PROJECT_DIR)

from environment import CommerceOpsEnv  # noqa: E402
from tasks import get_task_bundle       # noqa: E402

# ═══════════════════════════════════════════════════════════════════════════
# CHECKPOINT 1: Verify imports came from the right place
# ═══════════════════════════════════════════════════════════════════════════
import environment as _env_module
print(f"[CHECKPOINT] environment.py loaded from: {_env_module.__file__}")
print(f"[CHECKPOINT] FAST_DEV={FAST_DEV}, TRAIN_STEPS={TRAIN_STEPS}, TRAIN_TASKS={TRAIN_TASKS}")

# ---------------------------------------------------------------------------
# Prompt / action utilities (inline — no circular import from notebooks)
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a fulfillment operations manager. Output ONLY a JSON action — no prose, no markdown.

Examples of valid outputs:

Example 1 — Assign an order to a warehouse:
{"action_type": "assign_warehouse", "order_id": "O1", "warehouse_id": "W1"}

Example 2 — Split an order across two warehouses:
{"action_type": "split_shipment", "order_id": "O1", "allocations": [{"warehouse_id": "W1", "quantity": 5}, {"warehouse_id": "W2", "quantity": 3}]}

Example 3 — Delay an order:
{"action_type": "delay_order", "order_id": "O1", "reason": "stock_unavailable"}

Example 4 — Do nothing this step:
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
    
    # Strip markdown code fences anywhere in the text
    text = re.sub(r"```(?:json)?", "", text)
    text = text.strip()
    
    # Try 1: Direct parse (ideal case - model outputs pure JSON)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Try 2: Find the OUTERMOST balanced braces (handles nested JSON)
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
    
    # Fallback: return noop (valid for all task types)
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
        
        # Debug: show what the model generated and how it was parsed
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
    for task_id in ["task_1", "task_2"]:
        for seed in EVAL_SEEDS:
            # Debug first episode only to see what the model is generating
            r = _run_episode(model, tokenizer, task_id=task_id, seed=seed, debug=first_episode)
            first_episode = False
            records.append(r)
            print(f"  {label}  {task_id} seed={seed}  score={r['score']:.3f}  "
                  f"reward={r['total_reward']:+.3f}  invalid={r['invalid_actions']}")

    by_task: Dict[str, Any] = {}
    for task_id in ["task_1", "task_2"]:
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
# GRPO reward function
# ---------------------------------------------------------------------------


def grpo_reward_fn(
    prompts: List[str],
    completions: List[str],
    **kwargs,
) -> List[float]:
    """Single-step reward: env.reset() then env.step(model_action), use that reward.

    Diagnostic confirmed this signal works: V2 prompt yields +0.95 on task_1 baseline,
    -0.35 on invalid actions. Plenty of variance for GRPO group advantages.
    """
    task_ids = kwargs.get("task_id", ["task_1"] * len(completions))
    seeds    = kwargs.get("seed",    [0]        * len(completions))
    rewards  = []
    for completion, task_id, seed in zip(completions, task_ids, seeds):
        action = _extract_action(completion)
        try:
            env = CommerceOpsEnv()
            env.reset(task_id=task_id, seed=int(seed))
            obs = env.step(action)
            rewards.append(float(obs.reward))
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
# Plotting (saves to file, then uploads)
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

    tasks = ["task_1", "task_2"]
    x = np.arange(len(tasks))
    w = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Score comparison bars
    ax = axes[0]
    b_scores  = [baseline["by_task"].get(t, {}).get("mean_score", 0.0)  for t in tasks]
    t_scores  = [trained["by_task"].get(t,  {}).get("mean_score", 0.0)  for t in tasks]
    ax.bar(x - w/2, b_scores, w, label="Baseline",     color="#e07b54", alpha=0.85)
    ax.bar(x + w/2, t_scores, w, label="GRPO trained", color="#5b9bd5", alpha=0.85)
    ax.axhline(0.99, color="green", linestyle="--", alpha=0.4, label="Oracle (0.99)")
    ax.set_xticks(x); ax.set_xticklabels(tasks)
    ax.set_ylim(0, 1.1); ax.set_ylabel("Mean Score"); ax.set_title("Episode Score: Baseline vs Trained")
    ax.legend()

    # Reward comparison bars
    ax2 = axes[1]
    b_rew = [baseline["by_task"].get(t, {}).get("mean_reward", 0.0) for t in tasks]
    t_rew = [trained["by_task"].get(t,  {}).get("mean_reward", 0.0) for t in tasks]
    ax2.bar(x - w/2, b_rew, w, label="Baseline",     color="#e07b54", alpha=0.85)
    ax2.bar(x + w/2, t_rew, w, label="GRPO trained", color="#5b9bd5", alpha=0.85)
    ax2.set_xticks(x); ax2.set_xticklabels(tasks)
    ax2.set_ylabel("Mean Total Reward"); ax2.set_title("Total Reward: Baseline vs Trained")
    ax2.legend()

    plt.suptitle("CommerceOps-Env GRPO Training — Before / After", fontsize=14, fontweight="bold")
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

    # ── 1. Load model ─────────────────────────────────────────────────
    print(f"\nLoading {MODEL_NAME} …")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LEN,
        dtype=None,
        load_in_4bit=True,
        token=HF_TOKEN or None,
    )

    # ── 2. Baseline eval ──────────────────────────────────────────────
    FastLanguageModel.for_inference(model)
    baseline_result = _evaluate(model, tokenizer, label="baseline")

    # Reset inference flag on the base model before PEFT wrapping;
    # for_training on a PeftModelForCausalLM fails because the flag lives
    # on the inner base model, not the wrapper.
    FastLanguageModel.for_training(model)

    # ── 3. LoRA adapters ──────────────────────────────────────────────
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R, target_modules=["q_proj","k_proj","v_proj","o_proj",
                                   "gate_proj","up_proj","down_proj"],
        lora_alpha=LORA_ALPHA, lora_dropout=0, bias="none",
        use_gradient_checkpointing="unsloth", random_state=42,
    )

    # ── 4. Dataset ────────────────────────────────────────────────────
    prompts = [
        {"prompt": _make_prompt(tid, seed, tokenizer), "task_id": tid, "seed": seed}
        for tid in TRAIN_TASKS
        for seed in TRAIN_SEEDS
    ]
    train_dataset = Dataset.from_list(prompts)
    print(f"\nTraining dataset: {len(train_dataset)} prompts  steps={TRAIN_STEPS}")

    # ── 5. GRPO training ──────────────────────────────────────────────
    training_args = GRPOConfig(
        output_dir="/tmp/grpo_output",
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
        # Force diversity in the N samples per prompt — last run had
        # frac_reward_zero_std=0.8, meaning groups had no variance to learn from.
        temperature=GRPO_TEMP,
        top_p=GRPO_TOP_P,
        report_to="none",
        seed=42,
    )
    model.train()

    # Snapshot a LoRA A matrix before training so we can verify weights actually moved.
    _lora_before = None
    for name, p in model.named_parameters():
        if "lora_A" in name and p.requires_grad:
            _lora_before = (name, p.detach().clone())
            break

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        reward_funcs=grpo_reward_fn,
        processing_class=tokenizer,
    )
    print(f"\nStarting GRPO — {TRAIN_STEPS} steps …")
    train_result = trainer.train()
    print(f"Training done.  loss={round(train_result.training_loss, 4)}")

    if _lora_before is not None:
        name, before = _lora_before
        after = dict(model.named_parameters())[name].detach()
        delta = (after - before).norm().item()
        rel   = delta / (before.norm().item() + 1e-9)
        print(f"[SANITY] LoRA weight delta on {name}: ||Δ||={delta:.4e}  rel={rel:.4%}")
        print(f"         (rel ≈ 0% → adapters didn't learn; > 0.1% → real movement)")

    # ── 6. Post-training eval ─────────────────────────────────────────
    FastLanguageModel.for_inference(model)
    trained_result = _evaluate(model, tokenizer, label="trained")

    # ── 7. Comparison table ───────────────────────────────────────────
    print(f"\n{'='*54}")
    print(f"  BEFORE vs AFTER")
    print(f"{'='*54}")
    print(f"{'Task':<12} {'Metric':<20} {'Baseline':>10} {'Trained':>10} {'Delta':>8}")
    print("-" * 62)
    for task_id in ["task_1", "task_2"]:
        b = baseline_result["by_task"].get(task_id, {})
        t = trained_result["by_task"].get(task_id, {})
        for metric in ("mean_score", "mean_reward", "invalid_rate"):
            bv, tv = b.get(metric, 0.0), t.get(metric, 0.0)
            delta  = tv - bv
            print(f"{task_id:<12} {metric:<20} {bv:>10.4f} {tv:>10.4f} "
                  f"{'+'if delta>=0 else ''}{delta:>7.4f}")
    print("=" * 62)

    # ── 8. Save and push results ──────────────────────────────────────
    with tempfile.TemporaryDirectory() as tmpdir:
        # JSON summary
        summary = {
            "model": MODEL_NAME,
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

        # Reward curve PNG
        plot_path = os.path.join(tmpdir, "reward_curves.png")
        _plot(baseline_result, trained_result, path=plot_path)

        if HF_TOKEN:
            # Push results dataset
            api.create_repo(HUB_RESULTS_REPO, repo_type="dataset", exist_ok=True)
            api.upload_file(path_or_fileobj=json_path,
                            path_in_repo="results.json",
                            repo_id=HUB_RESULTS_REPO, repo_type="dataset")
            if os.path.exists(plot_path):
                api.upload_file(path_or_fileobj=plot_path,
                                path_in_repo="reward_curves.png",
                                repo_id=HUB_RESULTS_REPO, repo_type="dataset")
            print(f"\nResults pushed → https://huggingface.co/datasets/{HUB_RESULTS_REPO}")

            # Push model
            trainer.save_model("/tmp/grpo_output/final")
            tokenizer.save_pretrained("/tmp/grpo_output/final")
            api.create_repo(HUB_MODEL_REPO, exist_ok=True)
            api.upload_folder(folder_path="/tmp/grpo_output/final",
                              repo_id=HUB_MODEL_REPO)
            print(f"Model pushed    → https://huggingface.co/{HUB_MODEL_REPO}")
        else:
            print("\nHF_TOKEN not set — skipping Hub push. Results in /tmp/grpo_output/")
            print(json.dumps({
                "baseline": {t: baseline_result["by_task"].get(t) for t in ["task_1","task_2"]},
                "trained":  {t: trained_result["by_task"].get(t)  for t in ["task_1","task_2"]},
            }, indent=2))

    print("\nDone.")


if __name__ == "__main__":
    main()
