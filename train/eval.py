"""Standalone evaluation script for CommerceOps-Env.

Owned by Tendulkar per context.md team split.

Runs three policies against the environment:
  oracle  — follows the ground-truth reference plan exactly
  random  — picks random actions from the allowed set (lower bound)
  model   — (optional) runs an LLM loaded from a local path or Hub repo

Produces:
  - A before/after comparison table printed to stdout
  - ``eval_results.json`` with per-episode data for the demo notebook
  - ``reward_curves.png`` showing score distributions and cumulative reward

Usage (no model, just baselines — enough for the demo):
    python train/eval.py

Usage (with a trained checkpoint):
    python train/eval.py --model ./grpo_output/final --model-label grpo_trained

Usage (quick smoke test):
    python train/eval.py --fast-dev

All evaluation is deterministic given the same seeds.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional

# Ensure repo root is on the path regardless of where the script is called from.
_REPO_ROOT = os.path.join(os.path.dirname(__file__), "..")
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from environment import CommerceOpsEnv
from models import ActionType, ALLOWED_ACTIONS_BY_TASK, TaskType
from tasks import get_task_bundle

try:
    from train.metrics import (
        EpisodeRecord,
        MetricsLogger,
        StepRecord,
        TrainingMetricsTracker,
        compare_policies,
    )
except ImportError:
    from metrics import (  # type: ignore[no-redef]
        EpisodeRecord,
        MetricsLogger,
        StepRecord,
        TrainingMetricsTracker,
        compare_policies,
    )


# ---------------------------------------------------------------------------
# Policy implementations
# ---------------------------------------------------------------------------


def oracle_policy(env: CommerceOpsEnv) -> Dict[str, Any]:
    """Follow the ground-truth reference plan exactly."""
    state = env.state
    gt = state.ground_truth
    kind = gt.get("kind", "")

    if kind == "warehouse_assignment":
        oid  = gt.get("order_id", "O1")
        best = gt.get("best_warehouse")
        return {"action_type": "assign_warehouse", "order_id": oid, "warehouse_id": best}

    if kind == "multi_order_triage":
        plan = gt.get("plan", {})
        for order in state.orders:
            oid = order.order_id
            if order.status != "pending":
                continue
            exp = plan.get(oid)
            if exp is None:
                continue
            action: Dict[str, Any] = {"action_type": exp["action_type"], "order_id": oid}
            if exp["action_type"] == "assign_warehouse":
                action["warehouse_id"] = exp["warehouse_id"]
            elif exp["action_type"] == "split_shipment":
                action["allocations"] = exp["allocations"]
            elif exp["action_type"] == "delay_order":
                action["reason"] = "oracle_delay"
            return action

    if kind == "cascade_recovery":
        exp_sup = gt.get("expected_supplier_escalation")
        if exp_sup and not state.policy_flags.get("supplier_escalated"):
            return {"action_type": "escalate_supplier", "supplier_id": exp_sup}
        exp_actions = gt.get("expected_actions", {})
        for order in state.orders:
            if order.order_id in exp_actions and order.status == "pending":
                exp = exp_actions[order.order_id]
                action = {"action_type": exp["action_type"], "order_id": order.order_id}
                if exp["action_type"] == "reroute_order":
                    action["warehouse_id"] = exp.get("warehouse_id", "W2")
                elif exp["action_type"] == "refund_or_compensate":
                    action["compensation_type"] = exp.get("compensation_type", "credit_10")
                return action

    return {"action_type": "noop"}


def random_policy(env: CommerceOpsEnv, rng: random.Random) -> Dict[str, Any]:
    """Choose a random valid action from the allowed set."""
    state = env.state
    allowed = [a for a in state.allowed_actions if a != "noop"]
    if not allowed:
        return {"action_type": "noop"}

    action_type = rng.choice(allowed)
    pending_orders = [o for o in state.orders if o.status == "pending"]
    warehouses     = [w.warehouse_id for w in state.warehouses]

    if not pending_orders:
        return {"action_type": "noop"}

    order = rng.choice(pending_orders)

    if action_type == "assign_warehouse":
        return {
            "action_type": "assign_warehouse",
            "order_id":    order.order_id,
            "warehouse_id": rng.choice(warehouses) if warehouses else "W1",
        }

    if action_type == "split_shipment" and len(warehouses) >= 2:
        whs = rng.sample(warehouses, 2)
        qty = order.quantity_requested
        q1  = max(1, qty // 2)
        q2  = max(1, qty - q1)
        return {
            "action_type": "split_shipment",
            "order_id":    order.order_id,
            "allocations": [
                {"warehouse_id": whs[0], "quantity": q1},
                {"warehouse_id": whs[1], "quantity": q2},
            ],
        }

    if action_type == "delay_order":
        return {"action_type": "delay_order", "order_id": order.order_id,
                "reason": "random_delay"}

    if action_type == "prioritize_order":
        return {"action_type": "prioritize_order", "order_id": order.order_id}

    if action_type == "reroute_order":
        return {"action_type": "reroute_order", "order_id": order.order_id,
                "warehouse_id": rng.choice(warehouses) if warehouses else "W1"}

    if action_type == "escalate_supplier":
        return {"action_type": "escalate_supplier", "supplier_id": "SUP-RANDOM"}

    if action_type == "refund_or_compensate":
        return {"action_type": "refund_or_compensate", "order_id": order.order_id,
                "compensation_type": rng.choice(["credit_5", "credit_10", "credit_25"])}

    return {"action_type": "noop"}


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------


def run_episode_with_policy(
    task_id: str,
    seed: int,
    policy_fn,
    policy_name: str,
    run_id: str,
    episode_idx: int,
    logger: Optional[MetricsLogger] = None,
    **policy_kwargs: Any,
) -> EpisodeRecord:
    env = CommerceOpsEnv()
    obs = env.reset(task_id=task_id, seed=seed)
    step_idx = 0
    total_reward = 0.0

    while not obs.done:
        action = policy_fn(env, **policy_kwargs)
        obs    = env.step(action)
        total_reward += obs.reward
        step_idx += 1

        if logger is not None:
            logger.log_step(StepRecord(
                run_id=run_id,
                policy=policy_name,
                task_id=task_id,
                seed=seed,
                episode_idx=episode_idx,
                step=step_idx,
                action_type=action.get("action_type", "unknown"),
                reward=obs.reward,
                cumulative_reward=obs.cumulative_reward,
                is_invalid=obs.last_action_error == "invalid_action",
                is_repeat=obs.reward_breakdown.get("repeat_penalty", 0.0) < 0,
                error=obs.last_action_error,
            ))

    result = env.final_score()
    record = EpisodeRecord(
        run_id=run_id,
        policy=policy_name,
        task_id=task_id,
        seed=seed,
        episode_idx=episode_idx,
        score=result["score"],
        total_reward=round(total_reward, 6),
        steps=result["steps"],
        invalid_actions=result["invalid_actions"],
        repeat_actions=result["repeat_actions"],
        breakdown=result["breakdown"],
    )

    if logger is not None:
        logger.log_episode(record)

    return record


# ---------------------------------------------------------------------------
# Model-based evaluator (optional — only imported when --model is set)
# ---------------------------------------------------------------------------


def run_episode_with_model(
    task_id: str,
    seed: int,
    model,
    tokenizer,
    policy_name: str,
    run_id: str,
    episode_idx: int,
    logger: Optional[MetricsLogger] = None,
    max_new_tokens: int = 256,
) -> EpisodeRecord:
    """Run a single episode using a loaded LLM."""
    import json as _json
    import re
    import torch

    # Inline minimal obs→prompt (avoids circular import from train/ notebook)
    def _build_prompt(obs) -> str:
        snap = {
            "task": obs.task_id,
            "step": obs.step,
            "steps_remaining": obs.steps_remaining,
            "description": obs.task_description,
            "policy_flags": obs.policy_flags,
            "allowed_actions": [
                a.value if hasattr(a, "value") else a for a in obs.allowed_actions
            ],
            "orders":     [o.model_dump() for o in obs.orders],
            "warehouses": [w.model_dump() for w in obs.warehouses],
            "stock":      [s.model_dump() for s in obs.stock],
        }
        messages = [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user",   "content": _json.dumps(snap, indent=2)},
        ]
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def _extract(text: str):
        text = re.sub(r"^```(?:json)?\n?", "", text.strip())
        text = re.sub(r"\n?```$", "", text)
        try:
            return _json.loads(text)
        except _json.JSONDecodeError:
            m = re.search(r"\{[^{}]*\}", text, re.DOTALL)
            if m:
                try:
                    return _json.loads(m.group())
                except _json.JSONDecodeError:
                    pass
        return {"action_type": "noop"}

    env = CommerceOpsEnv()
    obs = env.reset(task_id=task_id, seed=seed)
    step_idx = 0
    total_reward = 0.0

    while not obs.done:
        prompt = _build_prompt(obs)
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
        with torch.no_grad():
            out = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        gen_text = tokenizer.decode(out[0][input_ids.shape[-1]:], skip_special_tokens=True)
        action   = _extract(gen_text)
        obs      = env.step(action)
        total_reward += obs.reward
        step_idx += 1

        if logger is not None:
            logger.log_step(StepRecord(
                run_id=run_id,
                policy=policy_name,
                task_id=task_id,
                seed=seed,
                episode_idx=episode_idx,
                step=step_idx,
                action_type=action.get("action_type", "unknown"),
                reward=obs.reward,
                cumulative_reward=obs.cumulative_reward,
                is_invalid=obs.last_action_error == "invalid_action",
                is_repeat=obs.reward_breakdown.get("repeat_penalty", 0.0) < 0,
                error=obs.last_action_error,
            ))

    result = env.final_score()
    record = EpisodeRecord(
        run_id=run_id,
        policy=policy_name,
        task_id=task_id,
        seed=seed,
        episode_idx=episode_idx,
        score=result["score"],
        total_reward=round(total_reward, 6),
        steps=result["steps"],
        invalid_actions=result["invalid_actions"],
        repeat_actions=result["repeat_actions"],
        breakdown=result["breakdown"],
    )
    if logger is not None:
        logger.log_episode(record)
    return record


_SYSTEM_PROMPT = (
    "You are a fulfillment operations manager. "
    "Respond with a single JSON action object and nothing else."
)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_results(
    all_records: Dict[str, List[EpisodeRecord]],
    output_path: str = "./train/logs/reward_curves.png",
) -> None:
    """Generate the before/after chart saved to ``output_path``."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not installed — skipping plot.")
        return

    policies = list(all_records.keys())
    tasks    = sorted({r.task_id for recs in all_records.values() for r in recs})
    colors   = plt.cm.tab10.colors  # type: ignore[attr-defined]

    fig, axes = plt.subplots(len(tasks), 2, figsize=(12, 4 * len(tasks)))
    if len(tasks) == 1:
        axes = [axes]

    for row_idx, task_id in enumerate(tasks):
        ax_score  = axes[row_idx][0]
        ax_reward = axes[row_idx][1]

        for col_idx, policy in enumerate(policies):
            recs = [r for r in all_records[policy] if r.task_id == task_id]
            if not recs:
                continue
            color = colors[col_idx % len(colors)]
            scores  = [r.score         for r in recs]
            rewards = [r.total_reward  for r in recs]

            # Box + scatter for scores
            bp = ax_score.boxplot(
                scores,
                positions=[col_idx],
                widths=0.5,
                patch_artist=True,
                boxprops=dict(facecolor=color, alpha=0.4),
                medianprops=dict(color="black", linewidth=2),
                whiskerprops=dict(color=color),
                capprops=dict(color=color),
                flierprops=dict(marker="o", color=color, markersize=4),
            )
            jitter = np.random.default_rng(42).uniform(-0.15, 0.15, len(scores))
            ax_score.scatter(
                [col_idx + j for j in jitter], scores,
                color=color, alpha=0.6, s=20, zorder=3
            )

            # Cumulative reward line (sorted by seed for consistent x-axis)
            ax_reward.plot(
                range(len(rewards)), rewards,
                marker="o", ms=4, color=color, alpha=0.7, label=policy
            )

        ax_score.set_xticks(range(len(policies)))
        ax_score.set_xticklabels(policies, rotation=15, ha="right")
        ax_score.set_title(f"{task_id} — Episode Score")
        ax_score.set_ylabel("Score (0–1)")
        ax_score.set_ylim(-0.05, 1.05)
        ax_score.axhline(y=1.0, color="green", linestyle="--", alpha=0.3, label="perfect")
        ax_score.legend(fontsize=7)

        ax_reward.set_title(f"{task_id} — Total Episode Reward")
        ax_reward.set_xlabel("Episode (by seed)")
        ax_reward.set_ylabel("Total reward")
        ax_reward.legend(fontsize=7)

    plt.suptitle("CommerceOps-Env — Policy Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Chart saved: {output_path}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate oracle, random, and (optionally) a trained model."
    )
    parser.add_argument("--tasks",    nargs="+", default=["task_1", "task_2"],
                        help="Task IDs to evaluate (default: task_1 task_2)")
    parser.add_argument("--seeds",    nargs="+", type=int,
                        default=list(range(8)),
                        help="Seeds to evaluate (default: 0..7)")
    parser.add_argument("--model",    default=None,
                        help="Path or Hub repo for trained model (optional)")
    parser.add_argument("--model-label", default="grpo_trained",
                        help="Label for the trained model in the output table")
    parser.add_argument("--log-dir",  default="./train/logs",
                        help="Directory for JSONL logs and output chart")
    parser.add_argument("--out-json", default="./train/logs/eval_results.json",
                        help="Path for JSON summary output")
    parser.add_argument("--no-plot",  action="store_true",
                        help="Skip matplotlib chart generation")
    parser.add_argument("--fast-dev", action="store_true",
                        help="Use only 2 seeds for quick smoke test")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    seeds   = [0, 1] if args.fast_dev else args.seeds
    tasks   = args.tasks
    run_id  = f"eval_{int(time.time())}"
    log_dir = args.log_dir

    print(f"\n{'='*62}")
    print(f"  CommerceOps-Env Evaluation  run_id={run_id}")
    print(f"  tasks={tasks}  seeds={seeds}")
    print(f"{'='*62}\n")

    all_records: Dict[str, List[EpisodeRecord]] = {}
    all_summaries: Dict[str, Dict[str, Any]] = {}

    # ── Oracle ────────────────────────────────────────────────────────
    print("--- Oracle policy ---")
    oracle_logger = MetricsLogger(run_id=run_id, policy="oracle", log_dir=log_dir)
    oracle_records: List[EpisodeRecord] = []
    for task_id in tasks:
        for ep_idx, seed in enumerate(seeds):
            rec = run_episode_with_policy(
                task_id, seed,
                policy_fn=lambda env: oracle_policy(env),
                policy_name="oracle",
                run_id=run_id,
                episode_idx=ep_idx,
                logger=oracle_logger,
            )
            oracle_records.append(rec)
            print(f"  oracle  {task_id} seed={seed:2d}  score={rec.score:.3f}  "
                  f"reward={rec.total_reward:+.3f}  invalid={rec.invalid_actions}")
    all_records["oracle"] = oracle_records
    all_summaries["oracle"] = oracle_logger.summary()

    # ── Random ────────────────────────────────────────────────────────
    print("\n--- Random policy ---")
    random_logger = MetricsLogger(run_id=run_id, policy="random", log_dir=log_dir)
    random_records: List[EpisodeRecord] = []
    for task_id in tasks:
        for ep_idx, seed in enumerate(seeds):
            rng = random.Random(seed * 137 + 7)  # deterministic but different from env seed
            rec = run_episode_with_policy(
                task_id, seed,
                policy_fn=lambda env, rng=rng: random_policy(env, rng),
                policy_name="random",
                run_id=run_id,
                episode_idx=ep_idx,
                logger=random_logger,
            )
            random_records.append(rec)
            print(f"  random  {task_id} seed={seed:2d}  score={rec.score:.3f}  "
                  f"reward={rec.total_reward:+.3f}  invalid={rec.invalid_actions}")
    all_records["random"] = random_records
    all_summaries["random"] = random_logger.summary()

    # ── Trained model (optional) ──────────────────────────────────────
    if args.model:
        print(f"\n--- Model: {args.model} ---")
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: PLC0415
            import torch  # noqa: PLC0415

            tokenizer = AutoTokenizer.from_pretrained(args.model)
            model = AutoModelForCausalLM.from_pretrained(
                args.model,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None,
            )
            model.eval()

            model_logger = MetricsLogger(run_id=run_id, policy=args.model_label, log_dir=log_dir)
            model_records: List[EpisodeRecord] = []
            for task_id in tasks:
                for ep_idx, seed in enumerate(seeds):
                    rec = run_episode_with_model(
                        task_id, seed,
                        model=model,
                        tokenizer=tokenizer,
                        policy_name=args.model_label,
                        run_id=run_id,
                        episode_idx=ep_idx,
                        logger=model_logger,
                    )
                    model_records.append(rec)
                    print(f"  {args.model_label}  {task_id} seed={seed:2d}  "
                          f"score={rec.score:.3f}  reward={rec.total_reward:+.3f}")
            all_records[args.model_label] = model_records
            all_summaries[args.model_label] = model_logger.summary()
        except Exception as exc:
            print(f"  Model eval failed: {exc}")

    # ── Comparison table ──────────────────────────────────────────────
    print("\n" + compare_policies(*all_summaries.values()))

    # ── Per-task detail ───────────────────────────────────────────────
    print("\nPer-task breakdown:")
    print(f"  {'Policy':<18} {'Task':<12} {'MeanScore':>10} {'MeanReward':>12} {'InvalidRate':>13}")
    print("  " + "-" * 67)
    for policy_name, logger in [
        ("oracle",           oracle_logger),
        ("random",           random_logger),
        *( [(args.model_label, model_logger)] if args.model and model_records else [] ),
    ]:
        for task_id in tasks:
            s = logger.summary(task_id)
            print(f"  {policy_name:<18} {task_id:<12} "
                  f"{s.get('mean_score', 0):>10.4f} "
                  f"{s.get('mean_total_reward', 0):>12.4f} "
                  f"{s.get('invalid_rate', 0):>13.4f}")

    # ── Save JSON ─────────────────────────────────────────────────────
    output = {
        "run_id":    run_id,
        "tasks":     tasks,
        "seeds":     seeds,
        "summaries": {
            policy: {task_id: logger.summary(task_id).copy() for task_id in tasks}
            for policy, logger in [
                ("oracle", oracle_logger),
                ("random", random_logger),
            ]
        },
        "episodes": {
            policy: [r.to_dict() for r in recs]
            for policy, recs in all_records.items()
        },
    }
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved: {args.out_json}")

    # ── Plot ──────────────────────────────────────────────────────────
    if not args.no_plot:
        chart_path = os.path.join(log_dir, "reward_curves.png")
        plot_results(all_records, output_path=chart_path)

    print("\nDone.")


if __name__ == "__main__":
    main()
