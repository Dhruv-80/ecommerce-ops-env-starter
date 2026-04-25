"""Step-level reward combiner for CommerceOps-Env.

Takes signals from ``verifier.verify_step()`` and produces a single scalar
reward plus a breakdown dict for logging and manual rollout inspection.

Design (context.md §Reward Design):
  +0.10  schema compliance (valid action format — guaranteed before this is
         called, so awarded every time we reach here)
  +0.20  correct entity targeted
  +0.30  correct action type for that entity
  +0.40  state update matches ground truth (full credit)
  partial credit: some signals carry a ``partial_credit`` float (0–1) that
         scales the state-update component.
  ─────
   1.00  max per step

Penalties (context.md §Anti-Reward-Hacking):
  -0.10  repeat action  (same action_type+order_id as the immediately prior step)
  -0.05  step penalty   (small constant to prefer shorter solutions)
  -0.20  collateral damage flag from env (e.g. assign infeasible order)
  -0.50  budget exceeded (step beyond max_steps — env should end before, but belt-and-suspenders)

Reward range is intentionally kept in [-0.85, 1.00] so the GRPO training
signal stays within a predictable band and the baseline clearly struggles
(≤ 0 average per step without guidance) while a good agent trends positive.
"""

from __future__ import annotations

from typing import Any, Dict


# Component weights
_W_SCHEMA = 0.10
_W_ENTITY = 0.20
_W_ACTION = 0.30
_W_STATE  = 0.40

# Penalty magnitudes
_P_REPEAT    = 0.10
_P_STEP      = 0.05
_P_COLLAT    = 0.20
_P_BUDGET    = 0.50


def compute_step_reward(
    *,
    signals: Dict[str, Any],
) -> Dict[str, Any]:
    """Convert verifier signals into a scalar reward and breakdown dict.

    Parameters
    ----------
    signals:
        Dict returned by ``verifier.verify_step()``, possibly augmented by
        the environment with ``_is_repeat``, ``_within_budget``, and
        ``_collateral_damage`` keys before being passed here.
    """
    reward = 0.0
    breakdown: Dict[str, float] = {
        "schema_compliance": 0.0,
        "correct_entity": 0.0,
        "correct_action": 0.0,
        "state_update": 0.0,
        "repeat_penalty": 0.0,
        "step_penalty": 0.0,
        "collateral_penalty": 0.0,
        "budget_penalty": 0.0,
    }

    # ----- Positive components -----

    # Schema compliance: always awarded when we reach reward computation
    # (the env rejects invalid schemas before calling us).
    reward += _W_SCHEMA
    breakdown["schema_compliance"] = _W_SCHEMA

    if signals.get("correct_entity"):
        reward += _W_ENTITY
        breakdown["correct_entity"] = _W_ENTITY

    if signals.get("correct_action_for_entity"):
        reward += _W_ACTION
        breakdown["correct_action"] = _W_ACTION

    if signals.get("state_update_correct"):
        partial = float(signals.get("partial_credit", 1.0))
        component = round(_W_STATE * partial, 6)
        reward += component
        breakdown["state_update"] = component
    elif signals.get("partial_credit") is not None:
        # partial_credit present but state_update_correct is False — give a
        # smaller fractional credit to keep gradient alive.
        partial = float(signals["partial_credit"]) * 0.5
        component = round(_W_STATE * partial, 6)
        reward += component
        breakdown["state_update"] = component

    # ----- Penalties -----

    if signals.get("is_repeat"):
        reward -= _P_REPEAT
        breakdown["repeat_penalty"] = -_P_REPEAT

    # Constant step cost (encourages shorter paths).
    reward -= _P_STEP
    breakdown["step_penalty"] = -_P_STEP

    if signals.get("collateral_damage"):
        reward -= _P_COLLAT
        breakdown["collateral_penalty"] = -_P_COLLAT

    if not signals.get("within_budget", True):
        reward -= _P_BUDGET
        breakdown["budget_penalty"] = -_P_BUDGET

    return {
        "reward": round(reward, 6),
        "breakdown": breakdown,
        "error": signals.get("error"),
    }


def compute_invalid_action_reward() -> Dict[str, Any]:
    """Reward for a structurally invalid action (failed schema validation).

    No positive components. Penalty is distinct from the normal step penalty
    so we can log invalid rate separately.
    """
    breakdown = {
        "schema_compliance": 0.0,
        "correct_entity": 0.0,
        "correct_action": 0.0,
        "state_update": 0.0,
        "repeat_penalty": 0.0,
        "step_penalty": -_P_STEP,
        "collateral_penalty": 0.0,
        "budget_penalty": 0.0,
        "invalid_action_penalty": -0.30,
    }
    return {
        "reward": round(-_P_STEP - 0.30, 6),
        "breakdown": breakdown,
        "error": "invalid_action",
    }


__all__ = [
    "compute_invalid_action_reward",
    "compute_step_reward",
]
