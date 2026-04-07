from typing import Any, Dict


def compute_step_reward(*, action: Dict[str, Any], outcome: Dict[str, Any], ground_truth: Dict[str, Any]) -> Dict[str, Any]:
    """IMPLEMENT: keep reward shaping simple and deterministic."""
    reward = 0.0
    breakdown = {
        "action_type": 0.0,
        "target_entity": 0.0,
        "final_state": 0.0,
        "collateral_damage": 0.0,
        "efficiency": 0.0,
        "penalties": 0.0,
    }

    if outcome.get("valid_action"):
        reward += 0.15
        breakdown["action_type"] = 0.15

    if outcome.get("correct_target"):
        reward += 0.20
        breakdown["target_entity"] = 0.20

    if outcome.get("state_matches_ground_truth"):
        reward += 0.25
        breakdown["final_state"] = 0.25

    if not outcome.get("collateral_damage"):
        reward += 0.15
        breakdown["collateral_damage"] = 0.15

    if outcome.get("within_budget", True):
        reward += 0.10
        breakdown["efficiency"] = 0.10

    if outcome.get("wrong_entity"):
        reward -= 0.30
        breakdown["penalties"] -= 0.30
    if outcome.get("wrong_inventory"):
        reward -= 0.20
        breakdown["penalties"] -= 0.20
    if outcome.get("unnecessary_escalation"):
        reward -= 0.15
        breakdown["penalties"] -= 0.15
    if outcome.get("destructive_cancel"):
        reward -= 0.25
        breakdown["penalties"] -= 0.25
    if outcome.get("repeat_action"):
        reward -= 0.10
        breakdown["penalties"] -= 0.10

    return {"reward": round(reward, 10), "breakdown": breakdown, "error": outcome.get("error")}
