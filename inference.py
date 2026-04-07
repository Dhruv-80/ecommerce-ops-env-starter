import json
import os
import re
from typing import Any, Dict, List, Optional, Set, Tuple

import requests
from openai import OpenAI

BENCHMARK = "ecommerce-ops-env"

SYSTEM_PROMPT = """You are an e-commerce operations agent.
At each turn, output exactly one JSON object describing the next action.
Valid actions: process_refund, reject_refund, update_inventory, route_order,
apply_substitute, cancel_order, flag_dispute, escalate_to_human,
send_compensation, inspect_order.
Do not output markdown. Do not explain. Output JSON only.
If you need missing order detail, use inspect_order first.
Wrong entity targeting is heavily penalized, so do not guess.
"""

ALLOWED_ACTION_FIELDS = {
    "action_type",
    "order_id",
    "ticket_id",
    "sku",
    "warehouse",
    "quantity",
    "reason",
    "compensation_type",
}


def _invalid_action(reason: str) -> Dict[str, Any]:
    return {"action_type": "invalid_action", "reason": reason}


def _safe_json_load(text: str) -> Optional[Any]:
    try:
        return json.loads(text)
    except Exception:
        return None


def _extract_first_json_object(text: str) -> Optional[str]:
    start = -1
    depth = 0
    in_string = False
    escape = False

    for idx, char in enumerate(text):
        if start < 0:
            if char == "{":
                start = idx
                depth = 1
            continue

        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]
    return None


def _sanitize_action(raw: Any) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        return _invalid_action("parsed_value_not_object")
    action_type = raw.get("action_type")
    if not isinstance(action_type, str) or not action_type.strip():
        return _invalid_action("missing_action_type")
    cleaned: Dict[str, Any] = {"action_type": action_type.strip()}
    for key in ALLOWED_ACTION_FIELDS:
        if key == "action_type":
            continue
        if key in raw:
            cleaned[key] = raw[key]
    return cleaned


def parse_action(text: str) -> Tuple[Dict[str, Any], str]:
    normalized = text.strip()
    if not normalized:
        return _invalid_action("empty_model_response"), "fallback_invalid"

    # 1) Direct JSON
    parsed = _safe_json_load(normalized)
    if parsed is not None:
        action = _sanitize_action(parsed)
        mode = "direct_json" if action.get("action_type") != "invalid_action" else "fallback_invalid"
        return action, mode

    # 2) XML-wrapped JSON
    match = re.search(r"<action>\s*(\{.*?\})\s*</action>", normalized, re.DOTALL | re.IGNORECASE)
    if match:
        parsed = _safe_json_load(match.group(1).strip())
        if parsed is not None:
            action = _sanitize_action(parsed)
            mode = "xml_wrapped" if action.get("action_type") != "invalid_action" else "fallback_invalid"
            return action, mode

    # 3) First JSON object in text
    candidate = _extract_first_json_object(normalized)
    if candidate:
        parsed = _safe_json_load(candidate)
        if parsed is not None:
            action = _sanitize_action(parsed)
            mode = "first_json_object" if action.get("action_type") != "invalid_action" else "fallback_invalid"
            return action, mode

    # 4) Invalid action fallback
    return _invalid_action("unable_to_parse_json_action"), "fallback_invalid"


def _post_json(base_url: str, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    response = requests.post(f"{base_url}{path}", json=payload, timeout=60)
    response.raise_for_status()
    return response.json()


def _model_client_from_env(*, llm_api_base_url: str, hf_token: str) -> Optional[OpenAI]:
    token = hf_token.strip()
    if not token or token.lower() in {"dummy", "none"}:
        return None
    return OpenAI(base_url=llm_api_base_url, api_key=token)


def _query_model(
    client: OpenAI,
    model_name: str,
    task_id: str,
    observation: Dict[str, Any],
) -> str:
    payload = {"task_id": task_id, "observation": observation}
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=True)},
        ],
        temperature=0,
    )
    return (response.choices[0].message.content or "").strip()


def _fallback_action(task_id: str, observation: Dict[str, Any], memory: Dict[str, Any]) -> Dict[str, Any]:
    tickets = observation.get("open_tickets", []) or []
    orders = observation.get("orders", []) or []
    inventory = observation.get("inventory", []) or []
    metadata = observation.get("metadata", {}) or {}

    inspected_orders: Dict[str, Dict[str, Any]] = memory.setdefault("inspected_orders", {})
    sent_comp: Set[Tuple[str, str]] = memory.setdefault("sent_comp", set())
    resolved_orders: Set[str] = memory.setdefault("resolved_orders", set())

    inspected = metadata.get("inspected_order")
    if isinstance(inspected, dict) and inspected.get("order_id"):
        inspected_orders[str(inspected["order_id"])] = inspected

    if task_id == "task_1":
        if tickets:
            ticket = sorted(tickets, key=lambda t: t.get("ticket_id", ""))[0]
            days = int(ticket.get("created_days_ago", 0))
            action_type = "process_refund" if days <= 30 else "reject_refund"
            return {
                "action_type": action_type,
                "ticket_id": ticket.get("ticket_id"),
                "order_id": ticket.get("order_id"),
                "reason": f"policy_{days}_days",
            }
        if orders:
            return {"action_type": "inspect_order", "order_id": orders[0].get("order_id")}
        return {"action_type": "escalate_to_human", "ticket_id": "UNKNOWN", "reason": "no_open_tickets"}

    if task_id == "task_2":
        pending = next((order for order in orders if not order.get("status") == "ROUTED"), None)
        if pending:
            wh = "W1"
            if inventory:
                wh = sorted(inventory, key=lambda row: (-int(row.get("quantity", 0)), row.get("warehouse", "")))[0].get("warehouse", "W1")
            return {"action_type": "route_order", "order_id": pending.get("order_id"), "warehouse": wh}
        if orders:
            return {"action_type": "inspect_order", "order_id": orders[0].get("order_id")}
        return {"action_type": "update_inventory", "sku": "SKU-DEFAULT", "warehouse": "W1", "quantity": 0}

    if task_id == "task_3":
        tier_to_comp = {"standard": "coupon_5", "premium": "coupon_15", "loyalty": "priority_support"}

        # Prefer acting on already-inspected orders.
        candidate_ids = [str(o.get("order_id")) for o in orders if o.get("order_id")]
        for order_id in candidate_ids:
            full = inspected_orders.get(order_id)
            if not isinstance(full, dict):
                continue

            items = full.get("items", []) or []
            cancelled_skus = [str(it.get("sku")) for it in items if str(it.get("status", "")).upper() in {"CANCELLED", "AFFECTED"}]
            shipped_present = any(str(it.get("status", "")).upper() == "SHIPPED" for it in items)
            tier = str(full.get("customer_tier", "standard")).strip().lower() or "standard"
            desired_comp = tier_to_comp.get(tier, "coupon_5")

            # If there are cancellations, resolve first.
            if cancelled_skus and order_id not in resolved_orders:
                # Heuristic: fully-cancelled premium orders are cancelled; otherwise substitute.
                if tier == "premium" and not shipped_present and len(cancelled_skus) == len(items):
                    return {"action_type": "cancel_order", "order_id": order_id, "reason": "supplier_cancelled"}
                sub_sku = f"{cancelled_skus[0]}-SUB"
                return {"action_type": "apply_substitute", "order_id": order_id, "sku": sub_sku}

            # If resolved but missing compensation, send it.
            if (order_id, desired_comp) not in sent_comp:
                # If we can see existing compensation, avoid duplicates.
                existing = set(str(x) for x in (full.get("compensation", []) or []))
                if desired_comp not in existing:
                    sent_comp.add((order_id, desired_comp))
                    return {"action_type": "send_compensation", "order_id": order_id, "compensation_type": desired_comp}
                sent_comp.add((order_id, desired_comp))

        # Otherwise inspect the next likely-affected order (ignore untouched/unaffected orders later).
        unresolved = next(
            (o for o in orders if str(o.get("status", "")).upper() not in {"RESOLVED", "CANCELLED", "REFUNDED", "REJECTED"}),
            None,
        )
        if unresolved and unresolved.get("order_id"):
            return {"action_type": "inspect_order", "order_id": unresolved.get("order_id")}
        return {"action_type": "inspect_order", "order_id": "UNKNOWN"}

    return {"action_type": "inspect_order", "order_id": "UNKNOWN"}


def _choose_action(
    *,
    llm_client: Optional[OpenAI],
    model_name: str,
    task_id: str,
    observation: Dict[str, Any],
    memory: Dict[str, Any],
) -> Tuple[Dict[str, Any], str]:
    if llm_client is not None:
        try:
            model_text = _query_model(llm_client, model_name, task_id, observation)
            parsed_action, parse_mode = parse_action(model_text)
            if parsed_action.get("action_type") != "invalid_action":
                return parsed_action, f"llm:{parse_mode}"
        except Exception:
            pass
    return _fallback_action(task_id, observation, memory), "fallback_policy"


def _format_action(action: Dict[str, Any]) -> str:
    atype = action.get("action_type", "unknown")
    parts = [f"{k}={v}" for k, v in action.items() if k != "action_type" and v is not None]
    if parts:
        return f"{atype}({','.join(parts)})"
    return atype


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


def main():
    llm_api_base_url = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1").rstrip("/")
    model_name = os.getenv("MODEL_NAME", "gpt-4o-mini")
    hf_token = os.getenv("HF_TOKEN") or ""
    env_base_url = os.getenv("ENV_BASE_URL", "http://localhost:7860").rstrip("/")
    task_ids = ["task_1", "task_2", "task_3"]

    llm_client = _model_client_from_env(llm_api_base_url=llm_api_base_url, hf_token=hf_token)

    for task_id in task_ids:
        memory: Dict[str, Any] = {}
        rewards: List[float] = []
        step_count = 0
        score = 0.0
        success = False

        log_start(task=task_id, env=BENCHMARK, model=model_name)

        try:
            observation = _post_json(env_base_url, "/reset", {"task_id": task_id})
            max_steps = int(observation.get("steps_remaining", 0))

            while not observation.get("done", False) and step_count < max_steps:
                action, _source = _choose_action(
                    llm_client=llm_client,
                    model_name=model_name,
                    task_id=task_id,
                    observation=observation,
                    memory=memory,
                )
                if task_id == "task_3":
                    resolved_orders: Set[str] = memory.setdefault("resolved_orders", set())
                    action_type = str(action.get("action_type", ""))
                    order_id = action.get("order_id")
                    if order_id and action_type in {"apply_substitute", "cancel_order"}:
                        resolved_orders.add(str(order_id))

                observation = _post_json(env_base_url, "/step", action)
                step_count += 1
                reward = float(observation.get("reward", 0.0))
                done = bool(observation.get("done", False))
                error = observation.get("last_action_error")
                rewards.append(reward)

                log_step(
                    step=step_count,
                    action=_format_action(action),
                    reward=reward,
                    done=done,
                    error=error,
                )

            grade = _post_json(env_base_url, "/grader", {})
            score = max(0.0, min(1.0, float(grade.get("score", 0.0))))
            success = score > 0.0

        finally:
            log_end(success=success, steps=step_count, score=score, rewards=rewards)


if __name__ == "__main__":
    main()
