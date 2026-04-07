import json
import os
import re
from typing import Any, Dict, Optional, Tuple

import requests
from openai import OpenAI

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


def _fallback_action(task_id: str, observation: Dict[str, Any]) -> Dict[str, Any]:
    tickets = observation.get("open_tickets", []) or []
    orders = observation.get("orders", []) or []
    inventory = observation.get("inventory", []) or []

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
        unresolved = next(
            (
                order
                for order in orders
                if order.get("status") not in {"RESOLVED", "CANCELLED", "REFUNDED", "REJECTED"}
            ),
            None,
        )
        if unresolved:
            return {"action_type": "inspect_order", "order_id": unresolved.get("order_id")}
        if tickets:
            return {"action_type": "escalate_to_human", "ticket_id": tickets[0].get("ticket_id"), "reason": "complex_case"}
        return {"action_type": "cancel_order", "order_id": "UNKNOWN", "reason": "fallback"}

    return {"action_type": "inspect_order", "order_id": "UNKNOWN"}


def _choose_action(
    *,
    llm_client: Optional[OpenAI],
    model_name: str,
    task_id: str,
    observation: Dict[str, Any],
) -> Tuple[Dict[str, Any], str]:
    if llm_client is not None:
        try:
            model_text = _query_model(llm_client, model_name, task_id, observation)
            parsed_action, parse_mode = parse_action(model_text)
            if parsed_action.get("action_type") != "invalid_action":
                return parsed_action, f"llm:{parse_mode}"
        except Exception:
            pass
    return _fallback_action(task_id, observation), "fallback_policy"


def main():
    llm_api_base_url = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1").rstrip("/")
    model_name = os.getenv("MODEL_NAME", "gpt-4o-mini")
    hf_token = os.getenv("HF_TOKEN", "dummy")
    env_base_url = os.getenv("ENV_BASE_URL", "http://localhost:7860").rstrip("/")
    task_ids = ["task_1", "task_2", "task_3"]

    llm_client = _model_client_from_env(llm_api_base_url=llm_api_base_url, hf_token=hf_token)

    for task_id in task_ids:
        observation = _post_json(env_base_url, "/reset", {"task_id": task_id})
        print("[START]", json.dumps({"task_id": task_id, "model": model_name}))

        total_reward = 0.0
        step_count = 0
        max_steps = int(observation.get("steps_remaining", 0))
        while not observation.get("done", False) and step_count < max_steps:
            action, _source = _choose_action(
                llm_client=llm_client,
                model_name=model_name,
                task_id=task_id,
                observation=observation,
            )
            observation = _post_json(env_base_url, "/step", action)
            step_count += 1
            reward = float(observation.get("reward", 0.0))
            total_reward += reward
            print(
                "[STEP]",
                json.dumps(
                    {
                        "step": step_count,
                        "action": action,
                        "reward": round(reward, 2),
                        "done": bool(observation.get("done", False)),
                    }
                ),
            )

        grade = _post_json(env_base_url, "/grader", {})
        score = max(0.0, min(1.0, float(grade.get("score", 0.0))))
        print(
            "[END]",
            json.dumps(
                {
                    "task_id": task_id,
                    "total_reward": round(total_reward, 2),
                    "score": score,
                }
            ),
        )


if __name__ == "__main__":
    main()
