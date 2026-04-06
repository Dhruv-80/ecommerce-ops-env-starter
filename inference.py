import json
import os
import re
from typing import Optional

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


def parse_action(text: str) -> Optional[dict]:
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    match = re.search(r"<action>(.*?)</action>", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1).strip())
        except Exception:
            pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except Exception:
            return None
    return None


def main():
    api_base_url = os.getenv("API_BASE_URL", "http://localhost:7860")
    model_name = os.getenv("MODEL_NAME", "gpt-4o-mini")
    hf_token = os.getenv("HF_TOKEN", "dummy")
    task_ids = ["task_1", "task_2", "task_3"]
    max_steps = {"task_1": 10, "task_2": 15, "task_3": 18}

    client = OpenAI(base_url=api_base_url, api_key=hf_token)

    for task_id in task_ids:
        print("[START]", json.dumps({"task_id": task_id, "model": model_name}))
        # IMPLEMENT: call your env reset and LLM loop here
        total_reward = 0.0
        for step in range(max_steps[task_id]):
            action = {"action_type": "inspect_order", "order_id": "TODO"}
            print("[STEP]", json.dumps({"step": step + 1, "action": action, "reward": round(0.0, 2), "done": False}))
            break
        print("[END]", json.dumps({"task_id": task_id, "total_reward": round(total_reward, 2), "score": 0.0}))


if __name__ == "__main__":
    main()
