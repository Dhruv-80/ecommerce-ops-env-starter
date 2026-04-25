# /// script
# dependencies = [
#   "transformers>=4.45",
#   "torch",
#   "accelerate",
#   "huggingface-hub",
# ]
# ///
"""Diagnostic-only HF Job: load Qwen2.5-3B, see what it outputs.

NO training. NO Unsloth. Just inference + show the raw outputs so we know
exactly what the baseline model produces. Costs ~3 minutes on l4x1.

Compares two system prompts:
  V1: current abstract prompt (just lists action types)
  V2: few-shot prompt (shows actual JSON examples)

Outputs the % of valid actions for each, plus 6 sample raw generations.
"""

import json
import os
import re
import sys
from typing import Any, Dict

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Clone env repo
REPO_URL = os.environ.get(
    "ENV_REPO_URL",
    "https://github.com/Dhruv-80/ecommerce-ops-env-starter.git",
)
REPO_DIR = "/tmp/commerce-ops-env"
if not os.path.exists(REPO_DIR):
    os.system(f"git clone --depth 1 {REPO_URL} {REPO_DIR}")
else:
    os.system(f"git -C {REPO_DIR} fetch --depth 1 origin main && git -C {REPO_DIR} reset --hard origin/main")
sys.path.insert(0, REPO_DIR)

from environment import CommerceOpsEnv  # noqa: E402

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

PROMPT_V1_ABSTRACT = """\
You are a fulfillment operations manager for an e-commerce platform.
Respond with a single JSON action object and nothing else.

Action schema:
  {"action_type": "<type>", ...fields}

Allowed types: assign_warehouse, split_shipment, delay_order,
               prioritize_order, reroute_order, escalate_supplier,
               refund_or_compensate, noop
"""

PROMPT_V2_FEW_SHOT = """\
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


def extract_action(text: str) -> Dict[str, Any]:
    text = text.strip()
    text = re.sub(r"```(?:json)?", "", text).strip()
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
                    try:
                        return json.loads(text[start:i+1])
                    except json.JSONDecodeError:
                        break
    return {"action_type": "noop"}


def obs_to_text(obs) -> str:
    return json.dumps({
        "task": obs.task_id,
        "step": obs.step,
        "steps_remaining": obs.steps_remaining,
        "description": obs.task_description,
        "allowed_actions": [a.value if hasattr(a, "value") else a for a in obs.allowed_actions],
        "orders":     [o.model_dump() for o in obs.orders],
        "warehouses": [w.model_dump() for w in obs.warehouses],
        "stock":      [s.model_dump() for s in obs.stock],
    }, indent=2)


def generate(model, tokenizer, system_prompt: str, user_msg: str) -> str:
    msgs = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_msg},
    ]
    inputs = tokenizer.apply_chat_template(
        msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt"
    ).to(model.device)
    with torch.no_grad():
        out = model.generate(
            inputs, max_new_tokens=200, do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    return tokenizer.decode(out[0][inputs.shape[-1]:], skip_special_tokens=True)


def test_prompt(model, tokenizer, name: str, system_prompt: str):
    print(f"\n{'='*78}\n  {name}\n{'='*78}")
    valid = 0
    total = 0
    samples = []
    for task_id in ["task_1", "task_2"]:
        for seed in range(3):
            env = CommerceOpsEnv()
            obs = env.reset(task_id=task_id, seed=seed)
            raw = generate(model, tokenizer, system_prompt, obs_to_text(obs))
            action = extract_action(raw)
            new_obs = env.step(action)
            is_valid = new_obs.last_action_error is None and action.get("action_type") != "noop"
            total += 1
            if is_valid:
                valid += 1
            samples.append({
                "task": task_id, "seed": seed,
                "raw": raw, "parsed": action,
                "valid": is_valid, "error": new_obs.last_action_error,
                "reward": new_obs.reward,
            })

    print(f"\nValid actions (non-noop, no env error): {valid}/{total}  ({100*valid/total:.0f}%)\n")
    for s in samples:
        print(f"--- {s['task']} seed={s['seed']}  valid={s['valid']}  reward={s['reward']:+.3f} ---")
        print(f"RAW:    {s['raw']!r}")
        print(f"PARSED: {s['parsed']}")
        print(f"ERROR:  {s['error']}\n")
    return valid, total


def main():
    print(f"Loading {MODEL_ID} on {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()

    v1_valid, v1_total = test_prompt(model, tokenizer, "V1: ABSTRACT (current)", PROMPT_V1_ABSTRACT)
    v2_valid, v2_total = test_prompt(model, tokenizer, "V2: FEW-SHOT (proposed)", PROMPT_V2_FEW_SHOT)

    print(f"\n{'='*78}\n  VERDICT\n{'='*78}")
    print(f"  V1 abstract  : {v1_valid}/{v1_total}  ({100*v1_valid/v1_total:.0f}%)")
    print(f"  V2 few-shot  : {v2_valid}/{v2_total}  ({100*v2_valid/v2_total:.0f}%)")
    if v2_valid >= 2 and v2_valid > v1_valid:
        print(f"\n  ✓ Few-shot prompt produces valid actions. Format priming via prompt is enough.")
        print(f"  ✓ GRPO will have signal to learn from. Proceed to training.")
    elif v2_valid == 0 and v1_valid == 0:
        print(f"\n  ✗ Both prompts produce 0 valid actions. Need light SFT before GRPO.")
    else:
        print(f"\n  ⚠ Marginal — investigate sample outputs above to decide.")


if __name__ == "__main__":
    main()
