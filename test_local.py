#!/usr/bin/env python3
"""Local debugging: run Qwen2.5-3B locally to see what it actually outputs.

Goal: prove what the baseline model produces BEFORE we add training.
This eliminates guesswork. Runs on Mac MPS (or CPU fallback).

Usage:
    python test_local.py
"""

import json
import os
import re
import sys
from typing import Any, Dict

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from environment import CommerceOpsEnv

MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"

# ---------------------------------------------------------------------------
# Two prompts: V1 (current, abstract) vs V2 (with few-shot examples)
# ---------------------------------------------------------------------------

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
You are a fulfillment operations manager. Output ONLY a JSON action — no prose, no markdown, no explanation.

Examples of valid outputs:

Example 1 — Assign an order to a warehouse:
{"action_type": "assign_warehouse", "order_id": "O1", "warehouse_id": "W1"}

Example 2 — Split an order across two warehouses:
{"action_type": "split_shipment", "order_id": "O1", "allocations": [{"warehouse_id": "W1", "quantity": 5}, {"warehouse_id": "W2", "quantity": 3}]}

Example 3 — Delay an order:
{"action_type": "delay_order", "order_id": "O1", "reason": "stock_unavailable"}

Example 4 — Do nothing this step:
{"action_type": "noop"}

Output format: a single JSON object on one line. No code fences. No reasoning text.
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


def test_prompt(model, tokenizer, prompt_name: str, system_prompt: str, n_seeds: int = 3):
    print(f"\n{'='*78}")
    print(f"  TESTING: {prompt_name}")
    print(f"{'='*78}")
    
    valid_count = 0
    total_count = 0
    samples = []
    
    for task_id in ["task_1", "task_2"]:
        for seed in range(n_seeds):
            env = CommerceOpsEnv()
            obs = env.reset(task_id=task_id, seed=seed)
            
            user_msg = obs_to_text(obs)
            raw = generate(model, tokenizer, system_prompt, user_msg)
            action = extract_action(raw)
            
            new_obs = env.step(action)
            is_valid = new_obs.last_action_error is None and action.get("action_type") != "noop"
            
            total_count += 1
            if is_valid:
                valid_count += 1
            
            samples.append({
                "task": task_id,
                "seed": seed,
                "raw_first_150": raw[:150],
                "parsed": action,
                "valid": is_valid,
                "error": new_obs.last_action_error,
                "reward": new_obs.reward,
            })
    
    print(f"\nValid actions: {valid_count}/{total_count}")
    print(f"\nSample outputs:")
    for s in samples[:3]:
        print(f"\n  [{s['task']} seed={s['seed']}]  valid={s['valid']}  reward={s['reward']:+.3f}")
        print(f"    raw    : {s['raw_first_150']!r}")
        print(f"    parsed : {s['parsed']}")
        print(f"    error  : {s['error']}")
    
    return valid_count, total_count


def main():
    print("Loading Qwen2.5-3B-Instruct (this takes ~30s on first run, downloads ~6GB)...")
    
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16 if device == "mps" else torch.float32,
        device_map=device,
    )
    model.eval()
    
    v1_valid, v1_total = test_prompt(model, tokenizer, "V1 (current — abstract)", PROMPT_V1_ABSTRACT)
    v2_valid, v2_total = test_prompt(model, tokenizer, "V2 (few-shot)", PROMPT_V2_FEW_SHOT)
    
    print(f"\n{'='*78}")
    print(f"  SUMMARY")
    print(f"{'='*78}")
    print(f"  V1 (abstract prompt) : {v1_valid}/{v1_total} valid actions ({100*v1_valid/v1_total:.0f}%)")
    print(f"  V2 (few-shot prompt) : {v2_valid}/{v2_total} valid actions ({100*v2_valid/v2_total:.0f}%)")
    print(f"\n  → If V2 >> V1, format priming via prompt is sufficient (no SFT needed).")
    print(f"  → If both ~0%, we need light SFT before GRPO.")


if __name__ == "__main__":
    main()
