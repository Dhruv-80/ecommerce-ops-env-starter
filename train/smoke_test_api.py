# /// script
# dependencies = ["httpx"]
# ///
"""Smoke-test the live CommerceOps-Env Space API.

Run as an HF Job (cpu-basic) or locally:
  python train/smoke_test_api.py

Env vars:
  ENV_BASE_URL  default: http://localhost:7860
"""

from __future__ import annotations

import os
import sys

import httpx

BASE = os.environ.get("ENV_BASE_URL", "http://localhost:7860").rstrip("/")
TIMEOUT = 30


def check(label: str, resp: httpx.Response, expect_status: int = 200) -> dict:
    ok = resp.status_code == expect_status
    mark = "✅" if ok else "❌"
    print(f"  {mark}  {label:<45} {resp.status_code}")
    if not ok:
        print(f"       Response: {resp.text[:200]}")
    return resp.json() if ok else {}


def main() -> None:
    print(f"\n{'='*60}")
    print(f"  CommerceOps-Env API Smoke Test")
    print(f"  base: {BASE}")
    print(f"{'='*60}\n")

    failures = 0
    c = httpx.Client(base_url=BASE, timeout=TIMEOUT)

    # Health
    r = c.get("/health")
    body = check("GET /health", r)
    if body.get("ok") is not True:
        print("  ❌ health.ok != True"); failures += 1

    # Tasks
    r = c.get("/tasks")
    body = check("GET /tasks", r)
    if set(body.keys()) != {"task_1", "task_2", "task_3"}:
        print(f"  ❌ tasks keys mismatch: {set(body.keys())}"); failures += 1

    # Schema
    r = c.get("/schema")
    body = check("GET /schema", r)
    if "action" not in body:
        print("  ❌ schema missing 'action'"); failures += 1

    # Metadata
    r = c.get("/metadata")
    body = check("GET /metadata", r)
    if body.get("version") != "2.0.0":
        print(f"  ⚠️  metadata.version={body.get('version')} (expected 2.0.0)")

    # Reset T1
    r = c.post("/reset", json={"task_id": "task_1", "seed": 0})
    body = check("POST /reset task_1 seed=0", r)
    if body.get("task_id") != "task_1":
        print("  ❌ reset.task_id != task_1"); failures += 1
    if body.get("done") is not False:
        print("  ❌ reset.done should be False"); failures += 1
    if not body.get("orders"):
        print("  ❌ reset.orders is empty"); failures += 1

    # Step valid
    r = c.post("/step", json={"action_type": "assign_warehouse",
                               "order_id": "O1", "warehouse_id": "W1"})
    body = check("POST /step valid action", r)
    if "reward" not in body:
        print("  ❌ step missing 'reward'"); failures += 1

    # State
    r = c.get("/state")
    body = check("GET /state", r)
    if "orders" not in body or "stock" not in body:
        print("  ❌ state missing 'orders' or 'stock'"); failures += 1

    # Grader
    r = c.post("/grader")
    body = check("POST /grader", r)
    if "score" not in body:
        print("  ❌ grader missing 'score'"); failures += 1
    elif not (0.0 < body["score"] <= 1.0):
        print(f"  ❌ grader score out of range: {body['score']}"); failures += 1

    # Reset T2
    r = c.post("/reset", json={"task_id": "task_2", "seed": 0})
    body = check("POST /reset task_2 seed=0", r)
    if body.get("task_id") != "task_2":
        print("  ❌ reset.task_id != task_2"); failures += 1

    # Invalid action (disallowed for T2)
    r = c.post("/step", json={"action_type": "reroute_order",
                               "order_id": "O1", "warehouse_id": "W1"})
    body = check("POST /step invalid-for-task action", r)
    if body.get("reward", 0) >= 0:
        print(f"  ❌ invalid action should give negative reward, got {body.get('reward')}"); failures += 1

    # Baseline (oracle run on all tasks)
    print("\n  Running /baseline (oracle policy) — may take ~15s …")
    r = c.get("/baseline", timeout=120)
    body = check("GET /baseline", r)
    for task_id in ("task_1", "task_2", "task_3"):
        if task_id not in body:
            print(f"  ❌ baseline missing {task_id}"); failures += 1
            continue
        score = body[task_id].get("score", 0)
        print(f"       {task_id} oracle score={score:.3f}")
        if score < 0.5:
            print(f"  ⚠️  {task_id} oracle score unexpectedly low")

    print(f"\n{'='*60}")
    if failures == 0:
        print("  ✅  All checks passed — Space API is healthy")
    else:
        print(f"  ❌  {failures} check(s) failed")
        sys.exit(1)
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
