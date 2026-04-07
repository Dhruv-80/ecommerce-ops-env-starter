"""Pre-submission validation script.

Runs every check from the submission checklist locally:
  1. Required files present
  2. openenv.yaml well-formed
  3. .env.example declares API_BASE_URL, MODEL_NAME, HF_TOKEN
  4. FastAPI app imports and endpoints return correct shapes (ASGI)
  5. inference.py produces [START]/[STEP]/[END] logs for 3 tasks with scores in [0,1]
  6. Docker build (skipped when daemon is unavailable)
  7. pytest suite passes

Usage:
    .venv/bin/python pre_submission_validate.py
"""

import asyncio
import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent
PASSED: List[str] = []
WARNINGS: List[str] = []


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _fail(msg: str) -> None:
    raise SystemExit(f"\n[FAIL] {msg}")


def _ok(msg: str) -> None:
    PASSED.append(msg)
    print(f"  [OK] {msg}")


def _warn(msg: str) -> None:
    WARNINGS.append(msg)
    print(f"  [WARN] {msg}")


def _python_exec() -> str:
    venv_py = REPO_ROOT / ".venv" / "bin" / "python"
    if venv_py.exists():
        return str(venv_py)
    return sys.executable


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ── 1. Required files ────────────────────────────────────────────────

def check_required_files() -> None:
    print("\n── Required files ──")
    required = [
        "openenv.yaml", "Dockerfile", "inference.py",
        "server/app.py", "models.py", "requirements.txt",
    ]
    missing = [n for n in required if not (REPO_ROOT / n).exists()]
    if missing:
        _fail(f"Missing required files: {', '.join(missing)}")
    _ok("All required files present")


# ── 2. openenv.yaml ──────────────────────────────────────────────────

def check_openenv_yaml() -> None:
    print("\n── openenv.yaml ──")
    text = _read_text(REPO_ROOT / "openenv.yaml")
    for key in ("spec_version:", "name:", "runtime:", "app:", "port:"):
        if key not in text:
            _fail(f"openenv.yaml missing `{key.rstrip(':')}`")
    compact = re.sub(r"\s+", "", text)
    if "app:server.app:app" not in compact:
        _fail("openenv.yaml `app` must be `server.app:app`")
    _ok("openenv.yaml valid")


# ── 3. Env vars template ─────────────────────────────────────────────

def check_env_vars() -> None:
    print("\n── Environment variables ──")
    path = REPO_ROOT / ".env.example"
    if not path.exists():
        _warn(".env.example not found; skipping")
        return
    text = _read_text(path)
    for key in ("API_BASE_URL", "MODEL_NAME", "HF_TOKEN"):
        if not re.search(rf"^{re.escape(key)}=", text, re.MULTILINE):
            _fail(f".env.example must define `{key}`")
    _ok("API_BASE_URL, MODEL_NAME, HF_TOKEN declared")


# ── 4. Endpoint contract via ASGI ────────────────────────────────────

def check_endpoints() -> None:
    print("\n── Endpoint contract (ASGI) ──")
    sys.path.insert(0, str(REPO_ROOT))
    try:
        from server.app import app  # type: ignore
    except Exception as exc:
        _fail(f"Cannot import server.app:app — {exc}")

    import httpx

    async def _run() -> None:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            r = await c.get("/health")
            assert r.status_code == 200, f"GET /health → {r.status_code}"

            r = await c.post("/reset", json={"task_id": "task_1"})
            assert r.status_code == 200, f"POST /reset → {r.status_code}"
            obs = r.json()
            for k in ("done", "reward", "task_id", "steps_remaining"):
                assert k in obs, f"/reset response missing `{k}`"

            r = await c.post("/step", json={"action_type": "inspect_order", "order_id": "O1"})
            assert r.status_code == 200, f"POST /step → {r.status_code}"

            r = await c.get("/state")
            assert r.status_code == 200, f"GET /state → {r.status_code}"

            r = await c.get("/tasks")
            assert r.status_code == 200, f"GET /tasks → {r.status_code}"
            tasks = r.json()
            assert isinstance(tasks, dict), "/tasks must return dict"
            assert {"task_1", "task_2", "task_3"}.issubset(tasks.keys()), "/tasks missing task IDs"

            r = await c.post("/grader", json={})
            assert r.status_code == 200, f"POST /grader → {r.status_code}"
            score = float(r.json().get("score", -1))
            assert 0.0 <= score <= 1.0, f"/grader score {score} out of [0,1]"

    try:
        asyncio.run(_run())
    except AssertionError as exc:
        _fail(str(exc))

    _ok("GET /health — 200")
    _ok("POST /reset — 200, observation shape correct")
    _ok("POST /step — 200")
    _ok("GET /state — 200")
    _ok("GET /tasks — 200, contains task_1/task_2/task_3")
    _ok("POST /grader — 200, score in [0,1]")


# ── 5. Grader per-task scores ─────────────────────────────────────────

def check_graders_per_task() -> None:
    print("\n── Per-task grader scores ──")
    import httpx
    from server.app import app  # type: ignore

    async def _run_task(task_id: str) -> float:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            await c.post("/reset", json={"task_id": task_id})
            r = await c.post("/grader", json={})
            return float(r.json().get("score", -1))

    for tid in ("task_1", "task_2", "task_3"):
        score = asyncio.run(_run_task(tid))
        if not (0.0 <= score <= 1.0):
            _fail(f"Grader score for {tid} = {score}, expected [0,1]")
        _ok(f"{tid} grader score = {score:.4f}")


# ── 6. inference.py log format ────────────────────────────────────────

def _parse_log_line(prefix: str, line: str) -> Dict[str, Any]:
    if not line.startswith(prefix + " "):
        _fail(f"Log line must start with `{prefix} `: {line[:120]}")
    payload = line[len(prefix) + 1:].strip()
    try:
        return json.loads(payload)
    except Exception:
        _fail(f"Payload after `{prefix}` must be valid JSON: {payload[:120]}")
    return {}


def check_inference_logs() -> None:
    print("\n── inference.py baseline run ──")

    port = _free_port()

    # Start uvicorn in a background thread using the same process to avoid
    # sandbox issues with subprocess port-binding.
    import uvicorn  # type: ignore

    server_ready = threading.Event()

    class _ReadyServer(uvicorn.Server):
        def startup(self, sockets=None):  # type: ignore
            result = super().startup(sockets)
            server_ready.set()
            return result

    config = uvicorn.Config(
        "server.app:app", host="127.0.0.1", port=port,
        log_level="error",
    )
    server = uvicorn.Server(config)

    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()

    # Poll until the port is actually listening.
    deadline = time.time() + 20
    while time.time() < deadline:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=1):
                break
        except OSError:
            time.sleep(0.2)
    else:
        _fail(f"Uvicorn server did not start on port {port} within 20s")

    _ok(f"Environment server started on port {port}")

    env = os.environ.copy()
    env["ENV_BASE_URL"] = f"http://127.0.0.1:{port}"
    env.setdefault("API_BASE_URL", "https://router.huggingface.co/v1")
    env.setdefault("MODEL_NAME", "gpt-4o-mini")
    env.setdefault("HF_TOKEN", "dummy")

    t0 = time.time()
    try:
        proc = subprocess.run(
            [_python_exec(), str(REPO_ROOT / "inference.py")],
            cwd=str(REPO_ROOT),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=1200,  # 20 min hard cap
        )
    except subprocess.TimeoutExpired:
        _fail("inference.py exceeded 20-minute time limit")
    finally:
        server.should_exit = True

    elapsed = time.time() - t0
    if proc.returncode != 0:
        _fail(f"inference.py exited with code {proc.returncode}.\n{proc.stdout[-2000:]}")
    _ok(f"inference.py completed in {elapsed:.1f}s (exit 0)")

    lines = [ln.strip() for ln in proc.stdout.splitlines() if ln.strip()]
    starts = [ln for ln in lines if ln.startswith("[START] ")]
    steps = [ln for ln in lines if ln.startswith("[STEP] ")]
    ends = [ln for ln in lines if ln.startswith("[END] ")]

    if len(starts) != 3:
        _fail(f"Expected 3 [START] lines, got {len(starts)}")
    if len(ends) != 3:
        _fail(f"Expected 3 [END] lines, got {len(ends)}")
    if len(steps) < 3:
        _fail(f"Expected >= 3 [STEP] lines, got {len(steps)}")

    _ok(f"Log structure: {len(starts)} [START], {len(steps)} [STEP], {len(ends)} [END]")

    ended = [_parse_log_line("[END]", ln) for ln in ends]
    task_ids_found = {obj.get("task_id") for obj in ended}
    if task_ids_found != {"task_1", "task_2", "task_3"}:
        _fail(f"[END] task_ids must be task_1/task_2/task_3; got {task_ids_found}")

    for obj in ended:
        tid = obj.get("task_id", "?")
        score = float(obj.get("score", -1))
        total_reward = float(obj.get("total_reward", 0))
        if not (0.0 <= score <= 1.0):
            _fail(f"{tid}: score {score} out of [0,1]")
        _ok(f"{tid}: score={score:.4f}  total_reward={total_reward:.2f}")

    # Verify [START] payloads contain task_id and model.
    for ln in starts:
        obj = _parse_log_line("[START]", ln)
        for key in ("task_id", "model"):
            if key not in obj:
                _fail(f"[START] payload missing `{key}`")

    # Verify [STEP] payloads contain step, action, reward, done.
    for ln in steps[:6]:  # sample check
        obj = _parse_log_line("[STEP]", ln)
        for key in ("step", "action", "reward", "done"):
            if key not in obj:
                _fail(f"[STEP] payload missing `{key}`")

    _ok("All log payloads have correct fields")


# ── 7. pytest ─────────────────────────────────────────────────────────

def check_tests() -> None:
    print("\n── pytest ──")
    proc = subprocess.run(
        [_python_exec(), "-m", "pytest", "-q", "--tb=short"],
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        timeout=120,
    )
    if proc.returncode != 0:
        _fail(f"pytest failed (exit {proc.returncode}):\n{proc.stdout[-2000:]}")
    # Extract summary line like "18 passed in 0.01s"
    for line in reversed(proc.stdout.splitlines()):
        if "passed" in line:
            _ok(line.strip())
            break
    else:
        _ok("pytest passed")


# ── 8. Docker build ──────────────────────────────────────────────────

def check_docker() -> None:
    print("\n── Docker build ──")
    if not (REPO_ROOT / "Dockerfile").exists():
        _fail("Dockerfile missing")
    try:
        info = subprocess.run(
            ["docker", "info"], stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, timeout=10,
        )
    except Exception:
        _warn("Docker CLI not found; skipping build check")
        return
    if info.returncode != 0:
        _warn("Docker daemon not reachable; skipping build check")
        return

    build = subprocess.run(
        ["docker", "build", "-t", "ecommerce-ops-env:validate", "."],
        cwd=str(REPO_ROOT),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, timeout=900,
    )
    if build.returncode != 0:
        _fail(f"docker build failed:\n{build.stdout[-2000:]}")
    _ok("Docker image built successfully")


# ── main ──────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("  Pre-Submission Validator — ecommerce-ops-env")
    print("=" * 60)

    check_required_files()
    check_openenv_yaml()
    check_env_vars()
    check_endpoints()
    check_graders_per_task()
    check_tests()
    check_inference_logs()
    check_docker()

    print("\n" + "=" * 60)
    if WARNINGS:
        print(f"  Warnings: {len(WARNINGS)}")
        for w in WARNINGS:
            print(f"    - {w}")
    print(f"  Checks passed: {len(PASSED)}")
    print("\n  [PASS] All pre-submission checks passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
