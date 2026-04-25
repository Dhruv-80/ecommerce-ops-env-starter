"""Submit a GRPO training job to HF Jobs using LOCAL code.

This script reads your local environment source files, inlines them
into the training script, and submits everything to HF as a single
self-contained UV script — no git push required.

Usage:
    # Load your .env first
    export $(grep -v '^#' .env | xargs)

    # Smoke test (5 steps, ~$0.80, ~20 min)
    python train/submit_job.py --fast-dev

    # Full run (200 steps, ~$4, ~90 min)
    python train/submit_job.py

    # Custom steps
    python train/submit_job.py --steps 50
"""

from __future__ import annotations

import argparse
import os
import sys
import textwrap
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).parent.parent  # project root

# Local env source files that will be bundled into the job
ENV_FILES = [
    "__init__.py",
    "models.py",
    "tasks.py",
    "verifier.py",
    "reward.py",
    "environment.py",
]

HF_TOKEN        = os.environ.get("HF_TOKEN", "")
HUB_MODEL_REPO  = os.environ.get("HUB_MODEL_REPO",  "dhruvnadamuni/commerce-ops-grpo")
HUB_RESULTS_REPO= os.environ.get("HUB_RESULTS_REPO","dhruvnadamuni/commerce-ops-results")
MODEL_NAME      = os.environ.get("MODEL_NAME",       "Qwen/Qwen2.5-3B-Instruct")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _read_local_files() -> dict[str, str]:
    """Read all local env source files and return {filename: content}."""
    files = {}
    for fname in ENV_FILES:
        fpath = ROOT / fname
        if not fpath.exists():
            print(f"  ⚠️  Missing local file: {fpath} — skipping.")
            continue
        files[fname] = fpath.read_text(encoding="utf-8")
        print(f"  ✅  Bundled {fname} ({len(files[fname])} chars)")
    return files


def _build_script(env_files: dict[str, str], train_steps: int, fast_dev: bool) -> str:
    """
    Construct a self-contained UV script by:
      1. Writing the PEP 723 dependency header.
      2. Adding a preamble that writes env files to /tmp/commerce-ops-env.
      3. Appending the main training logic from hf_train.py (with the git-clone
         section removed, since we bundle files instead).
    """
    train_py = (ROOT / "train" / "hf_train.py").read_text(encoding="utf-8")

    # ── 1. PEP 723 header (keep first block from hf_train.py) ──────────────
    header_end = train_py.index("# ///\n") + len("# ///\n")
    header = train_py[:header_end]

    # ── 2. File-bundle preamble (replaces git clone) ────────────────────────
    files_repr = repr(env_files)  # dict literal, safe to embed in script
    preamble = textwrap.dedent(f"""
        # ── Bundled local env files (injected by submit_job.py) ─────────────
        import os, sys as _sys

        _BUNDLED = {files_repr}

        _ENV_DIR = "/tmp/commerce-ops-env"
        os.makedirs(_ENV_DIR, exist_ok=True)
        for _fname, _content in _BUNDLED.items():
            with open(os.path.join(_ENV_DIR, _fname), "w") as _fh:
                _fh.write(_content)
        _sys.path.insert(0, _ENV_DIR)
        del _BUNDLED, _ENV_DIR, _fname, _content, _fh
        # ── End of bundled files ─────────────────────────────────────────────
    """)

    # ── 3. Body — strip everything up to and including the git-clone block ──
    # We remove from "from __future__" down to the sys.path.insert after clone.
    body_start_marker = "from environment import CommerceOpsEnv"
    body_start = train_py.index(body_start_marker)

    # Also strip the first `from __future__` block and torch/hub imports
    # since the preamble section begins fresh. We keep from SYSTEM_PROMPT onward
    # but we need the imports (torch, re, json, etc.) which come before it.
    # Easiest: take from the original imports section (after the header).
    imports_marker = "from __future__ import annotations"
    imports_start  = train_py.index(imports_marker)

    # The git-clone section starts at "REPO_URL" and ends after sys.path.insert
    clone_start = train_py.index("REPO_URL = os.environ.get(")
    clone_end   = train_py.index(body_start_marker) + len(body_start_marker) + 1

    body = (
        train_py[imports_start:clone_start]   # imports + config constants
        + f"\n{body_start_marker}\n"           # the import that was at clone_end
        + "from tasks import get_task_bundle\n"
        + train_py[clone_end:]                 # rest of the script
    )

    # ── 4. Override key constants with submission-time values ───────────────
    overrides = textwrap.dedent(f"""
        # ── Submission-time overrides ────────────────────────────────────────
        FAST_DEV    = {fast_dev!r}
        TRAIN_STEPS = {train_steps!r}
        MODEL_NAME  = {MODEL_NAME!r}
        HUB_MODEL_REPO   = {HUB_MODEL_REPO!r}
        HUB_RESULTS_REPO = {HUB_RESULTS_REPO!r}
        # ── End overrides ────────────────────────────────────────────────────
    """)

    # Insert overrides just before the EVAL_SEEDS line (which uses FAST_DEV)
    eval_seeds_marker = "EVAL_SEEDS"
    insert_at = body.index(eval_seeds_marker)
    body = body[:insert_at] + overrides + body[insert_at:]

    return header + "\n" + preamble + "\n" + body


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Submit GRPO training job to HF Jobs.")
    parser.add_argument("--fast-dev", action="store_true",
                        help="Smoke test: 5 steps, 2 seeds (~$0.80, ~20 min)")
    parser.add_argument("--steps", type=int, default=None,
                        help="Override training steps (default: 5 if fast-dev, else 200)")
    parser.add_argument("--flavor", default="l4x1",
                        help="HF Jobs hardware flavor (default: l4x1)")
    parser.add_argument("--timeout", default=None,
                        help="Job timeout, e.g. 2h (default: 45m for fast-dev, 3h for full)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Build the script and print it but don't submit")
    args = parser.parse_args()

    fast_dev    = args.fast_dev
    train_steps = args.steps if args.steps is not None else (5 if fast_dev else 200)
    timeout     = args.timeout or ("45m" if fast_dev else "3h")
    flavor      = args.flavor

    if not HF_TOKEN:
        print("❌  HF_TOKEN not set. Run: export $(grep -v '^#' .env | xargs)")
        sys.exit(1)

    print(f"\n{'='*58}")
    print(f"  CommerceOps-Env — HF Jobs Submit")
    print(f"  fast_dev   : {fast_dev}")
    print(f"  steps      : {train_steps}")
    print(f"  flavor     : {flavor}")
    print(f"  timeout    : {timeout}")
    print(f"  model      : {MODEL_NAME}")
    print(f"  hub_model  : {HUB_MODEL_REPO}")
    print(f"  hub_results: {HUB_RESULTS_REPO}")
    print(f"{'='*58}\n")

    print("Reading local source files …")
    env_files = _read_local_files()

    if not env_files:
        print("❌  No env files found. Run from the project root.")
        sys.exit(1)

    print("\nBuilding self-contained training script …")
    script = _build_script(env_files, train_steps=train_steps, fast_dev=fast_dev)
    print(f"  Script size: {len(script):,} chars  ({len(script)//1024} KB)")

    if args.dry_run:
        out_path = ROOT / "train" / "_bundled_job.py"
        out_path.write_text(script, encoding="utf-8")
        print(f"\n[dry-run] Script written to {out_path}")
        print("  Review it before submitting.")
        return

    print("\nSubmitting to HF Jobs …")
    try:
        from huggingface_hub import run_uv_job
    except ImportError:
        print("❌  huggingface_hub not installed. Run: pip install huggingface_hub")
        sys.exit(1)

    job = run_uv_job(
        script,
        flavor=flavor,
        timeout=timeout,
        secrets={"HF_TOKEN": HF_TOKEN},
        env={"HF_HUB_DISABLE_EXPERIMENTAL_WARNING": "1"},
    )

    print(f"\n{'='*58}")
    print(f"  ✅  Job submitted!")
    print(f"  Job ID : {job.id}")
    print(f"  View   : https://huggingface.co/jobs/{job.owner}/{job.id}")
    print(f"\n  Monitor logs:")
    print(f"    hf jobs logs {job.id}")
    print(f"\n  Cancel if needed:")
    print(f"    hf jobs cancel {job.id}")
    print(f"{'='*58}\n")


if __name__ == "__main__":
    main()
