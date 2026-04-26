#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

resolve_env_file() {
  if [[ -n "${ENV_FILE:-}" ]]; then
    if [[ ! -f "$ENV_FILE" ]]; then
      echo "ENV_FILE was set but not found: $ENV_FILE" >&2
      exit 1
    fi
    printf '%s\n' "$ENV_FILE"
    return
  fi

  local dir="$ROOT_DIR"
  while true; do
    local candidate="$dir/.env"
    if [[ -f "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return
    fi

    local parent
    parent="$(dirname "$dir")"
    if [[ "$parent" == "$dir" ]]; then
      break
    fi
    dir="$parent"
  done

  echo "Could not find a .env file from $ROOT_DIR upward. Set ENV_FILE=/absolute/path/to/.env" >&2
  exit 1
}

ENV_FILE_PATH="$(resolve_env_file)"
echo "Using env file: $ENV_FILE_PATH"

set -a
source "$ENV_FILE_PATH"
set +a

echo ""
echo "=== Loaded Environment Configuration ==="
echo "MODEL_NAME:   $MODEL_NAME"
echo "TRAIN_STEPS:  $TRAIN_STEPS"
echo "HF_FLAVOR:    $HF_FLAVOR"
echo "HF_TIMEOUT:   $HF_TIMEOUT"
echo "ENV_REPO_URL: $ENV_REPO_URL"
echo "HUB_MODEL_REPO:  $HUB_MODEL_REPO"
echo "HUB_RESULTS_REPO: $HUB_RESULTS_REPO"
echo "========================================"
echo ""

require_env() {
  local key="$1"
  if [[ -z "${!key:-}" ]]; then
    echo "Missing required key in .env: $key" >&2
    exit 1
  fi
}

required=(
  HF_TOKEN
  ENV_REPO_URL
  HUB_MODEL_REPO
  HUB_RESULTS_REPO
  MODEL_NAME
  TRAIN_STEPS
  FAST_DEV
  HF_FLAVOR
  HF_TIMEOUT
)

for key in "${required[@]}"; do
  require_env "$key"
done

HF_HUB_DISABLE_EXPERIMENTAL_WARNING=1 hf jobs uv run train/hf_train.py \
  --flavor "$HF_FLAVOR" \
  --timeout "$HF_TIMEOUT" \
  --secret HF_TOKEN="$HF_TOKEN" \
  --env HF_TOKEN="$HF_TOKEN" \
  --env ENV_REPO_URL="$ENV_REPO_URL" \
  --env HUB_MODEL_REPO="$HUB_MODEL_REPO" \
  --env HUB_RESULTS_REPO="$HUB_RESULTS_REPO" \
  --env MODEL_NAME="$MODEL_NAME" \
  --env TRAIN_STEPS="$TRAIN_STEPS" \
  --env FAST_DEV="$FAST_DEV"
