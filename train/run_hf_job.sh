#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -f ".env" ]]; then
  echo "Missing .env file at project root." >&2
  exit 1
fi

set -a
source .env
set +a

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
  --env ENV_REPO_URL="$ENV_REPO_URL" \
  --env HUB_MODEL_REPO="$HUB_MODEL_REPO" \
  --env HUB_RESULTS_REPO="$HUB_RESULTS_REPO" \
  --env MODEL_NAME="$MODEL_NAME" \
  --env TRAIN_STEPS="$TRAIN_STEPS" \
  --env FAST_DEV="$FAST_DEV"
