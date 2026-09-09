#!/usr/bin/env bash
set -Eeuo pipefail

# Reuse completed checkpoints. No training and no dependency reinstall.
# Example:
#   RUN_ROOT=outputs/autodl_research_20260909-023128 bash scripts/autodl_run_fast_ensemble.sh

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

RUN_ROOT="${RUN_ROOT:-}"
DATA_ROOT="${DATA_ROOT:-archive}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-8}"
WEIGHT_STEP="${WEIGHT_STEP:-0.1}"
SEED="${SEED:-42}"

if [[ -z "${RUN_ROOT}" ]]; then
  echo "Usage: RUN_ROOT=outputs/autodl_research_YYYYMMDD-HHMMSS bash scripts/autodl_run_fast_ensemble.sh"
  exit 2
fi
if [[ ! -d "${RUN_ROOT}" ]]; then
  echo "Run directory not found: ${RUN_ROOT}"
  exit 2
fi

python tools/evaluate_fast_ensemble.py \
  --run-root "${RUN_ROOT}" \
  --image-dir "${DATA_ROOT}/train_images" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --weight-step "${WEIGHT_STEP}" \
  --seed "${SEED}"

echo "Done: ${RUN_ROOT}/reports/fast_ensemble/report.md"
