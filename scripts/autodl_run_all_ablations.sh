#!/usr/bin/env bash
set -euo pipefail

# AutoDL/Linux one-command ablation runner.
# Usage:
#   bash scripts/autodl_run_all_ablations.sh
#
# Optional env vars:
#   EPOCHS=20 BATCH_SIZE=8 NUM_WORKERS=8 DATA_ROOT=archive bash scripts/autodl_run_all_ablations.sh

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

RUN_TS="$(date +%Y%m%d-%H%M%S)"

EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-8}"
DATA_ROOT="${DATA_ROOT:-archive}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/ablations}"
TOKEN_POOL_SIZE="${TOKEN_POOL_SIZE:-16}"
BACKBONE_STAGE="${BACKBONE_STAGE:-layer3_layer4}"

mkdir -p logs outputs/reports/ablation "${OUTPUT_ROOT}"

echo "Project root: ${PROJECT_ROOT}"
echo "Python: $(which python)"
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    print("cuda:", torch.version.cuda)
PY

echo "Install/verify Python dependencies..."
python -m pip install -r requirements.txt

echo "Dry-run ablation commands..."
python tools/run_ablation.py \
  --dry-run \
  --data-root "${DATA_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --token-pool-size "${TOKEN_POOL_SIZE}" \
  --backbone-stage "${BACKBONE_STAGE}"

echo "Start all ablation experiments..."
python tools/run_ablation.py \
  --run \
  --data-root "${DATA_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --token-pool-size "${TOKEN_POOL_SIZE}" \
  --backbone-stage "${BACKBONE_STAGE}" \
  2>&1 | tee "logs/autodl_ablation_${RUN_TS}.log"

echo "Generate ablation report..."
python tools/generate_ablation_report.py --csv "${OUTPUT_ROOT}/ablation_results.csv"

echo "Organize report files..."
python tools/organize_outputs.py || true

echo "Package ablation outputs..."
tar -czf "ablation_results_${RUN_TS}.tar.gz" "${OUTPUT_ROOT}" outputs/reports "logs/autodl_ablation_${RUN_TS}.log"

echo "Done. Key files:"
echo "  ${OUTPUT_ROOT}/ablation_results.csv"
echo "  outputs/reports/ablation/"
echo "  logs/"
echo "  ablation_results_${RUN_TS}.tar.gz"
