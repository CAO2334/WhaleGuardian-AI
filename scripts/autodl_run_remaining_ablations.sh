#!/usr/bin/env bash
set -Eeuo pipefail

# Run the ten high-value follow-up ablations (seed 42 only) after the completed
# controlled 01-07 suite.  Results are written to a fresh run directory so the
# original reports remain unchanged.  Resume safely with the same RUN_ROOT.
#
# Example:
#   bash scripts/autodl_run_remaining_ablations.sh
#   RUN_ROOT=outputs/autodl_remaining_20260909-023128 \
#     bash scripts/autodl_run_remaining_ablations.sh

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

RUN_TS="$(date +%Y%m%d-%H%M%S)"
DATA_ROOT="${DATA_ROOT:-archive}"
RUN_ROOT="${RUN_ROOT:-outputs/autodl_remaining_${RUN_TS}}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-8}"
IMAGE_SIZE="${IMAGE_SIZE:-512}"
VAL_RATIO="${VAL_RATIO:-0.10}"
TEST_RATIO="${TEST_RATIO:-0.10}"
SPLIT_SEED="${SPLIT_SEED:-42}"
INSTALL_DEPS="${INSTALL_DEPS:-1}"
REQUIRE_CUDA="${REQUIRE_CUDA:-1}"
ONLY="${ONLY:-}"

ABLATION_ROOT="${RUN_ROOT}/ablations"
REPORT_ROOT="${RUN_ROOT}/reports/ablation"
LOG_DIR="${RUN_ROOT}/logs"
mkdir -p "${ABLATION_ROOT}" "${REPORT_ROOT}" "${LOG_DIR}"
exec > >(tee -a "${LOG_DIR}/remaining_ablations.log") 2>&1

python - "${DATA_ROOT}" "${REQUIRE_CUDA}" <<'PY'
import shutil
import sys
from pathlib import Path

import torch

data_root = Path(sys.argv[1])
if not (data_root / "train.csv").is_file() or not (data_root / "train_images").is_dir():
    raise SystemExit(f"Dataset missing: expected {data_root / 'train.csv'} and {data_root / 'train_images'}")
if sys.argv[2] == "1" and not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable. Set REQUIRE_CUDA=0 only for a deliberate CPU smoke test.")
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
print(f"free disk: {shutil.disk_usage(Path.cwd()).free / (1024 ** 3):.1f} GiB")
PY

if [[ "${INSTALL_DEPS}" == "1" ]]; then
  python -m pip install -r requirements-autodl.txt
fi
python -m pip freeze > "${RUN_ROOT}/environment.txt"

ABLATION_ARGS=(
  --run
  --suite remaining
  --data-root "${DATA_ROOT}"
  --output-root "${ABLATION_ROOT}"
  --epochs "${EPOCHS}"
  --batch-size "${BATCH_SIZE}"
  --image-size "${IMAGE_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --val-ratio "${VAL_RATIO}"
  --test-ratio "${TEST_RATIO}"
  --split-seed "${SPLIT_SEED}"
  --seeds 42
  --split-strategy group
  --group-col individual_id
  --deterministic
  --skip-completed
)
if [[ -n "${ONLY}" ]]; then
  ABLATION_ARGS+=(--only ${ONLY})
fi

python tools/run_ablation.py "${ABLATION_ARGS[@]}"
python tools/generate_ablation_report.py \
  --csv "${ABLATION_ROOT}/ablation_results.csv" \
  --output "${REPORT_ROOT}/report.md" \
  --plot "${REPORT_ROOT}/macro_f1.png"

echo "Remaining ablation report: ${REPORT_ROOT}/report.md"
echo "Remaining ablation CSV: ${ABLATION_ROOT}/ablation_results.csv"
