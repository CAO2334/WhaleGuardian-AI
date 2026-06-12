#!/usr/bin/env bash
set -euo pipefail

# AutoDL/Linux one-command main-model runner.
# It trains only the final ResNet50-Transformer model and then generates reports.
#
# Usage:
#   bash scripts/autodl_run_main_model.sh
#
# Optional env vars:
#   EPOCHS=20 BATCH_SIZE=8 NUM_WORKERS=8 DATA_ROOT=archive bash scripts/autodl_run_main_model.sh
#   AUTO_SHUTDOWN=1 EPOCHS=20 BATCH_SIZE=8 NUM_WORKERS=8 bash scripts/autodl_run_main_model.sh

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

RUN_TS="$(date +%Y%m%d-%H%M%S)"

EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-8}"
DATA_ROOT="${DATA_ROOT:-archive}"
IMAGE_SIZE="${IMAGE_SIZE:-512}"
TOKEN_POOL_SIZE="${TOKEN_POOL_SIZE:-16}"
BACKBONE_STAGE="${BACKBONE_STAGE:-layer3_layer4}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/main_model_${RUN_TS}}"
REPORT_DIR="${REPORT_DIR:-outputs/reports/main_model_${RUN_TS}}"
ARTIFACT_DIR="${ARTIFACT_DIR:-artifacts/whale_resnet50_transformer_${RUN_TS}}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-main_resnet50_transformer_cls_layer3_layer4_tp16_focal_mixup_cutout_ema}"
AUTO_SHUTDOWN="${AUTO_SHUTDOWN:-0}"
SHUTDOWN_ON_FAILURE="${SHUTDOWN_ON_FAILURE:-0}"
SHUTDOWN_CMD="${SHUTDOWN_CMD:-/usr/bin/shutdown}"

mkdir -p logs "${OUTPUT_DIR}" "${REPORT_DIR}/analysis" "${REPORT_DIR}/evaluation" "${REPORT_DIR}/interpretability" "${ARTIFACT_DIR}"
LOG_FILE="logs/autodl_main_model_${RUN_TS}.log"

# Save complete stdout/stderr. After AutoDL shutdown, terminal output is gone,
# but this log is included in the downloadable package.
exec > >(tee -a "${LOG_FILE}") 2>&1

shutdown_if_requested() {
  local status="$1"
  if [[ "${AUTO_SHUTDOWN}" != "1" ]]; then
    return 0
  fi

  if [[ "${status}" == "success" || "${SHUTDOWN_ON_FAILURE}" == "1" ]]; then
    sync
    echo "AUTO_SHUTDOWN=1, call ${SHUTDOWN_CMD} now. status=${status}"
    "${SHUTDOWN_CMD}"
  else
    echo "Training/report failed, skip shutdown. Set SHUTDOWN_ON_FAILURE=1 to shutdown even on failure."
  fi
}

on_error() {
  local exit_code=$?
  echo "Script failed with exit code ${exit_code}."
  shutdown_if_requested "failure"
  exit "${exit_code}"
}

trap on_error ERR

echo "Project root: ${PROJECT_ROOT}"
echo "Output dir: ${OUTPUT_DIR}"
echo "Report dir: ${REPORT_DIR}"
echo "Artifact dir: ${ARTIFACT_DIR}"
echo "Log file: ${LOG_FILE}"
echo "Auto shutdown: ${AUTO_SHUTDOWN}"
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

echo "Start main model training..."
python train.py \
  --data-root "${DATA_ROOT}" \
  --output-dir "${OUTPUT_DIR}" \
  --experiment-name "${EXPERIMENT_NAME}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE}" \
  --image-size "${IMAGE_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --split-strategy group \
  --group-col individual_id \
  --model-type transformer \
  --loss-type focal \
  --mixup-alpha 0.4 \
  --cutout-p 0.5 \
  --backbone-stage "${BACKBONE_STAGE}" \
  --transformer-pooling cls \
  --token-pool-size "${TOKEN_POOL_SIZE}"

CHECKPOINT="${OUTPUT_DIR}/best_model.pth"
CLASS_MAP="${OUTPUT_DIR}/class_to_idx.json"
METRICS="${OUTPUT_DIR}/metrics.json"

echo "Generate dataset and per-class analysis..."
python tools/analyze_dataset.py \
  --csv "${DATA_ROOT}/train.csv" \
  --image-dir "${DATA_ROOT}/train_images" \
  --output-dir "${REPORT_DIR}/analysis" \
  --checkpoint "${CHECKPOINT}" \
  --class-map "${CLASS_MAP}" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --split-strategy group \
  --group-col individual_id

echo "Generate confusion matrix..."
python tools/eval_confusion_matrix.py \
  --checkpoint "${CHECKPOINT}" \
  --class-map "${CLASS_MAP}" \
  --eval-csv "${DATA_ROOT}/train.csv" \
  --image-dir "${DATA_ROOT}/train_images" \
  --output "${REPORT_DIR}/evaluation/confusion_matrix.png" \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --split-strategy group \
  --group-col individual_id \
  --normalize

SAMPLE_IMAGE="$(python - <<PY
import pandas as pd
from pathlib import Path
csv_path = Path("${DATA_ROOT}") / "train.csv"
image_dir = Path("${DATA_ROOT}") / "train_images"
df = pd.read_csv(csv_path)
for name in df["image"].astype(str):
    path = image_dir / name
    if path.exists():
        print(path.as_posix())
        break
PY
)"

if [[ -n "${SAMPLE_IMAGE}" ]]; then
  echo "Generate Grad-CAM for ${SAMPLE_IMAGE}..."
  python tools/generate_gradcam.py \
    --image "${SAMPLE_IMAGE}" \
    --checkpoint "${CHECKPOINT}" \
    --class-map "${CLASS_MAP}" \
    --output "${REPORT_DIR}/interpretability/gradcam.jpg"

  echo "Generate Transformer attention map for ${SAMPLE_IMAGE}..."
  python tools/generate_attention_map.py \
    --image "${SAMPLE_IMAGE}" \
    --checkpoint "${CHECKPOINT}" \
    --class-map "${CLASS_MAP}" \
    --output "${REPORT_DIR}/interpretability/attention_map.jpg"
else
  echo "Warning: no sample image found, skip Grad-CAM and attention map."
fi

echo "Export ONNX artifact..."
python tools/export_onnx.py \
  --checkpoint "${CHECKPOINT}" \
  --class-map "${CLASS_MAP}" \
  --metrics "${METRICS}" \
  --artifact-dir "${ARTIFACT_DIR}" \
  --version "${RUN_TS}"

echo "Package main model outputs..."
tar -czf "main_model_results_${RUN_TS}.tar.gz" "${OUTPUT_DIR}" "${REPORT_DIR}" "${ARTIFACT_DIR}" "${LOG_FILE}"

echo "Done."
echo "Checkpoint: ${CHECKPOINT}"
echo "Metrics: ${METRICS}"
echo "Reports: ${REPORT_DIR}"
echo "Artifact: ${ARTIFACT_DIR}"
echo "Download package: main_model_results_${RUN_TS}.tar.gz"

shutdown_if_requested "success"
