#!/usr/bin/env bash
set -euo pipefail

# Package reports/artifacts from an existing trained checkpoint.
#
# Default target is ablation experiment 04:
#   outputs/ablations/04_transformer_mean_focal_mixup_cutout/
#
# Usage:
#   bash scripts/autodl_package_existing_model.sh
#
# Override example:
#   RUN_NAME=best_exp \
#   CHECKPOINT=outputs/ablations/04_transformer_mean_focal_mixup_cutout/best_model.pth \
#   CLASS_MAP=outputs/ablations/04_transformer_mean_focal_mixup_cutout/class_to_idx.json \
#   METRICS=outputs/ablations/04_transformer_mean_focal_mixup_cutout/metrics.json \
#   bash scripts/autodl_package_existing_model.sh

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

RUN_TS="$(date +%Y%m%d-%H%M%S)"

DATA_ROOT="${DATA_ROOT:-archive}"
RUN_NAME="${RUN_NAME:-ablation04_transformer_mean_focal_mixup_cutout}"
CHECKPOINT="${CHECKPOINT:-outputs/ablations/04_transformer_mean_focal_mixup_cutout/best_model.pth}"
CLASS_MAP="${CLASS_MAP:-outputs/ablations/04_transformer_mean_focal_mixup_cutout/class_to_idx.json}"
METRICS="${METRICS:-outputs/ablations/04_transformer_mean_focal_mixup_cutout/metrics.json}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-8}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/package_${RUN_NAME}_${RUN_TS}}"
REPORT_DIR="${REPORT_DIR:-outputs/reports/${RUN_NAME}_${RUN_TS}}"
ARTIFACT_DIR="${ARTIFACT_DIR:-artifacts/${RUN_NAME}_${RUN_TS}}"

mkdir -p logs "${OUTPUT_DIR}" "${REPORT_DIR}/analysis" "${REPORT_DIR}/evaluation" "${REPORT_DIR}/interpretability" "${ARTIFACT_DIR}"
LOG_FILE="logs/package_${RUN_NAME}_${RUN_TS}.log"

exec > >(tee -a "${LOG_FILE}") 2>&1

echo "Project root: ${PROJECT_ROOT}"
echo "Run name: ${RUN_NAME}"
echo "Checkpoint: ${CHECKPOINT}"
echo "Class map: ${CLASS_MAP}"
echo "Metrics: ${METRICS}"
echo "Report dir: ${REPORT_DIR}"
echo "Artifact dir: ${ARTIFACT_DIR}"
echo "Log file: ${LOG_FILE}"

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Checkpoint not found: ${CHECKPOINT}"
  exit 1
fi

if [[ ! -f "${CLASS_MAP}" ]]; then
  echo "Class map not found: ${CLASS_MAP}"
  exit 1
fi

echo "Verify Python environment..."
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

echo "Copy checkpoint-side files into package output..."
cp "${CHECKPOINT}" "${OUTPUT_DIR}/best_model.pth"
cp "${CLASS_MAP}" "${OUTPUT_DIR}/class_to_idx.json"
if [[ -f "${METRICS}" ]]; then
  cp "${METRICS}" "${OUTPUT_DIR}/metrics.json"
fi

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
if [[ -f "${METRICS}" ]]; then
  python tools/export_onnx.py \
    --checkpoint "${CHECKPOINT}" \
    --class-map "${CLASS_MAP}" \
    --metrics "${METRICS}" \
    --artifact-dir "${ARTIFACT_DIR}" \
    --version "${RUN_TS}"
else
  python tools/export_onnx.py \
    --checkpoint "${CHECKPOINT}" \
    --class-map "${CLASS_MAP}" \
    --artifact-dir "${ARTIFACT_DIR}" \
    --version "${RUN_TS}"
fi

echo "Package outputs..."
tar -czf "${RUN_NAME}_${RUN_TS}.tar.gz" "${OUTPUT_DIR}" "${REPORT_DIR}" "${ARTIFACT_DIR}" "${LOG_FILE}"

echo "Done."
echo "Package file: ${RUN_NAME}_${RUN_TS}.tar.gz"
echo "Checkpoint copy: ${OUTPUT_DIR}/best_model.pth"
echo "Reports: ${REPORT_DIR}"
echo "Artifact: ${ARTIFACT_DIR}"
