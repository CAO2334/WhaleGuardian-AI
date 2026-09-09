#!/usr/bin/env bash
set -Eeuo pipefail

# Generate reports/artifact for an existing checkpoint. If the checkpoint was
# trained by the new pipeline, its frozen test split is used automatically.
# Legacy checkpoints have no untouched test split and are explicitly labelled
# non-independent instead of presenting a re-split validation score as a test.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
RUN_TS="$(date +%Y%m%d-%H%M%S)"

DATA_ROOT="${DATA_ROOT:-archive}"
RUN_NAME="${RUN_NAME:-existing_model}"
CHECKPOINT="${CHECKPOINT:-outputs/reports/final_model_04/best_model.pth}"
CLASS_MAP="${CLASS_MAP:-$(dirname "${CHECKPOINT}")/class_to_idx.json}"
METRICS="${METRICS:-$(dirname "${CHECKPOINT}")/metrics.json}"
TRAIN_SPLIT="${TRAIN_SPLIT:-$(dirname "${CHECKPOINT}")/splits/train.csv}"
EVAL_CSV="${EVAL_CSV:-$(dirname "${CHECKPOINT}")/splits/test.csv}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-8}"
INSTALL_DEPS="${INSTALL_DEPS:-1}"
RUN_ROOT="${RUN_ROOT:-outputs/package_${RUN_NAME}_${RUN_TS}}"
REPORT_DIR="${RUN_ROOT}/report"
ARTIFACT_DIR="${RUN_ROOT}/artifact"
FINAL_MODEL_DIR="${RUN_ROOT}/final_model"
LOG_FILE="${RUN_ROOT}/package.log"

mkdir -p "${REPORT_DIR}" "${ARTIFACT_DIR}" "${FINAL_MODEL_DIR}" "${RUN_ROOT}/interpretability"
exec > >(tee -a "${LOG_FILE}") 2>&1

for required in "${CHECKPOINT}" "${CLASS_MAP}" "${DATA_ROOT}/train.csv" "${DATA_ROOT}/train_images"; do
  if [[ ! -e "${required}" ]]; then
    echo "Required path missing: ${required}"
    exit 1
  fi
done

if [[ "${INSTALL_DEPS}" == "1" ]]; then
  python -m pip install -r requirements-autodl.txt
fi
cp "${CHECKPOINT}" "${FINAL_MODEL_DIR}/best_model.pth"
cp "${CLASS_MAP}" "${FINAL_MODEL_DIR}/class_to_idx.json"

if [[ -s "${EVAL_CSV}" ]]; then
  SPLIT_NAME="independent_test"
  INDEPENDENT_ARG="--independent-test"
else
  EVAL_CSV="${RUN_ROOT}/legacy_reconstructed_validation.csv"
  TRAIN_SPLIT="${RUN_ROOT}/legacy_reconstructed_train.csv"
  SPLIT_NAME="legacy_reconstructed_validation_non_independent"
  INDEPENDENT_ARG=""
  echo "WARNING: legacy checkpoint has no frozen untouched test CSV."
  echo "Reconstructing its historical validation split. Scores must not be reported as test performance."
  python - "${CHECKPOINT}" "${DATA_ROOT}/train.csv" "${TRAIN_SPLIT}" "${EVAL_CSV}" <<'PY'
import sys
import torch
import pandas as pd
from data.dataset import normalize_species_column, split_train_val

checkpoint = torch.load(sys.argv[1], map_location="cpu")
cfg = checkpoint.get("config", {})
frame = normalize_species_column(pd.read_csv(sys.argv[2]), fix_typos=bool(cfg.get("fix_species_typos", True)))
train, validation = split_train_val(
    frame,
    label_col="species",
    val_ratio=float(cfg.get("val_ratio", 0.2)),
    seed=int(cfg.get("split_seed", cfg.get("seed", 42))),
    split_strategy=str(cfg.get("split_strategy", "group")),
    group_col=str(cfg.get("group_col", "individual_id")),
)
train.to_csv(sys.argv[3], index=False, encoding="utf-8-sig")
validation.to_csv(sys.argv[4], index=False, encoding="utf-8-sig")
PY
fi
if [[ ! -s "${TRAIN_SPLIT}" ]]; then
  TRAIN_SPLIT="${DATA_ROOT}/train.csv"
fi

python tools/generate_model_report.py \
  --checkpoint "${CHECKPOINT}" \
  --class-map "${CLASS_MAP}" \
  --eval-csv "${EVAL_CSV}" \
  --train-csv "${TRAIN_SPLIT}" \
  --image-dir "${DATA_ROOT}/train_images" \
  --output-dir "${REPORT_DIR}" \
  --split-name "${SPLIT_NAME}" \
  ${INDEPENDENT_ARG} \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --bootstrap-iters 1000

SAMPLE_IMAGE="$(python -c 'import pandas as pd,sys; from pathlib import Path; d=pd.read_csv(sys.argv[1]); print(Path(sys.argv[2]) / str(d.iloc[0]["image"]))' "${REPORT_DIR}/predictions.csv" "${DATA_ROOT}/train_images")"
python tools/generate_gradcam.py \
  --image "${SAMPLE_IMAGE}" \
  --checkpoint "${CHECKPOINT}" \
  --class-map "${CLASS_MAP}" \
  --output "${RUN_ROOT}/interpretability/gradcam.jpg"

MODEL_TYPE="$(python -c 'import torch,sys; c=torch.load(sys.argv[1], map_location="cpu"); print(c.get("config",{}).get("model_type","transformer"))' "${CHECKPOINT}")"
if [[ "${MODEL_TYPE}" == "transformer" ]]; then
  python tools/generate_attention_map.py \
    --image "${SAMPLE_IMAGE}" \
    --checkpoint "${CHECKPOINT}" \
    --class-map "${CLASS_MAP}" \
    --output "${RUN_ROOT}/interpretability/attention_map.jpg"
fi

EXPORT_ARGS=(
  --checkpoint "${CHECKPOINT}"
  --class-map "${CLASS_MAP}"
  --artifact-dir "${ARTIFACT_DIR}"
  --version "${RUN_TS}"
)
if [[ -f "${METRICS}" ]]; then
  EXPORT_ARGS+=(--metrics "${METRICS}")
else
  EXPORT_ARGS+=(--metrics "${REPORT_DIR}/metrics.json")
fi
python tools/export_onnx.py "${EXPORT_ARGS[@]}"

PACKAGE_FILE="${RUN_NAME}_${RUN_TS}.tar.gz"
tar -czf "${PACKAGE_FILE}" "${RUN_ROOT}"
echo "Done. Download ${PACKAGE_FILE}"
