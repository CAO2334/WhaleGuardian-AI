#!/usr/bin/env bash
set -Eeuo pipefail

# AutoDL end-to-end research pipeline:
# preflight -> controlled training -> validation-only selection -> independent
# test report -> interpretability -> ONNX artifact -> downloadable archive.
#
# Default (one seed):
#   bash scripts/autodl_run_research_pipeline.sh
# Research run (recommended before an interview/paper):
#   SEEDS="42 123 3407" EPOCHS=20 BATCH_SIZE=8 bash scripts/autodl_run_research_pipeline.sh

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

RUN_TS="$(date +%Y%m%d-%H%M%S)"
DATA_ROOT="${DATA_ROOT:-archive}"
EPOCHS="${EPOCHS:-20}"
BATCH_SIZE="${BATCH_SIZE:-8}"
NUM_WORKERS="${NUM_WORKERS:-8}"
IMAGE_SIZE="${IMAGE_SIZE:-512}"
VAL_RATIO="${VAL_RATIO:-0.10}"
TEST_RATIO="${TEST_RATIO:-0.10}"
SPLIT_SEED="${SPLIT_SEED:-42}"
SEEDS="${SEEDS:-42}"
SUITE="${SUITE:-controlled}"
ONLY="${ONLY:-}"
INSTALL_DEPS="${INSTALL_DEPS:-1}"
REQUIRE_CUDA="${REQUIRE_CUDA:-1}"
KEEP_ALL_CHECKPOINTS_IN_PACKAGE="${KEEP_ALL_CHECKPOINTS_IN_PACKAGE:-0}"
AUTO_SHUTDOWN="${AUTO_SHUTDOWN:-0}"
SHUTDOWN_ON_FAILURE="${SHUTDOWN_ON_FAILURE:-0}"
SHUTDOWN_CMD="${SHUTDOWN_CMD:-/usr/bin/shutdown}"
EXPORT_ENSEMBLE_ARTIFACTS="${EXPORT_ENSEMBLE_ARTIFACTS:-1}"

RUN_ROOT="${RUN_ROOT:-outputs/autodl_research_${RUN_TS}}"
ABLATION_ROOT="${RUN_ROOT}/ablations"
REPORT_ROOT="${RUN_ROOT}/reports"
ARTIFACT_DIR="${RUN_ROOT}/artifact"
FINAL_MODEL_DIR="${RUN_ROOT}/final_model"
LOG_DIR="${RUN_ROOT}/logs"
LOG_FILE="${LOG_DIR}/pipeline.log"
SELECTION_JSON="${RUN_ROOT}/best_experiment.json"
PACKAGE_FILE="whale_research_${RUN_TS}.tar.gz"

mkdir -p "${ABLATION_ROOT}" "${REPORT_ROOT}/dataset" "${REPORT_ROOT}/ablation" \
  "${REPORT_ROOT}/final_test" "${REPORT_ROOT}/interpretability" "${ARTIFACT_DIR}" \
  "${FINAL_MODEL_DIR}" "${LOG_DIR}"
exec > >(tee -a "${LOG_FILE}") 2>&1

shutdown_if_requested() {
  local status="$1"
  if [[ "${AUTO_SHUTDOWN}" != "1" ]]; then
    return 0
  fi
  if [[ "${status}" == "success" || "${SHUTDOWN_ON_FAILURE}" == "1" ]]; then
    sync
    "${SHUTDOWN_CMD}"
  else
    echo "Pipeline failed; AutoDL instance is left running for diagnosis."
  fi
}

on_error() {
  local exit_code=$?
  echo "Pipeline failed with exit code ${exit_code}. See ${LOG_FILE}."
  printf 'Resume completed stages without retraining: RUN_ROOT=%q INSTALL_DEPS=0 bash scripts/autodl_run_research_pipeline.sh\n' "${RUN_ROOT}"
  shutdown_if_requested "failure"
  exit "${exit_code}"
}
trap on_error ERR

echo "Project: ${PROJECT_ROOT}"
echo "Run root: ${RUN_ROOT}"
echo "Suite: ${SUITE}; seeds: ${SEEDS}; epochs: ${EPOCHS}"

python - "${DATA_ROOT}" "${REQUIRE_CUDA}" <<'PY'
import shutil
import sys
from pathlib import Path

import torch

data_root = Path(sys.argv[1])
require_cuda = sys.argv[2] == "1"
csv_path = data_root / "train.csv"
image_dir = data_root / "train_images"
if not csv_path.is_file() or not image_dir.is_dir():
    raise SystemExit(f"Dataset missing: expected {csv_path} and {image_dir}")
if require_cuda and not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable. Set REQUIRE_CUDA=0 only for a deliberate CPU smoke test.")
free_gb = shutil.disk_usage(Path.cwd()).free / (1024 ** 3)
print("python:", sys.version.split()[0])
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    print("cuda runtime:", torch.version.cuda)
print(f"free disk: {free_gb:.1f} GiB")
PY

if [[ "${INSTALL_DEPS}" == "1" ]]; then
  python -m pip install -r requirements-autodl.txt
fi
python -m pip freeze | tee "${RUN_ROOT}/environment.txt"

python tools/analyze_dataset.py \
  --csv "${DATA_ROOT}/train.csv" \
  --image-dir "${DATA_ROOT}/train_images" \
  --output-dir "${REPORT_ROOT}/dataset" \
  --val-ratio "${VAL_RATIO}" \
  --split-strategy group \
  --group-col individual_id \
  --seed "${SPLIT_SEED}"

ABLATION_ARGS=(
  --run
  --suite "${SUITE}"
  --data-root "${DATA_ROOT}"
  --output-root "${ABLATION_ROOT}"
  --epochs "${EPOCHS}"
  --batch-size "${BATCH_SIZE}"
  --image-size "${IMAGE_SIZE}"
  --num-workers "${NUM_WORKERS}"
  --val-ratio "${VAL_RATIO}"
  --test-ratio "${TEST_RATIO}"
  --split-seed "${SPLIT_SEED}"
  --seeds ${SEEDS}
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
  --output "${REPORT_ROOT}/ablation/report.md" \
  --plot "${REPORT_ROOT}/ablation/macro_f1.png"

python tools/select_best_experiment.py \
  --csv "${ABLATION_ROOT}/ablation_results.csv" \
  --output "${SELECTION_JSON}"

CHECKPOINT="$(python -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["representative_checkpoint"])' "${SELECTION_JSON}")"
BEST_MODEL_TYPE="$(python -c 'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))["model_type"])' "${SELECTION_JSON}")"
BEST_DIR="$(dirname "${CHECKPOINT}")"
CLASS_MAP="${BEST_DIR}/class_to_idx.json"
TRAIN_SPLIT="${BEST_DIR}/splits/train.csv"
TEST_SPLIT="${BEST_DIR}/splits/test.csv"

if [[ ! -s "${TEST_SPLIT}" ]]; then
  echo "Independent test split is empty. Use TEST_RATIO > 0."
  exit 2
fi

python tools/generate_model_report.py \
  --checkpoint "${CHECKPOINT}" \
  --class-map "${CLASS_MAP}" \
  --eval-csv "${TEST_SPLIT}" \
  --train-csv "${TRAIN_SPLIT}" \
  --image-dir "${DATA_ROOT}/train_images" \
  --output-dir "${REPORT_ROOT}/final_test" \
  --split-name independent_test \
  --independent-test \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --bootstrap-iters 1000 \
  --group-col individual_id

mapfile -t SAMPLE_IMAGES < <(python - "${REPORT_ROOT}/final_test/predictions.csv" "${DATA_ROOT}/train_images" <<'PY'
import sys
from pathlib import Path
import pandas as pd

predictions = pd.read_csv(sys.argv[1])
image_dir = Path(sys.argv[2])
picked = []
wrong = predictions.loc[~predictions["correct"].astype(bool)].sort_values("confidence", ascending=False)
correct = predictions.loc[predictions["correct"].astype(bool)].sort_values("confidence")
for frame in (wrong.head(1), correct.head(1), correct.tail(1)):
    if not frame.empty:
        path = image_dir / str(frame.iloc[0]["image"])
        if path.exists() and str(path) not in picked:
            picked.append(str(path))
for path in picked:
    print(path)
PY
)

sample_index=0
for sample_image in "${SAMPLE_IMAGES[@]}"; do
  sample_index=$((sample_index + 1))
  python tools/generate_gradcam.py \
    --image "${sample_image}" \
    --checkpoint "${CHECKPOINT}" \
    --class-map "${CLASS_MAP}" \
    --output "${REPORT_ROOT}/interpretability/gradcam_${sample_index}.jpg"
done

if [[ "${BEST_MODEL_TYPE}" == "transformer" && ${#SAMPLE_IMAGES[@]} -gt 0 ]]; then
  python tools/generate_attention_map.py \
    --image "${SAMPLE_IMAGES[0]}" \
    --checkpoint "${CHECKPOINT}" \
    --class-map "${CLASS_MAP}" \
    --output "${REPORT_ROOT}/interpretability/attention_map.jpg"
fi

python tools/export_onnx.py \
  --checkpoint "${CHECKPOINT}" \
  --class-map "${CLASS_MAP}" \
  --metrics "${REPORT_ROOT}/final_test/metrics.json" \
  --artifact-dir "${ARTIFACT_DIR}" \
  --version "${RUN_TS}"

if [[ "${EXPORT_ENSEMBLE_ARTIFACTS}" == "1" ]]; then
  python tools/export_ensemble_artifacts.py \
    --run-root "${RUN_ROOT}" \
    --output-dir "${RUN_ROOT}/ensemble_artifacts" \
    --seed "${SPLIT_SEED}"
fi

cp "${CHECKPOINT}" "${FINAL_MODEL_DIR}/best_model.pth"
cp "${CLASS_MAP}" "${FINAL_MODEL_DIR}/class_to_idx.json"
cp "${BEST_DIR}/metrics.json" "${FINAL_MODEL_DIR}/training_metrics.json"
cp "${BEST_DIR}/splits/split_summary.json" "${FINAL_MODEL_DIR}/split_summary.json"

python - "${RUN_ROOT}" "${SELECTION_JSON}" "${PACKAGE_FILE}" <<'PY'
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

run_root = Path(sys.argv[1])
selection_path = Path(sys.argv[2])
selection = json.loads(selection_path.read_text(encoding="utf-8"))
selection["packaged_checkpoint"] = str(run_root / "final_model" / "best_model.pth")
selection_path.write_text(json.dumps(selection, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
manifest = {
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
    "platform": platform.platform(),
    "python": platform.python_version(),
    "selection": selection,
    "package": sys.argv[3],
    "test_set_used_for_selection": False,
}
(run_root / "run_manifest.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
PY

if [[ "${KEEP_ALL_CHECKPOINTS_IN_PACKAGE}" == "1" ]]; then
  tar -czf "${PACKAGE_FILE}" "${RUN_ROOT}"
else
  tar --exclude='*/ablations/*/seed_*/best_model.pth' -czf "${PACKAGE_FILE}" "${RUN_ROOT}"
fi
echo "Done. Download ${PACKAGE_FILE}"
echo "Final report: ${REPORT_ROOT}/final_test/report.md"
echo "Best experiment: ${SELECTION_JSON}"
shutdown_if_requested "success"
