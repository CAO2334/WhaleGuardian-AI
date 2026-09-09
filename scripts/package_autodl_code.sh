#!/usr/bin/env bash
set -Eeuo pipefail

# Package only the tracked technical project files for upload to AutoDL.
# Dataset, checkpoints, outputs and local interview materials are excluded.
# Run after committing the desired code version:
#   bash scripts/package_autodl_code.sh

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

OUTPUT="${1:-autodl_whale_project_$(date +%Y%m%d-%H%M%S).tar.gz}"
if [[ "${OUTPUT}" != /* ]]; then
  OUTPUT="${PROJECT_ROOT}/${OUTPUT}"
fi

git diff --quiet || {
  echo "工作区存在未提交修改，请先提交或明确检查后再打包。" >&2
  exit 1
}
git diff --cached --quiet || {
  echo "暂存区存在未提交修改，请先提交或明确检查后再打包。" >&2
  exit 1
}

git archive --format=tar.gz --prefix=AI鲸鱼/ HEAD -o "${OUTPUT}"
echo "AutoDL 代码包已生成: ${OUTPUT}"
echo "包内不包含 archive/、outputs/、权重、ONNX 和本地面试材料。"
