#!/usr/bin/env bash
set -Eeuo pipefail

# Compatibility entry point: train the historically best Transformer recipe,
# then run the same independent-test/report/export pipeline as the full suite.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export SUITE="controlled"
export ONLY="${ONLY:-07_previous_best_transformer_recipe}"
exec bash "${PROJECT_ROOT}/scripts/autodl_run_research_pipeline.sh"
