#!/usr/bin/env bash
set -Eeuo pipefail

# Compatibility entry point for the complete controlled AutoDL study.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export SUITE="${SUITE:-controlled}"
exec bash "${PROJECT_ROOT}/scripts/autodl_run_research_pipeline.sh"
