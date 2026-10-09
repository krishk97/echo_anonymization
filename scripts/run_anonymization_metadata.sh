#!/usr/bin/env bash
# Script 1 (CPU): index DICOM headers, extend the anonymization crosswalk, write the work manifest.
# Safe to rerun: completed chunks are skipped.
set -euo pipefail
cd "$(dirname "$0")/.."
CONFIG="${1:-configs/config.yaml}"

uv run python create_anonymization_plan.py --config "$CONFIG"
