#!/usr/bin/env bash
# Script 2 (GPU): preprocess, classify and save every pending manifest record, then build the final table.
# Safe to rerun: finished records are skipped. Pass --retry-errors as the 2nd argument to redo error records.
set -euo pipefail
cd "$(dirname "$0")/.."
CONFIG="${1:-configs/config.yaml}"
shift || true

uv run python run_echo_preprocessing_and_anonymization_with_classifier.py run --config "$CONFIG" "$@"
uv run python run_echo_preprocessing_and_anonymization_with_classifier.py merge --config "$CONFIG"
