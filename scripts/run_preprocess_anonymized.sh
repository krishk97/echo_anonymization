#!/usr/bin/env bash
# Preprocess an already-anonymized DICOM dataset from its record list (no crosswalk, no date shift),
# then build the final lists. All arguments are passed to run_preprocessing_on_anonymized.py, e.g.
#   scripts/run_preprocess_anonymized.sh --record-list echo-record-list.csv --output-root OUT --work-dir WORK
# Safe to rerun: the work manifest is reused and finished records are skipped.
set -euo pipefail
cd "$(dirname "$0")/.."

uv run python run_preprocessing_on_anonymized.py "$@"
