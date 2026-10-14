#!/usr/bin/env bash
# Preprocess MIMIC-IV-ECHO (already anonymized) with run_preprocessing_on_anonymized.py: crop/mask and save every
# listed DICOM, then build echo-video-list.csv, echo-image-list.csv and a filtered echo-study-list.csv.
# Run from anywhere:  scripts/run_preprocess_mimic_iv_echo.sh [--classify] [--retry-errors]
# Extra arguments are passed through (--classify also runs the view classifiers and needs GPUs).
# Locations can be overridden with environment variables, e.g.
#   OUTPUT_ROOT=/scratch/me/mimic-echo scripts/run_preprocess_mimic_iv_echo.sh
# Safe to rerun: the work manifest is reused and finished records are skipped.
set -euo pipefail
cd "$(dirname "$0")/.."

MIMIC_ROOT="${MIMIC_ROOT:-/scratch/shared/physionet.org/files/mimic-iv-echo/1.0.1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-/scratch/$USER/mimic-iv-echo-preprocessed}"
# Holds absolute source paths; keep it outside OUTPUT_ROOT.
WORK_DIR="${WORK_DIR:-/scratch/$USER/mimic-iv-echo-work}"
CONFIG="${CONFIG:-configs/config_preprocess_anonymized.yaml}"

if [[ ! -f "$MIMIC_ROOT/echo-record-list.csv" ]]; then
  echo "No echo-record-list.csv in $MIMIC_ROOT (set MIMIC_ROOT)" >&2
  exit 1
fi
STUDY_LIST=()
if [[ -f "$MIMIC_ROOT/echo-study-list.csv" ]]; then
  STUDY_LIST=(--study-list "$MIMIC_ROOT/echo-study-list.csv")
fi

uv run python run_preprocessing_on_anonymized.py \
  --config "$CONFIG" \
  --record-list "$MIMIC_ROOT/echo-record-list.csv" \
  "${STUDY_LIST[@]}" \
  --path-column dicom_filepath --patient-column subject_id --study-column study_id \
  --dicom-root "$MIMIC_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --work-dir "$WORK_DIR" \
  "$@"
