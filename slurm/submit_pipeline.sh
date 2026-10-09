#! /bin/bash -l
# Submit the full pipeline: Script 1, then Script 2 (GPU) once Script 1 has finished successfully.
# Run from the repo root:  slurm/submit_pipeline.sh [configs/config.yaml]
set -euo pipefail
cd "$(dirname "$0")/.."
CONFIG="${1:-configs/config.yaml}"
mkdir -p slurm/logs

plan=$(sbatch --parsable slurm/01_create_anonymization_plan.sbatch "$CONFIG")
run=$(sbatch --parsable --dependency=afterok:"$plan" slurm/02_run_with_classifier.sbatch "$CONFIG")
echo "Submitted plan job $plan and run job $run (starts after $plan succeeds)"
