#! /bin/bash -l
# Submit script 1
# Run from the repo root:  slurm/submit_01.sh [configs/config.yaml]
set -euo pipefail
cd "$(dirname "$0")/.."
CONFIG="${1:-configs/config.yaml}"
mkdir -p slurm/logs

plan=$(sbatch --parsable slurm/01_create_anonymization_plan.sbatch "$CONFIG")
echo "Submitted plan job $plan"
