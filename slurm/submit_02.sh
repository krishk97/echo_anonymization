#! /bin/bash -l
# Submit script 2
# Run from the repo root:  slurm/submit_02.sh [configs/config.yaml]
set -euo pipefail
cd "$(dirname "$0")/.."
CONFIG="${1:-configs/config.yaml}"
mkdir -p slurm/logs

run=$(sbatch --parsable slurm/02_run_with_classifier.sbatch "$CONFIG")
echo "Submitted run job $run"
