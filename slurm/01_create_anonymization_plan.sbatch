#!/usr/bin/env bash
# Script 1 (CPU): index DICOM headers, extend the anonymization crosswalk, write the work manifest.
# Submit from the repo root:  sbatch slurm/01_create_anonymization_plan.sbatch [configs/config.yaml]
# Safe to resubmit: completed chunks are skipped.
#
# --cpus-per-task should match index.num_workers in the config (+ a little headroom for the main process).
#SBATCH --job-name=echo-anon-plan
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=24:00:00
##SBATCH --partition=<cpu-partition>
##SBATCH --account=<account>
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
CONFIG="${1:-configs/config.yaml}"

# module load ...   # if uv is not already on PATH on the compute nodes

uv run python create_anonymization_plan.py --config "$CONFIG"
