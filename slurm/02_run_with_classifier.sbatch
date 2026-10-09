#!/usr/bin/env bash
# Script 2 (GPU): preprocess, classify and save every pending manifest record, then build the final table.
# Submit from the repo root:  sbatch slurm/02_run_with_classifier.sbatch [configs/config.yaml] [--retry-errors]
# Safe to resubmit (e.g. after a timeout): finished records are skipped.
#
# The GPU count in --gres must be at least preprocess.num_gpus in the config; the run fails otherwise.
# Each GPU process starts preprocess.num_workers DataLoader workers, so give it
# num_gpus * (num_workers + 1) CPUs or more (4 * (4 + 1) = 20 with the defaults).
#SBATCH --job-name=echo-anon-run
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=48:00:00
##SBATCH --partition=<gpu-partition>
##SBATCH --account=<account>
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
CONFIG="${1:-configs/config.yaml}"
shift || true

# module load ...   # if uv is not already on PATH on the compute nodes

nvidia-smi -L
uv run python run_echo_preprocessing_and_anonymization_with_classifier.py run --config "$CONFIG" "$@"
uv run python run_echo_preprocessing_and_anonymization_with_classifier.py merge --config "$CONFIG"
