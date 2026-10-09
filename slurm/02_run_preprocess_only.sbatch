#!/usr/bin/env bash
# Script 2 without the view classifiers (CPU only): preprocess and save every pending manifest record,
# then build the final table. A later 02_run_with_classifier.sbatch classifies these records.
# Submit from the repo root:  sbatch slurm/02_run_preprocess_only.sbatch [configs/config.yaml] [--retry-errors]
# Safe to resubmit: finished records are skipped. Don't run it at the same time as 02_run_with_classifier.sbatch.
#
# Each of the preprocess.num_cpu_procs processes starts preprocess.num_workers DataLoader workers, so give it
# num_cpu_procs * (num_workers + 1) CPUs or more (8 * (4 + 1) = 40 with the defaults).
#SBATCH --job-name=echo-anon-preprocess
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=40
#SBATCH --mem=128G
#SBATCH --time=48:00:00
##SBATCH --partition=<cpu-partition>
##SBATCH --account=<account>
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
CONFIG="${1:-configs/config.yaml}"
shift || true

# module load ...   # if uv is not already on PATH on the compute nodes

uv run python run_echo_preprocessing_and_anonymization.py --config "$CONFIG" "$@"
uv run python run_echo_preprocessing_and_anonymization_with_classifier.py merge --config "$CONFIG"
