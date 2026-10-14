#!/usr/bin/env bash
# Preprocess MIMIC-IV-ECHO (already anonymized) on CPU: wraps scripts/run_preprocess_mimic_iv_echo.sh.
# Submit from the repo root:
#   mkdir -p slurm/logs
#   sbatch slurm/03_preprocess_mimic_iv_echo.sbatch [--retry-errors]
# With the view classifiers, request GPUs on the command line (it overrides the #SBATCH lines):
#   sbatch --gres=gpu:4 --partition=<gpu-partition> slurm/03_preprocess_mimic_iv_echo.sbatch --classify
# Locations default to MIMIC_ROOT=/scratch/shared/physionet.org/files/mimic-iv-echo/1.0.1,
# OUTPUT_ROOT=/scratch/$USER/mimic-iv-echo-preprocessed and WORK_DIR=/scratch/$USER/mimic-iv-echo-work. Override them
# (and CONFIG) as environment variables, which sbatch passes to the job:
#   OUTPUT_ROOT=/scratch/me/out sbatch slurm/03_preprocess_mimic_iv_echo.sbatch
# Safe to resubmit (e.g. after a timeout): the work manifest is reused and finished records are skipped.
#
# CPUs: header reading uses index.num_workers processes; preprocessing uses num_cpu_procs (or num_gpus with
# --classify) * (num_workers + 1). With configs/config_preprocess_anonymized.yaml that is 32 and 8 * (7 + 1) = 64.
#SBATCH --job-name=mimic-echo-preprocess
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH --time=48:00:00
##SBATCH --partition=<cpu-partition>
##SBATCH --account=<account>
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

# module load ...   # if uv is not already on PATH on the compute nodes

scripts/run_preprocess_mimic_iv_echo.sh "$@"
