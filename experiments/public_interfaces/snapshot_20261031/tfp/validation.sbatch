#!/bin/bash
#SBATCH --job-name=tfp-public-gate
#SBATCH --account=ionides3
#SBATCH --partition=spgpu
#SBATCH --gpus=a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/ionides_root/ionides1/hkunyang/model024_20261006/implementation_comparison_20261030/tfp/logs/validation-%j.log
set -euo pipefail
TASK=/scratch/ionides_root/ionides1/hkunyang/model024_20261006/implementation_comparison_20261030/tfp
TFP_VENV=/scratch/ionides_root/ionides1/hkunyang/model024_20261006/competitor_estimation_20261028/tfp/venv
module purge
module load gcc cuda/12.8.1
unset PYTHONPATH LD_LIBRARY_PATH LD_PRELOAD JAX_PLATFORMS
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 TF_CPP_MIN_LOG_LEVEL=1
cd "$TASK"
nvidia-smi
"$TFP_VENV/bin/python" gpu_validation.py
