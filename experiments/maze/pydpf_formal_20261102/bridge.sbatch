#!/bin/bash
#SBATCH --account=ionides3
#SBATCH --partition=spgpu
#SBATCH --gpus=a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=00:20:00
#SBATCH --no-requeue
#SBATCH --output=bridge-%j.log
set -euo pipefail
module purge
module load gcc/13.2.0 cuda/12.8.1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
unset LD_LIBRARY_PATH LD_PRELOAD
TORCH_ROOT=/scratch/ionides_root/ionides1/hkunyang/model024_20261006/competitor_estimation_20261028/pydpf
export RUNTIME_DEPS="$TORCH_ROOT/runtime-deps"
export PYTHONPATH="$PWD/source/upstream:$RUNTIME_DEPS"
python3 verify_source.py
nvidia-smi
"$TORCH_ROOT/venv/bin/python" launch.py --stage bridge
