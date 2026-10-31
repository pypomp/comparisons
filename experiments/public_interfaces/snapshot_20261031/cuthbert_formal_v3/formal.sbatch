#!/bin/bash
#SBATCH --job-name=cuthbert-public-ABC-v3
#SBATCH --account=ionides3
#SBATCH --partition=spgpu
#SBATCH --gpus=a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=00:56:00
#SBATCH --array=0-11%3
#SBATCH --output=logs/%x-%A_%a.out
#SBATCH --error=logs/%x-%A_%a.err
set -euo pipefail
module purge
module load gcc cuda/12.8.1
source /scratch/ionides_root/ionides1/hkunyang/model024_20261006/model024_panel_cute_20261023/venv/bin/activate
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="$CUDA_HOME/lib64/libcusparse.so.12.5.8.93"
export XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda JAX_ENABLE_X64=False
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
nvidia-smi
python -u execute_case.py "$1"
