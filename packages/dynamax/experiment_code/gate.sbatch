#!/bin/bash
#SBATCH --account=ionides3
#SBATCH --partition=spgpu
#SBATCH --gpus=a40:1
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --job-name=dynamax-dacca-scalar-gate
#SBATCH --time=00:05:00
#SBATCH --output=/scratch/ionides_root/ionides1/hkunyang/model024_20261006/competitor_estimation_20261028/dynamax/dacca/formal_fixed_v1/logs/gate-%j.log
set -euo pipefail
TASK=/scratch/ionides_root/ionides1/hkunyang/model024_20261006/competitor_estimation_20261028/dynamax/dacca/formal_fixed_v1
BASE=/scratch/ionides_root/ionides1/hkunyang/model024_20261006/competitor_estimation_20261028/dynamax
OLD=/scratch/ionides_root/ionides1/hkunyang/model024_20261006/cuthbert_dacca_20261027
VENV=/scratch/ionides_root/ionides1/hkunyang/model024_20261006/model024_panel_cute_20261023/venv
module purge
module load gcc cuda/12.8.1
source "$VENV/bin/activate"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/targets/x86_64-linux/lib:${LD_LIBRARY_PATH:-}"
export LD_PRELOAD="$CUDA_HOME/lib64/libcusparse.so.12.5.8.93"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS=cuda
export PYTHONPATH="$TASK/source:$BASE/deps:$OLD/deps"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export JAX_COMPILATION_CACHE_DIR="$TASK/jax_cache"
cd "$TASK"
python verify_manifest.py
nvidia-smi
python gate.py
python verify_manifest.py
