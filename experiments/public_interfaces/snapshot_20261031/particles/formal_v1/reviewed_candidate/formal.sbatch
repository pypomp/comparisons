#!/bin/bash
#SBATCH --job-name=particles-newIF2-fit
#SBATCH --account=ionides3
#SBATCH --partition=standard
#SBATCH --array=0-23%6
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:45:00
#SBATCH --signal=B:TERM@60
#SBATCH --output=/scratch/ionides_root/ionides1/hkunyang/model024_20261006/implementation_comparison_20261030/particles/logs/formal-%A_%a.log
set -euo pipefail
BASE=/scratch/ionides_root/ionides1/hkunyang/model024_20261006
TASK="$BASE/implementation_comparison_20261030/particles"
module purge
module load gcc
source "$BASE/model024_panel_cute_20261023/venv/bin/activate"
export PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" JAX_PLATFORMS=cpu
export PYTHONPATH="$TASK/source/candidate:$BASE/competitor_estimation_20261028/particles/local-deps:$BASE/cuthbert_dacca_20261027/deps"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export NUMBA_CACHE_DIR="$TASK/formal_v1/cache/$SLURM_ARRAY_TASK_ID/numba"
mkdir -p "$NUMBA_CACHE_DIR"
cd "$TASK/formal_v1"
python verify.py
exec python fit.py --task-id "$SLURM_ARRAY_TASK_ID"
