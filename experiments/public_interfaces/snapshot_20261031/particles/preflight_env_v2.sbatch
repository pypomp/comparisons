#!/bin/bash
#SBATCH --job-name=particles-newIF2-check
#SBATCH --account=ionides3
#SBATCH --partition=standard
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:29:00
#SBATCH --output=/scratch/ionides_root/ionides1/hkunyang/model024_20261006/implementation_comparison_20261030/particles/logs/preflight-%j.log
set -euo pipefail
BASE=/scratch/ionides_root/ionides1/hkunyang/model024_20261006
TASK="$BASE/implementation_comparison_20261030/particles"
module purge
module load gcc
source "$BASE/model024_panel_cute_20261023/venv/bin/activate"
export PYTHONDONTWRITEBYTECODE=1
export CUDA_VISIBLE_DEVICES=""
export JAX_PLATFORMS=cpu
export PARTICLES_RUNTIME_DEPS="$BASE/competitor_estimation_20261028/particles/local-deps:$BASE/cuthbert_dacca_20261027/deps"
export PYTHONPATH="$TASK/source/candidate:$PARTICLES_RUNTIME_DEPS"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export NUMBA_CACHE_DIR="$TASK/results/preflight/numba_cache"
mkdir -p "$NUMBA_CACHE_DIR"
cd "$TASK"
python verify_manifest.py
exec python preflight.py
