#!/bin/bash
#SBATCH --job-name=dacca-pypfilt-formal
#SBATCH --account=ionides3
#SBATCH --partition=standard
#SBATCH --array=0-11%3
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=00:50:00
#SBATCH --signal=B:TERM@60
#SBATCH --output=/scratch/ionides_root/ionides1/hkunyang/model024_20261006/competitor_estimation_20261028/pypfilt/formal_v1/logs/formal-%A_%a.log
set -euo pipefail
export TASK_ALLOCATION_START=$(date +%s)
BASE=/scratch/ionides_root/ionides1/hkunyang/model024_20261006
TASK="$BASE/competitor_estimation_20261028/pypfilt/formal_v1"
module purge
module load gcc
source "$BASE/model024_panel_cute_20261023/venv/bin/activate"
export CUDA_VISIBLE_DEVICES=""
export JAX_PLATFORMS=cpu
export PYTHONPATH="$TASK/source:$TASK/../deps-linux:$BASE/cuthbert_dacca_paper_20261028/source:$BASE/cuthbert_dacca_20261027/deps"
export PFB_REFERENCE_SOURCE="$BASE/cuthbert_dacca_paper_20261028/source"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1
export TF_NUM_INTEROP_THREADS=1
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false"
export JAX_COMPILATION_CACHE_DIR="$TASK/results/tasks/$SLURM_ARRAY_TASK_ID/jax_cache"
export PFB_DACCA_OBS_DIR="$TASK/results/tasks/$SLURM_ARRAY_TASK_ID/runtime"
export TMPDIR="$TASK/results/tasks/$SLURM_ARRAY_TASK_ID/tmp"
mkdir -p "$JAX_COMPILATION_CACHE_DIR" "$PFB_DACCA_OBS_DIR" "$TMPDIR"
cd "$TASK"
python verify_manifest.py
exec python fit.py --protocol protocol.json --task-id "$SLURM_ARRAY_TASK_ID" --output results --allocation-seconds 3000
