#!/bin/bash
#SBATCH --job-name=particles-newIF2-score
#SBATCH --account=ionides3
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/ionides_root/ionides1/hkunyang/model024_20261006/implementation_comparison_20261030/particles/logs/score-%j.log
set -euo pipefail
BASE=/scratch/ionides_root/ionides1/hkunyang/model024_20261006
TASK="$BASE/implementation_comparison_20261030/particles"
module purge
module load gcc
source "$BASE/model024_panel_cute_20261023/venv/bin/activate"
export PYTHONDONTWRITEBYTECODE=1 JAX_PLATFORMS=cuda
export PYTHONPATH="$TASK/source/reference:$BASE/cuthbert_dacca_paper_20261028/source:$BASE/cuthbert_dacca_20261027/deps"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
cd "$TASK/formal_v1"
python verify.py
python assemble_endpoints.py
exec python score.py --endpoints endpoints.json --output scoring --seed 94981000
