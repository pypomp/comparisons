#!/bin/bash
#SBATCH --account=ionides3
#SBATCH --partition=spgpu
#SBATCH --gpus=a40:1
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --job-name=dynamax-dacca-formal
#SBATCH --time=00:59:30
#SBATCH --array=0-11%3
#SBATCH --signal=B:USR1@120
#SBATCH --output=/scratch/ionides_root/ionides1/hkunyang/model024_20261006/competitor_estimation_20261028/dynamax/dacca/formal_fixed_v1/logs/formal-%A_%a.log
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
test -f gate_results/PASSED.json
python -c 'import json,hashlib; from pathlib import Path; r=Path("."); p=json.load(open("gate_results/PASSED.json")); assert p["passed"] and p["source_manifest_sha256"]==hashlib.sha256((r/"source_manifest.json").read_bytes()).hexdigest()'
case $((SLURM_ARRAY_TASK_ID % 3)) in
  0) METHODS=(IF2 IFAD_097 SGD_alpha1);;
  1) METHODS=(IFAD_097 SGD_alpha1 IF2);;
  2) METHODS=(SGD_alpha1 IF2 IFAD_097);;
esac
TASK_STOP=0
trap 'TASK_STOP=1; if test -n "${WORKER_PID:-}"; then kill -USR1 "$WORKER_PID" 2>/dev/null || true; fi' USR1 TERM
for METHOD in "${METHODS[@]}"; do
  python worker.py --config config.json --method "$METHOD" --batch-index "$SLURM_ARRAY_TASK_ID" &
  WORKER_PID=$!
  wait "$WORKER_PID" || { RC=$?; if test "$TASK_STOP" -eq 1; then wait "$WORKER_PID" || true; exit 0; else exit "$RC"; fi; }
  if test "$TASK_STOP" -eq 1; then exit 0; fi
done
python verify_manifest.py
