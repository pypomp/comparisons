#!/bin/bash
#SBATCH --job-name=pydpf-api-validation
#SBATCH --account=ionides3
#SBATCH --partition=spgpu
#SBATCH --gpus=a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=00:55:00
#SBATCH --output=/scratch/ionides_root/ionides1/hkunyang/model024_20261006/implementation_comparison_20261030/pydpf/repair_v2/logs/preflight-%j.log
set -euo pipefail
BASE=/scratch/ionides_root/ionides1/hkunyang/model024_20261006
TASK="$BASE/implementation_comparison_20261030/pydpf/repair_v2"
RUNTIME="$BASE/competitor_estimation_20261028/pydpf"
module purge
module load gcc cuda/12.8.1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTEST_DISABLE_PLUGIN_AUTOLOAD=1
export PYTHONDONTWRITEBYTECODE=1
unset LD_PRELOAD
unset LD_LIBRARY_PATH
export PYTHONPATH="$TASK/candidate/pydpf:$TASK/test_deps:$RUNTIME/runtime-deps"
cd "$TASK"
"$RUNTIME/venv/bin/python" verify_manifest.py
nvidia-smi
set +e
"$RUNTIME/venv/bin/python" run_preflight.py
TASK_EXIT=$?
set -e
"$RUNTIME/venv/bin/python" verify_manifest.py
exit "$TASK_EXIT"
