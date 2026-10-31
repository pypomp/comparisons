#!/bin/bash
#SBATCH --account=ionides3
#SBATCH --partition=spgpu
#SBATCH --gpus=a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --signal=B:TERM@120
#SBATCH --output=/scratch/ionides_root/ionides1/hkunyang/model024_20261006/implementation_comparison_20261030/pydpf/formal_v1/logs/%x-%j.log
set -euo pipefail
TASK=/scratch/ionides_root/ionides1/hkunyang/model024_20261006/implementation_comparison_20261030/pydpf/formal_v1
PHASE=${PYDPF_PHASE:?missing reviewed phase}
module purge
module load gcc cuda/12.8.1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONDONTWRITEBYTECODE=1
unset LD_PRELOAD
unset LD_LIBRARY_PATH
mkdir -p "$TASK/results/$PHASE/jax_cache" "$TASK/results/$PHASE/tmp"
export JAX_COMPILATION_CACHE_DIR="$TASK/results/$PHASE/jax_cache"
export TMPDIR="$TASK/results/$PHASE/tmp"
cd "$TASK"
nvidia-smi
CHILD_PID=""
stop_phase() {
  if test -n "$CHILD_PID"; then kill -TERM "$CHILD_PID" 2>/dev/null || true; wait "$CHILD_PID" || true; fi
  exit 75
}
trap stop_phase TERM INT
python3 run_phase.py --phase "$PHASE" &
CHILD_PID=$!
wait "$CHILD_PID"
