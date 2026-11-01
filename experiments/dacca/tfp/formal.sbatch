#!/bin/bash
#SBATCH --job-name=tfp-public-fit
#SBATCH --account=ionides3
#SBATCH --partition=spgpu
#SBATCH --gpus=a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --no-requeue
#SBATCH --output=/scratch/ionides_root/ionides1/hkunyang/model024_20261006/implementation_comparison_20261030/tfp/formal_v2/logs/formal-%j.log
set -euo pipefail
METHOD=${1:?method required}
TASK=/scratch/ionides_root/ionides1/hkunyang/model024_20261006/implementation_comparison_20261030/tfp/formal_v2
TFP_VENV=/scratch/ionides_root/ionides1/hkunyang/model024_20261006/competitor_estimation_20261028/tfp/venv
REFERENCE_PYTHON=/scratch/ionides_root/ionides1/hkunyang/model024_20261006/model024_panel_cute_20261023/venv/bin/python
module purge
module load gcc cuda/12.8.1
unset PYTHONPATH LD_LIBRARY_PATH LD_PRELOAD JAX_PLATFORMS
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 TF_CPP_MIN_LOG_LEVEL=1
cd "$TASK"
nvidia-smi
"$TFP_VENV/bin/python" runtime_guard.py --method "$METHOD"
"$TFP_VENV/bin/python" run.py --method "$METHOD"
if test "$METHOD" != prefix; then
 export PYTHONPATH=/scratch/ionides_root/ionides1/hkunyang/model024_20261006/cuthbert_dacca_paper_20261028/source:/scratch/ionides_root/ionides1/hkunyang/model024_20261006/cuthbert_dacca_20261027/source:/scratch/ionides_root/ionides1/hkunyang/model024_20261006/cuthbert_dacca_20261027/deps
 export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/targets/x86_64-linux/lib"
 export LD_PRELOAD="$CUDA_HOME/lib64/libcusparse.so.12.5.8.93"
 export JAX_PLATFORMS=cuda JAX_ENABLE_X64=False
 "$REFERENCE_PYTHON" score.py --endpoints "results/$METHOD/endpoints.json" --output "results/$METHOD/independent_scores" --seed 94881000
fi
"$TFP_VENV/bin/python" terminal_manifest.py --method "$METHOD"
