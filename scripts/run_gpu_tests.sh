#!/bin/bash
# Run the MetVane test-suite (incl. CUDA tests) on one GPU of the IAIM partition.
#   sbatch scripts/run_gpu_tests.sh            (from the repository root)
#SBATCH -J metvane_gpu_tests
#SBATCH -p IAIM
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -t 00:30:00
#SBATCH -o metvane_gpu_tests_%j.log
set -euo pipefail
PY=${METVANE_PY:-/public/home/liangzm/tools/miniconda3/envs/weatherai/bin/python}
unset PYTHONPATH
export PYTHONNOUSERSITE=1
cd "${SLURM_SUBMIT_DIR:-$(dirname "$0")/..}"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv
PYTHONPATH=src "$PY" -m pytest -p no:cacheprovider -W error::RuntimeWarning -rs tests
