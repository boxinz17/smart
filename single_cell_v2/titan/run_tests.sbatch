#!/bin/bash
# Synthetic test suite for code/single_cell_v2 (folds, transforms, leakage, end-to-end runs).
#
#   sbatch --parsable code/single_cell_v2/titan/run_tests.sbatch
#
#SBATCH --job-name=single-cell-v2-tests
#SBATCH --comment=single-cell-v2-tests
#SBATCH --account=mladen
#SBATCH --qos=normal
#SBATCH --partition=debug
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --open-mode=append

set -euo pipefail
echo "[$(date -Is)] job=$SLURM_JOB_ID node=$(hostname)"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1
export PATH=/shared/home/mladen.kolar/miniforge3/envs/smart-singlecell/bin:$PATH
cd /shared/home/mladen.kolar/SMART_BoxinJinchi/code/single_cell_v2
python -m pytest -q -p no:cacheprovider tests
echo "[$(date -Is)] done"
