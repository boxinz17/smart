#!/bin/bash
# Reproducibility probe for park-grid-20261002-v1: Discovery BLAS convention on Titan.
#SBATCH --account=mladen
#SBATCH --qos=normal
#SBATCH --partition=debug
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:05:00
#SBATCH --output=logs/%x-%j.out
set -u
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PATH=/shared/home/mladen.kolar/miniforge3/envs/smart-singlecell/bin:$PATH
cd /shared/home/mladen.kolar/SMART_BoxinJinchi
RUN=runs/park-grid-20261002-v1
CHK="code/simulation/reviewer_revision_20260913/park_grid_fingerprint_check.py --plan $RUN/plan.json --reference $RUN/per_replication.csv --tasks 900,901,1080,2190"
export OPENBLAS_CORETYPE=Haswell NPY_DISABLE_CPU_FEATURES=X86_V3,X86_V4,AVX512_ICL,AVX512_SPR
for attempt in 1 2 3; do
  echo "== attempt $attempt (plain)"; python $CHK 2>&1 | grep -v "cpu features" | tail -8; echo "exit=${PIPESTATUS[0]}"
done
echo "== with faulthandler"; python -X faulthandler $CHK 2>&1 | grep -v "cpu features" | tail -30
