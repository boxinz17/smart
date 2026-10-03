#!/bin/bash
# Fingerprint diagnostic for park-grid-20261002-v1: native BLAS dispatch versus
# the Discovery convention (Haswell OpenBLAS kernels, NumPy optional SIMD off).
#
#SBATCH --account=mladen
#SBATCH --qos=normal
#SBATCH --partition=debug
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --open-mode=append

set -uo pipefail
echo "[$(date -Is)] job=$SLURM_JOB_ID node=$(hostname)"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PATH=/shared/home/mladen.kolar/miniforge3/envs/smart-singlecell/bin:$PATH
cd /shared/home/mladen.kolar/SMART_BoxinJinchi
RUN=runs/park-grid-20261002-v1
ARGS="--plan $RUN/plan.json --reference $RUN/per_replication.csv --tasks 900,901,1080,2190"
echo "== OPENBLAS_CORETYPE=Haswell only"
OPENBLAS_CORETYPE=Haswell python code/simulation/reviewer_revision_20260913/park_grid_fingerprint_check.py $ARGS
echo "== NPY_DISABLE_CPU_FEATURES only"
NPY_DISABLE_CPU_FEATURES=X86_V3,X86_V4,AVX512_ICL,AVX512_SPR python code/simulation/reviewer_revision_20260913/park_grid_fingerprint_check.py $ARGS
echo "== Haswell + NPY_DISABLE_CPU_FEATURES=X86_V4,AVX512_ICL,AVX512_SPR"
OPENBLAS_CORETYPE=Haswell NPY_DISABLE_CPU_FEATURES=X86_V4,AVX512_ICL,AVX512_SPR python code/simulation/reviewer_revision_20260913/park_grid_fingerprint_check.py $ARGS
echo "[$(date -Is)] done"
