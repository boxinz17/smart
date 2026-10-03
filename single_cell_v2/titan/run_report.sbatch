#!/bin/bash
# Regenerate the application tables and figures from a completed run (reporting
# only: no fitting). Submit from /shared/home/mladen.kolar/SMART_BoxinJinchi:
#
#   sbatch --parsable --job-name=TAG --comment=TAG \
#       --export=ALL,RUN=runs/sc-lodo-clr-20261002-v1,WITHIN=runs/sc-lodo-20261001-v1,REPORT=report-smart \
#       code/single_cell_v2/titan/run_report.sbatch
#
#SBATCH --account=mladen
#SBATCH --qos=normal
#SBATCH --partition=debug
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:15:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --open-mode=append

set -euo pipefail
echo "[$(date -Is)] job=$SLURM_JOB_ID restart=${SLURM_RESTART_COUNT:-0} node=$(hostname) run=$RUN report=$REPORT"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PATH=/shared/home/mladen.kolar/miniforge3/envs/smart-singlecell/bin:$PATH
cd /shared/home/mladen.kolar/SMART_BoxinJinchi
python code/single_cell_v2/report_application.py --run "$RUN" --within-cell "$WITHIN" \
    --folds data/single_cell/folds.json --out "$RUN/$REPORT"
echo "[$(date -Is)] done"
