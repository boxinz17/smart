#!/bin/bash
# Preprocessing exploration: every unfinished (variant, fold) pair as a single-threaded
# worker, one per physical core. Completed tasks are skipped, so a requeue or a rerun
# fits only what is missing. Submit from the project directory on Titan:
#
#   sbatch --parsable --job-name=RUNTAG --comment=RUNTAG \
#       --export=ALL,RUNTAG=RUNTAG,MODE=production code/single_cell_v2/titan/run_explore.sbatch
#
# Smoke canary: add TASKS=variant:fold+variant:fold and MODE=smoke ("+", since --export splits on commas) to --export, and pass
# --partition=debug --cpus-per-task=2 --mem=4G --time=00:15:00.
#
#SBATCH --account=mladen
#SBATCH --qos=normal
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=64
#SBATCH --mem=96G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --open-mode=append

set -euo pipefail
echo "[$(date -Is)] job=$SLURM_JOB_ID restart=${SLURM_RESTART_COUNT:-0} node=$(hostname) run=$RUNTAG mode=$MODE"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PATH=/shared/home/mladen.kolar/miniforge3/envs/smart-singlecell/bin:$PATH
cd /shared/home/mladen.kolar/SMART_BoxinJinchi

COUNTS=data/single_cell/nk_ilc1_counts.npz
ISOTYPES=data/single_cell/nk_ilc1_counts_isotypes.npz
FOLDS=data/single_cell/folds.json
OUT=runs/$RUNTAG
mkdir -p "$OUT/logs"
WORKERS=$((SLURM_CPUS_PER_TASK / 2))
STEP_MEM=$(( (SLURM_MEM_PER_NODE - 256) / WORKERS ))M
PENDING="$OUT/pending-$SLURM_JOB_ID.txt"   # per job: several jobs may share one run

if [ -n "${TASKS:-}" ]; then
    echo "$TASKS" | tr '+' '\n' | tr ':' ' ' > "$PENDING"
else
    # VARIANTS=a+b+c limits this job to those variants ("+", since --export splits on commas).
    python code/single_cell_v2/explore_preprocessing.py pending --folds "$FOLDS" --out "$OUT" \
        ${VARIANTS:+--variants "$(echo "$VARIANTS" | tr '+' ',')"} > "$PENDING"
fi
echo "pending=$(wc -l < "$PENDING") workers=$WORKERS step_mem=$STEP_MEM variants=${VARIANTS:-all}"

export COUNTS ISOTYPES FOLDS OUT MODE STEP_MEM
# One step per task on its own core; a failed task makes xargs exit 123 (job FAILED)
# while every other task's outputs are kept.
xargs -P "$WORKERS" -L1 bash -c \
    'srun --exact -n1 -c2 --mem="$STEP_MEM" python code/single_cell_v2/explore_preprocessing.py task \
        --counts "$COUNTS" --isotypes "$ISOTYPES" --folds "$FOLDS" --out "$OUT" \
        --variant "$1" --fold "$2" --mode "$MODE" >> "$OUT/logs/$1-$2.log" 2>&1' _ < "$PENDING"
echo "[$(date -Is)] done"
