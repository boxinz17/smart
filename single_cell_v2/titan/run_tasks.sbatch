#!/bin/bash
# All leave-one-donor-out tasks (folds x variants) as single-threaded workers,
# one per physical core. Completed tasks are skipped, so a requeue or a rerun
# fits only what is missing. Submit from the project directory on Titan:
#
#   sbatch --parsable --job-name=RUNTAG --comment=RUNTAG \
#       --export=ALL,RUNTAG=RUNTAG,MODE=production code/single_cell_v2/titan/run_tasks.sbatch
#
# Smoke canary (no test scoring): add TASKS=0,MODE=smoke to --export (several tasks: TASKS=0+5;
# --export splits on commas) and
# pass --partition=debug --cpus-per-task=2 --mem=4G --time=00:15:00.
#
#SBATCH --account=mladen
#SBATCH --qos=normal
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=32
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --open-mode=append

set -euo pipefail
echo "[$(date -Is)] job=$SLURM_JOB_ID restart=${SLURM_RESTART_COUNT:-0} node=$(hostname) run=$RUNTAG mode=$MODE"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PATH=/shared/home/mladen.kolar/miniforge3/envs/smart-singlecell/bin:$PATH
cd /shared/home/mladen.kolar/SMART_BoxinJinchi

COUNTS=data/single_cell/nk_ilc1_counts.npz
FOLDS=data/single_cell/folds.json
OUT=runs/$RUNTAG
mkdir -p "$OUT/logs"
N_TASKS=$(python -c "import sys; sys.path.insert(0, 'code/single_cell_v2'); \
from folds import load_folds, tasks; print(len(tasks(load_folds('$FOLDS'))))")
WORKERS=$((SLURM_CPUS_PER_TASK / 2))
STEP_MEM=$(( (SLURM_MEM_PER_NODE - 256) / WORKERS ))M
echo "tasks=$N_TASKS workers=$WORKERS step_mem=$STEP_MEM"
PENDING="$OUT/pending-$SLURM_JOB_ID.txt"   # per job: several jobs may share one run

if [ -n "${TASKS:-}" ]; then
    echo "$TASKS" | tr '+' '\n' > "$PENDING"
else
python - "$FOLDS" "$OUT" <<'EOF' > "$PENDING"
import sys
from pathlib import Path
sys.path.insert(0, "code/single_cell_v2")
from folds import load_folds, tasks
for index, task in enumerate(tasks(load_folds(sys.argv[1]))):
    if not (Path(sys.argv[2]) / task["variant"] / task["fold"] / "result.json").exists():
        print(index)
EOF
fi
echo "pending=$(wc -l < "$PENDING")"

export COUNTS FOLDS OUT MODE STEP_MEM
# One single-threaded step per task on its own core; a failed task makes xargs
# exit 123 (job FAILED) while every other task's outputs are kept.
xargs -P "$WORKERS" -I{} bash -c \
    'srun --exact -n1 -c2 --mem="$STEP_MEM" python code/single_cell_v2/run_fold.py \
        --counts "$COUNTS" --folds "$FOLDS" --out "$OUT" --task-index "$1" --mode "$MODE" \
        --threads 1 >> "$OUT/logs/task-$1.log" 2>&1' _ {} < "$PENDING"
echo "[$(date -Is)] done"
