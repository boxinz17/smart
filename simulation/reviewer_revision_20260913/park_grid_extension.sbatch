#!/bin/bash
# Park et al. comparator with a wider pooled-penalty grid (audit finding M-9).
# One single-threaded worker per physical core; each srun step processes one
# chunk of tasks. Completed tasks are skipped, so a requeue redoes only missing
# work. Submit from /shared/home/mladen.kolar/SMART_BoxinJinchi:
#
#   sbatch --parsable --job-name=RUNTAG --comment=RUNTAG \
#       --export=ALL,RUNTAG=RUNTAG code/simulation/reviewer_revision_20260913/park_grid_extension.sbatch
#
# Canary: add CHUNKS=<one chunk, tasks joined by +> to --export and pass
# --partition=debug --cpus-per-task=2 --mem=4G --time=00:15:00.
#
#SBATCH --account=mladen
#SBATCH --qos=normal
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=6
#SBATCH --mem=8G
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x-%j.out
#SBATCH --open-mode=append

set -euo pipefail
echo "[$(date -Is)] job=$SLURM_JOB_ID restart=${SLURM_RESTART_COUNT:-0} node=$(hostname) run=$RUNTAG"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PATH=/shared/home/mladen.kolar/miniforge3/envs/smart-singlecell/bin:$PATH
cd /shared/home/mladen.kolar/SMART_BoxinJinchi

RUN=runs/$RUNTAG
OUT=$RUN/tasks
mkdir -p "$OUT" "$RUN/logs"
WORKERS=$((SLURM_CPUS_PER_TASK / 2))
STEP_MEM=$(( (SLURM_MEM_PER_NODE - 256) / WORKERS ))M
PENDING="$RUN/pending-$SLURM_JOB_ID.txt"   # per job: never shared between jobs

if [ -n "${CHUNKS:-}" ]; then
    echo "$CHUNKS" > "$PENDING"
else
    # Raw-source (fitted) tasks without a completed result, in chunks of 10.
    python - "$RUN/plan.json" "$OUT" <<'EOF' > "$PENDING"
import json, sys
from pathlib import Path
plan = json.loads(Path(sys.argv[1]).read_text())
out = Path(sys.argv[2])
todo = []
for i, t in enumerate(plan["tasks"]):
    if plan["cases"][t["case_index"]].get("source_mode") != "fitted":
        continue
    f = out / f"task-{i:06d}.json"
    if f.exists() and json.loads(f.read_text()).get("complete"):
        continue
    todo.append(i)
for k in range(0, len(todo), 10):
    print("+".join(str(i) for i in todo[k:k + 10]))
EOF
fi
echo "workers=$WORKERS step_mem=$STEP_MEM chunks=$(wc -l < "$PENDING")"

export RUN OUT STEP_MEM
# A failed chunk makes xargs exit 123 (job FAILED); every completed task is kept.
xargs -P "$WORKERS" -I{} bash -c \
    'srun --exact -n1 -c2 --mem="$STEP_MEM" python code/simulation/reviewer_revision_20260913/park_grid_extension.py \
        --plan "$RUN/plan.json" --reference "$RUN/per_replication.csv" --out "$OUT" --tasks "$1" \
        >> "$RUN/logs/chunk-$(echo "$1" | cut -d+ -f1).log" 2>&1' _ {} < "$PENDING"
echo "[$(date -Is)] done"
