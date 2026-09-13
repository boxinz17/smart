#!/usr/bin/env bash
set -euo pipefail
run_dir=$1
workers=$2
chunk_index=$3
chunk_count=$4
study=${5:-main}
export SMART_SOURCE_ROOT="$run_dir/source"
export VENV=/home1/mkolar/envs/smart
export TMPDIR="$run_dir/tmp/${SLURM_JOB_ID}"
mkdir -p "$TMPDIR" "$run_dir/logs"
source "$SMART_SOURCE_ROOT/hpc/discovery/env.sh"
module load parallel/20240522
module load r/4.4.3
export R_LIBS_USER="$VENV/R/library"
export PYTHONPATH="$SMART_SOURCE_ROOT/simulation:$SMART_SOURCE_ROOT/sparse-smart-v2/src:$PYTHONPATH"
if [[ "$study" == rank ]]; then
  module_name=reviewer_revision_20260913.rank_study
  plan_file="$run_dir/rank-plan.json"
else
  module_name=reviewer_revision_20260913.runner
  plan_file="$run_dir/plan.json"
fi
items="$run_dir/${study}-chunk-${chunk_index}.txt"
"$VENV/bin/python" - "$plan_file" "$items" "$chunk_index" "$chunk_count" <<'PY'
import json,pathlib,sys
plan=json.loads(pathlib.Path(sys.argv[1]).read_text())
index,count=map(int,sys.argv[3:])
assert 0<=index<count
path=pathlib.Path(sys.argv[2]);content=''.join(f'{i}\n' for i in range(len(plan['tasks'])) if i%count==index)
if path.exists(): assert path.read_text()==content
else: path.write_text(content)
PY
parallel --plain --jobs "$workers" --delay .05 --halt never \
  --joblog "$run_dir/logs/${study}-chunk-${chunk_index}-${SLURM_JOB_ID}.tsv" \
  srun --exclusive --exact -N1 -n1 -c1 --mem=4G \
  --output "$run_dir/logs/${study}-task-{1}-%j.out" \
  --error "$run_dir/logs/${study}-task-{1}-%j.err" \
  "$VENV/bin/python" -m "$module_name" run "$run_dir" '{1}' :::: "$items"
