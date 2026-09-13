#!/usr/bin/env bash
set -euo pipefail
run_dir=$1
workers=$2
export SMART_SOURCE_ROOT="$run_dir/source"
export VENV=/home1/mkolar/envs/smart
export TMPDIR="$run_dir/tmp/${SLURM_JOB_ID}"
mkdir -p "$TMPDIR" "$run_dir/logs"
source "$SMART_SOURCE_ROOT/hpc/discovery/env.sh"
module load parallel/20240522
module load r/4.4.3
export R_LIBS_USER="$VENV/R/library"
export PYTHONPATH="$SMART_SOURCE_ROOT/simulation:$SMART_SOURCE_ROOT/sparse-smart-v2/src:$PYTHONPATH"
cd "$run_dir"
parallel --plain --jobs "$workers" --delay 0.05 --halt never \
  --joblog "$run_dir/logs/parallel-${SLURM_JOB_ID}.tsv" \
  srun --exclusive --exact -N1 -n1 -c1 --mem=4G \
  --output "$run_dir/logs/task-{1}-%j.out" \
  --error "$run_dir/logs/task-{1}-%j.err" \
  "$VENV/bin/python" -m reviewer_revision_20260913.runner run "$run_dir" '{1}' \
  :::: "$run_dir/work-items.txt"
