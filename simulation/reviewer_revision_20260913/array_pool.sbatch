#!/usr/bin/env bash
set -euo pipefail
run_dir=$1
workers=$2
study=$3
if [[ "$study" == main ]]; then
  exec bash "$run_dir/chunk-pool.sbatch" "$run_dir" "$workers" "$SLURM_ARRAY_TASK_ID" 24 main
fi
export SMART_SOURCE_ROOT="$run_dir/source"
export VENV=/home1/mkolar/envs/smart
export TMPDIR="$run_dir/tmp/${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"
source "$SMART_SOURCE_ROOT/hpc/discovery/env.sh"
export PYTHONPATH="$SMART_SOURCE_ROOT/simulation:$SMART_SOURCE_ROOT/sparse-smart-v2/src:$PYTHONPATH"
exec "$VENV/bin/python" -m reviewer_revision_20260913.rank_study run "$run_dir" "$SLURM_ARRAY_TASK_ID"
