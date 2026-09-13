#!/usr/bin/env bash
set -euo pipefail
run_dir=$1
workers=$2
export SMART_SOURCE_ROOT="$run_dir/source"
export VENV=/home1/mkolar/envs/smart
export TMPDIR="$run_dir/tmp/${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"
source "$SMART_SOURCE_ROOT/hpc/discovery/env.sh"
export PYTHONPATH="$SMART_SOURCE_ROOT/simulation:$SMART_SOURCE_ROOT/sparse-smart-v2/src:$PYTHONPATH"
"$VENV/bin/python" -m reviewer_revision_20260913.runner plan "$run_dir" --runtime --seeds 3000,3001,3002,3003,3004 --iterations 500
exec bash "$SMART_SOURCE_ROOT/simulation/reviewer_revision_20260913/pool.sbatch" "$run_dir" "$workers"
