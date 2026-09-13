#!/usr/bin/env bash
set -euo pipefail
run_dir=$1
study=${2:-main}
export SMART_SOURCE_ROOT="$run_dir/source"
export VENV=/home1/mkolar/envs/smart
export TMPDIR="$run_dir/tmp/${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"
source "$SMART_SOURCE_ROOT/hpc/discovery/env.sh"
export PYTHONPATH="$SMART_SOURCE_ROOT/simulation:$SMART_SOURCE_ROOT/sparse-smart-v2/src:$PYTHONPATH"
if [[ "$study" == rank ]]; then
  "$VENV/bin/python" -m reviewer_revision_20260913.rank_audit "$run_dir"
else
  "$VENV/bin/python" -m reviewer_revision_20260913.audit "$run_dir"
fi
