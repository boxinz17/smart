#!/usr/bin/env bash
set -euo pipefail
run_dir=$1
export SMART_SOURCE_ROOT="$run_dir/source"
export VENV=/home1/mkolar/envs/smart
export TMPDIR="$run_dir/tmp/${SLURM_JOB_ID}"
mkdir -p "$TMPDIR" "$run_dir/smoke"
source "$SMART_SOURCE_ROOT/hpc/discovery/env.sh"
export PYTHONPATH="$SMART_SOURCE_ROOT/simulation:$SMART_SOURCE_ROOT/sparse-smart-v2/src:$PYTHONPATH"
env -u SLURM_JOB_ID "$VENV/bin/python" -m pytest "$SMART_SOURCE_ROOT/simulation/reviewer_revision_20260913/tests/test_budget_study.py" --basetemp="/tmp/reviewer-budget-tests-${SLURM_JOB_ID}" -q
"$VENV/bin/python" -m reviewer_revision_20260913.budget_study plan "$run_dir/smoke" --seeds 2000 --budgets 100 --cases reference_exact_n80
"$VENV/bin/python" -m reviewer_revision_20260913.budget_study run "$run_dir/smoke" 0
"$VENV/bin/python" -m reviewer_revision_20260913.budget_study summarize "$run_dir/smoke"
"$VENV/bin/python" -m reviewer_revision_20260913.budget_study plan "$run_dir"
