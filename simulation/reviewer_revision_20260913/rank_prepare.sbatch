#!/usr/bin/env bash
set -euo pipefail
run_dir=$1
export SMART_SOURCE_ROOT="$run_dir/source"
export VENV=/home1/mkolar/envs/smart
export TMPDIR="$run_dir/tmp/${SLURM_JOB_ID}"
mkdir -p "$TMPDIR" "$run_dir/smoke"
source "$SMART_SOURCE_ROOT/hpc/discovery/env.sh"
export PYTHONPATH="$SMART_SOURCE_ROOT/simulation:$SMART_SOURCE_ROOT/sparse-smart-v2/src:$PYTHONPATH"
env -u SLURM_JOB_ID "$VENV/bin/python" -m pytest "$SMART_SOURCE_ROOT/simulation/reviewer_revision_20260913/tests/test_rank_study.py" "$SMART_SOURCE_ROOT/simulation/reviewer_revision_20260913/tests/test_rank_audit.py" -q
"$VENV/bin/python" -m reviewer_revision_20260913.rank_study plan "$run_dir/smoke" --seeds 0 --ranks 3,7 --source-ranks 10,15 --iterations 2 --case-ids reference_exact_n80
"$VENV/bin/python" -m reviewer_revision_20260913.rank_study run "$run_dir/smoke" 0
"$VENV/bin/python" -m reviewer_revision_20260913.rank_audit "$run_dir/smoke"
"$VENV/bin/python" -m reviewer_revision_20260913.rank_study plan "$run_dir" --seeds 1000,1001,1002,1003,1004,1005,1006,1007,1008,1009,1010,1011,1012,1013,1014,1015,1016,1017,1018,1019,1020,1021,1022,1023,1024,1025,1026,1027,1028,1029 --iterations 500
