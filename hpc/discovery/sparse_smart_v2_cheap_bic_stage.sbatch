#!/usr/bin/env bash
set -euo pipefail
[[ $# == 2 && -n ${SLURM_JOB_ID:-} && ${SLURM_NTASKS:-} == 1 && ${SLURM_CPUS_PER_TASK:-1} == 1 ]] || exit 2
run_dir=$(cd -- "$1" && pwd -P)
stage=$2
[[ "$stage" == prepare || "$stage" == develop || "$stage" == assess ]] || exit 2
attempt_dir="$run_dir/campaign/attempts/$stage-attempt-0001"
[[ -s "$attempt_dir/submission.json" ]] || exit 2
mkdir -p "$attempt_dir/tmp" "$attempt_dir/matplotlib"
finish() { status=$?; trap - EXIT; printf '%s\n' "$status" > "$attempt_dir/process-exit-code.txt"; exit "$status"; }
trap finish EXIT
trap 'exit 143' TERM
export TMPDIR="$attempt_dir/tmp" MPLCONFIGDIR="$attempt_dir/matplotlib"
export VENV=${VENV:-/home1/mkolar/envs/smart} SMART_SOURCE_ROOT="$run_dir/source"
source "$SMART_SOURCE_ROOT/hpc/discovery/env.sh"
export PYTHONPATH="$SMART_SOURCE_ROOT/sparse-smart-v2/src:$SMART_SOURCE_ROOT/simulation:$PYTHONPATH"
cd "$run_dir"
if [[ "$stage" == prepare ]]; then
    "$VENV/bin/python" -m pytest -q --import-mode=importlib -p no:cacheprovider \
        --basetemp="$attempt_dir/tmp/pytest" \
        "$SMART_SOURCE_ROOT/simulation/tests/test_sparse_smart_v2_cheap_bic_plan.py" \
        "$SMART_SOURCE_ROOT/simulation/tests/test_sparse_smart_v2_bic_runner.py" \
        "$SMART_SOURCE_ROOT/simulation/tests/test_sparse_smart_v2_cheap_bic_operations.py" \
        "$SMART_SOURCE_ROOT/simulation/tests/test_summarize_sparse_smart_v2_cheap_bic.py"
fi
"$VENV/bin/python" "$SMART_SOURCE_ROOT/simulation/run_sparse_smart_v2_cheap_bic.py" "$stage" --root "$run_dir"
