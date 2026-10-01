#!/usr/bin/env bash
set -euo pipefail
[[ $# == 3 && "$3" =~ ^[1-9][0-9]*$ && -n ${SLURM_JOB_ID:-} ]] || exit 2
run_dir=$(cd -- "$1" && pwd -P)
mode=$2
workers=$3
[[ "$mode" == prepare || "$mode" == preflight || "$mode" == pool || "$mode" == summary ]] || exit 2
attempt_dir="$run_dir/campaign/attempts/$mode-attempt-0001"
[[ -s "$attempt_dir/submission.json" ]] || exit 2
mkdir -p "$attempt_dir/tmp" "$attempt_dir/matplotlib" "$run_dir/logs/cases"
export TMPDIR="$attempt_dir/tmp" MPLCONFIGDIR="$attempt_dir/matplotlib"
export VENV=${VENV:-/home1/mkolar/envs/smart} SMART_SOURCE_ROOT="$run_dir/source" SLURM_EXPORT_ENV=ALL
verified=0
verify() {
    "$VENV/bin/python" - "$run_dir" "$mode" "$workers" <<'PY'
import importlib.util,json,pathlib,sys
root=pathlib.Path(sys.argv[1]);kind=sys.argv[2];workers=int(sys.argv[3])
spec=importlib.util.spec_from_file_location('retained_ops',root/'source/hpc/discovery/submit_sparse_smart_v2_retained_state_bic.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
print(json.dumps(module.validate_submission(root,kind,workers),sort_keys=True))
PY
}
finish() {
    status=$?
    trap - EXIT
    if (( verified )); then
        set +e
        verify > "$attempt_dir/source-integrity-after.json" 2> "$attempt_dir/source-integrity-after.err"
        integrity=$?
        set -e
        printf '%s\n' "$integrity" > "$attempt_dir/source-integrity-exit-code.txt"
        if (( integrity != 0 )); then status=74; fi
    fi
    printf '%s\n' "$status" > "$attempt_dir/exit-code.txt"
    printf 'job_id\texit_code\tfinished_utc\n%s\t%s\t%s\n' \
        "$SLURM_JOB_ID" "$status" "$(date -u +%FT%TZ)" > "$attempt_dir/process-status.tsv"
    exit "$status"
}
trap finish EXIT
trap 'exit 143' TERM
verify > "$attempt_dir/source-integrity-before.json"
verified=1
source "$SMART_SOURCE_ROOT/hpc/discovery/env.sh"
export PYTHONPATH="$SMART_SOURCE_ROOT/sparse-smart-v2/src:$SMART_SOURCE_ROOT/simulation:${PYTHONPATH:-}"
cd "$run_dir"
runner="$SMART_SOURCE_ROOT/simulation/run_sparse_smart_v2_retained_state_bic.py"
if [[ "$mode" == prepare ]]; then
    "$VENV/bin/python" -m pytest -q --import-mode=importlib -p no:cacheprovider --basetemp="$attempt_dir/tmp/pytest" \
        "$SMART_SOURCE_ROOT/sparse-smart-v2/tests/test_bic_selection.py" \
        "$SMART_SOURCE_ROOT/simulation/tests/test_sparse_smart_v2_retained_state_scoring.py" \
        "$SMART_SOURCE_ROOT/simulation/tests/test_sparse_smart_v2_retained_state_bic.py" \
        "$SMART_SOURCE_ROOT/simulation/tests/test_summarize_sparse_smart_v2_retained_state_bic.py" \
        "$SMART_SOURCE_ROOT/simulation/tests/test_sparse_smart_v2_retained_state_bic_operations.py"
    "$VENV/bin/python" "$runner" --plan "$run_dir/plan.json" --prepare
elif [[ "$mode" == preflight || "$mode" == pool ]]; then
    module load "${SMART_PARALLEL_MODULE:-parallel/20240522}"
    parallel --plain --version > "$attempt_dir/parallel-version.txt"
    if [[ "$mode" == preflight ]]; then work_file=preflight-work-items.tsv; else work_file=production-work-items.tsv; fi
    parallel --plain --quote --jobs "$workers" --delay 0.1 --halt never \
        --joblog "$attempt_dir/parallel-joblog.tsv" \
        bash ./source/hpc/discovery/sparse_smart_v2_retained_state_bic_worker.sh dispatch . '{}' \
        :::: "$run_dir/$work_file"
    if [[ "$mode" == preflight ]]; then
        "$VENV/bin/python" - "$run_dir" <<'PY'
import importlib.util,json,pathlib,sys
root=pathlib.Path(sys.argv[1]);spec=importlib.util.spec_from_file_location('retained_ops',root/'source/hpc/discovery/submit_sparse_smart_v2_retained_state_bic.py')
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
print(json.dumps(module.record_preflight(root),sort_keys=True))
PY
    fi
else
    "$VENV/bin/python" "$SMART_SOURCE_ROOT/simulation/summarize_sparse_smart_v2_retained_state_bic.py" --root "$run_dir"
fi
