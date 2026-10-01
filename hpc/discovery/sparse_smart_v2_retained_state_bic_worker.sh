#!/usr/bin/env bash
# Rescore one existing case in an exclusive single-CPU step; never fit.
set -euo pipefail
[[ $# == 3 && ( "$1" == dispatch || "$1" == execute ) && "$3" =~ ^m[0-9]+_e[0-9]+_k[0-9]+_s[0-9]+$ && -n ${SLURM_JOB_ID:-} ]] || exit 2
mode=$1
run_dir=$(cd -- "$2" && pwd -P)
case_id=$3
mkdir -p "$run_dir/logs/cases"
exec >> "$run_dir/logs/cases/$case_id.log" 2>&1
finish() {
    status=$?
    trap - EXIT
    if [[ "$mode" == dispatch ]]; then
        printf '%s\n' "$status" > "$run_dir/logs/cases/$case_id.exit-code.txt"
        printf 'job_id\tcase_id\texit_code\tfinished_utc\n%s\t%s\t%s\t%s\n' \
            "$SLURM_JOB_ID" "$case_id" "$status" "$(date -u +%FT%TZ)" > "$run_dir/logs/cases/$case_id.launcher-status.tsv"
    else
        printf '%s\n' "$status" > "$run_dir/logs/cases/$case_id.process-exit-code.txt"
        printf 'job_id\tstep_id\tcase_id\texit_code\tfinished_utc\n%s\t%s\t%s\t%s\t%s\n' \
            "$SLURM_JOB_ID" "${SLURM_STEP_ID:-unknown}" "$case_id" "$status" "$(date -u +%FT%TZ)" > "$run_dir/logs/cases/$case_id.process-status.tsv"
    fi
    exit "$status"
}
trap finish EXIT
trap 'exit 143' TERM
if [[ "$mode" == dispatch ]]; then
    srun --exclusive --exact --nodes=1 --ntasks=1 --cpus-per-task=1 --kill-on-bad-exit=0 \
        bash "$run_dir/source/hpc/discovery/sparse_smart_v2_retained_state_bic_worker.sh" execute "$run_dir" "$case_id" </dev/null
else
    [[ -n ${SLURM_STEP_ID:-} && ${SLURM_CPUS_PER_TASK:-1} == 1 ]] || exit 2
    "$VENV/bin/python" "$run_dir/source/simulation/run_sparse_smart_v2_retained_state_bic.py" \
        --plan "$run_dir/plan.json" --case-id "$case_id"
fi
