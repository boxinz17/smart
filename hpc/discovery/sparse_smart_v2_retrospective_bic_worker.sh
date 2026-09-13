#!/usr/bin/env bash
# GNU Parallel dispatches one independent case to each one-CPU Slurm step.
set -euo pipefail
[[ $# == 2 && "$2" =~ ^m[0-9]+_e[0-9]+_k[0-9]+_s[0-9]+$ && -n ${SLURM_JOB_ID:-} ]] || exit 2
run_dir=$(cd -- "$1" && pwd -P)
case_id=$2
exec > "$run_dir/logs/cases/$case_id.log" 2>&1
finish() {
    status=$?
    trap - EXIT
    printf '%s\n' "$status" > "$run_dir/logs/cases/$case_id.exit-code.txt"
    printf 'Finished job=%s case=%s exit=%s at=%s\n' "$SLURM_JOB_ID" "$case_id" "$status" "$(date -u +%FT%TZ)"
    exit "$status"
}
trap finish EXIT
trap 'exit 143' TERM
printf 'Dispatch job=%s case=%s at=%s\n' "$SLURM_JOB_ID" "$case_id" "$(date -u +%FT%TZ)"
srun --exclusive --exact --nodes=1 --ntasks=1 --cpus-per-task=1 --kill-on-bad-exit=0 \
    "$VENV/bin/python" "$run_dir/source/simulation/run_sparse_smart_v2_retrospective_bic.py" \
    --plan "$run_dir/plan.json" --case-id "$case_id" </dev/null
