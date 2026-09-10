#!/usr/bin/env bash
# GNU Parallel dispatches independent steps within one Slurm allocation.
# The worker count and memory are supplied by submit.sh.
# No fixed node count or per-node task count is imposed.
set -euo pipefail

[[ $# == 1 && -f "$1/job-config.sh" ]] || { printf 'Use hpc/discovery/submit.sh to submit this job.\n' >&2; exit 2; }
source "$1/job-config.sh"
[[ "$pool_workers" =~ ^[1-9][0-9]*$ ]] || { printf 'Invalid saved pool worker count\n' >&2; exit 2; }
[[ -n ${SLURM_JOB_ID:-} && ${SLURM_NTASKS:-} == "$pool_workers" ]] || { printf 'Expected a Slurm allocation with the saved worker count\n' >&2; exit 2; }
[[ -s "$run_dir/work-items.tsv" ]] || { printf 'Missing work manifest\n' >&2; exit 2; }
mkdir -p "$run_dir/logs/steps"

finish() {
    status=$?
    trap - EXIT
    printf 'Pool job %s finished with exit code %s at %s\n' "${SLURM_JOB_ID:-unknown}" "$status" "$(date -u +%FT%TZ)"
    if ! mkdir -p "$archive_dir/logs" || ! rsync -a "$run_dir/logs/" "$archive_dir/logs/"; then
        printf 'Pool log archive failed; recover logs from %s\n' "$run_dir/logs" >&2
        [[ "$status" -ne 0 ]] || status=74
    fi
    exit "$status"
}
trap finish EXIT
trap 'exit 143' TERM

if ! type module >/dev/null 2>&1; then
    source /etc/profile.d/modules.sh
fi
module purge
module load gcc/13.3.0 "${SMART_PARALLEL_MODULE:-parallel/20240522}"
command -v parallel >/dev/null || { printf 'GNU Parallel is unavailable\n' >&2; exit 2; }
parallel --plain --version > "$run_dir/logs/parallel-version.txt"
cd "$run_dir"

printf 'Pool job %s dispatches up to %s concurrent items on allocation %s at %s\n' \
    "${SLURM_JOB_ID:-unknown}" "$pool_workers" "${SLURM_JOB_NODELIST:-unknown}" "$(date -u +%FT%TZ)"
# Ignore personal Parallel profiles, quote literal paths, and keep dispatching
# after an item fails. Each completed slot receives the next manifest row.
# A fresh submission always has a fresh job log; no resume or retry is implicit.
# Fixed relative command paths avoid interpreting braces in user-supplied roots
# as GNU Parallel replacement strings. The saved config restores absolute paths.
parallel --plain --quote --jobs "$pool_workers" --delay 0.2 --halt never \
    --joblog "$run_dir/logs/parallel-joblog.tsv" --colsep '\t' \
    bash ./source/hpc/discovery/pool_worker.sh . '{1}' '{2}' '{3}' \
    :::: "$run_dir/work-items.tsv"
