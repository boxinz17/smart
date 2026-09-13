#!/usr/bin/env bash
# One-CPU preparation and focused verification before the fit allocation starts.
set -euo pipefail
[[ $# == 1 && -n ${SLURM_JOB_ID:-} && ${SLURM_NTASKS:-} == 1 && ${SLURM_CPUS_PER_TASK:-1} == 1 ]] || { printf 'Expected one-CPU Slurm preparation and RUN_DIR.\n' >&2; exit 2; }
run_dir=$(cd -- "$1" && pwd -P)
[[ -s "$run_dir/plan.json" && -s "$run_dir/source-manifest.json" ]] || { printf 'Missing frozen plan or source manifest.\n' >&2; exit 2; }
mkdir -p "$run_dir/logs" "$run_dir/tmp/preparation" "$run_dir/matplotlib/preparation"
finish() {
    status=$?
    trap - EXIT
    printf '%s\n' "$status" > "$run_dir/preparation-exit-code.txt"
    printf 'job_id\texit_code\tfinished_utc\n%s\t%s\t%s\n' "$SLURM_JOB_ID" "$status" "$(date -u +%FT%TZ)" > "$run_dir/preparation-process-status.tsv"
    exit "$status"
}
trap finish EXIT
trap 'exit 143' TERM
export TMPDIR="$run_dir/tmp/preparation" MPLCONFIGDIR="$run_dir/matplotlib/preparation"
export VENV=${VENV:-/home1/mkolar/envs/smart} SMART_SOURCE_ROOT="$run_dir/source"
source "$SMART_SOURCE_ROOT/hpc/discovery/env.sh"
export PYTHONPATH="$SMART_SOURCE_ROOT/sparse-smart-v2/src:$PYTHONPATH"
cd "$run_dir"
"$VENV/bin/python" -c 'import numpy, scipy, sklearn, sparse_smart_v2; print("V2 numerical imports ready:", sparse_smart_v2.__file__)'
"$VENV/bin/python" -m pytest -q --import-mode=importlib -p no:cacheprovider \
    --basetemp="$run_dir/tmp/preparation/pytest" \
    "$SMART_SOURCE_ROOT/sparse-smart-v2/tests" \
    "$SMART_SOURCE_ROOT/simulation/tests/test_sparse_smart_v2_pilot.py" \
    "$SMART_SOURCE_ROOT/simulation/tests/test_sparse_smart_v2_launcher.py"
"$VENV/bin/python" "$SMART_SOURCE_ROOT/simulation/run_sparse_smart_v2_pilot.py" prepare --root "$run_dir"
[[ -s "$run_dir/preparation.json" ]] || { printf 'Preparation did not produce its success marker.\n' >&2; exit 2; }
