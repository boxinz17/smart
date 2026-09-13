#!/usr/bin/env bash
# Validate the frozen source and prepare shared data references, using one CPU.
set -euo pipefail
[[ $# == 2 && -n ${SLURM_JOB_ID:-} && ${SLURM_NTASKS:-} == 1 && ${SLURM_CPUS_PER_TASK:-1} == 1 ]] || { printf 'Expected one-CPU Slurm preparation and RUN_DIR ATTEMPT_TAG.\n' >&2; exit 2; }
run_dir=$(cd -- "$1" && pwd -P)
attempt_tag=$2
[[ "$attempt_tag" =~ ^prepare-attempt-[0-9]{4}$ ]] || { printf 'Invalid preparation attempt tag.\n' >&2; exit 2; }
attempt_dir="$run_dir/campaign/attempts/$attempt_tag"
[[ -s "$attempt_dir/submission.json" ]] || { printf 'Missing exact submission record.\n' >&2; exit 2; }
mkdir -p "$attempt_dir/tmp" "$attempt_dir/matplotlib"
finish() {
    status=$?
    trap - EXIT
    printf '%s\n' "$status" > "$attempt_dir/preparation-exit-code.txt"
    printf 'job_id\texit_code\tfinished_utc\n%s\t%s\t%s\n' "$SLURM_JOB_ID" "$status" "$(date -u +%FT%TZ)" > "$attempt_dir/preparation-process-status.tsv"
    exit "$status"
}
trap finish EXIT
trap 'exit 143' TERM
export TMPDIR="$attempt_dir/tmp" MPLCONFIGDIR="$attempt_dir/matplotlib"
export VENV=${VENV:-/home1/mkolar/envs/smart} SMART_SOURCE_ROOT="$run_dir/source"
source "$SMART_SOURCE_ROOT/hpc/discovery/env.sh"
export PYTHONPATH="$SMART_SOURCE_ROOT/simulation:$SMART_SOURCE_ROOT/sparse-smart-v2/src:$PYTHONPATH"
cd "$run_dir"
"$VENV/bin/python" - "$run_dir" "$attempt_dir/submission.json" <<'PY'
import hashlib, importlib.util, json, pathlib, sys
root, record_path = map(pathlib.Path, sys.argv[1:])
record = json.loads(record_path.read_text())
digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
if (record.get('kind') != 'prepare' or record.get('tag') != record_path.parent.name
        or record.get('plan_sha256') != digest(root / 'plan.json')
        or record.get('source_manifest_sha256') != digest(root / 'source-manifest.json')
        or record.get('chunk_plan_sha256') != digest(root / 'campaign/chunks.json')):
    raise SystemExit('Preparation inputs differ from exact submitted record')
spec = importlib.util.spec_from_file_location('bic_launcher', root / 'source/hpc/discovery/submit_paper_source_comparators.py')
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
module.load_inputs(root)
PY
"$VENV/bin/python" -m pytest -q --import-mode=importlib -p no:cacheprovider \
    --basetemp="$attempt_dir/tmp/pytest" \
    "$SMART_SOURCE_ROOT/simulation/paper_source_comparators_20260913/tests"
"$VENV/bin/python" -m paper_source_comparators_20260913.runtime prepare --root "$run_dir"
[[ -s "$run_dir/preparation.json" ]] || { printf 'Preparation did not produce its marker.\n' >&2; exit 2; }

"$VENV/bin/python" -m paper_source_comparators_20260913.runtime canary --root "$run_dir"
