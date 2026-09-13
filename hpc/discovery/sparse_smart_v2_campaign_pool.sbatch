#!/usr/bin/env bash
# Fit-only pool: one exclusive single-CPU Slurm step per GNU Parallel task.
set -euo pipefail
[[ $# == 4 && "$2" =~ ^[1-9][0-9]*$ ]] || { printf 'Expected RUN_DIR WORKERS TASK_FILE_BASENAME ATTEMPT_TAG.\n' >&2; exit 2; }
run_dir=$(cd -- "$1" && pwd -P)
workers=$2
task_file=$3
attempt_tag=$4
[[ "$attempt_tag" =~ ^chunk-[0-9]{4}-attempt-[0-9]{4}$ ]] || { printf 'Invalid campaign attempt tag.\n' >&2; exit 2; }
attempt_dir="$run_dir/campaign/attempts/$attempt_tag"
[[ -s "$attempt_dir/submission.json" ]] || { printf 'Missing exact campaign submission record.\n' >&2; exit 2; }
[[ "$task_file" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || { printf 'Task file must be a simple basename within RUN_DIR.\n' >&2; exit 2; }
[[ -n ${SLURM_JOB_ID:-} && ${SLURM_NTASKS:-} == "$workers" && ${SLURM_CPUS_PER_TASK:-1} == 1 ]] || { printf 'Unexpected single-CPU-task allocation.\n' >&2; exit 2; }
mkdir -p "$run_dir/logs/steps" "$attempt_dir/tmp" "$attempt_dir/matplotlib"
export VENV=${VENV:-/home1/mkolar/envs/smart} SLURM_EXPORT_ENV=ALL
export TMPDIR="$attempt_dir/tmp" MPLCONFIGDIR="$attempt_dir/matplotlib"

verify_preparation() {
    "$VENV/bin/python" - "$run_dir" "$task_file" "${1:-}" "$attempt_dir/submission.json" <<'PY'
import hashlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
task_name, expected_path, submission_path = sys.argv[2:]
submission = json.loads(pathlib.Path(submission_path).read_text())
digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
plan = json.loads((root / 'plan.json').read_text())
manifest = json.loads((root / 'source-manifest.json').read_text())
ready = json.loads((root / 'preparation.json').read_text())
if ready.get('status') != 'complete' or ready.get('success') is not True:
    raise SystemExit('Preparation is incomplete')
for field, name in [('plan_sha256', 'plan.json'), ('source_manifest_sha256', 'source-manifest.json')]:
    if ready.get(field) != digest(root / name):
        raise SystemExit('Preparation does not match ' + name)
if ready.get('n_tasks') != len(plan['tasks']) or ready.get('n_cases') != len(plan['cases']):
    raise SystemExit('Preparation counts do not match plan')
if (root / 'work-items.tsv').read_text().splitlines() != [str(i) for i in range(ready['n_tasks'])]:
    raise SystemExit('Work-item IDs do not match plan')
task_path = root / task_name
if task_path.is_symlink() or not task_path.is_file() or task_path.resolve().parent != root:
    raise SystemExit('Task file must be a regular file directly within RUN_DIR')
task_bytes = task_path.read_bytes()
try:
    rows = task_bytes.decode('ascii').split('\n')
except UnicodeDecodeError:
    raise SystemExit('Task file must contain canonical decimal task IDs')
if rows[-1] == '':
    rows.pop()
if not rows:
    raise SystemExit('Task file must not be empty')
if any(not row.isdecimal() or str(int(row)) != row for row in rows):
    raise SystemExit('Task file must contain headerless canonical decimal task IDs')
if len(set(rows)) != len(rows):
    raise SystemExit('Task file must contain distinct task IDs')
if any(int(row) >= ready['n_tasks'] for row in rows):
    raise SystemExit('Task file contains an out-of-range task ID')
if (submission.get('plan_sha256') != digest(root / 'plan.json')
        or submission.get('source_manifest_sha256') != digest(root / 'source-manifest.json')
        or submission.get('task_file') != task_name
        or submission.get('task_file_sha256') != hashlib.sha256(task_bytes).hexdigest()
        or submission.get('task_ids') != [int(row) for row in rows]):
    raise SystemExit('Task subset differs from the exact submitted record')
chunks = json.loads((root / 'campaign/chunks.json').read_text())
if (submission.get('kind') != 'pool'
        or submission.get('tag') != pathlib.Path(submission_path).parent.name
        or submission.get('chunk_plan_sha256') != digest(root / 'campaign/chunks.json')
        or not set(submission['task_ids']).issubset(chunks['chunks'][submission['chunk_id']]['task_ids'])):
    raise SystemExit('Submitted record differs from immutable campaign chunks')
source = (root / 'source').resolve()
if manifest.get('schema_version') != 1 or pathlib.Path(manifest.get('source_root', '')).resolve() != source:
    raise SystemExit('Manifest identifies a different source root')
files = manifest.get('files')
if not isinstance(files, dict) or not files:
    raise SystemExit('Empty source manifest')
for relative, expected in files.items():
    part = pathlib.Path(relative)
    path = (source / part).resolve()
    if part.is_absolute() or '..' in part.parts or not path.is_relative_to(source) or digest(path) != expected:
        raise SystemExit('Frozen source changed: ' + relative)
report = {'verified_files': len(files), 'n_tasks': ready['n_tasks'], 'n_cases': ready['n_cases'],
          'task_file': task_name, 'task_file_sha256': hashlib.sha256(task_bytes).hexdigest(),
          'n_selected_tasks': len(rows), 'work_items_sha256': digest(root / 'work-items.tsv')}
unchanged = True
if expected_path:
    expected = json.loads(pathlib.Path(expected_path).read_text())
    unchanged = all(report[key] == expected.get(key) for key in
                    ('task_file', 'task_file_sha256', 'n_selected_tasks', 'work_items_sha256'))
    report['task_selection_unchanged'] = unchanged
print(json.dumps(report), flush=True)
if not unchanged:
    raise SystemExit('Task selection changed during pool execution')
PY
}
integrity_started=0
finish() {
    status=$?
    trap - EXIT
    if (( integrity_started )); then
        set +e
        verify_preparation "$attempt_dir/source-integrity-before.json" > "$attempt_dir/source-integrity-after.json" 2> "$attempt_dir/source-integrity-after.err"
        integrity_status=$?
        set -e
        printf '%s\n' "$integrity_status" > "$attempt_dir/source-integrity-exit-code.txt"
        if (( integrity_status != 0 )); then status=74; fi
    fi
    printf '%s\n' "$status" > "$attempt_dir/pool-exit-code.txt"
    printf 'job_id\texit_code\tfinished_utc\n%s\t%s\t%s\n' "$SLURM_JOB_ID" "$status" "$(date -u +%FT%TZ)" > "$attempt_dir/pool-process-status.tsv"
    if (( status == 0 )); then printf 'finished\n'; else printf 'failed\n'; fi > "$attempt_dir/fit-status.txt"
    exit "$status"
}
trap finish EXIT
trap 'exit 143' TERM
if ! type module >/dev/null 2>&1; then source /etc/profile.d/modules.sh; fi
module purge
module load gcc/13.3.0 python/3.12.8 "${SMART_PARALLEL_MODULE:-parallel/20240522}"
verify_preparation > "$attempt_dir/source-integrity-before.json"
integrity_started=1
command -v parallel >/dev/null || { printf 'GNU Parallel is unavailable.\n' >&2; exit 2; }
parallel --plain --version > "$attempt_dir/parallel-version.txt"
printf 'running\n' > "$attempt_dir/fit-status.txt"
cd "$run_dir"
# Relative command paths prevent braces in a literal run directory from being
# interpreted as replacement strings. Numeric work items have no shell syntax.
parallel --plain --quote --jobs "$workers" --delay 0.1 --halt never \
    --joblog "$attempt_dir/parallel-joblog.tsv" --colsep '\t' \
    bash ./source/hpc/discovery/sparse_smart_v2_worker.sh dispatch . '{1}' \
    :::: "$run_dir/$task_file"
