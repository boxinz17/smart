#!/usr/bin/env bash
# Submit preparation first, then a fit-only GNU Parallel pool. No data copying.
set -euo pipefail

usage() {
    cat <<'USAGE'
Usage: submit_sparse_smart_v2_pilot.sh --run-dir ROOT [options]
  ROOT already contains source/, source-manifest.json, plan.json, work-items.tsv.
  --workers N       Concurrent single-CPU fit steps (default 100)
  --time TIME       Main pool wall time (default 12:00:00)
  --mem SIZE        Memory per main-pool CPU (default 4G)
  --account NAME    Slurm account (default mkolar_1314)
  --partition NAME Slurm partition (default main)
  --dry-run         Write/print commands only; permit a local test root
Actual submissions require an existing absolute root under /scratch2.
Preparation uses one CPU, 8G, and 00:15:00. Its success releases the pool.
USAGE
}

run_dir= workers=100 wall_time=12:00:00 memory=4G
account=mkolar_1314 partition=main dry_run=0
while (($#)); do
    case "$1" in
        --run-dir|--workers|--time|--mem|--account|--partition)
            [[ $# -ge 2 && -n "$2" ]] || { usage >&2; exit 2; }
            case "$1" in
                --run-dir) run_dir=$2 ;; --workers) workers=$2 ;;
                --time) wall_time=$2 ;; --mem) memory=$2 ;;
                --account) account=$2 ;; --partition) partition=$2 ;;
            esac
            shift 2 ;;
        --dry-run) dry_run=1; shift ;;
        --help|-h) usage; exit 0 ;;
        *) printf 'Unknown argument: %s\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
done
[[ -n "$run_dir" && "$run_dir" == /* && -d "$run_dir" ]] || { printf 'An existing absolute --run-dir is required.\n' >&2; exit 2; }
run_dir=$(cd -- "$run_dir" && pwd -P)
if (( ! dry_run )); then
    case "$run_dir" in /scratch2/*) ;; *) printf 'Actual submission requires a resolved /scratch2 run directory.\n' >&2; exit 2 ;; esac
fi
[[ "$workers" =~ ^[1-9][0-9]*$ ]] || { printf 'Workers must be a positive integer.\n' >&2; exit 2; }
[[ "$wall_time" =~ ^([0-9]+-)?[0-9]+(:[0-9]{2}){0,2}$ ]] || { printf 'Invalid Slurm time.\n' >&2; exit 2; }
[[ "$memory" =~ ^[1-9][0-9]*[KMGT]?$ ]] || { printf 'Invalid memory per CPU.\n' >&2; exit 2; }
[[ "$account" =~ ^[A-Za-z0-9_.-]+$ && "$partition" =~ ^[A-Za-z0-9_.-]+$ ]] || { printf 'Invalid account or partition.\n' >&2; exit 2; }
for name in plan.json source-manifest.json work-items.tsv \
    source/hpc/discovery/sparse_smart_v2_prepare.sbatch \
    source/hpc/discovery/sparse_smart_v2_pool.sbatch \
    source/hpc/discovery/sparse_smart_v2_worker.sh \
    source/hpc/discovery/env.sh source/simulation/run_sparse_smart_v2_pilot.py; do
    [[ -s "$run_dir/$name" ]] || { printf 'Missing run input: %s\n' "$name" >&2; exit 2; }
done
export VENV=${VENV:-/home1/mkolar/envs/smart}
plan_python=${SMART_V2_PLAN_PYTHON:-python3}
"$plan_python" - "$run_dir" <<'PY'
import json, pathlib, sys
root = pathlib.Path(sys.argv[1])
plan = json.loads((root / 'plan.json').read_text())
manifest = json.loads((root / 'source-manifest.json').read_text())
if not isinstance(plan.get('tasks'), list) or not plan['tasks']:
    raise SystemExit('Plan must contain nonempty tasks')
rows = (root / 'work-items.tsv').read_text().splitlines()
if rows != [str(i) for i in range(len(plan['tasks']))]:
    raise SystemExit('Work items must be headerless consecutive integer task IDs')
if manifest.get('schema_version') != 1 or pathlib.Path(manifest.get('source_root', '')).resolve() != (root / 'source').resolve():
    raise SystemExit('Source manifest must identify this frozen source root')
if not isinstance(manifest.get('files'), dict) or not manifest['files']:
    raise SystemExit('Source manifest must contain file hashes')
PY
mkdir -p "$run_dir/logs" "$run_dir/tmp" "$run_dir/matplotlib"
prepare_command=(sbatch --parsable --job-name=smart-v2-prepare --account="$account" --partition="$partition"
    --ntasks=1 --cpus-per-task=1 --mem=8G --time=00:15:00 --export=ALL --chdir="$run_dir"
    --output="$run_dir/logs/prepare-%j.out" --error="$run_dir/logs/prepare-%j.err"
    "$run_dir/source/hpc/discovery/sparse_smart_v2_prepare.sbatch" "$run_dir")
pool_command=(sbatch --parsable --job-name=smart-v2-pool --account="$account" --partition="$partition"
    --ntasks="$workers" --cpus-per-task=1 --mem-per-cpu="$memory" --time="$wall_time"
    --export=ALL --chdir="$run_dir" --output="$run_dir/logs/pool-%j.out" --error="$run_dir/logs/pool-%j.err")

record() {
    local stage=$1 job_id=$2
    shift 2
    "$plan_python" - "$run_dir" "$stage" "$job_id" "$dry_run" "$@" <<'PY'
import datetime, json, os, pathlib, sys
root, stage, job_id, dry_run = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4] == '1'
path = root / ('submission-plan.json' if dry_run else 'submission.json')
data = json.loads(path.read_text()) if path.exists() else {'schema_version': 1, 'run_dir': str(root), 'dry_run': dry_run}
data[stage] = {'job_id': job_id or None, 'command': sys.argv[5:], 'recorded_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
data['venv'] = os.environ['VENV']
temporary = path.with_suffix('.tmp')
temporary.write_text(json.dumps(data, indent=2) + '\n')
temporary.replace(path)
PY
}
print_command() { printf '%q ' "$@"; printf '\n'; }
if (( dry_run )); then
    pool_command+=(--dependency=afterok:PREPARE_JOB_ID "$run_dir/source/hpc/discovery/sparse_smart_v2_pool.sbatch" "$run_dir" "$workers")
    record prepare '' "${prepare_command[@]}"
    record pool '' "${pool_command[@]}"
    print_command "${prepare_command[@]}"
    print_command "${pool_command[@]}"
    exit 0
fi
command -v sbatch >/dev/null || { printf 'sbatch is unavailable.\n' >&2; exit 2; }
mkdir "$run_dir/submission.lock" || { printf 'Run was already submitted or submission was attempted; inspect submission.json.\n' >&2; exit 2; }
trap 'status=$?; printf "%s\n" "$status" > "$run_dir/submission-exit-code.txt"' EXIT
record prepare '' "${prepare_command[@]}"
"${prepare_command[@]}" > "$run_dir/logs/prepare-submission.out" 2> "$run_dir/logs/prepare-submission.err"
prepare_reply=$(< "$run_dir/logs/prepare-submission.out")
prepare_id=${prepare_reply%%;*}
[[ "$prepare_id" =~ ^[0-9]+$ ]] || { printf 'Unrecognized preparation job ID: %s\n' "$prepare_reply" >&2; exit 2; }
record prepare "$prepare_id" "${prepare_command[@]}"
printf '%s\n' "$prepare_id" > "$run_dir/prepare-job-id.txt"
pool_command+=(--dependency="afterok:$prepare_id" "$run_dir/source/hpc/discovery/sparse_smart_v2_pool.sbatch" "$run_dir" "$workers")
record pool '' "${pool_command[@]}"
"${pool_command[@]}" > "$run_dir/logs/pool-submission.out" 2> "$run_dir/logs/pool-submission.err"
pool_reply=$(< "$run_dir/logs/pool-submission.out")
pool_id=${pool_reply%%;*}
[[ "$pool_id" =~ ^[0-9]+$ ]] || { printf 'Unrecognized pool job ID: %s\n' "$pool_reply" >&2; exit 2; }
record pool "$pool_id" "${pool_command[@]}"
printf '%s\n' "$pool_id" > "$run_dir/pool-job-id.txt"
printf 'Preparation job: %s\nFit-pool job: %s\nRun directory: %s\n' "$prepare_id" "$pool_id" "$run_dir"
