#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: submit.sh RUNNER MODEL EXPERIMENT [options] [-- runner arguments...]

RUNNER is a supported run_*.py basename inside simulation/.
Model and experiment IDs are fixed for each submission.

Options:
  --seeds IDS          Slurm seed list/ranges, e.g. 0,2,5-9 (default: 0)
  --mode array|pool    Separate array jobs or one GNU Parallel pool job (default: array)
  --concurrency N      Array mode: maximum concurrent jobs (default: 100)
  --workers N          Pool mode: single-CPU workers (default: 128; capped by work items)
  --settings IDS|all   Pool mode: split seed/setting pairs into separate work items
  --time LIMIT         Slurm time limit (default: 02:00:00)
  --mem SIZE           Memory per single-CPU worker (default: 4G)
  --account ACCOUNT    Default: SMART_ACCOUNT or mkolar_1314
  --partition NAME     Default: SMART_PARTITION or main
  --run-root PATH      Default: SMART_RUN_ROOT or /scratch1/$USER/smart/runs
  --archive-root PATH  Default: SMART_ARCHIVE_ROOT or $HOME/smart-results
  --dry-run            Create/review snapshot and command without submitting

Examples:
  submit.sh run_restricted_rrr.py 0 0 -- --setting-index 0
  submit.sh run_sparse_smart_external.py 0 3 --seeds 0-4 -- --iterations 500
  submit.sh run_sparse_smart_external.py 0 3 --mode pool --seeds 0-99 --settings all --workers 128 --dry-run

Runner --output-root and --seed-file overrides are rejected: each task keeps
its own results and uses the archived seed file. Extra arguments are literal.
EOF
}

die() { printf 'Error: %s\n' "$*" >&2; exit 2; }
need_value() { [[ $# -ge 2 && -n "$2" ]] || die "$1 needs a value"; }

if [[ ${1:-} == --help || ${1:-} == -h ]]; then usage; exit 0; fi
[[ $# -ge 3 ]] || { usage >&2; exit 2; }
runner=$1 model=$2 experiment=$3
shift 3
[[ "$runner" =~ ^run_[A-Za-z0-9_]+\.(py|R)$ ]] || die 'RUNNER must be a simulation runner basename'
[[ "$model" =~ ^[0-2]$ ]] || die 'MODEL must be 0, 1, or 2'
[[ "$experiment" =~ ^[0-3]$ ]] || die 'EXPERIMENT must be 0, 1, 2, or 3'
case "$runner" in
    run_SMART.py|run_SMARTCV.py|run_RRR.R|run_SRRR.R|run_RSSVD.R|run_SOFAR.R)
        die 'Paper-method reruns are deferred. Use the saved paper_reference curves for comparison.' ;;
    run_restricted_rrr.py|run_restricted_rrr_gauss_newton.py|run_sparse_smart.py|run_sparse_smart_tuned.py|run_sparse_smart_external.py) ;;
    *) die 'Unsupported per-seed runner. Use restricted RRR/GN or sparse SMART/tuned/external.' ;;
esac

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
repo=$(cd "$script_dir/../.." && pwd)
[[ -f "$repo/simulation/$runner" ]] || die "No runner at $repo/simulation/$runner"
seeds=0 concurrency=100 time_limit=02:00:00 memory=4G dry_run=0
mode=array workers=128 settings='' workers_given=0 concurrency_given=0
account=${SMART_ACCOUNT:-mkolar_1314}
partition=${SMART_PARTITION:-main}
run_root=${SMART_RUN_ROOT:-/scratch1/${USER}/smart/runs}
archive_root=${SMART_ARCHIVE_ROOT:-$HOME/smart-results}
extra=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --seeds) need_value "$@"; seeds=$2; shift 2 ;;
        --mode) need_value "$@"; mode=$2; shift 2 ;;
        --concurrency) need_value "$@"; concurrency=$2; concurrency_given=1; shift 2 ;;
        --workers) need_value "$@"; workers=$2; workers_given=1; shift 2 ;;
        --settings) need_value "$@"; settings=$2; shift 2 ;;
        --time) need_value "$@"; time_limit=$2; shift 2 ;;
        --mem) need_value "$@"; memory=$2; shift 2 ;;
        --account) need_value "$@"; account=$2; shift 2 ;;
        --partition) need_value "$@"; partition=$2; shift 2 ;;
        --run-root) need_value "$@"; run_root=$2; shift 2 ;;
        --archive-root) need_value "$@"; archive_root=$2; shift 2 ;;
        --dry-run) dry_run=1; shift ;;
        --) shift; extra=("$@"); break ;;
        *) die "Unknown launcher option: $1 (put runner options after --)" ;;
    esac
done
[[ "$concurrency" =~ ^[1-9][0-9]*$ ]] || die '--concurrency must be positive'
[[ "$mode" == array || "$mode" == pool ]] || die '--mode must be array or pool'
[[ "$workers" =~ ^[1-9][0-9]*$ && ${#workers} -le 6 ]] || die '--workers must be a positive integer of at most six digits'
if [[ "$mode" == array ]]; then
    (( workers_given == 0 )) && [[ -z "$settings" ]] || die '--workers and --settings require --mode pool'
else
    (( concurrency_given == 0 )) || die 'Use --workers for pool mode; --concurrency applies only to arrays'
fi
[[ "$time_limit" =~ ^[0-9:-]+$ ]] || die 'Invalid --time'
[[ "$memory" =~ ^[1-9][0-9]*[KkMmGgTt]?$ ]] || die 'Invalid --mem'
[[ "$account" =~ ^[A-Za-z0-9_.-]+$ ]] || die 'Invalid --account'
[[ "$partition" =~ ^[A-Za-z0-9_.-]+$ ]] || die 'Invalid --partition'
for argument in ${extra[@]+"${extra[@]}"}; do
    option=${argument%%=*}
    if [[ "$option" == --* && ( '--output-root' == "$option"* || '--seed-file' == "$option"* ) ]]; then
        die '--output-root and --seed-file (including abbreviations) are managed by the launcher'
    fi
    if [[ -n "$settings" && "$option" == --* && '--setting-index' == "$option"* ]]; then
        die 'Use --settings or runner --setting-index, not both'
    fi
done

# Expand zero-based lists/ranges without importing or executing simulation code.
parse_indices() {
    local spec=$1 maximum=$2 label=$3 part stride first last index seen=','
    local -a parts
    [[ "$spec" =~ ^[0-9]+(-[0-9]+(:[0-9]+)?)?(,[0-9]+(-[0-9]+(:[0-9]+)?)?)*$ ]] || die "Invalid $label list/range"
    parsed_indices=()
    IFS=',' read -r -a parts <<< "$spec"
    for part in "${parts[@]}"; do
        stride=1
        if [[ "$part" == *:* ]]; then stride=${part##*:}; part=${part%:*}; fi
        first=${part%%-*}; last=${part##*-}
        [[ ${#first} -le 3 && ${#last} -le 3 && ${#stride} -le 3 ]] || die "$label IDs must be 0-$maximum"
        first=$((10#$first)); last=$((10#$last)); stride=$((10#$stride))
        (( first <= last && last <= maximum && stride > 0 )) || die "$label IDs must be 0-$maximum, with ascending ranges and positive strides"
        for ((index=first; index<=last; index+=stride)); do
            [[ "$seen" != *",$index,"* ]] || die "Duplicate $label ID $index"
            seen+="$index,"; parsed_indices+=("$index")
        done
    done
}
parse_indices "$seeds" 99 seed
seed_ids=("${parsed_indices[@]}")
array_ids=$(IFS=','; printf '%s' "${seed_ids[*]}")
setting_ids=('-')
if [[ -n "$settings" ]]; then
    # Shared experiment_settings() grids: 5 sample sizes, 6 ranks, 7 source ranks,
    # 6 source-noise levels. All supported runners use those same grids.
    setting_counts=(5 6 7 6)
    setting_max=$((${setting_counts[$experiment]} - 1))
    if [[ "$settings" == all ]]; then settings="0-$setting_max"; fi
    parse_indices "$settings" "$setting_max" setting
    setting_ids=("${parsed_indices[@]}")
fi
work_count=$((${#seed_ids[@]} * ${#setting_ids[@]}))
pool_workers=0
if [[ "$mode" == pool ]]; then
    pool_workers=$workers
    if (( pool_workers > work_count )); then pool_workers=$work_count; fi
fi

command -v rsync >/dev/null || die 'rsync is required'
if (( ! dry_run )); then command -v sbatch >/dev/null || die 'Run this launcher on Discovery (sbatch unavailable)'; fi
mkdir -p "$run_root" "$archive_root"
run_root=$(cd "$run_root" && pwd)
archive_root=$(cd "$archive_root" && pwd)
[[ "$run_root" != "$archive_root" ]] || die 'Run and archive roots must differ'
for location in "$run_root" "$archive_root"; do
    [[ "$location" != "$repo" && "$location" != "$repo/"* ]] || die 'Run/archive roots must be outside the source repository'
done
run_dir=$(mktemp -d "$run_root/$(date -u +%Y%m%dT%H%M%SZ)_${runner%.*}_m${model}e${experiment}_${mode}_XXXXXX")
archive_dir="$archive_root/${run_dir##*/}"
mkdir -p "$run_dir/source" "$run_dir/logs" "$archive_dir"

# Snapshot current files, including uncommitted source edits. Copy only source,
# small reference data, tests, and metadata; never environments/results/raw data.
rsync -a --prune-empty-dirs \
    --exclude='.git/' --exclude='.venv/' --exclude='venv/' --exclude='env/' \
    --exclude='__pycache__/' --exclude='.pytest_cache/' --exclude='*.egg-info/' \
    --exclude='build/' --exclude='dist/' --exclude='result/' --exclude='results/' \
    --exclude='logs/' --exclude='fig/' --exclude='single_cell/' \
    --include='*/' --include='*.py' --include='*.R' --include='*.r' \
    --include='*.sh' --include='*.sbatch' --include='*.toml' --include='*.md' \
    --include='*.txt' --include='*.csv' --include='*.json' --include='*.yaml' \
    --include='*.yml' --include='.python-version' --include='py.typed' --include='LICENSE' --exclude='*' \
    "$repo/" "$run_dir/source/"
printf '%s\0' "$runner" "$model" "$experiment" ${extra[@]+"${extra[@]}"} > "$run_dir/invocation.args"
{
    printf 'run_dir=%q\narchive_dir=%q\n' "$run_dir" "$archive_dir"
    printf 'mode=%q\npool_workers=%q\n' "$mode" "$pool_workers"
    printf 'export VENV=%q\n' "${VENV:-$HOME/envs/smart}"
    printf 'export SMART_PARALLEL_MODULE=%q\n' "${SMART_PARALLEL_MODULE:-parallel/20240522}"
} > "$run_dir/job-config.sh"
{
    printf 'Created UTC: %s\nSource: %s\nRun: %s\nArchive: %s\n' "$(date -u +%FT%TZ)" "$repo" "$run_dir" "$archive_dir"
    printf 'Account: %s\nPartition: %s\nSeeds: %s\nConcurrency: %s\nTime: %s\nMemory: %s\n' "$account" "$partition" "$array_ids" "$concurrency" "$time_limit" "$memory"
    printf 'Mode: %s\nWork items: %s\nRequested pool worker option: %s\nEffective pool worker request: %s\nSplit settings: %s\n' "$mode" "$work_count" "$workers" "$pool_workers" "${settings:-none}"
    if [[ "$mode" == pool ]]; then
        printf 'Dispatcher: GNU Parallel\nParallel module: %s\nLaunch delay: 0.2 seconds\n' "${SMART_PARALLEL_MODULE:-parallel/20240522}"
    fi
    printf 'Runner invocation per task: '; printf '%q ' "$runner" "$model" "$experiment" '<seed ID>' ${extra[@]+"${extra[@]}"}; printf '\n'
    if git -C "$repo" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        printf '\nGit HEAD:\n'; git -C "$repo" rev-parse HEAD
        printf '\nGit status (includes untracked source):\n'; git -C "$repo" status --short
    fi
} > "$run_dir/manifest.txt"
: > "$run_dir/work-items.tsv"
for seed in "${seed_ids[@]}"; do
    for setting in "${setting_ids[@]}"; do
        task_id=$seed
        if [[ "$setting" != '-' ]]; then task_id="s${seed}_k${setting}"; fi
        printf '%s\t%s\t%s\n' "$task_id" "$seed" "$setting" >> "$run_dir/work-items.tsv"
        mkdir -p "$run_dir/tasks/$task_id"
    done
done
rsync -a "$run_dir/source/" "$archive_dir/source/"
rsync -a "$run_dir/manifest.txt" "$run_dir/invocation.args" "$run_dir/job-config.sh" "$run_dir/work-items.tsv" "$archive_dir/"

submit=(sbatch --parsable --account="$account" --partition="$partition"
    --job-name="${runner#run_}_m${model}e${experiment}" --cpus-per-task=1
    --time="$time_limit" --chdir="$run_dir")
if [[ "$mode" == array ]]; then
    submit+=(--array="${array_ids}%${concurrency}" --nodes=1 --ntasks=1 --mem="$memory"
        --output="$run_dir/logs/array-%A_%a.out" --error="$run_dir/logs/array-%A_%a.err"
        "$run_dir/source/hpc/discovery/simulation.sbatch" "$run_dir")
else
    # No node count, tasks-per-node, nodelist, or exclusive-node reservation:
    # Slurm places the worker CPUs and memory across available suitable nodes.
    submit+=(--ntasks="$pool_workers" --mem-per-cpu="$memory"
        --output="$run_dir/logs/pool-%j.out" --error="$run_dir/logs/pool-%j.err"
        "$run_dir/source/hpc/discovery/pool.sbatch" "$run_dir")
    printf 'Pool: %s work items, --workers=%s, %s workers in allocation request.\n' "$work_count" "$workers" "$pool_workers"
fi
printf 'Run directory: %s\nArchive directory: %s\nCommand: ' "$run_dir" "$archive_dir"
printf '%q ' "${submit[@]}"; printf '\n'
if (( dry_run )); then printf 'Dry run: source snapshot created; no jobs submitted.\n'; exit 0; fi
job_id=$("${submit[@]}")
printf '%s\n' "$job_id" | tee "$run_dir/job-id.txt"
cp "$run_dir/job-id.txt" "$archive_dir/job-id.txt"
printf 'Submitted %s job %s. Inspect it with: squeue -j %s\n' "$mode" "$job_id" "${job_id%%;*}"
