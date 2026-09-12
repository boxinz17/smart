#!/usr/bin/env bash
# Plan and snapshot the continuous SparseSMART budget study before submission.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: submit_budget_study.sh --workers N [options]

One fit-only GNU Parallel pool; each Slurm step fits a tuning subset of one data case.
The full defaults request 7,200 cells, including 900 inapplicable records.
The 100 penalty combinations are continuous trajectories, each capped at 2,000;
by default each combination gets its own work item (630,900 for 100 seeds).
This is a provisional pilot grid; preview a restricted three-seed scope first.

Study options:
  --tuning-preset NAME     expanded (default), source-rank-5, or source-rank-7
                          Rank probes require --models with one ID and select
                          experiment 2, setting 2 (rank 5) or 3 (rank 7)
                          Rank 5 adds .001
                          to U/V. Explicit tuning options override preset values
  --models IDS             IDs 0-2 (default: 0-2)
  --experiments IDS        IDs 0-3 (default: 0-3)
  --seeds IDS              Saved seed IDs 0-99 (default: 0-99)
  --setting-index N        Restrict one model and experiment to one grid setting
  --iteration-budgets IDS  Increasing caps (default: 500,1000,2000)
  --checkpoint-interval N  Full checkpoint interval (default: 250)
  --validation-interval N  Validation evaluation interval (default: 50)
  --validation-patience N  Iterations without significant improvement (default: 300)
  --validation-min-iterations N
                          Earliest validation stop (default: 500)
  --validation-min-relative-improvement X
                          Improvement needed to reset patience (default: .001)
  --no-validation-stop    Disable validation stopping for fixed-budget controls
  --n-validation N        Independent validation observations (default: 200)
  --validation-iterations VALUES
                          Extra times (default: 1,2,5,10,15,20,25,50,100,150,200)
                          Use none for only regular checkpoints and budget caps
  --init-penalties VALUES  Initializer grid (default: .01,.03,.1,.3)
  --penalties-u VALUES     Left penalty grid (default: .0025,.01,.04,.16,.32)
  --penalties-v VALUES     Right penalty grid (default: .0025,.01,.04,.16,.32)
  --stationarity-tol X     Certified early-stop tolerance (default: 1e-6)
  --tuning-task-size N|all Maximum penalty combinations per work item (default: 1)
                          N chunks V while fixing initializer and U; all keeps
                          every combination for a data case in one work item

Lists accept comma-separated or space-separated values; ID lists also accept
ascending ranges and strides, such as 0,2,5-9 or 0-98:2. Profile is always full.

Allocation options:
  --workers N             Required: maximum simultaneous single-CPU steps
                          Choose 16, 32, 64, etc. for this submission
  --time LIMIT            Wall-time limit for the entire pool (default: 24:00:00)
  --mem SIZE              Memory per single-CPU step (default: 8G)
  --account ACCOUNT       Default: SMART_ACCOUNT or mkolar_1314
  --partition NAME        Default: SMART_PARTITION or main
  --run-root PATH         Default: SMART_RUN_ROOT or /scratch1/$USER/smart/runs
  --archive-root PATH     Deprecated: record this value only; no archive is made
  --dry-run               Create plan/snapshot and print command; never submit

SMART_PLAN_PYTHON selects the existing Python used for metadata-only planning.
VENV and SMART_PARALLEL_MODULE select existing runtime installations.
No packages are installed. Each task uses one CPU and runner --workers 1;
Slurm distributes steps across the allocated nodes. No node count is fixed.
The CPU allocation is fixed once granted; choose --workers based on current
availability. Values above the number of planned work items are capped to that count.
Results, source provenance, and logs stay under the run directory. The job exits
when fitting finishes; aggregation, scientific summaries, and archival are deferred.
EOF
}
die() { printf 'Error: %s\n' "$*" >&2; exit 2; }
need_value() { [[ $# -ge 2 && -n "$2" && "$2" != --* ]] || die "$1 needs a value"; }

models=0-2 experiments=0-3 seeds=0-99 settings=''
tuning_preset=expanded models_explicit=0 experiments_explicit=0
budgets='' checkpoint_interval=250
validation_interval=50 validation_patience=300 validation_min_iterations=500
validation_min_relative_improvement=.001 n_validation=200 no_validation_stop=0
validation_iterations=1,2,5,10,15,20,25,50,100,150,200
init_penalties='' penalties_u='' penalties_v=''
stationarity_tol=1e-6 tuning_task_size=1 workers='' time_limit=24:00:00 memory=8G dry_run=0
account=${SMART_ACCOUNT:-mkolar_1314}
partition=${SMART_PARTITION:-main}
run_root=${SMART_RUN_ROOT:-/scratch1/${USER}/smart/runs}
archive_root=${SMART_ARCHIVE_ROOT:-}
planner=${SMART_PLAN_PYTHON:-python3}
while [[ $# -gt 0 ]]; do
    case "$1" in
        --help|-h) usage; exit 0 ;;
        --models|--experiments|--seeds|--iteration-budgets|--validation-iterations|--init-penalties|--penalties-u|--penalties-v)
            flag=$1; shift; values=()
            while [[ $# -gt 0 && "$1" != --* ]]; do values+=("$1"); shift; done
            [[ ${#values[@]} -gt 0 ]] || die "$flag needs a list"
            joined=$(IFS=','; printf '%s' "${values[*]}")
            case "$flag" in
                --models) models=$joined; models_explicit=1 ;;
                --experiments) experiments=$joined; experiments_explicit=1 ;;
                --seeds) seeds=$joined ;;
                --iteration-budgets) budgets=$joined ;;
                --validation-iterations) validation_iterations=$joined ;;
                --init-penalties) init_penalties=$joined ;;
                --penalties-u) penalties_u=$joined ;;
                --penalties-v) penalties_v=$joined ;;
            esac ;;
        --tuning-preset) need_value "$@"; tuning_preset=$2; shift 2 ;;
        --setting-index) need_value "$@"; settings=$2; shift 2 ;;
        --checkpoint-interval) need_value "$@"; checkpoint_interval=$2; shift 2 ;;
        --validation-interval) need_value "$@"; validation_interval=$2; shift 2 ;;
        --validation-patience) need_value "$@"; validation_patience=$2; shift 2 ;;
        --validation-min-iterations) need_value "$@"; validation_min_iterations=$2; shift 2 ;;
        --validation-min-relative-improvement) need_value "$@"; validation_min_relative_improvement=$2; shift 2 ;;
        --n-validation) need_value "$@"; n_validation=$2; shift 2 ;;
        --no-validation-stop) no_validation_stop=1; shift ;;
        --stationarity-tol) need_value "$@"; stationarity_tol=$2; shift 2 ;;
        --tuning-task-size) need_value "$@"; tuning_task_size=$2; shift 2 ;;
        --workers) need_value "$@"; workers=$2; shift 2 ;;
        --time) need_value "$@"; time_limit=$2; shift 2 ;;
        --mem) need_value "$@"; memory=$2; shift 2 ;;
        --account) need_value "$@"; account=$2; shift 2 ;;
        --partition) need_value "$@"; partition=$2; shift 2 ;;
        --run-root) need_value "$@"; run_root=$2; shift 2 ;;
        --archive-root) need_value "$@"; archive_root=$2; shift 2 ;;
        --dry-run) dry_run=1; shift ;;
        *) die "Unknown option: $1" ;;
    esac
done

parse_indices() {
    local spec=$1 maximum=$2 label=$3 part first last stride index seen=','
    local -a parts
    [[ "$spec" =~ ^[0-9]+(-[0-9]+(:[0-9]+)?)?(,[0-9]+(-[0-9]+(:[0-9]+)?)?)*$ ]] || die "Invalid $label list"
    parsed_indices=()
    IFS=',' read -r -a parts <<< "$spec"
    for part in "${parts[@]}"; do
        stride=1
        if [[ "$part" == *:* ]]; then stride=${part##*:}; part=${part%:*}; fi
        first=${part%%-*}; last=${part##*-}
        [[ ${#first} -le 3 && ${#last} -le 3 && ${#stride} -le 3 ]] || die "$label IDs must be 0-$maximum"
        first=$((10#$first)); last=$((10#$last)); stride=$((10#$stride))
        (( first <= last && last <= maximum && stride > 0 )) || die "Invalid $label range"
        for ((index=first; index<=last; index+=stride)); do
            [[ "$seen" != *",$index,"* ]] || die "Duplicate $label ID $index"
            seen+="$index,"; parsed_indices+=("$index")
        done
    done
}
parse_indices "$models" 2 model; model_ids=("${parsed_indices[@]}")
case "$tuning_preset" in
    expanded) preset_budgets=500,1000,2000
              preset_penalties=.0025,.01,.04,.16,.32 ;;
    source-rank-5|source-rank-7)
        (( models_explicit && ${#model_ids[@]} == 1 )) || die '--tuning-preset source-rank probes require --models with exactly one model ID'
        if [[ "$tuning_preset" == source-rank-5 ]]; then
            preset_setting=2; preset_penalties=.001,.0025,.01,.04,.16,.32
        else
            preset_setting=3; preset_penalties=.0025,.01,.04,.16,.32
        fi
        preset_budgets=500,1000,2000
        if (( ! experiments_explicit )); then experiments=2; fi
        [[ -z "$settings" || "$settings" == "$preset_setting" ]] || die "--tuning-preset $tuning_preset requires --setting-index $preset_setting"
        settings=$preset_setting ;;
    *) die 'Invalid --tuning-preset; choose expanded, source-rank-5, or source-rank-7' ;;
esac
# Resolve defaults after parsing so explicit grid/cap overrides are order-independent.
budgets=${budgets:-$preset_budgets}
init_penalties=${init_penalties:-.01,.03,.1,.3}
penalties_u=${penalties_u:-$preset_penalties}
penalties_v=${penalties_v:-$preset_penalties}
parse_indices "$experiments" 3 experiment; experiment_ids=("${parsed_indices[@]}")
if [[ "$tuning_preset" != expanded ]]; then
    [[ ${#experiment_ids[@]} == 1 && "${experiment_ids[0]}" == 2 ]] || die "--tuning-preset $tuning_preset requires --experiments 2"
fi
parse_indices "$seeds" 99 seed; seed_ids=("${parsed_indices[@]}")
[[ -n "$workers" ]] || die '--workers is required; choose the CPU concurrency for this submission (for example, 16, 32, or 64)'
[[ "$workers" =~ ^[1-9][0-9]*$ && ${#workers} -le 6 ]] || die '--workers must be a positive integer of at most six digits'
[[ "$tuning_task_size" == all || "$tuning_task_size" =~ ^[1-9][0-9]*$ ]] || die '--tuning-task-size must be a positive integer or all'
[[ "$time_limit" =~ ^[0-9:-]+$ ]] || die 'Invalid --time'
[[ "$memory" =~ ^[1-9][0-9]*[KkMmGgTt]?$ ]] || die 'Invalid --mem'
[[ "$account" =~ ^[A-Za-z0-9_.-]+$ ]] || die 'Invalid --account'
[[ "$partition" =~ ^[A-Za-z0-9_.-]+$ ]] || die 'Invalid --partition'
[[ "$budgets" =~ ^[1-9][0-9]*(,[1-9][0-9]*)*$ ]] || die 'Invalid --iteration-budgets'
IFS=',' read -r -a budget_ids <<< "$budgets"
if [[ -n "$settings" ]]; then
    [[ "$settings" =~ ^[0-6]$ && ${#model_ids[@]} == 1 && ${#experiment_ids[@]} == 1 ]] || die '--setting-index requires one model and one experiment'
fi
command -v "$planner" >/dev/null || die "Planning Python is unavailable: $planner"
"$planner" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 9) else "Planning requires Python 3.9 or newer; set SMART_PLAN_PYTHON to an existing compatible interpreter")'
command -v rsync >/dev/null || die 'rsync is required'
if (( ! dry_run )); then command -v sbatch >/dev/null || die 'Submit on Discovery, or use --dry-run'; fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
repo=$(cd -- "$script_dir/../.." && pwd -P)
mkdir -p -- "$run_root"
run_root=$(cd -- "$run_root" && pwd -P)
[[ "$run_root" != "$repo" && "$run_root" != "$repo/"* ]] || die 'Run root must be outside the source repository'
run_dir=$(mktemp -d "$run_root/$(date -u +%Y%m%dT%H%M%SZ)_sparse_smart_budget_XXXXXX")
mkdir -p "$run_dir/source" "$run_dir/logs" "$run_dir/tasks"
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

budget_args=(--iteration-budgets "${budget_ids[@]}" --checkpoint-interval "$checkpoint_interval"
    --validation-interval "$validation_interval" --validation-patience "$validation_patience"
    --validation-min-iterations "$validation_min_iterations"
    --validation-min-relative-improvement "$validation_min_relative_improvement" --n-validation "$n_validation"
    --validation-iterations "$validation_iterations"
    --init-penalties "$init_penalties" --penalties-u "$penalties_u" --penalties-v "$penalties_v"
    --stationarity-tol "$stationarity_tol")
if (( no_validation_stop )); then budget_args+=(--no-validation-stop); fi
plan_args=("$planner" "$run_dir/source/simulation/discovery_budget_study.py" plan
    --output-root "$run_dir" --models "${model_ids[@]}" --experiments "${experiment_ids[@]}"
    --seed-ids "${seed_ids[@]}" --profile full --tuning-task-size "$tuning_task_size" "${budget_args[@]}")
if [[ -n "$settings" ]]; then plan_args+=(--setting-index "$settings"); fi
PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 "${plan_args[@]}"
[[ -s "$run_dir/work-items.tsv" && -s "$run_dir/study-plan.json" ]] || die 'Planner did not produce the expected artifacts'
work_count=$(wc -l < "$run_dir/work-items.tsv")
work_count=${work_count//[[:space:]]/}
[[ "$work_count" =~ ^[1-9][0-9]*$ ]] || die 'Planner produced no work items'
pool_workers=$workers
if (( pool_workers > work_count )); then pool_workers=$work_count; fi
{
    printf 'run_dir=%q\npool_workers=%q\n' "$run_dir" "$pool_workers"
    # Compatibility metadata only: no fit-only script opens this destination.
    printf 'requested_archive_root=%q\n' "$archive_root"
    printf 'export VENV=%q\n' "${VENV:-$HOME/envs/smart}"
    printf 'export SMART_PARALLEL_MODULE=%q\n' "${SMART_PARALLEL_MODULE:-parallel/20240522}"
    printf 'budget_args=('; printf '%q ' "${budget_args[@]}"; printf ')\n'
} > "$run_dir/budget-job-config.sh"
{
    printf 'Created UTC: %s\nSource: %s\nRun: %s\n' "$(date -u +%FT%TZ)" "$repo" "$run_dir"
    printf 'Execution: fit-only\nPostprocessing: deferred\nArchival: deferred\nRequested archive root (unused): %s\n' "$archive_root"
    printf 'Work items: %s\nRequested workers: %s\nEffective workers: %s\nMemory per CPU: %s\nWhole-pool time: %s\n' "$work_count" "$workers" "$pool_workers" "$memory" "$time_limit"
    printf 'Account: %s\nPartition: %s\n' "$account" "$partition"
    printf 'Tuning combinations per task: %s\n' "$tuning_task_size"
    printf 'Tuning preset: %s\n' "$tuning_preset"
    printf 'Plan command: '; printf '%q ' "${plan_args[@]}"; printf '\n'
    if git -C "$repo" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
        printf '\nGit HEAD:\n'; git -C "$repo" rev-parse HEAD
        printf '\nGit status (snapshot includes uncommitted source):\n'; git -C "$repo" status --short
    fi
} > "$run_dir/manifest.txt"
printf 'deferred\n' > "$run_dir/postprocessing-status.txt"
printf 'deferred\n' > "$run_dir/archive-status.txt"
slurm_log_root=${run_dir//%/%%}
submit=(sbatch --parsable --account="$account" --partition="$partition" --job-name=sparse-smart-budget
    --ntasks="$pool_workers" --cpus-per-task=1 --mem-per-cpu="$memory" --time="$time_limit"
    --chdir="$run_dir" --output="$slurm_log_root/logs/pool-%j.out" --error="$slurm_log_root/logs/pool-%j.err"
    "$run_dir/source/hpc/discovery/budget_pool.sbatch" "$run_dir")
{ printf '%q ' "${submit[@]}"; printf '\n'; } > "$run_dir/submission-command.txt"
printf 'Run directory: %s\nPostprocessing: deferred; archival: deferred\nWork items: %s; simultaneous single-CPU steps: %s\nCommand: ' "$run_dir" "$work_count" "$pool_workers"
printf '%q ' "${submit[@]}"; printf '\n'
if (( dry_run )); then printf 'Dry run: plan and source saved in run directory; no jobs submitted.\n'; exit 0; fi
job_id=$("${submit[@]}")
printf '%s\n' "$job_id" | tee "$run_dir/job-id.txt"
printf 'Submitted pool job %s. Inspect with: squeue -j %s\n' "$job_id" "${job_id%%;*}"
