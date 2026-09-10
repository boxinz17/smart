"""Plan and collect isolated Discovery budget-study cells without fitting models.

The planner uses only the standard library. Its explicit grid/configuration
contract is checked against the runner in tests, and source-content identities
bind each plan to the implementation copied with it. Collection checks identity
and execution provenance; the existing summarizer performs the scientific audit.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timezone
import hashlib
from itertools import product
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from uuid import uuid4

HERE = Path(__file__).resolve().parent
METHOD = "SparseSMARTBudgetStudy"
SCHEME = "sparse-smart-source-content-v1"
PLAN_SCHEME = "discovery-budget-study-plan-v1"
TABLE_FIELDS = ("task_id", "model", "experiment", "seed", "setting")
SOURCE_FILES = (
    "run_restricted_rrr.py", "run_sparse_smart.py", "run_sparse_smart_tuned.py",
    "run_sparse_smart_external.py", "external_validation_data.py",
    "run_sparse_smart_budget_study.py", "batch_manifest.py",
    "sparse_smart_selection.py", "sparse_smart_provenance.py",
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def read_json(path):
    value = json.loads(path.read_text(), parse_constant=lambda value: (_ for _ in ()).throw(
        ValueError(f"Nonfinite JSON value: {value}")))
    require(isinstance(value, dict), f"Expected a JSON object: {path}")
    return value


def atomic_json(value, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def inside(path, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    require(path.is_relative_to(root), f"Path escapes declared root: {path}")
    return path


def configuration(*, iteration_budgets=(500, 1000, 2000, 4000, 8000), checkpoint_interval=250,
                  init_penalties=(.03,), penalties_u=(.0025, .01, .04),
                  penalties_v=(.0025, .01, .04), stationarity_tol=1e-6):
    budgets = list(iteration_budgets)
    require(budgets and all(type(x) is int and x > 0 for x in budgets)
            and all(a < b for a, b in zip(budgets, budgets[1:])),
            "Iteration budgets must be positive, unique, and increasing")
    require(type(checkpoint_interval) is int and checkpoint_interval > 0,
            "Checkpoint interval must be a positive integer")
    grids = dict(init_penalties=list(init_penalties), penalties_u=list(penalties_u),
                 penalties_v=list(penalties_v))
    for name, values in grids.items():
        require(values and all(type(x) in (int, float) and math.isfinite(x)
                and x >= 0 and (name != "init_penalties" or x > 0) for x in values)
                and len(set(values)) == len(values), f"Invalid {name}")
    require(type(stationarity_tol) in (int, float) and math.isfinite(stationarity_tol)
            and stationarity_tol > 0, "Stationarity tolerance must be finite and positive")
    return dict(iteration_budgets=budgets, checkpoint_interval=checkpoint_interval, **grids,
                inverse_step=20., stationarity_tol=stationarity_tol, n_validation=100,
                validation_seed_tag=1397970481, checkpoint_execution="continuous",
                initialization_spectrum="projected", refinement_solver="anchor_projected",
                spectral_step="projected", strict_source_check=False)


def experiment_settings(model, experiment):
    """The existing paper grid; no estimator or generator is imported."""
    require(type(model) is int and model in range(3), "Invalid model")
    require(type(experiment) is int and experiment in range(4), "Invalid experiment")
    p, q = ((100, 50), (150, 100), (300, 200))[model]
    ns = ((200, 400, 600, 800, 1000), (300, 500, 700, 1000, 1200),
          (500, 700, 1000, 1200, 1500))[model]
    rows = []
    values = (ns, (1, 3, 5, 7, 9, 11), (0, 3, 5, 7, 10, 15, 20),
              (0., .01, .02, .05, .1, .5))[experiment]
    for value in values:
        row = dict(n=ns[0], p=p, q=q, sigma0=.01, target_rank=5, source_rank=10)
        key, prefix = (("n", "n"), ("target_rank", "r"),
                       ("source_rank", "rs"), ("sigma0", "sigma0"))[experiment]
        row[key] = value
        row["suffix"] = f"{prefix}={value}"
        rows.append(row)
    return rows


def inapplicability(setting):
    if setting["source_rank"] < 1:
        return "source_rank_must_be_positive"
    if setting["target_rank"] > setting["source_rank"]:
        return "target_rank_exceeds_source_rank"
    return None


def resolved_configuration(setting, config):
    rank, source = setting["target_rank"], setting["source_rank"]
    nu, nv = (source, source) if setting["sigma0"] == 0 else (setting["p"], setting["q"])
    counts = [max(0, (nu-rank)*rank), max(0, (nv-rank)*rank)]
    grid = [dict(init_penalty=li, penalty_u=lu, penalty_v=lv, support_limits=counts,
                 step_size_inverse=config["inverse_step"])
            for li, lu, lv in product(config["init_penalties"], config["penalties_u"], config["penalties_v"])]
    sparsity = min(source*rank, max(rank, 5))
    return dict(runner=config, iterations=config["iteration_budgets"][-1],
        margins=dict(d_lower=.05, d_upper=12., gap=.01, anchor_min=.005, trial_radius=1.),
        rank=rank, source_rank=source, sparsity=[sparsity, sparsity], support_limits=[counts],
        actual_complement_counts=counts, candidate_grid=grid, trajectory_count=len(grid),
        candidate_count=len(grid)*len(config["iteration_budgets"]), checkpoint_execution="continuous",
        validation_interval=config["checkpoint_interval"], n_train=setting["n"],
        n_validation=config["n_validation"], fit_sample="all_supplied_training_rows",
        validation_mode="independent_external", refit_on_all_data=False, tuning_uses_truth=False,
        selection_metric="mean((Y_validation-X_validation@C_hat)**2)",
        selection_inputs=["training_X", "training_Y", "observed_source", "validation_X", "validation_Y"],
        rank_semantics="fitted dimensions; generator truth remains rank 5 and source rank 10")


def source_metadata(source_root):
    root = Path(source_root).resolve()
    sources = {f"simulation/{name}": inside(root/"simulation"/name, root) for name in SOURCE_FILES}
    for role, directory in (("generator/smart", root/"smart"/"smart"),
                            ("sparse_smart", root/"sparse-smart"/"src"/"sparse_smart")):
        require(directory.is_dir(), f"Missing source package: {directory}")
        for path in sorted(directory.rglob("*.py")):
            sources[f"{role}/{path.relative_to(directory).as_posix()}"] = inside(path, root)
    hashes = {name: hashlib.sha256(path.read_bytes()).hexdigest() for name, path in sorted(sources.items())}
    manifests = {}
    for applicable in (True, False):
        files = [dict(name=name, sha256=sha) for name, sha in hashes.items()
                 if applicable or not name.startswith("sparse_smart/")]
        manifest = dict(api="source-package" if applicable else "not-used", generator="generate_data", files=files)
        manifests[str(applicable).lower()] = dict(manifest=manifest,
            fingerprint=digest(dict(scheme=SCHEME, manifest=manifest)))
    return dict(scheme=SCHEME, files=hashes, implementation=manifests)


def expected_cells(models, experiments, seed_ids, seeds, profile, setting_index):
    require(profile in ("full", "difficult"), "Invalid profile")
    require(setting_index is None or (type(setting_index) is int and len(models) == len(experiments) == 1),
            "Setting index requires exactly one model and experiment")
    cells = []
    for model, experiment in product(models, experiments):
        grid = experiment_settings(model, experiment)
        require(setting_index is None or 0 <= setting_index < len(grid), "Invalid setting index")
        for seed, (index, setting) in product(seed_ids, enumerate(grid)):
            if setting_index is not None and index != setting_index:
                continue
            if profile == "difficult" and not ((experiment == 2 and setting["source_rank"] in (5, 7))
                                               or (experiment == 3 and setting["sigma0"] == .5)):
                continue
            cells.append(dict(task_id=f"m{model}_e{experiment}_s{seed}_k{index}", model=model,
                experiment=experiment, seed=seed, setting=index, random_seed=seeds[seed],
                simulation_setting=setting, inapplicability_reason=inapplicability(setting)))
    require(cells, "Requested scope contains no cells")
    return cells


def make_plan(*, source_root=HERE.parent, models=(0, 1, 2), experiments=(0, 1, 2, 3),
              seed_ids=tuple(range(100)), profile="full", setting_index=None, config=None, seed_file=None):
    models, experiments, seed_ids = map(list, (models, experiments, seed_ids))
    for values, limit, name in ((models, 3, "models"), (experiments, 4, "experiments"), (seed_ids, 100, "seed IDs")):
        require(values and all(type(x) is int and 0 <= x < limit for x in values)
                and len(set(values)) == len(values), f"Require unique valid {name}")
    root = Path(source_root).resolve()
    seed_path = inside(seed_file or root/"simulation/data/random_seeds/experiment_seeds.csv", root)
    with seed_path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    require(rows and all(int(row["rd_seed_id"]) == i for i, row in enumerate(rows)), "Malformed saved seed IDs")
    seeds = [int(row["seed"]) for row in rows]
    require(max(seed_ids) < len(seeds), "Requested saved seed is unavailable")
    require(all(0 <= seed < 2**32 for seed in seeds), "Invalid saved random seed")
    config = configuration() if config is None else config
    cells = expected_cells(models, experiments, seed_ids, seeds, profile, setting_index)
    excluded = sum(cell["inapplicability_reason"] is not None for cell in cells)
    identity = dict(schema_version=1, method=METHOD, models=models, experiments=experiments,
        seed_ids=seed_ids, seed_file_sha256=hashlib.sha256(seed_path.read_bytes()).hexdigest(),
        seed_file_relative=seed_path.relative_to(root).as_posix(), configuration=config, profile=profile,
        setting_index=setting_index, expected_cells=len(cells), expected_applicable=len(cells)-excluded,
        expected_inapplicable=excluded, cells=cells, source=source_metadata(root))
    return dict(identity, plan_scheme=PLAN_SCHEME, plan_fingerprint=digest(identity),
                created=datetime.now(timezone.utc).isoformat(), source_root=str(root), no_fits_performed=True)


def write_plan(plan, output_root):
    validate_plan(plan)
    root = Path(output_root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    path = inside(root/"study-plan.json", root)
    table = inside(root/"work-items.tsv", root)
    if path.exists():
        old = read_json(path)
        validate_plan(old)
        require(old["plan_fingerprint"] == plan["plan_fingerprint"],
                "Existing plan has different scope, configuration, seeds, or source; use a fresh run root")
        plan = old
    else:
        atomic_json(plan, path)
    from io import StringIO
    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=TABLE_FIELDS, delimiter="\t", lineterminator="\n", extrasaction="ignore")
    writer.writerows(plan["cells"])
    content = stream.getvalue()
    require(not table.exists() or table.read_text() == content, "Existing work-item table differs from plan")
    table.write_text(content)
    return plan


def validate_plan(plan):
    require(isinstance(plan, dict), "Study plan must be a JSON object")
    require(plan.get("plan_scheme") == PLAN_SCHEME, "Unsupported study plan")
    identity = {key: value for key, value in plan.items()
                if key not in ("plan_scheme", "plan_fingerprint", "created", "source_root", "no_fits_performed")}
    require(digest(identity) == plan.get("plan_fingerprint"), "Study plan fingerprint mismatch")
    require(plan["schema_version"] == 1 and plan["method"] == METHOD, "Wrong study plan")
    for name, limit in (("models", 3), ("experiments", 4), ("seed_ids", 100)):
        values = plan[name]
        require(values and all(type(value) is int and 0 <= value < limit for value in values)
                and len(set(values)) == len(values), f"Invalid planned {name}")
    config = plan["configuration"]
    expected = configuration(**{key: config[key] for key in ("iteration_budgets", "checkpoint_interval",
                            "init_penalties", "penalties_u", "penalties_v", "stationarity_tol")})
    require(config == expected, "Unsupported study configuration")
    seeds = [None]*100
    for cell in plan["cells"]:
        seed = cell["seed"]
        require(type(seed) is int and 0 <= seed < 100, "Invalid planned seed")
        require(seeds[seed] is None or seeds[seed] == cell["random_seed"], "Inconsistent planned random seed")
        seeds[seed] = cell["random_seed"]
    cells = expected_cells(plan["models"], plan["experiments"], plan["seed_ids"], seeds,
                           plan["profile"], plan["setting_index"])
    require(cells == plan["cells"] and len(cells) == plan["expected_cells"], "Study plan cell mapping mismatch")
    excluded = sum(cell["inapplicability_reason"] is not None for cell in cells)
    require(plan["expected_applicable"] == len(cells)-excluded and plan["expected_inapplicable"] == excluded,
            "Study plan applicability mismatch")


def relative_result(cell):
    model, experiment = f"model{cell['model']+1}", f"exp{cell['experiment']+1}"
    suffix = cell["simulation_setting"]["suffix"]
    return Path(model)/experiment/f"BudgetStudy_result_{model}_{experiment}_{suffix}_rd_seed_id={cell['seed']}.json"


def validate_record_identity(record, cell, plan):
    require(isinstance(record, dict), "Record must be a JSON object")
    setting = cell["simulation_setting"]
    identity = dict(schema_version=1, method=METHOD, model=f"model{cell['model']+1}",
        experiment=f"exp{cell['experiment']+1}", rd_seed_id=cell["seed"], random_seed=cell["random_seed"],
        setting=setting, configuration=resolved_configuration(setting, plan["configuration"]),
        generator_arguments=dict(n=setting["n"], p=setting["p"], q=setting["q"], sigma0=setting["sigma0"],
                                 sigma=.5, r_star=5, r0_star=10, random_seed=cell["random_seed"]))
    require(all(record.get(key) == value for key, value in identity.items())
            and record.get("configuration_fingerprint") == digest(identity), "Record cell/configuration identity mismatch")
    applicable = cell["inapplicability_reason"] is None
    source = plan["source"]["implementation"][str(applicable).lower()]
    require(record.get("implementation_fingerprint_scheme") == SCHEME
            and record.get("implementation_manifest") == source["manifest"]
            and record.get("implementation_fingerprint") == source["fingerprint"]
            and source["fingerprint"] == digest(dict(scheme=SCHEME, manifest=source["manifest"])),
            "Record implementation does not match planned source")
    require(type(record.get("applicable")) is bool and record["applicable"] == applicable,
            "Record applicability mismatch")
    require(type(record.get("success")) is bool, "Missing record success status")
    valid = ("complete", "partial", "all_candidates_failed") if applicable else ("inapplicable",)
    require(record.get("status") in valid, "Invalid record status")
    require(record["success"] == (record["status"] in ("complete", "partial")), "Record success/status mismatch")
    if not applicable:
        require(record.get("failure_reason") == cell["inapplicability_reason"], "Wrong inapplicability reason")
    return record["status"]


def inspect_cell_manifest(task_root, cell, plan):
    canonical = inside(task_root/"budget_study_manifest.json", task_root)
    if canonical.exists():
        path = canonical
    else:
        directory = inside(task_root/"budget_study_manifest_attempts", task_root)
        attempts = sorted(directory.glob("*.json"))
        path = inside(attempts[-1], task_root) if attempts else None
    if path is None:
        return None, [dict(task_id=cell["task_id"], error="missing_cell_manifest")]
    manifest = read_json(path)
    expected = dict(schema_version=1, method=METHOD, models=[cell["model"]], experiments=[cell["experiment"]],
        seed_ids=[cell["seed"]], profile=plan["profile"], setting_index=cell["setting"],
        expected_cells=1, seed_file_sha256=plan["seed_file_sha256"], configuration=plan["configuration"])
    require(all(manifest.get(key) == value for key, value in expected.items()), "Per-cell manifest scope/configuration mismatch")
    expected_row = dict(model=f"model{cell['model']+1}", experiment=f"exp{cell['experiment']+1}",
                        setting=cell["simulation_setting"]["suffix"], seed_id=cell["seed"])
    manifest_cells = manifest.get("cells", [])
    require(isinstance(manifest_cells, list) and len(manifest_cells) <= 1
            and all(isinstance(row, dict) and all(row.get(key) == value for key, value in expected_row.items())
                    for row in manifest_cells),
            "Per-cell manifest contains an unexpected/duplicate cell")
    require(manifest.get("attempt_status") != "completed" or len(manifest_cells) == 1,
            "Completed per-cell manifest omits its result cell")
    manifest_errors = manifest.get("errors", [])
    require(isinstance(manifest_errors, list) and all(isinstance(error, dict) for error in manifest_errors),
            "Per-cell manifest errors must be a list of objects")
    errors = [dict(task_id=cell["task_id"], error="driver_error", detail=error) for error in manifest_errors]
    if manifest.get("attempt_status") != "completed":
        errors.append(dict(task_id=cell["task_id"], error="cell_execution_incomplete",
                           attempt_status=manifest.get("attempt_status")))
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                attempt_status=manifest.get("attempt_status")), errors


def aggregate(run_root, output_root=None):
    root = Path(run_root).resolve()
    output = Path(output_root).absolute() if output_root else root/"results"
    require(output.parent.resolve() == root and output.name not in ("source", "tasks", "logs"),
            "Aggregate output must be a dedicated direct child of the run root")
    output = inside(output, root)
    plan = read_json(inside(root/"study-plan.json", root))
    validate_plan(plan)
    table = inside(root/"work-items.tsv", root)
    with table.open(newline="") as stream:
        rows = list(csv.reader(stream, delimiter="\t"))
    require(rows == [[str(cell[key]) for key in TABLE_FIELDS] for cell in plan["cells"]],
            "Work-item table differs from study plan")
    if output.exists():
        previous = inside(output/"aggregation-report.json", output)
        require(previous.is_file() and read_json(previous).get("plan_fingerprint") == plan["plan_fingerprint"],
                "Refusing to replace an unrelated aggregate output directory")
    staging = Path(tempfile.mkdtemp(prefix=f".{output.name}-collect-", dir=root))
    errors, missing, copied, counts = [], [], [], Counter()
    try:
        for cell in plan["cells"]:
            task_id = cell["task_id"]
            try:
                task_root = inside(root/"tasks"/task_id/"results", root)
                for name in ("process-exit-code.txt", "exit-code.txt", "launcher-exit-code.txt"):
                    code_path = inside(root/"tasks"/task_id/name, root)
                    if not code_path.is_file():
                        errors.append(dict(task_id=task_id, error="missing_execution_exit_code", file=name))
                    else:
                        try:
                            code = int(code_path.read_text().strip())
                        except ValueError:
                            code = None
                        if code != 0:
                            errors.append(dict(task_id=task_id, error="execution_failure", file=name, exit_code=code))
                path = inside(task_root/relative_result(cell), task_root)
                manifest, manifest_errors = inspect_cell_manifest(task_root, cell, plan)
                errors.extend(manifest_errors)
                found = [inside(item, task_root) for item in task_root.rglob("BudgetStudy_result_*.json")]
                require(set(found) <= {path}, "Unexpected/duplicate result in per-cell directory")
                if not path.is_file():
                    missing.append(task_id)
                    continue
                raw = path.read_bytes()
                record = json.loads(raw)
                status = validate_record_identity(record, cell, plan)
                destination = staging/relative_result(cell)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(raw)
                counts[status] += 1
                copied.append(dict(task_id=task_id, model=record["model"], experiment=record["experiment"],
                    setting=record["setting"]["suffix"], seed_id=cell["seed"], status=status,
                    success=record["success"], path=str(output/relative_result(cell)),
                    source_relative=path.relative_to(root).as_posix(), sha256=hashlib.sha256(raw).hexdigest(),
                    cell_manifest=manifest))
            except (ValueError, KeyError, TypeError, OSError) as error:
                errors.append(dict(task_id=task_id, error="invalid_cell_artifact", message=str(error)))
                missing.append(task_id)
        execution_complete = not errors and not missing and len(copied) == plan["expected_cells"]
        coverage_complete = execution_complete and not any(counts[key] for key in ("partial", "all_candidates_failed"))
        report = dict(schema_version=1, plan_fingerprint=plan["plan_fingerprint"], no_fits_performed=True,
            expected_cells=plan["expected_cells"], expected_applicable=plan["expected_applicable"],
            expected_inapplicable=plan["expected_inapplicable"], recorded_cells=len(copied), missing_cells=missing,
            errors=errors, status_counts=dict(counts), execution_complete=execution_complete,
            budget_coverage_complete=coverage_complete, scientific_validation="pending_existing_summarizer",
            cells=copied, finished=datetime.now(timezone.utc).isoformat())
        manifest = {key: plan[key] for key in ("schema_version", "method", "models", "experiments", "seed_ids",
                    "seed_file_sha256", "configuration", "profile", "setting_index", "expected_cells",
                    "expected_applicable", "expected_inapplicable")}
        manifest.update(plan_fingerprint=plan["plan_fingerprint"], output_root=str(output), cells=copied,
            errors=errors, missing_cells=missing, status_counts=dict(counts),
            attempt_status="completed" if execution_complete else "incomplete", finished=report["finished"],
            execution_complete=execution_complete, budget_coverage_complete=coverage_complete,
            scientific_validation="pending_existing_summarizer", no_fits_performed=True)
        atomic_json(manifest, staging/"budget_study_manifest.json")
        atomic_json(report, staging/"aggregation-report.json")
        if output.exists():
            archive = inside(root/f"{output.name}_aggregation_attempts", root)
            archive.mkdir(exist_ok=True)
            os.replace(output, archive/uuid4().hex)
        os.replace(staging, output)
        return report
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def float_grid(value):
    try:
        return tuple(float(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("Expected a comma-separated numeric grid") from error


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("plan", help="Write a full-scope plan without importing estimators")
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--source-root", type=Path, default=HERE.parent)
    p.add_argument("--seed-file", type=Path)
    p.add_argument("--models", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--experiments", type=int, nargs="+", default=[0, 1, 2, 3])
    p.add_argument("--seed-ids", type=int, nargs="+", default=list(range(100)))
    p.add_argument("--profile", choices=("full", "difficult"), default="full")
    p.add_argument("--setting-index", type=int)
    p.add_argument("--iteration-budgets", type=int, nargs="+", default=[500, 1000, 2000, 4000, 8000])
    p.add_argument("--checkpoint-interval", type=int, default=250)
    p.add_argument("--init-penalties", type=float_grid, default=(.03,))
    p.add_argument("--penalties-u", type=float_grid, default=(.0025, .01, .04))
    p.add_argument("--penalties-v", type=float_grid, default=(.0025, .01, .04))
    p.add_argument("--stationarity-tol", type=float, default=1e-6)
    a = sub.add_parser("aggregate", help="Collect records and audit missing/failed cell execution")
    a.add_argument("--run-root", type=Path, required=True)
    a.add_argument("--output-root", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == "plan":
            values = vars(args).copy()
            values.pop("command")
            output = values.pop("output_root")
            config = configuration(**{key: values.pop(key) for key in ("iteration_budgets", "checkpoint_interval",
                "init_penalties", "penalties_u", "penalties_v", "stationarity_tol")})
            plan = write_plan(make_plan(config=config, **values), output)
            print(json.dumps({key: plan[key] for key in ("plan_fingerprint", "expected_cells", "expected_applicable", "expected_inapplicable")}))
            return 0
        report = aggregate(args.run_root, args.output_root)
        print(json.dumps({key: report[key] for key in ("expected_cells", "recorded_cells", "execution_complete", "budget_coverage_complete")}))
        return int(not report["execution_complete"])
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    raise SystemExit(main())
