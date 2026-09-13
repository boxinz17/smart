"""Operational rank/source-dimension selection on a predeclared finite grid.

This is a separate, Slurm-only study. Outputs live in ``rank_tasks`` and never
alter the fixed-rank campaign. All model selection uses target validation data;
truth is evaluated only after selection. Runtime reports distinguish elapsed
workflow time (including evidence I/O) from recorded fitting-call time. The
source-independent RRR endpoint is explicitly labeled, so a winning endpoint
does not misleadingly appear to select a meaningful source dimension.

Example, using an already frozen code/environment snapshot::

    python -m reviewer_revision_20260913.rank_study plan /scratch2/.../rank-run \
        --seeds 0,1 --ranks 3,5,7 --source-ranks 10,15 --iterations 500
    python -m reviewer_revision_20260913.rank_study run /scratch2/.../rank-run 0

The plan command is inexpensive and may run locally. Production fitting is
guarded by Slurm and a /scratch2 output-root check. No SSH is performed here.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
from itertools import product
import json
import os
from pathlib import Path
import platform
import resource
import socket
import sys
import time
import traceback

import numpy as np

CODE_ROOT = Path(__file__).resolve().parents[2]
for _path in (CODE_ROOT / "simulation", CODE_ROOT / "sparse-smart/src",
              CODE_ROOT / "sparse-smart-v2/src"):
    sys.path.insert(0, str(_path))

from reviewer_revision_20260913 import competitors, design, runner


IMPLEMENTATION_VERSION = "rank-study-v2"
V2_LABELS = ("v2", "v2_transfer_only", "initializer_only")
DEFAULT_RANKS = (3, 5, 7)
DEFAULT_SOURCE_RANKS = (10, 15)


def _integer_grid(values, name, maximum=None):
    result = []
    for value in values:
        if (isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer)) or value < 1
                or (maximum is not None and value > maximum)):
            raise ValueError(f"{name} must contain positive admissible integers")
        if int(value) not in result:
            result.append(int(value))
    if not result:
        raise ValueError(f"{name} must be nonempty")
    return result


def build_rank_candidates(case, config):
    """Return every declared grid cell, including explicit inadmissible cells.

    Only physical dimensions from ``case`` are read. In particular target_rank,
    latent_source_rank, source_fit_rank and evaluation data are never consulted.
    Grid order is preserved after removing duplicate values; it fixes tie rules.
    Every requested target rank must have at least one admissible source grid
    cell, so the source-independent RRR endpoint remains available at every rank.
    """
    maximum = min(int(case["p"]), int(case["q"]))
    ranks = _integer_grid(config["rank_grid"], "rank_grid", maximum)
    source_ranks = _integer_grid(config["source_truncation_grid"],
                                 "source_truncation_grid", maximum)
    if max(ranks) > max(source_ranks):
        raise ValueError("Every target-rank candidate needs an admissible source dimension")
    return [dict(cell_index=i, fitted_rank=r, source_rank=s,
                 admissible=r <= s,
                 exclusion_reason=None if r <= s else "fitted_rank_exceeds_source_dimension")
            for i, (r, s) in enumerate(product(ranks, source_ranks))]


def choose_rank_results(records, X_validation, Y_validation):
    """Select successful finite coefficients using validation only.

    Supplied validation scores, training scores, truth metrics, and rank labels
    never determine selection. Exact ties retain the first eligible record.
    A missing/nonfinite/misshaped coefficient is an explicit exclusion. With no
    eligible record, return success=False and the complete selection audit.
    """
    xv, yv = np.asarray(X_validation, dtype=float), np.asarray(Y_validation, dtype=float)
    if (xv.ndim != 2 or yv.ndim != 2 or len(xv) == 0 or len(xv) != len(yv)
            or not np.isfinite(xv).all() or not np.isfinite(yv).all()):
        raise ValueError("Validation arrays must be finite matrices with matching nonzero rows")
    audit, winner, best_prediction = [], None, None
    records = list(records)
    for index, record in enumerate(records):
        row = dict(record_index=index, eligible=False,
                   cell_index=record.get("cell_index"), status=record.get("status", "ineligible"))
        if not bool(record.get("eligible", record.get("success", False))):
            row["exclusion_reason"] = record.get("error", record.get("exclusion_reason", "fit_ineligible"))
        else:
            try:
                coefficient = np.asarray(record["coefficient"], dtype=float)
                if coefficient.shape != (xv.shape[1], yv.shape[1]) or not np.isfinite(coefficient).all():
                    raise ValueError("Coefficient must have finite shape (predictors,responses)")
                with np.errstate(over="raise", invalid="raise"):
                    prediction = xv @ coefficient
                    loss = float(np.mean((yv - prediction) ** 2))
                    # The common response-squared offset cancels before summing.
                    difference = (None if best_prediction is None else
                        float(np.mean((prediction - best_prediction)
                                      * (prediction + best_prediction - 2 * yv))))
                if not np.isfinite(loss) or (difference is not None and not np.isfinite(difference)):
                    raise ValueError("Nonfinite validation score")
                row.update(eligible=True, status="ok", validation_mse=loss)
                if winner is None or difference < 0:
                    winner, best_prediction = index, prediction
            except (KeyError, TypeError, ValueError, FloatingPointError) as error:
                row.update(status="invalid_coefficient", exclusion_reason=str(error))
        audit.append(row)
    if winner is None:
        return dict(success=False, selected_index=None, coefficient=None, audit=audit,
                    error="No eligible rank/source-dimension candidate")
    return dict(success=True, selected_index=winner,
                coefficient=np.asarray(records[winner]["coefficient"], dtype=float).copy(),
                selected_record=records[winner], validation_mse=audit[winner]["validation_mse"],
                audit=audit)


def rank_configuration(iterations=500, ranks=DEFAULT_RANKS,
                       source_ranks=DEFAULT_SOURCE_RANKS):
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 0:
        raise ValueError("iterations must be a nonnegative integer")
    config = runner.configuration(iterations)
    config.update(rank_grid=_integer_grid(ranks, "ranks"),
                  source_truncation_grid=_integer_grid(source_ranks, "source_ranks"),
                  rank_selection="joint_validation_over_predeclared_rank_and_source_dimension_grid",
                  rank_known=False, source_dimension_known=False, refit_on_validation=False)
    return config


def representative_cases():
    """Six predeclared cases, selected from metadata without viewing outcomes.

    "reference" means exact population containment/alignment with the existing
    nonzero Gaussian source perturbation, not an exact observed source matrix.
    "fitted_source" uses n0=300 and source-only rank/ridge tuning from design.py.
    """
    return [case for case in design.case_manifest()
            if case["n_train"] in (80, 200) and (
                case["family"] == "reference"
                or (case["family"] == "containment" and case["left_angle_deg"] == 30
                    and case["right_angle_deg"] == 30)
                or (case["family"] == "fitted_source" and case["n_source"] == 300))]


def competitor_libraries(case, config):
    """Comparable operational rank grids, with no irrelevant duplicate dimensions.

    Ridge-to-source and nuclear contrast do not impose rank and are tuned once.
    The clean-source oracle is rank-tuned but explicitly uses clean source spans.
    Only estimated-source-subspace methods tune the operational source dimension.
    """
    cells = build_rank_candidates(case, config)
    ranks = list(dict.fromkeys(cell["fitted_rank"] for cell in cells))
    source_ranks = list(dict.fromkeys(cell["source_rank"] for cell in cells))
    methods = ("target_rrr", "target_ridge_rrr", "source_subspace_rrr",
               "source_subspace_ridge_rrr", "oracle_subspace_rrr",
               "ridge_to_source", "source_target_mixture", "nuclear_contrast")
    libraries = {}
    for method in methods:
        libraries[method] = competitors.candidate_grid(
            method, ranks=ranks, ridges=config["benchmark_ridges"],
            source_ranks=source_ranks, alphas=config["mixture_weights"],
            nuclear_penalties=config["nuclear_penalties"])
    return libraries


def _runtime_environment():
    return dict(host=socket.gethostname(), python=sys.version, platform=platform.platform(),
                numpy=np.__version__, slurm_job_id=os.environ.get("SLURM_JOB_ID"),
                thread_environment={key: os.environ.get(key) for key in (
                    "SLURM_CPUS_PER_TASK", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                    "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS")})


def _snapshot_files():
    """Hash actual local implementation files without reading git or other runs."""
    paths = list((CODE_ROOT / "sparse-smart-v2" / "src").rglob("*.py"))
    paths += list((CODE_ROOT / "sparse-smart" / "src").rglob("*.py"))
    paths += [Path(__file__), Path(runner.__file__), Path(design.__file__), Path(competitors.__file__)]
    return {str(path.relative_to(CODE_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(set(paths))}


def build_plan(root, seeds=(0, 1), iterations=500, ranks=DEFAULT_RANKS,
               source_ranks=DEFAULT_SOURCE_RANKS, case_ids=None):
    root = Path(root)
    config = rank_configuration(iterations, ranks, source_ranks)
    seed_ids = []
    for seed in seeds:
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or seed < 0:
            raise ValueError("Seeds must be nonnegative integers")
        if int(seed) not in seed_ids:
            seed_ids.append(int(seed))
    if not seed_ids:
        raise ValueError("At least one seed is required")
    cases = representative_cases()
    if case_ids is not None:
        requested = list(dict.fromkeys(case_ids))
        lookup = {case["case_id"]: case for case in cases}
        if not requested or any(name not in lookup for name in requested):
            raise ValueError("case_ids must name nonempty representative cases")
        cases = [lookup[name] for name in requested]
    grids = {case["case_id"]: build_rank_candidates(case, config) for case in cases}
    libraries = {case["case_id"]: competitor_libraries(case, config) for case in cases}
    v2_libraries = {}
    for case in cases:
        v2_libraries[case["case_id"]] = [
            dict(cell=cell, candidates=list(runner.v2_specs(
                dict(case, fitted_rank=cell["fitted_rank"], source_rank=cell["source_rank"]),
                config, "full_caps")) if cell["admissible"] else [])
            for cell in grids[case["case_id"]]]
    tasks = [dict(case_index=i, seed=seed, expected_methods=list(libraries[case["case_id"]])
                  + list(V2_LABELS)) for i, case in enumerate(cases) for seed in seed_ids]
    plan = dict(schema=1, study="operational_rank_selection", implementation=IMPLEMENTATION_VERSION,
                cases=cases, tasks=tasks, configuration=config, rank_cells=grids,
                competitor_libraries=libraries, v2_libraries=v2_libraries,
                code_sha256=_snapshot_files(), seed_ids=seed_ids, n_cases=len(cases), n_tasks=len(tasks),
                source_root=str(CODE_ROOT), output_namespace="rank_tasks",
                theorem_certified=False,
                runtime_policy="serial within task; fitting-call and wall times both saved; no cold-fit claim",
                selection_policy="minimum validation MSE; first eligible candidate on exact ties; no test/refit")
    root.mkdir(parents=True, exist_ok=True)
    if (root / "rank-plan.json").exists():
        raise FileExistsError("Use a fresh output root for each rank study plan")
    runner.atomic_json(root / "rank-plan.json", plan)
    (root / "rank-work-items.txt").write_text("".join(f"{i}\n" for i in range(len(tasks))))
    return plan


def _without_coefficients(record):
    return {key: value for key, value in record.items()
            if key not in ("coefficient", "prediction", "terminal_coefficient")}


def _fit_competitor_library(data, library, directory, observed, oracle_left=None, oracle_right=None):
    """Fit/audit/save every candidate, then select using validation only.

    Uncertified but finite returned fits are saved too; they remain ineligible.
    Per-call times exclude archive writes, while the caller records wall time.
    This helper receives no truth or test objects; the explicitly oracle method
    can receive only its clean source frames as separately named arguments.
    """
    directory.mkdir(parents=True, exist_ok=True)
    audit, pool, fits = [], [], []
    for index, spec in enumerate(library):
        began = time.perf_counter()
        row = dict(candidate_index=index, candidate=spec, status="failed")
        fit = None
        try:
            options = dict(spec, source_coefficient=data["C0"])
            if spec["method"].startswith("oracle_"):
                options.update(oracle_left=oracle_left, oracle_right=oracle_right)
            elif spec["method"].startswith("source_subspace"):
                options.update(observed_left=observed.left, observed_right=observed.right)
            fit = competitors.fit_competitor(data["X"], data["Y"], **options)
            loss = float(np.mean((data["Y_validation"] - data["X_validation"] @ fit.coefficient)**2))
            if not np.isfinite(loss):
                raise ValueError("Nonfinite validation score")
            certified = bool(fit.diagnostics.get("certified", False))
            row.update(status="ok" if certified else "uncertified", validation_mse=loss,
                       validation_loss=loss * data["Y_validation"].shape[1] / 2,
                       diagnostics=fit.diagnostics, coefficient_saved=True)
            if not certified:
                row["exclusion_reason"] = "numerical_certificate_not_met"
        except Exception as error:
            row.update(error=str(error), error_type=type(error).__name__, coefficient_saved=False)
            fit = None
        row["elapsed_seconds"] = time.perf_counter() - began
        if fit is not None:
            np.savez_compressed(directory / f"candidate-{index:04d}.npz", coefficient=fit.coefficient)
        fits.append(fit)
        pool.append(dict(eligible=row["status"] == "ok", status=row["status"],
                         coefficient=None if fit is None else fit.coefficient,
                         error=row.get("error", row.get("exclusion_reason"))))
        audit.append(row)
        runner.atomic_json(directory / "candidates.json", dict(complete=False, candidates=audit))
    selected = choose_rank_results(pool, data["X_validation"], data["Y_validation"])
    total = sum(row["elapsed_seconds"] for row in audit)
    runner.atomic_json(directory / "candidates.json", dict(complete=True, candidates=audit,
        expected_n_candidates=len(library), specs_sha256=hashlib.sha256(runner.canonical(library).encode()).hexdigest(),
        n_eligible=sum(row["status"] == "ok" for row in audit), total_seconds=total,
        selected_index=selected["selected_index"]))
    if not selected["success"]:
        error = ValueError(selected["error"])
        error.candidate_results = tuple(audit)
        raise error
    index = selected["selected_index"]
    return competitors.SelectionResult(fits[index], index, audit[index]["validation_loss"], tuple(audit), total, False)


def _fit_rank_v2(case, data, config, cells, directory):
    """Fit each admissible grid cell once and keep its independent label winners."""
    pool = {label: [] for label in V2_LABELS}
    cell_records = []
    all_call_seconds = 0.
    all_wall_seconds = 0.
    initializer_seconds = 0.
    initializer_attempts = 0
    initializer_eligible = 0
    n_candidates = n_eligible = 0
    for cell in cells:
        record = dict(cell)
        cell_dir = directory / f"cell-{cell['cell_index']:03d}"
        if not cell["admissible"]:
            record.update(status="inadmissible", wall_seconds=0., fitting_call_seconds=0.)
            winners, candidates = {}, []
            elapsed = 0.
        else:
            began = time.perf_counter()
            try:
                operational_case = dict(case, fitted_rank=cell["fitted_rank"],
                                        source_rank=cell["source_rank"])
                winners, candidates, elapsed = runner.fit_v2(
                    operational_case, data, config, cell_dir, "full_caps")
                record.update(status="completed", fitting_call_seconds=elapsed,
                              n_candidates=len(candidates), n_eligible=sum(c["eligible"] for c in candidates))
            except Exception as error:
                winners, candidates, elapsed = {}, [], 0.
                record.update(status="cell_exception", error=str(error), traceback=traceback.format_exc(),
                              fitting_call_seconds=None)
                # A partial upstream audit survives a cell-level exception.
                if (cell_dir / "candidates.json").exists():
                    partial = json.loads((cell_dir / "candidates.json").read_text())
                    candidates = partial.get("candidates", [])
                    elapsed = sum(row.get("elapsed_seconds", 0.) for row in candidates)
                    record["partial_fitting_call_seconds"] = elapsed
            record["wall_seconds"] = time.perf_counter() - began
            all_call_seconds += elapsed
            all_wall_seconds += record["wall_seconds"]
            n_candidates += len(candidates)
            n_eligible += sum(c.get("eligible", False) for c in candidates)
            initial_path = cell_dir / "initializers.json"
            if initial_path.exists():
                initial = json.loads(initial_path.read_text())
                initializer_seconds += initial.get("total_seconds", 0.)
                initial_rows = initial.get("candidates", [])
                initializer_attempts += len(initial_rows)
                initializer_eligible += sum(c.get("eligible", False) for c in initial_rows)
        for label in V2_LABELS:
            winner = winners.get(label)
            if winner is None:
                pool[label].append(dict(cell_index=cell["cell_index"], eligible=False,
                    status=record["status"], error=record.get("error", cell["exclusion_reason"] or "No eligible candidate")))
                continue
            candidate = dict(winner, cell_index=cell["cell_index"], eligible=True, status="ok")
            candidate["cached_selected_candidate_call_seconds"] = (
                candidates[winner["candidate_index"]].get("elapsed_seconds")
                if label != "initializer_only" and winner["candidate_index"] < len(candidates) else None)
            pool[label].append(candidate)
        cell_records.append(record)
        runner.atomic_json(directory / "rank-cells.json", dict(complete=False, cells=cell_records))
    runtime = dict(shared_computation_group="v2_rank_source_grid", fitting_call_seconds=all_call_seconds,
                   cell_wall_seconds=all_wall_seconds, initializer_pass_wall_seconds=initializer_seconds,
                   n_candidates=n_candidates, n_eligible=n_eligible,
                   n_initializer_candidates=initializer_attempts,
                   n_initializer_eligible=initializer_eligible,
                   cold_selected_fit_seconds=None,
                   timing_note="Calls reuse caches within each cell; cells run serially with fresh caches. Shared totals must not be summed over labels.")
    runner.atomic_json(directory / "rank-cells.json", dict(complete=True, cells=cell_records, runtime=runtime))
    return pool, runtime


def run_task(root, task_index):
    root = Path(root).resolve()
    if not os.environ.get("SLURM_JOB_ID") or not root.is_relative_to(Path("/scratch2")):
        raise RuntimeError("Production rank studies require Slurm and a /scratch2 results root")
    plan_path = root / "rank-plan.json"
    plan = json.loads(plan_path.read_text())
    if not 0 <= task_index < len(plan["tasks"]):
        raise ValueError("task_index outside rank plan")
    if plan["code_sha256"] != _snapshot_files():
        raise RuntimeError("Code differs from frozen rank plan; create a new plan for the new snapshot")
    digest = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    task, config = plan["tasks"][task_index], plan["configuration"]
    case = plan["cases"][task["case_index"]]
    directory = root / "rank_tasks" / f"task-{task_index:06d}"
    directory.mkdir(parents=True, exist_ok=True)
    result_path = directory / "result.json"
    if result_path.exists():
        existing = json.loads(result_path.read_text())
        if existing.get("complete") and existing.get("plan_sha256") == digest:
            return existing
        raise RuntimeError("Existing result has a different or incomplete task identity")
    started = time.perf_counter()
    bundle = design.generate_case(case, task["seed"])
    generation_seconds = time.perf_counter() - started
    data, evaluation = bundle["fit_data"], bundle["evaluation"]
    np.savez_compressed(directory / "data.npz", **{key: value for key, value in data.items() if isinstance(value, np.ndarray)})
    np.savez_compressed(directory / "truth.npz", **evaluation)
    result = dict(schema=1, study=plan["study"], task=task, case=case, plan_sha256=digest,
                  metadata=bundle["metadata"], runtime_environment=_runtime_environment(),
                  generation_and_source_fit_seconds=generation_seconds,
                  fit_data_fingerprint=runner.fingerprint(data), truth_fingerprint=runner.fingerprint(evaluation),
                  executed_code_sha256=_snapshot_files(), rank_evidence_schema=2,
                  methods={}, complete=False, theorem_certified=False)
    from sparse_smart_v2 import ObservedSource, prepare_source
    began = time.perf_counter()
    observed = prepare_source(ObservedSource(data["C0"]), p=case["p"], q=case["q"],
                              source_rank=max(config["source_truncation_grid"]))
    result["shared_source_decomposition_seconds"] = time.perf_counter() - began
    source_frames = dict(left=observed.left, right=observed.right,
                         singular_values=observed.source_singular_values)
    np.savez_compressed(directory / "observed-source-frames.npz", **source_frames)
    result["observed_source_frames_fingerprint"] = runner.fingerprint(source_frames)
    coefficients = {}
    for name, library in plan["competitor_libraries"][case["case_id"]].items():
        began = time.perf_counter()
        try:
            selected = _fit_competitor_library(data, library, directory / "benchmarks" / name, observed,
                oracle_left=evaluation["U0"] if name.startswith("oracle_") else None,
                oracle_right=evaluation["V0"] if name.startswith("oracle_") else None)
            coefficients[name] = selected.fit.coefficient
            candidate_times = [row.get("elapsed_seconds", 0.) for row in selected.candidate_results]
            result["methods"][name] = dict(success=True, selected_index=selected.selected_index,
                parameters=selected.fit.parameters, diagnostics=selected.fit.diagnostics,
                metrics=runner.evaluate(selected.fit.coefficient, data, evaluation),
                audit=selected.candidate_results, n_candidates=len(library),
                n_eligible=sum(row.get("status") == "ok" for row in selected.candidate_results),
                fitting_call_seconds=sum(candidate_times), wall_seconds=time.perf_counter() - began,
                selected_candidate_call_seconds=candidate_times[selected.selected_index],
                selected_fitted_rank=selected.fit.parameters.get("rank") if name not in ("ridge_to_source", "nuclear_contrast") else None,
                selected_source_dimension=selected.fit.parameters.get("source_rank")
                    if name.startswith("source_subspace") else None,
                oracle=name.startswith("oracle_"))
        except Exception as error:
            result["methods"][name] = dict(success=False, error=str(error), traceback=traceback.format_exc(),
                audit=getattr(error, "candidate_results", []), n_candidates=len(library),
                fitting_call_seconds=sum(row.get("elapsed_seconds", 0.) for row in getattr(error, "candidate_results", [])),
                n_eligible=sum(row.get("status") == "ok" for row in getattr(error, "candidate_results", [])),
                wall_seconds=time.perf_counter() - began)
        runner.atomic_json(directory / "partial.json", result)
    pool, runtime = _fit_rank_v2(case, data, config, plan["rank_cells"][case["case_id"]], directory / "v2")
    for label in V2_LABELS:
        selected = choose_rank_results(pool[label], data["X_validation"], data["Y_validation"])
        row = dict(success=selected["success"], selection_audit=selected["audit"], runtime=runtime,
                   n_grid_cells=len(pool[label]), n_successful_cells=sum(item["eligible"] for item in pool[label]))
        if selected["success"]:
            winner = selected["selected_record"]
            coefficients[label] = selected["coefficient"]
            endpoint = winner["method"] == "target_rrr"
            row.update(selected_index=selected["selected_index"], selected_cell_index=winner["cell_index"],
                selected_inner_candidate_index=winner["candidate_index"], parameters=winner["spec"],
                selected_fitted_rank=winner["spec"]["rank"],
                selected_source_dimension=None if endpoint else winner["spec"]["source_rank"],
                source_dimension_interpretation="not_used_by_target_rrr_endpoint" if endpoint else "validation_selected_initializer_dimension",
                selected_method=winner["method"], selected_iteration=winner["selected_iteration"],
                metrics=runner.evaluate(selected["coefficient"], data, evaluation),
                cached_selected_candidate_call_seconds=winner["cached_selected_candidate_call_seconds"],
                history=winner.get("history", []), oracle=False)
        else:
            row["error"] = selected["error"]
        result["methods"][label] = row
        runner.atomic_json(directory / f"{label}-rank-selection.json",
            dict(complete=True, candidates=[_without_coefficients(candidate) for candidate in pool[label]],
                 selection=row))
    np.savez_compressed(directory / "coefficients.npz", **coefficients)
    result.update(complete=True, wall_seconds=time.perf_counter() - started,
                  peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    runner.atomic_json(result_path, result)
    runner.atomic_json(directory / "summary.json", dict(case_id=case["case_id"], seed=task["seed"],
        methods={name: {key: value for key, value in row.items() if key in (
            "success", "error", "selected_fitted_rank", "selected_source_dimension", "selected_method",
            "metrics", "fitting_call_seconds", "wall_seconds", "runtime", "n_candidates", "n_eligible")}
            for name, row in result["methods"].items()}))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan")
    plan.add_argument("root")
    plan.add_argument("--seeds", default="0,1")
    plan.add_argument("--ranks", default="3,5,7")
    plan.add_argument("--source-ranks", default="10,15")
    plan.add_argument("--iterations", type=int, default=500)
    plan.add_argument("--case-ids", help="Comma-separated subset of representative case IDs")
    run = commands.add_parser("run")
    run.add_argument("root")
    run.add_argument("task", type=int)
    args = parser.parse_args()
    if args.command == "plan":
        plan = build_plan(args.root, [int(value) for value in args.seeds.split(",")], args.iterations,
                          [int(value) for value in args.ranks.split(",")],
                          [int(value) for value in args.source_ranks.split(",")],
                          None if args.case_ids is None else args.case_ids.split(","))
        print(runner.canonical(dict(n_cases=plan["n_cases"], n_tasks=plan["n_tasks"],
                                    rank_grid=plan["configuration"]["rank_grid"],
                                    source_grid=plan["configuration"]["source_truncation_grid"])))
    else:
        result = run_task(args.root, args.task)
        print(runner.canonical(dict(event="rank_task_complete", task=args.task, complete=result["complete"])))


if __name__ == "__main__":
    main()
