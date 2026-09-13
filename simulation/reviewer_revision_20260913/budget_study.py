"""Separate Slurm-only iteration sensitivity on a fixed six-specification grid.

Each (case, seed, budget) is a distinct task with six fresh, noncached fits.
A failure at a larger budget cannot censor a separately successful smaller
budget fit. Retained outputs from failed fits are saved for diagnosis only and
never enter that budget's refined-estimator selection. Raw converged Lasso-SVD
initializers are assessed independently of later chart/refinement failures.

This is a reduced-grid diagnostic, not a retuning of the main comparison, a
hard-cap ablation, or a global-convergence claim. Validation patience is disabled
to expose iteration sensitivity; the declared fixed-point stop remains enabled.

Metadata-only preparation, from a frozen source snapshot::

    python -m reviewer_revision_20260913.budget_study plan /scratch2/.../run
    python -m reviewer_revision_20260913.budget_study run /scratch2/.../run 0
    python -m reviewer_revision_20260913.budget_study summarize /scratch2/.../run
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, is_dataclass
import hashlib
from itertools import product
import json
import os
from pathlib import Path
import platform
import resource
import socket
import sys
import tempfile
import time
import traceback

import numpy as np


CODE_ROOT = Path(__file__).resolve().parents[2]
for _path in (CODE_ROOT / "simulation", CODE_ROOT / "sparse-smart/src",
              CODE_ROOT / "sparse-smart-v2/src"):
    sys.path.insert(0, str(_path))

from reviewer_revision_20260913 import design, runner


IMPLEMENTATION_VERSION = "budget-study-v1"
DEFAULT_BUDGETS = (100, 500, 1000, 2000)
DEFAULT_SEEDS = (2000, 2001, 2002, 2003, 2004)
CASE_IDS = ("reference_exact_n80", "containment_both30_n80",
            "fitted_source_n0100_n80", "sparse_stress_exact_n40")
INITIAL_PENALTIES = (.1, .3)
FACTOR_PENALTY_PAIRS = ((.01, .01), (.04, .04), (.16, .16))


def _integers(values, name, *, minimum=0):
    result = []
    for value in values:
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
            raise ValueError(f"{name} must contain integers >= {minimum}")
        if int(value) not in result:
            result.append(int(value))
    if not result:
        raise ValueError(f"{name} must not be empty")
    return result


def checkpoint_schedule(budget):
    budget = _integers([budget], "budget", minimum=1)[0]
    early = (0, 1, 2, 5, 10, 15, 20, 25, 50, 100, 150, 200)
    return sorted({t for t in early if t <= budget} | set(range(50, budget + 1, 50)) | {budget})


def budget_configuration(budget):
    base = runner.configuration(budget)
    fields = ("iterations", "validation_interval", "validation_iterations",
              "inverse_step", "max_backtracks", "stationarity_tol", "margins",
              "adaptive_anchors", "max_anchor_switches", "refinement_solver")
    config = {name: base[name] for name in fields}
    config.update(schema=1, iterations=int(budget), rrr_shortcut=False,
                  validation_patience=None, validation_min_iterations=0,
                  validation_min_relative_improvement=0.,
                  checkpoint_iterations=checkpoint_schedule(budget),
                  init_penalties=list(INITIAL_PENALTIES),
                  factor_penalty_pairs=[list(pair) for pair in FACTOR_PENALTY_PAIRS],
                  initialization_cache=False, partial_fit_selection=False,
                  refit_on_validation=False, theorem_certified=False,
                  study_scope="reduced_six_specification_full_cap_iteration_diagnostic")
    return config


def representative_cases():
    lookup = {case["case_id"]: case for case in design.case_manifest()}
    return [lookup[name] for name in CASE_IDS]


def candidate_specs(case):
    rank, source_rank = int(case["target_rank"]), int(case["source_rank"])
    free = max(rank, min(source_rank, 10))
    limits = ((int(case["p"]) - free) * rank, (int(case["q"]) - free) * rank)
    return [dict(rank=rank, source_rank=source_rank, free_directions=(free, free),
                 support_limits=limits, init_penalty=initial, penalty=penalty)
            for initial, penalty in product(INITIAL_PENALTIES, FACTOR_PENALTY_PAIRS)]


def make_model(spec, config):
    # The common factory preserves the main solver's constructor options. This
    # study explicitly adds every validation checkpoint to the saved trajectory.
    model = runner.make_v2(spec, config)
    model.checkpoint_iterations = tuple(config["checkpoint_iterations"])
    return model


def _snapshot_files():
    paths = list((CODE_ROOT / "sparse-smart-v2/src").rglob("*.py"))
    paths += list((CODE_ROOT / "sparse-smart/src").rglob("*.py"))
    paths += [Path(__file__), Path(runner.__file__), Path(design.__file__),
              Path(runner.competitors.__file__), Path(__file__).with_name("budget_pool.sbatch")]
    return {str(path.relative_to(CODE_ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(set(paths))}


def build_plan(root, *, seeds=DEFAULT_SEEDS, budgets=DEFAULT_BUDGETS, case_ids=None):
    """Write a complete declared plan without generating data or fitting models."""
    seed_ids = _integers(seeds, "seeds")
    budget_values = _integers(budgets, "budgets", minimum=1)
    if any(seed not in DEFAULT_SEEDS for seed in seed_ids):
        raise ValueError(f"seeds must be a subset of {DEFAULT_SEEDS}")
    if any(budget not in DEFAULT_BUDGETS for budget in budget_values):
        raise ValueError(f"budgets must be a subset of {DEFAULT_BUDGETS}")
    cases = representative_cases()
    if case_ids is not None:
        requested = list(dict.fromkeys(case_ids))
        lookup = {case["case_id"]: case for case in cases}
        if not requested or any(name not in lookup for name in requested):
            raise ValueError("case_ids must be a nonempty subset of the predeclared cases")
        cases = [lookup[name] for name in requested]
    tasks = [dict(case_index=i, seed=seed, budget=budget, expected_candidates=6)
             for i, _case in enumerate(cases) for seed in seed_ids for budget in budget_values]
    plan = dict(schema=1, implementation=IMPLEMENTATION_VERSION, study="iteration_budget_diagnostic",
                cases=cases, tasks=tasks, seeds=seed_ids, budgets=budget_values,
                configurations={str(budget): budget_configuration(budget) for budget in budget_values},
                candidate_libraries={case["case_id"]: candidate_specs(case) for case in cases},
                n_cases=len(cases), n_tasks=len(tasks), expected_fits=6 * len(tasks),
                code_sha256=_snapshot_files(), source_root=str(CODE_ROOT),
                output_namespace="budget_tasks", theorem_certified=False,
                budget_policy="fresh_independent_fit_per_candidate_and_budget_no_partial_fit_selection",
                runtime_policy="fresh_non_cached_fits_including_source_decomposition_and_initialization",
                selection_policy="within_budget_validation_selected_successful_candidates_no_truth_or_refit",
                stopping_policy="validation_patience_disabled_fixed_point_tolerance_retained",
                scope="reduced_grid_diagnostic_not_main_comparison_retuning")
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    if (root / "budget-plan.json").exists():
        raise FileExistsError("Use a fresh budget plan/output namespace")
    runner.atomic_json(root / "budget-plan.json", plan)
    (root / "budget-work-items.txt").write_text("".join(f"{i}\n" for i in range(len(tasks))))
    return plan


def _finite_coefficient(value, shape):
    if value is None or np.iscomplexobj(value):
        return None
    try:
        matrix = np.asarray(value, dtype=float)
    except (TypeError, ValueError):
        return None
    return matrix.copy() if matrix.shape == shape and np.isfinite(matrix).all() else None


def _record_dict(value):
    if value is None:
        return None
    return runner.jsonable(asdict(value) if is_dataclass(value) else value)


def _checkpoint_coefficient(source, checkpoint, *, selected=False):
    chart = checkpoint.selected_chart if selected else checkpoint.chart
    state = checkpoint.selected_state if selected else checkpoint.state
    left, values, right = chart.reconstruct(state)
    return ((source.left @ left) * values) @ (source.right @ right).T


def collect_outputs(model, spec, shape, exception=None):
    """Extract finite diagnostic arrays without accepting failed model outputs.

    This helper reads no truth, test sample, or future-budget result. A raw
    converged initializer has its own eligibility, independent of model.success_.
    Checkpoints of a failed complete-budget fit are explicitly diagnostic only.
    """
    arrays, checkpoints = {}, []
    good = exception is None and bool(getattr(model, "success_", False))
    selected = _finite_coefficient(getattr(model, "coefficient_", None), shape)
    terminal = _finite_coefficient(getattr(model, "last_coefficient_", None), shape)
    if selected is not None:
        arrays["selected_coefficient"] = selected
    if terminal is not None:
        arrays["terminal_coefficient"] = terminal
    extraction_errors = []
    initial = getattr(model, "initialization_", None)
    source = getattr(model, "source_", None)
    raw_initial_eligible = False
    if initial is not None and source is not None:
        try:
            raw = source.leading_left @ ((initial.P * initial.d) @ initial.Q.T) @ source.leading_right.T
            raw = _finite_coefficient(raw, shape)
            if raw is not None:
                arrays["raw_initializer"] = raw
                raw_initial_eligible = bool(initial.converged)
        except (AttributeError, ValueError, TypeError, np.linalg.LinAlgError) as error:
            extraction_errors.append(dict(stage="raw_initializer", error=str(error)))
    if source is not None:
        for iteration, checkpoint in sorted(getattr(model, "checkpoints_", {}).items()):
            row = dict(iteration=int(iteration), selected_iteration=int(checkpoint.selected_iteration),
                       status=checkpoint.status, termination_reason=checkpoint.termination_reason,
                       eligible_for_this_budget=bool(good), diagnostic_only=not good,
                       chart_epoch=getattr(checkpoint, "chart_epoch", None),
                       selected_chart_epoch=getattr(checkpoint, "selected_chart_epoch", None),
                       endpoint_record=_record_dict(getattr(checkpoint, "endpoint_record", None)))
            try:
                endpoint = _finite_coefficient(_checkpoint_coefficient(source, checkpoint), shape)
                prefix_selected = _finite_coefficient(_checkpoint_coefficient(source, checkpoint, selected=True), shape)
                if endpoint is None or prefix_selected is None:
                    raise ValueError("checkpoint coefficient is nonfinite or misshaped")
                endpoint_key, selected_key = f"checkpoint_{iteration:06d}", f"prefix_selected_{iteration:06d}"
                arrays[endpoint_key], arrays[selected_key] = endpoint, prefix_selected
                row.update(coefficient_key=endpoint_key, selected_coefficient_key=selected_key)
            except (AttributeError, ValueError, TypeError, np.linalg.LinAlgError) as error:
                row.update(extraction_error=str(error), eligible_for_this_budget=False)
                extraction_errors.append(dict(stage=f"checkpoint_{iteration}", error=str(error)))
            checkpoints.append(row)
    history = [_record_dict(record) for record in getattr(model, "history_", [])]
    terminal_record = _record_dict(getattr(model, "terminal_record_", None))
    eligible = good and selected is not None and terminal is not None
    status = getattr(model, "status_", "exception")
    if good and not eligible:
        extraction_errors.append(dict(stage="successful_output", error="successful model lacks finite selected/terminal coefficient"))
        status = "invalid_successful_output"
    changes = [row["objective_change"] for row in history
               if row.get("objective_change") is not None]
    row = dict(spec=spec, eligible=eligible, status=status,
               exception=exception, extraction_errors=extraction_errors,
               message=getattr(model, "message_", None),
               last_rejection=getattr(getattr(model, "result_", None), "last_rejection", None),
               method=getattr(model, "method_", None), n_iter=getattr(model, "n_iter_", None),
               selected_iteration=getattr(model, "selected_iteration_", None),
               termination_reason=getattr(model, "termination_reason_", None),
               optimization_converged=bool(getattr(model, "optimization_converged_", False)),
               selected_output_converged=bool(getattr(model, "converged_", False)),
               raw_initializer_eligible=raw_initial_eligible,
               raw_initializer_converged=bool(getattr(initial, "converged", False)),
               raw_initializer_kkt_residual=getattr(initial, "kkt_residual", None),
               raw_initializer_iterations=getattr(initial, "n_iter", None),
               raw_initializer_cost_scope="side_product_of_fresh_full_fit_no_separate_time_claim",
               retained_failed_arrays_are_diagnostic_only=not eligible,
               history=history, history_chart_epochs=getattr(model, "history_chart_epochs_", []),
               validation_history=getattr(model, "validation_history_", []),
               checkpoints=checkpoints, terminal_record=terminal_record,
               accepted_update_backtracks=sum(int(record.get("backtracks", 0)) for record in history),
               positive_reported_objective_change_count=sum(change > 0 for change in changes),
               max_reported_objective_change=max(changes) if changes else None,
               objective_comparison_scope="accepted_update_changes_with_chart_epochs_recorded_no_global_optimality_claim",
               fixed_point_diagnostic_scope="feasible_reference_hard_prox_mapping_not_chart_KKT_or_global_optimality",
               metadata=getattr(model, "metadata_", {}),
               numerical_work=getattr(model, "numerical_work_", {}))
    return runner.jsonable(row), arrays


def fit_candidate(data, spec, config, *, model_factory=make_model):
    """One fresh full call with no cache and no input from simulation truth."""
    from sparse_smart_v2 import ObservedSource
    model, exception = None, None
    began = time.perf_counter()
    try:
        model = model_factory(spec, config)
        model.fit(data["X"], data["Y"], source=ObservedSource(data["C0"]),
                  validation_data=(data["X_validation"], data["Y_validation"]))
    except Exception as error:
        exception = dict(type=type(error).__name__, message=str(error), traceback=traceback.format_exc())
    fitting_seconds = time.perf_counter() - began
    row, arrays = collect_outputs(model, spec, (data["X"].shape[1], data["Y"].shape[1]), exception)
    row["fitting_call_seconds"] = fitting_seconds
    row["fresh_fit"] = True
    row["initialization_cache_used"] = False
    return row, arrays


def choose_candidate(records, arrays, X_validation, Y_validation, *, initializer=False):
    """Recompute validation decisions; supplied scores and truth do not decide."""
    from sparse_smart.validation import validation_loss_difference
    xv, yv = np.asarray(X_validation, dtype=float), np.asarray(Y_validation, dtype=float)
    if xv.ndim != 2 or yv.ndim != 2 or len(xv) == 0 or len(xv) != len(yv) or not np.isfinite(xv).all() or not np.isfinite(yv).all():
        raise ValueError("validation arrays must be finite matrices with matching rows")
    audit, winner, best_prediction = [], None, None
    eligibility = "raw_initializer_eligible" if initializer else "eligible"
    key = "raw_initializer" if initializer else "selected_coefficient"
    seen_initializers = set()
    for index, (record, saved) in enumerate(zip(records, arrays, strict=True)):
        row = dict(candidate_index=index, eligible=False)
        initial_penalty = record["spec"]["init_penalty"]
        if initializer and initial_penalty in seen_initializers:
            row["exclusion_reason"] = "duplicate_initializer_penalty"
        elif not record.get(eligibility, False):
            row["exclusion_reason"] = "raw_initializer_ineligible" if initializer else "full_budget_fit_ineligible"
        else:
            coefficient = _finite_coefficient(saved.get(key), (xv.shape[1], yv.shape[1]))
            if coefficient is None:
                row["exclusion_reason"] = "missing_or_nonfinite_coefficient"
            else:
                try:
                    with np.errstate(over="raise", invalid="raise"):
                        prediction = xv @ coefficient
                        loss = float(np.mean((yv - prediction)**2))
                        difference = None if best_prediction is None else validation_loss_difference(
                            prediction, best_prediction, yv)
                    if not np.isfinite(loss):
                        raise FloatingPointError("nonfinite validation loss")
                except (ValueError, FloatingPointError) as error:
                    row.update(exclusion_reason="nonfinite_validation_arithmetic", error=str(error))
                else:
                    row.update(eligible=True, validation_mse=loss,
                               validation_loss_difference_to_incumbent=difference)
                    if initializer:
                        seen_initializers.add(initial_penalty)
                    if winner is None or difference < 0:
                        winner, best_prediction = index, prediction
        audit.append(row)
    return dict(success=winner is not None, selected_index=winner, coefficient_key=key,
                validation_mse=None if winner is None else audit[winner]["validation_mse"],
                audit=audit, selection_uses_truth=False)


def _atomic_npz(path, arrays):
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=".partial-", suffix=".npz", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_task(root, task_index):
    root = Path(root).resolve()
    if not os.environ.get("SLURM_JOB_ID") or not root.is_relative_to(Path("/scratch2")):
        raise RuntimeError("Budget-study fitting requires Slurm and a /scratch2 results root")
    plan_path = root / "budget-plan.json"
    plan = json.loads(plan_path.read_text())
    if not 0 <= task_index < len(plan["tasks"]):
        raise ValueError("task_index outside budget plan")
    if plan["code_sha256"] != _snapshot_files():
        raise RuntimeError("Code differs from frozen budget plan; use a fresh plan")
    digest = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    task = plan["tasks"][task_index]
    case = plan["cases"][task["case_index"]]
    config = plan["configurations"][str(task["budget"])]
    specs = plan["candidate_libraries"][case["case_id"]]
    directory = root / "budget_tasks" / f"task-{task_index:06d}"
    directory.mkdir(parents=True, exist_ok=True)
    result_path = directory / "result.json"
    if result_path.exists():
        previous = json.loads(result_path.read_text())
        if not previous.get("complete") or previous.get("plan_sha256") != digest:
            raise RuntimeError("Existing budget result has an incompatible identity")
        for name, expected in previous["evidence_sha256"].items():
            if hashlib.sha256((directory / name).read_bytes()).hexdigest() != expected:
                raise RuntimeError(f"Saved budget evidence differs: {name}")
        return previous
    lock_path = directory / ".running"
    lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(lock_fd, "w") as stream:
        stream.write(runner.canonical(dict(job=os.environ["SLURM_JOB_ID"], pid=os.getpid())) + "\n")
    started = time.perf_counter()
    try:
        bundle = design.generate_case(case, task["seed"])
        generation_seconds = time.perf_counter() - started
        data, evaluation = bundle["fit_data"], bundle["evaluation"]
        evidence = {}
        evidence["data.npz"] = _atomic_npz(directory / "data.npz", {k: v for k, v in data.items() if isinstance(v, np.ndarray)})
        evidence["truth.npz"] = _atomic_npz(directory / "truth.npz", evaluation)
        result = dict(schema=1, study=plan["study"], implementation=IMPLEMENTATION_VERSION,
                      complete=False, plan_sha256=digest, task=task, case=case, configuration=config,
                      metadata=bundle["metadata"], fit_data_fingerprint=runner.fingerprint(data),
                      truth_fingerprint=runner.fingerprint(evaluation), code_sha256=plan["code_sha256"],
                      generation_and_source_fit_seconds=generation_seconds,
                      runtime_environment=dict(host=socket.gethostname(), python=sys.version,
                          platform=platform.platform(), numpy=np.__version__, slurm_job_id=os.environ["SLURM_JOB_ID"],
                          threads={name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")}),
                      candidates=[], methods={}, evidence_sha256=evidence, theorem_certified=False)
        rows, saved_arrays = [], []
        for index, spec in enumerate(specs):
            row, arrays = fit_candidate(data, spec, config)
            row["candidate_index"] = index
            asset = f"candidate-{index:04d}.npz"
            evidence[asset] = _atomic_npz(directory / asset, arrays)
            row.update(coefficient_archive=asset, coefficient_keys=sorted(arrays))
            rows.append(row)
            saved_arrays.append(arrays)
            result["candidates"] = rows
            runner.atomic_json(directory / "partial.json", result)
            print(runner.canonical(dict(event="budget_candidate", case_id=case["case_id"], seed=task["seed"],
                                         budget=task["budget"], index=index, status=row["status"],
                                         eligible=row["eligible"], seconds=row["fitting_call_seconds"])), flush=True)
        # All selection occurs before truth-based diagnostic evaluation.
        selections = {"v2_reduced_grid": choose_candidate(rows, saved_arrays, data["X_validation"], data["Y_validation"]),
                      "raw_initializer_selected": choose_candidate(rows, saved_arrays, data["X_validation"], data["Y_validation"], initializer=True)}
        selected_arrays = {}
        for label, selected in selections.items():
            output = dict(selected)
            if selected["success"]:
                index = selected["selected_index"]
                coefficient = saved_arrays[index][selected["coefficient_key"]]
                selected_arrays[label] = coefficient
                output.update(parameters=rows[index]["spec"],
                              selected_iteration=0 if label == "raw_initializer_selected" else rows[index]["selected_iteration"],
                              metrics=runner.evaluate(coefficient, data, evaluation),
                              selected_candidate_full_fit_seconds=rows[index]["fitting_call_seconds"])
                if label == "v2_reduced_grid":
                    output.update(terminal_metrics=runner.evaluate(saved_arrays[index]["terminal_coefficient"], data, evaluation),
                                  optimization_converged=rows[index]["optimization_converged"],
                                  selected_output_converged=rows[index]["selected_output_converged"],
                                  terminal_record=rows[index]["terminal_record"])
            result["methods"][label] = output
        # Population coefficient/prediction risk is the trajectory diagnostic.
        # Test-sample multiplication is reserved for reported selected/terminal
        # fits, avoiding repeated large test predictions at every checkpoint.
        diagnostic_evaluation = {key: value for key, value in evaluation.items()
                                 if key not in ("X_test", "Y_test")}
        for row, arrays in zip(rows, saved_arrays, strict=True):
            row["diagnostic_metrics"] = {key: runner.evaluate(value, data, diagnostic_evaluation)
                                         for key, value in arrays.items()}
            row["diagnostic_metrics_scope"] = "coefficient_population_training_validation_no_test_sample"
        evidence["selected-coefficients.npz"] = _atomic_npz(directory / "selected-coefficients.npz", selected_arrays)
        result.update(complete=True, candidates=rows, expected_candidates=len(specs),
                      n_eligible=sum(row["eligible"] for row in rows),
                      n_raw_initializer_eligible=sum(row["raw_initializer_eligible"] for row in rows),
                      candidate_status_counts=dict(Counter(row["status"] for row in rows)),
                      failed_candidate_status_counts=dict(Counter(row["status"] for row in rows if not row["eligible"])),
                      fitting_call_seconds=sum(row["fitting_call_seconds"] for row in rows),
                      wall_seconds=time.perf_counter() - started,
                      max_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                      runtime_note="full fresh calls include repeated initialization/source decomposition; wall time also includes generation and evidence I/O",
                      later_budget_failures_censor_this_result=False,
                      partial_fit_selection=False)
        runner.atomic_json(result_path, result)
        return result
    except Exception as error:
        runner.atomic_json(directory / "task-error.json", dict(complete=False, plan_sha256=digest, task=task,
                           error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc(),
                           wall_seconds=time.perf_counter() - started))
        raise
    finally:
        lock_path.unlink(missing_ok=True)


def _mean_mcse(values):
    values = np.asarray(values, dtype=float)
    return dict(n=len(values), mean=float(np.mean(values)) if len(values) else None,
                monte_carlo_se=float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else None)


def summarize(root):
    """Audit complete task receipts and summarize; no fitting or imputation."""
    root = Path(root)
    plan_path = root / "budget-plan.json"
    plan = json.loads(plan_path.read_text())
    digest = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    results, issues = [], []
    for index, task in enumerate(plan["tasks"]):
        directory = root / "budget_tasks" / f"task-{index:06d}"
        path = directory / "result.json"
        if not path.exists():
            issues.append(dict(task_index=index, status="missing", task=task))
            continue
        result = json.loads(path.read_text())
        if not result.get("complete") or result.get("plan_sha256") != digest or result.get("task") != task or len(result.get("candidates", [])) != task["expected_candidates"]:
            issues.append(dict(task_index=index, status="invalid_completion", task=task))
            continue
        bad = [name for name, expected in result["evidence_sha256"].items()
               if not (directory / name).exists() or hashlib.sha256((directory / name).read_bytes()).hexdigest() != expected]
        if bad:
            issues.append(dict(task_index=index, status="evidence_hash_mismatch", files=bad))
            continue
        results.append(result)
    summaries, paired = [], []
    for case in plan["cases"]:
        for budget in plan["budgets"]:
            group = [row for row in results if row["case"]["case_id"] == case["case_id"] and row["task"]["budget"] == budget]
            successful = [row for row in group if row["methods"]["v2_reduced_grid"]["success"]]
            summaries.append(dict(case_id=case["case_id"], budget=budget, requested=len(plan["seeds"]),
                completed=len(group), successful=len(successful),
                population_prediction_excess=_mean_mcse([row["methods"]["v2_reduced_grid"]["metrics"]["population_prediction_excess"] for row in successful]),
                fitting_call_seconds=_mean_mcse([row["fitting_call_seconds"] for row in group]),
                selected_iterations=[row["methods"]["v2_reduced_grid"]["selected_iteration"] for row in successful],
                selected_candidate_terminal_stationarity_count=sum(row["methods"]["v2_reduced_grid"]["optimization_converged"] for row in successful),
                failed_candidate_status_counts=dict(sum((Counter(row["failed_candidate_status_counts"]) for row in group), Counter()))))
        ordered = sorted(plan["budgets"])
        for smaller, larger in zip(ordered[:-1], ordered[1:]):
            def index_budget(budget):
                return {row["task"]["seed"]: row for row in results if row["case"]["case_id"] == case["case_id"] and row["task"]["budget"] == budget}
            low, high = index_budget(smaller), index_budget(larger)
            complete = sorted(set(low) & set(high))
            mismatch = [seed for seed in complete if low[seed]["fit_data_fingerprint"] != high[seed]["fit_data_fingerprint"] or low[seed]["truth_fingerprint"] != high[seed]["truth_fingerprint"]]
            common = [seed for seed in complete if seed not in mismatch and low[seed]["methods"]["v2_reduced_grid"]["success"] and high[seed]["methods"]["v2_reduced_grid"]["success"]]
            differences = [high[seed]["methods"]["v2_reduced_grid"]["metrics"]["population_prediction_excess"] - low[seed]["methods"]["v2_reduced_grid"]["metrics"]["population_prediction_excess"] for seed in common]
            paired.append(dict(case_id=case["case_id"], lower_budget=smaller, upper_budget=larger,
                               common_success_seeds=common, fingerprint_mismatch_seeds=mismatch,
                               higher_minus_lower_prediction_risk=_mean_mcse(differences),
                               lower_success_higher_failure_seeds=[seed for seed in complete if low[seed]["methods"]["v2_reduced_grid"]["success"] and not high[seed]["methods"]["v2_reduced_grid"]["success"]]))
    summary = dict(schema=1, study=plan["study"], plan_sha256=digest, expected_tasks=len(plan["tasks"]),
                   audited_tasks=len(results), issues=issues, by_case_and_budget=summaries,
                   paired_budget_differences=paired, theorem_certified=False,
                   interpretation="descriptive_reduced_grid_budget_sensitivity_not_global_convergence_evidence")
    runner.atomic_json(root / "budget-summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    plan = sub.add_parser("plan")
    plan.add_argument("root", type=Path)
    plan.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    plan.add_argument("--budgets", default=",".join(map(str, DEFAULT_BUDGETS)))
    plan.add_argument("--cases", default=None)
    run = sub.add_parser("run")
    run.add_argument("root", type=Path)
    run.add_argument("task_index", type=int)
    summary = sub.add_parser("summarize")
    summary.add_argument("root", type=Path)
    args = parser.parse_args()
    if args.command == "plan":
        result = build_plan(args.root, seeds=[int(value) for value in args.seeds.split(",")],
                            budgets=[int(value) for value in args.budgets.split(",")],
                            case_ids=None if args.cases is None else args.cases.split(","))
        print(runner.canonical({key: result[key] for key in ("n_cases", "n_tasks", "expected_fits", "budgets", "seeds")}))
    elif args.command == "run":
        run_task(args.root, args.task_index)
    else:
        result = summarize(args.root)
        print(runner.canonical(dict(expected=result["expected_tasks"], audited=result["audited_tasks"], issues=len(result["issues"]))))


if __name__ == "__main__":
    main()
