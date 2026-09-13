"""Slurm-only simulation runner; truth is used only after model selection.

Each case is reproducible from its frozen manifest and seed. Candidate outcomes
are saved incrementally. Failures remain visible and never become substitutions.
The explicit target-RRR endpoint belongs to the declared v2 candidate library.
"""
from __future__ import annotations

import argparse
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
for path in (CODE_ROOT / "simulation", CODE_ROOT / "sparse-smart/src", CODE_ROOT / "sparse-smart-v2/src"):
    sys.path.insert(0, str(path))

from reviewer_revision_20260913 import design, competitors


def jsonable(value):
    if is_dataclass(value):
        return jsonable(asdict(value))
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def canonical(value):
    return json.dumps(jsonable(value), sort_keys=True, separators=(",", ":"), allow_nan=False)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".partial-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(canonical(value) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def fingerprint(arrays):
    digest = hashlib.sha256()
    for name, value in sorted(arrays.items()):
        if not isinstance(value, np.ndarray):
            continue
        a = np.ascontiguousarray(value)
        digest.update(canonical([name, a.shape, a.dtype.str]).encode())
        digest.update(a.tobytes())
    return digest.hexdigest()


def evaluate(coefficient, data, evaluation, intercept=None):
    """Called after validation selection; no model-fitting inputs from truth."""
    C = np.asarray(coefficient)
    delta = C - evaluation["C_star"]
    q = C.shape[1]
    intercept = np.zeros(q) if intercept is None else np.asarray(intercept)
    risk = float((np.sum(delta * (evaluation["Sigma_x"] @ delta)) + np.sum(intercept**2)) / q)
    result = dict(coefficient_rmse=float(np.sqrt(np.mean(delta ** 2))),
                  coefficient_squared_error=float(np.sum(delta ** 2)),
                  population_prediction_excess=risk,
                  validation_mse=float(np.mean((data["Y_validation"] - data["X_validation"] @ C - intercept) ** 2)),
                  training_mse=float(np.mean((data["Y"] - data["X"] @ C - intercept) ** 2)),
                  numerical_rank=int(np.linalg.matrix_rank(C)))
    if "X_test" in evaluation:
        result["test_mse"] = float(np.mean((evaluation["Y_test"] - evaluation["X_test"] @ C - intercept) ** 2))
    return result


def configuration(iterations=500):
    return dict(schema=1, init_penalties=[.003, .03, .1, .3, 1., 3.],
        penalties_u=[0., .001, .0025, .01, .04, .16], penalties_v=[0., .001, .0025, .01, .04, .16],
        iterations=iterations, validation_interval=50,
        validation_iterations=[1, 2, 5, 10, 15, 20, 25, 50, 100, 150, 200],
        validation_patience=300, validation_min_iterations=min(500, iterations),
        validation_min_relative_improvement=.001,
        inverse_step=20., max_backtracks=60, stationarity_tol=1e-6,
        margins=dict(d_lower=.001, d_upper=12., gap=.0001, anchor_min=.04, trial_radius=1.),
        adaptive_anchors=True, max_anchor_switches=16,
        refinement_solver="masked_chart_spectral_soft_hard", rrr_shortcut=True,
        main_variant="full_caps", tuning_uses_truth=False,
        rank_grid=[3, 5, 7], source_truncation_grid=[5, 10, 15],
        benchmark_ridges=[0., .0001, .001, .01, .1, 1., 10., 100.],
        mixture_weights=[0., .1, .25, .5, .75, .9, 1.],
        nuclear_penalties=[.001, .01, .1, 1., 3., 10.],
        park_lambdas=[.01,.03,.1,.3,1.,3.,10.],
        refit_on_validation=False)


def v2_specs(case, config, variant):
    rank = int(case.get("fitted_rank", case["target_rank"]))
    r0 = int(case["source_rank"])
    free = max(rank, min(r0, 10))
    full = ((case["p"] - free) * rank, (case["q"] - free) * rank)
    if variant in ("active_caps", "cap_control"):
        free = rank
        full = ((case["p"] - free) * rank, (case["q"] - free) * rank)
        cap_levels = [rank, 2 * rank, 4 * rank] if variant=="active_caps" else [max(full)]
        for initial, penalty, cap in product(config["init_penalties"], [0., .0025, .01, .04], cap_levels):
            yield dict(rank=rank, source_rank=r0, free_directions=(free, free),
                       init_penalty=initial, penalty=(penalty, penalty),
                       support_limits=tuple(min(cap, f) for f in full))
    else:
        # One direct endpoint per rank. It is independent of initialization.
        yield dict(rank=rank, source_rank=r0, free_directions=(free, free),
                   init_penalty=0., penalty=(0., 0.), support_limits=full)
        for initial, lu, lv in product(config["init_penalties"], config["penalties_u"], config["penalties_v"]):
            if lu == lv == 0:
                continue
            yield dict(rank=rank, source_rank=r0, free_directions=(free, free),
                       init_penalty=initial, penalty=(lu, lv), support_limits=full)


def make_v2(spec, config):
    from sparse_smart_v2 import SparseSMARTv2, Margins, PracticalCalibration
    return SparseSMARTv2(rank=spec["rank"], source_rank=spec["source_rank"],
        free_directions=spec["free_directions"], margins=Margins(**config["margins"]),
        calibration=PracticalCalibration(spec["init_penalty"], spec["penalty"],
                                         config["inverse_step"], spec["support_limits"]),
        iterations=config["iterations"], validation_interval=config["validation_interval"],
        validation_iterations=[t for t in config["validation_iterations"] if t <= config["iterations"]],
        validation_patience=config["validation_patience"],
        validation_min_iterations=config["validation_min_iterations"],
        validation_min_relative_improvement=config["validation_min_relative_improvement"],
        checkpoint_iterations=tuple(sorted(set([0, config["iterations"]]))),
        max_backtracks=config["max_backtracks"], stationarity_tol=config["stationarity_tol"],
        adaptive_anchors=config["adaptive_anchors"], max_anchor_switches=config["max_anchor_switches"],
        refinement_solver=config["refinement_solver"], rrr_shortcut=config["rrr_shortcut"])


def fit_v2(case, data, config, directory, variant="full_caps"):
    from sparse_smart_v2 import ObservedSource
    from sparse_smart.validation import validation_loss_difference
    source = ObservedSource(data["C0"])
    if variant in ("active_caps", "cap_control"):
        config = dict(config, rrr_shortcut=False)
    cache, records, winners = {}, [], {}
    start = time.perf_counter()
    # Independently select initialization, including when its later refinement
    # fails. Do not let final optimization eligibility censor this comparator.
    init_records, init_seen = [], set()
    for spec in v2_specs(case, config, variant):
        key = spec["init_penalty"]
        if key in init_seen or spec["init_penalty"] not in config["init_penalties"]:
            continue
        init_seen.add(key)
        initial_config = dict(config, iterations=0, validation_min_iterations=0, rrr_shortcut=False)
        initial_model, initial_error = None, None
        began = time.perf_counter()
        try:
            initial_model = make_v2(spec, initial_config)
            initial_model.fit(data["X"], data["Y"], source=source,
                validation_data=(data["X_validation"], data["Y_validation"]), _initialization_cache=cache)
        except Exception as exc:
            initial_error = str(exc)
        initial = getattr(initial_model,"initialization_",None)
        eligible = initial is not None and bool(initial.converged)
        row = dict(index=len(init_records), spec=spec, eligible=eligible, error=initial_error,
                   status=getattr(initial_model,"status_","exception"), elapsed_seconds=time.perf_counter()-began)
        if eligible:
            C = initial_model.source_.leading_left @ ((initial.P * initial.d) @ initial.Q.T) @ initial_model.source_.leading_right.T
            directory.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(directory / f"initializer-{row['index']:04d}.npz",coefficient=C)
            prediction = data["X_validation"] @ C
            row["validation_mse"] = float(np.mean((data["Y_validation"] - prediction)**2))
            old = winners.get("initializer_only")
            if old is None or validation_loss_difference(prediction, old["prediction"], data["Y_validation"]) < 0:
                winners["initializer_only"] = dict(coefficient=C, prediction=prediction,
                    candidate_index=row["index"], spec=spec, selected_iteration=0,
                    method="lasso_svd_initializer", history=[])
        init_records.append(row)
    initialization_seconds = sum(row['elapsed_seconds'] for row in init_records)
    atomic_json(directory / "initializers.json",dict(complete=True,candidates=init_records,
                total_seconds=initialization_seconds))
    for index, spec in enumerate(v2_specs(case, config, variant)):
        began = time.perf_counter()
        model, err = None, None
        try:
            model = make_v2(spec, config)
            model.fit(data["X"], data["Y"], source=source,
                      validation_data=(data["X_validation"], data["Y_validation"]),
                      _initialization_cache=cache)
        except Exception as exc:
            err = dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc())
        good = err is None and bool(getattr(model, "success_", False))
        record = dict(index=index, spec=spec, eligible=good, exception=err,
            elapsed_seconds=time.perf_counter()-began, status=getattr(model, "status_", "exception"),
            method=getattr(model, "method_", None), n_iter=getattr(model, "n_iter_", None),
            selected_iteration=getattr(model, "selected_iteration_", None),
            termination_reason=getattr(model, "termination_reason_", None),
            validation_history=getattr(model, "validation_history_", []),
            metadata=getattr(model, "metadata_", {}),
            numerical_work=getattr(model, "numerical_work_", {}))
        if good:
            coefficient = model.coefficient_.copy()
            if not np.isfinite(coefficient).all():
                raise ValueError("Successful v2 result is nonfinite")
            pred = data["X_validation"] @ coefficient
            record["validation_mse"] = float(np.mean((data["Y_validation"]-pred)**2))
            for label in (["v2", "v2_transfer_only"] if record["method"] != "target_rrr" else ["v2"]):
                old = winners.get(label)
                if old is None or validation_loss_difference(pred, old["prediction"], data["Y_validation"]) < 0:
                    winners[label] = dict(coefficient=coefficient, prediction=pred,
                        candidate_index=index, spec=spec, selected_iteration=record["selected_iteration"],
                        method=record["method"], history=jsonable(getattr(model, "history_", [])),
                        terminal_coefficient=getattr(model, "last_coefficient_", coefficient).copy())
            np.savez_compressed(directory / f"candidate-{index:04d}.npz", coefficient=coefficient)
        records.append(record)
        atomic_json(directory / "candidates.json", dict(variant=variant, candidates=records, complete=False))
        print(canonical(dict(event="v2_candidate", case_id=case["case_id"], variant=variant,
                             index=index, status=record["status"], seconds=record["elapsed_seconds"])), flush=True)
    elapsed = sum(row['elapsed_seconds'] for row in records) + sum(row['elapsed_seconds'] for row in init_records)
    atomic_json(directory / "candidates.json", dict(variant=variant, candidates=records, complete=True,
                expected_n_candidates=len(list(v2_specs(case,config,variant))),
                specs_sha256=hashlib.sha256(canonical(list(v2_specs(case,config,variant))).encode()).hexdigest(),
                total_seconds=elapsed, n_eligible=sum(r["eligible"] for r in records)))
    if "initializer_only" in winners:
        winners["initializer_only"]["own_seconds"] = initialization_seconds
        winners["initializer_only"]["own_n_candidates"] = len(init_records)
        winners["initializer_only"]["own_n_eligible"] = sum(row['eligible'] for row in init_records)
    return winners, records, elapsed


def run_case(root, task_index):
    root = Path(root).resolve()
    if not os.environ.get("SLURM_JOB_ID") or not str(root).startswith("/scratch2/"):
        raise RuntimeError("Production simulations require Slurm and a /scratch2 results root")
    plan = json.loads((root / "plan.json").read_text())
    task = plan["tasks"][task_index]
    case = plan["cases"][task["case_index"]]
    directory = root / "tasks" / f"task-{task_index:06d}"
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "result.json").exists():
        result = json.loads((directory / "result.json").read_text())
        if result.get("complete") and result.get("plan_sha256") == hashlib.sha256((root / "plan.json").read_bytes()).hexdigest():
            print("Already complete with this plan", flush=True)
            return
        raise RuntimeError("Existing result does not match completed task identity")
    began = time.perf_counter()
    bundle = design.generate_case(case, task["seed"])
    generation_seconds=time.perf_counter()-began
    data, evaluation = bundle["fit_data"], bundle["evaluation"]
    np.savez_compressed(directory / "data.npz", **{k:v for k,v in data.items() if isinstance(v, np.ndarray)})
    np.savez_compressed(directory / "truth.npz", **{k:v for k,v in evaluation.items() if isinstance(v, np.ndarray)})
    result = dict(schema=1, task=task, case=case, metadata=bundle["metadata"],
        plan_sha256=hashlib.sha256((root / "plan.json").read_bytes()).hexdigest(),
        fit_data_fingerprint=fingerprint(data), truth_fingerprint=fingerprint(evaluation),
        slurm_job_id=os.environ["SLURM_JOB_ID"], host=socket.gethostname(),
        python=sys.version, platform=platform.platform(), methods={}, complete=False)
    result['generation_and_source_fit_seconds']=generation_seconds
    from sparse_smart_v2 import ObservedSource, prepare_source
    source_started=time.perf_counter()
    observed=prepare_source(ObservedSource(data['C0']),p=case['p'],q=case['q'],source_rank=case['source_rank'])
    result['common_source_decomposition_seconds']=time.perf_counter()-source_started
    # Fitting routines receive training/validation data only, except the labelled oracle.
    coefficient_arrays = {}
    method_specs = benchmark_specs(case, plan["configuration"])
    for name, candidates in method_specs.items():
        started = time.perf_counter()
        try:
            selected = competitors.select_competitor(data["X"], data["Y"],
                data["X_validation"], data["Y_validation"], candidates,
                source_coefficient=data["C0"],
                observed_left=observed.left, observed_right=observed.right,
                oracle_left=evaluation["U0"] if name == "oracle_subspace_rrr" else None,
                oracle_right=evaluation["V0"] if name == "oracle_subspace_rrr" else None)
            # Interface normalized after competitor module integration.
            fit = selected.fit
            coefficient_arrays[name] = fit.coefficient
            result["methods"][name] = dict(success=True, metrics=evaluate(fit.coefficient, data, evaluation),
                parameters=fit.parameters, diagnostics=fit.diagnostics,
                selected_index=selected.selected_index,
                audit=jsonable(selected.candidate_results), total_seconds=time.perf_counter()-started)
        except Exception as exc:
            result["methods"][name] = dict(success=False, error=str(exc), traceback=traceback.format_exc(),
                                            audit=jsonable(getattr(exc,"candidate_results",[])),
                                            total_seconds=time.perf_counter()-started)
        atomic_json(directory / "partial.json", result)
    if case.get('source_mode')=='fitted':
        from reviewer_revision_20260913.published_park import fit_park_validation
        name='park_two_stage_nr_external_validation'
        started=time.perf_counter()
        try:
            selected=fit_park_validation(data['X'],data['Y'],data['X_validation'],data['Y_validation'],
                data['source_X'],data['source_Y'],
                lambda_w=plan['configuration']['park_lambdas'],lambda_delta=plan['configuration']['park_lambdas'],
                max_iter=1000,tolerance=1e-6,require_convergence=True,timeout_seconds=1800)
            fit=selected.fit
            coefficient_arrays[name]=fit.coefficient
            coefficient_arrays[name+'__intercept']=fit.intercept
            result['methods'][name]=dict(success=True,
                metrics=evaluate(fit.coefficient,data,evaluation,fit.intercept),diagnostics=fit.diagnostics,
                candidates=jsonable(selected.candidate_results),selected_index=selected.selected_index,
                validation_score=2*selected.validation_loss/case['q'],
                intercept=fit.intercept,total_seconds=time.perf_counter()-started)
        except Exception as exc:
            result['methods'][name]=dict(success=False,error=str(exc),
                candidates=jsonable(getattr(exc,'diagnostics',{}).get('candidate_results',[])),
                diagnostics=getattr(exc,'diagnostics',{}),traceback=traceback.format_exc(),total_seconds=time.perf_counter()-started)
    for variant in task.get("variants", ["full_caps"]):
        winners, records, elapsed = fit_v2(case, data, plan["configuration"], directory/variant, variant)
        for name, winner in winners.items():
            label = name if variant == "full_caps" else name + "_" + variant
            coefficient_arrays[label] = winner["coefficient"]
            result["methods"][label] = dict(success=True,
                metrics=evaluate(winner["coefficient"], data, evaluation),
                parameters=winner["spec"], selected_iteration=winner["selected_iteration"],
                selected_method=winner["method"], candidate_index=winner["candidate_index"],
                total_seconds=winner.get("own_seconds",elapsed), n_candidates=winner.get("own_n_candidates",len(records)),
                n_eligible=winner.get("own_n_eligible",sum(r["eligible"] for r in records)),
                history=winner["history"], theorem_certified=False)
            if "terminal_coefficient" in winner:
                result["methods"][label]["terminal_metrics"] = evaluate(winner["terminal_coefficient"], data, evaluation)
        for name in ("v2", "v2_transfer_only", "initializer_only"):
            if name not in winners:
                label=name if variant=="full_caps" else name+"_"+variant
                result["methods"][label] = dict(success=False,
                    error="No eligible candidate", n_candidates=len(records), total_seconds=elapsed)
    np.savez_compressed(directory / "coefficients.npz", **coefficient_arrays)
    result.update(complete=True, elapsed_seconds=time.perf_counter()-began,
                  peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    atomic_json(directory / "result.json", result)
    print(canonical(dict(event="case_complete", task=task_index, seconds=result["elapsed_seconds"])), flush=True)


def benchmark_specs(case, config):
    rank = int(case.get("fitted_rank", case["target_rank"]))
    sr = int(case["source_rank"])
    grids = {}
    for method in ("target_rrr", "oracle_subspace_rrr", "source_subspace_rrr"):
        grids[method] = [dict(method=method, rank=rank, source_rank=sr)]
    for method in ("target_ridge_rrr", "source_subspace_ridge_rrr", "ridge_to_source"):
        grids[method] = [dict(method=method, rank=rank, source_rank=sr, ridge=lam)
                         for lam in config["benchmark_ridges"] if method != "ridge_to_source" or lam > 0]
    grids["source_target_mixture"] = [dict(method="source_target_mixture", rank=rank,
        ridge=lam, alpha=alpha) for lam, alpha in product(config["benchmark_ridges"], config["mixture_weights"])]
    grids["nuclear_contrast"] = [dict(method="nuclear_contrast", rank=rank, nuclear_penalty=lam)
                                 for lam in config["nuclear_penalties"]]
    return grids


def build_plan(root, seeds, iterations, pilot=False, runtime=False):
    root = Path(root)
    cases = design.runtime_manifest() if runtime else design.case_manifest()
    if pilot:
        # Explicit representative endpoints, selected from metadata rather than outcomes.
        seen, chosen = set(), []
        for c in cases:
            family = c["family"]
            if family not in seen:
                seen.add(family)
                same = [x for x in cases if x["family"]==family and x["n_train"]==c["n_train"]]
                chosen.append(same[-1] if family in ("containment","diffuse_alignment","source_internal_gap","target_internal_gap") else c)
        chosen += [c for c in cases if c["family"]=="source_truncation" and c["source_rank"]==5 and c["n_train"]==80]
        cases = chosen
    tasks = [dict(case_index=i, seed=seed, variants=["full_caps"] +
                  (["active_caps", "cap_control"] if "sparse" in c["family"] or "sparsity" in c["family"] else []))
             for i,c in enumerate(cases) for seed in seeds]
    for task in tasks:
        task["expected_methods"] = list(benchmark_specs(cases[task["case_index"]], configuration(iterations)))
        if cases[task['case_index']].get('source_mode')=='fitted':
            task['expected_methods'].append('park_two_stage_nr_external_validation')
        for variant in task["variants"]:
            task["expected_methods"] += [name if variant=="full_caps" else name+"_"+variant
                                         for name in ("v2","v2_transfer_only","initializer_only")]
    plan = dict(schema=1, cases=cases, tasks=tasks, configuration=configuration(iterations),
                purpose="Reviewer numerical revision", pilot=pilot, runtime_study=runtime,seed_ids=seeds,
                source_root=str(CODE_ROOT), n_cases=len(cases), n_tasks=len(tasks))
    root.mkdir(parents=True, exist_ok=True)
    if (root/"plan.json").exists():
        raise FileExistsError("Use a new results root for each plan")
    atomic_json(root/"plan.json", plan)
    (root/"work-items.txt").write_text("".join(f"{i}\n" for i in range(len(tasks))))
    print(canonical(dict(n_cases=len(cases), n_tasks=len(tasks))))


def main():
    parser=argparse.ArgumentParser()
    sub=parser.add_subparsers(dest="command", required=True)
    p=sub.add_parser("plan"); p.add_argument("root"); p.add_argument("--seeds", default="0,1")
    p.add_argument("--iterations", type=int, default=500); p.add_argument("--pilot", action="store_true")
    p.add_argument("--runtime",action="store_true")
    p=sub.add_parser("run"); p.add_argument("root"); p.add_argument("task", type=int)
    args=parser.parse_args()
    if args.command=="plan":
        build_plan(args.root, [int(s) for s in args.seeds.split(",")], args.iterations, args.pilot,args.runtime)
    else:
        run_case(args.root,args.task)


if __name__=="__main__":
    main()
