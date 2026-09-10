"""Audit external-validation SparseSMART on every legacy paper-grid setting.

Only the data generator is run: hashes are checked against fresh legacy training
and independent tuning observations, cached per distinct generating setting.
No estimator is fitted. Existing summaries and result files are read only.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import asdict
from itertools import product
import json
import math
from pathlib import Path

import numpy as np

import external_validation_data
import run_sparse_smart_external as runner
from run_restricted_rrr import DEFAULT_SEED_FILE, experiment_settings, load_experiment_seeds
from run_sparse_smart import _digest_json, _json_value
from summarize_sparse_smart import DEFAULT_REFERENCE, _stats
from summarize_sparse_smart_external import _config, _expected_configuration, _fingerprint, _paper_provenance
from summarize_sparse_smart_tuned import (_finite, _integer, _paper_reference, _require,
    _validate_validation_history, _candidate_grid, _candidate_budget, _selected_budget, _validate_selected_budget,
    _validate_checkpoint_metadata)

HERE = Path(__file__).resolve().parent
MODEL_DIMS = ((100, 50), (150, 100), (300, 200))


def _data_audit(setting, seed, config, cache, generator):
    # Fitted target/source ranks do not enter the original generating process.
    key = (setting.n, setting.p, setting.q, setting.sigma0, int(seed),
           config.n_validation, config.validation_seed_tag)
    if key not in cache:
        data = external_validation_data.generate_external_validation(
            n_train=setting.n, p=setting.p, q=setting.q, sigma0=setting.sigma0,
            random_seed=int(seed), n_validation=config.n_validation, sigma=.5,
            r_star=5, r0_star=10, seed_tag=config.validation_seed_tag, generate_data_fn=generator)
        cache[key] = dict(
            training_observed_input_fingerprint=runner.old_runner._array_fingerprint(data, ("X", "Y", "C0")),
            validation_observed_input_fingerprint=runner.old_runner._array_fingerprint(data, ("X_validation", "Y_validation")),
            evaluation_truth_fingerprint=runner.old_runner._array_fingerprint(data, ("C_star",)),
            validation_seed_metadata=_json_value(data["validation_seed_metadata"]),
            _truth=np.asarray(data["C_star"]).copy(),
        )
    return cache[key]


def _validate_selection(record, setting, config, resolved):
    status, reason = record["status"], setting.inapplicability_reason()
    _require(status in ("complete", "failed", "all_candidates_failed", "inapplicable"), "Unknown run status")
    _require(type(record["success"]) is bool and record["success"] == (status == "complete"),
             "Inconsistent success flag")
    _require(type(record["applicable"]) is bool and record["applicable"] == (reason is None),
             "Applicability mismatch")
    if reason is not None:
        _require(status == "inapplicable" and record["failure_reason"] == reason,
                 "Invalid dimensions must be marked inapplicable without clamping")
        _require(record["selection_history"] == [] and record["fit_errors"] == []
                 and record["C_hat"] is None and record["avg_err"] is None
                 and record["best_params"] is None and record["split"] is None
                 and record["n_iter"] == 0 and record["all_candidates_failed"] is False,
                 "Inapplicable record contains fitted or successful output")
        _validate_checkpoint_metadata(record, config, 1)  # There are no candidate positions.
        return
    _require(status != "inapplicable", "Applicable setting is marked inapplicable")
    candidates = record["selection_history"]
    grid = _candidate_grid(config, resolved["support_limits"])
    _require(len(candidates) <= len(grid), "Too many candidate records")
    if status in ("complete", "all_candidates_failed"):
        _require(len(candidates) == len(grid), "Incomplete candidate search")
    for index, candidate in enumerate(candidates):
        budget, params = grid[index]
        _require(_candidate_budget(candidate, config) == budget, "Candidate budget order mismatch")
        _require(candidate["candidate_id"] == index and candidate["params"] == params,
                 "Candidate identity or tuning grid mismatch")
        _require(type(candidate["success"]) is bool, "Invalid candidate success flag")
        _integer(candidate["n_iter"], "candidate iterations", budget)
        if candidate["success"] and candidate["termination_reason"] == "max_iterations":
            _require(candidate["n_iter"] == budget, "Candidate stopped below its iteration budget")
        if candidate["success"]:
            _require(candidate["status"] in ("completed", "converged"), "Failed candidate marked successful")
            _validate_validation_history(candidate)
        else:
            _require(candidate["validation_mse"] is None,
                     "Failed candidate must not have an eligible validation score")
    _require(record["fit_errors"] == [c for c in candidates if not c["success"]],
              "fit_errors does not match candidate failures")
    _validate_checkpoint_metadata(record, config,
                                 len(grid) // len(config.iteration_budgets or (config.iterations,)))
    eligible = [c for c in candidates if c["success"]]
    if not record["success"]:
        _require(record["avg_err"] is None, "Failed partial error cannot count as successful")
        if status == "all_candidates_failed":
            _require(record["all_candidates_failed"] is True and not eligible
                     and record["best_params"] is None and record["C_hat"] is None,
                     "All-candidates-failed record contains a selected model")
        return
    _require(record["all_candidates_failed"] is False and eligible, "No eligible winner")
    winner = min(eligible, key=lambda c: c["validation_mse"])
    winner_budget = _validate_selected_budget(record, winner, config)
    _require(record["best_params"] == winner["params"], "Selected candidate is not the validation winner")
    _require(record["selected_iteration"] == winner["selected_iteration"]
             and record["n_iter"] == winner["n_iter"], "Winner iteration mismatch")
    _require(math.isclose(_finite(record["validation_loss"], "selected validation loss"),
                         winner["validation_mse"], rel_tol=1e-10, abs_tol=1e-12), "Winner score mismatch")
    _require(record["validation_history"] == winner["validation_history"], "Winner validation history mismatch")
    _finite(record["avg_err"], "completed coefficient error")
    _finite(record["initial_avg_err"], "initializer coefficient error")
    coefficient = np.asarray(record["C_hat"], dtype=float)
    _require(coefficient.shape == (setting.p, setting.q) and np.isfinite(coefficient).all(),
             "Invalid completed coefficient shape or entries")
    history = record["history"]
    _require([item["iteration"] for item in history] == list(range(record["n_iter"]+1)),
             "Incomplete selected optimization history")
    selected = history[record["selected_iteration"]]
    for side, maximum, cap in zip(("u", "v"), resolved["actual_complement_counts"],
                                 record["best_params"]["support_limits"]):
        indices = record["selected_supports"][side]
        _require(all(type(i) is int and 0 <= i < maximum for i in indices)
                 and indices == sorted(set(indices)) and len(indices) <= cap
                 and len(indices) == selected[f"support_{side}"], "Selected support mismatch")
    diagnostics = record["diagnostics"]
    _require(type(diagnostics["optimization_converged"]) is bool
             and type(diagnostics["selected_converged"]) is bool, "Missing convergence diagnostics")
    _require(diagnostics["optimization_converged"] == (record["termination_reason"] == "stationarity"),
             "Optimizer convergence/termination mismatch")
    if record["termination_reason"] == "max_iterations":
        _require(record["n_iter"] == winner_budget, "Iteration limit recorded below configured budget")


def validate_record(record, path, *, setting, model_id, exp_id, seed_id, random_seed,
                    data_cache, generate_data_fn):
    model, experiment = f"model{model_id+1}", f"exp{exp_id+1}"
    _require(record["schema_version"] == 1 and record["method"] == "SparseSMARTExternal",
             "Unexpected external result schema or method")
    _require(record["model"] == model and record["experiment"] == experiment
             and record["rd_seed_id"] == seed_id and record["random_seed"] == int(random_seed),
             "Cell identity or saved random seed mismatch")
    _require(record["setting"] == asdict(setting), "Setting does not match the legacy grid")
    expected = runner.result_path(Path("."), model=model, experiment=experiment,
                                  setting=setting, seed_id=seed_id)
    _require(path.name == expected.name and path.parent.name == experiment
             and path.parent.parent.name == model, "Result filename or directory mismatch")
    config = _config(record["configuration"]["runner"])
    _require(config.n_validation == 100, "This paper-grid pilot requires 100 additional validation rows")
    resolved = _expected_configuration(setting, config, record["configuration"])
    _require(record["configuration"] == resolved, "Resolved configuration mismatch")
    arguments = dict(n=setting.n, p=setting.p, q=setting.q, sigma0=setting.sigma0,
                     sigma=.5, r_star=5, r0_star=10, random_seed=int(random_seed))
    _require(record["generator_arguments"] == arguments, "Generator arguments mismatch")
    identity = {key:record[key] for key in ("schema_version", "method", "model", "experiment", "rd_seed_id",
        "random_seed", "setting", "configuration", "generator_arguments")}
    _require(record["configuration_fingerprint"] == _digest_json(identity), "Configuration fingerprint mismatch")
    for key in ("configuration_fingerprint", "implementation_fingerprint", "training_observed_input_fingerprint",
                "validation_observed_input_fingerprint", "evaluation_truth_fingerprint", "input_fingerprint"):
        _fingerprint(record[key], key)
    _require(record["input_fingerprint"] == _digest_json(dict(
        training_observed=record["training_observed_input_fingerprint"],
        validation_observed=record["validation_observed_input_fingerprint"],
        evaluation_truth=record["evaluation_truth_fingerprint"])), "Combined input fingerprint mismatch")
    _require(record["n_train"] == setting.n and record["n_validation"] == 100
             and record["all_training_rows_used"] is True and record["training_matches_legacy"] is True
             and record["refit_on_all_data"] is False, "Training/validation counts or reuse flags mismatch")
    expected_seed = dict(seed_sequence_entropy=[int(random_seed), config.validation_seed_tag],
        bit_generator="PCG64", covariance="AR1", covariance_rho=.5, design_factorization="cholesky",
        noise_std=.5, conditional_on_same_coefficient=True, source_reused=True)
    _require(record["validation_seed_metadata"] == expected_seed, "Independent validation seed metadata mismatch")
    expected_split = dict(mode="independent_external_validation", n_train=setting.n, n_validation=100,
        train_indices=None, validation_indices=None, all_training_rows_used=True, refit_on_all_data=False)
    if record["split"] is not None:
        _require(record["split"] == dict(**expected_split, fingerprint=_digest_json(expected_split)),
                 "External split metadata mismatch")
    elif record["status"] in ("complete", "all_candidates_failed"):
        raise ValueError("Missing split for a completed tuning search")
    _require(record["theorem_certified"] is False, "Empirical result cannot be a theorem certificate")
    _require(record["source_check_mode"] == ("strict" if config.strict_source_check else "empirical"),
             "Source-check mode mismatch")
    _validate_selection(record, setting, config, resolved)
    actual = _data_audit(setting, random_seed, config, data_cache, generate_data_fn)
    for key, value in actual.items():
        if key == "_truth":
            continue
        _require(record[key] == value, f"Regenerated {key} differs from the saved result")
    if record["success"]:
        error = np.linalg.norm(np.asarray(record["C_hat"])-actual["_truth"],ord="fro") / np.sqrt(setting.p*setting.q)
        _require(math.isclose(record["avg_err"],float(error),rel_tol=1e-10,abs_tol=1e-12),
                 "Saved coefficient error disagrees with regenerated truth and the paper metric")
    return config


def _duplicates(records, models, seeds):
    verified = []
    for model_id in models:
        base_n = experiment_settings(model_id, 0)[0].n
        labels = (("exp1", f"n={base_n}"), ("exp2", "r=5"),
                  ("exp3", "rs=10"), ("exp4", "sigma0=0.01"))
        for seed in seeds:
            available = [(experiment, suffix, records[(model_id, experiment, suffix, seed)])
                         for experiment, suffix in labels
                         if (model_id, experiment, suffix, seed) in records]
            if len(available) < 2:
                continue
            reference = available[0][2]
            for _, _, record in available[1:]:
                for key in ("training_observed_input_fingerprint", "validation_observed_input_fingerprint",
                            "evaluation_truth_fingerprint", "input_fingerprint", "configuration", "status",
                            "success", "best_params", "selected_iteration", "n_iter", "termination_reason"):
                    _require(record[key] == reference[key], f"Duplicate default cell disagrees: model{model_id+1} seed{seed} {key}")
                for key in ("avg_err", "initial_avg_err", "validation_loss"):
                    a,b = record[key],reference[key]
                    _require((a is None and b is None) or (a is not None and b is not None
                             and math.isclose(a,b,rel_tol=1e-10,abs_tol=1e-12)), f"Duplicate default cell disagrees: {key}")
                if reference["C_hat"] is not None:
                    _require(np.allclose(record["C_hat"],reference["C_hat"],rtol=1e-10,atol=1e-12),
                             "Duplicate default cell disagrees: coefficient")
            verified.append(dict(model_id=model_id, seed_id=seed, compared_cells=[f"{e}:{s}" for e,s,_ in available]))
    return verified


def _counts(values):
    return json.dumps(dict(Counter(values)), sort_keys=True)


def summarize(result_root, reference_path=DEFAULT_REFERENCE, *, model_ids=(0,1,2),
              experiments=(0,1,2,3), seed_ids=(0,1,2,3,4), seed_file=DEFAULT_SEED_FILE,
              generate_data_fn=None):
    """Return aggregates, paper rows and audit after cached input regeneration."""
    model_ids, experiments, seed_ids = tuple(model_ids),tuple(experiments),tuple(seed_ids)
    for values, count, name in ((model_ids,3,"model IDs"),(experiments,4,"experiments"),(seed_ids,100,"seed IDs")):
        _require(values and len(set(values)) == len(values)
                 and all(type(v) is int and 0 <= v < count for v in values), f"Invalid {name}")
    seeds = load_experiment_seeds(seed_file)
    _require(max(seed_ids) < len(seeds), "Requested seed is unavailable")
    settings = {(m,f"exp{e+1}",s.suffix):(e,s) for m in model_ids for e in experiments for s in experiment_settings(m,e)}
    records, cache, configs = {}, {}, set()
    implementations = {True:set(),False:set()}
    generator = generate_data_fn
    for m in model_ids:
        for e in experiments:
            paths = sorted((Path(result_root)/f"model{m+1}"/f"exp{e+1}").glob("SparseSMARTExternal_result_*.json"))
            for path in paths:
                try:
                    record = json.loads(path.read_text())
                    if record["rd_seed_id"] not in seed_ids:
                        continue
                    key = (m,record["experiment"],record["setting"]["suffix"],record["rd_seed_id"])
                    _require(key[:3] in settings and key not in records, "Unknown or duplicate result cell")
                    exp_id,setting = settings[key[:3]]
                    if generator is None:
                        generator = runner.old_runner._load_generator()
                    config = validate_record(record,path,setting=setting,model_id=m,exp_id=exp_id,
                        seed_id=key[-1],random_seed=seeds[key[-1]],data_cache=cache,generate_data_fn=generator)
                    records[key] = record
                    configs.add(json.dumps(_json_value(asdict(config)),sort_keys=True))
                    implementations[record["applicable"]].add(record["implementation_fingerprint"])
                except (KeyError,TypeError,ValueError,OverflowError) as error:
                    raise ValueError(f"Invalid external-grid result {path}: {error}") from error
    _require(len(configs) <= 1, "Cannot pool different tuning configurations")
    _require(all(len(values) <= 1 for values in implementations.values()),
             "Cannot pool different implementations within an applicability class")
    config = json.loads(next(iter(configs))) if configs else None
    duplicates = _duplicates(records,model_ids,seed_ids)
    paper = [r for m in model_ids for r in _paper_reference(reference_path,m,experiments)]
    pdfs = _paper_provenance(reference_path)
    rows = []
    for (m,experiment,suffix),(exp_id,setting) in settings.items():
        cell = [records[(m,experiment,suffix,seed)] for seed in seed_ids if (m,experiment,suffix,seed) in records]
        successful = [r for r in cell if r["success"]]
        candidates = [c for r in cell for c in r["selection_history"]]
        mean,se = _stats([r["avg_err"] for r in successful])
        initial_mean,_ = _stats([r["initial_avg_err"] for r in successful])
        applicable = setting.inapplicability_reason() is None
        row = dict(model=f"model{m+1}",model_id=m,p=setting.p,q=setting.q,experiment=experiment,
            setting=suffix,x=float(suffix.split("=",1)[1]),n_train=setting.n,n_validation=100,
            fitted_rank=setting.target_rank,fitted_source_rank=setting.source_rank,
            applicable=applicable,inapplicability_reason=setting.inapplicability_reason(),
            expected_runs=len(seed_ids),recorded_runs=len(cell),complete=len(successful),
            failed=sum(r["status"] in ("failed","all_candidates_failed") for r in cell),
            all_candidates_failed=sum(r["status"] == "all_candidates_failed" for r in cell),
            inapplicable=sum(r["status"] == "inapplicable" for r in cell),missing=len(seed_ids)-len(cell),
            final_mean=mean,final_se=se,initial_mean=initial_mean,
            mean_is_conditional_on_success=any(r["applicable"] and not r["success"] for r in cell),
            chosen_initializer=sum(r["selected_iteration"] == 0 for r in successful),
            optimization_converged=sum(r["diagnostics"]["optimization_converged"] for r in successful),
            selected_converged=sum(r["diagnostics"]["selected_converged"] for r in successful),
            max_iterations=sum(r["termination_reason"] == "max_iterations" for r in successful),
            selected_near_budget=sum(r["selected_iteration"] >= .9*_selected_budget(r) for r in successful) if config else 0,
            selected_budget_counts=_counts(_selected_budget(r) for r in successful),
            selected_at_budget=sum(r["selected_iteration"] == _selected_budget(r) for r in successful) if config else 0,
            candidate_total=len(candidates),candidate_failed=sum(not c["success"] for c in candidates),
            candidate_failure_counts=_counts(c["status"] for c in candidates if not c["success"]),
            initialization_spectrum_failures=sum(c["status"] == "initialization_spectrum_failed" for c in candidates),
            candidate_termination_counts=_counts(c["termination_reason"] for c in candidates),
            winner_termination_counts=_counts(r["termination_reason"] for r in successful),
            failure_reasons=_counts(r["failure_reason"] for r in cell if r["failure_reason"]),
            selected_params_counts=_counts(json.dumps(r["best_params"],sort_keys=True) for r in successful))
        for name,values in (("selected_iteration",[r["selected_iteration"] for r in successful]),
                            ("optimizer_iteration",[r["n_iter"] for r in successful]),
                            ("validation_mse",[r["validation_loss"] for r in successful]),
                            ("fit_seconds",[r["fit_time_sec"] for r in cell])):
            row[name+"_mean"],_ = _stats(values)
            row[name+"_min"],row[name+"_max"] = (min(values),max(values)) if values else (None,None)
        for side in ("u","v"):
            supports = [len(r["selected_supports"][side]) for r in successful]
            row[f"support_{side}_mean"],_ = _stats(supports)
            row[f"support_{side}_min"],row[f"support_{side}_max"] = (min(supports),max(supports)) if supports else (None,None)
            row[f"penalty_{side}_counts"] = _counts(str(r["best_params"][f"penalty_{side}"]) for r in successful)
            refined = [r["best_params"][f"penalty_{side}"] for r in successful if r["selected_iteration"] > 0]
            row[f"penalty_{side}_refined_lower_edge"] = sum(v == min(config[f"penalties_{side}"]) for v in refined) if config else 0
            row[f"penalty_{side}_refined_upper_edge"] = sum(v == max(config[f"penalties_{side}"]) for v in refined) if config else 0
        for method in ("SMART","SMART_fixed","RRR","SRRR","SOFAR","RSSVD"):
            ref = next(r for r in paper if int(r["model_id"]) == m and r["experiment"] == experiment
                       and r["method"] == method and abs(float(r["x"])-row["x"]) < 1e-9)
            row[f"paper_{method}_mean"],row[f"paper_{method}_se"] = float(ref["mean"]),float(ref["se"])
        rows.append(row)
    metadata = dict(model_ids=list(model_ids),experiments=list(experiments),seed_ids=list(seed_ids),
        expected_runs=len(settings)*len(seed_ids),recorded_runs=len(records),
        expected_applicable_runs=sum(s.inapplicability_reason() is None for _,s in settings.values())*len(seed_ids),
        expected_inapplicable_runs=sum(s.inapplicability_reason() is not None for _,s in settings.values())*len(seed_ids),
        runner_config=config,implementation_fingerprints={str(k):list(v) for k,v in implementations.items()},
        regenerated_unique_datasets=len(cache),verified_input_records=len(records),
        expected_unique_datasets=len({(s.n,s.p,s.q,s.sigma0,seed) for _,s in settings.values() for seed in seed_ids}),
        duplicate_default_cells_verified=duplicates,
        missing_cells=[dict(model_id=m,experiment=e,setting=suffix,seed_id=seed)
                       for m,e,suffix in settings for seed in seed_ids if (m,e,suffix,seed) not in records],
        actual_base_training_n={f"model{m+1}":experiment_settings(m,0)[0].n for m in model_ids},
        additional_validation_rows=100,refit_on_all_data=False,paper_repetitions=100,
        paper_pdf_hashes_verified=pdfs,comparison_has_equal_tuning_data=False,
        source_rank_semantics_match_paper=False,
        artifact_validation="identities/configurations/seeds/splits/candidates/winners; actual regenerated training/validation/truth hashes and coefficient errors; duplicate default cells; original PDF hashes",
        result_root=str(Path(result_root).resolve()),paper_reference=str(Path(reference_path).resolve()))
    return rows,paper,metadata


def _fmt(value,digits=6):
    return "NA" if value is None else f"{value:.{digits}f}"


def write_outputs(rows,paper,metadata,output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True,exist_ok=True)
    with (output_dir/"comparison.csv").open("w",newline="") as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (output_dir/"validation_audit.json").write_text(json.dumps(metadata,indent=2,sort_keys=True)+"\n")
    report = ["# SparseSMART across the legacy paper grid", "",
        f"Recorded {metadata['recorded_runs']}/{metadata['expected_runs']} requested cells: {sum(r['complete'] for r in rows)} successful fits, {sum(r['failed'] for r in rows)} failed runs/searches, {sum(r['inapplicable'] for r in rows)} inapplicable records and {sum(r['missing'] for r in rows)} missing cells. Only SparseSMART was fitted; paper methods are existing digitized reference curves.", "",
        f"Each setting uses seed IDs {', '.join(map(str,metadata['seed_ids']))}, with 100 additional independent tuning observations per fit and no full-data refit. The plotted training sample size is the full original legacy sample, not a split. Base training sizes are 200, 300 and 500 for Models I, II and III. The manuscript prose states a common default of 200; these runs follow the existing Python grids, so the actual model-specific training sizes are explicit.", "",
        "The paper references summarize 100 repetitions, whereas new means/SEs use only the requested seeds. New fits receive 100 extra observations for penalty/iterate selection: this is not an equal-data-budget comparison with the paper. Validation loss is a reused selection score, not independent test performance. Coefficient error is ||C_hat-C_star||_F / sqrt(p*q). Figure aggregates do not permit paired seed-level significance tests.", "",
        "Experiment 3 is SparseSMART source-rank sensitivity on the paper grid. Its source_rank controls the reduced initializer and source-coordinate construction. The paper's r_s instead specifies leading source directions left unpenalized. These parameters have different meanings; the experiment is not an identical parameter comparison. Automatically ranked paper SMART is a horizontal reference in Experiments 2 and 3; fixed-rank SMART is the closer rank-sensitivity reference.", "",
        "Fitted target rank 11 exceeds source rank 10, and source ranks 0/3 are below fitted rank 5. These cells are inapplicable and are never clamped or assigned zero error. Failures at otherwise applicable settings remain failures; initialization_spectrum_failed means the initializer violates the declared spectral margins before refinement. Means/SEs include successful fits only, with denominators and failure counts shown. Where failures occur, means are conditional on success. Missing results remain distinct.", "",
        f"The audit regenerated {metadata['regenerated_unique_datasets']} distinct training/validation datasets for {metadata['verified_input_records']} saved records, sharing cached data across fitted-rank settings. Training, validation and evaluation-truth hashes were checked directly. Repeated default cells across the four experiments were checked for matching fingerprints and outcomes (numerical agreement tolerance 1e-10). No earlier 160/40 result files are required.", ""]
    for model_id in metadata["model_ids"]:
        report += [f"## Model {['I','II','III'][model_id]}", "",
            "| Experiment / setting | Training n | Success / requested | Failed / inapplicable / missing | Error (SE) | Paper fixed-rank SMART | Paper SMART |",
            "|---|---:|---:|---:|---:|---:|---:|"]
        for r in rows:
            if r["model_id"] != model_id:continue
            error = f"{_fmt(r['final_mean'])} ({_fmt(r['final_se'])})" if r["applicable"] else "Not applicable"
            report.append(f"| {r['experiment']}: {r['setting']} | {r['n_train']} | {r['complete']}/{r['expected_runs']} | {r['failed']} / {r['inapplicable']} / {r['missing']} | {error} | {_fmt(r['paper_SMART_fixed_mean'])} | {_fmt(r['paper_SMART_mean'])} |")
        report += ["", f"![Model {model_id+1} comparison](comparison_model{model_id+1}.png)", ""]
    report += ["## Selection and optimization diagnostics", "",
        "Iteration-limit termination is not convergence. Optimizer convergence describes its final state; selected convergence describes the iterate retained by validation. Selecting iteration zero does not identify a preference for refinement penalties when initializers tie. The CSV preserves selected parameter counts, support ranges, candidate failure types, initialization-spectrum failures, iteration ranges, convergence and timings.", "",
        "| Model / experiment / setting | Initializer selections | Optimizer converged / max iterations | Selected converged | Selected iteration mean (range) | Candidate failures / total | Mean supports U / V |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        if not r["applicable"]:continue
        report.append(f"| {r['model']}: {r['experiment']} {r['setting']} | {r['chosen_initializer']} | {r['optimization_converged']} / {r['max_iterations']} | {r['selected_converged']} | {_fmt(r['selected_iteration_mean'],1)} ({r['selected_iteration_min']}--{r['selected_iteration_max']}) | {r['candidate_failed']}/{r['candidate_total']} | {_fmt(r['support_u_mean'],1)} / {_fmt(r['support_v_mean'],1)} |")
    report += ["", "Empirical noisy-source mode bypasses the source-accuracy sufficient condition when configured below. Numerical success is not theorem certification. The package, original runners, saved results, old summaries and manuscript files were preserved.", "",
        "The inherited validation_fraction and split_seed fields in the configuration below are inactive with explicit external validation. Every original training row is used for fitting; all 100 independently generated validation rows are used for tuning. The separate validation_seed_tag controls that external sample.", "",
        "```json",json.dumps(metadata["runner_config"],indent=2,sort_keys=True),"```",""]
    (output_dir/"comparison.md").write_text("\n".join(report))
    for model_id in metadata["model_ids"]:
        _plot(rows,metadata,model_id,output_dir/f"comparison_model{model_id+1}.png")
    return output_dir/"comparison.csv"


def _plot(rows,metadata,model_id,destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(2,2,figsize=(14,10.5))
    titles = ("Experiment 1: sample size", "Experiment 2: fitted target rank",
              "Experiment 3: SparseSMART source-rank sensitivity", "Experiment 4: source noise")
    labels = ("Training sample size n", "Fitted target rank", "SparseSMART source_rank (paper grid)", "Source noise (tested levels)")
    for exp_id,ax in enumerate(axes.flat):
        cell = [r for r in rows if r["model_id"] == model_id and r["experiment"] == f"exp{exp_id+1}"]
        if not cell:
            ax.set_visible(False);continue
        raw_x = np.array([r["x"] for r in cell])
        x = np.arange(len(cell)) if exp_id == 3 else raw_x
        for method,color,label in (("SMART","#318553","Paper SMART"),("SMART_fixed","#78828c","Paper fixed-rank SMART"),("RRR","#ba9158","Paper RRR")):
            ax.plot(x,[r[f"paper_{method}_mean"] for r in cell],ls="--",color=color,lw=1.3,label=label)
        ax.errorbar(x,[np.nan if r["final_mean"] is None else r["final_mean"] for r in cell],
                    yerr=[0 if r["final_se"] is None else r["final_se"] for r in cell],
                    marker="o",lw=2,ms=5,capsize=3,color="#143652",label="SparseSMART (mean ± SE)")
        for xx,r in zip(x,cell):
            if not r["applicable"] or r["complete"] == 0:
                ax.text(xx,.045,"N/A" if not r["applicable"] else f"0/{r['expected_runs']}",
                        transform=ax.get_xaxis_transform(),ha="center",fontsize=8,color="#8e4545")
            elif r["complete"] < r["expected_runs"]:
                ax.annotate(f"{r['complete']}/{r['expected_runs']}",(xx,r["final_mean"]),
                            xytext=(4,7),textcoords="offset points",fontsize=8)
        ax.set_title(titles[exp_id],fontsize=11)
        ax.set_xlabel(labels[exp_id],fontsize=10)
        ax.set_ylabel(r"$\|\widehat C-C^*\|_F/\sqrt{pq}$",fontsize=11)
        ax.set_xticks(x,[f"{v:g}" for v in raw_x])
        ax.set_ylim(bottom=0);ax.grid(alpha=.16);ax.spines[["top","right"]].set_visible(False)
    handles,legend = next(ax for ax in axes.flat if ax.get_visible()).get_legend_handles_labels()
    p,q=MODEL_DIMS[model_id]
    fig.suptitle(f"Model {['I','II','III'][model_id]} (p={p}, q={q}): SparseSMART across the paper grid",fontsize=16,y=.968)
    fig.text(.5,.933,f"{len(metadata['seed_ids'])} seeds; full legacy training sample (base n={metadata['actual_base_training_n'][f'model{model_id+1}']}); +100 independent tuning rows",ha="center",fontsize=11)
    fig.legend(handles,legend,loc="lower center",bbox_to_anchor=(.5,.063),ncol=4,frameon=False,fontsize=10)
    fig.text(.5,.038,"Fractions show successful/requested fits. N/A: dimensions unsupported by SparseSMART.",ha="center",fontsize=10)
    fig.text(.5,.016,"Paper: 100 repetitions, different tuning data. No other methods rerun.",ha="center",fontsize=10)
    fig.tight_layout(rect=(.01,.135,.99,.91),h_pad=3,w_pad=2)
    fig.savefig(destination,dpi=170);plt.close(fig)


def _indices(text):
    try:return tuple(int(value) for value in text.split(","))
    except ValueError as error:raise argparse.ArgumentTypeError("Expected comma-separated integer IDs") from error


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root",type=Path,default=HERE/"result"/"sparse_smart_external")
    parser.add_argument("--paper-reference",type=Path,default=DEFAULT_REFERENCE)
    parser.add_argument("--output-dir",type=Path,default=HERE/"result"/"sparse_smart_external"/"grid_summary")
    parser.add_argument("--models",type=_indices,default=(0,1,2))
    parser.add_argument("--experiments",type=_indices,default=(0,1,2,3))
    parser.add_argument("--seed-count",type=int,default=5)
    parser.add_argument("--seed-file",type=Path,default=DEFAULT_SEED_FILE)
    args=parser.parse_args(argv)
    if not 1 <= args.seed_count <= 100:parser.error("seed-count must lie between 1 and 100")
    rows,paper,metadata=summarize(args.result_root,args.paper_reference,model_ids=args.models,
        experiments=args.experiments,seed_ids=tuple(range(args.seed_count)),seed_file=args.seed_file)
    print(write_outputs(rows,paper,metadata,args.output_dir))
    return 0


if __name__ == "__main__":raise SystemExit(main())
