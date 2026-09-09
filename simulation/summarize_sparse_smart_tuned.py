"""Validate and summarize tuned SparseSMART runs against existing paper curves.

This module reads result artifacts only. It never fits an estimator or changes
the runner, package, paper references, or earlier fixed-tuning summaries.
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
import re

import numpy as np

from run_restricted_rrr import DEFAULT_SEED_FILE, experiment_settings, load_experiment_seeds
from run_sparse_smart import _digest_json, _json_value
from run_sparse_smart_tuned import RunnerConfig, resolved_configuration, result_path
from summarize_sparse_smart import DEFAULT_REFERENCE, _stats

HERE = Path(__file__).resolve().parent
PAPER_METHODS = ("SMART", "SMART_fixed", "RRR", "SRRR", "SOFAR", "RSSVD")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _finite(value, name):
    _require(not isinstance(value, bool) and isinstance(value, (int, float))
             and math.isfinite(value) and value >= 0, f"Invalid {name}")
    return float(value)


def _integer(value, name, maximum=None):
    _require(type(value) is int and value >= 0 and (maximum is None or value <= maximum),
             f"Invalid {name}")
    return value


def _runner_config(value):
    values = dict(value)
    for key in ("init_penalties", "penalties_u", "penalties_v"):
        values[key] = tuple(values[key])
    if values["support_limits"] is not None:
        values["support_limits"] = tuple(tuple(pair) for pair in values["support_limits"])
    config = RunnerConfig(**values)
    config.validate()
    return config


def _validate_validation_history(record):
    n_iter = _integer(record["n_iter"], "candidate n_iter")
    selected = _integer(record["selected_iteration"], "candidate selected_iteration", n_iter)
    history = record["validation_history"]
    _require([item["iteration"] for item in history] == list(range(n_iter + 1)),
             "Incomplete candidate validation history")
    scores = [_finite(item["loss"], "candidate validation loss") for item in history]
    _require(selected == min(range(len(scores)), key=scores.__getitem__),
             "Selected iterate is not the earliest validation minimum")
    score = _finite(record["validation_mse"], "candidate validation_mse")
    _require(math.isclose(score, scores[selected], rel_tol=1e-10, abs_tol=1e-12),
             "Candidate score disagrees with selected validation iterate")


def validate_record(record, path, *, setting, random_seed, model, experiment, seed_id):
    """Validate provenance, split isolation, and successful-selection semantics."""
    _require(record["schema_version"] == 1 and record["method"] == "SparseSMARTTuned",
             "Unexpected tuned result schema or method")
    _require(record["model"] == model and record["experiment"] == experiment
             and record["rd_seed_id"] == seed_id and record["random_seed"] == int(random_seed),
             "Cell identity or saved random seed mismatch")
    _require(record["setting"] == asdict(setting), "Cell setting does not match the V1 grid")
    expected_name = result_path(Path("."), model=model, experiment=experiment,
                                setting=setting, seed_id=seed_id).name
    _require(path.name == expected_name and path.parent.name == experiment
             and path.parent.parent.name == model, "Result filename or directory does not match cell identity")
    config = _runner_config(record["configuration"]["runner"])
    expected_config = _json_value(resolved_configuration(setting, config))
    _require(record["configuration"] == expected_config, "Resolved configuration mismatch")
    arguments = dict(n=setting.n, p=setting.p, q=setting.q, sigma0=setting.sigma0,
                     sigma=.5, r_star=5, r0_star=10, random_seed=int(random_seed))
    _require(record["generator_arguments"] == arguments, "Generator arguments do not match cell identity")
    identity = {key: record[key] for key in (
        "schema_version", "method", "model", "experiment", "rd_seed_id", "random_seed",
        "setting", "configuration", "generator_arguments")}
    _require(record["configuration_fingerprint"] == _digest_json(identity), "Configuration fingerprint mismatch")
    for field in ("configuration_fingerprint", "implementation_fingerprint", "observed_input_fingerprint",
                  "evaluation_truth_fingerprint", "input_fingerprint"):
        _require(isinstance(record[field], str) and re.fullmatch(r"[0-9a-f]{64}", record[field]) is not None,
                 f"Malformed {field}")
    _require(record["input_fingerprint"] == _digest_json(dict(
        observed=record["observed_input_fingerprint"], evaluation_truth=record["evaluation_truth_fingerprint"])),
        "Combined input fingerprint mismatch")
    _require(record["source_check_mode"] == ("strict" if config.strict_source_check else "empirical"),
             "Source-check mode differs from configuration")
    _require(record["theorem_certified"] is False, "Empirical result cannot be a theorem certificate")
    status = record["status"]
    _require(status in ("complete", "failed", "all_candidates_failed", "inapplicable"), "Unknown tuned run status")
    _require(type(record["success"]) is bool and record["success"] == (status == "complete"),
             "Inconsistent success flag")
    _require(record["applicable"] == (setting.inapplicability_reason() is None), "Applicability mismatch")
    if not record["success"]:
        _require(record["avg_err"] is None, "Failed partial error must not be stored as a successful avg_err")
    else:
        _finite(record["avg_err"], "completed coefficient error")
        _finite(record["initial_avg_err"], "initializer coefficient error")
        coefficient = np.asarray(record["C_hat"], dtype=float)
        _require(coefficient.shape == (setting.p, setting.q) and np.isfinite(coefficient).all(),
                 "Invalid completed coefficient")
    if record["split"] is not None:
        split = record["split"]
        nv = int(math.ceil(setting.n * config.validation_fraction))
        order = np.random.default_rng(config.split_seed).permutation(setting.n)
        expected_indices = dict(train_indices=np.sort(order[nv:]).tolist(),
                                validation_indices=np.sort(order[:nv]).tolist())
        _require(split == dict(**expected_indices, n_train=setting.n-nv, n_validation=nv,
                               fingerprint=_digest_json(expected_indices), refit_on_all_data=False),
                 "Training/validation split mismatch or overlap")
    if status in ("complete", "all_candidates_failed"):
        _require(record["split"] is not None, "Missing split for a completed tuning search")
    candidates = record["selection_history"]
    supports = expected_config["support_limits"]
    parameter_grid = [dict(init_penalty=li, penalty_u=lu, penalty_v=lv,
                           support_limits=pair, step_size_inverse=config.inverse_step)
                      for li, lu, lv, pair in product(config.init_penalties, config.penalties_u,
                                                     config.penalties_v, supports)]
    _require(len(candidates) <= len(parameter_grid), "Too many candidate records")
    if status in ("complete", "all_candidates_failed"):
        _require(len(candidates) == len(parameter_grid), "Incomplete candidate search")
    for index, candidate in enumerate(candidates):
        _require(candidate["candidate_id"] == index and candidate["params"] == parameter_grid[index],
                 "Candidate identity or tuning grid mismatch")
        _require(type(candidate["success"]) is bool, "Invalid candidate success flag")
        if candidate["success"]:
            _require(candidate["status"] in ("completed", "converged"), "Failed candidate marked successful")
            _validate_validation_history(candidate)
        else:
            _require(candidate["validation_mse"] is None, "Failed candidate must not have an eligible validation score")
        _integer(candidate["n_iter"], "candidate iterations", config.iterations)
    _require(record["fit_errors"] == [c for c in candidates if not c["success"]],
             "fit_errors does not match failed candidate records")
    eligible = [candidate for candidate in candidates if candidate["success"]]
    if status == "all_candidates_failed":
        _require(record["all_candidates_failed"] is True and not eligible
                 and record["best_params"] is None and record["C_hat"] is None,
                 "All-candidates-failed record contains a selected model")
    if record["success"]:
        _require(record["all_candidates_failed"] is False and bool(eligible), "No eligible successful winner")
        winner = min(eligible, key=lambda c: c["validation_mse"])
        _require(record["best_params"] == winner["params"], "Selected candidate is not the validation winner")
        _require(record["selected_iteration"] == winner["selected_iteration"]
                 and record["n_iter"] == winner["n_iter"], "Winner iteration mismatch")
        _require(math.isclose(_finite(record["validation_loss"], "selected validation loss"),
                             winner["validation_mse"], rel_tol=1e-10, abs_tol=1e-12), "Winner validation score mismatch")
        _require(record["validation_history"] == winner["validation_history"], "Winner validation history mismatch")
        history = record["history"]
        _require([h["iteration"] for h in history] == list(range(record["n_iter"] + 1)),
                 "Incomplete selected optimization history")
        selected_history = history[record["selected_iteration"]]
        for side, maximum, cap in zip(("u", "v"), expected_config["actual_complement_counts"],
                                     record["best_params"]["support_limits"]):
            indices = record["selected_supports"][side]
            _require(all(type(i) is int and 0 <= i < maximum for i in indices)
                     and indices == sorted(set(indices)) and len(indices) <= cap
                     and len(indices) == selected_history[f"support_{side}"],
                     "Selected support does not match its budget or selected iteration")
        diag = record["diagnostics"]
        _require(type(diag["optimization_converged"]) is bool and type(diag["selected_converged"]) is bool,
                 "Missing or invalid convergence diagnostics")
        _require(diag["optimization_converged"] == (record["termination_reason"] == "stationarity"),
                 "Optimization convergence/termination mismatch")
        if record["termination_reason"] == "max_iterations":
            _require(record["n_iter"] == config.iterations, "Maximum-iteration termination below configured budget")
    return config


def _paper_reference(path, model_id, experiments):
    with Path(path).open() as stream:
        paper = [r for r in csv.DictReader(stream) if int(r["model_id"]) == model_id
                 and r["experiment"] in {f"exp{i+1}" for i in experiments}]
    for exp_id in experiments:
        for setting in experiment_settings(model_id, exp_id):
            x = float(setting.suffix.split("=", 1)[1])
            for method in PAPER_METHODS:
                matches = [r for r in paper if r["experiment"] == f"exp{exp_id+1}" and r["method"] == method
                           and abs(float(r["x"])-x) < 1e-9]
                _require(len(matches) == 1, f"Missing or duplicate paper curve: exp{exp_id+1} {method} {x}")
                ref = matches[0]
                _require(int(ref["p"]) == setting.p and int(ref["q"]) == setting.q
                         and int(ref["paper_repetitions"]) == 100, "Paper dimensions/repetition metadata mismatch")
                _finite(float(ref["mean"]), "paper mean")
                _finite(float(ref["se"]), "paper SE")
    return paper


def _counter(values):
    return json.dumps(dict(Counter(values)), sort_keys=True)


def summarize(result_root, reference_path=DEFAULT_REFERENCE, *, model_id=0,
              seed_ids=(0, 1, 2, 3, 4), experiments=(0, 3), seed_file=DEFAULT_SEED_FILE,
              fixed_result_root=None):
    """Return (cell rows, paper rows, audit metadata), rejecting mixed artifacts."""
    seed_ids = tuple(seed_ids)
    _require(seed_ids and len(set(seed_ids)) == len(seed_ids)
             and all(type(s) is int and 0 <= s < 100 for s in seed_ids), "Invalid requested seed IDs")
    _require(experiments and len(set(experiments)) == len(experiments), "Invalid requested experiments")
    seeds = load_experiment_seeds(seed_file)
    _require(max(seed_ids) < len(seeds), "Requested seed is unavailable")
    model = f"model{model_id+1}"
    settings = {(f"exp{eid+1}", s.suffix): s for eid in experiments for s in experiment_settings(model_id, eid)}
    records, configs, implementations = {}, set(), set()
    for path in sorted((Path(result_root) / model).glob("exp*/SparseSMARTTuned_result_*.json")):
        try:
            record = json.loads(path.read_text())
            if record["rd_seed_id"] not in seed_ids or record["experiment"] not in {key[0] for key in settings}:
                continue
            key = (record["experiment"], record["setting"]["suffix"], record["rd_seed_id"])
            _require(key[:2] in settings, "Unknown cell setting")
            _require(key not in records, f"Duplicate cell: {key}")
            config = validate_record(record, path, setting=settings[key[:2]], random_seed=seeds[key[2]],
                                     model=model, experiment=key[0], seed_id=key[2])
            configs.add(json.dumps(_json_value(asdict(config)), sort_keys=True))
            implementations.add(record["implementation_fingerprint"])
            records[key] = record
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"Invalid tuned result {path}: {error}") from error
    _require(len(configs) <= 1 and len(implementations) <= 1,
             "Cannot pool runs with different tuning configurations or implementations")
    config = json.loads(next(iter(configs))) if configs else None
    shared_pairs = []
    if ("exp1", "n=200") in settings and ("exp4", "sigma0=0.01") in settings:
        for seed in seed_ids:
            a, b = records.get(("exp1", "n=200", seed)), records.get(("exp4", "sigma0=0.01", seed))
            if a is None or b is None:
                continue
            for field in ("observed_input_fingerprint", "evaluation_truth_fingerprint", "input_fingerprint",
                          "status", "success", "C_hat", "best_params", "selected_iteration", "avg_err"):
                _require(a[field] == b[field], f"Shared exp1/exp4 cell disagrees for seed {seed}: {field}")
            shared_pairs.append(seed)
    paper = _paper_reference(reference_path, model_id, experiments)
    fixed = {}
    if fixed_result_root is not None:
        from summarize_sparse_smart import summarize as fixed_summarize
        fixed_rows, _, _ = fixed_summarize(fixed_result_root, reference_path, model_id=model_id,
                                          seed_ids=seed_ids, experiments=experiments)
        fixed = {(r["experiment"], r["setting"]): r for r in fixed_rows}
    rows = []
    for (experiment, suffix), setting in settings.items():
        cell = [records[(experiment, suffix, seed)] for seed in seed_ids if (experiment, suffix, seed) in records]
        successful = [r for r in cell if r["success"]]
        candidates = [c for r in cell for c in r["selection_history"]]
        mean, se = _stats([r["avg_err"] for r in successful])
        initial_mean, _ = _stats([r["initial_avg_err"] for r in successful])
        x = float(suffix.split("=", 1)[1])
        row = dict(model=model, experiment=experiment, setting=suffix, x=x, n_total=setting.n,
                   expected_runs=len(seed_ids), recorded_runs=len(cell), complete=len(successful),
                   failed=sum(r["status"] in ("failed", "all_candidates_failed") for r in cell),
                   all_candidates_failed=sum(r["status"] == "all_candidates_failed" for r in cell),
                   inapplicable=sum(r["status"] == "inapplicable" for r in cell), missing=len(seed_ids)-len(cell),
                   initial_mean=initial_mean, final_mean=mean, final_se=se,
                   mean_is_conditional_on_success=len(successful) < len(cell),
                   n_train=None if config is None else setting.n-math.ceil(setting.n*config["validation_fraction"]),
                   n_validation=None if config is None else math.ceil(setting.n*config["validation_fraction"]),
                   chosen_initializer=sum(r["selected_iteration"] == 0 for r in successful),
                   refined_selections=sum(r["selected_iteration"] > 0 for r in successful),
                   selected_final_iteration=sum(r["selected_iteration"] == r["n_iter"] for r in successful),
                   selected_at_budget=sum(r["selected_iteration"] == config["iterations"] for r in successful) if config else 0,
                   selected_near_budget=sum(r["selected_iteration"] >= .9*config["iterations"] for r in successful) if config else 0,
                   optimization_converged=sum(r["diagnostics"]["optimization_converged"] for r in successful),
                   selected_converged=sum(r["diagnostics"]["selected_converged"] for r in successful),
                   max_iterations=sum(r["termination_reason"] == "max_iterations" for r in successful),
                   candidate_total=len(candidates), candidate_success=sum(c["success"] for c in candidates),
                   candidate_failed=sum(not c["success"] for c in candidates),
                   candidate_status_counts=_counter(c["status"] for c in candidates),
                   candidate_failure_status_counts=_counter(c["status"] for c in candidates if not c["success"]),
                   candidate_termination_counts=_counter(c["termination_reason"] for c in candidates),
                   winner_termination_counts=_counter(r["termination_reason"] for r in successful),
                   failure_reasons=_counter(r["failure_reason"] for r in cell if r["failure_reason"]),
                   source_check_bypassed=sum(bool(r["diagnostics"].get("calibration", {}).get("source_accuracy_bypassed")) for r in successful))
        for field, values in (
            ("selected_iteration", [r["selected_iteration"] for r in successful]),
            ("optimization_iteration", [r["n_iter"] for r in successful]),
            ("fit_seconds", [r["fit_time_sec"] for r in cell]),
            ("validation_mse", [r["validation_loss"] for r in successful]),
        ):
            row[field+"_mean"], _ = _stats(values)
            row[field+"_min"] = min(values) if values else None
            row[field+"_max"] = max(values) if values else None
        for side in ("u", "v"):
            selected = [r["best_params"][f"penalty_{side}"] for r in successful]
            counts = [len(r["selected_supports"][side]) for r in successful]
            row[f"penalty_{side}_counts"] = _counter(str(v) for v in selected)
            row[f"penalty_{side}_lower_edge"] = sum(v == min(config[f"penalties_{side}"]) for v in selected) if config else 0
            row[f"penalty_{side}_upper_edge"] = sum(v == max(config[f"penalties_{side}"]) for v in selected) if config else 0
            refined_penalties = [r["best_params"][f"penalty_{side}"] for r in successful if r["selected_iteration"] > 0]
            row[f"penalty_{side}_refined_lower_edge"] = sum(v == min(config[f"penalties_{side}"]) for v in refined_penalties) if config else 0
            row[f"penalty_{side}_refined_upper_edge"] = sum(v == max(config[f"penalties_{side}"]) for v in refined_penalties) if config else 0
            row[f"support_{side}_mean"], _ = _stats(counts)
            row[f"support_{side}_min"] = min(counts) if counts else None
            row[f"support_{side}_max"] = max(counts) if counts else None
            row[f"support_{side}_at_cap"] = sum(len(r["selected_supports"][side]) == r["best_params"]["support_limits"][0 if side == "u" else 1] for r in successful)
        row["penalty_pair_counts"] = _counter(f"{r['best_params']['penalty_u']},{r['best_params']['penalty_v']}" for r in successful)
        for method, label in (("SMART", "paper_smart"), ("SMART_fixed", "paper_smart_fixed")):
            ref = next(r for r in paper if r["experiment"] == experiment and r["method"] == method and abs(float(r["x"])-x) < 1e-9)
            row[label], row[label+"_se"] = float(ref["mean"]), float(ref["se"])
        previous = fixed.get((experiment, suffix))
        row["fixed_pilot_mean"] = previous["final_mean"] if previous else None
        row["fixed_pilot_se"] = previous["final_se"] if previous else None
        row["fixed_pilot_complete"] = previous["complete"] if previous else None
        rows.append(row)
    metadata = dict(model=model, seed_ids=list(seed_ids), runner_config=config,
                    implementation_fingerprint=next(iter(implementations)) if implementations else None,
                    expected_runs=len(settings)*len(seed_ids), recorded_runs=len(records),
                    missing_cells=[dict(experiment=e, setting=s, seed_id=i) for e, s in settings for i in seed_ids
                                   if (e, s, i) not in records],
                    artifact_validation="cell/configuration/seed/split/candidate/winner consistency; no data regeneration",
                    fixed_pilot_included=fixed_result_root is not None,
                    nominal_cells=len(settings)*len(seed_ids),
                    unique_requested_datasets=len({(s.n,s.p,s.q,s.sigma0,seed) for s in settings.values() for seed in seed_ids}),
                    shared_exp1_exp4_seeds_verified=shared_pairs,
                    paper_reference=str(Path(reference_path).resolve()), paper_repetitions=100)
    return rows, paper, metadata


def _fmt(value, digits=6):
    return "NA" if value is None else f"{value:.{digits}f}"


def write_outputs(rows, paper, metadata, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "validation_audit.json").write_text(json.dumps(metadata, indent=2, sort_keys=True)+"\n")
    n_success = sum(r["complete"] for r in rows)
    total = metadata["expected_runs"]
    config = metadata["runner_config"]
    if config is None:
        split_description = "No fitted configuration is available to describe the training/validation split."
    else:
        fraction = config["validation_fraction"]
        train_pct, validation_pct = 100*(1-fraction), 100*fraction
        validation_200 = math.ceil(200*fraction)
        split_description = (
            f"Each candidate fits {train_pct:g}% of the target observations and selects its iteration using the held-out {validation_pct:g}%, including iteration zero. "
            f"The same validation split selects penalties; no full-data refit occurs. At n=200 this means {200-validation_200} training and {validation_200} validation rows. "
            "The validation score is a selection criterion, not an independent test error. Coefficient truth is used only after selection for evaluation."
        )
    report = ["# Validation-tuned SparseSMART pilot versus V1 paper results", "",
        f"The requested pilot has {n_success}/{total} successful fits, {sum(r['failed'] for r in rows)} failed searches/runs, and {sum(r['missing'] for r in rows)} missing cells. Only SparseSMART was run; all other method curves were read from existing paper figure vectors.", "",
        f"New results use seed IDs {', '.join(map(str,metadata['seed_ids']))}: {len(metadata['seed_ids'])} repetitions per cell, versus 100 repetitions for the paper. The {metadata['nominal_cells']} nominal cells contain {metadata['unique_requested_datasets']} unique datasets: exp1 n=200 repeats exp4 source noise=0.01. Available shared cells were checked for identical fingerprints, selected coefficients, parameters, and errors. Paper values are approximate figure aggregates, not paired per-seed observations. This comparison does not support paired significance tests.", "",
        "The coefficient metric is ||C_hat-C_star||_F / sqrt(p*q). New means and SEs include successful selected fits only. Whenever a cell has failed runs, its reported mean is conditional on success; failures and missing counts remain explicit. Failed partial fits never become successful observations.", "",
        split_description, "",
        "Noisy-source fits use explicit empirical source-check mode when configured below. Noise, gap, and the failed sufficient condition remain recorded; no theorem certification is claimed. The full actual complement is available unless a support grid is supplied.", "",
        "| Setting | Success / expected | Failed (all candidates) / missing | Train / validation n | Initial error | Selected error (SE) | Paper SMART | Paper fixed rank |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        report.append(f"| {r['experiment']}: {r['setting']} | {r['complete']}/{r['expected_runs']} | {r['failed']} ({r['all_candidates_failed']}) / {r['missing']} | {r['n_train']} / {r['n_validation']} | {_fmt(r['initial_mean'])} | {_fmt(r['final_mean'])} ({_fmt(r['final_se'])}) | {_fmt(r['paper_smart'])} | {_fmt(r['paper_smart_fixed'])} |")
    report += ["", "## Selection and optimization diagnostics", "",
        "Optimization convergence describes the last optimizer state. Selected convergence describes the earlier or final validation-selected state; these are distinct. Reaching the iteration limit is not convergence. Near-budget means the selected iteration is at least 90% of the configured budget. Penalty edge counts below include only refined selections (iteration > 0); their denominator is successful fits minus initializer selections. Initializer selection does not identify a preference for refinement penalties: tied initializers may select the first grid pair. Frequent edge selections among refined fits motivate a later, separately declared wider grid.", "",
        "| Setting | Initializer selected | Optimizer converged / max iterations | Selected converged | Selected iteration mean (range) | Near budget / at budget | Candidate failures / total | Selected supports U / V, mean | Refined lambda U lower / upper | Refined lambda V lower / upper |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        report.append(f"| {r['experiment']}: {r['setting']} | {r['chosen_initializer']}/{r['complete']} | {r['optimization_converged']} / {r['max_iterations']} | {r['selected_converged']}/{r['complete']} | {_fmt(r['selected_iteration_mean'],1)} ({r['selected_iteration_min']}--{r['selected_iteration_max']}) | {r['selected_near_budget']} / {r['selected_at_budget']} | {r['candidate_failed']}/{r['candidate_total']} | {_fmt(r['support_u_mean'],1)} / {_fmt(r['support_v_mean'],1)} | {r['penalty_u_refined_lower_edge']} / {r['penalty_u_refined_upper_edge']} | {r['penalty_v_refined_lower_edge']} / {r['penalty_v_refined_upper_edge']} |")
    if metadata["fixed_pilot_included"]:
        report += ["", "The optional orange curve reads the earlier fixed-tuning SparseSMART pilot from saved results. That pilot trained on all observations with a 25-entry correction cap and 50 updates. It is a separate configuration, not an extra estimator rerun or a controlled single-factor comparison."]
    report += ["", "Exact candidate failure/termination counts, penalty selections, support bounds, timing, and missingness are in comparison.csv. validation_audit.json records the common configuration/implementation and missing cell identities. All input artifacts passed identity, fingerprint, deterministic split, candidate eligibility, and winner consistency checks; this summarizer did not regenerate simulation data.", "",
               "```json", json.dumps(metadata["runner_config"], indent=2, sort_keys=True), "```", "",
               "Paper reference provenance: code/simulation/paper_reference/README.md. Model I uses the repository generator's AR(1) predictor covariance and target noise 0.5.", ""]
    (output_dir / "comparison.md").write_text("\n".join(report))
    _plot(rows, paper, metadata, output_dir / "comparison.png")
    return output_dir / "comparison.csv"


def _plot(rows, paper, metadata, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 12, "axes.titlesize": 15, "axes.labelsize": 12,
                         "xtick.labelsize": 11, "ytick.labelsize": 11})
    experiments = list(dict.fromkeys(r["experiment"] for r in rows))
    fig, axes = plt.subplots(1, len(experiments), figsize=(7.5*len(experiments), 6.8), squeeze=False)
    colors = dict(SMART="#258348", SMART_fixed="#67717d", RRR="#4575b4", SRRR="#a45b8c", SOFAR="#b47d33", RSSVD="#8d77b3")
    labels = dict(SMART="Paper SMART", SMART_fixed="Paper fixed-rank SMART", RRR="Paper RRR",
                  SRRR="Paper SRRR", SOFAR="Paper SOFAR", RSSVD="Paper RSSVD")
    for ax, experiment in zip(axes[0], experiments):
        cell = [r for r in rows if r["experiment"] == experiment]
        raw_x = np.array([r["x"] for r in cell])
        x = np.arange(len(cell)) if experiment == "exp4" else raw_x
        positions = dict(zip(raw_x, x))
        for method in ("SMART", "SMART_fixed", "RRR"):
            refs = sorted((r for r in paper if r["experiment"] == experiment and r["method"] == method), key=lambda r:float(r["x"]))
            ax.plot([positions[float(r["x"])] for r in refs], [float(r["mean"]) for r in refs],
                    color=colors[method], ls="--", lw=1.5, alpha=1. if method.startswith("SMART") else .65, label=labels[method])
        if metadata["fixed_pilot_included"]:
            ax.plot(x, [np.nan if r["fixed_pilot_mean"] is None else r["fixed_pilot_mean"] for r in cell],
                    "o:", color="#c36617", lw=1.6, ms=4, label="Earlier fixed SparseSMART")
        ax.errorbar(x, [np.nan if r["final_mean"] is None else r["final_mean"] for r in cell],
                    yerr=[0. if r["final_se"] is None else r["final_se"] for r in cell],
                    marker="o", color="#102a43", lw=2.5, ms=6, capsize=4,
                    label="Tuned SparseSMART (mean ± SE)")
        for xx, r in zip(x, cell):
            if r["complete"] < r["expected_runs"]:
                ax.annotate(f"{r['complete']}/{r['expected_runs']} fits", (xx, r["final_mean"] or 0),
                            xytext=(4, 10), textcoords="offset points", fontsize=10)
        ax.set_title("Target sample size" if experiment == "exp1" else "Source noise")
        ax.set_xlabel("Total target sample size n" if experiment == "exp1" else "Source noise (tested levels)")
        ax.set_ylabel(r"Coefficient error $\|\widehat C-C^*\|_F/\sqrt{pq}$")
        ax.set_ylim(bottom=0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=.17)
        if experiment == "exp1":
            ax.set_xticks(x)
        elif experiment == "exp4":
            ax.set_xticks(x, [f"{value:g}" for value in raw_x])
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle(f"Model I: validation-tuned SparseSMART ({len(metadata['seed_ids'])} seeds) versus paper curves", fontsize=18, y=.965)
    config = metadata["runner_config"]
    split_label = ("No fitted configuration available" if config is None else
                   f"Training: {100*(1-config['validation_fraction']):g}% of target rows; validation: {100*config['validation_fraction']:g}%; no full-data refit")
    fig.text(.5,.907,split_label, ha="center", fontsize=12)
    fig.legend(handles, legend_labels, loc="lower center", bbox_to_anchor=(.5,.073), ncol=3, frameon=False, fontsize=11)
    source_label = "unknown" if config is None else "strict" if config["strict_source_check"] else "empirical"
    fig.text(.5,.02,f"Paper: 100 repetitions, extracted figure summaries. New noisy fits: {source_label} mode. No baseline reruns.", ha="center", fontsize=11)
    fig.tight_layout(rect=(0,.20,1,.88))
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=HERE / "result" / "sparse_smart_tuned")
    parser.add_argument("--paper-reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output-dir", type=Path, default=HERE / "result" / "sparse_smart_tuned" / "summary")
    parser.add_argument("--fixed-result-root", type=Path)
    parser.add_argument("--seed-file", type=Path, default=DEFAULT_SEED_FILE)
    parser.add_argument("--seed-count", type=int, default=5)
    args = parser.parse_args()
    if not 1 <= args.seed_count <= 100:
        parser.error("seed-count must lie between 1 and 100")
    rows, paper, metadata = summarize(args.result_root, args.paper_reference,
        seed_ids=tuple(range(args.seed_count)), seed_file=args.seed_file, fixed_result_root=args.fixed_result_root)
    print(write_outputs(rows, paper, metadata, args.output_dir))


if __name__ == "__main__":
    main()
