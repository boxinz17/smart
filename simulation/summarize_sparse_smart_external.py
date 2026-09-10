"""Audit the 200-training/100-independent-validation SparseSMART noise pilot.

Read-only with respect to all input runs and paper references. Output files are
new summaries; no data generation or estimator fitting occurs in this module.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import asdict
import hashlib
from itertools import product
import json
import math
from pathlib import Path
import re

import numpy as np

import run_sparse_smart_external as external_runner
from run_restricted_rrr import DEFAULT_SEED_FILE, experiment_settings, load_experiment_seeds
from run_sparse_smart import _digest_json, _json_value
from run_sparse_smart_tuned import result_path as previous_result_path
from summarize_sparse_smart import DEFAULT_REFERENCE, _stats
from summarize_sparse_smart_tuned import (
    _finite, _integer, _paper_reference, _require, _validate_validation_history,
    _candidate_grid, _candidate_budget, _selected_budget, _validate_selected_budget,
    _validate_checkpoint_metadata,
    validate_record as validate_previous_record,
)

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent.parent
HISTORICAL_SOLVER_SETTINGS = {"initialization_spectrum": "reject", "refinement_solver": "chart"}


def _config(value):
    value = dict(value)
    if value.get("iteration_budgets") is not None:
        value["iteration_budgets"] = tuple(value["iteration_budgets"])
    for key, historical_value in HISTORICAL_SOLVER_SETTINGS.items():
        value.setdefault(key, historical_value)
    for key in ("init_penalties", "penalties_u", "penalties_v"):
        value[key] = tuple(value[key])
    if value["support_limits"] is not None:
        value["support_limits"] = tuple(tuple(pair) for pair in value["support_limits"])
    result = external_runner.RunnerConfig(**value)
    result.validate()
    return result


def _expected_configuration(setting, config, saved):
    """Normalize absent historical solver and checkpoint-budget settings.

    Decoding assigns the historical algorithms, not today's auto defaults.
    The saved object remains untouched for the original identity/hash check.
    """
    expected = _json_value(external_runner.resolved_configuration(setting, config))
    for key in (*HISTORICAL_SOLVER_SETTINGS, "iteration_budgets"):
        if key not in saved["runner"]:
            del expected["runner"][key]
    return expected


def _fingerprint(value, name):
    _require(isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
             f"Malformed {name}")


def _validate_selection(record, config, resolved):
    status = record["status"]
    _require(status in ("complete", "failed", "all_candidates_failed", "inapplicable"),
             "Unknown external run status")
    _require(type(record["success"]) is bool and record["success"] == (status == "complete"),
             "Inconsistent success flag")
    _require(record["applicable"] is True, "All six requested Model I settings must be applicable")
    _require(status != "inapplicable", "An applicable external setting is marked inapplicable")
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
            _require(candidate["status"] in ("completed", "converged"),
                     "Failed candidate marked successful")
            _validate_validation_history(candidate)
        else:
            _require(candidate["validation_mse"] is None,
                     "Failed candidate must not have an eligible validation score")
    _require(record["fit_errors"] == [c for c in candidates if not c["success"]],
              "fit_errors does not match failed candidate records")
    _validate_checkpoint_metadata(record, config,
                                 len(grid) // len(config.iteration_budgets or (config.iterations,)))
    eligible = [c for c in candidates if c["success"]]
    if not record["success"]:
        _require(record["avg_err"] is None, "Failed partial error must not count as a successful error")
        if status == "all_candidates_failed":
            _require(record["all_candidates_failed"] is True and not eligible
                     and record["C_hat"] is None and record["best_params"] is None,
                     "All-candidates-failed record contains a selected model")
        return
    _require(record["all_candidates_failed"] is False and eligible, "No successful candidate winner")
    winner = min(eligible, key=lambda c: c["validation_mse"])
    winner_budget = _validate_selected_budget(record, winner, config)
    _require(record["best_params"] == winner["params"], "Selected candidate is not the validation winner")
    _require(record["selected_iteration"] == winner["selected_iteration"]
             and record["n_iter"] == winner["n_iter"], "Winner iteration mismatch")
    _require(math.isclose(_finite(record["validation_loss"], "selected validation loss"),
                         winner["validation_mse"], rel_tol=1e-10, abs_tol=1e-12),
             "Winner validation score mismatch")
    _require(record["validation_history"] == winner["validation_history"],
             "Winner validation history mismatch")
    _finite(record["avg_err"], "completed coefficient error")
    _finite(record["initial_avg_err"], "initializer coefficient error")
    coefficient = np.asarray(record["C_hat"], dtype=float)
    _require(coefficient.shape == (100, 50) and np.isfinite(coefficient).all(),
             "Invalid completed coefficient")
    history = record["history"]
    _require([item["iteration"] for item in history] == list(range(record["n_iter"] + 1)),
             "Incomplete selected optimization history")
    selected = history[record["selected_iteration"]]
    for side, maximum, cap in zip(("u", "v"), resolved["actual_complement_counts"],
                                 record["best_params"]["support_limits"]):
        indices = record["selected_supports"][side]
        _require(all(type(i) is int and 0 <= i < maximum for i in indices)
                 and indices == sorted(set(indices)) and len(indices) <= cap
                 and len(indices) == selected[f"support_{side}"], "Selected support mismatch")
    diag = record["diagnostics"]
    _require(type(diag["optimization_converged"]) is bool and type(diag["selected_converged"]) is bool,
             "Missing convergence diagnostics")
    _require(diag["optimization_converged"] == (record["termination_reason"] == "stationarity"),
             "Optimizer convergence/termination mismatch")
    if record["termination_reason"] == "max_iterations":
        _require(record["n_iter"] == winner_budget, "Iteration limit recorded below configured budget")


def validate_record(record, path, *, setting, seed_id, random_seed, previous):
    """Validate external-data selection and exact reuse of the original 200 rows."""
    _require(record["schema_version"] == 1 and record["method"] == "SparseSMARTExternal",
             "Unexpected external result schema or method")
    _require(record["model"] == "model1" and record["experiment"] == "exp4"
             and record["rd_seed_id"] == seed_id and record["random_seed"] == int(random_seed),
             "Cell identity or saved random seed mismatch")
    _require(setting.n == 200 and record["setting"] == asdict(setting), "Unexpected external setting")
    expected = external_runner.result_path(Path("."), model="model1", experiment="exp4",
                                          setting=setting, seed_id=seed_id)
    _require(path.name == expected.name and path.parent.name == "exp4"
             and path.parent.parent.name == "model1", "Result filename or directory mismatch")
    config = _config(record["configuration"]["runner"])
    _require(config.n_validation == 100, "This summary requires exactly 100 independent validation rows")
    resolved = _expected_configuration(setting, config, record["configuration"])
    _require(record["configuration"] == resolved, "Resolved configuration mismatch")
    arguments = dict(n=200, p=100, q=50, sigma0=setting.sigma0, sigma=.5,
                     r_star=5, r0_star=10, random_seed=int(random_seed))
    _require(record["generator_arguments"] == arguments, "Generator arguments mismatch")
    identity = {key: record[key] for key in ("schema_version", "method", "model", "experiment",
        "rd_seed_id", "random_seed", "setting", "configuration", "generator_arguments")}
    _require(record["configuration_fingerprint"] == _digest_json(identity), "Configuration fingerprint mismatch")
    for key in ("configuration_fingerprint", "implementation_fingerprint",
                "training_observed_input_fingerprint", "validation_observed_input_fingerprint",
                "evaluation_truth_fingerprint", "input_fingerprint"):
        _fingerprint(record[key], key)
    _require(record["input_fingerprint"] == _digest_json(dict(
        training_observed=record["training_observed_input_fingerprint"],
        validation_observed=record["validation_observed_input_fingerprint"],
        evaluation_truth=record["evaluation_truth_fingerprint"])), "Combined input fingerprint mismatch")
    _require(record["training_observed_input_fingerprint"] == previous["observed_input_fingerprint"]
             and record["evaluation_truth_fingerprint"] == previous["evaluation_truth_fingerprint"],
             "Training data or evaluation truth differs from the original 200-row dataset")
    _require(record["n_train"] == 200 and record["n_validation"] == 100
             and record["all_training_rows_used"] is True and record["training_matches_legacy"] is True,
             "External training/validation counts or reuse flags mismatch")
    _require(record["refit_on_all_data"] is False, "External selected fit must not refit on tuning observations")
    expected_seed_metadata = dict(seed_sequence_entropy=[int(random_seed), config.validation_seed_tag],
        bit_generator="PCG64", covariance="AR1", covariance_rho=.5,
        design_factorization="cholesky", noise_std=.5,
        conditional_on_same_coefficient=True, source_reused=True)
    _require(record["validation_seed_metadata"] == expected_seed_metadata,
             "Independent validation seed metadata mismatch")
    split = record["split"]
    expected_split = dict(mode="independent_external_validation", n_train=200, n_validation=100,
        train_indices=None, validation_indices=None, all_training_rows_used=True, refit_on_all_data=False)
    if split is not None:
        _require(split == dict(**expected_split, fingerprint=_digest_json(expected_split)),
                 "External split metadata mismatch")
    elif record["status"] in ("complete", "all_candidates_failed"):
        raise ValueError("Missing external split for a completed tuning search")
    _require(record["theorem_certified"] is False, "Empirical result cannot be a theorem certificate")
    _require(record["source_check_mode"] == ("strict" if config.strict_source_check else "empirical"),
             "Source-check mode differs from configuration")
    _validate_selection(record, config, resolved)
    return config


def _paper_provenance(reference_path):
    path = Path(reference_path).with_name("provenance.json")
    _require(path.is_file(), "Missing paper reference provenance")
    provenance = json.loads(path.read_text())
    checked = []
    for source in provenance["sources"]:
        pdf = REPO_ROOT / source["pdf"]
        _require(pdf.is_file() and hashlib.sha256(pdf.read_bytes()).hexdigest() == source["sha256"],
                 f"Paper PDF hash differs: {source['pdf']}")
        checked.append(dict(pdf=source["pdf"], sha256=source["sha256"]))
    _require(len(checked) == 3, "Incomplete paper reference provenance")
    return checked


def _counts(values):
    return json.dumps(dict(Counter(values)), sort_keys=True)


def summarize(result_root, reference_path=DEFAULT_REFERENCE, *, seed_ids=(0, 1, 2, 3, 4),
              previous_root=HERE / "result" / "sparse_smart_tuned", seed_file=DEFAULT_SEED_FILE):
    """Return six aggregate rows, paper rows and audit; never pool mismatched runs."""
    seed_ids = tuple(seed_ids)
    _require(seed_ids and len(set(seed_ids)) == len(seed_ids)
             and all(type(seed) is int and 0 <= seed < 100 for seed in seed_ids), "Invalid seed IDs")
    seeds = load_experiment_seeds(seed_file)
    _require(max(seed_ids) < len(seeds), "Requested seed is unavailable")
    settings = {setting.suffix: setting for setting in experiment_settings(0, 3)}
    old, records, configurations, implementations = {}, {}, set(), set()
    old_configurations, old_implementations = set(), set()
    for suffix, setting in settings.items():
        for seed in seed_ids:
            path = previous_result_path(previous_root, model="model1", experiment="exp4",
                                        setting=setting, seed_id=seed)
            if not path.exists():
                continue
            value = json.loads(path.read_text())
            validate_previous_record(value, path, setting=setting, random_seed=seeds[seed],
                                     model="model1", experiment="exp4", seed_id=seed)
            _require(value["split"] is None or (value["split"]["n_train"] == 160
                     and value["split"]["n_validation"] == 40), "Earlier reference is not the 160/40 pilot")
            old[(suffix, seed)] = value
            old_configurations.add(json.dumps(value["configuration"]["runner"], sort_keys=True))
            old_implementations.add(value["implementation_fingerprint"])
    _require(len(old_configurations) <= 1 and len(old_implementations) <= 1,
             "Cannot pool different earlier configurations or implementations")
    for path in sorted((Path(result_root) / "model1" / "exp4").glob("SparseSMARTExternal_result_*.json")):
        try:
            value = json.loads(path.read_text())
            if value["rd_seed_id"] not in seed_ids:
                continue
            key = (value["setting"]["suffix"], value["rd_seed_id"])
            _require(key[0] in settings and key not in records, "Unknown or duplicate external cell")
            _require(key in old, "Missing earlier 200-row input reference for training-fingerprint verification")
            config = validate_record(value, path, setting=settings[key[0]], seed_id=key[1],
                                     random_seed=seeds[key[1]], previous=old[key])
            records[key] = value
            configurations.add(json.dumps(_json_value(asdict(config)), sort_keys=True))
            implementations.add(value["implementation_fingerprint"])
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            raise ValueError(f"Invalid external result {path}: {error}") from error
    _require(len(configurations) <= 1 and len(implementations) <= 1,
             "Cannot pool different external configurations or implementations")
    config = json.loads(next(iter(configurations))) if configurations else None
    paper = _paper_reference(reference_path, 0, (3,))
    provenance = _paper_provenance(reference_path)
    rows = []
    for suffix, setting in settings.items():
        cell = [records[(suffix, seed)] for seed in seed_ids if (suffix, seed) in records]
        successes = [r for r in cell if r["success"]]
        candidates = [c for r in cell for c in r["selection_history"]]
        previous = [old[(suffix, seed)] for seed in seed_ids if (suffix, seed) in old]
        previous_success = [r for r in previous if r["success"]]
        mean, se = _stats([r["avg_err"] for r in successes])
        old_mean, old_se = _stats([r["avg_err"] for r in previous_success])
        initial_mean, initial_se = _stats([r["initial_avg_err"] for r in successes])
        row = dict(model="model1", experiment="exp4", setting=suffix, sigma0=setting.sigma0,
            n_train=200, n_validation=100, expected_runs=len(seed_ids), recorded_runs=len(cell),
            complete=len(successes), failed=sum(not r["success"] for r in cell),
            all_candidates_failed=sum(r["status"] == "all_candidates_failed" for r in cell),
            missing=len(seed_ids)-len(cell), final_mean=mean, final_se=se,
            initial_mean=initial_mean, initial_se=initial_se,
            mean_is_conditional_on_success=len(successes) < len(cell),
            earlier_n_train=160, earlier_n_validation=40, earlier_mean=old_mean, earlier_se=old_se,
            earlier_complete=len(previous_success), earlier_failed=len(previous)-len(previous_success),
            earlier_missing=len(seed_ids)-len(previous),
            chosen_initializer=sum(r["selected_iteration"] == 0 for r in successes),
            selected_budget_counts=_counts(_selected_budget(r) for r in successes),
            selected_at_budget=sum(r["selected_iteration"] == _selected_budget(r) for r in successes) if config else 0,
            selected_near_budget=sum(r["selected_iteration"] >= .9*_selected_budget(r) for r in successes) if config else 0,
            optimization_converged=sum(r["diagnostics"]["optimization_converged"] for r in successes),
            selected_converged=sum(r["diagnostics"]["selected_converged"] for r in successes),
            max_iterations=sum(r["termination_reason"] == "max_iterations" for r in successes),
            winner_termination_counts=_counts(r["termination_reason"] for r in successes),
            candidate_total=len(candidates), candidate_failed=sum(not c["success"] for c in candidates),
            candidate_failure_counts=_counts(c["status"] for c in candidates if not c["success"]),
            failure_reasons=_counts(r["failure_reason"] for r in cell if r["failure_reason"]))
        for name, values in (("selected_iteration", [r["selected_iteration"] for r in successes]),
                             ("optimization_iteration", [r["n_iter"] for r in successes]),
                             ("validation_mse", [r["validation_loss"] for r in successes]),
                             ("fit_seconds", [r["fit_time_sec"] for r in cell])):
            row[name+"_mean"], _ = _stats(values)
            row[name+"_min"], row[name+"_max"] = (min(values), max(values)) if values else (None, None)
        for side in ("u", "v"):
            supports = [len(r["selected_supports"][side]) for r in successes]
            row[f"support_{side}_mean"], _ = _stats(supports)
            row[f"support_{side}_min"], row[f"support_{side}_max"] = (min(supports), max(supports)) if supports else (None, None)
            row[f"penalty_{side}_counts"] = _counts(str(r["best_params"][f"penalty_{side}"]) for r in successes)
            refined = [r["best_params"][f"penalty_{side}"] for r in successes if r["selected_iteration"] > 0]
            row[f"penalty_{side}_refined_lower_edge"] = sum(v == min(config[f"penalties_{side}"]) for v in refined) if config else 0
            row[f"penalty_{side}_refined_upper_edge"] = sum(v == max(config[f"penalties_{side}"]) for v in refined) if config else 0
        row["selected_params_counts"] = _counts(json.dumps(r["best_params"], sort_keys=True) for r in successes)
        for method in ("SMART", "SMART_fixed", "RRR", "SRRR", "SOFAR", "RSSVD"):
            ref = next(r for r in paper if r["method"] == method and float(r["x"]) == setting.sigma0)
            row[f"paper_{method}_mean"], row[f"paper_{method}_se"] = float(ref["mean"]), float(ref["se"])
        rows.append(row)
    metadata = dict(seed_ids=list(seed_ids), expected_runs=6*len(seed_ids), recorded_runs=len(records),
        runner_config=config, implementation_fingerprint=next(iter(implementations)) if implementations else None,
        earlier_runner_config=json.loads(next(iter(old_configurations))) if old_configurations else None,
        earlier_implementation_fingerprint=next(iter(old_implementations)) if old_implementations else None,
        training_rows=200, additional_validation_rows=100, refit_on_all_data=False,
        training_fingerprints_verified_against_previous=len(records), previous_root=str(Path(previous_root).resolve()),
        result_root=str(Path(result_root).resolve()), paper_reference=str(Path(reference_path).resolve()),
        paper_pdf_hashes_verified=provenance, paper_repetitions=100,
        missing_cells=[dict(setting=suffix,seed_id=seed) for suffix in settings for seed in seed_ids
                       if (suffix,seed) not in records],
        comparison_has_equal_tuning_data=False,
        artifact_validation="identities, seeds, configurations, fingerprints, external split, candidate/iterate selection and prior training-data equality; no data regenerated")
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
    report = ["# SparseSMART with 200 training and 100 additional tuning observations", "",
        f"The six-noise-level Model I pilot contains {sum(r['complete'] for r in rows)}/{metadata['expected_runs']} successful runs, {sum(r['failed'] for r in rows)} failed runs/searches and {sum(r['missing'] for r in rows)} missing cells. Only SparseSMART was run. Seed IDs: {', '.join(map(str,metadata['seed_ids']))}.", "",
        "Each new coefficient estimate is fitted on the complete original 200 training observations. A separate sample of 100 observations from the same target coefficient is used to select penalties and an iterate. There is no full-data refit and the validation score is a reused selection score, not an independent test-performance estimate. Training/source and evaluation-truth fingerprints were matched against the earlier saved 200-row datasets.", "",
        "Historical references use different data budgets: the earlier SparseSMART pilot split 200 observations into 160 training and 40 validation rows; this pilot adds 100 tuning observations. The paper uses 200 target observations and its own tuning procedures. This is not an equal-data-budget comparison with the paper, and changes from the earlier pilot cannot be attributed to the optimizer alone. Other methods were not rerun.", "",
        "The reported coefficient metric is ||C_hat-C_star||_F / sqrt(p*q). New and earlier pilot means/SEs use the requested seeds; paper means/SEs summarize 100 repetitions and were digitized from existing figure vectors. Figure aggregates do not permit paired seed-level significance tests. Failed partial fits never count as successful observations; means are conditional on success wherever failures occur.", "",
        "| Source noise | Complete / requested | Failed (all candidates) / missing | New error (SE) | Earlier 160/40 error (SE) | Paper SMART | Paper fixed-rank SMART |",
        "|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        report.append(f"| {r['sigma0']:g} | {r['complete']}/{r['expected_runs']} | {r['failed']} ({r['all_candidates_failed']}) / {r['missing']} | {_fmt(r['final_mean'])} ({_fmt(r['final_se'])}) | {_fmt(r['earlier_mean'])} ({_fmt(r['earlier_se'])}) | {_fmt(r['paper_SMART_mean'])} | {_fmt(r['paper_SMART_fixed_mean'])} |")
    report += ["", "## Selection and optimization", "",
        "Reaching the iteration limit is not convergence. Optimizer convergence concerns its final state, whereas selected convergence concerns the iterate chosen by validation. Initializer selections (iteration zero) do not identify a preference for refinement penalties; tied initializers can select the first grid pair. Exact parameter counts and failure reasons appear in comparison.csv.", "",
        "| Noise | Initializer chosen | Selected iteration mean (range) | Optimizer converged / at limit | Selected converged | Candidate failures / total | Mean supports U / V |",
        "|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        report.append(f"| {r['sigma0']:g} | {r['chosen_initializer']} | {_fmt(r['selected_iteration_mean'],1)} ({r['selected_iteration_min']}--{r['selected_iteration_max']}) | {r['optimization_converged']} / {r['max_iterations']} | {r['selected_converged']} | {r['candidate_failed']}/{r['candidate_total']} | {_fmt(r['support_u_mean'],1)} / {_fmt(r['support_v_mean'],1)} |")
    report += ["", "Noisy-source empirical mode, when configured below, bypasses the sufficient source-accuracy gate. Numerical success is not theorem certification. Full complement budgets allow all 475/225 noisy-source coordinates (25/25 for exact sources); penalties determine which entries remain nonzero.", "",
        "The CSV includes selected parameter frequencies, support ranges, iteration ranges, termination counts, candidate failures, timings, initializer errors and every paper reference method. validation_audit.json records the common implementation/configuration and verified source-PDF hashes. Original runners, results and summaries were preserved.", "",
        "The inherited validation_fraction and split_seed fields below are inactive. Explicit validation data overrides internal splitting: every candidate fits 200 rows and is scored on the separate 100 rows.", "",
        "```json", json.dumps(metadata["runner_config"], indent=2, sort_keys=True), "```", ""]
    (output_dir / "comparison.md").write_text("\n".join(report))
    _plot(rows, paper, metadata, output_dir / "comparison.png")
    return output_dir / "comparison.csv"


def _plot(rows, paper, metadata, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    x = np.arange(len(rows))
    fig, ax = plt.subplots(figsize=(11.5, 7.8))
    for method, color, label in (("SMART", "#318553", "Paper SMART"),
                                  ("SMART_fixed", "#718096", "Paper fixed-rank SMART"),
                                  ("RRR", "#b38a52", "Paper RRR")):
        ax.plot(x, [r[f"paper_{method}_mean"] for r in rows], ls="--", color=color, lw=1.5, label=label)
    ax.errorbar(x, [np.nan if r["earlier_mean"] is None else r["earlier_mean"] for r in rows],
                yerr=[0 if r["earlier_se"] is None else r["earlier_se"] for r in rows],
                ls=":", marker="s", ms=4, capsize=3, color="#b06042", label="Earlier SparseSMART: 160 train + 40 tune")
    ax.errorbar(x, [np.nan if r["final_mean"] is None else r["final_mean"] for r in rows],
                yerr=[0 if r["final_se"] is None else r["final_se"] for r in rows],
                marker="o", ms=6, lw=2.4, capsize=4, color="#112f4b", label="SparseSMART: 200 train + 100 tune (mean ± SE)")
    for xx, row in zip(x, rows):
        if row["complete"] < row["expected_runs"]:
            ax.annotate(f"{row['complete']}/{row['expected_runs']} fits", (xx, row["final_mean"] or 0),
                        xytext=(4, 9), textcoords="offset points", fontsize=9)
    ax.set_xticks(x, [f"{r['sigma0']:g}" for r in rows])
    ax.set_xlabel("Source-noise standard deviation (tested levels)", fontsize=12)
    ax.set_ylabel(r"Coefficient error $\|\widehat C-C^*\|_F/\sqrt{pq}$", fontsize=12)
    ax.set_ylim(bottom=0)
    ax.grid(alpha=.18)
    ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Model I: SparseSMART with 200 training observations", fontsize=17, y=.965)
    fig.text(.5, .915, f"100 additional independent tuning observations; {len(metadata['seed_ids'])} seeds; no full-data refit", ha="center", fontsize=12)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(.5, .075), ncol=2, frameon=False, fontsize=10)
    fig.text(.5, .026, "Historical curves use different tuning data. Paper: 100 repetitions, digitized figures; no baseline reruns.", ha="center", fontsize=10)
    fig.tight_layout(rect=(.015, .225, .985, .89))
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=HERE / "result" / "sparse_smart_external")
    parser.add_argument("--previous-root", type=Path, default=HERE / "result" / "sparse_smart_tuned")
    parser.add_argument("--paper-reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output-dir", type=Path, default=HERE / "result" / "sparse_smart_external" / "summary")
    parser.add_argument("--seed-file", type=Path, default=DEFAULT_SEED_FILE)
    parser.add_argument("--seed-count", type=int, default=5)
    args = parser.parse_args(argv)
    if not 1 <= args.seed_count <= 100:
        parser.error("seed-count must lie between 1 and 100")
    rows, paper, metadata = summarize(args.result_root, args.paper_reference,
        seed_ids=tuple(range(args.seed_count)), previous_root=args.previous_root, seed_file=args.seed_file)
    print(write_outputs(rows, paper, metadata, args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
