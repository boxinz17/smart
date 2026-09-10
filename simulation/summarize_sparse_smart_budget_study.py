"""Audit continuous SparseSMART budget studies without fitting estimators.

Validation gains describe selection on the reused tuning sample. Coefficient
truth is used only to audit descriptive errors, never to choose a budget.
"""
from __future__ import annotations

import argparse
from collections import Counter, OrderedDict, defaultdict
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from sparse_smart_selection import (selection_value, selection_values, selection_keys, validation_winner,
                                    validate_selected_score, prediction_selection_score, pairwise_selection,
                                    history_winner, PAIRWISE_RULE, pairwise_loss_difference)

import external_validation_data
import run_sparse_smart_budget_study as runner
import run_sparse_smart_external as external_runner
from batch_manifest import manifest_for_resume
from run_restricted_rrr import DEFAULT_SEED_FILE, experiment_settings, load_experiment_seeds
from run_sparse_smart import _atomic_json_dump, _digest_json, _json_value

HERE = Path(__file__).resolve().parent
DEFAULT_ROOT = HERE / "result" / "sparse_smart_budget_study"
DIFFICULT_SETTINGS = ((2, "rs=5"), (2, "rs=7"), (3, "sigma0=0.5"))
DEFAULT_BUDGETS = (500, 1000, 2000, 4000, 8000)


class _RegeneratedDataCache(OrderedDict):
    """Bound retained audit arrays while counting every distinct dataset key."""

    def __init__(self, max_entries=16):
        super().__init__()
        if type(max_entries) is not int or max_entries < 1:
            raise ValueError("max_entries must be a positive integer")
        self.max_entries = max_entries
        self.seen_keys = set()

    def __getitem__(self, key):
        value = super().__getitem__(key)
        self.move_to_end(key)
        return value

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self.move_to_end(key)
        self.seen_keys.add(key)
        while len(self) > self.max_entries:
            self.popitem(last=False)


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _number(value, label, *, nonnegative=True):
    _require(not isinstance(value, bool) and isinstance(value, (int, float))
             and math.isfinite(value) and (not nonnegative or value >= 0), f"Invalid {label}")
    return float(value)


def _integer(value, label, minimum=0):
    _require(type(value) is int and value >= minimum, f"Invalid {label}")
    return value


def _same(a, b, label):
    _require(math.isclose(_number(a, label, nonnegative=False), _number(b, label, nonnegative=False),
                         rel_tol=1e-10, abs_tol=1e-12), f"{label} mismatch")


def _fingerprint(value, label):
    _require(isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value),
             f"Invalid {label}")


def _stats(values):
    values = list(values)
    if not values:
        return dict(n=0, mean=None, se=None)
    return dict(n=len(values), mean=float(np.mean(values)),
                se=float(np.std(values, ddof=1) / np.sqrt(len(values))) if len(values) > 1 else None)


def _thresholds(absolute, relative):
    return (_number(absolute, "absolute gain threshold"), _number(relative, "relative gain threshold"))


def compare_caps(base, extended, *, absolute_threshold=1e-4, relative_threshold=1e-3):
    """Describe a validation gain; unresolved searches cannot establish a plateau."""
    absolute, relative = _thresholds(absolute_threshold, relative_threshold)
    available = bool(base["success"] and extended["success"])
    complete = available and base["coverage_complete"] and extended["coverage_complete"]
    gain = relative_gain = threshold = None
    if available:
        a, b = _number(base["validation_mse"], "base score"), _number(extended["validation_mse"], "extended score")
        if pairwise_selection([base, extended]):
            difference = _number(extended["validation_comparisons"][str(base["iteration_budget"])],
                                 "pairwise cap loss difference", nonnegative=False)
            _require(difference <= 0., "Cumulative eligible validation minimum increased")
            gain = -difference
        else:
            ranking = selection_values([base, extended])
            keys = selection_keys([base, extended])
            _require(keys[1] <= keys[0], "Cumulative eligible validation minimum increased")
            gain = max(0., ranking[0]-ranking[1] if ranking[0] != ranking[1] else a-b)
        relative_gain = gain/a if a > 0 else None
        threshold = max(absolute, relative*a)
    material = gain > threshold if complete else None
    return dict(base_budget=base["iteration_budget"], extended_budget=extended["iteration_budget"],
        available_eligible_pair=available, complete_pair=bool(complete), validation_gain=gain,
        relative_validation_gain=relative_gain, absolute_threshold=absolute, relative_threshold=relative,
        material_gain_threshold=threshold, material_gain=material,
        interpretation=("unresolved_extension" if not complete else
                        "material_validation_gain" if material else "below_declared_material_threshold"),
        base_validation_mse=base["validation_mse"], extended_validation_mse=extended["validation_mse"],
        base_coefficient_error=base.get("coefficient_error"), extended_coefficient_error=extended.get("coefficient_error"),
        coefficient_error_change=(extended["coefficient_error"]-base["coefficient_error"]
                                  if available and base.get("coefficient_error") is not None
                                  and extended.get("coefficient_error") is not None else None))


def _budget_pairs(budgets):
    pairs=list(zip(budgets,budgets[1:]))
    if 2000 in budgets and 8000 in budgets and (2000,8000) not in pairs:
        pairs.append((2000,8000))
    return pairs


def _transitions(caps,absolute,relative):
    by_budget={cap["iteration_budget"]:cap for cap in caps}
    return [compare_caps(by_budget[a],by_budget[b],absolute_threshold=absolute,relative_threshold=relative)
            for a,b in _budget_pairs(tuple(by_budget))]


def _configuration(record, setting):
    options = dict(record["configuration"]["runner"])
    legacy_validation = "validation_iterations" not in options
    if legacy_validation:
        options["validation_iterations"] = ()
    for name in ("iteration_budgets", "init_penalties", "penalties_u", "penalties_v", "validation_iterations"):
        if name in options:
            options[name] = tuple(options[name])
    config = runner.RunnerConfig(**options)
    config.validate()
    expected = _json_value(runner.resolved_configuration(setting, config))
    if legacy_validation:
        expected["runner"].pop("validation_iterations")
        for key in ("validation_iterations", "validation_schedule", "validation_state_policy"):
            expected.pop(key)
    _require(record["configuration"] == expected,
             "Resolved study configuration mismatch")
    _require(record["configuration"]["runner"]["checkpoint_execution"] == "continuous",
             "Budget study requires continuous candidate trajectories")
    return config


_VALIDATION_AUDIT_COUNTS = ("evaluated_points", "factor_verified_points", "metadata_only_points",
                          "verified_pairwise_comparisons", "metadata_only_pairwise_comparisons")


def _validation_audit_totals(values=()):
    values = list(values)
    result = {key: sum(value.get(key, 0) for value in values) for key in _VALIDATION_AUDIT_COUNTS}
    result["all_validation_points_factor_verified"] = result["metadata_only_points"] == 0
    return result


def _validation_bests(history):
    """Replay scalar decisions, retaining every historical incumbent iteration."""
    history_winner(history)  # Includes ordered pairwise-incumbent consistency checks.
    best = None
    retained = set()
    pairwise = pairwise_selection(history)
    keys = selection_keys(history, "loss") if not pairwise else None
    for i, row in enumerate(history):
        improves = (best is None or (row["selection_comparison"]["loss_difference"] < 0
                    if pairwise else keys[i] < keys[best]))
        if improves:
            best = i
            retained.add(row["iteration"])
    return retained


def _data(setting, seed, config, cache, generator):
    key = (setting.n, setting.p, setting.q, setting.sigma0, int(seed), config.n_validation, config.validation_seed_tag)
    if key not in cache:
        data = external_validation_data.generate_external_validation(
            n_train=setting.n, p=setting.p, q=setting.q, sigma0=setting.sigma0, random_seed=int(seed),
            n_validation=config.n_validation, sigma=.5, r_star=5, r0_star=10,
            seed_tag=config.validation_seed_tag, generate_data_fn=generator)
        cache[key] = dict(
            training_observed_input_fingerprint=external_runner.old_runner._array_fingerprint(data, ("X", "Y", "C0")),
            validation_observed_input_fingerprint=external_runner.old_runner._array_fingerprint(data, ("X_validation", "Y_validation")),
            evaluation_truth_fingerprint=external_runner.old_runner._array_fingerprint(data, ("C_star",)),
            validation_seed_metadata=_json_value(data["validation_seed_metadata"]),
            X=data["X_validation"], Y=data["Y_validation"], truth=data["C_star"])
    return cache[key]


def _ordered(rows, last, label):
    indices = [_integer(row["iteration"], f"{label} iteration") for row in rows]
    _require(indices == sorted(set(indices)) and all(i <= last for i in indices), f"Unordered {label}")
    return indices


def _diagnostics(record):
    names = ("objective", "objective_change", "relative_step_norm", "projected_gradient_norm",
             "mapping_displacement", "proximal_uncertainty", "mapping_precision_limited", "step_norm")
    result = {name: record.get(name) for name in names}
    for name, value in result.items():
        if value is not None and name != "mapping_precision_limited":
            _number(value, name, nonnegative=name not in ("objective_change",))
    if result["mapping_precision_limited"] is not None:
        _require(type(result["mapping_precision_limited"]) is bool, "Invalid precision-limited flag")
    m, u, total = (result[k] for k in ("mapping_displacement", "proximal_uncertainty", "projected_gradient_norm"))
    if all(v is not None for v in (m, u, total)):
        _same(total, m+u, "Mapping displacement plus uncertainty")
    return result


def _interval(cp):
    return {name:cp[name] for name in ("interval_start_iteration","interval_accepted_steps",
            "interval_objective_change_sum","interval_max_relative_step_norm")}


def validate_record(record, path, *, setting, model_id, exp_id, seed_id, random_seed,
                    data_cache, generate_data_fn):
    """Return audited per-cap summaries; regenerate data only, never fit a model."""
    _require(record["schema_version"] == 1 and record["method"] == "SparseSMARTBudgetStudy", "Wrong study schema")
    _require(record["model"] == f"model{model_id+1}" and record["experiment"] == f"exp{exp_id+1}"
             and record["rd_seed_id"] == seed_id and record["random_seed"] == int(random_seed)
             and record["setting"] == asdict(setting), "Study cell identity mismatch")
    expected_path = runner.result_path(Path('.'), model=record["model"], experiment=record["experiment"],
                                       setting=setting, seed_id=seed_id)
    _require(path.name == expected_path.name and path.parent.name == record["experiment"]
             and path.parent.parent.name == record["model"], "Study filename mismatch")
    config = _configuration(record, setting)
    budgets = tuple(config.iteration_budgets)
    interval = config.checkpoint_interval
    _require(record["n_train"] == setting.n and record["n_validation"] == config.n_validation == 100,
             "Study training/validation counts mismatch")
    _require(record["all_training_rows_used"] is True and record["training_matches_legacy"] is True
             and record["refit_on_all_data"] is False and record["theorem_certified"] is False
             and record["source_check_mode"] == "empirical",
             "Study data-use or applicability flags mismatch")
    inapplicability = setting.inapplicability_reason()
    _require(type(record["applicable"]) is bool and record["applicable"] == (inapplicability is None),
             "Study structural applicability mismatch")
    arguments = dict(n=setting.n, p=setting.p, q=setting.q, sigma0=setting.sigma0,
                     sigma=.5, r_star=5, r0_star=10, random_seed=int(random_seed))
    _require(record["generator_arguments"] == arguments, "Study generator arguments mismatch")
    identity = {k:record[k] for k in ("schema_version","method","model","experiment","rd_seed_id",
                                     "random_seed","setting","configuration","generator_arguments")}
    _require(record["configuration_fingerprint"] == _digest_json(identity), "Configuration fingerprint mismatch")
    for key in ("configuration_fingerprint","implementation_fingerprint","training_observed_input_fingerprint",
                "validation_observed_input_fingerprint","evaluation_truth_fingerprint","input_fingerprint"):
        _fingerprint(record[key], key)
    _require(record["input_fingerprint"] == _digest_json(dict(training_observed=record["training_observed_input_fingerprint"],
        validation_observed=record["validation_observed_input_fingerprint"], evaluation_truth=record["evaluation_truth_fingerprint"])),
        "Combined input fingerprint mismatch")
    data = _data(setting, random_seed, config, data_cache, generate_data_fn)
    for key in ("training_observed_input_fingerprint","validation_observed_input_fingerprint",
                "evaluation_truth_fingerprint","validation_seed_metadata"):
        _require(record[key] == data[key], f"Regenerated {key} mismatch")
    if inapplicability is not None:
        _require(record["status"] == "inapplicable" and record["success"] is False
                 and record["failure_reason"] == inapplicability, "Inapplicable study status mismatch")
        _require(all(record[key] == [] for key in ("selection_history", "trajectories", "cap_outcomes"))
                 and record["tuning_diagnostics"] == {}, "Inapplicable study contains fitted trajectories")
        _require(all(record[key] is None for key in ("selected_budget", "selected_candidate_id",
                 "selected_iteration", "validation_loss", "avg_err")), "Inapplicable study contains a selected fit")
        _require(_number(record["fit_time_sec"], "inapplicable fit time") == 0,
                 "Inapplicable study reports fitting time")
        return dict(model=record["model"], model_id=model_id, experiment=record["experiment"],
            setting=setting.suffix, seed_id=seed_id, status="inapplicable", applicable=False,
            inapplicability_reason=inapplicability, missing=False, caps=[],
            trajectory_status_counts={}, trajectory_final_diagnostics=[], failed_trajectories=0,
            verified_factor_states=0, validation_audit=_validation_audit_totals(),
            configuration=record["configuration"]["runner"],
            implementation_fingerprint=record["implementation_fingerprint"])
    grid = record["configuration"]["candidate_grid"]
    trajectories = record["trajectories"]
    _require(len(trajectories) == len(grid), "Missing candidate trajectory")
    stable_selection = "selection_score" in record
    pairwise = pairwise_selection([record])
    reference = record.get("validation_reference_prediction")
    reference_mapping = record.get("trajectory_validation_references")
    merged = "tuning_shard_merge" in record
    if merged:
        _require(pairwise and stable_selection
                 and record.get("validation_reference_scope") == "trajectory"
                 and record.get("selection_reference") == {
                     "kind": "per_trajectory", "mapping": "trajectory_validation_references"}
                 and reference is None,
                 "Merged tuning record must declare per-trajectory validation references")
        _require(isinstance(reference_mapping, dict)
                 and set(reference_mapping) == {str(j) for j in range(len(grid))},
                 "Missing or malformed trajectory validation reference mapping")
        for item in reference_mapping.values():
            _require(isinstance(item, dict) and set(item) == {
                "validation_reference_prediction", "selection_reference"},
                "Malformed trajectory validation reference entry")
    else:
        _require(reference_mapping is None and record.get("validation_reference_scope") in (None, "fit"),
                 "Ordinary tuning record cannot carry per-trajectory validation references")
    points = {}
    verified_factors = 0
    validation_audit = _validation_audit_totals()
    for j, trajectory in enumerate(trajectories):
        _require(trajectory["grid_candidate_id"] == j and trajectory["params"] == grid[j], "Trajectory grid mismatch")
        last = _integer(trajectory["n_iter"], "trajectory iterations")
        _require(last <= budgets[-1] and type(trajectory["success"]) is bool, "Invalid terminal trajectory")
        history = trajectory["validation_history"]
        if merged:
            reference_entry = reference_mapping[str(j)]
            reference = reference_entry["validation_reference_prediction"]
            if history:
                reference_array = np.asarray(reference, dtype=float)
                _require(reference_array.shape == data["Y"].shape and np.isfinite(reference_array).all()
                         and isinstance(reference_entry["selection_reference"], dict),
                         "Invalid trajectory validation reference")
        indices = _ordered(history, last, "validation history")
        losses = {row["iteration"]:_number(row["loss"], "validation loss") for row in history}
        ranking = dict(zip(indices, selection_values(history, "loss")))
        _require(pairwise_selection(history) == pairwise or not history, "Trajectory comparison rule mismatch")
        _require(all(("selection_score" in row) == stable_selection for row in history),
                 "Trajectory validation selection scheme mismatch")
        if indices:
            _require(indices[0] == 0, "Validation history omits initialization")
        optimization = trajectory["history"]
        _ordered(optimization, last, "optimization history")
        optimization_by_iteration = {row["iteration"]:row for row in optimization}
        for row in optimization:
            _diagnostics(row)
        factor_scores = {}
        for key, factor in trajectory["factor_states"].items():
            t = _integer(factor["iteration"], "factor iteration")
            _require(key == str(t) and t <= last, "Factor key/iteration mismatch")
            left, d, right = (np.asarray(factor[name], dtype=float) for name in ("left","singular_values","right"))
            _require(left.shape == (setting.p,setting.target_rank) and d.shape == (setting.target_rank,)
                     and right.shape == (setting.q,setting.target_rank) and np.all(d > 0)
                     and all(np.isfinite(a).all() for a in (left,d,right)), "Invalid factor dimensions or values")
            _require(np.allclose(left.T@left,np.eye(len(d)),atol=1e-8,rtol=1e-8)
                     and np.allclose(right.T@right,np.eye(len(d)),atol=1e-8,rtol=1e-8), "Nonorthonormal coefficient factors")
            coefficient = (left*d)@right.T
            prediction = ((data["X"]@left)*d)@right.T
            score = float(np.mean((data["Y"]-prediction)**2))
            error = float(np.linalg.norm(coefficient-data["truth"])/np.sqrt(setting.p*setting.q))
            factor_scores[key] = score,error
            if stable_selection and t in ranking:
                relative_score = prediction_selection_score(prediction, data["Y"], reference)
                _same(relative_score, ranking[t], "Factor prediction/selection history")
            if t in losses:
                _same(score,losses[t],"Factor prediction/validation history")
            verified_factors += 1
        checkpoints = trajectory["checkpoints"]
        checkpoint_indices = [_integer(cp["checkpoint_iteration"],"checkpoint iteration") for cp in checkpoints]
        _require(checkpoint_indices == sorted(set(checkpoint_indices)) and all(t <= last for t in checkpoint_indices),
                 "Unordered checkpoints")
        _require(trajectory["checkpoint_iterations"] == checkpoint_indices, "Checkpoint index list mismatch")
        terminal_stationary = (trajectory["success"] and trajectory["status"] == "converged"
                               and trajectory["termination_reason"] == "stationarity")
        scheduled = {0, *budgets, *range(interval, budgets[-1]+1, interval)}
        validation_scheduled = set(runner.validation_schedule(config))
        if trajectory["success"]:
            _require(terminal_stationary or (last == budgets[-1] and trajectory["status"] == "completed"
                     and trajectory["termination_reason"] == "max_iterations"), "Inconsistent successful trajectory termination")
            expected = {t for t in scheduled if t <= last} | ({last} if terminal_stationary else set())
            expected_validation = {t for t in validation_scheduled if t <= last} | ({last} if terminal_stationary else set())
            _require(set(checkpoint_indices) == expected and set(indices) == expected_validation,
                     "Successful trajectory omits scheduled checkpoint or validation")
        elif optimization:
            # A terminal failed accepted state may lack validation/capture, but
            # failure cannot erase earlier scheduled, completed prefixes.
            required = {t for t in scheduled if t < last}
            allowed = {t for t in scheduled if t <= last}
            validation_required = {t for t in validation_scheduled if t < last}
            validation_allowed = {t for t in validation_scheduled if t <= last}
            _require(required <= set(checkpoint_indices) <= allowed
                     and validation_required <= set(indices) <= validation_allowed,
                     "Failed trajectory omits earlier scheduled checkpoint or validation")
        else:
            _require(last == 0 and not checkpoints and not history and not factor_scores,
                     "Trajectory without optimization history contains refinement states")
        best_iterations = _validation_bests(history)
        expected_factors = set(checkpoint_indices) | best_iterations
        _require(set(factor_scores) == {str(t) for t in expected_factors},
                 "Factor states differ from certified checkpoints and historical validation bests")
        _require(expected_factors <= set(optimization_by_iteration),
                 "Retained validation state omits optimization diagnostics")
        def prediction(row):
            return runner._factor_prediction(trajectory["factor_states"][str(row["iteration"])], data["X"])
        # The full scalar decision chain was checked above. Periodic states and
        # every historical best are independently rescored; early losers have
        # no saved parameter state and must not be called factor-verified.
        validation_audit["evaluated_points"] += len(indices)
        validation_audit["factor_verified_points"] += len(factor_scores)
        validation_audit["metadata_only_points"] += len(indices)-len(factor_scores)
        if pairwise:
            by_iteration = {row["iteration"]: row for row in history}
            for row in history:
                comparison = row["selection_comparison"]
                incumbent = comparison["incumbent_iteration"]
                if incumbent is None:
                    continue
                if str(row["iteration"]) in factor_scores and str(incumbent) in factor_scores:
                    actual = pairwise_loss_difference(prediction(row), data["Y"], prediction(by_iteration[incumbent]))
                    difference = comparison["loss_difference"]
                    _same(actual, difference, "Pairwise validation difference disagrees with saved factors")
                    _require((actual < 0) == (difference < 0), "Pairwise validation difference sign mismatch")
                    validation_audit["verified_pairwise_comparisons"] += 1
                else:
                    validation_audit["metadata_only_pairwise_comparisons"] += 1
        previous_checkpoint = 0
        for cp in checkpoints:
            t, selected = cp["checkpoint_iteration"], _integer(cp["selected_iteration"],"selected iteration")
            _require(cp["interval_start_iteration"] == previous_checkpoint
                     and cp["interval_accepted_steps"] == t-previous_checkpoint, "Checkpoint interval mismatch")
            if cp["interval_objective_change_sum"] is not None:
                _require(_number(cp["interval_objective_change_sum"],"interval objective change",nonnegative=False) <= 1e-12,
                         "Positive accepted objective change sum")
            if cp["interval_max_relative_step_norm"] is not None:
                _number(cp["interval_max_relative_step_norm"],"interval maximum relative step")
            previous_checkpoint = t
            _require(t in losses and selected in losses and selected <= t, "Checkpoint validation point missing")
            best = history_winner(row for row in history if row["iteration"] <= t)["iteration"]
            _require(selected == best, "Checkpoint does not retain earliest validation minimum")
            _require(pairwise_selection([cp]) == pairwise, "Checkpoint selection rule mismatch")
            _require(cp["terminal_factor_key"] == str(t) and cp["selected_factor_key"] == str(selected), "Checkpoint factor key mismatch")
            _same(cp["terminal_validation_mse"],losses[t],"Checkpoint terminal score")
            _same(cp["selected_validation_mse"],losses[selected],"Checkpoint selected score")
            if stable_selection:
                _same(cp["selection_score"], ranking[selected], "Checkpoint selected relative score")
                _same(cp["terminal_selection_score"], ranking[t], "Checkpoint terminal relative score")
            for which in ("terminal","selected"):
                score,error = factor_scores[cp[f"{which}_factor_key"]]
                _same(cp[f"{which}_validation_mse"],score,f"{which} prediction MSE")
                _same(cp[f"{which}_coefficient_error"],error,f"{which} coefficient error")
                _require(cp[f"{which}_record"]["iteration"] == (t if which == "terminal" else selected),
                         "Checkpoint diagnostic iteration mismatch")
                _diagnostics(cp[f"{which}_record"])
                _require(cp[f"{which}_record"] == optimization_by_iteration.get(t if which == "terminal" else selected),
                         "Checkpoint diagnostic record differs from trajectory history")
            _require(type(cp["success"]) is bool and type(cp["optimization_converged"]) is bool
                     and type(cp["selected_converged"]) is bool, "Checkpoint success/convergence flag invalid")
            stationary = cp["termination_reason"] == "stationarity"
            _require(cp["success"] and cp["status"] in ("completed","converged")
                     and cp["endpoint_selected"] == (selected == t), "Invalid certified checkpoint")
            _require(cp["optimization_converged"] == stationary
                     and cp["selected_converged"] == (stationary and selected == t), "Checkpoint convergence inconsistency")
            if stationary:
                residual = _number(cp["terminal_record"]["projected_gradient_norm"],"stationary residual")
                _require(cp["status"] == "converged" and residual <= config.stationarity_tol
                         and terminal_stationary and t == last
                         and cp["terminal_record"].get("mapping_domain_reason") is None,
                         "False checkpoint stationarity")
            else:
                _require(cp["status"] == "completed", "Nonstationary checkpoint marked converged")
            points[(j,t)] = cp
    rows = record["selection_history"]
    _require(len(rows) == len(budgets)*len(grid), "Missing analysis-cap candidate rows")
    for i,row in enumerate(rows):
        budget,j = budgets[i//len(grid)],i%len(grid)
        _require(row["candidate_id"] == i and row["grid_candidate_id"] == j
                 and row["iteration_budget"] == budget and row["params"] == grid[j], "Cap/grid candidate identity mismatch")
        _require(type(row["success"]) is bool and type(row["budget_reached"]) is bool, "Cap eligibility/coverage flag invalid")
        trajectory = trajectories[j]
        _require(row["trajectory_status"] == trajectory["status"]
                 and row["trajectory_termination_reason"] == trajectory["termination_reason"],
                 "Cap trajectory termination mismatch")
        available = [t for (grid_id,t), cp in points.items() if grid_id == j and t <= budget
                     and cp["success"] and (t > 0 or cp["termination_reason"] == "stationarity")]
        _require(row["success"] == bool(available), "Cap hides or invents an eligible prefix")
        if row["success"]:
            t = row["trajectory_checkpoint_iteration"]
            _require(t == max(available), "Cap row does not use latest attained certified prefix")
            cp = points[(j,t)]
            _require(t <= budget and cp["success"] and (t > 0 or cp["termination_reason"] == "stationarity"),
                     "Failed/uncertified partial prefix made eligible")
            _require(row["n_iter"] == t and row["selected_iteration"] == cp["selected_iteration"], "Retained prefix iteration mismatch")
            _same(row["validation_mse"],cp["selected_validation_mse"],"Retained prefix score")
            validate_selected_score(row, cp)
            reached = t == budget or cp["termination_reason"] == "stationarity"
            _require(row["budget_reached"] == reached, "Unreached cap marked covered")
        else:
            _require(row["validation_mse"] is None and not row["budget_reached"], "Failed partial is eligible or covered")
    def candidate_prediction(row):
        cp = points[(row["grid_candidate_id"], row["trajectory_checkpoint_iteration"])]
        factor = trajectories[row["grid_candidate_id"]]["factor_states"][cp["selected_factor_key"]]
        return runner._factor_prediction(factor, data["X"])
    if pairwise:
        eligible = [row for row in rows if row["success"]]
        validation_winner(eligible, prediction=candidate_prediction, response=data["Y"])
        for budget in budgets:
            validation_winner([row for row in eligible if row["iteration_budget"] == budget],
                              comparison_key="budget_selection_comparison",
                              prediction=candidate_prediction, response=data["Y"])
    outcomes = record["cap_outcomes"]
    _require([c["iteration_budget"] for c in outcomes] == list(budgets), "Cap outcomes missing or unordered")
    caps = []
    for cap in outcomes:
        budget = cap["iteration_budget"]
        current = [r for r in rows if r["iteration_budget"] == budget]
        eligible = [r for r in rows if r["iteration_budget"] <= budget and r["success"]]
        unresolved = [r["grid_candidate_id"] for r in current if not r["budget_reached"]]
        _require(cap["coverage_complete"] == (not unresolved) and cap["unresolved_grid_candidate_ids"] == unresolved
                 and cap["budget_reached_count"] == len(grid)-len(unresolved), "Cap coverage audit mismatch")
        _require(type(cap["coverage_complete"]) is bool and type(cap["success"]) is bool
                 and cap["status"] == ("unresolved" if unresolved else "resolved")
                 and cap["eligible_candidate_count"] == len(eligible)
                 and cap["eligible_grid_count"] == len({r["grid_candidate_id"] for r in eligible}),
                 "Cap status or eligibility count mismatch")
        _require(cap["success"] == bool(eligible), "Cap success/eligibility mismatch")
        result = dict(iteration_budget=budget, success=bool(eligible), coverage_complete=not unresolved,
            reached_candidates=len(grid)-len(unresolved), unresolved_candidate_ids=unresolved,
            validation_mse=None, coefficient_error=None, selected_iteration=None, selected_checkpoint=None,
            selected_near_cap=None, selected_at_cap=None, optimizer_converged=None, selected_converged=None,
            endpoint_diagnostics=None, selected_diagnostics=None, winner_candidate_id=None,
            candidate_endpoints=[])
        for row in current:
            cp = points[(row["grid_candidate_id"],row["trajectory_checkpoint_iteration"]) ] if row["success"] else None
            result["candidate_endpoints"].append(dict(grid_candidate_id=row["grid_candidate_id"],
                budget_reached=row["budget_reached"],eligible=row["success"],
                checkpoint_iteration=cp["checkpoint_iteration"] if cp else None,
                optimization_converged=cp["optimization_converged"] if cp else None,
                diagnostics=_diagnostics(cp["terminal_record"]) if cp else None,
                checkpoint_interval=_interval(cp) if cp else None))
        if eligible:
            winner = validation_winner(eligible)
            cp = points[(winner["grid_candidate_id"],winner["trajectory_checkpoint_iteration"])]
            for name,expected in (("winner_candidate_id",winner["candidate_id"]),
                ("winner_grid_candidate_id",winner["grid_candidate_id"]),("winner_origin_budget",winner["iteration_budget"]),
                ("winner_checkpoint_iteration",winner["trajectory_checkpoint_iteration"]),
                ("selected_iteration",winner["selected_iteration"]),("selected_factor_key",cp["selected_factor_key"])):
                _require(cap[name] == expected,f"Cumulative winner {name} mismatch")
            _same(cap["validation_mse"],winner["validation_mse"],"Cumulative validation minimum")
            validate_selected_score(cap, winner)
            if stable_selection:
                result["selection_score"] = winner["selection_score"]
            _same(cap["coefficient_error"],cp["selected_coefficient_error"],"Cumulative coefficient error")
            for name in ("terminal_validation_mse","terminal_coefficient_error"):
                _same(cap[name],cp[name],name)
            _require(cap["optimization_converged"] == cp["optimization_converged"]
                     and cap["selected_converged"] == cp["selected_converged"],"Cumulative convergence mismatch")
            result.update(validation_mse=winner["validation_mse"],coefficient_error=cp["selected_coefficient_error"],
                selected_iteration=winner["selected_iteration"],selected_checkpoint=winner["trajectory_checkpoint_iteration"],
                winner_candidate_id=winner["candidate_id"],selected_near_cap=budget-winner["selected_iteration"] <= interval,
                selected_at_cap=winner["selected_iteration"] == budget,
                optimizer_converged=cp["optimization_converged"],selected_converged=cp["selected_converged"],
                endpoint_diagnostics=_diagnostics(cp["terminal_record"]),selected_diagnostics=_diagnostics(cp["selected_record"]),
                checkpoint_interval=_interval(cp))
        if pairwise:
            _require(pairwise_selection([cap]), "Cap selection rule mismatch")
            comparisons = cap["validation_comparisons"]
            _require(set(comparisons) == {str(base["iteration_budget"]) for base in caps},
                     "Cap pairwise comparison scope mismatch")
            for base in caps:
                difference = comparisons[str(base["iteration_budget"])]
                if base["success"] and result["success"]:
                    actual = pairwise_loss_difference(candidate_prediction(rows[result["winner_candidate_id"]]),
                        data["Y"], candidate_prediction(rows[base["winner_candidate_id"]]))
                    _same(difference, actual, "Pairwise cap validation difference")
                    _require((difference < 0) == (actual < 0), "Pairwise cap comparison sign mismatch")
                else:
                    _require(difference is None, "Failed cap has a pairwise loss comparison")
            result.update(selection_rule=PAIRWISE_RULE, validation_comparisons=comparisons)
        caps.append(result)
    final = outcomes[-1]
    if final["success"]:
        validate_selected_score(record, final)
    _require(type(record["success"]) is bool and record["success"] == final["success"]
             and record["status"] == ("complete" if final["coverage_complete"] else
                                     "partial" if final["success"] else "all_candidates_failed"),
             "Overall status/coverage mismatch")
    for key, other in (("selected_budget","winner_origin_budget"),("selected_candidate_id","winner_candidate_id"),
                       ("selected_iteration","selected_iteration"),("validation_loss","validation_mse"),("avg_err","coefficient_error")):
        _require(record[key] == final[other],f"Overall winner {key} mismatch")
    return dict(model=record["model"],model_id=model_id,experiment=record["experiment"],setting=setting.suffix,
        seed_id=seed_id,status=record["status"],applicable=True,inapplicability_reason=None,missing=False,caps=caps,
        trajectory_status_counts=dict(Counter(t["status"] for t in trajectories)),
        trajectory_final_diagnostics=[dict(grid_candidate_id=t["grid_candidate_id"],status=t["status"],
            success=t["success"],n_iter=t["n_iter"],termination_reason=t["termination_reason"],
            diagnostics=_diagnostics(t["history"][-1]) if t["history"] else None) for t in trajectories],
        failed_trajectories=sum(not t["success"] for t in trajectories),verified_factor_states=verified_factors,
        validation_audit=_validation_audit_totals([validation_audit]),
        configuration=record["configuration"]["runner"],implementation_fingerprint=record["implementation_fingerprint"])


def _aggregate(cells, budgets, absolute, relative, *, model=False):
    cells = [cell for cell in cells if cell.get("applicable", True)]
    if not cells:
        return []
    reports = []
    for a,b in _budget_pairs(budgets):
        pairs = [(c,next(t for t in c["transitions"] if t["base_budget"] == a and t["extended_budget"] == b))
                 for c in cells if not c["missing"]]
        complete = [(c,t) for c,t in pairs if t["complete_pair"]]
        values = [t["validation_gain"] for _,t in complete]
        error_changes = [t["coefficient_error_change"] for _,t in complete]
        unit = "paired seeds within this setting"
        if model:
            grouped = defaultdict(list)
            for c,t in complete:
                grouped[c["seed_id"]].append(t)
            needed = len({(c["experiment"],c["setting"]) for c in cells})
            usable = [ts for ts in grouped.values() if len(ts) == needed]
            values = [float(np.mean([t["validation_gain"] for t in ts])) for ts in usable]
            error_changes = [float(np.mean([t["coefficient_error_change"] for t in ts])) for ts in usable]
            unit = "paired seed averages over all requested settings; incomplete seed averages excluded"
        reports.append(dict(base_budget=a,extended_budget=b,expected_cell_pairs=len(cells),
            recorded_eligible_pairs=sum(t["available_eligible_pair"] for _,t in pairs),complete_cell_pairs=len(complete),
            unresolved_or_missing_cell_pairs=len(cells)-len(complete),
            material_gain_pairs=sum(t["material_gain"] is True for _,t in complete),
            below_threshold_pairs=sum(t["material_gain"] is False for _,t in complete),
            paired_validation_gain=_stats(values),paired_coefficient_error_change=_stats(error_changes),
            statistical_unit=unit,absolute_threshold=absolute,relative_threshold=relative))
    return reports


def summarize(result_root=DEFAULT_ROOT, *, model_ids=(0,1,2),seed_ids=(0,1,2,3,4),seed_file=DEFAULT_SEED_FILE,
              absolute_threshold=1e-4,relative_threshold=1e-3,generate_data_fn=None,manifest_scope=False):
    absolute,relative = _thresholds(absolute_threshold,relative_threshold)
    manifest = None
    manifest_path = Path(result_root)/"budget_study_manifest.json"
    manifest_info = None
    if manifest_scope:
        manifest_path = manifest_for_resume(manifest_path)
        _require(manifest_path is not None, "Missing budget-study manifest or resumable attempt")
        manifest = json.loads(manifest_path.read_text())
        _require(manifest["schema_version"] == 1 and manifest["method"] == "SparseSMARTBudgetStudy", "Wrong study manifest")
        _require(manifest["seed_file_sha256"] == hashlib.sha256(Path(seed_file).read_bytes()).hexdigest(),
                 "Manifest saved-seed fingerprint mismatch")
        model_ids,seed_ids = manifest["models"],manifest["seed_ids"]
        manifest_info = dict(path=str(manifest_path.resolve()),sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                             finished=manifest.get("finished"),errors=manifest.get("errors",[]))
    model_ids,seed_ids = tuple(model_ids),tuple(seed_ids)
    for ids,n,label in ((model_ids,3,"models"),(seed_ids,100,"seeds")):
        _require(ids and len(set(ids)) == len(ids) and all(type(i) is int and 0 <= i < n for i in ids),f"Invalid {label}")
    seeds = load_experiment_seeds(seed_file)
    _require(max(seed_ids) < len(seeds),"Unavailable saved seed")
    requested = []
    for m in model_ids:
        if manifest is None:
            requested.extend((m,e,next(s for s in experiment_settings(m,e) if s.suffix == suffix)) for e,suffix in DIFFICULT_SETTINGS)
        else:
            experiments=manifest["experiments"]
            _require(experiments and len(set(experiments)) == len(experiments)
                     and all(type(e) is int and 0 <= e < 4 for e in experiments),"Invalid manifest experiments")
            for e in experiments:
                selected=runner.settings_for_profile(m,e,manifest["profile"])
                index=manifest["setting_index"]
                if index is not None:
                    _require(len(model_ids) == len(experiments) == 1 and type(index) is int
                             and 0 <= index < len(experiment_settings(m,e)),"Invalid manifest setting index")
                    target=experiment_settings(m,e)[index]
                    selected=tuple(s for s in selected if s==target)
                requested.extend((m,e,s) for s in selected)
    if manifest is not None:
        _require(requested and manifest["expected_cells"] == len(requested)*len(seed_ids),"Manifest scope/count mismatch")
        expected_applicable = sum(s.inapplicability_reason() is None for _, _, s in requested)*len(seed_ids)
        for key, expected in (("expected_applicable", expected_applicable),
                              ("expected_inapplicable", len(requested)*len(seed_ids)-expected_applicable)):
            if key in manifest:
                _require(manifest[key] == expected, "Manifest applicability/count mismatch")
    cells,files,cache,configs = [],[],_RegeneratedDataCache(),set()
    implementations = {True: set(), False: set()}
    generator = generate_data_fn
    budgets = tuple(manifest["configuration"]["iteration_budgets"]) if manifest else None
    for m,e,setting in requested:
        suffix = setting.suffix
        for seed in seed_ids:
            path = runner.result_path(Path(result_root),model=f"model{m+1}",experiment=f"exp{e+1}",setting=setting,seed_id=seed)
            if not path.exists():
                reason = setting.inapplicability_reason()
                cells.append(dict(model=f"model{m+1}",model_id=m,experiment=f"exp{e+1}",setting=suffix,
                    seed_id=seed,applicable=reason is None,inapplicability_reason=reason,
                    missing=True,caps=[],transitions=[]))
                continue
            try:
                raw = path.read_bytes()
                record = json.loads(raw)
                if generator is None:
                    generator = external_runner.old_runner._load_generator()
                cell = validate_record(record,path,setting=setting,model_id=m,exp_id=e,seed_id=seed,
                    random_seed=seeds[seed],data_cache=cache,generate_data_fn=generator)
                configs.add(json.dumps(cell["configuration"],sort_keys=True))
                if manifest is not None:
                    _require(cell["configuration"] == manifest["configuration"], "Record configuration differs from declared manifest")
                implementations[cell["applicable"]].add(cell["implementation_fingerprint"])
                budgets = tuple(cell["configuration"]["iteration_budgets"])
                cell["transitions"] = _transitions(cell["caps"],absolute,relative)
                cells.append(cell)
                files.append(dict(path=str(path.resolve()),sha256=hashlib.sha256(raw).hexdigest()))
            except (KeyError,TypeError,ValueError,OverflowError) as error:
                raise ValueError(f"Invalid budget study record {path}: {error}") from error
    _require(len(configs) <= 1 and all(len(values) <= 1 for values in implementations.values()),
             "Cannot pool different study configurations or implementations within an applicability class")
    budgets = budgets or DEFAULT_BUDGETS
    groups = []
    for m,e,setting in requested:
        suffix=setting.suffix
        group = [c for c in cells if c["model_id"] == m and c["experiment"] == f"exp{e+1}" and c["setting"] == suffix]
        groups.append(dict(model=f"model{m+1}",experiment=f"exp{e+1}",setting=suffix,
            applicable=setting.inapplicability_reason() is None,
            inapplicability_reason=setting.inapplicability_reason(),
            expected_cells=len(group),recorded_cells=sum(not c["missing"] for c in group),
            transitions=_aggregate(group,budgets,absolute,relative)))
    return dict(schema_version=1,method="SparseSMARTBudgetStudySummary",no_fits_performed=True,
        result_root=str(Path(result_root).resolve()),model_ids=list(model_ids),seed_ids=list(seed_ids),
        manifest_scope=manifest_scope,manifest=manifest_info,
        iteration_budgets=list(budgets),absolute_threshold=absolute,relative_threshold=relative,
        material_rule="gain > max(absolute_threshold, relative_threshold * base_validation_mse)",
        threshold_scope="predeclared descriptive practical thresholds; not a significance test",
        expected_cells=len(cells),recorded_cells=len(files),missing_cells=sum(c["missing"] for c in cells),
        expected_applicable_cells=sum(c["applicable"] for c in cells),
        expected_inapplicable_cells=sum(not c["applicable"] for c in cells),
        recorded_inapplicable_cells=sum(not c["applicable"] and not c["missing"] for c in cells),
        missing_inapplicable_cells=sum(not c["applicable"] and c["missing"] for c in cells),
        regenerated_unique_datasets=len(cache.seen_keys),verified_factor_states=sum(c.get("verified_factor_states",0) for c in cells),
        validation_audit=_validation_audit_totals(c.get("validation_audit", {}) for c in cells),
        configurations=[json.loads(c) for c in configs],
        implementation_fingerprints=sorted(set.union(*implementations.values())),
        implementation_fingerprints_by_applicability={"applicable" if key else "inapplicable": sorted(values)
                                                     for key, values in implementations.items()},
        input_files=files,cells=cells,per_setting=groups,
        per_model=[dict(model=f"model{m+1}",transitions=_aggregate([c for c in cells if c["model_id"]==m],budgets,absolute,relative,model=True)) for m in model_ids],
        interpretation="Unreached or failed extensions cannot establish a plateau; retained earlier fits remain eligible. Validation is reused for tuning. Coefficient truth is descriptive only. No universal sufficient-budget or convergence claim follows.")


def _fmt(value):
    return "NA" if value is None else f"{value:.6g}"


def write_outputs(report, output_root):
    output = Path(output_root)
    output.mkdir(parents=True,exist_ok=True)
    _atomic_json_dump(report,output/"summary.json")
    lines = ["# SparseSMART continuous iteration-budget study", "",
        f"Audited {report['recorded_cells']}/{report['expected_cells']} expected cells; {report['missing_cells']} missing. No estimators were fitted by this summary.", "",
        f"The scope contains {report['expected_applicable_cells']} structurally applicable cells and {report['expected_inapplicable_cells']} inapplicable cells ({report['recorded_inapplicable_cells']} recorded, {report['missing_inapplicable_cells']} records missing). Recorded inapplicable cells are audited and excluded from validation-gain denominators; they are not failed fits or unresolved optimization caps.", "",
        ("Scope follows exactly the manifest's declared settings and seeds, including requested cells whose result is still missing."
         if report["manifest_scope"] else "Scope includes the three declared difficult settings for every requested model and saved seed."), "",
        f"A gain is material when it exceeds max({_fmt(report['absolute_threshold'])}, {_fmt(report['relative_threshold'])} × the earlier validation MSE). Both absolute and relative gains are retained per cell in summary.json. These are predeclared practical thresholds, not significance tests.", "",
        "Cumulative minima retain successful earlier checkpoints, including before a later failure. Current records compare each prediction directly with the incumbent using stable pairwise validation loss differences; zero differences preserve budget-major candidate order and the earliest evaluated iterate. The full logged decision chain is checked for consistency. Saved factors independently verify retained models, backed comparisons and pairwise cap gains, even when reported MSE and reference-relative scores round equal. Nonimproving extra validation points retain scalar metadata only: their scores and rejection decisions cannot be independently recomputed from parameter states. Historical records retain their recorded selection rules. A plateau is not established when any candidate has failed or not reached the comparison cap. Genuine earlier stationarity is checked separately.", "",
        (f"Validation evidence: {report.get('validation_audit', {}).get('evaluated_points', 0)} evaluations; "
         f"{report.get('validation_audit', {}).get('factor_verified_points', 0)} independently factor-verified, "
         f"{report.get('validation_audit', {}).get('metadata_only_points', 0)} scalar-only. "
         "Detailed comparison-coverage counts are recorded in summary.json."), "",
        "## Paired validation gains", "",
        "The 2000 → 8000 comparison is included alongside adjacent caps whenever available: two adjacent gains below the threshold can jointly exceed it.", "",
        "Means and SEs below use only seeds with complete candidate coverage at both caps. Incomplete pairs are explicitly excluded; one pair has no estimable SE. Coefficient errors are descriptive diagnostics and never select stopping or a budget.", "",
        "| Model | Setting | Budgets | Complete / requested | Mean gain ± SE | Material | Below threshold | Unresolved / missing |",
        "|---|---|---|---:|---:|---:|---:|---:|"]
    for group in report["per_setting"]:
        for t in group["transitions"]:
            s=t["paired_validation_gain"]
            lines.append(f"| {group['model']} | {group['setting']} | {t['base_budget']} → {t['extended_budget']} | {t['complete_cell_pairs']}/{t['expected_cell_pairs']} | {_fmt(s['mean'])} ± {_fmt(s['se'])} | {t['material_gain_pairs']} | {t['below_threshold_pairs']} | {t['unresolved_or_missing_cell_pairs']} |")
    lines += ["", "## Trajectory outcomes", "",
        "A cell can retain an eligible winner despite a failed candidate trajectory. Every candidate must reach both caps (or stop at verified stationarity) for that cell to contribute plateau evidence.", "",
        "| Model | Setting | Seed | Cell status | Failed trajectories | Trajectory status counts |",
        "|---|---|---:|---|---:|---|"]
    for c in report["cells"]:
        if not c["missing"]:
            statuses=", ".join(f"{status}: {n}" for status,n in sorted(c['trajectory_status_counts'].items()))
            lines.append(f"| {c['model']} | {c['setting']} | {c['seed_id']} | {c['status']} | {c['failed_trajectories']} | {statuses} |")
    lines += ["", "## Optimization and endpoint evidence", "",
        "Small validation gains do not certify optimization convergence. Selected-state proximity to a cap uses one declared checkpoint interval. Stable objective-change sums and maximum relative steps over each checkpoint interval, constrained residuals, displacement and inner-proximal uncertainty are reported separately in each cell's cap records. An earlier retained winner's diagnostics describe that checkpoint, not an unvisited later endpoint; candidate_endpoints reports each candidate's latest prefix at the cap.", "",
        "| Model | Setting | Seed | Cap | Covered candidates | Validation MSE | Selected iterate | Near cap | Stationary selected | Endpoint residual | Inner uncertainty |",
        "|---|---|---:|---:|---:|---:|---:|---|---|---:|---:|"]
    for c in report["cells"]:
        if c["missing"]:
            continue
        for cap in c["caps"]:
            diagnostic=cap["endpoint_diagnostics"] or {}
            lines.append(f"| {c['model']} | {c['setting']} | {c['seed_id']} | {cap['iteration_budget']} | {cap['reached_candidates']} | {_fmt(cap['validation_mse'])} | {cap['selected_iteration']} | {cap['selected_near_cap']} | {cap['selected_converged']} | {_fmt(diagnostic.get('projected_gradient_norm'))} | {_fmt(diagnostic.get('proximal_uncertainty'))} |")
    lines += ["", "## Interpretation and audit", "", report["interpretation"], "",
        "The default design is three models, three difficult settings and five saved seeds, with 100 additional independent tuning observations per cell. Five-seed summaries do not establish a universal budget from a single smoke test. Comparisons with the paper's 100 repetitions have different tuning data and procedures; no competing method is rerun.", "",
        f"Regenerated {report['regenerated_unique_datasets']} distinct data settings and independently scored {report['verified_factor_states']} saved factor states. Configuration identities, seed/data hashes, checkpoint eligibility, coverage and cumulative winners were audited. Per-model paired statistics average all requested settings within each seed before computing SEs, avoiding treating repeated settings as independent seeds.", "",
        "Full per-cell, per-setting and per-model results, numerical diagnostics, failed-trajectory counts and input-file SHA256 hashes are in [summary.json](summary.json).", ""]
    (output/"report.md").write_text("\n".join(lines))
    return output/"report.md"


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root",type=Path,default=DEFAULT_ROOT)
    parser.add_argument("--output-root",type=Path,default=DEFAULT_ROOT/"summary")
    parser.add_argument("--models",type=int,nargs="+",default=[0,1,2])
    parser.add_argument("--seed-ids",type=int,nargs="+",default=[0,1,2,3,4])
    parser.add_argument("--seed-file",type=Path,default=DEFAULT_SEED_FILE)
    parser.add_argument("--manifest-scope",action="store_true",help="Use exactly the manifest's declared models/settings/seeds, including a single-cell smoke check")
    parser.add_argument("--absolute-threshold",type=float,default=1e-4)
    parser.add_argument("--relative-threshold",type=float,default=1e-3)
    args=parser.parse_args(argv)
    report=summarize(args.result_root,model_ids=args.models,seed_ids=args.seed_ids,seed_file=args.seed_file,
        absolute_threshold=args.absolute_threshold,relative_threshold=args.relative_threshold,manifest_scope=args.manifest_scope)
    print(write_outputs(report,args.output_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
