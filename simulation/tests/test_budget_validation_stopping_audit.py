"""Saved-record stopping audits; construct factors, never run optimization."""
from copy import deepcopy
from dataclasses import replace
import importlib.util
from pathlib import Path

import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location("_stop_audit_fixture",
    Path(__file__).with_name("test_summarize_sparse_smart_budget_study.py"))
fixtures = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(fixtures)
runner, summary = fixtures.runner, fixtures.summary


def refresh_identity(value):
    value["configuration_fingerprint"] = fixtures._digest_json({key: value[key] for key in
        ("schema_version", "method", "model", "experiment", "rd_seed_id", "random_seed",
         "setting", "configuration", "generator_arguments")})


def stopped_record():
    # Constant fitted predictions: no validation improvement at either check.
    value = fixtures.fixture(stops={0: 4, 1: 4}, tied=True)
    setting = runner.SimulationSetting(**value["setting"])
    config = replace(summary._configuration(value, setting), n_validation=200,
        validation_patience=4, validation_min_iterations=4)
    data = summary._data(setting, value["random_seed"], config, {}, fixtures.generator)
    value.update(configuration=fixtures._json_value(runner.resolved_configuration(setting, config)),
                 n_validation=200)
    for key in ("training_observed_input_fingerprint", "validation_observed_input_fingerprint",
                "evaluation_truth_fingerprint", "validation_seed_metadata"):
        value[key] = data[key]
    value["input_fingerprint"] = fixtures._digest_json(dict(
        training_observed=value["training_observed_input_fingerprint"],
        validation_observed=value["validation_observed_input_fingerprint"],
        evaluation_truth=value["evaluation_truth_fingerprint"]))
    for trajectory in value["trajectories"]:
        factor = trajectory["factor_states"]["0"]
        score = float(np.mean((data["Y"]-runner._factor_prediction(factor, data["X"]))**2))
        trajectory.update(status="completed", success=True, termination_reason="validation_stop")
        for row in trajectory["validation_history"]:
            t = row["iteration"]
            row.update(loss=score, validation_stopping=dict(significant_improvement=t == 0,
                reference_loss=score, last_significant_iteration=0,
                iterations_since_improvement=t, stop_requested=t == 4))
        trajectory["validation_stopping"] = dict(enabled=True, patience=4, min_iterations=4,
            min_relative_improvement=.001, patience_units="accepted_iterations", metric="raw_validation_mse",
            checks=3, best_loss=score, best_iteration=0, reference_loss=score,
            last_significant_iteration=0, last_check_iteration=4, stopped=True, stop_iteration=4)
        for cp in trajectory["checkpoints"]:
            cp.update(terminal_validation_mse=score, selected_validation_mse=score)
            if cp["checkpoint_iteration"] == 4:
                cp["termination_reason"] = "validation_stop"
    for row in value["selection_history"]:
        trajectory = value["trajectories"][row["grid_candidate_id"]]
        cp = next(cp for cp in trajectory["checkpoints"]
                  if cp["checkpoint_iteration"] == row["trajectory_checkpoint_iteration"])
        row.update(validation_mse=cp["selected_validation_mse"], termination_reason=cp["termination_reason"],
            trajectory_status="completed", trajectory_termination_reason="validation_stop")
    value["cap_outcomes"] = runner._cap_outcomes(value["selection_history"], value["trajectories"],
        value["configuration"], config)
    last = value["cap_outcomes"][-1]
    value.update(status="policy_complete", failure_reason=None, validation_loss=last["validation_mse"],
        selected_budget=last["winner_origin_budget"], selected_candidate_id=last["winner_candidate_id"],
        selected_iteration=last["selected_iteration"], avg_err=last["coefficient_error"])
    refresh_identity(value)
    return value


def audit(value):
    setting = runner.SimulationSetting(**value["setting"])
    path = runner.result_path(Path('.'), model=value["model"], experiment=value["experiment"],
        setting=setting, seed_id=0)
    return summary.validate_record(value, path, setting=setting,
        model_id=0, exp_id=2, seed_id=0, random_seed=value["random_seed"],
        data_cache={}, generate_data_fn=fixtures.generator)


def test_validation_stop_is_policy_complete_with_200_validation_rows():
    value = stopped_record()
    result = audit(value)
    assert result["status"] == "policy_complete"
    assert result["validation_stopped_trajectories"] == 2
    assert result["failed_trajectories"] == 0
    assert result["caps"][-1]["policy_complete"] is True
    assert result["caps"][-1]["coverage_complete"] is False
    assert result["caps"][-1]["validation_stopped_count"] == 2
    assert result["caps"][-1]["optimizer_converged"] is False
    assert result["caps"][-1]["selected_iteration"] == 0


def test_stationarity_takes_precedence_when_patience_expires_at_same_endpoint():
    value = stopped_record()
    for trajectory in value["trajectories"]:
        fixtures.mark_checkpoint_stationary(trajectory, 4)
        trajectory.update(status="converged", termination_reason="stationarity")
        trajectory["validation_stopping"].update(stopped=False, stop_iteration=None)
        trajectory["validation_history"][-1]["validation_stopping"]["stop_requested"] = False
    for row in value["selection_history"]:
        row.update(trajectory_status="converged", trajectory_termination_reason="stationarity")
        if row["trajectory_checkpoint_iteration"] == 4:
            row.update(status="converged", termination_reason="stationarity", budget_reached=True)
    config = summary._configuration(value, runner.SimulationSetting(**value["setting"]))
    value["cap_outcomes"] = runner._cap_outcomes(value["selection_history"], value["trajectories"],
        value["configuration"], config)
    value["status"] = "complete"
    result = audit(value)
    assert result["validation_stopped_trajectories"] == 0
    assert result["caps"][-1]["coverage_complete"] is True
    assert result["caps"][-1]["policy_complete"] is True
    # The selected model is still the earlier initializer, not the stationary
    # terminal model; stationarity never transfers to the selected state.
    assert result["caps"][-1]["optimizer_converged"] is False
    assert result["caps"][-1]["selected_converged"] is False


def test_verified_stationarity_cannot_be_relabelled_as_validation_stop():
    value = stopped_record()
    for trajectory in value["trajectories"]:
        fixtures.mark_checkpoint_stationary(trajectory, 4)
    with pytest.raises(ValueError, match="stopping decision"):
        audit(value)


@pytest.mark.parametrize("mutation", ["decision", "snapshot", "coverage", "policy", "endpoint", "continue"])
def test_stopping_audit_rejects_inconsistent_saved_evidence(mutation):
    value = stopped_record()
    trajectory = value["trajectories"][0]
    if mutation == "decision":
        trajectory["validation_history"][-1]["validation_stopping"]["stop_requested"] = False
    elif mutation == "snapshot":
        trajectory["validation_stopping"]["stop_iteration"] = 2
    elif mutation == "coverage":
        value["cap_outcomes"][-1]["coverage_complete"] = True
    elif mutation == "policy":
        value["cap_outcomes"][-1]["policy_complete"] = False
    elif mutation == "endpoint":
        trajectory["checkpoints"][-1]["termination_reason"] = "checkpoint"
    else:
        trajectory["validation_history"].append(deepcopy(trajectory["validation_history"][-1]))
    with pytest.raises(ValueError):
        audit(value)


def test_legacy_records_keep_100_rows_and_disabled_stopping():
    value = fixtures.fixture()
    for name in ("validation_interval", "validation_patience", "validation_min_iterations",
                 "validation_min_relative_improvement"):
        del value["configuration"]["runner"][name]
    for cap in value["cap_outcomes"]:
        del cap["policy_complete"], cap["validation_stopped_count"]
    refresh_identity(value)
    reconstructed = summary._configuration(value, runner.SimulationSetting(**value["setting"]))
    assert reconstructed.n_validation == 100
    assert reconstructed.validation_patience is None
    assert reconstructed.validation_interval == reconstructed.checkpoint_interval == 2
    assert audit(value)["status"] == "complete"


def test_merging_stopped_tuning_shards_preserves_policy_completion():
    import merge_sparse_smart_budget_shards as merger
    from sparse_smart_selection import PAIRWISE_RULE
    full = stopped_record()
    setting = runner.SimulationSetting(**full["setting"])
    config = summary._configuration(full, setting)
    data = summary._data(setting, full["random_seed"], config, {}, fixtures.generator)
    records = []
    for j, penalty in enumerate(config.penalties_u):
        local_config = replace(config, penalties_u=(penalty,))
        value = deepcopy(full)
        value["configuration"] = fixtures._json_value(runner.resolved_configuration(setting, local_config))
        trajectory = deepcopy(full["trajectories"][j])
        trajectory["grid_candidate_id"] = 0
        prediction = runner._factor_prediction(trajectory["factor_states"]["0"], data["X"])
        value.update(trajectories=[trajectory], selection_rule=PAIRWISE_RULE, selection_score=0.,
            validation_reference_prediction=prediction.tolist(), selection_reference=dict(
                kind="first_evaluated_initializer", grid_candidate_id=0, iteration=0))
        for row in trajectory["validation_history"]:
            row.update(selection_rule=PAIRWISE_RULE, selection_score=0., selection_comparison=dict(
                incumbent_iteration=None if row["iteration"] == 0 else 0,
                loss_difference=None if row["iteration"] == 0 else 0.))
        for cp in trajectory["checkpoints"]:
            cp.update(selection_rule=PAIRWISE_RULE, selection_score=0., terminal_selection_score=0.)
        value["selection_history"] = []
        for row in full["selection_history"]:
            if row["grid_candidate_id"] != j:
                continue
            row = deepcopy(row)
            index = len(value["selection_history"])
            row.update(candidate_id=index, grid_candidate_id=0, selection_rule=PAIRWISE_RULE,
                selection_score=0., selection_comparison=dict(incumbent_candidate_id=None if index == 0 else 0,
                    loss_difference=None if index == 0 else 0.),
                budget_selection_comparison=dict(incumbent_candidate_id=None, loss_difference=None))
            value["selection_history"].append(row)
        value["cap_outcomes"] = runner._cap_outcomes(value["selection_history"], value["trajectories"],
            value["configuration"], local_config, dict(X_validation=data["X"], Y_validation=data["Y"]))
        value["selected_candidate_id"] = 0
        refresh_identity(value)
        records.append(value)
    merged, result = merger.merge_records_with_audit(records, config=config,
        generate_data_fn=fixtures.generator)
    assert merged["status"] == result["status"] == "policy_complete"
    assert merged["failure_reason"] is None
    assert merged["cap_outcomes"][-1]["coverage_complete"] is False
    assert merged["cap_outcomes"][-1]["policy_complete"] is True
    assert result["validation_stopped_trajectories"] == 2
