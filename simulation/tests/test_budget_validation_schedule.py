"""Dense validation evidence fixtures: construct saved records without fitting."""
from copy import deepcopy
from dataclasses import replace
import importlib.util
from pathlib import Path

import numpy as np
import pytest


_fixture_spec = importlib.util.spec_from_file_location(
    "_budget_schedule_fixture", Path(__file__).with_name("test_summarize_sparse_smart_budget_study.py"))
fixtures = importlib.util.module_from_spec(_fixture_spec)
_fixture_spec.loader.exec_module(fixtures)
runner, summary = fixtures.runner, fixtures.summary

from sparse_smart_selection import PAIRWISE_RULE, pairwise_loss_difference, prediction_selection_score


def _refresh_identity(record):
    identity = {key: record[key] for key in (
        "schema_version", "method", "model", "experiment", "rd_seed_id",
        "random_seed", "setting", "configuration", "generator_arguments")}
    record["configuration_fingerprint"] = fixtures._digest_json(identity)


def early_validation_record():
    """Iteration 1 improves, 2 supersedes it, 3 loses, and 4 is the first full checkpoint."""
    record = fixtures.fixture()
    setting = runner.SimulationSetting(**record["setting"])
    config = runner.RunnerConfig(iteration_budgets=(4, 8), checkpoint_interval=4, validation_interval=4, validation_patience=None, n_validation=100,
        validation_iterations=(1, 2, 3), init_penalties=(.03,), penalties_u=(.0025, .01), penalties_v=(.0025,))
    record["configuration"] = fixtures._json_value(runner.resolved_configuration(setting, config))
    data = fixtures.generated(0)
    reference = np.zeros_like(data["Y"])
    record.update(selection_rule=PAIRWISE_RULE, validation_reference_prediction=reference.tolist())
    predictions = {}
    for j, trajectory in enumerate(record["trajectories"]):
        states, scores, errors, rows = {}, {}, {}, []
        selected = None
        for t, scale in ((0, 1.), (1, .9), (2, .8), (3, 1.1), (4, .84), (8, .68)):
            factor = dict(iteration=t, left=np.eye(setting.p, setting.target_rank).tolist(),
                right=np.eye(setting.q, setting.target_rank).tolist(),
                singular_values=(np.arange(5, 0, -1) * (scale+j*.01)).tolist())
            prediction = runner._factor_prediction(factor, data["X"])
            predictions[j, t] = prediction
            score = float(np.mean((data["Y"]-prediction)**2))
            coefficient = (np.asarray(factor["left"])*factor["singular_values"]) @ np.asarray(factor["right"]).T
            scores[t] = score
            errors[t] = float(np.linalg.norm(coefficient-data["truth"])/np.sqrt(setting.p*setting.q))
            difference = None if selected is None else pairwise_loss_difference(
                prediction, data["Y"], predictions[j, selected])
            rows.append(dict(iteration=t, loss=score, selection_rule=PAIRWISE_RULE,
                selection_score=prediction_selection_score(prediction, data["Y"], reference),
                selection_comparison=dict(incumbent_iteration=selected, loss_difference=difference)))
            improving = selected is None or difference < 0
            if improving:
                selected = t
            if improving or t in (0, 4, 8):
                states[str(t)] = factor
        assert set(states) == {"0", "1", "2", "4", "8"}
        trajectory.update(checkpoint_iterations=[0, 4, 8], factor_states=states,
            validation_history=rows, history=[fixtures.diagnostic(t) for t in (0, 1, 2, 4, 8)])
        trajectory["checkpoints"] = []
        previous = 0
        for t, winner in ((0, 0), (4, 2), (8, 8)):
            row = next(row for row in rows if row["iteration"] == t)
            chosen = next(row for row in rows if row["iteration"] == winner)
            trajectory["checkpoints"].append(dict(checkpoint_iteration=t, status="completed", success=True,
                termination_reason="checkpoint", interval_start_iteration=previous,
                interval_accepted_steps=t-previous, interval_objective_change_sum=-.1*(t-previous),
                interval_max_relative_step_norm=.01 if t else 0., selected_iteration=winner,
                terminal_factor_key=str(t), selected_factor_key=str(winner),
                terminal_validation_mse=scores[t], selected_validation_mse=scores[winner],
                terminal_coefficient_error=errors[t], selected_coefficient_error=errors[winner],
                terminal_record=fixtures.diagnostic(t), selected_record=fixtures.diagnostic(winner),
                optimization_converged=False, selected_converged=False, endpoint_selected=t == winner,
                selection_rule=PAIRWISE_RULE, selection_score=chosen["selection_score"],
                terminal_selection_score=row["selection_score"]))
            previous = t
    record["selection_history"] = []
    overall_winner = None
    for budget in config.iteration_budgets:
        budget_winner = None
        for j, trajectory in enumerate(record["trajectories"]):
            point = next(cp for cp in trajectory["checkpoints"] if cp["checkpoint_iteration"] == budget)
            row = dict(candidate_id=len(record["selection_history"]), grid_candidate_id=j,
                iteration_budget=budget, params=trajectory["params"], success=True, budget_reached=True,
                trajectory_checkpoint_iteration=budget, n_iter=budget,
                selected_iteration=point["selected_iteration"], validation_mse=point["selected_validation_mse"],
                status="completed", termination_reason="checkpoint", trajectory_status="completed",
                trajectory_termination_reason="max_iterations", selection_rule=PAIRWISE_RULE,
                selection_score=point["selection_score"])
            for key, incumbent in (("selection_comparison", overall_winner),
                                   ("budget_selection_comparison", budget_winner)):
                difference = None if incumbent is None else pairwise_loss_difference(
                    predictions[j, row["selected_iteration"]], data["Y"],
                    predictions[incumbent["grid_candidate_id"], incumbent["selected_iteration"]])
                row[key] = dict(incumbent_candidate_id=incumbent["candidate_id"] if incumbent else None,
                                loss_difference=difference)
            if overall_winner is None or row["selection_comparison"]["loss_difference"] < 0:
                overall_winner = row
            if budget_winner is None or row["budget_selection_comparison"]["loss_difference"] < 0:
                budget_winner = row
            record["selection_history"].append(row)
    fixtures.refresh_caps(record)
    for index, cap in enumerate(record["cap_outcomes"]):
        winner = record["selection_history"][cap["winner_candidate_id"]]
        cap.update(selection_rule=PAIRWISE_RULE, selection_score=winner["selection_score"],
            validation_comparisons={str(base["iteration_budget"]): pairwise_loss_difference(
                predictions[cap["winner_grid_candidate_id"], cap["selected_iteration"]], data["Y"],
                predictions[base["winner_grid_candidate_id"], base["selected_iteration"]])
                for base in record["cap_outcomes"][:index]})
    record["selection_score"] = record["cap_outcomes"][-1]["selection_score"]
    _refresh_identity(record)
    return record


def _audit(tmp_path, record):
    path = fixtures.write(tmp_path, record)
    return summary.validate_record(record, path, setting=runner.SimulationSetting(**record["setting"]),
        model_id=0, exp_id=2, seed_id=0, random_seed=record["random_seed"],
        data_cache={}, generate_data_fn=fixtures.generator)


def test_early_winners_and_scalar_only_losers_have_explicit_audit_coverage(tmp_path):
    record = early_validation_record()
    cell = _audit(tmp_path, record)
    assert cell["status"] == "complete"
    assert cell["caps"][0]["selected_iteration"] == 2
    assert cell["caps"][0]["selected_checkpoint"] == 4
    assert cell["verified_factor_states"] == 10
    audit = cell["validation_audit"]
    assert audit["evaluated_points"] == 12
    assert audit["factor_verified_points"] == 10
    assert audit["metadata_only_points"] == 2
    assert audit["verified_pairwise_comparisons"] == 8
    assert audit["metadata_only_pairwise_comparisons"] == 2
    assert audit["all_validation_points_factor_verified"] is False


@pytest.mark.parametrize("iteration", [1, 2])
def test_every_early_incumbent_requires_saved_factors_even_when_superseded(tmp_path, iteration):
    record = early_validation_record()
    trajectory = record["trajectories"][0]
    assert all(cp["selected_iteration"] != 1 for cp in trajectory["checkpoints"])
    del trajectory["factor_states"][str(iteration)]
    with pytest.raises(ValueError, match="[Ff]actor|[Ii]ncumbent|[Ww]inner|[Ss]tate"):
        _audit(tmp_path, record)


@pytest.mark.parametrize("field", ["loss", "selection_score"])
def test_backed_early_validation_score_forgery_is_rejected(tmp_path, field):
    record = early_validation_record()
    record["trajectories"][0]["validation_history"][1][field] += 1.
    with pytest.raises(ValueError, match="Factor prediction"):
        _audit(tmp_path, record)


def test_backed_early_pairwise_comparison_forgery_is_rejected(tmp_path):
    record = early_validation_record()
    record["trajectories"][0]["validation_history"][1]["selection_comparison"]["loss_difference"] -= 1.
    with pytest.raises(ValueError, match="[Pp]airwise.*difference"):
        _audit(tmp_path, record)


def test_scalar_only_loser_is_not_misrepresented_as_independently_rescored(tmp_path):
    record = early_validation_record()
    loser = record["trajectories"][0]["validation_history"][3]
    assert loser["iteration"] == 3 and "3" not in record["trajectories"][0]["factor_states"]
    # The missing coefficient cannot verify this numerical difference. A changed
    # positive value leaves the logged decision chain valid but remains unaudited.
    loser["selection_comparison"]["loss_difference"] *= 2
    cell = _audit(tmp_path, record)
    assert cell["validation_audit"]["metadata_only_pairwise_comparisons"] == 2
    assert cell["validation_audit"]["all_validation_points_factor_verified"] is False


def test_scalar_only_loser_cannot_claim_an_unretained_improvement(tmp_path):
    record = early_validation_record()
    record["trajectories"][0]["validation_history"][3]["selection_comparison"]["loss_difference"] = -1.
    with pytest.raises(ValueError):
        _audit(tmp_path, record)


def test_missing_extra_validation_evaluation_is_rejected(tmp_path):
    record = early_validation_record()
    del record["trajectories"][0]["validation_history"][3]
    with pytest.raises(ValueError, match="[Ss]cheduled|[Vv]alidation"):
        _audit(tmp_path, record)


def test_missing_selected_early_diagnostic_record_is_rejected(tmp_path):
    record = early_validation_record()
    record["trajectories"][0]["history"] = [row for row in record["trajectories"][0]["history"]
                                            if row["iteration"] != 2]
    with pytest.raises(ValueError, match="diagnostic"):
        _audit(tmp_path, record)


def test_legacy_configuration_without_new_schedule_fields_remains_auditable(tmp_path):
    record = fixtures.fixture()
    record["configuration"]["runner"].pop("validation_iterations", None)
    for key in ("validation_iterations", "validation_schedule", "validation_state_policy"):
        record["configuration"].pop(key, None)
    _refresh_identity(record)
    original_configuration = deepcopy(record["configuration"]["runner"])
    cell = _audit(tmp_path, record)
    assert cell["configuration"] == original_configuration
    assert "validation_iterations" not in cell["configuration"]
    assert cell["verified_factor_states"] == 10
    assert cell["validation_audit"]["metadata_only_points"] == 0
    assert cell["validation_audit"]["all_validation_points_factor_verified"] is True
    fixtures.manifest(tmp_path, record)
    report = summary.summarize(tmp_path, manifest_scope=True, generate_data_fn=fixtures.generator)
    assert report["recorded_cells"] == 1 and report["missing_cells"] == 0


@pytest.mark.parametrize("values", [(-1,), (True,), (1.,), (2, 1), (1, 1), "1,2"])
def test_invalid_extra_validation_schedule_is_rejected(values):
    with pytest.raises(ValueError, match="validation_iterations"):
        replace(runner.RunnerConfig(), validation_iterations=values).validate()


def test_extra_validation_schedule_may_be_empty_or_extend_beyond_short_budget():
    replace(runner.RunnerConfig(), iteration_budgets=(4, 8), validation_iterations=()).validate()
    config = replace(runner.RunnerConfig(), iteration_budgets=(4, 8), validation_iterations=(1, 3, 10))
    config.validate()
    setting = runner.SimulationSetting(**fixtures.fixture()["setting"])
    resolved = runner.resolved_configuration(setting, config)
    assert 10 not in resolved["validation_schedule"]
