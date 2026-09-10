"""Selection and artifact verification when a common residual hides MSE changes."""
from copy import deepcopy
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sparse_smart_selection import (prediction_selection_score, selection_keys,
                                    validation_winner, PAIRWISE_RULE)
from summarize_sparse_smart_tuned import _validate_validation_history
import run_sparse_smart_budget_study as runner
import summarize_sparse_smart_budget_study as summary
import run_sparse_smart_tuned as tuned_runner
import run_sparse_smart_external as external_runner
import summarize_sparse_smart_tuned as tuned_summary
import summarize_sparse_smart_external_grid as external_summary
import summarize_sparse_smart_iteration_pilot as iteration_summary


def test_relative_scores_rank_hidden_improvement_and_absolute_mse_breaks_rounded_ties():
    reference = np.array([[1.], [0.]])
    response = np.array([[2.], [1e12]])
    scores = [prediction_selection_score(np.array([[c], [0.]]), response, reference) for c in (1., 2.)]
    assert scores == [0., -.5]
    rows = [dict(validation_mse=5e23, selection_score=s) for s in scores]
    assert validation_winner(rows) is rows[1]
    rows = [dict(validation_mse=mse, selection_score=-1.) for mse in (1e-17, 1e-24, 1e-24)]
    assert validation_winner(rows) is rows[1]
    legacy = [dict(validation_mse=mse) for mse in (2., 1., 1.)]
    assert validation_winner(legacy) is legacy[1]
    with pytest.raises(ValueError, match="Mixed"):
        selection_keys([legacy[0], rows[0]])


def test_history_validator_accepts_hidden_improvement_and_rejects_wrong_selected_score():
    candidate = dict(n_iter=1, selected_iteration=1, validation_mse=5e23, selection_score=-.5,
        validation_history=[dict(iteration=0, loss=5e23, selection_score=0.),
                            dict(iteration=1, loss=5e23, selection_score=-.5)])
    _validate_validation_history(candidate)
    candidate['selected_iteration'] = 0
    with pytest.raises(ValueError, match="validation minimum"):
        _validate_validation_history(candidate)
    candidate['selected_iteration'] = 1
    candidate['selection_score'] = 0.
    with pytest.raises(ValueError, match="selection score mismatch"):
        _validate_validation_history(candidate)


def scalar_generator(*, n, p, q, **kwargs):
    return dict(X=np.ones((n, p)), Y=np.full((n, q), 2.),
                C0=np.ones((p, q)), C_star=np.full((p, q), 2.))


def test_real_budget_runner_and_summary_select_and_verify_hidden_validation_gain(tmp_path, monkeypatch):
    # Replace only the simulated observations. The real estimator, tuner,
    # serializers, data fingerprinting and independent summary all execute.
    original = runner.external_validation_data.generate_external_validation
    def external_data(**kwargs):
        data = original(**kwargs)
        data['X_validation'][:] = 1.
        data['Y_validation'][:] = 2.
        data['X_validation'][0] = 0.
        data['Y_validation'][0] = 1e12
        return data
    monkeypatch.setattr(runner.external_validation_data, 'generate_external_validation', external_data)
    setting = runner.SimulationSetting(4, 1, 1, 0., 1, 1, 'scalar')
    config = runner.RunnerConfig(iteration_budgets=(1, 3), checkpoint_interval=1,
        init_penalties=(1., .5), penalties_u=(0.,), penalties_v=(0.,), inverse_step=2.)
    path = runner.result_path(tmp_path, model='model1', experiment='exp1', setting=setting, seed_id=0)
    _, record = runner.run_setting(setting=setting, model='model1', experiment='exp1',
        seed_id=0, random_seed=123, destination=path, config=config, generate_data_fn=scalar_generator)
    assert record['success'] and record['selected_iteration'] == 3
    assert len({row['validation_mse'] for row in record['selection_history']}) == 1
    assert record['selected_candidate_id'] == 3
    def validate(value):
        return summary.validate_record(value, path, setting=setting, model_id=0, exp_id=0,
            seed_id=0, random_seed=123, data_cache={}, generate_data_fn=scalar_generator)
    audited = validate(record)
    assert audited['verified_factor_states'] == 8
    comparison = summary.compare_caps(*audited['caps'], absolute_threshold=0., relative_threshold=0.)
    assert comparison['validation_gain'] > 0
    assert comparison['material_gain']
    broken = deepcopy(record)
    broken['trajectories'][0]['validation_history'][1]['selection_score'] -= .1
    with pytest.raises(ValueError, match='Factor prediction/selection history'):
        validate(broken)
    broken = deepcopy(record)
    broken['validation_reference_prediction'][1][0] += .1
    with pytest.raises(ValueError, match='Factor prediction/selection history'):
        validate(broken)

    external_config = external_runner.RunnerConfig(iterations=3, iteration_budgets=(1, 3),
        init_penalties=(1., .5), penalties_u=(0.,), penalties_v=(0.,), inverse_step=2.)
    external_path = external_runner.result_path(tmp_path, model='model1', experiment='exp1', setting=setting, seed_id=0)
    _, external_record = external_runner.run_setting(setting=setting, model='model1', experiment='exp1',
        seed_id=0, random_seed=123, destination=external_path, config=external_config, generate_data_fn=scalar_generator)
    assert external_record['success'] and external_record['selected_candidate_id'] == 3
    external_summary.validate_record(external_record, external_path, setting=setting, model_id=0,
        exp_id=0, seed_id=0, random_seed=123, data_cache={}, generate_data_fn=scalar_generator)


def test_real_internal_holdout_artifact_has_consistent_selection_diagnostics(tmp_path):
    setting = runner.SimulationSetting(4, 1, 1, 0., 1, 1, 'scalar')
    config = tuned_runner.RunnerConfig(iterations=3, iteration_budgets=(1, 3),
        init_penalties=(1., .5), penalties_u=(0.,), penalties_v=(0.,), inverse_step=2.)
    path = tuned_runner.result_path(tmp_path, model='model1', experiment='exp1', setting=setting, seed_id=0)
    _, record = tuned_runner.run_setting(setting=setting, model='model1', experiment='exp1',
        seed_id=0, random_seed=123, destination=path, config=config, generate_data_fn=scalar_generator)
    assert record['success'] and record['selected_candidate_id'] == 3
    tuned_summary.validate_record(record, path, setting=setting, model='model1', experiment='exp1',
                                  seed_id=0, random_seed=123)


def test_pairwise_budget_and_iteration_artifacts_preserve_improvements_when_both_scores_tie(tmp_path, monkeypatch):
    original = runner.external_validation_data.generate_external_validation
    def external_data(**kwargs):
        data = original(**kwargs)
        data['X_validation'][:] = 1.
        data['Y_validation'][:] = 2.
        data['X_validation'][0] = 0.
        data['Y_validation'][0] = 1e12
        return data
    monkeypatch.setattr(runner.external_validation_data, 'generate_external_validation', external_data)
    setting = runner.SimulationSetting(4, 1, 1, 0., 1, 1, 'scalar')
    config = runner.RunnerConfig(iteration_budgets=(27, 40), checkpoint_interval=1,
        init_penalties=(1.,), penalties_u=(0.,), penalties_v=(0.,), inverse_step=2.,
        stationarity_tol=1e-16)
    path = runner.result_path(tmp_path, model='model1', experiment='exp1', setting=setting, seed_id=0)
    _, record = runner.run_setting(setting=setting, model='model1', experiment='exp1',
        seed_id=0, random_seed=123, destination=path, config=config, generate_data_fn=scalar_generator)
    assert record['success'] and record['selected_iteration'] == 40
    assert record['selection_rule'] == PAIRWISE_RULE
    first, last = record['cap_outcomes']
    assert first['validation_mse'] == last['validation_mse']
    assert first['selection_score'] == last['selection_score']
    def validate(value):
        return summary.validate_record(value, path, setting=setting, model_id=0, exp_id=0,
            seed_id=0, random_seed=123, data_cache={}, generate_data_fn=scalar_generator)
    audited = validate(record)
    comparison = summary.compare_caps(*audited['caps'], absolute_threshold=0., relative_threshold=0.)
    assert comparison['validation_gain'] > 0 and comparison['material_gain']
    broken = deepcopy(record)
    broken['trajectories'][0]['validation_history'][40]['selection_comparison']['loss_difference'] = 0.
    with pytest.raises(ValueError, match='Pairwise validation difference'):
        validate(broken)
    for key in ('selection_comparison', 'budget_selection_comparison'):
        broken = deepcopy(record)
        broken['selection_history'][1][key]['incumbent_candidate_id'] = 99
        with pytest.raises(ValueError, match='comparison incumbent mismatch'):
            validate(broken)
    broken = deepcopy(record)
    broken['cap_outcomes'][1]['validation_comparisons']['27'] = 0.
    with pytest.raises(ValueError, match='Pairwise cap comparison sign'):
        validate(broken)

    external_records = []
    for budget in (27, 40):
        external_config = external_runner.RunnerConfig(iterations=budget,
            init_penalties=(1.,), penalties_u=(0.,), penalties_v=(0.,), inverse_step=2.,
            stationarity_tol=1e-16)
        external_path = external_runner.result_path(tmp_path / str(budget), model='model1',
            experiment='exp1', setting=setting, seed_id=0)
        _, result = external_runner.run_setting(setting=setting, model='model1', experiment='exp1',
            seed_id=0, random_seed=123, destination=external_path, config=external_config,
            generate_data_fn=scalar_generator)
        external_summary.validate_record(result, external_path, setting=setting, model_id=0,
            exp_id=0, seed_id=0, random_seed=123, data_cache={}, generate_data_fn=scalar_generator)
        external_records.append(result)
    Xv, Yv = np.ones((100, 1)), np.full((100, 1), 2.)
    Xv[0], Yv[0] = 0., 1e12
    paired = iteration_summary.validate_pair(*external_records, base_budget=27, extended_budget=40,
                                            validation_data=(Xv, Yv))
    assert paired['selected_after_base_budget'] and paired['validation_change'] < 0
    assert paired['validation_change_basis'] == 'pairwise_validation_loss_difference'
