"""Checkpoint-budget provenance and eligibility without fitting estimators."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import run_sparse_smart_tuned as tuned
import run_sparse_smart_external as external
import run_sparse_smart_external_grid as batch
import summarize_sparse_smart_tuned as tuned_summary
import summarize_sparse_smart_external as external_summary
import summarize_sparse_smart_external_grid as grid_summary
import summarize_sparse_smart_iteration_pilot as iteration_summary
from run_sparse_smart import _digest_json, _json_value
import test_run_sparse_smart_external as runner_fixture
import test_summarize_sparse_smart_tuned as tuned_fixture
import test_summarize_sparse_smart_external as external_fixture
import test_summarize_sparse_smart_external_grid as grid_fixture


def rehash(value):
    value['configuration_fingerprint'] = _digest_json({key: value[key] for key in (
        'schema_version', 'method', 'model', 'experiment', 'rd_seed_id', 'random_seed',
        'setting', 'configuration', 'generator_arguments')})


@pytest.mark.parametrize('budgets', [(), (0, 2), (2, 2), (2, 1), (1,), (True, 2), (1., 2), '12'])
def test_invalid_schedule_rejected_before_execution(budgets):
    with pytest.raises(ValueError, match='iteration_budgets'):
        external.RunnerConfig(iterations=2, iteration_budgets=budgets).validate()


def test_candidate_count_expands_budget_major_grid_and_default_remains_single():
    setting = external.experiment_settings(0, 3)[1]
    config = external.RunnerConfig(iterations=2000, iteration_budgets=(500, 2000))
    resolved = external.resolved_configuration(setting, config)
    assert resolved['candidate_count'] == 18
    grid = tuned_summary._candidate_grid(config, resolved['support_limits'])
    assert [budget for budget, _ in grid] == [500] * 9 + [2000] * 9
    assert [params for _, params in grid[:9]] == [params for _, params in grid[9:]]
    assert external.resolved_configuration(setting, replace(config, iteration_budgets=None))['candidate_count'] == 9


@pytest.mark.parametrize('runner', [tuned, external])
def test_cell_cli_reports_schedule_and_count_without_writing(tmp_path, capsys, runner):
    runner.main(['0', '3', '0', '--setting-index', '1', '--iterations', '2000',
                 '--iteration-budgets', '500', '2000', '--output-root', str(tmp_path/'absent'), '--dry-run'])
    result = json.loads(capsys.readouterr().out)
    assert result['configuration']['runner']['iteration_budgets'] == [500, 2000]
    assert result['configuration']['candidate_count'] == 18
    assert not (tmp_path/'absent').exists()


def test_batch_cli_persists_schedule_before_any_fits(tmp_path, capsys):
    assert batch.main(['--models', '0', '--profile', 'difficult', '--iterations', '2000',
        '--iteration-budgets', '500', '2000', '--output-root', str(tmp_path/'absent'), '--dry-run']) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['configuration']['iteration_budgets'] == [500, 2000]
    assert result['expected_applicable'] == 15
    assert not (tmp_path/'absent').exists()
    with pytest.raises(SystemExit):
        batch.main(['--iterations', '500', '--iteration-budgets', '500', '2000',
                    '--output-root', str(tmp_path/'absent'), '--dry-run'])


def test_runner_saves_selected_checkpoint_metadata_and_refuses_different_resume(tmp_path):
    calls = {}
    api = runner_fixture.api(calls)
    factory = api.SparseSMARTTuner
    def make(**kwargs):
        tuner = factory(**kwargs)
        fit = tuner.fit
        def run(*args, **fit_kwargs):
            fit(*args, **fit_kwargs)
            tuner.selected_budget_ = 1
            tuner.selected_candidate_id_ = 1
            tuner.diagnostics_ = dict(iteration_budgets=[1, 2], selected_budget=1,
                                      selected_checkpoint_retained_after_failure=True)
            for row in tuner.selection_history_:
                row['iteration_budget'] = 1
            return tuner
        tuner.fit = run
        return tuner
    api.SparseSMARTTuner = make
    config = external.RunnerConfig(iterations=2, iteration_budgets=(1, 2))
    _, value = runner_fixture.run(tmp_path, calls=calls, config=config, fit_api=api)
    assert calls['config']['iteration_budgets'] == (1, 2)
    assert value['selected_budget'] == 1 and value['selected_candidate_id'] == 1
    assert value['tuning_diagnostics']['selected_checkpoint_retained_after_failure']
    assert all(row['iteration_budget'] == 1 for row in value['selection_history'])
    with pytest.raises(ValueError, match='configuration differs'):
        runner_fixture.run(tmp_path, config=replace(config, iteration_budgets=None), fit_api=api)


def checkpoint_record(kind):
    if kind == 'tuned':
        config = tuned.RunnerConfig(iterations=2, penalties_u=(.0025, .04), penalties_v=(.01,))
        value = tuned_fixture.record(config=config, selected=1)
        setting = tuned.experiment_settings(0, 0)[0]
    elif kind == 'external':
        config = external.RunnerConfig(iterations=2, penalties_u=(.0025, .04), penalties_v=(.01,))
        value, _ = external_fixture.record(config=config, selected=1)
        setting = external.experiment_settings(0, 3)[1]
    else:
        config = external.RunnerConfig(iterations=2, penalties_u=(.01, .04), penalties_v=(.01,))
        value = grid_fixture.record(config=config, selected=1)
        setting = external.experiment_settings(0, 0)[0]
    config = replace(config, iteration_budgets=(1, 2))
    resolve = tuned.resolved_configuration if kind == 'tuned' else external.resolved_configuration
    value['configuration'] = _json_value(resolve(setting, config))
    original = deepcopy(value['selection_history'])
    candidates = []
    for budget in config.iteration_budgets:
        for local_id, item in enumerate(original):
            row = deepcopy(item)
            row.update(candidate_id=len(candidates), grid_candidate_id=local_id, iteration_budget=budget,
                       n_iter=budget, validation_history=deepcopy(item['validation_history'][:budget+1]))
            if budget == 2 and local_id == 0:
                # Its excellent partial score is ineligible; the earlier successful
                # checkpoint at the same parameter values remains selectable.
                row.update(success=False, status='numerical_stagnation', validation_mse=None,
                           partial_validation_mse=.00001, has_partial_coefficient=True,
                           termination_reason='numerical_stagnation')
            candidates.append(row)
    value.update(selection_history=candidates, fit_errors=[row for row in candidates if not row['success']],
                 selected_budget=1, selected_candidate_id=0, n_iter=1,
                 validation_history=deepcopy(candidates[0]['validation_history']), history=value['history'][:2])
    rehash(value)
    return value, setting, config


def validate(kind, value, setting, config):
    if kind == 'tuned':
        path = tuned.result_path(Path('.'), model='model1', experiment='exp1', setting=setting, seed_id=0)
        return tuned_summary.validate_record(value, path, setting=setting, random_seed=tuned_fixture.SEEDS[0],
                                            model='model1', experiment='exp1', seed_id=0)
    if kind == 'external':
        return external_summary._validate_selection(value, config, value['configuration'])
    return grid_summary._validate_selection(value, setting, config, value['configuration'])


@pytest.mark.parametrize('kind', ['tuned', 'external', 'grid'])
def test_earlier_success_survives_later_failure_and_max_iteration_limit_is_its_own(kind):
    value, setting, config = checkpoint_record(kind)
    validate(kind, value, setting, config)
    assert value['selected_budget'] < config.iterations


@pytest.mark.parametrize('kind', ['tuned', 'external', 'grid'])
@pytest.mark.parametrize('mutation, message', [
    (lambda v: v.update(selected_budget=2), 'Selected budget'),
    (lambda v: v.update(selected_candidate_id=2), 'Selected candidate ID'),
    (lambda v: v.pop('selected_budget'), 'Missing selected budget'),
    (lambda v: v['selection_history'][0].update(iteration_budget=2), 'budget order'),
    (lambda v: v['selection_history'][0].pop('iteration_budget'), 'candidate iteration budget'),
    (lambda v: v['selection_history'][2].update(validation_mse=.00001), 'Failed candidate'),
    (lambda v: v['selection_history'][0].update(n_iter=2), 'candidate iterations|budget'),
])
def test_budget_and_winner_forgery_rejected(kind, mutation, message):
    value, setting, config = checkpoint_record(kind)
    mutation(value)
    with pytest.raises(ValueError, match=message):
        validate(kind, value, setting, config)


def test_full_grid_summary_counts_selected_checkpoint_edges(tmp_path):
    value, _, _ = checkpoint_record('grid')
    grid_fixture.write(tmp_path, value)
    rows, _, meta = grid_fixture.summarize(tmp_path, model_ids=(0,), seed_ids=(0,))
    row = rows[0]
    assert row['candidate_total'] == 4 and row['candidate_failed'] == 1
    assert row['selected_at_budget'] == row['selected_near_budget'] == 1
    assert json.loads(row['selected_budget_counts']) == {'1': 1}
    assert meta['runner_config']['iteration_budgets'] == [1, 2]


@pytest.mark.parametrize('kind', ['tuned', 'external', 'grid'])
def test_absent_historical_budget_is_normalized_without_mutating_saved_identity(kind):
    if kind == 'tuned':
        value = tuned_fixture.record()
        decode = tuned_summary._runner_config
        setting = tuned.experiment_settings(0, 0)[0]
    elif kind == 'external':
        value, _ = external_fixture.record()
        decode = external_summary._config
        setting = external.experiment_settings(0, 3)[1]
    else:
        value = grid_fixture.record()
        decode = external_summary._config
        setting = external.experiment_settings(0, 0)[0]
    del value['configuration']['runner']['iteration_budgets']
    rehash(value)
    before = deepcopy(value)
    config = decode(value['configuration']['runner'])
    assert config.iteration_budgets is None
    validate(kind, value, setting, config)
    assert value == before
    if kind != 'tuned':
        assert external_summary._expected_configuration(setting, config, value['configuration']) == value['configuration']


def test_narrow_iteration_comparison_refuses_checkpoint_tuning():
    value, _, _ = checkpoint_record('grid')
    with pytest.raises(ValueError, match='single-budget'):
        iteration_summary._validate_iteration_semantics(value, 2)
