"""Stable validation ranking across iterates, candidates, and budgets."""

import numpy as np
import pytest

from sparse_smart import ExactSource, Margins, PracticalCalibration, SparseSMART, SparseSMARTTuner


def scalar_data(constant):
    return (np.ones((4, 1)), np.full((4, 1), 2.),
            (np.array([[1.], [0.]]), np.array([[2.], [constant]])),
            ExactSource(np.eye(1), np.eye(1)))


def test_large_irreducible_loss_does_not_hide_better_iterates_or_change_training():
    options = dict(rank=1, source_rank=1, sparsity=(1, 1), margins=Margins(.05, 5., .01),
                   calibration=PracticalCalibration(1., 0., 1., (0, 0)), iterations=3,
                   checkpoint_iterations=(1, 3))
    fits = []
    for constant in (0., 1e12):
        X, Y, validation, source = scalar_data(constant)
        fits.append(SparseSMART(**options).fit(X, Y, source=source, validation_data=validation))
    clean, shifted = fits
    assert clean.success_ and shifted.success_
    assert shifted.selected_iteration_ == clean.selected_iteration_ == 1
    np.testing.assert_array_equal(shifted.coefficient_, [[2.]])
    np.testing.assert_array_equal(shifted.last_state_, clean.last_state_)
    assert shifted.best_selection_score_ == clean.best_selection_score_ == -.5
    assert [row['selection_score'] for row in shifted.validation_history_] == [0., -.5, -.5, -.5]
    assert len({row['loss'] for row in shifted.validation_history_}) == 1
    np.testing.assert_array_equal(shifted.validation_reference_prediction_, [[1.], [0.]])
    prefix = shifted.checkpoint_model(1)
    assert prefix.best_selection_score_ == shifted.checkpoints_[1].best_selection_score == -.5
    assert not hasattr(prefix, 'validation_reference_prediction_')
    shifted.fit(X, Y, source=source)
    assert shifted.best_selection_score_ is None and not hasattr(shifted, 'validation_reference_prediction_')


@pytest.mark.parametrize('mode', ['independent', 'continuous'])
def test_candidates_and_budgets_share_one_stable_reference_with_large_absolute_ties(mode):
    options = dict(rank=1, source_rank=1, sparsity=(1, 1), margins=Margins(.05, 5., .01),
                   init_penalties=(1., .5), penalties_u=(0.,), penalties_v=(0.,),
                   step_size_inverse=2., iterations=2, iteration_budgets=(1, 2),
                   stationarity_tol=None, checkpoint_execution=mode,
                   checkpoint_interval=1 if mode == 'continuous' else None)
    fits = []
    for constant in (0., 1e12):
        X, Y, validation, source = scalar_data(constant)
        fits.append(SparseSMARTTuner(**options).fit(X, Y, source=source, validation_data=validation))
    clean, shifted = fits
    assert clean.success_ and shifted.success_
    assert shifted.selected_candidate_id_ == clean.selected_candidate_id_ == 3
    assert shifted.selected_budget_ == 2 and shifted.best_params_['init_penalty'] == .5
    np.testing.assert_array_equal(shifted.coefficient_, clean.coefficient_)
    np.testing.assert_array_equal(shifted.coefficient_, [[1.875]])
    assert [r['selection_score'] for r in shifted.selection_history_] == [r['selection_score'] for r in clean.selection_history_]
    assert len({r['validation_mse'] for r in shifted.selection_history_}) == 1
    reference = shifted.validation_reference_prediction_
    np.testing.assert_array_equal(reference, [[1.], [0.]])
    metadata = shifted.diagnostics_['selection_reference']
    assert metadata['kind'] == 'first_evaluated_initializer' and metadata['grid_candidate_id'] == 0
    assert metadata['iteration'] == 0
    for budget, model in shifted.checkpoints_.items():
        assert not hasattr(model, 'validation_reference_prediction_')
        delta = model.predict(validation[0]) - reference
        independent_score = float(np.mean(2 * (reference - validation[1]) * delta + delta * delta))
        assert model.best_selection_score_ == independent_score
        assert shifted.diagnostics_['budget_statuses'][budget - 1]['best_selection_score'] == independent_score


@pytest.mark.parametrize('mode', ['independent', 'continuous'])
def test_exact_prediction_ties_keep_earliest_candidate_and_budget(mode):
    X, Y, validation, source = scalar_data(1e12)
    fit = SparseSMARTTuner(rank=1, source_rank=1, sparsity=(1, 1), margins=Margins(.05, 5., .01),
        init_penalties=(1., .5), penalties_u=(0.,), penalties_v=(0.,), step_size_inverse=1.,
        iterations=2, iteration_budgets=(1, 2), stationarity_tol=None,
        checkpoint_execution=mode, checkpoint_interval=1 if mode == 'continuous' else None).fit(
            X, Y, source=source, validation_data=validation)
    assert fit.success_ and fit.selected_candidate_id_ == 0 and fit.selected_budget_ == 1
    assert fit.selected_iteration_ == 1 and fit.best_selection_score_ == -.5


@pytest.mark.parametrize('mode', ['standalone', 'independent', 'continuous'])
def test_pairwise_ranking_retains_late_clean_data_improvements(mode):
    X, Y = np.ones((4, 1)), np.full((4, 1), 2.)
    validation = (np.ones((1, 1)), np.full((1, 1), 2.))
    source = ExactSource(np.eye(1), np.eye(1))
    common = dict(rank=1, source_rank=1, sparsity=(1, 1), margins=Margins(.05, 5., .01),
                  iterations=40, stationarity_tol=None)
    if mode == 'standalone':
        fit = SparseSMART(**common, calibration=PracticalCalibration(1., 0., 2., (0, 0))).fit(
            X, Y, source=source, validation_data=validation)
    else:
        fit = SparseSMARTTuner(**common, init_penalties=(1., .5), penalties_u=(0.,), penalties_v=(0.,),
            step_size_inverse=2., iteration_budgets=(30, 40), checkpoint_execution=mode,
            checkpoint_interval=1 if mode == 'continuous' else None).fit(
                X, Y, source=source, validation_data=validation)
        assert fit.selected_candidate_id_ == 3 and fit.selected_budget_ == 40
    assert fit.success_ and fit.selected_iteration_ == 40
    assert fit.best_validation_loss_ < 1e-24
    assert fit.best_selection_score_ == -1.


@pytest.mark.parametrize('mode', ['standalone', 'independent', 'continuous'])
def test_pairwise_ranking_keeps_late_improvements_when_both_reporting_scores_tie(mode):
    X, Y, validation, source = scalar_data(1e12)
    common = dict(rank=1, source_rank=1, sparsity=(1, 1), margins=Margins(.05, 5., .01),
                  iterations=40, stationarity_tol=None)
    if mode == 'standalone':
        fit = SparseSMART(**common, calibration=PracticalCalibration(1., 0., 2., (0, 0)),
                          checkpoint_iterations=(30, 40)).fit(X, Y, source=source, validation_data=validation)
    else:
        fit = SparseSMARTTuner(**common, init_penalties=(1., .5), penalties_u=(0.,), penalties_v=(0.,),
            step_size_inverse=2., iteration_budgets=(30, 40), checkpoint_execution=mode,
            checkpoint_interval=1 if mode == 'continuous' else None).fit(
                X, Y, source=source, validation_data=validation)
        assert fit.selected_candidate_id_ == 3 and fit.selected_budget_ == 40
        last = fit.selection_history_[-1]
        assert last['selection_comparison']['loss_difference'] < 0
        assert last['budget_selection_comparison']['loss_difference'] < 0
    assert fit.success_ and fit.selected_iteration_ == 40
    assert 0 < 2.-fit.coefficient_[0, 0] < 1e-12
    assert len({row['loss'] for row in fit.validation_history_}) == 1
    assert fit.validation_history_[27]['selection_score'] == fit.validation_history_[-1]['selection_score']
    final = fit.validation_history_[-1]
    assert final['selection_comparison']['incumbent_iteration'] == 39
    assert final['selection_comparison']['loss_difference'] < 0
    assert final['selection_rule'] == fit.selection_rule_ == 'pairwise-validation-loss-v1'
    for checkpoint in fit.checkpoints_.values():
        assert not hasattr(checkpoint, 'validation_reference_prediction_')


@pytest.mark.parametrize('mode', ['independent', 'continuous'])
def test_pairwise_candidate_ranking_is_not_limited_by_a_distant_common_reference(mode):
    X, Y = np.ones((4, 1)), np.full((4, 1), 1e8)
    fit = SparseSMARTTuner(rank=1, source_rank=1, sparsity=(1, 1), margins=Margins(.01, 2e8, .01),
        iterations=0, init_penalties=(9e7, .5, 1e-9), penalties_u=(0.,), penalties_v=(0.,),
        checkpoint_execution=mode, checkpoint_interval=1 if mode == 'continuous' else None).fit(
            X, Y, source=ExactSource(np.eye(1), np.eye(1)),
            validation_data=(np.array([[1.], [0.]]), np.array([[1e8], [1e12]])))
    assert fit.success_ and fit.selected_candidate_id_ == 2
    np.testing.assert_array_equal(fit.coefficient_, [[1e8]])
    before, after = fit.selection_history_[1:]
    assert before['selection_score'] == after['selection_score']
    assert before['validation_mse'] == after['validation_mse']
    assert after['selection_comparison'] == {'incumbent_candidate_id': 1, 'loss_difference': -.125}


def test_pairwise_comparison_retains_exact_loss_ties_with_different_predictions():
    from sparse_smart.validation import validation_loss_difference
    response = np.array([[2.], [1e12]])
    assert validation_loss_difference(np.array([[3.], [0.]]), np.array([[1.], [0.]]), response) == 0.


@pytest.mark.parametrize('mode', ['standalone', 'independent', 'continuous'])
def test_scaled_multidimensional_reference_uses_public_prediction_association(mode):
    from sparse_smart.validation import ValidationReference

    rng = np.random.default_rng(347)
    left = np.linalg.qr(rng.normal(size=(20, 8)))[0]
    right = np.linalg.qr(rng.normal(size=(15, 8)))[0]
    X = 300 * rng.normal(size=(40, 20))
    XV = 300 * rng.normal(size=(30, 20))
    coefficient = (left[:, :2] * [2., 1.]) @ right[:, :2].T
    Y = X @ coefficient + rng.normal(scale=.05, size=(40, 15))
    YV = np.zeros((30, 15))
    source = ExactSource(left, right)
    common = dict(rank=2, source_rank=8, sparsity=(2, 2), margins=Margins(.05, 6., .01),
                  iterations=0, stationarity_tol=None)
    if mode == 'standalone':
        fit = SparseSMART(**common, calibration=PracticalCalibration(.03, 0., 20., (12, 12))).fit(
            X, Y, source=source, validation_data=(XV, YV))
    else:
        fit = SparseSMARTTuner(**common, init_penalties=(.03,), penalties_u=(0.,), penalties_v=(0.,),
            step_size_inverse=20., checkpoint_execution=mode,
            checkpoint_interval=1 if mode == 'continuous' else None).fit(
                X, Y, source=source, validation_data=(XV, YV))
    assert fit.success_
    np.testing.assert_array_equal(fit.predict(XV), fit.validation_reference_prediction_)
    reference = ValidationReference()
    reference.score(fit.validation_reference_prediction_, YV)
    assert reference.score(fit.predict(XV), YV) == fit.best_selection_score_ == 0.
