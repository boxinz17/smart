"""Continuous trajectories, sparse validation selection, and cap coverage."""
from types import SimpleNamespace

import numpy as np
import pytest

from sparse_smart import FitFailure
from sparse_smart import tuning
from .test_tuning import _tuner, data


@pytest.fixture
def trajectories(monkeypatch):
    class FakeTrajectory:
        instances, outcomes, checkpoint_calls = [], {}, []

        def __init__(self, **options):
            self.options = options
            self.instances.append(self)

        def fit(self, X, Y, *, source, validation_data):
            self.training = (X.copy(), Y.copy())
            self.validation = tuple(v.copy() for v in validation_data)
            self.validation_reference = self._fit_cache.validation_reference
            self.validation_context = self._fit_cache.validation_context.copy()
            self.source = source
            self.outcome = self.outcomes.get(self.options['calibration'].penalty[0], {})
            self.success_ = self.outcome.get('success', True)
            self.n_iter_ = self.outcome.get('n_iter', self.options['iterations'])
            self.termination_reason_ = self.outcome.get('reason',
                'max_iterations' if self.success_ else 'numerical_stagnation')
            self.status_ = ('converged' if self.termination_reason_ == 'stationarity' else
                            'completed' if self.success_ else 'numerical_stagnation')
            self.message_ = self.status_
            self.diagnostics_ = {'terminal': self.n_iter_}
            self.coefficient_ = np.zeros((X.shape[1], Y.shape[1]))
            self.checkpoint_iterations_ = tuple(t for t in self.options['checkpoint_iterations'] if t <= self.n_iter_)
            if self.termination_reason_ == 'stationarity':
                self.checkpoint_iterations_ = tuple(sorted({*self.checkpoint_iterations_, self.n_iter_}))
            if self.outcome.get('exception'):
                raise FloatingPointError('validation failed after a completed checkpoint')
            return self

        def _checkpoint_view(self, endpoint):
            assert endpoint in self.checkpoint_iterations_
            points = [t for t in self.checkpoint_iterations_ if t <= endpoint]
            values = self.outcome.get('values', {})
            scores = {t: values.get(t, 10.-t)**2/2 for t in points}
            selection_scores = {}
            for t in points:
                candidate = self.coefficient_.copy()
                candidate[0, 0] = values.get(t, 10.-t)
                selection_scores[t] = self.validation_reference.score(
                    self.validation[0] @ candidate, self.validation[1], metadata=self.validation_context)
            selected = min(points, key=lambda t: (selection_scores[t], scores[t]))
            coefficient = self.coefficient_.copy()
            coefficient[0, 0] = values.get(selected, 10.-selected)
            stationary = self.termination_reason_ == 'stationarity' and endpoint == self.n_iter_
            return SimpleNamespace(success_=True, status_='converged' if stationary else 'completed',
                message_='completed prefix', termination_reason_='stationarity' if stationary else 'max_iterations',
                coefficient_=coefficient, n_iter_=endpoint, selected_iteration_=selected,
                best_selection_score_=selection_scores[selected],
                validation_history_=[{'iteration': t, 'loss': scores[t],
                                      'selection_score': selection_scores[t]} for t in points],
                diagnostics_={'terminal': endpoint}, predict=lambda X, **kwargs: X @ coefficient)

        def _checkpoint_summary(self, endpoint):
            view = self._checkpoint_view(endpoint)
            return dict(success=view.success_, status=view.status_, message=view.message_,
                validation_mse=min(row['loss'] for row in view.validation_history_),
                selection_score=view.best_selection_score_,
                selected_iteration=view.selected_iteration_, n_iter=view.n_iter_,
                termination_reason=view.termination_reason_, validation_history=view.validation_history_,
                diagnostics=view.diagnostics_)

        def _checkpoint_prediction(self, endpoint, X):
            return self._checkpoint_view(endpoint).predict(X)

        def checkpoint_model(self, endpoint):
            self.checkpoint_calls.append((self.options['calibration'].penalty[0], endpoint))
            if self.outcome.get('materialization_error'):
                raise FloatingPointError('checkpoint reconstruction failed')
            return self._checkpoint_view(endpoint)

    monkeypatch.setattr(tuning, 'SparseSMART', FakeTrajectory)
    return FakeTrajectory


def continuous(**options):
    defaults = dict(iterations=8, iteration_budgets=(4, 8), checkpoint_interval=2,
                    checkpoint_execution='continuous')
    defaults.update(options)
    return _tuner(**defaults)


def test_each_grid_point_is_fit_once_without_data_or_state_restart(data, trajectories):
    X, Y, source = data
    Xv, Yv = X[:3]+10., Y[:3]
    # The first predictor stays at 1 so the mock scores equal its stored scores.
    Xv[:, 0] = 1.
    fitted = continuous().fit(X, Y, source=source, validation_data=(Xv, Yv))
    assert len(trajectories.instances) == 2 < fitted.n_candidates_ == 4
    for model in trajectories.instances:
        assert model.options['iterations'] == 8
        assert model.options['checkpoint_iterations'] == [0, 2, 4, 6, 8]
        assert model.options['validation_interval'] == 2
        np.testing.assert_array_equal(model.training[0], X)
        np.testing.assert_array_equal(model.validation[0], Xv)
        assert model.source is source
    assert fitted.diagnostics_['trajectory_fits'] == 2
    assert fitted.diagnostics_['checkpoint_execution'] == 'continuous_trajectory'
    assert fitted.selected_budget_ == 8 and fitted.selected_checkpoint_iteration_ == 8
    assert all(r['budget_reached'] for r in fitted.selection_history_)
    assert [r['grid_candidate_id'] for r in fitted.selection_history_] == [0, 1, 0, 1]
    assert all(r['elapsed_time_sec'] is None for r in fitted.selection_history_)


def test_failure_before_first_cap_retains_dense_prefix_but_cannot_prove_plateau(data, trajectories):
    X, Y, source = data
    trajectories.outcomes = {p: dict(success=False, n_iter=3, values={0: 10., 2: 1.}) for p in (.01, .02)}
    fitted = continuous().fit(X, Y, source=source)
    assert fitted.success_ and fitted.selected_budget_ == 4 and fitted.selected_checkpoint_iteration_ == 2
    assert fitted.best_score_ == .5 and fitted.n_iter_ == 2
    assert len(fitted.checkpoints_) == 2
    assert all(r['success'] and not r['budget_reached'] for r in fitted.selection_history_)
    assert all(r['trajectory_status'] == 'numerical_stagnation' for r in fitted.selection_history_)
    assert all(r['status'] == 'completed' and r['trajectory_checkpoint_iteration'] == 2
               for r in fitted.selection_history_)
    assert fitted.diagnostics_['failed_trajectories'] == 2
    assert fitted.diagnostics_['selected_checkpoint_retained_after_failure']
    assert all(not row['budget_fully_covered'] for row in fitted.diagnostics_['budget_statuses'])


def test_failure_before_any_positive_checkpoint_does_not_make_initializer_eligible(data, trajectories):
    X, Y, source = data
    trajectories.outcomes = {p: dict(success=False, n_iter=1, values={0: 0.}) for p in (.01, .02)}
    fitted = continuous().fit(X, Y, source=source)
    assert not fitted.success_ and not fitted.checkpoints_
    assert all(not r['success'] and r['validation_mse'] is None for r in fitted.selection_history_)
    with pytest.raises(FitFailure):
        fitted.predict(X)


def test_caught_exception_keeps_completed_prefix_and_failure_coverage(data, trajectories):
    X, Y, source = data
    trajectories.outcomes = {p: dict(n_iter=3, exception=True) for p in (.01, .02)}
    fitted = continuous().fit(X, Y, source=source)
    assert fitted.success_ and fitted.selected_checkpoint_iteration_ == 2
    assert all(t['status'] == 'FloatingPointError' and t['n_iter'] == 3
               and not t['success'] for t in fitted.trajectory_history_)
    assert all(r['success'] and not r['budget_reached'] and r['trajectory_n_iter'] == 3
               for r in fitted.selection_history_)


@pytest.mark.parametrize('stop', [0, 1])
def test_certified_early_stationarity_covers_later_caps_even_off_grid(data, trajectories, stop):
    X, Y, source = data
    trajectories.outcomes = {p: dict(n_iter=stop, reason='stationarity') for p in (.01, .02)}
    fitted = continuous().fit(X, Y, source=source)
    assert fitted.success_ and fitted.n_iter_ == stop and fitted.selected_budget_ == 4
    assert all(r['budget_reached'] and r['trajectory_checkpoint_iteration'] == stop
               and r['termination_reason'] == 'stationarity' for r in fitted.selection_history_)


def test_best_prefix_can_be_an_interior_dense_checkpoint_with_earliest_ties(data, trajectories):
    X, Y, source = data
    trajectories.outcomes = {p: dict(values={0: 5., 2: 2., 4: 3., 6: 1., 8: 1.}) for p in (.01, .02)}
    fitted = continuous().fit(X, Y, source=source)
    assert fitted.selected_budget_ == 8 and fitted.selected_iteration_ == 6
    assert fitted.selected_candidate_id_ == 2 and fitted.best_params_['penalty_u'] == .01
    assert fitted.checkpoints_[4].selected_iteration_ == 2


def test_only_final_budget_winners_materialize_models(data, trajectories):
    X, Y, source = data
    trajectories.outcomes = {.01: dict(values={0: 5., 4: 3., 8: 3.}),
                             .02: dict(values={0: 5., 4: 1., 8: .5})}
    fitted = continuous().fit(X, Y, source=source)
    assert fitted.success_ and fitted.selected_candidate_id_ == 3
    assert trajectories.checkpoint_calls == [(.02, 4), (.02, 8)]
    assert len(fitted.selection_history_) == 4


def test_checkpoint_materialization_failure_tries_next_candidate(data, trajectories):
    X, Y, source = data
    trajectories.outcomes = {.01: dict(values={0: 5., 4: .1, 8: .1}, materialization_error=True),
                             .02: dict(values={0: 5., 4: 1., 8: .5})}
    fitted = continuous().fit(X, Y, source=source)
    assert fitted.success_ and fitted.best_params_['penalty_u'] == .02
    assert all(not fitted.selection_history_[i]['success'] for i in (0, 2))
    assert trajectories.checkpoint_calls == [(.01, 4), (.02, 4), (.01, 8), (.02, 8)]


def test_refitting_clears_old_trajectory_models_and_checkpoints(data, trajectories):
    X, Y, source = data
    fitted = continuous().fit(X, Y, source=source)
    old = tuple(fitted.trajectory_models_)
    trajectories.outcomes = {p: dict(success=False, n_iter=1) for p in (.01, .02)}
    fitted.fit(X, Y, source=source)
    assert not fitted.success_ and not fitted.checkpoints_
    assert not any(model in old for model in fitted.trajectory_models_)


@pytest.mark.parametrize('options', [dict(checkpoint_execution='bad'),
    dict(checkpoint_execution='independent', checkpoint_interval=2),
    dict(checkpoint_interval=0), dict(checkpoint_interval=True), dict(checkpoint_interval=2.5)])
def test_invalid_continuous_options_are_rejected(data, options):
    X, Y, source = data
    with pytest.raises(ValueError):
        continuous(**options).fit(X, Y, source=source)


def test_real_continuous_tuner_matches_independent_optimization_prefixes():
    from sparse_smart import ExactSource, Margins, PracticalCalibration, SparseSMART
    rng = np.random.default_rng(712)
    X = rng.normal(size=(45, 3))
    Y = np.column_stack((2.*X[:, 0], .15*X[:, 1])) + rng.normal(scale=.05, size=(45, 2))
    source = ExactSource(np.eye(3)[:, :2], np.eye(2))
    options = dict(rank=1, source_rank=2, sparsity=(1, 1), margins=Margins(.05, 6., .01),
        init_penalties=(.03,), penalties_u=(.01,), penalties_v=(.01,),
        step_size_inverse=20., stationarity_tol=None, refinement_solver='anchor_projected')
    fitted = tuning.SparseSMARTTuner(**options, iterations=8, iteration_budgets=(4, 8),
        checkpoint_execution='continuous', checkpoint_interval=2).fit(
            X[:35], Y[:35], source=source, validation_data=(X[35:], Y[35:]))
    assert fitted.success_ and len(fitted.trajectory_models_) == 1
    path = fitted.trajectory_models_[0]
    assert path.checkpoint_iterations_ == (0, 2, 4, 6, 8)
    for budget in (4, 8):
        independent = SparseSMART(rank=1, source_rank=2, sparsity=(1, 1), margins=options['margins'],
            calibration=PracticalCalibration(.03, (.01, .01), 20., (1, 1)), iterations=budget,
            stationarity_tol=None, refinement_solver='anchor_projected', validation_interval=2,
            checkpoint_iterations=(0, *range(2, budget+1, 2))).fit(
                X[:35], Y[:35], source=source, validation_data=(X[35:], Y[35:]))
        assert independent.success_
        prefix = path.checkpoint_model(budget)
        np.testing.assert_array_equal(independent.last_state_, prefix.last_state_)
        np.testing.assert_array_equal(independent.coefficient_, prefix.coefficient_)
        assert independent.validation_history_ == prefix.validation_history_
        assert independent.history_ == prefix.history_
