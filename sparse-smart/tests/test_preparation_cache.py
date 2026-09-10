"""Fit-scoped preparation reuse preserves candidate and checkpoint semantics."""

import numpy as np
import pytest

from sparse_smart import ExactSource, Margins, SparseSMARTTuner
from sparse_smart import estimator


def data():
    rng = np.random.default_rng(712)
    X = rng.normal(size=(40, 3))
    Y = np.column_stack((2. * X[:, 0], .15 * X[:, 1])) + rng.normal(scale=.05, size=(40, 2))
    return X[:30], Y[:30], X[30:], Y[30:], ExactSource(np.eye(3)[:, :2], np.eye(2))


def tuner(mode):
    return SparseSMARTTuner(rank=1, source_rank=2, sparsity=(1, 1), margins=Margins(.05, 6., .01),
        init_penalties=(.01, .03), penalties_u=(.001, .01), penalties_v=(.002, .02),
        support_limits=((0, 0), (1, 1)), iterations=4, iteration_budgets=(2, 4),
        checkpoint_execution=mode, checkpoint_interval=1 if mode == 'continuous' else None,
        stationarity_tol=None)


@pytest.mark.parametrize('mode', ['independent', 'continuous'])
def test_preparation_is_shared_and_matches_uncached_candidates(monkeypatch, mode):
    X, Y, Xv, Yv, source = data()
    counts = {'prepare_source': 0, 'reduced_lasso': 0, 'select_anchor': 0}
    for name in counts:
        original = getattr(estimator, name)

        def counted(*args, _name=name, _original=original, **kwargs):
            counts[_name] += 1
            return _original(*args, **kwargs)

        monkeypatch.setattr(estimator, name, counted)
    computed = []
    original_get = estimator._FitPreparationCache.get

    def observe_cache(self, key, compute):
        if key not in self.entries:
            computed.append(key[0])
        return original_get(self, key, compute)

    monkeypatch.setattr(estimator._FitPreparationCache, 'get', observe_cache)
    cached = tuner(mode).fit(X, Y, source=source, validation_data=(Xv, Yv))
    assert cached.success_
    assert counts == {'prepare_source': 1, 'reduced_lasso': 2, 'select_anchor': 4}
    for name in ('leading_data', 'training_data'):
        assert computed.count(name) == 1
    assert 'validation_data' not in computed  # Validation uses the public factor association.
    assert cached.n_candidates_ == 32

    monkeypatch.setattr(estimator._FitPreparationCache, 'get', lambda self, key, compute: compute())
    uncached = tuner(mode).fit(X, Y, source=source, validation_data=(Xv, Yv))
    assert uncached.success_ and cached.selected_candidate_id_ == uncached.selected_candidate_id_
    for left, right in zip(cached.selection_history_, uncached.selection_history_):
        assert left['status'] == right['status']
        assert left['selected_iteration'] == right['selected_iteration']
        assert left['validation_mse'] == right['validation_mse']
        assert left['validation_history'] == right['validation_history']
    for budget in (2, 4):
        np.testing.assert_array_equal(cached.checkpoints_[budget].coefficient_,
                                      uncached.checkpoints_[budget].coefficient_)
        assert cached.checkpoints_[budget].history_ == uncached.checkpoints_[budget].history_


def test_cache_does_not_survive_refit_or_alias_public_candidates(monkeypatch):
    X, Y, Xv, Yv, source = data()
    original = estimator.prepare_source
    calls = []

    def prepare(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(estimator, 'prepare_source', prepare)
    fitted = tuner('continuous').fit(X, Y, source=source, validation_data=(Xv, Yv))
    a, b = fitted.trajectory_models_[:2]
    assert not hasattr(a, '_fit_cache')
    assert not np.shares_memory(a.source_.left, b.source_.left)
    assert not np.shares_memory(a.initialization_.P, b.initialization_.P)
    assert not np.shares_memory(a.anchors_['u'].center, b.anchors_['u'].center)
    old = fitted.coefficient_.copy()
    Y *= .7
    fitted.fit(X, Y, source=source, validation_data=(Xv, Yv))
    assert len(calls) == 2 and fitted.success_
    assert not np.array_equal(fitted.coefficient_, old)
    fresh = tuner('continuous').fit(X, Y, source=source, validation_data=(Xv, Yv))
    np.testing.assert_array_equal(fitted.coefficient_, fresh.coefficient_)


def test_estimator_support_metadata_distinguishes_numerical_and_raw_counts():
    X, Y, Xv, Yv, source = data()
    fitted = tuner('continuous').fit(X, Y, source=source, validation_data=(Xv, Yv))
    model = next(m for m in fitted.trajectory_models_ if m.refinement_solver_ == 'anchor_projected')
    state = model.state_.copy()
    state[model.chart_.z_u_slice] = 1e-12
    before = state.copy()
    supports, raw, tolerances, diagnostics = model._refinement_metadata(
        model.result_, state, model.selected_iteration_, model.best_validation_loss_)
    assert len(supports['u']) == 0 and len(raw['u']) == 1
    assert tolerances['u'] == 1e-10
    assert diagnostics['support_reporting'] == 'effective_numerical'
    np.testing.assert_array_equal(state, before)
    for fitted_model in (model, model.checkpoint_model(2)):
        assert hasattr(fitted_model, 'raw_supports_') and hasattr(fitted_model, 'support_tolerances_')
        np.testing.assert_allclose(fitted_model.predict(X), X @ fitted_model.coefficient_, atol=1e-12)
