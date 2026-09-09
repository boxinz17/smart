import numpy as np
import pytest

from sparse_smart import ExactSource, Margins, PracticalCalibration, SparseSMART


def scalar_model():
    return SparseSMART(rank=1, source_rank=1, sparsity=(1, 1),
        margins=Margins(.1, 5., .1, trial_radius=2.),
        calibration=PracticalCalibration(.1, (0., 0.), 1., (0, 0)), iterations=4)


def test_validation_can_keep_initializer_without_changing_training_trajectory():
    X, Y = np.ones((4, 1)), np.full((4, 1), 2.)
    source = ExactSource(np.eye(1), np.eye(1))
    baseline = scalar_model().fit(X, Y, source=source)
    validation = (np.ones((3, 1)), np.full((3, 1), 1.9))
    selected = scalar_model().fit(X, Y, source=source, validation_data=validation)
    assert selected.success_ and selected.selected_iteration_ == 0
    assert selected.n_iter_ == baseline.n_iter_ == 4
    np.testing.assert_array_equal(selected.last_state_, baseline.state_)
    np.testing.assert_allclose(selected.coefficient_, [[1.9]], atol=1e-12)
    np.testing.assert_allclose(selected.last_coefficient_, [[2.]], atol=1e-12)
    np.testing.assert_allclose(selected.predict(validation[0]), validation[1], atol=1e-12)
    assert selected.best_validation_loss_ < 1e-24
    assert [item['iteration'] for item in selected.validation_history_] == list(range(5))
    assert [item.objective for item in selected.history_] == [item.objective for item in baseline.history_]
    assert selected.termination_reason_ == 'max_iterations'
    assert not selected.converged_


def test_validation_ties_keep_earliest_best_iterate_and_refit_clears_selection():
    X, Y = np.ones((4, 1)), np.full((4, 1), 2.)
    source = ExactSource(np.eye(1), np.eye(1))
    fitted = scalar_model().fit(X, Y, source=source, validation_data=(X, Y))
    assert fitted.selected_iteration_ == 1
    fitted.fit(X, Y, source=source)
    assert fitted.selected_iteration_ == 4
    assert fitted.validation_history_ == [] and fitted.best_validation_loss_ is None


def test_selected_initializer_does_not_inherit_terminal_stationarity():
    X, Y = np.ones((4, 1)), np.full((4, 1), 2.)
    fitted = scalar_model()
    fitted.stationarity_tol = 1e-6
    fitted.fit(X, Y, source=ExactSource(np.eye(1), np.eye(1)),
               validation_data=(X, np.full((4, 1), 1.9)))
    assert fitted.termination_reason_ == 'stationarity'
    assert fitted.optimization_converged_ and not fitted.converged_
    assert fitted.selected_iteration_ == 0 and fitted.n_iter_ == 1
    assert fitted.diagnostics_['projected_gradient_norm'] > .09
    assert fitted.diagnostics_['last_projected_gradient_norm'] < 1e-10


@pytest.mark.parametrize('validation', [(), (np.ones((2, 2)), np.ones((2, 1))),
    (np.ones((2, 1)), np.ones((3, 1))), (np.ones((2, 1)), np.ones((2, 2))),
    (np.ones((2, 1)), np.full((2, 1), np.nan))])
def test_validation_shapes_and_values_are_checked(validation):
    with pytest.raises(ValueError):
        scalar_model().fit(np.ones((4, 1)), np.ones((4, 1)),
            source=ExactSource(np.eye(1), np.eye(1)), validation_data=validation)


@pytest.mark.parametrize('option', [{'spectral_step': 'unknown'}, {'stationarity_tol': True},
                                  {'stationarity_tol': -1.}])
def test_solver_options_are_checked(option):
    estimator = scalar_model()
    for key, value in option.items():
        setattr(estimator, key, value)
    with pytest.raises(ValueError):
        estimator.fit(np.ones((4, 1)), np.ones((4, 1)), source=ExactSource(np.eye(1), np.eye(1)))
