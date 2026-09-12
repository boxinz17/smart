from copy import deepcopy

import numpy as np
import pytest

from sparse_smart.source import ExactSource
from sparse_smart_v2.calibration import Margins, PracticalCalibration
from sparse_smart_v2.estimator import FitFailure, SparseSMARTv2


def problem():
    # Source order is fixed while the target anchors are deliberately not row 0.
    X = 2 * np.eye(4)
    source = np.diag([5., 4., 3., 2.])
    P = np.array([0., .8, .6, 0.])
    Q = np.array([0., .6, .8, 0.])
    coefficient = 2 * np.outer(P, Q)
    return X, X @ coefficient, source, coefficient


def model(**overrides):
    options = dict(rank=1, source_rank=3, margins=Margins(.1, 8., .1),
        calibration=PracticalCalibration(0., 0., 20., (3, 3)), iterations=4)
    options.update(overrides)
    return SparseSMARTv2(**options)


def test_full_frames_arbitrary_anchors_and_unpenalized_rows_survive_zero_caps():
    X, Y, source, coefficient = problem()
    fit = model(free_directions=(3, 3), iterations=0,
        calibration=PracticalCalibration(0., 7., 20., (0, 0))).fit(X, Y, source=source)
    assert fit.success_, fit.message_
    assert fit.source_.left.shape == (4, 4)
    assert fit.source_.right.shape == (4, 4)
    np.testing.assert_array_equal(fit.anchors_["u"].indices, [1])
    np.testing.assert_array_equal(fit.anchors_["v"].indices, [2])
    np.testing.assert_array_equal(fit.free_rows_.rows_u, [0, 1, 2])
    np.testing.assert_array_equal(fit.free_rows_.rows_v, [0, 1, 2])
    np.testing.assert_allclose(fit.coefficient_, coefficient, atol=1e-12)
    assert fit.metadata_["fitted_support_counts"] == (0, 0)
    assert fit.metadata_["free_dimension"] == 5
    assert not fit.metadata_["theorem_certified"]
    assert not fit.metadata_["initialization_spectrum_repaired"]


def test_initial_hard_caps_remove_only_outside_coordinates_and_keep_orthogonality():
    X, Y, source, _ = problem()
    fit = model(free_directions=(1, 3), iterations=0,
        calibration=PracticalCalibration(0., 0., 20., (0, 0))).fit(X, Y, source=source)
    assert fit.success_, fit.message_
    P, d, Q = fit.chart_.reconstruct(fit.state_)
    np.testing.assert_allclose(P.T @ P, np.eye(1), atol=1e-13)
    np.testing.assert_allclose(Q.T @ Q, np.eye(1), atol=1e-13)
    assert np.count_nonzero(P[[0, 2, 3]]) == 0
    assert Q[1, 0] != 0  # Nonanchor, but protected by the larger right free set.
    assert np.count_nonzero(fit.state_ - fit.untruncated_initial_state_) == 1
    np.testing.assert_allclose(fit.predict(X), X @ fit.coefficient_, atol=1e-13)


def test_source_frames_remain_read_only_in_fit_and_checkpoint_views():
    X, Y, source, _ = problem()
    fit = model(iterations=0).fit(X, Y, source=source)
    for fitted in (fit, fit.checkpoint_model(0)):
        for name in ("left", "right", "leading_left", "leading_right", "source_singular_values"):
            assert not getattr(fitted.source_, name).flags.writeable
        with pytest.raises(ValueError):
            fitted.source_.left[0, 0] = 0


def test_exact_rank_two_fixed_point_succeeds_without_optional_tolerance():
    X = 2 * np.eye(4)
    Y = X @ np.diag([3., 1., 0., 0.])
    fit = model(rank=2, calibration=PracticalCalibration(0., 0., 20., (4, 4)),
        validation_patience=1, validation_min_iterations=0).fit(
            X, Y, source=np.diag([5., 4., 3., 2.]), validation_data=(X, Y))
    assert fit.success_, fit.message_
    assert fit.n_iter_ == fit.selected_iteration_ == 0
    assert fit.termination_reason_ == "stationarity"
    assert fit.optimization_converged_ and fit.converged_
    assert not fit.validation_stopping_["stopped"]
    assert fit.checkpoint_model(0).termination_reason_ == "stationarity"
    np.testing.assert_allclose(fit.predict(X), Y, atol=1e-13)


def test_exact_source_can_refine_outside_initializer_span():
    X = 2 * np.eye(4)
    U = np.eye(4)[:, :3]
    P = np.array([0., np.sqrt(.9), 0., np.sqrt(.1)])
    C = 2 * np.outer(P, [0., 0., 1., 0.])
    fit = model(iterations=3).fit(X, X @ C, source=ExactSource(U, U))
    assert fit.success_, fit.message_
    assert fit.source_.left.shape == (4, 4)
    assert np.linalg.norm(fit.last_coefficient_[3]) > 1e-4
    for checkpoint in fit.checkpoints_.values():
        P, _, Q = fit.chart_.reconstruct(checkpoint.state)
        np.testing.assert_allclose(P.T @ P, np.eye(1), atol=1e-12)
        np.testing.assert_allclose(Q.T @ Q, np.eye(1), atol=1e-12)
    assert all(b.objective <= a.objective + 1e-12 for a, b in zip(fit.history_, fit.history_[1:]))


@pytest.mark.parametrize("scale", [0., .01, 10.])
def test_initializer_spectrum_is_rejected_without_projection(scale):
    X, Y, source, _ = problem()
    fit = model().fit(X, scale * Y, source=source)
    assert not fit.success_
    assert fit.status_ == "initialization_spectrum_failed"
    assert not hasattr(fit, "coefficient_")
    assert not fit.metadata_["initialization_spectrum_repaired"]
    with pytest.raises(FitFailure):
        fit.predict(X, allow_partial=True)


def test_adjacent_initializer_gap_is_checked_and_raising_policy_is_preserved():
    X = 2 * np.eye(4)
    Y = X @ np.diag([2., 1.95, 0., 0.])
    fit = model(rank=2, iterations=0).fit(X, Y, source=np.diag([5., 4., 3., 2.]))
    assert fit.status_ == "initialization_spectrum_failed"
    X, Y, source, _ = problem()
    with pytest.raises(FitFailure, match="initialization_spectrum_failed"):
        model(raise_on_failure=True).fit(X, 0 * Y, source=source)


def test_uncertified_original_anchor_fails_instead_of_lowering_floor():
    X = 3 * np.eye(9)
    Y = X @ (2 * np.ones((9, 9)) / 9)
    fit = model(source_rank=9, margins=Margins(.1, 8., .1, anchor_min=.05),
        calibration=PracticalCalibration(0., 0., 20., (8, 8))).fit(
            X, Y, source=np.diag(np.arange(9, 0, -1)))
    assert fit.status_ == "anchor_selection_failed"
    assert not hasattr(fit, "coefficient_")


def validation_problem():
    X, Y, source, _ = problem()
    calibration = PracticalCalibration(.08, 0., 20., (3, 3))
    warm = model(iterations=0, calibration=calibration).fit(X, Y, source=source)
    assert warm.success_, warm.message_
    return X, Y, source, calibration, warm.predict(X)


def test_validation_selects_early_state_and_checkpoints_are_independent():
    X, Y, source, calibration, YV = validation_problem()
    fit = model(calibration=calibration, checkpoint_iterations=(1, 3),
        validation_interval=1).fit(X, Y, source=source, validation_data=(X, YV))
    assert fit.success_, fit.message_
    assert fit.n_iter_ == 4 and fit.selected_iteration_ == 0
    assert fit.termination_reason_ == "max_iterations"
    assert not fit.converged_
    assert fit.best_validation_loss_ == pytest.approx(0., abs=1e-25)
    assert np.linalg.norm(fit.coefficient_ - fit.last_coefficient_) > 1e-4
    assert fit.checkpoint_iterations_ == (0, 1, 3, 4)
    prefix = fit.checkpoint_model(1)
    assert prefix.success_ and prefix.n_iter_ == 1 and prefix.selected_iteration_ == 0
    assert max(row["iteration"] for row in prefix.validation_history_) == 1
    original = fit.coefficient_.copy()
    prefix.coefficient_[:] = 0
    np.testing.assert_array_equal(fit.coefficient_, original)
    with pytest.raises(FitFailure, match="checkpoint_unavailable"):
        fit.checkpoint_model(2)


def test_validation_patience_stops_updates_without_claiming_convergence():
    X, Y, source, calibration, YV = validation_problem()
    fit = model(calibration=calibration, iterations=20, validation_interval=1,
        validation_patience=2, validation_min_iterations=0).fit(
            X, Y, source=source, validation_data=(X, YV))
    assert fit.success_, fit.message_
    assert fit.n_iter_ == 2 and fit.selected_iteration_ == 0
    assert fit.termination_reason_ == "validation_stop"
    assert not fit.optimization_converged_ and not fit.converged_
    assert fit.validation_stopping_["stopped"]
    assert fit.checkpoint_model(2).termination_reason_ == "validation_stop"


def test_final_off_schedule_validation_is_measured_but_does_not_change_termination():
    X, Y, source, calibration, _ = validation_problem()
    fit = model(calibration=calibration, validation_interval=25, validation_iterations=(2, 100)).fit(
        X, Y, source=source, validation_data=(X, Y))
    assert fit.success_, fit.message_
    assert [row["iteration"] for row in fit.validation_history_] == [0, 2, 4]
    assert fit.termination_reason_ == "max_iterations"


def test_refinement_failure_requires_partial_opt_in_but_preserves_successful_prefix(monkeypatch):
    import sparse_smart_v2.estimator as module
    original_refine = module.refine

    def failed(*args, **kwargs):
        kwargs["iterations"] = 0
        result = original_refine(*args, **kwargs)
        result.status, result.message = "line_search_failed", "Simulated rejected trials"
        result.termination_reason = "line_search_failed"
        return result

    monkeypatch.setattr(module, "refine", failed)
    X, Y, source, _ = problem()
    fit = model().fit(X, Y, source=source)
    assert not fit.success_ and fit.status_ == "line_search_failed"
    with pytest.raises(FitFailure, match="allow_partial"):
        fit.predict(X)
    assert np.isfinite(fit.predict(X, allow_partial=True)).all()
    assert fit.checkpoint_model(0).success_


def test_refit_clears_stale_success_results_even_for_invalid_input():
    X, Y, source, _ = problem()
    fit = model(iterations=0).fit(X, Y, source=source)
    first = deepcopy(fit.coefficient_)
    fit.fit(X, Y, source=source)
    np.testing.assert_array_equal(fit.coefficient_, first)
    with pytest.raises(ValueError, match="matching"):
        fit.fit(X, Y[:-1], source=source)
    assert not hasattr(fit, "coefficient_") and not fit.checkpoints_
    with pytest.raises(FitFailure):
        fit.predict(X)
    fit.fit(X, Y, source=source)
    fit.fit(X, 0 * Y, source=source)
    assert not hasattr(fit, "coefficient_") and not fit.checkpoints_


def test_tuner_cache_reuses_source_and_lasso_without_sharing_mutable_fitted_state(monkeypatch):
    import sparse_smart_v2.estimator as module
    source_calls, init_calls = [], []
    prepare, initialize = module.prepare_source, module.reduced_lasso

    def counted_source(*args, **kwargs):
        source_calls.append(1)
        return prepare(*args, **kwargs)

    def counted_init(*args, **kwargs):
        init_calls.append(1)
        return initialize(*args, **kwargs)

    monkeypatch.setattr(module, "prepare_source", counted_source)
    monkeypatch.setattr(module, "reduced_lasso", counted_init)
    X, Y, source, _ = problem()
    cache = {}
    first = model(iterations=0).fit(X, Y, source=source, _initialization_cache=cache)
    original_P = first.initialization_.P.copy()
    first.initialization_.P[:] = 0
    second = model(iterations=0).fit(X, Y, source=source, _initialization_cache=cache)
    assert first.success_ and second.success_
    assert len(source_calls) == len(init_calls) == 1
    np.testing.assert_array_equal(second.initialization_.P, original_P)
    assert not any(value is cache for value in vars(second).values())
    with pytest.raises(ValueError, match="different training"):
        second.fit(X.copy(), Y, source=source, _initialization_cache=cache)


@pytest.mark.parametrize("override", [
    {"rank": True}, {"source_rank": 5}, {"iterations": -1},
    {"validation_interval": 0}, {"checkpoint_iterations": [0, 0]},
    {"checkpoint_iterations": [5]}, {"stationarity_tol": float("nan")},
    {"validation_patience": 2},
])
def test_invalid_inputs_raise(override):
    X, Y, source, _ = problem()
    with pytest.raises(ValueError):
        model(**override).fit(X, Y, source=source)
