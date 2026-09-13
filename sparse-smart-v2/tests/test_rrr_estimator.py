"""The unregularized endpoint is target-only RRR, outside the source chart."""

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

from sparse_smart_v2 import Margins, PracticalCalibration, SparseSMARTv2
from sparse_smart_v2.estimator import FitFailure
from sparse_smart_v2.tuning import SparseSMARTv2Tuner


def problem():
    rng = np.random.default_rng(282)
    X = rng.normal(size=(40, 6)) * [4., 2., 1., .5, .2, .1]
    C = rng.normal(size=(6, 5))
    Y = X @ C + .15 * rng.normal(size=(40, 5))
    XV = rng.normal(size=(11, 6))
    YV = XV @ C + .15 * rng.normal(size=(11, 5))
    source = np.zeros((6, 5))
    source[:5] = np.diag(np.arange(5, 0, -1))
    return X, Y, source, XV, YV


def oracle(X, Y, rank):
    """Independent response-projection expression for ordinary target RRR."""
    fitted_response = X @ np.linalg.lstsq(X, Y, rcond=None)[0]
    _, _, vt = np.linalg.svd(fitted_response, full_matrices=False)
    response_projection = vt[:rank].T @ vt[:rank]
    return np.linalg.lstsq(X, Y @ response_projection, rcond=None)[0]


def model(**overrides):
    options = dict(rank=2, source_rank=3, free_directions=(3, 3),
        margins=Margins(.01, 20., .001, anchor_min=.04),
        calibration=PracticalCalibration(.03, (0., 0.), 20., (6, 4)),
        iterations=10, checkpoint_iterations=(2, 5), validation_interval=1)
    options.update(overrides)
    return SparseSMARTv2(**options)


@pytest.mark.parametrize("solver", ["masked_chart_spectral_soft_hard", "masked_anchor_projected"])
def test_zero_penalties_full_caps_return_rrr_before_source_or_initializer(monkeypatch, solver):
    import sparse_smart_v2.estimator as module

    def forbidden(*args, **kwargs):
        raise AssertionError("The unregularized endpoint must not use the source initializer or chart")

    monkeypatch.setattr(module, "prepare_source", forbidden)
    monkeypatch.setattr(module, "reduced_lasso", forbidden)
    monkeypatch.setattr(module, "refine", forbidden)
    X, Y, _, _, _ = problem()
    fit = model(refinement_solver=solver).fit(X, Y, source=None)
    assert fit.success_ and fit.status_ == "converged"
    assert fit.method_ == "target_rrr"
    assert fit.n_iter_ == fit.selected_iteration_ == 0
    assert fit.optimization_converged_ and fit.converged_
    assert fit.termination_reason_ == "target_rrr_closed_form"
    assert fit.metadata_["fit_method"] == "target_rrr"
    assert fit.metadata_["refinement_solver"] == "target_rrr_closed_form"
    assert fit.metadata_["requested_refinement_solver"] == solver
    assert fit.metadata_["selection_rule"] == "single_target_rrr_fit"
    for name in ("source_", "initialization_", "chart_", "state_", "selected_chart_", "free_rows_"):
        assert not hasattr(fit, name), f"Direct RRR should not manufacture {name}"
    assert fit.rrr_result_ is not None and fit.rrr_certificate_
    np.testing.assert_allclose(fit.coefficient_, oracle(X, Y, 2), rtol=2e-12, atol=2e-12)
    np.testing.assert_array_equal(fit.last_coefficient_, fit.coefficient_)
    np.testing.assert_allclose(fit.predict(X), X @ fit.coefficient_, atol=1e-13)
    np.testing.assert_allclose((fit.left_factors_ * fit.factors_["d"]) @ fit.right_factors_.T,
                               fit.coefficient_, atol=2e-12)


@pytest.mark.parametrize("free,penalties,caps", [
    ((6, 5), (9., 13.), (0, 0)),
    ((6, 3), (9., 0.), (0, 4)),
    ((3, 5), (0., 13.), (6, 0)),
])
def test_empty_penalty_masks_count_as_zero_effective_penalties(free, penalties, caps):
    X, Y, _, _, _ = problem()
    fit = model(free_directions=free,
        calibration=PracticalCalibration(1e12, penalties, 20., caps)).fit(X, Y, source=None)
    assert fit.success_ and fit.method_ == "target_rrr"
    np.testing.assert_allclose(fit.coefficient_, oracle(X, Y, 2), atol=2e-12)


@pytest.mark.parametrize("free,penalties,caps", [
    ((3, 3), (0., 0.), (5, 4)),
    ((3, 3), (0., 0.), (6, 3)),
    ((3, 3), (1e-16, 0.), (6, 4)),
    ((6, 3), (7., .01), (0, 4)),
])
def test_restrictive_caps_or_nonzero_effective_penalty_preserve_chart_path(monkeypatch, free, penalties, caps):
    import sparse_smart_v2.estimator as module

    class ReachedSource(Exception):
        pass

    def source_path(*args, **kwargs):
        raise ReachedSource

    monkeypatch.setattr(module, "prepare_source", source_path)
    X, Y, source, _, _ = problem()
    with pytest.raises(ReachedSource):
        model(free_directions=free,
            calibration=PracticalCalibration(.03, penalties, 20., caps)).fit(X, Y, source=source)


def test_source_rank_margins_and_initialization_cannot_exclude_rrr():
    X, Y, source, _, _ = problem()
    narrow = Margins(100., 1000., 2., anchor_min=.06)
    fit = model(source_rank=1, margins=narrow,
        calibration=PracticalCalibration(1e12, 0., 20., (6, 4))).fit(X, Y, source=source)
    assert fit.success_ and fit.method_ == "target_rrr"
    assert fit.factors_["d"][-1] < narrow.d_lower
    np.testing.assert_allclose(fit.coefficient_, oracle(X, Y, 2), atol=2e-12)
    # Opting out restores the source-dimension requirement, not an implicit fallback.
    with pytest.raises(ValueError, match="rank.*source_rank"):
        model(source_rank=1, rrr_shortcut=False).fit(X, Y, source=source)


def test_explicit_optout_retains_initialization_rejection():
    X, Y, source, _, _ = problem()
    fit = model(rrr_shortcut=False, margins=Margins(100., 1000., 2.)).fit(X, Y, source=source)
    assert not fit.success_ and fit.status_ == "initialization_spectrum_failed"
    assert not hasattr(fit, "rrr_result_")


@pytest.mark.parametrize("overrides", [
    {"free_directions": (1, 3)}, {"free_directions": (7, 3)},
    {"free_directions": (3, True)}, {"rank": 6}, {"source_rank": 6},
    {"rrr_shortcut": 1},
    {"calibration": PracticalCalibration(.03, 0., 20., (7, 4))},
    {"free_directions": (6, 5), "calibration": PracticalCalibration(.03, 4., 20., (1, 0))},
])
def test_direct_path_still_checks_rank_masks_caps_and_configuration(overrides):
    X, Y, _, _, _ = problem()
    with pytest.raises(ValueError):
        model(**overrides).fit(X, Y, source=None)


def test_validation_measures_only_rrr_and_cannot_select_a_better_initializer():
    X, Y, _, XV, _ = problem()
    expected = oracle(X, Y, 2)
    # Zero validation response would favor a heavily shrunk initializer. It
    # must not change which estimator is returned for this endpoint.
    YV = np.zeros((len(XV), Y.shape[1]))
    fit = model(validation_patience=1, validation_min_iterations=0,
        calibration=PracticalCalibration(1e12, 0., 20., (6, 4))).fit(
            X, Y, source=None, validation_data=(XV, YV))
    np.testing.assert_allclose(fit.coefficient_, expected, atol=2e-12)
    assert len(fit.validation_history_) == 1
    assert fit.validation_history_[0]["iteration"] == 0
    assert fit.best_validation_loss_ == pytest.approx(np.mean((XV @ expected - YV)**2))
    assert not fit.validation_stopping_["stopped"]
    assert fit.termination_reason_ == "target_rrr_closed_form"


def test_validation_patience_without_validation_is_irrelevant_to_direct_fit():
    X, Y, _, _, _ = problem()
    fit = model(validation_patience=3).fit(X, Y, source=None)
    assert fit.success_ and fit.validation_history_ == []
    assert fit.best_validation_loss_ is None
    assert not fit.validation_stopping_["stopped"]


def test_zero_response_is_a_valid_lower_rank_rrr_solution():
    X, Y, _, _, _ = problem()
    fit = model().fit(X, np.zeros_like(Y), source=None)
    assert fit.success_ and fit.optimization_converged_
    np.testing.assert_array_equal(fit.coefficient_, np.zeros((X.shape[1], Y.shape[1])))
    np.testing.assert_array_equal(fit.predict(X), np.zeros_like(Y))
    assert not hasattr(fit, "initialization_")


@pytest.mark.parametrize("bad_validation", ["not_data", (np.ones((2, 6)), np.ones((3, 5))),
    (np.ones((2, 7)), np.ones((2, 5))), (np.ones((2, 6)), np.full((2, 5), np.nan))])
def test_direct_fit_does_not_ignore_malformed_validation(bad_validation):
    X, Y, _, _, _ = problem()
    with pytest.raises(ValueError):
        model().fit(X, Y, source=None, validation_data=bad_validation)


def test_checkpoint_is_independent_direct_fit_and_later_iterates_are_unavailable():
    X, Y, _, XV, YV = problem()
    fit = model().fit(X, Y, source=None, validation_data=(XV, YV))
    assert fit.checkpoint_iterations_ == (0,)
    before = fit.coefficient_.copy()
    checkpoint = fit.checkpoint_model(0)
    assert checkpoint.success_ and checkpoint.method_ == "target_rrr"
    assert checkpoint.n_iter_ == checkpoint.selected_iteration_ == 0
    assert not hasattr(checkpoint, "chart_")
    np.testing.assert_array_equal(checkpoint.predict(XV), fit.predict(XV))
    checkpoint.coefficient_[:] = 0
    checkpoint.metadata_["fit_method"] = "changed"
    np.testing.assert_array_equal(fit.coefficient_, before)
    assert fit.metadata_["fit_method"] == "target_rrr"
    for iteration in (1, 2, 5, 10):
        with pytest.raises(FitFailure, match="checkpoint_unavailable"):
            fit.checkpoint_model(iteration)


def test_refit_clears_direct_state_after_invalid_input_and_switching_to_chart():
    X, Y, source, _, _ = problem()
    fit = model().fit(X, Y, source=None)
    first = fit.coefficient_.copy()
    fit.fit(X, Y, source=None)
    np.testing.assert_array_equal(fit.coefficient_, first)
    with pytest.raises(ValueError, match="matching"):
        fit.fit(X, Y[:-1], source=None)
    assert not hasattr(fit, "coefficient_") and not hasattr(fit, "rrr_result_")
    with pytest.raises(FitFailure):
        fit.predict(X)
    fit.fit(X, Y, source=None)
    fit.rrr_shortcut = False
    fit.margins = Margins(100., 1000., 2.)
    fit.fit(X, Y, source=source)
    assert fit.status_ == "initialization_spectrum_failed"
    assert not hasattr(fit, "rrr_result_") and not hasattr(fit, "coefficient_")


def test_uncertified_rrr_result_fails_without_exposing_prior_or_partial_coefficient(monkeypatch):
    import sparse_smart_v2.estimator as module

    X, Y, _, _, _ = problem()
    fit = model().fit(X, Y, source=None)
    assert fit.success_ and hasattr(fit, "coefficient_")
    original = module.target_rrr

    def uncertified(*args, **kwargs):
        result = original(*args, **kwargs)
        return replace(result, certificate={**result.certificate, "certified": False})

    monkeypatch.setattr(module, "target_rrr", uncertified)
    fit.fit(X, Y, source=None)
    assert not fit.success_ and fit.status_ == "numerical_failure"
    assert not fit.optimization_converged_
    assert not hasattr(fit, "coefficient_") and not hasattr(fit, "rrr_result_")
    assert not fit.checkpoints_
    with pytest.raises(FitFailure):
        fit.predict(X, allow_partial=True)


def test_direct_fit_cache_reuse_is_independent_and_bound_to_training_observations():
    X, Y, _, _, _ = problem()
    cache = {}
    first = model().fit(X, Y, source=None, _initialization_cache=cache)
    original = first.coefficient_.copy()
    first.coefficient_[:] = 0
    second = model(calibration=PracticalCalibration(1e12, 0., 20., (6, 4))).fit(
        X, Y, source=None, _initialization_cache=cache)
    assert second.success_
    assert first.numerical_work_["rrr_solves"] == 1
    assert second.numerical_work_["rrr_solves"] == 0
    assert second.numerical_work_["rrr_cache_hits"] == 1
    np.testing.assert_array_equal(second.coefficient_, original)
    assert not any(value is cache for value in vars(second).values())
    with pytest.raises(ValueError, match="different training"):
        second.fit(X.copy(), Y, source=None, _initialization_cache=cache)


def test_tuner_all_direct_candidates_never_prepare_source_or_lasso(monkeypatch):
    import sparse_smart_v2.estimator as estimator_module
    import sparse_smart_v2.tuning as tuning_module

    def forbidden(*args, **kwargs):
        raise AssertionError("RRR-only tuning must not inspect or initialize from the source")

    monkeypatch.setattr(estimator_module, "prepare_source", forbidden)
    monkeypatch.setattr(estimator_module, "reduced_lasso", forbidden)
    monkeypatch.setattr(tuning_module, "prepare_source", forbidden)
    X, Y, _, XV, YV = problem()
    candidates = [model(), model(free_directions=(6, 5),
        calibration=PracticalCalibration(1e12, 9., 20., (0, 0)))]
    originals = [deepcopy(vars(candidate)) for candidate in candidates]
    fitted = SparseSMARTv2Tuner(candidates).fit(X, Y, source=None, validation_data=(XV, YV))
    assert fitted.success_ and fitted.best_index_ == 0
    assert all(row["eligible"] for row in fitted.results_)
    assert fitted.best_estimator_.method_ == "target_rrr"
    np.testing.assert_allclose(fitted.coefficient_, oracle(X, Y, 2), atol=2e-12)
    assert fitted.best_score_ == pytest.approx(np.mean((XV @ fitted.coefficient_ - YV)**2))
    assert [vars(candidate) for candidate in candidates] == originals
    assert all(row["params"]["rrr_shortcut"] for row in fitted.results_)
    assert [row["numerical_work"]["rrr_solves"] for row in fitted.results_] == [1, 0]
    assert [row["numerical_work"]["rrr_cache_hits"] for row in fitted.results_] == [0, 1]
    assert fitted.metadata_["source_preparations"] == 0


def test_tuner_rrr_uses_training_rows_only_with_internal_holdout():
    X, Y, _, _, _ = problem()
    fit = SparseSMARTv2Tuner([model()], validation_fraction=.25, random_state=17).fit(
        X, Y, source=None)
    assert fit.success_, fit.results_
    train, validation = fit.train_indices_, fit.validation_indices_
    expected = oracle(X[train], Y[train], 2)
    np.testing.assert_allclose(fit.coefficient_, expected, atol=2e-12)
    assert fit.best_score_ == pytest.approx(np.mean((X[validation] @ expected - Y[validation])**2))
    assert np.linalg.norm(expected - oracle(X, Y, 2)) > .001


@pytest.mark.parametrize("direct_first", [True, False])
def test_bad_source_for_regularized_candidate_does_not_exclude_direct_candidate(direct_first):
    X, Y, _, XV, YV = problem()
    direct = model()
    chart = model(calibration=PracticalCalibration(.03, (.01, .01), 20., (6, 4)))
    candidates = [direct, chart] if direct_first else [chart, direct]
    fit = SparseSMARTv2Tuner(candidates).fit(X, Y, source=None, validation_data=(XV, YV))
    assert fit.success_, fit.results_
    assert fit.best_index_ == (0 if direct_first else 1)
    assert fit.best_estimator_.method_ == "target_rrr"
    assert sum(row["eligible"] for row in fit.results_) == 1
    assert fit.metadata_["source_preparations"] == 1
    np.testing.assert_allclose(fit.coefficient_, oracle(X, Y, 2), atol=2e-12)
