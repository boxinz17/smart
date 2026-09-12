"""Validation isolation, reproducibility, and failed-candidate exclusion."""

from copy import deepcopy

import numpy as np
import pytest

from sparse_smart_v2 import tuning
from sparse_smart_v2.tuning import SparseSMARTv2Tuner


class Candidate:
    """Small estimator protocol double with inspectable training calls."""

    calls = []

    def __init__(self, value, *, success=True, exception=None, mutate=False):
        self.source_rank = 1
        self.rank = 1
        self.value = value
        self.succeeds = success
        self.exception = exception
        self.mutate = mutate
        self.iterations = 5
        self.free_directions = (1, 1)

    def fit(self, X, Y, *, source, validation_data, _initialization_cache=None):
        Xv, Yv = validation_data
        self.calls.append((X.copy(), Y.copy(), Xv.copy(), Yv.copy(), source, self))
        self.cache_identity = id(_initialization_cache)
        if self.mutate:
            X[:] = 999
            Y[:] = 999
            Xv[:] = 999
            Yv[:] = 999
        if self.exception:
            raise self.exception
        self.success_ = self.succeeds
        self.status_ = "iteration_limit" if self.success_ else "line_search_failed"
        self.selected_iteration_, self.n_iter_ = 2, 5
        self.validation_history_ = [{"iteration": 0, "loss": 3.}, {"iteration": 2, "loss": 1.}]
        self.history_ = [{"iteration": 5}]
        self.coefficient_ = np.zeros((X.shape[1], Y.shape[1]))
        self.coefficient_[0, 0] = self.value
        return self

    def predict(self, X):
        return X @ self.coefficient_


@pytest.fixture
def sample(monkeypatch):
    Candidate.calls = []
    preparations = []
    prepared = object()

    def prepare(source, **kwargs):
        preparations.append((source, kwargs))
        return prepared

    monkeypatch.setattr(tuning, "prepare_source", prepare)
    X = np.column_stack([np.ones(12), np.arange(12)])
    Y = np.zeros((12, 1))
    return X, Y, object(), prepared, preparations


def test_same_disjoint_split_and_immutable_candidate_templates(sample):
    X, Y, source, prepared, preparations = sample
    candidates = [Candidate(1), Candidate(2)]
    before = [deepcopy(vars(candidate)) for candidate in candidates]
    fitted = SparseSMARTv2Tuner(candidates).fit(X, Y, source=source)
    assert fitted.success_ and fitted.best_index_ == 0
    assert fitted.best_score_ == 1
    train, validation = fitted.train_indices_, fitted.validation_indices_
    assert (len(train), len(validation)) == (8, 4)
    assert not set(train) & set(validation)
    np.testing.assert_array_equal(np.sort(np.r_[train, validation]), np.arange(len(X)))
    assert len(preparations) == 1 and len(Candidate.calls) == 2
    assert len({call[-1].cache_identity for call in Candidate.calls}) == 1
    for Xtr, Ytr, Xv, Yv, shared, fitted_candidate in Candidate.calls:
        np.testing.assert_array_equal(Xtr, X[train])
        np.testing.assert_array_equal(Ytr, Y[train])
        np.testing.assert_array_equal(Xv, X[validation])
        np.testing.assert_array_equal(Yv, Y[validation])
        assert shared is prepared
        assert all(fitted_candidate is not template for template in candidates)
    assert [vars(candidate) for candidate in candidates] == before
    assert fitted.best_estimator_ is fitted.estimator_ is fitted.model_
    assert fitted.candidate_results_ is fitted.results_
    assert fitted.selected_iteration_ == 2
    assert not fitted.metadata_["refit_on_all_data"]
    assert not fitted.metadata_["theorem_certified"]
    assert not hasattr(fitted, "candidate_models_")
    assert "history" not in fitted.results_[0]
    np.testing.assert_array_equal(fitted.predict(X), X @ fitted.coefficient_)


def test_reproducible_splits_and_deterministic_ties(sample):
    X, Y, source, _, _ = sample
    candidates = [Candidate(1), Candidate(1)]
    a = SparseSMARTv2Tuner(candidates, random_state=7).fit(X, Y, source=source)
    b = SparseSMARTv2Tuner(candidates, random_state=7).fit(X, Y, source=source)
    c = SparseSMARTv2Tuner(candidates, random_state=8).fit(X, Y, source=source)
    np.testing.assert_array_equal(a.validation_indices_, b.validation_indices_)
    assert not np.array_equal(a.validation_indices_, c.validation_indices_)
    assert a.best_index_ == b.best_index_ == c.best_index_ == 0


def test_explicit_validation_selects_winner_without_refitting(sample):
    X, Y, source, _, _ = sample
    Xv, Yv = X[:3] + [0, 100], np.full((3, 1), 2.)
    fitted = SparseSMARTv2Tuner([Candidate(1), Candidate(2)]).fit(
        X, Y, source=source, validation_data=(Xv, Yv))
    assert fitted.best_index_ == 1 and fitted.best_score_ == 0
    assert fitted.split_mode_ == "explicit_validation"
    assert fitted.train_indices_ is fitted.validation_indices_ is None
    assert len(Candidate.calls) == 2
    for Xtr, Ytr, actual_Xv, actual_Yv, _, _ in Candidate.calls:
        np.testing.assert_array_equal(Xtr, X)
        np.testing.assert_array_equal(Ytr, Y)
        np.testing.assert_array_equal(actual_Xv, Xv)
        np.testing.assert_array_equal(actual_Yv, Yv)


def test_failed_fit_cannot_win_even_with_better_partial_coefficient(sample):
    X, Y, source, _, _ = sample
    fitted = SparseSMARTv2Tuner([Candidate(0, success=False), Candidate(1)]).fit(X, Y, source=source)
    assert fitted.best_index_ == 1
    assert fitted.results_[0]["validation_mse"] is None
    assert not fitted.results_[0]["eligible"]
    assert fitted.results_[0]["status"] == "line_search_failed"


def test_pairwise_comparison_detects_improvement_hidden_by_common_large_loss(sample):
    X, _, source, _, _ = sample
    Y = np.zeros((len(X), 2))
    Yv = np.tile([2., 1e12], (3, 1))
    fitted = SparseSMARTv2Tuner([Candidate(1), Candidate(2)]).fit(
        X, Y, source=source, validation_data=(X[:3], Yv))
    assert fitted.results_[0]["validation_mse"] == fitted.results_[1]["validation_mse"]
    assert fitted.results_[1]["loss_difference"] == -0.5
    assert fitted.best_index_ == 1


def test_exceptions_and_nonfinite_predictions_are_audited(sample):
    X, Y, source, _, _ = sample
    candidates = [Candidate(0, exception=RuntimeError("bad fit")), Candidate(np.inf), Candidate(1)]
    with np.errstate(invalid="ignore"):
        fitted = SparseSMARTv2Tuner(candidates).fit(X, Y, source=source)
    assert fitted.best_index_ == 2
    assert fitted.results_[0]["status"] == "RuntimeError"
    assert "bad fit" in fitted.results_[0]["message"]
    assert fitted.results_[1]["status"] == "ValueError"
    assert all(not record["eligible"] for record in fitted.results_[:2])


def test_mutating_candidate_cannot_change_shared_split_or_caller_data(sample):
    X, Y, source, _, _ = sample
    original_X, original_Y = X.copy(), Y.copy()
    fitted = SparseSMARTv2Tuner([Candidate(1, mutate=True), Candidate(2)]).fit(X, Y, source=source)
    np.testing.assert_array_equal(Candidate.calls[1][0], original_X[fitted.train_indices_])
    np.testing.assert_array_equal(Candidate.calls[1][3], original_Y[fitted.validation_indices_])
    np.testing.assert_array_equal(X, original_X)
    np.testing.assert_array_equal(Y, original_Y)


def test_repeat_fit_clears_stale_winner_and_raise_policy(sample):
    X, Y, source, _, _ = sample
    tuner = SparseSMARTv2Tuner([Candidate(1)]).fit(X, Y, source=source)
    assert hasattr(tuner, "coefficient_")
    tuner.candidates = [Candidate(0, success=False)]
    tuner.fit(X, Y, source=source)
    assert not tuner.success_ and tuner.best_estimator_ is None
    assert tuner.status_ == "no_successful_candidate"
    assert not hasattr(tuner, "coefficient_")
    assert len(tuner.results_) == 1
    with pytest.raises(tuning.FitFailure, match="No successful"):
        tuner.predict(X)
    tuner.raise_on_failure = True
    with pytest.raises(tuning.FitFailure, match="No successful"):
        tuner.fit(X, Y, source=source)
    assert len(tuner.results_) == 1


@pytest.mark.parametrize("options", [
    {"validation_fraction": 0}, {"validation_fraction": 1},
    {"validation_fraction": True}, {"validation_fraction": np.nan},
    {"random_state": -1}, {"random_state": True},
])
def test_invalid_split_is_rejected_before_fit(sample, options):
    X, Y, source, _, preparations = sample
    with pytest.raises(ValueError):
        SparseSMARTv2Tuner([Candidate(1)], **options).fit(X, Y, source=source)
    assert not Candidate.calls and not preparations


def test_invalid_candidates_data_and_validation_are_rejected(sample):
    X, Y, source, _, _ = sample
    with pytest.raises(ValueError, match="nonempty"):
        SparseSMARTv2Tuner([]).fit(X, Y, source=source)
    other = Candidate(1)
    other.source_rank = 2
    with pytest.raises(ValueError, match="same source_rank"):
        SparseSMARTv2Tuner([Candidate(1), other]).fit(X, Y, source=source)
    tuner = SparseSMARTv2Tuner([Candidate(1)])
    with pytest.raises(ValueError, match="validation data"):
        tuner.fit(X, Y, source=source, validation_data=(X[:2], Y[:3]))
    with pytest.raises(ValueError, match="matching rows"):
        tuner.fit(X, Y[:2], source=source)
    with pytest.raises(ValueError, match="one training and one validation"):
        tuner.fit(X[:1], Y[:1], source=source)
    with pytest.raises(TypeError, match="C_star"):
        tuner.fit(X, Y, source=source, C_star=np.zeros((2, 1)))


def test_real_fits_share_source_and_training_initializer_without_retaining_cache(monkeypatch):
    from sparse_smart_v2 import Margins, ObservedSource, PracticalCalibration, SparseSMARTv2
    from sparse_smart_v2 import estimator as estimator_module
    from sparse_smart_v2 import source as source_module

    rng = np.random.default_rng(44)
    X = rng.normal(size=(72, 5))
    coefficient = np.zeros((5, 4))
    coefficient[0, 0] = 2.
    Y = X @ coefficient + 0.03 * rng.normal(size=(72, 4))
    Xv = rng.normal(size=(16, 5))
    Yv = Xv @ coefficient + 0.03 * rng.normal(size=(16, 4))
    matrix = np.zeros((5, 4))
    matrix[:3, :3] = np.diag([3., 2., 1.])
    calls = {"svd": 0, "initialization": 0, "source_validation": 0}
    original_svd = source_module.deterministic_svd
    original_lasso = estimator_module.reduced_lasso
    original_prepare = estimator_module.prepare_source

    def svd(*args, **kwargs):
        calls["svd"] += 1
        return original_svd(*args, **kwargs)

    def lasso(design, response, *args, **kwargs):
        calls["initialization"] += 1
        # Diagonal source has canonical leading coordinate axes; all input
        # rows to the cached initializer must be training rows only.
        np.testing.assert_allclose(design, X[:, :3])
        np.testing.assert_allclose(response, Y[:, :3])
        return original_lasso(design, response, *args, **kwargs)

    def prepare(*args, **kwargs):
        calls["source_validation"] += 1
        return original_prepare(*args, **kwargs)

    monkeypatch.setattr(source_module, "deterministic_svd", svd)
    monkeypatch.setattr(estimator_module, "reduced_lasso", lasso)
    monkeypatch.setattr(estimator_module, "prepare_source", prepare)
    candidates = [
        SparseSMARTv2(
            rank=1, source_rank=3, free_directions=free,
            margins=Margins(.01, 8., .005, anchor_min=.005),
            calibration=PracticalCalibration(init, .01, 20., (1, 1)),
            iterations=2, validation_interval=1,
        )
        for init in (.01, .02) for free in ((1, 1), (2, 2))
    ]
    fitted = SparseSMARTv2Tuner(candidates).fit(
        X, Y, source=ObservedSource(matrix), validation_data=(Xv, Yv))
    assert fitted.success_, fitted.results_
    assert all(record["eligible"] for record in fitted.results_)
    assert calls == {"svd": 1, "initialization": 2, "source_validation": 1}
    assert not hasattr(fitted.best_estimator_, "_initialization_cache")
    assert not any(hasattr(candidate, "coefficient_") for candidate in candidates)
    expected = np.mean((fitted.predict(Xv) - Yv) ** 2)
    assert fitted.best_score_ == pytest.approx(expected)
