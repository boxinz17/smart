"""Training/validation isolation, candidate eligibility, and real-fit checks."""

import numpy as np
import pytest

from sparse_smart import ExactSource, FitFailure, Margins, NoisySource
from sparse_smart import tuning
from sparse_smart.tuning import SparseSMARTTuner


def _tuner(**kwargs):
    options = dict(rank=1, source_rank=2, sparsity=(1, 1),
                   margins=Margins(.05, 6., .01), init_penalties=(.03,),
                   penalties_u=(.01, .02), penalties_v=(.01,), iterations=3)
    options.update(kwargs)
    return SparseSMARTTuner(**options)


@pytest.fixture
def data():
    X = np.column_stack((np.ones(10), np.arange(10), np.zeros(10)))
    Y = np.zeros((10, 2))
    source = ExactSource(np.eye(3)[:, :2], np.eye(2))
    return X, Y, source


@pytest.fixture
def fake_estimator(monkeypatch):
    class FakeEstimator:
        instances = []
        outcomes = {}

        def __init__(self, **kwargs):
            self.options = kwargs
            self.calibration = kwargs["calibration"]
            self.instances.append(self)

        def fit(self, X, Y, *, source, validation_data):
            self.training_X, self.training_Y = X.copy(), Y.copy()
            self.validation_X, self.validation_Y = (value.copy() for value in validation_data)
            self.source = source
            pu, pv = self.calibration.penalty
            outcome = self.outcomes.get((pu, pv), {})
            if "exception" in outcome:
                raise outcome["exception"]
            self.success_ = outcome.get("success", True)
            self.status_ = "converged" if self.success_ else "line_search_failed"
            self.message_, self.termination_reason_ = self.status_, self.status_
            self.n_iter_, self.selected_iteration_ = 3, outcome.get("selected_iteration", 1)
            self.validation_history_ = [{"iteration": 0, "loss": 9.}, {"iteration": 1, "loss": 1.}]
            if outcome.get("has_coefficient", True):
                self.coefficient_ = np.zeros((X.shape[1], Y.shape[1]))
                self.coefficient_[0, 0] = outcome.get("coefficient", pu + pv)
            return self

        def predict(self, X, *, allow_partial=False):
            if not self.success_ and not allow_partial:
                raise FitFailure(self.status_, "failed prediction without opt-in")
            return X @ self.coefficient_

    monkeypatch.setattr(tuning, "SparseSMART", FakeEstimator)
    return FakeEstimator


def test_every_candidate_uses_same_disjoint_training_split_without_refit(data, fake_estimator):
    X, Y, source = data
    tuned = _tuner(init_penalties=(.01, .03), penalties_v=(.01, .04)).fit(X, Y, source=source)
    assert tuned.success_ and tuned.n_candidates_ == 8
    assert len(fake_estimator.instances) == 8
    train, validation = tuned.train_indices_, tuned.validation_indices_
    assert not set(train) & set(validation)
    np.testing.assert_array_equal(np.sort(np.concatenate((train, validation))), np.arange(len(X)))
    assert (len(train), len(validation)) == (8, 2)
    for candidate in fake_estimator.instances:
        np.testing.assert_array_equal(candidate.training_X, X[train])
        np.testing.assert_array_equal(candidate.training_Y, Y[train])
        np.testing.assert_array_equal(candidate.validation_X, X[validation])
        np.testing.assert_array_equal(candidate.validation_Y, Y[validation])
        assert candidate.source is source
        assert candidate.options["spectral_step"] == "projected"
        assert candidate.options["stationarity_tol"] == 1e-6
        assert candidate.options["initialization_spectrum"] == "auto"
        assert candidate.options["refinement_solver"] == "auto"
    assert tuned.best_params_["init_penalty"] == .01  # earliest tie
    assert tuned.best_params_["penalty_u"] == .01
    assert tuned.best_params_["penalty_v"] == .01
    assert tuned.best_score_ == pytest.approx(.02 ** 2 / 2)
    assert tuned.selected_iteration_ == 1
    assert not tuned.diagnostics_["refit_on_all_data"]
    assert not tuned.diagnostics_["validation_score_is_independent_test_estimate"]
    assert tuned.estimator_ is tuned.model_
    np.testing.assert_allclose(tuned.predict(X), X @ tuned.coefficient_)


def test_holdout_split_is_reproducible_and_random_state_changes_it(data, fake_estimator):
    X, Y, source = data
    a = _tuner(random_state=19).fit(X, Y, source=source)
    b = _tuner(random_state=19).fit(X, Y, source=source)
    c = _tuner(random_state=41).fit(X, Y, source=source)
    np.testing.assert_array_equal(a.train_indices_, b.train_indices_)
    np.testing.assert_array_equal(a.validation_indices_, b.validation_indices_)
    assert not np.array_equal(a.validation_indices_, c.validation_indices_)


def test_explicit_validation_drives_selection_but_never_enters_training(data, fake_estimator):
    X, Y, source = data
    fake_estimator.outcomes = {(.01, .01): {"coefficient": 1.}, (.02, .01): {"coefficient": 2.}}
    Xv = X[:3].copy()
    Yv = np.column_stack((np.full(3, 2.), np.zeros(3)))
    tuned = _tuner().fit(X, Y, source=source, validation_data=(Xv, Yv))
    assert tuned.train_indices_ is None and tuned.validation_indices_ is None
    assert tuned.split_mode_ == "explicit_validation"
    assert tuned.best_params_["penalty_u"] == .02
    assert tuned.best_score_ == 0
    assert len(fake_estimator.instances) == 2
    for candidate in fake_estimator.instances:
        np.testing.assert_array_equal(candidate.training_X, X)
        np.testing.assert_array_equal(candidate.training_Y, Y)
        np.testing.assert_array_equal(candidate.validation_Y, Yv)


@pytest.mark.parametrize("noisy,expected", [(False, (1, 1)), (True, (2, 1))])
def test_default_supports_use_actual_working_dimensions_and_source_gate_stays_on(
        data, fake_estimator, noisy, expected):
    X, Y, source = data
    if noisy:
        source = NoisySource(np.eye(3, 2), .01, 1.)
    tuned = _tuner().fit(X, Y, source=source)
    for candidate in fake_estimator.instances:
        assert candidate.calibration.support_limits == expected
        assert candidate.calibration.enforce_source_accuracy
    assert tuned.diagnostics_["maximum_complement_support"] == expected


def test_candidate_support_pairs_are_searched_independently(data, fake_estimator):
    X, Y, source = data
    tuned = _tuner(support_limits=((0, 0), (1, 1))).fit(X, Y, source=source)
    assert tuned.n_candidates_ == 4
    assert [r["params"]["support_limits"] for r in tuned.selection_history_] == [(0, 0), (1, 1)] * 2


def test_failed_partial_with_better_validation_score_cannot_win(data, fake_estimator):
    X, Y, source = data
    fake_estimator.outcomes = {
        (.01, .01): {"success": False, "coefficient": 0.},
        (.02, .01): {"success": True, "coefficient": 1.},
    }
    tuned = _tuner().fit(X, Y, source=source)
    assert tuned.success_ and tuned.best_params_["penalty_u"] == .02
    failed = tuned.selection_history_[0]
    assert not failed["success"] and failed["status"] == "line_search_failed"
    assert failed["has_partial_coefficient"]
    assert failed["partial_validation_mse"] == 0
    assert failed["validation_mse"] is None
    assert tuned.best_score_ == .5


def test_all_failures_remain_explicit_and_refit_clears_previous_winner(data, fake_estimator):
    X, Y, source = data
    tuned = _tuner().fit(X, Y, source=source)
    assert tuned.success_ and hasattr(tuned, "coefficient_")
    fake_estimator.outcomes = {
        (.01, .01): {"success": False, "coefficient": 0.},
        (.02, .01): {"exception": FloatingPointError("validation overflow")},
    }
    tuned.fit(X, Y, source=source)
    assert not tuned.success_ and tuned.status_ == "no_successful_candidate"
    assert tuned.model_ is None and tuned.best_params_ is None and tuned.best_score_ is None
    assert not hasattr(tuned, "coefficient_")
    assert len(tuned.selection_history_) == 2
    assert tuned.selection_history_[1]["status"] == "FloatingPointError"
    with pytest.raises(FitFailure, match="No successful"):
        tuned.predict(X)


def test_nonfinite_validation_candidate_is_recorded_and_not_selected(data, fake_estimator):
    X, Y, source = data
    fake_estimator.outcomes = {(.01, .01): {"coefficient": float("inf")}}
    tuned = _tuner().fit(X, Y, source=source)
    assert tuned.success_ and tuned.best_params_["penalty_u"] == .02
    assert not tuned.selection_history_[0]["success"]
    assert tuned.selection_history_[0]["validation_mse"] is None


@pytest.mark.parametrize("options", [
    {"init_penalties": ()}, {"init_penalties": (0,)}, {"penalties_u": (-1,)},
    {"penalties_v": (float("nan"),)}, {"penalties_u": (True,)},
    {"support_limits": ()}, {"support_limits": ((2, 1),)},
    {"support_limits": ((True, 1),)}, {"support_limits": ((0,),)},
    {"validation_fraction": 1.}, {"validation_fraction": 0.},
    {"random_state": True}, {"random_state": -1},
    {"initialization_spectrum": "unknown"}, {"refinement_solver": "unknown"},
    {"initialization_spectrum": np.array(["auto"])}, {"refinement_solver": np.array(["auto"])},
])
def test_invalid_search_or_split_inputs_fail_before_fitting(data, fake_estimator, options):
    X, Y, source = data
    with pytest.raises(ValueError):
        _tuner(**options).fit(X, Y, source=source)
    assert not fake_estimator.instances


def test_validation_dimensions_and_single_row_split_are_rejected(data, fake_estimator):
    X, Y, source = data
    with pytest.raises(ValueError, match="validation data"):
        _tuner().fit(X, Y, source=source, validation_data=(X[:2], Y[:3]))
    with pytest.raises(ValueError, match="one training and one validation"):
        _tuner().fit(X[:1], Y[:1], source=source)


def test_truth_is_not_an_input(data):
    X, Y, source = data
    with pytest.raises(TypeError, match="C_star"):
        _tuner().fit(X, Y, source=source, C_star=np.zeros((3, 2)))


def test_real_fit_selects_a_recorded_validation_iterate_and_preserves_training_size():
    rng = np.random.default_rng(194)
    X = rng.normal(size=(100, 4))
    C = np.zeros((4, 3))
    C[0, 0] = 2.
    Y = X @ C + .05 * rng.normal(size=(100, 3))
    source = ExactSource(np.eye(4)[:, :2], np.eye(3)[:, :2])
    fitted = SparseSMARTTuner(
        rank=1, source_rank=2, sparsity=(1, 1), margins=Margins(.05, 6., .01),
        init_penalties=(.01,), penalties_u=(.001, .01), penalties_v=(.003,),
        iterations=10, step_size_inverse=5., random_state=22,
    ).fit(X, Y, source=source)
    assert fitted.success_, fitted.selection_history_
    assert fitted.training_sample_count_ == 80 and fitted.validation_sample_count_ == 20
    prediction = fitted.predict(X[fitted.validation_indices_])
    score = np.mean((prediction - Y[fitted.validation_indices_]) ** 2)
    assert fitted.best_score_ == pytest.approx(score)
    assert fitted.best_score_ == pytest.approx(fitted.model_.best_validation_loss_)
    assert fitted.selected_iteration_ == fitted.model_.selected_iteration_
    assert fitted.model_.calibration_.support_limits == (1, 1)
    assert len(fitted.model_.validation_history_) == fitted.model_.n_iter_ + 1
    assert not fitted.diagnostics_["theorem_certified"]
