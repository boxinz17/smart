import numpy as np
import pytest
from sparse_smart import (SparseSMART, ExactSource, NoisySource, Margins,
                          PracticalCalibration, PrescribedCalibration, FitFailure)


def example(seed=21):
    rng = np.random.default_rng(seed)
    n, p, q, r0, r = 120, 10, 9, 7, 2
    U = np.linalg.qr(rng.normal(size=(p, r0)))[0]
    V = np.linalg.qr(rng.normal(size=(q, r0)))[0]
    P, Q = np.zeros((r0, r)), np.zeros((r0, r))
    P[:4] = np.linalg.qr(np.array([[1., .2], [.4, 1.], [.3, -.2], [.1, .3]]))[0]
    Q[:3] = np.linalg.qr(np.array([[1., .3], [.2, 1.], [.4, -.2]]))[0]
    C = (U @ P * [3., 1.2]) @ (V @ Q).T
    X = rng.normal(size=(n, p))
    Y = X @ C + .01 * rng.normal(size=(n, q))
    return X, Y, U, V, C


def model(**kwargs):
    options = dict(rank=2, source_rank=7, sparsity=(8, 6),
        margins=Margins(.2, 6., .3, trial_radius=.5),
        calibration=PracticalCalibration(.003, .001, 10., (8, 8)), iterations=15)
    options.update(kwargs)
    return SparseSMART(**options)


def test_exact_fit_predict_and_every_iterate_contract():
    X, Y, U, V, C = example()
    fitted = model().fit(X, Y, source=ExactSource(U, V))
    assert fitted.success_, fitted.message_
    assert fitted.coefficient_.shape == C.shape
    assert fitted.n_iter_ == 15
    assert np.linalg.norm(fitted.coefficient_ - C) < .1
    np.testing.assert_allclose(fitted.predict(X), X @ fitted.coefficient_, atol=1e-12)
    final_objective = np.sum((Y - X @ fitted.coefficient_)**2) / (2 * len(X))
    final_objective += .001 * (np.abs(fitted.state_[fitted.chart_.z_u_slice]).sum()
                              + np.abs(fitted.state_[fitted.chart_.z_v_slice]).sum())
    assert fitted.history_[-1].objective == pytest.approx(final_objective, abs=1e-12)
    for record in fitted.history_:
        assert record.support_u <= 8 and record.support_v <= 8
        assert min(record.anchor_min_u, record.anchor_min_v) >= .01
    assert all(b.objective <= a.objective for a, b in zip(fitted.history_, fitted.history_[1:]))
    assert not fitted.diagnostics_["theorem_certified"]
    np.testing.assert_array_equal(fitted.source_.left, U)


def test_deterministic_repeated_fit_and_T_zero_handoff():
    X, Y, U, V, _ = example()
    m = model(iterations=0).fit(X, Y, source=ExactSource(U, V))
    assert m.success_ and m.n_iter_ == 0
    first = m.coefficient_.copy()
    np.testing.assert_array_equal(m.state_, m.initial_state_)
    m.fit(X, Y, source=ExactSource(U, V))
    np.testing.assert_array_equal(m.coefficient_, first)


def test_n_less_than_source_rank_without_gram_inverse():
    n, r0 = 5, 12
    X = np.concatenate((np.sqrt(n) * np.eye(n), np.zeros((n, r0 - n))), axis=1)
    C = np.zeros((r0, r0))
    C[0, 0] = 2.
    m = SparseSMART(rank=1, source_rank=r0, sparsity=(1, 1), margins=Margins(.1, 5, .2),
        calibration=PracticalCalibration(.02, 0., 1., (0, 0)), iterations=2)
    m.fit(X, X @ C, source=ExactSource(np.eye(r0), np.eye(r0)))
    assert m.success_, m.message_
    np.testing.assert_allclose(m.coefficient_, C, atol=1e-12)


def test_noisy_fit_uses_full_frames_and_leaves_estimated_spans():
    rng = np.random.default_rng(81)
    p, q, r0 = 5, 4, 2
    # Actual source perturbation rotates the observed leading left vector.
    clean = np.zeros((p, q))
    clean[0, 0], clean[1, 1] = 5., 3.
    observed = clean.copy()
    observed[3, 0] = .01
    C = np.zeros((p, q))
    C[0, 0] = 2.
    X = rng.normal(size=(150, p))
    m = SparseSMART(rank=1, source_rank=r0, sparsity=(1, 1), margins=Margins(.1, 5., .2),
        calibration=PracticalCalibration(.001, 0., 3., (p - 1, q - 1)), iterations=12)
    m.fit(X, X @ C, source=NoisySource(observed, noise_std=.001, gap_lower=2.))
    assert m.success_, m.message_
    assert m.source_.left.shape == (p, p) and m.source_.right.shape == (q, q)
    projection = m.source_.leading_left @ m.source_.leading_left.T
    assert np.linalg.norm((np.eye(p) - projection) @ m.coefficient_) > 1e-4
    assert np.linalg.norm(m.coefficient_ - C) < .01


def test_cluster_mode_and_source_accuracy_failure():
    X, Y, U, V, _ = example()
    source = (U * [6., 6., 4., 3., 2., 1., .5]) @ V.T
    m = model(iterations=0).fit(X, Y, source=NoisySource(source, .00001, .5, cluster_size=2))
    assert m.success_, m.message_
    assert m.source_.cluster_size == 2
    m.fit(X, Y, source=NoisySource(source, 1., .01))
    assert m.status_ == "source_accuracy_failed"
    assert not hasattr(m, "coefficient_")
    with pytest.raises(FitFailure):
        m.predict(X)


def test_spectrum_failure_and_raising_policy():
    X, Y, U, V, _ = example()
    m = model(initialization_spectrum="reject").fit(X, np.zeros_like(Y), source=ExactSource(U, V))
    assert m.status_ == "initialization_spectrum_failed"
    with pytest.raises(FitFailure, match="initialization_spectrum_failed"):
        model(raise_on_failure=True, initialization_spectrum="reject").fit(
            X, np.zeros_like(Y), source=ExactSource(U, V))
    with pytest.raises(ValueError):
        model().fit(X, Y[:-1], source=ExactSource(U, V))


def test_partial_fit_requires_explicit_prediction_opt_in():
    X, Y, U, V, _ = example()
    m = model(calibration=PracticalCalibration(.003, .001, 1e-15, (8, 8)),
              max_backtracks=0).fit(X, Y, source=ExactSource(U, V))
    assert m.status_ == "line_search_failed" and hasattr(m, "coefficient_")
    with pytest.raises(FitFailure):
        m.predict(X)
    assert np.isfinite(m.predict(X, allow_partial=True)).all()


def test_prescribed_calibration_is_available_but_not_a_certificate():
    X, Y, U, V, _ = example()
    m = model(calibration=PrescribedCalibration(.00001), iterations=0)
    m.fit(X, Y, source=ExactSource(U, V))
    assert m.success_, m.message_
    assert m.calibration_.mode == "prescribed"
    assert not m.diagnostics_["theorem_certified"]


@pytest.mark.parametrize("name", ["lasso_tol", "orthogonality_tol", "tie_tol"])
@pytest.mark.parametrize("value", [True, np.bool_(True), "large"])
def test_rejects_non_numeric_tolerances(name, value):
    X, Y, U, V, _ = example()
    with pytest.raises(ValueError):
        model(**{name: value}).fit(X, Y, source=ExactSource(U, V))
