"""Practical spectral repair remains separate from the reduced-Lasso result."""

import numpy as np
import pytest

from sparse_smart import ExactSource, Margins, PracticalCalibration, PrescribedCalibration, SparseSMART


def diagonal_problem(values=(3., .2, 0., 0.)):
    X = 2. * np.eye(4)
    return X, X @ np.diag(values), ExactSource(np.eye(4), np.eye(4))


def estimator(**options):
    parameters = dict(rank=3, source_rank=4, sparsity=(3, 3),
        margins=Margins(.2, 8., .3),
        calibration=PracticalCalibration(.1, 0., 10., (3, 3)),
        iterations=0, refinement_solver="chart")
    parameters.update(options)
    return SparseSMART(**parameters)


def test_projected_initialization_repairs_spectrum_without_changing_lasso_result():
    X, Y, source = diagonal_problem()
    fitted = estimator(initialization_spectrum="projected").fit(X, Y, source=source)
    assert fitted.success_, fitted.message_
    diagnostics = fitted.diagnostics_
    assert diagnostics["initialization_spectrum_requested"] == "projected"
    assert diagnostics["initialization_spectrum"] == "projected"
    assert diagnostics["initialization_spectrum_repaired"]
    np.testing.assert_allclose(fitted.initialization_.d, [2.9, .1, 0.], atol=1e-14)
    np.testing.assert_allclose(fitted.initialization_.coefficient, np.diag([2.9, .1, 0., 0.]), atol=1e-14)
    np.testing.assert_allclose(diagnostics["initialization_singular_values_original"], [2.9, .1, 0.], atol=1e-14)
    np.testing.assert_allclose(diagnostics["initialization_singular_values_projected"], [2.9, .5, .2], atol=1e-14)
    assert diagnostics["initialization_spectrum_correction_norm"] == pytest.approx(np.sqrt(.2))
    P, d, Q = fitted.chart_.reconstruct(fitted.initial_state_)
    np.testing.assert_allclose(P, fitted.initialization_.P, atol=1e-14)
    np.testing.assert_allclose(Q, fitted.initialization_.Q, atol=1e-14)
    np.testing.assert_allclose(d, [2.9, .5, .2], atol=1e-14)
    np.testing.assert_allclose(fitted.coefficient_, np.diag([2.9, .5, .2, 0.]), atol=1e-14)
    assert np.all(d[:-1] - d[1:] >= fitted.margins.gap)
    assert d[-1] >= fitted.margins.d_lower
    assert not diagnostics["theorem_certified"]


def test_reject_policy_preserves_spectral_failure_and_original_values():
    X, Y, source = diagonal_problem()
    fitted = estimator(initialization_spectrum="reject").fit(X, Y, source=source)
    assert fitted.status_ == "initialization_spectrum_failed"
    assert not fitted.success_ and not hasattr(fitted, "coefficient_")
    assert fitted.diagnostics_["initialization_spectrum"] == "reject"
    assert not fitted.diagnostics_["initialization_spectrum_repaired"]
    assert fitted.diagnostics_["initialization_spectrum_correction_norm"] == 0.
    np.testing.assert_array_equal(fitted.diagnostics_["initialization_singular_values_original"],
                                  fitted.initialization_.d)


@pytest.mark.parametrize("policy", ["auto", "projected", "reject"])
def test_well_conditioned_initializer_is_exact_noop(policy):
    X, Y, source = diagonal_problem((3., 2., 1., 0.))
    fitted = estimator(initialization_spectrum=policy).fit(X, Y, source=source)
    assert fitted.success_, fitted.message_
    assert not fitted.diagnostics_["initialization_spectrum_repaired"]
    assert fitted.diagnostics_["initialization_spectrum_correction_norm"] == 0.
    np.testing.assert_array_equal(fitted.state_[fitted.chart_.d_slice], fitted.initialization_.d)
    np.testing.assert_array_equal(fitted.diagnostics_["initialization_singular_values_projected"],
                                  fitted.initialization_.d)


def test_auto_policy_resolves_projected_for_practical_calibration():
    X, Y, source = diagonal_problem()
    fitted = estimator().fit(X, Y, source=source)
    assert fitted.success_, fitted.message_
    assert fitted.initialization_spectrum_ == "projected"
    assert fitted.diagnostics_["initialization_spectrum_requested"] == "auto"
    assert fitted.diagnostics_["initialization_spectrum_repaired"]


def test_auto_policy_retains_rejection_for_prescribed_calibration():
    X, Y, source = diagonal_problem()
    fitted = estimator(calibration=PrescribedCalibration(1e-10)).fit(X, Y, source=source)
    assert fitted.initialization_spectrum_ == "reject"
    assert fitted.status_ == "initialization_spectrum_failed"
    assert fitted.diagnostics_["initialization_spectrum_requested"] == "auto"
    assert not fitted.diagnostics_["initialization_spectrum_repaired"]
    assert not fitted.diagnostics_["practical_extension"]


def test_prescribed_calibration_requires_explicit_projection_override():
    X, Y, source = diagonal_problem()
    fitted = estimator(calibration=PrescribedCalibration(1e-10),
                       initialization_spectrum="projected").fit(X, Y, source=source)
    assert fitted.success_, fitted.message_
    assert fitted.initialization_spectrum_ == "projected"
    assert fitted.diagnostics_["initialization_spectrum_repaired"]
    assert fitted.diagnostics_["practical_extension"]


@pytest.mark.parametrize("name", ["initialization_spectrum", "refinement_solver"])
@pytest.mark.parametrize("value", [None, True, 1, "invalid", [], np.array(["projected"])])
def test_invalid_policy_is_rejected_before_fit(name, value):
    with pytest.raises(ValueError, match=name):
        estimator(**{name: value})


@pytest.mark.parametrize("name", ["initialization_spectrum", "refinement_solver"])
def test_mutated_policy_is_validated_again_before_initialization(name):
    X, Y, source = diagonal_problem()
    fitted = estimator()
    setattr(fitted, name, "invalid")
    with pytest.raises(ValueError, match=name):
        fitted.fit(X, Y, source=source)
