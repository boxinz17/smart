"""Independent checks of declared calibration and failure boundaries."""

import math

import pytest

from sparse_smart.calibration import (
    CalibrationError,
    Margins,
    PracticalCalibration,
    PrescribedCalibration,
    SourceAccuracyError,
    resolve_calibration,
)


@pytest.fixture
def inputs():
    return dict(n=100_000, p=8, q=7, rank=2, source_rank=4,
                sparsity=(4, 5), margins=Margins(0.1, 5.0, 0.1))


def test_exact_prescription_matches_displayed_formulas(inputs):
    result = resolve_calibration(PrescribedCalibration(0.2), **inputs)
    diag = result.diagnostics
    # Use the actual small combinatorial count to independently check the
    # implementation's log-gamma evaluation of the anchor multiplicity.
    delta_w = 0.05 / 32
    h_anchor = math.log(math.comb(4, 2) ** 2 * 2 ** 2)
    ell = math.log(64 * math.exp(h_anchor) * (4 + 4 + 1) / delta_w)
    expected_init = 8 * 0.2 * math.sqrt(math.log(16 * 4 ** 2 / delta_w) / 100_000)
    assert result.init_penalty == pytest.approx(expected_init)
    assert diag["H_anc"] == pytest.approx(h_anchor)
    assert diag["ell"] == pytest.approx(ell)
    assert diag["lambda_b"] == pytest.approx(0.2 * math.sqrt(ell / 100_000))
    assert diag["coordinate_counts"] == (4, 4)
    assert diag["clean_complement_budgets"] == (2, 3)
    assert diag["reference_support"] == (2, 3)
    assert diag["correction_budgets"] == (0, 0)
    assert diag["eta_0"] == 0
    assert diag["epsilon"] == (0, 0)
    assert diag["beta"] == 0
    assert result.support_limits == (4, 4)
    assert diag["gamma"] == 0
    assert diag["primitive_checks"]["thresholding"]
    assert not diag["theorem_certified"]
    assert not diag["all_primitive_checks"]
    expected_g = 2 ** 30 * (2 + 2 + 5 + 10 + 100 + 10) ** 30
    expected_hm = 4 * math.log(5 * math.e * 4 / 4) + math.log(128 / delta_w)
    expected_kappa = 4 * (1 + math.sqrt(2 * expected_hm / 100_000)) ** 2
    assert diag["G"] == pytest.approx(expected_g)
    assert diag["kappa_upper"] == pytest.approx(expected_kappa)
    assert result.penalty == pytest.approx(8 * 0.2 * expected_g
                                           * math.sqrt(expected_kappa * ell / 100_000))
    assert math.isfinite(result.step_size_inverse)


def test_noisy_source_calibration_balances_and_bounds_tail(inputs):
    sigma, noise, gap = 0.2, 1e-8, 1.0
    result = resolve_calibration(PrescribedCalibration(sigma, support_enlargement=2),
                                 **inputs, source_noise=noise, source_gap=gap)
    diag = result.diagnostics
    eta = 4 * noise * math.sqrt((8 + 7) * math.log(9) + math.log(64 / (0.05 / 32)))
    expected_eps = 8 * math.sqrt(4) * eta / gap
    assert diag["eta_0"] == pytest.approx(eta)
    assert diag["epsilon"] == pytest.approx((expected_eps, expected_eps))
    assert diag["working_dimensions"] == (8, 7)
    assert diag["coordinate_counts"] == (12, 10)
    expected_errors = 5 * expected_eps
    assert diag["E"] == pytest.approx((expected_errors, expected_errors))
    assert diag["T"] == pytest.approx((math.sqrt(12) * expected_errors,
                                       math.sqrt(10) * expected_errors))
    assert diag["correction_budgets"] == (1, 1)
    assert diag["reference_support"] == (3, 4)
    assert result.support_limits == (6, 8)
    assert diag["beta_blocks"] == pytest.approx((expected_errors, expected_errors))


def test_exact_empty_complements_do_not_divide_by_source_gap(inputs):
    inputs.update(rank=4, sparsity=(4, 4))
    result = resolve_calibration(PrescribedCalibration(0.1), **inputs)
    assert result.support_limits == (0, 0)
    assert result.penalty > 0
    assert result.init_penalty > 0
    assert result.diagnostics["beta"] == 0
    assert result.diagnostics["correction_budgets"] == (0, 0)


@pytest.mark.parametrize("config", [PrescribedCalibration(1.0),
    PracticalCalibration(0.1, 0.1, 10.0, (2, 2))])
def test_source_accuracy_rejection_applies_to_both_modes(inputs, config):
    with pytest.raises(SourceAccuracyError, match="source accuracy fails"):
        resolve_calibration(config, **inputs, source_noise=0.1, source_gap=1.0)


def test_practical_source_accuracy_opt_in_preserves_supplied_bounds(inputs):
    config = PracticalCalibration(0.01, 0.02, 5.0, (3, 4),
                                  enforce_source_accuracy=False)
    result = resolve_calibration(config, **inputs, source_noise=0.1, source_gap=1.0)
    diag = result.diagnostics
    eta = 4 * 0.1 * math.sqrt(15 * math.log(9) + math.log(64 / (0.05 / 32)))
    assert diag["eta_0"] == pytest.approx(eta)
    assert diag["source_noise"] == 0.1
    assert diag["source_gap"] == 1.0
    assert diag["epsilon"] == pytest.approx((16 * eta, 16 * eta))
    assert not diag["source_accuracy_passed"]
    assert not diag["source_accuracy_enforced"]
    assert diag["source_accuracy_bypassed"]
    assert not diag["theorem_certified"]
    assert "explicitly bypassed" in diag["calibration_note"]
    assert result.mode == "practical"
    assert result.support_limits == (3, 4)
    assert (result.init_penalty, result.penalty, result.step_size_inverse) == (0.01, 0.02, 5.0)


@pytest.mark.parametrize("noise", [0.0, 1e-8])
@pytest.mark.parametrize("enforce", [False, True])
def test_passing_source_accuracy_distinguishes_enforcement_from_bypass(inputs, noise, enforce):
    config = PracticalCalibration(0.01, 0.02, 5.0, (3, 4),
                                  enforce_source_accuracy=enforce)
    diag = resolve_calibration(config, **inputs, source_noise=noise,
                               source_gap=1.0).diagnostics
    assert diag["source_accuracy_passed"]
    assert diag["source_accuracy_enforced"] is enforce
    assert not diag["source_accuracy_bypassed"]


def test_prescribed_source_accuracy_enforcement_cannot_be_disabled(inputs):
    with pytest.raises(TypeError, match="enforce_source_accuracy"):
        PrescribedCalibration(0.2, enforce_source_accuracy=False)
    diag = resolve_calibration(PrescribedCalibration(0.2), **inputs,
                               source_noise=1e-8, source_gap=1.0).diagnostics
    assert diag["source_accuracy_enforced"]
    assert diag["source_accuracy_passed"]
    assert not diag["source_accuracy_bypassed"]


@pytest.mark.parametrize("source_args", [
    {"source_noise": 0.1},
    {"source_noise": -0.1, "source_gap": 1.0},
    {"source_noise": float("nan"), "source_gap": 1.0},
    {"source_noise": 0.1, "source_gap": 0.0},
    {"source_noise": 0.1, "source_gap": float("inf")},
])
def test_empirical_source_option_still_rejects_malformed_bounds(inputs, source_args):
    config = PracticalCalibration(0.01, 0.02, 5.0, (3, 4),
                                  enforce_source_accuracy=False)
    with pytest.raises(CalibrationError):
        resolve_calibration(config, **inputs, **source_args)


def test_practical_values_are_preserved_and_source_bounds_reported(inputs):
    config = PracticalCalibration(0.01, 0.02, 5.0, (3, 4), delta=0.1)
    result = resolve_calibration(config, **inputs, source_noise=1e-8, source_gap=1)
    assert (result.init_penalty, result.penalty, result.step_size_inverse) == (0.01, 0.02, 5.0)
    assert result.support_limits == (3, 4)
    assert result.mode == "practical"
    assert result.diagnostics["eta_0"] > 0
    assert result.diagnostics["delta_w"] == 0.1 / 32
    assert "G" not in result.diagnostics
    assert not result.diagnostics["prescribed_calibration"]


def test_clusters_inflate_all_counts_and_saturate(inputs):
    inputs.update(sparsity=(2, 2))
    result = resolve_calibration(PrescribedCalibration(0.1), **inputs, cluster_size=4)
    assert result.diagnostics["effective_sparsity"] == (8, 8)
    assert result.diagnostics["row_bounds"] == (4, 4)
    assert result.diagnostics["clean_complement_budgets"] == (4, 4)
    assert result.support_limits == (4, 4)


def test_universal_anchor_check_is_reported_without_claiming_stability(inputs):
    inputs.update(p=60, q=60, source_rank=50, sparsity=(50, 50))
    result = resolve_calibration(PracticalCalibration(0.1, 0.1, 1.0, (3, 3)), **inputs)
    assert not result.diagnostics["universal_anchor_condition"]
    assert "not certified" in result.diagnostics["anchor_assumption_note"]


def test_huge_enlargement_is_capped_before_integer_conversion(inputs):
    result = resolve_calibration(PrescribedCalibration(0.2), **inputs)
    assert result.diagnostics["log_support_enlargement"] > 100
    assert result.support_limits == (4, 4)
    assert "saturated-support" in result.diagnostics["support_enlargement_rule"]
    assert result.step_size_inverse <= result.diagnostics["L_saturated_upper"]


def test_nearly_one_user_enlargement_still_increases_nonempty_support(inputs):
    result = resolve_calibration(PrescribedCalibration(0.2,
        support_enlargement=math.nextafter(1.0, 2.0)), **inputs)
    assert result.support_limits == (3, 4)
    assert math.isfinite(result.diagnostics["gamma"])
    assert not result.diagnostics["primitive_checks"]["thresholding"]


@pytest.mark.parametrize("enlargement,sparsity,expected", [
    (3.0, 2, 3),
    (math.nextafter(3.0, 2.0), 2, 3),
    (math.nextafter(3.0, 4.0), 2, 4),
    (1.5, 3, 3),
    (math.nextafter(1.5, 1.0), 3, 3),
    (math.nextafter(1.5, 2.0), 3, 4),
    (4.0, 2, 4),
    (5.0, 2, 4),
    (1e308, 2, 4),
    (3.0, 1, 0),
])
def test_explicit_enlargement_preserves_exact_ceil_and_saturates(
        enlargement, sparsity, expected):
    result = resolve_calibration(
        PrescribedCalibration(0.1, support_enlargement=enlargement),
        n=1000, p=5, q=5, rank=1, source_rank=5,
        sparsity=(sparsity, sparsity), margins=Margins(0.1, 5.0, 0.1),
    )
    assert result.support_limits == (expected, expected)
    assert result.diagnostics["support_enlargement"] == enlargement


def test_unsafe_geometry_constants_fail_explicitly_but_practical_mode_works(inputs):
    inputs["margins"] = Margins(1e-20, 5.0, 0.1)
    with pytest.raises(CalibrationError, match="overflow"):
        resolve_calibration(PrescribedCalibration(0.2), **inputs)
    assert resolve_calibration(PracticalCalibration(0.1, 0.1, 1, (2, 2)),
                               **inputs).mode == "practical"


@pytest.mark.parametrize("field,value", [
    ("n", True), ("n", 0), ("p", 3), ("rank", 5), ("source_rank", 9),
    ("sparsity", (1, 4)), ("sparsity", (9, 4)), ("sparsity", (True, 4)),
    ("cluster_size", 0), ("cluster_size", 5), ("cluster_size", True),
    ("source_noise", float("nan")), ("source_noise", -1),
    ("source_gap", 0),
])
def test_invalid_inputs_fail_before_computation(inputs, field, value):
    inputs[field] = value
    with pytest.raises(CalibrationError):
        resolve_calibration(PrescribedCalibration(0.2), **inputs)


@pytest.mark.parametrize("constructor", [
    lambda: PracticalCalibration(0, 0, 1, (0, 0)),
    lambda: PracticalCalibration(1, -1, 1, (0, 0)),
    lambda: PracticalCalibration(1, 1, 0, (0, 0)),
    lambda: PracticalCalibration(1, 1, 1, (False, 0)),
    lambda: PracticalCalibration(1, 1, 1, (0, 0), delta=0.25),
    lambda: PracticalCalibration(1, 1, 1, (0, 0), enforce_source_accuracy=0),
    lambda: PracticalCalibration(1, 1, 1, (0, 0), enforce_source_accuracy="false"),
    lambda: PrescribedCalibration(float("inf")),
    lambda: PrescribedCalibration(-1),
    lambda: PrescribedCalibration(0),
    lambda: PrescribedCalibration(1, delta=True),
    lambda: PrescribedCalibration(1, support_enlargement=1),
    lambda: PrescribedCalibration(1, design_lower=0),
    lambda: Margins(1, 4, 1),
    lambda: Margins(0.1, 5, 0),
    lambda: Margins(0.1, 5, 1, anchor_min=1 / 16),
    lambda: Margins(0.1, 5, 1, qr_threshold=1),
])
def test_invalid_configuration_scalars(constructor):
    with pytest.raises(CalibrationError):
        constructor()


def test_practical_budgets_cannot_exceed_working_coordinates(inputs):
    with pytest.raises(CalibrationError, match="complement sizes"):
        resolve_calibration(PracticalCalibration(0.1, 0.1, 1.0, (5, 4)), **inputs)


def test_positive_source_noise_requires_gap(inputs):
    with pytest.raises(CalibrationError, match="requires a positive source_gap"):
        resolve_calibration(PrescribedCalibration(0.2), **inputs, source_noise=1e-8)
