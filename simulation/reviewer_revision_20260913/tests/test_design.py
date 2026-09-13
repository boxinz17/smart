"""Tiny algebraic fixtures only; production simulations belong on Slurm."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

_spec = importlib.util.spec_from_file_location(
    "revision_design", Path(__file__).parents[1] / "design.py")
design = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(design)


def tiny(**updates):
    result = dict(p=20, q=15, target_rank=4, latent_source_rank=6,
                  source_rank=6, n_train=9, n_validation=11, n_test=13)
    result.update(updates)
    return result


@pytest.mark.parametrize("updates", [dict(), dict(diffuse_strength=1),
    dict(left_angle_deg=30, right_angle_deg=60), dict(target_specific=2),
    dict(left_angle_deg=90, right_angle_deg=90)])
def test_rotations_preserve_orthogonality_rank_and_spectrum(updates):
    generated = design.generate_case(tiny(**updates), 781)
    evaluation = generated["evaluation"]
    for name in ("U_star", "V_star", "U0", "V0"):
        frame = evaluation[name]
        np.testing.assert_allclose(frame.T @ frame, np.eye(frame.shape[1]), atol=2e-14)
    values = np.linalg.svd(evaluation["C_star"], compute_uv=False)
    np.testing.assert_allclose(values[:4], np.linspace(5, 3, 4), atol=2e-14)
    assert np.linalg.matrix_rank(evaluation["C_star"]) == 4


def test_unrelated_endpoint_is_orthogonal_to_both_source_spans():
    data = design.generate_case(tiny(left_angle_deg=90, right_angle_deg=90), 77)
    truth = data["evaluation"]
    np.testing.assert_allclose(truth["U0"].T @ truth["C_star"], 0, atol=2e-14)
    np.testing.assert_allclose(truth["C_star"] @ truth["V0"], 0, atol=2e-14)
    np.testing.assert_allclose(data["metadata"]["containment_joint"],
                               np.linalg.norm(truth["C_star"]))


def test_independent_sides_have_declared_angles():
    data = design.generate_case(tiny(left_angle_deg=20, right_angle_deg=60), 66)
    evaluation = data["evaluation"]
    for source, target, angle in (("U0", "U_star", 20), ("V0", "V_star", 60)):
        cosines = np.linalg.svd(evaluation[source].T @ evaluation[target],
                                compute_uv=False)
        np.testing.assert_allclose(cosines, np.cos(np.deg2rad(angle)), atol=2e-14)


def test_diffusion_changes_coordinate_concentration_but_keeps_containment():
    base = design.generate_case(tiny(), 114)
    diffuse = design.generate_case(tiny(diffuse_strength=1), 114)
    for side in ("left", "right"):
        assert base["metadata"][f"alignment_{side}"]["tail_after_top1"] == 0
        assert diffuse["metadata"][f"alignment_{side}"]["tail_after_top1"] > 0.5
        assert diffuse["metadata"][f"containment_{side}"] < 1e-13
    np.testing.assert_allclose(base["metadata"]["target_coefficient_energy"],
                               diffuse["metadata"]["target_coefficient_energy"])


def test_internal_gaps_preserve_total_energy_and_boundary_exactly():
    original = np.linspace(10, 1, 10)
    for factor in (1, 0.2, 0.02, 0):
        spectrum = design._spectrum(original, factor)
        np.testing.assert_allclose(spectrum @ spectrum, original @ original, rtol=2e-15)
        assert spectrum[-1] == original[-1]
        assert np.all(np.diff(spectrum) <= 0)
    tied = design._spectrum(original, 0)
    assert tied[0] == tied[1]
    assert tied[2] == tied[3]
    assert tied[-2] > tied[-1]


def test_boundary_gap_sweep_preserves_total_energy_and_nonzero_rank():
    original = np.linspace(5, 3, 5)
    spectrum = design._spectrum(original, boundary_factor=0.02)
    np.testing.assert_allclose(spectrum @ spectrum, original @ original, rtol=2e-15)
    assert spectrum[-1] == original[-1] * 0.02
    assert np.all(spectrum > 0)


def test_common_random_numbers_and_independent_splits():
    first = design.generate_case(tiny(), 999)
    changed = design.generate_case(tiny(diffuse_strength=0.2, left_angle_deg=30), 999)
    for key in ("X", "X_validation", "C0"):
        np.testing.assert_array_equal(first["fit_data"][key], changed["fit_data"][key])
    for x_key, y_key in (("X", "Y"), ("X_validation", "Y_validation")):
        errors = [data["fit_data"][y_key] - data["fit_data"][x_key] @
                  data["evaluation"]["C_star"] for data in (first, changed)]
        np.testing.assert_allclose(*errors, atol=4e-15)
    assert not np.array_equal(first["fit_data"]["X"][:9],
                              first["fit_data"]["X_validation"][:9])
    assert not np.array_equal(first["fit_data"]["X"][:9],
                              first["evaluation"]["X_test"][:9])


def test_truth_is_separate_from_fit_payload_and_source_ranks_are_distinct():
    generated = design.generate_case(tiny(source_rank=10), 332)
    fit = generated["fit_data"]
    assert set(fit) == {"X", "Y", "X_validation", "Y_validation", "C0",
                        "source_rank", "source_X", "source_Y"}
    assert fit["source_rank"] == 10
    assert generated["metadata"]["latent_source_rank"] == 6
    assert generated["metadata"]["source_numerical_rank"] == 15
    assert generated["metadata"]["fitted_source_rank"] is None


def test_fitted_source_uses_raw_observations_with_no_target_tuning():
    original = design.generate_case(tiny(source_mode="fitted", n_source=40), 737)
    changed = design.generate_case(tiny(source_mode="fitted", n_source=40,
        target_internal_gap=0, left_angle_deg=60, n_train=12), 737)
    for key in ("source_X", "source_Y", "C0"):
        np.testing.assert_array_equal(original["fit_data"][key], changed["fit_data"][key])
    assert len(original["fit_data"]["source_X"]) == 40
    assert original["metadata"]["source_fit"]["n_refit"] == 40
    assert original["metadata"]["fitted_source_rank"] in (1, 3, 5, 10, 15)
    assert original["metadata"]["source_fit"]["n_fit"] + \
           original["metadata"]["source_fit"]["n_validation"] == 40


def test_full_rank_ridge_matches_normal_equations():
    rng = np.random.default_rng(111)
    X, Y = rng.normal(size=(11, 7)), rng.normal(size=(11, 5))
    alpha = 0.1
    actual = next(design._ridge_rrr_path(X, Y, [alpha], [5]))[2]
    expected = np.linalg.solve(X.T @ X + len(X) * alpha * np.eye(7), X.T @ Y)
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-14)


def test_population_loss_and_coefficient_close_regime():
    data = design.generate_case(tiny(relationship="coefficient_close"), 115)
    evaluation = data["evaluation"]
    metrics = design.error_metrics(evaluation["C_star"], evaluation)
    assert metrics["coefficient_rmse"] == metrics["prediction_excess"] == 0
    assert data["metadata"]["source_target_distance"] < 0.15 * np.linalg.norm(evaluation["C_star"])
    zero_metrics = design.error_metrics(np.zeros_like(evaluation["C_star"]), evaluation)
    assert zero_metrics["prediction_excess"] == data["metadata"]["target_population_signal"]


def test_manifest_has_unique_keys_and_explicit_operational_dimensions():
    manifest = design.case_manifest()
    assert len({case["case_id"] for case in manifest}) == len(manifest)
    assert {case["n_train"] for case in manifest} == {40, 80, 200}
    for case in manifest:
        assert {"case_id", "family", "level", "n_train", "p", "q",
                "target_rank", "source_rank", "n_validation"} <= set(case)
        assert case["n_validation"] == 200
    assert {case["level"] for case in manifest if case["family"] == "containment"} == {15, 30, 60, 90}
