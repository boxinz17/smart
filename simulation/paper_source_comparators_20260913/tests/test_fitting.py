"""Tiny deterministic algebraic fixtures; no paper-data or Monte Carlo fits."""

import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

SIMULATION = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SIMULATION))
sys.path.insert(0, str(SIMULATION.parent / "sparse-smart" / "src"))
sys.path.insert(0, str(SIMULATION.parent / "sparse-smart-v2" / "src"))
from paper_source_comparators_20260913 import fitting as f


def fixture(p=4, q=3):
    # Distinct design scales make coefficient-SVD and design-whitened RRR differ.
    x = np.diag(np.arange(1, p + 1, dtype=float))
    c = np.zeros((p, q))
    for j in range(min(p, q)):
        c[j, j] = .7 / (j + 1)
    data = dict(X=x, Y=x @ c, C0=np.zeros((p, q)),
                X_validation=np.eye(p), Y_validation=c)
    return data, dict(left=np.eye(p), right=np.eye(q))


def small_configuration(**overrides):
    config = f.default_configuration()
    config.update(ridge_multipliers=[0., .1], source_dimensions=[1, 3],
                  mixture_alphas=[0., .5, 1.], nuclear_penalty_fractions=[0., .1, 1.],
                  initializer_penalties=[.03, .3])
    config.update(overrides)
    return config


def test_default_grids_are_explicit_fresh_and_training_scaled():
    a, b = f.default_configuration(), f.default_configuration()
    assert a["ridge_multipliers"] == [0, .0001, .001, .01, .1, 1, 10, 100]
    assert a["source_dimensions"] == [1, 3, 5, 7, 10, 11, 15, 20]
    assert a["initializer_penalties"] == [.003, .03, .1, .3, 1, 3]
    a["ridge_multipliers"].append(99)
    assert 99 not in b["ridge_multipliers"]
    data, frames = fixture()
    record, _ = f.fit_method(data, frames, "target_ridge_rrr", 2, small_configuration())
    scale = 16 / 4
    assert record["training_scales"]["ridge"] == pytest.approx(scale)
    assert [r["parameters"]["ridge"] for r in record["candidate_results"]] == [0, .1 * scale]


@pytest.mark.parametrize("name", ["C_star", "truth", "source_truth", "unused"])
def test_truth_and_extra_data_are_rejected_before_any_fit(monkeypatch, name):
    data, frames = fixture()
    data[name] = np.ones((4, 3))
    monkeypatch.setattr(f, "fit_competitor", lambda *a, **k: pytest.fail("fit accessed forbidden input"))
    with pytest.raises(ValueError, match="truth or extra"):
        f.fit_method(data, frames, "target_rrr", 2)


@pytest.mark.parametrize("method", ["target_rrr", "target_ridge_rrr"])
def test_target_methods_ignore_even_malformed_source_objects(method):
    data, frames = fixture()
    expected, coefficient = f.fit_method(data, frames, method, 2, small_configuration())
    data["C0"] = object()
    actual, other = f.fit_method(data, object(), method, 2, small_configuration())
    np.testing.assert_array_equal(coefficient, other)
    assert actual["validation_mse"] == expected["validation_mse"]
    assert actual["information_regime"] == "target_only"


def test_validation_selection_no_refit_and_first_tie(monkeypatch):
    data, frames = fixture()
    calls = []
    coefficient = np.ones((4, 3)) * .01
    def fit(x, y, method, **kwargs):
        calls.append((x.shape, kwargs))
        return SimpleNamespace(coefficient=coefficient, diagnostics={"certified": True, "converged": True})
    monkeypatch.setattr(f, "fit_competitor", fit)
    record, chosen = f.fit_method(data, frames, "target_ridge_rrr", 2, small_configuration())
    assert len(calls) == 2
    assert all(shape == data["X"].shape for shape, _ in calls)
    assert record["selected_index"] == 0 and not record["refit"]
    assert not chosen.flags.writeable
    assert record["validation_mse"] == pytest.approx(np.mean((data["Y_validation"] - chosen) ** 2))
    assert record["validation_mse_recomputed"] == record["validation_mse"]
    assert record["boundary_indicators"]["ridge_multiplier"]["at_lower"]


def test_selected_candidate_is_validation_minimum_not_training_optimum():
    data, frames = fixture()
    data["Y_validation"] = np.zeros_like(data["Y_validation"])
    record, _ = f.fit_method(data, frames, "target_ridge_rrr", 2,
                             small_configuration(ridge_multipliers=[0., .1, 100.]))
    assert record["selected_index"] == 2
    assert record["boundary_indicators"]["ridge_multiplier"]["at_upper"]
    eligible = [a["validation_mse"] for a in record["candidate_results"] if a["status"] == "eligible"]
    assert record["validation_mse"] == min(eligible)


@pytest.mark.parametrize("method", ["source_subspace_rrr", "source_subspace_ridge_rrr"])
def test_source_dimension_below_rank_is_valid_and_full_endpoint_matches_target(method):
    data, frames = fixture()
    config = small_configuration(ridge_multipliers=[0.])
    record, coefficient = f.fit_method(data, frames, method, 3, config)
    attempts = record["candidate_results"]
    assert [a["parameters"]["source_rank"] for a in attempts] == [[1, 1], [3, 3], [4, 3]]
    assert all(a["status"] == "eligible" for a in attempts)
    assert attempts[0]["effective_rank"] <= 1
    assert all(a["effective_rank"] <= 3 for a in attempts)
    assert attempts[-1]["diagnostics"]["includes_observed_null_completion"]
    _, target = f.fit_method(data, None, "target_rrr", 3, config)
    # Force only the full rectangular endpoint to inspect its coefficient.
    _, full = f.fit_method(data, frames, method, 3,
                           small_configuration(source_dimensions=[], ridge_multipliers=[0.]))
    np.testing.assert_allclose(full, target, atol=1e-13)


def test_nonorthonormal_or_incomplete_source_frames_are_input_errors():
    data, frames = fixture()
    frames["left"] = frames["left"][:, :3]
    with pytest.raises(ValueError, match="full observed"):
        f.fit_method(data, frames, "source_subspace_rrr", 2)
    frames["left"] = 2 * np.eye(4)
    with pytest.raises(ValueError, match="orthonormal"):
        f.fit_method(data, frames, "initializer_only", 2)


def test_every_failure_and_uncertified_attempt_is_preserved(monkeypatch):
    data, frames = fixture()
    counter = iter(range(3))
    def fit(*args, **kwargs):
        i = next(counter)
        if i == 0:
            raise np.linalg.LinAlgError("declared failure")
        return SimpleNamespace(coefficient=np.zeros((4, 3)),
                               diagnostics={"certified": i == 2, "converged": i == 2, "duality_gap": .2})
    monkeypatch.setattr(f, "fit_competitor", fit)
    record, coefficient = f.fit_method(data, frames, "target_ridge_rrr", 2,
                                        small_configuration(ridge_multipliers=[0., 1., 2.]))
    assert [r["status"] for r in record["candidate_results"]] == ["failed", "uncertified", "eligible"]
    assert record["selected_index"] == 2 and record["status"] == "success"
    assert (record["n_candidates"], record["n_eligible"], record["n_failed"]) == (3, 1, 2)
    assert record["candidate_results"][1]["diagnostics"]["duality_gap"] == .2
    json.dumps(record, allow_nan=False)


def test_all_uncertified_has_no_fallback(monkeypatch):
    data, frames = fixture()
    monkeypatch.setattr(f, "fit_competitor", lambda *a, **k: SimpleNamespace(
        coefficient=np.zeros((4, 3)), diagnostics={"certified": False, "converged": False}))
    record, coefficient = f.fit_method(data, frames, "target_rrr", 2)
    assert record["status"] == "no_eligible_candidate" and coefficient is None
    assert record["selected_index"] is None and record["n_eligible"] == 0


def test_rank_violation_is_rejected_even_if_solver_claims_certification(monkeypatch):
    data, frames = fixture()
    monkeypatch.setattr(f, "fit_competitor", lambda *a, **k: SimpleNamespace(
        coefficient=np.eye(4, 3), diagnostics={"certified": True, "converged": True}))
    record, coefficient = f.fit_method(data, frames, "target_rrr", 1)
    assert coefficient is None and record["candidate_results"][0]["status"] == "failed"
    assert "rank bound" in record["candidate_results"][0]["error"]


def test_selection_verification_mismatch_returns_no_coefficient(monkeypatch):
    data, frames = fixture()
    values = iter([1., 2.])
    monkeypatch.setattr(f, "_validation_mse", lambda *a: next(values))
    record, coefficient = f.fit_method(data, frames, "target_rrr", 2)
    assert coefficient is None and record["status"] == "selection_verification_failed"
    assert record["validation_mse"] == 1. and record["validation_mse_recomputed"] == 2.


def test_ridge_to_source_positive_grid_preserves_unobserved_source_component():
    data, frames = fixture()
    data["X"] = data["X"][:2]
    data["Y"] = data["Y"][:2]
    data["C0"][3, 1] = 5.
    record, coefficient = f.fit_method(data, frames, "ridge_to_source", None,
                                        small_configuration(ridge_multipliers=[0., .1, 1.]))
    assert len(record["candidate_results"]) == 2
    assert all(a["parameters"]["ridge"] > 0 for a in record["candidate_results"])
    assert coefficient[3, 1] == pytest.approx(5.)
    assert record["rank_constraint_scope"] == "none"
    assert record["requested_rank"] is None


def test_zero_design_source_ridge_preserves_declared_failures():
    data, frames = fixture()
    data["X"][:] = 0
    record, coefficient = f.fit_method(data, frames, "ridge_to_source", None, small_configuration())
    assert coefficient is None and record["n_failed"] == 1
    assert "zero training scale" in record["candidate_results"][0]["error"]


def test_mixture_joint_grid_endpoints_and_rank_scope():
    data, frames = fixture()
    data["C0"] = np.eye(4, 3)
    data["Y_validation"] = data["C0"].copy()
    record, coefficient = f.fit_method(data, frames, "source_target_mixture", 1, small_configuration())
    assert record["n_candidates"] == 6
    assert record["selected_index"] == 2  # first alpha=1 endpoint
    assert record["effective_rank"] == 3
    assert record["rank_constraint_scope"] == "target_component_only"
    np.testing.assert_array_equal(coefficient, data["C0"])
    assert all(a["diagnostics"]["target_component_certificate"]["certified"]
               for a in record["candidate_results"])


def test_nuclear_training_scale_zero_correction_and_dual_certificate():
    data, frames = fixture()
    data["C0"][3, 1] = .5
    data["X"] = data["X"][:3]
    data["Y"] = data["Y"][:3]
    config = small_configuration(nuclear_penalty_fractions=[0., 1.])
    record, coefficient = f.fit_method(data, frames, "nuclear_contrast", None, config)
    expected_scale = np.linalg.norm(data["X"].T @ (data["Y"] - data["X"] @ data["C0"]) / 3, 2)
    assert record["training_scales"]["nuclear_contrast"] == pytest.approx(expected_scale)
    first, last = record["candidate_results"]
    assert first["diagnostics"]["algorithm"] == "minimum_norm_unpenalized_source_correction"
    assert last["diagnostics"]["certified"] and last["diagnostics"]["duality_gap"] <= 1e-7
    assert first["status"] == "eligible" and last["status"] == "eligible"
    assert coefficient[3, 1] == pytest.approx(.5)


def test_initializer_uses_truncated_factors_and_observed_prefixes(monkeypatch):
    data, frames = fixture(p=12, q=11)
    frames["left"] = np.eye(12)[:, ::-1]
    frames["right"] = np.eye(11)[:, ::-1]
    calls = []
    def lasso(x, y, rank, penalty, **kwargs):
        calls.append((x.copy(), y.copy(), rank, penalty))
        # Raw matrix deliberately differs: the returned fit must use P*d@Q.T.
        return SimpleNamespace(P=np.eye(10)[:, :2], Q=np.eye(10)[:, :2], d=np.array([2., 1.]),
                               coefficient=np.eye(10) * 7., dual_gaps=np.zeros(10),
                               kkt_residual=0., converged=True, n_iter=np.ones(10, dtype=int))
    monkeypatch.setattr(f, "reduced_lasso", lasso)
    record, coefficient = f.fit_method(data, frames, "initializer_only", 2,
                                        small_configuration(initializer_penalties=[.03]))
    expected = (frames["left"][:, :2] * [2., 1.]) @ frames["right"][:, :2].T
    np.testing.assert_array_equal(coefficient, expected)
    np.testing.assert_array_equal(calls[0][0], data["X"] @ frames["left"][:, :10])
    np.testing.assert_array_equal(calls[0][1], data["Y"] @ frames["right"][:, :10])
    assert record["source_dimensions"] == [10, 10]
    assert record["effective_rank"] == 2
    assert "untruncated" in record["candidate_results"][0]["diagnostics"]["certificate_scope"]


def test_exact_zero_public_initializer_is_eligible_without_spectral_floor():
    data, frames = fixture()
    data["Y"][:] = 0
    record, coefficient = f.fit_method(data, frames, "initializer_only", 2, small_configuration())
    assert record["status"] == "success" and record["selected_index"] == 0
    assert record["effective_rank"] == 0
    np.testing.assert_array_equal(coefficient, np.zeros((4, 3)))
    diagnostics = record["candidate_results"][0]["diagnostics"]
    assert diagnostics["converged"] and diagnostics["zero_initializer"]
    assert not diagnostics["chart_or_spectral_floor_restrictions_applied"]


def test_public_initializer_actual_nonzero_reconstruction_and_validation():
    data, frames = fixture()
    record, coefficient = f.fit_method(data, frames, "initializer_only", 2,
                                        small_configuration(initializer_penalties=[.03]))
    raw = f.reduced_lasso(data["X"], data["Y"], 2, .03)
    np.testing.assert_allclose(coefficient, (raw.P * raw.d) @ raw.Q.T, atol=1e-14)
    assert record["candidate_results"][0]["diagnostics"]["converged"]
    assert record["validation_mse"] == pytest.approx(np.mean((data["Y_validation"] - coefficient) ** 2))


def test_rank_deficient_target_rrr_matches_minimum_norm_lift():
    data, frames = fixture()
    data["X"] = np.array([[1., 1., 0., 0.], [2., 2., 0., 0.], [0., 0., 1., 0.]])
    data["Y"] = np.array([[1., 2., 0.], [2., 4., 0.], [0., 1., 2.]])
    record, coefficient = f.fit_method(data, frames, "target_rrr", 3)
    np.testing.assert_allclose(coefficient, np.linalg.lstsq(data["X"], data["Y"], rcond=None)[0], atol=1e-13)
    assert record["candidate_results"][0]["diagnostics"]["design_rank"] == 2
    assert record["effective_rank"] == 2


def test_zero_nuclear_scale_retains_all_declared_fraction_attempts():
    data, frames = fixture()
    data["C0"] = data["Y_validation"].copy()
    record, coefficient = f.fit_method(data, frames, "nuclear_contrast", None)
    assert record["n_candidates"] == record["n_eligible"] == 7
    assert record["training_scales"]["nuclear_contrast"] == 0.
    assert record["selected_index"] == 0
    np.testing.assert_array_equal(coefficient, data["C0"])


def test_initializer_prefix_expands_to_requested_paper_rank(monkeypatch):
    data, frames = fixture(p=12, q=11)
    observed = []
    original = f.reduced_lasso
    def lasso(x, y, rank, penalty, **kwargs):
        observed.append((x.shape[1], y.shape[1], rank))
        return original(x, y, rank, penalty, **kwargs)
    monkeypatch.setattr(f, "reduced_lasso", lasso)
    record, _ = f.fit_method(data, frames, "initializer_only", 11,
                             small_configuration(initializer_penalties=[.03]))
    assert observed == [(11, 11, 11)]
    assert record["source_dimensions"] == [11, 11]


def test_initializer_nonconvergence_is_preserved_and_excluded(monkeypatch):
    data, frames = fixture()
    def lasso(x, y, rank, penalty, **kwargs):
        return SimpleNamespace(P=np.eye(4)[:, :rank], Q=np.eye(3)[:, :rank], d=np.ones(rank),
                               coefficient=np.eye(4, 3), dual_gaps=np.ones(3), kkt_residual=1.,
                               converged=False, n_iter=np.ones(3, dtype=int))
    monkeypatch.setattr(f, "reduced_lasso", lasso)
    record, coefficient = f.fit_method(data, frames, "initializer_only", 2, small_configuration())
    assert coefficient is None
    assert all(a["status"] == "uncertified" for a in record["candidate_results"])
    assert all(a["diagnostics"]["kkt_residual"] == 1. for a in record["candidate_results"])


@pytest.mark.parametrize("field,value", [("schema_version", 2), ("ridge_multipliers", [-1.]),
    ("nuclear_tolerance", 0.), ("mixture_alphas", [1.1]), ("initializer_penalties", [0.]),
    ("source_dimensions", [False]), ("extra", 1)])
def test_invalid_configuration_rejected(field, value):
    data, frames = fixture()
    with pytest.raises(ValueError):
        f.fit_method(data, frames, "target_rrr", 1, {field: value})
