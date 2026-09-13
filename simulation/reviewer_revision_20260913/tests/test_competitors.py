"""Small algebraic correctness tests, not Monte Carlo performance experiments."""

import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "competitors.py"
SPEC = importlib.util.spec_from_file_location("reviewer_revision_competitors", MODULE_PATH)
competitors = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = competitors
SPEC.loader.exec_module(competitors)
fit_competitor = competitors.fit_competitor
select_competitor = competitors.select_competitor
candidate_grid = competitors.candidate_grid


def _rrr_reference(x, y, rank, ridge):
    # Independent augmented least-squares construction. Fitted responses, not
    # OLS coefficients, are truncated before the minimum-norm lift.
    n, p = x.shape
    augmented_x = np.vstack((x, np.sqrt(n * ridge) * np.eye(p)))
    augmented_y = np.vstack((y, np.zeros((p, y.shape[1]))))
    ols = np.linalg.lstsq(augmented_x, augmented_y, rcond=None)[0]
    left, singular, right_t = np.linalg.svd(augmented_x @ ols, full_matrices=False)
    fitted = (left[:, :rank] * singular[:rank]) @ right_t[:rank]
    return np.linalg.lstsq(augmented_x, fitted, rcond=None)[0]


def test_rrr_uses_design_metric_not_coefficient_truncation():
    x = np.diag([100., 1.])
    y = np.diag([2., 1.])
    fitted = fit_competitor(x, y, "target_rrr", rank=1)
    np.testing.assert_allclose(fitted.coefficient, np.diag([.02, 0.]), atol=1e-14)
    assert fitted.diagnostics["training_loss"] == pytest.approx(.25)
    assert fitted.diagnostics["certified"]


@pytest.mark.parametrize("shape", [(12, 5, 4), (5, 12, 4), (9, 4, 7)])
@pytest.mark.parametrize("ridge", [0., .03, 10.])
def test_rrr_matches_independent_augmented_design_optimum(shape, ridge):
    n, p, q = shape
    rng = np.random.default_rng(602)
    x = rng.normal(size=(n, p)) * np.geomspace(.2, 4., p)
    y = rng.normal(size=(n, q))
    result = fit_competitor(x, y, "target_ridge_rrr", rank=2, ridge=ridge)
    expected = _rrr_reference(x, y, 2, ridge)
    np.testing.assert_allclose(result.coefficient, expected, rtol=2e-11, atol=2e-12)
    assert result.diagnostics["effective_rank"] <= 2
    assert result.diagnostics["certified"]
    assert abs(result.diagnostics["objective_gap"]) <= result.diagnostics["objective_tolerance"]


def test_rank_deficiency_minimum_norm_and_out_of_sample_predictions_differ():
    x = np.array([[1., 1., 0.], [2., 2., 0.], [0., 0., 1.]])
    y = np.array([[1., 2.], [2., 4.], [2., -1.]])
    result = fit_competitor(x, y, "target_rrr", rank=2)
    np.testing.assert_allclose(result.coefficient, np.linalg.lstsq(x, y, rcond=None)[0], atol=1e-13)
    shift = np.array([[1., 2.], [-1., -2.], [0., 0.]])
    np.testing.assert_allclose(x @ (result.coefficient + shift), result.predict(x), atol=1e-13)
    assert np.linalg.norm(result.coefficient + shift) > np.linalg.norm(result.coefficient)
    assert np.linalg.norm(np.eye(3) @ shift) > 0
    assert result.diagnostics["design_rank"] == 2


@pytest.mark.parametrize("rank,zero_x,zero_y", [(0, False, False), (2, True, False), (2, False, True)])
def test_zero_limits(rank, zero_x, zero_y):
    x = np.zeros((4, 3)) if zero_x else np.arange(12.).reshape(4, 3)
    y = np.zeros((4, 2)) if zero_y else np.arange(8.).reshape(4, 2)
    result = fit_competitor(x, y, "target_rrr", rank=rank)
    np.testing.assert_allclose(result.coefficient, 0., atol=0)
    assert result.diagnostics["certified"]


def test_full_rank_is_ols_and_ridge_zero_limit_matches_rrr():
    rng = np.random.default_rng(78)
    x, y = rng.normal(size=(10, 4)), rng.normal(size=(10, 3))
    full = fit_competitor(x, y, "target_rrr", rank=3)
    np.testing.assert_allclose(full.coefficient, np.linalg.lstsq(x, y, rcond=None)[0], atol=1e-13)
    rrr = fit_competitor(x, y, "target_rrr", rank=2)
    ridge = fit_competitor(x, y, "target_ridge_rrr", rank=2, ridge=0)
    np.testing.assert_array_equal(rrr.coefficient, ridge.coefficient)


def test_tied_cutoff_is_canonical_in_response_coordinates():
    result = fit_competitor(np.eye(3), np.eye(3), "target_rrr", rank=1)
    np.testing.assert_allclose(result.coefficient, np.diag([1., 0., 0.]), atol=1e-14)
    assert result.diagnostics["rank_cutoff_tied"]


def test_source_restriction_matches_reduced_problem_and_full_space_limit():
    rng = np.random.default_rng(879)
    x, y = rng.normal(size=(9, 4)), rng.normal(size=(9, 4))
    source = np.diag([4., 3., 2., 1.])
    restricted = fit_competitor(x, y, "source_subspace_ridge_rrr", rank=2, ridge=.2,
                                source_coefficient=source, source_rank=(3, 2))
    reduced = _rrr_reference(x[:, :3], y[:, :2], 2, .2)
    expected = np.zeros((4, 4))
    expected[:3, :2] = reduced
    np.testing.assert_allclose(restricted.coefficient, expected, atol=1e-12)
    assert restricted.diagnostics["certified"]
    full = fit_competitor(x, y, "source_subspace_rrr", rank=2, source_coefficient=source, source_rank=4)
    target = fit_competitor(x, y, "target_rrr", rank=2)
    np.testing.assert_allclose(full.coefficient, target.coefficient, atol=1e-12)
    empty = fit_competitor(x, y, "source_subspace_rrr", rank=2, source_coefficient=source, source_rank=0)
    np.testing.assert_array_equal(empty.coefficient, np.zeros((4, 4)))


def test_oracle_subspaces_are_rotation_invariant_and_labeled():
    rng = np.random.default_rng(908)
    x, y = rng.normal(size=(8, 5)), rng.normal(size=(8, 4))
    left = np.linalg.qr(rng.normal(size=(5, 3)))[0]
    right = np.linalg.qr(rng.normal(size=(4, 3)))[0]
    rotation = np.linalg.qr(rng.normal(size=(3, 3)))[0]
    one = fit_competitor(x, y, "oracle_subspace_rrr", rank=2, oracle_left=left, oracle_right=right)
    two = fit_competitor(x, y, "oracle_subspace_rrr", rank=2,
                         oracle_left=left @ rotation, oracle_right=right @ rotation.T)
    np.testing.assert_allclose(one.coefficient, two.coefficient, atol=1e-12)
    assert one.diagnostics["oracle"]
    with pytest.raises(ValueError, match="only be supplied to an oracle"):
        fit_competitor(x, y, "target_rrr", rank=2, oracle_left=left)


def test_ridge_to_source_exact_normal_equation_and_nullspace_limit():
    rng = np.random.default_rng(501)
    x, y, source = rng.normal(size=(4, 7)), rng.normal(size=(4, 3)), rng.normal(size=(7, 3))
    ridge = .7
    result = fit_competitor(x, y, "ridge_to_source", ridge=ridge, source_coefficient=source)
    expected = np.linalg.solve(x.T @ x / len(x) + ridge * np.eye(7), x.T @ y / len(x) + ridge * source)
    np.testing.assert_allclose(result.coefficient, expected, atol=1e-12)
    zero = fit_competitor(x, y, "ridge_to_source", ridge=0, source_coefficient=source)
    ols = fit_competitor(x, y, "target_rrr", rank=3)
    np.testing.assert_allclose(zero.predict(x), ols.predict(x), atol=1e-12)
    assert np.linalg.norm(zero.coefficient - ols.coefficient) > 1
    row_projector = np.linalg.pinv(x) @ x
    np.testing.assert_allclose((np.eye(7) - row_projector) @ zero.coefficient,
                               (np.eye(7) - row_projector) @ source, atol=1e-12)
    large = fit_competitor(x, y, "ridge_to_source", ridge=1e12, source_coefficient=source)
    np.testing.assert_allclose(large.coefficient, source, atol=2e-11)


def test_mixture_endpoints_and_rank_scope():
    x, y, source = np.eye(3), np.diag([3., 2., 1.]), np.diag([1., 2., 3.])
    zero = fit_competitor(x, y, "source_target_mixture", rank=1, alpha=0, source_coefficient=source)
    one = fit_competitor(x, y, "source_target_mixture", rank=1, alpha=1, source_coefficient=source)
    np.testing.assert_allclose(zero.coefficient, np.diag([3., 0., 0.]), atol=1e-14)
    np.testing.assert_array_equal(one.coefficient, source)
    assert one.diagnostics["effective_rank"] == 3
    assert one.diagnostics["rank_constraint_scope"] == "target_component_only"


def test_nuclear_contrast_has_closed_form_for_orthonormal_design():
    n = 4
    x = np.sqrt(n) * np.eye(n)
    source = np.diag([.3, .2, .1, 0.])
    contrast = np.diag([3., 1., .2, 0.])
    y = x @ (source + contrast)
    penalty = .5
    result = fit_competitor(x, y, "nuclear_contrast", source_coefficient=source, nuclear_penalty=penalty)
    np.testing.assert_allclose(result.coefficient, source + np.diag([2.5, .5, 0., 0.]), atol=1e-12)
    assert result.diagnostics["converged"]
    assert result.diagnostics["duality_gap"] < 1e-12
    huge = fit_competitor(x, y, "nuclear_contrast", source_coefficient=source, nuclear_penalty=10)
    np.testing.assert_allclose(huge.coefficient, source, atol=1e-12)


def test_nuclear_contrast_dual_certificate_and_iteration_cap():
    rng = np.random.default_rng(91)
    x, y, source = rng.normal(size=(15, 7)), rng.normal(size=(15, 5)), rng.normal(size=(7, 5))
    result = fit_competitor(x, y, "nuclear_contrast", source_coefficient=source,
                            nuclear_penalty=.2, tolerance=1e-8, max_iter=2000)
    assert result.diagnostics["certified"]
    assert result.diagnostics["duality_gap"] <= result.diagnostics["duality_gap_threshold"]
    cap = fit_competitor(x, y, "nuclear_contrast", source_coefficient=source,
                         nuclear_penalty=.2, tolerance=1e-14, max_iter=1)
    assert not cap.diagnostics["certified"]
    assert cap.diagnostics["termination_reason"] == "iteration_cap"


def test_selection_ties_failures_truth_isolation_and_no_refit_default():
    x, y = np.eye(3), np.diag([3., 2., 1.])
    xv, yv = np.eye(3), np.zeros((3, 3))
    candidates = [{"method": "not_a_method"}, {"method": "target_rrr", "rank": 0},
                  {"method": "target_rrr", "rank": 0}, {"method": "target_rrr", "rank": 2}]
    result = select_competitor(x, y, xv, yv, candidates)
    assert result.selected_index == 1
    assert result.validation_loss == 0
    assert result.candidate_results[0]["status"] == "failed"
    assert len(result.candidate_results) == len(candidates)
    assert result.fit.diagnostics["n_train"] == len(x)
    assert not result.refit
    assert result.elapsed_seconds >= 0
    # All fitted target coefficients depend on training only; changing held-out
    # responses changes the selected index without changing candidate fits.
    changed = select_competitor(x, y, xv, y, candidates)
    assert changed.selected_index == 3
    direct = fit_competitor(x, y, "target_rrr", rank=2)
    np.testing.assert_array_equal(changed.coefficient, direct.coefficient)


def test_refit_is_explicit_and_selection_retains_pre_refit_score():
    x, y = np.eye(2), np.eye(2)
    xv, yv = np.eye(2), 3 * np.eye(2)
    result = select_competitor(x, y, xv, yv, [{"method": "target_rrr", "rank": 2}], refit=True)
    np.testing.assert_allclose(result.coefficient, 2 * np.eye(2), atol=1e-13)
    assert result.validation_loss == pytest.approx(2.)
    assert result.fit.diagnostics["n_train"] == 4


def test_candidate_grid_is_ordered_and_does_not_hide_oracles():
    grid = candidate_grid("source_subspace_ridge_rrr", ranks=(1, 2), ridges=(0, .1), source_ranks=(2, 3))
    assert len(grid) == 8
    assert grid[0] == dict(method="source_subspace_ridge_rrr", rank=1, ridge=0, source_rank=2)
    assert grid[-1] == dict(method="source_subspace_ridge_rrr", rank=2, ridge=.1, source_rank=3)
    assert all(not option["method"].startswith("oracle_") for option in grid)


@pytest.mark.parametrize("options", [
    {"rank": -1}, {"rank": True}, {"rank": 1.2}, {"rank": 3},
    {"ridge": -1}, {"ridge": np.nan}, {"ridge": .1},
])
def test_invalid_inputs_are_rejected(options):
    with pytest.raises(ValueError):
        fit_competitor(np.eye(2), np.eye(2), "target_rrr", **options)


def test_source_rank_cannot_invent_null_source_directions():
    with pytest.raises(ValueError, match="numerical rank"):
        fit_competitor(np.eye(3), np.eye(3), "source_subspace_rrr", rank=1,
                       source_coefficient=np.diag([1., 0., 0.]), source_rank=2)


def test_common_observed_frames_allow_shared_null_completion_without_oracle_label():
    source = np.diag([1., 0., 0.])
    result = fit_competitor(np.eye(3), np.diag([1., 2., 3.]), "source_subspace_rrr",
                            rank=2, source_coefficient=source, source_rank=2,
                            observed_left=np.eye(3), observed_right=np.eye(3))
    np.testing.assert_allclose(result.coefficient, np.diag([1., 2., 0.]), atol=1e-14)
    assert not result.diagnostics["oracle"]
    assert result.diagnostics["includes_observed_null_completion"]
    assert result.diagnostics["source_numerical_rank"] == 1
    assert result.diagnostics["source_frame_convention"] == "supplied_common_observed_source_decomposition"
    selected = select_competitor(np.eye(3), np.diag([1., 2., 3.]), np.eye(3), np.eye(3),
                                 [{"method": "source_subspace_rrr", "rank": 2, "source_rank": 2}],
                                 source_coefficient=source, observed_left=np.eye(3), observed_right=np.eye(3))
    np.testing.assert_array_equal(selected.coefficient, result.coefficient)


def test_result_coefficient_is_independent_and_read_only():
    source = np.eye(3)
    result = fit_competitor(np.eye(3), np.eye(3), "source_target_mixture", source_coefficient=source, alpha=1)
    source[:] = 0
    np.testing.assert_array_equal(result.coefficient, np.eye(3))
    assert not result.coefficient.flags.writeable


def test_uncertified_candidates_are_visible_and_excluded():
    rng = np.random.default_rng(52)
    x, y = rng.normal(size=(10, 5)), rng.normal(size=(10, 3))
    source = np.zeros((5, 3))
    candidates = [{"method": "nuclear_contrast", "nuclear_penalty": .01, "max_iter": 1, "tolerance": 1e-15},
                  {"method": "target_rrr", "rank": 1}]
    result = select_competitor(x, y, x, y, candidates, source_coefficient=source)
    assert result.candidate_results[0]["status"] == "uncertified"
    assert result.selected_index == 1
    with pytest.raises(ValueError) as caught:
        select_competitor(x, y, x, y, candidates[:1], source_coefficient=source)
    assert caught.value.candidate_results[0]["status"] == "uncertified"
