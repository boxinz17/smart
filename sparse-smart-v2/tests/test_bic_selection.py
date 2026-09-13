import json

import numpy as np
import pytest

from sparse_smart_v2.rrr import target_rrr
from sparse_smart_v2.selection import bic_from_rss, bic_score


def _chart_options():
    # p=5, q=4, r=2; expanded free sets protect one additional row each.
    return dict(rank=2, free_directions=(3, 3),
                weighted_u=np.array([[100., 100.], [1., 0.], [0., -2.]]),
                weighted_v=np.array([[100., 100.], [0., 3.]]),
                penalized_u=np.array([[False, False], [True, True], [True, True]]),
                penalized_v=np.array([[False, False], [True, True]]))


def _data():
    rng = np.random.default_rng(638)
    return rng.normal(size=(12, 5)), rng.normal(size=(12, 4)), np.zeros((5, 4))


def test_observed_residual_and_actual_masked_support_not_coefficient_nonzeros():
    x, y, c = _data()
    options = _chart_options()
    score = bic_score(x, y, c, **options)
    expected_rss = np.linalg.norm(y, "fro") ** 2
    assert score.rss == pytest.approx(expected_rss)
    assert (score.support_u, score.support_v) == (2, 1)
    assert score.free_dimension == 8
    assert score.model_dimension == 11
    assert score.score == pytest.approx(48 * np.log(expected_rss / 48) + 11 * np.log(48))
    assert score.support_tolerance == 0
    assert score.method == "sparse_smart_v2"
    assert json.loads(json.dumps(score.as_dict(), allow_nan=False))["rss"] == score.rss
    # Values in free rows carry no extra support charge and are not modified.
    original = options["weighted_u"].copy()
    options["weighted_u"][0] = 0
    assert bic_score(x, y, c, **options).score == score.score
    np.testing.assert_array_equal(original[1:], options["weighted_u"][1:])


def test_support_tolerance_is_absolute_strict_and_does_not_change_fitted_values():
    x, y, c = _data()
    options = _chart_options()
    options["weighted_u"][1] = [1e-10, 2e-10]
    original = options["weighted_u"].copy()
    literal = bic_score(x, y, c, **options)
    effective = bic_score(x, y, c, **options, support_tolerance=1e-10)
    assert literal.support_u == 3
    assert effective.support_u == 2
    assert effective.rss == literal.rss
    assert literal.score - effective.score == pytest.approx(np.log(48))
    np.testing.assert_array_equal(options["weighted_u"], original)


def test_changing_free_counts_retains_unpenalized_model_dimension():
    x, y, c = _data()
    options = _chart_options()
    first = bic_score(x, y, c, **options)
    options["free_directions"] = (4, 3)
    options["penalized_u"][1] = False
    second = bic_score(x, y, c, **options)
    assert second.free_dimension == first.free_dimension + 2
    assert second.support_u == first.support_u - 1
    assert second.model_dimension == first.model_dimension + 1
    assert second.score - first.score == pytest.approx(np.log(48))


def test_direct_rrr_pays_dense_rank_model_dimension_and_matches_training_loss():
    x, y, _ = _data()
    fit = target_rrr(x, y, 2)
    score = bic_score(x, y, fit.coefficient, rank=2, direct_rrr=True)
    assert score.design_rank == 5
    assert score.model_dimension == 2 * (5 + 4 - 2)
    assert score.rss == pytest.approx(2 * len(x) * fit.training_loss)
    assert score.support_u is None and score.support_v is None
    # Free-set labels cannot give identical RRR fits cheaper complexity.
    full = bic_score(x, y, fit.coefficient, rank=2, direct_rrr=True,
                     free_directions=(5, 4), design_rank=fit.design_rank)
    assert score == full


def test_rank_deficient_rrr_charges_identifiable_prediction_model_dimension():
    rng = np.random.default_rng(745)
    basis = rng.normal(size=(15, 2))
    x = np.column_stack((basis, basis[:, 0], basis[:, 1], np.zeros(15)))
    y = rng.normal(size=(15, 4))
    fit = target_rrr(x, y, 4)
    score = bic_score(x, y, fit.coefficient, rank=4, direct_rrr=True)
    assert score.design_rank == fit.design_rank == 2
    assert score.rank == 4 and score.model_rank == 2
    assert score.model_dimension == 2 * (2 + 4 - 2)
    assert score.model_dimension != 4 * (5 + 4 - 4)


def test_rrr_dimension_is_model_rank_not_realized_response_rank():
    rng = np.random.default_rng(986)
    x = rng.normal(size=(20, 5))
    # The response has rank one, with a residual outside the design span.
    y = np.outer(rng.normal(size=20), rng.normal(size=4))
    fit = target_rrr(x, y, 3)
    assert fit.effective_rank == 1
    score = bic_score(x, y, fit.coefficient, rank=3, direct_rrr=True)
    assert score.model_rank == 3
    assert score.model_dimension == 3 * (5 + 4 - 3)


def test_rrr_zero_design_and_zero_requested_rank_have_zero_dimension():
    _, y, c = _data()
    for x, rank in ((np.zeros((12, 5)), 2), (np.ones((12, 5)), 0)):
        score = bic_score(x, y, c, rank=rank, direct_rrr=True)
        assert score.model_dimension == 0
        assert score.rss == pytest.approx(np.sum(y ** 2))


def test_bic_uses_observed_training_response_only():
    x, y, c = _data()
    first = bic_score(x, y, c, **_chart_options())
    second = bic_score(x, 2 * y, c, **_chart_options())
    assert second.score - first.score == pytest.approx(48 * np.log(4))
    with pytest.raises(TypeError):
        bic_score(x, y, c, **_chart_options(), validation_data=(x, y))
    with pytest.raises(TypeError):
        bic_score(x, y, c, **_chart_options(), true_coefficient=c)


@pytest.mark.parametrize("override", [
    {"rank": True}, {"rank": -1}, {"rank": 5}, {"rank": 1.5}, {"rank": 0},
    {"free_directions": (1, 3)}, {"free_directions": (6, 3)},
    {"free_directions": (3, True)}, {"free_directions": None},
    {"weighted_u": None}, {"weighted_u": np.ones((5, 2))},
    {"weighted_v": [[0, 0], [np.inf, 1]]},
    {"penalized_u": np.ones((3, 2))},
    {"penalized_u": np.array([[False, True], [True, False], [True, True]])},
    {"penalized_v": np.ones((2, 2), dtype=bool)},
    {"support_tolerance": -1}, {"support_tolerance": np.nan},
    {"support_tolerance": True}, {"design_rank": 5}, {"direct_rrr": 1},
])
def test_invalid_chart_scoring_inputs_rejected(override):
    options = _chart_options()
    options.update(override)
    with pytest.raises(ValueError):
        bic_score(*_data(), **options)


@pytest.mark.parametrize("override", [
    {"design_rank": 6}, {"design_rank": -1}, {"design_rank": True},
    {"weighted_u": np.zeros((3, 2))}, {"free_directions": (1, 2)},
])
def test_invalid_rrr_scoring_inputs_rejected(override):
    options = dict(rank=2, direct_rrr=True)
    options.update(override)
    with pytest.raises(ValueError):
        bic_score(*_data(), **options)


@pytest.mark.parametrize("position,replacement", [
    (0, np.ones((0, 5))), (0, np.ones((12, 0))), (0, np.ones(12)),
    (0, np.full((12, 5), np.nan)), (1, np.ones((11, 4))),
    (1, np.ones((12, 0))), (2, np.ones((4, 5))),
    (2, np.zeros((5, 4), dtype=complex)), (2, np.full((5, 4), np.inf)),
])
def test_invalid_observed_data_rejected(position, replacement):
    data = list(_data())
    data[position] = replacement
    with pytest.raises(ValueError):
        bic_score(*data, **_chart_options())


@pytest.mark.parametrize("override", [
    {"rss": 0}, {"rss": -1}, {"rss": np.inf}, {"rss": np.nan},
    {"rss": True}, {"n": 0}, {"q": -1}, {"n": 1.5},
    {"model_dimension": -1}, {"model_dimension": True},
])
def test_invalid_rss_or_bic_dimensions_rejected(override):
    options = dict(rss=1., n=12, q=4, model_dimension=10)
    options.update(override)
    with pytest.raises(ValueError):
        bic_from_rss(**options)


def test_zero_rss_and_overflow_do_not_silently_win():
    with pytest.raises(ValueError, match="rss"):
        bic_score(np.eye(2), np.eye(2), np.eye(2), rank=2, direct_rrr=True)
    with pytest.raises(ValueError, match="rss"):
        bic_score(np.eye(2), np.full((2, 2), 1e308), np.eye(2), rank=1, direct_rrr=True)
    assert np.isfinite(bic_from_rss(np.nextafter(0., 1.), n=100, q=20, model_dimension=3))
