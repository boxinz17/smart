"""Numerical regressions for stable objective changes and solver diagnostics."""
import numpy as np
import pytest

from sparse_smart import AnchorChart, Margins, ResolvedCalibration
import sparse_smart.anchor_solver as solver


def calibration(chart, inverse=1., penalties=(0., 0.)):
    counts = ((chart.n_u-chart.rank)*chart.rank, (chart.n_v-chart.rank)*chart.rank)
    return ResolvedCalibration(.03, penalties, inverse, counts, 'practical', {})


def scalar_problem(curvature=1.):
    chart = AnchorChart(1, 1, [0], [0], np.eye(1), np.eye(1))
    state = chart.pack([], [], [1.], np.empty((0, 1)), np.empty((0, 1)))
    design = np.full((4, 1), np.sqrt(curvature))
    return chart, state, design, 2.*design, Margins(.05, 5., .01, trial_radius=10.)


def objective(chart, state, design, response, penalties):
    return chart.loss(state, design, response) + sum(p*np.abs(state[sl]).sum()
        for p, sl in zip(penalties, (chart.z_u_slice, chart.z_v_slice)))


def test_stable_objective_change_matches_full_difference_in_regular_regime():
    rng = np.random.default_rng(135)
    chart = AnchorChart(4, 3, [0, 2], [0, 1], np.eye(2), np.eye(2))
    h = chart.pack([.12], [-.09], [2.2, 1.2], [[.08, -.2], [.11, .03]], [[-.07, .16]])
    state = solver._to_z(chart, h)
    trial = solver._to_z(chart, h+rng.normal(scale=.01, size=h.size))
    design, response = rng.normal(size=(8, 4)), rng.normal(size=(8, 3))
    penalties = (.13, .37)
    delta = objective(chart, trial, design, response, penalties)-objective(chart, state, design, response, penalties)
    stable = solver._objective_change(chart, state, trial, design, response, penalties)
    cached = solver._objective_change(chart, state, trial, design, response, penalties,
        context=solver._loss_context(chart, state, design, response))
    assert abs(delta) > 1e-4
    assert stable == pytest.approx(delta, rel=1e-12, abs=1e-14)
    assert cached == stable
    assert solver._objective_change(chart, trial, state, design, response, penalties) == pytest.approx(-stable)


def test_unmodellable_huge_residual_does_not_hide_decrease_or_accept_increase():
    chart, state, _, _, _ = scalar_problem()
    # The zero-design row has a huge residual constant for all coefficients.
    design, response = np.array([[1.], [0.]]), np.array([[2.], [1e16]])
    for d, expected in ((1.5, -.1875), (.5, .3125)):
        trial = state.copy()
        trial[chart.d_slice] = d
        assert chart.loss(trial, design, response) == chart.loss(state, design, response)
        assert solver._objective_change(chart, state, trial, design, response, (0., 0.)) == pytest.approx(expected, abs=1e-15)
    assert solver._objective_change(chart, state, state, design, response, (0., 0.)) == 0.


def test_l1_change_handles_sign_crossings_in_original_z_coordinates():
    chart = AnchorChart(3, 3, [0], [0], np.eye(1), np.eye(1))
    state = chart.pack([], [], [2.], [[.5], [-.2]], [[-.4], [.1]])
    trial = chart.pack([], [], [1.7], [[-.25], [.4]], [[.1], [-.2]])
    design, response = np.zeros((4, 3)), np.zeros((4, 3))
    # Signed-gradient linearization is wrong across zero; block L1 changes are
    # -.05 and -.2 despite the d change and all four signs crossing zero.
    expected = .7*(-.05)+1.1*(-.2)
    assert solver._objective_change(chart, state, trial, design, response, (.7, 1.1)) == pytest.approx(expected)
    assert solver._objective_change(chart, trial, state, design, response, (.7, 1.1)) == pytest.approx(-expected)


def test_solver_accepts_certified_decrease_when_recorded_objectives_round_equal():
    chart, state, _, _, margins = scalar_problem()
    design, response = np.array([[1.], [0.]]), np.array([[2.], [1e16]])
    result = solver.refine_anchor_projected(chart, state, design, response,
        calibration=calibration(chart), margins=margins, iterations=6)
    assert result.success and result.n_iter == 6
    assert len({row.objective for row in result.history}) == 1
    assert np.linalg.norm(result.state[chart.d_slice]-2.) < .02
    for row in result.history[1:]:
        assert row.step_norm > 0. and row.relative_step_norm > 0.
        assert row.objective_change < 0.
        assert row.objective_change <= -.25*row.step_size_inverse*row.step_norm**2


def test_reset_search_matches_repeated_one_step_solves_with_fixed_mapping_metric():
    chart, state, design, response, margins = scalar_problem(curvature=1000.)
    options = dict(calibration=calibration(chart), margins=margins, max_backtracks=20)
    result = solver.refine_anchor_projected(chart, state, design, response, iterations=6, **options)
    assert result.success and result.n_iter == 6
    reset_state, reset_rejections = state.copy(), 0
    for row in result.history[1:]:
        reset = solver.refine_anchor_projected(chart, reset_state, design, response, iterations=1, **options)
        assert reset.success
        reset_state = reset.state
        reset_rejections += reset.history[-1].backtracks
        assert row.line_search_start_inverse == 1.
        assert row.step_size_inverse == reset.history[-1].step_size_inverse
        assert row.backtracks == reset.history[-1].backtracks
        assert row.mapping_step_size_inverse == 1.
        assert row.objective_change <= -.25*row.step_size_inverse*row.step_norm**2
    np.testing.assert_array_equal(result.state, reset_state)
    assert sum(row.backtracks for row in result.history[1:]) == reset_rejections
    assert result.history[1].step_size_inverse > 100.


def test_reset_recovers_large_descent_step_after_transient_anchor_curvature():
    chart = AnchorChart(2, 1, [0], [0], np.eye(1), np.eye(1))
    h = np.sqrt(1.-.01**2)
    state = chart.pack([], [], [1.], [[h]], np.empty((0, 1)))
    design = np.sqrt(2.)*np.eye(2)
    response = design[:, :1]
    margins = Margins(.05, 5., .01, anchor_min=.001, trial_radius=10.)
    states = []
    # C=(d*sqrt(1-H**2), d*H). Near the initial small anchor, the
    # square-root curvature forces a tiny step. The first accepted move
    # leaves that region, making a much larger second step valid. Neither
    # the objective nor the proximal/line-search operations are mocked.
    result = solver.refine_anchor_projected(chart, state, design, response,
        calibration=calibration(chart), margins=margins, iterations=2, max_backtracks=30,
        iterate_callback=lambda iteration, current, record: states.append(current.copy()))
    assert result.success and result.n_iter == 2
    first, second = result.history[1:]
    assert first.step_size_inverse > 10_000.
    assert second.objective < first.objective-.1
    assert chart.loss(result.state, design, response) < .8
    assert second.step_norm > 8.*first.step_norm
    assert second.step_size_inverse < first.step_size_inverse/32.
    for row in (first, second):
        assert row.objective_change <= -.25*row.step_size_inverse*row.step_norm**2
        assert row.mapping_step_size_inverse == 1.
    for current in states:
        assert chart.domain_reason(current, d_lower=margins.d_lower, d_upper=margins.d_upper,
            gap=margins.gap, anchor_min=margins.anchor_min) is None


def mapping_with_stub(monkeypatch, outcomes, tolerance=1., inverse=10.):
    calls = []
    def trial(*args, **kwargs):
        calls.append(kwargs.get('absolute_gap_tol'))
        outcome = outcomes[len(calls)-1]
        if isinstance(outcome, Exception):
            raise outcome
        movement, uncertainty = outcome
        return np.array([movement/inverse, 0.]), uncertainty
    monkeypatch.setattr(solver, '_block_trial', trial)
    result = solver._mapping(object(), np.zeros(2), np.array([3., 4.]), inverse,
        (0., 0.), Margins(.05, 5., .01), 1e-11, tolerance)
    return result, calls


@pytest.mark.parametrize('tolerance,movement,uncertainty', [
    (None, .1, 2.), (1., 2., .1), (1., .1, .2), (1., 0., 0.)])
def test_mapping_skips_tightening_when_uncertainty_cannot_change_decision(monkeypatch, tolerance, movement, uncertainty):
    result, calls = mapping_with_stub(monkeypatch, [(movement, uncertainty)], tolerance)
    assert calls == [None]
    assert result[:4] == (movement+uncertainty, 5., 10., None)
    assert result[4:] == (movement, uncertainty, 0, False)


def test_mapping_tightens_blocking_uncertainty_and_keeps_residual_components(monkeypatch):
    result, calls = mapping_with_stub(monkeypatch, [(.25, 2.), (.3, .1)])
    residual, raw, inverse, reason, movement, uncertainty, refinements, limited = result
    assert len(calls) == 2 and calls[0] is None and 0. < calls[1] < 1e-3
    assert residual == pytest.approx(movement+uncertainty)
    assert (movement, uncertainty, raw, inverse, reason, refinements, limited) == (.3, .1, 5., 10., None, 1, False)
    assert residual < 1.


def test_unresolved_uncertainty_remains_limited_after_two_bounded_attempts(monkeypatch):
    result, calls = mapping_with_stub(monkeypatch, [(.25, 2.), (.25, 1.), (.25, .9)])
    assert len(calls) == 3 and 0. < calls[2] < calls[1]
    assert result[0] == pytest.approx(1.15) and result[3] is None
    assert result[4:] == (.25, .9, 2, True)
    assert result[0] > 1.


@pytest.mark.parametrize('failure', [ArithmeticError('unresolved inner solve'),
    FloatingPointError('precision limit'), ValueError('bad trial')])
def test_failed_tightening_preserves_original_finite_diagnostic(monkeypatch, failure):
    result, calls = mapping_with_stub(monkeypatch, [(.25, 2.), failure])
    assert len(calls) == 2
    assert result == (2.25, 5., 10., None, .25, 2., 1, True)


def test_nonimproving_tightening_keeps_better_diagnostic(monkeypatch):
    result, calls = mapping_with_stub(monkeypatch, [(.25, 2.), (3., .01)])
    assert len(calls) == 2
    assert result == (2.25, 5., 10., None, .25, 2., 1, True)


def test_active_kkt_solution_retains_uncertainty_when_absolute_target_is_unresolvable():
    weights, radius = np.array([.13, .37]), .8
    solution = np.array([[.5, 0.], [.4, .5], [0., .2]])
    solution *= radius/np.linalg.norm(solution, 2)
    left, _, right = np.linalg.svd(solution, full_matrices=False)
    value = solution + weights*np.sign(solution) + .6*np.outer(left[:, 0], right[0])
    prox = solver._weighted_l1_ball_prox(value, weights, radius, tolerance=1e-12, absolute_gap_tol=1e-30)
    assert prox.converged
    np.testing.assert_allclose(prox.value, solution, atol=2e-11)
    assert prox.gap_roundoff > 0. and prox.duality_gap+prox.gap_roundoff > 1e-30
    assert prox.error_bound > np.sqrt(2e-30)
    assert np.linalg.norm(prox.value-solution) <= prox.error_bound
    assert prox.error_bound == pytest.approx(np.sqrt(2*(prox.duality_gap+prox.gap_roundoff))+prox.direct_error)


def test_exact_stationary_point_with_unresolved_roundoff_floor_cannot_accept_zero_steps():
    chart = AnchorChart(2, 1, [0], [0], np.eye(1), np.eye(1))
    state = chart.pack([], [], [1.], [[0.]], np.empty((0, 1)))
    design = np.sqrt(2.)*np.eye(2)
    response = design[:, :1]
    options = dict(calibration=calibration(chart), margins=Margins(.05, 5., .01), iterations=3)
    ordinary = solver.refine_anchor_projected(chart, state, design, response, stationarity_tol=1e-7, **options)
    assert ordinary.status == 'converged' and ordinary.n_iter == 0
    tiny = solver.refine_anchor_projected(chart, state, design, response, stationarity_tol=1e-16, **options)
    assert tiny.status == 'numerical_stagnation' and not tiny.success and tiny.n_iter == 0
    assert len(tiny.history) == 1 and tiny.history[0].mapping_displacement == 0.
    assert tiny.proximal_uncertainty > 1e-16 and tiny.mapping_precision_limited
    np.testing.assert_array_equal(tiny.state, state)


def test_default_fixed_budget_accepts_exact_optimum_without_claiming_stationarity():
    from sparse_smart import ExactSource, PracticalCalibration, SparseSMART

    options = dict(rank=1, source_rank=2, sparsity=(1, 1), margins=Margins(.01, 5., .01),
                   calibration=PracticalCalibration(.01, 0., .5, (1, 1)), iterations=3)
    source = ExactSource(np.eye(2), np.eye(2))
    result = SparseSMART(**options).fit(np.eye(2), np.diag([.5, 0.]), source=source)
    assert result.success_ and result.n_iter_ == 3 and result.status_ == 'completed'
    assert not result.converged_ and result.termination_reason_ == 'max_iterations'
    np.testing.assert_array_equal(result.coefficient_, np.diag([.5, 0.]))
    assert result.result_.raw_gradient_norm == result.result_.mapping_displacement == 0.
    assert result.result_.proximal_uncertainty > 0.  # Allowance remains reported.
    assert all(row.step_norm == 0. for row in result.history_[2:])
    strict = SparseSMART(**options, stationarity_tol=1e-16).fit(
        np.eye(2), np.diag([.5, 0.]), source=source)
    assert not strict.success_ and strict.status_ == 'numerical_stagnation'


def test_mapping_refines_when_error_interval_straddles_tolerance_even_above_it(monkeypatch):
    # The approximate movement exceeds tolerance, but its proximal error can
    # explain that excess; a tighter solve may establish stationarity.
    result, calls = mapping_with_stub(monkeypatch, [(1.2, .4), (.8, .1)])
    assert len(calls) == 2 and calls[0] is None and calls[1] > 0.
    assert result[0] == pytest.approx(.9)
    assert result[4:] == (.8, .1, 1, False)


def test_default_fixed_budget_accepts_exact_bounded_optimum():
    from sparse_smart import ExactSource, PracticalCalibration, SparseSMART

    options = dict(rank=1, source_rank=2, sparsity=(1, 1), margins=Margins(.01, .5, .01),
                   calibration=PracticalCalibration(.3, 0., .5, (1, 1)), iterations=3)
    data = (np.eye(2), np.diag([1., 0.]))
    source = ExactSource(np.eye(2), np.eye(2))
    model = SparseSMART(**options).fit(*data, source=source)
    assert model.success_ and model.status_ == "completed" and model.n_iter_ == 3
    np.testing.assert_array_equal(model.coefficient_, np.diag([.5, 0.]))
    assert model.result_.raw_gradient_norm == .25
    assert model.result_.mapping_displacement == 0.
    assert model.result_.proximal_uncertainty > 0.
    assert not model.converged_
    strict = SparseSMART(**options, stationarity_tol=1e-16).fit(*data, source=source)
    assert strict.status_ == "numerical_stagnation" and not strict.success_


def test_default_fixed_budget_accepts_exact_l1_optimum():
    chart = AnchorChart(2, 2, [0], [0], np.eye(1), np.eye(1))
    state = chart.initial_state([[np.sqrt(.75)], [.5]], [1.], [[1.], [0.]])
    design = np.sqrt(2.) * np.eye(2)
    # The unconstrained entrywise L1 solution is exactly the initial coefficient.
    response = design @ np.array([[np.sqrt(.75), 0.], [.75, 0.]])
    options = dict(calibration=calibration(chart, penalties=(.25, .25)),
                   margins=Margins(.01, 5., .01), iterations=3)
    result = solver.refine_anchor_projected(chart, state, design, response, **options)
    assert result.success and result.n_iter == 3
    np.testing.assert_array_equal(result.state, state)
    assert result.raw_gradient_norm > .2
    assert result.history[-1].penalty_value == .125
    assert result.mapping_displacement == 0. and result.proximal_uncertainty > 0.
    strict = solver.refine_anchor_projected(chart, state, design, response,
                                            stationarity_tol=1e-16, **options)
    assert strict.status == "numerical_stagnation" and not strict.success


def test_uncertain_iterative_prox_cannot_manufacture_fixed_budget_noop(monkeypatch):
    from dataclasses import replace

    original = solver._weighted_l1_ball_prox

    def uncertain(*args, **kwargs):
        # Even an unchanged approximate solution is not a closed-form fixed point.
        return replace(original(*args, **kwargs), closed_form=False)

    monkeypatch.setattr(solver, "_weighted_l1_ball_prox", uncertain)
    chart = AnchorChart(2, 2, [0], [0], np.eye(1), np.eye(1))
    state = chart.pack([], [], [.5], [[0.]], [[0.]])
    result = solver.refine_anchor_projected(chart, state, np.eye(2), np.diag([.5, 0.]),
        calibration=calibration(chart), margins=Margins(.01, 5., .01), iterations=3)
    assert result.status == "numerical_stagnation" and not result.success
    assert result.mapping_displacement == 0. and result.proximal_uncertainty > 0.


@pytest.mark.parametrize("inverse", [.1, 10., 10000.])
def test_reference_trial_reuse_preserves_states_and_backtracking(monkeypatch, inverse):
    chart, state, design, response, margins = scalar_problem(curvature=1000.)
    original_trial, original_mapping = solver._block_trial, solver._mapping
    calls = []

    def counted(*args, **kwargs):
        calls.append(args[3])
        return original_trial(*args, **kwargs)

    monkeypatch.setattr(solver, "_block_trial", counted)
    options = dict(calibration=calibration(chart, inverse=inverse), margins=margins, iterations=5)
    reused = solver.refine_anchor_projected(chart, state, design, response, **options)
    reused_calls = len(calls)

    def uncached(*args, **kwargs):
        kwargs.pop("trial_cache", None)
        return original_mapping(*args, **kwargs)

    monkeypatch.setattr(solver, "_mapping", uncached)
    calls.clear()
    baseline = solver.refine_anchor_projected(chart, state, design, response, **options)
    assert reused.success and baseline.success
    np.testing.assert_array_equal(reused.state, baseline.state)
    assert reused.history == baseline.history
    expected_savings = reused.n_iter if 1. <= inverse <= 1000. else 0
    assert len(calls) - reused_calls == expected_savings


def test_mapping_refinement_keeps_original_line_search_trial(monkeypatch):
    initial = solver._BlockTrialResult(np.array([.9]), .3, False)
    refined = solver._BlockTrialResult(np.array([.8]), .1, False)
    calls = []

    def trial(*args, absolute_gap_tol=None):
        calls.append(absolute_gap_tol)
        return initial if absolute_gap_tol is None else refined

    monkeypatch.setattr(solver, "_block_trial", trial)
    cache = {}
    diagnostic = solver._mapping(object(), np.zeros(1), np.ones(1), 1., (0., 0.),
                                  Margins(.01, 5., .01), 1e-11, 1., trial_cache=cache)
    assert diagnostic[0] == pytest.approx(.9)
    assert diagnostic[4:6] == (.8, .1)
    assert cache["result"] is initial
    assert len(calls) == 2 and calls[0] is None and calls[1] > 0.
