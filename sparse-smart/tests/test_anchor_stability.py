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


def test_warm_start_preserves_quadratic_steps_with_fewer_rejections_and_fixed_mapping_metric():
    chart, state, design, response, margins = scalar_problem(curvature=1000.)
    options = dict(calibration=calibration(chart), margins=margins, max_backtracks=20)
    warm = solver.refine_anchor_projected(chart, state, design, response, iterations=6, **options)
    assert warm.success and warm.n_iter == 6
    reset_state, reset_rejections = state.copy(), 0
    for row, previous in zip(warm.history[1:], warm.history):
        reset = solver.refine_anchor_projected(chart, reset_state, design, response, iterations=1, **options)
        assert reset.success
        reset_state = reset.state
        reset_rejections += reset.history[-1].backtracks
        assert row.line_search_start_inverse == max(1., previous.step_size_inverse/2.)
        assert row.mapping_step_size_inverse == 1.
        assert row.objective_change <= -.25*row.step_size_inverse*row.step_norm**2
    np.testing.assert_array_equal(warm.state, reset_state)
    assert sum(row.backtracks for row in warm.history[1:]) < reset_rejections/2
    assert warm.history[1].step_size_inverse > 100.


def test_inherited_unresolved_zero_step_retries_original_inverse(monkeypatch):
    chart, state, design, response, margins = scalar_problem(curvature=20.)
    real_trial, real_mapping = solver._block_trial, solver._mapping
    huge_inverse = float(2**48)
    flags = {'mapping': False, 'injected_zero': False}
    def mapping(*args, **kwargs):
        flags['mapping'] = True
        try:
            return real_mapping(*args, **kwargs)
        finally:
            flags['mapping'] = False
    def trial(chart, current, gradient, inverse, *args, **kwargs):
        if not flags['mapping']:
            if np.array_equal(current, state) and inverse < huge_inverse:
                # A transient inner-solver issue forced a huge first inverse.
                raise ArithmeticError('transient initial proximal failure')
            if not np.array_equal(current, state) and inverse == huge_inverse/2 and not flags['injected_zero']:
                flags['injected_zero'] = True
                return current.copy(), 0.
        return real_trial(chart, current, gradient, inverse, *args, **kwargs)
    monkeypatch.setattr(solver, '_mapping', mapping)
    monkeypatch.setattr(solver, '_block_trial', trial)
    result = solver.refine_anchor_projected(chart, state, design, response,
        calibration=calibration(chart), margins=margins, iterations=2, max_backtracks=60)
    assert result.success and result.n_iter == 2 and flags['injected_zero']
    assert result.history[1].step_size_inverse == huge_inverse
    final = result.history[2]
    assert final.line_search_start_inverse == huge_inverse/2
    assert any('reset inverse step' in reason for reason in final.rejections)
    assert final.step_size_inverse < 100. and final.mapping_step_size_inverse == 1.
    assert final.objective_change <= -.25*final.step_size_inverse*final.step_norm**2


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


def test_mapping_refines_when_error_interval_straddles_tolerance_even_above_it(monkeypatch):
    # The approximate movement exceeds tolerance, but its proximal error can
    # explain that excess; a tighter solve may establish stationarity.
    result, calls = mapping_with_stub(monkeypatch, [(1.2, .4), (.8, .1)])
    assert len(calls) == 2 and calls[0] is None and calls[1] > 0.
    assert result[0] == pytest.approx(.9)
    assert result[4:] == (.8, .1, 1, False)
