import numpy as np
import pytest
from sparse_smart import AnchorChart, Margins, ResolvedCalibration, refine
from sparse_smart.spectral import project_singular_values


def scalar_problem(d=1.0, target=2.0):
    chart = AnchorChart(1, 1, [0], [0], np.eye(1), np.eye(1))
    state = chart.pack([], [], [d], np.empty((0, 1)), np.empty((0, 1)))
    return chart, state, np.ones((4, 1)), np.full((4, 1), target)


def calibration(L=1., penalty=0., supports=(0, 0)):
    return ResolvedCalibration(.01, penalty, L, supports, "practical", {})


def test_fixed_T_and_exact_zero_steps():
    args = scalar_problem()
    result = refine(*args, calibration=calibration(), margins=Margins(.1, 5, .2, trial_radius=2), iterations=4)
    assert result.success and result.n_iter == 4 and len(result.history) == 5
    assert result.state[0] == 2.
    assert result.history[1].step_norm == 1.
    assert result.history[-1].step_norm == 0.


def test_backtracking_resets_and_obeys_required_decrease():
    result = refine(*scalar_problem(), calibration=calibration(.1),
                    margins=Margins(.1, 5, .2, trial_radius=.3), iterations=5)
    assert result.success
    for previous, record in zip(result.history, result.history[1:]):
        assert record.objective <= previous.objective - .25 * record.step_size_inverse * record.step_norm**2
        assert record.step_norm <= .3
        assert record.step_size_inverse == .1 * 2**record.backtracks


def test_failures_preserve_last_accepted_state():
    args = scalar_problem()
    result = refine(*args, calibration=calibration(.01), margins=Margins(.1, 5, .2),
                    iterations=2, max_backtracks=0)
    assert result.status == "line_search_failed"
    assert result.n_iter == 0
    np.testing.assert_array_equal(result.state, args[1])
    result = refine(*args, calibration=calibration(1e300), margins=Margins(.1, 5, .2), iterations=1)
    assert result.status == "numerical_stagnation"
    result = refine(*args, calibration=calibration(1e300, 1e100), margins=Margins(.1, 5, .2), iterations=1)
    assert result.status == "numerical_stagnation"  # lambda does not act on the core
    bad = args[1].copy()
    bad[0] = -1
    result = refine(args[0], bad, args[2], args[3], calibration=calibration(),
                    margins=Margins(.1, 5, .2))
    assert result.status == "invalid_initial_state"


def test_full_gradient_allows_inactive_coordinate_to_enter():
    chart = AnchorChart(3, 1, [0], [0], np.eye(1), np.eye(1))
    x = chart.pack([], [], [1.], [[0.], [0.]], np.empty((0, 1)))
    Z = np.sqrt(3.) * np.eye(3)
    W = Z @ np.array([[.98], [.2], [0.]])
    result = refine(chart, x, Z, W, calibration=calibration(2., .001, (1, 0)),
                    margins=Margins(.1, 5, .2), iterations=3)
    assert result.success
    assert result.state[chart.z_u_slice][0] != 0.
    assert result.history[-1].support_u == 1


def test_offset_does_not_change_acceptance():
    opts = dict(calibration=calibration(), margins=Margins(.1, 5, .2, trial_radius=2), iterations=2)
    reference = refine(*scalar_problem(), **opts)
    shifted = refine(*scalar_problem(), **opts, loss_offset=1e30)
    np.testing.assert_array_equal(shifted.state, reference.state)
    assert shifted.history[0].smooth_loss >= 1e30


def test_numerical_overflow_is_reported_and_malformed_supports_rejected():
    chart, state, _, response = scalar_problem()
    result = refine(chart, state, np.full((4, 1), 1e200), response,
                    calibration=calibration(), margins=Margins(.1, 5, .2))
    assert result.status == "numerical_failure"
    with pytest.raises(ValueError, match="exactly two"):
        refine(*scalar_problem(), calibration=calibration(supports=(0,)), margins=Margins(.1, 5, .2))
    with pytest.raises(ValueError, match="columns"):
        refine(chart, state, np.ones((4, 2)), response,
               calibration=calibration(), margins=Margins(.1, 5, .2))


def test_infinite_threshold_ratio_has_zero_limit():
    chart = AnchorChart(2, 2, [0], [0], np.eye(1), np.eye(1))
    x = chart.pack([], [], [1.], [[0.]], [[0.]])
    result = refine(chart, x, np.zeros((2, 2)), np.zeros((2, 2)),
                    calibration=calibration(1e-320, 1., (1, 1)),
                    margins=Margins(.1, 5, .2), iterations=1)
    assert result.success
    np.testing.assert_array_equal(result.state, x)


def test_spectral_projection_analytic_pooling_and_bounds():
    np.testing.assert_allclose(project_singular_values([1., 3., 2.], d_lower=0., d_upper=6., gap=1.), [3., 2., 1.])
    np.testing.assert_allclose(project_singular_values([100., -100., 100.], d_lower=1., d_upper=5., gap=1.), [5., 2., 1.])
    np.testing.assert_array_equal(project_singular_values([-4.], d_lower=.1, d_upper=5., gap=20.), [.1])
    with pytest.raises(ValueError, match="no feasible"):
        project_singular_values([1., 2., 3.], d_lower=1., d_upper=2., gap=1.)


def test_spectral_projection_matches_independent_active_set_qp():
    # Enumerate linear-constraint active sets, independently of PAVA.
    import itertools
    rank, lower, upper, gap = 3, .1, 4., .3
    A = np.array([[1., 0., 0.], [0., 0., -1.], [-1., 1., 0.], [0., -1., 1.]])
    b = np.array([upper, -lower, -gap, -gap])
    rng = np.random.default_rng(18)
    for raw in rng.normal(1., 4., size=(20, rank)):
        candidates = []
        for bits in itertools.product((False, True), repeat=len(b)):
            rows = np.flatnonzero(bits)
            if len(rows):
                active = A[rows]
                multipliers = np.linalg.lstsq(active @ active.T, active @ raw - b[rows], rcond=None)[0]
                candidate = raw - active.T @ multipliers
                if np.min(multipliers) < -1e-9 or np.max(np.abs(active @ candidate - b[rows])) > 1e-8:
                    continue
            else:
                candidate = raw.copy()
            if np.max(A @ candidate - b) <= 1e-8:
                candidates.append(candidate)
        expected = min(candidates, key=lambda value: np.sum((value - raw)**2))
        actual = project_singular_values(raw, d_lower=lower, d_upper=upper, gap=gap)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-8)
        assert np.min(actual[:-1] - actual[1:]) >= gap
        assert np.min(actual) >= lower and np.max(actual) <= upper


def gap_boundary_problem():
    chart = AnchorChart(3, 2, [0, 1], [0, 1], np.eye(2), np.eye(2))
    state = chart.pack([0.], [0.], [2., 1.], [[0., 0.]], np.empty((0, 2)))
    X = np.sqrt(3.) * np.eye(3)
    Y = X @ np.array([[1., 0.], [0., 1.], [.2, 0.]])
    return chart, state, X, Y


def test_projected_spectral_step_descends_beyond_rejection_gap_stall():
    options = dict(calibration=calibration(2., 0., (2, 0)),
                   margins=Margins(.1, 5., 1., trial_radius=2.), iterations=10)
    rejected = refine(*gap_boundary_problem(), **options, spectral_step="reject")
    projected = refine(*gap_boundary_problem(), **options, spectral_step="projected")
    assert not rejected.success and rejected.termination_reason == "active_gap_stall"
    assert projected.success and projected.termination_reason == "max_iterations"
    assert projected.history[-1].objective < rejected.history[-1].objective - .1
    assert all(record.objective <= old.objective for old, record in zip(projected.history, projected.history[1:]))
    assert projected.state[-2] != 0.  # Correction continues at the active gap.


def test_stationarity_uses_projected_mapping_including_spectral_normal_cone():
    result = refine(*scalar_problem(d=.1, target=0.), calibration=calibration(),
                    margins=Margins(.1, 5., .2), iterations=10, stationarity_tol=1e-8)
    assert result.success and result.status == "converged" and result.n_iter == 0
    assert result.termination_reason == "stationarity"
    assert result.projected_gradient_norm == 0. and result.raw_gradient_norm == .1
    huge_step = refine(*scalar_problem(), calibration=calibration(1e300),
                       margins=Margins(.1, 5., .2), iterations=10, stationarity_tol=1e-8)
    assert not huge_step.success and huge_step.termination_reason == "numerical_stagnation"
    assert huge_step.projected_gradient_norm > .9
    assert huge_step.history[0].mapping_step_size_inverse == 1000.


def test_non_spectral_boundary_is_reported_separately_from_stationarity():
    chart = AnchorChart(2, 2, [0, 1], [0, 1], np.eye(2), np.eye(2))
    empty = np.empty((0, 2))
    state = chart.pack([.5 * np.sqrt(2)], [0.], [2., 1.], empty, empty)
    target = chart.pack([.7 * np.sqrt(2)], [0.], [2., 1.], empty, empty)
    P, d, Q = chart.reconstruct(target)
    X = np.sqrt(2.) * np.eye(2)
    Y = X @ ((P * d) @ Q.T)
    result = refine(chart, state, X, Y, calibration=calibration(2.),
                    margins=Margins(.1, 5., .1), iterations=10, stationarity_tol=1e-8)
    assert not result.success
    assert result.termination_reason == "active_constraint_stall"
    assert "Cayley" in result.history[-1].mapping_domain_reason
    assert result.projected_gradient_norm > 1e-4


def test_callbacks_receive_initial_and_accepted_states_without_mutating_solver():
    seen = []
    def callback(iteration, state, record):
        seen.append((iteration, state.copy(), record))
        state[:] = np.nan
    options = dict(calibration=calibration(), margins=Margins(.1, 5., .2, trial_radius=2.), iterations=4)
    result = refine(*scalar_problem(), **options, iterate_callback=callback)
    assert result.success and result.n_iter == 4
    assert [row[0] for row in seen] == [0, 1, 2, 3, 4]
    np.testing.assert_array_equal(result.state, [2.])
    assert seen[0][1][0] == 1. and seen[-1][1][0] == 2.
    assert all(record.projected_gradient_norm is not None for _, _, record in seen)


def test_asymmetric_penalties_apply_to_the_correct_complement():
    chart = AnchorChart(2, 2, [0], [0], np.eye(1), np.eye(1))
    state = chart.pack([], [], [1.], [[0.]], [[0.]])
    X = np.sqrt(2.) * np.eye(2)
    Y = X @ np.array([[1., .1], [.1, 0.]])
    result = refine(chart, state, X, Y, calibration=calibration(2., (.2, 0.), (1, 1)),
                    margins=Margins(.1, 5., .1), iterations=1)
    assert result.success
    assert result.state[chart.z_u_slice][0] == 0.
    assert result.state[chart.z_v_slice][0] > 0.
    assert result.history[-1].penalty_value == 0.


@pytest.mark.parametrize("options", [{"spectral_step": "bad"}, {"stationarity_tol": 0.},
    {"stationarity_tol": float("nan")}, {"stationarity_tol": True}, {"iterate_callback": 3}])
def test_invalid_new_solver_options_rejected(options):
    with pytest.raises(ValueError):
        refine(*scalar_problem(), calibration=calibration(), margins=Margins(.1, 5., .1), **options)
