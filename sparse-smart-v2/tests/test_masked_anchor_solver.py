"""Numerical invariants for the opt-in masked anchor-constrained metric."""
import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from sparse_smart.chart import AnchorChart
from sparse_smart.stopping import ValidationStopRequest
from sparse_smart_v2 import anchor_solver as solver
from sparse_smart_v2.calibration import Margins, PracticalCalibration
from sparse_smart_v2.solver import objective_change, penalty_value
from sparse_smart_v2.support import FreeRows, choose_free_rows


def problem(rank=2):
    rng = np.random.default_rng(271 + rank)
    chart = AnchorChart(6, 5, np.arange(1, rank + 1), np.arange(rank), np.eye(rank), np.eye(rank))
    rotations = rank * (rank - 1) // 2
    state = chart.pack(.02 * rng.normal(size=rotations), .02 * rng.normal(size=rotations),
        np.arange(rank, 0, -1, dtype=float), .08 * rng.normal(size=(6-rank, rank)),
        .08 * rng.normal(size=(5-rank, rank)))
    free = choose_free_rows(chart, (rank + 1, rank + 1))
    X, Y = rng.normal(size=(20, 6)), rng.normal(size=(20, 5))
    return chart, state, X, Y, free


def calibration(free, penalties=(.03, .07), L=20.):
    return PracticalCalibration(.003, penalties, L, free.capacities)


@pytest.mark.parametrize('rank', [1, 2, 3])
def test_masked_H_objective_gradient_matches_original_ambient_objective(rank):
    chart, z, X, Y, free = problem(rank)
    h = solver._to_h(chart, z)
    value, gradient = solver._value_gradient_h(chart, h, X, Y)
    penalties = (.03, .07)
    _, _, d, hu, hv = chart.unpack(h)
    for sl, mask, lam, block in ((chart.z_u_slice, free.penalized_u, .03, hu),
                                (chart.z_v_slice, free.penalized_v, .07, hv)):
        gradient[sl] += (lam * mask * np.sign(block) * d).ravel()
        gradient[chart.d_slice] += lam * (mask * np.abs(block)).sum(axis=0)
    outside = (np.setdiff1d(np.arange(chart.n_u), free.rows_u),
               np.setdiff1d(np.arange(chart.n_v), free.rows_v))

    def actual_objective(internal):
        original = solver._to_z(chart, internal)
        P, d, Q = chart.reconstruct(original)
        return (np.sum((X @ ((P * d) @ Q.T) - Y)**2) / (2 * len(X))
                + .03*np.abs((P*d)[outside[0]]).sum()
                + .07*np.abs((Q*d)[outside[1]]).sum())

    assert_allclose(actual_objective(h), value + penalty_value(chart, z, free, penalties), atol=1e-12)
    rng = np.random.default_rng(rank)
    for _ in range(4):
        direction = rng.normal(size=h.size)
        step = 1e-6
        finite = (actual_objective(h+step*direction)-actual_objective(h-step*direction))/(2*step)
        assert_allclose(finite, gradient@direction, rtol=3e-7, atol=3e-7)


def test_block_step_masks_both_H_weights_and_d_penalty_gradient():
    chart, z, _, _, free = problem()
    h = solver._to_h(chart, z)
    gradient = np.zeros_like(h)
    margins = Margins(.001, 10., .0001, anchor_min=.04)
    result = solver._block_trial(chart, h, gradient, 20., (.03, .07), free, margins, 1e-11)
    old_ou, old_ov, d, hu, hv = chart.unpack(h)
    ou, ov, actual_d, actual_u, actual_v = chart.unpack(result.value)
    assert_allclose(ou, old_ou, atol=1e-16)
    assert_allclose(ov, old_ov, atol=1e-16)
    columns = .03*(free.penalized_u*np.abs(hu)).sum(0)+.07*(free.penalized_v*np.abs(hv)).sum(0)
    assert_allclose(actual_d, d-columns/20., atol=1e-15)
    for old, actual, mask, lam in ((hu, actual_u, free.penalized_u, .03),
                                  (hv, actual_v, free.penalized_v, .07)):
        expected = np.sign(old)*np.maximum(np.abs(old)-lam*mask*d/20., 0)
        assert_allclose(actual, expected, atol=1e-15)
        assert_array_equal(actual[~mask], old[~mask])


def test_active_anchor_allows_tangent_descent_without_lowering_floor():
    chart = AnchorChart(3, 2, [0], [0], np.eye(1), np.eye(1))
    margins = Margins(.001, 10., .0001, anchor_min=.04, trial_radius=2.)
    radius = np.sqrt(1-.04**2)*(1-2e-13)
    state = chart.pack([], [], [1.], radius*np.array([[.6], [.8]]), [[0.]])
    X = np.sqrt(3.)*np.eye(3)
    Y = X@np.array([[0., 0.], [2., 0.], [0., 0.]])
    free = choose_free_rows(chart, (2, 1))
    points = []
    def observe(t, point, record):
        assert chart.domain_reason(point, d_lower=.001, d_upper=10., gap=.0001, anchor_min=.04) is None
        points.append(point.copy())
        point[:] = np.nan  # callback must never alias internal state
    result = solver.refine(chart, state, X, Y, calibration=calibration(free, (0., 0.)),
        margins=margins, free_rows=free, iterations=8, iterate_callback=observe)
    assert result.success, (result.status, result.last_rejection)
    assert result.n_iter == 8 and result.termination_reason == 'max_iterations'
    assert result.history[-1].objective < result.history[0].objective - .5
    assert all(r.anchor_min_u >= .04 for r in result.history)
    assert result.history[-1].anchor_min_u < .04000001
    assert max(r.step_size_inverse for r in result.history) <= 20.
    assert_array_equal(points[-1], result.state)
    for before, after, record in zip(points, points[1:], result.history[1:]):
        h_step = np.linalg.norm(solver._to_h(chart, after)-solver._to_h(chart, before))
        assert_allclose(h_step, record.step_norm, atol=1e-14)
        assert record.objective_change <= -.25*record.step_size_inverse*h_step**2 + 1e-14


def test_masked_metadata_and_objective_match_every_saved_Z_checkpoint():
    chart, state, X, Y, free = problem(3)
    points = []
    result = solver.refine(chart, state, X, Y, calibration=calibration(free),
        margins=Margins(.001, 10., .0001, anchor_min=.04), free_rows=free,
        iterations=5, iterate_callback=lambda t, x, rec: points.append(x))
    assert result.success, (result.status, result.last_rejection)
    for point, record in zip(points, result.history):
        assert_allclose(record.objective, chart.loss(point, X, Y)+penalty_value(chart, point, free, (.03,.07)))
        counts = [np.count_nonzero(point[sl].reshape(mask.shape)[mask]) for sl, mask in
                  ((chart.z_u_slice, free.penalized_u), (chart.z_v_slice, free.penalized_v))]
        assert (record.support_u, record.support_v) == tuple(counts)
        assert (record.raw_support_u, record.raw_support_v) == tuple(counts)
        assert record.support_tolerance_u == record.support_tolerance_v == 0.
    for before, after, rec in zip(points, points[1:], result.history[1:]):
        assert_allclose(rec.objective_change, objective_change(chart,before,after,X,Y,free,(.03,.07)))
    assert result.numerical_work['solver'] == 'masked_anchor_projected'
    assert result.numerical_work['totals']['block_calls'] > 0
    assert result.numerical_work['totals']['u']['calls'] > 0
    assert 'iterations' not in result.numerical_work


def test_zero_penalty_trajectory_independent_of_declared_free_rows():
    chart, state, X, Y, _ = problem(2)
    results = []
    for counts in ((2,2), (3,4), (6,5)):
        free = choose_free_rows(chart, counts)
        results.append(solver.refine(chart, state, X, Y, calibration=calibration(free,(0.,0.)),
            margins=Margins(.001,10.,.0001,anchor_min=.04),free_rows=free,iterations=4))
    assert all(result.success for result in results)
    for result in results[1:]:
        assert_array_equal(result.state, results[0].state)


def test_restrictive_hard_caps_are_rejected_even_for_zero_penalty():
    chart,state,X,Y,free=problem()
    with pytest.raises(ValueError,match='restrictive hard sparsity caps are unsupported'):
        solver.refine(chart,state,X,Y,
            calibration=PracticalCalibration(.003,0.,20.,(free.capacities[0]-1,free.capacities[1])),
            margins=Margins(.001,10.,.0001),free_rows=free,iterations=1)


def test_masks_must_match_chart_not_only_block_shape():
    chart,state,X,Y,free=problem()
    bad=FreeRows(free.rows_u,free.rows_v,~free.penalized_u,free.penalized_v)
    with pytest.raises(ValueError,match='does not match chart rows'):
        solver.refine(chart,state,X,Y,calibration=calibration(bad),
            margins=Margins(.001,10.,.0001),free_rows=bad,iterations=1)


def test_validation_stop_has_distinct_success_status_and_Z_state():
    chart,state,X,Y,free=problem()
    result=solver.refine(chart,state,X,Y,calibration=calibration(free),
        margins=Margins(.001,10.,.0001),free_rows=free,iterations=10,
        iterate_callback=lambda t,x,r: ValidationStopRequest() if t==2 else None)
    assert result.success and result.status=='completed' and result.n_iter==2
    assert result.termination_reason=='validation_stop'
    assert chart.domain_reason(result.state,d_lower=.001,d_upper=10.,gap=.0001,anchor_min=.01) is None


def test_huge_inverse_step_does_not_manufacture_convergence():
    chart,state,X,Y,free=problem()
    result=solver.refine(chart,state,X,Y,calibration=calibration(free,(0.,0.),1e25),
        margins=Margins(.001,10.,.0001),free_rows=free,iterations=10,stationarity_tol=1e-6)
    assert result.status=='numerical_stagnation' and result.n_iter==0
    assert result.projected_gradient_norm > 1e-6


def test_diagonal_mapping_retains_cancellation_safe_spectral_gradient():
    chart=AnchorChart(1,1,[0],[0],np.eye(1),np.eye(1))
    state=chart.pack([],[],[1e12],np.empty((0,1)),np.empty((0,1)))
    gradient=np.full_like(state,1e-6)
    diagnostic=solver._mapping(chart,state,gradient,20.,(0.,0.),choose_free_rows(chart),
        Margins(.001,1e14,.0001),1e-11)
    assert diagnostic[0] == pytest.approx(1e-6)
    assert diagnostic[3] is None


def test_fixed_budget_does_not_accept_rounded_noop_with_nonzero_diagonal_mapping():
    chart=AnchorChart(1,1,[0],[0],np.eye(1),np.eye(1))
    state=chart.pack([],[],[1e12],np.empty((0,1)),np.empty((0,1)))
    free=choose_free_rows(chart)
    result=solver.refine(chart,state,np.array([[1e-6]]),np.array([[1e6-1.]]),
        calibration=calibration(free,(0.,0.)),margins=Margins(.001,1e14,.0001),
        free_rows=free,iterations=5)
    assert result.status=='numerical_stagnation' and result.n_iter==0
    assert result.projected_gradient_norm == pytest.approx(1e-6)


def test_active_spectral_gap_is_projected_without_changing_declared_bound():
    chart=AnchorChart(2,2,[0,1],[0,1],np.eye(2),np.eye(2))
    state=chart.pack([0.],[0.],[3.,2.5],np.empty((0,2)),np.empty((0,2)))
    free=choose_free_rows(chart)
    result=solver.refine(chart,state,np.sqrt(2)*np.eye(2),np.sqrt(2)*np.eye(2),
        calibration=calibration(free,(0.,0.),2.),margins=Margins(.01,10.,.5),
        free_rows=free,iterations=100,stationarity_tol=1e-9)
    assert result.success and result.termination_reason=='stationarity'
    assert_allclose(result.state[chart.d_slice],[1.25,.75],atol=2e-9)
    assert max(r.step_size_inverse for r in result.history)<=4.


def test_reference_trial_cache_records_reuse_without_extra_prox_solves():
    chart,state,X,Y,free=problem()
    work={}
    result=solver.refine(chart,state,X,Y,calibration=calibration(free),
        margins=Margins(.001,10.,.0001),free_rows=free,iterations=2,numerical_work=work)
    assert result.success and result.numerical_work is work
    assert work['totals']['cache_success_hits'] >= 1
    assert len(work['iterations']) == 3


@pytest.mark.parametrize('kwargs',[{'iterations':True},{'max_backtracks':-1},{'stationarity_tol':0},
    {'line_search_strategy':'unrecognized'},{'cache_failed_trials':'yes'}])
def test_invalid_controls_raise(kwargs):
    chart,state,X,Y,free=problem()
    with pytest.raises(ValueError):
        solver.refine(chart,state,X,Y,calibration=calibration(free),margins=Margins(.001,10.,.0001),
            free_rows=free,**kwargs)
