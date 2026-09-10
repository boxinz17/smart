import numpy as np
import pytest
from scipy.linalg import polar

from sparse_smart.chart import AnchorChart, skew_coordinates, skew_matrix


def make_chart(n_u=7, n_v=6, rank=3):
    rng = np.random.default_rng(310)
    center_u = np.linalg.qr(rng.normal(size=(rank, rank)))[0]
    center_v = np.linalg.qr(rng.normal(size=(rank, rank)))[0]
    chart = AnchorChart(n_u, n_v, np.arange(rank), np.arange(n_v - rank, n_v), center_u, center_v)
    d = np.linspace(3.0, 1.0, rank)
    omega_u = rng.normal(scale=0.07, size=rank * (rank - 1) // 2)
    omega_v = rng.normal(scale=0.07, size=rank * (rank - 1) // 2)
    z_u = rng.normal(scale=0.06, size=(n_u - rank, rank)) * d
    z_v = rng.normal(scale=0.06, size=(n_v - rank, rank)) * d
    return chart, chart.pack(omega_u, omega_v, d, z_u, z_v)


def test_skew_basis_is_frobenius_orthonormal():
    coordinates = np.array([0.3, -0.2, 0.1, -0.07, 0.9, 0.0])
    matrix = skew_matrix(coordinates, 4)
    np.testing.assert_allclose(matrix, -matrix.T)
    np.testing.assert_allclose(np.linalg.norm(matrix, "fro"), np.linalg.norm(coordinates))
    np.testing.assert_allclose(skew_coordinates(matrix), coordinates)


def test_chart_orthogonality_weighted_zeros_and_packing():
    chart, x = make_chart()
    x[chart.z_u_slice.start + 1] = 0.0
    x[chart.z_v_slice.start + 2] = 0.0
    blocks = chart.unpack(x)
    np.testing.assert_array_equal(chart.pack(*blocks), x)
    P, d, Q = chart.reconstruct(x)
    np.testing.assert_allclose(P.T @ P, np.eye(chart.rank), atol=2e-14)
    np.testing.assert_allclose(Q.T @ Q, np.eye(chart.rank), atol=2e-14)
    np.testing.assert_allclose(P[chart.complement_u] * d, blocks[3], atol=1e-15)
    np.testing.assert_allclose(Q[chart.complement_v] * d, blocks[4], atol=1e-15)
    assert P[chart.complement_u[0], 1] == 0.0
    assert Q[chart.complement_v[0], 2] == 0.0


@pytest.mark.parametrize("n_u,n_v,rank", [(7, 6, 3), (6, 8, 2), (1, 1, 1), (1, 5, 1), (4, 1, 1), (3, 3, 3)])
def test_analytic_gradient_matches_directional_derivatives(n_u, n_v, rank):
    chart, x = make_chart(n_u, n_v, rank)
    rng = np.random.default_rng(93)
    design = rng.normal(size=(11, n_u))
    response = rng.normal(size=(11, n_v))
    loss, gradient = chart.value_gradient(x, design, response)
    np.testing.assert_allclose(loss, chart.loss(x, design, response), rtol=1e-14)
    for _ in range(8):
        direction = rng.normal(size=chart.size)
        direction /= np.linalg.norm(direction)
        epsilon = 2e-6
        numerical = (chart.loss(x + epsilon * direction, design, response) - chart.loss(x - epsilon * direction, design, response)) / (2 * epsilon)
        np.testing.assert_allclose(gradient @ direction, numerical, rtol=2e-6, atol=2e-8)


def test_gradient_with_repeated_square_root_eigenvalues_and_inactive_entries():
    chart, x = make_chart(6, 6, 3)
    # H'H is a scalar identity on both sides, so square-root eigenvalues repeat.
    d = x[chart.d_slice]
    x[chart.z_u_slice] = (0.2 * np.eye(3) * d).ravel()
    x[chart.z_v_slice] = (0.1 * np.eye(3) * d).ravel()
    rng = np.random.default_rng(4)
    design, response = rng.normal(size=(9, 6)), rng.normal(size=(9, 6))
    _, gradient = chart.value_gradient(x, design, response)
    numerical = np.empty_like(x)
    epsilon = 1e-6
    for j in range(chart.size):
        direction = np.zeros_like(x)
        direction[j] = epsilon
        numerical[j] = (chart.loss(x + direction, design, response) - chart.loss(x - direction, design, response)) / (2 * epsilon)
    np.testing.assert_allclose(gradient, numerical, rtol=2e-6, atol=2e-8)
    assert np.any(np.abs(gradient[chart.z_u_slice][x[chart.z_u_slice] == 0]) > 1e-3)


def test_initial_state_recovers_supplied_factors():
    rng = np.random.default_rng(11)
    P = np.linalg.qr(rng.normal(size=(7, 2)))[0]
    Q = np.linalg.qr(rng.normal(size=(5, 2)))[0]
    anchors_u, anchors_v = [1, 4], [0, 3]
    center_u, _ = polar(P[anchors_u])
    center_v, _ = polar(Q[anchors_v])
    chart = AnchorChart(7, 5, anchors_u, anchors_v, center_u, center_v)
    x = chart.initial_state(P, np.array([2.0, 0.7]), Q)
    P_rebuilt, d_rebuilt, Q_rebuilt = chart.reconstruct(x)
    np.testing.assert_allclose(P_rebuilt, P, atol=1e-12)
    np.testing.assert_allclose(Q_rebuilt, Q, atol=1e-12)
    np.testing.assert_array_equal(d_rebuilt, [2.0, 0.7])
    with pytest.raises(ValueError, match="centers do not match"):
        AnchorChart(7, 5, anchors_u, anchors_v, -center_u, center_v).initial_state(P, [2.0, 0.7], Q)


def test_domain_rejects_all_distinct_failure_modes():
    chart, x = make_chart(4, 4, 2)
    bounds = dict(d_lower=0.2, d_upper=4.0, gap=0.1, anchor_min=0.05)
    assert chart.domain_reason(x, **bounds) is None
    bad = x.copy()
    bad[chart.d_slice.start] = 0.0
    assert "singular values" in chart.domain_reason(bad, **bounds)
    with pytest.raises(ValueError, match="positive"):
        chart.reconstruct(bad)
    bad = x.copy()
    bad[chart.d_slice] = [1.0, 1.0]
    assert "gap" in chart.domain_reason(bad, **bounds)
    bad = x.copy()
    bad[0] = 1.0
    assert "Cayley" in chart.domain_reason(bad, **bounds)
    bad = x.copy()
    bad[chart.z_u_slice] = 0.0
    bad[chart.z_u_slice.start] = x[chart.d_slice.start] * 1.01
    assert "positive definite square root" in chart.domain_reason(bad, **bounds)
    with pytest.raises(ValueError, match="positive definite square root"):
        chart.reconstruct(bad)
    bad[chart.z_u_slice.start] = x[chart.d_slice.start] * np.sqrt(1 - 0.02**2)
    assert "anchor" in chart.domain_reason(bad, **bounds)
    bad[0] = np.nan
    assert "finite" in chart.domain_reason(bad, **bounds)


def test_rank_one_has_no_gap_requirement():
    chart, x = make_chart(1, 1, 1)
    assert chart.size == 1
    assert chart.domain_reason(x, d_lower=0.1, d_upper=5.0, gap=100.0, anchor_min=0.1) is None


def test_chart_rejects_malformed_arrays_and_metadata():
    with pytest.raises(ValueError, match="increasing"):
        AnchorChart(3, 3, [1, 0], [0, 1], np.eye(2), np.eye(2))
    with pytest.raises(ValueError, match="orthogonal"):
        AnchorChart(3, 3, [0, 1], [0, 1], np.ones((2, 2)), np.eye(2))
    chart, x = make_chart()
    with pytest.raises(ValueError, match="real"):
        chart.reconstruct(x.astype(complex))
    with pytest.raises(ValueError, match="response"):
        chart.value_gradient(x, np.zeros((2, 7)), np.zeros((3, 6)))


def test_repeated_state_evaluation_reuses_geometry_without_aliasing(monkeypatch):
    chart, state = make_chart()
    calls = []
    original = chart._side

    def counted(*args, **kwargs):
        calls.append(None)
        return original(*args, **kwargs)

    monkeypatch.setattr(chart, "_side", counted)
    design, response = np.eye(chart.n_u), np.zeros((chart.n_u, chart.n_v))
    expected = chart.reconstruct(state)
    value, gradient = chart.value_gradient(state, design, response)
    assert chart.loss(state, design, response) == value
    assert chart.domain_reason(state, d_lower=.1, d_upper=5., gap=.1, anchor_min=.01) is None
    assert len(calls) == 2
    assert np.isfinite(gradient).all()
    exposed = chart.reconstruct(state)
    for array in exposed:
        assert array.flags.writeable
        array[:] = -999
    for actual, wanted in zip(chart.reconstruct(state), expected):
        np.testing.assert_array_equal(actual, wanted)
    # Mutation of an existing state object must not be mistaken for a cache hit.
    state[chart.d_slice] += .1
    altered = chart.reconstruct(state)
    assert len(calls) == 4
    assert not np.array_equal(altered[1], expected[1])


def test_chart_copy_recomputes_after_geometry_mutation():
    from copy import deepcopy
    chart, state = make_chart()
    original = chart.reconstruct(state)
    copied = deepcopy(chart)
    copied.reconstruct(state)
    copied.center_u.setflags(write=True)
    copied.center_u *= -1
    left, singular, right = copied.reconstruct(state)
    np.testing.assert_array_equal(left[copied.anchors_u], -original[0][copied.anchors_u])
    np.testing.assert_array_equal(left[copied.complement_u], original[0][copied.complement_u])
    np.testing.assert_array_equal(singular, original[1])
    np.testing.assert_array_equal(right, original[2])
    np.testing.assert_array_equal(chart.reconstruct(state)[0], original[0])
