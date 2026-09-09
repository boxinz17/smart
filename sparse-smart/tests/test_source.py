import numpy as np
import pytest

from sparse_smart.source import ExactSource, NoisySource, complete_basis, deterministic_svd, prepare_source


def test_exact_source_preserves_supplied_frame_and_coordinates():
    U = np.array([[0., 1.], [1., 0.], [0., 0.]])
    V = np.eye(4)[:, :2]
    bases = prepare_source(ExactSource(U, V), 2, 3, 4)
    np.testing.assert_array_equal(bases.left, U)
    np.testing.assert_array_equal(bases.right, V)
    assert bases.mode == "exact"
    assert bases.noise_std == 0
    U[0, 1] = 8
    assert bases.left[0, 1] == 1


def test_noisy_source_has_full_complements_and_ranked_leading_frames():
    rng = np.random.default_rng(10)
    matrix = rng.normal(size=(5, 4))
    bases = prepare_source(NoisySource(matrix, 0.1, 0.5, cluster_size=2), 2, 5, 4)
    assert bases.left.shape == (5, 5)
    assert bases.right.shape == (4, 4)
    assert bases.leading_left.shape == (5, 2)
    np.testing.assert_allclose(bases.left.T @ bases.left, np.eye(5), atol=1e-12)
    np.testing.assert_allclose(bases.right.T @ bases.right, np.eye(4), atol=1e-12)
    np.testing.assert_allclose(bases.left[:, :2], bases.leading_left)
    np.testing.assert_allclose(bases.right[:, :2], bases.leading_right)
    assert np.all(np.diff(bases.source_singular_values) <= 0)
    assert bases.cluster_size == 2


def test_basis_completion_uses_coordinate_order():
    frame = np.array([[1.], [1.], [0.]]) / np.sqrt(2)
    complete = complete_basis(frame)
    np.testing.assert_allclose(complete, np.array([[1 / np.sqrt(2), 1 / np.sqrt(2), 0],
                                                [1 / np.sqrt(2), -1 / np.sqrt(2), 0],
                                                [0, 0, 1]]), atol=1e-14)


def test_svd_ties_and_nullspaces_use_canonical_coordinates():
    matrix = np.diag([0., 3., 3.])
    U, d, Vt = deterministic_svd(matrix)
    expected = np.eye(3)[:, [1, 2, 0]]
    np.testing.assert_allclose(U, expected, atol=1e-14)
    np.testing.assert_allclose(Vt.T, expected, atol=1e-14)
    np.testing.assert_allclose((U * d) @ Vt, matrix, atol=1e-14)


def test_svd_reconstruction_orthogonality_and_signs_rectangular():
    rng = np.random.default_rng(7)
    U0 = np.linalg.qr(rng.normal(size=(7, 4)))[0]
    V0 = np.linalg.qr(rng.normal(size=(5, 4)))[0]
    matrix = (U0 * [3., 3., 1., 0.]) @ V0.T
    U, d, Vt = deterministic_svd(matrix)
    np.testing.assert_allclose((U * d) @ Vt, matrix, atol=1e-12)
    np.testing.assert_allclose(U.T @ U, np.eye(5), atol=1e-12)
    np.testing.assert_allclose(Vt @ Vt.T, np.eye(5), atol=1e-12)
    for vector in U.T:
        assert vector[np.flatnonzero(np.abs(vector) > 1e-12)[0]] > 0


@pytest.mark.parametrize("source", [ExactSource(np.ones((3, 2)), np.eye(3)[:, :2]),
                                    ExactSource(np.eye(3), np.eye(3)),
                                    NoisySource(np.eye(3), 0, 1),
                                    NoisySource(np.eye(3), 1, -1),
                                    NoisySource(np.eye(3), 1, 1, 3),
                                    NoisySource(np.eye(3) * np.nan, 1, 1)])
def test_source_validation(source):
    with pytest.raises(ValueError):
        prepare_source(source, 2, 3, 3)
