"""Anchor charts and analytic derivatives for sparse SMART refinement.

This implements equations (chart) and (outer) of
``notes/v1_sparse_two_stage.tex``.  Complement coordinates are weighted by
the singular values; reconstructing factors therefore preserves their zeros.
The reverse derivative of the positive square root uses an r-by-r Sylvester
solve, including when the square root has repeated eigenvalues.
"""

from __future__ import annotations

from numbers import Integral

import numpy as np
from scipy.linalg import solve_sylvester


def _real_array(value, name: str) -> np.ndarray:
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real")
    result = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    return result


def skew_matrix(coordinates, rank: int) -> np.ndarray:
    """Expand upper-triangular coordinates in a Frobenius-orthonormal basis.

    Coordinates follow ``np.triu_indices(rank, 1)`` and multiply
    ``(E_ij - E_ji) / sqrt(2)``.  Their Euclidean norm is consequently the
    Frobenius norm of the returned skew-symmetric matrix.
    """
    if isinstance(rank, (bool, np.bool_)) or not isinstance(rank, Integral) or rank < 1:
        raise ValueError("rank must be a positive integer")
    coordinates = _real_array(coordinates, "skew coordinates")
    if coordinates.shape != (rank * (rank - 1) // 2,):
        raise ValueError("skew coordinates have the wrong shape")
    upper = np.triu_indices(rank, 1)
    matrix = np.zeros((rank, rank), dtype=np.float64)
    matrix[upper] = coordinates / np.sqrt(2.0)
    matrix[(upper[1], upper[0])] = -matrix[upper]
    return matrix


def skew_coordinates(matrix) -> np.ndarray:
    """Extract Frobenius-orthonormal coordinates from a skew matrix."""
    matrix = _real_array(matrix, "skew matrix")
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("skew matrix must be square")
    if not np.allclose(matrix, -matrix.T, rtol=0.0, atol=1e-12):
        raise ValueError("matrix must be skew-symmetric")
    upper = np.triu_indices(matrix.shape[0], 1)
    return (matrix[upper] - matrix[(upper[1], upper[0])]) / np.sqrt(2.0)


class AnchorChart:
    """Fixed-anchor local coordinates for a rank-r coefficient matrix.

    ``design`` in :meth:`value_gradient` is ``X @ F_L``; ``response`` is
    ``Y @ F_R``.  For a rectangular orthonormal ``F_R``, the omitted physical
    loss is an additive constant and has no effect on gradients or acceptance.

    Flat coordinates are ``(omega_u, omega_v, d, z_u, z_v)``.  Each omega is
    a vector in the orthonormal skew basis; each z block is flattened in C
    order.  Anchor indices must be in increasing order, and the supplied
    centers use that same row order.
    """

    def __init__(self, n_u, n_v, anchors_u, anchors_v, center_u, center_v):
        for value, name in ((n_u, "n_u"), (n_v, "n_v")):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.n_u, self.n_v = int(n_u), int(n_v)
        self.anchors_u = self._anchors(anchors_u, self.n_u, "anchors_u")
        self.anchors_v = self._anchors(anchors_v, self.n_v, "anchors_v")
        self.rank = len(self.anchors_u)
        if self.rank != len(self.anchors_v):
            raise ValueError("left and right anchors must have the same size")
        self.complement_u = np.setdiff1d(np.arange(self.n_u), self.anchors_u)
        self.complement_v = np.setdiff1d(np.arange(self.n_v), self.anchors_v)
        self.center_u = self._center(center_u, "center_u")
        self.center_v = self._center(center_v, "center_v")
        self._skew_size = self.rank * (self.rank - 1) // 2
        self._upper = np.triu_indices(self.rank, 1)
        self.omega_u_slice = slice(0, self._skew_size)
        self.omega_v_slice = slice(self._skew_size, 2 * self._skew_size)
        self.d_slice = slice(2 * self._skew_size, self.rank**2)
        start = self.rank**2
        self.z_u_slice = slice(start, start + (self.n_u - self.rank) * self.rank)
        start = self.z_u_slice.stop
        self.z_v_slice = slice(start, start + (self.n_v - self.rank) * self.rank)
        self.size = self.z_v_slice.stop
        for array in (
            self.anchors_u, self.anchors_v, self.complement_u,
            self.complement_v, self.center_u, self.center_v,
        ):
            array.flags.writeable = False

    @staticmethod
    def _anchors(value, dimension, name):
        array = np.asarray(value)
        if array.ndim != 1 or array.size == 0 or array.dtype.kind not in "iu":
            raise ValueError(f"{name} must be a nonempty integer vector")
        if np.any(array < 0) or np.any(array >= dimension):
            raise ValueError(f"{name} contains an out-of-range index")
        if np.any(np.diff(array.astype(np.int64)) <= 0):
            raise ValueError(f"{name} must contain distinct indices in increasing order")
        return array.astype(np.intp, copy=True)

    def _center(self, value, name):
        matrix = _real_array(value, name)
        if matrix.shape != (self.rank, self.rank):
            raise ValueError(f"{name} has the wrong shape")
        if not np.allclose(matrix.T @ matrix, np.eye(self.rank), rtol=0.0, atol=1e-10):
            raise ValueError(f"{name} must be orthogonal")
        return matrix.copy()

    def pack(self, omega_u, omega_v, d, z_u, z_v) -> np.ndarray:
        """Pack two skew-coordinate vectors, singular values and matrix blocks."""
        arrays = [
            _real_array(value, name)
            for value, name in zip(
                (omega_u, omega_v, d, z_u, z_v),
                ("omega_u", "omega_v", "d", "z_u", "z_v"),
            )
        ]
        shapes = (
            (self._skew_size,), (self._skew_size,), (self.rank,),
            (self.n_u - self.rank, self.rank),
            (self.n_v - self.rank, self.rank),
        )
        if any(value.shape != shape for value, shape in zip(arrays, shapes)):
            raise ValueError(f"coordinate block shapes must be {shapes}")
        return np.concatenate([value.ravel(order="C") for value in arrays])

    def unpack(self, x) -> tuple[np.ndarray, ...]:
        """Return coordinate blocks (views when x is already float64)."""
        x = _real_array(x, "x")
        if x.shape != (self.size,):
            raise ValueError(f"x must have shape ({self.size},)")
        return (
            x[self.omega_u_slice], x[self.omega_v_slice], x[self.d_slice],
            x[self.z_u_slice].reshape(self.n_u - self.rank, self.rank),
            x[self.z_v_slice].reshape(self.n_v - self.rank, self.rank),
        )

    def initial_state(self, P, d, Q) -> np.ndarray:
        """Extract weighted complements with zero Cayley rotations.

        Centers must be the polar factors of the supplied anchor blocks.  This
        is checked by reconstruction, so an incompatible center cannot silently
        change the supplied initializer.
        """
        P, d, Q = (_real_array(value, name) for value, name in ((P, "P"), (d, "d"), (Q, "Q")))
        if P.shape != (self.n_u, self.rank) or Q.shape != (self.n_v, self.rank):
            raise ValueError("initial factor shapes do not match the chart")
        if d.shape != (self.rank,) or np.any(d <= 0):
            raise ValueError("initial singular values must be positive with shape (rank,)")
        x = self.pack(
            np.zeros(self._skew_size), np.zeros(self._skew_size), d,
            P[self.complement_u] * d, Q[self.complement_v] * d,
        )
        P_rebuilt, _, Q_rebuilt = self.reconstruct(x)
        if not np.allclose(P, P_rebuilt, rtol=1e-10, atol=1e-10) or not np.allclose(Q, Q_rebuilt, rtol=1e-10, atol=1e-10):
            raise ValueError("initial factors are not orthonormal or chart centers do not match their anchor polar factors")
        return x

    def _side(self, omega, d, z, center, anchors, complement, dimension):
        omega_matrix = skew_matrix(omega, self.rank)
        identity = np.eye(self.rank)
        inverse = np.linalg.solve(identity - omega_matrix, identity)
        rotation = center @ (2.0 * inverse - identity)
        with np.errstate(over="raise", divide="raise", invalid="raise"):
            try:
                H = z / d
                gram = identity - H.T @ H
            except FloatingPointError as error:
                raise ValueError("nonfinite chart complement Gram matrix") from error
        gram = 0.5 * (gram + gram.T)
        eigenvalues, eigenvectors = np.linalg.eigh(gram)
        if np.any(eigenvalues <= 0.0):
            raise ValueError("complement Gram matrix has no positive definite square root")
        roots = np.sqrt(eigenvalues)
        W = (eigenvectors * roots) @ eigenvectors.T
        factor = np.empty((dimension, self.rank), dtype=np.float64)
        factor[anchors] = rotation @ W
        factor[complement] = H
        return factor, (H, W, inverse, rotation, omega_matrix, roots)

    def _parts(self, x):
        omega_u, omega_v, d, z_u, z_v = self.unpack(x)
        if np.any(d <= 0.0):
            raise ValueError("singular values must be positive")
        P, u_parts = self._side(
            omega_u, d, z_u, self.center_u, self.anchors_u, self.complement_u, self.n_u,
        )
        Q, v_parts = self._side(
            omega_v, d, z_v, self.center_v, self.anchors_v, self.complement_v, self.n_v,
        )
        return P, d, Q, u_parts, v_parts

    def reconstruct(self, x) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return orthonormal factors P, positive singular values d and Q.

        Undefined positive square roots raise ``ValueError``; their eigenvalues
        are never clipped to manufacture a feasible trial.
        """
        P, d, Q, _, _ = self._parts(x)
        return P, d.copy(), Q

    def domain_reason(self, x, *, d_lower, d_upper, gap, anchor_min) -> str | None:
        """Return None in the manuscript domain, otherwise the failed check."""
        bounds = np.asarray([d_lower, d_upper, gap, anchor_min], dtype=np.float64)
        if not np.all(np.isfinite(bounds)) or d_lower <= 0 or d_upper < d_lower or gap < 0 or not 0 < anchor_min <= 1:
            raise ValueError("domain bounds require 0 < d_lower <= d_upper, gap >= 0 and 0 < anchor_min <= 1")
        try:
            omega_u, omega_v, d, _, _ = self.unpack(x)
            if np.any(d < d_lower) or np.any(d > d_upper):
                return "singular values are outside [d_lower, d_upper]"
            if self.rank > 1 and np.any(d[:-1] - d[1:] < gap):
                return "adjacent singular-value gap is below gap"
            for omega, side in ((omega_u, "left"), (omega_v, "right")):
                if np.linalg.norm(skew_matrix(omega, self.rank), ord=2) > 0.5:
                    return f"{side} Cayley coordinate has operator norm above 1/2"
            _, _, _, u_parts, v_parts = self._parts(x)
            for parts, side in ((u_parts, "left"), (v_parts, "right")):
                if np.min(parts[-1]) < anchor_min:
                    return f"{side} anchor singular value is below anchor_min"
        except (ValueError, np.linalg.LinAlgError) as error:
            return str(error)
        return None

    def _data(self, design, response):
        design = _real_array(design, "design")
        response = _real_array(response, "response")
        if design.ndim != 2 or design.shape[1] != self.n_u or design.shape[0] == 0:
            raise ValueError("design must have shape (n, n_u) with n > 0")
        if response.shape != (design.shape[0], self.n_v):
            raise ValueError("response must have shape (n, n_v)")
        return design, response

    def loss(self, x, design, response) -> float:
        """Smooth projected loss ||response - design P D Q.T||_F^2 / (2n)."""
        design, response = self._data(design, response)
        P, d, Q = self.reconstruct(x)
        residual = ((design @ P) * d) @ Q.T - response
        value = float(np.sum(residual * residual) / (2.0 * design.shape[0]))
        if not np.isfinite(value):
            raise ValueError("loss is nonfinite")
        return value

    def _side_pullback(self, factor_gradient, d, parts, center, anchors, complement):
        H, W, inverse, rotation, _, _ = parts
        anchor_gradient = factor_gradient[anchors]
        rotation_gradient = anchor_gradient @ W
        square_root_gradient = rotation.T @ anchor_gradient
        square_root_gradient = 0.5 * (square_root_gradient + square_root_gradient.T)
        gram_gradient = solve_sylvester(W, W, square_root_gradient)
        complement_gradient = factor_gradient[complement] - H @ (gram_gradient + gram_gradient.T)
        z_gradient = complement_gradient / d
        d_gradient = -np.sum(complement_gradient * H, axis=0) / d
        omega_gradient = 2.0 * inverse.T @ center.T @ rotation_gradient @ inverse.T
        upper = self._upper
        omega_coordinates = (omega_gradient[upper] - omega_gradient[(upper[1], upper[0])]) / np.sqrt(2.0)
        return omega_coordinates, d_gradient, z_gradient

    def value_gradient(self, x, design, response) -> tuple[float, np.ndarray]:
        """Evaluate loss and its full analytic coordinate gradient.

        The calculation uses low-rank matrix products and two r-by-r Sylvester
        solves.  Every complement coordinate receives a gradient, even when its
        current value is zero, allowing support to change during refinement.
        """
        design, response = self._data(design, response)
        P, d, Q, u_parts, v_parts = self._parts(x)
        n = design.shape[0]
        projected_factor = design @ P
        residual = (projected_factor * d) @ Q.T - response
        value = float(np.sum(residual * residual) / (2.0 * n))
        projected_residual = residual @ Q
        P_gradient = (design.T @ projected_residual) * (d / n)
        Q_gradient = (residual.T @ projected_factor) * (d / n)
        d_gradient = np.sum(projected_factor * projected_residual, axis=0) / n
        omega_u_gradient, d_u_gradient, z_u_gradient = self._side_pullback(
            P_gradient, d, u_parts, self.center_u, self.anchors_u, self.complement_u,
        )
        omega_v_gradient, d_v_gradient, z_v_gradient = self._side_pullback(
            Q_gradient, d, v_parts, self.center_v, self.anchors_v, self.complement_v,
        )
        gradient = self.pack(
            omega_u_gradient, omega_v_gradient,
            d_gradient + d_u_gradient + d_v_gradient, z_u_gradient, z_v_gradient,
        )
        if not np.isfinite(value):
            raise ValueError("loss is nonfinite")
        return value, gradient
