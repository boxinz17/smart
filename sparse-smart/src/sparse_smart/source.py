"""Source coordinates and deterministic numerical basis conventions."""

from dataclasses import dataclass
from numbers import Real

import numpy as np


@dataclass(frozen=True)
class ExactSource:
    """Specified orthonormal source frames, each with ``source_rank`` columns."""

    U: np.ndarray
    V: np.ndarray


@dataclass(frozen=True)
class NoisySource:
    """Observed source coefficient and supplied source uncertainty bounds."""

    coefficient: np.ndarray
    noise_std: float
    gap_lower: float
    cluster_size: int = 1


@dataclass(frozen=True)
class SourceBases:
    left: np.ndarray
    right: np.ndarray
    leading_left: np.ndarray
    leading_right: np.ndarray
    mode: str
    noise_std: float
    gap_lower: float | None
    cluster_size: int
    source_singular_values: np.ndarray | None


def _real_matrix(value, name):
    array = np.asarray(value)
    if np.iscomplexobj(array):
        raise ValueError(f"{name} must be real")
    try:
        array = np.array(array, dtype=float, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite real matrix") from exc
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite real matrix")
    return array


def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _positive_real(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real) or not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive real number")
    return float(value)


def _reorthogonalize(vector, basis, tol):
    """Two matrix projection passes, with scalar fallback near dependence."""
    original = vector.copy()
    for _ in range(2):
        vector -= basis @ (basis.T @ vector)
    # Preserve coordinate-order decisions at nearly dependent candidates.
    # Such rare residuals are sensitive to the accumulation order, so retain
    # the original modified Gram-Schmidt calculation there.
    if np.linalg.norm(vector) <= max(8 * tol, np.sqrt(np.finfo(float).eps) * np.linalg.norm(original)):
        vector = original
        for _ in range(2):
            for column in range(basis.shape[1]):
                existing = basis[:, column]
                vector -= existing * np.dot(existing, vector)
    return vector


def _canonical_subspace(frame, tol):
    """Coordinate-order orthogonalization in an orthonormal column space."""
    dimension = frame.shape[1]
    selected = np.empty((frame.shape[0], dimension), order="F")
    count = 0
    for row in range(frame.shape[0]):
        vector = frame @ frame[row, :]
        vector = _reorthogonalize(vector, selected[:, :count], tol)
        norm = np.linalg.norm(vector)
        if norm > tol:
            selected[:, count] = vector / norm
            count += 1
        if count == dimension:
            return selected
    if dimension == 0:
        return np.empty((frame.shape[0], 0))
    raise ValueError("Could not determine a numerical coordinate-order basis; reduce tie_tol")


def complete_basis(frame, *, tol=1e-12, n_columns=None):
    """Append canonical complements up to ``n_columns`` (all rows by default).

    A partial completion is exactly the corresponding prefix of a full
    completion, including the supplied input columns and their orientation.
    """
    frame = _real_matrix(frame, "frame")
    tol = _positive_real(tol, "tol")
    rows, columns = frame.shape
    if columns > rows or not np.isfinite(tol) or tol <= 0:
        raise ValueError("frame dimensions or tol are invalid")
    if n_columns is None:
        n_columns = rows
    if (isinstance(n_columns, (bool, np.bool_))
            or not isinstance(n_columns, (int, np.integer))
            or not columns <= n_columns <= rows):
        raise ValueError("n_columns must be an integer between the input column and row counts")
    if not np.allclose(frame.T @ frame, np.eye(columns), atol=max(1e-10, 10 * tol), rtol=0):
        raise ValueError("frame must have orthonormal columns")
    if n_columns == columns:
        return frame
    # Column-major storage gives every partial completion the same contiguous
    # prefixes and projection order as the corresponding full completion.
    basis = np.empty((rows, n_columns), order="F")
    basis[:, :columns] = frame
    count = columns
    for index in range(rows):
        vector = np.zeros(rows)
        vector[index] = 1.0
        vector = _reorthogonalize(vector, basis[:, :count], tol)
        norm = np.linalg.norm(vector)
        if norm > tol:
            basis[:, count] = vector / norm
            count += 1
        if count == n_columns:
            break
    if count != n_columns:
        raise ValueError("Could not complete source basis; reduce tie_tol")
    return basis


def deterministic_svd(matrix, *, tie_tol=1e-12):
    """Thin SVD with coordinate-order tied subspaces and paired signs.

    Values within ``tie_tol * largest_singular_value`` are numerical ties.
    Rotating these subspaces can change reconstruction at that tolerance.
    Numerically zero left/right spaces are completed independently.
    """
    matrix = _real_matrix(matrix, "matrix")
    tie_tol = _positive_real(tie_tol, "tie_tol")
    if min(matrix.shape) < 1 or not np.isfinite(tie_tol) or tie_tol <= 0:
        raise ValueError("matrix must be nonempty and tie_tol must be positive")
    U, values, Vt = np.linalg.svd(matrix, full_matrices=False)
    V = Vt.T.copy()
    threshold = tie_tol * values[0]
    positive = int(np.count_nonzero(values > threshold)) if values[0] > 0 else 0
    start = 0
    while start < positive:
        stop = start + 1
        while stop < positive and values[start] - values[stop] <= threshold:
            stop += 1
        if stop - start > 1:
            canonical = _canonical_subspace(U[:, start:stop], tie_tol)
            rotation = U[:, start:stop].T @ canonical
            U[:, start:stop] = canonical
            V[:, start:stop] = V[:, start:stop] @ rotation
        start = stop
    if positive < len(values):
        U[:, positive:] = complete_basis(U[:, :positive], tol=tie_tol, n_columns=len(values))[:, positive:]
        V[:, positive:] = complete_basis(V[:, :positive], tol=tie_tol, n_columns=len(values))[:, positive:]
    for column in range(len(values)):
        nonzero = np.flatnonzero(np.abs(U[:, column]) > tie_tol)
        if nonzero.size and U[nonzero[0], column] < 0:
            U[:, column] *= -1
            V[:, column] *= -1
    return U, values, V.T


def prepare_source(source, source_rank, p, q, *, orthogonality_tol=1e-9, tie_tol=1e-12):
    """Build fixed exact or full noisy-source working bases."""
    source_rank = _positive_integer(source_rank, "source_rank")
    p = _positive_integer(p, "p")
    q = _positive_integer(q, "q")
    orthogonality_tol = _positive_real(orthogonality_tol, "orthogonality_tol")
    tie_tol = _positive_real(tie_tol, "tie_tol")
    if source_rank > min(p, q):
        raise ValueError("source_rank must not exceed min(p, q)")
    if not np.isfinite(orthogonality_tol) or orthogonality_tol <= 0:
        raise ValueError("orthogonality_tol must be positive")
    if isinstance(source, ExactSource):
        left = _real_matrix(source.U, "source.U")
        right = _real_matrix(source.V, "source.V")
        for name, frame, shape in (("source.U", left, (p, source_rank)), ("source.V", right, (q, source_rank))):
            if frame.shape != shape:
                raise ValueError(f"{name} must have shape {shape}")
            if not np.allclose(frame.T @ frame, np.eye(source_rank), atol=orthogonality_tol, rtol=0):
                raise ValueError(f"{name} must have orthonormal columns")
        return SourceBases(left, right, left.copy(), right.copy(), "exact", 0.0, None, 1, None)
    if isinstance(source, NoisySource):
        _positive_real(source.noise_std, "source.noise_std")
        _positive_real(source.gap_lower, "source.gap_lower")
        matrix = _real_matrix(source.coefficient, "source.coefficient")
        if matrix.shape != (p, q):
            raise ValueError(f"source.coefficient must have shape {(p, q)}")
        if not np.isfinite(source.noise_std) or source.noise_std <= 0:
            raise ValueError("source.noise_std must be positive")
        if not np.isfinite(source.gap_lower) or source.gap_lower <= 0:
            raise ValueError("source.gap_lower must be positive")
        cluster_size = _positive_integer(source.cluster_size, "source.cluster_size")
        if cluster_size > source_rank:
            raise ValueError("source.cluster_size must not exceed source_rank")
        U, values, Vt = deterministic_svd(matrix, tie_tol=tie_tol)
        leading_left = U[:, :source_rank]
        leading_right = Vt.T[:, :source_rank]
        return SourceBases(
            complete_basis(leading_left, tol=tie_tol), complete_basis(leading_right, tol=tie_tol),
            leading_left.copy(), leading_right.copy(), "noisy", float(source.noise_std),
            float(source.gap_lower), cluster_size, values.copy(),
        )
    raise TypeError("source must be an ExactSource or NoisySource")
