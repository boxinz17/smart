"""Energy screening and finite strong-RRQR row selection."""

from dataclasses import dataclass

import numpy as np

from .source import _positive_integer, _positive_real, _real_matrix


class AnchorFailure(ValueError):
    """No anchor passed the prescribed original-factor check."""


@dataclass(frozen=True)
class AnchorSelection:
    indices: np.ndarray
    center: np.ndarray
    screened_indices: np.ndarray
    min_singular_value: float
    n_exchanges: int


def _pivot_columns(matrix, indices):
    """Deterministic modified Gram-Schmidt CPQR, with original-index ties."""
    residuals = matrix.copy()
    chosen = []
    vectors = []
    available = list(range(matrix.shape[1]))
    for _ in range(matrix.shape[0]):
        norms = np.sum(residuals[:, available] ** 2, axis=0)
        largest = float(norms.max())
        tie_tolerance = 32 * np.finfo(float).eps * max(1.0, largest)
        candidates = [available[j] for j in range(len(available)) if largest - norms[j] <= tie_tolerance]
        pivot = min(candidates, key=lambda j: int(indices[j]))
        vector = matrix[:, pivot].copy()
        for _ in range(2):
            for existing in vectors:
                vector -= existing * np.dot(existing, vector)
        norm = np.linalg.norm(vector)
        if norm <= 32 * np.finfo(float).eps:
            raise AnchorFailure("Screened factor is numerically rank deficient")
        vector /= norm
        chosen.append(pivot)
        vectors.append(vector)
        available.remove(pivot)
        if available:
            residuals[:, available] -= np.outer(vector, vector @ residuals[:, available])
    return chosen


def select_anchor(F, *, anchor_min, qr_threshold=2.0, max_exchanges=10000):
    """Select rows after Gram-energy screening and whitening.

    Certification and the polar center use the original unthresholded F.
    Returned selected indices are in increasing order, matching chart rows.
    """
    F = _real_matrix(F, "F")
    anchor_min = _positive_real(anchor_min, "anchor_min")
    qr_threshold = _positive_real(qr_threshold, "qr_threshold")
    rows, rank = F.shape
    if rank < 1 or rows < rank or not np.allclose(F.T @ F, np.eye(rank), atol=1e-9, rtol=0):
        raise ValueError("F must have nonempty orthonormal columns")
    if not np.isfinite(anchor_min) or anchor_min <= 0:
        raise ValueError("anchor_min must be positive")
    if not np.isfinite(qr_threshold) or qr_threshold <= 1:
        raise ValueError("qr_threshold must be greater than one")
    max_exchanges = _positive_integer(max_exchanges, "max_exchanges")
    energies = np.sum(F * F, axis=1)
    order = np.lexsort((np.arange(rows), -energies))
    gram = np.zeros((rank, rank))
    screened = None
    for count, index in enumerate(order, start=1):
        gram += np.outer(F[index], F[index])
        if count >= rank and np.linalg.eigvalsh(gram)[0] >= 0.75 - 32 * np.finfo(float).eps:
            screened = order[:count]
            break
    if screened is None:
        raise AnchorFailure("Energy screening failed to retain three quarters of factor energy")
    values, vectors = np.linalg.eigh(gram)
    inverse_root = (vectors * (1.0 / np.sqrt(values))) @ vectors.T
    matrix = (F[screened] @ inverse_root).T
    selected = _pivot_columns(matrix, screened)
    n_exchanges = 0
    while True:
        # Fixed index order makes both exchange rows and lexicographic ties
        # independent of transient QR pivot positions.
        selected.sort(key=lambda j: int(screened[j]))
        remaining = sorted(set(range(len(screened))) - set(selected), key=lambda j: int(screened[j]))
        if not remaining:
            break
        try:
            transform = np.linalg.solve(matrix[:, selected], matrix[:, remaining])
        except np.linalg.LinAlgError as exc:
            raise AnchorFailure("Selected anchor is numerically singular") from exc
        magnitudes = np.abs(transform)
        largest = float(magnitudes.max())
        if largest <= qr_threshold:
            break
        if n_exchanges >= max_exchanges:
            raise AnchorFailure("Strong-RRQR exchange limit reached")
        ties = np.argwhere(largest - magnitudes <= 32 * np.finfo(float).eps * max(1.0, largest))
        i, j = (int(value) for value in ties[0])
        selected[i] = remaining[j]
        n_exchanges += 1
    indices = np.sort(screened[selected])
    block = F[indices]
    U, values, Vt = np.linalg.svd(block)
    smallest = float(values[-1])
    if smallest < 8 * anchor_min:
        raise AnchorFailure(f"No anchor certified: smallest singular value {smallest:.6g} is below 8b={8 * anchor_min:.6g}")
    return AnchorSelection(indices, U @ Vt, screened.copy(), smallest, n_exchanges)
