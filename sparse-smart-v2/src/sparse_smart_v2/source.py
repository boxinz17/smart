"""Fixed full source frames retaining all observed ordered SVD directions."""

from dataclasses import dataclass

import numpy as np

from sparse_smart.source import (
    ExactSource, NoisySource, SourceBases, complete_basis, deterministic_svd,
)

from .calibration import _integer, _real


@dataclass(frozen=True)
class ObservedSource:
    """An observed coefficient without an asserted source uncertainty bound."""

    coefficient: np.ndarray


def _matrix(value, name):
    array = np.asarray(value)
    if np.iscomplexobj(array):
        raise ValueError(f"{name} must be real")
    try:
        array = np.array(array, dtype=float, copy=True)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{name} must be a finite real matrix") from exc
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite real matrix")
    return array


def _frozen(array):
    if array is None:
        return None
    result = np.array(array, copy=True)
    result.setflags(write=False)
    return result


def _freeze(bases):
    return SourceBases(
        _frozen(bases.left), _frozen(bases.right),
        _frozen(bases.leading_left), _frozen(bases.leading_right),
        bases.mode, bases.noise_std, bases.gap_lower, bases.cluster_size,
        _frozen(bases.source_singular_values),
    )


def _check_frames(left, right, p, q, columns, tolerance):
    for name, frame, rows in (("left", left, p), ("right", right, q)):
        count = rows if columns is None else columns
        if frame.shape != (rows, count):
            raise ValueError(f"{name} source frame must have shape {(rows, count)}")
        if not np.allclose(frame.T @ frame, np.eye(count), atol=tolerance, rtol=0):
            raise ValueError(f"{name} source frame must have orthonormal columns")


def _prepared(source, p, q, source_rank, orthogonality_tol):
    left, right = _matrix(source.left, "left"), _matrix(source.right, "right")
    _check_frames(left, right, p, q, None, orthogonality_tol)
    leading_left = _matrix(source.leading_left, "leading_left")
    leading_right = _matrix(source.leading_right, "leading_right")
    if not np.array_equal(leading_left, left[:, :source_rank]) or not np.array_equal(leading_right, right[:, :source_rank]):
        raise ValueError("prepared leading frames must equal the requested full-frame prefixes")
    cluster = _integer(source.cluster_size, "cluster_size", minimum=1)
    if cluster > source_rank:
        raise ValueError("cluster_size must not exceed source_rank")
    if source.mode == "observed":
        if source.noise_std is not None or source.gap_lower is not None:
            raise ValueError("observed source must not assert uncertainty metadata")
    elif source.mode == "exact":
        if source.noise_std != 0 or source.gap_lower is not None:
            raise ValueError("exact source metadata must have zero noise and no gap bound")
    elif source.mode == "noisy":
        _real(source.noise_std, "noise_std", positive=True)
        _real(source.gap_lower, "gap_lower", positive=True)
    else:
        raise ValueError("prepared source mode must be exact, noisy, or observed")
    values = source.source_singular_values
    if values is not None:
        if np.iscomplexobj(values):
            raise ValueError("source singular values must be real")
        values = np.array(values, dtype=float, copy=True)
        if (values.shape != (min(p, q),) or not np.isfinite(values).all()
                or np.any(values < 0) or np.any(np.diff(values) > 0)):
            raise ValueError("source singular values must be a finite nonnegative descending vector")
    elif source.mode != "exact":
        raise ValueError("observed and noisy prepared sources require their singular values")
    return _freeze(SourceBases(left, right, leading_left, leading_right, source.mode,
                               source.noise_std, source.gap_lower, cluster, values))


def prepare_source(source, *, p, q, source_rank, orthogonality_tol=1e-9, tie_tol=1e-12):
    """Prepare full frames once, accepting a validated prepared source cache.

    Matrix inputs are equivalent to :class:`ObservedSource`. The reused
    ``sparse_smart`` deterministic SVD canonicalizes numerical ties/null
    directions at ``tie_tol``; all its observed singular directions are kept,
    including those beyond ``source_rank``. Only the leading prefixes enter
    initialization. Unknown observed noise/gap metadata is ``None``.
    Returned arrays are independent, read-only copies.
    """
    p, q = _integer(p, "p", minimum=1), _integer(q, "q", minimum=1)
    source_rank = _integer(source_rank, "source_rank", minimum=1)
    orthogonality_tol = _real(orthogonality_tol, "orthogonality_tol", positive=True)
    tie_tol = _real(tie_tol, "tie_tol", positive=True)
    if source_rank > min(p, q):
        raise ValueError("source_rank must not exceed min(p, q)")
    if isinstance(source, SourceBases):
        return _prepared(source, p, q, source_rank, orthogonality_tol)
    if isinstance(source, ExactSource):
        left, right = _matrix(source.U, "source.U"), _matrix(source.V, "source.V")
        _check_frames(left, right, p, q, source_rank, orthogonality_tol)
        return _freeze(SourceBases(
            complete_basis(left, tol=tie_tol), complete_basis(right, tol=tie_tol),
            left, right, "exact", 0.0, None, 1, None,
        ))
    if isinstance(source, NoisySource):
        noise = _real(source.noise_std, "source.noise_std", positive=True)
        gap = _real(source.gap_lower, "source.gap_lower", positive=True)
        cluster = _integer(source.cluster_size, "source.cluster_size", minimum=1)
        if cluster > source_rank:
            raise ValueError("source.cluster_size must not exceed source_rank")
        matrix, mode = source.coefficient, "noisy"
    else:
        matrix = source.coefficient if isinstance(source, ObservedSource) else source
        noise, gap, cluster, mode = None, None, 1, "observed"
    matrix = _matrix(matrix, "source.coefficient")
    if matrix.shape != (p, q):
        raise ValueError(f"source.coefficient must have shape {(p, q)}")
    U, values, Vt = deterministic_svd(matrix, tie_tol=tie_tol)
    left, right = complete_basis(U, tol=tie_tol), complete_basis(Vt.T, tol=tie_tol)
    return _freeze(SourceBases(left, right, left[:, :source_rank], right[:, :source_rank],
                               mode, noise, gap, cluster, values))


__all__ = ["ExactSource", "NoisySource", "ObservedSource", "SourceBases", "prepare_source"]
