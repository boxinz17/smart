"""Training-only BIC-style comparison of fitted SparseSMART v2 candidates.

The parameter count is a model-dimension approximation, not a proved effective
degrees-of-freedom formula for the adaptive constrained estimator. The API has
no validation responses, simulated truth, or prediction-error input. It scores
the supplied coefficient using the *observed* training response.
"""

from dataclasses import asdict, dataclass
from math import log

import numpy as np

from .calibration import _integer, _integer_pair, _real
from .source import _matrix


# Matches the literal masked-coordinate support reported by both v2 solvers.
# A different absolute threshold must be fixed before comparing candidates.
DEFAULT_SUPPORT_TOLERANCE = 0.0


@dataclass(frozen=True)
class BICScore:
    score: float
    rss: float
    model_dimension: int
    free_dimension: int | None
    support_u: int | None
    support_v: int | None
    n: int
    p: int
    q: int
    rank: int
    model_rank: int
    design_rank: int | None
    support_tolerance: float
    method: str

    def as_dict(self):
        """Return finite JSON-compatible scoring inputs and the resulting BIC."""
        return {
            "criterion": "bic",
            "dimension_interpretation": "model_dimension_approximation",
            **asdict(self),
        }


def bic_from_rss(rss, *, n, q, model_dimension):
    """Return ``n*q*log(rss/(n*q)) + model_dimension*log(n*q)``.

    Candidate-independent terms are omitted. Exact zero RSS has no finite
    Gaussian BIC with an estimated positive variance and is rejected rather
    than assigned an arbitrary numerical floor. Nonfinite arithmetic also
    raises, so an invalid score cannot silently win selection.
    """
    n = _integer(n, "n", minimum=1)
    q = _integer(q, "q", minimum=1)
    dimension = _integer(model_dimension, "model_dimension")
    rss = _real(rss, "rss", positive=True)
    observations = n * q
    # This equivalent form avoids underflow when rss / observations is tiny.
    score = observations * (log(rss) - log(observations)) + dimension * log(observations)
    if not np.isfinite(score):
        raise ValueError("BIC arithmetic produced a nonfinite score")
    return float(score)


def _masked_support(values, mask, *, dimension, rank, free_count, tolerance, side):
    block = _matrix(values, f"weighted_{side}")
    mask = np.asarray(mask)
    expected_shape = (dimension - rank, rank)
    if block.shape != expected_shape or mask.shape != expected_shape or mask.dtype.kind != "b":
        raise ValueError(f"weighted_{side} and boolean penalized_{side} must have shape {expected_shape}")
    # Every factor coordinate in a row is either free or penalized. Requiring
    # the full mask size also catches disagreement with the declared counts.
    if (mask.size and np.any(mask != mask[:, :1])) or int(mask.sum()) != (dimension - free_count) * rank:
        raise ValueError(f"penalized_{side} does not match the declared free-row count")
    return int(np.count_nonzero(np.abs(block[mask]) > tolerance))


def bic_score(design, response, coefficient, *, rank, free_directions=None,
              weighted_u=None, weighted_v=None, penalized_u=None, penalized_v=None,
              direct_rrr=False, design_rank=None, support_tolerance=DEFAULT_SUPPORT_TOLERANCE):
    """Score one fitted candidate from observed training data only.

    For a chart candidate, ``weighted_u`` and ``weighted_v`` are the original
    Z blocks from ``chart.unpack(state)[-2:]``, with shapes ``(p-r,r)`` and
    ``(q-r,r)``. The matching masks come from the selected chart's ``FreeRows``;
    entries in unpenalized rows never contribute to the support count. The
    dimension is ``r*(free_u+free_v-r) + support_u + support_v``.

    Set ``direct_rrr=True`` for the explicit target-only RRR branch, including
    the all-rows-unpenalized shortcut. Its dimension is
    ``k*(rank(design)+q-k)``, where ``k=min(rank, rank(design), q)`` is the
    admissible model rank, not the realized numerical rank of its response.
    A previously certified numerical ``design_rank`` can be supplied to avoid
    another design SVD. Otherwise its threshold matches ``target_rrr``.

    The absolute support tolerance is prespecified and recorded. Zero, the
    default, matches the solver's literal support diagnostic exactly; scoring
    never thresholds or changes the fitted coefficient. The caller is
    responsible for admitting only eligible terminal fits and for training-
    only fitting/stopping before invoking this function.
    """
    x = _matrix(design, "design")
    y = _matrix(response, "response")
    c = _matrix(coefficient, "coefficient")
    n, p = x.shape
    if n == 0 or p == 0 or y.shape[0] != n or y.shape[1] == 0:
        raise ValueError("design and response must be nonempty with the same row count")
    q = y.shape[1]
    if c.shape != (p, q):
        raise ValueError("coefficient must have shape (p, q)")
    rank = _integer(rank, "rank")
    if rank > min(p, q):
        raise ValueError("rank must not exceed min(p, q)")
    if not isinstance(direct_rrr, (bool, np.bool_)):
        raise ValueError("direct_rrr must be boolean")
    tolerance = _real(support_tolerance, "support_tolerance")
    blocks = (weighted_u, weighted_v, penalized_u, penalized_v)

    if direct_rrr:
        if any(value is not None for value in blocks):
            raise ValueError("direct RRR uses its full model dimension, not chart support inputs")
        if free_directions is not None:
            free = _integer_pair(free_directions, "free_directions")
            if any(not rank <= count <= size for count, size in zip(free, (p, q))):
                raise ValueError("free_directions must lie between rank and the corresponding dimension")
        if design_rank is None:
            sx = np.linalg.svd(x, compute_uv=False)
            threshold = np.finfo(float).eps * max(x.shape) * sx[0]
            design_rank = int(np.count_nonzero(sx > threshold))
        else:
            design_rank = _integer(design_rank, "design_rank")
            if design_rank > min(n, p):
                raise ValueError("design_rank exceeds the design dimensions")
        model_rank = min(rank, design_rank, q)
        dimension = model_rank * (design_rank + q - model_rank)
        free_dimension = support_u = support_v = None
        method = "target_rrr"
    else:
        if rank == 0:
            raise ValueError("a chart candidate requires positive rank")
        if design_rank is not None:
            raise ValueError("design_rank is only used to score direct RRR")
        if free_directions is None or any(value is None for value in blocks):
            raise ValueError("a chart candidate requires free_directions, weighted blocks and masks")
        free = _integer_pair(free_directions, "free_directions")
        if any(not rank <= count <= size for count, size in zip(free, (p, q))):
            raise ValueError("free_directions must lie between rank and the corresponding dimension")
        support_u = _masked_support(weighted_u, penalized_u, dimension=p, rank=rank,
                                    free_count=free[0], tolerance=tolerance, side="u")
        support_v = _masked_support(weighted_v, penalized_v, dimension=q, rank=rank,
                                    free_count=free[1], tolerance=tolerance, side="v")
        free_dimension = rank * (sum(free) - rank)
        dimension = free_dimension + support_u + support_v
        model_rank = rank
        method = "sparse_smart_v2"

    with np.errstate(over="ignore", invalid="ignore"):
        residual = y - x @ c
        rss = float(np.sum(residual * residual))
    score = bic_from_rss(rss, n=n, q=q, model_dimension=dimension)
    return BICScore(score, rss, dimension, free_dimension, support_u, support_v,
                    n, p, q, rank, model_rank, design_rank, tolerance, method)


__all__ = ["BICScore", "DEFAULT_SUPPORT_TOLERANCE", "bic_from_rss", "bic_score"]
