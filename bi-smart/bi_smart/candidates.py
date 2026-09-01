"""Candidate-library safeguards and deterministic validation selection.

This module implements the final stage of Complete automatic BI-SMART
(``alg:bismart-complete`` and Eq. ``bismart-validation-selector``).  Candidate
matrices are intentionally *not* deduplicated: two branches may yield the same
matrix but keep different labels, and the earlier label wins an exact
validation tie.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Iterable, Tuple

import numpy as np

from .types import Candidate, CandidateStatus, FoldData


def zero_candidate(
    n_features: int,
    n_responses: int,
    *,
    order_key: tuple = (),
) -> Candidate:
    """Construct the mandatory zero-matrix safeguard.

    The zero candidate is always algebraically successful.  It is useful when
    every fitted signal is weaker than validation noise and, more importantly,
    guarantees that the validation library is never empty.
    """

    # PSEUDOCODE 1: Validate the target coefficient dimensions.
    if n_features < 1 or n_responses < 1:
        raise ValueError("Zero-candidate dimensions must both be positive.")

    # PSEUDOCODE 2: Retain zero under its own label even if another branch also
    # happens to produce an all-zero coefficient matrix.
    return Candidate.successful(
        label="safeguard/zero",
        matrix=np.zeros((n_features, n_responses), dtype=float),
        order_key=order_key,
        kind="zero",
    )


def validation_loss(candidate: Candidate, validation: FoldData) -> float:
    """Return validation Frobenius loss divided by the fold size.

    Unsuccessful candidates are excluded by the complete appendix algorithm
    before validation, so passing one here is a programming error rather than
    an infinite-loss convention.
    """

    # PSEUDOCODE 1: Enforce the appendix rule that failed branches are deleted.
    if candidate.status is not CandidateStatus.SUCCESSFUL or candidate.matrix is None:
        raise ValueError("validation_loss requires a successful candidate.")

    # PSEUDOCODE 2: Check coefficient dimensions before multiplying X_val C.
    expected = (validation.n_features, validation.n_responses)
    if candidate.matrix.shape != expected:
        raise ValueError(
            f"Candidate {candidate.label!r} has shape {candidate.matrix.shape}; "
            f"validation expects {expected}."
        )

    # PSEUDOCODE 3: Evaluate Eq. (bismart-validation-selector).
    residual = validation.Y - validation.X @ candidate.matrix
    return float(np.linalg.norm(residual, ord="fro") ** 2 / validation.n_samples)


def score_and_select_candidates(
    candidates: Iterable[Candidate],
    validation: FoldData,
) -> Tuple[Candidate, Tuple[Candidate, ...]]:
    """Score successful candidates and return the first validation minimizer.

    Input order is the deterministic order required by the appendix
    (budget--partition--weight--radius--cap--iteration--safeguard).  This
    function deliberately preserves that order and uses ``numpy.argmin``, whose
    first-minimum behavior supplies the paper's tie rule.

    Returns
    -------
    selected, scored_candidates
        ``scored_candidates`` contains only successful branches.  Each returned
        candidate has a ``validation_loss`` entry added to its immutable
        metadata mapping.
    """

    # PSEUDOCODE 1: Delete every candidate marked unsuccessful, preserving the
    # relative order and retaining duplicate successful matrices.
    successful = tuple(
        candidate
        for candidate in candidates
        if candidate.status is CandidateStatus.SUCCESSFUL
    )
    if not successful:
        raise ValueError("The validation library contains no successful candidate.")

    # PSEUDOCODE 2: Compute losses without mutating caller-owned candidate data.
    losses = np.asarray(
        [validation_loss(candidate, validation) for candidate in successful],
        dtype=float,
    )
    if not np.all(np.isfinite(losses)):
        raise ValueError("Validation produced a non-finite loss.")

    scored = []
    for candidate, loss in zip(successful, losses):
        metadata = dict(candidate.metadata)
        metadata["validation_loss"] = float(loss)
        scored.append(replace(candidate, metadata=metadata))
    scored_tuple = tuple(scored)

    # PSEUDOCODE 3: np.argmin returns the first minimum, exactly matching the
    # deterministic tie convention for the already ordered candidate library.
    selected_index = int(np.argmin(losses))
    return scored_tuple[selected_index], scored_tuple


__all__ = [
    "score_and_select_candidates",
    "validation_loss",
    "zero_candidate",
]
