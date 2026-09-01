"""Public API and deterministic validation tests."""

import numpy as np
import pytest

import bi_smart
from bi_smart import BISMART, BISMARTConfig, Candidate, FoldData
from bi_smart.candidates import score_and_select_candidates, zero_candidate


def test_public_version_and_estimator_exports():
    assert bi_smart.__version__ == "0.1.0.dev0"
    assert bi_smart.BISMART is BISMART


def test_validation_keeps_first_candidate_on_exact_tie():
    fold = FoldData(
        X=np.eye(2),
        Y=np.zeros((2, 1)),
        name="validation",
    )
    first = Candidate.successful(
        label="first",
        matrix=np.zeros((2, 1)),
        kind="duplicate_a",
    )
    second = Candidate.successful(
        label="second",
        matrix=np.zeros((2, 1)),
        kind="duplicate_b",
    )

    selected, scored = score_and_select_candidates((first, second), fold)

    assert selected.label == "first"
    assert len(scored) == 2  # Duplicate matrices retain distinct labels.
    assert scored[0].metadata["validation_loss"] == 0.0
    assert scored[1].metadata["validation_loss"] == 0.0


def test_unfitted_estimator_and_zero_safeguard():
    config = BISMARTConfig(
        target_rank=1,
        source_rank=1,
        source_error_bound=0.0,
        budget_path=((1, 1),),
        source_weights=(1.0,),
    )
    model = BISMART(np.array([[2.0]]), config)

    with pytest.raises(RuntimeError, match="not been fitted"):
        model.get_estimates()

    candidate = zero_candidate(3, 2)
    assert candidate.label == "safeguard/zero"
    assert candidate.matrix.shape == (3, 2)
    assert np.count_nonzero(candidate.matrix) == 0
