"""Tests for the RankSelectorRSC estimator."""

import numpy as np
from numpy.random import default_rng

from smart import RankSelectorRSC


def _generate_low_rank_data(n, p, q, r, sigma, seed):
    rng = default_rng(seed)
    U, _ = np.linalg.qr(rng.standard_normal((p, r)))
    V, _ = np.linalg.qr(rng.standard_normal((q, r)))
    D = np.diag(rng.uniform(1.0, 2.0, size=r))
    A = U @ D @ V.T
    X = rng.standard_normal((n, p))
    E = rng.standard_normal((n, q)) * sigma
    return X, X @ A + E, A


def test_rsc_recovers_rank_on_clear_signal():
    """RSC should recover the true rank with high probability in an easy regime."""
    r_true = 5
    hits = 0
    n_trials = 20
    for trial in range(n_trials):
        X, Y, _ = _generate_low_rank_data(
            n=300, p=60, q=30, r=r_true, sigma=0.5, seed=trial
        )
        r_hat = RankSelectorRSC().select_rank(X, Y)
        hits += int(r_hat == r_true)
    # Allow a small fraction of misses due to random noise.
    assert hits >= int(0.8 * n_trials), f"only {hits}/{n_trials} recovered r_true"
