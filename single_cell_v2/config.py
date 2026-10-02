"""Frozen protocol for the NK -> ILC1 leave-one-donor-out application.

Design and rationale: response_r1/APPLICATION_ANALYSIS_PLAN.md. Every task
records the SHA-256 of this dictionary, so a changed protocol is visible in
the results. Grids reused from the simulation campaigns keep their values;
the SparseSMART and initializer penalties and spectral margins are rescaled to
the data by `scale_rules` using target training cells only.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy

CONFIG = dict(
    schema_version=1,
    cell_types=dict(source="NK", target="ILC1"),
    donor_column="DonorID",
    # Donors with fewer target cells stay in training pools and are never a test fold.
    min_test_cells=15,
    # Whole donors chosen so their target-cell share of the pool is closest to this.
    validation_share=0.225,
    # Donor-blocked share of source cells held out to tune the source fit.
    source_holdout_share=0.2,
    gex_target_sum=1e4,
    hvg_n_top_genes=3000,
    screen_top_k=150,
    variants=dict(
        main=dict(source_estimator="reduced_rank_ridge", rank_offset=0),
        source_ridge=dict(source_estimator="ridge", rank_offset=0),
        rank_minus2=dict(source_estimator="reduced_rank_ridge", rank_offset=-2),
        rank_plus2=dict(source_estimator="reduced_rank_ridge", rank_offset=2),
    ),
    # Simulation source-fit grids (reviewer_revision_20260913/design.py), with
    # ranks 30 and 50 added because the source has 134 proteins.
    source_fit=dict(alphas=[0.0, 1e-4, 1e-3, 1e-2, 1e-1, 1.0],
                    ranks=[1, 3, 5, 10, 15, 20, 30, 50]),
    sparse_smart=dict(
        iterations=500,
        # The structural campaign's grids assume noise SD 0.5 and top target
        # singular value 5; penalties scale by sigma_hat/0.5, margins by d1_hat/5.
        reference_sigma=0.5,
        reference_top_singular_value=5.0,
        scaled_margins=["d_lower", "d_upper", "gap", "trial_radius"],
    ),
    comparators=dict(
        methods=["target_rrr", "target_ridge_rrr", "source_subspace_rrr",
                 "source_subspace_ridge_rrr", "ridge_to_source", "source_target_mixture",
                 "nuclear_contrast", "initializer_only"],
        # Original-design grid; the source dimension r0 is added per fold.
        source_dimensions=[1, 3, 5, 7, 10, 11, 15, 20],
    ),
    lasso=dict(n_alphas=30, min_ratio=1e-4, max_iter=20000, tol=1e-6),
    park=dict(
        # Simulation grid extended upward: its penalty sat at the grid maximum in
        # 178 of 180 fitted-source tasks.
        lambdas=[0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0, 300.0],
        max_iter=1000, tolerance=1e-6, timeout_seconds=7200,
    ),
    diagnostics=dict(max_rank_curve=20, random_reference_draws=200, seed=20261001),
    # Proteins: centered log-ratio across cells, the standard CITE-seq normalization
    # (Stoeckius et al., 2017; Seurat), with constants from training cells. The first
    # protocol (run sc-lodo-20261001-v1) used within-cell CLR; see
    # response_r1/APPLICATION_PRESENTATION_PLAN.md.
    preprocessing=dict(protein_norm="clr_across"),
    # Smoke mode checks that every step runs; it never evaluates test cells.
    smoke=dict(sparse_smart_candidates=[0, 1, 2, 36, 105, 210], iterations=30,
               park_lambdas=[0.1, 10.0]),
)


def resolved(overrides=None):
    """Return a deep copy of CONFIG with top-level overrides (used by tests)."""
    config = deepcopy(CONFIG)
    for key, value in (overrides or {}).items():
        if key not in config:
            raise KeyError(f"unknown config key: {key}")
        config[key] = deepcopy(value)
    return config


def fingerprint(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
