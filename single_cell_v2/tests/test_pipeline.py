"""Synthetic checks: folds, per-cell transforms, no leakage, and full task runs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import modeling  # noqa: E402  (sets the import paths for the simulation packages)
from config import resolved  # noqa: E402
from data import from_arrays  # noqa: E402
from folds import build_folds, choose_holdout  # noqa: E402
from preprocess import fit_fold, per_cell  # noqa: E402
from run_fold import run_task  # noqa: E402

DONORS = {101: (150, 40), 102: (120, 35), 103: (90, 8), 104: (140, 45), 105: (100, 30), 106: (130, 38),
          107: (110, 33)}  # donor: (source cells, target cells); donor 103 is below the test threshold
TEST_CONFIG = dict(hvg_n_top_genes=150, screen_top_k=30)


def synthetic(seed=0, genes=400, proteins=20, factors=4):
    rng = np.random.default_rng(seed)
    gene_loadings = rng.normal(0, 0.6, (factors, genes)) * (rng.random(genes) < 0.3)
    source_effect = rng.normal(0, 0.5, (factors, proteins))
    target_effect = source_effect * np.array([2.0, 0.0, 1.0, 0.5])[:, None]  # same modules, new strengths
    rows, cell_type, donor = [], [], []
    for d, (n_source, n_target) in DONORS.items():
        shift = rng.normal(0, 0.1, genes)
        for kind, n, effect in (("NK", n_source, source_effect), ("ILC1", n_target, target_effect)):
            z = rng.normal(size=(n, factors))
            gene_rate = np.exp(0.5 + shift + z @ gene_loadings) * rng.uniform(0.5, 1.5, (n, 1))
            protein_rate = np.exp(2.0 + z @ effect)
            rows.append(np.hstack([rng.poisson(gene_rate), rng.poisson(protein_rate)]))
            cell_type += [kind] * n
            donor += [d] * n
    counts = np.vstack(rows).astype(float)
    feature_type = ["GEX"] * genes + ["ADT"] * proteins
    return from_arrays(counts, cell_type, donor, feature_type)


@pytest.fixture(scope="module")
def extract():
    return synthetic()


@pytest.fixture(scope="module")
def config():
    return resolved(TEST_CONFIG)


def test_choose_holdout_prefers_closest_share_then_fewer_donors():
    assert choose_holdout({1: 50, 2: 30, 3: 20}, 0.2) == [3]
    assert choose_holdout({1: 10, 2: 10, 3: 20}, 0.5) == [3]


def test_folds_hold_out_whole_donors(extract, config):
    folds = build_folds(extract, config)
    assert folds["never_tested"] == [103]
    assert [f["test_donor"] for f in folds["folds"]] == [101, 102, 104, 105, 106, 107]
    for fold in folds["folds"]:
        test = fold["test_donor"]
        assert test not in fold["validation_donors"] + fold["training_donors"] + fold["source_donors"]
        assert set(fold["validation_donors"]).isdisjoint(fold["training_donors"])
        assert set(fold["source_holdout_donors"]) < set(fold["source_donors"])
        assert abs(fold["n_validation"] / (fold["n_validation"] + fold["n_training"]) - 0.225) < 0.15


def test_per_cell_transforms(extract, config):
    gex, clr, _, _ = per_cell(extract, config["gex_target_sum"])
    totals = np.asarray(np.expm1(gex.toarray()).sum(axis=1)).ravel()
    assert np.allclose(totals, config["gex_target_sum"])
    assert np.allclose(clr.mean(axis=1), 0)


def _fitted(extract, fold, config):
    fd = fit_fold(extract, fold, config)
    coefficient, info = modeling.fit_source(fd, "reduced_rank_ridge", config)
    rank, _ = modeling.rsc(fd.X_train, fd.Y_train)
    return fd, coefficient, info, rank


def test_no_leakage_from_test_donor_or_validation_cells(extract, config):
    fold = build_folds(extract, config)["folds"][0]
    base = _fitted(extract, fold, config)
    # Replace every count of the test donor (both cell types) and of the target
    # validation cells; nothing fitted may change.
    changed = extract.counts.toarray()
    rows = (extract.donor_id == fold["test_donor"]) | (
        (extract.cell_type == "ILC1") & np.isin(extract.donor_id, fold["validation_donors"]))
    changed[rows] = np.random.default_rng(1).poisson(3.0, size=changed[rows].shape)
    other = from_arrays(changed, extract.cell_type, extract.donor_id, extract.feature_type,
                        feature_name=extract.feature_name)
    again = _fitted(other, fold, config)
    for name in ("genes", "proteins", "X_source", "Y_source", "X_train", "Y_train"):
        assert np.array_equal(getattr(base[0], name), getattr(again[0], name)), name
    assert np.array_equal(base[1], again[1])
    assert base[2]["selected_rank"] == again[2]["selected_rank"] and base[3] == again[3]
    assert not np.array_equal(base[0].Y_test, again[0].Y_test)  # the perturbation did reach test cells


@pytest.mark.parametrize("mode", ["smoke", "production"])
def test_run_task_end_to_end(extract, config, tmp_path, mode):
    folds = build_folds(extract, config)
    if mode == "production":  # shorter trajectories keep the full 211-candidate library quick
        config = resolved(dict(TEST_CONFIG, sparse_smart=dict(config["sparse_smart"], iterations=20),
                               park=dict(config["park"], lambdas=[0.1, 1.0, 10.0])))
    fold = folds["folds"][0]["fold"]
    result = run_task(extract, folds, fold, "main", mode, config, tmp_path)
    assert result["complete"]
    methods = result["methods"]
    expected = {"sparse_smart", "sparse_smart_transfer_only", "target_rrr", "target_ridge_rrr", "target_ridge",
                "source_subspace_rrr", "source_subspace_ridge_rrr", "ridge_to_source", "source_target_mixture",
                "nuclear_contrast", "initializer_only", "lasso", "source_only", "park"}
    assert set(methods) == expected
    assert methods["sparse_smart"]["success"] and methods["target_rrr"]["success"]
    assert all(("test" in m) == (mode == "production" and m.get("success")) for m in methods.values())
    assert "diagnostics" in result
    saved = json.loads((tmp_path / "main" / fold / "result.json").read_text())
    assert saved["config_sha256"] == result["config_sha256"]
    if mode == "production":
        assert result["sparse_smart"]["n_fitted"] == result["sparse_smart"]["n_candidates"] == 211
        assert (tmp_path / "main" / fold / "predictions.npz").exists()


def test_summarize_full_production_run(extract, tmp_path):
    from summarize import summarize

    config = resolved(dict(TEST_CONFIG, sparse_smart=dict(resolved()["sparse_smart"], iterations=10),
                           park=dict(resolved()["park"], lambdas=[0.1, 10.0]),
                           variants={k: v for k, v in resolved()["variants"].items() if k in ("main", "rank_plus2")}))
    folds = build_folds(extract, config)
    for variant in config["variants"]:
        for fold in folds["folds"]:
            run_task(extract, folds, fold["fold"], variant, "production", config, tmp_path / "run")
    summary = summarize(tmp_path / "run", folds, tmp_path / "summary", config)
    main = {row["method"]: row for row in summary["variants"]["main"]}
    assert main["sparse_smart"]["mean_difference_vs_sparse_smart"] == 0
    assert all(row["n_folds_succeeded"] == len(folds["folds"]) for row in main.values())
    assert (tmp_path / "summary" / "summary.md").exists() and len(summary["selection"]) == len(folds["folds"])
