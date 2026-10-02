"""Per-fold preprocessing fitted on the fold's fitting cells only.

Fitting cells are the source cells of every donor except the test donor and
the target cells of the training donors. Validation and test cells are only
transformed. Steps follow v1 (normalize, log1p, 3,000 highly variable genes,
zero-variance filter, scaling, marginal screening with top 150 genes per
population), except that each statistic now comes from fitting cells, both
modalities start from raw counts, and each population is centered by its own
fitting-cell means (the intercept is the target training mean).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import anndata as ad
import numpy as np
import scanpy as sc
import scipy.sparse as sp

from smart import MultitaskMarginalRegression


@dataclass
class FoldData:
    fold: dict
    genes: np.ndarray
    proteins: np.ndarray
    X_source: np.ndarray
    Y_source: np.ndarray
    source_donor: np.ndarray
    train_donor: np.ndarray
    X_train: np.ndarray
    Y_train: np.ndarray
    X_val: np.ndarray
    Y_val: np.ndarray
    X_test: np.ndarray
    Y_test: np.ndarray
    test_cells: np.ndarray
    stats: dict = field(default_factory=dict)


def per_cell(extract, target_sum):
    """Library-size normalization and log1p for genes; CLR for proteins.

    Each cell is transformed using only its own counts.
    """
    gex_cols = extract.feature_type == "GEX"
    adt_cols = extract.feature_type == "ADT"
    gex = sp.csr_matrix(extract.counts[:, gex_cols], dtype=np.float64)
    totals = np.asarray(gex.sum(axis=1)).ravel()
    if np.any(totals <= 0):
        raise ValueError("a cell has no gene counts")
    gex = sp.csr_matrix(sp.diags(target_sum / totals) @ gex)
    gex.data = np.log1p(gex.data)
    log_adt = np.log1p(np.asarray(extract.counts[:, adt_cols].todense(), dtype=np.float64))
    clr = log_adt - log_adt.mean(axis=1, keepdims=True)
    return gex, clr, extract.feature_name[gex_cols], extract.feature_name[adt_cols]


def marginal_top(X, Y, k):
    """v1's screening rule: rank genes by the l2 norm of their Pearson correlations."""
    model = MultitaskMarginalRegression(X, Y, marginal_method="pearson")
    model.compute_marginal_scores()
    return model.ranked_features[:k]


def cell_masks(extract, fold, config):
    source, target = config["cell_types"]["source"], config["cell_types"]["target"]
    donor = extract.donor_id
    is_source, is_target = extract.cell_type == source, extract.cell_type == target
    masks = dict(
        source=is_source & np.isin(donor, fold["source_donors"]),
        train=is_target & np.isin(donor, fold["training_donors"]),
        val=is_target & np.isin(donor, fold["validation_donors"]),
        test=is_target & (donor == fold["test_donor"]),
    )
    if np.any(masks["source"] & np.isin(donor, [fold["test_donor"]])):
        raise AssertionError("test donor in source cells")
    if masks["train"].sum() == 0 or masks["val"].sum() == 0 or masks["test"].sum() == 0:
        raise ValueError("empty training, validation or test set")
    masks["fit"] = masks["source"] | masks["train"]
    return masks


# Alternatives explored after the first run (APPLICATION_ANALYSIS_PLAN.md is the original
# protocol). The defaults reproduce it. Options marked "transductive" use the RNA of
# validation and test cells (never their proteins).
PREPROCESSING_DEFAULTS = dict(
    rna="screened_genes",      # "svd": truncated-SVD components of the standardized HVG matrix
    svd_components=100,
    svd_fit="fitting",         # "all": fit the SVD on every cell's RNA (transductive)
    cell_zscore=False,         # z-score each cell's feature vector (challenge winner, scLinear)
    donor_center=False,        # center features within each donor and cell type (transductive)
    knn_smooth=0,              # average RNA over k neighbours within donor and cell type (transductive)
    protein_norm="clr_cell",   # "clr_across" (Seurat, per protein across cells) or "dsb_like"
    protein_min_q90=None,      # keep proteins whose 90th-percentile count in fitting cells reaches this
)


def preprocessing_options(config):
    unknown = set(config.get("preprocessing", {})) - set(PREPROCESSING_DEFAULTS)
    if unknown:
        raise KeyError(f"unknown preprocessing options: {sorted(unknown)}")
    return {**PREPROCESSING_DEFAULTS, **config.get("preprocessing", {})}


def knn_smooth(X, groups, fit_rows, k, n_components=50):
    """Average each cell over its k nearest neighbours (itself included) within its group."""
    from sklearn.neighbors import NearestNeighbors
    from sklearn.utils.extmath import randomized_svd

    mean, sd = X[fit_rows].mean(axis=0), X[fit_rows].std(axis=0)
    Z = (X - mean) / np.where(sd > 0, sd, 1.0)
    _, _, vt = randomized_svd(Z[fit_rows], n_components, random_state=0)
    scores = Z @ vt.T
    smoothed = np.empty_like(X)
    for group in np.unique(groups):
        rows = np.flatnonzero(groups == group)
        neighbours = NearestNeighbors(n_neighbors=min(k, len(rows))).fit(scores[rows])
        index = neighbours.kneighbors(scores[rows], return_distance=False)
        smoothed[rows] = X[rows][index].mean(axis=1)
    return smoothed


def dsb_like(counts, fit_rows, isotypes):
    """dsb's model-negative normalization, reimplemented (Mulè et al., 2022).

    Step 1 standardizes each protein by the background component of a two-part
    Gaussian mixture of its log(count + 10), dsb's default pseudocount (fitted on
    fitting cells). Step 2 forms a
    per-cell technical factor, the first principal component of the cell's
    background level and its isotype values, and regresses it out of every
    protein (coefficients from fitting cells). The isotype values in this file
    are already normalized, so step 2 approximates dsb's use of raw isotype counts.
    """
    from sklearn.mixture import GaussianMixture

    logs = np.log(counts + 10.0)
    centre, spread = np.empty(logs.shape[1]), np.empty(logs.shape[1])
    for j in range(logs.shape[1]):
        values = logs[fit_rows, j].reshape(-1, 1)
        if np.ptp(values) == 0:
            centre[j], spread[j] = values[0, 0], 1.0
            continue
        # reg_covar floors the spread at 0.1, about one count above zero on this scale;
        # without it, proteins that are almost always zero get a near-zero spread.
        mixture = GaussianMixture(2, reg_covar=0.01, random_state=0).fit(values)
        low = int(np.argmin(mixture.means_.ravel()))
        centre[j] = mixture.means_.ravel()[low]
        spread[j] = np.sqrt(mixture.covariances_.ravel()[low])
    standardized = (logs - centre) / spread
    background = np.array([GaussianMixture(2, random_state=0).fit(row.reshape(-1, 1)).means_.min()
                           for row in standardized])
    iso_sd = isotypes[fit_rows].std(axis=0)
    iso = (isotypes - isotypes[fit_rows].mean(axis=0)) / np.where(iso_sd > 0, iso_sd, 1.0)
    technical = np.column_stack([(background - background[fit_rows].mean()) / background[fit_rows].std(), iso])
    technical -= technical[fit_rows].mean(axis=0)
    _, _, vt = np.linalg.svd(technical[fit_rows], full_matrices=False)
    factor = technical @ vt[0]
    slopes = factor[fit_rows] @ (standardized[fit_rows] - standardized[fit_rows].mean(axis=0)) / (factor[fit_rows] @ factor[fit_rows])
    return standardized - np.outer(factor, slopes)


def protein_matrix(extract, clr_cell, masks, options, isotypes):
    counts = np.asarray(extract.counts[:, extract.feature_type == "ADT"].todense(), dtype=np.float64)
    if options["protein_norm"] == "clr_cell":
        Y = clr_cell
    elif options["protein_norm"] == "clr_across":
        geometric = np.exp(np.log1p(counts[masks["fit"]]).mean(axis=0))
        Y = np.log1p(counts / geometric)
    elif options["protein_norm"] == "dsb_like":
        if isotypes is None:
            raise ValueError("dsb_like needs the isotype values")
        Y = dsb_like(counts, masks["fit"], isotypes)
    else:
        raise ValueError(f"unknown protein_norm: {options['protein_norm']}")
    keep = np.ones(Y.shape[1], dtype=bool)
    if options["protein_min_q90"] is not None:
        keep = np.quantile(counts[masks["fit"]], 0.9, axis=0) >= options["protein_min_q90"]
    return Y, keep


def fit_fold(extract, fold, config, transformed=None, isotypes=None):
    """Return centered fold matrices; `transformed` reuses per_cell output."""
    options = preprocessing_options(config)
    masks = cell_masks(extract, fold, config)
    gex, clr, gene_names, protein_names = transformed or per_cell(extract, config["gex_target_sum"])
    stats = {f"n_{name}_cells": int(mask.sum()) for name, mask in masks.items()}
    stats["preprocessing"] = options
    groups = np.char.add(extract.cell_type.astype(str), extract.donor_id.astype(str))
    cells = masks["fit"] | masks["val"] | masks["test"]

    # Highly variable genes on fitting cells (v1 used all 90,261 cells).
    fit_gex = ad.AnnData(sp.csr_matrix(gex[masks["fit"]]))
    fit_gex.uns["log1p"] = {"base": None}
    sc.pp.highly_variable_genes(fit_gex, n_top_genes=config["hvg_n_top_genes"], flavor="seurat")
    hvg = np.flatnonzero(fit_gex.var["highly_variable"].to_numpy())
    X = np.asarray(gex[:, hvg].todense())
    stats["n_hvg"] = int(len(hvg))
    if options["knn_smooth"]:
        X[cells] = knn_smooth(X[cells], groups[cells], masks["fit"][cells], options["knn_smooth"])

    Y, keep_proteins = protein_matrix(extract, clr, masks, options, isotypes)

    # Drop genes and proteins with no variation in either population's fitting cells.
    keep_genes = (X[masks["source"]].var(axis=0) > 0) & (X[masks["train"]].var(axis=0) > 0)
    keep_proteins &= (Y[masks["source"]].var(axis=0) > 0) & (Y[masks["train"]].var(axis=0) > 0)
    X, Y = X[:, keep_genes], Y[:, keep_proteins]
    genes, proteins = gene_names[hvg][keep_genes], protein_names[keep_proteins]
    stats.update(n_genes_nonconstant=int(keep_genes.sum()), n_proteins=int(keep_proteins.sum()))

    if options["rna"] == "screened_genes":
        X = X / X[masks["fit"]].std(axis=0, ddof=1)
        k = config["screen_top_k"]
        top_source = marginal_top(X[masks["source"]], Y[masks["source"]], k)
        top_target = marginal_top(X[masks["train"]], Y[masks["train"]], k)
        selected = np.union1d(top_source, top_target)
        X, genes = X[:, selected], genes[selected]
        stats.update(n_genes=int(len(selected)),
                     screened_overlap=int(len(np.intersect1d(top_source, top_target))))
    elif options["rna"] == "svd":
        from sklearn.utils.extmath import randomized_svd

        Z = (X - X[masks["fit"]].mean(axis=0)) / X[masks["fit"]].std(axis=0, ddof=1)
        rows = masks["fit"] if options["svd_fit"] == "fitting" else cells
        _, _, vt = randomized_svd(Z[rows], options["svd_components"], random_state=0)
        X = Z @ vt.T
        genes = np.array([f"SVD{i + 1}" for i in range(X.shape[1])])
        stats["n_genes"] = int(X.shape[1])
    else:
        raise ValueError(f"unknown rna option: {options['rna']}")

    if options["cell_zscore"]:
        X = (X - X.mean(axis=1, keepdims=True)) / X.std(axis=1, keepdims=True)
    if options["donor_center"]:
        for group in np.unique(groups[cells]):
            rows = np.flatnonzero((groups == group) & cells)
            X[rows] -= X[rows].mean(axis=0)

    mean_source = (X[masks["source"]].mean(axis=0), Y[masks["source"]].mean(axis=0))
    mean_target = (X[masks["train"]].mean(axis=0), Y[masks["train"]].mean(axis=0))

    def target(name):
        return X[masks[name]] - mean_target[0], Y[masks[name]] - mean_target[1]

    X_train, Y_train = target("train")
    X_val, Y_val = target("val")
    X_test, Y_test = target("test")
    return FoldData(
        fold=fold, genes=genes, proteins=proteins,
        X_source=X[masks["source"]] - mean_source[0], Y_source=Y[masks["source"]] - mean_source[1],
        source_donor=extract.donor_id[masks["source"]], train_donor=extract.donor_id[masks["train"]],
        X_train=X_train, Y_train=Y_train, X_val=X_val, Y_val=Y_val, X_test=X_test, Y_test=Y_test,
        test_cells=np.flatnonzero(masks["test"]), stats=stats)
