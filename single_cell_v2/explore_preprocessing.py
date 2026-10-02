"""Exploration after the first run: which preprocessing helps SparseSMART most.

Each variant changes preprocessing only (options in preprocess.PREPROCESSING_DEFAULTS),
and every method is refitted on the same preprocessed data, so comparisons within a
variant are like for like. Variants that change the protein normalization or the
protein set are not comparable on raw RMSE; compare them through SparseSMART's
standing against the competitors and through variance explained. This is not a
preregistered analysis: variants were chosen after seeing the first run's results.

Runs on Titan only (titan/run_explore.sbatch): `pending` lists unfinished tasks and
`task` runs one (variant, fold) pair as a single-threaded Slurm step.

    python code/single_cell_v2/explore_preprocessing.py pending --folds F --out DIR
    python code/single_cell_v2/explore_preprocessing.py task --counts C --isotypes I --folds F --out DIR \
        --variant V --fold FOLD --mode production
    python code/single_cell_v2/explore_preprocessing.py table --folds F --out DIR --base BASE_RUN
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from copy import deepcopy
from pathlib import Path

VARIANTS = {
    "svd50_z": dict(rna="svd", svd_components=50, cell_zscore=True),
    "svd100_z": dict(rna="svd", svd_components=100, cell_zscore=True),
    "svd300_z": dict(rna="svd", svd_components=300, cell_zscore=True),
    "svd100": dict(rna="svd", svd_components=100),
    "svd100_z_all": dict(rna="svd", svd_components=100, cell_zscore=True, svd_fit="all"),
    "genes_z": dict(cell_zscore=True),
    "donor_center": dict(donor_center=True),
    "knn15": dict(knn_smooth=15),
    "clr_across": dict(protein_norm="clr_across"),
    "clr_across_q90_10": dict(protein_norm="clr_across", protein_min_q90=10),
    "dsb": dict(protein_norm="dsb_like"),
    # Combinations of the best single changes (second exploration run).
    "clr_across_knn15": dict(protein_norm="clr_across", knn_smooth=15),
    "clr_across_z": dict(protein_norm="clr_across", cell_zscore=True),
    "clr_across_knn15_z": dict(protein_norm="clr_across", knn_smooth=15, cell_zscore=True),
    "clr_across_r2": dict(protein_norm="clr_across"),
    "clr_across_knn15_r2": dict(protein_norm="clr_across", knn_smooth=15),
    "clr_across_z_r2": dict(protein_norm="clr_across", cell_zscore=True),
}
# Settings changed for one variant only. On kNN-smoothed (nearly collinear) genes the
# target-only Lasso did not converge within 10 minutes in the smoke run; capped, a
# non-converged Lasso drops out of that variant's comparison instead of stalling the job.
# Variants ending in _r2 fit every rank-constrained method at the selected rank + 2.
LASSO_CAP = {"lasso": dict(n_alphas=30, min_ratio=1e-4, max_iter=2000, tol=1e-6)}
RANK_PLUS2 = {"variants": {"main": dict(source_estimator="reduced_rank_ridge", rank_offset=2)}}
VARIANT_CONFIG = {"knn15": LASSO_CAP, "clr_across_knn15": LASSO_CAP, "clr_across_knn15_z": LASSO_CAP,
                  "clr_across_r2": RANK_PLUS2, "clr_across_knn15_r2": {**LASSO_CAP, **RANK_PLUS2},
                  "clr_across_z_r2": RANK_PLUS2}
COMPETITORS = ["initializer_only", "target_rrr", "target_ridge_rrr", "target_ridge", "lasso", "source_only",
               "source_subspace_rrr", "source_subspace_ridge_rrr", "ridge_to_source", "source_target_mixture",
               "nuclear_contrast", "park"]

def pending(args):
    folds = json.loads(Path(args.folds).read_text())["folds"]
    names = args.variants.split(",") if args.variants else list(VARIANTS)
    for variant in names:
        for fold in folds:
            if not (Path(args.out) / variant / "main" / fold["fold"] / "result.json").exists():
                print(variant, fold["fold"])


def task(args):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import numpy as np
    from config import CONFIG
    from data import load_extract
    from folds import load_folds
    from run_fold import run_task

    started = time.perf_counter()
    config = deepcopy(CONFIG)
    config["preprocessing"] = VARIANTS[args.variant]
    config.update(deepcopy(VARIANT_CONFIG.get(args.variant, {})))
    print(f"{args.variant}/{args.fold} {args.mode}", flush=True)
    run_task(load_extract(args.counts), load_folds(args.folds), args.fold, "main", args.mode, config,
             Path(args.out) / args.variant, isotypes=np.load(args.isotypes)["isotypes"])
    print(f"done in {time.perf_counter() - started:.0f}s", flush=True)


def _load(directory, fold):
    import numpy as np

    result = json.loads((directory / "main" / fold / "result.json").read_text())
    methods = result["methods"]
    if any("r2" not in m.get("test", {"r2": 0}) for m in methods.values()):  # base run predates r2
        saved = np.load(directory / "main" / fold / "predictions.npz")
        Y = saved["Y_test"]
        for name, method in methods.items():
            if method.get("success"):
                method["test"]["r2"] = float(1 - np.mean((saved[name] - Y) ** 2) / np.mean(Y ** 2))
    return result


def table(args):
    import numpy as np

    folds = [f["fold"] for f in json.loads(Path(args.folds).read_text())["folds"]]
    sources = {"base": Path(args.base)} if args.base else {}
    for directory in [*(args.also or []), args.out]:  # earlier runs first; this run's variants win
        sources.update({v: Path(directory) / v for v in VARIANTS if (Path(directory) / v).exists()})
    rows = []
    for variant, directory in sources.items():
        if not all((directory / "main" / f / "result.json").exists() for f in folds):
            print(f"skip {variant}: incomplete")
            continue
        results = [_load(directory, f) for f in folds]

        def metric(method, key, split="test"):
            return np.array([r["methods"][method][split][key] for r in results])

        ours = metric("sparse_smart", "rmse")
        competitors = {c: metric(c, "rmse") for c in COMPETITORS if all(r["methods"].get(c, {}).get("success") for r in results)}
        best = min(competitors, key=lambda c: competitors[c].mean())
        val_ours = metric("sparse_smart", "rmse", "validation")
        val_comp = {c: metric(c, "rmse", "validation") for c in competitors}
        val_best = min(val_comp, key=lambda c: val_comp[c].mean())
        rows.append(dict(
            variant=variant, n_genes=int(np.mean([r["preprocessing"]["n_genes"] for r in results])),
            n_proteins=int(np.mean([r["preprocessing"]["n_proteins"] for r in results])),
            ranks=[r["ranks"]["rank"] for r in results],
            sparse_rmse=float(ours.mean()), sparse_r2=float(metric("sparse_smart", "r2").mean()),
            sparse_corr=float(np.nanmean(np.array([r["methods"]["sparse_smart"]["test"]["median_protein_correlation"]
                                                    for r in results], dtype=float))),
            best_competitor=best, gap_to_best=float(ours.mean() - competitors[best].mean()),
            donors_beating_best=int(np.sum(ours < competitors[best])),
            gap_to_ridge=float(ours.mean() - competitors["target_ridge"].mean()),
            gap_to_park=float(ours.mean() - competitors["park"].mean()) if "park" in competitors else None,
            rank_among_methods=int(1 + sum(v.mean() < ours.mean() for v in competitors.values())),
            validation_gap_to_best=float(val_ours.mean() - val_comp[val_best].mean()),
            validation_best=val_best,
            selected=[r["methods"]["sparse_smart"].get("record", {}).get("selected_iteration") for r in results],
        ))
    rows.sort(key=lambda r: r["gap_to_best"])
    lines = ["| Variant | Genes | Proteins | Ranks | SparseSMART RMSE | R² | Median corr | Rank of 13 | Gap to best (which) | Donors beating best | Gap to target ridge | Gap to Park | Validation gap |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        park = f"{r['gap_to_park']:+.4f}" if r["gap_to_park"] is not None else ""
        lines.append(f"| {r['variant']} | {r['n_genes']} | {r['n_proteins']} | {min(r['ranks'])}–{max(r['ranks'])} | "
                     f"{r['sparse_rmse']:.4f} | {r['sparse_r2']:.3f} | {r['sparse_corr']:.3f} | {r['rank_among_methods']} | "
                     f"{r['gap_to_best']:+.4f} ({r['best_competitor']}) | {r['donors_beating_best']}/8 | "
                     f"{r['gap_to_ridge']:+.4f} | {park} | {r['validation_gap_to_best']:+.4f} |")
    text = "\n".join(lines) + "\n"
    (Path(args.out) / "exploration.md").write_text(text)
    (Path(args.out) / "exploration.json").write_text(json.dumps(rows, indent=1) + "\n")
    print(text)


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=("pending", "task", "table"))
    parser.add_argument("--counts", type=Path)
    parser.add_argument("--isotypes", type=Path)
    parser.add_argument("--folds", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--base", type=Path, help="original run (its main/ folder) for the baseline row")
    parser.add_argument("--also", type=Path, action="append", help="earlier exploration run to include")
    parser.add_argument("--variants")
    parser.add_argument("--variant")
    parser.add_argument("--fold")
    parser.add_argument("--mode", choices=("smoke", "production"), default="production")
    args = parser.parse_args()
    dict(pending=pending, task=task, table=table)[args.command](args)


if __name__ == "__main__":
    main()
