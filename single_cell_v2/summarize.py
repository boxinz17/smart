"""Aggregate a production run into per-donor tables and paired comparisons.

    code/.venv/bin/python code/single_cell_v2/summarize.py --run <run dir> --folds folds.json --out <dir>

Writes summary.json, summary.md and per_donor.csv. Donors are the independent
units: intervals are t-intervals across held-out donors of paired differences
against SparseSMART. Refuses incomplete or mixed-protocol runs.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from config import CONFIG, fingerprint  # noqa: E402
from folds import load_folds, tasks  # noqa: E402

REFERENCE = "sparse_smart"
ORDER = ["sparse_smart", "sparse_smart_transfer_only", "initializer_only", "target_rrr", "target_ridge_rrr",
         "target_ridge", "lasso", "source_only", "source_subspace_rrr", "source_subspace_ridge_rrr",
         "ridge_to_source", "source_target_mixture", "nuclear_contrast", "park"]


def load(run, folds, config=CONFIG):
    results, missing = {}, []
    for task in tasks(folds, config):
        path = Path(run) / task["variant"] / task["fold"] / "result.json"
        if not path.exists():
            missing.append(f"{task['variant']}/{task['fold']}")
            continue
        result = json.loads(path.read_text())
        if not result.get("complete") or result["mode"] != "production":
            raise ValueError(f"{path} is not a complete production result")
        results[(task["variant"], task["fold"])] = result
    if missing:
        raise ValueError(f"missing tasks: {missing}")
    shas = {r["config_sha256"] for r in results.values()}
    if shas != {fingerprint(config)}:
        raise ValueError(f"results use a different protocol: {sorted(shas)}")
    return results


def variant_summary(results, folds, variant):
    fold_names = [f["fold"] for f in folds["folds"]]
    labels = folds["donor_labels"]
    sizes = np.array([next(f["n_test"] for f in folds["folds"] if f["fold"] == name) for name in fold_names])
    rows, per_donor = [], []
    reference = np.array([results[(variant, name)]["methods"][REFERENCE]["test"]["rmse"] for name in fold_names])
    for method in ORDER:
        records = [results[(variant, name)]["methods"].get(method) for name in fold_names]
        if all(r is None for r in records):  # not fitted in this variant (Park et al. outside main)
            continue
        records = [r or {} for r in records]
        ok = [bool(r.get("success")) for r in records]
        rmse = np.array([r["test"]["rmse"] if good else np.nan for r, good in zip(records, ok)])
        mse = np.array([r["test"]["mse"] if good else np.nan for r, good in zip(records, ok)])
        correlation = np.array([r["test"]["median_protein_correlation"] if good else np.nan
                                for r, good in zip(records, ok)], dtype=float)
        for name, value in zip(fold_names, rmse):
            per_donor.append(dict(variant=variant, method=method, fold=name,
                                  donor=labels[next(f["test_donor"] for f in folds["folds"] if f["fold"] == name)],
                                  test_rmse=None if np.isnan(value) else float(value)))
        row = dict(method=method, n_folds_succeeded=int(sum(ok)), seconds_mean=float(np.mean(
            [r.get("elapsed_seconds", np.nan) for r in records])))
        if all(ok):
            difference = rmse - reference
            half = (stats.t.ppf(0.975, len(difference) - 1) * difference.std(ddof=1) / np.sqrt(len(difference))
                    if len(difference) > 1 else np.nan)
            row.update(mean_rmse=float(rmse.mean()), pooled_rmse=float(np.sqrt(np.sum(sizes * mse) / sizes.sum())),
                       median_protein_correlation=float(np.nanmean(correlation)),
                       mean_difference_vs_sparse_smart=float(difference.mean()),
                       difference_ci95=[float(difference.mean() - half), float(difference.mean() + half)],
                       donors_where_sparse_smart_better=int(np.sum(difference > 0)))
        rows.append(row)
    return rows, per_donor


def selection_audit(results, folds):
    audit = []
    for fold in folds["folds"]:
        r = results[("main", fold["fold"])]
        winner = r["methods"][REFERENCE].get("record", {})
        audit.append(dict(fold=fold["fold"], donor=folds["donor_labels"][fold["test_donor"]],
                          selected_method=winner.get("selected_method"),
                          selected_iteration=winner.get("selected_iteration"),
                          candidate_index=winner.get("candidate_index"), ranks=r["ranks"],
                          source_rank=r["source_fit"]["selected_rank"], n_genes=r["preprocessing"]["n_genes"],
                          status_counts=r["sparse_smart"]["status_counts"]))
    return audit


def diagnostics_summary(results, folds):
    out = []
    for fold in folds["folds"]:
        d = results[("main", fold["fold"])]["diagnostics"]
        c, a = d["containment"], d["alignment"]
        floor = c["half_split_floor"] or {}
        out.append(dict(
            fold=fold["fold"], target_rsc_rank=d["low_rank"]["target_rsc"]["rank"],
            source_rsc_rank=d["low_rank"]["source_rsc"]["rank"],
            left_cosines=c["left_cosines"], left_random_mean=c["random_left"]["mean"],
            left_half_split=floor.get("left"), right_cosines=c["right_cosines"],
            right_random_mean=c["random_right"]["mean"], right_half_split=floor.get("right"),
            left_in_leading_r0=a["left"]["in_leading_r0"], right_in_leading_r0=a["right"]["in_leading_r0"],
            left_n_for_90_percent=a["left"]["n_for_90_percent"], right_n_for_90_percent=a["right"]["n_for_90_percent"]))
    return out


def markdown(summary):
    lines = ["# NK -> ILC1 leave-one-donor-out results", "",
             f"Protocol SHA-256 `{summary['config_sha256']}`; {summary['n_folds']} held-out donors.", ""]
    for variant, rows in summary["variants"].items():
        lines += [f"## Variant `{variant}`", "",
                  "| Method | Mean test RMSE | Pooled RMSE | Difference vs SparseSMART (95% CI) | Donors where SparseSMART is better |",
                  "|---|---|---|---|---|"]
        for row in rows:
            if "mean_rmse" not in row:
                lines.append(f"| {row['method']} | failed in {summary['n_folds'] - row['n_folds_succeeded']} folds | | | |")
                continue
            low, high = row["difference_ci95"]
            lines.append(f"| {row['method']} | {row['mean_rmse']:.4f} | {row['pooled_rmse']:.4f} | "
                         f"{row['mean_difference_vs_sparse_smart']:+.4f} ({low:+.4f}, {high:+.4f}) | "
                         f"{row['donors_where_sparse_smart_better']}/{summary['n_folds']} |")
        lines.append("")
    return "\n".join(lines) + "\n"


def summarize(run, folds, out, config=CONFIG):
    results = load(run, folds, config)
    summary = dict(config_sha256=fingerprint(config), n_folds=len(folds["folds"]), variants={},
                   selection=selection_audit(results, folds), diagnostics=diagnostics_summary(results, folds))
    per_donor = []
    for variant in config["variants"]:
        summary["variants"][variant], rows = variant_summary(results, folds, variant)
        per_donor += rows
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    (out / "summary.md").write_text(markdown(summary))
    with open(out / "per_donor.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(per_donor[0]))
        writer.writeheader()
        writer.writerows(per_donor)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--folds", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    print(markdown(summarize(args.run, load_folds(args.folds), args.out)))


if __name__ == "__main__":
    main()
