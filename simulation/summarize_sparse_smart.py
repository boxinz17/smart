"""Compare new-method seed runs with curves extracted from the existing paper.

No baseline estimators are invoked. Paper values are vector-digitized figure
summaries, not reconstructed raw Monte Carlo observations.
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np

from run_restricted_rrr import experiment_settings
from paper_reference import read_reference

HERE = Path(__file__).resolve().parent
DEFAULT_REFERENCE = HERE / "paper_reference" / "v1_simulation_curves.csv"


def _stats(values):
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return None, None
    return float(values.mean()), (float(values.std(ddof=1) / np.sqrt(len(values))) if len(values) > 1 else None)


def summarize(result_root, reference_path, *, model_id=0, seed_ids=(0, 1, 2, 3, 4), experiments=(0, 3)):
    model = f"model{model_id+1}"
    records = {}
    configs = set()
    implementations = set()
    for path in sorted((Path(result_root) / model).glob("exp*/SparseSMART_result_*.json")):
        record = json.loads(path.read_text())
        if record["rd_seed_id"] not in seed_ids or int(record["experiment"][3:]) - 1 not in experiments:
            continue
        if record["model"] != model or record["status"] not in ("complete", "failed", "inapplicable"):
            raise ValueError(f"Invalid model or run status in {path}")
        if (record["status"] == "complete") != bool(record["success"]):
            raise ValueError(f"Inconsistent success flag in {path}")
        if record["status"] == "complete" and (record["avg_err"] is None
                or not np.isfinite(record["avg_err"]) or record["avg_err"] < 0):
            raise ValueError(f"Missing or invalid completed-fit error in {path}")
        key = (record["experiment"], record["setting"]["suffix"], record["rd_seed_id"])
        if key in records:
            raise ValueError(f"Duplicate cell: {key}")
        configs.add(json.dumps(record["configuration"]["runner"], sort_keys=True))
        implementations.add(record["implementation_fingerprint"])
        records[key] = record
    if len(configs) > 1 or len(implementations) > 1:
        raise ValueError("Cannot pool runs with different tuning or implementations")
    reference_rows, _ = read_reference(reference_path)
    paper = [row for row in reference_rows if int(row["model_id"]) == model_id]
    rows = []
    for exp_id in experiments:
        experiment = f"exp{exp_id+1}"
        for setting in experiment_settings(model_id, exp_id):
            x = float(setting.suffix.split("=", 1)[1])
            cell = [records[(experiment, setting.suffix, seed)] for seed in seed_ids
                    if (experiment, setting.suffix, seed) in records]
            completed = [r for r in cell if r["status"] == "complete"]
            # Failures are counted explicitly. Partial errors are kept separately.
            mean, se = _stats([r["avg_err"] for r in completed])
            initial_mean, _ = _stats([r["initial_avg_err"] for r in cell if r["initial_avg_err"] is not None])
            partial_mean, _ = _stats([r["last_accepted_avg_err"] for r in cell if r["last_accepted_avg_err"] is not None])
            fit_mean, _ = _stats([r["fit_time_sec"] for r in cell if r["applicable"]])
            row = dict(model=model, experiment=experiment, setting=setting.suffix, x=x,
                expected_runs=len(seed_ids), recorded_runs=len(cell), complete=len(completed),
                failed=sum(r["status"] == "failed" for r in cell),
                inapplicable=sum(r["status"] == "inapplicable" for r in cell),
                missing=len(seed_ids)-len(cell), initial_mean=initial_mean, final_mean=mean,
                final_se=se, last_accepted_mean=partial_mean, fit_seconds_mean=fit_mean,
                source_check_bypassed=sum(bool(r.get("diagnostics", {}).get("calibration", {}).get("source_accuracy_bypassed")) for r in cell),
                failure_reasons=json.dumps(dict(Counter(r["failure_reason"] for r in cell if r["failure_reason"])),sort_keys=True))
            for method, label in (("SMART", "paper_smart"), ("SMART_fixed", "paper_smart_fixed")):
                matches = [r for r in paper if r["experiment"] == experiment
                           and r["method"] == method and abs(float(r["x"])-x) < 1e-9]
                if len(matches) != 1:
                    raise ValueError(f"Missing or duplicate paper reference: {experiment} {method} {x}")
                row[label] = float(matches[0]["mean"])
                row[label+"_se"] = float(matches[0]["se"])
            rows.append(row)
    return rows, paper, (json.loads(next(iter(configs))) if configs else None)


def write_outputs(rows, paper, config, output_dir, *, seed_ids):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "comparison.csv"
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = ["# SparseSMART pilot versus existing V1 figure results", "",
        "Only SparseSMART was run. All other methods below come from vector extraction of the existing paper PDF figures; no baseline estimator was rerun.", "",
        f"New runs use existing seed IDs {', '.join(map(str,seed_ids))}. Paper curves summarize 100 repetitions. Figure-derived values are approximate plotted summaries, not raw per-seed results. This is an exploratory comparison, not a paired significance test.", "",
        "The reported metric is ||C_hat-C_star||_F / sqrt(p*q). Tuning was fixed before the evaluation runs and uses supplied ranks/budgets; the unknown coefficient is used only to calculate errors. New means/SEs use completed runs, with failure and missing counts exposed in the table. last_accepted_mean in the CSV includes available partial fits separately.", "",
        "Noisy runs use explicit empirical mode outside the source-accuracy condition. The actual noise/gap and gate result remain in each record. Other feasibility and line-search checks remain active; no theorem coverage is claimed.", "",
        "```json", json.dumps(config, indent=2, sort_keys=True), "```", "",
        "| Setting | Completed / expected | Failed / missing | Initial error | Refined error (SE) | Paper SMART | Paper fixed-rank SMART |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    fmt = lambda value: "NA" if value is None else f"{value:.6f}"
    for row in rows:
        report.append(f"| {row['experiment']}: {row['setting']} | {row['complete']}/{row['expected_runs']} | "
            f"{row['failed']}/{row['missing']} | {fmt(row['initial_mean'])} | {fmt(row['final_mean'])} ({fmt(row['final_se'])}) | "
            f"{fmt(row['paper_smart'])} | {fmt(row['paper_smart_fixed'])} |")
    report += ["", "The pilot uses the Python repository generator, including its AR(1) predictor covariance and target noise 0.5. Model I avoids the default-sample-size ambiguity between the prose and higher-dimensional figure panels. See paper_reference/README.md for extraction provenance and precision.", ""]
    (output_dir / "comparison.md").write_text("\n".join(report))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    experiments = list(dict.fromkeys(r["experiment"] for r in rows))
    fig, axes = plt.subplots(1, len(experiments), figsize=(7 * len(experiments), 5.4), squeeze=False)
    colors = {"SMART": "#16834a", "SMART_fixed": "#657381", "RRR": "#4477aa",
              "SRRR": "#bb6688", "SOFAR": "#cc9955", "RSSVD": "#9977bb"}
    labels = {"SMART": "Paper: SMART", "SMART_fixed": "Paper: fixed-rank SMART",
              "RRR": "Paper: RRR", "SRRR": "Paper: SRRR", "SOFAR": "Paper: SOFAR", "RSSVD": "Paper: RSSVD"}
    for ax, experiment in zip(axes[0], experiments):
        cell = [r for r in rows if r["experiment"] == experiment]
        x = np.array([r["x"] for r in cell])
        for method in colors:
            ref = sorted([r for r in paper if r["experiment"] == experiment and r["method"] == method], key=lambda r:float(r["x"]))
            ax.plot([float(r["x"]) for r in ref], [float(r["mean"]) for r in ref],
                    color=colors[method], linestyle="--", linewidth=1.3,
                    alpha=1. if method.startswith("SMART") else .55, label=labels[method])
        ax.plot(x, [r["initial_mean"] for r in cell], "o:", color="#c55b0b", markersize=4, label="New: initializer")
        ax.errorbar(x, [np.nan if r["final_mean"] is None else r["final_mean"] for r in cell],
                    yerr=[0. if r["final_se"] is None else r["final_se"] for r in cell],
                    color="#101c2d", marker="o", linewidth=2.2, capsize=3, label="New: refined (mean +/- SE)")
        for xx, row in zip(x, cell):
            if row["complete"] < row["expected_runs"]:
                ax.annotate(f"{row['complete']}/{row['expected_runs']} complete", (xx, row["final_mean"] or 0), xytext=(3,8),textcoords="offset points",fontsize=8)
        ax.set_title("Sample-size sweep" if experiment == "exp1" else "Source-noise sweep", fontsize=13)
        ax.set_xlabel("Target sample size n" if experiment == "exp1" else "Source noise sigma0")
        ax.set_ylabel("Normalized Frobenius coefficient error")
        ax.set_ylim(bottom=0)
        ax.grid(alpha=.15)
        ax.spines[["top", "right"]].set_visible(False)
        if experiment == "exp1":
            ax.set_xticks(x)
    handles, labels_list = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, labels_list, loc="lower center", ncol=4, fontsize=9, frameon=False, bbox_to_anchor=(.5,.06))
    fig.suptitle(f"Model I pilot: {len(seed_ids)} seeds for SparseSMART; existing paper curves", fontsize=15)
    fig.text(.5,.015,"Paper curves: 100 repetitions, extracted from PDF vectors. New noisy fits: empirical mode; fixed tuning; no baseline reruns.",ha="center",fontsize=9)
    fig.tight_layout(rect=(0,.18,1,.92))
    fig.savefig(output_dir / "comparison.png", dpi=180)
    plt.close(fig)
    return csv_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-root", type=Path, default=HERE / "result" / "sparse_smart_pilot")
    parser.add_argument("--paper-reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--output-dir", type=Path, default=HERE / "result" / "sparse_smart_pilot" / "summary")
    parser.add_argument("--seed-count", type=int, default=5)
    args = parser.parse_args()
    if not 1 <= args.seed_count <= 100:
        parser.error("seed-count must lie between 1 and 100")
    seeds = tuple(range(args.seed_count))
    rows, paper, config = summarize(args.result_root, args.paper_reference, seed_ids=seeds)
    print(write_outputs(rows, paper, config, args.output_dir, seed_ids=seeds))


if __name__ == "__main__":
    main()
