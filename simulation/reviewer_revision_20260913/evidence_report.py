"""Render audited simulation evidence without loading arrays or fitting models.

Inputs are plan/audit/summary JSON, per_replication.csv, and optional saved solver
JSON. The report does not combine independent scenarios as if they were extra
Monte Carlo replications. Family summaries first average log risk ratios over
all family levels within each seed, then quantify uncertainty across seeds.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import textwrap

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "smart-revision-matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
import numpy as np


Z = 1.959963984540054
REFERENCES = ("target_ridge_rrr", "source_subspace_ridge_rrr", "initializer_only", "v2_cap_control")
OBSERVED_SUBSPACE_METHODS = frozenset(("source_subspace_rrr", "source_subspace_ridge_rrr"))
LABELS = dict(v2="Sparse SMART v2", v2_transfer_only="v2 transfer only",
    initializer_only="Initializer only", target_rrr="Target RRR",
    target_ridge_rrr="Target ridge RRR", source_subspace_rrr="Source subspace RRR",
    source_subspace_ridge_rrr="Source subspace ridge RRR", ridge_to_source="Ridge toward source",
    source_target_mixture="Source–target mixture", nuclear_contrast="Nuclear contrast",
    oracle_subspace_rrr="Clean subspace oracle", v2_active_caps="v2 active caps",
    v2_cap_control="v2 cap control", park_two_stage_nr="Park, authors' internal CV",
    park_two_stage_nr_external_validation="Park, shared holdout")
COLORS = dict(v2="#0072B2", initializer_only="#D55E00", source_subspace_ridge_rrr="#009E73",
              target_ridge_rrr="#444444", oracle_subspace_rrr="#888888",
              park_two_stage_nr_external_validation="#CC79A7", park_two_stage_nr="#CC79A7",
              v2_active_caps="#0072B2", v2_cap_control="#D55E00")
FAMILIES = dict(reference="Exact structure", source_noise="Source matrix noise",
    containment="Two-sided leakage", containment_side="One-sided leakage",
    target_specific="Target-specific directions", diffuse_alignment="Diffuse alignment",
    source_internal_gap="Source internal gaps", target_internal_gap="Target internal gaps",
    source_boundary_gap="Source boundary gap", target_boundary_gap="Target boundary gap",
    source_truncation="Source truncation", fitted_source="Fitted source sample size",
    coefficient_close="Close coefficients", spectral_shift="Shared vectors, changed spectrum",
    fitted_relationship="Fitted source: relationship", sparse_stress="Sparse stress", runtime="Dimension scaling")
METHOD_ORDER = ("v2", "v2_transfer_only", "initializer_only", "target_rrr", "target_ridge_rrr",
    "source_subspace_rrr", "source_subspace_ridge_rrr", "ridge_to_source",
    "source_target_mixture", "nuclear_contrast", "oracle_subspace_rrr",
    "park_two_stage_nr_external_validation", "park_two_stage_nr", "v2_active_caps", "v2_cap_control")


def _json(path):
    return json.loads(Path(path).read_text())


def _number(value):
    if value is None or value == "":
        return None
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def _read_rows(path):
    with Path(path).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    numeric = ("task_index", "seed", "n_train", "p", "q", "source_rank", "target_rank",
               "coefficient_rmse", "coefficient_squared_error", "population_prediction_excess",
               "validation_mse", "training_mse", "test_mse", "numerical_rank", "total_seconds",
               "n_candidates", "n_eligible", "selected_iteration", "candidate_index")
    for row in rows:
        for key in numeric:
            if key in row:
                row[key] = _number(row[key])
        for key in ("task_index", "seed", "n_train", "p", "q"):
            if row.get(key) is not None:
                row[key] = int(row[key])
    return rows


def _stats(values):
    values = np.asarray(values, dtype=float)
    n = len(values)
    if not n:
        return dict(n=0, mean=None, mcse=None, low=None, high=None)
    mean = float(np.mean(values))
    se = float(np.std(values, ddof=1)/np.sqrt(n)) if n > 1 else None
    return dict(n=n, mean=mean, mcse=se, low=None if se is None else mean-Z*se,
                high=None if se is None else mean+Z*se)


def attach_workflow_timing(root, rows, plan, plan_sha256):
    """Charge shared observed-source preparation once to methods requiring it.

    Original audited total_seconds is retained. Other methods already include
    their own source preparation, need only C0, or use labelled oracle frames.
    Missing preparation records produce unavailable workflow times for the two
    observed-subspace baselines, rather than silently assigning zero cost.
    """
    task_times = {}
    for index, task in enumerate(plan["tasks"]):
        path = root / "tasks" / f"task-{index:06d}" / "result.json"
        result = _json(path) if path.exists() else {}
        if result and (result.get("plan_sha256") != plan_sha256 or
                result.get("case", {}).get("case_id") != plan["cases"][task["case_index"]]["case_id"]):
            raise ValueError(f"Timing result identity mismatch for task {index}")
        item = dict(task_index=index, case_id=plan["cases"][task["case_index"]]["case_id"],
                    seed=task["seed"], result_json_available=bool(result))
        for metric in ("common_source_decomposition_seconds", "generation_and_source_fit_seconds"):
            value = _number(result.get(metric))
            if value is not None and value < 0:
                raise ValueError(f"Negative {metric} for task {index}")
            item[metric] = value
        task_times[index] = item
    for row in rows:
        timing = task_times[row["task_index"]]
        needs_preparation = row["method"] in OBSERVED_SUBSPACE_METHODS
        addition = timing["common_source_decomposition_seconds"] if needs_preparation else 0.
        original = row.get("total_seconds")
        row["candidate_call_seconds"] = original
        row["source_preparation_added_seconds"] = addition
        row["workflow_tuning_seconds"] = original+addition if original is not None and addition is not None else None
        row["workflow_timing_status"] = ("available" if row["workflow_tuning_seconds"] is not None else
                                         "source_preparation_unavailable" if addition is None else "candidate_call_unavailable")
    summary = dict(n_planned_tasks=len(task_times),
                   n_result_json_available=sum(item["result_json_available"] for item in task_times.values()))
    for metric in ("common_source_decomposition_seconds", "generation_and_source_fit_seconds"):
        values = [item[metric] for item in task_times.values() if item[metric] is not None]
        summary[metric] = dict(n=len(values), mean=float(np.mean(values)) if values else None,
                               median=float(np.median(values)) if values else None,
                               minimum=min(values) if values else None, maximum=max(values) if values else None,
                               n_unavailable=len(task_times)-len(values))
    summary["aggregation_scope"] = "Descriptive clocks over the planned tasks; no Monte Carlo interval pools different cases"
    summary["generation_scope"] = "Combined simulation data generation and source estimation, not an isolated source-fitting clock"
    summary["charging_rule"] = "Observed-source preparation added once only to source_subspace_rrr and source_subspace_ridge_rrr"
    return list(task_times.values()), summary


def _wilson(count, n):
    if not n:
        return None, None
    rate = count/n
    denominator = 1+Z*Z/n
    center = (rate+Z*Z/(2*n))/denominator
    width = Z*math.sqrt(rate*(1-rate)/n+Z*Z/(4*n*n))/denominator
    # The analytic interval contains the empirical rate. At the two endpoints,
    # roundoff in center +/- width can otherwise produce negative plot extents.
    low = 0. if count == 0 else max(0., center-width)
    high = 1. if count == n else min(1., center+width)
    return low, high


def _geometric(log_ratios):
    stats = _stats(log_ratios)
    return dict(n=stats["n"], geometric_ratio=None if stats["mean"] is None else math.exp(stats["mean"]),
        log_ratio_mcse=stats["mcse"], ratio_low=None if stats["low"] is None else math.exp(stats["low"]),
        ratio_high=None if stats["high"] is None else math.exp(stats["high"]))


def paired_evidence(rows):
    lookup = {(row["task_index"], row["method"]):row for row in rows}
    groups, observations = defaultdict(list), []
    for row in rows:
        for reference in REFERENCES:
            if reference == row["method"]:
                continue
            other = lookup.get((row["task_index"], reference))
            item = {key:row[key] for key in ("task_index", "seed", "case_id", "family", "level", "n_train", "method")}
            item.update(reference=reference, method_status=row["status"],
                        reference_status=other["status"] if other else "not_planned", difference=None,
                        risk_ratio=None, log_ratio=None)
            if row["status"] == "success" and other and other["status"] == "success":
                first, second = row.get("population_prediction_excess"), other.get("population_prediction_excess")
                if first is not None and second is not None:
                    item["difference"] = first-second
                    if first > 0 and second > 0:
                        item.update(risk_ratio=first/second, log_ratio=math.log(first/second))
            groups[row["case_id"], row["method"], reference].append(item)
            observations.append(item)
    summaries = []
    for members in groups.values():
        base = {key:members[0][key] for key in ("case_id", "family", "level", "n_train", "method", "reference")}
        differences = [row["difference"] for row in members if row["difference"] is not None]
        logs = [row["log_ratio"] for row in members if row["log_ratio"] is not None]
        harm = sum(value > 1e-12 for value in differences)
        low, high = _wilson(harm, len(differences))
        stats = _stats(differences)
        summaries.append(dict(base, n_planned=len(members), n_paired=len(differences),
            n_unavailable=len(members)-len(differences), n_ratio_unavailable=len(members)-len(logs),
            n_method_success=sum(row["method_status"] == "success" for row in members),
            n_reference_success=sum(row["reference_status"] == "success" for row in members),
            mean_difference=stats["mean"], difference_mcse=stats["mcse"],
            difference_low=stats["low"], difference_high=stats["high"],
            harm_frequency=harm/len(differences) if differences else None,
            harm_count=harm, harm_low=low, harm_high=high,
            mean_positive_excess=float(np.mean(np.maximum(differences, 0))) if differences else None,
            maximum_harm=max(0., max(differences)) if differences else None, **_geometric(logs)))
    # A seed, not a case/seed pair, is the independent replicate in a family summary.
    family_groups = defaultdict(list)
    for row in observations:
        family_groups[row["family"], row["n_train"], row["method"], row["reference"]].append(row)
    families = []
    for (family, n_train, method, reference), members in family_groups.items():
        expected_cases = {row["case_id"] for row in members}
        seeds = defaultdict(list)
        for row in members:
            seeds[row["seed"]].append(row)
        logs = []
        for group in seeds.values():
            valid = [row for row in group if row["log_ratio"] is not None]
            if {row["case_id"] for row in valid} == expected_cases:
                logs.append(float(np.mean([row["log_ratio"] for row in valid])))
        families.append(dict(family=family, n_train=n_train, method=method, reference=reference,
            n_cases=len(expected_cases), n_planned_seeds=len(seeds), n_complete_seeds=len(logs),
            n_unavailable_seeds=len(seeds)-len(logs), **_geometric(logs)))
    return summaries, families, observations


def method_statistics(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row["case_id"], row["method"]].append(row)
    output = []
    for members in groups.values():
        base = {key:members[0][key] for key in ("case_id", "family", "level", "n_train", "p", "q", "method")}
        counts = Counter(row["status"] for row in members)
        item = dict(base, n_planned=len(members), n_success=counts["success"],
                    n_failed=counts["failed"], n_missing=sum(counts[s] for s in ("missing_task", "missing_method", "incomplete_task")),
                    n_invalid=counts["audit_invalid"])
        for metric in ("population_prediction_excess", "coefficient_rmse", "total_seconds",
                       "candidate_call_seconds", "source_preparation_added_seconds", "workflow_tuning_seconds"):
            item[metric] = _stats([row[metric] for row in members if row["status"] == "success" and row.get(metric) is not None])
        output.append(item)
    return output


def solver_diagnostics(root, rows, plan):
    """Read JSON only. One candidate census per task/variant avoids duplication."""
    lookup = {(row["task_index"], row["method"]):row for row in rows}
    output = []
    for index, task in enumerate(plan["tasks"]):
        case = plan["cases"][task["case_index"]]
        directory = root / "tasks" / f"task-{index:06d}"
        result = _json(directory / "result.json") if (directory / "result.json").exists() else {}
        for variant in task.get("variants", ["full_caps"]):
            method = "v2" if variant == "full_caps" else "v2_"+variant
            row = lookup.get((index, method), {})
            fitted = result.get("methods", {}).get(method, {})
            candidate_path = directory / variant / "candidates.json"
            archive = _json(candidate_path) if candidate_path.exists() else {}
            candidates = archive.get("candidates", [])
            selected = next((candidate for candidate in candidates if candidate.get("index") == fitted.get("candidate_index")), {})
            metadata = selected.get("metadata", {})
            spec = fitted.get("parameters", {})
            supports, limits = metadata.get("fitted_support_counts"), spec.get("support_limits")
            free, rank = spec.get("free_directions"), spec.get("rank")
            capacity = [(dimension-count)*rank for dimension, count in zip((case["p"], case["q"]), free)] if free and rank else None
            restrictive = any(a < b for a,b in zip(limits, capacity)) if limits and capacity else None
            binding = restrictive and any(a >= b for a,b in zip(supports, limits)) if supports and limits else None
            n_candidates = len(candidates) if candidates else row.get("n_candidates")
            n_eligible = sum(candidate.get("eligible") is True for candidate in candidates) if candidates else row.get("n_eligible")
            output.append(dict(task_index=index, case_id=case["case_id"], family=case["family"],
                level=case["level"], n_train=case["n_train"], seed=task["seed"], variant=variant,
                status=row.get("status", "missing_method"),
                n_candidates=n_candidates, n_eligible=n_eligible,
                n_ineligible=n_candidates-n_eligible if n_candidates is not None and n_eligible is not None else None,
                candidate_census_available=bool(candidates),
                candidate_statuses=dict(Counter(candidate.get("status", "missing") for candidate in candidates)),
                termination_counts=dict(Counter(candidate.get("termination_reason", "missing") or "missing" for candidate in candidates)),
                selected_rrr=row.get("selected_method") == "target_rrr" if row.get("status") == "success" else None,
                selected_iteration=row.get("selected_iteration"), selected_termination=selected.get("termination_reason"),
                terminal_iteration=selected.get("n_iter"), selected_stationarity=None,
                selected_support_u=supports[0] if supports else None,
                selected_support_v=supports[1] if supports else None,
                cap_u=limits[0] if limits else None, cap_v=limits[1] if limits else None,
                restrictive_cap=restrictive, binding_cap=binding,
                candidate_anchor_switches=selected.get("numerical_work", {}).get("anchor_switches"),
                total_seconds=row.get("total_seconds")))
            history = fitted.get("history", [])
            selected_history = next((item for item in history if item.get("iteration") == fitted.get("selected_iteration")), {})
            output[-1]["selected_stationarity"] = selected_history.get("projected_gradient_norm")
    return output


def _write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key:json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value for key,value in row.items()})


def _style():
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":9, "axes.titlesize":10,
        "axes.labelsize":9, "xtick.labelsize":8, "ytick.labelsize":8,
        "legend.fontsize":8, "axes.spines.top":False, "axes.spines.right":False,
        "axes.linewidth":.7, "grid.linewidth":.45, "grid.alpha":.25,
        "pdf.fonttype":42, "ps.fonttype":42, "savefig.dpi":200})


def _label(method):
    return LABELS.get(method, method.replace("_", " "))


def _interval(ax, x, mean, low, high, *, color, label=None, marker="o", offset=0):
    if mean is None:
        return
    error = None if low is None or high is None else [[max(0., mean-low)], [max(0., high-mean)]]
    ax.errorbar([x+offset], [mean], yerr=error, fmt=marker, ms=4, color=color,
                capsize=2, lw=1, label=label)


def _finish(fig, title, development, footer):
    fig.suptitle(title, fontsize=12, x=.04, ha="left", y=.95 if development else .97)
    footer=textwrap.fill(footer,width=max(85,int(fig.get_figwidth()*20)))
    fig.text(.04, .012, footer, fontsize=7, color="#444444", va="bottom")
    if development:
        fig.text(.97, .995, "DEVELOPMENT PILOT — NOT FINAL EVIDENCE", color="#9B2226",
                 ha="right", va="top", fontsize=7, weight="bold")
    fig.subplots_adjust(top=.84, bottom=.19, left=.08, right=.98, hspace=.5, wspace=.3)


def _paired_lookup(pairs):
    return {(row["case_id"], row["method"], row["reference"]):row for row in pairs}


def plot_reference(methods, development):
    selected = ["v2", "initializer_only", "source_subspace_ridge_rrr", "target_ridge_rrr", "oracle_subspace_rrr"]
    ns = sorted({row["n_train"] for row in methods if row["family"] == "reference"})
    if not ns:
        return None
    fig, axes = plt.subplots(2, len(ns), figsize=(max(6.4, 5.6*len(ns)), 6.2), squeeze=False)
    for column,n in enumerate(ns):
        for row_index,metric in enumerate(("population_prediction_excess", "workflow_tuning_seconds")):
            ax = axes[row_index,column]
            for x,method in enumerate(selected):
                item = next((item for item in methods if item["family"] == "reference" and item["n_train"] == n and item["method"] == method), None)
                if item:
                    stats = item[metric]
                    _interval(ax, x, stats["mean"], stats["low"], stats["high"], color=COLORS.get(method,"#444444"))
            ax.set_xticks(range(len(selected)), ["v2", "Initializer", "Source ridge", "Target ridge", "Oracle"], rotation=20)
            ax.set_yscale("log")
            ax.grid(axis="y")
            ax.set_ylabel("Population excess prediction risk" if row_index == 0 else "Target-stage workflow tuning time (s)")
            ax.set_title(f"n = {n}")
    _finish(fig, "Exact structure: prediction and computational cost", development,
            "Successful-fit means and pointwise 95% Monte Carlo intervals. Source-subspace times include observed-source preparation once; counts are in the tables.")
    return fig


def plot_sweeps(cases, pairs, families, title, development, methods=None, reference="target_ridge_rrr"):
    methods = methods or ("v2", "source_subspace_ridge_rrr", "initializer_only", "oracle_subspace_rrr")
    lookup = _paired_lookup(pairs)
    ns = sorted({case["n_train"] for case in cases if case["family"] in families})
    if not ns:
        return None
    fig, axes = plt.subplots(len(ns), len(families), figsize=(max(6.5, 3.65*len(families)), 2.75*len(ns)+.8), squeeze=False)
    for row,n in enumerate(ns):
        for column,family in enumerate(families):
            ax = axes[row,column]
            chosen = [case for case in cases if case["family"] == family and case["n_train"] == n]
            if family in ("source_noise", "containment", "diffuse_alignment", "source_internal_gap", "target_internal_gap", "source_boundary_gap", "target_boundary_gap"):
                for case in cases:
                    if case["family"] == "reference" and case["n_train"] == n:
                        baseline=case.get("source_noise_sd",.01) if family == "source_noise" else 1 if "gap" in family else 0
                        # A pilot without any noise-sweep case should not imply that a sweep was run.
                        if chosen or family != "source_noise":
                            chosen.append(dict(case,level=baseline))
            chosen.sort(key=lambda case: float(case["level"]))
            if not chosen:
                ax.text(.5, .5, "Not present in audited study", ha="center", va="center", transform=ax.transAxes)
            for method in methods:
                points = []
                for x,case in enumerate(chosen):
                    item = lookup.get((case["case_id"], method, reference))
                    if item and item["geometric_ratio"] is not None:
                        _interval(ax, x, item["geometric_ratio"], item["ratio_low"], item["ratio_high"],
                                  color=COLORS.get(method,"#555555"))
                        points.append((x,item["geometric_ratio"]))
                if points:
                    ax.plot(*zip(*points), color=COLORS.get(method,"#555555"), lw=1, alpha=.85)
            ax.axhline(1, color="#555555", lw=.7, ls="--")
            ax.set_yscale("log")
            ax.set_xticks(range(len(chosen)), [f"{float(case['level']):g}" for case in chosen])
            ax.set_title(f"{FAMILIES.get(family,family)}  |  n = {n}")
            ax.set_ylabel("Paired geometric risk ratio")
            xlabel={"source_noise":"Source noise SD", "containment":"Angle (degrees)",
                    "diffuse_alignment":"Rotation strength", "fitted_source":"Source observations"}.get(family,"Gap multiplier")
            ax.set_xlabel(xlabel)
            ax.grid(axis="y")
    handles = [Line2D([0],[0],color=COLORS.get(method,"#555555"),marker="o",ms=4,lw=1,label=_label(method)) for method in methods]
    legend_columns=min(4 if fig.get_figwidth()>=9 else 2,len(handles))
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5,.075), ncol=legend_columns, frameon=False)
    footer=f"Risk relative to {_label(reference)}; below 1 favors the displayed method. Bars use independent seeds. No bars are drawn at one replicate."
    if "fitted_source" in families:
        footer += " Park uses raw source observations; other methods use the fitted source matrix."
    _finish(fig,title,development,footer)
    fig.subplots_adjust(bottom=.29 if len(ns)==1 else .22,hspace=.55)
    return fig


def plot_incremental(family_pairs, development):
    references = REFERENCES[:3]
    present = [row for row in family_pairs if row["method"] == "v2" and row["reference"] in references and row["geometric_ratio"] is not None]
    if any(row["n_train"] >= 80 for row in present):
        present=[row for row in present if row["n_train"]>=80]
    ns = sorted({row["n_train"] for row in present})
    if not ns:
        return None
    order = [family for family in FAMILIES if any(row["family"] == family for row in present)]
    fig, axes = plt.subplots(1, len(ns), figsize=(max(7., 5.*len(ns)), max(5., .35*len(order)+1.8)), squeeze=False)
    colors = ("#444444", "#009E73", "#D55E00")
    for column,n in enumerate(ns):
        ax = axes[0,column]
        for j,reference in enumerate(references):
            for y,family in enumerate(order):
                item = next((row for row in present if row["n_train"] == n and row["family"] == family and row["reference"] == reference), None)
                if item:
                    center = item["geometric_ratio"]
                    error = None if item["ratio_low"] is None else [[center-item["ratio_low"]],[item["ratio_high"]-center]]
                    ax.errorbar([center], [y+(j-1)*.2], xerr=error, fmt="o", ms=3.6, capsize=2, color=colors[j], lw=.9)
        ax.axvline(1,color="#777777",ls="--",lw=.8)
        ax.set_xscale("log")
        ax.set_yticks(range(len(order)), [FAMILIES[family] for family in order] if column == 0 else [])
        ax.invert_yaxis(); ax.grid(axis="x"); ax.set_title(f"n = {n}")
        ax.set_xlabel("v2 / comparator: geometric risk ratio")
    handles = [Line2D([0],[0],color=color,marker="o",lw=0,label=_label(reference)) for reference,color in zip(references,colors)]
    fig.legend(handles=handles,loc="lower center",bbox_to_anchor=(.5,.065),ncol=3,frameon=False)
    _finish(fig, "v2 relative to transfer and initialization baselines", development,
            "Equal weight to tested family levels within each seed; intervals across complete seeds. Below 1 favors v2; crossing 1 leaves the direction uncertain.")
    fig.subplots_adjust(left=.22 if len(ns)<3 else .17, wspace=.14)
    return fig


def _harm_cases(cases):
    levels = dict(source_noise=1., containment=90, diffuse_alignment=1., source_internal_gap=0.,
        target_internal_gap=0., source_boundary_gap=.02, target_boundary_gap=.02,
        fitted_source=100, target_specific=3, source_truncation=3)
    chosen = []
    for case in cases:
        family = case["family"]
        if family in ("reference", "coefficient_close", "spectral_shift", "fitted_relationship", "sparse_stress") or \
                family in levels and _number(case["level"]) == levels[family]:
            chosen.append(case)
    return chosen


def _harm_label(case):
    family, level = case["family"], case["level"]
    if family == "fitted_relationship":
        return "Fitted source: " + {"coefficient_close":"close coefficients",
            "spectral_shift":"spectral shift"}.get(str(level), str(level).replace("_", " "))
    if family == "reference":
        return f"Reference (source SD {case.get('source_noise_sd', .01):g})"
    templates = dict(source_noise="Source noise SD {level:g}",
        containment="Two-sided leakage {level:g}°", diffuse_alignment="Diffuse alignment {level:g}",
        source_boundary_gap="Source boundary multiplier {level:g}",
        target_boundary_gap="Target boundary multiplier {level:g}",
        fitted_source="Fitted source n₀ = {level:g}", target_specific="{level:g} target-specific directions",
        source_truncation="Source dimension s = {level:g}")
    if family in templates:
        return templates[family].format(level=float(level))
    if family in ("source_internal_gap", "target_internal_gap") and float(level) == 0:
        return "Source internal ties" if family == "source_internal_gap" else "Target internal ties"
    return FAMILIES[family] + (f" ({level})" if family == "sparse_stress" else "")


def plot_harm(cases, pairs, development):
    chosen = _harm_cases(cases)
    if any(case["n_train"] >= 80 for case in chosen):
        chosen=[case for case in chosen if case["n_train"]>=80]
    ns = sorted({case["n_train"] for case in chosen})
    if not ns:
        return None
    lookup = _paired_lookup(pairs)
    height = max(sum(case["n_train"] == n for case in chosen) for n in ns)
    fig,axes = plt.subplots(1,len(ns),figsize=(max(7.,4.9*len(ns)),max(4.6,.35*height+1.7)),squeeze=False)
    first_keys = [(case["family"], str(case["level"])) for case in chosen if case["n_train"] == ns[0]]
    for column,n in enumerate(ns):
        ax=axes[0,column]
        rows=[case for case in chosen if case["n_train"] == n]
        labels=[]
        for y,case in enumerate(rows):
            item=lookup.get((case["case_id"],"v2","target_ridge_rrr"))
            labels.append(_harm_label(case))
            if item and item["harm_frequency"] is not None:
                rate=item["harm_frequency"]
                ax.errorbar([rate],[y],xerr=[[rate-item["harm_low"]],[item["harm_high"]-rate]],fmt="o",ms=4,color=COLORS["v2"],capsize=2)
                ax.text(1.04,y,f"{item['harm_count']}/{item['n_paired']}",va="center",fontsize=7)
        same_rows = [(case["family"], str(case["level"])) for case in rows] == first_keys
        ax.set_yticks(range(len(rows)), labels if column == 0 or not same_rows else [])
        ax.invert_yaxis();ax.set_xlim(-.03,1.28)
        ax.set_xticks([0,.25,.5,.75,1]);ax.set_xlabel("Observed harm frequency")
        ax.set_title(f"n = {n}");ax.grid(axis="x")
    _finish(fig,"Negative transfer: v2 versus target ridge RRR",development,
        "Structural endpoints chosen by stress level; points are per-case harm frequencies and bars Wilson 95% intervals. Counts show harms / available pairs; full denominators are in CSV.")
    fig.subplots_adjust(left=.28,wspace=.16)
    return fig


def plot_solver(diagnostics, pairs, development):
    if not diagnostics:
        return None
    fig,axes=plt.subplots(1,3,figsize=(12,4.2))
    variants=[variant for variant in ("full_caps","active_caps","cap_control") if any(row["variant"] == variant for row in diagnostics)]
    for x,variant in enumerate(variants):
        members=[row for row in diagnostics if row["variant"] == variant]
        attempted=sum(row["n_candidates"] or 0 for row in members)
        failed=sum(row["n_ineligible"] or 0 for row in members)
        good=[row for row in members if row["status"] == "success"]
        axes[0].bar(x,failed/attempted if attempted else np.nan,color="#7E8C8D",width=.55)
        axes[0].text(x,failed/attempted if attempted else 0,f"{failed}/{attempted}",ha="center",va="bottom",fontsize=7)
        if good:
            rr=sum(row["selected_rrr"] is True for row in good)/len(good)
            zero=sum(row["selected_iteration"] == 0 and row["selected_rrr"] is False for row in good)/len(good)
            axes[1].bar(x-.15,rr,width=.3,color="#0072B2")
            axes[1].bar(x+.15,zero,width=.3,color="#D55E00")
    short=[variant.replace("_"," ") for variant in variants]
    for ax in axes[:2]:
        ax.set_xticks(range(len(variants)),short,rotation=15);ax.set_ylim(0,1.14);ax.grid(axis="y")
    axes[0].set_ylabel("Ineligible / declared candidate slots")
    axes[0].set_title("Candidate ineligibility")
    axes[1].set_title("Selected endpoint or chart iteration zero")
    axes[1].set_ylabel("Fraction of successful outputs")
    axes[1].legend(handles=[Line2D([0],[0],color="#0072B2",lw=5,label="Target RRR"),Line2D([0],[0],color="#D55E00",lw=5,label="Chart iteration 0")],frameon=False,loc="upper right")
    cap_pairs=[row for row in pairs if row["method"] == "v2_active_caps" and row["reference"] == "v2_cap_control"]
    for x,row in enumerate(cap_pairs):
        _interval(axes[2],x,row["geometric_ratio"],row["ratio_low"],row["ratio_high"],color=COLORS["v2"])
    axes[2].set_xticks(range(len(cap_pairs)),[f"{row['level']}, n={row['n_train']}" for row in cap_pairs],rotation=20)
    axes[2].axhline(1,color="#777777",ls="--",lw=.8);axes[2].set_yscale("log");axes[2].grid(axis="y")
    axes[2].set_title("Active caps / matched cap control")
    axes[2].set_ylabel("Paired geometric prediction risk ratio")
    _finish(fig,"Solver evidence and the contribution of hard caps",development,
        "Ineligible slots include unstarted fits. Selection bars are disjoint and use successful outputs as denominator. Chart iteration zero may include support projection. Fractions are descriptive, not independent-binomial inference.")
    return fig


def plot_runtime(methods, development):
    rows=[row for row in methods if row["family"] == "runtime"]
    if not rows:
        return None
    methods_to_plot=("v2","initializer_only","target_ridge_rrr","source_subspace_ridge_rrr")
    dimensions=sorted({(row["p"],row["q"]) for row in rows})
    fig,ax=plt.subplots(figsize=(7.5,4.7))
    for method in methods_to_plot:
        values=[]
        for x,(p,q) in enumerate(dimensions):
            row=next((row for row in rows if row["method"] == method and row["p"] == p and row["q"] == q),None)
            if row:
                stats=row["workflow_tuning_seconds"]
                _interval(ax,x,stats["mean"],stats["low"],stats["high"],color=COLORS[method])
                if stats["mean"] is not None:values.append((x,stats["mean"]))
        if values:ax.plot(*zip(*values),color=COLORS[method],label=_label(method),lw=1)
    ax.set_xticks(range(len(dimensions)),[f"p={p}, q={q}" for p,q in dimensions]);ax.set_yscale("log")
    ax.set_ylabel("Target-stage workflow tuning time (s)");ax.grid(axis="y");ax.legend(frameon=False)
    _finish(fig,"Dimensional computation study",development,
            "Same grids at each dimension; n=200, rank 5. Source-subspace times include observed-source preparation once. Bars: pointwise Monte Carlo intervals.")
    return fig


def _tex(value):
    return str(value).replace("\\", "\\textbackslash{}").replace("_", "\\_").replace("%", "\\%").replace("&", "\\&").replace("–", "--")


def _formatted(stats):
    if stats["mean"] is None:return "--"
    return f"{stats['mean']:.3g}" + (f" ({stats['mcse']:.2g})" if stats["mcse"] is not None else "")


def tables(prefix, methods, family_pairs, diagnostics, development):
    row_end = " " + chr(92)*2
    marker="Development pilot; one-replicate results are descriptive only." if development else "Audited study; uncertainty uses independent Monte Carlo seeds."
    rows=["% "+marker,
          "% Workflow time includes observed-source preparation once for the two source-subspace methods.",
          "% Original candidate call times and timing availability counts are retained in the CSV appendices.",
          "\\begin{tabular}{lrrrr}","\\toprule",
          "Method & Valid/planned & Prediction risk & Coefficient RMSE & Workflow (s)"+row_end,"\\midrule"]
    for n in sorted({row["n_train"] for row in methods if row["family"] == "reference"}):
        rows.append(f"\\multicolumn{{5}}{{l}}{{Exact structure, $n={n}$}}"+row_end)
        for method in METHOD_ORDER:
            item=next((row for row in methods if row["family"] == "reference" and row["n_train"] == n and row["method"] == method),None)
            if item:
                rows.append(f"{_tex(_label(method))} & {item['n_success']}/{item['n_planned']} & {_formatted(item['population_prediction_excess'])} & {_formatted(item['coefficient_rmse'])} & {_formatted(item['workflow_tuning_seconds'])}"+row_end)
        rows.append("\\midrule")
    rows[-1]="\\bottomrule";rows.append("\\end{tabular}")
    Path(str(prefix)+"-methods.tex").write_text("\n".join(rows)+"\n")
    rows=["% "+marker,"% Ratios average log risk ratios over family levels within seed, then across complete seeds.",
          "\\begin{tabular}{llrrr}","\\toprule","Family & $n$ & Seeds & v2/source ridge & 95\\% interval"+row_end,"\\midrule"]
    for item in sorted(family_pairs,key=lambda row:(row["n_train"],row["family"])):
        if item["method"] == "v2" and item["reference"] == "source_subspace_ridge_rrr":
            ratio="--" if item["geometric_ratio"] is None else f"{item['geometric_ratio']:.3f}"
            interval="--" if item["ratio_low"] is None else f"[{item['ratio_low']:.3f}, {item['ratio_high']:.3f}]"
            rows.append(f"{_tex(FAMILIES.get(item['family'],item['family']))} & {item['n_train']} & {item['n_complete_seeds']}/{item['n_planned_seeds']} & {ratio} & {interval}"+row_end)
    rows.extend(["\\bottomrule","\\end{tabular}"])
    Path(str(prefix)+"-source-advantage.tex").write_text("\n".join(rows)+"\n")
    rows=["% Descriptive census; candidate libraries are counted once per task and variant.",
          "\\begin{tabular}{lrrrrr}","\\toprule","Variant & Valid/planned & Ineligible/attempted & RRR & Iteration 0 & Binding cap"+row_end,"\\midrule"]
    for variant in ("full_caps","active_caps","cap_control"):
        members=[row for row in diagnostics if row["variant"] == variant]
        if not members:continue
        good=[row for row in members if row["status"] == "success"]
        attempted=sum(row["n_candidates"] or 0 for row in members);failed=sum(row["n_ineligible"] or 0 for row in members)
        binding=[row for row in good if row["binding_cap"] is not None]
        rows.append(f"{_tex(variant)} & {len(good)}/{len(members)} & {failed}/{attempted} & {sum(row['selected_rrr'] is True for row in good)}/{len(good)} & {sum(row['selected_iteration']==0 for row in good)}/{len(good)} & {sum(row['binding_cap'] is True for row in binding)}/{len(binding)}"+row_end)
    rows.extend(["\\bottomrule","\\end{tabular}"])
    Path(str(prefix)+"-diagnostics.tex").write_text("\n".join(rows)+"\n")


def build_report(root, output_prefix, *, runtime_root=None, allow_incomplete=False):
    root, prefix=Path(root).resolve(),Path(output_prefix).resolve()
    prefix.parent.mkdir(parents=True,exist_ok=True)
    plan,audit,summary=(_json(root/name) for name in ("plan.json","audit.json","summary.json"))
    digest=hashlib.sha256((root/"plan.json").read_bytes()).hexdigest()
    if audit.get("plan_sha256") != digest or summary.get("plan_sha256") != digest:
        raise ValueError("Report input plan fingerprints do not match audited artifacts")
    complete=audit.get("audit_passed") is True and audit.get("all_planned_tasks_complete") is True
    if not complete and not allow_incomplete:
        raise ValueError("The report requires a complete passing audit; --allow-incomplete labels all output as development")
    development=bool(plan.get("pilot")) or not complete
    rows=_read_rows(root/"per_replication.csv")
    counts=Counter(row["status"] for row in rows)
    if any(counts[status] != count for status,count in summary.get("status_counts",{}).items()):
        raise ValueError("Per-replication status counts differ from audited summary")
    task_timing,timing_summary=attach_workflow_timing(root,rows,plan,digest)
    pairs,family_pairs,observations=paired_evidence(rows)
    methods=method_statistics(rows)
    diagnostics=solver_diagnostics(root,rows,plan)
    runtime_methods=[]
    runtime_evidence=dict(status="not_supplied",root=None)
    if runtime_root:
        other=Path(runtime_root).resolve();runtime_audit=_json(other/"audit.json")
        runtime_evidence=dict(status="incomplete",root=str(other))
        if runtime_audit.get("audit_passed") and runtime_audit.get("all_planned_tasks_complete"):
            runtime_plan=_json(other/"plan.json")
            runtime_digest=hashlib.sha256((other/"plan.json").read_bytes()).hexdigest()
            if runtime_audit.get("plan_sha256") != runtime_digest:
                raise ValueError("Runtime plan fingerprint does not match audited artifact")
            runtime_rows=_read_rows(other/"per_replication.csv")
            _,runtime_timing=attach_workflow_timing(other,runtime_rows,runtime_plan,runtime_digest)
            runtime_methods=method_statistics(runtime_rows)
            runtime_evidence["timing_summary"]=runtime_timing
            runtime_evidence["status"]="audited_complete"
        elif not allow_incomplete:
            raise ValueError("Runtime study does not have a complete passing audit")
    _style()
    figures=[]
    def add(name,figure):
        if figure is not None:figures.append((name,figure))
    add("reference",plot_reference(methods,development))
    add("structure",plot_sweeps(plan["cases"],pairs,("source_noise","containment","diffuse_alignment"),
        "Robustness to source noise, leakage, and diffuse alignment",development))
    add("gaps",plot_sweeps(plan["cases"],pairs,("source_internal_gap","target_internal_gap"),
        "Internal spectral ties with fixed rank-boundary gaps",development,reference="source_subspace_ridge_rrr",
        methods=("v2","initializer_only","oracle_subspace_rrr")))
    park="park_two_stage_nr_external_validation" if any(row["method"] == "park_two_stage_nr_external_validation" for row in rows) else "park_two_stage_nr"
    add("fitted-source",plot_sweeps(plan["cases"],pairs,("fitted_source",),"Learning the source model from finite data",development,
        methods=("v2","source_subspace_ridge_rrr","initializer_only",park)))
    add("incremental",plot_incremental(family_pairs,development))
    add("harm",plot_harm(plan["cases"],pairs,development))
    add("solver",plot_solver(diagnostics,pairs,development))
    add("runtime",plot_runtime(runtime_methods or methods,development))
    output_files=[]
    with PdfPages(str(prefix)+"-figures.pdf",metadata={"Title":"Audited SMART revision numerical evidence", "Author":"SMART numerical revision"}) as book:
        for name,fig in figures:
            for extension in ("pdf","png"):
                path=Path(str(prefix)+f"-{name}.{extension}")
                fig.savefig(path,bbox_inches="tight",pad_inches=.12)
                output_files.append(str(path))
            book.savefig(fig,bbox_inches="tight",pad_inches=.12)
            plt.close(fig)
    tables(prefix,methods,family_pairs,diagnostics,development)
    for suffix,items in (("pairs",pairs),("family-pairs",family_pairs),("paired-observations",observations),("method-statistics",methods),("solver-diagnostics",diagnostics),("task-timing",task_timing)):
        _write_csv(Path(str(prefix)+f"-{suffix}.csv"),items)
    _write_csv(Path(str(prefix)+"-replications.csv"),rows)
    n200=[row for row in family_pairs if row["method"] == "v2" and row["reference"] == "source_subspace_ridge_rrr" and row["n_train"] == 200]
    interpretations=dict(n200_source_comparison_available=bool(n200),
        n200_families_improvement_interval_below_one=[row["family"] for row in n200 if row["ratio_high"] is not None and row["ratio_high"]<1],
        n200_families_harm_interval_above_one=[row["family"] for row in n200 if row["ratio_low"] is not None and row["ratio_low"]>1],
        n200_families_direction_unresolved=[row["family"] for row in n200 if row["ratio_low"] is None or row["ratio_low"]<=1<=row["ratio_high"]],
        restriction="Intervals are pointwise, not simultaneous; family-level findings are descriptive of the declared settings and do not establish uniform dominance.")
    payload=dict(schema=1,stage="development_pilot" if development else "audited_study",root=str(root),plan_sha256=digest,
        n_cases=len(plan["cases"]),n_tasks=len(plan["tasks"]),status_counts=dict(counts),runtime_evidence=runtime_evidence,
        timing_summary=timing_summary,
        provenance={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in ("plan.json","audit.json","summary.json","per_replication.csv")},
        figures=output_files,paired_summaries=pairs,family_summaries=family_pairs,interpretation=interpretations,
        conventions=dict(risk="Population prediction excess, including any fitted intercept; response noise excluded",
            ratio="Paired geometric mean of method risk / comparator risk across independent seeds",
            family="Average log ratios equally across all tested family levels within seed; only seeds complete over every level enter family intervals",
            failure="Failed, missing, invalid, and zero-risk/unavailable ratios retain explicit denominators; no estimate is substituted",
            harm="Per-case frequency of positive risk difference >1e-12 with Wilson 95% intervals",
            runtime="Target-stage workflow tuning time: original recorded candidate-library call time plus common_source_decomposition_seconds once for source_subspace_rrr and source_subspace_ridge_rrr only. Missing required preparation costs make workflow time unavailable. Source generation/fitting is separate; libraries shared by reported v2 variants are not additive",
            candidate_anchor_switches="Anchor-switch count over the entire trajectory of the selected candidate, including iterations after the selected validation checkpoint",
            iteration_zero="May include chart/support projection; it is not evidence that later optimization updates improve risk"),
        limitations=(["Development outputs do not substitute for the final replicated campaign."] if development else [])+
                    (["A complete dimensional runtime study was not supplied to this report."] if not runtime_methods else []))
    Path(str(prefix)+"-summary.json").write_text(json.dumps(payload,sort_keys=True,indent=2,allow_nan=False)+"\n")
    return dict(stage=payload["stage"],n_figures=len(figures),prefix=str(prefix),n_rows=len(rows),interpretation=interpretations)


def _read_compact_report_csv(path):
    """Decode this module's already-generated CSV cells without loading fits."""
    with Path(path).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        for key, value in row.items():
            if value == "":
                row[key] = None
            elif value in ("True", "False"):
                row[key] = value == "True"
            elif value.startswith(("{", "[")):
                row[key] = json.loads(value)
            elif (number := _number(value)) is not None:
                row[key] = int(number) if number.is_integer() else number
    return rows


def render_cached_figures(root, output_prefix, *, runtime_root=None, cached_prefix=None):
    """Regenerate figures from a complete report's compact CSVs only.

    Main task/candidate receipts and arrays are never read. Existing statistical
    CSVs, TeX tables and report-summary.json are preserved. The optional runtime
    study uses its audited CSV and result.json timing records, without arrays.
    """
    root, prefix = Path(root).resolve(), Path(output_prefix).resolve()
    source_prefix = Path(cached_prefix).resolve() if cached_prefix is not None else prefix
    audit = _json(root / "audit.json")
    plan = _json(root / "plan.json")
    digest = hashlib.sha256((root / "plan.json").read_bytes()).hexdigest()
    cached_path = Path(str(source_prefix) + "-summary.json")
    cached = _json(cached_path)
    if (audit.get("audit_passed") is not True or audit.get("all_planned_tasks_complete") is not True
            or audit.get("plan_sha256") != digest or cached.get("plan_sha256") != digest
            or cached.get("stage") != "audited_study"):
        raise ValueError("Cached figure rendering requires the matching complete passing audit")
    inputs = {str(cached_path): hashlib.sha256(cached_path.read_bytes()).hexdigest()}
    for name in ("plan.json", "audit.json", "summary.json", "per_replication.csv"):
        path = root / name
        fingerprint = hashlib.sha256(path.read_bytes()).hexdigest()
        if cached.get("provenance", {}).get(name) != fingerprint:
            raise ValueError(f"Cached report provenance mismatch: {name}")
        inputs[str(path)] = fingerprint
    loaded = {}
    for suffix in ("method-statistics", "pairs", "family-pairs", "solver-diagnostics"):
        path = Path(str(source_prefix) + f"-{suffix}.csv")
        loaded[suffix] = _read_compact_report_csv(path)
        inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    methods, pairs, family_pairs, diagnostics = (loaded[key] for key in
        ("method-statistics", "pairs", "family-pairs", "solver-diagnostics"))
    runtime_methods = []
    if cached.get("runtime_evidence", {}).get("status") == "audited_complete" and runtime_root is None:
        raise ValueError("Supply --runtime-root to retain the cached report's runtime figure")
    if runtime_root is not None:
        other = Path(runtime_root).resolve()
        runtime_plan, runtime_audit = _json(other / "plan.json"), _json(other / "audit.json")
        runtime_digest = hashlib.sha256((other / "plan.json").read_bytes()).hexdigest()
        if (runtime_audit.get("audit_passed") is not True or
                runtime_audit.get("all_planned_tasks_complete") is not True or
                runtime_audit.get("plan_sha256") != runtime_digest):
            raise ValueError("Runtime figure requires its matching complete passing audit")
        runtime_rows = _read_rows(other / "per_replication.csv")
        attach_workflow_timing(other, runtime_rows, runtime_plan, runtime_digest)
        runtime_methods = method_statistics(runtime_rows)
        runtime_inputs = [other / name for name in ("plan.json", "audit.json", "per_replication.csv")]
        runtime_inputs.extend(other / "tasks" / f"task-{index:06d}" / "result.json"
                              for index in range(len(runtime_plan["tasks"])))
        for path in runtime_inputs:
            if path.exists():
                inputs[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    _style()
    figures = [("reference", plot_reference(methods, False)),
        ("structure", plot_sweeps(plan["cases"], pairs,
            ("source_noise", "containment", "diffuse_alignment"),
            "Robustness to source noise, leakage, and diffuse alignment", False)),
        ("gaps", plot_sweeps(plan["cases"], pairs, ("source_internal_gap", "target_internal_gap"),
            "Internal spectral ties with fixed rank-boundary gaps", False,
            reference="source_subspace_ridge_rrr", methods=("v2", "initializer_only", "oracle_subspace_rrr")))]
    park = "park_two_stage_nr_external_validation" if any(
        row["method"] == "park_two_stage_nr_external_validation" for row in methods) else "park_two_stage_nr"
    figures.extend([("fitted-source", plot_sweeps(plan["cases"], pairs, ("fitted_source",),
        "Learning the source model from finite data", False,
        methods=("v2", "source_subspace_ridge_rrr", "initializer_only", park))),
        ("incremental", plot_incremental(family_pairs, False)),
        ("harm", plot_harm(plan["cases"], pairs, False)),
        ("solver", plot_solver(diagnostics, pairs, False)),
        ("runtime", plot_runtime(runtime_methods or methods, False))])
    figures = [(name, figure) for name, figure in figures if figure is not None]
    prefix.parent.mkdir(parents=True, exist_ok=True)
    output_paths = [Path(str(prefix) + "-figures.pdf")]
    with PdfPages(output_paths[0], metadata={"Title":"Audited SMART revision numerical evidence"}) as book:
        for name, figure in figures:
            for extension in ("pdf", "png"):
                path = Path(str(prefix) + f"-{name}.{extension}")
                figure.savefig(path, bbox_inches="tight", pad_inches=.12)
                output_paths.append(path)
            book.savefig(figure, bbox_inches="tight", pad_inches=.12)
            plt.close(figure)
    manifest = dict(mode="cached_figures_only", plan_sha256=digest, inputs_sha256=inputs,
        outputs_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in output_paths},
        renderer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        n_figures=len(figures), unchanged="Statistical CSVs, TeX tables and report-summary.json")
    Path(str(prefix) + "-render-provenance.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return dict(n_figures=len(figures), prefix=str(prefix), mode=manifest["mode"])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root");parser.add_argument("--output-prefix",required=True)
    parser.add_argument("--runtime-root");parser.add_argument("--allow-incomplete",action="store_true")
    parser.add_argument("--render-cached-figures", action="store_true",
                        help="Render only figures from a previously audited report's compact CSVs")
    parser.add_argument("--cached-prefix", help="Read compact report CSVs from this prefix while writing figures to --output-prefix")
    args=parser.parse_args()
    if args.render_cached_figures:
        print(json.dumps(render_cached_figures(args.root, args.output_prefix, runtime_root=args.runtime_root,
                                              cached_prefix=args.cached_prefix), sort_keys=True))
        return
    print(json.dumps(build_report(args.root,args.output_prefix,runtime_root=args.runtime_root,
                                 allow_incomplete=args.allow_incomplete),sort_keys=True))


if __name__ == "__main__":
    main()
