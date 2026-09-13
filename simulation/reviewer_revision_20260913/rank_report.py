"""Report a completed operational-rank study from audited compact artifacts.

No fitting code or array archives are imported or read. Optional audited task
result.json files supply recorded fitting clocks that the original CSV omits.
Missing task JSON leaves these clocks unavailable; wall time is never used as
a substitute for fitting-call time. Initializer passes have an explicitly
labelled wall clock because no separate complete fitting-call clock was saved.
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

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "smart-rank-report-matplotlib"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch
import numpy as np


Z = 1.959963984540054
V2 = ("v2", "v2_transfer_only", "initializer_only")
SOURCE_METHODS = ("source_subspace_rrr", "source_subspace_ridge_rrr")
REFERENCES = ("target_ridge_rrr", "source_subspace_ridge_rrr", "initializer_only")
PRIMARY = ("v2", "initializer_only", "source_subspace_ridge_rrr", "target_ridge_rrr")
LABELS = dict(v2="v2", v2_transfer_only="v2 transfer only", initializer_only="Initializer",
    source_subspace_ridge_rrr="Source ridge RRR", source_subspace_rrr="Source RRR",
    target_ridge_rrr="Target ridge RRR", target_rrr="Target RRR",
    oracle_subspace_rrr="Clean-source oracle", ridge_to_source="Ridge to source",
    source_target_mixture="Source–target mixture", nuclear_contrast="Nuclear contrast")
COLORS = ("#0072B2", "#D55E00", "#009E73", "#444444")
STATUSES = {"success", "failed", "audit_invalid", "missing_task", "incomplete_task", "missing_method"}


def _json(path):
    return json.loads(Path(path).read_text())


def _number(value):
    if value is None or value == "":
        return None
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def _stats(values):
    values = np.asarray(values, dtype=float)
    n = len(values)
    mean = float(values.mean()) if n else None
    se = float(values.std(ddof=1)/math.sqrt(n)) if n > 1 else None
    return dict(n=n, mean=mean, mcse=se, low=None if se is None else mean-Z*se,
                high=None if se is None else mean+Z*se)


def _source_value(value):
    if isinstance(value, str):
        value = json.loads(value) if value else None
    if value is None:
        return None
    values = list(value) if isinstance(value, (tuple, list)) else [value]
    if not values or any(_number(item) is None or float(item) <= 0 or int(item) != float(item) for item in values):
        raise ValueError("Invalid selected source dimension")
    return int(values[0]) if len(set(values)) == 1 else "×".join(str(int(item)) for item in values)


def _source_relevant(row):
    return row["method"] in SOURCE_METHODS or (row["method"] in V2 and row.get("selected_method") != "target_rrr")


def _read_rows(path):
    with Path(path).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    integers = ("task_index", "seed", "n_train", "p", "q", "selected_fitted_rank", "numerical_rank")
    numbers = ("population_prediction_excess", "coefficient_rmse", "total_seconds")
    for row in rows:
        for name in (*integers, *numbers):
            value = _number(row.get(name))
            if name in integers and value is not None and value != int(value):
                raise ValueError(f"Noninteger {name}")
            row[name] = int(value) if name in integers and value is not None else value
        row["selected_source_dimension"] = _source_value(row.get("selected_source_dimension"))
    return rows


def _attach_timing(root, plan, rows, digest):
    """Use only complete, matching, audited task metadata; never load NPZ files."""
    by_task = defaultdict(list)
    for row in rows:
        by_task[row["task_index"]].append(row)
    provenance = {}
    for index, task in enumerate(plan["tasks"]):
        path = root / "rank_tasks" / f"task-{index:06d}" / "result.json"
        result = _json(path) if path.exists() else None
        if result is not None:
            if (result.get("complete") is not True or result.get("plan_sha256") != digest or
                    result.get("task") != task or result.get("case") != plan["cases"][task["case_index"]] or
                    set(result.get("methods", {})) != set(task["expected_methods"])):
                raise ValueError(f"Timing metadata identity or method inventory mismatch: task {index}")
            provenance[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
        for row in by_task[index]:
            row.update(fitting_call_seconds=None, source_preparation_added_seconds=None,
                       tuning_seconds=None, tuning_scope="unavailable_missing_task_json")
            if result is None:
                continue
            recorded = result["methods"][row["method"]]
            if (recorded.get("success") is True) != (row["status"] == "success"):
                raise ValueError(f"Timing metadata success differs from audited CSV: task {index}")
            if row["status"] == "success":
                value = _number(recorded.get("metrics", {}).get("population_prediction_excess"))
                if value is None or not math.isclose(value, row["population_prediction_excess"], rel_tol=2e-9, abs_tol=2e-11):
                    raise ValueError(f"Timing metadata selected risk differs from audited CSV: task {index}")
            method = row["method"]
            if method == "initializer_only":
                clock = _number(recorded.get("runtime", {}).get("initializer_pass_wall_seconds"))
                scope = "initializer_pass_wall_time"
            else:
                clock = _number(recorded.get("runtime", {}).get("fitting_call_seconds")) if method in V2 else _number(recorded.get("fitting_call_seconds"))
                scope = "fitting_calls_plus_required_source_preparation"
            preparation = _number(result.get("shared_source_decomposition_seconds")) if method in SOURCE_METHODS else 0.
            if clock is None or clock < 0 or preparation is None or preparation < 0:
                raise ValueError(f"Invalid recorded tuning clock: task {index}, {method}")
            wall = row.get("total_seconds")
            if wall is not None and clock > wall + 1e-7*max(1., wall):
                raise ValueError(f"Fitting/pass clock exceeds audited wall clock: task {index}, {method}")
            row.update(fitting_call_seconds=None if method == "initializer_only" else clock,
                       source_preparation_added_seconds=preparation, tuning_seconds=clock+preparation,
                       tuning_scope=scope)
    return provenance


def load_study(root):
    root = Path(root).resolve()
    names = ("rank-plan.json", "rank-audit.json", "rank-summary.json", "rank-per-replication.csv")
    plan, audit, summary = (_json(root/name) for name in names[:3])
    if not plan.get("cases") or not plan.get("tasks") or len({case["case_id"] for case in plan["cases"]}) != len(plan["cases"]):
        raise ValueError("Rank report requires nonempty, uniquely named planned cases and tasks")
    if len({(task["case_index"], task["seed"]) for task in plan["tasks"]}) != len(plan["tasks"]):
        raise ValueError("A case/seed pair must be a single independent replication")
    digest = hashlib.sha256((root/names[0]).read_bytes()).hexdigest()
    if audit.get("audit_passed") is not True or audit.get("all_planned_tasks_complete") is not True or audit.get("issues"):
        raise ValueError("Rank report requires a completed passing rank audit")
    if audit.get("coverage_scope") == "task_subset" or audit.get("plan_sha256") != digest:
        raise ValueError("Rank audit plan hash or coverage mismatch")
    if "plan_sha256" in summary and summary["plan_sha256"] != digest:
        raise ValueError("Rank summary plan hash mismatch")
    inventory = audit.get("tasks", [])
    if (Counter(item.get("task_index") for item in inventory) != Counter(range(len(plan["tasks"]))) or
            any(item.get("state") != "complete" for item in inventory)):
        raise ValueError("Rank audit task inventory is incomplete or duplicated")
    for item in inventory:
        case = plan["cases"][plan["tasks"][item["task_index"]]["case_index"]]
        if item.get("case_id") != case["case_id"]:
            raise ValueError("Rank audit task case identity mismatch")
    rows = _read_rows(root/names[3])
    expected = Counter((index, method) for index, task in enumerate(plan["tasks"]) for method in task["expected_methods"])
    if Counter((row["task_index"], row["method"]) for row in rows) != expected:
        raise ValueError("Rank CSV method coverage is missing, duplicated, or unplanned")
    if (summary.get("n_planned_tasks") != len(plan["tasks"]) or summary.get("n_planned_method_outcomes") != len(rows) or
            Counter(row["status"] for row in rows) != Counter(summary.get("status_counts", {}))):
        raise ValueError("Rank CSV counts differ from audited summary")
    for row in rows:
        task = plan["tasks"][row["task_index"]]
        case = plan["cases"][task["case_index"]]
        if row["seed"] != task["seed"] or row["case_id"] != case["case_id"] or any(row[name] != case[name] for name in ("n_train", "p", "q")):
            raise ValueError("Rank CSV task identity mismatch")
        if row["status"] not in STATUSES:
            raise ValueError("Unrecognized rank outcome status")
        if row["status"] == "success" and any(row.get(name) is None or row[name] < 0 for name in ("population_prediction_excess", "coefficient_rmse", "numerical_rank")):
            raise ValueError("Successful rank outcome has unavailable or invalid metrics")
        if row["status"] in {"audit_invalid", "missing_task", "incomplete_task", "missing_method"}:
            raise ValueError("A passing rank audit cannot contain unaudited or missing outcomes")
    timing = _attach_timing(root, plan, rows, digest)
    provenance = {name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in names}
    provenance.update(timing)
    return plan, rows, provenance


def summarize_rows(plan, rows):
    groups = defaultdict(list)
    lookup = {(row["task_index"], row["method"]): row for row in rows}
    for row in rows:
        groups[row["case_id"], row["method"]].append(row)
    methods, pairs, selections = [], [], []
    for (case_id, method), members in groups.items():
        success = [row for row in members if row["status"] == "success"]
        valid_clocks = [row for row in members if row["status"] in ("success", "failed") and row.get("tuning_seconds") is not None]
        base = dict(case_id=case_id, method=method, family=members[0].get("family"), n_train=members[0]["n_train"])
        n_source = sum(_source_relevant(row) for row in success)
        methods.append(dict(base, n_planned=len(members), n_success=len(success),
            n_failed=sum(row["status"] == "failed" for row in members),
            n_unavailable=sum(row["status"] not in ("success", "failed") for row in members),
            n_timing_unavailable=len(members)-len(valid_clocks), source_applicable_successes=n_source,
            source_irrelevant_successes=len(success)-n_source,
            risk=_stats([row["population_prediction_excess"] for row in success]),
            coefficient_rmse=_stats([row["coefficient_rmse"] for row in success if row.get("coefficient_rmse") is not None]),
            tuning_seconds=_stats([row["tuning_seconds"] for row in valid_clocks]),
            tuning_scopes=sorted({row.get("tuning_scope", "unspecified") for row in valid_clocks}),
            wall_seconds=_stats([row["total_seconds"] for row in members if row["status"] in ("success", "failed") and row.get("total_seconds") is not None])))
        for dimension in ("fitted_rank", "source_dimension", "numerical_rank"):
            relevant = [row for row in success if dimension != "source_dimension" or _source_relevant(row)]
            if dimension == "fitted_rank" and method in ("ridge_to_source", "nuclear_contrast"):
                relevant = []
            key = "numerical_rank" if dimension == "numerical_rank" else "selected_"+dimension
            values = [_source_value(row.get(key)) if dimension == "source_dimension" else row.get(key) for row in relevant]
            counts = Counter(value for value in values if value is not None)
            for value, count in sorted(counts.items(), key=lambda item: str(item[0])):
                selections.append(dict(base, dimension=dimension, value=value, count=count,
                    n_planned=len(members), n_success=len(success), n_applicable=len(relevant),
                    n_irrelevant=len(success)-len(relevant), n_missing=sum(value is None for value in values)))
        if method not in ("v2", "v2_transfer_only"):
            continue
        for reference in REFERENCES:
            differences, logs, reference_success = [], [], 0
            for row in members:
                other = lookup.get((row["task_index"], reference))
                reference_success += bool(other and other["status"] == "success")
                if row["status"] == "success" and other and other["status"] == "success":
                    first, second = row["population_prediction_excess"], other["population_prediction_excess"]
                    differences.append(first-second)
                    if first > 0 and second > 0:
                        logs.append(math.log(first/second))
            delta, log_stats = _stats(differences), _stats(logs)
            pairs.append(dict(base, reference=reference, n_planned=len(members), n_paired=len(differences),
                n_unavailable=len(members)-len(differences), n_ratio_available=len(logs),
                n_ratio_unavailable=len(members)-len(logs), n_method_success=len(success),
                n_reference_success=reference_success, mean_difference=delta["mean"], difference_mcse=delta["mcse"],
                difference_low=delta["low"], difference_high=delta["high"], harm_count=sum(value > 1e-12 for value in differences),
                mean_positive_excess=float(np.mean(np.maximum(differences, 0))) if differences else None,
                geometric_ratio=math.exp(log_stats["mean"]) if logs else None, log_ratio_mcse=log_stats["mcse"],
                ratio_low=math.exp(log_stats["low"]) if log_stats["low"] is not None else None,
                ratio_high=math.exp(log_stats["high"]) if log_stats["high"] is not None else None))
    return dict(methods=methods, paired=pairs, selections=selections)


def _case_name(case):
    return {"reference": "Reference", "containment": "30° leakage", "fitted_source": "Fitted source (n₀=300)"}.get(case.get("family"), case["case_id"])


def _panels(plan, title):
    cases = sorted(plan["cases"], key=lambda case: (case["n_train"], {"reference": 0, "containment": 1, "fitted_source": 2}.get(case.get("family"), 3), case["case_id"]))
    columns = min(3, len(cases))
    nrows = math.ceil(len(cases)/columns)
    fig, axes = plt.subplots(nrows, columns, figsize=(4.0*columns, 3.0*nrows+.9), squeeze=False)
    fig.suptitle(title, x=.04, ha="left", fontsize=12)
    for ax, case in zip(axes.flat, cases):
        ax.set_title(f"{_case_name(case)}  |  n={case['n_train']}", fontsize=10)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in list(axes.flat)[len(cases):]:
        ax.set_visible(False)
    fig.subplots_adjust(left=.07, right=.98, top=.89, bottom=.19, hspace=.55, wspace=.34)
    return fig, list(zip(axes.flat, cases))


def _error(ax, x, value, low, high, color):
    if value is None:
        return
    error = None if low is None else [[max(0., value-low)], [max(0., high-value)]]
    ax.errorbar(x, value, yerr=error, fmt="o", color=color, capsize=2, ms=4, lw=1)


def _footer(fig, text):
    import textwrap
    fig.text(.04, .025, textwrap.fill(text, width=180), fontsize=7, va="bottom", color="#444444")


def figures(plan, summaries):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "xtick.labelsize": 8,
        "ytick.labelsize": 8, "pdf.fonttype": 42, "savefig.dpi": 200, "grid.alpha": .25})
    methods = {(row["case_id"], row["method"]): row for row in summaries["methods"]}
    result = []
    for metric, name, title in (("risk", "risk", "Operational rank selection: prediction risk"),
            ("tuning_seconds", "timing", "Joint tuning across ranks and source dimensions")):
        fig, panels = _panels(plan, title)
        for ax, case in panels:
            labels = []
            for x, (method, color) in enumerate(zip(PRIMARY, COLORS)):
                row = methods.get((case["case_id"], method))
                label = {"v2": "v2", "initializer_only": "Init.*" if metric == "tuning_seconds" else "Init.",
                         "source_subspace_ridge_rrr": "Source ridge", "target_ridge_rrr": "Target ridge"}[method]
                if row:
                    stats = row[metric]
                    _error(ax, x, stats["mean"], stats["low"], stats["high"], color)
                    label += f"\n{stats['n']}/{row['n_planned']}"
                labels.append(label)
            ax.set_xticks(range(len(PRIMARY)), labels, rotation=20)
            ax.set_yscale("log"); ax.grid(axis="y")
            ax.set_ylabel("Population excess prediction risk" if metric == "risk" else "Joint tuning seconds")
        _footer(fig, "Means and pointwise normal 95% Monte Carlo intervals across independent seeds. Counts show available/planned observations; failures remain explicit in tables." if metric == "risk" else
            "Means and pointwise 95% Monte Carlo intervals over all valid attempted clocks, including failed fits. Fitting-call totals include required shared SVD for source-subspace methods. *Initializer uses its separately recorded pass wall time. No missing fitting clock is replaced by CSV wall time.")
        result.append((name, fig))
    fig, panels = _panels(plan, "Paired risk of v2 after operational rank selection")
    for ax, case in panels:
        for x, reference in enumerate(REFERENCES):
            row = next((item for item in summaries["paired"] if item["case_id"] == case["case_id"] and item["method"] == "v2" and item["reference"] == reference), None)
            if row:
                _error(ax, x, row["geometric_ratio"], row["ratio_low"], row["ratio_high"], COLORS[x])
        ax.axhline(1., color="#777777", ls="--", lw=.8); ax.set_yscale("log"); ax.grid(axis="y")
        ax.set_xticks(range(3), ["Target ridge", "Source ridge", "Initializer"], rotation=20)
        ax.set_ylabel("Paired geometric risk ratio")
    _footer(fig, "v2 risk divided by each jointly tuned comparator; below one favors v2. Intervals use paired log ratios across seeds. Zero risks enter arithmetic differences but cannot enter log ratios. These are pointwise, not simultaneous, comparisons.")
    result.append(("paired", fig))
    for dimension, title, selected_methods in (("fitted_rank", "Selected fitted-rank distributions", PRIMARY),
            ("source_dimension", "Operational source dimensions among source-using selections", PRIMARY[:3])):
        fig, panels = _panels(plan, title)
        grid_name = "rank_grid" if dimension == "fitted_rank" else "source_truncation_grid"
        values = sorted(set(plan.get("configuration", {}).get(grid_name, [])) |
            {row["value"] for row in summaries["selections"] if row["dimension"] == dimension and row["method"] in selected_methods}, key=str)
        palette = ["#56B4E9", "#009E73", "#CC79A7", "#E69F00", "#0072B2"]
        for ax, case in panels:
            labels = []
            for x, method in enumerate(selected_methods):
                items = [row for row in summaries["selections"] if row["case_id"] == case["case_id"] and row["method"] == method and row["dimension"] == dimension]
                applicable = items[0]["n_applicable"] if items else 0
                record = methods.get((case["case_id"], method), {})
                labels.append(LABELS[method].replace(" RRR", "")+f"\n{applicable}/{record.get('n_planned', 0)}")
                bottom = 0.
                for value, color in zip(values, palette):
                    count = sum(row["count"] for row in items if row["value"] == value)
                    height = count/applicable if applicable else 0.
                    ax.bar(x, height, bottom=bottom, color=color, width=.65)
                    if height > .12:
                        ax.text(x, bottom+height/2, str(count), ha="center", va="center", fontsize=8)
                    bottom += height
            ax.set_xticks(range(len(selected_methods)), labels, rotation=20); ax.set_ylim(0, 1.05)
            ax.set_ylabel("Fraction of applicable selections"); ax.grid(axis="y")
        fig.legend(handles=[Patch(color=color, label=str(value)) for value, color in zip(values, palette)],
                   loc="lower center", bbox_to_anchor=(.5, .075), ncol=max(1, len(values)), frameon=False,
                   title="Fitted rank" if dimension == "fitted_rank" else "Selected source dimension")
        fig.subplots_adjust(bottom=.28, hspace=.6)
        _footer(fig, "Counts within bars; applicable/planned counts below labels. Rank-free estimators are excluded from the fitted-rank distribution; actual numerical coefficient ranks remain in the CSV appendix." if dimension == "fitted_rank" else
            "Counts within bars; applicable/planned counts below labels. Source-independent RRR endpoints and target/oracle methods have no selected operational source dimension and are excluded, not counted as zero. Failures remain in planned denominators.")
        result.append((dimension.replace("_", "-"), fig))
    return result


def _tex(value):
    return str(value).replace("_", "\\_").replace("%", "\\%").replace("&", "\\&").replace("°", "$^\\circ$").replace("₀", "$_0$").replace("×", "$\\times$").replace("–", "--")


def _format(stats):
    return "--" if stats["mean"] is None else f"{stats['mean']:.3g}"+(f" ({stats['mcse']:.2g})" if stats["mcse"] is not None else "")


def _csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        writer.writerows({key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value for key, value in row.items()} for row in rows)


def tables(prefix, plan, summaries):
    end = " " + chr(92)*2
    lookup = {(row["case_id"], row["method"]): row for row in summaries["methods"]}
    for metric, suffix in (("risk", "risk"), ("tuning_seconds", "timing"), (None, "counts")):
        headings = "Case & $n$ & v2 (s) & Init. wall (s) & Source ridge (s) & Target ridge (s)" if metric == "tuning_seconds" else "Case & $n$ & v2 & Initializer & Source ridge & Target ridge"
        lines = ["% Means (MCSE), or valid/planned counts. All cases use operational rank selection.",
            "% Timing: fitting calls plus required observed SVD; initializer column uses pass wall time. Missing clocks are --.",
            "\\begin{tabular}{llrrrr}", "\\toprule", headings+end, "\\midrule"]
        for case in plan["cases"]:
            values = []
            for method in PRIMARY:
                row = lookup.get((case["case_id"], method))
                values.append("--" if row is None else _format(row[metric]) if metric else f"{row['n_success']}/{row['n_planned']}")
            lines.append(" & ".join([_tex(_case_name(case)), str(case["n_train"]), *values])+end)
        lines.extend(["\\bottomrule", "\\end{tabular}"])
        Path(str(prefix)+f"-{suffix}.tex").write_text("\n".join(lines)+"\n")
    lines = ["% Paired geometric risk ratios and arithmetic v2-minus-initializer differences.",
        "\\begin{tabular}{llrrrr}", "\\toprule", "Case & $n$ & v2/target & v2/source & v2/init. & Difference (MCSE)"+end, "\\midrule"]
    for case in plan["cases"]:
        values, last = [], None
        for reference in REFERENCES:
            row = next((row for row in summaries["paired"] if row["case_id"] == case["case_id"] and row["method"] == "v2" and row["reference"] == reference), None)
            values.append("--" if not row or row["geometric_ratio"] is None else f"{row['geometric_ratio']:.3f}")
            last = row
        difference = "--" if not last else _format(dict(mean=last["mean_difference"], mcse=last["difference_mcse"]))
        lines.append(" & ".join([_tex(_case_name(case)), str(case["n_train"]), *values, difference])+end)
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    Path(str(prefix)+"-paired.tex").write_text("\n".join(lines)+"\n")


def build_report(root, output_prefix):
    root, prefix = Path(root).resolve(), Path(output_prefix).resolve()
    plan, rows, provenance = load_study(root)
    summary = summarize_rows(plan, rows)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    outputs = []
    with PdfPages(str(prefix)+"-figures.pdf") as book:
        for name, figure in figures(plan, summary):
            for extension in ("pdf", "png"):
                path = str(prefix)+f"-{name}.{extension}"
                figure.savefig(path, bbox_inches="tight", pad_inches=.12); outputs.append(path)
            book.savefig(figure, bbox_inches="tight", pad_inches=.12); plt.close(figure)
    tables(prefix, plan, summary)
    for key in ("methods", "paired", "selections"):
        _csv(str(prefix)+f"-{key}.csv", summary[key])
    _csv(str(prefix)+"-replications.csv", rows)
    summary.update(schema=1, study="audited_operational_rank_selection", root=str(root), provenance=provenance,
        n_cases=len(plan["cases"]), n_tasks=len(plan["tasks"]), cases=plan["cases"],
        configuration=plan.get("configuration", {}), figures=outputs,
        conventions=dict(risk="Exact population prediction excess risk; responses' observation noise excluded",
            paired="Per-case paired geometric risk ratios and arithmetic method-minus-reference differences across independent seeds; pointwise normal 95% intervals",
            unavailable="All planned outcome counts remain explicit; failed fits are excluded from risk means but included in available attempted-runtime summaries",
            selection="Selected fitted rank is a tuning parameter and is distinct from numerical coefficient rank, notably for source-target mixtures. Rank-free methods have no fitted-rank choice; source-independent endpoints and target/oracle methods have no operational source-dimension choice",
            timing="Recorded complete fitting-call totals, plus shared observed-source decomposition once for two source-subspace methods. Initializer uses labelled pass wall time. CSV total_seconds remains wall time. Missing JSON gives unavailable tuning clocks, never substituted wall time",
            shared_work="v2 and v2_transfer_only share their complete grid; their clocks must not be summed. The winning candidate is not a cold standalone fit",
            source_fit="Source-data generation and source estimation are outside these target-stage tuning clocks"))
    Path(str(prefix)+"-summary.json").write_text(json.dumps(summary, sort_keys=True, indent=2, allow_nan=False)+"\n")
    return dict(prefix=str(prefix), n_cases=len(plan["cases"]), n_tasks=len(plan["tasks"]), n_figures=len(outputs)//2,
                n_timing_unavailable=sum(row["tuning_seconds"] is None for row in rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root"); parser.add_argument("--output-prefix", required=True)
    args = parser.parse_args()
    print(json.dumps(build_report(args.root, args.output_prefix), sort_keys=True))


if __name__ == "__main__":
    main()
