"""Report realized generator metadata from a passing main-study audit.

Reads JSON receipts only: no data arrays, generation, fitting, or optimization.
Every planned task must have a complete matching receipt. Scalar diagnostics
and summaries of recorded numerical vectors are exported without pooling
different conditions into Monte Carlo replicates. Full original metadata is
retained in the JSON output. Selected penalty-boundary counts exclude the
explicit target-RRR endpoint, where those penalties were not used.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import tempfile
import os


STRUCTURAL_FAMILIES = frozenset(("reference", "source_noise", "containment", "containment_side",
    "target_specific", "diffuse_alignment", "source_internal_gap", "target_internal_gap",
    "source_boundary_gap", "target_boundary_gap", "source_truncation", "sparse_stress"))
REQUIRED_DIAGNOSTICS = ("latent_source_rank", "operational_source_rank", "source_numerical_rank",
    "source_error_frobenius", "source_error_operator", "target_population_signal",
    "target_coefficient_energy", "source_coefficient_energy", "containment_left", "containment_right", "containment_joint")


def _json(path):
    return json.loads(Path(path).read_text())


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _stats(values, planned):
    values = list(values)
    mean = statistics.mean(values) if values else None
    mcse = statistics.stdev(values) / math.sqrt(len(values)) if len(values) > 1 else None
    return dict(n=len(values), n_planned=planned, n_unavailable=planned-len(values), mean=mean, mcse=mcse,
                minimum=min(values) if values else None, maximum=max(values) if values else None)


def flatten_numeric(value, prefix=""):
    """Flatten scalars; retain vector min/max/mean/length rather than index means."""
    output = {}
    if isinstance(value, dict):
        for key, child in value.items():
            if key not in ("case", "candidates", "seed"):
                output.update(flatten_numeric(child, f"{prefix}.{key}" if prefix else key))
    elif _number(value):
        output[prefix] = value
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"Nonfinite metadata: {prefix}")
    elif isinstance(value, list) and value and all(_number(child) for child in value):
        output.update({prefix+".minimum": min(value), prefix+".maximum": max(value),
                       prefix+".mean": statistics.mean(value), prefix+".length": len(value)})
    elif isinstance(value, list) and any(isinstance(child, float) and not math.isfinite(child) for child in value):
        raise ValueError(f"Nonfinite metadata vector: {prefix}")
    return output


def _realized(metadata):
    result = flatten_numeric(metadata)
    target_norm = math.sqrt(metadata["target_coefficient_energy"])
    source_norm = math.sqrt(metadata["source_coefficient_energy"])
    if target_norm > 0:
        for side in ("left", "right", "joint"):
            result["containment_"+side+"_relative_frobenius"] = metadata["containment_"+side] / target_norm
    if source_norm > 0:
        result["source_error_relative_frobenius"] = metadata["source_error_frobenius"] / source_norm
    values = metadata.get("observed_source_singular_values", [])
    dimension = metadata["operational_source_rank"]
    if type(dimension) is not int or dimension < 1:
        raise ValueError("Invalid operational source dimension")
    if len(values) >= dimension and all(_number(value) for value in values):
        last = values[dimension-1]
        following = values[dimension] if len(values) > dimension else 0.
        result["observed_source_retained_smallest_singular_value"] = last
        result["observed_source_retained_boundary_gap"] = last-following
        if dimension > 1:
            result["observed_source_retained_minimum_internal_gap"] = min(a-b for a,b in zip(values[:dimension-1], values[1:dimension]))
        if last > 0:
            result["observed_source_retained_condition_number"] = values[0]/last
    return result


def _penalty_grids(config, variant):
    if variant in ("active_caps", "cap_control"):
        factors = [0., .0025, .01, .04]
        return dict(init_penalty=config["init_penalties"], penalty_u=factors, penalty_v=factors)
    return dict(init_penalty=config["init_penalties"], penalty_u=config["penalties_u"], penalty_v=config["penalties_v"])


def _selected_penalties(result, task, config):
    output = []
    for variant in task.get("variants", ["full_caps"]):
        suffix = "" if variant == "full_caps" else "_"+variant
        for label in ("v2", "v2_transfer_only"):
            method = label+suffix
            fit = result.get("methods", {}).get(method)
            if fit is None:
                raise ValueError(f"Missing selected-method receipt: {method}")
            success = fit.get("success") is True
            endpoint = success and fit.get("selected_method") == "target_rrr"
            item = dict(method=method, variant=variant, success=success,
                        target_rrr_endpoint=endpoint, penalties_applicable=success and not endpoint,
                        selected_iteration=fit.get("selected_iteration"), boundaries={})
            if item["penalties_applicable"]:
                parameters = fit.get("parameters", {})
                pair = parameters.get("penalty")
                if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                    raise ValueError(f"Missing factor penalties for {method}")
                values = dict(init_penalty=parameters.get("init_penalty"), penalty_u=pair[0], penalty_v=pair[1])
                for key, grid in _penalty_grids(config, variant).items():
                    if not grid or not all(_number(value) for value in grid) or values[key] not in grid:
                        raise ValueError(f"Selected {key} does not belong to frozen {method} grid")
                    item["boundaries"][key] = dict(value=values[key], minimum=min(grid), maximum=max(grid),
                        at_lower=values[key] == min(grid), at_upper=values[key] == max(grid),
                        singleton_grid=len(set(grid)) == 1)
            output.append(item)
    return output


def load_study(root):
    root = Path(root).resolve()
    plan_path = root / "plan.json"
    plan, audited = _json(plan_path), _json(root / "audit.json")
    digest = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    if audited.get("audit_passed") is not True or audited.get("all_planned_tasks_complete") is not True:
        raise ValueError("Metadata reporting requires a passing complete main-study audit")
    if audited.get("plan_sha256") != digest or audited.get("n_planned_tasks") != len(plan["tasks"]):
        raise ValueError("Audit plan identity/count mismatch")
    reports = audited.get("tasks", [])
    if [row.get("task_index") for row in reports] != list(range(len(plan["tasks"]))) or any(row.get("state") != "complete" for row in reports):
        raise ValueError("Audit does not cover every original task exactly once")
    if len({(task["case_index"], task["seed"], tuple(task.get("variants", ["full_caps"]))) for task in plan["tasks"]}) != len(plan["tasks"]):
        raise ValueError("Repeated task identity in plan")
    rows, raw, selections = [], [], []
    for index, task in enumerate(plan["tasks"]):
        case = plan["cases"][task["case_index"]]
        path = root / "tasks" / f"task-{index:06d}" / "result.json"
        result = _json(path)
        if result.get("complete") is not True or result.get("plan_sha256") != digest or result.get("task") != task or result.get("case") != case:
            raise ValueError(f"Result identity mismatch at task {index}")
        metadata = result.get("metadata", {})
        if metadata.get("case_id") != case["case_id"] or metadata.get("seed") != task["seed"] or metadata.get("case") != case:
            raise ValueError(f"Generator metadata identity mismatch at task {index}")
        if any(not _number(metadata.get(key)) for key in REQUIRED_DIAGNOSTICS):
            raise ValueError(f"Missing/nonfinite required generator diagnostics at task {index}")
        base = dict(task_index=index, case_id=case["case_id"], family=case.get("family"), level=case.get("level"),
                    seed=task["seed"], n_train=case["n_train"], p=case["p"], q=case["q"],
                    source_mode=case.get("source_mode"),
                    n_source=case.get("n_source") if case.get("source_mode") == "fitted" else 0)
        diagnostics = _realized(metadata)
        rows.append(dict(base, **diagnostics))
        raw.append(dict(base, metadata=metadata, result_json_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        selections.extend(dict(base, **row) for row in _selected_penalties(result, task, plan["configuration"]))
    provenance = dict(plan_sha256=digest, audit_json_sha256=hashlib.sha256((root / "audit.json").read_bytes()).hexdigest(),
                      n_planned_tasks=len(plan["tasks"]), n_receipts=len(rows), audit_passed=True,
                      source="Passing audited result.json generator metadata; no arrays loaded or fits repeated")
    return plan, rows, raw, selections, provenance


def summarize_rows(plan, rows, selections):
    groups, selected_groups = defaultdict(list), defaultdict(list)
    for row in rows:
        groups[row["case_id"]].append(row)
    for row in selections:
        selected_groups[row["case_id"], row["method"]].append(row)
    condition_rows = []
    descriptive_keys = {"task_index", "case_id", "family", "level", "seed", "n_train", "p", "q", "source_mode", "n_source"}
    for case in plan["cases"]:
        members = groups[case["case_id"]]
        planned = sum(plan["cases"][task["case_index"]]["case_id"] == case["case_id"] for task in plan["tasks"])
        keys = sorted({key for row in members for key in row if key not in descriptive_keys})
        summary = dict(case_id=case["case_id"], family=case.get("family"), level=case.get("level"),
            n_train=case["n_train"], n_source=case.get("n_source") if case.get("source_mode") == "fitted" else 0,
            source_mode=case.get("source_mode"),
            n_planned=planned, n_available=len(members),
            metrics={key: _stats([row[key] for row in members if _number(row.get(key))], planned) for key in keys})
        summary["source_fit_rank_counts"] = dict(Counter(str(row["source_fit.selected_rank"]) for row in members if "source_fit.selected_rank" in row))
        summary["source_fit_ridge_counts"] = dict(Counter(str(row["source_fit.selected_alpha"]) for row in members if "source_fit.selected_alpha" in row))
        condition_rows.append(summary)
    boundary_rows = []
    for (case_id, method), members in sorted(selected_groups.items()):
        applicable = [row for row in members if row["penalties_applicable"]]
        for parameter in ("init_penalty", "penalty_u", "penalty_v"):
            edges = [row["boundaries"][parameter] for row in applicable]
            boundary_rows.append(dict(case_id=case_id, method=method, parameter=parameter,
                n_planned=len(members), n_success=sum(row["success"] for row in members),
                n_endpoint=sum(row["target_rrr_endpoint"] for row in members), n_applicable=len(edges),
                n_failed=sum(not row["success"] for row in members),
                lower_count=sum(row["at_lower"] for row in edges), upper_count=sum(row["at_upper"] for row in edges),
                singleton_count=sum(row["singleton_grid"] for row in edges),
                selected_value_counts=dict(Counter(str(row["value"]) for row in edges))))
    return dict(conditions=condition_rows, penalty_boundaries=boundary_rows,
        interpretation="Condition-wise mean and Monte Carlo SE across seeds; no pooling of different severity levels. Scalar generator diagnostics are recorded receipt values and are not independently recomputed from arrays by this reporter. Angles, alignment and containment describe clean generator frames; observed-source approximation is separately measured by source errors and spectral diagnostics. Grid-edge selection motivates disclosure, not automatic post-confirmation grid expansion.")


def _atomic_text(path, content):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".metadata-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            stream.write(content)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _write_csv(path, rows):
    import io
    output = io.StringIO()
    fields = list(dict.fromkeys(key for row in rows for key in row))
    writer = csv.DictWriter(output, fieldnames=fields)
    writer.writeheader()
    writer.writerows({key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
                     for key, value in row.items()} for row in rows)
    _atomic_text(path, output.getvalue())


def _tex(value):
    return str(value).replace("\\", r"\textbackslash{}").replace("_", r"\_").replace("%", r"\%").replace("&", r"\&").replace("#", r"\#")


def _cell(row, name):
    stats = row["metrics"].get(name, {})
    if stats.get("mean") is None:
        return "--"
    mean, se = stats["mean"], stats["mcse"]
    return f"{mean:.3g}" + (f" ({se:.2g})" if se is not None else "")


def _table(rows, headings, fields, caption):
    labels = dict(fitted_source="Fitted source", fitted_relationship="Fitted relationship",
        reference="Reference", source_noise="Source noise", containment="Both-side angle",
        containment_side="One-side angle", target_specific="New directions", diffuse_alignment="Diffusion",
        source_internal_gap="Source internal", target_internal_gap="Target internal",
        source_boundary_gap="Source boundary", target_boundary_gap="Target boundary",
        source_truncation="Source truncation", sparse_stress="Sparse stress")
    lines = [r"\begin{longtable}{@{}l"+"r"*len(fields)+r"@{}}", r"\caption{"+caption+r"}\\",
             r"\toprule", "Condition / target $n$ & "+" & ".join(headings)+r" \\", r"\midrule\endhead"]
    for row in rows:
        label = _tex(f"{labels.get(row['family'], row['family'])} {row['level']}; {row['n_train']}")
        lines.append(label+" & "+" & ".join(_cell(row, field) for field in fields)+r" \\")
    lines += [r"\bottomrule", r"\end{longtable}"]
    return "\n".join(lines)+"\n"


def build_report(root, output_prefix):
    plan, rows, raw, selections, provenance = load_study(root)
    summary = summarize_rows(plan, rows, selections)
    prefix = Path(output_prefix)
    payload = dict(provenance=provenance, summary=summary, per_task=rows,
                   selected_penalties=selections, original_generator_metadata=raw)
    _atomic_text(Path(str(prefix)+".json"), json.dumps(payload, indent=2, sort_keys=True, allow_nan=False)+"\n")
    _write_csv(Path(str(prefix)+"-per-task.csv"), rows)
    flattened = [dict(case_id=row["case_id"], family=row["family"], level=row["level"], n_train=row["n_train"], metric=metric, **stats)
                 for row in summary["conditions"] for metric, stats in row["metrics"].items()]
    _write_csv(Path(str(prefix)+"-condition-metrics.csv"), flattened)
    _write_csv(Path(str(prefix)+"-penalty-boundaries.csv"), summary["penalty_boundaries"])
    fitted = [row for row in summary["conditions"] if row["source_mode"] == "fitted"]
    _atomic_text(Path(str(prefix)+"-fitted-source.tex"), _table(fitted,
        [r"$\|S-S^*\|_F$", r"$\|S-S^*\|_{\rm op}$", "Fit rank", "Retained gap"],
        ["source_error_frobenius", "source_error_operator", "source_fit.selected_rank", "observed_source_retained_boundary_gap"],
        r"Realized fitted-source diagnostics; means (Monte Carlo SE). Fit rank is distinct from the operational retained dimension. Full rank/ridge selection counts and all sample sizes are in the companion JSON."))
    structural = [row for row in summary["conditions"] if row["family"] in STRUCTURAL_FAMILIES]
    _atomic_text(Path(str(prefix)+"-structure.tex"), _table(structural,
        ["Relative leakage", r"Left tail$_2$", r"Right tail$_2$", "Source gap", "Target gap"],
        ["containment_joint_relative_frobenius", "alignment_left.tail_after_top2", "alignment_right.tail_after_top2", "source_internal_gaps.minimum", "target_internal_gaps.minimum"],
        r"Realized structural diagnostics; means (Monte Carlo SE). Leakage is the clean-source two-sided residual divided by target Frobenius norm. Tails are source-coordinate factor Frobenius mass after the two largest entries per column. Gaps are minimum clean internal gaps, not observed retained boundary gaps."))
    return dict(**provenance, n_conditions=len(summary["conditions"]), n_penalty_rows=len(summary["penalty_boundaries"]), output_prefix=str(prefix))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root")
    parser.add_argument("output_prefix")
    args = parser.parse_args()
    print(json.dumps(build_report(args.root, args.output_prefix), sort_keys=True))


if __name__ == "__main__":
    main()
