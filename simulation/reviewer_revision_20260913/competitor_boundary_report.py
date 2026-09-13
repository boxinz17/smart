"""Read audited JSON receipts to summarize raw-source comparator grid limits.

No arrays, fitting code, or numerical libraries are read. Generic comparator
summaries cover the same raw-source tasks as Park, not every main-study task.
"""
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics

PARK = "park_two_stage_nr_external_validation"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def numeric_summary(values):
    values = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return dict(n=len(values), minimum=min(values), median=statistics.median(values),
                mean=statistics.mean(values), maximum=max(values)) if values else dict(n=0)


def summarize(rows):
    out = dict(n_tasks=len(rows), park_successes=sum(r["park_success"] for r in rows))
    out["park_candidate_statuses"] = dict(sum((Counter(r["candidate_statuses"]) for r in rows), Counter()))
    out["park_eligible_counts"] = dict(Counter(str(r["n_eligible"]) for r in rows))
    out["park_actual_admm_fits"] = sum(r["n_admm_fits"] for r in rows)
    out["park_actual_admm_stops_met"] = sum(r["n_admm_stops_met"] for r in rows)
    out["park_selected_stops_met"] = sum(r.get("selected_stops_met") is True for r in rows)
    out["boundaries"] = {}
    for key in ("lambda_w", "lambda_delta", "ridge_to_source", "nuclear_contrast"):
        selected = [r["boundaries"][key] for r in rows if key in r["boundaries"]]
        out["boundaries"][key] = dict(n_selected=len(selected),
            lower_count=sum(v["lower"] for v in selected), upper_count=sum(v["upper"] for v in selected),
            selected_value_counts=dict(Counter(str(v["value"]) for v in selected)))
    out["selected_numerical_diagnostics"] = {
        key: numeric_summary([r.get(key) for r in rows]) for key in (
            "pooled_iterations", "correction_iterations", "pooled_squared_update", "correction_squared_update",
            "pooled_duality_gap", "correction_duality_gap", "pooled_gap_relative_to_primal", "correction_gap_relative_to_primal")}
    out["generic_candidate_statuses"] = {
        method: dict(sum((Counter(r["generic_statuses"].get(method, {})) for r in rows), Counter()))
        for method in ("ridge_to_source", "nuclear_contrast")}
    return out


def inspect(root):
    root = Path(root)
    hashes, issues, rows = {}, [], []

    def read(path):
        hashes[str(path)] = sha(path)
        return json.loads(path.read_text())

    def require(condition, message):
        if not condition:
            issues.append(message)

    plan = read(root / "plan.json")
    audit = read(root / "audit.json")
    digest = hashes[str(root / "plan.json")]
    require(audit.get("audit_passed") is True and audit.get("all_planned_tasks_complete") is True
            and audit.get("issues") == [], "Complete passing zero-issue numerical audit required")
    require(audit.get("plan_sha256") == digest, "Audit-plan hash mismatch")
    planned = [(i, t) for i, t in enumerate(plan["tasks"]) if PARK in t["expected_methods"]]
    park_grid = plan["configuration"]["park_lambdas"]
    expected_pairs = [(float(w), float(d)) for w in park_grid for d in park_grid]

    def boundary(value, grid):
        value = float(value)
        require(value in grid, "Selected value outside frozen grid")
        return dict(value=value, lower=value == min(grid), upper=value == max(grid))

    for index, task in planned:
        case = plan["cases"][task["case_index"]]
        path = root / "tasks" / f"task-{index:06d}" / "result.json"
        result = read(path)
        require(result.get("complete") is True and result.get("plan_sha256") == digest,
                f"Unbound/incomplete result {index}")
        require(result.get("case") == case and result.get("task") == task, f"Identity mismatch {index}")
        fit = result["methods"][PARK]
        diag, candidates = fit.get("diagnostics", {}), fit.get("candidates", [])
        require([(float(c["lambda_w"]), float(c["lambda_delta"])) for c in candidates] == expected_pairs,
                f"Park candidate grid/order mismatch {index}")
        actual = diag.get("admm_fits", [])
        row = dict(task_index=index, case_id=case["case_id"], seed=task["seed"],
                   park_success=fit.get("success") is True,
                   candidate_statuses=dict(Counter(c["status"] for c in candidates)),
                   n_eligible=sum(c["status"] == "ok" for c in candidates),
                   n_admm_fits=len(actual), n_admm_stops_met=sum(a["stopping_criterion_met"] is True for a in actual),
                   boundaries={}, generic_statuses={})
        if row["park_success"]:
            chosen = candidates[fit["selected_index"]]
            eligible = [(float(c["validation_loss"]), j) for j, c in enumerate(candidates) if c["status"] == "ok"]
            require(bool(eligible) and min(eligible)[1] == fit["selected_index"], f"Park validation selection mismatch {index}")
            row["selected_stops_met"] = diag.get("selected_admm_stopping_criteria_met")
            require(row["selected_stops_met"] is True, f"Selected Park stopping threshold failed {index}")
            for key in ("lambda_w", "lambda_delta"):
                require(float(chosen[key]) == float(diag["selected_" + key]), f"Selected Park penalty mismatch {index}")
                row["boundaries"][key] = boundary(chosen[key], park_grid)
            for stage in ("pooled", "correction"):
                # R records ADMM invocation indices starting at one.
                item = actual[chosen[stage + "_admm_index"] - 1]
                row[stage + "_iterations"] = item["iterations"]
                row[stage + "_squared_update"] = item["last_max_squared_update"]
                row[stage + "_duality_gap"] = item.get("duality_gap")
                primal = item.get("standardized_primal_objective")
                row[stage + "_gap_relative_to_primal"] = (
                    item["duality_gap"] / primal if primal and item.get("duality_gap") is not None else None)
        for method, parameter in (("ridge_to_source", "ridge"), ("nuclear_contrast", "nuclear_penalty")):
            generic = result["methods"][method]
            records = generic.get("audit", [])
            row["generic_statuses"][method] = dict(Counter(c["status"] for c in records))
            grid = sorted({float(c["candidate"][parameter]) for c in records})
            if generic.get("success"):
                row["boundaries"][method] = boundary(generic["parameters"][parameter], grid)
        rows.append(row)
    by_case = defaultdict(list)
    for row in rows:
        by_case[row["case_id"]].append(row)
    return dict(schema=1, metadata_checks_passed=not issues, issues=issues,
                n_planned_raw_source_tasks=len(planned), n_inspected_tasks=len(rows),
                summary=summarize(rows), by_case={k: summarize(v) for k, v in by_case.items()},
                replications=rows, input_sha256=hashes,
                scope="Metadata only, conditional on complete numerical audit; raw-source tasks only. "
                      "ADMM invocation counts account for pooled-fit reuse. No arrays, fits, or response losses recomputed.",
                limitations=["Finite penalty grids and authors' squared-update eligibility; no global optimality assertion.",
                             "Duality-gap summaries describe the separately standardized stage objectives, not target prediction risk."])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = inspect(args.root)
    report["auditor_sha256"] = sha(Path(__file__))
    report["created_utc"] = datetime.now(timezone.utc).isoformat()
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: report[k] for k in ("metadata_checks_passed", "issues", "n_inspected_tasks", "summary")}))
    raise SystemExit(0 if report["metadata_checks_passed"] else 2)
